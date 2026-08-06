#!/usr/bin/env python3
"""Audit the timepoint-specific S2 channel contract used by legacy training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.multisensor import TriSensorTemporalCsvDataset  # noqa: E402
from src.data.sensor_transforms import S2_PRECOMPUTED_STATS  # noqa: E402


SLOTS = (
    ("t0", "s2_0_path"),
    ("seasonal", "s2_90_path"),
    ("year", "s2_360_path"),
)
T0_LAYOUT = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8A",
    "ZERO_COMPAT_1",
    "ZERO_COMPAT_2",
    "B11",
    "B12",
]
HISTORY_LAYOUT = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--materializer-audit", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-positives", type=int, required=True)
    parser.add_argument("--expected-negatives", type=int, required=True)
    parser.add_argument("--reference-samples-per-slot", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_chw(path: Path) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    if image.shape[-1:] == (12,) and image.shape[0] != 12:
        image = image.transpose(2, 0, 1)
    return image


def validate_slot(image: np.ndarray, path: Path, slot: str) -> None:
    if image.shape != (12, 224, 224):
        raise ValueError(f"{path}: unexpected shape {image.shape}")
    if image.dtype != np.float32:
        raise ValueError(f"{path}: unexpected dtype {image.dtype}")
    if not np.isfinite(image).all():
        raise ValueError(f"{path}: non-finite values")
    all_zero = [index for index in range(12) if not np.any(image[index])]
    if slot == "t0":
        if all_zero != [8, 9]:
            raise ValueError(
                f"{path}: t0 all-zero channels {all_zero}, expected [8, 9]"
            )
    elif all_zero:
        raise ValueError(f"{path}: history all-zero channels {all_zero}")
    if float((image[11] == 0).mean()) >= 0.20:
        raise ValueError(f"{path}: B12 zero ratio >= 0.20")


def uniform_indices(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, count).round().astype(int).tolist()))


def audit_frame(
    frame: pd.DataFrame,
    *,
    all_rows: bool,
    samples_per_slot: int,
) -> dict:
    report = {}
    for slot, column in SLOTS:
        if all_rows:
            indices = list(range(len(frame)))
        else:
            valid_indices = [
                index
                for index, value in frame[column].items()
                if pd.notna(value)
                and str(value).strip()
                and str(value).strip().lower() not in {"nan", "none", "null"}
            ]
            selected_positions = uniform_indices(
                len(valid_indices), samples_per_slot
            )
            indices = [valid_indices[position] for position in selected_positions]
        if not indices:
            raise ValueError(f"No readable files for {slot}/{column}")
        channel_sums = np.zeros(12, dtype=np.float64)
        channel_pixels = 0
        for index in indices:
            path = Path(str(frame.iloc[index][column]))
            if not path.is_file():
                raise FileNotFoundError(path)
            image = read_chw(path)
            validate_slot(image, path, slot)
            channel_sums += image.reshape(12, -1).sum(axis=1, dtype=np.float64)
            channel_pixels += image.shape[1] * image.shape[2]
        report[slot] = {
            "column": column,
            "files_checked": len(indices),
            "layout": T0_LAYOUT if slot == "t0" else HISTORY_LAYOUT,
            "channel_means": (channel_sums / channel_pixels).tolist(),
            "expected_all_zero_channels": [8, 9] if slot == "t0" else [],
        }
    return report


def main() -> None:
    args = parse_args()
    candidate_path = args.candidate_manifest.resolve()
    candidate = pd.read_csv(candidate_path, low_memory=False)
    labels = candidate["label"].astype(int)
    if len(candidate) != args.expected_rows:
        raise ValueError("Candidate row count mismatch")
    if int((labels == 1).sum()) != args.expected_positives:
        raise ValueError("Candidate positive count mismatch")
    if int((labels == 0).sum()) != args.expected_negatives:
        raise ValueError("Candidate negative count mismatch")

    materializer = json.loads(args.materializer_audit.read_text())
    if materializer.get("pass") is not True:
        raise ValueError("Materializer audit did not pass")
    if Path(materializer["output_manifest"]).resolve() != candidate_path:
        raise ValueError("Materializer audit points to a different manifest")
    if materializer["output_manifest_sha256"] != sha256(candidate_path):
        raise ValueError("Candidate manifest hash differs from materializer audit")
    materializer_contract = materializer.get("contract", {})
    materializer_t0_layout = materializer.get("t0_layout")
    if materializer_t0_layout is None and materializer_contract.get(
        "t0_expected_all_zero_channels"
    ) == [8, 9]:
        materializer_t0_layout = T0_LAYOUT
    materializer_history_layout = materializer.get("historical_layout")
    if materializer_history_layout is None and materializer_contract.get(
        "historical_expected_all_zero_channels"
    ) == []:
        materializer_history_layout = HISTORY_LAYOUT
    if materializer_t0_layout != T0_LAYOUT:
        raise ValueError("Materializer t0 layout mismatch")
    if materializer_history_layout != HISTORY_LAYOUT:
        raise ValueError("Materializer history layout mismatch")

    candidate_report = audit_frame(
        candidate, all_rows=True, samples_per_slot=len(candidate)
    )
    training = pd.read_csv(args.training_manifest, low_memory=False)
    training_report = audit_frame(
        training,
        all_rows=False,
        samples_per_slot=args.reference_samples_per_slot,
    )

    dataset = TriSensorTemporalCsvDataset(
        csv_path=str(candidate_path),
        pad_to_multiple=14,
    )
    sensor_samples, loaded_label = dataset[0]
    if int(loaded_label) != int(candidate.iloc[0]["label"]):
        raise ValueError("Loader changed label")
    if len(sensor_samples) != 1 or sensor_samples[0][0] != "s2":
        raise ValueError("Loader did not return exactly one S2 sensor")
    frames = sensor_samples[0][1]
    if len(frames) != 3:
        raise ValueError("Loader did not preserve three temporal frames")
    means = np.asarray(S2_PRECOMPUTED_STATS[0], dtype=np.float32)[:, None, None]
    stds = np.asarray(S2_PRECOMPUTED_STATS[1], dtype=np.float32)[:, None, None]
    loader_diffs = {}
    first_row = candidate.iloc[0]
    for (slot, column), loaded in zip(SLOTS, frames):
        stored = read_chw(Path(str(first_row[column]))).astype(np.float32)
        expected = torch.from_numpy((stored - means) / stds)
        actual = loaded["imgs"]
        difference = float(torch.max(torch.abs(actual - expected)).item())
        if difference > 2e-6:
            raise ValueError(f"Loader normalization mismatch at {slot}: {difference}")
        loader_diffs[slot] = difference

    contract = {
        "scale_m": 360,
        "crop_pixels": 36,
        "output_size": 224,
        "stored_dtype": "float32",
        "resize": "legacy endpoint-aligned bilinear",
        "temporal_order": ["t0", "seasonal/t-90", "year/t-360"],
        "t0_layout": T0_LAYOUT,
        "historical_layout": HISTORY_LAYOUT,
        "normalization": "S2_PRECOMPUTED_STATS",
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = {
        "pass": True,
        "contract": contract,
        "contract_sha256": contract_hash,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": sha256(candidate_path),
        "materializer_audit": str(args.materializer_audit.resolve()),
        "materializer_audit_sha256": sha256(args.materializer_audit),
        "candidate": candidate_report,
        "training_reference_manifest": str(args.training_manifest.resolve()),
        "training_reference": training_report,
        "loader_normalization_max_abs_diff": loader_diffs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pass": True,
                "contract_sha256": contract_hash,
                "candidate_files_checked": {
                    slot: data["files_checked"]
                    for slot, data in candidate_report.items()
                },
                "training_reference_files_checked": {
                    slot: data["files_checked"]
                    for slot, data in training_report.items()
                },
                "candidate_expected_zero_channels": {
                    slot: data["expected_all_zero_channels"]
                    for slot, data in candidate_report.items()
                },
                "training_expected_zero_channels": {
                    slot: data["expected_all_zero_channels"]
                    for slot, data in training_report.items()
                },
                "loader_normalization_max_abs_diff": loader_diffs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
