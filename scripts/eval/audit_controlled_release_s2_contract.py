#!/usr/bin/env python3
"""Fail-closed audit for controlled-release Sentinel-2 model inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.multisensor import TriSensorTemporalCsvDataset  # noqa: E402
from src.data.sensor_transforms import S2_PRECOMPUTED_STATS  # noqa: E402


SLOTS = {
    "t0": ("t0_raw_path", "s2_0_path"),
    "seasonal": ("seasonal_raw_path", "s2_90_path"),
    "year": ("year_raw_path", "s2_360_path"),
}
LEGACY_ZERO_CHANNELS = (8, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-table", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--scale-m", type=int, required=True)
    parser.add_argument("--crop-pixels", type=int, required=True)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument(
        "--resize-contract",
        choices=("torch_align_corners_false", "legacy_endpoint"),
        default="torch_align_corners_false",
    )
    parser.add_argument("--expected-label", type=int, choices=(0, 1), required=True)
    parser.add_argument("--training-manifest", default="")
    parser.add_argument("--training-reference-limit", type=int, default=64)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def read_chw(path: Path) -> np.ndarray:
    array = tifffile.imread(path)
    if array.ndim != 3:
        raise ValueError(f"{path}: expected 3-D TIFF, got {array.shape}")
    if array.shape[0] == 12:
        return array
    if array.shape[-1] == 12:
        return array.transpose(2, 0, 1)
    raise ValueError(f"{path}: expected 12 channels, got {array.shape}")


def recompute(
    raw: np.ndarray,
    crop_pixels: int,
    target_size: int,
    resize_contract: str,
) -> np.ndarray:
    _, height, width = raw.shape
    y0 = height // 2 - crop_pixels // 2
    x0 = width // 2 - crop_pixels // 2
    crop = raw[:, y0 : y0 + crop_pixels, x0 : x0 + crop_pixels]
    if crop.shape != (12, crop_pixels, crop_pixels):
        raise ValueError(f"unexpected centered crop shape: {crop.shape}")
    crop = crop.astype(np.float32, copy=False)
    if resize_contract == "torch_align_corners_false":
        tensor = torch.from_numpy(crop)[None]
        return (
            F.interpolate(
                tensor,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )[0]
            .numpy()
            .astype(np.float32)
        )
    if resize_contract == "legacy_endpoint":
        _, height, width = crop.shape
        ys = np.linspace(0, height - 1, target_size)
        xs = np.linspace(0, width - 1, target_size)
        y0 = np.floor(ys).astype(np.int64)
        x0 = np.floor(xs).astype(np.int64)
        y1 = np.clip(y0 + 1, 0, height - 1)
        x1 = np.clip(x0 + 1, 0, width - 1)
        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]
        output = np.empty(
            (crop.shape[0], target_size, target_size), dtype=np.float32
        )
        for band_index in range(crop.shape[0]):
            band = crop[band_index]
            a = band[y0[:, None], x0[None, :]]
            b = band[y0[:, None], x1[None, :]]
            c = band[y1[:, None], x0[None, :]]
            d = band[y1[:, None], x1[None, :]]
            output[band_index] = (
                a * (1 - wx) * (1 - wy)
                + b * wx * (1 - wy)
                + c * (1 - wx) * wy
                + d * wx * wy
            )
        return output
    raise ValueError(f"unsupported resize contract: {resize_contract}")


def assert_legacy_zero_channels(array: np.ndarray, path: Path) -> None:
    for channel in LEGACY_ZERO_CHANNELS:
        if np.any(array[channel] != 0):
            raise ValueError(
                f"{path}: legacy compatibility channel {channel} is not identically zero"
            )


def audit_training_reference(path: Path, limit: int, target_size: int) -> dict[str, Any]:
    columns = [output_column for _, output_column in SLOTS.values()]
    frame = pd.read_csv(path, usecols=columns, nrows=max(1000, limit * 8), low_memory=False)
    paths: list[Path] = []
    for column in columns:
        for value in frame[column].dropna().astype(str):
            candidate = Path(value)
            if candidate.is_file():
                paths.append(candidate)
                if len(paths) >= limit:
                    break
        if len(paths) >= limit:
            break
    if not paths:
        raise ValueError(f"No readable S2 training references in {path}")

    dtypes: set[str] = set()
    shapes: set[tuple[int, ...]] = set()
    for image_path in paths:
        array = read_chw(image_path)
        dtypes.add(str(array.dtype))
        shapes.add(tuple(int(v) for v in array.shape))
        if array.shape != (12, target_size, target_size):
            raise ValueError(f"{image_path}: historical shape mismatch {array.shape}")
        if array.dtype != np.float32:
            raise ValueError(f"{image_path}: historical dtype mismatch {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"{image_path}: historical input contains non-finite values")
        assert_legacy_zero_channels(array, image_path)
    return {
        "manifest": str(path),
        "checked_files": len(paths),
        "shapes": [list(v) for v in sorted(shapes)],
        "dtypes": sorted(dtypes),
        "legacy_zero_channels": list(LEGACY_ZERO_CHANNELS),
    }


def main() -> int:
    args = parse_args()
    selection_path = Path(args.selection_table)
    manifest_path = Path(args.candidate_manifest)
    selection = pd.read_csv(selection_path, low_memory=False)
    if "selection_status" in selection.columns:
        selection = selection[selection["selection_status"].astype(str).eq("complete")].copy()
    candidates = pd.read_csv(manifest_path, low_memory=False)

    if len(selection) != len(candidates):
        raise ValueError(
            f"row count mismatch: selection={len(selection)} candidate={len(candidates)}"
        )
    if selection["matched_ground_truth_sample_id"].astype(str).duplicated().any():
        raise ValueError("selection has duplicate matched_ground_truth_sample_id values")
    if candidates["plume_id"].astype(str).duplicated().any():
        raise ValueError("candidate manifest has duplicate plume_id values")
    labels = pd.to_numeric(candidates["label"], errors="raise").astype(int)
    if not labels.eq(int(args.expected_label)).all():
        raise ValueError(f"candidate manifest contains labels other than {args.expected_label}")

    by_truth = selection.set_index(
        selection["matched_ground_truth_sample_id"].astype(str), drop=False
    )
    candidate_ids = set(candidates["plume_id"].astype(str))
    selection_ids = set(by_truth.index.astype(str))
    if candidate_ids != selection_ids:
        raise ValueError(
            f"candidate/selection identities differ: "
            f"candidate_only={sorted(candidate_ids - selection_ids)[:5]}, "
            f"selection_only={sorted(selection_ids - candidate_ids)[:5]}"
        )

    checked_files = 0
    max_recompute_abs_diff = 0.0
    raw_dtypes: set[str] = set()
    output_dtypes: set[str] = set()
    raw_shapes: set[tuple[int, ...]] = set()
    output_shapes: set[tuple[int, ...]] = set()
    b12_zero_ratio_max = 0.0

    for row in candidates.to_dict("records"):
        truth_id = str(row["plume_id"])
        source = by_truth.loc[truth_id]
        for _, (raw_column, output_column) in SLOTS.items():
            raw_path = Path(str(source[raw_column]))
            output_path = Path(str(row[output_column]))
            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)
            if not output_path.is_file():
                raise FileNotFoundError(output_path)

            raw = read_chw(raw_path)
            output = read_chw(output_path)
            raw_shapes.add(tuple(int(v) for v in raw.shape))
            output_shapes.add(tuple(int(v) for v in output.shape))
            raw_dtypes.add(str(raw.dtype))
            output_dtypes.add(str(output.dtype))
            if raw.shape != (12, 512, 512):
                raise ValueError(f"{raw_path}: raw shape mismatch {raw.shape}")
            if output.shape != (12, args.target_size, args.target_size):
                raise ValueError(f"{output_path}: output shape mismatch {output.shape}")
            if output.dtype != np.float32:
                raise ValueError(f"{output_path}: output dtype mismatch {output.dtype}")
            if not np.isfinite(output).all():
                raise ValueError(f"{output_path}: non-finite values")
            assert_legacy_zero_channels(raw, raw_path)
            assert_legacy_zero_channels(output, output_path)

            expected = recompute(
                raw,
                int(args.crop_pixels),
                int(args.target_size),
                args.resize_contract,
            )
            difference = float(np.max(np.abs(output - expected)))
            max_recompute_abs_diff = max(max_recompute_abs_diff, difference)
            if difference != 0.0:
                raise ValueError(
                    f"{output_path}: not an exact legacy center-crop reconstruction; "
                    f"max_abs_diff={difference}"
                )
            b12_zero_ratio = float((output[11] == 0).mean())
            b12_zero_ratio_max = max(b12_zero_ratio_max, b12_zero_ratio)
            if b12_zero_ratio >= 0.20:
                raise ValueError(
                    f"{output_path}: B12 zero ratio {b12_zero_ratio:.6f} >= 0.20"
                )
            checked_files += 1

    dataset = TriSensorTemporalCsvDataset(
        csv_path=str(manifest_path),
        pad_to_multiple=14,
    )
    sensor_samples, loaded_label = dataset[0]
    if int(loaded_label) != int(args.expected_label):
        raise ValueError("loader changed the label")
    if len(sensor_samples) != 1 or sensor_samples[0][0] != "s2":
        raise ValueError(f"unexpected loader sensors: {[name for name, _ in sensor_samples]}")
    frames = sensor_samples[0][1]
    if len(frames) != 3:
        raise ValueError(f"loader temporal frame count mismatch: {len(frames)}")
    loaded = frames[0]["imgs"]
    first_output = read_chw(Path(str(candidates.iloc[0]["s2_0_path"]))).astype(np.float32)
    means = np.asarray(S2_PRECOMPUTED_STATS[0], dtype=np.float32)[:, None, None]
    stds = np.asarray(S2_PRECOMPUTED_STATS[1], dtype=np.float32)[:, None, None]
    expected_loaded = torch.from_numpy((first_output - means) / stds)
    loader_norm_abs_diff = float(torch.max(torch.abs(loaded - expected_loaded)).item())
    if loader_norm_abs_diff > 2e-6:
        raise ValueError(
            f"loader normalization mismatch: max_abs_diff={loader_norm_abs_diff}"
        )

    training_reference = None
    if args.training_manifest:
        training_reference = audit_training_reference(
            Path(args.training_manifest),
            int(args.training_reference_limit),
            int(args.target_size),
        )

    contract = {
        "source_product_level": "Sentinel-2 L2A surface reflectance",
        "raw_grid": "R20m",
        "raw_shape": [12, 512, 512],
        "legacy_channel_layout": [
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
        ],
        "temporal_order": ["t0", "minus_90_day_selection", "minus_360_day_selection"],
        "crop": {
            "centered": True,
            "pixels": int(args.crop_pixels),
            "scale_name_m": int(args.scale_m),
        },
        "resize": {
            "target": [int(args.target_size), int(args.target_size)],
            "mode": "bilinear",
            "align_corners": args.resize_contract == "legacy_endpoint",
            "implementation": args.resize_contract,
        },
        "stored_dtype": "float32",
        "loader": {
            "scale_to_unit_divisor": 65535.0,
            "normalization": "S2_PRECOMPUTED_STATS",
            "pad_to_multiple": 14,
            "loaded_frame_shape": list(loaded.shape),
            "temporal_frames": 3,
        },
    }
    contract_sha256 = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "pass": True,
        "contract_sha256": contract_sha256,
        "contract": contract,
        "selection_table": str(selection_path),
        "candidate_manifest": str(manifest_path),
        "rows": len(candidates),
        "checked_files": checked_files,
        "raw_shapes": [list(v) for v in sorted(raw_shapes)],
        "output_shapes": [list(v) for v in sorted(output_shapes)],
        "raw_dtypes": sorted(raw_dtypes),
        "output_dtypes": sorted(output_dtypes),
        "max_recompute_abs_diff": max_recompute_abs_diff,
        "loader_normalization_max_abs_diff": loader_norm_abs_diff,
        "max_b12_zero_ratio": b12_zero_ratio_max,
        "training_reference": training_reference,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
