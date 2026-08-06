#!/usr/bin/env python3
"""Build paired S2 brightness counterfactuals for the legacy-360 classifier.

The probe deliberately transforms already-materialized 224 x 224 model inputs.
This keeps the crop, texture, temporal acquisition, and label fixed while changing
only radiometric magnitude.  These counterfactuals are sensitivity tests, not new
physical satellite observations and must not be reported as ordinary test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile


FRAME_COLUMNS = (
    ("t0", "s2_0_path", (8, 9)),
    ("t90", "s2_90_path", ()),
    ("t360", "s2_360_path", ()),
)
SCALE_FACTORS = (0.70, 0.85, 1.00, 1.15, 1.30)
EXPECTED_SHAPE = (12, 224, 224)
VALUE_MIN = 0.0
VALUE_MAX = 65535.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-reference-per-label", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_chw(path: str | Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    if array.shape != EXPECTED_SHAPE:
        raise ValueError(f"{path}: expected {EXPECTED_SHAPE}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: contains non-finite values")
    return array.astype(np.float32, copy=False)


def channel_mean(path: str | Path) -> np.ndarray:
    return read_chw(path).mean(axis=(1, 2), dtype=np.float64)


def parallel_channel_means(paths: Iterable[str], workers: int) -> np.ndarray:
    path_list = list(paths)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(executor.map(channel_mean, path_list))
    return np.stack(values)


def training_centroids(
    manifest: pd.DataFrame,
    per_label: int,
    seed: int,
    workers: int,
) -> tuple[dict[str, dict[str, list[float]]], dict[int, pd.DataFrame]]:
    required = [column for _, column, _ in FRAME_COLUMNS]
    candidates = manifest.dropna(subset=required).copy()
    sampled: dict[int, pd.DataFrame] = {}
    centroids: dict[str, dict[str, list[float]]] = {"0": {}, "1": {}}
    for label in (0, 1):
        group = candidates[candidates["label"].astype(int) == label]
        if len(group) < per_label:
            raise ValueError(
                f"label {label}: requested {per_label} references, only {len(group)} available"
            )
        selected = group.sample(n=per_label, random_state=seed + label).copy()
        sampled[label] = selected
        for frame_name, column, _ in FRAME_COLUMNS:
            means = parallel_channel_means(selected[column].astype(str), workers)
            centroids[str(label)][frame_name] = means.mean(axis=0).tolist()
    return centroids, sampled


def mode_name_for_factor(factor: float) -> str:
    return f"scale_{factor:.2f}".replace(".", "p")


def transform_scale(
    source: np.ndarray,
    factor: float,
    forced_zero_channels: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, float]]:
    output = np.clip(source * np.float32(factor), VALUE_MIN, VALUE_MAX)
    if forced_zero_channels:
        output[list(forced_zero_channels)] = 0.0
    clipped = np.count_nonzero(source * np.float32(factor) > VALUE_MAX)
    return output.astype(np.float32, copy=False), {
        "clipped_high_fraction": float(clipped / source.size),
        "clipped_low_fraction": 0.0,
    }


def transform_centroid_match(
    source: np.ndarray,
    target: np.ndarray,
    forced_zero_channels: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, float]]:
    output = source.copy()
    source_means = source.mean(axis=(1, 2), dtype=np.float64)
    deltas = target.astype(np.float64) - source_means
    clipped_low = 0
    clipped_high = 0
    forced_zero = set(forced_zero_channels)
    for channel in range(source.shape[0]):
        if channel in forced_zero:
            output[channel] = 0.0
            continue
        valid = source[channel] != 0.0
        shifted = source[channel].astype(np.float64) + deltas[channel]
        clipped_low += int(np.count_nonzero(valid & (shifted < VALUE_MIN)))
        clipped_high += int(np.count_nonzero(valid & (shifted > VALUE_MAX)))
        output[channel, valid] = np.clip(
            shifted[valid], VALUE_MIN, VALUE_MAX
        ).astype(np.float32)
    return output, {
        "clipped_high_fraction": float(clipped_high / source.size),
        "clipped_low_fraction": float(clipped_low / source.size),
    }


def atomic_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part.tif")
    tifffile.imwrite(temporary, array.astype(np.float32, copy=False))
    temporary.replace(path)


def json_dump(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    input_manifest = args.input_manifest.resolve()
    training_manifest = args.training_manifest.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)

    source = pd.read_csv(input_manifest, low_memory=False)
    train = pd.read_csv(
        training_manifest,
        usecols=["id", "plume_id", "label", *[c for _, c, _ in FRAME_COLUMNS]],
        low_memory=False,
    )
    required = {"id", "label", *[column for _, column, _ in FRAME_COLUMNS]}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"input manifest missing columns: {missing}")

    centroids, sampled = training_centroids(
        train,
        per_label=args.training_reference_per_label,
        seed=args.seed,
        workers=args.workers,
    )
    json_dump(
        output_root / "training_reference_centroids.json",
        {
            "training_manifest": str(training_manifest),
            "training_manifest_sha256": sha256(training_manifest),
            "seed": args.seed,
            "rows_per_label": args.training_reference_per_label,
            "sampled_ids": {
                str(label): rows["id"].astype(str).tolist()
                for label, rows in sampled.items()
            },
            "centroids": centroids,
        },
    )

    modes = [
        {
            "name": mode_name_for_factor(factor),
            "kind": "identity" if factor == 1.0 else "global_scale",
            "factor": factor,
        }
        for factor in SCALE_FACTORS
    ]
    modes.extend(
        [
            {"name": "match_train_negative_centroid", "kind": "centroid", "label": 0},
            {"name": "match_train_positive_centroid", "kind": "centroid", "label": 1},
        ]
    )

    output_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for row in source.to_dict(orient="records"):
        parent_id = str(row["id"])
        arrays = {
            frame_name: read_chw(str(row[column]))
            for frame_name, column, _ in FRAME_COLUMNS
        }
        for mode in modes:
            mode_name = str(mode["name"])
            transformed_paths: dict[str, str] = {}
            frame_diagnostics: dict[str, object] = {}
            for frame_name, column, zero_channels in FRAME_COLUMNS:
                original_path = Path(str(row[column])).resolve()
                original = arrays[frame_name]
                if mode["kind"] == "identity":
                    transformed_paths[column] = str(original_path)
                    transformed = original
                    clip_info = {
                        "clipped_high_fraction": 0.0,
                        "clipped_low_fraction": 0.0,
                    }
                elif mode["kind"] == "global_scale":
                    transformed, clip_info = transform_scale(
                        original, float(mode["factor"]), zero_channels
                    )
                    output_path = (
                        output_root
                        / "crops"
                        / mode_name
                        / parent_id
                        / f"{frame_name}.tif"
                    )
                    atomic_tiff(output_path, transformed)
                    transformed_paths[column] = str(output_path)
                else:
                    target_label = str(int(mode["label"]))
                    target = np.asarray(
                        centroids[target_label][frame_name], dtype=np.float64
                    )
                    transformed, clip_info = transform_centroid_match(
                        original, target, zero_channels
                    )
                    output_path = (
                        output_root
                        / "crops"
                        / mode_name
                        / parent_id
                        / f"{frame_name}.tif"
                    )
                    atomic_tiff(output_path, transformed)
                    transformed_paths[column] = str(output_path)

                if zero_channels and any(np.any(transformed[index]) for index in zero_channels):
                    raise ValueError(
                        f"{parent_id}/{mode_name}/{frame_name}: forced-zero channel changed"
                    )
                valid_channels = [
                    index for index in range(12) if index not in set(zero_channels)
                ]
                original_flat = original[valid_channels].reshape(-1).astype(np.float64)
                transformed_flat = transformed[valid_channels].reshape(-1).astype(np.float64)
                correlation = float(np.corrcoef(original_flat, transformed_flat)[0, 1])
                frame_diagnostics[frame_name] = {
                    **clip_info,
                    "original_mean": float(original_flat.mean()),
                    "transformed_mean": float(transformed_flat.mean()),
                    "spatial_spectral_correlation": correlation,
                }

            output_row = dict(row)
            output_row.update(transformed_paths)
            output_row["id"] = f"{parent_id}__{mode_name}"
            output_row["parent_id"] = parent_id
            output_row["brightness_mode"] = mode_name
            output_row["brightness_transform_kind"] = str(mode["kind"])
            output_row["brightness_factor"] = mode.get("factor", np.nan)
            output_row["brightness_target_label"] = mode.get("label", np.nan)
            output_row["counterfactual_probe_only"] = mode["kind"] != "identity"
            output_rows.append(output_row)
            diagnostics.append(
                {
                    "parent_id": parent_id,
                    "label": int(row["label"]),
                    "mode": mode_name,
                    "frames": frame_diagnostics,
                }
            )

    output = pd.DataFrame.from_records(output_rows)
    manifest_path = output_root / "manifest_all_modes.csv"
    output.to_csv(manifest_path, index=False)
    mode_manifests: dict[str, str] = {}
    for mode_name, group in output.groupby("brightness_mode", sort=False):
        path = output_root / f"manifest_{mode_name}.csv"
        group.to_csv(path, index=False)
        mode_manifests[str(mode_name)] = str(path)

    audit = {
        "pass": True,
        "purpose": "paired counterfactual sensitivity test for radiometric brightness",
        "warning": (
            "Counterfactual rows are model probes, not physical satellite observations; "
            "do not mix their metrics with ordinary test accuracy."
        ),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": sha256(input_manifest),
        "training_manifest": str(training_manifest),
        "rows_source": int(len(source)),
        "source_label_counts": {
            str(key): int(value)
            for key, value in source["label"].astype(int).value_counts().sort_index().items()
        },
        "modes": modes,
        "rows_output": int(len(output)),
        "mode_manifests": mode_manifests,
        "expected_shape": list(EXPECTED_SHAPE),
        "stored_dtype": "float32",
        "t0_forced_zero_channels": [8, 9],
        "diagnostics": diagnostics,
    }
    json_dump(output_root / "dataset_audit.json", audit)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "manifest": str(manifest_path),
                "rows": len(output),
                "modes": len(modes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
