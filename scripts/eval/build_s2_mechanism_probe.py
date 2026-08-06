#!/usr/bin/env python3
"""Build paired counterfactuals that separate S2 radiometry, texture, and time.

The source TIFFs are already-materialized legacy-360 model inputs.  Every
counterfactual therefore keeps the exact crop/resize contract and changes one
mechanism at a time.  The generated rows are diagnostic interventions, not
physical satellite observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tifffile


FRAMES = (("t0", "s2_0_path"), ("t90", "s2_90_path"), ("t360", "s2_360_path"))
EXPECTED_SHAPE = (12, 224, 224)
T0_FORCED_ZERO = (8, 9)
SWIR = (10, 11)
VALUE_MIN = 0.0
VALUE_MAX = 65535.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--training-reference-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def read_chw(path: str | Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    if array.shape != EXPECTED_SHAPE:
        raise ValueError(f"{path}: expected {EXPECTED_SHAPE}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: contains non-finite values")
    return array.astype(np.float32, copy=False)


def atomic_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part.tif")
    tifffile.imwrite(temporary, array.astype(np.float32, copy=False))
    temporary.replace(path)


def json_dump(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def force_contract(array: np.ndarray, frame: str) -> np.ndarray:
    out = np.clip(array, VALUE_MIN, VALUE_MAX).astype(np.float32, copy=False)
    if frame == "t0":
        out[list(T0_FORCED_ZERO)] = 0.0
    return out


def spatial_constant(array: np.ndarray, channels: tuple[int, ...]) -> np.ndarray:
    out = array.copy()
    means = array.mean(axis=(1, 2), dtype=np.float64)
    for channel in channels:
        out[channel].fill(np.float32(means[channel]))
    return out


def patch_shuffle(array: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    channels, height, width = array.shape
    patch = 14
    if height % patch or width % patch:
        raise ValueError(f"patch shuffle needs multiples of {patch}, got {array.shape}")
    gh, gw = height // patch, width // patch
    patches = (
        array.reshape(channels, gh, patch, gw, patch)
        .transpose(0, 1, 3, 2, 4)
        .reshape(channels, gh * gw, patch, patch)
    )
    shuffled = patches[:, permutation]
    return (
        shuffled.reshape(channels, gh, gw, patch, patch)
        .transpose(0, 1, 3, 2, 4)
        .reshape(channels, height, width)
    )


def pixel_shuffle(array: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    channels, height, width = array.shape
    return array.reshape(channels, height * width)[:, permutation].reshape(array.shape)


def channel_stds(path: str) -> np.ndarray:
    return read_chw(path).std(axis=(1, 2), dtype=np.float64)


def training_negative_targets(
    reference_json: Path, workers: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    reference = json.loads(reference_json.read_text(encoding="utf-8"))
    training_manifest = Path(reference["training_manifest"])
    expected_hash = reference.get("training_manifest_sha256")
    actual_hash = sha256(training_manifest)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("training manifest hash no longer matches the brightness reference")

    sampled_ids = [str(value) for value in reference["sampled_ids"]["0"]]
    train = pd.read_csv(training_manifest, low_memory=False)
    train["id"] = train["id"].astype(str)
    by_id = train.drop_duplicates("id").set_index("id")
    missing = sorted(set(sampled_ids) - set(by_id.index))
    if missing:
        raise ValueError(f"{len(missing)} sampled negative IDs missing from training manifest")
    selected = by_id.loc[sampled_ids]

    target_means: dict[str, np.ndarray] = {}
    target_stds: dict[str, np.ndarray] = {}
    for frame, column in FRAMES:
        paths = selected[column].astype(str).tolist()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            per_image_stds = list(executor.map(channel_stds, paths))
        target_stds[frame] = np.stack(per_image_stds).mean(axis=0)
        target_means[frame] = np.asarray(reference["centroids"]["0"][frame], dtype=np.float64)
    metadata = {
        "reference_json": str(reference_json.resolve()),
        "training_manifest": str(training_manifest.resolve()),
        "training_manifest_sha256": actual_hash,
        "sample_count": len(sampled_ids),
        "sampled_ids_sha256": hashlib.sha256("\n".join(sampled_ids).encode()).hexdigest(),
    }
    return target_means, target_stds, metadata


def match_mean_std(array: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray) -> np.ndarray:
    out = np.empty_like(array)
    source_mean = array.mean(axis=(1, 2), dtype=np.float64)
    source_std = array.std(axis=(1, 2), dtype=np.float64)
    for channel in range(array.shape[0]):
        if source_std[channel] <= 1e-8:
            out[channel].fill(np.float32(target_mean[channel]))
        else:
            normalized = (array[channel].astype(np.float64) - source_mean[channel]) / source_std[channel]
            out[channel] = np.clip(
                normalized * target_std[channel] + target_mean[channel], VALUE_MIN, VALUE_MAX
            ).astype(np.float32)
    return out


def correlation(original: np.ndarray, transformed: np.ndarray) -> float | None:
    a = original.ravel().astype(np.float64)
    b = transformed.ravel().astype(np.float64)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    args = parse_args()
    input_manifest = args.input_manifest.resolve()
    output_root = args.output_root.resolve()
    reference_json = args.training_reference_json.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)

    source = pd.read_csv(input_manifest, low_memory=False)
    required = {"id", "label", *(column for _, column in FRAMES)}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"input manifest missing columns: {missing}")

    target_means, target_stds, reference_metadata = training_negative_targets(
        reference_json, args.workers
    )
    json_dump(
        output_root / "training_negative_moment_targets.json",
        {
            **reference_metadata,
            "means": {key: value.tolist() for key, value in target_means.items()},
            "stds": {key: value.tolist() for key, value in target_stds.items()},
        },
    )

    modes = (
        "original",
        "spatial_constant_all",
        "pixel_shuffle_joint",
        "patch_shuffle_14",
        "swir_spatial_constant",
        "nonswir_spatial_constant",
        "t0_equal_t90",
        "t0_equal_t360",
        "t0_swir_equal_t90",
        "t0_swir_equal_t360",
        "swap_t90_t360",
        "match_train_negative_mean_std",
    )

    output_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for row_index, row in enumerate(source.to_dict(orient="records")):
        parent_id = str(row.get("parent_id") or row["id"]).split("__scale_1p00")[0]
        arrays = {frame: read_chw(str(row[column])) for frame, column in FRAMES}
        rng = np.random.default_rng(args.seed + row_index)
        pixel_perm = rng.permutation(EXPECTED_SHAPE[1] * EXPECTED_SHAPE[2])
        patch_perm = rng.permutation((EXPECTED_SHAPE[1] // 14) * (EXPECTED_SHAPE[2] // 14))

        for mode in modes:
            transformed: dict[str, np.ndarray] = {}
            if mode == "original":
                transformed = {key: value for key, value in arrays.items()}
            elif mode == "spatial_constant_all":
                transformed = {key: spatial_constant(value, tuple(range(12))) for key, value in arrays.items()}
            elif mode == "pixel_shuffle_joint":
                transformed = {key: pixel_shuffle(value, pixel_perm) for key, value in arrays.items()}
            elif mode == "patch_shuffle_14":
                transformed = {key: patch_shuffle(value, patch_perm) for key, value in arrays.items()}
            elif mode == "swir_spatial_constant":
                transformed = {key: spatial_constant(value, SWIR) for key, value in arrays.items()}
            elif mode == "nonswir_spatial_constant":
                transformed = {key: spatial_constant(value, tuple(range(10))) for key, value in arrays.items()}
            elif mode == "t0_equal_t90":
                transformed = {**arrays, "t0": arrays["t90"].copy()}
            elif mode == "t0_equal_t360":
                transformed = {**arrays, "t0": arrays["t360"].copy()}
            elif mode == "t0_swir_equal_t90":
                t0 = arrays["t0"].copy()
                t0[list(SWIR)] = arrays["t90"][list(SWIR)]
                transformed = {**arrays, "t0": t0}
            elif mode == "t0_swir_equal_t360":
                t0 = arrays["t0"].copy()
                t0[list(SWIR)] = arrays["t360"][list(SWIR)]
                transformed = {**arrays, "t0": t0}
            elif mode == "swap_t90_t360":
                transformed = {"t0": arrays["t0"], "t90": arrays["t360"], "t360": arrays["t90"]}
            elif mode == "match_train_negative_mean_std":
                transformed = {
                    key: match_mean_std(value, target_means[key], target_stds[key])
                    for key, value in arrays.items()
                }
            else:
                raise AssertionError(mode)

            paths: dict[str, str] = {}
            frame_diag: dict[str, object] = {}
            for frame, column in FRAMES:
                result = force_contract(transformed[frame].copy(), frame)
                source_path = Path(str(row[column])).resolve()
                can_reuse = mode == "original" or (
                    mode == "swap_t90_t360" and frame != "t0"
                )
                if can_reuse:
                    if mode == "swap_t90_t360":
                        reuse_frame = "t360" if frame == "t90" else "t90"
                        reuse_column = dict(FRAMES)[reuse_frame]
                        paths[column] = str(Path(str(row[reuse_column])).resolve())
                    else:
                        paths[column] = str(source_path)
                elif np.array_equal(result, arrays[frame]):
                    paths[column] = str(source_path)
                else:
                    path = output_root / "crops" / mode / parent_id / f"{frame}.tif"
                    atomic_tiff(path, result)
                    paths[column] = str(path)
                frame_diag[frame] = {
                    "source_mean": float(arrays[frame].mean(dtype=np.float64)),
                    "result_mean": float(result.mean(dtype=np.float64)),
                    "source_std": float(arrays[frame].std(dtype=np.float64)),
                    "result_std": float(result.std(dtype=np.float64)),
                    "correlation": correlation(arrays[frame], result),
                    "t0_contract_ok": bool(frame != "t0" or np.all(result[list(T0_FORCED_ZERO)] == 0)),
                }

            out = dict(row)
            out.update(paths)
            out["id"] = f"{parent_id}__{mode}"
            out["parent_id"] = parent_id
            out["probe_mode"] = mode
            out["counterfactual_probe_only"] = mode != "original"
            output_rows.append(out)
            diagnostics.append(
                {"parent_id": parent_id, "label": int(row["label"]), "mode": mode, "frames": frame_diag}
            )

    combined = pd.DataFrame(output_rows)
    combined.to_csv(output_root / "manifest_all_modes.csv", index=False)
    for mode, group in combined.groupby("probe_mode", sort=False):
        group.to_csv(output_root / f"manifest_{mode}.csv", index=False)

    json_dump(
        output_root / "dataset_audit.json",
        {
            "input_manifest": str(input_manifest),
            "input_manifest_sha256": sha256(input_manifest),
            "rows": len(combined),
            "source_rows": len(source),
            "modes": list(modes),
            "seed": args.seed,
            "expected_shape": list(EXPECTED_SHAPE),
            "t0_forced_zero_channels": list(T0_FORCED_ZERO),
            "swir_channels": {"B11": 10, "B12": 11},
            "warning": "Counterfactual mechanism probe only; transformed rows are not physical observations.",
            "training_reference": reference_metadata,
            "diagnostics": diagnostics,
        },
    )
    (output_root / "README.md").write_text(
        "# S2 legacy-360 mechanism probe\n\n"
        "Paired diagnostic interventions over 23 already-materialized 224x224 inputs. "
        "The 12 modes separately test spatial texture, SWIR texture, temporal ordering, "
        "t0/history dependence, and training-negative radiometric moments. Transformed "
        "rows are counterfactual diagnostics, not satellite observations.\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(combined), "modes": len(modes), "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
