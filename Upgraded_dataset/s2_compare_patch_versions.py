#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


IMAGE_COLUMNS = (
    "path_t0",
    "path_prev1",
    "path_prev2",
    "path_prev3",
    "path_seasonal",
    "path_year",
)
ALL_BANDS = tuple(range(12))


def to_chw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"expected 3D image, got {array.shape}")
    if array.shape[0] == 12:
        return array
    if array.shape[-1] == 12:
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"cannot infer channel axis for {array.shape}")


def sample_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["split"].astype(str)
        + "/"
        + frame["plume_id"].astype(str)
        + "/"
        + frame["path"].map(lambda value: Path(str(value)).name)
    )


def normalized_manifest(
    frame: pd.DataFrame,
    patch_root: str,
) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["_sample_key"] = sample_key(normalized)
    for column in normalized.columns:
        if normalized[column].dtype == object:
            normalized[column] = normalized[column].map(
                lambda value: (
                    value.replace(patch_root, "<PATCH_ROOT>")
                    if isinstance(value, str)
                    else value
                )
            )
    return normalized.sort_values("_sample_key").reset_index(drop=True)


def compare_pair(
    row: dict[str, Any],
    *,
    changed_bands: tuple[int, ...],
) -> dict[str, Any]:
    unchanged_bands = tuple(
        band for band in ALL_BANDS if band not in changed_bands
    )
    result: dict[str, Any] = {
        "files": 0,
        "errors": [],
    }
    for band in changed_bands:
        result[f"old_band{band}_nonzero"] = 0
        result[f"new_band{band}_nonzero"] = 0
        result[f"band{band}_changed"] = 0
    try:
        old_mask = np.asarray(tifffile.imread(row["path_plume_old"]))
        new_mask = np.asarray(tifffile.imread(row["path_plume_new"]))
        if not np.array_equal(old_mask, new_mask):
            result["errors"].append("mask_changed")
        for column in IMAGE_COLUMNS:
            old_image = to_chw(tifffile.imread(row[f"{column}_old"]))
            new_image = to_chw(tifffile.imread(row[f"{column}_new"]))
            result["files"] += 1
            if not np.array_equal(
                old_image[list(unchanged_bands)],
                new_image[list(unchanged_bands)],
            ):
                result["errors"].append(f"{column}:non_target_bands_changed")
            for band in changed_bands:
                if np.any(old_image[band]):
                    result[f"old_band{band}_nonzero"] += 1
                if np.any(new_image[band]):
                    result[f"new_band{band}_nonzero"] += 1
                if not np.array_equal(old_image[band], new_image[band]):
                    result[f"band{band}_changed"] += 1
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}:{exc}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-train", required=True)
    parser.add_argument("--old-test", required=True)
    parser.add_argument("--new-train", required=True)
    parser.add_argument("--new-test", required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--sample-rows", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--changed-bands", default="8,9")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    changed_bands = tuple(
        int(value.strip())
        for value in args.changed_bands.split(",")
        if value.strip()
    )
    if not changed_bands or any(
        band not in ALL_BANDS for band in changed_bands
    ):
        raise ValueError(f"Invalid --changed-bands={args.changed_bands!r}")

    old = pd.concat(
        [
            pd.read_csv(args.old_train, low_memory=False),
            pd.read_csv(args.old_test, low_memory=False),
        ],
        ignore_index=True,
    )
    new = pd.concat(
        [
            pd.read_csv(args.new_train, low_memory=False),
            pd.read_csv(args.new_test, low_memory=False),
        ],
        ignore_index=True,
    )
    old_normalized = normalized_manifest(old, args.old_root)
    new_normalized = normalized_manifest(new, args.new_root)
    old_keys = set(old_normalized["_sample_key"])
    new_keys = set(new_normalized["_sample_key"])
    shared_columns = sorted(
        set(old_normalized.columns) & set(new_normalized.columns)
    )
    metadata_differences = 0
    if old_keys == new_keys and len(old_normalized) == len(new_normalized):
        old_values = old_normalized[shared_columns].fillna("<NA>").astype(str)
        new_values = new_normalized[shared_columns].fillna("<NA>").astype(str)
        metadata_differences = int((old_values != new_values).to_numpy().sum())

    old_paths = old.copy()
    new_paths = new.copy()
    old_paths["_sample_key"] = sample_key(old_paths)
    new_paths["_sample_key"] = sample_key(new_paths)
    selected_columns = ["_sample_key", "path_plume", *IMAGE_COLUMNS]
    merged = old_paths[selected_columns].merge(
        new_paths[selected_columns],
        on="_sample_key",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    if 0 < args.sample_rows < len(merged):
        merged = merged.sample(args.sample_rows, random_state=args.seed)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(
                partial(compare_pair, changed_bands=changed_bands),
                merged.to_dict("records"),
            )
        )

    totals: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    for row, result in zip(merged.to_dict("records"), results):
        totals.update(
            {
                key: int(value)
                for key, value in result.items()
                if key != "errors"
            }
        )
        if result["errors"]:
            errors.append(
                {
                    "sample_key": row["_sample_key"],
                    "errors": result["errors"],
                }
            )
    files = totals["files"]
    report = {
        "old_rows": len(old),
        "new_rows": len(new),
        "old_only_keys": len(old_keys - new_keys),
        "new_only_keys": len(new_keys - old_keys),
        "metadata_differences": metadata_differences,
        "sampled_rows": len(merged),
        "sampled_image_pairs": files,
        "changed_bands": changed_bands,
        "band_changes": {
            str(band): {
                "old_nonzero_files": totals[f"old_band{band}_nonzero"],
                "new_nonzero_files": totals[f"new_band{band}_nonzero"],
                "changed_files": totals[f"band{band}_changed"],
            }
            for band in changed_bands
        },
        "comparison_errors": len(errors),
        "comparison_error_examples": errors[:20],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    return int(
        bool(
            report["old_only_keys"]
            or report["new_only_keys"]
            or report["metadata_differences"]
            or report["comparison_errors"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
