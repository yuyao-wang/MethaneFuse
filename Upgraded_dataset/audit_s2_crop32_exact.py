#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

import s2_6time_cdse_legacy512_rebuild as pipeline


def clean(value: Any) -> str:
    return pipeline.clean(value)


def add_failure(failures: list[dict[str, Any]], **failure: Any) -> None:
    if len(failures) < 50:
        failures.append(failure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-csv", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sample-plumes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    patch_path = Path(args.patch_csv)
    source_path = Path(args.source_csv)
    patches = pd.read_csv(patch_path, low_memory=False)
    sources = pd.read_csv(source_path, low_memory=False)
    pipeline.required_columns(
        patches,
        [
            "sample_id",
            "plume_id",
            "label",
            "crop_x",
            "crop_y",
            "plume_mask_sum",
            "path_plume",
            *pipeline.PATH_COLUMNS.values(),
        ],
        patch_path,
    )
    pipeline.required_columns(
        sources,
        [
            "plume_id",
            "resized_512x512_path",
            *[timepoint.std_col for timepoint in pipeline.TIMEPOINTS],
        ],
        source_path,
    )
    if patches["sample_id"].astype(str).duplicated().any():
        raise ValueError("patch CSV contains duplicate sample_id values")
    if sources["plume_id"].astype(str).duplicated().any():
        raise ValueError("source CSV contains duplicate plume_id values")

    patch_ids = set(patches["plume_id"].astype(str))
    source_ids = set(sources["plume_id"].astype(str))
    missing_sources = sorted(patch_ids - source_ids)
    if missing_sources:
        raise ValueError(f"patch plumes missing from source CSV: {missing_sources[:20]}")

    invalid_labels = sorted(set(patches["label"].astype(int)) - {0, 1})
    invalid_coordinates = patches[
        (patches["crop_x"].astype(int) < 0)
        | (patches["crop_y"].astype(int) < 0)
        | (patches["crop_x"].astype(int) > pipeline.WINDOW_SIZE - pipeline.PATCH_SIZE)
        | (patches["crop_y"].astype(int) > pipeline.WINDOW_SIZE - pipeline.PATCH_SIZE)
    ]
    if invalid_labels:
        raise ValueError(f"invalid labels: {invalid_labels}")
    if not invalid_coordinates.empty:
        raise ValueError(f"invalid crop coordinates: {len(invalid_coordinates)}")

    plume_ids = pd.Series(sorted(patch_ids))
    if 0 < int(args.sample_plumes) < len(plume_ids):
        plume_ids = plume_ids.sample(
            n=int(args.sample_plumes), random_state=int(args.seed)
        ).sort_values()
    selected_ids = plume_ids.astype(str).tolist()
    source_by_id = sources.set_index(sources["plume_id"].astype(str), drop=False)
    grouped_patches = patches.groupby(patches["plume_id"].astype(str), sort=False)

    failures: list[dict[str, Any]] = []
    checked_patches = 0
    checked_images = 0
    checked_masks = 0
    label_counts = {0: 0, 1: 0}

    for plume_id in selected_ids:
        source_row = source_by_id.loc[plume_id]
        source_images = {
            timepoint.name: pipeline.read_chw_512(
                Path(clean(source_row[timepoint.std_col]))
            )
            for timepoint in pipeline.TIMEPOINTS
        }
        source_mask = pipeline.read_mask_512(
            Path(clean(source_row["resized_512x512_path"]))
        )
        for _, patch in grouped_patches.get_group(plume_id).iterrows():
            sample_id = clean(patch["sample_id"])
            x = int(patch["crop_x"])
            y = int(patch["crop_y"])
            label = int(patch["label"])
            expected_label = int(
                pipeline.legacy_center_contained(
                    x,
                    y,
                    patch_size=pipeline.PATCH_SIZE,
                    center_size=10,
                )
            )
            if label != expected_label:
                add_failure(
                    failures,
                    sample_id=sample_id,
                    kind="label_coordinate_mismatch",
                    expected=expected_label,
                    actual=label,
                )
            for timepoint in pipeline.TIMEPOINTS:
                expected_image = pipeline.crop_array(
                    source_images[timepoint.name], x, y, pipeline.PATCH_SIZE
                )
                actual_image = pipeline.to_bhw(
                    tifffile.imread(Path(clean(patch[pipeline.PATH_COLUMNS[timepoint.name]])))
                ).astype(np.float32, copy=False)
                if not np.array_equal(actual_image, expected_image):
                    add_failure(
                        failures,
                        sample_id=sample_id,
                        kind="image_slice_mismatch",
                        timepoint=timepoint.name,
                        max_abs_error=float(
                            np.max(np.abs(actual_image - expected_image))
                        ),
                    )
                checked_images += 1

            expected_mask = pipeline.crop_array(
                source_mask, x, y, pipeline.PATCH_SIZE
            )
            if label == 0:
                expected_mask = np.zeros_like(expected_mask)
            actual_mask = tifffile.imread(Path(clean(patch["path_plume"])))
            if actual_mask.shape != (pipeline.PATCH_SIZE, pipeline.PATCH_SIZE):
                add_failure(
                    failures,
                    sample_id=sample_id,
                    kind="mask_shape_mismatch",
                    shape=list(actual_mask.shape),
                )
            elif not np.array_equal(actual_mask, expected_mask):
                add_failure(
                    failures,
                    sample_id=sample_id,
                    kind="mask_slice_mismatch",
                    expected_sum=int(expected_mask.sum()),
                    actual_sum=int(actual_mask.sum()),
                )
            if int(actual_mask.sum()) != int(patch["plume_mask_sum"]):
                add_failure(
                    failures,
                    sample_id=sample_id,
                    kind="mask_sum_metadata_mismatch",
                    csv_sum=int(patch["plume_mask_sum"]),
                    actual_sum=int(actual_mask.sum()),
                )
            if label == 1 and int(actual_mask.sum()) <= 0:
                add_failure(
                    failures,
                    sample_id=sample_id,
                    kind="positive_mask_is_empty",
                )
            if label == 0 and int(actual_mask.sum()) != 0:
                add_failure(
                    failures,
                    sample_id=sample_id,
                    kind="negative_mask_is_nonzero",
                )
            checked_masks += 1
            checked_patches += 1
            label_counts[label] += 1

    report = {
        "patch_csv": str(patch_path),
        "source_csv": str(source_path),
        "total_patch_rows": int(len(patches)),
        "total_patch_plumes": int(len(patch_ids)),
        "sampled_plumes": int(len(selected_ids)),
        "checked_patches": int(checked_patches),
        "checked_images": int(checked_images),
        "checked_masks": int(checked_masks),
        "checked_label_counts": label_counts,
        "failure_count": int(len(failures)),
        "failure_examples": failures,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
