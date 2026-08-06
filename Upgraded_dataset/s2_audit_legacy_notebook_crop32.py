#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


IMAGE_COLUMNS = [
    "path_t0",
    "path_prev1",
    "path_prev2",
    "path_prev3",
    "path_seasonal",
    "path_year",
]
ALL_PATH_COLUMNS = [*IMAGE_COLUMNS, "path_plume"]
VALID_BANDS = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)


def center_contained(crop_x: int, crop_y: int) -> bool:
    return (
        crop_x <= 251
        and crop_y <= 251
        and crop_x + 32 >= 261
        and crop_y + 32 >= 261
    )


def file_ok(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def to_chw(path: str) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    if array.ndim == 3 and array.shape[0] == 12:
        return array
    if array.ndim == 3 and array.shape[-1] == 12:
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"expected 12-band TIFF, got {array.shape}: {path}")


def inspect_row(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_id": str(row["sample_id"]),
        "ok": True,
        "muted_nonzero": 0,
        "valid_nonzero": 0,
        "dtypes": set(),
    }
    try:
        for column in IMAGE_COLUMNS:
            array = to_chw(str(row[column]))
            if array.shape != (12, 32, 32):
                raise ValueError(f"{column} shape={array.shape}")
            result["muted_nonzero"] += int(
                np.count_nonzero(array[[8, 9]])
            )
            result["valid_nonzero"] += int(
                np.count_nonzero(array[list(VALID_BANDS)])
            )
            result["dtypes"].add(str(array.dtype))
        mask = np.squeeze(np.asarray(tifffile.imread(str(row["path_plume"]))))
        if mask.shape != (32, 32):
            raise ValueError(f"mask shape={mask.shape}")
        result["mask_sum"] = float(mask.sum())
        result["dtypes"] = sorted(result["dtypes"])
    except Exception as error:
        result["ok"] = False
        result["error"] = f"{type(error).__name__}:{error}"
        result["dtypes"] = sorted(result["dtypes"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False)
    train["split"] = "train"
    test["split"] = "test"
    combined = pd.concat([train, test], ignore_index=True)

    train_events = set(train["event_group_id"].astype(str))
    test_events = set(test["event_group_id"].astype(str))
    event_overlap = sorted(train_events & test_events)
    train_plumes = set(train["plume_id"].astype(str))
    test_plumes = set(test["plume_id"].astype(str))
    plume_overlap = sorted(train_plumes & test_plumes)

    positive_geometry_bad = combined[
        combined["crop_kind"].eq("positive")
        & (
            ~combined["crop_x"].between(234, 246)
            | ~combined["crop_y"].between(234, 246)
            | ~combined["label"].eq(1)
        )
    ]
    random_expected = combined.apply(
        lambda row: int(center_contained(int(row["crop_x"]), int(row["crop_y"]))),
        axis=1,
    )
    random_geometry_bad = combined[
        combined["crop_kind"].eq("random")
        & ~combined["label"].astype(int).eq(random_expected)
    ]

    paths = [
        str(value)
        for column in ALL_PATH_COLUMNS
        for value in combined[column]
    ]
    path_statuses: list[bool] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for start in range(0, len(paths), 20000):
            path_statuses.extend(
                executor.map(file_ok, paths[start : start + 20000])
            )
    missing_paths = [
        path for path, status in zip(paths, path_statuses) if not status
    ]

    sample = combined.sample(
        n=min(args.sample_rows, len(combined)),
        random_state=20260724,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        inspections = list(executor.map(inspect_row, sample.to_dict("records")))
    invalid_inspections = [
        inspection for inspection in inspections if not inspection["ok"]
    ]
    muted_nonzero = int(
        sum(inspection["muted_nonzero"] for inspection in inspections)
    )
    valid_empty = int(
        sum(inspection["valid_nonzero"] == 0 for inspection in inspections)
    )

    label_counts = combined["label"].value_counts().sort_index().to_dict()
    positive_fraction = float(combined["label"].mean())
    per_plume = combined.groupby(["split", "plume_id"]).size()
    failures: list[str] = []
    if event_overlap:
        failures.append("event-group leakage")
    if plume_overlap:
        failures.append("plume leakage")
    if missing_paths:
        failures.append(f"{len(missing_paths)} missing files")
    if len(positive_geometry_bad):
        failures.append("positive crop geometry mismatch")
    if len(random_geometry_bad):
        failures.append("random crop labels do not match old notebook")
    if invalid_inspections:
        failures.append("sampled TIFF read/shape failures")
    if muted_nonzero:
        failures.append("bands 8/9 are not muted")
    if valid_empty:
        failures.append("sampled rows contain empty valid bands")
    if not 0.40 <= positive_fraction <= 0.60:
        failures.append("class balance outside 40%-60%")

    report = {
        "passed": not failures,
        "failures": failures,
        "rows": {
            "all": len(combined),
            "train": len(train),
            "test": len(test),
        },
        "plumes": {
            "train": len(train_plumes),
            "test": len(test_plumes),
            "overlap": len(plume_overlap),
            "minimum_rows": int(per_plume.min()),
            "median_rows": float(per_plume.median()),
            "maximum_rows": int(per_plume.max()),
        },
        "events": {
            "train": len(train_events),
            "test": len(test_events),
            "overlap": len(event_overlap),
        },
        "labels": label_counts,
        "positive_fraction": positive_fraction,
        "path_checks": {
            "files": len(paths),
            "missing": len(missing_paths),
            "missing_examples": missing_paths[:20],
        },
        "geometry": {
            "positive_bad": len(positive_geometry_bad),
            "random_bad": len(random_geometry_bad),
        },
        "sample_inspection": {
            "rows": len(inspections),
            "invalid": len(invalid_inspections),
            "muted_nonzero": muted_nonzero,
            "valid_empty": valid_empty,
            "dtypes": sorted(
                {
                    dtype
                    for inspection in inspections
                    for dtype in inspection["dtypes"]
                }
            ),
            "invalid_examples": invalid_inspections[:20],
        },
        "args": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "failures": failures,
                "rows": report["rows"],
                "plumes": report["plumes"],
                "events": report["events"],
                "labels": label_counts,
                "path_checks": report["path_checks"],
                "sample_inspection": report["sample_inspection"],
            },
            indent=2,
        ),
        flush=True,
    )
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
