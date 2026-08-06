#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


TIMEPOINTS = {
    "t0": ("path_t0", "s2_0_std_512"),
    "prev1": ("path_prev1", "s2_-7_std_512"),
    "prev2": ("path_prev2", "s2_prev2_std_512"),
    "prev3": ("path_prev3", "s2_prev3_std_512"),
    "seasonal": ("path_seasonal", "s2_-90_std_512"),
    "year": ("path_year", "s2_-360_std_512"),
}


def to_chw(path: str) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    if array.ndim != 3:
        raise ValueError(f"expected 3D TIFF, got {array.shape}: {path}")
    if array.shape[0] == 12:
        return array.astype(np.float32, copy=False)
    if array.shape[-1] == 12:
        return np.moveaxis(array, -1, 0).astype(np.float32, copy=False)
    raise ValueError(f"expected 12 bands, got {array.shape}: {path}")


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
    if int(valid.sum()) < 64:
        return float("-inf")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def compare(record: dict, source_record: dict, maximum_shift: int) -> dict:
    crop_x = int(record["crop_x"])
    crop_y = int(record["crop_y"])
    output = {
        "plume_id": str(record["plume_id"]),
        "label": int(record["label"]),
        "crop_x": crop_x,
        "crop_y": crop_y,
        "timepoints": {},
    }
    for timepoint, (patch_column, source_column) in TIMEPOINTS.items():
        patch = to_chw(str(record[patch_column]))
        source = to_chw(str(source_record[source_column]))
        best = (-2.0, 0, 0)
        for shift_y in range(-maximum_shift, maximum_shift + 1):
            for shift_x in range(-maximum_shift, maximum_shift + 1):
                source_crop = source[
                    11,
                    crop_y + shift_y : crop_y + shift_y + 32,
                    crop_x + shift_x : crop_x + shift_x + 32,
                ]
                if source_crop.shape != (32, 32):
                    continue
                value = correlation(patch[11], source_crop)
                if value > best[0]:
                    best = (value, shift_x, shift_y)
        zero_crop = source[:, crop_y : crop_y + 32, crop_x : crop_x + 32]
        output["timepoints"][timepoint] = {
            "same_coordinate_band11_correlation": correlation(
                patch[11],
                zero_crop[11],
            ),
            "best_band11_correlation": best[0],
            "best_shift_x": best[1],
            "best_shift_y": best[2],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-csv", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--maximum-plumes", type=int, default=50)
    parser.add_argument("--maximum-shift", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    patches = pd.read_csv(args.patch_csv, low_memory=False)
    patches = patches[pd.to_numeric(patches["label"]).eq(1)]
    patches = patches.sort_values(["plume_id", "crop_index"], kind="stable")
    patches = patches.groupby("plume_id", sort=False).head(1)
    patches = patches.sample(
        n=min(args.maximum_plumes, len(patches)),
        random_state=20260724,
    )
    source = pd.read_csv(args.source_csv, low_memory=False)
    source_lookup = {
        str(record["plume_id"]): record for record in source.to_dict("records")
    }
    tasks = [
        (record, source_lookup[str(record["plume_id"])])
        for record in patches.to_dict("records")
        if str(record["plume_id"]) in source_lookup
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        records = list(
            executor.map(
                lambda task: compare(task[0], task[1], args.maximum_shift),
                tasks,
            )
        )

    summary = {}
    for timepoint in TIMEPOINTS:
        values = [record["timepoints"][timepoint] for record in records]
        shifts = [(value["best_shift_x"], value["best_shift_y"]) for value in values]
        summary[timepoint] = {
            "same_coordinate_correlation_median": float(
                np.median(
                    [value["same_coordinate_band11_correlation"] for value in values]
                )
            ),
            "best_correlation_median": float(
                np.median([value["best_band11_correlation"] for value in values])
            ),
            "zero_shift_best": int(sum(shift == (0, 0) for shift in shifts)),
            "best_shift_x_median": float(np.median([shift[0] for shift in shifts])),
            "best_shift_y_median": float(np.median([shift[1] for shift in shifts])),
        }
    report = {
        "records": len(records),
        "maximum_shift": args.maximum_shift,
        "summary": summary,
        "details": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"records": len(records), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
