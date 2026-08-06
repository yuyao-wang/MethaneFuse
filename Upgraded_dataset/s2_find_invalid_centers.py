#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


TIMEPOINTS = {
    "t0": "t0_raw_path",
    "prev1": "prev1_raw_path",
    "prev2": "prev2_raw_path",
    "prev3": "prev3_raw_path",
    "seasonal": "seasonal_raw_path",
    "year": "year_raw_path",
}
VALID_BANDS = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)


def inspect(task: tuple[str, str, str]) -> dict[str, Any]:
    plume_id, timepoint, path = task
    try:
        array = np.asarray(tifffile.memmap(path))
        if array.shape == (512, 512, 12):
            array = np.moveaxis(array, -1, 0)
        if array.shape != (12, 512, 512):
            raise ValueError(f"unexpected shape {array.shape}")
        center = array[list(VALID_BANDS), 240:272, 240:272]
        zero_fractions = np.mean(center == 0, axis=(1, 2))
        return {
            "plume_id": plume_id,
            "timepoint": timepoint,
            "path": path,
            "ok": True,
            "max_center_zero_fraction": float(np.max(zero_fractions)),
            "center_zero_fractions": zero_fractions.tolist(),
        }
    except Exception as error:
        return {
            "plume_id": plume_id,
            "timepoint": timepoint,
            "path": path,
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--timepoints", default="t0,prev1,prev2,prev3,seasonal,year")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv, low_memory=False)
    timepoints = [
        value.strip() for value in args.timepoints.split(",") if value.strip()
    ]
    tasks = [
        (
            str(row["plume_id"]),
            timepoint,
            str(row[TIMEPOINTS[timepoint]]),
        )
        for row in frame.to_dict("records")
        for timepoint in timepoints
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        records = list(executor.map(inspect, tasks))
    unreadable = [record for record in records if not record["ok"]]
    invalid = [
        record
        for record in records
        if record["ok"]
        and record["max_center_zero_fraction"] >= args.threshold
    ]
    report = {
        "rows": len(frame),
        "tasks": len(tasks),
        "threshold": args.threshold,
        "unreadable": unreadable,
        "invalid_centers": invalid,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "unreadable": len(unreadable),
                "invalid_centers": len(invalid),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
