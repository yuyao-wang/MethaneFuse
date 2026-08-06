#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from s2_point_center_recut_diagnostic import plume_shift_pixels, to_chw


TIMEPOINTS = {
    "seasonal": (
        "seasonal_product_id",
        ("seasonal_v8_direct_path", "seasonal_v9_target_path"),
        "seasonal_product_name",
        "s2_-90.tif",
    ),
    "year": (
        "year_product_id",
        ("year_v8_direct_path", "year_v9_target_path"),
        "year_product_name",
        "s2_-360.tif",
    ),
}


def recenter(array: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    output = np.zeros_like(array)
    source_x0 = max(0, shift_x)
    source_y0 = max(0, shift_y)
    source_x1 = min(512, 512 + shift_x)
    source_y1 = min(512, 512 + shift_y)
    target_x0 = max(0, -shift_x)
    target_y0 = max(0, -shift_y)
    width = source_x1 - source_x0
    height = source_y1 - source_y0
    if width <= 0 or height <= 0:
        raise ValueError(f"shift outside source: x={shift_x} y={shift_y}")
    output[
        :,
        target_y0 : target_y0 + height,
        target_x0 : target_x0 + width,
    ] = array[:, source_y0:source_y1, source_x0:source_x1]
    output[[8, 9]] = 0
    return output


def write_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.partial.tif"
    )
    tifffile.imwrite(temporary, array)
    os.replace(temporary, path)


def process(
    task: tuple[dict[str, Any], str],
    target_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    row, timepoint = task
    _, source_columns, product_column, filename = TIMEPOINTS[timepoint]
    plume_id = str(row["plume_id"])
    source = next(
        (
            Path(str(row[column]))
            for column in source_columns
            if isinstance(row.get(column), str)
            and row[column].strip()
            and Path(row[column]).is_file()
        ),
        None,
    )
    if source is None:
        raise FileNotFoundError(
            f"no trusted legacy source for {plume_id} {timepoint}"
        )
    cached = cache_root / plume_id / timepoint / source.name
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.is_file() or cached.stat().st_size == 0:
        temporary = cached.with_name(cached.name + f".tmp.{os.getpid()}")
        shutil.copy2(source, temporary)
        os.replace(temporary, cached)
    array = to_chw(tifffile.imread(cached))
    if array.shape != (12, 512, 512):
        raise ValueError(f"unexpected source shape {array.shape}: {source}")
    shift_x, shift_y, distance_km = plume_shift_pixels(row, product_column)
    output = recenter(array, shift_x, shift_y)
    target = target_root / timepoint / plume_id / filename
    write_atomic(target, output)
    shutil.rmtree(cache_root / plume_id / timepoint, ignore_errors=True)
    return {
        "plume_id": plume_id,
        "timepoint": timepoint,
        "source_path": str(source),
        "target_path": str(target),
        "shift_x": shift_x,
        "shift_y": shift_y,
        "offset_km": distance_km,
        "status": "legacy_512_recentered",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.table, low_memory=False)
    tasks = [
        (row, timepoint)
        for row in frame.to_dict("records")
        for timepoint, (product_column, source_columns, _, _) in TIMEPOINTS.items()
        if (
            pd.isna(row.get(product_column))
            or str(row.get(f"{timepoint}_recrop_status", "")).strip()
            == "failed"
        )
        and any(
            isinstance(row.get(column), str) and row[column].strip()
            for column in source_columns
        )
    ]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                process,
                task,
                Path(args.target_root),
                Path(args.cache_root),
            ): (str(task[0]["plume_id"]), task[1])
            for task in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            plume_id, timepoint = futures[future]
            try:
                records.append(future.result())
            except Exception as error:
                failures.append(
                    {
                        "plume_id": plume_id,
                        "timepoint": timepoint,
                        "error": f"{type(error).__name__}:{error}",
                    }
                )
            if index % 10 == 0 or index == len(futures):
                print(
                    f"{index}/{len(futures)} ok={len(records)} "
                    f"failed={len(failures)}",
                    flush=True,
                )

    report = {
        "tasks": len(tasks),
        "successful": len(records),
        "failed": len(failures),
        "records": sorted(
            records, key=lambda value: (value["plume_id"], value["timepoint"])
        ),
        "failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "successful": len(records),
                "failed": len(failures),
            },
            indent=2,
        ),
        flush=True,
    )
    if failures:
        raise RuntimeError(f"{len(failures)} legacy recenter failures")


if __name__ == "__main__":
    main()
