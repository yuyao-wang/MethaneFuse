#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from affine import Affine
from rasterio.warp import transform


PATH_MAPPING = {
    "s2_0_std_512": "t0_v8_source_path",
    "s2_-7_std_512": "prev1_v8_source_path",
    "s2_prev2_std_512": "prev2_v8_source_path",
    "s2_prev3_std_512": "prev3_v8_source_path",
    "s2_-90_std_512": "seasonal_v8_source_path",
    "s2_-360_std_512": "year_v8_source_path",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def source_class(path: str) -> str:
    markers = [
        "S2_point_center_exact_v3",
        "S2_canonical_6510_local_safe_recrop_v4",
        "plume_raw_s2_90360_fixed_512",
        "plume_s2_CDSE0_gee90360_original_512",
    ]
    for marker in markers:
        if marker in path:
            return marker
    return "other"


def file_status(path: str) -> tuple[str, bool, int]:
    candidate = Path(path)
    try:
        size = candidate.stat().st_size
    except OSError:
        return path, False, 0
    return path, candidate.is_file() and size > 0, int(size)


def plume_pixel(record: dict[str, Any]) -> tuple[str, float, float] | None:
    path = Path(clean(record["s2_0_std_512"]))
    if "S2_point_center_exact_v3" not in str(path):
        return None
    sidecar = path.with_name(path.name + ".georef.json")
    metadata = json.loads(sidecar.read_text())
    affine = Affine(*metadata["transform"][:6])
    x_values, y_values = transform(
        "EPSG:4326",
        metadata["crs_wkt"],
        [float(record["plume_longitude"])],
        [float(record["plume_latitude"])],
    )
    column, row = (~affine) * (float(x_values[0]), float(y_values[0]))
    return clean(record["plume_id"]), float(column), float(row)


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q10": float(np.quantile(array, 0.10)),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def prepare_split(path: str, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for destination, source in PATH_MAPPING.items():
        if source not in frame.columns:
            raise ValueError(f"missing source column {source}: {path}")
        frame[destination] = frame[source].map(clean)
    frame["split"] = split
    frame["center_mode"] = "plume_point"
    frame["image_geometry_version"] = "point_center_v11"
    for destination in PATH_MAPPING:
        frame[f"{destination}_source_class"] = frame[destination].map(source_class)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9-train-csv", required=True)
    parser.add_argument("--v9-test-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    train = prepare_split(args.v9_train_csv, "train")
    test = prepare_split(args.v9_test_csv, "test")
    combined = pd.concat([train, test], ignore_index=True)

    train_events = set(train["event_group_id"].astype(str))
    test_events = set(test["event_group_id"].astype(str))
    event_overlap = sorted(train_events & test_events)
    if event_overlap:
        raise ValueError(f"event leakage: {event_overlap[:10]}")

    paths = sorted(
        {
            clean(value)
            for column in [*PATH_MAPPING, "resized_512x512_path"]
            for value in combined[column]
            if clean(value)
        }
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        statuses = list(executor.map(file_status, paths))
    missing = [path for path, ok, _ in statuses if not ok]
    if missing:
        raise FileNotFoundError(f"{len(missing)} source files missing: {missing[:10]}")

    point_records = [
        record
        for record in combined.to_dict("records")
        if "S2_point_center_exact_v3" in clean(record["s2_0_std_512"])
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        point_pixels = [
            value
            for value in executor.map(plume_pixel, point_records)
            if value is not None
        ]
    x_values = [value[1] for value in point_pixels]
    y_values = [value[2] for value in point_pixels]
    off_center = [
        value
        for value in point_pixels
        if abs(value[1] - 256.0) > 1.01 or abs(value[2] - 256.0) > 1.01
    ]
    if off_center:
        raise ValueError(f"{len(off_center)} point-centered t0 files are misaligned")

    train_path = output_root / "s2_v11_train_512.csv"
    test_path = output_root / "s2_v11_test_512.csv"
    all_path = output_root / "s2_v11_all_512.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    combined.to_csv(all_path, index=False)

    source_counts: dict[str, dict[str, int]] = {}
    for destination in PATH_MAPPING:
        source_counts[destination] = dict(
            Counter(combined[f"{destination}_source_class"]).most_common()
        )
    report = {
        "status": "ok",
        "rows": {
            "all": len(combined),
            "train": len(train),
            "test": len(test),
        },
        "events": {
            "train": len(train_events),
            "test": len(test_events),
            "overlap": len(event_overlap),
        },
        "unique_source_files": len(paths),
        "missing_source_files": len(missing),
        "point_t0_georef_rows": len(point_pixels),
        "point_t0_x": quantiles(x_values),
        "point_t0_y": quantiles(y_values),
        "point_t0_off_center": len(off_center),
        "source_counts": source_counts,
        "outputs": {
            "all": str(all_path),
            "train": str(train_path),
            "test": str(test_path),
        },
    }
    report_path = output_root / "prepare_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
