#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


TIMEPOINTS = {
    "seasonal": ("s2_90_std_512.tif", "s2_-90.tif"),
    "year": ("s2_360_std_512.tif", "s2_-360.tif"),
}


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def to_chw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.shape == (12, 512, 512):
        return array
    if array.shape == (512, 512, 12):
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"expected 12x512x512 image, got {array.shape}")


def copy_one(
    source: Path,
    target: Path,
    scratch: Path,
    overwrite: bool,
) -> None:
    if target.is_file() and target.stat().st_size > 0 and not overwrite:
        return
    image = to_chw(tifffile.imread(source))
    if any(not np.any(image[index]) for index in (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)):
        raise ValueError(f"valid spectral band is empty in {source}")
    scratch.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(scratch, image)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    shutil.copy2(scratch, temporary)
    os.replace(temporary, target)
    scratch.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    frame = pd.read_csv(csv_path, low_memory=False)
    jobs: list[tuple[int, str, Path, Path, Path]] = []
    mutable_columns = [
        f"{timepoint}_{suffix}"
        for timepoint in TIMEPOINTS
        for suffix in (
            "original_raw_path",
            "raw_path",
            "path_source",
            "local_status",
            "download_target_raw_path",
            "matched_old_timepoint",
            "recrop_status",
            "recrop_message",
        )
    ]
    for column in mutable_columns:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].astype(object)

    for index, row in frame.iterrows():
        plume_id = clean(row.get("plume_id"))
        for timepoint, (legacy_name, target_name) in TIMEPOINTS.items():
            product_id = clean(row.get(f"{timepoint}_product_id"))
            product_name = clean(row.get(f"{timepoint}_product_name"))
            if product_id and product_name:
                continue
            source = Path(args.legacy_root) / plume_id / legacy_name
            target = Path(args.target_root) / timepoint / plume_id / target_name
            scratch = Path(args.scratch_root) / plume_id / target_name
            if not source.is_file():
                raise FileNotFoundError(source)
            jobs.append((index, timepoint, source, target, scratch))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(
            executor.map(
                lambda job: copy_one(job[2], job[3], job[4], args.overwrite),
                jobs,
            )
        )

    for index, timepoint, source, target, _ in jobs:
        frame.at[index, f"{timepoint}_original_raw_path"] = str(source)
        frame.at[index, f"{timepoint}_raw_path"] = str(target)
        frame.at[index, f"{timepoint}_path_source"] = "legacy_fixed_512_exact"
        frame.at[index, f"{timepoint}_local_status"] = "available"
        frame.at[index, f"{timepoint}_download_target_raw_path"] = ""
        frame.at[index, f"{timepoint}_matched_old_timepoint"] = timepoint
        frame.at[index, f"{timepoint}_recrop_status"] = "downloaded"
        frame.at[index, f"{timepoint}_recrop_message"] = (
            f"legacy_fixed_512_exact; source={source}"
        )

    temporary = csv_path.with_name(csv_path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, csv_path)
    print(f"legacy_timepoints={len(jobs)} rows={len(frame)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
