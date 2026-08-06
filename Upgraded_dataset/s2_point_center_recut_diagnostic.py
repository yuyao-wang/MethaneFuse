#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from rasterio.warp import transform


TIMEPOINTS = {
    "t0": ("s2_0_std_512", "t0_product_name", "s2_0.tif"),
    "seasonal": ("s2_-90_std_512", "seasonal_product_name", "s2_seasonal.tif"),
    "year": ("s2_-360_std_512", "year_product_name", "s2_year.tif"),
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def parse_bounds(value: Any) -> tuple[float, float, float, float]:
    parsed = ast.literal_eval(clean(value))
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        raise ValueError(f"invalid plume_bounds: {value}")
    return tuple(float(item) for item in parsed)


def product_epsg(product_name: Any, latitude: float, longitude: float) -> int:
    match = re.search(r"_T(\d{2})([A-Z])", clean(product_name))
    if match:
        zone = int(match.group(1))
        north = match.group(2) >= "N"
    else:
        zone = max(1, min(60, int((longitude + 180.0) // 6.0) + 1))
        north = latitude >= 0
    return (32600 if north else 32700) + zone


def plume_shift_pixels(
    row: dict[str, Any],
    product_name_column: str,
) -> tuple[int, int, float]:
    latitude = float(row["plume_latitude"])
    longitude = float(row["plume_longitude"])
    min_lon, min_lat, max_lon, max_lat = parse_bounds(row["plume_bounds"])
    epsg = product_epsg(row.get(product_name_column), latitude, longitude)
    xs, ys = transform(
        "EPSG:4326",
        f"EPSG:{epsg}",
        [min_lon, max_lon, longitude],
        [max_lat, min_lat, latitude],
    )
    center_x = (xs[0] + xs[1]) / 2.0
    center_y = (ys[0] + ys[1]) / 2.0
    shift_x = int(round((xs[2] - center_x) / 20.0))
    shift_y = int(round((center_y - ys[2]) / 20.0))
    distance_km = math.hypot(xs[2] - center_x, ys[2] - center_y) / 1000.0
    return shift_x, shift_y, distance_km


def to_chw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"expected 3D image, got {array.shape}")
    if array.shape[0] in {1, 10, 12, 13}:
        return array
    if array.shape[-1] in {1, 10, 12, 13}:
        return np.transpose(array, (2, 0, 1))
    raise ValueError(f"cannot identify channel axis: {array.shape}")


def write_tiff_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{random.randrange(1 << 30)}.part.tif")
    try:
        tifffile.imwrite(temporary, np.asarray(array, dtype=np.float32))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def crop_valid(x: int, y: int, size: int, shift_x: int, shift_y: int) -> bool:
    source_x = x + shift_x
    source_y = y + shift_y
    return (
        source_x >= 0
        and source_y >= 0
        and source_x + size <= 512
        and source_y + size <= 512
    )


def band_valid(crop: np.ndarray) -> bool:
    return float(np.count_nonzero(crop[11] == 0)) / float(crop[11].size) < 0.20


def stable_seed(seed: int, plume_id: str) -> int:
    value = seed
    for character in plume_id.encode("utf-8"):
        value = ((value * 131) + character) & 0xFFFFFFFF
    return value


def select_plumes(
    source: pd.DataFrame,
    patch_ids: set[str],
    maximum: int,
    seed: int,
    min_offset_km: float,
    max_offset_km: float,
    max_per_event: int,
) -> pd.DataFrame:
    selected = source[source["plume_id"].astype(str).isin(patch_ids)].copy()
    offsets = []
    valid = []
    for row in selected.to_dict("records"):
        try:
            shifts = [
                plume_shift_pixels(row, TIMEPOINTS[name][1])
                for name in TIMEPOINTS
            ]
            offsets.append(shifts[0][2])
            valid.append(
                min_offset_km <= shifts[0][2] <= max_offset_km
                and all(abs(shift_x) <= 240 and abs(shift_y) <= 240 for shift_x, shift_y, _ in shifts)
            )
        except Exception:
            offsets.append(float("nan"))
            valid.append(False)
    selected["t0_offset_km"] = offsets
    selected = selected[np.asarray(valid)].copy()
    selected = selected.sample(frac=1.0, random_state=seed)
    selected = selected.groupby("event_group_id", sort=False).head(max_per_event)
    if maximum > 0:
        selected = selected.head(maximum)
    return selected.reset_index(drop=True)


def process_plume(
    source_row: dict[str, Any],
    output_root: Path,
    patches_per_class: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    plume_id = clean(source_row["plume_id"])
    arrays: dict[str, np.ndarray] = {}
    shifts: dict[str, tuple[int, int, float]] = {}
    for name, (path_column, product_column, _) in TIMEPOINTS.items():
        arrays[name] = to_chw(tifffile.imread(clean(source_row[path_column]))).astype(
            np.float32, copy=False
        )
        if arrays[name].shape != (12, 512, 512):
            raise ValueError(f"{plume_id} {name} has shape {arrays[name].shape}")
        shifts[name] = plume_shift_pixels(source_row, product_column)

    mask = np.asarray(tifffile.imread(clean(source_row["resized_512x512_path"])))
    if mask.ndim == 3:
        mask = np.max(to_chw(mask), axis=0)
    if mask.shape != (512, 512):
        raise ValueError(f"{plume_id} mask has shape {mask.shape}")
    mask = np.nan_to_num(mask) > 0

    rng = random.Random(stable_seed(seed, plume_id))
    coordinates: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()

    def acceptable(x: int, y: int) -> bool:
        if (x, y) in seen:
            return False
        for name in TIMEPOINTS:
            shift_x, shift_y, _ = shifts[name]
            if not crop_valid(x, y, 32, shift_x, shift_y):
                return False
            original_crop = arrays[name][:, y : y + 32, x : x + 32]
            recentered_crop = arrays[name][
                :,
                y + shift_y : y + shift_y + 32,
                x + shift_x : x + shift_x + 32,
            ]
            if not band_valid(original_crop) or not band_valid(recentered_crop):
                return False
        return True

    attempts = 0
    while sum(label == 1 for label, _, _ in coordinates) < patches_per_class and attempts < 1000:
        attempts += 1
        x = rng.randint(234, 246)
        y = rng.randint(234, 246)
        if mask[y : y + 32, x : x + 32].sum() <= 0 or not acceptable(x, y):
            continue
        seen.add((x, y))
        coordinates.append((1, x, y))

    attempts = 0
    while sum(label == 0 for label, _, _ in coordinates) < patches_per_class and attempts < 5000:
        attempts += 1
        x = rng.randint(112, 368)
        y = rng.randint(112, 368)
        if mask[y : y + 32, x : x + 32].sum() > 0 or not acceptable(x, y):
            continue
        seen.add((x, y))
        coordinates.append((0, x, y))

    positive_count = sum(label == 1 for label, _, _ in coordinates)
    negative_count = sum(label == 0 for label, _, _ in coordinates)
    if positive_count < patches_per_class or negative_count < patches_per_class:
        return [], [], {
            "plume_id": plume_id,
            "status": "insufficient_valid_patches",
            "positive": positive_count,
            "negative": negative_count,
        }

    recentered_records: list[dict[str, Any]] = []
    original_records: list[dict[str, Any]] = []
    counters = {0: 0, 1: 0}
    for label, crop_x, crop_y in coordinates:
        crop_index = counters[label]
        counters[label] += 1
        kind = "positive" if label == 1 else "negative"
        sample_id = f"{plume_id}__{kind}_{crop_index:02d}_x{crop_x}_y{crop_y}"
        recentered_dir = output_root / "recentered" / plume_id / sample_id
        original_dir = output_root / "original" / plume_id / sample_id
        recentered_paths: dict[str, str] = {}
        original_paths: dict[str, str] = {}
        for name, (_, _, filename) in TIMEPOINTS.items():
            shift_x, shift_y, _ = shifts[name]
            recentered_crop = arrays[name][
                :,
                crop_y + shift_y : crop_y + shift_y + 32,
                crop_x + shift_x : crop_x + shift_x + 32,
            ]
            original_crop = arrays[name][
                :,
                crop_y : crop_y + 32,
                crop_x : crop_x + 32,
            ]
            recentered_destination = recentered_dir / filename
            original_destination = original_dir / filename
            write_tiff_atomic(recentered_destination, recentered_crop)
            write_tiff_atomic(original_destination, original_crop)
            recentered_paths[name] = str(recentered_destination)
            original_paths[name] = str(original_destination)
        common = {
            "sample_id": sample_id,
            "id": sample_id,
            "plume_id": plume_id,
            "event_group_id": clean(source_row.get("event_group_id")),
            "label": label,
            "crop_x": crop_x,
            "crop_y": crop_y,
            "t0_center_offset_km": shifts["t0"][2],
            "t0_shift_x": shifts["t0"][0],
            "t0_shift_y": shifts["t0"][1],
        }
        recentered_records.append(
            {
                **common,
                "path_t0": recentered_paths["t0"],
                "path_prev1": recentered_paths["t0"],
                "path_seasonal": recentered_paths["seasonal"],
                "path_year": recentered_paths["year"],
            }
        )
        original_records.append(
            {
                **common,
                "path_t0": original_paths["t0"],
                "path_prev1": original_paths["t0"],
                "path_seasonal": original_paths["seasonal"],
                "path_year": original_paths["year"],
            }
        )
    return recentered_records, original_records, {
        "plume_id": plume_id,
        "status": "ok",
        "patches": len(recentered_records),
        **{
            f"{name}_shift_x": shifts[name][0]
            for name in TIMEPOINTS
        },
        **{
            f"{name}_shift_y": shifts[name][1]
            for name in TIMEPOINTS
        },
    }


def run_split(
    split: str,
    patch_csv: str,
    source: pd.DataFrame,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    patches = pd.read_csv(patch_csv, low_memory=False)
    patches["label"] = pd.to_numeric(patches["label"], errors="raise").astype(int)
    lower = 256 - args.center_box // 2 - 16
    upper = 256 + args.center_box // 2 - 16
    patches = patches[
        patches["label"].eq(1)
        | (
            patches["label"].eq(0)
            & patches["crop_x"].between(lower, upper)
            & patches["crop_y"].between(lower, upper)
        )
    ].copy()
    plume_ids = set(patches["plume_id"].astype(str))
    selected = select_plumes(
        source,
        plume_ids,
        args.max_train_plumes if split == "train" else args.max_test_plumes,
        args.seed + (0 if split == "train" else 1000),
        args.min_offset_km,
        args.max_offset_km,
        args.max_per_event,
    )
    recentered: list[dict[str, Any]] = []
    original: list[dict[str, Any]] = []
    qa: list[dict[str, Any]] = []
    split_root = output_root / split
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_plume,
                row,
                split_root,
                args.patches_per_class,
                args.seed,
            ): clean(row["plume_id"])
            for row in selected.to_dict("records")
        }
        for index, future in enumerate(as_completed(futures), start=1):
            plume_id = futures[future]
            try:
                new_records, old_records, record = future.result()
                recentered.extend(new_records)
                original.extend(old_records)
                qa.append(record)
            except Exception as error:
                qa.append(
                    {
                        "plume_id": plume_id,
                        "status": "error",
                        "reason": f"{type(error).__name__}:{error}",
                    }
                )
            if index % 25 == 0 or index == len(futures):
                print(
                    f"[{split}] {index}/{len(futures)} successful_plumes="
                    f"{sum(record.get('status') == 'ok' for record in qa)} patches={len(recentered)}",
                    flush=True,
                )
    recentered_frame = pd.DataFrame(recentered).sort_values("sample_id")
    original_frame = pd.DataFrame(original).sort_values("sample_id")
    qa_frame = pd.DataFrame(qa).sort_values("plume_id")
    recentered_csv = output_root / f"{split}_recentered.csv"
    original_csv = output_root / f"{split}_original_matched.csv"
    qa_csv = output_root / f"{split}_qa.csv"
    recentered_frame.to_csv(recentered_csv, index=False)
    original_frame.to_csv(original_csv, index=False)
    qa_frame.to_csv(qa_csv, index=False)
    return {
        "split": split,
        "selected_plumes": len(selected),
        "successful_plumes": int(qa_frame["status"].eq("ok").sum()),
        "rows": len(recentered_frame),
        "labels": recentered_frame["label"].value_counts().sort_index().to_dict(),
        "recentered_csv": str(recentered_csv),
        "original_csv": str(original_csv),
        "qa_csv": str(qa_csv),
    }


def main(args: argparse.Namespace) -> None:
    started = time.time()
    source = pd.read_csv(args.source_csv, low_memory=False)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        run_split("train", args.train_csv, source, output_root, args),
        run_split("test", args.test_csv, source, output_root, args),
    ]
    summary = {
        "seconds": round(time.time() - started, 2),
        "args": vars(args),
        "results": results,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-train-plumes", type=int, default=600)
    parser.add_argument("--max-test-plumes", type=int, default=300)
    parser.add_argument("--max-per-event", type=int, default=2)
    parser.add_argument("--patches-per-class", type=int, default=8)
    parser.add_argument("--center-box", type=int, default=256)
    parser.add_argument("--min-offset-km", type=float, default=0.32)
    parser.add_argument("--max-offset-km", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260723)
    main(parser.parse_args())
