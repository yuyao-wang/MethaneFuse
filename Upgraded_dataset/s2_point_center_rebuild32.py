#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from s2_point_center_recut_diagnostic import (
    band_valid,
    clean,
    plume_shift_pixels,
    stable_seed,
    to_chw,
    write_tiff_atomic,
)


TIMEPOINTS = {
    "t0": ("s2_0_std_512", "t0_product_name", "s2_0.tif", "t0_image_time"),
    "prev1": ("s2_-7_std_512", "prev1_product_name", "s2_prev1.tif", "prev1_image_time"),
    "prev2": ("s2_prev2_std_512", "prev2_product_name", "s2_prev2.tif", "prev2_image_time"),
    "prev3": ("s2_prev3_std_512", "prev3_product_name", "s2_prev3.tif", "prev3_image_time"),
    "seasonal": (
        "s2_-90_std_512",
        "seasonal_product_name",
        "s2_seasonal.tif",
        "seasonal_image_time",
    ),
    "year": ("s2_-360_std_512", "year_product_name", "s2_year.tif", "year_image_time"),
}


def load_mask(path: str) -> np.ndarray:
    mask = np.asarray(tifffile.imread(path))
    if mask.ndim == 3:
        mask = np.max(to_chw(mask), axis=0)
    if mask.shape != (512, 512):
        raise ValueError(f"expected 512x512 mask, got {mask.shape}: {path}")
    return np.nan_to_num(mask) > 0


def valid_coordinate_limits(
    shifts: dict[str, tuple[int, int, float]],
) -> tuple[int, int, int, int]:
    x_min = max(0, max(-shift_x for shift_x, _, _ in shifts.values()))
    x_max = min(480, min(480 - shift_x for shift_x, _, _ in shifts.values()))
    y_min = max(0, max(-shift_y for _, shift_y, _ in shifts.values()))
    y_max = min(480, min(480 - shift_y for _, shift_y, _ in shifts.values()))
    return x_min, x_max, y_min, y_max


def crops_valid(
    arrays: dict[str, np.ndarray],
    shifts: dict[str, tuple[int, int, float]],
    crop_x: int,
    crop_y: int,
) -> bool:
    for name, array in arrays.items():
        shift_x, shift_y, _ = shifts[name]
        crop = array[
            :,
            crop_y + shift_y : crop_y + shift_y + 32,
            crop_x + shift_x : crop_x + shift_x + 32,
        ]
        if crop.shape != (12, 32, 32) or not band_valid(crop):
            return False
    return True


def choose_coordinates(
    arrays: dict[str, np.ndarray],
    mask: np.ndarray,
    shifts: dict[str, tuple[int, int, float]],
    plume_id: str,
    seed: int,
    count: int,
) -> list[tuple[int, int, int, int]]:
    x_min, x_max, y_min, y_max = valid_coordinate_limits(shifts)
    positive_x_min = max(234, x_min)
    positive_x_max = min(246, x_max)
    positive_y_min = max(234, y_min)
    positive_y_max = min(246, y_max)
    if (
        positive_x_min > positive_x_max
        or positive_y_min > positive_y_max
        or x_min > x_max
        or y_min > y_max
    ):
        raise ValueError(
            f"plume center is outside recoverable intersection: "
            f"x={x_min}:{x_max} y={y_min}:{y_max}"
        )

    rng = random.Random(stable_seed(seed, plume_id))
    coordinates: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()

    attempts = 0
    while sum(label == 1 for label, _, _, _ in coordinates) < count and attempts < 2000:
        attempts += 1
        crop_x = rng.randint(positive_x_min, positive_x_max)
        crop_y = rng.randint(positive_y_min, positive_y_max)
        if (crop_x, crop_y) in seen:
            continue
        mask_crop = mask[crop_y : crop_y + 32, crop_x : crop_x + 32]
        if mask_crop.sum() <= 0 or not crops_valid(arrays, shifts, crop_x, crop_y):
            continue
        seen.add((crop_x, crop_y))
        coordinates.append((1, crop_x, crop_y, int(mask_crop.sum())))

    attempts = 0
    while sum(label == 0 for label, _, _, _ in coordinates) < count and attempts < 10000:
        attempts += 1
        crop_x = rng.randint(x_min, x_max)
        crop_y = rng.randint(y_min, y_max)
        if (crop_x, crop_y) in seen:
            continue
        mask_crop = mask[crop_y : crop_y + 32, crop_x : crop_x + 32]
        if mask_crop.sum() > 0 or not crops_valid(arrays, shifts, crop_x, crop_y):
            continue
        seen.add((crop_x, crop_y))
        coordinates.append((0, crop_x, crop_y, 0))

    positive_count = sum(label == 1 for label, _, _, _ in coordinates)
    negative_count = sum(label == 0 for label, _, _, _ in coordinates)
    if positive_count != count or negative_count != count:
        raise ValueError(
            f"insufficient coordinates positive={positive_count}/{count} "
            f"negative={negative_count}/{count}"
        )
    return coordinates


def build_record(
    row: dict[str, Any],
    split: str,
    label: int,
    index: int,
    crop_x: int,
    crop_y: int,
    mask_sum: int,
    image_paths: dict[str, str],
    mask_path: str,
    shifts: dict[str, tuple[int, int, float]],
) -> dict[str, Any]:
    kind = "positive" if label == 1 else "negative"
    sample_id = f"{row['plume_id']}__{kind}_{index:02d}_x{crop_x}_y{crop_y}"
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "id": sample_id,
        "plume_id": row["plume_id"],
        "event_group_id": row["event_group_id"],
        "cohort": row.get("cohort", "cdse_raw_or_legacy512"),
        "source_pattern": row.get("source_pattern", ""),
        "split": split,
        "label": label,
        "crop_kind": kind,
        "crop_index": index,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "plume_mask_sum": mask_sum,
        "path_plume": mask_path,
        "plume_mask_path": mask_path,
        "mask_path": mask_path,
        "mask_path_512": row["resized_512x512_path"],
        "plume_latitude": row["plume_latitude"],
        "plume_longitude": row["plume_longitude"],
        "latitude": row["plume_latitude"],
        "longitude": row["plume_longitude"],
        "event_time": row["event_time"],
        "datetime": row["event_time"],
        "source": "bounds_center_recovered_to_plume_point",
        "image_center_mode": "plume_point",
    }
    for name, (_, _, _, time_column) in TIMEPOINTS.items():
        record[f"path_{name}"] = image_paths[name]
        record[time_column] = row[time_column]
        record[f"{name}_shift_x"] = shifts[name][0]
        record[f"{name}_shift_y"] = shifts[name][1]
    record["image_path"] = record["path_t0"]
    record["s2_path"] = record["path_t0"]
    record["s2_-7_path"] = record["path_prev1"]
    record["s2_pre_path"] = record["path_seasonal"]
    record["s2_pre_pre_path"] = record["path_year"]
    return record


def process_plume(
    row: dict[str, Any],
    split: str,
    output_root: Path,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plume_id = clean(row["plume_id"])
    arrays: dict[str, np.ndarray] = {}
    shifts: dict[str, tuple[int, int, float]] = {}
    for name, (path_column, product_column, _, _) in TIMEPOINTS.items():
        array = to_chw(tifffile.imread(clean(row[path_column]))).astype(np.float32, copy=False)
        if array.shape != (12, 512, 512):
            raise ValueError(f"{name} has shape {array.shape}")
        arrays[name] = array
        shifts[name] = plume_shift_pixels(row, product_column)
    mask = load_mask(clean(row["resized_512x512_path"]))
    coordinates = choose_coordinates(arrays, mask, shifts, plume_id, seed, count)

    records: list[dict[str, Any]] = []
    counters = {0: 0, 1: 0}
    for label, crop_x, crop_y, mask_sum in coordinates:
        index = counters[label]
        counters[label] += 1
        kind = "positive" if label == 1 else "negative"
        sample_dir = output_root / split / plume_id / f"{kind}_{index:02d}_x{crop_x}_y{crop_y}"
        image_paths: dict[str, str] = {}
        for name, (_, _, filename, _) in TIMEPOINTS.items():
            shift_x, shift_y, _ = shifts[name]
            crop = arrays[name][
                :,
                crop_y + shift_y : crop_y + shift_y + 32,
                crop_x + shift_x : crop_x + shift_x + 32,
            ]
            destination = sample_dir / filename
            write_tiff_atomic(destination, crop)
            image_paths[name] = str(destination)
        mask_crop = (
            mask[crop_y : crop_y + 32, crop_x : crop_x + 32].astype(np.uint8)
            if label == 1
            else np.zeros((32, 32), dtype=np.uint8)
        )
        mask_destination = sample_dir / "plume.tif"
        write_tiff_atomic(mask_destination, mask_crop[None, :, :])
        records.append(
            build_record(
                row,
                split,
                label,
                index,
                crop_x,
                crop_y,
                mask_sum,
                image_paths,
                str(mask_destination),
                shifts,
            )
        )
    return records, {
        "plume_id": plume_id,
        "event_group_id": row["event_group_id"],
        "split": split,
        "status": "ok",
        "rows": len(records),
        "t0_offset_km": shifts["t0"][2],
        "max_abs_shift": max(
            max(abs(shift_x), abs(shift_y))
            for shift_x, shift_y, _ in shifts.values()
        ),
    }


def run_split(
    split: str,
    split_csv: str,
    source: pd.DataFrame,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_ids = set(
        pd.read_csv(split_csv, usecols=["plume_id"], low_memory=False)["plume_id"].astype(str)
    )
    selected = source[source["plume_id"].astype(str).isin(split_ids)].copy()
    selected = selected.sort_values(["event_time", "plume_id"], kind="stable")
    if args.limit > 0:
        selected = selected.head(args.limit)
    records: list[dict[str, Any]] = []
    qa: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_plume,
                row,
                split,
                output_root,
                args.patches_per_class,
                args.seed,
            ): clean(row["plume_id"])
            for row in selected.to_dict("records")
        }
        for index, future in enumerate(as_completed(futures), start=1):
            plume_id = futures[future]
            try:
                plume_records, qa_record = future.result()
                records.extend(plume_records)
                qa.append(qa_record)
            except Exception as error:
                qa.append(
                    {
                        "plume_id": plume_id,
                        "split": split,
                        "status": "fail",
                        "reason": f"{type(error).__name__}:{error}",
                    }
                )
            if index % args.progress_every == 0 or index == len(futures):
                successful = sum(record["status"] == "ok" for record in qa)
                print(
                    f"[{split}] {index}/{len(futures)} successful={successful} "
                    f"failed={len(qa) - successful} rows={len(records)}",
                    flush=True,
                )
    manifest = pd.DataFrame(records).sort_values(
        ["event_group_id", "plume_id", "label", "crop_index"], kind="stable"
    )
    qa_frame = pd.DataFrame(qa).sort_values("plume_id", kind="stable")
    manifest_path = output_root / f"{split}.csv"
    qa_path = output_root / f"{split}_qa.csv"
    manifest.to_csv(manifest_path, index=False)
    qa_frame.to_csv(qa_path, index=False)
    return {
        "split": split,
        "input_plumes": len(selected),
        "successful_plumes": int(qa_frame["status"].eq("ok").sum()),
        "failed_plumes": int(qa_frame["status"].ne("ok").sum()),
        "rows": len(manifest),
        "labels": manifest["label"].value_counts().sort_index().to_dict(),
        "manifest": str(manifest_path),
        "qa": str(qa_path),
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
    train_groups = set(pd.read_csv(results[0]["manifest"])["event_group_id"].astype(str))
    test_groups = set(pd.read_csv(results[1]["manifest"])["event_group_id"].astype(str))
    overlap = sorted(train_groups & test_groups)
    summary = {
        "seconds": round(time.time() - started, 2),
        "event_group_overlap": len(overlap),
        "event_group_overlap_examples": overlap[:20],
        "results": results,
        "args": vars(args),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if overlap:
        raise RuntimeError(f"event-group leakage detected: {overlap[:20]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--patches-per-class", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int, default=0)
    main(parser.parse_args())
