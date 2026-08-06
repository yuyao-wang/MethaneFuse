#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import tifffile
from affine import Affine
from pyproj import Transformer


TIMEPOINT_PATH_COLUMNS = [
    "t0_raw_path",
    "prev1_raw_path",
    "prev2_raw_path",
    "prev3_raw_path",
    "seasonal_raw_path",
    "year_raw_path",
]
PATCH_PATH_COLUMNS = [
    "path_t0",
    "path_prev1",
    "path_prev2",
    "path_prev3",
    "path_seasonal",
    "path_year",
]
VALID_BANDS = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)


def to_chw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"expected 3D image, got {array.shape}")
    if array.shape[0] in (10, 12, 13):
        return array
    if array.shape[-1] in (10, 12, 13):
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"cannot infer channel axis for {array.shape}")


def file_ok(value: Any) -> bool:
    try:
        path = Path(str(value))
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def sampled(values: list[str], maximum: int, seed: int) -> list[str]:
    if maximum <= 0 or len(values) <= maximum:
        return values
    return random.Random(seed).sample(values, maximum)


def audit_image(path_text: str, expected_size: int) -> str:
    try:
        image = to_chw(tifffile.imread(path_text))
        if image.shape != (12, expected_size, expected_size):
            return f"shape={image.shape}"
        empty = [band for band in VALID_BANDS if not np.any(image[band])]
        if empty:
            return f"empty_valid_bands={empty}"
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}:{exc}"


def write_report(report: dict[str, Any], output: str) -> None:
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


def audit_recrop(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.csv, low_memory=False)
    failures: list[str] = []
    if len(frame) != args.expected_rows:
        failures.append(f"rows={len(frame)} expected={args.expected_rows}")

    all_paths: list[str] = []
    for column in TIMEPOINT_PATH_COLUMNS:
        if column not in frame.columns:
            failures.append(f"missing_column={column}")
            continue
        all_paths.extend(frame[column].astype(str).tolist())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        flags = list(pool.map(file_ok, all_paths))
    missing_files = int(len(flags) - sum(flags))
    if missing_files:
        failures.append(f"missing_files={missing_files}")

    t0_sidecars_missing = 0
    center_offsets: list[float] = []
    transformers: dict[str, Transformer] = {}
    for row in frame.to_dict("records"):
        t0_path = Path(str(row.get("t0_raw_path", "")))
        sidecar = t0_path.with_name(t0_path.name + ".georef.json")
        if not file_ok(sidecar):
            t0_sidecars_missing += 1
            continue
        metadata = json.loads(sidecar.read_text())
        transform = Affine(*[float(value) for value in metadata["transform"][:6]])
        crs_wkt = str(metadata["crs_wkt"])
        transformer = transformers.get(crs_wkt)
        if transformer is None:
            transformer = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)
            transformers[crs_wkt] = transformer
        x, y = transformer.transform(
            float(row["plume_longitude"]),
            float(row["plume_latitude"]),
        )
        col, image_row = (~transform) * (x, y)
        center_offsets.append(max(abs(float(col) - 256.0), abs(float(image_row) - 256.0)))
    if t0_sidecars_missing:
        failures.append(f"missing_t0_sidecars={t0_sidecars_missing}")
    max_center_offset = max(center_offsets, default=math.inf)
    if max_center_offset > args.max_center_offset:
        failures.append(
            f"max_center_offset={max_center_offset:.6f}>{args.max_center_offset}"
        )

    image_paths = sampled(all_paths, args.sample_images, args.seed)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        image_errors = list(
            pool.map(lambda path: audit_image(path, 512), image_paths)
        )
    bad_images = [
        {"path": path, "error": error}
        for path, error in zip(image_paths, image_errors)
        if error
    ]
    if bad_images:
        failures.append(f"bad_sampled_images={len(bad_images)}")

    report = {
        "kind": "recrop",
        "rows": len(frame),
        "files": len(all_paths),
        "missing_files": missing_files,
        "t0_sidecars_missing": t0_sidecars_missing,
        "max_center_offset_pixels": max_center_offset,
        "sampled_images": len(image_paths),
        "bad_sampled_images": bad_images[:20],
        "failures": failures,
    }
    write_report(report, args.output)
    return 1 if failures else 0


def read_mask_metrics(path_text: str, center_box: int) -> tuple[int, int, str]:
    try:
        with rasterio.open(path_text) as dataset:
            mask = dataset.read(1)
        if mask.shape != (512, 512):
            return 0, 0, f"shape={mask.shape}"
        mask = np.asarray(mask) > 0
        half = center_box // 2
        center = mask[256 - half : 256 + half, 256 - half : 256 + half]
        return int(mask.sum()), int(center.sum()), ""
    except Exception as exc:
        return 0, 0, f"{type(exc).__name__}:{exc}"


def audit_masks(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.csv, low_memory=False)
    failures: list[str] = []
    if len(frame) != args.expected_rows:
        failures.append(f"rows={len(frame)} expected={args.expected_rows}")
    if "s2_mask_512_path" not in frame.columns:
        failures.append("missing_column=s2_mask_512_path")
        paths: list[str] = []
    else:
        paths = frame["s2_mask_512_path"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        metrics = list(
            pool.map(lambda path: read_mask_metrics(path, args.center_box), paths)
        )
    zero_masks = sum(total <= 0 for total, _, _ in metrics)
    no_center_signal = sum(center <= 0 for _, center, _ in metrics)
    read_errors = [
        {"path": path, "error": error}
        for path, (_, _, error) in zip(paths, metrics)
        if error
    ]
    if zero_masks:
        failures.append(f"zero_masks={zero_masks}")
    if no_center_signal:
        failures.append(f"masks_without_center_signal={no_center_signal}")
    if read_errors:
        failures.append(f"mask_read_errors={len(read_errors)}")
    report = {
        "kind": "masks",
        "rows": len(frame),
        "zero_masks": zero_masks,
        "masks_without_center_signal": no_center_signal,
        "positive_pixels_min": min((value[0] for value in metrics), default=0),
        "positive_pixels_median": float(
            np.median([value[0] for value in metrics])
        )
        if metrics
        else 0.0,
        "read_errors": read_errors[:20],
        "failures": failures,
    }
    write_report(report, args.output)
    return 1 if failures else 0


def audit_split(args: argparse.Namespace) -> int:
    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False)
    failures: list[str] = []
    for column in ("event_group_id", "plume_id"):
        if column not in train.columns or column not in test.columns:
            failures.append(f"missing_column={column}")
    event_overlap = set(train["event_group_id"].astype(str)) & set(
        test["event_group_id"].astype(str)
    )
    plume_overlap = set(train["plume_id"].astype(str)) & set(
        test["plume_id"].astype(str)
    )
    ratio = len(train) / max(1, len(train) + len(test))
    if event_overlap:
        failures.append(f"event_overlap={len(event_overlap)}")
    if plume_overlap:
        failures.append(f"plume_overlap={len(plume_overlap)}")
    if not args.min_ratio <= ratio <= args.max_ratio:
        failures.append(f"train_ratio={ratio:.6f}")
    report = {
        "kind": "split",
        "train_rows": len(train),
        "test_rows": len(test),
        "train_ratio": ratio,
        "train_events": int(train["event_group_id"].nunique()),
        "test_events": int(test["event_group_id"].nunique()),
        "event_overlap": len(event_overlap),
        "plume_overlap": len(plume_overlap),
        "failures": failures,
    }
    write_report(report, args.output)
    return 1 if failures else 0


def audit_patch_row(
    row: dict[str, Any],
    expected_size: int,
    check_mask_label_consistency: bool,
    check_all_valid_bands: bool,
    valid_bands: tuple[int, ...],
    quality_path_columns: set[str],
    quality_band_index: int,
    quality_zero_ratio_thresh: float,
) -> str:
    label = int(row["label"])
    try:
        mask = np.asarray(tifffile.imread(str(row["path_plume"])))
        if mask.shape != (expected_size, expected_size):
            return f"mask_shape={mask.shape}"
        mask_sum = float(mask.sum())
        if check_mask_label_consistency:
            if label == 1 and mask_sum <= 0:
                return "positive_mask_is_zero"
            if label == 0 and mask_sum != 0:
                return f"negative_mask_sum={mask_sum}"
        for column in PATCH_PATH_COLUMNS:
            image = to_chw(tifffile.imread(str(row[column])))
            if image.shape != (12, expected_size, expected_size):
                return f"{column}:shape={image.shape}"
            if check_all_valid_bands:
                empty = [band for band in valid_bands if not np.any(image[band])]
                if empty:
                    return f"{column}:empty_valid_bands={empty}"
            if column in quality_path_columns:
                if quality_band_index >= image.shape[0]:
                    return (
                        f"{column}:missing_quality_band="
                        f"{quality_band_index}/{image.shape[0]}"
                    )
                zero_ratio = float(
                    (image[quality_band_index] == 0).mean()
                )
                if zero_ratio >= quality_zero_ratio_thresh:
                    return (
                        f"{column}:quality_band_zero_ratio="
                        f"{zero_ratio:.6f}"
                    )
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}:{exc}"


def audit_patches(args: argparse.Namespace) -> int:
    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False)
    failures: list[str] = []
    event_overlap = set(train["event_group_id"].astype(str)) & set(
        test["event_group_id"].astype(str)
    )
    plume_overlap = set(train["plume_id"].astype(str)) & set(
        test["plume_id"].astype(str)
    )
    if event_overlap:
        failures.append(f"event_overlap={len(event_overlap)}")
    if plume_overlap:
        failures.append(f"plume_overlap={len(plume_overlap)}")
    expected_per_plume = args.n_pos + args.n_neg
    plume_count_stats: dict[str, dict[str, float | int]] = {}
    for split_name, frame in (("train", train), ("test", test)):
        counts = frame.groupby("plume_id").size()
        plume_count_stats[split_name] = {
            "plumes": int(len(counts)),
            "minimum": int(counts.min()) if len(counts) else 0,
            "median": float(counts.median()) if len(counts) else 0.0,
            "maximum": int(counts.max()) if len(counts) else 0,
            "partial": int((counts < expected_per_plume).sum()),
        }
        if args.allow_partial_plumes:
            bad_counts = int(
                ((counts < 1) | (counts > expected_per_plume)).sum()
            )
            bad_balance = 0
        else:
            bad_counts = int((counts != expected_per_plume).sum())
            labels = frame.groupby(["plume_id", "label"]).size().unstack(fill_value=0)
            bad_balance = int(
                (
                    labels.reindex(columns=[0, 1], fill_value=0)
                    != [args.n_neg, args.n_pos]
                ).any(axis=1).sum()
            )
        if bad_counts:
            failures.append(f"{split_name}_bad_plume_counts={bad_counts}")
        if bad_balance:
            failures.append(f"{split_name}_bad_label_balance={bad_balance}")

    combined = pd.concat([train, test], ignore_index=True)
    quality_path_columns = {
        value.strip()
        for value in args.quality_path_columns.split(",")
        if value.strip()
    }
    try:
        valid_bands = tuple(
            int(value.strip())
            for value in args.valid_band_indices.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid --valid-band-indices={args.valid_band_indices!r}"
        ) from exc
    invalid_bands = sorted(
        band for band in valid_bands if band < 0 or band >= 12
    )
    if invalid_bands:
        failures.append(f"invalid_valid_band_indices={invalid_bands}")
    unknown_quality_columns = sorted(
        quality_path_columns - set(PATCH_PATH_COLUMNS)
    )
    if unknown_quality_columns:
        failures.append(
            f"unknown_quality_path_columns={unknown_quality_columns}"
        )
    if args.notebook_cell7_geometry:
        required = {"source", "crop_x", "crop_y", "label"}
        missing = sorted(required - set(combined.columns))
        if missing:
            failures.append(f"notebook_geometry_missing_columns={missing}")
        else:
            positive = combined["source"].eq("notebook_cell7_positive")
            random_source = combined["source"].eq("notebook_cell7_random")
            unknown_source = ~(positive | random_source)
            bad_positive = positive & (
                combined["label"].ne(1)
                | ~combined["crop_x"].between(234, 246)
                | ~combined["crop_y"].between(234, 246)
            )
            expected_random_label = (
                combined["crop_x"].le(251)
                & combined["crop_y"].le(251)
                & (combined["crop_x"] + 32).ge(261)
                & (combined["crop_y"] + 32).ge(261)
            ).astype(int)
            bad_random = random_source & combined["label"].ne(
                expected_random_label
            )
            if int(unknown_source.sum()):
                failures.append(
                    f"notebook_geometry_unknown_source={int(unknown_source.sum())}"
                )
            if int(bad_positive.sum()):
                failures.append(
                    f"notebook_geometry_bad_positive={int(bad_positive.sum())}"
                )
            if int(bad_random.sum()):
                failures.append(
                    f"notebook_geometry_bad_random={int(bad_random.sum())}"
                )

    if args.sample_rows <= 0 or args.sample_rows >= len(combined):
        sampled_frame = combined
    else:
        sampled_frame = combined.sample(
            n=args.sample_rows,
            random_state=args.seed,
        )
    rows = sampled_frame.to_dict("records")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        errors = list(
            pool.map(
                lambda row: audit_patch_row(
                    row,
                    args.expected_size,
                    args.check_mask_label_consistency,
                    args.check_all_valid_bands,
                    valid_bands,
                    quality_path_columns,
                    args.quality_band_index,
                    args.quality_zero_ratio_thresh,
                ),
                rows,
            )
        )
    bad_rows = [
        {
            "split": row["split"],
            "plume_id": row["plume_id"],
            "error": error,
        }
        for row, error in zip(rows, errors)
        if error
    ]
    bad_plume_ids_by_split = {
        split_name: sorted(
            {
                str(row["plume_id"])
                for row in bad_rows
                if str(row["split"]) == split_name
            }
        )
        for split_name in ("train", "test")
    }
    if bad_rows:
        failures.append(f"bad_sampled_rows={len(bad_rows)}")
    report = {
        "kind": "patches",
        "train_rows": len(train),
        "test_rows": len(test),
        "train_labels": train["label"].value_counts().sort_index().to_dict(),
        "test_labels": test["label"].value_counts().sort_index().to_dict(),
        "plume_count_stats": plume_count_stats,
        "event_overlap": len(event_overlap),
        "plume_overlap": len(plume_overlap),
        "sampled_rows": len(rows),
        "valid_band_indices": valid_bands,
        "bad_rows": len(bad_rows),
        "bad_sampled_rows": bad_rows[:20],
        "bad_plume_ids_by_split": bad_plume_ids_by_split,
        "failures": failures,
    }
    write_report(report, args.output)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    recrop = subparsers.add_parser("recrop")
    recrop.add_argument("--csv", required=True)
    recrop.add_argument("--expected-rows", type=int, default=4450)
    recrop.add_argument("--max-center-offset", type=float, default=1.5)
    recrop.add_argument("--sample-images", type=int, default=1000)
    recrop.add_argument("--workers", type=int, default=32)
    recrop.add_argument("--seed", type=int, default=20260723)
    recrop.add_argument("--output", default="")
    recrop.set_defaults(func=audit_recrop)

    masks = subparsers.add_parser("masks")
    masks.add_argument("--csv", required=True)
    masks.add_argument("--expected-rows", type=int, default=4450)
    masks.add_argument("--center-box", type=int, default=256)
    masks.add_argument("--workers", type=int, default=32)
    masks.add_argument("--output", default="")
    masks.set_defaults(func=audit_masks)

    split = subparsers.add_parser("split")
    split.add_argument("--train-csv", required=True)
    split.add_argument("--test-csv", required=True)
    split.add_argument("--min-ratio", type=float, default=0.80)
    split.add_argument("--max-ratio", type=float, default=0.90)
    split.add_argument("--output", default="")
    split.set_defaults(func=audit_split)

    patches = subparsers.add_parser("patches")
    patches.add_argument("--train-csv", required=True)
    patches.add_argument("--test-csv", required=True)
    patches.add_argument("--n-pos", type=int, default=16)
    patches.add_argument("--n-neg", type=int, default=16)
    patches.add_argument(
        "--allow-partial-plumes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    patches.add_argument(
        "--notebook-cell7-geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    patches.add_argument("--expected-size", type=int, default=32)
    patches.add_argument(
        "--check-mask-label-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    patches.add_argument(
        "--check-all-valid-bands",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    patches.add_argument(
        "--valid-band-indices",
        default=",".join(str(band) for band in VALID_BANDS),
    )
    patches.add_argument("--quality-path-columns", default="")
    patches.add_argument("--quality-band-index", type=int, default=11)
    patches.add_argument(
        "--quality-zero-ratio-thresh",
        type=float,
        default=0.20,
    )
    patches.add_argument("--sample-rows", type=int, default=2000)
    patches.add_argument("--workers", type=int, default=32)
    patches.add_argument("--seed", type=int, default=20260723)
    patches.add_argument("--output", default="")
    patches.set_defaults(func=audit_patches)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
