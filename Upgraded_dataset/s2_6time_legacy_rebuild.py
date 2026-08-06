#!/usr/bin/env python3
"""Rebuild the six-time S2 dataset with the legacy notebook geometry and labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
import tifffile
from affine import Affine
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform
from rasterio.windows import Window


METHANE_ROOT = Path("/home/yuyao/methane_train")
CSV_ROOT = METHANE_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_legacy_rebuild"

DEFAULT_LEGACY_CSV = METHANE_ROOT / "preprocess_dataset_s2" / "CM_S2_L2A_-7_gee90360_std512.csv"
DEFAULT_LEGACY_MASK_CSV = METHANE_ROOT / "preprocess_dataset_s2" / "raw_s2_90360_cleaned_fixed.csv"
DEFAULT_NEW_CSV = METHANE_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_all6_available_paths_std512_complete.csv"
DEFAULT_MAIN_CSV = METHANE_ROOT / "Upgrade_data_pipeline" / "csv" / "carbon_mapper_plumes_20160101_20260530_with_t0_flags.csv"
DEFAULT_GEE_LOCAL_CSV = METHANE_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_gee_local_paths.csv"
DEFAULT_GEE_EXPORT_MANIFESTS = [
    METHANE_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_gee_export_manifest.csv",
    METHANE_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_gee_export_retry.csv",
    CSV_ROOT / "s2_6time_legacy_prev123_gee_export_manifest.csv",
]
DEFAULT_LEGACY_512_ROOT = Path("/mnt/engg-niulab/yuyao/preprocessed_512/S2")
DEFAULT_GEE_RAW_ROOT = Path("/mnt/engg-niulab/yuyao/sensors_raw_data/S2_GEE_6time")
DEFAULT_CM_ROOT = Path("/mnt/engg-niulab/yuyao/sensors_raw_data/CM")
DEFAULT_OUT_512_ROOT = Path("/mnt/engg-niulab/yuyao/preprocessed_512/S2_6time_legacy_exact")
DEFAULT_OUT_32_ROOT = Path("/mnt/engg-niulab/yuyao/final_crop/s2_6time_legacy_exact_32")
DEFAULT_OUT_224_ROOT = Path("/mnt/engg-niulab/yuyao/final_crop/s2_6time_legacy_exact_32_to_224")

DEFAULT_SOURCE_CSV = CSV_ROOT / "s2_6time_legacy_exact_sources.csv"
DEFAULT_NEW_SOURCE_AUDIT = CSV_ROOT / "s2_6time_new_all6_cdse_sources.csv"
DEFAULT_LEGACY_EXPORT_INPUT = CSV_ROOT / "s2_6time_legacy_prev123_gee_export_input.csv"
DEFAULT_QA_512_CSV = CSV_ROOT / "s2_6time_legacy_exact_512_qa.csv"
DEFAULT_COMPLETE_512_CSV = CSV_ROOT / "s2_6time_legacy_exact_512_complete.csv"
DEFAULT_SPLIT_ROOT = CSV_ROOT / "temporal_split"

WINDOW_SIZE = 512
PATCH_SIZE = 32
TARGET_SIZE = 224
EXPECTED_BANDS = 12


@dataclass(frozen=True)
class Timepoint:
    name: str
    raw_filename: str
    std_col: str
    std_filename: str
    image_time_col: str
    nominal_offset_days: int


TIMEPOINTS = [
    Timepoint("t0", "s2_0.tif", "s2_0_std_512", "s2_0_std_512.tif", "t0_image_time", 0),
    Timepoint("prev1", "s2_prev1.tif", "s2_-7_std_512", "s2_-7_std_512.tif", "prev1_image_time", -7),
    Timepoint("prev2", "s2_prev2.tif", "s2_prev2_std_512", "s2_prev2_std_512.tif", "prev2_image_time", -14),
    Timepoint("prev3", "s2_prev3.tif", "s2_prev3_std_512", "s2_prev3_std_512.tif", "prev3_image_time", -21),
    Timepoint("seasonal", "s2_seasonal.tif", "s2_-90_std_512", "s2_-90_std_512.tif", "seasonal_image_time", -90),
    Timepoint("year", "s2_year.tif", "s2_-360_std_512", "s2_-360_std_512.tif", "year_image_time", -360),
]

LEGACY_EXISTING_TIMEPOINTS = {"t0", "seasonal", "year"}
GEE_FILL_TIMEPOINTS = {"prev1", "prev2", "prev3"}
LEGACY_COHORT = "legacy_existing3_plus_gee_prev123"
NEW_COHORT = "new_all6_cdse"
PATH_COLUMNS = {
    "t0": "path_t0",
    "prev1": "path_prev1",
    "prev2": "path_prev2",
    "prev3": "path_prev3",
    "seasonal": "path_seasonal",
    "year": "path_year",
}


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def file_ok(value: Any) -> bool:
    text = clean(value)
    if not text:
        return False
    try:
        path = Path(text)
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def parse_time(value: Any) -> pd.Timestamp:
    text = clean(value)
    if not text:
        return pd.NaT
    try:
        timestamp = pd.to_datetime(text, utc=True, errors="coerce", format="mixed")
    except TypeError:
        timestamp = pd.NaT
    if pd.isna(timestamp):
        timestamp = pd.to_datetime(text, utc=True, errors="coerce")
    return timestamp


def iso_time(value: Any) -> str:
    timestamp = parse_time(value)
    if pd.isna(timestamp):
        return ""
    return timestamp.isoformat()


def nominal_time(event_time: Any, days: int) -> str:
    timestamp = parse_time(event_time)
    if pd.isna(timestamp):
        return ""
    return (timestamp + pd.Timedelta(days=days)).isoformat()


def fallback_group_id(plume_id: str) -> str:
    parts = plume_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1] and len(parts[1]) <= 4:
        return parts[0]
    return plume_id


def stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def expected_gee_path(gee_root: Path, plume_id: str, tp: Timepoint) -> Path:
    return gee_root / tp.name / plume_id / tp.raw_filename


def expected_legacy_512_path(legacy_root: Path, plume_id: str, tp: Timepoint) -> Path:
    return legacy_root / plume_id / tp.std_filename


def expected_output_512_path(out_root: Path, plume_id: str, tp: Timepoint) -> Path:
    return out_root / plume_id / tp.std_filename


def load_main_metadata(path: Path) -> dict[str, dict[str, str]]:
    columns = [
        "plume_id",
        "event_group_id",
        "plume_latitude",
        "plume_longitude",
        "datetime",
        "plume_bounds",
        "s2_t0_time",
    ]
    df = pd.read_csv(path, usecols=lambda column: column in columns, low_memory=False)
    records: dict[str, dict[str, str]] = {}
    for row in df.to_dict("records"):
        plume_id = clean(row.get("plume_id"))
        if plume_id:
            records[plume_id] = {key: clean(value) for key, value in row.items()}
    return records


def load_legacy_masks(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, usecols=lambda column: column in {"plume_id", "resized_512x512_path"}, low_memory=False)
    return {
        clean(row["plume_id"]): clean(row["resized_512x512_path"])
        for row in df.to_dict("records")
        if clean(row.get("plume_id"))
    }


def truthy_flag(value: Any) -> bool:
    return clean(value).lower() in {"1", "1.0", "true", "yes"}


def load_gee_available(paths: Iterable[str]) -> set[tuple[str, str]]:
    available: set[tuple[str, str]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        for tp in TIMEPOINTS:
            column = f"gee_{tp.name}_raw_path"
            if column not in df.columns:
                continue
            flag_column = next(
                (
                    candidate
                    for candidate in (f"has_{tp.name}_gee_raw", f"has_gee_{tp.name}")
                    if candidate in df.columns
                ),
                "",
            )
            columns = ["plume_id", column, *([flag_column] if flag_column else [])]
            for row in df[columns].to_dict("records"):
                plume_id = clean(row.get("plume_id"))
                value = clean(row.get(column))
                is_available = truthy_flag(row.get(flag_column)) if flag_column else file_ok(value)
                if plume_id and value and is_available:
                    available.add((clean(plume_id), tp.name))
    return available


def load_gee_selected_times(paths: Iterable[str]) -> dict[tuple[str, str], str]:
    selected: dict[tuple[str, str], str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        df = pd.read_csv(
            path,
            usecols=lambda column: column in {"plume_id", "timepoint", "selected_time_utc"},
            low_memory=False,
        )
        if not {"plume_id", "timepoint", "selected_time_utc"}.issubset(df.columns):
            continue
        for plume_id, timepoint, value in df.itertuples(index=False, name=None):
            key = (clean(plume_id), clean(timepoint))
            timestamp = iso_time(value)
            if key[0] and key[1] in PATH_COLUMNS and timestamp:
                selected[key] = timestamp
    return selected


def parallel_file_flags(paths: Iterable[str], workers: int) -> dict[str, int]:
    unique_paths = sorted({clean(path) for path in paths if clean(path)})
    if not unique_paths:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        values = pool.map(file_ok, unique_paths)
        return {path: int(ok) for path, ok in zip(unique_paths, values)}


def base_record(plume_id: str, event_time: Any, latitude: Any, longitude: Any, bounds: Any, group_id: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "plume_id": plume_id,
        "event_group_id": clean(group_id) or fallback_group_id(plume_id),
        "event_time": iso_time(event_time),
        "datetime": iso_time(event_time),
        "plume_latitude": latitude,
        "plume_longitude": longitude,
        "plume_bounds": clean(bounds),
    }
    return record


def build_manifest(args: argparse.Namespace) -> int:
    legacy = pd.read_csv(args.legacy_csv, low_memory=False)
    new = pd.read_csv(args.new_csv, low_memory=False)
    main = load_main_metadata(Path(args.main_csv))
    legacy_masks = load_legacy_masks(Path(args.legacy_mask_csv))
    gee_available = load_gee_available([args.gee_local_csv, *args.extra_gee_local_csv])
    gee_selected_times = load_gee_selected_times(args.gee_export_manifest)
    legacy_ids = set(legacy["plume_id"].astype(str))
    new_ids = set(new["plume_id"].astype(str))
    overlap = legacy_ids & new_ids
    if overlap:
        raise RuntimeError(f"legacy/new plume overlap must be empty, found {len(overlap)}")

    legacy_root = Path(args.legacy_512_root)
    gee_root = Path(args.gee_raw_root)
    out_root = Path(args.out_512_root)
    cm_root = Path(args.cm_root)
    records: list[dict[str, Any]] = []

    for row in legacy.to_dict("records"):
        plume_id = clean(row.get("plume_id"))
        metadata = main.get(plume_id, {})
        event_time = clean(row.get("datetime")) or metadata.get("datetime")
        latitude = row.get("plume_latitude", metadata.get("plume_latitude", ""))
        longitude = row.get("plume_longitude", metadata.get("plume_longitude", ""))
        record = base_record(
            plume_id,
            event_time,
            latitude,
            longitude,
            row.get("plume_bounds", metadata.get("plume_bounds", "")),
            metadata.get("event_group_id", ""),
        )
        record["cohort"] = LEGACY_COHORT
        record["t0_image_time"] = iso_time(metadata.get("s2_t0_time")) or record["event_time"]
        record["prev1_image_time"] = nominal_time(event_time, -7)
        record["prev2_image_time"] = nominal_time(event_time, -14)
        record["prev3_image_time"] = nominal_time(event_time, -21)
        record["seasonal_image_time"] = nominal_time(event_time, -90)
        record["year_image_time"] = nominal_time(event_time, -360)
        record["legacy_mask_path"] = legacy_masks.get(plume_id, clean(row.get("resized_512x512_path")))
        record["raw_cm_mask_path"] = str(cm_root / plume_id / "plume.tif")
        for tp in TIMEPOINTS:
            record[tp.image_time_col] = gee_selected_times.get((plume_id, tp.name), record[tp.image_time_col])
            legacy_path = expected_legacy_512_path(legacy_root, plume_id, tp) if tp.name in LEGACY_EXISTING_TIMEPOINTS else Path("")
            gee_path = expected_gee_path(gee_root, plume_id, tp)
            record[f"legacy_{tp.name}_512_path"] = str(legacy_path) if tp.name in LEGACY_EXISTING_TIMEPOINTS else ""
            record[f"cdse_{tp.name}_512_path"] = ""
            record[f"has_cdse_{tp.name}"] = 0
            record[f"gee_{tp.name}_raw_path"] = str(gee_path)
            record[f"has_gee_{tp.name}"] = int((plume_id, tp.name) in gee_available)
            record[tp.std_col] = str(expected_output_512_path(out_root, plume_id, tp))
        record["has_required_legacy3"] = 1
        record["has_required_gee"] = int(all(record[f"has_gee_{name}"] for name in GEE_FILL_TIMEPOINTS))
        record["has_required_cdse6"] = 1
        records.append(record)

    for row in new.to_dict("records"):
        plume_id = clean(row.get("plume_id"))
        metadata = main.get(plume_id, {})
        event_time = clean(row.get("event_time")) or clean(row.get("datetime")) or metadata.get("datetime")
        record = base_record(
            plume_id,
            event_time,
            row.get("plume_latitude", metadata.get("plume_latitude", "")),
            row.get("plume_longitude", metadata.get("plume_longitude", "")),
            row.get("plume_bounds", metadata.get("plume_bounds", "")),
            metadata.get("event_group_id", ""),
        )
        record["cohort"] = NEW_COHORT
        for tp in TIMEPOINTS:
            record[tp.image_time_col] = iso_time(row.get(tp.image_time_col)) or nominal_time(
                event_time, tp.nominal_offset_days
            )
        record["legacy_mask_path"] = ""
        record["raw_cm_mask_path"] = str(cm_root / plume_id / "plume.tif")
        for tp in TIMEPOINTS:
            cdse_path = clean(row.get(tp.std_col)) or clean(row.get(f"{tp.name}_512_path"))
            record[f"legacy_{tp.name}_512_path"] = ""
            record[f"cdse_{tp.name}_512_path"] = cdse_path
            record[f"has_cdse_{tp.name}"] = 0
            record[f"gee_{tp.name}_raw_path"] = ""
            record[f"has_gee_{tp.name}"] = 0
            record[tp.std_col] = str(expected_output_512_path(out_root, plume_id, tp))
        record["has_required_legacy3"] = 1
        record["has_required_gee"] = 1
        record["has_required_cdse6"] = 0
        records.append(record)

    source = pd.DataFrame(records).sort_values(["event_time", "plume_id"], kind="stable").reset_index(drop=True)
    legacy_rows = source["cohort"].eq(LEGACY_COHORT)
    new_rows = source["cohort"].eq(NEW_COHORT)
    cdse_paths = [
        path
        for tp in TIMEPOINTS
        for path in source.loc[new_rows, f"cdse_{tp.name}_512_path"].astype(str)
    ]
    source_path_flags = parallel_file_flags(
        [
            *source["raw_cm_mask_path"].astype(str),
            *source.loc[legacy_rows, "legacy_t0_512_path"].astype(str),
            *source.loc[legacy_rows, "legacy_seasonal_512_path"].astype(str),
            *source.loc[legacy_rows, "legacy_year_512_path"].astype(str),
            *source.loc[legacy_rows, "legacy_mask_path"].astype(str),
            *cdse_paths,
        ],
        int(args.manifest_stat_workers),
    )
    source["has_raw_cm_mask"] = source["raw_cm_mask_path"].map(lambda path: source_path_flags.get(clean(path), 0))
    source.loc[legacy_rows, "has_required_legacy3"] = source.loc[legacy_rows].apply(
        lambda row: int(
            all(
                source_path_flags.get(clean(row[column]), 0)
                for column in ("legacy_t0_512_path", "legacy_seasonal_512_path", "legacy_year_512_path")
            )
        ),
        axis=1,
    )
    for tp in TIMEPOINTS:
        source.loc[new_rows, f"has_cdse_{tp.name}"] = source.loc[new_rows, f"cdse_{tp.name}_512_path"].map(
            lambda path: source_path_flags.get(clean(path), 0)
        )
    source.loc[new_rows, "has_required_cdse6"] = source.loc[
        new_rows, [f"has_cdse_{tp.name}" for tp in TIMEPOINTS]
    ].min(axis=1)
    source["has_legacy_mask"] = source["legacy_mask_path"].map(lambda path: source_path_flags.get(clean(path), 0))
    source["has_usable_mask"] = np.where(
        legacy_rows,
        np.maximum(source["has_legacy_mask"], source["has_raw_cm_mask"]),
        source["has_raw_cm_mask"],
    ).astype(np.int8)
    source["has_required_images"] = np.where(
        legacy_rows,
        source[["has_required_legacy3", "has_required_gee"]].min(axis=1),
        source["has_required_cdse6"],
    ).astype(np.int8)
    source["ready_for_512"] = source[["has_required_images", "has_usable_mask"]].min(axis=1)
    atomic_csv(source, Path(args.source_csv))
    atomic_csv(source[new_rows].copy(), Path(args.new_source_audit))
    legacy_export = source[legacy_rows].copy()
    for timepoint in TIMEPOINTS:
        if timepoint.name in GEE_FILL_TIMEPOINTS:
            legacy_export[timepoint.image_time_col] = ""
    atomic_csv(legacy_export, Path(args.legacy_export_input))

    summary = {
        "rows": len(source),
        "unique_plume_ids": int(source["plume_id"].nunique()),
        "unique_event_groups": int(source["event_group_id"].nunique()),
        "cohorts": source["cohort"].value_counts().to_dict(),
        "cross_cohort_event_groups": int(
            len(
                set(
                    source.loc[legacy_rows, "event_group_id"].astype(str)
                )
                & set(source.loc[new_rows, "event_group_id"].astype(str))
            )
        ),
        "ready_for_512": int(source["ready_for_512"].sum()),
        "missing_raw_cm_mask": int((source["has_raw_cm_mask"] == 0).sum()),
        "missing_usable_mask": int((source["has_usable_mask"] == 0).sum()),
        "legacy_missing_existing3": int(
            (legacy_rows & (source["has_required_legacy3"] == 0)).sum()
        ),
        "new_cdse_missing_by_timepoint": {
            tp.name: int((new_rows & (source[f"has_cdse_{tp.name}"] == 0)).sum()) for tp in TIMEPOINTS
        },
        "legacy_missing_required_gee_by_timepoint": {
            tp.name: int(
                (
                    (source[f"has_gee_{tp.name}"] == 0)
                    & legacy_rows
                    & (tp.name in GEE_FILL_TIMEPOINTS)
                ).sum()
            )
            for tp in TIMEPOINTS
        },
    }
    atomic_json(summary, Path(args.source_csv).with_suffix(".audit.json"))
    log(json.dumps(summary, ensure_ascii=False))
    return 0


def to_bhw(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"unexpected image shape {array.shape}")
    if array.shape[0] in {1, 3, 4, 12, 13}:
        return array
    if array.shape[-1] in {1, 3, 4, 12, 13}:
        return np.transpose(array, (2, 0, 1))
    return array


def center_crop_or_pad(array: np.ndarray, size: int = WINDOW_SIZE) -> tuple[np.ndarray, str]:
    bands, height, width = array.shape
    if height == size and width == size:
        return array, "noop"
    if height < size or width < size:
        out = np.zeros((bands, size, size), dtype=array.dtype)
        y0 = max(0, (size - height) // 2)
        x0 = max(0, (size - width) // 2)
        src_y0 = max(0, (height - size) // 2)
        src_x0 = max(0, (width - size) // 2)
        copy_h = min(height, size)
        copy_w = min(width, size)
        out[:, y0 : y0 + copy_h, x0 : x0 + copy_w] = array[:, src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
        return out, f"pad_center:{height}x{width}"
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return array[:, y0 : y0 + size, x0 : x0 + size], f"crop_center:{height}x{width}"


def read_geotiff_bhw(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        if dataset.width >= WINDOW_SIZE and dataset.height >= WINDOW_SIZE:
            col0 = (dataset.width - WINDOW_SIZE) // 2
            row0 = (dataset.height - WINDOW_SIZE) // 2
            array = dataset.read(window=Window(col0, row0, WINDOW_SIZE, WINDOW_SIZE))
        else:
            array = dataset.read()
    array = to_bhw(array)
    if array.shape[0] == 1:
        try:
            tif_array = to_bhw(tifffile.imread(path))
            if tif_array.shape[0] in {12, 13}:
                array = tif_array
        except Exception:
            pass
    return array.astype(np.float32, copy=False)


def write_multiband_atomic(path: Path, array: np.ndarray, compression: str = "deflate") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".part.{os.getpid()}.{random.randrange(1 << 30)}")
    bands, height, width = array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": str(array.dtype),
        "transform": from_origin(0, 0, 1, 1),
        "tiled": True,
        "blockxsize": min(256, width),
        "blockysize": min(256, height),
        "BIGTIFF": "IF_SAFER",
    }
    if compression != "none":
        profile["compress"] = compression
        profile["predictor"] = 2 if np.issubdtype(array.dtype, np.floating) else 1
    try:
        with rasterio.open(tmp, "w", **profile) as dataset:
            dataset.write(array)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def standardize_gee_to_512(src: Path, dst: Path, overwrite: bool) -> str:
    if file_ok(dst) and not overwrite:
        return "exists"
    array = read_geotiff_bhw(src)
    if array.shape[0] != EXPECTED_BANDS:
        raise ValueError(f"expected {EXPECTED_BANDS} bands, got {array.shape} from {src}")
    array, note = center_crop_or_pad(array, WINDOW_SIZE)
    write_multiband_atomic(dst, array.astype(np.float32, copy=False))
    return note


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> str:
    if not file_ok(src):
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if not overwrite and file_ok(dst):
            return "exists"
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        tmp = dst.with_name(dst.name + f".part.{os.getpid()}")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return "copied"
    if mode == "hardlink":
        os.link(src, dst)
        return "hardlinked"
    os.symlink(src, dst)
    return "symlinked"


def paste_centered(canvas: np.ndarray, source: np.ndarray, start_x: int, start_y: int) -> None:
    src_h, src_w = source.shape
    dst_h, dst_w = canvas.shape
    dst_x0 = max(0, start_x)
    dst_y0 = max(0, start_y)
    dst_x1 = min(dst_w, start_x + src_w)
    dst_y1 = min(dst_h, start_y + src_h)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        raise ValueError("reprojected plume mask does not intersect 512 canvas")
    src_x0 = dst_x0 - start_x
    src_y0 = dst_y0 - start_y
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0 : src_y0 + (dst_y1 - dst_y0), src_x0 : src_x0 + (dst_x1 - dst_x0)]


def build_exact_legacy_mask(raw_mask: Path, out_path: Path, latitude: float, longitude: float, overwrite: bool) -> dict[str, Any]:
    if file_ok(out_path) and not overwrite:
        with rasterio.open(out_path) as dataset:
            mask = dataset.read(1)
        return {
            "mask_action": "exists",
            "mask_positive_pixels": int((mask > 0).sum()),
            "mask_center20_sum": int((mask[246:266, 246:266] > 0).sum()),
        }
    with rasterio.open(raw_mask) as source:
        if source.crs is None:
            raise ValueError(f"raw CM mask has no CRS: {raw_mask}")
        temporary_transform, _, _ = calculate_default_transform(
            source.crs,
            source.crs,
            source.width,
            source.height,
            resolution=(20, 20),
            left=source.bounds.left,
            bottom=source.bounds.bottom,
            right=source.bounds.right,
            top=source.bounds.top,
        )
        width = max(1, int((source.bounds.right - source.bounds.left) / 20))
        height = max(1, int((source.bounds.top - source.bounds.bottom) / 20))
        projected = np.zeros((source.count, height, width), dtype=np.uint8)
        for band_index in range(source.count):
            reproject(
                source=source.read(band_index + 1),
                destination=projected[band_index],
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=temporary_transform,
                dst_crs=source.crs,
                resampling=Resampling.bilinear,
            )
        binary = np.max(projected > 0, axis=0).astype(np.uint8)
        center_x, center_y = transform("EPSG:4326", source.crs, [float(longitude)], [float(latitude)])
        center_x = float(center_x[0])
        center_y = float(center_y[0])
        center_pixel_x = int((center_x - source.bounds.left) / 20)
        center_pixel_y = int((source.bounds.top - center_y) / 20)
        start_x = WINDOW_SIZE // 2 - center_pixel_x
        start_y = WINDOW_SIZE // 2 - center_pixel_y
        canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.uint8)
        paste_centered(canvas, binary, start_x, start_y)
        output_transform = Affine(
            20,
            0,
            center_x - (WINDOW_SIZE // 2) * 20,
            0,
            -20,
            center_y + (WINDOW_SIZE // 2) * 20,
        )
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            transform=output_transform,
            crs=source.crs,
            width=WINDOW_SIZE,
            height=WINDOW_SIZE,
            count=1,
            dtype="uint8",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".part.{os.getpid()}.{random.randrange(1 << 30)}")
    try:
        with rasterio.open(tmp, "w", **profile) as destination:
            destination.write(canvas, 1)
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {
        "mask_action": "rebuilt_exact_legacy_geometry",
        "mask_positive_pixels": int(canvas.sum()),
        "mask_center20_sum": int(canvas[246:266, 246:266].sum()),
    }


def validate_512_image(path: Path) -> tuple[bool, str]:
    if not file_ok(path):
        return False, "missing"
    try:
        with rasterio.open(path) as dataset:
            if (dataset.count, dataset.height, dataset.width) != (EXPECTED_BANDS, WINDOW_SIZE, WINDOW_SIZE):
                return False, f"shape={dataset.count}x{dataset.height}x{dataset.width}"
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def process_512_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    plume_id = clean(row.get("plume_id"))
    cohort = clean(row.get("cohort"))
    result: dict[str, Any] = {
        "plume_id": plume_id,
        "event_group_id": clean(row.get("event_group_id")),
        "cohort": cohort,
        "status": "fail",
        "reason": "",
    }
    errors: list[str] = []
    actions: list[str] = []
    for tp in TIMEPOINTS:
        dst = Path(clean(row.get(tp.std_col)))
        try:
            if cohort == LEGACY_COHORT and tp.name in LEGACY_EXISTING_TIMEPOINTS:
                src = Path(clean(row.get(f"legacy_{tp.name}_512_path")))
                action = link_or_copy(src, dst, args.legacy_link_mode, bool(args.overwrite))
            elif cohort == NEW_COHORT:
                src = Path(clean(row.get(f"cdse_{tp.name}_512_path")))
                action = link_or_copy(src, dst, args.legacy_link_mode, bool(args.overwrite))
            else:
                src = Path(clean(row.get(f"gee_{tp.name}_raw_path")))
                if not file_ok(src):
                    raise FileNotFoundError(src)
                action = standardize_gee_to_512(src, dst, bool(args.overwrite))
            ok, message = validate_512_image(dst)
            if not ok:
                raise ValueError(message)
            actions.append(f"{tp.name}:{action}")
            result[tp.std_col] = str(dst)
        except Exception as exc:
            result[tp.std_col] = ""
            errors.append(f"{tp.name}:{type(exc).__name__}:{exc}")

    mask_path = Path(args.out_512_root) / plume_id / "resized_512x512.tif"
    try:
        legacy_mask = Path(clean(row.get("legacy_mask_path")))
        if cohort == LEGACY_COHORT and file_ok(legacy_mask):
            action = link_or_copy(legacy_mask, mask_path, args.legacy_link_mode, bool(args.overwrite))
            with rasterio.open(mask_path) as dataset:
                mask = dataset.read(1)
            mask_info = {
                "mask_action": f"legacy_{action}",
                "mask_positive_pixels": int((mask > 0).sum()),
                "mask_center20_sum": int((mask[246:266, 246:266] > 0).sum()),
            }
        else:
            raw_mask = Path(clean(row.get("raw_cm_mask_path")))
            if not file_ok(raw_mask):
                raise FileNotFoundError(raw_mask)
            mask_info = build_exact_legacy_mask(
                raw_mask,
                mask_path,
                float(row["plume_latitude"]),
                float(row["plume_longitude"]),
                bool(args.overwrite),
            )
        result.update(mask_info)
        result["resized_512x512_path"] = str(mask_path)
        if int(result.get("mask_positive_pixels", 0)) <= 0:
            errors.append("mask:zero_positive_pixels")
    except Exception as exc:
        result["resized_512x512_path"] = ""
        errors.append(f"mask:{type(exc).__name__}:{exc}")

    result["actions"] = ";".join(actions)
    result["reason"] = ";".join(errors)
    result["status"] = "ok" if not errors else "fail"
    return result


def build_512(args: argparse.Namespace) -> int:
    source = pd.read_csv(args.source_csv, low_memory=False)
    if args.plume_id:
        source = source[source["plume_id"].astype(str).eq(str(args.plume_id))].copy()
    if args.limit:
        source = source.head(args.limit).copy()
    rows = source.to_dict("records")
    results: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_512_row, row, args): row["plume_id"] for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            plume_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"plume_id": plume_id, "status": "fail", "reason": f"worker:{type(exc).__name__}:{exc}"}
            results.append(result)
            if index % max(1, int(args.progress_every)) == 0 or index == len(futures):
                ok_count = sum(item.get("status") == "ok" for item in results)
                log(f"build512 {index}/{len(futures)} ok={ok_count} fail={len(results) - ok_count}")

    qa = pd.DataFrame(results).sort_values("plume_id", kind="stable")
    atomic_csv(qa, Path(args.qa_csv))
    successful = qa[qa["status"].eq("ok")].copy()
    complete = source.merge(
        successful[["plume_id", "resized_512x512_path", "mask_positive_pixels", "mask_center20_sum"]],
        on="plume_id",
        how="inner",
        suffixes=("", "_qa"),
    )
    complete["has_all6_512"] = 1
    atomic_csv(complete, Path(args.complete_csv))
    log(f"build512 complete={len(complete)}/{len(source)} qa={args.qa_csv} output={args.complete_csv}")
    return 0 if len(complete) == len(source) else 2


def choose_temporal_cutoff(df: pd.DataFrame, target_ratio: float, min_ratio: float, max_ratio: float) -> tuple[pd.Timestamp, set[str], dict[str, Any]]:
    work = df.copy()
    work["_event_time"] = work["event_time"].map(parse_time)
    if work["_event_time"].isna().any():
        examples = work.loc[work["_event_time"].isna(), ["plume_id", "event_time"]].head(10).to_dict("records")
        raise ValueError(f"invalid event_time values: {examples}")
    group_stats = work.groupby("event_group_id", as_index=False).agg(
        group_time=("_event_time", "min"),
        group_time_max=("_event_time", "max"),
        row_count=("plume_id", "size"),
    )
    if (group_stats["group_time_max"] - group_stats["group_time"] > pd.Timedelta(days=1)).any():
        raise ValueError("at least one event_group_id spans more than one day")
    group_stats["date"] = group_stats["group_time"].dt.floor("D")
    by_date = group_stats.groupby("date", as_index=False)["row_count"].sum().sort_values("date")
    by_date["cumulative"] = by_date["row_count"].cumsum()
    total = int(by_date["row_count"].sum())
    candidates: list[tuple[float, pd.Timestamp, float]] = []
    for row in by_date.itertuples(index=False):
        ratio = int(row.cumulative) / total
        if min_ratio <= ratio <= max_ratio:
            cutoff = pd.Timestamp(row.date) + pd.Timedelta(days=1)
            candidates.append((abs(ratio - target_ratio), cutoff, ratio))
    if not candidates:
        raise ValueError(f"no temporal cutoff yields ratio in [{min_ratio}, {max_ratio}]")
    _, cutoff, ratio = min(candidates, key=lambda item: (item[0], item[1]))
    train_groups = set(group_stats.loc[group_stats["group_time"] < cutoff, "event_group_id"].astype(str))
    audit = {
        "cutoff_utc": cutoff.isoformat(),
        "rule": "train event_group_time < cutoff; test event_group_time >= cutoff",
        "target_ratio": target_ratio,
        "actual_ratio": ratio,
        "total_rows": total,
        "total_event_groups": int(len(group_stats)),
    }
    return cutoff, train_groups, audit


def split_temporal(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.complete_csv, low_memory=False)
    cutoff, train_groups, audit = choose_temporal_cutoff(df, args.target_ratio, args.min_ratio, args.max_ratio)
    group_values = df["event_group_id"].astype(str)
    train = df[group_values.isin(train_groups)].copy()
    test = df[~group_values.isin(train_groups)].copy()
    train["split"] = "train"
    test["split"] = "test"
    event_overlap = set(train["event_group_id"].astype(str)) & set(test["event_group_id"].astype(str))
    plume_overlap = set(train["plume_id"].astype(str)) & set(test["plume_id"].astype(str))
    if event_overlap or plume_overlap:
        raise RuntimeError(f"split leakage event={len(event_overlap)} plume={len(plume_overlap)}")
    out_root = Path(args.split_root)
    cutoff_tag = cutoff.strftime("%Y-%m-%d")
    train_path = out_root / f"s2_legacy_exact_train_cutoff_{cutoff_tag}.csv"
    test_path = out_root / f"s2_legacy_exact_test_cutoff_{cutoff_tag}.csv"
    atomic_csv(train, train_path)
    atomic_csv(test, test_path)
    audit.update(
        {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_ratio": len(train) / len(df),
            "train_event_groups": int(train["event_group_id"].nunique()),
            "test_event_groups": int(test["event_group_id"].nunique()),
            "event_group_overlap": len(event_overlap),
            "plume_id_overlap": len(plume_overlap),
            "train_csv": str(train_path),
            "test_csv": str(test_path),
        }
    )
    atomic_json(audit, out_root / "split_audit.json")
    log(json.dumps(audit, ensure_ascii=False))
    return 0


def read_chw(path: Path) -> np.ndarray:
    array = to_bhw(tifffile.imread(path)).astype(np.float32, copy=False)
    if array.shape != (EXPECTED_BANDS, WINDOW_SIZE, WINDOW_SIZE):
        raise ValueError(f"expected 12x512x512, got {array.shape}: {path}")
    return array


def read_mask(path: Path) -> np.ndarray:
    array = tifffile.imread(path)
    if array.ndim == 3:
        array = array[0] if array.shape[0] == 1 else np.max(array, axis=0)
    if array.shape != (WINDOW_SIZE, WINDOW_SIZE):
        raise ValueError(f"expected 512x512 mask, got {array.shape}: {path}")
    return (array > 0).astype(np.uint8)


def legacy_positive_crop(rng: random.Random, patch_size: int = PATCH_SIZE, center_size: int = 20) -> tuple[int, int]:
    center_x = WINDOW_SIZE // 2
    center_y = WINDOW_SIZE // 2
    center_left = center_x - center_size // 2
    center_right = center_x + center_size // 2
    center_top = center_y - center_size // 2
    center_bottom = center_y + center_size // 2
    left_min = max(0, center_right - patch_size)
    left_max = min(center_left, WINDOW_SIZE - patch_size)
    top_min = max(0, center_bottom - patch_size)
    top_max = min(center_top, WINDOW_SIZE - patch_size)
    return rng.randint(left_min, left_max), rng.randint(top_min, top_max)


def legacy_center_contained(x: int, y: int, patch_size: int = PATCH_SIZE, center_size: int = 10) -> bool:
    center_x = WINDOW_SIZE // 2
    center_y = WINDOW_SIZE // 2
    center_x1 = center_x - center_size // 2
    center_y1 = center_y - center_size // 2
    center_x2 = center_x + center_size // 2
    center_y2 = center_y + center_size // 2
    return x <= center_x1 and y <= center_y1 and x + patch_size >= center_x2 and y + patch_size >= center_y2


def band_zero_too_much(array: np.ndarray, band_index: int, threshold: float) -> bool:
    band = array[band_index]
    return float(np.count_nonzero(band == 0)) / float(band.size) >= threshold


def tifffile_write_atomic(path: Path, array: np.ndarray, compression: str = "deflate") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".part.{os.getpid()}.{random.randrange(1 << 30)}")
    kwargs = {} if compression == "none" else {"compression": compression}
    try:
        tifffile.imwrite(tmp, array, **kwargs)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def crop_patch(array: np.ndarray, x: int, y: int, size: int = PATCH_SIZE) -> np.ndarray:
    return array[..., y : y + size, x : x + size]


def patch_record(row: dict[str, Any], split: str, kind: str, index: int, x: int, y: int, label: int, paths: dict[str, str], mask_path: str, mask_sum: int) -> dict[str, Any]:
    sample_id = f"{row['plume_id']}__{kind}_{index:02d}"
    record = {
        "sample_id": sample_id,
        "id": sample_id,
        "plume_id": row["plume_id"],
        "event_group_id": row["event_group_id"],
        "cohort": row["cohort"],
        "split": split,
        "label": int(label),
        "crop_kind": kind,
        "crop_index": int(index),
        "crop_x": int(x),
        "crop_y": int(y),
        "plume_mask_sum": int(mask_sum),
        "path_plume": mask_path,
        "plume_mask_path": mask_path,
        "mask_path": mask_path,
        "latitude": row["plume_latitude"],
        "longitude": row["plume_longitude"],
        "datetime": row["event_time"],
    }
    for tp in TIMEPOINTS:
        record[PATH_COLUMNS[tp.name]] = paths[tp.name]
        record[tp.image_time_col] = row[tp.image_time_col]
    record["image_path"] = record["path_t0"]
    record["s2_path"] = record["path_t0"]
    record["s2_pre_path"] = record["path_seasonal"]
    record["s2_pre_pre_path"] = record["path_year"]
    return record


def process_crop_row(row: dict[str, Any], split: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    plume_id = clean(row.get("plume_id"))
    try:
        images = {tp.name: read_chw(Path(clean(row[tp.std_col]))) for tp in TIMEPOINTS}
        mask = read_mask(Path(clean(row["resized_512x512_path"])))
    except Exception as exc:
        return [], f"read:{type(exc).__name__}:{exc}"
    rng = random.Random(stable_seed(int(args.seed), split, plume_id))
    coordinates: list[tuple[str, int, int, int, int]] = []
    for index in range(int(args.n_pos)):
        x, y = legacy_positive_crop(rng, int(args.patch_size), int(args.positive_center_size))
        coordinates.append(("positive", index, x, y, 1))
    for index in range(int(args.n_random)):
        x = rng.randint(0, WINDOW_SIZE - int(args.patch_size))
        y = rng.randint(0, WINDOW_SIZE - int(args.patch_size))
        label = int(legacy_center_contained(x, y, int(args.patch_size), int(args.label_center_size)))
        coordinates.append(("random", index, x, y, label))

    output_root = Path(args.out_32_root) / split / plume_id
    records: list[dict[str, Any]] = []
    quality_drops = 0
    for kind, index, x, y, label in coordinates:
        crops = {name: crop_patch(image, x, y, int(args.patch_size)) for name, image in images.items()}
        if any(crop.shape[-2:] != (int(args.patch_size), int(args.patch_size)) for crop in crops.values()):
            quality_drops += 1
            continue
        if any(band_zero_too_much(crop, int(args.band_index), float(args.zero_ratio_thresh)) for crop in crops.values()):
            quality_drops += 1
            continue
        mask_crop = crop_patch(mask, x, y, int(args.patch_size))
        if kind == "random" and label == 0:
            mask_crop = np.zeros_like(mask_crop)
        sample_dir = output_root / f"{kind}_{index:02d}"
        paths: dict[str, str] = {}
        try:
            for tp in TIMEPOINTS:
                path = sample_dir / f"{tp.name}.tif"
                if args.overwrite or not file_ok(path):
                    tifffile_write_atomic(path, crops[tp.name].astype(np.float32, copy=False), args.compression)
                paths[tp.name] = str(path)
            plume_path = sample_dir / "plume.tif"
            if args.overwrite or not file_ok(plume_path):
                tifffile_write_atomic(plume_path, mask_crop.astype(np.uint8, copy=False), args.compression)
            records.append(
                patch_record(
                    row,
                    split,
                    kind,
                    index,
                    x,
                    y,
                    label,
                    paths,
                    str(plume_path),
                    int(mask_crop.sum()),
                )
            )
        except Exception as exc:
            return [], f"write:{type(exc).__name__}:{exc}"
    return records, "" if not quality_drops else f"quality_drops={quality_drops}"


def crop_32(args: argparse.Namespace) -> int:
    split_inputs = [("train", Path(args.train_csv)), ("test", Path(args.test_csv))]
    all_outputs: list[pd.DataFrame] = []
    for split, input_path in split_inputs:
        df = pd.read_csv(input_path, low_memory=False)
        if args.limit:
            df = df.head(args.limit).copy()
        rows = df.to_dict("records")
        records: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {pool.submit(process_crop_row, row, split, args): row["plume_id"] for row in rows}
            for index, future in enumerate(as_completed(futures), start=1):
                plume_id = futures[future]
                try:
                    new_records, message = future.result()
                except Exception as exc:
                    new_records, message = [], f"worker:{type(exc).__name__}:{exc}"
                records.extend(new_records)
                if not new_records or message:
                    failures.append({"plume_id": plume_id, "message": message, "samples": str(len(new_records))})
                if index % max(1, int(args.progress_every)) == 0 or index == len(futures):
                    log(f"crop32 {split} {index}/{len(futures)} samples={len(records)} issues={len(failures)}")
        output = pd.DataFrame(records).sort_values(["event_group_id", "plume_id", "crop_kind", "crop_index"], kind="stable")
        output["id"] = np.arange(1, len(output) + 1, dtype=np.int64)
        out_csv = Path(args.out_32_root) / f"{split}_patches_32.csv"
        atomic_csv(output, out_csv)
        atomic_csv(pd.DataFrame(failures), Path(args.out_32_root) / f"{split}_crop_issues.csv")
        all_outputs.append(output)
        log(f"crop32 wrote {out_csv} rows={len(output)} labels={output['label'].value_counts().to_dict()}")
    combined = pd.concat(all_outputs, ignore_index=True)
    atomic_csv(combined, Path(args.out_32_root) / "all_patches_32.csv")
    return 0


def resize_bilinear_chw(image: np.ndarray, out_h: int = TARGET_SIZE, out_w: int = TARGET_SIZE) -> np.ndarray:
    image = to_bhw(image).astype(np.float32, copy=False)
    channels, height, width = image.shape
    y = np.linspace(0, height - 1, out_h)
    x = np.linspace(0, width - 1, out_w)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = (x - x0)[None, :]
    wy = (y - y0)[:, None]
    top_left = image[:, y0[:, None], x0[None, :]]
    top_right = image[:, y0[:, None], x1[None, :]]
    bottom_left = image[:, y1[:, None], x0[None, :]]
    bottom_right = image[:, y1[:, None], x1[None, :]]
    output = (
        top_left * (1 - wx) * (1 - wy)
        + top_right * wx * (1 - wy)
        + bottom_left * (1 - wx) * wy
        + bottom_right * wx * wy
    )
    return output.astype(np.float32, copy=False)


def map_resized_path(src: str, old_root: Path, new_root: Path) -> Path:
    path = Path(src)
    try:
        relative = path.relative_to(old_root)
    except ValueError as exc:
        raise ValueError(f"path is outside 32 root: {path}") from exc
    return new_root / relative


def resize_one(src: Path, dst: Path, overwrite: bool, compression: str) -> tuple[bool, str]:
    try:
        if file_ok(dst) and not overwrite:
            return True, "exists"
        image = tifffile.imread(src)
        output = resize_bilinear_chw(image, TARGET_SIZE, TARGET_SIZE)
        tifffile_write_atomic(dst, output, compression)
        return True, "written"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def resize_224(args: argparse.Namespace) -> int:
    old_root = Path(args.out_32_root)
    new_root = Path(args.out_224_root)
    for split in ["train", "test"]:
        input_csv = old_root / f"{split}_patches_32.csv"
        df = pd.read_csv(input_csv, low_memory=False)
        if args.limit:
            df = df.head(args.limit).copy()
        for column in PATH_COLUMNS.values():
            new_values = []
            for src in df[column].astype(str):
                dst = map_resized_path(src, old_root, new_root)
                new_values.append(str(dst))
            df[column] = new_values
        df["image_path"] = df["path_t0"]
        df["s2_path"] = df["path_t0"]
        df["s2_pre_path"] = df["path_seasonal"]
        df["s2_pre_pre_path"] = df["path_year"]
        failures: list[dict[str, str]] = []
        completed = 0
        total = len(df) * len(PATH_COLUMNS)
        batch_rows = max(1, int(args.batch_rows))
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            for start in range(0, len(df), batch_rows):
                stop = min(start + batch_rows, len(df))
                futures = {}
                for row_index in range(start, stop):
                    row = df.iloc[row_index]
                    for column in PATH_COLUMNS.values():
                        dst = Path(row[column])
                        src = map_resized_path(str(dst), new_root, old_root)
                        futures[pool.submit(resize_one, src, dst, bool(args.overwrite), args.compression)] = (src, dst)
                for future in as_completed(futures):
                    src, dst = futures[future]
                    ok, message = future.result()
                    completed += 1
                    if not ok:
                        failures.append({"src": str(src), "dst": str(dst), "message": message})
                    if completed % max(1, int(args.progress_every)) == 0 or completed == total:
                        log(f"resize224 {split} {completed}/{total} failures={len(failures)}")
        if failures:
            atomic_csv(pd.DataFrame(failures), new_root / f"{split}_resize_failures.csv")
            raise RuntimeError(f"resize224 {split} failed files={len(failures)}")
        out_csv = new_root / f"{split}_patches_224.csv"
        atomic_csv(df, out_csv)
        log(f"resize224 wrote {out_csv} rows={len(df)}")
    train = pd.read_csv(new_root / "train_patches_224.csv", low_memory=False)
    test = pd.read_csv(new_root / "test_patches_224.csv", low_memory=False)
    atomic_csv(pd.concat([train, test], ignore_index=True), new_root / "all_patches_224.csv")
    return 0


def audit_dataset(args: argparse.Namespace) -> int:
    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False)
    event_overlap = set(train["event_group_id"].astype(str)) & set(test["event_group_id"].astype(str))
    plume_overlap = set(train["plume_id"].astype(str)) & set(test["plume_id"].astype(str))
    required = [*PATH_COLUMNS.values(), "path_plume"]
    missing_columns = sorted(set(required) - set(train.columns) | (set(required) - set(test.columns)))
    if missing_columns:
        raise ValueError(f"missing columns: {missing_columns}")
    audit = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_labels": train["label"].value_counts().sort_index().to_dict(),
        "test_labels": test["label"].value_counts().sort_index().to_dict(),
        "train_event_groups": int(train["event_group_id"].nunique()),
        "test_event_groups": int(test["event_group_id"].nunique()),
        "event_group_overlap": len(event_overlap),
        "plume_id_overlap": len(plume_overlap),
        "train_zero_mask_positive_labels": int(((train["label"] == 1) & (train["plume_mask_sum"] == 0)).sum()) if "plume_mask_sum" in train else None,
        "test_zero_mask_positive_labels": int(((test["label"] == 1) & (test["plume_mask_sum"] == 0)).sum()) if "plume_mask_sum" in test else None,
    }
    if event_overlap or plume_overlap:
        raise RuntimeError(f"leakage detected: {audit}")
    train_labels = set(pd.to_numeric(train["label"], errors="coerce").dropna().astype(int))
    test_labels = set(pd.to_numeric(test["label"], errors="coerce").dropna().astype(int))
    if not train_labels.issubset({0, 1}) or not test_labels.issubset({0, 1}):
        raise RuntimeError(f"invalid labels detected: train={train_labels} test={test_labels}")
    if audit["train_zero_mask_positive_labels"] or audit["test_zero_mask_positive_labels"]:
        raise RuntimeError(f"positive labels with empty plume masks detected: {audit}")
    atomic_json(audit, Path(args.audit_json))
    log(json.dumps(audit, ensure_ascii=False))
    return 0


def add_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--legacy-csv", default=str(DEFAULT_LEGACY_CSV))
    parser.add_argument("--legacy-mask-csv", default=str(DEFAULT_LEGACY_MASK_CSV))
    parser.add_argument("--new-csv", default=str(DEFAULT_NEW_CSV))
    parser.add_argument("--main-csv", default=str(DEFAULT_MAIN_CSV))
    parser.add_argument("--gee-local-csv", default=str(DEFAULT_GEE_LOCAL_CSV))
    parser.add_argument("--extra-gee-local-csv", nargs="*", default=[])
    parser.add_argument("--gee-export-manifest", nargs="*", default=[str(path) for path in DEFAULT_GEE_EXPORT_MANIFESTS])
    parser.add_argument("--manifest-stat-workers", type=int, default=32)
    parser.add_argument("--legacy-512-root", default=str(DEFAULT_LEGACY_512_ROOT))
    parser.add_argument("--gee-raw-root", default=str(DEFAULT_GEE_RAW_ROOT))
    parser.add_argument("--cm-root", default=str(DEFAULT_CM_ROOT))
    parser.add_argument("--out-512-root", default=str(DEFAULT_OUT_512_ROOT))
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE_CSV))
    parser.add_argument("--new-source-audit", default=str(DEFAULT_NEW_SOURCE_AUDIT))
    parser.add_argument("--legacy-export-input", default=str(DEFAULT_LEGACY_EXPORT_INPUT))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("build-manifest")
    add_manifest_args(manifest)
    manifest.set_defaults(func=build_manifest)

    build = subparsers.add_parser("build-512")
    build.add_argument("--source-csv", default=str(DEFAULT_SOURCE_CSV))
    build.add_argument("--out-512-root", default=str(DEFAULT_OUT_512_ROOT))
    build.add_argument("--qa-csv", default=str(DEFAULT_QA_512_CSV))
    build.add_argument("--complete-csv", default=str(DEFAULT_COMPLETE_512_CSV))
    build.add_argument("--workers", type=int, default=12)
    build.add_argument("--progress-every", type=int, default=100)
    build.add_argument("--legacy-link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--limit", type=int, default=0)
    build.add_argument("--plume-id", default="")
    build.set_defaults(func=build_512)

    split = subparsers.add_parser("split")
    split.add_argument("--complete-csv", default=str(DEFAULT_COMPLETE_512_CSV))
    split.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    split.add_argument("--target-ratio", type=float, default=0.85)
    split.add_argument("--min-ratio", type=float, default=0.80)
    split.add_argument("--max-ratio", type=float, default=0.90)
    split.set_defaults(func=split_temporal)

    crop = subparsers.add_parser("crop-32")
    crop.add_argument("--train-csv", required=True)
    crop.add_argument("--test-csv", required=True)
    crop.add_argument("--out-32-root", default=str(DEFAULT_OUT_32_ROOT))
    crop.add_argument("--workers", type=int, default=16)
    crop.add_argument("--progress-every", type=int, default=50)
    crop.add_argument("--seed", type=int, default=0)
    crop.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    crop.add_argument("--n-pos", type=int, default=16)
    crop.add_argument("--n-random", type=int, default=16)
    crop.add_argument("--positive-center-size", type=int, default=20)
    crop.add_argument("--label-center-size", type=int, default=10)
    crop.add_argument("--band-index", type=int, default=11)
    crop.add_argument("--zero-ratio-thresh", type=float, default=0.20)
    crop.add_argument("--compression", choices=["none", "deflate", "zlib", "lzma", "zstd"], default="deflate")
    crop.add_argument("--overwrite", action="store_true")
    crop.add_argument("--limit", type=int, default=0)
    crop.set_defaults(func=crop_32)

    resize = subparsers.add_parser("resize-224")
    resize.add_argument("--out-32-root", default=str(DEFAULT_OUT_32_ROOT))
    resize.add_argument("--out-224-root", default=str(DEFAULT_OUT_224_ROOT))
    resize.add_argument("--workers", type=int, default=12)
    resize.add_argument("--batch-rows", type=int, default=256)
    resize.add_argument("--progress-every", type=int, default=500)
    resize.add_argument("--compression", choices=["none", "deflate", "zlib", "lzma", "zstd"], default="deflate")
    resize.add_argument("--overwrite", action="store_true")
    resize.add_argument("--limit", type=int, default=0)
    resize.set_defaults(func=resize_224)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--train-csv", required=True)
    audit.add_argument("--test-csv", required=True)
    audit.add_argument("--audit-json", required=True)
    audit.set_defaults(func=audit_dataset)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
