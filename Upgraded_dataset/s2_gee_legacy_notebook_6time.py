#!/usr/bin/env python3
"""Pure-GEE six-time adaptation of the legacy S2 notebook from cell 4 onward."""

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
from rasterio import windows
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


WINDOW_SIZE = 512
PATCH_SIZE = 32
TARGET_SIZE = 224
EXPECTED_BANDS = 12
GEE_ROOT_TOKEN = "S2_GEE_6time"
GEE_BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

REPO_ROOT = Path(__file__).resolve().parents[1]
METHANE_ROOT = Path("/home/yuyao/methane_train")
DEFAULT_INPUT_CSV = (
    METHANE_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_gee_local_paths.csv"
)
DEFAULT_METADATA_ROOT = REPO_ROOT / "Upgraded_dataset" / "s2_gee_legacy_notebook_6time"
DEFAULT_SELECTED_CSV = DEFAULT_METADATA_ROOT / "cell04_gee_all6_physical_4449.csv"
DEFAULT_SELECTED_QA = DEFAULT_METADATA_ROOT / "cell04_gee_all6_physical_qa.csv"
DEFAULT_512_ROOT = Path("/diniuvol/yuyao/s2_gee_legacy_notebook_6time_std512")
DEFAULT_512_CSV = DEFAULT_METADATA_ROOT / "cell05_gee_6time_std512.csv"
DEFAULT_512_QA = DEFAULT_METADATA_ROOT / "cell05_gee_6time_std512_qa.csv"
DEFAULT_COMPLETE_CSV = DEFAULT_METADATA_ROOT / "cell06_gee_6time_std512_complete.csv"
DEFAULT_COMPLETE_QA = DEFAULT_METADATA_ROOT / "cell06_gee_6time_std512_complete_qa.csv"
DEFAULT_32_ROOT = Path("/diniuvol/yuyao/s2_gee_legacy_notebook_6time_32")
DEFAULT_32_CSV = DEFAULT_METADATA_ROOT / "cell08_gee_6time_patches_32.csv"
DEFAULT_32_QA = DEFAULT_METADATA_ROOT / "cell08_gee_6time_crop_qa.csv"
DEFAULT_224_ROOT = Path("/diniuvol/yuyao/s2_gee_legacy_notebook_6time_224")
DEFAULT_224_CSV = DEFAULT_METADATA_ROOT / "cell09_gee_6time_patches_224.csv"
DEFAULT_RESIZE_QA = DEFAULT_METADATA_ROOT / "cell09_gee_6time_resize_qa.csv"
DEFAULT_SPLIT_ROOT = DEFAULT_METADATA_ROOT / "temporal_cutoff_split"


@dataclass(frozen=True)
class Timepoint:
    name: str
    raw_col: str
    std_col: str
    std_filename: str
    patch_col: str
    patch_filename: str
    image_time_col: str


TIMEPOINTS = [
    Timepoint("t0", "gee_t0_raw_path", "s2_0_std_512", "s2_0_std_512.tif", "path_t0", "s2_0.tif", "t0_image_time"),
    Timepoint("prev1", "gee_prev1_raw_path", "s2_-7_std_512", "s2_-7_std_512.tif", "path_prev1", "s2_prev1.tif", "prev1_image_time"),
    Timepoint("prev2", "gee_prev2_raw_path", "s2_prev2_std_512", "s2_prev2_std_512.tif", "path_prev2", "s2_prev2.tif", "prev2_image_time"),
    Timepoint("prev3", "gee_prev3_raw_path", "s2_prev3_std_512", "s2_prev3_std_512.tif", "path_prev3", "s2_prev3.tif", "prev3_image_time"),
    Timepoint("seasonal", "gee_seasonal_raw_path", "s2_-90_std_512", "s2_-90_std_512.tif", "path_seasonal", "s2_seasonal.tif", "seasonal_image_time"),
    Timepoint("year", "gee_year_raw_path", "s2_-360_std_512", "s2_-360_std_512.tif", "path_year", "s2_year.tif", "year_image_time"),
]
PATCH_COLUMNS = [timepoint.patch_col for timepoint in TIMEPOINTS]


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def file_ok(value: Any) -> bool:
    path_text = clean(value)
    if not path_text:
        return False
    try:
        path = Path(path_text)
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    temporary = path.with_name(path.name + f".part.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(data: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    temporary = path.with_name(path.name + f".part.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join([str(seed), *parts]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def fallback_event_group_id(plume_id: str) -> str:
    pieces = plume_id.rsplit("-", 1)
    if len(pieces) == 2 and pieces[1] and len(pieces[1]) <= 4:
        return pieces[0]
    return plume_id


def normalize_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "event_time" not in result.columns:
        result["event_time"] = result.get("datetime", "")
    if "datetime" not in result.columns:
        result["datetime"] = result["event_time"]
    if "event_group_id" not in result.columns:
        result["event_group_id"] = result["plume_id"].astype(str).map(fallback_event_group_id)
    result["event_group_id"] = result["event_group_id"].fillna("").astype(str)
    missing_group = result["event_group_id"].str.strip().eq("")
    result.loc[missing_group, "event_group_id"] = (
        result.loc[missing_group, "plume_id"].astype(str).map(fallback_event_group_id)
    )
    return result


def source_row_status(row: dict[str, Any]) -> dict[str, Any]:
    plume_id = clean(row.get("plume_id"))
    result: dict[str, Any] = {
        "plume_id": plume_id,
        "event_group_id": clean(row.get("event_group_id")),
        "status": "ok",
        "reason": "",
    }
    reasons: list[str] = []
    for timepoint in TIMEPOINTS:
        source = clean(row.get(timepoint.raw_col))
        result[f"physical_{timepoint.name}"] = int(file_ok(source))
        result[f"gee_only_{timepoint.name}"] = int(GEE_ROOT_TOKEN in source)
        if not file_ok(source):
            reasons.append(f"{timepoint.name}:missing")
        elif GEE_ROOT_TOKEN not in source:
            reasons.append(f"{timepoint.name}:not_gee")
    mask_path = clean(row.get("raw_cm_mask_path"))
    result["physical_raw_cm_mask"] = int(file_ok(mask_path))
    if not file_ok(mask_path):
        reasons.append("raw_cm_mask:missing")
    if not plume_id:
        reasons.append("plume_id:missing")
    result["reason"] = ";".join(reasons)
    result["status"] = "ok" if not reasons else "fail"
    return result


def cell04_select(args: argparse.Namespace) -> int:
    source = normalize_metadata(pd.read_csv(args.input_csv, low_memory=False))
    rows = source.to_dict("records")
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        results = list(pool.map(source_row_status, rows))
    qa = pd.DataFrame(results)
    keep_ids = set(qa.loc[qa["status"].eq("ok"), "plume_id"].astype(str))
    selected = source[source["plume_id"].astype(str).isin(keep_ids)].copy()
    selected = selected.sort_values(["event_time", "event_group_id", "plume_id"], kind="stable")
    selected["data_source"] = "gee_only_COPERNICUS_S2_SR_HARMONIZED"
    selected["gee_band_order"] = ",".join(GEE_BAND_NAMES)
    atomic_csv(qa.sort_values("plume_id", kind="stable"), Path(args.qa_csv))
    atomic_csv(selected, Path(args.output_csv))
    summary = {
        "input_rows": int(len(source)),
        "physical_all6_gee_rows": int(len(selected)),
        "failed_rows": int(len(source) - len(selected)),
        "unique_plumes": int(selected["plume_id"].nunique()),
        "unique_events": int(selected["event_group_id"].nunique()),
        "gee_band_order": GEE_BAND_NAMES,
        "non_gee_rows_kept": 0,
    }
    atomic_json(summary, Path(args.output_csv).with_suffix(".summary.json"))
    log(f"cell04 selected={len(selected)}/{len(source)} unique_events={summary['unique_events']}")
    if args.expect_rows and len(selected) != int(args.expect_rows):
        raise RuntimeError(f"expected {args.expect_rows} physical all-six rows, found {len(selected)}")
    return 0


def centered_window(dataset: rasterio.io.DatasetReader) -> windows.Window:
    col0 = (int(dataset.width) - WINDOW_SIZE) // 2
    row0 = (int(dataset.height) - WINDOW_SIZE) // 2
    return windows.Window(col0, row0, WINDOW_SIZE, WINDOW_SIZE)


def read_gee_center_512(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as dataset:
        if dataset.count != EXPECTED_BANDS:
            raise ValueError(f"expected 12 bands, got {dataset.count}: {path}")
        window = centered_window(dataset)
        array = dataset.read(window=window, boundless=True, fill_value=0)
        profile = dataset.profile.copy()
        profile.update(
            height=WINDOW_SIZE,
            width=WINDOW_SIZE,
            count=EXPECTED_BANDS,
            transform=windows.transform(window, dataset.transform),
            crs=dataset.crs,
        )
    if array.shape != (EXPECTED_BANDS, WINDOW_SIZE, WINDOW_SIZE):
        raise ValueError(f"unexpected centered shape {array.shape}: {path}")
    return array.astype(np.float32, copy=False), profile


def write_gdal_multiband_tif(path: Path, array: np.ndarray, compression: str = "deflate") -> None:
    ensure_parent(path)
    temporary = path.with_name(f".{path.name}.part.{os.getpid()}.{random.randrange(1 << 30)}.tif")
    bands, height, width = array.shape
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": int(height),
        "width": int(width),
        "count": int(bands),
        "dtype": str(array.dtype),
        "transform": from_origin(0, 0, 1, 1),
        "tiled": True,
        "blockxsize": min(256, int(width)),
        "blockysize": min(256, int(height)),
        "BIGTIFF": "IF_SAFER",
    }
    if compression != "none":
        profile["compress"] = compression
        profile["predictor"] = 2 if np.issubdtype(array.dtype, np.floating) else 1
    try:
        with rasterio.open(temporary, "w", **profile) as destination:
            destination.write(array)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_mask_tif(path: Path, mask: np.ndarray, profile: dict[str, Any]) -> None:
    ensure_parent(path)
    temporary = path.with_name(f".{path.name}.part.{os.getpid()}.{random.randrange(1 << 30)}.tif")
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        height=WINDOW_SIZE,
        width=WINDOW_SIZE,
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        predictor=1,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    try:
        with rasterio.open(temporary, "w", **output_profile) as destination:
            destination.write(mask.astype(np.uint8, copy=False), 1)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_aligned_mask(raw_mask_path: Path, t0_profile: dict[str, Any]) -> np.ndarray:
    with rasterio.open(raw_mask_path) as source:
        if source.crs is None:
            raise ValueError(f"raw CM mask has no CRS: {raw_mask_path}")
        source_array = source.read(1).astype(np.float32)
        destination = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
        reproject(
            source=source_array,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=t0_profile["transform"],
            dst_crs=t0_profile["crs"],
            resampling=Resampling.bilinear,
        )
    return ((destination > 0) & np.isfinite(destination)).astype(np.uint8)


def validate_tiff(path: Path, shape: tuple[int, ...]) -> tuple[bool, str]:
    if not file_ok(path):
        return False, "missing"
    try:
        array = tifffile.imread(path)
        actual_shape = tuple(array.shape)
        accepted_shapes = {tuple(shape)}
        if len(shape) == 3:
            accepted_shapes.add((shape[1], shape[2], shape[0]))
        if actual_shape not in accepted_shapes:
            return False, f"shape={array.shape}"
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def process_cell05_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    plume_id = clean(row.get("plume_id"))
    output_dir = Path(args.out_512_root) / plume_id
    result: dict[str, Any] = {
        "plume_id": plume_id,
        "status": "ok",
        "reason": "",
        "mask_positive_pixels": 0,
        "mask_center20_sum": 0,
    }
    errors: list[str] = []
    t0_profile: dict[str, Any] | None = None
    for timepoint in TIMEPOINTS:
        source_path = Path(clean(row.get(timepoint.raw_col)))
        output_path = output_dir / timepoint.std_filename
        result[timepoint.std_col] = str(output_path)
        try:
            if GEE_ROOT_TOKEN not in str(source_path):
                raise ValueError(f"non-GEE source rejected: {source_path}")
            if not file_ok(source_path):
                raise FileNotFoundError(source_path)
            if not args.overwrite:
                valid, _ = validate_tiff(
                    output_path,
                    (EXPECTED_BANDS, WINDOW_SIZE, WINDOW_SIZE),
                )
                if valid:
                    if timepoint.name == "t0":
                        _, t0_profile = read_gee_center_512(source_path)
                    continue
            array, source_profile = read_gee_center_512(source_path)
            if timepoint.name == "t0":
                t0_profile = source_profile
            write_gdal_multiband_tif(output_path, array, args.compression)
        except Exception as exc:
            errors.append(f"{timepoint.name}:{type(exc).__name__}:{exc}")

    mask_path = output_dir / "resized_512x512.tif"
    result["resized_512x512_path"] = str(mask_path)
    try:
        if t0_profile is None:
            _, t0_profile = read_gee_center_512(Path(clean(row.get("gee_t0_raw_path"))))
        if args.overwrite or not validate_tiff(mask_path, (WINDOW_SIZE, WINDOW_SIZE))[0]:
            mask = build_aligned_mask(Path(clean(row.get("raw_cm_mask_path"))), t0_profile)
            write_mask_tif(mask_path, mask, t0_profile)
        else:
            mask = (tifffile.imread(mask_path) > 0).astype(np.uint8)
        result["mask_positive_pixels"] = int(mask.sum())
        result["mask_center20_sum"] = int(mask[246:266, 246:266].sum())
        if int(mask.sum()) <= 0:
            errors.append("mask:zero_positive_pixels")
    except Exception as exc:
        errors.append(f"mask:{type(exc).__name__}:{exc}")

    result["reason"] = ";".join(errors)
    result["status"] = "ok" if not errors else "fail"
    return result


def cell05_std512(args: argparse.Namespace) -> int:
    source = normalize_metadata(pd.read_csv(args.input_csv, low_memory=False))
    if args.plume_id:
        source = source[source["plume_id"].astype(str).eq(str(args.plume_id))].copy()
    if args.limit:
        source = source.head(int(args.limit)).copy()
    rows = source.to_dict("records")
    results: list[dict[str, Any]] = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(process_cell05_row, row, args): clean(row.get("plume_id")) for row in rows}
        for completed, future in enumerate(as_completed(futures), start=1):
            plume_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "plume_id": plume_id,
                    "status": "fail",
                    "reason": f"worker:{type(exc).__name__}:{exc}",
                }
            results.append(result)
            if completed % max(1, int(args.progress_every)) == 0 or completed == len(futures):
                ok_count = sum(item.get("status") == "ok" for item in results)
                elapsed = max(time.time() - start, 1e-6)
                rate = completed / elapsed
                eta = (len(futures) - completed) / max(rate, 1e-9)
                log(
                    f"cell05 {completed}/{len(futures)} ok={ok_count} fail={completed-ok_count} "
                    f"rate={rate:.2f} plumes/s eta={eta/60:.1f} min"
                )
                if args.flush_every and completed % int(args.flush_every) == 0:
                    atomic_csv(pd.DataFrame(results), Path(args.qa_csv))
    qa = pd.DataFrame(results).sort_values("plume_id", kind="stable")
    atomic_csv(qa, Path(args.qa_csv))
    successful = qa[qa["status"].eq("ok")].copy()
    output = source.merge(
        successful[
            [
                "plume_id",
                *[timepoint.std_col for timepoint in TIMEPOINTS],
                "resized_512x512_path",
                "mask_positive_pixels",
                "mask_center20_sum",
            ]
        ],
        on="plume_id",
        how="inner",
        suffixes=("", "_new"),
    )
    for column in [timepoint.std_col for timepoint in TIMEPOINTS] + ["resized_512x512_path"]:
        replacement = f"{column}_new"
        if replacement in output.columns:
            output[column] = output[replacement]
            output.drop(columns=[replacement], inplace=True)
    output["std512_source"] = "legacy_notebook_cell05_center_window_GEE_only"
    atomic_csv(output, Path(args.output_csv))
    log(f"cell05 wrote rows={len(output)} failures={len(source)-len(output)} to {args.output_csv}")
    return 0


def validate_cell06_row(row: dict[str, Any]) -> dict[str, Any]:
    plume_id = clean(row.get("plume_id"))
    reasons: list[str] = []
    for timepoint in TIMEPOINTS:
        valid, message = validate_tiff(
            Path(clean(row.get(timepoint.std_col))),
            (EXPECTED_BANDS, WINDOW_SIZE, WINDOW_SIZE),
        )
        if not valid:
            reasons.append(f"{timepoint.name}:{message}")
    valid_mask, mask_message = validate_tiff(
        Path(clean(row.get("resized_512x512_path"))),
        (WINDOW_SIZE, WINDOW_SIZE),
    )
    if not valid_mask:
        reasons.append(f"mask:{mask_message}")
    return {
        "plume_id": plume_id,
        "status": "ok" if not reasons else "fail",
        "reason": ";".join(reasons),
    }


def cell06_complete(args: argparse.Namespace) -> int:
    source = normalize_metadata(pd.read_csv(args.input_csv, low_memory=False))
    rows = source.to_dict("records")
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        results = list(pool.map(validate_cell06_row, rows))
    qa = pd.DataFrame(results)
    keep_ids = set(qa.loc[qa["status"].eq("ok"), "plume_id"].astype(str))
    complete = source[source["plume_id"].astype(str).isin(keep_ids)].copy()
    atomic_csv(qa.sort_values("plume_id", kind="stable"), Path(args.qa_csv))
    atomic_csv(complete, Path(args.output_csv))
    log(f"cell06 complete={len(complete)}/{len(source)}")
    return 0


def read_chw_512(path: Path) -> np.ndarray:
    array = tifffile.imread(path)
    if array.ndim == 3 and array.shape[-1] == EXPECTED_BANDS and array.shape[0] != EXPECTED_BANDS:
        array = np.transpose(array, (2, 0, 1))
    if array.shape != (EXPECTED_BANDS, WINDOW_SIZE, WINDOW_SIZE):
        raise ValueError(f"expected 12x512x512, got {array.shape}: {path}")
    return array.astype(np.float32, copy=False)


def read_mask_512(path: Path) -> np.ndarray:
    array = tifffile.imread(path)
    if array.ndim == 3:
        array = array[0] if array.shape[0] == 1 else np.max(array, axis=0)
    if array.shape != (WINDOW_SIZE, WINDOW_SIZE):
        raise ValueError(f"expected 512x512 mask, got {array.shape}: {path}")
    return (array > 0).astype(np.uint8)


def legacy_positive_crop(
    rng: random.Random,
    patch_size: int = PATCH_SIZE,
    center_size: int = 20,
) -> tuple[int, int]:
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


def legacy_center_contained(
    x: int,
    y: int,
    patch_size: int = PATCH_SIZE,
    center_size: int = 10,
) -> bool:
    center_x = WINDOW_SIZE // 2
    center_y = WINDOW_SIZE // 2
    center_x1 = center_x - center_size // 2
    center_y1 = center_y - center_size // 2
    center_x2 = center_x + center_size // 2
    center_y2 = center_y + center_size // 2
    return (
        x <= center_x1
        and y <= center_y1
        and x + patch_size >= center_x2
        and y + patch_size >= center_y2
    )


def band_zero_too_much(array: np.ndarray, band_index: int, threshold: float) -> bool:
    band = array[band_index]
    return float(np.count_nonzero(band == 0)) / float(band.size) >= threshold


def write_tifffile_atomic(path: Path, array: np.ndarray, compression: str) -> None:
    ensure_parent(path)
    temporary = path.with_name(f".{path.name}.part.{os.getpid()}.{random.randrange(1 << 30)}.tif")
    kwargs: dict[str, Any] = {}
    if compression != "none":
        kwargs["compression"] = compression
    try:
        tifffile.imwrite(temporary, array, **kwargs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def crop_patch(array: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    return array[..., y : y + size, x : x + size]


def crop_record(
    row: dict[str, Any],
    kind: str,
    index: int,
    x: int,
    y: int,
    label: int,
    paths: dict[str, str],
    mask_path: str,
    mask_sum: int,
) -> dict[str, Any]:
    sample_id = f"{row['plume_id']}__{kind}_{index:02d}_x{x}_y{y}"
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "id": sample_id,
        "plume_id": row["plume_id"],
        "event_group_id": row["event_group_id"],
        "data_source": "gee_only_COPERNICUS_S2_SR_HARMONIZED",
        "label": int(label),
        "crop_kind": kind,
        "crop_index": int(index),
        "crop_x": int(x),
        "crop_y": int(y),
        "plume_mask_sum": int(mask_sum),
        "path_plume": mask_path,
        "plume_mask_path": mask_path,
        "mask_path": mask_path,
        "latitude": row.get("plume_latitude", row.get("latitude", "")),
        "longitude": row.get("plume_longitude", row.get("longitude", "")),
        "event_time": row.get("event_time", row.get("datetime", "")),
        "datetime": row.get("event_time", row.get("datetime", "")),
        "gee_band_order": ",".join(GEE_BAND_NAMES),
    }
    for timepoint in TIMEPOINTS:
        record[timepoint.patch_col] = paths[timepoint.name]
        record[timepoint.image_time_col] = row.get(timepoint.image_time_col, "")
    record["image_path"] = record["path_t0"]
    record["s2_path"] = record["path_t0"]
    record["s2_pre_path"] = record["path_seasonal"]
    record["s2_pre_pre_path"] = record["path_year"]
    return record


def process_cell08_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plume_id = clean(row.get("plume_id"))
    qa: dict[str, Any] = {
        "plume_id": plume_id,
        "status": "ok",
        "reason": "",
        "requested": int(args.n_pos) + int(args.n_random),
        "written": 0,
        "quality_drops": 0,
    }
    try:
        images = {
            timepoint.name: read_chw_512(Path(clean(row[timepoint.std_col])))
            for timepoint in TIMEPOINTS
        }
        mask = read_mask_512(Path(clean(row["resized_512x512_path"])))
    except Exception as exc:
        qa["status"] = "fail"
        qa["reason"] = f"read:{type(exc).__name__}:{exc}"
        return [], qa

    rng = random.Random(stable_seed(int(args.seed), plume_id))
    coordinates: list[tuple[str, int, int, int, int]] = []
    for index in range(int(args.n_pos)):
        x, y = legacy_positive_crop(rng, int(args.patch_size), int(args.positive_center_size))
        coordinates.append(("positive", index, x, y, 1))
    for index in range(int(args.n_random)):
        x = rng.randint(0, WINDOW_SIZE - int(args.patch_size))
        y = rng.randint(0, WINDOW_SIZE - int(args.patch_size))
        label = int(
            legacy_center_contained(
                x,
                y,
                int(args.patch_size),
                int(args.label_center_size),
            )
        )
        coordinates.append(("random", index, x, y, label))

    records: list[dict[str, Any]] = []
    output_root = Path(args.out_32_root) / plume_id
    for kind, index, x, y, label in coordinates:
        crops = {
            name: crop_patch(image, x, y, int(args.patch_size))
            for name, image in images.items()
        }
        if any(
            crop.shape[-2:] != (int(args.patch_size), int(args.patch_size))
            for crop in crops.values()
        ):
            qa["quality_drops"] += 1
            continue
        if any(
            band_zero_too_much(
                crop,
                int(args.band_index),
                float(args.zero_ratio_thresh),
            )
            for crop in crops.values()
        ):
            qa["quality_drops"] += 1
            continue
        mask_crop = crop_patch(mask, x, y, int(args.patch_size))
        if kind == "random" and label == 0:
            mask_crop = np.zeros_like(mask_crop)
        sample_dir = output_root / f"{kind}_{index:02d}_x{x}_y{y}"
        paths: dict[str, str] = {}
        try:
            for timepoint in TIMEPOINTS:
                path = sample_dir / timepoint.patch_filename
                if args.overwrite or not file_ok(path):
                    write_tifffile_atomic(
                        path,
                        crops[timepoint.name].astype(np.float32, copy=False),
                        args.compression,
                    )
                paths[timepoint.name] = str(path)
            plume_path = sample_dir / "plume.tif"
            if args.overwrite or not file_ok(plume_path):
                write_tifffile_atomic(
                    plume_path,
                    mask_crop.astype(np.uint8, copy=False),
                    args.compression,
                )
            records.append(
                crop_record(
                    row,
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
            qa["status"] = "fail"
            qa["reason"] = f"write:{type(exc).__name__}:{exc}"
            return records, qa
    qa["written"] = len(records)
    if not records:
        qa["status"] = "fail"
        qa["reason"] = "all_crops_filtered"
    elif int(qa["quality_drops"]) > 0:
        qa["reason"] = f"quality_drops={qa['quality_drops']}"
    return records, qa


def cell08_crop32(args: argparse.Namespace) -> int:
    source = normalize_metadata(pd.read_csv(args.input_csv, low_memory=False))
    if args.limit:
        source = source.head(int(args.limit)).copy()
    rows = source.to_dict("records")
    all_records: list[dict[str, Any]] = []
    qa_records: list[dict[str, Any]] = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(process_cell08_row, row, args): clean(row.get("plume_id")) for row in rows}
        for completed, future in enumerate(as_completed(futures), start=1):
            plume_id = futures[future]
            try:
                records, qa = future.result()
            except Exception as exc:
                records = []
                qa = {
                    "plume_id": plume_id,
                    "status": "fail",
                    "reason": f"worker:{type(exc).__name__}:{exc}",
                    "written": 0,
                }
            all_records.extend(records)
            qa_records.append(qa)
            if completed % max(1, int(args.progress_every)) == 0 or completed == len(futures):
                elapsed = max(time.time() - start, 1e-6)
                rate = completed / elapsed
                eta = (len(futures) - completed) / max(rate, 1e-9)
                failures = sum(item.get("status") != "ok" for item in qa_records)
                log(
                    f"cell08 {completed}/{len(futures)} patches={len(all_records)} "
                    f"issues={failures} rate={rate:.2f} plumes/s eta={eta/60:.1f} min"
                )
                if args.flush_every and completed % int(args.flush_every) == 0:
                    atomic_csv(pd.DataFrame(qa_records), Path(args.qa_csv))
    output = pd.DataFrame(all_records)
    if output.empty:
        raise RuntimeError("cell08 produced no patches")
    output = output.sort_values(
        ["event_time", "event_group_id", "plume_id", "crop_kind", "crop_index"],
        kind="stable",
    ).reset_index(drop=True)
    output["id"] = np.arange(1, len(output) + 1, dtype=np.int64)
    qa = pd.DataFrame(qa_records).sort_values("plume_id", kind="stable")
    atomic_csv(qa, Path(args.qa_csv))
    atomic_csv(output, Path(args.output_csv))
    log(
        f"cell08 wrote rows={len(output)} plumes={output['plume_id'].nunique()} "
        f"labels={output['label'].value_counts().sort_index().to_dict()}"
    )
    return 0


def map_root(path_text: str, source_root: Path, destination_root: Path) -> Path:
    path = Path(path_text)
    try:
        relative = path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"path outside source root: {path}") from exc
    return destination_root / relative


def resize_write_one(
    destination: Path,
    array: np.ndarray,
    compression: str,
    overwrite: bool,
) -> tuple[bool, str]:
    try:
        if file_ok(destination) and not overwrite:
            return True, "exists"
        write_tifffile_atomic(destination, array.astype(np.float32, copy=False), compression)
        return True, "written"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def cell09_resize224(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as torch_functional

    source_root = Path(args.in_32_root)
    destination_root = Path(args.out_224_root)
    frame = pd.read_csv(args.input_csv, low_memory=False)
    if args.limit:
        frame = frame.head(int(args.limit)).copy()
    tasks: list[tuple[Path, Path]] = []
    for column in PATCH_COLUMNS:
        resized_paths: list[str] = []
        for source_text in frame[column].astype(str):
            source_path = Path(source_text)
            destination_path = map_root(source_text, source_root, destination_root)
            tasks.append((source_path, destination_path))
            resized_paths.append(str(destination_path))
        frame[column] = resized_paths
    frame["image_path"] = frame["path_t0"]
    frame["s2_path"] = frame["path_t0"]
    frame["s2_pre_path"] = frame["path_seasonal"]
    frame["s2_pre_pre_path"] = frame["path_year"]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    batch_size = max(1, int(args.batch_images))
    failures: list[dict[str, str]] = []
    completed = 0
    written = 0
    skipped = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(args.write_workers))) as writer_pool:
        for batch_start in range(0, len(tasks), batch_size):
            batch_tasks = tasks[batch_start : batch_start + batch_size]
            pending_tasks: list[tuple[Path, Path]] = []
            pending_arrays: list[np.ndarray] = []
            for source_path, destination_path in batch_tasks:
                if file_ok(destination_path) and not args.overwrite:
                    skipped += 1
                    completed += 1
                    continue
                try:
                    array = tifffile.imread(source_path)
                    if array.ndim == 3 and array.shape[-1] == EXPECTED_BANDS and array.shape[0] != EXPECTED_BANDS:
                        array = np.transpose(array, (2, 0, 1))
                    if array.shape != (EXPECTED_BANDS, PATCH_SIZE, PATCH_SIZE):
                        raise ValueError(f"shape={array.shape}")
                    pending_arrays.append(array.astype(np.float32, copy=False))
                    pending_tasks.append((source_path, destination_path))
                except Exception as exc:
                    failures.append(
                        {
                            "src": str(source_path),
                            "dst": str(destination_path),
                            "message": f"read:{type(exc).__name__}:{exc}",
                        }
                    )
                    completed += 1
            if pending_arrays:
                tensor = torch.from_numpy(np.stack(pending_arrays, axis=0)).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=False,
                )
                with torch.inference_mode():
                    resized = torch_functional.interpolate(
                        tensor,
                        size=(TARGET_SIZE, TARGET_SIZE),
                        mode="bilinear",
                        align_corners=True,
                    )
                output_arrays = resized.cpu().numpy()
                del tensor, resized
                futures = {
                    writer_pool.submit(
                        resize_write_one,
                        destination_path,
                        output_arrays[index],
                        args.compression,
                        bool(args.overwrite),
                    ): (source_path, destination_path)
                    for index, (source_path, destination_path) in enumerate(pending_tasks)
                }
                for future in as_completed(futures):
                    source_path, destination_path = futures[future]
                    ok, message = future.result()
                    completed += 1
                    if ok:
                        written += int(message == "written")
                        skipped += int(message == "exists")
                    else:
                        failures.append(
                            {
                                "src": str(source_path),
                                "dst": str(destination_path),
                                "message": message,
                            }
                        )
                del output_arrays
            if completed % max(1, int(args.progress_every)) < len(batch_tasks) or completed == len(tasks):
                elapsed = max(time.time() - start, 1e-6)
                rate = completed / elapsed
                eta = (len(tasks) - completed) / max(rate, 1e-9)
                log(
                    f"cell09 {completed}/{len(tasks)} written={written} skipped={skipped} "
                    f"fail={len(failures)} rate={rate:.1f} images/s eta={eta/60:.1f} min"
                )
    atomic_csv(pd.DataFrame(failures), Path(args.qa_csv))
    if failures:
        raise RuntimeError(f"cell09 failed images={len(failures)}; see {args.qa_csv}")
    atomic_csv(frame, Path(args.output_csv))
    log(f"cell09 wrote rows={len(frame)} images={len(tasks)} to {args.output_csv}")
    return 0


def choose_cutoff(
    frame: pd.DataFrame,
    target_test_ratio: float,
    min_test_ratio: float,
    max_test_ratio: float,
) -> tuple[pd.Timestamp, set[str], dict[str, Any]]:
    work = normalize_metadata(frame)
    work["_event_datetime"] = pd.to_datetime(work["event_time"], utc=True, errors="coerce")
    if work["_event_datetime"].isna().any():
        bad = work.loc[work["_event_datetime"].isna(), ["plume_id", "event_time"]].head(10)
        raise ValueError(f"invalid event times: {bad.to_dict('records')}")
    group_summary = (
        work.groupby("event_group_id", as_index=False)
        .agg(
            group_time=("_event_datetime", "max"),
            rows=("event_group_id", "size"),
            plumes=("plume_id", "nunique"),
        )
        .sort_values(["group_time", "event_group_id"], kind="stable")
        .reset_index(drop=True)
    )
    total_rows = int(group_summary["rows"].sum())
    candidates: list[tuple[float, pd.Timestamp, float, set[str]]] = []
    unique_times = group_summary["group_time"].drop_duplicates().sort_values()
    for cutoff in unique_times.iloc[:-1]:
        test_groups = set(
            group_summary.loc[group_summary["group_time"] > cutoff, "event_group_id"].astype(str)
        )
        test_rows = int(
            group_summary.loc[group_summary["event_group_id"].astype(str).isin(test_groups), "rows"].sum()
        )
        test_ratio = test_rows / max(total_rows, 1)
        if min_test_ratio <= test_ratio <= max_test_ratio:
            candidates.append(
                (
                    abs(test_ratio - target_test_ratio),
                    pd.Timestamp(cutoff),
                    test_ratio,
                    test_groups,
                )
            )
    if not candidates:
        raise RuntimeError(
            f"no temporal cutoff gives test ratio in [{min_test_ratio}, {max_test_ratio}]"
        )
    _, cutoff, test_ratio, test_groups = min(candidates, key=lambda item: (item[0], item[1]))
    report = {
        "cutoff_utc": cutoff.isoformat(),
        "target_test_ratio": float(target_test_ratio),
        "actual_test_ratio": float(test_ratio),
        "total_rows": int(total_rows),
        "total_plumes": int(work["plume_id"].nunique()),
        "total_events": int(work["event_group_id"].nunique()),
    }
    return cutoff, test_groups, report


def temporal_split(args: argparse.Namespace) -> int:
    frame = normalize_metadata(pd.read_csv(args.input_csv, low_memory=False))
    cutoff, test_groups, report = choose_cutoff(
        frame,
        float(args.target_test_ratio),
        float(args.min_test_ratio),
        float(args.max_test_ratio),
    )
    is_test = frame["event_group_id"].astype(str).isin(test_groups)
    train = frame.loc[~is_test].copy()
    test = frame.loc[is_test].copy()
    overlap_events = set(train["event_group_id"].astype(str)) & set(test["event_group_id"].astype(str))
    overlap_plumes = set(train["plume_id"].astype(str)) & set(test["plume_id"].astype(str))
    if overlap_events or overlap_plumes:
        raise RuntimeError(
            f"split leakage events={len(overlap_events)} plumes={len(overlap_plumes)}"
        )
    split_root = Path(args.split_root) / f"cutoff_{cutoff.date().isoformat()}"
    train_path = split_root / "train.csv"
    test_path = split_root / "test.csv"
    atomic_csv(train, train_path)
    atomic_csv(test, test_path)
    report.update(
        {
            "train_csv": str(train_path),
            "test_csv": str(test_path),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_plumes": int(train["plume_id"].nunique()),
            "test_plumes": int(test["plume_id"].nunique()),
            "train_events": int(train["event_group_id"].nunique()),
            "test_events": int(test["event_group_id"].nunique()),
            "train_label_counts": {
                str(key): int(value)
                for key, value in train["label"].value_counts().sort_index().items()
            },
            "test_label_counts": {
                str(key): int(value)
                for key, value in test["label"].value_counts().sort_index().items()
            },
            "event_overlap": 0,
            "plume_overlap": 0,
            "train_max_event_time": str(pd.to_datetime(train["event_time"], utc=True).max()),
            "test_min_event_time": str(pd.to_datetime(test["event_time"], utc=True).min()),
        }
    )
    atomic_json(report, split_root / "split_report.json")
    log(
        f"split cutoff={report['cutoff_utc']} train={len(train)} test={len(test)} "
        f"test_ratio={len(test)/len(frame):.3%} event_overlap=0 plume_overlap=0"
    )
    return 0


def audit(args: argparse.Namespace) -> int:
    train = normalize_metadata(pd.read_csv(args.train_csv, low_memory=False))
    test = normalize_metadata(pd.read_csv(args.test_csv, low_memory=False))
    event_overlap = sorted(
        set(train["event_group_id"].astype(str)) & set(test["event_group_id"].astype(str))
    )
    plume_overlap = sorted(set(train["plume_id"].astype(str)) & set(test["plume_id"].astype(str)))
    path_columns = [column for column in PATCH_COLUMNS if column in train.columns and column in test.columns]
    missing_paths: list[dict[str, str]] = []
    sample = pd.concat([train, test], ignore_index=True)
    if args.path_sample and int(args.path_sample) < len(sample):
        sample = sample.sample(int(args.path_sample), random_state=73)
    for row in sample.to_dict("records"):
        for column in path_columns:
            if not file_ok(row.get(column)):
                missing_paths.append(
                    {
                        "sample_id": clean(row.get("sample_id", row.get("id"))),
                        "column": column,
                        "path": clean(row.get(column)),
                    }
                )
    report = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_plumes": int(train["plume_id"].nunique()),
        "test_plumes": int(test["plume_id"].nunique()),
        "train_events": int(train["event_group_id"].nunique()),
        "test_events": int(test["event_group_id"].nunique()),
        "event_overlap_count": int(len(event_overlap)),
        "plume_overlap_count": int(len(plume_overlap)),
        "event_overlap_examples": event_overlap[:20],
        "plume_overlap_examples": plume_overlap[:20],
        "checked_path_rows": int(len(sample)),
        "missing_path_count": int(len(missing_paths)),
        "train_label_counts": {
            str(key): int(value)
            for key, value in train["label"].value_counts().sort_index().items()
        },
        "test_label_counts": {
            str(key): int(value)
            for key, value in test["label"].value_counts().sort_index().items()
        },
        "positive_zero_mask_rows": int(
            pd.concat([train, test], ignore_index=True)
            .query("label == 1 and plume_mask_sum <= 0")
            .shape[0]
        ),
        "negative_nonzero_mask_rows": int(
            pd.concat([train, test], ignore_index=True)
            .query("label == 0 and plume_mask_sum > 0")
            .shape[0]
        ),
        "gee_band_order": GEE_BAND_NAMES,
    }
    atomic_json(report, Path(args.output_json))
    if event_overlap or plume_overlap or missing_paths:
        raise RuntimeError(
            f"audit failed event_overlap={len(event_overlap)} plume_overlap={len(plume_overlap)} "
            f"missing_paths={len(missing_paths)}"
        )
    log(
        f"audit passed rows={len(train)+len(test)} events_overlap=0 plumes_overlap=0 "
        f"missing_paths=0"
    )
    return 0


def add_common_limit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=0)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cell04 = subparsers.add_parser("cell04-select-gee")
    cell04.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    cell04.add_argument("--output-csv", default=str(DEFAULT_SELECTED_CSV))
    cell04.add_argument("--qa-csv", default=str(DEFAULT_SELECTED_QA))
    cell04.add_argument("--workers", type=int, default=64)
    cell04.add_argument("--expect-rows", type=int, default=4449)
    cell04.set_defaults(func=cell04_select)

    cell05 = subparsers.add_parser("cell05-std512")
    cell05.add_argument("--input-csv", default=str(DEFAULT_SELECTED_CSV))
    cell05.add_argument("--out-512-root", default=str(DEFAULT_512_ROOT))
    cell05.add_argument("--output-csv", default=str(DEFAULT_512_CSV))
    cell05.add_argument("--qa-csv", default=str(DEFAULT_512_QA))
    cell05.add_argument("--workers", type=int, default=24)
    cell05.add_argument("--progress-every", type=int, default=50)
    cell05.add_argument("--flush-every", type=int, default=200)
    cell05.add_argument("--compression", choices=["none", "deflate", "zstd", "lzma"], default="deflate")
    cell05.add_argument("--overwrite", action="store_true")
    cell05.add_argument("--plume-id", default="")
    add_common_limit(cell05)
    cell05.set_defaults(func=cell05_std512)

    cell06 = subparsers.add_parser("cell06-complete")
    cell06.add_argument("--input-csv", default=str(DEFAULT_512_CSV))
    cell06.add_argument("--output-csv", default=str(DEFAULT_COMPLETE_CSV))
    cell06.add_argument("--qa-csv", default=str(DEFAULT_COMPLETE_QA))
    cell06.add_argument("--workers", type=int, default=32)
    cell06.set_defaults(func=cell06_complete)

    cell08 = subparsers.add_parser("cell08-crop32")
    cell08.add_argument("--input-csv", default=str(DEFAULT_COMPLETE_CSV))
    cell08.add_argument("--out-32-root", default=str(DEFAULT_32_ROOT))
    cell08.add_argument("--output-csv", default=str(DEFAULT_32_CSV))
    cell08.add_argument("--qa-csv", default=str(DEFAULT_32_QA))
    cell08.add_argument("--workers", type=int, default=12)
    cell08.add_argument("--progress-every", type=int, default=25)
    cell08.add_argument("--flush-every", type=int, default=200)
    cell08.add_argument("--seed", type=int, default=73)
    cell08.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    cell08.add_argument("--n-pos", type=int, default=16)
    cell08.add_argument("--n-random", type=int, default=16)
    cell08.add_argument("--positive-center-size", type=int, default=20)
    cell08.add_argument("--label-center-size", type=int, default=10)
    cell08.add_argument("--band-index", type=int, default=11)
    cell08.add_argument("--zero-ratio-thresh", type=float, default=0.20)
    cell08.add_argument("--compression", choices=["none", "deflate", "zstd", "lzma"], default="none")
    cell08.add_argument("--overwrite", action="store_true")
    add_common_limit(cell08)
    cell08.set_defaults(func=cell08_crop32)

    cell09 = subparsers.add_parser("cell09-resize224")
    cell09.add_argument("--input-csv", default=str(DEFAULT_32_CSV))
    cell09.add_argument("--in-32-root", default=str(DEFAULT_32_ROOT))
    cell09.add_argument("--out-224-root", default=str(DEFAULT_224_ROOT))
    cell09.add_argument("--output-csv", default=str(DEFAULT_224_CSV))
    cell09.add_argument("--qa-csv", default=str(DEFAULT_RESIZE_QA))
    cell09.add_argument("--device", default="cuda:0")
    cell09.add_argument("--batch-images", type=int, default=96)
    cell09.add_argument("--write-workers", type=int, default=24)
    cell09.add_argument("--progress-every", type=int, default=1000)
    cell09.add_argument("--compression", choices=["none", "deflate", "zstd", "lzma"], default="deflate")
    cell09.add_argument("--overwrite", action="store_true")
    add_common_limit(cell09)
    cell09.set_defaults(func=cell09_resize224)

    split = subparsers.add_parser("temporal-split")
    split.add_argument("--input-csv", default=str(DEFAULT_224_CSV))
    split.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    split.add_argument("--target-test-ratio", type=float, default=0.15)
    split.add_argument("--min-test-ratio", type=float, default=0.10)
    split.add_argument("--max-test-ratio", type=float, default=0.20)
    split.set_defaults(func=temporal_split)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--train-csv", required=True)
    audit_parser.add_argument("--test-csv", required=True)
    audit_parser.add_argument("--output-json", required=True)
    audit_parser.add_argument("--path-sample", type=int, default=5000)
    audit_parser.set_defaults(func=audit)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
