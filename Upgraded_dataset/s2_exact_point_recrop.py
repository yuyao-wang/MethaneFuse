#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
import os
import re
import resource
import shutil
import sys
import threading
import time
import types
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import requests
import tifffile
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.warp import reproject, transform as transform_coordinates


TIMEPOINTS = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
RAW_FILENAMES = {
    "t0": "s2.tif",
    "prev1": "s2_-7.tif",
    "prev2": "s2_prev2.tif",
    "prev3": "s2_prev3.tif",
    "seasonal": "s2_-90.tif",
    "year": "s2_-360.tif",
}
SUCCESS_SOURCE = "download_s2_missing_from_clean_table"
DEFAULT_EXISTING_PRODUCT_ROOTS = ",".join(
    [
        "/mnt/engg-niulab/yuyao/sensors_raw_data/S2/raw_data_dir_s2",
        "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_s2_90360",
        "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_s2",
        "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2",
        "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2_-7",
    ]
)
SPECTRAL_BAND_20M_RE = re.compile(r".*_B(?:0?[1-9]|1[0-2]|8A)_20m\.jp2$")
S2_PRODUCT_RE = re.compile(
    r"^S2(?P<satellite>[ABC])_MSIL2A_"
    r"(?P<sensing>\d{8}T\d{6})_"
    r"N(?P<baseline>\d{4})_"
    r"R(?P<orbit>\d{3})_"
    r"T(?P<tile>\d{2}[A-Z]{3})_"
    r"(?P<generation>\d{8}T\d{6})\.SAFE$"
)
AWS_R20M_BANDS = (
    ("B01", 0),
    ("B02", 1),
    ("B03", 2),
    ("B04", 3),
    ("B05", 4),
    ("B06", 5),
    ("B07", 6),
    ("B8A", 8),
    ("B11", 10),
    ("B12", 11),
)
STAC_COG_BANDS = AWS_R20M_BANDS
VALID_MODEL_BAND_INDICES = tuple(
    band_index for _, band_index in STAC_COG_BANDS
)
LEGACY_R20M_BANDS = (
    ("B01", 0),
    ("B02", 1),
    ("B03", 2),
    ("B04", 3),
    ("B05", 4),
    ("B06", 5),
    ("B07", 6),
    ("B8A", 7),
    ("B11", 10),
    ("B12", 11),
)
EARTH_SEARCH_ITEM_URL = (
    "https://earth-search.aws.element84.com/v1/collections/"
    "sentinel-2-l2a/items/{item_id}"
)
AWS_SENTINEL_L2A_HTTPS_ROOT = "https://sentinel-s2-l2a.s3.amazonaws.com"
_http_local = threading.local()
_cdse_stage_locks_guard = threading.Lock()
_cdse_stage_locks: dict[str, Any] = {}
_cdse_product_id_lock = threading.Lock()
_cdse_product_ids: dict[str, str] = {}
_cdse_catalogue_semaphore = threading.BoundedSemaphore(8)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def uncovered_pixel_count(image: np.ndarray) -> int:
    valid = image[list(VALID_MODEL_BAND_INDICES)]
    return int(np.count_nonzero(~np.any(valid > 0, axis=0)))


def center_is_covered(image: np.ndarray) -> bool:
    valid = image[list(VALID_MODEL_BAND_INDICES)]
    center_row = valid.shape[1] // 2
    center_col = valid.shape[2] // 2
    center = valid[
        :,
        center_row - 1 : center_row + 1,
        center_col - 1 : center_col + 1,
    ]
    return bool(np.any(center > 0))


def apply_dn_add(image: np.ndarray, dn_add: int) -> np.ndarray:
    if not dn_add:
        return image
    adjusted = image.astype(np.int32, copy=True)
    valid = adjusted > 0
    adjusted[valid] += int(dn_add)
    return np.clip(
        adjusted,
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)


def has_value(value: Any) -> bool:
    return bool(clean(value))


def existing_path(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    path = Path(text)
    try:
        if path.exists() and path.stat().st_size > 0:
            return str(path)
    except OSError:
        return ""
    return ""


def georef_sidecar_path(tif_path: Path) -> Path:
    return tif_path.with_name(tif_path.name + ".georef.json")


def target_is_complete(target: Path, timepoint: str) -> bool:
    try:
        if not target.is_file() or target.stat().st_size <= 0:
            return False
        if timepoint == "t0":
            sidecar = georef_sidecar_path(target)
            return sidecar.is_file() and sidecar.stat().st_size > 0
        return True
    except OSError:
        return False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_component(value: Any) -> str:
    text = clean(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:180] if text else "missing"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_bounds(row: pd.Series) -> list[float]:
    raw = clean(row.get("plume_bounds", ""))
    if raw:
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 4:
                return [float(v) for v in parsed]
        except Exception:
            pass
    lat = float(row["plume_latitude"])
    lon = float(row["plume_longitude"])
    return [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]


def point_centered_crop_bounds(row: pd.Series) -> list[float]:
    latitude = float(row["plume_latitude"])
    longitude = float(row["plume_longitude"])
    return [
        longitude - 0.01,
        latitude - 0.01,
        longitude + 0.01,
        latitude + 0.01,
    ]


def copy_file_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def move_file_cross_fs(src: Path, dst: Path) -> None:
    copy_file_atomic(src, dst)
    src.unlink()


def write_json_atomic(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + f".tmp.{os.getpid()}.{threading.get_ident()}"
    )
    temporary.write_text(json.dumps(data, sort_keys=True) + "\n")
    os.replace(temporary, path)


def cleanup_product(product_scratch_dir: Path, product_name: str) -> None:
    for path in [product_scratch_dir / product_name, product_scratch_dir / f"{product_name}.zip"]:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except Exception as exc:
            print(f"warning: failed to clean scratch product {path}: {exc}", flush=True)


def cleanup_crop_scratch(crop_scratch_dir: Path, product_id: str) -> None:
    path = crop_scratch_dir / safe_component(product_id)
    try:
        if path.is_dir():
            shutil.rmtree(path)
    except Exception as exc:
        print(f"warning: failed to clean crop scratch {path}: {exc}", flush=True)


def parse_existing_product_roots(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def product_band_files(product_dir: Path) -> list[Path]:
    if not product_dir.is_dir():
        return []
    return sorted(
        path
        for path in product_dir.iterdir()
        if path.is_file() and SPECTRAL_BAND_20M_RE.fullmatch(path.name)
    )


def find_existing_product(product_name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        candidate = root / product_name
        if product_band_files(candidate):
            return candidate
    return None


def product_acquisition_key(product_name: str) -> tuple[str, str, str, str] | None:
    match = S2_PRODUCT_RE.fullmatch(product_name)
    if match is None:
        return None
    return (
        match.group("satellite"),
        match.group("sensing"),
        match.group("orbit"),
        match.group("tile"),
    )


def build_existing_acquisition_index(
    roots: list[Path],
) -> dict[tuple[str, str, str, str], Path]:
    index: dict[tuple[str, str, str, str], Path] = {}
    for root in roots:
        try:
            entries = list(os.scandir(root))
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            key = product_acquisition_key(entry.name)
            if key is not None:
                index.setdefault(key, Path(entry.path))
    return index


def stage_existing_product(
    source_dir: Path,
    product_scratch_dir: Path,
    product_name: str,
) -> Path:
    destination = product_scratch_dir / product_name
    marker = destination / ".download_complete"
    if marker.is_file() and product_band_files(destination):
        return destination

    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    temporary = product_scratch_dir / (
        f".{safe_component(product_name)}.stage.{os.getpid()}.{threading.get_ident()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        source_files = product_band_files(source_dir)
        if not source_files:
            raise RuntimeError(f"no spectral 20 m bands found in {source_dir}")
        for source_file in source_files:
            shutil.copy2(source_file, temporary / source_file.name)
        (temporary / ".download_complete").write_text("staged\n")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def point_centered_window(
    legacy: Any,
    dataset: Any,
    plume_bounds: list[float],
) -> Window | None:
    top_left = legacy.latlon_to_pixel(plume_bounds[3], plume_bounds[0], dataset)
    bottom_right = legacy.latlon_to_pixel(plume_bounds[1], plume_bounds[2], dataset)
    center_x = (top_left[0] + bottom_right[0]) / 2
    center_y = (top_left[1] + bottom_right[1]) / 2

    window_size = 512
    half_window = window_size // 2
    col_start = int(np.floor(center_x - half_window))
    row_start = int(np.floor(center_y - half_window))
    col_end = col_start + window_size
    row_end = row_start + window_size
    if (
        col_start < 0
        or row_start < 0
        or col_end > dataset.width
        or row_end > dataset.height
    ):
        return None
    return Window(col_start, row_start, window_size, window_size)


def point_centered_unbounded_window(
    legacy: Any,
    dataset: Any,
    plume_bounds: list[float],
) -> Window:
    top_left = legacy.latlon_to_pixel(
        plume_bounds[3],
        plume_bounds[0],
        dataset,
    )
    bottom_right = legacy.latlon_to_pixel(
        plume_bounds[1],
        plume_bounds[2],
        dataset,
    )
    center_x = (top_left[0] + bottom_right[0]) / 2
    center_y = (top_left[1] + bottom_right[1]) / 2
    return Window(
        int(np.floor(center_x - 256)),
        int(np.floor(center_y - 256)),
        512,
        512,
    )


def http_session() -> requests.Session:
    session = getattr(_http_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "panopticon-s2-point-recrop/1.0"
        _http_local.session = session
    return session


def parse_s2_product_name(product_name: str) -> dict[str, str]:
    match = S2_PRODUCT_RE.fullmatch(product_name)
    if match is None:
        raise ValueError(f"unsupported Sentinel-2 product name: {product_name}")
    return match.groupdict()


def product_match_score(
    wanted: dict[str, str],
    candidate_name: str,
) -> tuple[int, int] | None:
    try:
        candidate = parse_s2_product_name(candidate_name)
    except ValueError:
        return None
    identity_fields = ("satellite", "sensing", "orbit", "tile")
    if any(candidate[field] != wanted[field] for field in identity_fields):
        return None
    baseline_penalty = 0 if candidate["baseline"] == wanted["baseline"] else 1
    wanted_generation = datetime.strptime(wanted["generation"], "%Y%m%dT%H%M%S")
    candidate_generation = datetime.strptime(
        candidate["generation"],
        "%Y%m%dT%H%M%S",
    )
    generation_delta = int(
        abs((candidate_generation - wanted_generation).total_seconds())
    )
    return baseline_penalty, generation_delta


def stac_item_base_url(item: dict[str, Any]) -> str:
    assets = item.get("assets", {})
    metadata_href = clean(
        assets.get("product_metadata", {}).get("href", "")
    )
    candidate_hrefs = [metadata_href]
    for asset_name in (
        "rededge1-jp2",
        "swir16-jp2",
        "blue-jp2",
        "green-jp2",
    ):
        candidate_hrefs.append(
            clean(assets.get(asset_name, {}).get("href", ""))
        )
    for href in candidate_hrefs:
        if not href.startswith("s3://sentinel-s2-l2a/"):
            continue
        key = href[len("s3://sentinel-s2-l2a/") :]
        if "/" not in key:
            continue
        return (
            f"{AWS_SENTINEL_L2A_HTTPS_ROOT}/"
            f"{key.rsplit('/', 1)[0]}"
        )
    raise RuntimeError("STAC item has no original Sentinel-2 JP2 asset")


def earth_search_r20m_source(
    product_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    wanted = parse_s2_product_name(product_name)
    date_tag = wanted["sensing"][:8]
    candidates: list[tuple[tuple[int, int], dict[str, Any]]] = []
    consecutive_missing = 0
    session = http_session()

    for item_index in range(max(1, int(args.aws_item_max_index))):
        item_id = (
            f"S2{wanted['satellite']}_{wanted['tile']}_"
            f"{date_tag}_{item_index}_L2A"
        )
        url = EARTH_SEARCH_ITEM_URL.format(item_id=item_id)
        response = None
        for attempt in range(1, int(args.aws_read_retries) + 2):
            try:
                response = session.get(
                    url,
                    timeout=float(args.aws_request_timeout),
                )
                if response.status_code == 404:
                    break
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt > int(args.aws_read_retries):
                    raise
                time.sleep(min(8.0, 2.0 ** (attempt - 1)))
        if response is None:
            continue
        if response.status_code == 404:
            consecutive_missing += 1
            if candidates and consecutive_missing >= 3:
                break
            continue
        consecutive_missing = 0
        item = response.json()
        candidate_name = clean(
            item.get("properties", {}).get("s2:product_uri", "")
        )
        score = product_match_score(wanted, candidate_name)
        if score is None:
            continue
        try:
            base_url = stac_item_base_url(item)
        except RuntimeError:
            continue
        exact = candidate_name == product_name
        record = {
            "item_id": item_id,
            "source_product_name": candidate_name,
            "base_url": base_url,
            "exact_product": exact,
            "score": score,
        }
        if exact:
            return record
        candidates.append((score, record))

    if not candidates:
        raise RuntimeError(
            f"no public AWS Sentinel-2 L2A item matches {product_name}"
        )
    return min(candidates, key=lambda item: item[0])[1]


def read_aws_r20m_band(
    legacy: Any,
    args: argparse.Namespace,
    pending: list[dict[str, Any]],
    base_url: str,
    band_name: str,
    band_index: int,
) -> tuple[int, str, list[tuple[tuple[int, str], np.ndarray, dict[str, Any]]]]:
    url = f"{base_url}/R20m/{band_name}.jp2"
    last_error: Exception | None = None
    for attempt in range(1, int(args.aws_read_retries) + 2):
        try:
            records: list[
                tuple[tuple[int, str], np.ndarray, dict[str, Any]]
            ] = []
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".jp2",
                GDAL_HTTP_MULTIRANGE="YES",
                GDAL_HTTP_MAX_RETRY=str(args.aws_read_retries),
                GDAL_HTTP_RETRY_DELAY="1",
                GDAL_HTTP_TIMEOUT=str(args.aws_request_timeout),
                VSI_CACHE="FALSE",
            ):
                with rasterio.open(url) as dataset:
                    for task in pending:
                        key = (
                            int(task["row_index"]),
                            task["timepoint"],
                        )
                        window = point_centered_window(
                            legacy,
                            dataset,
                            task["plume_bounds"],
                        )
                        if window is None:
                            raise RuntimeError(
                                "empty point-centered crop for "
                                f"{task['plume_id']} from {url}"
                            )
                        clipped = dataset.read(1, window=window)
                        crop_transform = rasterio.windows.transform(
                            window,
                            dataset.transform,
                        )
                        georef = {
                            "crs_wkt": (
                                dataset.crs.to_wkt()
                                if dataset.crs
                                else ""
                            ),
                            "transform": list(crop_transform),
                            "height": int(clipped.shape[0]),
                            "width": int(clipped.shape[1]),
                            "plume_id": task["plume_id"],
                            "timepoint": task["timepoint"],
                            "product_id": task["product_id"],
                            "product_name": task["product_name"],
                            "reference_band": url,
                        }
                        records.append((key, clipped, georef))
            return band_index, band_name, records
        except Exception as exc:
            last_error = exc
            if attempt > int(args.aws_read_retries):
                break
            time.sleep(min(8.0, 2.0 ** (attempt - 1)))
    raise RuntimeError(
        f"AWS R20m read failed after {int(args.aws_read_retries) + 1} "
        f"attempts for {url}: {last_error}"
    )


def crop_aws_product_group_to_window_cache(
    legacy: Any,
    args: argparse.Namespace,
    group: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ready: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in group:
        target = Path(task["target_raw_path"])
        if target_is_complete(target, task["timepoint"]) and not args.overwrite:
            ready.append(
                {
                    **task,
                    "status": "target_exists",
                    "raw_path": str(target),
                    "message": "",
                }
            )
        else:
            pending.append(task)
    if not pending:
        return ready

    direct_item_id = clean(group[0].get("stac_item_id"))
    direct_base_url = clean(group[0].get("stac_base_url"))
    direct_source_product = clean(
        group[0].get("stac_source_product_name")
    )
    if direct_item_id and direct_base_url and direct_source_product:
        source = {
            "item_id": direct_item_id,
            "source_product_name": direct_source_product,
            "base_url": direct_base_url,
            "exact_product": direct_source_product
            == group[0]["product_name"],
            "score": (0, 0),
        }
    else:
        source = earth_search_r20m_source(
            group[0]["product_name"],
            args,
        )
    if args.aws_require_exact and not source["exact_product"]:
        raise RuntimeError(
            "public AWS item is not the exact selected SAFE product: "
            f"wanted={group[0]['product_name']} "
            f"found={source['source_product_name']}"
        )
    valid_pending: list[dict[str, Any]] = []
    validation_url = f"{source['base_url']}/R20m/B05.jp2"
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".jp2",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MAX_RETRY=str(args.aws_read_retries),
        GDAL_HTTP_RETRY_DELAY="1",
        GDAL_HTTP_TIMEOUT=str(args.aws_request_timeout),
        VSI_CACHE="FALSE",
    ):
        with rasterio.open(validation_url) as dataset:
            for task in pending:
                window = point_centered_window(
                    legacy,
                    dataset,
                    task["plume_bounds"],
                )
                if window is None:
                    ready.append(
                        {
                            **task,
                            "status": "failed",
                            "raw_path": "",
                            "message": (
                                "AWS tile cannot provide centered 512 "
                                f"crop for {task['plume_id']}"
                            ),
                        }
                    )
                    continue
                valid_pending.append(task)
    pending = valid_pending
    if not pending:
        return ready
    arrays = {
        (int(task["row_index"]), task["timepoint"]): np.zeros(
            (12, 512, 512),
            dtype=np.uint16,
        )
        for task in pending
    }
    georefs: dict[tuple[int, str], dict[str, Any]] = {}
    max_band_workers = min(
        len(AWS_R20M_BANDS),
        max(1, int(args.aws_band_workers)),
    )
    with ThreadPoolExecutor(max_workers=max_band_workers) as executor:
        futures = {
            executor.submit(
                read_aws_r20m_band,
                legacy,
                args,
                pending,
                source["base_url"],
                band_name,
                band_index,
            )
            for band_name, band_index in AWS_R20M_BANDS
        }
        for future in as_completed(futures):
            band_index, _, records = future.result()
            for key, clipped, georef in records:
                if clipped.shape != (512, 512):
                    raise RuntimeError(
                        f"unexpected AWS crop shape for {key}: "
                        f"{clipped.shape}"
                    )
                arrays[key][band_index] = clipped
                georefs.setdefault(key, georef)
            futures.remove(future)

    source_kind = (
        "exact"
        if source["exact_product"]
        else "same_acquisition_fallback"
    )
    source_label = (
        f"aws_sentinel_s2_l2a:{source['item_id']}:{source_kind}:"
        f"{source['source_product_name']}"
    )
    for task in pending:
        key = (int(task["row_index"]), task["timepoint"])
        image = arrays[key]
        empty_valid_bands = [
            band_index
            for _, band_index in AWS_R20M_BANDS
            if not np.any(image[band_index])
        ]
        if empty_valid_bands:
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": (
                        "empty AWS valid bands: "
                        f"{empty_valid_bands}"
                    ),
                }
            )
            continue
        uncovered_pixels = uncovered_pixel_count(image)
        if not center_is_covered(image):
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": (
                        "AWS crop center is uncovered; "
                        f"total_uncovered_pixels={uncovered_pixels}"
                    ),
                }
            )
            continue
        product_key = safe_component(task["product_id"])
        scratch_tif = (
            Path(args.crop_scratch_dir)
            / product_key
            / task["plume_id"]
            / f"{task['timepoint']}.tif"
        )
        scratch_tif.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(scratch_tif, image)
        target = Path(task["target_raw_path"])
        move_file_cross_fs(scratch_tif, target)
        if task["timepoint"] == "t0":
            metadata = dict(georefs[key])
            metadata.update(
                source_item_id=source["item_id"],
                source_product_name=source["source_product_name"],
                source_product_exact=bool(source["exact_product"]),
            )
            write_json_atomic(metadata, georef_sidecar_path(target))
        ready.append(
            {
                **task,
                "status": "downloaded",
                "raw_path": str(target),
                "message": f"512x512; product_source={source_label}",
            }
        )
    return ready


def read_stac_cog_band(
    args: argparse.Namespace,
    pending: list[dict[str, Any]],
    cog_base_url: str,
    reference: dict[str, Any],
    windows: dict[tuple[int, str], Window],
    band_name: str,
    band_index: int,
    dn_add: int,
) -> tuple[int, list[tuple[tuple[int, str], np.ndarray]]]:
    url = f"{cog_base_url}/{band_name}.tif"
    last_error: Exception | None = None
    for attempt in range(1, int(args.aws_read_retries) + 2):
        try:
            records: list[tuple[tuple[int, str], np.ndarray]] = []
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MULTIRANGE="YES",
                GDAL_HTTP_MAX_RETRY=str(args.aws_read_retries),
                GDAL_HTTP_RETRY_DELAY="1",
                GDAL_HTTP_TIMEOUT=str(args.aws_request_timeout),
                VSI_CACHE="FALSE",
            ):
                with rasterio.open(url) as dataset:
                    source_resolution = max(
                        abs(float(dataset.transform.a)),
                        abs(float(dataset.transform.e)),
                    )
                    if source_resolution < 19.0:
                        resampling = Resampling.average
                    elif source_resolution > 21.0:
                        resampling = Resampling.bilinear
                    else:
                        resampling = Resampling.nearest
                    with WarpedVRT(
                        dataset,
                        crs=reference["crs"],
                        transform=reference["transform"],
                        width=reference["width"],
                        height=reference["height"],
                        src_nodata=0,
                        nodata=0,
                        resampling=resampling,
                    ) as warped:
                        for task in pending:
                            key = (
                                int(task["row_index"]),
                                task["timepoint"],
                            )
                            clipped = warped.read(
                                1,
                                window=windows[key],
                            )
                            if clipped.shape != (512, 512):
                                raise RuntimeError(
                                    f"unexpected COG crop shape for "
                                    f"{key}: {clipped.shape}"
                                )
                            clipped = apply_dn_add(clipped, dn_add)
                            records.append((key, clipped))
            return band_index, records
        except Exception as exc:
            last_error = exc
            if attempt > int(args.aws_read_retries):
                break
            time.sleep(min(8.0, 2.0 ** (attempt - 1)))
    raise RuntimeError(
        f"STAC COG read failed after {int(args.aws_read_retries) + 1} "
        f"attempts for {url}: {last_error}"
    )


def crop_stac_cog_product_group(
    legacy: Any,
    args: argparse.Namespace,
    group: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ready: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in group:
        target = Path(task["target_raw_path"])
        if target_is_complete(target, task["timepoint"]) and not args.overwrite:
            ready.append(
                {
                    **task,
                    "status": "target_exists",
                    "raw_path": str(target),
                    "message": "",
                }
            )
        else:
            pending.append(task)
    if not pending:
        return ready

    cog_base_url = clean(group[0].get("stac_cog_base_url"))
    if not cog_base_url:
        raise RuntimeError("missing STAC COG base URL")
    dn_add = int(float(group[0].get("stac_dn_add", 0) or 0))
    reference_url = f"{cog_base_url}/B05.tif"
    windows: dict[tuple[int, str], Window] = {}
    georefs: dict[tuple[int, str], dict[str, Any]] = {}
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MAX_RETRY=str(args.aws_read_retries),
        GDAL_HTTP_RETRY_DELAY="1",
        GDAL_HTTP_TIMEOUT=str(args.aws_request_timeout),
        VSI_CACHE="FALSE",
    ):
        with rasterio.open(reference_url) as dataset:
            reference = {
                "crs": dataset.crs,
                "transform": dataset.transform,
                "width": int(dataset.width),
                "height": int(dataset.height),
            }
            for task in pending:
                key = (
                    int(task["row_index"]),
                    task["timepoint"],
                )
                window = point_centered_window(
                    legacy,
                    dataset,
                    task["plume_bounds"],
                )
                if window is None:
                    ready.append(
                        {
                            **task,
                            "status": "failed",
                            "raw_path": "",
                            "message": (
                                "COG tile cannot provide centered 512 "
                                f"crop for {task['plume_id']}"
                            ),
                        }
                    )
                    continue
                windows[key] = window
                crop_transform = rasterio.windows.transform(
                    window,
                    dataset.transform,
                )
                georefs[key] = {
                    "crs_wkt": (
                        dataset.crs.to_wkt()
                        if dataset.crs
                        else ""
                    ),
                    "transform": list(crop_transform),
                    "height": 512,
                    "width": 512,
                    "plume_id": task["plume_id"],
                    "timepoint": task["timepoint"],
                    "product_id": task["product_id"],
                    "product_name": task["product_name"],
                    "reference_band": reference_url,
                }

    pending = [
        task
        for task in pending
        if (int(task["row_index"]), task["timepoint"]) in windows
    ]
    if not pending:
        return ready
    arrays = {
        (int(task["row_index"]), task["timepoint"]): np.zeros(
            (12, 512, 512),
            dtype=np.uint16,
        )
        for task in pending
    }
    max_band_workers = min(
        len(STAC_COG_BANDS),
        max(1, int(args.aws_band_workers)),
    )
    with ThreadPoolExecutor(max_workers=max_band_workers) as executor:
        futures = {
            executor.submit(
                read_stac_cog_band,
                args,
                pending,
                cog_base_url,
                reference,
                windows,
                band_name,
                band_index,
                dn_add,
            )
            for band_name, band_index in STAC_COG_BANDS
        }
        for future in as_completed(futures):
            band_index, records = future.result()
            for key, clipped in records:
                arrays[key][band_index] = clipped

    source_label = (
        "stac_cog_resampled:"
        f"{clean(group[0].get('stac_item_id'))}:"
        f"{clean(group[0].get('stac_source_product_name'))}"
    )
    for task in pending:
        key = (int(task["row_index"]), task["timepoint"])
        image = arrays[key]
        empty_valid_bands = [
            band_index
            for _, band_index in STAC_COG_BANDS
            if not np.any(image[band_index])
        ]
        if empty_valid_bands:
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": (
                        "empty COG valid bands: "
                        f"{empty_valid_bands}"
                    ),
                }
            )
            continue
        uncovered_pixels = uncovered_pixel_count(image)
        if not center_is_covered(image):
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": (
                        "COG crop center is uncovered; "
                        f"total_uncovered_pixels={uncovered_pixels}"
                    ),
                }
            )
            continue
        product_key = safe_component(task["product_id"])
        scratch_tif = (
            Path(args.crop_scratch_dir)
            / product_key
            / task["plume_id"]
            / f"{task['timepoint']}.tif"
        )
        scratch_tif.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(scratch_tif, image)
        target = Path(task["target_raw_path"])
        move_file_cross_fs(scratch_tif, target)
        if task["timepoint"] == "t0":
            metadata = dict(georefs[key])
            metadata.update(
                source_item_id=clean(task.get("stac_item_id")),
                source_product_name=clean(
                    task.get("stac_source_product_name")
                ),
                source_product_exact=True,
                source_kind="stac_cog_resampled",
                source_dn_add=dn_add,
            )
            write_json_atomic(
                metadata,
                georef_sidecar_path(target),
            )
        ready.append(
            {
                **task,
                "status": "downloaded",
                "raw_path": str(target),
                "message": (
                    f"512x512; product_source={source_label}; "
                    f"dn_add={dn_add}"
                ),
            }
        )
    return ready


def local_mosaic_band_path(
    product_dir: str,
    band_name: str,
) -> Path:
    matches = sorted(
        Path(product_dir).glob(f"*_{band_name}_20m.jp2")
    )
    if not matches:
        raise FileNotFoundError(
            f"missing {band_name} R20m in {product_dir}"
        )
    return matches[0]


def read_local_mosaic_band(
    sources: list[dict[str, Any]],
    reference_crs: Any,
    union_transform: Any,
    union_height: int,
    union_width: int,
    task_slices: dict[
        tuple[int, str],
        tuple[slice, slice],
    ],
    band_name: str,
    band_index: int,
) -> tuple[int, list[tuple[tuple[int, str], np.ndarray]]]:
    mosaic = np.zeros(
        (union_height, union_width),
        dtype=np.uint16,
    )
    for source in sources:
        path = local_mosaic_band_path(
            source["product_dir"],
            band_name,
        )
        temporary = np.zeros(
            (union_height, union_width),
            dtype=np.uint16,
        )
        with rasterio.open(path) as dataset:
            reproject(
                source=rasterio.band(dataset, 1),
                destination=temporary,
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                src_nodata=0,
                dst_transform=union_transform,
                dst_crs=reference_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
                num_threads=1,
                init_dest_nodata=True,
            )
        fill = (mosaic == 0) & (temporary > 0)
        mosaic[fill] = temporary[fill]
    records = [
        (key, mosaic[row_slice, col_slice])
        for key, (row_slice, col_slice) in task_slices.items()
    ]
    return band_index, records


def crop_local_mosaic_product_group(
    legacy: Any,
    args: argparse.Namespace,
    group: list[dict[str, Any]],
    source_label: str = "local_r20m_mosaic",
) -> list[dict[str, Any]]:
    ready: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in group:
        target = Path(task["target_raw_path"])
        if target_is_complete(target, task["timepoint"]) and not args.overwrite:
            ready.append(
                {
                    **task,
                    "status": "target_exists",
                    "raw_path": str(target),
                    "message": "",
                }
            )
        else:
            pending.append(task)
    if not pending:
        return ready

    source_map: dict[str, dict[str, Any]] = {}
    for task in pending:
        raw = clean(task.get("local_mosaic_json"))
        if not raw:
            raise RuntimeError("missing local mosaic source list")
        for source in json.loads(raw):
            product_name = clean(source.get("product_name"))
            if product_name:
                source_map[product_name] = source
    reference_product = pending[0]["product_name"]
    sources = sorted(
        source_map.values(),
        key=lambda source: clean(source.get("product_name"))
        != reference_product,
    )
    if not sources:
        raise RuntimeError("empty local mosaic source list")

    reference_path = local_mosaic_band_path(
        sources[0]["product_dir"],
        "B05",
    )
    windows: dict[tuple[int, str], Window] = {}
    georefs: dict[tuple[int, str], dict[str, Any]] = {}
    with rasterio.open(reference_path) as dataset:
        reference_crs = dataset.crs
        reference_transform = dataset.transform
        for task in pending:
            key = (
                int(task["row_index"]),
                task["timepoint"],
            )
            window = point_centered_unbounded_window(
                legacy,
                dataset,
                task["plume_bounds"],
            )
            windows[key] = window
            crop_transform = rasterio.windows.transform(
                window,
                dataset.transform,
            )
            georefs[key] = {
                "crs_wkt": (
                    dataset.crs.to_wkt()
                    if dataset.crs
                    else ""
                ),
                "transform": list(crop_transform),
                "height": 512,
                "width": 512,
                "plume_id": task["plume_id"],
                "timepoint": task["timepoint"],
                "product_id": task["product_id"],
                "product_name": task["product_name"],
                "reference_band": str(reference_path),
            }

    min_col = min(int(window.col_off) for window in windows.values())
    min_row = min(int(window.row_off) for window in windows.values())
    max_col = max(
        int(window.col_off + window.width)
        for window in windows.values()
    )
    max_row = max(
        int(window.row_off + window.height)
        for window in windows.values()
    )
    union_width = max_col - min_col
    union_height = max_row - min_row
    if union_width > 4096 or union_height > 4096:
        raise RuntimeError(
            f"local mosaic canvas too large: "
            f"{union_width}x{union_height}"
        )
    union_window = Window(
        min_col,
        min_row,
        union_width,
        union_height,
    )
    union_transform = rasterio.windows.transform(
        union_window,
        reference_transform,
    )
    task_slices = {
        key: (
            slice(
                int(window.row_off) - min_row,
                int(window.row_off) - min_row + 512,
            ),
            slice(
                int(window.col_off) - min_col,
                int(window.col_off) - min_col + 512,
            ),
        )
        for key, window in windows.items()
    }
    arrays = {
        (int(task["row_index"]), task["timepoint"]): np.zeros(
            (12, 512, 512),
            dtype=np.uint16,
        )
        for task in pending
    }
    with ThreadPoolExecutor(
        max_workers=min(
            len(STAC_COG_BANDS),
            max(1, int(args.aws_band_workers)),
        )
    ) as executor:
        futures = {
            executor.submit(
                read_local_mosaic_band,
                sources,
                reference_crs,
                union_transform,
                union_height,
                union_width,
                task_slices,
                band_name,
                band_index,
            )
            for band_name, band_index in STAC_COG_BANDS
        }
        for future in as_completed(futures):
            band_index, records = future.result()
            for key, clipped in records:
                if clipped.shape != (512, 512):
                    raise RuntimeError(
                        f"unexpected local mosaic crop shape: "
                        f"{clipped.shape}"
                    )
                arrays[key][band_index] = clipped

    source_names = ",".join(
        clean(source.get("product_name"))
        for source in sources
    )
    for task in pending:
        key = (int(task["row_index"]), task["timepoint"])
        image = arrays[key]
        empty_valid_bands = [
            band_index
            for _, band_index in STAC_COG_BANDS
            if not np.any(image[band_index])
        ]
        if empty_valid_bands:
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": (
                        "empty local mosaic valid bands: "
                        f"{empty_valid_bands}"
                    ),
                }
            )
            continue
        uncovered_pixels = uncovered_pixel_count(image)
        if not center_is_covered(image):
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": (
                        "local mosaic crop center is uncovered; "
                        f"total_uncovered_pixels={uncovered_pixels}"
                    ),
                }
            )
            continue
        product_key = safe_component(task["product_id"])
        scratch_tif = (
            Path(args.crop_scratch_dir)
            / product_key
            / task["plume_id"]
            / f"{task['timepoint']}.tif"
        )
        scratch_tif.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(scratch_tif, image)
        target = Path(task["target_raw_path"])
        move_file_cross_fs(scratch_tif, target)
        if task["timepoint"] == "t0":
            metadata = dict(georefs[key])
            metadata.update(
                source_product_name=task["product_name"],
                source_product_exact=True,
                source_kind=source_label,
                source_product_names=source_names.split(","),
            )
            write_json_atomic(
                metadata,
                georef_sidecar_path(target),
            )
        ready.append(
            {
                **task,
                "status": "downloaded",
                "raw_path": str(target),
                "message": (
                    f"512x512; product_source={source_label}:"
                    f"{source_names}"
                ),
            }
        )
    return ready


def read_stac_mosaic_band(
    args: argparse.Namespace,
    sources: list[dict[str, Any]],
    reference_crs: Any,
    union_transform: Any,
    union_height: int,
    union_width: int,
    task_slices: dict[
        tuple[int, str],
        tuple[slice, slice],
    ],
    band_name: str,
    band_index: int,
) -> tuple[int, list[tuple[tuple[int, str], np.ndarray]]]:
    mosaic = np.zeros(
        (union_height, union_width),
        dtype=np.uint16,
    )
    for source in sources:
        url = f"{source['stac_cog_base_url']}/{band_name}.tif"
        dn_add = int(float(source.get("stac_dn_add", 0) or 0))
        last_error: Exception | None = None
        for attempt in range(1, int(args.aws_read_retries) + 2):
            try:
                temporary = np.zeros(
                    (union_height, union_width),
                    dtype=np.float32,
                )
                with rasterio.Env(
                    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                    GDAL_HTTP_MULTIRANGE="YES",
                    GDAL_HTTP_MAX_RETRY=str(args.aws_read_retries),
                    GDAL_HTTP_RETRY_DELAY="1",
                    GDAL_HTTP_TIMEOUT=str(args.aws_request_timeout),
                    VSI_CACHE="FALSE",
                ):
                    with rasterio.open(url) as dataset:
                        source_resolution = max(
                            abs(float(dataset.transform.a)),
                            abs(float(dataset.transform.e)),
                        )
                        if source_resolution < 19.0:
                            resampling = Resampling.average
                        elif source_resolution > 21.0:
                            resampling = Resampling.bilinear
                        else:
                            resampling = Resampling.nearest
                        reproject(
                            source=rasterio.band(dataset, 1),
                            destination=temporary,
                            src_transform=dataset.transform,
                            src_crs=dataset.crs,
                            src_nodata=0,
                            dst_transform=union_transform,
                            dst_crs=reference_crs,
                            dst_nodata=0,
                            resampling=resampling,
                            num_threads=1,
                            init_dest_nodata=True,
                        )
                valid = temporary > 0
                if dn_add:
                    temporary[valid] += dn_add
                rounded = np.clip(
                    np.rint(temporary),
                    0,
                    np.iinfo(np.uint16).max,
                ).astype(np.uint16)
                fill = (mosaic == 0) & valid
                mosaic[fill] = rounded[fill]
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt > int(args.aws_read_retries):
                    break
                time.sleep(min(8.0, 2.0 ** (attempt - 1)))
        if last_error is not None:
            raise RuntimeError(
                f"STAC mosaic read failed for {url}: {last_error}"
            )
    records = [
        (key, mosaic[row_slice, col_slice])
        for key, (row_slice, col_slice) in task_slices.items()
    ]
    return band_index, records


def crop_stac_mosaic_product_group(
    legacy: Any,
    args: argparse.Namespace,
    group: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ready: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in group:
        target = Path(task["target_raw_path"])
        if target_is_complete(target, task["timepoint"]) and not args.overwrite:
            ready.append(
                {
                    **task,
                    "status": "target_exists",
                    "raw_path": str(target),
                    "message": "",
                }
            )
        else:
            pending.append(task)
    if not pending:
        return ready

    source_map: dict[str, dict[str, Any]] = {}
    for task in pending:
        raw = clean(task.get("stac_mosaic_json"))
        if not raw:
            raise RuntimeError("missing STAC mosaic source list")
        for source in json.loads(raw):
            item_id = clean(source.get("stac_item_id"))
            if item_id:
                source_map[item_id] = source
    reference_item_id = clean(pending[0].get("stac_item_id"))
    sources = sorted(
        source_map.values(),
        key=lambda source: clean(source.get("stac_item_id"))
        != reference_item_id,
    )
    if not sources:
        raise RuntimeError("empty STAC mosaic source list")

    reference_url = (
        f"{clean(pending[0].get('stac_cog_base_url'))}/B05.tif"
    )
    windows: dict[tuple[int, str], Window] = {}
    georefs: dict[tuple[int, str], dict[str, Any]] = {}
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MAX_RETRY=str(args.aws_read_retries),
        GDAL_HTTP_RETRY_DELAY="1",
        GDAL_HTTP_TIMEOUT=str(args.aws_request_timeout),
        VSI_CACHE="FALSE",
    ):
        with rasterio.open(reference_url) as dataset:
            reference_crs = dataset.crs
            reference_transform = dataset.transform
            for task in pending:
                key = (
                    int(task["row_index"]),
                    task["timepoint"],
                )
                window = point_centered_unbounded_window(
                    legacy,
                    dataset,
                    task["plume_bounds"],
                )
                windows[key] = window
                crop_transform = rasterio.windows.transform(
                    window,
                    dataset.transform,
                )
                georefs[key] = {
                    "crs_wkt": (
                        dataset.crs.to_wkt()
                        if dataset.crs
                        else ""
                    ),
                    "transform": list(crop_transform),
                    "height": 512,
                    "width": 512,
                    "plume_id": task["plume_id"],
                    "timepoint": task["timepoint"],
                    "product_id": task["product_id"],
                    "product_name": task["product_name"],
                    "reference_band": reference_url,
                }

    min_col = min(int(window.col_off) for window in windows.values())
    min_row = min(int(window.row_off) for window in windows.values())
    max_col = max(
        int(window.col_off + window.width)
        for window in windows.values()
    )
    max_row = max(
        int(window.row_off + window.height)
        for window in windows.values()
    )
    union_width = max_col - min_col
    union_height = max_row - min_row
    if union_width > 4096 or union_height > 4096:
        raise RuntimeError(
            f"mosaic canvas too large: {union_width}x{union_height}"
        )
    union_window = Window(
        min_col,
        min_row,
        union_width,
        union_height,
    )
    union_transform = rasterio.windows.transform(
        union_window,
        reference_transform,
    )
    task_slices = {
        key: (
            slice(
                int(window.row_off) - min_row,
                int(window.row_off) - min_row + 512,
            ),
            slice(
                int(window.col_off) - min_col,
                int(window.col_off) - min_col + 512,
            ),
        )
        for key, window in windows.items()
    }

    arrays = {
        (int(task["row_index"]), task["timepoint"]): np.zeros(
            (12, 512, 512),
            dtype=np.uint16,
        )
        for task in pending
    }
    max_band_workers = min(
        len(STAC_COG_BANDS),
        max(1, int(args.aws_band_workers)),
    )
    with ThreadPoolExecutor(max_workers=max_band_workers) as executor:
        futures = {
            executor.submit(
                read_stac_mosaic_band,
                args,
                sources,
                reference_crs,
                union_transform,
                union_height,
                union_width,
                task_slices,
                band_name,
                band_index,
            )
            for band_name, band_index in STAC_COG_BANDS
        }
        for future in as_completed(futures):
            band_index, records = future.result()
            for key, clipped in records:
                if clipped.shape != (512, 512):
                    raise RuntimeError(
                        f"unexpected mosaic crop shape: {clipped.shape}"
                    )
                arrays[key][band_index] = clipped

    source_ids = ",".join(
        clean(source.get("stac_item_id"))
        for source in sources
    )
    for task in pending:
        key = (int(task["row_index"]), task["timepoint"])
        image = arrays[key]
        empty_valid_bands = [
            band_index
            for _, band_index in STAC_COG_BANDS
            if not np.any(image[band_index])
        ]
        if empty_valid_bands:
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": (
                        "empty STAC mosaic valid bands: "
                        f"{empty_valid_bands}"
                    ),
                }
            )
            continue
        uncovered_pixels = uncovered_pixel_count(image)
        if not center_is_covered(image):
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": (
                        "STAC mosaic crop center is uncovered; "
                        f"total_uncovered_pixels={uncovered_pixels}"
                    ),
                }
            )
            continue
        product_key = safe_component(task["product_id"])
        scratch_tif = (
            Path(args.crop_scratch_dir)
            / product_key
            / task["plume_id"]
            / f"{task['timepoint']}.tif"
        )
        scratch_tif.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(scratch_tif, image)
        target = Path(task["target_raw_path"])
        move_file_cross_fs(scratch_tif, target)
        if task["timepoint"] == "t0":
            metadata = dict(georefs[key])
            metadata.update(
                source_item_id=reference_item_id,
                source_product_name=clean(
                    task.get("stac_source_product_name")
                ),
                source_product_exact=True,
                source_kind="stac_cog_mosaic",
                source_item_ids=source_ids.split(","),
            )
            write_json_atomic(
                metadata,
                georef_sidecar_path(target),
            )
        ready.append(
            {
                **task,
                "status": "downloaded",
                "raw_path": str(target),
                "message": (
                    "512x512; product_source=stac_cog_mosaic:"
                    f"{source_ids}"
                ),
            }
        )
    return ready


def cdse_nodes_base_url(product_id: str, product_name: str) -> str:
    return (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/"
        f"Products({product_id})/Nodes({product_name})"
    )


def valid_cdse_product_id(product_id: str) -> bool:
    try:
        uuid.UUID(clean(product_id))
        return True
    except (ValueError, AttributeError):
        return False


def resolve_cdse_product_id(
    args: argparse.Namespace,
    product_id: str,
    product_name: str,
) -> str:
    if valid_cdse_product_id(product_id):
        return product_id
    with _cdse_product_id_lock:
        cached = _cdse_product_ids.get(product_name)
    if cached:
        return cached

    try:
        wanted = parse_s2_product_name(product_name)
        filter_text = (
            f"contains(Name,'S2{wanted['satellite']}_MSIL2A_"
            f"{wanted['sensing']}') and "
            f"contains(Name,'_R{wanted['orbit']}_T{wanted['tile']}_')"
        )
    except ValueError:
        escaped_name = product_name.replace("'", "''")
        filter_text = f"Name eq '{escaped_name}'"
    last_error: Exception | None = None
    for attempt in range(1, int(args.auth_retries) + 2):
        session = requests.Session()
        try:
            with _cdse_catalogue_semaphore:
                response = session.get(
                    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
                    params={
                        "$filter": filter_text,
                        "$select": "Id,Name",
                        "$top": 20,
                    },
                    timeout=(
                        30.0,
                        min(120.0, float(args.node_request_timeout)),
                    ),
                )
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "0") or 0)
                response.close()
                time.sleep(max(5, min(120, retry_after or 5 * attempt)))
                raise RuntimeError("CDSE catalogue rate limited")
            response.raise_for_status()
            values = response.json().get("value", [])
            candidates = []
            for value in values:
                candidate_id = clean(value.get("Id"))
                candidate_name = clean(value.get("Name"))
                if not valid_cdse_product_id(candidate_id):
                    continue
                if candidate_name == product_name:
                    score = (-1, -1)
                else:
                    try:
                        score = product_match_score(
                            parse_s2_product_name(product_name),
                            candidate_name,
                        )
                    except ValueError:
                        score = None
                if score is not None:
                    candidates.append((score, candidate_name, candidate_id))
            if not candidates:
                raise RuntimeError(
                    f"no matching CDSE product for {product_name}"
                )
            candidates.sort()
            resolved = candidates[0][2]
            with _cdse_product_id_lock:
                _cdse_product_ids[product_name] = resolved
            return resolved
        except Exception as exc:
            last_error = exc
            if attempt > int(args.auth_retries):
                break
            time.sleep(min(20.0, 2.0 ** (attempt - 1)))
        finally:
            session.close()
    raise RuntimeError(
        f"cannot resolve real CDSE UUID for {product_name}: {last_error}"
    )


def follow_cdse_redirects(
    session: requests.Session,
    url: str,
    *,
    stream: bool,
    timeout: float,
) -> requests.Response:
    current_url = url
    for _ in range(8):
        response = session.get(
            current_url,
            allow_redirects=False,
            stream=stream,
            timeout=(min(60.0, timeout), timeout),
        )
        if response.status_code not in (301, 302, 303, 307, 308):
            return response
        location = clean(response.headers.get("Location", ""))
        response.close()
        if not location:
            raise RuntimeError(f"CDSE redirect without Location: {current_url}")
        current_url = location
    raise RuntimeError(f"too many CDSE redirects: {url}")


def cdse_nodes_request(
    token: Any,
    args: argparse.Namespace,
    url: str,
    *,
    stream: bool,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, int(args.auth_retries) + 2):
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {token.get()}"
        try:
            response = follow_cdse_redirects(
                session,
                url,
                stream=stream,
                timeout=float(args.node_request_timeout),
            )
            if response.status_code == 401:
                response.close()
                token.update()
                raise RuntimeError("CDSE Nodes token expired")
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "0") or 0)
                response.close()
                time.sleep(max(5, min(120, retry_after or 10 * attempt)))
                raise RuntimeError("CDSE Nodes rate limited")
            response.raise_for_status()
            setattr(response, "_panopticon_session", session)
            return response
        except Exception as exc:
            session.close()
            last_error = exc
            if attempt > int(args.auth_retries):
                break
            time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise RuntimeError(
        f"CDSE Nodes request failed after {int(args.auth_retries) + 1} "
        f"attempts for {url}: {last_error}"
    )


def close_cdse_response(response: requests.Response) -> None:
    session = getattr(response, "_panopticon_session", None)
    response.close()
    if session is not None:
        session.close()


def cdse_r20m_node_paths(
    token: Any,
    args: argparse.Namespace,
    product_id: str,
    product_name: str,
) -> dict[str, list[str]]:
    url = (
        f"{cdse_nodes_base_url(product_id, product_name)}"
        "/Nodes(MTD_MSIL2A.xml)/$value"
    )
    response = cdse_nodes_request(token, args, url, stream=False)
    try:
        root = ET.fromstring(response.content)
    finally:
        close_cdse_response(response)

    paths: dict[str, list[str]] = {}
    wanted = {band_name for band_name, _ in AWS_R20M_BANDS}
    for element in root.iter():
        if "IMAGE_FILE" not in element.tag.upper():
            continue
        value = clean(element.text)
        if "/R20m/" not in value:
            continue
        match = re.search(r"_B(01|02|03|04|05|06|07|8A|11|12)_20m$", value)
        if match is None:
            continue
        band_name = f"B{match.group(1)}"
        if band_name in wanted:
            paths[band_name] = (value + ".jp2").split("/")
    missing = sorted(wanted - set(paths))
    if missing:
        raise RuntimeError(
            f"MTD_MSIL2A.xml missing R20m bands {missing}: {product_name}"
        )
    return paths


def download_cdse_node_band(
    token: Any,
    args: argparse.Namespace,
    base_url: str,
    node_parts: list[str],
    destination: Path,
) -> Path:
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    url = (
        base_url
        + "".join(f"/Nodes({part})" for part in node_parts)
        + "/$value"
    )
    with args.node_band_semaphore:
        response = cdse_nodes_request(token, args, url, stream=True)
        temporary = destination.with_name(
            destination.name
            + f".part.{os.getpid()}.{threading.get_ident()}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_size = int(response.headers.get("Content-Length", "0") or 0)
        written = 0
        try:
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        written += len(chunk)
            if written <= 0:
                raise RuntimeError(f"empty CDSE node response: {url}")
            if expected_size and written != expected_size:
                raise RuntimeError(
                    f"incomplete CDSE node {written}/{expected_size}: {url}"
                )
            os.replace(temporary, destination)
            return destination
        finally:
            close_cdse_response(response)
            if temporary.exists():
                temporary.unlink()


def stage_cdse_r20m_nodes(
    token: Any,
    args: argparse.Namespace,
    product_id: str,
    product_name: str,
) -> Path:
    product_id = resolve_cdse_product_id(
        args,
        product_id,
        product_name,
    )
    destination = Path(args.product_scratch_dir) / product_name
    marker = destination / ".download_complete"
    if marker.is_file() and len(product_band_files(destination)) == len(
        AWS_R20M_BANDS
    ):
        return destination
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)

    node_paths = cdse_r20m_node_paths(
        token,
        args,
        product_id,
        product_name,
    )
    base_url = cdse_nodes_base_url(product_id, product_name)
    max_workers = min(
        len(AWS_R20M_BANDS),
        max(1, int(args.node_band_workers)),
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for band_name, _ in AWS_R20M_BANDS:
            node_parts = node_paths[band_name]
            source_name = node_parts[-1]
            futures.append(
                executor.submit(
                    download_cdse_node_band,
                    token,
                    args,
                    base_url,
                    node_parts,
                    destination / source_name,
                )
            )
        for future in as_completed(futures):
            future.result()
    band_files = product_band_files(destination)
    if len(band_files) != len(AWS_R20M_BANDS):
        raise RuntimeError(
            f"CDSE Nodes staged {len(band_files)}/{len(AWS_R20M_BANDS)} "
            f"bands for {product_name}"
        )
    marker.write_text("cdse_nodes_r20m_complete\n")
    return destination


def crop_cdse_nodes_product_group(
    legacy: Any,
    token: Any,
    args: argparse.Namespace,
    group: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    product_name = group[0]["product_name"]
    product_id = group[0]["product_id"]
    local_product = stage_cdse_r20m_nodes(
        token,
        args,
        product_id,
        product_name,
    )
    return crop_existing_product_group_to_window_cache(
        legacy,
        args,
        group,
        local_product,
        source_label=f"cdse_nodes_r20m:{product_name}",
    )


def cdse_stage_lock(product_name: str) -> Any:
    with _cdse_stage_locks_guard:
        lock = _cdse_stage_locks.get(product_name)
        if lock is None:
            lock = threading.RLock()
            _cdse_stage_locks[product_name] = lock
        return lock


def crop_cdse_mosaic_product_group(
    legacy: Any,
    token: Any,
    args: argparse.Namespace,
    group: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_map: dict[str, dict[str, str]] = {}
    for task in group:
        raw = clean(task.get("cdse_mosaic_json"))
        if not raw:
            raise RuntimeError("missing CDSE mosaic source list")
        for source in json.loads(raw):
            product_name = clean(source.get("product_name"))
            product_id = clean(source.get("product_id"))
            if product_name and product_id:
                source_map[product_name] = {
                    "product_name": product_name,
                    "product_id": product_id,
                }
    if not source_map:
        raise RuntimeError("empty CDSE mosaic source list")

    local_sources: list[dict[str, str]] = []
    staged_products: set[str] = set()
    source_locks = [
        cdse_stage_lock(product_name)
        for product_name in sorted(source_map)
    ]
    for lock in source_locks:
        lock.acquire()
    try:
        for source in source_map.values():
            product_name = source["product_name"]
            local_product = find_existing_product(
                product_name,
                args.existing_product_root_list,
            )
            if local_product is None:
                local_product = stage_cdse_r20m_nodes(
                    token,
                    args,
                    source["product_id"],
                    product_name,
                )
                staged_products.add(product_name)
            local_sources.append(
                {
                    "product_name": product_name,
                    "product_dir": str(local_product),
                }
            )
        local_json = json.dumps(
            local_sources,
            separators=(",", ":"),
        )
        local_group = [
            {
                **task,
                "local_mosaic_json": local_json,
            }
            for task in group
        ]
        return crop_local_mosaic_product_group(
            legacy,
            args,
            local_group,
            source_label="cdse_r20m_mosaic",
        )
    finally:
        if args.cleanup_product:
            for product_name in staged_products:
                cleanup_product(
                    Path(args.product_scratch_dir),
                    product_name,
                )
        for lock in reversed(source_locks):
            lock.release()


def crop_existing_product_group_to_window_cache(
    legacy: Any,
    args: argparse.Namespace,
    group: list[dict[str, Any]],
    source_dir: Path,
    source_label: str | None = None,
) -> list[dict[str, Any]]:
    ready: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in group:
        target = Path(task["target_raw_path"])
        if target_is_complete(target, task["timepoint"]) and not args.overwrite:
            ready.append(
                {
                    **task,
                    "status": "target_exists",
                    "raw_path": str(target),
                    "message": "",
                }
            )
        else:
            pending.append(task)
    if not pending:
        return ready

    arrays: dict[tuple[int, str], np.ndarray] = {}
    shapes: dict[tuple[int, str], tuple[int, int]] = {}
    georefs: dict[tuple[int, str], dict[str, Any]] = {}
    task_failures: dict[tuple[int, str], str] = {}
    band_files = product_band_files(source_dir)
    if not band_files:
        raise RuntimeError(f"no spectral 20 m bands found in {source_dir}")

    for band_file in band_files:
        match = re.search(r".*B([0-9A-Za-z]+)_20m\.jp2$", band_file.name)
        if match is None:
            continue
        spectrum_text = match.group(1)
        band_name = f"B{spectrum_text}"
        band_indices = dict(AWS_R20M_BANDS)
        if band_name not in band_indices:
            continue
        band_index = band_indices[band_name]
        with rasterio.open(band_file) as dataset:
            for task in pending:
                key = (int(task["row_index"]), task["timepoint"])
                if key in task_failures:
                    continue
                window = point_centered_window(
                    legacy,
                    dataset,
                    task["plume_bounds"],
                )
                if window is None:
                    task_failures[key] = (
                        "empty point-centered crop for "
                        f"{task['plume_id']} from {band_file}"
                    )
                    arrays.pop(key, None)
                    shapes.pop(key, None)
                    georefs.pop(key, None)
                    continue
                clipped = dataset.read(1, window=window)
                if key not in arrays:
                    shapes[key] = clipped.shape
                    arrays[key] = np.zeros(
                        (12, clipped.shape[0], clipped.shape[1]),
                        dtype=clipped.dtype,
                    )
                    crop_transform = rasterio.windows.transform(
                        window,
                        dataset.transform,
                    )
                    georefs[key] = {
                        "crs_wkt": dataset.crs.to_wkt() if dataset.crs else "",
                        "transform": list(crop_transform),
                        "height": int(clipped.shape[0]),
                        "width": int(clipped.shape[1]),
                        "plume_id": task["plume_id"],
                        "timepoint": task["timepoint"],
                        "product_id": task["product_id"],
                        "product_name": task["product_name"],
                        "reference_band": band_file.name,
                    }
                if clipped.shape != shapes[key]:
                    task_failures[key] = (
                        f"band shape mismatch for {task['plume_id']}: "
                        f"{clipped.shape} != {shapes[key]}"
                    )
                    arrays.pop(key, None)
                    shapes.pop(key, None)
                    georefs.pop(key, None)
                    continue
                arrays[key][band_index] = clipped

    if source_label is None:
        source_label = f"window_cache:{source_dir.parent}"
    for task in pending:
        key = (int(task["row_index"]), task["timepoint"])
        if key in task_failures:
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": task_failures[key],
                }
            )
            continue
        image = arrays.get(key)
        if image is None:
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": "no bands cropped",
                }
            )
            continue
        if image.shape != (12, 512, 512):
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": f"unexpected crop shape: {image.shape}",
                }
            )
            continue
        empty_valid_bands = [
            band_index
            for band_index in VALID_MODEL_BAND_INDICES
            if not np.any(image[band_index])
        ]
        if empty_valid_bands:
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": (
                        "empty local valid bands: "
                        f"{empty_valid_bands}"
                    ),
                }
            )
            continue
        uncovered_pixels = uncovered_pixel_count(image)
        if not center_is_covered(image):
            ready.append(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": (
                        "local crop center is uncovered; "
                        f"total_uncovered_pixels={uncovered_pixels}"
                    ),
                }
            )
            continue
        product_key = safe_component(task["product_id"])
        scratch_tif = (
            Path(args.crop_scratch_dir)
            / product_key
            / task["plume_id"]
            / f"{task['timepoint']}.tif"
        )
        scratch_tif.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(scratch_tif, image)
        target = Path(task["target_raw_path"])
        move_file_cross_fs(scratch_tif, target)
        if task["timepoint"] == "t0":
            write_json_atomic(georefs[key], georef_sidecar_path(target))
        ready.append(
            {
                **task,
                "status": "downloaded",
                "raw_path": str(target),
                "message": f"512x512; product_source={source_label}",
            }
        )
    return ready


def load_legacy_s2(repo_root: Path) -> Any:
    candidates = [
        repo_root
        / "data_preprocess"
        / "carbon_mapper_sentinel2_90360_plume_download.py",
        Path(
            "/home/yuyao/methane_train/data_preprocess/"
            "carbon_mapper_sentinel2_90360_plume_download.py"
        ),
    ]
    for path in candidates:
        if path.is_file():
            try:
                return load_module("legacy_s2_clean_table_download", path)
            except ModuleNotFoundError as exc:
                if exc.name != "pyproj":
                    raise
                compatibility_module = types.ModuleType("pyproj")

                class RasterioTransformer:
                    def __init__(self, source_crs: Any, target_crs: Any) -> None:
                        self.source_crs = source_crs
                        self.target_crs = target_crs

                    @classmethod
                    def from_crs(
                        cls,
                        source_crs: Any,
                        target_crs: Any,
                        always_xy: bool = True,
                    ) -> "RasterioTransformer":
                        if not always_xy:
                            raise ValueError("the compatibility transformer requires always_xy=True")
                        return cls(source_crs, target_crs)

                    def transform(self, longitude: float, latitude: float) -> tuple[float, float]:
                        eastings, northings = transform_coordinates(
                            self.source_crs,
                            self.target_crs,
                            [longitude],
                            [latitude],
                        )
                        return float(eastings[0]), float(northings[0])

                compatibility_module.Transformer = RasterioTransformer
                previous_module = sys.modules.get("pyproj")
                sys.modules["pyproj"] = compatibility_module
                try:
                    return load_module("legacy_s2_clean_table_download", path)
                finally:
                    if previous_module is None:
                        sys.modules.pop("pyproj", None)
                    else:
                        sys.modules["pyproj"] = previous_module
    raise FileNotFoundError(
        "legacy S2 module not found: "
        + ", ".join(str(path) for path in candidates)
    )


def load_config(legacy: Any, path: str) -> dict[str, Any]:
    if path and Path(path).exists():
        return legacy.load_config(path)
    return {}


def load_cdse_credentials(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, str]]:
    if args.cdse_username or args.cdse_password:
        if not args.cdse_username or not args.cdse_password:
            raise RuntimeError("provide both --cdse-username and --cdse-password")
        return [{"username": args.cdse_username, "password": args.cdse_password}]
    out: list[dict[str, str]] = []
    idx = args.cdse_env_index
    while True:
        username = os.environ.get(f"CDSE_USERNAME{idx}") or config.get(f"cdse_username{idx}")
        password = os.environ.get(f"CDSE_PASSWORD{idx}") or config.get(f"cdse_password{idx}")
        if username and password:
            out.append({"username": username, "password": password})
            idx += 1
            continue
        if username or password:
            raise RuntimeError(f"incomplete CDSE credential pair at index {idx}")
        break
    if not out:
        legacy_user = config.get("cdse_username")
        legacy_pass = config.get("cdse_password")
        if legacy_user and legacy_pass:
            out.append({"username": legacy_user, "password": legacy_pass})
    if not out:
        raise RuntimeError(f"CDSE credentials not found. Set CDSE_USERNAME{args.cdse_env_index}/CDSE_PASSWORD{args.cdse_env_index}.")
    return out


def start_tokens(legacy: Any, credentials: list[dict[str, str]]) -> list[Any]:
    tokens = []
    for cred in credentials:
        token = legacy.RefreshableAccessToken(cred["username"], cred["password"])
        tokens.append(token)
        thread = threading.Thread(target=legacy.refresh_variable, args=(token,), daemon=True)
        thread.start()
    return tokens


def sync_existing_targets(df: pd.DataFrame, timepoints: list[str]) -> int:
    changed = 0
    for idx, row in df.iterrows():
        for tp in timepoints:
            target_col = f"{tp}_download_target_raw_path"
            raw_col = f"{tp}_raw_path"
            target = existing_path(row.get(target_col, ""))
            if not target:
                continue
            if not existing_path(row.get(raw_col, "")):
                df.at[idx, raw_col] = target
                df.at[idx, f"{tp}_path_source"] = SUCCESS_SOURCE + ":target_exists_on_resume"
                df.at[idx, f"{tp}_local_status"] = "available"
                df.at[idx, f"{tp}_download_needed"] = 0
                df.at[idx, target_col] = ""
                changed += 1
    return changed


def save_table_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, path)


def build_tasks(
    df: pd.DataFrame,
    timepoints: list[str],
    overwrite: bool,
    limit: int,
    target_root: str,
    center_mode: str,
    stac_dn_add_override: int | None,
) -> tuple[list[dict[str, Any]], int]:
    tasks: list[dict[str, Any]] = []
    synced_targets = 0
    for idx, row in df.iterrows():
        for tp in timepoints:
            if target_root:
                target = str(
                    Path(target_root)
                    / tp
                    / clean(row["plume_id"])
                    / RAW_FILENAMES[tp]
                )
            else:
                target = clean(row.get(f"{tp}_download_target_raw_path", ""))
                if not target and overwrite:
                    target = clean(row.get(f"{tp}_raw_path", ""))
            if not target:
                continue
            if (
                not target_root
                and int(float(row.get(f"{tp}_download_needed", 0) or 0)) != 1
                and not overwrite
            ):
                continue
            if not target_root and existing_path(row.get(f"{tp}_raw_path", "")) and not overwrite:
                continue
            if target_is_complete(Path(target), tp) and not overwrite:
                if target_root:
                    df.at[idx, f"{tp}_raw_path"] = target
                    df.at[idx, f"{tp}_path_source"] = (
                        SUCCESS_SOURCE + ":target_exists_on_resume"
                    )
                    df.at[idx, f"{tp}_local_status"] = "available"
                    df.at[idx, f"{tp}_download_needed"] = 0
                    df.at[idx, f"{tp}_download_target_raw_path"] = ""
                    df.at[idx, f"{tp}_matched_old_timepoint"] = (
                        "product_crops"
                    )
                    df.at[idx, f"{tp}_recrop_status"] = (
                        "target_exists"
                    )
                    df.at[idx, f"{tp}_recrop_message"] = ""
                    synced_targets += 1
                continue
            product_id = clean(row.get(f"{tp}_product_id", ""))
            product_name = clean(row.get(f"{tp}_product_name", ""))
            if not product_id or not product_name:
                continue
            tasks.append(
                {
                    "row_index": idx,
                    "plume_id": clean(row["plume_id"]),
                    "timepoint": tp,
                    "product_id": product_id,
                    "product_name": product_name,
                    "product_local_path": clean(
                        row.get(f"{tp}_product_local_path", "")
                    ),
                    "stac_item_id": clean(
                        row.get(f"{tp}_stac_item_id", "")
                    ),
                    "stac_base_url": clean(
                        row.get(f"{tp}_stac_base_url", "")
                    ),
                    "stac_cog_base_url": clean(
                        row.get(f"{tp}_stac_cog_base_url", "")
                    ),
                    "stac_dn_add": (
                        str(stac_dn_add_override)
                        if stac_dn_add_override is not None
                        else clean(row.get(f"{tp}_stac_dn_add", "0")) or "0"
                    ),
                    "stac_source_product_name": clean(
                        row.get(
                            f"{tp}_stac_source_product_name",
                            "",
                        )
                    ),
                    "stac_mosaic_json": clean(
                        row.get(f"{tp}_stac_mosaic_json", "")
                    ),
                    "local_mosaic_json": clean(
                        row.get(f"{tp}_local_mosaic_json", "")
                    ),
                    "cdse_mosaic_json": clean(
                        row.get(f"{tp}_cdse_mosaic_json", "")
                    ),
                    "target_raw_path": target,
                    "plume_bounds": (
                        parse_bounds(row)
                        if center_mode == "plume_bounds"
                        else point_centered_crop_bounds(row)
                    ),
                }
            )
            if limit and len(tasks) >= limit:
                return tasks, synced_targets
    return tasks, synced_targets


def group_tasks_by_product(tasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in tasks:
        groups.setdefault((task["product_id"], task["product_name"]), []).append(task)
    return list(groups.values())


def download_one_crop(
    legacy: Any,
    token: Any,
    args: argparse.Namespace,
    task: dict[str, Any],
) -> dict[str, Any]:
    target = Path(task["target_raw_path"])
    if target_is_complete(target, task["timepoint"]) and not args.overwrite:
        return {**task, "status": "target_exists", "raw_path": str(target), "message": ""}

    product_key = safe_component(task["product_id"])
    scratch_tif = Path(args.crop_scratch_dir) / product_key / task["plume_id"] / f"{task['timepoint']}.tif"
    scratch_tif.parent.mkdir(parents=True, exist_ok=True)
    if scratch_tif.exists():
        scratch_tif.unlink()

    for attempt in range(1, args.auth_retries + 2):
        dims = legacy.download(
            token.get(),
            args.product_scratch_dir,
            task["plume_id"],
            task["product_id"],
            task["product_name"],
            task["plume_bounds"],
            str(scratch_tif),
            cleanup_product=False,
        )
        if dims is not None and scratch_tif.exists() and scratch_tif.stat().st_size > 0:
            move_file_cross_fs(scratch_tif, target)
            product_source = clean(task.get("product_cache_source", "cdse"))
            return {
                **task,
                "status": "downloaded",
                "raw_path": str(target),
                "message": (
                    f"{int(dims[0])}x{int(dims[1])}; attempts={attempt}; "
                    f"product_source={product_source}"
                ),
            }
        if attempt <= args.auth_retries:
            try:
                token.update()
            except Exception as exc:
                return {**task, "status": "failed", "raw_path": "", "message": f"token refresh failed: {exc}"}
            time.sleep(args.auth_retry_sleep)
    return {**task, "status": "failed", "raw_path": "", "message": "legacy download returned no crop"}


def process_product_group(
    legacy: Any,
    token: Any,
    args: argparse.Namespace,
    group: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not group:
        return []
    product_name = group[0]["product_name"]
    product_id = group[0]["product_id"]
    results: list[dict[str, Any]] = []
    lock = legacy.get_product_lock(product_name)
    stage_lock = (
        nullcontext()
        if clean(group[0].get("cdse_mosaic_json"))
        else cdse_stage_lock(product_name)
    )
    with lock, stage_lock:
        try:
            if clean(group[0].get("cdse_mosaic_json")):
                return crop_cdse_mosaic_product_group(
                    legacy,
                    token,
                    args,
                    group,
                )
            if clean(group[0].get("local_mosaic_json")):
                return crop_local_mosaic_product_group(
                    legacy,
                    args,
                    group,
                )
            if clean(group[0].get("stac_mosaic_json")):
                return crop_stac_mosaic_product_group(
                    legacy,
                    args,
                    group,
                )
            existing_source = args.existing_product_source_map.get(product_name)
            use_window_cache = (
                existing_source is not None
                and (
                    args.existing_cache_mode == "window"
                    or (
                        args.existing_cache_mode == "adaptive"
                        and len(group) < args.full_stage_min_tasks
                    )
                )
            )
            if use_window_cache:
                return crop_existing_product_group_to_window_cache(
                    legacy,
                    args,
                    group,
                    existing_source,
                )
            if existing_source is not None:
                local_product = stage_existing_product(
                    existing_source,
                    Path(args.product_scratch_dir),
                    product_name,
                )
                return crop_existing_product_group_to_window_cache(
                    legacy,
                    args,
                    group,
                    local_product,
                    source_label=f"full_cache:{existing_source.parent}",
                )
            if args.missing_source == "skip":
                message = "product is absent from all existing SAFE roots"
                return [
                    {
                        **task,
                        "status": "missing_local_product",
                        "raw_path": "",
                        "message": message,
                    }
                    for task in group
                ]
            if args.missing_source == "aws":
                if clean(group[0].get("stac_cog_base_url")):
                    return crop_stac_cog_product_group(
                        legacy,
                        args,
                        group,
                    )
                try:
                    return crop_aws_product_group_to_window_cache(
                        legacy,
                        args,
                        group,
                    )
                except Exception:
                    if clean(group[0].get("stac_cog_base_url")):
                        return crop_stac_cog_product_group(
                            legacy,
                            args,
                            group,
                        )
                    raise
            if args.missing_source == "cdse_nodes":
                return crop_cdse_nodes_product_group(
                    legacy,
                    token,
                    args,
                    group,
                )
            if args.missing_source == "hybrid":
                if clean(group[0].get("stac_cog_base_url")):
                    return crop_stac_cog_product_group(
                        legacy,
                        args,
                        group,
                    )
                try:
                    return crop_aws_product_group_to_window_cache(
                        legacy,
                        args,
                        group,
                    )
                except Exception as exc:
                    if clean(group[0].get("stac_cog_base_url")):
                        try:
                            print(
                                f"{product_name}: original R20m unavailable; "
                                "using STAC COG 20 m reconstruction: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            return crop_stac_cog_product_group(
                                legacy,
                                args,
                                group,
                            )
                        except Exception as cog_exc:
                            if str(product_id).startswith("aws_stac:"):
                                raise RuntimeError(
                                    "both original R20m and STAC COG "
                                    f"failed for {product_name}: {cog_exc}"
                                ) from cog_exc
                    print(
                        f"{product_name}: AWS window source unavailable; "
                        f"falling back to CDSE Nodes: {type(exc).__name__}: "
                        f"{exc}",
                        flush=True,
                    )
                    return crop_cdse_nodes_product_group(
                        legacy,
                        token,
                        args,
                        group,
                    )
            first_result = download_one_crop(
                legacy,
                token,
                args,
                {**group[0], "product_cache_source": "cdse"},
            )
            if first_result.get("status") not in {"downloaded", "target_exists"}:
                results.append(first_result)
                message = clean(first_result.get("message")) or "CDSE product download failed"
                results.extend(
                    {
                        **task,
                        "status": "failed",
                        "raw_path": "",
                        "message": message,
                    }
                    for task in group[1:]
                )
            elif group[0]["timepoint"] == "t0":
                local_product = Path(args.product_scratch_dir) / product_name
                results.extend(
                    crop_existing_product_group_to_window_cache(
                        legacy,
                        args,
                        group,
                        local_product,
                        source_label="cdse_local_cache",
                    )
                )
            else:
                results.append(first_result)
                if len(group) > 1:
                    local_product = Path(args.product_scratch_dir) / product_name
                    results.extend(
                        crop_existing_product_group_to_window_cache(
                            legacy,
                            args,
                            group[1:],
                            local_product,
                            source_label="cdse_local_cache",
                        )
                    )
        except Exception as exc:
            message = f"product group failed: {type(exc).__name__}: {exc}"
            print(f"{product_name}: {message}", flush=True)
            results.extend(
                {
                    **task,
                    "status": "failed",
                    "raw_path": "",
                    "message": message,
                }
                for task in group[len(results):]
            )
        finally:
            if args.cleanup_product:
                cleanup_product(Path(args.product_scratch_dir), product_name)
                cleanup_crop_scratch(Path(args.crop_scratch_dir), product_id)
    return results


def apply_results_to_table(df: pd.DataFrame, results: list[dict[str, Any]]) -> int:
    changed = 0
    for record in results:
        idx = int(record["row_index"])
        tp = record["timepoint"]
        status = clean(record.get("status"))
        df.at[idx, f"{tp}_recrop_status"] = status
        df.at[idx, f"{tp}_recrop_message"] = clean(record.get("message"))
        if status not in {"downloaded", "target_exists"}:
            continue
        raw_path = existing_path(record.get("raw_path", ""))
        if not raw_path:
            continue
        df.at[idx, f"{tp}_raw_path"] = raw_path
        df.at[idx, f"{tp}_path_source"] = SUCCESS_SOURCE
        df.at[idx, f"{tp}_local_status"] = "available"
        df.at[idx, f"{tp}_download_needed"] = 0
        df.at[idx, f"{tp}_download_target_raw_path"] = ""
        df.at[idx, f"{tp}_matched_old_timepoint"] = "product_crops"
        changed += 1
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Point-center recrop S2 timepoints from grouped CDSE SAFE products."
    )
    parser.add_argument("--table", default="Upgrade_data_pipeline/csv/s2_6time_clean_paths.csv")
    parser.add_argument("--output-table", default="")
    parser.add_argument("--target-root", default="")
    parser.add_argument("--timepoints", default="t0,prev1,prev2,prev3,seasonal,year")
    parser.add_argument(
        "--center-mode",
        choices=("point", "plume_bounds"),
        default="point",
        help=(
            "Center 512 crops on the plume point or reproduce the historical "
            "pipeline by centering on the Carbon Mapper plume bounds."
        ),
    )
    parser.add_argument(
        "--stac-dn-add-override",
        type=int,
        default=None,
        help=(
            "Override table stac_dn_add for STAC COG reads. Use 0 for raw COG "
            "values; the historical engg-leung S2 DN distribution requires "
            "+1000 for the current Earth Search COGs."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--legacy-config", default="")
    parser.add_argument("--product-scratch-dir", default="/diniuvol/yuyao/s2_download_scratch/products")
    parser.add_argument("--crop-scratch-dir", default="/diniuvol/yuyao/s2_download_scratch/crops")
    parser.add_argument(
        "--existing-product-roots",
        default=DEFAULT_EXISTING_PRODUCT_ROOTS,
        help="Comma-separated extracted SAFE roots, searched before CDSE download.",
    )
    parser.add_argument(
        "--existing-cache-mode",
        choices=("adaptive", "window", "full"),
        default="adaptive",
        help="Cache remote SAFE data as needed windows, full products, or adapt by reuse.",
    )
    parser.add_argument(
        "--full-stage-min-tasks",
        type=int,
        default=32,
        help="In adaptive mode, fully stage products reused by at least this many crops.",
    )
    parser.add_argument(
        "--missing-source",
        choices=("skip", "aws", "cdse_nodes", "hybrid", "cdse"),
        default="cdse",
        help=(
            "Source for products absent from existing SAFE roots. "
            "skip records them as unavailable without network access; "
            "aws reads public R20m JP2 windows; cdse_nodes downloads only "
            "the ten R20m files; hybrid tries aws then cdse_nodes; "
            "cdse downloads full ZIPs."
        ),
    )
    parser.add_argument(
        "--aws-band-workers",
        type=int,
        default=2,
        help="Concurrent public AWS R20m bands within each missing product.",
    )
    parser.add_argument(
        "--aws-item-max-index",
        type=int,
        default=12,
        help="Maximum Earth Search item indices checked per tile and day.",
    )
    parser.add_argument(
        "--aws-read-retries",
        type=int,
        default=3,
        help="Retries for Earth Search metadata and public S3 JP2 reads.",
    )
    parser.add_argument(
        "--aws-request-timeout",
        type=float,
        default=90.0,
        help="HTTP timeout in seconds for public AWS source discovery and reads.",
    )
    parser.add_argument(
        "--aws-require-exact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject same-acquisition AWS fallbacks and use another source.",
    )
    parser.add_argument(
        "--node-band-workers",
        type=int,
        default=4,
        help="Concurrent R20m band downloads for each CDSE Nodes product.",
    )
    parser.add_argument(
        "--node-global-workers",
        type=int,
        default=48,
        help="Global cap on simultaneous CDSE Nodes band streams.",
    )
    parser.add_argument(
        "--node-request-timeout",
        type=float,
        default=300.0,
        help="Read timeout in seconds for individual CDSE Nodes band files.",
    )
    parser.add_argument("--cdse-env-index", type=int, default=1)
    parser.add_argument("--cdse-username", default="")
    parser.add_argument("--cdse-password", default="")
    parser.add_argument("--auth-retries", type=int, default=3)
    parser.add_argument("--auth-retry-sleep", type=float, default=2.0)
    parser.add_argument("--sync-interval", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-cleanup-product", dest="cleanup_product", action="store_false")
    parser.set_defaults(cleanup_product=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--legacy-r20m-layout",
        action="store_true",
        help=(
            "Reproduce the historical CDSE R20m tensor layout: "
            "B01..B07,B8A,0,0,B11,B12."
        ),
    )
    return parser.parse_args()


def main() -> int:
    global AWS_R20M_BANDS, STAC_COG_BANDS, VALID_MODEL_BAND_INDICES
    args = parse_args()
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    requested_limit = min(hard_limit, 65536)
    if soft_limit < requested_limit:
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (requested_limit, hard_limit),
        )
    print(
        f"[info] open-file soft limit="
        f"{resource.getrlimit(resource.RLIMIT_NOFILE)[0]}",
        flush=True,
    )
    if args.legacy_r20m_layout:
        AWS_R20M_BANDS = LEGACY_R20M_BANDS
        STAC_COG_BANDS = LEGACY_R20M_BANDS
        VALID_MODEL_BAND_INDICES = tuple(
            band_index for _, band_index in LEGACY_R20M_BANDS
        )
    args.existing_product_root_list = parse_existing_product_roots(
        args.existing_product_roots
    )
    args.node_band_semaphore = threading.BoundedSemaphore(
        max(1, int(args.node_global_workers))
    )
    timepoints = [tp.strip() for tp in args.timepoints.split(",") if tp.strip()]
    bad = sorted(set(timepoints) - set(TIMEPOINTS))
    if bad:
        raise ValueError(f"unsupported timepoints: {bad}")
    Path(args.product_scratch_dir).mkdir(parents=True, exist_ok=True)
    Path(args.crop_scratch_dir).mkdir(parents=True, exist_ok=True)

    table_path = Path(args.table)
    output_table_path = Path(args.output_table) if args.output_table else table_path
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    load_path = output_table_path if output_table_path.exists() else table_path
    df = pd.read_csv(load_path, low_memory=False)
    for tp in timepoints:
        mutable_text_columns = [
            f"{tp}_raw_path",
            f"{tp}_path_source",
            f"{tp}_local_status",
            f"{tp}_download_target_raw_path",
            f"{tp}_matched_old_timepoint",
        ]
        for column in mutable_text_columns:
            if column not in df.columns:
                df[column] = ""
            df[column] = df[column].astype(object)
        original_column = f"{tp}_original_raw_path"
        if original_column not in df.columns:
            df[original_column] = df.get(f"{tp}_raw_path", "")
        for suffix in ("recrop_status", "recrop_message"):
            column = f"{tp}_{suffix}"
            if column not in df.columns:
                df[column] = ""
    if not args.target_root:
        changed = sync_existing_targets(df, timepoints)
        if changed:
            save_table_atomic(df, output_table_path)
            print(f"synced existing target files into table: {changed}", flush=True)

    tasks, synced_targets = build_tasks(
        df,
        timepoints,
        args.overwrite,
        args.limit,
        args.target_root,
        args.center_mode,
        args.stac_dn_add_override,
    )
    if synced_targets:
        save_table_atomic(df, output_table_path)
        print(
            f"synced target-root files into table: {synced_targets}",
            flush=True,
        )
    groups = group_tasks_by_product(tasks)
    product_root = Path(args.product_scratch_dir)
    product_sources = {}
    for group in groups:
        product_name = group[0]["product_name"]
        direct_path = clean(group[0].get("product_local_path"))
        product_sources[product_name] = (
            Path(direct_path)
            if direct_path
            else find_existing_product(
                product_name,
                args.existing_product_root_list,
            )
        )
    args.existing_product_source_map = product_sources
    unresolved_product_names = [
        product_name
        for product_name, source in product_sources.items()
        if source is None
    ]
    acquisition_fallbacks = 0
    if unresolved_product_names:
        acquisition_index = build_existing_acquisition_index(
            args.existing_product_root_list
        )
        for product_name in unresolved_product_names:
            key = product_acquisition_key(product_name)
            candidate = acquisition_index.get(key) if key is not None else None
            if candidate is not None and product_band_files(candidate):
                product_sources[product_name] = candidate
                acquisition_fallbacks += 1
    groups.sort(
        key=lambda group: (
            not (product_root / group[0]["product_name"]).is_dir(),
            product_sources[group[0]["product_name"]] is None,
        )
    )
    cached_groups = sum(
        product_sources[group[0]["product_name"]] is not None
        for group in groups
    )
    window_groups = sum(
        source is not None
        and (
            args.existing_cache_mode == "window"
            or (
                args.existing_cache_mode == "adaptive"
                and len(group) < args.full_stage_min_tasks
            )
        )
        for group, source in (
            (group, product_sources[group[0]["product_name"]])
            for group in groups
        )
    )
    source_classes = Counter()
    for group in groups:
        first = group[0]
        product_name = first["product_name"]
        if clean(first.get("cdse_mosaic_json")):
            source_classes["cdse_mosaic"] += 1
        elif clean(first.get("local_mosaic_json")):
            source_classes["local_mosaic"] += 1
        elif clean(first.get("stac_mosaic_json")):
            source_classes["stac_mosaic"] += 1
        elif product_sources[product_name] is not None:
            source_classes["existing_safe"] += 1
        elif clean(first.get("stac_cog_base_url")):
            source_classes["stac_cog"] += 1
        else:
            source_classes["unresolved_product"] += 1
    print(
        f"tasks={len(tasks)} product_groups={len(groups)} cached_groups={cached_groups} "
        f"window_groups={window_groups} full_cache_groups={cached_groups - window_groups} "
        f"missing_groups={len(groups) - cached_groups} "
        f"acquisition_fallbacks={acquisition_fallbacks} "
        f"source_classes={dict(sorted(source_classes.items()))} "
        f"missing_source={args.missing_source} "
        f"aws_band_workers={args.aws_band_workers} "
        f"node_band_workers={args.node_band_workers} "
        f"node_global_workers={args.node_global_workers} "
        f"timepoints={','.join(timepoints)}",
        flush=True,
    )
    if args.dry_run:
        for tp in timepoints:
            subset = [t for t in tasks if t["timepoint"] == tp]
            print(f"{tp}: tasks={len(subset)} unique_products={len({t['product_id'] for t in subset})}", flush=True)
        return 0
    if not tasks:
        return 0

    repo_root = Path.cwd()
    legacy = load_legacy_s2(repo_root)
    config = load_config(legacy, args.legacy_config)
    with legacy.proxy_manager_lock:
        legacy.proxy_manager = legacy.build_proxy_manager(config)
    if args.missing_source in {"cdse", "cdse_nodes", "hybrid"}:
        credentials = load_cdse_credentials(args, config)
        tokens = start_tokens(legacy, credentials)
    else:
        tokens = [None]

    completed_groups = 0
    completed_tasks = 0
    failed_tasks = 0
    pending_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for idx, group in enumerate(groups):
            token = tokens[idx % len(tokens)]
            futures.append(executor.submit(process_product_group, legacy, token, args, group))
        try:
            for future in as_completed(futures):
                records = future.result()
                completed_groups += 1
                completed_tasks += len(records)
                pending_records.extend(records)
                ok = sum(1 for r in records if r.get("status") in {"downloaded", "target_exists"})
                failed = len(records) - ok
                failed_tasks += failed
                print(
                    f"group {completed_groups}/{len(groups)} tasks={len(records)} ok={ok} failed={failed} "
                    f"done_tasks={completed_tasks}/{len(tasks)}",
                    flush=True,
                )
                if len(pending_records) >= args.sync_interval:
                    applied = apply_results_to_table(df, pending_records)
                    save_table_atomic(df, output_table_path)
                    print(f"synced table records={len(pending_records)} applied={applied}", flush=True)
                pending_records.clear()
        finally:
            if pending_records:
                applied = apply_results_to_table(df, pending_records)
                save_table_atomic(df, output_table_path)
                print(f"synced table records={len(pending_records)} applied={applied}", flush=True)
    if failed_tasks:
        print(f"completed with failed_tasks={failed_tasks}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
