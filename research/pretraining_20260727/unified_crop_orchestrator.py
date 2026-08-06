#!/usr/bin/env python3
"""Plan leak-free, geographically aligned six-timepoint crop tasks.

This is a dry-run-first orchestrator.  It creates a deterministic task
manifest and can read a small number of tasks back as a smoke test, but it
never bulk-materializes image crops.  A task has one canonical WGS84 query
centre and one east/north metric offset; every available sensor and all six
timepoints consume that same geographic query.

Labels come only from spatial overlap with the local Carbon Mapper plume mask.
Sensor masks are diagnostics and grid references, not competing label
authorities.  Sentinel-5P pixels must be resolved independently for every
query centre and every timepoint from the NetCDF latitude/longitude arrays;
the old fixed nearest plume-centre indices are explicitly forbidden.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import tempfile
import threading
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import rasterio
import tifffile
from affine import Affine
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import geometry_mask
from rasterio.warp import transform
from rasterio.windows import Window

try:
    from netCDF4 import Dataset
except Exception:
    Dataset = None

try:
    import h5py
except Exception:
    h5py = None


warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

SCHEMA_VERSION = "methanefuse_unified_crop_task_v1"
TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
SENSOR_ORDER = ("s2", "l89", "emit", "s5p")
RASTER_SENSORS = ("s2", "l89", "emit")
SENSOR_BITS = {sensor: index for index, sensor in enumerate(SENSOR_ORDER)}
S5P_IO_LOCK = threading.Lock()
CH4_CANDIDATES = (
    "methane_mixing_ratio_bias_corrected",
    "methane_mixing_ratio",
    "xch4",
)

DEFAULT_WIDE_MANIFEST = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/manifests/"
    "multisensor_6time_512_wide.csv"
)
DEFAULT_S2_MASK_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/"
    "cache/s2_512_masks"
)
DEFAULT_CM_ROOT = Path(
    "/mnt/engg-niulab/yuyao/sensors_raw_data/CM"
)
DEFAULT_TASK_CSV = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/manifests/"
    "unified_crop_tasks.csv"
)
DEFAULT_SCHEMA_JSON = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/manifests/"
    "unified_crop_tasks.schema.json"
)
DEFAULT_AUDIT_JSON = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/manifests/"
    "unified_crop_tasks.audit.json"
)


@dataclass(frozen=True)
class RasterGrid:
    sensor: str
    reference_path: Path
    grid_source: str
    crs: CRS
    transform: Affine
    width: int
    height: int
    pixel_size_x: float
    pixel_size_y: float


@dataclass(frozen=True)
class QueryGeometry:
    label: int
    longitude: float
    latitude: float
    east_m: float
    north_m: float
    positive_pixels: int
    query_pixels: int
    footprint_wgs84: tuple[tuple[float, float], ...]


@dataclass
class PlumePlan:
    plume_id: str
    tasks: list[dict[str, Any]]
    status: str
    reason: str
    sensors: tuple[str, ...]
    requested_positive: int
    requested_negative: int
    made_positive: int
    made_negative: int
    dropped_sensors: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wide-manifest",
        type=Path,
        default=DEFAULT_WIDE_MANIFEST,
    )
    parser.add_argument(
        "--s2-mask-root",
        type=Path,
        default=DEFAULT_S2_MASK_ROOT,
    )
    parser.add_argument("--cm-root", type=Path, default=DEFAULT_CM_ROOT)
    parser.add_argument("--task-csv", type=Path, default=DEFAULT_TASK_CSV)
    parser.add_argument(
        "--schema-json",
        type=Path,
        default=DEFAULT_SCHEMA_JSON,
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=DEFAULT_AUDIT_JSON,
    )
    parser.add_argument(
        "--smoke-validation-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--write-task-manifest",
        action="store_true",
        help=(
            "Write task metadata only. This never materializes image crops."
        ),
    )
    parser.add_argument("--query-size-m", type=float, default=360.0)
    parser.add_argument("--n-pos", type=int, default=2)
    parser.add_argument("--n-neg", type=int, default=2)
    parser.add_argument("--max-offset-m", type=float, default=4000.0)
    parser.add_argument("--max-negative-attempts", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--plume-id",
        action="append",
        default=[],
        help="Exact selection; repeat for multiple plume IDs.",
    )
    parser.add_argument(
        "--sample-plumes",
        type=int,
        default=0,
        help="Stable hash sample from the selected wide manifest.",
    )
    parser.add_argument(
        "--sample-per-combination",
        type=int,
        default=0,
        help=(
            "Stable sample from each sensor combination. Intended for "
            "partial-sensor smoke tests."
        ),
    )
    parser.add_argument(
        "--smoke-validate-tasks",
        type=int,
        default=0,
        help=(
            "Read sources and validate this many planned tasks in memory. "
            "No crop files are written."
        ),
    )
    return parser.parse_args()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def _parse_bool(value: Any) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "y"}


def _file_ok(value: Any) -> bool:
    text = _clean(value)
    if not text:
        return False
    try:
        path = Path(text)
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _safe_component(value: Any) -> str:
    text = _clean(value)
    safe = "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in text
    )
    return safe[:220] if safe else "missing"


def _stable_digest(seed: int, *parts: Any) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_rng(seed: int, *parts: Any) -> np.random.Generator:
    digest = _stable_digest(seed, *parts)
    value = int(digest[:16], 16)
    return np.random.default_rng(value)


def _time_path_columns(sensor: str) -> list[str]:
    return [f"{sensor}_{timepoint}_path" for timepoint in TIMEPOINTS]


def _required_wide_columns() -> set[str]:
    columns = {
        "plume_id",
        "event_group_id",
        "event_time",
        "split",
        "plume_latitude",
        "plume_longitude",
        "plume_bounds",
        "has_s2",
        "has_l89",
        "has_emit",
        "has_s5p",
        "l89_mask_path",
        "l89_mask_exists",
        "emit_mask_path",
        "emit_mask_exists",
    }
    for sensor in SENSOR_ORDER:
        columns.update(_time_path_columns(sensor))
    return columns


def _optional_wide_columns() -> set[str]:
    columns = {
        "available_sensors",
        "crop_ready_sensors",
        "plume_tif",
    }
    for timepoint in TIMEPOINTS:
        for suffix in (
            "qc_center_iy",
            "qc_center_ix",
            "qc_center_distance_km",
            "qc_patch_missing_ratio",
        ):
            columns.add(f"s5p_{timepoint}_{suffix}")
    return columns


def _require_unique(frame: pd.DataFrame, name: str) -> None:
    if frame["plume_id"].astype(str).duplicated().any():
        duplicates = (
            frame.loc[
                frame["plume_id"].astype(str).duplicated(keep=False),
                "plume_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"{name} has duplicate plume_id values: {duplicates[:20]}"
        )


def _availability_for_row(
    row: Mapping[str, Any],
    s2_mask_root: Path,
) -> tuple[tuple[str, ...], dict[str, str]]:
    masks: dict[str, str] = {}
    sensors: list[str] = []
    s2_mask = (
        s2_mask_root
        / _safe_component(row.get("plume_id", ""))
        / "s2_mask_512.tif"
    )
    if _parse_bool(row.get("has_s2", "")) and _file_ok(s2_mask):
        sensors.append("s2")
        masks["s2"] = str(s2_mask)
    if (
        _parse_bool(row.get("has_l89", ""))
        and _parse_bool(row.get("l89_mask_exists", ""))
        and _clean(row.get("l89_mask_path", ""))
    ):
        sensors.append("l89")
        masks["l89"] = _clean(row.get("l89_mask_path", ""))
    if (
        _parse_bool(row.get("has_emit", ""))
        and _parse_bool(row.get("emit_mask_exists", ""))
        and _clean(row.get("emit_mask_path", ""))
    ):
        sensors.append("emit")
        masks["emit"] = _clean(row.get("emit_mask_path", ""))
    if _parse_bool(row.get("has_s5p", "")):
        sensors.append("s5p")
    return tuple(sensor for sensor in SENSOR_ORDER if sensor in sensors), masks


def _sensor_combo(sensors: Sequence[str]) -> str:
    return "|".join(sensor for sensor in SENSOR_ORDER if sensor in sensors)


def _load_wide(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not args.wide_manifest.is_file():
        raise FileNotFoundError(
            f"wide manifest does not exist: {args.wide_manifest}"
        )
    header = list(pd.read_csv(args.wide_manifest, nrows=0).columns)
    required = _required_wide_columns()
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"wide manifest is missing columns: {missing}")
    selected = [
        column
        for column in header
        if column in required | _optional_wide_columns()
    ]
    frame = pd.read_csv(
        args.wide_manifest,
        usecols=selected,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    _require_unique(frame, "wide manifest")
    all_rows = len(frame)

    group_split_counts = frame.groupby(
        "event_group_id", observed=True
    )["split"].nunique()
    split_conflicts = group_split_counts[group_split_counts.gt(1)]
    if not split_conflicts.empty:
        raise ValueError(
            f"{len(split_conflicts)} event groups span multiple splits"
        )

    availability = [
        _availability_for_row(row, args.s2_mask_root)
        for row in frame.to_dict("records")
    ]
    frame["_ready_sensors"] = [item[0] for item in availability]
    frame["_sensor_masks"] = [item[1] for item in availability]
    frame["_sensor_combo"] = frame["_ready_sensors"].map(_sensor_combo)
    frame = frame[frame["_sensor_combo"].ne("")].copy()
    ready_rows = len(frame)

    selection_modes = sum(
        bool(value)
        for value in (
            args.plume_id,
            args.sample_plumes,
            args.sample_per_combination,
        )
    )
    if selection_modes > 1:
        raise ValueError(
            "use only one of --plume-id, --sample-plumes, or "
            "--sample-per-combination"
        )
    requested_missing: list[str] = []
    if args.plume_id:
        requested = set(args.plume_id)
        present = set(frame["plume_id"].astype(str))
        requested_missing = sorted(requested - present)
        frame = frame[frame["plume_id"].isin(requested)].copy()
    elif args.sample_plumes:
        if args.sample_plumes < 1:
            raise ValueError("--sample-plumes must be positive")
        frame["_rank"] = frame["plume_id"].map(
            lambda value: _stable_digest(args.seed, value)
        )
        frame = (
            frame.sort_values(["_rank", "plume_id"], kind="mergesort")
            .head(args.sample_plumes)
            .drop(columns="_rank")
            .copy()
        )
    elif args.sample_per_combination:
        if args.sample_per_combination < 1:
            raise ValueError("--sample-per-combination must be positive")
        frame["_rank"] = frame["plume_id"].map(
            lambda value: _stable_digest(args.seed, value)
        )
        frame = (
            frame.sort_values(
                ["_sensor_combo", "_rank", "plume_id"],
                kind="mergesort",
            )
            .groupby("_sensor_combo", observed=True, sort=True)
            .head(args.sample_per_combination)
            .drop(columns="_rank")
            .copy()
        )
    if frame.empty:
        raise ValueError("no crop-ready rows remain after selection")
    frame = frame.sort_values("plume_id", kind="mergesort").reset_index(
        drop=True
    )
    audit = {
        "path": str(args.wide_manifest),
        "rows": int(all_rows),
        "rows_with_at_least_one_crop_ready_sensor": int(ready_rows),
        "selected_rows": int(len(frame)),
        "selected_sensor_combinations": {
            str(key): int(value)
            for key, value in frame["_sensor_combo"]
            .value_counts()
            .sort_index()
            .items()
        },
        "requested_missing_or_not_crop_ready": requested_missing,
        "event_groups_spanning_splits": 0,
    }
    return frame, audit


def _raw_binary(dataset: rasterio.io.DatasetReader) -> np.ndarray:
    dataset_mask = dataset.dataset_mask()
    if np.any(dataset_mask == 0):
        binary = dataset_mask > 0
    else:
        band = dataset.read(1)
        binary = np.isfinite(band) & (band > 0)
    return binary.astype(np.uint8)


def _real_grid(
    sensor: str,
    path: Path,
    *,
    grid_source: str,
) -> RasterGrid:
    if not _file_ok(path):
        raise RuntimeError(f"missing_mask:{path}")
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise RuntimeError(f"mask_missing_crs:{path}")
        if dataset.transform.almost_equals(Affine.identity()):
            raise RuntimeError(f"mask_identity_transform:{path}")
        if dataset.width != 512 or dataset.height != 512:
            raise RuntimeError(
                f"mask_not_512:{dataset.height}x{dataset.width}:{path}"
            )
        pixel_x = math.hypot(dataset.transform.a, dataset.transform.d)
        pixel_y = math.hypot(dataset.transform.b, dataset.transform.e)
        if not (
            math.isfinite(pixel_x)
            and math.isfinite(pixel_y)
            and pixel_x > 0
            and pixel_y > 0
        ):
            raise RuntimeError(f"invalid_pixel_size:{path}")
        return RasterGrid(
            sensor=sensor,
            reference_path=path,
            grid_source=grid_source,
            crs=dataset.crs,
            transform=dataset.transform,
            width=int(dataset.width),
            height=int(dataset.height),
            pixel_size_x=float(pixel_x),
            pixel_size_y=float(pixel_y),
        )


def _query_window(
    longitude: float,
    latitude: float,
    grid: RasterGrid,
    query_size_m: float,
) -> dict[str, Any] | None:
    x_values, y_values = transform(
        "EPSG:4326",
        grid.crs,
        [longitude],
        [latitude],
    )
    column, row = (~grid.transform) * (
        float(x_values[0]),
        float(y_values[0]),
    )
    patch_width = max(1, int(math.ceil(query_size_m / grid.pixel_size_x)))
    patch_height = max(
        1, int(math.ceil(query_size_m / grid.pixel_size_y))
    )
    x0 = int(math.floor(float(column) - patch_width / 2))
    y0 = int(math.floor(float(row) - patch_height / 2))
    if (
        x0 < 0
        or y0 < 0
        or x0 + patch_width > grid.width
        or y0 + patch_height > grid.height
    ):
        return None
    return {
        "query_col_t0": float(column),
        "query_row_t0": float(row),
        "crop_x0": x0,
        "crop_y0": y0,
        "crop_width_px": patch_width,
        "crop_height_px": patch_height,
        "grid_crs": grid.crs.to_string(),
        "grid_transform": json.dumps(
            [float(value) for value in grid.transform[:6]]
        ),
        "grid_reference_path": str(grid.reference_path),
        "grid_source": grid.grid_source,
        "pixel_size_x_m": grid.pixel_size_x,
        "pixel_size_y_m": grid.pixel_size_y,
    }


def _offset_to_lonlat(
    base_longitude: float,
    base_latitude: float,
    east_m: float,
    north_m: float,
) -> tuple[float, float]:
    if east_m == 0 and north_m == 0:
        return base_longitude, base_latitude
    longitude, latitude = transform(
        _local_metric_crs(base_longitude, base_latitude),
        "EPSG:4326",
        [east_m],
        [north_m],
    )
    return float(longitude[0]), float(latitude[0])


def _lonlat_to_offset(
    base_longitude: float,
    base_latitude: float,
    longitude: float,
    latitude: float,
) -> tuple[float, float]:
    east, north = transform(
        "EPSG:4326",
        _local_metric_crs(base_longitude, base_latitude),
        [longitude],
        [latitude],
    )
    return float(east[0]), float(north[0])


@lru_cache(maxsize=8192)
def _local_metric_crs(
    base_longitude: float,
    base_latitude: float,
) -> CRS:
    """Local WGS84 azimuthal-equidistant frame for canonical metre offsets."""

    return CRS.from_string(
        "+proj=aeqd "
        f"+lat_0={base_latitude:.12f} "
        f"+lon_0={base_longitude:.12f} "
        "+datum=WGS84 +units=m +no_defs"
    )


def _canonical_query(
    *,
    raw_binary: np.ndarray,
    raw_crs: CRS,
    raw_transform: Affine,
    base_longitude: float,
    base_latitude: float,
    longitude: float,
    latitude: float,
    query_size_m: float,
) -> QueryGeometry:
    center_x, center_y = transform(
        "EPSG:4326",
        raw_crs,
        [longitude],
        [latitude],
    )
    x = float(center_x[0])
    y = float(center_y[0])
    half = query_size_m / 2.0
    projected_ring = (
        (x - half, y - half),
        (x + half, y - half),
        (x + half, y + half),
        (x - half, y + half),
        (x - half, y - half),
    )
    geometry = {
        "type": "Polygon",
        "coordinates": [projected_ring],
    }
    inside = geometry_mask(
        [geometry],
        out_shape=raw_binary.shape,
        transform=raw_transform,
        invert=True,
        all_touched=True,
    )
    query_pixels = int(np.count_nonzero(inside))
    positive_pixels = int(
        np.count_nonzero((raw_binary > 0) & inside)
    )
    xs = [point[0] for point in projected_ring]
    ys = [point[1] for point in projected_ring]
    lons, lats = transform(raw_crs, "EPSG:4326", xs, ys)
    footprint_wgs84 = tuple(
        (float(lon), float(lat)) for lon, lat in zip(lons, lats)
    )
    east, north = _lonlat_to_offset(
        base_longitude,
        base_latitude,
        longitude,
        latitude,
    )
    return QueryGeometry(
        label=int(positive_pixels > 0),
        longitude=longitude,
        latitude=latitude,
        east_m=east,
        north_m=north,
        positive_pixels=positive_pixels,
        query_pixels=query_pixels,
        footprint_wgs84=footprint_wgs84,
    )


def _fits_all_rasters(
    query: QueryGeometry,
    grids: Mapping[str, RasterGrid],
    query_size_m: float,
) -> tuple[bool, dict[str, dict[str, Any]]]:
    windows: dict[str, dict[str, Any]] = {}
    for sensor in RASTER_SENSORS:
        if sensor not in grids:
            continue
        window = _query_window(
            query.longitude,
            query.latitude,
            grids[sensor],
            query_size_m,
        )
        if window is None:
            return False, {}
        windows[sensor] = window
    return True, windows


def _positive_candidates(
    *,
    raw_binary: np.ndarray,
    raw_crs: CRS,
    raw_transform: Affine,
    base_longitude: float,
    base_latitude: float,
    query_size_m: float,
    max_offset_m: float,
    grids: Mapping[str, RasterGrid],
    rng: np.random.Generator,
    target: int,
) -> list[tuple[QueryGeometry, dict[str, dict[str, Any]]]]:
    positive_rows, positive_columns = np.nonzero(raw_binary)
    if len(positive_rows) == 0:
        return []
    order = rng.permutation(len(positive_rows))
    candidates: list[
        tuple[QueryGeometry, dict[str, dict[str, Any]]]
    ] = []
    seen: set[tuple[int, int]] = set()
    for index in order:
        row = int(positive_rows[index])
        column = int(positive_columns[index])
        x, y = rasterio.transform.xy(
            raw_transform,
            row,
            column,
            offset="center",
        )
        longitudes, latitudes = transform(
            raw_crs,
            "EPSG:4326",
            [float(x)],
            [float(y)],
        )
        query = _canonical_query(
            raw_binary=raw_binary,
            raw_crs=raw_crs,
            raw_transform=raw_transform,
            base_longitude=base_longitude,
            base_latitude=base_latitude,
            longitude=float(longitudes[0]),
            latitude=float(latitudes[0]),
            query_size_m=query_size_m,
        )
        if query.label != 1:
            continue
        if math.hypot(query.east_m, query.north_m) > max_offset_m:
            continue
        key = (round(query.east_m), round(query.north_m))
        if key in seen:
            continue
        fits, windows = _fits_all_rasters(
            query,
            grids,
            query_size_m,
        )
        if not fits:
            continue
        seen.add(key)
        candidates.append((query, windows))
        if len(candidates) >= target:
            break
    return candidates


def _negative_candidates(
    *,
    raw_binary: np.ndarray,
    raw_crs: CRS,
    raw_transform: Affine,
    base_longitude: float,
    base_latitude: float,
    query_size_m: float,
    max_offset_m: float,
    grids: Mapping[str, RasterGrid],
    rng: np.random.Generator,
    target: int,
    max_attempts: int,
) -> list[tuple[QueryGeometry, dict[str, dict[str, Any]]]]:
    candidates: list[
        tuple[QueryGeometry, dict[str, dict[str, Any]]]
    ] = []
    seen: set[tuple[int, int]] = set()
    for _ in range(max_attempts):
        east = float(rng.uniform(-max_offset_m, max_offset_m))
        north = float(rng.uniform(-max_offset_m, max_offset_m))
        key = (round(east), round(north))
        if key in seen:
            continue
        longitude, latitude = _offset_to_lonlat(
            base_longitude,
            base_latitude,
            east,
            north,
        )
        query = _canonical_query(
            raw_binary=raw_binary,
            raw_crs=raw_crs,
            raw_transform=raw_transform,
            base_longitude=base_longitude,
            base_latitude=base_latitude,
            longitude=longitude,
            latitude=latitude,
            query_size_m=query_size_m,
        )
        if query.label != 0:
            continue
        fits, windows = _fits_all_rasters(
            query,
            grids,
            query_size_m,
        )
        if not fits:
            continue
        seen.add(key)
        candidates.append((query, windows))
        if len(candidates) >= target:
            break
    candidates.sort(
        key=lambda item: math.hypot(
            item[0].east_m,
            item[0].north_m,
        ),
        reverse=True,
    )
    return candidates[:target]


def _sensor_presence_mask(sensors: Sequence[str]) -> str:
    sensor_set = set(sensors)
    return "".join(
        "1" if sensor in sensor_set else "0"
        for sensor in SENSOR_ORDER
    )


def _task_record(
    *,
    row: Mapping[str, Any],
    query: QueryGeometry,
    windows: Mapping[str, Mapping[str, Any]],
    sensors: Sequence[str],
    masks: Mapping[str, str],
    label_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    plume_id = _clean(row.get("plume_id", ""))
    event_group_id = _clean(row.get("event_group_id", ""))
    task_id = _stable_digest(
        args.seed,
        SCHEMA_VERSION,
        plume_id,
        query.label,
        label_index,
        f"{query.east_m:.6f}",
        f"{query.north_m:.6f}",
    )[:24]
    sensor_set = set(sensors)
    task: dict[str, Any] = {
        "task_id": task_id,
        "schema_version": SCHEMA_VERSION,
        "plume_id": plume_id,
        "event_group_id": event_group_id,
        "event_time": _clean(row.get("event_time", "")),
        "split": _clean(row.get("split", "")),
        "label": int(query.label),
        "label_index": int(label_index),
        "query_size_m": float(args.query_size_m),
        "query_center_longitude": query.longitude,
        "query_center_latitude": query.latitude,
        "offset_east_m": query.east_m,
        "offset_north_m": query.north_m,
        "offset_east_normalized": query.east_m / args.max_offset_m,
        "offset_north_normalized": query.north_m / args.max_offset_m,
        "offset_radius_m": math.hypot(query.east_m, query.north_m),
        "query_footprint_wgs84": json.dumps(
            query.footprint_wgs84
        ),
        "canonical_mask_path": str(
            args.cm_root / plume_id / "plume.tif"
        ),
        "canonical_positive_pixels": int(query.positive_pixels),
        "canonical_query_pixels": int(query.query_pixels),
        "canonical_overlap_fraction": (
            query.positive_pixels / max(1, query.query_pixels)
        ),
        "canonical_label_rule": (
            "label=1 iff Carbon Mapper binary footprint intersects "
            "the physical query square"
        ),
        "available_sensors": _sensor_combo(sensors),
        "sensor_presence_order": "|".join(SENSOR_ORDER),
        "sensor_presence_mask": _sensor_presence_mask(sensors),
        "num_available_sensors": int(len(sensors)),
        "timepoint_order": "|".join(TIMEPOINTS),
        "s5p_query_mapping_policy": (
            "nearest valid geolocation pixel to task query_center_latitude/"
            "longitude, recomputed independently per timepoint"
            if "s5p" in sensor_set
            else ""
        ),
        "s5p_fixed_plume_center_forbidden": bool("s5p" in sensor_set),
        "execution_status": "planned",
    }
    for sensor in SENSOR_ORDER:
        present = sensor in sensor_set
        task[f"has_{sensor}"] = present
        task[f"{sensor}_source_time_presence_mask"] = (
            "111111" if present else "000000"
        )
        for timepoint in TIMEPOINTS:
            task[f"{sensor}_{timepoint}_path"] = (
                _clean(row.get(f"{sensor}_{timepoint}_path", ""))
                if present
                else ""
            )
        if sensor in RASTER_SENSORS:
            task[f"{sensor}_mask_path"] = (
                masks.get(sensor, "") if present else ""
            )
            window = windows.get(sensor, {})
            for key in (
                "query_col_t0",
                "query_row_t0",
                "crop_x0",
                "crop_y0",
                "crop_width_px",
                "crop_height_px",
                "grid_crs",
                "grid_transform",
                "grid_reference_path",
                "grid_source",
                "pixel_size_x_m",
                "pixel_size_y_m",
            ):
                task[f"{sensor}_{key}"] = (
                    window.get(key, "") if present else ""
                )
        if sensor == "s5p":
            for timepoint in TIMEPOINTS:
                for suffix in (
                    "qc_center_iy",
                    "qc_center_ix",
                    "qc_center_distance_km",
                    "qc_patch_missing_ratio",
                ):
                    task[f"s5p_source_{timepoint}_{suffix}"] = (
                        _clean(
                            row.get(
                                f"s5p_{timepoint}_{suffix}",
                                "",
                            )
                        )
                        if present
                        else ""
                    )
    return task


def _plan_plume(
    row: Mapping[str, Any],
    args: argparse.Namespace,
) -> PlumePlan:
    plume_id = _clean(row.get("plume_id", ""))
    requested_sensors = tuple(row["_ready_sensors"])
    masks = dict(row["_sensor_masks"])
    dropped: list[str] = []
    grids: dict[str, RasterGrid] = {}
    for sensor in requested_sensors:
        if not all(
            _file_ok(row.get(f"{sensor}_{timepoint}_path", ""))
            for timepoint in TIMEPOINTS
        ):
            dropped.append(sensor)
    for sensor in RASTER_SENSORS:
        if sensor not in requested_sensors or sensor in dropped:
            continue
        try:
            grids[sensor] = _real_grid(
                sensor,
                Path(row[f"{sensor}_t0_path"]),
                grid_source="t0_image",
            )
        except Exception:
            try:
                grids[sensor] = _real_grid(
                    sensor,
                    Path(masks[sensor]),
                    grid_source="canonical_mask_fallback",
                )
            except Exception:
                dropped.append(sensor)
    sensors = tuple(
        sensor
        for sensor in requested_sensors
        if sensor not in dropped
    )
    if not sensors:
        return PlumePlan(
            plume_id=plume_id,
            tasks=[],
            status="fail",
            reason="no_sensor_after_grid_validation",
            sensors=(),
            requested_positive=args.n_pos,
            requested_negative=args.n_neg,
            made_positive=0,
            made_negative=0,
            dropped_sensors=tuple(dropped),
        )
    try:
        base_latitude = float(row["plume_latitude"])
        base_longitude = float(row["plume_longitude"])
    except (KeyError, TypeError, ValueError):
        return PlumePlan(
            plume_id=plume_id,
            tasks=[],
            status="fail",
            reason="invalid_plume_coordinates",
            sensors=sensors,
            requested_positive=args.n_pos,
            requested_negative=args.n_neg,
            made_positive=0,
            made_negative=0,
            dropped_sensors=tuple(dropped),
        )
    raw_path = args.cm_root / plume_id / "plume.tif"
    try:
        with rasterio.open(raw_path) as raw:
            if raw.crs is None:
                raise RuntimeError("raw_mask_missing_crs")
            if raw.transform.almost_equals(Affine.identity()):
                raise RuntimeError("raw_mask_identity_transform")
            raw_binary = _raw_binary(raw)
            raw_crs = raw.crs
            raw_transform = raw.transform
    except Exception as exc:
        return PlumePlan(
            plume_id=plume_id,
            tasks=[],
            status="fail",
            reason=f"raw_mask_invalid:{type(exc).__name__}:{str(exc)[:160]}",
            sensors=sensors,
            requested_positive=args.n_pos,
            requested_negative=args.n_neg,
            made_positive=0,
            made_negative=0,
            dropped_sensors=tuple(dropped),
        )
    if not np.any(raw_binary):
        return PlumePlan(
            plume_id=plume_id,
            tasks=[],
            status="fail",
            reason="raw_mask_empty",
            sensors=sensors,
            requested_positive=args.n_pos,
            requested_negative=args.n_neg,
            made_positive=0,
            made_negative=0,
            dropped_sensors=tuple(dropped),
        )

    rng = _stable_rng(
        args.seed,
        _clean(row.get("event_group_id", "")),
        plume_id,
    )
    positives = _positive_candidates(
        raw_binary=raw_binary,
        raw_crs=raw_crs,
        raw_transform=raw_transform,
        base_longitude=base_longitude,
        base_latitude=base_latitude,
        query_size_m=args.query_size_m,
        max_offset_m=args.max_offset_m,
        grids=grids,
        rng=rng,
        target=args.n_pos,
    )
    negatives = _negative_candidates(
        raw_binary=raw_binary,
        raw_crs=raw_crs,
        raw_transform=raw_transform,
        base_longitude=base_longitude,
        base_latitude=base_latitude,
        query_size_m=args.query_size_m,
        max_offset_m=args.max_offset_m,
        grids=grids,
        rng=rng,
        target=args.n_neg,
        max_attempts=args.max_negative_attempts,
    )
    tasks: list[dict[str, Any]] = []
    for label, candidates in ((1, positives), (0, negatives)):
        for label_index, (query, windows) in enumerate(candidates):
            tasks.append(
                _task_record(
                    row=row,
                    query=query,
                    windows=windows,
                    sensors=sensors,
                    masks=masks,
                    label_index=label_index,
                    args=args,
                )
            )
    status = (
        "ok"
        if len(positives) == args.n_pos
        and len(negatives) == args.n_neg
        else "short"
    )
    reason = (
        "ok"
        if status == "ok"
        else (
            f"made_pos={len(positives)}/{args.n_pos};"
            f"made_neg={len(negatives)}/{args.n_neg}"
        )
    )
    return PlumePlan(
        plume_id=plume_id,
        tasks=tasks,
        status=status,
        reason=reason,
        sensors=sensors,
        requested_positive=args.n_pos,
        requested_negative=args.n_neg,
        made_positive=len(positives),
        made_negative=len(negatives),
        dropped_sensors=tuple(dropped),
    )


def _to_chw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        raise RuntimeError(f"unexpected_array_shape:{array.shape}")
    if array.shape[-2:] == (512, 512):
        return array
    if array.shape[:2] == (512, 512):
        return np.moveaxis(array, -1, 0)
    raise RuntimeError(f"not_spatial_512:{array.shape}")


def _read_raster_smoke_crop(
    sensor: str,
    path: Path,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    expected_crs: str,
    expected_transform: Affine,
) -> tuple[int, tuple[int, int, int]]:
    if sensor == "s2":
        array = _to_chw(tifffile.imread(path))
        crop = array[:, y0 : y0 + height, x0 : x0 + width]
    else:
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise RuntimeError(f"source_missing_crs:{path}")
            if dataset.crs != CRS.from_string(expected_crs):
                raise RuntimeError(
                    f"source_crs_mismatch:{dataset.crs}:"
                    f"{expected_crs}:{path}"
                )
            if not dataset.transform.almost_equals(expected_transform):
                raise RuntimeError(
                    f"source_transform_mismatch:{path}"
                )
            crop = dataset.read(
                window=Window(x0, y0, width, height)
            )
    if crop.shape[-2:] != (height, width):
        raise RuntimeError(
            f"crop_shape={crop.shape},expected={height}x{width}"
        )
    return int(np.count_nonzero(np.isfinite(crop))), tuple(crop.shape)


def _s5p_nearest_query_pixel(
    path: Path,
    *,
    query_longitude: float,
    query_latitude: float,
) -> dict[str, Any]:
    if Dataset is None and h5py is None:
        raise RuntimeError("no_netcdf4_or_h5py_backend")
    with S5P_IO_LOCK:
        if Dataset is not None:
            dataset = Dataset(str(path), "r")
            try:
                product = dataset.groups.get("PRODUCT", dataset)
                latitude_variable = product.variables.get("latitude")
                longitude_variable = product.variables.get("longitude")
                if (
                    latitude_variable is None
                    or longitude_variable is None
                ):
                    raise RuntimeError("missing_latitude_longitude")
                latitudes = np.ma.filled(
                    latitude_variable[:],
                    np.nan,
                ).astype(np.float64)
                longitudes = np.ma.filled(
                    longitude_variable[:],
                    np.nan,
                ).astype(np.float64)
                variable_names = product.variables
                backend = "netCDF4"
                close = dataset.close
            except Exception:
                dataset.close()
                raise
        else:
            dataset = h5py.File(str(path), "r")
            try:
                product = dataset.get("PRODUCT", dataset)
                latitude_variable = product.get("latitude")
                longitude_variable = product.get("longitude")
                if (
                    latitude_variable is None
                    or longitude_variable is None
                ):
                    raise RuntimeError("missing_latitude_longitude")
                latitudes = _h5_numeric(latitude_variable)
                longitudes = _h5_numeric(longitude_variable)
                variable_names = product
                backend = "h5py"
                close = dataset.close
            except Exception:
                dataset.close()
                raise
        try:
            latitudes = np.squeeze(latitudes)
            longitudes = np.squeeze(longitudes)
            if latitudes.ndim != 2 or longitudes.shape != latitudes.shape:
                raise RuntimeError(
                    f"bad_geolocation_shape:{latitudes.shape}:"
                    f"{longitudes.shape}"
                )
            distance_km = _haversine_grid_km(
                latitudes,
                longitudes,
                query_latitude=query_latitude,
                query_longitude=query_longitude,
            )
            flat_index = int(np.argmin(distance_km))
            iy, ix = np.unravel_index(flat_index, latitudes.shape)
            if not math.isfinite(float(distance_km[iy, ix])):
                raise RuntimeError("no_valid_geolocation")
            if (
                iy < 1
                or ix < 1
                or iy + 1 >= latitudes.shape[0]
                or ix + 1 >= latitudes.shape[1]
            ):
                raise RuntimeError(f"nearest_pixel_on_edge:{iy},{ix}")
            ch4_name = next(
                (
                    name
                    for name in CH4_CANDIDATES
                    if name in variable_names
                ),
                None,
            )
            if ch4_name is None:
                raise RuntimeError("missing_ch4_variable")
            variable = variable_names[ch4_name]
            if variable.ndim == 3:
                selection = (
                    0,
                    slice(iy - 1, iy + 2),
                    slice(ix - 1, ix + 2),
                )
            elif variable.ndim == 2:
                selection = (
                    slice(iy - 1, iy + 2),
                    slice(ix - 1, ix + 2),
                )
            else:
                raise RuntimeError(f"bad_ch4_dims:{variable.shape}")
            if backend == "netCDF4":
                patch_array = np.ma.filled(
                    variable[selection],
                    np.nan,
                ).astype(np.float32)
            else:
                patch_array = _h5_numeric(
                    variable,
                    selection=selection,
                ).astype(np.float32)
            missing_ratio = float(
                1.0
                - np.count_nonzero(np.isfinite(patch_array))
                / patch_array.size
            )
            return {
                "iy": int(iy),
                "ix": int(ix),
                "distance_km": float(distance_km[iy, ix]),
                "patch_missing_ratio": missing_ratio,
                "ch4_variable": ch4_name,
                "geolocation_shape": list(latitudes.shape),
                "io_backend": backend,
            }
        finally:
            close()


def _h5_numeric(
    variable: Any,
    *,
    selection: Any = None,
) -> np.ndarray:
    array = np.asarray(
        variable[...] if selection is None else variable[selection],
        dtype=np.float64,
    )
    for attribute in ("_FillValue", "missing_value"):
        if attribute not in variable.attrs:
            continue
        for value in np.asarray(variable.attrs[attribute]).reshape(-1):
            array[array == float(value)] = np.nan
    for attribute, comparison in (
        ("valid_min", np.less),
        ("valid_max", np.greater),
    ):
        if attribute in variable.attrs:
            threshold = float(
                np.asarray(variable.attrs[attribute]).reshape(-1)[0]
            )
            array[comparison(array, threshold)] = np.nan
    if "scale_factor" in variable.attrs:
        array *= float(
            np.asarray(variable.attrs["scale_factor"]).reshape(-1)[0]
        )
    if "add_offset" in variable.attrs:
        array += float(
            np.asarray(variable.attrs["add_offset"]).reshape(-1)[0]
        )
    return array


def _haversine_grid_km(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    query_latitude: float,
    query_longitude: float,
) -> np.ndarray:
    valid = np.isfinite(latitudes) & np.isfinite(longitudes)
    latitude_radians = np.deg2rad(latitudes)
    query_latitude_radians = math.radians(query_latitude)
    delta_latitude = latitude_radians - query_latitude_radians
    delta_longitude = np.deg2rad(
        (longitudes - query_longitude + 180.0) % 360.0 - 180.0
    )
    haversine = (
        np.sin(delta_latitude / 2.0) ** 2
        + math.cos(query_latitude_radians)
        * np.cos(latitude_radians)
        * np.sin(delta_longitude / 2.0) ** 2
    )
    distance_km = (
        2.0
        * 6371.0088
        * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    )
    distance_km[~valid] = np.inf
    return distance_km


def _smoke_validate_task(task: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_id": task["task_id"],
        "plume_id": task["plume_id"],
        "label": int(task["label"]),
        "available_sensors": task["available_sensors"],
        "status": "ok",
        "reason": "ok",
        "raster_time_shapes": "",
        "raster_valid_time_masks": "",
        "sensor_mask_positive_pixels": "",
        "s5p_query_pixels": "",
    }
    raster_shapes: dict[str, dict[str, Any]] = {}
    raster_valid_time_masks: dict[str, str] = {}
    sensor_mask_counts: dict[str, int] = {}
    warnings_found: list[str] = []
    try:
        sensors = set(_clean(task["available_sensors"]).split("|"))
        for sensor in RASTER_SENSORS:
            if sensor not in sensors:
                continue
            x0 = int(task[f"{sensor}_crop_x0"])
            y0 = int(task[f"{sensor}_crop_y0"])
            width = int(task[f"{sensor}_crop_width_px"])
            height = int(task[f"{sensor}_crop_height_px"])
            expected_crs = _clean(task[f"{sensor}_grid_crs"])
            expected_transform = Affine(
                *json.loads(task[f"{sensor}_grid_transform"])
            )
            time_shapes: dict[str, Any] = {}
            valid_time_bits: list[str] = []
            for timepoint in TIMEPOINTS:
                path = Path(task[f"{sensor}_{timepoint}_path"])
                finite_count, shape = _read_raster_smoke_crop(
                    sensor,
                    path,
                    x0=x0,
                    y0=y0,
                    width=width,
                    height=height,
                    expected_crs=expected_crs,
                    expected_transform=expected_transform,
                )
                time_shapes[timepoint] = {
                    "shape": list(shape),
                    "finite_values": finite_count,
                }
                valid_time_bits.append("1" if finite_count > 0 else "0")
                if finite_count <= 0:
                    warnings_found.append(
                        f"zero_finite:{sensor}:{timepoint}"
                    )
            raster_shapes[sensor] = time_shapes
            raster_valid_time_masks[sensor] = "".join(valid_time_bits)
            mask_path = Path(task[f"{sensor}_mask_path"])
            mask_grid = _real_grid(
                sensor,
                mask_path,
                grid_source="diagnostic_mask",
            )
            mask_window = _query_window(
                float(task["query_center_longitude"]),
                float(task["query_center_latitude"]),
                mask_grid,
                float(task["query_size_m"]),
            )
            if mask_window is None:
                raise RuntimeError(
                    f"query_outside_diagnostic_mask:{sensor}"
                )
            with rasterio.open(mask_path) as mask_file:
                mask_crop = mask_file.read(
                    1,
                    window=Window(
                        int(mask_window["crop_x0"]),
                        int(mask_window["crop_y0"]),
                        int(mask_window["crop_width_px"]),
                        int(mask_window["crop_height_px"]),
                    ),
                )
            sensor_mask_counts[sensor] = int(
                np.count_nonzero(mask_crop)
            )
        result["raster_time_shapes"] = json.dumps(
            raster_shapes,
            sort_keys=True,
        )
        result["raster_valid_time_masks"] = json.dumps(
            raster_valid_time_masks,
            sort_keys=True,
        )
        result["sensor_mask_positive_pixels"] = json.dumps(
            sensor_mask_counts,
            sort_keys=True,
        )

        if "s5p" in sensors:
            mappings: dict[str, Any] = {}
            query_longitude = float(task["query_center_longitude"])
            query_latitude = float(task["query_center_latitude"])
            for timepoint in TIMEPOINTS:
                path = Path(task[f"s5p_{timepoint}_path"])
                mappings[timepoint] = _s5p_nearest_query_pixel(
                    path,
                    query_longitude=query_longitude,
                    query_latitude=query_latitude,
                )
                if mappings[timepoint]["patch_missing_ratio"] >= 1.0:
                    warnings_found.append(
                        f"all_missing:s5p:{timepoint}"
                    )
            result["s5p_query_pixels"] = json.dumps(
                mappings,
                sort_keys=True,
            )
        if warnings_found:
            result["status"] = "warning"
            result["reason"] = ";".join(warnings_found)
    except Exception as exc:
        result["status"] = "fail"
        result["reason"] = (
            f"{type(exc).__name__}:{str(exc)[:300]}"
        )
    return result


def _select_smoke_tasks(
    tasks: pd.DataFrame,
    maximum: int,
) -> pd.DataFrame:
    if maximum <= 0 or tasks.empty:
        return tasks.head(0)
    selected: list[pd.DataFrame] = []
    used_ids: set[str] = set()
    for _, group in tasks.groupby(
        "available_sensors",
        observed=True,
        sort=True,
    ):
        row = group.sort_values(
            ["label", "task_id"],
            ascending=[False, True],
            kind="mergesort",
        ).head(1)
        selected.append(row)
        used_ids.update(row["task_id"].astype(str))
        if sum(len(item) for item in selected) >= maximum:
            break
    selected_frame = (
        pd.concat(selected, ignore_index=True)
        if selected
        else tasks.head(0)
    )
    if len(selected_frame) < maximum:
        remaining = tasks[
            ~tasks["task_id"].astype(str).isin(used_ids)
        ].sort_values("task_id", kind="mergesort")
        selected_frame = pd.concat(
            [
                selected_frame,
                remaining.head(maximum - len(selected_frame)),
            ],
            ignore_index=True,
        )
    return selected_frame.head(maximum)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _counter(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(_clean(value) or "<empty>" for value in values)
    return {key: int(counts[key]) for key in sorted(counts)}


def _schema(task_columns: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "One row is one canonical geographic query consumed by every "
            "present sensor and all six timepoints."
        ),
        "sensor_order": list(SENSOR_ORDER),
        "timepoint_order": list(TIMEPOINTS),
        "global_invariants": [
            "split is inherited from the event_group_id-wide global split",
            "one WGS84 query centre and east/north offset per task",
            "all present sensors consume exactly that query centre",
            "partial sensors are represented by sensor_presence_mask",
            "all present sensors require six source files",
            "query-level finite-data time masks are explicit and never "
            "silently inferred from file presence",
            "canonical Carbon Mapper geometry is the only label authority",
            "sensor masks are grid/diagnostic data and may disagree by resolution",
            "S5P nearest pixel is recomputed per query and per timepoint",
            "fixed plume-centre S5P indices are forbidden",
        ],
        "label_definition": (
            "label=1 iff the physical query square has at least one "
            "all-touched overlap pixel with the binary Carbon Mapper plume "
            "footprint; otherwise label=0"
        ),
        "offset_definition": (
            "offset_east_m/offset_north_m are WGS84 geodesic offsets from "
            "the catalogue plume point to query_center"
        ),
        "raster_execution_contract": (
            "map query_center WGS84 to the real t0 image grid when present; "
            "use the canonical repaired mask grid only for identity-array "
            "products such as S2; apply the stored window to all six arrays; "
            "executor must revalidate shapes/bounds and record a six-bit "
            "finite-data mask for each sensor"
        ),
        "s5p_execution_contract": (
            "for every task and every timepoint, read that NetCDF's own "
            "latitude/longitude arrays and find the nearest valid pixel to "
            "query_center; never reuse source qc_center indices"
        ),
        "columns": list(task_columns),
        "materialized_crop_files": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.query_size_m <= 0:
        raise ValueError("--query-size-m must be positive")
    if args.n_pos < 0 or args.n_neg < 0:
        raise ValueError("--n-pos/--n-neg cannot be negative")
    if args.n_pos + args.n_neg <= 0:
        raise ValueError("at least one query per plume is required")
    if args.max_offset_m <= args.query_size_m / 2:
        raise ValueError(
            "--max-offset-m must exceed half the query size"
        )
    if args.max_negative_attempts < args.n_neg:
        raise ValueError("--max-negative-attempts is too small")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    frame, input_audit = _load_wide(args)
    rows = frame.to_dict("records")
    print(
        f"[crop-plan] mode="
        f"{'write-task-manifest' if args.write_task_manifest else 'dry-run'} "
        f"plumes={len(rows)} workers={args.workers}",
        flush=True,
    )
    plans: list[PlumePlan] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, plan in enumerate(
            executor.map(lambda row: _plan_plume(row, args), rows),
            start=1,
        ):
            plans.append(plan)
            if index % 100 == 0 or index == len(rows):
                print(
                    f"[crop-plan] planned={index}/{len(rows)}",
                    flush=True,
                )

    task_records = [
        task for plan in plans for task in plan.tasks
    ]
    tasks = pd.DataFrame(task_records)
    if tasks.empty:
        raise RuntimeError("no crop tasks could be planned")
    tasks = tasks.sort_values(
        ["event_time", "event_group_id", "plume_id", "label", "label_index"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    task_group_splits = tasks.groupby(
        "event_group_id", observed=True
    )["split"].nunique()
    if task_group_splits.gt(1).any():
        raise RuntimeError("planned tasks introduced event split leakage")

    schema = _schema(tasks.columns)
    _atomic_write_json(schema, args.schema_json)
    if args.write_task_manifest:
        _atomic_write_csv(tasks, args.task_csv)

    smoke = pd.DataFrame()
    smoke_path = args.smoke_validation_csv
    if smoke_path is None:
        smoke_path = args.task_csv.with_name(
            f"{args.task_csv.stem}.smoke_validation.csv"
        )
    if args.smoke_validate_tasks:
        selected_smoke = _select_smoke_tasks(
            tasks,
            args.smoke_validate_tasks,
        )
        smoke_records = [
            _smoke_validate_task(row)
            for row in selected_smoke.to_dict("records")
        ]
        smoke = pd.DataFrame(smoke_records)
        _atomic_write_csv(smoke, smoke_path)

    plan_status = pd.DataFrame(
        [
            {
                "plume_id": plan.plume_id,
                "status": plan.status,
                "reason": plan.reason,
                "available_sensors": _sensor_combo(plan.sensors),
                "dropped_sensors": _sensor_combo(
                    plan.dropped_sensors
                ),
                "made_positive": plan.made_positive,
                "made_negative": plan.made_negative,
            }
            for plan in plans
        ]
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "mode": (
            "task_manifest_written"
            if args.write_task_manifest
            else "dry_run"
        ),
        "input": input_audit,
        "parameters": {
            "query_size_m": args.query_size_m,
            "n_pos": args.n_pos,
            "n_neg": args.n_neg,
            "max_offset_m": args.max_offset_m,
            "max_negative_attempts": args.max_negative_attempts,
            "seed": args.seed,
            "workers": args.workers,
        },
        "old_pipeline_rejections": [
            "three-timepoint-only SENSOR_COLS",
            "global shared RNG across worker threads",
            "centre-box labels instead of canonical mask geometry",
            "requiring all raster masks to agree on the label",
            "S5P plume nearest_iy/nearest_ix reused for every query",
            "S5P t0 pixel coordinates reused across historical timepoints",
        ],
        "invariants": schema["global_invariants"],
        "results": {
            "selected_plumes": int(len(plans)),
            "plume_plan_status_counts": _counter(
                plan_status["status"]
            ),
            "plume_plan_reason_counts": _counter(
                plan_status["reason"]
            ),
            "dropped_sensor_counts": _counter(
                plan_status.loc[
                    plan_status["dropped_sensors"].ne(""),
                    "dropped_sensors",
                ]
            ),
            "tasks": int(len(tasks)),
            "tasks_by_label": {
                str(int(key)): int(value)
                for key, value in tasks["label"]
                .value_counts()
                .sort_index()
                .items()
            },
            "tasks_by_split": _counter(tasks["split"]),
            "tasks_by_sensor_combination": _counter(
                tasks["available_sensors"]
            ),
            "event_groups": int(tasks["event_group_id"].nunique()),
            "event_groups_spanning_splits": 0,
            "sensor_presence_masks": _counter(
                tasks["sensor_presence_mask"]
            ),
            "s5p_tasks": int(tasks["has_s5p"].sum()),
            "s5p_fixed_plume_center_forbidden_true": int(
                tasks["s5p_fixed_plume_center_forbidden"].sum()
            ),
        },
        "smoke_validation": {
            "requested_tasks": int(args.smoke_validate_tasks),
            "rows": int(len(smoke)),
            "status_counts": (
                _counter(smoke["status"]) if not smoke.empty else {}
            ),
            "reason_counts": (
                _counter(smoke["reason"]) if not smoke.empty else {}
            ),
            "csv": str(smoke_path) if not smoke.empty else "",
        },
        "outputs": {
            "task_csv": (
                str(args.task_csv)
                if args.write_task_manifest
                else ""
            ),
            "schema_json": str(args.schema_json),
            "audit_json": str(args.audit_json),
            "materialized_crop_files": False,
        },
    }
    _atomic_write_json(audit, args.audit_json)
    summary = {
        "mode": audit["mode"],
        "selected_plumes": len(plans),
        "tasks": len(tasks),
        "tasks_by_label": audit["results"]["tasks_by_label"],
        "tasks_by_sensor_combination": audit["results"][
            "tasks_by_sensor_combination"
        ],
        "plume_plan_status_counts": audit["results"][
            "plume_plan_status_counts"
        ],
        "smoke_status_counts": audit["smoke_validation"][
            "status_counts"
        ],
        "task_csv": audit["outputs"]["task_csv"],
        "schema_json": str(args.schema_json),
        "audit_json": str(args.audit_json),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return audit


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
