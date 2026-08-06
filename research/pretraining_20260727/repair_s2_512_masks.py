#!/usr/bin/env python3
"""Repair canonical S2 masks on the exact point-centred 512 grid.

The materialized S2 v14 TIFFs are tifffile arrays.  Rasterio sees them as a
single-band, identity-transform raster, so their rasterio metadata is never
used to invent a grid.  Instead, this script resolves a trusted point-centred
provenance reference (normally ``t0_v8_source_path``) and validates its
``.georef.json`` sidecar before reprojecting the local Carbon Mapper plume mask.

Safety defaults:

* no mask is written unless ``--write`` is given;
* only rows with ``has_s2`` in the unified wide manifest are selected;
* a sidecar/reference must be 512x512, projected, non-identity, centred on the
  catalogue plume point, and geographically overlap the raw plume mask;
* an empty reprojected mask is a failure, never a successful all-zero mask;
* output files are written atomically and never overwrite without
  ``--overwrite``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import tempfile
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import rasterio
import tifffile
from affine import Affine
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import array_bounds, xy
from rasterio.warp import Resampling, reproject, transform, transform_bounds
from rasterio.windows import Window, bounds as window_bounds


warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


SCHEMA_VERSION = "methanefuse_s2_canonical_mask_repair_v1"
WINDOW_SIZE = 512
DEFAULT_UNIFIED_MANIFEST = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/manifests/"
    "multisensor_6time_512_wide.csv"
)
DEFAULT_S2_SOURCE_MANIFEST = Path(
    "/home/yuyao/panopticon/Upgraded_dataset/"
    "s2_historical_point_center_v14/s2_v14_all_512.csv"
)
DEFAULT_CM_ROOTS = (
    Path("/mnt/engg-niulab/yuyao/sensors_raw_data/CM"),
)
DEFAULT_OUTPUT_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/"
    "cache/s2_512_masks"
)
DEFAULT_AUDIT_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/manifests"
)

# These columns are provenance links to point-centred products.  Bounds-native
# t0_input_path/t0_raw_path are intentionally absent because they are not the
# v14 point-centred target grid.
TRUSTED_REFERENCE_COLUMNS = (
    ("t0_v8_source_path", "exact_v3_source"),
    ("t0_original_raw_path", "exact_v8_original"),
    ("t0_v8_target_path", "exact_v8_target"),
    ("t0_v8_direct_path", "exact_v8_direct"),
)

S2_SOURCE_OPTIONAL_COLUMNS = (
    "event_group_id",
    "event_time",
    "plume_latitude",
    "plume_longitude",
    "plume_bounds",
    "plume_tif",
    "center_mode",
    "image_geometry_version",
    "s2_0_std_512_source_class",
    *(column for column, _ in TRUSTED_REFERENCE_COLUMNS),
)


@dataclass(frozen=True)
class Grid:
    crs: CRS
    transform: Affine
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    bounds_wgs84: tuple[float, float, float, float]
    point_col: float
    point_row: float
    center_offset_pixels: float


@dataclass
class Resolution:
    record: dict[str, Any]
    row: dict[str, Any]
    grid: Grid | None = None
    raw_mask_path: Path | None = None
    reference_path: Path | None = None
    sidecar_path: Path | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unified-manifest",
        type=Path,
        default=DEFAULT_UNIFIED_MANIFEST,
    )
    parser.add_argument(
        "--s2-source-manifest",
        type=Path,
        default=DEFAULT_S2_SOURCE_MANIFEST,
    )
    parser.add_argument(
        "--cm-root",
        type=Path,
        action="append",
        default=None,
        help="Local Carbon Mapper root; may be repeated.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--resolution-csv",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "s2_mask_repair_resolution.csv",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "s2_mask_repair.audit.json",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write masks. Without this flag the run is audit-only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only after all source checks pass.",
    )
    parser.add_argument(
        "--subpixel-point-fallback",
        action="store_true",
        help=(
            "If nearest-neighbour reprojection is empty, place each positive "
            "raw-pixel centre into its containing target cell. This is an "
            "explicit minimum-cell fallback; it does not dilate or invent "
            "polygon coverage."
        ),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--max-center-offset-pixels",
        type=float,
        default=1.5,
        help="Maximum plume-point distance from the expected pixel centre.",
    )
    parser.add_argument(
        "--min-pixel-size",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--max-pixel-size",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--plume-id",
        action="append",
        default=[],
        help="Exact smoke/debug selection; repeat for multiple plume IDs.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Deterministic hash sample after has_s2 filtering.",
    )
    parser.add_argument("--seed", type=int, default=20260727)
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


def _safe_component(value: Any) -> str:
    text = _clean(value)
    safe = "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in text
    )
    return safe[:220] if safe else "missing"


def _file_ok(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _require_unique(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    empty = frame["plume_id"].map(_clean).eq("")
    if empty.any():
        raise ValueError(f"{name} contains {int(empty.sum())} empty plume_id")
    duplicate = frame["plume_id"].astype(str).duplicated(keep=False)
    duplicate_ids = sorted(
        frame.loc[duplicate, "plume_id"].astype(str).unique().tolist()
    )
    if duplicate_ids:
        raise ValueError(
            f"{name} has {len(duplicate_ids)} duplicate plume_id values: "
            f"{duplicate_ids[:20]}"
        )
    return {
        "rows": int(len(frame)),
        "unique_plume_ids": int(frame["plume_id"].nunique()),
        "duplicate_plume_ids": 0,
    }


def _load_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    for name, path in (
        ("unified manifest", args.unified_manifest),
        ("S2 source manifest", args.s2_source_manifest),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    unified_header = _read_header(args.unified_manifest)
    unified_required = {
        "plume_id",
        "has_s2",
        "s2_t0_path",
        "event_group_id",
        "event_time",
        "plume_latitude",
        "plume_longitude",
        "plume_bounds",
    }
    missing = sorted(unified_required - set(unified_header))
    if missing:
        raise ValueError(
            f"unified manifest is missing required columns: {missing}"
        )
    unified_optional = {"split", "plume_tif"}
    unified_columns = [
        column
        for column in unified_header
        if column in unified_required | unified_optional
    ]
    unified = pd.read_csv(
        args.unified_manifest,
        usecols=unified_columns,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    unified_audit = _require_unique(unified, "unified manifest")
    has_s2 = unified["has_s2"].map(_parse_bool)
    all_rows = len(unified)
    unified = unified.loc[has_s2].copy()
    has_s2_rows = len(unified)

    requested = set(args.plume_id)
    missing_requested: list[str] = []
    if requested:
        present = set(unified["plume_id"].astype(str))
        missing_requested = sorted(requested - present)
        unified = unified[unified["plume_id"].isin(requested)].copy()
    if args.sample:
        if requested:
            raise ValueError("--sample and --plume-id cannot be used together")
        if args.sample < 1:
            raise ValueError("--sample must be positive")
        unified["_sample_rank"] = unified["plume_id"].astype(str).map(
            lambda plume_id: hashlib.sha256(
                f"{args.seed}|{plume_id}".encode("utf-8")
            ).hexdigest()
        )
        unified = (
            unified.sort_values(
                ["_sample_rank", "plume_id"],
                kind="mergesort",
            )
            .head(args.sample)
            .drop(columns="_sample_rank")
            .copy()
        )
    if unified.empty:
        raise ValueError("no has_s2 rows remain after selection")

    source_header = _read_header(args.s2_source_manifest)
    source_required = {"plume_id"}
    missing = sorted(source_required - set(source_header))
    if missing:
        raise ValueError(f"S2 source manifest is missing columns: {missing}")
    source_columns = [
        column
        for column in source_header
        if column in source_required | set(S2_SOURCE_OPTIONAL_COLUMNS)
    ]
    if "t0_v8_source_path" not in source_columns:
        raise ValueError(
            "S2 source manifest has no t0_v8_source_path provenance column"
        )
    source = pd.read_csv(
        args.s2_source_manifest,
        usecols=source_columns,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    source_audit = _require_unique(source, "S2 source manifest")

    source = source.rename(
        columns={
            column: f"source_{column}"
            for column in source.columns
            if column != "plume_id"
        }
    )
    joined = unified.merge(
        source,
        on="plume_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    joined = joined.sort_values("plume_id", kind="mergesort").reset_index(
        drop=True
    )
    source_missing = int(joined["_merge"].ne("both").sum())
    joined = joined.drop(columns="_merge")

    audit = {
        "unified": {
            **_fingerprint(args.unified_manifest),
            **unified_audit,
            "rows_before_has_s2_filter": int(all_rows),
            "has_s2_rows": int(has_s2_rows),
        },
        "s2_source": {
            **_fingerprint(args.s2_source_manifest),
            **source_audit,
            "columns_read": source_columns,
        },
        "selection": {
            "rows": int(len(joined)),
            "requested_plume_ids": sorted(requested),
            "requested_not_in_has_s2": missing_requested,
            "sample": int(args.sample),
            "seed": int(args.seed),
            "rows_missing_s2_source_join": source_missing,
        },
    }
    return joined, audit


def _storage_shape(path: Path) -> tuple[tuple[int, ...], str, bool]:
    with tifffile.TiffFile(path) as dataset:
        series = dataset.series[0]
        shape = tuple(int(value) for value in series.shape)
        dtype = str(series.dtype)
    spatial_ok = False
    if shape == (WINDOW_SIZE, WINDOW_SIZE):
        spatial_ok = True
    elif len(shape) == 3:
        spatial_ok = (
            shape[-2:] == (WINDOW_SIZE, WINDOW_SIZE)
            or shape[:2] == (WINDOW_SIZE, WINDOW_SIZE)
        )
    return shape, dtype, spatial_ok


def _has_real_georef(
    crs: CRS | None,
    affine: Affine,
) -> bool:
    return bool(
        crs is not None
        and not affine.almost_equals(Affine.identity())
        and np.all(np.isfinite(np.asarray(affine)[:6]))
    )


def _inspect_target(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "target_s2_path": str(path),
        "target_exists": False,
        "target_storage_shape": "",
        "target_dtype": "",
        "target_spatial_shape_ok": False,
        "target_rasterio_count": "",
        "target_rasterio_shape": "",
        "target_rasterio_crs": "",
        "target_rasterio_transform": "",
        "target_rasterio_has_real_georef": False,
    }
    if not _file_ok(path):
        return result
    result["target_exists"] = True
    shape, dtype, spatial_ok = _storage_shape(path)
    result["target_storage_shape"] = json.dumps(shape)
    result["target_dtype"] = dtype
    result["target_spatial_shape_ok"] = bool(spatial_ok)
    with rasterio.open(path) as dataset:
        result["target_rasterio_count"] = int(dataset.count)
        result["target_rasterio_shape"] = (
            f"{int(dataset.height)}x{int(dataset.width)}"
        )
        result["target_rasterio_crs"] = (
            dataset.crs.to_string() if dataset.crs else ""
        )
        result["target_rasterio_transform"] = json.dumps(
            [float(value) for value in dataset.transform[:6]]
        )
        result["target_rasterio_has_real_georef"] = _has_real_georef(
            dataset.crs,
            dataset.transform,
        )
    return result


def _parse_plume_bounds(value: Any) -> tuple[float, float, float, float] | None:
    text = _clean(value)
    if not text:
        return None
    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        return None
    try:
        bounds = tuple(float(item) for item in parsed)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bounds):
        return None
    left, bottom, right, top = bounds
    if left >= right or bottom >= top:
        return None
    return left, bottom, right, top


def _normalize_bounds(
    bounds: Sequence[float],
) -> tuple[float, float, float, float]:
    left, bottom, right, top = (float(value) for value in bounds)
    return (
        min(left, right),
        min(bottom, top),
        max(left, right),
        max(bottom, top),
    )


def _bounds_overlap(
    first: Sequence[float],
    second: Sequence[float],
) -> bool:
    a_left, a_bottom, a_right, a_top = _normalize_bounds(first)
    b_left, b_bottom, b_right, b_top = _normalize_bounds(second)
    return (
        min(a_right, b_right) > max(a_left, b_left)
        and min(a_top, b_top) > max(a_bottom, b_bottom)
    )


def _grid_from_components(
    *,
    crs: CRS,
    affine: Affine,
    width: int,
    height: int,
    longitude: float,
    latitude: float,
    max_center_offset: float,
    min_pixel_size: float,
    max_pixel_size: float,
) -> Grid:
    if width != WINDOW_SIZE or height != WINDOW_SIZE:
        raise RuntimeError(f"reference_not_512:{height}x{width}")
    if not _has_real_georef(crs, affine):
        raise RuntimeError("reference_missing_real_georef")
    if not crs.is_projected:
        raise RuntimeError(f"reference_crs_not_projected:{crs}")
    determinant = affine.a * affine.e - affine.b * affine.d
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        raise RuntimeError("reference_transform_singular")
    pixel_x = math.hypot(affine.a, affine.d)
    pixel_y = math.hypot(affine.b, affine.e)
    if not (
        min_pixel_size <= pixel_x <= max_pixel_size
        and min_pixel_size <= pixel_y <= max_pixel_size
    ):
        raise RuntimeError(
            f"reference_pixel_size_out_of_range:{pixel_x:.6f},{pixel_y:.6f}"
        )
    x_values, y_values = transform(
        "EPSG:4326",
        crs,
        [longitude],
        [latitude],
    )
    point_col, point_row = (~affine) * (
        float(x_values[0]),
        float(y_values[0]),
    )
    center_offset = max(
        abs(float(point_col) - width / 2),
        abs(float(point_row) - height / 2),
    )
    if not (
        0 <= point_col <= width
        and 0 <= point_row <= height
    ):
        raise RuntimeError(
            f"plume_point_outside_reference:{point_col:.6f},{point_row:.6f}"
        )
    if center_offset > max_center_offset:
        raise RuntimeError(
            f"reference_not_point_centered:offset={center_offset:.6f}"
        )
    bounds = _normalize_bounds(array_bounds(height, width, affine))
    bounds_wgs84 = _normalize_bounds(
        transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    )
    return Grid(
        crs=crs,
        transform=affine,
        width=width,
        height=height,
        bounds=bounds,
        bounds_wgs84=bounds_wgs84,
        point_col=float(point_col),
        point_row=float(point_row),
        center_offset_pixels=float(center_offset),
    )


def _sidecar_candidates(reference_path: Path) -> list[Path]:
    candidates = [
        reference_path.with_name(reference_path.name + ".georef.json"),
        reference_path.with_suffix(".georef.json"),
    ]
    return list(dict.fromkeys(candidates))


def _grid_from_sidecar(
    sidecar: Path,
    *,
    plume_id: str,
    longitude: float,
    latitude: float,
    args: argparse.Namespace,
) -> Grid:
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_plume_id = _clean(metadata.get("plume_id", ""))
    if sidecar_plume_id and sidecar_plume_id != plume_id:
        raise RuntimeError(
            f"sidecar_plume_id_mismatch:{sidecar_plume_id}"
        )
    timepoint = _clean(metadata.get("timepoint", ""))
    if timepoint and timepoint != "t0":
        raise RuntimeError(f"sidecar_not_t0:{timepoint}")
    transform_values = metadata.get("transform")
    if not isinstance(transform_values, list) or len(transform_values) < 6:
        raise RuntimeError("sidecar_missing_transform")
    try:
        affine = Affine(
            *(float(value) for value in transform_values[:6])
        )
        width = int(metadata["width"])
        height = int(metadata["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("sidecar_invalid_grid_fields") from exc
    crs_text = _clean(metadata.get("crs_wkt", ""))
    if not crs_text:
        crs_text = _clean(metadata.get("crs", ""))
    if not crs_text and metadata.get("epsg"):
        crs_text = f"EPSG:{int(metadata['epsg'])}"
    if not crs_text:
        raise RuntimeError("sidecar_missing_crs")
    try:
        crs = CRS.from_user_input(crs_text)
    except Exception as exc:
        raise RuntimeError("sidecar_invalid_crs") from exc
    return _grid_from_components(
        crs=crs,
        affine=affine,
        width=width,
        height=height,
        longitude=longitude,
        latitude=latitude,
        max_center_offset=args.max_center_offset_pixels,
        min_pixel_size=args.min_pixel_size,
        max_pixel_size=args.max_pixel_size,
    )


def _grid_from_raster(
    path: Path,
    *,
    longitude: float,
    latitude: float,
    args: argparse.Namespace,
) -> Grid:
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise RuntimeError("raster_reference_missing_crs")
        return _grid_from_components(
            crs=dataset.crs,
            affine=dataset.transform,
            width=int(dataset.width),
            height=int(dataset.height),
            longitude=longitude,
            latitude=latitude,
            max_center_offset=args.max_center_offset_pixels,
            min_pixel_size=args.min_pixel_size,
            max_pixel_size=args.max_pixel_size,
        )


def _resolve_reference(
    row: dict[str, Any],
    *,
    plume_id: str,
    longitude: float,
    latitude: float,
    args: argparse.Namespace,
) -> tuple[Grid, Path, Path | None, str, list[str]]:
    failures: list[str] = []
    for column, label in TRUSTED_REFERENCE_COLUMNS:
        source_column = f"source_{column}"
        path_text = _clean(row.get(source_column, ""))
        if not path_text:
            failures.append(f"{column}:empty")
            continue
        path = Path(path_text)
        if not _file_ok(path):
            failures.append(f"{column}:missing_file")
            continue
        try:
            _, _, spatial_ok = _storage_shape(path)
            if not spatial_ok:
                raise RuntimeError("reference_array_not_spatial_512")
        except Exception as exc:
            failures.append(
                f"{column}:array:{type(exc).__name__}:{str(exc)[:120]}"
            )
            continue

        sidecars = _sidecar_candidates(path)
        sidecar_found = False
        for sidecar in sidecars:
            if not _file_ok(sidecar):
                continue
            sidecar_found = True
            try:
                grid = _grid_from_sidecar(
                    sidecar,
                    plume_id=plume_id,
                    longitude=longitude,
                    latitude=latitude,
                    args=args,
                )
                return (
                    grid,
                    path,
                    sidecar,
                    f"{label}:georef_sidecar",
                    failures,
                )
            except Exception as exc:
                failures.append(
                    f"{column}:sidecar:{type(exc).__name__}:"
                    f"{str(exc)[:160]}"
                )
        try:
            grid = _grid_from_raster(
                path,
                longitude=longitude,
                latitude=latitude,
                args=args,
            )
            return grid, path, None, f"{label}:raster_georef", failures
        except Exception as exc:
            if not sidecar_found:
                failures.append(f"{column}:missing_sidecar")
            failures.append(
                f"{column}:raster:{type(exc).__name__}:{str(exc)[:160]}"
            )
    raise RuntimeError(
        "no_trusted_georef_reference|" + "|".join(failures[:12])
    )


def _resolve_raw_mask(
    plume_id: str,
    row: dict[str, Any],
    roots: Sequence[Path],
) -> Path:
    candidates = [
        root / plume_id / "plume.tif"
        for root in roots
    ]
    local_plume_tif = _clean(row.get("plume_tif", ""))
    if local_plume_tif.startswith("/"):
        candidates.append(Path(local_plume_tif))
    for candidate in candidates:
        if _file_ok(candidate):
            return candidate
    raise RuntimeError(
        "missing_raw_cm_plume:"
        + "|".join(str(path) for path in candidates)
    )


def _raw_mask_metadata(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise RuntimeError("raw_cm_mask_missing_crs")
        if not _has_real_georef(dataset.crs, dataset.transform):
            raise RuntimeError("raw_cm_mask_missing_real_georef")
        bounds = _normalize_bounds(dataset.bounds)
        bounds_wgs84 = _normalize_bounds(
            transform_bounds(
                dataset.crs,
                "EPSG:4326",
                *bounds,
                densify_pts=21,
            )
        )
        return {
            "crs": dataset.crs,
            "transform": dataset.transform,
            "bounds": bounds,
            "bounds_wgs84": bounds_wgs84,
            "width": int(dataset.width),
            "height": int(dataset.height),
            "count": int(dataset.count),
        }


def _resolve_row(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    cm_roots: Sequence[Path],
) -> Resolution:
    plume_id = _clean(row.get("plume_id", ""))
    output_path = (
        args.output_root
        / _safe_component(plume_id)
        / "s2_mask_512.tif"
    )
    record: dict[str, Any] = {
        "plume_id": plume_id,
        "event_group_id": _clean(row.get("event_group_id", "")),
        "event_time": _clean(row.get("event_time", "")),
        "split": _clean(row.get("split", "")),
        "resolution_status": "fail",
        "resolution_reason": "",
        "write_status": "not_requested" if not args.write else "not_run",
        "final_status": "fail",
        "final_reason": "",
        "raw_cm_mask_path": "",
        "raw_cm_crs": "",
        "raw_cm_shape": "",
        "raw_cm_bounds_wgs84": "",
        "reference_column": "",
        "reference_kind": "",
        "reference_path": "",
        "reference_sidecar": "",
        "reference_crs": "",
        "reference_transform": "",
        "reference_pixel_size_x": "",
        "reference_pixel_size_y": "",
        "reference_bounds_wgs84": "",
        "plume_point_col": "",
        "plume_point_row": "",
        "center_offset_pixels": "",
        "plume_point_inside_target": False,
        "raw_bbox_overlaps_target": False,
        "catalogue_plume_bounds_overlap_target": False,
        "output_mask_path": str(output_path),
        "mask_reprojection_method": "",
        "subpixel_fallback_triggered": False,
        "fallback_source_positive_pixels": "",
        "fallback_source_centers_inside_target": "",
        "fallback_target_positive_cells": "",
        "positive_pixels": "",
        "positive_fraction": "",
        "center_10x10_positive_pixels": "",
        "positive_bounds_wgs84": "",
        "positive_bounds_overlap_raw_cm": False,
        "positive_bounds_overlap_catalogue_plume": False,
        "output_shape": "",
        "output_crs": "",
        "output_transform": "",
    }
    if not plume_id:
        record["resolution_reason"] = "missing_plume_id"
        record["final_reason"] = record["resolution_reason"]
        return Resolution(record=record, row=row)
    if _clean(row.get("source_t0_v8_source_path", "")) == "":
        record["resolution_reason"] = "missing_s2_source_join_or_provenance"
        record["final_reason"] = record["resolution_reason"]
        return Resolution(record=record, row=row)
    try:
        latitude = float(row["plume_latitude"])
        longitude = float(row["plume_longitude"])
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        record["resolution_reason"] = "invalid_plume_coordinates"
        record["final_reason"] = record["resolution_reason"]
        return Resolution(record=record, row=row)

    target_path = Path(_clean(row.get("s2_t0_path", "")))
    try:
        target_info = _inspect_target(target_path)
        record.update(target_info)
    except Exception as exc:
        record["resolution_reason"] = (
            f"target_inspection_failed:{type(exc).__name__}:{str(exc)[:180]}"
        )
        record["final_reason"] = record["resolution_reason"]
        return Resolution(record=record, row=row)
    if not record["target_exists"]:
        record["resolution_reason"] = "missing_v14_target"
        record["final_reason"] = record["resolution_reason"]
        return Resolution(record=record, row=row)
    if not record["target_spatial_shape_ok"]:
        record["resolution_reason"] = (
            f"v14_target_not_spatial_512:{record['target_storage_shape']}"
        )
        record["final_reason"] = record["resolution_reason"]
        return Resolution(record=record, row=row)

    try:
        raw_path = _resolve_raw_mask(plume_id, row, cm_roots)
        raw_info = _raw_mask_metadata(raw_path)
        record["raw_cm_mask_path"] = str(raw_path)
        record["raw_cm_crs"] = raw_info["crs"].to_string()
        record["raw_cm_shape"] = (
            f"{raw_info['count']}x{raw_info['height']}x{raw_info['width']}"
        )
        record["raw_cm_bounds_wgs84"] = json.dumps(
            raw_info["bounds_wgs84"]
        )
    except Exception as exc:
        record["resolution_reason"] = (
            f"raw_cm_mask_invalid:{type(exc).__name__}:{str(exc)[:220]}"
        )
        record["final_reason"] = record["resolution_reason"]
        return Resolution(record=record, row=row)

    try:
        grid, reference, sidecar, reference_kind, prior_failures = (
            _resolve_reference(
                row,
                plume_id=plume_id,
                longitude=longitude,
                latitude=latitude,
                args=args,
            )
        )
        reference_column = next(
            (
                column
                for column, _ in TRUSTED_REFERENCE_COLUMNS
                if _clean(row.get(f"source_{column}", ""))
                == str(reference)
            ),
            "",
        )
        record["reference_column"] = reference_column
        record["reference_kind"] = reference_kind
        record["reference_path"] = str(reference)
        record["reference_sidecar"] = str(sidecar or "")
        record["reference_crs"] = grid.crs.to_string()
        record["reference_transform"] = json.dumps(
            [float(value) for value in grid.transform[:6]]
        )
        record["reference_pixel_size_x"] = float(
            math.hypot(grid.transform.a, grid.transform.d)
        )
        record["reference_pixel_size_y"] = float(
            math.hypot(grid.transform.b, grid.transform.e)
        )
        record["reference_bounds_wgs84"] = json.dumps(grid.bounds_wgs84)
        record["plume_point_col"] = grid.point_col
        record["plume_point_row"] = grid.point_row
        record["center_offset_pixels"] = grid.center_offset_pixels
        record["plume_point_inside_target"] = True
        record["reference_prior_failures"] = "|".join(prior_failures)
    except Exception as exc:
        record["resolution_reason"] = (
            f"trusted_reference_invalid:{type(exc).__name__}:"
            f"{str(exc)[:500]}"
        )
        record["final_reason"] = record["resolution_reason"]
        return Resolution(
            record=record,
            row=row,
            raw_mask_path=raw_path,
        )

    try:
        raw_bounds_in_target = _normalize_bounds(
            transform_bounds(
                raw_info["crs"],
                grid.crs,
                *raw_info["bounds"],
                densify_pts=21,
            )
        )
        raw_overlap = _bounds_overlap(raw_bounds_in_target, grid.bounds)
        record["raw_bbox_overlaps_target"] = bool(raw_overlap)
        plume_bounds = _parse_plume_bounds(row.get("plume_bounds", ""))
        plume_overlap = (
            _bounds_overlap(plume_bounds, grid.bounds_wgs84)
            if plume_bounds is not None
            else False
        )
        record["catalogue_plume_bounds_overlap_target"] = bool(plume_overlap)
        if not raw_overlap:
            raise RuntimeError("raw_cm_bbox_does_not_overlap_target")
        if plume_bounds is None:
            raise RuntimeError("invalid_catalogue_plume_bounds")
        if not plume_overlap:
            raise RuntimeError(
                "catalogue_plume_bounds_do_not_overlap_target"
            )
        if record["target_rasterio_has_real_georef"]:
            with rasterio.open(target_path) as target_dataset:
                if (
                    target_dataset.crs != grid.crs
                    or not target_dataset.transform.almost_equals(
                        grid.transform
                    )
                ):
                    raise RuntimeError(
                        "v14_real_georef_conflicts_with_trusted_reference"
                    )
    except Exception as exc:
        record["resolution_reason"] = (
            f"geographic_qa_failed:{type(exc).__name__}:{str(exc)[:220]}"
        )
        record["final_reason"] = record["resolution_reason"]
        return Resolution(
            record=record,
            row=row,
            grid=grid,
            raw_mask_path=raw_path,
            reference_path=reference,
            sidecar_path=sidecar,
        )

    record["resolution_status"] = "ready"
    record["resolution_reason"] = "ready"
    record["final_status"] = "ready" if not args.write else "pending"
    record["final_reason"] = "ready" if not args.write else "pending_write"
    return Resolution(
        record=record,
        row=row,
        grid=grid,
        raw_mask_path=raw_path,
        reference_path=reference,
        sidecar_path=sidecar,
    )


def _source_binary_mask(dataset: rasterio.io.DatasetReader) -> np.ndarray:
    dataset_mask = dataset.dataset_mask()
    if np.any(dataset_mask == 0):
        binary = dataset_mask > 0
    else:
        band = dataset.read(1)
        binary = np.isfinite(band) & (band > 0)
    return binary.astype(np.uint8)


def _subpixel_positive_centres_to_target_cells(
    source_mask: np.ndarray,
    *,
    source_transform: Affine,
    source_crs: CRS,
    grid: Grid,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Quantise positive source-pixel centres onto the target grid.

    This fallback is intentionally narrower than polygon/all-touched
    rasterisation: it marks a target cell only when the centre of a real
    positive source pixel falls inside that cell.  It is used only when the
    normal nearest-neighbour reprojection is empty.
    """

    source_rows, source_columns = np.nonzero(source_mask)
    if source_rows.size == 0:
        raise RuntimeError("raw_cm_mask_empty")
    source_x, source_y = xy(
        source_transform,
        source_rows,
        source_columns,
        offset="center",
    )
    if source_crs != grid.crs:
        source_x, source_y = transform(
            source_crs,
            grid.crs,
            source_x,
            source_y,
        )

    inverse = ~grid.transform
    target_rows: list[int] = []
    target_columns: list[int] = []
    for projected_x, projected_y in zip(source_x, source_y):
        column, row = inverse * (projected_x, projected_y)
        target_row = math.floor(row)
        target_column = math.floor(column)
        if (
            0 <= target_row < grid.height
            and 0 <= target_column < grid.width
        ):
            target_rows.append(target_row)
            target_columns.append(target_column)

    if not target_rows:
        raise RuntimeError("subpixel_source_centres_outside_target")
    destination = np.zeros(
        (grid.height, grid.width),
        dtype=np.uint8,
    )
    destination[
        np.asarray(target_rows, dtype=np.int64),
        np.asarray(target_columns, dtype=np.int64),
    ] = 1
    return destination, {
        "subpixel_fallback_triggered": True,
        "fallback_source_positive_pixels": int(source_rows.size),
        "fallback_source_centers_inside_target": int(len(target_rows)),
        "fallback_target_positive_cells": int(
            np.count_nonzero(destination)
        ),
    }


def _validate_existing_output(
    path: Path,
    grid: Grid,
) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise RuntimeError(f"output_count={dataset.count}")
        if (dataset.height, dataset.width) != (
            WINDOW_SIZE,
            WINDOW_SIZE,
        ):
            raise RuntimeError(
                f"output_shape={dataset.height}x{dataset.width}"
            )
        if dataset.crs != grid.crs:
            raise RuntimeError(
                f"output_crs_mismatch:{dataset.crs}!={grid.crs}"
            )
        if not dataset.transform.almost_equals(grid.transform):
            raise RuntimeError("output_transform_mismatch")
        array = dataset.read(1)
    unique = set(np.unique(array).tolist())
    if not unique.issubset({0, 1}):
        raise RuntimeError(f"output_not_binary:{sorted(unique)[:20]}")
    if not np.any(array):
        raise RuntimeError("output_mask_empty")
    return array.astype(np.uint8), {
        "output_shape": f"{array.shape[0]}x{array.shape[1]}",
        "output_crs": grid.crs.to_string(),
        "output_transform": json.dumps(
            [float(value) for value in grid.transform[:6]]
        ),
    }


def _positive_bounds(
    mask: np.ndarray,
    grid: Grid,
) -> tuple[float, float, float, float]:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        raise RuntimeError("reprojected_mask_empty")
    window = Window(
        col_off=int(columns.min()),
        row_off=int(rows.min()),
        width=int(columns.max() - columns.min() + 1),
        height=int(rows.max() - rows.min() + 1),
    )
    projected = _normalize_bounds(window_bounds(window, grid.transform))
    return _normalize_bounds(
        transform_bounds(
            grid.crs,
            "EPSG:4326",
            *projected,
            densify_pts=21,
        )
    )


def _mask_qa(
    mask: np.ndarray,
    resolution: Resolution,
) -> dict[str, Any]:
    if resolution.grid is None or resolution.raw_mask_path is None:
        raise RuntimeError("internal_missing_resolution")
    grid = resolution.grid
    positive_pixels = int(np.count_nonzero(mask))
    if positive_pixels <= 0:
        raise RuntimeError("reprojected_mask_empty")
    positive_bounds = _positive_bounds(mask, grid)
    raw_bounds = json.loads(
        resolution.record["raw_cm_bounds_wgs84"]
    )
    plume_bounds = _parse_plume_bounds(
        resolution.row.get("plume_bounds", "")
    )
    raw_overlap = _bounds_overlap(positive_bounds, raw_bounds)
    plume_overlap = (
        _bounds_overlap(positive_bounds, plume_bounds)
        if plume_bounds is not None
        else False
    )
    if not raw_overlap:
        raise RuntimeError("positive_mask_bounds_do_not_overlap_raw_cm")
    if not plume_overlap:
        raise RuntimeError(
            "positive_mask_bounds_do_not_overlap_catalogue_plume"
        )
    half = 5
    center = mask[
        WINDOW_SIZE // 2 - half : WINDOW_SIZE // 2 + half,
        WINDOW_SIZE // 2 - half : WINDOW_SIZE // 2 + half,
    ]
    return {
        "positive_pixels": positive_pixels,
        "positive_fraction": positive_pixels / float(mask.size),
        "center_10x10_positive_pixels": int(np.count_nonzero(center)),
        "positive_bounds_wgs84": json.dumps(positive_bounds),
        "positive_bounds_overlap_raw_cm": bool(raw_overlap),
        "positive_bounds_overlap_catalogue_plume": bool(plume_overlap),
    }


def _write_mask(resolution: Resolution, args: argparse.Namespace) -> Resolution:
    record = resolution.record
    if record["resolution_status"] != "ready":
        record["write_status"] = "not_run_resolution_failed"
        return resolution
    if resolution.grid is None or resolution.raw_mask_path is None:
        record["write_status"] = "fail"
        record["final_status"] = "fail"
        record["final_reason"] = "internal_missing_resolution"
        return resolution

    output_path = Path(record["output_mask_path"])
    try:
        if _file_ok(output_path) and not args.overwrite:
            mask, output_qa = _validate_existing_output(
                output_path,
                resolution.grid,
            )
            record.update(output_qa)
            record.update(_mask_qa(mask, resolution))
            record["write_status"] = "ok_existing"
            record["final_status"] = "ok"
            record["final_reason"] = "existing_output_valid"
            return resolution

        grid = resolution.grid
        with rasterio.open(resolution.raw_mask_path) as source:
            source_mask = _source_binary_mask(source)
            if not np.any(source_mask):
                raise RuntimeError("raw_cm_mask_empty")
            destination = np.zeros(
                (grid.height, grid.width),
                dtype=np.uint8,
            )
            reproject(
                source=source_mask,
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=0,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
                num_threads=1,
                init_dest_nodata=True,
            )
            method = "raw_cm_nearest"
            if (
                not np.any(destination)
                and args.subpixel_point_fallback
            ):
                destination, fallback_qa = (
                    _subpixel_positive_centres_to_target_cells(
                        source_mask,
                        source_transform=source.transform,
                        source_crs=source.crs,
                        grid=grid,
                    )
                )
                record.update(fallback_qa)
                method = "subpixel_positive_centres_minimum_target_cells"
        qa = _mask_qa(destination, resolution)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{output_path.name}.",
                suffix=".tmp.tif",
                dir=output_path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            profile = {
                "driver": "GTiff",
                "height": grid.height,
                "width": grid.width,
                "count": 1,
                "dtype": "uint8",
                "crs": grid.crs,
                "transform": grid.transform,
                "nodata": 0,
                "compress": "deflate",
                "predictor": 1,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "IF_SAFER",
            }
            with rasterio.open(temporary, "w", **profile) as destination_file:
                destination_file.write(destination, 1)
            readback, output_qa = _validate_existing_output(
                temporary,
                grid,
            )
            readback_qa = _mask_qa(readback, resolution)
            if int(readback_qa["positive_pixels"]) != int(
                qa["positive_pixels"]
            ):
                raise RuntimeError("readback_positive_pixel_mismatch")
            os.replace(temporary, output_path)
            temporary = None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

        record.update(qa)
        record.update(output_qa)
        record["mask_reprojection_method"] = method
        record["write_status"] = "ok_written"
        record["final_status"] = "ok"
        record["final_reason"] = f"reprojected_{method}"
    except Exception as exc:
        record["write_status"] = "fail"
        record["final_status"] = "fail"
        record["final_reason"] = (
            f"write_or_qa_failed:{type(exc).__name__}:{str(exc)[:300]}"
        )
    return resolution


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


def _quantiles(values: Iterable[Any]) -> dict[str, float] | None:
    numeric = np.asarray(
        [
            float(value)
            for value in values
            if _clean(value) and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if numeric.size == 0:
        return None
    return {
        "min": float(numeric.min()),
        "q10": float(np.quantile(numeric, 0.10)),
        "median": float(np.median(numeric)),
        "q90": float(np.quantile(numeric, 0.90)),
        "max": float(numeric.max()),
    }


def _counter(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(_clean(value) or "<empty>" for value in values)
    return {key: int(counts[key]) for key in sorted(counts)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.sample < 0:
        raise ValueError("--sample cannot be negative")
    if args.max_center_offset_pixels <= 0:
        raise ValueError("--max-center-offset-pixels must be positive")
    if not 0 < args.min_pixel_size <= args.max_pixel_size:
        raise ValueError("invalid pixel-size limits")
    cm_roots = tuple(args.cm_root or DEFAULT_CM_ROOTS)

    rows, input_audit = _load_rows(args)
    records = rows.to_dict("records")
    print(
        f"[s2-mask] mode={'write' if args.write else 'dry-run'} "
        f"rows={len(records)} workers={args.workers}",
        flush=True,
    )

    resolutions: list[Resolution] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, resolution in enumerate(
            executor.map(
                lambda row: _resolve_row(
                    row,
                    args=args,
                    cm_roots=cm_roots,
                ),
                records,
            ),
            start=1,
        ):
            resolutions.append(resolution)
            if index % 250 == 0 or index == len(records):
                print(
                    f"[s2-mask] resolve={index}/{len(records)}",
                    flush=True,
                )

    if args.write:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            written: list[Resolution] = []
            for index, resolution in enumerate(
                executor.map(
                    lambda item: _write_mask(item, args),
                    resolutions,
                ),
                start=1,
            ):
                written.append(resolution)
                if index % 50 == 0 or index == len(resolutions):
                    print(
                        f"[s2-mask] write={index}/{len(resolutions)}",
                        flush=True,
                    )
            resolutions = written

    qa = pd.DataFrame([resolution.record for resolution in resolutions])
    qa = qa.sort_values("plume_id", kind="mergesort").reset_index(drop=True)
    _atomic_write_csv(qa, args.resolution_csv)

    ready = qa["resolution_status"].eq("ready")
    final_ok = qa["final_status"].eq("ok")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "mode": "write" if args.write else "dry_run",
        "inputs": input_audit,
        "policy": {
            "selected_rows": "has_s2=True in unified wide manifest",
            "trusted_reference_columns": [
                {"column": column, "kind": kind}
                for column, kind in TRUSTED_REFERENCE_COLUMNS
            ],
            "excluded_as_georef": [
                "s2_t0_path (v14 tifffile array)",
                "t0_input_path/t0_raw_path (bounds-native grid)",
            ],
            "v14_use": "array spatial-shape validation only",
            "max_center_offset_pixels": args.max_center_offset_pixels,
            "pixel_size_range": [
                args.min_pixel_size,
                args.max_pixel_size,
            ],
            "reprojection_resampling": "nearest",
            "subpixel_point_fallback": {
                "enabled": bool(args.subpixel_point_fallback),
                "trigger": "nearest reprojection is empty",
                "policy": (
                    "mark only target cells containing centres of real "
                    "positive source pixels; no dilation and no polygon "
                    "or all-touched area fabrication"
                ),
            },
            "raw_binary_source": (
                "rasterio dataset_mask/alpha; if all pixels are valid, "
                "finite band-1 values > 0"
            ),
            "empty_output_allowed": False,
            "overwrite": bool(args.overwrite),
        },
        "cm_roots": [str(path) for path in cm_roots],
        "outputs": {
            "mask_root": str(args.output_root),
            "resolution_csv": str(args.resolution_csv),
            "audit_json": str(args.audit_json),
        },
        "results": {
            "rows": int(len(qa)),
            "resolution_ready": int(ready.sum()),
            "resolution_failed": int((~ready).sum()),
            "resolution_reason_counts": _counter(
                qa["resolution_reason"]
            ),
            "reference_kind_counts": _counter(
                qa.loc[ready, "reference_kind"]
            ),
            "target_rasterio_real_georef_counts": _counter(
                qa["target_rasterio_has_real_georef"]
            ),
            "final_status_counts": _counter(qa["final_status"]),
            "final_reason_counts": _counter(qa["final_reason"]),
            "written_or_valid_existing": int(final_ok.sum()),
            "center_offset_pixels": _quantiles(
                qa.loc[ready, "center_offset_pixels"]
            ),
            "positive_pixels": _quantiles(
                qa.loc[final_ok, "positive_pixels"]
            ),
            "positive_fraction": _quantiles(
                qa.loc[final_ok, "positive_fraction"]
            ),
            "all_ready_point_inside_target": bool(
                qa.loc[ready, "plume_point_inside_target"].map(
                    _parse_bool
                ).all()
            ),
            "all_ready_raw_bbox_overlap": bool(
                qa.loc[ready, "raw_bbox_overlaps_target"].map(
                    _parse_bool
                ).all()
            ),
            "all_ready_catalogue_bounds_overlap": bool(
                qa.loc[
                    ready,
                    "catalogue_plume_bounds_overlap_target",
                ]
                .map(_parse_bool)
                .all()
            ),
            "all_ok_positive_bounds_overlap_raw": bool(
                qa.loc[
                    final_ok,
                    "positive_bounds_overlap_raw_cm",
                ]
                .map(_parse_bool)
                .all()
            ),
            "all_ok_positive_bounds_overlap_catalogue": bool(
                qa.loc[
                    final_ok,
                    "positive_bounds_overlap_catalogue_plume",
                ]
                .map(_parse_bool)
                .all()
            ),
        },
    }
    _atomic_write_json(audit, args.audit_json)
    summary = {
        "mode": audit["mode"],
        "rows": int(len(qa)),
        "resolution_ready": int(ready.sum()),
        "resolution_failed": int((~ready).sum()),
        "written_or_valid_existing": int(final_ok.sum()),
        "resolution_reason_counts": audit["results"][
            "resolution_reason_counts"
        ],
        "final_reason_counts": audit["results"]["final_reason_counts"],
        "resolution_csv": str(args.resolution_csv),
        "audit_json": str(args.audit_json),
        "output_root": str(args.output_root),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return audit


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
