#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import rasterio
import requests
from affine import Affine
try:
    from pyproj import CRS, Transformer
except ImportError:
    from rasterio.crs import CRS as RasterioCRS
    from rasterio.warp import transform as rasterio_transform

    class CRS:
        def __init__(self, value: Any):
            self.value = value

        @classmethod
        def from_user_input(cls, value: Any) -> "CRS":
            return cls(value)

        def to_string(self) -> str:
            return RasterioCRS.from_user_input(self.value).to_string()

    class Transformer:
        def __init__(self, source: Any, destination: Any):
            self.source = source
            self.destination = destination

        @classmethod
        def from_crs(
            cls,
            source: Any,
            destination: Any,
            always_xy: bool = True,
        ) -> "Transformer":
            return cls(source, destination)

        def transform(self, x: float, y: float) -> tuple[float, float]:
            xs, ys = rasterio_transform(
                self.source,
                self.destination,
                [float(x)],
                [float(y)],
            )
            return float(xs[0]), float(ys[0])

try:
    from shapely.geometry import Point, box, shape
    from shapely.ops import transform as transform_geometry
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
DEFAULT_EXISTING_ROOTS = (
    "/mnt/engg-niulab/yuyao/sensors_raw_data/S2/raw_data_dir_s2",
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_s2_90360",
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_s2",
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2",
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2_-7",
)
PRODUCT_RE = re.compile(
    r"^S2(?P<satellite>[ABC])_MSIL2A_"
    r"(?P<sensing>\d{8}T\d{6})_"
    r"N(?P<baseline>\d{4})_"
    r"R(?P<orbit>\d{3})_"
    r"T(?P<tile>\d{2}[A-Z]{3})_"
    r"(?P<generation>\d{8}T\d{6})\.SAFE$"
)
R20M_RE = re.compile(r".*_B(?:0?[1-9]|1[0-2]|8A)_20m\.jp2$")
ITEM_URL = (
    "https://earth-search.aws.element84.com/v1/collections/"
    "sentinel-2-l2a/items/{item_id}"
)
SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
CDSE_PRODUCTS_URL = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
)
S3_PREFIX = "s3://sentinel-s2-l2a/"
HTTPS_PREFIX = "https://sentinel-s2-l2a.s3.amazonaws.com"
_thread_local = threading.local()
_transformer_lock = threading.Lock()
_transformers: dict[str, Transformer] = {}
_request_lock = threading.Lock()
_next_request_time = 0.0
_request_interval = 0.1
_http_cache_root: Path | None = None


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: Affine
    height: int
    width: int


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def parse_product(product_name: str) -> dict[str, str]:
    match = PRODUCT_RE.fullmatch(product_name)
    if match is None:
        raise ValueError(f"unsupported product name: {product_name}")
    return match.groupdict()


def session() -> requests.Session:
    value = getattr(_thread_local, "session", None)
    if value is None:
        value = requests.Session()
        value.headers["User-Agent"] = "panopticon-s2-tile-resolver/1.0"
        _thread_local.session = value
    return value


def request_cache_path(
    url: str,
    params: dict[str, Any] | None,
) -> Path | None:
    if _http_cache_root is None:
        return None
    payload = json.dumps(
        [url, params or {}],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _http_cache_root / (
        hashlib.sha256(payload).hexdigest() + ".json"
    )


def wait_for_request_slot() -> None:
    global _next_request_time
    with _request_lock:
        now = time.monotonic()
        if now < _next_request_time:
            time.sleep(_next_request_time - now)
        _next_request_time = time.monotonic() + _request_interval


def get_json(url: str, params: dict[str, Any] | None, retries: int) -> dict[str, Any] | None:
    cache_path = request_cache_path(url, params)
    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        return None if cached.get("__not_found__") else cached

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            wait_for_request_slot()
            response = session().get(url, params=params, timeout=60)
            if response.status_code == 404:
                if cache_path is not None:
                    temporary = cache_path.with_name(
                        cache_path.name + f".tmp.{os.getpid()}"
                    )
                    temporary.write_text(
                        json.dumps({"__not_found__": True})
                    )
                    os.replace(temporary, cache_path)
                return None
            if response.status_code in {403, 429}:
                retry_after = int(response.headers.get("Retry-After", "0") or 0)
                time.sleep(
                    max(
                        10,
                        min(120, retry_after or 15 * (attempt + 1)),
                    )
                )
                continue
            response.raise_for_status()
            payload = response.json()
            if cache_path is not None:
                temporary = cache_path.with_name(
                    cache_path.name
                    + f".tmp.{os.getpid()}.{threading.get_ident()}"
                )
                temporary.write_text(json.dumps(payload))
                os.replace(temporary, cache_path)
            return payload
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"request failed for {url}: {last_error}")


def product_score(wanted: dict[str, str], candidate_name: str) -> tuple[int, int, int] | None:
    try:
        candidate = parse_product(candidate_name)
    except ValueError:
        return None
    for field in ("satellite", "sensing", "orbit"):
        if candidate[field] != wanted[field]:
            return None
    baseline_penalty = int(candidate["baseline"] != wanted["baseline"])
    generation_delta = int(
        abs(
            (
                datetime.strptime(candidate["generation"], "%Y%m%dT%H%M%S")
                - datetime.strptime(wanted["generation"], "%Y%m%dT%H%M%S")
            ).total_seconds()
        )
    )
    exact_processing = int(
        candidate["baseline"] != wanted["baseline"]
        or candidate["generation"] != wanted["generation"]
    )
    return exact_processing, baseline_penalty, generation_delta


def item_base_url(item: dict[str, Any]) -> str:
    assets = item.get("assets", {})
    for asset_name in (
        "rededge1-jp2",
        "swir16-jp2",
        "blue-jp2",
        "green-jp2",
        "product_metadata",
    ):
        href = clean(assets.get(asset_name, {}).get("href", ""))
        if not href.startswith(S3_PREFIX):
            continue
        key = href[len(S3_PREFIX) :]
        return f"{HTTPS_PREFIX}/{key.rsplit('/', 1)[0]}"
    raise ValueError("missing original JP2 asset")


def item_cog_base_url(item: dict[str, Any]) -> str:
    assets = item.get("assets", {})
    for asset_name in ("rededge1", "swir16", "blue", "green"):
        href = clean(assets.get(asset_name, {}).get("href", ""))
        if href.startswith("https://") and "/" in href:
            return href.rsplit("/", 1)[0]
    raise ValueError("missing Sentinel-2 COG asset")


def item_dn_add(item: dict[str, Any]) -> int:
    assets = item.get("assets", {})
    for asset_name in ("rededge1", "swir16", "blue", "green"):
        bands = assets.get(asset_name, {}).get("raster:bands", [])
        if not bands:
            continue
        scale = float(bands[0].get("scale", 0.0001) or 0.0001)
        offset = float(bands[0].get("offset", 0.0) or 0.0)
        return int(round(offset / scale))
    return 0


def item_grid(item: dict[str, Any]) -> Grid:
    properties = item.get("properties", {})
    assets = item.get("assets", {})
    for asset_name in ("rededge1-jp2", "swir16-jp2", "rededge1", "swir16"):
        asset = assets.get(asset_name, {})
        shape = asset.get("proj:shape")
        transform = asset.get("proj:transform")
        epsg = asset.get("proj:epsg") or properties.get("proj:epsg")
        if (
            isinstance(shape, list)
            and len(shape) == 2
            and isinstance(transform, list)
            and len(transform) >= 6
            and epsg
        ):
            return Grid(
                crs=f"EPSG:{int(epsg)}",
                transform=Affine(*[float(value) for value in transform[:6]]),
                height=int(shape[0]),
                width=int(shape[1]),
            )
    raise ValueError("missing 20 m projection metadata")


def transformer_for(crs: str) -> Transformer:
    with _transformer_lock:
        value = _transformers.get(crs)
        if value is None:
            value = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            _transformers[crs] = value
        return value


def crop_metrics(grid: Grid, longitude: float, latitude: float) -> tuple[float, float]:
    transformer = transformer_for(grid.crs)
    top_left = transformer.transform(longitude - 0.01, latitude + 0.01)
    bottom_right = transformer.transform(longitude + 0.01, latitude - 0.01)
    tl_col, tl_row = (~grid.transform) * top_left
    br_col, br_row = (~grid.transform) * bottom_right
    center_col = (tl_col + br_col) / 2.0
    center_row = (tl_row + br_row) / 2.0
    col_start = math.floor(center_col - 256)
    row_start = math.floor(center_row - 256)
    margin = min(
        col_start,
        row_start,
        grid.width - (col_start + 512),
        grid.height - (row_start + 512),
    )
    point_x, point_y = transformer.transform(longitude, latitude)
    point_col, point_row = (~grid.transform) * (point_x, point_y)
    center_offset = max(
        abs(point_col - col_start - 256),
        abs(point_row - row_start - 256),
    )
    return float(margin), float(center_offset)


def point_inside_grid(
    grid: Grid,
    longitude: float,
    latitude: float,
) -> bool:
    transformer = transformer_for(grid.crs)
    x, y = transformer.transform(longitude, latitude)
    col, row = (~grid.transform) * (x, y)
    return 0 <= col < grid.width and 0 <= row < grid.height


def local_product_dir(product_name: str, roots: tuple[Path, ...]) -> Path | None:
    for root in roots:
        candidate = root / product_name
        try:
            if not candidate.is_dir():
                continue
            if any(
                path.is_file() and R20M_RE.fullmatch(path.name)
                for path in candidate.iterdir()
            ):
                return candidate
        except OSError:
            continue
    return None


def local_grid(product_dir: Path) -> Grid:
    candidates = sorted(
        path
        for path in product_dir.iterdir()
        if path.is_file()
        and (
            path.name.endswith("_B05_20m.jp2")
            or path.name.endswith("_B11_20m.jp2")
        )
    )
    if not candidates:
        candidates = sorted(
            path
            for path in product_dir.iterdir()
            if path.is_file() and R20M_RE.fullmatch(path.name)
        )
    if not candidates:
        raise FileNotFoundError(f"no R20m bands in {product_dir}")
    with rasterio.open(candidates[0]) as dataset:
        if dataset.crs is None:
            raise ValueError(f"missing CRS in {candidates[0]}")
        return Grid(
            crs=CRS.from_user_input(dataset.crs).to_string(),
            transform=dataset.transform,
            height=dataset.height,
            width=dataset.width,
        )


def local_sibling_products(
    wanted: dict[str, str],
    roots: tuple[Path, ...],
) -> list[dict[str, Any]]:
    pattern = (
        f"S2{wanted['satellite']}_MSIL2A_{wanted['sensing']}_"
        f"N{wanted['baseline']}_R{wanted['orbit']}_T*_"
        f"{wanted['generation']}.SAFE"
    )
    products: dict[str, dict[str, Any]] = {}
    for root in roots:
        try:
            candidates = root.glob(pattern)
            for path in candidates:
                if path.name in products or not path.is_dir():
                    continue
                try:
                    grid = local_grid(path)
                except (OSError, ValueError, FileNotFoundError):
                    continue
                products[path.name] = {
                    "product_name": path.name,
                    "product_dir": str(path),
                    "grid": grid,
                }
        except OSError:
            continue
    return list(products.values())


def repair_with_local_siblings(
    wanted: dict[str, str],
    tasks: list[dict[str, Any]],
    roots: tuple[Path, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    siblings = local_sibling_products(wanted, roots)
    if not siblings:
        return [], tasks
    repaired: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    sibling_identity = ",".join(
        sorted(value["product_name"] for value in siblings)
    )
    digest = hashlib.sha256(
        sibling_identity.encode()
    ).hexdigest()[:16]
    mosaic_sources = [
        {
            "product_name": value["product_name"],
            "product_dir": value["product_dir"],
        }
        for value in siblings
    ]
    for task in tasks:
        choices: list[tuple[float, float, dict[str, Any]]] = []
        point_covering: list[
            tuple[float, float, dict[str, Any]]
        ] = []
        for sibling in siblings:
            grid = sibling["grid"]
            margin, center_offset = crop_metrics(
                grid,
                task["longitude"],
                task["latitude"],
            )
            if point_inside_grid(
                grid,
                task["longitude"],
                task["latitude"],
            ):
                point_covering.append(
                    (margin, center_offset, sibling)
                )
            if margin >= 0 and center_offset <= 1.5:
                choices.append((margin, center_offset, sibling))
        if choices:
            margin, center_offset, selected = max(
                choices,
                key=lambda value: value[0],
            )
            repaired.append(
                {
                    **task,
                    "selected_product_name": selected[
                        "product_name"
                    ],
                    "selected_product_id": (
                        "local_sibling:"
                        + hashlib.sha256(
                            selected["product_name"].encode()
                        ).hexdigest()[:16]
                    ),
                    "tile_repair_status": (
                        "local_point_repaired"
                    ),
                    "tile_margin_pixels": margin,
                    "center_offset_pixels": center_offset,
                }
            )
            continue
        if point_covering and len(siblings) >= 2:
            margin, center_offset, reference = max(
                point_covering,
                key=lambda value: value[0],
            )
            ordered_sources = sorted(
                mosaic_sources,
                key=lambda source: source["product_name"]
                != reference["product_name"],
            )
            spatial_bin = (
                f"{round(task['longitude'], 1):.1f}_"
                f"{round(task['latitude'], 1):.1f}"
            )
            repaired.append(
                {
                    **task,
                    "local_mosaic_json": json.dumps(
                        ordered_sources,
                        separators=(",", ":"),
                    ),
                    "selected_product_name": reference[
                        "product_name"
                    ],
                    "selected_product_id": (
                        f"local_mosaic:{digest}:{spatial_bin}"
                    ),
                    "tile_repair_status": (
                        "local_mosaic_repaired"
                    ),
                    "tile_margin_pixels": margin,
                    "center_offset_pixels": center_offset,
                }
            )
            continue
        unresolved.append(task)
    return repaired, unresolved


def exact_stac_item(
    product_name: str,
    item_max_index: int,
    retries: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    wanted = parse_product(product_name)
    date_tag = wanted["sensing"][:8]
    best: tuple[tuple[int, int, int], dict[str, Any]] | None = None
    found_item = False
    consecutive_missing = 0
    for item_index in range(item_max_index):
        item_id = f"S2{wanted['satellite']}_{wanted['tile']}_{date_tag}_{item_index}_L2A"
        item = get_json(ITEM_URL.format(item_id=item_id), None, retries)
        if item is None:
            consecutive_missing += 1
            if found_item and consecutive_missing >= 3:
                break
            continue
        found_item = True
        consecutive_missing = 0
        candidate_name = clean(item.get("properties", {}).get("s2:product_uri"))
        score = product_score(wanted, candidate_name)
        if score is None:
            continue
        item["_resolved_item_id"] = item_id
        if candidate_name == product_name:
            return item, item
        if best is None or score < best[0]:
            best = (score, item)
    return None, None if best is None else best[1]


def point_items(
    wanted: dict[str, str],
    longitude: float,
    latitude: float,
    retries: int,
    epsilon: float = 0.00001,
) -> list[dict[str, Any]]:
    date_tag = wanted["sensing"][:8]
    payload = get_json(
        SEARCH_URL,
        {
            "collections": "sentinel-2-l2a",
            "bbox": (
                f"{longitude - epsilon},{latitude - epsilon},"
                f"{longitude + epsilon},{latitude + epsilon}"
            ),
            "datetime": (
                f"{date_tag[:4]}-{date_tag[4:6]}-{date_tag[6:8]}"
                "T00:00:00Z/"
                f"{date_tag[:4]}-{date_tag[4:6]}-{date_tag[6:8]}"
                "T23:59:59Z"
            ),
            "limit": 100,
        },
        retries,
    )
    if payload is None:
        return []
    return list(payload.get("features", []))


def source_record(item: dict[str, Any]) -> dict[str, str]:
    return {
        "stac_item_id": clean(item.get("id") or item.get("_resolved_item_id")),
        "stac_base_url": item_base_url(item),
        "stac_cog_base_url": item_cog_base_url(item),
        "stac_dn_add": str(item_dn_add(item)),
        "stac_source_product_name": clean(
            item.get("properties", {}).get("s2:product_uri")
        ),
    }


def cdse_point_record(
    wanted: dict[str, str],
    task: dict[str, Any],
    retries: int,
    force_mosaic: bool = False,
) -> dict[str, Any] | None:
    if not SHAPELY_AVAILABLE:
        return None
    sensing = datetime.strptime(
        wanted["sensing"],
        "%Y%m%dT%H%M%S",
    ).replace(tzinfo=timezone.utc)
    start = sensing - timedelta(minutes=5)
    end = sensing + timedelta(minutes=5)
    longitude = float(task["longitude"])
    latitude = float(task["latitude"])
    utm_zone = max(
        1,
        min(60, int(math.floor((longitude + 180.0) / 6.0)) + 1),
    )
    epsg = (32600 if latitude >= 0 else 32700) + utm_zone
    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True,
    )
    point = Point(*transformer.transform(longitude, latitude))
    crop_half_extent_m = 512.0 * 20.0 / 2.0
    crop_region = box(
        point.x - crop_half_extent_m,
        point.y - crop_half_extent_m,
        point.x + crop_half_extent_m,
        point.y + crop_half_extent_m,
    )
    inverse_transformer = Transformer.from_crs(
        f"EPSG:{epsg}",
        "EPSG:4326",
        always_xy=True,
    )
    query_region = transform_geometry(
        inverse_transformer.transform,
        crop_region,
    )
    coordinates = ",".join(
        f"{x:.8f} {y:.8f}"
        for x, y in query_region.exterior.coords
    )
    filter_text = (
        "Collection/Name eq 'SENTINEL-2' and "
        "OData.CSC.Intersects("
        "area=geography'SRID=4326;"
        f"POLYGON(({coordinates}))') and "
        f"ContentDate/Start gt {start.isoformat(timespec='milliseconds').replace('+00:00', 'Z')} "
        f"and ContentDate/Start lt {end.isoformat(timespec='milliseconds').replace('+00:00', 'Z')} "
        "and contains(Name,'MSIL2A')"
    )
    payload = get_json(
        CDSE_PRODUCTS_URL,
        {
            "$filter": filter_text,
            "$orderby": "ContentDate/Start asc",
            "$top": 100,
        },
        retries,
    )
    if payload is None:
        return None

    candidates: list[
        tuple[
            tuple[int, int, int],
            float,
            float,
            Any,
            dict[str, Any],
        ]
    ] = []
    for product in payload.get("value", []):
        product_name = clean(product.get("Name"))
        score = product_score(wanted, product_name)
        footprint_data = product.get("GeoFootprint")
        product_id = clean(product.get("Id"))
        if score is None or not product_id or not footprint_data:
            continue
        try:
            footprint = transform_geometry(
                transformer.transform,
                shape(footprint_data),
            )
        except Exception:
            continue
        if not footprint.intersects(crop_region):
            continue
        boundary_distance = (
            float(point.distance(footprint.boundary))
            if footprint.covers(point)
            else -float(point.distance(footprint))
        )
        overlap_area = float(
            footprint.intersection(crop_region).area
        )
        candidates.append(
            (
                score,
                boundary_distance,
                overlap_area,
                footprint,
                product,
            )
        )
    if not candidates:
        return None

    point_candidates = [
        value for value in candidates if value[3].covers(point)
    ]
    if not point_candidates:
        return None
    original_name = clean(
        task.get("original_product_name")
        or task.get("product_name")
    )
    original_tile = wanted["tile"]
    force_use_mosaic = False
    if force_mosaic:
        by_baseline: dict[
            str,
            list[
                tuple[
                    tuple[int, int, int],
                    float,
                    float,
                    Any,
                    dict[str, Any],
                ]
            ],
        ] = {}
        for candidate in candidates:
            candidate_name = clean(candidate[4].get("Name"))
            candidate_metadata = parse_product(candidate_name)
            by_baseline.setdefault(
                candidate_metadata["baseline"],
                [],
            ).append(candidate)

        selected = None
        same_processing = []
        for baseline, baseline_candidates in sorted(
            by_baseline.items(),
            key=lambda item: (
                item[0] != wanted["baseline"],
                min(value[0] for value in item[1]),
            ),
        ):
            by_tile: dict[
                str,
                tuple[
                    tuple[int, int, int],
                    float,
                    float,
                    Any,
                    dict[str, Any],
                ],
            ] = {}
            for candidate in baseline_candidates:
                candidate_name = clean(
                    candidate[4].get("Name")
                )
                tile = parse_product(candidate_name)["tile"]
                current = by_tile.get(tile)
                if current is None or candidate[0] < current[0]:
                    by_tile[tile] = candidate
            cohort = list(by_tile.values())
            cohort_point = [
                value for value in cohort if value[3].covers(point)
            ]
            full_alternates = [
                value
                for value in cohort_point
                if parse_product(
                    clean(value[4].get("Name"))
                )["tile"] != original_tile
                and value[3].covers(crop_region)
            ]
            if full_alternates:
                same_processing = cohort
                (
                    _,
                    boundary_distance,
                    _,
                    _,
                    selected,
                ) = max(
                    full_alternates,
                    key=lambda value: value[1],
                )
                break
            if len(cohort) < 2 or not cohort_point:
                continue
            coverage = unary_union(
                [value[3] for value in cohort]
            )
            if not coverage.buffer(1.0).covers(crop_region):
                continue
            same_processing = cohort
            (
                _,
                boundary_distance,
                _,
                _,
                selected,
            ) = max(cohort_point, key=lambda value: value[1])
            force_use_mosaic = True
            break
        if selected is None:
            return None
    else:
        best_score = min(value[0] for value in point_candidates)
        same_processing = [
            value for value in candidates if value[0] == best_score
        ]
        same_processing_point = [
            value
            for value in same_processing
            if value[3].covers(point)
        ]
        _, boundary_distance, _, _, selected = max(
            same_processing_point,
            key=lambda value: value[1],
        )
    selected_name = clean(selected.get("Name"))
    selected_id = clean(selected.get("Id"))
    margin_pixels = boundary_distance / 20.0 - 256.0
    use_mosaic = (
        force_use_mosaic
        or margin_pixels < 0
    )
    if use_mosaic and len(same_processing) >= 2:
        coverage = unary_union(
            [value[3] for value in same_processing]
        )
        if not coverage.covers(crop_region):
            return None
        sources = [
            {
                "product_name": clean(product.get("Name")),
                "product_id": clean(product.get("Id")),
            }
            for _, _, _, _, product in sorted(
                same_processing,
                key=lambda value: value[2],
                reverse=True,
            )
        ]
        identity = ",".join(
            source["product_id"] for source in sources
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        spatial_bin = (
            f"{round(longitude, 1):.1f}_"
            f"{round(latitude, 1):.1f}"
        )
        return {
            **task,
            "cdse_mosaic_json": json.dumps(
                sources,
                separators=(",", ":"),
            ),
            "selected_product_name": selected_name,
            "selected_product_id": (
                f"cdse_mosaic:{digest}:{spatial_bin}"
            ),
            "tile_repair_status": "cdse_mosaic_repaired",
            "tile_margin_pixels": margin_pixels,
            "center_offset_pixels": 0.0,
            "tile_repair_note": (
                "selected CDSE crop-intersection mosaic; "
                f"tiles={len(sources)}; "
                f"forced_after_pixel_gap={int(force_mosaic)}; "
                f"best_boundary_distance_m={boundary_distance:.3f}"
            ),
        }
    return {
        **task,
        "selected_product_name": selected_name,
        "selected_product_id": selected_id,
        "tile_repair_status": "cdse_point_repaired",
        "tile_margin_pixels": margin_pixels,
        "center_offset_pixels": 0.0,
        "tile_repair_note": (
            "selected by CDSE crop-region coverage; "
            f"boundary_distance_m={boundary_distance:.3f}"
        ),
    }


def mosaic_record(
    wanted: dict[str, str],
    task: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates: list[
        tuple[
            tuple[int, int, int],
            float,
            float,
            dict[str, Any],
        ]
    ] = []
    for item in items:
        candidate_name = clean(
            item.get("properties", {}).get("s2:product_uri")
        )
        score = product_score(wanted, candidate_name)
        if score is None:
            continue
        try:
            grid = item_grid(item)
            if not point_inside_grid(
                grid,
                task["longitude"],
                task["latitude"],
            ):
                continue
            margin, center_offset = crop_metrics(
                grid,
                task["longitude"],
                task["latitude"],
            )
            source_record(item)
        except (ValueError, KeyError):
            continue
        candidates.append((score, margin, center_offset, item))
    if not candidates:
        raise RuntimeError(
            "no point-covering STAC tiles for "
            f"{task['plume_id']} {task['timepoint']}"
        )

    best_processing = min(value[0] for value in candidates)
    same_processing = [
        value for value in candidates if value[0] == best_processing
    ]
    reference = max(same_processing, key=lambda value: value[1])
    sources = []
    seen: set[str] = set()
    for _, _, _, item in same_processing:
        source = source_record(item)
        item_id = source["stac_item_id"]
        if item_id in seen:
            continue
        seen.add(item_id)
        sources.append(source)
    reference_source = source_record(reference[3])
    sources.sort(
        key=lambda source: source["stac_item_id"]
        != reference_source["stac_item_id"]
    )
    identity = ",".join(source["stac_item_id"] for source in sources)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    spatial_bin = (
        f"{round(task['longitude'], 1):.1f}_"
        f"{round(task['latitude'], 1):.1f}"
    )
    return {
        **task,
        **reference_source,
        "stac_mosaic_json": json.dumps(
            sources,
            separators=(",", ":"),
        ),
        "selected_product_name": reference_source[
            "stac_source_product_name"
        ],
        "selected_product_id": (
            f"aws_mosaic:{digest}:{spatial_bin}"
        ),
        "tile_repair_status": "stac_mosaic_repaired",
        "tile_margin_pixels": reference[1],
        "center_offset_pixels": reference[2],
    }


def resolve_group(
    product_name: str,
    tasks: list[dict[str, Any]],
    roots: tuple[Path, ...],
    item_max_index: int,
    retries: int,
    force_stac_mosaic: bool,
) -> list[dict[str, Any]]:
    wanted = parse_product(product_name)
    if force_stac_mosaic:
        return [
            mosaic_record(
                wanted,
                task,
                point_items(
                    wanted,
                    task["longitude"],
                    task["latitude"],
                    retries,
                    epsilon=0.12,
                ),
            )
            for task in tasks
        ]
    product_dir = local_product_dir(product_name, roots)
    exact_item: dict[str, Any] | None = None
    fallback_item: dict[str, Any] | None = None
    if product_dir is not None:
        original_grid = local_grid(product_dir)
        original_source: dict[str, str] = {}
        source_kind = "local_exact"
    else:
        exact_item, fallback_item = exact_stac_item(
            product_name,
            item_max_index,
            retries,
        )
        metadata_item = exact_item or fallback_item
        if metadata_item is None:
            original_grid = None
            original_source = {}
            source_kind = "unresolved_original"
        else:
            original_grid = item_grid(metadata_item)
            original_source = source_record(exact_item) if exact_item else {}
            source_kind = "stac_exact" if exact_item else "stac_grid_only"

    records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for task in tasks:
        if original_grid is None:
            unresolved.append(task)
            continue
        margin, center_offset = crop_metrics(
            original_grid,
            task["longitude"],
            task["latitude"],
        )
        if margin >= 0 and center_offset <= 1.5:
            records.append(
                {
                    **task,
                    **original_source,
                    "selected_product_name": product_name,
                    "selected_product_id": task["product_id"],
                    "tile_repair_status": source_kind,
                    "tile_margin_pixels": margin,
                    "center_offset_pixels": center_offset,
                }
            )
        else:
            unresolved.append(task)

    local_repairs, unresolved = repair_with_local_siblings(
        wanted,
        unresolved,
        roots,
    )
    records.extend(local_repairs)

    def strict_original_record(
        task: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        if product_dir is not None:
            raise RuntimeError(
                f"no full 512 tile for {product_name}: "
                f"{task['plume_id']} {task['timepoint']}; {reason}"
            )
        if original_grid is None:
            margin = -1_000_000.0
            center_offset = 1_000_000.0
        else:
            margin, center_offset = crop_metrics(
                original_grid,
                task["longitude"],
                task["latitude"],
            )
        return {
            **task,
            **original_source,
            "selected_product_name": product_name,
            "selected_product_id": task["product_id"],
            "tile_repair_status": "remote_original_strict_pending",
            "tile_margin_pixels": margin,
            "center_offset_pixels": center_offset,
            "tile_repair_note": reason,
        }

    candidate_items: dict[str, dict[str, Any]] = {}
    while unresolved:
        seed = unresolved[0]
        for item in point_items(
            wanted,
            seed["longitude"],
            seed["latitude"],
            retries,
        ):
            item_id = clean(item.get("id"))
            candidate_name = clean(
                item.get("properties", {}).get("s2:product_uri")
            )
            if item_id and product_score(wanted, candidate_name) is not None:
                candidate_items[item_id] = item

        newly_resolved: list[dict[str, Any]] = []
        still_unresolved: list[dict[str, Any]] = []
        for task in unresolved:
            choices: list[
                tuple[tuple[int, int, int, float], dict[str, Any], float, float]
            ] = []
            for item in candidate_items.values():
                candidate_name = clean(
                    item.get("properties", {}).get("s2:product_uri")
                )
                score = product_score(wanted, candidate_name)
                if score is None:
                    continue
                try:
                    margin, center_offset = crop_metrics(
                        item_grid(item),
                        task["longitude"],
                        task["latitude"],
                    )
                except (ValueError, KeyError):
                    continue
                if margin < 0 or center_offset > 1.5:
                    continue
                choices.append(
                    ((*score, -margin), item, margin, center_offset)
                )
            if not choices:
                still_unresolved.append(task)
                continue
            _, selected, margin, center_offset = min(
                choices,
                key=lambda value: value[0],
            )
            selected_source = source_record(selected)
            selected_product_name = selected_source[
                "stac_source_product_name"
            ]
            newly_resolved.append(
                {
                    **task,
                    **selected_source,
                    "selected_product_name": selected_product_name,
                    "selected_product_id": (
                        "aws_stac:"
                        + selected_source["stac_item_id"]
                    ),
                    "tile_repair_status": "stac_point_repaired",
                    "tile_margin_pixels": margin,
                    "center_offset_pixels": center_offset,
                }
            )

        if not newly_resolved:
            mosaics: list[dict[str, Any]] = []
            for task in still_unresolved:
                for item in point_items(
                    wanted,
                    task["longitude"],
                    task["latitude"],
                    retries,
                    epsilon=0.12,
                ):
                    item_id = clean(item.get("id"))
                    candidate_name = clean(
                        item.get("properties", {}).get(
                            "s2:product_uri"
                        )
                    )
                    if (
                        item_id
                        and product_score(wanted, candidate_name)
                        is not None
                    ):
                        candidate_items[item_id] = item
                try:
                    mosaics.append(
                        mosaic_record(
                            wanted,
                            task,
                            list(candidate_items.values()),
                        )
                    )
                except RuntimeError as exc:
                    cdse_repair = cdse_point_record(
                        wanted,
                        task,
                        retries,
                    )
                    if cdse_repair is not None:
                        mosaics.append(cdse_repair)
                    else:
                        mosaics.append(
                            strict_original_record(task, str(exc))
                        )
            records.extend(mosaics)
            unresolved = []
            break
        records.extend(newly_resolved)
        unresolved = still_unresolved
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--existing-roots",
        default=",".join(DEFAULT_EXISTING_ROOTS),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--item-max-index", type=int, default=12)
    parser.add_argument("--request-retries", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--http-cache",
        default="/diniuvol/yuyao/s2_stac_resolver_cache",
    )
    parser.add_argument("--force-stac-mosaic", action="store_true")
    return parser.parse_args()


def main() -> int:
    global _http_cache_root
    global _request_interval
    args = parse_args()
    _request_interval = max(0.0, float(args.min_request_interval))
    _http_cache_root = Path(args.http_cache)
    _http_cache_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input_csv, low_memory=False)
    roots = tuple(
        Path(value.strip())
        for value in args.existing_roots.split(",")
        if value.strip()
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    skipped_missing_product = 0
    for index, row in frame.iterrows():
        for timepoint in TIMEPOINTS:
            product_name = clean(row.get(f"{timepoint}_product_name"))
            product_id = clean(row.get(f"{timepoint}_product_id"))
            if not product_name or not product_id:
                skipped_missing_product += 1
                continue
            task = {
                "row_index": int(index),
                "plume_id": clean(row.get("plume_id")),
                "timepoint": timepoint,
                "product_name": product_name,
                "product_id": product_id,
                "longitude": float(row["plume_longitude"]),
                "latitude": float(row["plume_latitude"]),
            }
            groups.setdefault(product_name, []).append(task)

    started = time.time()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                resolve_group,
                product_name,
                tasks,
                roots,
                args.item_max_index,
                args.request_retries,
                args.force_stac_mosaic,
            ): product_name
            for product_name, tasks in groups.items()
        }
        for future in as_completed(futures):
            product_name = futures[future]
            completed += 1
            try:
                records.extend(future.result())
            except Exception as exc:
                failure = {
                    "product_name": product_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                print(
                    "FAIL "
                    f"product={failure['product_name']} "
                    f"error={failure['error']}",
                    flush=True,
                )
            if (
                completed % args.progress_every == 0
                or completed == len(groups)
            ):
                elapsed = time.time() - started
                print(
                    f"groups={completed}/{len(groups)} "
                    f"records={len(records)} failures={len(failures)} "
                    f"elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )

    if failures:
        report = {
            "groups": len(groups),
            "records": len(records),
            "failures": failures,
        }
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(
            f"tile resolution failed for {len(failures)} products"
        )

    for timepoint in TIMEPOINTS:
        for suffix in (
            "product_name",
            "product_id",
            "stac_item_id",
            "stac_base_url",
            "stac_cog_base_url",
            "stac_dn_add",
            "stac_source_product_name",
            "stac_mosaic_json",
            "local_mosaic_json",
            "cdse_mosaic_json",
            "tile_repair_status",
            "tile_repair_note",
            "tile_margin_pixels",
            "center_offset_pixels",
            "tile_repair_original_product_name",
            "tile_repair_original_product_id",
        ):
            column = f"{timepoint}_{suffix}"
            if column not in frame.columns:
                frame[column] = ""
            frame[column] = frame[column].astype(object)

    repaired = 0
    status_counts: dict[str, int] = {}
    margins: list[float] = []
    center_offsets: list[float] = []
    for record in records:
        index = record["row_index"]
        timepoint = record["timepoint"]
        original_name = record["product_name"]
        original_id = record["product_id"]
        selected_name = record["selected_product_name"]
        selected_id = record["selected_product_id"]
        frame.at[
            index,
            f"{timepoint}_tile_repair_original_product_name",
        ] = original_name
        frame.at[
            index,
            f"{timepoint}_tile_repair_original_product_id",
        ] = original_id
        frame.at[index, f"{timepoint}_product_name"] = selected_name
        frame.at[index, f"{timepoint}_product_id"] = selected_id
        for suffix in (
            "stac_item_id",
            "stac_base_url",
            "stac_cog_base_url",
            "stac_dn_add",
            "stac_source_product_name",
            "stac_mosaic_json",
            "local_mosaic_json",
            "cdse_mosaic_json",
            "tile_repair_status",
            "tile_repair_note",
            "tile_margin_pixels",
            "center_offset_pixels",
        ):
            frame.at[index, f"{timepoint}_{suffix}"] = record.get(
                suffix,
                "",
            )
        repaired += int(selected_name != original_name)
        status = str(record["tile_repair_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        margins.append(float(record["tile_margin_pixels"]))
        center_offsets.append(float(record["center_offset_pixels"]))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        output_path.name + f".tmp.{os.getpid()}"
    )
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output_path)

    report = {
        "rows": len(frame),
        "tasks": len(records),
        "groups": len(groups),
        "skipped_missing_product": skipped_missing_product,
        "repaired_tasks": repaired,
        "status_counts": status_counts,
        "minimum_tile_margin_pixels": min(margins, default=None),
        "maximum_center_offset_pixels": max(
            center_offsets,
            default=None,
        ),
        "elapsed_seconds": time.time() - started,
        "failures": [],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
