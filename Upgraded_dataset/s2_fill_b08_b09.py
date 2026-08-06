#!/usr/bin/env python3

"""Restore canonical S2 channels 7:10 as B08, B8A, and B09."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import requests
import tifffile
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from pyproj import Transformer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgraded_dataset.s2_exact_point_recrop import (
    EARTH_SEARCH_ITEM_URL,
    apply_dn_add,
    cdse_nodes_base_url,
    cdse_nodes_request,
    close_cdse_response,
    download_cdse_node_band,
    earth_search_r20m_source,
    http_session,
    load_cdse_credentials,
    load_config,
    load_legacy_s2,
    start_tokens,
)
from Upgraded_dataset.s2_resolve_point_covering_tiles import (
    CDSE_PRODUCTS_URL,
    item_cog_base_url,
    item_dn_add,
)


TIMEPOINTS = {
    "t0": ("s2_0_std_512", "t0_input_kind"),
    "prev1": ("s2_-7_std_512", "prev1_input_kind"),
    "prev2": ("s2_prev2_std_512", "prev2_input_kind"),
    "prev3": ("s2_prev3_std_512", "prev3_input_kind"),
    "seasonal": ("s2_-90_std_512", "seasonal_input_kind"),
    "year": ("s2_-360_std_512", "year_input_kind"),
}
REPAIR_KINDS = {
    "point_exact_v3_harmonized",
    "local_safe_point_recrop_v4",
}
MODEL_BAND_ORDER = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)
BANDS = tuple(
    (band_name, MODEL_BAND_ORDER.index(band_name))
    for band_name in ("B08", "B8A", "B09")
)
CDSE_PRODUCT_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@dataclass(frozen=True)
class RepairTask:
    plume_id: str
    timepoint: str
    target: Path
    sidecar: Path
    metadata: dict[str, Any]
    product_name: str
    product_id: str
    item_id: str
    cog_base_url: str
    dn_add: int | None
    plume_latitude: float
    plume_longitude: float


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def to_chw(array: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"expected 3D TIFF, got {array.shape}")
    if array.shape[0] in {10, 12, 13}:
        return array, False
    if array.shape[-1] in {10, 12, 13}:
        return np.transpose(array, (2, 0, 1)), True
    raise ValueError(f"cannot identify channel axis: {array.shape}")


def bands_are_complete(path: Path) -> bool:
    try:
        with tifffile.TiffFile(path) as tiff:
            if len(tiff.pages) > 9:
                b08 = tiff.pages[7].asarray()
                b8a = tiff.pages[8].asarray()
                b09 = tiff.pages[9].asarray()
                return (
                    bool(np.any(b08))
                    and bool(np.any(b8a))
                    and bool(np.any(b09))
                )
            array, _ = to_chw(tiff.asarray())
            return (
                array.shape[0] > 9
                and bool(np.any(array[7]))
                and bool(np.any(array[8]))
                and bool(np.any(array[9]))
            )
    except Exception:
        return False


def read_sidecar(task: dict[str, Any]) -> RepairTask:
    plume_id = task["plume_id"]
    timepoint = task["timepoint"]
    target = task["target"]
    sidecar = target.with_name(target.name + ".georef.json")
    metadata = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    fallback = task["fallback"]
    product_name = clean(
        metadata.get("canonical_s2_band_source_product_name")
        or metadata.get("b8a_b09_source_product_name")
        or metadata.get("b08_b09_source_product_name")
        or metadata.get("source_product_name")
        or metadata.get("product_name")
        or fallback.get(f"{timepoint}_stac_source_product_name")
        or fallback.get(f"{timepoint}_product_name")
        or fallback.get(f"{timepoint}_tile_repair_original_product_name")
    )
    if not product_name:
        raise ValueError(f"missing product name: {sidecar}")
    product_id = ""
    product_pairs = (
        (
            clean(metadata.get("canonical_s2_band_source_product_name")),
            clean(metadata.get("canonical_s2_band_source_product_id")),
        ),
        (
            clean(metadata.get("b8a_b09_source_product_name")),
            clean(metadata.get("b8a_b09_source_product_id")),
        ),
        (
            clean(metadata.get("b08_b09_source_product_name")),
            clean(metadata.get("b08_b09_source_product_id")),
        ),
        (
            clean(metadata.get("source_product_name")),
            clean(metadata.get("source_product_id")),
        ),
        (
            clean(metadata.get("product_name")),
            clean(metadata.get("product_id")),
        ),
        (
            clean(fallback.get(f"{timepoint}_stac_source_product_name")),
            clean(fallback.get(f"{timepoint}_product_id")),
        ),
        (
            clean(fallback.get(f"{timepoint}_product_name")),
            clean(fallback.get(f"{timepoint}_product_id")),
        ),
        (
            clean(fallback.get(f"{timepoint}_tile_repair_original_product_name")),
            clean(fallback.get(f"{timepoint}_tile_repair_original_product_id")),
        ),
    )
    for candidate_name, candidate_id in product_pairs:
        if (
            candidate_name == product_name
            and CDSE_PRODUCT_ID_RE.fullmatch(candidate_id)
        ):
            product_id = candidate_id
            break
    item_id = clean(
        metadata.get("canonical_s2_band_source_item_id")
        or metadata.get("b8a_b09_source_item_id")
        or metadata.get("b08_b09_source_item_id")
        or metadata.get("source_item_id")
        or fallback.get(f"{timepoint}_stac_item_id")
    )
    cog_base_url = clean(
        metadata.get("canonical_s2_band_cog_base_url")
        or metadata.get("b8a_b09_cog_base_url")
        or metadata.get("b08_b09_cog_base_url")
    )
    reference_band = clean(metadata.get("reference_band"))
    if (
        not cog_base_url
        and reference_band.startswith("https://")
        and reference_band.endswith(".tif")
    ):
        cog_base_url = reference_band.rsplit("/", 1)[0]
    if not cog_base_url:
        cog_base_url = clean(fallback.get(f"{timepoint}_stac_cog_base_url"))
    raw_dn_add = (
        metadata.get("canonical_s2_band_dn_add")
        if metadata.get("canonical_s2_band_dn_add") is not None
        else metadata.get("b8a_b09_dn_add")
        if metadata.get("b8a_b09_dn_add") is not None
        else metadata.get("b08_b09_dn_add")
        if metadata.get("b08_b09_dn_add") is not None
        else metadata.get("source_dn_add")
    )
    try:
        dn_add = int(raw_dn_add) if raw_dn_add is not None else None
    except (TypeError, ValueError):
        dn_add = None
    return RepairTask(
        plume_id=plume_id,
        timepoint=timepoint,
        target=target,
        sidecar=sidecar,
        metadata=metadata,
        product_name=product_name,
        product_id=product_id,
        item_id=item_id,
        cog_base_url=cog_base_url,
        dn_add=dn_add,
        plume_latitude=float(task["plume_latitude"]),
        plume_longitude=float(task["plume_longitude"]),
    )


def load_metadata_lookup(paths: str) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for value in paths.split(","):
        path = Path(value.strip())
        if not path.is_file():
            continue
        frame = pd.read_csv(path, low_memory=False)
        for row in frame.to_dict("records"):
            plume_id = clean(row.get("plume_id"))
            if not plume_id:
                continue
            merged = lookup.setdefault(plume_id, {})
            for key, value in row.items():
                if clean(value) and not clean(merged.get(key)):
                    merged[key] = value
    return lookup


def discover_tasks(args: argparse.Namespace) -> tuple[list[RepairTask], dict[str, int]]:
    frame = pd.read_csv(args.manifest, low_memory=False)
    metadata_lookup = load_metadata_lookup(args.metadata_tables)
    forced_targets: set[tuple[str, str]] = set()
    if args.force_targets_json:
        raw_targets = json.loads(Path(args.force_targets_json).read_text())
        forced_targets = {
            (clean(item[0]), clean(item[1]))
            for item in raw_targets
            if isinstance(item, list) and len(item) >= 2
        }
    skipped_targets: set[tuple[str, str]] = set()
    if args.skip_targets_json:
        raw_targets = json.loads(Path(args.skip_targets_json).read_text())
        skipped_targets = {
            (clean(item[0]), clean(item[1]))
            for item in raw_targets
            if isinstance(item, list) and len(item) >= 2
        }
    requested_timepoints = [
        value.strip() for value in args.timepoints.split(",") if value.strip()
    ]
    candidates: list[dict[str, Any]] = []
    counts = {
        "rows": int(len(frame)),
        "candidate_targets": 0,
        "already_complete": 0,
        "unsupported_kind": 0,
        "missing_target": 0,
        "sidecar_failures": 0,
        "sidecar_failure_targets": [],
        "preexisting_complete": 0,
        "explicitly_skipped": 0,
    }
    for row in frame.to_dict("records"):
        plume_id = clean(row.get("plume_id"))
        fallback = metadata_lookup.get(plume_id, {})
        for timepoint in requested_timepoints:
            path_column, kind_column = TIMEPOINTS[timepoint]
            if clean(row.get(kind_column)) not in REPAIR_KINDS:
                counts["unsupported_kind"] += 1
                continue
            target = Path(clean(row.get(path_column)))
            if not args.trust_target_paths and not target.is_file():
                counts["missing_target"] += 1
                continue
            counts["candidate_targets"] += 1
            candidates.append(
                {
                    "plume_id": plume_id,
                    "timepoint": timepoint,
                    "target": target,
                    "plume_latitude": row.get("plume_latitude", row.get("latitude")),
                    "plume_longitude": row.get(
                        "plume_longitude", row.get("longitude")
                    ),
                    "fallback": fallback,
                }
            )
    if args.limit_tasks > 0:
        candidates = candidates[: args.limit_tasks]

    tasks: list[RepairTask] = []
    failure_examples: list[str] = []

    def inspect_candidate(
        candidate: dict[str, Any],
    ) -> tuple[str, RepairTask | None]:
        forced = (candidate["plume_id"], candidate["timepoint"]) in forced_targets
        if (candidate["plume_id"], candidate["timepoint"]) in skipped_targets:
            return "skip", None
        sidecar = candidate["target"].with_name(
            candidate["target"].name + ".georef.json"
        )
        try:
            metadata = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        except Exception:
            metadata = {}
        if (
            not args.overwrite
            and not forced
            and bool(metadata.get("canonical_s2_band_fill_complete"))
        ):
            return "marker", None
        if (
            not args.overwrite
            and not forced
            and args.preserve_existing_bands
            and bands_are_complete(candidate["target"])
        ):
            return "preexisting", None
        return "task", read_sidecar(candidate)

    with ThreadPoolExecutor(max_workers=max(1, args.discovery_workers)) as executor:
        future_map = {
            executor.submit(inspect_candidate, candidate): candidate
            for candidate in candidates
        }
        discovery_started = time.monotonic()
        for discovery_index, future in enumerate(
            as_completed(future_map),
            start=1,
        ):
            try:
                status, task = future.result()
                if status == "marker":
                    counts["already_complete"] += 1
                elif status == "preexisting":
                    counts["already_complete"] += 1
                    counts["preexisting_complete"] += 1
                elif status == "skip":
                    counts["already_complete"] += 1
                    counts["explicitly_skipped"] += 1
                elif task is not None:
                    tasks.append(task)
            except Exception as exc:
                counts["sidecar_failures"] += 1
                candidate = future_map[future]
                counts["sidecar_failure_targets"].append(
                    [
                        candidate["plume_id"],
                        candidate["timepoint"],
                        str(candidate["target"]),
                        f"{type(exc).__name__}: {exc}",
                    ]
                )
                if len(failure_examples) < 20:
                    failure_examples.append(
                        f"{candidate['plume_id']} {candidate['timepoint']}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            if (
                discovery_index % max(1, args.discovery_progress_every) == 0
                or discovery_index == len(future_map)
            ):
                elapsed = max(time.monotonic() - discovery_started, 1e-6)
                rate = discovery_index / elapsed
                remaining = (len(future_map) - discovery_index) / max(rate, 1e-6)
                print(
                    f"[Discover] checked={discovery_index}/{len(future_map)} "
                    f"preexisting={counts['preexisting_complete']} "
                    f"marked={counts['already_complete'] - counts['preexisting_complete']} "
                    f"failures={counts['sidecar_failures']} "
                    f"rate={rate:.1f}/s eta_min={remaining / 60:.1f}",
                    flush=True,
                )
    if failure_examples:
        print(
            "[Discover] failure examples:\n  "
            + "\n  ".join(failure_examples),
            flush=True,
        )
    tasks.sort(key=lambda task: (task.product_name, task.plume_id, task.timepoint))
    return tasks, counts


def fetch_item(item_id: str, args: argparse.Namespace) -> dict[str, Any]:
    response = http_session().get(
        EARTH_SEARCH_ITEM_URL.format(item_id=item_id),
        timeout=float(args.request_timeout),
    )
    response.raise_for_status()
    return response.json()


def resolve_source(
    product_name: str,
    group: list[RepairTask],
    args: argparse.Namespace,
) -> dict[str, Any]:
    direct_cog = next((task.cog_base_url for task in group if task.cog_base_url), "")
    direct_item = next((task.item_id for task in group if task.item_id), "")
    direct_dn = next((task.dn_add for task in group if task.dn_add is not None), None)
    if direct_cog and direct_dn is not None:
        return {
            "cog_base_url": direct_cog,
            "item_id": direct_item,
            "source_product_name": product_name,
            "dn_add": int(direct_dn),
            "exact_product": True,
        }

    if direct_item:
        item = fetch_item(direct_item, args)
        source_product_name = clean(
            item.get("properties", {}).get("s2:product_uri")
        )
        return {
            "cog_base_url": item_cog_base_url(item),
            "item_id": direct_item,
            "source_product_name": source_product_name,
            "dn_add": item_dn_add(item),
            "exact_product": source_product_name == product_name,
        }

    resolver_args = argparse.Namespace(
        aws_item_max_index=args.item_max_index,
        aws_read_retries=args.read_retries,
        aws_request_timeout=args.request_timeout,
    )
    source = earth_search_r20m_source(product_name, resolver_args)
    item = fetch_item(source["item_id"], args)
    return {
        "cog_base_url": item_cog_base_url(item),
        "item_id": source["item_id"],
        "source_product_name": source["source_product_name"],
        "dn_add": item_dn_add(item),
        "exact_product": bool(source["exact_product"]),
    }


def atomic_write_tiff(
    target: Path,
    array: np.ndarray,
    cache_dir: Path,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tif"
    )
    remote_temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.part"
    )
    try:
        tifffile.imwrite(cache_path, array)
        shutil.copyfile(cache_path, remote_temporary)
        os.replace(remote_temporary, target)
    finally:
        cache_path.unlink(missing_ok=True)
        remote_temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.part"
    )
    try:
        temporary.write_text(json.dumps(data, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_band_for_task(
    dataset: rasterio.DatasetReader,
    reference_dataset: rasterio.DatasetReader,
    task: RepairTask,
    dn_add: int,
    resampling: Resampling,
) -> np.ndarray:
    transform_values = task.metadata.get("transform")
    crs_wkt = clean(task.metadata.get("crs_wkt"))
    if isinstance(transform_values, list) and len(transform_values) >= 6 and crs_wkt:
        transform = Affine(*[float(value) for value in transform_values[:6]])
        target_crs = crs_wkt
        height = int(task.metadata.get("height", 512))
        width = int(task.metadata.get("width", 512))
    else:
        reference_transform = reference_dataset.transform
        if not np.isclose(abs(reference_transform.a), 20.0) or not np.isclose(
            abs(reference_transform.e), 20.0
        ):
            reference_transform = Affine(
                20.0,
                0.0,
                reference_transform.c,
                0.0,
                -20.0,
                reference_transform.f,
            )
        transformer = Transformer.from_crs(
            "EPSG:4326",
            reference_dataset.crs,
            always_xy=True,
        )
        x, y = transformer.transform(task.plume_longitude, task.plume_latitude)
        column, row = (~reference_transform) * (x, y)
        window = Window(
            int(np.floor(column - 256)),
            int(np.floor(row - 256)),
            512,
            512,
        )
        transform = rasterio.windows.transform(
            window,
            reference_transform,
        )
        target_crs = reference_dataset.crs
        height = 512
        width = 512
    with WarpedVRT(
        dataset,
        crs=target_crs,
        transform=transform,
        width=width,
        height=height,
        src_nodata=0,
        nodata=0,
        resampling=resampling,
    ) as warped:
        image = warped.read(1)
    return apply_dn_add(image, dn_add)


def cdse_b8a_b09_node_paths(
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
    patterns = {
        "B08": re.compile(r"/R10m/.*_B08_10m$"),
        "B8A": re.compile(r"/R20m/.*_B8A_20m$"),
        "B09": re.compile(r"/R60m/.*_B09_60m$"),
    }
    for element in root.iter():
        if "IMAGE_FILE" not in element.tag.upper():
            continue
        value = clean(element.text)
        for band_name, pattern in patterns.items():
            if pattern.search(value):
                paths[band_name] = (value + ".jp2").split("/")
    missing = sorted(set(patterns) - set(paths))
    if missing:
        raise RuntimeError(
            f"MTD_MSIL2A.xml missing bands {missing}: {product_name}"
        )
    return paths


def stage_cdse_b8a_b09(
    token: Any,
    args: argparse.Namespace,
    product_id: str,
    product_name: str,
) -> dict[str, Path]:
    destination = Path(args.cdse_cache_dir) / product_name
    marker = destination / ".canonical_nir_bands_complete"
    existing = {
        band_name: next(destination.glob(f"*_B{band_name[1:]}_*m.jp2"), None)
        for band_name, _ in BANDS
    }
    if marker.is_file() and all(
        path is not None and path.is_file() and path.stat().st_size > 0
        for path in existing.values()
    ):
        return {key: value for key, value in existing.items() if value is not None}
    destination.mkdir(parents=True, exist_ok=True)
    node_paths = cdse_b8a_b09_node_paths(
        token,
        args,
        product_id,
        product_name,
    )
    base_url = cdse_nodes_base_url(product_id, product_name)
    paths = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_map = {}
        for band_name, _ in BANDS:
            node_parts = node_paths[band_name]
            destination_path = destination / node_parts[-1]
            future = executor.submit(
                download_cdse_node_band,
                token,
                args,
                base_url,
                node_parts,
                destination_path,
            )
            future_map[future] = band_name
        for future in as_completed(future_map):
            paths[future_map[future]] = future.result()
    marker.write_text("complete\n")
    return paths


def resolve_cdse_product_id(
    product_name: str,
    args: argparse.Namespace,
) -> str:
    response = requests.get(
        CDSE_PRODUCTS_URL,
        params={
            "$filter": f"Name eq '{product_name}'",
            "$select": "Id,Name",
            "$top": 2,
        },
        timeout=float(args.request_timeout),
    )
    response.raise_for_status()
    matches = [
        clean(item.get("Id"))
        for item in response.json().get("value", [])
        if clean(item.get("Name")) == product_name
        and CDSE_PRODUCT_ID_RE.fullmatch(clean(item.get("Id")))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"CDSE catalogue returned {len(matches)} exact IDs for {product_name}"
        )
    return matches[0]


def repair_cdse_product_group(
    product_name: str,
    group: list[RepairTask],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    product_ids = {task.product_id for task in group if task.product_id}
    if not product_ids:
        product_ids = {resolve_cdse_product_id(product_name, args)}
    if len(product_ids) != 1:
        raise RuntimeError(
            f"missing/ambiguous CDSE product id for {product_name}: "
            f"{sorted(product_ids)}"
        )
    product_id = next(iter(product_ids))
    token = args.cdse_tokens[
        sum(product_name.encode("utf-8")) % len(args.cdse_tokens)
    ]
    band_paths = stage_cdse_b8a_b09(
        token,
        args,
        product_id,
        product_name,
    )
    results = []
    with (
        rasterio.open(band_paths["B08"]) as b08_dataset,
        rasterio.open(band_paths["B8A"]) as b8a_dataset,
        rasterio.open(band_paths["B09"]) as b09_dataset,
    ):
        for task in group:
            b08 = read_band_for_task(
                b08_dataset,
                b8a_dataset,
                task,
                -1000,
                Resampling.nearest,
            )
            b8a = read_band_for_task(
                b8a_dataset,
                b8a_dataset,
                task,
                -1000,
                Resampling.nearest,
            )
            b09 = read_band_for_task(
                b09_dataset,
                b8a_dataset,
                task,
                -1000,
                Resampling.nearest,
            )
            if not np.any(b08) or not np.any(b8a) or not np.any(b09):
                raise RuntimeError(
                    f"empty exact CDSE band for {task.plume_id} {task.timepoint}"
                )
            original = tifffile.imread(task.target)
            channels, was_hwc = to_chw(original)
            channels = channels.copy()
            channels[7] = b08.astype(channels.dtype, copy=False)
            channels[8] = b8a.astype(channels.dtype, copy=False)
            channels[9] = b09.astype(channels.dtype, copy=False)
            output = np.transpose(channels, (1, 2, 0)) if was_hwc else channels
            atomic_write_tiff(task.target, output, Path(args.cache_dir))
            metadata = dict(task.metadata)
            metadata.update(
                {
                    "canonical_s2_band_fill_complete": True,
                    "canonical_s2_band_source": "cdse_nodes_exact",
                    "canonical_s2_band_source_product_id": product_id,
                    "canonical_s2_band_source_product_name": product_name,
                    "canonical_s2_band_dn_add": -1000,
                    "b08_resampling": "nearest",
                    "b8a_resampling": "nearest",
                    "b09_resampling": "nearest",
                }
            )
            atomic_write_json(task.sidecar, metadata)
            results.append(
                {
                    "plume_id": task.plume_id,
                    "timepoint": task.timepoint,
                    "target": str(task.target),
                    "status": "repaired",
                    "product_name": product_name,
                    "source_product_name": product_name,
                    "source_item_id": "",
                    "exact_product": True,
                    "dn_add": -1000,
                    "b08_nonzero": float((b08 != 0).mean()),
                    "b8a_nonzero": float((b8a != 0).mean()),
                    "b09_nonzero": float((b09 != 0).mean()),
                }
            )
    return results


def repair_product_group(
    product_name: str,
    group: list[RepairTask],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.source_mode == "cdse":
        return repair_cdse_product_group(product_name, group, args)
    source = resolve_source(product_name, group, args)
    if args.require_exact_product and not source["exact_product"]:
        raise RuntimeError(
            f"public source is not exact: wanted={product_name} "
            f"found={source['source_product_name']}"
        )
    band_urls = {
        band_name: f"{source['cog_base_url']}/{band_name}.tif"
        for band_name, _ in BANDS
    }
    reference_url = f"{source['cog_base_url']}/B05.tif"
    results = []
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MAX_RETRY=str(args.read_retries),
        GDAL_HTTP_RETRY_DELAY="1",
        GDAL_HTTP_TIMEOUT=str(args.request_timeout),
        VSI_CACHE="FALSE",
    ):
        with (
            rasterio.open(reference_url) as reference_dataset,
            rasterio.open(band_urls["B08"]) as b08_dataset,
            rasterio.open(band_urls["B8A"]) as b8a_dataset,
            rasterio.open(band_urls["B09"]) as b09_dataset,
        ):
            for task in group:
                b08 = read_band_for_task(
                    b08_dataset,
                    reference_dataset,
                    task,
                    int(source["dn_add"]),
                    Resampling.nearest,
                )
                b8a = read_band_for_task(
                    b8a_dataset,
                    reference_dataset,
                    task,
                    int(source["dn_add"]),
                    Resampling.nearest,
                )
                b09 = read_band_for_task(
                    b09_dataset,
                    reference_dataset,
                    task,
                    int(source["dn_add"]),
                    Resampling.nearest,
                )
                if not np.any(b08) or not np.any(b8a) or not np.any(b09):
                    raise RuntimeError(
                        f"empty repaired band for {task.plume_id} {task.timepoint}"
                    )
                original = tifffile.imread(task.target)
                channels, was_hwc = to_chw(original)
                channels = channels.copy()
                channels[7] = b08.astype(channels.dtype, copy=False)
                channels[8] = b8a.astype(channels.dtype, copy=False)
                channels[9] = b09.astype(channels.dtype, copy=False)
                output = np.transpose(channels, (1, 2, 0)) if was_hwc else channels
                atomic_write_tiff(
                    task.target,
                    output,
                    Path(args.cache_dir),
                )
                metadata = dict(task.metadata)
                metadata.update(
                    {
                        "canonical_s2_band_fill_complete": True,
                        "canonical_s2_band_source_item_id": source["item_id"],
                        "canonical_s2_band_source_product_name": source[
                            "source_product_name"
                        ],
                        "canonical_s2_band_cog_base_url": source["cog_base_url"],
                        "canonical_s2_band_dn_add": int(source["dn_add"]),
                        "b08_resampling": "nearest",
                        "b8a_resampling": "nearest",
                        "b09_resampling": "nearest",
                    }
                )
                atomic_write_json(task.sidecar, metadata)
                results.append(
                    {
                        "plume_id": task.plume_id,
                        "timepoint": task.timepoint,
                        "target": str(task.target),
                        "status": "repaired",
                        "product_name": product_name,
                        "source_product_name": source["source_product_name"],
                        "source_item_id": source["item_id"],
                        "exact_product": bool(source["exact_product"]),
                        "dn_add": int(source["dn_add"]),
                        "b08_nonzero": float((b08 != 0).mean()),
                        "b8a_nonzero": float((b8a != 0).mean()),
                        "b09_nonzero": float((b09 != 0).mean()),
                    }
                )
    return results


def save_progress(
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    args: argparse.Namespace,
    discovery: dict[str, int],
    started: float,
) -> None:
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".part")
    report = {
        "manifest": args.manifest,
        "discovery": discovery,
        "completed": len(records),
        "repaired": sum(record["status"] == "repaired" for record in records),
        "already_complete": sum(
            record["status"] == "already_complete" for record in records
        ),
        "failures": failures,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, report_path)


def main(args: argparse.Namespace) -> int:
    started = time.monotonic()
    args.cdse_tokens = []
    tasks, discovery = discover_tasks(args)
    groups: dict[str, list[RepairTask]] = {}
    for task in tasks:
        groups.setdefault(task.product_name, []).append(task)
    group_items = sorted(groups.items())
    if args.limit_products > 0:
        group_items = group_items[: args.limit_products]
        tasks = [task for _, group in group_items for task in group]
    print(
        f"[Discover] tasks={len(tasks)} products={len(group_items)} "
        f"candidate={discovery['candidate_targets']} "
        f"already={discovery['already_complete']} "
        f"sidecar_failures={discovery['sidecar_failures']}",
        flush=True,
    )
    if args.dry_run or not tasks:
        save_progress([], [], args, discovery, started)
        return 0
    if args.source_mode == "cdse":
        legacy = load_legacy_s2(REPO_ROOT)
        config = load_config(legacy, args.legacy_config)
        credentials = load_cdse_credentials(args, config)
        args.cdse_tokens = start_tokens(legacy, credentials)
        args.node_band_semaphore = threading.BoundedSemaphore(
            max(1, args.node_global_workers)
        )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed_groups = 0
    completed_tasks = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(repair_product_group, product_name, group, args): (
                product_name,
                group,
            )
            for product_name, group in group_items
        }
        for future in as_completed(future_map):
            product_name, group = future_map[future]
            try:
                group_records = future.result()
                records.extend(group_records)
                completed_tasks += len(group_records)
            except Exception as exc:
                failures.append(
                    {
                        "product_name": product_name,
                        "tasks": len(group),
                        "plume_ids": sorted({task.plume_id for task in group}),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                completed_tasks += len(group)
            completed_groups += 1
            if (
                completed_groups % max(1, args.progress_every) == 0
                or completed_groups == len(group_items)
            ):
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = completed_tasks / elapsed
                remaining = (len(tasks) - completed_tasks) / max(rate, 1e-6)
                print(
                    f"[Progress] groups={completed_groups}/{len(group_items)} "
                    f"tasks={completed_tasks}/{len(tasks)} "
                    f"repaired={sum(r['status'] == 'repaired' for r in records)} "
                    f"failed_groups={len(failures)} rate={rate:.2f}/s "
                    f"eta_min={remaining / 60:.1f}",
                    flush=True,
                )
                save_progress(records, failures, args, discovery, started)
    save_progress(records, failures, args, discovery, started)
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--metadata-tables",
        default=(
            "/home/yuyao/methane_train/Upgrade_data_pipeline/csv/"
            "s2_6time_point_covering_tiles_v3.csv,"
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_restore_dropped_resolved_v3.csv,"
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_canonical_6510_recrop_retry_local_v4.csv,"
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_canonical_6510_recrop_tasks_v4.csv"
        ),
    )
    parser.add_argument("--timepoints", default="t0,prev1,prev2,prev3,seasonal,year")
    parser.add_argument("--cache-dir", default="/diniuvol/yuyao/s2_b8a_b09_repair_cache")
    parser.add_argument(
        "--source-mode",
        choices=("public", "cdse"),
        default="public",
    )
    parser.add_argument(
        "--cdse-cache-dir",
        default="/diniuvol/yuyao/s2_b8a_b09_cdse_cache",
    )
    parser.add_argument(
        "--legacy-config",
        default=(
            "/home/yuyao/methane_train/data_preprocess/configs/"
            "carbon_mapper_sentinel2_plume_download.yaml"
        ),
    )
    parser.add_argument("--cdse-env-index", type=int, default=1)
    parser.add_argument("--cdse-username", default="")
    parser.add_argument("--cdse-password", default="")
    parser.add_argument("--auth-retries", type=int, default=3)
    parser.add_argument("--node-request-timeout", type=float, default=300.0)
    parser.add_argument("--node-global-workers", type=int, default=16)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--discovery-workers", type=int, default=32)
    parser.add_argument("--discovery-progress-every", type=int, default=1000)
    parser.add_argument("--item-max-index", type=int, default=12)
    parser.add_argument("--read-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--limit-products", type=int, default=0)
    parser.add_argument(
        "--trust-target-paths",
        action="store_true",
        help="Skip the serial exists() scan for manifests whose source paths were audited.",
    )
    parser.add_argument("--force-targets-json", default="")
    parser.add_argument("--skip-targets-json", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preserve-existing-bands",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-exact-product",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(parser.parse_args()))
