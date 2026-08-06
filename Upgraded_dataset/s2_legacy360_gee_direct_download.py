#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import ee
import pandas as pd
import rasterio
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgraded_dataset.s2_legacy360_gee_export import (
    CANONICAL_FILENAME,
    GEE_BANDS,
    GEE_COLLECTION,
    LEGACY_TIMEPOINTS,
    TIMEPOINT_BY_NAME,
    image_property_str,
    image_time_iso,
    initialize_ee,
    parse_timepoints,
    query_window as legacy_query_window,
    select_closest_image,
    sentinel_collection,
    square_region,
)
from Upgraded_dataset.s2_6time_gee_export import (
    query_window as exact_time_query_window,
)


OUT_FIELDS = [
    "plume_id",
    "timepoint",
    "status",
    "raw_path",
    "collection",
    "bands",
    "query_start_utc",
    "query_end_utc",
    "selected_time_utc",
    "selected_id",
    "cloud_pct",
    "nodata_pct",
    "candidates",
    "attempts",
    "size",
    "shape",
    "message",
]
MANIFEST_LOCK = threading.Lock()
THREAD_LOCAL = threading.local()


def log(message: str) -> None:
    print(f"[Direct GEE] {message}", flush=True)


def session() -> requests.Session:
    value = getattr(THREAD_LOCAL, "session", None)
    if value is None:
        value = requests.Session()
        value.headers.update({"User-Agent": "s2-legacy360-matched-downloader/1.0"})
        THREAD_LOCAL.session = value
    return value


def target_path(raw_root: Path, plume_id: str, timepoint: str) -> Path:
    return (
        raw_root
        / "S2_GEE_6time"
        / timepoint
        / plume_id
        / CANONICAL_FILENAME[timepoint]
    )


def validate_tiff(path: Path, expected_bands: int, minimum_size: int) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size <= 0:
        return False, "missing_or_empty"
    try:
        with rasterio.open(path) as dataset:
            bands = dataset.count
            height = dataset.height
            width = dataset.width
            dtypes = dataset.dtypes
    except Exception as exc:
        return False, f"read_error:{type(exc).__name__}:{exc}"
    if bands != expected_bands or height < minimum_size or width < minimum_size:
        return False, f"shape=({bands},{height},{width})"
    return True, f"shape=({bands},{height},{width}),dtype={dtypes}"


def decode_download(content: bytes, content_type: str) -> bytes:
    if content[:4] == b"PK\x03\x04" or "zip" in content_type.casefold():
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".tif", ".tiff"))
            ]
            if len(names) != 1:
                raise ValueError(f"expected one TIFF in archive, found {names}")
            return archive.read(names[0])
    return content


def append_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_LOCK:
        exists = path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
            if not exists:
                writer.writeheader()
            for record in records:
                writer.writerow({field: record.get(field, "") for field in OUT_FIELDS})


def download_image(
    image,
    region,
    destination: Path,
    scale: float,
    crs: str,
    expected_bands: int,
    expected_size: int,
    retries: int,
) -> tuple[int, int, str]:
    parameters = {
        "name": destination.stem,
        "bands": GEE_BANDS,
        "region": region,
        "scale": scale,
        "crs": crs,
        "filePerBand": False,
        "format": "GEO_TIFF",
    }
    last_message = ""
    for attempt in range(1, retries + 1):
        temporary = destination.with_name(destination.name + ".part")
        try:
            url = image.getDownloadURL(parameters)
            response = session().get(url, timeout=(30, 300))
            response.raise_for_status()
            payload = decode_download(
                response.content,
                response.headers.get("content-type", ""),
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(payload)
            ok, message = validate_tiff(
                temporary,
                expected_bands=expected_bands,
                minimum_size=expected_size,
            )
            if not ok:
                raise ValueError(message)
            os.replace(temporary, destination)
            return attempt, destination.stat().st_size, message
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            last_message = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(last_message)


def process_plume(
    row: dict[str, Any],
    timepoint_names: list[str],
    raw_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    plume_id = str(row["plume_id"]).strip()
    latitude = float(row[args.lat_col])
    longitude = float(row[args.lon_col])
    region = square_region(
        longitude,
        latitude,
        int(args.export_pixels),
        float(args.scale),
    )
    records: list[dict[str, Any]] = []
    for timepoint_name in timepoint_names:
        timepoint = TIMEPOINT_BY_NAME[timepoint_name]
        destination = target_path(raw_root, plume_id, timepoint_name)
        if args.resume:
            ok, validation = validate_tiff(
                destination,
                expected_bands=len(GEE_BANDS),
                minimum_size=int(args.export_pixels),
            )
            if ok:
                records.append(
                    {
                        "plume_id": plume_id,
                        "timepoint": timepoint_name,
                        "status": "skip_existing_valid",
                        "raw_path": str(destination),
                        "collection": GEE_COLLECTION,
                        "bands": ",".join(GEE_BANDS),
                        "size": destination.stat().st_size,
                        "shape": validation,
                    }
                )
                continue

        try:
            if args.selection_mode == "exact_time":
                target, start, end, source = exact_time_query_window(
                    row, timepoint, args
                )
            else:
                target, start, end, source = legacy_query_window(
                    row, timepoint, args
                )
            collection = sentinel_collection(region, start, end, args)
            candidates = int(collection.size().getInfo())
            if candidates <= 0:
                records.append(
                    {
                        "plume_id": plume_id,
                        "timepoint": timepoint_name,
                        "status": "no_image",
                        "raw_path": str(destination),
                        "collection": GEE_COLLECTION,
                        "bands": ",".join(GEE_BANDS),
                        "query_start_utc": start.isoformat(),
                        "query_end_utc": end.isoformat(),
                        "candidates": candidates,
                        "message": source,
                    }
                )
                continue
            image = (
                select_closest_image(collection, target)
                if target is not None
                else ee.Image(collection.first())
            )
            selected_id = image_property_str(image, "system:index")
            selected_time = image_time_iso(image)
            cloud = image_property_str(image, "CLOUDY_PIXEL_PERCENTAGE")
            nodata = image_property_str(image, "NODATA_PIXEL_PERCENTAGE")
            image = (
                image.select(GEE_BANDS)
                .clip(region)
                .reproject(crs=args.crs, scale=float(args.scale))
                .clip(region)
            )
            attempts, size, shape = download_image(
                image,
                region,
                destination,
                scale=float(args.scale),
                crs=args.crs,
                expected_bands=len(GEE_BANDS),
                expected_size=int(args.export_pixels),
                retries=int(args.retries),
            )
            records.append(
                {
                    "plume_id": plume_id,
                    "timepoint": timepoint_name,
                    "status": "downloaded",
                    "raw_path": str(destination),
                    "collection": GEE_COLLECTION,
                    "bands": ",".join(GEE_BANDS),
                    "query_start_utc": start.isoformat(),
                    "query_end_utc": end.isoformat(),
                    "selected_time_utc": selected_time,
                    "selected_id": selected_id,
                    "cloud_pct": cloud,
                    "nodata_pct": nodata,
                    "candidates": candidates,
                    "attempts": attempts,
                    "size": size,
                    "shape": shape,
                    "message": source,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "plume_id": plume_id,
                    "timepoint": timepoint_name,
                    "status": "failed",
                    "raw_path": str(destination),
                    "collection": GEE_COLLECTION,
                    "bands": ",".join(GEE_BANDS),
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    return records


def latest_status_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, low_memory=False)
    latest = frame.drop_duplicates(["plume_id", "timepoint"], keep="last")
    return {
        str(key): int(value)
        for key, value in latest["status"].value_counts().sort_index().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--timepoints", default="t0,seasonal,year")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit-plumes", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--lat-col", default="plume_latitude")
    parser.add_argument("--lon-col", default="plume_longitude")
    parser.add_argument("--cloud-pct", type=float, default=20.0)
    parser.add_argument("--sort-property", default="NODATA_PIXEL_PERCENTAGE")
    parser.add_argument(
        "--selection-mode",
        choices=("legacy_window", "exact_time"),
        default="legacy_window",
    )
    parser.add_argument("--image-time-before-hours", type=float, default=1.0)
    parser.add_argument("--image-time-after-hours", type=float, default=1.0)
    parser.add_argument("--export-pixels", type=int, default=512)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--crs", default="EPSG:4326")
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument("--ee-project", default="")
    args = parser.parse_args()

    initialize_ee(args)
    timepoints = parse_timepoints(args.timepoints)
    unsupported = sorted({timepoint.name for timepoint in timepoints} - LEGACY_TIMEPOINTS)
    if unsupported:
        raise ValueError(f"unsupported timepoints: {unsupported}")
    names = [timepoint.name for timepoint in timepoints]
    frame = pd.read_csv(args.input_csv, low_memory=False)
    if args.limit_plumes > 0:
        frame = frame.head(args.limit_plumes)
    rows = frame.to_dict("records")
    raw_root = Path(args.raw_root)
    manifest = Path(args.manifest)
    completed_plumes = 0
    completed_jobs = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(process_plume, row, names, raw_root, args): str(row["plume_id"])
            for row in rows
        }
        for future in as_completed(futures):
            records = future.result()
            append_records(manifest, records)
            completed_plumes += 1
            completed_jobs += len(records)
            if (
                completed_plumes % max(1, args.progress_every) == 0
                or completed_plumes == len(futures)
            ):
                log(
                    f"plumes={completed_plumes}/{len(futures)} "
                    f"jobs={completed_jobs}/{len(futures) * len(names)} "
                    f"latest_status={latest_status_counts(manifest)}"
                )

    summary = {
        "input_csv": args.input_csv,
        "raw_root": args.raw_root,
        "manifest": args.manifest,
        "plumes": len(rows),
        "timepoints": names,
        "latest_status": latest_status_counts(manifest),
    }
    manifest.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
