#!/usr/bin/env python3
"""Submit six-time Sentinel-2 exports from Google Earth Engine.

This is a 6-time wrapper around the old GEE Sentinel-2 export flow in
``data_preprocess/gee_download_3_90_365_l2a.py``.  It intentionally uses
``COPERNICUS/S2_SR_HARMONIZED`` and exports the same 12-band order that the
old final datasets used:

    B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12

The input is the new S2 six-time table.  For each row and timepoint, the
script queries GEE around the table's ``*_image_time`` and submits one Drive
export.  It writes a local manifest so reruns can skip already submitted
(plume_id, timepoint) pairs.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    import ee
except ImportError as exc:  # pragma: no cover - runtime environment issue
    raise SystemExit("earthengine-api is required: pip install earthengine-api") from exc

try:
    import geemap  # noqa: F401  # kept for parity with the old GEE script environment
except ImportError:
    geemap = None


REPO_ROOT = Path("/home/yuyao/methane_train")
DEFAULT_INPUT_CSV = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_all6_available_paths_std512_complete.csv"
DEFAULT_OUT_MANIFEST = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_gee_export_manifest.csv"
DEFAULT_DRIVE_FOLDER = "CM_S2_L2A_6TIME_GEE"
DEFAULT_RAW_ROOT = "/mnt/engg-niulab/yuyao/sensors_raw_data"

GEE_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
GEE_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]


@dataclass(frozen=True)
class Timepoint:
    name: str
    image_time_col: str
    filename_suffix: str
    fallback_offset_days: int
    fallback_window_before_days: int
    fallback_window_after_days: int


TIMEPOINTS = [
    Timepoint("t0", "t0_image_time", "s2_0", 0, 1, 1),
    Timepoint("prev1", "prev1_image_time", "s2_prev1", -7, 10, 1),
    Timepoint("prev2", "prev2_image_time", "s2_prev2", -14, 20, 1),
    Timepoint("prev3", "prev3_image_time", "s2_prev3", -21, 30, 1),
    Timepoint("seasonal", "seasonal_image_time", "s2_seasonal", -90, 60, 1),
    Timepoint("year", "year_image_time", "s2_year", -365, 90, 1),
]
TIMEPOINT_BY_NAME = {tp.name: tp for tp in TIMEPOINTS}
EXPANDED_WINDOW_DAYS = {
    "t0": 3,
    "prev1": 15,
    "prev2": 20,
    "prev3": 30,
    "seasonal": 60,
    "year": 90,
}
CANONICAL_FILENAME = {
    "t0": "s2_0.tif",
    "prev1": "s2_prev1.tif",
    "prev2": "s2_prev2.tif",
    "prev3": "s2_prev3.tif",
    "seasonal": "s2_seasonal.tif",
    "year": "s2_year.tif",
}

OUT_FIELDS = [
    "plume_id",
    "timepoint",
    "status",
    "task_id",
    "task_state",
    "drive_folder",
    "file_prefix",
    "collection",
    "bands",
    "target_time_utc",
    "query_start_utc",
    "query_end_utc",
    "selected_time_utc",
    "selected_id",
    "cloud_pct",
    "nodata_pct",
    "plume_latitude",
    "plume_longitude",
    "message",
]


_manifest_lock = threading.Lock()
_task_queue_lock = threading.Lock()
_last_task_poll = 0.0
_last_active_count: Optional[int] = None
_drive_backpressure_lock = threading.Lock()
_drive_backpressure_service: Any = None
_drive_backpressure_folder_id = ""
_last_drive_file_poll = 0.0
_last_drive_file_count: Optional[int] = None


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def parse_utc(value: Any) -> Optional[datetime]:
    if not has_value(value):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ee_date(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def safe_component(value: Any, max_len: int = 180) -> str:
    text = str(value or "").strip()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    clean = "".join(out).strip("_")
    return (clean or "missing")[:max_len]


def parse_timepoints(value: str) -> list[Timepoint]:
    names = [x.strip() for x in value.split(",") if x.strip()]
    if not names or names == ["all"]:
        return list(TIMEPOINTS)
    bad = sorted(set(names) - set(TIMEPOINT_BY_NAME))
    if bad:
        raise ValueError(f"Unknown timepoints: {bad}. Valid: {sorted(TIMEPOINT_BY_NAME)}")
    return [TIMEPOINT_BY_NAME[name] for name in names]


def load_done_keys(
    path: Path,
    resume: bool,
    task_states: Optional[dict[str, str]] = None,
    dry_run: bool = False,
) -> set[tuple[str, str]]:
    if not resume or not path.exists():
        return set()
    latest: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            plume_id = str(row.get("plume_id", "")).strip()
            tp = str(row.get("timepoint", "")).strip()
            if plume_id and tp:
                latest[(plume_id, tp)] = row

    done: set[tuple[str, str]] = set()
    for key, row in latest.items():
        status = str(row.get("status", "")).strip()
        if status in {"drive_exists", "local_exists"}:
            done.add(key)
            continue
        if status == "dry_run":
            if dry_run:
                done.add(key)
            continue
        if status not in {"submitted", "already_submitted"}:
            continue
        if task_states is None:
            done.add(key)
            continue
        task_id = str(row.get("task_id", "")).strip()
        if task_states.get(task_id) in {"READY", "RUNNING", "COMPLETED"}:
            done.add(key)
    return done


def current_task_states() -> dict[str, str]:
    return {
        str(getattr(task, "id", "")): str(getattr(task, "state", ""))
        for task in ee.batch.Task.list()
        if getattr(task, "id", None)
    }



def file_prefix_for(plume_id: str, tp: Timepoint, args: argparse.Namespace) -> str:
    return safe_component(args.prefix_template.format(plume_id=plume_id, timepoint=tp.name, suffix=tp.filename_suffix))


def local_target_path(raw_root: str, plume_id: str, timepoint: str) -> Path:
    return Path(raw_root) / "S2_GEE_6time" / timepoint / plume_id / CANONICAL_FILENAME[timepoint]


def existing_local_keys(df: pd.DataFrame, timepoints: list[Timepoint], args: argparse.Namespace) -> set[tuple[str, str]]:
    if not args.check_local:
        return set()
    candidates: list[tuple[tuple[str, str], Path]] = []
    for _, row_obj in df.iterrows():
        row = row_obj.to_dict()
        plume_id = str(row.get("plume_id", "")).strip()
        if not plume_id:
            continue
        for tp in timepoints:
            candidates.append(((plume_id, tp.name), local_target_path(args.raw_root, plume_id, tp.name)))

    def _exists(item: tuple[tuple[str, str], Path]) -> Optional[tuple[str, str]]:
        key, path = item
        try:
            return key if path.is_file() and path.stat().st_size > 0 else None
        except OSError:
            return None

    with ThreadPoolExecutor(max_workers=max(1, int(args.resume_check_workers))) as pool:
        return {key for key in pool.map(_exists, candidates) if key is not None}


def build_drive_service(args: argparse.Namespace):
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive deps missing. Install: pip install google-api-python-client google-auth google-auth-oauthlib"
        ) from exc

    scopes = ["https://www.googleapis.com/auth/drive"]
    if args.service_account:
        creds = service_account.Credentials.from_service_account_file(args.credentials, scopes=scopes)
    else:
        creds = None
        if os.path.exists(args.token):
            creds = Credentials.from_authorized_user_file(args.token, scopes)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(args.credentials, scopes)
                creds = flow.run_console()
            Path(args.token).parent.mkdir(parents=True, exist_ok=True)
            with open(args.token, "w") as fh:
                fh.write(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_drive_folder_id(service: Any, folder_id: str, folder_name: str) -> str:
    if folder_id:
        return folder_id
    safe_name = folder_name.replace("'", "\\'")
    query = f"mimeType='application/vnd.google-apps.folder' and name='{safe_name}' and trashed=false"
    resp = service.files().list(q=query, fields="files(id,name,parents)").execute()
    matches = resp.get("files", [])
    if not matches:
        return ""
    if len(matches) > 1:
        names = ", ".join(f"{m['name']}({m['id']})" for m in matches)
        raise RuntimeError(f"multiple Drive folders named {folder_name}: {names}; pass --drive-folder-id")
    return matches[0]["id"]


def drive_file_stem(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".tif"):
        return name[:-4]
    if lower.endswith(".tiff"):
        return name[:-5]
    return name


def load_drive_prefixes(args: argparse.Namespace) -> set[str]:
    if not args.check_drive:
        return set()
    service = build_drive_service(args)
    folder_id = resolve_drive_folder_id(service, args.drive_folder_id, args.drive_folder)
    if not folder_id:
        log(f"Drive folder not found yet: {args.drive_folder}; drive resume keys=0")
        return set()
    prefixes: set[str] = set()
    page_token = None
    query = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id,name,mimeType)"
    while True:
        resp = service.files().list(q=query, fields=fields, pageToken=page_token).execute()
        for item in resp.get("files", []):
            if item.get("mimeType") == "application/vnd.google-apps.folder":
                continue
            stem = drive_file_stem(str(item.get("name", "")))
            if stem:
                prefixes.add(stem)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return prefixes


def existing_drive_keys(df: pd.DataFrame, timepoints: list[Timepoint], prefixes: set[str], args: argparse.Namespace) -> set[tuple[str, str]]:
    if not prefixes:
        return set()
    out: set[tuple[str, str]] = set()
    for _, row_obj in df.iterrows():
        row = row_obj.to_dict()
        plume_id = str(row.get("plume_id", "")).strip()
        if not plume_id:
            continue
        for tp in timepoints:
            prefix = file_prefix_for(plume_id, tp, args)
            if prefix in prefixes or any(name.startswith(prefix + "-") for name in prefixes):
                out.add((plume_id, tp.name))
    return out


def append_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _manifest_lock:
        exists = path.exists()
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in OUT_FIELDS})


def initialize_ee(args: argparse.Namespace) -> None:
    if args.authenticate:
        ee.Authenticate()
    kwargs = {}
    if args.ee_project:
        kwargs["project"] = args.ee_project
    ee.Initialize(**kwargs)


def active_gee_task_count(min_poll_interval_sec: float = 15.0) -> int:
    global _last_task_poll, _last_active_count
    now = time.time()
    with _task_queue_lock:
        if _last_active_count is not None and now - _last_task_poll < min_poll_interval_sec:
            return _last_active_count
        tasks = ee.batch.Task.list()
        active = sum(1 for task in tasks if getattr(task, "state", None) in {"RUNNING", "READY"})
        _last_task_poll = now
        _last_active_count = active
        return active


def wait_for_task_capacity(max_active_tasks: int, poll_sec: float) -> None:
    if max_active_tasks <= 0:
        return
    while True:
        active = active_gee_task_count()
        if active < max_active_tasks:
            return
        log(f"GEE queue active={active} >= {max_active_tasks}; sleeping {poll_sec:.0f}s")
        time.sleep(poll_sec)
        with _task_queue_lock:
            global _last_active_count
            _last_active_count = None


def current_drive_file_count(args: argparse.Namespace, min_poll_interval_sec: float = 30.0) -> int:
    global _drive_backpressure_service
    global _drive_backpressure_folder_id
    global _last_drive_file_poll
    global _last_drive_file_count

    now = time.time()
    with _drive_backpressure_lock:
        if _last_drive_file_count is not None and now - _last_drive_file_poll < min_poll_interval_sec:
            return _last_drive_file_count
        if _drive_backpressure_service is None:
            _drive_backpressure_service = build_drive_service(args)
        if not _drive_backpressure_folder_id:
            _drive_backpressure_folder_id = resolve_drive_folder_id(
                _drive_backpressure_service,
                args.drive_folder_id,
                args.drive_folder,
            )
        if not _drive_backpressure_folder_id:
            count = 0
        else:
            count = 0
            page_token = None
            query = f"'{_drive_backpressure_folder_id}' in parents and trashed=false"
            while True:
                response = _drive_backpressure_service.files().list(
                    q=query,
                    fields="nextPageToken,files(id)",
                    pageSize=1000,
                    pageToken=page_token,
                ).execute()
                count += len(response.get("files", []))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        _last_drive_file_poll = now
        _last_drive_file_count = count
        return count


def wait_for_submission_capacity(args: argparse.Namespace) -> None:
    poll_sec = float(args.queue_poll_sec)
    while True:
        active = active_gee_task_count()
        if int(args.max_active_tasks) > 0 and active >= int(args.max_active_tasks):
            log(f"GEE queue active={active} >= {args.max_active_tasks}; sleeping {poll_sec:.0f}s")
            time.sleep(poll_sec)
            with _task_queue_lock:
                global _last_active_count
                _last_active_count = None
            continue

        if int(args.max_drive_files) > 0:
            drive_files = current_drive_file_count(args)
            if drive_files >= int(args.max_drive_files):
                log(
                    f"Drive backlog files={drive_files} >= {args.max_drive_files}; "
                    f"waiting for local puller to drain it"
                )
                time.sleep(float(args.drive_backpressure_poll_sec))
                with _drive_backpressure_lock:
                    global _last_drive_file_count
                    _last_drive_file_count = None
                continue
        return


def square_region(lon: float, lat: float, export_pixels: int, scale_m: float):
    # Same geometry style as the old script: buffer a WGS84 point by half the
    # target side length, then use its bounding rectangle for export.
    point = ee.Geometry.Point(lon, lat)
    half_size_m = export_pixels * scale_m / 2.0
    bounds = point.buffer(half_size_m).bounds()
    coords = bounds.getInfo()["coordinates"][0]
    rect_bounds = [coords[0][0], coords[0][1], coords[2][0], coords[2][1]]
    return ee.Geometry.Rectangle(rect_bounds)


def query_window(row: dict[str, Any], tp: Timepoint, args: argparse.Namespace) -> tuple[Optional[datetime], datetime, datetime, str]:
    target = parse_utc(row.get(tp.image_time_col))
    if target is not None:
        start = target - timedelta(hours=float(args.image_time_before_hours))
        end = target + timedelta(hours=float(args.image_time_after_hours))
        return target, start, end, "image_time"

    event_time = parse_utc(row.get("event_time"))
    if event_time is None:
        raise ValueError("missing event_time and image_time")
    target = event_time + timedelta(days=tp.fallback_offset_days)
    start = target - timedelta(days=tp.fallback_window_before_days)
    end = target + timedelta(days=tp.fallback_window_after_days)
    return target, start, end, "fallback_offset"


def select_closest_image(collection, target: datetime):
    target_ms = int(target.timestamp() * 1000)
    with_delta = collection.map(
        lambda image: image.set(
            "_abs_time_delta",
            ee.Number(image.get("system:time_start")).subtract(target_ms).abs(),
        )
    )
    return ee.Image(with_delta.sort("_abs_time_delta", True).first())


def sentinel_collection(region: Any, start: datetime, end: datetime, args: argparse.Namespace):
    collection = (
        ee.ImageCollection(GEE_COLLECTION)
        .filterBounds(region)
        .filterDate(ee_date(start), ee_date(end))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", float(args.cloud_pct)))
    )
    if args.sort_property:
        collection = collection.sort(args.sort_property, True)
    return collection


def image_property_str(image, name: str) -> str:
    try:
        value = image.get(name).getInfo()
    except Exception:
        return ""
    if value is None:
        return ""
    return str(value)


def image_time_iso(image) -> str:
    try:
        millis = image.get("system:time_start").getInfo()
    except Exception:
        return ""
    if millis is None:
        return ""
    return iso_z(datetime.fromtimestamp(float(millis) / 1000.0, tz=timezone.utc))


def submit_one(row: dict[str, Any], tp: Timepoint, args: argparse.Namespace) -> dict[str, Any]:
    plume_id = str(row.get("plume_id", "")).strip()
    if not plume_id:
        raise ValueError("missing plume_id")
    lat = float(row[args.lat_col])
    lon = float(row[args.lon_col])
    target, start, end, source = query_window(row, tp, args)

    region = square_region(lon, lat, int(args.export_pixels), float(args.scale))
    collection = sentinel_collection(region, start, end, args)
    count = int(collection.size().getInfo())
    if count <= 0 and target is not None:
        expanded_days = EXPANDED_WINDOW_DAYS[tp.name]
        expanded_start = target - timedelta(days=expanded_days)
        expanded_end = target + timedelta(days=expanded_days)
        collection = sentinel_collection(region, expanded_start, expanded_end, args)
        count = int(collection.size().getInfo())
        if count > 0:
            start = expanded_start
            end = expanded_end
            source = f"{source}_expanded_{expanded_days}d"
    if count <= 0:
        return {
            "plume_id": plume_id,
            "timepoint": tp.name,
            "status": "no_image",
            "collection": GEE_COLLECTION,
            "bands": ",".join(GEE_BANDS),
            "target_time_utc": iso_z(target) if target else "",
            "query_start_utc": iso_z(start),
            "query_end_utc": iso_z(end),
            "plume_latitude": lat,
            "plume_longitude": lon,
            "message": f"no GEE image in {source} window",
        }

    image = select_closest_image(collection, target) if target is not None else ee.Image(collection.first())
    image_id = image_property_str(image, "system:index")
    selected_time = image_time_iso(image)
    cloud = image_property_str(image, "CLOUDY_PIXEL_PERCENTAGE")
    nodata = image_property_str(image, "NODATA_PIXEL_PERCENTAGE")
    image = image.select(GEE_BANDS).clip(region).reproject(crs=args.crs, scale=float(args.scale)).clip(region)

    file_prefix = file_prefix_for(plume_id, tp, args)
    if args.dry_run:
        return {
            "plume_id": plume_id,
            "timepoint": tp.name,
            "status": "dry_run",
            "drive_folder": args.drive_folder,
            "file_prefix": file_prefix,
            "collection": GEE_COLLECTION,
            "bands": ",".join(GEE_BANDS),
            "target_time_utc": iso_z(target) if target else "",
            "query_start_utc": iso_z(start),
            "query_end_utc": iso_z(end),
            "selected_time_utc": selected_time,
            "selected_id": image_id,
            "cloud_pct": cloud,
            "nodata_pct": nodata,
            "plume_latitude": lat,
            "plume_longitude": lon,
            "message": f"would submit; candidates={count}; source={source}",
        }

    wait_for_submission_capacity(args)
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=file_prefix,
        folder=args.drive_folder,
        fileNamePrefix=file_prefix,
        region=region,
        scale=float(args.scale),
        crs=args.crs,
        maxPixels=int(args.max_pixels),
    )
    task.start()
    status = task.status()
    return {
        "plume_id": plume_id,
        "timepoint": tp.name,
        "status": "submitted",
        "task_id": status.get("id", ""),
        "task_state": status.get("state", ""),
        "drive_folder": args.drive_folder,
        "file_prefix": file_prefix,
        "collection": GEE_COLLECTION,
        "bands": ",".join(GEE_BANDS),
        "target_time_utc": iso_z(target) if target else "",
        "query_start_utc": iso_z(start),
        "query_end_utc": iso_z(end),
        "selected_time_utc": selected_time,
        "selected_id": image_id,
        "cloud_pct": cloud,
        "nodata_pct": nodata,
        "plume_latitude": lat,
        "plume_longitude": lon,
        "message": f"submitted; candidates={count}; source={source}",
    }


def iter_jobs(df: pd.DataFrame, timepoints: list[Timepoint], done: set[tuple[str, str]], limit_rows: int, limit_jobs: int):
    rows_seen = 0
    jobs_seen = 0
    for _, row_obj in df.iterrows():
        row = row_obj.to_dict()
        plume_id = str(row.get("plume_id", "")).strip()
        if not plume_id:
            continue
        rows_seen += 1
        if limit_rows and rows_seen > limit_rows:
            break
        for tp in timepoints:
            key = (plume_id, tp.name)
            if key in done:
                continue
            jobs_seen += 1
            if limit_jobs and jobs_seen > limit_jobs:
                return
            yield row, tp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit GEE Sentinel-2 exports for the new six-time S2 table.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--out-manifest", default=str(DEFAULT_OUT_MANIFEST))
    parser.add_argument("--drive-folder", default=DEFAULT_DRIVE_FOLDER)
    parser.add_argument("--drive-folder-id", default="")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--timepoints", default="all", help="Comma list: t0,prev1,prev2,prev3,seasonal,year or all")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--limit-jobs", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--lat-col", default="plume_latitude")
    parser.add_argument("--lon-col", default="plume_longitude")
    parser.add_argument("--cloud-pct", type=float, default=20.0)
    parser.add_argument("--sort-property", default="NODATA_PIXEL_PERCENTAGE")
    parser.add_argument("--image-time-before-hours", type=float, default=36.0)
    parser.add_argument("--image-time-after-hours", type=float, default=36.0)
    parser.add_argument("--export-pixels", type=int, default=512)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--crs", default="EPSG:4326")
    parser.add_argument("--max-pixels", type=int, default=1000000000)
    parser.add_argument("--prefix-template", default="{plume_id}_{suffix}")

    parser.add_argument("--check-local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-drive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-task-states", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-check-workers", type=int, default=32)
    parser.add_argument("--credentials", default="/home/yuyao/methane_train/data_downloading/credentials.json")
    parser.add_argument("--token", default="/home/yuyao/methane_train/data_downloading/token.json")
    parser.add_argument("--service-account", action="store_true")
    parser.add_argument("--max-active-tasks", type=int, default=500)
    parser.add_argument(
        "--max-drive-files",
        type=int,
        default=500,
        help="Pause submissions while this many export files await the local Drive puller; <=0 disables.",
    )
    parser.add_argument("--drive-backpressure-poll-sec", type=float, default=60.0)
    parser.add_argument("--queue-poll-sec", type=float, default=100.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--authenticate", action="store_true", help="Run ee.Authenticate() before ee.Initialize().")
    parser.add_argument("--ee-project", default="", help="Optional Earth Engine Cloud project for ee.Initialize(project=...).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    out_manifest = Path(args.out_manifest)
    timepoints = parse_timepoints(args.timepoints)

    initialize_ee(args)
    df = pd.read_csv(input_csv, low_memory=False)
    task_states = None
    if args.resume and args.refresh_task_states and out_manifest.exists():
        task_states = current_task_states()
        state_counts = pd.Series(list(task_states.values())).value_counts().to_dict()
        log(f"refreshed GEE task states: {state_counts}")
    manifest_done = load_done_keys(out_manifest, bool(args.resume), task_states, bool(args.dry_run))
    local_done = existing_local_keys(df, timepoints, args) if args.resume else set()
    drive_prefixes = load_drive_prefixes(args) if args.resume else set()
    drive_done = existing_drive_keys(df, timepoints, drive_prefixes, args) if args.resume else set()
    done = manifest_done | local_done | drive_done
    jobs = list(iter_jobs(df, timepoints, done, int(args.limit_rows), int(args.limit_jobs)))
    log(
        f"input_rows={len(df)} timepoints={','.join(tp.name for tp in timepoints)} "
        f"resume_manifest={len(manifest_done)} resume_local={len(local_done)} "
        f"resume_drive={len(drive_done)} jobs={len(jobs)} manifest={out_manifest}"
    )
    if not jobs:
        return 0

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(submit_one, row, tp, args): (row.get("plume_id", ""), tp.name) for row, tp in jobs}
        for idx, fut in enumerate(as_completed(futures), start=1):
            plume_id, tp_name = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:
                record = {
                    "plume_id": plume_id,
                    "timepoint": tp_name,
                    "status": "failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            append_manifest(out_manifest, record)
            if record.get("status") in {"submitted", "dry_run"}:
                ok += 1
            else:
                failed += 1
            if idx % max(1, int(args.progress_every)) == 0 or idx == len(futures):
                log(f"done={idx}/{len(futures)} ok={ok} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
