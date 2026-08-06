#!/usr/bin/env python3
"""Pull completed six-time S2 GEE Drive exports into local raw directories.

This is the second line after ``s2_6time_gee_export.py``:

  GEE -> Google Drive -> local raw TIFFs -> wide local paths CSV

It reads ``s2_6time_gee_export_manifest.csv`` and matches Drive files by the
``file_prefix`` recorded when the export task was submitted.  Downloaded files
are written as:

  {raw_root}/S2_GEE_6time/{timepoint}/{plume_id}/{canonical_filename}.tif

The wide output CSV keeps one row per plume and can be used by the later
standardization/crop stage.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
except ImportError as exc:  # pragma: no cover - runtime environment issue
    raise SystemExit(
        "Missing Google Drive deps. Install with:\n"
        "  pip install google-api-python-client google-auth google-auth-oauthlib"
    ) from exc

try:
    import rasterio
except ImportError:  # pragma: no cover - validation falls back to size only
    rasterio = None


SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPORT_MANIFEST = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_gee_export_manifest.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_gee_drive_pull_manifest.csv"
DEFAULT_WIDE_CSV = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_gee_local_paths.csv"
DEFAULT_INPUT_TABLE = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_all6_available_paths_std512_complete.csv"
DEFAULT_RAW_ROOT = "/mnt/engg-niulab/yuyao/sensors_raw_data"
DEFAULT_DRIVE_FOLDER = "CM_S2_L2A_6TIME_GEE"

TIMEPOINTS = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
CANONICAL_FILENAME = {
    "t0": "s2_0.tif",
    "prev1": "s2_prev1.tif",
    "prev2": "s2_prev2.tif",
    "prev3": "s2_prev3.tif",
    "seasonal": "s2_seasonal.tif",
    "year": "s2_year.tif",
}
WIDE_PATH_COL = {
    "t0": "gee_t0_raw_path",
    "prev1": "gee_prev1_raw_path",
    "prev2": "gee_prev2_raw_path",
    "prev3": "gee_prev3_raw_path",
    "seasonal": "gee_seasonal_raw_path",
    "year": "gee_year_raw_path",
}
SUCCESS_STATUSES = {
    "downloaded",
    "downloaded_deleted_drive",
    "skip_existing_valid",
    "skip_existing_valid_deleted_drive",
    "resume_skip_completed",
    "resume_skip_completed_deleted_drive",
}
OUT_FIELDS = [
    "plume_id",
    "timepoint",
    "status",
    "drive_file_id",
    "drive_file_name",
    "drive_folder",
    "file_prefix",
    "raw_path",
    "size",
    "validate_message",
    "message",
]

_worker_local = threading.local()


def log(message: str) -> None:
    print(message, flush=True)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def parse_drive_time(value: Any) -> Optional[datetime]:
    if not has_value(value):
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def should_skip_by_age(item: dict[str, Any], min_age_seconds: int) -> bool:
    if min_age_seconds <= 0:
        return False
    modified = parse_drive_time(item.get("modifiedTime"))
    if modified is None:
        return False
    return (datetime.now(timezone.utc) - modified).total_seconds() < min_age_seconds


def build_drive_service(args: argparse.Namespace):
    if args.service_account:
        creds = service_account.Credentials.from_service_account_file(args.credentials, scopes=SCOPES)
    else:
        creds = None
        if os.path.exists(args.token):
            creds = Credentials.from_authorized_user_file(args.token, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(args.credentials, SCOPES)
                creds = flow.run_console()
            Path(args.token).parent.mkdir(parents=True, exist_ok=True)
            with open(args.token, "w") as fh:
                fh.write(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def worker_drive_service(args: argparse.Namespace):
    service = getattr(_worker_local, "drive_service", None)
    if service is None:
        service = build_drive_service(args)
        _worker_local.drive_service = service
    return service


def resolve_folder_id(service: Any, folder_id: str, folder_name: str) -> str:
    if folder_id:
        return folder_id
    safe_name = folder_name.replace("'", "\\'")
    query = f"mimeType='{FOLDER_MIME_TYPE}' and name='{safe_name}' and trashed=false"
    resp = service.files().list(q=query, fields="files(id,name,parents)").execute()
    matches = resp.get("files", [])
    if not matches:
        raise RuntimeError(f"Drive folder not found: {folder_name}")
    if len(matches) > 1:
        found = ", ".join(f"{m['name']}({m['id']})" for m in matches)
        raise RuntimeError(f"multiple Drive folders named {folder_name}: {found}; pass --drive-folder-id")
    return matches[0]["id"]


def iter_folder_files(service: Any, folder_id: str) -> Iterable[dict[str, Any]]:
    page_token = None
    query = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id,name,size,mimeType,modifiedTime)"
    while True:
        resp = service.files().list(q=query, fields=fields, pageToken=page_token, orderBy="createdTime").execute()
        yield from resp.get("files", [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def file_stem(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".tif"):
        return name[:-4]
    if lower.endswith(".tiff"):
        return name[:-5]
    return name


def local_target_path(raw_root: str, plume_id: str, timepoint: str) -> Path:
    return Path(raw_root) / "S2_GEE_6time" / timepoint / plume_id / CANONICAL_FILENAME[timepoint]


def target_from_prefix(prefix: str, raw_root: str) -> Optional[dict[str, str]]:
    for timepoint, filename in sorted(CANONICAL_FILENAME.items(), key=lambda item: len(file_stem(item[1])), reverse=True):
        suffix = "_" + file_stem(filename)
        if not prefix.endswith(suffix):
            continue
        plume_id = prefix[: -len(suffix)].strip()
        if not plume_id:
            return None
        return {
            "plume_id": plume_id,
            "timepoint": timepoint,
            "file_prefix": prefix,
            "raw_path": str(local_target_path(raw_root, plume_id, timepoint)),
        }
    return None


def load_expected_exports(export_manifest: Path, raw_root: str, resume_submitted_only: bool = True) -> dict[str, dict[str, str]]:
    if not export_manifest.exists():
        raise FileNotFoundError(export_manifest)
    df = pd.read_csv(export_manifest, low_memory=False)
    required = {"plume_id", "timepoint", "file_prefix", "status"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{export_manifest} missing columns: {sorted(missing)}")

    expected: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        status = str(row.get("status", "")).strip()
        if resume_submitted_only and status not in {"submitted", "already_submitted", "dry_run"}:
            continue
        plume_id = str(row.get("plume_id", "")).strip()
        tp = str(row.get("timepoint", "")).strip()
        prefix = str(row.get("file_prefix", "")).strip()
        if not plume_id or tp not in TIMEPOINTS or not prefix:
            continue
        expected[prefix] = {
            "plume_id": plume_id,
            "timepoint": tp,
            "file_prefix": prefix,
            "raw_path": str(local_target_path(raw_root, plume_id, tp)),
        }
    return expected


def file_size_matches(item: dict[str, Any], local_path: Path) -> bool:
    if "size" not in item:
        return True
    try:
        return local_path.stat().st_size == int(item["size"])
    except OSError:
        return False


def download_file(service: Any, item: dict[str, Any], dest_path: Path, chunk_size: int) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=item["id"])
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    with io.FileIO(tmp_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    os.replace(tmp_path, dest_path)


def validate_tif(path: Path, expected_bands: int) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.stat().st_size <= 0:
        return False, "empty_file"
    if rasterio is None:
        return True, "size_only"
    try:
        with rasterio.open(path) as ds:
            if ds.width <= 0 or ds.height <= 0 or ds.count <= 0:
                return False, f"invalid_shape:{ds.count}x{ds.height}x{ds.width}"
            if expected_bands > 0 and ds.count != expected_bands:
                return False, f"unexpected_band_count:{ds.count} expected={expected_bands}"
            return True, f"rasterio_ok:{ds.count}x{ds.height}x{ds.width}"
    except Exception as exc:
        return False, f"rasterio_error:{type(exc).__name__}:{exc}"


def load_completed_records(out_csv: Path) -> dict[tuple[str, str, str], Path]:
    if not out_csv.exists():
        return {}
    try:
        df = pd.read_csv(out_csv, low_memory=False)
    except Exception:
        return {}
    required = {"plume_id", "timepoint", "drive_file_name", "raw_path", "status"}
    if not required.issubset(df.columns):
        return {}
    df = df[df["status"].astype(str).isin(SUCCESS_STATUSES)]
    df = df.drop_duplicates(["plume_id", "timepoint", "drive_file_name"], keep="last")
    completed: dict[tuple[str, str, str], Path] = {}
    for row in df.itertuples(index=False):
        plume_id = str(getattr(row, "plume_id", "")).strip()
        tp = str(getattr(row, "timepoint", "")).strip()
        name = str(getattr(row, "drive_file_name", "")).strip()
        raw_path = Path(str(getattr(row, "raw_path", "")).strip())
        if plume_id and tp in TIMEPOINTS and name:
            completed[(plume_id, tp, name)] = raw_path
    return completed


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in OUT_FIELDS} for row in rows)


def successful_pull_rows(out_csv: Path) -> pd.DataFrame:
    if not out_csv.exists():
        return pd.DataFrame(columns=OUT_FIELDS)
    df = pd.read_csv(out_csv, low_memory=False)
    if "status" not in df.columns:
        return pd.DataFrame(columns=OUT_FIELDS)
    return df[df["status"].astype(str).isin(SUCCESS_STATUSES)].copy()


def write_wide_paths(input_table: Path, out_csv: Path, wide_csv: Path) -> int:
    pulls = successful_pull_rows(out_csv)
    if pulls.empty:
        return 0
    pulls = pulls.sort_values("raw_path").drop_duplicates(["plume_id", "timepoint"], keep="last")
    by_key = {(str(r.plume_id), str(r.timepoint)): str(r.raw_path) for r in pulls.itertuples(index=False)}

    if input_table.exists():
        wide = pd.read_csv(input_table, low_memory=False)
    else:
        plume_ids = sorted({k[0] for k in by_key})
        wide = pd.DataFrame({"plume_id": plume_ids})
    for tp in TIMEPOINTS:
        col = WIDE_PATH_COL[tp]
        wide[col] = [by_key.get((str(pid), tp), "") for pid in wide["plume_id"].astype(str)]
        wide[f"has_{tp}_gee_raw"] = wide[col].astype(str).map(lambda p: int(bool(p) and Path(p).exists()))
    wide["has_all6_gee_raw"] = wide[[f"has_{tp}_gee_raw" for tp in TIMEPOINTS]].min(axis=1)
    wide_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = wide_csv.with_suffix(wide_csv.suffix + ".part")
    wide.to_csv(tmp, index=False)
    tmp.replace(wide_csv)
    return int(wide["has_all6_gee_raw"].sum())


def collect_drive_jobs(
    service: Any,
    folder_id: str,
    expected: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> list[list[tuple[dict[str, Any], dict[str, str]]]]:
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, str]]]] = {}
    eligible = 0
    for item in iter_folder_files(service, folder_id):
        if item.get("mimeType") == FOLDER_MIME_TYPE:
            continue
        if should_skip_by_age(item, int(args.min_age_seconds)):
            continue
        name = str(item.get("name", ""))
        prefix = file_stem(name)
        target = expected.get(prefix)
        if target is None:
            if not args.allow_unplanned:
                continue
            target = target_from_prefix(prefix, args.raw_root)
            if target is None:
                continue

        key = (target["plume_id"], target["timepoint"])
        grouped.setdefault(key, []).append((item, target))
        eligible += 1
        if args.max_files and eligible >= int(args.max_files):
            break
    return list(grouped.values())


def process_drive_item(
    service: Any,
    item: dict[str, Any],
    target: dict[str, str],
    completed: dict[tuple[str, str, str], Path],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    name = str(item.get("name", ""))
    plume_id = target["plume_id"]
    tp = target["timepoint"]
    dest_path = Path(target["raw_path"])
    record = {
        "plume_id": plume_id,
        "timepoint": tp,
        "status": "",
        "drive_file_id": item.get("id", ""),
        "drive_file_name": name,
        "drive_folder": args.drive_folder,
        "file_prefix": target["file_prefix"],
        "raw_path": str(dest_path),
        "size": item.get("size", ""),
        "validate_message": "",
        "message": "",
    }

    try:
        done_path = completed.get((plume_id, tp, name))
        if args.resume and done_path is not None and not args.overwrite:
            ok, msg = validate_tif(done_path, int(args.expected_bands))
            record["raw_path"] = str(done_path)
            record["validate_message"] = msg
            if ok:
                record["status"] = "resume_skip_completed"
                if args.delete_after and not args.dry_run:
                    service.files().delete(fileId=item["id"]).execute()
                    record["status"] = "resume_skip_completed_deleted_drive"
                return record, 1

        if dest_path.exists() and not args.overwrite:
            if file_size_matches(item, dest_path):
                ok, msg = validate_tif(dest_path, int(args.expected_bands))
                record["validate_message"] = msg
                record["status"] = "skip_existing_valid" if ok else "existing_invalid"
                if ok and args.delete_after and not args.dry_run:
                    service.files().delete(fileId=item["id"]).execute()
                    record["status"] = "skip_existing_valid_deleted_drive"
                return record, int(ok)
            record["status"] = "existing_size_mismatch"
            return record, 0

        if args.dry_run:
            record["status"] = "dry_run"
            log(f"dry-run download {name} -> {dest_path}")
            return record, 1

        log(f"downloading {name} -> {dest_path}")
        download_file(service, item, dest_path, int(args.chunk_size_mb) * 1024 * 1024)
        if not file_size_matches(item, dest_path):
            record["status"] = "size_mismatch"
            return record, 0
        ok, msg = validate_tif(dest_path, int(args.expected_bands))
        record["validate_message"] = msg
        if not ok:
            record["status"] = "invalid_tif"
            return record, 0
        if args.delete_after:
            service.files().delete(fileId=item["id"]).execute()
            record["status"] = "downloaded_deleted_drive"
        else:
            record["status"] = "downloaded"
        return record, 1
    except HttpError as exc:
        record["status"] = "drive_error"
        record["message"] = str(exc)
    except Exception as exc:
        record["status"] = "download_error"
        record["message"] = str(exc)
    return record, 0


def process_drive_group(
    group: list[tuple[dict[str, Any], dict[str, str]]],
    completed: dict[tuple[str, str, str], Path],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int]:
    service = worker_drive_service(args)
    records: list[dict[str, Any]] = []
    processed = 0
    for item, target in group:
        record, count = process_drive_item(service, item, target, completed, args)
        records.append(record)
        processed += count
    return records, processed


def process_once(service: Any, folder_id: str, expected: dict[str, dict[str, str]], completed: dict[tuple[str, str, str], Path], args: argparse.Namespace) -> int:
    groups = collect_drive_jobs(service, folder_id, expected, args)
    if not groups:
        return 0

    processed = 0
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(process_drive_group, group, completed, args) for group in groups]
        for future in as_completed(futures):
            group_records, group_processed = future.result()
            records.extend(group_records)
            processed += group_processed

    append_rows(Path(args.out_csv), records)
    if records and not args.no_wide_update:
        complete = write_wide_paths(Path(args.input_table), Path(args.out_csv), Path(args.wide_out_csv))
        log(f"updated wide local paths: complete_all6={complete} -> {args.wide_out_csv}")
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull S2 six-time GEE Drive exports into local raw files.")
    parser.add_argument("--export-manifest", default=str(DEFAULT_EXPORT_MANIFEST))
    parser.add_argument("--input-table", default=str(DEFAULT_INPUT_TABLE), help="Optional original wide table to preserve metadata in --wide-out-csv.")
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--wide-out-csv", default=str(DEFAULT_WIDE_CSV))
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--drive-folder", default=DEFAULT_DRIVE_FOLDER)
    parser.add_argument("--drive-folder-id", default="")
    parser.add_argument("--credentials", default="/home/yuyao/methane_train/data_downloading/credentials.json")
    parser.add_argument("--token", default="/home/yuyao/methane_train/data_downloading/token.json")
    parser.add_argument("--service-account", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--min-age-seconds", type=int, default=30)
    parser.add_argument("--chunk-size-mb", type=int, default=16)
    parser.add_argument("--expected-bands", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delete-after", dest="delete_after", action="store_true", default=True)
    parser.add_argument("--no-delete", dest="delete_after", action="store_false")
    parser.add_argument("--allow-unplanned", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wide-update", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    completed = load_completed_records(Path(args.out_csv)) if args.resume else {}
    if args.resume:
        log(f"loaded completed pull records for resume: {len(completed)}")
    service = build_drive_service(args)
    folder_id = resolve_folder_id(service, args.drive_folder_id, args.drive_folder)
    previous_expected_count = -1

    while True:
        try:
            expected = load_expected_exports(
                Path(args.export_manifest),
                args.raw_root,
                resume_submitted_only=not args.allow_unplanned,
            )
            if len(expected) != previous_expected_count:
                log(f"loaded expected S2 GEE exports: {len(expected)} from {args.export_manifest}")
                previous_expected_count = len(expected)
            count = process_once(service, folder_id, expected, completed, args)
            if args.once:
                break
            if count == 0:
                time.sleep(int(args.poll_seconds))
        except KeyboardInterrupt:
            break
        except Exception as exc:
            log(f"sync loop error: {type(exc).__name__}: {exc}")
            if args.once:
                break
            time.sleep(int(args.poll_seconds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
