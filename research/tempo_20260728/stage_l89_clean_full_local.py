#!/usr/bin/env python3
"""Gate and resumably stage clean-L89 train-only imagery to /diniuvol.

The audit phase performs remote filesystem operations on at most 512
deterministically selected files that are absent from the existing local
six-role cache.  It copies at most 64 of those files as the formal resumable
prefix.  Full staging is allowed only when the persisted feasibility gates
pass.  No test/sealed/holdout path is ever constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "Upgraded_dataset/l89_6time_full_event_balanced_split/"
    "L89_temporal_train_full_event_balanced.csv"
)
DEFAULT_TARGET = Path(
    "/diniuvol/yuyao/methanefuse_research_20260728/"
    "l89_clean_full_local_v1"
)
OLD_3TIME = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/cache/l89_3time"
)
OLD_EXTRA = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/cache/l89_6time_extra"
)
ROLE_LAYOUT = {
    "path_t0": (OLD_3TIME, "l89_0.tif"),
    "path_prev1": (OLD_3TIME, "l89_prev1.tif"),
    "path_prev2": (OLD_EXTRA, "l89_prev2.tif"),
    "path_prev3": (OLD_EXTRA, "l89_prev3.tif"),
    "path_seasonal": (OLD_3TIME, "l89_seasonal.tif"),
    "path_year": (OLD_EXTRA, "l89_year.tif"),
}
HELD_OUT_RE = re.compile(
    r"(^|[._-])(test|sealed|holdout|outer)([._-]|$)", re.IGNORECASE
)
GIB = 1024**3
MIB = 1024**2
AUDIT_SEED = 20260728


@dataclass(frozen=True)
class FileRecord:
    source: str
    destination: str
    role: str
    row_folder: str


def safe_path(value: str | Path, *, purpose: str) -> Path:
    path = Path(value).expanduser().resolve()
    offending = [part for part in path.parts if HELD_OUT_RE.search(part)]
    if offending:
        raise ValueError(
            f"{purpose} contains held-out marker {sorted(offending)}: {path}"
        )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def stable_rank(record: FileRecord) -> str:
    return hashlib.sha256(
        f"{AUDIT_SEED}|{record.source}".encode("utf-8")
    ).hexdigest()


def build_missing_records(
    input_path: Path, target_root: Path
) -> tuple[list[FileRecord], dict[str, Any]]:
    frame = pd.read_csv(input_path, low_memory=False)
    required = {"path", *ROLE_LAYOUT}
    missing_columns = sorted(required - set(frame))
    if missing_columns:
        raise ValueError(f"Source train CSV is missing {missing_columns}.")
    unique: dict[str, FileRecord] = {}
    collision_guard: dict[str, str] = {}
    existing_role_counts = {role: 0 for role in ROLE_LAYOUT}
    rows_complete = 0
    for row in frame.to_dict(orient="records"):
        folder = Path(str(row["path"])).name
        if not folder:
            raise ValueError("Empty row folder in source train CSV.")
        row_complete = True
        for role, (old_root, expected_name) in ROLE_LAYOUT.items():
            source_text = str(row[role]).strip()
            if not source_text:
                raise ValueError(f"Empty {role} source path.")
            source_name = Path(source_text).name
            if source_name != expected_name:
                raise ValueError(
                    f"{role} source basename {source_name} != {expected_name}"
                )
            old_local = old_root / folder / expected_name
            if old_local.is_file():
                existing_role_counts[role] += 1
                continue
            row_complete = False
            destination = target_root / folder / expected_name
            destination_text = str(destination)
            prior_source = collision_guard.setdefault(
                destination_text, source_text
            )
            if prior_source != source_text:
                raise RuntimeError(
                    f"Destination collision: {destination_text}"
                )
            unique.setdefault(
                source_text,
                FileRecord(
                    source=source_text,
                    destination=destination_text,
                    role=role,
                    row_folder=folder,
                ),
            )
        rows_complete += int(row_complete)
    records = sorted(unique.values(), key=lambda item: item.source)
    return records, {
        "rows": int(len(frame)),
        "roles": int(len(ROLE_LAYOUT)),
        "old_local_complete_rows": int(rows_complete),
        "old_local_complete_fraction": float(rows_complete / len(frame)),
        "old_local_files_by_role": existing_role_counts,
        "unique_missing_source_files": int(len(records)),
        "destination_collisions": 0,
    }


def stat_record(record: FileRecord) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = os.stat(record.source)
        if not os.path.isfile(record.source):
            return {
                **asdict(record),
                "exists_regular": False,
                "error": "not a regular file",
                "elapsed_seconds": time.monotonic() - started,
            }
        return {
            **asdict(record),
            "exists_regular": True,
            "bytes": int(result.st_size),
            "elapsed_seconds": time.monotonic() - started,
        }
    except OSError as error:
        return {
            **asdict(record),
            "exists_regular": False,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.monotonic() - started,
        }


def encode_manifest_record(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def append_manifest(path: Path, value: dict[str, Any]) -> None:
    encoded = encode_manifest_record(value)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640
    )
    with os.fdopen(descriptor, "ab", buffering=0) as stream:
        stream.write(encoded)
        os.fsync(stream.fileno())


def copy_atomic(record: FileRecord) -> dict[str, Any]:
    source = Path(record.source)
    destination = Path(record.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if destination.is_file():
        size = int(destination.stat().st_size)
        digest = sha256_file(destination)
        return {
            **asdict(record),
            "status": "verified_existing_destination",
            "bytes": size,
            "sha256": digest,
            "elapsed_seconds": time.monotonic() - started,
        }
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.partial"
    )
    digest = hashlib.sha256()
    total = 0
    try:
        with (
            source.open("rb", buffering=0) as input_stream,
            temporary.open("xb", buffering=0) as output_stream,
        ):
            while True:
                block = input_stream.read(16 * MIB)
                if not block:
                    break
                output_stream.write(block)
                digest.update(block)
                total += len(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        **asdict(record),
        "status": "copied_atomic",
        "bytes": int(total),
        "sha256": digest.hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
    }


def percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(float(fraction) * len(ordered)) - 1),
    )
    return float(ordered[index])


def command_audit(args: argparse.Namespace) -> None:
    input_path = safe_path(args.input_csv, purpose="source train CSV")
    target_root = safe_path(args.target_root, purpose="local staging target")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    target_root.mkdir(parents=True, exist_ok=True)
    input_sha = sha256_file(input_path)
    records, inventory = build_missing_records(input_path, target_root)
    ranked = sorted(records, key=stable_rank)
    stat_sample = ranked[: min(int(args.stat_sample), len(ranked))]
    if len(stat_sample) > 512:
        raise ValueError("Remote stat sample is capped at 512 files.")
    stat_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        stat_results = list(executor.map(stat_record, stat_sample))
    stat_wall = time.monotonic() - stat_started
    existing = [
        result for result in stat_results if result["exists_regular"]
    ]
    sizes = [int(result["bytes"]) for result in existing]
    existence_fraction = (
        float(len(existing) / len(stat_results)) if stat_results else 0.0
    )
    if not sizes:
        raise RuntimeError("No sampled source file exists.")

    copy_candidates_by_source = {
        result["source"]: result for result in existing
    }
    copy_records = [
        record
        for record in stat_sample
        if record.source in copy_candidates_by_source
    ][: min(int(args.copy_sample), 64)]
    copy_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        copy_results = list(executor.map(copy_atomic, copy_records))
    copy_wall = time.monotonic() - copy_started
    manifest_path = target_root / "COPY_MANIFEST.jsonl"
    for result in copy_results:
        append_manifest(
            manifest_path,
            {
                **result,
                "phase": "feasibility_prefix",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            },
        )

    unique_missing = int(inventory["unique_missing_source_files"])
    mean_size = float(statistics.fmean(sizes))
    estimated_total_bytes = int(round(mean_size * unique_missing))
    copied_bytes = int(sum(int(item["bytes"]) for item in copy_results))
    throughput = copied_bytes / max(copy_wall, 1e-9)
    estimated_remaining = max(0, estimated_total_bytes - copied_bytes)
    eta_seconds = estimated_remaining / max(throughput, 1e-9)
    disk = shutil.disk_usage(target_root)
    projected_free = int(disk.free - estimated_remaining)
    gates = {
        "estimated_total_at_most_180_gib": bool(
            estimated_total_bytes <= 180 * GIB
        ),
        "projected_free_at_least_180_gib": bool(
            projected_free >= 180 * GIB
        ),
        "sample_source_existence_at_least_99_9_percent": bool(
            existence_fraction >= 0.999
        ),
        "estimated_16_worker_eta_at_most_90_minutes": bool(
            eta_seconds <= 90 * 60
        ),
    }
    audit = {
        "schema_version": "l89-clean-staging-feasibility-v1",
        "source_train": {
            "path": str(input_path),
            "sha256": input_sha,
        },
        "target_root": str(target_root),
        "inventory": inventory,
        "remote_access_boundary": {
            "only_missing_role_paths_sampled": True,
            "remote_directories_listed": False,
            "remote_stat_sample": int(len(stat_results)),
            "remote_copy_sample": int(len(copy_results)),
            "held_out_paths_accessed": False,
        },
        "stat_sample": {
            "deterministic_seed": AUDIT_SEED,
            "existing_regular_files": int(len(existing)),
            "missing_or_error_files": int(len(stat_results) - len(existing)),
            "existence_fraction": existence_fraction,
            "wall_seconds": float(stat_wall),
            "mean_bytes": mean_size,
            "median_bytes": float(statistics.median(sizes)),
            "p95_bytes": percentile(sizes, 0.95),
            "error_sample": [
                {
                    "source": item["source"],
                    "error": item.get("error", "unknown"),
                }
                for item in stat_results
                if not item["exists_regular"]
            ][:20],
        },
        "copy_prefix": {
            "files": int(len(copy_results)),
            "bytes": copied_bytes,
            "wall_seconds": float(copy_wall),
            "aggregate_mib_per_second": float(throughput / MIB),
            "manifest": str(manifest_path),
            "retained_as_formal_resumable_prefix": True,
        },
        "projection": {
            "estimated_total_bytes": estimated_total_bytes,
            "estimated_total_gib": float(estimated_total_bytes / GIB),
            "estimated_remaining_bytes_after_prefix": estimated_remaining,
            "disk_free_bytes_after_prefix": int(disk.free),
            "disk_free_gib_after_prefix": float(disk.free / GIB),
            "projected_free_bytes_after_completion": projected_free,
            "projected_free_gib_after_completion": float(
                projected_free / GIB
            ),
            "estimated_16_worker_eta_seconds": float(eta_seconds),
            "estimated_16_worker_eta_minutes": float(eta_seconds / 60.0),
        },
        "gates": gates,
        "ready_for_full_staging": bool(all(gates.values())),
        "full_staging_not_started_by_audit_command": True,
        "test_or_sealed_or_holdout_access": False,
    }
    audit["audit_contract_sha256"] = canonical_json_digest(audit)
    audit_path = target_root / "FEASIBILITY_AUDIT.json"
    atomic_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


def load_completed_manifest(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed manifest line {line_number}: {error}"
                ) from error
            if value.get("status") in {
                "copied_atomic",
                "verified_existing_destination",
            }:
                completed[str(value["source"])] = value
    return completed


def copy_catching(record: FileRecord) -> dict[str, Any]:
    try:
        return copy_atomic(record)
    except Exception as error:
        return {
            **asdict(record),
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": 0.0,
        }


def command_stage(args: argparse.Namespace) -> None:
    input_path = safe_path(args.input_csv, purpose="source train CSV")
    target_root = safe_path(args.target_root, purpose="local staging target")
    audit_path = target_root / "FEASIBILITY_AUDIT.json"
    if not audit_path.is_file():
        raise FileNotFoundError("Run the bounded feasibility audit first.")
    with audit_path.open("r", encoding="utf-8") as stream:
        audit = json.load(stream)
    audit_digest = audit.pop("audit_contract_sha256")
    if canonical_json_digest(audit) != audit_digest:
        raise RuntimeError("Feasibility audit digest changed.")
    audit["audit_contract_sha256"] = audit_digest
    if not bool(audit["ready_for_full_staging"]):
        raise RuntimeError("Feasibility gates did not all pass.")
    if sha256_file(input_path) != audit["source_train"]["sha256"]:
        raise RuntimeError("Source train CSV changed after feasibility audit.")
    if int(args.workers) > 16:
        raise ValueError("Full staging is capped at 16 workers.")

    records, inventory = build_missing_records(input_path, target_root)
    if (
        int(inventory["unique_missing_source_files"])
        != int(audit["inventory"]["unique_missing_source_files"])
    ):
        raise RuntimeError("Missing-file inventory changed after audit.")
    manifest_path = target_root / "COPY_MANIFEST.jsonl"
    completed = load_completed_manifest(manifest_path)
    pending: list[FileRecord] = []
    for record in records:
        previous = completed.get(record.source)
        destination = Path(record.destination)
        if (
            previous is not None
            and destination.is_file()
            and int(destination.stat().st_size) == int(previous["bytes"])
        ):
            continue
        pending.append(record)

    started = time.monotonic()
    copied = 0
    copied_bytes = 0
    errors: list[dict[str, Any]] = []
    progress_path = target_root / "STAGING_PROGRESS.json"
    # A per-record fsync caps manifest consumption at only a few files per
    # second on /diniuvol even when the 16 copy workers are healthy.  Persist
    # one append stream and fsync at every progress checkpoint instead.  A
    # crash can leave at most progress_interval completed destinations
    # unrecorded; copy_atomic verifies and records those existing files on the
    # next resumable run.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        manifest_path.open("ab", buffering=1024 * 1024) as manifest_stream,
        ThreadPoolExecutor(max_workers=int(args.workers)) as executor,
    ):
        for result in executor.map(copy_catching, pending):
            record = {
                **result,
                "phase": "full_staging",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            manifest_stream.write(encode_manifest_record(record))
            if result["status"] == "error":
                errors.append(record)
            else:
                copied += 1
                copied_bytes += int(result["bytes"])
            processed = copied + len(errors)
            if processed % int(args.progress_interval) == 0:
                manifest_stream.flush()
                os.fsync(manifest_stream.fileno())
                atomic_json(
                    progress_path,
                    {
                        "status": "running",
                        "expected_unique_missing_files": int(len(records)),
                        "already_complete_before_run": int(len(completed)),
                        "pending_at_start": int(len(pending)),
                        "processed_this_run": int(processed),
                        "copied_this_run": int(copied),
                        "errors_this_run": int(len(errors)),
                        "bytes_this_run": int(copied_bytes),
                        "elapsed_seconds": float(time.monotonic() - started),
                        "workers": int(args.workers),
                        "manifest_fsync_interval": int(
                            args.progress_interval
                        ),
                        "test_or_sealed_or_holdout_access": False,
                    },
                )
        manifest_stream.flush()
        os.fsync(manifest_stream.fileno())
    elapsed = time.monotonic() - started
    final_completed = load_completed_manifest(manifest_path)
    verified = 0
    for record in records:
        previous = final_completed.get(record.source)
        destination = Path(record.destination)
        if (
            previous is not None
            and destination.is_file()
            and int(destination.stat().st_size) == int(previous["bytes"])
        ):
            verified += 1
    status = "complete" if verified == len(records) and not errors else "incomplete"
    final = {
        "schema_version": "l89-clean-staging-completion-v1",
        "status": status,
        "source_train_sha256": audit["source_train"]["sha256"],
        "feasibility_audit_contract_sha256": audit_digest,
        "expected_unique_missing_files": int(len(records)),
        "verified_complete_files": int(verified),
        "errors_this_run": int(len(errors)),
        "error_sample": errors[:20],
        "workers": int(args.workers),
        "elapsed_seconds_this_run": float(elapsed),
        "copy_manifest": str(manifest_path),
        "copy_manifest_sha256": sha256_file(manifest_path),
        "test_or_sealed_or_holdout_access": False,
    }
    atomic_json(
        target_root
        / ("STAGING_COMPLETE.json" if status == "complete" else "STAGING_INCOMPLETE.json"),
        final,
    )
    atomic_json(progress_path, final)
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    if status != "complete":
        raise RuntimeError(
            f"Staging incomplete: {verified}/{len(records)} files."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.set_defaults(function=command_audit)
    audit.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    audit.add_argument("--target-root", default=str(DEFAULT_TARGET))
    audit.add_argument("--workers", type=int, default=16)
    audit.add_argument("--stat-sample", type=int, default=512)
    audit.add_argument("--copy-sample", type=int, default=64)

    stage = subparsers.add_parser("stage")
    stage.set_defaults(function=command_stage)
    stage.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    stage.add_argument("--target-root", default=str(DEFAULT_TARGET))
    stage.add_argument("--workers", type=int, default=16)
    stage.add_argument("--progress-interval", type=int, default=256)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if int(args.workers) < 1 or int(args.workers) > 16:
        raise ValueError("workers must be between 1 and 16.")
    args.function(args)


if __name__ == "__main__":
    main()
