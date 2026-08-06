#!/usr/bin/env python3
"""Audit legacy-360 payload availability and build deterministic safe manifests.

The historical CSVs occasionally name payloads that are no longer present on
the remote filesystem.  Classification requires all three optical temporal
roles for a declared sensor (S5P stores the three roles in one NPZ).  If any
required payload is unavailable, this script clears that complete sensor arm.
A row is removed only when no sensor remains.

The read-only hashed cache is accepted as a payload source even if the remote
file has subsequently disappeared.  The cache is never modified here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SENSOR_ORDER = ("s2", "l89", "emit", "s5p")
REQUIRED_PATHS = {
    "s2": ("s2_0_path", "s2_90_path", "s2_360_path"),
    "l89": ("l89_0_path", "l89_90_path", "l89_360_path"),
    "emit": ("emit_0_path", "emit_90_path", "emit_360_path"),
    "s5p": ("s5p_0_path",),
}
CLEAR_PATHS = {
    sensor: tuple(
        f"{sensor}_{role}_path" for role in ("0", "90", "360")
    )
    for sensor in SENSOR_ORDER
}
OPTICAL_MIN_BYTES = 1024
S5P_MIN_BYTES = 128
NPZ_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def cached_path(cache_dir: Path, source: str) -> Path:
    source_path = Path(source).expanduser().absolute()
    digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()
    suffix = (
        "".join(source_path.suffixes[-2:]) if source_path.suffixes else ""
    )
    if len(suffix) > 24:
        suffix = source_path.suffix
    return cache_dir / digest[:2] / f"{digest}{suffix}"


def _candidate_status(
    path: Path, *, is_s5p: bool
) -> tuple[bool, int, str]:
    minimum = S5P_MIN_BYTES if is_s5p else OPTICAL_MIN_BYTES
    try:
        if not path.is_file():
            return False, 0, "missing"
        size = int(path.stat().st_size)
        if size < minimum:
            return False, size, "too_small"
        if is_s5p:
            if path.suffix.casefold() != ".npz":
                return False, size, "not_npz"
            with path.open("rb") as stream:
                if stream.read(4) not in NPZ_MAGIC:
                    return False, size, "bad_npz_magic"
        return True, size, "ok"
    except OSError:
        return False, 0, "stat_or_header_error"


def payload_status(source: str, cache_dir: Path) -> tuple[str, int]:
    is_s5p = Path(source).suffix.casefold() == ".npz"
    cached = cached_path(cache_dir, source)
    valid, size, cache_reason = _candidate_status(
        cached, is_s5p=is_s5p
    )
    if valid:
        return "cache", size
    source_path = Path(source).expanduser().absolute()
    valid, source_size, source_reason = _candidate_status(
        source_path, is_s5p=is_s5p
    )
    if valid:
        return "source", source_size
    return f"cache_{cache_reason}__source_{source_reason}", source_size


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
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


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def label_counts(frame: pd.DataFrame) -> dict[str, int]:
    values = pd.to_numeric(frame["label"], errors="raise").astype(int)
    return {
        str(key): int(value)
        for key, value in values.value_counts().sort_index().items()
    }


def declared(frame: pd.DataFrame, sensor: str) -> pd.Series:
    columns = REQUIRED_PATHS[sensor]
    return frame[list(columns)].apply(
        lambda row: any(bool(clean(value)) for value in row), axis=1
    )


def complete_declared(frame: pd.DataFrame, sensor: str) -> pd.Series:
    columns = REQUIRED_PATHS[sensor]
    return frame[list(columns)].apply(
        lambda row: all(bool(clean(value)) for value in row), axis=1
    )


def all_paths(frames: Iterable[pd.DataFrame]) -> list[str]:
    values: set[str] = set()
    required_columns = {
        column
        for columns in REQUIRED_PATHS.values()
        for column in columns
    }
    for frame in frames:
        for column in required_columns:
            values.update(
                text
                for text in frame[column].map(clean).tolist()
                if text
            )
    return sorted(values)


def sanitize_one(
    frame: pd.DataFrame,
    *,
    statuses: dict[str, tuple[str, int]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = frame.copy()
    frame = frame.copy()
    sensor_audit: dict[str, Any] = {}
    retained_masks: dict[str, pd.Series] = {}
    for sensor in SENSOR_ORDER:
        declared_mask = declared(frame, sensor)
        complete_mask = complete_declared(frame, sensor)
        valid_mask = frame[list(REQUIRED_PATHS[sensor])].apply(
            lambda row: bool(complete_mask.loc[row.name])
            and all(
                statuses.get(clean(value), ("missing", 0))[0]
                in {"cache", "source"}
                for value in row
            ),
            axis=1,
        )
        retained_masks[sensor] = valid_mask
        failed_mask = declared_mask & ~valid_mask
        failed_values = frame.loc[
            failed_mask, list(REQUIRED_PATHS[sensor])
        ].copy()
        for column in CLEAR_PATHS[sensor]:
            if column in frame.columns:
                frame.loc[~valid_mask, column] = ""
        failed_paths = Counter()
        for column in REQUIRED_PATHS[sensor]:
            for value in failed_values[column].map(clean):
                if value:
                    failed_paths[
                        statuses.get(value, ("missing", 0))[0]
                    ] += 1
        sensor_audit[sensor] = {
            "declared_rows": int(declared_mask.sum()),
            "complete_declared_rows": int(complete_mask.sum()),
            "retained_rows": int(valid_mask.sum()),
            "removed_sensor_rows": int(failed_mask.sum()),
            "failed_path_references_by_status": dict(
                sorted(failed_paths.items())
            ),
        }
    keep = pd.concat(retained_masks, axis=1).any(axis=1)
    frame = frame.loc[keep].copy().reset_index(drop=True)
    frame["availability_signature"] = [
        "+".join(
            sensor
            for sensor in SENSOR_ORDER
            if all(clean(row[column]) for column in REQUIRED_PATHS[sensor])
        )
        for _, row in frame.iterrows()
    ]
    if (frame["availability_signature"].str.len() == 0).any():
        raise AssertionError("sanitization retained a row without a sensor")
    dropped = before.loc[~keep]
    audit = {
        "rows_before": int(len(before)),
        "rows_after": int(len(frame)),
        "dropped_rows": int(len(before) - len(frame)),
        "labels_before": label_counts(before),
        "labels_after": label_counts(frame),
        "labels_dropped": label_counts(dropped),
        "events": {
            "unique_before": int(before["event_id"].astype(str).nunique()),
            "unique_after": int(frame["event_id"].astype(str).nunique()),
            "unique_in_dropped_rows": int(
                dropped["event_id"].astype(str).nunique()
            ),
        },
        "sensors": sensor_audit,
        "availability_after": {
            str(key): int(value)
            for key, value in frame["availability_signature"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    return frame, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-core", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "train_core": Path(args.train_core).expanduser().absolute(),
        "dev": Path(args.dev).expanduser().absolute(),
        "test": Path(args.test).expanduser().absolute(),
    }
    frames = {
        name: pd.read_csv(
            path, dtype=str, keep_default_na=False, low_memory=False
        )
        for name, path in inputs.items()
    }
    needed_columns = {
        "id",
        "plume_id",
        "label",
        "event_id",
        "query360_index",
        *(
            column
            for columns in REQUIRED_PATHS.values()
            for column in columns
        ),
    }
    for name, frame in frames.items():
        missing = sorted(needed_columns - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    paths = all_paths(frames.values())
    cache_dir = Path(args.cache_dir).expanduser().absolute()
    started = time.monotonic()
    print(
        f"[payload-audit-v2] checking {len(paths)} unique paths with "
        f"{args.workers} workers",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        results = executor.map(
            lambda source: payload_status(source, cache_dir), paths
        )
        statuses = dict(zip(paths, results))
    output_dir = Path(args.output_dir).expanduser().absolute()
    audit: dict[str, Any] = {
        "schema_version": "legacy360-payload-sanitize-v1",
        "cache_dir": str(cache_dir),
        "workers": int(args.workers),
        "elapsed_payload_scan_seconds": float(time.monotonic() - started),
        "unique_paths": len(paths),
        "minimum_bytes": {
            "optical": OPTICAL_MIN_BYTES,
            "s5p": S5P_MIN_BYTES,
        },
        "s5p_validation": "regular .npz, minimum size, ZIP magic; no array decode",
        "path_status": dict(
            sorted(Counter(status for status, _ in statuses.values()).items())
        ),
        "splits": {},
    }
    outputs: dict[str, Path] = {}
    next_query_index = 0
    for name, frame in frames.items():
        sanitized, split_audit = sanitize_one(frame, statuses=statuses)
        sanitized.insert(
            sanitized.columns.get_loc("query360_index"),
            "source_query360_index",
            sanitized["query360_index"].astype(str),
        )
        sanitized["query360_index"] = range(
            next_query_index, next_query_index + len(sanitized)
        )
        next_query_index += len(sanitized)
        destination = output_dir / f"legacy360_{name}_sanitized.csv"
        atomic_csv(destination, sanitized)
        outputs[name] = destination
        split_audit.update(
            {
                "source": str(inputs[name]),
                "source_sha256": sha256_file(inputs[name]),
                "output": str(destination),
                "output_sha256": sha256_file(destination),
            }
        )
        audit["splits"][name] = split_audit
    audit["outputs"] = {name: str(path) for name, path in outputs.items()}
    audit["query360_index"] = {
        "rule": "global contiguous train_core/dev/test after sanitization",
        "rows": int(next_query_index),
        "min": 0 if next_query_index else None,
        "max": next_query_index - 1 if next_query_index else None,
    }
    atomic_json(output_dir / "legacy360_payload_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
