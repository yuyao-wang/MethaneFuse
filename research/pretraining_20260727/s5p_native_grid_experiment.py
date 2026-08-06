#!/usr/bin/env python3
"""Matched-arm S5P six-timepoint screening on an approximate 3x3 grid.

The staged NPZ arrays are 224x224 fields produced by an earlier upsampling
step.  This script summarizes those fields with finite-mask-weighted adaptive
pooling to 3x3.  The result is *not* an inverse of that upsampling and must not
be described as exact native S5P values; it is only an approximate native-scale
summary that preserves finite-value support.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_VERSION = 2
RESEARCH_ROOT = Path("/diniuvol/yuyao/methanefuse_research_20260727")
DEFAULT_TRAIN_CSV = RESEARCH_ROOT / "manifests_staged/s5p/train.csv"
DEFAULT_VAL_CSV = RESEARCH_ROOT / "manifests_staged/s5p/val.csv"
DEFAULT_LOCAL_NPZ_ROOT = RESEARCH_ROOT / "cache/s5p_npz"
DEFAULT_CACHE_DIR = RESEARCH_ROOT / "cache/s5p_native_grid_experiment"

ROLES = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
ROLE_PATH_COLUMNS = tuple(f"{role}_path" for role in ROLES)
ROLE_TO_ID = {role: index for index, role in enumerate(ROLES)}
FALLBACK_OFFSETS_DAYS = {
    "t0": 0.0,
    "prev1": -1.0,
    "prev2": -2.0,
    "prev3": -3.0,
    "seasonal": -90.0,
    "year": -365.0,
}
S5P_TIMESTAMP_RE = re.compile(r"(\d{8}T\d{6})")
EVENT_SUFFIX_RE = re.compile(r"-[A-Za-z0-9]+$")
ARM_NAMES = (
    "t0_masked",
    "role_only",
    "delta_time",
    "history_shuffle_train",
)
NATIVE_GRID_DISCLAIMER = (
    "The cached 3x3 values are finite-mask-weighted adaptive averages of "
    "224x224 fields that were already upsampled upstream. They are an "
    "approximate native-scale summary, not exact original/native S5P values "
    "and not an inverse interpolation."
)


class ArmInputs(NamedTuple):
    features: torch.Tensor
    valid_mask: torch.Tensor
    delta_days: torch.Tensor
    roles: torch.Tensor
    use_delta: bool


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        json_safe(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                json_safe(payload), indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def torch_load_local(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def finite_mask_weighted_adaptive_pool(
    field: torch.Tensor | np.ndarray,
    output_size: tuple[int, int] = (3, 3),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool finite values only and return zero-filled values plus cell validity."""

    values = torch.as_tensor(field, dtype=torch.float32)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3:
        raise ValueError(
            f"Expected field=(T,H,W) or (H,W), got {tuple(values.shape)}"
        )
    if values.shape[-2] < output_size[0] or values.shape[-1] < output_size[1]:
        raise ValueError(
            f"Cannot pool spatial shape {tuple(values.shape[-2:])} "
            f"to {output_size}"
        )

    finite = torch.isfinite(values)
    finite_values = torch.where(finite, values, torch.zeros_like(values))
    numerator = F.adaptive_avg_pool2d(
        finite_values.unsqueeze(1), output_size
    ).squeeze(1)
    finite_fraction = F.adaptive_avg_pool2d(
        finite.to(dtype=torch.float32).unsqueeze(1), output_size
    ).squeeze(1)
    valid = finite_fraction > 0
    pooled = torch.where(
        valid,
        numerator / finite_fraction.clamp_min(torch.finfo(torch.float32).eps),
        torch.zeros_like(numerator),
    )
    if not torch.isfinite(pooled).all():
        raise RuntimeError("Finite-mask pooling produced non-finite output")
    return pooled.contiguous(), valid.contiguous()


def _valid_text(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def canonical_event_id(plume_id: Any) -> str:
    """Match the repository's strip-final-suffix canonical event rule."""

    text = str(plume_id).strip()
    if not text:
        raise ValueError("Cannot canonicalize an empty plume_id")
    return EVENT_SUFFIX_RE.sub("", text)


def parse_s5p_utc_timestamp(path: Any) -> Optional[pd.Timestamp]:
    if not _valid_text(path):
        return None
    match = S5P_TIMESTAMP_RE.search(Path(str(path)).name)
    if match is None:
        return None
    timestamp = pd.to_datetime(
        match.group(1),
        format="%Y%m%dT%H%M%S",
        utc=True,
        errors="coerce",
    )
    return None if pd.isna(timestamp) else timestamp


def utc_delta_days_from_record(
    record: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return acquisition UTC deltas relative to t0 and a fallback mask."""

    plume_time = pd.to_datetime(
        record.get("plume_time"), utc=True, errors="coerce"
    )
    if pd.isna(plume_time):
        raise ValueError(
            f"Invalid plume_time for row {record.get('_row_id')}: "
            f"{record.get('plume_time')!r}"
        )

    timestamps: list[pd.Timestamp] = []
    fallback_mask: list[bool] = []
    for role, column in zip(ROLES, ROLE_PATH_COLUMNS):
        timestamp = parse_s5p_utc_timestamp(record.get(column))
        used_fallback = timestamp is None
        if used_fallback:
            timestamp = plume_time + pd.Timedelta(
                days=FALLBACK_OFFSETS_DAYS[role]
            )
        timestamps.append(timestamp)
        fallback_mask.append(used_fallback)

    reference = timestamps[0]
    deltas = torch.tensor(
        [
            (timestamp - reference).total_seconds() / 86400.0
            for timestamp in timestamps
        ],
        dtype=torch.float32,
    )
    deltas[0] = 0.0
    return deltas, torch.tensor(fallback_mask, dtype=torch.bool)


def _selected_frame_sha256(frame: pd.DataFrame) -> str:
    columns = [
        "_row_id",
        "_event_id",
        "image_path",
        "label",
        "plume_id",
        "plume_time",
        *ROLE_PATH_COLUMNS,
    ]
    hashes = pd.util.hash_pandas_object(
        frame.loc[:, columns].fillna("").astype(str),
        index=False,
    ).to_numpy(dtype=np.uint64, copy=False)
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def _balanced_limit(
    frame: pd.DataFrame,
    limit: int,
    *,
    seed: int,
) -> pd.DataFrame:
    if limit == 0:
        return frame.copy()
    if limit < 2 or limit % 2:
        raise ValueError("A non-zero row limit must be an even integer >= 2")
    per_class = limit // 2
    selected = []
    for label in (0, 1):
        group = frame[frame["label"] == label]
        if len(group) < per_class:
            raise ValueError(
                f"Requested {per_class} label={label} rows but found {len(group)}"
            )
        selected.append(
            group.sample(
                n=per_class,
                replace=False,
                random_state=seed + label,
            )
        )
    return (
        pd.concat(selected, ignore_index=False)
        .sort_values("_row_id", kind="stable")
        .reset_index(drop=True)
    )


def read_staged_manifests(
    train_csv: Path,
    val_csv: Path,
    *,
    max_train: int,
    max_val: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = [
        "image_path",
        "label",
        "plume_id",
        "plume_time",
        *ROLE_PATH_COLUMNS,
    ]
    selected_frames = []
    full_frames = []
    manifest_sha = {}
    for split, path, limit, split_seed in (
        ("train", train_csv, max_train, seed),
        ("val", val_csv, max_val, seed + 10_000),
    ):
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=required, low_memory=False)
        if frame.empty:
            raise ValueError(f"Empty {split} manifest: {path}")
        if frame[required].isna().any().any():
            bad = {
                column: int(frame[column].isna().sum())
                for column in required
                if frame[column].isna().any()
            }
            raise ValueError(f"{split} manifest has missing values: {bad}")
        frame["label"] = pd.to_numeric(
            frame["label"], errors="raise"
        ).astype(np.int64)
        if not frame["label"].isin((0, 1)).all():
            raise ValueError(f"{split} labels are not binary")
        frame["_row_id"] = np.arange(len(frame), dtype=np.int64)
        frame["_event_id"] = frame["plume_id"].map(canonical_event_id)
        if (frame["_event_id"].str.len() == 0).any():
            raise ValueError(f"{split} contains an empty canonical event ID")
        full_frames.append(frame)
        selected_frames.append(
            _balanced_limit(frame, limit, seed=split_seed)
        )
        manifest_sha[split] = file_sha256(path)

    train_full, val_full = full_frames
    train, val = selected_frames
    overlap = set(train_full["_event_id"].astype(str)) & set(
        val_full["_event_id"].astype(str)
    )
    if overlap:
        raise ValueError(
            f"Train/val canonical event overlap={len(overlap)}; "
            f"examples={sorted(overlap)[:10]}"
        )

    audit = {
        "manifest_sha256": manifest_sha,
        "full_rows": {"train": len(train_full), "val": len(val_full)},
        "selected_rows": {"train": len(train), "val": len(val)},
        "selected_sha256": {
            "train": _selected_frame_sha256(train),
            "val": _selected_frame_sha256(val),
        },
        "labels": {
            split: {
                str(label): int(count)
                for label, count in frame["label"]
                .value_counts()
                .sort_index()
                .items()
            }
            for split, frame in (("train", train), ("val", val))
        },
        "canonical_event_rule": "strip-final-hyphen-alphanumeric-suffix-v1",
        "canonical_events": {
            "train": int(train_full["_event_id"].nunique()),
            "val": int(val_full["_event_id"].nunique()),
        },
        "canonical_event_overlap": 0,
        "selection_seed": int(seed),
        "selection": (
            "all rows when limit=0; otherwise deterministic balanced "
            "selection with original row IDs preserved"
        ),
    }
    return train, val, audit


def _resolved_under(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def validate_local_npz_paths(
    frame: pd.DataFrame,
    *,
    local_npz_root: Path,
) -> list[Path]:
    root = local_npz_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Local NPZ root does not exist: {root}")
    paths = []
    for row_id, value in zip(frame["_row_id"], frame["image_path"]):
        path = Path(str(value)).expanduser().resolve()
        if not _resolved_under(path, root):
            raise ValueError(
                f"row_id={row_id} NPZ is outside local root {root}: {path}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"row_id={row_id} missing local NPZ: {path}")
        paths.append(path)
    return paths


def npz_identity_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for index, path in enumerate(paths):
        stat = path.stat()
        identity = (
            f"{index}\0{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        ).encode("utf-8")
        digest.update(identity)
    return digest.hexdigest()


def _load_npz_from_bytes(
    path: Path,
    *,
    data_key: str,
) -> tuple[np.ndarray, str]:
    content = path.read_bytes()
    content_sha = hashlib.sha256(content).hexdigest()
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        if data_key not in archive:
            raise KeyError(
                f"{path}: missing key {data_key!r}; keys={archive.files}"
            )
        array = np.asarray(archive[data_key], dtype=np.float32).copy()
    if array.shape != (6, 224, 224):
        raise ValueError(f"{path}: expected ch4=(6,224,224), got {array.shape}")
    return array, content_sha


def _extract_record(
    record: Mapping[str, Any],
    *,
    data_key: str,
) -> dict[str, Any]:
    path = Path(str(record["image_path"])).expanduser().resolve()
    array, content_sha = _load_npz_from_bytes(path, data_key=data_key)
    features, valid_mask = finite_mask_weighted_adaptive_pool(array, (3, 3))
    delta_days, timestamp_fallback_mask = utc_delta_days_from_record(record)
    return {
        "features": features,
        "valid_mask": valid_mask,
        "delta_days": delta_days,
        "roles": torch.arange(len(ROLES), dtype=torch.long),
        "label": int(record["label"]),
        "row_id": int(record["_row_id"]),
        "plume_id": str(record["plume_id"]),
        "event_id": str(
            record.get("_event_id", canonical_event_id(record["plume_id"]))
        ),
        "timestamp_fallback_mask": timestamp_fallback_mask,
        "npz_content_sha256": content_sha,
    }


def _cache_signature(
    *,
    split: str,
    csv_sha256: str,
    selected_sha256: str,
    identity_sha256: str,
    data_key: str,
) -> str:
    return canonical_sha256(
        {
            "script_version": SCRIPT_VERSION,
            "split": split,
            "manifest_sha256": csv_sha256,
            "selected_rows_sha256": selected_sha256,
            "npz_identity_sha256": identity_sha256,
            "data_key": data_key,
            "source_shape": [6, 224, 224],
            "pool": "finite-mask-weighted adaptive_avg_pool2d",
            "output_shape": [6, 3, 3],
        }
    )


def validate_cache_payload(
    payload: Mapping[str, Any],
    *,
    signature: str,
) -> None:
    required = (
        "features",
        "valid_mask",
        "delta_days",
        "roles",
        "labels",
        "row_ids",
        "plume_ids",
        "event_ids",
        "timestamp_fallback_mask",
        "meta",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Feature cache is missing keys: {missing}")
    if payload["meta"].get("cache_signature_sha256") != signature:
        raise ValueError("Feature cache signature mismatch")
    features = payload["features"]
    valid_mask = payload["valid_mask"]
    rows = features.shape[0]
    if tuple(features.shape[1:]) != (6, 3, 3):
        raise ValueError(f"Unexpected cache feature shape: {features.shape}")
    if valid_mask.shape != features.shape or valid_mask.dtype != torch.bool:
        raise ValueError("Cache valid_mask shape/dtype mismatch")
    if payload["delta_days"].shape != (rows, 6):
        raise ValueError("Cache delta_days shape mismatch")
    if payload["roles"].shape != (rows, 6):
        raise ValueError("Cache roles shape mismatch")
    if payload["labels"].shape != (rows,):
        raise ValueError("Cache labels shape mismatch")
    if len(payload["plume_ids"]) != rows or len(payload["event_ids"]) != rows:
        raise ValueError("Cache plume/event ID length mismatch")
    if not torch.isfinite(features).all():
        raise ValueError("Cache features contain non-finite values")


def build_or_load_feature_cache(
    split: str,
    frame: pd.DataFrame,
    *,
    csv_sha256: str,
    selected_sha256: str,
    local_npz_root: Path,
    cache_dir: Path,
    data_key: str,
    workers: int,
    progress_every: int,
    rebuild: bool,
) -> tuple[dict[str, Any], Path]:
    paths = validate_local_npz_paths(
        frame, local_npz_root=local_npz_root
    )
    identity_sha = npz_identity_sha256(paths)
    signature = _cache_signature(
        split=split,
        csv_sha256=csv_sha256,
        selected_sha256=selected_sha256,
        identity_sha256=identity_sha,
        data_key=data_key,
    )
    cache_path = (
        cache_dir.expanduser().resolve()
        / f"{split}_{signature[:20]}.pt"
    )
    if cache_path.is_file() and not rebuild:
        payload = torch_load_local(cache_path)
        validate_cache_payload(payload, signature=signature)
        print(f"[cache:{split}] reuse {cache_path}", flush=True)
        return payload, cache_path

    records = frame.to_dict("records")
    extract = partial(_extract_record, data_key=data_key)
    if workers <= 1:
        iterator = map(extract, records)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        iterator = executor.map(extract, records)

    count = len(records)
    features = torch.empty((count, 6, 3, 3), dtype=torch.float32)
    valid_mask = torch.empty((count, 6, 3, 3), dtype=torch.bool)
    delta_days = torch.empty((count, 6), dtype=torch.float32)
    roles = torch.empty((count, 6), dtype=torch.long)
    labels = torch.empty((count,), dtype=torch.long)
    row_ids = torch.empty((count,), dtype=torch.long)
    timestamp_fallback_mask = torch.empty((count, 6), dtype=torch.bool)
    plume_ids: list[str] = []
    event_ids: list[str] = []
    content_digest = hashlib.sha256()
    started = time.time()
    try:
        for index, result in enumerate(iterator):
            features[index] = result["features"]
            valid_mask[index] = result["valid_mask"]
            delta_days[index] = result["delta_days"]
            roles[index] = result["roles"]
            labels[index] = result["label"]
            row_ids[index] = result["row_id"]
            timestamp_fallback_mask[index] = result[
                "timestamp_fallback_mask"
            ]
            plume_ids.append(result["plume_id"])
            event_ids.append(result["event_id"])
            content_digest.update(
                f"{result['row_id']}\0".encode("utf-8")
            )
            content_digest.update(
                bytes.fromhex(result["npz_content_sha256"])
            )
            if progress_every and (
                (index + 1) % progress_every == 0 or index + 1 == count
            ):
                print(
                    f"[cache:{split}] {index + 1}/{count} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    npz_content_sha = content_digest.hexdigest()
    input_sha = canonical_sha256(
        {
            "manifest_sha256": csv_sha256,
            "selected_rows_sha256": selected_sha256,
            "npz_content_sha256": npz_content_sha,
            "feature_semantics": NATIVE_GRID_DISCLAIMER,
        }
    )
    meta = {
        "script_version": SCRIPT_VERSION,
        "split": split,
        "rows": count,
        "cache_signature_sha256": signature,
        "manifest_sha256": csv_sha256,
        "selected_rows_sha256": selected_sha256,
        "npz_identity_sha256": identity_sha,
        "npz_content_sha256": npz_content_sha,
        "input_sha256": input_sha,
        "data_key": data_key,
        "source_shape": [6, 224, 224],
        "feature_shape": [6, 3, 3],
        "pooling": (
            "numerator=adaptive_avg_pool(finite_values); "
            "denominator=adaptive_avg_pool(finite_mask); "
            "feature=numerator/denominator where denominator>0"
        ),
        "valid_mask": "pooled cell has at least one finite source value",
        "finite_zero_policy": "finite zero is valid",
        "native_grid_disclaimer": NATIVE_GRID_DISCLAIMER,
        "delta_days": (
            "UTC acquisition timestamp minus t0 acquisition timestamp, "
            "in fractional days; NC filename timestamp preferred"
        ),
        "roles": list(ROLES),
        "timestamp_fallback_cells": int(
            timestamp_fallback_mask.sum().item()
        ),
        "created_unix": time.time(),
    }
    payload = {
        "features": features,
        "valid_mask": valid_mask,
        "delta_days": delta_days,
        "roles": roles,
        "labels": labels,
        "row_ids": row_ids,
        "plume_ids": plume_ids,
        "event_ids": event_ids,
        "timestamp_fallback_mask": timestamp_fallback_mask,
        "meta": meta,
    }
    validate_cache_payload(payload, signature=signature)
    atomic_torch_save(cache_path, payload)
    atomic_json(cache_path.with_suffix(".meta.json"), meta)
    print(
        f"[cache:{split}] wrote {cache_path} input_sha256={input_sha}",
        flush=True,
    )
    return payload, cache_path


def build_cross_event_donor_indices(
    event_ids: Sequence[str],
    *,
    seed: int,
) -> torch.Tensor:
    """Map every row to a deterministic donor from another canonical event."""

    groups: dict[str, list[int]] = {}
    for index, event_id in enumerate(event_ids):
        groups.setdefault(str(event_id), []).append(index)
    if len(groups) < 2:
        raise ValueError("History shuffling requires at least two events")
    rng = random.Random(int(seed))
    event_order = sorted(groups)
    rng.shuffle(event_order)
    donor_event = {
        event_id: event_order[(position + 1) % len(event_order)]
        for position, event_id in enumerate(event_order)
    }
    donors = torch.empty(len(event_ids), dtype=torch.long)
    for event_id, recipient_indices in groups.items():
        candidates = list(groups[donor_event[event_id]])
        rng.shuffle(candidates)
        offset = rng.randrange(len(candidates))
        for position, recipient_index in enumerate(recipient_indices):
            donors[recipient_index] = candidates[
                (offset + position) % len(candidates)
            ]
    for index, donor_index in enumerate(donors.tolist()):
        if str(event_ids[index]) == str(event_ids[donor_index]):
            raise RuntimeError("Cross-event donor construction failed")
    return donors


def shuffle_history_only(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle donor history values/masks while preserving every t0 exactly."""

    if features.ndim != 4 or features.shape[1:] != (6, 3, 3):
        raise ValueError(f"Expected features=(B,6,3,3), got {features.shape}")
    if valid_mask.shape != features.shape:
        raise ValueError("valid_mask must match features")
    if permutation.shape != (features.shape[0],):
        raise ValueError("permutation must have shape (B,)")
    if permutation.device != features.device:
        permutation = permutation.to(features.device)

    shuffled_features = features.clone()
    shuffled_valid = valid_mask.clone()
    shuffled_features[:, 1:] = features[permutation, 1:]
    shuffled_valid[:, 1:] = valid_mask[permutation, 1:]
    return shuffled_features, shuffled_valid


def replace_history_with_donors(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    donor_features: torch.Tensor,
    donor_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace only history with already-aligned cross-event donor rows."""

    if donor_features.shape != features.shape:
        raise ValueError("donor_features must match recipient feature shape")
    if donor_valid_mask.shape != valid_mask.shape:
        raise ValueError("donor_valid_mask must match recipient validity shape")
    shuffled_features = features.clone()
    shuffled_valid = valid_mask.clone()
    shuffled_features[:, 1:] = donor_features[:, 1:]
    shuffled_valid[:, 1:] = donor_valid_mask[:, 1:]
    return shuffled_features, shuffled_valid


def apply_arm_inputs(
    arm: str,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    delta_days: torch.Tensor,
    roles: torch.Tensor,
    *,
    training: bool,
    donor_features: Optional[torch.Tensor] = None,
    donor_valid_mask: Optional[torch.Tensor] = None,
) -> ArmInputs:
    if arm not in ARM_NAMES:
        raise ValueError(f"Unknown arm={arm!r}; expected {ARM_NAMES}")
    if features.shape != valid_mask.shape:
        raise ValueError("features and valid_mask shapes differ")

    use_delta = arm in {"delta_time", "history_shuffle_train"}
    output_features = features
    output_valid = valid_mask
    if arm == "t0_masked":
        output_features = features.clone()
        output_valid = valid_mask.clone()
        output_features[:, 1:] = 0.0
        output_valid[:, 1:] = False
    elif arm == "history_shuffle_train" and training:
        if donor_features is None or donor_valid_mask is None:
            raise ValueError(
                "history_shuffle_train requires aligned cross-event donors"
            )
        output_features, output_valid = replace_history_with_donors(
            features,
            valid_mask,
            donor_features,
            donor_valid_mask,
        )

    return ArmInputs(
        features=output_features,
        valid_mask=output_valid,
        delta_days=delta_days,
        roles=roles,
        use_delta=use_delta,
    )


class NativeGridTemporalClassifier(nn.Module):
    """Small two-layer Transformer over six approximate native-grid frames."""

    def __init__(
        self,
        *,
        train_mean: float,
        train_std: float,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not math.isfinite(train_std) or train_std <= 0:
            raise ValueError(f"Invalid train_std={train_std}")
        self.register_buffer(
            "train_mean", torch.tensor(float(train_mean), dtype=torch.float32)
        )
        self.register_buffer(
            "train_std", torch.tensor(float(train_std), dtype=torch.float32)
        )
        self.grid_projection = nn.Sequential(
            nn.Linear(18, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.role_embedding = nn.Embedding(len(ROLES), hidden_dim)
        self.delta_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=2, norm=nn.LayerNorm(hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        delta_days: torch.Tensor,
        roles: torch.Tensor,
        *,
        use_delta: bool,
    ) -> torch.Tensor:
        if features.ndim != 4 or features.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected features=(B,T,3,3), got {features.shape}"
            )
        if valid_mask.shape != features.shape:
            raise ValueError("valid_mask must match features")
        batch, timepoints = features.shape[:2]
        if delta_days.shape != (batch, timepoints):
            raise ValueError("delta_days shape mismatch")
        if roles.shape != (batch, timepoints):
            raise ValueError("roles shape mismatch")

        normalized = (features - self.train_mean) / self.train_std
        normalized = torch.where(
            valid_mask, normalized, torch.zeros_like(normalized)
        )
        grid_input = torch.cat(
            (
                normalized.flatten(2),
                valid_mask.to(dtype=normalized.dtype).flatten(2),
            ),
            dim=-1,
        )
        tokens = self.grid_projection(grid_input)
        tokens = tokens + self.role_embedding(roles)
        if use_delta:
            scaled_delta = (
                delta_days.div(365.0).clamp(min=-2.0, max=2.0).unsqueeze(-1)
            )
            tokens = tokens + self.delta_projection(scaled_delta)

        time_valid = valid_mask.flatten(2).any(dim=-1)
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat((cls, tokens), dim=1)
        key_padding_mask = torch.cat(
            (
                torch.zeros(
                    (batch, 1), dtype=torch.bool, device=features.device
                ),
                ~time_valid,
            ),
            dim=1,
        )
        encoded = self.temporal_encoder(
            sequence, src_key_padding_mask=key_padding_mask
        )
        return self.classifier(encoded[:, 0]).squeeze(-1)


def training_normalization(
    train_cache: Mapping[str, Any],
) -> dict[str, float | int]:
    features = train_cache["features"]
    valid_mask = train_cache["valid_mask"]
    values = features[valid_mask].to(dtype=torch.float64)
    if values.numel() == 0:
        raise ValueError("Training cache has no valid pooled values")
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    if not math.isfinite(std) or std <= 1e-8:
        raise ValueError(f"Invalid training pooled-value std={std}")
    return {
        "mean": mean,
        "std": std,
        "valid_values": int(values.numel()),
        "source": "selected training cache finite pooled cells only",
    }


def best_macro_f1_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != labels.shape:
        raise ValueError("labels/probabilities must be matching vectors")
    if labels.size == 0:
        raise ValueError("Cannot select threshold from empty arrays")

    order = np.argsort(-probabilities, kind="stable")
    sorted_prob = probabilities[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels == 1)
    cumulative_fp = np.cumsum(sorted_labels == 0)
    group_ends = np.flatnonzero(
        np.r_[sorted_prob[1:] != sorted_prob[:-1], True]
    )
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())

    thresholds = np.r_[1.0, sorted_prob[group_ends]]
    tp = np.r_[0, cumulative_tp[group_ends]].astype(np.float64)
    fp = np.r_[0, cumulative_fp[group_ends]].astype(np.float64)
    fn = positives - tp
    tn = negatives - fp
    positive_den = 2 * tp + fp + fn
    negative_den = 2 * tn + fp + fn
    f1_positive = np.divide(
        2 * tp,
        positive_den,
        out=np.zeros_like(tp),
        where=positive_den > 0,
    )
    f1_negative = np.divide(
        2 * tn,
        negative_den,
        out=np.zeros_like(tn),
        where=negative_den > 0,
    )
    macro_f1 = (f1_positive + f1_negative) / 2.0
    tpr = np.divide(
        tp,
        positives,
        out=np.zeros_like(tp),
        where=positives > 0,
    )
    tnr = np.divide(
        tn,
        negatives,
        out=np.zeros_like(tn),
        where=negatives > 0,
    )
    balanced = (tpr + tnr) / 2.0

    best_value = float(macro_f1.max())
    candidates = np.flatnonzero(np.isclose(macro_f1, best_value))
    best_index = max(
        candidates.tolist(),
        key=lambda index: (
            float(balanced[index]),
            -abs(float(thresholds[index]) - 0.5),
            -index,
        ),
    )
    return float(thresholds[best_index])


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    threshold = best_macro_f1_threshold(labels, probabilities)
    predictions = (probabilities >= threshold).astype(np.int64)
    try:
        ap = float(average_precision_score(labels, probabilities))
    except ValueError:
        ap = float("nan")
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "rows": int(labels.size),
        "positive_rate": float(labels.mean()),
        "ap": ap,
        "auc": auc,
        "macro_f1": float(
            f1_score(
                labels, predictions, average="macro", zero_division=0
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "pred_positive_rate": float(predictions.mean()),
        "best_threshold": threshold,
        "threshold_selection": (
            "maximize validation macro-F1; tie-break by balanced accuracy "
            "then proximity to 0.5"
        ),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _cache_batch(
    cache: Mapping[str, Any],
    indices: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        cache["features"][indices],
        cache["valid_mask"][indices],
        cache["delta_days"][indices],
        cache["roles"][indices],
        cache["labels"][indices],
    )


def evaluate_arm(
    model: NativeGridTemporalClassifier,
    cache: Mapping[str, Any],
    *,
    arm: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.arange(len(cache["labels"]))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    probabilities = []
    targets = []
    with torch.inference_mode():
        for (indices,) in loader:
            features, valid, delta, roles, labels = _cache_batch(
                cache, indices
            )
            inputs = apply_arm_inputs(
                arm,
                features,
                valid,
                delta,
                roles,
                training=False,
            )
            logits = model(
                inputs.features.to(device),
                inputs.valid_mask.to(device),
                inputs.delta_days.to(device),
                inputs.roles.to(device),
                use_delta=inputs.use_delta,
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels.numpy())
    return classification_metrics(
        np.concatenate(targets), np.concatenate(probabilities)
    )


def seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def train_one_arm(
    arm: str,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    *,
    normalization: Mapping[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, Any]:
    seed_everything(seed, device)
    model = NativeGridTemporalClassifier(
        train_mean=float(normalization["mean"]),
        train_std=float(normalization["std"]),
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)
    initial_state_sha = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_labels = train_cache["labels"]
    positives = int((train_labels == 1).sum().item())
    negatives = int((train_labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        raise ValueError("Training cache must contain both labels")
    pos_weight = torch.tensor(
        negatives / positives, dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.arange(len(train_labels))),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )

    history = []
    best_record = None
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        donor_indices = None
        if arm == "history_shuffle_train":
            donor_indices = build_cross_event_donor_indices(
                train_cache["event_ids"],
                seed=seed + 700_001 + epoch,
            )
        for (indices,) in loader:
            features, valid, delta, roles, labels = _cache_batch(
                train_cache, indices
            )
            donor_features = None
            donor_valid = None
            if donor_indices is not None:
                selected_donors = donor_indices[indices]
                donor_features = train_cache["features"][selected_donors]
                donor_valid = train_cache["valid_mask"][selected_donors]
                recipient_events = [
                    train_cache["event_ids"][index]
                    for index in indices.tolist()
                ]
                donor_events = [
                    train_cache["event_ids"][index]
                    for index in selected_donors.tolist()
                ]
                if any(
                    recipient == donor
                    for recipient, donor in zip(
                        recipient_events, donor_events
                    )
                ):
                    raise RuntimeError("History donor shares recipient event")
            inputs = apply_arm_inputs(
                arm,
                features,
                valid,
                delta,
                roles,
                training=True,
                donor_features=donor_features,
                donor_valid_mask=donor_valid,
            )
            labels_device = labels.to(device=device, dtype=torch.float32)
            logits = model(
                inputs.features.to(device),
                inputs.valid_mask.to(device),
                inputs.delta_days.to(device),
                inputs.roles.to(device),
                use_delta=inputs.use_delta,
            )
            loss = criterion(logits, labels_device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            rows = int(labels.numel())
            total_loss += float(loss.item()) * rows
            total_rows += rows

        metrics = evaluate_arm(
            model,
            val_cache,
            arm=arm,
            batch_size=batch_size,
            device=device,
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_rows, 1),
            "val": metrics,
        }
        history.append(record)
        score = (
            -math.inf if not math.isfinite(metrics["ap"]) else metrics["ap"],
            metrics["macro_f1"],
        )
        if best_record is None:
            is_best = True
        else:
            prior = (
                (
                    -math.inf
                    if not math.isfinite(best_record["val"]["ap"])
                    else best_record["val"]["ap"]
                ),
                best_record["val"]["macro_f1"],
            )
            is_best = score > prior
        if is_best:
            best_record = record
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"[{arm}] epoch={epoch}/{epochs} "
            f"loss={record['train_loss']:.6f} "
            f"AP={metrics['ap']:.6f} AUC={metrics['auc']:.6f} "
            f"macro-F1={metrics['macro_f1']:.6f} "
            f"BA={metrics['balanced_accuracy']:.6f} "
            f"pred+={metrics['pred_positive_rate']:.6f} "
            f"threshold={metrics['best_threshold']:.6f}",
            flush=True,
        )

    if best_record is None or best_state is None:
        raise RuntimeError(f"No epoch completed for arm={arm}")
    checkpoint = {
        "arm": arm,
        "model": best_state,
        "best_epoch": int(best_record["epoch"]),
        "normalization": dict(normalization),
        "model_config": {
            "layers": 2,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "dropout": dropout,
        },
        "seed": seed,
        "train_input_sha256": train_cache["meta"]["input_sha256"],
        "val_input_sha256": val_cache["meta"]["input_sha256"],
    }
    atomic_torch_save(checkpoint_path, checkpoint)
    return {
        "arm": arm,
        "initial_state_sha256": initial_state_sha,
        "history": history,
        "best": best_record,
        "checkpoint": str(checkpoint_path.expanduser().resolve()),
    }


def parse_arms(value: str) -> tuple[str, ...]:
    arms = tuple(part.strip() for part in value.split(",") if part.strip())
    if not arms:
        raise ValueError("At least one arm is required")
    unknown = [arm for arm in arms if arm not in ARM_NAMES]
    if unknown:
        raise ValueError(f"Unknown arms={unknown}; expected {ARM_NAMES}")
    if len(set(arms)) != len(arms):
        raise ValueError(f"Duplicate arms are not allowed: {arms}")
    return arms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_csv", "--train-csv", type=Path, default=DEFAULT_TRAIN_CSV
    )
    parser.add_argument(
        "--val_csv", "--val-csv", type=Path, default=DEFAULT_VAL_CSV
    )
    parser.add_argument(
        "--local_npz_root",
        "--local-npz-root",
        type=Path,
        default=DEFAULT_LOCAL_NPZ_ROOT,
    )
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
    )
    parser.add_argument("--output_json", "--output-json", type=Path, required=True)
    parser.add_argument("--data_key", "--data-key", default="ch4")
    parser.add_argument("--max_train", "--max-train", type=int, default=0)
    parser.add_argument("--max_val", "--max-val", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature_workers", "--feature-workers", type=int, default=8)
    parser.add_argument("--progress_every", "--progress-every", type=int, default=500)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=256)
    parser.add_argument("--learning_rate", "--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", "--hidden-dim", type=int, default=64)
    parser.add_argument("--num_heads", "--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--arms", default=",".join(ARM_NAMES))
    parser.add_argument("--rebuild_cache", "--rebuild-cache", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    if not 1 <= args.epochs <= 5:
        raise ValueError("--epochs must be between 1 and 5")
    if args.max_train < 0 or args.max_val < 0:
        raise ValueError("--max_train/--max_val must be non-negative")
    if (args.max_train and args.max_train % 2) or (
        args.max_val and args.max_val % 2
    ):
        raise ValueError("Non-zero row limits must be even")
    if not 1 <= args.feature_workers <= 32:
        raise ValueError("--feature_workers must be in [1,32]")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer hyperparameters")
    if args.hidden_dim <= 0 or args.hidden_dim % args.num_heads:
        raise ValueError("--hidden_dim must be positive and divisible by --num_heads")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0,1)")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return parse_arms(args.arms)


def arm_protocol() -> dict[str, str]:
    return {
        "t0_masked": (
            "Only t0 is attention-visible; all five history feature and "
            "validity tokens are masked."
        ),
        "role_only": (
            "All six coherent frames plus learned role embeddings; actual "
            "UTC delta encoding is disabled."
        ),
        "delta_time": (
            "All six coherent frames plus learned roles and actual UTC "
            "fractional delta-days relative to t0."
        ),
        "history_shuffle_train": (
            "Same model and validation input as delta_time, but training-only "
            "history feature/validity tensors come from deterministic donors "
            "in different canonical events. t0, labels, roles, and delta "
            "schedules stay with the recipient row."
        ),
    }


def main() -> None:
    args = parse_args()
    arms = validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output_json = args.output_json.expanduser().resolve()
    print(f"[native-grid warning] {NATIVE_GRID_DISCLAIMER}", flush=True)

    train, val, manifest_audit = read_staged_manifests(
        args.train_csv,
        args.val_csv,
        max_train=args.max_train,
        max_val=args.max_val,
        seed=args.seed,
    )
    starting = {
        "status": "building_cache",
        "script_version": SCRIPT_VERSION,
        "command": [sys.executable, *sys.argv],
        "args": vars(args),
        "native_grid_disclaimer": NATIVE_GRID_DISCLAIMER,
        "manifest_audit": manifest_audit,
        "arm_protocol": arm_protocol(),
    }
    atomic_json(output_json, starting)

    train_cache, train_cache_path = build_or_load_feature_cache(
        "train",
        train,
        csv_sha256=manifest_audit["manifest_sha256"]["train"],
        selected_sha256=manifest_audit["selected_sha256"]["train"],
        local_npz_root=args.local_npz_root,
        cache_dir=args.cache_dir,
        data_key=args.data_key,
        workers=args.feature_workers,
        progress_every=args.progress_every,
        rebuild=args.rebuild_cache,
    )
    val_cache, val_cache_path = build_or_load_feature_cache(
        "val",
        val,
        csv_sha256=manifest_audit["manifest_sha256"]["val"],
        selected_sha256=manifest_audit["selected_sha256"]["val"],
        local_npz_root=args.local_npz_root,
        cache_dir=args.cache_dir,
        data_key=args.data_key,
        workers=args.feature_workers,
        progress_every=args.progress_every,
        rebuild=args.rebuild_cache,
    )
    normalization = training_normalization(train_cache)
    provenance = {
        **starting,
        "status": "training",
        "device": str(device),
        "cache": {
            "train": str(train_cache_path),
            "val": str(val_cache_path),
            "train_input_sha256": train_cache["meta"]["input_sha256"],
            "val_input_sha256": val_cache["meta"]["input_sha256"],
        },
        "normalization": normalization,
        "model": {
            "type": "two-layer temporal Transformer",
            "layers": 2,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "grid_input": (
                "9 normalized pooled values + 9 cell-validity indicators "
                "per timepoint"
            ),
        },
        "matched_arm_controls": {
            "same_initialization_seed": args.seed,
            "same_optimizer": "AdamW",
            "same_epoch_count": args.epochs,
            "same_train_order_seed": args.seed,
            "history_shuffle_donors": (
                "deterministic full-training-set cross-canonical-event map "
                "per epoch; does not perturb model/dropout RNG"
            ),
        },
    }
    atomic_json(output_json, provenance)

    results = {}
    checkpoint_dir = output_json.parent / f"{output_json.stem}_checkpoints"
    for arm in arms:
        result = train_one_arm(
            arm,
            train_cache,
            val_cache,
            normalization=normalization,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            seed=args.seed,
            device=device,
            checkpoint_path=checkpoint_dir / f"{arm}_best.pt",
        )
        results[arm] = result
        atomic_json(
            output_json,
            {
                **provenance,
                "status": "training",
                "completed_arms": list(results),
                "arms": results,
            },
        )

    initial_hashes = {
        result["initial_state_sha256"] for result in results.values()
    }
    if len(initial_hashes) != 1:
        raise RuntimeError(
            "Matched arms did not start from identical model parameters"
        )
    summary = {
        arm: result["best"]["val"] for arm, result in results.items()
    }
    t0_metrics = summary.get("t0_masked")
    deltas_vs_t0 = {}
    if t0_metrics is not None:
        for arm, metrics in summary.items():
            deltas_vs_t0[arm] = {
                metric: metrics[metric] - t0_metrics[metric]
                for metric in (
                    "ap",
                    "auc",
                    "macro_f1",
                    "balanced_accuracy",
                    "pred_positive_rate",
                )
            }
    final = {
        **provenance,
        "status": "complete",
        "completed_unix": time.time(),
        "arms": results,
        "best_metrics": summary,
        "deltas_vs_t0_masked": deltas_vs_t0,
        "matched_initial_state_sha256": next(iter(initial_hashes)),
        "formal_gpu_run_launched_by_this_command": device.type == "cuda",
    }
    atomic_json(output_json, final)
    print(f"[done] wrote {output_json}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        output_path = None
        for flag in ("--output_json", "--output-json"):
            if flag in sys.argv:
                index = sys.argv.index(flag)
                if index + 1 < len(sys.argv):
                    output_path = Path(sys.argv[index + 1])
                    break
        if output_path is not None:
            prior: dict[str, Any] = {}
            try:
                if output_path.is_file():
                    prior = json.loads(
                        output_path.read_text(encoding="utf-8")
                    )
            except Exception:
                prior = {}
            prior.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_unix": time.time(),
                }
            )
            atomic_json(output_path, prior)
        raise
