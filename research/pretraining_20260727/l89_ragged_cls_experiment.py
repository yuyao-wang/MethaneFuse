#!/usr/bin/env python3
"""Leakage-safe frozen-CLS experiments for ragged L89 temporal sequences.

This runner has two deliberately separate stages:

``cache``
    Read a train or inner-validation CSV, encode every requested L89 frame
    independently with a frozen Panopticon backbone, and atomically materialize
    a provenance-rich feature cache.

``train-heads``
    Train parameter-identical two-layer current-query temporal heads from a
    train/validation cache pair.  The four primary arms differ only in their
    input intervention: t0-only masking, role-only timing, continuous delta
    time, or cross-event history shuffling during training.

The runner never accepts a ``test``/``sealed`` split and refuses CSV/cache paths
with a test-like path component.  The original sensor test manifests therefore
cannot be opened accidentally by this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid xFormers discovery delays in the inherited Panopticon implementation.
os.environ.setdefault("XFORMERS_DISABLED", "1")

from Upgraded_dataset.dino_classifier_head_l89_temporal_satmae import (  # noqa: E402
    DEFAULT_PATH_COLUMNS,
    DEFAULT_TIME_COLUMNS,
    PRECOMPUTED_STATS,
    StaticAnchoredCache,
    load_backbone,
)
from thirdparty.dinov2.data.datasets.s2_csv import S2CsvDataset  # noqa: E402


SCRIPT_VERSION = "l89-ragged-cls-v1"
CACHE_FORMAT_VERSION = 1
ARM_NAMES = (
    "t0_masked",
    "role_only",
    "delta_time",
    "history_shuffle_train",
)
NAT_INT64 = np.iinfo(np.int64).min
EVENT_SUFFIX_RE = re.compile(r"-[A-Za-z0-9]+$")
TEST_LIKE_RE = re.compile(r"(^|[._-])(sealed|test)([._-]|$)", re.IGNORECASE)
DAY_NS = 86_400_000_000_000


def parse_columns(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = value
    columns = tuple(str(part).strip() for part in parts if str(part).strip())
    if not columns:
        raise ValueError("At least one column is required.")
    if len(set(columns)) != len(columns):
        raise ValueError(f"Columns must be unique, got {columns}.")
    return columns


def parse_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        values = tuple(int(part) for part in value)
    if not values or len(set(values)) != len(values) or min(values) < 0:
        raise ValueError(f"Expected unique non-negative integers, got {values}.")
    return values


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(contiguous.shape)))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def atomic_csv_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def torch_load_trusted(path: Path) -> Any:
    """Load a cache created locally by this runner across PyTorch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def assert_not_sealed_path(path: Path, *, purpose: str) -> None:
    resolved = path.expanduser().resolve()
    offending_parts = [part for part in resolved.parts if TEST_LIKE_RE.search(part)]
    if offending_parts:
        raise ValueError(
            f"{purpose} path looks like a sealed/test artifact and is refused: "
            f"{resolved}; offending components={offending_parts}."
        )


def role_from_column(column: str) -> str:
    role = str(column).strip().lower()
    for prefix in ("path_", "time_"):
        if role.startswith(prefix):
            role = role[len(prefix) :]
    for suffix in ("_path", "_image_time", "_time", "_timestamp"):
        if role.endswith(suffix):
            role = role[: -len(suffix)]
    return role


def find_t0_index(path_columns: Sequence[str]) -> int:
    roles = tuple(role_from_column(column) for column in path_columns)
    indices = [index for index, role in enumerate(roles) if role == "t0"]
    if len(indices) != 1:
        raise ValueError(
            f"Exactly one path column must resolve to role 't0'; "
            f"columns={tuple(path_columns)}, roles={roles}."
        )
    return indices[0]


def validate_role_alignment(
    path_columns: Sequence[str], time_columns: Sequence[str]
) -> tuple[str, ...]:
    if len(path_columns) != len(time_columns):
        raise ValueError(
            f"path/time column counts differ: {len(path_columns)} != {len(time_columns)}"
        )
    path_roles = tuple(role_from_column(column) for column in path_columns)
    time_roles = tuple(role_from_column(column) for column in time_columns)
    if path_roles != time_roles:
        raise ValueError(
            "Path/time columns must describe the same ordered roles; "
            f"path_roles={path_roles}, time_roles={time_roles}."
        )
    return path_roles


@dataclass(frozen=True)
class TemporalMetadata:
    timestamps_utc_ns: torch.Tensor
    timestamps_utc_iso: list[list[str]]
    timestamp_valid_mask: torch.Tensor
    delta_days: torch.Tensor
    role_names: tuple[str, ...]
    role_index: torch.Tensor
    t0_index: int


def build_temporal_metadata(
    frame: pd.DataFrame,
    path_columns: Sequence[str],
    time_columns: Sequence[str],
) -> TemporalMetadata:
    role_names = validate_role_alignment(path_columns, time_columns)
    t0_index = find_t0_index(path_columns)
    missing = [column for column in time_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing timestamp columns: {missing}.")

    timestamp_columns: list[np.ndarray] = []
    iso_columns: list[list[str]] = []
    valid_columns: list[np.ndarray] = []
    for column in time_columns:
        parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
        index = pd.DatetimeIndex(parsed)
        values = index.asi8.copy()
        valid = values != NAT_INT64
        timestamp_columns.append(values)
        valid_columns.append(valid)
        iso_columns.append(
            [
                timestamp.isoformat().replace("+00:00", "Z")
                if not pd.isna(timestamp)
                else ""
                for timestamp in index
            ]
        )

    timestamp_ns = np.stack(timestamp_columns, axis=1)
    timestamp_valid = np.stack(valid_columns, axis=1)
    t0_ns = timestamp_ns[:, t0_index]
    t0_valid = timestamp_valid[:, t0_index]
    delta = np.full(timestamp_ns.shape, np.nan, dtype=np.float64)
    pair_valid = timestamp_valid & t0_valid[:, None]
    row_indices, time_indices = np.nonzero(pair_valid)
    delta[row_indices, time_indices] = (
        timestamp_ns[row_indices, time_indices] - t0_ns[row_indices]
    ) / float(DAY_NS)
    delta[:, t0_index] = np.where(t0_valid, 0.0, np.nan)

    iso_by_row = [
        [iso_columns[time_index][row_index] for time_index in range(len(time_columns))]
        for row_index in range(len(frame))
    ]
    return TemporalMetadata(
        timestamps_utc_ns=torch.from_numpy(timestamp_ns.astype(np.int64, copy=False)),
        timestamps_utc_iso=iso_by_row,
        timestamp_valid_mask=torch.from_numpy(timestamp_valid),
        delta_days=torch.from_numpy(delta.astype(np.float32)),
        role_names=role_names,
        role_index=torch.arange(len(role_names), dtype=torch.long),
        t0_index=t0_index,
    )


def compute_duplicate_unique_masks(
    timestamps_utc_ns: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    t0_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse exact-time aliases among valid observations.

    Returns ``duplicate_mask`` (suppressed aliases), ``unique_mask`` (one
    retained observation per timestamp), and ``duplicate_group_mask`` (all
    members of a repeated-timestamp group).  t0 wins a tie, followed by stable
    input-role order.
    """

    if timestamps_utc_ns.shape != valid_mask.shape or timestamps_utc_ns.ndim != 2:
        raise ValueError(
            "timestamps_utc_ns and valid_mask must have the same (N,T) shape."
        )
    timestamps = timestamps_utc_ns.detach().cpu().numpy()
    valid = valid_mask.detach().cpu().numpy().astype(bool, copy=False)
    duplicate = np.zeros_like(valid)
    duplicate_group = np.zeros_like(valid)
    unique = np.zeros_like(valid)
    for row_index in range(timestamps.shape[0]):
        groups: dict[int, list[int]] = {}
        for time_index in range(timestamps.shape[1]):
            timestamp = int(timestamps[row_index, time_index])
            if not valid[row_index, time_index] or timestamp == NAT_INT64:
                continue
            groups.setdefault(timestamp, []).append(time_index)
        for indices in groups.values():
            ordered = sorted(indices, key=lambda index: (index != t0_index, index))
            keeper = ordered[0]
            unique[row_index, keeper] = True
            if len(ordered) > 1:
                duplicate_group[row_index, ordered] = True
                duplicate[row_index, ordered[1:]] = True
    return (
        torch.from_numpy(duplicate),
        torch.from_numpy(unique),
        torch.from_numpy(duplicate_group),
    )


def canonical_event_ids(
    frame: pd.DataFrame,
    *,
    event_column: str,
    plume_id_column: str,
) -> tuple[list[str], str]:
    if event_column and event_column in frame.columns:
        values = frame[event_column].astype("string").str.strip()
        source = event_column
    elif plume_id_column in frame.columns:
        values = (
            frame[plume_id_column]
            .astype("string")
            .str.strip()
            .str.replace(EVENT_SUFFIX_RE, "", regex=True)
        )
        source = f"{plume_id_column}:strip-final-hyphen-suffix"
    else:
        raise ValueError(
            f"Cannot build event IDs: neither {event_column!r} nor "
            f"{plume_id_column!r} exists."
        )
    invalid = values.isna() | values.eq("")
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} rows have empty canonical event IDs.")
    return values.astype(str).tolist(), source


def string_column(
    frame: pd.DataFrame,
    column: str,
    *,
    fallback: Optional[Sequence[str]] = None,
) -> list[str]:
    if column and column in frame.columns:
        values = frame[column].fillna("").astype(str).str.strip()
        if values.eq("").any():
            raise ValueError(f"Column {column!r} contains empty values.")
        return values.tolist()
    if fallback is not None:
        return [str(value) for value in fallback]
    return [str(index) for index in range(len(frame))]


def validate_binary_labels(frame: pd.DataFrame, label_column: str) -> torch.Tensor:
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column {label_column!r}.")
    numeric = pd.to_numeric(frame[label_column], errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{int(numeric.isna().sum())} labels are non-numeric.")
    labels = numeric.to_numpy(dtype=np.int64)
    observed = set(labels.tolist())
    if not observed.issubset({0, 1}):
        raise ValueError(f"Labels must be binary, got {sorted(observed)}.")
    return torch.from_numpy(labels)


def deterministic_stratified_limit(
    frame: pd.DataFrame,
    *,
    label_column: str,
    maximum_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Select a deterministic approximately balanced subset for smoke caches."""

    maximum_rows = int(maximum_rows)
    if maximum_rows <= 0 or maximum_rows >= len(frame):
        return frame.copy()
    labels = validate_binary_labels(frame, label_column).numpy()
    rng = np.random.default_rng(int(seed))
    by_class = {
        label: np.flatnonzero(labels == label)
        for label in (0, 1)
    }
    target = {
        0: maximum_rows // 2,
        1: maximum_rows - maximum_rows // 2,
    }
    selected: list[int] = []
    for label in (0, 1):
        count = min(target[label], len(by_class[label]))
        if count:
            selected.extend(
                rng.choice(by_class[label], size=count, replace=False).tolist()
            )
    if len(selected) < maximum_rows:
        remaining = np.setdiff1d(
            np.arange(len(frame), dtype=np.int64),
            np.asarray(selected, dtype=np.int64),
            assume_unique=False,
        )
        fill = min(maximum_rows - len(selected), len(remaining))
        selected.extend(rng.choice(remaining, size=fill, replace=False).tolist())
    if len(selected) != maximum_rows:
        raise RuntimeError(
            f"Could select only {len(selected)}/{maximum_rows} requested rows."
        )
    # Retain canonical CSV order after deterministic sampling.
    return frame.iloc[sorted(selected)].copy()


def load_normalization_stats(path: Optional[Path]) -> tuple[list[float], list[float], str]:
    if path is None:
        return list(PRECOMPUTED_STATS[0]), list(PRECOMPUTED_STATS[1]), "precomputed"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        mean = payload.get("mean")
        std = payload.get("std")
    elif isinstance(payload, Sequence) and len(payload) == 2:
        mean, std = payload
    else:
        raise ValueError(f"Unsupported normalization JSON schema: {path}.")
    mean_values = [float(value) for value in mean]
    std_values = [float(value) for value in std]
    if len(mean_values) != len(std_values) or not mean_values:
        raise ValueError("Normalization mean/std lengths are invalid.")
    if not all(math.isfinite(value) for value in mean_values + std_values):
        raise ValueError("Normalization mean/std must be finite.")
    if min(std_values) <= 0:
        raise ValueError("Normalization std values must be positive.")
    return mean_values, std_values, f"json:{path.resolve()}:{sha256_file(path)}"


class L89FrameCacheDataset(Dataset):
    """Strict row-preserving loader that never substitutes another sample."""

    STATUS_MISSING = 0
    STATUS_USABLE = 1
    STATUS_LOW_COVERAGE = 2
    STATUS_READ_ERROR = 3

    def __init__(
        self,
        csv_path: Path,
        frame: pd.DataFrame,
        *,
        path_columns: Sequence[str],
        band_indices: Sequence[int],
        mean: Sequence[float],
        std: Sequence[float],
        image_size: int,
        min_valid_fraction: float,
        validity_band_index: int,
        local_file_cache: Optional[StaticAnchoredCache],
        local_cache_bypass_root: Optional[Path],
        zero_invalid_pixels: bool,
    ):
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.path_columns = tuple(path_columns)
        self.band_indices = tuple(int(index) for index in band_indices)
        self.image_size = int(image_size)
        self.min_valid_fraction = float(min_valid_fraction)
        self.local_file_cache = local_file_cache
        self.local_cache_bypass_root = (
            local_cache_bypass_root.expanduser().resolve()
            if local_cache_bypass_root is not None
            else None
        )
        self.zero_invalid_pixels = bool(zero_invalid_pixels)
        self.reader = S2CsvDataset(
            csv_path=str(csv_path),
            ds_cfg_name="landsat89_7band",
            path_column=self.path_columns[0],
            normalize_stats=None,
            scale_to_unit=False,
            compute_stats=False,
            pad_to_multiple=None,
            skip_invalid_samples=False,
            path_columns_for_validation=self.path_columns,
        )
        available = int(self.reader.chn_ids.shape[0])
        if any(index >= available for index in self.band_indices):
            raise ValueError(
                f"Band indices {self.band_indices} exceed available channels 0..{available - 1}."
            )
        if validity_band_index not in self.band_indices:
            raise ValueError(
                f"validity_band_index={validity_band_index} is absent from "
                f"selected bands {self.band_indices}."
            )
        self.validity_band_position = self.band_indices.index(validity_band_index)
        if len(mean) != available or len(std) != available:
            raise ValueError(
                f"Expected {available} normalization values, got {len(mean)}/{len(std)}."
            )
        self.mean = torch.tensor(mean, dtype=torch.float32)[
            list(self.band_indices)
        ].view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32)[
            list(self.band_indices)
        ].view(-1, 1, 1)
        self.channel_ids = torch.as_tensor(self.reader.chn_ids, dtype=torch.float32)[
            list(self.band_indices)
        ]

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_path(self, value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None
        path = value.strip()
        if self.local_cache_bypass_root is not None:
            try:
                Path(os.path.abspath(os.path.expanduser(path))).relative_to(
                    self.local_cache_bypass_root
                )
                return path
            except ValueError:
                pass
        if self.local_file_cache is not None:
            path = self.local_file_cache.ensure_local(path)
        return path

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        timepoints = len(self.path_columns)
        channels = len(self.band_indices)
        images = torch.zeros(
            timepoints,
            channels,
            self.image_size,
            self.image_size,
            dtype=torch.float32,
        )
        image_valid = torch.zeros(timepoints, dtype=torch.bool)
        valid_fraction = torch.zeros(timepoints, dtype=torch.float32)
        status = torch.full(
            (timepoints,), self.STATUS_MISSING, dtype=torch.int8
        )
        for time_index, column in enumerate(self.path_columns):
            path = self._resolve_path(row[column])
            if path is None:
                continue
            try:
                raw = self.reader._read_image_raw(path)[list(self.band_indices)]
                native_valid = torch.isfinite(raw) & raw.ne(0)
                fraction = native_valid[self.validity_band_position].float().mean()
                valid_fraction[time_index] = fraction
                if float(fraction) < self.min_valid_fraction:
                    status[time_index] = self.STATUS_LOW_COVERAGE
                    continue
                normalized = (torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0) - self.mean) / self.std
                if self.zero_invalid_pixels:
                    normalized = torch.where(
                        native_valid,
                        normalized,
                        torch.zeros((), dtype=normalized.dtype),
                    )
                normalized = F.interpolate(
                    normalized.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                images[time_index] = normalized
                image_valid[time_index] = True
                status[time_index] = self.STATUS_USABLE
            except Exception:
                status[time_index] = self.STATUS_READ_ERROR
        return (
            torch.tensor(index, dtype=torch.long),
            images,
            image_valid,
            valid_fraction,
            status,
        )


def input_table_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    selected = frame.loc[:, list(columns)].copy()
    for column in selected.columns:
        selected[column] = selected[column].fillna("").astype(str)
    payload = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256_bytes(payload)


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def cache_features(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv).expanduser().resolve()
    output_path = Path(args.output_cache).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    assert_not_sealed_path(csv_path, purpose="CSV")
    assert_not_sealed_path(output_path, purpose="cache")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"A real frozen Panopticon checkpoint is required: {weights_path}"
        )
    csv_sha_before = sha256_file(csv_path)
    weights_sha_before = sha256_file(weights_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Cache already exists: {output_path}; pass --overwrite to replace it atomically."
        )

    path_columns = parse_columns(args.path_columns)
    time_columns = parse_columns(args.time_columns)
    if len(path_columns) not in {3, 6}:
        raise ValueError(
            f"Ragged L89 caches support exactly 3 or 6 roles, got {len(path_columns)}."
        )
    band_indices = parse_ints(args.band_indices)
    frame = pd.read_csv(csv_path, low_memory=False)
    source_rows = len(frame)
    frame = deterministic_stratified_limit(
        frame,
        label_column=args.label_column,
        maximum_rows=args.max_rows,
        seed=args.row_selection_seed,
    )
    if frame.empty:
        raise ValueError(f"No rows selected from {csv_path}.")
    required = set(path_columns) | set(time_columns) | {args.label_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}.")

    temporal = build_temporal_metadata(frame, path_columns, time_columns)
    labels = validate_binary_labels(frame, args.label_column)
    plume_ids = string_column(frame, args.plume_id_column)
    ids = string_column(frame, args.id_column, fallback=plume_ids)
    event_ids, event_rule = canonical_event_ids(
        frame,
        event_column=args.event_column,
        plume_id_column=args.plume_id_column,
    )
    mean, std, stats_source = load_normalization_stats(
        Path(args.stats_json).expanduser().resolve() if args.stats_json else None
    )

    cache_object = None
    if args.local_cache_mode != "off":
        cache_object = StaticAnchoredCache(
            args.local_cache_dir,
            min_free_gb=args.local_cache_min_free_gb,
            async_mode=args.local_cache_mode == "async",
            max_workers=args.local_cache_workers,
        )
    dataset = L89FrameCacheDataset(
        csv_path,
        frame,
        path_columns=path_columns,
        band_indices=band_indices,
        mean=mean,
        std=std,
        image_size=args.image_size,
        min_valid_fraction=args.min_valid_fraction,
        validity_band_index=args.validity_band_index,
        local_file_cache=cache_object,
        local_cache_bypass_root=(
            Path(args.local_cache_bypass_root)
            if args.local_cache_bypass_root
            else None
        ),
        zero_invalid_pixels=args.zero_invalid_pixels,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers
    loader = DataLoader(dataset, **loader_kwargs)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    backbone = load_backbone(str(weights_path), device=device, debug=args.debug).to(device)
    backbone.requires_grad_(False)
    backbone.eval()

    rows = len(frame)
    timepoints = len(path_columns)
    feature_dim = int(backbone.embed_dim)
    storage_dtype = torch.float16 if args.storage_dtype == "float16" else torch.float32
    features = torch.zeros(rows, timepoints, feature_dim, dtype=storage_dtype)
    image_valid_mask = torch.zeros(rows, timepoints, dtype=torch.bool)
    valid_fraction = torch.zeros(rows, timepoints, dtype=torch.float32)
    load_status = torch.zeros(rows, timepoints, dtype=torch.int8)
    seen = torch.zeros(rows, dtype=torch.bool)
    channel_ids = dataset.channel_ids

    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            indices, images, batch_image_valid, batch_fraction, batch_status = batch
            batch_size, batch_timepoints, channels, height, width = images.shape
            flat_images = images.reshape(
                batch_size * batch_timepoints, channels, height, width
            ).to(device, non_blocking=True)
            flat_channel_ids = (
                channel_ids.view(1, -1)
                .expand(batch_size * batch_timepoints, -1)
                .clone()
                .to(device, non_blocking=True)
            )
            with autocast_context(device, args.amp_dtype):
                output = backbone.forward_features(
                    {"imgs": flat_images, "chn_ids": flat_channel_ids}
                )
                batch_features = output["x_norm_clstoken"].reshape(
                    batch_size, batch_timepoints, feature_dim
                )
            batch_features = batch_features.float().cpu()
            batch_features[~batch_image_valid] = 0.0
            indices = indices.long()
            if seen[indices].any():
                raise RuntimeError("A cache row was emitted more than once.")
            seen[indices] = True
            features[indices] = batch_features.to(storage_dtype)
            image_valid_mask[indices] = batch_image_valid
            valid_fraction[indices] = batch_fraction
            load_status[indices] = batch_status
            if (
                batch_index % max(1, args.log_interval) == 0
                or batch_index == len(loader)
            ):
                print(
                    f"[cache] batches={batch_index}/{len(loader)} "
                    f"rows={int(seen.sum())}/{rows} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    if not seen.all():
        missing_rows = torch.nonzero(~seen, as_tuple=False).flatten().tolist()[:20]
        raise RuntimeError(f"Feature cache missed rows: {missing_rows}.")

    valid_mask = image_valid_mask & temporal.timestamp_valid_mask
    duplicate_mask, unique_mask, duplicate_group_mask = compute_duplicate_unique_masks(
        temporal.timestamps_utc_ns,
        valid_mask,
        t0_index=temporal.t0_index,
    )
    features[~unique_mask] = 0.0
    t0_invalid = int((~unique_mask[:, temporal.t0_index]).sum())
    read_errors = int(
        load_status.eq(L89FrameCacheDataset.STATUS_READ_ERROR).sum()
    )
    if t0_invalid > int(args.max_invalid_t0):
        raise RuntimeError(
            f"{t0_invalid} rows have unusable t0 evidence, exceeding "
            f"--max-invalid-t0={args.max_invalid_t0}; no cache was written."
        )
    if read_errors > int(args.max_read_errors):
        raise RuntimeError(
            f"{read_errors} declared source artifacts failed to read, exceeding "
            f"--max-read-errors={args.max_read_errors}; no cache was written."
        )

    csv_sha = sha256_file(csv_path)
    weights_sha = sha256_file(weights_path)
    if csv_sha != csv_sha_before:
        raise RuntimeError("CSV contents changed during feature extraction.")
    if weights_sha != weights_sha_before:
        raise RuntimeError("Panopticon checkpoint changed during feature extraction.")
    table_columns = [
        args.id_column,
        args.label_column,
        args.plume_id_column,
        *path_columns,
        *time_columns,
    ]
    table_columns = [column for column in dict.fromkeys(table_columns) if column in frame]
    table_sha = input_table_sha256(frame, table_columns)
    input_contract = {
        "script_version": SCRIPT_VERSION,
        "csv_sha256": csv_sha,
        "weights_sha256": weights_sha,
        "input_table_sha256": table_sha,
        "path_columns": list(path_columns),
        "time_columns": list(time_columns),
        "role_names": list(temporal.role_names),
        "band_indices": list(band_indices),
        "channel_ids": channel_ids.tolist(),
        "normalization_mean": mean,
        "normalization_std": std,
        "normalization_source": stats_source,
        "image_size": int(args.image_size),
        "min_valid_fraction": float(args.min_valid_fraction),
        "validity_band_index": int(args.validity_band_index),
        "zero_invalid_pixels": bool(args.zero_invalid_pixels),
        "local_cache_bypass_root": (
            str(Path(args.local_cache_bypass_root).expanduser().resolve())
            if args.local_cache_bypass_root
            else ""
        ),
        "duplicate_rule": "exact-utc-ns-valid-evidence-t0-then-role-order-v1",
        "source_rows": int(source_rows),
        "selected_rows": int(len(frame)),
        "row_selection": (
            "all"
            if not args.max_rows or int(args.max_rows) >= source_rows
            else "deterministic-label-stratified-v1"
        ),
        "row_selection_seed": int(args.row_selection_seed),
    }
    input_contract_sha = sha256_bytes(canonical_json_bytes(input_contract))
    payload: dict[str, Any] = {
        "format_version": CACHE_FORMAT_VERSION,
        "script_version": SCRIPT_VERSION,
        "split": args.split,
        "features": features,
        "labels": labels,
        "ids": ids,
        "plume_ids": plume_ids,
        "event_ids": event_ids,
        "event_id_rule": event_rule,
        "timestamps_utc_ns": temporal.timestamps_utc_ns,
        "timestamps_utc_iso": temporal.timestamps_utc_iso,
        "timestamp_valid_mask": temporal.timestamp_valid_mask,
        "delta_days": temporal.delta_days,
        "role_names": list(temporal.role_names),
        "role_index": temporal.role_index,
        "t0_index": int(temporal.t0_index),
        "image_valid_mask": image_valid_mask,
        "valid_mask": valid_mask,
        "duplicate_mask": duplicate_mask,
        "duplicate_group_mask": duplicate_group_mask,
        "unique_mask": unique_mask,
        "valid_fraction": valid_fraction,
        "load_status": load_status,
        "path_columns": list(path_columns),
        "time_columns": list(time_columns),
        "input_contract": input_contract,
        "input_contract_sha256": input_contract_sha,
        "csv_path": str(csv_path),
        "csv_sha256": csv_sha,
        "weights_path": str(weights_path),
        "weights_sha256": weights_sha,
        "input_table_sha256": table_sha,
        "feature_sha256": tensor_sha256(features),
        "label_sha256": tensor_sha256(labels),
        "timestamp_sha256": tensor_sha256(temporal.timestamps_utc_ns),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "elapsed_seconds": float(time.monotonic() - started),
    }
    atomic_torch_save(output_path, payload)
    cache_sha = sha256_file(output_path)
    summary = {
        "cache_path": str(output_path),
        "cache_sha256": cache_sha,
        "split": args.split,
        "rows": rows,
        "timepoints": timepoints,
        "feature_dim": feature_dim,
        "storage_dtype": str(features.dtype),
        "usable_t0_rows": int(unique_mask[:, temporal.t0_index].sum()),
        "invalid_t0_rows": t0_invalid,
        "usable_observations": int(valid_mask.sum()),
        "unique_observations": int(unique_mask.sum()),
        "suppressed_duplicate_observations": int(duplicate_mask.sum()),
        "read_errors": read_errors,
        "input_contract_sha256": input_contract_sha,
        "csv_sha256": csv_sha,
        "weights_sha256": weights_sha,
        "feature_sha256": payload["feature_sha256"],
    }
    atomic_json_write(output_path.with_suffix(output_path.suffix + ".json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


REQUIRED_CACHE_KEYS = {
    "format_version",
    "script_version",
    "split",
    "features",
    "labels",
    "ids",
    "plume_ids",
    "event_ids",
    "timestamps_utc_ns",
    "delta_days",
    "role_names",
    "role_index",
    "t0_index",
    "valid_mask",
    "duplicate_mask",
    "unique_mask",
    "input_contract",
    "input_contract_sha256",
    "csv_sha256",
    "weights_sha256",
    "feature_sha256",
}


def validate_cache_payload(
    payload: Mapping[str, Any],
    *,
    path: Path,
    expected_split: str,
) -> None:
    missing = sorted(REQUIRED_CACHE_KEYS - set(payload))
    if missing:
        raise ValueError(f"{path} is missing cache keys: {missing}.")
    if int(payload["format_version"]) != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"{path} format_version={payload['format_version']} != {CACHE_FORMAT_VERSION}."
        )
    if payload["split"] != expected_split:
        raise ValueError(
            f"{path} split={payload['split']!r}; expected {expected_split!r}."
        )
    features = payload["features"]
    labels = payload["labels"]
    timestamps = payload["timestamps_utc_ns"]
    delta = payload["delta_days"]
    valid = payload["valid_mask"]
    unique = payload["unique_mask"]
    duplicate = payload["duplicate_mask"]
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError(f"{path} features must be a tensor shaped (N,T,D).")
    rows, timepoints, feature_dim = features.shape
    if rows <= 0 or timepoints not in {3, 6} or feature_dim <= 0:
        raise ValueError(
            f"{path} invalid feature shape {tuple(features.shape)}; expected T=3 or 6."
        )
    for name, tensor in (
        ("timestamps_utc_ns", timestamps),
        ("delta_days", delta),
        ("valid_mask", valid),
        ("unique_mask", unique),
        ("duplicate_mask", duplicate),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (rows, timepoints):
            raise ValueError(
                f"{path} {name} shape must be {(rows, timepoints)}, "
                f"got {getattr(tensor, 'shape', None)}."
            )
    if labels.shape != (rows,):
        raise ValueError(f"{path} labels shape must be {(rows,)}, got {labels.shape}.")
    for name in ("ids", "plume_ids", "event_ids"):
        if len(payload[name]) != rows:
            raise ValueError(f"{path} {name} length does not equal N={rows}.")
    if len(payload["role_names"]) != timepoints:
        raise ValueError(f"{path} role_names length does not equal T={timepoints}.")
    if payload["role_index"].shape != (timepoints,):
        raise ValueError(f"{path} role_index must be shaped {(timepoints,)}.")
    t0_index = int(payload["t0_index"])
    if not 0 <= t0_index < timepoints:
        raise ValueError(f"{path} has invalid t0_index={t0_index}.")
    if not torch.equal(unique & duplicate, torch.zeros_like(unique, dtype=torch.bool)):
        raise ValueError(f"{path} unique_mask overlaps suppressed duplicate_mask.")
    if (unique & ~valid).any():
        raise ValueError(f"{path} unique_mask is not a subset of valid_mask.")
    observed = set(labels.long().tolist())
    if observed != {0, 1}:
        raise ValueError(f"{path} must contain both binary classes, got {observed}.")
    if tensor_sha256(features) != payload["feature_sha256"]:
        raise ValueError(f"{path} feature SHA does not match its cache metadata.")
    expected_contract_sha = sha256_bytes(
        canonical_json_bytes(payload["input_contract"])
    )
    if expected_contract_sha != payload["input_contract_sha256"]:
        raise ValueError(f"{path} input contract SHA is invalid.")


def comparable_input_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {
        "csv_sha256",
        "input_table_sha256",
        "source_rows",
        "selected_rows",
        "row_selection",
        "row_selection_seed",
    }
    return {key: value for key, value in contract.items() if key not in ignored}


def load_cache_pair(
    train_path: Path,
    val_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    assert_not_sealed_path(train_path, purpose="train cache")
    assert_not_sealed_path(val_path, purpose="validation cache")
    train = torch_load_trusted(train_path)
    val = torch_load_trusted(val_path)
    if not isinstance(train, dict) or not isinstance(val, dict):
        raise ValueError("Both cache files must contain dictionaries.")
    validate_cache_payload(train, path=train_path, expected_split="train")
    validate_cache_payload(val, path=val_path, expected_split="val")
    for key in ("weights_sha256", "role_names", "t0_index", "path_columns", "time_columns"):
        if train.get(key) != val.get(key):
            raise ValueError(f"Train/validation cache mismatch for {key!r}.")
    if train["features"].shape[1:] != val["features"].shape[1:]:
        raise ValueError("Train/validation feature shapes differ.")
    if comparable_input_contract(train["input_contract"]) != comparable_input_contract(
        val["input_contract"]
    ):
        raise ValueError(
            "Train/validation input contracts differ beyond their CSV/table hashes."
        )
    overlap = sorted(set(train["event_ids"]) & set(val["event_ids"]))
    if overlap:
        raise ValueError(
            f"Train/validation caches overlap by {len(overlap)} canonical events; "
            f"examples={overlap[:10]}."
        )
    audit = {
        "train_cache": str(train_path),
        "train_cache_sha256": sha256_file(train_path),
        "validation_cache": str(val_path),
        "validation_cache_sha256": sha256_file(val_path),
        "weights_sha256": train["weights_sha256"],
        "event_overlap": 0,
        "timepoints": int(train["features"].shape[1]),
        "feature_dim": int(train["features"].shape[2]),
        "role_names": list(train["role_names"]),
        "t0_index": int(train["t0_index"]),
    }
    return train, val, audit


class ContinuousDeltaEncoder(nn.Module):
    """Fourier and signed-log embedding for real-valued temporal lags."""

    def __init__(
        self,
        output_dim: int,
        periods_days: Sequence[float],
        *,
        maximum_days: float = 4000.0,
    ):
        super().__init__()
        periods = torch.tensor(tuple(float(value) for value in periods_days))
        if periods.numel() == 0 or torch.any(periods <= 0):
            raise ValueError("All Fourier periods must be positive.")
        self.register_buffer("periods_days", periods, persistent=True)
        self.maximum_days = float(maximum_days)
        input_dim = 2 + 2 * int(periods.numel())
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, delta_days: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(delta_days)
        safe = torch.where(finite, delta_days, torch.zeros_like(delta_days))
        safe = safe.clamp(min=-self.maximum_days, max=self.maximum_days)
        denominator = math.log1p(self.maximum_days)
        signed_log = torch.sign(safe) * torch.log1p(torch.abs(safe)) / denominator
        absolute_log = torch.log1p(torch.abs(safe)) / denominator
        angle = (
            2.0
            * math.pi
            * safe.unsqueeze(-1)
            / self.periods_days.to(device=safe.device, dtype=safe.dtype)
        )
        encoded = torch.cat(
            (
                signed_log.unsqueeze(-1),
                absolute_log.unsqueeze(-1),
                torch.sin(angle),
                torch.cos(angle),
            ),
            dim=-1,
        )
        output = self.mlp(encoded)
        return output * finite.unsqueeze(-1).to(output.dtype)


class CurrentQueryBlock(nn.Module):
    """One cross-attention/FFN block that updates only the current query."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        hidden_dim = int(round(model_dim * mlp_ratio))
        self.query_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        query = query + self.attention_dropout(attended)
        return query + self.ffn(self.ffn_norm(query))


class RaggedCurrentQueryHead(nn.Module):
    """Parameter-identical head shared by all input-intervention arms."""

    def __init__(
        self,
        feature_dim: int,
        num_roles: int,
        *,
        model_dim: int = 256,
        num_heads: int = 8,
        depth: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        periods_days: Sequence[float] = (1, 3, 7, 30, 90, 365),
        t0_index: int = 0,
    ):
        super().__init__()
        if depth != 2:
            raise ValueError("The preregistered current-query head has exactly 2 layers.")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads.")
        if not 0 <= t0_index < num_roles:
            raise ValueError("t0_index is out of range.")
        self.t0_index = int(t0_index)
        self.input_projection = nn.Linear(feature_dim, model_dim)
        self.role_embedding = nn.Embedding(num_roles, model_dim)
        self.delta_encoder = ContinuousDeltaEncoder(model_dim, periods_days)
        self.input_norm = nn.LayerNorm(model_dim)
        self.blocks = nn.ModuleList(
            [
                CurrentQueryBlock(
                    model_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        role_index: torch.Tensor,
        delta_days: torch.Tensor,
        *,
        enable_delta: bool,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape (B,T,D).")
        batch_size, timepoints, _ = features.shape
        if valid_mask.shape != (batch_size, timepoints):
            raise ValueError("valid_mask shape does not match features.")
        if delta_days.shape != (batch_size, timepoints):
            raise ValueError("delta_days shape does not match features.")
        if role_index.ndim == 1:
            role_index = role_index.view(1, -1).expand(batch_size, -1)
        if role_index.shape != (batch_size, timepoints):
            raise ValueError("role_index shape does not match features.")
        if not valid_mask[:, self.t0_index].all():
            raise ValueError("Every head input must retain a valid t0 query.")

        delta_embedding = self.delta_encoder(delta_days)
        # Always execute the delta encoder.  Multiplication, rather than a
        # separate model, keeps parameter shapes and dense forward work matched.
        delta_embedding = delta_embedding * float(bool(enable_delta))
        context = (
            self.input_projection(features)
            + self.role_embedding(role_index)
            + delta_embedding
        )
        context = self.input_norm(context)
        query = context[:, self.t0_index : self.t0_index + 1]
        for block in self.blocks:
            query = block(query, context, valid_mask)
        return self.classifier(self.output_norm(query[:, 0])).squeeze(-1)


def model_parameter_signature(model: nn.Module) -> dict[str, Any]:
    shapes = {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
        }
        for name, parameter in model.named_parameters()
    }
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "parameter_shapes": shapes,
        "shape_sha256": sha256_bytes(canonical_json_bytes(shapes)),
    }


def build_cross_event_donor_indices(
    event_ids: Sequence[str],
    *,
    seed: int,
) -> torch.Tensor:
    """Map every row to a donor from a different canonical event."""

    groups: dict[str, list[int]] = {}
    for index, event_id in enumerate(event_ids):
        groups.setdefault(str(event_id), []).append(index)
    if len(groups) < 2:
        raise ValueError("History shuffling requires at least two canonical events.")
    rng = random.Random(int(seed))
    event_order = sorted(groups)
    rng.shuffle(event_order)
    donor_event = {
        event_id: event_order[(position + 1) % len(event_order)]
        for position, event_id in enumerate(event_order)
    }
    donors = torch.empty(len(event_ids), dtype=torch.long)
    for event_id, target_indices in groups.items():
        candidates = list(groups[donor_event[event_id]])
        rng.shuffle(candidates)
        offset = rng.randrange(len(candidates))
        for position, target_index in enumerate(target_indices):
            donors[target_index] = candidates[(offset + position) % len(candidates)]
    for index, donor_index in enumerate(donors.tolist()):
        if str(event_ids[index]) == str(event_ids[donor_index]):
            raise RuntimeError("Cross-event donor construction failed.")
    return donors


def apply_history_donors(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    unique_mask: torch.Tensor,
    delta_days: torch.Tensor,
    donor_indices: torch.Tensor,
    *,
    t0_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if donor_indices.shape != (features.shape[0],):
        raise ValueError("donor_indices must have one entry per row.")
    history = [index for index in range(features.shape[1]) if index != t0_index]
    shuffled_features = features.clone()
    shuffled_valid = valid_mask.clone()
    shuffled_unique = unique_mask.clone()
    shuffled_delta = delta_days.clone()
    shuffled_features[:, history] = features[donor_indices][:, history]
    shuffled_valid[:, history] = valid_mask[donor_indices][:, history]
    shuffled_unique[:, history] = unique_mask[donor_indices][:, history]
    shuffled_delta[:, history] = delta_days[donor_indices][:, history]
    return shuffled_features, shuffled_valid, shuffled_unique, shuffled_delta


def prepare_arm_inputs(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    unique_mask: torch.Tensor,
    delta_days: torch.Tensor,
    *,
    arm: str,
    t0_index: int,
    donor_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if arm not in ARM_NAMES:
        raise ValueError(f"Unknown arm {arm!r}.")
    working_features = features.clone()
    working_valid = valid_mask.clone()
    working_unique = unique_mask.clone()
    working_delta = delta_days.clone()
    if arm == "history_shuffle_train":
        if donor_indices is None:
            raise ValueError("history_shuffle_train requires donor_indices.")
        (
            working_features,
            working_valid,
            working_unique,
            working_delta,
        ) = apply_history_donors(
            working_features,
            working_valid,
            working_unique,
            working_delta,
            donor_indices,
            t0_index=t0_index,
        )

    effective_valid = working_valid & working_unique
    history = [index for index in range(features.shape[1]) if index != t0_index]
    if arm == "t0_masked":
        working_features[:, history] = 0.0
        effective_valid[:, history] = False
        working_delta[:, history] = 0.0
    enable_delta = arm in {"delta_time", "history_shuffle_train"}
    if not enable_delta:
        working_delta.zero_()
    if not effective_valid[:, t0_index].all():
        raise ValueError("Prepared arm inputs contain an invalid t0.")
    return working_features, effective_valid, working_delta, enable_delta


def select_usable_rows(payload: Mapping[str, Any]) -> torch.Tensor:
    t0_index = int(payload["t0_index"])
    usable = payload["valid_mask"][:, t0_index] & payload["unique_mask"][:, t0_index]
    indices = torch.nonzero(usable, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError("No rows have usable unique t0 evidence.")
    labels = payload["labels"][indices].long()
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Usable rows must contain both classes.")
    return indices


def take_rows(payload: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    index_list = indices.tolist()
    return {
        "features": payload["features"][indices].float(),
        "labels": payload["labels"][indices].float(),
        "valid_mask": payload["valid_mask"][indices].bool(),
        "unique_mask": payload["unique_mask"][indices].bool(),
        "delta_days": payload["delta_days"][indices].float(),
        "ids": [payload["ids"][index] for index in index_list],
        "plume_ids": [payload["plume_ids"][index] for index in index_list],
        "event_ids": [payload["event_ids"][index] for index in index_list],
    }


def fixed_epoch_batches(
    rows: int,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> list[torch.Tensor]:
    if shuffle:
        generator = torch.Generator().manual_seed(int(seed) + int(epoch))
        order = torch.randperm(rows, generator=generator)
    else:
        order = torch.arange(rows)
    return list(order.split(batch_size))


def event_balanced_row_weights(event_ids: Sequence[str]) -> np.ndarray:
    """Give every canonical event total weight one, irrespective of row count."""
    values = np.asarray([str(value) for value in event_ids], dtype=object)
    if values.ndim != 1 or len(values) == 0 or any(not value for value in values):
        raise ValueError("event_ids must be a non-empty vector of non-blank strings.")
    unique, inverse, counts = np.unique(
        values, return_inverse=True, return_counts=True
    )
    if len(unique) == 0 or np.any(counts <= 0):
        raise ValueError("Could not construct event-balanced row weights.")
    return 1.0 / counts[inverse].astype(np.float64)


def best_weighted_positive_f1_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray,
    *,
    tie_target: float = 0.5,
) -> tuple[float, float]:
    """Exact weighted positive-F1 maximizer for probability >= threshold."""
    target = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if (
        target.ndim != 1
        or probability.shape != target.shape
        or weights.shape != target.shape
        or len(target) == 0
    ):
        raise ValueError("labels, probabilities, and weights must share non-empty shape.")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("Weighted F1 threshold selection requires both classes.")
    if not np.isfinite(probability).all() or not np.isfinite(weights).all():
        raise ValueError("Threshold inputs must be finite.")
    if np.any(weights <= 0):
        raise ValueError("Threshold sample weights must be positive.")
    order = np.argsort(-probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_target = target[order]
    sorted_weights = weights[order]
    cumulative_tp = np.cumsum(sorted_weights * sorted_target)
    cumulative_fp = np.cumsum(sorted_weights * (1 - sorted_target))
    total_positive = float(np.sum(weights * target))
    ends = np.flatnonzero(
        np.r_[sorted_probability[:-1] != sorted_probability[1:], True]
    )
    tp = cumulative_tp[ends]
    fp = cumulative_fp[ends]
    fn = total_positive - tp
    denominator = 2.0 * tp + fp + fn
    scores = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp),
        where=denominator > 0,
    )
    thresholds = sorted_probability[ends]
    best_score = float(scores.max())
    tied = np.flatnonzero(np.isclose(scores, best_score, atol=1e-12, rtol=0.0))
    distances = np.abs(thresholds[tied] - float(tie_target))
    closest = tied[np.flatnonzero(distances == distances.min())]
    chosen = int(closest[np.argmax(thresholds[closest])])
    return float(thresholds[chosen]), best_score


def evaluate_head(
    model: RaggedCurrentQueryHead,
    data: Mapping[str, Any],
    *,
    arm: str,
    role_index: torch.Tensor,
    t0_index: int,
    batch_size: int,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    loss_sum = 0.0
    rows = int(data["labels"].shape[0])
    batches = fixed_epoch_batches(
        rows, batch_size=batch_size, seed=0, epoch=0, shuffle=False
    )
    with torch.inference_mode():
        for indices in batches:
            evaluation_arm = (
                "delta_time" if arm == "history_shuffle_train" else arm
            )
            (
                features,
                effective_valid,
                delta_days,
                enable_delta,
            ) = prepare_arm_inputs(
                data["features"][indices],
                data["valid_mask"][indices],
                data["unique_mask"][indices],
                data["delta_days"][indices],
                arm=evaluation_arm,
                t0_index=t0_index,
            )
            labels = data["labels"][indices].to(device)
            logits = model(
                features.to(device),
                effective_valid.to(device),
                role_index.to(device),
                delta_days.to(device),
                enable_delta=enable_delta,
            )
            loss = criterion(logits, labels)
            loss_sum += float(loss) * len(indices)
            probabilities.append(torch.sigmoid(logits).cpu())
            targets.append(labels.cpu())
    probability = torch.cat(probabilities).numpy()
    target = torch.cat(targets).numpy().astype(np.int64)
    prediction = (probability >= 0.5).astype(np.int64)
    event_weights = event_balanced_row_weights(data["event_ids"])
    event_threshold, event_selected_f1 = best_weighted_positive_f1_threshold(
        target, probability, event_weights
    )
    event_selected_prediction = (probability >= event_threshold).astype(np.int64)
    tn = int(np.sum((target == 0) & (prediction == 0)))
    fp = int(np.sum((target == 0) & (prediction == 1)))
    fn = int(np.sum((target == 1) & (prediction == 0)))
    tp = int(np.sum((target == 1) & (prediction == 1)))
    metrics = {
        "loss": loss_sum / rows,
        "ap": float(average_precision_score(target, probability)),
        "auc": float(roc_auc_score(target, probability)),
        "macro_f1_at_0_5": float(
            f1_score(target, prediction, average="macro", zero_division=0)
        ),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(target, prediction)
        ),
        "event_balanced_ap": float(
            average_precision_score(
                target, probability, sample_weight=event_weights
            )
        ),
        "event_balanced_auc": float(
            roc_auc_score(target, probability, sample_weight=event_weights)
        ),
        "event_balanced_positive_f1_at_0_5": float(
            f1_score(
                target,
                prediction,
                average="binary",
                sample_weight=event_weights,
                zero_division=0,
            )
        ),
        "event_balanced_macro_f1_at_0_5": float(
            f1_score(
                target,
                prediction,
                labels=[0, 1],
                average="macro",
                sample_weight=event_weights,
                zero_division=0,
            )
        ),
        "event_balanced_selected_positive_f1": float(event_selected_f1),
        "event_balanced_selected_threshold": float(event_threshold),
        "event_balanced_selected_macro_f1": float(
            f1_score(
                target,
                event_selected_prediction,
                labels=[0, 1],
                average="macro",
                sample_weight=event_weights,
                zero_division=0,
            )
        ),
        "canonical_event_count": int(len(set(data["event_ids"]))),
        "pred_positive_rate_at_0_5": float(prediction.mean()),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "probability_mean": float(probability.mean()),
        "probability_std": float(probability.std()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "rows": rows,
    }
    return metrics, probability


def train_one_arm(
    arm: str,
    *,
    args: argparse.Namespace,
    train_data: Mapping[str, Any],
    val_data: Mapping[str, Any],
    role_index: torch.Tensor,
    t0_index: int,
    initial_state: Mapping[str, torch.Tensor],
    parameter_signature: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    periods = tuple(float(value) for value in args.delta_periods.split(","))
    model = RaggedCurrentQueryHead(
        feature_dim=int(train_data["features"].shape[-1]),
        num_roles=int(train_data["features"].shape[1]),
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        depth=2,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        periods_days=periods,
        t0_index=t0_index,
    ).to(args.resolved_device)
    model.load_state_dict(initial_state, strict=True)
    if model_parameter_signature(model) != parameter_signature:
        raise RuntimeError(f"{arm}: model parameter signature changed.")
    # Give each arm the same dropout and optimizer stochastic stream.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    labels = train_data["labels"]
    positives = float(labels.sum())
    negatives = float(labels.numel() - positives)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            negatives / max(positives, 1.0), device=args.resolved_device
        )
    )
    history: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        donor_indices = None
        if arm == "history_shuffle_train":
            donor_indices = build_cross_event_donor_indices(
                train_data["event_ids"],
                seed=args.seed + 100_003 * epoch,
            )
            (
                epoch_features,
                epoch_valid,
                epoch_delta,
                epoch_enable_delta,
            ) = prepare_arm_inputs(
                train_data["features"],
                train_data["valid_mask"],
                train_data["unique_mask"],
                train_data["delta_days"],
                arm=arm,
                t0_index=t0_index,
                donor_indices=donor_indices,
            )
        else:
            epoch_features = None
            epoch_valid = None
            epoch_delta = None
            epoch_enable_delta = False
        batches = fixed_epoch_batches(
            int(labels.shape[0]),
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=epoch,
            shuffle=True,
        )
        if args.max_train_steps:
            batches = batches[: int(args.max_train_steps)]
        train_loss_sum = 0.0
        train_rows = 0
        for step, indices in enumerate(batches, 1):
            if arm == "history_shuffle_train":
                features = epoch_features[indices]
                effective_valid = epoch_valid[indices]
                delta_days = epoch_delta[indices]
                enable_delta = epoch_enable_delta
            else:
                (
                    features,
                    effective_valid,
                    delta_days,
                    enable_delta,
                ) = prepare_arm_inputs(
                    train_data["features"][indices],
                    train_data["valid_mask"][indices],
                    train_data["unique_mask"][indices],
                    train_data["delta_days"][indices],
                    arm=arm,
                    t0_index=t0_index,
                )
            batch_labels = labels[indices].to(args.resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                features.to(args.resolved_device),
                effective_valid.to(args.resolved_device),
                role_index.to(args.resolved_device),
                delta_days.to(args.resolved_device),
                enable_delta=enable_delta,
            )
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{arm} epoch={epoch} step={step} loss is non-finite.")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(indices)
            train_rows += len(indices)
        val_metrics, probabilities = evaluate_head(
            model,
            val_data,
            arm=arm,
            role_index=role_index,
            t0_index=t0_index,
            batch_size=args.eval_batch_size,
            device=args.resolved_device,
            criterion=criterion,
        )
        record = {
            "arm": arm,
            "epoch": epoch,
            "optimizer_steps": len(batches),
            "train_rows_seen": train_rows,
            "train_loss": train_loss_sum / max(train_rows, 1),
            "validation": val_metrics,
            "elapsed_seconds": float(time.monotonic() - started),
        }
        history.append(record)
        if best is None or val_metrics["ap"] > best["validation"]["ap"]:
            best = copy.deepcopy(record)
            checkpoint = {
                "script_version": SCRIPT_VERSION,
                "arm": arm,
                "epoch": epoch,
                "model": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                "model_parameter_signature": dict(parameter_signature),
                "initial_state_sha256": state_dict_sha256(initial_state),
                "cache_audit": dict(cache_audit),
                "validation": val_metrics,
                "args": {
                    key: value
                    for key, value in vars(args).items()
                    if key != "resolved_device"
                },
            }
            atomic_torch_save(arm_dir / "checkpoint_best_ap.pt", checkpoint)
            predictions = pd.DataFrame(
                {
                    "id": val_data["ids"],
                    "plume_id": val_data["plume_ids"],
                    "event_id": val_data["event_ids"],
                    "label": val_data["labels"].long().tolist(),
                    "probability": probabilities,
                    "prediction_at_0_5": (probabilities >= 0.5).astype(np.int64),
                    "prediction_at_event_balanced_selected_threshold": (
                        probabilities
                        >= val_metrics["event_balanced_selected_threshold"]
                    ).astype(np.int64),
                    "arm": arm,
                    "epoch": epoch,
                }
            )
            atomic_csv_write(arm_dir / "validation_best_ap_predictions.csv", predictions)
        atomic_json_write(arm_dir / "metrics_history.json", history)
        print(
            f"[head] arm={arm} epoch={epoch}/{args.epochs} "
            f"loss={record['train_loss']:.6f} "
            f"AP={val_metrics['ap']:.6f} AUC={val_metrics['auc']:.6f} "
            f"macroF1={val_metrics['macro_f1_at_0_5']:.6f} "
            f"BA={val_metrics['balanced_accuracy_at_0_5']:.6f} "
            f"pred+={val_metrics['pred_positive_rate_at_0_5']:.6f}",
            flush=True,
        )
    if best is None:
        raise RuntimeError(f"{arm} produced no epoch result.")
    return history, best


def train_heads(args: argparse.Namespace) -> None:
    train_path = Path(args.train_cache).expanduser().resolve()
    val_path = Path(args.val_cache).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("Both --train-cache and --val-cache must exist.")
    assert_not_sealed_path(output_dir, purpose="head output")
    if (output_dir / "summary.json").exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_dir}/summary.json already exists; pass --overwrite to update atomically."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(
        output_dir / "run_status.json",
        {
            "status": "running",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "script_version": SCRIPT_VERSION,
        },
    )
    try:
        train_cache, val_cache, cache_audit = load_cache_pair(train_path, val_path)
        train_indices = select_usable_rows(train_cache)
        val_indices = select_usable_rows(val_cache)
        train_data = take_rows(train_cache, train_indices)
        val_data = take_rows(val_cache, val_indices)
        t0_index = int(train_cache["t0_index"])
        role_index = train_cache["role_index"].long()
        arms = parse_columns(args.arms)
        unknown_arms = sorted(set(arms) - set(ARM_NAMES))
        if unknown_arms:
            raise ValueError(f"Unknown arms: {unknown_arms}; expected {ARM_NAMES}.")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = torch.device(args.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        args.resolved_device = device
        periods = tuple(float(value) for value in args.delta_periods.split(","))
        prototype = RaggedCurrentQueryHead(
            feature_dim=int(train_data["features"].shape[-1]),
            num_roles=int(train_data["features"].shape[1]),
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            depth=2,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            periods_days=periods,
            t0_index=t0_index,
        )
        initial_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in prototype.state_dict().items()
        }
        parameter_signature = model_parameter_signature(prototype)
        initial_state_sha = state_dict_sha256(initial_state)
        del prototype

        run_config = {
            "script_version": SCRIPT_VERSION,
            "arms": list(arms),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "model_dim": int(args.model_dim),
            "num_heads": int(args.num_heads),
            "depth": 2,
            "mlp_ratio": float(args.mlp_ratio),
            "dropout": float(args.dropout),
            "delta_periods_days": list(periods),
            "train_rows_total": int(train_cache["features"].shape[0]),
            "train_rows_usable": int(train_indices.numel()),
            "validation_rows_total": int(val_cache["features"].shape[0]),
            "validation_rows_usable": int(val_indices.numel()),
            "parameter_signature": parameter_signature,
            "initial_state_sha256": initial_state_sha,
            "matched_compute_contract": {
                "same_initial_state": True,
                "same_model_parameter_shapes": True,
                "same_static_timepoints": int(train_data["features"].shape[1]),
                "same_epoch_batches": True,
                "same_optimizer_and_steps": True,
                "delta_encoder_executed_for_every_arm": True,
                "t0_history_features_zeroed": True,
                "t0_history_validity_zeroed": True,
                "history_shuffle_scope": "training-only-cross-canonical-event",
                "validation_input": (
                    "arm-matched; history_shuffle_train uses unshuffled "
                    "full-delta validation"
                ),
            },
            "cache_audit": cache_audit,
        }
        atomic_json_write(output_dir / "run_config.json", run_config)

        all_history: dict[str, list[dict[str, Any]]] = {}
        best_by_arm: dict[str, dict[str, Any]] = {}
        for arm in arms:
            history, best = train_one_arm(
                arm,
                args=args,
                train_data=train_data,
                val_data=val_data,
                role_index=role_index,
                t0_index=t0_index,
                initial_state=initial_state,
                parameter_signature=parameter_signature,
                cache_audit=cache_audit,
                output_dir=output_dir,
            )
            all_history[arm] = history
            best_by_arm[arm] = best
            atomic_json_write(output_dir / "metrics_history.json", all_history)
            atomic_json_write(
                output_dir / "summary.json",
                {
                    "script_version": SCRIPT_VERSION,
                    "best_by_arm": best_by_arm,
                    "selection_metric": "validation_ap",
                    "sealed_test_read": False,
                    "initial_state_sha256": initial_state_sha,
                    "parameter_signature": parameter_signature,
                    "cache_audit": cache_audit,
                },
            )
        atomic_json_write(
            output_dir / "run_status.json",
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "arms": list(arms),
                "sealed_test_read": False,
            },
        )
    except Exception as error:
        atomic_json_write(
            output_dir / "run_status.json",
            {
                "status": "failed",
                "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "sealed_test_read": False,
            },
        )
        raise


def add_boolean_pair(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    destination = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=destination, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{name}",
        dest=destination,
        action="store_false",
        help=f"Disable: {help_text}",
    )
    parser.set_defaults(**{destination: default})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Panopticon per-frame CLS cache and matched-compute ragged "
            "L89 temporal-head experiments."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser(
        "cache", help="Stage A: atomically create a frozen per-frame CLS cache."
    )
    cache_parser.add_argument("--csv", required=True)
    cache_parser.add_argument("--split", choices=("train", "val"), required=True)
    cache_parser.add_argument("--output-cache", required=True)
    cache_parser.add_argument(
        "--weights", default="weights/panopticon_vitb14_teacher.pth"
    )
    cache_parser.add_argument("--path-columns", default=DEFAULT_PATH_COLUMNS)
    cache_parser.add_argument("--time-columns", default=DEFAULT_TIME_COLUMNS)
    cache_parser.add_argument("--label-column", default="label")
    cache_parser.add_argument("--id-column", default="id")
    cache_parser.add_argument("--plume-id-column", default="plume_id")
    cache_parser.add_argument("--event-column", default="event_group_id")
    cache_parser.add_argument("--band-indices", default="0,1,2,3,4,5,6")
    cache_parser.add_argument("--stats-json")
    cache_parser.add_argument("--image-size", type=int, default=224)
    cache_parser.add_argument("--min-valid-fraction", type=float, default=0.75)
    cache_parser.add_argument("--validity-band-index", type=int, default=0)
    add_boolean_pair(
        cache_parser,
        "zero-invalid-pixels",
        default=True,
        help_text="Zero native invalid pixels after normalization.",
    )
    cache_parser.add_argument("--batch-size", type=int, default=16)
    cache_parser.add_argument("--num-workers", type=int, default=8)
    cache_parser.add_argument("--prefetch-factor", type=int, default=2)
    add_boolean_pair(
        cache_parser,
        "persistent-workers",
        default=True,
        help_text="Keep DataLoader workers alive for the cache pass.",
    )
    cache_parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    cache_parser.add_argument(
        "--amp-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    cache_parser.add_argument(
        "--storage-dtype", choices=("float16", "float32"), default="float16"
    )
    cache_parser.add_argument(
        "--local-cache-dir", default="/diniuvol/yuyao/l89_ragged_cls_file_cache"
    )
    cache_parser.add_argument(
        "--local-cache-bypass-root",
        default="/diniuvol/yuyao",
        help="Files already below this local root are read in place, not copied again.",
    )
    cache_parser.add_argument(
        "--local-cache-mode", choices=("off", "sync", "async"), default="sync"
    )
    cache_parser.add_argument("--local-cache-workers", type=int, default=2)
    cache_parser.add_argument("--local-cache-min-free-gb", type=float, default=20.0)
    cache_parser.add_argument("--max-rows", type=int, default=0)
    cache_parser.add_argument("--row-selection-seed", type=int, default=20260727)
    cache_parser.add_argument("--max-invalid-t0", type=int, default=0)
    cache_parser.add_argument(
        "--max-read-errors",
        type=int,
        default=0,
        help=(
            "Fail closed when more declared source artifacts fail to read; "
            "explicitly raise only for an audited quarantine smoke."
        ),
    )
    cache_parser.add_argument("--log-interval", type=int, default=20)
    cache_parser.add_argument("--overwrite", action="store_true")
    cache_parser.add_argument("--debug", action="store_true")
    cache_parser.set_defaults(handler=cache_features)

    train_parser = subparsers.add_parser(
        "train-heads",
        help="Stage B: train parameter-identical temporal heads from two caches.",
    )
    train_parser.add_argument("--train-cache", required=True)
    train_parser.add_argument("--val-cache", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--arms", default=",".join(ARM_NAMES))
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--eval-batch-size", type=int, default=512)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.05)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--model-dim", type=int, default=256)
    train_parser.add_argument("--num-heads", type=int, default=8)
    train_parser.add_argument("--mlp-ratio", type=float, default=2.0)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--delta-periods", default="1,3,7,30,90,365")
    train_parser.add_argument("--max-train-steps", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=20260727)
    train_parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.set_defaults(handler=train_heads)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
