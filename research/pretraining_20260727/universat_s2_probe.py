#!/usr/bin/env python3
"""Frozen UniverSat probe for six-date, 12-band Sentinel-2 methane crops.

The runner intentionally separates the expensive frozen-backbone pass from the
small supervised probe:

1. Read staged CSV manifests and 12-band TIFFs.
2. Compute per-band mean/std from the selected training rows only.
3. Extract mean+max pooled UniverSat features into resumable shards.
4. Train a small classification head for a few epochs.
5. Write validation AP, AUROC, positive-class F1, and the best threshold to JSON.

Network access is disabled by default.  The official UniverSat repository and
checkpoint may both be supplied as local paths; a Hugging Face cache can also
be used with ``local_files_only=True``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, Subset


SCRIPT_VERSION = 1
DEFAULT_RESEARCH_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727"
)
DEFAULT_UNIVERSAT_REPO = DEFAULT_RESEARCH_ROOT / "external/UniverSat"
DEFAULT_MODEL_SOURCE = DEFAULT_RESEARCH_ROOT / "external/UniverSat_checkpoint"
DEFAULT_FEATURE_CACHE = (
    DEFAULT_RESEARCH_ROOT / "cache/universat_s2_features"
)
DEFAULT_HF_CACHE = DEFAULT_RESEARCH_ROOT / "cache/huggingface"

DEFAULT_PATH_COLUMNS = (
    "path_t0",
    "path_prev1",
    "path_prev2",
    "path_prev3",
    "path_seasonal",
    "path_year",
)
DEFAULT_TIME_COLUMNS = (
    "t0_image_time",
    "prev1_image_time",
    "prev2_image_time",
    "prev3_image_time",
    "seasonal_image_time",
    "year_image_time",
)

# Local S2 v14 order:
# B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12.
# Values are micrometres, matching UniverSat's official sensor registry.
S2_WAVELENGTHS_UM = (
    0.442,
    0.490,
    0.560,
    0.665,
    0.705,
    0.740,
    0.783,
    0.833,
    0.865,
    0.944,
    1.612,
    2.190,
)


def parse_columns(value: str | Sequence[str]) -> tuple[str, ...]:
    """Parse a comma-separated column list while accepting existing tuples."""
    if isinstance(value, str):
        columns = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        columns = tuple(str(part).strip() for part in value if str(part).strip())
    if not columns:
        raise argparse.ArgumentTypeError("At least one column is required")
    return columns


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_artifact_identity(path: Path) -> dict[str, Any]:
    """Return a cheap identity without hashing the 804 MB checkpoint."""
    identity: dict[str, Any] = {"path": str(path.resolve())}
    if not path.exists():
        identity["exists"] = False
        return identity
    identity["exists"] = True
    candidates = (
        (path / "config.json", path / "model.safetensors")
        if path.is_dir()
        else (path,)
    )
    files = []
    for candidate in candidates:
        if candidate.is_file():
            stat = candidate.stat()
            item = {
                "name": candidate.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
            if stat.st_size <= 1024 * 1024:
                item["sha256"] = sha256_file(candidate)
            files.append(item)
    identity["files"] = files
    return identity


def manifest_fingerprint(
    csv_path: Path,
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    stat = csv_path.stat()
    selected = frame.loc[:, list(columns)].fillna("").astype(str)
    row_hashes = pd.util.hash_pandas_object(selected, index=False).to_numpy(
        dtype=np.uint64, copy=False
    )
    digest = hashlib.sha256()
    digest.update(str(csv_path.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def stratified_limit(
    frame: pd.DataFrame,
    limit: int | None,
    *,
    label_column: str,
    seed: int,
) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame.copy()
    if limit <= 0:
        raise ValueError("Sample limits must be positive")

    labels = sorted(frame[label_column].unique().tolist())
    if limit < len(labels):
        raise ValueError(
            f"Limit {limit} is smaller than the number of labels {len(labels)}"
        )
    groups = {
        label: frame.index[frame[label_column] == label].to_numpy()
        for label in labels
    }
    total = len(frame)
    ideal = {label: limit * len(indices) / total for label, indices in groups.items()}
    quotas = {
        label: min(len(groups[label]), max(1, int(math.floor(ideal[label]))))
        for label in labels
    }

    while sum(quotas.values()) > limit:
        candidates = [label for label in labels if quotas[label] > 1]
        if not candidates:
            raise RuntimeError("Could not construct a stratified sample")
        label = max(candidates, key=lambda item: (quotas[item] - ideal[item], item))
        quotas[label] -= 1
    while sum(quotas.values()) < limit:
        candidates = [
            label for label in labels if quotas[label] < len(groups[label])
        ]
        if not candidates:
            break
        label = max(candidates, key=lambda item: (ideal[item] - quotas[item], -item))
        quotas[label] += 1

    rng = np.random.default_rng(seed)
    chosen = []
    for label in labels:
        chosen.extend(
            rng.choice(groups[label], size=quotas[label], replace=False).tolist()
        )
    return frame.loc[sorted(chosen)].copy()


def label_counts(frame: pd.DataFrame, label_column: str) -> dict[str, int]:
    return {
        str(int(label)): int(count)
        for label, count in frame[label_column].value_counts().sort_index().items()
    }


def read_and_validate_manifests(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_csv = args.train_csv.expanduser().resolve()
    eval_csv = args.eval_csv.expanduser().resolve()
    if not train_csv.is_file():
        raise FileNotFoundError(train_csv)
    if not eval_csv.is_file():
        raise FileNotFoundError(eval_csv)

    train_full = pd.read_csv(train_csv, low_memory=False)
    eval_full = pd.read_csv(eval_csv, low_memory=False)
    required = {
        args.label_column,
        *args.path_columns,
        *args.time_columns,
    }
    if len(args.path_columns) != len(args.time_columns):
        raise ValueError("--path_columns and --time_columns must have equal length")
    if len(args.path_columns) != 6:
        raise ValueError(
            f"This runner expects six timepoints, got {len(args.path_columns)}"
        )

    for split_name, frame in (("train", train_full), ("eval", eval_full)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{split_name} manifest is missing columns {missing}")
        numeric_labels = pd.to_numeric(frame[args.label_column], errors="raise")
        if not numeric_labels.isin([0, 1]).all():
            raise ValueError(f"{split_name} labels must be binary 0/1")
        frame[args.label_column] = numeric_labels.astype(np.int64)
        for column in (*args.path_columns, *args.time_columns):
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{split_name} has missing values in {column}")

    event_audit: dict[str, Any] = {
        "event_column": args.event_column,
        "checked": False,
    }
    if args.event_column:
        if args.event_column not in train_full or args.event_column not in eval_full:
            raise ValueError(
                f"--event_column={args.event_column!r} is absent from a manifest"
            )
        train_events = set(
            train_full[args.event_column].fillna("").astype(str).str.strip()
        )
        eval_events = set(
            eval_full[args.event_column].fillna("").astype(str).str.strip()
        )
        if "" in train_events or "" in eval_events:
            raise ValueError("Empty event identifiers are not allowed")
        overlap = train_events & eval_events
        event_audit = {
            "event_column": args.event_column,
            "checked": True,
            "train_events": len(train_events),
            "eval_events": len(eval_events),
            "overlap_count": len(overlap),
            "overlap_examples": sorted(overlap)[:20],
        }
        if overlap and not args.allow_event_overlap:
            raise ValueError(
                f"Train/eval overlap in {len(overlap)} {args.event_column} values"
            )

    train_full = train_full.copy()
    eval_full = eval_full.copy()
    train_full["_source_index"] = np.arange(len(train_full), dtype=np.int64)
    eval_full["_source_index"] = np.arange(len(eval_full), dtype=np.int64)
    train = stratified_limit(
        train_full,
        args.max_train,
        label_column=args.label_column,
        seed=args.seed,
    ).reset_index(drop=True)
    evaluation = stratified_limit(
        eval_full,
        args.max_eval_samples,
        label_column=args.label_column,
        seed=args.seed + 1,
    ).reset_index(drop=True)

    for split_name, frame in (("train", train), ("eval", evaluation)):
        if frame[args.label_column].nunique() != 2:
            raise ValueError(
                f"{split_name} selection must contain both classes; "
                f"counts={label_counts(frame, args.label_column)}"
            )

    audit = {
        "train_csv": str(train_csv),
        "eval_csv": str(eval_csv),
        "full_rows": {"train": len(train_full), "eval": len(eval_full)},
        "selected_rows": {"train": len(train), "eval": len(evaluation)},
        "selected_labels": {
            "train": label_counts(train, args.label_column),
            "eval": label_counts(evaluation, args.label_column),
        },
        "events": event_audit,
    }
    return train, evaluation, audit


def load_s2_tiff(
    path: str,
    *,
    expected_channels: int,
    expected_size: int,
) -> torch.Tensor:
    """Load tifffile-written CHW or HWC arrays and return float32 CHW."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    array = np.asarray(tifffile.imread(source))
    array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"{source}: expected a 3-D TIFF, got shape {array.shape}")

    if array.shape[0] == expected_channels:
        chw = array
    elif array.shape[-1] == expected_channels:
        chw = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(
            f"{source}: expected {expected_channels} channels in CHW or HWC, "
            f"got shape {array.shape}"
        )
    if chw.shape[1:] != (expected_size, expected_size):
        raise ValueError(
            f"{source}: expected spatial size {expected_size}x{expected_size}, "
            f"got {tuple(chw.shape[1:])}"
        )
    chw = np.ascontiguousarray(chw, dtype=np.float32)
    return torch.from_numpy(chw)


def parse_mixed_utc_datetimes(values: pd.Series, *, column: str) -> pd.Series:
    """Parse ISO timestamps with or without fractional seconds across pandas versions."""
    try:
        # pandas >=2 supports mixed per-element ISO-8601 formats explicitly.
        parsed = pd.to_datetime(
            values,
            utc=True,
            errors="raise",
            format="mixed",
        )
    except (TypeError, ValueError):
        # pandas 1.x does not understand format="mixed".  Parsing each scalar
        # avoids the single-format inference that also rejects a column mixing
        # timestamps with and without fractional seconds.
        parsed = values.map(
            lambda value: pd.to_datetime(value, utc=True, errors="raise")
        )
    if parsed.isna().any():
        raise ValueError(f"NaT values in {column}")
    return parsed


class S2TemporalDataset(Dataset):
    """CSV-backed current or six-timepoint S2 dataset."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        path_columns: Sequence[str],
        time_columns: Sequence[str],
        input_mode: str,
        label_column: str,
        id_column: str,
        expected_channels: int,
        expected_size: int,
        max_relative_days: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.path_columns = tuple(path_columns)
        self.time_columns = tuple(time_columns)
        self.input_mode = input_mode
        self.label_column = label_column
        self.expected_channels = expected_channels
        self.expected_size = expected_size

        if input_mode == "current":
            if "path_t0" in self.path_columns:
                active = (self.path_columns.index("path_t0"),)
            else:
                active = (0,)
        elif input_mode == "raw":
            active = tuple(range(len(self.path_columns)))
        else:
            raise ValueError(f"Unsupported input_mode={input_mode!r}")
        self.active_indices = active
        self.active_path_columns = tuple(self.path_columns[index] for index in active)
        self.active_time_columns = tuple(self.time_columns[index] for index in active)
        self.paths = self.frame.loc[:, list(self.active_path_columns)].astype(str).to_numpy()

        parsed_dates = []
        for column in self.active_time_columns:
            values = parse_mixed_utc_datetimes(
                self.frame[column],
                column=column,
            )
            parsed_dates.append(values.array.asi8)
        date_ns = np.stack(parsed_dates, axis=1)
        relative_float = (
            date_ns - date_ns.min(axis=1, keepdims=True)
        ) / float(86_400 * 10**9)
        relative_days = np.rint(relative_float).astype(np.int64)
        self.date_clip_count = int((relative_days > max_relative_days).sum())
        self.relative_days = np.clip(
            relative_days, 0, max_relative_days
        ).astype(np.int64)

        self.labels = self.frame[label_column].to_numpy(dtype=np.int64)
        self.source_indices = self.frame["_source_index"].to_numpy(dtype=np.int64)
        if id_column in self.frame:
            self.sample_ids = self.frame[id_column].astype(str).tolist()
        else:
            self.sample_ids = [str(index) for index in self.source_indices]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, str]:
        sample_id = self.sample_ids[index]
        images = []
        try:
            for path in self.paths[index]:
                images.append(
                    load_s2_tiff(
                        str(path),
                        expected_channels=self.expected_channels,
                        expected_size=self.expected_size,
                    )
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed loading sample_id={sample_id!r}, row={index}"
            ) from exc
        return (
            torch.stack(images, dim=0),
            torch.from_numpy(self.relative_days[index].copy()),
            int(self.labels[index]),
            int(self.source_indices[index]),
            sample_id,
        )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def compute_band_stats(
    dataset: S2TemporalDataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    progress_every: int,
) -> dict[str, Any]:
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=False,
    )
    channels = dataset.expected_channels
    count = torch.zeros(channels, dtype=torch.float64)
    total = torch.zeros(channels, dtype=torch.float64)
    total_sq = torch.zeros(channels, dtype=torch.float64)
    started = time.time()

    for step, (images, _dates, _labels, _indices, _ids) in enumerate(loader, 1):
        values = images.to(torch.float64)
        finite = torch.isfinite(values)
        safe = torch.where(finite, values, torch.zeros_like(values))
        reduce_dims = (0, 1, 3, 4)
        count += finite.sum(dim=reduce_dims)
        total += safe.sum(dim=reduce_dims)
        total_sq += (safe * safe).sum(dim=reduce_dims)
        if progress_every and (step % progress_every == 0 or step == len(loader)):
            print(
                f"[stats] batches={step}/{len(loader)} "
                f"rows={min(step * batch_size, len(dataset))}/{len(dataset)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    if (count == 0).any():
        bad = torch.nonzero(count == 0).flatten().tolist()
        raise ValueError(f"No finite training pixels for bands {bad}")
    mean = total / count
    variance = (total_sq / count - mean.square()).clamp_min(0.0)
    std = variance.sqrt()
    near_constant = std <= 1e-6
    near_constant_bands = torch.nonzero(near_constant).flatten().tolist()
    if near_constant_bands:
        # Tiny smoke subsets can legitimately contain a band that is constant
        # across all selected crops.  Unit scale maps that band to zero after
        # mean subtraction without introducing infinities or aborting the run.
        std = torch.where(near_constant, torch.ones_like(std), std)
        print(
            "[stats] warning: using unit scale for near-constant bands "
            f"{near_constant_bands}",
            flush=True,
        )
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "finite_count": count.to(torch.int64).tolist(),
        "near_constant_bands": near_constant_bands,
    }


def load_or_compute_stats(
    args: argparse.Namespace,
    train_dataset: S2TemporalDataset,
    train_fingerprint: str,
) -> tuple[dict[str, Any], Path, str]:
    stats_config = {
        "script_version": SCRIPT_VERSION,
        "train_fingerprint": train_fingerprint,
        "input_mode": args.input_mode,
        "active_path_columns": list(train_dataset.active_path_columns),
        "expected_channels": args.expected_channels,
        "expected_size": args.expected_size,
        "selected_rows": len(train_dataset),
    }
    stats_key = config_hash(stats_config)
    stats_path = (
        args.feature_cache_dir.expanduser().resolve()
        / "band_stats"
        / f"{stats_key}.json"
    )
    if stats_path.is_file() and not args.recompute_stats:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        if payload.get("stats_key") != stats_key:
            raise ValueError(f"Stats fingerprint mismatch in {stats_path}")
        print(f"[stats] reusing {stats_path}", flush=True)
        return payload["stats"], stats_path, stats_key

    print(
        f"[stats] computing train-only {args.expected_channels}-band statistics "
        f"from {len(train_dataset)} rows",
        flush=True,
    )
    stats = compute_band_stats(
        train_dataset,
        batch_size=args.io_batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        progress_every=args.progress_every,
    )
    payload = {
        "stats_key": stats_key,
        "config": stats_config,
        "stats": stats,
        "created_unix": time.time(),
    }
    atomic_write_json(stats_path, payload)
    print(f"[stats] wrote {stats_path}", flush=True)
    return stats, stats_path, stats_key


def import_universat_hub(repo: Path):
    hubconf = repo.expanduser().resolve() / "hubconf.py"
    if not hubconf.is_file():
        raise FileNotFoundError(f"UniverSat hubconf.py not found: {hubconf}")
    if str(repo.resolve()) not in sys.path:
        sys.path.insert(0, str(repo.resolve()))
    spec = importlib.util.spec_from_file_location(
        "methanefuse_universat_hubconf", hubconf
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {hubconf}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_universat(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    version_parts = torch.__version__.split("+", 1)[0].split(".")
    version_tuple = tuple(int(part) for part in version_parts[:2])
    if version_tuple < (2, 2) or not hasattr(torch, "compiler"):
        raise RuntimeError(
            f"UniverSat requires torch>=2.2; found torch={torch.__version__}. "
            "Use the methane or train environment, not the old panopticon env."
        )

    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    args.hf_cache_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    hub = import_universat_hub(args.universat_repo)

    model_source_path = args.model_source.expanduser()
    if model_source_path.exists():
        source: str | Path = model_source_path.resolve()
        source_identity: dict[str, Any] = local_artifact_identity(source)
    else:
        source = str(args.model_source)
        source_identity = {"repo_id": source}
    load_kwargs: dict[str, Any] = {
        "cache_dir": args.hf_cache_dir.expanduser().resolve(),
        "local_files_only": not args.allow_download,
    }
    if args.model_revision:
        load_kwargs["revision"] = args.model_revision
    print(
        f"[model] loading source={source} local_only={not args.allow_download}",
        flush=True,
    )
    model = hub.UniverSat.from_pretrained(source, **load_kwargs)
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    identity = {
        "source": source_identity,
        "repo": local_artifact_identity(args.universat_repo / "hubconf.py"),
        "revision": args.model_revision,
        "parameters": int(parameter_count),
    }
    print(
        f"[model] loaded {parameter_count:,} frozen parameters on {device}",
        flush=True,
    )
    return model, identity


def autocast_context(
    device: torch.device, amp_dtype: str
) -> contextlib.AbstractContextManager:
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[amp_dtype]
    return torch.autocast(device_type="cuda", dtype=dtype)


def cache_tensor_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def write_feature_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_unix"] = time.time()
    atomic_write_json(path, manifest)


def validate_existing_shards(
    manifest: dict[str, Any], split_dir: Path
) -> None:
    row_sum = 0
    for shard in manifest.get("shards", []):
        path = split_dir / shard["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached feature shard: {path}")
        if int(path.stat().st_size) != int(shard["size_bytes"]):
            raise ValueError(f"Cached feature shard size changed: {path}")
        row_sum += int(shard["rows"])
    if row_sum != int(manifest.get("processed_rows", 0)):
        raise ValueError("Feature manifest processed_rows does not match shard rows")


def extract_feature_cache(
    *,
    split_name: str,
    dataset: S2TemporalDataset,
    dataset_fingerprint: str,
    model: nn.Module,
    model_identity: dict[str, Any],
    stats: dict[str, Any],
    stats_key: str,
    run_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    split_dir = run_root / split_name
    manifest_path = split_dir / "manifest.json"
    extraction_config = {
        "script_version": SCRIPT_VERSION,
        "split": split_name,
        "dataset_fingerprint": dataset_fingerprint,
        "rows": len(dataset),
        "input_mode": args.input_mode,
        "active_path_columns": list(dataset.active_path_columns),
        "active_time_columns": list(dataset.active_time_columns),
        "date_clip_count": dataset.date_clip_count,
        "max_relative_days": args.max_relative_days,
        "stats_key": stats_key,
        "model": model_identity,
        "modality_name": args.modality_name,
        "wavelengths_um": list(S2_WAVELENGTHS_UM),
        "input_resolution_m": args.input_resolution_m,
        "patch_size_m": args.patch_size_m,
        "output_grid": args.output_grid,
        "normalized_clip": args.normalized_clip,
        "cache_dtype": args.cache_dtype,
    }
    extraction_key = config_hash(extraction_config)

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("extraction_key") != extraction_key:
            raise ValueError(
                f"Feature cache configuration mismatch in {manifest_path}; "
                "use a different --feature_cache_dir or --cache_tag"
            )
        validate_existing_shards(manifest, split_dir)
        if manifest.get("complete"):
            if int(manifest["processed_rows"]) != len(dataset):
                raise ValueError(f"Completed cache row mismatch in {manifest_path}")
            print(
                f"[features:{split_name}] reusing {manifest['processed_rows']} rows "
                f"from {split_dir}",
                flush=True,
            )
            return manifest_path
    else:
        manifest = {
            "extraction_key": extraction_key,
            "config": extraction_config,
            "complete": False,
            "processed_rows": 0,
            "feature_dim": None,
            "label_counts": {},
            "shards": [],
            "created_unix": time.time(),
        }
        write_feature_manifest(manifest_path, manifest)

    processed = int(manifest["processed_rows"])
    if processed > len(dataset):
        raise ValueError(f"Cache has more rows than dataset: {manifest_path}")
    remaining = Subset(dataset, range(processed, len(dataset)))
    loader = make_loader(
        remaining,
        batch_size=args.extract_batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=device.type == "cuda",
    )
    mean = torch.tensor(
        stats["mean"], dtype=torch.float32, device=device
    ).view(1, 1, -1, 1, 1)
    std = torch.tensor(
        stats["std"], dtype=torch.float32, device=device
    ).view(1, 1, -1, 1, 1)
    shard_dtype = cache_tensor_dtype(args.cache_dtype)
    next_shard = len(manifest["shards"])
    buffered_features: list[torch.Tensor] = []
    buffered_labels: list[torch.Tensor] = []
    buffered_indices: list[torch.Tensor] = []
    buffered_ids: list[str] = []
    buffered_rows = 0
    counts = Counter(
        {int(key): int(value) for key, value in manifest["label_counts"].items()}
    )
    started = time.time()

    def flush_shard() -> None:
        nonlocal next_shard, processed, buffered_rows
        if buffered_rows == 0:
            return
        features = torch.cat(buffered_features, dim=0).to(shard_dtype)
        labels = torch.cat(buffered_labels, dim=0).to(torch.int64)
        source_indices = torch.cat(buffered_indices, dim=0).to(torch.int64)
        shard_name = f"shard_{next_shard:06d}.pt"
        shard_path = split_dir / shard_name
        atomic_torch_save(
            shard_path,
            {
                "features": features,
                "labels": labels,
                "source_indices": source_indices,
                "sample_ids": list(buffered_ids),
            },
        )
        for label, count in zip(*torch.unique(labels, return_counts=True)):
            counts[int(label.item())] += int(count.item())
        rows = int(labels.numel())
        processed += rows
        manifest["processed_rows"] = processed
        manifest["feature_dim"] = int(features.shape[1])
        manifest["label_counts"] = {
            str(key): int(value) for key, value in sorted(counts.items())
        }
        manifest["shards"].append(
            {
                "file": shard_name,
                "rows": rows,
                "size_bytes": int(shard_path.stat().st_size),
            }
        )
        write_feature_manifest(manifest_path, manifest)
        print(
            f"[features:{split_name}] cached={processed}/{len(dataset)} "
            f"shards={len(manifest['shards'])} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        next_shard += 1
        buffered_features.clear()
        buffered_labels.clear()
        buffered_indices.clear()
        buffered_ids.clear()
        buffered_rows = 0

    for step, (images, dates, labels, source_indices, sample_ids) in enumerate(
        loader, 1
    ):
        images = images.to(device, non_blocking=True, dtype=torch.float32)
        dates = dates.to(device, non_blocking=True, dtype=torch.int64)
        images = (images - mean) / std
        images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
        if args.normalized_clip > 0:
            images = images.clamp(
                min=-args.normalized_clip, max=args.normalized_clip
            )

        with torch.inference_mode(), autocast_context(device, args.amp_dtype):
            tokens, _extras = model.encode(
                {
                    args.modality_name: images,
                    f"{args.modality_name}_dates": dates,
                },
                wavelengths={
                    args.modality_name: list(S2_WAVELENGTHS_UM)
                },
                input_res={args.modality_name: args.input_resolution_m},
                subpatches={args.modality_name: 1},
                patch_size=args.patch_size_m,
                output_grid=args.output_grid,
            )
            if tokens.ndim != 3:
                raise ValueError(
                    f"Expected UniverSat tokens (B,N,D), got {tuple(tokens.shape)}"
                )
            pooled = torch.cat(
                [tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1
            ).to(torch.float32)

        buffered_features.append(pooled.cpu())
        buffered_labels.append(labels.to(torch.int64).cpu())
        buffered_indices.append(source_indices.to(torch.int64).cpu())
        buffered_ids.extend(str(value) for value in sample_ids)
        buffered_rows += int(labels.numel())
        if buffered_rows >= args.cache_shard_size:
            flush_shard()
        elif args.progress_every and step % args.progress_every == 0:
            print(
                f"[features:{split_name}] extracting batch={step}/{len(loader)} "
                f"committed={processed} buffered={buffered_rows}",
                flush=True,
            )

    flush_shard()
    if processed != len(dataset):
        raise RuntimeError(
            f"Feature extraction ended at {processed}/{len(dataset)} rows"
        )
    manifest["complete"] = True
    write_feature_manifest(manifest_path, manifest)
    return manifest_path


def load_feature_shard(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def iter_feature_batches(
    manifest_path: Path,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"Incomplete feature cache: {manifest_path}")
    split_dir = manifest_path.parent
    shard_indices = list(range(len(manifest["shards"])))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(shard_indices)
    for shard_index in shard_indices:
        shard_info = manifest["shards"][shard_index]
        payload = load_feature_shard(split_dir / shard_info["file"])
        features = payload["features"].to(torch.float32)
        labels = payload["labels"].to(torch.int64)
        order = np.arange(len(labels))
        if shuffle:
            rng.shuffle(order)
        order_tensor = torch.from_numpy(order)
        for start in range(0, len(order), batch_size):
            selected = order_tensor[start : start + batch_size]
            yield features[selected], labels[selected]


class ProbeHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.layers = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.layers = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 2),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def best_f1_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        predictions = probabilities >= 0.5
        return 0.5, float(f1_score(labels, predictions, zero_division=0))
    numer = 2.0 * precision[:-1] * recall[:-1]
    denom = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        numer,
        denom,
        out=np.zeros_like(numer),
        where=denom > 0,
    )
    best_value = float(np.max(f1_values))
    candidate_indices = np.flatnonzero(np.isclose(f1_values, best_value))
    best_index = min(
        candidate_indices.tolist(),
        key=lambda index: (abs(float(thresholds[index]) - 0.5), index),
    )
    return float(thresholds[best_index]), best_value


def evaluate_head(
    head: nn.Module,
    manifest_path: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    head.eval()
    probabilities = []
    targets = []
    with torch.inference_mode():
        for features, labels in iter_feature_batches(
            manifest_path,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            logits = head(features.to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1)[:, 1]
            probabilities.append(probs.cpu().numpy())
            targets.append(labels.numpy())
    y_true = np.concatenate(targets).astype(np.int64, copy=False)
    y_prob = np.concatenate(probabilities).astype(np.float64, copy=False)
    if np.unique(y_true).size != 2:
        raise ValueError("Evaluation cache must contain both classes")

    threshold, best_f1 = best_f1_threshold(y_true, y_prob)
    predictions = (y_prob >= threshold).astype(np.int64)
    predictions_05 = (y_prob >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "rows": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "negative_rows": int((y_true == 0).sum()),
        "ap": float(average_precision_score(y_true, y_prob)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "best_threshold": threshold,
        "f1": best_f1,
        "f1_at_0_5": float(
            f1_score(y_true, predictions_05, zero_division=0)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def train_probe(
    args: argparse.Namespace,
    *,
    train_manifest_path: Path,
    eval_manifest_path: Path,
    device: torch.device,
    output_json: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    train_manifest = json.loads(
        train_manifest_path.read_text(encoding="utf-8")
    )
    feature_dim = int(train_manifest["feature_dim"])
    eval_manifest = json.loads(
        eval_manifest_path.read_text(encoding="utf-8")
    )
    if int(eval_manifest["feature_dim"]) != feature_dim:
        raise ValueError("Train/eval feature dimensions differ")

    head = ProbeHead(
        feature_dim,
        hidden_dim=args.head_hidden_dim,
        dropout=args.head_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay
    )
    counts = {
        int(key): int(value)
        for key, value in train_manifest["label_counts"].items()
    }
    if set(counts) != {0, 1}:
        raise ValueError(f"Training cache must contain both labels: {counts}")
    if args.class_weight == "balanced":
        total = counts[0] + counts[1]
        class_weights = torch.tensor(
            [total / (2 * counts[0]), total / (2 * counts[1])],
            dtype=torch.float32,
            device=device,
        )
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        total_rows = 0
        for features, labels in iter_feature_batches(
            train_manifest_path,
            batch_size=args.head_batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = head(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows = int(labels.numel())
            total_loss += float(loss.item()) * rows
            total_rows += rows

        metrics = evaluate_head(
            head,
            eval_manifest_path,
            batch_size=args.head_batch_size,
            device=device,
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_rows, 1),
            "eval": metrics,
        }
        history.append(epoch_result)
        print(
            f"[head] epoch={epoch}/{args.epochs} "
            f"loss={epoch_result['train_loss']:.6f} "
            f"AP={metrics['ap']:.6f} AUROC={metrics['auroc']:.6f} "
            f"F1={metrics['f1']:.6f} threshold={metrics['best_threshold']:.6f}",
            flush=True,
        )
        if best is None or (
            metrics["ap"], metrics["f1"]
        ) > (
            best["eval"]["ap"], best["eval"]["f1"]
        ):
            best = epoch_result
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }

        in_progress = {
            "status": "running",
            **provenance,
            "history": history,
            "best": best,
        }
        atomic_write_json(output_json, in_progress)

    if best is None or best_state is None:
        raise RuntimeError("No head epoch completed")
    head_path = output_json.with_name(output_json.stem + "_head_best.pt")
    atomic_torch_save(
        head_path,
        {
            "head": best_state,
            "feature_dim": feature_dim,
            "best_epoch": int(best["epoch"]),
            "args": json_safe(vars(args)),
        },
    )
    result = {
        "status": "complete",
        **provenance,
        "history": history,
        "best": best,
        "head_checkpoint": str(head_path),
    }
    atomic_write_json(output_json, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--eval_csv", type=Path, required=True)
    parser.add_argument(
        "--input_mode", choices=("current", "raw"), default="raw"
    )
    parser.add_argument(
        "--path_columns",
        type=parse_columns,
        default=DEFAULT_PATH_COLUMNS,
        help="Comma-separated six TIFF path columns.",
    )
    parser.add_argument(
        "--time_columns",
        type=parse_columns,
        default=DEFAULT_TIME_COLUMNS,
        help="Comma-separated six acquisition-time columns.",
    )
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--id_column", default="sample_id")
    parser.add_argument("--event_column", default="event_group_id")
    parser.add_argument("--allow_event_overlap", action="store_true")
    parser.add_argument(
        "--max_train",
        "--max-train",
        type=int,
        help="Optional deterministic stratified training-row cap.",
    )
    parser.add_argument(
        "--max_eval_samples",
        "--max-eval-samples",
        type=int,
        help="Optional deterministic stratified evaluation-row cap.",
    )
    parser.add_argument("--seed", type=int, default=20260727)

    parser.add_argument(
        "--universat_repo", type=Path, default=DEFAULT_UNIVERSAT_REPO
    )
    parser.add_argument(
        "--model_source",
        type=Path,
        default=DEFAULT_MODEL_SOURCE,
        help="Local checkpoint directory, or HF repo ID if downloading is allowed.",
    )
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--hf_cache_dir", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow Hugging Face network access. Default is local-files-only.",
    )

    parser.add_argument("--expected_channels", type=int, default=12)
    parser.add_argument("--expected_size", type=int, default=32)
    parser.add_argument("--max_relative_days", type=int, default=365)
    parser.add_argument("--modality_name", default="methane_s2")
    parser.add_argument("--input_resolution_m", type=float, default=10.0)
    parser.add_argument("--patch_size_m", type=float, default=40.0)
    parser.add_argument("--output_grid", type=int, default=8)
    parser.add_argument(
        "--normalized_clip",
        type=float,
        default=0.0,
        help="Clip normalized inputs symmetrically; <=0 disables clipping.",
    )

    parser.add_argument(
        "--feature_cache_dir", type=Path, default=DEFAULT_FEATURE_CACHE
    )
    parser.add_argument(
        "--cache_tag",
        default="",
        help="Optional cache namespace suffix for manual checkpoint disambiguation.",
    )
    parser.add_argument("--cache_shard_size", type=int, default=4096)
    parser.add_argument(
        "--cache_dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--recompute_stats", action="store_true")
    parser.add_argument("--io_batch_size", type=int, default=16)
    parser.add_argument("--extract_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument(
        "--amp_dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--head_batch_size", type=int, default=512)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--head_hidden_dim", type=int, default=0)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument(
        "--class_weight", choices=("none", "balanced"), default="balanced"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output_json",
        type=Path,
        help="Defaults to results.json inside the fingerprinted cache directory.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "--expected_channels": args.expected_channels,
        "--expected_size": args.expected_size,
        "--max_relative_days": args.max_relative_days,
        "--output_grid": args.output_grid,
        "--cache_shard_size": args.cache_shard_size,
        "--io_batch_size": args.io_batch_size,
        "--extract_batch_size": args.extract_batch_size,
        "--head_batch_size": args.head_batch_size,
        "--epochs": args.epochs,
        "--prefetch_factor": args.prefetch_factor,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative")
    if len(S2_WAVELENGTHS_UM) != args.expected_channels:
        raise ValueError(
            "Official wavelength list and --expected_channels disagree"
        )
    patch_pixels = args.patch_size_m / args.input_resolution_m
    if not math.isclose(patch_pixels, round(patch_pixels)):
        raise ValueError("patch_size_m / input_resolution_m must be integral")
    if args.expected_size % int(round(patch_pixels)) != 0:
        raise ValueError(
            "expected_size must be divisible by the physical patch size in pixels"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.path_columns = parse_columns(args.path_columns)
    args.time_columns = parse_columns(args.time_columns)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    train_frame, eval_frame, manifest_audit = read_and_validate_manifests(args)
    fingerprint_columns = (
        args.label_column,
        "_source_index",
        *args.path_columns,
        *args.time_columns,
    )
    train_fingerprint = manifest_fingerprint(
        args.train_csv.expanduser().resolve(),
        train_frame,
        fingerprint_columns,
    )
    eval_fingerprint = manifest_fingerprint(
        args.eval_csv.expanduser().resolve(),
        eval_frame,
        fingerprint_columns,
    )
    train_dataset = S2TemporalDataset(
        train_frame,
        path_columns=args.path_columns,
        time_columns=args.time_columns,
        input_mode=args.input_mode,
        label_column=args.label_column,
        id_column=args.id_column,
        expected_channels=args.expected_channels,
        expected_size=args.expected_size,
        max_relative_days=args.max_relative_days,
    )
    eval_dataset = S2TemporalDataset(
        eval_frame,
        path_columns=args.path_columns,
        time_columns=args.time_columns,
        input_mode=args.input_mode,
        label_column=args.label_column,
        id_column=args.id_column,
        expected_channels=args.expected_channels,
        expected_size=args.expected_size,
        max_relative_days=args.max_relative_days,
    )

    stats, stats_path, stats_key = load_or_compute_stats(
        args, train_dataset, train_fingerprint
    )
    model_source_path = args.model_source.expanduser()
    model_source_identity = (
        local_artifact_identity(model_source_path)
        if model_source_path.exists()
        else {"repo_id": str(args.model_source)}
    )
    run_config = {
        "script_version": SCRIPT_VERSION,
        "input_mode": args.input_mode,
        "train_fingerprint": train_fingerprint,
        "eval_fingerprint": eval_fingerprint,
        "stats_key": stats_key,
        "model_source": model_source_identity,
        "model_revision": args.model_revision,
        "repo_hubconf": local_artifact_identity(
            args.universat_repo.expanduser().resolve() / "hubconf.py"
        ),
        "wavelengths_um": list(S2_WAVELENGTHS_UM),
        "input_resolution_m": args.input_resolution_m,
        "patch_size_m": args.patch_size_m,
        "output_grid": args.output_grid,
        "cache_dtype": args.cache_dtype,
        "normalized_clip": args.normalized_clip,
        "cache_tag": args.cache_tag,
    }
    run_key = config_hash(run_config)
    run_root = (
        args.feature_cache_dir.expanduser().resolve()
        / f"{args.input_mode}_{run_key[:16]}"
    )
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else run_root / "results.json"
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested ({device}) but CUDA is unavailable"
            )
        torch.cuda.set_device(device)
    model, model_identity = load_universat(args, device)

    train_feature_manifest = extract_feature_cache(
        split_name="train",
        dataset=train_dataset,
        dataset_fingerprint=train_fingerprint,
        model=model,
        model_identity=model_identity,
        stats=stats,
        stats_key=stats_key,
        run_root=run_root,
        args=args,
        device=device,
    )
    eval_feature_manifest = extract_feature_cache(
        split_name="eval",
        dataset=eval_dataset,
        dataset_fingerprint=eval_fingerprint,
        model=model,
        model_identity=model_identity,
        stats=stats,
        stats_key=stats_key,
        run_root=run_root,
        args=args,
        device=device,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    provenance = {
        "script_version": SCRIPT_VERSION,
        "run_key": run_key,
        "run_root": str(run_root),
        "args": json_safe(vars(args)),
        "manifest_audit": manifest_audit,
        "relative_days": {
            "policy": "round(days_since_earliest), then clamp",
            "range": [0, args.max_relative_days],
            "train_clipped_values": train_dataset.date_clip_count,
            "eval_clipped_values": eval_dataset.date_clip_count,
        },
        "band_stats": {
            "source": "selected training rows only",
            "path": str(stats_path),
            "mean": stats["mean"],
            "std": stats["std"],
            "finite_count": stats["finite_count"],
            "near_constant_bands": stats.get("near_constant_bands", []),
        },
        "model": model_identity,
        "feature_cache": {
            "train_manifest": str(train_feature_manifest),
            "eval_manifest": str(eval_feature_manifest),
        },
    }
    result = train_probe(
        args,
        train_manifest_path=train_feature_manifest,
        eval_manifest_path=eval_feature_manifest,
        device=device,
        output_json=output_json,
        provenance=provenance,
    )
    metrics = result["best"]["eval"]
    print(
        f"[done] output={output_json} best_epoch={result['best']['epoch']} "
        f"AP={metrics['ap']:.6f} AUROC={metrics['auroc']:.6f} "
        f"F1={metrics['f1']:.6f} threshold={metrics['best_threshold']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
