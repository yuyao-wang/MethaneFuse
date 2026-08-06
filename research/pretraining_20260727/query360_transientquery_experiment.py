#!/usr/bin/env python3
"""Leakage-safe 360 m multi-sensor TransientQuery screening experiment.

The runner has four explicit stages:

``split``
    Derive a geographically disjoint inner validation set from one train-only
    macro-region manifest.  The data module refuses test/sealed/holdout paths.

``warm-cache``
    Copy only the selected classification inputs to a size-verified hashed
    cache.  Plume masks are never copied.

``cache-features``
    Encode every valid sensor frame independently with either the official
    frozen Panopticon checkpoint or a reproducibly random frozen Panopticon.
    Frames are grouped by native sensor channel count; no cross-sensor padding
    can enter Panopticon's spectral attention.

``train-heads``
    Train parameter- and initialization-matched current-only,
    TransientQuery, and history-shuffle heads on cached features.  Model
    selection uses inner-validation AP and never opens a test manifest.

The random frozen encoder is a fast representation control, not an end-to-end
scratch training claim.  That distinction is embedded in every artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from query360_data import (  # noqa: E402
    DEFAULT_WV3_SRF,
    ROLE_NAMES,
    SENSOR_ORDER,
    Query360Dataset,
    StrictHashedFileCache,
    assert_safe_path,
    derive_inner_split,
    query360_collate,
    sha256_file,
    warm_classification_cache,
)
from query360_model import (  # noqa: E402
    ARM_NAMES,
    TransientQuery360Head,
    build_matched_models,
    evaluate_head,
    model_parameter_signature,
    state_dict_sha256,
    train_head_epoch,
)
from src.utils.training import _load_backbone  # noqa: E402


SCRIPT_VERSION = "query360-transientquery-v1"
FEATURE_CACHE_VERSION = 1
DEFAULT_SOURCE_TRAIN = (
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/"
    "datasets_360m_cluster_split/"
    "manifest_time_train_360m_macroregion_by_plume.csv"
)
DEFAULT_OUTPUT_ROOT = (
    "/diniuvol/yuyao/methanefuse_research_20260727/query360_tq_v1"
)
DEFAULT_WEIGHTS = str(REPO_ROOT / "weights" / "panopticon_vitb14_teacher.pth")
EVENT_SUFFIX_RE = re.compile(r"-[A-Za-z0-9]+$")


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


def atomic_json_write(path: Path, value: Any) -> None:
    path = assert_safe_path(path, purpose="JSON output")
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
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def atomic_csv_write(path: Path, frame: pd.DataFrame) -> None:
    path = assert_safe_path(path, purpose="CSV output")
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
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def atomic_torch_save(path: Path, value: Any) -> None:
    path = assert_safe_path(path, purpose="torch output")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def torch_load_trusted(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_csv_list(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not parsed:
        raise ValueError("Expected at least one comma-separated value.")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Values must be unique, got {parsed}.")
    return parsed


def parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part) for part in parse_csv_list(value))
    if min(parsed) < 0:
        raise ValueError("Seeds must be non-negative.")
    return parsed


def script_provenance() -> dict[str, str]:
    files = (
        Path(__file__).resolve(),
        SCRIPT_DIR / "query360_data.py",
        SCRIPT_DIR / "query360_model.py",
    )
    return {str(path): sha256_file(path) for path in files}


def configure_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_split(args: argparse.Namespace) -> None:
    output_root = assert_safe_path(args.output_root, purpose="experiment root")
    manifest_dir = output_root / "manifests"
    artifacts = derive_inner_split(
        args.source_train_csv,
        manifest_dir / "inner_train.csv",
        manifest_dir / "inner_val.csv",
        manifest_dir / "split_audit.json",
        seed=args.seed,
        target_val_fraction=args.val_fraction,
        min_val_regions=args.min_val_regions,
        search_trials=args.search_trials,
        overwrite=args.overwrite,
    )
    print(json.dumps(artifacts.__dict__, indent=2, sort_keys=True), flush=True)


def run_warm_cache(args: argparse.Namespace) -> None:
    cache_dir = assert_safe_path(args.cache_dir, purpose="raw file cache")
    reports = []
    for manifest_value in parse_csv_list(args.manifests):
        manifest = assert_safe_path(manifest_value, purpose="cache manifest")
        started = time.monotonic()
        report = warm_classification_cache(
            manifest, cache_dir, max_workers=args.workers
        )
        record = {
            **report.__dict__,
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "elapsed_seconds": float(time.monotonic() - started),
        }
        reports.append(record)
        print(
            f"[warm-cache] manifest={manifest.name} "
            f"paths={report.requested_paths} copied={report.copied_paths} "
            f"reused={report.reused_paths} "
            f"GiB={report.total_source_bytes / 2**30:.3f} "
            f"elapsed={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
    audit = {
        "schema_version": "query360-raw-cache-v1",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "cache_dir": str(cache_dir),
        "reports": reports,
        "plume_masks_cached": False,
        "external_test_manifest_read": False,
        "script_provenance": script_provenance(),
    }
    atomic_json_write(cache_dir / "warmup_audit.json", audit)


def _select_smoke_positions(frame: pd.DataFrame, maximum_rows: int) -> list[int]:
    if maximum_rows <= 0 or maximum_rows >= len(frame):
        return list(range(len(frame)))
    labels = frame["label"].astype(int).to_numpy()
    target_zero = maximum_rows // 2
    target_one = maximum_rows - target_zero
    selected = list(np.flatnonzero(labels == 0)[:target_zero])
    selected.extend(np.flatnonzero(labels == 1)[:target_one])
    if len(selected) < maximum_rows:
        already = set(selected)
        selected.extend(
            index
            for index in range(len(frame))
            if index not in already
        )
    return sorted(selected[:maximum_rows])


def _build_encoder(
    condition: str,
    *,
    weights: str,
    encoder_seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    random.seed(encoder_seed)
    np.random.seed(encoder_seed)
    torch.manual_seed(encoder_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(encoder_seed)

    if condition == "panopticon_pretrained":
        weights_path = assert_safe_path(weights, purpose="encoder weights")
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)
        backbone = _load_backbone(str(weights_path), strict=True)
        metadata = {
            "condition": condition,
            "interpretation": "official frozen Panopticon encoder",
            "weights_path": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
            "encoder_seed": int(encoder_seed),
        }
    elif condition == "random_frozen":
        backbone = _load_backbone("random", strict=True)
        metadata = {
            "condition": condition,
            "interpretation": (
                "reproducibly random frozen Panopticon encoder; "
                "not end-to-end scratch training"
            ),
            "weights_path": "",
            "weights_sha256": None,
            "encoder_seed": int(encoder_seed),
        }
    else:
        raise ValueError(f"Unknown encoder condition: {condition}")
    metadata["loaded_state_sha256"] = state_dict_sha256(backbone.state_dict())
    metadata["embed_dim"] = int(backbone.embed_dim)
    return backbone, metadata


def _metadata_values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        raise ValueError(f"Manifest lacks metadata column {column!r}.")
    values = frame[column].fillna("").astype(str).str.strip().tolist()
    if any(not value for value in values):
        raise ValueError(f"Manifest metadata column {column!r} contains blanks.")
    return values


def canonical_event_ids(plume_ids: Sequence[str]) -> list[str]:
    """Collapse terminal plume variants to their acquisition/event identity."""

    event_ids = [EVENT_SUFFIX_RE.sub("", str(value).strip()) for value in plume_ids]
    if any(not value for value in event_ids):
        raise ValueError("Canonical event derivation produced an empty identifier.")
    return event_ids


def run_cache_features(args: argparse.Namespace) -> None:
    manifest_path = assert_safe_path(args.manifest, purpose="feature manifest")
    output_path = assert_safe_path(args.output_cache, purpose="feature cache")
    raw_cache_dir = assert_safe_path(args.raw_cache_dir, purpose="raw file cache")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Feature cache already exists: {output_path}; pass --overwrite."
        )
    if args.split not in {"train", "inner_val"}:
        raise ValueError("--split must be train or inner_val.")

    local_cache = StrictHashedFileCache(raw_cache_dir)
    dataset = Query360Dataset(
        manifest_path,
        local_cache=local_cache,
        wv3_srf_csv=args.wv3_srf_csv,
        pad_to_multiple=14,
    )
    positions = _select_smoke_positions(dataset.frame, args.max_rows)
    selected_frame = dataset.frame.iloc[positions].reset_index(drop=True).copy()
    selected_dataset = (
        dataset if len(positions) == len(dataset) else Subset(dataset, positions)
    )
    global_indices = selected_frame["query360_index"].astype(np.int64).tolist()
    global_to_local = {
        int(global_index): local_index
        for local_index, global_index in enumerate(global_indices)
    }
    if len(global_to_local) != len(selected_frame):
        raise RuntimeError("Selected query360_index values are not unique.")

    loader_kwargs: dict[str, Any] = {
        "dataset": selected_dataset,
        "batch_size": args.row_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "collate_fn": query360_collate,
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )
    loader = DataLoader(**loader_kwargs)

    device = configure_device(args.device)
    backbone, encoder_metadata = _build_encoder(
        args.encoder_condition,
        weights=args.weights,
        encoder_seed=args.encoder_seed,
    )
    backbone = backbone.to(device)
    backbone.requires_grad_(False)
    backbone.eval()

    rows = len(selected_frame)
    sensors = len(SENSOR_ORDER)
    roles = len(ROLE_NAMES)
    feature_dim = int(backbone.embed_dim)
    storage_dtype = (
        torch.float16 if args.storage_dtype == "float16" else torch.float32
    )
    features = torch.zeros(
        (rows, sensors, roles, feature_dim), dtype=storage_dtype
    )
    valid_mask = torch.zeros((rows, sensors, roles), dtype=torch.bool)
    finite_fraction = torch.zeros((rows, sensors, roles), dtype=torch.float32)
    seen_rows = torch.zeros(rows, dtype=torch.bool)
    emitted_observations = 0
    started = time.monotonic()

    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            batch_global = batch["index"].long().tolist()
            try:
                batch_local = torch.tensor(
                    [global_to_local[int(index)] for index in batch_global],
                    dtype=torch.long,
                )
            except KeyError as error:
                raise RuntimeError(
                    f"DataLoader emitted an undeclared query360_index: {error}"
                ) from error
            if seen_rows[batch_local].any():
                raise RuntimeError("A selected manifest row was emitted twice.")
            seen_rows[batch_local] = True
            valid_mask[batch_local] = batch["valid_mask"].bool()
            finite_fraction[batch_local] = batch["finite_fraction"].float()

            for sensor_index, sensor_name in enumerate(SENSOR_ORDER):
                sensor_batch = batch["sensor_batches"][sensor_name]
                images = sensor_batch["images"]
                if images.shape[0] == 0:
                    continue
                observation_local = torch.tensor(
                    [
                        global_to_local[int(index)]
                        for index in sensor_batch["rows"].long().tolist()
                    ],
                    dtype=torch.long,
                )
                observation_roles = sensor_batch["roles"].long()
                channel_ids = sensor_batch["channel_ids"].reshape(-1)
                if images.shape[1] != channel_ids.numel():
                    raise RuntimeError(
                        f"{sensor_name}: image/channel-id count mismatch."
                    )
                for start in range(0, len(images), args.encoder_microbatch):
                    stop = min(start + args.encoder_microbatch, len(images))
                    image_chunk = images[start:stop].to(
                        device, non_blocking=True
                    )
                    channel_chunk = (
                        channel_ids.view(1, -1)
                        .expand(stop - start, -1)
                        .clone()
                        .to(device, non_blocking=True)
                    )
                    with autocast_context(device, args.amp_dtype):
                        output = backbone.forward_features(
                            {"imgs": image_chunk, "chn_ids": channel_chunk}
                        )
                        encoded = output["x_norm_clstoken"]
                    features[
                        observation_local[start:stop],
                        sensor_index,
                        observation_roles[start:stop],
                    ] = encoded.float().cpu().to(storage_dtype)
                    emitted_observations += stop - start

            if (
                batch_number % max(1, args.log_interval) == 0
                or batch_number == len(loader)
            ):
                elapsed = time.monotonic() - started
                print(
                    f"[cache-features] condition={args.encoder_condition} "
                    f"split={args.split} batches={batch_number}/{len(loader)} "
                    f"rows={int(seen_rows.sum())}/{rows} "
                    f"observations={emitted_observations} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if not seen_rows.all():
        missing = torch.nonzero(~seen_rows, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Feature cache missed rows: {missing[:20]}.")
    feature_presence = features.float().abs().sum(dim=-1).ne(0)
    if (feature_presence & ~valid_mask).any():
        raise RuntimeError("A feature was written outside the declared valid mask.")
    if (valid_mask & ~feature_presence).any():
        missing = torch.nonzero(
            valid_mask & ~feature_presence, as_tuple=False
        ).tolist()
        raise RuntimeError(
            f"Valid observations produced all-zero features: {missing[:20]}."
        )
    usable = valid_mask[:, :, 0].any(dim=1)
    if not usable.any():
        raise RuntimeError("No row retains a valid current observation.")
    excluded_indices = torch.nonzero(~usable, as_tuple=False).flatten().tolist()
    if excluded_indices:
        print(
            f"[cache-features] excluding {len(excluded_indices)} rows with no "
            "valid current sensor.",
            flush=True,
        )

    keep = torch.nonzero(usable, as_tuple=False).flatten()
    keep_list = keep.tolist()
    selected_frame = selected_frame.iloc[keep_list].reset_index(drop=True)
    features = features[keep]
    valid_mask = valid_mask[keep]
    finite_fraction = finite_fraction[keep]
    if not torch.isfinite(features.float()).all():
        raise RuntimeError("Feature cache contains non-finite embeddings.")
    labels = torch.tensor(
        selected_frame["label"].astype(int).tolist(), dtype=torch.long
    )
    if set(labels.tolist()) != {0, 1}:
        raise RuntimeError("A formal feature cache must contain both classes.")

    plume_ids = _metadata_values(selected_frame, "plume_id")
    payload = {
        "format_version": FEATURE_CACHE_VERSION,
        "script_version": SCRIPT_VERSION,
        "split": args.split,
        "encoder": encoder_metadata,
        "features": features,
        "valid_mask": valid_mask,
        "finite_fraction": finite_fraction,
        "labels": labels,
        "ids": _metadata_values(selected_frame, "id"),
        "plume_ids": plume_ids,
        "event_ids": canonical_event_ids(plume_ids),
        "event_id_rule": "strip-terminal-hyphen-alphanumeric-suffix-v1",
        "cluster_ids": _metadata_values(selected_frame, "cluster_id"),
        "macro_region_ids": _metadata_values(
            selected_frame, "macro_region_id"
        ),
        "availability_signatures": _metadata_values(
            selected_frame, "availability_signature"
        ),
        "query360_indices": torch.tensor(
            selected_frame["query360_index"].astype(np.int64).tolist(),
            dtype=torch.long,
        ),
        "sensor_names": list(SENSOR_ORDER),
        "role_names": [
            "current",
            "short_history",
            "sensor_specific_long_history",
        ],
        "role_semantics": {
            "s2": ["t0", "approximately_t_minus_90d", "approximately_t_minus_360d"],
            "l89": ["t0", "approximately_t_minus_90d", "approximately_t_minus_360d"],
            "emit": ["t0", "approximately_t_minus_90d", "approximately_t_minus_180d"],
            "s5p": ["t0", "approximately_t_minus_90d", "approximately_t_minus_360d"],
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "source_rows": int(len(dataset.frame)),
            "selected_rows_before_validity": int(rows),
            "cached_rows": int(len(selected_frame)),
            "excluded_no_current_indices": excluded_indices,
        },
        "extraction": {
            "row_batch_size": int(args.row_batch_size),
            "encoder_microbatch": int(args.encoder_microbatch),
            "num_workers": int(args.num_workers),
            "amp_dtype": args.amp_dtype,
            "storage_dtype": args.storage_dtype,
            "emitted_valid_observations": int(emitted_observations),
            "elapsed_seconds": float(time.monotonic() - started),
            "raw_cache_dir": str(raw_cache_dir),
            "sensor_grouped_native_channels": True,
            "cross_sensor_channel_padding": False,
        },
        "safety": {
            "external_test_manifest_read": False,
            "test_sealed_holdout_paths_refused": True,
            "random_row_replacement": False,
            "plume_masks_read": False,
        },
        "script_provenance": script_provenance(),
    }
    atomic_torch_save(output_path, payload)
    audit = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "features",
            "valid_mask",
            "finite_fraction",
            "labels",
            "query360_indices",
        }
    }
    audit["tensor_shapes"] = {
        "features": list(features.shape),
        "valid_mask": list(valid_mask.shape),
        "finite_fraction": list(finite_fraction.shape),
        "labels": list(labels.shape),
    }
    audit["output_cache"] = str(output_path)
    audit["output_cache_sha256"] = sha256_file(output_path)
    atomic_json_write(output_path.with_suffix(output_path.suffix + ".audit.json"), audit)
    print(
        f"[cache-features] wrote {output_path} rows={len(labels)} "
        f"observations={int(valid_mask.sum())} "
        f"sha256={audit['output_cache_sha256']}",
        flush=True,
    )


def _validate_feature_cache(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expected_split: str,
) -> None:
    assert_safe_path(path, purpose="feature cache")
    if int(payload.get("format_version", -1)) != FEATURE_CACHE_VERSION:
        raise ValueError(f"Unsupported feature cache format: {path}")
    if payload.get("split") != expected_split:
        raise ValueError(
            f"{path}: split={payload.get('split')!r}, expected {expected_split!r}."
        )
    features = payload.get("features")
    valid = payload.get("valid_mask")
    labels = payload.get("labels")
    if not isinstance(features, torch.Tensor) or features.ndim != 4:
        raise ValueError(f"{path}: malformed features tensor.")
    if not isinstance(valid, torch.Tensor) or valid.shape != features.shape[:3]:
        raise ValueError(f"{path}: malformed valid_mask.")
    if not isinstance(labels, torch.Tensor) or labels.shape != features.shape[:1]:
        raise ValueError(f"{path}: malformed labels.")
    if tuple(payload.get("sensor_names", ())) != tuple(SENSOR_ORDER):
        raise ValueError(f"{path}: sensor ordering differs.")
    if not valid[:, :, 0].any(dim=1).all():
        raise ValueError(f"{path}: row without current sensor evidence.")
    if not torch.isfinite(features.float()).all():
        raise ValueError(f"{path}: non-finite features.")
    if set(labels.long().tolist()) != {0, 1}:
        raise ValueError(f"{path}: both binary classes are required.")
    rows = int(features.shape[0])
    for key in (
        "ids",
        "plume_ids",
        "event_ids",
        "cluster_ids",
        "macro_region_ids",
        "availability_signatures",
    ):
        if len(payload.get(key, ())) != rows:
            raise ValueError(f"{path}: metadata length mismatch for {key}.")
    manifest = payload.get("manifest", {})
    manifest_path = assert_safe_path(
        manifest.get("path", ""), purpose="cached source manifest"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if sha256_file(manifest_path) != manifest.get("sha256"):
        raise RuntimeError(f"{path}: source manifest changed after extraction.")
    safety = payload.get("safety", {})
    if safety.get("external_test_manifest_read") is not False:
        raise RuntimeError(f"{path}: missing no-test-read safety declaration.")


def _load_cache_pair(
    train_path: Path, val_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    train = torch_load_trusted(train_path)
    val = torch_load_trusted(val_path)
    _validate_feature_cache(train_path, train, expected_split="train")
    _validate_feature_cache(val_path, val, expected_split="inner_val")
    train_encoder = train["encoder"]
    val_encoder = val["encoder"]
    if train_encoder != val_encoder:
        raise ValueError("Train/validation encoder provenance differs.")
    if train["features"].shape[1:] != val["features"].shape[1:]:
        raise ValueError("Train/validation feature shapes are incompatible.")
    overlap = {}
    for key in ("event_ids", "plume_ids", "cluster_ids", "macro_region_ids"):
        values = sorted(set(train[key]) & set(val[key]))
        overlap[key] = values
    nonempty = {key: values[:20] for key, values in overlap.items() if values}
    if nonempty:
        raise RuntimeError(f"Feature-cache leakage detected: {nonempty}")
    audit = {
        "train_cache": str(train_path),
        "train_cache_sha256": sha256_file(train_path),
        "val_cache": str(val_path),
        "val_cache_sha256": sha256_file(val_path),
        "encoder": dict(train_encoder),
        "cross_split_overlap": {key: 0 for key in overlap},
    }
    return train, val, audit


def _list_sha256(values: Sequence[Any]) -> str:
    return sha256_bytes(canonical_json_bytes([str(value) for value in values]))


def _tensor_contract_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(tensor.shape)))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _dataset_contract(
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
) -> dict[str, Any]:
    def split_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "manifest_sha256": payload["manifest"]["sha256"],
            "query360_indices_sha256": _tensor_contract_sha256(
                payload["query360_indices"]
            ),
            "labels_sha256": _tensor_contract_sha256(
                payload["labels"].long()
            ),
            "valid_mask_sha256": _tensor_contract_sha256(
                payload["valid_mask"].bool()
            ),
            "ids_sha256": _list_sha256(payload["ids"]),
            "event_ids_sha256": _list_sha256(payload["event_ids"]),
            "rows": int(payload["features"].shape[0]),
        }

    return {
        "train": split_contract(train_cache),
        "inner_val": split_contract(val_cache),
        "sensor_names": list(train_cache["sensor_names"]),
        "role_names": list(train_cache["role_names"]),
    }


def _metric_value(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    value = metrics.get("overall", {}).get(key)
    return None if value is None else float(value)


def _train_one_head_arm(
    *,
    arm: str,
    model: TransientQuery360Head,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
    initial_state_sha256: str,
    parameter_signature: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
) -> dict[str, Any]:
    device = configure_device(args.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    labels = train_cache["labels"].float()
    positives = float(labels.sum())
    negatives = float(labels.numel() - positives)
    pos_weight = negatives / max(positives, 1.0)
    arm_dir = output_dir / f"seed_{seed}" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best: Optional[dict[str, Any]] = None
    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_head_epoch(
            model,
            optimizer,
            features=train_cache["features"],
            valid_mask=train_cache["valid_mask"],
            labels=train_cache["labels"].float(),
            arm=arm,
            event_ids=train_cache["event_ids"],
            plume_ids=train_cache["plume_ids"],
            batch_size=args.batch_size,
            seed=seed,
            epoch=epoch,
            auxiliary_weight=args.auxiliary_weight,
            pos_weight=pos_weight,
            grad_clip=args.grad_clip,
            max_steps=args.max_train_steps or None,
        )
        validation, probabilities = evaluate_head(
            model,
            features=val_cache["features"],
            valid_mask=val_cache["valid_mask"],
            labels=val_cache["labels"].float(),
            arm=arm,
            batch_size=args.eval_batch_size,
            auxiliary_weight=args.auxiliary_weight,
            pos_weight=pos_weight,
        )
        record = {
            "epoch": int(epoch),
            "train": train_metrics,
            "validation": validation,
            "elapsed_seconds": float(time.monotonic() - started),
        }
        history.append(record)
        ap = _metric_value(validation, "ap")
        if ap is None:
            raise RuntimeError("Validation AP is undefined.")
        if best is None or ap > float(best["validation"]["overall"]["ap"]):
            best = copy.deepcopy(record)
            checkpoint = {
                "schema_version": "query360-head-checkpoint-v1",
                "script_version": SCRIPT_VERSION,
                "encoder": dict(train_cache["encoder"]),
                "arm": arm,
                "seed": int(seed),
                "epoch": int(epoch),
                "model": {
                    key: tensor.detach().cpu()
                    for key, tensor in model.state_dict().items()
                },
                "optimizer_steps": int(train_metrics["steps"]),
                "initial_state_sha256": initial_state_sha256,
                "parameter_signature": dict(parameter_signature),
                "cache_audit": dict(cache_audit),
                "validation": validation,
            }
            atomic_torch_save(arm_dir / "checkpoint_best_ap.pt", checkpoint)
            predictions = pd.DataFrame(
                {
                    "id": val_cache["ids"],
                    "plume_id": val_cache["plume_ids"],
                    "cluster_id": val_cache["cluster_ids"],
                    "macro_region_id": val_cache["macro_region_ids"],
                    "availability_signature": val_cache[
                        "availability_signatures"
                    ],
                    "label": val_cache["labels"].long().tolist(),
                    "probability": probabilities,
                    "prediction_at_0_5": (
                        probabilities >= 0.5
                    ).astype(np.int64),
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch,
                }
            )
            atomic_csv_write(
                arm_dir / "validation_best_ap_predictions.csv", predictions
            )
        atomic_json_write(arm_dir / "metrics_history.json", history)
        overall = validation["overall"]
        print(
            f"[head] encoder={train_cache['encoder']['condition']} "
            f"seed={seed} arm={arm} epoch={epoch}/{args.epochs} "
            f"loss={train_metrics['loss']:.6f} "
            f"AP={overall['ap']:.6f} AUC={overall['auc']:.6f} "
            f"F1@.5={overall['macro_f1_at_0_5']:.6f} "
            f"bestF1={overall['best_macro_f1']:.6f}",
            flush=True,
        )
    if best is None:
        raise RuntimeError(f"{arm}: no training result.")
    return best


def _paired_delta(
    seed_results: Sequence[Mapping[str, Any]],
    left_arm: str,
    right_arm: str,
    metric: str,
) -> dict[str, Any]:
    values = []
    for result in seed_results:
        left = result["arms"][left_arm]["validation"]["overall"].get(metric)
        right = result["arms"][right_arm]["validation"]["overall"].get(metric)
        if left is not None and right is not None:
            values.append(float(left) - float(right))
    return {
        "left": left_arm,
        "right": right_arm,
        "metric": metric,
        "per_seed": values,
        "mean": float(np.mean(values)) if values else None,
        "sample_std": (
            float(np.std(values, ddof=1)) if len(values) >= 2 else None
        ),
    }


def run_train_heads(args: argparse.Namespace) -> None:
    train_path = assert_safe_path(args.train_cache, purpose="train feature cache")
    val_path = assert_safe_path(args.val_cache, purpose="validation feature cache")
    output_dir = assert_safe_path(args.output_dir, purpose="head output")
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("Both train and inner-validation caches are required.")
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists; pass --overwrite.")
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
        train_cache, val_cache, cache_audit = _load_cache_pair(
            train_path, val_path
        )
        requested_arms = parse_csv_list(args.arms)
        unknown = sorted(set(requested_arms) - set(ARM_NAMES))
        if unknown:
            raise ValueError(f"Unknown arms: {unknown}; expected {ARM_NAMES}.")
        seeds = parse_int_list(args.seeds)
        model_kwargs = {
            "feature_dim": int(train_cache["features"].shape[-1]),
            "num_sensors": int(train_cache["features"].shape[1]),
            "num_roles": int(train_cache["features"].shape[2]),
            "model_dim": int(args.model_dim),
            "num_heads": int(args.num_heads),
            "depth": 2,
            "mlp_ratio": float(args.mlp_ratio),
            "dropout": float(args.dropout),
        }
        seed_results = []
        for seed in seeds:
            models, initial_sha, signature = build_matched_models(
                model_kwargs=model_kwargs,
                seed=seed,
                arms=requested_arms,
            )
            arm_results = {}
            for arm in requested_arms:
                arm_results[arm] = _train_one_head_arm(
                    arm=arm,
                    model=models[arm],
                    train_cache=train_cache,
                    val_cache=val_cache,
                    args=args,
                    seed=seed,
                    output_dir=output_dir,
                    initial_state_sha256=initial_sha,
                    parameter_signature=signature,
                    cache_audit=cache_audit,
                )
            seed_results.append(
                {
                    "seed": int(seed),
                    "initial_state_sha256": initial_sha,
                    "parameter_signature": signature,
                    "arms": arm_results,
                }
            )

        comparisons = []
        for metric in (
            "ap",
            "auc",
            "macro_f1_at_0_5",
            "best_macro_f1",
        ):
            if {
                "transient_query",
                "current_only",
            }.issubset(requested_arms):
                comparisons.append(
                    _paired_delta(
                        seed_results,
                        "transient_query",
                        "current_only",
                        metric,
                    )
                )
            if {
                "transient_query",
                "history_shuffle_train",
            }.issubset(requested_arms):
                comparisons.append(
                    _paired_delta(
                        seed_results,
                        "transient_query",
                        "history_shuffle_train",
                        metric,
                    )
                )
            if {
                "scale_aware_transient_query",
                "current_only",
            }.issubset(requested_arms):
                comparisons.append(
                    _paired_delta(
                        seed_results,
                        "scale_aware_transient_query",
                        "current_only",
                        metric,
                    )
                )
            if {
                "scale_aware_transient_query",
                "transient_query",
            }.issubset(requested_arms):
                comparisons.append(
                    _paired_delta(
                        seed_results,
                        "scale_aware_transient_query",
                        "transient_query",
                        metric,
                    )
                )
        training_contract = {
            "arms": list(requested_arms),
            "seeds": list(seeds),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "auxiliary_weight": float(args.auxiliary_weight),
            "grad_clip": float(args.grad_clip),
            "model_dim": int(args.model_dim),
            "num_heads": int(args.num_heads),
            "depth": 2,
            "mlp_ratio": float(args.mlp_ratio),
            "dropout": float(args.dropout),
            "max_train_steps": int(args.max_train_steps),
            "selection_metric": "inner_validation_ap",
        }
        summary = {
            "schema_version": "query360-head-summary-v1",
            "script_version": SCRIPT_VERSION,
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "encoder": dict(train_cache["encoder"]),
            "train_rows": int(train_cache["features"].shape[0]),
            "validation_rows": int(val_cache["features"].shape[0]),
            "arms": list(requested_arms),
            "seeds": list(seeds),
            "epochs": int(args.epochs),
            "selection_metric": "inner_validation_ap",
            "training_contract": training_contract,
            "dataset_contract": _dataset_contract(train_cache, val_cache),
            "seed_results": seed_results,
            "paired_deltas": comparisons,
            "cache_audit": cache_audit,
            "matched_contract": {
                "same_head_architecture": True,
                "same_initial_state_within_seed": True,
                "same_epoch_batches_within_seed": True,
                "same_optimizer_hyperparameters": True,
                "same_dense_feature_projection": True,
                "history_is_the_only_arm_intervention": True,
            },
            "safety": {
                "external_test_manifest_read": False,
                "test_sealed_holdout_paths_refused": True,
                "checkpoint_selected_on_inner_validation_only": True,
            },
            "script_provenance": script_provenance(),
        }
        atomic_json_write(summary_path, summary)
        atomic_json_write(
            output_dir / "run_status.json",
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "summary": str(summary_path),
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
            },
        )
        raise


def _summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    encoder = summary["encoder"]["condition"]
    for seed_result in summary["seed_results"]:
        seed = int(seed_result["seed"])
        for arm, best in seed_result["arms"].items():
            overall = best["validation"]["overall"]
            row = {
                "encoder": encoder,
                "seed": seed,
                "arm": arm,
                "best_epoch": int(best["epoch"]),
            }
            for metric in (
                "ap",
                "auc",
                "macro_f1_at_0_5",
                "best_macro_f1",
                "best_macro_f1_threshold",
                "balanced_accuracy_at_0_5",
            ):
                row[metric] = overall.get(metric)
            for stratum in ("single_sensor", "multisensor"):
                metrics = best["validation"]["sensor_count"][stratum]
                row[f"{stratum}_rows"] = metrics.get("rows")
                row[f"{stratum}_ap"] = metrics.get("ap")
                row[f"{stratum}_macro_f1_at_0_5"] = metrics.get(
                    "macro_f1_at_0_5"
                )
            rows.append(row)
    return rows


def run_compare(args: argparse.Namespace) -> None:
    summary_paths = [
        assert_safe_path(path, purpose="head summary")
        for path in parse_csv_list(args.summaries)
    ]
    summaries = []
    for path in summary_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as stream:
            summaries.append(json.load(stream))
    if len(summaries) != 2:
        raise ValueError("A formal encoder comparison requires exactly two summaries.")
    expected_conditions = {"panopticon_pretrained", "random_frozen"}
    observed_conditions = {
        summary.get("encoder", {}).get("condition") for summary in summaries
    }
    if observed_conditions != expected_conditions:
        raise ValueError(
            "Formal comparison requires exactly the pretrained and random-frozen "
            f"encoder conditions, got {sorted(str(value) for value in observed_conditions)}."
        )
    reference_dataset = summaries[0].get("dataset_contract")
    reference_training = summaries[0].get("training_contract")
    if not reference_dataset or not reference_training:
        raise ValueError("Summary lacks formal dataset/training provenance contracts.")
    for summary in summaries[1:]:
        if summary.get("dataset_contract") != reference_dataset:
            raise ValueError(
                "Encoder summaries do not use identical manifests, rows, labels, "
                "validity masks, or sensor/role ordering."
            )
        if summary.get("training_contract") != reference_training:
            raise ValueError(
                "Encoder summaries do not use identical arms, seeds, epochs, "
                "head architecture, or optimizer hyperparameters."
            )
    frame = pd.DataFrame(
        [row for summary in summaries for row in _summary_rows(summary)]
    )
    output_dir = assert_safe_path(args.output_dir, purpose="comparison output")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv_write(output_dir / "all_best_seed_results.csv", frame)
    numeric_metrics = [
        "ap",
        "auc",
        "macro_f1_at_0_5",
        "best_macro_f1",
        "balanced_accuracy_at_0_5",
        "single_sensor_ap",
        "multisensor_ap",
    ]
    aggregate = (
        frame.groupby(["encoder", "arm"])[numeric_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in aggregate.columns
    ]
    atomic_csv_write(output_dir / "aggregate_metrics.csv", aggregate)
    comparison = {
        "schema_version": "query360-encoder-comparison-v1",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "summaries": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in summary_paths
        ],
        "rows": frame.to_dict("records"),
        "within_encoder_paired_deltas": {
            summary["encoder"]["condition"]: summary["paired_deltas"]
            for summary in summaries
        },
        "matched_dataset_contract": reference_dataset,
        "matched_training_contract": reference_training,
        "interpretation_guardrail": (
            "random_frozen is a representation control and is not "
            "end-to-end scratch training"
        ),
        "external_test_manifest_read": False,
    }
    atomic_json_write(output_dir / "comparison.json", comparison)
    print(aggregate.to_string(index=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--source-train-csv", default=DEFAULT_SOURCE_TRAIN)
    split_parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    split_parser.add_argument("--seed", type=int, default=360)
    split_parser.add_argument("--val-fraction", type=float, default=0.15)
    split_parser.add_argument("--min-val-regions", type=int, default=8)
    split_parser.add_argument("--search-trials", type=int, default=8192)
    split_parser.add_argument("--overwrite", action="store_true")
    split_parser.set_defaults(function=run_split)

    warm_parser = subparsers.add_parser("warm-cache")
    warm_parser.add_argument("--manifests", required=True)
    warm_parser.add_argument("--cache-dir", required=True)
    warm_parser.add_argument("--workers", type=int, default=32)
    warm_parser.set_defaults(function=run_warm_cache)

    cache_parser = subparsers.add_parser("cache-features")
    cache_parser.add_argument("--manifest", required=True)
    cache_parser.add_argument("--split", choices=["train", "inner_val"], required=True)
    cache_parser.add_argument(
        "--encoder-condition",
        choices=["panopticon_pretrained", "random_frozen"],
        required=True,
    )
    cache_parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    cache_parser.add_argument("--encoder-seed", type=int, default=360)
    cache_parser.add_argument("--raw-cache-dir", required=True)
    cache_parser.add_argument("--output-cache", required=True)
    cache_parser.add_argument("--wv3-srf-csv", default=str(DEFAULT_WV3_SRF))
    cache_parser.add_argument("--device", default="cuda:0")
    cache_parser.add_argument(
        "--amp-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    cache_parser.add_argument(
        "--storage-dtype", choices=["float16", "float32"], default="float16"
    )
    cache_parser.add_argument("--row-batch-size", type=int, default=32)
    cache_parser.add_argument("--encoder-microbatch", type=int, default=64)
    cache_parser.add_argument("--num-workers", type=int, default=12)
    cache_parser.add_argument("--prefetch-factor", type=int, default=2)
    cache_parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=True
    )
    cache_parser.add_argument("--log-interval", type=int, default=10)
    cache_parser.add_argument("--max-rows", type=int, default=0)
    cache_parser.add_argument("--overwrite", action="store_true")
    cache_parser.set_defaults(function=run_cache_features)

    head_parser = subparsers.add_parser("train-heads")
    head_parser.add_argument("--train-cache", required=True)
    head_parser.add_argument("--val-cache", required=True)
    head_parser.add_argument("--output-dir", required=True)
    head_parser.add_argument(
        "--arms", default="current_only,transient_query,history_shuffle_train"
    )
    head_parser.add_argument("--seeds", default="17,29,43")
    head_parser.add_argument("--epochs", type=int, default=5)
    head_parser.add_argument("--batch-size", type=int, default=256)
    head_parser.add_argument("--eval-batch-size", type=int, default=512)
    head_parser.add_argument("--learning-rate", type=float, default=3e-4)
    head_parser.add_argument("--weight-decay", type=float, default=0.05)
    head_parser.add_argument("--auxiliary-weight", type=float, default=0.3)
    head_parser.add_argument("--grad-clip", type=float, default=1.0)
    head_parser.add_argument("--model-dim", type=int, default=256)
    head_parser.add_argument("--num-heads", type=int, default=8)
    head_parser.add_argument("--mlp-ratio", type=float, default=2.0)
    head_parser.add_argument("--dropout", type=float, default=0.1)
    head_parser.add_argument("--max-train-steps", type=int, default=0)
    head_parser.add_argument("--device", default="cuda:0")
    head_parser.add_argument("--overwrite", action="store_true")
    head_parser.set_defaults(function=run_train_heads)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--summaries", required=True)
    compare_parser.add_argument("--output-dir", required=True)
    compare_parser.set_defaults(function=run_compare)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
