#!/usr/bin/env python3
"""Shard legacy-360 extraction work and strictly merge feature caches.

The feature extractor is deliberately left unchanged.  This utility provides
the two operations needed to run it on two GPUs:

``shard-manifest``
    Assign complete acquisition events to two deterministic train shards.
    The greedy scheduler starts GPU 1 with the estimated cost of the complete
    development manifest, so that ``GPU0 train`` and ``GPU1 train + dev`` have
    nearly equal estimated work.  Existing global ``query360_index`` values
    are retained byte-for-byte.

``merge-cache``
    Validate independently extracted shard caches, including the exact source
    manifests named inside them, then concatenate every row tensor and row
    metadata list.  IDs and global query indices must be globally unique;
    plume IDs may repeat within a shard but may not cross shard boundaries.
    The merged cache is always labelled ``train_core``.

All CSV, PT, JSON, and SHA sidecar writes are atomic replacements in the
destination filesystem.  Existing outputs are refused unless ``--overwrite``
is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


SCRIPT_VERSION = "legacy360-feature-shard-merge-v1"
FEATURE_SCHEMA = "query360-two-axis-feature-cache-v1"
SENSOR_NAMES = ("s2", "l89", "emit", "s5p")
SENSOR_COSTS = {"s2": 5, "l89": 4, "emit": 4, "s5p": 4}
OPTICAL_ROLES = ("0", "90", "360")
OPTICAL_COLUMNS = {
    sensor: tuple(f"{sensor}_{role}_path" for role in OPTICAL_ROLES)
    for sensor in ("s2", "l89", "emit")
}
S5P_COLUMN = "s5p_0_path"
EMPTY_TOKENS = {"", "nan", "none", "null"}

REQUIRED_ROW_LIST_FIELDS = (
    "ids",
    "plume_ids",
    "availability_signatures",
)
OPTIONAL_ROW_LIST_FIELDS = ("event_ids", "query360_indices")

# These are intentional aliases in query360_two_axis_full_legacy.py.  Keeping
# them as aliases after merging avoids duplicating multi-gigabyte tensors.
TENSOR_ALIASES = {
    "features_hybrid": "features",
    "base_sensor_logits_hybrid": "base_sensor_logits",
    "base_sensor_valid_hybrid": "base_sensor_valid",
    "base_fused_logits": "base_hybrid_logits",
}


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.casefold() in EMPTY_TOKENS else text


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalise_for_fingerprint(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_for_fingerprint(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_for_fingerprint(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__type__": type(value).__name__, "__repr__": repr(value)}


def object_fingerprint(value: Any) -> str:
    canonical = json.dumps(
        _normalise_for_fingerprint(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _temporary_path(destination: Path, suffix: str) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=suffix,
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary_name = _temporary_path(path, ".csv.tmp")
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


def atomic_json(path: Path, payload: Any) -> None:
    descriptor, temporary_name = _temporary_path(path, ".json.tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
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


def atomic_text(path: Path, text: str) -> None:
    descriptor, temporary_name = _temporary_path(path, ".txt.tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_torch(path: Path, payload: Any) -> None:
    descriptor, temporary_name = _temporary_path(path, ".pt.tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def refuse_existing(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing outputs: " + ", ".join(existing)
        )


def read_manifest(path: Path, role: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(
        path, dtype=str, keep_default_na=False, low_memory=False
    )
    required = {
        "id",
        "plume_id",
        "label",
        "query360_index",
        "availability_signature",
        S5P_COLUMN,
        *(column for columns in OPTICAL_COLUMNS.values() for column in columns),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{role} manifest misses columns: {missing}")
    if frame.empty:
        raise ValueError(f"{role} manifest is empty: {path}")

    for column in ("id", "plume_id", "query360_index"):
        values = frame[column].map(clean)
        if values.eq("").any():
            raise ValueError(f"{role}.{column} contains blanks")
        frame[column] = values
    if frame["id"].duplicated().any():
        examples = frame.loc[
            frame["id"].duplicated(keep=False), "id"
        ].head(20)
        raise ValueError(f"{role} has duplicate ids: {examples.tolist()}")
    if not frame["query360_index"].str.fullmatch(r"[+-]?\d+").all():
        examples = frame.loc[
            ~frame["query360_index"].str.fullmatch(r"[+-]?\d+"),
            "query360_index",
        ].head(20)
        raise ValueError(
            f"{role} has non-integer query360_index values: {examples.tolist()}"
        )
    query_indices = frame["query360_index"].astype(np.int64)
    if query_indices.duplicated().any():
        examples = query_indices[query_indices.duplicated(keep=False)].head(20)
        raise ValueError(
            f"{role} has duplicate query360_index values: {examples.tolist()}"
        )

    labels = pd.to_numeric(frame["label"], errors="raise").astype(np.int64)
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{role}.label is not binary")
    frame["label"] = labels

    costs: list[int] = []
    signatures: list[str] = []
    for row_number, row in frame.iterrows():
        available: list[str] = []
        for sensor, columns in OPTICAL_COLUMNS.items():
            presence = [bool(clean(row[column])) for column in columns]
            if any(presence) and not all(presence):
                raise ValueError(
                    f"{role} row {row_number} has a partial {sensor} triplet"
                )
            if all(presence):
                available.append(sensor)
        if clean(row[S5P_COLUMN]):
            available.append("s5p")
        signature = "+".join(available)
        if not signature:
            raise ValueError(f"{role} row {row_number} has no sensor")
        declared_signature = clean(row["availability_signature"])
        if signature != declared_signature:
            raise ValueError(
                f"{role} row {row_number} availability mismatch: "
                f"paths={signature!r}, column={declared_signature!r}"
            )
        signatures.append(signature)
        costs.append(sum(SENSOR_COSTS[sensor] for sensor in available))
    frame["_estimated_cost"] = np.asarray(costs, dtype=np.int64)
    frame["_source_position"] = np.arange(len(frame), dtype=np.int64)
    return frame


def label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame["label"].value_counts().sort_index().items()
    }


def signature_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame["availability_signature"]
        .value_counts()
        .sort_index()
        .items()
    }


def public_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=["_estimated_cost", "_source_position"], errors="ignore"
    )


def command_shard_manifest(args: argparse.Namespace) -> None:
    train_path = Path(args.train_manifest).expanduser().absolute()
    dev_path = Path(args.dev_manifest).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    shard0_path = (
        Path(args.shard0).expanduser().absolute()
        if args.shard0
        else output_dir / f"{args.prefix}_gpu0.csv"
    )
    shard1_path = (
        Path(args.shard1).expanduser().absolute()
        if args.shard1
        else output_dir / f"{args.prefix}_gpu1.csv"
    )
    audit_path = (
        Path(args.audit).expanduser().absolute()
        if args.audit
        else output_dir / f"{args.prefix}_shard_audit.json"
    )
    audit_sha_path = audit_path.with_suffix(audit_path.suffix + ".sha256")
    refuse_existing(
        (shard0_path, shard1_path, audit_path, audit_sha_path),
        bool(args.overwrite),
    )

    train = read_manifest(train_path, "train_core")
    dev = read_manifest(dev_path, "dev")
    train_ids = set(train["id"])
    dev_ids = set(dev["id"])
    if train_ids & dev_ids:
        raise ValueError("train_core and dev contain overlapping ids")
    train_queries = set(train["query360_index"].astype(np.int64))
    dev_queries = set(dev["query360_index"].astype(np.int64))
    if train_queries & dev_queries:
        raise ValueError("train_core and dev contain overlapping query indices")
    train_plumes = set(train["plume_id"])
    dev_plumes = set(dev["plume_id"])
    if train_plumes & dev_plumes:
        raise ValueError("train_core and dev contain overlapping plume ids")

    # Use event_id where available so related plume variants cannot be split
    # across devices.  Falling back to plume_id still guarantees no plume
    # overlap between shards.
    group_column = "event_id" if "event_id" in train.columns else "plume_id"
    if train[group_column].map(clean).eq("").any():
        raise ValueError(f"train_core.{group_column} contains blanks")
    group_costs = (
        train.groupby(group_column, sort=False, observed=True)["_estimated_cost"]
        .sum()
        .astype(np.int64)
    )
    groups = sorted(
        ((str(group), int(cost)) for group, cost in group_costs.items()),
        key=lambda item: (-item[1], item[0]),
    )

    dev_cost = int(dev["_estimated_cost"].sum())
    loads = [0, dev_cost]
    assignment: dict[str, int] = {}
    for group, cost in groups:
        target = 0 if loads[0] <= loads[1] else 1
        assignment[group] = target
        loads[target] += cost

    shard_index = train[group_column].map(assignment)
    if shard_index.isna().any():
        raise AssertionError("greedy assignment missed groups")
    shards = [
        train.loc[shard_index.eq(index)]
        .sort_values("_source_position", kind="stable")
        .copy()
        for index in (0, 1)
    ]
    if any(shard.empty for shard in shards):
        raise ValueError("greedy assignment produced an empty shard")
    if set(shards[0]["plume_id"]) & set(shards[1]["plume_id"]):
        raise AssertionError("plume id crossed train shards")
    if (
        "event_id" in train.columns
        and set(shards[0]["event_id"]) & set(shards[1]["event_id"])
    ):
        raise AssertionError("event id crossed train shards")

    original_queries = train["query360_index"].astype(np.int64).tolist()
    output_queries = sorted(
        shards[0]["query360_index"].astype(np.int64).tolist()
        + shards[1]["query360_index"].astype(np.int64).tolist()
    )
    if sorted(original_queries) != output_queries:
        raise AssertionError("sharding changed the global query index universe")

    atomic_csv(shard0_path, public_manifest(shards[0]))
    atomic_csv(shard1_path, public_manifest(shards[1]))
    shard_costs = [int(shard["_estimated_cost"].sum()) for shard in shards]
    audit = {
        "schema_version": "legacy360-extraction-shards-v1",
        "script_version": SCRIPT_VERSION,
        "scheduler": {
            "algorithm": (
                "deterministic largest-event-first greedy list scheduling"
            ),
            "group_column": group_column,
            "sensor_costs": SENSOR_COSTS,
            "initial_loads": {"gpu0": 0, "gpu1": dev_cost},
            "goal": "gpu0_train ~= gpu1_train + dev",
        },
        "inputs": {
            "train_core": {
                "path": str(train_path),
                "sha256": sha256_file(train_path),
                "rows": int(len(train)),
                "estimated_cost": int(train["_estimated_cost"].sum()),
            },
            "dev": {
                "path": str(dev_path),
                "sha256": sha256_file(dev_path),
                "rows": int(len(dev)),
                "estimated_cost": dev_cost,
            },
        },
        "outputs": {
            "gpu0": {
                "path": str(shard0_path),
                "sha256": sha256_file(shard0_path),
                "rows": int(len(shards[0])),
                "groups": int(shards[0][group_column].nunique()),
                "plumes": int(shards[0]["plume_id"].nunique()),
                "labels": label_counts(shards[0]),
                "availability_signatures": signature_counts(shards[0]),
                "train_cost": shard_costs[0],
                "scheduled_total_cost": shard_costs[0],
            },
            "gpu1": {
                "path": str(shard1_path),
                "sha256": sha256_file(shard1_path),
                "rows": int(len(shards[1])),
                "groups": int(shards[1][group_column].nunique()),
                "plumes": int(shards[1]["plume_id"].nunique()),
                "labels": label_counts(shards[1]),
                "availability_signatures": signature_counts(shards[1]),
                "train_cost": shard_costs[1],
                "dev_cost": dev_cost,
                "scheduled_total_cost": shard_costs[1] + dev_cost,
            },
        },
        "balance": {
            "absolute_cost_delta": abs(
                shard_costs[0] - (shard_costs[1] + dev_cost)
            ),
            "relative_cost_delta": abs(
                shard_costs[0] - (shard_costs[1] + dev_cost)
            )
            / max(1, max(shard_costs[0], shard_costs[1] + dev_cost)),
            "maximum_group_cost": max(cost for _group, cost in groups),
        },
        "integrity": {
            "rows_preserved": int(len(shards[0]) + len(shards[1]))
            == int(len(train)),
            "query360_index_preserved": True,
            "query360_index_sha256_sorted": sha256_lines(
                sorted(original_queries)
            ),
            "id_overlap_between_shards": 0,
            "plume_overlap_between_shards": 0,
            "dev_plume_overlap": 0,
        },
    }
    atomic_json(audit_path, audit)
    audit_sha = sha256_file(audit_path)
    atomic_text(
        audit_sha_path,
        f"{audit_sha}  {audit_path.name}\n",
    )
    result = {
        "shard0": str(shard0_path),
        "shard1": str(shard1_path),
        "audit": str(audit_path),
        "audit_sha256": audit_sha,
        "scheduled_costs": {
            "gpu0": shard_costs[0],
            "gpu1_plus_dev": shard_costs[1] + dev_cost,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: feature cache is not a dictionary")
    return payload


def _tensor_fields(payload: Mapping[str, Any]) -> set[str]:
    return {
        str(key)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
    }


def _row_list_fields(payload: Mapping[str, Any]) -> set[str]:
    fields = {
        key
        for key in (*REQUIRED_ROW_LIST_FIELDS, *OPTIONAL_ROW_LIST_FIELDS)
        if key in payload
    }
    missing = sorted(set(REQUIRED_ROW_LIST_FIELDS) - fields)
    if missing:
        raise ValueError(f"feature cache misses row metadata: {missing}")
    return fields


def _tensors_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if (
        left.device.type == "cpu"
        and right.device.type == "cpu"
        and left.untyped_storage().data_ptr()
        == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and left.stride() == right.stride()
    ):
        return True
    return bool(torch.equal(left, right))


def _tensor_storage_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        tensor.device.type,
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _tensor_is_finite(tensor: torch.Tensor, row_chunk: int = 256) -> bool:
    """Bound temporary memory while auditing multi-gigabyte feature tensors."""

    for start in range(0, int(tensor.shape[0]), row_chunk):
        if not bool(torch.isfinite(tensor[start : start + row_chunk]).all()):
            return False
    return True


def _read_cache_manifest(
    cache_path: Path,
    payload: Mapping[str, Any],
    rows: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[int]]:
    metadata = payload.get("manifest")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{cache_path}: manifest metadata is missing")
    if "path" not in metadata or "sha256" not in metadata:
        raise ValueError(f"{cache_path}: malformed manifest metadata")
    manifest_path = Path(str(metadata["path"])).expanduser().absolute()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{cache_path}: source manifest is unavailable: {manifest_path}"
        )
    actual_sha = sha256_file(manifest_path)
    if str(metadata["sha256"]) != actual_sha:
        raise ValueError(f"{cache_path}: source manifest SHA256 mismatch")
    if int(metadata.get("rows", -1)) != rows:
        raise ValueError(f"{cache_path}: manifest row count metadata mismatch")
    frame = read_manifest(manifest_path, f"cache {cache_path.name}")
    if len(frame) != rows:
        raise ValueError(f"{cache_path}: source manifest length mismatch")

    ids = [str(value) for value in payload["ids"]]
    plumes = [str(value) for value in payload["plume_ids"]]
    if ids != frame["id"].astype(str).tolist():
        raise ValueError(f"{cache_path}: cached IDs do not match its manifest")
    if plumes != frame["plume_id"].astype(str).tolist():
        raise ValueError(
            f"{cache_path}: cached plume IDs do not match its manifest"
        )
    labels = payload["labels"].detach().cpu().long().tolist()
    if labels != frame["label"].astype(np.int64).tolist():
        raise ValueError(f"{cache_path}: cached labels do not match its manifest")
    signatures = [
        str(value) for value in payload["availability_signatures"]
    ]
    if signatures != frame["availability_signature"].astype(str).tolist():
        raise ValueError(
            f"{cache_path}: cached availability does not match its manifest"
        )
    if "event_ids" in payload:
        expected_events = (
            frame["event_id"].astype(str).tolist()
            if "event_id" in frame
            else frame["plume_id"].astype(str).tolist()
        )
        if [str(value) for value in payload["event_ids"]] != expected_events:
            raise ValueError(
                f"{cache_path}: cached event IDs do not match its manifest"
            )
    queries = frame["query360_index"].astype(np.int64).tolist()
    if "query360_indices" in payload:
        cached_queries = [
            int(value) for value in payload["query360_indices"]
        ]
        if cached_queries != queries:
            raise ValueError(
                f"{cache_path}: cached query indices do not match its manifest"
            )
    return frame, {
        "path": str(manifest_path),
        "sha256": actual_sha,
        "rows": rows,
    }, queries


def _validate_cache(
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_version") != FEATURE_SCHEMA:
        raise ValueError(
            f"{path}: schema={payload.get('schema_version')!r}, "
            f"expected={FEATURE_SCHEMA!r}"
        )
    script_version = clean(payload.get("script_version"))
    if not script_version:
        raise ValueError(f"{path}: blank script_version")
    if payload.get("split") != "train_core":
        raise ValueError(
            f"{path}: split={payload.get('split')!r}, expected='train_core'"
        )
    if list(payload.get("sensor_names", [])) != list(SENSOR_NAMES):
        raise ValueError(f"{path}: wrong or reordered sensor_names")
    if not isinstance(payload.get("encoder"), Mapping):
        raise ValueError(f"{path}: encoder metadata is missing")
    if bool(payload.get("sealed_test_read", False)):
        raise ValueError(f"{path}: refusing a cache marked as sealed-test read")

    tensors = _tensor_fields(payload)
    if not {"features", "valid_mask", "labels"}.issubset(tensors):
        raise ValueError(f"{path}: required row tensors are missing")
    labels = payload["labels"]
    if labels.ndim != 1:
        raise ValueError(f"{path}: labels must be rank one")
    rows = int(labels.shape[0])
    if rows < 1:
        raise ValueError(f"{path}: empty feature cache")
    checked_float_storages: set[tuple[Any, ...]] = set()
    for key in sorted(tensors):
        tensor = payload[key]
        if tensor.ndim < 1 or int(tensor.shape[0]) != rows:
            raise ValueError(
                f"{path}: tensor {key!r} is not row-aligned ({tensor.shape})"
            )
        if tensor.is_floating_point():
            storage_signature = _tensor_storage_signature(tensor)
            if storage_signature not in checked_float_storages:
                if not _tensor_is_finite(tensor):
                    raise ValueError(
                        f"{path}: tensor {key!r} contains non-finite values"
                    )
                checked_float_storages.add(storage_signature)

    row_lists = _row_list_fields(payload)
    for key in sorted(row_lists):
        value = payload[key]
        if not isinstance(value, (list, tuple)) or len(value) != rows:
            raise ValueError(f"{path}: row metadata {key!r} is malformed")

    frame, manifest, queries = _read_cache_manifest(path, payload, rows)
    return {
        "rows": rows,
        "tensor_fields": tensors,
        "row_list_fields": row_lists,
        "script_version": script_version,
        "encoder_fingerprint": object_fingerprint(payload["encoder"]),
        "base_definitions_fingerprint": object_fingerprint(
            payload.get("base_definitions")
        ),
        "frame": frame,
        "manifest": manifest,
        "queries": queries,
    }


def command_merge_cache(args: argparse.Namespace) -> None:
    input_paths = [
        Path(value).expanduser().absolute() for value in args.input_cache
    ]
    if len(input_paths) < 2:
        raise ValueError("merge-cache requires at least two --input-cache values")
    if len(set(input_paths)) != len(input_paths):
        raise ValueError("the same input cache was supplied more than once")
    output_path = Path(args.output_cache).expanduser().absolute()
    audit_path = (
        Path(args.audit).expanduser().absolute()
        if args.audit
        else output_path.with_suffix(output_path.suffix + ".audit.json")
    )
    audit_sha_path = audit_path.with_suffix(audit_path.suffix + ".sha256")
    manifest_path = (
        Path(args.output_manifest).expanduser().absolute()
        if args.output_manifest
        else output_path.with_suffix(output_path.suffix + ".manifest.csv")
    )
    refuse_existing(
        (output_path, manifest_path, audit_path, audit_sha_path),
        bool(args.overwrite),
    )
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path == output_path:
            raise ValueError("an input cache cannot also be the output cache")

    input_hashes = {
        path: sha256_file(path)
        for path in input_paths
    }
    payloads = [_load_torch(path) for path in input_paths]
    validations = [
        _validate_cache(path, payload)
        for path, payload in zip(input_paths, payloads)
    ]
    reference = validations[0]
    for path, validation in zip(input_paths[1:], validations[1:]):
        for key in (
            "tensor_fields",
            "row_list_fields",
            "script_version",
            "encoder_fingerprint",
            "base_definitions_fingerprint",
        ):
            if validation[key] != reference[key]:
                raise ValueError(
                    f"{path}: {key} differs from {input_paths[0]}"
                )

    # Exact trailing shape/dtype agreement for every row tensor.
    for key in sorted(reference["tensor_fields"]):
        first = payloads[0][key]
        for path, payload in zip(input_paths[1:], payloads[1:]):
            candidate = payload[key]
            if (
                candidate.dtype != first.dtype
                or candidate.shape[1:] != first.shape[1:]
            ):
                raise ValueError(
                    f"{path}: tensor {key!r} shape/dtype differs"
                )

    id_sets: list[set[str]] = []
    plume_sets: list[set[str]] = []
    query_sets: list[set[int]] = []
    for path, payload, validation in zip(
        input_paths, payloads, validations
    ):
        ids = [str(value) for value in payload["ids"]]
        queries = [int(value) for value in validation["queries"]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate IDs inside cache")
        if len(set(queries)) != len(queries):
            raise ValueError(f"{path}: duplicate query indices inside cache")
        id_sets.append(set(ids))
        plume_sets.append(set(str(value) for value in payload["plume_ids"]))
        query_sets.append(set(queries))
    for right in range(1, len(input_paths)):
        for left in range(right):
            if id_sets[left] & id_sets[right]:
                raise ValueError(
                    f"ID overlap between {input_paths[left]} and "
                    f"{input_paths[right]}"
                )
            if query_sets[left] & query_sets[right]:
                raise ValueError(
                    f"query360_index overlap between {input_paths[left]} and "
                    f"{input_paths[right]}"
                )
            if plume_sets[left] & plume_sets[right]:
                raise ValueError(
                    f"plume overlap between {input_paths[left]} and "
                    f"{input_paths[right]}"
                )

    row_tensor_fields = sorted(reference["tensor_fields"])
    alias_fields: dict[str, str] = {}
    for alias, canonical in TENSOR_ALIASES.items():
        if alias not in row_tensor_fields or canonical not in row_tensor_fields:
            continue
        if all(
            _tensors_equal(payload[alias], payload[canonical])
            for payload in payloads
        ):
            alias_fields[alias] = canonical

    merged: dict[str, Any] = {
        key: value
        for key, value in payloads[0].items()
        if key
        not in (
            set(row_tensor_fields)
            | set(reference["row_list_fields"])
            | {"manifest", "extraction", "split", "sealed_test_read"}
        )
    }
    for key in row_tensor_fields:
        if key in alias_fields:
            continue
        merged[key] = torch.cat([payload[key] for payload in payloads], dim=0)
    for alias, canonical in alias_fields.items():
        merged[alias] = merged[canonical]
    for key in sorted(reference["row_list_fields"]):
        if key == "query360_indices":
            # Re-derived below from the SHA-verified manifests.
            continue
        merged[key] = [
            item for payload in payloads for item in list(payload[key])
        ]
    merged_queries = [
        int(value)
        for validation in validations
        for value in validation["queries"]
    ]
    merged["query360_indices"] = merged_queries
    merged["split"] = "train_core"
    merged["sealed_test_read"] = False

    merged_frame = pd.concat(
        [validation["frame"] for validation in validations],
        ignore_index=True,
        sort=False,
    )
    # Internal validation columns came from read_manifest, not the source CSV.
    merged_frame = public_manifest(merged_frame)
    if merged_frame["id"].astype(str).tolist() != [
        str(value) for value in merged["ids"]
    ]:
        raise AssertionError("merged manifest/cache order mismatch")
    atomic_csv(manifest_path, merged_frame)
    merged["manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "rows": int(len(merged_frame)),
        "parts": [validation["manifest"] for validation in validations],
    }
    merged["extraction"] = {
        "merged_feature_shards": True,
        "merge_script_version": SCRIPT_VERSION,
        "input_caches": [
            {
                "path": str(path),
                "sha256": input_hashes[path],
                "rows": int(validation["rows"]),
                "extraction": payload.get("extraction"),
            }
            for path, payload, validation in zip(
                input_paths, payloads, validations
            )
        ],
        "observations": int(
            sum(
                int(payload.get("extraction", {}).get("observations", 0))
                for payload in payloads
            )
        ),
        "elapsed_seconds_sum": float(
            sum(
                float(
                    payload.get("extraction", {}).get(
                        "elapsed_seconds", 0.0
                    )
                )
                for payload in payloads
            )
        ),
        "durable_local_feature_cache": True,
        "training_remote_image_io": False,
    }

    total_rows = sum(int(value["rows"]) for value in validations)
    for key in row_tensor_fields:
        if int(merged[key].shape[0]) != total_rows:
            raise AssertionError(f"merged tensor {key!r} has wrong length")
    for key in (*REQUIRED_ROW_LIST_FIELDS, "query360_indices"):
        if len(merged[key]) != total_rows:
            raise AssertionError(f"merged metadata {key!r} has wrong length")

    atomic_torch(output_path, merged)
    cache_sha = sha256_file(output_path)
    audit = {
        "schema_version": "legacy360-merged-feature-cache-audit-v1",
        "script_version": SCRIPT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA,
        "extractor_script_version": reference["script_version"],
        "split": "train_core",
        "inputs": [
            {
                "path": str(path),
                "sha256": input_hashes[path],
                "rows": int(validation["rows"]),
                "manifest": validation["manifest"],
            }
            for path, validation in zip(input_paths, validations)
        ],
        "output": {
            "path": str(output_path),
            "sha256": cache_sha,
            "rows": total_rows,
            "manifest": merged["manifest"],
        },
        "compatibility": {
            "encoder_fingerprint": reference["encoder_fingerprint"],
            "sensor_names": list(SENSOR_NAMES),
            "tensor_fields": {
                key: {
                    "shape": list(merged[key].shape),
                    "dtype": str(merged[key].dtype),
                    "alias_of": alias_fields.get(key),
                }
                for key in row_tensor_fields
            },
            "row_list_fields": sorted(
                set(reference["row_list_fields"]) | {"query360_indices"}
            ),
        },
        "integrity": {
            "ids_unique": len(set(merged["ids"])) == total_rows,
            "query360_indices_unique": len(set(merged_queries)) == total_rows,
            "plumes_disjoint_between_inputs": True,
            "id_sha256": sha256_lines(merged["ids"]),
            "query360_index_sha256": sha256_lines(merged_queries),
            "query360_index_min": min(merged_queries),
            "query360_index_max": max(merged_queries),
        },
    }
    atomic_json(audit_path, audit)
    audit_sha = sha256_file(audit_path)
    atomic_text(audit_sha_path, f"{audit_sha}  {audit_path.name}\n")
    print(
        json.dumps(
            {
                "cache": str(output_path),
                "cache_sha256": cache_sha,
                "manifest": str(manifest_path),
                "audit": str(audit_path),
                "audit_sha256": audit_sha,
                "rows": total_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard = subparsers.add_parser(
        "shard-manifest",
        help="balance train extraction against dev on two GPUs",
    )
    shard.add_argument("--train-manifest", required=True)
    shard.add_argument("--dev-manifest", required=True)
    shard.add_argument("--output-dir", required=True)
    shard.add_argument("--prefix", default="legacy360_train_core")
    shard.add_argument("--shard0", default="")
    shard.add_argument("--shard1", default="")
    shard.add_argument("--audit", default="")
    shard.add_argument("--overwrite", action="store_true")
    shard.set_defaults(function=command_shard_manifest)

    merge = subparsers.add_parser(
        "merge-cache",
        help="strictly validate and concatenate train_core feature shards",
    )
    merge.add_argument(
        "--input-cache",
        action="append",
        required=True,
        help="repeat once per shard, in desired merged row order",
    )
    merge.add_argument("--output-cache", required=True)
    merge.add_argument("--output-manifest", default="")
    merge.add_argument("--audit", default="")
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(function=command_merge_cache)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
