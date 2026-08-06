#!/usr/bin/env python3
"""Lock-bound two-way extraction and merge for the legacy-360 sealed test.

This utility deliberately does *not* evaluate a model.  It has two commands:

``shard-manifest``
    Validate the sole cross-family development winner receipt, its exact
    selection lock, and its checkpoint before opening the explicitly
    authorized sealed manifest. Complete event/plume groups are assigned to
    exactly two manifests.

``merge-cache``
    Validate the same immutable selection artifacts, the shard plan, and two
    extractor-produced ``split=test`` caches, then emit one sealed feature
    cache.  No prediction, metric, threshold selection, or threshold search
    exists in this module.

There is intentionally no overwrite or retry flag.  Exclusive claim files
make both the manifest opening and the cache merge auditable one-time actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from shard_and_merge_feature_cache import (
    FEATURE_SCHEMA,
    OPTIONAL_ROW_LIST_FIELDS,
    REQUIRED_ROW_LIST_FIELDS,
    SENSOR_NAMES,
    TENSOR_ALIASES,
    _load_torch,
    _row_list_fields,
    _tensor_fields,
    _tensor_is_finite,
    _tensor_storage_signature,
    _tensors_equal,
    atomic_csv,
    atomic_json,
    atomic_torch,
    clean,
    object_fingerprint,
    public_manifest,
    read_manifest,
    refuse_existing,
    sha256_file,
    sha256_lines,
)


SCRIPT_VERSION = "legacy360-sealed-feature-shards-v1"
PLAN_SCHEMA = "legacy360-sealed-feature-shard-plan-v1"
CLAIM_SCHEMA = "legacy360-sealed-feature-manifest-claim-v1"
MERGE_USE_SCHEMA = "legacy360-sealed-feature-merge-use-v1"
MERGE_AUDIT_SCHEMA = "legacy360-sealed-feature-merge-audit-v1"
MERGE_RECEIPT_SCHEMA = "legacy360-sealed-feature-merge-receipt-v1"
MASTER_SELECTION_SCHEMA = (
    "legacy360-full-cache-master-selection-receipt-v1"
)
SOURCE_POSITION_COLUMN = "__sealed_source_position"

LOCK_TO_CHECKPOINT_SCHEMA = {
    "query360-two-axis-selection-lock-v1": "query360-two-axis-head-v1",
    "query360-gated-delta-selection-lock-v1": "query360-gated-delta-head-v1",
}
GATED_LOCK_SCHEMA = "query360-gated-delta-selection-lock-v1"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _absolute(value: str | Path) -> Path:
    return Path(value).expanduser().absolute()


def _json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} is unavailable: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{role} is not a JSON object: {path}")
    return payload


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a durable JSON claim without a check-then-create race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
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
    except Exception:
        # The claim is deliberately retained if writing may have become
        # visible.  An operator must inspect it rather than silently retry.
        raise


def _master_manifest_claim_path(master_receipt_path: Path) -> Path:
    return master_receipt_path.with_suffix(
        master_receipt_path.suffix + ".sealed_feature_claim.json"
    )


def _merge_use_path(master_receipt_path: Path) -> Path:
    return master_receipt_path.with_suffix(
        master_receipt_path.suffix + ".sealed_feature_merge_use.json"
    )


def _cache_use_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(
        cache_path.suffix + ".sealed_feature_merge_use.json"
    )


def _require_unopened_selection_state(
    *,
    lock_path: Path,
    lock: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a completed dev-only selection before any sealed path is read."""

    summary_path = lock_path.parent / "summary.json"
    status_path = lock_path.parent / "run_status.json"
    summary = _json(summary_path, "selection summary")
    status = _json(status_path, "selection run status")
    if summary.get("status") != "complete":
        raise ValueError("selection summary is not complete")
    if status.get("status") != "complete":
        raise ValueError("selection run status is not complete")
    for role, payload in (("summary", summary), ("run status", status)):
        if payload.get("sealed_test_read") is not False:
            raise PermissionError(f"selection {role} is not sealed-test clean")
        if int(payload.get("sealed_test_evaluations", -1)) != 0:
            raise PermissionError(
                f"selection {role} reports a sealed-test evaluation"
            )
    if lock.get("test_cache_read_before_lock") is not False:
        raise PermissionError(
            "selection lock does not attest test_cache_read_before_lock=false"
        )
    if summary.get("selection_lock") != lock:
        raise ValueError("summary's embedded selection lock differs")
    for forbidden in (
        lock_path.parent / "sealed_test_result.json",
        lock_path.parent / "sealed_test_predictions.csv",
        lock_path.parent / "locked_eval_status.json",
    ):
        if forbidden.exists():
            raise PermissionError(
                "selection directory already contains sealed evaluation "
                f"artifacts: {forbidden}"
            )
    return summary, status


def _validate_selection_artifacts(
    *,
    selection_lock_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Validate lock/checkpoint/status without touching a sealed manifest."""

    lock = _json(selection_lock_path, "selection lock")
    lock_schema = str(lock.get("schema_version", ""))
    checkpoint_schema = LOCK_TO_CHECKPOINT_SCHEMA.get(lock_schema)
    if checkpoint_schema is None:
        raise ValueError(f"unsupported selection lock schema: {lock_schema!r}")

    _require_unopened_selection_state(lock_path=selection_lock_path, lock=lock)
    locked_checkpoint_path = _absolute(str(lock.get("checkpoint", "")))
    if locked_checkpoint_path != checkpoint_path:
        raise ValueError(
            "provided checkpoint path differs from the path in selection lock"
        )
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != lock.get("checkpoint_sha256"):
        raise ValueError("checkpoint SHA256 differs from selection lock")
    checkpoint = _load_torch(checkpoint_path)
    if checkpoint.get("schema_version") != checkpoint_schema:
        raise ValueError("checkpoint schema differs from selection lock family")

    if int(checkpoint.get("epoch", -1)) != int(lock.get("best_epoch", -2)):
        raise ValueError("checkpoint epoch differs from selection lock")
    for key in ("arm", "base_mode", "model_config"):
        if checkpoint.get(key) != lock.get(key):
            raise ValueError(f"checkpoint {key} differs from selection lock")
    if not clean(lock.get("arm")) or not clean(lock.get("base_mode")):
        raise ValueError("selection lock has a blank arm/base_mode")
    if not isinstance(lock.get("model_config"), Mapping):
        raise ValueError("selection lock has no model_config")
    if not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError("checkpoint has no model state")
    if not isinstance(checkpoint.get("parameter_signature"), Mapping):
        raise ValueError("checkpoint has no parameter signature")

    encoder = lock.get("encoder")
    if not isinstance(encoder, Mapping):
        raise ValueError("selection lock has no encoder provenance")
    state_sha = str(encoder.get("state_sha256", "")).lower()
    if HEX_SHA256.fullmatch(state_sha) is None:
        raise ValueError(
            "selection lock encoder.state_sha256 is not a SHA256 digest"
        )
    checkpoint_encoder = checkpoint.get("encoder")
    if checkpoint_encoder is not None and checkpoint_encoder != encoder:
        raise ValueError("checkpoint encoder differs from selection lock")
    if lock_schema == GATED_LOCK_SCHEMA and checkpoint_encoder != encoder:
        raise ValueError("gated-delta checkpoint has no locked encoder binding")
    encoder_binding = (
        "checkpoint_and_selection_lock"
        if checkpoint_encoder == encoder
        else "sha_locked_legacy_checkpoint_plus_selection_lock"
    )

    if lock_schema == GATED_LOCK_SCHEMA:
        if checkpoint.get("base_contract") != lock.get("base_contract"):
            raise ValueError(
                "checkpoint base_contract differs from selection lock"
            )
    locked_threshold = float(lock.get("locked_threshold", math.nan))
    checkpoint_threshold = float(
        checkpoint.get("locked_threshold_candidate", math.nan)
    )
    if (
        not math.isfinite(locked_threshold)
        or not 0.0 <= locked_threshold <= 1.0
        or locked_threshold != checkpoint_threshold
    ):
        raise ValueError("checkpoint threshold differs from selection lock")
    selection_metric = str(lock.get("selection_metric", ""))
    checkpoint_dev = checkpoint.get("dev")
    if (
        not selection_metric
        or not isinstance(checkpoint_dev, Mapping)
        or selection_metric not in checkpoint_dev
        or float(lock.get("selection_score", math.nan))
        != float(checkpoint_dev[selection_metric])
    ):
        raise ValueError("checkpoint selection score differs from lock")

    return {
        "lock": lock,
        "checkpoint": checkpoint,
        "selection_lock_path": str(selection_lock_path),
        "selection_lock_sha256": sha256_file(selection_lock_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "lock_schema": lock_schema,
        "checkpoint_schema": checkpoint_schema,
        "arm": str(lock["arm"]),
        "base_mode": str(lock["base_mode"]),
        "model_config_fingerprint": object_fingerprint(lock["model_config"]),
        "encoder": dict(encoder),
        "encoder_fingerprint": object_fingerprint(encoder),
        "encoder_binding": encoder_binding,
    }


def _validate_master_selection_receipt(
    *,
    master_receipt_path: Path,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the family-local lock to the sole cross-family dev winner."""

    if master_receipt_path.is_symlink():
        raise ValueError("master selection receipt cannot be a symlink")
    receipt = _json(master_receipt_path, "master selection receipt")
    receipt_sha = sha256_file(master_receipt_path)
    sidecar = master_receipt_path.with_suffix(
        master_receipt_path.suffix + ".sha256"
    )
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"master selection receipt SHA sidecar is missing: {sidecar}"
        )
    tokens = sidecar.read_text(encoding="utf-8").split()
    if (
        len(tokens) < 2
        or tokens[0] != receipt_sha
        or tokens[1] != master_receipt_path.name
    ):
        raise ValueError("master selection receipt SHA sidecar mismatch")
    if (
        receipt.get("schema_version") != MASTER_SELECTION_SCHEMA
        or receipt.get("status") != "complete"
    ):
        raise ValueError("master selection receipt is not complete/supported")
    if (
        receipt.get("sealed_test_read") is not False
        or int(receipt.get("sealed_test_evaluations", -1)) != 0
    ):
        raise PermissionError("master selection receipt is not dev-only")
    winner = receipt.get("winner")
    dispatch = receipt.get("locked_dispatch")
    common = receipt.get("common_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (winner, dispatch, common)
    ):
        raise ValueError("master selection receipt is malformed")
    assert isinstance(winner, Mapping)
    assert isinstance(dispatch, Mapping)
    assert isinstance(common, Mapping)
    family_to_lock = {
        "gated_delta": GATED_LOCK_SCHEMA,
        "compact_axial": "query360-two-axis-selection-lock-v1",
    }
    family = str(winner.get("family", ""))
    if family_to_lock.get(family) != artifacts["lock_schema"]:
        raise ValueError("master winner family differs from selection lock")
    winner_lock = winner.get("selection_lock")
    winner_checkpoint = winner.get("checkpoint")
    if not isinstance(winner_lock, Mapping) or not isinstance(
        winner_checkpoint, Mapping
    ):
        raise ValueError("master winner lacks lock/checkpoint binding")
    expected_pairs = (
        (
            _absolute(str(winner_lock.get("path", ""))),
            _absolute(artifacts["selection_lock_path"]),
            "selection lock path",
        ),
        (
            str(winner_lock.get("sha256", "")),
            artifacts["selection_lock_sha256"],
            "selection lock SHA256",
        ),
        (
            _absolute(str(winner_checkpoint.get("path", ""))),
            _absolute(artifacts["checkpoint_path"]),
            "checkpoint path",
        ),
        (
            str(winner_checkpoint.get("sha256", "")),
            artifacts["checkpoint_sha256"],
            "checkpoint SHA256",
        ),
        (
            str(winner.get("lock_schema", "")),
            artifacts["lock_schema"],
            "lock schema",
        ),
        (
            str(winner.get("checkpoint_schema", "")),
            artifacts["checkpoint_schema"],
            "checkpoint schema",
        ),
        (str(winner.get("arm", "")), artifacts["arm"], "arm"),
        (
            str(winner.get("base_mode", "")),
            artifacts["base_mode"],
            "base mode",
        ),
    )
    for observed, expected, role in expected_pairs:
        if observed != expected:
            raise ValueError(f"master winner {role} mismatch")
    lock = artifacts["lock"]
    for key in ("best_epoch", "selection_score", "locked_threshold"):
        if float(winner.get(key, math.nan)) != float(lock.get(key, math.nan)):
            raise ValueError(f"master winner {key} mismatch")
    if common.get("encoder") != artifacts["encoder"]:
        raise ValueError("master common encoder differs from locked encoder")
    if (
        common.get("encoder_fingerprint")
        != artifacts["encoder_fingerprint"]
    ):
        raise ValueError("master encoder fingerprint differs")
    if (
        dispatch.get("selection_lock") != winner_lock
        or dispatch.get("checkpoint") != winner_checkpoint
        or dispatch.get("evaluator_family") != family
    ):
        raise ValueError("master locked dispatch differs from winner")
    return {
        "path": str(master_receipt_path),
        "sha256": receipt_sha,
        "sidecar": str(sidecar),
        "family": family,
        "winner_run": str(winner.get("run", "")),
        "receipt": receipt,
    }


def _normalise_manifest_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _manifest_fingerprint(frame: pd.DataFrame) -> str:
    """Order-independent, column-sensitive fingerprint of public rows."""

    visible = public_manifest(frame).drop(
        columns=[SOURCE_POSITION_COLUMN], errors="ignore"
    )
    columns = sorted(str(column) for column in visible.columns)
    records = [
        {
            column: _normalise_manifest_value(row[column])
            for column in columns
        }
        for _index, row in visible.iterrows()
    ]
    records.sort(
        key=lambda row: (
            int(row["query360_index"]),
            row["id"],
            json.dumps(row, sort_keys=True, separators=(",", ":")),
        )
    )
    return object_fingerprint({"columns": columns, "records": records})


def _extract_argv(
    *,
    python: str,
    extractor: Path,
    manifest: Path,
    output_cache: Path,
    device: str,
    encoder: Mapping[str, Any],
    raw_cache_dir: str,
    raw_cache_readonly_fallback: bool,
    row_batch_size: int,
    encoder_microbatch: int,
    num_workers: int,
    prefetch_factor: int,
) -> list[str]:
    argv = [
        python,
        str(extractor),
        "extract",
        "--manifest",
        str(manifest),
        "--split",
        "test",
        "--sealed-test",
        "--output-cache",
        str(output_cache),
        "--device",
        device,
        "--row-batch-size",
        str(row_batch_size),
        "--encoder-microbatch",
        str(encoder_microbatch),
        "--num-workers",
        str(num_workers),
        "--prefetch-factor",
        str(prefetch_factor),
    ]
    default_weights = clean(encoder.get("default_weights"))
    if default_weights:
        argv.extend(["--weights", default_weights])
    sensor_weights = encoder.get("sensor_weights")
    if isinstance(sensor_weights, Mapping) and sensor_weights:
        encoded = ",".join(
            f"{key}={sensor_weights[key]}" for key in sorted(sensor_weights)
        )
        argv.extend(["--sensor-weights", encoded])
    if raw_cache_dir:
        argv.extend(["--raw-cache-dir", raw_cache_dir])
        if raw_cache_readonly_fallback:
            argv.append("--raw-cache-readonly-fallback")
    return argv


def command_shard_manifest(args: argparse.Namespace) -> None:
    if not args.sealed_test:
        raise PermissionError(
            "refusing sealed-test sharding without explicit --sealed-test"
        )
    selection_lock_path = _absolute(args.selection_lock)
    checkpoint_path = _absolute(args.checkpoint)
    artifacts = _validate_selection_artifacts(
        selection_lock_path=selection_lock_path,
        checkpoint_path=checkpoint_path,
    )
    master_receipt_path = _absolute(args.master_selection_receipt)
    master = _validate_master_selection_receipt(
        master_receipt_path=master_receipt_path,
        artifacts=artifacts,
    )

    manifest_path = _absolute(args.manifest)
    output_dir = _absolute(args.output_dir)
    prefix = str(args.prefix)
    shard_paths = [
        output_dir / f"{prefix}_gpu{index}.csv" for index in (0, 1)
    ]
    cache_paths = [
        output_dir / f"{prefix}_gpu{index}.features.pt" for index in (0, 1)
    ]
    plan_path = (
        _absolute(args.plan)
        if args.plan
        else output_dir / f"{prefix}_plan.json"
    )
    claim_path = _master_manifest_claim_path(master_receipt_path)
    extractor_path = _absolute(args.extractor)
    if not extractor_path.is_file():
        raise FileNotFoundError(f"extractor is unavailable: {extractor_path}")
    if not clean(args.python):
        raise ValueError("--python cannot be blank")
    for name in (
        "row_batch_size",
        "encoder_microbatch",
        "num_workers",
        "prefetch_factor",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.raw_cache_readonly_fallback and not args.raw_cache_dir:
        raise ValueError(
            "--raw-cache-readonly-fallback requires --raw-cache-dir"
        )
    planned_paths = [*shard_paths, *cache_paths, plan_path]
    if len(set(planned_paths)) != len(planned_paths):
        raise ValueError("sealed shard output paths must be distinct")
    refuse_existing(
        (*shard_paths, *cache_paths, plan_path, claim_path),
        overwrite=False,
    )

    initial_claim = {
        "schema_version": CLAIM_SCHEMA,
        "script_version": SCRIPT_VERSION,
        "status": "opening_manifest",
        "sealed_test_authorized": True,
        "selection_lock_path": str(selection_lock_path),
        "selection_lock_sha256": artifacts["selection_lock_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": artifacts["checkpoint_sha256"],
        "master_selection_receipt_path": str(master_receipt_path),
        "master_selection_receipt_sha256": master["sha256"],
        "master_winner_run": master["winner_run"],
        "master_winner_family": master["family"],
        "manifest_path": str(manifest_path),
        "plan_path": str(plan_path),
        "metrics_computed": False,
        "threshold_search_performed": False,
    }
    _exclusive_json(claim_path, initial_claim)

    try:
        frame = read_manifest(manifest_path, "sealed test")
        if SOURCE_POSITION_COLUMN in frame.columns:
            raise ValueError(
                f"sealed manifest already contains reserved column "
                f"{SOURCE_POSITION_COLUMN!r}"
            )
        group_column = (
            "event_id" if "event_id" in frame.columns else "plume_id"
        )
        groups = frame[group_column].map(clean)
        if groups.eq("").any():
            raise ValueError(f"sealed test.{group_column} contains blanks")
        frame[group_column] = groups
        if group_column == "event_id":
            events_per_plume = (
                frame.groupby("plume_id", observed=True)["event_id"].nunique()
            )
            if bool(events_per_plume.gt(1).any()):
                raise ValueError("a plume_id maps to multiple event_id groups")

        group_costs = (
            frame.groupby(group_column, sort=False, observed=True)[
                "_estimated_cost"
            ]
            .sum()
            .astype(np.int64)
        )
        ranked_groups = sorted(
            (
                (str(group), int(cost))
                for group, cost in group_costs.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if len(ranked_groups) < 2:
            raise ValueError(
                "sealed manifest needs at least two complete groups"
            )
        loads = [0, 0]
        assignment: dict[str, int] = {}
        for group, cost in ranked_groups:
            target = 0 if loads[0] <= loads[1] else 1
            assignment[group] = target
            loads[target] += cost
        shard_index = frame[group_column].map(assignment)
        if shard_index.isna().any():
            raise AssertionError("group scheduler missed sealed rows")

        frame[SOURCE_POSITION_COLUMN] = frame["_source_position"].astype(
            np.int64
        )
        shards = [
            frame.loc[shard_index.eq(index)]
            .sort_values("_source_position", kind="stable")
            .copy()
            for index in (0, 1)
        ]
        if any(shard.empty for shard in shards):
            raise ValueError("group scheduler produced an empty shard")
        for column in ("plume_id", group_column):
            if set(shards[0][column]) & set(shards[1][column]):
                raise AssertionError(f"{column} crossed sealed shards")
        if (
            set(shards[0]["id"]) & set(shards[1]["id"])
            or set(
                shards[0]["query360_index"].astype(np.int64).tolist()
            )
            & set(shards[1]["query360_index"].astype(np.int64).tolist())
        ):
            raise AssertionError("sealed ID/query universe crossed shards")

        for path, shard in zip(shard_paths, shards):
            atomic_csv(path, public_manifest(shard))

        commands = [
            _extract_argv(
                python=str(args.python),
                extractor=extractor_path,
                manifest=shard_paths[index],
                output_cache=cache_paths[index],
                device=f"cuda:{index}",
                encoder=artifacts["encoder"],
                raw_cache_dir=str(args.raw_cache_dir),
                raw_cache_readonly_fallback=bool(
                    args.raw_cache_readonly_fallback
                ),
                row_batch_size=int(args.row_batch_size),
                encoder_microbatch=int(args.encoder_microbatch),
                num_workers=int(args.num_workers),
                prefetch_factor=int(args.prefetch_factor),
            )
            for index in (0, 1)
        ]
        plan = {
            "schema_version": PLAN_SCHEMA,
            "script_version": SCRIPT_VERSION,
            "sealed_test_authorized": True,
            "selection": {
                "master_selection_receipt_path": str(master_receipt_path),
                "master_selection_receipt_sha256": master["sha256"],
                "master_winner_run": master["winner_run"],
                "master_winner_family": master["family"],
                "selection_lock_path": str(selection_lock_path),
                "selection_lock_sha256": artifacts[
                    "selection_lock_sha256"
                ],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": artifacts["checkpoint_sha256"],
                "lock_schema": artifacts["lock_schema"],
                "checkpoint_schema": artifacts["checkpoint_schema"],
                "arm": artifacts["arm"],
                "base_mode": artifacts["base_mode"],
                "model_config_fingerprint": artifacts[
                    "model_config_fingerprint"
                ],
                "encoder": artifacts["encoder"],
                "encoder_fingerprint": artifacts["encoder_fingerprint"],
                "encoder_binding": artifacts["encoder_binding"],
            },
            "source_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "rows": int(len(frame)),
                "columns": sorted(
                    str(column)
                    for column in public_manifest(frame)
                    .drop(
                        columns=[SOURCE_POSITION_COLUMN],
                        errors="ignore",
                    )
                    .columns
                ),
                "content_fingerprint": _manifest_fingerprint(frame),
                "id_universe_sha256": sha256_lines(
                    sorted(frame["id"].astype(str).tolist())
                ),
                "query_universe_sha256": sha256_lines(
                    sorted(
                        frame["query360_index"]
                        .astype(np.int64)
                        .tolist()
                    )
                ),
            },
            "scheduler": {
                "algorithm": (
                    "deterministic largest-complete-group-first greedy"
                ),
                "group_column": group_column,
                "shards": 2,
            },
            "shards": [
                {
                    "index": index,
                    "manifest_path": str(shard_paths[index]),
                    "manifest_sha256": sha256_file(shard_paths[index]),
                    "rows": int(len(shards[index])),
                    "groups": int(shards[index][group_column].nunique()),
                    "plumes": int(shards[index]["plume_id"].nunique()),
                    "estimated_cost": int(
                        shards[index]["_estimated_cost"].sum()
                    ),
                    "expected_cache_path": str(cache_paths[index]),
                    "extract_argv": commands[index],
                    "sealed_test_flag_present": (
                        "--sealed-test" in commands[index]
                    ),
                }
                for index in (0, 1)
            ],
            "integrity": {
                "rows_preserved": int(sum(len(shard) for shard in shards))
                == int(len(frame)),
                "event_or_plume_groups_disjoint": True,
                "plumes_disjoint": True,
                "ids_disjoint": True,
                "query_indices_disjoint": True,
                "manifest_union_fingerprint_recorded": True,
            },
            "metrics_computed": False,
            "threshold_search_performed": False,
        }
        atomic_json(plan_path, plan)
        final_claim = {
            **initial_claim,
            "status": "planned",
            "manifest_sha256": plan["source_manifest"]["sha256"],
            "manifest_rows": int(len(frame)),
            "plan_sha256": sha256_file(plan_path),
            "shard_manifest_sha256": [
                shard["manifest_sha256"] for shard in plan["shards"]
            ],
        }
        atomic_json(claim_path, final_claim)
        print(
            json.dumps(
                {
                    "plan": str(plan_path),
                    "plan_sha256": sha256_file(plan_path),
                    "claim": str(claim_path),
                    "shards": [str(path) for path in shard_paths],
                    "expected_caches": [str(path) for path in cache_paths],
                    "metrics_computed": False,
                    "threshold_search_performed": False,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as error:
        atomic_json(
            claim_path,
            {
                **initial_claim,
                "status": "failed_after_manifest_claim",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def _validate_test_cache(
    *,
    cache_path: Path,
    payload: Mapping[str, Any],
    expected_shard: Mapping[str, Any],
    locked_encoder: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_version") != FEATURE_SCHEMA:
        raise ValueError(f"{cache_path}: unsupported feature cache schema")
    if payload.get("split") != "test":
        raise ValueError(f"{cache_path}: cache split is not test")
    if payload.get("sealed_test_read") is not True:
        raise ValueError(f"{cache_path}: sealed_test_read is not true")
    extraction = payload.get("extraction")
    if (
        not isinstance(extraction, Mapping)
        or extraction.get("sealed_test_authorized") is not True
    ):
        raise ValueError(
            f"{cache_path}: extraction lacks sealed-test authorization"
        )
    if list(payload.get("sensor_names", [])) != list(SENSOR_NAMES):
        raise ValueError(f"{cache_path}: wrong or reordered sensor_names")
    if payload.get("encoder") != locked_encoder:
        raise ValueError(f"{cache_path}: encoder differs from selection lock")
    if not clean(payload.get("script_version")):
        raise ValueError(f"{cache_path}: blank extractor script_version")

    tensor_fields = _tensor_fields(payload)
    if not {"features", "valid_mask", "labels"}.issubset(tensor_fields):
        raise ValueError(f"{cache_path}: required row tensors are missing")
    labels = payload["labels"]
    if labels.ndim != 1 or int(labels.shape[0]) < 1:
        raise ValueError(f"{cache_path}: labels are malformed")
    rows = int(labels.shape[0])
    checked_float_storages: set[tuple[Any, ...]] = set()
    for key in sorted(tensor_fields):
        tensor = payload[key]
        if tensor.ndim < 1 or int(tensor.shape[0]) != rows:
            raise ValueError(
                f"{cache_path}: tensor {key!r} is not row-aligned"
            )
        if tensor.is_floating_point():
            signature = _tensor_storage_signature(tensor)
            if signature not in checked_float_storages:
                if not _tensor_is_finite(tensor):
                    raise ValueError(
                        f"{cache_path}: tensor {key!r} is non-finite"
                    )
                checked_float_storages.add(signature)
    row_list_fields = _row_list_fields(payload)
    if "event_ids" not in row_list_fields:
        raise ValueError(f"{cache_path}: event_ids metadata is required")
    for key in sorted(row_list_fields):
        value = payload[key]
        if not isinstance(value, (list, tuple)) or len(value) != rows:
            raise ValueError(
                f"{cache_path}: row metadata {key!r} is malformed"
            )

    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{cache_path}: manifest metadata is absent")
    expected_manifest_path = _absolute(expected_shard["manifest_path"])
    observed_manifest_path = _absolute(str(manifest.get("path", "")))
    if observed_manifest_path != expected_manifest_path:
        raise ValueError(f"{cache_path}: cache names the wrong shard manifest")
    expected_manifest_sha = str(expected_shard["manifest_sha256"])
    actual_manifest_sha = sha256_file(expected_manifest_path)
    if (
        manifest.get("sha256") != expected_manifest_sha
        or actual_manifest_sha != expected_manifest_sha
    ):
        raise ValueError(f"{cache_path}: shard manifest SHA256 mismatch")
    if int(manifest.get("rows", -1)) != rows or int(
        expected_shard["rows"]
    ) != rows:
        raise ValueError(f"{cache_path}: shard manifest row count mismatch")

    frame = read_manifest(
        expected_manifest_path, f"sealed cache {cache_path.name}"
    )
    if len(frame) != rows:
        raise ValueError(f"{cache_path}: shard manifest length mismatch")
    expected_lists = {
        "ids": frame["id"].astype(str).tolist(),
        "plume_ids": frame["plume_id"].astype(str).tolist(),
        "availability_signatures": frame[
            "availability_signature"
        ].astype(str).tolist(),
    }
    for key, expected in expected_lists.items():
        if [str(value) for value in payload[key]] != expected:
            raise ValueError(
                f"{cache_path}: cached {key} differ from shard manifest"
            )
    if payload["labels"].detach().cpu().long().tolist() != frame[
        "label"
    ].astype(np.int64).tolist():
        raise ValueError(
            f"{cache_path}: cached labels differ from shard manifest"
        )
    expected_events = (
        frame["event_id"].astype(str).tolist()
        if "event_id" in frame.columns
        else frame["plume_id"].astype(str).tolist()
    )
    if "event_ids" in payload and [
        str(value) for value in payload["event_ids"]
    ] != expected_events:
        raise ValueError(
            f"{cache_path}: cached event_ids differ from shard manifest"
        )
    queries = frame["query360_index"].astype(np.int64).tolist()
    if "query360_indices" in payload and [
        int(value) for value in payload["query360_indices"]
    ] != queries:
        raise ValueError(
            f"{cache_path}: cached query indices differ from shard manifest"
        )
    if SOURCE_POSITION_COLUMN not in frame.columns:
        raise ValueError(
            f"{cache_path}: shard manifest has no source-position seal"
        )
    positions = pd.to_numeric(
        frame[SOURCE_POSITION_COLUMN], errors="raise"
    ).astype(np.int64)
    if positions.duplicated().any() or bool(positions.lt(0).any()):
        raise ValueError(
            f"{cache_path}: invalid sealed source positions"
        )
    return {
        "rows": rows,
        "tensor_fields": tensor_fields,
        "row_list_fields": row_list_fields,
        "frame": frame,
        "queries": queries,
        "positions": positions.tolist(),
        "script_version": str(payload["script_version"]),
        "base_definitions_fingerprint": object_fingerprint(
            payload.get("base_definitions")
        ),
    }


def _assert_pair_compatibility(
    *,
    cache_paths: Sequence[Path],
    payloads: Sequence[Mapping[str, Any]],
    validations: Sequence[Mapping[str, Any]],
) -> None:
    first = validations[0]
    for path, validation in zip(cache_paths[1:], validations[1:]):
        for key in (
            "tensor_fields",
            "row_list_fields",
            "script_version",
            "base_definitions_fingerprint",
        ):
            if validation[key] != first[key]:
                raise ValueError(f"{path}: {key} differs between test shards")
    for key in sorted(first["tensor_fields"]):
        reference = payloads[0][key]
        for path, payload in zip(cache_paths[1:], payloads[1:]):
            candidate = payload[key]
            if (
                candidate.dtype != reference.dtype
                or candidate.shape[1:] != reference.shape[1:]
            ):
                raise ValueError(
                    f"{path}: tensor {key!r} shape/dtype differs"
                )

    id_sets: list[set[str]] = []
    plume_sets: list[set[str]] = []
    event_sets: list[set[str]] = []
    query_sets: list[set[int]] = []
    position_sets: list[set[int]] = []
    for path, payload, validation in zip(
        cache_paths, payloads, validations
    ):
        ids = [str(value) for value in payload["ids"]]
        plumes = [str(value) for value in payload["plume_ids"]]
        events = [
            str(value)
            for value in payload.get("event_ids", plumes)
        ]
        queries = [int(value) for value in validation["queries"]]
        positions = [int(value) for value in validation["positions"]]
        for name, values in (
            ("IDs", ids),
            ("query indices", queries),
            ("source positions", positions),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{path}: duplicate {name} inside cache")
        id_sets.append(set(ids))
        plume_sets.append(set(plumes))
        event_sets.append(set(events))
        query_sets.append(set(queries))
        position_sets.append(set(positions))
    for left, right in ((0, 1),):
        for name, sets in (
            ("ID", id_sets),
            ("plume", plume_sets),
            ("event", event_sets),
            ("query360_index", query_sets),
            ("source position", position_sets),
        ):
            if sets[left] & sets[right]:
                raise ValueError(
                    f"{name} overlap between the two sealed caches"
                )


def command_merge_cache(args: argparse.Namespace) -> None:
    if not args.sealed_test:
        raise PermissionError(
            "refusing sealed-test merge without explicit --sealed-test"
        )
    selection_lock_path = _absolute(args.selection_lock)
    checkpoint_path = _absolute(args.checkpoint)
    artifacts = _validate_selection_artifacts(
        selection_lock_path=selection_lock_path,
        checkpoint_path=checkpoint_path,
    )
    master_receipt_path = _absolute(args.master_selection_receipt)
    master = _validate_master_selection_receipt(
        master_receipt_path=master_receipt_path,
        artifacts=artifacts,
    )

    plan_path = _absolute(args.plan)
    claim_path = _master_manifest_claim_path(master_receipt_path)
    claim = _json(claim_path, "sealed manifest claim")
    if claim.get("schema_version") != CLAIM_SCHEMA:
        raise ValueError("unsupported sealed manifest claim")
    if claim.get("status") != "planned":
        raise PermissionError("sealed manifest claim is not in planned state")
    if (
        claim.get("selection_lock_sha256")
        != artifacts["selection_lock_sha256"]
        or claim.get("checkpoint_sha256")
        != artifacts["checkpoint_sha256"]
        or claim.get("master_selection_receipt_sha256")
        != master["sha256"]
        or _absolute(str(claim.get("plan_path", ""))) != plan_path
        or claim.get("plan_sha256") != sha256_file(plan_path)
    ):
        raise ValueError("sealed manifest claim differs from lock/plan")

    plan = _json(plan_path, "sealed shard plan")
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported sealed shard plan")
    if plan.get("sealed_test_authorized") is not True:
        raise PermissionError("sealed shard plan lacks explicit authorization")
    selection = plan.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("sealed shard plan has no selection binding")
    for key in (
        "selection_lock_sha256",
        "checkpoint_sha256",
        "arm",
        "base_mode",
        "model_config_fingerprint",
        "encoder_fingerprint",
    ):
        if selection.get(key) != artifacts[key]:
            raise ValueError(f"sealed shard plan selection.{key} differs")
    if selection.get("encoder") != artifacts["encoder"]:
        raise ValueError("sealed shard plan encoder differs from lock")
    if (
        selection.get("master_selection_receipt_path")
        != str(master_receipt_path)
        or selection.get("master_selection_receipt_sha256")
        != master["sha256"]
        or selection.get("master_winner_run") != master["winner_run"]
        or selection.get("master_winner_family") != master["family"]
    ):
        raise ValueError("sealed shard plan differs from master winner")
    if (
        plan.get("metrics_computed") is not False
        or plan.get("threshold_search_performed") is not False
    ):
        raise ValueError("sealed shard plan violates the metric guardrail")

    shards = plan.get("shards")
    if (
        not isinstance(shards, list)
        or len(shards) != 2
        or sorted(int(item.get("index", -1)) for item in shards) != [0, 1]
    ):
        raise ValueError("sealed shard plan must name exactly two shards")
    for shard in shards:
        argv = shard.get("extract_argv")
        if (
            shard.get("sealed_test_flag_present") is not True
            or not isinstance(argv, list)
            or "--sealed-test" not in argv
            or "--overwrite" in argv
            or "--split" not in argv
            or argv[argv.index("--split") + 1] != "test"
        ):
            raise ValueError(
                "shard extraction command violates sealed-test guardrails"
            )
    shards_by_cache = {
        _absolute(str(shard["expected_cache_path"])): shard
        for shard in shards
    }
    if len(shards_by_cache) != 2:
        raise ValueError("sealed shard plan has duplicate cache paths")
    input_paths = [_absolute(value) for value in args.input_cache]
    if len(input_paths) != 2 or len(set(input_paths)) != 2:
        raise ValueError(
            "merge-cache requires exactly two distinct --input-cache values"
        )
    if set(input_paths) != set(shards_by_cache):
        raise ValueError(
            "input caches differ from the two paths committed in the plan"
        )
    input_paths.sort(key=lambda path: int(shards_by_cache[path]["index"]))

    output_path = _absolute(args.output_cache)
    manifest_path = (
        _absolute(args.output_manifest)
        if args.output_manifest
        else output_path.with_suffix(output_path.suffix + ".manifest.csv")
    )
    audit_path = (
        _absolute(args.audit)
        if args.audit
        else output_path.with_suffix(output_path.suffix + ".audit.json")
    )
    receipt_path = (
        _absolute(args.receipt)
        if args.receipt
        else output_path.with_suffix(output_path.suffix + ".receipt.json")
    )
    merge_use_path = _merge_use_path(master_receipt_path)
    cache_use_paths = [_cache_use_path(path) for path in input_paths]
    produced_paths = [
        output_path,
        manifest_path,
        audit_path,
        receipt_path,
        merge_use_path,
        *cache_use_paths,
    ]
    if len(set(produced_paths)) != len(produced_paths):
        raise ValueError("sealed merge output/use paths must be distinct")
    refuse_existing(
        (
            output_path,
            manifest_path,
            audit_path,
            receipt_path,
            merge_use_path,
            *cache_use_paths,
        ),
        overwrite=False,
    )
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path == output_path:
            raise ValueError("an input cache cannot be the output cache")

    use_record = {
        "schema_version": MERGE_USE_SCHEMA,
        "script_version": SCRIPT_VERSION,
        "status": "merging",
        "sealed_test_authorized": True,
        "selection_lock_path": str(selection_lock_path),
        "selection_lock_sha256": artifacts["selection_lock_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": artifacts["checkpoint_sha256"],
        "master_selection_receipt_path": str(master_receipt_path),
        "master_selection_receipt_sha256": master["sha256"],
        "master_winner_run": master["winner_run"],
        "master_winner_family": master["family"],
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "input_caches": [str(path) for path in input_paths],
        "output_cache": str(output_path),
        "metrics_computed": False,
        "threshold_search_performed": False,
    }
    _exclusive_json(merge_use_path, use_record)
    for path, use_path in zip(input_paths, cache_use_paths):
        _exclusive_json(
            use_path,
            {
                **use_record,
                "cache_path": str(path),
                "global_merge_use_path": str(merge_use_path),
            },
        )

    try:
        input_hashes = {path: sha256_file(path) for path in input_paths}
        payloads = [_load_torch(path) for path in input_paths]
        validations = [
            _validate_test_cache(
                cache_path=path,
                payload=payload,
                expected_shard=shards_by_cache[path],
                locked_encoder=artifacts["encoder"],
            )
            for path, payload in zip(input_paths, payloads)
        ]
        _assert_pair_compatibility(
            cache_paths=input_paths,
            payloads=payloads,
            validations=validations,
        )

        combined_frame = pd.concat(
            [validation["frame"] for validation in validations],
            ignore_index=True,
            sort=False,
        )
        source = plan.get("source_manifest")
        if not isinstance(source, Mapping):
            raise ValueError("sealed shard plan has no source manifest")
        total_rows = int(sum(item["rows"] for item in validations))
        positions = pd.to_numeric(
            combined_frame[SOURCE_POSITION_COLUMN], errors="raise"
        ).astype(np.int64)
        if (
            total_rows != int(source.get("rows", -1))
            or sorted(positions.tolist()) != list(range(total_rows))
            or _manifest_fingerprint(combined_frame)
            != source.get("content_fingerprint")
            or sha256_lines(
                sorted(combined_frame["id"].astype(str).tolist())
            )
            != source.get("id_universe_sha256")
            or sha256_lines(
                sorted(
                    combined_frame["query360_index"]
                    .astype(np.int64)
                    .tolist()
                )
            )
            != source.get("query_universe_sha256")
        ):
            raise ValueError(
                "two shard manifests are not the exact committed union"
            )
        # Restore the source manifest order so that there is a single
        # canonical merged cache regardless of greedy shard assignment.
        concatenated_positions = positions.to_numpy(dtype=np.int64)
        source_order = np.argsort(concatenated_positions, kind="stable")
        if not np.array_equal(
            concatenated_positions[source_order],
            np.arange(total_rows, dtype=np.int64),
        ):
            raise AssertionError("sealed source-order restoration failed")
        source_order_tensor = torch.from_numpy(source_order.copy()).long()

        reference = validations[0]
        row_tensor_fields = sorted(reference["tensor_fields"])
        alias_fields: dict[str, str] = {}
        for alias, canonical in TENSOR_ALIASES.items():
            if (
                alias in row_tensor_fields
                and canonical in row_tensor_fields
                and all(
                    _tensors_equal(payload[alias], payload[canonical])
                    for payload in payloads
                )
            ):
                alias_fields[alias] = canonical

        merged: dict[str, Any] = {
            key: value
            for key, value in payloads[0].items()
            if key
            not in (
                set(row_tensor_fields)
                | set(reference["row_list_fields"])
                | {
                    "manifest",
                    "extraction",
                    "split",
                    "sealed_test_read",
                }
            )
        }
        for key in row_tensor_fields:
            if key not in alias_fields:
                concatenated = torch.cat(
                    [payload[key] for payload in payloads], dim=0
                )
                merged[key] = concatenated.index_select(
                    0, source_order_tensor
                )
        for alias, canonical in alias_fields.items():
            merged[alias] = merged[canonical]
        for key in sorted(reference["row_list_fields"]):
            if key != "query360_indices":
                concatenated_values = [
                    item
                    for payload in payloads
                    for item in list(payload[key])
                ]
                merged[key] = [
                    concatenated_values[int(index)]
                    for index in source_order
                ]
        concatenated_queries = [
            int(query)
            for validation in validations
            for query in validation["queries"]
        ]
        merged_queries = [
            concatenated_queries[int(index)] for index in source_order
        ]
        merged["query360_indices"] = merged_queries
        merged["split"] = "test"
        merged["sealed_test_read"] = True

        combined_frame = combined_frame.iloc[source_order].reset_index(
            drop=True
        )
        output_frame = public_manifest(combined_frame).drop(
            columns=[SOURCE_POSITION_COLUMN], errors="raise"
        )
        if output_frame["id"].astype(str).tolist() != [
            str(value) for value in merged["ids"]
        ]:
            raise AssertionError("merged test manifest/cache order differs")
        atomic_csv(manifest_path, output_frame)
        merged["manifest"] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "rows": total_rows,
            "sealed_source": {
                "path": str(source["path"]),
                "sha256": str(source["sha256"]),
                "content_fingerprint": str(
                    source["content_fingerprint"]
                ),
            },
            "parts": [
                {
                    "manifest_path": str(
                        shards_by_cache[path]["manifest_path"]
                    ),
                    "manifest_sha256": str(
                        shards_by_cache[path]["manifest_sha256"]
                    ),
                    "cache_path": str(path),
                    "cache_sha256": input_hashes[path],
                }
                for path in input_paths
            ],
        }
        merged["extraction"] = {
            "merged_sealed_feature_shards": True,
            "sealed_test_authorized": True,
            "merge_script_version": SCRIPT_VERSION,
            "master_selection_receipt_path": str(master_receipt_path),
            "master_selection_receipt_sha256": master["sha256"],
            "master_winner_run": master["winner_run"],
            "master_winner_family": master["family"],
            "selection_lock_sha256": artifacts[
                "selection_lock_sha256"
            ],
            "checkpoint_sha256": artifacts["checkpoint_sha256"],
            "plan_sha256": sha256_file(plan_path),
            "input_caches": [
                {
                    "path": str(path),
                    "sha256": input_hashes[path],
                    "rows": int(validation["rows"]),
                }
                for path, validation in zip(input_paths, validations)
            ],
            "observations": int(
                sum(
                    int(
                        payload.get("extraction", {}).get(
                            "observations", 0
                        )
                    )
                    for payload in payloads
                )
            ),
            "durable_local_feature_cache": True,
            "training_remote_image_io": False,
            "metrics_computed": False,
            "threshold_search_performed": False,
        }
        for key in row_tensor_fields:
            if int(merged[key].shape[0]) != total_rows:
                raise AssertionError(
                    f"merged tensor {key!r} has the wrong length"
                )
        for key in (*REQUIRED_ROW_LIST_FIELDS, "query360_indices"):
            if len(merged[key]) != total_rows:
                raise AssertionError(
                    f"merged row metadata {key!r} has the wrong length"
                )

        atomic_torch(output_path, merged)
        output_sha = sha256_file(output_path)
        audit = {
            "schema_version": MERGE_AUDIT_SCHEMA,
            "script_version": SCRIPT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA,
            "split": "test",
            "sealed_test_read": True,
            "selection": {
                "master_selection_receipt_path": str(master_receipt_path),
                "master_selection_receipt_sha256": master["sha256"],
                "master_winner_run": master["winner_run"],
                "master_winner_family": master["family"],
                "selection_lock_path": str(selection_lock_path),
                "selection_lock_sha256": artifacts[
                    "selection_lock_sha256"
                ],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": artifacts["checkpoint_sha256"],
                "arm": artifacts["arm"],
                "base_mode": artifacts["base_mode"],
                "model_config_fingerprint": artifacts[
                    "model_config_fingerprint"
                ],
                "encoder_fingerprint": artifacts[
                    "encoder_fingerprint"
                ],
                "encoder_binding": artifacts["encoder_binding"],
            },
            "plan": {
                "path": str(plan_path),
                "sha256": sha256_file(plan_path),
            },
            "inputs": [
                {
                    "path": str(path),
                    "sha256": input_hashes[path],
                    "rows": int(validation["rows"]),
                    "manifest_path": str(
                        shards_by_cache[path]["manifest_path"]
                    ),
                    "manifest_sha256": str(
                        shards_by_cache[path]["manifest_sha256"]
                    ),
                }
                for path, validation in zip(input_paths, validations)
            ],
            "output": {
                "path": str(output_path),
                "sha256": output_sha,
                "rows": total_rows,
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            },
            "integrity": {
                "exact_manifest_union": True,
                "plumes_disjoint_between_shards": True,
                "events_disjoint_between_shards": True,
                "ids_unique": len(set(merged["ids"])) == total_rows,
                "query_indices_unique": (
                    len(set(merged_queries)) == total_rows
                ),
                "encoder_matches_lock": True,
            },
            "metrics_computed": False,
            "threshold_search_performed": False,
        }
        atomic_json(audit_path, audit)
        receipt = {
            "schema_version": MERGE_RECEIPT_SCHEMA,
            "script_version": SCRIPT_VERSION,
            "status": "complete",
            "one_time_merge_use_path": str(merge_use_path),
            "selection_lock_sha256": artifacts[
                "selection_lock_sha256"
            ],
            "checkpoint_sha256": artifacts["checkpoint_sha256"],
            "master_selection_receipt_path": str(master_receipt_path),
            "master_selection_receipt_sha256": master["sha256"],
            "plan_sha256": sha256_file(plan_path),
            "output_cache": str(output_path),
            "output_cache_sha256": output_sha,
            "output_manifest": str(manifest_path),
            "output_manifest_sha256": sha256_file(manifest_path),
            "audit": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "sealed_test_read": True,
            "sealed_test_evaluations": 0,
            "metrics_computed": False,
            "threshold_search_performed": False,
        }
        atomic_json(receipt_path, receipt)
        atomic_json(
            merge_use_path,
            {
                **use_record,
                "status": "complete",
                "output_cache_sha256": output_sha,
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
            },
        )
        atomic_json(
            claim_path,
            {
                **claim,
                "status": "merged",
                "merge_use_path": str(merge_use_path),
                "output_cache": str(output_path),
                "output_cache_sha256": output_sha,
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
            },
        )
        print(
            json.dumps(
                {
                    "cache": str(output_path),
                    "cache_sha256": output_sha,
                    "manifest": str(manifest_path),
                    "audit": str(audit_path),
                    "receipt": str(receipt_path),
                    "rows": total_rows,
                    "metrics_computed": False,
                    "threshold_search_performed": False,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as error:
        atomic_json(
            merge_use_path,
            {
                **use_record,
                "status": "failed_after_cache_use_claim",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard = subparsers.add_parser(
        "shard-manifest",
        help="claim and split a sealed manifest into two complete-group shards",
    )
    shard.add_argument("--manifest", required=True)
    shard.add_argument("--selection-lock", required=True)
    shard.add_argument("--checkpoint", required=True)
    shard.add_argument("--master-selection-receipt", required=True)
    shard.add_argument("--output-dir", required=True)
    shard.add_argument("--prefix", default="legacy360_sealed_test")
    shard.add_argument("--plan", default="")
    shard.add_argument("--sealed-test", action="store_true")
    shard.add_argument(
        "--extractor",
        default=str(
            Path(__file__).resolve().parent.parent
            / "query360_two_axis_full_legacy.py"
        ),
    )
    shard.add_argument("--python", default=sys.executable)
    shard.add_argument("--raw-cache-dir", default="")
    shard.add_argument(
        "--raw-cache-readonly-fallback", action="store_true"
    )
    shard.add_argument("--row-batch-size", type=int, default=64)
    shard.add_argument("--encoder-microbatch", type=int, default=256)
    # Four workers/prefetch-one per extractor is the audited stable ceiling.
    # The older 16/2 example amplified rogue-S2 host memory and can OOM.
    shard.add_argument("--num-workers", type=int, default=4)
    shard.add_argument("--prefetch-factor", type=int, default=1)
    shard.set_defaults(function=command_shard_manifest)

    merge = subparsers.add_parser(
        "merge-cache",
        help="one-time strict merge of exactly two sealed test feature caches",
    )
    merge.add_argument(
        "--input-cache",
        action="append",
        required=True,
        help="repeat exactly twice; paths must match the committed plan",
    )
    merge.add_argument("--plan", required=True)
    merge.add_argument("--selection-lock", required=True)
    merge.add_argument("--checkpoint", required=True)
    merge.add_argument("--master-selection-receipt", required=True)
    merge.add_argument("--output-cache", required=True)
    merge.add_argument("--output-manifest", default="")
    merge.add_argument("--audit", default="")
    merge.add_argument("--receipt", default="")
    merge.add_argument("--sealed-test", action="store_true")
    merge.set_defaults(function=command_merge_cache)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
