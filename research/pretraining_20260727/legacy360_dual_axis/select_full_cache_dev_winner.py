#!/usr/bin/env python3
"""Select one immutable dev winner across the four full-cache candidates.

This is a development-only campaign auditor.  It has no held-out cache,
manifest, prediction, or evaluation argument.  It validates the four exact
promotion configurations, their complete status/history/selection locks, and
their SHA-bound checkpoints before writing one master selection receipt.

Ranking is deterministic:

1. higher development ``best_binary_f1``;
2. higher development AP;
3. higher development ROC-AUC;
4. earlier selected epoch;
5. fewer trainable head parameters;
6. earlier position in the explicit candidate table below.

The receipt points to the original winning lock/checkpoint; it does not create
a new model lock and cannot change the locked epoch or threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from shard_and_merge_feature_cache import (
    atomic_json,
    atomic_text,
    object_fingerprint,
    sha256_file,
)


SCRIPT_VERSION = "legacy360-full-cache-dev-winner-v1"
RECEIPT_SCHEMA = "legacy360-full-cache-master-selection-receipt-v1"
CAMPAIGN_STATUS_SCHEMA = "legacy360-full-cache-promotion-status-v1"
RUN_LAUNCH_STATUS_SCHEMA = "legacy360-full-cache-promotion-run-status-v1"
SELECTION_METRIC = "best_binary_f1"
EXPECTED_TRAIN_ROWS = 113843
EXPECTED_DEV_ROWS = 12621
MINIMUM_READY_BEST_F1 = 0.895


class SelectionAuditError(ValueError):
    """The promotion campaign is not eligible for immutable selection."""


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    arm: str
    learning_rate: float
    lock_schema: str
    checkpoint_schema: str
    summary_schema: str
    evaluator: str


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        "promote_gdelta_universal_lr1e4_cap1p5",
        "gated_delta",
        "scale_aware_gated_delta",
        1e-4,
        "query360-gated-delta-selection-lock-v1",
        "query360-gated-delta-head-v1",
        "query360-gated-delta-summary-v1",
        "query360_gated_delta_runner.py",
    ),
    CandidateSpec(
        "promote_gdelta_universal_lr3e4_cap1p5",
        "gated_delta",
        "scale_aware_gated_delta",
        3e-4,
        "query360-gated-delta-selection-lock-v1",
        "query360-gated-delta-head-v1",
        "query360-gated-delta-summary-v1",
        "query360_gated_delta_runner.py",
    ),
    CandidateSpec(
        "promote_compact_scale_aware_d64_lr1e4",
        "compact_axial",
        "scale_aware_two_axis_query",
        1e-4,
        "query360-two-axis-selection-lock-v1",
        "query360-two-axis-head-v1",
        "query360-two-axis-full-legacy-summary-v2",
        "query360_two_axis_full_legacy.py",
    ),
    CandidateSpec(
        "promote_compact_two_axis_d64_lr3e5",
        "compact_axial",
        "two_axis_query",
        3e-5,
        "query360-two-axis-selection-lock-v1",
        "query360-two-axis-head-v1",
        "query360-two-axis-full-legacy-summary-v2",
        "query360_two_axis_full_legacy.py",
    ),
)


def _absolute(value: str | Path) -> Path:
    return Path(value).expanduser().absolute()


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SelectionAuditError(
            f"value is not finite canonical JSON: {error}"
        ) from error


def _same(left: Any, right: Any, role: str) -> None:
    if _canonical(left) != _canonical(right):
        raise SelectionAuditError(f"{role}: artifacts differ")


def _mapping(value: Any, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionAuditError(f"{role}: expected a JSON object")
    return value


def _finite(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionAuditError(f"{role}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SelectionAuditError(f"{role}: expected a finite number")
    return result


def _unit(value: Any, role: str) -> float:
    result = _finite(value, role)
    if not 0.0 <= result <= 1.0:
        raise SelectionAuditError(f"{role}: expected a value in [0,1]")
    return result


def _assert_dev_only(value: Any, role: str) -> None:
    """Reject any positive held-out-use marker in campaign JSON."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_role = f"{role}.{key}"
            if key == "sealed_test_read" and child is not False:
                raise SelectionAuditError(f"{child_role}: must be false")
            if key == "sealed_test_evaluations" and child != 0:
                raise SelectionAuditError(f"{child_role}: must be zero")
            if key in {
                "test_cache_read_before_lock",
                "test_cache_read_after_selection_lock",
                "test_threshold_search_performed",
            } and child is not False:
                raise SelectionAuditError(f"{child_role}: must be false")
            if key == "sealed_test" and child is not None:
                raise SelectionAuditError(f"{child_role}: must be null")
            _assert_dev_only(child, child_role)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_dev_only(child, f"{role}[{index}]")


def _read_json(path: Path, role: str) -> tuple[Any, str]:
    if path.is_symlink():
        raise SelectionAuditError(f"{role}: symlinks are not accepted")
    if not path.is_file():
        raise SelectionAuditError(f"{role}: missing artifact {path}")
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SelectionAuditError(f"{role}: artifact changed while read")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectionAuditError(f"{role}: invalid JSON: {error}") from error
    _assert_dev_only(value, role)
    import hashlib

    return value, hashlib.sha256(data).hexdigest()


def _load_checkpoint(path: Path, expected_sha: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionAuditError(f"checkpoint is unavailable: {path}")
    before = path.stat()
    first_sha = sha256_file(path)
    if first_sha != expected_sha:
        raise SelectionAuditError("checkpoint SHA256 differs from lock")
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    after = path.stat()
    second_sha = sha256_file(path)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or first_sha != second_sha:
        raise SelectionAuditError("checkpoint changed while audited")
    return _mapping(checkpoint, "checkpoint")


def _expected_best(
    history: Sequence[Any], metric: str, role: str
) -> Mapping[str, Any]:
    best: Mapping[str, Any] | None = None
    best_score = -math.inf
    observed_epochs: list[int] = []
    for index, raw_record in enumerate(history):
        record = _mapping(raw_record, f"{role}[{index}]")
        epoch = record.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise SelectionAuditError(f"{role}[{index}].epoch is invalid")
        observed_epochs.append(epoch)
        if epoch == 0 and record.get("train") is not None:
            raise SelectionAuditError(f"{role}: epoch 0 train must be null")
        if epoch > 0 and not isinstance(record.get("train"), Mapping):
            raise SelectionAuditError(
                f"{role}: trained epoch {epoch} has no train metrics"
            )
        dev = _mapping(record.get("dev"), f"{role}[{index}].dev")
        score = _unit(dev.get(metric), f"{role}[{index}].dev.{metric}")
        _unit(dev.get("ap"), f"{role}[{index}].dev.ap")
        _unit(dev.get("auc"), f"{role}[{index}].dev.auc")
        _unit(
            dev.get("best_binary_f1_threshold"),
            f"{role}[{index}].dev.best_binary_f1_threshold",
        )
        if best is None or score > best_score:
            best = record
            best_score = score
    if not observed_epochs or observed_epochs != list(
        range(observed_epochs[-1] + 1)
    ):
        raise SelectionAuditError(f"{role}: epochs are not contiguous from 0")
    assert best is not None
    return best


def _validate_config(
    *,
    spec: CandidateSpec,
    lock: Mapping[str, Any],
    summary: Mapping[str, Any],
    epochs: int,
) -> None:
    training = _mapping(summary.get("training"), f"{spec.name}.training")
    model = _mapping(summary.get("model"), f"{spec.name}.model")
    config = _mapping(lock.get("model_config"), f"{spec.name}.model_config")
    if model.get("arm") != spec.arm or model.get("base_mode") != "universal":
        raise SelectionAuditError(f"{spec.name}: summary arm/base mismatch")
    if lock.get("arm") != spec.arm or lock.get("base_mode") != "universal":
        raise SelectionAuditError(f"{spec.name}: lock arm/base mismatch")
    if not math.isclose(
        _finite(training.get("learning_rate"), f"{spec.name}.learning_rate"),
        spec.learning_rate,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise SelectionAuditError(f"{spec.name}: learning rate mismatch")
    if training.get("selection_metric") != SELECTION_METRIC:
        raise SelectionAuditError(f"{spec.name}: selection metric mismatch")
    if int(summary.get("train_rows", -1)) != EXPECTED_TRAIN_ROWS:
        raise SelectionAuditError(f"{spec.name}: train row count mismatch")
    if int(summary.get("dev_rows", -1)) != EXPECTED_DEV_ROWS:
        raise SelectionAuditError(f"{spec.name}: dev row count mismatch")

    if spec.family == "gated_delta":
        if int(training.get("epochs_requested", -1)) != epochs:
            raise SelectionAuditError(f"{spec.name}: epoch budget mismatch")
        expected = {
            "bottleneck_dim": 32,
            "residual_cap": 1.5,
            "dropout": 0.05,
        }
        for key, value in expected.items():
            observed = config.get(key)
            if isinstance(value, int):
                equal = int(observed) == value
            else:
                equal = math.isclose(
                    float(observed), value, rel_tol=0.0, abs_tol=1e-15
                )
            if not equal:
                raise SelectionAuditError(
                    f"{spec.name}: model_config.{key} mismatch"
                )
        for key, value in {
            "weight_decay": 0.01,
            "sensor_aux_weight": 0.1,
            "residual_l2": 1e-3,
        }.items():
            if not math.isclose(
                float(training.get(key, math.nan)),
                value,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise SelectionAuditError(
                    f"{spec.name}: training.{key} mismatch"
                )
    else:
        if int(training.get("epochs", -1)) != epochs:
            raise SelectionAuditError(f"{spec.name}: epoch budget mismatch")
        expected = {
            "model_dim": 64,
            "num_heads": 4,
            "temporal_depth": 1,
            "mlp_ratio": 1.0,
            "dropout": 0.0,
        }
        for key, value in expected.items():
            observed = config.get(key)
            if isinstance(value, int):
                equal = int(observed) == value
            else:
                equal = math.isclose(
                    float(observed), value, rel_tol=0.0, abs_tol=1e-15
                )
            if not equal:
                raise SelectionAuditError(
                    f"{spec.name}: model_config.{key} mismatch"
                )
        for key, value in {
            "sensor_aux_weight": 0.05,
            "axis_aux_weight": 0.05,
        }.items():
            if not math.isclose(
                float(training.get(key, math.nan)),
                value,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise SelectionAuditError(
                    f"{spec.name}: training.{key} mismatch"
                )


def _validate_candidate(
    spec: CandidateSpec,
    campaign_root: Path,
    epochs: int,
) -> dict[str, Any]:
    run_dir = campaign_root / spec.name
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise SelectionAuditError(f"{spec.name}: run directory is missing")
    filenames = (
        "launcher_status.json",
        "run_status.json",
        "summary.json",
        "metrics_history.json",
        "selection_lock.json",
    )
    payloads: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for filename in filenames:
        payload, digest = _read_json(
            run_dir / filename, f"{spec.name}.{filename}"
        )
        payloads[filename] = payload
        hashes[filename] = digest
    launch = _mapping(payloads["launcher_status.json"], f"{spec.name}.launch")
    status = _mapping(payloads["run_status.json"], f"{spec.name}.status")
    summary = _mapping(payloads["summary.json"], f"{spec.name}.summary")
    lock = _mapping(payloads["selection_lock.json"], f"{spec.name}.lock")
    history = payloads["metrics_history.json"]
    if not isinstance(history, list):
        raise SelectionAuditError(f"{spec.name}: history is not a list")

    if launch.get("schema_version") != RUN_LAUNCH_STATUS_SCHEMA:
        raise SelectionAuditError(f"{spec.name}: launcher schema mismatch")
    for payload, role in (
        (launch, "launcher"),
        (status, "runner"),
        (summary, "summary"),
    ):
        if payload.get("status") != "complete":
            raise SelectionAuditError(f"{spec.name}: {role} is incomplete")
    if launch.get("run") != spec.name or launch.get("family") != spec.family:
        raise SelectionAuditError(f"{spec.name}: launch identity mismatch")
    if launch.get("epoch0_base_verified") is not True:
        raise SelectionAuditError(f"{spec.name}: epoch 0 was not verified")
    if summary.get("schema_version") != spec.summary_schema:
        raise SelectionAuditError(f"{spec.name}: summary schema mismatch")
    if lock.get("schema_version") != spec.lock_schema:
        raise SelectionAuditError(f"{spec.name}: lock schema mismatch")
    if lock.get("selection_metric") != SELECTION_METRIC:
        raise SelectionAuditError(f"{spec.name}: lock metric mismatch")
    _same(summary.get("history"), history, f"{spec.name}: summary history")
    _same(
        summary.get("selection_lock"),
        lock,
        f"{spec.name}: embedded selection lock",
    )
    _validate_config(spec=spec, lock=lock, summary=summary, epochs=epochs)

    expected_best = _expected_best(
        history, SELECTION_METRIC, f"{spec.name}.history"
    )
    _same(summary.get("best"), expected_best, f"{spec.name}: selected best")
    best_epoch = int(expected_best["epoch"])
    if spec.family == "compact_axial" and len(history) != epochs + 1:
        raise SelectionAuditError(
            f"{spec.name}: axial history did not complete all epochs"
        )
    if len(history) > epochs + 1:
        raise SelectionAuditError(f"{spec.name}: history exceeds epoch budget")
    if int(lock.get("best_epoch", -1)) != best_epoch:
        raise SelectionAuditError(f"{spec.name}: locked epoch mismatch")
    if int(status.get("best_epoch", -1)) != best_epoch:
        raise SelectionAuditError(f"{spec.name}: status epoch mismatch")
    best_metrics = _mapping(
        expected_best.get("dev"), f"{spec.name}.best.dev"
    )
    epoch0_metrics = _mapping(history[0].get("dev"), f"{spec.name}.epoch0.dev")
    _same(
        status.get("best_dev_metrics"),
        best_metrics,
        f"{spec.name}: status best metrics",
    )
    score = _unit(lock.get("selection_score"), f"{spec.name}.selection_score")
    if score != float(best_metrics[SELECTION_METRIC]):
        raise SelectionAuditError(f"{spec.name}: locked score mismatch")
    threshold = _unit(
        lock.get("locked_threshold"), f"{spec.name}.locked_threshold"
    )
    if threshold != float(best_metrics["best_binary_f1_threshold"]):
        raise SelectionAuditError(f"{spec.name}: locked threshold mismatch")
    if score < float(epoch0_metrics[SELECTION_METRIC]):
        raise SelectionAuditError(f"{spec.name}: selected worse than epoch 0")

    checkpoint_path = _absolute(str(lock.get("checkpoint", "")))
    expected_checkpoint = (run_dir / "checkpoint_best.pth").absolute()
    if checkpoint_path != expected_checkpoint:
        raise SelectionAuditError(f"{spec.name}: checkpoint path mismatch")
    checkpoint_sha = str(lock.get("checkpoint_sha256", ""))
    checkpoint = _load_checkpoint(checkpoint_path, checkpoint_sha)
    if checkpoint.get("schema_version") != spec.checkpoint_schema:
        raise SelectionAuditError(f"{spec.name}: checkpoint schema mismatch")
    for key, expected in (
        ("epoch", best_epoch),
        ("arm", spec.arm),
        ("base_mode", "universal"),
        ("model_config", lock.get("model_config")),
    ):
        if checkpoint.get(key) != expected:
            raise SelectionAuditError(
                f"{spec.name}: checkpoint {key} mismatch"
            )
    _same(checkpoint.get("dev"), best_metrics, f"{spec.name}: checkpoint dev")
    if float(checkpoint.get("locked_threshold_candidate", math.nan)) != threshold:
        raise SelectionAuditError(
            f"{spec.name}: checkpoint threshold mismatch"
        )
    signature = _mapping(
        checkpoint.get("parameter_signature"),
        f"{spec.name}.parameter_signature",
    )
    parameter_count = signature.get("parameter_count")
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count < 1
    ):
        raise SelectionAuditError(f"{spec.name}: parameter count is invalid")
    encoder = _mapping(lock.get("encoder"), f"{spec.name}.encoder")
    state_sha = str(encoder.get("state_sha256", ""))
    if len(state_sha) != 64 or any(
        character not in "0123456789abcdef" for character in state_sha
    ):
        raise SelectionAuditError(f"{spec.name}: encoder SHA256 is invalid")
    if (
        checkpoint.get("encoder") is not None
        and checkpoint.get("encoder") != encoder
    ):
        raise SelectionAuditError(f"{spec.name}: checkpoint encoder mismatch")
    if spec.family == "gated_delta":
        if checkpoint.get("encoder") != encoder:
            raise SelectionAuditError(
                f"{spec.name}: gated checkpoint lacks encoder binding"
            )
        if checkpoint.get("base_contract") != lock.get("base_contract"):
            raise SelectionAuditError(
                f"{spec.name}: gated base contract mismatch"
            )

    for role in ("train_manifest", "dev_manifest", "split_guard"):
        if not isinstance(lock.get(role), Mapping):
            raise SelectionAuditError(f"{spec.name}: lock lacks {role}")

    return {
        "candidate_index": CANDIDATES.index(spec),
        "run": spec.name,
        "family": spec.family,
        "arm": spec.arm,
        "base_mode": "universal",
        "learning_rate": spec.learning_rate,
        "selection_metric": SELECTION_METRIC,
        "selection_score": score,
        "ap": _unit(best_metrics.get("ap"), f"{spec.name}.best.ap"),
        "auc": _unit(best_metrics.get("auc"), f"{spec.name}.best.auc"),
        "best_epoch": best_epoch,
        "locked_threshold": threshold,
        "parameter_count": int(parameter_count),
        "epoch0": dict(epoch0_metrics),
        "encoder": dict(encoder),
        "encoder_fingerprint": object_fingerprint(encoder),
        "train_manifest": lock["train_manifest"],
        "dev_manifest": lock["dev_manifest"],
        "split_guard": lock["split_guard"],
        "lock_schema": spec.lock_schema,
        "checkpoint_schema": spec.checkpoint_schema,
        "selection_lock": {
            "path": str((run_dir / "selection_lock.json").absolute()),
            "sha256": hashes["selection_lock.json"],
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
        },
        "evaluator_script": spec.evaluator,
        "artifacts": {
            filename: {
                "path": str((run_dir / filename).absolute()),
                "sha256": hashes[filename],
            }
            for filename in filenames
        },
    }


def _rank_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(candidate["selection_score"]),
        -float(candidate["ap"]),
        -float(candidate["auc"]),
        int(candidate["best_epoch"]),
        int(candidate["parameter_count"]),
        int(candidate["candidate_index"]),
    )


def command_select(args: argparse.Namespace) -> None:
    campaign_root = _absolute(args.campaign_root)
    if campaign_root.is_symlink() or not campaign_root.is_dir():
        raise SelectionAuditError("campaign root is unavailable")
    receipt_path = (
        _absolute(args.output)
        if args.output
        else campaign_root / "master_dev_selection_receipt.json"
    )
    receipt_sha_path = receipt_path.with_suffix(
        receipt_path.suffix + ".sha256"
    )
    existing = [
        str(path)
        for path in (receipt_path, receipt_sha_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite master selection artifacts: "
            + ", ".join(existing)
        )

    campaign_status_path = campaign_root / "promotion_launcher_status.json"
    campaign_status, campaign_status_sha = _read_json(
        campaign_status_path, "campaign status"
    )
    campaign_status = _mapping(campaign_status, "campaign status")
    if campaign_status.get("schema_version") != CAMPAIGN_STATUS_SCHEMA:
        raise SelectionAuditError("campaign status schema mismatch")
    if (
        campaign_status.get("status") != "complete"
        or campaign_status.get("candidate_count") != 4
        or campaign_status.get("epoch0_base_required") is not True
    ):
        raise SelectionAuditError("campaign status is not selection-ready")
    epochs = campaign_status.get("epochs")
    if isinstance(epochs, bool) or epochs not in (1, 2, 3):
        raise SelectionAuditError("campaign epoch budget is invalid")

    candidates = [
        _validate_candidate(spec, campaign_root, int(epochs))
        for spec in CANDIDATES
    ]
    reference = candidates[0]
    for candidate in candidates[1:]:
        for key in (
            "encoder",
            "train_manifest",
            "dev_manifest",
            "split_guard",
            "epoch0",
        ):
            _same(
                candidate[key],
                reference[key],
                f"matched campaign {key}",
            )

    ranked = sorted(candidates, key=_rank_key)
    winner = ranked[0]
    if float(winner["selection_score"]) < MINIMUM_READY_BEST_F1:
        raise SelectionAuditError(
            "no candidate meets the precommitted development readiness "
            f"floor {MINIMUM_READY_BEST_F1:.3f}"
        )
    runner_root = Path(__file__).resolve().parent.parent
    feature_sharder = (
        Path(__file__).resolve().parent / "sealed_test_feature_shards.py"
    )
    evaluator = runner_root / str(winner["evaluator_script"])
    if not feature_sharder.is_file() or not evaluator.is_file():
        raise SelectionAuditError("downstream locked tooling is unavailable")

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "selected_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "scope": "development-only cross-family selection",
            "selection_metric": SELECTION_METRIC,
            "candidate_count": 4,
            "epoch_budget": int(epochs),
            "tie_break": [
                "higher development best_binary_f1",
                "higher development average precision",
                "higher development ROC-AUC",
                "earlier development-selected epoch",
                "fewer trainable head parameters",
                "earlier index in the explicit four-candidate table",
            ],
            "rank_key": (
                "(-best_binary_f1,-ap,-auc,best_epoch,"
                "parameter_count,candidate_index)"
            ),
            "minimum_ready_best_binary_f1": MINIMUM_READY_BEST_F1,
            "epoch0_exact_base_common_across_candidates": True,
            "no_retraining_after_selection": True,
        },
        "campaign": {
            "root": str(campaign_root),
            "status_path": str(campaign_status_path),
            "status_sha256": campaign_status_sha,
        },
        "common_contract": {
            "train_rows": EXPECTED_TRAIN_ROWS,
            "dev_rows": EXPECTED_DEV_ROWS,
            "encoder": reference["encoder"],
            "encoder_fingerprint": reference["encoder_fingerprint"],
            "train_manifest": reference["train_manifest"],
            "dev_manifest": reference["dev_manifest"],
            "split_guard": reference["split_guard"],
            "epoch0": reference["epoch0"],
        },
        "ranked_candidates": [
            {
                key: candidate[key]
                for key in (
                    "run",
                    "candidate_index",
                    "family",
                    "arm",
                    "base_mode",
                    "learning_rate",
                    "selection_metric",
                    "selection_score",
                    "ap",
                    "auc",
                    "best_epoch",
                    "locked_threshold",
                    "parameter_count",
                    "selection_lock",
                    "checkpoint",
                )
            }
            for candidate in ranked
        ],
        "winner": {
            key: winner[key]
            for key in (
                "run",
                "candidate_index",
                "family",
                "arm",
                "base_mode",
                "learning_rate",
                "selection_metric",
                "selection_score",
                "ap",
                "auc",
                "best_epoch",
                "locked_threshold",
                "parameter_count",
                "lock_schema",
                "checkpoint_schema",
                "selection_lock",
                "checkpoint",
            )
        },
        "locked_dispatch": {
            "feature_sharder": str(feature_sharder),
            "evaluator": str(evaluator),
            "evaluator_family": winner["family"],
            "selection_lock": winner["selection_lock"],
            "checkpoint": winner["checkpoint"],
            "required_explicit_authorization_flag": "--sealed-test",
            "feature_stage_computes_metrics": False,
            "feature_stage_searches_threshold": False,
            "evaluation_uses_locked_threshold": True,
        },
        "sealed_test_read": False,
        "sealed_test_evaluations": 0,
    }
    atomic_json(receipt_path, receipt)
    receipt_sha = sha256_file(receipt_path)
    atomic_text(
        receipt_sha_path,
        f"{receipt_sha}  {receipt_path.name}\n",
    )
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "receipt_sha256": receipt_sha,
                "winner": winner["run"],
                "family": winner["family"],
                "selection_score": winner["selection_score"],
                "selection_lock": winner["selection_lock"],
                "checkpoint": winner["checkpoint"],
                "evaluator": str(evaluator),
                "sealed_test_read": False,
                "sealed_test_evaluations": 0,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument(
        "--output",
        default="",
        help=(
            "new receipt path; defaults to "
            "CAMPAIGN_ROOT/master_dev_selection_receipt.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    command_select(args)


if __name__ == "__main__":
    main()
