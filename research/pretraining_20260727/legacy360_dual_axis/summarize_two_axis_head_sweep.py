#!/usr/bin/env python3
"""Audit and summarize the formal legacy-360 development-only head sweep.

This program deliberately has no test-cache or sealed-evaluation input.  It
reads exactly four JSON artifacts from each of the ten predeclared run
directories:

* ``run_status.json``
* ``summary.json``
* ``metrics_history.json``
* ``selection_lock.json``

All inputs are validated before any report is written.  In particular, a
failed/incomplete run, any indication that sealed test data was read, or any
cross-artifact disagreement makes the whole command fail closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SUMMARY_SCHEMA = "query360-two-axis-full-legacy-summary-v2"
LOCK_SCHEMA = "query360-two-axis-selection-lock-v1"
REPORT_SCHEMA = "query360-two-axis-dev-sweep-report-v1"
EXPECTED_SELECTION_METRIC = "best_binary_f1"
SENSORS = ("s2", "l89", "emit", "s5p")
INPUT_FILENAMES = (
    "run_status.json",
    "summary.json",
    "metrics_history.json",
    "selection_lock.json",
)
DELTA_METRICS = (
    "binary_f1_at_0_5",
    "macro_f1_at_0_5",
    "best_binary_f1",
    "best_macro_f1_at_binary_threshold",
    "ap",
    "auc",
)
DISPLAY_METRICS = (
    "binary_f1_at_0_5",
    "macro_f1_at_0_5",
    "best_binary_f1",
    "best_binary_f1_threshold",
    "best_macro_f1_at_binary_threshold",
    "ap",
    "auc",
)


class SweepValidationError(ValueError):
    """The sweep cannot be summarized without violating the audit contract."""


@dataclass(frozen=True)
class RunSpec:
    name: str
    base_mode: str
    arm: str
    learning_rate: float


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec("hybrid_current_only_lr1e4", "hybrid", "current_only", 1e-4),
    RunSpec("hybrid_current_only_lr3e4", "hybrid", "current_only", 3e-4),
    RunSpec(
        "hybrid_two_axis_lr1e4", "hybrid", "two_axis_query", 1e-4
    ),
    RunSpec(
        "hybrid_two_axis_lr3e4", "hybrid", "two_axis_query", 3e-4
    ),
    RunSpec(
        "hybrid_scale_aware_lr1e4",
        "hybrid",
        "scale_aware_two_axis_query",
        1e-4,
    ),
    RunSpec(
        "hybrid_scale_aware_lr3e4",
        "hybrid",
        "scale_aware_two_axis_query",
        3e-4,
    ),
    RunSpec(
        "universal_current_only_lr1e4",
        "universal",
        "current_only",
        1e-4,
    ),
    RunSpec(
        "universal_current_only_lr3e4",
        "universal",
        "current_only",
        3e-4,
    ),
    RunSpec(
        "universal_scale_aware_lr1e4",
        "universal",
        "scale_aware_two_axis_query",
        1e-4,
    ),
    RunSpec(
        "universal_scale_aware_lr3e4",
        "universal",
        "scale_aware_two_axis_query",
        3e-4,
    ),
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _same(left: Any, right: Any, *, source: str) -> None:
    try:
        equal = _canonical(left) == _canonical(right)
    except (TypeError, ValueError) as error:
        raise SweepValidationError(
            f"{source}: value is not finite JSON: {error}"
        ) from error
    if not equal:
        raise SweepValidationError(f"{source}: artifacts disagree")


def _mapping(value: Any, *, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SweepValidationError(f"{source}: expected a JSON object")
    return value


def _sequence(value: Any, *, source: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise SweepValidationError(f"{source}: expected a JSON list")
    return value


def _finite_number(value: Any, *, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepValidationError(f"{source}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SweepValidationError(f"{source}: expected a finite number")
    return result


def _unit_metric(value: Any, *, source: str) -> float:
    result = _finite_number(value, source=source)
    if result < 0.0 or result > 1.0:
        raise SweepValidationError(
            f"{source}: metric must lie in [0, 1], got {result}"
        )
    return result


def _integer(value: Any, *, source: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SweepValidationError(f"{source}: expected an integer")
    if value < minimum:
        raise SweepValidationError(
            f"{source}: expected >= {minimum}, got {value}"
        )
    return int(value)


def _assert_false(value: Any, *, source: str) -> None:
    if value is not False:
        raise SweepValidationError(f"{source}: must be exactly false")


def _assert_zero(value: Any, *, source: str) -> None:
    if isinstance(value, bool) or value != 0:
        raise SweepValidationError(f"{source}: must be exactly zero")


def _assert_no_test_use(value: Any, *, source: str) -> None:
    """Recursively reject positive test-use markers in the four artifacts."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_source = f"{source}.{key}"
            if key == "sealed_test_read":
                _assert_false(child, source=child_source)
            elif key == "sealed_test_evaluations":
                _assert_zero(child, source=child_source)
            elif key in {
                "test_cache_read_before_lock",
                "test_cache_read_after_selection_lock",
                "test_threshold_search_performed",
            }:
                _assert_false(child, source=child_source)
            elif key == "sealed_test" and child is not None:
                raise SweepValidationError(
                    f"{child_source}: must remain null for a dev-only sweep"
                )
            _assert_no_test_use(child, source=child_source)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_test_use(child, source=f"{source}[{index}]")


def _read_json_stable(path: Path) -> tuple[Any, str]:
    if path.is_symlink():
        raise SweepValidationError(f"{path}: symlink inputs are not accepted")
    if not path.is_file():
        raise SweepValidationError(f"{path}: required artifact is missing")
    before = path.stat()
    payload_bytes = path.read_bytes()
    after = path.stat()
    signature_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    signature_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if signature_before != signature_after:
        raise SweepValidationError(f"{path}: artifact changed while read")
    try:
        value = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SweepValidationError(f"{path}: invalid JSON: {error}") from error
    _assert_no_test_use(value, source=str(path))
    return value, hashlib.sha256(payload_bytes).hexdigest()


def _validate_metric_block(
    value: Any, *, source: str, require_by_sensor: bool
) -> Mapping[str, Any]:
    metrics = _mapping(value, source=source)
    for name in DISPLAY_METRICS:
        _unit_metric(metrics.get(name), source=f"{source}.{name}")
    for name in (
        "binary_f1",
        "macro_f1",
        "balanced_accuracy",
        "predicted_positive_rate",
        "balanced_accuracy_at_0_5",
        "predicted_positive_rate_at_0_5",
    ):
        _unit_metric(metrics.get(name), source=f"{source}.{name}")
    _integer(metrics.get("rows"), source=f"{source}.rows", minimum=1)
    positives = _integer(
        metrics.get("positives"), source=f"{source}.positives"
    )
    if positives > int(metrics["rows"]):
        raise SweepValidationError(
            f"{source}.positives: cannot exceed row count"
        )
    decision_threshold = _unit_metric(
        metrics.get("decision_threshold"),
        source=f"{source}.decision_threshold",
    )
    if decision_threshold != 0.5:
        raise SweepValidationError(
            f"{source}.decision_threshold: expected fixed 0.5"
        )
    if float(metrics["binary_f1"]) != float(metrics["binary_f1_at_0_5"]):
        raise SweepValidationError(
            f"{source}: binary_f1 alias differs from binary_f1_at_0_5"
        )
    if float(metrics["macro_f1"]) != float(metrics["macro_f1_at_0_5"]):
        raise SweepValidationError(
            f"{source}: macro_f1 alias differs from macro_f1_at_0_5"
        )
    if require_by_sensor:
        by_sensor = _mapping(
            metrics.get("by_sensor"), source=f"{source}.by_sensor"
        )
        if set(by_sensor) != set(SENSORS):
            raise SweepValidationError(
                f"{source}.by_sensor: expected exactly {SENSORS}, "
                f"got {tuple(sorted(map(str, by_sensor)))}"
            )
        for sensor in SENSORS:
            _validate_metric_block(
                by_sensor[sensor],
                source=f"{source}.by_sensor.{sensor}",
                require_by_sensor=False,
            )
    return metrics


def _expected_best(
    history: Sequence[Any], selection_metric: str, *, source: str
) -> Mapping[str, Any]:
    best: Mapping[str, Any] | None = None
    best_score = -math.inf
    for index, raw_record in enumerate(history):
        record = _mapping(raw_record, source=f"{source}[{index}]")
        dev = _mapping(record.get("dev"), source=f"{source}[{index}].dev")
        score = _finite_number(
            dev.get(selection_metric),
            source=f"{source}[{index}].dev.{selection_metric}",
        )
        # This reproduces the trainer's strict-greater update rule, including
        # its deterministic preference for the earliest epoch on an exact tie.
        if best is None or score > best_score:
            best = record
            best_score = score
    if best is None:
        raise SweepValidationError(f"{source}: history is empty")
    return best


def _validate_run(spec: RunSpec, run_dir: Path) -> dict[str, Any]:
    if run_dir.is_symlink():
        raise SweepValidationError(f"{run_dir}: symlink run dirs are rejected")
    if not run_dir.is_dir():
        raise SweepValidationError(f"{run_dir}: expected run directory")

    payloads: dict[str, Any] = {}
    digests: dict[str, str] = {}
    for filename in INPUT_FILENAMES:
        value, digest = _read_json_stable(run_dir / filename)
        payloads[filename] = value
        digests[filename] = digest

    status = _mapping(
        payloads["run_status.json"], source=f"{spec.name}.run_status"
    )
    summary = _mapping(
        payloads["summary.json"], source=f"{spec.name}.summary"
    )
    history = _sequence(
        payloads["metrics_history.json"],
        source=f"{spec.name}.metrics_history",
    )
    lock = _mapping(
        payloads["selection_lock.json"],
        source=f"{spec.name}.selection_lock",
    )

    if status.get("status") != "complete":
        raise SweepValidationError(
            f"{spec.name}.run_status: run is not complete "
            f"(status={status.get('status')!r})"
        )
    _assert_false(
        status.get("sealed_test_read"),
        source=f"{spec.name}.run_status.sealed_test_read",
    )
    _assert_zero(
        status.get("sealed_test_evaluations"),
        source=f"{spec.name}.run_status.sealed_test_evaluations",
    )
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise SweepValidationError(
            f"{spec.name}.summary: unsupported schema"
        )
    if summary.get("status") != "complete":
        raise SweepValidationError(f"{spec.name}.summary: status is not complete")
    _assert_false(
        summary.get("sealed_test_read"),
        source=f"{spec.name}.summary.sealed_test_read",
    )
    _assert_zero(
        summary.get("sealed_test_evaluations"),
        source=f"{spec.name}.summary.sealed_test_evaluations",
    )
    if summary.get("sealed_test", object()) is not None:
        raise SweepValidationError(
            f"{spec.name}.summary.sealed_test: must be explicitly null"
        )
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise SweepValidationError(
            f"{spec.name}.selection_lock: unsupported schema"
        )
    _assert_false(
        lock.get("test_cache_read_before_lock"),
        source=f"{spec.name}.selection_lock.test_cache_read_before_lock",
    )

    summary_history = _sequence(
        summary.get("history"), source=f"{spec.name}.summary.history"
    )
    _same(
        summary_history,
        history,
        source=f"{spec.name}: summary/history vs metrics_history",
    )
    _same(
        summary.get("selection_lock"),
        lock,
        source=f"{spec.name}: embedded vs external selection lock",
    )

    model = _mapping(summary.get("model"), source=f"{spec.name}.summary.model")
    training = _mapping(
        summary.get("training"), source=f"{spec.name}.summary.training"
    )
    if model.get("base_mode") != spec.base_mode:
        raise SweepValidationError(
            f"{spec.name}: base_mode={model.get('base_mode')!r}, "
            f"expected {spec.base_mode!r}"
        )
    if model.get("arm") != spec.arm:
        raise SweepValidationError(
            f"{spec.name}: arm={model.get('arm')!r}, expected {spec.arm!r}"
        )
    learning_rate = _finite_number(
        training.get("learning_rate"),
        source=f"{spec.name}.summary.training.learning_rate",
    )
    if not math.isclose(
        learning_rate, spec.learning_rate, rel_tol=0.0, abs_tol=1e-15
    ):
        raise SweepValidationError(
            f"{spec.name}: learning_rate={learning_rate}, "
            f"expected {spec.learning_rate}"
        )
    selection_metric = str(training.get("selection_metric", ""))
    if selection_metric != EXPECTED_SELECTION_METRIC:
        raise SweepValidationError(
            f"{spec.name}: selection_metric={selection_metric!r}, "
            f"expected {EXPECTED_SELECTION_METRIC!r}"
        )
    if lock.get("arm") != spec.arm or lock.get("base_mode") != spec.base_mode:
        raise SweepValidationError(
            f"{spec.name}: selection lock arm/base disagrees with run plan"
        )
    if lock.get("selection_metric") != selection_metric:
        raise SweepValidationError(
            f"{spec.name}: lock selection metric disagrees with summary"
        )
    if lock.get("encoder") != summary.get("encoder"):
        raise SweepValidationError(
            f"{spec.name}: summary/lock encoder provenance differs"
        )

    epochs = _integer(
        training.get("epochs"),
        source=f"{spec.name}.summary.training.epochs",
        minimum=1,
    )
    expected_epochs = list(range(epochs + 1))
    observed_epochs: list[int] = []
    validated_records: list[Mapping[str, Any]] = []
    for index, raw_record in enumerate(history):
        record = _mapping(
            raw_record, source=f"{spec.name}.metrics_history[{index}]"
        )
        epoch = _integer(
            record.get("epoch"),
            source=f"{spec.name}.metrics_history[{index}].epoch",
        )
        observed_epochs.append(epoch)
        if epoch == 0 and record.get("train") is not None:
            raise SweepValidationError(
                f"{spec.name}: epoch 0 train record must be null"
            )
        if epoch > 0 and not isinstance(record.get("train"), Mapping):
            raise SweepValidationError(
                f"{spec.name}: trained epoch {epoch} lacks train metrics"
            )
        _validate_metric_block(
            record.get("dev"),
            source=f"{spec.name}.metrics_history[{index}].dev",
            require_by_sensor=True,
        )
        validated_records.append(record)
    if observed_epochs != expected_epochs:
        raise SweepValidationError(
            f"{spec.name}: expected epochs {expected_epochs}, "
            f"got {observed_epochs}"
        )

    expected_best = _expected_best(
        validated_records,
        selection_metric,
        source=f"{spec.name}.metrics_history",
    )
    summary_best = _mapping(
        summary.get("best"), source=f"{spec.name}.summary.best"
    )
    _same(
        summary_best,
        expected_best,
        source=f"{spec.name}: summary best vs dev argmax",
    )
    best_epoch = _integer(
        expected_best.get("epoch"), source=f"{spec.name}.best.epoch"
    )
    best_metrics = _mapping(
        expected_best.get("dev"), source=f"{spec.name}.best.dev"
    )
    epoch0_metrics = _mapping(
        validated_records[0].get("dev"), source=f"{spec.name}.epoch0.dev"
    )
    lock_best_epoch = _integer(
        lock.get("best_epoch"),
        source=f"{spec.name}.selection_lock.best_epoch",
    )
    status_best_epoch = _integer(
        status.get("best_epoch"),
        source=f"{spec.name}.run_status.best_epoch",
    )
    if lock_best_epoch != best_epoch or status_best_epoch != best_epoch:
        raise SweepValidationError(
            f"{spec.name}: selected epoch differs across artifacts"
        )
    _same(
        status.get("best_dev_metrics"),
        best_metrics,
        source=f"{spec.name}: status best metrics vs history",
    )

    selection_score = _finite_number(
        lock.get("selection_score"),
        source=f"{spec.name}.selection_lock.selection_score",
    )
    if selection_score != float(best_metrics[selection_metric]):
        raise SweepValidationError(
            f"{spec.name}: locked score differs from selected dev metric"
        )
    locked_threshold = _unit_metric(
        lock.get("locked_threshold"),
        source=f"{spec.name}.selection_lock.locked_threshold",
    )
    if locked_threshold != float(best_metrics["best_binary_f1_threshold"]):
        raise SweepValidationError(
            f"{spec.name}: locked threshold differs from selected dev threshold"
        )
    status_threshold = _unit_metric(
        status.get("locked_threshold"),
        source=f"{spec.name}.run_status.locked_threshold",
    )
    if status_threshold != locked_threshold:
        raise SweepValidationError(
            f"{spec.name}: status/selection-lock threshold differs"
        )

    checkpoint = Path(str(lock.get("checkpoint", ""))).expanduser()
    expected_checkpoint = run_dir / "checkpoint_best.pth"
    if checkpoint.absolute() != expected_checkpoint.absolute():
        raise SweepValidationError(
            f"{spec.name}: lock checkpoint does not name this run's "
            "checkpoint_best.pth"
        )
    checkpoint_sha = str(lock.get("checkpoint_sha256", ""))
    if len(checkpoint_sha) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha
    ):
        raise SweepValidationError(
            f"{spec.name}: malformed checkpoint SHA-256"
        )

    return {
        "run": spec.name,
        "base_mode": spec.base_mode,
        "arm": spec.arm,
        "learning_rate": learning_rate,
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "selection_score": selection_score,
        "locked_threshold": locked_threshold,
        "epoch0": epoch0_metrics,
        "best": best_metrics,
        "configuration": {
            "model": model,
            "training": training,
            "model_config": lock.get("model_config"),
            "train_rows": summary.get("train_rows"),
            "dev_rows": summary.get("dev_rows"),
            "encoder": summary.get("encoder"),
            "train_manifest": lock.get("train_manifest"),
            "dev_manifest": lock.get("dev_manifest"),
        },
        "artifacts": {
            filename: {
                "path": str((run_dir / filename).absolute()),
                "sha256": digests[filename],
            }
            for filename in INPUT_FILENAMES
        },
    }


def _validate_matched_protocol(runs: Sequence[Mapping[str, Any]]) -> None:
    if len(runs) != len(RUN_SPECS):
        raise SweepValidationError(
            f"expected {len(RUN_SPECS)} runs, got {len(runs)}"
        )
    first = runs[0]
    invariant_training_keys = (
        "seed",
        "epochs",
        "batch_size",
        "sensor_aux_weight",
        "axis_aux_weight",
        "selection_metric",
    )
    invariant_model_keys = (
        "model_dim",
        "num_heads",
        "temporal_depth",
        "parameter_signature",
        "initial_state_sha256",
    )
    for run in runs[1:]:
        for key in invariant_training_keys:
            _same(
                run["configuration"]["training"].get(key),
                first["configuration"]["training"].get(key),
                source=f"matched protocol: training.{key}",
            )
        for key in invariant_model_keys:
            _same(
                run["configuration"]["model"].get(key),
                first["configuration"]["model"].get(key),
                source=f"matched protocol: model.{key}",
            )
        for key in (
            "train_rows",
            "dev_rows",
            "encoder",
            "train_manifest",
            "dev_manifest",
            "model_config",
        ):
            _same(
                run["configuration"].get(key),
                first["configuration"].get(key),
                source=f"matched protocol: {key}",
            )

    epoch0_by_base: dict[str, Any] = {}
    for run in runs:
        base = str(run["base_mode"])
        if base not in epoch0_by_base:
            epoch0_by_base[base] = run["epoch0"]
        else:
            _same(
                run["epoch0"],
                epoch0_by_base[base],
                source=f"epoch-0 exact-base consistency for {base}",
            )


def _compute_deltas(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            str(run["base_mode"]),
            float(run["learning_rate"]),
            str(run["arm"]),
        ): run
        for run in runs
    }
    deltas: list[dict[str, Any]] = []
    for candidate in runs:
        if candidate["arm"] == "current_only":
            continue
        key = (
            str(candidate["base_mode"]),
            float(candidate["learning_rate"]),
            "current_only",
        )
        control = indexed.get(key)
        if control is None:
            raise SweepValidationError(
                f"{candidate['run']}: missing same-base/LR current-only control"
            )
        metric_delta = {
            metric: float(candidate["best"][metric])
            - float(control["best"][metric])
            for metric in DELTA_METRICS
        }
        selection_metric = str(candidate["selection_metric"])
        deltas.append(
            {
                "candidate_run": candidate["run"],
                "control_run": control["run"],
                "base_mode": candidate["base_mode"],
                "learning_rate": candidate["learning_rate"],
                "candidate_arm": candidate["arm"],
                "selection_metric": selection_metric,
                "selection_metric_delta": (
                    float(candidate["best"][selection_metric])
                    - float(control["best"][selection_metric])
                ),
                "metric_delta": metric_delta,
            }
        )
    return deltas


def _winner(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selection_metrics = {str(run["selection_metric"]) for run in runs}
    if selection_metrics != {EXPECTED_SELECTION_METRIC}:
        raise SweepValidationError(
            f"selection metrics differ across runs: {selection_metrics}"
        )
    best_score = max(float(run["selection_score"]) for run in runs)
    tied = [
        str(run["run"])
        for run in runs
        if math.isclose(
            float(run["selection_score"]),
            best_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    recommended = tied[0]
    winner_run = next(run for run in runs if run["run"] == recommended)
    return {
        "recommended_run": recommended,
        "tied_runs": tied,
        "selection_metric": EXPECTED_SELECTION_METRIC,
        "selection_score": best_score,
        "best_epoch": winner_run["best_epoch"],
        "locked_threshold": winner_run["locked_threshold"],
        "basis": (
            "maximum development selection metric only; no test or sealed-test "
            "artifact was accepted or read"
        ),
        "tie_break": (
            "first run in the predeclared sweep order"
            if len(tied) > 1
            else None
        ),
        "recommendation_scope": (
            "candidate for the single later locked evaluation, not a test "
            "result or a leakage-free SOTA claim"
        ),
    }


def summarize_sweep(heads_root: Path | str) -> dict[str, Any]:
    root = Path(heads_root).expanduser().absolute()
    if root.is_symlink():
        raise SweepValidationError(f"{root}: symlink sweep roots are rejected")
    if not root.is_dir():
        raise SweepValidationError(f"{root}: heads root is missing")
    runs = [_validate_run(spec, root / spec.name) for spec in RUN_SPECS]
    _validate_matched_protocol(runs)
    deltas = _compute_deltas(runs)
    winner = _winner(runs)
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "formal legacy-360 ten-run development-only head sweep",
        "heads_root": str(root),
        "run_count": len(runs),
        "sealed_test_read": False,
        "sealed_test_evaluations": 0,
        "input_contract": {
            "per_run_files": list(INPUT_FILENAMES),
            "test_inputs_supported": False,
            "failed_or_incomplete_runs_accepted": False,
            "all_inputs_validated_before_report_write": True,
        },
        "winner_recommendation": winner,
        "runs": runs,
        "matched_deltas_vs_current_only": deltas,
    }


def _csv_run_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    delta_by_candidate = {
        str(delta["candidate_run"]): delta
        for delta in report["matched_deltas_vs_current_only"]
    }
    rows: list[dict[str, Any]] = []
    for run in report["runs"]:
        epoch0 = run["epoch0"]
        best = run["best"]
        row: dict[str, Any] = {
            "run": run["run"],
            "base_mode": run["base_mode"],
            "arm": run["arm"],
            "learning_rate": run["learning_rate"],
            "selection_metric": run["selection_metric"],
            "epoch0_binary_f1_at_0_5": epoch0["binary_f1_at_0_5"],
            "epoch0_macro_f1_at_0_5": epoch0["macro_f1_at_0_5"],
            "epoch0_best_binary_f1": epoch0["best_binary_f1"],
            "epoch0_best_binary_f1_threshold": epoch0[
                "best_binary_f1_threshold"
            ],
            "epoch0_best_macro_f1_at_binary_threshold": epoch0[
                "best_macro_f1_at_binary_threshold"
            ],
            "epoch0_ap": epoch0["ap"],
            "epoch0_auc": epoch0["auc"],
            "best_epoch": run["best_epoch"],
            "best_binary_f1_at_0_5": best["binary_f1_at_0_5"],
            "best_macro_f1_at_0_5": best["macro_f1_at_0_5"],
            "best_binary_f1": best["best_binary_f1"],
            "best_binary_f1_threshold": best[
                "best_binary_f1_threshold"
            ],
            "best_macro_f1_at_binary_threshold": best[
                "best_macro_f1_at_binary_threshold"
            ],
            "best_ap": best["ap"],
            "best_auc": best["auc"],
            "selection_delta_vs_epoch0": (
                float(best[run["selection_metric"]])
                - float(epoch0[run["selection_metric"]])
            ),
            "control_run": "",
            "delta_vs_current_selection_metric": "",
            "epoch0_by_sensor_json": _canonical(epoch0["by_sensor"]),
            "best_by_sensor_json": _canonical(best["by_sensor"]),
        }
        delta = delta_by_candidate.get(str(run["run"]))
        if delta is not None:
            row["control_run"] = delta["control_run"]
            row["delta_vs_current_selection_metric"] = delta[
                "selection_metric_delta"
            ]
        for sensor in SENSORS:
            for prefix, metrics in (("epoch0", epoch0), ("best", best)):
                sensor_metrics = metrics["by_sensor"][sensor]
                for metric in DISPLAY_METRICS:
                    row[f"{prefix}_{sensor}_{metric}"] = sensor_metrics[metric]
        rows.append(row)
    return rows


def _csv_delta_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for delta in report["matched_deltas_vs_current_only"]:
        row = {
            "candidate_run": delta["candidate_run"],
            "control_run": delta["control_run"],
            "base_mode": delta["base_mode"],
            "learning_rate": delta["learning_rate"],
            "candidate_arm": delta["candidate_arm"],
            "selection_metric": delta["selection_metric"],
            "selection_metric_delta": delta["selection_metric_delta"],
        }
        row.update(
            {
                f"delta_{metric}": value
                for metric, value in delta["metric_delta"].items()
            }
        )
        rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown(report: Mapping[str, Any]) -> str:
    winner = report["winner_recommendation"]
    lines = [
        "# Legacy 360 m two-axis head sweep: development-only summary",
        "",
        (
            "**Guardrail:** this report read only the four declared development "
            "artifacts per run. No test or sealed-test input is supported. "
            "All ten runs passed completion, no-test-use, and cross-artifact "
            "consistency checks before this file was written."
        ),
        "",
        "## Development-selected candidate",
        "",
        f"- Run: `{winner['recommended_run']}`",
        (
            f"- Selection: `{winner['selection_metric']}` = "
            f"{_fmt(winner['selection_score'])}, epoch "
            f"{winner['best_epoch']}, locked development threshold "
            f"{_fmt(winner['locked_threshold'])}"
        ),
        f"- Scope: {winner['recommendation_scope']}",
    ]
    if len(winner["tied_runs"]) > 1:
        lines.extend(
            [
                (
                    "- Exact/near tie (1e-12 tolerance): "
                    + ", ".join(f"`{name}`" for name in winner["tied_runs"])
                ),
                f"- Tie break: {winner['tie_break']}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Runs",
            "",
            (
                "| Run | Base | Arm | LR | Epoch-0 F1@0.5 | Best epoch | "
                "F1@0.5 | Dev-best F1 | Threshold | Macro F1@0.5 | "
                "Macro F1@dev threshold | AP | AUC |"
            ),
            (
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
                "---:|---:|---:|"
            ),
        ]
    )
    for run in report["runs"]:
        base = run["epoch0"]
        best = run["best"]
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{run['run']}`",
                    str(run["base_mode"]),
                    str(run["arm"]),
                    _fmt(run["learning_rate"]),
                    _fmt(base["binary_f1_at_0_5"]),
                    str(run["best_epoch"]),
                    _fmt(best["binary_f1_at_0_5"]),
                    _fmt(best["best_binary_f1"]),
                    _fmt(best["best_binary_f1_threshold"]),
                    _fmt(best["macro_f1_at_0_5"]),
                    _fmt(best["best_macro_f1_at_binary_threshold"]),
                    _fmt(best["ap"]),
                    _fmt(best["auc"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Selected-epoch sensor-containing strata",
            "",
            (
                "| Run | S2 F1@0.5 | L89 F1@0.5 | EMIT F1@0.5 | "
                "S5P F1@0.5 |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for run in report["runs"]:
        by_sensor = run["best"]["by_sensor"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{run['run']}`",
                    *[
                        _fmt(by_sensor[sensor]["binary_f1_at_0_5"])
                        for sensor in SENSORS
                    ],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Matched two-axis deltas versus current-only",
            "",
            (
                "Each delta compares independently development-selected epochs "
                "at the same base and learning rate."
            ),
            "",
            (
                "| Candidate | Control | Base | LR | Δ selection F1 | "
                "Δ F1@0.5 | Δ macro F1@0.5 | Δ AP | Δ AUC |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for delta in report["matched_deltas_vs_current_only"]:
        values = delta["metric_delta"]
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{delta['candidate_run']}`",
                    f"`{delta['control_run']}`",
                    str(delta["base_mode"]),
                    _fmt(delta["learning_rate"]),
                    _fmt(delta["selection_metric_delta"]),
                    _fmt(values["binary_f1_at_0_5"]),
                    _fmt(values["macro_f1_at_0_5"]),
                    _fmt(values["ap"]),
                    _fmt(values["auc"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "The JSON report retains the complete epoch-0 and selected "
                "`by_sensor` dictionaries plus SHA-256 provenance for every "
                "input artifact. The run CSV flattens the same sensor metrics."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise SweepValidationError("cannot write an empty CSV")
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_reports(
    report: Mapping[str, Any],
    output_dir: Path | str,
    *,
    prefix: str = "two_axis_dev_sweep_summary",
    overwrite: bool = False,
) -> dict[str, Path]:
    if not prefix or any(character in prefix for character in "/\\"):
        raise SweepValidationError("output prefix must be one plain filename")
    destination = Path(output_dir).expanduser().absolute()
    heads_root = Path(str(report["heads_root"])).absolute()
    if _is_within(destination, heads_root):
        raise SweepValidationError(
            "report output must be outside the heads root so run inputs remain "
            "read-only"
        )
    paths = {
        "json": destination / f"{prefix}.json",
        "runs_csv": destination / f"{prefix}_runs.csv",
        "deltas_csv": destination / f"{prefix}_deltas.csv",
        "markdown": destination / f"{prefix}.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise SweepValidationError(
            "refusing to overwrite report artifacts: "
            + ", ".join(map(str, existing))
        )
    destination.mkdir(parents=True, exist_ok=True)
    payloads = {
        "json": (
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
        "runs_csv": _csv_bytes(_csv_run_rows(report)),
        "deltas_csv": _csv_bytes(_csv_delta_rows(report)),
        "markdown": _markdown(report).encode("utf-8"),
    }
    for role, path in paths.items():
        _atomic_bytes(path, payloads[role])
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed audit and dev-only summary of the fixed ten-run "
            "legacy-360 two-axis head sweep. This CLI has no test input."
        )
    )
    parser.add_argument(
        "--heads-root",
        default=(
            "/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/"
            "formal_v2/heads"
        ),
        help="Directory containing the ten predeclared run directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/"
            "formal_v2/reports"
        ),
        help="Report directory; must be outside --heads-root.",
    )
    parser.add_argument(
        "--prefix",
        default="two_axis_dev_sweep_summary",
        help="Plain filename prefix for JSON, Markdown, and CSV outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace prior report outputs, never input run artifacts.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = summarize_sweep(args.heads_root)
    paths = write_reports(
        report,
        args.output_dir,
        prefix=args.prefix,
        overwrite=bool(args.overwrite),
    )
    winner = report["winner_recommendation"]
    print(
        "validated 10/10 dev-only runs; "
        f"candidate={winner['recommended_run']} "
        f"{winner['selection_metric']}={winner['selection_score']:.6f}"
    )
    for role, path in paths.items():
        print(f"{role}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
