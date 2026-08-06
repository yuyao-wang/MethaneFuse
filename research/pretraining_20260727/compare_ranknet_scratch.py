#!/usr/bin/env python3
"""CPU-only promotion gate for RankNet+BCE versus matched BCE scratch.

The checker is intentionally input-read-only.  It reads two training
histories, the RankNet checkpoint, and two same-checkpoint validation
evaluation JSON/CSV pairs.  Metrics are recomputed from the CSV labels and
probabilities; no dataset, imagery, runner, or sealed-test artifact is read.

Exit codes are 0 for pass, 1 for fail, and 2 for pending.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# A gate must never deserialize checkpoint tensors onto a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402


EXPECTED_SENSORS = ("s2", "l89", "emit", "s5p")
OBJECTIVE_SIGNATURE = {
    "objective_version": (
        "sensorwise_weighted_bce_plus_all_pairs_ranknet_v1"
    ),
    "rank_weight": 0.5,
    "rank_temperature": 1.0,
    "pair_scope": "within_sensor_current_batch",
    "pair_selection": "all_positive_negative_pairs",
    "one_class_policy": "connected_zero_rank_loss",
}
FULL_INPUT_CONTRACT = "full_current_recent_seasonal_v1"
CURRENT_ONLY_INPUT_CONTRACT = (
    "current_preserved_recent_seasonal_zeroed_after_normalization_v1"
)
ARTIFACT_TYPE = "ranknet_bce_scratch_gate_decision"
GATE_SPEC = {
    "name": "four-sensor-ranknet-bce-e1-promotion-v1",
    "primary_metric": "macro_mean_over_sensor_average_precision",
    "macro_ap_min_delta": 0.010,
    "minimum_nondecreasing_sensor_ap_count": 3,
    "worst_sensor_ap_min_delta": -0.010,
    "macro_best_f1_min_delta": -0.005,
    "macro_f1_at_0p5_min_delta": -0.010,
    "minimum_paired_batch_fraction": 0.99,
    "full_temporal_macro_ap_advantage_min": 0.005,
    "minimum_positive_temporal_sensor_count": 2,
    "notes": (
        "One epoch and one seed are a promotion screen.  Every threshold is "
        "preregistered; failure is a hard stop rather than a tuning signal."
    ),
}
CLASSIFICATION_METRICS = {
    "ap": "ap",
    "auroc": "auroc",
    "best_f1": "f1",
    "f1_at_0p5": "f1_0p5",
}
EVALUATION_NUMERIC_FIELDS = (
    "positive_rate",
    "ap",
    "auroc",
    "f1",
    "f1_0p5",
    "threshold",
    "probability_min",
    "probability_max",
    "probability_mean",
    "probability_std",
)
EVALUATION_COUNT_FIELDS = (
    "samples",
    "predicted_positive_at_0p5",
    "predicted_positive_at_best",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a globally purged RankNet+BCE run with matched BCE "
            "scratch and audit same-checkpoint temporal use."
        )
    )
    parser.add_argument("--scratch-history", type=Path)
    parser.add_argument("--ranknet-history", type=Path)
    parser.add_argument("--ranknet-checkpoint", type=Path)
    parser.add_argument("--full-temporal-evaluation", type=Path)
    parser.add_argument("--current-only-evaluation", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic CPU-only pass/fail/pending tests.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def add_check(
    checks: Dict[str, Dict[str, Any]],
    name: str,
    passed: bool,
    *,
    observed: Any,
    expected: Any,
    detail: str = "",
) -> None:
    checks[name] = {
        "pass": bool(passed),
        "observed": observed,
        "expected": expected,
        "detail": detail,
    }


def input_records(paths: Mapping[str, Path]) -> Dict[str, Any]:
    return {
        role: {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            **({"sha256": sha256_file(path)} if path.is_file() else {}),
        }
        for role, path in paths.items()
    }


def pending_result(
    paths: Mapping[str, Path],
    reasons: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "created_utc": utc_now(),
        "status": "pending",
        "preregistered_gate": GATE_SPEC,
        "inputs": input_records(paths),
        "pending_reasons": list(reasons),
        "checks": {},
        "metrics": None,
        "gate_decision": None,
    }


def fail_result(
    paths: Mapping[str, Path],
    error: Exception,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "created_utc": utc_now(),
        "status": "fail",
        "preregistered_gate": GATE_SPEC,
        "inputs": input_records(paths),
        "errors": [f"{type(error).__name__}: {error}"],
        "checks": {},
        "metrics": None,
        "gate_decision": {
            "pass": False,
            "decision": "do_not_promote",
            "failed_criteria": ["artifact_parse_or_schema_error"],
        },
    }


def last_epoch(history: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    epochs = history.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError(f"{role} history has no completed epochs.")
    if not all(isinstance(record, Mapping) for record in epochs):
        raise ValueError(f"{role} history contains a non-object epoch.")
    return max(epochs, key=lambda record: int(record.get("epoch", -1)))


def canonical_supervised_resume(signature: Any) -> Any:
    """Remove only the registered RankNet objective and its schema bump."""

    if not isinstance(signature, Mapping):
        return signature
    canonical = copy.deepcopy(dict(signature))
    canonical.pop("schema_version", None)
    canonical.pop("supervised_objective", None)
    return canonical


def resolve_prediction_csv(
    evaluation_path: Path,
    evaluation: Mapping[str, Any],
) -> Path:
    raw_path = evaluation.get("predictions_csv")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(
            f"Evaluation lacks predictions_csv: {evaluation_path}"
        )
    path = Path(raw_path)
    if not path.is_absolute():
        path = evaluation_path.parent / path
    return path.resolve()


def load_prediction_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = ["sensor", "row_index", "label", "probability"]
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"{path} columns={reader.fieldnames}, "
                f"expected={expected_columns}"
            )
        records: List[Dict[str, Any]] = []
        for line_number, row in enumerate(reader, start=2):
            sensor = row["sensor"]
            if sensor not in EXPECTED_SENSORS:
                raise ValueError(
                    f"{path}:{line_number} invalid sensor={sensor!r}"
                )
            try:
                row_index = int(row["row_index"])
                label_float = float(row["label"])
                probability = float(row["probability"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number} cannot parse row={row}"
                ) from error
            if str(row_index) != str(row["row_index"]):
                raise ValueError(
                    f"{path}:{line_number} row_index is not canonical integer"
                )
            if label_float not in (0.0, 1.0):
                raise ValueError(
                    f"{path}:{line_number} non-binary label={label_float}"
                )
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"{path}:{line_number} invalid probability={probability}"
                )
            records.append(
                {
                    "sensor": sensor,
                    "row_index": row_index,
                    "label": int(label_float),
                    "probability": probability,
                }
            )
    if not records:
        raise ValueError(f"Prediction CSV is empty: {path}")
    return records


def binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    true_positive = int(np.sum((labels == 1) & predictions))
    false_positive = int(np.sum((labels == 0) & predictions))
    false_negative = int(np.sum((labels == 1) & (~predictions)))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2.0 * true_positive / denominator


def compute_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> Dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if labels_array.size == 0:
        raise ValueError("Cannot compute metrics for an empty sensor.")
    if np.unique(labels_array).size != 2:
        raise ValueError("Full validation must contain both classes.")
    thresholds = np.unique(
        np.concatenate(
            (
                np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
                probabilities_array,
            )
        )
    )
    f1_values = np.asarray(
        [
            binary_f1(labels_array, probabilities_array >= threshold)
            for threshold in thresholds
        ],
        dtype=np.float64,
    )
    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])
    predictions_at_0p5 = probabilities_array >= 0.5
    predictions_at_best = probabilities_array >= best_threshold
    return {
        "samples": int(labels_array.size),
        "positive_rate": float(labels_array.mean()),
        "ap": float(
            average_precision_score(labels_array, probabilities_array)
        ),
        "auroc": float(roc_auc_score(labels_array, probabilities_array)),
        "f1": float(f1_values[best_index]),
        "f1_0p5": float(binary_f1(labels_array, predictions_at_0p5)),
        "threshold": best_threshold,
        "probability_min": float(probabilities_array.min()),
        "probability_max": float(probabilities_array.max()),
        "probability_mean": float(probabilities_array.mean()),
        "probability_std": float(probabilities_array.std()),
        "predicted_positive_at_0p5": int(predictions_at_0p5.sum()),
        "predicted_positive_at_best": int(predictions_at_best.sum()),
    }


def validate_prediction_order(
    records: Sequence[Mapping[str, Any]],
    role: str,
) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {
        sensor: [] for sensor in EXPECTED_SENSORS
    }
    expected_sensor_indices = {
        sensor: index for index, sensor in enumerate(EXPECTED_SENSORS)
    }
    previous_sensor_index = -1
    for record in records:
        sensor = str(record["sensor"])
        sensor_index = expected_sensor_indices[sensor]
        if sensor_index < previous_sensor_index:
            raise ValueError(
                f"{role} CSV sensor blocks are not in canonical order."
            )
        previous_sensor_index = sensor_index
        grouped[sensor].append(record)
    for sensor, sensor_records in grouped.items():
        observed_indices = [
            int(record["row_index"]) for record in sensor_records
        ]
        expected_indices = list(range(len(sensor_records)))
        if observed_indices != expected_indices:
            raise ValueError(
                f"{role}/{sensor} row_index must increase from zero: "
                f"observed_head={observed_indices[:10]}"
            )
    return grouped


def recompute_evaluation(
    records: Sequence[Mapping[str, Any]],
    role: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    grouped = validate_prediction_order(records, role)
    by_sensor: Dict[str, Dict[str, Any]] = {}
    for sensor in EXPECTED_SENSORS:
        by_sensor[sensor] = compute_metrics(
            [int(record["label"]) for record in grouped[sensor]],
            [float(record["probability"]) for record in grouped[sensor]],
        )
    macro = {
        metric: float(
            np.mean([by_sensor[sensor][metric] for sensor in EXPECTED_SENSORS])
        )
        for metric in ("ap", "auroc", "f1", "f1_0p5")
    }
    return by_sensor, macro


def metric_mapping_matches(
    observed: Any,
    recomputed: Mapping[str, Any],
    *,
    tolerance: float = 1e-9,
) -> Tuple[bool, Dict[str, Any]]:
    comparison: Dict[str, Any] = {}
    if not isinstance(observed, Mapping):
        return False, {"error": "logged metrics are not a mapping"}
    passed = True
    for key in EVALUATION_COUNT_FIELDS:
        logged = observed.get(key)
        expected = recomputed[key]
        key_pass = (
            isinstance(logged, int)
            and not isinstance(logged, bool)
            and logged == expected
        )
        passed = passed and key_pass
        comparison[key] = {
            "logged": logged,
            "recomputed": expected,
            "pass": key_pass,
        }
    for key in EVALUATION_NUMERIC_FIELDS:
        logged = observed.get(key)
        expected = recomputed[key]
        error = (
            abs(float(logged) - float(expected))
            if is_finite_number(logged)
            else None
        )
        key_pass = error is not None and error <= tolerance
        passed = passed and key_pass
        comparison[key] = {
            "logged": logged,
            "recomputed": expected,
            "absolute_error": error,
            "pass": key_pass,
        }
    return passed, comparison


def extract_history_classification(
    history: Mapping[str, Any],
    role: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Any]]:
    epoch = last_epoch(history, role)
    validation = epoch.get("val")
    logged_macro = epoch.get("macro_over_sensor")
    if not isinstance(validation, Mapping):
        raise ValueError(f"{role} epoch lacks val mapping.")
    if not isinstance(logged_macro, Mapping):
        raise ValueError(f"{role} epoch lacks macro_over_sensor mapping.")
    by_sensor: Dict[str, Dict[str, float]] = {}
    samples: Dict[str, int] = {}
    for sensor in EXPECTED_SENSORS:
        sensor_metrics = validation.get(sensor)
        if not isinstance(sensor_metrics, Mapping):
            raise ValueError(f"{role} validation lacks {sensor}.")
        sample_count = sensor_metrics.get("samples")
        if not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError(
                f"{role}/{sensor} invalid samples={sample_count!r}"
            )
        samples[sensor] = sample_count
        by_sensor[sensor] = {}
        for output_name, source_name in CLASSIFICATION_METRICS.items():
            value = sensor_metrics.get(source_name)
            if not is_finite_number(value):
                raise ValueError(
                    f"{role}/{sensor} invalid {source_name}={value!r}"
                )
            by_sensor[sensor][output_name] = float(value)
    macro = {
        metric: float(
            np.mean([by_sensor[sensor][metric] for sensor in EXPECTED_SENSORS])
        )
        for metric in CLASSIFICATION_METRICS
    }
    macro_comparison: Dict[str, Any] = {}
    for output_name, source_name in CLASSIFICATION_METRICS.items():
        logged = logged_macro.get(source_name)
        if not is_finite_number(logged):
            raise ValueError(
                f"{role} invalid macro {source_name}={logged!r}"
            )
        macro_comparison[output_name] = {
            "logged": float(logged),
            "recomputed": macro[output_name],
            "absolute_error": abs(float(logged) - macro[output_name]),
        }
    return by_sensor, macro, {
        "samples": samples,
        "macro": macro_comparison,
    }


def audit_pair_diagnostics(
    ranknet: Mapping[str, Any],
) -> Tuple[bool, Dict[str, Any], float, List[str]]:
    epoch = last_epoch(ranknet, "ranknet")
    diagnostics = epoch.get("train_objective_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False, {}, 0.0, [
            "train_objective_diagnostics is missing"
        ]
    resume = ranknet.get("resume_signature")
    optimization = (
        resume.get("optimization")
        if isinstance(resume, Mapping)
        else None
    )
    batch_size = (
        optimization.get("batch_size")
        if isinstance(optimization, Mapping)
        else None
    )
    expected_batches = ranknet.get("balanced_rounds_per_epoch")
    failures: List[str] = []
    observed: Dict[str, Any] = {}
    total_batches = 0
    total_paired_batches = 0
    for sensor in EXPECTED_SENSORS:
        record = diagnostics.get(sensor)
        observed[sensor] = record
        if not isinstance(record, Mapping):
            failures.append(f"{sensor}: diagnostics missing")
            continue
        numeric_losses = (
            "total_loss",
            "weighted_bce_loss",
            "ranknet_loss",
        )
        for key in numeric_losses:
            value = record.get(key)
            if not is_finite_number(value) or float(value) < 0.0:
                failures.append(f"{sensor}: invalid {key}={value!r}")
        count_keys = (
            "positive_samples",
            "negative_samples",
            "pair_count",
            "batches",
            "paired_batches",
            "one_class_batches",
        )
        for key in count_keys:
            value = record.get(key)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                failures.append(f"{sensor}: invalid {key}={value!r}")
        if failures and not all(
            isinstance(record.get(key), int)
            and not isinstance(record.get(key), bool)
            for key in count_keys
        ):
            continue
        batches = int(record["batches"])
        paired_batches = int(record["paired_batches"])
        one_class_batches = int(record["one_class_batches"])
        positive_samples = int(record["positive_samples"])
        negative_samples = int(record["negative_samples"])
        pair_count = int(record["pair_count"])
        paired_fraction = record.get("paired_batch_fraction")
        one_class_fraction = record.get("one_class_batch_fraction")
        expected_paired_fraction = (
            paired_batches / batches if batches > 0 else float("nan")
        )
        expected_one_class_fraction = (
            one_class_batches / batches if batches > 0 else float("nan")
        )
        if batches != expected_batches:
            failures.append(
                f"{sensor}: batches={batches}, expected={expected_batches}"
            )
        if paired_batches + one_class_batches != batches:
            failures.append(
                f"{sensor}: paired + one_class != batches"
            )
        if (
            not is_finite_number(paired_fraction)
            or abs(float(paired_fraction) - expected_paired_fraction) > 1e-12
        ):
            failures.append(
                f"{sensor}: paired_batch_fraction mismatch"
            )
        if (
            not is_finite_number(one_class_fraction)
            or abs(
                float(one_class_fraction) - expected_one_class_fraction
            )
            > 1e-12
        ):
            failures.append(
                f"{sensor}: one_class_batch_fraction mismatch"
            )
        total_samples = positive_samples + negative_samples
        if (
            not isinstance(batch_size, int)
            or batch_size <= 0
            or not (batches - 1) * batch_size
            < total_samples
            <= batches * batch_size
        ):
            failures.append(
                f"{sensor}: sample totals are incompatible with at most one "
                "partial final batch"
            )
        if pair_count <= 0:
            failures.append(f"{sensor}: pair_count is not positive")
        # The runner records total/BCE as sample-weighted means and RankNet
        # as a mean over pair-bearing batches.  A partial last batch therefore
        # makes their aggregate arithmetic intentionally non-identical.
        total_batches += batches
        total_paired_batches += paired_batches
    global_fraction = (
        total_paired_batches / total_batches if total_batches > 0 else 0.0
    )
    return not failures, observed, global_fraction, failures


def compare_artifacts(
    *,
    scratch_history_path: Path,
    ranknet_history_path: Path,
    ranknet_checkpoint_path: Path,
    full_temporal_evaluation_path: Path,
    current_only_evaluation_path: Path,
) -> Dict[str, Any]:
    paths: Dict[str, Path] = {
        "scratch_history": Path(scratch_history_path),
        "ranknet_history": Path(ranknet_history_path),
        "ranknet_checkpoint": Path(ranknet_checkpoint_path),
        "full_temporal_evaluation": Path(full_temporal_evaluation_path),
        "current_only_evaluation": Path(current_only_evaluation_path),
    }
    missing = [
        {
            "role": role,
            "path": str(path.resolve()),
            "reason": "file_not_found",
        }
        for role, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        return pending_result(paths, missing)

    try:
        scratch = load_json(paths["scratch_history"])
        ranknet = load_json(paths["ranknet_history"])
        full_evaluation = load_json(paths["full_temporal_evaluation"])
        current_evaluation = load_json(paths["current_only_evaluation"])
    except Exception as error:
        return fail_result(paths, error)

    running_reasons: List[Dict[str, Any]] = []
    for role, payload in (
        ("scratch_history", scratch),
        ("ranknet_history", ranknet),
        ("full_temporal_evaluation", full_evaluation),
        ("current_only_evaluation", current_evaluation),
    ):
        status = payload.get("status")
        if status != "completed":
            running_reasons.append(
                {
                    "role": role,
                    "path": str(paths[role].resolve()),
                    "reason": "artifact_not_completed",
                    "observed_status": status,
                }
            )
    if running_reasons:
        return pending_result(paths, running_reasons)

    try:
        full_csv_path = resolve_prediction_csv(
            paths["full_temporal_evaluation"],
            full_evaluation,
        )
        current_csv_path = resolve_prediction_csv(
            paths["current_only_evaluation"],
            current_evaluation,
        )
    except Exception as error:
        return fail_result(paths, error)
    paths["full_temporal_predictions"] = full_csv_path
    paths["current_only_predictions"] = current_csv_path
    missing_predictions = [
        {
            "role": role,
            "path": str(path),
            "reason": "file_not_found",
        }
        for role, path in (
            ("full_temporal_predictions", full_csv_path),
            ("current_only_predictions", current_csv_path),
        )
        if not path.is_file()
    ]
    if missing_predictions:
        return pending_result(paths, missing_predictions)

    try:
        full_records = load_prediction_csv(full_csv_path)
        current_records = load_prediction_csv(current_csv_path)
        full_by_sensor, full_macro = recompute_evaluation(
            full_records,
            "full_temporal",
        )
        current_by_sensor, current_macro = recompute_evaluation(
            current_records,
            "current_only",
        )
        checkpoint = torch.load(
            paths["ranknet_checkpoint"],
            map_location=torch.device("cpu"),
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError("RankNet checkpoint is not a mapping.")
        scratch_by_sensor, scratch_macro, scratch_audit = (
            extract_history_classification(scratch, "scratch")
        )
        ranknet_by_sensor, ranknet_macro, ranknet_audit = (
            extract_history_classification(ranknet, "ranknet")
        )
    except Exception as error:
        return fail_result(paths, error)

    checks: Dict[str, Dict[str, Any]] = {}
    inputs = input_records(paths)
    histories = {"scratch": scratch, "ranknet": ranknet}
    add_check(
        checks,
        "history_status_completed",
        all(history.get("status") == "completed" for history in histories.values()),
        observed={role: history.get("status") for role, history in histories.items()},
        expected={"scratch": "completed", "ranknet": "completed"},
    )
    add_check(
        checks,
        "history_modes_and_sharing",
        all(
            history.get("mode") == "supervised"
            and history.get("sharing") == "shared"
            and history.get("sensors") == list(EXPECTED_SENSORS)
            for history in histories.values()
        ),
        observed={
            role: {
                "mode": history.get("mode"),
                "sharing": history.get("sharing"),
                "sensors": history.get("sensors"),
            }
            for role, history in histories.items()
        },
        expected={
            role: {
                "mode": "supervised",
                "sharing": "shared",
                "sensors": list(EXPECTED_SENSORS),
            }
            for role in histories
        },
    )
    add_check(
        checks,
        "registered_history_schema_delta",
        scratch.get("schema_version") == 2
        and ranknet.get("schema_version") == 4,
        observed={
            "scratch": scratch.get("schema_version"),
            "ranknet": ranknet.get("schema_version"),
        },
        expected={"scratch": 2, "ranknet": 4},
    )
    objective_records = {
        "scratch_history": scratch.get("supervised_objective"),
        "scratch_resume": (
            scratch.get("resume_signature", {}).get("supervised_objective")
            if isinstance(scratch.get("resume_signature"), Mapping)
            else None
        ),
        "ranknet_history": ranknet.get("supervised_objective"),
        "ranknet_resume": (
            ranknet.get("resume_signature", {}).get("supervised_objective")
            if isinstance(ranknet.get("resume_signature"), Mapping)
            else None
        ),
        "checkpoint": checkpoint.get("supervised_objective"),
    }
    add_check(
        checks,
        "ranknet_objective_exact_and_scratch_bce_only",
        objective_records["scratch_history"] is None
        and objective_records["scratch_resume"] is None
        and objective_records["ranknet_history"] == OBJECTIVE_SIGNATURE
        and objective_records["ranknet_resume"] == OBJECTIVE_SIGNATURE
        and objective_records["checkpoint"] == OBJECTIVE_SIGNATURE,
        observed=objective_records,
        expected={
            "scratch_history": None,
            "scratch_resume": None,
            "ranknet_history": OBJECTIVE_SIGNATURE,
            "ranknet_resume": OBJECTIVE_SIGNATURE,
            "checkpoint": OBJECTIVE_SIGNATURE,
        },
    )
    scratch_resume = scratch.get("resume_signature")
    ranknet_resume = ranknet.get("resume_signature")
    resume_schemas = {
        "scratch": (
            scratch_resume.get("schema_version")
            if isinstance(scratch_resume, Mapping)
            else None
        ),
        "ranknet": (
            ranknet_resume.get("schema_version")
            if isinstance(ranknet_resume, Mapping)
            else None
        ),
    }
    add_check(
        checks,
        "registered_resume_schema_delta",
        resume_schemas == {"scratch": 1, "ranknet": 2},
        observed=resume_schemas,
        expected={"scratch": 1, "ranknet": 2},
    )
    canonical_resumes = {
        "scratch": canonical_supervised_resume(scratch_resume),
        "ranknet": canonical_supervised_resume(ranknet_resume),
    }
    add_check(
        checks,
        "matched_resume_after_registered_objective_projection",
        isinstance(canonical_resumes["scratch"], Mapping)
        and canonical_resumes["scratch"] == canonical_resumes["ranknet"],
        observed={
            role: json_fingerprint(value)
            for role, value in canonical_resumes.items()
        },
        expected=(
            "identical after removing only supervised_objective and its "
            "registered schema bump"
        ),
    )
    signatures = {
        name: {
            role: history.get(name)
            for role, history in histories.items()
        }
        for name in ("data_signature", "encoder_signature")
    }
    for signature_name, values in signatures.items():
        add_check(
            checks,
            f"matched_{signature_name}",
            isinstance(values["scratch"], Mapping)
            and values["scratch"] == values["ranknet"],
            observed={
                role: json_fingerprint(value)
                for role, value in values.items()
            },
            expected="scratch == ranknet",
        )
    overlap_audit = {
        role: {
            "event_protocol": history.get("event_protocol", {}).get(
                "global_train_val_overlap_after"
            ),
            "data_signature": history.get("data_signature", {}).get(
                "global_train_val_overlap_after"
            ),
        }
        for role, history in histories.items()
    }
    add_check(
        checks,
        "global_event_overlap_zero",
        all(
            audit["event_protocol"] == 0
            and audit["data_signature"] == 0
            for audit in overlap_audit.values()
        ),
        observed=overlap_audit,
        expected={
            role: {"event_protocol": 0, "data_signature": 0}
            for role in histories
        },
    )
    budgets = {
        role: {
            "rounds": history.get("balanced_rounds_per_epoch"),
            "optimizer_steps": history.get("optimizer_steps_per_epoch"),
            "model_parameters": history.get("model_parameters"),
            "trainable_parameters": history.get("trainable_parameters"),
        }
        for role, history in histories.items()
    }
    add_check(
        checks,
        "matched_model_and_training_budget",
        budgets["scratch"] == budgets["ranknet"]
        and budgets["ranknet"]["rounds"] == 157
        and budgets["ranknet"]["optimizer_steps"] == 628,
        observed=budgets,
        expected=(
            "scratch == ranknet with 157 rounds and 628 optimizer steps"
        ),
    )
    configs = {
        role: (
            history.get("config")
            if isinstance(history.get("config"), Mapping)
            else {}
        )
        for role, history in histories.items()
    }
    run_controls = {
        role: {
            "epochs": config.get("epochs"),
            "seed": config.get("seed"),
            "max_val_batches": config.get("max_val_batches"),
            "init_checkpoint": config.get("init_checkpoint"),
            "resume": config.get("resume"),
            "completed_epochs": [
                record.get("epoch")
                for record in histories[role].get("epochs", [])
                if isinstance(record, Mapping)
            ],
            "has_init_report": isinstance(
                histories[role].get("init_checkpoint"),
                Mapping,
            ),
        }
        for role, config in configs.items()
    }
    expected_controls = {
        role: {
            "epochs": 1,
            "seed": 20260727,
            "max_val_batches": 0,
            "init_checkpoint": None,
            "resume": None,
            "completed_epochs": [0],
            "has_init_report": False,
        }
        for role in histories
    }
    add_check(
        checks,
        "matched_one_epoch_initialization_and_full_validation",
        run_controls == expected_controls,
        observed=run_controls,
        expected=expected_controls,
    )
    add_check(
        checks,
        "matched_initial_classification_head",
        is_sha256(scratch.get("initial_head_sha256"))
        and scratch.get("initial_head_sha256")
        == ranknet.get("initial_head_sha256"),
        observed={
            role: history.get("initial_head_sha256")
            for role, history in histories.items()
        },
        expected="same nonempty SHA-256",
    )
    macro_audit = {
        "scratch": scratch_audit["macro"],
        "ranknet": ranknet_audit["macro"],
    }
    add_check(
        checks,
        "history_macro_metrics_match_sensor_recomputation",
        all(
            metric["absolute_error"] <= 1e-9
            for role in macro_audit.values()
            for metric in role.values()
        ),
        observed=macro_audit,
        expected="absolute error <= 1e-9",
    )
    add_check(
        checks,
        "matched_full_validation_sample_counts",
        scratch_audit["samples"] == ranknet_audit["samples"],
        observed={
            "scratch": scratch_audit["samples"],
            "ranknet": ranknet_audit["samples"],
        },
        expected="identical positive samples per sensor",
    )
    add_check(
        checks,
        "checkpoint_protocol_identity",
        checkpoint.get("mode") == "supervised"
        and checkpoint.get("epoch") == 0
        and isinstance(checkpoint.get("model"), Mapping)
        and checkpoint.get("data_signature") == ranknet.get("data_signature")
        and checkpoint.get("encoder_signature")
        == ranknet.get("encoder_signature")
        and checkpoint.get("resume_signature")
        == ranknet.get("resume_signature"),
        observed={
            "mode": checkpoint.get("mode"),
            "epoch": checkpoint.get("epoch"),
            "has_model": isinstance(checkpoint.get("model"), Mapping),
            "data_signature": json_fingerprint(
                checkpoint.get("data_signature")
            ),
            "encoder_signature": json_fingerprint(
                checkpoint.get("encoder_signature")
            ),
            "resume_signature": json_fingerprint(
                checkpoint.get("resume_signature")
            ),
        },
        expected={
            "mode": "supervised",
            "epoch": 0,
            "has_model": True,
            "data_signature": json_fingerprint(
                ranknet.get("data_signature")
            ),
            "encoder_signature": json_fingerprint(
                ranknet.get("encoder_signature")
            ),
            "resume_signature": json_fingerprint(
                ranknet.get("resume_signature")
            ),
        },
    )

    ranknet_checkpoint_sha = inputs["ranknet_checkpoint"]["sha256"]
    ranknet_history_sha = inputs["ranknet_history"]["sha256"]
    event_protocol_fingerprint = ranknet.get("event_protocol", {}).get(
        "fingerprint"
    )
    evaluation_integrity: Dict[str, Any] = {}
    for role, evaluation, evaluation_path, prediction_path in (
        (
            "full_temporal",
            full_evaluation,
            paths["full_temporal_evaluation"],
            full_csv_path,
        ),
        (
            "current_only",
            current_evaluation,
            paths["current_only_evaluation"],
            current_csv_path,
        ),
    ):
        expected_mode = role
        expected_input = (
            FULL_INPUT_CONTRACT
            if role == "full_temporal"
            else CURRENT_ONLY_INPUT_CONTRACT
        )
        source_path = evaluation.get("source_history_path")
        checkpoint_path = evaluation.get("checkpoint_path")
        evaluation_integrity[role] = {
            "artifact_type": evaluation.get("artifact_type"),
            "schema_version": evaluation.get("schema_version"),
            "status": evaluation.get("status"),
            "evaluation_mode": evaluation.get("evaluation_mode"),
            "input_contract": evaluation.get("input_contract"),
            "checkpoint_path_matches": (
                isinstance(checkpoint_path, str)
                and Path(checkpoint_path).resolve()
                == paths["ranknet_checkpoint"].resolve()
            ),
            "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
            "checkpoint_epoch": evaluation.get("checkpoint_epoch"),
            "source_history_path_matches": (
                isinstance(source_path, str)
                and Path(source_path).resolve()
                == paths["ranknet_history"].resolve()
            ),
            "source_history_sha256": evaluation.get(
                "source_history_sha256"
            ),
            "predictions_csv_path_matches": (
                prediction_path.resolve()
                == resolve_prediction_csv(evaluation_path, evaluation)
            ),
            "predictions_csv_sha256": evaluation.get(
                "predictions_csv_sha256"
            ),
            "prediction_rows": evaluation.get("prediction_rows"),
            "objective_matches": (
                evaluation.get("supervised_objective")
                == OBJECTIVE_SIGNATURE
            ),
            "encoder_signature_matches": (
                evaluation.get("encoder_signature")
                == ranknet.get("encoder_signature")
            ),
            "data_signature_matches": (
                evaluation.get("data_signature")
                == ranknet.get("data_signature")
            ),
            "resume_signature_matches": (
                evaluation.get("resume_signature")
                == ranknet.get("resume_signature")
            ),
            "encoder_fingerprint_matches": (
                evaluation.get("encoder_signature_fingerprint")
                == json_fingerprint(ranknet.get("encoder_signature"))
            ),
            "data_fingerprint_matches": (
                evaluation.get("data_signature_fingerprint")
                == json_fingerprint(ranknet.get("data_signature"))
            ),
            "resume_fingerprint_matches": (
                evaluation.get("resume_signature_fingerprint")
                == json_fingerprint(ranknet.get("resume_signature"))
            ),
            "event_protocol_fingerprint": evaluation.get(
                "event_protocol_fingerprint"
            ),
            "runtime_is_mapping": isinstance(
                evaluation.get("runtime"),
                Mapping,
            ),
            "normalization_stats_path": evaluation.get(
                "normalization_stats_path"
            ),
            "normalization_stats_sha256": evaluation.get(
                "normalization_stats_sha256"
            ),
            "full_sample_counts": evaluation.get("full_sample_counts"),
            "expected": {
                "evaluation_mode": expected_mode,
                "input_contract": expected_input,
            },
        }
    evaluation_integrity_pass = True
    for role, audit in evaluation_integrity.items():
        expected_mode = audit["expected"]["evaluation_mode"]
        expected_input = audit["expected"]["input_contract"]
        evaluation_integrity_pass = evaluation_integrity_pass and (
            audit["artifact_type"]
            == "ranknet_validation_evaluation_v1"
            and audit["schema_version"] == 1
            and audit["status"] == "completed"
            and audit["evaluation_mode"] == expected_mode
            and audit["input_contract"] == expected_input
            and audit["checkpoint_path_matches"]
            and audit["checkpoint_sha256"] == ranknet_checkpoint_sha
            and audit["checkpoint_epoch"] == 0
            and audit["source_history_path_matches"]
            and audit["source_history_sha256"] == ranknet_history_sha
            and audit["predictions_csv_path_matches"]
            and audit["predictions_csv_sha256"]
            == inputs[f"{role}_predictions"]["sha256"]
            and audit["prediction_rows"]
            == len(full_records if role == "full_temporal" else current_records)
            and audit["objective_matches"]
            and audit["encoder_signature_matches"]
            and audit["data_signature_matches"]
            and audit["resume_signature_matches"]
            and audit["encoder_fingerprint_matches"]
            and audit["data_fingerprint_matches"]
            and audit["resume_fingerprint_matches"]
            and audit["event_protocol_fingerprint"]
            == event_protocol_fingerprint
            and audit["runtime_is_mapping"]
            and isinstance(audit["normalization_stats_path"], str)
            and Path(audit["normalization_stats_path"]).resolve()
            == Path(str(ranknet.get("normalization_stats"))).resolve()
            and audit["normalization_stats_sha256"]
            == ranknet.get("data_signature", {}).get(
                "normalization_stats_sha256"
            )
            and audit["full_sample_counts"] == ranknet_audit["samples"]
        )
    add_check(
        checks,
        "same_checkpoint_evaluation_artifact_integrity",
        evaluation_integrity_pass,
        observed=evaluation_integrity,
        expected=(
            "two completed mode-specific artifacts tied by path/SHA/signatures "
            "to the same epoch-0 RankNet checkpoint and history"
        ),
    )
    full_keys_labels = [
        (record["sensor"], record["row_index"], record["label"])
        for record in full_records
    ]
    current_keys_labels = [
        (record["sensor"], record["row_index"], record["label"])
        for record in current_records
    ]
    add_check(
        checks,
        "full_current_rows_order_and_labels_identical",
        full_keys_labels == current_keys_labels,
        observed={
            "full_rows": len(full_keys_labels),
            "current_rows": len(current_keys_labels),
            "identical": full_keys_labels == current_keys_labels,
        },
        expected="identical sensor,row_index,label sequence",
    )

    evaluation_metric_audits: Dict[str, Any] = {}
    evaluation_metrics_pass = True
    for role, evaluation, recomputed_by_sensor, recomputed_macro in (
        (
            "full_temporal",
            full_evaluation,
            full_by_sensor,
            full_macro,
        ),
        (
            "current_only",
            current_evaluation,
            current_by_sensor,
            current_macro,
        ),
    ):
        logged_by_sensor = evaluation.get("per_sensor")
        logged_macro = evaluation.get("macro_over_sensor")
        role_audit: Dict[str, Any] = {"per_sensor": {}, "macro": {}}
        if not isinstance(logged_by_sensor, Mapping):
            evaluation_metrics_pass = False
            role_audit["error"] = "per_sensor is not a mapping"
        else:
            for sensor in EXPECTED_SENSORS:
                sensor_pass, sensor_audit = metric_mapping_matches(
                    logged_by_sensor.get(sensor),
                    recomputed_by_sensor[sensor],
                )
                evaluation_metrics_pass = (
                    evaluation_metrics_pass and sensor_pass
                )
                role_audit["per_sensor"][sensor] = sensor_audit
        if not isinstance(logged_macro, Mapping):
            evaluation_metrics_pass = False
            role_audit["macro"]["error"] = "not a mapping"
        else:
            for key, expected in recomputed_macro.items():
                logged = logged_macro.get(key)
                error = (
                    abs(float(logged) - expected)
                    if is_finite_number(logged)
                    else None
                )
                key_pass = error is not None and error <= 1e-9
                evaluation_metrics_pass = (
                    evaluation_metrics_pass and key_pass
                )
                role_audit["macro"][key] = {
                    "logged": logged,
                    "recomputed": expected,
                    "absolute_error": error,
                    "pass": key_pass,
                }
        evaluation_metric_audits[role] = role_audit
    add_check(
        checks,
        "evaluation_metrics_recomputed_from_csv",
        evaluation_metrics_pass,
        observed=evaluation_metric_audits,
        expected="all per-sensor and macro absolute errors <= 1e-9",
    )

    full_history_consistency: Dict[str, Any] = {}
    full_history_consistency_pass = True
    ranknet_epoch_val = last_epoch(ranknet, "ranknet").get("val", {})
    for sensor in EXPECTED_SENSORS:
        logged = ranknet_epoch_val.get(sensor, {})
        comparison: Dict[str, Any] = {}
        for source_key in (
            "samples",
            "positive_rate",
            "ap",
            "auroc",
            "f1",
            "f1_0p5",
            "threshold",
        ):
            expected = full_by_sensor[sensor][source_key]
            observed = (
                logged.get(source_key)
                if isinstance(logged, Mapping)
                else None
            )
            if source_key == "samples":
                key_pass = observed == expected
                error = None
            else:
                error = (
                    abs(float(observed) - float(expected))
                    if is_finite_number(observed)
                    else None
                )
                key_pass = error is not None and error <= 1e-9
            full_history_consistency_pass = (
                full_history_consistency_pass and key_pass
            )
            comparison[source_key] = {
                "history": observed,
                "recomputed": expected,
                "absolute_error": error,
                "pass": key_pass,
            }
        full_history_consistency[sensor] = comparison
    add_check(
        checks,
        "ranknet_history_full_validation_matches_csv",
        full_history_consistency_pass,
        observed=full_history_consistency,
        expected="all logged full-validation metrics match within 1e-9",
    )

    pair_audit_pass, pair_audit, paired_fraction, pair_failures = (
        audit_pair_diagnostics(ranknet)
    )
    add_check(
        checks,
        "ranknet_pair_diagnostics_internally_consistent",
        pair_audit_pass,
        observed={
            "per_sensor": pair_audit,
            "global_paired_batch_fraction": paired_fraction,
            "failures": pair_failures,
        },
        expected=(
            "per-sensor finite decomposed losses and exact sample/batch/pair "
            "audit over 157 batches"
        ),
    )

    per_sensor: Dict[str, Any] = {}
    for sensor in EXPECTED_SENSORS:
        per_sensor[sensor] = {
            "scratch": scratch_by_sensor[sensor],
            "ranknet": ranknet_by_sensor[sensor],
            "delta_ranknet_minus_scratch": {
                metric: (
                    ranknet_by_sensor[sensor][metric]
                    - scratch_by_sensor[sensor][metric]
                )
                for metric in CLASSIFICATION_METRICS
            },
            "full_temporal_recomputed": full_by_sensor[sensor],
            "current_only_recomputed": current_by_sensor[sensor],
            "delta_full_minus_current_ap": (
                full_by_sensor[sensor]["ap"]
                - current_by_sensor[sensor]["ap"]
            ),
        }
    macro_delta = {
        metric: ranknet_macro[metric] - scratch_macro[metric]
        for metric in CLASSIFICATION_METRICS
    }
    temporal_ap_deltas = {
        sensor: per_sensor[sensor]["delta_full_minus_current_ap"]
        for sensor in EXPECTED_SENSORS
    }
    temporal_macro_delta = full_macro["ap"] - current_macro["ap"]
    metrics = {
        "per_sensor": per_sensor,
        "macro_mean_over_sensors": {
            "scratch": scratch_macro,
            "ranknet": ranknet_macro,
            "delta_ranknet_minus_scratch": macro_delta,
            "full_temporal_recomputed": full_macro,
            "current_only_recomputed": current_macro,
            "delta_full_minus_current_ap": temporal_macro_delta,
        },
        "global_paired_batch_fraction": paired_fraction,
    }

    ap_deltas = {
        sensor: per_sensor[sensor]["delta_ranknet_minus_scratch"]["ap"]
        for sensor in EXPECTED_SENSORS
    }
    nondecreasing_count = sum(delta >= 0.0 for delta in ap_deltas.values())
    worst_sensor = min(ap_deltas, key=ap_deltas.get)
    positive_temporal_count = sum(
        delta > 0.0 for delta in temporal_ap_deltas.values()
    )
    s5p = full_by_sensor["s5p"]
    s5p_samples = int(s5p["samples"])
    s5p_non_degenerate = (
        s5p["probability_std"] > 1e-8
        and s5p["probability_max"] > s5p["probability_min"]
        and s5p["ap"] > s5p["positive_rate"]
        and s5p["auroc"] > 0.5
        and 0 < s5p["predicted_positive_at_best"] < s5p_samples
    )
    structural_pass = all(check["pass"] for check in checks.values())
    criteria = {
        "artifact_protocol_metric_and_objective_integrity": {
            "pass": structural_pass,
            "observed": {
                "passed_checks": sum(
                    check["pass"] for check in checks.values()
                ),
                "total_checks": len(checks),
            },
            "expected": "all checks pass",
        },
        "macro_ap_delta_at_least_0p010": {
            "pass": macro_delta["ap"] >= GATE_SPEC["macro_ap_min_delta"],
            "observed": macro_delta["ap"],
            "expected": f">= {GATE_SPEC['macro_ap_min_delta']}",
        },
        "at_least_three_sensor_ap_deltas_nondecreasing": {
            "pass": nondecreasing_count
            >= GATE_SPEC["minimum_nondecreasing_sensor_ap_count"],
            "observed": {
                "count": nondecreasing_count,
                "deltas": ap_deltas,
            },
            "expected": (
                f">= {GATE_SPEC['minimum_nondecreasing_sensor_ap_count']} "
                "sensors"
            ),
        },
        "worst_sensor_ap_delta_at_least_minus_0p010": {
            "pass": ap_deltas[worst_sensor]
            >= GATE_SPEC["worst_sensor_ap_min_delta"],
            "observed": {
                "sensor": worst_sensor,
                "delta": ap_deltas[worst_sensor],
            },
            "expected": f">= {GATE_SPEC['worst_sensor_ap_min_delta']}",
        },
        "macro_best_f1_delta_at_least_minus_0p005": {
            "pass": macro_delta["best_f1"]
            >= GATE_SPEC["macro_best_f1_min_delta"],
            "observed": macro_delta["best_f1"],
            "expected": f">= {GATE_SPEC['macro_best_f1_min_delta']}",
        },
        "macro_f1_at_0p5_delta_at_least_minus_0p010": {
            "pass": macro_delta["f1_at_0p5"]
            >= GATE_SPEC["macro_f1_at_0p5_min_delta"],
            "observed": macro_delta["f1_at_0p5"],
            "expected": f">= {GATE_SPEC['macro_f1_at_0p5_min_delta']}",
        },
        "paired_batch_fraction_at_least_0p99": {
            "pass": paired_fraction
            >= GATE_SPEC["minimum_paired_batch_fraction"],
            "observed": paired_fraction,
            "expected": f">= {GATE_SPEC['minimum_paired_batch_fraction']}",
        },
        "full_temporal_macro_ap_advantage_at_least_0p005": {
            "pass": temporal_macro_delta
            >= GATE_SPEC["full_temporal_macro_ap_advantage_min"],
            "observed": temporal_macro_delta,
            "expected": (
                f">= "
                f"{GATE_SPEC['full_temporal_macro_ap_advantage_min']}"
            ),
        },
        "at_least_two_sensors_have_positive_temporal_ap_contribution": {
            "pass": positive_temporal_count
            >= GATE_SPEC["minimum_positive_temporal_sensor_count"],
            "observed": {
                "count": positive_temporal_count,
                "deltas": temporal_ap_deltas,
            },
            "expected": (
                f">= {GATE_SPEC['minimum_positive_temporal_sensor_count']} "
                "sensors with delta > 0"
            ),
        },
        "s5p_non_degenerate_probability_and_ranking_audit": {
            "pass": s5p_non_degenerate,
            "observed": {
                "samples": s5p_samples,
                "positive_rate": s5p["positive_rate"],
                "ap": s5p["ap"],
                "auroc": s5p["auroc"],
                "probability_min": s5p["probability_min"],
                "probability_max": s5p["probability_max"],
                "probability_std": s5p["probability_std"],
                "predicted_positive_at_best": s5p[
                    "predicted_positive_at_best"
                ],
            },
            "expected": (
                "nonconstant probabilities, AP > prevalence, AUROC > 0.5, "
                "and best-F1 predictions contain both classes"
            ),
        },
    }
    failed_criteria = [
        name for name, criterion in criteria.items() if not criterion["pass"]
    ]
    passed = not failed_criteria
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "created_utc": utc_now(),
        "status": "pass" if passed else "fail",
        "preregistered_gate": GATE_SPEC,
        "inputs": inputs,
        "checks": checks,
        "metrics": metrics,
        "gate_decision": {
            "pass": passed,
            "decision": (
                "promote_to_multiseed_confirmation"
                if passed
                else "do_not_promote"
            ),
            "criteria": criteria,
            "failed_criteria": failed_criteria,
            "interpretation": (
                "Eligible only for multi-seed confirmation; this remains a "
                "supervised ranking-aware classifier, not pretraining."
                if passed
                else (
                    "Hard stop: do not tune RankNet, add epochs, or start "
                    "another objective search."
                )
            ),
        },
    }


def _synthetic_classification_epoch(
    per_sensor: Mapping[str, Mapping[str, Any]],
    *,
    diagnostics: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    macro = {
        source_name: float(
            np.mean(
                [
                    float(per_sensor[sensor][source_name])
                    for sensor in EXPECTED_SENSORS
                ]
            )
        )
        for source_name in ("ap", "auroc", "f1", "f1_0p5")
    }
    epoch: Dict[str, Any] = {
        "epoch": 0,
        "train_loss": {sensor: 0.75 for sensor in EXPECTED_SENSORS},
        "val": {
            sensor: {
                **dict(per_sensor[sensor]),
                "loss": 0.5,
            }
            for sensor in EXPECTED_SENSORS
        },
        "macro_over_sensor": {**macro, "loss": 0.5},
    }
    if diagnostics is not None:
        epoch["train_objective_diagnostics"] = copy.deepcopy(diagnostics)
    return epoch


def _write_prediction_csv(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sensor", "row_index", "label", "probability"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sensor": record["sensor"],
                    "row_index": record["row_index"],
                    "label": record["label"],
                    "probability": repr(float(record["probability"])),
                }
            )


def _evaluation_payload(
    *,
    mode: str,
    input_contract: str,
    evaluation_path: Path,
    prediction_path: Path,
    records: Sequence[Mapping[str, Any]],
    history_path: Path,
    checkpoint_path: Path,
    data_signature: Mapping[str, Any],
    encoder_signature: Mapping[str, Any],
    resume_signature: Mapping[str, Any],
    event_protocol_fingerprint: str,
    normalization_stats_path: Path,
) -> Dict[str, Any]:
    by_sensor, macro = recompute_evaluation(records, mode)
    return {
        "artifact_type": "ranknet_validation_evaluation_v1",
        "schema_version": 1,
        "status": "completed",
        "evaluation_mode": mode,
        "input_contract": input_contract,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": 0,
        "source_history_path": str(history_path.resolve()),
        "source_history_sha256": sha256_file(history_path),
        "supervised_objective": OBJECTIVE_SIGNATURE,
        "encoder_signature": encoder_signature,
        "data_signature": data_signature,
        "resume_signature": resume_signature,
        "encoder_signature_fingerprint": json_fingerprint(
            encoder_signature
        ),
        "data_signature_fingerprint": json_fingerprint(data_signature),
        "resume_signature_fingerprint": json_fingerprint(resume_signature),
        "event_protocol_fingerprint": event_protocol_fingerprint,
        "normalization_stats_path": str(normalization_stats_path.resolve()),
        "normalization_stats_sha256": data_signature[
            "normalization_stats_sha256"
        ],
        "predictions_csv": str(prediction_path.resolve()),
        "predictions_csv_sha256": sha256_file(prediction_path),
        "prediction_rows": len(records),
        "full_sample_counts": {
            sensor: int(by_sensor[sensor]["samples"])
            for sensor in EXPECTED_SENSORS
        },
        "per_sensor": by_sensor,
        "macro_over_sensor": macro,
        "completed_unix": 1.0,
        "runtime": {"device": "cpu", "synthetic": True},
        "_self_test_path": str(evaluation_path),
    }


def run_self_test() -> None:
    event_fingerprint = "synthetic-global-purge-v1"
    data_signature = {
        "schema_version": 2,
        "event_protocol_fingerprint": event_fingerprint,
        "global_train_val_overlap_after": 0,
        "manifest_sha256": {
            sensor: {
                "train": f"{sensor}-synthetic-train",
                "val": f"{sensor}-synthetic-val",
            }
            for sensor in EXPECTED_SENSORS
        },
        "normalization_stats_sha256": "synthetic-normalization",
        "representation": {
            "streams": [
                "normalized_t0",
                "t0-prev1",
                "t0-seasonal",
            ],
        },
    }
    encoder_signature = {
        "schema_version": 1,
        "sharing": "shared",
        "embed_dim": 8,
        "sensors": list(EXPECTED_SENSORS),
    }
    optimization = {
        "batch_size": 64,
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "grad_clip": 1.0,
        "balanced_pos_weight": True,
        "amp": True,
        "augment": True,
        "seed": 20260727,
        "balanced_rounds": 157,
        "optimizer_steps_per_epoch": 628,
        "loader_lengths": {
            sensor: 157 for sensor in EXPECTED_SENSORS
        },
    }
    scratch_resume = {
        "schema_version": 1,
        "mode": "supervised",
        "sharing": "shared",
        "sensors": list(EXPECTED_SENSORS),
        "encoder_signature": encoder_signature,
        "pretrain_model": None,
        "data_signature": data_signature,
        "optimization": optimization,
        "evaluation": {"max_val_batches": 0},
    }
    ranknet_resume = copy.deepcopy(scratch_resume)
    ranknet_resume["schema_version"] = 2
    ranknet_resume["supervised_objective"] = OBJECTIVE_SIGNATURE

    full_records: List[Dict[str, Any]] = []
    current_records: List[Dict[str, Any]] = []
    for sensor in EXPECTED_SENSORS:
        labels = [0] * 20 + [1] * 20
        full_probabilities = [
            0.05 + index * 0.005 for index in range(20)
        ] + [
            0.75 + index * 0.005 for index in range(20)
        ]
        current_probabilities = list(full_probabilities)
        current_probabilities[0], current_probabilities[20] = (
            current_probabilities[20],
            current_probabilities[0],
        )
        for row_index, (label, full_probability, current_probability) in (
            enumerate(
                zip(
                    labels,
                    full_probabilities,
                    current_probabilities,
                )
            )
        ):
            full_records.append(
                {
                    "sensor": sensor,
                    "row_index": row_index,
                    "label": label,
                    "probability": full_probability,
                }
            )
            current_records.append(
                {
                    "sensor": sensor,
                    "row_index": row_index,
                    "label": label,
                    "probability": current_probability,
                }
            )
    full_by_sensor, _full_macro = recompute_evaluation(
        full_records,
        "full_temporal",
    )
    scratch_metrics = {
        sensor: {
            "samples": full_by_sensor[sensor]["samples"],
            "positive_rate": full_by_sensor[sensor]["positive_rate"],
            "ap": full_by_sensor[sensor]["ap"] - 0.02,
            "auroc": full_by_sensor[sensor]["auroc"] - 0.02,
            "f1": full_by_sensor[sensor]["f1"] - 0.02,
            "f1_0p5": full_by_sensor[sensor]["f1_0p5"] - 0.02,
            "threshold": full_by_sensor[sensor]["threshold"],
        }
        for sensor in EXPECTED_SENSORS
    }
    ranknet_metrics = {
        sensor: {
            key: full_by_sensor[sensor][key]
            for key in (
                "samples",
                "positive_rate",
                "ap",
                "auroc",
                "f1",
                "f1_0p5",
                "threshold",
            )
        }
        for sensor in EXPECTED_SENSORS
    }
    pair_diagnostics = {
        sensor: {
            "total_loss": 0.75,
            "weighted_bce_loss": 0.5,
            "ranknet_loss": 0.5,
            "positive_samples": 5024,
            "negative_samples": 5024,
            "pair_count": 160768,
            "batches": 157,
            "paired_batches": 157,
            "one_class_batches": 0,
            "paired_batch_fraction": 1.0,
            "one_class_batch_fraction": 0.0,
        }
        for sensor in EXPECTED_SENSORS
    }
    common_history = {
        "status": "completed",
        "mode": "supervised",
        "sharing": "shared",
        "sensors": list(EXPECTED_SENSORS),
        "event_protocol": {
            "fingerprint": event_fingerprint,
            "global_train_val_overlap_after": 0,
        },
        "data_signature": data_signature,
        "encoder_signature": encoder_signature,
        "model_parameters": 1000,
        "trainable_parameters": 900,
        "balanced_rounds_per_epoch": 157,
        "optimizer_steps_per_epoch": 628,
        "initial_head_sha256": "b" * 64,
        "config": {
            "epochs": 1,
            "seed": 20260727,
            "max_val_batches": 0,
            "init_checkpoint": None,
            "resume": None,
        },
    }
    scratch = {
        **common_history,
        "schema_version": 2,
        "resume_signature": scratch_resume,
        "epochs": [_synthetic_classification_epoch(scratch_metrics)],
    }
    ranknet = {
        **common_history,
        "schema_version": 4,
        "resume_signature": ranknet_resume,
        "supervised_objective": OBJECTIVE_SIGNATURE,
        "epochs": [
            _synthetic_classification_epoch(
                ranknet_metrics,
                diagnostics=pair_diagnostics,
            )
        ],
    }

    with tempfile.TemporaryDirectory(
        prefix="ranknet_gate_self_test."
    ) as directory:
        root = Path(directory)
        normalization_stats_path = root / "normalization_stats.json"
        scratch_path = root / "scratch.json"
        ranknet_path = root / "ranknet.json"
        checkpoint_path = root / "ranknet.pth"
        full_csv_path = root / "full.csv"
        current_csv_path = root / "current.csv"
        full_json_path = root / "full.json"
        current_json_path = root / "current.json"
        atomic_json_dump(
            {"synthetic": "normalization"},
            normalization_stats_path,
        )
        scratch["normalization_stats"] = str(
            normalization_stats_path.resolve()
        )
        ranknet["normalization_stats"] = str(
            normalization_stats_path.resolve()
        )
        atomic_json_dump(scratch, scratch_path)
        atomic_json_dump(ranknet, ranknet_path)
        torch.save(
            {
                "mode": "supervised",
                "epoch": 0,
                "model": {},
                "data_signature": data_signature,
                "encoder_signature": encoder_signature,
                "resume_signature": ranknet_resume,
                "supervised_objective": OBJECTIVE_SIGNATURE,
            },
            checkpoint_path,
        )
        _write_prediction_csv(full_csv_path, full_records)
        _write_prediction_csv(current_csv_path, current_records)

        def write_evaluations(
            *,
            history_path: Path = ranknet_path,
            full_path: Path = full_json_path,
            current_path: Path = current_json_path,
            full_rows: Sequence[Mapping[str, Any]] = full_records,
            current_rows: Sequence[Mapping[str, Any]] = current_records,
        ) -> None:
            full_payload = _evaluation_payload(
                mode="full_temporal",
                input_contract=FULL_INPUT_CONTRACT,
                evaluation_path=full_path,
                prediction_path=full_csv_path,
                records=full_rows,
                history_path=history_path,
                checkpoint_path=checkpoint_path,
                data_signature=data_signature,
                encoder_signature=encoder_signature,
                resume_signature=ranknet_resume,
                event_protocol_fingerprint=event_fingerprint,
                normalization_stats_path=normalization_stats_path,
            )
            current_payload = _evaluation_payload(
                mode="current_only",
                input_contract=CURRENT_ONLY_INPUT_CONTRACT,
                evaluation_path=current_path,
                prediction_path=current_csv_path,
                records=current_rows,
                history_path=history_path,
                checkpoint_path=checkpoint_path,
                data_signature=data_signature,
                encoder_signature=encoder_signature,
                resume_signature=ranknet_resume,
                event_protocol_fingerprint=event_fingerprint,
                normalization_stats_path=normalization_stats_path,
            )
            atomic_json_dump(full_payload, full_path)
            atomic_json_dump(current_payload, current_path)

        write_evaluations()

        def compare(
            *,
            scratch_input: Path = scratch_path,
            ranknet_input: Path = ranknet_path,
            full_input: Path = full_json_path,
            current_input: Path = current_json_path,
        ) -> Dict[str, Any]:
            return compare_artifacts(
                scratch_history_path=scratch_input,
                ranknet_history_path=ranknet_input,
                ranknet_checkpoint_path=checkpoint_path,
                full_temporal_evaluation_path=full_input,
                current_only_evaluation_path=current_input,
            )

        passing = compare()
        if passing.get("status") != "pass":
            raise AssertionError(f"Synthetic passing gate failed: {passing}")

        pending = compare(current_input=root / "not-created.json")
        if pending.get("status") != "pending":
            raise AssertionError("Missing evaluation did not return pending.")

        objective_mismatch = copy.deepcopy(ranknet)
        objective_mismatch["supervised_objective"]["rank_weight"] = 0.6
        objective_path = root / "ranknet_objective_mismatch.json"
        atomic_json_dump(objective_mismatch, objective_path)
        objective_result = compare(ranknet_input=objective_path)
        if objective_result.get("status") != "fail":
            raise AssertionError("Objective mismatch was not rejected.")

        comparator_mutations = {
            "manifest": (
                "data_signature",
                "manifest_sha256",
                "s2",
                "train",
            ),
            "stats": (
                "data_signature",
                "normalization_stats_sha256",
            ),
            "protocol": (
                "data_signature",
                "event_protocol_fingerprint",
            ),
            "encoder": ("encoder_signature", "embed_dim"),
            "head": ("initial_head_sha256",),
            "seed": ("config", "seed"),
            "rounds": ("balanced_rounds_per_epoch",),
            "steps": ("optimizer_steps_per_epoch",),
        }
        comparator_results: Dict[str, str] = {}
        for name, key_path in comparator_mutations.items():
            mutated = copy.deepcopy(scratch)
            target: Dict[str, Any] = mutated
            for key in key_path[:-1]:
                target = target[key]
            leaf = key_path[-1]
            current_value = target[leaf]
            target[leaf] = (
                current_value + 1
                if isinstance(current_value, int)
                else f"{current_value}-changed"
            )
            mutation_path = root / f"scratch_{name}.json"
            atomic_json_dump(mutated, mutation_path)
            result = compare(scratch_input=mutation_path)
            comparator_results[name] = result["status"]
            if result.get("status") != "fail":
                raise AssertionError(
                    f"Comparator mutation {name} was not rejected."
                )

        resume_delta = copy.deepcopy(scratch)
        resume_delta["resume_signature"]["optimization"][
            "learning_rate"
        ] = 1e-3
        resume_delta_path = root / "scratch_resume_delta.json"
        atomic_json_dump(resume_delta, resume_delta_path)
        resume_delta_result = compare(scratch_input=resume_delta_path)
        if resume_delta_result.get("status") != "fail":
            raise AssertionError(
                "Unregistered resume-signature delta was not rejected."
            )

        pair_failure = copy.deepcopy(ranknet)
        for sensor in EXPECTED_SENSORS:
            diagnostic = pair_failure["epochs"][0][
                "train_objective_diagnostics"
            ][sensor]
            diagnostic["paired_batches"] = 155
            diagnostic["one_class_batches"] = 2
            diagnostic["paired_batch_fraction"] = 155 / 157
            diagnostic["one_class_batch_fraction"] = 2 / 157
        pair_failure_path = root / "ranknet_pair_failure.json"
        atomic_json_dump(pair_failure, pair_failure_path)
        pair_full_path = root / "pair_full.json"
        pair_current_path = root / "pair_current.json"
        write_evaluations(
            history_path=pair_failure_path,
            full_path=pair_full_path,
            current_path=pair_current_path,
        )
        pair_result = compare(
            ranknet_input=pair_failure_path,
            full_input=pair_full_path,
            current_input=pair_current_path,
        )
        if pair_result.get("status") != "fail":
            raise AssertionError("Sub-99% paired batches did not fail.")
        pair_criterion = pair_result["gate_decision"]["criteria"][
            "paired_batch_fraction_at_least_0p99"
        ]
        if pair_criterion.get("pass"):
            raise AssertionError("Pair-fraction criterion did not fail.")

        metric_tamper = load_json(full_json_path)
        metric_tamper["per_sensor"]["s2"]["ap"] -= 0.01
        metric_tamper_path = root / "full_metric_tamper.json"
        atomic_json_dump(metric_tamper, metric_tamper_path)
        metric_result = compare(full_input=metric_tamper_path)
        if metric_result.get("status") != "fail":
            raise AssertionError("Tampered metric was not rejected.")

    print(
        json.dumps(
            {
                "self_test": "passed",
                "device": "cpu",
                "real_data_or_sealed_test_read": 0,
                "passing_case": passing["status"],
                "missing_case": pending["status"],
                "objective_mismatch_case": objective_result["status"],
                "comparator_mutations": comparator_results,
                "unregistered_resume_delta": resume_delta_result["status"],
                "pair_fraction_case": pair_result["status"],
                "metric_tamper_case": metric_result["status"],
            },
            sort_keys=True,
        )
    )


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    required = {
        "scratch_history": args.scratch_history,
        "ranknet_history": args.ranknet_history,
        "ranknet_checkpoint": args.ranknet_checkpoint,
        "full_temporal_evaluation": args.full_temporal_evaluation,
        "current_only_evaluation": args.current_only_evaluation,
    }
    omitted = [role for role, path in required.items() if path is None]
    if omitted:
        raise SystemExit(
            "Missing required arguments outside --self-test: "
            + ", ".join(omitted)
        )
    result = compare_artifacts(
        scratch_history_path=args.scratch_history,
        ranknet_history_path=args.ranknet_history,
        ranknet_checkpoint_path=args.ranknet_checkpoint,
        full_temporal_evaluation_path=args.full_temporal_evaluation,
        current_only_evaluation_path=args.current_only_evaluation,
    )
    if args.output_json is not None:
        output_resolved = args.output_json.resolve()
        input_resolved = {
            Path(path).resolve()
            for path in required.values()
            if path is not None
        }
        if output_resolved in input_resolved:
            raise ValueError("--output-json must not overwrite a gate input.")
        if args.output_json.is_file():
            existing = load_json(args.output_json)
            if existing.get("artifact_type") != ARTIFACT_TYPE:
                raise ValueError(
                    "Refusing to overwrite a different artifact type: "
                    f"{args.output_json}"
                )
        atomic_json_dump(result, args.output_json)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["status"] == "pass":
        return 0
    if result["status"] == "pending":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
