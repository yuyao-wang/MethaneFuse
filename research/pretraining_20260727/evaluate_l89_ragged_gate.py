#!/usr/bin/env python3
"""Integrity-first validation gate for L89 and explicit EMIT ragged pilots.

This evaluator only consumes inner-validation artifacts emitted by
``l89_ragged_cls_experiment.py`` or the explicitly versioned EMIT wrapper.  It
refuses sealed/test-like paths, verifies the four-arm evidence chain,
recomputes the fixed-threshold metrics, and makes the preregistered pilot
decision with an event-cluster paired bootstrap.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickletools
import random
import re
import sys
import tempfile
import zipfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import l89_ragged_cls_experiment as runner


ARTIFACT_SCHEMA_VERSION = 1
ARM_NAMES = tuple(runner.ARM_NAMES)
COMPARATOR_ARMS = ("t0_masked", "role_only", "history_shuffle_train")
CANDIDATE_ARM = "delta_time"
CHECKPOINT_NAME = "checkpoint_best_ap.pt"
PREDICTION_NAME = "validation_best_ap_predictions.csv"
FIXED_THRESHOLD = 0.5
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20_260_727
DEFAULT_JSON_NAME = "l89_ragged_gate.json"
DEFAULT_CSV_NAME = "l89_ragged_gate.csv"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PREDICTION_COLUMNS = (
    "id",
    "plume_id",
    "event_id",
    "label",
    "probability",
    "prediction_at_0_5",
    "arm",
    "epoch",
)
METRIC_KEYS = (
    "ap",
    "auc",
    "macro_f1_at_0_5",
    "balanced_accuracy_at_0_5",
    "pred_positive_rate_at_0_5",
)
CACHE_AUDIT_KEYS = {
    "train_cache",
    "train_cache_sha256",
    "validation_cache",
    "validation_cache_sha256",
    "weights_sha256",
    "event_overlap",
    "timepoints",
    "feature_dim",
    "role_names",
    "t0_index",
}
EMIT_SCRIPT_VERSION = "emit-ragged-cls-v1"
EMIT_SENSOR = "emit32"
EMIT_CACHE_AUDIT_PROVENANCE_KEYS = {
    "cache_script_version",
    "sensor",
}
SUPPORTED_RUN_SCRIPT_VERSIONS = {
    runner.SCRIPT_VERSION,
    EMIT_SCRIPT_VERSION,
}
GATE_THRESHOLDS = {
    "delta_ap_vs_t0_masked": 0.010,
    "delta_ap_vs_role_only": 0.005,
    "delta_ap_vs_history_shuffle_train": 0.005,
    "delta_time_balanced_accuracy_at_0_5": 0.55,
    "delta_time_pred_positive_rate_min": 0.05,
    "delta_time_pred_positive_rate_max": 0.95,
}
SERIALIZED_HANDLER_GLOBALS = {
    "__main__ train_heads",
    (
        "research.pretraining_20260727.l89_ragged_cls_experiment "
        "train_heads"
    ),
}


def _require_mapping(value: Any, *, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must contain a JSON/object mapping.")
    return value


def _validate_sha(value: Any, *, source: str) -> str:
    text = str(value)
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"{source} is not a lowercase SHA-256 digest.")
    return text


def _stable_file_sha(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return runner.sha256_file(path)


def _read_json_stable(path: Path) -> tuple[dict[str, Any], str]:
    before = _stable_file_sha(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    after = _stable_file_sha(path)
    if before != after:
        raise ValueError(f"{path} changed while it was being read.")
    return dict(_require_mapping(value, source=str(path))), before


def _read_prediction_stable(path: Path) -> tuple[pd.DataFrame, str]:
    before = _stable_file_sha(path)
    frame = pd.read_csv(
        path,
        dtype={
            "id": str,
            "plume_id": str,
            "event_id": str,
            "arm": str,
        },
        keep_default_na=False,
    )
    after = _stable_file_sha(path)
    if before != after:
        raise ValueError(f"{path} changed while it was being read.")
    return frame, before


def _serialized_handler_globals(data_pickle: bytes) -> list[str]:
    observed: list[str] = []
    try:
        for opcode, argument, _ in pickletools.genops(data_pickle):
            if (
                opcode.name == "GLOBAL"
                and argument in SERIALIZED_HANDLER_GLOBALS
            ):
                observed.append(str(argument))
    except Exception as error:
        raise ValueError(f"Checkpoint pickle stream is malformed: {error}") from error
    return observed


def _modern_torch_pickle_member(path: Path) -> tuple[str, bytes]:
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"{path} is not a modern ZIP-based torch archive and is refused."
        )
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"{path} contains duplicate ZIP members.")
        pickle_names = [
            name for name in names if name == "data.pkl" or name.endswith("/data.pkl")
        ]
        if len(pickle_names) != 1:
            raise ValueError(
                f"{path} must contain exactly one checkpoint data.pkl member."
            )
        pickle_name = pickle_names[0]
        return pickle_name, archive.read(pickle_name)


def _sanitized_checkpoint_copy(path: Path) -> tuple[Path | None, bool]:
    """Replace the legacy function reference with an inert allowlisted global.

    PyTorch 2.0's weights-only unpickler cannot allowlist a historical
    ``__main__.train_heads`` reference.  The runner only stored that reference
    as the value of ``args.handler``.  Replacing its GLOBAL opcode with the
    already allowlisted ``collections.OrderedDict`` permits a weights-only load
    without invoking the original handler or a general-purpose pickle loader.
    """

    pickle_name, data_pickle = _modern_torch_pickle_member(path)
    with zipfile.ZipFile(path, "r") as archive:
        handler_globals = _serialized_handler_globals(data_pickle)
        if not handler_globals:
            return None, False
        if len(handler_globals) != 1:
            raise ValueError(
                f"{path} has {len(handler_globals)} serialized handler globals; "
                "expected one."
            )
        module_name, function_name = handler_globals[0].split(" ", 1)
        handler_opcode = (
            b"c"
            + module_name.encode("utf-8")
            + b"\n"
            + function_name.encode("utf-8")
            + b"\n"
        )
        if data_pickle.count(handler_opcode) != 1:
            raise ValueError(
                f"{path} serialized handler opcode could not be sanitized exactly."
            )
        sanitized_pickle = data_pickle.replace(
            handler_opcode,
            b"ccollections\nOrderedDict\n",
            1,
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".l89-ragged-sanitized-checkpoint.",
            suffix=".pt",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with zipfile.ZipFile(temporary_path, "w") as output:
                for member in archive.infolist():
                    payload = (
                        sanitized_pickle
                        if member.filename == pickle_name
                        else archive.read(member.filename)
                    )
                    output.writestr(member, payload)
        except Exception:
            with suppress(FileNotFoundError):
                temporary_path.unlink()
            raise
    return temporary_path, True


def _torch_load_weights_only(
    path: Path,
    *,
    permit_legacy_handler: bool,
) -> tuple[Any, bool]:
    """Load tensors and primitive containers without general pickle execution."""

    sanitized_path: Path | None = None
    sanitized = False
    try:
        _, data_pickle = _modern_torch_pickle_member(path)
        if permit_legacy_handler:
            sanitized_path, sanitized = _sanitized_checkpoint_copy(path)
        elif _serialized_handler_globals(data_pickle):
            raise ValueError(f"{path} unexpectedly contains a CLI handler.")
        load_path = sanitized_path if sanitized_path is not None else path
        try:
            value = torch.load(
                load_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as error:
            raise RuntimeError(
                "This evaluator requires torch.load(..., weights_only=True)."
            ) from error
        except Exception as error:
            raise ValueError(
                f"Restricted weights-only load failed for {path}: {error}"
            ) from error
        return value, sanitized
    finally:
        if sanitized_path is not None:
            with suppress(FileNotFoundError):
                sanitized_path.unlink()


def _load_cache_pair_secure(
    train_path: Path,
    validation_path: Path,
    *,
    expected_script_version: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Secure equivalent of the runner's cache-pair loader."""

    if expected_script_version not in SUPPORTED_RUN_SCRIPT_VERSIONS:
        raise ValueError(
            f"Unsupported run script_version={expected_script_version!r}."
        )
    runner.assert_not_sealed_path(train_path, purpose="train cache")
    runner.assert_not_sealed_path(validation_path, purpose="validation cache")
    train_before = _stable_file_sha(train_path)
    validation_before = _stable_file_sha(validation_path)
    train, train_sanitized = _torch_load_weights_only(
        train_path, permit_legacy_handler=False
    )
    validation, validation_sanitized = _torch_load_weights_only(
        validation_path, permit_legacy_handler=False
    )
    if train_sanitized or validation_sanitized:
        raise ValueError("Cache files unexpectedly required checkpoint sanitization.")
    if train_before != _stable_file_sha(train_path):
        raise ValueError(f"{train_path} changed while it was being read.")
    if validation_before != _stable_file_sha(validation_path):
        raise ValueError(f"{validation_path} changed while it was being read.")
    if not isinstance(train, dict) or not isinstance(validation, dict):
        raise ValueError("Both cache files must contain dictionaries.")
    runner.validate_cache_payload(
        train, path=train_path, expected_split="train"
    )
    runner.validate_cache_payload(
        validation, path=validation_path, expected_split="val"
    )
    for name, payload in (("train", train), ("validation", validation)):
        if payload.get("script_version") != expected_script_version:
            raise ValueError(
                f"{name} cache script_version does not match the run."
            )
        if expected_script_version == EMIT_SCRIPT_VERSION:
            if payload.get("sensor") != EMIT_SENSOR:
                raise ValueError(
                    f"{name} EMIT cache sensor must equal {EMIT_SENSOR!r}."
                )
            contract = _require_mapping(
                payload.get("input_contract"),
                source=f"{name} EMIT cache input_contract",
            )
            if (
                contract.get("script_version") != EMIT_SCRIPT_VERSION
                or contract.get("sensor") != EMIT_SENSOR
            ):
                raise ValueError(
                    f"{name} EMIT cache input contract provenance is invalid."
                )
    for key in (
        "weights_sha256",
        "role_names",
        "t0_index",
        "path_columns",
        "time_columns",
    ):
        if train.get(key) != validation.get(key):
            raise ValueError(f"Train/validation cache mismatch for {key!r}.")
    if train["features"].shape[1:] != validation["features"].shape[1:]:
        raise ValueError("Train/validation feature shapes differ.")
    if runner.comparable_input_contract(
        train["input_contract"]
    ) != runner.comparable_input_contract(validation["input_contract"]):
        raise ValueError(
            "Train/validation input contracts differ beyond split provenance."
        )
    overlap = sorted(set(train["event_ids"]) & set(validation["event_ids"]))
    if overlap:
        raise ValueError(
            "Train/validation event overlap is nonzero: "
            f"count={len(overlap)}, examples={overlap[:10]}."
        )
    audit = {
        "train_cache": str(train_path),
        "train_cache_sha256": train_before,
        "validation_cache": str(validation_path),
        "validation_cache_sha256": validation_before,
        "weights_sha256": train["weights_sha256"],
        "event_overlap": 0,
        "timepoints": int(train["features"].shape[1]),
        "feature_dim": int(train["features"].shape[2]),
        "role_names": list(train["role_names"]),
        "t0_index": int(train["t0_index"]),
    }
    if expected_script_version == EMIT_SCRIPT_VERSION:
        audit.update(
            {
                "cache_script_version": EMIT_SCRIPT_VERSION,
                "sensor": EMIT_SENSOR,
            }
        )
    return train, validation, audit


def recompute_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    labels_array = np.asarray(labels)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if labels_array.ndim != 1 or probability_array.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")
    if labels_array.shape != probability_array.shape or labels_array.size == 0:
        raise ValueError("Labels and probabilities must have the same nonzero length.")
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("Probabilities contain non-finite values.")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    numeric_labels = labels_array.astype(np.float64)
    if not np.all(np.isfinite(numeric_labels)):
        raise ValueError("Labels contain non-finite values.")
    if not np.all(numeric_labels == np.floor(numeric_labels)):
        raise ValueError("Labels must be binary integers.")
    integer_labels = numeric_labels.astype(np.int64)
    if set(integer_labels.tolist()) != {0, 1}:
        raise ValueError("Metrics require both binary classes.")
    predictions = (probability_array >= FIXED_THRESHOLD).astype(np.int64)
    return {
        "ap": float(average_precision_score(integer_labels, probability_array)),
        "auc": float(roc_auc_score(integer_labels, probability_array)),
        "macro_f1_at_0_5": float(
            f1_score(
                integer_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(integer_labels, predictions)
        ),
        "pred_positive_rate_at_0_5": float(predictions.mean()),
    }


def build_event_cluster_bootstrap_plan(
    labels: Sequence[int] | np.ndarray,
    event_ids: Sequence[str] | np.ndarray,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Draw canonical events with replacement and retain complete clusters."""

    if int(replicates) != replicates or int(replicates) <= 0:
        raise ValueError("Bootstrap replicates must be a positive integer.")
    labels_array = np.asarray(labels, dtype=np.int64)
    events_array = np.asarray(event_ids, dtype=str)
    if labels_array.ndim != 1 or events_array.ndim != 1:
        raise ValueError("Bootstrap labels/event_ids must be one-dimensional.")
    if labels_array.shape != events_array.shape or labels_array.size == 0:
        raise ValueError("Bootstrap labels/event_ids must have equal nonzero length.")
    if set(labels_array.tolist()) != {0, 1}:
        raise ValueError("Bootstrap source must contain both binary classes.")
    if np.any(np.char.strip(events_array) == ""):
        raise ValueError("Bootstrap event IDs must be nonempty.")

    event_order = list(dict.fromkeys(events_array.tolist()))
    clusters = {
        event_id: np.flatnonzero(events_array == event_id).astype(np.int64)
        for event_id in event_order
    }
    rng = np.random.default_rng(int(seed))
    accepted: list[np.ndarray] = []
    attempts = 0
    maximum_attempts = max(1_000, int(replicates) * 100)
    while len(accepted) < int(replicates) and attempts < maximum_attempts:
        attempts += 1
        sampled_positions = rng.integers(
            0, len(event_order), size=len(event_order)
        )
        indices = np.concatenate(
            [clusters[event_order[int(position)]] for position in sampled_positions]
        )
        if np.unique(labels_array[indices]).size != 2:
            continue
        accepted.append(indices)
    if len(accepted) != int(replicates):
        raise RuntimeError(
            "Could not obtain the requested number of two-class event-cluster "
            f"bootstrap replicates: accepted={len(accepted)}, attempts={attempts}."
        )
    return {
        "indices": accepted,
        "attempts": attempts,
        "discarded_single_class": attempts - len(accepted),
        "replicates": int(replicates),
        "seed": int(seed),
        "cluster_count": len(event_order),
    }


def _bootstrap_metric_samples(
    labels: np.ndarray,
    probabilities: np.ndarray,
    plan: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    ap_values = np.empty(len(plan["indices"]), dtype=np.float64)
    auc_values = np.empty(len(plan["indices"]), dtype=np.float64)
    for position, indices in enumerate(plan["indices"]):
        sample_labels = labels[indices]
        sample_probabilities = probabilities[indices]
        ap_values[position] = average_precision_score(
            sample_labels, sample_probabilities
        )
        auc_values[position] = roc_auc_score(
            sample_labels, sample_probabilities
        )
    return {"ap": ap_values, "auc": auc_values}


def _comparison_from_bootstrap_samples(
    candidate_metrics: Mapping[str, float],
    reference_metrics: Mapping[str, float],
    candidate_samples: Mapping[str, np.ndarray],
    reference_samples: Mapping[str, np.ndarray],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("ap", "auc"):
        differences = candidate_samples[metric] - reference_samples[metric]
        lower, upper = np.quantile(differences, [0.025, 0.975])
        point_delta = float(
            candidate_metrics[metric] - reference_metrics[metric]
        )
        result[metric] = {
            "candidate": float(candidate_metrics[metric]),
            "reference": float(reference_metrics[metric]),
            "delta": point_delta,
            "ci_95": {
                "lower": float(lower),
                "upper": float(upper),
            },
            "ci_lower_gt_zero": bool(float(lower) > 0.0),
        }
    result["bootstrap"] = {
        "unit": "canonical_event_cluster",
        "paired": True,
        "replicates": int(plan["replicates"]),
        "seed": int(plan["seed"]),
        "cluster_count": int(plan["cluster_count"]),
        "attempts": int(plan["attempts"]),
        "discarded_single_class": int(plan["discarded_single_class"]),
    }
    return result


def paired_cluster_bootstrap(
    labels: Sequence[int] | np.ndarray,
    event_ids: Sequence[str] | np.ndarray,
    candidate_probabilities: Sequence[float] | np.ndarray,
    reference_probabilities: Sequence[float] | np.ndarray,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    candidate_array = np.asarray(candidate_probabilities, dtype=np.float64)
    reference_array = np.asarray(reference_probabilities, dtype=np.float64)
    candidate_metrics = recompute_metrics(labels_array, candidate_array)
    reference_metrics = recompute_metrics(labels_array, reference_array)
    plan = build_event_cluster_bootstrap_plan(
        labels_array,
        event_ids,
        replicates=replicates,
        seed=seed,
    )
    return _comparison_from_bootstrap_samples(
        candidate_metrics,
        reference_metrics,
        _bootstrap_metric_samples(labels_array, candidate_array, plan),
        _bootstrap_metric_samples(labels_array, reference_array, plan),
        plan,
    )


def _validate_cache_audit(
    audit: Any,
    *,
    source: str,
    run_script_version: str,
) -> dict[str, Any]:
    if run_script_version not in SUPPORTED_RUN_SCRIPT_VERSIONS:
        raise ValueError(
            f"{source} has unsupported run script_version="
            f"{run_script_version!r}."
        )
    audit_mapping = dict(_require_mapping(audit, source=source))
    expected_keys = set(CACHE_AUDIT_KEYS)
    if run_script_version == EMIT_SCRIPT_VERSION:
        expected_keys.update(EMIT_CACHE_AUDIT_PROVENANCE_KEYS)
    if set(audit_mapping) != expected_keys:
        missing = sorted(expected_keys - set(audit_mapping))
        extra = sorted(set(audit_mapping) - expected_keys)
        raise ValueError(
            f"{source} cache audit schema mismatch; missing={missing}, extra={extra}."
        )
    if run_script_version == EMIT_SCRIPT_VERSION:
        if audit_mapping["cache_script_version"] != EMIT_SCRIPT_VERSION:
            raise ValueError(
                f"{source}.cache_script_version must equal "
                f"{EMIT_SCRIPT_VERSION!r}."
            )
        if audit_mapping["sensor"] != EMIT_SENSOR:
            raise ValueError(
                f"{source}.sensor must equal {EMIT_SENSOR!r}."
            )
    if audit_mapping["event_overlap"] != 0:
        raise ValueError(f"{source} cache audit has event_overlap != 0.")
    for key in (
        "train_cache_sha256",
        "validation_cache_sha256",
        "weights_sha256",
    ):
        _validate_sha(audit_mapping[key], source=f"{source}.{key}")
    for key in ("train_cache", "validation_cache"):
        if not isinstance(audit_mapping[key], str) or not audit_mapping[key].strip():
            raise ValueError(f"{source}.{key} must be a nonempty path string.")
    if int(audit_mapping["timepoints"]) not in {3, 6}:
        raise ValueError(f"{source} cache audit has invalid timepoints.")
    if int(audit_mapping["feature_dim"]) <= 0:
        raise ValueError(f"{source} cache audit has invalid feature_dim.")
    if len(audit_mapping["role_names"]) != int(audit_mapping["timepoints"]):
        raise ValueError(f"{source} cache audit role_names length is invalid.")
    if not 0 <= int(audit_mapping["t0_index"]) < int(
        audit_mapping["timepoints"]
    ):
        raise ValueError(f"{source} cache audit t0_index is invalid.")
    return audit_mapping


def _validate_parameter_signature(
    signature: Any,
    *,
    source: str,
) -> dict[str, Any]:
    mapping = dict(_require_mapping(signature, source=source))
    expected_keys = {"parameter_count", "parameter_shapes", "shape_sha256"}
    if set(mapping) != expected_keys:
        raise ValueError(f"{source} parameter signature schema mismatch.")
    shapes = dict(
        _require_mapping(
            mapping["parameter_shapes"],
            source=f"{source}.parameter_shapes",
        )
    )
    recomputed_sha = runner.sha256_bytes(runner.canonical_json_bytes(shapes))
    if recomputed_sha != mapping["shape_sha256"]:
        raise ValueError(f"{source} parameter signature shape SHA is invalid.")
    _validate_sha(mapping["shape_sha256"], source=f"{source}.shape_sha256")
    parameter_count = 0
    for name, entry in shapes.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{source} parameter signature has an invalid name.")
        item = dict(
            _require_mapping(entry, source=f"{source}.parameter_shapes[{name}]")
        )
        if set(item) != {"shape", "dtype", "numel"}:
            raise ValueError(
                f"{source} parameter signature entry {name!r} is invalid."
            )
        shape = [int(dimension) for dimension in item["shape"]]
        if any(dimension < 0 for dimension in shape):
            raise ValueError(
                f"{source} parameter signature entry {name!r} has invalid shape."
            )
        numel = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if int(item["numel"]) != numel:
            raise ValueError(
                f"{source} parameter signature entry {name!r} has invalid numel."
            )
        if not isinstance(item["dtype"], str) or not item["dtype"].startswith(
            "torch."
        ):
            raise ValueError(
                f"{source} parameter signature entry {name!r} has invalid dtype."
            )
        parameter_count += numel
    if int(mapping["parameter_count"]) != parameter_count:
        raise ValueError(f"{source} parameter signature count is invalid.")
    return mapping


def _prototype_evidence(
    run_config: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    if int(run_config["depth"]) != 2:
        raise ValueError("run_config.depth must equal the preregistered depth=2.")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        seed = int(run_config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        prototype = runner.RaggedCurrentQueryHead(
            feature_dim=int(cache_audit["feature_dim"]),
            num_roles=int(cache_audit["timepoints"]),
            model_dim=int(run_config["model_dim"]),
            num_heads=int(run_config["num_heads"]),
            depth=int(run_config["depth"]),
            mlp_ratio=float(run_config["mlp_ratio"]),
            dropout=float(run_config["dropout"]),
            periods_days=tuple(
                float(value) for value in run_config["delta_periods_days"]
            ),
            t0_index=int(cache_audit["t0_index"]),
        )
        state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in prototype.state_dict().items()
        }
        signature = runner.model_parameter_signature(prototype)
        initial_sha = runner.state_dict_sha256(state)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
    return state, signature, initial_sha


def _assert_metric_report(
    recomputed: Mapping[str, float],
    reported: Any,
    *,
    source: str,
    tolerance: float = 1e-10,
) -> None:
    reported_mapping = _require_mapping(reported, source=source)
    for key in METRIC_KEYS:
        if key not in reported_mapping:
            raise ValueError(f"{source} is missing metric {key!r}.")
        try:
            value = float(reported_mapping[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source}.{key} is not numeric.") from error
        if not np.isfinite(value):
            raise ValueError(f"{source}.{key} is non-finite.")
        if not np.isclose(
            recomputed[key],
            value,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                f"{source}.{key}={value} differs from recomputed "
                f"{recomputed[key]}."
            )


def _validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    arm: str,
    expected_rows: Sequence[tuple[str, str, str, int]],
    expected_epoch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise ValueError(
            f"{arm}: prediction columns differ from {PREDICTION_COLUMNS}."
        )
    if len(frame) != len(expected_rows):
        raise ValueError(
            f"{arm}: prediction row pairing/order count mismatch; "
            f"observed={len(frame)}, expected={len(expected_rows)}."
        )
    for column in ("id", "plume_id", "event_id"):
        values = frame[column].astype(str)
        if values.str.strip().eq("").any():
            raise ValueError(f"{arm}: prediction {column} contains empty values.")

    try:
        label_values = pd.to_numeric(frame["label"], errors="raise").to_numpy(
            dtype=np.float64
        )
        probability_values = pd.to_numeric(
            frame["probability"], errors="raise"
        ).to_numpy(dtype=np.float64)
        prediction_values = pd.to_numeric(
            frame["prediction_at_0_5"], errors="raise"
        ).to_numpy(dtype=np.float64)
        epoch_values = pd.to_numeric(frame["epoch"], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{arm}: prediction numeric columns are invalid.") from error
    if not np.all(np.isfinite(label_values)):
        raise ValueError(f"{arm}: prediction labels contain non-finite values.")
    if not np.all(label_values == np.floor(label_values)):
        raise ValueError(f"{arm}: prediction labels are not integers.")
    labels = label_values.astype(np.int64)
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError(f"{arm}: prediction labels are not binary.")
    if not np.all(np.isfinite(probability_values)):
        raise ValueError(f"{arm}: probabilities contain non-finite values.")
    if np.any((probability_values < 0.0) | (probability_values > 1.0)):
        raise ValueError(f"{arm}: probabilities lie outside [0, 1].")
    if not np.all(np.isfinite(prediction_values)) or not np.all(
        prediction_values == np.floor(prediction_values)
    ):
        raise ValueError(f"{arm}: fixed-threshold predictions are invalid.")
    predictions = prediction_values.astype(np.int64)
    expected_predictions = (
        probability_values >= FIXED_THRESHOLD
    ).astype(np.int64)
    if not np.array_equal(predictions, expected_predictions):
        raise ValueError(
            f"{arm}: prediction_at_0_5 does not use fixed threshold 0.5."
        )
    if not np.all(np.isfinite(epoch_values)) or not np.all(
        epoch_values == np.floor(epoch_values)
    ):
        raise ValueError(f"{arm}: prediction epochs are invalid.")
    epochs = epoch_values.astype(np.int64)
    if not np.all(epochs == int(expected_epoch)):
        raise ValueError(
            f"{arm}: prediction epoch does not match selected checkpoint epoch."
        )
    if frame["arm"].tolist() != [arm] * len(frame):
        raise ValueError(f"{arm}: prediction arm column is not constant/correct.")

    observed_rows = list(
        zip(
            frame["id"].astype(str).tolist(),
            frame["plume_id"].astype(str).tolist(),
            frame["event_id"].astype(str).tolist(),
            labels.tolist(),
        )
    )
    if observed_rows != list(expected_rows):
        mismatch = next(
            (
                index
                for index, (observed, expected) in enumerate(
                    zip(observed_rows, expected_rows)
                )
                if observed != expected
            ),
            None,
        )
        raise ValueError(
            f"{arm}: prediction row pairing/order mismatch at row {mismatch}."
        )
    return labels, probability_values, predictions


def _validate_checkpoint_model(
    model_state: Any,
    *,
    arm: str,
    prototype_state: Mapping[str, torch.Tensor],
    parameter_signature: Mapping[str, Any],
) -> None:
    state = _require_mapping(model_state, source=f"{arm} checkpoint.model")
    if set(state) != set(prototype_state):
        missing = sorted(set(prototype_state) - set(state))
        extra = sorted(set(state) - set(prototype_state))
        raise ValueError(
            f"{arm}: checkpoint model state schema mismatch; "
            f"missing={missing}, extra={extra}."
        )
    for name, expected in prototype_state.items():
        observed = state[name]
        if not isinstance(observed, torch.Tensor):
            raise ValueError(f"{arm}: checkpoint state {name!r} is not a tensor.")
        if observed.shape != expected.shape or observed.dtype != expected.dtype:
            raise ValueError(
                f"{arm}: checkpoint state {name!r} shape/dtype mismatch."
            )
    for name, entry in parameter_signature["parameter_shapes"].items():
        observed = state[name]
        if list(observed.shape) != list(entry["shape"]):
            raise ValueError(
                f"{arm}: checkpoint parameter signature shape mismatch for {name}."
            )
        if str(observed.dtype) != str(entry["dtype"]):
            raise ValueError(
                f"{arm}: checkpoint parameter signature dtype mismatch for {name}."
            )
        if int(observed.numel()) != int(entry["numel"]):
            raise ValueError(
                f"{arm}: checkpoint parameter signature numel mismatch for {name}."
            )


def _plan_sha256(plan: Mapping[str, Any]) -> str:
    serializable = {
        "indices": [indices.astype(np.int64).tolist() for indices in plan["indices"]],
        "attempts": int(plan["attempts"]),
        "discarded_single_class": int(plan["discarded_single_class"]),
        "replicates": int(plan["replicates"]),
        "seed": int(plan["seed"]),
        "cluster_count": int(plan["cluster_count"]),
    }
    return runner.sha256_bytes(runner.canonical_json_bytes(serializable))


def _evaluate_gate(
    per_arm: Mapping[str, Mapping[str, float]],
    comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidate = per_arm[CANDIDATE_ARM]
    criteria: dict[str, dict[str, Any]] = {}
    comparator_thresholds = {
        "t0_masked": GATE_THRESHOLDS["delta_ap_vs_t0_masked"],
        "role_only": GATE_THRESHOLDS["delta_ap_vs_role_only"],
        "history_shuffle_train": GATE_THRESHOLDS[
            "delta_ap_vs_history_shuffle_train"
        ],
    }
    for comparator, threshold in comparator_thresholds.items():
        key = f"delta_ap_vs_{comparator}"
        observed = float(
            comparisons[f"delta_time_minus_{comparator}"]["ap"]["delta"]
        )
        criteria[key] = {
            "observed": observed,
            "operator": ">=",
            "threshold": float(threshold),
            "pass": bool(observed >= threshold),
        }
    balanced_accuracy = float(candidate["balanced_accuracy_at_0_5"])
    criteria["delta_time_balanced_accuracy_at_0_5"] = {
        "observed": balanced_accuracy,
        "operator": ">=",
        "threshold": GATE_THRESHOLDS[
            "delta_time_balanced_accuracy_at_0_5"
        ],
        "pass": bool(
            balanced_accuracy
            >= GATE_THRESHOLDS["delta_time_balanced_accuracy_at_0_5"]
        ),
    }
    positive_rate = float(candidate["pred_positive_rate_at_0_5"])
    positive_rate_pass = bool(
        GATE_THRESHOLDS["delta_time_pred_positive_rate_min"]
        <= positive_rate
        <= GATE_THRESHOLDS["delta_time_pred_positive_rate_max"]
    )
    criteria["delta_time_pred_positive_rate_at_0_5"] = {
        "observed": positive_rate,
        "operator": "inclusive_between",
        "minimum": GATE_THRESHOLDS["delta_time_pred_positive_rate_min"],
        "maximum": GATE_THRESHOLDS["delta_time_pred_positive_rate_max"],
        "pass": positive_rate_pass,
    }
    failed = [key for key, value in criteria.items() if not value["pass"]]
    ci_diagnostics = {
        f"delta_time_minus_{comparator}": {
            "ap_ci_lower_gt_zero": bool(
                comparisons[f"delta_time_minus_{comparator}"]["ap"][
                    "ci_lower_gt_zero"
                ]
            ),
            "auc_ci_lower_gt_zero": bool(
                comparisons[f"delta_time_minus_{comparator}"]["auc"][
                    "ci_lower_gt_zero"
                ]
            ),
        }
        for comparator in COMPARATOR_ARMS
    }
    return {
        "fixed_probability_threshold": FIXED_THRESHOLD,
        "preregistered_thresholds": copy.deepcopy(GATE_THRESHOLDS),
        "criteria": criteria,
        "pass": not failed,
        "failed_criteria": failed,
        "ci_diagnostics": {
            "part_of_preregistered_gate": False,
            "comparisons": ci_diagnostics,
        },
    }


def _comparison_csv_frame(artifact: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    decision = artifact["gate_decision"]
    comparison_specs: list[
        tuple[str, str, str, str, Mapping[str, Any], bool | None]
    ] = []
    for reference in COMPARATOR_ARMS:
        name = f"delta_time_minus_{reference}"
        comparison_specs.append(
            (
                "preregistered",
                CANDIDATE_ARM,
                reference,
                name,
                artifact["metrics"]["comparisons"][name],
                decision["criteria"][f"delta_ap_vs_{reference}"]["pass"],
            )
        )
    for name, comparison in artifact["exploratory_not_preregistered"][
        "comparisons"
    ].items():
        reference = name.removeprefix("role_only_minus_")
        comparison_specs.append(
            (
                "exploratory_not_preregistered",
                "role_only",
                reference,
                name,
                comparison,
                None,
            )
        )

    for (
        analysis_scope,
        candidate,
        reference,
        comparison_name,
        comparison,
        comparison_gate_pass,
    ) in comparison_specs:
        rows.append(
            {
                "analysis_scope": analysis_scope,
                "comparison": comparison_name,
                "part_of_preregistered_gate": analysis_scope == "preregistered",
                "candidate_arm": candidate,
                "reference_arm": reference,
                "fixed_threshold": FIXED_THRESHOLD,
                "candidate_ap": comparison["ap"]["candidate"],
                "reference_ap": comparison["ap"]["reference"],
                "delta_ap": comparison["ap"]["delta"],
                "delta_ap_ci_95_lower": comparison["ap"]["ci_95"]["lower"],
                "delta_ap_ci_95_upper": comparison["ap"]["ci_95"]["upper"],
                "delta_ap_ci_lower_gt_zero": comparison["ap"][
                    "ci_lower_gt_zero"
                ],
                "candidate_auc": comparison["auc"]["candidate"],
                "reference_auc": comparison["auc"]["reference"],
                "delta_auc": comparison["auc"]["delta"],
                "delta_auc_ci_95_lower": comparison["auc"]["ci_95"]["lower"],
                "delta_auc_ci_95_upper": comparison["auc"]["ci_95"]["upper"],
                "delta_auc_ci_lower_gt_zero": comparison["auc"][
                    "ci_lower_gt_zero"
                ],
                "candidate_balanced_accuracy_at_0_5": artifact["metrics"][
                    "per_arm"
                ][candidate]["balanced_accuracy_at_0_5"],
                "candidate_pred_positive_rate_at_0_5": artifact["metrics"][
                    "per_arm"
                ][candidate]["pred_positive_rate_at_0_5"],
                "comparison_ap_gate_pass": comparison_gate_pass,
                "overall_gate_pass": decision["pass"],
                "gate_status": artifact["status"],
                "bootstrap_replicates": comparison["bootstrap"]["replicates"],
                "bootstrap_seed": comparison["bootstrap"]["seed"],
                "bootstrap_unit": comparison["bootstrap"]["unit"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_run(
    run_dir: str | Path,
    *,
    output_json: str | Path,
    output_csv: str | Path,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate a completed four-arm run and atomically write its gate decision."""

    run_path = Path(run_dir).expanduser().resolve()
    runner.assert_not_sealed_path(run_path, purpose="ragged run")
    if not run_path.is_dir():
        raise NotADirectoryError(run_path)
    json_output = Path(output_json).expanduser().resolve()
    csv_output = Path(output_csv).expanduser().resolve()
    if json_output == csv_output:
        raise ValueError("JSON and CSV output paths must differ.")
    for path in (json_output, csv_output):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True/--overwrite."
            )

    run_config, run_config_sha = _read_json_stable(run_path / "run_config.json")
    summary, summary_sha = _read_json_stable(run_path / "summary.json")
    run_status, run_status_sha = _read_json_stable(
        run_path / "run_status.json"
    )
    run_script_version = str(run_config.get("script_version", ""))
    if run_script_version not in SUPPORTED_RUN_SCRIPT_VERSIONS:
        raise ValueError(
            f"run_config script_version={run_script_version!r} is unsupported."
        )
    if summary.get("script_version") != run_script_version:
        raise ValueError("summary script_version does not match run_config.")
    if (
        run_status.get("script_version", run_script_version)
        != run_script_version
    ):
        raise ValueError("run_status script_version does not match run_config.")
    if run_status.get("status") != "complete":
        raise ValueError("run_status is not complete.")
    if run_status.get("sealed_test_read") is not False:
        raise ValueError("run_status must record sealed_test_read=false.")
    if summary.get("sealed_test_read") is not False:
        raise ValueError("summary must record sealed_test_read=false.")
    if summary.get("selection_metric") != "validation_ap":
        raise ValueError("summary selection_metric must be validation_ap.")
    if tuple(run_config.get("arms", ())) != ARM_NAMES:
        raise ValueError(f"run_config must contain exactly four arms: {ARM_NAMES}.")
    if tuple(run_status.get("arms", ())) != ARM_NAMES:
        raise ValueError(f"run_status must contain exactly four arms: {ARM_NAMES}.")
    best_by_arm = dict(
        _require_mapping(summary.get("best_by_arm"), source="summary.best_by_arm")
    )
    if set(best_by_arm) != set(ARM_NAMES):
        raise ValueError("summary.best_by_arm does not contain exactly four arms.")

    config_audit = _validate_cache_audit(
        run_config.get("cache_audit"),
        source="run_config",
        run_script_version=run_script_version,
    )
    summary_audit = _validate_cache_audit(
        summary.get("cache_audit"),
        source="summary",
        run_script_version=run_script_version,
    )
    if summary_audit != config_audit:
        raise ValueError("summary cache audit differs from run_config cache audit.")
    train_cache_path = Path(config_audit["train_cache"]).expanduser().resolve()
    validation_cache_path = Path(
        config_audit["validation_cache"]
    ).expanduser().resolve()
    protected_inputs = {
        (run_path / "run_config.json").resolve(),
        (run_path / "summary.json").resolve(),
        (run_path / "run_status.json").resolve(),
        train_cache_path,
        validation_cache_path,
    }
    for arm in ARM_NAMES:
        protected_inputs.add((run_path / arm / CHECKPOINT_NAME).resolve())
        protected_inputs.add((run_path / arm / PREDICTION_NAME).resolve())
    for output_path in (json_output, csv_output):
        if output_path in protected_inputs:
            raise ValueError(
                f"Output path would overwrite a protected input: {output_path}."
            )
    train_cache, validation_cache, actual_audit = _load_cache_pair_secure(
        train_cache_path,
        validation_cache_path,
        expected_script_version=run_script_version,
    )
    if actual_audit != config_audit:
        raise ValueError(
            "Declared cache audit differs from the independently verified cache audit."
        )
    actual_overlap = set(train_cache["event_ids"]) & set(
        validation_cache["event_ids"]
    )
    if actual_overlap:
        raise ValueError("Actual train/validation event overlap is nonzero.")

    train_indices = runner.select_usable_rows(train_cache)
    validation_indices = runner.select_usable_rows(validation_cache)
    validation_data = runner.take_rows(validation_cache, validation_indices)
    expected_rows = list(
        zip(
            [str(value) for value in validation_data["ids"]],
            [str(value) for value in validation_data["plume_ids"]],
            [str(value) for value in validation_data["event_ids"]],
            validation_data["labels"].long().tolist(),
        )
    )
    if int(run_config.get("train_rows_total", -1)) != int(
        train_cache["features"].shape[0]
    ):
        raise ValueError("run_config train_rows_total is incorrect.")
    if int(run_config.get("train_rows_usable", -1)) != int(
        train_indices.numel()
    ):
        raise ValueError("run_config train_rows_usable is incorrect.")
    if int(run_config.get("validation_rows_total", -1)) != int(
        validation_cache["features"].shape[0]
    ):
        raise ValueError("run_config validation_rows_total is incorrect.")
    if int(run_config.get("validation_rows_usable", -1)) != len(expected_rows):
        raise ValueError("run_config validation_rows_usable is incorrect.")

    config_signature = _validate_parameter_signature(
        run_config.get("parameter_signature"),
        source="run_config",
    )
    summary_signature = _validate_parameter_signature(
        summary.get("parameter_signature"),
        source="summary",
    )
    if summary_signature != config_signature:
        raise ValueError(
            "summary parameter signature differs from run_config parameter signature."
        )
    config_initial_sha = _validate_sha(
        run_config.get("initial_state_sha256"),
        source="run_config initial-state SHA",
    )
    summary_initial_sha = _validate_sha(
        summary.get("initial_state_sha256"),
        source="summary initial-state SHA",
    )
    if summary_initial_sha != config_initial_sha:
        raise ValueError("summary initial-state SHA differs from run_config.")
    prototype_state, prototype_signature, prototype_initial_sha = (
        _prototype_evidence(run_config, config_audit)
    )
    if prototype_signature != config_signature:
        raise ValueError(
            "Independently reconstructed parameter signature differs from run_config."
        )
    if prototype_initial_sha != config_initial_sha:
        raise ValueError(
            "Independently reconstructed initial-state SHA differs from run_config."
        )

    matched_compute = _require_mapping(
        run_config.get("matched_compute_contract"),
        source="run_config.matched_compute_contract",
    )
    for key in (
        "same_initial_state",
        "same_model_parameter_shapes",
        "same_epoch_batches",
        "same_optimizer_and_steps",
        "delta_encoder_executed_for_every_arm",
    ):
        if matched_compute.get(key) is not True:
            raise ValueError(f"Matched-compute contract requires {key}=true.")
    if (
        matched_compute.get("history_shuffle_scope")
        != "training-only-cross-canonical-event"
    ):
        raise ValueError("Matched-compute history_shuffle_scope is invalid.")
    if "same_static_timepoints" in matched_compute and int(
        matched_compute["same_static_timepoints"]
    ) != int(config_audit["timepoints"]):
        raise ValueError("Matched-compute static timepoint count is invalid.")
    compute_by_arm: dict[str, dict[str, int]] = {}
    for arm in ARM_NAMES:
        best_record = _require_mapping(
            best_by_arm[arm],
            source=f"summary.best_by_arm.{arm}",
        )
        try:
            optimizer_steps = int(best_record["optimizer_steps"])
            train_rows_seen = int(best_record["train_rows_seen"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{arm}: matched-compute steps/rows evidence is invalid."
            ) from error
        if optimizer_steps <= 0 or train_rows_seen <= 0:
            raise ValueError(
                f"{arm}: matched-compute steps/rows must be positive."
            )
        if train_rows_seen > int(train_indices.numel()):
            raise ValueError(
                f"{arm}: matched-compute train_rows_seen exceeds usable rows."
            )
        compute_by_arm[arm] = {
            "optimizer_steps": optimizer_steps,
            "train_rows_seen": train_rows_seen,
        }
    compute_signatures = {
        (record["optimizer_steps"], record["train_rows_seen"])
        for record in compute_by_arm.values()
    }
    if len(compute_signatures) != 1:
        raise ValueError(
            "Four-arm matched-compute optimizer_steps/train_rows_seen differ."
        )

    labels_by_arm: dict[str, np.ndarray] = {}
    probabilities_by_arm: dict[str, np.ndarray] = {}
    per_arm_metrics: dict[str, dict[str, float]] = {}
    arm_evidence: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        arm_dir = run_path / arm
        if not arm_dir.is_dir():
            raise ValueError(f"Missing arm directory: {arm_dir}.")
        checkpoint_path = arm_dir / CHECKPOINT_NAME
        prediction_path = arm_dir / PREDICTION_NAME
        checkpoint_sha_before = _stable_file_sha(checkpoint_path)
        checkpoint, legacy_handler_sanitized = _torch_load_weights_only(
            checkpoint_path,
            permit_legacy_handler=True,
        )
        if checkpoint_sha_before != _stable_file_sha(checkpoint_path):
            raise ValueError(f"{checkpoint_path} changed while it was being read.")
        checkpoint = dict(
            _require_mapping(checkpoint, source=f"{arm} checkpoint")
        )
        if checkpoint.get("script_version") != run_script_version:
            raise ValueError(f"{arm}: checkpoint script_version is invalid.")
        if checkpoint.get("arm") != arm:
            raise ValueError(f"{arm}: checkpoint arm field is invalid.")
        checkpoint_audit = _validate_cache_audit(
            checkpoint.get("cache_audit"),
            source=f"{arm} checkpoint",
            run_script_version=run_script_version,
        )
        if checkpoint_audit != config_audit:
            raise ValueError(f"{arm}: checkpoint cache audit mismatch.")
        checkpoint_initial_sha = _validate_sha(
            checkpoint.get("initial_state_sha256"),
            source=f"{arm} checkpoint initial-state SHA",
        )
        if checkpoint_initial_sha != config_initial_sha:
            raise ValueError(f"{arm}: checkpoint initial-state SHA mismatch.")
        checkpoint_signature = _validate_parameter_signature(
            checkpoint.get("model_parameter_signature"),
            source=f"{arm} checkpoint",
        )
        if checkpoint_signature != config_signature:
            raise ValueError(f"{arm}: checkpoint parameter signature mismatch.")
        _validate_checkpoint_model(
            checkpoint.get("model"),
            arm=arm,
            prototype_state=prototype_state,
            parameter_signature=config_signature,
        )

        best_record = dict(
            _require_mapping(
                best_by_arm[arm], source=f"summary.best_by_arm.{arm}"
            )
        )
        if best_record.get("arm") != arm:
            raise ValueError(f"{arm}: summary best record arm is invalid.")
        try:
            checkpoint_epoch = int(checkpoint["epoch"])
            summary_epoch = int(best_record["epoch"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{arm}: selected epoch is invalid.") from error
        if checkpoint_epoch <= 0 or checkpoint_epoch != summary_epoch:
            raise ValueError(
                f"{arm}: checkpoint epoch differs from summary selected epoch."
            )

        frame, prediction_sha = _read_prediction_stable(prediction_path)
        labels, probabilities, _ = _validate_prediction_frame(
            frame,
            arm=arm,
            expected_rows=expected_rows,
            expected_epoch=checkpoint_epoch,
        )
        if labels_by_arm and not np.array_equal(
            labels, next(iter(labels_by_arm.values()))
        ):
            raise ValueError(f"{arm}: four-arm prediction labels are not paired.")
        metrics = recompute_metrics(labels, probabilities)
        _assert_metric_report(
            metrics,
            best_record.get("validation"),
            source=f"summary.best_by_arm.{arm}.validation",
        )
        _assert_metric_report(
            metrics,
            checkpoint.get("validation"),
            source=f"{arm} checkpoint.validation",
        )
        labels_by_arm[arm] = labels
        probabilities_by_arm[arm] = probabilities
        per_arm_metrics[arm] = metrics
        arm_evidence[arm] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha_before,
            "legacy_handler_sanitized_before_weights_only_load": bool(
                legacy_handler_sanitized
            ),
            "prediction_csv": str(prediction_path),
            "prediction_csv_sha256": prediction_sha,
            "selected_epoch": checkpoint_epoch,
            "validation_rows": len(frame),
        }

    labels = labels_by_arm[CANDIDATE_ARM]
    event_ids = np.asarray(
        [row[2] for row in expected_rows],
        dtype=str,
    )
    plan = build_event_cluster_bootstrap_plan(
        labels,
        event_ids,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    bootstrap_samples = {
        arm: _bootstrap_metric_samples(
            labels,
            probabilities_by_arm[arm],
            plan,
        )
        for arm in ARM_NAMES
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for comparator in COMPARATOR_ARMS:
        comparison_name = f"delta_time_minus_{comparator}"
        comparisons[comparison_name] = _comparison_from_bootstrap_samples(
            per_arm_metrics[CANDIDATE_ARM],
            per_arm_metrics[comparator],
            bootstrap_samples[CANDIDATE_ARM],
            bootstrap_samples[comparator],
            plan,
        )
    exploratory_comparisons: dict[str, dict[str, Any]] = {}
    for comparator in ("t0_masked", "history_shuffle_train"):
        comparison_name = f"role_only_minus_{comparator}"
        exploratory_comparisons[comparison_name] = (
            _comparison_from_bootstrap_samples(
                per_arm_metrics["role_only"],
                per_arm_metrics[comparator],
                bootstrap_samples["role_only"],
                bootstrap_samples[comparator],
                plan,
            )
        )
    plan_sha = _plan_sha256(plan)
    gate_decision = _evaluate_gate(per_arm_metrics, comparisons)
    status = "pass" if gate_decision["pass"] else "fail"
    artifact: dict[str, Any] = {
        "artifact_type": "l89_ragged_pilot_gate_decision",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": status,
        "runner_script_version": run_script_version,
        "sealed_test_read": False,
        "inputs": {
            "run_dir": str(run_path),
            "run_config": {
                "path": str(run_path / "run_config.json"),
                "sha256": run_config_sha,
            },
            "summary": {
                "path": str(run_path / "summary.json"),
                "sha256": summary_sha,
            },
            "run_status": {
                "path": str(run_path / "run_status.json"),
                "sha256": run_status_sha,
            },
            "cache_audit": config_audit,
            "arms": arm_evidence,
        },
        "integrity": {
            "all_checks_pass": True,
            "checks": {
                "sealed_test_refused": True,
                "four_arm_schema_and_order": True,
                "cache_audit_matches_files": True,
                "train_validation_event_overlap_zero": True,
                "initial_state_sha_reconstructed": True,
                "parameter_shape_sha_reconstructed": True,
                "checkpoint_models_match_signature": True,
                "four_arm_optimizer_steps_and_rows_match": True,
                "validation_rows_paired_to_cache": True,
                "reported_metrics_recomputed": True,
            },
            "matched_compute_by_arm": compute_by_arm,
        },
        "metrics": {
            "fixed_probability_threshold": FIXED_THRESHOLD,
            "per_arm": per_arm_metrics,
            "comparisons": comparisons,
        },
        "bootstrap": {
            "unit": "canonical_event_cluster",
            "paired": True,
            "replicates": int(plan["replicates"]),
            "seed": int(plan["seed"]),
            "cluster_count": int(plan["cluster_count"]),
            "attempts": int(plan["attempts"]),
            "discarded_single_class": int(plan["discarded_single_class"]),
            "plan_sha256": plan_sha,
        },
        "exploratory_not_preregistered": {
            "not_part_of_gate": True,
            "reason": "Post-hoc report; does not alter the preregistered gate.",
            "comparisons": exploratory_comparisons,
        },
        "gate_decision": gate_decision,
        "outputs": {
            "json": str(json_output),
            "csv": str(csv_output),
        },
    }
    csv_frame = _comparison_csv_frame(artifact)
    runner.atomic_csv_write(csv_output, csv_frame)
    artifact["outputs"]["csv_sha256"] = _stable_file_sha(csv_output)
    artifact["outputs"]["json_is_commit_marker_for_csv_sha256"] = True
    runner.atomic_json_write(json_output, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a completed L89 or explicitly versioned EMIT ragged "
            "four-arm inner-validation run and apply the preregistered pilot "
            "gate."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        help=f"Default: RUN_DIR/{DEFAULT_JSON_NAME}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help=f"Default: RUN_DIR/{DEFAULT_CSV_NAME}",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    output_json = (
        args.output_json
        if args.output_json is not None
        else run_dir / DEFAULT_JSON_NAME
    )
    output_csv = (
        args.output_csv
        if args.output_csv is not None
        else run_dir / DEFAULT_CSV_NAME
    )
    artifact = evaluate_run(
        run_dir,
        output_json=output_json,
        output_csv=output_csv,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "gate_pass": artifact["gate_decision"]["pass"],
                "output_json": str(Path(output_json).expanduser().resolve()),
                "output_csv": str(Path(output_csv).expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
