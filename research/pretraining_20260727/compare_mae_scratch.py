#!/usr/bin/env python3
"""CPU-only, input-read-only gate for Residual-MAE versus matched scratch.

The checker reads exactly three ``metrics_history.json`` files and the MAE
source checkpoint supplied on the command line.  It never resolves manifests,
loads imagery, imports the training runner, or evaluates a model.  By default
the complete JSON decision is printed to stdout; ``--output-json`` optionally
writes the same derived report atomically without modifying any input.

The legacy profile requires all three run signatures to be byte-for-byte
equivalent.  The validity-masked profile is deliberately narrower: masked
pretraining and its finetuning run must have the exact same validity objective
contract, while the legacy scratch run may omit only the registered
reconstruction-only fields.  Every supervised data, encoder architecture,
initialization-control, optimization, and evaluation field must still match.

Exit codes are 0 for pass, 1 for fail, and 2 for pending.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Enforce CPU-only checkpoint deserialization even on a GPU host.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402


EXPECTED_SENSORS = ("s2", "l89", "emit", "s5p")
LEGACY_PROFILE = "legacy"
VALIDITY_MASKED_PROFILE = "validity-masked"
VALIDITY_OBJECTIVE_VERSION = "masked_valid_element_per_sample_stream_v1"
VALIDITY_SEMANTICS = (
    "per_channel_native_validity;"
    "tiff_finite_nonzero;"
    "s5p_finite;"
    "residual_validity_intersection;"
    "resize_area_down_nearest_up;"
    f"{VALIDITY_OBJECTIVE_VERSION}"
)
EXPECTED_VALIDITY_NATIVE = {
    "tiff": "finite_and_nonzero",
    "s5p": "finite_including_zero",
}
EXPECTED_VALIDITY_STREAMS = {
    "current": "t0",
    "recent": "t0_and_prev1",
    "seasonal": "t0_and_seasonal",
}
EXPECTED_VALIDITY_RESIZE = {
    "downsample": "area",
    "upsample": "nearest",
}
EXPECTED_VALIDITY_NORMALIZATION = (
    "valid_elements_per_sample_then_equal_valid_samples_per_stream"
    "_then_equal_nonempty_streams"
)
BASE_GATE_SPEC = {
    "name": "four-sensor-residual-mae-e1-promotion-v1",
    "primary_metric": "macro_mean_over_sensor_average_precision",
    "macro_ap_min_delta": 0.010,
    "minimum_nondecreasing_sensor_ap_count": 3,
    "worst_sensor_ap_min_delta": -0.010,
    "macro_best_f1_min_delta": -0.005,
    "notes": (
        "One epoch and one seed are a promotion screen, not a paper claim. "
        "AUROC and F1@0.5 are reported but are not promotion thresholds."
    ),
}
CLASSIFICATION_METRICS = {
    "ap": "ap",
    "auroc": "auroc",
    "best_f1": "f1",
    "f1_at_0p5": "f1_0p5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare globally event-purged MAE finetuning with matched scratch "
            "without reading any dataset or using a GPU."
        )
    )
    parser.add_argument("--pretrain-history", type=Path, required=False)
    parser.add_argument("--pretrain-checkpoint", type=Path, required=False)
    parser.add_argument("--scratch-history", type=Path, required=False)
    parser.add_argument("--finetune-history", type=Path, required=False)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional atomic path for the derived decision; stdout is always emitted.",
    )
    parser.add_argument(
        "--comparison-profile",
        choices=(LEGACY_PROFILE, VALIDITY_MASKED_PROFILE),
        default=LEGACY_PROFILE,
        help=(
            "Use 'validity-masked' only for a masked pretrain + masked "
            "finetune comparison against the registered legacy scratch run."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic CPU-only pass/fail/pending tests in a temporary directory.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gate_spec(profile: str) -> Dict[str, Any]:
    spec = copy.deepcopy(BASE_GATE_SPEC)
    if profile == VALIDITY_MASKED_PROFILE:
        spec["name"] = (
            "four-sensor-validity-masked-residual-mae-e1-promotion-v1"
        )
        spec["notes"] = (
            BASE_GATE_SPEC["notes"]
            + " The scratch control may omit only preregistered "
            "reconstruction-objective signature fields."
        )
    return spec


def artifact_type(profile: str) -> str:
    if profile == VALIDITY_MASKED_PROFILE:
        return "validity_masked_mae_scratch_gate_decision"
    return "mae_scratch_gate_decision"


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
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def last_epoch(history: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    epochs = history.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError(f"{role} history has no completed epoch records.")
    records = [record for record in epochs if isinstance(record, Mapping)]
    if len(records) != len(epochs):
        raise ValueError(f"{role} history contains a non-object epoch record.")
    return max(records, key=lambda record: int(record.get("epoch", -1)))


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


def pending_result(
    paths: Mapping[str, Path],
    reasons: Sequence[Mapping[str, Any]],
    *,
    profile: str,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": artifact_type(profile),
        "comparison_profile": profile,
        "created_utc": utc_now(),
        "status": "pending",
        "preregistered_gate": gate_spec(profile),
        "inputs": {
            role: {"path": str(path.resolve()), "exists": path.is_file()}
            for role, path in paths.items()
        },
        "pending_reasons": list(reasons),
        "checks": {},
        "metrics": None,
        "gate_decision": None,
    }


def fail_result(
    paths: Mapping[str, Path],
    error: Exception,
    *,
    profile: str,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": artifact_type(profile),
        "comparison_profile": profile,
        "created_utc": utc_now(),
        "status": "fail",
        "preregistered_gate": gate_spec(profile),
        "inputs": {
            role: {"path": str(path.resolve()), "exists": path.is_file()}
            for role, path in paths.items()
        },
        "errors": [f"{type(error).__name__}: {error}"],
        "checks": {},
        "metrics": None,
        "gate_decision": {
            "pass": False,
            "failed_criteria": ["artifact_parse_or_schema_error"],
        },
    }


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def mapping_at(value: Any, *keys: str) -> Optional[Mapping[str, Any]]:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def validity_data_contract(
    data_signature: Any,
) -> Optional[Mapping[str, Any]]:
    return mapping_at(
        data_signature,
        "representation",
        "validity_reconstruction",
    )


def canonical_data_signature(data_signature: Any) -> Any:
    """Remove only the registered reconstruction-only data contract."""

    if not isinstance(data_signature, Mapping):
        return data_signature
    canonical = copy.deepcopy(dict(data_signature))
    # A masked signature bumps its schema solely to register the nested
    # objective contract.  The exact +1 relation is checked separately.
    canonical.pop("schema_version", None)
    representation = canonical.get("representation")
    if isinstance(representation, Mapping):
        canonical_representation = dict(representation)
        canonical_representation.pop("validity_reconstruction", None)
        canonical["representation"] = canonical_representation
    return canonical


def validity_encoder_contract(
    encoder_signature: Any,
) -> Optional[Mapping[str, Any]]:
    return mapping_at(encoder_signature, "validity_adapter")


def canonical_encoder_signature(encoder_signature: Any) -> Any:
    """Project a masked signature to the parameter-bearing architecture."""

    if not isinstance(encoder_signature, Mapping):
        return encoder_signature
    canonical = copy.deepcopy(dict(encoder_signature))
    # The masked signature bumps its schema solely to register this adapter.
    # Schema compatibility is checked separately before removing the version.
    canonical.pop("validity_adapter", None)
    canonical.pop("schema_version", None)
    return canonical


def canonical_supervised_resume_signature(resume_signature: Any) -> Any:
    """Remove only objective-only fields before FT versus scratch matching."""

    if not isinstance(resume_signature, Mapping):
        return resume_signature
    canonical = copy.deepcopy(dict(resume_signature))
    # The exact masked-versus-legacy +1 schema relation is checked separately.
    canonical.pop("schema_version", None)
    canonical.pop("validity_reconstruction", None)
    canonical["data_signature"] = canonical_data_signature(
        canonical.get("data_signature")
    )
    canonical["encoder_signature"] = canonical_encoder_signature(
        canonical.get("encoder_signature")
    )
    return canonical


def validity_contract_is_masked(contract: Any) -> bool:
    if not isinstance(contract, Mapping):
        return False
    source_sha = contract.get("adapter_source_sha256")
    return (
        contract.get("enabled") is True
        and contract.get("objective_version")
        == VALIDITY_OBJECTIVE_VERSION
        and contract.get("semantics") == VALIDITY_SEMANTICS
        and is_sha256(source_sha)
        and contract.get("native_validity") == EXPECTED_VALIDITY_NATIVE
        and contract.get("stream_validity") == EXPECTED_VALIDITY_STREAMS
        and contract.get("resize") == EXPECTED_VALIDITY_RESIZE
        and contract.get("loss_normalization")
        == EXPECTED_VALIDITY_NORMALIZATION
    )


def objective_version_from_contract(contract: Any) -> Any:
    if not isinstance(contract, Mapping):
        return None
    return contract.get("objective_version")


def extract_reconstruction(
    history: Mapping[str, Any], sensors: Sequence[str]
) -> Dict[str, Any]:
    epoch = last_epoch(history, "pretrain")
    train_loss = epoch.get("train_loss")
    validation = epoch.get("val")
    if not isinstance(train_loss, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("Pretrain epoch lacks train_loss or validation mappings.")
    output: Dict[str, Any] = {"epoch": int(epoch.get("epoch", -1)), "sensors": {}}
    for sensor in sensors:
        val_sensor = validation.get(sensor)
        if not isinstance(val_sensor, Mapping):
            raise ValueError(f"Pretrain validation lacks sensor {sensor}.")
        train_value = train_loss.get(sensor)
        val_value = val_sensor.get("reconstruction_loss")
        samples = val_sensor.get("samples")
        if not is_finite_number(train_value):
            raise ValueError(f"Non-finite pretrain train loss for {sensor}: {train_value}")
        if not is_finite_number(val_value):
            raise ValueError(
                f"Non-finite pretrain validation reconstruction for {sensor}: {val_value}"
            )
        if not isinstance(samples, int) or samples <= 0:
            raise ValueError(
                f"Pretrain validation has invalid sample count for {sensor}: {samples}"
            )
        output["sensors"][sensor] = {
            "train_loss": float(train_value),
            "validation_reconstruction_loss": float(val_value),
            "validation_samples": samples,
        }
    return output


def audit_validity_diagnostics(
    history: Mapping[str, Any],
    sensors: Sequence[str],
) -> Tuple[bool, Dict[str, Any], List[str]]:
    epoch = last_epoch(history, "pretrain")
    validation = epoch.get("val")
    if not isinstance(validation, Mapping):
        return False, {}, ["pretrain validation mapping is missing"]

    observed: Dict[str, Any] = {}
    failures: List[str] = []
    for sensor in sensors:
        sensor_record = validation.get(sensor)
        diagnostics = (
            sensor_record.get("validity_diagnostics")
            if isinstance(sensor_record, Mapping)
            else None
        )
        observed[sensor] = diagnostics
        if not isinstance(diagnostics, Mapping):
            failures.append(f"{sensor}: validity_diagnostics is missing")
            continue
        sample_count = sensor_record.get("samples")
        expected_streams = (
            f"{sensor}_current",
            f"{sensor}_recent",
            f"{sensor}_seasonal",
        )
        for stream in expected_streams:
            loss = diagnostics.get(f"loss_{stream}")
            valid_fraction = diagnostics.get(f"valid_fraction_{stream}")
            valid_elements = diagnostics.get(
                f"valid_masked_elements_{stream}"
            )
            valid_samples = diagnostics.get(f"valid_samples_{stream}")
            excluded_samples = diagnostics.get(
                f"excluded_samples_{stream}"
            )
            batch_samples = diagnostics.get(f"batch_samples_{stream}")
            if not is_finite_number(loss) or float(loss) < 0.0:
                failures.append(f"{sensor}/{stream}: invalid loss={loss!r}")
            if (
                not is_finite_number(valid_fraction)
                or not 0.0 < float(valid_fraction) <= 1.0
            ):
                failures.append(
                    f"{sensor}/{stream}: invalid valid_fraction="
                    f"{valid_fraction!r}"
                )
            if (
                not is_finite_number(valid_elements)
                or float(valid_elements) <= 0.0
            ):
                failures.append(
                    f"{sensor}/{stream}: no valid masked elements="
                    f"{valid_elements!r}"
                )
            if (
                not is_finite_number(valid_samples)
                or float(valid_samples) <= 0.0
                or not isinstance(sample_count, int)
                or not is_finite_number(excluded_samples)
                or float(excluded_samples) < 0.0
                or not is_finite_number(batch_samples)
                or float(batch_samples) != float(sample_count)
                or abs(
                    float(valid_samples)
                    + float(excluded_samples)
                    - float(batch_samples)
                )
                > 1e-6
            ):
                failures.append(
                    f"{sensor}/{stream}: inconsistent sample audit "
                    f"valid={valid_samples!r}, excluded="
                    f"{excluded_samples!r}, batch={batch_samples!r}, "
                    f"validation={sample_count!r}"
                )
        valid_streams = diagnostics.get("valid_streams")
        empty_streams = diagnostics.get("empty_streams")
        total_streams = diagnostics.get("total_streams")
        if (
            not is_finite_number(valid_streams)
            or float(valid_streams) <= 0.0
            or not is_finite_number(empty_streams)
            or float(empty_streams) < 0.0
            or not is_finite_number(total_streams)
            or float(total_streams) <= 0.0
            or abs(
                float(valid_streams)
                + float(empty_streams)
                - float(total_streams)
            )
            > 1e-6
        ):
            failures.append(
                f"{sensor}: inconsistent stream audit valid="
                f"{valid_streams!r}, empty={empty_streams!r}, "
                f"total={total_streams!r}"
            )
    return not failures, observed, failures


def extract_classification(
    history: Mapping[str, Any],
    role: str,
    sensors: Sequence[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Any]]:
    epoch = last_epoch(history, role)
    validation = epoch.get("val")
    if not isinstance(validation, Mapping):
        raise ValueError(f"{role} epoch lacks validation mapping.")
    by_sensor: Dict[str, Dict[str, float]] = {}
    for sensor in sensors:
        sensor_metrics = validation.get(sensor)
        if not isinstance(sensor_metrics, Mapping):
            raise ValueError(f"{role} validation lacks sensor {sensor}.")
        samples = sensor_metrics.get("samples")
        if not isinstance(samples, int) or samples <= 0:
            raise ValueError(f"{role}/{sensor} invalid validation samples: {samples}")
        by_sensor[sensor] = {}
        for output_name, history_name in CLASSIFICATION_METRICS.items():
            value = sensor_metrics.get(history_name)
            if not is_finite_number(value):
                raise ValueError(
                    f"{role}/{sensor} has non-finite {history_name}: {value}"
                )
            by_sensor[sensor][output_name] = float(value)

    macro = {
        metric: sum(by_sensor[sensor][metric] for sensor in sensors) / len(sensors)
        for metric in CLASSIFICATION_METRICS
    }
    logged = epoch.get("macro_over_sensor")
    logged_comparison: Dict[str, Any] = {}
    if not isinstance(logged, Mapping):
        raise ValueError(f"{role} epoch lacks macro_over_sensor.")
    for output_name, history_name in CLASSIFICATION_METRICS.items():
        logged_value = logged.get(history_name)
        if not is_finite_number(logged_value):
            raise ValueError(
                f"{role} has non-finite logged macro {history_name}: {logged_value}"
            )
        logged_comparison[output_name] = {
            "recomputed": macro[output_name],
            "logged": float(logged_value),
            "absolute_error": abs(macro[output_name] - float(logged_value)),
        }
    return by_sensor, macro, logged_comparison


def compare_artifacts(
    *,
    pretrain_history_path: Path,
    pretrain_checkpoint_path: Path,
    scratch_history_path: Path,
    finetune_history_path: Path,
    comparison_profile: str = LEGACY_PROFILE,
) -> Dict[str, Any]:
    if comparison_profile not in (LEGACY_PROFILE, VALIDITY_MASKED_PROFILE):
        raise ValueError(f"Unsupported comparison profile: {comparison_profile}")
    paths = {
        "pretrain_history": Path(pretrain_history_path),
        "pretrain_checkpoint": Path(pretrain_checkpoint_path),
        "scratch_history": Path(scratch_history_path),
        "finetune_history": Path(finetune_history_path),
    }
    missing = [
        {"role": role, "path": str(path.resolve()), "reason": "file_not_found"}
        for role, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        return pending_result(paths, missing, profile=comparison_profile)

    try:
        pretrain = load_json(paths["pretrain_history"])
        scratch = load_json(paths["scratch_history"])
        finetune = load_json(paths["finetune_history"])
    except Exception as error:
        return fail_result(paths, error, profile=comparison_profile)

    running = []
    for role, history in (
        ("pretrain_history", pretrain),
        ("scratch_history", scratch),
        ("finetune_history", finetune),
    ):
        status = history.get("status")
        epochs = history.get("epochs")
        if status == "running" or not isinstance(epochs, list) or not epochs:
            running.append(
                {
                    "role": role,
                    "path": str(paths[role].resolve()),
                    "reason": "run_not_completed",
                    "observed_status": status,
                    "completed_epochs": len(epochs) if isinstance(epochs, list) else None,
                }
            )
    if running:
        return pending_result(paths, running, profile=comparison_profile)

    checks: Dict[str, Dict[str, Any]] = {}
    inputs = {
        role: {
            "path": str(path.resolve()),
            "exists": True,
            "sha256": sha256_file(path),
        }
        for role, path in paths.items()
    }
    histories = {
        "pretrain": pretrain,
        "scratch": scratch,
        "finetune": finetune,
    }

    add_check(
        checks,
        "history_status_completed",
        all(history.get("status") == "completed" for history in histories.values()),
        observed={role: history.get("status") for role, history in histories.items()},
        expected={role: "completed" for role in histories},
    )
    add_check(
        checks,
        "history_modes",
        pretrain.get("mode") == "pretrain"
        and scratch.get("mode") == "supervised"
        and finetune.get("mode") == "supervised",
        observed={role: history.get("mode") for role, history in histories.items()},
        expected={
            "pretrain": "pretrain",
            "scratch": "supervised",
            "finetune": "supervised",
        },
    )
    add_check(
        checks,
        "shared_encoder_protocol",
        all(history.get("sharing") == "shared" for history in histories.values()),
        observed={role: history.get("sharing") for role, history in histories.items()},
        expected={role: "shared" for role in histories},
    )
    add_check(
        checks,
        "sensor_order",
        all(
            history.get("sensors") == list(EXPECTED_SENSORS)
            for history in histories.values()
        ),
        observed={role: history.get("sensors") for role, history in histories.items()},
        expected=list(EXPECTED_SENSORS),
    )

    overlaps = {
        role: history.get("event_protocol", {}).get(
            "global_train_val_overlap_after"
        )
        for role, history in histories.items()
    }
    add_check(
        checks,
        "global_train_validation_overlap_zero",
        all(value == 0 for value in overlaps.values()),
        observed=overlaps,
        expected={role: 0 for role in histories},
    )
    signature_overlap = {
        role: history.get("data_signature", {}).get(
            "global_train_val_overlap_after"
        )
        for role, history in histories.items()
    }
    add_check(
        checks,
        "signed_global_overlap_zero",
        all(value == 0 for value in signature_overlap.values()),
        observed=signature_overlap,
        expected={role: 0 for role in histories},
    )

    data_signatures = {
        role: history.get("data_signature") for role, history in histories.items()
    }
    data_fingerprints = {
        role: json_fingerprint(value) for role, value in data_signatures.items()
    }
    encoder_signatures = {
        role: history.get("encoder_signature")
        for role, history in histories.items()
    }
    encoder_fingerprints = {
        role: json_fingerprint(value) for role, value in encoder_signatures.items()
    }

    scratch_resume = scratch.get("resume_signature")
    finetune_resume = finetune.get("resume_signature")
    if comparison_profile == LEGACY_PROFILE:
        add_check(
            checks,
            "data_signatures_identical",
            all(
                isinstance(value, Mapping)
                for value in data_signatures.values()
            )
            and len(set(data_fingerprints.values())) == 1,
            observed=data_fingerprints,
            expected="three non-null mappings with one identical fingerprint",
        )
        add_check(
            checks,
            "encoder_signatures_identical",
            all(
                isinstance(value, Mapping)
                for value in encoder_signatures.values()
            )
            and len(set(encoder_fingerprints.values())) == 1,
            observed=encoder_fingerprints,
            expected="three non-null mappings with one identical fingerprint",
        )
        add_check(
            checks,
            "matched_supervised_run_signature",
            scratch_resume == finetune_resume and scratch_resume is not None,
            observed={
                "scratch": json_fingerprint(scratch_resume),
                "finetune": json_fingerprint(finetune_resume),
            },
            expected="identical non-null fingerprints",
            detail=(
                "Only encoder initialization may differ between finetune "
                "and scratch."
            ),
        )
    else:
        pretrain_data_contract = validity_data_contract(
            data_signatures["pretrain"]
        )
        finetune_data_contract = validity_data_contract(
            data_signatures["finetune"]
        )
        scratch_data_contract = validity_data_contract(
            data_signatures["scratch"]
        )
        add_check(
            checks,
            "masked_data_objective_pretrain_finetune_exact",
            validity_contract_is_masked(pretrain_data_contract)
            and pretrain_data_contract == finetune_data_contract,
            observed={
                "pretrain": pretrain_data_contract,
                "finetune": finetune_data_contract,
            },
            expected={
                "enabled": True,
                "objective_version": VALIDITY_OBJECTIVE_VERSION,
                "semantics": VALIDITY_SEMANTICS,
                "adapter_source_sha256": "64 lowercase hex characters",
                "native_validity": EXPECTED_VALIDITY_NATIVE,
                "stream_validity": EXPECTED_VALIDITY_STREAMS,
                "resize": EXPECTED_VALIDITY_RESIZE,
                "loss_normalization": EXPECTED_VALIDITY_NORMALIZATION,
                "relationship": "pretrain == finetune",
            },
        )
        add_check(
            checks,
            "scratch_omits_masked_reconstruction_objective",
            scratch_data_contract is None,
            observed=scratch_data_contract,
            expected=None,
            detail=(
                "The registered legacy scratch run need not be rebuilt with "
                "a reconstruction loss it never optimizes."
            ),
        )
        history_contracts = {
            role: history.get("validity_reconstruction")
            for role, history in histories.items()
        }
        add_check(
            checks,
            "masked_history_objective_pretrain_finetune_exact",
            history_contracts["pretrain"]
            == history_contracts["finetune"]
            == pretrain_data_contract
            and history_contracts["scratch"] is None,
            observed=history_contracts,
            expected=(
                "pretrain == finetune == masked data contract; "
                "legacy scratch omitted"
            ),
        )
        history_schema_versions = {
            role: history.get("schema_version")
            for role, history in histories.items()
        }
        scratch_history_schema = history_schema_versions["scratch"]
        add_check(
            checks,
            "masked_history_schema_bump_is_registered_only_delta",
            isinstance(scratch_history_schema, int)
            and history_schema_versions["pretrain"]
            == history_schema_versions["finetune"]
            == scratch_history_schema + 1,
            observed=history_schema_versions,
            expected=(
                "masked pretrain == masked finetune == legacy scratch + 1"
            ),
        )
        data_schema_versions = {
            role: (
                value.get("schema_version")
                if isinstance(value, Mapping)
                else None
            )
            for role, value in data_signatures.items()
        }
        scratch_data_schema = data_schema_versions["scratch"]
        add_check(
            checks,
            "masked_data_schema_bump_is_registered_only_delta",
            isinstance(scratch_data_schema, int)
            and data_schema_versions["pretrain"]
            == data_schema_versions["finetune"]
            == scratch_data_schema + 1,
            observed=data_schema_versions,
            expected=(
                "masked pretrain == masked finetune == legacy scratch + 1"
            ),
        )
        canonical_data = {
            role: canonical_data_signature(value)
            for role, value in data_signatures.items()
        }
        canonical_data_fingerprints = {
            role: json_fingerprint(value)
            for role, value in canonical_data.items()
        }
        add_check(
            checks,
            "matched_base_data_protocol_after_objective_projection",
            all(
                isinstance(value, Mapping)
                for value in canonical_data.values()
            )
            and len(set(canonical_data_fingerprints.values())) == 1,
            observed=canonical_data_fingerprints,
            expected=(
                "identical manifest/stat/protocol/representation signatures "
                "after removing only representation.validity_reconstruction"
            ),
        )

        pretrain_encoder_contract = validity_encoder_contract(
            encoder_signatures["pretrain"]
        )
        finetune_encoder_contract = validity_encoder_contract(
            encoder_signatures["finetune"]
        )
        scratch_encoder_contract = validity_encoder_contract(
            encoder_signatures["scratch"]
        )
        adapter_objectives = {
            "pretrain": objective_version_from_contract(
                pretrain_encoder_contract
            ),
            "finetune": objective_version_from_contract(
                finetune_encoder_contract
            ),
        }
        add_check(
            checks,
            "masked_encoder_adapter_pretrain_finetune_exact",
            isinstance(pretrain_encoder_contract, Mapping)
            and pretrain_encoder_contract == finetune_encoder_contract
            and pretrain_encoder_contract.get("class")
            == "ValidityMaskedMethaneResidualMAE"
            and is_sha256(
                pretrain_encoder_contract.get("source_sha256")
            )
            and pretrain_encoder_contract.get("source_sha256")
            == (
                pretrain_data_contract.get("adapter_source_sha256")
                if isinstance(pretrain_data_contract, Mapping)
                else None
            )
            and all(
                value == VALIDITY_OBJECTIVE_VERSION
                for value in adapter_objectives.values()
            ),
            observed={
                "pretrain": pretrain_encoder_contract,
                "finetune": finetune_encoder_contract,
            },
            expected={
                "relationship": "pretrain == finetune",
                "class": "ValidityMaskedMethaneResidualMAE",
                "source_sha256": (
                    "64 lowercase hex characters matching the data contract"
                ),
                "objective_version": VALIDITY_OBJECTIVE_VERSION,
            },
        )
        add_check(
            checks,
            "scratch_omits_validity_adapter",
            scratch_encoder_contract is None,
            observed=scratch_encoder_contract,
            expected=None,
        )
        encoder_schema_versions = {
            role: (
                value.get("schema_version")
                if isinstance(value, Mapping)
                else None
            )
            for role, value in encoder_signatures.items()
        }
        scratch_encoder_schema = encoder_schema_versions["scratch"]
        add_check(
            checks,
            "masked_encoder_schema_bump_is_registered_only_delta",
            isinstance(scratch_encoder_schema, int)
            and encoder_schema_versions["pretrain"]
            == encoder_schema_versions["finetune"]
            == scratch_encoder_schema + 1,
            observed=encoder_schema_versions,
            expected=(
                "masked pretrain == masked finetune == legacy scratch + 1"
            ),
        )
        canonical_encoders = {
            role: canonical_encoder_signature(value)
            for role, value in encoder_signatures.items()
        }
        canonical_encoder_fingerprints = {
            role: json_fingerprint(value)
            for role, value in canonical_encoders.items()
        }
        add_check(
            checks,
            "matched_parameter_bearing_encoder_architecture",
            all(
                isinstance(value, Mapping)
                for value in canonical_encoders.values()
            )
            and len(set(canonical_encoder_fingerprints.values())) == 1,
            observed=canonical_encoder_fingerprints,
            expected=(
                "identical encoder signatures after removing only "
                "validity_adapter and its registered schema bump"
            ),
        )

        pretrain_resume = pretrain.get("resume_signature")
        resume_objectives = {
            role: (
                resume.get("validity_reconstruction")
                if isinstance(resume, Mapping)
                else None
            )
            for role, resume in (
                ("pretrain", pretrain_resume),
                ("finetune", finetune_resume),
            )
        }
        add_check(
            checks,
            "masked_resume_objective_pretrain_finetune_exact",
            isinstance(resume_objectives["pretrain"], Mapping)
            and resume_objectives["pretrain"]
            == resume_objectives["finetune"]
            == pretrain_data_contract,
            observed=resume_objectives,
            expected=(
                "pretrain resume == finetune resume == masked data contract"
            ),
        )
        resume_schema_versions = {
            "pretrain": (
                pretrain_resume.get("schema_version")
                if isinstance(pretrain_resume, Mapping)
                else None
            ),
            "scratch": (
                scratch_resume.get("schema_version")
                if isinstance(scratch_resume, Mapping)
                else None
            ),
            "finetune": (
                finetune_resume.get("schema_version")
                if isinstance(finetune_resume, Mapping)
                else None
            ),
        }
        scratch_resume_schema = resume_schema_versions["scratch"]
        add_check(
            checks,
            "masked_resume_schema_bump_is_registered_only_delta",
            isinstance(scratch_resume_schema, int)
            and resume_schema_versions["pretrain"]
            == resume_schema_versions["finetune"]
            == scratch_resume_schema + 1,
            observed=resume_schema_versions,
            expected=(
                "masked pretrain == masked finetune == legacy scratch + 1"
            ),
        )
        pretrain_model_signature = (
            pretrain_resume.get("pretrain_model")
            if isinstance(pretrain_resume, Mapping)
            else None
        )
        add_check(
            checks,
            "pretrain_model_declares_masked_objective",
            isinstance(pretrain_model_signature, Mapping)
            and pretrain_model_signature.get(
                "reconstruction_objective"
            )
            == VALIDITY_OBJECTIVE_VERSION,
            observed=(
                pretrain_model_signature.get("reconstruction_objective")
                if isinstance(pretrain_model_signature, Mapping)
                else None
            ),
            expected=VALIDITY_OBJECTIVE_VERSION,
        )
        optimization_signatures = {
            role: (
                resume.get("optimization")
                if isinstance(resume, Mapping)
                else None
            )
            for role, resume in (
                ("pretrain", pretrain_resume),
                ("scratch", scratch_resume),
                ("finetune", finetune_resume),
            )
        }
        add_check(
            checks,
            "matched_loader_optimizer_and_seed_controls_across_modes",
            all(
                isinstance(value, Mapping)
                for value in optimization_signatures.values()
            )
            and len(
                {
                    json_fingerprint(value)
                    for value in optimization_signatures.values()
                }
            )
            == 1,
            observed={
                role: json_fingerprint(value)
                for role, value in optimization_signatures.items()
            },
            expected=(
                "identical batch size, optimizer hyperparameters, "
                "augmentation, seed, balanced rounds, steps, and loader "
                "lengths across pretrain/scratch/finetune"
            ),
        )
        canonical_supervised_resumes = {
            "scratch": canonical_supervised_resume_signature(scratch_resume),
            "finetune": canonical_supervised_resume_signature(finetune_resume),
        }
        canonical_supervised_resume_fingerprints = {
            role: json_fingerprint(value)
            for role, value in canonical_supervised_resumes.items()
        }
        add_check(
            checks,
            "matched_supervised_run_signature_after_objective_projection",
            all(
                isinstance(value, Mapping)
                for value in canonical_supervised_resumes.values()
            )
            and canonical_supervised_resumes["scratch"]
            == canonical_supervised_resumes["finetune"],
            observed=canonical_supervised_resume_fingerprints,
            expected=(
                "scratch == finetune after removing only registered "
                "reconstruction-objective fields"
            ),
            detail=(
                "This retains the complete supervised data, architecture, "
                "optimizer, seed, augmentation, and evaluation budget."
            ),
        )
    supervised_budget = {
        role: {
            "rounds": history.get("balanced_rounds_per_epoch"),
            "optimizer_steps": history.get("optimizer_steps_per_epoch"),
            "model_parameters": history.get("model_parameters"),
            "trainable_parameters": history.get("trainable_parameters"),
        }
        for role, history in (("scratch", scratch), ("finetune", finetune))
    }
    add_check(
        checks,
        "matched_supervised_update_budget_and_model_size",
        supervised_budget["scratch"] == supervised_budget["finetune"],
        observed=supervised_budget,
        expected="scratch == finetune",
    )
    if comparison_profile == VALIDITY_MASKED_PROFILE:
        configs = {
            role: (
                history.get("config")
                if isinstance(history.get("config"), Mapping)
                else {}
            )
            for role, history in histories.items()
        }
        validity_flags = {
            role: config.get("validity_masked_reconstruction", False)
            for role, config in configs.items()
        }
        add_check(
            checks,
            "masked_cli_flag_pretrain_finetune_only",
            validity_flags
            == {
                "pretrain": True,
                "scratch": False,
                "finetune": True,
            },
            observed=validity_flags,
            expected={
                "pretrain": True,
                "scratch": False,
                "finetune": True,
            },
        )
        epoch_controls = {
            role: {
                "configured_epochs": configs[role].get("epochs"),
                "completed_epoch_indices": [
                    record.get("epoch")
                    for record in history.get("epochs", [])
                    if isinstance(record, Mapping)
                ],
            }
            for role, history in histories.items()
        }
        add_check(
            checks,
            "matched_one_epoch_screen",
            all(
                control["configured_epochs"] == 1
                and control["completed_epoch_indices"] == [0]
                for control in epoch_controls.values()
            ),
            observed=epoch_controls,
            expected={
                role: {
                    "configured_epochs": 1,
                    "completed_epoch_indices": [0],
                }
                for role in histories
            },
        )
        seeds = {
            role: config.get("seed") for role, config in configs.items()
        }
        add_check(
            checks,
            "matched_seed",
            all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in seeds.values()
            )
            and len(set(seeds.values())) == 1,
            observed=seeds,
            expected="one identical integer seed across all three runs",
        )
        initialization_controls = {
            role: {
                "config_init_checkpoint": configs[role].get(
                    "init_checkpoint"
                ),
                "config_resume": configs[role].get("resume"),
                "has_init_report": isinstance(
                    history.get("init_checkpoint"), Mapping
                ),
            }
            for role, history in histories.items()
        }
        add_check(
            checks,
            "controlled_initialization_paths",
            initialization_controls["pretrain"]
            == {
                "config_init_checkpoint": None,
                "config_resume": None,
                "has_init_report": False,
            }
            and initialization_controls["scratch"]
            == {
                "config_init_checkpoint": None,
                "config_resume": None,
                "has_init_report": False,
            }
            and initialization_controls["finetune"][
                "config_resume"
            ]
            is None
            and isinstance(
                initialization_controls["finetune"][
                    "config_init_checkpoint"
                ],
                str,
            )
            and bool(
                initialization_controls["finetune"][
                    "config_init_checkpoint"
                ]
            )
            and initialization_controls["finetune"]["has_init_report"],
            observed=initialization_controls,
            expected=(
                "pretrain/scratch start without init or resume; finetune "
                "uses exactly one init checkpoint and no resume"
            ),
        )
        validation_sample_counts = {
            role: {
                sensor: last_epoch(history, role).get("val", {})
                .get(sensor, {})
                .get("samples")
                for sensor in EXPECTED_SENSORS
            }
            for role, history in (
                ("scratch", scratch),
                ("finetune", finetune),
            )
        }
        add_check(
            checks,
            "matched_supervised_validation_sample_counts",
            validation_sample_counts["scratch"]
            == validation_sample_counts["finetune"]
            and all(
                isinstance(value, int) and value > 0
                for value in validation_sample_counts["scratch"].values()
            ),
            observed=validation_sample_counts,
            expected="identical positive full-validation counts per sensor",
        )

    try:
        checkpoint = torch.load(
            paths["pretrain_checkpoint"], map_location=torch.device("cpu")
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Pretrain checkpoint is not a mapping.")
    except Exception as error:
        return fail_result(paths, error, profile=comparison_profile)

    add_check(
        checks,
        "checkpoint_declares_pretrain",
        checkpoint.get("mode") == "pretrain",
        observed=checkpoint.get("mode"),
        expected="pretrain",
    )
    add_check(
        checks,
        "checkpoint_data_signature",
        isinstance(checkpoint.get("data_signature"), Mapping)
        and checkpoint.get("data_signature") == pretrain.get("data_signature"),
        observed=json_fingerprint(checkpoint.get("data_signature")),
        expected=json_fingerprint(pretrain.get("data_signature")),
    )
    add_check(
        checks,
        "checkpoint_encoder_signature",
        isinstance(checkpoint.get("encoder_signature"), Mapping)
        and checkpoint.get("encoder_signature")
        == pretrain.get("encoder_signature"),
        observed=json_fingerprint(checkpoint.get("encoder_signature")),
        expected=json_fingerprint(pretrain.get("encoder_signature")),
    )
    pretrain_epoch = last_epoch(pretrain, "pretrain")
    add_check(
        checks,
        "checkpoint_matches_last_pretrain_epoch",
        int(checkpoint.get("epoch", -2)) == int(pretrain_epoch.get("epoch", -1)),
        observed=checkpoint.get("epoch"),
        expected=pretrain_epoch.get("epoch"),
    )
    add_check(
        checks,
        "checkpoint_has_model_state",
        isinstance(checkpoint.get("model"), Mapping),
        observed=type(checkpoint.get("model")).__name__,
        expected="mapping",
    )
    if comparison_profile == VALIDITY_MASKED_PROFILE:
        checkpoint_config = checkpoint.get("config")
        checkpoint_resume = checkpoint.get("resume_signature")
        add_check(
            checks,
            "checkpoint_declares_masked_objective",
            isinstance(checkpoint_config, Mapping)
            and checkpoint_config.get(
                "validity_masked_reconstruction"
            )
            is True,
            observed=(
                checkpoint_config.get("validity_masked_reconstruction")
                if isinstance(checkpoint_config, Mapping)
                else None
            ),
            expected=True,
        )
        add_check(
            checks,
            "checkpoint_masked_contract_matches_pretrain",
            checkpoint.get("validity_reconstruction")
            == pretrain_data_contract,
            observed=checkpoint.get("validity_reconstruction"),
            expected=pretrain_data_contract,
        )
        add_check(
            checks,
            "checkpoint_resume_signature_matches_pretrain",
            isinstance(checkpoint_resume, Mapping)
            and checkpoint_resume == pretrain.get("resume_signature"),
            observed=json_fingerprint(checkpoint_resume),
            expected=json_fingerprint(pretrain.get("resume_signature")),
        )

    init_report = finetune.get("init_checkpoint")
    if not isinstance(init_report, Mapping):
        init_report = {}
    checkpoint_sha = inputs["pretrain_checkpoint"]["sha256"]
    init_path = init_report.get("path")
    path_matches = False
    if isinstance(init_path, str) and init_path:
        path_matches = Path(init_path).resolve() == paths["pretrain_checkpoint"].resolve()
    add_check(
        checks,
        "finetune_source_checkpoint_identity",
        path_matches and init_report.get("checkpoint_sha256") == checkpoint_sha,
        observed={
            "path": init_path,
            "sha256": init_report.get("checkpoint_sha256"),
        },
        expected={
            "path": str(paths["pretrain_checkpoint"].resolve()),
            "sha256": checkpoint_sha,
        },
    )
    coverage = init_report.get("encoder_coverage")
    loaded_numel = init_report.get("loaded_encoder_numel")
    target_numel = init_report.get("target_encoder_numel")
    add_check(
        checks,
        "complete_encoder_transfer",
        is_finite_number(coverage)
        and float(coverage) == 1.0
        and isinstance(loaded_numel, int)
        and loaded_numel > 0
        and loaded_numel == target_numel,
        observed={
            "coverage": coverage,
            "loaded_encoder_numel": loaded_numel,
            "target_encoder_numel": target_numel,
        },
        expected={
            "coverage": 1.0,
            "loaded_encoder_numel": "positive and equal to target_encoder_numel",
        },
    )
    decoder_exclusions = init_report.get("excluded_decoder_keys")
    checkpoint_head_exclusions = init_report.get(
        "excluded_checkpoint_head_keys"
    )
    add_check(
        checks,
        "transfer_exclusions_audited",
        isinstance(decoder_exclusions, list)
        and len(decoder_exclusions) > 0
        and isinstance(checkpoint_head_exclusions, list),
        observed={
            "decoder_key_count": (
                len(decoder_exclusions)
                if isinstance(decoder_exclusions, list)
                else None
            ),
            "legacy_checkpoint_head_key_count": (
                len(checkpoint_head_exclusions)
                if isinstance(checkpoint_head_exclusions, list)
                else None
            ),
        },
        expected="nonempty decoder list and an explicit legacy-head list",
    )
    head_before = init_report.get("target_head_sha256_before")
    head_after = init_report.get("target_head_sha256_after")
    scratch_head = scratch.get("initial_head_sha256")
    finetune_head = finetune.get("initial_head_sha256")
    add_check(
        checks,
        "head_unchanged_and_matches_scratch",
        isinstance(head_before, str)
        and bool(head_before)
        and head_before == head_after == scratch_head == finetune_head,
        observed={
            "transfer_before": head_before,
            "transfer_after": head_after,
            "scratch_initial": scratch_head,
            "finetune_initial": finetune_head,
        },
        expected="all four SHA-256 values identical and nonempty",
    )

    try:
        reconstruction = extract_reconstruction(pretrain, EXPECTED_SENSORS)
        scratch_by_sensor, scratch_macro, scratch_logged = extract_classification(
            scratch, "scratch", EXPECTED_SENSORS
        )
        finetune_by_sensor, finetune_macro, finetune_logged = (
            extract_classification(finetune, "finetune", EXPECTED_SENSORS)
        )
    except Exception as error:
        return fail_result(paths, error, profile=comparison_profile)

    add_check(
        checks,
        "finite_four_sensor_reconstruction",
        True,
        observed={
            sensor: {
                "train_loss": values["train_loss"],
                "validation_reconstruction_loss": values[
                    "validation_reconstruction_loss"
                ],
            }
            for sensor, values in reconstruction["sensors"].items()
        },
        expected="finite train and validation reconstruction losses for all sensors",
    )
    if comparison_profile == VALIDITY_MASKED_PROFILE:
        (
            validity_diagnostics_pass,
            validity_diagnostics,
            validity_diagnostic_failures,
        ) = audit_validity_diagnostics(pretrain, EXPECTED_SENSORS)
        add_check(
            checks,
            "nonzero_plausible_four_sensor_validity_diagnostics",
            validity_diagnostics_pass,
            observed={
                "by_sensor": validity_diagnostics,
                "failures": validity_diagnostic_failures,
            },
            expected=(
                "finite nonnegative loss, 0 < valid_fraction <= 1, positive "
                "valid masked elements/samples, and internally consistent "
                "sample/stream count audits for each sensor"
            ),
        )
    macro_errors = {
        "scratch": {
            metric: values["absolute_error"]
            for metric, values in scratch_logged.items()
        },
        "finetune": {
            metric: values["absolute_error"]
            for metric, values in finetune_logged.items()
        },
    }
    add_check(
        checks,
        "logged_macro_metrics_match_recomputed",
        all(
            error <= 1e-9
            for role_errors in macro_errors.values()
            for error in role_errors.values()
        ),
        observed=macro_errors,
        expected="absolute error <= 1e-9",
    )

    per_sensor: Dict[str, Any] = {}
    for sensor in EXPECTED_SENSORS:
        per_sensor[sensor] = {
            "scratch": scratch_by_sensor[sensor],
            "finetune": finetune_by_sensor[sensor],
            "delta_finetune_minus_scratch": {
                metric: finetune_by_sensor[sensor][metric]
                - scratch_by_sensor[sensor][metric]
                for metric in CLASSIFICATION_METRICS
            },
        }
    macro_delta = {
        metric: finetune_macro[metric] - scratch_macro[metric]
        for metric in CLASSIFICATION_METRICS
    }
    metrics = {
        "per_sensor": per_sensor,
        "macro_mean_over_sensors": {
            "scratch": scratch_macro,
            "finetune": finetune_macro,
            "delta_finetune_minus_scratch": macro_delta,
        },
        "logged_macro_consistency": {
            "scratch": scratch_logged,
            "finetune": finetune_logged,
        },
    }

    ap_deltas = {
        sensor: per_sensor[sensor]["delta_finetune_minus_scratch"]["ap"]
        for sensor in EXPECTED_SENSORS
    }
    nondecreasing_count = sum(delta >= 0.0 for delta in ap_deltas.values())
    worst_sensor = min(ap_deltas, key=ap_deltas.get)
    structural_pass = all(check["pass"] for check in checks.values())
    active_gate_spec = gate_spec(comparison_profile)
    criteria = {
        "artifact_protocol_and_transfer_integrity": {
            "pass": structural_pass,
            "observed": {
                "passed_checks": sum(check["pass"] for check in checks.values()),
                "total_checks": len(checks),
            },
            "expected": "all checks pass",
        },
        "macro_ap_delta_at_least_0p010": {
            "pass": (
                macro_delta["ap"]
                >= active_gate_spec["macro_ap_min_delta"]
            ),
            "observed": macro_delta["ap"],
            "expected": f">= {active_gate_spec['macro_ap_min_delta']}",
        },
        "at_least_three_sensor_ap_deltas_nondecreasing": {
            "pass": nondecreasing_count
            >= active_gate_spec["minimum_nondecreasing_sensor_ap_count"],
            "observed": {
                "count": nondecreasing_count,
                "deltas": ap_deltas,
            },
            "expected": (
                f">= "
                f"{active_gate_spec['minimum_nondecreasing_sensor_ap_count']} "
                "sensors"
            ),
        },
        "worst_sensor_ap_delta_at_least_minus_0p010": {
            "pass": ap_deltas[worst_sensor]
            >= active_gate_spec["worst_sensor_ap_min_delta"],
            "observed": {
                "sensor": worst_sensor,
                "delta": ap_deltas[worst_sensor],
            },
            "expected": (
                f">= {active_gate_spec['worst_sensor_ap_min_delta']}"
            ),
        },
        "macro_best_f1_delta_at_least_minus_0p005": {
            "pass": macro_delta["best_f1"]
            >= active_gate_spec["macro_best_f1_min_delta"],
            "observed": macro_delta["best_f1"],
            "expected": (
                f">= {active_gate_spec['macro_best_f1_min_delta']}"
            ),
        },
    }
    failed_criteria = [
        name for name, criterion in criteria.items() if not criterion["pass"]
    ]
    passed = not failed_criteria
    return {
        "schema_version": 1,
        "artifact_type": artifact_type(comparison_profile),
        "comparison_profile": comparison_profile,
        "created_utc": utc_now(),
        "status": "pass" if passed else "fail",
        "preregistered_gate": active_gate_spec,
        "inputs": inputs,
        "checks": checks,
        "reconstruction": reconstruction,
        "metrics": metrics,
        "gate_decision": {
            "pass": passed,
            "criteria": criteria,
            "failed_criteria": failed_criteria,
            "interpretation": (
                "Promote only to multi-seed 2-3 epoch confirmation."
                if passed
                else "Do not promote this one-epoch MAE implementation."
            ),
        },
    }


def synthetic_classification_epoch(
    values: Mapping[str, Mapping[str, float]]
) -> Dict[str, Any]:
    macro = {
        history_name: sum(
            values[sensor][output_name] for sensor in EXPECTED_SENSORS
        )
        / len(EXPECTED_SENSORS)
        for output_name, history_name in CLASSIFICATION_METRICS.items()
    }
    return {
        "epoch": 0,
        "train_loss": {sensor: 0.5 for sensor in EXPECTED_SENSORS},
        "val": {
            sensor: {
                "samples": 16,
                "positive_rate": 0.5,
                "ap": values[sensor]["ap"],
                "auroc": values[sensor]["auroc"],
                "f1": values[sensor]["best_f1"],
                "f1_0p5": values[sensor]["f1_at_0p5"],
                "loss": 0.6,
            }
            for sensor in EXPECTED_SENSORS
        },
        "macro_over_sensor": macro,
    }


def run_self_test() -> None:
    data_signature = {
        "schema_version": 2,
        "global_train_val_overlap_after": 0,
        "manifest_sha256": {"synthetic": "no-real-data"},
        "representation": {"streams": ["synthetic_current"]},
    }
    encoder_signature = {
        "schema_version": 1,
        "synthetic_encoder": "v1",
    }
    event_protocol = {"global_train_val_overlap_after": 0}
    common = {
        "schema_version": 2,
        "status": "completed",
        "sharing": "shared",
        "sensors": list(EXPECTED_SENSORS),
        "event_protocol": event_protocol,
        "data_signature": data_signature,
        "encoder_signature": encoder_signature,
        "balanced_rounds_per_epoch": 2,
        "optimizer_steps_per_epoch": 8,
    }
    pretrain = {
        **common,
        "mode": "pretrain",
        "model_parameters": 120,
        "trainable_parameters": 120,
        "resume_signature": {"mode": "pretrain"},
        "epochs": [
            {
                "epoch": 0,
                "train_loss": {
                    sensor: 0.4 + index * 0.01
                    for index, sensor in enumerate(EXPECTED_SENSORS)
                },
                "val": {
                    sensor: {
                        "samples": 8,
                        "reconstruction_loss": 0.5 + index * 0.01,
                    }
                    for index, sensor in enumerate(EXPECTED_SENSORS)
                },
                "macro_over_sensor": {"reconstruction_loss": 0.515},
            }
        ],
    }
    scratch_values = {
        sensor: {
            "ap": 0.50 + index * 0.02,
            "auroc": 0.60 + index * 0.02,
            "best_f1": 0.55 + index * 0.01,
            "f1_at_0p5": 0.50 + index * 0.01,
        }
        for index, sensor in enumerate(EXPECTED_SENSORS)
    }
    finetune_values = {
        sensor: {
            metric: value + (0.02 if metric != "f1_at_0p5" else 0.01)
            for metric, value in scratch_values[sensor].items()
        }
        for sensor in EXPECTED_SENSORS
    }
    supervised_common = {
        **common,
        "mode": "supervised",
        "model_parameters": 100,
        "trainable_parameters": 100,
        "resume_signature": {"matched_supervised": "v1"},
        "initial_head_sha256": "synthetic-matched-head-sha",
    }
    scratch = {
        **supervised_common,
        "epochs": [synthetic_classification_epoch(scratch_values)],
    }

    with tempfile.TemporaryDirectory(prefix="mae_gate_self_test.") as directory:
        root = Path(directory)
        checkpoint_path = root / "pretrain.pth"
        torch.save(
            {
                "epoch": 0,
                "mode": "pretrain",
                "data_signature": data_signature,
                "encoder_signature": encoder_signature,
                "model": {},
            },
            checkpoint_path,
        )
        checkpoint_sha = sha256_file(checkpoint_path)
        finetune = {
            **supervised_common,
            "init_checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "loaded_encoder_keys": 1,
                "loaded_encoder_numel": 10,
                "target_encoder_numel": 10,
                "encoder_coverage": 1.0,
                "excluded_decoder_keys": ["backbone.decoder_pred.weight"],
                "excluded_checkpoint_head_keys": [],
                "target_head_keys": ["heads.s2.weight"],
                "target_head_sha256_before": "synthetic-matched-head-sha",
                "target_head_sha256_after": "synthetic-matched-head-sha",
            },
            "epochs": [synthetic_classification_epoch(finetune_values)],
        }
        history_paths = {
            "pretrain": root / "pretrain.json",
            "scratch": root / "scratch.json",
            "finetune": root / "finetune.json",
        }
        for role, payload in (
            ("pretrain", pretrain),
            ("scratch", scratch),
            ("finetune", finetune),
        ):
            atomic_json_dump(payload, history_paths[role])

        passed = compare_artifacts(
            pretrain_history_path=history_paths["pretrain"],
            pretrain_checkpoint_path=checkpoint_path,
            scratch_history_path=history_paths["scratch"],
            finetune_history_path=history_paths["finetune"],
        )
        if passed.get("status") != "pass":
            raise AssertionError(f"Synthetic passing gate failed: {passed}")

        failed_finetune = copy.deepcopy(finetune)
        failed_finetune["initial_head_sha256"] = "different-head"
        failed_path = root / "finetune_fail.json"
        atomic_json_dump(failed_finetune, failed_path)
        failed = compare_artifacts(
            pretrain_history_path=history_paths["pretrain"],
            pretrain_checkpoint_path=checkpoint_path,
            scratch_history_path=history_paths["scratch"],
            finetune_history_path=failed_path,
        )
        if failed.get("status") != "fail":
            raise AssertionError("Synthetic head mismatch was not rejected.")

        pending = compare_artifacts(
            pretrain_history_path=history_paths["pretrain"],
            pretrain_checkpoint_path=checkpoint_path,
            scratch_history_path=history_paths["scratch"],
            finetune_history_path=root / "not_created_yet.json",
        )
        if pending.get("status") != "pending":
            raise AssertionError("Missing finetune history did not report pending.")

        adapter_sha = "a" * 64
        validity_contract = {
            "enabled": True,
            "objective_version": VALIDITY_OBJECTIVE_VERSION,
            "semantics": VALIDITY_SEMANTICS,
            "adapter_source_sha256": adapter_sha,
            "native_validity": EXPECTED_VALIDITY_NATIVE,
            "stream_validity": EXPECTED_VALIDITY_STREAMS,
            "resize": EXPECTED_VALIDITY_RESIZE,
            "loss_normalization": EXPECTED_VALIDITY_NORMALIZATION,
        }
        masked_data_signature = copy.deepcopy(data_signature)
        masked_data_signature["schema_version"] = 3
        masked_data_signature["representation"][
            "validity_reconstruction"
        ] = validity_contract
        masked_encoder_signature = {
            **encoder_signature,
            "schema_version": 2,
            "validity_adapter": {
                "class": "ValidityMaskedMethaneResidualMAE",
                "source_sha256": adapter_sha,
                "objective_version": VALIDITY_OBJECTIVE_VERSION,
            },
        }
        optimization = {
            "batch_size": 4,
            "learning_rate": 3e-4,
            "weight_decay": 0.05,
            "grad_clip": 1.0,
            "balanced_pos_weight": True,
            "amp": True,
            "augment": True,
            "seed": 20260727,
            "balanced_rounds": 2,
            "optimizer_steps_per_epoch": 8,
            "loader_lengths": {
                sensor: 2 for sensor in EXPECTED_SENSORS
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
        masked_finetune_resume = copy.deepcopy(scratch_resume)
        masked_finetune_resume[
            "encoder_signature"
        ] = masked_encoder_signature
        masked_finetune_resume[
            "data_signature"
        ] = masked_data_signature
        masked_finetune_resume[
            "validity_reconstruction"
        ] = validity_contract
        masked_finetune_resume["schema_version"] = 2
        masked_pretrain_resume = copy.deepcopy(
            masked_finetune_resume
        )
        masked_pretrain_resume["mode"] = "pretrain"
        masked_pretrain_resume["evaluation"] = {"max_val_batches": 30}
        masked_pretrain_resume["pretrain_model"] = {
            "mask_ratio": 0.6,
            "reconstruction_objective": VALIDITY_OBJECTIVE_VERSION,
        }

        masked_pretrain = copy.deepcopy(pretrain)
        masked_pretrain["data_signature"] = masked_data_signature
        masked_pretrain["encoder_signature"] = masked_encoder_signature
        masked_pretrain["resume_signature"] = masked_pretrain_resume
        masked_pretrain["schema_version"] = 3
        masked_pretrain["validity_reconstruction"] = validity_contract
        masked_pretrain["config"] = {
            "validity_masked_reconstruction": True,
            "epochs": 1,
            "seed": 20260727,
            "init_checkpoint": None,
            "resume": None,
        }
        for sensor in EXPECTED_SENSORS:
            diagnostics = {
                "valid_streams": 3.0,
                "empty_streams": 0.0,
                "total_streams": 3.0,
            }
            for suffix in ("current", "recent", "seasonal"):
                stream = f"{sensor}_{suffix}"
                diagnostics.update(
                    {
                        f"loss_{stream}": 0.5,
                        f"valid_fraction_{stream}": 0.75,
                        f"valid_masked_elements_{stream}": 100.0,
                        f"valid_samples_{stream}": 8.0,
                        f"excluded_samples_{stream}": 0.0,
                        f"batch_samples_{stream}": 8.0,
                        f"empty_stream_{stream}": 0.0,
                    }
                )
            masked_pretrain["epochs"][0]["val"][sensor][
                "validity_diagnostics"
            ] = diagnostics

        masked_scratch = copy.deepcopy(scratch)
        masked_scratch["resume_signature"] = scratch_resume
        masked_scratch["config"] = {
            "validity_masked_reconstruction": False,
            "epochs": 1,
            "seed": 20260727,
            "init_checkpoint": None,
            "resume": None,
        }
        masked_checkpoint_path = root / "masked_pretrain.pth"
        torch.save(
            {
                "epoch": 0,
                "mode": "pretrain",
                "data_signature": masked_data_signature,
                "encoder_signature": masked_encoder_signature,
                "resume_signature": masked_pretrain_resume,
                "config": {
                    "validity_masked_reconstruction": True,
                },
                "validity_reconstruction": validity_contract,
                "model": {},
            },
            masked_checkpoint_path,
        )
        masked_checkpoint_sha = sha256_file(masked_checkpoint_path)
        masked_finetune = copy.deepcopy(finetune)
        masked_finetune["data_signature"] = masked_data_signature
        masked_finetune["encoder_signature"] = masked_encoder_signature
        masked_finetune["resume_signature"] = masked_finetune_resume
        masked_finetune["schema_version"] = 3
        masked_finetune["validity_reconstruction"] = validity_contract
        masked_finetune["config"] = {
            "validity_masked_reconstruction": True,
            "epochs": 1,
            "seed": 20260727,
            "init_checkpoint": str(masked_checkpoint_path.resolve()),
            "resume": None,
        }
        masked_finetune["init_checkpoint"][
            "path"
        ] = str(masked_checkpoint_path.resolve())
        masked_finetune["init_checkpoint"][
            "checkpoint_sha256"
        ] = masked_checkpoint_sha

        masked_paths = {
            "pretrain": root / "masked_pretrain.json",
            "scratch": root / "masked_scratch.json",
            "finetune": root / "masked_finetune.json",
        }
        for role, payload in (
            ("pretrain", masked_pretrain),
            ("scratch", masked_scratch),
            ("finetune", masked_finetune),
        ):
            atomic_json_dump(payload, masked_paths[role])
        masked_pass = compare_artifacts(
            pretrain_history_path=masked_paths["pretrain"],
            pretrain_checkpoint_path=masked_checkpoint_path,
            scratch_history_path=masked_paths["scratch"],
            finetune_history_path=masked_paths["finetune"],
            comparison_profile=VALIDITY_MASKED_PROFILE,
        )
        if masked_pass.get("status") != "pass":
            raise AssertionError(
                f"Synthetic masked passing gate failed: {masked_pass}"
            )

        wrong_objective_finetune = copy.deepcopy(masked_finetune)
        wrong_objective_finetune["data_signature"]["representation"][
            "validity_reconstruction"
        ]["objective_version"] = "wrong-objective"
        wrong_objective_path = root / "wrong_objective_finetune.json"
        atomic_json_dump(wrong_objective_finetune, wrong_objective_path)
        wrong_objective = compare_artifacts(
            pretrain_history_path=masked_paths["pretrain"],
            pretrain_checkpoint_path=masked_checkpoint_path,
            scratch_history_path=masked_paths["scratch"],
            finetune_history_path=wrong_objective_path,
            comparison_profile=VALIDITY_MASKED_PROFILE,
        )
        if wrong_objective.get("status") != "fail":
            raise AssertionError(
                "Masked pretrain/finetune objective mismatch was not rejected."
            )

        legacy_as_masked = compare_artifacts(
            pretrain_history_path=history_paths["pretrain"],
            pretrain_checkpoint_path=checkpoint_path,
            scratch_history_path=history_paths["scratch"],
            finetune_history_path=history_paths["finetune"],
            comparison_profile=VALIDITY_MASKED_PROFILE,
        )
        if legacy_as_masked.get("status") != "fail":
            raise AssertionError(
                "Legacy unmasked artifacts passed the masked profile."
            )

    print(
        json.dumps(
            {
                "self_test": "passed",
                "device": "cpu",
                "real_manifests_or_tests_read": 0,
                "passing_case": passed["status"],
                "head_mismatch_case": failed["status"],
                "missing_artifact_case": pending["status"],
                "validity_masked_passing_case": masked_pass["status"],
                "masked_objective_mismatch_case": wrong_objective["status"],
                "legacy_as_masked_case": legacy_as_masked["status"],
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
        "pretrain_history": args.pretrain_history,
        "pretrain_checkpoint": args.pretrain_checkpoint,
        "scratch_history": args.scratch_history,
        "finetune_history": args.finetune_history,
    }
    omitted = [role for role, path in required.items() if path is None]
    if omitted:
        raise SystemExit(
            "Missing required arguments outside --self-test: " + ", ".join(omitted)
        )
    result = compare_artifacts(
        pretrain_history_path=args.pretrain_history,
        pretrain_checkpoint_path=args.pretrain_checkpoint,
        scratch_history_path=args.scratch_history,
        finetune_history_path=args.finetune_history,
        comparison_profile=args.comparison_profile,
    )
    if args.output_json is not None:
        output_resolved = args.output_json.resolve()
        input_paths = {
            Path(path).resolve()
            for path in required.values()
            if path is not None
        }
        if output_resolved in input_paths:
            raise ValueError(
                "--output-json must not overwrite a gate input artifact."
            )
        if (
            args.comparison_profile == VALIDITY_MASKED_PROFILE
            and args.output_json.is_file()
        ):
            try:
                existing_artifact_type = load_json(
                    args.output_json
                ).get("artifact_type")
            except Exception as error:
                raise ValueError(
                    "Refusing to overwrite an existing non-readable output "
                    f"with the validity-masked gate: {args.output_json}"
                ) from error
            if existing_artifact_type != artifact_type(
                VALIDITY_MASKED_PROFILE
            ):
                raise ValueError(
                    "Refusing to overwrite an existing non-masked gate "
                    f"artifact: {args.output_json}"
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
