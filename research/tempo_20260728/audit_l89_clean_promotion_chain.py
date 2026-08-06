#!/usr/bin/env python3
"""CPU-only formal promotion-chain audit for the clean L89 replicate.

The audit has exactly one preregistered branch:

* qualify the fresh P5 response expert, then test fixed P5 + mean-seed D1
  against P5; or
* when P5 is not qualified, skip promotion and automatically audit the fixed
  P0 + mean-seed D1 mechanism-only fallback.

No checkpoint is executed.  Sidecar checkpoints are opened on CPU only to
verify their capability AP and matched fresh initialization.  All prediction
arithmetic is fixed in logit space, all thresholds are point-selected once,
and all uncertainty uses 5,000 canonical-event-cluster replicates with seed
2026072808.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.pretraining_20260727 import l89_ragged_cls_experiment as cache
from research.pretraining_20260727 import (
    rctp_l89_event_balanced_head_followup as head_runner,
)
from research.tempo_20260728 import audit_l89_clean_fixed_ensemble as fixed
from research.tempo_20260728 import tempo_l89_global as tempo


FROZEN_SEEDS = (20260727, 20260728, 20260729)
FROZEN_BOOTSTRAP_REPLICATES = 5000
FROZEN_BOOTSTRAP_SEED = 2026072808
FROZEN_REFERENCE_WEIGHT = 0.5
FROZEN_MEAN_D1_WEIGHT = 0.5
SIDECAR_ARTIFACT_ARMS = {
    "p4": "p4_response_scrambled",
    "p5": "p5_correct_response",
}
FORBIDDEN_COMPONENT = re.compile(
    r"(^|[._-])(test|sealed|holdout|outer)([._-]|$)", re.IGNORECASE
)
GATE_METRICS = (
    "event_balanced_ap",
    "event_balanced_auc",
    "event_balanced_positive_f1_selected",
    "event_balanced_macro_f1_selected",
    "all_negative_fp_mass",
)
HEAD_ALIASES = {
    "event_balanced_ap": ("event_balanced_ap",),
    "event_balanced_auc": ("event_balanced_auc",),
    "event_balanced_macro_f1_selected": (
        "event_balanced_macro_f1_selected",
        "event_balanced_selected_macro_f1",
    ),
    "event_balanced_positive_f1_selected": (
        "event_balanced_positive_f1_selected",
        "event_balanced_selected_positive_f1",
    ),
    "selected_threshold": (
        "selected_threshold",
        "event_balanced_selected_threshold",
    ),
}


def assert_inner_path(path: Path, *, purpose: str) -> Path:
    resolved = path.expanduser().resolve()
    tempo.assert_development_path(resolved, purpose=purpose)
    offending = [
        component
        for component in resolved.parts
        if FORBIDDEN_COMPONENT.search(component)
    ]
    if offending:
        raise ValueError(
            f"{purpose} contains held-out/outer-like path components: "
            f"{resolved}; offending={offending}"
        )
    return resolved


def read_json(path: Path, *, purpose: str) -> Mapping[str, Any]:
    resolved = assert_inner_path(path, purpose=purpose)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{purpose} must contain a JSON object.")
    return payload


def finite_unit(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0,1], got {result}.")
    return result


def finite_value(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}.")
    return result


def get_alias(
    payload: Mapping[str, Any], canonical: str, *, source: str
) -> float:
    aliases = HEAD_ALIASES.get(canonical, (canonical,))
    present = [name for name in aliases if name in payload]
    if not present:
        raise KeyError(
            f"{source} must contain an alias for {canonical}: "
            f"{aliases}; present={present}."
        )
    values = [
        finite_value(payload[name], name=f"{source}.{name}")
        for name in present
    ]
    if any(
        not math.isclose(value, values[0], rel_tol=0.0, abs_tol=1e-12)
        for value in values[1:]
    ):
        raise ValueError(
            f"{source} contains disagreeing aliases for {canonical}: "
            f"{dict(zip(present, values))}."
        )
    return values[0]


def values_close(left: float, right: float, *, tolerance: float = 1e-8) -> bool:
    return bool(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance))


def require_false_flag(payload: Mapping[str, Any], key: str, *, source: str) -> None:
    if key in payload and payload[key] is not False:
        raise ValueError(f"{source}.{key} must be exactly false.")


def audit_embedded_absolute_paths(value: Any, *, source: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            audit_embedded_absolute_paths(child, source=f"{source}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            audit_embedded_absolute_paths(child, source=f"{source}[{index}]")
    elif isinstance(value, str) and value.startswith("/"):
        assert_inner_path(Path(value), purpose=f"embedded path {source}")


def extract_sidecar_capability(
    *,
    arm: str,
    summary_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if arm not in SIDECAR_ARTIFACT_ARMS:
        raise ValueError(f"Unsupported logical sidecar arm {arm!r}.")
    artifact_arm = SIDECAR_ARTIFACT_ARMS[arm]
    summary_path = assert_inner_path(
        summary_path, purpose=f"{arm} sidecar summary"
    )
    checkpoint_path = assert_inner_path(
        checkpoint_path, purpose=f"{arm} sidecar checkpoint"
    )
    summary = read_json(summary_path, purpose=f"{arm} sidecar summary")
    audit_embedded_absolute_paths(summary, source=f"{arm}_summary")
    if str(summary.get("arm", "")).lower() != artifact_arm:
        raise ValueError(f"{arm} sidecar summary has the wrong arm.")
    require_false_flag(
        summary,
        "test_or_sealed_or_holdout_read",
        source=f"{arm}_summary",
    )
    expected_sha = str(summary.get("sidecar_checkpoint_sha256", ""))
    observed_sha = cache.sha256_file(checkpoint_path)
    if not expected_sha or expected_sha != observed_sha:
        raise ValueError(f"{arm} summary/checkpoint SHA mismatch.")
    recorded_checkpoint = summary.get("sidecar_checkpoint")
    if recorded_checkpoint is not None:
        recorded = assert_inner_path(
            Path(str(recorded_checkpoint)),
            purpose=f"{arm} recorded sidecar checkpoint",
        )
        if recorded != checkpoint_path:
            raise ValueError(f"{arm} summary points to a different checkpoint.")

    checkpoint = cache.torch_load_trusted(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{arm} sidecar checkpoint must be a mapping.")
    if str(checkpoint.get("arm", "")).lower() != artifact_arm:
        raise ValueError(f"{arm} sidecar checkpoint has the wrong arm.")
    require_false_flag(
        checkpoint,
        "test_or_sealed_or_holdout_read",
        source=f"{arm}_checkpoint",
    )
    summary_metrics = summary.get("best_dev_metrics")
    checkpoint_metrics = checkpoint.get("best_dev_metrics")
    if not isinstance(summary_metrics, Mapping) or not isinstance(
        checkpoint_metrics, Mapping
    ):
        raise KeyError(
            f"{arm} sidecar summary/checkpoint lack best capability metrics."
        )
    summary_ap = finite_unit(
        summary_metrics["average_precision"],
        name=f"{arm}.summary.best_dev_metrics.average_precision",
    )
    checkpoint_ap = finite_unit(
        checkpoint_metrics["average_precision"],
        name=f"{arm}.checkpoint.best_dev_metrics.average_precision",
    )
    if not values_close(summary_ap, checkpoint_ap, tolerance=1e-12):
        raise ValueError(f"{arm} summary/checkpoint capability AP mismatch.")
    summary_initial = str(summary.get("combined_initial_state_sha256", ""))
    checkpoint_initial = str(
        checkpoint.get("combined_initial_state_sha256", "")
    )
    if not summary_initial or summary_initial != checkpoint_initial:
        raise ValueError(f"{arm} fresh initial-state receipt mismatch.")
    selected_epoch = int(summary.get("best_epoch", -1))
    if selected_epoch not in (1, 2) or int(checkpoint.get("epoch", -1)) != selected_epoch:
        raise ValueError(f"{arm} selected sidecar epoch is not a valid matched epoch.")
    return {
        "arm": arm,
        "artifact_arm": artifact_arm,
        "synthetic_capability_ap": summary_ap,
        "selected_epoch": selected_epoch,
        "fresh_combined_initial_state_sha256": summary_initial,
        "summary": {
            "path": str(summary_path),
            "sha256": cache.sha256_file(summary_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": observed_sha,
        },
        "selection_panel": "fixed inner-train-only synthetic capability panel",
    }


def validate_sidecar_pair(
    p4: Mapping[str, Any], p5: Mapping[str, Any]
) -> None:
    if (
        p4["fresh_combined_initial_state_sha256"]
        != p5["fresh_combined_initial_state_sha256"]
    ):
        raise ValueError("P4/P5 fresh sidecar initial states differ.")


def validate_head_comparison(
    path: Path,
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    payload = read_json(path, purpose="matched head comparison")
    audit_embedded_absolute_paths(payload, source="head_comparison")
    if (
        payload.get("script_version") != "l89-clean-inner-matched-head-v1"
        or payload.get("clean_inner_replicate_exploratory") is not True
        or payload.get("post_hoc_exploratory") is not False
    ):
        raise ValueError(
            "Head comparison is not a fresh clean-inner matched-head artifact."
        )
    require_false_flag(payload, "test_or_sealed_read", source="head_comparison")
    require_false_flag(
        payload,
        "test_or_sealed_or_holdout_read",
        source="head_comparison",
    )
    matching = payload.get("matching")
    if not isinstance(matching, Mapping):
        raise KeyError("Matched head comparison lacks matching receipts.")
    required_matching = (
        "identity_and_labels_equal",
        "same_initial_state",
        "same_parameter_signature",
        "same_batch_plan_sha256",
        "same_seed_optimizer_epochs",
    )
    failed = [key for key in required_matching if matching.get(key) is not True]
    if failed:
        raise ValueError(f"Matched P0/P4/P5 contract failed: {failed}.")
    point = payload.get("point_metrics")
    if not isinstance(point, Mapping) or set(point) != {"p0", "p4", "p5"}:
        raise ValueError("Head comparison must contain exactly P0/P4/P5 metrics.")
    normalized: dict[str, Mapping[str, Any]] = {}
    for arm in ("p0", "p4", "p5"):
        metrics = point[arm]
        if not isinstance(metrics, Mapping):
            raise TypeError(f"Head metrics for {arm} must be a mapping.")
        normalized[arm] = metrics
        for metric in (
            "event_balanced_ap",
            "event_balanced_auc",
            "event_balanced_macro_f1_selected",
        ):
            value = get_alias(metrics, metric, source=f"head.{arm}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"head.{arm}.{metric} is outside [0,1].")
    return payload, normalized


def fixed_reference_mean_d1(
    reference_probability: np.ndarray,
    d1_probabilities: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if len(d1_probabilities) != len(FROZEN_SEEDS):
        raise ValueError("Exactly three preregistered D1 predictions are required.")
    reference_logit = fixed.logit(reference_probability)
    mean_d1_logit = np.stack(
        [fixed.logit(value) for value in d1_probabilities], axis=0
    ).mean(axis=0)
    candidate_logit = (
        FROZEN_REFERENCE_WEIGHT * reference_logit
        + FROZEN_MEAN_D1_WEIGHT * mean_d1_logit
    )
    return fixed.sigmoid(mean_d1_logit), fixed.sigmoid(candidate_logit)


def condition(
    *,
    observed: float,
    operator: str,
    boundary: float,
    description: str,
) -> dict[str, Any]:
    if operator == ">":
        passed = observed > boundary
    elif operator == ">=":
        passed = observed >= boundary - 1e-12
    elif operator == "<=":
        passed = observed <= boundary + 1e-12
    else:
        raise ValueError(f"Unsupported gate operator {operator!r}.")
    return {
        "description": description,
        "observed": float(observed),
        "operator": operator,
        "boundary": float(boundary),
        "pass": bool(passed),
    }


def evaluate_p5_qualification(
    *,
    p4_synthetic_ap: float,
    p5_synthetic_ap: float,
    formal_gate_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    p0_ap = get_alias(
        formal_gate_metrics["p0"], "event_balanced_ap", source="formal.p0"
    )
    p4_ap = get_alias(
        formal_gate_metrics["p4"], "event_balanced_ap", source="formal.p4"
    )
    p5_ap = get_alias(
        formal_gate_metrics["p5"], "event_balanced_ap", source="formal.p5"
    )
    p0_auc = get_alias(
        formal_gate_metrics["p0"], "event_balanced_auc", source="formal.p0"
    )
    p4_auc = get_alias(
        formal_gate_metrics["p4"], "event_balanced_auc", source="formal.p4"
    )
    p5_auc = get_alias(
        formal_gate_metrics["p5"], "event_balanced_auc", source="formal.p5"
    )
    p0_macro = get_alias(
        formal_gate_metrics["p0"],
        "event_balanced_macro_f1_selected",
        source="formal.p0",
    )
    p4_macro = get_alias(
        formal_gate_metrics["p4"],
        "event_balanced_macro_f1_selected",
        source="formal.p4",
    )
    p5_macro = get_alias(
        formal_gate_metrics["p5"],
        "event_balanced_macro_f1_selected",
        source="formal.p5",
    )
    p0_fp_mass = get_alias(
        formal_gate_metrics["p0"],
        "all_negative_fp_mass",
        source="formal.p0",
    )
    p5_fp_mass = get_alias(
        formal_gate_metrics["p5"],
        "all_negative_fp_mass",
        source="formal.p5",
    )
    conditions = {
        "synthetic_ap_p5_gt_p4": condition(
            observed=p5_synthetic_ap - p4_synthetic_ap,
            operator=">",
            boundary=0.0,
            description="synthetic_AP(P5) - synthetic_AP(P4)",
        ),
        "real_ap_p5_gt_p0_p4": condition(
            observed=p5_ap - max(p0_ap, p4_ap),
            operator=">",
            boundary=0.0,
            description="real_AP(P5) - max(real_AP(P0), real_AP(P4))",
        ),
        "real_auc_p5_gt_p0_p4": condition(
            observed=p5_auc - max(p0_auc, p4_auc),
            operator=">",
            boundary=0.0,
            description="real_AUC(P5) - max(real_AUC(P0), real_AUC(P4))",
        ),
        "real_macro_p5_tolerance": condition(
            observed=p5_macro - max(p0_macro, p4_macro),
            operator=">=",
            boundary=-0.003,
            description="real_macro(P5) - max(real_macro(P0), real_macro(P4))",
        ),
        "p5_fp_mass_tolerance": condition(
            observed=float(p5_fp_mass) - float(p0_fp_mass),
            operator="<=",
            boundary=0.5,
            description="FPmass(P5) - FPmass(P0)",
        ),
    }
    return {
        "P5_qualified": all(item["pass"] for item in conditions.values()),
        "all_conditions_required": True,
        "conditions": conditions,
        "source_values": {
            "synthetic_ap": {
                "p4": float(p4_synthetic_ap),
                "p5": float(p5_synthetic_ap),
            },
            "real_event_balanced": {
                "p0": {"ap": p0_ap, "auc": p0_auc, "macro_f1": p0_macro},
                "p4": {"ap": p4_ap, "auc": p4_auc, "macro_f1": p4_macro},
                "p5": {
                    "ap": p5_ap,
                    "auc": p5_auc,
                    "macro_f1": p5_macro,
                    "all_negative_fp_mass": float(p5_fp_mass),
                },
                "p0_all_negative_fp_mass": float(p0_fp_mass),
            },
        },
    }


def evaluate_promotion_gate(
    *,
    point_metrics: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = point_metrics["fixed_p5_mean_d1"]
    reference = point_metrics["p5"]
    comparison = bootstrap["candidate_minus_p5"]
    ap_delta = float(candidate["event_balanced_ap"]) - float(
        reference["event_balanced_ap"]
    )
    auc_delta = float(candidate["event_balanced_auc"]) - float(
        reference["event_balanced_auc"]
    )
    macro_delta = float(
        candidate["event_balanced_macro_f1_selected"]
    ) - float(reference["event_balanced_macro_f1_selected"])
    fp_delta = float(candidate["all_negative_fp_mass"]) - float(
        reference["all_negative_fp_mass"]
    )
    ap_ci_low = finite_value(
        comparison["event_balanced_ap"]["ci_95_low"],
        name="candidate_minus_p5.AP.ci_95_low",
    )
    conditions = {
        "ap_point_delta_positive": condition(
            observed=ap_delta,
            operator=">",
            boundary=0.0,
            description="AP(P5+D1) - AP(P5)",
        ),
        "auc_point_delta_positive": condition(
            observed=auc_delta,
            operator=">",
            boundary=0.0,
            description="AUC(P5+D1) - AUC(P5)",
        ),
        "ap_cluster_bootstrap_ci95_low_positive": condition(
            observed=ap_ci_low,
            operator=">",
            boundary=0.0,
            description="AP paired event-cluster bootstrap 95% lower bound",
        ),
        "macro_f1_tolerance": condition(
            observed=macro_delta,
            operator=">=",
            boundary=-0.003,
            description="macro-F1(P5+D1) - macro-F1(P5)",
        ),
        "fp_mass_tolerance": condition(
            observed=fp_delta,
            operator="<=",
            boundary=0.5,
            description="FPmass(P5+D1) - FPmass(P5)",
        ),
    }
    return {
        "reference": "new_p5",
        "candidate": "fixed_0.5_p5_plus_0.5_d1_three_seed_logit_ensemble",
        "promotion": all(item["pass"] for item in conditions.values()),
        "all_conditions_required": True,
        "conditions": conditions,
    }


def compare_metric_bundle(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    source: str,
    keys: Sequence[str] = (*GATE_METRICS, "selected_threshold"),
    tolerance: float = 1e-8,
) -> None:
    for key in keys:
        left = get_alias(observed, key, source=source)
        right = get_alias(expected, key, source=f"{source}.recomputed")
        if not values_close(left, right, tolerance=tolerance):
            raise ValueError(
                f"{source}.{key} differs from prediction recomputation: "
                f"{left} != {right}."
            )


def compare_bootstrap(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    source: str,
) -> None:
    if set(observed) != set(expected):
        raise ValueError(f"{source} comparison keys differ.")
    for comparison in expected:
        if set(observed[comparison]) != set(expected[comparison]):
            raise ValueError(f"{source}.{comparison} metric keys differ.")
        for metric in expected[comparison]:
            for key in (
                "point",
                "ci_95_low",
                "ci_95_high",
                "win_probability",
            ):
                left = finite_value(
                    observed[comparison][metric][key],
                    name=f"{source}.{comparison}.{metric}.{key}",
                )
                right = finite_value(
                    expected[comparison][metric][key],
                    name=f"recomputed.{comparison}.{metric}.{key}",
                )
                if not values_close(left, right, tolerance=1e-12):
                    raise ValueError(
                        f"{source}.{comparison}.{metric}.{key} is not "
                        "the deterministic frozen-bootstrap value."
                    )
            if (
                observed[comparison][metric].get("better_direction")
                != expected[comparison][metric].get("better_direction")
            ):
                raise ValueError(
                    f"{source}.{comparison}.{metric} direction differs."
                )


def strictly_aligned_prediction_frames(
    frames: Mapping[str, pd.DataFrame],
) -> None:
    if not frames:
        raise ValueError("No prediction frames supplied.")
    required = ("id", "plume_id", "event_id", "label")
    for name, frame in frames.items():
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(
                f"{name} prediction table lacks strict identity columns {missing}."
            )
        if frame["id"].astype(str).duplicated().any():
            raise ValueError(f"{name} prediction IDs are not unique.")
    reference_name = next(iter(frames))
    reference = frames[reference_name]
    for name, candidate in frames.items():
        if name == reference_name:
            continue
        for column in ("id", "plume_id", "event_id"):
            if (
                reference[column].astype(str).tolist()
                != candidate[column].astype(str).tolist()
            ):
                raise ValueError(
                    f"{name} differs from {reference_name} in ordered {column}."
                )
        if not np.array_equal(
            reference["label"].to_numpy(dtype=np.int64),
            candidate["label"].to_numpy(dtype=np.int64),
        ):
            raise ValueError(
                f"{name} differs from {reference_name} in ordered label."
            )


def validate_p0_probability_replay(
    *,
    reference_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    reference_path: Path,
    candidate_path: Path,
    source: str,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Independently close one P0 replay CSV to the fresh matched-head P0."""

    strictly_aligned_prediction_frames(
        {"fresh_p0_head": reference_frame, source: candidate_frame}
    )
    reference_probability = reference_frame["probability"].to_numpy(
        dtype=np.float64
    )
    candidate_probability = candidate_frame["probability"].to_numpy(
        dtype=np.float64
    )
    if len(reference_probability) == 0:
        raise ValueError(f"{source} P0 replay has no rows.")
    maximum_error = float(
        np.max(np.abs(candidate_probability - reference_probability))
    )
    if maximum_error > float(tolerance):
        raise ValueError(
            f"{source} P0 replay probability differs from fresh P0 head: "
            f"maximum_absolute_probability_error={maximum_error:.9g}, "
            f"tolerance={float(tolerance):.9g}."
        )
    return {
        "source": source,
        "fresh_p0_head_path": str(reference_path),
        "fresh_p0_head_sha256": cache.sha256_file(reference_path),
        "replay_path": str(candidate_path),
        "replay_sha256": cache.sha256_file(candidate_path),
        "strict_ordered_identity_and_label_match": True,
        "rows": int(len(reference_probability)),
        "maximum_absolute_probability_error": maximum_error,
        "probability_tolerance": float(tolerance),
        "pass": True,
    }


def load_prediction_inputs(
    *,
    p0_path: Path,
    p0_head_path: Path,
    p4_path: Path,
    p5_path: Path,
    d1_template: str,
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    paths = {
        "p0": assert_inner_path(p0_path, purpose="P0 prediction input"),
        "p0_head": assert_inner_path(
            p0_head_path, purpose="matched-head P0 prediction input"
        ),
        "p4": assert_inner_path(p4_path, purpose="P4 prediction input"),
        "p5": assert_inner_path(p5_path, purpose="P5 prediction input"),
    }
    for seed in FROZEN_SEEDS:
        paths[f"d1_seed_{seed}"] = assert_inner_path(
            Path(str(d1_template).format(seed=seed)),
            purpose=f"D1 seed {seed} prediction input",
        )
    if len(set(paths.values())) != len(paths):
        # The fixed-ensemble P0 replay is allowed to be the exact matched-head
        # P0 CSV, but no other logical prediction input may alias another.
        aliases = {}
        for name, path in paths.items():
            aliases.setdefault(path, []).append(name)
        invalid = [
            names
            for names in aliases.values()
            if len(names) > 1 and set(names) != {"p0", "p0_head"}
        ]
        if invalid:
            raise ValueError(
                "P4, P5, and all three D1 prediction paths must be distinct."
            )
    frames = {name: tempo.load_prediction_table(path) for name, path in paths.items()}
    strictly_aligned_prediction_frames(frames)
    return paths, frames


def validate_fixed_result(
    *,
    result_path: Path,
    prediction_paths: Mapping[str, Path],
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    payload = read_json(result_path, purpose="fixed P5+D1 result")
    audit_embedded_absolute_paths(payload, source="fixed_ensemble_result")
    if payload.get("schema_version") != "l89-clean-fixed-p5-mean-d1-v1":
        raise ValueError("Unexpected fixed-ensemble result schema.")
    arithmetic = payload.get("arithmetic")
    if not isinstance(arithmetic, Mapping):
        raise KeyError("Fixed result lacks arithmetic receipt.")
    if tuple(arithmetic.get("d1_seeds", ())) != FROZEN_SEEDS:
        raise ValueError("Fixed result D1 seeds are not preregistered.")
    if (
        float(arithmetic.get("p5_logit_weight", math.nan)) != 0.5
        or float(arithmetic.get("mean_d1_logit_weight", math.nan)) != 0.5
        or arithmetic.get("weights_refit") is not False
    ):
        raise ValueError("Fixed result violates the 0.5/0.5 no-refit contract.")
    bootstrap_receipt = payload.get("bootstrap")
    if not isinstance(bootstrap_receipt, Mapping):
        raise KeyError("Fixed result lacks bootstrap receipt.")
    if (
        int(bootstrap_receipt.get("replicates", -1))
        != FROZEN_BOOTSTRAP_REPLICATES
        or int(bootstrap_receipt.get("seed", -1)) != FROZEN_BOOTSTRAP_SEED
        or bootstrap_receipt.get("point_selected_thresholds_fixed_per_system")
        is not True
        or bootstrap_receipt.get("thresholds_refit_per_replicate") is not False
    ):
        raise ValueError("Fixed result violates the frozen bootstrap contract.")
    if payload.get("test_or_sealed_or_holdout_or_outer_read") is not False:
        raise ValueError("Fixed result does not attest zero held-out access.")
    if payload.get("formal_inner_development_only") is not True:
        raise ValueError("Fixed result is not marked formal-inner-development-only.")

    provenance = payload.get("input_provenance")
    if not isinstance(provenance, Mapping):
        raise KeyError("Fixed result lacks prediction provenance.")
    fixed_input_names = {
        "p0",
        "p5",
        *(f"d1_seed_{seed}" for seed in FROZEN_SEEDS),
    }
    for name in sorted(fixed_input_names):
        path = prediction_paths[name]
        record = provenance.get(name)
        if not isinstance(record, Mapping):
            raise KeyError(f"Fixed result lacks provenance for {name}.")
        recorded = assert_inner_path(
            Path(str(record["path"])), purpose=f"fixed result {name} provenance"
        )
        if recorded != path or str(record.get("sha256", "")) != cache.sha256_file(path):
            raise ValueError(f"Fixed result provenance mismatch for {name}.")

    if set(provenance) != fixed_input_names:
        raise ValueError("Fixed result has an unexpected prediction-input set.")

    reference = frames["p5"]
    labels = reference["label"].to_numpy(dtype=np.int64)
    events = reference["event_id"].astype(str).tolist()
    p0_probability = frames["p0"]["probability"].to_numpy(dtype=np.float64)
    p5_probability = frames["p5"]["probability"].to_numpy(dtype=np.float64)
    d1_probability = [
        frames[f"d1_seed_{seed}"]["probability"].to_numpy(dtype=np.float64)
        for seed in FROZEN_SEEDS
    ]
    mean_d1, candidate = fixed_reference_mean_d1(
        p5_probability, d1_probability
    )
    probabilities = {
        "p0": p0_probability,
        "p5": p5_probability,
        "mean_seed_d1": mean_d1,
        "fixed_p5_mean_d1": candidate,
    }
    point_metrics = {
        name: tempo.metric_bundle(labels, values, events)
        for name, values in probabilities.items()
    }
    recorded_metrics = payload.get("point_metrics")
    if not isinstance(recorded_metrics, Mapping) or set(recorded_metrics) != set(
        point_metrics
    ):
        raise ValueError("Fixed result point-system set differs.")
    for name in point_metrics:
        compare_metric_bundle(
            recorded_metrics[name],
            point_metrics[name],
            source=f"fixed_result.point_metrics.{name}",
        )

    recomputed_bootstrap = fixed.fixed_threshold_event_bootstrap(
        labels,
        events,
        probabilities,
        point_metrics,
        (
            ("candidate_minus_p5", "fixed_p5_mean_d1", "p5"),
            ("candidate_minus_p0", "fixed_p5_mean_d1", "p0"),
        ),
        replicates=FROZEN_BOOTSTRAP_REPLICATES,
        seed=FROZEN_BOOTSTRAP_SEED,
    )
    recorded_bootstrap = payload.get("paired_event_cluster_bootstrap")
    if not isinstance(recorded_bootstrap, Mapping):
        raise KeyError("Fixed result lacks paired bootstrap.")
    compare_bootstrap(
        recorded_bootstrap,
        recomputed_bootstrap,
        source="fixed_result.paired_event_cluster_bootstrap",
    )

    prediction_record = payload.get("prediction_output")
    if not isinstance(prediction_record, Mapping):
        raise KeyError("Fixed result lacks candidate prediction output.")
    prediction_path = assert_inner_path(
        Path(str(prediction_record["path"])),
        purpose="fixed P5+D1 candidate prediction",
    )
    if str(prediction_record.get("sha256", "")) != cache.sha256_file(
        prediction_path
    ):
        raise ValueError("Fixed candidate prediction SHA mismatch.")
    candidate_frame = tempo.load_prediction_table(prediction_path)
    strictly_aligned_prediction_frames(
        {"p5": reference, "candidate": candidate_frame}
    )
    if not np.allclose(
        candidate_frame["probability"].to_numpy(dtype=np.float64),
        candidate,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Fixed candidate CSV does not replay frozen arithmetic.")
    return payload, point_metrics, recomputed_bootstrap


def recompute_head_metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities = frame["probability"].to_numpy(dtype=np.float64)
    events = frame["event_id"].astype(str).tolist()
    metrics = head_runner.prediction_metrics(labels, probabilities, events)
    weights = head_runner.mean_one_event_weights(events)
    positive_weight = head_runner.event_balanced_positive_weight(labels, events)
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    per_row = -(
        positive_weight * labels * np.log(clipped)
        + (1 - labels) * np.log1p(-clipped)
    )
    metrics["event_balanced_bce"] = float(np.mean(per_row * weights))
    return metrics


def validate_head_predictions(
    *,
    comparison: Mapping[str, Any],
    head_metrics: Mapping[str, Mapping[str, Any]],
    prediction_paths: Mapping[str, Path],
    frames: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    provenance = comparison.get("prediction_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "p0",
        "p4",
        "p5",
    }:
        raise ValueError(
            "Matched-head comparison lacks exact P0/P4/P5 prediction provenance."
        )
    frame_names = {"p0": "p0_head", "p4": "p4", "p5": "p5"}
    output: dict[str, Any] = {}
    formal_gate_metrics: dict[str, Mapping[str, Any]] = {}
    for arm, frame_name in frame_names.items():
        path = prediction_paths[frame_name]
        record = provenance.get(arm)
        if not isinstance(record, Mapping):
            raise TypeError(f"Head prediction provenance for {arm} is invalid.")
        recorded_path = assert_inner_path(
            Path(str(record.get("path", ""))),
            purpose=f"matched-head {arm} prediction provenance",
        )
        observed_sha = cache.sha256_file(path)
        if (
            recorded_path != path
            or str(record.get("sha256", "")) != observed_sha
        ):
            raise ValueError(
                f"Matched-head {arm} prediction path/SHA does not match comparison."
            )
        recomputed = recompute_head_metric_bundle(frames[frame_name])
        recorded_metrics = head_metrics[arm]
        metric_keys = set(recorded_metrics) - {"best_epoch"}
        if metric_keys != set(recomputed):
            raise ValueError(
                f"Head comparison {arm} metric schema differs from independent "
                f"recomputation: recorded={sorted(metric_keys)}, "
                f"recomputed={sorted(recomputed)}."
            )
        for metric in sorted(metric_keys):
            observed = finite_value(
                recorded_metrics[metric],
                name=f"head.{arm}.{metric}",
            )
            expected = finite_value(
                recomputed[metric],
                name=f"head_recomputed.{arm}.{metric}",
            )
            tolerance = 2e-5 if metric == "event_balanced_bce" else 1e-8
            if not values_close(observed, expected, tolerance=tolerance):
                raise ValueError(
                    f"Head comparison {arm}.{metric} does not match its "
                    f"prediction CSV: {observed} != {expected}."
                )
        if "epoch" in frames[frame_name].columns:
            epochs = set(
                frames[frame_name]["epoch"].to_numpy(dtype=np.int64).tolist()
            )
            if epochs != {int(recorded_metrics["best_epoch"])}:
                raise ValueError(
                    f"Head prediction {arm} epoch does not match comparison."
                )
        if "arm" in frames[frame_name].columns and set(
            frames[frame_name]["arm"].astype(str)
        ) != {arm}:
            raise ValueError(f"Head prediction {arm} arm column is invalid.")
        labels = frames[frame_name]["label"].to_numpy(dtype=np.int64)
        probabilities = frames[frame_name]["probability"].to_numpy(
            dtype=np.float64
        )
        events = frames[frame_name]["event_id"].astype(str).tolist()
        formal_bundle = tempo.metric_bundle(labels, probabilities, events)
        formal_bundle_sha = cache.sha256_bytes(
            cache.canonical_json_bytes(formal_bundle)
        )
        formal_gate_metrics[arm] = formal_bundle
        output[arm] = {
            "path": str(path),
            "sha256": observed_sha,
            "strict_identity_columns": [
                "id",
                "plume_id",
                "event_id",
                "label",
            ],
            "all_comparison_metrics_recomputed": True,
            "recomputed_metrics": recomputed,
            "formal_gate_metrics": formal_bundle,
            "formal_gate_metrics_sha256": formal_bundle_sha,
            "formal_gate_metrics_independently_recomputed": True,
        }
    return output, formal_gate_metrics


def load_d1_mechanism_receipts(
    *,
    d1_paths: Mapping[str, Path],
    d1_summary_template: Optional[str],
    point_metrics: Mapping[str, Mapping[str, Any]],
    p0_metrics: Mapping[str, Any],
    head_comparison_path: Path,
    p0_head_prediction_path: Path,
    p0_head_frame: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    provenance: dict[str, dict[str, str]] = {}
    expected_p0_checkpoint = assert_inner_path(
        head_comparison_path.parent
        / "p0"
        / "checkpoint_best_event_balanced_ap.pt",
        purpose="fresh matched-head P0 parent checkpoint",
    )
    if not expected_p0_checkpoint.is_file():
        raise FileNotFoundError(expected_p0_checkpoint)
    expected_parent_file_sha = cache.sha256_file(expected_p0_checkpoint)
    parent_checkpoint = cache.torch_load_trusted(expected_p0_checkpoint)
    if (
        not isinstance(parent_checkpoint, Mapping)
        or parent_checkpoint.get("arm") != "p0"
        or not isinstance(parent_checkpoint.get("model"), Mapping)
    ):
        raise ValueError("Fresh P0 parent checkpoint payload is invalid.")
    require_false_flag(
        parent_checkpoint,
        "test_or_sealed_or_holdout_read",
        source="p0_parent_checkpoint",
    )
    expected_parent_model_sha = cache.state_dict_sha256(
        parent_checkpoint["model"]
    )
    expected_parent_epoch = int(parent_checkpoint.get("epoch", -1))
    if expected_parent_epoch <= 0:
        raise ValueError("Fresh P0 parent selected epoch is invalid.")
    expected_p0_prediction_sha = cache.sha256_file(p0_head_prediction_path)
    root_paths: set[Path] = set()
    base_audits: list[Mapping[str, Any]] = []
    seed_p0_replays: dict[str, dict[str, Any]] = {}

    for seed in FROZEN_SEEDS:
        prediction_path = d1_paths[f"d1_seed_{seed}"]
        seed_p0_prediction_path = assert_inner_path(
            prediction_path.parent / "p0_base_predictions.csv",
            purpose=f"D1 seed {seed} sibling P0 replay",
        )
        seed_p0_frame = tempo.load_prediction_table(
            seed_p0_prediction_path
        )
        seed_p0_replay = validate_p0_probability_replay(
            reference_frame=p0_head_frame,
            candidate_frame=seed_p0_frame,
            reference_path=p0_head_prediction_path,
            candidate_path=seed_p0_prediction_path,
            source=f"d1_seed_{seed}_sibling_p0",
        )
        seed_p0_replays[str(seed)] = seed_p0_replay
        summary_path = (
            Path(str(d1_summary_template).format(seed=seed))
            if d1_summary_template
            else prediction_path.parent / "summary.json"
        )
        summary_path = assert_inner_path(
            summary_path, purpose=f"D1 seed {seed} summary"
        )
        root_paths.add(summary_path.parent.parent)
        summary = read_json(summary_path, purpose=f"D1 seed {seed} summary")
        audit_embedded_absolute_paths(summary, source=f"d1_summary_{seed}")
        if int(summary.get("seed", -1)) != seed:
            raise ValueError(f"D1 summary seed mismatch for {seed}.")
        require_false_flag(
            summary,
            "test_or_sealed_or_holdout_read",
            source=f"d1_summary_{seed}",
        )
        results = summary.get("results")
        if not isinstance(results, Mapping):
            raise KeyError(f"D1 seed {seed} summary lacks results.")
        d1 = results.get("d1_gated_delta")
        p0 = results.get("p0_base")
        if not isinstance(d1, Mapping) or not isinstance(p0, Mapping):
            raise KeyError(f"D1 seed {seed} lacks D1/P0 records.")
        if (
            str(d1.get("arm", "")) != "d1_gated_delta"
            or int(d1.get("seed", -1)) != seed
            or str(p0.get("arm", "")) != "p0_base"
            or int(p0.get("seed", -1)) != seed
        ):
            raise ValueError(f"D1 seed {seed} arm/seed receipts are invalid.")
        coherent = d1.get("best", {}).get("validation")
        shuffled = d1.get("history_shuffle_fixed_model_and_threshold")
        replay = p0.get("epoch_zero_exact_frozen_base")
        if not isinstance(coherent, Mapping) or not isinstance(
            shuffled, Mapping
        ) or not isinstance(replay, Mapping):
            raise KeyError(f"D1 seed {seed} lacks shuffle or epoch-zero receipt.")
        if replay.get("exact_within_tolerance") is not True:
            raise ValueError(f"D1 seed {seed} failed exact P0 epoch-zero replay.")
        error = finite_value(
            replay.get("replay_max_abs_probability_error"),
            name=f"D1 seed {seed} replay error",
        )
        tolerance = finite_value(
            replay.get("replay_tolerance"),
            name=f"D1 seed {seed} replay tolerance",
        )
        if error > tolerance:
            raise ValueError(f"D1 seed {seed} replay error exceeds tolerance.")

        top_base_audit = summary.get("frozen_role_only_base_audit")
        nested_base_audit = p0.get("frozen_role_only_base_audit")
        if not isinstance(top_base_audit, Mapping) or not isinstance(
            nested_base_audit, Mapping
        ):
            raise KeyError(f"D1 seed {seed} lacks frozen P0 parent audit.")
        if cache.canonical_json_bytes(top_base_audit) != cache.canonical_json_bytes(
            nested_base_audit
        ):
            raise ValueError(f"D1 seed {seed} has inconsistent P0 parent audits.")
        audit_embedded_absolute_paths(
            top_base_audit, source=f"d1_summary_{seed}.p0_parent"
        )
        recorded_parent_path = assert_inner_path(
            Path(str(top_base_audit.get("checkpoint", ""))),
            purpose=f"D1 seed {seed} P0 parent checkpoint",
        )
        recorded_parent_prediction = assert_inner_path(
            Path(str(top_base_audit.get("prediction_csv", ""))),
            purpose=f"D1 seed {seed} P0 parent prediction",
        )
        if (
            recorded_parent_path != expected_p0_checkpoint
            or str(top_base_audit.get("checkpoint_sha256", ""))
            != expected_parent_file_sha
            or str(top_base_audit.get("model_state_sha256", ""))
            != expected_parent_model_sha
            or int(top_base_audit.get("checkpoint_epoch", -1))
            != expected_parent_epoch
            or recorded_parent_prediction != p0_head_prediction_path
            or str(top_base_audit.get("prediction_csv_sha256", ""))
            != expected_p0_prediction_sha
            or top_base_audit.get("replay_numerically_exact") is not True
            or str(top_base_audit.get("base_kind", ""))
            != "event_balanced_p0"
            or top_base_audit.get("response_sidecar_used") is not False
        ):
            raise ValueError(
                f"D1 seed {seed} does not close to the exact fresh P0 parent."
            )
        if (
            int(replay.get("source_checkpoint_epoch", -1))
            != expected_parent_epoch
            or str(summary.get("base_initial_state_sha256", ""))
            != expected_parent_model_sha
        ):
            raise ValueError(f"D1 seed {seed} epoch-zero parent identity differs.")
        temporal_replay = d1.get("epoch_zero_exact_p0_replay")
        top_temporal_replay = summary.get(
            "zero_initialized_d1_prototype_exact_p0_replay"
        )
        p0_temporal_replay = p0.get(
            "zero_initialized_d1_prototype_exact_p0_replay"
        )
        if (
            not isinstance(temporal_replay, Mapping)
            or not isinstance(top_temporal_replay, Mapping)
            or not isinstance(p0_temporal_replay, Mapping)
            or cache.canonical_json_bytes(temporal_replay)
            != cache.canonical_json_bytes(top_temporal_replay)
            or cache.canonical_json_bytes(temporal_replay)
            != cache.canonical_json_bytes(p0_temporal_replay)
        ):
            raise ValueError(
                f"D1 seed {seed} lacks one exact zero-init prototype replay receipt."
            )
        probability_error = finite_value(
            temporal_replay.get("maximum_absolute_probability_error"),
            name=f"D1 seed {seed} zero-init probability error",
        )
        probability_tolerance = finite_value(
            temporal_replay.get("probability_tolerance"),
            name=f"D1 seed {seed} zero-init probability tolerance",
        )
        logit_error = finite_value(
            temporal_replay.get("maximum_absolute_logit_error"),
            name=f"D1 seed {seed} zero-init logit error",
        )
        logit_tolerance = finite_value(
            temporal_replay.get("logit_tolerance"),
            name=f"D1 seed {seed} zero-init logit tolerance",
        )
        if (
            temporal_replay.get("pass") is not True
            or temporal_replay.get(
                "evaluated_before_any_temporal_optimizer_step"
            )
            is not True
            or temporal_replay.get("zero_initialized_residual_output")
            is not True
            or str(temporal_replay.get("arm", "")) != "d1_gated_delta"
            or str(temporal_replay.get("fresh_p0_checkpoint", ""))
            != str(expected_p0_checkpoint)
            or str(temporal_replay.get("fresh_p0_checkpoint_sha256", ""))
            != expected_parent_file_sha
            or str(temporal_replay.get("fresh_p0_model_state_sha256", ""))
            != expected_parent_model_sha
            or str(temporal_replay.get("temporal_initial_state_sha256", ""))
            != str(summary.get("temporal_initial_state_sha256", ""))
            or int(temporal_replay.get("rows", -1))
            != int(p0_metrics["rows"])
            or probability_error > probability_tolerance
            or logit_error > logit_tolerance
        ):
            raise ValueError(
                f"D1 seed {seed} zero-init prototype did not exactly replay P0."
            )
        base_audits.append(top_base_audit)

        compare_metric_bundle(
            coherent,
            point_metrics[f"d1_seed_{seed}"],
            source=f"d1_summary_{seed}.coherent",
        )
        p0_summary_metrics = p0.get("best", {}).get("validation")
        if not isinstance(p0_summary_metrics, Mapping):
            raise KeyError(f"D1 seed {seed} lacks P0 base metrics.")
        compare_metric_bundle(
            p0_summary_metrics,
            p0_metrics,
            source=f"d1_summary_{seed}.p0_base",
        )

        early = d1.get("early_stop_receipt")
        if not isinstance(early, Mapping) or early.get("valid") is not True:
            raise ValueError(f"D1 seed {seed} lacks a valid early-stop receipt.")
        history_path = assert_inner_path(
            Path(str(early.get("history_path", ""))),
            purpose=f"D1 seed {seed} D1 history",
        )
        expected_history_path = (
            summary_path.parent / "d1_gated_delta_metrics_history.json"
        ).resolve()
        if history_path != expected_history_path or not history_path.is_file():
            raise ValueError(f"D1 seed {seed} history path is not canonical.")
        history_sha = cache.sha256_file(history_path)
        if str(early.get("history_sha256", "")) != history_sha:
            raise ValueError(f"D1 seed {seed} history SHA mismatch.")
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history_payload, list) or not history_payload:
            raise ValueError(f"D1 seed {seed} history is empty or invalid.")
        observed_epochs = [
            int(record.get("epoch", -1)) for record in history_payload
        ]
        if observed_epochs != list(range(1, len(history_payload) + 1)):
            raise ValueError(f"D1 seed {seed} epochs are not consecutive.")
        if not 1 <= len(history_payload) <= 4:
            raise ValueError(f"D1 seed {seed} violates max_epochs=4.")
        observed_ap = [
            finite_unit(
                record["validation"]["event_balanced_ap"],
                name=f"D1 seed {seed} epoch AP",
            )
            for record in history_payload
        ]
        selected_ap = max(observed_ap)
        selected_epoch = min(
            epoch
            for epoch, ap in zip(observed_epochs, observed_ap)
            if ap == selected_ap
        )
        first_non_improvement: Optional[int] = None
        running_best = -math.inf
        for epoch, ap in zip(observed_epochs, observed_ap):
            if ap > running_best:
                running_best = ap
            else:
                first_non_improvement = epoch
                break
        expected_stop_reason = (
            "patience_exhausted"
            if first_non_improvement is not None
            else "max_epochs_reached"
        )
        if first_non_improvement is not None:
            if first_non_improvement != observed_epochs[-1]:
                raise ValueError(
                    f"D1 seed {seed} continued after patience=1 was exhausted."
                )
        elif len(history_payload) != 4:
            raise ValueError(
                f"D1 seed {seed} stopped before max4 without AP non-improvement."
            )
        if (
            int(early.get("max_epochs", -1)) != 4
            or int(early.get("patience", -1)) != 1
            or int(early.get("epochs_observed", -1)) != len(history_payload)
            or list(early.get("observed_epochs", ())) != observed_epochs
            or list(early.get("observed_event_balanced_ap", ()))
            != observed_ap
            or int(early.get("selected_epoch", -1)) != selected_epoch
            or not values_close(
                float(early.get("selected_event_balanced_ap", math.nan)),
                selected_ap,
                tolerance=1e-12,
            )
            or str(early.get("stop_reason", "")) != expected_stop_reason
            or int(d1.get("best", {}).get("epoch", -1)) != selected_epoch
        ):
            raise ValueError(f"D1 seed {seed} early-stop receipt is inconsistent.")
        best_history_record = history_payload[selected_epoch - 1]
        compare_metric_bundle(
            d1["best"]["validation"],
            best_history_record["validation"],
            source=f"d1_summary_{seed}.selected_history",
        )

        d1_checkpoint_path = (
            summary_path.parent / "d1_gated_delta_best_event_ap.pt"
        ).resolve()
        if not d1_checkpoint_path.is_file():
            raise FileNotFoundError(d1_checkpoint_path)
        d1_checkpoint = cache.torch_load_trusted(d1_checkpoint_path)
        if (
            not isinstance(d1_checkpoint, Mapping)
            or d1_checkpoint.get("arm") != "d1_gated_delta"
            or int(d1_checkpoint.get("seed", -1)) != seed
            or int(d1_checkpoint.get("epoch", -1)) != selected_epoch
            or str(d1_checkpoint.get("initial_state_sha256", ""))
            != str(d1.get("initial_state_sha256", ""))
            or str(d1_checkpoint.get("initial_state_sha256", ""))
            != str(summary.get("temporal_initial_state_sha256", ""))
            or str(d1_checkpoint.get("frozen_base_checkpoint", ""))
            != str(expected_p0_checkpoint)
            or str(d1_checkpoint.get("frozen_base_checkpoint_sha256", ""))
            != expected_parent_file_sha
            or str(d1_checkpoint.get("frozen_base_model_state_sha256", ""))
            != expected_parent_model_sha
        ):
            raise ValueError(f"D1 seed {seed} checkpoint lineage is invalid.")
        require_false_flag(
            d1_checkpoint,
            "test_or_sealed_or_holdout_read",
            source=f"d1_checkpoint_{seed}",
        )

        diagnostic_metrics: dict[str, Any] = {}
        recorded_delta = d1.get("history_shuffle_delta")
        shuffle_validity = d1.get("history_shuffle_validity")
        required_shuffle_validity = (
            "all_target_donor_strata_equal",
            "all_cross_event",
            "all_donors_from_different_canonical_event",
            "ordered_id_preserved",
            "ordered_plume_id_preserved",
            "ordered_event_id_preserved",
            "labels_preserved",
            "shapes_preserved",
            "t0_feature_preserved",
            "availability_pattern_preserved",
            "acquisition_metadata_preserved",
            "only_history_features_replaced",
        )
        shuffle_strata = (
            shuffle_validity.get("strata")
            if isinstance(shuffle_validity, Mapping)
            else None
        )
        strata_count = (
            int(shuffle_validity.get("strata_count", -1))
            if isinstance(shuffle_validity, Mapping)
            else -1
        )
        strata_receipts_valid = (
            isinstance(shuffle_strata, list)
            and len(shuffle_strata) == strata_count
            and all(
                isinstance(record, Mapping)
                and int(record.get("rows", 0)) > 0
                and int(record.get("canonical_event_count", 0)) >= 2
                and len(record.get("valid_history_pattern", ())) == 5
                and len(record.get("unique_history_pattern", ())) == 5
                and record.get("all_cross_event") is True
                and re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(record.get("donor_indices_sha256", "")),
                )
                is not None
                for record in (shuffle_strata or ())
            )
        )
        if (
            not isinstance(shuffle_validity, Mapping)
            or shuffle_validity.get("valid") is not True
            or int(shuffle_validity.get("seed", -1)) != seed + 8_191
            or int(shuffle_validity.get("rows", -1))
            != int(coherent["rows"])
            or len(shuffle_validity.get("history_indices", ())) != 5
            or strata_count < 1
            or int(shuffle_validity.get("pattern_count", -1))
            != strata_count
            or not strata_receipts_valid
            or int(
                shuffle_validity.get(
                    "availability_mismatch_rows", -1
                )
            )
            != 0
            or int(
                shuffle_validity.get(
                    "availability_pattern_mismatch_count", -1
                )
            )
            != 0
            or list(shuffle_validity.get("failed_strata", ())) != []
            or any(
                shuffle_validity.get(key) is not True
                for key in required_shuffle_validity
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(shuffle_validity.get("donor_indices_sha256", "")),
            )
        ):
            raise ValueError(
                f"D1 seed {seed} history shuffle identity/label/shape "
                "validity receipt failed."
            )
        if not isinstance(recorded_delta, Mapping) or set(recorded_delta) != set(
            GATE_METRICS
        ):
            raise ValueError(f"D1 seed {seed} shuffle delta receipt is invalid.")
        if not values_close(
            get_alias(shuffled, "selected_threshold", source=f"d1.shuffle.{seed}"),
            get_alias(coherent, "selected_threshold", source=f"d1.coherent.{seed}"),
            tolerance=1e-12,
        ):
            raise ValueError(f"D1 seed {seed} shuffle refit the threshold.")
        for invariant in (
            "rows",
            "events",
            "all_negative_event_count",
            "positive_event_count",
        ):
            if int(shuffled.get(invariant, -1)) != int(coherent.get(invariant, -2)):
                raise ValueError(
                    f"D1 seed {seed} shuffle changed invariant {invariant}."
                )
        for metric in GATE_METRICS:
            coherent_value = get_alias(
                coherent, metric, source=f"d1_summary_{seed}.coherent"
            )
            shuffled_value = get_alias(
                shuffled, metric, source=f"d1_summary_{seed}.shuffle"
            )
            if metric != "all_negative_fp_mass" and not (
                0.0 <= shuffled_value <= 1.0
            ):
                raise ValueError(
                    f"D1 seed {seed} shuffled {metric} is outside [0,1]."
                )
            delta = shuffled_value - coherent_value
            if not values_close(
                float(recorded_delta[metric]), delta, tolerance=1e-12
            ):
                raise ValueError(
                    f"D1 seed {seed} shuffle delta for {metric} is inconsistent."
                )
            diagnostic_metrics[metric] = {
                "coherent": coherent_value,
                "matched_cross_event_history_shuffle": shuffled_value,
                "shuffle_minus_coherent": delta,
            }
        diagnostics[str(seed)] = {
            "sibling_p0_base_probability_replay": seed_p0_replay,
            "epoch_zero_exact_p0_replay": {
                "pass": True,
                "maximum_absolute_probability_error": error,
                "tolerance": tolerance,
            },
            "zero_initialized_d1_prototype_exact_p0_replay": {
                **dict(temporal_replay),
                "independently_validated": True,
            },
            "fresh_p0_parent": {
                "checkpoint_path": str(expected_p0_checkpoint),
                "checkpoint_file_sha256": expected_parent_file_sha,
                "model_state_sha256": expected_parent_model_sha,
                "selected_epoch": expected_parent_epoch,
                "path_file_and_model_state_match": True,
            },
            "early_stop": {
                **dict(early),
                "history_independently_replayed": True,
                "patience_1_max_epochs_4_valid": True,
                "observed_event_balanced_ap": observed_ap,
                "selected_event_balanced_ap": selected_ap,
                "checkpoint_epoch": int(d1_checkpoint["epoch"]),
                "checkpoint_path": str(d1_checkpoint_path),
                "checkpoint_sha256": cache.sha256_file(d1_checkpoint_path),
            },
            "matched_cross_event_history_shuffle": diagnostic_metrics,
            "history_shuffle_validity": {
                **dict(shuffle_validity),
                "independently_validated": True,
            },
            "history_shuffle_valid": True,
            "selection_or_gate_role": "diagnostic only",
        }
        provenance[str(seed)] = {
            "path": str(summary_path),
            "sha256": cache.sha256_file(summary_path),
            "history_path": str(history_path),
            "history_sha256": history_sha,
            "checkpoint_path": str(d1_checkpoint_path),
            "checkpoint_sha256": cache.sha256_file(d1_checkpoint_path),
            "sibling_p0_replay_path": str(seed_p0_prediction_path),
            "sibling_p0_replay_sha256": cache.sha256_file(
                seed_p0_prediction_path
            ),
        }
    if len(root_paths) != 1:
        raise ValueError("D1 summaries do not share one current-run root.")
    if any(
        cache.canonical_json_bytes(audit)
        != cache.canonical_json_bytes(base_audits[0])
        for audit in base_audits[1:]
    ):
        raise ValueError("Three D1 seeds do not share the exact fresh P0 parent.")
    d1_root = next(iter(root_paths))
    run_config_path = assert_inner_path(
        d1_root / "run_config.json", purpose="D1 formal run config"
    )
    run_status_path = assert_inner_path(
        d1_root / "run_status.json", purpose="D1 formal run status"
    )
    run_config = read_json(run_config_path, purpose="D1 formal run config")
    run_status = read_json(run_status_path, purpose="D1 formal run status")
    configured_seeds = tuple(
        int(value.strip())
        for value in str(run_config.get("seeds", "")).split(",")
        if value.strip()
    )
    if (
        configured_seeds != FROZEN_SEEDS
        or tuple(run_config.get("resolved_arms", ()))
        != ("p0_base", "d1_gated_delta")
        or str(run_config.get("base_kind", "")) != "event_balanced_p0"
        or int(run_config.get("epochs", -1)) != 4
        or int(run_config.get("patience", -1)) != 1
        or assert_inner_path(
            Path(str(run_config.get("event_base_checkpoint", ""))),
            purpose="D1 configured P0 parent",
        )
        != expected_p0_checkpoint
        or run_config.get("test_or_sealed_or_holdout_read") is not False
    ):
        raise ValueError("D1 run config violates the frozen three-seed contract.")
    if (
        run_status.get("status") != "complete"
        or tuple(int(value) for value in run_status.get("seeds", ()))
        != FROZEN_SEEDS
        or tuple(run_status.get("arms", ()))
        != ("p0_base", "d1_gated_delta")
        or run_status.get("test_or_sealed_or_holdout_read") is not False
    ):
        raise ValueError("D1 formal run status is incomplete or invalid.")
    formal_validation = {
        "validated_for_all_promotion_branches": True,
        "seeds": list(FROZEN_SEEDS),
        "run_config": {
            "path": str(run_config_path),
            "sha256": cache.sha256_file(run_config_path),
        },
        "run_status": {
            "path": str(run_status_path),
            "sha256": cache.sha256_file(run_status_path),
        },
        "fresh_p0_parent": {
            "checkpoint_path": str(expected_p0_checkpoint),
            "checkpoint_file_sha256": expected_parent_file_sha,
            "model_state_sha256": expected_parent_model_sha,
            "prediction_path": str(p0_head_prediction_path),
            "prediction_sha256": expected_p0_prediction_sha,
        },
        "all_seed_parent_audits_identical": True,
        "per_seed_sibling_p0_probability_replays": seed_p0_replays,
        "all_seed_sibling_p0_probability_replays_match_fresh_head": True,
        "all_seed_zero_initialized_d1_prototypes_exactly_replay_p0": True,
        "all_seed_history_shuffle_receipts_valid": True,
        "all_seed_patience_1_max4_receipts_valid": True,
    }
    return diagnostics, provenance, formal_validation


def fallback_audit(
    *,
    frames: Mapping[str, pd.DataFrame],
    d1_diagnostics: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    reference = frames["p0"]
    labels = reference["label"].to_numpy(dtype=np.int64)
    events = reference["event_id"].astype(str).tolist()
    p0_probability = reference["probability"].to_numpy(dtype=np.float64)
    d1_probability = [
        frames[f"d1_seed_{seed}"]["probability"].to_numpy(dtype=np.float64)
        for seed in FROZEN_SEEDS
    ]
    mean_d1, candidate = fixed_reference_mean_d1(
        p0_probability, d1_probability
    )
    probabilities: dict[str, np.ndarray] = {"p0": p0_probability}
    probabilities.update(
        {
            f"d1_seed_{seed}": values
            for seed, values in zip(FROZEN_SEEDS, d1_probability)
        }
    )
    probabilities["mean_seed_d1"] = mean_d1
    probabilities["fixed_p0_mean_d1"] = candidate
    point_metrics = {
        name: tempo.metric_bundle(labels, values, events)
        for name, values in probabilities.items()
    }
    bootstrap = fixed.fixed_threshold_event_bootstrap(
        labels,
        events,
        probabilities,
        point_metrics,
        (("fallback_minus_p0", "fixed_p0_mean_d1", "p0"),),
        replicates=FROZEN_BOOTSTRAP_REPLICATES,
        seed=FROZEN_BOOTSTRAP_SEED,
    )
    output_frame = reference.copy()
    output_frame["probability"] = candidate
    output_frame["selected_threshold"] = float(
        point_metrics["fixed_p0_mean_d1"]["selected_threshold"]
    )
    output_frame["system"] = "fixed_p0_mean_d1"
    payload = {
        "formula": "0.5*logit(P0)+0.5*mean_seed(logit(D1))",
        "arithmetic": {
            "d1_seed_aggregation": "mean of exactly three D1 logits before fusion",
            "d1_seeds": list(FROZEN_SEEDS),
            "p0_logit_weight": 0.5,
            "mean_d1_logit_weight": 0.5,
            "weights_refit": False,
        },
        "point_metrics": point_metrics,
        "paired_event_cluster_bootstrap": bootstrap,
        "mechanism_diagnostics": dict(d1_diagnostics),
        "bootstrap": {
            "unit": "canonical event_id",
            "replicates": FROZEN_BOOTSTRAP_REPLICATES,
            "seed": FROZEN_BOOTSTRAP_SEED,
            "point_selected_thresholds_fixed_per_system": True,
            "thresholds_refit_per_replicate": False,
        },
    }
    return payload, output_frame


def render_markdown(payload: Mapping[str, Any]) -> str:
    qualification = payload["p5_qualification"]
    lines = [
        "# Clean L89 promotion-chain audit",
        "",
        f"- `P5_qualified`: `{str(qualification['P5_qualified']).lower()}`",
        f"- `audit_type`: `{payload['audit_type']}`",
        f"- `promotion_eligible`: "
        f"`{str(payload['promotion_eligible']).lower()}`",
        f"- `promotion`: `{str(payload['promotion']).lower()}`",
        f"- `outer_evaluation_eligible`: "
        f"`{str(payload['outer_evaluation_eligible']).lower()}`",
        "",
        "## P5 qualification",
        "",
        "| Condition | Observed | Requirement | Pass |",
        "|---|---:|---:|:---:|",
    ]
    for name, item in qualification["conditions"].items():
        lines.append(
            f"| {name} | {item['observed']:+.6f} | "
            f"{item['operator']} {item['boundary']:+.6f} | "
            f"{'yes' if item['pass'] else 'no'} |"
        )
    if qualification["P5_qualified"]:
        gate = payload["promotion_gate"]
        lines.extend(
            [
                "",
                "## P5+D1 promotion gate",
                "",
                "| Condition | Observed | Requirement | Pass |",
                "|---|---:|---:|:---:|",
            ]
        )
        for name, item in gate["conditions"].items():
            lines.append(
                f"| {name} | {item['observed']:+.6f} | "
                f"{item['operator']} {item['boundary']:+.6f} | "
                f"{'yes' if item['pass'] else 'no'} |"
            )
    else:
        fallback = payload["fallback"]
        metrics = fallback["point_metrics"]
        lines.extend(
            [
                "",
                "## Fixed P0+D1 mechanism-only fallback",
                "",
                "`0.5*logit(P0) + 0.5*mean_seed(logit(D1))`",
                "",
                "| System | AP | AUC | Positive-F1 | Macro-F1 | FP mass |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        ordered = [
            "p0",
            *(f"d1_seed_{seed}" for seed in FROZEN_SEEDS),
            "mean_seed_d1",
            "fixed_p0_mean_d1",
        ]
        for name in ordered:
            point = metrics[name]
            lines.append(
                f"| {name} | {point['event_balanced_ap']:.6f} | "
                f"{point['event_balanced_auc']:.6f} | "
                f"{point['event_balanced_positive_f1_selected']:.6f} | "
                f"{point['event_balanced_macro_f1_selected']:.6f} | "
                f"{point['all_negative_fp_mass']:.6f} |"
            )
    lines.extend(
        [
            "",
            "This is an event-disjoint train-only inner replicate. It is not an",
            "outer or confirmatory evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_formal_context(args: argparse.Namespace) -> Optional[dict[str, Any]]:
    """Bind a formal CLI run to its protocol, split, weights, and load ledger."""

    names = (
        "protocol",
        "train_manifest",
        "dev_manifest",
        "split_receipt",
        "model_ledger_final",
        "base_weights",
    )
    supplied = {
        name: getattr(args, name, None)
        for name in names
        if getattr(args, name, None)
    }
    if not supplied:
        # Unit-level direct calls exercise gate arithmetic without constructing
        # a complete formal lineage. The CLI parser requires every field.
        return None
    if set(supplied) != set(names):
        raise ValueError("Formal context is only valid when all fields are supplied.")
    paths = {
        name: assert_inner_path(Path(value), purpose=f"formal {name}")
        for name, value in supplied.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    split = read_json(paths["split_receipt"], purpose="formal split receipt")
    if (
        split.get("schema_version") != "l89-clean-resolved-manifests-v1"
        or split.get("event_overlap") != 0
        or split.get("complete") is not True
        or split.get("formal_full_manifest") is not True
        or int(split.get("missing_role_files", -1)) != 0
    ):
        raise ValueError("Formal resolved split receipt is incomplete.")
    for name, key in (("train_manifest", "train"), ("dev_manifest", "dev")):
        output = split.get("outputs", {}).get(key, {})
        if (
            assert_inner_path(
                Path(str(output.get("path", ""))),
                purpose=f"formal split {key} output",
            )
            != paths[name]
            or output.get("sha256") != cache.sha256_file(paths[name])
        ):
            raise ValueError(f"Formal split receipt does not bind {key}.")
    ledger = read_json(
        paths["model_ledger_final"], purpose="formal model-load ledger"
    )
    if (
        ledger.get("status") != "finalized"
        or ledger.get("known_old_checkpoint_sha_match") is not False
        or ledger.get("old_sidecar_checkpoint_loaded") is not False
        or ledger.get("old_head_checkpoint_loaded") is not False
        or ledger.get("old_d1_checkpoint_loaded") is not False
        or ledger.get("test_or_sealed_or_holdout_or_outer_read") is not False
    ):
        raise ValueError("Formal model-load ledger did not close cleanly.")
    weights_sha = cache.sha256_file(paths["base_weights"])
    expected_weights_sha = (
        "55024f411a7f383ed1a646d9b833b65683a4443846603fc7d37591d5afe9d26e"
    )
    if weights_sha != expected_weights_sha:
        raise ValueError("Formal public Panopticon PTH SHA changed.")
    return {
        name: {"path": str(path), "sha256": cache.sha256_file(path)}
        for name, path in paths.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = assert_inner_path(
        Path(args.output_dir), purpose="promotion-chain output"
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    formal_context = validate_formal_context(args)
    head_comparison_path = assert_inner_path(
        Path(args.head_comparison), purpose="matched head comparison"
    )
    p0_head_prediction_path = assert_inner_path(
        (
            Path(args.p0_head_predictions)
            if args.p0_head_predictions
            else (
                head_comparison_path.parent
                / "p0"
                / "validation_best_event_balanced_ap_predictions.csv"
            )
        ),
        purpose="matched-head P0 prediction input",
    )
    fixed_result_path = assert_inner_path(
        Path(args.fixed_ensemble_result), purpose="fixed ensemble result"
    )

    sidecars = {
        "p4": extract_sidecar_capability(
            arm="p4",
            summary_path=Path(args.p4_sidecar_summary),
            checkpoint_path=Path(args.p4_sidecar_checkpoint),
        ),
        "p5": extract_sidecar_capability(
            arm="p5",
            summary_path=Path(args.p5_sidecar_summary),
            checkpoint_path=Path(args.p5_sidecar_checkpoint),
        ),
    }
    validate_sidecar_pair(sidecars["p4"], sidecars["p5"])
    head_payload, head_metrics = validate_head_comparison(head_comparison_path)
    prediction_paths, frames = load_prediction_inputs(
        p0_path=Path(args.p0_predictions),
        p0_head_path=p0_head_prediction_path,
        p4_path=Path(args.p4_predictions),
        p5_path=Path(args.p5_predictions),
        d1_template=args.d1_template,
    )
    (
        head_prediction_audit,
        formal_head_gate_metrics,
    ) = validate_head_predictions(
        comparison=head_payload,
        head_metrics=head_metrics,
        prediction_paths=prediction_paths,
        frames=frames,
    )
    fixed_p0_replay_receipt = validate_p0_probability_replay(
        reference_frame=frames["p0_head"],
        candidate_frame=frames["p0"],
        reference_path=p0_head_prediction_path,
        candidate_path=prediction_paths["p0"],
        source="fixed_ensemble_p0",
    )
    fixed_payload, primary_metrics, primary_bootstrap = validate_fixed_result(
        result_path=fixed_result_path,
        prediction_paths=prediction_paths,
        frames=frames,
    )
    d1_labels = frames["p0"]["label"].to_numpy(dtype=np.int64)
    d1_events = frames["p0"]["event_id"].astype(str).tolist()
    d1_point_metrics = {
        f"d1_seed_{seed}": tempo.metric_bundle(
            d1_labels,
            frames[f"d1_seed_{seed}"]["probability"].to_numpy(
                dtype=np.float64
            ),
            d1_events,
        )
        for seed in FROZEN_SEEDS
    }
    d1_receipt_metrics = {
        "p0": primary_metrics["p0"],
        **d1_point_metrics,
    }
    (
        d1_diagnostics,
        d1_summary_provenance,
        d1_formal_validation,
    ) = load_d1_mechanism_receipts(
        d1_paths=prediction_paths,
        d1_summary_template=args.d1_summary_template,
        point_metrics=d1_receipt_metrics,
        p0_metrics=primary_metrics["p0"],
        head_comparison_path=head_comparison_path,
        p0_head_prediction_path=p0_head_prediction_path,
        p0_head_frame=frames["p0_head"],
    )
    qualification = evaluate_p5_qualification(
        p4_synthetic_ap=float(sidecars["p4"]["synthetic_capability_ap"]),
        p5_synthetic_ap=float(sidecars["p5"]["synthetic_capability_ap"]),
        formal_gate_metrics=formal_head_gate_metrics,
    )

    payload: dict[str, Any] = {
        "schema_version": "l89-clean-promotion-chain-v1",
        "protocol_id": "l89-clean-train-only-replicate-v1",
        "p5_qualification": qualification,
        "sidecar_capability_evidence": sidecars,
        "formal_inner_development_only": True,
        "test_or_sealed_or_holdout_or_outer_read": False,
        "outer_evaluation_eligible": False,
        "old_checkpoint_reused": False,
        "claim_boundary": (
            "event-disjoint train-only inner replicate from the same source "
            "training cohort; not outer confirmation and not SOTA evidence"
        ),
        "input_provenance": {
            "head_comparison": {
                "path": str(head_comparison_path),
                "sha256": cache.sha256_file(head_comparison_path),
            },
            "fixed_ensemble_result": {
                "path": str(fixed_result_path),
                "sha256": cache.sha256_file(fixed_result_path),
            },
            "predictions": {
                name: {"path": str(path), "sha256": cache.sha256_file(path)}
                for name, path in prediction_paths.items()
            },
            "d1_summaries": d1_summary_provenance,
        },
        "validated_upstream_receipts": {
            "head_comparison_matching": dict(head_payload["matching"]),
            "head_prediction_recomputation": head_prediction_audit,
            "fresh_head_formal_gate_metrics": {
                arm: {
                    "metrics": dict(metrics),
                    "sha256": cache.sha256_bytes(
                        cache.canonical_json_bytes(metrics)
                    ),
                }
                for arm, metrics in formal_head_gate_metrics.items()
            },
            "fixed_ensemble_p0_probability_replay": (
                fixed_p0_replay_receipt
            ),
            "fixed_ensemble_arithmetic": dict(fixed_payload["arithmetic"]),
            "fixed_ensemble_bootstrap": dict(fixed_payload["bootstrap"]),
            "d1_formal_validation": d1_formal_validation,
        },
        "d1_mechanism_diagnostics": d1_diagnostics,
    }
    if formal_context is not None:
        payload["formal_context"] = formal_context
    if qualification["P5_qualified"]:
        promotion_gate = evaluate_promotion_gate(
            point_metrics=primary_metrics,
            bootstrap=primary_bootstrap,
        )
        payload.update(
            {
                "audit_type": "P5+D1 primary promotion audit",
                "promotion_eligible": True,
                "promotion": bool(promotion_gate["promotion"]),
                "promotion_gate": promotion_gate,
                "primary_candidate": {
                    "point_metrics": primary_metrics,
                    "paired_event_cluster_bootstrap": primary_bootstrap,
                },
                "fallback_triggered": False,
                "final_decision": (
                    "promote_train_only_mechanism"
                    if promotion_gate["promotion"]
                    else "do_not_promote"
                ),
            }
        )
    else:
        fallback, fallback_frame = fallback_audit(
            frames=frames,
            d1_diagnostics=d1_diagnostics,
        )
        payload.update(
            {
                "audit_type": "P0+D1 mechanism-only fallback",
                "promotion_eligible": False,
                "promotion": False,
                "promotion_gate": {
                    "evaluated": False,
                    "reason": "P5_qualified=false",
                },
                "fallback_triggered": True,
                "fallback": fallback,
                "final_decision": "do_not_promote_p5_unqualified",
            }
        )

    output_dir.mkdir(parents=True)
    if payload["fallback_triggered"]:
        fallback_path = output_dir / "fixed_p0_mean_d1_predictions.csv"
        cache.atomic_csv_write(fallback_path, fallback_frame)
        payload["fallback"]["prediction_output"] = {
            "path": str(fallback_path),
            "sha256": cache.sha256_file(fallback_path),
        }
    cache.atomic_json_write(output_dir / "RESULT.json", payload)
    cache.atomic_json_write(output_dir / "PROMOTION_DECISION.json", payload)
    (output_dir / "RESULT.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p4-sidecar-summary", required=True)
    parser.add_argument("--p4-sidecar-checkpoint", required=True)
    parser.add_argument("--p5-sidecar-summary", required=True)
    parser.add_argument("--p5-sidecar-checkpoint", required=True)
    parser.add_argument("--head-comparison", required=True)
    parser.add_argument("--fixed-ensemble-result", required=True)
    parser.add_argument("--p0-predictions", required=True)
    parser.add_argument(
        "--p0-head-predictions",
        help=(
            "Fresh matched-head P0 prediction CSV; defaults to the canonical "
            "P0 CSV beside comparison.json."
        ),
    )
    parser.add_argument("--p4-predictions", required=True)
    parser.add_argument("--p5-predictions", required=True)
    parser.add_argument(
        "--d1-template",
        required=True,
        help="D1 prediction path template containing {seed}.",
    )
    parser.add_argument(
        "--d1-summary-template",
        help=(
            "Optional per-seed summary path template containing {seed}; "
            "fallback defaults to summary.json beside each D1 prediction."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--split-receipt", required=True)
    parser.add_argument("--model-ledger-final", required=True)
    parser.add_argument("--base-weights", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
