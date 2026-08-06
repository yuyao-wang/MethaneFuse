#!/usr/bin/env python3
"""Development-only R4 availability and sensor-history mechanism audit.

This audit never selects a threshold, weight, checkpoint, or hyperparameter.
It consumes the frozen four-seed R4 aggregate, its immutable per-seed
checkpoints/predictions, and the development feature cache. P0 and R4 retain
their already selected global development thresholds.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.tempo_20260728.tempo_legacy360_global import (
    SENSOR_ORDER,
    TempoGlobalHead,
    _features_for_mode,
    _logit,
    atomic_json,
    guard_development_path,
    load_feature_cache,
    predict_tempo,
    sha256_file,
)


SCRIPT_VERSION = "tempo-r4-mechanism-availability-audit-v1"
DEFAULT_AGGREGATE = Path(
    "/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1"
    "/final_r4_4seed_v5/final_aggregate.json"
)
DEFAULT_RUN_ROOT = DEFAULT_AGGREGATE.parent
DEFAULT_DEV_CACHE = Path(
    "/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5"
    "/features/dev_universal_s2hybrid.pt"
)
DEFAULT_OUTPUT = Path(
    "/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1"
    "/mechanism_availability_audit_v6"
)


def history_availability_category(
    valid_mask: np.ndarray,
    sensor_index: int,
) -> np.ndarray:
    """Return exclusive availability categories for one sensor."""

    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 3 or valid.shape[1:] != (4, 3):
        raise ValueError("valid_mask must have shape [N,4,3].")
    current = valid[:, sensor_index, 0]
    short = valid[:, sensor_index, 1]
    long = valid[:, sensor_index, 2]
    if np.any((short | long) & ~current):
        raise RuntimeError("Historical token is valid without current token.")
    output = np.full(len(valid), "absent", dtype=object)
    output[current & ~short & ~long] = "t0_only"
    output[current & short & ~long] = "short_only"
    output[current & ~short & long] = "long_only"
    output[current & short & long] = "both"
    return output


def shuffled_sensor_history_subset(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    sensor_index: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle one sensor's valid history marginals and return affected rows."""

    if features.ndim != 4 or valid_mask.shape != features.shape[:3]:
        raise ValueError("Malformed feature/mask tensors.")
    if not 0 <= sensor_index < features.shape[1]:
        raise ValueError("sensor_index is out of range.")
    affected = torch.nonzero(
        valid_mask[:, sensor_index, 1:].any(dim=1)
    ).flatten()
    subset = features[affected].clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for role in range(1, features.shape[2]):
        local = torch.nonzero(
            valid_mask[affected, sensor_index, role]
        ).flatten()
        if local.numel() < 2:
            continue
        donor_local = local[
            torch.randperm(local.numel(), generator=generator)
        ]
        donor_global = affected[donor_local]
        subset[local, sensor_index, role] = features[
            donor_global, sensor_index, role
        ]
    return affected, subset


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _metric_delta(
    candidate: float | None,
    reference: float | None,
) -> float | None:
    if candidate is None or reference is None:
        return None
    return float(candidate - reference)


def canonical_all_negative_audit(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    event_ids: np.ndarray,
    threshold: float,
    canonical_all_negative_events: set[str],
) -> dict[str, Any]:
    """Audit only events known to be all-negative in the full development set."""

    target = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    events = np.asarray(event_ids, dtype=str)
    in_canonical_negative = np.asarray(
        [event in canonical_all_negative_events for event in events],
        dtype=bool,
    )
    negative_events = events[in_canonical_negative]
    negative_probability = probability[in_canonical_negative]
    hard_fp = negative_probability >= float(threshold)
    represented = sorted(set(negative_events.tolist()))
    group_events = sorted(set(events.tolist()))
    if represented:
        frame = pd.DataFrame(
            {
                "event_id": negative_events,
                "hard_fp": hard_fp.astype(np.float64),
            }
        )
        per_event = frame.groupby("event_id", sort=True)["hard_fp"].agg(
            ["mean", "max", "size"]
        )
        fp_event_count = int(per_event["max"].sum())
        equal_event_fp_density = float(per_event["mean"].mean())
    else:
        fp_event_count = 0
        equal_event_fp_density = None
    return {
        "canonical_all_negative_events_represented": int(len(represented)),
        "canonical_all_negative_event_fraction_of_group_events": _safe_divide(
            len(represented), len(group_events)
        ),
        "canonical_all_negative_rows": int(in_canonical_negative.sum()),
        "canonical_all_negative_row_fraction_of_group": _safe_divide(
            int(in_canonical_negative.sum()), len(target)
        ),
        "hard_fp_event_count": fp_event_count,
        "hard_fp_event_density": _safe_divide(
            fp_event_count, len(represented)
        ),
        "hard_fp_rows": int(hard_fp.sum()),
        "hard_fp_row_density": _safe_divide(
            int(hard_fp.sum()), int(in_canonical_negative.sum())
        ),
        "equal_event_weight_fp_density": equal_event_fp_density,
    }


def fixed_model_group_metrics(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    event_ids: np.ndarray,
    threshold: float,
    canonical_all_negative_events: set[str],
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    prediction = probability >= float(threshold)
    positive = target == 1
    negative = ~positive
    tp = int(np.sum(positive & prediction))
    fp = int(np.sum(negative & prediction))
    fn = int(np.sum(positive & ~prediction))
    tn = int(np.sum(negative & ~prediction))
    binary_f1 = float(
        f1_score(target, prediction, zero_division=0)
    )
    macro_f1 = float(
        f1_score(
            target,
            prediction,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )
    ap = (
        float(average_precision_score(target, probability))
        if positive.any()
        else None
    )
    auc = (
        float(roc_auc_score(target, probability))
        if positive.any() and negative.any()
        else None
    )
    return {
        "threshold": float(threshold),
        "binary_f1": binary_f1,
        "macro_f1": macro_f1,
        "ap": ap,
        "auc": auc,
        "confusion": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "false_positive_rows": fp,
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "canonical_all_negative": canonical_all_negative_audit(
            labels=target,
            probabilities=probability,
            event_ids=event_ids,
            threshold=threshold,
            canonical_all_negative_events=canonical_all_negative_events,
        ),
    }


def group_transition_audit(
    *,
    labels: np.ndarray,
    event_ids: np.ndarray,
    p0_probability: np.ndarray,
    r4_probability: np.ndarray,
    p0_threshold: float,
    r4_threshold: float,
    canonical_all_negative_events: set[str],
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    events = np.asarray(event_ids, dtype=str)
    p0_prediction = np.asarray(p0_probability) >= float(p0_threshold)
    r4_prediction = np.asarray(r4_probability) >= float(r4_threshold)
    positive = target == 1
    negative = ~positive
    p0 = fixed_model_group_metrics(
        labels=target,
        probabilities=p0_probability,
        event_ids=events,
        threshold=p0_threshold,
        canonical_all_negative_events=canonical_all_negative_events,
    )
    r4 = fixed_model_group_metrics(
        labels=target,
        probabilities=r4_probability,
        event_ids=events,
        threshold=r4_threshold,
        canonical_all_negative_events=canonical_all_negative_events,
    )
    p0_null = p0["canonical_all_negative"]
    r4_null = r4["canonical_all_negative"]
    return {
        "rows": int(len(target)),
        "events": int(len(set(events.tolist()))),
        "positives": int(positive.sum()),
        "negatives": int(negative.sum()),
        "p0": p0,
        "r4": r4,
        "r4_minus_p0": {
            "binary_f1": _metric_delta(r4["binary_f1"], p0["binary_f1"]),
            "macro_f1": _metric_delta(r4["macro_f1"], p0["macro_f1"]),
            "ap": _metric_delta(r4["ap"], p0["ap"]),
            "auc": _metric_delta(r4["auc"], p0["auc"]),
            "false_positive_rows": int(
                r4["false_positive_rows"] - p0["false_positive_rows"]
            ),
            "canonical_all_negative_hard_fp_events": int(
                r4_null["hard_fp_event_count"]
                - p0_null["hard_fp_event_count"]
            ),
            "canonical_all_negative_hard_fp_rows": int(
                r4_null["hard_fp_rows"] - p0_null["hard_fp_rows"]
            ),
            "canonical_all_negative_equal_event_fp_density": _metric_delta(
                r4_null["equal_event_weight_fp_density"],
                p0_null["equal_event_weight_fp_density"],
            ),
        },
        "decision_transitions": {
            "false_negative_corrected_to_true_positive": int(
                np.sum(positive & ~p0_prediction & r4_prediction)
            ),
            "true_positive_lost_to_false_negative": int(
                np.sum(positive & p0_prediction & ~r4_prediction)
            ),
            "false_positive_corrected_to_true_negative": int(
                np.sum(negative & p0_prediction & ~r4_prediction)
            ),
            "new_false_positive_from_true_negative": int(
                np.sum(negative & ~p0_prediction & r4_prediction)
            ),
            "net_true_positive_rows": int(
                np.sum(positive & r4_prediction)
                - np.sum(positive & p0_prediction)
            ),
            "net_false_positive_rows": int(
                np.sum(negative & r4_prediction)
                - np.sum(negative & p0_prediction)
            ),
        },
    }


def audit_partition(
    *,
    group_labels: np.ndarray,
    labels: np.ndarray,
    event_ids: np.ndarray,
    p0_probability: np.ndarray,
    r4_probability: np.ndarray,
    p0_threshold: float,
    r4_threshold: float,
    canonical_all_negative_events: set[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value in sorted(
        set(group_labels.tolist()),
        key=lambda item: (-int(np.sum(group_labels == item)), str(item)),
    ):
        mask = group_labels == value
        output[str(value)] = group_transition_audit(
            labels=labels[mask],
            event_ids=event_ids[mask],
            p0_probability=p0_probability[mask],
            r4_probability=r4_probability[mask],
            p0_threshold=p0_threshold,
            r4_threshold=r4_threshold,
            canonical_all_negative_events=canonical_all_negative_events,
        )
    if sum(group["rows"] for group in output.values()) != len(labels):
        raise RuntimeError("Availability partition does not cover development.")
    return output


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_frozen_seed_models(
    *,
    run_root: Path,
    seeds: Sequence[int],
    identity: pd.DataFrame,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = run_root / f"r4_seed{int(seed)}"
        result_path = run_dir / "result.json"
        prediction_path = run_dir / "dev_predictions_best.csv"
        checkpoint_path = run_dir / "checkpoint_best.pth"
        for path, role in (
            (result_path, "R4 development result"),
            (prediction_path, "R4 development prediction"),
            (checkpoint_path, "R4 development checkpoint"),
        ):
            guard_development_path(path, role)
        result = _load_json(result_path)
        prediction = pd.read_csv(prediction_path)
        identity_columns = ["id", "event_id", "label"]
        prediction_identity = prediction[identity_columns].copy()
        prediction_identity["id"] = prediction_identity["id"].astype(str)
        prediction_identity["event_id"] = prediction_identity[
            "event_id"
        ].astype(str)
        prediction_identity["label"] = prediction_identity["label"].astype(
            np.int64
        )
        if not prediction_identity.equals(identity[identity_columns]):
            raise RuntimeError(f"Seed {seed} prediction identities changed.")
        if result["arm"] != "r4" or int(result["seed"]) != int(seed):
            raise RuntimeError(f"Seed {seed} result metadata changed.")
        if bool(result["protocol"]["test_or_sealed_read"]):
            raise RuntimeError(f"Seed {seed} reports test/sealed read.")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        config = checkpoint["model_config"]
        model = TempoGlobalHead(
            int(config["feature_dim"]),
            num_sensors=int(config["num_sensors"]),
            model_dim=int(config["model_dim"]),
            residual_cap=float(config["residual_cap"]),
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        probability = prediction["probability"].to_numpy(dtype=np.float64)
        residual = prediction["residual"].to_numpy(dtype=np.float64)
        recovered_base_logit = _logit(probability) - residual
        records.append(
            {
                "seed": int(seed),
                "model": model,
                "base_fused_logits": torch.from_numpy(
                    recovered_base_logit.astype(np.float32)
                ),
                "source_probability": probability,
                "result_path": str(result_path),
                "result_sha256": sha256_file(result_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )
    return records


def shuffle_transition_audit(
    *,
    labels: np.ndarray,
    event_ids: np.ndarray,
    unshuffled_probability: np.ndarray,
    shuffled_probability: np.ndarray,
    threshold: float,
    canonical_all_negative_events: set[str],
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    unshuffled_prediction = unshuffled_probability >= float(threshold)
    shuffled_prediction = shuffled_probability >= float(threshold)
    positive = target == 1
    negative = ~positive
    baseline = fixed_model_group_metrics(
        labels=target,
        probabilities=unshuffled_probability,
        event_ids=event_ids,
        threshold=threshold,
        canonical_all_negative_events=canonical_all_negative_events,
    )
    shuffled = fixed_model_group_metrics(
        labels=target,
        probabilities=shuffled_probability,
        event_ids=event_ids,
        threshold=threshold,
        canonical_all_negative_events=canonical_all_negative_events,
    )
    base_null = baseline["canonical_all_negative"]
    shuffle_null = shuffled["canonical_all_negative"]
    return {
        "unshuffled": baseline,
        "shuffled": shuffled,
        "shuffled_minus_unshuffled": {
            "binary_f1": _metric_delta(
                shuffled["binary_f1"], baseline["binary_f1"]
            ),
            "macro_f1": _metric_delta(
                shuffled["macro_f1"], baseline["macro_f1"]
            ),
            "ap": _metric_delta(shuffled["ap"], baseline["ap"]),
            "auc": _metric_delta(shuffled["auc"], baseline["auc"]),
            "false_positive_rows": int(
                shuffled["false_positive_rows"]
                - baseline["false_positive_rows"]
            ),
            "canonical_all_negative_hard_fp_events": int(
                shuffle_null["hard_fp_event_count"]
                - base_null["hard_fp_event_count"]
            ),
            "canonical_all_negative_hard_fp_rows": int(
                shuffle_null["hard_fp_rows"] - base_null["hard_fp_rows"]
            ),
        },
        "decision_transitions_after_shuffle": {
            "true_positive_lost": int(
                np.sum(positive & unshuffled_prediction & ~shuffled_prediction)
            ),
            "true_positive_gained": int(
                np.sum(positive & ~unshuffled_prediction & shuffled_prediction)
            ),
            "false_positive_added": int(
                np.sum(negative & ~unshuffled_prediction & shuffled_prediction)
            ),
            "false_positive_removed": int(
                np.sum(negative & unshuffled_prediction & ~shuffled_prediction)
            ),
            "total_threshold_crossings": int(
                np.sum(unshuffled_prediction != shuffled_prediction)
            ),
        },
    }


def run_per_sensor_shuffle(
    *,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    labels: np.ndarray,
    event_ids: np.ndarray,
    ensemble_probability: np.ndarray,
    ensemble_threshold: float,
    canonical_all_negative_events: set[str],
    seed_models: list[dict[str, Any]],
    shuffle_seed: int,
    eval_batch_size: int,
) -> tuple[dict[str, Any], float]:
    output: dict[str, Any] = {}
    replay_error = 0.0
    device = torch.device("cpu")
    for sensor_index, sensor in enumerate(SENSOR_ORDER):
        permutation_seed = int(shuffle_seed + 1009 * sensor_index)
        affected, shuffled_features = shuffled_sensor_history_subset(
            features,
            valid_mask,
            sensor_index=sensor_index,
            seed=permutation_seed,
        )
        affected_numpy = affected.numpy()
        shuffled_seed_logits: list[np.ndarray] = []
        for record in seed_models:
            sample = affected[: min(64, len(affected))]
            if len(sample):
                replay_probability, _ = predict_tempo(
                    record["model"],
                    features[sample],
                    valid_mask[sample],
                    record["base_fused_logits"][sample],
                    torch.zeros(len(sample), 4),
                    arm="r4",
                    batch_size=eval_batch_size,
                    device=device,
                )
                expected = record["source_probability"][sample.numpy()]
                replay_error = max(
                    replay_error,
                    float(np.max(np.abs(replay_probability - expected))),
                )
            if len(affected):
                _, shuffled_residual = predict_tempo(
                    record["model"],
                    shuffled_features,
                    valid_mask[affected],
                    record["base_fused_logits"][affected],
                    torch.zeros(len(affected), 4),
                    arm="r4",
                    batch_size=eval_batch_size,
                    device=device,
                )
                seed_logit = (
                    record["base_fused_logits"][affected].numpy()
                    + shuffled_residual
                )
            else:
                seed_logit = np.empty(0, dtype=np.float64)
            shuffled_seed_logits.append(seed_logit)
        shuffled_probability = ensemble_probability.copy()
        if len(affected):
            mean_logit = np.mean(
                np.stack(shuffled_seed_logits, axis=0), axis=0
            )
            shuffled_probability[affected_numpy] = 1.0 / (
                1.0 + np.exp(-mean_logit)
            )
        full_audit = shuffle_transition_audit(
            labels=labels,
            event_ids=event_ids,
            unshuffled_probability=ensemble_probability,
            shuffled_probability=shuffled_probability,
            threshold=ensemble_threshold,
            canonical_all_negative_events=canonical_all_negative_events,
        )
        eligible_audit = shuffle_transition_audit(
            labels=labels[affected_numpy],
            event_ids=event_ids[affected_numpy],
            unshuffled_probability=ensemble_probability[affected_numpy],
            shuffled_probability=shuffled_probability[affected_numpy],
            threshold=ensemble_threshold,
            canonical_all_negative_events=canonical_all_negative_events,
        )
        change = np.abs(
            shuffled_probability[affected_numpy]
            - ensemble_probability[affected_numpy]
        )
        output[sensor] = {
            "sensor_index": int(sensor_index),
            "permutation_seed": permutation_seed,
            "eligible_rows": int(len(affected)),
            "eligible_events": int(
                len(set(event_ids[affected_numpy].tolist()))
            ),
            "valid_short_history_rows": int(
                valid_mask[:, sensor_index, 1].sum()
            ),
            "valid_long_history_rows": int(
                valid_mask[:, sensor_index, 2].sum()
            ),
            "mean_absolute_probability_change_on_eligible_rows": (
                float(change.mean()) if len(change) else 0.0
            ),
            "max_absolute_probability_change_on_eligible_rows": (
                float(change.max()) if len(change) else 0.0
            ),
            "full_development": full_audit,
            "eligible_subset": eligible_audit,
            "invariants": {
                "only_named_sensor_history_modified": True,
                "t0_unchanged": True,
                "valid_mask_unchanged": True,
                "other_sensor_histories_unchanged": True,
                "history_marginals_preserved": True,
                "global_threshold_refit": False,
                "ensemble_weights_refit": False,
            },
        }
        del shuffled_features
    return output, replay_error


def _format_float(value: float | None, digits: int = 5) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def availability_table(
    title: str,
    groups: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Group | Rows | Events | P0 F1 | R4 F1 | ΔF1 | P0 AP | R4 AP | "
        "ΔAP | FP rows P0→R4 | new TP | new FP | removed FP | "
        "all-neg FP events P0→R4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|",
    ]
    for name, group in groups.items():
        transition = group["decision_transitions"]
        p0_null = group["p0"]["canonical_all_negative"]
        r4_null = group["r4"]["canonical_all_negative"]
        lines.append(
            f"| {name} | {group['rows']} | {group['events']} | "
            f"{_format_float(group['p0']['binary_f1'])} | "
            f"{_format_float(group['r4']['binary_f1'])} | "
            f"{_format_float(group['r4_minus_p0']['binary_f1'])} | "
            f"{_format_float(group['p0']['ap'])} | "
            f"{_format_float(group['r4']['ap'])} | "
            f"{_format_float(group['r4_minus_p0']['ap'])} | "
            f"{group['p0']['false_positive_rows']}→"
            f"{group['r4']['false_positive_rows']} | "
            f"{transition['false_negative_corrected_to_true_positive']} | "
            f"{transition['new_false_positive_from_true_negative']} | "
            f"{transition['false_positive_corrected_to_true_negative']} | "
            f"{p0_null['hard_fp_event_count']}→"
            f"{r4_null['hard_fp_event_count']} |"
        )
    lines.append("")
    return lines


def write_markdown(path: Path, audit: Mapping[str, Any]) -> None:
    overall = audit["availability"]["overall"]
    lines = [
        "# R4 mechanism and availability audit",
        "",
        "Development only. No threshold, ensemble weight, checkpoint, or "
        "hyperparameter was refit; no test/sealed artifact was read.",
        "",
        (
            f"Frozen global thresholds: P0 "
            f"`{audit['thresholds']['p0']:.8f}`, R4 "
            f"`{audit['thresholds']['r4_ensemble']:.8f}`."
        ),
        "",
        "## Overall fixed-threshold result",
        "",
        "| Model | Rows | Events | F1 | Macro F1 | AP | FP rows | "
        "all-negative FP events/rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| P0 | {overall['rows']} | {overall['events']} | "
            f"{overall['p0']['binary_f1']:.5f} | "
            f"{overall['p0']['macro_f1']:.5f} | "
            f"{overall['p0']['ap']:.5f} | "
            f"{overall['p0']['false_positive_rows']} | "
            f"{overall['p0']['canonical_all_negative']['hard_fp_event_count']}"
            "/"
            f"{overall['p0']['canonical_all_negative']['hard_fp_rows']} |"
        ),
        (
            f"| R4 four-seed | {overall['rows']} | {overall['events']} | "
            f"{overall['r4']['binary_f1']:.5f} | "
            f"{overall['r4']['macro_f1']:.5f} | "
            f"{overall['r4']['ap']:.5f} | "
            f"{overall['r4']['false_positive_rows']} | "
            f"{overall['r4']['canonical_all_negative']['hard_fp_event_count']}"
            "/"
            f"{overall['r4']['canonical_all_negative']['hard_fp_rows']} |"
        ),
        "",
    ]
    lines.extend(
        availability_table(
            "Exclusive sensor-availability signatures",
            audit["availability"]["exclusive_signature"],
        )
    )
    lines.extend(
        availability_table(
            "Available sensor count",
            audit["availability"]["sensor_count"],
        )
    )
    for sensor in SENSOR_ORDER:
        lines.extend(
            availability_table(
                f"{sensor.upper()} history availability",
                audit["availability"]["sensor_history_availability"][sensor],
            )
        )
    lines.extend(
        availability_table(
            "S2 present versus absent",
            audit["availability"]["s2_presence"],
        )
    )
    missing = audit["missing_sensor_stress"]
    lines.extend(
        [
            "## Counterfactual sensor-drop stress",
            "",
            f"Status: **{missing['status']}**.",
            "",
            missing["reason"],
            "",
            "The natural partial-observation comparison above remains valid "
            "because those rows were encoded and scored with their genuinely "
            "available sensor subsets.",
            "",
        ]
    )
    lines.extend(
        [
            "## One-sensor-at-a-time history shuffle",
            "",
            "Only the named sensor's valid history tokens are permuted. "
            "Negative deltas mean matched history was useful.",
            "",
            "| Sensor | eligible rows/events | ΔF1 | Δmacro | ΔAP | ΔAUC | "
            "mean |Δp| | threshold flips | TP lost/gained | FP added/removed | "
            "Δ all-neg FP events/rows |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sensor in SENSOR_ORDER:
        record = audit["per_sensor_history_shuffle"][sensor]
        delta = record["full_development"]["shuffled_minus_unshuffled"]
        transition = record["full_development"][
            "decision_transitions_after_shuffle"
        ]
        lines.append(
            f"| {sensor.upper()} | {record['eligible_rows']}/"
            f"{record['eligible_events']} | "
            f"{_format_float(delta['binary_f1'], 6)} | "
            f"{_format_float(delta['macro_f1'], 6)} | "
            f"{_format_float(delta['ap'], 6)} | "
            f"{_format_float(delta['auc'], 6)} | "
            f"{record['mean_absolute_probability_change_on_eligible_rows']:.6f} | "
            f"{transition['total_threshold_crossings']} | "
            f"{transition['true_positive_lost']}/"
            f"{transition['true_positive_gained']} | "
            f"{transition['false_positive_added']}/"
            f"{transition['false_positive_removed']} | "
            f"{delta['canonical_all_negative_hard_fp_events']:+d}/"
            f"{delta['canonical_all_negative_hard_fp_rows']:+d} |"
        )
    lines.extend(
        [
            "",
            "All subgroup AP values use the original continuous scores. "
            "All F1/FP/transition values use the two frozen global thresholds.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def command_audit(args: argparse.Namespace) -> None:
    aggregate_path = Path(args.aggregate).expanduser().absolute()
    run_root = Path(args.run_root).expanduser().absolute()
    dev_cache_path = Path(args.dev_cache).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    for path, role in (
        (aggregate_path, "four-seed development aggregate"),
        (run_root, "four-seed development run root"),
        (dev_cache_path, "development feature cache"),
        (output_dir, "mechanism audit output"),
    ):
        guard_development_path(path, role)
    output_path = output_dir / "r4_mechanism_availability_audit.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing existing audit: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(int(args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    aggregate = _load_json(aggregate_path)
    if bool(aggregate["test_or_sealed_read"]):
        raise RuntimeError("Aggregate reports test/sealed read.")
    seeds = [int(seed) for seed in aggregate["seeds"]]
    if len(seeds) != 4:
        raise RuntimeError("Mechanism audit requires the frozen four-seed R4.")
    ensemble_path = aggregate_path.parent / "final_ensemble_predictions.csv"
    guard_development_path(ensemble_path, "development ensemble predictions")
    ensemble = pd.read_csv(ensemble_path)
    cache = load_feature_cache(dev_cache_path, ("dev", "evaluation"))
    if bool(cache.get("sealed_test_read", False)):
        raise RuntimeError("Development cache reports sealed-test read.")
    identity = pd.DataFrame(
        {
            "id": [str(value) for value in cache["ids"]],
            "event_id": [str(value) for value in cache["event_ids"]],
            "label": cache["labels"].numpy().astype(np.int64),
        }
    )
    ensemble_identity = ensemble[["id", "event_id", "label"]].copy()
    ensemble_identity["id"] = ensemble_identity["id"].astype(str)
    ensemble_identity["event_id"] = ensemble_identity["event_id"].astype(str)
    ensemble_identity["label"] = ensemble_identity["label"].astype(np.int64)
    if not ensemble_identity.equals(identity):
        raise RuntimeError("Aggregate prediction/cache identities changed.")

    labels = identity["label"].to_numpy(dtype=np.int64)
    event_ids = identity["event_id"].to_numpy(dtype=str)
    signatures = np.asarray(
        [str(value) for value in cache["availability_signatures"]],
        dtype=object,
    )
    valid_mask = cache["valid_mask"].bool()
    valid_numpy = valid_mask.numpy()
    features = _features_for_mode(cache, "universal")
    p0_probability = ensemble["p0_probability"].to_numpy(dtype=np.float64)
    r4_probability = ensemble["ensemble_probability"].to_numpy(
        dtype=np.float64
    )
    p0_threshold = float(
        aggregate["p0"]["metrics"]["best_binary_f1_threshold"]
    )
    r4_threshold = float(
        aggregate["logit_ensemble"]["metrics"]["best_binary_f1_threshold"]
    )
    event_frame = pd.DataFrame(
        {"event_id": event_ids, "label": labels}
    )
    canonical_all_negative_events = set(
        event_frame.groupby("event_id", sort=True)["label"]
        .max()
        .loc[lambda value: value.eq(0)]
        .index.astype(str)
    )
    overall = group_transition_audit(
        labels=labels,
        event_ids=event_ids,
        p0_probability=p0_probability,
        r4_probability=r4_probability,
        p0_threshold=p0_threshold,
        r4_threshold=r4_threshold,
        canonical_all_negative_events=canonical_all_negative_events,
    )
    signature_sensor_count = np.asarray(
        [len(signature.split("+")) for signature in signatures],
        dtype=np.int64,
    )
    mask_sensor_count = valid_numpy[:, :, 0].sum(axis=1)
    if not np.array_equal(signature_sensor_count, mask_sensor_count):
        raise RuntimeError("Signature and t0 sensor counts differ.")
    sensor_history = {
        sensor: audit_partition(
            group_labels=history_availability_category(
                valid_numpy, sensor_index
            ),
            labels=labels,
            event_ids=event_ids,
            p0_probability=p0_probability,
            r4_probability=r4_probability,
            p0_threshold=p0_threshold,
            r4_threshold=r4_threshold,
            canonical_all_negative_events=canonical_all_negative_events,
        )
        for sensor_index, sensor in enumerate(SENSOR_ORDER)
    }
    availability = {
        "overall": overall,
        "exclusive_signature": audit_partition(
            group_labels=signatures,
            labels=labels,
            event_ids=event_ids,
            p0_probability=p0_probability,
            r4_probability=r4_probability,
            p0_threshold=p0_threshold,
            r4_threshold=r4_threshold,
            canonical_all_negative_events=canonical_all_negative_events,
        ),
        "sensor_count": audit_partition(
            group_labels=mask_sensor_count.astype(object),
            labels=labels,
            event_ids=event_ids,
            p0_probability=p0_probability,
            r4_probability=r4_probability,
            p0_threshold=p0_threshold,
            r4_threshold=r4_threshold,
            canonical_all_negative_events=canonical_all_negative_events,
        ),
        "sensor_history_availability": sensor_history,
        "s2_presence": audit_partition(
            group_labels=np.where(
                valid_numpy[:, 0, 0], "present", "absent"
            ).astype(object),
            labels=labels,
            event_ids=event_ids,
            p0_probability=p0_probability,
            r4_probability=r4_probability,
            p0_threshold=p0_threshold,
            r4_threshold=r4_threshold,
            canonical_all_negative_events=canonical_all_negative_events,
        ),
    }
    seed_models = load_frozen_seed_models(
        run_root=run_root,
        seeds=seeds,
        identity=identity,
    )
    sensor_shuffle, replay_error = run_per_sensor_shuffle(
        features=features,
        valid_mask=valid_mask,
        labels=labels,
        event_ids=event_ids,
        ensemble_probability=r4_probability,
        ensemble_threshold=r4_threshold,
        canonical_all_negative_events=canonical_all_negative_events,
        seed_models=seed_models,
        shuffle_seed=int(args.shuffle_seed),
        eval_batch_size=int(args.eval_batch_size),
    )
    if replay_error > 5e-4:
        raise RuntimeError(
            f"Frozen checkpoint replay error is too large: {replay_error}"
        )
    audit = {
        "schema_version": "tempo-r4-mechanism-availability-audit-v1",
        "script_version": SCRIPT_VERSION,
        "scope": "legacy360 development only",
        "inputs": {
            "aggregate": str(aggregate_path),
            "aggregate_sha256": sha256_file(aggregate_path),
            "ensemble_predictions": str(ensemble_path),
            "ensemble_predictions_sha256": sha256_file(ensemble_path),
            "dev_cache": str(dev_cache_path),
            "dev_cache_sha256": sha256_file(dev_cache_path),
            "run_root": str(run_root),
            "seeds": seeds,
            "seed_artifacts": [
                {
                    key: value
                    for key, value in record.items()
                    if key
                    not in {
                        "model",
                        "base_fused_logits",
                        "source_probability",
                    }
                }
                for record in seed_models
            ],
        },
        "thresholds": {
            "p0": p0_threshold,
            "r4_ensemble": r4_threshold,
            "source": "frozen global development aggregate",
            "refit_for_subgroups": False,
            "refit_for_shuffle": False,
        },
        "canonical_event_definition": {
            "events": int(len(set(event_ids.tolist()))),
            "all_negative_events": int(len(canonical_all_negative_events)),
            "all_negative_determined_on_full_development_before_grouping": True,
        },
        "availability": availability,
        "missing_sensor_stress": {
            "status": "not_identifiable_end_to_end_from_cached_features",
            "reason": (
                "The frozen P0 shortcut uses a universal row-fusion head over "
                "the elementwise maximum of legacy concatenated-sensor "
                "features. The cache stores only the already fused base logit "
                "and per-role features, not those pre-fusion concatenated "
                "features. Dropping a mask only in P0/R4 adapters would leave "
                "the supposedly dropped sensor inside the base logit and "
                "would be a confounded adapter ablation. Exact counterfactual "
                "sensor dropping therefore requires raw-image re-encoding and "
                "was intentionally not fabricated in this CPU/cache-only audit."
            ),
            "natural_partial_observation_groups_reported": True,
            "adapter_only_drop_reported_as_end_to_end": False,
            "thresholds_refit": False,
            "test_or_sealed_read": False,
        },
        "per_sensor_history_shuffle": sensor_shuffle,
        "guardrails": {
            "test_or_sealed_read": False,
            "device": "cpu",
            "torch_threads": int(args.torch_threads),
            "ensemble_weights_refit": False,
            "thresholds_refit": False,
            "hyperparameters_changed": False,
            "t0_shuffled": False,
            "one_sensor_history_shuffled_at_a_time": True,
            "frozen_checkpoint_replay_max_abs_probability_error": replay_error,
            "frozen_checkpoint_cross_device_replay_tolerance": 5e-4,
            "exclusive_signature_rows_sum": int(
                sum(
                    value["rows"]
                    for value in availability[
                        "exclusive_signature"
                    ].values()
                )
            ),
            "sensor_count_rows_sum": int(
                sum(
                    value["rows"]
                    for value in availability["sensor_count"].values()
                )
            ),
        },
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    atomic_json(output_path, audit)
    write_markdown(output_dir / "R4_MECHANISM_AVAILABILITY_AUDIT.md", audit)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "rows": int(len(labels)),
                "events": int(len(set(event_ids.tolist()))),
                "seeds": seeds,
                "replay_error": replay_error,
                "test_or_sealed_read": False,
            },
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", default=str(DEFAULT_AGGREGATE))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--dev-cache", default=str(DEFAULT_DEV_CACHE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shuffle-seed", type=int, default=20260728)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command_audit(args)


if __name__ == "__main__":
    main()
