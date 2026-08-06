#!/usr/bin/env python3
"""Post-hoc CPU-only event-balanced head audit for frozen RCTP L89 caches.

This runner deliberately changes only the downstream optimization/selection
unit. It never reads test/sealed artifacts and must not be used as confirmatory
evidence because the validation split already motivated this diagnostic.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import l89_ragged_cls_experiment as base


SCRIPT_VERSION = "rctp-l89-event-balanced-head-posthoc-v1"
CLEAN_INNER_SCRIPT_VERSION = "l89-clean-inner-matched-head-v1"
ARMS = ("p0", "p4", "p5")


def artifact_metadata(args: argparse.Namespace) -> dict[str, Any]:
    clean = bool(getattr(args, "clean_inner_replicate", False))
    return {
        "script_version": (
            CLEAN_INNER_SCRIPT_VERSION if clean else SCRIPT_VERSION
        ),
        "scope": (
            "independent train-only clean inner replicate"
            if clean
            else "train/dev-only post-hoc event-balanced head diagnostic"
        ),
        "post_hoc_exploratory": not clean,
        "clean_inner_replicate_exploratory": clean,
    }


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def event_unit_row_weights(event_ids: Sequence[str]) -> np.ndarray:
    """Return positive row weights whose sum is exactly one in every event."""

    weights = base.event_balanced_row_weights(event_ids)
    frame = pd.DataFrame(
        {
            "event_id": [str(value) for value in event_ids],
            "weight": weights,
        }
    )
    totals = frame.groupby("event_id", sort=True)["weight"].sum().to_numpy()
    if not np.allclose(totals, 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("Inverse event-size weights do not total one per event.")
    return weights


def mean_one_event_weights(event_ids: Sequence[str]) -> np.ndarray:
    """Scale unit-event weights to mean one without changing event equality."""

    weights = event_unit_row_weights(event_ids)
    return weights / weights.mean()


def event_balanced_positive_weight(
    labels: np.ndarray, event_ids: Sequence[str]
) -> float:
    target = np.asarray(labels, dtype=np.int64)
    weights = event_unit_row_weights(event_ids)
    positive_mass = float(weights[target == 1].sum())
    negative_mass = float(weights[target == 0].sum())
    if positive_mass <= 0 or negative_mass <= 0:
        raise ValueError("Event-balanced class weighting requires both classes.")
    return negative_mass / positive_mass


def best_weighted_macro_f1_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray,
    *,
    tie_target: float = 0.5,
) -> tuple[float, float]:
    """Exact weighted macro-F1 maximizer over observed score thresholds."""

    target = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if (
        target.ndim != 1
        or probability.shape != target.shape
        or weights.shape != target.shape
        or len(target) == 0
    ):
        raise ValueError("labels, probabilities, and weights must share shape.")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("Macro-F1 threshold selection requires both classes.")
    if not np.isfinite(probability).all() or not np.isfinite(weights).all():
        raise ValueError("Threshold inputs must be finite.")
    if np.any(weights <= 0):
        raise ValueError("Threshold weights must be positive.")

    order = np.argsort(-probability, kind="mergesort")
    scores = probability[order]
    y = target[order]
    w = weights[order]
    cumulative_tp = np.cumsum(w * y)
    cumulative_fp = np.cumsum(w * (1 - y))
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    tp = cumulative_tp[ends]
    fp = cumulative_fp[ends]
    total_positive = float(np.sum(w * y))
    total_negative = float(np.sum(w * (1 - y)))
    fn = total_positive - tp
    tn = total_negative - fp
    positive_f1 = np.divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=(2.0 * tp + fp + fn) > 0,
    )
    negative_f1 = np.divide(
        2.0 * tn,
        2.0 * tn + fp + fn,
        out=np.zeros_like(tn),
        where=(2.0 * tn + fp + fn) > 0,
    )
    macro_f1 = (positive_f1 + negative_f1) / 2.0
    best_score = float(macro_f1.max())
    tied = np.flatnonzero(np.isclose(macro_f1, best_score, atol=1e-12, rtol=0.0))
    thresholds = scores[ends]
    distances = np.abs(thresholds[tied] - float(tie_target))
    closest = tied[np.flatnonzero(distances == distances.min())]
    chosen = int(closest[np.argmax(thresholds[closest])])
    return float(thresholds[chosen]), best_score


def build_initial_state(
    *,
    feature_dim: int,
    num_roles: int,
    model_dim: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    periods_days: Sequence[float],
    t0_index: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    set_seed(seed)
    model = base.RaggedCurrentQueryHead(
        feature_dim=feature_dim,
        num_roles=num_roles,
        model_dim=model_dim,
        num_heads=num_heads,
        depth=2,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        periods_days=periods_days,
        t0_index=t0_index,
    )
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    signature = base.model_parameter_signature(model)
    return state, signature


def instantiate_from_initial_state(
    initial_state: Mapping[str, torch.Tensor],
    *,
    feature_dim: int,
    num_roles: int,
    model_dim: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    periods_days: Sequence[float],
    t0_index: int,
) -> base.RaggedCurrentQueryHead:
    model = base.RaggedCurrentQueryHead(
        feature_dim=feature_dim,
        num_roles=num_roles,
        model_dim=model_dim,
        num_heads=num_heads,
        depth=2,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        periods_days=periods_days,
        t0_index=t0_index,
    )
    model.load_state_dict(initial_state, strict=True)
    return model


def prediction_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    event_ids: Sequence[str],
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    event_weights = event_unit_row_weights(event_ids)
    threshold, selected_macro = best_weighted_macro_f1_threshold(
        target, probability, event_weights
    )
    output: dict[str, Any] = {
        "rows": int(len(target)),
        "events": int(len(set(str(value) for value in event_ids))),
        "row_ap": float(average_precision_score(target, probability)),
        "row_auc": float(roc_auc_score(target, probability)),
        "event_balanced_ap": float(
            average_precision_score(
                target, probability, sample_weight=event_weights
            )
        ),
        "event_balanced_auc": float(
            roc_auc_score(target, probability, sample_weight=event_weights)
        ),
        "event_balanced_selected_threshold": float(threshold),
        "event_balanced_selected_macro_f1": float(selected_macro),
    }
    for name, value in (("fixed_0p5", 0.5), ("selected", threshold)):
        prediction = (probability >= value).astype(np.int64)
        output[f"row_positive_f1_{name}"] = float(
            f1_score(target, prediction, zero_division=0)
        )
        output[f"row_macro_f1_{name}"] = float(
            f1_score(
                target,
                prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )
        output[f"event_balanced_positive_f1_{name}"] = float(
            f1_score(
                target,
                prediction,
                sample_weight=event_weights,
                zero_division=0,
            )
        )
        output[f"event_balanced_macro_f1_{name}"] = float(
            f1_score(
                target,
                prediction,
                labels=[0, 1],
                average="macro",
                sample_weight=event_weights,
                zero_division=0,
            )
        )
    return output


def evaluate(
    model: base.RaggedCurrentQueryHead,
    data: Mapping[str, Any],
    *,
    role_index: torch.Tensor,
    t0_index: int,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for indices in base.fixed_epoch_batches(
            int(data["labels"].shape[0]),
            batch_size=batch_size,
            seed=0,
            epoch=0,
            shuffle=False,
        ):
            features, valid, delta, enable_delta = base.prepare_arm_inputs(
                data["features"][indices],
                data["valid_mask"][indices],
                data["unique_mask"][indices],
                data["delta_days"][indices],
                arm="role_only",
                t0_index=t0_index,
            )
            logits_parts.append(
                model(
                    features,
                    valid,
                    role_index,
                    delta,
                    enable_delta=enable_delta,
                ).cpu()
            )
    logits = torch.cat(logits_parts)
    probability = torch.sigmoid(logits).numpy()
    target = data["labels"].long().numpy()
    metrics = prediction_metrics(target, probability, data["event_ids"])
    weights = torch.from_numpy(
        mean_one_event_weights(data["event_ids"])
    ).to(dtype=logits.dtype)
    pos_weight = torch.tensor(
        event_balanced_positive_weight(target, data["event_ids"]),
        dtype=logits.dtype,
    )
    per_row = F.binary_cross_entropy_with_logits(
        logits,
        data["labels"].to(dtype=logits.dtype),
        pos_weight=pos_weight,
        reduction="none",
    )
    metrics["event_balanced_bce"] = float((per_row * weights).mean())
    return metrics, probability


def train_arm(
    arm: str,
    *,
    train_data: Mapping[str, Any],
    val_data: Mapping[str, Any],
    role_index: torch.Tensor,
    t0_index: int,
    initial_state: Mapping[str, torch.Tensor],
    parameter_signature: Mapping[str, Any],
    args: argparse.Namespace,
    arm_dir: Path,
    cache_audit: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    model = instantiate_from_initial_state(
        initial_state,
        feature_dim=int(train_data["features"].shape[-1]),
        num_roles=int(train_data["features"].shape[1]),
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        periods_days=args.periods_days,
        t0_index=t0_index,
    )
    if base.model_parameter_signature(model) != parameter_signature:
        raise RuntimeError(f"{arm}: parameter signature changed.")
    loaded_sha = base.state_dict_sha256(model.state_dict())
    initial_sha = base.state_dict_sha256(initial_state)
    if loaded_sha != initial_sha:
        raise RuntimeError(f"{arm}: initial state differs from matched prototype.")

    set_seed(args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_labels = train_data["labels"]
    raw_event_weights = event_unit_row_weights(train_data["event_ids"])
    scaled_weights = torch.from_numpy(
        mean_one_event_weights(train_data["event_ids"])
    ).to(dtype=torch.float32)
    pos_weight = torch.tensor(
        event_balanced_positive_weight(
            train_labels.long().numpy(), train_data["event_ids"]
        ),
        dtype=torch.float32,
    )
    event_count = len(set(str(value) for value in train_data["event_ids"]))
    per_event_scaled_total = float(len(train_labels) / event_count)
    weight_audit = {
        "events": int(event_count),
        "rows": int(len(train_labels)),
        "raw_min": float(raw_event_weights.min()),
        "raw_max": float(raw_event_weights.max()),
        "raw_per_event_total": 1.0,
        "scaled_mean": float(scaled_weights.mean()),
        "scaled_per_event_total": per_event_scaled_total,
        "event_balanced_pos_weight": float(pos_weight),
    }

    history: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    best_predictions: Optional[pd.DataFrame] = None
    started = time.monotonic()
    arm_dir.mkdir(parents=True, exist_ok=False)
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = base.fixed_epoch_batches(
            int(train_labels.shape[0]),
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=epoch,
            shuffle=True,
        )
        total_weighted_loss = 0.0
        rows_seen = 0
        for indices in batches:
            features, valid, delta, enable_delta = base.prepare_arm_inputs(
                train_data["features"][indices],
                train_data["valid_mask"][indices],
                train_data["unique_mask"][indices],
                train_data["delta_days"][indices],
                arm="role_only",
                t0_index=t0_index,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                features,
                valid,
                role_index,
                delta,
                enable_delta=enable_delta,
            )
            per_row = F.binary_cross_entropy_with_logits(
                logits,
                train_labels[indices],
                pos_weight=pos_weight,
                reduction="none",
            )
            # The global mean-one scale keeps every event's summed epoch weight
            # identical; unlike batch renormalization, it does not reweight a
            # row according to its batch composition.
            loss = (per_row * scaled_weights[indices]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{arm} epoch={epoch}: non-finite loss.")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_weighted_loss += float(
                (per_row.detach() * scaled_weights[indices]).sum()
            )
            rows_seen += len(indices)

        metrics, probabilities = evaluate(
            model,
            val_data,
            role_index=role_index,
            t0_index=t0_index,
            batch_size=args.eval_batch_size,
        )
        record = {
            "arm": arm,
            "epoch": epoch,
            "optimizer_steps": len(batches),
            "train_rows_seen": int(rows_seen),
            "train_event_balanced_bce": total_weighted_loss / rows_seen,
            "validation": metrics,
            "elapsed_seconds": float(time.monotonic() - started),
        }
        history.append(record)
        base.atomic_json_write(arm_dir / "metrics_history.json", history)
        if (
            best is None
            or metrics["event_balanced_ap"]
            > best["validation"]["event_balanced_ap"]
        ):
            best = copy.deepcopy(record)
            threshold = metrics["event_balanced_selected_threshold"]
            best_predictions = pd.DataFrame(
                {
                    "id": val_data["ids"],
                    "plume_id": val_data["plume_ids"],
                    "event_id": val_data["event_ids"],
                    "label": val_data["labels"].long().tolist(),
                    "probability": probabilities,
                    "prediction_at_0_5": (
                        probabilities >= 0.5
                    ).astype(np.int64),
                    "prediction_at_event_balanced_macro_threshold": (
                        probabilities >= threshold
                    ).astype(np.int64),
                    "arm": arm,
                    "epoch": epoch,
                }
            )
            base.atomic_torch_save(
                arm_dir / "checkpoint_best_event_balanced_ap.pt",
                {
                    **artifact_metadata(args),
                    "arm": arm,
                    "epoch": epoch,
                    "selection_metric": "event_balanced_ap",
                    "threshold_metric": "event_balanced_macro_f1",
                    "model": {
                        name: tensor.detach().cpu()
                        for name, tensor in model.state_dict().items()
                    },
                    "initial_state_sha256": initial_sha,
                    "parameter_signature": dict(parameter_signature),
                    "cache_audit": dict(cache_audit),
                    "weight_audit": weight_audit,
                    "validation": metrics,
                },
            )
            base.atomic_csv_write(
                arm_dir
                / "validation_best_event_balanced_ap_predictions.csv",
                best_predictions,
            )
        print(
            f"[posthoc-event-head] arm={arm} epoch={epoch}/{args.epochs} "
            f"trainBCE={record['train_event_balanced_bce']:.6f} "
            f"eventAP={metrics['event_balanced_ap']:.6f} "
            f"eventAUC={metrics['event_balanced_auc']:.6f} "
            f"eventMacroF1={metrics['event_balanced_selected_macro_f1']:.6f}",
            flush=True,
        )
    if best is None or best_predictions is None:
        raise RuntimeError(f"{arm} produced no checkpoint.")
    return history, best, best_predictions


def event_confusion(
    frame: pd.DataFrame, threshold: float
) -> tuple[list[str], np.ndarray]:
    events: list[str] = []
    rows: list[list[float]] = []
    for event_id, subset in frame.groupby("event_id", sort=True):
        target = subset["label"].to_numpy(dtype=np.int64)
        prediction = (
            subset["probability"].to_numpy(dtype=float) >= threshold
        ).astype(np.int64)
        weight = np.full(len(subset), 1.0 / len(subset), dtype=np.float64)
        tp = float(weight[(target == 1) & (prediction == 1)].sum())
        fp = float(weight[(target == 0) & (prediction == 1)].sum())
        fn = float(weight[(target == 1) & (prediction == 0)].sum())
        tn = float(weight[(target == 0) & (prediction == 0)].sum())
        events.append(str(event_id))
        rows.append([tp, fp, fn, tn])
    return events, np.asarray(rows, dtype=np.float64)


def f1_from_confusion(
    tp: float, fp: float, fn: float, tn: float
) -> tuple[float, float]:
    pos_denominator = 2 * tp + fp + fn
    neg_denominator = 2 * tn + fp + fn
    positive = 0.0 if pos_denominator <= 0 else 2 * tp / pos_denominator
    negative = 0.0 if neg_denominator <= 0 else 2 * tn / neg_denominator
    return float(positive), float((positive + negative) / 2.0)


def paired_event_bootstrap(
    predictions: Mapping[str, pd.DataFrame],
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tables: dict[str, np.ndarray] = {}
    canonical_events: Optional[list[str]] = None
    for arm in ARMS:
        events, table = event_confusion(
            predictions[arm],
            float(metrics[arm]["event_balanced_selected_threshold"]),
        )
        if canonical_events is None:
            canonical_events = events
        elif events != canonical_events:
            raise ValueError("Event order differs between arms.")
        tables[arm] = table
    assert canonical_events is not None
    rng = np.random.default_rng(seed)
    distributions: dict[str, dict[str, list[float]]] = {
        arm: {"positive_f1": [], "macro_f1": []} for arm in ARMS
    }
    for _ in range(int(replicates)):
        sample = rng.integers(0, len(canonical_events), size=len(canonical_events))
        for arm in ARMS:
            tp, fp, fn, tn = tables[arm][sample].sum(axis=0)
            positive, macro = f1_from_confusion(tp, fp, fn, tn)
            distributions[arm]["positive_f1"].append(positive)
            distributions[arm]["macro_f1"].append(macro)
    intervals: dict[str, Any] = {}
    for arm in ARMS:
        intervals[arm] = {}
        for name in ("positive_f1", "macro_f1"):
            values = np.asarray(distributions[arm][name], dtype=float)
            point_key = f"event_balanced_{name}_selected"
            intervals[arm][name] = {
                "point": float(metrics[arm][point_key]),
                "ci_95_low": float(np.percentile(values, 2.5)),
                "ci_95_high": float(np.percentile(values, 97.5)),
            }
    deltas: dict[str, Any] = {}
    for name, left, right in (
        ("p5_minus_p0", "p5", "p0"),
        ("p5_minus_p4", "p5", "p4"),
        ("p4_minus_p0", "p4", "p0"),
    ):
        deltas[name] = {}
        for metric_name in ("positive_f1", "macro_f1"):
            values = np.asarray(distributions[left][metric_name]) - np.asarray(
                distributions[right][metric_name]
            )
            point_key = f"event_balanced_{metric_name}_selected"
            deltas[name][metric_name] = {
                "point": float(
                    metrics[left][point_key] - metrics[right][point_key]
                ),
                "ci_95_low": float(np.percentile(values, 2.5)),
                "ci_95_high": float(np.percentile(values, 97.5)),
            }
    return intervals, deltas


def render_markdown(result: Mapping[str, Any]) -> str:
    metrics = result["point_metrics"]
    intervals = result["event_cluster_bootstrap_ci"]
    is_clean = bool(result.get("clean_inner_replicate_exploratory", False))
    lines = [
        (
            "# L89 clean-inner matched event-balanced head replicate"
            if is_clean
            else "# RCTP L89 post-hoc event-balanced head diagnostic"
        ),
        "",
        (
            "**Exploratory inner replicate only.** This is neither outer "
            "evaluation nor confirmatory/SOTA evidence."
            if is_clean
            else "**Exploratory only.** This train/dev diagnostic was "
            "motivated by the already-inspected dev result; it is not "
            "confirmatory evidence."
        ),
        "No test/sealed/holdout/outer artifact was read.",
        "",
        "| Arm | Best epoch | Event AP | Event AUC | Positive F1 (95% CI) | Macro F1 (95% CI) | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        point = metrics[arm]
        ci = intervals[arm]
        lines.append(
            f"| {arm.upper()} | {point['best_epoch']} | "
            f"{point['event_balanced_ap']:.4f} | "
            f"{point['event_balanced_auc']:.4f} | "
            f"{point['event_balanced_positive_f1_selected']:.4f} "
            f"[{ci['positive_f1']['ci_95_low']:.4f}, "
            f"{ci['positive_f1']['ci_95_high']:.4f}] | "
            f"{point['event_balanced_macro_f1_selected']:.4f} "
            f"[{ci['macro_f1']['ci_95_low']:.4f}, "
            f"{ci['macro_f1']['ci_95_high']:.4f}] | "
            f"{point['event_balanced_selected_threshold']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Paired delta | Positive F1 (95% CI) | Macro F1 (95% CI) |",
            "|---|---:|---:|",
        ]
    )
    for name in ("p5_minus_p0", "p5_minus_p4", "p4_minus_p0"):
        delta = result["paired_deltas"][name]
        lines.append(
            f"| {name} | {delta['positive_f1']['point']:+.4f} "
            f"[{delta['positive_f1']['ci_95_low']:+.4f}, "
            f"{delta['positive_f1']['ci_95_high']:+.4f}] | "
            f"{delta['macro_f1']['point']:+.4f} "
            f"[{delta['macro_f1']['ci_95_low']:+.4f}, "
            f"{delta['macro_f1']['ci_95_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "Matched change applied to every arm:",
            "",
            "- inverse canonical-event-size BCE (each event has equal base loss weight);",
            "- checkpoint selection by event-balanced AP;",
            "- threshold selection by event-balanced macro F1;",
            "- identical model initialization, batch order, seed, optimizer, and 3-epoch cap.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    base.assert_not_sealed_path(output_dir, purpose="post-hoc output")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing post-hoc run: {output_dir}"
        )
    if args.epochs != 3:
        raise ValueError("This matched diagnostic is frozen to exactly 3 epochs.")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive.")
    torch.set_num_threads(int(args.num_threads))
    args.periods_days = tuple(
        float(value.strip())
        for value in args.delta_periods.split(",")
        if value.strip()
    )
    if not args.periods_days:
        raise ValueError("At least one delta period is required.")

    cache_paths = {
        "p0": (Path(args.p0_train_cache), Path(args.p0_val_cache)),
        "p4": (Path(args.p4_train_cache), Path(args.p4_val_cache)),
        "p5": (Path(args.p5_train_cache), Path(args.p5_val_cache)),
    }
    loaded: dict[str, dict[str, Any]] = {}
    cache_audits: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        train_path = cache_paths[arm][0].expanduser().resolve()
        val_path = cache_paths[arm][1].expanduser().resolve()
        train_cache, val_cache, audit = base.load_cache_pair(train_path, val_path)
        train_indices = base.select_usable_rows(train_cache)
        val_indices = base.select_usable_rows(val_cache)
        loaded[arm] = {
            "train": base.take_rows(train_cache, train_indices),
            "val": base.take_rows(val_cache, val_indices),
            "role_index": train_cache["role_index"].long(),
            "t0_index": int(train_cache["t0_index"]),
        }
        cache_audits[arm] = audit

    identity_keys = ("ids", "plume_ids", "event_ids")
    reference = loaded["p0"]
    for arm in ("p4", "p5"):
        for split in ("train", "val"):
            for key in identity_keys:
                if loaded[arm][split][key] != reference[split][key]:
                    raise ValueError(f"{arm} {split} identity differs for {key}.")
            if not torch.equal(
                loaded[arm][split]["labels"], reference[split]["labels"]
            ):
                raise ValueError(f"{arm} {split} labels differ from P0.")
        if loaded[arm]["t0_index"] != reference["t0_index"]:
            raise ValueError(f"{arm} t0 index differs from P0.")
        if not torch.equal(loaded[arm]["role_index"], reference["role_index"]):
            raise ValueError(f"{arm} role indices differ from P0.")
        if (
            loaded[arm]["train"]["features"].shape
            != reference["train"]["features"].shape
        ):
            raise ValueError(f"{arm} train feature shape differs from P0.")

    train_data = reference["train"]
    feature_dim = int(train_data["features"].shape[-1])
    num_roles = int(train_data["features"].shape[1])
    t0_index = int(reference["t0_index"])
    role_index = reference["role_index"]
    initial_state, signature = build_initial_state(
        feature_dim=feature_dim,
        num_roles=num_roles,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        periods_days=args.periods_days,
        t0_index=t0_index,
        seed=args.seed,
    )
    initial_sha = base.state_dict_sha256(initial_state)
    arm_initial_shas = {}
    for arm in ARMS:
        model = instantiate_from_initial_state(
            initial_state,
            feature_dim=feature_dim,
            num_roles=num_roles,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            periods_days=args.periods_days,
            t0_index=t0_index,
        )
        arm_initial_shas[arm] = base.state_dict_sha256(model.state_dict())
    if set(arm_initial_shas.values()) != {initial_sha}:
        raise RuntimeError("Three-arm initialization SHA mismatch.")

    batch_plan = {
        str(epoch): [
            indices.tolist()
            for indices in base.fixed_epoch_batches(
                len(train_data["labels"]),
                batch_size=args.batch_size,
                seed=args.seed,
                epoch=epoch,
                shuffle=True,
            )
        ]
        for epoch in range(1, args.epochs + 1)
    }
    batch_plan_sha = base.sha256_bytes(base.canonical_json_bytes(batch_plan))

    output_dir.mkdir(parents=True, exist_ok=False)
    base.atomic_json_write(
        output_dir / "run_status.json",
        {
            "status": "running",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            **artifact_metadata(args),
            "test_or_sealed_read": False,
        },
    )
    run_config = {
        **artifact_metadata(args),
        "interpretation_guardrail": (
            (
                "Independent train-only clean inner replicate; exploratory "
                "only, never outer evaluation, confirmatory evidence, or SOTA."
            )
            if args.clean_inner_replicate
            else (
                "Protocol-only diagnosis motivated by an already-inspected "
                "dev result; never confirmatory evidence."
            )
        ),
        "test_or_sealed_read": False,
        "arms": list(ARMS),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "model_dim": int(args.model_dim),
        "num_heads": int(args.num_heads),
        "mlp_ratio": float(args.mlp_ratio),
        "dropout": float(args.dropout),
        "periods_days": list(args.periods_days),
        "num_threads": int(args.num_threads),
        "loss_weighting": "inverse_canonical_event_size_mean_one",
        "class_weighting": "event_balanced_negative_mass_over_positive_mass",
        "checkpoint_selection": "event_balanced_ap",
        "epoch_protocol": (
            "all three matched arms run exactly epochs 1,2,3; each arm selects "
            "the earliest epoch attaining its maximum event-balanced AP"
        ),
        "threshold_selection": "event_balanced_macro_f1",
        "initial_state_sha256": initial_sha,
        "initial_state_sha256_by_arm": arm_initial_shas,
        "parameter_signature": signature,
        "batch_plan_sha256": batch_plan_sha,
        "cache_audits": cache_audits,
    }
    base.atomic_json_write(output_dir / "run_config.json", run_config)

    try:
        best_by_arm: dict[str, Any] = {}
        histories: dict[str, Any] = {}
        predictions: dict[str, pd.DataFrame] = {}
        for arm in ARMS:
            history, best, frame = train_arm(
                arm,
                train_data=loaded[arm]["train"],
                val_data=loaded[arm]["val"],
                role_index=role_index,
                t0_index=t0_index,
                initial_state=initial_state,
                parameter_signature=signature,
                args=args,
                arm_dir=output_dir / arm,
                cache_audit=cache_audits[arm],
            )
            histories[arm] = history
            best_by_arm[arm] = best
            predictions[arm] = frame
            base.atomic_json_write(output_dir / "metrics_history.json", histories)

        point_metrics = {
            arm: {
                "best_epoch": int(best_by_arm[arm]["epoch"]),
                **best_by_arm[arm]["validation"],
            }
            for arm in ARMS
        }
        prediction_provenance = {}
        for arm in ARMS:
            prediction_path = (
                output_dir
                / arm
                / "validation_best_event_balanced_ap_predictions.csv"
            ).resolve()
            prediction_provenance[arm] = {
                "path": str(prediction_path),
                "sha256": base.sha256_file(prediction_path),
            }
        intervals, deltas = paired_event_bootstrap(
            predictions,
            point_metrics,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        comparison = {
            **artifact_metadata(args),
            "test_or_sealed_read": False,
            "matching": {
                "identity_and_labels_equal": True,
                "same_initial_state": len(set(arm_initial_shas.values())) == 1,
                "initial_state_sha256": initial_sha,
                "same_parameter_signature": True,
                "same_batch_plan_sha256": batch_plan_sha,
                "same_seed_optimizer_epochs": True,
            },
            "point_metrics": point_metrics,
            "prediction_provenance": prediction_provenance,
            "event_cluster_bootstrap_ci": intervals,
            "paired_deltas": deltas,
            "bootstrap": {
                "unit": "canonical event_id",
                "replicates": int(args.bootstrap_replicates),
                "seed": int(args.seed),
                "thresholds_refit_per_replicate": False,
            },
        }
        base.atomic_json_write(output_dir / "comparison.json", comparison)
        base.atomic_json_write(
            output_dir / "summary.json",
            {
                **artifact_metadata(args),
                "test_or_sealed_read": False,
                "selection_metric": "event_balanced_ap",
                "threshold_metric": "event_balanced_macro_f1",
                "best_by_arm": best_by_arm,
                "initial_state_sha256": initial_sha,
                "batch_plan_sha256": batch_plan_sha,
            },
        )
        (output_dir / "COMPARISON.md").write_text(
            render_markdown(comparison), encoding="utf-8"
        )
        base.atomic_json_write(
            output_dir / "run_status.json",
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                **artifact_metadata(args),
                "test_or_sealed_read": False,
            },
        )
    except Exception as error:
        base.atomic_json_write(
            output_dir / "run_status.json",
            {
                "status": "failed",
                "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                **artifact_metadata(args),
                "test_or_sealed_read": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    root = Path(
        "/diniuvol/yuyao/methanefuse_research_20260727/"
        "rctp_l89_real_cls_v1"
    )
    base_cache = Path(
        "/diniuvol/yuyao/methanefuse_research_20260727/"
        "cache/l89_ragged_cls_v1"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p0-train-cache", default=str(base_cache / "train.pt")
    )
    parser.add_argument("--p0-val-cache", default=str(base_cache / "val.pt"))
    parser.add_argument("--p4-train-cache", default=str(root / "cache/p4/train.pt"))
    parser.add_argument("--p4-val-cache", default=str(root / "cache/p4/val.pt"))
    parser.add_argument("--p5-train-cache", default=str(root / "cache/p5/train.pt"))
    parser.add_argument("--p5-val-cache", default=str(root / "cache/p5/val.pt"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--clean-inner-replicate",
        action="store_true",
        help=(
            "Label artifacts as an independent exploratory clean-inner "
            "replicate instead of the legacy post-hoc diagnostic."
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--delta-periods", default="1,3,7,30,90,365")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--num-threads", type=int, default=12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
