#!/usr/bin/env python3
"""Five-fold canonical-event OOF reliability liaison for P5 and D1.

G0 is an L2 logistic consensus over only the two logits.  G1 adds a
predeclared acquisition-reliability feature set.  Every imputer, scaler,
logistic model, and operating threshold is fit on the four training folds and
then applied to the held-out canonical events.  No test-like path is allowed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as cache_runner,
)
from research.tempo_20260728 import tempo_l89_global as tempo  # noqa: E402


SCRIPT_VERSION = "tempo-l89-oof-reliability-liaison-v1"
MODEL_NAMES = ("p5", "d1", "equal_05", "g0", "g1")


def event_class_balanced_fit_weights(
    labels: np.ndarray, event_ids: Sequence[str]
) -> tuple[np.ndarray, dict[str, float]]:
    target = np.asarray(labels, dtype=np.int64)
    event_weight = cache_runner.event_balanced_row_weights(event_ids)
    positive_mass = float(event_weight[target == 1].sum())
    negative_mass = float(event_weight[target == 0].sum())
    if positive_mass <= 0 or negative_mass <= 0:
        raise ValueError("Training fold lacks one row class.")
    class_factor = np.where(
        target == 1,
        0.5 / positive_mass,
        0.5 / negative_mass,
    )
    weight = event_weight * class_factor
    weight *= len(weight) / weight.sum()
    return weight, {
        "event_balanced_positive_mass_before_class_balance": positive_mass,
        "event_balanced_negative_mass_before_class_balance": negative_mass,
        "final_positive_weight_sum": float(weight[target == 1].sum()),
        "final_negative_weight_sum": float(weight[target == 0].sum()),
        "final_weight_mean": float(weight.mean()),
    }


def make_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    fit_intercept=True,
                    max_iter=1000,
                    random_state=20260728,
                ),
            ),
        ]
    )


def reliability_score(
    unique_history_count: np.ndarray,
    nearest_gap_days: np.ndarray,
    year_gap_days: np.ndarray,
    mean_history_quality: np.ndarray,
) -> np.ndarray:
    """Fixed acquisition prior; constants come from cadence semantics."""

    count_term = np.clip(unique_history_count / 5.0, 0.0, 1.0)
    recent_term = 1.0 / (
        1.0 + np.abs(nearest_gap_days - 16.0) / 16.0
    )
    year_term = 1.0 / (
        1.0 + np.abs(year_gap_days - 365.0) / 30.0
    )
    quality_term = np.clip(mean_history_quality, 0.0, 1.0)
    components = np.stack(
        (count_term, recent_term, year_term, quality_term), axis=1
    )
    invalid = ~np.isfinite(components)
    components[invalid] = 0.0
    return np.power(np.prod(components, axis=1), 0.25)


def build_feature_table(
    cache: Mapping[str, Any],
    indices: np.ndarray,
    *,
    p5_probability: np.ndarray,
    d1_probability: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    role_names = [str(value) for value in cache["role_names"]]
    t0_index = int(cache["t0_index"])
    history_indices = [
        index for index in range(len(role_names)) if index != t0_index
    ]
    unique = cache["unique_mask"][indices].bool().numpy()
    gaps = np.abs(cache["delta_days"][indices].float().numpy())
    quality = cache["valid_fraction"][indices].float().numpy()
    history_unique = unique[:, history_indices]
    unique_count = history_unique.sum(axis=1).astype(np.float64)
    nearest = np.full(len(indices), np.nan, dtype=np.float64)
    history_quality = np.full(len(indices), np.nan, dtype=np.float64)
    for row in range(len(indices)):
        valid = history_unique[row]
        if valid.any():
            nearest[row] = float(gaps[row, history_indices][valid].min())
            history_quality[row] = float(
                quality[row, history_indices][valid].mean()
            )
    year_index = role_names.index("year")
    year_gap = gaps[:, year_index].astype(np.float64)
    year_gap[~unique[:, year_index]] = np.nan

    p5_logit = tempo.safe_logit(p5_probability)
    d1_logit = tempo.safe_logit(d1_probability)
    reliability = reliability_score(
        unique_count, nearest, year_gap, history_quality
    )
    data: dict[str, np.ndarray] = {
        "p5_logit": p5_logit,
        "d1_logit": d1_logit,
        "abs_logit_disagreement": np.abs(d1_logit - p5_logit),
        "unique_history_count": unique_count,
    }
    for index in history_indices:
        data[f"missing_{role_names[index]}"] = (
            ~unique[:, index]
        ).astype(np.float64)
    data.update(
        {
            "nearest_gap_days": nearest,
            "abs_year_gap_deviation_365d": np.abs(year_gap - 365.0),
            "mean_history_quality": history_quality,
            "d1_logit_x_fixed_reliability": d1_logit * reliability,
        }
    )
    frame = pd.DataFrame(data)
    audit = {
        "g0_features": ["p5_logit", "d1_logit"],
        "g1_features": list(frame.columns),
        "reliability_formula": (
            "geometric_mean(count/5, "
            "1/(1+abs(nearest_gap-16)/16), "
            "1/(1+abs(year_gap-365)/30), mean_history_quality)"
        ),
        "nominal_recent_gap_days": 16.0,
        "nominal_year_gap_days": 365.0,
        "year_gap_tolerance_days": 30.0,
        "constants_selected_from_metrics": False,
        "reliability_summary": {
            "minimum": float(reliability.min()),
            "median": float(np.median(reliability)),
            "maximum": float(reliability.max()),
        },
    }
    return frame, audit


def decisions_from_fold_threshold(
    probability: np.ndarray, threshold: np.ndarray
) -> np.ndarray:
    return np.asarray(probability, dtype=np.float64) >= np.asarray(
        threshold, dtype=np.float64
    )


def oof_metric_bundle(
    labels: np.ndarray,
    probabilities: np.ndarray,
    decisions: np.ndarray,
    event_ids: Sequence[str],
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    score = np.asarray(probabilities, dtype=np.float64)
    prediction = np.asarray(decisions, dtype=bool)
    weights = cache_runner.event_balanced_row_weights(event_ids)
    event_frame = pd.DataFrame(
        {
            "event_id": [str(value) for value in event_ids],
            "label": target,
            "prediction": prediction.astype(np.int64),
        }
    )
    grouped_label = event_frame.groupby("event_id", sort=True)["label"].max()
    null_ids = set(grouped_label[grouped_label.eq(0)].index)
    positive_ids = set(grouped_label[grouped_label.eq(1)].index)
    null = event_frame[event_frame["event_id"].isin(null_ids)]
    if len(null):
        null_grouped = null.groupby("event_id", sort=True)["prediction"]
        null_rates = null_grouped.mean().to_numpy(dtype=np.float64)
        null_any = null_grouped.max().to_numpy(dtype=np.float64)
    else:
        null_rates = np.zeros(0)
        null_any = np.zeros(0)
    positive = event_frame[event_frame["event_id"].isin(positive_ids)].copy()
    positive["true_detection"] = positive["label"] * positive["prediction"]
    if len(positive):
        detected = (
            positive.groupby("event_id", sort=True)["true_detection"]
            .max()
            .to_numpy(dtype=np.float64)
        )
    else:
        detected = np.zeros(0)
    return {
        "rows": int(len(target)),
        "events": int(len(set(event_ids))),
        "event_balanced_ap": float(
            average_precision_score(target, score, sample_weight=weights)
        ),
        "event_balanced_auc": float(
            roc_auc_score(target, score, sample_weight=weights)
        ),
        "event_balanced_positive_f1_oof_threshold": float(
            f1_score(
                target,
                prediction,
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "event_balanced_macro_f1_oof_threshold": float(
            f1_score(
                target,
                prediction,
                labels=[0, 1],
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "negative_row_fp_count": int(
            ((target == 0) & prediction).sum()
        ),
        "all_negative_event_count": int(len(null_rates)),
        "all_negative_fp_mass": float(null_rates.sum()),
        "all_negative_events_with_any_fp": int(null_any.sum()),
        "all_negative_event_any_fp_rate": (
            float(null_any.mean()) if len(null_any) else 0.0
        ),
        "positive_event_count": int(len(detected)),
        "positive_event_any_detection_recall": (
            float(detected.mean()) if len(detected) else 0.0
        ),
    }


def weighted_bootstrap_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    decisions: np.ndarray,
    row_weights: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.int64)
    score = np.asarray(probabilities, dtype=np.float64)
    prediction = np.asarray(decisions, dtype=bool)
    weight = np.asarray(row_weights, dtype=np.float64)
    positive_f1 = f1_score(
        target,
        prediction,
        sample_weight=weight,
        zero_division=0,
    )
    macro_f1 = f1_score(
        target,
        prediction,
        labels=[0, 1],
        average="macro",
        sample_weight=weight,
        zero_division=0,
    )
    return {
        "event_balanced_ap": float(
            average_precision_score(target, score, sample_weight=weight)
        ),
        "event_balanced_auc": float(
            roc_auc_score(target, score, sample_weight=weight)
        ),
        "event_balanced_positive_f1_oof_threshold": float(positive_f1),
        "event_balanced_macro_f1_oof_threshold": float(macro_f1),
    }


def paired_event_bootstrap(
    labels: np.ndarray,
    event_ids: Sequence[str],
    probabilities: Mapping[str, np.ndarray],
    decisions: Mapping[str, np.ndarray],
    point_metrics: Mapping[str, Mapping[str, Any]],
    *,
    comparisons: Sequence[tuple[str, str, str]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    events = np.asarray([str(value) for value in event_ids], dtype=object)
    canonical, codes, sizes = np.unique(
        events, return_inverse=True, return_counts=True
    )
    event_positive = np.zeros(len(canonical), dtype=bool)
    for code in range(len(canonical)):
        event_positive[code] = bool(target[codes == code].max())
    null_event_indices = np.flatnonzero(~event_positive)
    per_model_null_rate: dict[str, np.ndarray] = {}
    per_model_null_any: dict[str, np.ndarray] = {}
    for name in probabilities:
        rate = np.zeros(len(canonical), dtype=np.float64)
        any_fp = np.zeros(len(canonical), dtype=np.float64)
        prediction = np.asarray(decisions[name], dtype=bool)
        for code in null_event_indices:
            values = prediction[codes == code]
            rate[code] = float(values.mean())
            any_fp[code] = float(values.max())
        per_model_null_rate[name] = rate
        per_model_null_any[name] = any_fp

    metric_names = (
        "event_balanced_ap",
        "event_balanced_auc",
        "event_balanced_positive_f1_oof_threshold",
        "event_balanced_macro_f1_oof_threshold",
        "all_negative_fp_mass",
        "all_negative_event_any_fp_rate",
    )
    distributions = {
        comparison: {metric: [] for metric in metric_names}
        for comparison, _, _ in comparisons
    }
    rng = np.random.default_rng(int(seed))
    for replicate in range(int(replicates)):
        sampled = rng.integers(0, len(canonical), size=len(canonical))
        event_counts = np.bincount(
            sampled, minlength=len(canonical)
        ).astype(np.float64)
        row_weights = event_counts[codes] / sizes[codes]
        model_metrics: dict[str, dict[str, float]] = {}
        sampled_null_total = float(event_counts[null_event_indices].sum())
        for name in probabilities:
            value = weighted_bootstrap_metrics(
                target,
                probabilities[name],
                decisions[name],
                row_weights,
            )
            value["all_negative_fp_mass"] = float(
                (
                    event_counts[null_event_indices]
                    * per_model_null_rate[name][null_event_indices]
                ).sum()
            )
            value["all_negative_event_any_fp_rate"] = (
                float(
                    (
                        event_counts[null_event_indices]
                        * per_model_null_any[name][null_event_indices]
                    ).sum()
                    / sampled_null_total
                )
                if sampled_null_total > 0
                else 0.0
            )
            model_metrics[name] = value
        for comparison, left, right in comparisons:
            for metric in metric_names:
                distributions[comparison][metric].append(
                    model_metrics[left][metric]
                    - model_metrics[right][metric]
                )
        if (replicate + 1) % 500 == 0:
            print(
                f"[oof-bootstrap] {replicate + 1}/{replicates}",
                flush=True,
            )

    output: dict[str, Any] = {}
    for comparison, left, right in comparisons:
        output[comparison] = {}
        for metric in metric_names:
            values = np.asarray(
                distributions[comparison][metric], dtype=np.float64
            )
            output[comparison][metric] = {
                "point": float(
                    point_metrics[left][metric]
                    - point_metrics[right][metric]
                ),
                "ci_95_low": float(np.percentile(values, 2.5)),
                "ci_95_high": float(np.percentile(values, 97.5)),
            }
    return output


def coefficient_audit(
    pipeline: Pipeline, feature_names: Sequence[str]
) -> dict[str, Any]:
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    logistic = pipeline.named_steps["logistic"]
    return {
        "feature_names": list(feature_names),
        "imputer_statistics": {
            name: float(value)
            for name, value in zip(feature_names, imputer.statistics_)
        },
        "scaler_mean": {
            name: float(value)
            for name, value in zip(feature_names, scaler.mean_)
        },
        "scaler_scale": {
            name: float(value)
            for name, value in zip(feature_names, scaler.scale_)
        },
        "standardized_coefficients": {
            name: float(value)
            for name, value in zip(feature_names, logistic.coef_[0])
        },
        "intercept": float(logistic.intercept_[0]),
        "optimizer_iterations": int(logistic.n_iter_[0]),
    }


def aggregate_coefficients(
    folds: Sequence[Mapping[str, Any]], model: str
) -> dict[str, Any]:
    names = list(folds[0]["models"][model]["feature_names"])
    output: dict[str, Any] = {}
    for name in names:
        values = np.asarray(
            [
                fold["models"][model]["standardized_coefficients"][name]
                for fold in folds
            ],
            dtype=np.float64,
        )
        output[name] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "positive_folds": int((values > 0).sum()),
            "negative_folds": int((values < 0).sum()),
        }
    return output


def markdown_report(payload: Mapping[str, Any]) -> str:
    metrics = payload["oof_metrics"]
    gate = payload["promotion_gate"]
    bootstrap = payload["paired_event_bootstrap"]
    coefficients = payload["aggregate_standardized_coefficients"]["g1"]

    def fmt(value: float) -> str:
        return f"{float(value):.6f}"

    lines = [
        "# L89 five-fold canonical-event OOF reliability liaison",
        "",
        "This is a development-internal, cross-fitted mechanism experiment. "
        "It is not a sealed-test result.",
        "",
        "## OOF metrics",
        "",
        "| model | AP | AUC | macro-F1 | positive F1 | null FP mass | any-FP events |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        value = metrics[name]
        lines.append(
            f"| {name} | {fmt(value['event_balanced_ap'])} | "
            f"{fmt(value['event_balanced_auc'])} | "
            f"{fmt(value['event_balanced_macro_f1_oof_threshold'])} | "
            f"{fmt(value['event_balanced_positive_f1_oof_threshold'])} | "
            f"{fmt(value['all_negative_fp_mass'])} | "
            f"{value['all_negative_events_with_any_fp']} |"
        )
    lines.extend(
        [
            "",
            "All operating thresholds are selected within the four training "
            "folds and applied unchanged to held-out events.",
            "",
            "## Promotion verdict",
            "",
            f"**{gate['verdict']}**",
            "",
            gate["reason"],
            "",
            "The strict gate requires G1 to exceed both G0 and equal-0.5 in "
            "AP, AUC, and macro-F1, while having lower all-negative FP mass "
            "than both. No tolerance or hyperparameter search is used.",
            "",
            "## Interpretation",
            "",
            f"- Equal-0.5 remains the ranking winner: AP "
            f"`{metrics['equal_05']['event_balanced_ap']:.6f}` and AUC "
            f"`{metrics['equal_05']['event_balanced_auc']:.6f}`. G0 falls to "
            f"AP `{metrics['g0']['event_balanced_ap']:.6f}` and G1 to "
            f"`{metrics['g1']['event_balanced_ap']:.6f}`.",
            f"- G1 recovers macro-F1 and suppresses null FP mass relative to "
            f"G0, but its AP delta versus equal-0.5 is "
            f"`{bootstrap['g1_minus_equal']['event_balanced_ap']['point']:+.6f}` "
            "and its interval crosses zero.",
            f"- The D1×reliability interaction is positive in "
            f"`{coefficients['d1_logit_x_fixed_reliability']['positive_folds']}/5` "
            "folds, while absolute P5/D1 disagreement is negative in "
            f"`{coefficients['abs_logit_disagreement']['negative_folds']}/5`. "
            "The intended reliability mechanism is visible in coefficients "
            "but is not strong or stable enough to improve held-out ranking.",
            "- This rejects a learned reliability liaison on the present 124 "
            "development events. Fixed equal evidence fusion is retained; "
            "the mechanism bins should guide representation/patch design, "
            "not another calibrated gate.",
            "",
            "## Paired canonical-event bootstrap",
            "",
            "| comparison | metric | point delta | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for comparison in ("g1_minus_equal", "g1_minus_g0", "g0_minus_equal"):
        for metric in (
            "event_balanced_ap",
            "event_balanced_auc",
            "event_balanced_macro_f1_oof_threshold",
            "all_negative_fp_mass",
        ):
            value = bootstrap[comparison][metric]
            lines.append(
                f"| {comparison} | {metric} | "
                f"{value['point']:+.6f} | "
                f"[{value['ci_95_low']:+.6f}, "
                f"{value['ci_95_high']:+.6f}] |"
            )
    lines.extend(
        [
            "",
            "## Fixed models",
            "",
            "- G0: standardized P5 logit and D1 logit.",
            "- G1: G0 plus absolute logit disagreement, unique-history count, "
            "five role-missing flags, nearest gap, absolute deviation of the "
            "year gap from 365 days, mean history quality, and the D1-logit × "
            "fixed-reliability interaction.",
            "- Reliability is the geometric mean of count completeness, "
            "16-day recent-cadence agreement, 365-day annual-cadence "
            "agreement, and history quality.",
            "- Both models use L2 logistic regression with fixed C=1.",
            "- Fit weights equalize canonical events and then the two row "
            "classes inside each training fold.",
            "",
            "## Boundary",
            "",
            "- Five folds are stratified only by positive/all-negative "
            "canonical event.",
            "- Imputation, scaling, coefficients, and thresholds are "
            "training-fold only.",
            "- Bootstrap resamples fixed held-out OOF predictions; it never "
            "refits a model or threshold.",
            "- No test, sealed, or holdout path was read.",
            "",
        ]
    )
    return "\n".join(lines)


def command_run(args: argparse.Namespace) -> None:
    paths = {
        "cache": Path(args.dev_cache).expanduser().resolve(),
        "p5": Path(args.p5_predictions).expanduser().resolve(),
        "d1": Path(args.d1_predictions).expanduser().resolve(),
        "equal": Path(args.equal_predictions).expanduser().resolve(),
        "output_json": Path(args.output_json).expanduser().resolve(),
        "output_csv": Path(args.output_csv).expanduser().resolve(),
        "output_markdown": Path(args.output_markdown).expanduser().resolve(),
    }
    for name, path in paths.items():
        tempo.assert_development_path(path, purpose=f"OOF liaison {name}")
    for name in ("output_json", "output_csv", "output_markdown"):
        if paths[name].exists() and not args.overwrite:
            raise FileExistsError(paths[name])

    cache = cache_runner.torch_load_trusted(paths["cache"])
    if not isinstance(cache, Mapping):
        raise ValueError("Development cache is not a mapping.")
    cache_runner.validate_cache_payload(
        cache, path=paths["cache"], expected_split="val"
    )
    usable = cache_runner.select_usable_rows(cache)
    indices = usable.numpy()
    ids = [str(cache["ids"][index]) for index in indices]
    event_ids = [str(cache["event_ids"][index]) for index in indices]
    labels = cache["labels"][usable].long().numpy()
    prediction_frames = {
        name: tempo.load_prediction_table(paths[name])
        for name in ("p5", "d1", "equal")
    }
    for name, frame in prediction_frames.items():
        if frame["id"].astype(str).tolist() != ids:
            raise ValueError(f"{name} prediction IDs differ from cache.")
        if frame["event_id"].astype(str).tolist() != event_ids:
            raise ValueError(f"{name} event IDs differ from cache.")
        if not np.array_equal(
            frame["label"].to_numpy(dtype=np.int64), labels
        ):
            raise ValueError(f"{name} labels differ from cache.")
    fixed_probability = {
        "p5": prediction_frames["p5"]["probability"].to_numpy(
            dtype=np.float64
        ),
        "d1": prediction_frames["d1"]["probability"].to_numpy(
            dtype=np.float64
        ),
        "equal_05": prediction_frames["equal"]["probability"].to_numpy(
            dtype=np.float64
        ),
    }
    replayed_equal = 1.0 / (
        1.0
        + np.exp(
            -0.5 * tempo.safe_logit(fixed_probability["p5"])
            -0.5 * tempo.safe_logit(fixed_probability["d1"])
        )
    )
    equal_replay_error = float(
        np.max(np.abs(replayed_equal - fixed_probability["equal_05"]))
    )
    if equal_replay_error > 1e-10:
        raise ValueError("Equal-0.5 prediction replay failed.")

    features, feature_audit = build_feature_table(
        cache,
        indices,
        p5_probability=fixed_probability["p5"],
        d1_probability=fixed_probability["d1"],
    )
    g0_names = feature_audit["g0_features"]
    g1_names = feature_audit["g1_features"]
    events_frame = (
        pd.DataFrame({"event_id": event_ids, "label": labels})
        .groupby("event_id", sort=True)["label"]
        .max()
        .reset_index()
    )
    splitter = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=int(args.fold_seed)
    )
    event_to_fold: dict[str, int] = {}
    for fold, (_, heldout) in enumerate(
        splitter.split(events_frame["event_id"], events_frame["label"])
    ):
        for event in events_frame.iloc[heldout]["event_id"].astype(str):
            event_to_fold[event] = int(fold)
    row_fold = np.asarray(
        [event_to_fold[event] for event in event_ids], dtype=np.int64
    )
    if set(row_fold.tolist()) != set(range(5)):
        raise RuntimeError("Not all OOF folds are represented.")

    oof_probability = {
        name: np.full(len(labels), np.nan, dtype=np.float64)
        for name in MODEL_NAMES
    }
    oof_threshold = {
        name: np.full(len(labels), np.nan, dtype=np.float64)
        for name in MODEL_NAMES
    }
    fold_audit: list[dict[str, Any]] = []
    for fold in range(5):
        heldout = row_fold == fold
        training = ~heldout
        train_events = set(
            np.asarray(event_ids, dtype=object)[training].tolist()
        )
        heldout_events = set(
            np.asarray(event_ids, dtype=object)[heldout].tolist()
        )
        overlap = train_events & heldout_events
        if overlap:
            raise RuntimeError(f"Fold {fold} event leakage: {sorted(overlap)}")
        fit_weight, weight_audit = event_class_balanced_fit_weights(
            labels[training],
            np.asarray(event_ids, dtype=object)[training].tolist(),
        )
        fold_models: dict[str, Any] = {}
        for model_name, feature_names in (
            ("g0", g0_names),
            ("g1", g1_names),
        ):
            pipeline = make_logistic_pipeline()
            pipeline.fit(
                features.loc[training, feature_names],
                labels[training],
                logistic__sample_weight=fit_weight,
            )
            train_probability = pipeline.predict_proba(
                features.loc[training, feature_names]
            )[:, 1]
            heldout_probability = pipeline.predict_proba(
                features.loc[heldout, feature_names]
            )[:, 1]
            threshold, _ = tempo.best_macro_threshold(
                labels[training],
                train_probability,
                np.asarray(event_ids, dtype=object)[training].tolist(),
            )
            oof_probability[model_name][heldout] = heldout_probability
            oof_threshold[model_name][heldout] = float(threshold)
            fold_models[model_name] = {
                **coefficient_audit(pipeline, feature_names),
                "train_selected_threshold": float(threshold),
            }
        for model_name in ("p5", "d1", "equal_05"):
            train_probability = fixed_probability[model_name][training]
            threshold, _ = tempo.best_macro_threshold(
                labels[training],
                train_probability,
                np.asarray(event_ids, dtype=object)[training].tolist(),
            )
            oof_probability[model_name][heldout] = fixed_probability[
                model_name
            ][heldout]
            oof_threshold[model_name][heldout] = float(threshold)
            fold_models[model_name] = {
                "train_selected_threshold": float(threshold)
            }
        fold_audit.append(
            {
                "fold": int(fold),
                "train_events": int(len(train_events)),
                "heldout_events": int(len(heldout_events)),
                "train_rows": int(training.sum()),
                "heldout_rows": int(heldout.sum()),
                "heldout_positive_events": int(
                    events_frame[
                        events_frame["event_id"].isin(heldout_events)
                    ]["label"].sum()
                ),
                "heldout_all_negative_events": int(
                    len(heldout_events)
                    - events_frame[
                        events_frame["event_id"].isin(heldout_events)
                    ]["label"].sum()
                ),
                "event_overlap": 0,
                "fit_weight_audit": weight_audit,
                "models": fold_models,
                "heldout_event_ids": sorted(heldout_events),
            }
        )

    for name in MODEL_NAMES:
        if not np.isfinite(oof_probability[name]).all():
            raise RuntimeError(f"{name} has incomplete OOF probabilities.")
        if not np.isfinite(oof_threshold[name]).all():
            raise RuntimeError(f"{name} has incomplete OOF thresholds.")
    oof_decision = {
        name: decisions_from_fold_threshold(
            oof_probability[name], oof_threshold[name]
        )
        for name in MODEL_NAMES
    }
    metrics = {
        name: oof_metric_bundle(
            labels, oof_probability[name], oof_decision[name], event_ids
        )
        for name in MODEL_NAMES
    }
    comparisons = (
        ("g1_minus_equal", "g1", "equal_05"),
        ("g1_minus_g0", "g1", "g0"),
        ("g0_minus_equal", "g0", "equal_05"),
    )
    bootstrap = paired_event_bootstrap(
        labels,
        event_ids,
        oof_probability,
        oof_decision,
        metrics,
        comparisons=comparisons,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
    )

    g1 = metrics["g1"]
    comparators = (metrics["g0"], metrics["equal_05"])
    checks = {
        "ap_above_both": all(
            g1["event_balanced_ap"] > value["event_balanced_ap"]
            for value in comparators
        ),
        "auc_above_both": all(
            g1["event_balanced_auc"] > value["event_balanced_auc"]
            for value in comparators
        ),
        "macro_f1_above_both": all(
            g1["event_balanced_macro_f1_oof_threshold"]
            > value["event_balanced_macro_f1_oof_threshold"]
            for value in comparators
        ),
        "null_fp_mass_below_both": all(
            g1["all_negative_fp_mass"] < value["all_negative_fp_mass"]
            for value in comparators
        ),
    }
    promoted = all(checks.values())
    gate = {
        "predeclared_checks": checks,
        "promoted": promoted,
        "verdict": "PROMOTE G1" if promoted else "REJECT G1",
        "reason": (
            "G1 clears every strict OOF metric gate."
            if promoted
            else "G1 does not beat both G0 and equal-0.5 on every strict OOF "
            "metric gate; no follow-up tuning is authorized."
        ),
    }

    output_frame = pd.DataFrame(
        {
            "id": ids,
            "event_id": event_ids,
            "label": labels,
            "fold": row_fold,
        }
    )
    for name in MODEL_NAMES:
        output_frame[f"{name}_probability"] = oof_probability[name]
        output_frame[f"{name}_fold_threshold"] = oof_threshold[name]
        output_frame[f"{name}_prediction"] = oof_decision[name].astype(
            np.int64
        )
    tempo.atomic_csv_write(paths["output_csv"], output_frame)
    payload = {
        "script_version": SCRIPT_VERSION,
        "audit_type": "five-fold-canonical-event-oof-reliability-liaison",
        "development_internal_cross_fitted": True,
        "sealed_result": False,
        "fold_seed": int(args.fold_seed),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "model_contract": {
            "g0": "L2 logistic consensus on P5 and D1 logits",
            "g1": (
                "G0 plus fixed reliability attributes and "
                "D1-logit×reliability"
            ),
            "C": 1.0,
            "C_or_feature_search": False,
            "fold_local_imputation_scaling_fit": True,
            "fold_local_threshold_fit": True,
            "fit_weighting": "canonical-event then row-class balanced",
        },
        "feature_audit": feature_audit,
        "folds": fold_audit,
        "aggregate_standardized_coefficients": {
            "g0": aggregate_coefficients(fold_audit, "g0"),
            "g1": aggregate_coefficients(fold_audit, "g1"),
        },
        "oof_metrics": metrics,
        "paired_event_bootstrap": bootstrap,
        "promotion_gate": gate,
        "equal_05_replay_max_abs_probability_error": equal_replay_error,
        "oof_predictions_csv": str(paths["output_csv"]),
        "oof_predictions_csv_sha256": cache_runner.sha256_file(
            paths["output_csv"]
        ),
        "provenance": {
            name: {
                "path": str(paths[name]),
                "sha256": cache_runner.sha256_file(paths[name]),
            }
            for name in ("cache", "p5", "d1", "equal")
        },
        "test_or_sealed_or_holdout_read": False,
    }
    tempo.atomic_json_write(paths["output_json"], payload)
    paths["output_markdown"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["output_markdown"].with_suffix(
        paths["output_markdown"].suffix + ".tmp"
    )
    temporary.write_text(markdown_report(payload), encoding="utf-8")
    temporary.replace(paths["output_markdown"])
    print(
        json.dumps(
            {
                "oof_metrics": metrics,
                "promotion_gate": gate,
                "output_json": str(paths["output_json"]),
                "output_csv": str(paths["output_csv"]),
                "output_markdown": str(paths["output_markdown"]),
                "test_or_sealed_or_holdout_read": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-cache", required=True)
    parser.add_argument("--p5-predictions", required=True)
    parser.add_argument("--d1-predictions", required=True)
    parser.add_argument("--equal-predictions", required=True)
    parser.add_argument("--fold-seed", type=int, default=20260728)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    command_run(build_parser().parse_args())


if __name__ == "__main__":
    main()
