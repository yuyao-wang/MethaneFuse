#!/usr/bin/env python3
"""CPU-only mechanism audit for fixed L89 P5/D1/TEMPO predictions.

This script is deliberately descriptive.  It never chooses cut points from
model performance, never refits a threshold inside a subgroup, and rejects
test/sealed/holdout paths.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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

from research.pretraining_20260727 import (
    l89_ragged_cls_experiment as cache_runner,
)
from research.tempo_20260728 import tempo_l89_global as tempo


SCRIPT_VERSION = "tempo-l89-mechanism-audit-v1"
OPTIONAL_PHYSICAL_RE = re.compile(
    r"(plume[_ -]?(size|area|width|length)|"
    r"concentration|enhancement|ppm|ppb|"
    r"emission[_ -]?rate|flux|integrated[_ -]?mass|(^|_)ime($|_))",
    re.IGNORECASE,
)


def finite_or_none(value: float | np.floating[Any]) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def safe_metric_delta(
    left: Mapping[str, Any], right: Mapping[str, Any], key: str
) -> float | None:
    if left.get(key) is None or right.get(key) is None:
        return None
    return float(left[key]) - float(right[key])


def fixed_threshold_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    events = frame["event_id"].astype(str).tolist()
    score = np.asarray(probability, dtype=np.float64)
    if score.shape != labels.shape:
        raise ValueError("Probability shape differs from subgroup labels.")
    weights = cache_runner.event_balanced_row_weights(events)
    prediction = score >= float(threshold)
    classes = np.unique(labels)
    ap = (
        float(average_precision_score(labels, score, sample_weight=weights))
        if classes.size == 2
        else None
    )
    auc = (
        float(roc_auc_score(labels, score, sample_weight=weights))
        if classes.size == 2
        else None
    )
    positive_f1 = float(
        f1_score(
            labels,
            prediction,
            sample_weight=weights,
            zero_division=0,
        )
    )
    macro_f1 = float(
        f1_score(
            labels,
            prediction,
            labels=[0, 1],
            average="macro",
            sample_weight=weights,
            zero_division=0,
        )
    )
    negative = labels == 0
    negative_weight = float(weights[negative].sum())
    false_positive_weight = float(weights[negative & prediction].sum())

    null = frame["event_kind"].eq("all_negative_event").to_numpy()
    null_frame = pd.DataFrame(
        {
            "event_id": frame.loc[null, "event_id"].astype(str).tolist(),
            "prediction": prediction[null].astype(np.int64),
        }
    )
    if len(null_frame):
        null_grouped = null_frame.groupby("event_id", sort=True)["prediction"]
        null_fp_rates = null_grouped.mean().to_numpy(dtype=np.float64)
        null_any_fp = null_grouped.max().to_numpy(dtype=np.float64)
    else:
        null_fp_rates = np.zeros(0, dtype=np.float64)
        null_any_fp = np.zeros(0, dtype=np.float64)

    positive_event = frame["event_kind"].eq("positive_event").to_numpy()
    true_positive = positive_event & (labels == 1) & prediction
    detection_frame = pd.DataFrame(
        {
            "event_id": frame.loc[positive_event, "event_id"]
            .astype(str)
            .tolist(),
            "true_positive": true_positive[positive_event].astype(np.int64),
        }
    )
    if len(detection_frame):
        detected = (
            detection_frame.groupby("event_id", sort=True)["true_positive"]
            .max()
            .to_numpy(dtype=np.float64)
        )
    else:
        detected = np.zeros(0, dtype=np.float64)
    return {
        "rows": int(len(frame)),
        "events": int(frame["event_id"].nunique()),
        "positive_rows": int((labels == 1).sum()),
        "negative_rows": int((labels == 0).sum()),
        "positive_prevalence": float(labels.mean()),
        "event_balanced_ap": ap,
        "event_balanced_auc": auc,
        "event_balanced_positive_f1_fixed_global_threshold": positive_f1,
        "event_balanced_macro_f1_fixed_global_threshold": macro_f1,
        "global_threshold": float(threshold),
        "negative_row_fp_count": int((negative & prediction).sum()),
        "event_balanced_negative_row_fp_rate": (
            false_positive_weight / negative_weight
            if negative_weight > 0
            else 0.0
        ),
        "all_negative_event_count": int(len(null_fp_rates)),
        "all_negative_fp_mass": float(null_fp_rates.sum()),
        "all_negative_events_with_any_fp": int(null_any_fp.sum()),
        "all_negative_event_any_fp_rate": (
            float(null_any_fp.mean()) if len(null_any_fp) else 0.0
        ),
        "positive_event_count": int(len(detected)),
        "positive_event_any_detection_recall": (
            float(detected.mean()) if len(detected) else 0.0
        ),
        "mean_positive_probability": (
            float(score[labels == 1].mean()) if (labels == 1).any() else None
        ),
        "mean_negative_probability": (
            float(score[labels == 0].mean()) if (labels == 0).any() else None
        ),
    }


def subgroup_metrics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    probabilities: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    values = frame[group_column].astype(str)
    for value in sorted(values.unique()):
        mask = values.eq(value).to_numpy()
        subset = frame.loc[mask].reset_index(drop=True)
        systems = {
            name: fixed_threshold_metrics(
                subset, probability[mask], threshold=thresholds[name]
            )
            for name, probability in probabilities.items()
        }
        delta_keys = (
            "event_balanced_ap",
            "event_balanced_auc",
            "event_balanced_positive_f1_fixed_global_threshold",
            "event_balanced_macro_f1_fixed_global_threshold",
            "event_balanced_negative_row_fp_rate",
            "all_negative_fp_mass",
            "all_negative_event_any_fp_rate",
            "positive_event_any_detection_recall",
        )
        output.append(
            {
                "bin": value,
                "systems": systems,
                "fusion_minus_p5": {
                    key: safe_metric_delta(
                        systems["fusion"], systems["p5"], key
                    )
                    for key in delta_keys
                },
                "d1_minus_p5": {
                    key: safe_metric_delta(systems["d1"], systems["p5"], key)
                    for key in delta_keys
                },
            }
        )
    return output


def semantic_lag_bin(value: float, *, nearest: bool) -> str:
    if not math.isfinite(value):
        return "00_no_unique_history"
    if nearest:
        if value <= 7:
            return "01_le_7d"
        if value <= 16:
            return "02_8_16d"
        if value <= 24:
            return "03_17_24d"
        if value <= 32:
            return "04_25_32d"
        return "05_gt_32d"
    if value <= 350:
        return "01_le_350d"
    if value <= 365:
        return "02_351_365d"
    return "03_gt_365d"


def physical_quartile_bins(
    values: pd.Series,
) -> tuple[pd.Series, dict[str, Any]]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if len(finite) < 20 or finite.nunique() < 4:
        return (
            pd.Series(["unavailable"] * len(values), index=values.index),
            {
                "usable": False,
                "reason": "fewer than 20 finite values or fewer than 4 unique values",
            },
        )
    edges = np.quantile(finite.to_numpy(dtype=np.float64), [0, 0.25, 0.5, 0.75, 1])
    edges = np.unique(edges)
    if len(edges) < 3:
        return (
            pd.Series(["unavailable"] * len(values), index=values.index),
            {"usable": False, "reason": "degenerate quartile edges"},
        )
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = [f"Q{index + 1}" for index in range(len(edges) - 1)]
    binned = pd.cut(
        numeric,
        bins=edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    ).astype("string")
    binned = binned.fillna("missing")
    return binned, {
        "usable": True,
        "method": "unlabeled development quartiles; no metric-based search",
        "finite_values": int(len(finite)),
        "edges": [finite_or_none(value) for value in edges],
    }


def category_summary(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    p5_probability: np.ndarray,
    d1_probability: np.ndarray,
    fusion_probability: np.ndarray,
    p5_prediction: np.ndarray,
    d1_prediction: np.ndarray,
    fusion_prediction: np.ndarray,
    limit_examples: int = 25,
) -> dict[str, Any]:
    subset = frame.loc[mask].copy()
    indices = np.flatnonzero(mask)
    labels = frame["label"].to_numpy(dtype=np.int64)
    global_weights = cache_runner.event_balanced_row_weights(
        frame["event_id"].astype(str).tolist()
    )
    subset["p5_probability"] = p5_probability[mask]
    subset["d1_probability"] = d1_probability[mask]
    subset["fusion_probability"] = fusion_probability[mask]
    subset["p5_prediction"] = p5_prediction[mask].astype(np.int64)
    subset["d1_prediction"] = d1_prediction[mask].astype(np.int64)
    subset["fusion_prediction"] = fusion_prediction[mask].astype(np.int64)
    signed = 2 * labels - 1
    improvement = signed * (
        fusion_probability - p5_probability
    )
    subset["fusion_true_class_probability_gain"] = improvement[mask]
    if len(subset):
        example_columns = [
            "id",
            "event_id",
            "label",
            "event_kind",
            "unique_history_count",
            "nearest_history_lag_bin",
            "maximum_history_lag_bin",
            "t0_quality_bin",
            "history_quality_bin",
            "role_pattern",
            "p5_probability",
            "d1_probability",
            "fusion_probability",
            "p5_prediction",
            "d1_prediction",
            "fusion_prediction",
            "fusion_true_class_probability_gain",
        ]
        examples = (
            subset.sort_values(
                "fusion_true_class_probability_gain", ascending=False
            )
            .head(limit_examples)[example_columns]
            .to_dict(orient="records")
        )
    else:
        examples = []
    return {
        "rows": int(mask.sum()),
        "events": int(frame.loc[mask, "event_id"].nunique()),
        "positive_rows": int(frame.loc[mask, "label"].sum()),
        "negative_rows": int(mask.sum() - frame.loc[mask, "label"].sum()),
        "global_event_balanced_row_mass": float(global_weights[indices].sum()),
        "mean_p5_probability": (
            float(p5_probability[mask].mean()) if mask.any() else None
        ),
        "mean_d1_probability": (
            float(d1_probability[mask].mean()) if mask.any() else None
        ),
        "mean_fusion_probability": (
            float(fusion_probability[mask].mean()) if mask.any() else None
        ),
        "feature_bin_counts": {
            column: {
                str(key): int(value)
                for key, value in frame.loc[mask, column]
                .astype(str)
                .value_counts()
                .items()
            }
            for column in (
                "unique_history_count",
                "nearest_history_lag_bin",
                "maximum_history_lag_bin",
                "t0_quality_bin",
                "history_quality_bin",
                "role_pattern",
                "event_kind",
            )
        },
        "examples_largest_true_class_probability_gain_first": examples,
    }


def event_transition_audit(
    frame: pd.DataFrame,
    p5_prediction: np.ndarray,
    fusion_prediction: np.ndarray,
) -> dict[str, Any]:
    working = frame[["event_id", "event_kind", "label"]].copy()
    working["p5_prediction"] = p5_prediction.astype(np.int64)
    working["fusion_prediction"] = fusion_prediction.astype(np.int64)
    null = working[working["event_kind"].eq("all_negative_event")]
    null_group = null.groupby("event_id", sort=True)[
        ["p5_prediction", "fusion_prediction"]
    ].max()
    positive = working[working["event_kind"].eq("positive_event")].copy()
    positive["p5_true_detection"] = (
        positive["label"] * positive["p5_prediction"]
    )
    positive["fusion_true_detection"] = (
        positive["label"] * positive["fusion_prediction"]
    )
    positive_group = positive.groupby("event_id", sort=True)[
        ["p5_true_detection", "fusion_true_detection"]
    ].max()

    def transitions(
        table: pd.DataFrame, left: str, right: str
    ) -> dict[str, Any]:
        categories = {
            "both_no": (table[left].eq(0) & table[right].eq(0)),
            "p5_yes_fusion_no": (
                table[left].eq(1) & table[right].eq(0)
            ),
            "p5_no_fusion_yes": (
                table[left].eq(0) & table[right].eq(1)
            ),
            "both_yes": (table[left].eq(1) & table[right].eq(1)),
        }
        return {
            name: {
                "events": int(mask.sum()),
                "event_ids": table.index[mask].astype(str).tolist(),
            }
            for name, mask in categories.items()
        }

    return {
        "all_negative_event_any_fp_transitions": transitions(
            null_group, "p5_prediction", "fusion_prediction"
        ),
        "positive_event_true_detection_transitions": transitions(
            positive_group, "p5_true_detection", "fusion_true_detection"
        ),
    }


def row_transition_rates_by_group(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    baseline_correct: np.ndarray,
    candidate_correct: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    target = frame["label"].to_numpy(dtype=np.int64).astype(bool)
    corrected = ~baseline_correct & candidate_correct
    introduced = baseline_correct & ~candidate_correct
    output: dict[str, list[dict[str, Any]]] = {}
    for column in group_columns:
        records: list[dict[str, Any]] = []
        values = frame[column].astype(str)
        for value in sorted(values.unique()):
            mask = values.eq(value).to_numpy()
            baseline_errors = int((mask & ~baseline_correct).sum())
            baseline_correct_rows = int((mask & baseline_correct).sum())
            corrected_mask = mask & corrected
            introduced_mask = mask & introduced
            records.append(
                {
                    "bin": value,
                    "rows": int(mask.sum()),
                    "events": int(frame.loc[mask, "event_id"].nunique()),
                    "baseline_errors": baseline_errors,
                    "corrected_errors": int(corrected_mask.sum()),
                    "correction_rate_among_baseline_errors": (
                        float(corrected_mask.sum() / baseline_errors)
                        if baseline_errors
                        else None
                    ),
                    "baseline_correct_rows": baseline_correct_rows,
                    "introduced_errors": int(introduced_mask.sum()),
                    "introduction_rate_among_baseline_correct": (
                        float(introduced_mask.sum() / baseline_correct_rows)
                        if baseline_correct_rows
                        else None
                    ),
                    "net_correct_rows": int(
                        corrected_mask.sum() - introduced_mask.sum()
                    ),
                    "corrected_false_negatives": int(
                        (corrected_mask & target).sum()
                    ),
                    "corrected_false_positives": int(
                        (corrected_mask & ~target).sum()
                    ),
                    "introduced_false_negatives": int(
                        (introduced_mask & target).sum()
                    ),
                    "introduced_false_positives": int(
                        (introduced_mask & ~target).sum()
                    ),
                }
            )
        output[column] = records
    return output


def markdown_report(payload: Mapping[str, Any]) -> str:
    overall = payload["overall_metrics"]
    error = payload["error_audit"]["p5_to_fusion"]
    null = payload["error_audit"]["event_transitions"][
        "all_negative_event_any_fp_transitions"
    ]
    positive = payload["error_audit"]["event_transitions"][
        "positive_event_true_detection_transitions"
    ]

    def get_bin(group: str, name: str) -> Mapping[str, Any]:
        for record in payload["subgroups"][group]:
            if record["bin"] == name:
                return record
        raise KeyError(f"Missing mechanism bin {group}/{name}")

    history_four = get_bin("unique_history_count", "4")
    history_five = get_bin("unique_history_count", "5")
    lag_mid = get_bin("nearest_history_lag_bin", "03_17_24d")
    lag_late = get_bin("nearest_history_lag_bin", "05_gt_32d")
    annual_short = get_bin("maximum_history_lag_bin", "01_le_350d")
    annual_nominal = get_bin("maximum_history_lag_bin", "02_351_365d")
    degraded_history = get_bin("history_quality_bin", "some_degraded")
    degraded_t0 = get_bin("t0_quality_bin", "degraded")
    missing_prev2 = get_bin(
        "role_pattern", "prev1+prev3+seasonal+year"
    )
    missing_prev3 = get_bin(
        "role_pattern", "prev1+prev2+seasonal+year"
    )
    physical_audit = payload["optional_csv_physical_columns"]
    if physical_audit["none_available"]:
        physical_note = (
            "The development CSV contains no plume-size, plume-area, "
            "concentration, enhancement, flux, or emission-rate column, so "
            "no physical-magnitude bin was fabricated."
        )
    else:
        physical_note = (
            "Physical-variable quartiles were added for: "
            + ", ".join(physical_audit["matched_columns"])
            + "."
        )

    rows: list[tuple[float, str, str, int, int]] = []
    for group_name, bins in payload["subgroups"].items():
        for record in bins:
            delta = record["fusion_minus_p5"]["event_balanced_ap"]
            p5_ap = record["systems"]["p5"]["event_balanced_ap"]
            if delta is None or p5_ap is None:
                continue
            rows.append(
                (
                    float(delta),
                    group_name,
                    record["bin"],
                    int(record["systems"]["p5"]["rows"]),
                    int(record["systems"]["p5"]["events"]),
                )
            )
    rows.sort(reverse=True)
    best = rows[:6]
    worst = list(reversed(rows[-6:]))

    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.6f}"

    lines = [
        "# L89 P5/D1 fixed-fusion mechanism audit",
        "",
        "This is an exploratory, development-only analysis. Bins were fixed "
        "from acquisition semantics or unlabeled quartiles; no cut point or "
        "threshold was selected from subgroup performance.",
        "",
        "## Overall fixed-threshold metrics",
        "",
        "| system | AP | AUC | macro-F1 | positive F1 | null FP mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("p5", "d1", "fusion"):
        value = overall[name]
        lines.append(
            f"| {name} | {fmt(value['event_balanced_ap'])} | "
            f"{fmt(value['event_balanced_auc'])} | "
            f"{fmt(value['event_balanced_macro_f1_fixed_global_threshold'])} | "
            f"{fmt(value['event_balanced_positive_f1_fixed_global_threshold'])} | "
            f"{fmt(value['all_negative_fp_mass'])} |"
        )
    lines.extend(
        [
            "",
            "## P5 → fusion row errors",
            "",
            f"- Corrected P5 errors: `{error['corrected_p5_error']['rows']}` "
            f"rows across `{error['corrected_p5_error']['events']}` events.",
            f"- New fusion errors: `{error['introduced_fusion_error']['rows']}` "
            f"rows across `{error['introduced_fusion_error']['events']}` events.",
            f"- Corrected false negatives: "
            f"`{error['corrected_false_negative']['rows']}`; corrected false "
            f"positives: `{error['corrected_false_positive']['rows']}`.",
            f"- New false negatives: "
            f"`{error['introduced_false_negative']['rows']}`; new false "
            f"positives: `{error['introduced_false_positive']['rows']}`.",
            f"- Of D1's P5-error corrections, fusion retains "
            f"`{payload['error_audit']['late_fusion_filtering']['p5_errors']['d1_and_fusion_correct']}` "
            f"and declines "
            f"`{payload['error_audit']['late_fusion_filtering']['p5_errors']['d1_correct_fusion_wrong']}`.",
            f"- Of D1's harmful overrides on P5-correct rows, fusion filters "
            f"`{payload['error_audit']['late_fusion_filtering']['p5_correct_rows']['d1_wrong_fusion_correct']}` "
            f"and retains "
            f"`{payload['error_audit']['late_fusion_filtering']['p5_correct_rows']['d1_and_fusion_wrong']}`.",
            "",
            "At canonical-event level:",
            "",
            f"- all-negative events whose any-FP is removed: "
            f"`{null['p5_yes_fusion_no']['events']}`;",
            f"- all-negative events receiving a new any-FP: "
            f"`{null['p5_no_fusion_yes']['events']}`;",
            f"- positive events newly detected: "
            f"`{positive['p5_no_fusion_yes']['events']}`;",
            f"- positive events whose detection is lost: "
            f"`{positive['p5_yes_fusion_no']['events']}`.",
            "- Therefore the lower all-negative FP mass comes from fewer "
            "false-positive rows inside already affected events; it does not "
            "yet eliminate false-positive events.",
            "",
            "## Mechanism localization",
            "",
            f"- Fusion removes `{overall['p5']['negative_row_fp_count'] - overall['fusion']['negative_row_fp_count']}` "
            "row-level false positives while losing "
            f"`{error['introduced_false_negative']['rows'] - error['corrected_false_negative']['rows']}` "
            "net true-positive rows. Its gain is therefore mainly improved "
            "precision/ranking, not broader event recall.",
            f"- With four unique histories, AP changes by "
            f"`{history_four['fusion_minus_p5']['event_balanced_ap']:+.6f}`; "
            f"with all five, by "
            f"`{history_five['fusion_minus_p5']['event_balanced_ap']:+.6f}`. "
            "The temporal path is especially useful under one suppressed or "
            "duplicated visit, although most net row corrections still occur "
            "in the much larger five-history group.",
            f"- A nearest usable history at 17–24 days has AP delta "
            f"`{lag_mid['fusion_minus_p5']['event_balanced_ap']:+.6f}`, "
            f"whereas >32 days has "
            f"`{lag_late['fusion_minus_p5']['event_balanced_ap']:+.6f}`.",
            f"- Nominal annual coverage (351–365 days) has AP delta "
            f"`{annual_nominal['fusion_minus_p5']['event_balanced_ap']:+.6f}`; "
            f"truncated coverage (≤350 days) has "
            f"`{annual_short['fusion_minus_p5']['event_balanced_ap']:+.6f}`. "
            "This is consistent with needing both a recent comparator and a "
            "genuine seasonal/year reference.",
            f"- Degraded historical quality has AP delta "
            f"`{degraded_history['fusion_minus_p5']['event_balanced_ap']:+.6f}` "
            f"over `{degraded_history['systems']['fusion']['rows']}` rows, "
            "consistent with the quality gate down-weighting a bad visit.",
            f"- Missing `prev2` has AP delta "
            f"`{missing_prev2['fusion_minus_p5']['event_balanced_ap']:+.6f}`; "
            f"missing `prev3` has "
            f"`{missing_prev3['fusion_minus_p5']['event_balanced_ap']:+.6f}`. "
            "Role identity matters; history count alone is insufficient.",
            f"- The degraded-t0 bin contains only "
            f"`{degraded_t0['systems']['fusion']['positive_rows']}` positive "
            f"rows out of `{degraded_t0['systems']['fusion']['rows']}`. Its "
            f"AP delta `{degraded_t0['fusion_minus_p5']['event_balanced_ap']:+.6f}` "
            "is too sparse for a mechanism claim.",
            "",
            "## Largest subgroup AP gains",
            "",
            "| grouping | bin | rows | events | fusion − P5 AP |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for delta, group, value, row_count, events in best:
        lines.append(
            f"| {group} | {value} | {row_count} | {events} | {delta:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Largest subgroup AP losses",
            "",
            "| grouping | bin | rows | events | fusion − P5 AP |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for delta, group, value, row_count, events in worst:
        lines.append(
            f"| {group} | {value} | {row_count} | {events} | {delta:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Reading boundary",
            "",
            "- No test, sealed, or holdout path was read.",
            "- Thresholds are the already selected global development "
            "thresholds and were not refit inside bins.",
            "- Small-bin deltas are descriptive and have no multiplicity "
            "correction or confidence interval.",
            f"- {physical_note}",
            "",
            "See the companion JSON for every bin, exact counts, provenance, "
            "and representative corrected/introduced rows.",
            "",
        ]
    )
    return "\n".join(lines)


def command_audit(args: argparse.Namespace) -> None:
    paths = {
        "cache": Path(args.dev_cache).expanduser().resolve(),
        "csv": Path(args.dev_csv).expanduser().resolve(),
        "p5": Path(args.p5_predictions).expanduser().resolve(),
        "d1": Path(args.d1_predictions).expanduser().resolve(),
        "fusion": Path(args.fusion_predictions).expanduser().resolve(),
        "output_json": Path(args.output_json).expanduser().resolve(),
        "output_markdown": Path(args.output_markdown).expanduser().resolve(),
    }
    for name, path in paths.items():
        tempo.assert_development_path(path, purpose=f"mechanism audit {name}")
    for name in ("output_json", "output_markdown"):
        if paths[name].exists() and not args.overwrite:
            raise FileExistsError(paths[name])

    cache = cache_runner.torch_load_trusted(paths["cache"])
    if not isinstance(cache, Mapping):
        raise ValueError("Development cache is not a mapping.")
    cache_runner.validate_cache_payload(
        cache, path=paths["cache"], expected_split="val"
    )
    indices = cache_runner.select_usable_rows(cache)
    index_list = indices.tolist()
    roles = [str(value) for value in cache["role_names"]]
    t0_index = int(cache["t0_index"])
    history_indices = [
        index for index in range(len(roles)) if index != t0_index
    ]
    unique = cache["unique_mask"][indices].bool().numpy()
    delta = cache["delta_days"][indices].float().numpy()
    valid_fraction = cache["valid_fraction"][indices].float().numpy()
    labels = cache["labels"][indices].long().numpy()
    event_ids = [str(cache["event_ids"][index]) for index in index_list]
    ids = [str(cache["ids"][index]) for index in index_list]
    plume_ids = [str(cache["plume_ids"][index]) for index in index_list]
    event_positive_lookup = (
        pd.DataFrame({"event_id": event_ids, "label": labels})
        .groupby("event_id", sort=True)["label"]
        .max()
        .to_dict()
    )

    frame = pd.DataFrame(
        {
            "id": ids,
            "plume_id": plume_ids,
            "event_id": event_ids,
            "label": labels,
            "event_kind": [
                (
                    "positive_event"
                    if int(event_positive_lookup[event]) == 1
                    else "all_negative_event"
                )
                for event in event_ids
            ],
        }
    )
    history_unique = unique[:, history_indices]
    frame["unique_history_count"] = history_unique.sum(axis=1).astype(int).astype(str)
    nearest: list[float] = []
    maximum: list[float] = []
    history_quality: list[float] = []
    role_patterns: list[str] = []
    for row in range(len(frame)):
        valid_history = history_unique[row]
        if valid_history.any():
            lag = np.abs(delta[row, history_indices][valid_history])
            quality = valid_fraction[row, history_indices][valid_history]
            nearest.append(float(lag.min()))
            maximum.append(float(lag.max()))
            history_quality.append(float(quality.mean()))
            role_patterns.append(
                "+".join(
                    roles[index]
                    for index, present in zip(history_indices, valid_history)
                    if present
                )
            )
        else:
            nearest.append(float("nan"))
            maximum.append(float("nan"))
            history_quality.append(float("nan"))
            role_patterns.append("none")
    frame["nearest_history_lag_days"] = nearest
    frame["maximum_history_lag_days"] = maximum
    frame["nearest_history_lag_bin"] = [
        semantic_lag_bin(value, nearest=True) for value in nearest
    ]
    frame["maximum_history_lag_bin"] = [
        semantic_lag_bin(value, nearest=False) for value in maximum
    ]
    frame["t0_valid_fraction"] = valid_fraction[:, t0_index]
    frame["mean_unique_history_valid_fraction"] = history_quality
    frame["t0_quality_bin"] = np.where(
        frame["t0_valid_fraction"].to_numpy() >= 1.0 - 1e-7,
        "perfect",
        "degraded",
    )
    frame["history_quality_bin"] = np.where(
        frame["mean_unique_history_valid_fraction"].to_numpy() >= 1.0 - 1e-7,
        "all_perfect",
        "some_degraded",
    )
    frame["role_pattern"] = role_patterns
    frame["row_label"] = np.where(labels == 1, "positive_row", "negative_row")

    prediction_frames: dict[str, pd.DataFrame] = {}
    for name in ("p5", "d1", "fusion"):
        prediction = tempo.load_prediction_table(paths[name])
        if prediction["id"].astype(str).tolist() != ids:
            raise ValueError(f"{name} prediction IDs differ from cache.")
        if prediction["event_id"].astype(str).tolist() != event_ids:
            raise ValueError(f"{name} canonical event IDs differ from cache.")
        if not np.array_equal(
            prediction["label"].to_numpy(dtype=np.int64), labels
        ):
            raise ValueError(f"{name} labels differ from cache.")
        prediction_frames[name] = prediction
    probabilities = {
        name: prediction["probability"].to_numpy(dtype=np.float64)
        for name, prediction in prediction_frames.items()
    }
    expected_fusion = 1.0 / (
        1.0
        + np.exp(
            -0.5 * tempo.safe_logit(probabilities["p5"])
            -0.5 * tempo.safe_logit(probabilities["d1"])
        )
    )
    fusion_replay_error = float(
        np.max(np.abs(expected_fusion - probabilities["fusion"]))
    )
    if fusion_replay_error > 1e-10:
        raise ValueError(
            f"Fixed 0.5 fusion replay differs by {fusion_replay_error:.3e}."
        )
    thresholds = {
        "p5": float(args.p5_threshold),
        "d1": float(args.d1_threshold),
        "fusion": float(args.fusion_threshold),
    }

    source_csv = pd.read_csv(paths["csv"])
    if source_csv["id"].astype(str).tolist() != ids:
        raise ValueError("Development CSV row IDs differ from cache.")
    optional_columns: dict[str, Any] = {}
    for column in source_csv.columns:
        if column == "plume_id" or not OPTIONAL_PHYSICAL_RE.search(column):
            continue
        binned, audit = physical_quartile_bins(source_csv[column])
        optional_columns[column] = audit
        if bool(audit["usable"]):
            output_column = f"csv_{column}_quartile"
            frame[output_column] = binned.astype(str)
            optional_columns[column]["output_group_column"] = output_column

    group_columns = (
        "unique_history_count",
        "nearest_history_lag_bin",
        "maximum_history_lag_bin",
        "t0_quality_bin",
        "history_quality_bin",
        "event_kind",
        "row_label",
        "role_pattern",
    ) + tuple(
        str(value["output_group_column"])
        for value in optional_columns.values()
        if value.get("usable")
    )
    subgroups = {
        column: subgroup_metrics(
            frame,
            group_column=column,
            probabilities=probabilities,
            thresholds=thresholds,
        )
        for column in group_columns
    }
    overall = {
        name: fixed_threshold_metrics(
            frame, probability, threshold=thresholds[name]
        )
        for name, probability in probabilities.items()
    }

    p5_prediction = probabilities["p5"] >= thresholds["p5"]
    d1_prediction = probabilities["d1"] >= thresholds["d1"]
    fusion_prediction = probabilities["fusion"] >= thresholds["fusion"]
    target = labels.astype(bool)
    p5_correct = p5_prediction == target
    d1_correct = d1_prediction == target
    fusion_correct = fusion_prediction == target
    p5_to_fusion_masks = {
        "corrected_p5_error": ~p5_correct & fusion_correct,
        "introduced_fusion_error": p5_correct & ~fusion_correct,
        "both_wrong": ~p5_correct & ~fusion_correct,
        "both_correct": p5_correct & fusion_correct,
        "corrected_false_negative": (
            target & ~p5_prediction & fusion_prediction
        ),
        "corrected_false_positive": (
            ~target & p5_prediction & ~fusion_prediction
        ),
        "introduced_false_negative": (
            target & p5_prediction & ~fusion_prediction
        ),
        "introduced_false_positive": (
            ~target & ~p5_prediction & fusion_prediction
        ),
    }
    d1_vs_p5_masks = {
        "d1_corrects_p5": ~p5_correct & d1_correct,
        "d1_breaks_p5": p5_correct & ~d1_correct,
        "both_wrong": ~p5_correct & ~d1_correct,
        "both_correct": p5_correct & d1_correct,
    }
    summary_kwargs = {
        "p5_probability": probabilities["p5"],
        "d1_probability": probabilities["d1"],
        "fusion_probability": probabilities["fusion"],
        "p5_prediction": p5_prediction,
        "d1_prediction": d1_prediction,
        "fusion_prediction": fusion_prediction,
    }
    error_audit = {
        "p5_to_fusion": {
            name: category_summary(frame, mask, **summary_kwargs)
            for name, mask in p5_to_fusion_masks.items()
        },
        "d1_vs_p5": {
            name: category_summary(frame, mask, **summary_kwargs)
            for name, mask in d1_vs_p5_masks.items()
        },
        "event_transitions": event_transition_audit(
            frame, p5_prediction, fusion_prediction
        ),
        "p5_to_fusion_transition_rates_by_group": row_transition_rates_by_group(
            frame,
            group_columns=group_columns,
            baseline_correct=p5_correct,
            candidate_correct=fusion_correct,
        ),
        "p5_to_d1_transition_rates_by_group": row_transition_rates_by_group(
            frame,
            group_columns=group_columns,
            baseline_correct=p5_correct,
            candidate_correct=d1_correct,
        ),
        "late_fusion_filtering": {
            "p5_errors": {
                "rows": int((~p5_correct).sum()),
                "d1_corrects": int((~p5_correct & d1_correct).sum()),
                "fusion_corrects": int((~p5_correct & fusion_correct).sum()),
                "d1_and_fusion_correct": int(
                    (~p5_correct & d1_correct & fusion_correct).sum()
                ),
                "d1_correct_fusion_wrong": int(
                    (~p5_correct & d1_correct & ~fusion_correct).sum()
                ),
                "d1_wrong_fusion_correct": int(
                    (~p5_correct & ~d1_correct & fusion_correct).sum()
                ),
            },
            "p5_correct_rows": {
                "rows": int(p5_correct.sum()),
                "d1_breaks": int((p5_correct & ~d1_correct).sum()),
                "fusion_breaks": int((p5_correct & ~fusion_correct).sum()),
                "d1_and_fusion_wrong": int(
                    (p5_correct & ~d1_correct & ~fusion_correct).sum()
                ),
                "d1_wrong_fusion_correct": int(
                    (p5_correct & ~d1_correct & fusion_correct).sum()
                ),
                "d1_correct_fusion_wrong": int(
                    (p5_correct & d1_correct & ~fusion_correct).sum()
                ),
            },
        },
    }

    payload = {
        "script_version": SCRIPT_VERSION,
        "audit_type": "fixed-semantic-bin-development-mechanism-audit",
        "exploratory": True,
        "canonical_seed": int(args.seed),
        "binning_contract": {
            "cutpoints_selected_from_performance": False,
            "threshold_refit_within_bin": False,
            "unique_history_count": "integer count after duplicate suppression",
            "nearest_history_lag_days": "<=7, 8-16, 17-24, 25-32, >32",
            "maximum_history_lag_days": "<=350, 351-365, >365",
            "quality": (
                "perfect (=1 within 1e-7) versus degraded; input-validity "
                "semantics, not metric search"
            ),
            "optional_csv_physical_variables": (
                "unlabeled quartiles only when a matching numeric column has "
                "at least 20 finite and 4 unique values"
            ),
        },
        "thresholds": thresholds,
        "overall_metrics": overall,
        "subgroups": subgroups,
        "error_audit": error_audit,
        "optional_csv_physical_columns": {
            "matched_columns": sorted(optional_columns),
            "details": optional_columns,
            "none_available": not bool(optional_columns),
            "source_columns": [str(value) for value in source_csv.columns],
        },
        "fusion_replay": {
            "formula": "sigmoid(0.5*logit(P5)+0.5*logit(D1))",
            "maximum_abs_probability_error": fusion_replay_error,
            "verified": True,
        },
        "data": {
            "rows": int(len(frame)),
            "events": int(frame["event_id"].nunique()),
            "positive_events": int(
                frame.loc[
                    frame["event_kind"].eq("positive_event"), "event_id"
                ].nunique()
            ),
            "all_negative_events": int(
                frame.loc[
                    frame["event_kind"].eq("all_negative_event"), "event_id"
                ].nunique()
            ),
            "role_names": roles,
            "cache": str(paths["cache"]),
            "cache_sha256": cache_runner.sha256_file(paths["cache"]),
            "cache_feature_sha256": str(cache["feature_sha256"]),
            "csv": str(paths["csv"]),
            "csv_sha256": cache_runner.sha256_file(paths["csv"]),
        },
        "prediction_provenance": {
            name: {
                "path": str(paths[name]),
                "sha256": cache_runner.sha256_file(paths[name]),
            }
            for name in ("p5", "d1", "fusion")
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
                "output_json": str(paths["output_json"]),
                "output_markdown": str(paths["output_markdown"]),
                "overall_metrics": overall,
                "error_counts": {
                    name: value["rows"]
                    for name, value in error_audit["p5_to_fusion"].items()
                },
                "optional_csv_physical_columns": payload[
                    "optional_csv_physical_columns"
                ],
                "test_or_sealed_or_holdout_read": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-cache", required=True)
    parser.add_argument("--dev-csv", required=True)
    parser.add_argument("--p5-predictions", required=True)
    parser.add_argument("--d1-predictions", required=True)
    parser.add_argument("--fusion-predictions", required=True)
    parser.add_argument("--p5-threshold", type=float, required=True)
    parser.add_argument("--d1-threshold", type=float, required=True)
    parser.add_argument("--fusion-threshold", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    command_audit(build_parser().parse_args())


if __name__ == "__main__":
    main()
