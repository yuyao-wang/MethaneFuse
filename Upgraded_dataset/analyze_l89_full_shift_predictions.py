#!/usr/bin/env python3
"""Analyze why the full L89 temporal test set is harder than the 2025-10 subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict


TIME_COLUMNS = (
    "t0_image_time",
    "prev1_image_time",
    "prev2_image_time",
    "prev3_image_time",
    "seasonal_image_time",
    "year_image_time",
)


def metrics(frame: pd.DataFrame, threshold: float = 0.5) -> dict[str, float | int]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    probs = frame["pred_prob1"].to_numpy(dtype=np.float64)
    preds = (probs >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    f1_den = 2 * tp + fp + fn
    result: dict[str, float | int] = {
        "count": int(len(frame)),
        "positive": int(labels.sum()),
        "acc": (tp + tn) / len(frame) if len(frame) else float("nan"),
        "f1": 2 * tp / f1_den if f1_den else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    result["auroc"] = (
        float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan")
    )
    return result


def add_features(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    result = frame.copy()
    result["event_time_dt"] = pd.to_datetime(result["event_time"], utc=True)
    result["event_id"] = (
        result["plume_id"].astype(str).str.replace(r"-[A-Za-z0-9]+$", "", regex=True)
        + "|"
        + result["event_time"].astype(str)
    )
    result["period"] = np.where(result["event_time_dt"] <= cutoff, "cutoff", "post_cutoff")
    result["month"] = result["event_time_dt"].dt.strftime("%Y-%m")
    result["lat_bin"] = pd.cut(result["latitude"], [-90, -30, 0, 30, 60, 90], include_lowest=True).astype(str)
    result["lon_bin"] = pd.cut(result["longitude"], [-180, -90, 0, 90, 180], include_lowest=True).astype(str)
    for column in TIME_COLUMNS:
        image_time = pd.to_datetime(result[column], utc=True, errors="coerce")
        result[f"age_{column}_days"] = (result["event_time_dt"] - image_time).dt.total_seconds() / 86400.0
    result["t0_abs_age_days"] = result["age_t0_image_time_days"].abs()
    result["max_abs_time_age_days"] = result[
        [f"age_{column}_days" for column in TIME_COLUMNS]
    ].abs().max(axis=1)
    result["prediction_error"] = (~result["pred_correct"].astype(bool)).astype(np.int64)
    return result


def grouped_metrics(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(column, dropna=False, observed=True):
        row = {column: str(value), **metrics(group)}
        row["mean_prob1"] = float(group["pred_prob1"].mean())
        row["event_time_min"] = str(group["event_time_dt"].min())
        row["latitude_mean"] = float(group["latitude"].mean())
        row["longitude_mean"] = float(group["longitude"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def best_threshold(frame: pd.DataFrame) -> tuple[float, dict[str, float | int]]:
    best = (0.5, metrics(frame, 0.5))
    for threshold in np.linspace(0.05, 0.95, 181):
        candidate = metrics(frame, float(threshold))
        if candidate["f1"] > best[1]["f1"]:
            best = (float(threshold), candidate)
    return best


def cap_event_label_rows(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    ranked = frame.copy()
    hashed = pd.util.hash_pandas_object(
        ranked[["id", "event_id", "label"]],
        index=False,
        hash_key="0123456789abcdef",
    ).astype(np.uint64)
    ranked["_sample_rank"] = hashed ^ np.uint64(seed)
    ranked = ranked.sort_values(["event_id", "label", "_sample_rank"], kind="stable")
    return (
        ranked.groupby(["event_id", "label"], observed=True)
        .head(cap)
        .drop(columns="_sample_rank")
    )


def domain_classifier(frame: pd.DataFrame) -> dict:
    feature_columns = [
        "latitude",
        "longitude",
        "source_x",
        "source_y",
        "label",
        "t0_abs_age_days",
        "max_abs_time_age_days",
        *[f"age_{column}_days" for column in TIME_COLUMNS],
    ]
    features = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).to_numpy(dtype=np.float64)
    target = (frame["period"] == "post_cutoff").to_numpy(dtype=np.int64)
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=20251031,
        n_jobs=-1,
    )
    groups = frame["event_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20251031)
    probabilities = cross_val_predict(
        classifier,
        features,
        target,
        groups=groups,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    classifier.fit(features, target)
    importances = sorted(
        zip(feature_columns, classifier.feature_importances_), key=lambda item: item[1], reverse=True
    )
    return {
        "grouped_cross_validated_auroc": float(roc_auc_score(target, probabilities)),
        "feature_importances": [{"feature": name, "importance": float(value)} for name, value in importances],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cutoff", default="2025-10-31T23:59:59Z")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.predictions_csv, low_memory=False)
    frame = add_features(frame, pd.Timestamp(args.cutoff))

    overall = metrics(frame)
    period_table = grouped_metrics(frame, "period")
    month_table = grouped_metrics(frame, "month")
    event_table = grouped_metrics(frame, "event_id")
    lat_table = grouped_metrics(frame, "lat_bin")
    lon_table = grouped_metrics(frame, "lon_bin")

    full_f1 = float(overall["f1"])
    exclusion_deltas = []
    for event_id, group in frame.groupby("event_id", observed=True):
        without = frame[frame["event_id"] != event_id]
        without_metrics = metrics(without)
        exclusion_deltas.append(
            {
                "event_id": event_id,
                "rows": int(len(group)),
                "errors": int(group["prediction_error"].sum()),
                "positive": int(group["label"].sum()),
                "event_f1": metrics(group)["f1"],
                "full_f1_without_event": without_metrics["f1"],
                "f1_gain_if_excluded": float(without_metrics["f1"]) - full_f1,
                "event_time": str(group["event_time_dt"].min()),
                "latitude": float(group["latitude"].mean()),
                "longitude": float(group["longitude"].mean()),
            }
        )
    exclusion_table = pd.DataFrame(exclusion_deltas).sort_values(
        ["f1_gain_if_excluded", "rows"], ascending=[False, False]
    )

    cutoff_frame = frame[frame["period"] == "cutoff"]
    post_frame = frame[frame["period"] == "post_cutoff"]
    threshold, cutoff_threshold_metrics = best_threshold(cutoff_frame)
    threshold_results = {
        "selected_on": "cutoff",
        "threshold": threshold,
        "cutoff": cutoff_threshold_metrics,
        "post_cutoff": metrics(post_frame, threshold),
        "full": metrics(frame, threshold),
    }
    balance_rows = []
    for cap in (8, 16, 32, 64):
        balanced = cap_event_label_rows(frame, cap, seed=20251031)
        for period, group in (
            ("full", balanced),
            ("cutoff", balanced[balanced["period"] == "cutoff"]),
            ("post_cutoff", balanced[balanced["period"] == "post_cutoff"]),
        ):
            balance_rows.append(
                {
                    "cap_per_event_label": cap,
                    "period": period,
                    **metrics(group),
                }
            )
    balance_table = pd.DataFrame(balance_rows)

    numeric_shift_columns = [
        "latitude",
        "longitude",
        "source_x",
        "source_y",
        "t0_abs_age_days",
        "max_abs_time_age_days",
        *[f"age_{column}_days" for column in TIME_COLUMNS],
    ]
    shift_rows = []
    for column in numeric_shift_columns:
        for period, group in frame.groupby("period", observed=True):
            values = group[column].dropna()
            shift_rows.append(
                {
                    "feature": column,
                    "period": period,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "p05": float(values.quantile(0.05)),
                    "p50": float(values.quantile(0.50)),
                    "p95": float(values.quantile(0.95)),
                }
            )

    period_table.to_csv(output_dir / "metrics_by_period.csv", index=False)
    month_table.to_csv(output_dir / "metrics_by_month.csv", index=False)
    event_table.to_csv(output_dir / "metrics_by_event.csv", index=False)
    exclusion_table.to_csv(output_dir / "event_exclusion_impact.csv", index=False)
    lat_table.to_csv(output_dir / "metrics_by_latitude_bin.csv", index=False)
    lon_table.to_csv(output_dir / "metrics_by_longitude_bin.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(output_dir / "metadata_shift_summary.csv", index=False)
    balance_table.to_csv(output_dir / "event_balance_sensitivity.csv", index=False)

    error_counts = event_table.assign(errors=event_table["fp"] + event_table["fn"]).sort_values(
        "errors", ascending=False
    )
    total_errors = int(error_counts["errors"].sum())
    error_concentration = {
        f"top_{count}_event_error_share": float(error_counts.head(count)["errors"].sum() / total_errors)
        for count in (1, 3, 5, 10, 20)
    }
    summary = {
        "overall": overall,
        "periods": {row["period"]: row for row in period_table.to_dict(orient="records")},
        "threshold_calibration": threshold_results,
        "event_error_concentration": error_concentration,
        "event_balance_cap_16": {
            row["period"]: row
            for row in balance_table[balance_table["cap_per_event_label"] == 16].to_dict(
                orient="records"
            )
        },
        "domain_classifier": domain_classifier(frame),
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
