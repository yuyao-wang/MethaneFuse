#!/usr/bin/env python3
"""Summarize paired predictions from build_s2_brightness_probe.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def finite_metric(function, labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(function(labels, scores))


def paired_test(delta: np.ndarray) -> dict[str, float | int | None]:
    nonzero = delta[np.abs(delta) > 1e-12]
    p_value = None
    if len(nonzero):
        p_value = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
    return {
        "count": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "fraction_increased": float((delta > 0).mean()),
        "fraction_decreased": float((delta < 0).mean()),
        "wilcoxon_two_sided_p": p_value,
    }


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.probe_manifest, low_memory=False)
    predictions = pd.read_csv(args.predictions, low_memory=False)
    required_manifest = {
        "id",
        "parent_id",
        "label",
        "brightness_mode",
        "brightness_factor",
    }
    required_predictions = {"id", "positive_probability"}
    if missing := sorted(required_manifest - set(manifest.columns)):
        raise ValueError(f"probe manifest missing columns: {missing}")
    if missing := sorted(required_predictions - set(predictions.columns)):
        raise ValueError(f"predictions missing columns: {missing}")
    frame = manifest.merge(
        predictions[["id", "positive_probability"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    if frame["positive_probability"].isna().any():
        missing_ids = frame.loc[frame["positive_probability"].isna(), "id"].tolist()
        raise ValueError(f"missing predictions for {len(missing_ids)} rows")

    summaries: list[dict[str, object]] = []
    for mode, group in frame.groupby("brightness_mode", sort=False):
        labels = group["label"].astype(int).to_numpy()
        scores = group["positive_probability"].astype(float).to_numpy()
        predictions_binary = (scores >= 0.5).astype(int)
        positives = labels == 1
        negatives = labels == 0
        summaries.append(
            {
                "brightness_mode": mode,
                "count": int(len(group)),
                "positive_mean": float(scores[positives].mean()),
                "negative_mean": float(scores[negatives].mean()),
                "recall_at_0p5": float(predictions_binary[positives].mean()),
                "fpr_at_0p5": float(predictions_binary[negatives].mean()),
                "auroc": finite_metric(roc_auc_score, labels, scores),
                "average_precision": finite_metric(
                    average_precision_score, labels, scores
                ),
            }
        )
    summary_frame = pd.DataFrame.from_records(summaries)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.output_csv, index=False)

    baseline = (
        frame[frame["brightness_mode"] == "scale_1p00"]
        .set_index("parent_id")["positive_probability"]
        .astype(float)
    )
    paired: dict[str, object] = {}
    for mode, group in frame.groupby("brightness_mode", sort=False):
        values = group.set_index("parent_id")["positive_probability"].astype(float)
        aligned = pd.concat(
            [baseline.rename("baseline"), values.rename("variant")], axis=1, join="inner"
        )
        paired[str(mode)] = {
            "all": paired_test((aligned["variant"] - aligned["baseline"]).to_numpy())
        }
        labels_by_parent = group.set_index("parent_id")["label"].astype(int)
        for label in (0, 1):
            parent_ids = labels_by_parent[labels_by_parent == label].index
            subset = aligned.loc[aligned.index.intersection(parent_ids)]
            paired[str(mode)][f"label_{label}"] = paired_test(
                (subset["variant"] - subset["baseline"]).to_numpy()
            )

    scalar = frame[frame["brightness_factor"].notna()].copy()
    scalar_trends: list[dict[str, object]] = []
    for parent_id, group in scalar.groupby("parent_id", sort=False):
        group = group.sort_values("brightness_factor")
        factors = group["brightness_factor"].astype(float).to_numpy()
        scores = group["positive_probability"].astype(float).to_numpy()
        slope = float(np.polyfit(factors, scores, deg=1)[0])
        rho = float(spearmanr(factors, scores).statistic)
        scalar_trends.append(
            {
                "parent_id": parent_id,
                "label": int(group["label"].iloc[0]),
                "slope_probability_per_scale_unit": slope,
                "spearman_rho": rho,
            }
        )
    trend_frame = pd.DataFrame.from_records(scalar_trends)
    trend_summary: dict[str, object] = {}
    for name, group in [
        ("all", trend_frame),
        ("label_0", trend_frame[trend_frame["label"] == 0]),
        ("label_1", trend_frame[trend_frame["label"] == 1]),
    ]:
        slopes = group["slope_probability_per_scale_unit"].to_numpy()
        rhos = group["spearman_rho"].to_numpy()
        trend_summary[name] = {
            "count": int(len(group)),
            "median_slope": float(np.median(slopes)),
            "mean_slope": float(np.mean(slopes)),
            "fraction_positive_slope": float((slopes > 0).mean()),
            "median_spearman_rho": float(np.median(rhos)),
        }

    result = {
        "probe_manifest": str(args.probe_manifest.resolve()),
        "predictions": str(args.predictions.resolve()),
        "rows": int(len(frame)),
        "per_mode": summaries,
        "paired_vs_scale_1p00": paired,
        "scalar_brightness_trends": {
            "summary": trend_summary,
            "per_parent": scalar_trends,
        },
        "interpretation_rule": {
            "brightness_supported": (
                "Scores rise monotonically for most parents under global scaling and/or "
                "fall consistently when matched to the training-negative centroid."
            ),
            "brightness_not_supported": (
                "Paired score changes are near zero or have inconsistent signs across parents."
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_name(args.output_json.name + ".part")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_json)
    print(json.dumps({"rows": len(frame), "modes": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
