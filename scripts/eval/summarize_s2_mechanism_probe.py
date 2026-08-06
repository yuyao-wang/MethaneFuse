#!/usr/bin/env python3
"""Summarize paired model scores for the S2 mechanism probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score-column", default="positive_probability")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def safe_wilcoxon(values: np.ndarray) -> float | None:
    if len(values) == 0 or np.allclose(values, 0.0):
        return None
    return float(wilcoxon(values, alternative="two-sided").pvalue)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.probe_manifest, low_memory=False)
    predictions = pd.read_csv(args.predictions, low_memory=False)
    needed = {"id", "label", "parent_id", "probe_mode"}
    missing = sorted(needed - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest missing {missing}")
    if args.score_column not in predictions.columns:
        raise ValueError(f"predictions missing score column {args.score_column!r}")
    predictions = predictions.rename(columns={args.score_column: "positive_probability"})
    merged = manifest[["id", "label", "parent_id", "probe_mode"]].merge(
        predictions[["id", "positive_probability"]], on="id", how="left", validate="one_to_one"
    )
    if merged["positive_probability"].isna().any():
        raise ValueError("some probe rows have no prediction")

    original = (
        merged[merged["probe_mode"] == "original"]
        .set_index("parent_id")["positive_probability"]
        .rename("original_probability")
    )
    merged = merged.join(original, on="parent_id", validate="many_to_one")
    merged["delta_vs_original"] = merged["positive_probability"] - merged["original_probability"]

    per_mode: list[dict[str, object]] = []
    paired: dict[str, dict[str, object]] = {}
    for mode, group in merged.groupby("probe_mode", sort=False):
        labels = group["label"].astype(int).to_numpy()
        scores = group["positive_probability"].to_numpy(dtype=float)
        record: dict[str, object] = {
            "probe_mode": mode,
            "count": len(group),
            "positive_mean": float(scores[labels == 1].mean()),
            "negative_mean": float(scores[labels == 0].mean()),
            "fpr_native_argmax": float(np.mean(scores[labels == 0] > 0.5)),
            "recall_native_argmax": float(np.mean(scores[labels == 1] > 0.5)),
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
        }
        per_mode.append(record)
        paired[mode] = {}
        for label_name, subset in (
            ("all", group),
            ("label_0", group[group["label"].astype(int) == 0]),
            ("label_1", group[group["label"].astype(int) == 1]),
        ):
            delta = subset["delta_vs_original"].to_numpy(dtype=float)
            paired[mode][label_name] = {
                "count": len(delta),
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "fraction_decreased": float(np.mean(delta < 0)),
                "fraction_increased": float(np.mean(delta > 0)),
                "wilcoxon_two_sided_p": safe_wilcoxon(delta),
            }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_mode).to_csv(args.output_csv, index=False)
    args.output_json.write_text(
        json.dumps(
            {
                "rows": len(merged),
                "probe_manifest": str(args.probe_manifest.resolve()),
                "predictions": str(args.predictions.resolve()),
                "decision_rule": "checkpoint-native two-class argmax; no fitted threshold stored",
                "per_mode": per_mode,
                "paired_vs_original": paired,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(merged), "modes": len(per_mode)}, indent=2))


if __name__ == "__main__":
    main()
