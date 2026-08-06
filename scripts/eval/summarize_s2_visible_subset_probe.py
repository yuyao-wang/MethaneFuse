#!/usr/bin/env python3
"""Compute exact four-band Shapley attribution from visible subset predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ((0, "B01"), (1, "B02"), (2, "B03"), (3, "B04"))
FULL_MASK = (1 << len(BANDS)) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score-column", default="positive_probability")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def shapley_for_row(scores: dict[int, float]) -> dict[str, float]:
    # v(S) is the score reduction after removing the bands in S.
    baseline = scores[0]
    values = {mask: baseline - score for mask, score in scores.items()}
    n = len(BANDS)
    result: dict[str, float] = {}
    for bit, name in BANDS:
        total = 0.0
        for subset in range(1 << n):
            if subset & (1 << bit):
                continue
            size = bin(subset).count("1")
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            total += weight * (values[subset | (1 << bit)] - values[subset])
        result[name] = float(total)
    return result


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.probe_manifest, low_memory=False)
    predictions = pd.read_csv(args.predictions, low_memory=False)
    if args.score_column not in predictions.columns:
        raise ValueError(f"missing score column {args.score_column!r}")
    merged = manifest[
        ["id", "parent_id", "label", "probe_mode", "removed_visible_mask", "removed_visible_bands"]
    ].merge(
        predictions[["id", args.score_column]], on="id", how="left", validate="one_to_one"
    )
    if merged[args.score_column].isna().any():
        raise ValueError("missing predictions")
    if merged.groupby("parent_id")["removed_visible_mask"].nunique().min() != 16:
        raise ValueError("every parent must contain all 16 subsets")

    wide = merged.pivot(index="parent_id", columns="removed_visible_mask", values=args.score_column)
    labels = merged.drop_duplicates("parent_id").set_index("parent_id")["label"].astype(int)
    rows: list[dict[str, object]] = []
    for parent_id, series in wide.iterrows():
        scores = {int(mask): float(score) for mask, score in series.items()}
        attribution = shapley_for_row(scores)
        row: dict[str, object] = {
            "parent_id": parent_id,
            "label": int(labels.loc[parent_id]),
            "baseline_score": scores[0],
            "all_visible_removed_score": scores[FULL_MASK],
            "total_score_drop": scores[0] - scores[FULL_MASK],
        }
        for bit, name in BANDS:
            row[f"{name}_shapley_score_drop"] = attribution[name]
            row[f"{name}_single_removal_score"] = scores[1 << bit]
            row[f"{name}_single_removal_delta"] = scores[1 << bit] - scores[0]
            only_remaining_mask = FULL_MASK ^ (1 << bit)
            row[f"{name}_only_texture_remaining_score"] = scores[only_remaining_mask]
        rows.append(row)
    attribution_frame = pd.DataFrame(rows)

    group_summary: dict[str, object] = {}
    for group_name, group in (
        ("all", attribution_frame),
        ("label_0", attribution_frame[attribution_frame["label"] == 0]),
        ("label_1", attribution_frame[attribution_frame["label"] == 1]),
    ):
        band_records = []
        for _, name in BANDS:
            shapley = group[f"{name}_shapley_score_drop"].to_numpy(dtype=float)
            band_records.append(
                {
                    "band": name,
                    "mean_shapley_score_drop": float(shapley.mean()),
                    "median_shapley_score_drop": float(np.median(shapley)),
                    "fraction_positive_shapley": float(np.mean(shapley > 0)),
                    "mean_single_removal_score": float(group[f"{name}_single_removal_score"].mean()),
                    "mean_single_removal_delta": float(group[f"{name}_single_removal_delta"].mean()),
                    "mean_only_texture_remaining_score": float(group[f"{name}_only_texture_remaining_score"].mean()),
                }
            )
        band_records.sort(key=lambda row: row["mean_shapley_score_drop"], reverse=True)
        group_summary[group_name] = {
            "count": len(group),
            "baseline_mean": float(group["baseline_score"].mean()),
            "all_visible_removed_mean": float(group["all_visible_removed_score"].mean()),
            "total_score_drop_mean": float(group["total_score_drop"].mean()),
            "bands_ranked_by_shapley": band_records,
            "shapley_sum_mean": float(
                group[[f"{name}_shapley_score_drop" for _, name in BANDS]].sum(axis=1).mean()
            ),
        }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    attribution_frame.to_csv(args.output_csv, index=False)
    args.output_json.write_text(
        json.dumps(
            {
                "rows": len(attribution_frame),
                "probe_manifest": str(args.probe_manifest.resolve()),
                "predictions": str(args.predictions.resolve()),
                "score_column": args.score_column,
                "intervention": "selected t0 visible band spatial texture replaced by its own spatial mean",
                "shapley_value": "exact contribution to positive-score drop across all 16 removal subsets",
                "groups": group_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(group_summary, indent=2))


if __name__ == "__main__":
    main()
