#!/usr/bin/env python3
"""Development-only audit of whether two prediction streams are complementary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tempo_l89_global import assert_development_path, metric_bundle


KEYS = ("id", "plume_id", "event_id", "label")


def load_predictions(path: Path, suffix: str) -> pd.DataFrame:
    assert_development_path(path, purpose=f"{suffix} predictions")
    frame = pd.read_csv(path, dtype={"id": str})
    missing = sorted(set((*KEYS, "probability")) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path} has duplicate ids")
    output = frame.loc[:, [*KEYS, "probability"]].copy()
    output = output.rename(columns={"probability": f"probability_{suffix}"})
    return output


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(probability) - np.log1p(-probability)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=41)
    parser.add_argument("--fixed-alphas", default="0.35,0.475")
    args = parser.parse_args()
    assert_development_path(args.output, purpose="fusion audit output")
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")

    left = load_predictions(args.left.resolve(), "left")
    right = load_predictions(args.right.resolve(), "right")
    merged = left.merge(
        right,
        on=list(KEYS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Prediction streams do not contain the same rows")

    label = merged["label"].to_numpy(dtype=np.int64)
    event_ids = merged["event_id"].astype(str).tolist()
    p_left = merged["probability_left"].to_numpy(dtype=np.float64)
    p_right = merged["probability_right"].to_numpy(dtype=np.float64)
    z_left = clipped_logit(p_left)
    z_right = clipped_logit(p_right)

    records: list[dict[str, object]] = []
    for mode in ("probability", "logit"):
        for alpha in np.linspace(0.0, 1.0, int(args.steps)):
            if mode == "probability":
                probability = (1.0 - alpha) * p_left + alpha * p_right
            else:
                logit = (1.0 - alpha) * z_left + alpha * z_right
                probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
            metrics = metric_bundle(label, probability, event_ids)
            records.append(
                {
                    "mode": mode,
                    "alpha_right": float(alpha),
                    "metrics": metrics,
                }
            )

    fixed_alphas = [
        float(value.strip())
        for value in str(args.fixed_alphas).split(",")
        if value.strip()
    ]
    if any(not 0.0 <= value <= 1.0 for value in fixed_alphas):
        raise ValueError("--fixed-alphas values must be within [0,1]")
    fixed: list[dict[str, object]] = []
    for mode in ("probability", "logit"):
        for alpha in fixed_alphas:
            if mode == "probability":
                probability = (1.0 - alpha) * p_left + alpha * p_right
            else:
                logit = (1.0 - alpha) * z_left + alpha * z_right
                probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
            fixed.append(
                {
                    "mode": mode,
                    "alpha_right": float(alpha),
                    "metrics": metric_bundle(label, probability, event_ids),
                }
            )

    def key(record: dict[str, object]) -> tuple[float, float, float]:
        metrics = record["metrics"]
        assert isinstance(metrics, dict)
        return (
            float(metrics["event_balanced_ap"]),
            float(metrics["event_balanced_macro_f1_selected"]),
            -float(metrics["all_negative_fp_mass"]),
        )

    best_ap = max(records, key=key)
    best_macro = max(
        records,
        key=lambda record: (
            float(record["metrics"]["event_balanced_macro_f1_selected"]),
            float(record["metrics"]["event_balanced_ap"]),
            -float(record["metrics"]["all_negative_fp_mass"]),
        ),
    )
    endpoints = [
        record
        for record in records
        if math.isclose(float(record["alpha_right"]), 0.0)
        or math.isclose(float(record["alpha_right"]), 1.0)
    ]
    payload = {
        "scope": "development-only exploratory score-complementarity audit",
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "rows": int(len(merged)),
        "events": int(len(set(event_ids))),
        "steps": int(args.steps),
        "best_event_balanced_ap": best_ap,
        "best_event_balanced_macro_f1": best_macro,
        "fixed_alpha_evaluations": fixed,
        "endpoints": endpoints,
        "test_or_sealed_or_holdout_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
