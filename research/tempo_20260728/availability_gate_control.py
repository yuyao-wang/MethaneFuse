#!/usr/bin/env python3
"""Frozen availability-gate control for the legacy360 development split.

Exactly-one-sensor rows use the frozen four-seed R4 ensemble; rows with two or
more available sensors use frozen P0. Each branch retains its own already
locked global threshold. No score weight or threshold is fitted here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.tempo_20260728.tempo_legacy360_global import (  # noqa: E402
    atomic_json,
    guard_development_path,
    sha256_file,
)


SCRIPT_VERSION = "tempo-availability-gate-control-v1"
DEFAULT_AGGREGATE = Path(
    "/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1"
    "/final_r4_4seed_v5/final_aggregate.json"
)
DEFAULT_PREDICTIONS = DEFAULT_AGGREGATE.parent / "final_ensemble_predictions.csv"
DEFAULT_OUTPUT = Path(
    "/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1"
    "/availability_gate_control_v7"
)


def fixed_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=bool)
    positive = target == 1
    negative = ~positive
    tp = int(np.sum(positive & prediction))
    fp = int(np.sum(negative & prediction))
    fn = int(np.sum(positive & ~prediction))
    tn = int(np.sum(negative & ~prediction))
    return {
        "binary_f1": float(f1_score(target, prediction, zero_division=0)),
        "macro_f1": float(
            f1_score(
                target,
                prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "ap": float(average_precision_score(target, score)),
        "auc": float(roc_auc_score(target, score)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "false_positive_rows": fp,
    }


def fixed_event_audit(
    *,
    labels: np.ndarray,
    event_ids: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=np.int64),
            "event_id": np.asarray(event_ids, dtype=str),
            "prediction": np.asarray(predictions, dtype=bool),
            "score": np.asarray(scores, dtype=np.float64),
        }
    )
    event_max = frame.groupby("event_id", sort=True)["label"].max()
    negative_ids = set(event_max.loc[event_max.eq(0)].index.astype(str))
    negative = frame.loc[frame["event_id"].isin(negative_ids)]
    per_event = negative.groupby("event_id", sort=True).agg(
        rows=("label", "size"),
        fp_rate=("prediction", "mean"),
        any_fp=("prediction", "max"),
        mean_score=("score", "mean"),
    )
    positive_ids = set(event_max.loc[event_max.eq(1)].index.astype(str))
    positive = frame.loc[frame["event_id"].isin(positive_ids)]
    detected = (
        positive.groupby("event_id", sort=True)["prediction"].max()
        if len(positive)
        else pd.Series(dtype=bool)
    )
    return {
        "canonical_events": int(len(event_max)),
        "all_negative": {
            "events": int(len(per_event)),
            "rows": int(len(negative)),
            "fp_event_count": int(per_event["any_fp"].sum()),
            "fp_event_fraction": float(per_event["any_fp"].mean()),
            "hard_fp_rows": int(negative["prediction"].sum()),
            "hard_fp_row_fraction": float(negative["prediction"].mean()),
            "equal_event_weight_fp_density": float(
                per_event["fp_rate"].mean()
            ),
            "mean_score": float(negative["score"].mean()),
        },
        "positive_or_mixed_events": int(len(positive_ids)),
        "positive_or_mixed_any_detection_count": int(detected.sum()),
        "positive_or_mixed_any_detection_recall": float(detected.mean()),
    }


def _weighted_fixed_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    row_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    target = np.asarray(labels, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=bool)
    weight = np.asarray(row_weights, dtype=np.float64)
    positive = target == 1
    negative = ~positive
    tp = weight @ (positive & prediction)
    fp = weight @ (negative & prediction)
    fn = weight @ (positive & ~prediction)
    tn = weight @ (negative & ~prediction)
    binary_f1 = np.divide(
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
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order]
    sorted_weight = weight[:, order]
    cumulative_tp = np.cumsum(
        sorted_weight * sorted_target.reshape(1, -1), axis=1
    )
    cumulative_fp = np.cumsum(
        sorted_weight * (1 - sorted_target).reshape(1, -1), axis=1
    )
    precision = np.divide(
        cumulative_tp,
        cumulative_tp + cumulative_fp,
        out=np.zeros_like(cumulative_tp),
        where=(cumulative_tp + cumulative_fp) > 0,
    )
    positive_weight = sorted_weight * sorted_target.reshape(1, -1)
    positive_mass = positive_weight.sum(axis=1)
    ap = np.divide(
        (precision * positive_weight).sum(axis=1),
        positive_mass,
        out=np.zeros_like(positive_mass),
        where=positive_mass > 0,
    )
    return {
        "binary_f1": binary_f1,
        "macro_f1": (binary_f1 + negative_f1) / 2.0,
        "ap": ap,
    }


def paired_event_bootstrap(
    *,
    labels: np.ndarray,
    event_ids: np.ndarray,
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    candidate_name: str,
    references: list[str],
    repeats: int,
    seed: int,
    batch_size: int = 64,
) -> dict[str, Any]:
    events = np.asarray(event_ids, dtype=str)
    unique_events = np.asarray(sorted(set(events.tolist())), dtype=object)
    lookup = {event: index for index, event in enumerate(unique_events)}
    row_event = np.asarray([lookup[event] for event in events], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    metrics = ("binary_f1", "macro_f1", "ap")
    values = {
        reference: {
            metric: np.empty(int(repeats), dtype=np.float64)
            for metric in metrics
        }
        for reference in references
    }
    uniform = np.full(len(unique_events), 1.0 / len(unique_events))
    for start in range(0, int(repeats), int(batch_size)):
        stop = min(start + int(batch_size), int(repeats))
        counts = rng.multinomial(
            len(unique_events), uniform, size=stop - start
        )
        weights = counts[:, row_event].astype(np.float64)
        candidate_score, candidate_prediction = models[candidate_name]
        candidate = _weighted_fixed_metrics(
            labels, candidate_score, candidate_prediction, weights
        )
        for reference_name in references:
            reference_score, reference_prediction = models[reference_name]
            reference = _weighted_fixed_metrics(
                labels, reference_score, reference_prediction, weights
            )
            for metric in metrics:
                values[reference_name][metric][start:stop] = (
                    candidate[metric] - reference[metric]
                )

    def summarize(array: np.ndarray) -> dict[str, Any]:
        return {
            "mean_delta": float(array.mean()),
            "ci95": [
                float(np.quantile(array, 0.025)),
                float(np.quantile(array, 0.975)),
            ],
            "win_probability": float(np.mean(array > 0)),
        }

    return {
        "unit": "canonical event cluster",
        "repeats": int(repeats),
        "seed": int(seed),
        "candidate": candidate_name,
        "thresholds_refit": False,
        "ensemble_weights_refit": False,
        "post_selection_diagnostic": True,
        "candidate_minus_reference": {
            reference: {
                metric: summarize(array)
                for metric, array in metric_values.items()
            }
            for reference, metric_values in values.items()
        },
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Frozen availability-gate control",
        "",
        "Development only; no threshold or score weight was fitted and no "
        "test/sealed artifact was read.",
        "",
        "Rule: use the frozen four-seed R4 ensemble only when exactly one "
        "sensor is available; otherwise use frozen P0. Each branch uses its "
        "own already locked global threshold.",
        "",
        "| Model | F1 | Macro F1 | AP | AUC | FP rows | "
        "all-neg FP events/rows | Positive events detected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("p0", "r4_global", "single_sensor_r4_else_p0"):
        record = result["models"][name]
        metric = record["metrics"]
        event = record["event_audit"]
        lines.append(
            f"| {name} | {metric['binary_f1']:.6f} | "
            f"{metric['macro_f1']:.6f} | {metric['ap']:.6f} | "
            f"{metric['auc']:.6f} | {metric['false_positive_rows']} | "
            f"{event['all_negative']['fp_event_count']}/"
            f"{event['all_negative']['hard_fp_rows']} | "
            f"{event['positive_or_mixed_any_detection_count']}/"
            f"{event['positive_or_mixed_events']} |"
        )
    lines.extend(
        [
            "",
            "## Paired canonical-event bootstrap",
            "",
            "| Candidate minus reference | Metric | Mean | 95% CI | Win prob. |",
            "|---|---|---:|---:|---:|",
        ]
    )
    bootstrap = result["paired_event_bootstrap"][
        "candidate_minus_reference"
    ]
    for reference in ("p0", "r4_global"):
        for metric in ("binary_f1", "macro_f1", "ap"):
            value = bootstrap[reference][metric]
            lines.append(
                f"| gate − {reference} | {metric} | "
                f"{value['mean_delta']:+.6f} | "
                f"[{value['ci95'][0]:+.6f},{value['ci95'][1]:+.6f}] | "
                f"{value['win_probability']:.4f} |"
            )
    lines.extend(
        [
            "",
            "AP/AUC use the raw logit-derived probability from whichever frozen "
            "branch the binary rule selects. F1/FP use that branch's locked "
            "global threshold.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def command_run(args: argparse.Namespace) -> None:
    aggregate_path = Path(args.aggregate).expanduser().absolute()
    prediction_path = Path(args.predictions).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    for path, role in (
        (aggregate_path, "development aggregate"),
        (prediction_path, "development predictions"),
        (output_dir, "availability gate output"),
    ):
        guard_development_path(path, role)
    output_path = output_dir / "availability_gate_control.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing existing result: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("r", encoding="utf-8") as stream:
        aggregate = json.load(stream)
    if bool(aggregate["test_or_sealed_read"]):
        raise RuntimeError("Aggregate reports test/sealed read.")
    frame = pd.read_csv(prediction_path)
    labels = frame["label"].to_numpy(dtype=np.int64)
    event_ids = frame["event_id"].astype(str).to_numpy()
    sensor_count = frame["availability_signature"].astype(str).map(
        lambda value: len(value.split("+"))
    ).to_numpy(dtype=np.int64)
    if not np.all(sensor_count >= 1):
        raise RuntimeError("Every row must have at least one sensor.")
    p0_score = frame["p0_probability"].to_numpy(dtype=np.float64)
    r4_score = frame["ensemble_probability"].to_numpy(dtype=np.float64)
    p0_threshold = float(
        aggregate["p0"]["metrics"]["best_binary_f1_threshold"]
    )
    r4_threshold = float(
        aggregate["logit_ensemble"]["metrics"]["best_binary_f1_threshold"]
    )
    p0_prediction = p0_score >= p0_threshold
    r4_prediction = r4_score >= r4_threshold
    use_r4 = sensor_count == 1
    gate_score = np.where(use_r4, r4_score, p0_score)
    gate_prediction = np.where(use_r4, r4_prediction, p0_prediction)
    model_arrays = {
        "p0": (p0_score, p0_prediction),
        "r4_global": (r4_score, r4_prediction),
        "single_sensor_r4_else_p0": (gate_score, gate_prediction),
    }
    models = {
        name: {
            "metrics": fixed_metrics(labels, score, prediction),
            "event_audit": fixed_event_audit(
                labels=labels,
                event_ids=event_ids,
                predictions=prediction,
                scores=score,
            ),
        }
        for name, (score, prediction) in model_arrays.items()
    }
    bootstrap = paired_event_bootstrap(
        labels=labels,
        event_ids=event_ids,
        models=model_arrays,
        candidate_name="single_sensor_r4_else_p0",
        references=["p0", "r4_global"],
        repeats=int(args.bootstrap_repeats),
        seed=int(args.bootstrap_seed),
    )
    result = {
        "schema_version": "tempo-availability-gate-control-v1",
        "script_version": SCRIPT_VERSION,
        "rule": {
            "exactly_one_available_sensor": "four-seed R4 ensemble",
            "two_or_more_available_sensors": "P0",
            "continuous_weights_fitted": False,
            "thresholds_fitted": False,
            "p0_threshold": p0_threshold,
            "r4_threshold": r4_threshold,
            "single_sensor_rows": int(use_r4.sum()),
            "multi_sensor_rows": int((~use_r4).sum()),
        },
        "models": models,
        "paired_event_bootstrap": bootstrap,
        "inputs": {
            "aggregate": str(aggregate_path),
            "aggregate_sha256": sha256_file(aggregate_path),
            "predictions": str(prediction_path),
            "predictions_sha256": sha256_file(prediction_path),
        },
        "test_or_sealed_read": False,
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    atomic_json(output_path, result)
    write_markdown(output_dir / "AVAILABILITY_GATE_CONTROL.md", result)
    print(json.dumps(result["models"], indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", default=str(DEFAULT_AGGREGATE))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    return parser


def main() -> None:
    command_run(build_parser().parse_args())


if __name__ == "__main__":
    main()
