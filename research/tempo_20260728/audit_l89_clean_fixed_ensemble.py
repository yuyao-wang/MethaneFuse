#!/usr/bin/env python3
"""Primary clean-L89 fixed ensemble and event-cluster uncertainty audit.

The three predeclared D1 prediction tables are first averaged in **logit
space**. Only then is the fixed two-stream candidate formed:

``candidate_logit = 0.5 * P5_logit + 0.5 * mean_seed(D1_logit)``.

This is intentionally different from averaging metrics of three independent
P5+D1 blends. The program writes one candidate prediction table and performs a
5,000-replicate canonical-event-cluster paired bootstrap against the same-run
P5 and P0 predictions. Point-selected thresholds are held fixed in every
bootstrap replicate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.pretraining_20260727 import l89_ragged_cls_experiment as cache
from research.tempo_20260728 import tempo_l89_global as tempo


FROZEN_SEEDS = (20260727, 20260728, 20260729)
FROZEN_P5_WEIGHT = 0.5
FROZEN_D1_WEIGHT = 0.5
FROZEN_BOOTSTRAP_REPLICATES = 5000
FROZEN_BOOTSTRAP_SEED = 2026072808


def logit(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.where(
        array >= 0,
        1.0 / (1.0 + np.exp(-array)),
        np.exp(array) / (1.0 + np.exp(array)),
    )


def fixed_ensemble(
    p5_probability: np.ndarray,
    d1_probabilities: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if len(d1_probabilities) != len(FROZEN_SEEDS):
        raise ValueError("Exactly three predeclared D1 predictions are required.")
    p5_logit = logit(p5_probability)
    d1_logits = np.stack([logit(value) for value in d1_probabilities], axis=0)
    mean_d1_logit = d1_logits.mean(axis=0)
    candidate_logit = (
        FROZEN_P5_WEIGHT * p5_logit
        + FROZEN_D1_WEIGHT * mean_d1_logit
    )
    return sigmoid(mean_d1_logit), sigmoid(candidate_logit)


def aligned_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    names = tuple(frames)
    if not names:
        raise ValueError("No prediction frames supplied.")
    reference = frames[names[0]]
    required = {"id", "plume_id", "event_id", "label", "probability"}
    for name, frame in frames.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} prediction frame is missing {missing}.")
    for name in names[1:]:
        candidate = frames[name]
        for column in ("id", "plume_id", "event_id"):
            if (
                reference[column].astype(str).tolist()
                != candidate[column].astype(str).tolist()
            ):
                raise ValueError(
                    f"{name} differs from {names[0]} in ordered {column}."
                )
        if not np.array_equal(
            reference["label"].to_numpy(dtype=np.int64),
            candidate["label"].to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"{name} labels differ from {names[0]}.")


def fixed_threshold_event_bootstrap(
    labels: np.ndarray,
    event_ids: Sequence[str],
    probabilities: Mapping[str, np.ndarray],
    point_metrics: Mapping[str, Mapping[str, Any]],
    comparisons: Sequence[tuple[str, str, str]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    events = np.asarray([str(value) for value in event_ids], dtype=object)
    canonical, codes, sizes = np.unique(
        events, return_inverse=True, return_counts=True
    )
    if len(canonical) < 2:
        raise ValueError("At least two canonical events are required.")
    thresholds = {
        name: float(point_metrics[name]["selected_threshold"])
        for name in probabilities
    }
    all_negative = np.zeros(len(canonical), dtype=bool)
    fp_rate: dict[str, np.ndarray] = {
        name: np.zeros(len(canonical), dtype=np.float64)
        for name in probabilities
    }
    for code in range(len(canonical)):
        rows = codes == code
        all_negative[code] = bool(np.max(target[rows]) == 0)
        if all_negative[code]:
            for name, values in probabilities.items():
                fp_rate[name][code] = float(
                    np.mean(np.asarray(values)[rows] >= thresholds[name])
                )
    metric_names = (
        "event_balanced_ap",
        "event_balanced_auc",
        "event_balanced_positive_f1_selected",
        "event_balanced_macro_f1_selected",
        "all_negative_fp_mass",
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
        metrics: dict[str, dict[str, float]] = {}
        for name, values in probabilities.items():
            metrics[name] = tempo.weighted_selected_metrics(
                target,
                np.asarray(values, dtype=np.float64),
                row_weights,
                thresholds[name],
            )
            metrics[name]["all_negative_fp_mass"] = float(
                np.sum(event_counts[all_negative] * fp_rate[name][all_negative])
            )
        for comparison, left, right in comparisons:
            for metric in metric_names:
                distributions[comparison][metric].append(
                    metrics[left][metric] - metrics[right][metric]
                )
        if (replicate + 1) % 500 == 0:
            print(
                f"[clean-bootstrap] {replicate + 1}/{replicates}",
                flush=True,
            )
    output: dict[str, Any] = {}
    for comparison, left, right in comparisons:
        output[comparison] = {}
        for metric in metric_names:
            values = np.asarray(
                distributions[comparison][metric], dtype=np.float64
            )
            lower_is_better = metric == "all_negative_fp_mass"
            output[comparison][metric] = {
                "point": float(
                    point_metrics[left][metric] - point_metrics[right][metric]
                ),
                "ci_95_low": float(np.percentile(values, 2.5)),
                "ci_95_high": float(np.percentile(values, 97.5)),
                "win_probability": float(
                    np.mean(values < 0) if lower_is_better else np.mean(values > 0)
                ),
                "better_direction": "lower" if lower_is_better else "higher",
            }
    return output


def render_markdown(payload: Mapping[str, Any]) -> str:
    metrics = payload["point_metrics"]
    deltas = payload["paired_event_cluster_bootstrap"]
    lines = [
        "# Clean L89 fixed P5 + mean-seed D1 audit",
        "",
        "Primary candidate arithmetic:",
        "",
        "`0.5 * logit(P5) + 0.5 * mean_seed(logit(D1))`",
        "",
        "| System | Event-balanced row AP | AUC | Positive F1 | Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("p0", "p5", "mean_seed_d1", "fixed_p5_mean_d1"):
        point = metrics[name]
        lines.append(
            f"| {name} | {point['event_balanced_ap']:.6f} | "
            f"{point['event_balanced_auc']:.6f} | "
            f"{point['event_balanced_positive_f1_selected']:.6f} | "
            f"{point['event_balanced_macro_f1_selected']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Canonical-event-cluster paired bootstrap (5,000 replicates; point",
            "thresholds fixed in every replicate):",
            "",
            "| Comparison | Metric | Delta | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for comparison in ("candidate_minus_p5", "candidate_minus_p0"):
        for metric, result in deltas[comparison].items():
            lines.append(
                f"| {comparison} | {metric} | {result['point']:+.6f} | "
                f"[{result['ci_95_low']:+.6f}, "
                f"{result['ci_95_high']:+.6f}] |"
            )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if tuple(args.seeds) != FROZEN_SEEDS:
        raise ValueError(f"Seeds must be exactly {FROZEN_SEEDS}.")
    if int(args.replicates) != FROZEN_BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"Formal bootstrap is frozen to {FROZEN_BOOTSTRAP_REPLICATES}."
        )
    if int(args.bootstrap_seed) != FROZEN_BOOTSTRAP_SEED:
        raise ValueError(
            f"Formal bootstrap seed is frozen to {FROZEN_BOOTSTRAP_SEED}."
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    tempo.assert_development_path(output_dir, purpose="clean ensemble output")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    paths = {
        "p0": Path(args.p0_predictions).expanduser().resolve(),
        "p5": Path(args.p5_predictions).expanduser().resolve(),
    }
    for seed in FROZEN_SEEDS:
        paths[f"d1_seed_{seed}"] = Path(
            str(args.d1_template).format(seed=seed)
        ).expanduser().resolve()
    frames = {
        name: tempo.load_prediction_table(path)
        for name, path in paths.items()
    }
    aligned_frames(frames)
    reference = frames["p5"]
    labels = reference["label"].to_numpy(dtype=np.int64)
    event_ids = reference["event_id"].astype(str).tolist()
    p0_probability = frames["p0"]["probability"].to_numpy(dtype=np.float64)
    p5_probability = frames["p5"]["probability"].to_numpy(dtype=np.float64)
    d1_probabilities = [
        frames[f"d1_seed_{seed}"]["probability"].to_numpy(dtype=np.float64)
        for seed in FROZEN_SEEDS
    ]
    mean_d1_probability, candidate_probability = fixed_ensemble(
        p5_probability, d1_probabilities
    )
    probabilities = {
        "p0": p0_probability,
        "p5": p5_probability,
        "mean_seed_d1": mean_d1_probability,
        "fixed_p5_mean_d1": candidate_probability,
    }
    point_metrics = {
        name: tempo.metric_bundle(labels, values, event_ids)
        for name, values in probabilities.items()
    }
    bootstrap = fixed_threshold_event_bootstrap(
        labels,
        event_ids,
        probabilities,
        point_metrics,
        (
            (
                "candidate_minus_p5",
                "fixed_p5_mean_d1",
                "p5",
            ),
            (
                "candidate_minus_p0",
                "fixed_p5_mean_d1",
                "p0",
            ),
        ),
        replicates=int(args.replicates),
        seed=int(args.bootstrap_seed),
    )

    output_dir.mkdir(parents=True)
    prediction_path = output_dir / "fixed_p5_mean_d1_predictions.csv"
    prediction_frame = reference.copy()
    prediction_frame["probability"] = candidate_probability
    prediction_frame["selected_threshold"] = float(
        point_metrics["fixed_p5_mean_d1"]["selected_threshold"]
    )
    cache.atomic_csv_write(prediction_path, prediction_frame)
    payload = {
        "schema_version": "l89-clean-fixed-p5-mean-d1-v1",
        "arithmetic": {
            "d1_seed_aggregation": "mean of three D1 logits before fusion",
            "d1_seeds": list(FROZEN_SEEDS),
            "p5_logit_weight": FROZEN_P5_WEIGHT,
            "mean_d1_logit_weight": FROZEN_D1_WEIGHT,
            "formula": "0.5*logit(P5)+0.5*mean_seed(logit(D1))",
            "weights_refit": False,
        },
        "point_metrics": point_metrics,
        "paired_event_cluster_bootstrap": bootstrap,
        "bootstrap": {
            "unit": "canonical event_id",
            "replicates": int(args.replicates),
            "seed": int(args.bootstrap_seed),
            "point_selected_thresholds_fixed_per_system": True,
            "thresholds_refit_per_replicate": False,
        },
        "prediction_output": {
            "path": str(prediction_path),
            "sha256": cache.sha256_file(prediction_path),
        },
        "input_provenance": {
            name: {"path": str(path), "sha256": cache.sha256_file(path)}
            for name, path in paths.items()
        },
        "event_balanced_metric_definition": (
            "row predictions with inverse canonical-event-size weights; "
            "not event-aggregated predictions"
        ),
        "formal_inner_development_only": True,
        "test_or_sealed_or_holdout_or_outer_read": False,
    }
    cache.atomic_json_write(output_dir / "RESULT.json", payload)
    (output_dir / "RESULT.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-predictions", required=True)
    parser.add_argument("--p5-predictions", required=True)
    parser.add_argument("--d1-template", required=True)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=FROZEN_SEEDS,
    )
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument(
        "--bootstrap-seed", type=int, default=FROZEN_BOOTSTRAP_SEED
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
