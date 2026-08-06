#!/usr/bin/env python3
"""Aggregate fixed single-sensor R4 seeds without retraining or held-out reads."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as cache_runner,
)
from research.tempo_20260728 import tempo_l89_global as metric_runner  # noqa: E402


SCRIPT_VERSION = "tempo-single-sensor-r4-aggregate-v1"
ARMS = ("raw_delta", "r4_motion_excitation")
METRICS = (
    "event_balanced_positive_f1_selected",
    "event_balanced_macro_f1_selected",
    "event_balanced_ap",
    "event_balanced_auc",
    "all_negative_fp_mass",
)
FORBIDDEN_RE = re.compile(
    r"(^|[._-])(test|sealed|holdout)([._-]|$)", re.IGNORECASE
)


def assert_development_path(path: Path, *, purpose: str) -> Path:
    resolved = path.expanduser().resolve()
    offending = [
        part for part in resolved.parts if FORBIDDEN_RE.search(part.lower())
    ]
    if offending:
        raise ValueError(
            f"{purpose} path contains held-out marker {offending}: {resolved}"
        )
    cache_runner.assert_not_sealed_path(resolved, purpose=purpose)
    return resolved


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7
    )
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.where(
        value >= 0,
        1.0 / (1.0 + np.exp(-value)),
        np.exp(value) / (1.0 + np.exp(value)),
    )


def summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": (
            float(array.std(ddof=1)) if len(array) > 1 else 0.0
        ),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def load_seed(
    root: Path, seed: int
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    seed_dir = assert_development_path(
        root / f"emit_3time_seed{seed}", purpose=f"seed {seed} root"
    )
    aggregate_path = seed_dir / "aggregate.json"
    if not aggregate_path.is_file():
        raise FileNotFoundError(aggregate_path)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate.get("test_or_sealed_or_holdout_read") is not False:
        raise RuntimeError(f"Seed {seed} is not development-only.")
    predictions: dict[str, pd.DataFrame] = {}
    for arm in ARMS:
        path = Path(aggregate["arms"][arm]["predictions"])
        path = assert_development_path(path, purpose=f"{arm} seed predictions")
        frame = pd.read_csv(path)
        required = {
            "id",
            "event_id",
            "label",
            "p0_probability",
            "probability",
            "history_shuffle_probability",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} missing columns {missing}")
        predictions[arm] = frame
    return aggregate, predictions


def paired_event_bootstrap(
    *,
    labels: np.ndarray,
    event_ids: np.ndarray,
    p0_probability: np.ndarray,
    candidate_probability: np.ndarray,
    p0_threshold: float,
    candidate_threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    unique_events = np.asarray(
        sorted(set(map(str, event_ids.tolist()))), dtype=object
    )
    positions = {
        event: np.flatnonzero(event_ids == event) for event in unique_events
    }
    event_is_positive = {
        event: bool(labels[positions[event]].max()) for event in unique_events
    }
    p0_fp_rate = {
        event: float(
            np.mean(
                p0_probability[positions[event]] >= float(p0_threshold)
            )
        )
        for event in unique_events
        if not event_is_positive[event]
    }
    candidate_fp_rate = {
        event: float(
            np.mean(
                candidate_probability[positions[event]]
                >= float(candidate_threshold)
            )
        )
        for event in unique_events
        if not event_is_positive[event]
    }

    def weighted_metrics(
        selected_labels: np.ndarray,
        selected_probability: np.ndarray,
        selected_weights: np.ndarray,
        *,
        threshold: float,
    ) -> dict[str, float]:
        prediction = selected_probability >= float(threshold)
        positive = selected_labels == 1
        negative = ~positive
        tp = float(selected_weights[prediction & positive].sum())
        fp = float(selected_weights[prediction & negative].sum())
        fn = float(selected_weights[(~prediction) & positive].sum())
        tn = float(selected_weights[(~prediction) & negative].sum())
        positive_denominator = 2.0 * tp + fp + fn
        negative_denominator = 2.0 * tn + fp + fn
        positive_f1 = (
            2.0 * tp / positive_denominator
            if positive_denominator > 0
            else 0.0
        )
        negative_f1 = (
            2.0 * tn / negative_denominator
            if negative_denominator > 0
            else 0.0
        )
        return {
            "event_balanced_positive_f1_selected": positive_f1,
            "event_balanced_macro_f1_selected": (
                positive_f1 + negative_f1
            )
            / 2.0,
            "event_balanced_ap": float(
                average_precision_score(
                    selected_labels,
                    selected_probability,
                    sample_weight=selected_weights,
                )
            ),
            "event_balanced_auc": float(
                roc_auc_score(
                    selected_labels,
                    selected_probability,
                    sample_weight=selected_weights,
                )
            ),
        }
    rng = np.random.default_rng(int(seed))
    deltas = {
        metric: np.empty(int(replicates), dtype=np.float64)
        for metric in METRICS
    }
    completed = 0
    attempts = 0
    while completed < int(replicates):
        attempts += 1
        selected = rng.choice(
            unique_events, size=len(unique_events), replace=True
        )
        row_indices: list[np.ndarray] = []
        row_weights: list[np.ndarray] = []
        for event in selected.tolist():
            rows = positions[str(event)]
            row_indices.append(rows)
            row_weights.append(
                np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
            )
        selected_rows = np.concatenate(row_indices)
        selected_weights = np.concatenate(row_weights)
        selected_labels = labels[selected_rows]
        if len(np.unique(selected_labels)) != 2:
            continue
        candidate = weighted_metrics(
            selected_labels,
            candidate_probability[selected_rows],
            threshold=float(candidate_threshold),
            selected_weights=selected_weights,
        )
        p0 = weighted_metrics(
            selected_labels,
            p0_probability[selected_rows],
            threshold=float(p0_threshold),
            selected_weights=selected_weights,
        )
        selected_negative_events = [
            str(event)
            for event in selected.tolist()
            if not event_is_positive[str(event)]
        ]
        candidate["all_negative_fp_mass"] = float(
            sum(candidate_fp_rate[event] for event in selected_negative_events)
        )
        p0["all_negative_fp_mass"] = float(
            sum(p0_fp_rate[event] for event in selected_negative_events)
        )
        for metric in METRICS:
            deltas[metric][completed] = float(candidate[metric] - p0[metric])
        completed += 1
        if completed % 1000 == 0:
            print(
                f"[bootstrap] {completed}/{replicates}", flush=True
            )
        if attempts > int(replicates) * 10:
            raise RuntimeError("Too many invalid bootstrap resamples.")
    output: dict[str, Any] = {
        "replicates": int(replicates),
        "canonical_events": int(len(unique_events)),
        "seed": int(seed),
        "thresholds_fixed_before_resampling": {
            "p0": float(p0_threshold),
            "candidate": float(candidate_threshold),
        },
        "refit_within_bootstrap": False,
        "metrics": {},
    }
    for metric, values in deltas.items():
        output["metrics"][metric] = {
            "mean_delta": float(values.mean()),
            "median_delta": float(np.median(values)),
            "ci95_percentile": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "win_probability": float(np.mean(values > 0)),
            "tie_probability": float(np.mean(values == 0)),
        }
    return output


def run(args: argparse.Namespace) -> None:
    root = assert_development_path(Path(args.root), purpose="seed root")
    output_dir = assert_development_path(
        Path(args.output_dir), purpose="aggregate output"
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    seeds = [int(value) for value in str(args.seeds).split(",")]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two unique seeds are required.")

    aggregates: dict[int, dict[str, Any]] = {}
    prediction_sets: dict[int, dict[str, pd.DataFrame]] = {}
    reference_ids: Optional[list[str]] = None
    reference_labels: Optional[np.ndarray] = None
    reference_events: Optional[np.ndarray] = None
    reference_p0: Optional[np.ndarray] = None
    cache_sha: Optional[tuple[str, str]] = None
    base_sha: Optional[str] = None
    for seed in seeds:
        aggregate, predictions = load_seed(root, seed)
        aggregates[seed] = aggregate
        prediction_sets[seed] = predictions
        frame = predictions[ARMS[0]]
        ids = frame["id"].astype(str).tolist()
        labels = frame["label"].to_numpy(dtype=np.int64)
        events = frame["event_id"].astype(str).to_numpy(dtype=object)
        p0 = frame["p0_probability"].to_numpy(dtype=np.float64)
        if reference_ids is None:
            reference_ids = ids
            reference_labels = labels
            reference_events = events
            reference_p0 = p0
            cache_sha = (
                str(aggregate["cache_audit"]["train_cache_sha256"]),
                str(aggregate["cache_audit"]["validation_cache_sha256"]),
            )
            base_sha = str(aggregate["base_audit"]["checkpoint_sha256"])
        else:
            if ids != reference_ids:
                raise RuntimeError(f"Seed {seed} identity order changed.")
            if not np.array_equal(labels, reference_labels):
                raise RuntimeError(f"Seed {seed} labels changed.")
            if not np.array_equal(events, reference_events):
                raise RuntimeError(f"Seed {seed} event IDs changed.")
            if not np.allclose(p0, reference_p0, atol=1e-9, rtol=0):
                raise RuntimeError(f"Seed {seed} P0 probabilities changed.")
            current_sha = (
                str(aggregate["cache_audit"]["train_cache_sha256"]),
                str(aggregate["cache_audit"]["validation_cache_sha256"]),
            )
            if current_sha != cache_sha:
                raise RuntimeError(f"Seed {seed} cache SHA changed.")
            if str(aggregate["base_audit"]["checkpoint_sha256"]) != base_sha:
                raise RuntimeError(f"Seed {seed} base checkpoint changed.")
        for arm in ARMS[1:]:
            other = predictions[arm]
            if other["id"].astype(str).tolist() != ids:
                raise RuntimeError(f"Seed {seed} arm identity order changed.")

    assert reference_ids is not None
    assert reference_labels is not None
    assert reference_events is not None
    assert reference_p0 is not None
    p0_metrics = metric_runner.metric_bundle(
        reference_labels, reference_p0, reference_events.tolist()
    )
    ensemble_results: dict[str, Any] = {}
    ensemble_columns: dict[str, np.ndarray] = {
        "id": np.asarray(reference_ids, dtype=object),
        "event_id": reference_events,
        "label": reference_labels,
        "p0_probability": reference_p0,
    }
    for arm in ARMS:
        probabilities = np.stack(
            [
                prediction_sets[seed][arm]["probability"].to_numpy(
                    dtype=np.float64
                )
                for seed in seeds
            ],
            axis=0,
        )
        shuffled = np.stack(
            [
                prediction_sets[seed][arm][
                    "history_shuffle_probability"
                ].to_numpy(dtype=np.float64)
                for seed in seeds
            ],
            axis=0,
        )
        ensemble_probability = sigmoid(logit(probabilities).mean(axis=0))
        shuffled_probability = sigmoid(logit(shuffled).mean(axis=0))
        metrics = metric_runner.metric_bundle(
            reference_labels,
            ensemble_probability,
            reference_events.tolist(),
        )
        shuffled_metrics = metric_runner.metric_bundle(
            reference_labels,
            shuffled_probability,
            reference_events.tolist(),
            threshold=float(metrics["selected_threshold"]),
        )
        per_seed = {
            metric: summarize(
                [
                    aggregates[seed]["arms"][arm]["best"]["validation"][
                        metric
                    ]
                    for seed in seeds
                ]
            )
            for metric in METRICS
        }
        ensemble_results[arm] = {
            "equal_logit_ensemble": metrics,
            "history_shuffle_fixed_threshold": shuffled_metrics,
            "history_shuffle_delta": {
                metric: float(shuffled_metrics[metric] - metrics[metric])
                for metric in METRICS
            },
            "per_seed_summary": per_seed,
            "selected_epochs": {
                str(seed): int(
                    aggregates[seed]["arms"][arm]["best"]["epoch"]
                )
                for seed in seeds
            },
        }
        ensemble_columns[f"{arm}_probability"] = ensemble_probability
        ensemble_columns[
            f"{arm}_history_shuffle_probability"
        ] = shuffled_probability

    promoted = ensemble_results["r4_motion_excitation"][
        "equal_logit_ensemble"
    ]
    bootstrap = paired_event_bootstrap(
        labels=reference_labels,
        event_ids=reference_events,
        p0_probability=reference_p0,
        candidate_probability=ensemble_columns[
            "r4_motion_excitation_probability"
        ],
        p0_threshold=float(p0_metrics["selected_threshold"]),
        candidate_threshold=float(promoted["selected_threshold"]),
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
    )
    cache_runner.atomic_csv_write(
        output_dir / "ensemble_predictions.csv",
        pd.DataFrame(ensemble_columns),
    )
    output = {
        "script_version": SCRIPT_VERSION,
        "seeds": seeds,
        "p0_exact": p0_metrics,
        "ensembles": ensemble_results,
        "r4_vs_p0_paired_canonical_event_bootstrap": bootstrap,
        "provenance": {
            "seed_root": str(root),
            "train_cache_sha256": cache_sha[0] if cache_sha else None,
            "dev_cache_sha256": cache_sha[1] if cache_sha else None,
            "p0_checkpoint_sha256": base_sha,
        },
        "test_or_sealed_or_holdout_read": False,
    }
    cache_runner.atomic_json_write(output_dir / "aggregate.json", output)

    r4 = ensemble_results["r4_motion_excitation"]
    raw = ensemble_results["raw_delta"]
    lines = [
        "# EMIT three-time R4 fixed multi-seed aggregate",
        "",
        "Development-only; no test, sealed, or holdout artifact was read.",
        "",
        f"Seeds: `{','.join(map(str, seeds))}`. All hyperparameters and the "
        "frozen P0 checkpoint were unchanged.",
        "",
        "| Model | event-F1 | event-macro | event-AP | event-AUC | FP mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (
        ("P0 exact", p0_metrics),
        ("Raw-delta equal-logit", raw["equal_logit_ensemble"]),
        ("R4 equal-logit", r4["equal_logit_ensemble"]),
        ("R4 history shuffle, fixed threshold", r4["history_shuffle_fixed_threshold"]),
    ):
        lines.append(
            f"| {name} | "
            f"{metrics['event_balanced_positive_f1_selected']:.6f} | "
            f"{metrics['event_balanced_macro_f1_selected']:.6f} | "
            f"{metrics['event_balanced_ap']:.6f} | "
            f"{metrics['event_balanced_auc']:.6f} | "
            f"{metrics['all_negative_fp_mass']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired canonical-event bootstrap: R4 minus P0",
            "",
            f"{int(args.bootstrap_replicates)} resamples; thresholds were "
            "fixed once before resampling and never refit.",
            "",
            "| Metric | mean delta | 95% CI | win probability |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        record = bootstrap["metrics"][metric]
        lines.append(
            f"| {metric} | {record['mean_delta']:+.6f} | "
            f"[{record['ci95_percentile'][0]:+.6f},"
            f"{record['ci95_percentile'][1]:+.6f}] | "
            f"{record['win_probability']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a three-time EMIT transfer pilot using t0, prev1, and "
            "seasonal frozen CLS features. It does not establish a six-time "
            "EMIT result.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=(
            "/diniuvol/yuyao/methanefuse_tempo_20260728/"
            "cross_sensor_r4_v1"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="17,42,73,20260728")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
