#!/usr/bin/env python3
"""Fixed post-hoc P5+D1-ensemble+D7 ceiling audit on L89 development.

This audit is deliberately narrow:

* P5, the fixed three-seed D1 logit ensemble, and one already-selected D7
  checkpoint each receive exactly one third of the final logit;
* the reference is the already-supported one-half P5 plus one-half
  three-seed D1 logit ensemble;
* no fusion weight, seed, checkpoint, feature, or threshold family is
  searched;
* each system receives only the standard event-balanced macro-F1 threshold
  returned by ``metric_bundle``;
* bootstrap replicates hold those two point-estimate thresholds fixed.

D7 already failed its preregistered promotion gate.  Therefore this audit can
only estimate a post-hoc ceiling and can never promote the three-expert model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research.tempo_20260728 import tempo_l89_global as global_runner


SCRIPT_VERSION = "tempo-l89-posthoc-three-expert-ceiling-v1"
DEFAULT_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1"
)
DEFAULT_P5 = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/"
    "rctp_l89_sidecar_fallback_v1/downstream_event_balanced_seed20260728/"
    "p5/validation_best_event_balanced_ap_predictions.csv"
)
DEFAULT_D1_TEMPLATE = (
    DEFAULT_ROOT
    / "eventbase_d1_multiseed_fixed/seed_{seed}/"
    "d1_gated_delta_best_event_ap_predictions.csv"
)
DEFAULT_D7 = (
    DEFAULT_ROOT
    / "eventbase_d7_sparse_slowfast_seed20260728_v1/seed_20260728/"
    "d7_sparse_slowfast_best_event_ap_predictions.csv"
)
DEFAULT_OUTPUT = Path(
    "research/tempo_20260728/"
    "l89_posthoc_three_expert_ceiling_v1.json"
)
DEFAULT_D1_SEEDS = (20260727, 20260728, 20260729)
METRICS = (
    "event_balanced_ap",
    "event_balanced_auc",
    "event_balanced_macro_f1_selected",
    "event_balanced_positive_f1_selected",
    "all_negative_fp_mass",
)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    positive = values >= 0.0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def fixed_fusions(
    p5_probability: np.ndarray,
    d1_probabilities: Sequence[np.ndarray],
    d7_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return exactly the locked two-expert reference and three-expert ceiling."""

    if len(d1_probabilities) != 3:
        raise ValueError("The ceiling audit requires exactly three D1 seeds.")
    p5_logit = global_runner.safe_logit(p5_probability)
    d1_logits = np.stack(
        [global_runner.safe_logit(value) for value in d1_probabilities],
        axis=0,
    )
    d1_ensemble_logit = d1_logits.mean(axis=0)
    d7_logit = global_runner.safe_logit(d7_probability)
    return {
        "d1_three_seed_logit_ensemble": sigmoid(d1_ensemble_logit),
        "p5_d1_equal_logit_reference": sigmoid(
            0.5 * p5_logit + 0.5 * d1_ensemble_logit
        ),
        "p5_d1_d7_equal_thirds_posthoc_ceiling": sigmoid(
            (p5_logit + d1_ensemble_logit + d7_logit) / 3.0
        ),
    }


def _all_negative_fp_rate_by_event(
    labels: np.ndarray,
    event_codes: np.ndarray,
    event_count: int,
    probability: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    event_positive = np.zeros(event_count, dtype=np.bool_)
    np.logical_or.at(event_positive, event_codes, labels.astype(bool))
    all_negative = ~event_positive
    hard_prediction = probability >= float(threshold)
    row_count = np.bincount(event_codes, minlength=event_count).astype(
        np.float64
    )
    fp_count = np.bincount(
        event_codes,
        weights=hard_prediction.astype(np.float64),
        minlength=event_count,
    )
    fp_rate = np.divide(
        fp_count,
        row_count,
        out=np.zeros_like(fp_count),
        where=row_count > 0,
    )
    return all_negative, fp_rate


def canonical_event_cluster_bootstrap(
    labels: np.ndarray,
    event_ids: Sequence[str],
    probabilities: Mapping[str, np.ndarray],
    point_metrics: Mapping[str, Mapping[str, Any]],
    *,
    candidate_name: str,
    reference_name: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired canonical-event bootstrap with fixed point thresholds.

    FP mass in a replicate is the multiplicity-weighted sum of within-event
    hard-FP rates over sampled all-negative canonical events.  This exactly
    equals the existing point definition when every event has multiplicity
    one and preserves the equal-event interpretation under cluster resampling.
    """

    target = np.asarray(labels, dtype=np.int64)
    events = np.asarray([str(value) for value in event_ids], dtype=object)
    canonical, codes, sizes = np.unique(
        events, return_inverse=True, return_counts=True
    )
    if len(canonical) < 2:
        raise ValueError("At least two canonical events are required.")
    names = tuple(probabilities)
    if candidate_name not in names or reference_name not in names:
        raise ValueError("Candidate/reference probabilities are missing.")
    for name, probability in probabilities.items():
        if np.asarray(probability).shape != target.shape:
            raise ValueError(f"{name} probabilities do not match labels.")

    thresholds = {
        name: float(point_metrics[name]["selected_threshold"])
        for name in names
    }
    all_negative: np.ndarray | None = None
    fp_rates: dict[str, np.ndarray] = {}
    for name in names:
        arm_negative, arm_rates = _all_negative_fp_rate_by_event(
            target,
            codes,
            len(canonical),
            np.asarray(probabilities[name], dtype=np.float64),
            thresholds[name],
        )
        if all_negative is None:
            all_negative = arm_negative
        elif not np.array_equal(all_negative, arm_negative):
            raise RuntimeError("All-negative event identity differs by arm.")
        fp_rates[name] = arm_rates
    assert all_negative is not None

    distributions = {metric: [] for metric in METRICS}
    rng = np.random.default_rng(int(seed))
    for replicate in range(int(replicates)):
        sampled = rng.integers(0, len(canonical), size=len(canonical))
        event_multiplicity = np.bincount(
            sampled, minlength=len(canonical)
        ).astype(np.float64)
        row_weights = event_multiplicity[codes] / sizes[codes]
        replicate_metrics: dict[str, dict[str, float]] = {}
        for name in (reference_name, candidate_name):
            values = global_runner.weighted_selected_metrics(
                target,
                np.asarray(probabilities[name], dtype=np.float64),
                row_weights,
                thresholds[name],
            )
            values["all_negative_fp_mass"] = float(
                np.sum(
                    event_multiplicity[all_negative]
                    * fp_rates[name][all_negative]
                )
            )
            replicate_metrics[name] = values
        for metric in METRICS:
            distributions[metric].append(
                replicate_metrics[candidate_name][metric]
                - replicate_metrics[reference_name][metric]
            )
        if (replicate + 1) % 500 == 0:
            print(
                f"[three-expert ceiling bootstrap] "
                f"{replicate + 1}/{replicates}",
                flush=True,
            )

    result: dict[str, Any] = {}
    for metric in METRICS:
        values = np.asarray(distributions[metric], dtype=np.float64)
        point = float(
            point_metrics[candidate_name][metric]
            - point_metrics[reference_name][metric]
        )
        lower_is_better = metric == "all_negative_fp_mass"
        candidate_wins = values < 0.0 if lower_is_better else values > 0.0
        result[metric] = {
            "point_delta": point,
            "ci_95_low": float(np.percentile(values, 2.5)),
            "ci_95_high": float(np.percentile(values, 97.5)),
            "better_direction": "lower" if lower_is_better else "higher",
            "candidate_win_probability": float(np.mean(candidate_wins)),
            "ties_probability": float(np.mean(values == 0.0)),
        }
    return {
        "unit": "canonical_event_cluster",
        "replicates": int(replicates),
        "seed": int(seed),
        "thresholds_refit_per_replicate": False,
        "fp_mass_definition": (
            "multiplicity-weighted sum of within-event hard-FP row rates over "
            "sampled all-negative canonical events"
        ),
        "candidate_minus_reference": result,
    }


def build_payload(
    *,
    p5_path: Path,
    d1_paths: Sequence[Path],
    d7_path: Path,
    replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    paths = {
        "p5": p5_path.expanduser().resolve(),
        **{
            f"d1_seed_{seed}": path.expanduser().resolve()
            for seed, path in zip(DEFAULT_D1_SEEDS, d1_paths)
        },
        "d7_seed_20260728": d7_path.expanduser().resolve(),
    }
    frames = {
        name: global_runner.load_prediction_table(path)
        for name, path in paths.items()
    }
    global_runner.validate_aligned_prediction_frames(frames)
    reference_frame = frames["p5"]
    labels = reference_frame["label"].to_numpy(dtype=np.int64)
    event_ids = reference_frame["event_id"].astype(str).tolist()
    p5_probability = reference_frame["probability"].to_numpy(dtype=np.float64)
    d1_probabilities = [
        frames[f"d1_seed_{seed}"]["probability"].to_numpy(dtype=np.float64)
        for seed in DEFAULT_D1_SEEDS
    ]
    d7_probability = frames["d7_seed_20260728"]["probability"].to_numpy(
        dtype=np.float64
    )
    fusion_probabilities = fixed_fusions(
        p5_probability, d1_probabilities, d7_probability
    )
    evaluated = {
        key: fusion_probabilities[key]
        for key in (
            "p5_d1_equal_logit_reference",
            "p5_d1_d7_equal_thirds_posthoc_ceiling",
        )
    }
    point_metrics = {
        name: global_runner.metric_bundle(labels, probability, event_ids)
        for name, probability in evaluated.items()
    }
    reference_name = "p5_d1_equal_logit_reference"
    candidate_name = "p5_d1_d7_equal_thirds_posthoc_ceiling"
    bootstrap = canonical_event_cluster_bootstrap(
        labels,
        event_ids,
        evaluated,
        point_metrics,
        candidate_name=candidate_name,
        reference_name=reference_name,
        replicates=replicates,
        seed=bootstrap_seed,
    )
    ap_better = bool(
        point_metrics[candidate_name]["event_balanced_ap"]
        > point_metrics[reference_name]["event_balanced_ap"]
    )
    macro_better = bool(
        point_metrics[candidate_name]["event_balanced_macro_f1_selected"]
        > point_metrics[reference_name]["event_balanced_macro_f1_selected"]
    )
    ceiling_screen_passed = bool(ap_better and macro_better)
    return {
        "script_version": SCRIPT_VERSION,
        "audit_type": "fixed-posthoc-three-expert-ceiling-development-only",
        "status": "rejected_not_promotable",
        "post_hoc_exploratory": True,
        "test_or_outer_or_sealed_or_holdout_read": False,
        "rows": int(len(labels)),
        "canonical_events": int(len(set(event_ids))),
        "d1_seeds": list(DEFAULT_D1_SEEDS),
        "d7_seeds": [20260728],
        "fixed_logit_formulas": {
            reference_name: "0.5*logit(P5) + 0.5*mean_seed(logit(D1))",
            candidate_name: (
                "(logit(P5) + mean_seed(logit(D1)) + "
                "logit(D7_seed20260728))/3"
            ),
        },
        "fixed_expert_weights": {
            reference_name: {"p5": 0.5, "d1_three_seed_ensemble": 0.5},
            candidate_name: {
                "p5": 1.0 / 3.0,
                "d1_three_seed_ensemble": 1.0 / 3.0,
                "d7_seed20260728": 1.0 / 3.0,
            },
        },
        "search_contract": {
            "fusion_weight_search": False,
            "threshold_combination_search": False,
            "checkpoint_search": False,
            "seed_search": False,
            "threshold_rule": (
                "one standard event-balanced macro-F1-maximizing threshold "
                "computed independently by metric_bundle for each point "
                "prediction; held fixed in bootstrap"
            ),
        },
        "point_metrics": point_metrics,
        "canonical_event_cluster_bootstrap": bootstrap,
        "ceiling_screen": {
            "rule": (
                "candidate must have strictly higher point event-balanced row "
                "AP and macro-F1 than the P5+D1 reference"
            ),
            "ap_strictly_better": ap_better,
            "macro_f1_strictly_better": macro_better,
            "passed": ceiling_screen_passed,
            "immediate_reject_if_either_not_better": True,
        },
        "promotion": {
            "eligible": False,
            "decision": "reject",
            "reason": (
                "D7 already failed its predeclared promotion gate; this "
                "post-hoc ceiling audit cannot reverse that decision "
                "regardless of point metrics."
            ),
        },
        "identity_contract": {
            "ordered_id_exact": True,
            "ordered_event_id_exact": True,
            "labels_exact": True,
        },
        "provenance": {
            name: {
                "path": str(path),
                "sha256": global_runner.cache_runner.sha256_file(path),
            }
            for name, path in paths.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p5-predictions", default=str(DEFAULT_P5))
    parser.add_argument("--d1-template", default=str(DEFAULT_D1_TEMPLATE))
    parser.add_argument("--d7-predictions", default=str(DEFAULT_D7))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026072807)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.replicates) < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")
    output = Path(args.output_json).expanduser().resolve()
    global_runner.assert_development_path(
        output, purpose="post-hoc ceiling audit output"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    d1_paths = [
        Path(str(args.d1_template).format(seed=seed))
        for seed in DEFAULT_D1_SEEDS
    ]
    payload = build_payload(
        p5_path=Path(args.p5_predictions),
        d1_paths=d1_paths,
        d7_path=Path(args.d7_predictions),
        replicates=int(args.replicates),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    global_runner.atomic_json_write(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
