#!/usr/bin/env python3
"""Audit the predeclared 0.5-logit P0 + single-seed D1 clean-dev fusion.

This is an exploratory, inner-development-only diagnostic.  It does not tune
the fusion weight.  Point thresholds are selected once per system and then
held fixed in every canonical-event-cluster bootstrap replicate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.pretraining_20260727 import l89_ragged_cls_experiment as cache
from research.tempo_20260728 import audit_l89_clean_fixed_ensemble as bootstrap
from research.tempo_20260728 import tempo_l89_global as tempo


FUSION_WEIGHT = 0.5
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2026072812


def logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values) - np.log1p(-values)


def sigmoid(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.where(
        array >= 0.0,
        1.0 / (1.0 + np.exp(-array)),
        np.exp(array) / (1.0 + np.exp(array)),
    )


def render_markdown(payload: Mapping[str, Any]) -> str:
    point = payload["point_metrics"]
    paired = payload["paired_event_cluster_bootstrap"]
    lines = [
        "# Fresh clean-L89 exploratory P0 + D1 audit",
        "",
        "This is a one-seed, inner-development-only diagnostic. The fusion is",
        "predeclared as `0.5*logit(P0) + 0.5*logit(D1)`; no fusion weight was",
        "searched.",
        "",
        "| System | Event-balanced AP | AUC | Macro-F1 | Positive F1 | Negative-event FP mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("p0", "d1", "fixed_p0_d1"):
        metric = point[name]
        lines.append(
            f"| {name} | {metric['event_balanced_ap']:.6f} | "
            f"{metric['event_balanced_auc']:.6f} | "
            f"{metric['event_balanced_macro_f1_selected']:.6f} | "
            f"{metric['event_balanced_positive_f1_selected']:.6f} | "
            f"{metric['all_negative_fp_mass']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Paired canonical-event-cluster bootstrap (2,000 replicates; each",
            "system's point threshold is fixed across replicates):",
            "",
            "| Comparison | Metric | Delta | 95% CI | Win probability |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparison in ("fusion_minus_p0", "fusion_minus_d1"):
        for metric, result in paired[comparison].items():
            lines.append(
                f"| {comparison} | {metric} | {result['point']:+.6f} | "
                f"[{result['ci_95_low']:+.6f}, "
                f"{result['ci_95_high']:+.6f}] | "
                f"{result['win_probability']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Interpretation guardrail: this result can motivate the architecture",
            "and the preregistered multi-seed/formal replicate, but it cannot be",
            "used as a test-set or SOTA claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.replicates) != BOOTSTRAP_REPLICATES:
        raise ValueError(f"Exploratory audit is frozen to {BOOTSTRAP_REPLICATES} replicates.")
    if int(args.bootstrap_seed) != BOOTSTRAP_SEED:
        raise ValueError(f"Exploratory audit seed is frozen to {BOOTSTRAP_SEED}.")

    p0_path = Path(args.p0_predictions).expanduser().resolve()
    d1_path = Path(args.d1_predictions).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path, purpose in (
        (p0_path, "clean exploratory P0 predictions"),
        (d1_path, "clean exploratory D1 predictions"),
        (output_dir, "clean exploratory fusion output"),
    ):
        tempo.assert_development_path(path, purpose=purpose)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    frames = {
        "p0": tempo.load_prediction_table(p0_path),
        "d1": tempo.load_prediction_table(d1_path),
    }
    bootstrap.aligned_frames(frames)
    reference = frames["p0"]
    labels = reference["label"].to_numpy(dtype=np.int64)
    event_ids = reference["event_id"].astype(str).tolist()
    p0_probability = frames["p0"]["probability"].to_numpy(dtype=np.float64)
    d1_probability = frames["d1"]["probability"].to_numpy(dtype=np.float64)
    fusion_probability = sigmoid(
        FUSION_WEIGHT * logit(p0_probability)
        + FUSION_WEIGHT * logit(d1_probability)
    )
    probabilities = {
        "p0": p0_probability,
        "d1": d1_probability,
        "fixed_p0_d1": fusion_probability,
    }
    point_metrics = {
        name: tempo.metric_bundle(labels, values, event_ids)
        for name, values in probabilities.items()
    }
    paired = bootstrap.fixed_threshold_event_bootstrap(
        labels,
        event_ids,
        probabilities,
        point_metrics,
        (
            ("fusion_minus_p0", "fixed_p0_d1", "p0"),
            ("fusion_minus_d1", "fixed_p0_d1", "d1"),
        ),
        replicates=int(args.replicates),
        seed=int(args.bootstrap_seed),
    )

    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "fixed_p0_d1_predictions.csv"
    predictions = reference.loc[:, ["id", "plume_id", "event_id", "label"]].copy()
    predictions["probability"] = fusion_probability
    predictions["selected_threshold"] = float(
        point_metrics["fixed_p0_d1"]["selected_threshold"]
    )
    cache.atomic_csv_write(predictions_path, predictions)
    payload = {
        "schema_version": "l89-clean-exploratory-p0-single-d1-fusion-v1",
        "scope": "fresh clean inner-development exploratory diagnostic",
        "arithmetic": {
            "formula": "0.5*logit(P0)+0.5*logit(single_seed_D1)",
            "p0_logit_weight": FUSION_WEIGHT,
            "d1_logit_weight": FUSION_WEIGHT,
            "weight_search_performed": False,
            "single_d1_seed": 20260728,
        },
        "point_metrics": point_metrics,
        "paired_event_cluster_bootstrap": paired,
        "bootstrap": {
            "unit": "canonical event_id",
            "replicates": int(args.replicates),
            "seed": int(args.bootstrap_seed),
            "point_selected_thresholds_fixed_per_system": True,
            "thresholds_refit_per_replicate": False,
        },
        "input_provenance": {
            "p0": {"path": str(p0_path), "sha256": cache.sha256_file(p0_path)},
            "d1": {"path": str(d1_path), "sha256": cache.sha256_file(d1_path)},
        },
        "prediction_output": {
            "path": str(predictions_path),
            "sha256": cache.sha256_file(predictions_path),
        },
        "rows": int(len(reference)),
        "events": int(len(set(event_ids))),
        "event_balanced_metric_definition": (
            "row predictions with inverse canonical-event-size weights"
        ),
        "test_or_sealed_or_holdout_or_outer_read": False,
        "promotion_eligible": False,
    }
    cache.atomic_json_write(output_dir / "RESULT.json", payload)
    (output_dir / "RESULT.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-predictions", required=True)
    parser.add_argument("--d1-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
