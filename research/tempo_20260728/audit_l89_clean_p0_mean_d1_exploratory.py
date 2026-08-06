#!/usr/bin/env python3
"""Audit fixed P0 + three-seed mean-D1 fusion on clean inner development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from research.pretraining_20260727 import l89_ragged_cls_experiment as cache
from research.tempo_20260728 import audit_l89_clean_fixed_ensemble as bootstrap
from research.tempo_20260728 import audit_l89_clean_p0_d1_exploratory as single
from research.tempo_20260728 import tempo_l89_global as tempo


SEEDS = (20260727, 20260728, 20260729)
FUSION_WEIGHT = 0.5
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2026072813


def render_markdown(payload: Mapping[str, Any]) -> str:
    point = payload["point_metrics"]
    paired = payload["paired_event_cluster_bootstrap"]
    lines = [
        "# Fresh clean-L89 P0 + three-seed mean-D1 audit",
        "",
        "**Exploratory inner development only; not a test or SOTA claim.**",
        "",
        "The D1 configuration was the mechanism-first default fixed before the",
        "CPU hyperparameter screen. D1 seeds are averaged in logit space, then",
        "combined with P0 using the fixed formula",
        "`0.5*logit(P0) + 0.5*mean_seed(logit(D1))`.",
        "",
        "| System | Event-balanced AP | AUC | Macro-F1 | Positive F1 | Negative-event FP mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("p0", "mean_seed_d1", "fixed_p0_mean_d1"):
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
            "Paired canonical-event-cluster bootstrap (2,000 replicates; point",
            "thresholds fixed):",
            "",
            "| Comparison | Metric | Delta | 95% CI | Win probability |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparison in ("fusion_minus_p0", "fusion_minus_mean_d1"):
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
            "Guardrail: the three-seed mean reduces initialization noise, but all",
            "three seeds still share one inner-development split. Promotion",
            "requires the frozen formal replicate and a separately authorized",
            "outer/sealed evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if tuple(args.seeds) != SEEDS:
        raise ValueError(f"Seeds must be exactly {SEEDS}.")
    if len(args.d1_predictions) != len(SEEDS):
        raise ValueError("Exactly three D1 prediction paths are required.")
    if int(args.replicates) != BOOTSTRAP_REPLICATES:
        raise ValueError(f"Audit is frozen to {BOOTSTRAP_REPLICATES} replicates.")
    if int(args.bootstrap_seed) != BOOTSTRAP_SEED:
        raise ValueError(f"Audit seed is frozen to {BOOTSTRAP_SEED}.")

    p0_path = Path(args.p0_predictions).expanduser().resolve()
    d1_paths = [
        Path(value).expanduser().resolve() for value in args.d1_predictions
    ]
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path, purpose in (
        (p0_path, "clean exploratory P0 predictions"),
        *[
            (path, f"clean exploratory D1 seed {seed} predictions")
            for seed, path in zip(SEEDS, d1_paths)
        ],
        (output_dir, "clean exploratory mean-D1 fusion output"),
    ):
        tempo.assert_development_path(path, purpose=purpose)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    frames = {"p0": tempo.load_prediction_table(p0_path)}
    for seed, path in zip(SEEDS, d1_paths):
        frames[f"d1_seed_{seed}"] = tempo.load_prediction_table(path)
    bootstrap.aligned_frames(frames)
    reference = frames["p0"]
    labels = reference["label"].to_numpy(dtype=np.int64)
    event_ids = reference["event_id"].astype(str).tolist()
    p0_probability = reference["probability"].to_numpy(dtype=np.float64)
    d1_logits = np.stack(
        [
            single.logit(
                frames[f"d1_seed_{seed}"]["probability"].to_numpy(
                    dtype=np.float64
                )
            )
            for seed in SEEDS
        ],
        axis=0,
    )
    mean_d1_probability = single.sigmoid(d1_logits.mean(axis=0))
    fusion_probability = single.sigmoid(
        FUSION_WEIGHT * single.logit(p0_probability)
        + FUSION_WEIGHT * d1_logits.mean(axis=0)
    )
    probabilities = {
        "p0": p0_probability,
        "mean_seed_d1": mean_d1_probability,
        "fixed_p0_mean_d1": fusion_probability,
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
            ("fusion_minus_p0", "fixed_p0_mean_d1", "p0"),
            (
                "fusion_minus_mean_d1",
                "fixed_p0_mean_d1",
                "mean_seed_d1",
            ),
        ),
        replicates=int(args.replicates),
        seed=int(args.bootstrap_seed),
    )

    output_dir.mkdir(parents=True)
    prediction_path = output_dir / "fixed_p0_mean_d1_predictions.csv"
    prediction_frame = reference.loc[
        :, ["id", "plume_id", "event_id", "label"]
    ].copy()
    prediction_frame["probability"] = fusion_probability
    prediction_frame["selected_threshold"] = float(
        point_metrics["fixed_p0_mean_d1"]["selected_threshold"]
    )
    cache.atomic_csv_write(prediction_path, prediction_frame)
    mean_d1_path = output_dir / "mean_seed_d1_predictions.csv"
    mean_d1_frame = prediction_frame.copy()
    mean_d1_frame["probability"] = mean_d1_probability
    mean_d1_frame["selected_threshold"] = float(
        point_metrics["mean_seed_d1"]["selected_threshold"]
    )
    cache.atomic_csv_write(mean_d1_path, mean_d1_frame)

    payload = {
        "schema_version": "l89-clean-exploratory-p0-mean-d1-fusion-v1",
        "scope": "fresh clean inner-development three-seed diagnostic",
        "arithmetic": {
            "d1_seed_aggregation": "mean of three logits",
            "d1_seeds": list(SEEDS),
            "formula": "0.5*logit(P0)+0.5*mean_seed(logit(D1))",
            "weight_search_performed": False,
            "configuration": {
                "name": "c0_default_mechanism_first",
                "temporal_dim": 192,
                "learning_rate": 8e-4,
                "weight_decay": 0.02,
                "dropout": 0.15,
            },
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
            "d1": {
                str(seed): {
                    "path": str(path),
                    "sha256": cache.sha256_file(path),
                }
                for seed, path in zip(SEEDS, d1_paths)
            },
        },
        "prediction_outputs": {
            "mean_seed_d1": {
                "path": str(mean_d1_path),
                "sha256": cache.sha256_file(mean_d1_path),
            },
            "fixed_p0_mean_d1": {
                "path": str(prediction_path),
                "sha256": cache.sha256_file(prediction_path),
            },
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


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-predictions", required=True)
    parser.add_argument("--d1-predictions", nargs="+", required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=SEEDS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
