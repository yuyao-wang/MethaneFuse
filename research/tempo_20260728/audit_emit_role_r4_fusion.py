#!/usr/bin/env python3
"""Fixed equal-logit audit of the old EMIT role-only expert and R4 motion."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as cache_runner,
)
from research.tempo_20260728 import tempo_l89_global as metric_runner  # noqa: E402
from research.tempo_20260728.aggregate_single_sensor_r4 import (  # noqa: E402
    paired_event_bootstrap,
)


SCRIPT_VERSION = "emit-role-r4-fixed-fusion-audit-v1"
FORBIDDEN_RE = re.compile(
    r"(^|[._-])(test|sealed|holdout)([._-]|$)", re.IGNORECASE
)
METRIC_KEYS = (
    "event_balanced_positive_f1_selected",
    "event_balanced_macro_f1_selected",
    "event_balanced_ap",
    "event_balanced_auc",
    "row_positive_f1_selected",
    "row_macro_f1_selected",
    "row_ap",
    "row_auc",
    "all_negative_fp_mass",
    "all_negative_events_with_any_fp",
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


def summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": (
            float(array.std(ddof=1)) if len(array) > 1 else 0.0
        ),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def run(args: argparse.Namespace) -> None:
    old_root = assert_development_path(
        Path(args.old_root), purpose="old EMIT result root"
    )
    r4_root = assert_development_path(
        Path(args.r4_root), purpose="R4 result root"
    )
    output_dir = assert_development_path(
        Path(args.output_dir), purpose="fusion audit output"
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    role_seeds = [
        int(value) for value in str(args.role_seeds).split(",")
    ]
    r4_seeds = [int(value) for value in str(args.r4_seeds).split(",")]

    role_frames: list[pd.DataFrame] = []
    role_artifacts: list[dict[str, Any]] = []
    for seed in role_seeds:
        prediction_path = assert_development_path(
            old_root
            / f"emit_ragged_cls_v1_seed{seed}"
            / "role_only"
            / "validation_best_ap_predictions.csv",
            purpose=f"role-only seed {seed} predictions",
        )
        checkpoint_path = assert_development_path(
            old_root
            / f"emit_ragged_cls_v1_seed{seed}"
            / "role_only"
            / "checkpoint_best_ap.pt",
            purpose=f"role-only seed {seed} checkpoint",
        )
        if not prediction_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing role-only artifact for seed {seed}"
            )
        role_frames.append(pd.read_csv(prediction_path))
        role_artifacts.append(
            {
                "seed": seed,
                "prediction_path": str(prediction_path),
                "prediction_sha256": cache_runner.sha256_file(
                    prediction_path
                ),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": cache_runner.sha256_file(
                    checkpoint_path
                ),
            }
        )

    reference = role_frames[0]
    ids = reference["id"].astype(str).tolist()
    labels = reference["label"].to_numpy(dtype=np.int64)
    event_ids = reference["event_id"].astype(str).to_numpy(dtype=object)
    for seed, frame in zip(role_seeds, role_frames):
        if frame["id"].astype(str).tolist() != ids:
            raise RuntimeError(f"Role seed {seed} identity order changed.")
        if not np.array_equal(
            frame["label"].to_numpy(dtype=np.int64), labels
        ):
            raise RuntimeError(f"Role seed {seed} labels changed.")
        if not np.array_equal(
            frame["event_id"].astype(str).to_numpy(dtype=object), event_ids
        ):
            raise RuntimeError(f"Role seed {seed} event IDs changed.")

    role_probabilities = [
        frame["probability"].to_numpy(dtype=np.float64)
        for frame in role_frames
    ]
    role_ensemble = sigmoid(
        np.mean([logit(value) for value in role_probabilities], axis=0)
    )
    role_metrics = metric_runner.metric_bundle(
        labels, role_ensemble, event_ids.tolist()
    )
    role_seed_metrics = [
        metric_runner.metric_bundle(labels, value, event_ids.tolist())
        for value in role_probabilities
    ]

    r4_frames: list[pd.DataFrame] = []
    r4_artifacts: list[dict[str, Any]] = []
    for seed in r4_seeds:
        prediction_path = assert_development_path(
            r4_root
            / f"emit_3time_seed{seed}"
            / f"r4_motion_excitation_seed{seed}"
            / "dev_predictions_best.csv",
            purpose=f"R4 seed {seed} predictions",
        )
        checkpoint_path = assert_development_path(
            r4_root
            / f"emit_3time_seed{seed}"
            / f"r4_motion_excitation_seed{seed}"
            / "checkpoint_best.pt",
            purpose=f"R4 seed {seed} checkpoint",
        )
        if not prediction_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing R4 artifact for seed {seed}")
        frame = pd.read_csv(prediction_path)
        if frame["id"].astype(str).tolist() != ids:
            raise RuntimeError(f"R4 seed {seed} identity order changed.")
        r4_frames.append(frame)
        r4_artifacts.append(
            {
                "seed": seed,
                "prediction_path": str(prediction_path),
                "prediction_sha256": cache_runner.sha256_file(
                    prediction_path
                ),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": cache_runner.sha256_file(
                    checkpoint_path
                ),
            }
        )
    r4_probabilities = [
        frame["probability"].to_numpy(dtype=np.float64)
        for frame in r4_frames
    ]
    r4_ensemble = sigmoid(
        np.mean([logit(value) for value in r4_probabilities], axis=0)
    )
    r4_metrics = metric_runner.metric_bundle(
        labels, r4_ensemble, event_ids.tolist()
    )

    # This is the only fusion evaluated.  No weight search is implemented.
    fusion_weight_role = 0.5
    fusion_weight_r4 = 0.5
    fusion_probability = sigmoid(
        fusion_weight_role * logit(role_ensemble)
        + fusion_weight_r4 * logit(r4_ensemble)
    )
    fusion_metrics = metric_runner.metric_bundle(
        labels, fusion_probability, event_ids.tolist()
    )
    exceeded = {
        "event_balanced_ap": bool(
            fusion_metrics["event_balanced_ap"]
            > role_metrics["event_balanced_ap"]
        ),
        "event_balanced_auc": bool(
            fusion_metrics["event_balanced_auc"]
            > role_metrics["event_balanced_auc"]
        ),
        "event_balanced_positive_f1_selected": bool(
            fusion_metrics["event_balanced_positive_f1_selected"]
            > role_metrics["event_balanced_positive_f1_selected"]
        ),
        "event_balanced_macro_f1_selected": bool(
            fusion_metrics["event_balanced_macro_f1_selected"]
            > role_metrics["event_balanced_macro_f1_selected"]
        ),
    }
    bootstrap = None
    if exceeded["event_balanced_ap"] and exceeded["event_balanced_auc"]:
        bootstrap = paired_event_bootstrap(
            labels=labels,
            event_ids=event_ids,
            p0_probability=role_ensemble,
            candidate_probability=fusion_probability,
            p0_threshold=float(role_metrics["selected_threshold"]),
            candidate_threshold=float(fusion_metrics["selected_threshold"]),
            replicates=int(args.bootstrap_replicates),
            seed=int(args.bootstrap_seed),
        )
        bootstrap["contrast"] = (
            "fixed 0.5 role-only/R4 equal-logit fusion minus "
            "role-only equal-logit ensemble"
        )
        bootstrap["thresholds_fixed_before_resampling"] = {
            "role_only": float(role_metrics["selected_threshold"]),
            "fusion": float(fusion_metrics["selected_threshold"]),
        }

    cache_runner.atomic_csv_write(
        output_dir / "fusion_predictions.csv",
        pd.DataFrame(
            {
                "id": ids,
                "event_id": event_ids,
                "label": labels,
                "role_only_3seed_probability": role_ensemble,
                "r4_4seed_probability": r4_ensemble,
                "fixed_equal_logit_probability": fusion_probability,
            }
        ),
    )
    output = {
        "script_version": SCRIPT_VERSION,
        "role_seeds": role_seeds,
        "r4_seeds": r4_seeds,
        "fixed_fusion": {
            "logit_weights": {
                "role_only": fusion_weight_role,
                "r4": fusion_weight_r4,
            },
            "weight_search_performed": False,
        },
        "role_only_per_seed_summary": {
            key: summary([metrics[key] for metrics in role_seed_metrics])
            for key in METRIC_KEYS
        },
        "role_only_equal_logit": role_metrics,
        "r4_equal_logit": r4_metrics,
        "fixed_role_r4_equal_logit": fusion_metrics,
        "fusion_minus_role_only": {
            key: float(fusion_metrics[key] - role_metrics[key])
            for key in METRIC_KEYS
        },
        "fusion_exceeded_role_only": exceeded,
        "paired_canonical_event_bootstrap": bootstrap,
        "artifacts": {
            "role_only": role_artifacts,
            "r4": r4_artifacts,
        },
        "metric_boundary": {
            "event_balanced": (
                "each canonical event has equal total row weight"
            ),
            "row": "ordinary row-weighted metric",
            "thresholds": (
                "each model selects one inner-dev event-balanced macro-F1 "
                "threshold before bootstrap; bootstrap never refits"
            ),
        },
        "test_or_sealed_or_holdout_read": False,
    }
    cache_runner.atomic_json_write(output_dir / "audit.json", output)

    lines = [
        "# EMIT role-only + R4 fixed fusion audit",
        "",
        "Development-only. No test, sealed, or holdout artifact was read.",
        "The only fusion is a predeclared `0.5/0.5` equal-logit average; no "
        "weight was searched.",
        "",
        "| Model | event-F1 | event-macro | event-AP | event-AUC | row-F1 | row-AP | FP mass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (
        ("Old role-only 3-seed", role_metrics),
        ("R4 4-seed", r4_metrics),
        ("Fixed 0.5 role+R4", fusion_metrics),
    ):
        lines.append(
            f"| {name} | "
            f"{metrics['event_balanced_positive_f1_selected']:.6f} | "
            f"{metrics['event_balanced_macro_f1_selected']:.6f} | "
            f"{metrics['event_balanced_ap']:.6f} | "
            f"{metrics['event_balanced_auc']:.6f} | "
            f"{metrics['row_positive_f1_selected']:.6f} | "
            f"{metrics['row_ap']:.6f} | "
            f"{metrics['all_negative_fp_mass']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The old role-only TransientQuery remains the strong primary "
            "baseline. R4 is evaluated as a complementary motion expert, not "
            "only against the weaker t0-only P0.",
            "",
        ]
    )
    if bootstrap is not None:
        lines.extend(
            [
                "## Paired canonical-event bootstrap: fusion minus role-only",
                "",
                "| Metric | mean delta | 95% CI | win probability |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, record in bootstrap["metrics"].items():
            lines.append(
                f"| {key} | {record['mean_delta']:+.6f} | "
                f"[{record['ci95_percentile'][0]:+.6f},"
                f"{record['ci95_percentile'][1]:+.6f}] | "
                f"{record['win_probability']:.4f} |"
            )
        lines.append("")
    (output_dir / "AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-root",
        default="/diniuvol/yuyao/methanefuse_research_20260727/results",
    )
    parser.add_argument(
        "--r4-root",
        default=(
            "/diniuvol/yuyao/methanefuse_tempo_20260728/"
            "cross_sensor_r4_v1"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--role-seeds", default="20260727,20260728,20260729"
    )
    parser.add_argument("--r4-seeds", default="17,42,73,20260728")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
