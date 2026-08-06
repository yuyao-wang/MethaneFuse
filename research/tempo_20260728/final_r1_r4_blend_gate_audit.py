#!/usr/bin/env python3
"""Final zero-training R1/R4 equal-logit and availability-gate audit.

This development-only control combines the final three-seed R1 ensemble and
final four-seed R4 ensemble with fixed 0.5/0.5 logit weights.  The blend's
single global development threshold is also used on the exactly-one-current-
sensor branch of a blend-else-P0 gate.  No ensemble weight or subgroup
threshold is searched, and no test/sealed artifact is accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.tempo_20260728.availability_gate_control import (  # noqa: E402
    fixed_event_audit,
    fixed_metrics,
    paired_event_bootstrap,
)
from research.tempo_20260728.tempo_legacy360_global import (  # noqa: E402
    atomic_json,
    guard_development_path,
    probability_metrics,
    sha256_file,
)


SCRIPT_VERSION = "tempo-final-r1-r4-blend-gate-audit-v1"
DEFAULT_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1"
)
DEFAULT_R1_ROOT = DEFAULT_ROOT / "final_r1_3seed_v3"
DEFAULT_R4_ROOT = DEFAULT_ROOT / "final_r4_4seed_v5"
DEFAULT_OUTPUT = DEFAULT_ROOT / "final_r1_r4_blend_gate_v11"


def logit(probability: np.ndarray) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    value = np.clip(value, 1e-8, 1.0 - 1e-8)
    return np.log(value) - np.log1p(-value)


def sigmoid(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-array))


def command_run(args: argparse.Namespace) -> None:
    r1_predictions_path = Path(args.r1_predictions).expanduser().absolute()
    r1_aggregate_path = Path(args.r1_aggregate).expanduser().absolute()
    r4_predictions_path = Path(args.r4_predictions).expanduser().absolute()
    r4_aggregate_path = Path(args.r4_aggregate).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    for path, role in (
        (r1_predictions_path, "final R1 development predictions"),
        (r1_aggregate_path, "final R1 development aggregate"),
        (r4_predictions_path, "final R4 development predictions"),
        (r4_aggregate_path, "final R4 development aggregate"),
        (output_dir, "development blend output"),
    ):
        guard_development_path(path, role)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    r1 = pd.read_csv(r1_predictions_path)
    r4 = pd.read_csv(r4_predictions_path)
    with r1_aggregate_path.open("r", encoding="utf-8") as stream:
        r1_aggregate = json.load(stream)
    with r4_aggregate_path.open("r", encoding="utf-8") as stream:
        r4_aggregate = json.load(stream)
    if bool(r1_aggregate["test_or_sealed_read"]) or bool(
        r4_aggregate["test_or_sealed_read"]
    ):
        raise RuntimeError("Input aggregate reports test/sealed read.")
    if [int(value) for value in r1_aggregate["seeds"]] != [17, 42, 73]:
        raise RuntimeError("Expected final R1 seeds 17/42/73.")
    if [int(value) for value in r4_aggregate["seeds"]] != [17, 42, 73, 101]:
        raise RuntimeError("Expected final R4 seeds 17/42/73/101.")
    identity_columns = [
        "id",
        "plume_id",
        "event_id",
        "availability_signature",
        "label",
    ]
    if not r1[identity_columns].equals(r4[identity_columns]):
        raise RuntimeError("Final R1/R4 development identities differ.")
    labels = r4["label"].to_numpy(dtype=np.int64)
    event_ids = r4["event_id"].astype(str).to_numpy()
    signatures = r4["availability_signature"].astype(str)
    sensor_count = signatures.map(
        lambda value: len(value.split("+"))
    ).to_numpy(dtype=np.int64)
    if not np.all(sensor_count >= 1):
        raise RuntimeError("Every row must have a current sensor.")
    p0 = r4["p0_probability"].to_numpy(dtype=np.float64)
    r1_score = r1["ensemble_probability"].to_numpy(dtype=np.float64)
    r4_score = r4["ensemble_probability"].to_numpy(dtype=np.float64)
    r1_shuffle = r1[
        "history_shuffle_ensemble_probability"
    ].to_numpy(dtype=np.float64)
    r4_shuffle = r4[
        "history_shuffle_ensemble_probability"
    ].to_numpy(dtype=np.float64)
    blend_score = sigmoid(0.5 * (logit(r1_score) + logit(r4_score)))
    blend_shuffle_score = sigmoid(
        0.5 * (logit(r1_shuffle) + logit(r4_shuffle))
    )
    blend_selected = probability_metrics(
        torch.from_numpy(labels),
        blend_score,
        signatures.tolist(),
    )
    blend_threshold = float(
        blend_selected["best_binary_f1_threshold"]
    )
    p0_threshold = float(
        r4_aggregate["p0"]["metrics"]["best_binary_f1_threshold"]
    )
    r4_threshold = float(
        r4_aggregate["logit_ensemble"]["metrics"][
            "best_binary_f1_threshold"
        ]
    )
    p0_prediction = p0 >= p0_threshold
    r4_prediction = r4_score >= r4_threshold
    blend_prediction = blend_score >= blend_threshold
    use_temporal = sensor_count == 1
    v7_gate_score = np.where(use_temporal, r4_score, p0)
    v7_gate_prediction = np.where(
        use_temporal, r4_prediction, p0_prediction
    )
    blend_gate_score = np.where(use_temporal, blend_score, p0)
    blend_gate_prediction = np.where(
        use_temporal, blend_prediction, p0_prediction
    )
    global_shuffle_prediction = blend_shuffle_score >= blend_threshold
    gate_shuffle_score = np.where(
        use_temporal, blend_shuffle_score, p0
    )
    gate_shuffle_prediction = np.where(
        use_temporal, global_shuffle_prediction, p0_prediction
    )
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "p0": (p0, p0_prediction),
        "frozen_v7_gate": (v7_gate_score, v7_gate_prediction),
        "r1_r4_equal_logit_global": (
            blend_score,
            blend_prediction,
        ),
        "single_sensor_blend_else_p0": (
            blend_gate_score,
            blend_gate_prediction,
        ),
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
        for name, (score, prediction) in arrays.items()
    }
    shuffle = {
        "r1_r4_equal_logit_global": fixed_metrics(
            labels, blend_shuffle_score, global_shuffle_prediction
        ),
        "single_sensor_blend_else_p0": fixed_metrics(
            labels, gate_shuffle_score, gate_shuffle_prediction
        ),
    }
    for name in shuffle:
        for metric in ("binary_f1", "macro_f1", "ap", "auc"):
            shuffle[name][f"delta_{metric}"] = float(
                shuffle[name][metric] - models[name]["metrics"][metric]
            )
    bootstraps = {
        candidate: paired_event_bootstrap(
            labels=labels,
            event_ids=event_ids,
            models=arrays,
            candidate_name=candidate,
            references=["p0", "frozen_v7_gate"],
            repeats=int(args.bootstrap_repeats),
            seed=int(args.bootstrap_seed),
        )
        for candidate in (
            "r1_r4_equal_logit_global",
            "single_sensor_blend_else_p0",
        )
    }
    v7_metrics = models["frozen_v7_gate"]["metrics"]
    gate_metrics = models["single_sensor_blend_else_p0"]["metrics"]
    promotion = bool(
        gate_metrics["binary_f1"]
        >= v7_metrics["binary_f1"] + float(args.minimum_f1_gain)
        and gate_metrics["ap"] >= v7_metrics["ap"]
        and gate_metrics["auc"] >= v7_metrics["auc"]
    )
    result: dict[str, Any] = {
        "schema_version": "tempo-final-r1-r4-blend-gate-audit-v1",
        "script_version": SCRIPT_VERSION,
        "definition": {
            "r1_seeds": [17, 42, 73],
            "r4_seeds": [17, 42, 73, 101],
            "r1_logit_weight": 0.5,
            "r4_logit_weight": 0.5,
            "weight_searched": False,
            "global_blend_threshold": blend_threshold,
            "global_blend_threshold_selected_on_development": True,
            "p0_threshold": p0_threshold,
            "gate_rule": (
                "exactly one current sensor -> fixed equal-logit blend at "
                "its global threshold; otherwise P0 at its global threshold"
            ),
            "subgroup_threshold_searched": False,
        },
        "models": models,
        "history_shuffle_at_frozen_thresholds": shuffle,
        "paired_event_bootstrap": bootstraps,
        "promotion_rule": {
            "minimum_f1_gain_over_v7": float(args.minimum_f1_gain),
            "require_no_ap_regression": True,
            "require_no_auc_regression": True,
            "candidate_passes": promotion,
            "decision": (
                "eligible_to_replace_locked_r4_gate"
                if promotion
                else "do_not_promote_keep_locked_r4"
            ),
        },
        "inputs": {
            "r1_predictions": str(r1_predictions_path),
            "r1_predictions_sha256": sha256_file(r1_predictions_path),
            "r1_aggregate": str(r1_aggregate_path),
            "r1_aggregate_sha256": sha256_file(r1_aggregate_path),
            "r4_predictions": str(r4_predictions_path),
            "r4_predictions_sha256": sha256_file(r4_predictions_path),
            "r4_aggregate": str(r4_aggregate_path),
            "r4_aggregate_sha256": sha256_file(r4_aggregate_path),
        },
        "test_or_sealed_read": False,
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    atomic_json(output_dir / "result.json", result)
    predictions = r4[identity_columns].copy()
    predictions["sensor_count"] = sensor_count
    predictions["p0_probability"] = p0
    predictions["r1_probability"] = r1_score
    predictions["r4_probability"] = r4_score
    predictions["blend_probability"] = blend_score
    predictions["blend_prediction"] = blend_prediction.astype(np.int8)
    predictions["blend_gate_probability"] = blend_gate_score
    predictions["blend_gate_prediction"] = (
        blend_gate_prediction.astype(np.int8)
    )
    predictions.to_csv(output_dir / "dev_predictions.csv", index=False)
    lines = [
        "# Final R1/R4 fixed equal-logit audit",
        "",
        "Development only. Weights are fixed at 0.5/0.5; no test/sealed "
        "artifact was read.",
        "",
        "| Model | F1 | Macro F1 | AP | AUC | FP rows | "
        "all-neg FP events/rows | positive events detected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "p0",
        "frozen_v7_gate",
        "r1_r4_equal_logit_global",
        "single_sensor_blend_else_p0",
    ):
        metric = models[name]["metrics"]
        event = models[name]["event_audit"]
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
            f"Promotion decision: **{result['promotion_rule']['decision']}**.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "models": models,
                "promotion_rule": result["promotion_rule"],
                "test_or_sealed_read": False,
            },
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r1-predictions",
        default=str(DEFAULT_R1_ROOT / "final_ensemble_predictions.csv"),
    )
    parser.add_argument(
        "--r1-aggregate",
        default=str(DEFAULT_R1_ROOT / "final_aggregate.json"),
    )
    parser.add_argument(
        "--r4-predictions",
        default=str(DEFAULT_R4_ROOT / "final_ensemble_predictions.csv"),
    )
    parser.add_argument(
        "--r4-aggregate",
        default=str(DEFAULT_R4_ROOT / "final_aggregate.json"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--minimum-f1-gain", type=float, default=0.001)
    return parser


def main() -> None:
    command_run(build_parser().parse_args())


if __name__ == "__main__":
    main()
