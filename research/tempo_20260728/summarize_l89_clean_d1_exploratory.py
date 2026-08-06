#!/usr/bin/env python3
"""Select and summarize the bounded clean-inner D1 hyperparameter screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.pretraining_20260727 import (
    l89_ragged_cls_experiment as cache_runner,
)
from research.tempo_20260728 import tempo_l89_global as tempo


SCRIPT_VERSION = "l89-clean-d1-exploratory-screen-summary-v1"
SCREEN_SEED = 20260728
THREE_SEEDS = (20260727, 20260728, 20260729)
AP_TIE_BAND = 5e-4
FORBIDDEN_RE = re.compile(
    r"(^|[._/\\-])(test|sealed|holdout|outer)([._/\\-]|$)",
    re.IGNORECASE,
)
CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "c0_default",
        "temporal_dim": 192,
        "learning_rate": 8e-4,
        "weight_decay": 0.02,
        "dropout": 0.15,
    },
    {
        "name": "c1_lr4e4",
        "temporal_dim": 192,
        "learning_rate": 4e-4,
        "weight_decay": 0.02,
        "dropout": 0.15,
    },
    {
        "name": "c2_lr12e3",
        "temporal_dim": 192,
        "learning_rate": 1.2e-3,
        "weight_decay": 0.02,
        "dropout": 0.15,
    },
    {
        "name": "c3_dim128",
        "temporal_dim": 128,
        "learning_rate": 8e-4,
        "weight_decay": 0.02,
        "dropout": 0.15,
    },
    {
        "name": "c4_dim256",
        "temporal_dim": 256,
        "learning_rate": 8e-4,
        "weight_decay": 0.02,
        "dropout": 0.15,
    },
    {
        "name": "c5_dropout30",
        "temporal_dim": 192,
        "learning_rate": 8e-4,
        "weight_decay": 0.02,
        "dropout": 0.30,
    },
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse(path: Path, *, purpose: str) -> Path:
    resolved = path.expanduser().resolve()
    if FORBIDDEN_RE.search(str(resolved)):
        raise ValueError(f"{purpose} is held-out-like and forbidden: {resolved}")
    cache_runner.assert_not_sealed_path(resolved, purpose=purpose)
    return resolved


def metric_projection(metrics: Mapping[str, Any]) -> dict[str, float]:
    names = (
        "event_balanced_ap",
        "event_balanced_auc",
        "event_balanced_macro_f1_selected",
        "event_balanced_positive_f1_selected",
        "all_negative_fp_mass",
        "selected_threshold",
    )
    return {name: float(metrics[name]) for name in names}


def validate_screen_run(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    config_root = root / "screen" / str(config["name"])
    status_path = config_root / "run_status.json"
    summary_path = config_root / f"seed_{SCREEN_SEED}" / "summary.json"
    run_config_path = config_root / "run_config.json"
    for path in (status_path, summary_path, run_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    status = read_json(status_path)
    summary = read_json(summary_path)
    run_config = read_json(run_config_path)
    if status.get("status") != "complete":
        raise ValueError(f"{config['name']} did not complete.")
    if run_config.get("resolved_arms") != ["p0_base", "d1_gated_delta"]:
        raise ValueError(f"{config['name']} arm contract changed.")
    if int(run_config["epochs"]) > 4 or int(run_config["patience"]) != 1:
        raise ValueError(f"{config['name']} exceeded the screen budget.")
    if int(run_config["seeds"]) != SCREEN_SEED:
        raise ValueError(f"{config['name']} did not use the screen seed.")
    for key in ("temporal_dim", "learning_rate", "weight_decay", "dropout"):
        observed = float(run_config[key])
        expected = float(config[key])
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{config['name']} {key}={observed} differs from {expected}."
            )
    if summary.get("test_or_sealed_or_holdout_read") is not False:
        raise ValueError(f"{config['name']} held-out guardrail is absent.")
    if int(summary["cache_audit"]["train_dev_event_overlap"]) != 0:
        raise ValueError(f"{config['name']} has event overlap.")
    d1 = summary["results"]["d1_gated_delta"]
    p0 = summary["results"]["p0_base"]
    replay = p0["zero_initialized_d1_prototype_exact_p0_replay"]
    early = d1["early_stop_receipt"]
    shuffle_validity = d1["history_shuffle_validity"]
    if replay.get("pass") is not True:
        raise ValueError(f"{config['name']} failed exact epoch-zero P0 replay.")
    if early.get("valid") is not True:
        raise ValueError(f"{config['name']} has invalid early stopping.")
    if shuffle_validity.get("valid") is not True:
        raise ValueError(f"{config['name']} has invalid history shuffle.")
    real = metric_projection(d1["best"]["validation"])
    shuffled = metric_projection(
        d1["history_shuffle_fixed_model_and_threshold"]
    )
    return {
        **dict(config),
        "selected_epoch": int(d1["best"]["epoch"]),
        "epochs_observed": int(early["epochs_observed"]),
        "real": real,
        "p0": metric_projection(p0["best"]["validation"]),
        "history_shuffle_fixed_threshold": shuffled,
        "history_shuffle_delta": {
            key: float(d1["history_shuffle_delta"][key])
            for key in (
                "event_balanced_ap",
                "event_balanced_auc",
                "event_balanced_macro_f1_selected",
                "event_balanced_positive_f1_selected",
                "all_negative_fp_mass",
            )
        },
        "exact_p0_epoch_zero_replay": True,
        "history_shuffle_valid": True,
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "prediction_csv": str(
            config_root
            / f"seed_{SCREEN_SEED}"
            / "d1_gated_delta_best_event_ap_predictions.csv"
        ),
    }


def write_screen_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# L89 clean-inner D1 bounded exploratory screen",
        "",
        "**Exploratory inner development only. No outer/test/sealed/holdout "
        "artifact was read.**",
        "",
        "Selection used event-balanced AP only. Macro-F1, positive-F1, FP mass, "
        "and history shuffle were recorded but were not used to choose a "
        "configuration. Configurations within 0.0005 AP of the maximum were "
        "treated as tied and resolved by the pre-registered configuration order.",
        "",
        "| config | dim | lr | dropout | epoch | AP | AUC | macro-F1 | pos-F1 | FPmass | shuffle ΔAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["screen"]:
        real = row["real"]
        lines.append(
            f"| {row['name']} | {row['temporal_dim']} | "
            f"{row['learning_rate']:.4g} | {row['dropout']:.2f} | "
            f"{row['selected_epoch']} | {real['event_balanced_ap']:.6f} | "
            f"{real['event_balanced_auc']:.6f} | "
            f"{real['event_balanced_macro_f1_selected']:.6f} | "
            f"{real['event_balanced_positive_f1_selected']:.6f} | "
            f"{real['all_negative_fp_mass']:.3f} | "
            f"{row['history_shuffle_delta']['event_balanced_ap']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Frozen choice: `{payload['selected_config']['name']}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def command_select(args: argparse.Namespace) -> None:
    root = refuse(Path(args.root), purpose="screen root")
    rows = [validate_screen_run(root, config) for config in CONFIGS]
    maximum_ap = max(row["real"]["event_balanced_ap"] for row in rows)
    eligible = [
        row
        for row in rows
        if row["real"]["event_balanced_ap"] >= maximum_ap - AP_TIE_BAND
    ]
    # ``rows`` preserves the pre-registered order; no diagnostic metric enters
    # this tie break.
    selected = eligible[0]
    p0_shas = {
        read_json(Path(row["summary"]))["frozen_role_only_base_audit"][
            "checkpoint_sha256"
        ]
        for row in rows
    }
    if len(p0_shas) != 1:
        raise ValueError("Screen configurations did not share one exact P0.")
    payload = {
        "script_version": SCRIPT_VERSION,
        "scope": "bounded clean-inner exploratory hyperparameter screen",
        "screen_seed": SCREEN_SEED,
        "configuration_count": len(rows),
        "maximum_epochs_per_configuration": 4,
        "patience": 1,
        "selection_metric": "event_balanced_ap",
        "selection_rule": (
            "maximum event-balanced AP; values within 0.0005 absolute AP are "
            "a tie, resolved by pre-registered CONFIGS order"
        ),
        "diagnostics_not_used_for_selection": [
            "event_balanced_auc",
            "event_balanced_macro_f1_selected",
            "event_balanced_positive_f1_selected",
            "all_negative_fp_mass",
            "history_shuffle_delta",
        ],
        "screen": rows,
        "maximum_observed_ap": maximum_ap,
        "tie_band": AP_TIE_BAND,
        "tie_eligible": [row["name"] for row in eligible],
        "selected_config": selected,
        "shared_p0_checkpoint_sha256": next(iter(p0_shas)),
        "next_step": (
            "Run exactly the frozen selected configuration at seeds "
            "20260727,20260728,20260729; do not reopen configuration search."
        ),
        "test_or_sealed_or_holdout_or_outer_read": False,
    }
    cache_runner.atomic_json_write(root / "SCREEN_SELECTION.json", payload)
    pd.DataFrame(
        [
            {
                "config": row["name"],
                "temporal_dim": row["temporal_dim"],
                "learning_rate": row["learning_rate"],
                "weight_decay": row["weight_decay"],
                "dropout": row["dropout"],
                "selected_epoch": row["selected_epoch"],
                **row["real"],
                "history_shuffle_delta_ap": row["history_shuffle_delta"][
                    "event_balanced_ap"
                ],
            }
            for row in rows
        ]
    ).to_csv(root / "SCREEN_TABLE.csv", index=False)
    write_screen_markdown(root / "SCREEN_RESULTS.md", payload)
    argument_lines = [
        "--temporal-dim",
        str(selected["temporal_dim"]),
        "--learning-rate",
        str(selected["learning_rate"]),
        "--weight-decay",
        str(selected["weight_decay"]),
        "--dropout",
        str(selected["dropout"]),
    ]
    (root / "BEST_ARGS.txt").write_text(
        "\n".join(argument_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def aligned_prediction_frames(paths: Sequence[Path]) -> list[pd.DataFrame]:
    frames = [pd.read_csv(path) for path in paths]
    tempo.validate_aligned_prediction_frames(
        {f"seed_{seed}": frame for seed, frame in zip(THREE_SEEDS, frames)}
    )
    return frames


def command_finalize(args: argparse.Namespace) -> None:
    root = refuse(Path(args.root), purpose="screen root")
    selection = read_json(root / "SCREEN_SELECTION.json")
    three_root = root / "three_seed"
    status = read_json(three_root / "run_status.json")
    aggregate = read_json(three_root / "aggregate.json")
    if status.get("status") != "complete":
        raise ValueError("Three-seed run is incomplete.")
    observed_seeds = tuple(int(value) for value in status["seeds"])
    if observed_seeds != THREE_SEEDS:
        raise ValueError(f"Unexpected three-seed order: {observed_seeds}")
    selected = selection["selected_config"]
    three_config = read_json(three_root / "run_config.json")
    for key in ("temporal_dim", "learning_rate", "weight_decay", "dropout"):
        if not math.isclose(
            float(three_config[key]),
            float(selected[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Three-seed {key} differs from frozen selection.")
    summaries = [
        read_json(three_root / f"seed_{seed}" / "summary.json")
        for seed in THREE_SEEDS
    ]
    if not all(
        summary["results"]["d1_gated_delta"]["early_stop_receipt"]["valid"]
        and summary["results"]["d1_gated_delta"]["history_shuffle_validity"][
            "valid"
        ]
        and summary["results"]["p0_base"][
            "zero_initialized_d1_prototype_exact_p0_replay"
        ]["pass"]
        for summary in summaries
    ):
        raise ValueError("Three-seed mechanism receipts are incomplete.")
    d1_paths = [
        three_root
        / f"seed_{seed}"
        / "d1_gated_delta_best_event_ap_predictions.csv"
        for seed in THREE_SEEDS
    ]
    frames = aligned_prediction_frames(d1_paths)
    reference = frames[0]
    labels = reference["label"].to_numpy(dtype=np.int64)
    events = reference["event_id"].astype(str).tolist()
    logits = np.stack(
        [
            tempo.safe_logit(frame["probability"].to_numpy(dtype=np.float64))
            for frame in frames
        ],
        axis=0,
    )
    ensemble_probability = 1.0 / (1.0 + np.exp(-logits.mean(axis=0)))
    ensemble_metrics = tempo.metric_bundle(labels, ensemble_probability, events)
    p0_path = three_root / f"seed_{THREE_SEEDS[0]}" / "p0_base_predictions.csv"
    p0_frame = pd.read_csv(p0_path)
    tempo.validate_aligned_prediction_frames(
        {"p0": p0_frame, "ensemble": reference}
    )
    p0_probability = p0_frame["probability"].to_numpy(dtype=np.float64)
    p0_metrics = tempo.metric_bundle(labels, p0_probability, events)
    bootstrap = tempo.paired_event_bootstrap_deltas(
        labels,
        events,
        {"p0": p0_probability, "d1_seed_logit_mean": ensemble_probability},
        {"p0": p0_metrics, "d1_seed_logit_mean": ensemble_metrics},
        (("d1_seed_logit_mean_minus_p0", "d1_seed_logit_mean", "p0"),),
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
    )
    ensemble_frame = reference.copy()
    ensemble_frame["probability"] = ensemble_probability
    ensemble_frame["selected_threshold"] = float(
        ensemble_metrics["selected_threshold"]
    )
    ensemble_frame["prediction"] = (
        ensemble_probability >= float(ensemble_metrics["selected_threshold"])
    ).astype(np.int64)
    ensemble_path = root / "THREE_SEED_LOGIT_ENSEMBLE_PREDICTIONS.csv"
    cache_runner.atomic_csv_write(ensemble_path, ensemble_frame)

    per_seed = []
    for seed, summary in zip(THREE_SEEDS, summaries):
        d1 = summary["results"]["d1_gated_delta"]
        per_seed.append(
            {
                "seed": seed,
                "selected_epoch": int(d1["best"]["epoch"]),
                "real": metric_projection(d1["best"]["validation"]),
                "history_shuffle_fixed_threshold": metric_projection(
                    d1["history_shuffle_fixed_model_and_threshold"]
                ),
                "history_shuffle_delta": {
                    key: float(d1["history_shuffle_delta"][key])
                    for key in (
                        "event_balanced_ap",
                        "event_balanced_auc",
                        "event_balanced_macro_f1_selected",
                        "event_balanced_positive_f1_selected",
                        "all_negative_fp_mass",
                    )
                },
            }
        )
    result = {
        "script_version": SCRIPT_VERSION,
        "scope": "bounded clean-inner exploratory result",
        "selected_config": selected,
        "seeds": list(THREE_SEEDS),
        "p0": p0_metrics,
        "per_seed": per_seed,
        "three_seed_metric_mean_and_sd": aggregate["arms"][
            "d1_gated_delta"
        ],
        "fixed_equal_seed_logit_ensemble": ensemble_metrics,
        "paired_event_bootstrap": {
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.bootstrap_seed),
            "thresholds_frozen_at_full_development_point": True,
            "deltas": bootstrap,
        },
        "ensemble_predictions": str(ensemble_path),
        "ensemble_predictions_sha256": sha256_file(ensemble_path),
        "history_shuffle_interpretation": (
            "availability-stratified cross-event donors; only history "
            "features replaced, t0/labels/availability/gaps/quality retained"
        ),
        "selection_closed_after_single_seed_screen": True,
        "no_additional_configuration_or_weight_search": True,
        "test_or_sealed_or_holdout_or_outer_read": False,
    }
    cache_runner.atomic_json_write(root / "FINAL_RESULT.json", result)
    lines = [
        "# L89 clean-inner D1 exploratory final",
        "",
        "**Inner development only; not outer, confirmatory, or a SOTA claim.**",
        "",
        f"Frozen config: `{selected['name']}`; dim={selected['temporal_dim']}, "
        f"lr={selected['learning_rate']}, dropout={selected['dropout']}.",
        "",
        "| result | AP | AUC | macro-F1 | positive-F1 | FPmass |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| exact P0 | {p0_metrics['event_balanced_ap']:.6f} | "
            f"{p0_metrics['event_balanced_auc']:.6f} | "
            f"{p0_metrics['event_balanced_macro_f1_selected']:.6f} | "
            f"{p0_metrics['event_balanced_positive_f1_selected']:.6f} | "
            f"{p0_metrics['all_negative_fp_mass']:.3f} |"
        ),
    ]
    for row in per_seed:
        value = row["real"]
        lines.append(
            f"| D1 seed {row['seed']} | "
            f"{value['event_balanced_ap']:.6f} | "
            f"{value['event_balanced_auc']:.6f} | "
            f"{value['event_balanced_macro_f1_selected']:.6f} | "
            f"{value['event_balanced_positive_f1_selected']:.6f} | "
            f"{value['all_negative_fp_mass']:.3f} |"
        )
    lines.append(
        f"| fixed 3-seed logit mean | "
        f"{ensemble_metrics['event_balanced_ap']:.6f} | "
        f"{ensemble_metrics['event_balanced_auc']:.6f} | "
        f"{ensemble_metrics['event_balanced_macro_f1_selected']:.6f} | "
        f"{ensemble_metrics['event_balanced_positive_f1_selected']:.6f} | "
        f"{ensemble_metrics['all_negative_fp_mass']:.3f} |"
    )
    lines.extend(
        [
            "",
            "The six-configuration screen was closed before the three-seed "
            "replicate. No blend weight or subgroup gate was tuned.",
            "",
        ]
    )
    (root / "FINAL_RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--root", required=True)
    select.set_defaults(handler=command_select)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--root", required=True)
    finalize.add_argument("--bootstrap-replicates", type=int, default=2000)
    finalize.add_argument("--bootstrap-seed", type=int, default=2026072817)
    finalize.set_defaults(handler=command_finalize)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
