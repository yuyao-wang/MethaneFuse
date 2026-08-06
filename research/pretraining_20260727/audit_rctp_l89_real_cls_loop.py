#!/usr/bin/env python3
"""Audit matched P0/P4/P5 real L89 dev predictions and report paired F1.

This collector refuses test/sealed paths, verifies encoder-cache provenance,
head initialization equality, prediction identity, and P4/P5 pretraining
equality before computing dev-only row and canonical-event-balanced metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import l89_ragged_cls_experiment as cache_runner


SCRIPT_VERSION = "audit-rctp-l89-real-cls-loop-v1"
ARMS = ("p0", "p4", "p5")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def canonical_identifier(series: pd.Series, name: str) -> pd.Series:
    values = series.astype("string").fillna("").str.strip()
    if values.eq("").any():
        raise ValueError(f"Blank {name} values")
    return values.astype(str)


def load_downstream_arm(
    arm: str,
    directory: Path,
    expected_weights: Path,
) -> dict[str, Any]:
    cache_runner.assert_not_sealed_path(directory, purpose=f"{arm} downstream")
    summary = read_json(directory / "summary.json")
    config = read_json(directory / "run_config.json")
    best = summary.get("best_by_arm", {}).get("role_only")
    if not isinstance(best, dict):
        raise ValueError(f"{arm} lacks a completed role_only result")
    epoch = int(best["epoch"])
    if not 1 <= epoch <= 3:
        raise ValueError(f"{arm} selected epoch {epoch}, outside bounded 1..3")
    prediction_path = (
        directory / "role_only" / "validation_best_ap_predictions.csv"
    )
    frame = pd.read_csv(prediction_path)
    required = {"id", "plume_id", "event_id", "label", "probability"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{arm} prediction CSV missing {missing}")
    frame = frame.copy()
    for column in ("id", "plume_id", "event_id"):
        frame[column] = canonical_identifier(frame[column], f"{arm}.{column}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{arm} has duplicate prediction IDs")
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError(f"{arm} labels are not binary")
    probability = pd.to_numeric(frame["probability"], errors="raise").astype(float)
    if not np.isfinite(probability).all() or not probability.between(0, 1).all():
        raise ValueError(f"{arm} probabilities are invalid")
    frame["label"] = labels
    frame["probability"] = probability

    expected_sha = cache_runner.sha256_file(expected_weights)
    observed_sha = summary.get("cache_audit", {}).get("weights_sha256")
    if observed_sha != expected_sha:
        raise ValueError(
            f"{arm} cache weights SHA {observed_sha} != expected {expected_sha}"
        )
    return {
        "arm": arm,
        "directory": str(directory),
        "summary": summary,
        "run_config": config,
        "best_epoch": epoch,
        "prediction_path": str(prediction_path),
        "predictions": frame.sort_values("id").reset_index(drop=True),
        "expected_weights": str(expected_weights),
        "expected_weights_sha256": expected_sha,
        "cache_weights_sha256": observed_sha,
        "initial_state_sha256": summary["initial_state_sha256"],
        "parameter_signature": summary["parameter_signature"],
    }


def audit_prediction_identity(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    reference = arms["p0"]["predictions"]
    identity_columns = ["id", "plume_id", "event_id", "label"]
    for arm in ("p4", "p5"):
        candidate = arms[arm]["predictions"]
        if len(candidate) != len(reference):
            raise ValueError(
                f"{arm} prediction rows {len(candidate)} != P0 {len(reference)}"
            )
        for column in identity_columns:
            if not candidate[column].equals(reference[column]):
                mismatch = np.flatnonzero(
                    candidate[column].to_numpy()
                    != reference[column].to_numpy()
                )[:10]
                raise ValueError(
                    f"{arm}/{column} differs from P0 at rows {mismatch.tolist()}"
                )
    return {
        "rows": len(reference),
        "events": int(reference["event_id"].nunique()),
        "plumes": int(reference["plume_id"].nunique()),
        "identity_columns_equal": identity_columns,
        "id_unique_each_arm": True,
    }


def audit_head_matching(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    initialization = {arms[arm]["initial_state_sha256"] for arm in ARMS}
    signatures = {
        json.dumps(arms[arm]["parameter_signature"], sort_keys=True)
        for arm in ARMS
    }
    if len(initialization) != 1:
        raise ValueError("P0/P4/P5 downstream head initial states differ")
    if len(signatures) != 1:
        raise ValueError("P0/P4/P5 downstream parameter signatures differ")
    comparable_keys = (
        "seed",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "model_dim",
        "num_heads",
        "depth",
        "mlp_ratio",
        "dropout",
    )
    values = {
        key: [arms[arm]["run_config"].get(key) for arm in ARMS]
        for key in comparable_keys
    }
    mismatched = {
        key: value for key, value in values.items() if len(set(value)) != 1
    }
    if mismatched:
        raise ValueError(f"Downstream configs are not matched: {mismatched}")
    return {
        "same_initial_state_sha256": next(iter(initialization)),
        "same_parameter_signature": True,
        "matched_config": {key: values[key][0] for key in comparable_keys},
    }


def audit_pretraining_matching(p4_dir: Path, p5_dir: Path) -> dict[str, Any]:
    for path, purpose in (
        (p4_dir, "P4 pretraining"),
        (p5_dir, "P5 pretraining"),
    ):
        cache_runner.assert_not_sealed_path(path, purpose=purpose)
    p4_config = read_json(p4_dir / "run_config.json")
    p5_config = read_json(p5_dir / "run_config.json")
    p4_history = json.loads(
        (p4_dir / "metrics_history.json").read_text(encoding="utf-8")
    )
    p5_history = json.loads(
        (p5_dir / "metrics_history.json").read_text(encoding="utf-8")
    )
    equality_keys = (
        "base_weights_sha256",
        "trainability",
        "trainable_initial_state_sha256",
        "probe_config",
        "probe_initial_state_sha256",
        "parameter_counts",
        "clean_anchor_contract",
        "train_plan_rows_per_epoch",
        "dev_plan_rows",
        "dev_plan_sha256",
        "planned_optimizer_steps",
    )
    mismatched = {
        key: {"p4": p4_config.get(key), "p5": p5_config.get(key)}
        for key in equality_keys
        if p4_config.get(key) != p5_config.get(key)
    }
    if mismatched:
        raise ValueError(f"P4/P5 pretraining inputs are not matched: {mismatched}")
    p4_plan_hashes = [item["train_plan_sha256"] for item in p4_history]
    p5_plan_hashes = [item["train_plan_sha256"] for item in p5_history]
    p4_steps = [item["train"]["global_optimizer_steps"] for item in p4_history]
    p5_steps = [item["train"]["global_optimizer_steps"] for item in p5_history]
    if p4_plan_hashes != p5_plan_hashes:
        raise ValueError("P4/P5 deterministic train plans differ")
    if p4_steps != p5_steps:
        raise ValueError("P4/P5 optimizer-step counts differ")
    if p4_config.get("objective_variant_order") == p5_config.get(
        "objective_variant_order"
    ):
        raise ValueError("P4/P5 objective response order unexpectedly identical")
    return {
        "matched_keys": list(equality_keys),
        "train_plan_sha256_by_epoch": p4_plan_hashes,
        "global_optimizer_steps_by_epoch": p4_steps,
        "only_expected_difference": (
            "objective response/variant order and resulting learned weights"
        ),
        "cached_cls_used_as_temporal_context_or_probe_input": False,
        "clean_anchor_contract": p4_config.get("clean_anchor_contract"),
    }


def f1_from_confusion(
    tp: float, fp: float, fn: float, tn: float
) -> tuple[float, float]:
    positive_denominator = 2 * tp + fp + fn
    negative_denominator = 2 * tn + fp + fn
    positive = 0.0 if positive_denominator <= 0 else 2 * tp / positive_denominator
    negative = 0.0 if negative_denominator <= 0 else 2 * tn / negative_denominator
    return float(positive), float((positive + negative) / 2.0)


def event_confusion_table(
    frame: pd.DataFrame, threshold: float
) -> tuple[list[str], np.ndarray]:
    events: list[str] = []
    rows: list[list[float]] = []
    for event, subset in frame.groupby("event_id", sort=True):
        target = subset["label"].to_numpy(dtype=np.int64)
        prediction = (
            subset["probability"].to_numpy(dtype=float) >= threshold
        ).astype(np.int64)
        weight = np.full(len(subset), 1.0 / len(subset), dtype=np.float64)
        tp = float(weight[(target == 1) & (prediction == 1)].sum())
        fp = float(weight[(target == 0) & (prediction == 1)].sum())
        fn = float(weight[(target == 1) & (prediction == 0)].sum())
        tn = float(weight[(target == 0) & (prediction == 0)].sum())
        events.append(str(event))
        rows.append([tp, fp, fn, tn])
    return events, np.asarray(rows, dtype=np.float64)


def compute_point_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame["label"].to_numpy(dtype=np.int64)
    probability = frame["probability"].to_numpy(dtype=np.float64)
    event_weights = cache_runner.event_balanced_row_weights(frame["event_id"])
    threshold, selected_positive_f1 = (
        cache_runner.best_weighted_positive_f1_threshold(
            target, probability, event_weights
        )
    )
    output: dict[str, Any] = {
        "rows": len(frame),
        "events": int(frame["event_id"].nunique()),
        "row_ap": float(average_precision_score(target, probability)),
        "row_auc": float(roc_auc_score(target, probability)),
        "event_balanced_ap": float(
            average_precision_score(
                target, probability, sample_weight=event_weights
            )
        ),
        "event_balanced_auc": float(
            roc_auc_score(target, probability, sample_weight=event_weights)
        ),
        "event_balanced_selected_threshold": threshold,
        "event_balanced_selected_positive_f1": selected_positive_f1,
    }
    for name, value in (("fixed_0p5", 0.5), ("dev_selected", threshold)):
        prediction = (probability >= value).astype(np.int64)
        output[f"row_positive_f1_{name}"] = float(
            f1_score(target, prediction, zero_division=0)
        )
        output[f"row_macro_f1_{name}"] = float(
            f1_score(
                target,
                prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )
        output[f"event_balanced_positive_f1_{name}"] = float(
            f1_score(
                target,
                prediction,
                sample_weight=event_weights,
                zero_division=0,
            )
        )
        output[f"event_balanced_macro_f1_{name}"] = float(
            f1_score(
                target,
                prediction,
                labels=[0, 1],
                average="macro",
                sample_weight=event_weights,
                zero_division=0,
            )
        )
    return output


def bootstrap_event_f1(
    arms: Mapping[str, Mapping[str, Any]],
    points: Mapping[str, Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = {
        arm: {
            "fixed_0p5": 0.5,
            "dev_selected": points[arm]["event_balanced_selected_threshold"],
        }
        for arm in ARMS
    }
    tables: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in ARMS}
    canonical_events: Optional[list[str]] = None
    for arm in ARMS:
        frame = arms[arm]["predictions"]
        for threshold_name, threshold in thresholds[arm].items():
            events, table = event_confusion_table(frame, threshold)
            if canonical_events is None:
                canonical_events = events
            elif events != canonical_events:
                raise ValueError("Event order differs between arms")
            tables[arm][threshold_name] = table
    assert canonical_events is not None
    rng = np.random.default_rng(seed)
    distributions: dict[str, dict[str, dict[str, list[float]]]] = {
        arm: {
            threshold_name: {"positive_f1": [], "macro_f1": []}
            for threshold_name in ("fixed_0p5", "dev_selected")
        }
        for arm in ARMS
    }
    for _ in range(int(replicates)):
        sample = rng.integers(0, len(canonical_events), size=len(canonical_events))
        for arm in ARMS:
            for threshold_name in ("fixed_0p5", "dev_selected"):
                tp, fp, fn, tn = tables[arm][threshold_name][sample].sum(axis=0)
                positive, macro = f1_from_confusion(tp, fp, fn, tn)
                distributions[arm][threshold_name]["positive_f1"].append(positive)
                distributions[arm][threshold_name]["macro_f1"].append(macro)

    intervals: dict[str, Any] = {}
    for arm in ARMS:
        intervals[arm] = {}
        for threshold_name in ("fixed_0p5", "dev_selected"):
            intervals[arm][threshold_name] = {}
            for metric in ("positive_f1", "macro_f1"):
                values = np.asarray(
                    distributions[arm][threshold_name][metric], dtype=float
                )
                point_key = f"event_balanced_{metric}_{threshold_name}"
                intervals[arm][threshold_name][metric] = {
                    "point": points[arm][point_key],
                    "ci_95_low": float(np.percentile(values, 2.5)),
                    "ci_95_high": float(np.percentile(values, 97.5)),
                }
    deltas: dict[str, Any] = {}
    for comparison, left, right in (
        ("p5_minus_p0", "p5", "p0"),
        ("p5_minus_p4", "p5", "p4"),
        ("p4_minus_p0", "p4", "p0"),
    ):
        deltas[comparison] = {}
        for threshold_name in ("fixed_0p5", "dev_selected"):
            deltas[comparison][threshold_name] = {}
            for metric in ("positive_f1", "macro_f1"):
                values = np.asarray(
                    distributions[left][threshold_name][metric]
                ) - np.asarray(distributions[right][threshold_name][metric])
                point_key = f"event_balanced_{metric}_{threshold_name}"
                deltas[comparison][threshold_name][metric] = {
                    "point": points[left][point_key] - points[right][point_key],
                    "ci_95_low": float(np.percentile(values, 2.5)),
                    "ci_95_high": float(np.percentile(values, 97.5)),
                }
    return intervals, deltas


def render_markdown(result: Mapping[str, Any]) -> str:
    anchor_contract = result["pretraining_matching"].get(
        "clean_anchor_contract", {}
    )
    anchor_description = (
        f"enabled, detached clean target only "
        f"(weight={anchor_contract.get('weight')}, "
        f"metric={anchor_contract.get('metric')})"
        if anchor_contract.get("enabled")
        else "disabled (weight=0)"
    )
    rows = []
    metrics = result["point_metrics"]
    intervals = result["event_cluster_bootstrap_ci"]
    for arm in ARMS:
        selected = intervals[arm]["dev_selected"]["positive_f1"]
        macro = intervals[arm]["dev_selected"]["macro_f1"]
        rows.append(
            f"| {arm.upper()} | {metrics[arm]['best_epoch']} | "
            f"{metrics[arm]['event_balanced_ap']:.4f} | "
            f"{selected['point']:.4f} [{selected['ci_95_low']:.4f}, "
            f"{selected['ci_95_high']:.4f}] | "
            f"{macro['point']:.4f} [{macro['ci_95_low']:.4f}, "
            f"{macro['ci_95_high']:.4f}] | "
            f"{metrics[arm]['event_balanced_selected_threshold']:.4f} |"
        )
    delta_rows = []
    for comparison in ("p5_minus_p0", "p5_minus_p4", "p4_minus_p0"):
        positive = result["paired_deltas"][comparison]["dev_selected"][
            "positive_f1"
        ]
        macro = result["paired_deltas"][comparison]["dev_selected"]["macro_f1"]
        delta_rows.append(
            f"| {comparison} | {positive['point']:+.4f} "
            f"[{positive['ci_95_low']:+.4f}, {positive['ci_95_high']:+.4f}] | "
            f"{macro['point']:+.4f} "
            f"[{macro['ci_95_low']:+.4f}, {macro['ci_95_high']:+.4f}] |"
        )
    return f"""# RCTP → real L89 classification: matched dev audit

This is a train/dev-only engineering screen. It is not a final test result.
P0, P4, and P5 use identical real L89 rows and byte-identical downstream-head
initialization. P4/P5 use matched online-context continue-pretraining; their
only intended difference is correct versus scrambled spectral response.

| Arm | Best head epoch | Event-balanced AP | Event-balanced positive F1 (95% CI) | Event-balanced macro F1 (95% CI) | Dev threshold |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

| Paired delta | Positive F1 (95% CI) | Macro F1 (95% CI) |
|---|---:|---:|
{chr(10).join(delta_rows)}

Intervals are canonical-event cluster percentiles with each arm's dev-selected
threshold held fixed. They quantify this dev split only. Threshold and checkpoint
selection both used dev, so the values are optimistic and require a separately
authorized final evaluation after freezing the protocol.

Leakage/provenance checks:

- prediction rows/events/plumes: {result['prediction_identity']['rows']:,} /
  {result['prediction_identity']['events']:,} /
  {result['prediction_identity']['plumes']:,};
- P0/P4/P5 identities and labels are exactly equal;
- cache weight SHA matches the declared PTH for every arm;
- downstream initialization, shapes, optimizer settings, batches, and epoch cap match;
- P4/P5 train plans, update counts, base weights, and trainable parameter set match;
- cached CLS used as temporal context or probe input: **no**;
- optional cached clean target: **{anchor_description}**;
- test/sealed artifacts read: **none**.
"""


def run(args: argparse.Namespace) -> None:
    directories = {
        "p0": Path(args.p0_dir).expanduser().resolve(),
        "p4": Path(args.p4_dir).expanduser().resolve(),
        "p5": Path(args.p5_dir).expanduser().resolve(),
    }
    weights = {
        "p0": Path(args.p0_weights).expanduser().resolve(),
        "p4": Path(args.p4_weights).expanduser().resolve(),
        "p5": Path(args.p5_weights).expanduser().resolve(),
    }
    arms = {
        arm: load_downstream_arm(arm, directories[arm], weights[arm])
        for arm in ARMS
    }
    identity = audit_prediction_identity(arms)
    head_matching = audit_head_matching(arms)
    pretrain_matching = audit_pretraining_matching(
        Path(args.p4_pretrain_dir).expanduser().resolve(),
        Path(args.p5_pretrain_dir).expanduser().resolve(),
    )
    point_metrics = {
        arm: {
            **compute_point_metrics(arms[arm]["predictions"]),
            "best_epoch": arms[arm]["best_epoch"],
        }
        for arm in ARMS
    }
    intervals, deltas = bootstrap_event_f1(
        arms,
        point_metrics,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    result = {
        "script_version": SCRIPT_VERSION,
        "scope": "train/dev-only matched real L89 classification",
        "test_or_sealed_read": False,
        "prediction_identity": identity,
        "head_matching": head_matching,
        "pretraining_matching": pretrain_matching,
        "weights": {
            arm: {
                "path": arms[arm]["expected_weights"],
                "sha256": arms[arm]["expected_weights_sha256"],
            }
            for arm in ARMS
        },
        "point_metrics": point_metrics,
        "bootstrap": {
            "unit": "canonical event_id",
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "thresholds_refit_per_replicate": False,
        },
        "event_cluster_bootstrap_ci": intervals,
        "paired_deltas": deltas,
        "interpretation_guardrail": (
            "Dev-selected checkpoint and threshold; engineering screen only, "
            "not an unbiased final result."
        ),
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    cache_runner.assert_not_sealed_path(output_json, purpose="comparison JSON")
    cache_runner.assert_not_sealed_path(output_md, purpose="comparison Markdown")
    cache_runner.atomic_json_write(output_json, result)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_md.with_suffix(output_md.suffix + ".tmp")
    temporary.write_text(render_markdown(result), encoding="utf-8")
    temporary.replace(output_md)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit matched P0/P4/P5 real L89 dev classification."
    )
    parser.add_argument("--p0-dir", required=True)
    parser.add_argument("--p4-dir", required=True)
    parser.add_argument("--p5-dir", required=True)
    parser.add_argument("--p4-pretrain-dir", required=True)
    parser.add_argument("--p5-pretrain-dir", required=True)
    parser.add_argument("--p0-weights", required=True)
    parser.add_argument("--p4-weights", required=True)
    parser.add_argument("--p5-weights", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.set_defaults(handler=run)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
