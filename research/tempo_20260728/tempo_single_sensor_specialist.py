#!/usr/bin/env python3
"""Development-only single-current-sensor R4 specialist.

The R4 head is trained only on rows for which exactly one sensor is available
at t0.  At development inference it is used only for that same branch; rows
with two or more current sensors retain the frozen promoted P0 score and
decision.  No test/sealed input is accepted.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.tempo_20260728.availability_gate_control import (  # noqa: E402
    fixed_event_audit,
    fixed_metrics,
    paired_event_bootstrap,
)
from research.tempo_20260728.tempo_legacy360_global import (  # noqa: E402
    SCRIPT_VERSION as HEAD_SCRIPT_VERSION,
    TempoGlobalHead,
    _features_for_mode,
    active_parameter_compute_contract,
    atomic_json,
    atomic_torch,
    build_matched_initial_state,
    fixed_epoch_batches,
    guard_development_path,
    load_feature_cache,
    model_parameter_signature,
    predict_tempo,
    probability_metrics,
    promoted_outputs,
    set_seed,
    sha256_file,
    shuffle_history,
    state_dict_sha256,
    validate_development_caches,
)


SCRIPT_VERSION = "tempo-single-sensor-r4-specialist-v1"
DEFAULT_FORMAL_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5"
)
DEFAULT_LEGACY_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1"
)
DEFAULT_OUTPUT = DEFAULT_LEGACY_ROOT / "single_sensor_specialist_v10"
DEFAULT_V7_PREDICTIONS = (
    DEFAULT_LEGACY_ROOT
    / "final_r4_4seed_v5/final_ensemble_predictions.csv"
)
DEFAULT_V7_AGGREGATE = (
    DEFAULT_LEGACY_ROOT / "final_r4_4seed_v5/final_aggregate.json"
)


def current_sensor_counts(valid_mask: torch.Tensor) -> np.ndarray:
    if valid_mask.ndim != 3:
        raise ValueError("valid_mask must have shape [N,S,R].")
    counts = valid_mask[:, :, 0].sum(dim=1).cpu().numpy().astype(np.int64)
    if not np.all(counts >= 1):
        raise RuntimeError("Every row must have a valid current sensor.")
    return counts


def _f1_from_confusion(
    tp: int, fp: int, fn: int, tn: int
) -> tuple[float, float]:
    positive = (
        2.0 * tp / (2.0 * tp + fp + fn)
        if (2 * tp + fp + fn)
        else 0.0
    )
    negative = (
        2.0 * tn / (2.0 * tn + fp + fn)
        if (2 * tn + fp + fn)
        else 0.0
    )
    return float(positive), float((positive + negative) / 2.0)


def best_branch_threshold(
    *,
    labels: np.ndarray,
    branch_scores: np.ndarray,
    branch_mask: np.ndarray,
    fixed_other_predictions: np.ndarray,
) -> float:
    """Optimize only the branch threshold while other decisions stay fixed."""

    target = np.asarray(labels, dtype=np.int64)
    score = np.asarray(branch_scores, dtype=np.float64)
    use_branch = np.asarray(branch_mask, dtype=bool)
    fixed = np.asarray(fixed_other_predictions, dtype=bool)
    if not (
        target.shape == score.shape == use_branch.shape == fixed.shape
    ):
        raise ValueError("Branch-threshold arrays must have equal shape.")
    positions = np.flatnonzero(use_branch)
    if not len(positions):
        raise ValueError("Branch has no rows.")
    other = ~use_branch
    tp = int(np.sum((target == 1) & fixed & other))
    fp = int(np.sum((target == 0) & fixed & other))
    fn = int(np.sum((target == 1) & ~fixed & other))
    tn = int(np.sum((target == 0) & ~fixed & other))
    branch_target = target[positions]
    branch_score = score[positions]
    fn += int(np.sum(branch_target == 1))
    tn += int(np.sum(branch_target == 0))
    maximum = float(np.max(branch_score))
    empty_threshold = float(np.nextafter(maximum, math.inf))
    binary, macro = _f1_from_confusion(tp, fp, fn, tn)
    best_key = (binary, macro, empty_threshold)
    best_threshold = empty_threshold
    order = np.argsort(-branch_score, kind="mergesort")
    sorted_score = branch_score[order]
    sorted_target = branch_target[order]
    start = 0
    while start < len(order):
        stop = start + 1
        while (
            stop < len(order)
            and sorted_score[stop] == sorted_score[start]
        ):
            stop += 1
        group = sorted_target[start:stop]
        positives = int(group.sum())
        negatives = int(len(group) - positives)
        tp += positives
        fn -= positives
        fp += negatives
        tn -= negatives
        binary, macro = _f1_from_confusion(tp, fp, fn, tn)
        threshold = float(sorted_score[start])
        key = (binary, macro, threshold)
        if key > best_key:
            best_key = key
            best_threshold = threshold
        start = stop
    return best_threshold


def evaluate_gate(
    *,
    labels: np.ndarray,
    specialist_scores: np.ndarray,
    p0_scores: np.ndarray,
    sensor_counts: np.ndarray,
    p0_threshold: float,
    specialist_threshold: float | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
    target = np.asarray(labels, dtype=np.int64)
    specialist = np.asarray(specialist_scores, dtype=np.float64)
    p0 = np.asarray(p0_scores, dtype=np.float64)
    counts = np.asarray(sensor_counts, dtype=np.int64)
    use_specialist = counts == 1
    p0_prediction = p0 >= float(p0_threshold)
    threshold = (
        best_branch_threshold(
            labels=target,
            branch_scores=specialist,
            branch_mask=use_specialist,
            fixed_other_predictions=p0_prediction,
        )
        if specialist_threshold is None
        else float(specialist_threshold)
    )
    specialist_prediction = specialist >= threshold
    score = np.where(use_specialist, specialist, p0)
    prediction = np.where(
        use_specialist, specialist_prediction, p0_prediction
    )
    return (
        fixed_metrics(target, score, prediction),
        score,
        prediction,
        threshold,
    )


def _selection_key(record: Mapping[str, Any]) -> tuple[float, float, float]:
    metrics = record["gate_metrics"]
    return (
        float(metrics["binary_f1"]),
        float(metrics["ap"]),
        float(metrics["auc"]),
    )


def _load_v7(
    *,
    predictions_path: Path,
    aggregate_path: Path,
    expected_ids: list[str],
) -> tuple[
    dict[str, Any],
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, float],
]:
    frame = pd.read_csv(predictions_path)
    if frame["id"].astype(str).tolist() != expected_ids:
        raise RuntimeError("Frozen v7 predictions do not match dev row order.")
    with aggregate_path.open("r", encoding="utf-8") as stream:
        aggregate = json.load(stream)
    if bool(aggregate["test_or_sealed_read"]):
        raise RuntimeError("Frozen v7 aggregate reports test/sealed read.")
    labels = frame["label"].to_numpy(dtype=np.int64)
    counts = (
        frame["availability_signature"]
        .astype(str)
        .map(lambda value: len(value.split("+")))
        .to_numpy(dtype=np.int64)
    )
    p0_score = frame["p0_probability"].to_numpy(dtype=np.float64)
    r4_score = frame["ensemble_probability"].to_numpy(dtype=np.float64)
    p0_threshold = float(
        aggregate["p0"]["metrics"]["best_binary_f1_threshold"]
    )
    r4_threshold = float(
        aggregate["logit_ensemble"]["metrics"][
            "best_binary_f1_threshold"
        ]
    )
    p0_prediction = p0_score >= p0_threshold
    r4_prediction = r4_score >= r4_threshold
    use_r4 = counts == 1
    gate_score = np.where(use_r4, r4_score, p0_score)
    gate_prediction = np.where(use_r4, r4_prediction, p0_prediction)
    models = {
        "p0": (p0_score, p0_prediction),
        "frozen_v7_gate": (gate_score, gate_prediction),
    }
    result = {
        name: fixed_metrics(labels, score, prediction)
        for name, (score, prediction) in models.items()
    }
    return (
        result,
        models,
        {"p0": p0_threshold, "r4": r4_threshold},
    )


def command_run(args: argparse.Namespace) -> None:
    train_path = Path(args.train_cache).expanduser().absolute()
    dev_path = Path(args.dev_cache).expanduser().absolute()
    promoted_path = Path(args.promoted_checkpoint).expanduser().absolute()
    v7_predictions_path = Path(args.v7_predictions).expanduser().absolute()
    v7_aggregate_path = Path(args.v7_aggregate).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    for path, role in (
        (train_path, "training cache"),
        (dev_path, "development cache"),
        (promoted_path, "promoted checkpoint"),
        (v7_predictions_path, "frozen v7 development predictions"),
        (v7_aggregate_path, "frozen v7 development aggregate"),
        (output_dir, "development output"),
    ):
        guard_development_path(path, role)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    atomic_json(
        status_path,
        {
            "status": "loading",
            "test_or_sealed_read": False,
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    )
    torch.set_num_threads(int(args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    train_cache = load_feature_cache(train_path, ("train_core", "train"))
    dev_cache = load_feature_cache(dev_path, ("dev", "evaluation"))
    validate_development_caches(train_cache, dev_cache)
    promoted_checkpoint = torch.load(
        promoted_path, map_location="cpu", weights_only=False
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(int(args.seed))
    train_base = promoted_outputs(
        promoted_checkpoint,
        train_cache,
        batch_size=int(args.eval_batch_size),
        device=device,
    )
    dev_base = promoted_outputs(
        promoted_checkpoint,
        dev_cache,
        batch_size=int(args.eval_batch_size),
        device=device,
    )
    p0_dev_probability = torch.sigmoid(dev_base[0]).numpy()
    train_counts = current_sensor_counts(train_cache["valid_mask"])
    dev_counts = current_sensor_counts(dev_cache["valid_mask"])
    train_positions = torch.from_numpy(
        np.flatnonzero(train_counts == 1)
    ).long()
    if not len(train_positions):
        raise RuntimeError("No exactly-one-current-sensor training rows.")
    train_features = _features_for_mode(train_cache, "universal")
    dev_features = _features_for_mode(dev_cache, "universal")
    train_valid = train_cache["valid_mask"]
    dev_valid = dev_cache["valid_mask"]
    train_labels = train_cache["labels"].float()
    dev_labels = dev_cache["labels"].numpy().astype(np.int64)
    initial_state, signature, initial_sha = build_matched_initial_state(
        int(train_features.shape[-1]),
        model_dim=int(args.model_dim),
        residual_cap=float(args.residual_cap),
        seed=int(args.seed),
    )
    model = TempoGlobalHead(
        int(train_features.shape[-1]),
        model_dim=int(args.model_dim),
        residual_cap=float(args.residual_cap),
    )
    model.load_state_dict(initial_state, strict=True)
    if model_parameter_signature(model) != signature:
        raise RuntimeError("Matched parameter signature changed.")
    if state_dict_sha256(model.state_dict()) != initial_sha:
        raise RuntimeError("Matched initialization changed.")
    active_contract = active_parameter_compute_contract(model, arm="r4")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    v7_metrics, v7_models, v7_thresholds = _load_v7(
        predictions_path=v7_predictions_path,
        aggregate_path=v7_aggregate_path,
        expected_ids=[str(value) for value in dev_cache["ids"]],
    )
    # Both the candidate and its frozen-v7 reference must use the identical
    # aggregate-locked P0 scores and threshold on the unchanged multi-sensor
    # branch. This also removes device-level roundoff at threshold ties.
    p0_threshold = float(v7_thresholds["p0"])
    gate_p0_probability, p0_prediction = v7_models["p0"]
    p0_metrics = v7_metrics["p0"]
    baseline_record = {
        "epoch": 0,
        "train": None,
        "gate_metrics": p0_metrics,
        "specialist_threshold": p0_threshold,
        "exact_p0": True,
    }
    history: list[dict[str, Any]] = [baseline_record]
    best_record = copy.deepcopy(baseline_record)
    best_state = copy.deepcopy(initial_state)
    best_specialist_probability = p0_dev_probability.copy()
    best_gate_score = gate_p0_probability.copy()
    best_gate_prediction = p0_prediction.copy()
    stale_epochs = 0
    started = time.monotonic()
    atomic_json(
        status_path,
        {
            "status": "training",
            "train_source_rows": int(len(train_counts)),
            "train_specialist_rows": int(len(train_positions)),
            "dev_rows": int(len(dev_counts)),
            "device": str(device),
            "test_or_sealed_read": False,
        },
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = fused_sum = sensor_sum = l2_sum = 0.0
        seen = 0
        batches = fixed_epoch_batches(
            len(train_positions),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            epoch=epoch,
            shuffle=True,
        )
        for relative in batches:
            source = train_positions[relative]
            target = train_labels[source].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                train_features[source].to(device=device, dtype=torch.float32),
                train_valid[source].to(device),
                train_base[0][source].to(device),
                train_base[1][source].to(device),
                arm="r4",
            )
            fused_loss = F.binary_cross_entropy_with_logits(
                output["fused_logits"], target
            )
            expanded = target[:, None].expand_as(output["sensor_logits"])
            sensor_raw = F.binary_cross_entropy_with_logits(
                output["sensor_logits"], expanded, reduction="none"
            )
            sensor_mask = output["sensor_valid"]
            sensor_loss = (
                (sensor_raw * sensor_mask).sum()
                / sensor_mask.sum().clamp_min(1)
            )
            l2 = output["residual"].square().mean()
            loss = (
                fused_loss
                + float(args.sensor_aux_weight) * sensor_loss
                + float(args.residual_l2) * l2
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(len(source))
            total += float(loss.detach()) * count
            fused_sum += float(fused_loss.detach()) * count
            sensor_sum += float(sensor_loss.detach()) * count
            l2_sum += float(l2.detach()) * count
            seen += count
        specialist_probability, residual = predict_tempo(
            model,
            dev_features,
            dev_valid,
            dev_base[0],
            dev_base[1],
            arm="r4",
            batch_size=int(args.eval_batch_size),
            device=device,
        )
        gate_metrics, gate_score, gate_prediction, threshold = evaluate_gate(
            labels=dev_labels,
            specialist_scores=specialist_probability,
            p0_scores=gate_p0_probability,
            sensor_counts=dev_counts,
            p0_threshold=p0_threshold,
            specialist_threshold=None,
        )
        raw_metrics = probability_metrics(
            dev_cache["labels"],
            specialist_probability,
            dev_cache["availability_signatures"],
        )
        frozen_threshold_metrics, _, _, _ = evaluate_gate(
            labels=dev_labels,
            specialist_scores=specialist_probability,
            p0_scores=gate_p0_probability,
            sensor_counts=dev_counts,
            p0_threshold=p0_threshold,
            specialist_threshold=float(v7_thresholds["r4"]),
        )
        record = {
            "epoch": int(epoch),
            "train": {
                "loss": total / seen,
                "fused_bce": fused_sum / seen,
                "sensor_bce": sensor_sum / seen,
                "residual_l2": l2_sum / seen,
                "rows": int(seen),
                "steps": int(len(batches)),
            },
            "gate_metrics": gate_metrics,
            "specialist_raw_metrics": raw_metrics,
            "gate_metrics_at_frozen_v7_r4_threshold": (
                frozen_threshold_metrics
            ),
            "specialist_threshold": float(threshold),
            "residual_abs_mean": float(np.abs(residual).mean()),
            "elapsed_seconds": float(time.monotonic() - started),
            "exact_p0": False,
        }
        history.append(record)
        improved = _selection_key(record) > _selection_key(best_record)
        if improved:
            best_record = copy.deepcopy(record)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_specialist_probability = specialist_probability.copy()
            best_gate_score = gate_score.copy()
            best_gate_prediction = gate_prediction.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
        atomic_json(output_dir / "metrics_history.json", history)
        print(
            f"[SPECIALIST] epoch={epoch}/{args.epochs} "
            f"rows={seen} loss={total / seen:.6f} "
            f"gate_F1={gate_metrics['binary_f1']:.6f} "
            f"macro={gate_metrics['macro_f1']:.6f} "
            f"AP={gate_metrics['ap']:.6f} AUC={gate_metrics['auc']:.6f} "
            f"thr={threshold:.6f}",
            flush=True,
        )
        if (
            int(args.early_stop_patience) > 0
            and stale_epochs >= int(args.early_stop_patience)
        ):
            print(
                f"[SPECIALIST] early_stop epoch={epoch} "
                f"patience={args.early_stop_patience}",
                flush=True,
            )
            break
    best_epoch = int(best_record["epoch"])
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    if best_epoch > 0:
        reloaded, _ = predict_tempo(
            model,
            dev_features,
            dev_valid,
            dev_base[0],
            dev_base[1],
            arm="r4",
            batch_size=int(args.eval_batch_size),
            device=device,
        )
        if not np.allclose(
            reloaded, best_specialist_probability, atol=1e-7
        ):
            raise RuntimeError("Reloaded specialist prediction changed.")
    event_ids = np.asarray(
        [str(value) for value in dev_cache["event_ids"]], dtype=str
    )
    candidate_model = (
        best_gate_score.astype(np.float64),
        best_gate_prediction.astype(bool),
    )
    models = {
        **v7_models,
        "single_sensor_specialist_gate": candidate_model,
    }
    bootstrap = paired_event_bootstrap(
        labels=dev_labels,
        event_ids=event_ids,
        models=models,
        candidate_name="single_sensor_specialist_gate",
        references=["p0", "frozen_v7_gate"],
        repeats=int(args.bootstrap_repeats),
        seed=int(args.bootstrap_seed),
    )
    candidate_metrics = fixed_metrics(
        dev_labels, best_gate_score, best_gate_prediction
    )
    v7 = v7_metrics["frozen_v7_gate"]
    promotion = bool(
        candidate_metrics["binary_f1"]
        >= float(v7["binary_f1"]) + float(args.minimum_f1_gain)
        and candidate_metrics["ap"] >= float(v7["ap"])
        and candidate_metrics["auc"] >= float(v7["auc"])
        and best_epoch > 0
    )
    specialist_threshold = float(best_record["specialist_threshold"])
    atomic_torch(
        output_dir / "checkpoint_best.pth",
        {
            "schema_version": "tempo-single-sensor-r4-specialist-v1",
            "script_version": SCRIPT_VERSION,
            "head_script_version": HEAD_SCRIPT_VERSION,
            "arm": "r4",
            "seed": int(args.seed),
            "epoch": best_epoch,
            "model_config": {
                "feature_dim": int(train_features.shape[-1]),
                "num_sensors": 4,
                "model_dim": int(args.model_dim),
                "residual_cap": float(args.residual_cap),
            },
            "model": best_state,
            "initial_state_sha256": initial_sha,
            "parameter_signature": signature,
            "active_parameter_compute_contract": active_contract,
            "training_filter": "exactly one valid current sensor",
            "locked_dev_specialist_threshold": specialist_threshold,
            "p0_threshold": p0_threshold,
            "test_or_sealed_read": False,
        },
    )
    shuffled_features = shuffle_history(
        dev_features, dev_valid, seed=int(args.shuffle_seed)
    )
    if best_epoch > 0:
        shuffled_specialist, _ = predict_tempo(
            model,
            shuffled_features,
            dev_valid,
            dev_base[0],
            dev_base[1],
            arm="r4",
            batch_size=int(args.eval_batch_size),
            device=device,
        )
    else:
        shuffled_specialist = p0_dev_probability.copy()
    (
        shuffled_metrics,
        _,
        shuffled_prediction,
        _,
    ) = evaluate_gate(
        labels=dev_labels,
        specialist_scores=shuffled_specialist,
        p0_scores=gate_p0_probability,
        sensor_counts=dev_counts,
        p0_threshold=p0_threshold,
        specialist_threshold=specialist_threshold,
    )
    predictions = pd.DataFrame(
        {
            "id": [str(value) for value in dev_cache["ids"]],
            "event_id": event_ids,
            "availability_signature": [
                str(value)
                for value in dev_cache["availability_signatures"]
            ],
            "sensor_count": dev_counts,
            "label": dev_labels,
            "p0_probability": gate_p0_probability,
            "specialist_probability": best_specialist_probability,
            "gate_probability": best_gate_score,
            "gate_prediction": best_gate_prediction.astype(np.int8),
            "history_shuffle_gate_prediction": (
                shuffled_prediction.astype(np.int8)
            ),
        }
    )
    predictions.to_csv(output_dir / "dev_predictions.csv", index=False)
    result = {
        "schema_version": "tempo-single-sensor-r4-specialist-result-v1",
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "best_epoch": best_epoch,
        "best_record": best_record,
        "models": {
            "p0": {
                "metrics": p0_metrics,
                "event_audit": fixed_event_audit(
                    labels=dev_labels,
                    event_ids=event_ids,
                    predictions=p0_prediction,
                    scores=gate_p0_probability,
                ),
            },
            "frozen_v7_gate": {
                "metrics": v7,
                "event_audit": fixed_event_audit(
                    labels=dev_labels,
                    event_ids=event_ids,
                    predictions=v7_models["frozen_v7_gate"][1],
                    scores=v7_models["frozen_v7_gate"][0],
                ),
            },
            "single_sensor_specialist_gate": {
                "metrics": candidate_metrics,
                "event_audit": fixed_event_audit(
                    labels=dev_labels,
                    event_ids=event_ids,
                    predictions=best_gate_prediction,
                    scores=best_gate_score,
                ),
            },
        },
        "history_shuffle_at_locked_specialist_threshold": {
            "metrics": shuffled_metrics,
            "delta_binary_f1": float(
                shuffled_metrics["binary_f1"]
                - candidate_metrics["binary_f1"]
            ),
            "delta_ap": float(
                shuffled_metrics["ap"] - candidate_metrics["ap"]
            ),
            "threshold_refit": False,
        },
        "paired_event_bootstrap": bootstrap,
        "promotion_rule": {
            "minimum_f1_gain_over_v7": float(args.minimum_f1_gain),
            "require_no_ap_regression": True,
            "require_no_auc_regression": True,
            "candidate_passes": promotion,
            "decision": (
                "eligible_for_additional_seeds"
                if promotion
                else "stop_after_seed42"
            ),
        },
        "training": {
            "source_rows": int(len(train_counts)),
            "exactly_one_current_sensor_rows": int(len(train_positions)),
            "positive_rows": int(
                train_cache["labels"][train_positions].sum()
            ),
            "negative_rows": int(
                len(train_positions)
                - train_cache["labels"][train_positions].sum()
            ),
            "epochs_requested": int(args.epochs),
            "epochs_completed": int(len(history) - 1),
            "early_stop_patience": int(args.early_stop_patience),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "sensor_aux_weight": float(args.sensor_aux_weight),
            "residual_l2": float(args.residual_l2),
            "selection": (
                "overall development gate F1; AP then AUC tie breaks; only "
                "the specialist-branch threshold varies"
            ),
        },
        "inputs": {
            "train_cache": str(train_path),
            "train_cache_sha256": sha256_file(train_path),
            "dev_cache": str(dev_path),
            "dev_cache_sha256": sha256_file(dev_path),
            "promoted_checkpoint": str(promoted_path),
            "promoted_checkpoint_sha256": sha256_file(promoted_path),
            "v7_predictions": str(v7_predictions_path),
            "v7_predictions_sha256": sha256_file(v7_predictions_path),
            "v7_aggregate": str(v7_aggregate_path),
            "v7_aggregate_sha256": sha256_file(v7_aggregate_path),
        },
        "test_or_sealed_read": False,
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    atomic_json(output_dir / "result.json", result)
    lines = [
        "# Single-current-sensor R4 specialist",
        "",
        "Development only; no test/sealed artifact was read.",
        "",
        "| Model | F1 | Macro F1 | AP | AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("p0", "frozen_v7_gate", "single_sensor_specialist_gate"):
        value = result["models"][name]["metrics"]
        lines.append(
            f"| {name} | {value['binary_f1']:.6f} | "
            f"{value['macro_f1']:.6f} | {value['ap']:.6f} | "
            f"{value['auc']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Best epoch: `{best_epoch}`. Specialist threshold selected on "
            f"development: `{specialist_threshold:.9f}`.",
            "",
            f"Promotion decision: **{result['promotion_rule']['decision']}**.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    atomic_json(
        status_path,
        {
            "status": "complete",
            "best_epoch": best_epoch,
            "candidate_passes": promotion,
            "test_or_sealed_read": False,
            "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "models": result["models"],
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
        "--train-cache",
        default=str(
            DEFAULT_FORMAL_ROOT
            / "features/train_core_universal_s2hybrid.pt"
        ),
    )
    parser.add_argument(
        "--dev-cache",
        default=str(
            DEFAULT_FORMAL_ROOT / "features/dev_universal_s2hybrid.pt"
        ),
    )
    parser.add_argument(
        "--promoted-checkpoint",
        default=str(
            DEFAULT_FORMAL_ROOT
            / "heads/full_cache_dev_promotion_v1"
            / "promote_compact_scale_aware_d64_lr1e4/checkpoint_best.pth"
        ),
    )
    parser.add_argument(
        "--v7-predictions", default=str(DEFAULT_V7_PREDICTIONS)
    )
    parser.add_argument("--v7-aggregate", default=str(DEFAULT_V7_AGGREGATE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sensor-aux-weight", type=float, default=0.05)
    parser.add_argument("--residual-l2", type=float, default=1e-3)
    parser.add_argument("--model-dim", type=int, default=48)
    parser.add_argument("--residual-cap", type=float, default=1.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--shuffle-seed", type=int, default=20260728)
    parser.add_argument("--minimum-f1-gain", type=float, default=0.001)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    command_run(build_parser().parse_args())


if __name__ == "__main__":
    main()
