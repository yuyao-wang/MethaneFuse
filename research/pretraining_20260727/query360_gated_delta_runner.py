#!/usr/bin/env python3
"""Strict fit/select/evaluate runner for the gated-delta 360 m head.

The ``train`` command intentionally accepts only ``train_core`` and ``dev``
feature caches. Epoch and threshold selection are performed on development
data, while epoch zero is the unchanged historical checkpoint supplied in the
cache. The separate ``evaluate-locked`` command requires an explicit
``--sealed-test`` authorization and evaluates the selected checkpoint exactly
once at the already locked development threshold.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from query360_gated_delta_model import (  # noqa: E402
    GATED_DELTA_ARMS,
    GatedDelta360Head,
)
from query360_model import (  # noqa: E402
    fixed_epoch_batches,
    model_parameter_signature,
    set_deterministic_seed,
    state_dict_sha256,
    transient_query_loss,
)
from query360_two_axis_full_legacy import (  # noqa: E402
    _base_logits_for_mode,
    _features_for_mode,
    _sensor_base_for_mode,
    atomic_json,
    atomic_torch,
    configure_device,
    load_feature_cache,
    sha256_file,
    stratified_locked_metrics,
    stratified_metrics,
)


SCRIPT_VERSION = "query360-gated-delta-locked-eval-v2"
CHECKPOINT_SCHEMA = "query360-gated-delta-head-v1"
LOCK_SCHEMA = "query360-gated-delta-selection-lock-v1"
LOCKED_TEST_SCHEMA = "query360-gated-delta-locked-test-v1"
_FORBIDDEN_SPLIT_TOKEN = re.compile(
    r"(^|[/_.-])(test|sealed)([/_.-]|$)", flags=re.IGNORECASE
)


def guard_development_path(path: Path, *, role: str) -> None:
    """Refuse paths that advertise test/sealed data before reading them."""

    if _FORBIDDEN_SPLIT_TOKEN.search(str(path)):
        raise PermissionError(
            f"Refusing {role} path that looks like test/sealed data: {path}"
        )


def _assert_cache_compatibility(
    train_cache: Mapping[str, Any],
    dev_cache: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    if train_cache["encoder"] != dev_cache["encoder"]:
        raise ValueError("train_core/dev encoder provenance differs")
    train_shape = tuple(train_cache["features"].shape[1:])
    dev_shape = tuple(dev_cache["features"].shape[1:])
    if train_shape != dev_shape:
        raise ValueError(
            f"train_core/dev feature shapes differ: {train_shape} vs {dev_shape}"
        )
    plume_overlap = set(train_cache["plume_ids"]) & set(
        dev_cache["plume_ids"]
    )
    if plume_overlap:
        raise RuntimeError(
            "train_core/dev plume overlap: "
            f"{sorted(str(value) for value in plume_overlap)[:20]}"
        )
    train_events = {
        str(value) for value in train_cache.get("event_ids", [])
    }
    dev_events = {str(value) for value in dev_cache.get("event_ids", [])}
    event_overlap = train_events & dev_events
    if train_events and dev_events and event_overlap:
        raise RuntimeError(
            "train_core/dev canonical event overlap: "
            f"{sorted(event_overlap)[:20]}"
        )
    return train_events, dev_events


def _build_model_config(
    cache: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    shape = cache["features"].shape
    return {
        "feature_dim": int(shape[-1]),
        "num_sensors": int(shape[1]),
        "num_roles": int(shape[2]),
        "bottleneck_dim": int(args.bottleneck_dim),
        "dropout": float(args.dropout),
        "residual_cap": float(args.residual_cap),
    }


def _base_contract_for_mode(
    cache: Mapping[str, Any], base_mode: str
) -> dict[str, Any]:
    """Return split-invariant semantics for the selected historical-PTH base."""

    definitions = cache.get("base_definitions")
    definition = (
        definitions.get(base_mode)
        if isinstance(definitions, Mapping)
        else None
    )
    sensor_names = cache.get("sensor_names")
    return {
        "base_mode": str(base_mode),
        "definition": None if definition is None else str(definition),
        "sensor_names": (
            None
            if sensor_names is None
            else [str(value) for value in sensor_names]
        ),
    }


def build_model(config: Mapping[str, Any]) -> GatedDelta360Head:
    required = {
        "feature_dim",
        "num_sensors",
        "num_roles",
        "bottleneck_dim",
        "dropout",
        "residual_cap",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"model config misses {missing}")
    return GatedDelta360Head(
        int(config["feature_dim"]),
        num_sensors=int(config["num_sensors"]),
        num_roles=int(config["num_roles"]),
        bottleneck_dim=int(config["bottleneck_dim"]),
        dropout=float(config["dropout"]),
        residual_cap=float(config["residual_cap"]),
    )


def predict_probabilities(
    model: GatedDelta360Head,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    arm: str,
    base_mode: str,
) -> np.ndarray:
    model.eval()
    mode_features = _features_for_mode(cache, base_mode)
    fused_base = _base_logits_for_mode(cache, base_mode)
    sensor_base = _sensor_base_for_mode(cache, base_mode)
    probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for indices in fixed_epoch_batches(
            int(mode_features.shape[0]),
            batch_size=int(batch_size),
            seed=0,
            epoch=0,
            shuffle=False,
        ):
            output = model(
                mode_features[indices].to(
                    device=device, dtype=torch.float32
                ),
                cache["valid_mask"][indices].to(device),
                arm=arm,
                base_fused_logits=fused_base[indices].to(device),
                base_sensor_logits=(
                    None
                    if sensor_base is None
                    else sensor_base[indices].to(device)
                ),
            )
            probabilities.append(torch.sigmoid(output.fused_logits).cpu())
    return torch.cat(probabilities).numpy()


def evaluate_dev(
    model: GatedDelta360Head,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    arm: str,
    base_mode: str,
) -> tuple[dict[str, Any], np.ndarray]:
    probabilities = predict_probabilities(
        model,
        cache,
        batch_size=batch_size,
        device=device,
        arm=arm,
        base_mode=base_mode,
    )
    metrics = stratified_metrics(
        cache["labels"],
        probabilities,
        cache["availability_signatures"],
    )
    return metrics, probabilities


def _checkpoint_payload(
    *,
    model: GatedDelta360Head,
    epoch: int,
    arm: str,
    base_mode: str,
    model_config: Mapping[str, Any],
    initial_sha: str,
    parameter_signature: Mapping[str, Any],
    dev_metrics: Mapping[str, Any],
    encoder: Mapping[str, Any],
    base_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "script_version": SCRIPT_VERSION,
        "epoch": int(epoch),
        "arm": str(arm),
        "base_mode": str(base_mode),
        "model_config": dict(model_config),
        "model": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "initial_state_sha256": initial_sha,
        "parameter_signature": dict(parameter_signature),
        # Keep the exact feature/base checkpoint provenance inside both the
        # checkpoint and the selection lock. Sealed evaluation must match all
        # three records: checkpoint, lock, and test cache.
        "encoder": dict(encoder),
        "base_contract": dict(base_contract),
        "dev": dict(dev_metrics),
        "locked_threshold_candidate": float(
            dev_metrics["best_binary_f1_threshold"]
        ),
        "axis_contract": {
            "time": "masked per-sensor current-minus-history gated delta",
            "sensor": "masked sample-dependent sensor softmax fusion",
            "base": "zero-initialized bounded linear residual over existing PTH",
        },
    }


def _write_best_predictions(
    path: Path,
    cache: Mapping[str, Any],
    probabilities: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "id": cache["ids"],
            "plume_id": cache["plume_ids"],
            "availability_signature": cache["availability_signatures"],
            "label": cache["labels"].tolist(),
            "probability": probabilities,
        }
    ).to_csv(path, index=False)


def command_train(args: argparse.Namespace) -> None:
    train_path = Path(args.train_cache).expanduser().absolute()
    dev_path = Path(args.dev_cache).expanduser().absolute()
    guard_development_path(train_path, role="train-cache")
    guard_development_path(dev_path, role="dev-cache")

    output_dir = Path(args.output_dir).expanduser().absolute()
    protected = [
        output_dir / "checkpoint_best.pth",
        output_dir / "selection_lock.json",
        output_dir / "summary.json",
    ]
    conflicts = [str(path) for path in protected if path.exists()]
    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite completed selection artifacts: "
            f"{conflicts}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run_status.json"
    atomic_json(
        status_path,
        {
            "status": "running",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "dev_only": True,
            "sealed_test_read": False,
            "sealed_test_evaluations": 0,
        },
    )

    try:
        train_cache = load_feature_cache(
            train_path, ("train_core", "train")
        )
        dev_cache = load_feature_cache(dev_path, ("dev", "evaluation"))
        train_events, dev_events = _assert_cache_compatibility(
            train_cache, dev_cache
        )
        # Validate the requested feature/base pair before initializing output.
        training_features = _features_for_mode(
            train_cache, args.base_mode
        )
        _features_for_mode(dev_cache, args.base_mode)
        training_fused_base = _base_logits_for_mode(
            train_cache, args.base_mode
        )
        _base_logits_for_mode(dev_cache, args.base_mode)
        training_sensor_base = _sensor_base_for_mode(
            train_cache, args.base_mode
        )
        base_contract = _base_contract_for_mode(
            train_cache, args.base_mode
        )
        if base_contract != _base_contract_for_mode(
            dev_cache, args.base_mode
        ):
            raise ValueError(
                "train_core/dev selected base semantics differ"
            )

        set_deterministic_seed(int(args.seed))
        model_config = _build_model_config(train_cache, args)
        model = build_model(model_config)
        initial_sha = state_dict_sha256(model.state_dict())
        parameter_signature = model_parameter_signature(model)
        device = configure_device(args.device)
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        labels = train_cache["labels"].float()
        if args.pos_weight:
            positives = float(labels.sum())
            pos_weight: Optional[float] = (
                float(labels.numel() - positives) / max(positives, 1.0)
            )
        else:
            pos_weight = None

        initial_metrics, initial_probabilities = evaluate_dev(
            model,
            dev_cache,
            batch_size=int(args.eval_batch_size),
            device=device,
            arm=args.arm,
            base_mode=args.base_mode,
        )
        history: list[dict[str, Any]] = [
            {
                "epoch": 0,
                "train": None,
                "dev": initial_metrics,
                "elapsed_seconds": 0.0,
                "interpretation": (
                    "exact unchanged checkpoint before zero-initialized "
                    "gated-delta residual training"
                ),
            }
        ]
        best = copy.deepcopy(history[0])
        checkpoint_path = output_dir / "checkpoint_best.pth"
        atomic_torch(
            checkpoint_path,
            _checkpoint_payload(
                model=model,
                epoch=0,
                arm=args.arm,
                base_mode=args.base_mode,
                model_config=model_config,
                initial_sha=initial_sha,
                parameter_signature=parameter_signature,
                dev_metrics=initial_metrics,
                encoder=train_cache["encoder"],
                base_contract=base_contract,
            ),
        )
        _write_best_predictions(
            output_dir / "dev_predictions_best.csv",
            dev_cache,
            initial_probabilities,
        )
        atomic_json(output_dir / "metrics_history.json", history)
        print(
            "[gated-delta] epoch=0 exact-PTH-base "
            f"binary_F1@.5={initial_metrics['binary_f1_at_0_5']:.6f} "
            f"best_binary_F1={initial_metrics['best_binary_f1']:.6f} "
            f"AP={initial_metrics['ap']:.6f} "
            f"params={parameter_signature['parameter_count']}",
            flush=True,
        )

        started = time.monotonic()
        epochs_without_improvement = 0
        stop_reason = "configured epoch budget completed"
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            total_sum = fused_sum = sensor_sum = trust_sum = 0.0
            seen = 0
            batches = fixed_epoch_batches(
                len(labels),
                batch_size=int(args.batch_size),
                seed=int(args.seed),
                epoch=epoch,
                shuffle=True,
            )
            for indices in batches:
                features = training_features[indices].to(
                    device=device, dtype=torch.float32
                )
                valid = train_cache["valid_mask"][indices].to(device)
                target = labels[indices].to(device)
                base_fused = training_fused_base[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    features,
                    valid,
                    arm=args.arm,
                    base_fused_logits=base_fused,
                    base_sensor_logits=(
                        None
                        if training_sensor_base is None
                        else training_sensor_base[indices].to(device)
                    ),
                )
                breakdown = transient_query_loss(
                    output,
                    target,
                    auxiliary_weight=float(args.sensor_aux_weight),
                    pos_weight=pos_weight,
                )
                valid_sensor = output.sensor_valid.to(
                    output.residual_sensor_logits.dtype
                )
                sensor_trust = (
                    output.residual_sensor_logits.square() * valid_sensor
                ).sum() / valid_sensor.sum().clamp_min(1.0)
                trust = (
                    output.residual_fused_logits.square().mean()
                    + 0.25 * sensor_trust
                )
                loss = breakdown.total + float(args.residual_l2) * trust
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                loss.backward()
                if float(args.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(args.grad_clip)
                    )
                optimizer.step()
                count = int(indices.numel())
                total_sum += float(loss.detach()) * count
                fused_sum += float(breakdown.fused.detach()) * count
                sensor_sum += float(breakdown.auxiliary.detach()) * count
                trust_sum += float(trust.detach()) * count
                seen += count

            metrics, probabilities = evaluate_dev(
                model,
                dev_cache,
                batch_size=int(args.eval_batch_size),
                device=device,
                arm=args.arm,
                base_mode=args.base_mode,
            )
            record = {
                "epoch": int(epoch),
                "train": {
                    "loss": total_sum / seen,
                    "fused_bce": fused_sum / seen,
                    "sensor_aux_bce": sensor_sum / seen,
                    "residual_trust": trust_sum / seen,
                    "rows": int(seen),
                    "steps": int(len(batches)),
                },
                "dev": metrics,
                "elapsed_seconds": float(time.monotonic() - started),
            }
            history.append(record)
            atomic_json(output_dir / "metrics_history.json", history)
            score = float(metrics[args.selection_metric])
            best_score = float(best["dev"][args.selection_metric])
            if score > best_score:
                best = copy.deepcopy(record)
                epochs_without_improvement = 0
                atomic_torch(
                    checkpoint_path,
                    _checkpoint_payload(
                        model=model,
                        epoch=epoch,
                        arm=args.arm,
                        base_mode=args.base_mode,
                        model_config=model_config,
                        initial_sha=initial_sha,
                        parameter_signature=parameter_signature,
                        dev_metrics=metrics,
                        encoder=train_cache["encoder"],
                        base_contract=base_contract,
                    ),
                )
                _write_best_predictions(
                    output_dir / "dev_predictions_best.csv",
                    dev_cache,
                    probabilities,
                )
            else:
                epochs_without_improvement += 1
            print(
                f"[gated-delta] epoch={epoch}/{args.epochs} "
                f"loss={record['train']['loss']:.6f} "
                f"binary_F1@.5={metrics['binary_f1_at_0_5']:.6f} "
                f"best_binary_F1={metrics['best_binary_f1']:.6f} "
                f"threshold={metrics['best_binary_f1_threshold']:.6f} "
                f"AP={metrics['ap']:.6f} AUC={metrics['auc']:.6f}",
                flush=True,
            )
            if (
                int(args.early_stop_patience) > 0
                and epochs_without_improvement
                >= int(args.early_stop_patience)
            ):
                stop_reason = (
                    "development selection metric did not improve for "
                    f"{epochs_without_improvement} epoch(s)"
                )
                break

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        locked_threshold = float(
            checkpoint["locked_threshold_candidate"]
        )
        lock = {
            "schema_version": LOCK_SCHEMA,
            "locked_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "best_epoch": int(checkpoint["epoch"]),
            "selection_metric": str(args.selection_metric),
            "selection_score": float(
                checkpoint["dev"][args.selection_metric]
            ),
            "locked_threshold": locked_threshold,
            "arm": str(args.arm),
            "base_mode": str(args.base_mode),
            "base_contract": checkpoint["base_contract"],
            "model_config": checkpoint["model_config"],
            "encoder": train_cache["encoder"],
            "split_guard": {
                "train_plume_ids": sorted(
                    {str(value) for value in train_cache["plume_ids"]}
                ),
                "dev_plume_ids": sorted(
                    {str(value) for value in dev_cache["plume_ids"]}
                ),
                "source_train_event_ids": sorted(
                    train_events | dev_events
                ),
            },
            "train_manifest": train_cache["manifest"],
            "dev_manifest": dev_cache["manifest"],
            "threshold_source": (
                "positive-class F1 maximization on development only at the "
                "development-selected epoch"
            ),
            "test_cache_read_before_lock": False,
            "dev_only_runner": True,
        }
        atomic_json(output_dir / "selection_lock.json", lock)
        summary = {
            "schema_version": "query360-gated-delta-summary-v1",
            "script_version": SCRIPT_VERSION,
            "status": "complete",
            "protocol": (
                "train_core/dev only; epoch and threshold selected on dev; "
                "sealed test is not read here and can only be evaluated by "
                "the separate evaluate-locked command"
            ),
            "axis_contract": checkpoint["axis_contract"],
            "train_rows": int(training_features.shape[0]),
            "dev_rows": int(dev_cache["features"].shape[0]),
            "model": {
                "arm": str(args.arm),
                "base_mode": str(args.base_mode),
                "config": model_config,
                "initial_state_sha256": initial_sha,
                "parameter_signature": parameter_signature,
            },
            "training": {
                "seed": int(args.seed),
                "epochs_requested": int(args.epochs),
                "epochs_completed": int(history[-1]["epoch"]),
                "early_stop_patience": int(args.early_stop_patience),
                "stop_reason": stop_reason,
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "sensor_aux_weight": float(args.sensor_aux_weight),
                "residual_l2": float(args.residual_l2),
                "selection_metric": str(args.selection_metric),
            },
            "best": best,
            "selection_lock": lock,
            "history": history,
            "sealed_test": None,
            "sealed_test_read": False,
            "sealed_test_evaluations": 0,
            "fallback_guarantee": (
                "epoch zero is an eligible checkpoint and exactly preserves "
                "the supplied PTH logits"
            ),
        }
        atomic_json(output_dir / "summary.json", summary)
        atomic_json(
            status_path,
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "best_epoch": int(checkpoint["epoch"]),
                "best_dev_metrics": checkpoint["dev"],
                "locked_threshold": locked_threshold,
                "dev_only": True,
                "sealed_test_read": False,
                "sealed_test_evaluations": 0,
            },
        )
    except Exception as error:
        atomic_json(
            status_path,
            {
                "status": "failed",
                "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "dev_only": True,
                "sealed_test_read": False,
                "sealed_test_evaluations": 0,
            },
        )
        raise


def _validate_locked_checkpoint(
    *,
    selection_lock_path: Path,
    checkpoint_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    GatedDelta360Head,
    str,
    str,
]:
    """Validate every selection artifact before a test-cache path is read."""

    with selection_lock_path.open("r", encoding="utf-8") as stream:
        lock = json.load(stream)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("unsupported selection lock schema")

    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != lock.get("checkpoint_sha256"):
        raise ValueError("checkpoint SHA does not match selection lock")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported locked checkpoint schema")

    if int(checkpoint.get("epoch", -1)) != int(lock.get("best_epoch", -2)):
        raise ValueError("checkpoint epoch differs from selection lock")
    arm = str(lock.get("arm"))
    if arm not in GATED_DELTA_ARMS:
        raise ValueError("selection lock has an unsupported arm")
    if checkpoint.get("arm") != arm:
        raise ValueError("checkpoint arm differs from selection lock")
    base_mode = str(lock.get("base_mode"))
    if base_mode not in {"universal", "hybrid"}:
        raise ValueError("selection lock has an unsupported base_mode")
    if checkpoint.get("base_mode") != base_mode:
        raise ValueError("checkpoint base_mode differs from selection lock")
    locked_base_contract = lock.get("base_contract")
    if not isinstance(locked_base_contract, Mapping):
        raise ValueError("selection lock has no base contract")
    if checkpoint.get("base_contract") != locked_base_contract:
        raise ValueError(
            "checkpoint base contract differs from selection lock"
        )

    model_config = lock.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("selection lock has no model_config")
    if checkpoint.get("model_config") != model_config:
        raise ValueError("checkpoint model_config differs from selection lock")

    locked_encoder = lock.get("encoder")
    if not isinstance(locked_encoder, Mapping):
        raise ValueError("selection lock has no encoder provenance")
    if checkpoint.get("encoder") != locked_encoder:
        raise ValueError(
            "checkpoint encoder provenance differs from selection lock"
        )

    locked_threshold = float(lock["locked_threshold"])
    if not np.isfinite(locked_threshold) or not 0.0 <= locked_threshold <= 1.0:
        raise ValueError("locked threshold must be finite and in [0,1]")
    if locked_threshold != float(
        checkpoint["locked_threshold_candidate"]
    ):
        raise ValueError(
            "locked threshold differs from checkpoint dev candidate"
        )
    selection_metric = str(lock.get("selection_metric"))
    checkpoint_dev = checkpoint.get("dev")
    if (
        not isinstance(checkpoint_dev, Mapping)
        or selection_metric not in checkpoint_dev
    ):
        raise ValueError("locked selection metric is absent from checkpoint")
    if float(lock["selection_score"]) != float(
        checkpoint_dev[selection_metric]
    ):
        raise ValueError(
            "locked selection score differs from checkpoint dev score"
        )

    model = build_model(model_config)
    if model_parameter_signature(model) != checkpoint.get(
        "parameter_signature"
    ):
        raise ValueError("locked model parameter signature differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    return lock, checkpoint, model, checkpoint_sha, sha256_file(
        selection_lock_path
    )


def _assert_locked_cache_contract(
    cache: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
) -> None:
    """Match the sealed cache to the locked encoder/base/model contract."""

    if cache["encoder"] != lock["encoder"]:
        raise ValueError("sealed-test encoder differs from locked encoder")
    config = lock["model_config"]
    expected_shape = (
        int(config["num_sensors"]),
        int(config["num_roles"]),
        int(config["feature_dim"]),
    )
    observed_shape = tuple(cache["features"].shape[1:])
    if observed_shape != expected_shape:
        raise ValueError(
            "sealed-test feature shape differs from locked model config: "
            f"{observed_shape} vs {expected_shape}"
        )

    # Resolve every selected base tensor now, before inference. This is a
    # fail-closed check that the requested historical-PTH branch is present.
    base_mode = str(lock["base_mode"])
    if _base_contract_for_mode(cache, base_mode) != lock["base_contract"]:
        raise ValueError(
            "sealed-test base contract differs from selection lock"
        )
    mode_features = _features_for_mode(cache, base_mode)
    if tuple(mode_features.shape[1:]) != expected_shape:
        raise ValueError(
            "sealed-test selected feature shape differs from locked config"
        )
    _base_logits_for_mode(cache, base_mode)
    _sensor_base_for_mode(cache, base_mode)


def command_evaluate_locked(args: argparse.Namespace) -> None:
    """Evaluate one locked model/threshold on the sealed cache exactly once."""

    if not args.sealed_test:
        raise PermissionError(
            "refusing sealed-test evaluation without explicit --sealed-test"
        )

    output_dir = Path(args.output_dir).expanduser().absolute()
    result_path = output_dir / "sealed_test_result.json"
    predictions_path = output_dir / "sealed_test_predictions.csv"
    status_path = output_dir / "locked_eval_status.json"
    conflicts = [
        str(path)
        for path in (result_path, predictions_path, status_path)
        if path.exists()
    ]
    if conflicts:
        raise FileExistsError(
            "refusing a second sealed-test evaluation; existing artifacts: "
            f"{conflicts}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    sealed_test_read = False
    atomic_json(
        status_path,
        {
            "status": "validating_lock",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "sealed_test_read": False,
            "sealed_test_evaluations": 0,
        },
    )

    try:
        lock_path = Path(args.selection_lock).expanduser().absolute()
        checkpoint_path = Path(args.checkpoint).expanduser().absolute()
        (
            lock,
            checkpoint,
            model,
            checkpoint_sha,
            lock_sha,
        ) = _validate_locked_checkpoint(
            selection_lock_path=lock_path,
            checkpoint_path=checkpoint_path,
        )
        locked_threshold = float(lock["locked_threshold"])
        device = configure_device(args.device)
        model.to(device)

        # The lock, checkpoint, model construction, state-dict loading, and
        # fixed threshold are all validated before touching the sealed path.
        sealed_test_read = True
        atomic_json(
            status_path,
            {
                "status": "sealed_test_running",
                "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "sealed_test_read": True,
                "sealed_test_evaluations": 0,
                "selection_lock_sha256": lock_sha,
                "checkpoint_sha256": checkpoint_sha,
                "locked_threshold": locked_threshold,
            },
        )
        test_cache_path = Path(args.test_cache).expanduser().absolute()
        test_cache = load_feature_cache(test_cache_path, "test")
        _assert_locked_cache_contract(test_cache, lock=lock)

        split_guard = lock.get("split_guard")
        if not isinstance(split_guard, Mapping):
            raise ValueError("selection lock has no split_guard")
        source_plumes = {
            str(value)
            for key in ("train_plume_ids", "dev_plume_ids")
            for value in split_guard.get(key, [])
        }
        test_plumes = {str(value) for value in test_cache["plume_ids"]}
        plume_overlap = source_plumes & test_plumes
        if plume_overlap:
            raise RuntimeError(
                "sealed-test plume overlap: "
                f"{sorted(plume_overlap)[:20]}"
            )
        source_train_events = {
            str(value)
            for value in split_guard.get("source_train_event_ids", [])
        }
        test_events = {
            str(value) for value in test_cache.get("event_ids", [])
        }
        event_overlap = source_train_events & test_events

        probabilities = predict_probabilities(
            model,
            test_cache,
            batch_size=int(args.eval_batch_size),
            device=device,
            arm=str(lock["arm"]),
            base_mode=str(lock["base_mode"]),
        )
        metrics = stratified_locked_metrics(
            test_cache["labels"],
            probabilities,
            test_cache["availability_signatures"],
            locked_threshold=locked_threshold,
        )
        pd.DataFrame(
            {
                "id": test_cache["ids"],
                "plume_id": test_cache["plume_ids"],
                "availability_signature": test_cache[
                    "availability_signatures"
                ],
                "label": test_cache["labels"].tolist(),
                "probability": probabilities,
                "locked_prediction": (
                    probabilities >= locked_threshold
                ).astype(np.int64),
            }
        ).to_csv(predictions_path, index=False)

        result = {
            "artifact_type": "sealed_test_result",
            "schema_version": LOCKED_TEST_SCHEMA,
            "evaluation_count": 1,
            "evaluated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "epoch": int(checkpoint["epoch"]),
                "model_config": checkpoint["model_config"],
                "encoder": checkpoint["encoder"],
                "base_contract": checkpoint["base_contract"],
            },
            "selection_lock": {
                "path": str(lock_path),
                "sha256": lock_sha,
                "locked_threshold": locked_threshold,
                "selection_metric": lock["selection_metric"],
                "selection_score": lock["selection_score"],
                "arm": lock["arm"],
                "base_mode": lock["base_mode"],
                "base_contract": lock["base_contract"],
            },
            "cache": {
                "path": str(test_cache_path),
                "manifest": test_cache["manifest"],
                "encoder": test_cache["encoder"],
            },
            "metrics": metrics,
            "plume_overlap_with_train_or_dev": 0,
            "canonical_event_overlap_with_source_train": int(
                len(event_overlap)
            ),
            "canonical_event_overlap_note": (
                "observed property only; never used for model, epoch, or "
                "threshold selection"
            ),
            "test_threshold_search_performed": False,
            "threshold_source": "locked development threshold",
            "test_cache_read_after_selection_lock": True,
        }
        atomic_json(result_path, result)
        atomic_json(
            status_path,
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "sealed_test_read": True,
                "sealed_test_evaluations": 1,
                "result": str(result_path),
            },
        )
    except Exception as error:
        atomic_json(
            status_path,
            {
                "status": "failed",
                "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "sealed_test_read": sealed_test_read,
                "sealed_test_evaluations": 0,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser(
        "train",
        help="fit and select using train_core/dev feature caches only",
    )
    train.add_argument("--train-cache", required=True)
    train.add_argument("--dev-cache", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument(
        "--arm",
        choices=GATED_DELTA_ARMS,
        default="scale_aware_gated_delta",
    )
    train.add_argument(
        "--base-mode",
        choices=["universal", "hybrid"],
        default="hybrid",
    )
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--early-stop-patience", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--eval-batch-size", type=int, default=4096)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--sensor-aux-weight", type=float, default=0.1)
    train.add_argument("--residual-l2", type=float, default=1e-3)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--bottleneck-dim", type=int, default=32)
    train.add_argument("--dropout", type=float, default=0.05)
    train.add_argument("--residual-cap", type=float, default=4.0)
    train.add_argument("--pos-weight", action="store_true")
    train.add_argument(
        "--selection-metric",
        choices=[
            "binary_f1_at_0_5",
            "macro_f1_at_0_5",
            "best_binary_f1",
            "ap",
            "auc",
        ],
        default="best_binary_f1",
    )
    train.add_argument("--device", default="cuda:0")
    train.set_defaults(function=command_train)

    locked = subparsers.add_parser(
        "evaluate-locked",
        help=(
            "evaluate one prelocked checkpoint/threshold on sealed test "
            "without training, epoch selection, or threshold search"
        ),
    )
    locked.add_argument("--checkpoint", required=True)
    locked.add_argument("--selection-lock", required=True)
    locked.add_argument("--test-cache", required=True)
    locked.add_argument("--output-dir", required=True)
    locked.add_argument(
        "--sealed-test",
        action="store_true",
        help="explicit authorization required for the one sealed evaluation",
    )
    locked.add_argument("--eval-batch-size", type=int, default=4096)
    locked.add_argument("--device", default="cuda:0")
    locked.set_defaults(function=command_evaluate_locked)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "evaluate-locked":
        if int(args.eval_batch_size) <= 0:
            raise ValueError("--eval-batch-size must be positive")
        return
    positive_integer_fields = (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "bottleneck_dim",
    )
    for name in positive_integer_fields:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.early_stop_patience) < 0:
        raise ValueError("--early-stop-patience cannot be negative")
    if float(args.learning_rate) <= 0:
        raise ValueError("--learning-rate must be positive")
    if float(args.weight_decay) < 0:
        raise ValueError("--weight-decay cannot be negative")
    if float(args.sensor_aux_weight) < 0:
        raise ValueError("--sensor-aux-weight cannot be negative")
    if float(args.residual_l2) < 0:
        raise ValueError("--residual-l2 cannot be negative")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)
    args.function(args)


if __name__ == "__main__":
    main()
