#!/usr/bin/env python3
"""Exactly-once locked evaluator for P0, four-seed R4, and availability gate.

The sealed evaluation command is deliberately difficult to invoke: it
requires an immutable lock-manifest SHA, a literal confirmation token, a new
output directory, and a path explicitly marked sealed/test. The source cache
is streamed exactly once into local staging while hashing; all three frozen
models/rules then share the same single in-memory cache payload.

Do not invoke ``evaluate-once`` until the experiment owner explicitly
authorizes the final lock SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.tempo_20260728.availability_gate_control import (  # noqa: E402
    fixed_event_audit,
    fixed_metrics,
)
from research.tempo_20260728.tempo_legacy360_global import (  # noqa: E402
    SCRIPT_VERSION as HEAD_SCRIPT_VERSION,
    TempoGlobalHead,
    _features_for_mode,
    atomic_json,
    guard_development_path,
    predict_tempo,
    promoted_outputs,
    sha256_file,
)


SCRIPT_VERSION = "tempo-exactly-once-locked-evaluator-v1"
CONFIRMATION_TOKEN = "RUN_EXACTLY_ONCE_LOCKED_SEALED_EVALUATION"
DEFAULT_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1"
)
DEFAULT_AGGREGATE = DEFAULT_ROOT / "final_r4_4seed_v5/final_aggregate.json"
DEFAULT_R4_ROOT = DEFAULT_ROOT / "final_r4_4seed_v5"
DEFAULT_PROMOTED = Path(
    "/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5/heads"
    "/full_cache_dev_promotion_v1/promote_compact_scale_aware_d64_lr1e4"
    "/checkpoint_best.pth"
)
DEFAULT_DEV_CACHE = Path(
    "/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5/features"
    "/dev_universal_s2hybrid.pt"
)
DEFAULT_LOCK_DIR = DEFAULT_ROOT / "locked_evaluator_v9"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _artifact(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role}: {path}")
    return {
        "role": role,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def build_lock_manifest(args: argparse.Namespace) -> None:
    aggregate_path = Path(args.aggregate).expanduser().absolute()
    promoted_path = Path(args.promoted_checkpoint).expanduser().absolute()
    r4_root = Path(args.r4_root).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    for path, role in (
        (aggregate_path, "development aggregate"),
        (promoted_path, "promoted P0 checkpoint"),
        (r4_root, "R4 development run root"),
        (output_dir, "lock output"),
    ):
        guard_development_path(path, role)
    manifest_path = output_dir / "LOCK_MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing existing lock: {manifest_path}")
    aggregate = _read_json(aggregate_path)
    if bool(aggregate["test_or_sealed_read"]):
        raise RuntimeError("Source aggregate reports test/sealed read.")
    seeds = [int(seed) for seed in aggregate["seeds"]]
    if seeds != [17, 42, 73, 101]:
        raise RuntimeError(f"Unexpected final R4 seeds: {seeds}")
    r4_checkpoints = [
        _artifact(
            r4_root / f"r4_seed{seed}" / "checkpoint_best.pth",
            f"R4 seed {seed} checkpoint",
        )
        for seed in seeds
    ]
    protocol = {
        "schema_version": "tempo-locked-evaluation-protocol-v1",
        "evaluator_script_version": SCRIPT_VERSION,
        "head_script_version_at_lock": HEAD_SCRIPT_VERSION,
        "models": {
            "p0": {
                "checkpoint": _artifact(
                    promoted_path, "promoted frozen P0 checkpoint"
                ),
                "feature_mode": "universal",
                "threshold": float(
                    aggregate["p0"]["metrics"][
                        "best_binary_f1_threshold"
                    ]
                ),
            },
            "r4_global": {
                "checkpoints": r4_checkpoints,
                "seeds": seeds,
                "logit_weights": [0.25, 0.25, 0.25, 0.25],
                "arm": "r4",
                "threshold": float(
                    aggregate["logit_ensemble"]["metrics"][
                        "best_binary_f1_threshold"
                    ]
                ),
            },
            "single_sensor_r4_else_p0": {
                "selector": (
                    "R4 iff exactly one current sensor is valid; "
                    "otherwise P0"
                ),
                "score": "selected branch probability",
                "decision": "selected branch locked global threshold",
            },
        },
        "metrics": {
            "row": ["binary_f1", "macro_f1", "ap", "auc", "confusion"],
            "event": [
                "all_negative_fp_events",
                "all_negative_fp_rows",
                "positive_or_mixed_any_detection",
            ],
            "positive_label": 1,
            "threshold_comparator": ">=",
        },
        "prohibitions": [
            "no threshold selection",
            "no ensemble-weight selection",
            "no epoch/checkpoint selection",
            "no subgroup-rule selection",
            "no per-sensor threshold",
            "no repeat evaluation under the same lock",
        ],
        "source_development_aggregate": _artifact(
            aggregate_path, "four-seed development aggregate"
        ),
        "sealed_input": {
            "path_locked": False,
            "content_sha256_known_before_read": False,
            "required_schema": (
                "feature cache with features_universal, valid_mask, "
                "base_universal_logits, base_sensor_logits_universal, labels, "
                "ids, event_ids, availability_signatures"
            ),
            "source_read_policy": (
                "one sequential source pass into local staging with SHA256; "
                "all models share one torch-loaded staged payload"
            ),
        },
    }
    manifest = {
        "schema_version": "tempo-lock-manifest-v1",
        "status": "locked_waiting_for_explicit_sealed_authorization",
        "protocol": protocol,
        "protocol_sha256": _json_digest(protocol),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "test_or_sealed_read": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    _write_text_atomic(
        output_dir / "LOCK_MANIFEST.sha256",
        f"{manifest_sha}  {manifest_path.name}\n",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "protocol_sha256": manifest["protocol_sha256"],
                "status": manifest["status"],
                "test_or_sealed_read": False,
            },
            indent=2,
        ),
        flush=True,
    )


def validate_lock(manifest_path: Path) -> tuple[dict[str, Any], str]:
    manifest = _read_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    if manifest["status"] != "locked_waiting_for_explicit_sealed_authorization":
        raise RuntimeError("Lock manifest status changed.")
    if bool(manifest["test_or_sealed_read"]):
        raise RuntimeError("Lock construction reports test/sealed read.")
    if _json_digest(manifest["protocol"]) != manifest["protocol_sha256"]:
        raise RuntimeError("Locked protocol digest changed.")
    protocol = manifest["protocol"]
    for record in [
        protocol["models"]["p0"]["checkpoint"],
        *protocol["models"]["r4_global"]["checkpoints"],
        protocol["source_development_aggregate"],
    ]:
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Locked artifact changed: {path}")
    if protocol["models"]["r4_global"]["logit_weights"] != [0.25] * 4:
        raise RuntimeError("R4 weights are not the locked equal weights.")
    return manifest, manifest_sha


def validate_cache_payload(cache: Mapping[str, Any]) -> None:
    required = {
        "features_universal",
        "valid_mask",
        "base_universal_logits",
        "base_sensor_logits_universal",
        "labels",
        "ids",
        "event_ids",
        "availability_signatures",
    }
    missing = sorted(required - set(cache))
    if missing:
        raise RuntimeError(f"Evaluation cache is missing keys: {missing}")
    features = cache["features_universal"]
    valid = cache["valid_mask"]
    labels = cache["labels"]
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 4
        or tuple(features.shape[1:3]) != (4, 3)
    ):
        raise RuntimeError("Malformed evaluation features.")
    if valid.shape != features.shape[:3] or valid.dtype != torch.bool:
        raise RuntimeError("Malformed evaluation valid mask.")
    if labels.shape != features.shape[:1]:
        raise RuntimeError("Malformed evaluation labels.")
    if not valid[:, :, 0].any(dim=1).all():
        raise RuntimeError("Every evaluation row needs a current sensor.")


def load_r4_model(checkpoint_path: Path) -> TempoGlobalHead:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint["model_config"]
    model = TempoGlobalHead(
        int(config["feature_dim"]),
        num_sensors=int(config["num_sensors"]),
        model_dim=int(config["model_dim"]),
        residual_cap=float(config["residual_cap"]),
    )
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected R4 checkpoint keys: {incompatible.unexpected_keys}"
        )
    allowed_missing = {
        name
        for name in model.state_dict()
        if name.startswith("attention_")
    }
    if set(incompatible.missing_keys) != allowed_missing:
        raise RuntimeError(
            "R4 checkpoint missing-key set changed: "
            f"{incompatible.missing_keys}"
        )
    model.eval()
    return model


def evaluate_locked_payload(
    *,
    cache: Mapping[str, Any],
    manifest: Mapping[str, Any],
    batch_size: int,
    torch_threads: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    validate_cache_payload(cache)
    torch.set_num_threads(int(torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    protocol = manifest["protocol"]
    device = torch.device("cpu")
    promoted_path = Path(
        protocol["models"]["p0"]["checkpoint"]["path"]
    )
    promoted_checkpoint = torch.load(
        promoted_path, map_location="cpu", weights_only=False
    )
    p0_fused, p0_sensor = promoted_outputs(
        promoted_checkpoint,
        cache,
        batch_size=int(batch_size),
        device=device,
    )
    features = _features_for_mode(cache, "universal")
    valid = cache["valid_mask"].bool()
    seed_logits: list[np.ndarray] = []
    for record in protocol["models"]["r4_global"]["checkpoints"]:
        model = load_r4_model(Path(record["path"]))
        _, residual = predict_tempo(
            model,
            features,
            valid,
            p0_fused,
            p0_sensor,
            arm="r4",
            batch_size=int(batch_size),
            device=device,
        )
        seed_logits.append(p0_fused.numpy().astype(np.float64) + residual)
        del model
    weights = np.asarray(
        protocol["models"]["r4_global"]["logit_weights"],
        dtype=np.float64,
    )
    r4_logit = np.average(
        np.stack(seed_logits, axis=0), axis=0, weights=weights
    )
    p0_logit = p0_fused.numpy().astype(np.float64)
    p0_probability = 1.0 / (1.0 + np.exp(-p0_logit))
    r4_probability = 1.0 / (1.0 + np.exp(-r4_logit))
    p0_threshold = float(protocol["models"]["p0"]["threshold"])
    r4_threshold = float(protocol["models"]["r4_global"]["threshold"])
    p0_prediction = p0_probability >= p0_threshold
    r4_prediction = r4_probability >= r4_threshold
    sensor_count = valid[:, :, 0].sum(dim=1).numpy().astype(np.int64)
    use_r4 = sensor_count == 1
    gate_probability = np.where(
        use_r4, r4_probability, p0_probability
    )
    gate_prediction = np.where(use_r4, r4_prediction, p0_prediction)
    labels = cache["labels"].numpy().astype(np.int64)
    event_ids = np.asarray(
        [str(value) for value in cache["event_ids"]], dtype=str
    )
    arrays = {
        "p0": (p0_probability, p0_prediction),
        "r4_global": (r4_probability, r4_prediction),
        "single_sensor_r4_else_p0": (
            gate_probability,
            gate_prediction,
        ),
    }
    metrics = {
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
    output = pd.DataFrame(
        {
            "id": [str(value) for value in cache["ids"]],
            "event_id": event_ids,
            "availability_signature": [
                str(value) for value in cache["availability_signatures"]
            ],
            "sensor_count": sensor_count,
            "label": labels,
            "p0_probability": p0_probability,
            "p0_prediction": p0_prediction.astype(np.int8),
            "r4_probability": r4_probability,
            "r4_prediction": r4_prediction.astype(np.int8),
            "gate_probability": gate_probability,
            "gate_prediction": gate_prediction.astype(np.int8),
            "gate_used_r4": use_r4.astype(np.int8),
        }
    )
    result = {
        "schema_version": "tempo-locked-evaluation-result-v1",
        "models": metrics,
        "rows": int(len(labels)),
        "events": int(len(set(event_ids.tolist()))),
        "thresholds": {
            "p0": p0_threshold,
            "r4": r4_threshold,
            "refit": False,
        },
        "r4_seeds": protocol["models"]["r4_global"]["seeds"],
        "r4_logit_weights": weights.tolist(),
        "availability_gate": {
            "single_sensor_rows": int(use_r4.sum()),
            "multi_sensor_rows": int((~use_r4).sum()),
            "rule_refit": False,
        },
        "selection_performed": False,
    }
    return result, output


def _exclusive_intent(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def stage_source_once(
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    """One sequential source read while copying to local staging and hashing."""

    if destination.exists():
        raise FileExistsError(f"Refusing existing staged cache: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb", buffering=0) as input_stream:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o640,
        )
        with os.fdopen(descriptor, "wb", buffering=0) as output_stream:
            while True:
                block = input_stream.read(16 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                output_stream.write(block)
                byte_count += len(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    return {
        "source_path": str(source),
        "source_bytes_read_once": int(byte_count),
        "source_sha256_during_staging": digest.hexdigest(),
        "staged_path": str(destination),
        "source_content_passes": 1,
    }


def write_evaluation_outputs(
    *,
    output_dir: Path,
    result: dict[str, Any],
    predictions: pd.DataFrame,
    receipt: dict[str, Any],
) -> None:
    metrics_path = output_dir / "LOCKED_METRICS.json"
    predictions_path = output_dir / "LOCKED_PREDICTIONS.csv"
    atomic_json(metrics_path, result)
    predictions.to_csv(predictions_path, index=False)
    receipt["outputs"] = {
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
    }
    receipt["status"] = "complete"
    receipt["completed_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    atomic_json(output_dir / "RECEIPT.json", receipt)


def command_dry_run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.lock_manifest).expanduser().absolute()
    cache_path = Path(args.cache).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    for path, role in (
        (manifest_path, "lock manifest"),
        (cache_path, "development dry-run cache"),
        (output_dir, "development dry-run output"),
    ):
        guard_development_path(path, role)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing dry-run output: {output_dir}")
    manifest, manifest_sha = validate_lock(manifest_path)
    cache_sha = sha256_file(cache_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    result, predictions = evaluate_locked_payload(
        cache=cache,
        manifest=manifest,
        batch_size=int(args.batch_size),
        torch_threads=int(args.torch_threads),
    )
    output_dir.mkdir(parents=True)
    receipt = {
        "schema_version": "tempo-locked-evaluation-receipt-v1",
        "mode": "development_dry_run",
        "status": "running",
        "lock_manifest": str(manifest_path),
        "lock_manifest_sha256": manifest_sha,
        "input": {
            "path": str(cache_path),
            "sha256": cache_sha,
            "staged": False,
        },
        "runtime": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "host": platform.node(),
        },
        "test_or_sealed_read": False,
        "selection_performed": False,
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    write_evaluation_outputs(
        output_dir=output_dir,
        result=result,
        predictions=predictions,
        receipt=receipt,
    )
    print(json.dumps(result["models"], indent=2), flush=True)


def command_evaluate_once(args: argparse.Namespace) -> None:
    manifest_path = Path(args.lock_manifest).expanduser().absolute()
    source_path = Path(args.sealed_cache).expanduser().absolute()
    staging_dir = Path(args.staging_dir).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    if args.confirm != CONFIRMATION_TOKEN:
        raise RuntimeError("Exactly-once confirmation token is missing.")
    lower_source = str(source_path).lower()
    if "sealed" not in lower_source and "test" not in lower_source:
        raise RuntimeError("Sealed evaluator requires an explicitly marked path.")
    if output_dir.exists():
        raise FileExistsError(
            f"Exactly-once output already exists; refusing rerun: {output_dir}"
        )
    manifest, manifest_sha = validate_lock(manifest_path)
    if args.authorized_lock_sha256 != manifest_sha:
        raise RuntimeError("Explicitly authorized lock SHA does not match.")
    output_dir.mkdir(parents=True)
    intent_path = output_dir / "EXACTLY_ONCE_INTENT.json"
    started = pd.Timestamp.now(tz="UTC").isoformat()
    _exclusive_intent(
        intent_path,
        {
            "status": "source_read_committed",
            "lock_manifest": str(manifest_path),
            "lock_manifest_sha256": manifest_sha,
            "source_path": str(source_path),
            "started_utc": started,
            "rerun_allowed": False,
        },
    )
    staged_path = staging_dir / (
        f"locked_eval_{manifest_sha[:16]}_{source_path.name}"
    )
    stage_receipt = stage_source_once(source_path, staged_path)
    # The source is never opened again. All models share this one payload.
    cache = torch.load(staged_path, map_location="cpu", weights_only=False)
    result, predictions = evaluate_locked_payload(
        cache=cache,
        manifest=manifest,
        batch_size=int(args.batch_size),
        torch_threads=int(args.torch_threads),
    )
    result["evaluation_scope"] = "sealed_or_test_exactly_once"
    receipt = {
        "schema_version": "tempo-locked-evaluation-receipt-v1",
        "mode": "sealed_exactly_once",
        "status": "running",
        "lock_manifest": str(manifest_path),
        "lock_manifest_sha256": manifest_sha,
        "input": stage_receipt,
        "runtime": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "host": platform.node(),
        },
        "test_or_sealed_read": True,
        "selection_performed": False,
        "rerun_allowed": False,
        "started_utc": started,
    }
    write_evaluation_outputs(
        output_dir=output_dir,
        result=result,
        predictions=predictions,
        receipt=receipt,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "receipt": str(output_dir / "RECEIPT.json"),
                "lock_manifest_sha256": manifest_sha,
            },
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    lock = subparsers.add_parser(
        "build-lock", help="Freeze artifacts, rules, weights, and thresholds."
    )
    lock.set_defaults(function=build_lock_manifest)
    lock.add_argument("--aggregate", default=str(DEFAULT_AGGREGATE))
    lock.add_argument("--promoted-checkpoint", default=str(DEFAULT_PROMOTED))
    lock.add_argument("--r4-root", default=str(DEFAULT_R4_ROOT))
    lock.add_argument("--output-dir", default=str(DEFAULT_LOCK_DIR))

    dry = subparsers.add_parser(
        "dry-run", help="Run locked protocol on a development cache."
    )
    dry.set_defaults(function=command_dry_run)
    dry.add_argument(
        "--lock-manifest",
        default=str(DEFAULT_LOCK_DIR / "LOCK_MANIFEST.json"),
    )
    dry.add_argument("--cache", default=str(DEFAULT_DEV_CACHE))
    dry.add_argument(
        "--output-dir", default=str(DEFAULT_LOCK_DIR / "dev_dry_run")
    )
    dry.add_argument("--batch-size", type=int, default=512)
    dry.add_argument("--torch-threads", type=int, default=16)

    sealed = subparsers.add_parser(
        "evaluate-once",
        help="Execute one authorized sealed evaluation; never use for tuning.",
    )
    sealed.set_defaults(function=command_evaluate_once)
    sealed.add_argument("--lock-manifest", required=True)
    sealed.add_argument("--sealed-cache", required=True)
    sealed.add_argument("--staging-dir", required=True)
    sealed.add_argument("--output-dir", required=True)
    sealed.add_argument("--authorized-lock-sha256", required=True)
    sealed.add_argument("--confirm", required=True)
    sealed.add_argument("--batch-size", type=int, default=512)
    sealed.add_argument("--torch-threads", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
