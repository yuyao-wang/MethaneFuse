#!/usr/bin/env python3
"""Read-only validation evaluator for one completed RankNet checkpoint.

The evaluator never trains or updates the model. It emits one prediction CSV
and one signed JSON result for either the full temporal input or the
preregistered current-only ablation. The ablation preserves the current tensor
and zeroes recent/seasonal values (and their validity, if present) at the model
entrance after normalization.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

import multisensor_residual_runner as runner


ARTIFACT_TYPE = runner.RANKNET_VALIDATION_ARTIFACT_TYPE
INPUT_CONTRACTS = {
    "full_temporal": runner.RANKNET_FULL_INPUT_CONTRACT,
    "current_only": (
        "current_preserved_recent_seasonal_zeroed_after_normalization_v1"
    ),
}
PREDICTION_FIELDS = runner.RANKNET_PREDICTION_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one completed BCE+RankNet checkpoint on full recent "
            "validation without training."
        )
    )
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--evaluation-mode",
        choices=tuple(INPUT_CONTRACTS),
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_json_mapping(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def require_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{name} mismatch: observed_sha256="
            f"{runner.json_fingerprint(observed)}, expected_sha256="
            f"{runner.json_fingerprint(expected)}"
        )


def validate_source_contract(
    history: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> Tuple[Tuple[str, ...], Mapping[str, Any]]:
    expected_objective = runner.build_ranknet_objective_signature()
    if history.get("status") != "completed":
        raise ValueError(
            f"RankNet source history is not completed: {history.get('status')!r}"
        )
    if history.get("mode") != "supervised":
        raise ValueError("RankNet validation requires a supervised history.")
    if checkpoint.get("mode") != "supervised":
        raise ValueError("RankNet validation requires a supervised checkpoint.")
    if history.get("schema_version") != 4:
        raise ValueError(
            f"Expected RankNet history schema 4, got "
            f"{history.get('schema_version')!r}"
        )
    require_equal(
        "history supervised objective",
        history.get("supervised_objective"),
        expected_objective,
    )
    require_equal(
        "checkpoint supervised objective",
        checkpoint.get("supervised_objective"),
        expected_objective,
    )
    require_equal(
        "history/checkpoint encoder signature",
        checkpoint.get("encoder_signature"),
        history.get("encoder_signature"),
    )
    require_equal(
        "history/checkpoint data signature",
        checkpoint.get("data_signature"),
        history.get("data_signature"),
    )
    require_equal(
        "history/checkpoint resume signature",
        checkpoint.get("resume_signature"),
        history.get("resume_signature"),
    )
    resume_signature = history.get("resume_signature")
    if not isinstance(resume_signature, Mapping):
        raise TypeError("History has no resume signature mapping.")
    if resume_signature.get("schema_version") != 2:
        raise ValueError(
            "RankNet resume signature must have schema_version=2."
        )
    require_equal(
        "resume supervised objective",
        resume_signature.get("supervised_objective"),
        expected_objective,
    )
    data_signature = history.get("data_signature")
    if not isinstance(data_signature, Mapping):
        raise TypeError("History has no data signature mapping.")
    if data_signature.get("global_train_val_overlap_after") != 0:
        raise ValueError(
            "RankNet source does not declare zero global train/validation "
            "canonical-event overlap."
        )

    sensors = tuple(history.get("sensors", ()))
    if sensors != runner.SENSOR_ORDER:
        raise ValueError(
            f"Expected sensor order {runner.SENSOR_ORDER}, got {sensors}"
        )
    if tuple(checkpoint.get("sensors", ())) != sensors:
        raise ValueError("History/checkpoint sensor order mismatch.")
    if int(checkpoint.get("epoch", -1)) != 0:
        raise ValueError(
            "The preregistered RankNet gate permits only epoch-0 checkpoint."
        )
    if len(history.get("epochs", ())) != 1:
        raise ValueError("RankNet source must contain exactly one epoch.")
    return sensors, expected_objective


def verify_signed_inputs(
    history: Mapping[str, Any],
    sensors: Sequence[str],
) -> Tuple[
    Dict[str, Dict[str, Path]],
    Path,
    Dict[str, Mapping[str, Any]],
]:
    resolved = history.get("resolved_csvs")
    if not isinstance(resolved, Mapping):
        raise TypeError("History has no resolved_csvs mapping.")
    csvs: Dict[str, Dict[str, Path]] = {}
    for sensor in sensors:
        sensor_paths = resolved.get(sensor)
        if not isinstance(sensor_paths, Mapping):
            raise TypeError(f"Missing resolved CSV mapping for {sensor}.")
        csvs[sensor] = {
            split: Path(str(sensor_paths[split]))
            for split in ("train", "val")
        }

    data_signature = history["data_signature"]
    expected_manifest_hashes = data_signature.get("manifest_sha256")
    if not isinstance(expected_manifest_hashes, Mapping):
        raise TypeError("Data signature has no manifest hashes.")
    for sensor in sensors:
        for split in ("train", "val"):
            path = csvs[sensor][split]
            observed_sha = runner.sha256_file(path)
            expected_sha = expected_manifest_hashes[sensor][split]
            if observed_sha != expected_sha:
                raise ValueError(
                    f"{sensor}/{split} manifest hash mismatch: "
                    f"observed={observed_sha}, expected={expected_sha}"
                )

    stats_path = Path(str(history["normalization_stats"]))
    observed_stats_sha = runner.sha256_file(stats_path)
    expected_stats_sha = data_signature.get("normalization_stats_sha256")
    if observed_stats_sha != expected_stats_sha:
        raise ValueError(
            "Normalization-statistics hash mismatch: "
            f"observed={observed_stats_sha}, expected={expected_stats_sha}"
        )
    stats_payload = load_json_mapping(stats_path)
    stats_container = stats_payload.get("sensors", stats_payload)
    if not isinstance(stats_container, Mapping):
        raise TypeError(
            f"Normalization statistics have no sensor mapping: {stats_path}"
        )
    stats: Dict[str, Mapping[str, Any]] = {}
    for sensor in sensors:
        value = stats_container.get(sensor)
        if not isinstance(value, Mapping):
            raise TypeError(f"Normalization statistics missing {sensor}.")
        stats[sensor] = value
    return csvs, stats_path, stats


def runtime_namespace(
    history: Mapping[str, Any],
    args: argparse.Namespace,
) -> Namespace:
    config = history.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("History has no config mapping.")
    values = dict(config)
    values.update(
        {
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "max_val_batches": 0,
            "augment": bool(config.get("augment", True)),
        }
    )
    if values.get("validity_masked_reconstruction"):
        raise ValueError(
            "The RankNet fallback must not use validity-masked reconstruction."
        )
    if int(args.batch_size) <= 0 or int(args.num_workers) < 0:
        raise ValueError("Invalid evaluator batch size or worker count.")
    return Namespace(**values)


def build_model(
    history: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    sensors: Sequence[str],
    device: torch.device,
) -> runner.MultiSensorResidualModel:
    config = history["config"]
    model = runner.MultiSensorResidualModel(
        sensors,
        mode="supervised",
        sharing=str(config["sharing"]),
        image_size=int(config["image_size"]),
        patch_size=int(config["patch_size"]),
        embed_dim=int(config["embed_dim"]),
        depth=int(config["depth"]),
        num_heads=int(config["num_heads"]),
        mlp_ratio=float(config["mlp_ratio"]),
        fuse_freq=int(config["fuse_freq"]),
        dropout=float(config["dropout"]),
        mask_ratio=float(config["mask_ratio"]),
        decoder_embed_dim=int(config["decoder_embed_dim"]),
        decoder_depth=int(config["decoder_depth"]),
        decoder_num_heads=int(config["decoder_num_heads"]),
        validity_masked_reconstruction=False,
    )
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint has no model state mapping.")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def apply_input_contract(
    streams: Mapping[str, torch.Tensor],
    validity_by_stream: Optional[Mapping[str, torch.Tensor]],
    evaluation_mode: str,
) -> Tuple[
    Dict[str, torch.Tensor],
    Optional[Dict[str, torch.Tensor]],
]:
    if evaluation_mode == "full_temporal":
        return dict(streams), (
            None
            if validity_by_stream is None
            else dict(validity_by_stream)
        )
    if evaluation_mode != "current_only":
        raise ValueError(f"Unsupported evaluation mode: {evaluation_mode}")

    expected_suffixes = set(runner.STREAM_SUFFIXES)
    observed_suffixes = {
        name.rsplit("_", 1)[-1] for name in streams
    }
    if observed_suffixes != expected_suffixes:
        raise ValueError(
            f"Current-only ablation expected streams {expected_suffixes}, "
            f"got {observed_suffixes}"
        )
    zeroed_streams = {
        name: (
            tensor
            if name.endswith("_current")
            else torch.zeros_like(tensor)
        )
        for name, tensor in streams.items()
    }
    zeroed_validity = None
    if validity_by_stream is not None:
        if set(validity_by_stream) != set(streams):
            raise ValueError("Value/validity keys differ before ablation.")
        zeroed_validity = {
            name: (
                tensor
                if name.endswith("_current")
                else torch.zeros_like(tensor)
            )
            for name, tensor in validity_by_stream.items()
        }
    return zeroed_streams, zeroed_validity


def probability_diagnostics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    return {
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_mean": float(probabilities.mean()),
        "probability_std": float(probabilities.std()),
        "predicted_positive_at_0p5": int((probabilities >= 0.5).sum()),
        "predicted_positive_at_best": int(
            (probabilities >= float(threshold)).sum()
        ),
    }


@torch.inference_mode()
def evaluate(
    model: runner.MultiSensorResidualModel,
    val_loaders: Mapping[str, torch.utils.data.DataLoader],
    *,
    sensors: Sequence[str],
    device: torch.device,
    amp: bool,
    image_size: int,
    validity_masked_reconstruction: bool,
    evaluation_mode: str,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    List[Dict[str, Any]],
]:
    per_sensor: Dict[str, Dict[str, Any]] = {}
    prediction_rows: List[Dict[str, Any]] = []
    for sensor in sensors:
        labels_all: List[float] = []
        probabilities_all: List[float] = []
        row_index = 0
        for model_inputs, labels in val_loaders[sensor]:
            streams, validity_by_stream = runner.move_model_inputs(
                model_inputs,
                device,
                image_size,
                validity_masked_reconstruction=validity_masked_reconstruction,
            )
            streams, validity_by_stream = apply_input_contract(
                streams,
                validity_by_stream,
                evaluation_mode,
            )
            labels_device = labels.to(device=device, non_blocking=True)
            with runner.amp_context(device, amp):
                logits = model(
                    sensor,
                    streams,
                    validity_by_stream=validity_by_stream,
                )["logits"]
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            labels_numpy = labels_device.float().cpu().numpy()
            for label, probability in zip(
                labels_numpy.tolist(),
                probabilities.tolist(),
            ):
                prediction_rows.append(
                    {
                        "sensor": sensor,
                        "row_index": row_index,
                        "label": int(label),
                        "probability": float(probability),
                    }
                )
                row_index += 1
            labels_all.extend(labels_numpy.tolist())
            probabilities_all.extend(probabilities.tolist())

        labels_array = np.asarray(labels_all, dtype=np.int64)
        probabilities_array = np.asarray(
            probabilities_all,
            dtype=np.float64,
        )
        if labels_array.size == 0:
            raise ValueError(f"No validation predictions for {sensor}.")
        metrics = runner.classification_metrics(
            labels_array,
            probabilities_array,
        )
        threshold = metrics.get("threshold")
        if threshold is None or not math.isfinite(float(threshold)):
            raise ValueError(f"No finite best-F1 threshold for {sensor}.")
        metrics.update(
            probability_diagnostics(
                labels_array,
                probabilities_array,
                float(threshold),
            )
        )
        per_sensor[sensor] = runner.json_safe(metrics)
    return per_sensor, prediction_rows


def write_prediction_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=PREDICTION_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    runner.atomic_text_dump(buffer.getvalue(), path)


def main() -> None:
    args = parse_args()
    for path in (args.history, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_json.resolve() == args.output_csv.resolve():
        raise ValueError("--output-json and --output-csv must differ.")
    existing = [
        path
        for path in (args.output_json, args.output_csv)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite evaluation artifacts: {existing}"
        )

    source_history_sha256 = runner.sha256_file(args.history)
    checkpoint_sha256 = runner.sha256_file(args.checkpoint)
    history = load_json_mapping(args.history)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint payload is not a mapping.")
    sensors, objective = validate_source_contract(history, checkpoint)
    csvs, stats_path, stats = verify_signed_inputs(history, sensors)

    runtime_args = runtime_namespace(history, args)
    device = runner.choose_device(args.device)
    runner.seed_everything(int(history["config"]["seed"]))
    (
        _train_datasets,
        val_datasets,
        _train_loaders,
        val_loaders,
    ) = runner.build_data(
        runtime_args,
        sensors,
        csvs,
        stats,
        device,
    )
    model = build_model(history, checkpoint, sensors, device)
    started = time.time()
    per_sensor, prediction_rows = evaluate(
        model,
        val_loaders,
        sensors=sensors,
        device=device,
        amp=bool(args.amp),
        image_size=int(runtime_args.image_size),
        validity_masked_reconstruction=bool(
            runtime_args.validity_masked_reconstruction
        ),
        evaluation_mode=args.evaluation_mode,
    )
    expected_counts = {
        sensor: len(val_datasets[sensor]) for sensor in sensors
    }
    observed_counts = {
        sensor: int(per_sensor[sensor]["samples"]) for sensor in sensors
    }
    if observed_counts != expected_counts:
        raise AssertionError(
            f"Full-validation sample-count mismatch: "
            f"observed={observed_counts}, expected={expected_counts}"
        )

    macro = {
        key: runner.finite_macro(per_sensor, key)
        for key in ("ap", "auroc", "f1", "f1_0p5")
    }
    write_prediction_csv(prediction_rows, args.output_csv)
    prediction_sha256 = runner.sha256_file(args.output_csv)
    result = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "completed",
        "evaluation_mode": args.evaluation_mode,
        "input_contract": INPUT_CONTRACTS[args.evaluation_mode],
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "source_history_path": str(args.history.resolve()),
        "source_history_sha256": source_history_sha256,
        "supervised_objective": objective,
        "encoder_signature": history["encoder_signature"],
        "data_signature": history["data_signature"],
        "resume_signature": history["resume_signature"],
        "encoder_signature_fingerprint": runner.json_fingerprint(
            history["encoder_signature"]
        ),
        "data_signature_fingerprint": runner.json_fingerprint(
            history["data_signature"]
        ),
        "resume_signature_fingerprint": runner.json_fingerprint(
            history["resume_signature"]
        ),
        "event_protocol_fingerprint": history["data_signature"][
            "event_protocol_fingerprint"
        ],
        "normalization_stats_path": str(stats_path.resolve()),
        "normalization_stats_sha256": runner.sha256_file(stats_path),
        "predictions_csv": str(args.output_csv.resolve()),
        "predictions_csv_sha256": prediction_sha256,
        "prediction_rows": len(prediction_rows),
        "full_sample_counts": observed_counts,
        "per_sensor": per_sensor,
        "macro_over_sensor": macro,
        "runtime": {
            "device": str(device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "amp": bool(args.amp),
            "elapsed_seconds": time.time() - started,
        },
        "completed_unix": time.time(),
    }
    runner.atomic_json_dump(runner.json_safe(result), args.output_json)
    print(
        "[RankNet validation] "
        + json.dumps(
            {
                "status": "completed",
                "evaluation_mode": args.evaluation_mode,
                "checkpoint_sha256": checkpoint_sha256,
                "prediction_rows": len(prediction_rows),
                "prediction_csv_sha256": prediction_sha256,
                "macro_over_sensor": macro,
                "result": str(args.output_json),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
