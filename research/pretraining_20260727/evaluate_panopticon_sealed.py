#!/usr/bin/env python3
"""Leakage-safe sealed evaluation for L89, EMIT32, and S5P Panopticon runs.

The protocol is deliberately split into two commands:

1. ``freeze-threshold`` verifies that an explicitly supplied checkpoint was
   selected by validation AP, runs that checkpoint on the full validation
   manifest, and writes a threshold artifact bound to the checkpoint,
   manifests, model/input schema, normalization, and training code.
2. ``evaluate-sealed`` accepts only that frozen artifact.  It has no threshold
   search or checkpoint discovery option and rejects development/test event
   overlap before inference.

Dataset retry logic is bypassed during both phases.  A missing or corrupt row
therefore aborts evaluation instead of silently replacing it with another row.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727.temporal_input_modes import (  # noqa: E402
    output_timepoints,
    parse_residual_slots,
)


PROTOCOL = "methanefuse-panopticon-sealed-v1"
CANONICAL_EVENT_RULE = "strip-final-hyphen-suffix-v1"
EVENT_SUFFIX_RE = re.compile(r"-[A-Za-z0-9]+$")


@dataclass(frozen=True)
class SensorAdapter:
    module_name: str
    dataset_class: str
    schema_keys: tuple[str, ...]


SENSORS = {
    "l89": SensorAdapter(
        module_name="Upgraded_dataset.dino_classifier_head_l89_temporal_satmae",
        dataset_class="L89TemporalSequenceDataset",
        schema_keys=(
            "path_columns",
            "time_columns",
            "band_indices",
            "input_mode",
            "residual_slots",
            "residual_clip",
            "pad_to_multiple",
            "time_base_year",
            "time_embed_dim",
            "embed_dim",
            "use_precomputed_stats",
            "stats_samples",
            "stats_seed",
            "stats_workers",
        ),
    ),
    "emit32": SensorAdapter(
        module_name="Upgraded_dataset.dino_classifier_head_emit32_temporal_satmae",
        dataset_class="Emit32TemporalSequenceDataset",
        schema_keys=(
            "path_columns",
            "time_columns",
            "input_mode",
            "residual_slots",
            "residual_clip",
            "pad_to_multiple",
            "time_base_year",
            "time_embed_dim",
            "embed_dim",
        ),
    ),
    "s5p": SensorAdapter(
        module_name="Upgraded_dataset.dino_classifier_head_s5p_temporal_satmae",
        dataset_class="S5PTemporalSequenceDataset",
        schema_keys=(
            "path_column",
            "time_path_columns",
            "data_key",
            "channel_last",
            "allow_pickle",
            "nan_to_num",
            "chn_id_value",
            "input_mode",
            "residual_slots",
            "residual_clip",
            "pad_to_multiple",
            "pad_value",
            "time_base_year",
            "time_embed_dim",
            "embed_dim",
            "use_precomputed_stats",
            "stats_samples",
            "stats_seed",
            "stats_workers",
        ),
    ),
}


class StrictIndexDataset(Dataset):
    """Call ``_get_one`` directly so evaluation never substitutes bad rows."""

    def __init__(self, dataset: Dataset):
        if not hasattr(dataset, "_get_one"):
            raise TypeError(
                f"{type(dataset).__name__} has no _get_one method required for strict evaluation"
            )
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset._get_one(index)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite protocol output: {path}. "
            "Use a new path so a frozen decision cannot be changed in place."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_predictions(
    path: Path,
    manifest_path: Path,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {path}")
    frame = pd.read_csv(manifest_path, low_memory=False)
    if len(frame) != len(probabilities):
        raise ValueError(
            f"Prediction count {len(probabilities)} != manifest rows {len(frame)}"
        )
    frame = frame.copy()
    frame["methanefuse_probability"] = probabilities
    frame["methanefuse_prediction"] = (
        probabilities >= threshold
    ).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def canonical_events(frame: pd.DataFrame, path: Path) -> list[str]:
    if "plume_id" not in frame.columns:
        raise ValueError(
            f"{path} has no plume_id column; sealed event-overlap audit cannot run"
        )
    values = (
        frame["plume_id"]
        .astype("string")
        .str.strip()
        .str.replace(EVENT_SUFFIX_RE, "", regex=True)
    )
    invalid = values.isna() | values.eq("")
    if invalid.any():
        raise ValueError(
            f"{path} has {int(invalid.sum())} empty canonical plume events"
        )
    return sorted(set(values.astype(str)))


def manifest_info(path: Path, *, include_events: bool = False) -> dict:
    path = path.resolve()
    frame = pd.read_csv(path, low_memory=False)
    if "label" not in frame.columns:
        raise ValueError(f"{path} has no label column")
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    unexpected = sorted(set(labels) - {0, 1})
    if unexpected:
        raise ValueError(f"{path} has non-binary labels: {unexpected}")
    events = canonical_events(frame, path)
    result = {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "labels": {
            str(int(label)): int(count)
            for label, count in labels.value_counts().sort_index().items()
        },
        "canonical_event_rule": CANONICAL_EVENT_RULE,
        "canonical_event_count": len(events),
        "canonical_event_sha256": json_fingerprint(events),
    }
    if include_events:
        result["canonical_events"] = events
    return result


def load_checkpoint(path: Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a dictionary: {path}")
    for key in ("backbone", "head", "args"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint {path} is missing key '{key}'")
    saved_args = dict(checkpoint["args"])
    return checkpoint, saved_args


def module_and_adapter(sensor: str):
    adapter = SENSORS[sensor]
    module = importlib.import_module(adapter.module_name)
    module_path = Path(module.__file__).resolve()
    return module, adapter, module_path


def saved_model_schema(
    sensor: str, saved_args: dict, module_path: Path
) -> dict:
    adapter = SENSORS[sensor]
    missing = [key for key in adapter.schema_keys if key not in saved_args]
    if missing:
        raise KeyError(
            f"Checkpoint args for {sensor} are missing schema keys: {missing}"
        )
    schema = {
        "sensor": sensor,
        "trainer_module": adapter.module_name,
        "trainer_code_path": str(module_path),
        "trainer_code_sha256": sha256_file(module_path),
        "arguments": {key: saved_args[key] for key in adapter.schema_keys},
    }
    if sensor in {"l89", "emit32"}:
        path_columns = parse_columns(saved_args["path_columns"])
    else:
        path_columns = parse_columns(saved_args["time_path_columns"])
    residual_slots = parse_residual_slots(saved_args["residual_slots"])
    schema["source_timepoints"] = len(path_columns)
    schema["model_timepoints"] = output_timepoints(
        saved_args["input_mode"], len(path_columns), residual_slots
    )
    schema["fingerprint"] = json_fingerprint(schema)
    return schema


def parse_columns(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        columns = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        columns = tuple(str(part).strip() for part in value if str(part).strip())
    if not columns:
        raise ValueError("Expected at least one column name")
    return columns


def verify_validation_ap_selection(
    checkpoint: dict, metrics_path: Path
) -> dict:
    history = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"Metrics history must be a non-empty list: {metrics_path}")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    records = [
        record
        for record in history
        if int(record.get("epoch", -2)) == checkpoint_epoch
    ]
    if len(records) != 1:
        raise ValueError(
            f"Expected one metrics record for checkpoint epoch {checkpoint_epoch}; "
            f"found {len(records)} in {metrics_path}"
        )
    finite_records = []
    for record in history:
        value = record.get("test_ap")
        if value is not None and math.isfinite(float(value)):
            finite_records.append((int(record["epoch"]), float(value)))
    if not finite_records:
        raise ValueError(f"No finite validation AP values in {metrics_path}")
    best_ap = max(value for _, value in finite_records)
    best_epochs = sorted(
        epoch
        for epoch, value in finite_records
        if math.isclose(value, best_ap, rel_tol=0.0, abs_tol=1e-12)
    )
    selected_ap = float(records[0]["test_ap"])
    if checkpoint_epoch not in best_epochs:
        raise ValueError(
            f"Checkpoint epoch {checkpoint_epoch} has validation AP={selected_ap}, "
            f"but AP-selected epoch(s) are {best_epochs} with AP={best_ap}"
        )
    saved_best = checkpoint.get("best_val_ap")
    if saved_best is not None and not math.isclose(
        float(saved_best), best_ap, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"Checkpoint best_val_ap={saved_best} disagrees with history best AP={best_ap}"
        )
    return {
        "rule": "maximum_validation_ap",
        "metrics_history_path": str(metrics_path.resolve()),
        "metrics_history_sha256": sha256_file(metrics_path),
        "checkpoint_epoch": checkpoint_epoch,
        "selected_validation_ap_logged": selected_ap,
        "best_validation_ap_logged": best_ap,
        "tied_best_epochs": best_epochs,
        "selected_epoch_record": records[0],
    }


def verify_development_manifests(
    saved_args: dict, validation_csv: Path
) -> tuple[dict, dict, list[str]]:
    if "train_csv" not in saved_args or "test_csv" not in saved_args:
        raise KeyError("Checkpoint args must contain train_csv and test_csv")
    train_path = Path(saved_args["train_csv"]).resolve()
    saved_validation_path = Path(saved_args["test_csv"]).resolve()
    validation_csv = validation_csv.resolve()
    for path in (train_path, saved_validation_path, validation_csv):
        if not path.is_file():
            raise FileNotFoundError(path)

    supplied_validation_sha = sha256_file(validation_csv)
    saved_validation_sha = sha256_file(saved_validation_path)
    if supplied_validation_sha != saved_validation_sha:
        raise ValueError(
            "--validation_csv is not byte-identical to the validation manifest "
            f"saved in the checkpoint args ({saved_validation_path})"
        )

    train_info = manifest_info(train_path, include_events=True)
    validation_info = manifest_info(validation_csv, include_events=True)
    train_events = set(train_info.pop("canonical_events"))
    validation_events = set(validation_info.pop("canonical_events"))
    overlap = train_events & validation_events
    if overlap:
        raise ValueError(
            f"Training/validation canonical event overlap is {len(overlap)}"
        )
    development_events = sorted(train_events | validation_events)
    return train_info, validation_info, development_events


def resolve_normalization(
    sensor: str, module, saved_args: dict
) -> tuple[list[float], list[float], dict]:
    train_csv = str(Path(saved_args["train_csv"]).resolve())
    if sensor == "emit32":
        mean, std = module.PRECOMPUTED_STATS
        provenance = {
            "kind": "trainer_precomputed_constant",
            "trainer_symbol": "PRECOMPUTED_STATS",
        }
    elif sensor == "l89" and bool(saved_args["use_precomputed_stats"]):
        mean, std = module.PRECOMPUTED_STATS
        provenance = {
            "kind": "trainer_precomputed_constant",
            "trainer_symbol": "PRECOMPUTED_STATS",
        }
    elif sensor == "l89":
        path_columns = parse_columns(saved_args["path_columns"])
        mean, std = module.compute_l89_temporal_stats(
            train_csv,
            path_columns=path_columns,
            n_samples=int(saved_args["stats_samples"]),
            seed=int(saved_args["stats_seed"]),
            num_workers=int(saved_args["stats_workers"]),
            local_file_cache=None,
        )
        provenance = {
            "kind": "deterministically_reconstructed_from_training_manifest",
            "stats_samples": int(saved_args["stats_samples"]),
            "stats_seed": int(saved_args["stats_seed"]),
            "stats_workers": int(saved_args["stats_workers"]),
        }
    elif sensor == "s5p":
        if bool(saved_args["use_precomputed_stats"]):
            if module.PRECOMPUTED_STATS is None:
                raise ValueError(
                    "Checkpoint requests precomputed S5P stats, but trainer has none"
                )
            mean, std = module.PRECOMPUTED_STATS
            provenance = {
                "kind": "trainer_precomputed_constant",
                "trainer_symbol": "PRECOMPUTED_STATS",
            }
        else:
            mean, std = module.compute_s5p_temporal_stats(
                train_csv,
                path_column=saved_args["path_column"],
                n_samples=int(saved_args["stats_samples"]),
                seed=int(saved_args["stats_seed"]),
                num_workers=int(saved_args["stats_workers"]),
                local_file_cache=None,
                data_key=saved_args["data_key"],
                channel_last=bool(saved_args["channel_last"]),
                allow_pickle=bool(saved_args["allow_pickle"]),
                num_timepoints=len(module.TIMEPOINTS),
            )
            provenance = {
                "kind": "deterministically_reconstructed_from_training_manifest",
                "stats_samples": int(saved_args["stats_samples"]),
                "stats_seed": int(saved_args["stats_seed"]),
                "stats_workers": int(saved_args["stats_workers"]),
            }
    else:
        raise ValueError(sensor)

    mean_values = [float(value) for value in np.asarray(mean).reshape(-1)]
    std_values = [float(value) for value in np.asarray(std).reshape(-1)]
    if (
        not mean_values
        or len(mean_values) != len(std_values)
        or not np.isfinite(mean_values).all()
        or not np.isfinite(std_values).all()
        or (np.asarray(std_values) <= 0).any()
    ):
        raise ValueError(
            f"Invalid normalization mean/std lengths or values: {mean_values}, {std_values}"
        )
    provenance["training_manifest_path"] = train_csv
    provenance["training_manifest_sha256"] = sha256_file(Path(train_csv))
    return mean_values, std_values, provenance


def build_dataset(
    sensor: str,
    module,
    saved_args: dict,
    csv_path: Path,
    normalization: tuple[Sequence[float], Sequence[float]],
) -> StrictIndexDataset:
    residual_slots = parse_residual_slots(saved_args["residual_slots"])
    common = {
        "csv_path": str(csv_path.resolve()),
        "normalize_stats": normalization,
        "time_base_year": int(saved_args["time_base_year"]),
        "pad_to_multiple": int(saved_args["pad_to_multiple"]),
        "skip_invalid_samples": False,
        "local_file_cache": None,
        "cache_prefetch_rows": 0,
        "input_mode": saved_args["input_mode"],
        "residual_slots": residual_slots,
        "residual_clip": float(saved_args["residual_clip"]),
    }
    if sensor == "l89":
        dataset = module.L89TemporalSequenceDataset(
            **common,
            path_columns=parse_columns(saved_args["path_columns"]),
            time_columns=parse_columns(saved_args["time_columns"]),
            band_indices=module.parse_band_indices(saved_args["band_indices"]),
        )
    elif sensor == "emit32":
        dataset = module.Emit32TemporalSequenceDataset(
            **common,
            path_columns=parse_columns(saved_args["path_columns"]),
            time_columns=parse_columns(saved_args["time_columns"]),
            skip_path_validation=False,
        )
    elif sensor == "s5p":
        dataset = module.S5PTemporalSequenceDataset(
            **common,
            path_column=saved_args["path_column"],
            time_path_columns=parse_columns(saved_args["time_path_columns"]),
            label_column="label",
            pad_value=float(saved_args["pad_value"]),
            data_key=saved_args["data_key"],
            channel_last=bool(saved_args["channel_last"]),
            allow_pickle=bool(saved_args["allow_pickle"]),
            nan_to_num=float(saved_args["nan_to_num"]),
            chn_id_value=float(saved_args["chn_id_value"]),
            max_retries=1,
        )
    else:
        raise ValueError(sensor)
    return StrictIndexDataset(dataset)


def build_model(
    sensor: str,
    module,
    saved_args: dict,
    checkpoint: dict,
    device: torch.device,
):
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # The selected checkpoint contains the entire trained backbone, so reading
    # the original teacher checkpoint first is unnecessary and error-prone.
    panopticon = module.load_backbone("scratch", device=device)
    if sensor in {"l89", "emit32"}:
        path_columns = parse_columns(saved_args["path_columns"])
    else:
        path_columns = parse_columns(saved_args["time_path_columns"])
    residual_slots = parse_residual_slots(saved_args["residual_slots"])
    backbone = module.TemporalPanopticonBackbone(
        panopticon,
        num_timepoints=output_timepoints(
            saved_args["input_mode"], len(path_columns), residual_slots
        ),
        time_embed_dim=int(saved_args["time_embed_dim"]),
    ).to(device)
    head = module.CLSHead(
        embed_dim=int(saved_args["embed_dim"]), num_classes=2
    ).to(device)
    module.load_state_dict_flexible(backbone, checkpoint["backbone"])
    module.load_state_dict_flexible(head, checkpoint["head"])
    backbone.eval()
    head.eval()
    return backbone, head


def infer(
    dataset: StrictIndexDataset,
    backbone,
    head,
    module,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        # Evaluation workers exit after the pass and do not add persistent CPU
        # pressure alongside the parallel experiments.
        loader_kwargs["persistent_workers"] = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    probability_batches = []
    target_batches = []
    with torch.inference_mode():
        for step, (x_dict, labels) in enumerate(loader, 1):
            x_dict = module.recursive_to_device(x_dict, device)
            with autocast(enabled=device.type == "cuda"):
                features = backbone(x_dict, is_training=True)
                logits = head(features["x_norm_clstoken"])
            probability_batches.append(
                F.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
            )
            target_batches.append(labels.numpy())
            if step % 100 == 0 or step == len(loader):
                print(
                    f"[Inference] batches={step}/{len(loader)} "
                    f"rows={sum(len(x) for x in target_batches)}/{len(dataset)}",
                    flush=True,
                )
    if not probability_batches:
        raise ValueError("Cannot evaluate an empty dataset")
    probabilities = np.concatenate(probability_batches).astype(np.float64)
    targets = np.concatenate(target_batches).astype(np.int64)
    if len(probabilities) != len(dataset):
        raise RuntimeError(
            f"Inference produced {len(probabilities)} predictions for {len(dataset)} rows"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Inference produced non-finite probabilities")
    if set(np.unique(targets)) - {0, 1}:
        raise ValueError("Inference targets are not binary")
    return targets, probabilities


def choose_validation_threshold(
    targets: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float, dict]:
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("Validation threshold selection requires both classes")
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    if thresholds.size == 0:
        raise ValueError("Validation precision-recall curve has no thresholds")
    f1_curve = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    index = int(np.nanargmax(f1_curve))
    threshold_index = min(index, thresholds.size - 1)
    threshold = float(thresholds[threshold_index])
    return (
        threshold,
        float(f1_curve[index]),
        {
            "rule": "sklearn_precision_recall_curve_first_f1_argmax",
            "argmax_index": index,
            "threshold_index": threshold_index,
            "candidate_threshold_count": int(thresholds.size),
            "prediction_comparison": "probability >= threshold",
        },
    )


def metrics_at_threshold(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict:
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("AP/AUROC evaluation requires both classes")
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (targets == 1)).sum())
    fp = int(((predictions == 1) & (targets == 0)).sum())
    fn = int(((predictions == 0) & (targets == 1)).sum())
    tn = int(((predictions == 0) & (targets == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    roc_fpr, roc_tpr, _ = roc_curve(targets, probabilities)
    within_01 = roc_tpr[roc_fpr <= 0.01]
    within_05 = roc_tpr[roc_fpr <= 0.05]
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(targets)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "recall": float(recall),
        "fpr": float(fpr),
        "auroc": float(roc_auc_score(targets, probabilities)),
        "ap": float(average_precision_score(targets, probabilities)),
        "recall_at_fpr_01": float(within_01.max()) if within_01.size else 0.0,
        "recall_at_fpr_05": float(within_05.max()) if within_05.size else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "samples": int(len(targets)),
        "positives": int((targets == 1).sum()),
        "negatives": int((targets == 0).sum()),
    }


def runtime_info(device: torch.device) -> dict:
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "device": str(device),
        "amp": device.type == "cuda",
    }
    if device.type == "cuda":
        result["cuda"] = torch.version.cuda
        result["gpu_name"] = torch.cuda.get_device_name(device)
    return result


def freeze_threshold(args) -> None:
    checkpoint_path = args.checkpoint.resolve()
    validation_csv = args.validation_csv.resolve()
    output_artifact = args.output_artifact.resolve()
    requested_outputs = [output_artifact]
    if args.predictions_csv is not None:
        requested_outputs.append(args.predictions_csv.resolve())
    if len(set(requested_outputs)) != len(requested_outputs):
        raise ValueError("Threshold artifact and predictions must use different paths")
    existing = [path for path in requested_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to run because protocol output already exists: {existing}"
        )
    metrics_path = (
        args.metrics_history.resolve()
        if args.metrics_history is not None
        else checkpoint_path.parent / "metrics_history.json"
    )
    checkpoint, saved_args = load_checkpoint(checkpoint_path)
    module, _, module_path = module_and_adapter(args.sensor)
    schema = saved_model_schema(args.sensor, saved_args, module_path)
    selection = verify_validation_ap_selection(checkpoint, metrics_path)
    train_info, validation_info, development_events = (
        verify_development_manifests(saved_args, validation_csv)
    )
    mean, std, normalization_provenance = resolve_normalization(
        args.sensor, module, saved_args
    )
    normalization = (mean, std)
    dataset = build_dataset(
        args.sensor, module, saved_args, validation_csv, normalization
    )
    device = torch.device(args.device)
    backbone, head = build_model(
        args.sensor, module, saved_args, checkpoint, device
    )
    targets, probabilities = infer(
        dataset,
        backbone,
        head,
        module,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    threshold, best_f1, threshold_rule = choose_validation_threshold(
        targets, probabilities
    )
    validation_metrics = metrics_at_threshold(targets, probabilities, threshold)
    if not math.isclose(
        validation_metrics["f1"], best_f1, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("Internal threshold/F1 consistency check failed")

    selected_record = selection["selected_epoch_record"]
    logged_threshold = selected_record.get("test_best_f1_threshold")
    logged_comparison = None
    if logged_threshold is not None and math.isfinite(float(logged_threshold)):
        logged_comparison = {
            "logged_threshold": float(logged_threshold),
            "reinferred_minus_logged": threshold - float(logged_threshold),
        }

    artifact = {
        "protocol": PROTOCOL,
        "artifact_type": "frozen_validation_threshold",
        "created_utc": utc_now(),
        "sensor": args.sensor,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint.get("epoch", -1)),
        },
        "checkpoint_selection": selection,
        "model_schema": schema,
        "normalization": {
            "mean": mean,
            "std": std,
            "provenance": normalization_provenance,
        },
        "training_manifest": train_info,
        "validation_manifest": validation_info,
        "development_events": {
            "canonical_event_rule": CANONICAL_EVENT_RULE,
            "count": len(development_events),
            "sha256": json_fingerprint(development_events),
            "values": development_events,
        },
        "threshold": threshold,
        "threshold_selection": threshold_rule,
        "logged_threshold_comparison": logged_comparison,
        "validation_metrics": validation_metrics,
        "runtime": runtime_info(device),
    }
    atomic_write_json(output_artifact, artifact)
    if args.predictions_csv is not None:
        atomic_write_predictions(
            args.predictions_csv.resolve(),
            validation_csv,
            probabilities,
            threshold,
        )
    print(json.dumps(artifact, indent=2), flush=True)


def load_and_verify_artifact(
    path: Path,
    sensor: str,
    checkpoint_path: Path,
    checkpoint: dict,
    schema: dict,
) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("protocol") != PROTOCOL:
        raise ValueError(
            f"Unsupported threshold artifact protocol: {artifact.get('protocol')}"
        )
    if artifact.get("artifact_type") != "frozen_validation_threshold":
        raise ValueError("Artifact is not a frozen validation threshold")
    if artifact.get("sensor") != sensor:
        raise ValueError(
            f"Artifact sensor={artifact.get('sensor')} != requested sensor={sensor}"
        )
    expected_checkpoint_sha = artifact["checkpoint"]["sha256"]
    actual_checkpoint_sha = sha256_file(checkpoint_path)
    if actual_checkpoint_sha != expected_checkpoint_sha:
        raise ValueError("Checkpoint SHA-256 does not match frozen artifact")
    if int(checkpoint.get("epoch", -1)) != int(artifact["checkpoint"]["epoch"]):
        raise ValueError("Checkpoint epoch does not match frozen artifact")
    if schema["fingerprint"] != artifact["model_schema"]["fingerprint"]:
        raise ValueError("Current checkpoint/input schema does not match artifact")
    if (
        schema["trainer_code_sha256"]
        != artifact["model_schema"]["trainer_code_sha256"]
    ):
        raise ValueError("Trainer/evaluator model code changed after threshold freeze")
    threshold = float(artifact["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Frozen threshold is non-finite")
    return artifact


def evaluate_sealed(args) -> None:
    checkpoint_path = args.checkpoint.resolve()
    eval_csv = args.eval_csv.resolve()
    artifact_path = args.threshold_artifact.resolve()
    output_json = args.output_json.resolve()
    requested_outputs = [output_json]
    if args.predictions_csv is not None:
        requested_outputs.append(args.predictions_csv.resolve())
    if len(set(requested_outputs)) != len(requested_outputs):
        raise ValueError("Result JSON and predictions must use different paths")
    existing = [path for path in requested_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to open sealed data because output already exists: {existing}"
        )
    checkpoint, saved_args = load_checkpoint(checkpoint_path)
    module, _, module_path = module_and_adapter(args.sensor)
    schema = saved_model_schema(args.sensor, saved_args, module_path)
    artifact = load_and_verify_artifact(
        artifact_path, args.sensor, checkpoint_path, checkpoint, schema
    )

    eval_info = manifest_info(eval_csv, include_events=True)
    eval_events = set(eval_info.pop("canonical_events"))
    development_events = set(artifact["development_events"]["values"])
    overlap = eval_events & development_events
    if overlap:
        examples = sorted(overlap)[:10]
        raise ValueError(
            f"Sealed manifest overlaps development data in {len(overlap)} "
            f"canonical events; examples={examples}"
        )
    if eval_info["sha256"] == artifact["validation_manifest"]["sha256"]:
        raise ValueError("Sealed manifest is byte-identical to validation manifest")

    mean = [float(value) for value in artifact["normalization"]["mean"]]
    std = [float(value) for value in artifact["normalization"]["std"]]
    normalization_training_sha = artifact["normalization"]["provenance"][
        "training_manifest_sha256"
    ]
    if normalization_training_sha != artifact["training_manifest"]["sha256"]:
        raise ValueError(
            "Normalization provenance is not bound to the frozen training manifest"
        )
    dataset = build_dataset(
        args.sensor, module, saved_args, eval_csv, (mean, std)
    )
    device = torch.device(args.device)
    backbone, head = build_model(
        args.sensor, module, saved_args, checkpoint, device
    )
    targets, probabilities = infer(
        dataset,
        backbone,
        head,
        module,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    threshold = float(artifact["threshold"])
    result = {
        "protocol": PROTOCOL,
        "artifact_type": "sealed_test_result",
        "created_utc": utc_now(),
        "sensor": args.sensor,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": artifact["checkpoint"]["sha256"],
            "epoch": int(checkpoint.get("epoch", -1)),
        },
        "checkpoint_selection": artifact["checkpoint_selection"],
        "model_schema_fingerprint": schema["fingerprint"],
        "threshold": threshold,
        "threshold_source": {
            "artifact_path": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "validation_manifest_sha256": artifact["validation_manifest"]["sha256"],
            "selection_rule": artifact["threshold_selection"],
        },
        "sealed_manifest": eval_info,
        "development_event_overlap": 0,
        "metrics": metrics_at_threshold(targets, probabilities, threshold),
        "runtime": runtime_info(device),
    }
    if args.predictions_csv is not None:
        atomic_write_predictions(
            args.predictions_csv.resolve(), eval_csv, probabilities, threshold
        )
        result["predictions_csv"] = str(args.predictions_csv.resolve())
    atomic_write_json(output_json, result)
    print(json.dumps(result, indent=2), flush=True)


def smoke(args) -> None:
    checkpoint_path = args.checkpoint.resolve()
    validation_csv = args.validation_csv.resolve()
    metrics_path = (
        args.metrics_history.resolve()
        if args.metrics_history is not None
        else checkpoint_path.parent / "metrics_history.json"
    )
    checkpoint, saved_args = load_checkpoint(checkpoint_path)
    module, _, module_path = module_and_adapter(args.sensor)
    schema = saved_model_schema(args.sensor, saved_args, module_path)
    selection = verify_validation_ap_selection(checkpoint, metrics_path)
    train_info, validation_info, development_events = (
        verify_development_manifests(saved_args, validation_csv)
    )
    mean, std, provenance = resolve_normalization(args.sensor, module, saved_args)
    dataset = build_dataset(
        args.sensor, module, saved_args, validation_csv, (mean, std)
    )
    row_count = min(max(1, args.rows), len(dataset))
    if row_count == 1:
        indices = [0]
    else:
        indices = np.linspace(0, len(dataset) - 1, row_count, dtype=int).tolist()
    rows = []
    for index in indices:
        x_dict, label = dataset[index]
        imgs = x_dict["imgs"]
        if not torch.isfinite(imgs).all():
            raise ValueError(f"Smoke row {index} contains non-finite model input")
        rows.append(
            {
                "index": int(index),
                "label": int(label),
                "imgs_shape": list(imgs.shape),
                "chn_ids_shape": list(x_dict["chn_ids"].shape),
                "timestamps_shape": list(x_dict["timestamps"].shape),
                "imgs_min": float(imgs.min()),
                "imgs_max": float(imgs.max()),
            }
        )
    model_check = "not_requested"
    if args.load_model:
        device = torch.device(args.device)
        backbone, head = build_model(
            args.sensor, module, saved_args, checkpoint, device
        )
        model_check = {
            "device": str(device),
            "backbone_parameters": int(
                sum(parameter.numel() for parameter in backbone.parameters())
            ),
            "head_parameters": int(
                sum(parameter.numel() for parameter in head.parameters())
            ),
        }
    result = {
        "protocol": PROTOCOL,
        "artifact_type": "setup_smoke_only_not_a_threshold",
        "sensor": args.sensor,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "selection": selection,
        "model_schema": schema,
        "training_manifest": train_info,
        "validation_manifest": validation_info,
        "development_event_count": len(development_events),
        "normalization": {
            "mean": mean,
            "std": std,
            "provenance": provenance,
        },
        "rows": rows,
        "model_load": model_check,
    }
    print(json.dumps(result, indent=2), flush=True)


def add_checkpoint_and_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sensor", choices=sorted(SENSORS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--metrics_history",
        type=Path,
        help="Defaults to metrics_history.json beside the selected checkpoint.",
    )


def add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--prefetch_factor", type=int, default=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-phase validation-threshold freezing and sealed evaluation for "
            "L89/EMIT32/S5P Panopticon checkpoints."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser(
        "freeze-threshold",
        help="Freeze a threshold from the full validation set; does not open test.",
    )
    add_checkpoint_and_selection_arguments(freeze_parser)
    add_inference_arguments(freeze_parser)
    freeze_parser.add_argument("--validation_csv", type=Path, required=True)
    freeze_parser.add_argument("--output_artifact", type=Path, required=True)
    freeze_parser.add_argument("--predictions_csv", type=Path)
    freeze_parser.set_defaults(func=freeze_threshold)

    eval_parser = subparsers.add_parser(
        "evaluate-sealed",
        help="Evaluate once with a frozen artifact; no threshold-search option exists.",
    )
    eval_parser.add_argument("--sensor", choices=sorted(SENSORS), required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    add_inference_arguments(eval_parser)
    eval_parser.add_argument("--threshold_artifact", type=Path, required=True)
    eval_parser.add_argument("--eval_csv", type=Path, required=True)
    eval_parser.add_argument("--output_json", type=Path, required=True)
    eval_parser.add_argument("--predictions_csv", type=Path)
    eval_parser.set_defaults(func=evaluate_sealed)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Read a few validation rows and optionally load model state; writes nothing.",
    )
    add_checkpoint_and_selection_arguments(smoke_parser)
    smoke_parser.add_argument("--validation_csv", type=Path, required=True)
    smoke_parser.add_argument("--rows", type=int, default=3)
    smoke_parser.add_argument("--load_model", action="store_true")
    smoke_parser.add_argument("--device", default="cpu")
    smoke_parser.set_defaults(func=smoke)

    args = parser.parse_args()
    for name in ("batch_size", "num_workers", "prefetch_factor", "rows"):
        if hasattr(args, name) and getattr(args, name) < (0 if name == "num_workers" else 1):
            parser.error(f"--{name} has an invalid value: {getattr(args, name)}")
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
