#!/usr/bin/env python3
"""Balanced four-sensor residual training on the event-held-out manifests.

This is intentionally a small runner, not a replacement for the existing
sensor-specific Panopticon programs.  It answers one focused question: can one
shared temporal encoder learn a sensor-independent plume-change representation
while retaining sensor-specific input stems and classification heads?

The default representation contains three streams for each sensor:

    current = normalize(t0)
    recent = normalize(t0) - normalize(prev1)
    seasonal = normalize(t0) - normalize(seasonal)

The encoder is the Residual-MAE implementation in ``/home/yuyao/NormWear``.
Its patch stems are sensor specific while every transformer block is shared.
Only one sensor is required at inference time; missing sensors are therefore a
normal execution path rather than an imputed input.

Examples
--------
Short supervised screen after all four caches are staged::

    python research/pretraining_20260727/multisensor_residual_runner.py \
      --mode supervised --device cuda:0 --epochs 3 --batch-size 64 \
      --output-dir /diniuvol/yuyao/methanefuse_research_20260727/shared_residual/e3

Tiny CPU integration smoke (one batch per sensor)::

    python research/pretraining_20260727/multisensor_residual_runner.py \
      --device cpu --epochs 1 --batch-size 2 --stats-samples 2 \
      --image-size 28 --patch-size 14 --embed-dim 32 --depth 1 \
      --num-heads 4 --max-train-rounds 1 --max-val-batches 1 \
      --num-workers 0 --no-require-staged \
      --output-dir /tmp/multisensor_residual_smoke
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


NORMWEAR_ROOT = Path("/home/yuyao/NormWear")
if str(NORMWEAR_ROOT) not in sys.path:
    # Insert at the front because both projects contain a top-level ``modules``
    # package.  The relative imports inside NormWear require its package.
    sys.path.insert(0, str(NORMWEAR_ROOT))

from modules.methane_residual_mae import MethaneResidualMAE  # noqa: E402

if __package__:
    from .methane_residual_mae_validity import (  # noqa: E402
        VALIDITY_OBJECTIVE_VERSION,
        VALIDITY_SEMANTICS,
        ValidityMaskedMethaneResidualMAE,
    )
else:
    # ``python research/.../multisensor_residual_runner.py`` places only this
    # file's directory on sys.path.  Import the sibling adapter directly so
    # the documented launch form works without requiring PYTHONPATH.
    LOCAL_MODULE_ROOT = Path(__file__).resolve().parent
    if str(LOCAL_MODULE_ROOT) not in sys.path:
        sys.path.insert(1, str(LOCAL_MODULE_ROOT))
    from methane_residual_mae_validity import (  # type: ignore[no-redef]  # noqa: E402
        VALIDITY_OBJECTIVE_VERSION,
        VALIDITY_SEMANTICS,
        ValidityMaskedMethaneResidualMAE,
    )


DEFAULT_WORK_ROOT = Path("/diniuvol/yuyao/methanefuse_research_20260727")
SENSOR_ORDER = ("s2", "l89", "emit", "s5p")
STREAM_SUFFIXES = ("current", "recent", "seasonal")
STAGED_DIRECTORY_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "s2": ("s2",),
    "l89": ("l89", "l89_3time"),
    "emit": ("emit", "emit_3time"),
    "s5p": ("s5p",),
}
CANONICAL_EVENT_RULE = "s2-event_group-else-strip-final-hyphen-suffix-v1"
EVENT_SUFFIX_RE = re.compile(r"-[A-Za-z0-9]+$")
PROTOCOL_SCHEMA_VERSION = 2
RANKNET_OBJECTIVE_VERSION = (
    "sensorwise_weighted_bce_plus_all_pairs_ranknet_v1"
)
RANKNET_FIXED_WEIGHT = 0.5
RANKNET_FIXED_TEMPERATURE = 1.0
RANKNET_VALIDATION_ARTIFACT_TYPE = "ranknet_validation_evaluation_v1"
RANKNET_FULL_INPUT_CONTRACT = "full_current_recent_seasonal_v1"
RANKNET_PREDICTION_FIELDS = (
    "sensor",
    "row_index",
    "label",
    "probability",
)


@dataclass(frozen=True)
class SensorSpec:
    name: str
    kind: str
    channels: int
    current_column: str
    recent_column: Optional[str]
    seasonal_column: Optional[str]
    image_column: Optional[str] = None
    npz_key: Optional[str] = None
    npz_current_index: int = 0
    npz_recent_index: int = 1
    npz_seasonal_index: int = 4


SENSOR_SPECS: Mapping[str, SensorSpec] = {
    "s2": SensorSpec("s2", "tiff", 12, "path_t0", "path_prev1", "path_seasonal"),
    "l89": SensorSpec("l89", "tiff", 10, "path_t0", "path_prev1", "path_seasonal"),
    "emit": SensorSpec("emit", "tiff", 32, "path_t0", "path_prev1", "path_seasonal"),
    "s5p": SensorSpec(
        "s5p",
        "npz",
        1,
        "t0",
        "prev1",
        "seasonal",
        image_column="image_path",
        npz_key="ch4",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensor-balanced shared Residual-MAE training for methane classification."
    )
    parser.add_argument("--mode", choices=("supervised", "pretrain"), default="supervised")
    parser.add_argument(
        "--sharing",
        choices=("shared", "independent"),
        default="shared",
        help=(
            "shared: sensor-specific patch stems plus one shared transformer; "
            "independent: one complete Residual-MAE per sensor as the control."
        ),
    )
    parser.add_argument("--sensors", default=",".join(SENSOR_ORDER))
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=DEFAULT_WORK_ROOT / "manifests_staged",
        help="Preferred staged layout: ROOT/{sensor}/{train,val}.csv.",
    )
    parser.add_argument(
        "--source-manifest-root",
        type=Path,
        default=DEFAULT_WORK_ROOT / "manifests",
        help="Fallback cap8/recent manifests when a staged manifest is not present.",
    )
    parser.add_argument(
        "--require-staged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refuse an implicit fallback to remote paths. Disable only for a "
            "small diagnostic or pass explicit CSV overrides."
        ),
    )
    parser.add_argument(
        "--cross-sensor-event-policy",
        choices=("error", "purge"),
        default="error",
        help=(
            "For shared models, audit the union of every sensor's train events "
            "against the union of every validation set. 'error' refuses any "
            "overlap; 'purge' deterministically removes all validation events "
            "from every training manifest before stats or training."
        ),
    )
    parser.add_argument(
        "--protocol-manifest-dir",
        type=Path,
        default=None,
        help=(
            "Directory for deterministic event-purged train manifests. Use one "
            "common directory for MAE, pretrained finetune, and matched scratch."
        ),
    )
    for sensor in SENSOR_ORDER:
        parser.add_argument(f"--{sensor}-train-csv", type=Path, default=None)
        parser.add_argument(f"--{sensor}-val-csv", type=Path, default=None)

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORK_ROOT / "shared_residual")
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--stats-samples", type=int, default=128)
    parser.add_argument("--stats-workers", type=int, default=4)
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--s5p-data-key", default="ch4")
    parser.add_argument("--residual-clip", type=float, default=5.0)
    parser.add_argument("--current-clip", type=float, default=8.0)
    parser.add_argument(
        "--resize-on-device",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resize collated batches on the training device instead of in each "
            "DataLoader sample; this reduces CPU pressure for 224x224 caches."
        ),
    )

    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--fuse-freq", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument(
        "--validity-masked-reconstruction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use per-channel native validity and reconstruct only valid "
            "masked elements. Disabled by default for exact compatibility "
            "with legacy runs; pass the flag to both masked pretraining and "
            "its supervised transfer run."
        ),
    )
    parser.add_argument("--decoder-embed-dim", type=int, default=128)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-num-heads", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--steps-per-sensor",
        type=int,
        default=0,
        help="Balanced batches per sensor per epoch; 0 uses the shortest loader.",
    )
    parser.add_argument(
        "--max-train-rounds",
        type=int,
        default=0,
        help="Smoke/debug cap applied after --steps-per-sensor; 0 disables it.",
    )
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--log-interval-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-pos-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ranknet-objective",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add the preregistered within-sensor all-positive/negative-pairs "
            "RankNet term to weighted BCE. This supervised fallback is "
            "authorized only after the validity-masked MAE gate fails."
        ),
    )
    parser.add_argument(
        "--ranknet-weight",
        type=float,
        default=RANKNET_FIXED_WEIGHT,
        help=(
            "Fixed RankNet weight. The authorized fallback requires exactly "
            f"{RANKNET_FIXED_WEIGHT}; other values are rejected."
        ),
    )
    parser.add_argument(
        "--ranknet-temperature",
        type=float,
        default=RANKNET_FIXED_TEMPERATURE,
        help=(
            "Fixed RankNet temperature. The authorized fallback requires "
            f"exactly {RANKNET_FIXED_TEMPERATURE}; other values are rejected."
        ),
    )
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize a supervised run from a pretrain checkpoint. Only the "
            "complete encoder is loaded; MAE decoder and classification heads "
            "are explicitly excluded."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Resume the same interrupted run, including model and optimizer. "
            "This is not transfer initialization."
        ),
    )
    parser.add_argument(
        "--resume-reason",
        default="",
        help="Audit note for why a stopped run is being resumed.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a CPU-only checkpoint-transfer self-test without reading manifests.",
    )
    return parser.parse_args()


def parse_sensors(value: str) -> Tuple[str, ...]:
    sensors = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not sensors:
        raise ValueError("--sensors must contain at least one sensor.")
    unknown = sorted(set(sensors) - set(SENSOR_ORDER))
    if unknown:
        raise ValueError(f"Unknown sensors: {unknown}; expected a subset of {SENSOR_ORDER}.")
    if len(set(sensors)) != len(sensors):
        raise ValueError(f"Duplicate sensors in --sensors: {sensors}")
    return sensors


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def atomic_text_dump(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(file_descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def artifact_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Keep disabled RankNet fields out of legacy run artifacts."""

    config = dict(vars(args))
    if not bool(config.get("ranknet_objective", False)):
        for key in (
            "ranknet_objective",
            "ranknet_weight",
            "ranknet_temperature",
        ):
            config.pop(key, None)
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_dict_fingerprint(
    state: Mapping[str, torch.Tensor], *, prefix: str
) -> str:
    """Hash named tensor bytes, used to prove transfer did not touch heads."""

    digest = hashlib.sha256()
    matched = 0
    for name in sorted(state):
        if not name.startswith(prefix):
            continue
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        matched += 1
    if matched == 0:
        raise ValueError(f"No state-dict keys start with {prefix!r}.")
    return digest.hexdigest()


def write_dataframe_once_or_verify(frame: pd.DataFrame, path: Path) -> str:
    """Materialize a deterministic protocol CSV without silently replacing it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(file_descriptor)
    try:
        frame.to_csv(temporary_name, index=False, lineterminator="\n")
        candidate_sha = sha256_file(Path(temporary_name))
        if path.exists():
            existing_sha = sha256_file(path)
            if existing_sha != candidate_sha:
                raise FileExistsError(
                    f"Protocol manifest exists with different content: {path}; "
                    f"existing_sha256={existing_sha}, candidate_sha256={candidate_sha}. "
                    "Use a new --protocol-manifest-dir for the changed protocol."
                )
            os.unlink(temporary_name)
            return existing_sha
        os.replace(temporary_name, path)
        return candidate_sha
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


EPOCH_TABLE_FIELDS = (
    "epoch",
    "sensor",
    "mode",
    "sharing",
    "train_loss",
    "val_loss",
    "reconstruction_loss",
    "samples",
    "positive_rate",
    "ap",
    "f1",
    "f1_0p5",
    "auroc",
    "threshold",
    "elapsed_seconds",
    "completed_unix",
)


def epoch_table_rows(history: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mode = history.get("mode")
    sharing = history.get("sharing")
    for epoch_record in history.get("epochs", []):
        epoch = epoch_record.get("epoch")
        train_losses = epoch_record.get("train_loss", {})
        val_by_sensor = epoch_record.get("val", {})
        for sensor, metrics in val_by_sensor.items():
            rows.append(
                {
                    "epoch": epoch,
                    "sensor": sensor,
                    "mode": mode,
                    "sharing": sharing,
                    "train_loss": train_losses.get(sensor),
                    "val_loss": metrics.get("loss"),
                    "reconstruction_loss": metrics.get("reconstruction_loss"),
                    "samples": metrics.get("samples"),
                    "positive_rate": metrics.get("positive_rate"),
                    "ap": metrics.get("ap"),
                    "f1": metrics.get("f1"),
                    "f1_0p5": metrics.get("f1_0p5"),
                    "auroc": metrics.get("auroc"),
                    "threshold": metrics.get("threshold"),
                    "elapsed_seconds": epoch_record.get("elapsed_seconds"),
                    "completed_unix": epoch_record.get("completed_unix"),
                }
            )
        macro = epoch_record.get("macro_over_sensor", {})
        rows.append(
            {
                "epoch": epoch,
                "sensor": "__macro__",
                "mode": mode,
                "sharing": sharing,
                "train_loss": (
                    float(np.mean(list(train_losses.values()))) if train_losses else None
                ),
                "val_loss": macro.get("loss"),
                "reconstruction_loss": macro.get("reconstruction_loss"),
                "samples": None,
                "positive_rate": None,
                "ap": macro.get("ap"),
                "f1": macro.get("f1"),
                "f1_0p5": macro.get("f1_0p5"),
                "auroc": macro.get("auroc"),
                "threshold": None,
                "elapsed_seconds": epoch_record.get("elapsed_seconds"),
                "completed_unix": epoch_record.get("completed_unix"),
            }
        )
    return json_safe(rows)


def write_epoch_tables(history: Mapping[str, Any], output_dir: Path) -> None:
    """Atomically rebuild compact JSONL and CSV views after every epoch."""

    rows = epoch_table_rows(history)
    jsonl = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )
    atomic_text_dump(jsonl, output_dir / "metrics_epochs.jsonl")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EPOCH_TABLE_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text_dump(buffer.getvalue(), output_dir / "metrics_epochs.csv")


def write_ranknet_full_validation_artifacts(
    *,
    history: Mapping[str, Any],
    history_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    epoch: int,
    per_sensor: Mapping[str, Mapping[str, Any]],
    macro_over_sensor: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> Dict[str, Any]:
    """Sign predictions captured by the existing RankNet validation pass."""

    expected_objective = build_ranknet_objective_signature()
    if history.get("supervised_objective") != expected_objective:
        raise ValueError(
            "Refusing to write RankNet predictions for a different objective."
        )
    if history.get("status") != "completed":
        raise ValueError(
            "Full RankNet validation artifacts require completed history."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"RankNet validation artifact requires checkpoint: "
            f"{checkpoint_path}"
        )
    output_csv = output_dir / "validation_full_temporal.csv"
    output_json = output_dir / "validation_full_temporal.json"
    existing = [
        path for path in (output_csv, output_json) if path.exists()
    ]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite RankNet validation artifacts: {existing}"
        )

    full_sample_counts = {
        sensor: int(per_sensor[sensor]["samples"])
        for sensor in history["sensors"]
    }
    if len(prediction_rows) != sum(full_sample_counts.values()):
        raise ValueError(
            "RankNet prediction-row count does not equal full validation "
            f"counts: rows={len(prediction_rows)}, "
            f"counts={full_sample_counts}"
        )
    for sensor in history["sensors"]:
        sensor_rows = [
            row for row in prediction_rows if row["sensor"] == sensor
        ]
        observed_indices = [int(row["row_index"]) for row in sensor_rows]
        expected_indices = list(range(full_sample_counts[sensor]))
        if observed_indices != expected_indices:
            raise ValueError(
                f"{sensor} RankNet prediction row_index is not a unique "
                "contiguous validation ordering."
            )

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=RANKNET_PREDICTION_FIELDS,
    )
    writer.writeheader()
    writer.writerows(prediction_rows)
    atomic_text_dump(buffer.getvalue(), output_csv)
    prediction_csv_sha256 = sha256_file(output_csv)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    history_sha256 = sha256_file(history_path)
    result = {
        "schema_version": 1,
        "artifact_type": RANKNET_VALIDATION_ARTIFACT_TYPE,
        "status": "completed",
        "evaluation_mode": "full_temporal",
        "input_contract": RANKNET_FULL_INPUT_CONTRACT,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(epoch),
        "source_history_path": str(history_path.resolve()),
        "source_history_sha256": history_sha256,
        "supervised_objective": expected_objective,
        "encoder_signature": history["encoder_signature"],
        "data_signature": history["data_signature"],
        "resume_signature": history["resume_signature"],
        "encoder_signature_fingerprint": json_fingerprint(
            history["encoder_signature"]
        ),
        "data_signature_fingerprint": json_fingerprint(
            history["data_signature"]
        ),
        "resume_signature_fingerprint": json_fingerprint(
            history["resume_signature"]
        ),
        "event_protocol_fingerprint": history["data_signature"][
            "event_protocol_fingerprint"
        ],
        "normalization_stats_path": str(
            Path(str(history["normalization_stats"])).resolve()
        ),
        "normalization_stats_sha256": history["data_signature"][
            "normalization_stats_sha256"
        ],
        "predictions_csv": str(output_csv.resolve()),
        "predictions_csv_sha256": prediction_csv_sha256,
        "prediction_rows": len(prediction_rows),
        "full_sample_counts": full_sample_counts,
        "per_sensor": per_sensor,
        "macro_over_sensor": macro_over_sensor,
        "runtime": runtime,
        "completed_unix": time.time(),
    }
    atomic_json_dump(json_safe(result), output_json)
    return {
        "json_path": str(output_json),
        "csv_path": str(output_csv),
        "checkpoint_sha256": checkpoint_sha256,
        "predictions_csv_sha256": prediction_csv_sha256,
        "prediction_rows": len(prediction_rows),
    }


def resolve_csvs(
    args: argparse.Namespace, sensors: Sequence[str]
) -> Dict[str, Dict[str, Path]]:
    resolved: Dict[str, Dict[str, Path]] = {}
    for sensor in sensors:
        resolved[sensor] = {}
        for split in ("train", "val"):
            override = getattr(args, f"{sensor}_{split}_csv")
            staged_candidates = [
                args.manifest_root / directory / f"{split}.csv"
                for directory in STAGED_DIRECTORY_ALIASES[sensor]
            ]
            staged = next(
                (candidate for candidate in staged_candidates if candidate.is_file()),
                staged_candidates[0],
            )
            fallback_suffix = "train_core_cap8" if split == "train" else "val_recent"
            fallback = args.source_manifest_root / f"{sensor}_{fallback_suffix}.csv"
            if override is not None:
                path = override
            elif staged.is_file():
                path = staged
            elif args.require_staged:
                raise FileNotFoundError(
                    f"Staged {split} manifest for {sensor} is not ready; checked "
                    f"{staged_candidates}. "
                    "Finish local staging, pass an explicit CSV, or use "
                    "--no-require-staged for a small diagnostic only."
                )
            else:
                path = fallback
                print(
                    f"[Data] WARNING: {sensor}/{split} uses fallback manifest "
                    f"{fallback}; its paths may be remote.",
                    flush=True,
                )
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"No {split} manifest for {sensor}. Checked explicit={override}, "
                    f"staged={staged}, fallback={fallback}."
                )
            resolved[sensor][split] = path.resolve()
    return resolved


def canonical_event_series(
    sensor: str, frame: pd.DataFrame, path: Path
) -> pd.Series:
    """Map legacy sensor manifests onto the shared Carbon Mapper event."""

    if sensor == "s2" and "event_group_id" in frame.columns:
        values = frame["event_group_id"].astype("string").str.strip()
    else:
        if "plume_id" not in frame.columns:
            raise ValueError(
                f"{path} has no plume_id column; global event isolation cannot run."
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
            f"{path} has {int(invalid.sum())} empty canonical event identifiers."
        )
    return values.astype(str)


def _validate_manifest_labels(frame: pd.DataFrame, path: Path) -> None:
    if "label" not in frame.columns:
        raise ValueError(f"{path} has no label column.")
    numeric = pd.to_numeric(frame["label"], errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{path} has {int(numeric.isna().sum())} nonnumeric labels.")
    observed = set(numeric.unique().tolist())
    if not observed.issubset({0, 1}):
        raise ValueError(f"{path} has non-binary labels: {sorted(observed)}")


def enforce_event_protocol(
    args: argparse.Namespace,
    sensors: Sequence[str],
    csvs: Mapping[str, Mapping[str, Path]],
) -> Tuple[Dict[str, Dict[str, Path]], Dict[str, Any]]:
    """Audit event isolation and optionally purge global validation events.

    A shared encoder makes every sensor's training rows part of one training
    set.  Therefore a validation event seen through *another* sensor is still
    leakage even when each legacy sensor split is internally disjoint.
    """

    frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    events: Dict[str, Dict[str, pd.Series]] = {}
    per_sensor: Dict[str, Dict[str, Any]] = {}
    for sensor in sensors:
        frames[sensor] = {}
        events[sensor] = {}
        per_sensor[sensor] = {}
        for split in ("train", "val"):
            path = Path(csvs[sensor][split])
            frame = pd.read_csv(path, low_memory=False)
            if frame.empty:
                raise ValueError(f"Empty {split} manifest: {path}")
            _validate_manifest_labels(frame, path)
            event_values = canonical_event_series(sensor, frame, path)
            frames[sensor][split] = frame
            events[sensor][split] = event_values
            per_sensor[sensor][split] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": int(len(frame)),
                "canonical_events": int(event_values.nunique()),
                "positive_rows": int(pd.to_numeric(frame["label"]).sum()),
            }
        within = set(events[sensor]["train"]) & set(events[sensor]["val"])
        per_sensor[sensor]["within_sensor_train_val_overlap"] = len(within)
        if within:
            raise ValueError(
                f"{sensor} train/val has {len(within)} canonical event overlaps; "
                f"examples={sorted(within)[:10]}"
            )

    validation_union = set().union(
        *(set(events[sensor]["val"]) for sensor in sensors)
    )
    training_union = set().union(
        *(set(events[sensor]["train"]) for sensor in sensors)
    )
    global_overlap = training_union & validation_union
    cross_pairs: Dict[str, int] = {}
    for train_sensor in sensors:
        for val_sensor in sensors:
            if train_sensor == val_sensor:
                continue
            count = len(
                set(events[train_sensor]["train"])
                & set(events[val_sensor]["val"])
            )
            if count:
                cross_pairs[f"{train_sensor}_train->{val_sensor}_val"] = count

    audit: Dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "canonical_event_rule": CANONICAL_EVENT_RULE,
        "sharing": args.sharing,
        "requested_policy": args.cross_sensor_event_policy,
        "global_validation_events": len(validation_union),
        "global_training_events_before": len(training_union),
        "global_train_val_overlap_before": len(global_overlap),
        "global_train_val_overlap_examples": sorted(global_overlap)[:20],
        "cross_sensor_overlap_pairs": cross_pairs,
        "sensors": per_sensor,
    }
    updated = {
        sensor: {split: Path(csvs[sensor][split]) for split in ("train", "val")}
        for sensor in sensors
    }

    if args.sharing == "shared" and global_overlap:
        if args.cross_sensor_event_policy == "error":
            raise ValueError(
                "Shared-encoder protocol leaks canonical events across sensors: "
                f"union(train) intersects union(val) by {len(global_overlap)} events; "
                f"pairs={cross_pairs}; examples={sorted(global_overlap)[:10]}. "
                "Use --cross-sensor-event-policy purge with one common "
                "--protocol-manifest-dir for pretrain, finetune, and scratch."
            )
        protocol_dir = (
            Path(args.protocol_manifest_dir)
            if args.protocol_manifest_dir is not None
            else Path(args.output_dir) / "protocol_manifests"
        )
        for sensor in sensors:
            keep = ~events[sensor]["train"].isin(validation_union)
            purged = frames[sensor]["train"].loc[keep].copy()
            if purged.empty:
                raise ValueError(f"Global event purge emptied {sensor} training data.")
            remaining_labels = set(
                pd.to_numeric(purged["label"]).astype(int).unique().tolist()
            )
            if remaining_labels != {0, 1}:
                raise ValueError(
                    f"Global event purge leaves {sensor} labels={sorted(remaining_labels)}; "
                    "classification would not be a valid matched comparison."
                )
            output_path = (protocol_dir / f"{sensor}_train_global_val_purged.csv").resolve()
            output_sha = write_dataframe_once_or_verify(purged, output_path)
            remaining_events = set(canonical_event_series(sensor, purged, output_path))
            post_overlap = remaining_events & validation_union
            if post_overlap:
                raise AssertionError(
                    f"Internal error: {sensor} still overlaps global validation."
                )
            updated[sensor]["train"] = output_path
            audit["sensors"][sensor]["purged_train"] = {
                "path": str(output_path),
                "sha256": output_sha,
                "rows": int(len(purged)),
                "canonical_events": len(remaining_events),
                "removed_rows": int((~keep).sum()),
                "removed_events": len(
                    set(events[sensor]["train"]) & validation_union
                ),
                "positive_rows": int(pd.to_numeric(purged["label"]).sum()),
            }
        audit["purge_applied"] = True
    else:
        # Independent models share no parameters, so cross-sensor overlap is
        # reported but does not leak information between their evaluations.
        audit["purge_applied"] = False

    post_training_union: set[str] = set()
    for sensor in sensors:
        if audit["purge_applied"]:
            post_frame = pd.read_csv(updated[sensor]["train"], low_memory=False)
            post_training_union.update(
                canonical_event_series(sensor, post_frame, updated[sensor]["train"])
            )
        else:
            post_training_union.update(events[sensor]["train"])
    post_overlap = post_training_union & validation_union
    audit["global_training_events_after"] = len(post_training_union)
    audit["global_train_val_overlap_after"] = len(post_overlap)
    if args.sharing == "shared" and post_overlap:
        raise AssertionError(
            f"Shared protocol still has {len(post_overlap)} global event overlaps."
        )
    audit["fingerprint"] = json_fingerprint(
        {
            "canonical_event_rule": CANONICAL_EVENT_RULE,
            "manifest_sha256": {
                sensor: {
                    split: sha256_file(updated[sensor][split])
                    for split in ("train", "val")
                }
                for sensor in sensors
            },
            "global_train_val_overlap_after": len(post_overlap),
        }
    )
    print(
        "[Protocol] "
        f"global_train_val_overlap={len(global_overlap)}->{len(post_overlap)} "
        f"purge={audit['purge_applied']} pairs={cross_pairs}",
        flush=True,
    )
    return updated, audit


def _read_tiff_chw(path: str, expected_channels: int) -> np.ndarray:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path)
    array = np.asarray(tifffile.imread(path))
    array = np.squeeze(array)
    if array.ndim == 2 and expected_channels == 1:
        array = array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF at {path}, got {array.shape}.")
    if array.shape[0] == expected_channels:
        pass
    elif array.shape[-1] == expected_channels:
        array = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(
            f"Could not locate {expected_channels} channels in TIFF {path} with shape {array.shape}."
        )
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def _extract_npz_array(path: str, key: str) -> np.ndarray:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            numeric_keys = []
            for candidate in archive.files:
                with contextlib.suppress(ValueError, TypeError):
                    if np.asarray(archive[candidate]).dtype.kind in "biufc":
                        numeric_keys.append(candidate)
            if len(numeric_keys) != 1:
                raise KeyError(
                    f"NPZ {path} has no key {key!r}; numeric candidates are {numeric_keys}."
                )
            key = numeric_keys[0]
        array = np.asarray(archive[key])
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D temporal NPZ array at {path}, got {array.shape}.")
    if array.shape[0] == 6:
        pass
    elif array.shape[-1] == 6:
        array = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"Could not locate six timepoints in NPZ {path} with shape {array.shape}.")
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def load_three_frames(
    row: pd.Series, spec: SensorSpec, s5p_data_key: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw frames ``(3,C,H,W)`` and a same-shaped valid-pixel mask."""

    if spec.kind == "tiff":
        columns = (spec.current_column, spec.recent_column, spec.seasonal_column)
        frames = [_read_tiff_chw(str(row[column]), spec.channels) for column in columns]
        shapes = {frame.shape for frame in frames}
        if len(shapes) != 1:
            raise ValueError(f"{spec.name} timepoint shapes disagree: {sorted(shapes)}")
        raw = np.stack(frames, axis=0)
        # The TIFF pipelines encode no-data as zero.
        valid = np.isfinite(raw) & (raw != 0)
    elif spec.kind == "npz":
        assert spec.image_column is not None
        stack = _extract_npz_array(
            str(row[spec.image_column]), s5p_data_key or spec.npz_key or "ch4"
        )
        indices = (
            spec.npz_current_index,
            spec.npz_recent_index,
            spec.npz_seasonal_index,
        )
        raw = stack[np.asarray(indices, dtype=np.int64), :, :][:, None, :, :]
        valid = np.isfinite(raw)
    else:
        raise ValueError(f"Unsupported sensor kind: {spec.kind}")
    return raw.astype(np.float32, copy=False), valid


def _row_channel_moments(
    row: pd.Series, spec: SensorSpec, s5p_data_key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw, valid = load_three_frames(row, spec, s5p_data_key)
    valid64 = valid.astype(np.float64, copy=False)
    raw64 = np.where(valid, raw, 0.0).astype(np.float64, copy=False)
    sums = (raw64 * valid64).sum(axis=(0, 2, 3))
    squared_sums = (raw64 * raw64 * valid64).sum(axis=(0, 2, 3))
    counts = valid64.sum(axis=(0, 2, 3))
    return sums, squared_sums, counts


def compute_sensor_stats(
    csv_path: Path,
    spec: SensorSpec,
    *,
    sample_count: int,
    workers: int,
    seed: int,
    s5p_data_key: str,
) -> Dict[str, Any]:
    dataframe = pd.read_csv(csv_path, low_memory=False)
    if len(dataframe) == 0:
        raise ValueError(f"Empty training manifest: {csv_path}")
    count = min(max(1, int(sample_count)), len(dataframe))
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(dataframe), size=count, replace=False).tolist())

    sums = np.zeros(spec.channels, dtype=np.float64)
    squared_sums = np.zeros(spec.channels, dtype=np.float64)
    counts = np.zeros(spec.channels, dtype=np.float64)

    def consume(result: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        nonlocal sums, squared_sums, counts
        row_sums, row_squared_sums, row_counts = result
        sums += row_sums
        squared_sums += row_squared_sums
        counts += row_counts

    worker_count = min(max(1, int(workers)), count)
    if worker_count == 1:
        for index in indices:
            consume(_row_channel_moments(dataframe.iloc[index], spec, s5p_data_key))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _row_channel_moments, dataframe.iloc[index], spec, s5p_data_key
                )
                for index in indices
            ]
            for future in as_completed(futures):
                consume(future.result())

    missing = counts <= 0
    means = np.zeros(spec.channels, dtype=np.float64)
    variances = np.ones(spec.channels, dtype=np.float64)
    means[~missing] = sums[~missing] / counts[~missing]
    variances[~missing] = (
        squared_sums[~missing] / counts[~missing] - means[~missing] ** 2
    )
    standard_deviations = np.sqrt(np.maximum(variances, 1e-12))
    standard_deviations[missing] = 1.0
    return {
        "sensor": spec.name,
        "train_csv": str(csv_path),
        "train_csv_sha256": sha256_file(csv_path),
        "sample_seed": int(seed),
        "requested_sample_rows": int(sample_count),
        "s5p_data_key": str(s5p_data_key),
        "sampled_rows": count,
        "frames_per_row": 3,
        "mean": means.astype(np.float32).tolist(),
        "std": standard_deviations.astype(np.float32).tolist(),
        "valid_pixel_count": counts.astype(np.int64).tolist(),
        "zero_is_nodata": spec.kind == "tiff",
    }


def load_or_compute_stats(
    args: argparse.Namespace,
    sensors: Sequence[str],
    csvs: Mapping[str, Mapping[str, Path]],
) -> Tuple[Path, Dict[str, Dict[str, Any]]]:
    stats_path = args.stats_json or (args.output_dir / "normalization_stats.json")
    payload: Dict[str, Any] = {}
    if stats_path.is_file() and not args.recompute_stats:
        with stats_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        payload = loaded.get("sensors", loaded)

    changed = False
    for sensor_index, sensor in enumerate(sensors):
        existing = payload.get(sensor)
        train_sha = sha256_file(csvs[sensor]["train"])
        expected_seed = args.seed + sensor_index * 1009
        if existing is not None:
            compatible = (
                len(existing.get("mean", [])) == SENSOR_SPECS[sensor].channels
                and len(existing.get("std", [])) == SENSOR_SPECS[sensor].channels
                and existing.get("train_csv_sha256") == train_sha
                and int(existing.get("sample_seed", -1)) == expected_seed
                and int(existing.get("requested_sample_rows", -1))
                == int(args.stats_samples)
                and str(existing.get("s5p_data_key", args.s5p_data_key))
                == str(args.s5p_data_key)
            )
            if compatible:
                print(f"[Stats] reuse {sensor}: {stats_path}", flush=True)
                continue
            raise ValueError(
                f"Saved stats for {sensor} are incompatible with the effective "
                f"training manifest or sampling protocol: {stats_path}. "
                "Pass --recompute-stats deliberately or use a new --stats-json."
            )
        print(
            f"[Stats] compute {sensor} from {csvs[sensor]['train']} "
            f"({args.stats_samples} sampled rows)",
            flush=True,
        )
        payload[sensor] = compute_sensor_stats(
            csvs[sensor]["train"],
            SENSOR_SPECS[sensor],
            sample_count=args.stats_samples,
            workers=args.stats_workers,
            seed=expected_seed,
            s5p_data_key=args.s5p_data_key,
        )
        changed = True

    if changed or not stats_path.is_file():
        atomic_json_dump(
            json_safe(
                {
                    "schema_version": 2,
                    "created_unix": time.time(),
                    "representation": "normalized_t0,t0-prev1,t0-seasonal",
                    "sensors": payload,
                }
            ),
            stats_path,
        )
    return stats_path, {sensor: payload[sensor] for sensor in sensors}


def resize_validity_tensor(
    validity: torch.Tensor,
    target_shape: Tuple[int, int],
) -> torch.Tensor:
    """Resize per-channel validity without inventing high-resolution support."""

    if validity.ndim != 4:
        raise ValueError(
            "Validity resize expects a 4D tensor (N,C,H,W), got "
            f"{tuple(validity.shape)}"
        )
    target_height, target_width = (int(target_shape[0]), int(target_shape[1]))
    source_height, source_width = validity.shape[-2:]
    if (source_height, source_width) == (target_height, target_width):
        return validity.float().clamp(0.0, 1.0)

    validity_float = validity.float()
    if target_height <= source_height and target_width <= source_width:
        resized = F.interpolate(
            validity_float,
            size=(target_height, target_width),
            mode="area",
        )
    elif target_height >= source_height and target_width >= source_width:
        resized = F.interpolate(
            validity_float,
            size=(target_height, target_width),
            mode="nearest",
        )
    else:
        raise ValueError(
            "Mixed-direction validity resize is unsupported: "
            f"source={(source_height, source_width)} "
            f"target={(target_height, target_width)}"
        )
    return resized.clamp(0.0, 1.0)


class TemporalResidualDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        spec: SensorSpec,
        stats: Mapping[str, Any],
        *,
        image_size: int,
        current_clip: float,
        residual_clip: float,
        s5p_data_key: str,
        augment: bool,
        resize_on_device: bool,
        return_validity: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.dataframe = pd.read_csv(self.csv_path, low_memory=False)
        if len(self.dataframe) == 0:
            raise ValueError(f"Empty manifest: {self.csv_path}")
        if "label" not in self.dataframe.columns:
            raise ValueError(f"Missing label column in {self.csv_path}")
        self.spec = spec
        self.image_size = int(image_size)
        self.current_clip = float(current_clip)
        self.residual_clip = float(residual_clip)
        self.s5p_data_key = str(s5p_data_key)
        self.augment = bool(augment)
        self.resize_on_device = bool(resize_on_device)
        self.return_validity = bool(return_validity)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, spec.channels, 1, 1)
        self.std = (
            torch.tensor(stats["std"], dtype=torch.float32)
            .clamp_min(1e-6)
            .view(1, spec.channels, 1, 1)
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[Any, torch.Tensor]:
        row = self.dataframe.iloc[index]
        raw_numpy, valid_numpy = load_three_frames(row, self.spec, self.s5p_data_key)
        raw = torch.from_numpy(raw_numpy.copy())
        valid = torch.from_numpy(valid_numpy.copy())
        normalized = (raw - self.mean) / self.std
        normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        current = normalized[0]
        current_valid = valid[0]
        recent_valid = valid[0] & valid[1]
        seasonal_valid = valid[0] & valid[2]
        recent = torch.where(
            recent_valid, normalized[0] - normalized[1], torch.zeros_like(current)
        )
        seasonal = torch.where(
            seasonal_valid, normalized[0] - normalized[2], torch.zeros_like(current)
        )
        current = current.clamp(-self.current_clip, self.current_clip)
        recent = recent.clamp(-self.residual_clip, self.residual_clip)
        seasonal = seasonal.clamp(-self.residual_clip, self.residual_clip)
        three = torch.stack((current, recent, seasonal), dim=0)
        three_validity = None
        if self.return_validity:
            three_validity = torch.stack(
                (current_valid, recent_valid, seasonal_valid),
                dim=0,
            ).float()

        if (
            not self.resize_on_device
            and tuple(three.shape[-2:]) != (self.image_size, self.image_size)
        ):
            three = F.interpolate(
                three,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if three_validity is not None:
                three_validity = resize_validity_tensor(
                    three_validity,
                    (self.image_size, self.image_size),
                )
        if self.augment:
            if torch.rand(()) < 0.5:
                three = torch.flip(three, dims=(-1,))
                if three_validity is not None:
                    three_validity = torch.flip(
                        three_validity,
                        dims=(-1,),
                    )
            if torch.rand(()) < 0.5:
                three = torch.flip(three, dims=(-2,))
                if three_validity is not None:
                    three_validity = torch.flip(
                        three_validity,
                        dims=(-2,),
                    )

        streams = {
            f"{self.spec.name}_{suffix}": three[stream_index]
            for stream_index, suffix in enumerate(STREAM_SUFFIXES)
        }
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        if three_validity is not None:
            validity_by_stream = {
                f"{self.spec.name}_{suffix}": three_validity[stream_index]
                for stream_index, suffix in enumerate(STREAM_SUFFIXES)
            }
            return {
                "values": streams,
                "validity": validity_by_stream,
            }, label
        return streams, label

    @property
    def labels(self) -> np.ndarray:
        return self.dataframe["label"].to_numpy(dtype=np.float32)


class MultiSensorResidualModel(nn.Module):
    def __init__(
        self,
        sensors: Sequence[str],
        *,
        mode: str,
        sharing: str,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        fuse_freq: int,
        dropout: float,
        mask_ratio: float,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        validity_masked_reconstruction: bool = False,
    ) -> None:
        super().__init__()
        self.sensors = tuple(sensors)
        self.mode = mode
        self.sharing = sharing
        self.validity_masked_reconstruction = bool(
            validity_masked_reconstruction
        )

        def make_backbone(backbone_sensors: Sequence[str]) -> MethaneResidualMAE:
            stream_channels = {
                f"{sensor}_{suffix}": SENSOR_SPECS[sensor].channels
                for sensor in backbone_sensors
                for suffix in STREAM_SUFFIXES
            }
            backbone_class = (
                ValidityMaskedMethaneResidualMAE
                if self.validity_masked_reconstruction
                else MethaneResidualMAE
            )
            return backbone_class(
                sensor_channels=stream_channels,
                img_size=(image_size, image_size),
                patch_size=(patch_size, patch_size),
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                decoder_embed_dim=decoder_embed_dim,
                decoder_depth=decoder_depth,
                decoder_num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                fuse_freq=fuse_freq,
                mask_ratio=mask_ratio,
                is_pretrain=mode == "pretrain",
                drop=dropout,
                temporal_fusion_mode="current_query",
            )

        if sharing == "shared":
            self.backbone = make_backbone(self.sensors)
            self.backbones = None
        elif sharing == "independent":
            self.backbone = None
            self.backbones = nn.ModuleDict(
                {sensor: make_backbone((sensor,)) for sensor in self.sensors}
            )
        else:
            raise ValueError(f"Unsupported sharing mode: {sharing}")
        # A pretraining checkpoint must not contain meaningless, random
        # classification heads.  The transfer loader also excludes heads from
        # older checkpoints created before this invariant was introduced.
        self.heads = nn.ModuleDict()
        if mode == "supervised":
            self.heads.update(
                {
                    sensor: nn.Sequential(
                        nn.LayerNorm(embed_dim),
                        nn.Dropout(dropout),
                        nn.Linear(embed_dim, 1),
                    )
                    for sensor in self.sensors
                }
            )

    def forward(
        self,
        sensor: str,
        streams: Mapping[str, torch.Tensor],
        validity_by_stream: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        backbone = (
            self.backbone
            if self.sharing == "shared"
            else self.backbones[sensor]
        )
        if self.mode == "pretrain":
            if self.validity_masked_reconstruction:
                loss, predictions, masks, stream_losses = backbone(
                    streams,
                    validity_by_stream=validity_by_stream,
                )
            else:
                loss, predictions, masks, stream_losses = backbone(streams)
            return {
                "loss": loss,
                "predictions": predictions,
                "masks": masks,
                "stream_losses": stream_losses,
            }
        features, encoded = backbone.forward_features(streams)
        logits = self.heads[sensor](features).squeeze(-1)
        return {"logits": logits, "encoded": encoded}


def cycle_loader(loader: DataLoader) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def move_streams(
    streams: Mapping[str, torch.Tensor],
    device: torch.device,
    image_size: int,
) -> Dict[str, torch.Tensor]:
    target_shape = (int(image_size), int(image_size))
    moved: Dict[str, torch.Tensor] = {}
    for name, tensor in streams.items():
        tensor = tensor.to(device=device, non_blocking=True)
        if tuple(tensor.shape[-2:]) != target_shape:
            tensor = F.interpolate(
                tensor,
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )
        moved[name] = tensor
    return moved


def move_validity_masks(
    validity_by_stream: Mapping[str, torch.Tensor],
    device: torch.device,
    image_size: int,
) -> Dict[str, torch.Tensor]:
    target_shape = (int(image_size), int(image_size))
    moved: Dict[str, torch.Tensor] = {}
    for name, tensor in validity_by_stream.items():
        tensor = tensor.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if tuple(tensor.shape[-2:]) != target_shape:
            tensor = resize_validity_tensor(tensor, target_shape)
        moved[name] = tensor.clamp(0.0, 1.0)
    return moved


def move_model_inputs(
    model_inputs: Mapping[str, Any],
    device: torch.device,
    image_size: int,
    *,
    validity_masked_reconstruction: bool,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    if not validity_masked_reconstruction:
        return move_streams(model_inputs, device, image_size), None

    if set(model_inputs) != {"values", "validity"}:
        raise ValueError(
            "Validity-masked batches require exactly {'values', 'validity'}, "
            f"got {sorted(model_inputs)}"
        )
    streams = model_inputs["values"]
    validity_by_stream = model_inputs["validity"]
    if not isinstance(streams, Mapping) or not isinstance(
        validity_by_stream,
        Mapping,
    ):
        raise TypeError("Validity-masked batch values and validity must be mappings")
    if set(streams) != set(validity_by_stream):
        raise ValueError(
            "Stream/value keys differ from validity keys: "
            f"values_only={sorted(set(streams) - set(validity_by_stream))}, "
            f"validity_only={sorted(set(validity_by_stream) - set(streams))}"
        )
    return (
        move_streams(streams, device, image_size),
        move_validity_masks(validity_by_stream, device, image_size),
    )


def build_data(
    args: argparse.Namespace,
    sensors: Sequence[str],
    csvs: Mapping[str, Mapping[str, Path]],
    stats: Mapping[str, Mapping[str, Any]],
    device: torch.device,
) -> Tuple[
    Dict[str, TemporalResidualDataset],
    Dict[str, TemporalResidualDataset],
    Dict[str, DataLoader],
    Dict[str, DataLoader],
]:
    train_datasets: Dict[str, TemporalResidualDataset] = {}
    val_datasets: Dict[str, TemporalResidualDataset] = {}
    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}
    for sensor_index, sensor in enumerate(sensors):
        common = {
            "spec": SENSOR_SPECS[sensor],
            "stats": stats[sensor],
            "image_size": args.image_size,
            "current_clip": args.current_clip,
            "residual_clip": args.residual_clip,
            "s5p_data_key": args.s5p_data_key,
            "resize_on_device": args.resize_on_device,
            "return_validity": args.validity_masked_reconstruction,
        }
        train_dataset = TemporalResidualDataset(
            csvs[sensor]["train"], augment=args.augment, **common
        )
        val_dataset = TemporalResidualDataset(
            csvs[sensor]["val"], augment=False, **common
        )
        generator = torch.Generator()
        generator.manual_seed(args.seed + sensor_index * 1009)
        val_generator = torch.Generator()
        val_generator.manual_seed(args.seed + 500_000 + sensor_index * 1009)
        loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        }
        if args.num_workers > 0:
            loader_kwargs.update(
                {
                    # Eight loaders exist in a four-sensor run. Persistent
                    # workers would leave all train and validation pools alive
                    # simultaneously (8 * num_workers processes), which harms
                    # throughput on the 32-core host.
                    "persistent_workers": False,
                    "prefetch_factor": 2,
                }
            )
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=False,
            generator=generator,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            # Several manifests are ordered by label. A prefix-limited screen
            # would otherwise evaluate a one-class subset (notably S5P).
            shuffle=args.max_val_batches > 0,
            drop_last=False,
            generator=val_generator,
            **loader_kwargs,
        )
        train_datasets[sensor] = train_dataset
        val_datasets[sensor] = val_dataset
        train_loaders[sensor] = train_loader
        val_loaders[sensor] = val_loader
        positives = int(train_dataset.labels.sum())
        print(
            f"[Data] {sensor}: train={len(train_dataset)} (positive={positives}) "
            f"val={len(val_dataset)} train_csv={csvs[sensor]['train']}",
            flush=True,
        )
    return train_datasets, val_datasets, train_loaders, val_loaders


def build_ranknet_objective_signature() -> Dict[str, Any]:
    """Return the immutable contract for the authorized supervised fallback."""

    return {
        "objective_version": RANKNET_OBJECTIVE_VERSION,
        "rank_weight": RANKNET_FIXED_WEIGHT,
        "rank_temperature": RANKNET_FIXED_TEMPERATURE,
        "pair_scope": "within_sensor_current_batch",
        "pair_selection": "all_positive_negative_pairs",
        "one_class_policy": "connected_zero_rank_loss",
    }


def all_pairs_ranknet_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Stable all-positive/negative-pairs RankNet loss for one sensor batch."""

    if logits.ndim != 1 or labels.ndim != 1:
        raise ValueError(
            "RankNet expects one-dimensional logits and labels, got "
            f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )
    if logits.shape != labels.shape:
        raise ValueError(
            f"RankNet logits/labels shape mismatch: "
            f"{tuple(logits.shape)} vs {tuple(labels.shape)}"
        )
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError(
            f"RankNet temperature must be finite and positive, got {temperature}"
        )

    positive_mask = labels == 1
    negative_mask = labels == 0
    positive_logits = logits[positive_mask]
    negative_logits = logits[negative_mask]
    positive_count = int(positive_logits.numel())
    negative_count = int(negative_logits.numel())
    if positive_count + negative_count != int(labels.numel()):
        raise ValueError("RankNet labels must contain only exact binary values 0 or 1.")

    pair_count = positive_count * negative_count
    if pair_count == 0:
        # Keep the result attached to the graph so a one-class batch can
        # continue through the ordinary BCE backward path without a special
        # optimizer branch.
        rank_loss = logits.sum() * 0.0
    else:
        # Compute the potentially large logit differences in float32 even
        # under BF16 autocast. F.softplus is stable for extreme magnitudes.
        pairwise_margin = (
            negative_logits.float().unsqueeze(0)
            - positive_logits.float().unsqueeze(1)
        ) / float(temperature)
        rank_loss = F.softplus(pairwise_margin).mean()

    return rank_loss, {
        "positive_samples": positive_count,
        "negative_samples": negative_count,
        "pair_count": pair_count,
        "one_class_batch": int(pair_count == 0),
    }


def supervised_classification_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    classification_loss: nn.Module,
    *,
    ranknet_enabled: bool,
    ranknet_weight: float,
    ranknet_temperature: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute legacy BCE or the isolated BCE + RankNet fallback objective."""

    weighted_bce = classification_loss(logits, labels)
    if not ranknet_enabled:
        return weighted_bce, {
            "weighted_bce_loss": weighted_bce.detach(),
            "ranknet_loss": logits.detach().sum() * 0.0,
            "positive_samples": int((labels == 1).sum().item()),
            "negative_samples": int((labels == 0).sum().item()),
            "pair_count": 0,
            "one_class_batch": 0,
        }

    ranknet_loss, pair_diagnostics = all_pairs_ranknet_loss(
        logits,
        labels,
        temperature=ranknet_temperature,
    )
    if float(ranknet_weight) == 0.0:
        # Preserve exact legacy loss semantics for the CPU regression test.
        total_loss = weighted_bce
    else:
        total_loss = weighted_bce + float(ranknet_weight) * ranknet_loss
    return total_loss, {
        "weighted_bce_loss": weighted_bce.detach(),
        "ranknet_loss": ranknet_loss.detach(),
        **pair_diagnostics,
    }


def make_classification_losses(
    datasets: Mapping[str, TemporalResidualDataset],
    *,
    balanced: bool,
    device: torch.device,
) -> Dict[str, nn.Module]:
    losses: Dict[str, nn.Module] = {}
    for sensor, dataset in datasets.items():
        labels = dataset.labels
        positives = float((labels == 1).sum())
        negatives = float((labels == 0).sum())
        if balanced and positives > 0:
            # Avoid an extreme weight if a diagnostic subset is nearly one-class.
            positive_weight = min(max(negatives / positives, 0.25), 4.0)
        else:
            positive_weight = 1.0
        losses[sensor] = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(positive_weight, device=device)
        )
        print(f"[Loss] {sensor}: pos_weight={positive_weight:.6f}", flush=True)
    return losses


def _binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels_bool = labels.astype(bool)
    predictions_bool = predictions.astype(bool)
    true_positive = float(np.logical_and(labels_bool, predictions_bool).sum())
    false_positive = float(np.logical_and(~labels_bool, predictions_bool).sum())
    false_negative = float(np.logical_and(labels_bool, ~predictions_bool).sum())
    denominator = 2.0 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2.0 * true_positive / denominator


def classification_metrics(labels: Sequence[float], probabilities: Sequence[float]) -> Dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if labels_array.size == 0:
        return {
            "samples": 0,
            "positive_rate": None,
            "ap": None,
            "auroc": None,
            "f1": None,
            "f1_0p5": None,
            "threshold": None,
        }

    fixed_f1 = _binary_f1(labels_array, probabilities_array >= 0.5)
    thresholds = np.unique(
        np.concatenate(
            (
                np.asarray([0.0, 0.5, 1.0]),
                probabilities_array,
            )
        )
    )
    f1_values = np.asarray(
        [_binary_f1(labels_array, probabilities_array >= threshold) for threshold in thresholds]
    )
    best_index = int(np.argmax(f1_values))
    one_class = np.unique(labels_array).size < 2
    return {
        "samples": int(labels_array.size),
        "positive_rate": float(labels_array.mean()),
        "ap": None if one_class else float(average_precision_score(labels_array, probabilities_array)),
        "auroc": None if one_class else float(roc_auc_score(labels_array, probabilities_array)),
        "f1": float(f1_values[best_index]),
        "f1_0p5": float(fixed_f1),
        "threshold": float(thresholds[best_index]),
    }


def finite_macro(sensor_metrics: Mapping[str, Mapping[str, Any]], key: str) -> Optional[float]:
    values = [
        float(metrics[key])
        for metrics in sensor_metrics.values()
        if metrics.get(key) is not None and math.isfinite(float(metrics[key]))
    ]
    return None if not values else float(np.mean(values))


def amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def accumulate_validity_diagnostics(
    totals: Dict[str, float],
    weights: Dict[str, float],
    diagnostics: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
) -> None:
    """Accumulate adapter diagnostics without hiding excluded samples/streams."""

    scalar_diagnostics = {
        key: float(value.detach().float().cpu())
        for key, value in diagnostics.items()
    }
    for key, scalar in scalar_diagnostics.items():
        if key.startswith("loss_"):
            stream = key[len("loss_") :]
            weight = scalar_diagnostics.get(f"valid_samples_{stream}", 0.0)
            totals[key] = totals.get(key, 0.0) + scalar * weight
            weights[key] = weights.get(key, 0.0) + weight
        elif key.startswith("valid_fraction_"):
            weight = float(batch_size)
            totals[key] = totals.get(key, 0.0) + scalar * weight
            weights[key] = weights.get(key, 0.0) + weight
        else:
            # Counts are deliberately summed across batches.  This includes
            # valid/excluded/batch samples, valid masked elements, and
            # valid/empty/total stream-batch counts.
            totals[key] = totals.get(key, 0.0) + scalar


def finalize_validity_diagnostics(
    totals: Mapping[str, float],
    weights: Mapping[str, float],
) -> Dict[str, float]:
    return {
        key: (
            float(total) / max(float(weights.get(key, 0.0)), 1.0)
            if key.startswith("loss_")
            or key.startswith("valid_fraction_")
            else float(total)
        )
        for key, total in sorted(totals.items())
    }


def train_one_epoch(
    model: MultiSensorResidualModel,
    train_loaders: Mapping[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    classification_losses: Mapping[str, nn.Module],
    *,
    sensors: Sequence[str],
    device: torch.device,
    amp: bool,
    grad_clip: float,
    rounds: int,
    mode: str,
    image_size: int,
    log_interval_rounds: int,
    validity_masked_reconstruction: bool,
    ranknet_enabled: bool,
    ranknet_weight: float,
    ranknet_temperature: float,
) -> Tuple[
    Dict[str, float],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, Any]],
]:
    model.train()
    iterators = {sensor: cycle_loader(train_loaders[sensor]) for sensor in sensors}
    totals = {sensor: 0.0 for sensor in sensors}
    counts = {sensor: 0 for sensor in sensors}
    latest_validity_diagnostics: Dict[str, Dict[str, float]] = {}
    diagnostic_totals: Dict[str, Dict[str, float]] = {
        sensor: {} for sensor in sensors
    }
    diagnostic_weights: Dict[str, Dict[str, float]] = {
        sensor: {} for sensor in sensors
    }
    ranknet_accumulators: Dict[str, Dict[str, float]] = {
        sensor: {
            "weighted_bce_sum": 0.0,
            "ranknet_valid_batch_sum": 0.0,
            "positive_samples": 0.0,
            "negative_samples": 0.0,
            "pair_count": 0.0,
            "batches": 0.0,
            "paired_batches": 0.0,
            "one_class_batches": 0.0,
        }
        for sensor in sensors
    }
    latest_ranknet_diagnostics: Dict[str, Dict[str, Any]] = {}
    started = time.time()

    # Every round contributes exactly one batch from every sensor.  Optimizer
    # steps are intentionally per-sensor so batches with incompatible channel
    # counts never need padding or concatenation.
    for _round_index in range(rounds):
        for sensor in sensors:
            model_inputs, labels = next(iterators[sensor])
            streams, validity_by_stream = move_model_inputs(
                model_inputs,
                device,
                image_size,
                validity_masked_reconstruction=(
                    validity_masked_reconstruction
                ),
            )
            labels = labels.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp):
                output = model(
                    sensor,
                    streams,
                    validity_by_stream=validity_by_stream,
                )
                if mode == "pretrain":
                    loss = output["loss"]
                    if validity_masked_reconstruction:
                        latest_validity_diagnostics[sensor] = {
                            key: float(value.detach().float().cpu())
                            for key, value in output[
                                "stream_losses"
                            ].items()
                        }
                        accumulate_validity_diagnostics(
                            diagnostic_totals[sensor],
                            diagnostic_weights[sensor],
                            output["stream_losses"],
                            batch_size=int(labels.shape[0]),
                        )
                else:
                    if ranknet_enabled:
                        (
                            loss,
                            batch_objective_diagnostics,
                        ) = supervised_classification_objective(
                            output["logits"],
                            labels,
                            classification_losses[sensor],
                            ranknet_enabled=True,
                            ranknet_weight=ranknet_weight,
                            ranknet_temperature=ranknet_temperature,
                        )
                        batch_size = int(labels.shape[0])
                        pair_count = int(
                            batch_objective_diagnostics["pair_count"]
                        )
                        accumulator = ranknet_accumulators[sensor]
                        accumulator["weighted_bce_sum"] += (
                            float(
                                batch_objective_diagnostics[
                                    "weighted_bce_loss"
                                ].float().cpu()
                            )
                            * batch_size
                        )
                        accumulator["positive_samples"] += int(
                            batch_objective_diagnostics["positive_samples"]
                        )
                        accumulator["negative_samples"] += int(
                            batch_objective_diagnostics["negative_samples"]
                        )
                        accumulator["pair_count"] += pair_count
                        accumulator["batches"] += 1
                        if pair_count > 0:
                            accumulator["ranknet_valid_batch_sum"] += float(
                                batch_objective_diagnostics[
                                    "ranknet_loss"
                                ].float().cpu()
                            )
                            accumulator["paired_batches"] += 1
                        else:
                            accumulator["one_class_batches"] += 1
                        latest_ranknet_diagnostics[sensor] = {
                            "weighted_bce_loss": float(
                                batch_objective_diagnostics[
                                    "weighted_bce_loss"
                                ].float().cpu()
                            ),
                            "ranknet_loss": float(
                                batch_objective_diagnostics[
                                    "ranknet_loss"
                                ].float().cpu()
                            ),
                            "positive_samples": int(
                                batch_objective_diagnostics[
                                    "positive_samples"
                                ]
                            ),
                            "negative_samples": int(
                                batch_objective_diagnostics[
                                    "negative_samples"
                                ]
                            ),
                            "pair_count": pair_count,
                        }
                    else:
                        # Keep the legacy supervised path unchanged when the
                        # explicitly authorized fallback is disabled.
                        loss = classification_losses[sensor](
                            output["logits"],
                            labels,
                        )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {sensor} loss: {loss.detach().cpu().item()}")
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_size = int(labels.shape[0])
            totals[sensor] += float(loss.detach().cpu()) * batch_size
            counts[sensor] += batch_size
        completed_rounds = _round_index + 1
        if (
            log_interval_rounds > 0
            and (
                completed_rounds % log_interval_rounds == 0
                or completed_rounds == rounds
            )
        ):
            print(
                f"[Train progress] rounds={completed_rounds}/{rounds} "
                f"optimizer_steps={completed_rounds * len(sensors)} "
                f"elapsed_seconds={time.time() - started:.1f}"
                + (
                    " validity="
                    + json.dumps(
                        json_safe(latest_validity_diagnostics),
                        sort_keys=True,
                    )
                    if validity_masked_reconstruction
                    and latest_validity_diagnostics
                    else ""
                )
                + (
                    " ranknet="
                    + json.dumps(
                        json_safe(latest_ranknet_diagnostics),
                        sort_keys=True,
                    )
                    if ranknet_enabled and latest_ranknet_diagnostics
                    else ""
                ),
                flush=True,
            )
    losses = {
        sensor: totals[sensor] / max(1, counts[sensor])
        for sensor in sensors
    }
    diagnostics = {
        sensor: finalize_validity_diagnostics(
            diagnostic_totals[sensor],
            diagnostic_weights[sensor],
        )
        for sensor in sensors
        if diagnostic_totals[sensor]
    }
    supervised_diagnostics: Dict[str, Dict[str, Any]] = {}
    if ranknet_enabled:
        for sensor in sensors:
            accumulator = ranknet_accumulators[sensor]
            batch_count = int(accumulator["batches"])
            paired_batch_count = int(accumulator["paired_batches"])
            sample_count = int(
                accumulator["positive_samples"]
                + accumulator["negative_samples"]
            )
            supervised_diagnostics[sensor] = {
                "total_loss": float(losses[sensor]),
                "weighted_bce_loss": (
                    float(accumulator["weighted_bce_sum"])
                    / max(1, sample_count)
                ),
                "ranknet_loss": (
                    float(accumulator["ranknet_valid_batch_sum"])
                    / max(1, paired_batch_count)
                ),
                "positive_samples": int(
                    accumulator["positive_samples"]
                ),
                "negative_samples": int(
                    accumulator["negative_samples"]
                ),
                "pair_count": int(accumulator["pair_count"]),
                "batches": batch_count,
                "paired_batches": paired_batch_count,
                "one_class_batches": int(
                    accumulator["one_class_batches"]
                ),
                "paired_batch_fraction": (
                    float(paired_batch_count) / max(1, batch_count)
                ),
                "one_class_batch_fraction": (
                    float(accumulator["one_class_batches"])
                    / max(1, batch_count)
                ),
            }
    return losses, diagnostics, supervised_diagnostics


@torch.inference_mode()
def evaluate_supervised(
    model: MultiSensorResidualModel,
    val_loaders: Mapping[str, DataLoader],
    classification_losses: Mapping[str, nn.Module],
    *,
    sensors: Sequence[str],
    device: torch.device,
    amp: bool,
    max_batches: int,
    image_size: int,
    validity_masked_reconstruction: bool,
    prediction_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    model.eval()
    output_metrics: Dict[str, Dict[str, Any]] = {}
    for sensor in sensors:
        labels_all: List[float] = []
        probabilities_all: List[float] = []
        loss_total = 0.0
        sample_count = 0
        for batch_index, (model_inputs, labels) in enumerate(
            val_loaders[sensor]
        ):
            if max_batches > 0 and batch_index >= max_batches:
                break
            streams, validity_by_stream = move_model_inputs(
                model_inputs,
                device,
                image_size,
                validity_masked_reconstruction=(
                    validity_masked_reconstruction
                ),
            )
            labels = labels.to(device=device, non_blocking=True)
            with amp_context(device, amp):
                logits = model(
                    sensor,
                    streams,
                    validity_by_stream=validity_by_stream,
                )["logits"]
                loss = classification_losses[sensor](logits, labels)
            probabilities = torch.sigmoid(logits.float())
            labels_all.extend(labels.detach().cpu().tolist())
            probabilities_all.extend(probabilities.detach().cpu().tolist())
            loss_total += float(loss.detach().cpu()) * int(labels.shape[0])
            sample_count += int(labels.shape[0])
        metrics = classification_metrics(labels_all, probabilities_all)
        metrics["loss"] = loss_total / max(1, sample_count)
        if prediction_rows is not None:
            labels_array = np.asarray(labels_all, dtype=np.int64)
            probabilities_array = np.asarray(
                probabilities_all,
                dtype=np.float64,
            )
            threshold = metrics.get("threshold")
            if (
                probabilities_array.size == 0
                or threshold is None
                or not math.isfinite(float(threshold))
            ):
                raise ValueError(
                    f"RankNet full validation produced no finite result for "
                    f"{sensor}."
                )
            metrics.update(
                {
                    "probability_min": float(probabilities_array.min()),
                    "probability_max": float(probabilities_array.max()),
                    "probability_mean": float(probabilities_array.mean()),
                    "probability_std": float(probabilities_array.std()),
                    "predicted_positive_at_0p5": int(
                        (probabilities_array >= 0.5).sum()
                    ),
                    "predicted_positive_at_best": int(
                        (
                            probabilities_array
                            >= float(threshold)
                        ).sum()
                    ),
                }
            )
            prediction_rows.extend(
                {
                    "sensor": sensor,
                    "row_index": row_index,
                    "label": int(label),
                    "probability": float(probability),
                }
                for row_index, (label, probability) in enumerate(
                    zip(labels_array.tolist(), probabilities_array.tolist())
                )
            )
        output_metrics[sensor] = metrics
    return output_metrics


@torch.inference_mode()
def evaluate_pretrain(
    model: MultiSensorResidualModel,
    val_loaders: Mapping[str, DataLoader],
    *,
    sensors: Sequence[str],
    device: torch.device,
    amp: bool,
    max_batches: int,
    image_size: int,
    validity_masked_reconstruction: bool,
) -> Dict[str, Dict[str, Any]]:
    model.eval()
    output_metrics: Dict[str, Dict[str, Any]] = {}
    for sensor in sensors:
        total = 0.0
        samples = 0
        diagnostic_totals: Dict[str, float] = {}
        diagnostic_weights: Dict[str, float] = {}
        for batch_index, (model_inputs, labels) in enumerate(
            val_loaders[sensor]
        ):
            if max_batches > 0 and batch_index >= max_batches:
                break
            streams, validity_by_stream = move_model_inputs(
                model_inputs,
                device,
                image_size,
                validity_masked_reconstruction=(
                    validity_masked_reconstruction
                ),
            )
            with amp_context(device, amp):
                model_output = model(
                    sensor,
                    streams,
                    validity_by_stream=validity_by_stream,
                )
                loss = model_output["loss"]
            batch_size = int(labels.shape[0])
            total += float(loss.detach().cpu()) * batch_size
            samples += batch_size
            if validity_masked_reconstruction:
                accumulate_validity_diagnostics(
                    diagnostic_totals,
                    diagnostic_weights,
                    model_output["stream_losses"],
                    batch_size=batch_size,
                )
        output_metrics[sensor] = {
            "samples": samples,
            "reconstruction_loss": total / max(1, samples),
        }
        if validity_masked_reconstruction:
            diagnostics = finalize_validity_diagnostics(
                diagnostic_totals,
                diagnostic_weights,
            )
            output_metrics[sensor]["validity_diagnostics"] = diagnostics
            # Preserve the exact masked objective across validation batches:
            # samples are averaged within each stream, then non-empty streams
            # receive equal weight.
            stream_losses = [
                value
                for key, value in diagnostics.items()
                if key.startswith("loss_")
                and diagnostics.get(
                    f"valid_samples_{key[len('loss_') :]}",
                    0.0,
                )
                > 0
            ]
            if stream_losses:
                output_metrics[sensor]["reconstruction_loss"] = float(
                    np.mean(stream_losses)
                )
    return output_metrics


def build_encoder_signature(
    args: argparse.Namespace, sensors: Sequence[str]
) -> Dict[str, Any]:
    source_path = NORMWEAR_ROOT / "modules" / "methane_residual_mae.py"
    signature = {
        "schema_version": 1,
        "sharing": args.sharing,
        "sensors": list(sensors),
        "stream_channels": {
            f"{sensor}_{suffix}": SENSOR_SPECS[sensor].channels
            for sensor in sensors
            for suffix in STREAM_SUFFIXES
        },
        "image_size": int(args.image_size),
        "patch_size": int(args.patch_size),
        "embed_dim": int(args.embed_dim),
        "depth": int(args.depth),
        "num_heads": int(args.num_heads),
        "mlp_ratio": float(args.mlp_ratio),
        "fuse_freq": int(args.fuse_freq),
        "dropout": float(args.dropout),
        "temporal_fusion_mode": "current_query",
        "normwear_source_sha256": sha256_file(source_path),
    }
    if args.validity_masked_reconstruction:
        adapter_path = Path(__file__).resolve().with_name(
            "methane_residual_mae_validity.py"
        )
        signature["schema_version"] = 2
        signature["validity_adapter"] = {
            "class": "ValidityMaskedMethaneResidualMAE",
            "source_sha256": sha256_file(adapter_path),
            "objective_version": VALIDITY_OBJECTIVE_VERSION,
        }
    return signature


def build_validity_reconstruction_signature() -> Dict[str, Any]:
    adapter_path = Path(__file__).resolve().with_name(
        "methane_residual_mae_validity.py"
    )
    return {
        "enabled": True,
        "objective_version": VALIDITY_OBJECTIVE_VERSION,
        "semantics": VALIDITY_SEMANTICS,
        "adapter_source_sha256": sha256_file(adapter_path),
        "native_validity": {
            "tiff": "finite_and_nonzero",
            "s5p": "finite_including_zero",
        },
        "stream_validity": {
            "current": "t0",
            "recent": "t0_and_prev1",
            "seasonal": "t0_and_seasonal",
        },
        "resize": {
            "downsample": "area",
            "upsample": "nearest",
        },
        "loss_normalization": (
            "valid_elements_per_sample_then_equal_valid_samples_per_stream"
            "_then_equal_nonempty_streams"
        ),
    }


def build_data_signature(
    args: argparse.Namespace,
    sensors: Sequence[str],
    csvs: Mapping[str, Mapping[str, Path]],
    stats_path: Path,
    event_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    signature = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "canonical_event_rule": CANONICAL_EVENT_RULE,
        "event_protocol_fingerprint": event_audit["fingerprint"],
        "global_train_val_overlap_after": int(
            event_audit["global_train_val_overlap_after"]
        ),
        "manifest_sha256": {
            sensor: {
                split: sha256_file(csvs[sensor][split])
                for split in ("train", "val")
            }
            for sensor in sensors
        },
        "normalization_stats_sha256": sha256_file(stats_path),
        "representation": {
            "streams": ["normalized_t0", "t0-prev1", "t0-seasonal"],
            "current_clip": float(args.current_clip),
            "residual_clip": float(args.residual_clip),
            "s5p_data_key": str(args.s5p_data_key),
        },
    }
    if args.validity_masked_reconstruction:
        signature["schema_version"] = PROTOCOL_SCHEMA_VERSION + 1
        signature["representation"]["validity_reconstruction"] = (
            build_validity_reconstruction_signature()
        )
    return signature


def build_resume_signature(
    args: argparse.Namespace,
    sensors: Sequence[str],
    encoder_signature: Mapping[str, Any],
    data_signature: Mapping[str, Any],
    *,
    rounds: int,
    loader_lengths: Mapping[str, int],
) -> Dict[str, Any]:
    """Settings that must be identical when continuing one interrupted run."""

    signature = {
        "schema_version": 1,
        "mode": args.mode,
        "sharing": args.sharing,
        "sensors": list(sensors),
        "encoder_signature": encoder_signature,
        "pretrain_model": (
            {
                "mask_ratio": float(args.mask_ratio),
                "decoder_embed_dim": int(args.decoder_embed_dim),
                "decoder_depth": int(args.decoder_depth),
                "decoder_num_heads": int(args.decoder_num_heads),
            }
            if args.mode == "pretrain"
            else None
        ),
        "data_signature": data_signature,
        "optimization": {
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "balanced_pos_weight": bool(args.balanced_pos_weight),
            "amp": bool(args.amp),
            "augment": bool(args.augment),
            "seed": int(args.seed),
            "balanced_rounds": int(rounds),
            "optimizer_steps_per_epoch": int(rounds * len(sensors)),
            "loader_lengths": dict(loader_lengths),
        },
        "evaluation": {
            "max_val_batches": int(args.max_val_batches),
        },
    }
    if args.validity_masked_reconstruction:
        signature["schema_version"] = 2
        signature["validity_reconstruction"] = (
            build_validity_reconstruction_signature()
        )
        if signature["pretrain_model"] is not None:
            signature["pretrain_model"]["reconstruction_objective"] = (
                VALIDITY_OBJECTIVE_VERSION
            )
    if bool(getattr(args, "ranknet_objective", False)):
        if args.mode != "supervised":
            raise ValueError(
                "The RankNet fallback is a supervised objective only."
            )
        if args.validity_masked_reconstruction:
            raise ValueError(
                "RankNet and validity-masked reconstruction cannot share one "
                "run signature."
            )
        signature["schema_version"] = 2
        signature["supervised_objective"] = (
            build_ranknet_objective_signature()
        )
    return signature


def _signature_error(
    name: str, expected: Mapping[str, Any], observed: Any
) -> ValueError:
    return ValueError(
        f"{name} mismatch: expected_sha256={json_fingerprint(expected)}, "
        f"checkpoint_sha256={json_fingerprint(observed)}. Refusing to mix "
        "different architecture/data/training protocols."
    )


def _transfer_exclusion_reason(key: str) -> Optional[str]:
    components = key.split(".")
    if components[0] == "heads":
        return "classification_head"
    if "mask_token" in components or any(
        component.startswith("decoder") for component in components
    ):
        return "mae_decoder"
    return None


def load_pretrained_encoder(
    model: MultiSensorResidualModel,
    path: Path,
    *,
    expected_encoder_signature: Mapping[str, Any],
    expected_data_signature: Mapping[str, Any],
) -> Dict[str, Any]:
    """Strict encoder-only MAE transfer with auditable exclusions."""

    if model.mode != "supervised":
        raise ValueError("--init-checkpoint is only valid for --mode supervised.")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected a checkpoint mapping at {path}.")
    if checkpoint.get("mode") != "pretrain":
        raise ValueError(
            f"Transfer source must declare mode='pretrain', got "
            f"{checkpoint.get('mode')!r}: {path}"
        )
    if checkpoint.get("encoder_signature") != expected_encoder_signature:
        raise _signature_error(
            "encoder signature",
            expected_encoder_signature,
            checkpoint.get("encoder_signature"),
        )
    if checkpoint.get("data_signature") != expected_data_signature:
        raise _signature_error(
            "data signature",
            expected_data_signature,
            checkpoint.get("data_signature"),
        )
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint has no model state dict: {path}")

    target_state = model.state_dict()
    target_encoder_keys = {
        key for key in target_state if _transfer_exclusion_reason(key) is None
    }
    filtered: Dict[str, torch.Tensor] = {}
    excluded: Dict[str, List[str]] = {
        "classification_head": [],
        "mae_decoder": [],
    }
    unknown_encoder_keys: List[str] = []
    shape_mismatches: List[str] = []
    for key, value in state.items():
        reason = _transfer_exclusion_reason(str(key))
        if reason is not None:
            excluded[reason].append(str(key))
            continue
        if key not in target_state:
            unknown_encoder_keys.append(str(key))
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            shape_mismatches.append(
                f"{key}: source={tuple(value.shape)} target={tuple(target_state[key].shape)}"
            )
            continue
        filtered[str(key)] = value

    missing_encoder = sorted(target_encoder_keys - set(filtered))
    if missing_encoder or unknown_encoder_keys or shape_mismatches:
        raise ValueError(
            "Pretrained encoder is not an exact transfer match: "
            f"missing={missing_encoder[:20]}, "
            f"unknown={sorted(unknown_encoder_keys)[:20]}, "
            f"shape_mismatches={shape_mismatches[:20]}"
        )
    head_before = state_dict_fingerprint(target_state, prefix="heads.")
    result = model.load_state_dict(filtered, strict=False)
    expected_missing = sorted(set(target_state) - target_encoder_keys)
    if sorted(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise AssertionError(
            "Internal transfer invariant failed: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    head_after = state_dict_fingerprint(model.state_dict(), prefix="heads.")
    if head_before != head_after:
        raise AssertionError("Classification heads changed during encoder-only transfer.")
    loaded_numel = sum(int(target_state[key].numel()) for key in filtered)
    encoder_numel = sum(int(target_state[key].numel()) for key in target_encoder_keys)
    report = {
        "path": str(Path(path).resolve()),
        "checkpoint_sha256": sha256_file(path),
        "loaded_encoder_keys": len(filtered),
        "loaded_encoder_numel": loaded_numel,
        "target_encoder_numel": encoder_numel,
        "encoder_coverage": loaded_numel / max(1, encoder_numel),
        "excluded_decoder_keys": sorted(excluded["mae_decoder"]),
        "excluded_checkpoint_head_keys": sorted(excluded["classification_head"]),
        "target_head_keys": expected_missing,
        "target_head_sha256_before": head_before,
        "target_head_sha256_after": head_after,
    }
    if report["encoder_coverage"] != 1.0:
        raise AssertionError(f"Encoder coverage is not 100%: {report}")
    return report


def run_self_test() -> None:
    """CPU-only proof of the transfer contract; reads no dataset manifests."""

    sensors = ("s2", "s5p")
    model_kwargs = {
        "sensors": sensors,
        "sharing": "shared",
        "image_size": 28,
        "patch_size": 14,
        "embed_dim": 32,
        "depth": 1,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "fuse_freq": 1,
        "dropout": 0.0,
        "mask_ratio": 0.5,
        "decoder_embed_dim": 16,
        "decoder_depth": 1,
        "decoder_num_heads": 4,
    }
    torch.manual_seed(17)
    pretrain = MultiSensorResidualModel(mode="pretrain", **model_kwargs)
    torch.manual_seed(23)
    finetune = MultiSensorResidualModel(mode="supervised", **model_kwargs)
    torch.manual_seed(23)
    scratch = MultiSensorResidualModel(mode="supervised", **model_kwargs)
    scratch_head_sha = state_dict_fingerprint(scratch.state_dict(), prefix="heads.")
    initial_head_sha = state_dict_fingerprint(finetune.state_dict(), prefix="heads.")
    if scratch_head_sha != initial_head_sha:
        raise AssertionError("Matched scratch and finetune heads do not initialize equally.")

    encoder_signature = {"self_test": "encoder-v1"}
    data_signature = {"self_test": "data-v1"}
    with tempfile.TemporaryDirectory(prefix="methane_residual_transfer_test.") as directory:
        checkpoint_path = Path(directory) / "pretrain.pth"
        atomic_torch_save(
            {
                "epoch": 0,
                "mode": "pretrain",
                "sharing": "shared",
                "sensors": sensors,
                "encoder_signature": encoder_signature,
                "data_signature": data_signature,
                "model": pretrain.state_dict(),
                "optimizer": {},
            },
            checkpoint_path,
        )
        report = load_pretrained_encoder(
            finetune,
            checkpoint_path,
            expected_encoder_signature=encoder_signature,
            expected_data_signature=data_signature,
        )
        bad_checkpoint = Path(directory) / "not_pretrain.pth"
        bad_payload = torch.load(checkpoint_path, map_location="cpu")
        bad_payload["mode"] = "supervised"
        atomic_torch_save(bad_payload, bad_checkpoint)
        try:
            load_pretrained_encoder(
                scratch,
                bad_checkpoint,
                expected_encoder_signature=encoder_signature,
                expected_data_signature=data_signature,
            )
        except ValueError as error:
            wrong_mode_rejected = "mode='pretrain'" in str(error)
        else:
            wrong_mode_rejected = False
        if not wrong_mode_rejected:
            raise AssertionError("A non-pretrain transfer source was not rejected.")

    for sensor in sensors:
        streams = {
            f"{sensor}_{suffix}": torch.randn(
                2, SENSOR_SPECS[sensor].channels, 28, 28
            )
            for suffix in STREAM_SUFFIXES
        }
        supervised_output = finetune(sensor, streams)
        pretrain_output = pretrain(sensor, streams)
        if not torch.isfinite(supervised_output["logits"]).all():
            raise AssertionError(f"Non-finite supervised output for {sensor}.")
        if not torch.isfinite(pretrain_output["loss"]):
            raise AssertionError(f"Non-finite pretrain output for {sensor}.")
    print(
        "[Self-test] "
        + json.dumps(
            {
                "status": "passed",
                "device": "cpu",
                "manifests_read": 0,
                "encoder_coverage": report["encoder_coverage"],
                "decoder_keys_excluded": len(report["excluded_decoder_keys"]),
                "head_sha_matches_scratch": (
                    report["target_head_sha256_after"] == scratch_head_sha
                ),
                "wrong_mode_rejected": wrong_mode_rejected,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({value}) but torch.cuda.is_available() is false.")
    return device


def validate_model_shape_args(args: argparse.Namespace) -> None:
    if args.init_checkpoint is not None and args.resume is not None:
        raise ValueError("--init-checkpoint and --resume are mutually exclusive.")
    if args.init_checkpoint is not None and args.mode != "supervised":
        raise ValueError("--init-checkpoint is only valid with --mode supervised.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.image_size <= 0 or args.patch_size <= 0:
        raise ValueError("--image-size and --patch-size must be positive.")
    if args.image_size % args.patch_size != 0:
        raise ValueError("--image-size must be divisible by --patch-size.")
    if args.embed_dim % args.num_heads != 0:
        raise ValueError("--embed-dim must be divisible by --num-heads.")
    if args.decoder_embed_dim % args.decoder_num_heads != 0:
        raise ValueError("--decoder-embed-dim must be divisible by --decoder-num-heads.")


def validate_ranknet_args(
    args: argparse.Namespace,
    sensors: Sequence[str],
) -> None:
    """Enforce the preregistered fallback instead of enabling a hidden sweep."""

    weight_matches = float(args.ranknet_weight) == RANKNET_FIXED_WEIGHT
    temperature_matches = (
        float(args.ranknet_temperature) == RANKNET_FIXED_TEMPERATURE
    )
    if not args.ranknet_objective:
        if not weight_matches or not temperature_matches:
            raise ValueError(
                "--ranknet-weight/--ranknet-temperature have no effect unless "
                "--ranknet-objective is enabled, and non-preregistered values "
                "are not accepted."
            )
        return

    failures: List[str] = []
    if args.mode != "supervised":
        failures.append("mode must be supervised")
    if args.sharing != "shared":
        failures.append("sharing must be shared")
    if tuple(sensors) != SENSOR_ORDER:
        failures.append(f"sensors must be exactly {SENSOR_ORDER}")
    if not weight_matches:
        failures.append(
            f"ranknet_weight must be exactly {RANKNET_FIXED_WEIGHT}"
        )
    if not temperature_matches:
        failures.append(
            "ranknet_temperature must be exactly "
            f"{RANKNET_FIXED_TEMPERATURE}"
        )
    if not args.balanced_pos_weight:
        failures.append("balanced positive weighting must remain enabled")
    if args.validity_masked_reconstruction:
        failures.append(
            "validity-masked reconstruction must remain disabled"
        )
    if args.init_checkpoint is not None:
        failures.append("the fallback must start from scratch")
    if args.resume is not None:
        failures.append("the one-epoch gate does not authorize resume")
    if args.no_save_checkpoint:
        failures.append(
            "the signed validation artifact requires a saved checkpoint"
        )
    if int(args.epochs) != 1:
        failures.append("the gate authorizes exactly one epoch")
    if int(args.seed) != 20260727:
        failures.append("the gate seed must be exactly 20260727")
    if not args.augment:
        failures.append("the matched augmentation setting must remain enabled")
    if failures:
        raise ValueError(
            "RankNet fallback contract violation: " + "; ".join(failures)
        )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    validate_model_shape_args(args)
    sensors = parse_sensors(args.sensors)
    validate_ranknet_args(args, sensors)
    history_path = args.output_dir / "metrics_history.json"
    checkpoint_path = args.output_dir / "checkpoint_latest.pth"
    if args.resume is None:
        run_artifact_paths = [
            history_path,
            checkpoint_path,
            args.output_dir / "metrics_epochs.jsonl",
            args.output_dir / "metrics_epochs.csv",
        ]
        if args.ranknet_objective:
            run_artifact_paths.extend(
                [
                    args.output_dir / "validation_full_temporal.json",
                    args.output_dir / "validation_full_temporal.csv",
                ]
            )
        stale_artifacts = [
            path
            for path in run_artifact_paths
            if path.exists()
        ]
        if stale_artifacts:
            raise FileExistsError(
                "Refusing to overwrite an existing run. Choose a new "
                f"--output-dir or use --resume: {stale_artifacts}"
            )
    else:
        if Path(args.resume).resolve().parent != args.output_dir.resolve():
            raise ValueError(
                "--resume must point inside the same --output-dir so history "
                "and checkpoint cannot be mixed across runs."
            )
        if not history_path.is_file():
            raise FileNotFoundError(
                f"Resume requires the original metrics history: {history_path}"
            )

    seed_everything(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csvs = resolve_csvs(args, sensors)
    csvs, event_audit = enforce_event_protocol(args, sensors, csvs)
    stats_path, stats = load_or_compute_stats(args, sensors, csvs)
    (
        train_datasets,
        _val_datasets,
        train_loaders,
        val_loaders,
    ) = build_data(args, sensors, csvs, stats, device)

    loader_lengths = {sensor: len(train_loaders[sensor]) for sensor in sensors}
    rounds = (
        int(args.steps_per_sensor)
        if args.steps_per_sensor > 0
        else min(loader_lengths.values())
    )
    if args.max_train_rounds > 0:
        rounds = min(rounds, int(args.max_train_rounds))
    if rounds <= 0:
        raise ValueError(f"No training rounds available; loader lengths={loader_lengths}")
    encoder_signature = build_encoder_signature(args, sensors)
    data_signature = build_data_signature(
        args, sensors, csvs, stats_path, event_audit
    )
    resume_signature = build_resume_signature(
        args,
        sensors,
        encoder_signature,
        data_signature,
        rounds=rounds,
        loader_lengths=loader_lengths,
    )

    model = MultiSensorResidualModel(
        sensors,
        mode=args.mode,
        sharing=args.sharing,
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        fuse_freq=args.fuse_freq,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        validity_masked_reconstruction=(
            args.validity_masked_reconstruction
        ),
    ).to(device)
    init_report: Optional[Dict[str, Any]] = None
    if args.init_checkpoint is not None:
        init_report = load_pretrained_encoder(
            model,
            args.init_checkpoint,
            expected_encoder_signature=encoder_signature,
            expected_data_signature=data_signature,
        )
        print(
            f"[Init] encoder-only {args.init_checkpoint}: "
            f"coverage={init_report['encoder_coverage']:.3f} "
            f"decoder_excluded={len(init_report['excluded_decoder_keys'])} "
            f"checkpoint_heads_excluded="
            f"{len(init_report['excluded_checkpoint_head_keys'])}",
            flush=True,
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"[Model] mode={args.mode} sharing={args.sharing} "
        f"sensor_specific_stems=true parameters={parameter_count:,} "
        f"trainable={trainable_count:,} device={device} "
        f"validity_masked_reconstruction="
        f"{args.validity_masked_reconstruction}"
        + (
            " ranknet_objective=true"
            if args.ranknet_objective
            else ""
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    classification_losses: Mapping[str, nn.Module] = {}
    if args.mode == "supervised":
        classification_losses = make_classification_losses(
            train_datasets, balanced=args.balanced_pos_weight, device=device
        )

    history: Dict[str, Any] = {
        "schema_version": (
            4
            if args.ranknet_objective
            else (3 if args.validity_masked_reconstruction else 2)
        ),
        "status": "running",
        "mode": args.mode,
        "sharing": args.sharing,
        "config": json_safe(artifact_config(args)),
        "sensors": list(sensors),
        "resolved_csvs": json_safe(csvs),
        "normalization_stats": str(stats_path),
        "event_protocol": json_safe(event_audit),
        "encoder_signature": json_safe(encoder_signature),
        "data_signature": json_safe(data_signature),
        "resume_signature": json_safe(resume_signature),
        "model_parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "balanced_rounds_per_epoch": rounds,
        "optimizer_steps_per_epoch": rounds * len(sensors),
        "epochs": [],
    }
    if args.validity_masked_reconstruction:
        history["validity_reconstruction"] = (
            build_validity_reconstruction_signature()
        )
    if args.ranknet_objective:
        history["supervised_objective"] = (
            build_ranknet_objective_signature()
        )
    if args.mode == "supervised":
        history["initial_head_sha256"] = state_dict_fingerprint(
            model.state_dict(), prefix="heads."
        )
    if init_report is not None:
        history["init_checkpoint"] = json_safe(init_report)
    start_epoch = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"Expected checkpoint mapping: {args.resume}")
        if checkpoint.get("resume_signature") != resume_signature:
            raise _signature_error(
                "resume signature",
                resume_signature,
                checkpoint.get("resume_signature"),
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if args.epochs <= start_epoch:
            raise ValueError(
                f"--epochs={args.epochs} leaves nothing to resume after "
                f"checkpoint epoch={checkpoint['epoch']}."
            )
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        if history.get("resume_signature") != resume_signature:
            raise _signature_error(
                "history resume signature",
                resume_signature,
                history.get("resume_signature"),
            )
        if history.get("status") == "failed":
            history.setdefault("interruptions", []).append(
                {
                    "reason": args.resume_reason or "resumed_after_failure",
                    "error": history.get("error"),
                    "error_type": history.get("error_type"),
                    "error_message": history.get("error_message"),
                    "traceback": history.get("traceback"),
                    "failed_unix": history.get("failed_unix"),
                }
            )
            for key in (
                "error",
                "error_type",
                "error_message",
                "traceback",
                "failed_unix",
            ):
                history.pop(key, None)
        # The metrics file is written before the checkpoint.  If a crash
        # happened between those atomic writes, discard records newer than
        # the checkpoint so resuming cannot duplicate an epoch.
        history["epochs"] = [
            record
            for record in history.get("epochs", [])
            if int(record.get("epoch", -1)) <= int(checkpoint["epoch"])
        ]
        history["status"] = "running"
        history.setdefault("resume_events", []).append(
            {
                "checkpoint": str(args.resume),
                "start_epoch": start_epoch,
                "resumed_unix": time.time(),
                "runtime_config": json_safe(artifact_config(args)),
            }
        )
        print(f"[Resume] {args.resume}: starting epoch {start_epoch}", flush=True)

    history["balanced_rounds_per_epoch"] = rounds
    history["optimizer_steps_per_epoch"] = rounds * len(sensors)
    atomic_json_dump(json_safe(history), history_path)
    write_epoch_tables(history, args.output_dir)
    print(
        f"[Train] balanced_rounds={rounds}; optimizer_steps_per_epoch="
        f"{rounds * len(sensors)}; loader_batches={loader_lengths}",
        flush=True,
    )

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_started = time.time()
            (
                train_losses,
                train_validity_diagnostics,
                train_objective_diagnostics,
            ) = train_one_epoch(
                model,
                train_loaders,
                optimizer,
                classification_losses,
                sensors=sensors,
                device=device,
                amp=args.amp,
                grad_clip=args.grad_clip,
                rounds=rounds,
                mode=args.mode,
                image_size=args.image_size,
                log_interval_rounds=args.log_interval_rounds,
                validity_masked_reconstruction=(
                    args.validity_masked_reconstruction
                ),
                ranknet_enabled=args.ranknet_objective,
                ranknet_weight=args.ranknet_weight,
                ranknet_temperature=args.ranknet_temperature,
            )
            if args.mode == "supervised":
                full_validation_prediction_rows = (
                    []
                    if (
                        args.ranknet_objective
                        and args.max_val_batches == 0
                    )
                    else None
                )
                val = evaluate_supervised(
                    model,
                    val_loaders,
                    classification_losses,
                    sensors=sensors,
                    device=device,
                    amp=args.amp,
                    max_batches=args.max_val_batches,
                    image_size=args.image_size,
                    validity_masked_reconstruction=(
                        args.validity_masked_reconstruction
                    ),
                    prediction_rows=full_validation_prediction_rows,
                )
                macro = {
                    key: finite_macro(val, key)
                    for key in ("ap", "f1", "f1_0p5", "auroc", "loss")
                }
            else:
                val = evaluate_pretrain(
                    model,
                    val_loaders,
                    sensors=sensors,
                    device=device,
                    amp=args.amp,
                    max_batches=args.max_val_batches,
                    image_size=args.image_size,
                    validity_masked_reconstruction=(
                        args.validity_masked_reconstruction
                    ),
                )
                macro = {
                    "reconstruction_loss": finite_macro(val, "reconstruction_loss")
                }

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_losses,
                "val": val,
                "macro_over_sensor": macro,
                "elapsed_seconds": time.time() - epoch_started,
                "completed_unix": time.time(),
            }
            if (
                args.mode == "pretrain"
                and args.validity_masked_reconstruction
            ):
                epoch_record["train_validity_diagnostics"] = (
                    train_validity_diagnostics
                )
            if args.ranknet_objective:
                epoch_record["train_objective_diagnostics"] = (
                    train_objective_diagnostics
                )
            history["epochs"].append(json_safe(epoch_record))
            history["status"] = "running" if epoch + 1 < args.epochs else "completed"
            atomic_json_dump(json_safe(history), history_path)
            write_epoch_tables(history, args.output_dir)

            if not args.no_save_checkpoint:
                checkpoint_payload = {
                    "epoch": epoch,
                    "mode": args.mode,
                    "sharing": args.sharing,
                    "sensors": sensors,
                    "encoder_signature": encoder_signature,
                    "data_signature": data_signature,
                    "resume_signature": resume_signature,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": artifact_config(args),
                }
                if args.validity_masked_reconstruction:
                    checkpoint_payload["validity_reconstruction"] = (
                        build_validity_reconstruction_signature()
                    )
                if args.ranknet_objective:
                    checkpoint_payload["supervised_objective"] = (
                        build_ranknet_objective_signature()
                    )
                atomic_torch_save(
                    checkpoint_payload,
                    args.output_dir / "checkpoint_latest.pth",
                )
            ranknet_validation_artifact = None
            if (
                args.ranknet_objective
                and args.max_val_batches == 0
            ):
                if full_validation_prediction_rows is None:
                    raise AssertionError(
                        "RankNet full validation did not capture predictions."
                    )
                ranknet_validation_artifact = (
                    write_ranknet_full_validation_artifacts(
                        history=history,
                        history_path=history_path,
                        checkpoint_path=checkpoint_path,
                        output_dir=args.output_dir,
                        epoch=epoch,
                        per_sensor=val,
                        macro_over_sensor=macro,
                        prediction_rows=full_validation_prediction_rows,
                        runtime={
                            "device": str(device),
                            "batch_size": int(args.batch_size),
                            "num_workers": int(args.num_workers),
                            "amp": bool(args.amp),
                            "elapsed_seconds": epoch_record[
                                "elapsed_seconds"
                            ],
                            "captured_during_training_validation": True,
                        },
                    )
                )
            epoch_summary = {
                "epoch": epoch,
                "train_loss": train_losses,
                "val": val,
                "macro_over_sensor": macro,
                "elapsed_seconds": epoch_record["elapsed_seconds"],
            }
            if args.ranknet_objective:
                epoch_summary["ranknet_validation_artifact"] = (
                    ranknet_validation_artifact
                )
            print(
                "[Epoch] "
                + json.dumps(
                    json_safe(epoch_summary),
                    sort_keys=True,
                ),
                flush=True,
            )
    except BaseException as error:
        history["status"] = "failed"
        error_message = str(error).strip() or repr(error)
        history["error"] = f"{type(error).__name__}: {error_message}"
        history["error_type"] = type(error).__name__
        history["error_message"] = error_message
        history["traceback"] = traceback.format_exc()
        history["failed_unix"] = time.time()
        atomic_json_dump(json_safe(history), history_path)
        write_epoch_tables(history, args.output_dir)
        raise

    print(f"[Done] metrics={history_path}", flush=True)


if __name__ == "__main__":
    main()
