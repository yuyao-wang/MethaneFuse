#!/usr/bin/env python3
"""Strict evaluation for the legacy three-timepoint S2-only classifiers.

The target checkpoints were trained by ``baselines/per_vit/s2_temporal.py``:
three normalized 12-band frames are concatenated in [t0, t-90, t-360] order
and passed to the Panopticon ViT-B/14 plus a two-class CLS head.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("XFORMERS_DISABLED", "1")

from baselines.per_vit.s2_temporal import (  # noqa: E402
    CLSHead,
    ConcatTemporalDataset,
    PRECOMPUTED_STATS,
    recursive_to_device,
)
from src.backbones import build_panopticon_vitb14  # noqa: E402
from thirdparty.dinov2.data.datasets.s2_csv import (  # noqa: E402
    S2TemporalCsvDataset,
)

PATH_COLUMNS = ("s2_0_path", "s2_90_path", "s2_360_path")
LEGACY_LAYOUT = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8A",
    "ZERO_COMPAT_1",
    "ZERO_COMPAT_2",
    "B11",
    "B12",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def validate_audits_and_manifest(
    eval_csv: Path,
    audit_paths: list[Path],
    expected_scale_m: int,
    expected_rows: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frame = pd.read_csv(eval_csv, low_memory=False)
    required = {"id", "label", *PATH_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {missing}")
    if len(frame) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} evaluation rows, found {len(frame)}"
        )
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("Evaluation CSV has duplicate IDs")
    labels = set(frame["label"].astype(int).unique().tolist())
    if labels != {0, 1}:
        raise ValueError(f"Expected both original labels 0 and 1, found {labels}")

    audits = []
    audited_frames = []
    for path in audit_paths:
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("pass") is not True:
            raise ValueError(f"Input-contract audit did not pass: {path}")
        contract = audit.get("contract", {})
        if int(contract.get("crop", {}).get("scale_name_m", -1)) != expected_scale_m:
            raise ValueError(f"Scale mismatch in input-contract audit: {path}")
        loader = contract.get("loader", {})
        if loader.get("loaded_frame_shape") != [12, 224, 224]:
            raise ValueError(f"Unexpected loaded frame shape in {path}")
        if int(loader.get("temporal_frames", -1)) != 3:
            raise ValueError(f"Unexpected temporal frame count in {path}")
        if int(loader.get("pad_to_multiple", -1)) != 14:
            raise ValueError(f"Unexpected loader padding contract in {path}")
        if contract.get("legacy_channel_layout") != LEGACY_LAYOUT:
            raise ValueError(f"Unexpected S2 channel layout in {path}")
        if contract.get("temporal_order") != [
            "t0",
            "minus_90_day_selection",
            "minus_360_day_selection",
        ]:
            raise ValueError(f"Unexpected temporal order in {path}")
        candidate_manifest = Path(audit["candidate_manifest"]).resolve()
        if not candidate_manifest.is_file():
            raise FileNotFoundError(candidate_manifest)
        audited_frames.append(pd.read_csv(candidate_manifest, low_memory=False))
        audits.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "candidate_manifest": str(candidate_manifest),
                "contract_sha256": audit["contract_sha256"],
                "rows": int(audit["rows"]),
                "max_recompute_abs_diff": float(
                    audit["max_recompute_abs_diff"]
                ),
                "loader_normalization_max_abs_diff": float(
                    audit["loader_normalization_max_abs_diff"]
                ),
            }
        )

    if audits:
        if len(audits) != 2:
            raise ValueError(
                "Provide exactly two audits (label-1 and label-0), or neither"
            )
        contract_hashes = {item["contract_sha256"] for item in audits}
        if len(contract_hashes) != 1:
            raise ValueError(
                f"Positive/negative input contracts differ: {contract_hashes}"
            )

        audited = pd.concat(audited_frames, ignore_index=True)
        if len(audited) != len(frame):
            raise ValueError(
                f"Audited rows ({len(audited)}) != evaluation rows ({len(frame)})"
            )
        comparison_columns = ["id", "label", *PATH_COLUMNS]
        expected_records = sorted(
            tuple(str(value) for value in row)
            for row in audited[comparison_columns].itertuples(
                index=False, name=None
            )
        )
        actual_records = sorted(
            tuple(str(value) for value in row)
            for row in frame[comparison_columns].itertuples(
                index=False, name=None
            )
        )
        if actual_records != expected_records:
            raise ValueError(
                "Evaluation rows/labels/paths do not exactly equal the two audited manifests"
            )

    for column in PATH_COLUMNS:
        for raw_path in frame[column].astype(str):
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            with tifffile.TiffFile(path) as handle:
                shape = tuple(handle.series[0].shape)
                dtype = str(handle.series[0].dtype)
            if shape != (12, 224, 224) or dtype != "float32":
                raise ValueError(
                    f"Input violates stored TIFF contract: {path}, "
                    f"shape={shape}, dtype={dtype}"
                )

    return frame, audits


def validate_checkpoint_args(
    checkpoint: dict[str, Any], expected_scale_m: int
) -> dict[str, Any]:
    required_keys = {"backbone", "head", "args", "epoch"}
    missing = sorted(required_keys - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {missing}")
    saved_args = dict(checkpoint["args"])
    train_csv = str(saved_args.get("train_csv", ""))
    test_csv = str(saved_args.get("test_csv", ""))
    expected_dataset_name = f"legacy_param_{expected_scale_m}m"
    for csv_path in (train_csv, test_csv):
        if "_518" in csv_path:
            raise ValueError(f"Refusing excluded 518 dataset checkpoint: {csv_path}")
        if expected_dataset_name not in csv_path:
            raise ValueError(
                f"Checkpoint dataset is not {expected_dataset_name}: {csv_path}"
            )
    expected = {
        "pad_to_multiple": 14,
        "t0_col": "path_t0",
        "t90_col": "path_t90",
        "t360_col": "path_t360",
        "embed_dim": 768,
        "train_backbone": True,
    }
    mismatches = {
        key: {"expected": value, "actual": saved_args.get(key)}
        for key, value in expected.items()
        if saved_args.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint input/model contract mismatch: {mismatches}")
    return saved_args


def compute_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    metrics = {
        "threshold": float(threshold),
        "samples": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float((tp + tn) / len(labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "recall": float(tp / positives),
        "specificity": float(tn / negatives),
        "fpr": float(fp / negatives),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "positive_probability_mean": float(probabilities[labels == 1].mean()),
        "negative_probability_mean": float(probabilities[labels == 0].mean()),
        "positive_probability_median": float(
            np.median(probabilities[labels == 1])
        ),
        "negative_probability_median": float(
            np.median(probabilities[labels == 0])
        ),
    }
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval_csv", type=Path, required=True)
    parser.add_argument("--positive_audit", type=Path)
    parser.add_argument("--negative_audit", type=Path)
    parser.add_argument("--expected_scale_m", type=int, required=True)
    parser.add_argument("--expected_rows", type=int, default=57)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--predictions_csv", type=Path, required=True)
    args = parser.parse_args()

    eval_csv = args.eval_csv.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if (args.positive_audit is None) != (args.negative_audit is None):
        raise ValueError(
            "--positive_audit and --negative_audit must be supplied together"
        )
    audit_paths = []
    if args.positive_audit is not None and args.negative_audit is not None:
        audit_paths = [
            args.positive_audit.resolve(),
            args.negative_audit.resolve(),
        ]
    source_frame, audits = validate_audits_and_manifest(
        eval_csv,
        audit_paths,
        args.expected_scale_m,
        args.expected_rows,
    )

    contract_mode = (
        "two_manifest_exact_union"
        if audits
        else "standalone_manifest_structure"
    )
    print(
        f"[Contract] {contract_mode} passed for {len(source_frame)} rows",
        flush=True,
    )
    print(f"[Checkpoint] loading {checkpoint_path}", flush=True)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    saved_args = validate_checkpoint_args(checkpoint, args.expected_scale_m)

    backbone = build_panopticon_vitb14()
    head = CLSHead(embed_dim=768, num_classes=2)
    backbone_state = checkpoint.pop("backbone")
    head_state = checkpoint.pop("head")
    backbone.load_state_dict(backbone_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    del backbone_state, head_state
    checkpoint.pop("optimizer", None)
    checkpoint.pop("scheduler", None)
    gc.collect()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    backbone = backbone.to(device).eval()
    head = head.to(device).eval()

    base_dataset = S2TemporalCsvDataset(
        csv_path=str(eval_csv),
        ds_cfg_name="s2_12band",
        normalize_stats=PRECOMPUTED_STATS,
        scale_to_unit=False,
        pad_to_multiple=14,
        compute_stats=False,
        path_columns=PATH_COLUMNS,
        skip_invalid_samples=False,
    )
    dataset = ConcatTemporalDataset(base_dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    probability_batches = []
    target_batches = []
    with torch.inference_mode():
        for x_dict, labels in loader:
            if tuple(x_dict["imgs"].shape[1:]) != (36, 224, 224):
                raise ValueError(
                    f"Unexpected concatenated model input: {x_dict['imgs'].shape}"
                )
            x_dict = recursive_to_device(x_dict, device)
            features = backbone(x_dict, is_training=True)
            logits = head(features["x_norm_clstoken"])
            probability_batches.append(
                F.softmax(logits, dim=1)[:, 1].float().cpu()
            )
            target_batches.append(labels.to(dtype=torch.int64).cpu())

    probabilities = torch.cat(probability_batches).numpy()
    labels = torch.cat(target_batches).numpy()
    csv_labels = source_frame["label"].astype(np.int64).to_numpy()
    if not np.array_equal(labels, csv_labels):
        raise ValueError("DataLoader labels do not preserve evaluation CSV order")
    metrics, predictions = compute_metrics(
        labels, probabilities, args.threshold
    )

    prediction_frame = source_frame.copy()
    prediction_frame["s2_only_probability"] = probabilities
    prediction_frame["s2_only_prediction"] = predictions
    prediction_frame["correct"] = predictions == labels
    atomic_write_csv(prediction_frame, args.predictions_csv.resolve())

    result = {
        "model_kind": "local_s2_only_panopticon_vitb14_cls",
        "scale_m": args.expected_scale_m,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint.get("epoch", -1)),
            "global_step": int(checkpoint.get("global_step", -1)),
            "best_train_accuracy": float(
                checkpoint.get("best_train_acc", float("nan"))
            ),
            "best_test_accuracy": float(
                checkpoint.get("best_test_acc", float("nan"))
            ),
            "training_csv": saved_args["train_csv"],
            "validation_csv": saved_args["test_csv"],
        },
        "input_contract": {
            "eval_csv": str(eval_csv),
            "eval_csv_sha256": sha256_file(eval_csv),
            "validation_mode": contract_mode,
            "audits": audits,
            "temporal_path_columns": list(PATH_COLUMNS),
            "temporal_order": ["t0", "t-90", "t-360"],
            "per_frame_shape": [12, 224, 224],
            "concatenated_shape": [36, 224, 224],
            "normalization": "legacy S2 PRECOMPUTED_STATS on stored DN",
            "scale_to_unit": False,
            "pad_to_multiple": 14,
            "precision": "float32",
            "excluded_dataset_suffix": "_518",
        },
        "metrics": metrics,
        "predictions_csv": str(args.predictions_csv.resolve()),
    }
    atomic_write_json(result, args.output_json.resolve())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
