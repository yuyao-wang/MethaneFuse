#!/usr/bin/env python3
"""Evaluate a selected S2 checkpoint once on a sealed CSV.

The decision threshold is recovered from the checkpoint's validation epoch,
never optimized on the evaluation CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgraded_dataset.dino_classifier_head_s2_temporal_satmae import (
    CLSHead,
    S2TemporalSequenceDataset,
    TemporalPanopticonBackbone,
    flatten_temporal_channels,
    load_backbone,
    load_state_dict_flexible,
    parse_channel_indices,
    parse_csv_columns,
    recursive_to_device,
    resize_imgs,
)
from research.pretraining_20260727.temporal_input_modes import (
    output_timepoints,
    parse_residual_slots,
)


def load_validation_threshold(
    checkpoint: dict, metrics_path: Path
) -> tuple[float, str]:
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    history = json.loads(metrics_path.read_text(encoding="utf-8"))
    matches = [
        record
        for record in history
        if int(record.get("epoch", -2)) == checkpoint_epoch
        and record.get("test_best_f1_threshold") is not None
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely recover validation threshold for checkpoint "
            f"epoch={checkpoint_epoch} from {metrics_path}"
        )
    threshold = float(matches[0]["test_best_f1_threshold"])
    if not np.isfinite(threshold):
        raise ValueError(f"Non-finite validation threshold: {threshold}")
    return threshold, f"{metrics_path}:epoch={checkpoint_epoch}"


def metrics_at_threshold(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (targets == 1)).sum())
    fp = int(((predictions == 1) & (targets == 0)).sum())
    fn = int(((predictions == 0) & (targets == 1)).sum())
    tn = int(((predictions == 0) & (targets == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / max(1, len(targets))
    positive_f1 = f1_score(targets, predictions, zero_division=0)
    macro_f1 = f1_score(
        targets, predictions, average="macro", zero_division=0
    )
    roc_fpr, roc_tpr, _ = roc_curve(targets, probabilities)
    within_01 = roc_tpr[roc_fpr <= 0.01]
    within_05 = roc_tpr[roc_fpr <= 0.05]
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "f1": float(positive_f1),
        "macro_f1": float(macro_f1),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--predictions_csv", type=Path)
    parser.add_argument("--metrics_history", type=Path)
    parser.add_argument("--normalization_stats", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_args = dict(checkpoint["args"])
    run_dir = args.checkpoint.parent
    stats_path = args.normalization_stats or run_dir / "normalization_stats.json"
    metrics_path = args.metrics_history or run_dir / "metrics_history.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    if args.threshold is None:
        threshold, threshold_source = load_validation_threshold(
            checkpoint, metrics_path
        )
    else:
        threshold = float(args.threshold)
        threshold_source = "explicit_cli"

    path_columns = parse_csv_columns(saved_args["path_columns"])
    time_columns = parse_csv_columns(saved_args["time_columns"])
    residual_slots = parse_residual_slots(saved_args["residual_slots"])
    channel_indices = parse_channel_indices(saved_args["channel_indices"])
    input_resize_size = int(saved_args.get("input_resize_size", 0))
    dataset = S2TemporalSequenceDataset(
        csv_path=str(args.eval_csv),
        path_columns=path_columns,
        time_columns=time_columns,
        normalize_stats=(stats["mean"], stats["std"]),
        time_base_year=int(saved_args.get("time_base_year", 2000)),
        pad_to_multiple=(
            None if input_resize_size > 0 else int(saved_args.get("pad_to_multiple", 14))
        ),
        skip_invalid_samples=False,
        local_file_cache=None,
        cache_prefetch_rows=0,
        channel_indices=channel_indices,
        input_mode=saved_args.get("input_mode", "raw"),
        residual_slots=residual_slots,
        residual_clip=float(saved_args.get("residual_clip", 5.0)),
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    panopticon = load_backbone(saved_args["weights"], device=device)
    if saved_args["temporal_fusion"] == "satmae":
        backbone = TemporalPanopticonBackbone(
            panopticon,
            num_timepoints=output_timepoints(
                saved_args.get("input_mode", "raw"),
                len(path_columns),
                residual_slots,
            ),
            time_embed_dim=int(saved_args.get("time_embed_dim", 384)),
        ).to(device)
    else:
        backbone = panopticon.to(device)
    head = CLSHead(
        embed_dim=int(saved_args.get("embed_dim", 768)), num_classes=2
    ).to(device)
    load_state_dict_flexible(backbone, checkpoint["backbone"])
    load_state_dict_flexible(head, checkpoint["head"])
    backbone.eval()
    head.eval()

    probabilities = []
    targets = []
    with torch.inference_mode():
        for x_dict, labels in loader:
            x_dict = recursive_to_device(x_dict, device)
            x_dict = resize_imgs(x_dict, input_resize_size)
            if saved_args["temporal_fusion"] == "concat_channels":
                x_dict = flatten_temporal_channels(x_dict)
            with autocast(enabled=device.type == "cuda"):
                features = backbone(x_dict, is_training=True)
                logits = head(features["x_norm_clstoken"])
            probabilities.append(
                F.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
            )
            targets.append(labels.numpy())
    probability_array = np.concatenate(probabilities)
    target_array = np.concatenate(targets).astype(np.int64)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "eval_csv": str(args.eval_csv.resolve()),
        "threshold_source": threshold_source,
        "input_mode": saved_args.get("input_mode", "raw"),
        "temporal_fusion": saved_args["temporal_fusion"],
        "metrics": metrics_at_threshold(
            target_array, probability_array, threshold
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(
        args.output_json.suffix + f".part.{os.getpid()}"
    )
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_json)

    if args.predictions_csv is not None:
        frame = pd.read_csv(args.eval_csv, low_memory=False)
        if len(frame) != len(probability_array):
            raise ValueError(
                f"Prediction count {len(probability_array)} != CSV rows {len(frame)}"
            )
        frame = frame.copy()
        frame["methanefuse_probability"] = probability_array
        frame["methanefuse_prediction"] = (
            probability_array >= threshold
        ).astype(np.int64)
        args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        temporary_predictions = args.predictions_csv.with_suffix(
            args.predictions_csv.suffix + f".part.{os.getpid()}"
        )
        frame.to_csv(temporary_predictions, index=False)
        os.replace(temporary_predictions, args.predictions_csv)

    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
