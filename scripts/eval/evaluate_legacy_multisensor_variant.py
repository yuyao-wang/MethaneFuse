#!/usr/bin/env python3
"""Evaluate legacy 480 m multi-sensor adapter variants on a wide-table CSV."""

from __future__ import annotations

import argparse
import gc
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
    balanced_accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
THIRDPARTY_ROOT = REPO_ROOT / "thirdparty"
if str(THIRDPARTY_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRDPARTY_ROOT))
DINOV2_ROOT = THIRDPARTY_ROOT / "dinov2"
if str(DINOV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DINOV2_ROOT))
PANOPTICON_HUB_ROOT = (
    Path.home() / ".cache" / "torch" / "hub" / "Panopticon-FM_panopticon_main"
)
if PANOPTICON_HUB_ROOT.is_dir() and str(PANOPTICON_HUB_ROOT) not in sys.path:
    # Keep the repository's extended ``dinov2`` package ahead of the cached
    # upstream checkout, while still exposing the cached ``hubconf.py``.
    sys.path.append(str(PANOPTICON_HUB_ROOT))

os.environ.setdefault("XFORMERS_DISABLED", "1")

from src.data.multisensor import (  # noqa: E402
    TriSensorTemporalCsvDataset,
    custom_collate_fn,
)
from src.data.sensor_transforms import (  # noqa: E402
    DEFAULT_WV3_BANDS,
    load_wv3_channel_ids_from_srf,
)
from src.utils.training import recursive_to_device  # noqa: E402


def build_model(variant: str, saved_args: dict, device: torch.device):
    if variant == "qv_lora":
        from experiments_tryout.universal_models_fusion import (
            multi_sensor_panopticon_4_lora_adapter as module,
        )

        model = module.MultiSensorPanopticonClassifier(
            backbone=module._load_backbone(saved_args["weights"]),
            adapter_blocks=int(saved_args["adapter_blocks"]),
            adapter_bottleneck_dim=int(saved_args["adapter_bottleneck_dim"]),
            adapter_alpha=float(saved_args["adapter_alpha"]),
            adapter_dropout=float(saved_args["adapter_dropout"]),
            enable_summary_head=bool(saved_args.get("summary_head", False)),
            summary_hidden_dim=int(saved_args.get("summary_hidden_dim", 128)),
            summary_dropout=float(saved_args.get("summary_dropout", 0.1)),
            summary_loss_weight=float(
                saved_args.get("summary_loss_weight", 1.0)
            ),
            row_fusion_mode=str(saved_args["row_fusion_mode"]),
        )
        return model.to(device)

    if variant == "postblock_moe":
        from experiments_tryout.universal_models_fusion import (
            multi_sensor_panopticon_4_nonlinear_moeadapter as module,
        )

        model = module.MultiSensorPanopticonClassifier(
            backbone=module._load_backbone(saved_args["weights"]),
            adapter_first_blocks=12,
            adapter_token_blocks=int(saved_args["adapter_token_blocks"]),
            adapter_bottleneck_dim=int(saved_args["adapter_bottleneck_dim"]),
            adapter_alpha=float(saved_args["adapter_alpha"]),
            adapter_dropout=float(saved_args["adapter_dropout"]),
            adapter_cls_only=bool(saved_args["adapter_cls_only"]),
            adapter_use_gelu=bool(saved_args["adapter_gelu"]),
            enable_summary_head=bool(saved_args.get("summary_head", False)),
            summary_hidden_dim=int(saved_args.get("summary_hidden_dim", 128)),
            summary_dropout=float(saved_args.get("summary_dropout", 0.1)),
            summary_loss_weight=float(
                saved_args.get("summary_loss_weight", 1.0)
            ),
            row_fusion_mode=str(saved_args["row_fusion_mode"]),
        )
        return model.to(device)

    raise ValueError(f"Unsupported variant: {variant}")


def metrics(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict:
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    return {
        "samples": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float((predictions == labels).mean()),
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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=("qv_lora", "postblock_moe"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval_csv", type=Path, required=True)
    parser.add_argument("--wv3_srf_csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--predictions_csv", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    saved_args = dict(checkpoint["args"])
    for key in ("train_csv", "test_csv"):
        source = str(saved_args.get(key, ""))
        if "_518" in source or "legacy_param_480m" not in source:
            raise ValueError(f"Rejected training data contract: {key}={source}")
    if any(
        key in checkpoint
        for key in ("threshold", "best_threshold", "decision_threshold")
    ):
        raise ValueError(
            "Checkpoint contains a threshold field; evaluator must explicitly support it"
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = build_model(args.variant, saved_args, device)
    state = checkpoint.pop("model")
    model.load_state_dict(state, strict=True)
    del state
    checkpoint.pop("optimizer", None)
    checkpoint.pop("scheduler", None)
    checkpoint.pop("scaler", None)
    gc.collect()
    model.eval()

    bands = [
        band.strip()
        for band in str(
            saved_args.get("wv3_bands", ",".join(DEFAULT_WV3_BANDS))
        ).split(",")
        if band.strip()
    ]
    wv3_chn_ids = load_wv3_channel_ids_from_srf(
        args.wv3_srf_csv, bands
    ).unsqueeze(-1)
    dataset = TriSensorTemporalCsvDataset(
        csv_path=str(args.eval_csv.resolve()),
        s5p_data_key=saved_args.get("s5p_data_key", "ch4"),
        s5p_chn_ids_key=saved_args.get("s5p_chn_ids_key", "chn_ids"),
        s5p_channels_last=bool(saved_args.get("s5p_channels_last", False)),
        align_l89_to_s2=bool(saved_args.get("align_l89_to_s2", False)),
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=custom_collate_fn,
    )

    labels_all = []
    predictions_all = []
    probabilities_all = []
    with torch.inference_mode():
        for x_dict, labels, sensors, sample_to_row in loader:
            labels = labels.to(device)
            sample_to_row = sample_to_row.to(device=device, dtype=torch.long)
            x_dict = recursive_to_device(x_dict, device)
            with autocast(enabled=device.type == "cuda"):
                outputs = model(x_dict, sensors=sensors)
                logits = model.compute_row_fused_logits(
                    outputs,
                    sample_to_row=sample_to_row,
                    num_rows=labels.size(0),
                    device=device,
                )
            labels_all.append(labels.cpu())
            predictions_all.append(logits.argmax(dim=1).cpu())
            probabilities_all.append(
                F.softmax(logits, dim=1)[:, 1].float().cpu()
            )

    labels = torch.cat(labels_all).numpy().astype(np.int64)
    predictions = torch.cat(predictions_all).numpy().astype(np.int64)
    probabilities = torch.cat(probabilities_all).numpy()
    frame = pd.read_csv(args.eval_csv, low_memory=False)
    csv_labels = frame["label"].astype(np.int64).to_numpy()
    if not np.array_equal(labels, csv_labels):
        raise ValueError("Prediction order differs from evaluation CSV")

    output_frame = frame[
        [
            column
            for column in (
                "id",
                "plume_id",
                "label",
                "latitude",
                "longitude",
                "datetime",
                "dx_anchor_px",
                "dy_anchor_px",
            )
            if column in frame.columns
        ]
    ].copy()
    output_frame["prediction"] = predictions
    output_frame["positive_probability"] = probabilities
    output_frame["correct"] = predictions == labels
    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(args.predictions_csv, index=False)

    result = {
        "variant": args.variant,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_test_accuracy": float(
            checkpoint.get("best_test_acc", float("nan"))
        ),
        "training_csv": saved_args["train_csv"],
        "validation_csv": saved_args["test_csv"],
        "eval_csv": str(args.eval_csv.resolve()),
        "row_fusion_mode": saved_args["row_fusion_mode"],
        "decision_rule": {
            "kind": "two_class_logit_argmax",
            "probability_equivalent_threshold": 0.5,
            "source": "original training/evaluation code; no fitted threshold stored",
        },
        "metrics": metrics(labels, predictions, probabilities),
        "predictions_csv": str(args.predictions_csv.resolve()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
