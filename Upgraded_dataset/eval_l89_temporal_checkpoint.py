#!/usr/bin/env python3
"""Evaluate a trained L89 temporal checkpoint on one CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgraded_dataset.dino_classifier_head_l89_temporal_satmae import (  # noqa: E402
    CLSHead,
    PRECOMPUTED_STATS,
    StaticAnchoredCache,
    TemporalPanopticonBackbone,
    build_dataset,
    load_backbone,
    load_state_dict_flexible,
    parse_csv_columns,
    recursive_to_device,
)


class ArgsProxy:
    pass


def compute_metrics(targets: torch.Tensor, probs: torch.Tensor, preds: torch.Tensor) -> dict:
    targets = targets.to(torch.long)
    preds = preds.to(torch.long)
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    total = int(targets.numel())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1_den = 2 * tp + fp + fn
    f1 = 2 * tp / f1_den if f1_den else 0.0
    acc = (tp + tn) / total if total else 0.0
    try:
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(targets.numpy(), probs.numpy()))
    except Exception:
        auroc = float("nan")
    return {
        "count": total,
        "acc": acc,
        "f1": f1,
        "recall": recall,
        "fpr": fpr,
        "auroc": auroc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_csv", required=True, help="Used only to match dataset/stat args.")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--weights", default="weights/panopticon_vitb14_teacher.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--path_columns", default="path_t0,path_prev1,path_prev2,path_prev3,path_seasonal,path_year")
    parser.add_argument(
        "--time_columns",
        default="t0_image_time,prev1_image_time,prev2_image_time,prev3_image_time,seasonal_image_time,year_image_time",
    )
    parser.add_argument("--time_base_year", type=int, default=2000)
    parser.add_argument("--time_embed_dim", type=int, default=384)
    parser.add_argument("--pad_to_multiple", type=int, default=14)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--output_predictions_csv",
        default=None,
        help="Optional row-aligned CSV containing probability, prediction, and correctness.",
    )
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--band_indices", default="0,1,2,3,4,5,6")
    parser.add_argument("--local_cache_dir", default="/diniuvol/yuyao/l89_temporal_cache")
    parser.add_argument("--local_cache_mode", choices=["off", "sync"], default="off")
    parser.add_argument("--local_cache_min_free_gb", type=float, default=200.0)
    args = parser.parse_args()

    path_columns = parse_csv_columns(args.path_columns)
    time_columns = parse_csv_columns(args.time_columns)
    device = torch.device(args.device)
    if device.type == "cuda":
        visible_count = torch.cuda.device_count()
        requested_index = 0 if device.index is None else device.index
        if not torch.cuda.is_available() or requested_index >= visible_count:
            raise SystemExit(
                f"Requested {args.device}, but PyTorch sees {visible_count} CUDA device(s). "
                "Use --device cuda:0, or set CUDA_VISIBLE_DEVICES to the physical GPU you want "
                "and still use --device cuda:0 inside Python."
            )
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    use_amp = device.type == "cuda"

    ds_args = ArgsProxy()
    ds_args.path_columns = args.path_columns
    ds_args.time_columns = args.time_columns
    ds_args.time_base_year = args.time_base_year
    ds_args.pad_to_multiple = args.pad_to_multiple
    ds_args.skip_invalid_samples = False
    ds_args.cache_prefetch_rows = 0
    ds_args.band_indices = args.band_indices
    cache_obj = None
    if args.local_cache_mode != "off" and args.local_cache_dir:
        cache_obj = StaticAnchoredCache(
            args.local_cache_dir,
            min_free_gb=args.local_cache_min_free_gb,
        )
    test_ds = build_dataset(ds_args, args.test_csv, cache_obj, PRECOMPUTED_STATS)

    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    panopticon = load_backbone(args.weights, device=device, debug=False)
    backbone = TemporalPanopticonBackbone(
        panopticon,
        num_timepoints=len(path_columns),
        time_embed_dim=args.time_embed_dim,
    ).to(device)
    head = CLSHead(embed_dim=args.embed_dim, num_classes=2).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    load_state_dict_flexible(backbone, ckpt["backbone"])
    load_state_dict_flexible(head, ckpt["head"])
    backbone.eval()
    head.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    loss_total = 0.0
    count = 0
    probs_all = []
    targets_all = []
    preds_all = []
    with torch.no_grad():
        total_batches = len(test_loader)
        for step, (x_dict, labels) in enumerate(test_loader, 1):
            labels = labels.to(device)
            x_dict = recursive_to_device(x_dict, device)
            with autocast(enabled=use_amp):
                feats = backbone(x_dict, is_training=True)
                logits = head(feats["x_norm_clstoken"])
                loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            loss_total += float(loss.item()) * int(labels.size(0))
            count += int(labels.size(0))
            probs_all.append(probs.detach().cpu())
            targets_all.append(labels.detach().cpu())
            preds_all.append(preds.detach().cpu())
            if args.max_eval_steps is not None and step >= args.max_eval_steps:
                break
            if args.progress_every > 0 and step % args.progress_every == 0:
                print(
                    f"[Eval] {step}/{total_batches} batches, rows={count}",
                    flush=True,
                )

    probs_all = torch.cat(probs_all)
    targets_all = torch.cat(targets_all)
    preds_all = torch.cat(preds_all)
    metrics = compute_metrics(targets_all, probs_all, preds_all)
    metrics["loss"] = loss_total / count if count else float("nan")
    metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
    metrics["test_csv"] = str(Path(args.test_csv).resolve())

    print(
        "Eval: "
        f"loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} f1={metrics['f1']:.4f} "
        f"recall={metrics['recall']:.4f} fpr={metrics['fpr']:.4f} auroc={metrics['auroc']:.4f} "
        f"count={metrics['count']}",
        flush=True,
    )
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    if args.output_predictions_csv:
        source_df = pd.read_csv(args.test_csv, low_memory=False).iloc[: metrics["count"]].copy()
        source_labels = torch.as_tensor(source_df["label"].to_numpy(), dtype=torch.long)
        if source_labels.shape != targets_all.shape or not torch.equal(source_labels, targets_all):
            raise RuntimeError(
                "Prediction export requires row-aligned evaluation, but evaluated labels do not match the test CSV."
            )
        source_df["pred_prob1"] = probs_all.numpy()
        source_df["pred_label"] = preds_all.numpy()
        source_df["pred_correct"] = source_df["pred_label"].to_numpy() == source_df["label"].to_numpy()
        output_predictions = Path(args.output_predictions_csv)
        output_predictions.parent.mkdir(parents=True, exist_ok=True)
        source_df.to_csv(output_predictions, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
