#!/usr/bin/env python
"""Evaluate MethaneFuse query-level classification checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("XFORMERS_DISABLED", "1")

from src.models.finetune_loramoe_adapter import (  # noqa: E402
    DEFAULT_WV3_BANDS,
    MultiSensorPanopticonClassifier,
    StaticAnchoredCache,
    TriSensorTemporalCsvDataset,
    _load_backbone,
    compute_split_metrics,
    custom_collate_fn,
    install_lora_moe_qv_adapters,
    load_model_checkpoint_flexible,
    load_wv3_channel_ids_from_srf,
    recursive_to_device,
)



def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value

def load_config(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    cfg = OmegaConf.load(path)
    return dict(OmegaConf.to_container(cfg, resolve=True))


def apply_config_defaults(parser: argparse.ArgumentParser, cfg: Mapping[str, Any]) -> None:
    valid = {action.dest for action in parser._actions}
    parser.set_defaults(**{k: v for k, v in cfg.items() if k in valid})


def build_parser(defaults: Optional[Mapping[str, Any]] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--eval_csv", required=defaults is None or "eval_csv" not in defaults)
    parser.add_argument("--checkpoint", required=defaults is None or "checkpoint" not in defaults)
    parser.add_argument("--weights", default="weights/panopticon_vitb14_teacher.pth")
    parser.add_argument("--stage", choices=("a", "b"), default="b")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--row_fusion_mode", choices=("map", "max"), default="max")
    parser.add_argument("--sensor_aux_loss_weight", type=float, default=0.3)
    parser.add_argument("--s5p_data_key", default="ch4")
    parser.add_argument("--s5p_chn_ids_key", default="chn_ids")
    parser.add_argument("--s5p_channels_last", action="store_true")
    parser.add_argument("--align_l89_to_s2", action="store_true")
    parser.add_argument("--wv3_srf_csv", default=str(REPO_ROOT / "WV3_VNIR_SWIR_response.csv"))
    parser.add_argument("--wv3_bands", default=",".join(DEFAULT_WV3_BANDS))
    parser.add_argument("--local_cache_dir", default=None)
    parser.add_argument("--local_cache_min_free_gb", type=float, default=5.0)
    if defaults:
        apply_config_defaults(parser, defaults)
    return parser


def parse_args() -> argparse.Namespace:
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", default=None)
    known, remaining = config_probe.parse_known_args()
    defaults = load_config(known.config)
    parser = build_parser(defaults)
    return parser.parse_args(remaining if known.config is None else ["--config", known.config, *remaining])


def main(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    wv3_band_names = [x.strip() for x in args.wv3_bands.split(",") if x.strip()]
    wv3_chn_ids = load_wv3_channel_ids_from_srf(args.wv3_srf_csv, wv3_band_names).unsqueeze(-1)
    cache_obj = None
    if args.local_cache_dir:
        cache_obj = StaticAnchoredCache(args.local_cache_dir, min_free_gb=args.local_cache_min_free_gb)

    eval_ds = TriSensorTemporalCsvDataset(
        csv_path=args.eval_csv,
        local_file_cache=cache_obj,
        s5p_data_key=args.s5p_data_key,
        s5p_chn_ids_key=args.s5p_chn_ids_key,
        s5p_channels_last=args.s5p_channels_last,
        align_l89_to_s2=args.align_l89_to_s2,
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=custom_collate_fn,
    )

    model = MultiSensorPanopticonClassifier(
        backbone=_load_backbone(args.weights),
        enable_summary_head=False,
        row_fusion_mode=args.row_fusion_mode,
    ).to(device)
    if args.stage == "b":
        installed = install_lora_moe_qv_adapters(
            model.backbone,
            num_experts=len(model.sensor_order),
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )
        print(
            f"[Stage B] Installed Q/V LoRA-MoE adapters in {installed} qkv modules "
            f"(experts={len(model.sensor_order)}, rank={args.lora_rank}, alpha={args.lora_alpha:g}).",
            flush=True,
        )
    load_model_checkpoint_flexible(Path(args.checkpoint), model, device)
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    labels_all: list[int] = []
    preds_all: list[int] = []
    scores_all: list[float] = []
    total_loss = 0.0
    total_rows = 0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for step, (x_dict, labels, sensors, sample_to_row) in enumerate(loader, 1):
            labels = labels.to(device)
            sample_to_row = sample_to_row.to(device=device, dtype=torch.long)
            x_dict = recursive_to_device(x_dict, device)
            with autocast(enabled=use_amp):
                outputs = model(x_dict, sensors=sensors)
                flat_labels = labels.index_select(0, sample_to_row)
                sensor_aux_loss, _ = model.loss_from_outputs(outputs, flat_labels, sensors, criterion)
                fused_logits = model.compute_row_fused_logits(
                    outputs,
                    sample_to_row=sample_to_row,
                    num_rows=labels.size(0),
                    device=device,
                )
                fused_loss = criterion(fused_logits, labels)
                loss = fused_loss + (args.sensor_aux_loss_weight * sensor_aux_loss)
            batch_rows = int(labels.size(0))
            total_loss += float(loss.item()) * batch_rows
            total_rows += batch_rows
            preds = fused_logits.argmax(dim=1)
            labels_all.extend(labels.detach().cpu().tolist())
            preds_all.extend(preds.detach().cpu().tolist())
            if fused_logits.shape[-1] == 2:
                scores_all.extend(torch.softmax(fused_logits, dim=1)[:, 1].detach().cpu().tolist())
            if args.max_eval_steps is not None and step >= args.max_eval_steps:
                break

    metrics = compute_split_metrics(labels_all, preds_all, scores_all)
    result = {
        "checkpoint": str(args.checkpoint),
        "eval_csv": str(args.eval_csv),
        "loss": total_loss / max(1, total_rows),
        "count": total_rows,
        "overall": metrics,
    }
    result = json_safe(result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main(parse_args())
