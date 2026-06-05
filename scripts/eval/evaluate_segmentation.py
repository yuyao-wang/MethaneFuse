#!/usr/bin/env python
"""Evaluate MethaneFuse segmentation checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("XFORMERS_DISABLED", "1")

from src.data.multisensor import StaticAnchoredCache, TriSensorTemporalCsvDataset  # noqa: E402
from src.data.segmentation import SingleSensorSegmentationDataset, parse_tasks, segmentation_collate_fn  # noqa: E402
from src.data.sensor_transforms import DEFAULT_WV3_BANDS, load_wv3_channel_ids_from_srf  # noqa: E402
from src.models.segmentation import PanopticonSegmentationModel, run_eval_epoch  # noqa: E402
from src.utils.training import _load_backbone  # noqa: E402



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
    parser.add_argument("--checkpoint", default=None, help="Single checkpoint path used when evaluating one task.")
    parser.add_argument("--checkpoint_dir", default=None, help="Directory containing <run_name>__<task>/ckpt_best_val_iou_plus.pth files.")
    parser.add_argument("--run_name", default=None, help="Base run name used with --checkpoint_dir.")
    parser.add_argument("--weights", default="weights/panopticon_vitb14_teacher.pth")
    parser.add_argument("--tasks", default="s2,l89,emit")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--eval_threshold", type=float, default=0.5)
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


def resolve_checkpoint(args: argparse.Namespace, task_name: str) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    if not args.checkpoint_dir or not args.run_name:
        raise ValueError("Provide either --checkpoint or both --checkpoint_dir and --run_name.")
    return Path(args.checkpoint_dir) / f"{args.run_name}__{task_name}" / "ckpt_best_val_iou_plus.pth"


def main(args: argparse.Namespace) -> dict[str, Any]:
    tasks = parse_tasks(args.tasks)
    if args.checkpoint and len(tasks) != 1:
        raise ValueError("--checkpoint can only be used with a single task. Use --checkpoint_dir/--run_name for multiple tasks.")

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

    base_ds = TriSensorTemporalCsvDataset(
        csv_path=args.eval_csv,
        local_file_cache=cache_obj,
        align_l89_to_s2=args.align_l89_to_s2,
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )

    results: dict[str, Any] = {}
    for task in tasks:
        task_ds = SingleSensorSegmentationDataset(base_ds, task)
        loader = DataLoader(
            task_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=segmentation_collate_fn,
        )
        model = PanopticonSegmentationModel(backbone=_load_backbone(args.weights)).to(device)
        checkpoint = resolve_checkpoint(args, task.name)
        ckpt = torch.load(checkpoint, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, Mapping) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        metrics = run_eval_epoch(
            model=model,
            loader=loader,
            device=device,
            use_amp=device.type == "cuda",
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            eval_threshold=args.eval_threshold,
            max_steps=args.max_eval_steps,
        )
        results[task.name] = {
            "checkpoint": str(checkpoint),
            "count": int(metrics["count"]),
            "loss": float(metrics["loss"]),
            "bce": float(metrics["bce"]),
            "dice": float(metrics["dice"]),
            "iou_plus": float(metrics["iou_plus"]),
        }

    output = {"eval_csv": str(args.eval_csv), "tasks": results}
    output = json_safe(output)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    main(parse_args())
