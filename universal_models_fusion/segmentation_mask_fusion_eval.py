"""Evaluate late fusion for sensor-specific segmentation checkpoints.

The script loads one segmentation checkpoint per sensor, predicts probability
masks on rows where multiple sensor masks are available, aligns all predicted
masks to one reference mask grid, fuses them, and reports IoU+.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from universal_models_fusion.multi_sensor_panopticon_4 import (  # noqa: E402
    DEFAULT_WV3_BANDS,
    StaticAnchoredCache,
    TriSensorTemporalCsvDataset,
    load_wv3_channel_ids_from_srf,
    recursive_to_device,
)
from universal_models_fusion.multi_sensor_panopticon_4_segmentation import (  # noqa: E402
    TASK_CONFIGS,
    PanopticonSegmentationModel,
    SingleSensorSegmentationDataset,
    TaskConfig,
    parse_tasks,
)
from universal_models_fusion.unet_multisensor_baseline_iou_plus import UNetBaseline  # noqa: E402


def _is_missing(value: object) -> bool:
    text = str(value).strip().lower()
    return text in ("", "nan", "none", "null")


def _sample_key(row, row_idx: int) -> str:
    value = row.get("id", row_idx)
    if _is_missing(value):
        return str(row_idx)
    return str(value)


def _to_2d_mask(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[0] == 1:
            return a[0]
        if a.shape[-1] == 1:
            return a[..., 0]
        return a[0]
    raise ValueError(f"Expected 2D/single-band mask, got shape={a.shape}")


def load_binary_mask(path: str) -> np.ndarray:
    arr = _to_2d_mask(tiff.imread(path))
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return (arr > 0).astype(np.float32)


def resize_array(arr: np.ndarray, shape: Tuple[int, int], *, mode: str) -> np.ndarray:
    if tuple(arr.shape) == tuple(shape):
        return arr.astype(np.float32, copy=False)
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    out = F.interpolate(t, size=shape, **kwargs).squeeze(0).squeeze(0)
    return out.cpu().numpy().astype(np.float32, copy=False)


def geospatial_reproject_or_resize(
    arr: np.ndarray,
    *,
    src_ref_path: str,
    dst_ref_path: str,
    dst_shape: Tuple[int, int],
    resampling: str,
) -> Tuple[np.ndarray, str]:
    if tuple(arr.shape) != tuple(dst_shape):
        fallback = resize_array(arr, dst_shape, mode="bilinear" if resampling != "nearest" else "nearest")
    else:
        fallback = arr.astype(np.float32, copy=False)

    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        with rasterio.open(src_ref_path) as src_ds, rasterio.open(dst_ref_path) as dst_ds:
            if src_ds.crs is None or dst_ds.crs is None:
                return fallback, "resize_no_crs"
            src_transform = src_ds.transform
            dst_transform = dst_ds.transform
            if src_transform.is_identity or dst_transform.is_identity:
                return fallback, "resize_identity_transform"
            dst = np.zeros(dst_shape, dtype=np.float32)
            reproject(
                source=np.asarray(arr, dtype=np.float32),
                destination=dst,
                src_transform=src_transform,
                src_crs=src_ds.crs,
                dst_transform=dst_transform,
                dst_crs=dst_ds.crs,
                resampling=getattr(Resampling, resampling),
                src_nodata=0,
                dst_nodata=0,
            )
            return dst, "reproject"
    except Exception:
        return fallback, "resize_fallback"


def iou_plus(pred: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(pred) > 0
    g = np.asarray(target) > 0
    if not g.any():
        return 1.0 if not p.any() else 0.0
    union = np.logical_or(p, g).sum()
    if union <= 0:
        return 0.0
    return float(np.logical_and(p, g).sum() / union)


def resolve_checkpoint(args, task_name: str) -> Path:
    explicit = getattr(args, f"{task_name}_ckpt")
    if explicit:
        return Path(explicit)
    if args.run_name is None:
        raise ValueError(f"Need --run_name or --{task_name}_ckpt for task={task_name}")
    return Path(args.checkpoint_dir) / f"{args.run_name}__{task_name}" / args.checkpoint_name


def load_model_for_task(args, task: TaskConfig, ds: SingleSensorSegmentationDataset, device: torch.device) -> nn.Module:
    ckpt_path = resolve_checkpoint(args, task.name)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint for {task.name}: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if args.model_type == "unet":
        sample_x, _, _ = ds[0]
        model: nn.Module = UNetBaseline(
            in_channels=int(sample_x["imgs"].shape[0]),
            base_channels=int(args.base_channels),
        )
    elif args.model_type == "panopticon":
        from universal_models_fusion.multi_sensor_panopticon_4 import _load_backbone

        model = PanopticonSegmentationModel(backbone=_load_backbone(args.weights))
    else:
        raise ValueError(f"Unsupported --model_type={args.model_type}")

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_task_probs(
    *,
    args,
    task: TaskConfig,
    model: nn.Module,
    ds: SingleSensorSegmentationDataset,
    device: torch.device,
    allowed_keys: Optional[set[str]] = None,
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    use_amp = device.type == "cuda" and (not args.no_amp)
    for ds_idx, row_idx in enumerate(ds.indices):
        row = ds.base_ds.df.iloc[row_idx]
        key = _sample_key(row, row_idx)
        if allowed_keys is not None and key not in allowed_keys:
            continue
        x_dict, _mask, sample_id = ds[ds_idx]
        x_dict = {k: v.unsqueeze(0) for k, v in x_dict.items()}
        x_dict = recursive_to_device(x_dict, device)
        with autocast(enabled=use_amp):
            logits = model(x_dict)
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().float().cpu().numpy()

        mask_path = str(row[task.mask_column]).strip()
        mask_path = ds.base_ds._maybe_cache_path(mask_path)
        original_mask = load_binary_mask(mask_path)
        probs = resize_array(probs, original_mask.shape, mode="bilinear")

        out[key] = {
            "row_idx": int(row_idx),
            "sample_id": str(sample_id),
            "task": task.name,
            "prob": probs,
            "own_iou_plus": iou_plus(probs >= float(args.threshold), original_mask),
            "mask_path": mask_path,
        }
    return out


def collect_candidate_rows(base_ds: TriSensorTemporalCsvDataset, tasks: Sequence[TaskConfig], min_sensors: int) -> List[str]:
    keys: List[str] = []
    for row_idx, row in base_ds.df.iterrows():
        present = 0
        for task in tasks:
            if task.mask_column in base_ds.df.columns and not _is_missing(row.get(task.mask_column, "")):
                present += 1
        if present >= min_sensors:
            keys.append(_sample_key(row, int(row_idx)))
    return keys


def choose_reference_task(row, available: Sequence[str], target: str) -> str:
    if target == "first":
        return available[0]
    if target == "anchor":
        anchor = str(row.get("anchor_sensor", "")).strip().lower()
        if anchor in available:
            return anchor
        return available[0]
    if target not in available:
        raise ValueError(f"Requested reference task '{target}' is unavailable for row id={row.get('id', '')}")
    return target


def fuse_probs(probs: Sequence[np.ndarray], method: str, threshold: float) -> np.ndarray:
    stack = np.stack([np.asarray(p, dtype=np.float32) for p in probs], axis=0)
    if method == "mean":
        return stack.mean(axis=0)
    if method == "max":
        return stack.max(axis=0)
    if method == "vote":
        return (stack >= threshold).mean(axis=0)
    raise ValueError(f"Unsupported fusion method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Late-fusion evaluation for sensor-specific segmentation outputs.")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--tasks", default="s2,l89,emit", help="Comma-separated subset of: s2,l89,emit")
    parser.add_argument("--model_type", choices=["unet", "panopticon"], default="unet")
    parser.add_argument("--checkpoint_dir", default="checkpoints/unet_multisensor_baseline")
    parser.add_argument("--run_name", default=None, help="Base run name; checkpoints are <run_name>__<task>/ckpt_best...")
    parser.add_argument("--checkpoint_name", default="ckpt_best_val_iou_plus.pth")
    parser.add_argument("--s2_ckpt", default=None)
    parser.add_argument("--l89_ckpt", default=None)
    parser.add_argument("--emit_ckpt", default=None)
    parser.add_argument("--weights", default="weights/panopticon_vitb14_teacher.pth")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--fusion", choices=["mean", "max", "vote"], default="mean")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--reference", choices=["anchor", "first", "s2", "l89", "emit"], default="anchor")
    parser.add_argument("--min_sensors", type=int, default=2)
    parser.add_argument("--reproject_resampling", choices=["nearest", "bilinear"], default="bilinear")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--local_cache_dir", default=None)
    parser.add_argument("--local_cache_min_free_gb", type=float, default=5.0)
    parser.add_argument("--align_l89_to_s2", action="store_true")
    parser.add_argument(
        "--wv3_srf_csv",
        default=str(REPO_ROOT / "WV3_VNIR_SWIR_response.csv"),
        help="Path to WV3 SRF CSV for emit channel IDs.",
    )
    parser.add_argument("--wv3_bands", default=",".join(DEFAULT_WV3_BANDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = parse_tasks(args.tasks)
    task_by_name = {t.name: t for t in tasks}
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
        csv_path=args.test_csv,
        local_file_cache=cache_obj,
        align_l89_to_s2=args.align_l89_to_s2,
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )

    candidate_keys = collect_candidate_rows(base_ds, tasks, min_sensors=args.min_sensors)
    if args.max_rows is not None:
        candidate_keys = candidate_keys[: args.max_rows]
    allowed_keys = set(candidate_keys)

    per_task_preds: Dict[str, Dict[str, Dict[str, object]]] = {}
    for task in tasks:
        ds = SingleSensorSegmentationDataset(base_ds, task)
        model = load_model_for_task(args, task, ds, device)
        per_task_preds[task.name] = predict_task_probs(
            args=args,
            task=task,
            model=model,
            ds=ds,
            device=device,
            allowed_keys=allowed_keys,
        )
        print(f"[{task.name}] predictions={len(per_task_preds[task.name])}", flush=True)

    rows: List[Dict[str, object]] = []
    score_sum = 0.0
    count = 0
    mode_counts: Dict[str, int] = {}
    own_score_sums: Dict[str, float] = {name: 0.0 for name in task_by_name}
    own_score_counts: Dict[str, int] = {name: 0 for name in task_by_name}
    key_to_row_idx = {_sample_key(row, int(i)): int(i) for i, row in base_ds.df.iterrows()}

    for key in candidate_keys:
        available = [name for name in task_by_name if key in per_task_preds[name]]
        if len(available) < args.min_sensors:
            continue
        for name in available:
            own_score_sums[name] += float(per_task_preds[name][key]["own_iou_plus"])
            own_score_counts[name] += 1
        row_idx = key_to_row_idx[key]
        row = base_ds.df.iloc[row_idx]
        ref_name = choose_reference_task(row, available, args.reference)
        ref_task = task_by_name[ref_name]
        ref_info = per_task_preds[ref_name][key]
        ref_mask_path = str(ref_info["mask_path"])
        target_mask = load_binary_mask(ref_mask_path)
        dst_shape = tuple(int(x) for x in target_mask.shape)

        aligned_probs: List[np.ndarray] = []
        align_modes: List[str] = []
        for name in available:
            info = per_task_preds[name][key]
            src = np.asarray(info["prob"], dtype=np.float32)
            aligned, align_mode = geospatial_reproject_or_resize(
                src,
                src_ref_path=str(info["mask_path"]),
                dst_ref_path=ref_mask_path,
                dst_shape=dst_shape,
                resampling=args.reproject_resampling,
            )
            aligned_probs.append(aligned)
            align_modes.append(align_mode)
            mode_counts[align_mode] = mode_counts.get(align_mode, 0) + 1

        fused_prob = fuse_probs(aligned_probs, args.fusion, args.threshold)
        fused_mask = fused_prob >= args.threshold
        score = iou_plus(fused_mask, target_mask)
        score_sum += score
        count += 1
        rows.append(
            {
                "id": key,
                "row_idx": row_idx,
                "plume_id": str(row.get("plume_id", "")),
                "label": str(row.get("label", "")),
                "reference": ref_name,
                "sensors": ",".join(available),
                "num_sensors": len(available),
                "target_mask": ref_task.mask_column,
                "target_shape": f"{dst_shape[0]}x{dst_shape[1]}",
                "fusion_iou_plus": score,
                "align_modes": ",".join(sorted(set(align_modes))),
            }
        )

    summary = {
        "model_type": args.model_type,
        "fusion": args.fusion,
        "threshold": args.threshold,
        "reference": args.reference,
        "tasks": [t.name for t in tasks],
        "min_sensors": args.min_sensors,
        "count": count,
        "fusion_iou_plus": score_sum / max(1, count),
        "own_iou_plus": {
            name: own_score_sums[name] / max(1, own_score_counts[name])
            for name in task_by_name
        },
        "own_iou_plus_counts": own_score_counts,
        "align_mode_counts": mode_counts,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "id",
                "row_idx",
                "plume_id",
                "label",
                "reference",
                "sensors",
                "num_sensors",
                "target_mask",
                "target_shape",
                "fusion_iou_plus",
                "align_modes",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
