"""Multi-task (sensor-split) segmentation finetuning with IoU+ selection.

This script reuses the preprocessing and backbone-loading utilities from
`multi_sensor_panopticon_4.py`, but trains segmentation models for three
independent tasks:
  - s2
  - l89
  - emit (mapped to wv3 preprocessing/channel IDs)

Each task:
  1) loads the same classification-pretrained backbone weights
  2) creates a fresh segmentation head (not shared)
  3) trains with BCEWithLogits + Dice
  4) selects checkpoints by validation IoU+ only
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("XFORMERS_DISABLED", "1")

from universal_models_fusion.multi_sensor_panopticon_4 import (  # noqa: E402
    DEFAULT_WV3_BANDS,
    StaticAnchoredCache,
    TriSensorTemporalCsvDataset,
    _load_backbone,
    collect_cache_paths_from_df,
    load_wv3_channel_ids_from_srf,
    recursive_to_device,
    set_trainable,
)


@dataclass(frozen=True)
class TaskConfig:
    name: str
    sensor_key: str
    image_columns: Tuple[str, str, str]
    mask_column: str


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "s2": TaskConfig(
        name="s2",
        sensor_key="s2",
        image_columns=("s2_0_path", "s2_90_path", "s2_360_path"),
        mask_column="s2_plume_path",
    ),
    "l89": TaskConfig(
        name="l89",
        sensor_key="l89",
        image_columns=("l89_0_path", "l89_90_path", "l89_360_path"),
        mask_column="l89_plume_path",
    ),
    "emit": TaskConfig(
        name="emit",
        sensor_key="wv3",
        image_columns=("emit_0_path", "emit_90_path", "emit_360_path"),
        mask_column="emit_plume_path",
    ),
}


def _is_missing(value: object) -> bool:
    text = str(value).strip().lower()
    return text in ("", "nan", "none", "null")


class SingleSensorSegmentationDataset(Dataset):
    """Wrap a wide-table multi-sensor dataset into one sensor-specific seg task."""

    def __init__(self, base_ds: TriSensorTemporalCsvDataset, task: TaskConfig):
        self.base_ds = base_ds
        self.task = task
        self.max_retries = max(1, int(getattr(base_ds, "max_retries", 3)))
        self.indices = self._collect_valid_indices()
        if len(self.indices) == 0:
            raise ValueError(f"No valid rows found for task={task.name}")

    def _collect_valid_indices(self) -> List[int]:
        idxs: List[int] = []
        cols = list(self.task.image_columns) + [self.task.mask_column]
        for i, row in self.base_ds.df.iterrows():
            ok = True
            for col in cols:
                if col not in self.base_ds.df.columns or _is_missing(row.get(col, "")):
                    ok = False
                    break
            if ok:
                idxs.append(int(i))
        return idxs

    def __len__(self) -> int:
        return len(self.indices)

    def _load_mask(self, row, imgs_hw: Tuple[int, int]) -> torch.Tensor:
        mask_path = str(row[self.task.mask_column]).strip()
        mask_path = self.base_ds._maybe_cache_path(mask_path)
        arr = tiff.imread(mask_path)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Mask must be 2D or single-band 3D, got shape={arr.shape} at {mask_path}")
        mask = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        mask = (mask > 0).float()
        if self.base_ds.pad_to_multiple is not None:
            mask = self.base_ds._pad_to_multiple(mask.unsqueeze(0), self.base_ds.pad_to_multiple).squeeze(0)
        if tuple(mask.shape) != tuple(imgs_hw):
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=imgs_hw,
                mode="nearest",
            ).squeeze(0).squeeze(0)
        return mask

    def __getitem__(self, idx: int):
        attempts = 0
        current_idx = idx
        last_exc: Optional[Exception] = None
        while attempts < self.max_retries:
            row_idx = self.indices[current_idx]
            row = self.base_ds.df.iloc[row_idx]
            try:
                x_list = [
                    self.base_ds._load_temporal_frame(
                        row=row,
                        column_name=col,
                        sensor=self.task.sensor_key,
                        sample_id=row_idx,
                    )
                    for col in self.task.image_columns
                ]
                imgs = torch.cat([x["imgs"] for x in x_list], dim=0)
                chn_ids = torch.cat([x["chn_ids"] for x in x_list], dim=0)
                mask = self._load_mask(row=row, imgs_hw=(imgs.shape[1], imgs.shape[2]))
                sample_id = row.get("id", row_idx)
                return {"imgs": imgs, "chn_ids": chn_ids}, mask.unsqueeze(0), sample_id
            except Exception as exc:
                last_exc = exc
                attempts += 1
                if attempts >= self.max_retries:
                    raise RuntimeError(
                        f"Failed to load sample after {self.max_retries} retries for task={self.task.name}, "
                        f"row_idx={row_idx}"
                    ) from exc
                current_idx = random.randint(0, len(self.indices) - 1)
        raise RuntimeError("Unreachable") from last_exc


def segmentation_collate_fn(batch):
    x_dicts: List[Dict[str, torch.Tensor]] = []
    masks: List[torch.Tensor] = []
    sample_ids: List[object] = []
    for x_dict, mask, sample_id in batch:
        x_dicts.append(x_dict)
        masks.append(mask)
        sample_ids.append(sample_id)

    max_channels = max(x["imgs"].shape[0] for x in x_dicts)
    max_h = max(x["imgs"].shape[1] for x in x_dicts)
    max_w = max(x["imgs"].shape[2] for x in x_dicts)

    padded_imgs = []
    padded_chn_ids = []
    padded_masks = []
    for x_dict, mask in zip(x_dicts, masks):
        img = x_dict["imgs"]
        chn_ids = x_dict["chn_ids"]
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        img = F.pad(img, (0, pad_w, 0, pad_h))
        mask = F.pad(mask, (0, pad_w, 0, pad_h))
        pad_c = max_channels - img.shape[0]
        if pad_c > 0:
            img = torch.cat([img, torch.zeros((pad_c, max_h, max_w), dtype=img.dtype)], dim=0)
            chn_ids = torch.cat(
                [chn_ids, torch.zeros((pad_c, *chn_ids.shape[1:]), dtype=chn_ids.dtype)],
                dim=0,
            )
        padded_imgs.append(img)
        padded_chn_ids.append(chn_ids)
        padded_masks.append(mask)

    batched_x = {"imgs": torch.stack(padded_imgs), "chn_ids": torch.stack(padded_chn_ids)}
    batched_mask = torch.stack(padded_masks)
    return batched_x, batched_mask, sample_ids


class PatchTokenSegmentationHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        hidden = max(embed_dim // 2, 128)
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [B, N, D]
        return self.net(patch_tokens)


class PanopticonSegmentationModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        embed_dim = int(getattr(backbone, "embed_dim", 768))
        self.seg_head = PatchTokenSegmentationHead(embed_dim=embed_dim)

    def _infer_patch_hw(self, n_tokens: int, h: int, w: int) -> Tuple[int, int]:
        patch_size = int(getattr(self.backbone, "patch_size", 14))
        ph = max(1, h // patch_size)
        pw = max(1, w // patch_size)
        if ph * pw == n_tokens:
            return ph, pw
        # Fallback for any mismatch caused by dynamic padding/interpolation internals.
        approx_h = max(1, int(round(np.sqrt((n_tokens * h) / max(1, w)))))
        approx_w = max(1, n_tokens // approx_h)
        if approx_h * approx_w != n_tokens:
            approx_h = int(round(np.sqrt(n_tokens)))
            approx_h = max(1, approx_h)
            approx_w = max(1, n_tokens // approx_h)
        if approx_h * approx_w != n_tokens:
            raise ValueError(f"Cannot infer patch grid for n_tokens={n_tokens}, input_hw=({h},{w})")
        return approx_h, approx_w

    def forward(self, x_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        feats = self.backbone(x_dict, is_training=True)
        patch_tokens = torch.nan_to_num(feats["x_norm_patchtokens"], nan=0.0, posinf=1e4, neginf=-1e4)
        b, n, _ = patch_tokens.shape
        h, w = x_dict["imgs"].shape[-2:]
        ph, pw = self._infer_patch_hw(n_tokens=n, h=h, w=w)
        patch_logits = self.seg_head(patch_tokens).transpose(1, 2).reshape(b, 1, ph, pw)
        logits = F.interpolate(patch_logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def iou_plus_scores_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> torch.Tensor:
    pred = torch.sigmoid(logits) >= float(threshold)
    gt = targets >= 0.5
    pred = pred.flatten(start_dim=1)
    gt = gt.flatten(start_dim=1)

    inter = torch.logical_and(pred, gt).sum(dim=1).float()
    union = torch.logical_or(pred, gt).sum(dim=1).float().clamp_min(1.0)
    iou = inter / union

    gt_empty = ~gt.any(dim=1)
    pred_empty = ~pred.any(dim=1)
    ones = torch.ones_like(iou)
    zeros = torch.zeros_like(iou)
    return torch.where(gt_empty, torch.where(pred_empty, ones, zeros), iou)


def parse_tasks(tasks_text: str) -> List[TaskConfig]:
    names = [x.strip().lower() for x in tasks_text.split(",") if x.strip()]
    if not names:
        raise ValueError("--tasks cannot be empty")
    out: List[TaskConfig] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        if name not in TASK_CONFIGS:
            raise ValueError(f"Unsupported task '{name}'. Available: {sorted(TASK_CONFIGS)}")
        out.append(TASK_CONFIGS[name])
        seen.add(name)
    return out


def default_run_name(args) -> str:
    train_stem = Path(args.train_csv).stem or "train"
    test_stem = Path(args.test_csv).stem or "test"
    return f"{train_stem}__{test_stem}__seg"


def build_optimizer(args, model: PanopticonSegmentationModel) -> torch.optim.Optimizer:
    param_groups = [
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": model.seg_head.parameters(), "lr": args.head_lr},
    ]
    return torch.optim.Adam(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    model: PanopticonSegmentationModel,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    best_val_iou_plus: float,
    args,
    task_name: str,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "best_val_iou_plus": best_val_iou_plus,
            "task": task_name,
            "args": vars(args),
        },
        path,
    )


def try_resume(
    path: Path,
    model: PanopticonSegmentationModel,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
) -> Tuple[int, int, float]:
    if not path.is_file():
        return 1, 0, float("-inf")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    global_step = int(ckpt.get("global_step", 0))
    best_val = float(ckpt.get("best_val_iou_plus", float("-inf")))
    print(f"Resumed from {path} at epoch {start_epoch - 1}", flush=True)
    return start_epoch, global_step, best_val


def run_train_epoch(
    *,
    model: PanopticonSegmentationModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    max_grad_norm: float,
    bce_weight: float,
    dice_weight: float,
    eval_threshold: float,
    log_interval: int,
    epoch: int,
    task_name: str,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou_plus = 0.0
    total_samples = 0

    for step_idx, (x_dict, masks, _) in enumerate(loader, 1):
        x_dict = recursive_to_device(x_dict, device)
        masks = masks.to(device=device, dtype=torch.float32)
        batch_size = masks.shape[0]

        with autocast(enabled=use_amp):
            logits = model(x_dict)
            bce = F.binary_cross_entropy_with_logits(logits, masks)
            dice = dice_loss_from_logits(logits, masks)
            loss = (bce_weight * bce) + (dice_weight * dice)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            iou_plus_batch = iou_plus_scores_from_logits(logits, masks, threshold=eval_threshold)

        total_loss += float(loss.item()) * batch_size
        total_bce += float(bce.item()) * batch_size
        total_dice += float(dice.item()) * batch_size
        total_iou_plus += float(iou_plus_batch.sum().item())
        total_samples += batch_size

        if log_interval > 0 and step_idx % log_interval == 0:
            print(
                f"[{task_name}] epoch={epoch} step={step_idx}/{max(1, len(loader))} "
                f"train_loss={total_loss/max(1,total_samples):.4f} "
                f"train_iou_plus={total_iou_plus/max(1,total_samples):.4f}",
                flush=True,
            )

        if max_steps is not None and step_idx >= max_steps:
            break

    return {
        "loss": total_loss / max(1, total_samples),
        "bce": total_bce / max(1, total_samples),
        "dice": total_dice / max(1, total_samples),
        "iou_plus": total_iou_plus / max(1, total_samples),
    }


@torch.no_grad()
def run_eval_epoch(
    *,
    model: PanopticonSegmentationModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    bce_weight: float,
    dice_weight: float,
    eval_threshold: float,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou_plus = 0.0
    total_samples = 0

    for step_idx, (x_dict, masks, _) in enumerate(loader, 1):
        x_dict = recursive_to_device(x_dict, device)
        masks = masks.to(device=device, dtype=torch.float32)
        batch_size = masks.shape[0]

        with autocast(enabled=use_amp):
            logits = model(x_dict)
            bce = F.binary_cross_entropy_with_logits(logits, masks)
            dice = dice_loss_from_logits(logits, masks)
            loss = (bce_weight * bce) + (dice_weight * dice)
        iou_plus_batch = iou_plus_scores_from_logits(logits, masks, threshold=eval_threshold)

        total_loss += float(loss.item()) * batch_size
        total_bce += float(bce.item()) * batch_size
        total_dice += float(dice.item()) * batch_size
        total_iou_plus += float(iou_plus_batch.sum().item())
        total_samples += batch_size

        if max_steps is not None and step_idx >= max_steps:
            break

    return {
        "loss": total_loss / max(1, total_samples),
        "bce": total_bce / max(1, total_samples),
        "dice": total_dice / max(1, total_samples),
        "iou_plus": total_iou_plus / max(1, total_samples),
        "count": float(total_samples),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_wandb(args, *, run_name: str, task_name: str):
    if not args.use_wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.wandb_run_name or default_run_name(args),
        config={**vars(args), "task": task_name},
    )


def train_single_task(
    *,
    args,
    task: TaskConfig,
    device: torch.device,
    train_base_ds: TriSensorTemporalCsvDataset,
    test_base_ds: TriSensorTemporalCsvDataset,
) -> Dict[str, float]:
    train_ds = SingleSensorSegmentationDataset(train_base_ds, task)
    test_ds = SingleSensorSegmentationDataset(test_base_ds, task)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=segmentation_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=segmentation_collate_fn,
    )

    model = PanopticonSegmentationModel(backbone=_load_backbone(args.weights)).to(device)
    optimizer = build_optimizer(args, model)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    base_run_name = args.wandb_run_name or default_run_name(args)
    task_run_name = f"{base_run_name}__{task.name}"
    ckpt_dir = Path(args.checkpoint_dir) / task_run_name
    latest_path = ckpt_dir / "ckpt_latest.pth"
    best_path = ckpt_dir / "ckpt_best_val_iou_plus.pth"

    start_epoch = 1
    global_step = 0
    best_val_iou_plus = float("-inf")
    if args.resume:
        start_epoch, global_step, best_val_iou_plus = try_resume(
            latest_path, model, optimizer, scaler if use_amp else None, device
        )

    wandb_run = init_wandb(args, run_name=task_run_name, task_name=task.name)
    print(
        f"[{task.name}] train_samples={len(train_ds)} test_samples={len(test_ds)} "
        f"checkpoints={ckpt_dir}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        freeze_backbone = epoch <= args.freeze_backbone_epochs
        if freeze_backbone:
            set_trainable(model.backbone, False)
            model.backbone.eval()
        else:
            set_trainable(model.backbone, True)
            model.backbone.train()
        model.seg_head.train()

        train_metrics = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            max_grad_norm=args.max_grad_norm,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            eval_threshold=args.eval_threshold,
            log_interval=args.log_interval,
            epoch=epoch,
            task_name=task.name,
            max_steps=args.max_train_steps,
        )
        val_metrics = run_eval_epoch(
            model=model,
            loader=test_loader,
            device=device,
            use_amp=use_amp,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            eval_threshold=args.eval_threshold,
            max_steps=args.max_eval_steps,
        )
        global_step += len(train_loader)

        val_iou_plus = val_metrics["iou_plus"]
        print(
            f"[{task.name}] epoch={epoch} freeze_backbone={freeze_backbone} "
            f"train_loss={train_metrics['loss']:.4f} train_iou_plus={train_metrics['iou_plus']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_iou_plus={val_iou_plus:.4f}",
            flush=True,
        )

        save_checkpoint(
            latest_path,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            best_val_iou_plus=max(best_val_iou_plus, val_iou_plus),
            args=args,
            task_name=task.name,
        )
        if val_iou_plus > best_val_iou_plus:
            best_val_iou_plus = val_iou_plus
            save_checkpoint(
                best_path,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scaler=scaler if use_amp else None,
                best_val_iou_plus=best_val_iou_plus,
                args=args,
                task_name=task.name,
            )
            print(
                f"[{task.name}] Saved new best checkpoint: {best_path} "
                f"(val_iou_plus={best_val_iou_plus:.4f})",
                flush=True,
            )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "task": task.name,
                    "freeze_backbone": float(freeze_backbone),
                    "train_loss": train_metrics["loss"],
                    "train_bce": train_metrics["bce"],
                    "train_dice": train_metrics["dice"],
                    "train_iou_plus": train_metrics["iou_plus"],
                    "val_loss": val_metrics["loss"],
                    "val_bce": val_metrics["bce"],
                    "val_dice": val_metrics["dice"],
                    "val_iou_plus": val_metrics["iou_plus"],
                    "val_count": val_metrics["count"],
                    "best_val_iou_plus": best_val_iou_plus,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()
    return {"best_val_iou_plus": best_val_iou_plus, "train_count": float(len(train_ds)), "test_count": float(len(test_ds))}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Panopticon segmentation finetuning on sensor-split tasks (s2/l89/emit) with IoU+."
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--tasks", default="s2,l89,emit", help="Comma-separated subset of: s2,l89,emit")
    parser.add_argument("--weights", default="weights/panopticon_vitb14_teacher.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2)
    parser.add_argument("--backbone_lr", type=float, default=5e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--eval_threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_dir", default="checkpoints/multi_sensor_seg")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="baselines")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--local_cache_dir", default=None)
    parser.add_argument("--local_cache_min_free_gb", type=float, default=5.0)
    parser.add_argument("--local_cache_warmup", action="store_true")
    parser.add_argument("--local_cache_workers", type=int, default=8)
    parser.add_argument("--align_l89_to_s2", action="store_true")
    parser.add_argument(
        "--wv3_srf_csv",
        default=str(REPO_ROOT / "WV3_VNIR_SWIR_response.csv"),
        help="Path to WV3 SRF CSV for emit(wv3) channel IDs.",
    )
    parser.add_argument(
        "--wv3_bands",
        default=",".join(DEFAULT_WV3_BANDS),
        help="Comma-separated WV3 SRF column names for emit(wv3).",
    )
    return parser.parse_args()


def main(args):
    set_seed(args.seed)
    tasks = parse_tasks(args.tasks)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    wv3_band_names = [x.strip() for x in args.wv3_bands.split(",") if x.strip()]
    if len(wv3_band_names) == 0:
        raise ValueError("--wv3_bands must provide at least one name")
    wv3_chn_ids = load_wv3_channel_ids_from_srf(args.wv3_srf_csv, wv3_band_names).unsqueeze(-1)

    cache_obj = None
    if args.local_cache_dir:
        cache_obj = StaticAnchoredCache(args.local_cache_dir, min_free_gb=args.local_cache_min_free_gb)
        print(
            f"[Cache] Enabled local cache dir={cache_obj.cache_dir} "
            f"(min_free_gb={args.local_cache_min_free_gb:.2f})",
            flush=True,
        )

    train_base_ds = TriSensorTemporalCsvDataset(
        csv_path=args.train_csv,
        local_file_cache=cache_obj,
        align_l89_to_s2=args.align_l89_to_s2,
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )
    test_base_ds = TriSensorTemporalCsvDataset(
        csv_path=args.test_csv,
        local_file_cache=cache_obj,
        align_l89_to_s2=args.align_l89_to_s2,
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )

    if args.local_cache_warmup and cache_obj is not None:
        task_cols = set()
        for task in tasks:
            task_cols.update(task.image_columns)
            task_cols.add(task.mask_column)
        warmup_cols = [c for c in sorted(task_cols) if c in train_base_ds.df.columns]
        all_paths = collect_cache_paths_from_df(train_base_ds.df, warmup_cols)
        print(
            f"[Cache] Warmup scanning {len(warmup_cols)} columns for selected tasks, "
            f"found {len(all_paths)} path entries.",
            flush=True,
        )
        cache_obj.warm_up(all_paths, max_workers=args.local_cache_workers)

    print(
        f"Using device={device}, tasks={[t.name for t in tasks]}, "
        f"freeze_backbone_epochs={args.freeze_backbone_epochs}, "
        f"backbone_lr={args.backbone_lr}, head_lr={args.head_lr}, "
        f"loss=BCEWithLogits+Dice, eval_metric=IoU+",
        flush=True,
    )

    summary: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        summary[task.name] = train_single_task(
            args=args,
            task=task,
            device=device,
            train_base_ds=train_base_ds,
            test_base_ds=test_base_ds,
        )

    print("=== Segmentation summary (best by val_iou_plus) ===", flush=True)
    for task_name, stats in summary.items():
        print(
            f"[{task_name}] best_val_iou_plus={stats['best_val_iou_plus']:.4f} "
            f"train_count={int(stats['train_count'])} test_count={int(stats['test_count'])}",
            flush=True,
        )


if __name__ == "__main__":
    main(parse_args())
