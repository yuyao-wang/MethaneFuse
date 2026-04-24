"""UNet segmentation baseline on sensor-split tasks with IoU+ selection.

Tasks are trained independently on wide-table CSVs:
  - s2
  - l89
  - emit

Evaluation metric is IoU+ only. Best checkpoint is selected by val_iou_plus.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from universal_models_fusion.multi_sensor_panopticon_4 import (  # noqa: E402
    DEFAULT_WV3_BANDS,
    StaticAnchoredCache,
    TriSensorTemporalCsvDataset,
    collect_cache_paths_from_df,
    load_wv3_channel_ids_from_srf,
    recursive_to_device,
)
from universal_models_fusion.multi_sensor_panopticon_4_segmentation import (  # noqa: E402
    TASK_CONFIGS,
    SingleSensorSegmentationDataset,
    TaskConfig,
    dice_loss_from_logits,
    iou_plus_scores_from_logits,
    parse_tasks,
    segmentation_collate_fn,
)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.shape[2] - x1.shape[2]
        diff_x = x2.shape[3] - x1.shape[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetBaseline(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.up1 = Up(c * 16 + c * 8, c * 8)
        self.up2 = Up(c * 8 + c * 4, c * 4)
        self.up3 = Up(c * 4 + c * 2, c * 2)
        self.up4 = Up(c * 2 + c, c)
        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x_dict) -> torch.Tensor:
        x = x_dict["imgs"]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(args, model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    model: nn.Module,
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


def try_resume(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: Optional[GradScaler], device: torch.device):
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
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    bce_weight: float,
    dice_weight: float,
    threshold: float,
    max_grad_norm: float,
    max_steps: Optional[int],
    log_interval: int,
    epoch: int,
    task_name: str,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou_plus = 0.0
    total = 0

    for step_idx, (x_dict, masks, _) in enumerate(loader, 1):
        x_dict = recursive_to_device(x_dict, device)
        masks = masks.to(device=device, dtype=torch.float32)
        n = int(masks.shape[0])

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
            iou_plus = iou_plus_scores_from_logits(logits, masks, threshold=threshold)

        total_loss += float(loss.item()) * n
        total_bce += float(bce.item()) * n
        total_dice += float(dice.item()) * n
        total_iou_plus += float(iou_plus.sum().item())
        total += n

        if log_interval > 0 and step_idx % log_interval == 0:
            print(
                f"[{task_name}] epoch={epoch} step={step_idx}/{max(1, len(loader))} "
                f"train_loss={total_loss/max(1,total):.4f} "
                f"train_bce={total_bce/max(1,total):.4f} "
                f"train_dice={total_dice/max(1,total):.4f} "
                f"train_iou_plus={total_iou_plus/max(1,total):.4f}",
                flush=True,
            )

        if max_steps is not None and step_idx >= max_steps:
            break

    return {
        "loss": total_loss / max(1, total),
        "bce": total_bce / max(1, total),
        "dice": total_dice / max(1, total),
        "iou_plus": total_iou_plus / max(1, total),
        "count": float(total),
    }


@torch.no_grad()
def run_eval_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    bce_weight: float,
    dice_weight: float,
    threshold: float,
    max_steps: Optional[int],
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou_plus = 0.0
    total = 0

    for step_idx, (x_dict, masks, _) in enumerate(loader, 1):
        x_dict = recursive_to_device(x_dict, device)
        masks = masks.to(device=device, dtype=torch.float32)
        n = int(masks.shape[0])

        with autocast(enabled=use_amp):
            logits = model(x_dict)
            bce = F.binary_cross_entropy_with_logits(logits, masks)
            dice = dice_loss_from_logits(logits, masks)
            loss = (bce_weight * bce) + (dice_weight * dice)
        iou_plus = iou_plus_scores_from_logits(logits, masks, threshold=threshold)

        total_loss += float(loss.item()) * n
        total_bce += float(bce.item()) * n
        total_dice += float(dice.item()) * n
        total_iou_plus += float(iou_plus.sum().item())
        total += n

        if max_steps is not None and step_idx >= max_steps:
            break

    return {
        "loss": total_loss / max(1, total),
        "bce": total_bce / max(1, total),
        "dice": total_dice / max(1, total),
        "iou_plus": total_iou_plus / max(1, total),
        "count": float(total),
    }


def init_wandb(args, run_name: str, task_name: str):
    if not args.use_wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.wandb_run_name or "unet_baseline",
        config={**vars(args), "task": task_name},
    )


def build_loaders(args, train_base_ds, test_base_ds, task: TaskConfig, device: torch.device):
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
    return train_ds, test_ds, train_loader, test_loader


def train_single_task(args, task: TaskConfig, train_base_ds, test_base_ds, device: torch.device) -> Dict[str, float]:
    train_ds, test_ds, train_loader, test_loader = build_loaders(args, train_base_ds, test_base_ds, task, device)
    sample_x, _, _ = train_ds[0]
    in_channels = int(sample_x["imgs"].shape[0])

    model = UNetBaseline(in_channels=in_channels, base_channels=args.base_channels).to(device)
    optimizer = build_optimizer(args, model)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    run_base = args.wandb_run_name or f"unet_{Path(args.train_csv).stem}__{Path(args.test_csv).stem}"
    run_name = f"{run_base}__{task.name}"
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    latest_path = ckpt_dir / "ckpt_latest.pth"
    best_path = ckpt_dir / "ckpt_best_val_iou_plus.pth"

    start_epoch = 1
    global_step = 0
    best_val_iou_plus = float("-inf")
    if args.resume:
        start_epoch, global_step, best_val_iou_plus = try_resume(
            latest_path,
            model,
            optimizer,
            scaler if use_amp else None,
            device,
        )

    wandb_run = init_wandb(args, run_name=run_name, task_name=task.name)

    print(
        f"[{task.name}] in_channels={in_channels} train_samples={len(train_ds)} test_samples={len(test_ds)} "
        f"ckpt_dir={ckpt_dir}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_m = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            threshold=args.eval_threshold,
            max_grad_norm=args.max_grad_norm,
            max_steps=args.max_train_steps,
            log_interval=args.log_interval,
            epoch=epoch,
            task_name=task.name,
        )
        val_m = run_eval_epoch(
            model=model,
            loader=test_loader,
            device=device,
            use_amp=use_amp,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            threshold=args.eval_threshold,
            max_steps=args.max_eval_steps,
        )
        global_step += int(args.max_train_steps or len(train_loader))

        print(
            f"[{task.name}] epoch={epoch} "
            f"train_loss={train_m['loss']:.4f} train_iou_plus={train_m['iou_plus']:.4f} "
            f"val_loss={val_m['loss']:.4f} val_iou_plus={val_m['iou_plus']:.4f}",
            flush=True,
        )

        save_checkpoint(
            latest_path,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            best_val_iou_plus=max(best_val_iou_plus, val_m["iou_plus"]),
            args=args,
            task_name=task.name,
        )
        if val_m["iou_plus"] > best_val_iou_plus:
            best_val_iou_plus = val_m["iou_plus"]
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
            print(f"[{task.name}] saved best checkpoint: {best_path}", flush=True)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "task": task.name,
                    "train_loss": train_m["loss"],
                    "train_bce": train_m["bce"],
                    "train_dice": train_m["dice"],
                    "train_iou_plus": train_m["iou_plus"],
                    "train_count": train_m["count"],
                    "val_loss": val_m["loss"],
                    "val_bce": val_m["bce"],
                    "val_dice": val_m["dice"],
                    "val_iou_plus": val_m["iou_plus"],
                    "val_count": val_m["count"],
                    "best_val_iou_plus": best_val_iou_plus,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()
    return {
        "best_val_iou_plus": best_val_iou_plus,
        "train_count": float(len(train_ds)),
        "test_count": float(len(test_ds)),
        "in_channels": float(in_channels),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="UNet segmentation baseline on sensor-split tasks with IoU+.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--tasks", default="s2,l89,emit", help="Comma-separated subset of: s2,l89,emit")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--eval_threshold", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50, help="Print train metrics every N steps (<=0 disables).")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_dir", default="checkpoints/unet_multisensor_baseline")
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
        help="Path to WV3 SRF CSV for emit channel IDs.",
    )
    parser.add_argument(
        "--wv3_bands",
        default=",".join(DEFAULT_WV3_BANDS),
        help="Comma-separated WV3 SRF column names.",
    )
    return parser.parse_args()


def main(args):
    set_seed(args.seed)
    tasks: List[TaskConfig] = parse_tasks(args.tasks)

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
            f"[Cache] Enabled local cache dir={cache_obj.cache_dir} (min_free_gb={args.local_cache_min_free_gb:.2f})",
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

    summary: Dict[str, Dict[str, float]] = {}
    print(
        f"UNet baseline start: device={device}, tasks={[t.name for t in tasks]}, "
        f"loss=BCEWithLogits+Dice, metric=IoU+, eval_threshold={args.eval_threshold}",
        flush=True,
    )
    for task in tasks:
        summary[task.name] = train_single_task(args, task, train_base_ds, test_base_ds, device)

    print("=== UNet baseline summary (best by val_iou_plus) ===", flush=True)
    for task_name in [t.name for t in tasks]:
        m = summary[task_name]
        print(
            f"[{task_name}] best_val_iou_plus={m['best_val_iou_plus']:.4f} "
            f"train_count={int(m['train_count'])} test_count={int(m['test_count'])} "
            f"in_channels={int(m['in_channels'])}",
            flush=True,
        )


if __name__ == "__main__":
    main(parse_args())
