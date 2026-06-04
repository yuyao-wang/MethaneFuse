#!/usr/bin/env python
"""SatMAE-FT + average-fusion baseline for the 480m multisensor query dataset.

The SatMAE multispectral encoder expects 10-channel 96x96 inputs. S2 and L89 are
mapped directly into that shape; EMIT and S5P are deliberately forced through
small trainable projection adapters. Each available sensor contributes one
feature per row, temporal frames are averaged per sensor, and row fusion is a
plain arithmetic mean over present sensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile as tiff
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

S2_STATS = (
    [786.128173828125, 1025.8876953125, 1593.730712890625, 2315.26123046875, 2710.462890625,
     3115.90087890625, 3289.0830078125, 3465.536376953125, 3495.579833984375, 3517.7958984375,
     4180.28564453125, 3567.866943359375],
    [435.72607421875, 597.6113891601562, 688.5059814453125, 840.1614990234375, 801.7208251953125,
     706.9466552734375, 689.823974609375, 727.5567626953125, 668.30224609375, 551.3565063476562,
     629.679931640625, 641.590087890625],
)

L89_STATS_7 = (
    [10729.92784546, 11384.64407242, 13172.77519667, 14892.25620267, 18149.92169893,
     20249.17615773, 18375.0669698],
    [1029.18232283, 1188.52313418, 1552.27685613, 1959.74400972, 1954.80410093,
     2098.98682671, 1895.56781996],
)

S5P_STATS = (
    [1908.944798144908, 362.20128748570943, 666.5645739544017],
    [52.2615669211965, 743.7553740768346, 901.8399543163595],
)

SATMAE_GROUPS = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))


def is_present(value: Any) -> bool:
    if not isinstance(value, str):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return False
        value = str(value)
    value = value.strip()
    return bool(value) and value.lower() != "nan"


def read_tiff_chw(path: str) -> torch.Tensor:
    arr = tiff.imread(path)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[0] > 64 and arr.shape[-1] <= 64:
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape {arr.shape} at {path}")
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def resize_chw(img: torch.Tensor, size: int) -> torch.Tensor:
    if img.shape[-2:] == (size, size):
        return img
    return F.interpolate(img.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)


def normalize(img: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).clamp(min=1e-6).view(-1, 1, 1)
    c = min(img.shape[0], mean_t.shape[0])
    out = img.clone()
    out[:c] = (out[:c] - mean_t[:c]) / std_t[:c]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def standardize_per_band(img: torch.Tensor) -> torch.Tensor:
    flat = img.flatten(1)
    mean = flat.mean(dim=1).view(-1, 1, 1)
    std = flat.std(dim=1).clamp(min=1e-6).view(-1, 1, 1)
    return torch.nan_to_num((img - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


class LocalFileCache:
    def __init__(self, root: Optional[str], *, max_gb: float, min_free_gb: float):
        self.root = Path(root) if root else None
        self.max_bytes = int(max_gb * (1024 ** 3))
        self.min_free_bytes = int(min_free_gb * (1024 ** 3))
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def get(self, path: str) -> str:
        if self.root is None:
            return path
        src = Path(path)
        if not src.is_file():
            return path
        try:
            stat = src.stat()
            key = f"{abs(hash(str(src))):016x}_{stat.st_size}{src.suffix}"
            dst = self.root / key
            if dst.is_file() and dst.stat().st_size == stat.st_size:
                return str(dst)
            usage = shutil.disk_usage(self.root)
            if usage.free - stat.st_size < self.min_free_bytes:
                return path
            marker = self.root / ".cache_bytes"
            used = int(marker.read_text().strip()) if marker.is_file() else 0
            if used + stat.st_size > self.max_bytes:
                return path
            tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            marker.write_text(str(used + stat.st_size))
            return str(dst)
        except Exception:
            return path


@dataclass
class RowSample:
    label: int
    row_id: Any
    plume_id: str
    anchor_sensor: str
    overlap_mode: bool
    sensors: Dict[str, torch.Tensor]


class SatMAE480mDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        *,
        image_size: int = 96,
        max_retries: int = 5,
        local_cache_dir: Optional[str] = None,
        local_cache_max_gb: float = 350.0,
        local_cache_min_free_gb: float = 100.0,
    ):
        self.csv_path = str(csv_path)
        self.df = pd.read_csv(csv_path, low_memory=False)
        self.image_size = int(image_size)
        self.max_retries = max(1, int(max_retries))
        self.cache = LocalFileCache(local_cache_dir, max_gb=local_cache_max_gb, min_free_gb=local_cache_min_free_gb)
        self.s2_select = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        self.s2_mean = [S2_STATS[0][i] for i in self.s2_select]
        self.s2_std = [S2_STATS[1][i] for i in self.s2_select]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> RowSample:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._get_one(idx)
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= self.max_retries:
                    raise
                idx = random.randrange(len(self.df))
        raise RuntimeError("unreachable") from last_exc

    def _read_tiff(self, path: str) -> torch.Tensor:
        return read_tiff_chw(self.cache.get(path))

    def _read_s5p(self, path: str) -> torch.Tensor:
        np_obj = np.load(self.cache.get(path), allow_pickle=False)
        try:
            arr = np.array(np_obj["ch4"] if isinstance(np_obj, np.lib.npyio.NpzFile) and "ch4" in np_obj else np_obj[np_obj.files[0]])
        finally:
            if isinstance(np_obj, np.lib.npyio.NpzFile):
                np_obj.close()
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[0] > 32 and arr.shape[-1] <= 16:
            arr = np.transpose(arr, (2, 0, 1))
        img = torch.from_numpy(arr.astype(np.float32, copy=False))
        if img.shape[0] != 3:
            repeat = math.ceil(3 / max(1, img.shape[0]))
            img = img.repeat(repeat, 1, 1)[:3]
        img = normalize(img, S5P_STATS[0], S5P_STATS[1])
        return resize_chw(img, self.image_size)

    def _get_one(self, idx: int) -> RowSample:
        row = self.df.iloc[idx]
        sensors: Dict[str, torch.Tensor] = {}

        if all(is_present(row.get(c)) for c in ("s2_0_path", "s2_90_path", "s2_360_path")):
            frames = []
            for col in ("s2_0_path", "s2_90_path", "s2_360_path"):
                img = self._read_tiff(str(row[col]))
                if img.shape[0] < 12:
                    raise ValueError(f"S2 image has {img.shape[0]} channels at {row[col]}")
                img = normalize(img[self.s2_select], self.s2_mean, self.s2_std)
                frames.append(resize_chw(img, self.image_size))
            sensors["s2"] = torch.stack(frames, dim=0)

        if all(is_present(row.get(c)) for c in ("l89_0_path", "l89_90_path", "l89_360_path")):
            frames = []
            for col in ("l89_0_path", "l89_90_path", "l89_360_path"):
                img = self._read_tiff(str(row[col]))[:10]
                first = normalize(img[:7], L89_STATS_7[0], L89_STATS_7[1])
                if img.shape[0] > 7:
                    rest = standardize_per_band(img[7:10])
                    img = torch.cat([first, rest], dim=0)
                else:
                    img = first
                if img.shape[0] < 10:
                    img = torch.cat([img, torch.zeros((10 - img.shape[0], *img.shape[1:]), dtype=img.dtype)], dim=0)
                frames.append(resize_chw(img[:10], self.image_size))
            sensors["l89"] = torch.stack(frames, dim=0)

        if all(is_present(row.get(c)) for c in ("emit_0_path", "emit_90_path", "emit_360_path")):
            frames = []
            for col in ("emit_0_path", "emit_90_path", "emit_360_path"):
                img = standardize_per_band(self._read_tiff(str(row[col]))[:16])
                frames.append(resize_chw(img, self.image_size))
            sensors["emit"] = torch.stack(frames, dim=0)

        s5p_frames = []
        for col in ("s5p_0_path", "s5p_90_path", "s5p_360_path"):
            if is_present(row.get(col)):
                s5p_frames.append(self._read_s5p(str(row[col])))
        if s5p_frames:
            while len(s5p_frames) < 3:
                s5p_frames.append(torch.zeros_like(s5p_frames[0]))
            sensors["s5p"] = torch.stack(s5p_frames[:3], dim=0)

        if not sensors:
            raise ValueError(f"No usable sensor paths in row index {idx}")

        return RowSample(
            label=int(row["label"]),
            row_id=row.get("id", idx),
            plume_id=str(row.get("plume_id", "")),
            anchor_sensor=str(row.get("anchor_sensor", "")),
            overlap_mode=str(row.get("overlap_mode", "")).strip().lower() == "true",
            sensors=sensors,
        )


def collate_rows(batch: Sequence[RowSample]) -> Dict[str, Any]:
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)
    tensors_by_sensor: Dict[str, List[torch.Tensor]] = {}
    sample_to_row: Dict[str, List[int]] = {}
    for row_idx, item in enumerate(batch):
        for sensor, tensor in item.sensors.items():
            tensors_by_sensor.setdefault(sensor, []).append(tensor)
            sample_to_row.setdefault(sensor, []).append(row_idx)
    return {
        "labels": labels,
        "sensors": {k: torch.stack(v, dim=0) for k, v in tensors_by_sensor.items()},
        "sample_to_row": {k: torch.tensor(v, dtype=torch.long) for k, v in sample_to_row.items()},
        "row_ids": [b.row_id for b in batch],
        "plume_ids": [b.plume_id for b in batch],
        "anchor_sensor": [b.anchor_sensor for b in batch],
        "overlap_mode": [b.overlap_mode for b in batch],
    }


class ConvBandAdapter(nn.Module):
    def __init__(self, in_chans: int, out_chans: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, out_chans, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SatMAEAvgFusionClassifier(nn.Module):
    def __init__(
        self,
        *,
        satmae_repo: str,
        pretrained: str,
        image_size: int = 96,
        patch_size: int = 8,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        freeze_satmae: bool = False,
        temporal_fusion: str = "input_mean",
    ):
        super().__init__()
        self.temporal_fusion = temporal_fusion
        repo = Path(satmae_repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        if not hasattr(np, "float"):
            np.float = float  # type: ignore[attr-defined]
        from models_vit_group_channels import GroupChannelsVisionTransformer
        from functools import partial

        self.satmae = GroupChannelsVisionTransformer(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=10,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            channel_groups=SATMAE_GROUPS,
            num_classes=2,
            global_pool=False,
        )
        self._load_pretrained(pretrained)
        self.emit_adapter = ConvBandAdapter(16, 10)
        self.s5p_adapter = ConvBandAdapter(3, 10)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))
        if freeze_satmae:
            self.set_satmae_trainable(False)

    def _load_pretrained(self, path: str) -> None:
        if not path:
            return
        path_obj = Path(path)
        if path_obj.suffix == ".safetensors":
            from safetensors.torch import load_file
            checkpoint_model = load_file(str(path_obj))
        else:
            ckpt = torch.load(path, map_location="cpu")
            checkpoint_model = ckpt.get("model", ckpt)
        checkpoint_model = {
            k: v for k, v in checkpoint_model.items()
            if not k.startswith("decoder_") and not k.startswith("mask_token") and not k.startswith("head.")
        }
        state = self.satmae.state_dict()
        filtered = {k: v for k, v in checkpoint_model.items() if k in state and state[k].shape == v.shape}
        msg = self.satmae.load_state_dict(filtered, strict=False)
        print(f"[SatMAE] loaded {len(filtered)} tensors from {path}; missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)

    def set_satmae_trainable(self, trainable: bool) -> None:
        for p in self.satmae.parameters():
            p.requires_grad_(trainable)

    def encode_frames(self, x: torch.Tensor, adapter: Optional[nn.Module] = None) -> torch.Tensor:
        if self.temporal_fusion == "input_mean":
            x = x.mean(dim=1, keepdim=True)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        if adapter is not None:
            x = adapter(x)
        feat = self.satmae.forward_features(x)
        return feat.reshape(b, t, -1).mean(dim=1)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        labels = batch["labels"]
        device = labels.device
        batch_size = int(labels.shape[0])
        embed_dim = self.head[0].normalized_shape[0]
        fused = torch.zeros((batch_size, embed_dim), device=device)
        counts = torch.zeros((batch_size, 1), device=device)
        sensors: Mapping[str, torch.Tensor] = batch["sensors"]
        sample_to_row: Mapping[str, torch.Tensor] = batch["sample_to_row"]

        adapters: Dict[str, Optional[nn.Module]] = {"s2": None, "l89": None, "emit": self.emit_adapter, "s5p": self.s5p_adapter}
        for sensor in ("s2", "l89", "emit", "s5p"):
            if sensor not in sensors:
                continue
            feat = self.encode_frames(sensors[sensor], adapters[sensor]).to(dtype=fused.dtype)
            rows = sample_to_row[sensor]
            fused.index_add_(0, rows, feat)
            counts.index_add_(0, rows, torch.ones((rows.numel(), 1), device=device, dtype=fused.dtype))
        return self.head(fused / counts.clamp(min=1.0))


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    out["labels"] = batch["labels"].to(device, non_blocking=True)
    out["sensors"] = {k: v.to(device, non_blocking=True) for k, v in batch["sensors"].items()}
    out["sample_to_row"] = {k: v.to(device, non_blocking=True) for k, v in batch["sample_to_row"].items()}
    return out


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    pred = (s >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    fpr = fp / max(1, fp + tn)
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
    except Exception:
        auroc = float("nan")
    return {
        "n": int(len(y)),
        "acc": float((tp + tn) / max(1, len(y))),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "auroc": auroc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, *, max_steps: Optional[int] = None) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    rows: List[Dict[str, Any]] = []
    for step, batch in enumerate(loader, 1):
        if max_steps is not None and step > max_steps:
            break
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        labels = batch["labels"]
        loss = criterion(logits, labels)
        probs = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy().tolist()
        preds = (np.asarray(probs) >= 0.5).astype(int).tolist()
        label_list = labels.detach().cpu().numpy().astype(int).tolist()
        total_loss += float(loss.item()) * len(label_list)
        total += len(label_list)
        for i, label in enumerate(label_list):
            rows.append({
                "id": batch["row_ids"][i],
                "plume_id": batch["plume_ids"][i],
                "label": int(label),
                "prob1": float(probs[i]),
                "pred": int(preds[i]),
                "anchor_sensor": batch["anchor_sensor"][i],
                "overlap_mode": bool(batch["overlap_mode"][i]),
            })
    y = [r["label"] for r in rows]
    s = [r["prob1"] for r in rows]
    metrics: Dict[str, Any] = {"loss": total_loss / max(1, total), "overall": binary_metrics(y, s), "by_overlap": {}, "by_anchor_sensor": {}}
    for name, keep in {"overlap": [r for r in rows if r["overlap_mode"]], "single": [r for r in rows if not r["overlap_mode"]]}.items():
        metrics["by_overlap"][name] = binary_metrics([r["label"] for r in keep], [r["prob1"] for r in keep]) if keep else None
    for anchor in sorted({str(r["anchor_sensor"]) for r in rows}):
        keep = [r for r in rows if str(r["anchor_sensor"]) == anchor]
        metrics["by_anchor_sensor"][anchor] = binary_metrics([r["label"] for r in keep], [r["prob1"] for r in keep])
    metrics["rows"] = rows
    return metrics


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def write_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "plume_id", "label", "prob1", "pred", "anchor_sensor", "overlap_mode"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    os.replace(tmp, path)


def init_wandb(args: argparse.Namespace):
    if not args.use_wandb:
        return None
    import wandb
    return wandb.init(project=args.wandb_project, name=args.wandb_run_name or args.run_name, config=vars(args))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    *,
    epoch: int,
    log_interval: int,
    max_steps: Optional[int],
    use_amp: bool,
    accum_steps: int,
    wandb_run=None,
) -> Tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        if max_steps is not None and step > max_steps:
            break
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]
        with autocast(enabled=use_amp):
            logits = model(batch)
            loss = criterion(logits, labels) / max(1, accum_steps)
        scaler.scale(loss).backward()
        if step % max(1, accum_steps) == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        bs = int(labels.shape[0])
        total += bs
        total_loss += float(loss.item()) * bs * max(1, accum_steps)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        if log_interval and step % log_interval == 0:
            msg = f"[train] epoch={epoch} step={step}/{len(loader)} loss={total_loss/max(1,total):.4f} acc={correct/max(1,total):.4f} elapsed_min={(time.time()-start)/60:.1f}"
            print(msg, flush=True)
            if wandb_run is not None:
                wandb_run.log({"train/loss_running": total_loss / max(1, total), "train/acc_running": correct / max(1, total), "epoch": epoch})
    return total_loss / max(1, total), correct / max(1, total), total


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_f1: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_f1": best_f1, "args": vars(args)}, path)


def main(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    run_name = args.run_name or f"satmae_ft_avg_480m_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.checkpoint_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "args.json", vars(args))

    train_ds = SatMAE480mDataset(
        args.train_csv,
        image_size=args.image_size,
        local_cache_dir=args.local_cache_dir,
        local_cache_max_gb=args.local_cache_max_gb,
        local_cache_min_free_gb=args.local_cache_min_free_gb,
    )
    test_ds = SatMAE480mDataset(
        args.test_csv,
        image_size=args.image_size,
        local_cache_dir=args.local_cache_dir,
        local_cache_max_gb=args.local_cache_max_gb,
        local_cache_min_free_gb=args.local_cache_min_free_gb,
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_rows, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_rows, persistent_workers=args.num_workers > 0)
    print(f"SatMAE-FT avg fusion | train={len(train_ds)} test={len(test_ds)} batch={args.batch_size} eval_batch={args.eval_batch_size} out_dir={out_dir}", flush=True)

    model = SatMAEAvgFusionClassifier(
        satmae_repo=args.satmae_repo,
        pretrained=args.pretrained,
        image_size=args.image_size,
        patch_size=args.patch_size,
        freeze_satmae=args.freeze_satmae,
        temporal_fusion=args.temporal_fusion,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    satmae_params = [p for n, p in model.named_parameters() if n.startswith("satmae.") and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("satmae.") and p.requires_grad]
    groups = []
    if satmae_params:
        groups.append({"params": satmae_params, "lr": args.lr_satmae})
    groups.append({"params": other_params, "lr": args.lr_head})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda" and args.amp)

    start_epoch = 1
    best_f1 = -1.0
    latest = out_dir / "ckpt_latest.pth"
    if args.resume and latest.is_file():
        ckpt = torch.load(latest, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_f1", -1.0))
        print(f"Resumed from {latest} at epoch {start_epoch}", flush=True)

    wandb_run = init_wandb(args)
    for epoch in range(start_epoch, args.epochs + 1):
        if args.freeze_satmae_epochs and epoch <= args.freeze_satmae_epochs:
            model.set_satmae_trainable(False)
        elif not args.freeze_satmae:
            model.set_satmae_trainable(True)
        train_loss, train_acc, train_n = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch=epoch, log_interval=args.log_interval, max_steps=args.max_train_steps,
            use_amp=device.type == "cuda" and args.amp, accum_steps=args.accum_steps, wandb_run=wandb_run,
        )
        metrics = evaluate(model, test_loader, criterion, device, max_steps=args.max_eval_steps)
        metrics_no_rows = {k: v for k, v in metrics.items() if k != "rows"}
        metrics_no_rows.update({"train_loss": train_loss, "train_acc": train_acc, "train_n": train_n, "epoch": epoch})
        print(
            f"[eval] epoch={epoch} loss={metrics_no_rows['loss']:.4f} "
            f"acc={metrics_no_rows['overall']['acc']:.4f} f1={metrics_no_rows['overall']['f1']:.4f} "
            f"recall={metrics_no_rows['overall']['recall']:.4f} fpr={metrics_no_rows['overall']['fpr']:.4f} "
            f"auroc={metrics_no_rows['overall']['auroc']:.4f}",
            flush=True,
        )
        write_json(out_dir / f"metrics_epoch_{epoch:03d}.json", metrics_no_rows)
        write_json(out_dir / "metrics_latest.json", metrics_no_rows)
        write_predictions(out_dir / f"predictions_epoch_{epoch:03d}.csv", metrics["rows"])
        write_predictions(out_dir / "predictions_latest.csv", metrics["rows"])
        save_checkpoint(latest, model, optimizer, epoch, best_f1, args)
        f1 = float(metrics_no_rows["overall"]["f1"])
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(out_dir / "ckpt_best_f1.pth", model, optimizer, epoch, best_f1, args)
            write_json(out_dir / "metrics_best_f1.json", metrics_no_rows)
            write_predictions(out_dir / "predictions_best_f1.csv", metrics["rows"])
        if wandb_run is not None:
            log_obj = {"epoch": epoch, "train/loss": train_loss, "train/acc": train_acc, "eval/loss": metrics_no_rows["loss"]}
            log_obj.update({f"eval/overall_{k}": v for k, v in metrics_no_rows["overall"].items() if isinstance(v, (int, float))})
            wandb_run.log(log_obj)
    if wandb_run is not None:
        wandb_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SatMAE-FT average-fusion 480m baseline.")
    parser.add_argument("--train_csv", default="/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m_518/manifest_time_train.csv")
    parser.add_argument("--test_csv", default="/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m_518/manifest_time_test.csv")
    parser.add_argument("--satmae_repo", default="/diniuvol/yuyao/SatMAE")
    parser.add_argument("--pretrained", default="/diniuvol/yuyao/satmae_weights/satmae-vitbase-multispec-pretrain.safetensors")
    parser.add_argument("--checkpoint_dir", default="/transferdiniu2/yuyao/checkpoints/satmae_ft_avg_480m")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--eval_batch_size", type=int, default=10)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lr_satmae", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--freeze_satmae", action="store_true")
    parser.add_argument("--freeze_satmae_epochs", type=int, default=1)
    parser.add_argument("--temporal_fusion", choices=("input_mean", "feature_mean"), default="input_mean")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_cache_dir", default=None)
    parser.add_argument("--local_cache_max_gb", type=float, default=350.0)
    parser.add_argument("--local_cache_min_free_gb", type=float, default=100.0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="query_dataset")
    parser.add_argument("--wandb_run_name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
