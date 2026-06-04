#!/usr/bin/env python
"""AnySat-FT + average-fusion baseline for the 480m multisensor query dataset.

This is intentionally pragmatic:
- S2 and L89 are routed through AnySat's pretrained S2/L8 projectors.
- EMIT and S5P are forced in through small trainable adapters that emit 768-d tile
  features.
- Row-level fusion is a plain arithmetic mean over all sensors present in that
  manifest row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
import shutil
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile as tiff
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("XFORMERS_DISABLED", "1")

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


def parse_datetime_to_doy(value: Any) -> int:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return 0
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return 0
    return int(dt.timetuple().tm_yday - 1)


def dates_for_row(value: Any) -> torch.Tensor:
    doy = parse_datetime_to_doy(value)
    return torch.tensor([doy % 366, (doy - 90) % 366, (doy - 360) % 366], dtype=torch.long)


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
    elif arr.ndim == 3:
        # Most local crops are already CHW. If not, move the smallest plausible channel axis.
        if arr.shape[0] > 64 and arr.shape[-1] <= 64:
            arr = np.transpose(arr, (2, 0, 1))
    else:
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


@dataclass
class RowSample:
    label: int
    row_id: Any
    plume_id: str
    anchor_sensor: str
    overlap_mode: bool
    sensors: Dict[str, torch.Tensor]
    dates: torch.Tensor


class AnySat480mDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        *,
        image_size_10m: int = 48,
        s5p_size: int = 48,
        max_retries: int = 5,
        processed_cache_dir: Optional[str] = None,
        processed_cache_min_free_gb: float = 100.0,
        processed_cache_readonly: bool = False,
    ):
        self.csv_path = str(csv_path)
        self.df = pd.read_csv(csv_path, low_memory=False)
        self.image_size_10m = int(image_size_10m)
        self.s5p_size = int(s5p_size)
        self.max_retries = max(1, int(max_retries))
        self.processed_cache_dir = Path(processed_cache_dir) if processed_cache_dir else None
        self.processed_cache_min_free_bytes = int(processed_cache_min_free_gb * (1024 ** 3))
        self.processed_cache_readonly = bool(processed_cache_readonly)
        if self.processed_cache_dir is not None:
            self.processed_cache_dir.mkdir(parents=True, exist_ok=True)

        self.s2_select = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        self.s2_mean = [S2_STATS[0][i] for i in self.s2_select]
        self.s2_std = [S2_STATS[1][i] for i in self.s2_select]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> RowSample:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._get_cached_or_one(idx)
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= self.max_retries:
                    raise
                idx = random.randrange(len(self.df))
        raise RuntimeError("unreachable") from last_exc

    def _cache_path(self, idx: int) -> Optional[Path]:
        if self.processed_cache_dir is None:
            return None
        row = self.df.iloc[idx]
        row_id = str(row.get("id", idx))
        key_text = f"anysat_v2|{self.csv_path}|{idx}|{row_id}|{self.image_size_10m}|{self.s5p_size}"
        key = hashlib.sha1(key_text.encode("utf-8")).hexdigest()
        return self.processed_cache_dir / f"{key}.pt"

    def _sample_to_cache_obj(self, sample: RowSample) -> Dict[str, Any]:
        return {
            "label": int(sample.label),
            "row_id": sample.row_id,
            "plume_id": sample.plume_id,
            "anchor_sensor": sample.anchor_sensor,
            "overlap_mode": bool(sample.overlap_mode),
            "sensors": sample.sensors,
            "dates": sample.dates,
        }

    def _sample_from_cache_obj(self, obj: Mapping[str, Any]) -> RowSample:
        return RowSample(
            label=int(obj["label"]),
            row_id=obj["row_id"],
            plume_id=str(obj["plume_id"]),
            anchor_sensor=str(obj["anchor_sensor"]),
            overlap_mode=bool(obj["overlap_mode"]),
            sensors=dict(obj["sensors"]),
            dates=obj["dates"],
        )

    def _get_cached_or_one(self, idx: int) -> RowSample:
        cache_path = self._cache_path(idx)
        if cache_path is not None and cache_path.is_file():
            try:
                return self._sample_from_cache_obj(torch.load(cache_path, map_location="cpu"))
            except Exception:
                pass
        sample = self._get_one(idx)
        if cache_path is not None and not self.processed_cache_readonly:
            try:
                if shutil.disk_usage(cache_path.parent).free > self.processed_cache_min_free_bytes:
                    tmp = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
                    torch.save(self._sample_to_cache_obj(sample), tmp)
                    os.replace(tmp, cache_path)
            except Exception:
                pass
        return sample

    def _get_one(self, idx: int) -> RowSample:
        row = self.df.iloc[idx]
        sensors: Dict[str, torch.Tensor] = {}
        dates = dates_for_row(row.get("datetime", ""))

        if all(is_present(row.get(c)) for c in ("s2_0_path", "s2_90_path", "s2_360_path")):
            frames = []
            for col in ("s2_0_path", "s2_90_path", "s2_360_path"):
                img = read_tiff_chw(str(row[col]))
                if img.shape[0] < 12:
                    raise ValueError(f"S2 image has {img.shape[0]} channels at {row[col]}")
                img = img[self.s2_select]
                img = normalize(img, self.s2_mean, self.s2_std)
                img = resize_chw(img, self.image_size_10m)
                frames.append(img)
            sensors["s2"] = torch.stack(frames, dim=0)  # T,C,H,W

        if all(is_present(row.get(c)) for c in ("l89_0_path", "l89_90_path", "l89_360_path")):
            frames = []
            for col in ("l89_0_path", "l89_90_path", "l89_360_path"):
                img = read_tiff_chw(str(row[col]))
                img = img[:10]
                img = normalize(img, L89_STATS_7[0], L89_STATS_7[1])
                if img.shape[0] < 10:
                    pad = torch.zeros((10 - img.shape[0], *img.shape[1:]), dtype=img.dtype)
                    img = torch.cat([img, pad], dim=0)
                img = torch.cat([img[:10], torch.zeros((1, *img.shape[1:]), dtype=img.dtype)], dim=0)
                img = resize_chw(img, self.image_size_10m)
                frames.append(img)
            sensors["l8"] = torch.stack(frames, dim=0)

        if all(is_present(row.get(c)) for c in ("emit_0_path", "emit_90_path", "emit_360_path")):
            frames = []
            for col in ("emit_0_path", "emit_90_path", "emit_360_path"):
                img = read_tiff_chw(str(row[col]))[:16]
                # Per-image standardization keeps this forced adapter numerically stable.
                flat = img.flatten(1)
                mean = flat.mean(dim=1).view(-1, 1, 1)
                std = flat.std(dim=1).clamp(min=1e-6).view(-1, 1, 1)
                img = (img - mean) / std
                img = resize_chw(torch.nan_to_num(img), self.image_size_10m)
                frames.append(img)
            sensors["emit"] = torch.stack(frames, dim=0)

        if is_present(row.get("s5p_0_path")):
            path = str(row["s5p_0_path"])
            np_obj = np.load(path, allow_pickle=False)
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
            sensors["s5p"] = resize_chw(img, self.s5p_size)

        if not sensors:
            raise ValueError(f"No usable sensor paths in row index {idx}")

        return RowSample(
            label=int(row["label"]),
            row_id=row.get("id", idx),
            plume_id=str(row.get("plume_id", "")),
            anchor_sensor=str(row.get("anchor_sensor", "")),
            overlap_mode=str(row.get("overlap_mode", "")).strip().lower() == "true",
            sensors=sensors,
            dates=dates,
        )


def collate_rows(batch: Sequence[RowSample]) -> Dict[str, Any]:
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)
    dates_by_sensor: Dict[str, List[torch.Tensor]] = {}
    tensors_by_sensor: Dict[str, List[torch.Tensor]] = {}
    sample_to_row: Dict[str, List[int]] = {}

    for row_idx, item in enumerate(batch):
        for sensor, tensor in item.sensors.items():
            tensors_by_sensor.setdefault(sensor, []).append(tensor)
            sample_to_row.setdefault(sensor, []).append(row_idx)
            if sensor in {"s2", "l8"}:
                dates_by_sensor.setdefault(sensor, []).append(item.dates)

    sensors = {k: torch.stack(v, dim=0) for k, v in tensors_by_sensor.items()}
    dates = {k: torch.stack(v, dim=0) for k, v in dates_by_sensor.items()}
    row_index = {k: torch.tensor(v, dtype=torch.long) for k, v in sample_to_row.items()}

    return {
        "labels": labels,
        "sensors": sensors,
        "dates": dates,
        "sample_to_row": row_index,
        "row_ids": [b.row_id for b in batch],
        "plume_ids": [b.plume_id for b in batch],
        "anchor_sensor": [b.anchor_sensor for b in batch],
        "overlap_mode": [b.overlap_mode for b in batch],
    }


class TemporalConvAdapter(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_chans, 96, kernel_size=(3, 5, 5), padding=(0, 2, 2), bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.LayerNorm(192), nn.Linear(192, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B,T,C,H,W -> B,C,T,H,W -> B,hidden,H,W after temporal kernel.
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.net[0](x).squeeze(2)
        for layer in self.net[1:]:
            x = layer(x)
        return self.proj(x)


class SpatialConvAdapter(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(128),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AnySatAvgFusionClassifier(nn.Module):
    def __init__(
        self,
        *,
        anysat_repo: str,
        pretrained: bool = True,
        embed_dim: int = 768,
        patch_size: int = 20,
        freeze_anysat: bool = False,
    ):
        super().__init__()
        self.anysat = torch.hub.load(
            anysat_repo,
            "anysat",
            source="local" if os.path.isdir(anysat_repo) else "github",
            pretrained=pretrained,
            flash_attn=False,
        )
        self.patch_size = int(patch_size)
        self.emit_adapter = TemporalConvAdapter(16, embed_dim=embed_dim)
        self.s5p_adapter = SpatialConvAdapter(3, embed_dim=embed_dim)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))
        if freeze_anysat:
            for p in self.anysat.parameters():
                p.requires_grad_(False)

    def set_anysat_trainable(self, trainable: bool) -> None:
        for p in self.anysat.parameters():
            p.requires_grad_(trainable)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        labels = batch["labels"]
        device = labels.device
        batch_size = int(labels.shape[0])
        fused = torch.zeros((batch_size, self.head[0].normalized_shape[0]), device=device)
        counts = torch.zeros((batch_size, 1), device=device)

        sensors: Mapping[str, torch.Tensor] = batch["sensors"]
        dates: Mapping[str, torch.Tensor] = batch["dates"]
        sample_to_row: Mapping[str, torch.Tensor] = batch["sample_to_row"]

        for sensor in ("s2", "l8"):
            if sensor not in sensors:
                continue
            x = {
                sensor: sensors[sensor],
                f"{sensor}_dates": dates[sensor],
            }
            feat = self.anysat(x, patch_size=self.patch_size, output="tile").to(dtype=fused.dtype)
            rows = sample_to_row[sensor]
            fused.index_add_(0, rows, feat)
            counts.index_add_(0, rows, torch.ones((rows.numel(), 1), device=device, dtype=fused.dtype))

        if "emit" in sensors:
            feat = self.emit_adapter(sensors["emit"]).to(dtype=fused.dtype)
            rows = sample_to_row["emit"]
            fused.index_add_(0, rows, feat)
            counts.index_add_(0, rows, torch.ones((rows.numel(), 1), device=device, dtype=fused.dtype))

        if "s5p" in sensors:
            feat = self.s5p_adapter(sensors["s5p"]).to(dtype=fused.dtype)
            rows = sample_to_row["s5p"]
            fused.index_add_(0, rows, feat)
            counts.index_add_(0, rows, torch.ones((rows.numel(), 1), device=device, dtype=fused.dtype))

        fused = fused / counts.clamp(min=1.0)
        return self.head(fused)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    out["labels"] = batch["labels"].to(device, non_blocking=True)
    out["sensors"] = {k: v.to(device, non_blocking=True) for k, v in batch["sensors"].items()}
    out["dates"] = {k: v.to(device, non_blocking=True) for k, v in batch["dates"].items()}
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
    total = max(1, len(y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    fpr = fp / max(1, fp + tn)
    acc = (tp + tn) / total
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
    except Exception:
        auroc = float("nan")
    return {
        "n": int(len(y)),
        "acc": float(acc),
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
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
    metrics: Dict[str, Any] = {
        "loss": total_loss / max(1, total),
        "overall": binary_metrics(y, s),
        "by_overlap": {},
        "by_anchor_sensor": {},
    }
    for name, keep in {
        "overlap": [r for r in rows if r["overlap_mode"]],
        "single": [r for r in rows if not r["overlap_mode"]],
    }.items():
        metrics["by_overlap"][name] = binary_metrics([r["label"] for r in keep], [r["prob1"] for r in keep]) if keep else None

    anchors = sorted({str(r["anchor_sensor"]) for r in rows})
    for anchor in anchors:
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
    return wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))


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
    wandb_run=None,
) -> Tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    start = time.time()
    for step, batch in enumerate(loader, 1):
        if max_steps is not None and step > max_steps:
            break
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(batch)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = int(labels.shape[0])
        total += bs
        total_loss += float(loss.item()) * bs
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        if log_interval and step % log_interval == 0:
            msg = (
                f"[train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={total_loss/max(1,total):.4f} acc={correct/max(1,total):.4f} "
                f"elapsed_min={(time.time()-start)/60:.1f}"
            )
            print(msg, flush=True)
            if wandb_run is not None:
                wandb_run.log({"train/loss_running": total_loss / max(1, total), "train/acc_running": correct / max(1, total), "epoch": epoch})
    return total_loss / max(1, total), correct / max(1, total), total


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_f1: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_f1": best_f1,
        "args": vars(args),
    }, path)


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

    run_name = args.run_name or f"anysat_ft_avg_480m_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.checkpoint_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "args.json", vars(args))

    train_ds = AnySat480mDataset(
        args.train_csv,
        image_size_10m=args.image_size_10m,
        s5p_size=args.s5p_size,
        processed_cache_dir=args.processed_cache_dir,
        processed_cache_min_free_gb=args.processed_cache_min_free_gb,
        processed_cache_readonly=args.processed_cache_readonly,
    )
    test_ds = AnySat480mDataset(
        args.test_csv,
        image_size_10m=args.image_size_10m,
        s5p_size=args.s5p_size,
        processed_cache_dir=args.processed_cache_dir,
        processed_cache_min_free_gb=args.processed_cache_min_free_gb,
        processed_cache_readonly=args.processed_cache_readonly,
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_rows, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_rows, persistent_workers=args.num_workers > 0)

    print(
        f"AnySat-FT avg fusion | train={len(train_ds)} test={len(test_ds)} "
        f"batch={args.batch_size} eval_batch={args.eval_batch_size} patch_size={args.patch_size} out_dir={out_dir}",
        flush=True,
    )

    model = AnySatAvgFusionClassifier(
        anysat_repo=args.anysat_repo,
        pretrained=not args.no_pretrained,
        patch_size=args.patch_size,
        freeze_anysat=args.freeze_anysat,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    anysat_params = [p for p in model.anysat.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("anysat.") and p.requires_grad]
    param_groups = []
    if anysat_params:
        param_groups.append({"params": anysat_params, "lr": args.lr_anysat})
    param_groups.append({"params": other_params, "lr": args.lr_head})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
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
        if args.freeze_anysat_epochs and epoch <= args.freeze_anysat_epochs:
            model.set_anysat_trainable(False)
        elif not args.freeze_anysat:
            model.set_anysat_trainable(True)

        train_loss, train_acc, train_n = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch=epoch, log_interval=args.log_interval, max_steps=args.max_train_steps,
            use_amp=device.type == "cuda" and args.amp, wandb_run=wandb_run,
        )
        metrics = evaluate(model, test_loader, criterion, device, max_steps=args.max_eval_steps)
        metrics_no_rows = {k: v for k, v in metrics.items() if k != "rows"}
        metrics_no_rows["train_loss"] = train_loss
        metrics_no_rows["train_acc"] = train_acc
        metrics_no_rows["train_n"] = train_n
        metrics_no_rows["epoch"] = epoch
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
            log_obj = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "eval/loss": metrics_no_rows["loss"],
            }
            for k, v in metrics_no_rows["overall"].items():
                if isinstance(v, (int, float)):
                    log_obj[f"eval/overall_{k}"] = v
            wandb_run.log(log_obj)

    if wandb_run is not None:
        wandb_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnySat-FT average-fusion 480m baseline.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--anysat_repo", required=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--checkpoint_dir", default="checkpoints/anysat_ft_avg_480m")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lr_anysat", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patch_size", type=int, default=20)
    parser.add_argument("--image_size_10m", type=int, default=48)
    parser.add_argument("--s5p_size", type=int, default=48)
    parser.add_argument("--freeze_anysat", action="store_true")
    parser.add_argument("--freeze_anysat_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--processed_cache_dir", default=None)
    parser.add_argument("--processed_cache_min_free_gb", type=float, default=100.0)
    parser.add_argument("--processed_cache_readonly", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="query_dataset")
    parser.add_argument("--wandb_run_name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
