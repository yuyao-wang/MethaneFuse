"""Multi-sensor Panopticon training script with sensor-specific embeddings and heads.

This module exposes reusable model components (``MultiSensorPanopticonClassifier``)
*and* a runnable training entry point that ingests mixed-sensor CSVs containing
Sentinel-2, Landsat 8/9, Sentinel-5P, and WV3 samples. Each sensor owns its own
Panopticon patch embedding and classifier head, while the DinoViT backbone is
shared and updated by the consensus of all datasets in a batch.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Make the repository root importable so examples work when executed directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid CUDA init stalls on CPU-only hosts.
os.environ.setdefault("XFORMERS_DISABLED", "1")

from thirdparty.dinov2.models.panopticon import PanopticonPE
from thirdparty.dinov2.models.vision_transformer import DinoVisionTransformer
from src.data.multisensor import (
    ConcatTemporalDataset,
    StaticAnchoredCache,
    TriSensorTemporalCsvDataset,
    collect_cache_paths_from_df,
    custom_collate_fn,
)
from src.data.sensor_transforms import DEFAULT_WV3_BANDS, load_wv3_channel_ids_from_srf
from src.evaluation.metrics import compute_binary_metrics, compute_split_metrics, mean_logits_by_row
from src.utils.training import _load_backbone, _slice_x_dict, recursive_to_device, set_trainable



# --------------------------------------------------------------------------------------
#  Model components
# --------------------------------------------------------------------------------------

class CLSHead(nn.Module):
    """LayerNorm + Linear classification head used per sensor."""

    def __init__(self, embed_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fc(self.norm(cls_token))

    @property
    def out_features(self) -> int:
        return self.fc.out_features


class LogitSummaryHead(nn.Module):
    """Fuse logits from all sensor heads into one final prediction."""

    def __init__(self, *, num_heads: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        in_dim = num_heads * num_classes
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class MaskedAttentionPooling(nn.Module):
    """Mask-aware attention pooling over a variable number of sensor features per row."""

    def __init__(self, *, embed_dim: int, num_sensors: int):
        super().__init__()
        self.sensor_embed = nn.Embedding(num_sensors, embed_dim)
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        feats: torch.Tensor,
        sensor_indices: torch.Tensor,
        sample_to_row: torch.Tensor,
        *,
        num_rows: int,
    ) -> torch.Tensor:
        if feats.ndim != 2:
            raise ValueError(f"feats must be [N,D], got shape={tuple(feats.shape)}")
        if sensor_indices.shape[0] != feats.shape[0] or sample_to_row.shape[0] != feats.shape[0]:
            raise ValueError(
                "MaskedAttentionPooling input length mismatch: "
                f"feats={feats.shape[0]}, sensor_indices={sensor_indices.shape[0]}, sample_to_row={sample_to_row.shape[0]}"
            )
        attn_in = feats + self.sensor_embed(sensor_indices)
        scores = self.score(attn_in).squeeze(-1)
        out = feats.new_zeros((num_rows, feats.shape[-1]))
        for row_idx in range(num_rows):
            mask = sample_to_row == row_idx
            if not torch.any(mask):
                continue
            weights = torch.softmax(scores[mask], dim=0)
            out[row_idx] = torch.sum(feats[mask] * weights.unsqueeze(-1), dim=0)
        return out


class RowwiseMaxPooling(nn.Module):
    """Row-wise max pooling over active sensor features."""

    def forward(
        self,
        feats: torch.Tensor,
        sensor_indices: torch.Tensor,
        sample_to_row: torch.Tensor,
        *,
        num_rows: int,
    ) -> torch.Tensor:
        if feats.ndim != 2:
            raise ValueError(f"feats must be [N,D], got shape={tuple(feats.shape)}")
        if sensor_indices.shape[0] != feats.shape[0] or sample_to_row.shape[0] != feats.shape[0]:
            raise ValueError(
                "RowwiseMaxPooling input length mismatch: "
                f"feats={feats.shape[0]}, sensor_indices={sensor_indices.shape[0]}, sample_to_row={sample_to_row.shape[0]}"
            )
        out = feats.new_zeros((num_rows, feats.shape[-1]))
        for row_idx in range(num_rows):
            mask = sample_to_row == row_idx
            if not torch.any(mask):
                continue
            out[row_idx] = torch.max(feats[mask], dim=0).values
        return out


HeadFactory = Callable[[int, int], nn.Module]


class MultiSensorPanopticonClassifier(nn.Module):
    """Shared DinoViT backbone with sensor-specific Panopticon PEs and heads."""

    def __init__(
        self,
        *,
        backbone: Optional[DinoVisionTransformer] = None,
        sensors: Sequence[str] = ("s2", "l89", "s5p", "wv3"),
        num_classes: Mapping[str, int] | int = 2,
        patch_embed_overrides: Optional[Mapping[str, PanopticonPE]] = None,
        head_factory: Optional[Union[HeadFactory, nn.Module]] = None,
        enable_summary_head: bool = True,
        summary_hidden_dim: int = 128,
        summary_dropout: float = 0.1,
        summary_loss_weight: float = 1.0,
        row_fusion_mode: str = "map",
    ):
        super().__init__()
        if backbone is None:
            backbone = _load_backbone()
        if not isinstance(backbone, DinoVisionTransformer):
            raise TypeError("backbone must be a DinoVisionTransformer instance")

        self.backbone = backbone
        self.sensor_order = list(sensors)
        if not self.sensor_order:
            raise ValueError("At least one sensor must be specified")
        self.sensor_to_idx = {sensor: idx for idx, sensor in enumerate(self.sensor_order)}

        base_patch_embed = backbone.patch_embed
        sensor_modules = nn.ModuleDict()
        for sensor in self.sensor_order:
            if patch_embed_overrides and sensor in patch_embed_overrides:
                sensor_modules[sensor] = patch_embed_overrides[sensor]
            elif sensor == self.sensor_order[0]:
                sensor_modules[sensor] = base_patch_embed
            else:
                sensor_modules[sensor] = copy.deepcopy(base_patch_embed)
        self.sensor_patch_embeds = sensor_modules
        self.backbone.patch_embed = self.sensor_patch_embeds[self.sensor_order[0]]

        embed_dim = getattr(self.backbone, "embed_dim", 768)
        if isinstance(num_classes, int):
            class_map = {sensor: num_classes for sensor in self.sensor_order}
        else:
            class_map = {sensor: num_classes[sensor] for sensor in self.sensor_order}

        def make_head(num_output_classes: int) -> nn.Module:
            if head_factory is None:
                return CLSHead(embed_dim=embed_dim, num_classes=num_output_classes)
            if isinstance(head_factory, nn.Module):
                return copy.deepcopy(head_factory)
            head = head_factory(embed_dim, num_output_classes)
            if not isinstance(head, nn.Module):
                raise TypeError("head_factory must create an nn.Module")
            return head

        self.heads = nn.ModuleDict()
        for sensor in self.sensor_order:
            classes = class_map[sensor]
            self.heads[sensor] = make_head(classes)
        self.summary_loss_weight = float(summary_loss_weight)
        self.summary_head: Optional[LogitSummaryHead] = None
        self.row_fusion_mode = str(row_fusion_mode).strip().lower()
        if self.row_fusion_mode not in ("map", "max"):
            raise ValueError(f"row_fusion_mode must be one of ['map', 'max'], got {row_fusion_mode!r}")
        self.row_fusion_pool: Optional[nn.Module] = None
        self.row_fusion_head: Optional[CLSHead] = None

        head_dims = [getattr(self.heads[sensor], "out_features", None) for sensor in self.sensor_order]
        can_build_summary = bool(head_dims) and None not in head_dims and len(set(head_dims)) == 1
        if enable_summary_head and can_build_summary:
            num_out_classes = int(head_dims[0])
            self.summary_head = LogitSummaryHead(
                num_heads=len(self.sensor_order),
                num_classes=num_out_classes,
                hidden_dim=summary_hidden_dim,
                dropout=summary_dropout,
            )
        elif enable_summary_head and not can_build_summary:
            warnings.warn(
                "Summary head disabled because sensor heads do not share the same out_features.",
                stacklevel=2,
            )
        if can_build_summary:
            num_out_classes = int(head_dims[0])
            if self.row_fusion_mode == "map":
                self.row_fusion_pool = MaskedAttentionPooling(embed_dim=embed_dim, num_sensors=len(self.sensor_order))
            else:
                self.row_fusion_pool = RowwiseMaxPooling()
            self.row_fusion_head = CLSHead(embed_dim=embed_dim, num_classes=num_out_classes)
        else:
            warnings.warn(
                "Row fusion head disabled because sensor heads do not share the same out_features.",
                stacklevel=2,
            )

    @contextmanager
    def _use_sensor(self, sensor: str):
        original = self.backbone.patch_embed
        self.backbone.patch_embed = self.sensor_patch_embeds[sensor]
        try:
            yield
        finally:
            self.backbone.patch_embed = original

    def encode_sensors(self, sensors: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
        idxs = [self.sensor_to_idx[s] for s in sensors]
        return torch.tensor(idxs, device=device, dtype=torch.long)

    def _normalize_sensors(self, sensors: Union[Sequence[str], torch.Tensor]) -> Sequence[str]:
        if isinstance(sensors, torch.Tensor):
            indices = sensors.detach().to("cpu").tolist()
            return [self.sensor_order[i] for i in indices]
        return list(sensors)

    def _build_sensor_batches(
        self,
        x_dict: MutableMapping[str, torch.Tensor],
        sensors: Sequence[str],
    ) -> Dict[str, "SensorBatch"]:
        device = next(iter(x_dict.values())).device
        batches: Dict[str, SensorBatch] = {}
        for sensor_name in self.sensor_order:
            idx = [i for i, s in enumerate(sensors) if s == sensor_name]
            if not idx:
                continue
            idx_tensor = torch.tensor(idx, device=device, dtype=torch.long)
            batches[sensor_name] = SensorBatch(idx_tensor, _slice_x_dict(x_dict, idx_tensor))
        return batches

    def forward(
        self,
        x_dict: MutableMapping[str, torch.Tensor],
        sensors: Sequence[str],
        *,
        return_features: bool = False,
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        sensor_labels = self._normalize_sensors(sensors)
        sensor_batches = self._build_sensor_batches(x_dict, sensor_labels)
        if not sensor_batches:
            raise ValueError("No samples matched the configured sensors")

        outputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]] = {}
        participating = list(sensor_batches.keys())
        merge_dims = [getattr(self.heads[sensor_name], "out_features", None) for sensor_name in participating]
        allow_merge = bool(merge_dims) and None not in merge_dims and len(set(merge_dims)) == 1
        merged_logits: Optional[torch.Tensor] = None
        batch_size = len(sensors)
        summary_inputs: Optional[torch.Tensor] = None

        for sensor_name, sensor_batch in sensor_batches.items():
            with self._use_sensor(sensor_name):
                feats = self.backbone(sensor_batch.x_dict, is_training=True)
            cls_token = torch.nan_to_num(feats["x_norm_clstoken"], nan=0.0, posinf=1e4, neginf=-1e4)
            head_logits = [
                torch.nan_to_num(self.heads[name](cls_token), nan=0.0, posinf=1e4, neginf=-1e4)
                for name in self.sensor_order
            ]
            logits = head_logits[self.sensor_to_idx[sensor_name]]
            if allow_merge:
                if merged_logits is None:
                    merged_logits = logits.new_zeros((batch_size, logits.shape[-1]))
                merged_logits.index_copy_(0, sensor_batch.indices, logits)
            if self.summary_head is not None:
                if summary_inputs is None:
                    summary_inputs = logits.new_zeros((batch_size, len(self.sensor_order) * logits.shape[-1]))
                summary_inputs.index_copy_(0, sensor_batch.indices, torch.cat(head_logits, dim=-1))
            outputs[sensor_name] = {
                "indices": sensor_batch.indices,
                "cls_token": cls_token,
                "logits": logits,
            }
            if return_features:
                outputs[sensor_name]["feats"] = feats["x_norm_patchtokens"]

        if merged_logits is not None:
            outputs["merged_logits"] = merged_logits
        if summary_inputs is not None and self.summary_head is not None:
            outputs["summary_logits"] = self.summary_head(summary_inputs)
        self._ensure_sensor_keys(outputs, x_dict, return_features=return_features)
        return outputs

    def compute_loss(
        self,
        x_dict: MutableMapping[str, torch.Tensor],
        sensors: Sequence[str],
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]]:
        sensor_labels = self._normalize_sensors(sensors)
        outputs = self.forward(x_dict, sensors=sensor_labels)
        total_loss, per_sensor_losses = self._loss_from_outputs(outputs, labels, sensor_labels, criterion)
        return total_loss, per_sensor_losses, outputs

    def loss_from_outputs(
        self,
        outputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        labels: torch.Tensor,
        sensors: Union[Sequence[str], torch.Tensor],
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sensor_labels = self._normalize_sensors(sensors)
        return self._loss_from_outputs(outputs, labels, sensor_labels, criterion)

    def _loss_from_outputs(
        self,
        outputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        labels: torch.Tensor,
        sensors: Sequence[str],
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss: Optional[torch.Tensor] = None
        per_sensor_losses: Dict[str, torch.Tensor] = {}
        for sensor_name in self.sensor_order:
            sensor_out = outputs.get(sensor_name)
            if not isinstance(sensor_out, dict):
                continue
            idx = sensor_out["indices"]
            if idx.numel() == 0:
                continue
            sensor_labels = labels.index_select(0, idx)
            loss = criterion(sensor_out["logits"], sensor_labels)
            per_sensor_losses[sensor_name] = loss
            total_loss = loss if total_loss is None else total_loss + loss
        summary_logits = outputs.get("summary_logits")
        if isinstance(summary_logits, torch.Tensor):
            summary_loss = criterion(summary_logits, labels)
            per_sensor_losses["summary"] = summary_loss
            weighted_summary_loss = summary_loss * self.summary_loss_weight
            total_loss = weighted_summary_loss if total_loss is None else total_loss + weighted_summary_loss
        if total_loss is None:
            raise RuntimeError("No loss terms were computed; check the sensor labels")
        return total_loss, per_sensor_losses

    def _ensure_sensor_keys(
        self,
        outputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        x_dict: MutableMapping[str, torch.Tensor],
        *,
        return_features: bool = False,
    ) -> None:
        device = next(iter(x_dict.values())).device
        embed_dim = getattr(self.backbone, "embed_dim", 768)
        for sensor_name in self.sensor_order:
            if sensor_name in outputs:
                continue
            head = self.heads[sensor_name]
            num_classes = head.out_features
            empty_idx = torch.empty((0,), dtype=torch.long, device=device)
            empty_cls = torch.empty((0, embed_dim), device=device)
            empty_logits = torch.empty((0, num_classes), device=device)
            placeholder: Dict[str, torch.Tensor] = {
                "indices": empty_idx,
                "cls_token": empty_cls,
                "logits": empty_logits,
            }
            if return_features:
                placeholder["feats"] = torch.empty((0, 0), device=device)
            outputs[sensor_name] = placeholder

    def compute_row_fused_logits(
        self,
        outputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        *,
        sample_to_row: torch.Tensor,
        num_rows: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.row_fusion_pool is None or self.row_fusion_head is None:
            raise RuntimeError("Row fusion modules are not initialized")
        if sample_to_row.ndim != 1:
            raise ValueError(f"sample_to_row must be rank-1, got shape={tuple(sample_to_row.shape)}")

        num_samples = int(sample_to_row.shape[0])
        embed_dim = getattr(self.backbone, "embed_dim", 768)
        flat_feats = torch.zeros((num_samples, embed_dim), device=device)
        flat_sensor_indices = torch.zeros((num_samples,), dtype=torch.long, device=device)

        for sensor_name in self.sensor_order:
            sensor_out = outputs.get(sensor_name)
            if not isinstance(sensor_out, dict):
                continue
            idx = sensor_out.get("indices")
            cls_token = sensor_out.get("cls_token")
            if not isinstance(idx, torch.Tensor) or not isinstance(cls_token, torch.Tensor) or idx.numel() == 0:
                continue
            flat_feats.index_copy_(0, idx, cls_token)
            flat_sensor_indices.index_fill_(0, idx, int(self.sensor_to_idx[sensor_name]))

        row_features = self.row_fusion_pool(
            flat_feats,
            flat_sensor_indices,
            sample_to_row,
            num_rows=num_rows,
        )
        return self.row_fusion_head(row_features)


@dataclass
class SensorBatch:
    indices: torch.Tensor
    x_dict: Dict[str, torch.Tensor]


# --------------------------------------------------------------------------------------
#  Dataset helpers (mixed temporal CSV + S5P NPZ support)
# --------------------------------------------------------------------------------------


def collect_cache_paths_from_df(df, columns: Sequence[str]) -> list[str]:
    valid_columns = [col for col in columns if col in df.columns]
    if len(valid_columns) == 0:
        return []
    out: list[str] = []
    for col in valid_columns:
        series = df[col]
        for value in series.tolist():
            if value is None:
                continue
            path = str(value).strip()
            if path == "" or path.lower() in ("nan", "none", "null"):
                continue
            out.append(path)
    return out


# --------------------------------------------------------------------------------------
#  Training utilities
# --------------------------------------------------------------------------------------

def init_wandb(args):
    if not args.use_wandb:
        return None
    import wandb

    return wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))


def default_run_name(args) -> str:
    train_stem = Path(args.train_csv).stem or "train"
    test_stem = Path(args.test_csv).stem or "test"
    mode = "ft" if args.train_backbone else "head"
    return f"{train_stem}__{test_stem}__{mode}"


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    best_train_acc: float,
    best_test_acc: float,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "best_train_acc": best_train_acc,
            "best_test_acc": best_test_acc,
            "args": vars(args),
        },
        path,
    )


def try_resume(path: Path, model: nn.Module, optimizer, scheduler, scaler, device):
    if not path.is_file():
        return 1, 0, 0.0, 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    global_step = ckpt.get("global_step", 0)
    best_train_acc = ckpt.get("best_train_acc", 0.0)
    best_test_acc = ckpt.get("best_test_acc", 0.0)
    print(f"Resumed from {path} at epoch {start_epoch-1}", flush=True)
    return start_epoch, global_step, best_train_acc, best_test_acc


def build_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "noam":
        warmup_steps = max(args.warmup_steps, 1)
        noam_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")



def gather_head_parameters(model: MultiSensorPanopticonClassifier) -> Sequence[nn.Parameter]:
    shared_patch = model.backbone.patch_embed
    extra_patch_params = []
    for sensor, module in model.sensor_patch_embeds.items():
        if module is shared_patch:
            continue
        extra_patch_params.extend(list(module.parameters()))
    summary_params = list(model.summary_head.parameters()) if model.summary_head is not None else []
    row_fusion_params = []
    if model.row_fusion_pool is not None:
        row_fusion_params.extend(list(model.row_fusion_pool.parameters()))
    if model.row_fusion_head is not None:
        row_fusion_params.extend(list(model.row_fusion_head.parameters()))
    head_params = list(model.heads.parameters()) + summary_params + row_fusion_params + extra_patch_params
    return head_params


# --------------------------------------------------------------------------------------
#  CLI / training loop
# --------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-sensor Panopticon finetuning (S2/L89/S5P/WV3).")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--weights", default="weights/panopticon_vitb14_teacher.pth")
    parser.add_argument(
        "--fusion_group_column",
        default="id",
        help="Deprecated and ignored. Group-based fusion is disabled; wide-table rows are treated as final units.",
    )
    parser.add_argument(
        "--s5p_data_key",
        default="ch4",
        help="NPZ key for S5P image tensor (defaults to 'ch4' for old 2025 S5P patches).",
    )
    parser.add_argument("--s5p_chn_ids_key", default="chn_ids")
    parser.add_argument("--s5p_channels_last", action="store_true")
    parser.add_argument(
        "--wv3_srf_csv",
        default=str(REPO_ROOT / "WV3_VNIR_SWIR_response.csv"),
        help="Path to WV3 SRF CSV used to derive per-band channel IDs.",
    )
    parser.add_argument(
        "--wv3_bands",
        default=",".join(DEFAULT_WV3_BANDS),
        help="Comma-separated WV3 SRF column names in channel order.",
    )
    parser.add_argument(
        "--align_l89_to_s2",
        action="store_true",
        help="Pad L89 to 12 channels and reuse S2 channel ids (legacy behavior).",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_dir", default="checkpoints/multi_sensor")
    parser.add_argument("--lr_scheduler", choices=["none", "noam"], default="noam")
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="panopticon-multisensor")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--local_cache_dir", default=None)
    parser.add_argument("--local_cache_min_free_gb", type=float, default=5.0)
    parser.add_argument("--local_cache_warmup", action="store_true")
    parser.add_argument("--local_cache_workers", type=int, default=8)
    parser.add_argument("--summary_head", action="store_true", dest="summary_head")
    parser.add_argument("--disable_summary_head", action="store_false", dest="summary_head")
    parser.set_defaults(summary_head=False)
    parser.add_argument("--summary_hidden_dim", type=int, default=128)
    parser.add_argument("--summary_dropout", type=float, default=0.1)
    parser.add_argument("--summary_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--row_fusion_mode",
        choices=["map", "max"],
        default="map",
        help="Row-level fusion pooling mode: 'map' for masked attention pooling, 'max' for max pooling.",
    )
    parser.add_argument(
        "--sensor_aux_loss_weight",
        type=float,
        default=0.3,
        help="Weight of per-sensor auxiliary CE loss when training row-level fusion.",
    )
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    wv3_band_names = [b.strip() for b in args.wv3_bands.split(",") if b.strip()]
    if len(wv3_band_names) == 0:
        raise ValueError("--wv3_bands must provide at least one WV3 band column name")
    wv3_chn_ids = load_wv3_channel_ids_from_srf(args.wv3_srf_csv, wv3_band_names).unsqueeze(-1)

    cache_obj = None
    if args.local_cache_dir:
        cache_obj = StaticAnchoredCache(args.local_cache_dir, min_free_gb=args.local_cache_min_free_gb)
        print(
            f"[Cache] Enabled local cache dir={cache_obj.cache_dir} (min_free_gb={args.local_cache_min_free_gb:.2f})",
            flush=True,
        )
        if not args.local_cache_warmup:
            print("[Cache] Warmup disabled; cache will populate lazily on first file access.", flush=True)

    base_train_ds = TriSensorTemporalCsvDataset(
        csv_path=args.train_csv,
        local_file_cache=cache_obj,
        s5p_data_key=args.s5p_data_key,
        s5p_chn_ids_key=args.s5p_chn_ids_key,
        s5p_channels_last=args.s5p_channels_last,
        align_l89_to_s2=args.align_l89_to_s2,
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )
    base_test_ds = TriSensorTemporalCsvDataset(
        csv_path=args.test_csv,
        local_file_cache=cache_obj,
        s5p_data_key=args.s5p_data_key,
        s5p_chn_ids_key=args.s5p_chn_ids_key,
        s5p_channels_last=args.s5p_channels_last,
        align_l89_to_s2=args.align_l89_to_s2,
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )

    if args.local_cache_warmup and cache_obj:
        warmup_columns: set[str] = set(col for col in base_train_ds.df.columns if str(col).endswith("_path"))
        wide_sensor_columns = getattr(base_train_ds, "_wide_sensor_columns", {})
        if isinstance(wide_sensor_columns, Mapping):
            for cols in wide_sensor_columns.values():
                warmup_columns.update(str(col) for col in cols if col in base_train_ds.df.columns)
        all_paths = collect_cache_paths_from_df(base_train_ds.df, sorted(warmup_columns))
        print(
            f"[Cache] Warmup scanning {len(warmup_columns)} path columns from train CSV, "
            f"found {len(all_paths)} path entries.",
            flush=True,
        )
        cache_obj.warm_up(all_paths, max_workers=args.local_cache_workers)

    train_ds = base_train_ds
    test_ds = base_test_ds

    core_model = MultiSensorPanopticonClassifier(
        backbone=_load_backbone(args.weights),
        enable_summary_head=False,
        summary_hidden_dim=args.summary_hidden_dim,
        summary_dropout=args.summary_dropout,
        summary_loss_weight=args.summary_loss_weight,
        row_fusion_mode=args.row_fusion_mode,
    ).to(device)
    model: nn.Module = core_model
    use_data_parallel = args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1
    if use_data_parallel:
        print(f"Enabling DataParallel across {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(core_model)
    elif args.data_parallel:
        print("DataParallel requested but insufficient CUDA devices; running single-device.", flush=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    head_params = gather_head_parameters(core_model)
    param_groups = [
        {"params": head_params, "lr": args.head_lr},
    ]
    if args.train_backbone:
        param_groups.insert(0, {"params": core_model.backbone.parameters(), "lr": args.backbone_lr})

    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay, betas=(args.momentum, 0.999))
    scheduler = build_scheduler(args, optimizer)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    run_name = args.wandb_run_name or default_run_name(args)
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    latest_path = ckpt_dir / "ckpt_latest.pth"
    best_path = ckpt_dir / "ckpt_best_test.pth"
    print(f"Checkpoints will be saved under: {ckpt_dir}", flush=True)

    start_epoch = 1
    global_step = 0
    best_train_acc = 0.0
    best_test_acc = float("-inf")

    if args.resume:
        start_epoch, global_step, best_train_acc, best_test_acc = try_resume(
            latest_path, core_model, optimizer, scheduler, scaler, device
        )

    wandb_run = init_wandb(args)

    sensors_list = core_model.sensor_order
    deprecated_opts = []
    if bool(args.summary_head):
        deprecated_opts.append("--summary_head")
    if str(args.fusion_group_column).strip() != "id":
        deprecated_opts.append("--fusion_group_column")
    if deprecated_opts:
        warnings.warn(
            "Deprecated options are ignored in this script: "
            + ", ".join(sorted(set(deprecated_opts))),
            stacklevel=2,
        )
    if args.sensor_aux_loss_weight < 0.0:
        raise ValueError(f"--sensor_aux_loss_weight must be >= 0, got {args.sensor_aux_loss_weight}.")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
    )
    train_steps = max(1, len(train_loader))
    test_steps = max(1, len(test_loader))

    print(
        f"Using device={device}, train_samples={len(train_ds)}, test_samples={len(test_ds)}, "
        f"sensors={sensors_list}, train_backbone={args.train_backbone}, "
        f"fusion={'masked_attention_pooling' if args.row_fusion_mode == 'map' else 'max_pooling'}, "
        f"sensor_aux_loss_weight={args.sensor_aux_loss_weight:.3f}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        freeze_backbone = (not args.train_backbone) or (
            args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs
        )
        if freeze_backbone:
            set_trainable(core_model.backbone, False)
            core_model.backbone.eval()
        else:
            set_trainable(core_model.backbone, True)
            core_model.backbone.train()
        model.train()
        core_model.heads.train()
        if core_model.row_fusion_pool is not None:
            core_model.row_fusion_pool.train()
        if core_model.row_fusion_head is not None:
            core_model.row_fusion_head.train()
        for sensor in core_model.sensor_patch_embeds.values():
            sensor.train()

        total_loss = 0.0
        total = 0
        correct = 0
        per_sensor_loss_accum = {sensor: 0.0 for sensor in sensors_list}
        per_sensor_count = {sensor: 0 for sensor in sensors_list}
        fused_loss_accum = 0.0
        fused_loss_count = 0
        sensor_aux_loss_accum = 0.0
        sensor_aux_loss_count = 0

        step_idx = 0
        for x_dict, labels, sensors, sample_to_row in train_loader:
            step_idx += 1
            labels = labels.to(device)
            sample_to_row = sample_to_row.to(device=device, dtype=torch.long)
            x_dict = recursive_to_device(x_dict, device)
            sensor_arg: Union[Sequence[str], torch.Tensor]
            if use_data_parallel:
                sensor_arg = core_model.encode_sensors(sensors, device=device)
            else:
                sensor_arg = sensors
            with autocast(enabled=use_amp):
                outputs = model(x_dict, sensors=sensor_arg)
                flat_labels = labels.index_select(0, sample_to_row)
                sensor_aux_loss, per_sensor_losses = core_model.loss_from_outputs(outputs, flat_labels, sensors, criterion)
                fused_logits = core_model.compute_row_fused_logits(
                    outputs,
                    sample_to_row=sample_to_row,
                    num_rows=labels.size(0),
                    device=device,
                )
                fused_loss = criterion(fused_logits, labels)
                loss = fused_loss + (args.sensor_aux_loss_weight * sensor_aux_loss)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(param_groups[0]["params"], args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            batch_rows = labels.size(0)
            total_loss += loss.item() * batch_rows
            total += batch_rows
            preds = fused_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            fused_loss_accum += fused_loss.item()
            fused_loss_count += 1
            sensor_aux_loss_accum += sensor_aux_loss.item()
            sensor_aux_loss_count += 1
            for sensor_name, sensor_loss in per_sensor_losses.items():
                if sensor_name in per_sensor_loss_accum:
                    per_sensor_loss_accum[sensor_name] += sensor_loss.item()
                    per_sensor_count[sensor_name] += 1

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break
            if args.log_interval and step_idx % args.log_interval == 0:
                sensor_loss_details = " ".join(
                    f"{name}_loss={per_sensor_losses[name].item():.4f}"
                    for name in sensors_list
                    if name in per_sensor_losses
                )
                sensor_loss_details = (
                    f"fused_loss={fused_loss.item():.4f} sensor_aux_loss={sensor_aux_loss.item():.4f} {sensor_loss_details}"
                ).strip()
                print(
                    f"Epoch {epoch} step {step_idx}/{train_steps} train_loss={total_loss/total:.4f} "
                    f"train_acc={correct/total:.4f} {sensor_loss_details}",
                    flush=True,
                )

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        test_loss_total = 0.0
        total_eval = 0
        correct_eval = 0
        eval_steps = 0
        fused_all_labels: list[int] = []
        fused_all_preds: list[int] = []
        fused_all_pos_scores: list[float] = []
        fused_overlap_labels: list[int] = []
        fused_overlap_preds: list[int] = []
        fused_overlap_pos_scores: list[float] = []
        fused_single_labels: list[int] = []
        fused_single_preds: list[int] = []
        fused_single_pos_scores: list[float] = []

        fused_all_sensor_labels: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        fused_all_sensor_preds: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        fused_all_sensor_pos_scores: Dict[str, list[float]] = {sensor: [] for sensor in sensors_list}
        fused_overlap_sensor_labels: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        fused_overlap_sensor_preds: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        fused_overlap_sensor_pos_scores: Dict[str, list[float]] = {sensor: [] for sensor in sensors_list}
        fused_single_sensor_labels: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        fused_single_sensor_preds: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        fused_single_sensor_pos_scores: Dict[str, list[float]] = {sensor: [] for sensor in sensors_list}

        ref_sensor_all_labels: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        ref_sensor_all_preds: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        ref_sensor_all_pos_scores: Dict[str, list[float]] = {sensor: [] for sensor in sensors_list}
        ref_sensor_overlap_labels: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        ref_sensor_overlap_preds: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        ref_sensor_overlap_pos_scores: Dict[str, list[float]] = {sensor: [] for sensor in sensors_list}
        ref_sensor_single_labels: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        ref_sensor_single_preds: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        ref_sensor_single_pos_scores: Dict[str, list[float]] = {sensor: [] for sensor in sensors_list}

        sensor_to_idx_eval = {sensor: idx for idx, sensor in enumerate(sensors_list)}
        with torch.no_grad():
            for step, (x_dict, labels, sensors, sample_to_row) in enumerate(test_loader, 1):
                labels = labels.to(device)
                sample_to_row = sample_to_row.to(device=device, dtype=torch.long)
                x_dict = recursive_to_device(x_dict, device)
                if use_data_parallel:
                    sensor_arg = core_model.encode_sensors(sensors, device=device)
                else:
                    sensor_arg = sensors
                with autocast(enabled=use_amp):
                    outputs = model(x_dict, sensors=sensor_arg)
                    flat_labels = labels.index_select(0, sample_to_row)
                    sensor_aux_loss, _ = core_model.loss_from_outputs(outputs, flat_labels, sensors, criterion)
                    fused_logits = core_model.compute_row_fused_logits(
                        outputs,
                        sample_to_row=sample_to_row,
                        num_rows=labels.size(0),
                        device=device,
                    )
                    fused_loss = criterion(fused_logits, labels)
                    loss = fused_loss + (args.sensor_aux_loss_weight * sensor_aux_loss)
                batch_rows = labels.size(0)
                test_loss_total += loss.item() * batch_rows
                total_eval += batch_rows
                preds = fused_logits.argmax(dim=1)
                correct_eval += (preds == labels).sum().item()
                pos_scores: Optional[torch.Tensor] = None
                if fused_logits.shape[-1] == 2:
                    pos_scores = torch.softmax(fused_logits, dim=1)[:, 1]
                row_has_sensor = torch.zeros((batch_rows, len(sensors_list)), dtype=torch.bool, device=device)
                for sample_idx, sensor_name in enumerate(sensors):
                    sensor_idx = sensor_to_idx_eval.get(str(sensor_name))
                    if sensor_idx is None:
                        continue
                    row_idx = int(sample_to_row[sample_idx].item())
                    row_has_sensor[row_idx, sensor_idx] = True
                row_sensor_count = row_has_sensor.sum(dim=1)
                overlap_mask = row_sensor_count >= 2
                single_mask = row_sensor_count == 1

                def _append_rows(mask: torch.Tensor, label_bucket: list[int], pred_bucket: list[int], score_bucket: list[float]) -> None:
                    if not torch.any(mask):
                        return
                    row_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    label_bucket.extend(labels.index_select(0, row_ids).detach().to("cpu").tolist())
                    pred_bucket.extend(preds.index_select(0, row_ids).detach().to("cpu").tolist())
                    if pos_scores is not None:
                        score_bucket.extend(pos_scores.index_select(0, row_ids).detach().to("cpu").tolist())

                _append_rows(torch.ones((batch_rows,), dtype=torch.bool, device=device), fused_all_labels, fused_all_preds, fused_all_pos_scores)
                _append_rows(overlap_mask, fused_overlap_labels, fused_overlap_preds, fused_overlap_pos_scores)
                _append_rows(single_mask, fused_single_labels, fused_single_preds, fused_single_pos_scores)

                for sensor_idx, sensor_name in enumerate(sensors_list):
                    sensor_mask = row_has_sensor[:, sensor_idx]
                    _append_rows(
                        sensor_mask,
                        fused_all_sensor_labels[sensor_name],
                        fused_all_sensor_preds[sensor_name],
                        fused_all_sensor_pos_scores[sensor_name],
                    )
                    _append_rows(
                        sensor_mask & overlap_mask,
                        fused_overlap_sensor_labels[sensor_name],
                        fused_overlap_sensor_preds[sensor_name],
                        fused_overlap_sensor_pos_scores[sensor_name],
                    )
                    _append_rows(
                        sensor_mask & single_mask,
                        fused_single_sensor_labels[sensor_name],
                        fused_single_sensor_preds[sensor_name],
                        fused_single_sensor_pos_scores[sensor_name],
                    )

                # Optional reference-only metrics from sensor-specific heads.
                for sensor_idx, sensor_name in enumerate(sensors_list):
                    sensor_out = outputs.get(sensor_name)
                    if not isinstance(sensor_out, dict):
                        continue
                    sensor_indices = sensor_out.get("indices")
                    sensor_logits = sensor_out.get("logits")
                    if not isinstance(sensor_indices, torch.Tensor) or not isinstance(sensor_logits, torch.Tensor):
                        continue
                    if sensor_indices.numel() == 0:
                        continue
                    sensor_rows = sample_to_row.index_select(0, sensor_indices)
                    row_ids, row_logits = mean_logits_by_row(sensor_logits, sensor_rows, num_rows=batch_rows)
                    if row_ids.numel() == 0:
                        continue
                    row_labels_ref = labels.index_select(0, row_ids)
                    row_preds_ref = row_logits.argmax(dim=1)
                    row_pos_scores: Optional[torch.Tensor] = None
                    if row_logits.shape[-1] == 2:
                        row_pos_scores = torch.softmax(row_logits, dim=1)[:, 1]
                    ref_sensor_all_labels[sensor_name].extend(row_labels_ref.detach().to("cpu").tolist())
                    ref_sensor_all_preds[sensor_name].extend(row_preds_ref.detach().to("cpu").tolist())
                    if row_pos_scores is not None:
                        ref_sensor_all_pos_scores[sensor_name].extend(row_pos_scores.detach().to("cpu").tolist())

                    row_overlap_mask_ref = overlap_mask.index_select(0, row_ids)
                    row_single_mask_ref = single_mask.index_select(0, row_ids)
                    if torch.any(row_overlap_mask_ref):
                        overlap_ids = torch.nonzero(row_overlap_mask_ref, as_tuple=False).squeeze(1)
                        ref_sensor_overlap_labels[sensor_name].extend(
                            row_labels_ref.index_select(0, overlap_ids).detach().to("cpu").tolist()
                        )
                        ref_sensor_overlap_preds[sensor_name].extend(
                            row_preds_ref.index_select(0, overlap_ids).detach().to("cpu").tolist()
                        )
                        if row_pos_scores is not None:
                            ref_sensor_overlap_pos_scores[sensor_name].extend(
                                row_pos_scores.index_select(0, overlap_ids).detach().to("cpu").tolist()
                            )
                    if torch.any(row_single_mask_ref):
                        single_ids = torch.nonzero(row_single_mask_ref, as_tuple=False).squeeze(1)
                        ref_sensor_single_labels[sensor_name].extend(
                            row_labels_ref.index_select(0, single_ids).detach().to("cpu").tolist()
                        )
                        ref_sensor_single_preds[sensor_name].extend(
                            row_preds_ref.index_select(0, single_ids).detach().to("cpu").tolist()
                        )
                        if row_pos_scores is not None:
                            ref_sensor_single_pos_scores[sensor_name].extend(
                                row_pos_scores.index_select(0, single_ids).detach().to("cpu").tolist()
                            )
                eval_steps += 1
                if args.max_eval_steps is not None and eval_steps >= args.max_eval_steps:
                    break

        test_loss = test_loss_total / max(1, total_eval)
        test_acc = correct_eval / max(1, total_eval)
        fused_all_overall_metrics = compute_split_metrics(fused_all_labels, fused_all_preds, fused_all_pos_scores)
        fused_overlap_overall_metrics = compute_split_metrics(
            fused_overlap_labels,
            fused_overlap_preds,
            fused_overlap_pos_scores,
        )
        fused_single_overall_metrics = compute_split_metrics(
            fused_single_labels,
            fused_single_preds,
            fused_single_pos_scores,
        )
        fused_all_sensor_metrics = {
            sensor: compute_split_metrics(
                fused_all_sensor_labels[sensor],
                fused_all_sensor_preds[sensor],
                fused_all_sensor_pos_scores[sensor],
            )
            for sensor in sensors_list
        }
        fused_overlap_sensor_metrics = {
            sensor: compute_split_metrics(
                fused_overlap_sensor_labels[sensor],
                fused_overlap_sensor_preds[sensor],
                fused_overlap_sensor_pos_scores[sensor],
            )
            for sensor in sensors_list
        }
        fused_single_sensor_metrics = {
            sensor: compute_split_metrics(
                fused_single_sensor_labels[sensor],
                fused_single_sensor_preds[sensor],
                fused_single_sensor_pos_scores[sensor],
            )
            for sensor in sensors_list
        }
        ref_sensor_all_metrics = {
            sensor: compute_split_metrics(
                ref_sensor_all_labels[sensor],
                ref_sensor_all_preds[sensor],
                ref_sensor_all_pos_scores[sensor],
            )
            for sensor in sensors_list
        }
        ref_sensor_overlap_metrics = {
            sensor: compute_split_metrics(
                ref_sensor_overlap_labels[sensor],
                ref_sensor_overlap_preds[sensor],
                ref_sensor_overlap_pos_scores[sensor],
            )
            for sensor in sensors_list
        }
        ref_sensor_single_metrics = {
            sensor: compute_split_metrics(
                ref_sensor_single_labels[sensor],
                ref_sensor_single_preds[sensor],
                ref_sensor_single_pos_scores[sensor],
            )
            for sensor in sensors_list
        }
        test_fpr = fused_all_overall_metrics["fpr"]
        test_recall = fused_all_overall_metrics["recall"]
        test_auroc = fused_all_overall_metrics["auroc"]
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"test_fpr={test_fpr:.4f} test_recall={test_recall:.4f} test_auroc={test_auroc:.4f} "
            f"train_fused_loss={fused_loss_accum/max(1,fused_loss_count):.4f} "
            f"train_sensor_aux_loss={sensor_aux_loss_accum/max(1,sensor_aux_loss_count):.4f}",
            flush=True,
        )
        print(
            f"[EvalSplit][FUSED][ALL][OVERALL] count={int(fused_all_overall_metrics['count'])} "
            f"acc={fused_all_overall_metrics['acc']:.4f} fpr={fused_all_overall_metrics['fpr']:.4f} "
            f"recall={fused_all_overall_metrics['recall']:.4f} auroc={fused_all_overall_metrics['auroc']:.4f}",
            flush=True,
        )
        print(
            f"[EvalSplit][FUSED][OVERLAP][OVERALL] count={int(fused_overlap_overall_metrics['count'])} "
            f"acc={fused_overlap_overall_metrics['acc']:.4f} fpr={fused_overlap_overall_metrics['fpr']:.4f} "
            f"recall={fused_overlap_overall_metrics['recall']:.4f} auroc={fused_overlap_overall_metrics['auroc']:.4f}",
            flush=True,
        )
        print(
            f"[EvalSplit][FUSED][SINGLE][OVERALL] count={int(fused_single_overall_metrics['count'])} "
            f"acc={fused_single_overall_metrics['acc']:.4f} fpr={fused_single_overall_metrics['fpr']:.4f} "
            f"recall={fused_single_overall_metrics['recall']:.4f} auroc={fused_single_overall_metrics['auroc']:.4f}",
            flush=True,
        )
        for sensor in sensors_list:
            m_all = fused_all_sensor_metrics[sensor]
            m_overlap = fused_overlap_sensor_metrics[sensor]
            m_single = fused_single_sensor_metrics[sensor]
            print(
                f"[EvalSplit][FUSED][ALL][SENSOR_{sensor.upper()}] count={int(m_all['count'])} "
                f"acc={m_all['acc']:.4f} fpr={m_all['fpr']:.4f} "
                f"recall={m_all['recall']:.4f} auroc={m_all['auroc']:.4f}",
                flush=True,
            )
            print(
                f"[EvalSplit][FUSED][OVERLAP][SENSOR_{sensor.upper()}] count={int(m_overlap['count'])} "
                f"acc={m_overlap['acc']:.4f} fpr={m_overlap['fpr']:.4f} "
                f"recall={m_overlap['recall']:.4f} auroc={m_overlap['auroc']:.4f}",
                flush=True,
            )
            print(
                f"[EvalSplit][FUSED][SINGLE][SENSOR_{sensor.upper()}] count={int(m_single['count'])} "
                f"acc={m_single['acc']:.4f} fpr={m_single['fpr']:.4f} "
                f"recall={m_single['recall']:.4f} auroc={m_single['auroc']:.4f}",
                flush=True,
            )
        for sensor in sensors_list:
            ref_all = ref_sensor_all_metrics[sensor]
            ref_overlap = ref_sensor_overlap_metrics[sensor]
            ref_single = ref_sensor_single_metrics[sensor]
            print(
                f"[EvalSplit][REF_SENSOR_HEAD][ALL][SENSOR_{sensor.upper()}] count={int(ref_all['count'])} "
                f"acc={ref_all['acc']:.4f} fpr={ref_all['fpr']:.4f} "
                f"recall={ref_all['recall']:.4f} auroc={ref_all['auroc']:.4f}",
                flush=True,
            )
            print(
                f"[EvalSplit][REF_SENSOR_HEAD][OVERLAP][SENSOR_{sensor.upper()}] count={int(ref_overlap['count'])} "
                f"acc={ref_overlap['acc']:.4f} fpr={ref_overlap['fpr']:.4f} "
                f"recall={ref_overlap['recall']:.4f} auroc={ref_overlap['auroc']:.4f}",
                flush=True,
            )
            print(
                f"[EvalSplit][REF_SENSOR_HEAD][SINGLE][SENSOR_{sensor.upper()}] count={int(ref_single['count'])} "
                f"acc={ref_single['acc']:.4f} fpr={ref_single['fpr']:.4f} "
                f"recall={ref_single['recall']:.4f} auroc={ref_single['auroc']:.4f}",
                flush=True,
            )

        prev_best_test = best_test_acc
        best_train_acc = max(best_train_acc, train_acc)
        best_test_acc = max(best_test_acc, test_acc)

        save_checkpoint(
            latest_path,
            epoch,
            global_step,
            core_model,
            optimizer,
            scheduler,
            scaler if use_amp else None,
            best_train_acc,
            best_test_acc,
            args,
        )
        if test_acc > prev_best_test:
            save_checkpoint(
                best_path,
                epoch,
                global_step,
                core_model,
                optimizer,
                scheduler,
                scaler if use_amp else None,
                best_train_acc,
                best_test_acc,
                args,
            )
            print(f"Saved new best checkpoint: {best_path} (test_acc={test_acc:.4f})", flush=True)
        if wandb_run is not None:
            log_payload = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_fpr": test_fpr,
                "test_recall": test_recall,
                "test_auroc": test_auroc,
                "test_fused_all_overall_count": fused_all_overall_metrics["count"],
                "test_fused_all_overall_acc": fused_all_overall_metrics["acc"],
                "test_fused_all_overall_fpr": fused_all_overall_metrics["fpr"],
                "test_fused_all_overall_recall": fused_all_overall_metrics["recall"],
                "test_fused_all_overall_auroc": fused_all_overall_metrics["auroc"],
                "test_fused_overlap_overall_count": fused_overlap_overall_metrics["count"],
                "test_fused_overlap_overall_acc": fused_overlap_overall_metrics["acc"],
                "test_fused_overlap_overall_fpr": fused_overlap_overall_metrics["fpr"],
                "test_fused_overlap_overall_recall": fused_overlap_overall_metrics["recall"],
                "test_fused_overlap_overall_auroc": fused_overlap_overall_metrics["auroc"],
                "test_fused_single_overall_count": fused_single_overall_metrics["count"],
                "test_fused_single_overall_acc": fused_single_overall_metrics["acc"],
                "test_fused_single_overall_fpr": fused_single_overall_metrics["fpr"],
                "test_fused_single_overall_recall": fused_single_overall_metrics["recall"],
                "test_fused_single_overall_auroc": fused_single_overall_metrics["auroc"],
                "train_fused_loss": fused_loss_accum / max(1, fused_loss_count),
                "train_sensor_aux_loss": sensor_aux_loss_accum / max(1, sensor_aux_loss_count),
            }
            for sensor in sensors_list:
                if per_sensor_count[sensor] > 0:
                    log_payload[f"train_loss_{sensor}"] = (
                        per_sensor_loss_accum[sensor] / per_sensor_count[sensor]
                    )
                fused_all_sensor = fused_all_sensor_metrics[sensor]
                fused_overlap_sensor = fused_overlap_sensor_metrics[sensor]
                fused_single_sensor = fused_single_sensor_metrics[sensor]
                ref_all_sensor = ref_sensor_all_metrics[sensor]
                ref_overlap_sensor = ref_sensor_overlap_metrics[sensor]
                ref_single_sensor = ref_sensor_single_metrics[sensor]
                log_payload[f"test_fused_all_sensor_{sensor}_count"] = fused_all_sensor["count"]
                log_payload[f"test_fused_all_sensor_{sensor}_acc"] = fused_all_sensor["acc"]
                log_payload[f"test_fused_all_sensor_{sensor}_fpr"] = fused_all_sensor["fpr"]
                log_payload[f"test_fused_all_sensor_{sensor}_recall"] = fused_all_sensor["recall"]
                log_payload[f"test_fused_all_sensor_{sensor}_auroc"] = fused_all_sensor["auroc"]
                log_payload[f"test_fused_overlap_sensor_{sensor}_count"] = fused_overlap_sensor["count"]
                log_payload[f"test_fused_overlap_sensor_{sensor}_acc"] = fused_overlap_sensor["acc"]
                log_payload[f"test_fused_overlap_sensor_{sensor}_fpr"] = fused_overlap_sensor["fpr"]
                log_payload[f"test_fused_overlap_sensor_{sensor}_recall"] = fused_overlap_sensor["recall"]
                log_payload[f"test_fused_overlap_sensor_{sensor}_auroc"] = fused_overlap_sensor["auroc"]
                log_payload[f"test_fused_single_sensor_{sensor}_count"] = fused_single_sensor["count"]
                log_payload[f"test_fused_single_sensor_{sensor}_acc"] = fused_single_sensor["acc"]
                log_payload[f"test_fused_single_sensor_{sensor}_fpr"] = fused_single_sensor["fpr"]
                log_payload[f"test_fused_single_sensor_{sensor}_recall"] = fused_single_sensor["recall"]
                log_payload[f"test_fused_single_sensor_{sensor}_auroc"] = fused_single_sensor["auroc"]
                log_payload[f"test_ref_sensor_head_all_{sensor}_count"] = ref_all_sensor["count"]
                log_payload[f"test_ref_sensor_head_all_{sensor}_acc"] = ref_all_sensor["acc"]
                log_payload[f"test_ref_sensor_head_all_{sensor}_fpr"] = ref_all_sensor["fpr"]
                log_payload[f"test_ref_sensor_head_all_{sensor}_recall"] = ref_all_sensor["recall"]
                log_payload[f"test_ref_sensor_head_all_{sensor}_auroc"] = ref_all_sensor["auroc"]
                log_payload[f"test_ref_sensor_head_overlap_{sensor}_count"] = ref_overlap_sensor["count"]
                log_payload[f"test_ref_sensor_head_overlap_{sensor}_acc"] = ref_overlap_sensor["acc"]
                log_payload[f"test_ref_sensor_head_overlap_{sensor}_fpr"] = ref_overlap_sensor["fpr"]
                log_payload[f"test_ref_sensor_head_overlap_{sensor}_recall"] = ref_overlap_sensor["recall"]
                log_payload[f"test_ref_sensor_head_overlap_{sensor}_auroc"] = ref_overlap_sensor["auroc"]
                log_payload[f"test_ref_sensor_head_single_{sensor}_count"] = ref_single_sensor["count"]
                log_payload[f"test_ref_sensor_head_single_{sensor}_acc"] = ref_single_sensor["acc"]
                log_payload[f"test_ref_sensor_head_single_{sensor}_fpr"] = ref_single_sensor["fpr"]
                log_payload[f"test_ref_sensor_head_single_{sensor}_recall"] = ref_single_sensor["recall"]
                log_payload[f"test_ref_sensor_head_single_{sensor}_auroc"] = ref_single_sensor["auroc"]
            wandb_run.log(log_payload)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
