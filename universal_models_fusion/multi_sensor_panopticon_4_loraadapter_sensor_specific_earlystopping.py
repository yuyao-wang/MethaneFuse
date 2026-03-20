"""Multi-sensor Panopticon training script with sensor-specific embeddings and heads.

This module exposes reusable model components (``MultiSensorPanopticonClassifier``)
*and* a runnable training entry point that ingests mixed-sensor CSVs containing
Sentinel-2, Landsat 8/9, Sentinel-5P, and WV3 samples. Each sensor owns its own
Panopticon patch embedding and classifier head, while the DinoViT backbone is
shared and updated by the consensus of all datasets in a batch.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import warnings
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile as tiff
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset, SubsetRandomSampler

# Make the repository root importable so examples work when executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid CUDA init stalls on CPU-only hosts.
os.environ.setdefault("XFORMERS_DISABLED", "1")

from dinov2.data.datasets.s2_csv import S2TemporalCsvDataset, _SkipSample
from dinov2.utils.data import extract_wavemus, load_ds_cfg
from dinov2.models.panopticon import PanopticonPE
from dinov2.models.vision_transformer import DinoVisionTransformer

S2_PRECOMPUTED_STATS = (
    [
        786.128173828125,
        1025.8876953125,
        1593.730712890625,
        2315.26123046875,
        2710.462890625,
        3115.90087890625,
        3289.0830078125,
        3465.536376953125,
        3495.579833984375,
        3517.7958984375,
        4180.28564453125,
        3567.866943359375,
    ],
    [
        435.72607421875,
        597.6113891601562,
        688.5059814453125,
        840.1614990234375,
        801.7208251953125,
        706.9466552734375,
        689.823974609375,
        727.5567626953125,
        668.30224609375,
        551.3565063476562,
        629.679931640625,
        641.590087890625,
    ],
)

L89_PRECOMPUTED_STATS = (
    [
        10729.92784546,
        11384.64407242,
        13172.77519667,
        14892.25620267,
        18149.92169893,
        20249.17615773,
        18375.0669698,
    ],
    [
        1029.18232283,
        1188.52313418,
        1552.27685613,
        1959.74400972,
        1954.80410093,
        2098.98682671,
        1895.56781996,
    ],
)

# Sentinel-5P CH4 stacked (t0, t-90, t-360) stats reused from existing examples.
S5P_PRECOMPUTED_STATS = (
    [1908.944798144908, 362.20128748570943, 666.5645739544017],
    [52.2615669211965, 743.7553740768346, 901.8399543163595],
)

DEFAULT_WV3_BANDS = [
    "Coastal (MS7)",
    "Blue (MS4)",
    "Green (MS3)",
    "Yellow (MS6)",
    "Red (MS2)",
    "Red Edge (MS5)",
    "NIR1 (MS1)",
    "NIR2 (MS8)",
    "SWIR1",
    "SWIR2",
    "SWIR3",
    "SWIR4",
    "SWIR5",
    "SWIR6",
    "SWIR7",
    "SWIR8",
]


def _parse_optional_float(text: str) -> Optional[float]:
    text = str(text).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_wv3_channel_ids_from_srf(csv_path: str, band_names: Sequence[str]) -> torch.Tensor:
    """Compute WV3 channel IDs (mu) from an SRF CSV file."""

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"WV3 SRF file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"WV3 SRF file is empty: {csv_path}") from exc

        header_idx = {name: idx for idx, name in enumerate(header) if name}
        missing = [name for name in band_names if name not in header_idx]
        if missing:
            raise ValueError(
                f"Missing WV3 band columns in {csv_path}: {missing}. Available columns include: {header[:18]}"
            )

        wavelengths = []
        responses = {name: [] for name in band_names}
        for row in reader:
            if not row:
                continue
            wavelength = _parse_optional_float(row[0] if len(row) > 0 else "")
            if wavelength is None:
                continue
            wavelengths.append(wavelength)
            for name in band_names:
                idx = header_idx[name]
                value = _parse_optional_float(row[idx] if idx < len(row) else "")
                responses[name].append(max(0.0, value if value is not None else 0.0))

    if len(wavelengths) == 0:
        raise ValueError(f"No numeric wavelength rows parsed from {csv_path}")

    wavelength_t = torch.tensor(wavelengths, dtype=torch.float64)
    mu_list = []
    for name in band_names:
        weights = torch.tensor(responses[name], dtype=torch.float64)
        total = torch.sum(weights)
        if total <= 0:
            raise ValueError(f"Band '{name}' has zero response everywhere in {csv_path}")
        mu = torch.sum(wavelength_t * weights) / total
        mu_list.append(mu)
    return torch.round(torch.stack(mu_list)).to(torch.int16)


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


HeadFactory = Callable[[int, int], nn.Module]


class TinyResidualAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int,
        *,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError("adapter_bottleneck_dim (LoRA rank) must be > 0")
        self.rank = int(bottleneck_dim)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.down = nn.Linear(embed_dim, self.rank, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(self.rank, embed_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.up(self.dropout(self.down(x))) * self.scaling


class SensorAdapterBlock(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        embed_dim: int,
        num_domains: int,
        bottleneck_dim: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        cls_only: bool = True,
    ):
        super().__init__()
        self.block = block
        self.adapters = nn.ModuleList(
            [
                TinyResidualAdapter(
                    embed_dim=embed_dim,
                    bottleneck_dim=bottleneck_dim,
                    alpha=alpha,
                    dropout=dropout,
                )
                for _ in range(num_domains)
            ]
        )
        self.cls_only = bool(cls_only)
        self._current_domain: Optional[torch.Tensor] = None

    def set_domain_id(self, domain_id: Union[int, torch.Tensor]) -> None:
        if not torch.is_tensor(domain_id):
            domain_id = torch.tensor(domain_id, dtype=torch.long)
        self._current_domain = domain_id

    def _resolve_domain(self, x: torch.Tensor) -> torch.Tensor:
        if self._current_domain is None:
            return torch.zeros((), device=x.device, dtype=torch.long)
        domain = self._current_domain
        if domain.device != x.device:
            domain = domain.to(device=x.device, dtype=torch.long)
        return domain

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.block(x)
        domain = self._resolve_domain(x)
        if domain.ndim == 0:
            adapter = self.adapters[int(domain.item())]
            if self.cls_only:
                out = x.clone()
                cls = out[:, :1, :]
                out[:, :1, :] = cls + adapter(cls)
                return out
            return x + adapter(x)

        if domain.shape[0] != x.shape[0]:
            raise ValueError(f"domain_id batch mismatch: expected {x.shape[0]}, got {domain.shape[0]}")
        out = x.clone()
        for domain_idx, adapter in enumerate(self.adapters):
            selector = domain == domain_idx
            if selector.any():
                x_sel = x[selector]
                if self.cls_only:
                    out_sel = x_sel.clone()
                    cls = out_sel[:, :1, :]
                    out_sel[:, :1, :] = cls + adapter(cls)
                    out[selector] = out_sel
                else:
                    out[selector] = x_sel + adapter(x_sel)
        return out


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
        adapter_first_blocks: int = 5,
        adapter_bottleneck_dim: int = 16,
        adapter_alpha: float = 16.0,
        adapter_dropout: float = 0.0,
        adapter_cls_only: bool = True,
        enable_summary_head: bool = True,
        summary_hidden_dim: int = 128,
        summary_dropout: float = 0.1,
        summary_loss_weight: float = 1.0,
        enable_overlap_head: bool = True,
        overlap_hidden_dim: int = 128,
        overlap_dropout: float = 0.1,
        overlap_loss_weight: float = 0.0,
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
        _attach_sensor_adapters(
            self.backbone,
            num_domains=len(self.sensor_order),
            adapter_first_blocks=adapter_first_blocks,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            adapter_alpha=adapter_alpha,
            adapter_dropout=adapter_dropout,
            adapter_cls_only=adapter_cls_only,
        )

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
        self.overlap_loss_weight = float(overlap_loss_weight)
        self.summary_head: Optional[LogitSummaryHead] = None
        self.overlap_head: Optional[LogitSummaryHead] = None

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
        if enable_overlap_head and can_build_summary:
            num_out_classes = int(head_dims[0])
            self.overlap_head = LogitSummaryHead(
                num_heads=len(self.sensor_order),
                num_classes=num_out_classes,
                hidden_dim=overlap_hidden_dim,
                dropout=overlap_dropout,
            )
        elif enable_overlap_head and not can_build_summary:
            warnings.warn(
                "Overlap head disabled because sensor heads do not share the same out_features.",
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

    def _set_backbone_domain_id(self, domain_id: torch.Tensor) -> None:
        # Resolve from registered modules each call so DataParallel replicas stay isolated.
        for _, _, block in _iter_transformer_blocks(self.backbone):
            if isinstance(block, SensorAdapterBlock):
                block.set_domain_id(domain_id)

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
                domain_ids = torch.full(
                    (sensor_batch.indices.shape[0],),
                    self.sensor_to_idx[sensor_name],
                    device=sensor_batch.indices.device,
                    dtype=torch.long,
                )
                self._set_backbone_domain_id(domain_ids)
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


@dataclass
class SensorBatch:
    indices: torch.Tensor
    x_dict: Dict[str, torch.Tensor]


# --------------------------------------------------------------------------------------
#  Backbone patching helpers (sensor-specific low-rank residual adapters)
# --------------------------------------------------------------------------------------

def _iter_transformer_blocks(backbone: DinoVisionTransformer) -> list[tuple[nn.ModuleList, int, nn.Module]]:
    blocks: list[tuple[nn.ModuleList, int, nn.Module]] = []
    for top_idx, block_chunk in enumerate(backbone.blocks):
        if isinstance(block_chunk, nn.ModuleList):
            for idx, block in enumerate(block_chunk):
                if isinstance(block, nn.Identity):
                    continue
                blocks.append((block_chunk, idx, block))
        else:
            blocks.append((backbone.blocks, top_idx, block_chunk))
    return blocks


def _attach_sensor_adapters(
    backbone: DinoVisionTransformer,
    *,
    num_domains: int,
    adapter_first_blocks: int,
    adapter_bottleneck_dim: int,
    adapter_alpha: float = 16.0,
    adapter_dropout: float = 0.0,
    adapter_cls_only: bool = True,
):
    if adapter_first_blocks <= 0 or num_domains <= 0:
        return backbone

    block_entries = _iter_transformer_blocks(backbone)
    if not block_entries:
        raise RuntimeError("No transformer blocks found when attaching adapters.")
    adapter_first_blocks = min(adapter_first_blocks, len(block_entries))
    embed_dim = int(getattr(backbone, "embed_dim", 768))

    wrapped_blocks: list[SensorAdapterBlock] = []
    for container, idx, base_block in block_entries[:adapter_first_blocks]:
        if isinstance(base_block, SensorAdapterBlock):
            wrapped = base_block
        else:
            wrapped = SensorAdapterBlock(
                block=base_block,
                embed_dim=embed_dim,
                num_domains=num_domains,
                bottleneck_dim=adapter_bottleneck_dim,
                alpha=adapter_alpha,
                dropout=adapter_dropout,
                cls_only=adapter_cls_only,
            )
            container[idx] = wrapped
        wrapped_blocks.append(wrapped)

    backbone._sensor_adapter_blocks = wrapped_blocks  # type: ignore[attr-defined]

    def _set_domain_id(self, domain_id):
        blocks = getattr(self, "_sensor_adapter_blocks", [])
        if not blocks:
            return
        if not torch.is_tensor(domain_id):
            domain_id_tensor = torch.tensor(domain_id, device=self.pos_embed.device, dtype=torch.long)
        else:
            domain_id_tensor = domain_id.to(self.pos_embed.device)
        for adapter_block in blocks:
            adapter_block.set_domain_id(domain_id_tensor)

    backbone._set_domain_id = MethodType(_set_domain_id, backbone)  # type: ignore[attr-defined]
    return backbone


# --------------------------------------------------------------------------------------
#  Dataset helpers (mixed temporal CSV + S5P NPZ support)
# --------------------------------------------------------------------------------------

class StaticAnchoredCache:
    def __init__(self, cache_dir: str, min_free_gb: float = 10.0):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_free_bytes = min_free_gb * (1024**3)

    def _get_free_space(self) -> int:
        return shutil.disk_usage(self.cache_dir).free

    def _hashed_path(self, original: str) -> Path:
        norm_path = os.path.abspath(original)
        digest = hashlib.sha1(norm_path.encode("utf-8")).hexdigest()
        subdir = digest[:2]
        suffix = Path(original).suffix
        return self.cache_dir / subdir / f"{digest}{suffix}"

    def ensure_local(self, original: str) -> str:
        dst = self._hashed_path(original)
        if dst.exists():
            return str(dst)
        if self._get_free_space() < self.min_free_bytes:
            return original
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(original, tmp)
            os.replace(tmp, dst)
        except Exception:
            with suppress(FileNotFoundError):
                tmp.unlink()
            return original
        return str(dst)

    def warm_up(self, paths: Sequence[str], max_workers: int = 8) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        unique_paths = sorted({os.path.abspath(p) for p in paths if isinstance(p, str)})
        if not unique_paths:
            return
        print(f"[Cache] Warming up (target: {len(unique_paths)})...", flush=True)

        def _copy_one(path: str):
            res = self.ensure_local(path)
            return res == path

        fallback_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_copy_one, p) for p in unique_paths]
            for fut in as_completed(futures):
                fallback_count += 1 if fut.result() else 0
        print(f"[Cache] Warmup complete. Cached: {len(unique_paths) - fallback_count}, Remote: {fallback_count}")


def _compute_mean_std(stats: Optional[Tuple[Sequence[float], Sequence[float]]]):
    if stats is None:
        return None, None
    mean, std = stats
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_tensor = torch.clamp(torch.tensor(std, dtype=torch.float32), min=1e-6).view(-1, 1, 1)
    return mean_tensor, std_tensor


class TriSensorTemporalCsvDataset(S2TemporalCsvDataset):
    """Temporal dataset that mixes Sentinel-2, Landsat 8/9, Sentinel-5P, and WV3 samples."""

    def __init__(
        self,
        *args,
        local_file_cache: Optional[StaticAnchoredCache] = None,
        s5p_data_key: Optional[str] = None,
        s5p_chn_ids_key: Optional[str] = "chn_ids",
        s5p_channels_last: bool = False,
        align_l89_to_s2: bool = False,
        wv3_chn_ids: Optional[torch.Tensor] = None,
        fusion_group_column: str = "id",
        sensor_stats_overrides: Optional[Mapping[str, Tuple[Sequence[float], Sequence[float]]]] = None,
        **kwargs,
    ):
        kwargs.pop("ds_cfg_name", None)
        kwargs.pop("normalize_stats", None)
        self.ds_cfg = None
        self.normalize_stats = None
        self._local_file_cache = local_file_cache
        self._s5p_data_key = s5p_data_key
        self._s5p_chn_ids_key = s5p_chn_ids_key
        self._s5p_channels_last = s5p_channels_last
        self._align_l89_to_s2 = bool(align_l89_to_s2)
        self._fusion_group_column = str(fusion_group_column).strip() or "id"
        self._sensor_stats_overrides = dict(sensor_stats_overrides) if sensor_stats_overrides is not None else {}
        self._table_mode = "long"
        self._wide_sensor_columns: Dict[str, Tuple[str, ...]] = {}
        self._wv3_chn_ids = None
        if wv3_chn_ids is not None:
            self._wv3_chn_ids = torch.as_tensor(wv3_chn_ids).clone().detach()
            if self._wv3_chn_ids.ndim == 1:
                self._wv3_chn_ids = self._wv3_chn_ids.unsqueeze(-1)
        super().__init__(*args, **kwargs)

        self._validate_sensor_column()
        if self._fusion_group_column not in self.df.columns:
            warnings.warn(
                f"Fusion group column '{self._fusion_group_column}' not found; falling back to per-row group ids.",
                stacklevel=2,
            )
        self.sensor_configs = self._build_sensor_configs()
        self._s5p_mean, self._s5p_std = _compute_mean_std(S5P_PRECOMPUTED_STATS)
        if "s5p" in self._sensor_stats_overrides:
            s5p_mean, s5p_std = _compute_mean_std(self._sensor_stats_overrides["s5p"])
            if s5p_mean is not None and s5p_std is not None:
                self._s5p_mean, self._s5p_std = s5p_mean, s5p_std

    def _validate_sensor_column(self) -> None:
        if "sensor" in self.df.columns:
            self._table_mode = "long"
            return
        self._table_mode = "wide"

        def _wide_cols(prefix: str) -> Optional[Tuple[str, str, str]]:
            cols = (f"{prefix}_0_path", f"{prefix}_90_path", f"{prefix}_360_path")
            return cols if all(col in self.df.columns for col in cols) else None

        s2_cols = _wide_cols("s2")
        l89_cols = _wide_cols("l89")
        s5p_col = ("s5p_0_path",) if "s5p_0_path" in self.df.columns else None
        wv3_cols = _wide_cols("wv3")
        if wv3_cols is None:
            emit_cols = _wide_cols("emit")
            if emit_cols is not None:
                warnings.warn(
                    "Wide-table uses EMIT columns; mapping emit_*_path to sensor key 'wv3' for this script.",
                    stacklevel=2,
                )
                wv3_cols = emit_cols

        if s2_cols is not None:
            self._wide_sensor_columns["s2"] = s2_cols
        if l89_cols is not None:
            self._wide_sensor_columns["l89"] = l89_cols
        if s5p_col is not None:
            self._wide_sensor_columns["s5p"] = s5p_col
        if wv3_cols is not None:
            self._wide_sensor_columns["wv3"] = wv3_cols

        if len(self._wide_sensor_columns) == 0:
            raise ValueError(
                "CSV has no 'sensor' column and no recognized wide-table sensor path columns "
                "(expected prefixes like s2_*, l89_*, s5p_0_path, wv3_* or emit_*)."
            )

    def _build_sensor_configs(self) -> Dict[str, Dict[str, torch.Tensor]]:
        l89_stats = self._sensor_stats_overrides.get("l89", L89_PRECOMPUTED_STATS)
        s2_stats = self._sensor_stats_overrides.get("s2", S2_PRECOMPUTED_STATS)
        configs = {
            "l89": {
                "ds_cfg": load_ds_cfg("landsat89_7band"),
                "normalize_stats": l89_stats,
                "scale_to_unit": True,
            },
            "s2": {
                "ds_cfg": load_ds_cfg("s2_12band"),
                "normalize_stats": s2_stats,
                "scale_to_unit": True,
            },
        }

        if self._wv3_chn_ids is not None:
            configs["wv3"] = {
                "ds_cfg": None,
                "normalize_stats": self._sensor_stats_overrides.get("wv3"),
                "chn_ids": self._wv3_chn_ids,
                "mean_tensor": None,
                "std_tensor": None,
                "scale_to_unit": False,
            }

        # Inject channel ids and optional scaling.
        for name, cfg in configs.items():
            if name == "wv3":
                continue
            cfg["normalize_stats"] = self._maybe_scale_stats(cfg.get("normalize_stats"))
            ds_cfg_obj = cfg["ds_cfg"]
            chn_ids = extract_wavemus(ds_cfg_obj, return_sigmas=False).unsqueeze(-1)
            ds_cfg_obj["chn_ids"] = chn_ids
            cfg["chn_ids"] = chn_ids

        if self._align_l89_to_s2:
            configs["l89"]["chn_ids"] = configs["s2"]["chn_ids"]

        for cfg in configs.values():
            mean_tensor, std_tensor = _compute_mean_std(cfg.get("normalize_stats"))
            cfg["mean_tensor"] = mean_tensor
            cfg["std_tensor"] = std_tensor

        return configs

    def _maybe_scale_stats(self, stats):
        if stats is None or not getattr(self, "scale_to_unit", False):
            return stats
        mean, std = stats
        return ([m / 65535.0 for m in mean], [s / 65535.0 for s in std])

    @contextmanager
    def _use_sensor_cfg(self, sensor: str):
        cfg = self.sensor_configs[sensor]
        original = (self.ds_cfg, self.normalize_stats, self.chn_ids, self._mean, self._std, self.scale_to_unit)
        self.ds_cfg = cfg["ds_cfg"]
        self.normalize_stats = cfg["normalize_stats"]
        self.chn_ids = cfg["chn_ids"]
        self._mean = cfg.get("mean_tensor")
        self._std = cfg.get("std_tensor")
        self.scale_to_unit = bool(cfg.get("scale_to_unit", self.scale_to_unit))
        try:
            yield
        finally:
            self.ds_cfg, self.normalize_stats, self.chn_ids, self._mean, self._std, self.scale_to_unit = original

    def _load_image(self, path: str, *, column_name: str, sample_id: int, sensor_override: Optional[str] = None):
        row = self.df.iloc[sample_id]
        sensor = sensor_override if sensor_override is not None else row.get("sensor")
        if sensor not in self.sensor_configs:
            raise ValueError(f"Sample {sample_id} has unknown sensor '{sensor}'")
        with self._use_sensor_cfg(sensor):
            path_to_use = self._maybe_cache_path(path)
            return super(TriSensorTemporalCsvDataset, self)._load_image(
                path_to_use, column_name=column_name, sample_id=sample_id
            )

    def __getitem__(self, idx):
        attempts = 0
        last_exc: Optional[Exception] = None
        while attempts < self.max_retries:
            try:
                row = self.df.iloc[idx]
                label = int(row[self.label_column])
                group_value = row.get(self._fusion_group_column, idx)
                if group_value is None or (isinstance(group_value, float) and np.isnan(group_value)):
                    group_value = idx
                group_id = str(group_value)

                if self._table_mode == "long":
                    sensor = row.get("sensor")
                    if sensor is None:
                        raise ValueError(f"Row {idx} is missing 'sensor' value")

                    if sensor == "s5p":
                        x_dict = self._load_s5p_sample(row, self.path_columns[0])
                        return [("s5p", [x_dict])], label, group_id

                    if sensor not in self.sensor_configs:
                        raise ValueError(f"Sample {idx} has unknown sensor '{sensor}'")

                    x_list = [self._load_temporal_frame(row, col, sensor, idx) for col in self.path_columns]
                    return [(str(sensor), x_list)], label, group_id

                sensor_samples: list[tuple[str, list[Dict[str, torch.Tensor]]]] = []
                for sensor_name in ("s2", "l89", "s5p", "wv3"):
                    cols = self._wide_sensor_columns.get(sensor_name)
                    if cols is None:
                        continue
                    if sensor_name == "s5p":
                        path_text = str(row.get(cols[0], "")).strip()
                        if path_text == "" or path_text.lower() in ("nan", "none", "null"):
                            continue
                        x_dict = self._load_s5p_sample(row, cols[0])
                        sensor_samples.append((sensor_name, [x_dict]))
                        continue

                    missing_path = False
                    for col in cols:
                        value = str(row.get(col, "")).strip()
                        if value == "" or value.lower() in ("nan", "none", "null"):
                            missing_path = True
                            break
                    if missing_path:
                        continue
                    x_list = [self._load_temporal_frame(row, col, sensor_name, idx) for col in cols]
                    sensor_samples.append((sensor_name, x_list))

                if not sensor_samples:
                    raise _SkipSample(f"No valid sensor paths found for wide-table row {idx}")
                return sensor_samples, label, group_id
            except _SkipSample as exc:
                last_exc = exc
                attempts += 1
                if attempts >= self.max_retries:
                    raise RuntimeError(f"Exceeded {self.max_retries} retries for temporal index {idx}") from exc
                idx = np.random.randint(0, len(self.df))
        raise RuntimeError("Unreachable") from last_exc

    def _load_temporal_frame(self, row, column_name: str, sensor: str, sample_id: int) -> Dict[str, torch.Tensor]:
        path = row[column_name]
        img = self._load_image(path, column_name=column_name, sample_id=sample_id, sensor_override=sensor)
        if sensor == "l89" and self._align_l89_to_s2:
            img = self._pad_l89_to_s2(img)

        # WV3 values can come in different scales across sources.
        # Auto-scale only when dynamic range is clearly raw-DN-like.
        if sensor == "wv3":
            abs_max = torch.amax(torch.abs(img))
            if torch.isfinite(abs_max) and abs_max.item() > 100.0:
                img = img / 65535.0

        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = torch.clamp(img, min=-50.0, max=50.0)
        x_dict = {
            "imgs": img,
            "chn_ids": self.sensor_configs[sensor]["chn_ids"],
        }
        if self.transform_each is not None:
            x_dict = self.transform_each(x_dict)
        return x_dict

    def _maybe_cache_path(self, path: str) -> str:
        if self._local_file_cache is not None and isinstance(path, str):
            return self._local_file_cache.ensure_local(path)
        return path

    def _load_s5p_sample(self, row, column_name: str) -> Dict[str, torch.Tensor]:
        img, chn_ids = self._load_s5p_raw_sample(row, column_name)
        img = (img - self._s5p_mean) / self._s5p_std
        if self.pad_to_multiple is not None:
            img = self._pad_to_multiple(img, self.pad_to_multiple)
        if chn_ids is None:
            chn_ids = torch.zeros((img.shape[0], 1), dtype=torch.float32)
        if chn_ids.ndim == 1:
            chn_ids = chn_ids.unsqueeze(-1)
        if chn_ids.shape[0] != img.shape[0]:
            if chn_ids.shape[0] == 1:
                chn_ids = chn_ids.repeat(img.shape[0], *([1] * (chn_ids.ndim - 1)))
            elif chn_ids.shape[0] > img.shape[0]:
                chn_ids = chn_ids[: img.shape[0]]
            else:
                repeat = math.ceil(img.shape[0] / max(1, chn_ids.shape[0]))
                chn_ids = chn_ids.repeat(repeat, *([1] * (chn_ids.ndim - 1)))[: img.shape[0]]
        x_dict = dict(imgs=img, chn_ids=chn_ids)
        if self.transform_each is not None:
            x_dict = self.transform_each(x_dict)
        return x_dict

    def _load_s5p_raw_sample(self, row, column_name: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        path = row.get(column_name)
        if not isinstance(path, str) or not path:
            raise ValueError(f"S5P sample missing value in '{column_name}'")
        path = self._maybe_cache_path(path)
        chn_ids = None
        lower_path = path.lower()
        if lower_path.endswith((".tif", ".tiff")):
            img_np = np.array(tiff.imread(path))
        else:
            try:
                np_obj = np.load(path, allow_pickle=False)
            except ValueError as exc:
                if "allow_pickle=False" in str(exc) or "pickled data" in str(exc):
                    np_obj = np.load(path, allow_pickle=True)
                else:
                    raise
            try:
                if isinstance(np_obj, np.lib.npyio.NpzFile):
                    img_np = self._extract_npz_array(np_obj, path)
                    if self._s5p_chn_ids_key is not None and self._s5p_chn_ids_key in np_obj:
                        chn_ids = torch.as_tensor(np_obj[self._s5p_chn_ids_key])
                else:
                    if isinstance(np_obj, np.ndarray) and np_obj.dtype == object and np_obj.size == 1:
                        obj = np_obj.item()
                        if isinstance(obj, Mapping):
                            if self._s5p_data_key is not None and self._s5p_data_key in obj:
                                img_np = np.asarray(obj[self._s5p_data_key])
                            else:
                                first_array = next(
                                    (v for v in obj.values() if isinstance(v, np.ndarray)),
                                    None,
                                )
                                if first_array is None:
                                    raise ValueError(f"Object array at {path} has no ndarray payload.")
                                img_np = np.asarray(first_array)
                            if self._s5p_chn_ids_key is not None and self._s5p_chn_ids_key in obj:
                                chn_ids = torch.as_tensor(obj[self._s5p_chn_ids_key])
                        else:
                            img_np = np.asarray(obj)
                    else:
                        img_np = np.array(np_obj)
            finally:
                if isinstance(np_obj, np.lib.npyio.NpzFile):
                    np_obj.close()
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, 0)
        elif img_np.ndim == 3 and self._s5p_channels_last:
            img_np = np.transpose(img_np, (2, 0, 1))
        elif img_np.ndim == 3:
            c_first, c_last = img_np.shape[0], img_np.shape[-1]
            if c_first > 32 and c_last <= 16:
                img_np = np.transpose(img_np, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported S5P sample shape {img_np.shape} from {path}")
        img = torch.from_numpy(img_np).to(dtype=torch.float32)
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        expected_channels = int(self._s5p_mean.shape[0])
        if img.shape[0] != expected_channels:
            if img.shape[0] == 1 and expected_channels > 1:
                img = img.repeat(expected_channels, 1, 1)
            elif img.shape[0] > expected_channels:
                img = img[:expected_channels]
            else:
                repeat = math.ceil(expected_channels / max(1, img.shape[0]))
                img = img.repeat(repeat, 1, 1)[:expected_channels]
        return img, chn_ids

    def _extract_npz_array(self, np_obj: np.lib.npyio.NpzFile, path: str) -> np.ndarray:
        if self._s5p_data_key is not None:
            if self._s5p_data_key not in np_obj:
                raise KeyError(f"Key '{self._s5p_data_key}' not found in NPZ file {path}")
            return np.array(np_obj[self._s5p_data_key])
        if len(np_obj.files) == 0:
            raise ValueError(f"No arrays found in NPZ file {path}")
        return np.array(np_obj[np_obj.files[0]])

    @staticmethod
    def _pad_l89_to_s2(img: torch.Tensor) -> torch.Tensor:
        if img.shape[0] != 7:
            return img
        device, dtype, h, w = img.device, img.dtype, img.shape[1], img.shape[2]
        out = torch.zeros((12, h, w), device=device, dtype=dtype)
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 7, 5: 10, 6: 11}
        for l89_idx, s2_idx in mapping.items():
            out[s2_idx] = img[l89_idx]
        return out


class ConcatTemporalDataset(Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        if (
            len(sample) >= 3
            and isinstance(sample[0], list)
            and (len(sample[0]) == 0 or isinstance(sample[0][0], tuple))
        ):
            sensor_samples, label, group_id = sample[:3]
        elif len(sample) == 4:
            x_list, label, sensor, group_id = sample
            sensor_samples = [(str(sensor), x_list)]
        else:
            x_list, label, sensor = sample
            sensor_samples = [(str(sensor), x_list)]
            group_id = str(idx)

        flattened = []
        for sensor, x_list in sensor_samples:
            imgs = torch.cat([x["imgs"] for x in x_list], dim=0)
            chn_ids = torch.cat([x["chn_ids"] for x in x_list], dim=0)
            x_dict = dict(imgs=imgs, chn_ids=chn_ids)
            flattened.append((x_dict, label, str(sensor), str(group_id)))
        return flattened


def custom_collate_fn(batch):
    flat_batch = []
    for item in batch:
        if isinstance(item, list):
            flat_batch.extend(item)
        else:
            flat_batch.append(item)
    if len(flat_batch) == 0:
        raise RuntimeError("Empty batch encountered in custom_collate_fn.")

    if len(flat_batch[0]) == 4:
        x_dicts, labels, sensors, group_ids = zip(*flat_batch)
    else:
        x_dicts, labels, sensors = zip(*flat_batch)
        group_ids = [str(i) for i in range(len(flat_batch))]
    max_channels = max(x["imgs"].shape[0] for x in x_dicts)
    max_h = max(x["imgs"].shape[1] for x in x_dicts)
    max_w = max(x["imgs"].shape[2] for x in x_dicts)

    padded_imgs = []
    padded_chn_ids = []
    for x_dict in x_dicts:
        img = x_dict["imgs"]
        chn_ids = x_dict["chn_ids"]
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        img = F.pad(img, (0, pad_w, 0, pad_h))
        pad_c = max_channels - c
        if pad_c:
            img = torch.cat([img, torch.zeros((pad_c, max_h, max_w), dtype=img.dtype)], dim=0)
            chn_ids = torch.cat([chn_ids, torch.zeros((pad_c, *chn_ids.shape[1:]), dtype=chn_ids.dtype)], dim=0)
        padded_imgs.append(img)
        padded_chn_ids.append(chn_ids)
    batched_x_dict = {"imgs": torch.stack(padded_imgs), "chn_ids": torch.stack(padded_chn_ids)}
    return batched_x_dict, torch.tensor(labels), list(sensors), list(group_ids)


def _resolve_base_dataset(dataset: Dataset) -> Dataset:
    base = dataset
    while hasattr(base, "base_ds"):
        base = base.base_ds  # type: ignore[attr-defined]
    return base


def count_overlap_rows(
    df,
    *,
    group_column: str,
    sensor_column: str = "sensor",
) -> Tuple[int, int, int]:
    """Return (total_rows, overlap_group_count, overlap_row_count)."""
    total_rows = int(len(df))
    if group_column not in df.columns or sensor_column not in df.columns or total_rows == 0:
        return total_rows, 0, 0

    group_to_sensors: Dict[str, set[str]] = {}
    group_to_count: Dict[str, int] = {}
    group_values = df[group_column].tolist()
    sensor_values = df[sensor_column].tolist()
    for i, (gid_raw, sensor_raw) in enumerate(zip(group_values, sensor_values)):
        if gid_raw is None or (isinstance(gid_raw, float) and np.isnan(gid_raw)):
            gid = f"__row_{i}"
        else:
            gid = str(gid_raw)
        sensor = str(sensor_raw)
        if gid not in group_to_sensors:
            group_to_sensors[gid] = set()
            group_to_count[gid] = 0
        group_to_sensors[gid].add(sensor)
        group_to_count[gid] += 1

    overlap_groups = [gid for gid, sensors in group_to_sensors.items() if len(sensors) >= 2]
    overlap_rows = sum(group_to_count[gid] for gid in overlap_groups)
    return total_rows, len(overlap_groups), int(overlap_rows)


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


def _gather_sensor_indices(dataset: Dataset, sensors: Sequence[str]) -> Dict[str, Sequence[int]]:
    base_ds = _resolve_base_dataset(dataset)
    df = getattr(base_ds, "df", None)
    if df is None:
        raise ValueError("Dataset must expose a DataFrame to split by sensor.")
    idx_map = {sensor: [] for sensor in sensors}
    if "sensor" in df.columns:
        for idx, sensor_name in enumerate(df["sensor"].tolist()):
            if sensor_name in idx_map:
                idx_map[sensor_name].append(idx)
        return idx_map
    if "anchor_sensor" in df.columns:
        for idx, sensor_name in enumerate(df["anchor_sensor"].tolist()):
            key = str(sensor_name)
            if key in idx_map:
                idx_map[key].append(idx)
        return idx_map
    wide_sensor_columns = getattr(base_ds, "_wide_sensor_columns", {})
    if isinstance(wide_sensor_columns, Mapping) and len(wide_sensor_columns) > 0:
        for idx, row in df.iterrows():
            for sensor_name, cols in wide_sensor_columns.items():
                if sensor_name not in idx_map:
                    continue
                if sensor_name == "s5p":
                    value = str(row.get(cols[0], "")).strip().lower()
                    if value not in ("", "nan", "none", "null"):
                        idx_map[sensor_name].append(int(idx))
                    continue
                valid = True
                for col in cols:
                    value = str(row.get(col, "")).strip().lower()
                    if value in ("", "nan", "none", "null"):
                        valid = False
                        break
                if valid:
                    idx_map[sensor_name].append(int(idx))
        return idx_map
    raise ValueError(
        "Dataset must expose either 'sensor', 'anchor_sensor', or recognized wide-table sensor columns to split by sensor."
    )


def parse_sensor_coverage_overrides(raw: str) -> Dict[str, float]:
    """Parse sensor coverage overrides from 'sensor=value' pairs."""
    if not raw or not raw.strip():
        return {}
    parsed: Dict[str, float] = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" in item:
            sensor_name, value_text = item.split("=", 1)
        elif ":" in item:
            sensor_name, value_text = item.split(":", 1)
        else:
            raise ValueError(
                f"Invalid --phase1_sensor_coverages entry '{item}'. Use sensor=value, e.g. wv3=0.4,s5p=0.7."
            )
        sensor = sensor_name.strip()
        value = value_text.strip()
        if not sensor:
            raise ValueError(f"Invalid --phase1_sensor_coverages entry '{item}': empty sensor name.")
        try:
            coverage = float(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid coverage value '{value}' for sensor '{sensor}' in --phase1_sensor_coverages."
            ) from exc
        if not (0.0 < coverage <= 1.0):
            raise ValueError(
                f"Coverage for sensor '{sensor}' must be in (0, 1], got {coverage}."
            )
        parsed[sensor] = coverage
    return parsed


def parse_sensor_weight_overrides(raw: str, *, arg_name: str = "--overlap_sensor_weights") -> Dict[str, float]:
    """Parse sensor weight overrides from 'sensor=value' pairs."""
    if not raw or not raw.strip():
        return {}
    parsed: Dict[str, float] = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" in item:
            sensor_name, value_text = item.split("=", 1)
        elif ":" in item:
            sensor_name, value_text = item.split(":", 1)
        else:
            raise ValueError(f"Invalid {arg_name} entry '{item}'. Use sensor=value, e.g. wv3=1.4,s5p=0.7.")
        sensor = sensor_name.strip()
        value = value_text.strip()
        if not sensor:
            raise ValueError(f"Invalid {arg_name} entry '{item}': empty sensor name.")
        try:
            weight = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid weight value '{value}' for sensor '{sensor}' in {arg_name}.") from exc
        if not (weight > 0.0):
            raise ValueError(f"Weight for sensor '{sensor}' must be > 0 in {arg_name}, got {weight}.")
        parsed[sensor] = weight
    return parsed


def _stats_accumulate(acc: Dict[str, torch.Tensor], img: torch.Tensor) -> None:
    if img.ndim != 3:
        raise ValueError(f"Expected CHW tensor for stats accumulation, got shape={tuple(img.shape)}")
    c = img.shape[0]
    img64 = img.to(dtype=torch.float64)
    if not acc:
        acc["sum"] = torch.zeros((c,), dtype=torch.float64)
        acc["sumsq"] = torch.zeros((c,), dtype=torch.float64)
        acc["count"] = torch.zeros((c,), dtype=torch.float64)
    if acc["sum"].shape[0] != c:
        raise ValueError(f"Channel mismatch during stats accumulation: expected {acc['sum'].shape[0]}, got {c}")
    flat = img64.view(c, -1)
    acc["sum"] += flat.sum(dim=1)
    acc["sumsq"] += (flat * flat).sum(dim=1)
    acc["count"] += float(flat.shape[1])


def _stats_finalize(acc: Dict[str, torch.Tensor]) -> Optional[Tuple[list[float], list[float]]]:
    if not acc:
        return None
    count = torch.clamp(acc["count"], min=1.0)
    mean = acc["sum"] / count
    var = (acc["sumsq"] / count) - (mean * mean)
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean.to(dtype=torch.float32).tolist(), std.to(dtype=torch.float32).tolist()


def _estimate_sensor_stats_from_train_dataset(
    dataset: TriSensorTemporalCsvDataset,
    *,
    sensors: Sequence[str],
    path_columns: Sequence[str],
    max_samples_per_sensor: int,
    seed: int,
) -> Dict[str, Tuple[list[float], list[float]]]:
    df = dataset.df
    rng = random.Random(seed)
    stats_acc: Dict[str, Dict[str, torch.Tensor]] = {s: {} for s in sensors}
    wide_sensor_columns = getattr(dataset, "_wide_sensor_columns", {})
    table_mode = str(getattr(dataset, "_table_mode", "long"))

    def _row_has_valid_path(row, col_name: str) -> bool:
        value = str(row.get(col_name, "")).strip().lower()
        return value not in ("", "nan", "none", "null")

    for sensor in sensors:
        if "sensor" in df.columns:
            indices = [i for i, s in enumerate(df["sensor"].tolist()) if s == sensor]
        elif table_mode == "wide" and isinstance(wide_sensor_columns, Mapping):
            cols = wide_sensor_columns.get(sensor)
            indices = []
            if cols is not None:
                for idx, row in df.iterrows():
                    if sensor == "s5p":
                        if _row_has_valid_path(row, cols[0]):
                            indices.append(int(idx))
                        continue
                    if all(_row_has_valid_path(row, col) for col in cols):
                        indices.append(int(idx))
        else:
            indices = []
        if not indices:
            continue
        if max_samples_per_sensor > 0 and len(indices) > max_samples_per_sensor:
            indices = rng.sample(indices, max_samples_per_sensor)
        skipped = 0
        for idx in indices:
            row = df.iloc[idx]
            try:
                if sensor == "s5p":
                    if table_mode == "wide" and isinstance(wide_sensor_columns, Mapping):
                        s5p_cols = wide_sensor_columns.get("s5p")
                        if s5p_cols is None:
                            continue
                        img, _ = dataset._load_s5p_raw_sample(row, s5p_cols[0])
                    else:
                        img, _ = dataset._load_s5p_raw_sample(row, path_columns[0])
                    _stats_accumulate(stats_acc[sensor], img)
                    continue
                if sensor not in dataset.sensor_configs:
                    continue
                if table_mode == "wide" and isinstance(wide_sensor_columns, Mapping):
                    sensor_cols = wide_sensor_columns.get(sensor)
                    if sensor_cols is None:
                        continue
                    chosen_path_columns = sensor_cols
                else:
                    chosen_path_columns = path_columns
                with dataset._use_sensor_cfg(sensor):
                    for col in chosen_path_columns:
                        path = row[col]
                        path = dataset._maybe_cache_path(path)
                        img = dataset._read_image_raw(path)
                        if sensor == "l89" and dataset._align_l89_to_s2:
                            img = dataset._pad_l89_to_s2(img)
                        if sensor == "wv3":
                            abs_max = torch.amax(torch.abs(img))
                            if torch.isfinite(abs_max) and abs_max.item() > 100.0:
                                img = img / 65535.0
                        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                        _stats_accumulate(stats_acc[sensor], img)
            except Exception:
                skipped += 1
                continue
        if skipped > 0:
            print(f"[SensorStats] skipped {skipped} samples while estimating stats for sensor={sensor}.", flush=True)

    result: Dict[str, Tuple[list[float], list[float]]] = {}
    for sensor in sensors:
        finalized = _stats_finalize(stats_acc[sensor])
        if finalized is None:
            continue
        mean_list, std_list = finalized
        result[sensor] = (mean_list, std_list)
    return result


def _save_sensor_stats_cache(path: Path, stats: Mapping[str, Tuple[Sequence[float], Sequence[float]]]) -> None:
    payload: Dict[str, Dict[str, list[float]]] = {}
    for sensor, value in stats.items():
        mean, std = value
        payload[sensor] = {
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)


def _load_sensor_stats_cache(path: Path) -> Dict[str, Tuple[list[float], list[float]]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid sensor stats cache format: expected object at {path}")
    parsed: Dict[str, Tuple[list[float], list[float]]] = {}
    for sensor, block in payload.items():
        if not isinstance(block, Mapping):
            continue
        mean = block.get("mean")
        std = block.get("std")
        if not isinstance(mean, list) or not isinstance(std, list):
            continue
        if len(mean) == 0 or len(mean) != len(std):
            continue
        parsed[str(sensor)] = ([float(v) for v in mean], [float(v) for v in std])
    return parsed


def _select_batch_logits(
    outputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
    *,
    batch_size: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    logits = outputs.get("summary_logits")
    if isinstance(logits, torch.Tensor):
        return logits
    logits = outputs.get("merged_logits")
    if isinstance(logits, torch.Tensor):
        return logits
    merged = torch.zeros((batch_size, num_classes), device=device)
    for sensor_out in outputs.values():
        if not isinstance(sensor_out, dict):
            continue
        idx = sensor_out.get("indices")
        sensor_logits = sensor_out.get("logits")
        if isinstance(idx, torch.Tensor) and isinstance(sensor_logits, torch.Tensor) and idx.numel() > 0:
            merged.index_copy_(0, idx, sensor_logits)
    return merged


def _build_overlap_group_inputs(
    logits: torch.Tensor,
    sensors: Sequence[str],
    group_ids: Sequence[Any],
    sensor_order: Sequence[str],
    *,
    sensor_weights: Optional[Mapping[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank-2 [B,C], got shape={tuple(logits.shape)}")
    if len(sensors) != logits.shape[0] or len(group_ids) != logits.shape[0]:
        raise ValueError(
            "Batch size mismatch in overlap fusion inputs: "
            f"logits={logits.shape[0]}, sensors={len(sensors)}, group_ids={len(group_ids)}"
        )
    sensor_to_idx = {sensor: idx for idx, sensor in enumerate(sensor_order)}
    group_to_idx: Dict[str, int] = {}
    sample_to_group: list[int] = []
    for gid in group_ids:
        key = str(gid)
        mapped = group_to_idx.get(key)
        if mapped is None:
            mapped = len(group_to_idx)
            group_to_idx[key] = mapped
        sample_to_group.append(mapped)

    num_groups = len(group_to_idx)
    num_sensors = len(sensor_order)
    num_classes = logits.shape[-1]
    sums = logits.new_zeros((num_groups, num_sensors, num_classes))
    counts = logits.new_zeros((num_groups, num_sensors, 1))
    for row_idx, (sensor_name, group_idx) in enumerate(zip(sensors, sample_to_group)):
        sensor_idx = sensor_to_idx.get(sensor_name)
        if sensor_idx is None:
            raise ValueError(f"Unknown sensor '{sensor_name}' encountered in overlap fusion")
        weight = float(sensor_weights.get(sensor_name, 1.0)) if sensor_weights is not None else 1.0
        sums[group_idx, sensor_idx] += logits[row_idx] * weight
        counts[group_idx, sensor_idx, 0] += weight

    means = sums / torch.clamp(counts, min=1e-6)
    means = torch.where(counts > 0, means, torch.zeros_like(means))
    features = means.reshape(num_groups, -1)
    sample_to_group_tensor = torch.tensor(sample_to_group, device=logits.device, dtype=torch.long)
    return features, sample_to_group_tensor


def _compute_overlap_head_logits(
    model: MultiSensorPanopticonClassifier,
    logits: torch.Tensor,
    sensors: Sequence[str],
    group_ids: Sequence[Any],
    *,
    sensor_weights: Optional[Mapping[str, float]] = None,
) -> Optional[torch.Tensor]:
    if model.overlap_head is None:
        return None
    group_inputs, sample_to_group = _build_overlap_group_inputs(
        logits,
        sensors,
        group_ids,
        model.sensor_order,
        sensor_weights=sensor_weights,
    )
    group_logits = model.overlap_head(group_inputs)
    return group_logits.index_select(0, sample_to_group)


def _fuse_logits_by_overlap(
    logits: torch.Tensor,
    sensors: Sequence[str],
    group_ids: Sequence[Any],
    *,
    method: str,
    sensor_weights: Optional[Mapping[str, float]] = None,
    overall_head_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mode = str(method).strip().lower()
    if mode in ("", "none") or logits.shape[0] == 0:
        return logits
    if len(sensors) != logits.shape[0] or len(group_ids) != logits.shape[0]:
        raise ValueError(
            "Batch size mismatch in overlap logits fusion: "
            f"logits={logits.shape[0]}, sensors={len(sensors)}, group_ids={len(group_ids)}"
        )

    if mode == "overall_head":
        if isinstance(overall_head_logits, torch.Tensor):
            return overall_head_logits
        mode = "logit_mean"

    group_to_indices: Dict[str, list[int]] = {}
    for i, gid in enumerate(group_ids):
        key = str(gid)
        if key not in group_to_indices:
            group_to_indices[key] = []
        group_to_indices[key].append(i)

    fused = logits.clone()
    for indices in group_to_indices.values():
        if len(indices) <= 1:
            continue
        idx_t = torch.tensor(indices, device=logits.device, dtype=torch.long)
        group_logits = logits.index_select(0, idx_t)
        if sensor_weights is None:
            weights = logits.new_ones((len(indices),))
        else:
            weights = logits.new_tensor([float(sensor_weights.get(sensors[i], 1.0)) for i in indices])
        weight_sum = torch.clamp(weights.sum(), min=1e-6)

        if mode == "logit_mean":
            fused_group = (group_logits * weights.unsqueeze(-1)).sum(dim=0, keepdim=True) / weight_sum
        elif mode == "prob_mean":
            probs = torch.softmax(group_logits, dim=-1)
            fused_probs = (probs * weights.unsqueeze(-1)).sum(dim=0, keepdim=True) / weight_sum
            fused_group = torch.log(torch.clamp(fused_probs, min=1e-6))
        elif mode == "majority_vote":
            num_classes = int(group_logits.shape[-1])
            vote_scores = logits.new_zeros((num_classes,))
            tie_scores = logits.new_zeros((num_classes,))
            probs = torch.softmax(group_logits, dim=-1)
            pred_idx = torch.argmax(group_logits, dim=-1)
            for row_idx in range(group_logits.shape[0]):
                cls_idx = int(pred_idx[row_idx].item())
                w = float(weights[row_idx].item())
                vote_scores[cls_idx] += w
                tie_scores += probs[row_idx] * w
            max_vote = torch.max(vote_scores)
            tied = torch.nonzero(vote_scores == max_vote, as_tuple=False).flatten()
            if tied.numel() > 1:
                tie_sub = tie_scores.index_select(0, tied)
                winner = int(tied[int(torch.argmax(tie_sub).item())].item())
            else:
                winner = int(tied[0].item())
            fused_group = vote_scores.unsqueeze(0) + (1e-3 * tie_scores.unsqueeze(0))
            fused_group[0, winner] = fused_group[0, winner] + 1e-2
        else:
            raise ValueError(f"Unsupported overlap fusion method '{method}'")

        fused.index_copy_(0, idx_t, fused_group.expand(len(indices), -1))
    return fused


def build_sensor_dataloaders(
    dataset: Dataset,
    sensors: Sequence[str],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    oversample_to_max_steps: bool = False,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    idx_map = _gather_sensor_indices(dataset, sensors)
    base_steps = {
        sensor: max(1, math.ceil(len(indices) / batch_size)) if indices else 0 for sensor, indices in idx_map.items()
    }
    max_steps = max(base_steps.values(), default=0)
    loaders: Dict[str, DataLoader] = {}
    steps_per_sensor: Dict[str, int] = {}
    for sensor_name, indices in idx_map.items():
        if not indices:
            continue
        subset = Subset(dataset, indices)
        target_steps = max_steps if oversample_to_max_steps else base_steps[sensor_name]
        if target_steps <= 0:
            continue
        sampler = None
        if oversample_to_max_steps and len(indices) > 0:
            num_samples = target_steps * batch_size
            sampler = RandomSampler(subset, replacement=True, num_samples=num_samples)
        loaders[sensor_name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn,
        )
        steps_per_sensor[sensor_name] = target_steps
    return loaders, steps_per_sensor


def build_balanced_mixed_dataloader(
    dataset: Dataset,
    sensors: Sequence[str],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    oversample_to_max: bool = True,
    epoch_coverage: float = 1.0,
    sensor_coverages: Optional[Mapping[str, float]] = None,
) -> Tuple[DataLoader, int]:
    """Create a single DataLoader that mixes sensors and oversamples minority ones.

    - If ``oversample_to_max`` is True (default), each sensor is repeated until it
      reaches the size of the largest sensor, balancing class counts.
    - Otherwise, uses the natural counts.
    Returns (loader, total_steps_per_epoch).
    """

    if not (0.0 < epoch_coverage <= 1.0):
        raise ValueError(f"epoch_coverage must be in (0, 1], got {epoch_coverage}.")

    idx_map = _gather_sensor_indices(dataset, sensors)
    counts = {s: len(v) for s, v in idx_map.items() if v}
    if not counts:
        raise ValueError("No samples found for the configured sensors.")

    target = max(counts.values()) if oversample_to_max else None
    balanced_indices: list[int] = []
    for sensor, indices in idx_map.items():
        if not indices:
            continue
        desired = target if target is not None else len(indices)
        repeat = math.ceil(desired / len(indices))
        expanded = (indices * repeat)[:desired]
        sensor_coverage = 1.0
        if sensor_coverages and sensor in sensor_coverages:
            sensor_coverage = float(sensor_coverages[sensor])
            if not (0.0 < sensor_coverage <= 1.0):
                raise ValueError(f"Coverage for sensor '{sensor}' must be in (0, 1], got {sensor_coverage}.")
        if sensor_coverage < 1.0 and expanded:
            keep = max(1, math.ceil(len(expanded) * sensor_coverage))
            if keep < len(expanded):
                expanded = random.sample(expanded, keep)
        balanced_indices.extend(expanded)

    if epoch_coverage < 1.0 and balanced_indices:
        keep = max(1, math.ceil(len(balanced_indices) * epoch_coverage))
        if keep < len(balanced_indices):
            balanced_indices = random.sample(balanced_indices, keep)

    sampler = SubsetRandomSampler(balanced_indices) if shuffle else balanced_indices
    steps = math.ceil(len(balanced_indices) / batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if isinstance(sampler, SubsetRandomSampler) else shuffle,
        sampler=sampler if isinstance(sampler, SubsetRandomSampler) else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
    )
    return loader, steps


# --------------------------------------------------------------------------------------
#  Training utilities
# --------------------------------------------------------------------------------------

def _load_backbone(weights_path: Optional[str] = None, *, strict: bool = True) -> DinoVisionTransformer:
    from hubconf import _panopticon_vitb14

    backbone = _panopticon_vitb14()
    if weights_path in (None, "", "none", "scratch", "random"):
        return backbone
    ckpt_path = Path(weights_path)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, Mapping) and "backbone" in state:
        state = state["backbone"]
    backbone.load_state_dict(state, strict=strict)
    return backbone


def _index_batch(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, idx)


def _slice_x_dict(x_dict: MutableMapping[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: _index_batch(v, idx) for k, v in x_dict.items() if isinstance(v, torch.Tensor)}


def recursive_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: recursive_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [recursive_to_device(v, device) for v in x]
        return type(x)(t)
    return x


def set_trainable(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


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
    extra_state: Optional[Mapping[str, Any]] = None,
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
            "extra_state": {} if extra_state is None else dict(extra_state),
        },
        path,
    )


def try_resume(path: Path, model: nn.Module, optimizer, scheduler, scaler, device):
    if not path.is_file():
        return 1, 0, 0.0, 0.0, {}
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
    extra_state = ckpt.get("extra_state", {})
    if not isinstance(extra_state, Mapping):
        extra_state = {}
    print(f"Resumed from {path} at epoch {start_epoch-1}", flush=True)
    return start_epoch, global_step, best_train_acc, best_test_acc, dict(extra_state)


def build_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "noam":
        warmup_steps = max(args.warmup_steps, 1)
        noam_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _binary_auroc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC for binary labels {0,1} using rank statistics (tie-aware)."""
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape[0] != scores.shape[0]:
        return float("nan")
    n = labels.shape[0]
    if n == 0:
        return float("nan")
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = float(ranks[pos_mask].sum())
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def compute_binary_metrics(labels: np.ndarray, preds: np.ndarray, pos_scores: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(np.int64, copy=False)
    preds = preds.astype(np.int64, copy=False)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {
        "fpr": _safe_ratio(fp, fp + tn),
        "recall": _safe_ratio(tp, tp + fn),
        "auroc": _binary_auroc_from_scores(labels, pos_scores),
    }


def _compute_subset_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
    pos_scores: Sequence[float],
    indices: Sequence[int],
) -> Dict[str, float]:
    if len(indices) == 0:
        return {"count": 0.0, "acc": float("nan"), "fpr": float("nan"), "recall": float("nan"), "auroc": float("nan")}
    y = np.asarray([labels[i] for i in indices], dtype=np.int64)
    p = np.asarray([preds[i] for i in indices], dtype=np.int64)
    count = int(y.shape[0])
    acc = float((p == y).mean()) if count > 0 else float("nan")
    if len(pos_scores) == len(labels):
        s = np.asarray([pos_scores[i] for i in indices], dtype=np.float64)
        bin_metrics = compute_binary_metrics(labels=y, preds=p, pos_scores=s)
    else:
        bin_metrics = {"fpr": float("nan"), "recall": float("nan"), "auroc": float("nan")}
    return {
        "count": float(count),
        "acc": acc,
        "fpr": float(bin_metrics["fpr"]),
        "recall": float(bin_metrics["recall"]),
        "auroc": float(bin_metrics["auroc"]),
    }


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(payload), ensure_ascii=True) + "\n")


def gather_head_parameters(model: MultiSensorPanopticonClassifier) -> Sequence[nn.Parameter]:
    shared_patch = model.backbone.patch_embed
    extra_patch_params = []
    for sensor, module in model.sensor_patch_embeds.items():
        if module is shared_patch:
            continue
        extra_patch_params.extend(list(module.parameters()))
    summary_params = list(model.summary_head.parameters()) if model.summary_head is not None else []
    overlap_params = list(model.overlap_head.parameters()) if model.overlap_head is not None else []
    head_params = list(model.heads.parameters()) + summary_params + overlap_params + extra_patch_params
    return head_params


def _gather_adapter_parameters_from_blocks(blocks: Sequence[nn.Module]) -> Sequence[nn.Parameter]:
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for block in blocks:
        if not isinstance(block, SensorAdapterBlock):
            continue
        for param in block.adapters.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            params.append(param)
    return params


def gather_adapter_parameters(model: MultiSensorPanopticonClassifier) -> Sequence[nn.Parameter]:
    blocks = list(getattr(model.backbone, "_sensor_adapter_blocks", []))
    return _gather_adapter_parameters_from_blocks(blocks)


def _gather_backbone_non_adapter_parameters(model: MultiSensorPanopticonClassifier) -> Sequence[nn.Parameter]:
    adapter_ids = {id(p) for p in gather_adapter_parameters(model)}
    return [p for p in model.backbone.parameters() if id(p) not in adapter_ids]


def _set_first_backbone_blocks_trainable(model: MultiSensorPanopticonClassifier, first_blocks: int, requires_grad: bool) -> None:
    entries = _iter_transformer_blocks(model.backbone)
    if first_blocks <= 0:
        return
    for _, _, block in entries[: min(first_blocks, len(entries))]:
        target = block.block if isinstance(block, SensorAdapterBlock) else block
        set_trainable(target, requires_grad)
        if requires_grad:
            target.train()
        else:
            target.eval()


def _set_sensor_adapter_trainable(
    model: MultiSensorPanopticonClassifier,
    sensor_name: str,
    *,
    requires_grad: bool,
) -> None:
    domain_idx = model.sensor_to_idx.get(sensor_name)
    if domain_idx is None:
        return
    for block in getattr(model.backbone, "_sensor_adapter_blocks", []):
        if not isinstance(block, SensorAdapterBlock):
            continue
        if domain_idx < 0 or domain_idx >= len(block.adapters):
            continue
        adapter = block.adapters[domain_idx]
        adapter.train(requires_grad)
        for param in adapter.parameters():
            param.requires_grad = requires_grad


def _set_sensor_head_trainable(
    model: MultiSensorPanopticonClassifier,
    sensor_name: str,
    *,
    requires_grad: bool,
) -> None:
    if sensor_name not in model.heads:
        return
    head = model.heads[sensor_name]
    head.train(requires_grad)
    for param in head.parameters():
        param.requires_grad = requires_grad


def _set_sensor_patch_embed_trainable(
    model: MultiSensorPanopticonClassifier,
    sensor_name: str,
    *,
    requires_grad: bool,
) -> None:
    if sensor_name not in model.sensor_patch_embeds:
        return
    patch_embed = model.sensor_patch_embeds[sensor_name]
    patch_embed.train(requires_grad)
    for param in patch_embed.parameters():
        param.requires_grad = requires_grad


def _collect_sensor_adapter_state(
    model: MultiSensorPanopticonClassifier,
    sensor_name: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    domain_idx = model.sensor_to_idx.get(sensor_name)
    state: Dict[str, Dict[str, torch.Tensor]] = {}
    if domain_idx is None:
        return state
    for block_idx, block in enumerate(getattr(model.backbone, "_sensor_adapter_blocks", [])):
        if not isinstance(block, SensorAdapterBlock):
            continue
        if domain_idx < 0 or domain_idx >= len(block.adapters):
            continue
        state[str(block_idx)] = copy.deepcopy(block.adapters[domain_idx].state_dict())
    return state


def _restore_sensor_adapter_state(
    model: MultiSensorPanopticonClassifier,
    sensor_name: str,
    state: Mapping[str, Mapping[str, torch.Tensor]],
) -> None:
    domain_idx = model.sensor_to_idx.get(sensor_name)
    if domain_idx is None:
        return
    blocks = list(getattr(model.backbone, "_sensor_adapter_blocks", []))
    for key, adapter_state in state.items():
        try:
            block_idx = int(key)
        except ValueError:
            continue
        if block_idx < 0 or block_idx >= len(blocks):
            continue
        block = blocks[block_idx]
        if not isinstance(block, SensorAdapterBlock):
            continue
        if domain_idx < 0 or domain_idx >= len(block.adapters):
            continue
        block.adapters[domain_idx].load_state_dict(dict(adapter_state))


def _collect_sensor_head_state(
    model: MultiSensorPanopticonClassifier,
    sensor_name: str,
) -> Dict[str, torch.Tensor]:
    if sensor_name not in model.heads:
        return {}
    head = model.heads[sensor_name]
    return copy.deepcopy(head.state_dict())


def _restore_sensor_head_state(
    model: MultiSensorPanopticonClassifier,
    sensor_name: str,
    state: Mapping[str, torch.Tensor],
) -> None:
    if sensor_name not in model.heads:
        return
    head = model.heads[sensor_name]
    head.load_state_dict(dict(state))


def _collect_sensor_patch_embed_state(
    model: MultiSensorPanopticonClassifier,
    sensor_name: str,
) -> Dict[str, torch.Tensor]:
    if sensor_name not in model.sensor_patch_embeds:
        return {}
    patch_embed = model.sensor_patch_embeds[sensor_name]
    return copy.deepcopy(patch_embed.state_dict())


def _restore_sensor_patch_embed_state(
    model: MultiSensorPanopticonClassifier,
    sensor_name: str,
    state: Mapping[str, torch.Tensor],
) -> None:
    if sensor_name not in model.sensor_patch_embeds:
        return
    patch_embed = model.sensor_patch_embeds[sensor_name]
    patch_embed.load_state_dict(dict(state))


def _save_sensor_adapter_checkpoint(
    path: Path,
    *,
    model: MultiSensorPanopticonClassifier,
    sensor_name: str,
    epoch: int,
    score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter_state = _collect_sensor_adapter_state(model, sensor_name)
    head_state = _collect_sensor_head_state(model, sensor_name)
    patch_embed_state = _collect_sensor_patch_embed_state(model, sensor_name)
    torch.save(
        {
            "sensor": sensor_name,
            "epoch": int(epoch),
            "score": float(score),
            "adapter_state": adapter_state,
            "head_state": head_state,
            "patch_embed_state": patch_embed_state,
        },
        path,
    )


def _load_sensor_adapter_checkpoint(
    path: Path,
    *,
    model: MultiSensorPanopticonClassifier,
    sensor_name: str,
    device: torch.device,
) -> Optional[float]:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location=device)
    adapter_state = payload.get("adapter_state")
    if isinstance(adapter_state, Mapping):
        _restore_sensor_adapter_state(model, sensor_name, adapter_state)
    head_state = payload.get("head_state")
    if isinstance(head_state, Mapping):
        _restore_sensor_head_state(model, sensor_name, head_state)
    patch_embed_state = payload.get("patch_embed_state")
    if isinstance(patch_embed_state, Mapping):
        _restore_sensor_patch_embed_state(model, sensor_name, patch_embed_state)
    score = payload.get("score")
    return float(score) if score is not None else None


# --------------------------------------------------------------------------------------
#  CLI / training loop
# --------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-sensor Panopticon finetuning (S2/L89/S5P/WV3).")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--weights", default="weights/panopticon_vitb14_teacher.pth")
    parser.add_argument("--t0_col", default="path_t0")
    parser.add_argument("--t90_col", default="path_t90")
    parser.add_argument("--t360_col", default="path_t360")
    parser.add_argument(
        "--fusion_group_column",
        default="id",
        help="CSV column used to group overlapping cross-sensor samples into one decision unit.",
    )
    parser.add_argument("--s5p_data_key", default=None)
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
    parser.add_argument(
        "--adapter_lr",
        type=float,
        default=None,
        help="Override adapter LR. If unset, uses backbone_lr * adapter_lr_multiplier.",
    )
    parser.add_argument("--adapter_lr_multiplier", type=float, default=2.0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--train_backbone", action="store_true", dest="train_backbone")
    parser.add_argument("--freeze_backbone", action="store_false", dest="train_backbone")
    parser.set_defaults(train_backbone=True)
    parser.add_argument("--phase1_backbone_epochs", type=int, default=4)
    parser.add_argument(
        "--phase1_train_coverage",
        type=float,
        default=1.0,
        help="Fraction of train samples seen per epoch during phase1 backbone training. Range: (0, 1].",
    )
    parser.add_argument(
        "--phase1_sensor_coverages",
        type=str,
        default="",
        help="Optional per-sensor phase1 coverage overrides (sensor=value), e.g. 'wv3=0.4,s5p=0.7'.",
    )
    parser.add_argument(
        "--freeze_backbone_first_blocks",
        type=int,
        default=12,
        help="Kept for compatibility; this script uses LoRA adapters on all 12 ViT blocks.",
    )
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    parser.add_argument(
        "--adapter_bottleneck_dim",
        "--lora_rank",
        dest="adapter_bottleneck_dim",
        type=int,
        default=16,
        help="Low-rank dimension (LoRA rank) for sensor-specific adapters.",
    )
    parser.add_argument(
        "--adapter_alpha",
        "--lora_alpha",
        dest="adapter_alpha",
        type=float,
        default=16.0,
        help="LoRA scaling factor (effective scale is adapter_alpha / adapter_bottleneck_dim).",
    )
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--adapter_cls_only", action="store_true", dest="adapter_cls_only")
    parser.add_argument("--disable_adapter_cls_only", action="store_false", dest="adapter_cls_only")
    parser.set_defaults(adapter_cls_only=True)
    parser.add_argument(
        "--sensor_adapter_early_stop_sensors",
        type=str,
        default="wv3,s5p,s2",
        help="Comma-separated sensors using adapter early stopping rollback.",
    )
    parser.add_argument("--sensor_adapter_early_stopping_patience", type=int, default=6)
    parser.add_argument("--sensor_adapter_early_stopping_warmup_epochs", type=int, default=5)
    parser.add_argument("--sensor_adapter_early_stopping_min_delta", type=float, default=1e-4)
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
    parser.add_argument(
        "--sensor_stats_cache",
        type=str,
        default="",
        help="Optional JSON cache for per-sensor mean/std estimated from train split.",
    )
    parser.add_argument(
        "--recompute_sensor_stats",
        action="store_true",
        help="Force recomputation of per-sensor mean/std from train split and overwrite cache.",
    )
    parser.add_argument(
        "--sensor_stats_max_samples_per_sensor",
        type=int,
        default=512,
        help="Max train rows sampled per sensor when estimating mean/std; <=0 means all rows.",
    )
    parser.add_argument("--sensor_stats_seed", type=int, default=42)
    parser.add_argument(
        "--overlap_report_jsonl",
        type=str,
        default="",
        help="Optional JSONL path to append overlap-only evaluation reports each epoch.",
    )
    parser.add_argument("--oversample_minority", action="store_true", dest="oversample_minority")
    parser.add_argument("--disable_oversample_minority", action="store_false", dest="oversample_minority")
    parser.set_defaults(oversample_minority=True)
    parser.add_argument("--sensor_switch_interval", type=int, default=100)
    parser.add_argument("--summary_head", action="store_true", dest="summary_head")
    parser.add_argument("--disable_summary_head", action="store_false", dest="summary_head")
    parser.set_defaults(summary_head=True)
    parser.add_argument("--summary_hidden_dim", type=int, default=128)
    parser.add_argument("--summary_dropout", type=float, default=0.1)
    parser.add_argument("--summary_loss_weight", type=float, default=1.0)
    parser.add_argument("--overlap_head", action="store_true", dest="overlap_head")
    parser.add_argument("--disable_overlap_head", action="store_false", dest="overlap_head")
    parser.set_defaults(overlap_head=True)
    parser.add_argument("--overlap_hidden_dim", type=int, default=128)
    parser.add_argument("--overlap_dropout", type=float, default=0.1)
    parser.add_argument("--overlap_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--overlap_fusion",
        choices=["none", "majority_vote", "logit_mean", "prob_mean", "overall_head"],
        default="logit_mean",
        help="How to fuse logits inside overlap groups at decision time (default: weighted logit mean).",
    )
    parser.add_argument(
        "--overlap_fusion_train",
        action="store_true",
        help="Also apply overlap fusion to train-time accuracy reporting (evaluation always follows --overlap_fusion).",
    )
    parser.add_argument(
        "--overlap_sensor_weights",
        type=str,
        default="",
        help="Optional sensor reliability weights for overlap fusion (sensor=value), e.g. 'wv3=1.5,s5p=0.6'.",
    )
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    path_columns = (args.t0_col, args.t90_col, args.t360_col)
    wv3_band_names = [b.strip() for b in args.wv3_bands.split(",") if b.strip()]
    if len(wv3_band_names) == 0:
        raise ValueError("--wv3_bands must provide at least one WV3 band column name")
    wv3_chn_ids = load_wv3_channel_ids_from_srf(args.wv3_srf_csv, wv3_band_names).unsqueeze(-1)

    cache_obj = None
    if args.local_cache_dir:
        cache_obj = StaticAnchoredCache(args.local_cache_dir, min_free_gb=args.local_cache_min_free_gb)

    ds_kwargs = dict(
        path_columns=path_columns,
        fusion_group_column=args.fusion_group_column,
        local_file_cache=cache_obj,
        s5p_data_key=args.s5p_data_key,
        s5p_chn_ids_key=args.s5p_chn_ids_key,
        s5p_channels_last=args.s5p_channels_last,
        align_l89_to_s2=args.align_l89_to_s2,
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )
    sensor_stats_overrides: Dict[str, Tuple[list[float], list[float]]] = {}
    sensor_stats_cache_path: Optional[Path] = None
    if args.sensor_stats_cache and args.sensor_stats_cache.strip():
        sensor_stats_cache_path = Path(args.sensor_stats_cache).expanduser()
    if sensor_stats_cache_path is not None and sensor_stats_cache_path.is_file() and not args.recompute_sensor_stats:
        sensor_stats_overrides = _load_sensor_stats_cache(sensor_stats_cache_path)
        print(
            f"[SensorStats] Loaded cached per-sensor mean/std from {sensor_stats_cache_path} "
            f"(sensors={sorted(sensor_stats_overrides.keys())})",
            flush=True,
        )
    if args.recompute_sensor_stats or (sensor_stats_cache_path is not None and not sensor_stats_cache_path.is_file()):
        probe_ds = TriSensorTemporalCsvDataset(
            csv_path=args.train_csv,
            **ds_kwargs,
        )
        sensor_stats_overrides = _estimate_sensor_stats_from_train_dataset(
            probe_ds,
            sensors=("s2", "l89", "s5p", "wv3"),
            path_columns=path_columns,
            max_samples_per_sensor=int(args.sensor_stats_max_samples_per_sensor),
            seed=int(args.sensor_stats_seed),
        )
        if sensor_stats_cache_path is not None:
            _save_sensor_stats_cache(sensor_stats_cache_path, sensor_stats_overrides)
            print(
                f"[SensorStats] Recomputed and saved per-sensor mean/std to {sensor_stats_cache_path} "
                f"(sensors={sorted(sensor_stats_overrides.keys())})",
                flush=True,
            )
        else:
            print(
                f"[SensorStats] Recomputed per-sensor mean/std (sensors={sorted(sensor_stats_overrides.keys())}).",
                flush=True,
            )

    base_train_ds = TriSensorTemporalCsvDataset(
        csv_path=args.train_csv,
        sensor_stats_overrides=sensor_stats_overrides if sensor_stats_overrides else None,
        **ds_kwargs,
    )
    base_test_ds = TriSensorTemporalCsvDataset(
        csv_path=args.test_csv,
        sensor_stats_overrides=sensor_stats_overrides if sensor_stats_overrides else None,
        **ds_kwargs,
    )
    train_total_rows, train_overlap_groups, train_overlap_rows = count_overlap_rows(
        base_train_ds.df,
        group_column=args.fusion_group_column,
    )
    test_total_rows, test_overlap_groups, test_overlap_rows = count_overlap_rows(
        base_test_ds.df,
        group_column=args.fusion_group_column,
    )
    train_overlap_ratio = (train_overlap_rows / max(1, train_total_rows)) * 100.0
    test_overlap_ratio = (test_overlap_rows / max(1, test_total_rows)) * 100.0
    print(
        f"[OverlapStats] train: overlap_rows={train_overlap_rows}/{train_total_rows} ({train_overlap_ratio:.2f}%), "
        f"overlap_groups={train_overlap_groups} | "
        f"test: overlap_rows={test_overlap_rows}/{test_total_rows} ({test_overlap_ratio:.2f}%), "
        f"overlap_groups={test_overlap_groups}",
        flush=True,
    )

    if args.local_cache_warmup and cache_obj:
        all_paths: list[str] = []
        candidate_columns = list(path_columns)
        for ds in (base_train_ds, base_test_ds):
            wide_sensor_columns = getattr(ds, "_wide_sensor_columns", {})
            if isinstance(wide_sensor_columns, Mapping):
                for cols in wide_sensor_columns.values():
                    candidate_columns.extend(list(cols))
        all_paths.extend(collect_cache_paths_from_df(base_train_ds.df, candidate_columns))
        all_paths.extend(collect_cache_paths_from_df(base_test_ds.df, candidate_columns))
        cache_obj.warm_up(all_paths, max_workers=args.local_cache_workers)

    train_ds = ConcatTemporalDataset(base_train_ds)
    test_ds = ConcatTemporalDataset(base_test_ds)

    adapter_layer_count = 12
    if int(args.freeze_backbone_first_blocks) != adapter_layer_count:
        print(
            f"[Info] Overriding --freeze_backbone_first_blocks={args.freeze_backbone_first_blocks} "
            f"to {adapter_layer_count} so LoRA adapters cover all ViT blocks.",
            flush=True,
        )

    core_model = MultiSensorPanopticonClassifier(
        backbone=_load_backbone(args.weights),
        adapter_first_blocks=adapter_layer_count,
        adapter_bottleneck_dim=args.adapter_bottleneck_dim,
        adapter_alpha=args.adapter_alpha,
        adapter_dropout=args.adapter_dropout,
        adapter_cls_only=args.adapter_cls_only,
        enable_summary_head=args.summary_head,
        summary_hidden_dim=args.summary_hidden_dim,
        summary_dropout=args.summary_dropout,
        summary_loss_weight=args.summary_loss_weight,
        enable_overlap_head=args.overlap_head,
        overlap_hidden_dim=args.overlap_hidden_dim,
        overlap_dropout=args.overlap_dropout,
        overlap_loss_weight=args.overlap_loss_weight,
    ).to(device)
    model: nn.Module = core_model
    use_data_parallel = args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1
    if use_data_parallel:
        print(f"Enabling DataParallel across {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(core_model)
    elif args.data_parallel:
        print("DataParallel requested but insufficient CUDA devices; running single-device.", flush=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    adapter_blocks = list(getattr(core_model.backbone, "_sensor_adapter_blocks", []))
    adapter_params = list(gather_adapter_parameters(core_model))
    head_params = gather_head_parameters(core_model)
    backbone_params = list(_gather_backbone_non_adapter_parameters(core_model))
    adapter_lr_effective = args.adapter_lr if args.adapter_lr is not None else (args.backbone_lr * args.adapter_lr_multiplier)
    param_groups = [
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": adapter_params, "lr": adapter_lr_effective},
        {"params": head_params, "lr": args.head_lr},
    ]
    param_groups = [group for group in param_groups if len(group["params"]) > 0]
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay, betas=(args.momentum, 0.999))
    scheduler = build_scheduler(args, optimizer)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    run_name = args.wandb_run_name or default_run_name(args)
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    latest_path = ckpt_dir / "ckpt_latest.pth"
    best_path = ckpt_dir / "ckpt_best_test.pth"

    start_epoch = 1
    global_step = 0
    best_train_acc = 0.0
    best_test_acc = float("-inf")
    resume_state: Dict[str, Any] = {}

    if args.resume:
        start_epoch, global_step, best_train_acc, best_test_acc, resume_state = try_resume(
            latest_path, core_model, optimizer, scheduler, scaler, device
        )

    wandb_run = init_wandb(args)

    sensors_list = core_model.sensor_order
    phase1_epochs = max(0, int(args.phase1_backbone_epochs))
    phase1_train_coverage = float(args.phase1_train_coverage)
    if not (0.0 < phase1_train_coverage <= 1.0):
        raise ValueError(f"--phase1_train_coverage must be in (0, 1], got {phase1_train_coverage}.")
    phase1_sensor_coverages = parse_sensor_coverage_overrides(args.phase1_sensor_coverages)
    unknown_phase1_coverage_sensors = [s for s in phase1_sensor_coverages if s not in sensors_list]
    if unknown_phase1_coverage_sensors:
        raise ValueError(
            "Unknown sensors in --phase1_sensor_coverages: "
            f"{unknown_phase1_coverage_sensors}. Valid sensors: {sensors_list}"
        )
    overlap_sensor_weights = parse_sensor_weight_overrides(args.overlap_sensor_weights)
    unknown_overlap_weight_sensors = [s for s in overlap_sensor_weights if s not in sensors_list]
    if unknown_overlap_weight_sensors:
        raise ValueError(
            "Unknown sensors in --overlap_sensor_weights: "
            f"{unknown_overlap_weight_sensors}. Valid sensors: {sensors_list}"
        )
    if args.overlap_loss_weight < 0.0:
        raise ValueError(f"--overlap_loss_weight must be >= 0, got {args.overlap_loss_weight}.")
    if args.overlap_fusion == "overall_head" and core_model.overlap_head is None:
        warnings.warn(
            "--overlap_fusion=overall_head requested but overlap head is unavailable; falling back to logit_mean.",
            stacklevel=2,
        )
    if args.overlap_loss_weight > 0.0 and core_model.overlap_head is None:
        warnings.warn(
            "overlap_loss_weight > 0 but overlap head is unavailable; overlap loss will be skipped.",
            stacklevel=2,
        )
    early_stop_warmup = max(1, int(args.sensor_adapter_early_stopping_warmup_epochs))
    early_stop_patience = max(1, int(args.sensor_adapter_early_stopping_patience))
    early_stop_min_delta = max(0.0, float(args.sensor_adapter_early_stopping_min_delta))
    requested_early_stop_sensors = [s.strip() for s in args.sensor_adapter_early_stop_sensors.split(",") if s.strip()]
    unknown_early_stop_sensors = [s for s in requested_early_stop_sensors if s not in sensors_list]
    if unknown_early_stop_sensors:
        print(f"[Warn] Ignoring unknown sensors in --sensor_adapter_early_stop_sensors: {unknown_early_stop_sensors}", flush=True)
    early_stop_sensors = [s for s in requested_early_stop_sensors if s in sensors_list]
    sensor_adapter_best_score = {sensor: float("-inf") for sensor in early_stop_sensors}
    sensor_adapter_bad_epochs = {sensor: 0 for sensor in early_stop_sensors}
    sensor_adapter_frozen = {sensor: False for sensor in early_stop_sensors}
    sensor_adapter_best_epoch = {sensor: 0 for sensor in early_stop_sensors}
    sensor_adapter_ckpt_paths = {
        sensor: ckpt_dir / f"ckpt_best_adapter_{sensor}.pth" for sensor in early_stop_sensors
    }
    if isinstance(resume_state, Mapping):
        saved_best = resume_state.get("sensor_adapter_best_score")
        if isinstance(saved_best, Mapping):
            for sensor in early_stop_sensors:
                value = saved_best.get(sensor)
                if value is not None:
                    sensor_adapter_best_score[sensor] = float(value)
        saved_bad = resume_state.get("sensor_adapter_bad_epochs")
        if isinstance(saved_bad, Mapping):
            for sensor in early_stop_sensors:
                value = saved_bad.get(sensor)
                if value is not None:
                    sensor_adapter_bad_epochs[sensor] = int(value)
        saved_frozen = resume_state.get("sensor_adapter_frozen")
        if isinstance(saved_frozen, Mapping):
            for sensor in early_stop_sensors:
                value = saved_frozen.get(sensor)
                if value is not None:
                    sensor_adapter_frozen[sensor] = bool(value)
        saved_best_epoch = resume_state.get("sensor_adapter_best_epoch")
        if isinstance(saved_best_epoch, Mapping):
            for sensor in early_stop_sensors:
                value = saved_best_epoch.get(sensor)
                if value is not None:
                    sensor_adapter_best_epoch[sensor] = int(value)
        # Ensure frozen adapters/heads/patch-embeds are restored to best checkpoint state.
        for sensor in early_stop_sensors:
            if sensor_adapter_frozen[sensor]:
                _load_sensor_adapter_checkpoint(
                    sensor_adapter_ckpt_paths[sensor],
                    model=core_model,
                    sensor_name=sensor,
                    device=device,
                )
                _set_sensor_adapter_trainable(core_model, sensor, requires_grad=False)
                _set_sensor_head_trainable(core_model, sensor, requires_grad=False)
                _set_sensor_patch_embed_trainable(core_model, sensor, requires_grad=False)

    pin_memory = device.type == "cuda"
    # Default train loader (full coverage), reused outside phase1 undersampling.
    train_loader_default, train_steps_default = build_balanced_mixed_dataloader(
        train_ds,
        sensors_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        oversample_to_max=args.oversample_minority,
    )
    phase1_dynamic_loader = (phase1_train_coverage < 1.0) or bool(phase1_sensor_coverages)
    test_loader, test_steps = build_balanced_mixed_dataloader(
        test_ds,
        sensors_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        oversample_to_max=False,
    )
    sensor_stats_mode = "precomputed_defaults"
    if sensor_stats_overrides:
        sensor_stats_mode = f"train_estimated({sorted(sensor_stats_overrides.keys())})"

    print(
        f"Using device={device}, train_samples={len(train_ds)}, test_samples={len(test_ds)}, "
        f"sensors={sensors_list}, phase1_backbone_epochs={phase1_epochs}, "
        f"phase1_train_coverage={phase1_train_coverage:.3f}, phase1_sensor_coverages={phase1_sensor_coverages}, "
        f"adapter_layers={adapter_layer_count}, adapter_lr={adapter_lr_effective:.2e}, "
        "phase2_mode=freeze_vit_backbone_train_adapters_sensor_heads_conv3d, "
        f"lora_rank={args.adapter_bottleneck_dim}, lora_alpha={args.adapter_alpha:.2f}, "
        f"adapter_early_stop_sensors={early_stop_sensors}, "
        f"overlap_group_column={args.fusion_group_column}, overlap_fusion={args.overlap_fusion}, "
        f"overlap_fusion_train={int(args.overlap_fusion_train)}, overlap_loss_weight={args.overlap_loss_weight:.3f}, "
        f"overlap_sensor_weights={overlap_sensor_weights}, sensor_stats={sensor_stats_mode}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        in_phase1 = epoch <= phase1_epochs
        if in_phase1 and phase1_dynamic_loader:
            train_loader, train_steps = build_balanced_mixed_dataloader(
                train_ds,
                sensors_list,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                shuffle=True,
                oversample_to_max=args.oversample_minority,
                epoch_coverage=phase1_train_coverage,
                sensor_coverages=phase1_sensor_coverages,
            )
        else:
            train_loader, train_steps = train_loader_default, train_steps_default
        model.train()
        set_trainable(core_model.heads, True)
        core_model.heads.train()
        if core_model.summary_head is not None:
            set_trainable(core_model.summary_head, True)
            core_model.summary_head.train()
        if core_model.overlap_head is not None:
            set_trainable(core_model.overlap_head, True)
            core_model.overlap_head.train()
        for sensor in core_model.sensor_patch_embeds.values():
            set_trainable(sensor, True)
            sensor.train()

        if in_phase1:
            set_trainable(core_model.backbone, args.train_backbone)
            if args.train_backbone:
                core_model.backbone.train()
            else:
                core_model.backbone.eval()
            for param in adapter_params:
                param.requires_grad = False
            for block in adapter_blocks:
                if isinstance(block, SensorAdapterBlock):
                    block.adapters.eval()
        else:
            # After phase 1: freeze ViT backbone, train sensor adapters and heads.
            set_trainable(core_model.backbone, False)
            core_model.backbone.eval()
            for sensor in core_model.sensor_patch_embeds.values():
                set_trainable(sensor, True)
                sensor.train()
            for param in adapter_params:
                param.requires_grad = True
            for block in adapter_blocks:
                if isinstance(block, SensorAdapterBlock):
                    block.adapters.train()
            for sensor_name, frozen in sensor_adapter_frozen.items():
                if frozen:
                    _set_sensor_adapter_trainable(core_model, sensor_name, requires_grad=False)
                    _set_sensor_head_trainable(core_model, sensor_name, requires_grad=False)
                    _set_sensor_patch_embed_trainable(core_model, sensor_name, requires_grad=False)

        total_loss = 0.0
        total = 0
        correct = 0
        per_sensor_loss_accum = {sensor: 0.0 for sensor in sensors_list}
        per_sensor_count = {sensor: 0 for sensor in sensors_list}
        summary_loss_accum = 0.0
        summary_loss_count = 0
        overlap_loss_accum = 0.0
        overlap_loss_count = 0
        train_sensor_correct = {sensor: 0 for sensor in sensors_list}
        train_sensor_total = {sensor: 0 for sensor in sensors_list}
        num_classes = core_model.heads[sensors_list[0]].out_features

        step_idx = 0
        for x_dict, labels, sensors, group_ids in train_loader:
            step_idx += 1
            labels = labels.to(device)
            x_dict = recursive_to_device(x_dict, device)
            sensor_arg: Union[Sequence[str], torch.Tensor]
            if use_data_parallel:
                sensor_arg = core_model.encode_sensors(sensors, device=device)
            else:
                sensor_arg = sensors
            with autocast(enabled=use_amp):
                outputs = model(x_dict, sensors=sensor_arg)
                loss, per_sensor_losses = core_model.loss_from_outputs(outputs, labels, sensors, criterion)
                batch = labels.size(0)
                base_logits = _select_batch_logits(
                    outputs,
                    batch_size=batch,
                    num_classes=num_classes,
                    device=device,
                )
                overlap_head_logits: Optional[torch.Tensor] = None
                need_overlap_head = (
                    core_model.overlap_head is not None
                    and (args.overlap_fusion == "overall_head" or args.overlap_loss_weight > 0.0)
                )
                if need_overlap_head:
                    overlap_head_logits = _compute_overlap_head_logits(
                        core_model,
                        base_logits,
                        sensors,
                        group_ids,
                        sensor_weights=overlap_sensor_weights if overlap_sensor_weights else None,
                    )
                if overlap_head_logits is not None and args.overlap_loss_weight > 0.0:
                    overlap_loss = criterion(overlap_head_logits, labels)
                    per_sensor_losses["overlap"] = overlap_loss
                    loss = loss + (core_model.overlap_loss_weight * overlap_loss)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in core_model.parameters() if p.requires_grad),
                    args.max_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            total_loss += loss.item() * batch
            total += batch
            pred_logits = base_logits
            if args.overlap_fusion_train and args.overlap_fusion != "none":
                pred_logits = _fuse_logits_by_overlap(
                    pred_logits,
                    sensors,
                    group_ids,
                    method=args.overlap_fusion,
                    sensor_weights=overlap_sensor_weights if overlap_sensor_weights else None,
                    overall_head_logits=overlap_head_logits,
                )
            preds = pred_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            for i, sensor_type in enumerate(sensors):
                train_sensor_total[sensor_type] += 1
                if preds[i] == labels[i]:
                    train_sensor_correct[sensor_type] += 1
            for sensor_name, sensor_loss in per_sensor_losses.items():
                if sensor_name == "summary":
                    summary_loss_accum += sensor_loss.item()
                    summary_loss_count += 1
                    continue
                if sensor_name == "overlap":
                    overlap_loss_accum += sensor_loss.item()
                    overlap_loss_count += 1
                    continue
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
                if "summary" in per_sensor_losses:
                    sensor_loss_details = f"{sensor_loss_details} summary_loss={per_sensor_losses['summary'].item():.4f}".strip()
                if "overlap" in per_sensor_losses:
                    sensor_loss_details = f"{sensor_loss_details} overlap_loss={per_sensor_losses['overlap'].item():.4f}".strip()
                print(
                    f"Epoch {epoch} step {step_idx}/{train_steps} train_loss={total_loss/total:.4f} "
                    f"train_acc={correct/total:.4f} {sensor_loss_details}",
                    flush=True,
                )

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)
        train_per_sensor_acc = {
            sensor: (train_sensor_correct[sensor] / train_sensor_total[sensor] if train_sensor_total[sensor] > 0 else float("nan"))
            for sensor in sensors_list
        }

        model.eval()
        sensor_correct = {sensor: 0 for sensor in sensors_list}
        sensor_total = {sensor: 0 for sensor in sensors_list}
        test_loss_total = 0.0
        total_eval = 0
        correct_eval = 0
        eval_steps = 0
        eval_labels: list[int] = []
        eval_preds: list[int] = []
        eval_pos_scores: list[float] = []
        eval_base_preds: list[int] = []
        eval_base_pos_scores: list[float] = []
        eval_group_ids: list[str] = []
        eval_sample_sensors: list[str] = []
        eval_sensor_labels = {sensor: [] for sensor in sensors_list}
        eval_sensor_preds = {sensor: [] for sensor in sensors_list}
        eval_sensor_pos_scores = {sensor: [] for sensor in sensors_list}
        with torch.no_grad():
            for step, (x_dict, labels, sensors, group_ids) in enumerate(test_loader, 1):
                labels = labels.to(device)
                x_dict = recursive_to_device(x_dict, device)
                if use_data_parallel:
                    sensor_arg = core_model.encode_sensors(sensors, device=device)
                else:
                    sensor_arg = sensors
                with autocast(enabled=use_amp):
                    outputs = model(x_dict, sensors=sensor_arg)
                    loss, _ = core_model.loss_from_outputs(outputs, labels, sensors, criterion)
                    batch = labels.size(0)
                    base_logits = _select_batch_logits(
                        outputs,
                        batch_size=batch,
                        num_classes=num_classes,
                        device=device,
                    )
                    overlap_head_logits: Optional[torch.Tensor] = None
                    need_overlap_head = (
                        core_model.overlap_head is not None
                        and (args.overlap_fusion == "overall_head" or args.overlap_loss_weight > 0.0)
                    )
                    if need_overlap_head:
                        overlap_head_logits = _compute_overlap_head_logits(
                            core_model,
                            base_logits,
                            sensors,
                            group_ids,
                            sensor_weights=overlap_sensor_weights if overlap_sensor_weights else None,
                        )
                    if overlap_head_logits is not None and args.overlap_loss_weight > 0.0:
                        loss = loss + (core_model.overlap_loss_weight * criterion(overlap_head_logits, labels))
                batch = labels.size(0)
                test_loss_total += loss.item() * batch
                total_eval += batch
                base_preds = base_logits.argmax(dim=1)
                logits = base_logits
                if args.overlap_fusion != "none":
                    logits = _fuse_logits_by_overlap(
                        logits,
                        sensors,
                        group_ids,
                        method=args.overlap_fusion,
                        sensor_weights=overlap_sensor_weights if overlap_sensor_weights else None,
                        overall_head_logits=overlap_head_logits,
                    )
                preds = logits.argmax(dim=1)
                correct_eval += (preds == labels).sum().item()
                eval_labels.extend(labels.detach().to("cpu").tolist())
                eval_preds.extend(preds.detach().to("cpu").tolist())
                eval_base_preds.extend(base_preds.detach().to("cpu").tolist())
                eval_group_ids.extend([str(g) for g in group_ids])
                eval_sample_sensors.extend(list(sensors))
                batch_pos_scores: Optional[list[float]] = None
                batch_base_pos_scores: Optional[list[float]] = None
                if logits.shape[-1] == 2:
                    pos_scores = torch.softmax(logits, dim=1)[:, 1]
                    batch_pos_scores = pos_scores.detach().to("cpu").tolist()
                    eval_pos_scores.extend(batch_pos_scores)
                if base_logits.shape[-1] == 2:
                    base_pos_scores = torch.softmax(base_logits, dim=1)[:, 1]
                    batch_base_pos_scores = base_pos_scores.detach().to("cpu").tolist()
                    eval_base_pos_scores.extend(batch_base_pos_scores)
                for i, sensor_type in enumerate(sensors):
                    sensor_total[sensor_type] += 1
                    if preds[i] == labels[i]:
                        sensor_correct[sensor_type] += 1
                    eval_sensor_labels[sensor_type].append(int(labels[i].item()))
                    eval_sensor_preds[sensor_type].append(int(preds[i].item()))
                    if batch_pos_scores is not None:
                        eval_sensor_pos_scores[sensor_type].append(float(batch_pos_scores[i]))
                eval_steps += 1
                if args.max_eval_steps is not None and eval_steps >= args.max_eval_steps:
                    break

        test_loss = test_loss_total / max(1, total_eval)
        test_acc = correct_eval / max(1, total_eval)
        per_sensor_acc = {
            sensor: (sensor_correct[sensor] / sensor_total[sensor] if sensor_total[sensor] > 0 else float("nan"))
            for sensor in sensors_list
        }
        test_fpr = float("nan")
        test_recall = float("nan")
        test_auroc = float("nan")
        if eval_labels and eval_preds and eval_pos_scores and len(eval_labels) == len(eval_pos_scores):
            metrics = compute_binary_metrics(
                labels=np.asarray(eval_labels, dtype=np.int64),
                preds=np.asarray(eval_preds, dtype=np.int64),
                pos_scores=np.asarray(eval_pos_scores, dtype=np.float64),
            )
            test_fpr = metrics["fpr"]
            test_recall = metrics["recall"]
            test_auroc = metrics["auroc"]
        per_sensor_binary_metrics = {
            sensor: {"fpr": float("nan"), "recall": float("nan"), "auroc": float("nan")}
            for sensor in sensors_list
        }
        for sensor in sensors_list:
            sensor_labels = eval_sensor_labels[sensor]
            sensor_preds = eval_sensor_preds[sensor]
            sensor_scores = eval_sensor_pos_scores[sensor]
            if sensor_labels and sensor_preds and sensor_scores and len(sensor_labels) == len(sensor_scores):
                per_sensor_binary_metrics[sensor] = compute_binary_metrics(
                    labels=np.asarray(sensor_labels, dtype=np.int64),
                    preds=np.asarray(sensor_preds, dtype=np.int64),
                    pos_scores=np.asarray(sensor_scores, dtype=np.float64),
                )

        group_sensor_sets: Dict[str, set[str]] = {}
        for gid, sensor_name in zip(eval_group_ids, eval_sample_sensors):
            if gid not in group_sensor_sets:
                group_sensor_sets[gid] = set()
            group_sensor_sets[gid].add(sensor_name)
        overlap_indices = [
            i for i, gid in enumerate(eval_group_ids) if len(group_sensor_sets.get(gid, set())) >= 2
        ]
        non_overlap_indices = [
            i for i, gid in enumerate(eval_group_ids) if len(group_sensor_sets.get(gid, set())) < 2
        ]
        overlap_metrics_fused = _compute_subset_metrics(
            labels=eval_labels,
            preds=eval_preds,
            pos_scores=eval_pos_scores,
            indices=overlap_indices,
        )
        overlap_metrics_base = _compute_subset_metrics(
            labels=eval_labels,
            preds=eval_base_preds,
            pos_scores=eval_base_pos_scores,
            indices=overlap_indices,
        )
        non_overlap_metrics_fused = _compute_subset_metrics(
            labels=eval_labels,
            preds=eval_preds,
            pos_scores=eval_pos_scores,
            indices=non_overlap_indices,
        )
        non_overlap_metrics_base = _compute_subset_metrics(
            labels=eval_labels,
            preds=eval_base_preds,
            pos_scores=eval_base_pos_scores,
            indices=non_overlap_indices,
        )
        overlap_per_sensor_compare: Dict[str, Dict[str, float]] = {}
        for sensor in sensors_list:
            sensor_overlap_indices = [i for i in overlap_indices if eval_sample_sensors[i] == sensor]
            fused_sensor = _compute_subset_metrics(
                labels=eval_labels,
                preds=eval_preds,
                pos_scores=eval_pos_scores,
                indices=sensor_overlap_indices,
            )
            base_sensor = _compute_subset_metrics(
                labels=eval_labels,
                preds=eval_base_preds,
                pos_scores=eval_base_pos_scores,
                indices=sensor_overlap_indices,
            )
            overlap_per_sensor_compare[sensor] = {
                "count": fused_sensor["count"],
                "fused_acc": fused_sensor["acc"],
                "base_acc": base_sensor["acc"],
                "fused_fpr": fused_sensor["fpr"],
                "base_fpr": base_sensor["fpr"],
                "fused_recall": fused_sensor["recall"],
                "base_recall": base_sensor["recall"],
                "fused_auroc": fused_sensor["auroc"],
                "base_auroc": base_sensor["auroc"],
            }
        non_overlap_per_sensor_compare: Dict[str, Dict[str, float]] = {}
        for sensor in sensors_list:
            sensor_non_overlap_indices = [i for i in non_overlap_indices if eval_sample_sensors[i] == sensor]
            fused_sensor = _compute_subset_metrics(
                labels=eval_labels,
                preds=eval_preds,
                pos_scores=eval_pos_scores,
                indices=sensor_non_overlap_indices,
            )
            base_sensor = _compute_subset_metrics(
                labels=eval_labels,
                preds=eval_base_preds,
                pos_scores=eval_base_pos_scores,
                indices=sensor_non_overlap_indices,
            )
            non_overlap_per_sensor_compare[sensor] = {
                "count": fused_sensor["count"],
                "fused_acc": fused_sensor["acc"],
                "base_acc": base_sensor["acc"],
                "fused_fpr": fused_sensor["fpr"],
                "base_fpr": base_sensor["fpr"],
                "fused_recall": fused_sensor["recall"],
                "base_recall": base_sensor["recall"],
                "fused_auroc": fused_sensor["auroc"],
                "base_auroc": base_sensor["auroc"],
            }

        newly_frozen_sensors: list[str] = []
        if (not in_phase1) and early_stop_sensors:
            for sensor in early_stop_sensors:
                if sensor_adapter_frozen[sensor]:
                    continue
                sensor_metric = float(per_sensor_acc.get(sensor, float("nan")))
                if not math.isfinite(sensor_metric):
                    continue
                if sensor_metric > (sensor_adapter_best_score[sensor] + early_stop_min_delta):
                    sensor_adapter_best_score[sensor] = sensor_metric
                    sensor_adapter_bad_epochs[sensor] = 0
                    sensor_adapter_best_epoch[sensor] = epoch
                    _save_sensor_adapter_checkpoint(
                        sensor_adapter_ckpt_paths[sensor],
                        model=core_model,
                        sensor_name=sensor,
                        epoch=epoch,
                        score=sensor_metric,
                    )
                else:
                    sensor_adapter_bad_epochs[sensor] += 1

                if epoch >= (phase1_epochs + early_stop_warmup) and sensor_adapter_bad_epochs[sensor] >= early_stop_patience:
                    restored_score = _load_sensor_adapter_checkpoint(
                        sensor_adapter_ckpt_paths[sensor],
                        model=core_model,
                        sensor_name=sensor,
                        device=device,
                    )
                    if restored_score is not None:
                        sensor_adapter_best_score[sensor] = restored_score
                    sensor_adapter_frozen[sensor] = True
                    _set_sensor_adapter_trainable(core_model, sensor, requires_grad=False)
                    _set_sensor_head_trainable(core_model, sensor, requires_grad=False)
                    _set_sensor_patch_embed_trainable(core_model, sensor, requires_grad=False)
                    newly_frozen_sensors.append(sensor)

        for sensor in newly_frozen_sensors:
            print(
                f"[SensorAdapterEarlyStop] sensor={sensor} adapter+head+conv3d frozen at epoch={epoch}, "
                f"best_acc={sensor_adapter_best_score[sensor]:.5f}, "
                f"best_epoch={sensor_adapter_best_epoch[sensor]}, "
                f"bad_epochs={sensor_adapter_bad_epochs[sensor]}",
                flush=True,
            )

        train_acc_str = " ".join(f"train_acc_{s}={train_per_sensor_acc[s]:.4f}" for s in sensors_list)
        test_acc_str = " ".join(f"test_acc_{s}={per_sensor_acc[s]:.4f}" for s in sensors_list)
        test_metric_str = " ".join(
            f"{s}(fpr={per_sensor_binary_metrics[s]['fpr']:.4f},recall={per_sensor_binary_metrics[s]['recall']:.4f},auroc={per_sensor_binary_metrics[s]['auroc']:.4f})"
            for s in sensors_list
        )
        overlap_sensor_compare_str = " ".join(
            f"{s}(n={int(overlap_per_sensor_compare[s]['count'])},fused_acc={overlap_per_sensor_compare[s]['fused_acc']:.4f},base_acc={overlap_per_sensor_compare[s]['base_acc']:.4f})"
            for s in sensors_list
        )
        non_overlap_sensor_compare_str = " ".join(
            f"{s}(n={int(non_overlap_per_sensor_compare[s]['count'])},fused_acc={non_overlap_per_sensor_compare[s]['fused_acc']:.4f},base_acc={non_overlap_per_sensor_compare[s]['base_acc']:.4f})"
            for s in sensors_list
        )
        sensor_freeze_str = " ".join(
            f"adapter_frozen_{sensor}={int(sensor_adapter_frozen[sensor])}" for sensor in early_stop_sensors
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"test_fpr={test_fpr:.4f} test_recall={test_recall:.4f} test_auroc={test_auroc:.4f} "
            f"overlap_n={int(overlap_metrics_fused['count'])} "
            f"overlap_fused(acc={overlap_metrics_fused['acc']:.4f},fpr={overlap_metrics_fused['fpr']:.4f},recall={overlap_metrics_fused['recall']:.4f},auroc={overlap_metrics_fused['auroc']:.4f}) "
            f"overlap_base(acc={overlap_metrics_base['acc']:.4f},fpr={overlap_metrics_base['fpr']:.4f},recall={overlap_metrics_base['recall']:.4f},auroc={overlap_metrics_base['auroc']:.4f}) "
            f"non_overlap_n={int(non_overlap_metrics_fused['count'])} "
            f"non_overlap_fused(acc={non_overlap_metrics_fused['acc']:.4f},fpr={non_overlap_metrics_fused['fpr']:.4f},recall={non_overlap_metrics_fused['recall']:.4f},auroc={non_overlap_metrics_fused['auroc']:.4f}) "
            f"non_overlap_base(acc={non_overlap_metrics_base['acc']:.4f},fpr={non_overlap_metrics_base['fpr']:.4f},recall={non_overlap_metrics_base['recall']:.4f},auroc={non_overlap_metrics_base['auroc']:.4f}) "
            f"{train_acc_str} {test_acc_str} test_metrics_per_sensor={test_metric_str} "
            f"overlap_sensor_compare={overlap_sensor_compare_str} "
            f"non_overlap_sensor_compare={non_overlap_sensor_compare_str} {sensor_freeze_str}",
            flush=True,
        )

        best_train_acc = max(best_train_acc, train_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            ckpt_extra_state = {
                "sensor_adapter_best_score": sensor_adapter_best_score,
                "sensor_adapter_bad_epochs": sensor_adapter_bad_epochs,
                "sensor_adapter_frozen": sensor_adapter_frozen,
                "sensor_adapter_best_epoch": sensor_adapter_best_epoch,
                "phase1_backbone_epochs": phase1_epochs,
                "freeze_backbone_first_blocks": adapter_layer_count,
            }
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
                extra_state=ckpt_extra_state,
            )
        ckpt_extra_state = {
            "sensor_adapter_best_score": sensor_adapter_best_score,
            "sensor_adapter_bad_epochs": sensor_adapter_bad_epochs,
            "sensor_adapter_frozen": sensor_adapter_frozen,
            "sensor_adapter_best_epoch": sensor_adapter_best_epoch,
            "phase1_backbone_epochs": phase1_epochs,
            "freeze_backbone_first_blocks": adapter_layer_count,
        }
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
            extra_state=ckpt_extra_state,
        )

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
                "overlap_count": overlap_metrics_fused["count"],
                "overlap_acc_fused": overlap_metrics_fused["acc"],
                "overlap_fpr_fused": overlap_metrics_fused["fpr"],
                "overlap_recall_fused": overlap_metrics_fused["recall"],
                "overlap_auroc_fused": overlap_metrics_fused["auroc"],
                "overlap_acc_base": overlap_metrics_base["acc"],
                "overlap_fpr_base": overlap_metrics_base["fpr"],
                "overlap_recall_base": overlap_metrics_base["recall"],
                "overlap_auroc_base": overlap_metrics_base["auroc"],
                "non_overlap_count": non_overlap_metrics_fused["count"],
                "non_overlap_acc_fused": non_overlap_metrics_fused["acc"],
                "non_overlap_fpr_fused": non_overlap_metrics_fused["fpr"],
                "non_overlap_recall_fused": non_overlap_metrics_fused["recall"],
                "non_overlap_auroc_fused": non_overlap_metrics_fused["auroc"],
                "non_overlap_acc_base": non_overlap_metrics_base["acc"],
                "non_overlap_fpr_base": non_overlap_metrics_base["fpr"],
                "non_overlap_recall_base": non_overlap_metrics_base["recall"],
                "non_overlap_auroc_base": non_overlap_metrics_base["auroc"],
                "phase1": int(in_phase1),
                "adapter_stage": int(not in_phase1),
                "overlap_fusion_enabled": int(args.overlap_fusion != "none"),
                "overlap_fusion_train": int(args.overlap_fusion_train),
            }
            for sensor in sensors_list:
                log_payload[f"test_acc_{sensor}"] = per_sensor_acc[sensor]
                log_payload[f"test_fpr_{sensor}"] = per_sensor_binary_metrics[sensor]["fpr"]
                log_payload[f"test_recall_{sensor}"] = per_sensor_binary_metrics[sensor]["recall"]
                log_payload[f"test_auroc_{sensor}"] = per_sensor_binary_metrics[sensor]["auroc"]
                if per_sensor_count[sensor] > 0:
                    log_payload[f"train_loss_{sensor}"] = (
                        per_sensor_loss_accum[sensor] / per_sensor_count[sensor]
                    )
                if train_sensor_total[sensor] > 0:
                    log_payload[f"train_acc_{sensor}"] = train_per_sensor_acc[sensor]
                sensor_cmp = overlap_per_sensor_compare.get(sensor)
                if sensor_cmp is not None:
                    log_payload[f"overlap_count_{sensor}"] = sensor_cmp["count"]
                    log_payload[f"overlap_fused_acc_{sensor}"] = sensor_cmp["fused_acc"]
                    log_payload[f"overlap_base_acc_{sensor}"] = sensor_cmp["base_acc"]
                    log_payload[f"overlap_fused_fpr_{sensor}"] = sensor_cmp["fused_fpr"]
                    log_payload[f"overlap_base_fpr_{sensor}"] = sensor_cmp["base_fpr"]
                    log_payload[f"overlap_fused_recall_{sensor}"] = sensor_cmp["fused_recall"]
                    log_payload[f"overlap_base_recall_{sensor}"] = sensor_cmp["base_recall"]
                    log_payload[f"overlap_fused_auroc_{sensor}"] = sensor_cmp["fused_auroc"]
                    log_payload[f"overlap_base_auroc_{sensor}"] = sensor_cmp["base_auroc"]
                sensor_non_overlap_cmp = non_overlap_per_sensor_compare.get(sensor)
                if sensor_non_overlap_cmp is not None:
                    log_payload[f"non_overlap_count_{sensor}"] = sensor_non_overlap_cmp["count"]
                    log_payload[f"non_overlap_fused_acc_{sensor}"] = sensor_non_overlap_cmp["fused_acc"]
                    log_payload[f"non_overlap_base_acc_{sensor}"] = sensor_non_overlap_cmp["base_acc"]
                    log_payload[f"non_overlap_fused_fpr_{sensor}"] = sensor_non_overlap_cmp["fused_fpr"]
                    log_payload[f"non_overlap_base_fpr_{sensor}"] = sensor_non_overlap_cmp["base_fpr"]
                    log_payload[f"non_overlap_fused_recall_{sensor}"] = sensor_non_overlap_cmp["fused_recall"]
                    log_payload[f"non_overlap_base_recall_{sensor}"] = sensor_non_overlap_cmp["base_recall"]
                    log_payload[f"non_overlap_fused_auroc_{sensor}"] = sensor_non_overlap_cmp["fused_auroc"]
                    log_payload[f"non_overlap_base_auroc_{sensor}"] = sensor_non_overlap_cmp["base_auroc"]
            for sensor in early_stop_sensors:
                log_payload[f"adapter_frozen_{sensor}"] = int(sensor_adapter_frozen[sensor])
                log_payload[f"adapter_best_acc_{sensor}"] = sensor_adapter_best_score[sensor]
                log_payload[f"adapter_bad_epochs_{sensor}"] = sensor_adapter_bad_epochs[sensor]
            if summary_loss_count > 0:
                log_payload["train_loss_summary"] = summary_loss_accum / summary_loss_count
            if overlap_loss_count > 0:
                log_payload["train_loss_overlap"] = overlap_loss_accum / overlap_loss_count
            wandb_run.log(log_payload)

        if args.overlap_report_jsonl and args.overlap_report_jsonl.strip():
            report_payload: Dict[str, Any] = {
                "epoch": int(epoch),
                "overlap_fusion": args.overlap_fusion,
                "overlap_fusion_train": bool(args.overlap_fusion_train),
                "overlap_count": int(overlap_metrics_fused["count"]),
                "non_overlap_count": int(non_overlap_metrics_fused["count"]),
                "overlap_fused": {
                    "acc": overlap_metrics_fused["acc"],
                    "fpr": overlap_metrics_fused["fpr"],
                    "recall": overlap_metrics_fused["recall"],
                    "auroc": overlap_metrics_fused["auroc"],
                },
                "overlap_base": {
                    "acc": overlap_metrics_base["acc"],
                    "fpr": overlap_metrics_base["fpr"],
                    "recall": overlap_metrics_base["recall"],
                    "auroc": overlap_metrics_base["auroc"],
                },
                "non_overlap_fused": {
                    "acc": non_overlap_metrics_fused["acc"],
                    "fpr": non_overlap_metrics_fused["fpr"],
                    "recall": non_overlap_metrics_fused["recall"],
                    "auroc": non_overlap_metrics_fused["auroc"],
                },
                "non_overlap_base": {
                    "acc": non_overlap_metrics_base["acc"],
                    "fpr": non_overlap_metrics_base["fpr"],
                    "recall": non_overlap_metrics_base["recall"],
                    "auroc": non_overlap_metrics_base["auroc"],
                },
                "per_sensor_overlap_compare": overlap_per_sensor_compare,
                "per_sensor_non_overlap_compare": non_overlap_per_sensor_compare,
            }
            _append_jsonl(Path(args.overlap_report_jsonl).expanduser(), report_payload)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
