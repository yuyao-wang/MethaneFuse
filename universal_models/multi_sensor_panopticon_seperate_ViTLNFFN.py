"""Multi-sensor Panopticon training script with sensor-specific embeddings and heads.

This module exposes reusable model components (``MultiSensorPanopticonClassifier``)
*and* a runnable training entry point that ingests mixed-sensor CSVs containing
Sentinel-2, Landsat 8/9, and Sentinel-5P samples. Each sensor owns its own
Panopticon patch embedding and classifier head, while the DinoViT backbone is
shared and updated by the consensus of all datasets in a batch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import math
import os
import shutil
import sys
import warnings
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

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


class DomainSpecificLayerNorm(nn.Module):
    """LayerNorm with per-domain affine parameters and shared statistics."""

    def __init__(self, normalized_shape, num_domains: int, eps: float = 1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        if len(self.normalized_shape) != 1:
            raise ValueError("DomainSpecificLayerNorm currently supports 1D normalized_shape.")
        if num_domains < 1:
            raise ValueError("num_domains must be >= 1")
        self.num_domains = num_domains
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_domains, *self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(num_domains, *self.normalized_shape))
        self._current_domain: Optional[torch.Tensor] = None

    def set_domain_id(self, domain_id: Union[int, torch.Tensor]) -> None:
        if not torch.is_tensor(domain_id):
            domain_id = torch.tensor(domain_id, device=self.weight.device, dtype=torch.long)
        self._current_domain = domain_id

    def _resolve_domain(self, x: torch.Tensor) -> torch.Tensor:
        if self._current_domain is None:
            return torch.zeros((), device=x.device, dtype=torch.long)
        dom = self._current_domain
        if dom.device != x.device:
            dom = dom.to(x.device)
        return dom

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        domain = self._resolve_domain(x)
        if domain.ndim == 0:
            w = self.weight[domain]
            b = self.bias[domain]
            return F.layer_norm(x, self.normalized_shape, w, b, self.eps)

        if domain.shape[0] != x.shape[0]:
            raise ValueError(f"domain_id batch mismatch: expected {x.shape[0]}, got {domain.shape[0]}")
        w = self.weight[domain].unsqueeze(1)  # B x 1 x D
        b = self.bias[domain].unsqueeze(1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * w + b

    def extra_repr(self) -> str:
        return f"num_domains={self.num_domains}, normalized_shape={self.normalized_shape}, eps={self.eps}"


class DomainSpecificLinear(nn.Module):
    """Per-domain Linear layer (only fc1 is domain-specific; fc2 shared).

    A domain id (int or 1D tensor) must be set via ``set_domain_id`` before forward.
    """

    def __init__(self, in_features: int, out_features: int, num_domains: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_domains = num_domains
        self.weight = nn.Parameter(torch.empty(num_domains, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_domains, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self._current_domain: Optional[torch.Tensor] = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_domain_id(self, domain_id: Union[int, torch.Tensor]) -> None:
        if not torch.is_tensor(domain_id):
            domain_id = torch.tensor(domain_id, dtype=torch.long, device=self.weight.device)
        self._current_domain = domain_id

    def _resolve_domain(self, x: torch.Tensor) -> torch.Tensor:
        if self._current_domain is None:
            return torch.zeros((), dtype=torch.long, device=x.device)
        dom = self._current_domain
        if dom.device != x.device:
            dom = dom.to(x.device)
        return dom

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        dom = self._resolve_domain(x)
        if dom.ndim == 0:
            w = self.weight[dom]
            b = None if self.bias is None else self.bias[dom]
            return F.linear(x, w, b)

        # dom is batch-sized tensor: shape [B]
        if dom.shape[0] != x.shape[0]:
            raise ValueError(f"Domain id shape {dom.shape} incompatible with input batch {x.shape}")
        outputs = []
        for i in range(x.shape[0]):
            w = self.weight[dom[i]]
            b = None if self.bias is None else self.bias[dom[i]]
            outputs.append(F.linear(x[i], w, b))
        return torch.stack(outputs, dim=0)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, num_domains={self.num_domains}, bias={self.bias is not None}"


def _convert_backbone_ffn_fc1(backbone: DinoVisionTransformer, num_domains: int):
    """Make MLP fc1 domain-specific while keeping fc2 shared."""
    if num_domains <= 1:
        return backbone
    if hasattr(backbone, "_domain_specific_fc1s"):
        return backbone

    ds_linears: list[DomainSpecificLinear] = []

    def _maybe_replace(mlp: nn.Module):
        fc1 = getattr(mlp, "fc1", None)
        if isinstance(fc1, DomainSpecificLinear):
            return
        if not isinstance(fc1, nn.Linear):
            return
        ds = DomainSpecificLinear(fc1.in_features, fc1.out_features, num_domains, bias=fc1.bias is not None)
        with torch.no_grad():
            ds.weight.copy_(fc1.weight.unsqueeze(0).expand(num_domains, -1, -1))
            if fc1.bias is not None:
                ds.bias.copy_(fc1.bias.unsqueeze(0).expand(num_domains, -1))
        mlp.fc1 = ds
        ds_linears.append(ds)

    for module in backbone.modules():
        mlp = getattr(module, "mlp", None)
        if mlp is not None:
            _maybe_replace(mlp)

    prev_setter = getattr(backbone, "_set_domain_id", None)

    def _set_domain_id(self, domain_id):
        if prev_setter:
            prev_setter(domain_id)
        for ds in ds_linears:
            ds.set_domain_id(domain_id)

    backbone._set_domain_id = MethodType(_set_domain_id, backbone)  # type: ignore[attr-defined]
    backbone._domain_specific_fc1s = ds_linears  # type: ignore[attr-defined]
    return backbone


class MultiSensorPanopticonClassifier(nn.Module):
    """Shared DinoViT backbone with sensor-specific Panopticon PEs and heads."""

    def __init__(
        self,
        *,
        backbone: Optional[DinoVisionTransformer] = None,
        sensors: Sequence[str] = ("s2", "l89", "s5p"),
        num_classes: Mapping[str, int] | int = 2,
        patch_embed_overrides: Optional[Mapping[str, PanopticonPE]] = None,
        head_factory: Optional[Union[HeadFactory, nn.Module]] = None,
        enable_summary_head: bool = True,
        summary_hidden_dim: int = 128,
        summary_dropout: float = 0.1,
        summary_loss_weight: float = 1.0,
        sensor_loss_weights: Optional[Mapping[str, float]] = None,
    ):
        super().__init__()
        self.sensor_order = list(sensors)
        if not self.sensor_order:
            raise ValueError("At least one sensor must be specified")
        self.sensor_to_idx = {sensor: idx for idx, sensor in enumerate(self.sensor_order)}
        num_domains = len(self.sensor_order)

        if backbone is None:
            backbone = _load_backbone(domain_specific_ln=True, num_domains=num_domains)
        if not isinstance(backbone, DinoVisionTransformer):
            raise TypeError("backbone must be a DinoVisionTransformer instance")
        if not hasattr(backbone, "_domain_specific_lns"):
            _convert_backbone_layernorms(backbone, num_domains=num_domains)

        # patch fc1 to be sensor-specific (domain-specific) while keeping fc2 shared
        _convert_backbone_ffn_fc1(backbone, num_domains=num_domains)

        self.backbone = backbone

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
        self.sensor_loss_weights = {sensor: 1.0 for sensor in self.sensor_order}
        if sensor_loss_weights is not None:
            for sensor_name, weight in sensor_loss_weights.items():
                if sensor_name in self.sensor_loss_weights:
                    self.sensor_loss_weights[sensor_name] = float(weight)

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
                if hasattr(self.backbone, "_set_domain_id"):
                    domain_id = torch.tensor(
                        self.sensor_to_idx[sensor_name],
                        device=sensor_batch.indices.device,
                        dtype=torch.long,
                    )
                    self.backbone._set_domain_id(domain_id)  # type: ignore[attr-defined]
                feats = self.backbone(sensor_batch.x_dict, is_training=True)
            cls_token = feats["x_norm_clstoken"]
            head_logits = [self.heads[name](cls_token) for name in self.sensor_order]
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
            weighted_loss = loss * self.sensor_loss_weights.get(sensor_name, 1.0)
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
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
        embed_dim = getattr(self.backbone, "embed_dim", outputs[next(iter(outputs))]["cls_token"].shape[-1])
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
#  Dataset helpers (mixed temporal CSV + S5P NPZ support)
# --------------------------------------------------------------------------------------

class StaticAnchoredCache:
    def __init__(self, cache_dir: str, min_free_gb: float = 10.0, max_gb: Optional[float] = None):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_free_bytes = min_free_gb * (1024**3)
        self.max_bytes = None if max_gb in (None, 0) else max_gb * (1024**3)

    def _get_free_space(self) -> int:
        return shutil.disk_usage(self.cache_dir).free

    def _get_cache_size(self) -> int:
        total = 0
        for root, _, files in os.walk(self.cache_dir):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except FileNotFoundError:
                    continue
        return total

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
        if self.max_bytes is not None:
            try:
                file_size = Path(original).stat().st_size
            except (FileNotFoundError, OSError):
                file_size = 0
            if self._get_cache_size() + file_size > self.max_bytes:
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


# --------------------------------------------------------------------------------------
#  Backbone patching helpers (Domain-specific LayerNorm)
# --------------------------------------------------------------------------------------

def _convert_backbone_layernorms(backbone: DinoVisionTransformer, num_domains: int):
    """Replace backbone nn.LayerNorm modules with domain-specific variants."""
    if hasattr(backbone, "_domain_specific_lns"):
        return backbone

    domain_lns: list[DomainSpecificLayerNorm] = []

    def _maybe_replace(module: nn.Module, name: str):
        child = getattr(module, name, None)
        if not isinstance(child, nn.LayerNorm):
            return
        dsl = DomainSpecificLayerNorm(child.normalized_shape, num_domains=num_domains, eps=child.eps)
        with torch.no_grad():
            dsl.weight.copy_(child.weight.unsqueeze(0).expand(num_domains, -1))
            dsl.bias.copy_(child.bias.unsqueeze(0).expand(num_domains, -1))
        setattr(module, name, dsl)
        domain_lns.append(dsl)

    def _recurse(module: nn.Module):
        for field in ("norm", "norm1", "norm2"):
            if hasattr(module, field):
                _maybe_replace(module, field)
        for _name, child in module.named_children():
            _recurse(child)

    if num_domains > 1:
        _recurse(backbone)
    backbone._domain_specific_lns = domain_lns  # type: ignore[attr-defined]

    prev_setter = getattr(backbone, "_set_domain_id", None)

    def _set_domain_id(self, domain_id):
        if prev_setter:
            prev_setter(domain_id)
        if not domain_lns:
            return
        if not torch.is_tensor(domain_id):
            domain_id_tensor = torch.tensor(domain_id, device=self.pos_embed.device, dtype=torch.long)
        else:
            domain_id_tensor = domain_id.to(self.pos_embed.device)
        for ln in domain_lns:
            ln.set_domain_id(domain_id_tensor)

    backbone._set_domain_id = MethodType(_set_domain_id, backbone)  # type: ignore[attr-defined]
    return backbone


def _expand_state_dict_for_domain_lns(state: MutableMapping[str, torch.Tensor], num_domains: int):
    """Expand LayerNorm params from [D] to [K,D] for domain-specific LayerNorm loads."""
    if num_domains <= 1:
        return
    for key, tensor in list(state.items()):
        if tensor.ndim != 1 or not key.endswith(("weight", "bias")):
            continue
        module_key = key.rsplit(".", 1)[0]
        if module_key.split(".")[-1] not in {"norm", "norm1", "norm2"}:
            continue
        state[key] = tensor.unsqueeze(0).expand(num_domains, -1).clone()


class TriSensorTemporalCsvDataset(S2TemporalCsvDataset):
    """Temporal dataset that mixes Sentinel-2, Landsat 8/9, and Sentinel-5P samples."""

    def __init__(
        self,
        *args,
        local_file_cache: Optional[StaticAnchoredCache] = None,
        s5p_data_key: Optional[str] = None,
        s5p_chn_ids_key: Optional[str] = "chn_ids",
        s5p_channels_last: bool = False,
        align_l89_to_s2: bool = False,
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
        super().__init__(*args, **kwargs)

        self._validate_sensor_column()
        self.sensor_configs = self._build_sensor_configs()
        self._s5p_mean, self._s5p_std = _compute_mean_std(S5P_PRECOMPUTED_STATS)

    def _validate_sensor_column(self) -> None:
        if "sensor" not in self.df.columns:
            raise ValueError("CSV must contain a 'sensor' column with values in {'s2','l89','s5p'}")

    def _build_sensor_configs(self) -> Dict[str, Dict[str, torch.Tensor]]:
        configs = {
            "l89": {
                "ds_cfg": load_ds_cfg("landsat89_7band"),
                "normalize_stats": L89_PRECOMPUTED_STATS,
            },
            "s2": {
                "ds_cfg": load_ds_cfg("s2_12band"),
                "normalize_stats": S2_PRECOMPUTED_STATS,
            },
        }

        # Inject channel ids and optional scaling.
        for name, cfg in configs.items():
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
        original = (self.ds_cfg, self.normalize_stats, self.chn_ids, self._mean, self._std)
        self.ds_cfg = cfg["ds_cfg"]
        self.normalize_stats = cfg["normalize_stats"]
        self.chn_ids = cfg["chn_ids"]
        self._mean = cfg.get("mean_tensor")
        self._std = cfg.get("std_tensor")
        try:
            yield
        finally:
            self.ds_cfg, self.normalize_stats, self.chn_ids, self._mean, self._std = original

    def _load_image(self, path: str, *, column_name: str, sample_id: int):
        row = self.df.iloc[sample_id]
        sensor = row.get("sensor")
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
                sensor = row.get("sensor")
                if sensor is None:
                    raise ValueError(f"Row {idx} is missing 'sensor' value")
                label = int(row[self.label_column])

                if sensor == "s5p":
                    x_dict = self._load_s5p_sample(row, self.path_columns[0])
                    return [x_dict], label, sensor

                if sensor not in self.sensor_configs:
                    raise ValueError(f"Sample {idx} has unknown sensor '{sensor}'")

                x_list = [self._load_temporal_frame(row, col, sensor, idx) for col in self.path_columns]
                return x_list, label, sensor
            except _SkipSample as exc:
                last_exc = exc
                attempts += 1
                if attempts >= self.max_retries:
                    raise RuntimeError(f"Exceeded {self.max_retries} retries for temporal index {idx}") from exc
                idx = np.random.randint(0, len(self.df))
        raise RuntimeError("Unreachable") from last_exc

    def _load_temporal_frame(self, row, column_name: str, sensor: str, sample_id: int) -> Dict[str, torch.Tensor]:
        path = row[column_name]
        img = self._load_image(path, column_name=column_name, sample_id=sample_id)
        if sensor == "l89" and self._align_l89_to_s2:
            img = self._pad_l89_to_s2(img)
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
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
        path = row.get(column_name)
        if not isinstance(path, str) or not path:
            raise ValueError(f"S5P sample missing value in '{column_name}'")
        path = self._maybe_cache_path(path)
        np_obj = np.load(path, allow_pickle=False)
        chn_ids = None
        try:
            if isinstance(np_obj, np.lib.npyio.NpzFile):
                img_np = self._extract_npz_array(np_obj, path)
                if self._s5p_chn_ids_key is not None and self._s5p_chn_ids_key in np_obj:
                    chn_ids = torch.as_tensor(np_obj[self._s5p_chn_ids_key])
            else:
                img_np = np.array(np_obj)
        finally:
            if isinstance(np_obj, np.lib.npyio.NpzFile):
                np_obj.close()
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, 0)
        elif img_np.ndim == 3 and self._s5p_channels_last:
            img_np = np.transpose(img_np, (2, 0, 1))
        img = torch.from_numpy(img_np).to(dtype=torch.float32)
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        if img.shape[0] != self._s5p_mean.shape[0]:
            repeat = self._s5p_mean.shape[0] // max(1, img.shape[0])
            img = img.repeat(repeat, 1, 1)
        img = (img - self._s5p_mean) / self._s5p_std
        if self.pad_to_multiple is not None:
            img = self._pad_to_multiple(img, self.pad_to_multiple)
        if chn_ids is None:
            chn_ids = torch.zeros((img.shape[0], 1), dtype=torch.float32)
        if chn_ids.ndim == 1:
            chn_ids = chn_ids.unsqueeze(-1)
        x_dict = dict(imgs=img, chn_ids=chn_ids)
        if self.transform_each is not None:
            x_dict = self.transform_each(x_dict)
        return x_dict

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
        x_list, label, sensor = self.base_ds[idx]
        imgs = torch.cat([x["imgs"] for x in x_list], dim=0)
        chn_ids = torch.cat([x["chn_ids"] for x in x_list], dim=0)
        x_dict = dict(imgs=imgs, chn_ids=chn_ids)
        return x_dict, label, sensor


def custom_collate_fn(batch):
    x_dicts, labels, sensors = zip(*batch)
    batch_size = len(x_dicts)
    max_channels = max(x["imgs"].shape[0] for x in x_dicts)
    max_h = max(x["imgs"].shape[1] for x in x_dicts)
    max_w = max(x["imgs"].shape[2] for x in x_dicts)
    chn_tail_shape = x_dicts[0]["chn_ids"].shape[1:]

    batched_imgs = x_dicts[0]["imgs"].new_zeros((batch_size, max_channels, max_h, max_w))
    batched_chn_ids = x_dicts[0]["chn_ids"].new_zeros((batch_size, max_channels, *chn_tail_shape))
    for i, x_dict in enumerate(x_dicts):
        img = x_dict["imgs"]
        chn_ids = x_dict["chn_ids"]
        c, h, w = img.shape
        batched_imgs[i, :c, :h, :w] = img
        batched_chn_ids[i, :c] = chn_ids
    batched_x_dict = {"imgs": batched_imgs, "chn_ids": batched_chn_ids}
    return batched_x_dict, torch.as_tensor(labels, dtype=torch.long), list(sensors)


def _resolve_base_dataset(dataset: Dataset) -> Dataset:
    base = dataset
    while hasattr(base, "base_ds"):
        base = base.base_ds  # type: ignore[attr-defined]
    return base


def _gather_sensor_indices(dataset: Dataset, sensors: Sequence[str]) -> Dict[str, Sequence[int]]:
    base_ds = _resolve_base_dataset(dataset)
    df = getattr(base_ds, "df", None)
    if df is None or "sensor" not in df.columns:
        raise ValueError("Dataset must expose a DataFrame with a 'sensor' column to split by sensor.")
    idx_map = {sensor: [] for sensor in sensors}
    for idx, sensor_name in enumerate(df["sensor"].tolist()):
        if sensor_name in idx_map:
            idx_map[sensor_name].append(idx)
    return idx_map


def build_sensor_dataloaders(
    dataset: Dataset,
    sensors: Sequence[str],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
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
        loader_kwargs = {}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            loader_kwargs["prefetch_factor"] = prefetch_factor
        loaders[sensor_name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn,
            **loader_kwargs,
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
    persistent_workers: bool,
    prefetch_factor: int,
    shuffle: bool,
    oversample_to_max: bool = True,
    sensor_epoch_ratio: Optional[Mapping[str, float]] = None,
) -> Tuple[DataLoader, int]:
    """Create a single DataLoader that mixes sensors and oversamples minority ones.

    - If ``oversample_to_max`` is True (default), each sensor is repeated until it
      reaches the size of the largest sensor, balancing class counts.
    - ``sensor_epoch_ratio`` can downsample/upsample each sensor per epoch
      (e.g. l89=0.35 means using about 35% of the L89 epoch budget).
    - Otherwise, uses the natural counts.
    Returns (loader, total_steps_per_epoch).
    """

    idx_map = _gather_sensor_indices(dataset, sensors)
    counts = {s: len(v) for s, v in idx_map.items() if v}
    if not counts:
        raise ValueError("No samples found for the configured sensors.")

    ratio_map = {sensor: 1.0 for sensor in sensors}
    if sensor_epoch_ratio:
        for sensor_name, value in sensor_epoch_ratio.items():
            if sensor_name in ratio_map:
                ratio_map[sensor_name] = max(0.0, float(value))

    target = max(counts.values()) if oversample_to_max else None
    balanced_indices: list[int] = []
    for sensor, indices in idx_map.items():
        if not indices:
            continue
        base_target = target if target is not None else len(indices)
        desired = int(math.ceil(base_target * ratio_map.get(sensor, 1.0)))
        if desired <= 0:
            continue
        repeat = math.ceil(desired / len(indices))
        expanded = (indices * repeat)[:desired]
        balanced_indices.extend(expanded)

    if not balanced_indices:
        raise ValueError("No samples selected for the mixed dataloader. Check sensor_epoch_ratio settings.")

    balanced_subset = Subset(dataset, balanced_indices)
    sampler = None
    if shuffle:
        sampler = RandomSampler(balanced_subset)
    steps = math.ceil(len(balanced_indices) / batch_size)
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(
        balanced_subset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
        **loader_kwargs,
    )
    return loader, steps


# --------------------------------------------------------------------------------------
#  Training utilities
# --------------------------------------------------------------------------------------

def _load_backbone(
    weights_path: Optional[str] = None,
    *,
    strict: bool = True,
    domain_specific_ln: bool = False,
    num_domains: int = 1,
) -> DinoVisionTransformer:
    from hubconf import _panopticon_vitb14

    backbone = _panopticon_vitb14()
    if domain_specific_ln:
        _convert_backbone_layernorms(backbone, num_domains=num_domains)
    if weights_path in (None, "", "none", "scratch", "random"):
        return backbone
    ckpt_path = Path(weights_path)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, Mapping) and "backbone" in state:
        state = state["backbone"]
    if domain_specific_ln:
        _expand_state_dict_for_domain_lns(state, num_domains=num_domains)
    backbone.load_state_dict(state, strict=strict)
    return backbone


def _index_batch(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, idx)


def _slice_x_dict(x_dict: MutableMapping[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: _index_batch(v, idx) for k, v in x_dict.items() if isinstance(v, torch.Tensor)}


def recursive_to_device(x, device, *, non_blocking: bool = False):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: recursive_to_device(v, device, non_blocking=non_blocking) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [recursive_to_device(v, device, non_blocking=non_blocking) for v in x]
        return type(x)(t)
    return x


def set_trainable(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def count_non_finite_params(module: nn.Module) -> int:
    non_finite = 0
    for p in module.parameters():
        if p is None:
            continue
        non_finite += int((~torch.isfinite(p.detach())).sum().item())
    return non_finite


def init_wandb(args):
    if not args.use_wandb:
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:
        warnings.warn(f"Failed to import wandb ({exc}); continuing without Weights & Biases logging.")
        return None

    wandb_init = getattr(wandb, "init", None)
    if not callable(wandb_init):
        module_file = getattr(wandb, "__file__", "<unknown>")
        warnings.warn(
            f"Imported module 'wandb' from {module_file}, but it has no callable init(). "
            "This usually means a local module is shadowing the W&B package. "
            "Continuing without Weights & Biases logging."
        )
        return None

    try:
        return wandb_init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    except Exception as exc:
        warnings.warn(f"wandb.init failed ({exc}); continuing without Weights & Biases logging.")
        return None


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


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float("nan") if denominator == 0 else float(numerator / denominator)


def _binary_auroc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    n = labels.size
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


def build_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "noam":
        warmup_steps = max(args.warmup_steps, 1)
        noam_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")


def parse_sensor_float_map(
    raw: Optional[str],
    sensors: Sequence[str],
    *,
    default: float = 1.0,
    map_name: str,
    min_value: float = 0.0,
) -> Dict[str, float]:
    values = {sensor: float(default) for sensor in sensors}
    if raw is None:
        return values
    text = raw.strip()
    if not text:
        return values
    for piece in text.split(","):
        token = piece.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid {map_name} entry '{token}'. Expected format sensor=value.")
        sensor_name, value_str = token.split("=", 1)
        sensor_name = sensor_name.strip()
        if sensor_name not in values:
            raise ValueError(f"Unknown sensor '{sensor_name}' in {map_name}. Valid sensors: {list(sensors)}")
        try:
            value = float(value_str)
        except ValueError as exc:
            raise ValueError(f"Invalid float value '{value_str}' for sensor '{sensor_name}' in {map_name}.") from exc
        if value < min_value:
            raise ValueError(f"{map_name} for sensor '{sensor_name}' must be >= {min_value}, got {value}.")
        values[sensor_name] = value
    return values


def gather_head_parameters(model: MultiSensorPanopticonClassifier) -> Sequence[nn.Parameter]:
    shared_patch = model.backbone.patch_embed
    extra_patch_params = []
    for sensor, module in model.sensor_patch_embeds.items():
        if module is shared_patch:
            continue
        extra_patch_params.extend(list(module.parameters()))
    summary_params = list(model.summary_head.parameters()) if model.summary_head is not None else []
    head_params = list(model.heads.parameters()) + summary_params + extra_patch_params
    return head_params


# --------------------------------------------------------------------------------------
#  CLI / training loop
# --------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-sensor Panopticon finetuning (S2/L89/S5P).")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--weights", default="weights/panopticon_vitb14_teacher.pth")
    parser.add_argument("--t0_col", default="path_t0")
    parser.add_argument("--t90_col", default="path_t90")
    parser.add_argument("--t360_col", default="path_t360")
    parser.add_argument("--s5p_data_key", default=None)
    parser.add_argument("--s5p_chn_ids_key", default="chn_ids")
    parser.add_argument("--s5p_channels_last", action="store_true")
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
    parser.add_argument("--amp", action="store_true", dest="amp")
    parser.add_argument("--disable_amp", action="store_false", dest="amp")
    parser.set_defaults(amp=True)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="panopticon-multisensor")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--local_cache_dir", default=None)
    parser.add_argument("--local_cache_min_free_gb", type=float, default=5.0)
    parser.add_argument("--local_cache_max_gb", type=float, default=None)
    parser.add_argument("--local_cache_warmup", action="store_true")
    parser.add_argument("--local_cache_workers", type=int, default=8)
    parser.add_argument("--oversample_minority", action="store_true", dest="oversample_minority")
    parser.add_argument("--disable_oversample_minority", action="store_false", dest="oversample_minority")
    parser.set_defaults(oversample_minority=True)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true", dest="persistent_workers")
    parser.add_argument("--disable_persistent_workers", action="store_false", dest="persistent_workers")
    parser.set_defaults(persistent_workers=True)
    parser.add_argument("--non_blocking_transfer", action="store_true", dest="non_blocking_transfer")
    parser.add_argument("--disable_non_blocking_transfer", action="store_false", dest="non_blocking_transfer")
    parser.set_defaults(non_blocking_transfer=True)
    parser.add_argument("--enable_tf32", action="store_true", dest="enable_tf32")
    parser.add_argument("--disable_tf32", action="store_false", dest="enable_tf32")
    parser.set_defaults(enable_tf32=True)
    parser.add_argument("--cudnn_benchmark", action="store_true", dest="cudnn_benchmark")
    parser.add_argument("--disable_cudnn_benchmark", action="store_false", dest="cudnn_benchmark")
    parser.set_defaults(cudnn_benchmark=True)
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--fused_optimizer", action="store_true", dest="fused_optimizer")
    parser.add_argument("--disable_fused_optimizer", action="store_false", dest="fused_optimizer")
    parser.set_defaults(fused_optimizer=False)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--compile_mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    parser.add_argument("--sensor_switch_interval", type=int, default=100)
    parser.add_argument("--summary_head", action="store_true", dest="summary_head")
    parser.add_argument("--disable_summary_head", action="store_false", dest="summary_head")
    parser.set_defaults(summary_head=True)
    parser.add_argument("--summary_hidden_dim", type=int, default=128)
    parser.add_argument("--summary_dropout", type=float, default=0.1)
    parser.add_argument("--summary_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--train_sensor_epoch_ratio",
        type=str,
        default="",
        help="Comma-separated per-sensor ratio for training samples per epoch, e.g. s2=1.0,l89=0.35,s5p=1.0",
    )
    parser.add_argument(
        "--sensor_loss_weights",
        type=str,
        default="",
        help="Comma-separated per-sensor loss weights, e.g. s2=1.0,l89=0.6,s5p=1.0",
    )
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = args.enable_tf32
        torch.backends.cudnn.allow_tf32 = args.enable_tf32
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)

    path_columns = (args.t0_col, args.t90_col, args.t360_col)
    cache_obj = None
    if args.local_cache_dir:
        cache_obj = StaticAnchoredCache(
            args.local_cache_dir,
            min_free_gb=args.local_cache_min_free_gb,
            max_gb=args.local_cache_max_gb,
        )

    base_train_ds = TriSensorTemporalCsvDataset(
        csv_path=args.train_csv,
        path_columns=path_columns,
        local_file_cache=cache_obj,
        s5p_data_key=args.s5p_data_key,
        s5p_chn_ids_key=args.s5p_chn_ids_key,
        s5p_channels_last=args.s5p_channels_last,
        align_l89_to_s2=args.align_l89_to_s2,
        pad_to_multiple=14,
    )
    base_test_ds = TriSensorTemporalCsvDataset(
        csv_path=args.test_csv,
        path_columns=path_columns,
        local_file_cache=cache_obj,
        s5p_data_key=args.s5p_data_key,
        s5p_chn_ids_key=args.s5p_chn_ids_key,
        s5p_channels_last=args.s5p_channels_last,
        align_l89_to_s2=args.align_l89_to_s2,
        pad_to_multiple=14,
    )

    if args.local_cache_warmup and cache_obj:
        all_paths = []
        for col in path_columns:
            if col in base_train_ds.df.columns:
                all_paths.extend(base_train_ds.df[col].dropna().astype(str).tolist())
        cache_obj.warm_up(all_paths, max_workers=args.local_cache_workers)

    train_ds = ConcatTemporalDataset(base_train_ds)
    test_ds = ConcatTemporalDataset(base_test_ds)

    default_sensors = ("s2", "l89", "s5p")
    train_sensor_epoch_ratio = parse_sensor_float_map(
        args.train_sensor_epoch_ratio,
        default_sensors,
        default=1.0,
        map_name="train_sensor_epoch_ratio",
        min_value=0.0,
    )
    sensor_loss_weights = parse_sensor_float_map(
        args.sensor_loss_weights,
        default_sensors,
        default=1.0,
        map_name="sensor_loss_weights",
        min_value=0.0,
    )
    core_model = MultiSensorPanopticonClassifier(
        backbone=_load_backbone(args.weights, domain_specific_ln=True, num_domains=len(default_sensors)),
        sensors=default_sensors,
        enable_summary_head=args.summary_head,
        summary_hidden_dim=args.summary_hidden_dim,
        summary_dropout=args.summary_dropout,
        summary_loss_weight=args.summary_loss_weight,
        sensor_loss_weights=sensor_loss_weights,
    ).to(device)
    init_non_finite = count_non_finite_params(core_model)
    if init_non_finite > 0:
        raise RuntimeError(f"Model has {init_non_finite} non-finite parameters after initialization/load.")
    model: nn.Module = core_model
    use_data_parallel = args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1
    if use_data_parallel:
        print(f"Enabling DataParallel across {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(core_model)
    elif args.data_parallel:
        print("DataParallel requested but insufficient CUDA devices; running single-device.", flush=True)
    if args.torch_compile and not use_data_parallel and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"Enabled torch.compile(mode={args.compile_mode})", flush=True)
        except Exception as exc:
            warnings.warn(f"Failed to enable torch.compile; continuing without it. Error: {exc}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    head_params = gather_head_parameters(core_model)
    param_groups = [
        {"params": head_params, "lr": args.head_lr},
    ]
    if args.train_backbone:
        param_groups.insert(0, {"params": core_model.backbone.parameters(), "lr": args.backbone_lr})

    adam_kwargs = {"weight_decay": args.weight_decay, "betas": (args.momentum, 0.999)}
    using_fused_adam = False
    if args.fused_optimizer and device.type == "cuda":
        adam_sig = inspect.signature(torch.optim.Adam)
        if "fused" in adam_sig.parameters:
            adam_kwargs["fused"] = True
            using_fused_adam = True
    optimizer = torch.optim.Adam(param_groups, **adam_kwargs)
    print(f"Optimizer: Adam(fused={using_fused_adam})", flush=True)
    scheduler = build_scheduler(args, optimizer)
    use_amp = device.type == "cuda" and args.amp
    scaler = GradScaler(enabled=use_amp)

    run_name = args.wandb_run_name or default_run_name(args)
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    latest_path = ckpt_dir / "ckpt_latest.pth"
    best_path = ckpt_dir / "ckpt_best_test.pth"

    start_epoch = 1
    global_step = 0
    best_train_acc = 0.0
    best_test_acc = float("-inf")

    if args.resume:
        start_epoch, global_step, best_train_acc, best_test_acc = try_resume(
            latest_path, core_model, optimizer, scheduler, scaler, device
        )
        resumed_non_finite = count_non_finite_params(core_model)
        if resumed_non_finite > 0:
            raise RuntimeError(f"Model has {resumed_non_finite} non-finite parameters after resume.")

    wandb_run = init_wandb(args)

    sensors_list = core_model.sensor_order
    pin_memory = device.type == "cuda"
    persistent_workers = args.persistent_workers and args.num_workers > 0
    # Single mixed loader with optional oversampling so that minority sensors are repeated
    # and batches contain mixed sensors.
    train_loader, train_steps = build_balanced_mixed_dataloader(
        train_ds,
        sensors_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=True,
        oversample_to_max=args.oversample_minority,
        sensor_epoch_ratio=train_sensor_epoch_ratio,
    )
    test_loader, test_steps = build_balanced_mixed_dataloader(
        test_ds,
        sensors_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=False,
        oversample_to_max=False,
    )

    print(
        f"Using device={device}, train_samples={len(train_ds)}, test_samples={len(test_ds)}, "
        f"sensors={sensors_list}, train_backbone={args.train_backbone}, "
        f"train_sensor_epoch_ratio={train_sensor_epoch_ratio}, sensor_loss_weights={sensor_loss_weights}",
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
        if core_model.summary_head is not None:
            core_model.summary_head.train()
        for sensor in core_model.sensor_patch_embeds.values():
            sensor.train()

        total_loss = 0.0
        total = 0
        correct = 0
        per_sensor_loss_accum = {sensor: 0.0 for sensor in sensors_list}
        per_sensor_count = {sensor: 0 for sensor in sensors_list}
        summary_loss_accum = 0.0
        summary_loss_count = 0
        train_sensor_correct = {sensor: 0 for sensor in sensors_list}
        train_sensor_total = {sensor: 0 for sensor in sensors_list}

        step_idx = 0
        for x_dict, labels, sensors in train_loader:
            step_idx += 1
            labels = labels.to(device, non_blocking=args.non_blocking_transfer)
            x_dict = recursive_to_device(x_dict, device, non_blocking=args.non_blocking_transfer)
            sensor_arg: Union[Sequence[str], torch.Tensor]
            if use_data_parallel:
                sensor_arg = core_model.encode_sensors(sensors, device=device)
            else:
                sensor_arg = sensors
            with autocast(enabled=use_amp):
                outputs = model(x_dict, sensors=sensor_arg)
                loss, per_sensor_losses = core_model.loss_from_outputs(outputs, labels, sensors, criterion)
            if not torch.isfinite(loss):
                imgs = x_dict["imgs"]
                msg = [
                    f"Non-finite train loss at epoch={epoch} step={step_idx}",
                    f"imgs_finite={bool(torch.isfinite(imgs).all().item())}",
                    f"imgs_abs_max={float(imgs.detach().abs().max().item()):.3e}",
                ]
                for sensor_name in sensors_list:
                    sensor_out = outputs.get(sensor_name)
                    if not isinstance(sensor_out, dict):
                        continue
                    logits = sensor_out["logits"]
                    msg.append(
                        f"{sensor_name}_logits_finite={bool(torch.isfinite(logits).all().item())}"
                    )
                    msg.append(f"{sensor_name}_logits_abs_max={float(logits.detach().abs().max().item()):.3e}")
                print(" ".join(msg), flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
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

            batch = labels.size(0)
            total_loss += loss.item() * batch
            total += batch
            logits = outputs.get("summary_logits")
            if logits is None:
                logits = outputs.get("merged_logits")
            if logits is None:
                logits = torch.zeros((batch, core_model.heads[sensors_list[0]].out_features), device=device)
                for sensor_name, sensor_out in outputs.items():
                    if not isinstance(sensor_out, dict):
                        continue
                    logits.index_copy_(0, sensor_out["indices"], sensor_out["logits"])
            preds = logits.argmax(dim=1)
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
        with torch.no_grad():
            for step, (x_dict, labels, sensors) in enumerate(test_loader, 1):
                labels = labels.to(device, non_blocking=args.non_blocking_transfer)
                x_dict = recursive_to_device(x_dict, device, non_blocking=args.non_blocking_transfer)
                if use_data_parallel:
                    sensor_arg = core_model.encode_sensors(sensors, device=device)
                else:
                    sensor_arg = sensors
                with autocast(enabled=use_amp):
                    outputs = model(x_dict, sensors=sensor_arg)
                    loss, _ = core_model.loss_from_outputs(outputs, labels, sensors, criterion)
                if not torch.isfinite(loss):
                    print(f"Non-finite eval loss at epoch={epoch} step={step}; skipping batch.", flush=True)
                    continue
                batch = labels.size(0)
                test_loss_total += loss.item() * batch
                total_eval += batch
                logits = outputs.get("summary_logits")
                if logits is None:
                    logits = outputs.get("merged_logits")
                if logits is None:
                    logits = torch.zeros((batch, core_model.heads[sensors_list[0]].out_features), device=device)
                    for sensor_name, sensor_out in outputs.items():
                        if not isinstance(sensor_out, dict):
                            continue
                        logits.index_copy_(0, sensor_out["indices"], sensor_out["logits"])
                preds = logits.argmax(dim=1)
                correct_eval += (preds == labels).sum().item()
                eval_labels.extend(labels.detach().to("cpu").tolist())
                eval_preds.extend(preds.detach().to("cpu").tolist())
                if logits.shape[-1] == 2:
                    pos_scores = torch.softmax(logits, dim=1)[:, 1]
                    eval_pos_scores.extend(pos_scores.detach().to("cpu").tolist())
                for i, sensor_type in enumerate(sensors):
                    sensor_total[sensor_type] += 1
                    if preds[i] == labels[i]:
                        sensor_correct[sensor_type] += 1
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

        train_acc_str = " ".join(f"train_acc_{s}={train_per_sensor_acc[s]:.4f}" for s in sensors_list)
        test_acc_str = " ".join(f"test_acc_{s}={per_sensor_acc[s]:.4f}" for s in sensors_list)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"test_fpr={test_fpr:.4f} test_recall={test_recall:.4f} test_auroc={test_auroc:.4f} "
            f"{train_acc_str} {test_acc_str}",
            flush=True,
        )

        save_checkpoint(
            latest_path,
            epoch,
            global_step,
            core_model,
            optimizer,
            scheduler,
            scaler if use_amp else None,
            train_acc,
            test_acc,
            args,
        )
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(
                best_path,
                epoch,
                global_step,
                core_model,
                optimizer,
                scheduler,
                scaler if use_amp else None,
                train_acc,
                test_acc,
                args,
            )

        best_train_acc = max(best_train_acc, train_acc)
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
                "FPR": test_fpr,
                "Recall": test_recall,
                "AUROC": test_auroc,
            }
            for sensor in sensors_list:
                log_payload[f"test_acc_{sensor}"] = per_sensor_acc[sensor]
                if per_sensor_count[sensor] > 0:
                    log_payload[f"train_loss_{sensor}"] = (
                        per_sensor_loss_accum[sensor] / per_sensor_count[sensor]
                    )
                if train_sensor_total[sensor] > 0:
                    log_payload[f"train_acc_{sensor}"] = train_per_sensor_acc[sensor]
            if summary_loss_count > 0:
                log_payload["train_loss_summary"] = summary_loss_accum / summary_loss_count
            wandb_run.log(log_payload)
            if math.isfinite(test_fpr):
                wandb_run.summary["FPR"] = test_fpr
            if math.isfinite(test_recall):
                wandb_run.summary["Recall"] = test_recall
            if math.isfinite(test_auroc):
                wandb_run.summary["AUROC"] = test_auroc

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
