#!/usr/bin/env python3
"""Small, leakage-safe RCTP mechanism screen on six-visit Landsat 8/9.

This is deliberately not a full foundation-model pretraining implementation.
It asks one narrow question before spending a large compute budget:

    Can a frozen Panopticon encoder plus a zero-initialized,
    response-conditioned residual adapter distinguish an exact synthetic
    methane counterfactual from energy-matched spectral nuisances?

For every reference row, exactly one usable visit is selected (including
history visits).  Four images share the identical scene and plume field:

* clean reference;
* methane-like L89 SWIR absorption;
* energy-matched achromatic attenuation;
* energy-matched wavelength-shuffled attenuation.

Only the selected visit is read online.  The other clean visit features come
from the existing frozen-CLS cache.  The clean visit is nevertheless encoded
online together with all three perturbations, so feature differences are an
exact same-checkpoint pair.  The Panopticon encoder is always frozen; only the
small probe is optimized.

The command accepts train and validation artifacts only and rejects test-like
paths.  Validation AP selects the checkpoint.  There is no test argument or
test evaluation code path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("XFORMERS_DISABLED", "1")

from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as cache_runner,
)
from thirdparty.dinov2.data.datasets.s2_csv import S2CsvDataset  # noqa: E402
from Upgraded_dataset.dino_classifier_head_l89_temporal_satmae import (  # noqa: E402
    load_backbone,
)


SCRIPT_VERSION = "rctp-l89-mechanism-screen-v1"
VARIANT_NAMES = ("methane", "achromatic", "wavelength_shuffled")
METHANE_VARIANT_INDEX = 0

# A deliberately explicit *screening* approximation, not a formal L8/L9 SRF
# renderer.  The first five reflectance bands receive no methane absorption,
# SWIR1 receives a weak response, and SWIR2 receives the strongest response.
# Formal RCTP must replace this vector with versioned SRF integration.
DEFAULT_L89_CH4_RESPONSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 1.0)
DEFAULT_WAVELENGTH_SHUFFLE = (3, 4, 5, 6, 0, 1, 2)


@dataclass(frozen=True)
class RendererConfig:
    response: tuple[float, ...] = DEFAULT_L89_CH4_RESPONSE
    wavelength_shuffle: tuple[int, ...] = DEFAULT_WAVELENGTH_SHUFFLE
    min_peak_drop: float = 0.01
    max_peak_drop: float = 0.08
    plume_min_length: float = 0.16
    plume_max_length: float = 0.42
    plume_min_width: float = 0.025
    plume_max_width: float = 0.075
    renderer_version: str = "elliptical-filament-multiplicative-absorption-v1"

    def validate(self, channels: int) -> None:
        if len(self.response) != channels:
            raise ValueError(
                f"Response length {len(self.response)} != channels {channels}."
            )
        if len(self.wavelength_shuffle) != channels:
            raise ValueError("wavelength_shuffle length does not match channels.")
        if sorted(self.wavelength_shuffle) != list(range(channels)):
            raise ValueError("wavelength_shuffle must be a permutation.")
        if tuple(self.wavelength_shuffle) == tuple(range(channels)):
            raise ValueError("wavelength_shuffle cannot be the identity.")
        if min(self.response) < 0 or max(self.response) <= 0:
            raise ValueError("Response values must be nonnegative and nonzero.")
        if not 0 < self.min_peak_drop <= self.max_peak_drop < 1:
            raise ValueError("Peak drop bounds must satisfy 0 < min <= max < 1.")
        if not (
            0 < self.plume_min_length <= self.plume_max_length < 1
            and 0 < self.plume_min_width <= self.plume_max_width < 1
        ):
            raise ValueError("Plume geometry bounds must lie in (0, 1).")


@dataclass(frozen=True)
class PlanEntry:
    row_index: int
    visit_index: int
    renderer_seed: int
    peak_drop: float


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device} requested but CUDA is unavailable.")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    return device


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def stable_int_seed(*parts: Any) -> int:
    encoded = cache_runner.canonical_json_bytes([str(part) for part in parts])
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & 0x7FFFFFFF


def deterministic_reference_plan(
    payload: Mapping[str, Any],
    *,
    seed: int,
    epoch: int,
    max_rows: int,
    reference_label: str,
    renderer: RendererConfig,
) -> list[PlanEntry]:
    """Choose rows and balance injection roles without reading any imagery."""

    labels = payload["labels"].long()
    unique = payload["unique_mask"].bool()
    if labels.ndim != 1 or unique.ndim != 2 or len(labels) != len(unique):
        raise ValueError("Cache labels/unique_mask shapes are inconsistent.")
    if reference_label not in {"negative", "all"}:
        raise ValueError("reference_label must be 'negative' or 'all'.")
    row_mask = unique.any(dim=1)
    if reference_label == "negative":
        row_mask &= labels.eq(0)
    candidates = torch.nonzero(row_mask, as_tuple=False).flatten().tolist()
    if not candidates:
        raise ValueError("No eligible reference rows remain after filtering.")

    rng = random.Random(stable_int_seed(seed, epoch, "rows"))
    rng.shuffle(candidates)
    if max_rows > 0:
        candidates = candidates[: min(int(max_rows), len(candidates))]
    # Canonical row order is intentionally not restored: the shuffled order
    # makes the round-robin target role independent of source-table ordering.
    timepoints = int(unique.shape[1])
    plan: list[PlanEntry] = []
    for rank, row_index in enumerate(candidates):
        preferred = (rank + int(epoch)) % timepoints
        usable = [
            (preferred + offset) % timepoints
            for offset in range(timepoints)
            if bool(unique[row_index, (preferred + offset) % timepoints])
        ]
        if not usable:
            continue
        visit_index = int(usable[0])
        item_seed = stable_int_seed(seed, epoch, row_index, visit_index, "renderer")
        item_rng = random.Random(item_seed)
        log_min = math.log(renderer.min_peak_drop)
        log_max = math.log(renderer.max_peak_drop)
        peak_drop = math.exp(item_rng.uniform(log_min, log_max))
        plan.append(
            PlanEntry(
                row_index=int(row_index),
                visit_index=visit_index,
                renderer_seed=item_seed,
                peak_drop=float(peak_drop),
            )
        )
    if not plan:
        raise ValueError("Reference plan is empty.")
    if len(plan) >= 2 and timepoints > 1:
        visits = {entry.visit_index for entry in plan}
        if not any(index > 0 for index in visits):
            raise RuntimeError("Plan unexpectedly contains no history injection.")
    return plan


def plan_sha256(plan: Sequence[PlanEntry]) -> str:
    return cache_runner.sha256_bytes(
        cache_runner.canonical_json_bytes([asdict(entry) for entry in plan])
    )


def make_soft_plume(
    height: int,
    width: int,
    *,
    seed: int,
    config: RendererConfig,
) -> torch.Tensor:
    """Render a deterministic curved-ish, downwind elliptical soft field."""

    if height <= 1 or width <= 1:
        raise ValueError("Plume image dimensions must exceed one pixel.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def uniform(low: float, high: float) -> float:
        value = torch.rand((), generator=generator).item()
        return float(low + (high - low) * value)

    yy = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
    xx = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    center_x = uniform(-0.25, 0.25)
    center_y = uniform(-0.25, 0.25)
    angle = uniform(-math.pi, math.pi)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx, dy = grid_x - center_x, grid_y - center_y
    along = cos_a * dx + sin_a * dy
    across = -sin_a * dx + cos_a * dy
    length = uniform(config.plume_min_length, config.plume_max_length)
    width_scale = uniform(config.plume_min_width, config.plume_max_width)
    # A small sinusoidal centerline bend avoids a pure Gaussian template.
    bend = uniform(-0.20, 0.20) * torch.sin(
        math.pi * torch.clamp(along / max(length, 1e-6), -1.0, 1.0)
    )
    across = across - bend * length
    core = torch.exp(
        -0.5 * (along / length).square()
        -0.5 * (across / width_scale).square()
    )
    # Prefer the downwind half while retaining a soft source shoulder.
    downwind = torch.sigmoid((along + 0.15 * length) / (0.08 * length))
    field = core * (0.25 + 0.75 * downwind)
    maximum = float(field.max())
    if not math.isfinite(maximum) or maximum <= 0:
        raise RuntimeError("Synthetic plume renderer produced an empty field.")
    return (field / maximum).clamp_(0.0, 1.0)


def _matched_delta(
    reference_delta: torch.Tensor,
    candidate_delta: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Scale each candidate to exactly match reference L2 energy per sample."""

    if reference_delta.shape != candidate_delta.shape or reference_delta.ndim != 4:
        raise ValueError("Matched deltas must both have shape (B,C,H,W).")
    reduce_dims = (1, 2, 3)
    reference_energy = reference_delta.square().sum(dim=reduce_dims, keepdim=True)
    candidate_energy = candidate_delta.square().sum(dim=reduce_dims, keepdim=True)
    if torch.any(reference_energy <= epsilon) or torch.any(candidate_energy <= epsilon):
        raise ValueError("Cannot energy-match an empty counterfactual delta.")
    return candidate_delta * torch.sqrt(reference_energy / candidate_energy)


def render_counterfactual_variants(
    clean_normalized: torch.Tensor,
    valid_pixels: torch.Tensor,
    plume_field: torch.Tensor,
    peak_drop: torch.Tensor,
    *,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    config: RendererConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Create exact same-scene methane and matched nuisance variants.

    Returns variants shaped ``(B,3,C,H,W)`` in ``VARIANT_NAMES`` order.
    """

    if clean_normalized.ndim != 4:
        raise ValueError("clean_normalized must have shape (B,C,H,W).")
    batch, channels, height, width = clean_normalized.shape
    expected = (batch, channels, height, width)
    if valid_pixels.shape != expected:
        raise ValueError("valid_pixels shape does not match clean images.")
    if plume_field.shape != (batch, height, width):
        raise ValueError("plume_field must have shape (B,H,W).")
    if peak_drop.shape != (batch,):
        raise ValueError("peak_drop must have shape (B,).")
    config.validate(channels)
    mean = normalization_mean.to(
        device=clean_normalized.device, dtype=clean_normalized.dtype
    ).reshape(1, channels, 1, 1)
    std = normalization_std.to(
        device=clean_normalized.device, dtype=clean_normalized.dtype
    ).reshape(1, channels, 1, 1)
    if torch.any(std <= 0):
        raise ValueError("Normalization standard deviations must be positive.")
    valid = valid_pixels.to(device=clean_normalized.device, dtype=torch.bool)
    plume = plume_field.to(
        device=clean_normalized.device, dtype=clean_normalized.dtype
    ).unsqueeze(1)
    strength = peak_drop.to(
        device=clean_normalized.device, dtype=clean_normalized.dtype
    ).reshape(batch, 1, 1, 1)
    raw = torch.clamp(clean_normalized * std + mean, min=0.0)

    def absorption_delta(response: torch.Tensor) -> torch.Tensor:
        response = response.to(
            device=clean_normalized.device, dtype=clean_normalized.dtype
        ).reshape(1, channels, 1, 1)
        transmission = torch.exp(-strength * response * plume)
        delta = raw * (transmission - 1.0) / std
        return torch.where(valid, delta, torch.zeros_like(delta))

    methane_response = torch.tensor(config.response)
    methane_delta = absorption_delta(methane_response)
    achromatic_delta = _matched_delta(
        methane_delta, absorption_delta(torch.ones(channels))
    )
    shuffled_response = methane_response[list(config.wavelength_shuffle)]
    shuffled_delta = _matched_delta(
        methane_delta, absorption_delta(shuffled_response)
    )
    deltas = torch.stack(
        (methane_delta, achromatic_delta, shuffled_delta), dim=1
    )
    variants = clean_normalized.unsqueeze(1) + deltas
    energies = deltas.square().sum(dim=(2, 3, 4))
    relative_error = (
        (energies - energies[:, :1]).abs()
        / energies[:, :1].clamp_min(1e-12)
    )
    diagnostics = {
        "deltas": deltas,
        "energies": energies,
        "max_energy_relative_error": relative_error.max(),
    }
    return variants, diagnostics


class RCTPReferenceDataset(Dataset):
    """Read only the planned visit while retaining strict cache row identity."""

    def __init__(
        self,
        csv_path: Path,
        frame: pd.DataFrame,
        payload: Mapping[str, Any],
        plan: Sequence[PlanEntry],
        *,
        renderer: RendererConfig,
    ):
        super().__init__()
        contract = payload["input_contract"]
        self.csv_path = csv_path
        self.frame = frame.reset_index(drop=True)
        self.payload = payload
        self.plan = list(plan)
        self.renderer = renderer
        self.path_columns = tuple(payload["path_columns"])
        self.band_indices = tuple(int(value) for value in contract["band_indices"])
        self.image_size = int(contract["image_size"])
        self.zero_invalid_pixels = bool(contract["zero_invalid_pixels"])
        self.mean = torch.tensor(
            contract["normalization_mean"], dtype=torch.float32
        )[list(self.band_indices)].view(-1, 1, 1)
        self.std = torch.tensor(
            contract["normalization_std"], dtype=torch.float32
        )[list(self.band_indices)].view(-1, 1, 1)
        self.channel_ids = torch.tensor(
            contract["channel_ids"], dtype=torch.float32
        )
        self.reader = S2CsvDataset(
            csv_path=str(csv_path),
            ds_cfg_name="landsat89_7band",
            path_column=self.path_columns[0],
            normalize_stats=None,
            scale_to_unit=False,
            compute_stats=False,
            pad_to_multiple=None,
            skip_invalid_samples=False,
            path_columns_for_validation=self.path_columns,
        )
        available = int(self.reader.chn_ids.shape[0])
        if available != len(contract["normalization_mean"]):
            raise ValueError("Reader channels differ from cached input contract.")
        renderer.validate(len(self.band_indices))

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.plan[index]
        row = self.frame.iloc[entry.row_index]
        column = self.path_columns[entry.visit_index]
        path = row[column]
        if not isinstance(path, str) or not path.strip():
            raise RuntimeError(
                f"Planned row {entry.row_index} visit {entry.visit_index} has no path."
            )
        raw = self.reader._read_image_raw(path.strip())[list(self.band_indices)]
        native_valid = torch.isfinite(raw) & raw.ne(0)
        clean = (
            torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0) - self.mean
        ) / self.std
        if self.zero_invalid_pixels:
            clean = torch.where(native_valid, clean, torch.zeros_like(clean))
        clean = F.interpolate(
            clean.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        valid = F.interpolate(
            native_valid.float().unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0).bool()
        plume = make_soft_plume(
            self.image_size,
            self.image_size,
            seed=entry.renderer_seed,
            config=self.renderer,
        )
        return {
            "row_index": torch.tensor(entry.row_index, dtype=torch.long),
            "visit_index": torch.tensor(entry.visit_index, dtype=torch.long),
            "renderer_seed": torch.tensor(entry.renderer_seed, dtype=torch.long),
            "peak_drop": torch.tensor(entry.peak_drop, dtype=torch.float32),
            "clean_image": clean,
            "valid_pixels": valid,
            "plume_field": plume,
        }


def response_metadata(
    response: Sequence[float],
    channel_ids_nm: Sequence[float],
    *,
    gsd_m: float = 30.0,
    timepoints: int = 6,
) -> torch.Tensor:
    response_tensor = torch.tensor(response, dtype=torch.float32)
    wavelength = torch.tensor(channel_ids_nm, dtype=torch.float32)
    if response_tensor.shape != wavelength.shape:
        raise ValueError("Response and wavelength metadata lengths differ.")
    wavelength = (wavelength - wavelength.mean()) / wavelength.std().clamp_min(1e-6)
    geometry = torch.tensor(
        [math.log(float(gsd_m)), math.log(float(timepoints))], dtype=torch.float32
    )
    return torch.cat((response_tensor, wavelength, geometry), dim=0)


class RCTPTemporalProbe(nn.Module):
    """Zero-init response adapter and heads over exact paired CLS differences."""

    def __init__(
        self,
        feature_dim: int,
        response_dim: int,
        num_roles: int,
        *,
        model_dim: int = 192,
        num_heads: int = 6,
        dropout: float = 0.1,
        periods_days: Sequence[float] = (1, 3, 7, 30, 90, 365),
    ):
        super().__init__()
        if feature_dim <= 0 or response_dim <= 0 or num_roles < 2:
            raise ValueError("Invalid feature/response/role dimensions.")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.feature_dim = int(feature_dim)
        self.response_dim = int(response_dim)
        self.num_roles = int(num_roles)
        self.delta_norm = nn.LayerNorm(feature_dim)
        self.clean_norm = nn.LayerNorm(feature_dim)
        self.delta_projection = nn.Linear(feature_dim, model_dim)
        self.clean_projection = nn.Linear(feature_dim, model_dim)
        self.response_encoder = nn.Sequential(
            nn.LayerNorm(response_dim),
            nn.Linear(response_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.role_embedding = nn.Embedding(num_roles, model_dim)
        self.delta_time_encoder = cache_runner.ContinuousDeltaEncoder(
            model_dim, periods_days
        )
        self.adapter_norm = nn.LayerNorm(model_dim)
        self.adapter_hidden = nn.Linear(model_dim, model_dim)
        self.adapter_out = nn.Linear(model_dim, model_dim)
        nn.init.zeros_(self.adapter_out.weight)
        nn.init.zeros_(self.adapter_out.bias)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=2 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.query = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.normal_(self.query, std=0.02)
        self.pool = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.type_head = nn.Linear(model_dim, 1)
        self.visit_head = nn.Linear(model_dim, 1)
        self.strength_head = nn.Linear(model_dim, 1)
        for head in (self.type_head, self.visit_head, self.strength_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        clean_features: torch.Tensor,
        delta_features: torch.Tensor,
        valid_mask: torch.Tensor,
        role_index: torch.Tensor,
        delta_days: torch.Tensor,
        response: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if clean_features.shape != delta_features.shape or clean_features.ndim != 3:
            raise ValueError("clean/delta features must share shape (B,T,D).")
        batch, timepoints, feature_dim = clean_features.shape
        if feature_dim != self.feature_dim or timepoints != self.num_roles:
            raise ValueError("Feature dimensions do not match probe configuration.")
        expected = (batch, timepoints)
        if valid_mask.shape != expected or delta_days.shape != expected:
            raise ValueError("valid_mask/delta_days shapes must be (B,T).")
        if role_index.ndim == 1:
            role_index = role_index.view(1, -1).expand(batch, -1)
        if role_index.shape != expected:
            raise ValueError("role_index must have shape (T,) or (B,T).")
        if response.ndim == 1:
            response = response.view(1, -1).expand(batch, -1)
        if response.shape != (batch, self.response_dim):
            raise ValueError("response metadata shape is invalid.")
        if not valid_mask.any(dim=1).all():
            raise ValueError("Every probe example needs at least one valid visit.")
        if torch.any(valid_mask & ~torch.isfinite(delta_days)):
            raise ValueError("A valid visit has non-finite delta_days.")

        delta_token = self.delta_projection(self.delta_norm(delta_features))
        clean_token = self.clean_projection(self.clean_norm(clean_features))
        response_token = self.response_encoder(response).unsqueeze(1)
        adapter_input = self.adapter_norm(delta_token + response_token)
        adapter = self.adapter_out(F.gelu(self.adapter_hidden(adapter_input)))
        roles = self.role_embedding(role_index.long())
        times = self.delta_time_encoder(
            torch.where(
                valid_mask.bool(),
                delta_days,
                torch.full_like(delta_days, float("nan")),
            )
        )
        tokens = delta_token + 0.10 * clean_token + roles + times + adapter
        padding = ~valid_mask.bool()
        tokens = self.temporal_encoder(tokens, src_key_padding_mask=padding)
        query = self.query.expand(batch, -1, -1)
        pooled, _ = self.pool(
            query, tokens, tokens, key_padding_mask=padding, need_weights=False
        )
        pooled = self.output_norm(pooled[:, 0])
        visit_logits = self.visit_head(tokens).squeeze(-1)
        visit_logits = visit_logits.masked_fill(padding, -1e4)
        return {
            "type_logit": self.type_head(pooled).squeeze(-1),
            "visit_logits": visit_logits,
            "strength_log": self.strength_head(pooled).squeeze(-1),
            "pooled": pooled,
        }


def assemble_probe_batch(
    cached_clean_features: torch.Tensor,
    cached_valid_mask: torch.Tensor,
    cached_delta_days: torch.Tensor,
    visit_index: torch.Tensor,
    online_clean_cls: torch.Tensor,
    online_variant_cls: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Expand B matched groups into B*3 probe examples."""

    if cached_clean_features.ndim != 3:
        raise ValueError("cached_clean_features must have shape (B,T,D).")
    batch, timepoints, feature_dim = cached_clean_features.shape
    variants = len(VARIANT_NAMES)
    if online_clean_cls.shape != (batch, feature_dim):
        raise ValueError("online_clean_cls shape is invalid.")
    if online_variant_cls.shape != (batch, variants, feature_dim):
        raise ValueError("online_variant_cls shape is invalid.")
    if cached_valid_mask.shape != (batch, timepoints):
        raise ValueError("cached_valid_mask shape is invalid.")
    if cached_delta_days.shape != (batch, timepoints):
        raise ValueError("cached_delta_days shape is invalid.")
    if visit_index.shape != (batch,):
        raise ValueError("visit_index shape is invalid.")
    if torch.any(visit_index < 0) or torch.any(visit_index >= timepoints):
        raise ValueError("visit_index is outside the temporal sequence.")
    selected_valid = cached_valid_mask[
        torch.arange(batch, device=visit_index.device), visit_index
    ]
    if not selected_valid.all():
        raise ValueError("An injected visit is not valid in the clean cache.")

    clean = cached_clean_features.clone()
    rows = torch.arange(batch, device=visit_index.device)
    clean[rows, visit_index] = online_clean_cls
    clean = (
        clean[:, None, :, :]
        .expand(batch, variants, timepoints, feature_dim)
        .reshape(batch * variants, timepoints, feature_dim)
        .contiguous()
    )
    delta = torch.zeros_like(clean)
    group = torch.arange(batch, device=visit_index.device)[:, None]
    arm = torch.arange(variants, device=visit_index.device)[None, :]
    flat = (group * variants + arm).reshape(-1)
    flat_visit = visit_index[:, None].expand(batch, variants).reshape(-1)
    exact_delta = (
        online_variant_cls - online_clean_cls[:, None, :]
    ).reshape(batch * variants, feature_dim)
    delta[flat, flat_visit] = exact_delta
    valid = (
        cached_valid_mask[:, None, :]
        .expand(batch, variants, timepoints)
        .reshape(batch * variants, timepoints)
        .contiguous()
    )
    delta_days = (
        cached_delta_days[:, None, :]
        .expand(batch, variants, timepoints)
        .reshape(batch * variants, timepoints)
        .contiguous()
    )
    type_target = torch.tensor(
        [1.0, 0.0, 0.0],
        device=clean.device,
        dtype=clean.dtype,
    ).repeat(batch)
    return {
        "clean_features": clean,
        "delta_features": delta,
        "valid_mask": valid,
        "delta_days": delta_days,
        "visit_target": flat_visit,
        "type_target": type_target,
    }


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best = (-1.0, 0.5)
    for threshold in thresholds:
        score = float(f1_score(labels, probabilities >= threshold, zero_division=0))
        candidate = (score, -abs(float(threshold) - 0.5))
        incumbent = (best[0], -abs(best[1] - 0.5))
        if candidate > incumbent:
            best = (score, float(threshold))
    return best[1], best[0]


def compute_screen_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    group_ids: Sequence[int],
    variant_indices: Sequence[int],
    visit_targets: Sequence[int],
    visit_predictions: Sequence[int],
    strength_targets: Sequence[float],
    strength_predictions: Sequence[float],
) -> dict[str, float | int]:
    labels_np = np.asarray(labels, dtype=np.int64)
    probs_np = np.asarray(probabilities, dtype=np.float64)
    groups_np = np.asarray(group_ids, dtype=np.int64)
    variants_np = np.asarray(variant_indices, dtype=np.int64)
    if set(np.unique(labels_np).tolist()) != {0, 1}:
        raise ValueError("Metrics require both methane and nuisance examples.")
    threshold, selected_f1 = best_f1_threshold(labels_np, probs_np)
    paired_wins: list[float] = []
    paired_margins: list[float] = []
    for group_id in np.unique(groups_np):
        select = groups_np == group_id
        group_prob = probs_np[select]
        group_variant = variants_np[select]
        positive = group_prob[group_variant == METHANE_VARIANT_INDEX]
        negative = group_prob[group_variant != METHANE_VARIANT_INDEX]
        if len(positive) != 1 or len(negative) != 2:
            raise ValueError("Every matched group must contain one methane and two nuisances.")
        paired_wins.append(float(positive[0] > negative.max()))
        paired_margins.append(float(positive[0] - negative.max()))
    visit_target_np = np.asarray(visit_targets, dtype=np.int64)
    visit_pred_np = np.asarray(visit_predictions, dtype=np.int64)
    strength_target_np = np.asarray(strength_targets, dtype=np.float64)
    strength_pred_np = np.asarray(strength_predictions, dtype=np.float64)
    methane = variants_np == METHANE_VARIANT_INDEX
    metrics: dict[str, float | int] = {
        "average_precision": float(average_precision_score(labels_np, probs_np)),
        "roc_auc": float(roc_auc_score(labels_np, probs_np)),
        "selected_f1": float(selected_f1),
        "selected_threshold": float(threshold),
        "fixed_0p5_f1": float(
            f1_score(labels_np, probs_np >= 0.5, zero_division=0)
        ),
        "paired_win_rate": float(np.mean(paired_wins)),
        "paired_probability_margin": float(np.mean(paired_margins)),
        "visit_accuracy": float(np.mean(visit_target_np == visit_pred_np)),
        "methane_strength_log_mae": float(
            np.mean(
                np.abs(strength_target_np[methane] - strength_pred_np[methane])
            )
        ),
        "examples": int(len(labels_np)),
        "matched_groups": int(len(np.unique(groups_np))),
    }
    # The mechanism is only interesting if it survives weak perturbations and
    # history injection.  These fixed, predeclared panels prevent an attractive
    # aggregate from hiding a high-strength or t0-only shortcut.
    peak_drop_np = np.exp(strength_target_np)
    panels = {
        "weak_le_0p02": peak_drop_np <= 0.02,
        "mid_0p02_to_0p04": (peak_drop_np > 0.02) & (peak_drop_np <= 0.04),
        "strong_gt_0p04": peak_drop_np > 0.04,
        "t0": visit_target_np == 0,
        "history": visit_target_np != 0,
    }
    for panel_name, panel_mask in panels.items():
        panel_labels = labels_np[panel_mask]
        if len(panel_labels) == 0 or set(np.unique(panel_labels).tolist()) != {0, 1}:
            metrics[f"{panel_name}_examples"] = int(len(panel_labels))
            continue
        panel_probs = probs_np[panel_mask]
        metrics[f"{panel_name}_examples"] = int(len(panel_labels))
        metrics[f"{panel_name}_average_precision"] = float(
            average_precision_score(panel_labels, panel_probs)
        )
        metrics[f"{panel_name}_roc_auc"] = float(
            roc_auc_score(panel_labels, panel_probs)
        )
        panel_groups = groups_np[panel_mask]
        panel_variants = variants_np[panel_mask]
        panel_wins: list[float] = []
        for group_id in np.unique(panel_groups):
            group_select = panel_groups == group_id
            group_prob = panel_probs[group_select]
            group_variant = panel_variants[group_select]
            positive = group_prob[group_variant == METHANE_VARIANT_INDEX]
            negative = group_prob[group_variant != METHANE_VARIANT_INDEX]
            if len(positive) == 1 and len(negative) == 2:
                panel_wins.append(float(positive[0] > negative.max()))
        if panel_wins:
            metrics[f"{panel_name}_paired_win_rate"] = float(np.mean(panel_wins))
        metrics[f"{panel_name}_visit_accuracy"] = float(
            np.mean(visit_target_np[panel_mask] == visit_pred_np[panel_mask])
        )
    for nuisance_index, nuisance_name in (
        (1, "achromatic"),
        (2, "wavelength_shuffled"),
    ):
        select = (variants_np == METHANE_VARIANT_INDEX) | (
            variants_np == nuisance_index
        )
        binary_labels = labels_np[select]
        binary_probs = probs_np[select]
        metrics[f"methane_vs_{nuisance_name}_average_precision"] = float(
            average_precision_score(binary_labels, binary_probs)
        )
        metrics[f"methane_vs_{nuisance_name}_roc_auc"] = float(
            roc_auc_score(binary_labels, binary_probs)
        )
    return metrics


def validate_frame_cache_alignment(
    csv_path: Path,
    payload: Mapping[str, Any],
) -> pd.DataFrame:
    cache_runner.assert_not_sealed_path(csv_path, purpose="RCTP CSV")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    csv_sha = cache_runner.sha256_file(csv_path)
    if csv_sha != payload["csv_sha256"]:
        raise ValueError(
            f"CSV SHA {csv_sha} does not match cache {payload['csv_sha256']}."
        )
    frame = pd.read_csv(csv_path, low_memory=False)
    if len(frame) != len(payload["features"]):
        raise ValueError("CSV row count differs from cache.")
    id_column = "id"
    if id_column not in frame:
        raise ValueError("RCTP manifest must contain an id column.")
    ids = frame[id_column].fillna("").astype(str).tolist()
    if ids != [str(value) for value in payload["ids"]]:
        raise ValueError("CSV/cache row IDs are not exactly aligned.")
    return frame


def _encode_exact_group(
    backbone: nn.Module,
    clean: torch.Tensor,
    variants: torch.Tensor,
    channel_ids: torch.Tensor,
    *,
    device: torch.device,
    amp_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_variants, channels, height, width = variants.shape
    images = torch.cat((clean.unsqueeze(1), variants), dim=1).reshape(
        batch * (num_variants + 1), channels, height, width
    )
    ids = channel_ids.view(1, -1).expand(len(images), -1).clone()
    with torch.inference_mode(), autocast_context(device, amp_dtype):
        output = backbone.forward_features(
            {
                "imgs": images.to(device, non_blocking=True),
                "chn_ids": ids.to(device, non_blocking=True),
            }
        )
        cls = output["x_norm_clstoken"].float().reshape(
            batch, num_variants + 1, -1
        )
    return cls[:, 0], cls[:, 1:]


def _loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    peak_drop: torch.Tensor,
    *,
    type_weight: float,
    visit_weight: float,
    strength_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["type_target"].float()
    positive_weight = torch.tensor(2.0, device=target.device)
    type_loss = F.binary_cross_entropy_with_logits(
        output["type_logit"], target, pos_weight=positive_weight
    )
    visit_loss = F.cross_entropy(output["visit_logits"], batch["visit_target"])
    strength_target = (
        peak_drop[:, None].expand(-1, len(VARIANT_NAMES)).reshape(-1).log()
    )
    methane = target.bool()
    strength_loss = F.smooth_l1_loss(
        output["strength_log"][methane], strength_target[methane]
    )
    total = (
        float(type_weight) * type_loss
        + float(visit_weight) * visit_loss
        + float(strength_weight) * strength_loss
    )
    return total, {
        "type_loss": float(type_loss.detach()),
        "visit_loss": float(visit_loss.detach()),
        "strength_loss": float(strength_loss.detach()),
        "total_loss": float(total.detach()),
    }


def _run_epoch(
    *,
    backbone: nn.Module,
    probe: RCTPTemporalProbe,
    loader: DataLoader,
    payload: Mapping[str, Any],
    renderer: RendererConfig,
    metadata: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
    optimizer: Optional[torch.optim.Optimizer],
    max_steps: int,
    global_step: int,
    type_weight: float,
    visit_weight: float,
    strength_weight: float,
    grad_clip: float,
    log_interval: int,
) -> tuple[dict[str, float | int], int, pd.DataFrame]:
    training = optimizer is not None
    probe.train(training)
    backbone.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    row_ids: list[int] = []
    group_ids: list[int] = []
    variant_indices: list[int] = []
    visit_targets: list[int] = []
    visit_predictions: list[int] = []
    strength_targets: list[float] = []
    strength_predictions: list[float] = []
    losses: list[dict[str, float]] = []
    started = time.monotonic()
    mean = torch.tensor(
        payload["input_contract"]["normalization_mean"], dtype=torch.float32
    )[payload["input_contract"]["band_indices"]]
    std = torch.tensor(
        payload["input_contract"]["normalization_std"], dtype=torch.float32
    )[payload["input_contract"]["band_indices"]]
    role_index = payload["role_index"].long().to(device)
    channel_ids = torch.tensor(
        payload["input_contract"]["channel_ids"], dtype=torch.float32
    )

    for batch_index, source in enumerate(loader, 1):
        if training and max_steps > 0 and global_step >= max_steps:
            break
        row_indices = source["row_index"].long()
        visit = source["visit_index"].long().to(device)
        clean = source["clean_image"].float().to(device, non_blocking=True)
        valid_pixels = source["valid_pixels"].bool().to(device, non_blocking=True)
        plume = source["plume_field"].float().to(device, non_blocking=True)
        peak_drop = source["peak_drop"].float().to(device, non_blocking=True)
        variants, diagnostics = render_counterfactual_variants(
            clean,
            valid_pixels,
            plume,
            peak_drop,
            normalization_mean=mean,
            normalization_std=std,
            config=renderer,
        )
        if float(diagnostics["max_energy_relative_error"]) > 5e-5:
            raise RuntimeError("Nuisance energy matching exceeded tolerance.")
        if not torch.isfinite(variants).all():
            raise RuntimeError("Counterfactual renderer produced non-finite pixels.")
        online_clean, online_variants = _encode_exact_group(
            backbone,
            clean,
            variants,
            channel_ids,
            device=device,
            amp_dtype=amp_dtype,
        )
        if not torch.isfinite(online_clean).all() or not torch.isfinite(
            online_variants
        ).all():
            raise RuntimeError("Frozen encoder produced non-finite CLS features.")
        cached_clean = payload["features"][row_indices].float().to(
            device, non_blocking=True
        )
        cached_valid = payload["unique_mask"][row_indices].bool().to(
            device, non_blocking=True
        )
        cached_delta = payload["delta_days"][row_indices].float().to(
            device, non_blocking=True
        )
        assembled = assemble_probe_batch(
            cached_clean,
            cached_valid,
            cached_delta,
            visit,
            online_clean,
            online_variants,
        )
        repeated_metadata = metadata.to(device).view(1, -1).expand(
            len(assembled["type_target"]), -1
        )
        with torch.set_grad_enabled(training), autocast_context(device, amp_dtype):
            output = probe(
                assembled["clean_features"],
                assembled["delta_features"],
                assembled["valid_mask"],
                role_index,
                assembled["delta_days"],
                repeated_metadata,
            )
            total_loss, loss_values = _loss(
                output,
                assembled,
                peak_drop,
                type_weight=type_weight,
                visit_weight=visit_weight,
                strength_weight=strength_weight,
            )
        if not torch.isfinite(total_loss):
            raise RuntimeError("RCTP objective became non-finite.")
        if training:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(probe.parameters(), float(grad_clip))
            optimizer.step()
            global_step += 1
        losses.append(loss_values)

        probability = torch.sigmoid(output["type_logit"]).detach().float().cpu()
        predictions = output["visit_logits"].argmax(dim=1).detach().cpu()
        targets = assembled["type_target"].detach().long().cpu()
        strength_target = (
            peak_drop[:, None]
            .expand(-1, len(VARIANT_NAMES))
            .reshape(-1)
            .log()
            .detach()
            .cpu()
        )
        batch_size = len(row_indices)
        labels.extend(targets.tolist())
        probabilities.extend(probability.tolist())
        repeated_rows = row_indices.repeat_interleave(len(VARIANT_NAMES)).tolist()
        row_ids.extend(repeated_rows)
        group_ids.extend(repeated_rows)
        variant_indices.extend(
            torch.arange(len(VARIANT_NAMES)).repeat(batch_size).tolist()
        )
        visit_targets.extend(assembled["visit_target"].detach().cpu().tolist())
        visit_predictions.extend(predictions.tolist())
        strength_targets.extend(strength_target.tolist())
        strength_predictions.extend(
            output["strength_log"].detach().float().cpu().tolist()
        )
        if batch_index % max(1, log_interval) == 0:
            print(
                f"[{'train' if training else 'dev'}] batch={batch_index}/{len(loader)} "
                f"step={global_step} loss={loss_values['total_loss']:.4f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    if not labels:
        raise RuntimeError("Epoch emitted no examples.")
    metrics = compute_screen_metrics(
        labels,
        probabilities,
        group_ids,
        variant_indices,
        visit_targets,
        visit_predictions,
        strength_targets,
        strength_predictions,
    )
    for name in ("type_loss", "visit_loss", "strength_loss", "total_loss"):
        metrics[name] = float(np.mean([value[name] for value in losses]))
    metrics["elapsed_seconds"] = float(time.monotonic() - started)
    metrics["optimizer_steps"] = int(global_step)
    prediction_frame = pd.DataFrame(
        {
            "cache_row_index": row_ids,
            "group_id": group_ids,
            "variant_index": variant_indices,
            "variant": [VARIANT_NAMES[index] for index in variant_indices],
            "target_is_methane": labels,
            "probability_methane": probabilities,
            "injected_visit_index": visit_targets,
            "predicted_visit_index": visit_predictions,
            "target_log_strength": strength_targets,
            "predicted_log_strength": strength_predictions,
        }
    )
    return metrics, global_step, prediction_frame


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    training: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(training),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": False,
        "generator": torch.Generator().manual_seed(int(seed)),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def run_screen(args: argparse.Namespace) -> None:
    train_csv = Path(args.train_csv).expanduser().resolve()
    dev_csv = Path(args.dev_csv).expanduser().resolve()
    train_cache = Path(args.train_cache).expanduser().resolve()
    dev_cache = Path(args.dev_cache).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    for path, purpose in (
        (train_csv, "train CSV"),
        (dev_csv, "dev CSV"),
        (train_cache, "train cache"),
        (dev_cache, "dev cache"),
        (output_dir, "output"),
    ):
        cache_runner.assert_not_sealed_path(path, purpose=purpose)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is nonempty: {output_dir}; pass --overwrite."
        )
    if not weights.is_file():
        raise FileNotFoundError(weights)
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_payload, dev_payload, cache_audit = cache_runner.load_cache_pair(
        train_cache, dev_cache
    )
    train_frame = validate_frame_cache_alignment(train_csv, train_payload)
    dev_frame = validate_frame_cache_alignment(dev_csv, dev_payload)
    weights_sha = cache_runner.sha256_file(weights)
    if weights_sha != train_payload["weights_sha256"]:
        raise ValueError(
            "The online PTH differs from the frozen-CLS cache PTH. Exact paired "
            "delta encoding remains valid, but the five cached context visits "
            "would be incompatible; rebuild caches with this PTH first."
        )
    response = tuple(
        float(part.strip())
        for part in args.ch4_response.split(",")
        if part.strip()
    )
    shuffle = tuple(
        int(part.strip())
        for part in args.wavelength_shuffle.split(",")
        if part.strip()
    )
    renderer = RendererConfig(
        response=response,
        wavelength_shuffle=shuffle,
        min_peak_drop=float(args.min_peak_drop),
        max_peak_drop=float(args.max_peak_drop),
    )
    channels = int(train_payload["features"].shape[2])
    input_channels = len(train_payload["input_contract"]["band_indices"])
    renderer.validate(input_channels)
    metadata = response_metadata(
        renderer.response,
        train_payload["input_contract"]["channel_ids"],
        timepoints=int(train_payload["features"].shape[1]),
    )
    backbone = load_backbone(str(weights), device=device, debug=args.debug).to(device)
    backbone.requires_grad_(False)
    backbone.eval()
    probe_config = {
        "feature_dim": channels,
        "response_dim": int(metadata.numel()),
        "num_roles": int(train_payload["features"].shape[1]),
        "model_dim": int(args.model_dim),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "periods_days": tuple(
            float(value.strip())
            for value in args.delta_periods.split(",")
            if value.strip()
        ),
    }
    probe = RCTPTemporalProbe(**probe_config).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_plan = deterministic_reference_plan(
        dev_payload,
        seed=args.seed + 100_003,
        epoch=0,
        max_rows=args.max_dev_rows,
        reference_label=args.reference_label,
        renderer=renderer,
    )
    dev_dataset = RCTPReferenceDataset(
        dev_csv, dev_frame, dev_payload, dev_plan, renderer=renderer
    )
    dev_loader = _loader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        training=False,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    history: list[dict[str, Any]] = []
    best_ap = -math.inf
    global_step = 0
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_metrics: Optional[dict[str, float | int]] = None
    best_dev_predictions: Optional[pd.DataFrame] = None
    for epoch in range(1, int(args.epochs) + 1):
        train_plan = deterministic_reference_plan(
            train_payload,
            seed=args.seed,
            epoch=epoch - 1,
            max_rows=args.max_train_rows,
            reference_label=args.reference_label,
            renderer=renderer,
        )
        train_dataset = RCTPReferenceDataset(
            train_csv, train_frame, train_payload, train_plan, renderer=renderer
        )
        train_loader = _loader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            training=True,
            seed=args.seed + epoch,
            pin_memory=device.type == "cuda",
        )
        train_metrics, global_step, _ = _run_epoch(
            backbone=backbone,
            probe=probe,
            loader=train_loader,
            payload=train_payload,
            renderer=renderer,
            metadata=metadata,
            device=device,
            amp_dtype=args.amp_dtype,
            optimizer=optimizer,
            max_steps=args.max_train_steps,
            global_step=global_step,
            type_weight=args.type_loss_weight,
            visit_weight=args.visit_loss_weight,
            strength_weight=args.strength_loss_weight,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
        )
        dev_metrics, _, dev_predictions = _run_epoch(
            backbone=backbone,
            probe=probe,
            loader=dev_loader,
            payload=dev_payload,
            renderer=renderer,
            metadata=metadata,
            device=device,
            amp_dtype=args.amp_dtype,
            optimizer=None,
            max_steps=0,
            global_step=global_step,
            type_weight=args.type_loss_weight,
            visit_weight=args.visit_loss_weight,
            strength_weight=args.strength_loss_weight,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
        )
        record = {
            "epoch": epoch,
            "train_plan_sha256": plan_sha256(train_plan),
            "train_rows": len(train_plan),
            "train": train_metrics,
            "dev": dev_metrics,
        }
        history.append(record)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        if dev_metrics["average_precision"] > best_ap:
            best_ap = float(dev_metrics["average_precision"])
            best_epoch = epoch
            best_metrics = copy.deepcopy(dev_metrics)
            best_dev_predictions = dev_predictions.copy()
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in probe.state_dict().items()
            }
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break
    if best_state is None or best_metrics is None or best_dev_predictions is None:
        raise RuntimeError("No checkpoint was selected.")
    checkpoint = {
        "script_version": SCRIPT_VERSION,
        "selection": "maximum dev average_precision",
        "best_epoch": best_epoch,
        "best_dev_metrics": best_metrics,
        "probe_config": probe_config,
        "probe_state": best_state,
        "renderer": asdict(renderer),
        "response_metadata": metadata,
        "weights_path": str(weights),
        "weights_sha256": weights_sha,
        "cache_audit": cache_audit,
        "dev_plan_sha256": plan_sha256(dev_plan),
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"handler"}
        },
    }
    checkpoint_path = output_dir / "checkpoint_best_dev_ap.pt"
    cache_runner.atomic_torch_save(checkpoint_path, checkpoint)
    prediction_path = output_dir / "best_dev_predictions.csv"
    cache_runner.atomic_csv_write(prediction_path, best_dev_predictions)
    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "complete_train_dev_only",
        "best_epoch": best_epoch,
        "best_dev_metrics": best_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": cache_runner.sha256_file(checkpoint_path),
        "best_dev_predictions": str(prediction_path),
        "best_dev_predictions_sha256": cache_runner.sha256_file(prediction_path),
        "history": history,
        "dev_rows": len(dev_plan),
        "dev_plan_sha256": plan_sha256(dev_plan),
        "renderer": asdict(renderer),
        "cache_audit": cache_audit,
        "weights_sha256": weights_sha,
        "test_evaluations": 0,
    }
    cache_runner.atomic_json_write(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/dev-only L89 RCTP counterfactual mechanism screen."
    )
    parser.add_argument(
        "--train-csv",
        default=(
            "/diniuvol/yuyao/methanefuse_research_20260727/"
            "manifests_staged/l89_6time/train.csv"
        ),
    )
    parser.add_argument(
        "--dev-csv",
        default=(
            "/diniuvol/yuyao/methanefuse_research_20260727/"
            "manifests_staged/l89_6time/val.csv"
        ),
    )
    parser.add_argument(
        "--train-cache",
        default=(
            "/diniuvol/yuyao/methanefuse_research_20260727/"
            "cache/l89_ragged_cls_v1/train.pt"
        ),
    )
    parser.add_argument(
        "--dev-cache",
        default=(
            "/diniuvol/yuyao/methanefuse_research_20260727/"
            "cache/l89_ragged_cls_v1/val.pt"
        ),
    )
    parser.add_argument(
        "--weights", default="weights/panopticon_vitb14_teacher.pth"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-rows", type=int, default=4096)
    parser.add_argument("--max-dev-rows", type=int, default=2048)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Global optimizer-step cap across epochs; zero means no cap.",
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--delta-periods", default="1,3,7,30,90,365")
    parser.add_argument("--type-loss-weight", type=float, default=1.0)
    parser.add_argument("--visit-loss-weight", type=float, default=0.25)
    parser.add_argument("--strength-loss-weight", type=float, default=0.10)
    parser.add_argument("--reference-label", choices=("negative", "all"), default="negative")
    parser.add_argument(
        "--ch4-response",
        default=",".join(str(value) for value in DEFAULT_L89_CH4_RESPONSE),
    )
    parser.add_argument(
        "--wavelength-shuffle",
        default=",".join(str(value) for value in DEFAULT_WAVELENGTH_SHUFFLE),
    )
    parser.add_argument("--min-peak-drop", type=float, default=0.01)
    parser.add_argument("--max-peak-drop", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(handler=run_screen)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
