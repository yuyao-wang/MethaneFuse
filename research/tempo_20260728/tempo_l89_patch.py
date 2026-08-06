#!/usr/bin/env python3
"""TEMPO: frozen patch-local temporal onset head for six-visit L89.

The expensive Panopticon backbone is never optimized by this program.  Its
final 768-dimensional patch tokens are projected online to a fixed 64-D
subspace and written to resumable local shards.  Raw 768-D patch tokens are
never persisted.

The trainable head asks a current (t0) patch to retrieve the best matching
background inside a small spatial neighbourhood of every valid historical
visit.  It then contrasts current-to-history change with explicit
history-to-history variation.  Real time lags and per-observation valid
fractions gate the history, top-k MIL preserves a small local onset, and an
all-negative canonical-event loss suppresses spurious patch activations.  The
new branch is added to a frozen base logit with an exact epoch-zero identity.
The default keeps the original zero scalar gate; a stability control instead
zeros the patch readout and fixes the residual gate to one.

This is deliberately a train/development-only implementation.  Every command
rejects paths containing ``test``, ``sealed`` or ``holdout`` and a cache pair
with canonical-event overlap is rejected.  Nothing in this file authorizes a
locked evaluation.
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
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as l89,
)
from research.pretraining_20260727 import (  # noqa: E402
    l89_patch_local_rctp_followup as patch_followup,
)
from research.pretraining_20260727 import (  # noqa: E402
    rctp_l89_event_balanced_head_followup as event_head_followup,
)
from research.pretraining_20260727.legacy360_patch_cache import (  # noqa: E402
    deterministic_orthogonal_projection,
    projection_metadata,
    tensor_sha256,
)
from Upgraded_dataset.dino_classifier_head_l89_temporal_satmae import (  # noqa: E402
    load_backbone,
)


SCRIPT_VERSION = "tempo-l89-patch-v1"
CACHE_VERSION = "tempo-l89-projected-patch-shards-v1"
SHARD_VERSION = "tempo-l89-projected-patch-shard-v1"
HEAD_VERSION = "tempo-l89-background-onset-head-v1"
OVERLAY_VERSION = "tempo-l89-base-logit-overlay-v1"
FORBIDDEN_PATH_TOKENS = ("test", "sealed", "holdout")
DEFAULT_LOCAL_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260728/cache"
)


def nested_extend_orthogonal_projection(
    input_dim: int,
    output_dim: int,
    *,
    base_dim: int = 128,
    base_seed: int = 36064,
    extension_seed: int = 36192,
) -> torch.Tensor:
    """Extend an existing Gaussian-QR basis without changing its columns."""

    if not 1 <= int(base_dim) < int(output_dim) <= int(input_dim):
        raise ValueError(
            "nested projection needs 1 <= base_dim < output_dim <= input_dim"
        )
    base = deterministic_orthogonal_projection(
        int(input_dim), int(base_dim), seed=int(base_seed)
    )
    base64 = base.double()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(extension_seed))
    candidate = torch.randn(
        int(input_dim),
        int(output_dim) - int(base_dim),
        generator=generator,
        dtype=torch.float64,
    )
    # Two projections guard against finite-precision leakage into the fixed
    # base subspace before the extension QR.
    candidate = candidate - base64 @ (base64.T @ candidate)
    candidate = candidate - base64 @ (base64.T @ candidate)
    extra, triangular = torch.linalg.qr(candidate, mode="reduced")
    diagonal = torch.diagonal(triangular)
    signs = torch.where(
        diagonal < 0,
        -torch.ones_like(diagonal),
        torch.ones_like(diagonal),
    )
    extra = (extra * signs.unsqueeze(0)).float().contiguous()
    projection = torch.cat((base, extra), dim=1).contiguous()
    if not torch.equal(projection[:, : int(base_dim)], base):
        raise AssertionError("nested projection changed the fixed base columns")
    return projection


def nested_projection_metadata(
    projection: torch.Tensor,
    *,
    base_dim: int,
    base_seed: int,
    extension_seed: int,
) -> dict[str, Any]:
    """Audit a nested extension, including exact prefix equality."""

    base = deterministic_orthogonal_projection(
        int(projection.shape[0]), int(base_dim), seed=int(base_seed)
    )
    prefix = projection[:, : int(base_dim)]
    identity = torch.eye(projection.shape[1], dtype=torch.float32)
    orthogonality_error = (
        projection.float().T @ projection.float() - identity
    ).abs().max()
    return {
        "input_dim": int(projection.shape[0]),
        "output_dim": int(projection.shape[1]),
        "seed": int(base_seed),
        "extension_seed": int(extension_seed),
        "base_dim": int(base_dim),
        "algorithm": (
            "nested_extend_fixed_gaussian_qr_base_"
            "plus_projected_gaussian_qr_extension"
        ),
        "sha256": tensor_sha256(projection),
        "base_projection_sha256": tensor_sha256(base),
        "extension_sha256": tensor_sha256(
            projection[:, int(base_dim) :].contiguous()
        ),
        "first_base_columns_array_equal": bool(torch.equal(prefix, base)),
        "first_base_columns_max_abs_error": float(
            (prefix - base).abs().max()
        ),
        "max_abs_qtq_minus_i": float(orthogonality_error),
    }


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def assert_development_path(path: Path, *, purpose: str) -> None:
    """Reject any path that could plausibly name held-out material."""

    lowered = str(path.expanduser().resolve()).casefold()
    hits = [token for token in FORBIDDEN_PATH_TOKENS if token in lowered]
    if hits:
        raise ValueError(
            f"{purpose} path contains held-out token(s) {hits}: {path}"
        )


def _load_torch_payload(path: Path) -> Mapping[str, Any]:
    return patch_followup._load_torch_payload(path)


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    """Numerically safe masked softmax which returns zero for an empty set."""

    mask = mask.to(device=scores.device, dtype=torch.bool)
    while mask.ndim < scores.ndim:
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, scores.shape)
    safe = torch.where(mask, scores, torch.full_like(scores, -1.0e4))
    maximum = safe.max(dim=dim, keepdim=True).values
    numerator = torch.where(mask, torch.exp(safe - maximum), torch.zeros_like(safe))
    denominator = numerator.sum(dim=dim, keepdim=True)
    return numerator / denominator.clamp_min(torch.finfo(scores.dtype).tiny)


def _shift_spatial(
    values: torch.Tensor,
    *,
    dy: int,
    dx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source ``(y+dy,x+dx)`` at every destination coordinate.

    ``values`` is ``[..., H, W, C]``.  The boolean return has shape ``[H,W]``
    and marks destinations whose shifted source lies inside the grid.
    """

    if values.ndim < 3:
        raise ValueError("spatial tensor needs shape [...,H,W,C]")
    height, width = int(values.shape[-3]), int(values.shape[-2])
    output = torch.zeros_like(values)
    valid = torch.zeros(height, width, device=values.device, dtype=torch.bool)
    dst_y0 = max(0, -int(dy))
    dst_y1 = min(height, height - int(dy))
    dst_x0 = max(0, -int(dx))
    dst_x1 = min(width, width - int(dx))
    if dst_y1 > dst_y0 and dst_x1 > dst_x0:
        src_y0, src_y1 = dst_y0 + int(dy), dst_y1 + int(dy)
        src_x0, src_x1 = dst_x0 + int(dx), dst_x1 + int(dx)
        output[..., dst_y0:dst_y1, dst_x0:dst_x1, :] = values[
            ..., src_y0:src_y1, src_x0:src_x1, :
        ]
        valid[dst_y0:dst_y1, dst_x0:dst_x1] = True
    return output, valid


def local_soft_match(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    radius: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-match every current patch inside each history's local window.

    Parameters are already in low-rank query/key/value spaces:

    - ``query``: ``[B,P,R]``
    - ``keys``: ``[B,H,P,R]``
    - ``values``: ``[B,H,P,V]``

    The implementation stores only the small ``[B,H,P,K]`` score tensor.  It
    accumulates value vectors offset-by-offset rather than materializing a
    potentially multi-GiB ``[B,H,P,K,V]`` tensor for a 5x5 window.
    """

    if query.ndim != 3 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("query/keys/values need [B,P,R]/[B,H,P,R]/[B,H,P,V]")
    batch, patches, rank = query.shape
    if keys.shape[:2] != values.shape[:2] or keys.shape[:3] != (
        batch,
        keys.shape[1],
        patches,
    ):
        raise ValueError("history tensors have incompatible leading dimensions")
    if int(keys.shape[-1]) != rank:
        raise ValueError("query/key ranks differ")
    height, width = (int(grid_shape[0]), int(grid_shape[1]))
    if height * width != patches:
        raise ValueError("grid_shape does not equal patch count")
    if radius not in {0, 1, 2}:
        raise ValueError(
            "TEMPO supports radius=0 (same pixel), 1 (3x3), or 2 (5x5)"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    histories = int(keys.shape[1])
    query = F.normalize(query.float(), dim=-1, eps=1e-6)
    keys = F.normalize(keys.float(), dim=-1, eps=1e-6)
    values = values.float()
    key_grid = keys.reshape(batch, histories, height, width, rank)
    value_dim = int(values.shape[-1])
    value_grid = values.reshape(batch, histories, height, width, value_dim)
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    logits: list[torch.Tensor] = []
    spatial_masks: list[torch.Tensor] = []
    for dy, dx in offsets:
        shifted_key, valid = _shift_spatial(key_grid, dy=dy, dx=dx)
        shifted_key = shifted_key.reshape(batch, histories, patches, rank)
        logits.append(
            (shifted_key * query[:, None]).sum(dim=-1) / float(temperature)
        )
        spatial_masks.append(valid.reshape(patches))
    score = torch.stack(logits, dim=-1)
    spatial_mask = torch.stack(spatial_masks, dim=-1).view(
        1, 1, patches, len(offsets)
    )
    weights = _masked_softmax(score, spatial_mask, dim=-1)
    aligned = torch.zeros(
        batch,
        histories,
        patches,
        value_dim,
        device=values.device,
        dtype=torch.float32,
    )
    for offset_index, (dy, dx) in enumerate(offsets):
        shifted_value, _ = _shift_spatial(value_grid, dy=dy, dx=dx)
        aligned = aligned + weights[..., offset_index, None] * shifted_value.reshape(
            batch, histories, patches, value_dim
        )
    entropy = -(
        weights * torch.log(weights.clamp_min(1e-8))
    ).sum(dim=-1)
    return aligned, entropy


def temporal_onset_components(
    current: torch.Tensor,
    aligned_history: torch.Tensor,
    history_weights: torch.Tensor,
    history_valid: torch.Tensor,
    *,
    normality_scale: float,
) -> dict[str, torch.Tensor]:
    """Contrast current change against explicit history-history variation."""

    if current.ndim != 3 or aligned_history.ndim != 4:
        raise ValueError("current/history need [B,P,D]/[B,H,P,D]")
    batch, histories, patches, dimension = aligned_history.shape
    if current.shape != (batch, patches, dimension):
        raise ValueError("current and aligned history shapes differ")
    if history_weights.shape != (batch, histories):
        raise ValueError("history_weights must have shape [B,H]")
    if history_valid.shape != (batch, histories):
        raise ValueError("history_valid must have shape [B,H]")

    valid = history_valid.bool()
    weights = torch.where(valid, history_weights, torch.zeros_like(history_weights))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    center = (aligned_history * weights[:, :, None, None]).sum(dim=1)
    signed = current - center
    current_absolute = (
        (current[:, None] - aligned_history).abs()
        * weights[:, :, None, None]
    ).sum(dim=1)

    # Explicit pairwise historical variation, not merely feature variance.
    normal_sum = torch.zeros_like(current_absolute)
    pair_mass = torch.zeros(batch, 1, 1, device=current.device)
    for left in range(histories):
        for right in range(left + 1, histories):
            pair_weight = (
                weights[:, left] * weights[:, right]
            ).view(batch, 1, 1)
            pair_ok = (
                valid[:, left] & valid[:, right]
            ).to(pair_weight.dtype).view(batch, 1, 1)
            pair_weight = pair_weight * pair_ok
            normal_sum = normal_sum + pair_weight * (
                aligned_history[:, left] - aligned_history[:, right]
            ).abs()
            pair_mass = pair_mass + pair_weight
    normality = normal_sum / pair_mass.clamp_min(1e-8)
    normality = torch.where(pair_mass.gt(0), normality, torch.zeros_like(normality))
    excess = F.relu(current_absolute - float(normality_scale) * normality)
    ratio = (
        current_absolute / (normality + 0.05)
    ).clamp(max=10.0) / 10.0
    has_history = valid.any(dim=1)
    history_multiplier = has_history[:, None, None].to(current.dtype)
    center = center * history_multiplier
    signed = signed * history_multiplier
    current_absolute = current_absolute * history_multiplier
    normality = normality * history_multiplier
    excess = excess * history_multiplier
    ratio = ratio * history_multiplier
    return {
        "history_center": center,
        "signed_change": signed,
        "current_absolute": current_absolute,
        "history_normality": normality,
        "excess": excess,
        "ratio": ratio,
        "has_history": has_history,
    }


class TemporalQualityGate(nn.Module):
    """Gate irregular history using real lag, season and valid fraction."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        delta_days: torch.Tensor,
        quality: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if delta_days.shape != quality.shape or delta_days.shape != valid.shape:
            raise ValueError("delta_days, quality and valid must share [B,H]")
        finite = torch.isfinite(delta_days)
        safe = torch.where(finite, delta_days, torch.zeros_like(delta_days)).float()
        log_lag = torch.log1p(safe.abs()).clamp(max=math.log1p(4000.0))
        log_lag = log_lag / math.log1p(4000.0)
        annual = 2.0 * math.pi * safe / 365.2425
        features = torch.stack(
            (
                torch.sign(safe) * log_lag,
                log_lag,
                torch.sin(annual),
                torch.cos(annual),
                quality.float().clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        usable = valid.bool() & finite
        logits = self.mlp(features).squeeze(-1)
        return _masked_softmax(logits, usable, dim=1)


def assemble_onset_features(
    component: Mapping[str, torch.Tensor],
    *,
    use_normality_features: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Assemble matched-width onset inputs for the normality ablation.

    The no-normality arm preserves parameter shape and initialization but
    zeroes both normality-derived channels and defines excess as the unadjusted
    current absolute change.  It therefore cannot recover normality through a
    nominally "disabled" input channel.
    """

    signed = component["signed_change"]
    current_absolute = component["current_absolute"]
    if use_normality_features:
        normality = component["history_normality"]
        excess = component["excess"]
        ratio = component["ratio"]
    else:
        normality = torch.zeros_like(current_absolute)
        excess = current_absolute
        ratio = torch.zeros_like(current_absolute)
    features = torch.cat(
        (signed, current_absolute, normality, excess, ratio), dim=-1
    )
    return features, {
        "history_normality": normality,
        "excess": excess,
        "ratio": ratio,
    }


@dataclass
class TempoOutput:
    logits: torch.Tensor
    residual: torch.Tensor
    mil_logit: torch.Tensor
    patch_scores: torch.Tensor
    history_weights: torch.Tensor
    history_normality: torch.Tensor
    excess: torch.Tensor
    topk_count: int


class TempoPatchHead(nn.Module):
    """Patch-local background-memory onset head with exact base fallback."""

    def __init__(
        self,
        feature_dim: int = 64,
        *,
        match_rank: int = 16,
        value_dim: int = 32,
        hidden_dim: int = 64,
        radius: int = 1,
        temperature: float = 0.10,
        topk_fraction: float = 0.10,
        normality_scale: float = 1.0,
        use_normality_features: bool = True,
        residual_cap: float = 1.5,
        zero_init_mode: str = "scalar",
    ):
        super().__init__()
        if feature_dim < 1 or match_rank < 1 or value_dim < 1:
            raise ValueError("feature dimensions must be positive")
        if radius not in {0, 1, 2}:
            raise ValueError("radius must be 0, 1, or 2")
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0,1]")
        if residual_cap <= 0:
            raise ValueError("residual_cap must be positive")
        if zero_init_mode not in {"scalar", "readout"}:
            raise ValueError("zero_init_mode must be 'scalar' or 'readout'")
        self.feature_dim = int(feature_dim)
        self.radius = int(radius)
        self.temperature = float(temperature)
        self.topk_fraction = float(topk_fraction)
        self.normality_scale = float(normality_scale)
        self.use_normality_features = bool(use_normality_features)
        self.residual_cap = float(residual_cap)
        self.zero_init_mode = str(zero_init_mode)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.query = nn.Linear(self.feature_dim, int(match_rank), bias=False)
        self.key = nn.Linear(self.feature_dim, int(match_rank), bias=False)
        self.value = nn.Linear(self.feature_dim, int(value_dim), bias=False)
        self.temporal_gate = TemporalQualityGate()
        onset_dim = int(value_dim) * 5
        self.onset = nn.Sequential(
            nn.LayerNorm(onset_dim),
            nn.Linear(onset_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.patch_readout = nn.Linear(int(hidden_dim), 1)
        if self.zero_init_mode == "scalar":
            # Original behavior: a zero scalar preserves random local
            # features/readout and receives the only first-step gradient.
            self.residual_gate = nn.Parameter(torch.zeros(()))
        else:
            # Stability control: remove the random readout-sign ambiguity.
            # The first step updates the zero readout directly while the
            # fixed unit gate keeps epoch zero bit-exact to the base.
            nn.init.zeros_(self.patch_readout.weight)
            nn.init.zeros_(self.patch_readout.bias)
            self.residual_gate = nn.Parameter(
                torch.ones(()), requires_grad=False
            )

    @property
    def exact_noop(self) -> bool:
        if self.zero_init_mode == "scalar":
            return bool(float(self.residual_gate.detach()) == 0.0)
        return bool(
            int(torch.count_nonzero(self.patch_readout.weight.detach())) == 0
            and int(torch.count_nonzero(self.patch_readout.bias.detach())) == 0
        )

    def forward(
        self,
        base_logits: torch.Tensor,
        patch_tokens: torch.Tensor,
        unique_mask: torch.Tensor,
        delta_days: torch.Tensor,
        quality: torch.Tensor,
        *,
        t0_index: int,
        grid_shape: tuple[int, int],
    ) -> TempoOutput:
        if patch_tokens.ndim != 4:
            raise ValueError("patch_tokens must have shape [B,T,P,D]")
        batch, visits, patches, dimension = patch_tokens.shape
        if dimension != self.feature_dim:
            raise ValueError("patch feature dimension differs from head")
        if base_logits.shape != (batch,):
            raise ValueError("base_logits must have shape [B]")
        for name, value in (
            ("unique_mask", unique_mask),
            ("delta_days", delta_days),
            ("quality", quality),
        ):
            if value.shape != (batch, visits):
                raise ValueError(f"{name} must have shape [B,T]")
        if not 0 <= int(t0_index) < visits:
            raise ValueError("t0_index is out of range")
        if not unique_mask[:, int(t0_index)].bool().all():
            raise ValueError("every TEMPO row requires a unique valid t0")
        history_indices = [
            index for index in range(visits) if index != int(t0_index)
        ]
        if not history_indices:
            raise ValueError("TEMPO needs at least one history visit")

        tokens = self.input_norm(patch_tokens.float())
        current_token = tokens[:, int(t0_index)]
        history_token = tokens[:, history_indices]
        current_value = self.value(current_token)
        aligned, _entropy = local_soft_match(
            self.query(current_token),
            self.key(history_token),
            self.value(history_token),
            grid_shape=grid_shape,
            radius=self.radius,
            temperature=self.temperature,
        )
        history_valid = unique_mask[:, history_indices].bool()
        history_weights = self.temporal_gate(
            delta_days[:, history_indices],
            quality[:, history_indices],
            history_valid,
        )
        component = temporal_onset_components(
            current_value,
            aligned,
            history_weights,
            history_valid,
            normality_scale=self.normality_scale,
        )
        onset_input, effective = assemble_onset_features(
            component,
            use_normality_features=self.use_normality_features,
        )
        hidden = self.onset(onset_input)
        patch_scores = self.patch_readout(hidden).squeeze(-1)
        patch_scores = patch_scores * component["has_history"][:, None].to(
            patch_scores.dtype
        )
        keep = max(1, int(math.ceil(patches * self.topk_fraction)))
        top_scores = torch.topk(
            patch_scores, k=keep, dim=1, largest=True
        ).values
        mil_logit = top_scores.mean(dim=1)
        raw_residual = self.residual_gate * mil_logit
        residual = self.residual_cap * torch.tanh(
            raw_residual / self.residual_cap
        )
        return TempoOutput(
            logits=base_logits.float() + residual,
            residual=residual,
            mil_logit=mil_logit,
            patch_scores=patch_scores,
            history_weights=history_weights,
            history_normality=effective["history_normality"],
            excess=effective["excess"],
            topk_count=keep,
        )


def all_negative_event_mask(
    labels: torch.Tensor,
    event_ids: Sequence[str],
) -> torch.Tensor:
    """Mark rows whose complete canonical event contains no positive row."""

    if labels.ndim != 1 or len(labels) != len(event_ids):
        raise ValueError("labels/event_ids differ in length")
    event_positive: dict[str, bool] = {}
    for label, event_id in zip(labels.long().tolist(), event_ids):
        key = str(event_id)
        if not key:
            raise ValueError("event IDs must be nonblank")
        event_positive[key] = event_positive.get(key, False) or bool(label)
    return torch.tensor(
        [not event_positive[str(event_id)] for event_id in event_ids],
        dtype=torch.bool,
    )


def negative_event_topk_null_loss(
    patch_scores: torch.Tensor,
    negative_event_rows: torch.Tensor,
    *,
    topk_fraction: float,
    margin: float = 0.0,
) -> torch.Tensor:
    """Penalize positive local evidence on entirely negative train events."""

    if patch_scores.ndim != 2:
        raise ValueError("patch_scores must have shape [B,P]")
    if negative_event_rows.shape != (patch_scores.shape[0],):
        raise ValueError("negative_event_rows must have shape [B]")
    selected = patch_scores[negative_event_rows.bool()]
    if selected.numel() == 0:
        return patch_scores.sum() * 0.0
    keep = max(1, int(math.ceil(selected.shape[1] * float(topk_fraction))))
    top = torch.topk(selected, k=keep, dim=1, largest=True).values
    return F.softplus(top - float(margin)).mean()


def _resource_guard(
    device: torch.device,
    *,
    max_allocated_gib: float,
    min_free_gib: float,
) -> int:
    if device.type != "cuda":
        return 0
    allocated = int(torch.cuda.max_memory_allocated(device))
    if allocated > float(max_allocated_gib) * (2**30):
        raise RuntimeError(
            f"TEMPO allocation {allocated / 2**30:.2f} GiB exceeds "
            f"{max_allocated_gib:.2f} GiB cap"
        )
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < float(min_free_gib) * (2**30):
        raise RuntimeError(
            f"GPU free memory {free_bytes / 2**30:.2f} GiB fell below "
            f"{min_free_gib:.2f} GiB"
        )
    return allocated


def _validate_local_inputs(
    frame: pd.DataFrame,
    path_columns: Sequence[str],
    *,
    required_root: Path,
) -> dict[str, Any]:
    return patch_followup._validate_all_local_paths(
        frame, path_columns, allowed_root=required_root
    )


def _manifest_configuration(
    *,
    args: argparse.Namespace,
    csv_path: Path,
    cls_cache_path: Path,
    weights_path: Path,
    base_head_path: Path,
    cls_cache: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "split": str(args.split),
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "cls_cache": str(cls_cache_path),
        "cls_cache_sha256": sha256_file(cls_cache_path),
        "weights": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "base_head": str(base_head_path),
        "base_head_sha256": sha256_file(base_head_path),
        "rows": int(cls_cache["features"].shape[0]),
        "timepoints": int(cls_cache["features"].shape[1]),
        "t0_index": int(cls_cache["t0_index"]),
        "projection": dict(projection),
        "projection_storage_dtype": "torch.float16",
        "shard_rows": int(args.shard_rows),
        "amp_dtype": str(args.amp_dtype),
        "input_contract_sha256": str(cls_cache["input_contract_sha256"]),
        "comparable_input_contract_sha256": canonical_json_sha256(
            l89.comparable_input_contract(cls_cache["input_contract"])
        ),
        "raw_768d_patch_tokens_persisted": False,
        "test_or_sealed_read": False,
    }


def _validate_existing_shard(
    path: Path,
    record: Mapping[str, Any],
    *,
    config_sha256: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"shard file SHA mismatch: {path}")
    payload = _load_torch_payload(path)
    if payload.get("format_version") != SHARD_VERSION:
        raise ValueError(f"unsupported shard format: {path}")
    if payload.get("configuration_sha256") != config_sha256:
        raise ValueError(f"shard configuration differs: {path}")
    projected = payload["patch_tokens"]
    if projected.dtype != torch.float16 or projected.ndim != 4:
        raise ValueError(f"invalid projected patch tensor: {path}")
    if tensor_sha256(projected) != payload["tensor_sha256"]["patch_tokens"]:
        raise ValueError(f"projected patch tensor SHA mismatch: {path}")
    if bool(payload.get("raw_768d_patch_tokens_persisted", True)):
        raise ValueError(f"raw patch-token persistence is forbidden: {path}")


def command_extract(args: argparse.Namespace) -> None:
    """Stream frozen final patch tokens into resumable projected shards."""

    set_seed(args.seed)
    csv_path = Path(args.csv).expanduser().resolve()
    cls_cache_path = Path(args.cls_cache).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    base_head_path = Path(args.base_head_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    required_root = Path(args.required_local_root).expanduser().resolve()
    for path, purpose in (
        (csv_path, "CSV"),
        (cls_cache_path, "CLS cache"),
        (weights_path, "weights"),
        (base_head_path, "base head"),
        (output_dir, "output"),
        (required_root, "required local root"),
    ):
        assert_development_path(path, purpose=purpose)
    for path in (csv_path, cls_cache_path, weights_path, base_head_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if int(args.shard_rows) < 1:
        raise ValueError("--shard-rows must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    cls_cache = _load_torch_payload(cls_cache_path)
    l89.validate_cache_payload(
        cls_cache, path=cls_cache_path, expected_split=args.split
    )
    if str(cls_cache["csv_path"]) != str(csv_path):
        raise ValueError("CLS cache CSV path differs from --csv")
    if str(cls_cache["weights_path"]) != str(weights_path):
        raise ValueError("CLS cache weights path differs from --weights")
    if sha256_file(csv_path) != str(cls_cache["csv_sha256"]):
        raise ValueError("CSV SHA differs from audited CLS cache")
    if sha256_file(weights_path) != str(cls_cache["weights_sha256"]):
        raise ValueError("weights SHA differs from audited CLS cache")
    unique_mask = cls_cache["unique_mask"].bool()
    t0_index = int(cls_cache["t0_index"])
    if not unique_mask[:, t0_index].all():
        raise ValueError("TEMPO cache requires every row to have a unique valid t0")

    frame = pd.read_csv(csv_path, low_memory=False)
    if len(frame) != int(cls_cache["features"].shape[0]):
        raise ValueError("CSV and CLS cache row counts differ")
    ids = l89.string_column(
        frame, "id", fallback=l89.string_column(frame, "plume_id")
    )
    if [str(value) for value in ids] != [
        str(value) for value in cls_cache["ids"]
    ]:
        raise ValueError("CSV and CLS cache row identities differ")
    contract = cls_cache["input_contract"]
    path_columns = tuple(str(value) for value in contract["path_columns"])
    local_audit = _validate_local_inputs(
        frame, path_columns, required_root=required_root
    )
    dataset = l89.L89FrameCacheDataset(
        csv_path,
        frame,
        path_columns=path_columns,
        band_indices=tuple(int(value) for value in contract["band_indices"]),
        mean=tuple(float(value) for value in contract["normalization_mean"]),
        std=tuple(float(value) for value in contract["normalization_std"]),
        image_size=int(contract["image_size"]),
        min_valid_fraction=float(contract["min_valid_fraction"]),
        validity_band_index=int(contract["validity_band_index"]),
        local_file_cache=None,
        local_cache_bypass_root=required_root,
        zero_invalid_pixels=bool(contract["zero_invalid_pixels"]),
    )
    base_checkpoint = _load_torch_payload(base_head_path)
    base_logits = patch_followup.frozen_base_logits(
        base_checkpoint,
        cls_cache,
        batch_size=int(args.base_eval_batch_size),
    )

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < float(args.min_cuda_free_gib) * (2**30):
            raise RuntimeError(
                f"{device} has only {free_bytes / 2**30:.2f} GiB free"
            )
    backbone = load_backbone(
        str(weights_path), device=device, debug=bool(args.debug)
    ).to(device)
    backbone.requires_grad_(False)
    backbone.eval()
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise AssertionError("TEMPO extraction backbone must remain frozen")
    projection_mode = str(
        getattr(args, "projection_mode", "gaussian_qr")
    )
    if projection_mode == "gaussian_qr":
        projection = deterministic_orthogonal_projection(
            int(backbone.embed_dim),
            int(args.projection_dim),
            seed=int(args.projection_seed),
        )
        projection_info = projection_metadata(
            projection, seed=int(args.projection_seed)
        )
    elif projection_mode == "nested_extend_128":
        if int(args.projection_dim) != 256:
            raise ValueError(
                "nested_extend_128 requires --projection-dim 256"
            )
        extension_seed = int(
            getattr(args, "projection_extension_seed", 36192)
        )
        projection = nested_extend_orthogonal_projection(
            int(backbone.embed_dim),
            int(args.projection_dim),
            base_dim=128,
            base_seed=int(args.projection_seed),
            extension_seed=extension_seed,
        )
        projection_info = nested_projection_metadata(
            projection,
            base_dim=128,
            base_seed=int(args.projection_seed),
            extension_seed=extension_seed,
        )
        if not bool(projection_info["first_base_columns_array_equal"]):
            raise AssertionError("nested projection prefix audit failed")
    else:
        raise ValueError(f"unknown projection mode {projection_mode!r}")
    configuration = _manifest_configuration(
        args=args,
        csv_path=csv_path,
        cls_cache_path=cls_cache_path,
        weights_path=weights_path,
        base_head_path=base_head_path,
        cls_cache=cls_cache,
        projection=projection_info,
    )
    configuration_sha = canonical_json_sha256(configuration)

    existing: dict[int, Mapping[str, Any]] = {}
    resume_grid_shape: Optional[tuple[int, int]] = None
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{manifest_path} exists; pass --resume or choose a new directory"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != CACHE_VERSION:
            raise ValueError("existing manifest has unsupported format")
        if manifest.get("configuration_sha256") != configuration_sha:
            raise ValueError("resume manifest configuration differs")
        manifest_records = list(manifest.get("shards", []))
        manifest_grid = tuple(int(value) for value in manifest["grid_shape"])
        if manifest_records:
            if len(manifest_grid) != 2 or min(manifest_grid) < 1:
                raise ValueError("resume manifest has an invalid patch grid")
            resume_grid_shape = manifest_grid
        elif manifest_grid != (0, 0):
            raise ValueError("empty resume manifest must have grid [0,0]")
        for record in manifest_records:
            shard_index = int(record["shard_index"])
            if shard_index in existing:
                raise ValueError(f"duplicate resume shard index {shard_index}")
            shard_path = output_dir / str(record["file"])
            _validate_existing_shard(
                shard_path, record, config_sha256=configuration_sha
            )
            existing[shard_index] = record
    elif any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is nonempty but has no compatible manifest"
        )

    rows = len(frame)
    shard_ranges = [
        (index, start, min(start + int(args.shard_rows), rows))
        for index, start in enumerate(range(0, rows, int(args.shard_rows)))
    ]
    records: dict[int, Mapping[str, Any]] = dict(existing)
    started = time.monotonic()
    maximum_allocated = 0
    grid_shape: Optional[tuple[int, int]] = resume_grid_shape
    projection_device = projection.to(device)
    channel_ids = dataset.channel_ids
    identity = {
        "ids": [str(value) for value in cls_cache["ids"]],
        "plume_ids": [str(value) for value in cls_cache["plume_ids"]],
        "event_ids": [str(value) for value in cls_cache["event_ids"]],
        "labels": cls_cache["labels"].long().tolist(),
    }
    identity["identity_sha256"] = canonical_json_sha256(
        {
            "ids": identity["ids"],
            "event_ids": identity["event_ids"],
            "labels": identity["labels"],
        }
    )

    def current_manifest(*, status: str) -> dict[str, Any]:
        return {
            "format_version": CACHE_VERSION,
            "script_version": SCRIPT_VERSION,
            "status": status,
            "configuration": configuration,
            "configuration_sha256": configuration_sha,
            "grid_shape": list(grid_shape or (0, 0)),
            "rows": rows,
            "shards": [
                dict(records[index]) for index in sorted(records)
            ],
            "identity": identity,
            "provenance": {
                "input_audit": local_audit,
                "backbone_trainable_parameters": 0,
                "token_layer": "x_norm_patchtokens_final",
                "base_head_arm": str(base_checkpoint.get("arm")),
                "base_head_epoch": int(base_checkpoint.get("epoch", -1)),
                "cuda_max_memory_allocated_bytes": int(maximum_allocated),
                "elapsed_seconds": float(time.monotonic() - started),
                "raw_768d_patch_tokens_persisted": False,
                "test_or_sealed_read": False,
            },
        }

    # Commit the configuration before the first shard.  If power is lost
    # between a shard rename and its manifest update, resume simply rewrites
    # that one unregistered shard instead of leaving an unusable directory.
    if not manifest_path.exists():
        atomic_json(manifest_path, current_manifest(status="incomplete"))

    for shard_index, start, stop in shard_ranges:
        if shard_index in existing:
            record = existing[shard_index]
            if (
                int(record["row_start"]) != start
                or int(record["row_stop"]) != stop
            ):
                raise ValueError(
                    f"resume shard {shard_index} has wrong row range"
                )
            print(
                f"[tempo-cache] resume skip shard={shard_index} rows={start}:{stop}",
                flush=True,
            )
            continue
        subset = Subset(dataset, list(range(start, stop)))
        loader_options: dict[str, Any] = {
            "dataset": subset,
            "batch_size": int(args.batch_size),
            "shuffle": False,
            "num_workers": int(args.num_workers),
            "pin_memory": device.type == "cuda",
        }
        if int(args.num_workers) > 0:
            loader_options["prefetch_factor"] = int(args.prefetch_factor)
            loader_options["persistent_workers"] = False
        loader = DataLoader(**loader_options)
        projected_parts: list[torch.Tensor] = []
        index_parts: list[torch.Tensor] = []
        quality_parts: list[torch.Tensor] = []
        status_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for indices, images, image_valid, fractions, status in loader:
                indices = indices.long()
                if not torch.equal(
                    image_valid.bool(),
                    cls_cache["image_valid_mask"][indices].bool(),
                ):
                    raise ValueError(
                        "online image validity differs from audited CLS cache"
                    )
                batch_rows, visits, channels, height, width = images.shape
                flat_images = images.reshape(
                    batch_rows * visits, channels, height, width
                ).to(device, non_blocking=True)
                flat_ids = (
                    channel_ids.view(1, -1)
                    .expand(batch_rows * visits, -1)
                    .clone()
                    .to(device, non_blocking=True)
                )
                with autocast_context(device, args.amp_dtype):
                    output = backbone.forward_features(
                        {"imgs": flat_images, "chn_ids": flat_ids}
                    )
                    raw = output["x_norm_patchtokens"].reshape(
                        batch_rows,
                        visits,
                        -1,
                        int(backbone.embed_dim),
                    )
                    projected = torch.matmul(
                        raw, projection_device.to(dtype=raw.dtype)
                    )
                patch_count = int(projected.shape[2])
                side = int(math.isqrt(patch_count))
                if side * side != patch_count:
                    raise ValueError(
                        f"TEMPO requires a square patch grid, got P={patch_count}"
                    )
                current_grid = (side, side)
                if grid_shape is None:
                    grid_shape = current_grid
                elif grid_shape != current_grid:
                    raise ValueError("patch grid changed between batches")
                projected = projected.masked_fill(
                    ~unique_mask[indices].to(device)[:, :, None, None],
                    0.0,
                )
                projected_parts.append(projected.cpu().to(torch.float16))
                index_parts.append(indices)
                quality_parts.append(fractions.to(torch.float16))
                status_parts.append(status.to(torch.int8))
                maximum_allocated = max(
                    maximum_allocated,
                    _resource_guard(
                        device,
                        max_allocated_gib=float(args.max_cuda_allocated_gib),
                        min_free_gib=float(args.min_cuda_free_gib),
                    ),
                )
        row_indices = torch.cat(index_parts)
        expected_indices = torch.arange(start, stop, dtype=torch.long)
        if not torch.equal(row_indices, expected_indices):
            raise RuntimeError("shard loader changed canonical row order")
        projected_shard = torch.cat(projected_parts)
        quality_shard = torch.cat(quality_parts)
        status_shard = torch.cat(status_parts)
        shard_payload = {
            "format_version": SHARD_VERSION,
            "script_version": SCRIPT_VERSION,
            "configuration_sha256": configuration_sha,
            "split": str(args.split),
            "shard_index": int(shard_index),
            "row_start": int(start),
            "row_stop": int(stop),
            "row_indices": row_indices,
            "patch_tokens": projected_shard,
            "base_logits": base_logits[start:stop].float(),
            "labels": cls_cache["labels"][start:stop].long(),
            "unique_mask": unique_mask[start:stop],
            "delta_days": cls_cache["delta_days"][start:stop].float(),
            "quality": quality_shard,
            "load_status": status_shard,
            "ids": [str(value) for value in cls_cache["ids"][start:stop]],
            "plume_ids": [
                str(value) for value in cls_cache["plume_ids"][start:stop]
            ],
            "event_ids": [
                str(value) for value in cls_cache["event_ids"][start:stop]
            ],
            "grid_shape": list(grid_shape or (0, 0)),
            "projection": projection_info,
            "raw_768d_patch_tokens_persisted": False,
            "tensor_sha256": {
                "patch_tokens": tensor_sha256(projected_shard),
                "base_logits": tensor_sha256(base_logits[start:stop].float()),
                "labels": tensor_sha256(cls_cache["labels"][start:stop].long()),
                "unique_mask": tensor_sha256(unique_mask[start:stop]),
                "delta_days": tensor_sha256(
                    cls_cache["delta_days"][start:stop].float()
                ),
                "quality": tensor_sha256(quality_shard),
            },
            "test_or_sealed_read": False,
        }
        shard_name = f"shard_{shard_index:05d}_{start:07d}_{stop:07d}.pt"
        shard_path = output_dir / shard_name
        atomic_torch(shard_path, shard_payload)
        record = {
            "shard_index": int(shard_index),
            "row_start": int(start),
            "row_stop": int(stop),
            "rows": int(stop - start),
            "file": shard_name,
            "sha256": sha256_file(shard_path),
            "patch_tokens_sha256": shard_payload["tensor_sha256"][
                "patch_tokens"
            ],
            "shape": list(projected_shard.shape),
        }
        records[shard_index] = record
        partial_manifest = current_manifest(
            status=(
                "complete"
                if len(records) == len(shard_ranges)
                else "incomplete"
            )
        )
        atomic_json(manifest_path, partial_manifest)
        print(
            f"[tempo-cache] wrote shard={shard_index} rows={start}:{stop} "
            f"shape={tuple(projected_shard.shape)}",
            flush=True,
        )

    final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(final_manifest["shards"]) != len(shard_ranges):
        raise RuntimeError("cache extraction ended with incomplete shard coverage")
    final_manifest["status"] = "complete"
    final_manifest["manifest_content_sha256"] = canonical_json_sha256(
        {
            "configuration_sha256": configuration_sha,
            "shards": final_manifest["shards"],
            "identity_sha256": final_manifest["identity"]["identity_sha256"],
        }
    )
    atomic_json(manifest_path, final_manifest)
    print(json.dumps(final_manifest, indent=2, sort_keys=True), flush=True)


def load_manifest(path: Path, *, expected_split: str) -> dict[str, Any]:
    assert_development_path(path, purpose=f"{expected_split} cache")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != CACHE_VERSION:
        raise ValueError(f"{path}: unsupported TEMPO cache format")
    if payload.get("status") != "complete":
        raise ValueError(f"{path}: cache is not complete")
    configuration_sha = canonical_json_sha256(payload["configuration"])
    if configuration_sha != payload.get("configuration_sha256"):
        raise ValueError(f"{path}: configuration SHA mismatch")
    if payload.get("configuration", {}).get("split") != expected_split:
        raise ValueError(f"{path}: expected split {expected_split}")
    if payload.get("configuration", {}).get(
        "raw_768d_patch_tokens_persisted"
    ) is not False:
        raise ValueError(f"{path}: raw patch persistence contract violated")
    rows = int(payload["rows"])
    expected_start = 0
    for record in sorted(payload["shards"], key=lambda item: item["shard_index"]):
        if int(record["row_start"]) != expected_start:
            raise ValueError(f"{path}: shard coverage is not contiguous")
        expected_start = int(record["row_stop"])
        shard_path = path.parent / str(record["file"])
        _validate_existing_shard(
            shard_path,
            record,
            config_sha256=str(payload["configuration_sha256"]),
        )
    if expected_start != rows:
        raise ValueError(f"{path}: shard coverage stops at {expected_start}/{rows}")
    identity = payload["identity"]
    if any(
        len(identity[key]) != rows
        for key in ("ids", "plume_ids", "event_ids", "labels")
    ):
        raise ValueError(f"{path}: manifest identity length mismatch")
    expected_identity_sha = canonical_json_sha256(
        {
            "ids": [str(value) for value in identity["ids"]],
            "event_ids": [str(value) for value in identity["event_ids"]],
            "labels": [int(value) for value in identity["labels"]],
        }
    )
    if expected_identity_sha != identity.get("identity_sha256"):
        raise ValueError(f"{path}: identity SHA mismatch")
    expected_content_sha = canonical_json_sha256(
        {
            "configuration_sha256": configuration_sha,
            "shards": payload["shards"],
            "identity_sha256": expected_identity_sha,
        }
    )
    if expected_content_sha != payload.get("manifest_content_sha256"):
        raise ValueError(f"{path}: manifest content SHA mismatch")
    return payload


def validate_cache_pair(
    train_manifest_path: Path,
    val_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train = load_manifest(train_manifest_path, expected_split="train")
    val = load_manifest(val_manifest_path, expected_split="val")
    train_config = train["configuration"]
    val_config = val["configuration"]
    for key in (
        "weights_sha256",
        "base_head_sha256",
        "projection",
        "timepoints",
        "t0_index",
        "comparable_input_contract_sha256",
        "projection_storage_dtype",
        "amp_dtype",
    ):
        if train_config[key] != val_config[key]:
            raise ValueError(f"train/val cache mismatch for {key}")
    if train["grid_shape"] != val["grid_shape"]:
        raise ValueError("train/val patch grids differ")
    overlap = sorted(
        set(train["identity"]["event_ids"])
        & set(val["identity"]["event_ids"])
    )
    if overlap:
        raise ValueError(
            f"train/val overlap by {len(overlap)} canonical events; "
            f"examples={overlap[:10]}"
        )
    audit = {
        "train_manifest": str(train_manifest_path),
        "train_manifest_sha256": sha256_file(train_manifest_path),
        "val_manifest": str(val_manifest_path),
        "val_manifest_sha256": sha256_file(val_manifest_path),
        "event_overlap": 0,
        "projection": train_config["projection"],
        "grid_shape": train["grid_shape"],
        "test_or_sealed_read": False,
    }
    return train, val, audit


def _verify_overlay_identity(
    manifest: Mapping[str, Any],
    *,
    ids: Sequence[str],
    labels: Sequence[int],
    event_ids: Sequence[str],
    purpose: str,
) -> None:
    identity = manifest["identity"]
    if [str(value) for value in ids] != [
        str(value) for value in identity["ids"]
    ]:
        raise ValueError(f"{purpose}: IDs differ from patch manifest")
    if [int(value) for value in labels] != [
        int(value) for value in identity["labels"]
    ]:
        raise ValueError(f"{purpose}: labels differ from patch manifest")
    if [str(value) for value in event_ids] != [
        str(value) for value in identity["event_ids"]
    ]:
        raise ValueError(f"{purpose}: canonical event IDs differ")


def _head_hyperparameters(
    checkpoint: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    config_path: Optional[Path],
) -> dict[str, Any]:
    source: Mapping[str, Any]
    if config_path is not None:
        source = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        raw = checkpoint.get("args")
        if not isinstance(raw, Mapping):
            raise ValueError(
                "head checkpoint has no embedded args; provide "
                "--head-config-json"
            )
        source = raw
    input_weight = state.get("input_projection.weight")
    role_weight = state.get("role_embedding.weight")
    if input_weight is None or role_weight is None:
        raise ValueError("head state lacks input/role projection weights")
    block_indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    if not block_indices:
        raise ValueError("head state has no temporal blocks")
    periods_tensor = state.get("delta_encoder.periods_days")
    if periods_tensor is not None:
        periods = tuple(float(value) for value in periods_tensor.tolist())
    else:
        raw_periods = source.get("delta_periods", "7,30,90,180,365,730")
        periods = tuple(
            float(value)
            for value in (
                raw_periods.split(",")
                if isinstance(raw_periods, str)
                else raw_periods
            )
        )
    return {
        "feature_dim": int(input_weight.shape[1]),
        "num_roles": int(role_weight.shape[0]),
        "model_dim": int(input_weight.shape[0]),
        "num_heads": int(source["num_heads"]),
        "depth": int(max(block_indices) + 1),
        "mlp_ratio": float(source["mlp_ratio"]),
        # Dropout has no effect in exact eval replay but is recorded.
        "dropout": float(source.get("dropout", 0.0)),
        "periods_days": periods,
    }


def replay_ragged_head_logits(
    *,
    feature_cache_path: Path,
    head_checkpoint_path: Path,
    head_config_path: Optional[Path],
    manifest: Mapping[str, Any],
    split: str,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exactly replay a frozen role-only head on a compatible CLS cache."""

    for path, purpose in (
        (feature_cache_path, "rebase feature cache"),
        (head_checkpoint_path, "rebase head checkpoint"),
    ):
        assert_development_path(path, purpose=purpose)
        if not path.is_file():
            raise FileNotFoundError(path)
    if head_config_path is not None:
        assert_development_path(head_config_path, purpose="head config")
        if not head_config_path.is_file():
            raise FileNotFoundError(head_config_path)
    feature_cache = _load_torch_payload(feature_cache_path)
    l89.validate_cache_payload(
        feature_cache,
        path=feature_cache_path,
        expected_split=split,
    )
    _verify_overlay_identity(
        manifest,
        ids=feature_cache["ids"],
        labels=feature_cache["labels"].long().tolist(),
        event_ids=feature_cache["event_ids"],
        purpose="feature-cache replay",
    )
    checkpoint = _load_torch_payload(head_checkpoint_path)
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("head checkpoint lacks a model state")
    hyperparameters = _head_hyperparameters(
        checkpoint, state, head_config_path
    )
    if int(feature_cache["features"].shape[-1]) != int(
        hyperparameters["feature_dim"]
    ):
        raise ValueError("head feature dimension differs from feature cache")
    if int(feature_cache["features"].shape[1]) != int(
        hyperparameters["num_roles"]
    ):
        raise ValueError("head role count differs from feature cache")
    model = l89.RaggedCurrentQueryHead(
        feature_dim=hyperparameters["feature_dim"],
        num_roles=hyperparameters["num_roles"],
        model_dim=hyperparameters["model_dim"],
        num_heads=hyperparameters["num_heads"],
        depth=hyperparameters["depth"],
        mlp_ratio=hyperparameters["mlp_ratio"],
        dropout=hyperparameters["dropout"],
        periods_days=hyperparameters["periods_days"],
        t0_index=int(feature_cache["t0_index"]),
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    rows = int(feature_cache["features"].shape[0])
    role_index = feature_cache["role_index"].long()
    t0_index = int(feature_cache["t0_index"])
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for indices in l89.fixed_epoch_batches(
            rows,
            batch_size=int(batch_size),
            seed=0,
            epoch=0,
            shuffle=False,
        ):
            features, valid, delta_days, enable_delta = l89.prepare_arm_inputs(
                feature_cache["features"][indices].float(),
                feature_cache["valid_mask"][indices].bool(),
                feature_cache["unique_mask"][indices].bool(),
                feature_cache["delta_days"][indices].float(),
                arm="role_only",
                t0_index=t0_index,
            )
            outputs.append(
                model(
                    features,
                    valid,
                    role_index,
                    delta_days,
                    enable_delta=enable_delta,
                ).float()
            )
    logits = torch.cat(outputs).contiguous()
    if logits.shape != (rows,) or not torch.isfinite(logits).all():
        raise RuntimeError("head replay produced invalid logits")
    feature_weights_sha = str(feature_cache.get("weights_sha256", ""))
    config_sha = (
        sha256_file(head_config_path) if head_config_path is not None else ""
    )
    provenance = {
        "source_kind": "exact_checkpoint_replay",
        "feature_cache": str(feature_cache_path),
        "feature_cache_sha256": sha256_file(feature_cache_path),
        "feature_weights_sha256": feature_weights_sha,
        "head_checkpoint": str(head_checkpoint_path),
        "head_checkpoint_sha256": sha256_file(head_checkpoint_path),
        "head_config": (
            str(head_config_path) if head_config_path is not None else None
        ),
        "head_config_sha256": config_sha or None,
        "head_hyperparameters": hyperparameters,
        "checkpoint_arm": str(checkpoint.get("arm", "")),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_state_sha256": l89.state_dict_sha256(state),
        "probability_to_logit_conversion": None,
    }
    return logits, provenance


def logits_from_prediction_csv(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    probability_column: str,
    clip: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Create float32 logits from a strictly identity-bound prediction CSV."""

    assert_development_path(path, purpose="rebase prediction CSV")
    if not path.is_file():
        raise FileNotFoundError(path)
    if not 0.0 < float(clip) < 0.5:
        raise ValueError("probability clip must be in (0,0.5)")
    frame = pd.read_csv(path)
    required = {"id", "label", "event_id", probability_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prediction CSV lacks required columns {missing}")
    _verify_overlay_identity(
        manifest,
        ids=frame["id"].astype(str).tolist(),
        labels=pd.to_numeric(frame["label"], errors="raise")
        .astype(np.int64)
        .tolist(),
        event_ids=frame["event_id"].astype(str).tolist(),
        purpose="prediction CSV",
    )
    original = pd.to_numeric(
        frame[probability_column], errors="raise"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(original).all() or np.any(original < 0) or np.any(
        original > 1
    ):
        raise ValueError("prediction probabilities must be finite in [0,1]")
    clipped = np.clip(original, float(clip), 1.0 - float(clip))
    probability_tensor = torch.from_numpy(clipped).float()
    logits = torch.logit(probability_tensor).contiguous()
    replayed = torch.sigmoid(logits).double().numpy()
    maximum = float(np.max(np.abs(replayed - clipped)))
    if maximum > 5.0e-7:
        raise ValueError(
            f"float32 probability/logit replay error {maximum:.3e} exceeds 5e-7"
        )
    provenance = {
        "source_kind": "prediction_csv_float32_logit",
        "prediction_csv": str(path),
        "prediction_csv_sha256": sha256_file(path),
        "probability_column": str(probability_column),
        "probability_to_logit_conversion": {
            "input_parse_dtype": "numpy.float64",
            "clip": float(clip),
            "logit_storage_dtype": "torch.float32",
            "roundtrip_max_abs_error_against_clipped_probability": maximum,
            "required_max_abs_error": 5.0e-7,
            "clipped_values": int(np.count_nonzero(original != clipped)),
        },
    }
    return logits, provenance


def _verify_probability_csv(
    logits: torch.Tensor,
    path: Path,
    manifest: Mapping[str, Any],
    *,
    probability_column: str,
) -> dict[str, Any]:
    assert_development_path(path, purpose="verification prediction CSV")
    frame = pd.read_csv(path)
    required = {"id", "label", "event_id", probability_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"verification CSV lacks columns {missing}")
    _verify_overlay_identity(
        manifest,
        ids=frame["id"].astype(str).tolist(),
        labels=pd.to_numeric(frame["label"], errors="raise")
        .astype(np.int64)
        .tolist(),
        event_ids=frame["event_id"].astype(str).tolist(),
        purpose="verification prediction CSV",
    )
    expected = pd.to_numeric(
        frame[probability_column], errors="raise"
    ).to_numpy(dtype=np.float64)
    observed = torch.sigmoid(logits.float()).double().numpy()
    maximum = float(np.max(np.abs(expected - observed)))
    if maximum > 5.0e-7:
        raise ValueError(
            f"replayed probability differs from CSV by {maximum:.3e}; "
            "required <=5e-7"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "probability_column": str(probability_column),
        "max_abs_probability_error": maximum,
        "required_max_abs_error": 5.0e-7,
        "verified": True,
    }


def command_rebase(args: argparse.Namespace) -> None:
    """Build a small CPU-only base-logit overlay without touching patches."""

    patch_manifest_path = Path(args.patch_manifest).expanduser().resolve()
    output_path = Path(args.output_overlay).expanduser().resolve()
    assert_development_path(patch_manifest_path, purpose="patch manifest")
    assert_development_path(output_path, purpose="base overlay output")
    if output_path.exists():
        raise FileExistsError(output_path)
    family = str(args.family).strip()
    if not family:
        raise ValueError("--family must be nonblank")
    manifest = load_manifest(
        patch_manifest_path, expected_split=str(args.split)
    )
    use_checkpoint = bool(args.feature_cache or args.head_checkpoint)
    use_predictions = bool(args.predictions_csv)
    if use_checkpoint == use_predictions:
        raise ValueError(
            "choose exactly one source: checkpoint replay or prediction CSV"
        )
    if use_checkpoint:
        if not args.feature_cache or not args.head_checkpoint:
            raise ValueError(
                "checkpoint replay needs --feature-cache and --head-checkpoint"
            )
        logits, provenance = replay_ragged_head_logits(
            feature_cache_path=Path(args.feature_cache).expanduser().resolve(),
            head_checkpoint_path=Path(
                args.head_checkpoint
            ).expanduser().resolve(),
            head_config_path=(
                Path(args.head_config_json).expanduser().resolve()
                if args.head_config_json
                else None
            ),
            manifest=manifest,
            split=str(args.split),
            batch_size=int(args.batch_size),
        )
    else:
        logits, provenance = logits_from_prediction_csv(
            Path(args.predictions_csv).expanduser().resolve(),
            manifest=manifest,
            probability_column=str(args.probability_column),
            clip=float(args.probability_clip),
        )
    verification = None
    if args.verify_predictions_csv:
        verification = _verify_probability_csv(
            logits,
            Path(args.verify_predictions_csv).expanduser().resolve(),
            manifest,
            probability_column=str(args.probability_column),
        )
    family_contract = {
        "family": family,
        "source_kind": provenance["source_kind"],
        "head_checkpoint_sha256": provenance.get(
            "head_checkpoint_sha256"
        ),
        "head_config_sha256": provenance.get("head_config_sha256"),
        "feature_weights_sha256": provenance.get(
            "feature_weights_sha256"
        ),
        "external_source_model_sha256": (
            str(args.source_model_sha256).strip() or None
        ),
        "conversion": provenance.get("probability_to_logit_conversion"),
    }
    # Split-specific roundtrip maxima and CSV paths must not make train/val
    # family contracts differ.
    conversion = family_contract.get("conversion")
    if isinstance(conversion, Mapping):
        family_contract["conversion"] = {
            key: value
            for key, value in conversion.items()
            if key
            not in {
                "roundtrip_max_abs_error_against_clipped_probability",
                "clipped_values",
            }
        }
    payload = {
        "format_version": OVERLAY_VERSION,
        "script_version": SCRIPT_VERSION,
        "split": str(args.split),
        "family": family,
        "family_contract": family_contract,
        "family_contract_sha256": canonical_json_sha256(family_contract),
        "base_logits": logits.float().contiguous(),
        "base_logits_sha256": tensor_sha256(logits.float()),
        "rows": int(logits.numel()),
        "patch_manifest": str(patch_manifest_path),
        "patch_manifest_sha256": sha256_file(patch_manifest_path),
        "patch_manifest_content_sha256": manifest[
            "manifest_content_sha256"
        ],
        "patch_identity_sha256": manifest["identity"]["identity_sha256"],
        "provenance": provenance,
        "verification_prediction_audit": verification,
        "test_or_sealed_read": False,
    }
    atomic_torch(output_path, payload)
    summary = {
        key: value
        for key, value in payload.items()
        if key != "base_logits"
    }
    summary["overlay"] = str(output_path)
    summary["overlay_sha256"] = sha256_file(output_path)
    atomic_json(output_path.with_suffix(output_path.suffix + ".json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def load_base_overlay(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    expected_split: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert_development_path(path, purpose=f"{expected_split} base overlay")
    payload = dict(_load_torch_payload(path))
    if payload.get("format_version") != OVERLAY_VERSION:
        raise ValueError(f"{path}: unsupported base overlay")
    if payload.get("split") != expected_split:
        raise ValueError(f"{path}: overlay split mismatch")
    if bool(payload.get("test_or_sealed_read", True)):
        raise ValueError(f"{path}: overlay held-out contract is invalid")
    if payload.get("patch_identity_sha256") != manifest["identity"][
        "identity_sha256"
    ]:
        raise ValueError(f"{path}: overlay identity differs from patch cache")
    if payload.get("patch_manifest_content_sha256") != manifest.get(
        "manifest_content_sha256"
    ):
        raise ValueError(f"{path}: overlay patch-manifest content differs")
    logits = payload["base_logits"].float().contiguous()
    if logits.shape != (int(manifest["rows"]),):
        raise ValueError(f"{path}: overlay logit shape differs")
    if tensor_sha256(logits) != payload.get("base_logits_sha256"):
        raise ValueError(f"{path}: overlay logit SHA mismatch")
    expected_family_sha = canonical_json_sha256(payload["family_contract"])
    if expected_family_sha != payload.get("family_contract_sha256"):
        raise ValueError(f"{path}: overlay family contract SHA mismatch")
    audit = {
        "path": str(path),
        "sha256": sha256_file(path),
        "family": str(payload["family"]),
        "family_contract_sha256": expected_family_sha,
        "source_kind": payload["provenance"]["source_kind"],
        "verification_prediction_audit": payload.get(
            "verification_prediction_audit"
        ),
    }
    return logits, audit


def _load_shard(path: Path) -> dict[str, Any]:
    payload = dict(_load_torch_payload(path))
    if payload.get("format_version") != SHARD_VERSION:
        raise ValueError(f"{path}: invalid shard")
    return payload


def iter_shard_batches(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    sample_weights: Optional[torch.Tensor] = None,
    negative_event_rows: Optional[torch.Tensor] = None,
    base_logits_override: Optional[torch.Tensor] = None,
) -> Iterator[dict[str, Any]]:
    """Load one local shard at a time and never materialize the full cache."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = list(manifest["shards"])
    generator = torch.Generator().manual_seed(int(seed) + 104729 * int(epoch))
    if shuffle:
        order = torch.randperm(len(records), generator=generator).tolist()
        records = [records[index] for index in order]
    for record in records:
        payload = _load_shard(manifest_path.parent / str(record["file"]))
        rows = int(payload["patch_tokens"].shape[0])
        if shuffle:
            row_order = torch.randperm(rows, generator=generator)
        else:
            row_order = torch.arange(rows)
        for start in range(0, rows, int(batch_size)):
            local = row_order[start : start + int(batch_size)]
            global_indices = payload["row_indices"][local].long()
            batch = {
                "row_indices": global_indices,
                "patch_tokens": payload["patch_tokens"][local],
                "base_logits": (
                    base_logits_override[global_indices]
                    if base_logits_override is not None
                    else payload["base_logits"][local]
                ),
                "labels": payload["labels"][local],
                "unique_mask": payload["unique_mask"][local],
                "delta_days": payload["delta_days"][local],
                "quality": payload["quality"][local],
                "event_ids": [
                    payload["event_ids"][int(index)] for index in local.tolist()
                ],
                "ids": [payload["ids"][int(index)] for index in local.tolist()],
            }
            if sample_weights is not None:
                batch["sample_weights"] = sample_weights[global_indices]
            if negative_event_rows is not None:
                batch["negative_event_rows"] = negative_event_rows[
                    global_indices
                ]
            yield batch


def _event_class_balanced_weights(
    labels: torch.Tensor,
    event_ids: Sequence[str],
) -> torch.Tensor:
    return patch_followup.event_balanced_training_weights(labels, event_ids)


def _selection_value(metrics: Mapping[str, Any], name: str) -> float:
    if name == "event_balanced_ap":
        return float(metrics["event_balanced_ap"])
    if name == "event_balanced_macro_f1":
        return float(metrics["event_balanced_macro_f1_selected"])
    raise ValueError(f"unsupported selection metric {name}")


def add_independently_selected_macro_f1(
    metrics: dict[str, Any],
    labels: torch.Tensor,
    probabilities: np.ndarray,
    event_ids: Sequence[str],
) -> None:
    """Record macro-F1 selection without changing the legacy positive-F1 view.

    The older patch audit selects its threshold by positive-class F1.  The
    global L89 audit independently maximizes event-balanced macro-F1.  Both
    are useful but are not interchangeable, so retain the former under an
    explicit name and add the latter using the exact global implementation.
    """

    target = labels.long().cpu().numpy().astype(np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    event_weights = l89.event_balanced_row_weights(event_ids)
    threshold, selected_macro = (
        event_head_followup.best_weighted_macro_f1_threshold(
            target, probability, event_weights
        )
    )
    prediction = (probability >= float(threshold)).astype(np.int64)
    metrics["event_balanced_macro_f1_selected"] = float(selected_macro)
    metrics["event_balanced_macro_f1_selected_threshold"] = float(threshold)
    metrics["event_balanced_at_macro_f1_selected"] = {
        "threshold": float(threshold),
        "positive_f1": float(
            f1_score(
                target,
                prediction,
                sample_weight=event_weights,
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                target,
                prediction,
                labels=[0, 1],
                average="macro",
                sample_weight=event_weights,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                target, prediction, sample_weight=event_weights
            )
        ),
        "predicted_positive_rate": float(prediction.mean()),
    }
    positive_selected = metrics["event_balanced_at_event_selected"]
    metrics["event_balanced_positive_f1_selected"] = float(
        positive_selected["positive_f1"]
    )
    metrics["event_balanced_positive_f1_selected_threshold"] = float(
        positive_selected["threshold"]
    )


def _all_negative_fp_audit(
    labels: torch.Tensor,
    event_ids: Sequence[str],
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "label": labels.long().numpy(),
            "event_id": [str(value) for value in event_ids],
            "probability": probabilities,
        }
    )
    negative_events = [
        event_id
        for event_id, group in frame.groupby("event_id", sort=True)
        if int(group["label"].max()) == 0
    ]
    per_event = []
    for event_id in negative_events:
        group = frame[frame["event_id"] == event_id]
        rate = float((group["probability"] >= float(threshold)).mean())
        per_event.append(rate)
    return {
        "all_negative_events": len(negative_events),
        "threshold": float(threshold),
        "mean_event_false_positive_rate": (
            float(np.mean(per_event)) if per_event else 0.0
        ),
        "false_positive_mass": float(np.sum(per_event)),
        "equal_event_weight": True,
    }


def evaluate_stream(
    model: TempoPatchHead,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    batch_size: int,
    device: torch.device,
    base_logits_override: Optional[torch.Tensor] = None,
    late_fusion_logits_override: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    rows = int(manifest["rows"])
    logits = torch.empty(rows, dtype=torch.float32)
    mil_logits = torch.empty(rows, dtype=torch.float32)
    grid_shape = tuple(int(value) for value in manifest["grid_shape"])
    t0_index = int(manifest["configuration"]["t0_index"])
    model.eval()
    with torch.inference_mode():
        for batch in iter_shard_batches(
            manifest,
            manifest_path,
            batch_size=batch_size,
            seed=0,
            epoch=0,
            shuffle=False,
            base_logits_override=base_logits_override,
        ):
            indices = batch["row_indices"]
            output = model(
                batch["base_logits"].to(device),
                batch["patch_tokens"].to(device),
                batch["unique_mask"].to(device),
                batch["delta_days"].to(device),
                batch["quality"].to(device),
                t0_index=t0_index,
                grid_shape=grid_shape,
            )
            logits[indices] = output.logits.cpu()
            mil_logits[indices] = output.mil_logit.cpu()
    labels = torch.tensor(manifest["identity"]["labels"], dtype=torch.long)
    event_ids = manifest["identity"]["event_ids"]
    metrics, probabilities = patch_followup.metrics_from_logits(
        labels, logits, event_ids
    )
    add_independently_selected_macro_f1(
        metrics, labels, probabilities, event_ids
    )
    threshold = float(
        metrics["event_balanced_at_event_selected"]["threshold"]
    )
    metrics["all_negative_event_fp"] = _all_negative_fp_audit(
        labels, event_ids, probabilities, threshold=threshold
    )
    if late_fusion_logits_override is not None:
        late_logits = late_fusion_logits_override.float().contiguous()
        if late_logits.shape != (rows,):
            raise ValueError("late-fusion overlay logit shape differs")
        # This is deliberately evaluation-only.  It is never passed to the
        # model or the optimizer, and therefore cannot influence training.
        fused_logits = 0.5 * (logits + late_logits)
        fused_metrics, _ = patch_followup.metrics_from_logits(
            labels, fused_logits, event_ids
        )
        fused_probability = torch.sigmoid(fused_logits).numpy()
        add_independently_selected_macro_f1(
            fused_metrics, labels, fused_probability, event_ids
        )
        fused_threshold = float(
            fused_metrics["event_balanced_at_event_selected"]["threshold"]
        )
        fused_metrics["all_negative_event_fp"] = _all_negative_fp_audit(
            labels,
            event_ids,
            fused_probability,
            threshold=fused_threshold,
        )
        metrics["fixed_0p5_late_logit_fusion"] = fused_metrics
    return metrics, probabilities, mil_logits.numpy(), logits.numpy()


def command_train(args: argparse.Namespace) -> None:
    """Train one train/dev-only TEMPO configuration with early stopping."""

    set_seed(args.seed)
    zero_init_mode = str(getattr(args, "zero_init_mode", "scalar"))
    train_path = Path(args.train_manifest).expanduser().resolve()
    val_path = Path(args.val_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path, purpose in (
        (train_path, "train manifest"),
        (val_path, "validation manifest"),
        (output_dir, "output"),
    ):
        assert_development_path(path, purpose=purpose)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists() and not args.resume:
        raise FileExistsError(result_path)
    train, val, cache_audit = validate_cache_pair(train_path, val_path)
    train_base_override: Optional[torch.Tensor] = None
    val_base_override: Optional[torch.Tensor] = None
    base_overlay_audit: Optional[dict[str, Any]] = None
    val_late_fusion_override: Optional[torch.Tensor] = None
    late_fusion_overlay_audit: Optional[dict[str, Any]] = None
    if bool(args.train_base_overlay) != bool(args.val_base_overlay):
        raise ValueError(
            "train and validation base overlays must be supplied together"
        )
    if args.train_base_overlay:
        train_overlay_path = Path(
            args.train_base_overlay
        ).expanduser().resolve()
        val_overlay_path = Path(args.val_base_overlay).expanduser().resolve()
        train_base_override, train_overlay_audit = load_base_overlay(
            train_overlay_path, train, expected_split="train"
        )
        val_base_override, val_overlay_audit = load_base_overlay(
            val_overlay_path, val, expected_split="val"
        )
        if (
            train_overlay_audit["family"]
            != val_overlay_audit["family"]
            or train_overlay_audit["family_contract_sha256"]
            != val_overlay_audit["family_contract_sha256"]
        ):
            raise ValueError(
                "train/validation base overlay family mismatch"
            )
        base_overlay_audit = {
            "train": train_overlay_audit,
            "val": val_overlay_audit,
            "same_family": True,
            "same_family_contract": True,
        }
    if getattr(args, "val_late_fusion_overlay", ""):
        late_fusion_path = Path(
            args.val_late_fusion_overlay
        ).expanduser().resolve()
        val_late_fusion_override, late_fusion_overlay_audit = load_base_overlay(
            late_fusion_path, val, expected_split="val"
        )
    labels = torch.tensor(train["identity"]["labels"], dtype=torch.long)
    event_ids = train["identity"]["event_ids"]
    sample_weights = _event_class_balanced_weights(labels, event_ids)
    negative_rows = all_negative_event_mask(labels, event_ids)
    feature_dim = int(train["configuration"]["projection"]["output_dim"])
    configuration = {
        "script_version": SCRIPT_VERSION,
        "train_manifest_sha256": sha256_file(train_path),
        "val_manifest_sha256": sha256_file(val_path),
        "feature_dim": feature_dim,
        "match_rank": int(args.match_rank),
        "value_dim": int(args.value_dim),
        "hidden_dim": int(args.hidden_dim),
        "radius": int(args.radius),
        "temperature": float(args.temperature),
        "topk_fraction": float(args.topk_fraction),
        "normality_scale": float(args.normality_scale),
        "use_normality_features": bool(args.use_normality_features),
        "residual_cap": float(args.residual_cap),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "null_weight": float(args.null_weight),
        "null_margin": float(args.null_margin),
        "batch_size": int(args.batch_size),
        "selection_metric": str(args.selection_metric),
        "seed": int(args.seed),
        "base_overlay_family": (
            base_overlay_audit["train"]["family"]
            if base_overlay_audit is not None
            else "embedded_role_only"
        ),
        "train_base_overlay_sha256": (
            base_overlay_audit["train"]["sha256"]
            if base_overlay_audit is not None
            else None
        ),
        "val_base_overlay_sha256": (
            base_overlay_audit["val"]["sha256"]
            if base_overlay_audit is not None
            else None
        ),
        "fixed_0p5_late_logit_fusion": (
            {
                "evaluation_only": True,
                "weight_primary_tempo_logit": 0.5,
                "weight_frozen_late_logit": 0.5,
                "selection_uses_primary_tempo_only": True,
                "overlay_family": late_fusion_overlay_audit["family"],
                "overlay_sha256": late_fusion_overlay_audit["sha256"],
            }
            if late_fusion_overlay_audit is not None
            else None
        ),
        "test_or_sealed_read": False,
    }
    # Keep the default scalar-mode configuration byte-compatible with runs
    # created before this stability control was added.
    if zero_init_mode != "scalar":
        configuration["zero_init_mode"] = zero_init_mode
    configuration_sha = canonical_json_sha256(configuration)
    device = torch.device(args.device)
    model = TempoPatchHead(
        feature_dim,
        match_rank=int(args.match_rank),
        value_dim=int(args.value_dim),
        hidden_dim=int(args.hidden_dim),
        radius=int(args.radius),
        temperature=float(args.temperature),
        topk_fraction=float(args.topk_fraction),
        normality_scale=float(args.normality_scale),
        use_normality_features=bool(args.use_normality_features),
        residual_cap=float(args.residual_cap),
        zero_init_mode=zero_init_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_probability: Optional[np.ndarray] = None
    best_mil: Optional[np.ndarray] = None
    best_logits: Optional[np.ndarray] = None
    last_path = output_dir / "checkpoint_last.pt"
    if args.resume and last_path.exists():
        checkpoint = _load_torch_payload(last_path)
        if checkpoint.get("configuration_sha256") != configuration_sha:
            raise ValueError("resume checkpoint configuration differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = list(checkpoint["history"])
        best = copy.deepcopy(checkpoint["best"])
        best_state = copy.deepcopy(checkpoint["best_state"])
        start_epoch = int(checkpoint["epoch"]) + 1

    if start_epoch == 1:
        metrics_zero, probability_zero, mil_zero, logits_zero = evaluate_stream(
            model,
            val,
            val_path,
            batch_size=int(args.eval_batch_size),
            device=device,
            base_logits_override=val_base_override,
            late_fusion_logits_override=val_late_fusion_override,
        )
        if not model.exact_noop:
            raise AssertionError("TEMPO must be exact no-op at epoch zero")
        # The identity contract is on logits: every streamed model output must
        # be bit-exact to its frozen base.  Sigmoid is audited separately with
        # a numerical tolerance because backend implementations need not be
        # bit-identical.
        base_logits_zero = np.empty(int(val["rows"]), dtype=np.float32)
        base_probabilities = np.empty(int(val["rows"]), dtype=np.float32)
        for batch in iter_shard_batches(
            val,
            val_path,
            batch_size=int(args.eval_batch_size),
            seed=0,
            epoch=0,
            shuffle=False,
            base_logits_override=val_base_override,
        ):
            indices = batch["row_indices"].numpy()
            frozen_logits = batch["base_logits"].float()
            base_logits_zero[indices] = frozen_logits.numpy()
            base_probabilities[indices] = torch.sigmoid(frozen_logits).numpy()
        if not np.array_equal(logits_zero, base_logits_zero):
            raise AssertionError("epoch-zero logits differ from frozen base")
        probability_max_abs = float(
            np.max(np.abs(probability_zero - base_probabilities))
        )
        # Two float32 ULPs cover legitimate shape/backend differences in the
        # auxiliary sigmoid audit.  The governing identity check above
        # remains bit-exact on logits.
        probability_tolerance = 2e-7
        if (
            not math.isfinite(probability_max_abs)
            or probability_max_abs > probability_tolerance
        ):
            raise AssertionError(
                "epoch-zero probabilities exceed frozen-base tolerance: "
                f"max_abs={probability_max_abs:.9g}"
            )
        zero = {
            "epoch": 0,
            "optimizer_steps": 0,
            "validation": metrics_zero,
            "exact_base_logit": True,
            "probability_max_abs_vs_frozen_base": probability_max_abs,
            "probability_tolerance": probability_tolerance,
        }
        history.append(zero)
        best = copy.deepcopy(zero)
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_probability = probability_zero.copy()
        best_mil = mil_zero.copy()
        best_logits = logits_zero.copy()

    no_improvement = 0
    started = time.monotonic()
    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        total = classification_total = null_total = 0.0
        rows_seen = steps = 0
        for batch in iter_shard_batches(
            train,
            train_path,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            epoch=epoch,
            shuffle=True,
            sample_weights=sample_weights,
            negative_event_rows=negative_rows,
            base_logits_override=train_base_override,
        ):
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["base_logits"].to(device),
                batch["patch_tokens"].to(device),
                batch["unique_mask"].to(device),
                batch["delta_days"].to(device),
                batch["quality"].to(device),
                t0_index=int(train["configuration"]["t0_index"]),
                grid_shape=tuple(int(value) for value in train["grid_shape"]),
            )
            target = batch["labels"].float().to(device)
            weight = batch["sample_weights"].float().to(device)
            per_row = F.binary_cross_entropy_with_logits(
                output.logits, target, reduction="none"
            )
            classification_loss = (per_row * weight).sum() / weight.sum()
            null_loss = negative_event_topk_null_loss(
                output.patch_scores,
                batch["negative_event_rows"].to(device),
                topk_fraction=float(args.topk_fraction),
                margin=float(args.null_margin),
            )
            loss = classification_loss + float(args.null_weight) * null_loss
            if not torch.isfinite(loss):
                raise RuntimeError("TEMPO produced non-finite train loss")
            loss.backward()
            if float(args.grad_clip) > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(args.grad_clip)
                )
            optimizer.step()
            count = len(batch["labels"])
            total += float(loss.detach()) * count
            classification_total += float(classification_loss.detach()) * count
            null_total += float(null_loss.detach()) * count
            rows_seen += count
            steps += 1
            if int(args.max_steps_per_epoch) > 0 and steps >= int(
                args.max_steps_per_epoch
            ):
                break
        metrics, probability, mil, evaluated_logits = evaluate_stream(
            model,
            val,
            val_path,
            batch_size=int(args.eval_batch_size),
            device=device,
            base_logits_override=val_base_override,
            late_fusion_logits_override=val_late_fusion_override,
        )
        record = {
            "epoch": int(epoch),
            "optimizer_steps": int(steps),
            "rows_seen": int(rows_seen),
            "train_loss": total / max(1, rows_seen),
            "classification_loss": classification_total / max(1, rows_seen),
            "null_loss": null_total / max(1, rows_seen),
            "validation": metrics,
            "residual_gate": float(model.residual_gate.detach().cpu()),
            "elapsed_seconds": float(time.monotonic() - started),
            "exact_base_logit": False,
        }
        history.append(record)
        assert best is not None
        current_value = _selection_value(metrics, args.selection_metric)
        best_value = _selection_value(best["validation"], args.selection_metric)
        if current_value > best_value + float(args.min_delta):
            best = copy.deepcopy(record)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_probability = probability.copy()
            best_mil = mil.copy()
            best_logits = evaluated_logits.copy()
            no_improvement = 0
        else:
            no_improvement += 1
        atomic_torch(
            last_path,
            {
                "format_version": HEAD_VERSION,
                "configuration": configuration,
                "configuration_sha256": configuration_sha,
                "epoch": int(epoch),
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best": best,
                "best_state": best_state,
                "test_or_sealed_read": False,
            },
        )
        print(
            f"[tempo-train] epoch={epoch}/{args.epochs} "
            f"AP={metrics['event_balanced_ap']:.6f} "
            f"macroF1_selected="
            f"{metrics['event_balanced_macro_f1_selected']:.6f} "
            f"positiveF1_selected="
            f"{metrics['event_balanced_positive_f1_selected']:.6f} "
            f"FPmass={metrics['all_negative_event_fp']['false_positive_mass']:.4f} "
            f"gate={record['residual_gate']:.5f}",
            flush=True,
        )
        if int(args.patience) >= 0 and no_improvement > int(args.patience):
            print(
                f"[tempo-train] early stop after {no_improvement} "
                "non-improving epoch(s)",
                flush=True,
            )
            break

    assert best is not None and best_state is not None
    if best_probability is None or best_mil is None or best_logits is None:
        # A resumed run may already be beyond --epochs.  Reproduce selected
        # predictions from its persisted best state.
        model.load_state_dict(best_state, strict=True)
        _, best_probability, best_mil, best_logits = evaluate_stream(
            model,
            val,
            val_path,
            batch_size=int(args.eval_batch_size),
            device=device,
            base_logits_override=val_base_override,
            late_fusion_logits_override=val_late_fusion_override,
        )
    atomic_torch(
        output_dir / "checkpoint_best.pt",
        {
            "format_version": HEAD_VERSION,
            "configuration": configuration,
            "configuration_sha256": configuration_sha,
            "epoch": int(best["epoch"]),
            "model": best_state,
            "selection_metric": str(args.selection_metric),
            "validation": best["validation"],
            "cache_audit": cache_audit,
            "base_overlay_audit": base_overlay_audit,
            "late_fusion_overlay_audit": late_fusion_overlay_audit,
            "backbone_frozen": True,
            "base_logits_frozen": True,
            "epoch_zero_exact_base_logit": True,
            "test_or_sealed_read": False,
        },
    )
    val_frame = pd.DataFrame(
        {
            "id": val["identity"]["ids"],
            "plume_id": val["identity"]["plume_ids"],
            "event_id": val["identity"]["event_ids"],
            "label": val["identity"]["labels"],
            "probability": best_probability,
            "logit": best_logits,
            "mil_logit": best_mil,
            "selected_epoch": int(best["epoch"]),
        }
    )
    if val_late_fusion_override is not None:
        fused_logits = 0.5 * (
            torch.from_numpy(best_logits).float()
            + val_late_fusion_override.float()
        )
        val_frame["fixed_0p5_late_logit_fusion_probability"] = (
            torch.sigmoid(fused_logits).numpy()
        )
    l89.atomic_csv_write(output_dir / "validation_predictions.csv", val_frame)
    result = {
        "format_version": HEAD_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "configuration": configuration,
        "configuration_sha256": configuration_sha,
        "cache_audit": cache_audit,
        "base_overlay_audit": base_overlay_audit,
        "late_fusion_overlay_audit": late_fusion_overlay_audit,
        "epoch_zero": history[0],
        "best": best,
        "epochs_completed": int(history[-1]["epoch"]),
        "early_stopped": int(history[-1]["epoch"]) < int(args.epochs),
        "matching_contract": {
            "frozen_final_patch_tokens": True,
            "neighbourhood": f"{2 * int(args.radius) + 1}x{2 * int(args.radius) + 1}",
            "low_rank_soft_match": True,
            "explicit_history_pairwise_normality": True,
            "normality_features_enabled": bool(
                args.use_normality_features
            ),
            "delta_days_quality_gate": True,
            "topk_mil": True,
            "all_negative_event_null_loss": float(args.null_weight) > 0,
            "zero_init_base_logit_residual": True,
            "zero_initialization_mode": zero_init_mode,
            "residual_gate_trainable": bool(
                model.residual_gate.requires_grad
            ),
            "epoch_zero_exact_base_logit": True,
            "fixed_0p5_late_logit_fusion_evaluation_only": (
                late_fusion_overlay_audit is not None
            ),
            "raw_768d_patch_tokens_persisted": False,
        },
        "interpretation_guardrail": (
            "Train/development-selected configuration, epoch and threshold. "
            "No held-out estimate and no SOTA claim."
        ),
        "test_or_sealed_read": False,
    }
    atomic_json(result_path, result)
    atomic_json(output_dir / "metrics_history.json", history)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def run_synthetic_dry_run(seed: int = 20260728) -> dict[str, Any]:
    """CPU-only shape, gradient, null-loss and exact-fallback smoke test."""

    set_seed(seed)
    batch, visits, height, width, dimension = 6, 6, 4, 4, 16
    patches = height * width
    tokens = torch.randn(batch, visits, patches, dimension)
    # Stable history plus a localized t0 onset in half the rows.
    tokens[:, 1:] = tokens[:, 1:2] + 0.02 * torch.randn_like(tokens[:, 1:])
    tokens[:3, 0] = tokens[:3, 1]
    tokens[3:, 0] = tokens[3:, 1]
    tokens[3:, 0, 5:7, :4] += 2.5
    valid = torch.ones(batch, visits, dtype=torch.bool)
    valid[0, 4:] = False
    delta = torch.tensor(
        [[0.0, -30.0, -90.0, -180.0, -365.0, -730.0]]
    ).expand(batch, -1)
    quality = torch.ones(batch, visits)
    base = torch.linspace(-0.5, 0.5, batch)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    events = ["n0", "n1", "n2", "p0", "p1", "p2"]
    model = TempoPatchHead(
        dimension,
        match_rank=8,
        value_dim=8,
        hidden_dim=16,
        radius=1,
        topk_fraction=0.125,
    )
    output = model(
        base,
        tokens,
        valid,
        delta,
        quality,
        t0_index=0,
        grid_shape=(height, width),
    )
    if not torch.equal(output.logits, base):
        raise AssertionError("dry-run epoch zero differs from frozen base")
    negative = all_negative_event_mask(labels, events)
    null = negative_event_topk_null_loss(
        output.patch_scores,
        negative,
        topk_fraction=0.125,
    )
    classification = F.binary_cross_entropy_with_logits(output.logits, labels.float())
    loss = classification + 0.1 * null
    loss.backward()
    gradient = float(
        sum(
            parameter.grad.abs().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    if not math.isfinite(gradient) or gradient <= 0:
        raise AssertionError("dry-run produced no finite gradient")
    return {
        "status": "ok",
        "device": "cpu",
        "shape": {
            "logits": list(output.logits.shape),
            "patch_scores": list(output.patch_scores.shape),
            "history_weights": list(output.history_weights.shape),
        },
        "topk_count": int(output.topk_count),
        "epoch_zero_exact_base_logit": True,
        "finite_positive_gradient_sum": gradient,
        "negative_event_rows": int(negative.sum()),
        "test_or_sealed_read": False,
    }


def command_dry_run(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            run_synthetic_dry_run(seed=int(args.seed)),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser(
        "extract", help="Stream frozen projected final-patch shards."
    )
    extract.add_argument("--split", choices=("train", "val"), required=True)
    extract.add_argument("--csv", required=True)
    extract.add_argument("--cls-cache", required=True)
    extract.add_argument("--weights", required=True)
    extract.add_argument("--base-head-checkpoint", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--required-local-root", default=str(DEFAULT_LOCAL_ROOT))
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--num-workers", type=int, default=4)
    extract.add_argument("--prefetch-factor", type=int, default=1)
    extract.add_argument("--shard-rows", type=int, default=96)
    extract.add_argument("--projection-dim", type=int, default=64)
    extract.add_argument("--projection-seed", type=int, default=36064)
    extract.add_argument(
        "--projection-mode",
        choices=("gaussian_qr", "nested_extend_128"),
        default="gaussian_qr",
    )
    extract.add_argument(
        "--projection-extension-seed",
        type=int,
        default=36192,
        help="independent seed for nested_extend_128 extra columns",
    )
    extract.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    extract.add_argument("--base-eval-batch-size", type=int, default=512)
    extract.add_argument("--min-cuda-free-gib", type=float, default=12.0)
    extract.add_argument("--max-cuda-allocated-gib", type=float, default=16.0)
    extract.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    extract.add_argument("--seed", type=int, default=20260728)
    extract.add_argument("--debug", action="store_true")
    extract.set_defaults(handler=command_extract)

    rebase = commands.add_parser(
        "rebase",
        help="Build a CPU-only exact base-logit overlay without re-extraction.",
    )
    rebase.add_argument("--split", choices=("train", "val"), required=True)
    rebase.add_argument("--patch-manifest", required=True)
    rebase.add_argument("--family", required=True)
    rebase.add_argument("--output-overlay", required=True)
    rebase.add_argument("--feature-cache", default="")
    rebase.add_argument("--head-checkpoint", default="")
    rebase.add_argument("--head-config-json", default="")
    rebase.add_argument("--predictions-csv", default="")
    rebase.add_argument("--verify-predictions-csv", default="")
    rebase.add_argument("--probability-column", default="probability")
    rebase.add_argument("--probability-clip", type=float, default=1e-7)
    rebase.add_argument("--source-model-sha256", default="")
    rebase.add_argument("--batch-size", type=int, default=512)
    rebase.set_defaults(handler=command_rebase)

    train = commands.add_parser(
        "train", help="Train one train/dev-only TEMPO head."
    )
    train.add_argument("--train-manifest", required=True)
    train.add_argument("--val-manifest", required=True)
    train.add_argument("--train-base-overlay", default="")
    train.add_argument("--val-base-overlay", default="")
    train.add_argument(
        "--val-late-fusion-overlay",
        default="",
        help=(
            "Optional frozen validation overlay for evaluation-only fixed "
            "0.5/0.5 logit fusion; it never enters training or selection."
        ),
    )
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--match-rank", type=int, default=16)
    train.add_argument("--value-dim", type=int, default=32)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--radius", type=int, choices=(0, 1, 2), default=1)
    train.add_argument("--temperature", type=float, default=0.10)
    train.add_argument("--topk-fraction", type=float, default=0.10)
    train.add_argument("--normality-scale", type=float, default=1.0)
    train.add_argument(
        "--use-normality-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--residual-cap", type=float, default=1.5)
    train.add_argument(
        "--zero-init-mode",
        choices=("scalar", "readout"),
        default="scalar",
        help=(
            "scalar keeps the original zero residual gate; readout zeros the "
            "patch readout and fixes the residual gate to one"
        ),
    )
    train.add_argument("--null-weight", type=float, default=0.10)
    train.add_argument("--null-margin", type=float, default=0.0)
    train.add_argument("--epochs", type=int, default=6)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--eval-batch-size", type=int, default=24)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--patience", type=int, default=1)
    train.add_argument("--min-delta", type=float, default=2e-4)
    train.add_argument("--max-steps-per-epoch", type=int, default=0)
    train.add_argument(
        "--selection-metric",
        choices=("event_balanced_ap", "event_balanced_macro_f1"),
        default="event_balanced_ap",
    )
    train.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--seed", type=int, default=20260728)
    train.set_defaults(handler=command_train)

    dry = commands.add_parser("dry-run", help="CPU-only synthetic contract run.")
    dry.add_argument("--seed", type=int, default=20260728)
    dry.set_defaults(handler=command_dry_run)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
