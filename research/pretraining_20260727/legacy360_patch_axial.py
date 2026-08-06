#!/usr/bin/env python3
"""Legacy-preserving patch-level time x sensor model for the 360 m protocol.

The important invariant in this runner is that the historical three-date
Panopticon path remains intact.  For every available sensor, the base branch
concatenates all valid dates in the channel dimension and applies the existing
Panopticon backbone and its existing classifier head.  Two new paths are added
as *zero-initialized residuals*:

1. final/early patch tokens from the individual dates are aligned by spatial
   patch and a current-date query attends along the temporal axis; and
2. the resulting per-sensor evidence is fused with masked sensor-set attention.

Consequently, before the first optimizer step:

* a one-sensor sample produces exactly the old checkpoint log-odds; and
* a multi-sensor sample produces the requested historical max/mean fusion of
  the old per-sensor log-odds.

This avoids the failure mode of first compressing every date to a CLS token.
The temporal operation is dense over corresponding spatial patches, while the
old 3*C channel-fusion route is retained as the high-performing base.

Supported initial checkpoints
-----------------------------
* raw official Panopticon state dict;
* historical single-sensor checkpoint with ``backbone`` and ``head``; and
* historical universal-360m checkpoint with ``model.backbone.*``,
  ``model.sensor_patch_embeds.*`` and ``model.heads.*``.

The command-line trainer selects checkpoints and thresholds on a development
split and touches the test split only once after training.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import query360_data as q360  # noqa: E402
from query360_data import (  # noqa: E402
    CLASSIFICATION_PATH_COLUMNS,
    DEFAULT_WV3_SRF,
    SENSOR_ORDER,
    Query360Dataset,
    StrictHashedFileCache,
    collect_classification_paths,
    query360_collate,
)
from src.backbones import build_panopticon_vitb14  # noqa: E402


SCRIPT_VERSION = "legacy360-patch-axial-v1"
SEALED_TEST_AUTHORIZATION = "engineering-historical-final-eval"
CANONICAL_EVENT_SUFFIX = re.compile(r"-[A-Za-z0-9]+$")
UNIVERSAL_SENSOR_NAMES = {
    "s2": "s2",
    "l89": "l89",
    "emit": "wv3",
    "s5p": "s5p",
}


def canonical_event(value: Any) -> str:
    """Historical event normalization used only for overlap auditing."""

    return CANONICAL_EVENT_SUFFIX.sub("", str(value).strip())


class AuthorizedSealedQuery360Dataset(Query360Dataset):
    """Explicitly authorized final-evaluation manifest reader.

    ``Query360Dataset`` intentionally rejects any manifest path containing
    test/sealed/holdout.  This subclass does not weaken that default.  It
    duplicates only manifest initialization after requiring an exact,
    user-visible authorization token; individual image paths still pass
    through the inherited fail-closed ``_resolve`` implementation.
    """

    def __init__(
        self,
        csv_path: os.PathLike[str] | str,
        *,
        authorization: str,
        local_cache: Optional[StrictHashedFileCache] = None,
        wv3_srf_csv: os.PathLike[str] | str = DEFAULT_WV3_SRF,
        pad_to_multiple: Optional[int] = 14,
        min_finite_fraction: float = 0.05,
    ) -> None:
        if authorization != SEALED_TEST_AUTHORIZATION:
            raise PermissionError(
                "Final evaluation requires explicit "
                f"--sealed_test_authorization {SEALED_TEST_AUTHORIZATION}"
            )
        text = os.fspath(csv_path).strip()
        if not text:
            raise ValueError("sealed evaluation manifest path is empty")
        self.csv_path = Path(os.path.abspath(os.path.expanduser(text)))
        if not self.csv_path.is_file():
            raise FileNotFoundError(self.csv_path)
        self.frame = pd.read_csv(
            self.csv_path, dtype=str, keep_default_na=False, low_memory=False
        )
        q360._ensure_columns(self.frame, q360.REQUIRED_SPLIT_COLUMNS)
        if "query360_index" not in self.frame.columns:
            self.frame["query360_index"] = np.arange(
                len(self.frame), dtype=np.int64
            )
        indices = pd.to_numeric(
            self.frame["query360_index"], errors="raise"
        ).astype(np.int64)
        if indices.duplicated().any():
            raise q360.Query360DataError(
                "query360_index must be unique within a manifest."
            )
        self.frame["query360_index"] = indices
        labels = pd.to_numeric(self.frame["label"], errors="raise").astype(
            np.int64
        )
        if not labels.isin([0, 1]).all():
            raise q360.Query360DataError("Only labels 0 and 1 are accepted.")
        self.frame["label"] = labels
        # The manifest is authorized, not arbitrary raw files.  Raw image paths
        # remain subject to Query360Dataset._resolve/assert_safe_path at access.
        recomputed = q360._signatures_from_matrix(
            q360._availability_matrix(self.frame)
        )
        if any(signature == "none" for signature in recomputed):
            raise q360.Query360DataError(
                "A dataset row has no available sensor."
            )
        if "availability_signature" in self.frame.columns:
            declared = (
                self.frame["availability_signature"]
                .map(q360._clean_text)
                .tolist()
            )
            mismatch = [
                index
                for index, (left, right) in enumerate(
                    zip(declared, recomputed)
                )
                if left != right
            ]
            if mismatch:
                raise q360.Query360DataError(
                    f"availability_signature mismatch at rows {mismatch[:10]}"
                )
        self.frame["availability_signature"] = recomputed
        self.local_cache = local_cache
        self.pad_to_multiple = pad_to_multiple
        self.min_finite_fraction = float(min_finite_fraction)
        if not 0.0 <= self.min_finite_fraction <= 1.0:
            raise ValueError("min_finite_fraction must be in [0, 1].")
        self.channel_ids = q360.get_sensor_channel_ids(wv3_srf_csv)
        self._s2_mean = torch.tensor(
            q360.S2_PRECOMPUTED_STATS[0], dtype=torch.float32
        ).view(-1, 1, 1)
        self._s2_std = torch.tensor(
            q360.S2_PRECOMPUTED_STATS[1], dtype=torch.float32
        ).clamp_min(1e-6).view(-1, 1, 1)
        self._l89_mean = torch.tensor(
            q360.L89_PRECOMPUTED_STATS[0], dtype=torch.float32
        ).view(-1, 1, 1)
        self._l89_std = torch.tensor(
            q360.L89_PRECOMPUTED_STATS[1], dtype=torch.float32
        ).clamp_min(1e-6).view(-1, 1, 1)
        self._s5p_mean = torch.tensor(
            q360.S5P_PRECOMPUTED_STATS[0], dtype=torch.float32
        )
        self._s5p_std = torch.tensor(
            q360.S5P_PRECOMPUTED_STATS[1], dtype=torch.float32
        ).clamp_min(1e-6)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class LegacyCLSHead(nn.Module):
    """The exact LayerNorm -> two-class Linear head used by legacy runs."""

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(value))


class SameLocationTemporalResidual(nn.Module):
    """Current-patch query over temporal roles, followed by fixed top-k pooling."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int = 8,
        num_roles: int = 3,
        topk_fraction: float = 0.25,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if num_roles < 2:
            raise ValueError("num_roles must be at least two")
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0, 1]")
        self.embed_dim = int(embed_dim)
        self.num_roles = int(num_roles)
        self.topk_fraction = float(topk_fraction)
        self.role_embedding = nn.Parameter(
            torch.zeros(1, num_roles, 1, embed_dim)
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.residual_norm = nn.LayerNorm(embed_dim)
        self.residual_projection = nn.Linear(embed_dim, embed_dim)
        nn.init.normal_(self.role_embedding, std=0.02)
        # This is the base-preserving switch.  It must remain exactly zero at
        # construction so upstream random attention cannot change old logits.
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        valid_roles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one residual vector per sample and temporal attention.

        Parameters
        ----------
        patch_tokens
            Tensor ``[N,T,P,D]``.  Patches must be spatially corresponding
            between dates.
        valid_roles
            Boolean tensor ``[N,T]``.  Role zero is the current observation and
            must be valid.
        """

        if patch_tokens.ndim != 4:
            raise ValueError("patch_tokens must have shape [N,T,P,D]")
        rows, roles, patches, dim = patch_tokens.shape
        if roles != self.num_roles or dim != self.embed_dim:
            raise ValueError(
                f"expected T={self.num_roles}, D={self.embed_dim}; "
                f"got {tuple(patch_tokens.shape)}"
            )
        if valid_roles.shape != (rows, roles) or valid_roles.dtype != torch.bool:
            raise ValueError("valid_roles must be boolean [N,T]")
        if not valid_roles[:, 0].all():
            raise ValueError("every temporal query needs a valid current role")

        values = patch_tokens + self.role_embedding.to(patch_tokens.dtype)
        # [N,T,P,D] -> [N*P,T,D], preserving patch correspondence.
        context = values.permute(0, 2, 1, 3).reshape(
            rows * patches, roles, dim
        )
        query = values[:, 0].reshape(rows * patches, 1, dim)
        padding_mask = (
            ~valid_roles[:, None, :]
            .expand(rows, patches, roles)
            .reshape(rows * patches, roles)
        )
        attended, attention = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        attended = attended[:, 0].reshape(rows, patches, dim)
        attention = attention[:, 0].reshape(rows, patches, roles)

        # Fixed energy top-k makes the readout plume-local without introducing
        # a learned patch scorer that can immediately overfit acquisition IDs.
        energy = attended.float().square().mean(dim=-1)
        keep = max(1, int(math.ceil(patches * self.topk_fraction)))
        selected = torch.topk(energy, k=keep, dim=1, largest=True).indices
        gather = selected.unsqueeze(-1).expand(rows, keep, dim)
        pooled = attended.gather(1, gather).mean(dim=1)
        residual = self.residual_projection(self.residual_norm(pooled))
        return residual, attention


class MaskedSensorSetResidual(nn.Module):
    """Masked cross-sensor attention whose scalar output starts at zero."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_sensors: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.sensor_embedding = nn.Parameter(
            torch.zeros(1, num_sensors, embed_dim)
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, 1)
        nn.init.normal_(self.sensor_embedding, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        evidence: torch.Tensor,
        valid_sensors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if evidence.ndim != 3:
            raise ValueError("evidence must have shape [B,S,D]")
        if (
            valid_sensors.shape != evidence.shape[:2]
            or valid_sensors.dtype != torch.bool
        ):
            raise ValueError("valid_sensors must be boolean [B,S]")
        if not valid_sensors.any(dim=1).all():
            raise ValueError("every row needs at least one valid current sensor")
        weight = valid_sensors.to(evidence.dtype).unsqueeze(-1)
        query = (evidence * weight).sum(dim=1, keepdim=True) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        context = evidence + self.sensor_embedding.to(evidence.dtype)
        fused, attention = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=~valid_sensors,
            need_weights=True,
            average_attn_weights=True,
        )
        residual = self.output(self.output_norm(fused[:, 0])).squeeze(-1)
        attention = attention[:, 0]
        attention = torch.where(
            valid_sensors, attention, torch.zeros_like(attention)
        )
        return residual, attention


@dataclass
class PatchAxialOutput:
    fused_logits: torch.Tensor
    base_fused_logits: torch.Tensor
    sensor_logits: torch.Tensor
    sensor_valid: torch.Tensor
    temporal_residual_norm: torch.Tensor
    sensor_attention: torch.Tensor


def _two_class_log_odds(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] != 2:
        raise ValueError("legacy classifier must emit exactly two logits")
    return logits[..., 1] - logits[..., 0]


class Legacy360PatchAxial(nn.Module):
    """Panopticon legacy branch plus patch-temporal and sensor-set residuals."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        embed_dim: int = 768,
        sensors: Sequence[str] = SENSOR_ORDER,
        num_roles: int = 3,
        temporal_heads: int = 8,
        sensor_heads: int = 8,
        temporal_frame_blocks: int = 4,
        topk_fraction: float = 0.25,
        base_fusion: str = "max",
        use_sensor_patch_embeds: bool = False,
    ) -> None:
        super().__init__()
        if base_fusion not in {"max", "mean"}:
            raise ValueError("base_fusion must be 'max' or 'mean'")
        self.backbone = backbone
        self.embed_dim = int(embed_dim)
        self.sensors = tuple(sensors)
        self.num_roles = int(num_roles)
        self.temporal_frame_blocks = int(temporal_frame_blocks)
        self.base_fusion = base_fusion
        self.heads = nn.ModuleDict(
            {sensor: LegacyCLSHead(embed_dim) for sensor in self.sensors}
        )
        # Universal-360m checkpoints classify an elementwise-max pooled row
        # feature with a separate head.  Single-sensor checkpoints do not have
        # this module and fall back to their sensor head.
        self.row_fusion_head: Optional[LegacyCLSHead] = None
        self.sensor_patch_embeds = nn.ModuleDict()
        if use_sensor_patch_embeds:
            if not hasattr(backbone, "patch_embed"):
                raise TypeError(
                    "sensor-specific patch embeddings require a ViT backbone"
                )
            self.sensor_patch_embeds = nn.ModuleDict(
                {
                    sensor: copy.deepcopy(backbone.patch_embed)
                    for sensor in self.sensors
                }
            )
        self.temporal = nn.ModuleDict(
            {
                sensor: SameLocationTemporalResidual(
                    embed_dim,
                    num_heads=temporal_heads,
                    num_roles=num_roles,
                    topk_fraction=topk_fraction,
                )
                for sensor in self.sensors
            }
        )
        self.sensor_fusion = MaskedSensorSetResidual(
            embed_dim,
            num_sensors=len(self.sensors),
            num_heads=sensor_heads,
        )

    @property
    def residual_is_zero(self) -> bool:
        parameters = [
            self.sensor_fusion.output.weight,
            self.sensor_fusion.output.bias,
        ]
        for temporal in self.temporal.values():
            parameters.extend(
                [
                    temporal.residual_projection.weight,
                    temporal.residual_projection.bias,
                ]
            )
        return all(bool(torch.count_nonzero(value).item() == 0) for value in parameters)

    def _channel_ids(
        self,
        channel_ids: torch.Tensor,
        *,
        batch_size: int,
        repeats: int = 1,
    ) -> torch.Tensor:
        ids = channel_ids
        if ids.ndim == 1:
            ids = ids.repeat(repeats).unsqueeze(0).expand(batch_size, -1)
        elif ids.ndim == 2 and ids.shape[0] == batch_size:
            ids = ids.repeat(1, repeats)
        elif ids.ndim == 2 and ids.shape[1] in (1, 2):
            ids = ids.repeat(repeats, 1).unsqueeze(0).expand(
                batch_size, -1, -1
            )
        elif ids.ndim == 3 and ids.shape[0] == batch_size:
            ids = ids.repeat(1, repeats, 1)
        else:
            raise ValueError(
                f"cannot normalize channel IDs of shape {tuple(ids.shape)}"
            )
        # Panopticon's wavelength embedding coarsens a local view in place.
        return ids.clone()

    def _forward_vit_with_patch_embed(
        self,
        patch_embed: nn.Module,
        x_dict: Mapping[str, torch.Tensor],
        *,
        max_blocks: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a DINO ViT with an explicitly selected Panopticon patch embed."""

        backbone = self.backbone
        tokens, height, width = patch_embed(dict(x_dict))
        tokens = torch.cat(
            (
                backbone.cls_token.expand(tokens.shape[0], -1, -1),
                tokens,
            ),
            dim=1,
        )
        tokens = tokens + backbone.interpolate_pos_encoding(
            tokens, width, height
        )
        if backbone.register_tokens is not None:
            tokens = torch.cat(
                (
                    tokens[:, :1],
                    backbone.register_tokens.expand(tokens.shape[0], -1, -1),
                    tokens[:, 1:],
                ),
                dim=1,
            )
        limit = len(backbone.blocks) if max_blocks is None else min(
            int(max_blocks), len(backbone.blocks)
        )
        for index, block in enumerate(backbone.blocks):
            if index >= limit:
                break
            tokens = block(tokens)
        if max_blocks is None or limit == len(backbone.blocks):
            tokens = backbone.norm(tokens)
        cls = tokens[:, 0]
        patch = tokens[:, backbone.num_register_tokens + 1 :]
        return cls, patch

    def _encode(
        self,
        sensor: str,
        images: torch.Tensor,
        channel_ids: torch.Tensor,
        *,
        max_blocks: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_dict = {"imgs": images, "chn_ids": channel_ids}
        patch_embed = self.sensor_patch_embeds[sensor] if sensor in self.sensor_patch_embeds else None
        can_use_native = patch_embed is None and (
            max_blocks is None
            or max_blocks >= len(getattr(self.backbone, "blocks", ()))
        )
        if can_use_native:
            output = self.backbone.forward_features(x_dict)
            return output["x_norm_clstoken"], output["x_norm_patchtokens"]
        if patch_embed is None:
            patch_embed = self.backbone.patch_embed
        return self._forward_vit_with_patch_embed(
            patch_embed, x_dict, max_blocks=max_blocks
        )

    @staticmethod
    def _row_positions(batch_indices: torch.Tensor) -> dict[int, int]:
        values = [int(value) for value in batch_indices.detach().cpu().tolist()]
        if len(values) != len(set(values)):
            raise ValueError("batch index contains duplicates")
        return {value: position for position, value in enumerate(values)}

    def _one_sensor(
        self,
        sensor: str,
        sensor_batch: Mapping[str, torch.Tensor],
        *,
        row_positions: Mapping[int, int],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a sensor while retaining both temporal constructions."""

        images = sensor_batch["images"]
        rows = sensor_batch["rows"]
        roles = sensor_batch["roles"]
        ids = sensor_batch["channel_ids"]
        device = images.device
        evidence = images.new_zeros((batch_size, self.embed_dim))
        valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
        residual_norm = images.new_zeros(batch_size)
        if images.shape[0] == 0:
            return evidence, valid, residual_norm

        local_rows = torch.tensor(
            [row_positions[int(value)] for value in rows.detach().cpu().tolist()],
            device=device,
            dtype=torch.long,
        )
        roles = roles.to(device=device, dtype=torch.long)
        by_sample: dict[int, dict[int, int]] = {}
        for observation, (sample, role) in enumerate(
            zip(local_rows.detach().cpu().tolist(), roles.detach().cpu().tolist())
        ):
            by_sample.setdefault(int(sample), {})[int(role)] = observation

        active = sorted(
            sample for sample, mapping in by_sample.items() if 0 in mapping
        )
        if not active:
            return evidence, valid, residual_norm
        active_tensor = torch.tensor(active, device=device, dtype=torch.long)
        valid[active_tensor] = True

        # ------------------------------------------------------------------
        # Historical base: concatenate the valid dates as channels.  Rows are
        # grouped by temporal mask so every Panopticon batch has fixed C.
        # ------------------------------------------------------------------
        base_cls = images.new_zeros((batch_size, self.embed_dim))
        groups: MutableMapping[tuple[int, ...], list[int]] = {}
        for sample in active:
            signature = tuple(sorted(by_sample[sample]))
            groups.setdefault(signature, []).append(sample)
        for signature, samples in groups.items():
            observation_indices = [
                [by_sample[sample][role] for role in signature]
                for sample in samples
            ]
            sample_images = [
                torch.cat(
                    [images[index] for index in row_observations], dim=0
                )
                for row_observations in observation_indices
            ]
            concatenated = torch.stack(sample_images, dim=0)
            concat_ids = self._channel_ids(
                ids,
                batch_size=len(samples),
                repeats=len(signature),
            ).to(device=device)
            cls, _ = self._encode(
                sensor,
                concatenated,
                concat_ids,
                max_blocks=None,
            )
            sample_index = torch.tensor(samples, device=device, dtype=torch.long)
            base_cls = base_cls.index_copy(0, sample_index, cls)

        # ------------------------------------------------------------------
        # Patch-temporal residual: all frames share one encoder call, then t0
        # queries t0/history independently at every spatial patch.
        # ------------------------------------------------------------------
        frame_ids = self._channel_ids(
            ids, batch_size=int(images.shape[0])
        ).to(device=device)
        _, frame_patches = self._encode(
            sensor,
            images,
            frame_ids,
            max_blocks=self.temporal_frame_blocks,
        )
        patch_count = int(frame_patches.shape[1])
        dense_patches = frame_patches.new_zeros(
            (len(active), self.num_roles, patch_count, self.embed_dim)
        )
        role_valid = torch.zeros(
            (len(active), self.num_roles), dtype=torch.bool, device=device
        )
        for active_position, sample in enumerate(active):
            for role, observation in by_sample[sample].items():
                if role < self.num_roles:
                    dense_patches[active_position, role] = frame_patches[observation]
                    role_valid[active_position, role] = True
        residual, _ = self.temporal[sensor](dense_patches, role_valid)
        combined = base_cls.index_select(0, active_tensor) + residual
        evidence = evidence.index_copy(0, active_tensor, combined)
        residual_norm = residual_norm.index_copy(
            0, active_tensor, residual.float().norm(dim=-1).to(residual_norm.dtype)
        )
        return evidence, valid, residual_norm

    def forward(self, batch: Mapping[str, Any]) -> PatchAxialOutput:
        if "index" not in batch or "sensor_batches" not in batch:
            raise ValueError("expected a query360_collate batch")
        batch_size = int(batch["index"].shape[0])
        row_positions = self._row_positions(batch["index"])
        device = next(self.parameters()).device
        evidence = torch.zeros(
            batch_size,
            len(self.sensors),
            self.embed_dim,
            device=device,
            dtype=next(self.parameters()).dtype,
        )
        sensor_valid = torch.zeros(
            batch_size, len(self.sensors), device=device, dtype=torch.bool
        )
        temporal_norm = torch.zeros(
            batch_size, len(self.sensors), device=device
        )
        for sensor_index, sensor in enumerate(self.sensors):
            if sensor not in batch["sensor_batches"]:
                continue
            sensor_evidence, valid, residual = self._one_sensor(
                sensor,
                batch["sensor_batches"][sensor],
                row_positions=row_positions,
                batch_size=batch_size,
            )
            evidence[:, sensor_index] = sensor_evidence
            sensor_valid[:, sensor_index] = valid
            temporal_norm[:, sensor_index] = residual
        if not sensor_valid.any(dim=1).all():
            bad = torch.nonzero(~sensor_valid.any(dim=1), as_tuple=False).flatten()
            raise ValueError(f"rows without a valid current sensor: {bad.tolist()}")

        sensor_logits = torch.stack(
            [
                _two_class_log_odds(self.heads[sensor](evidence[:, index]))
                for index, sensor in enumerate(self.sensors)
            ],
            dim=1,
        )
        sensor_logits = torch.where(
            sensor_valid, sensor_logits, torch.zeros_like(sensor_logits)
        )
        if self.row_fusion_head is not None:
            if self.base_fusion == "max":
                fused_evidence = evidence.masked_fill(
                    ~sensor_valid.unsqueeze(-1), -torch.inf
                ).max(dim=1).values
            else:
                weight = sensor_valid.to(evidence.dtype).unsqueeze(-1)
                fused_evidence = (evidence * weight).sum(dim=1) / weight.sum(
                    dim=1
                ).clamp_min(1.0)
            base = _two_class_log_odds(
                self.row_fusion_head(fused_evidence)
            )
        elif self.base_fusion == "max":
            base = sensor_logits.masked_fill(
                ~sensor_valid, -torch.inf
            ).max(dim=1).values
        else:
            weight = sensor_valid.to(sensor_logits.dtype)
            base = (sensor_logits * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        fusion_residual, sensor_attention = self.sensor_fusion(
            evidence, sensor_valid
        )
        return PatchAxialOutput(
            fused_logits=base + fusion_residual,
            base_fused_logits=base,
            sensor_logits=sensor_logits,
            sensor_valid=sensor_valid,
            temporal_residual_norm=temporal_norm,
            sensor_attention=sensor_attention,
        )


def _strip_prefix(
    state: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {
        str(key)[len(prefix) :]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }


def load_legacy_model(
    checkpoint_path: str | Path,
    *,
    single_sensor_name: str = "s2",
    share_single_head: bool = True,
    base_fusion: Optional[str] = None,
    temporal_frame_blocks: int = 4,
    topk_fraction: float = 0.25,
) -> tuple[Legacy360PatchAxial, dict[str, Any]]:
    """Build the model and load one of the three historical checkpoint forms."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{checkpoint}: checkpoint must be a mapping")
    backbone = build_panopticon_vitb14()
    kind: str
    universal_state: Optional[Mapping[str, torch.Tensor]] = None
    single_head_state: Optional[Mapping[str, torch.Tensor]] = None
    if "model" in payload and isinstance(payload["model"], Mapping):
        universal_state = payload["model"]
        backbone_state = _strip_prefix(universal_state, "backbone.")
        if not backbone_state:
            raise ValueError(f"{checkpoint}: model has no backbone.* state")
        backbone.load_state_dict(backbone_state, strict=True)
        kind = "universal_360m"
    elif "backbone" in payload and isinstance(payload["backbone"], Mapping):
        backbone.load_state_dict(payload["backbone"], strict=True)
        if "head" in payload and isinstance(payload["head"], Mapping):
            single_head_state = payload["head"]
        kind = "single_sensor"
    elif payload and all(torch.is_tensor(value) for value in payload.values()):
        backbone.load_state_dict(payload, strict=True)
        kind = "raw_panopticon"
    else:
        raise ValueError(f"{checkpoint}: unsupported checkpoint structure")

    requested_fusion = base_fusion
    if requested_fusion is None:
        requested_fusion = "max"
        if isinstance(payload.get("args"), Mapping):
            legacy_mode = str(payload["args"].get("row_fusion_mode", "max"))
            if legacy_mode in {"max", "mean"}:
                requested_fusion = legacy_mode
    model = Legacy360PatchAxial(
        backbone,
        temporal_frame_blocks=temporal_frame_blocks,
        topk_fraction=topk_fraction,
        base_fusion=requested_fusion,
        use_sensor_patch_embeds=universal_state is not None,
    )
    loaded_heads: list[str] = []
    loaded_row_fusion_head = False
    if universal_state is not None:
        for sensor, source_name in UNIVERSAL_SENSOR_NAMES.items():
            patch_state = _strip_prefix(
                universal_state, f"sensor_patch_embeds.{source_name}."
            )
            head_state = _strip_prefix(
                universal_state, f"heads.{source_name}."
            )
            if not patch_state or not head_state:
                raise ValueError(
                    f"{checkpoint}: missing universal state for {source_name}"
                )
            model.sensor_patch_embeds[sensor].load_state_dict(
                patch_state, strict=True
            )
            model.heads[sensor].load_state_dict(head_state, strict=True)
            loaded_heads.append(sensor)
        row_head_state = _strip_prefix(universal_state, "row_fusion_head.")
        if row_head_state:
            model.row_fusion_head = LegacyCLSHead(model.embed_dim)
            model.row_fusion_head.load_state_dict(row_head_state, strict=True)
            loaded_row_fusion_head = True
    elif single_head_state is not None:
        if single_sensor_name not in model.heads:
            raise ValueError(f"unknown single sensor {single_sensor_name!r}")
        targets = model.sensors if share_single_head else (single_sensor_name,)
        for sensor in targets:
            model.heads[sensor].load_state_dict(single_head_state, strict=True)
            loaded_heads.append(sensor)
    if not model.residual_is_zero:
        raise AssertionError("new residual paths must initialize to exact zero")
    provenance = {
        "checkpoint": str(checkpoint),
        "checkpoint_kind": kind,
        "checkpoint_epoch": payload.get("epoch"),
        "legacy_best_test_acc": payload.get("best_test_acc"),
        "base_fusion": requested_fusion,
        "loaded_heads": loaded_heads,
        "loaded_row_fusion_head": loaded_row_fusion_head,
        "single_sensor_name": single_sensor_name if kind == "single_sensor" else None,
        "share_single_head": bool(share_single_head),
        "temporal_frame_blocks": int(temporal_frame_blocks),
        "topk_fraction": float(topk_fraction),
        "zero_residual_verified": True,
    }
    return model, provenance


def move_batch(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {key: move_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch(value, device) for value in batch)
    return batch


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prediction = (probabilities >= float(threshold)).astype(np.int64)
    output = {
        "threshold": float(threshold),
        "binary_f1": float(
            f1_score(labels, prediction, pos_label=1, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, prediction)
        ),
        "positive_rate": float(prediction.mean()),
    }
    if len(np.unique(labels)) == 2:
        output["ap"] = float(average_precision_score(labels, probabilities))
        output["auc"] = float(roc_auc_score(labels, probabilities))
    else:
        output["ap"] = 0.0
        output["auc"] = 0.5
    return output


def select_f1_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, dict[str, float]]:
    candidates = np.unique(
        np.concatenate(
            (
                np.linspace(0.01, 0.99, 197, dtype=np.float64),
                np.asarray(probabilities, dtype=np.float64),
                np.asarray([0.5]),
            )
        )
    )
    best_threshold = 0.5
    best_metrics = classification_metrics(
        labels, probabilities, threshold=best_threshold
    )
    for threshold in candidates:
        metrics = classification_metrics(
            labels, probabilities, threshold=float(threshold)
        )
        key = (
            metrics["binary_f1"],
            metrics["macro_f1"],
            -abs(float(threshold) - 0.5),
        )
        best_key = (
            best_metrics["binary_f1"],
            best_metrics["macro_f1"],
            -abs(best_threshold - 0.5),
        )
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def evaluate(
    model: Legacy360PatchAxial,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with _autocast(device, amp):
                output = model(batch)
            labels.append(batch["labels"].detach().cpu())
            probabilities.append(
                torch.sigmoid(output.fused_logits.float()).detach().cpu()
            )
    return torch.cat(labels).numpy(), torch.cat(probabilities).numpy()


def _make_dataset(
    csv_path: str,
    *,
    local_cache: Optional[StrictHashedFileCache],
    wv3_srf_csv: str,
) -> Query360Dataset:
    return Query360Dataset(
        csv_path,
        local_cache=local_cache,
        wv3_srf_csv=wv3_srf_csv,
        pad_to_multiple=14,
    )


def _make_loader(
    dataset: Query360Dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=query360_collate,
        drop_last=False,
    )


def _warm_cache(
    cache: StrictHashedFileCache,
    csv_paths: Sequence[str],
    *,
    workers: int,
) -> dict[str, Any]:
    paths: set[str] = set()
    for csv_path in csv_paths:
        frame = pd.read_csv(
            csv_path,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
            usecols=lambda name: name in CLASSIFICATION_PATH_COLUMNS,
        )
        paths.update(collect_classification_paths(frame))
    report = cache.warm_up(sorted(paths), max_workers=workers)
    return asdict(report)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run_status.json"
    atomic_json(
        status_path,
        {
            "status": "running",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "script_version": SCRIPT_VERSION,
        },
    )
    try:
        if args.sealed_test_authorization != SEALED_TEST_AUTHORIZATION:
            raise PermissionError(
                "This runner will open the historical test manifest only after "
                "development selection. Pass the explicit authorization token: "
                f"--sealed_test_authorization {SEALED_TEST_AUTHORIZATION}"
            )
        local_cache = (
            StrictHashedFileCache(args.raw_cache_dir)
            if args.raw_cache_dir
            else None
        )
        cache_report: dict[str, Any] = {}
        if local_cache is not None and args.warm_cache:
            cache_report["development"] = _warm_cache(
                local_cache,
                (args.train_csv, args.dev_csv),
                workers=args.cache_workers,
            )
            atomic_json(output_dir / "cache_report.json", cache_report)
        train_dataset = _make_dataset(
            args.train_csv,
            local_cache=local_cache,
            wv3_srf_csv=args.wv3_srf_csv,
        )
        dev_dataset = _make_dataset(
            args.dev_csv,
            local_cache=local_cache,
            wv3_srf_csv=args.wv3_srf_csv,
        )
        train_plumes = set(train_dataset.frame["plume_id"].astype(str))
        dev_plumes = set(dev_dataset.frame["plume_id"].astype(str))
        if train_plumes & dev_plumes:
            raise RuntimeError("train/development plume IDs must be disjoint")
        train_loader = _make_loader(
            train_dataset,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=True,
            device=device,
        )
        dev_loader = _make_loader(
            dev_dataset,
            batch_size=args.eval_batch_size,
            workers=args.workers,
            shuffle=False,
            device=device,
        )
        model, provenance = load_legacy_model(
            args.checkpoint,
            single_sensor_name=args.single_sensor_name,
            share_single_head=args.share_single_head,
            base_fusion=args.base_fusion,
            temporal_frame_blocks=args.temporal_frame_blocks,
            topk_fraction=args.topk_fraction,
        )
        model.to(device)
        backbone_parameters = list(model.backbone.parameters()) + list(
            model.sensor_patch_embeds.parameters()
        )
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        residual_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in backbone_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": args.backbone_lr,
                },
                {
                    "params": residual_parameters,
                    "lr": args.residual_lr,
                },
            ],
            weight_decay=args.weight_decay,
        )
        positives = float(train_dataset.frame["label"].astype(int).sum())
        negatives = float(len(train_dataset) - positives)
        pos_weight = (
            torch.tensor(negatives / max(positives, 1.0), device=device)
            if args.pos_weight
            else None
        )
        history: list[dict[str, Any]] = []
        best: Optional[dict[str, Any]] = None
        best_state: Optional[dict[str, torch.Tensor]] = None
        started = time.monotonic()
        for epoch in range(1, args.epochs + 1):
            backbone_trainable = epoch > args.freeze_backbone_epochs
            for parameter in backbone_parameters:
                parameter.requires_grad_(backbone_trainable)
            model.train()
            if not backbone_trainable:
                model.backbone.eval()
                model.sensor_patch_embeds.eval()
            running_loss = 0.0
            running_main = 0.0
            running_aux = 0.0
            seen = 0
            for step, raw_batch in enumerate(train_loader, start=1):
                batch = move_batch(raw_batch, device)
                target = batch["labels"].float()
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, args.amp):
                    output = model(batch)
                    main_loss = F.binary_cross_entropy_with_logits(
                        output.fused_logits,
                        target,
                        pos_weight=pos_weight,
                    )
                    expanded_target = target[:, None].expand_as(
                        output.sensor_logits
                    )
                    per_sensor = F.binary_cross_entropy_with_logits(
                        output.sensor_logits,
                        expanded_target,
                        reduction="none",
                        pos_weight=pos_weight,
                    )
                    mask = output.sensor_valid.to(per_sensor.dtype)
                    aux_loss = (per_sensor * mask).sum() / mask.sum().clamp_min(1.0)
                    loss = main_loss + args.sensor_aux_weight * aux_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    args.grad_clip,
                )
                optimizer.step()
                count = int(target.numel())
                running_loss += float(loss.detach()) * count
                running_main += float(main_loss.detach()) * count
                running_aux += float(aux_loss.detach()) * count
                seen += count
                if args.max_train_steps and step >= args.max_train_steps:
                    break
            dev_labels, dev_probabilities = evaluate(
                model, dev_loader, device=device, amp=args.amp
            )
            threshold, dev_metrics = select_f1_threshold(
                dev_labels, dev_probabilities
            )
            record = {
                "epoch": epoch,
                "train": {
                    "loss": running_loss / max(seen, 1),
                    "main_loss": running_main / max(seen, 1),
                    "sensor_aux_loss": running_aux / max(seen, 1),
                    "rows": seen,
                    "backbone_trainable": backbone_trainable,
                },
                "development": dev_metrics,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(record)
            atomic_json(output_dir / "metrics_history.json", history)
            score = (
                dev_metrics["binary_f1"]
                if args.selection_metric == "binary_f1"
                else dev_metrics["macro_f1"]
            )
            if best is None or score > (
                best["development"][args.selection_metric]
            ):
                best = copy.deepcopy(record)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                atomic_torch(
                    output_dir / "checkpoint_best.pth",
                    {
                        "schema_version": SCRIPT_VERSION,
                        "epoch": epoch,
                        "threshold": threshold,
                        "provenance": provenance,
                        "model": best_state,
                        "development": dev_metrics,
                    },
                )
            print(
                f"[patch-axial] epoch={epoch}/{args.epochs} "
                f"loss={record['train']['loss']:.5f} "
                f"dev_binary_F1={dev_metrics['binary_f1']:.5f} "
                f"dev_macro_F1={dev_metrics['macro_f1']:.5f} "
                f"threshold={threshold:.4f} AP={dev_metrics['ap']:.5f}",
                flush=True,
            )
        if best is None or best_state is None:
            raise RuntimeError("no training epoch completed")
        model.load_state_dict(best_state, strict=True)
        best_threshold = float(best["development"]["threshold"])

        # Deliberately defer every test-manifest/cache read until architecture,
        # epoch, and threshold have been frozen on development.  The warm-start
        # checkpoint itself was historically selected with evaluation feedback,
        # so this remains an engineering protocol, not a leakage-free estimate.
        if local_cache is not None and args.warm_cache:
            cache_report["final_evaluation"] = _warm_cache(
                local_cache,
                (args.test_csv,),
                workers=args.cache_workers,
            )
            atomic_json(output_dir / "cache_report.json", cache_report)
        test_dataset = AuthorizedSealedQuery360Dataset(
            args.test_csv,
            authorization=args.sealed_test_authorization,
            local_cache=local_cache,
            wv3_srf_csv=args.wv3_srf_csv,
            pad_to_multiple=14,
        )
        test_loader = _make_loader(
            test_dataset,
            batch_size=args.eval_batch_size,
            workers=args.workers,
            shuffle=False,
            device=device,
        )
        test_plumes = set(test_dataset.frame["plume_id"].astype(str))
        train_events = {
            canonical_event(value) for value in train_dataset.frame["plume_id"]
        }
        dev_events = {
            canonical_event(value) for value in dev_dataset.frame["plume_id"]
        }
        test_events = {
            canonical_event(value) for value in test_dataset.frame["plume_id"]
        }
        overlap_audit = {
            "exact_plume_id": {
                "train_development": len(train_plumes & dev_plumes),
                "train_test": len(train_plumes & test_plumes),
                "development_test": len(dev_plumes & test_plumes),
            },
            "canonical_event_strip_final_suffix": {
                "train_development": len(train_events & dev_events),
                "train_test": len(train_events & test_events),
                "development_test": len(dev_events & test_events),
            },
        }
        test_labels, test_probabilities = evaluate(
            model, test_loader, device=device, amp=args.amp
        )
        test_metrics = classification_metrics(
            test_labels,
            test_probabilities,
            threshold=best_threshold,
        )
        test_at_half = classification_metrics(
            test_labels,
            test_probabilities,
            threshold=0.5,
        )
        summary = {
            "schema_version": SCRIPT_VERSION,
            "status": "complete",
            "protocol": "engineering_historical_warm_start_not_leakage_free",
            "protocol_warning": (
                "The supplied historical PTH was trained/selected under the old "
                "evaluation loop and the canonical-event audit may overlap the "
                "historical train/test split. Results are iteration evidence only, "
                "not a leakage-free SOTA claim."
            ),
            "selection": (
                "new residual epoch and threshold selected on development; "
                "historical test manifest opened once afterward"
            ),
            "overlap_audit": overlap_audit,
            "rows": {
                "train": len(train_dataset),
                "development": len(dev_dataset),
                "test": len(test_dataset),
            },
            "provenance": provenance,
            "best_development": best,
            "test_at_dev_threshold": test_metrics,
            "test_at_0_5": test_at_half,
            "cache_report": cache_report or None,
        }
        atomic_json(output_dir / "summary.json", summary)
        pd.DataFrame(
            {
                "id": test_dataset.frame["id"].astype(str).tolist(),
                "plume_id": test_dataset.frame["plume_id"].astype(str).tolist(),
                "label": test_labels,
                "probability": test_probabilities,
            }
        ).to_csv(output_dir / "test_predictions.csv", index=False)
        atomic_json(
            status_path,
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "best_epoch": best["epoch"],
                "test_binary_f1": test_metrics["binary_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
            },
        )
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        atomic_json(
            status_path,
            {
                "status": "failed",
                "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--dev_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument(
        "--sealed_test_authorization",
        default="",
        help=(
            "Required exact token for the one final historical evaluation: "
            f"{SEALED_TEST_AUTHORIZATION}"
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backbone_lr", type=float, default=1.0e-5)
    parser.add_argument("--residual_lr", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=1)
    parser.add_argument("--sensor_aux_weight", type=float, default=0.2)
    parser.add_argument("--temporal_frame_blocks", type=int, default=4)
    parser.add_argument("--topk_fraction", type=float, default=0.25)
    parser.add_argument(
        "--base_fusion",
        choices=("max", "mean"),
        default=None,
        help="Defaults to the mode recorded by the source checkpoint.",
    )
    parser.add_argument(
        "--selection_metric",
        choices=("binary_f1", "macro_f1"),
        default="binary_f1",
    )
    parser.add_argument("--single_sensor_name", choices=SENSOR_ORDER, default="s2")
    parser.add_argument(
        "--share_single_head",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pos_weight",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--raw_cache_dir", default="")
    parser.add_argument(
        "--warm_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cache_workers", type=int, default=32)
    parser.add_argument("--wv3_srf_csv", default=str(DEFAULT_WV3_SRF))
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
