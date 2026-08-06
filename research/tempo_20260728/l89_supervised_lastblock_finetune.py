#!/usr/bin/env python3
"""Exploratory supervised last-block L89 matched control.

This runner is deliberately development-only.  It accepts an event-disjoint
L89 train/development pair, refuses paths containing test/sealed/holdout, and
has no evaluation-only or test subcommand.

The two arms share one frozen Panopticon image trunk (patch embedding and ViT
blocks 0--10).  The shared trunk is executed exactly once under ``no_grad``.
Each arm owns a byte-identical copy of block 11, final norm, t0 head, and
``TEMPOGlobalResidual``:

``current_only``
    suppresses every history slot before the TEMPO residual;
``temporal_d1``
    exposes the audited irregular six-visit history to gated signed/absolute
    current-minus-history evidence.

Both TEMPO residuals are zero initialized, so the complete arms have identical
state and emit bit-identical logits at epoch 0.  During training they see the
same event-by-label sampled rows and the same stochastic masks.  Thus the
history mask is the only arm intervention.
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
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, SequentialSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid optional xFormers discovery in the inherited Panopticon code.
os.environ.setdefault("XFORMERS_DISABLED", "1")

from Upgraded_dataset.dino_classifier_head_l89_temporal_satmae import (  # noqa: E402
    load_backbone,
)
from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as l89,
)
from research.tempo_20260728 import tempo_l89_global as tempo  # noqa: E402


SCRIPT_VERSION = "l89-supervised-lastblock-matched-v1"
ARM_NAMES = ("current_only", "temporal_d1")
EXPECTED_BACKBONE_BLOCKS = 12
MAX_EPOCHS = 3
MAX_PATIENCE = 1
MAX_DRAWS_PER_EPOCH = 4096
MAX_BATCH_SIZE = 24
MAX_WALL_LIMIT_MINUTES = 45.0
MAX_CUDA_ALLOCATED_GIB = 28.0


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sha256_file(path: Path) -> str:
    return l89.sha256_file(path)


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return l89.state_dict_sha256(state)


def assert_development_path(path: Path, *, purpose: str) -> Path:
    resolved = path.expanduser().resolve()
    tempo.assert_development_path(resolved, purpose=purpose)
    return resolved


class RuntimeGuard:
    """Fail closed on wall time or this process's CUDA allocation peak."""

    def __init__(
        self,
        *,
        device: torch.device,
        wall_limit_minutes: float,
        max_cuda_allocated_gib: float,
        clock: Callable[[], float] = time.monotonic,
        cuda_peak_reader: Optional[Callable[[torch.device], int]] = None,
        cuda_peak_resetter: Optional[Callable[[torch.device], None]] = None,
    ):
        self.device = torch.device(device)
        self.wall_limit_seconds = float(wall_limit_minutes) * 60.0
        self.max_cuda_allocated_bytes = float(max_cuda_allocated_gib) * (
            2**30
        )
        self.clock = clock
        self.started = float(clock())
        self.check_count = 0
        self.last_phase = ""
        self.maximum_observed_cuda_allocated_bytes = 0
        if cuda_peak_reader is None:
            cuda_peak_reader = lambda value: int(
                torch.cuda.max_memory_allocated(value)
            )
        self.cuda_peak_reader = cuda_peak_reader
        if cuda_peak_resetter is None:
            cuda_peak_resetter = torch.cuda.reset_peak_memory_stats
        self.cuda_peak_resetter = cuda_peak_resetter
        if self.device.type == "cuda":
            self.cuda_peak_resetter(self.device)

    def check(self, phase: str) -> dict[str, Any]:
        self.check_count += 1
        self.last_phase = str(phase)
        elapsed = float(self.clock()) - self.started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError(
                f"Runtime guard observed invalid elapsed time at {phase!r}."
            )
        if elapsed > self.wall_limit_seconds:
            raise RuntimeError(
                "Wall-time guard exceeded after "
                f"{phase}: {elapsed / 60.0:.3f} min > "
                f"{self.wall_limit_seconds / 60.0:.3f} min."
            )
        peak = 0
        if self.device.type == "cuda":
            peak = int(self.cuda_peak_reader(self.device))
            self.maximum_observed_cuda_allocated_bytes = max(
                self.maximum_observed_cuda_allocated_bytes, peak
            )
            if peak > self.max_cuda_allocated_bytes:
                raise RuntimeError(
                    "CUDA allocation guard exceeded after "
                    f"{phase}: {peak / 2**30:.3f} GiB > "
                    f"{self.max_cuda_allocated_bytes / 2**30:.3f} GiB."
                )
        return {
            "phase": self.last_phase,
            "check_count": self.check_count,
            "elapsed_seconds": elapsed,
            "cuda_peak_allocated_bytes": peak,
        }

    def audit(self) -> dict[str, Any]:
        return {
            "wall_limit_minutes": self.wall_limit_seconds / 60.0,
            "max_cuda_allocated_gib": self.max_cuda_allocated_bytes / 2**30,
            "check_count": self.check_count,
            "last_phase": self.last_phase,
            "maximum_observed_cuda_allocated_gib": (
                self.maximum_observed_cuda_allocated_bytes / 2**30
            ),
        }


def _rng_state(device: torch.device) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    cpu_state = torch.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
    return cpu_state, cuda_state


def _restore_rng_state(
    state: tuple[torch.Tensor, Optional[torch.Tensor]], device: torch.device
) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if device.type == "cuda":
        if cuda_state is None:
            raise RuntimeError("CUDA RNG state is absent.")
        torch.cuda.set_rng_state(cuda_state, device)


class EventLabelSampler(Sampler[int]):
    """Deterministic balanced sampling over canonical event-by-label strata.

    Labels alternate exactly.  Within each label, an event-label stratum is
    sampled uniformly and then one row is sampled uniformly from that stratum.
    Mixed-label events therefore contribute one stratum to each observed label,
    instead of having all their rows pooled together.
    """

    def __init__(
        self,
        labels: Sequence[int] | torch.Tensor,
        event_ids: Sequence[str],
        *,
        draws_per_epoch: int,
        seed: int,
        indices: Optional[Sequence[int]] = None,
    ):
        label_values = torch.as_tensor(labels, dtype=torch.long).cpu()
        if label_values.ndim != 1 or len(label_values) != len(event_ids):
            raise ValueError("labels and event_ids must be aligned vectors.")
        if indices is None:
            selected = list(range(len(event_ids)))
        else:
            selected = [int(value) for value in indices]
        if not selected or min(selected) < 0 or max(selected) >= len(event_ids):
            raise ValueError("indices are empty or out of range.")
        if not 1 <= int(draws_per_epoch) <= MAX_DRAWS_PER_EPOCH:
            raise ValueError(
                f"draws_per_epoch must be in 1..{MAX_DRAWS_PER_EPOCH}."
            )

        strata: dict[int, dict[str, list[int]]] = {0: {}, 1: {}}
        for index in selected:
            label = int(label_values[index])
            if label not in strata:
                raise ValueError(f"Expected binary labels, got {label}.")
            event = str(event_ids[index]).strip()
            if not event:
                raise ValueError("event_ids cannot contain blank values.")
            strata[label].setdefault(event, []).append(index)
        if not strata[0] or not strata[1]:
            raise ValueError("Event-label sampling requires both labels.")

        self.strata = strata
        self.draws_per_epoch = int(draws_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def order(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * self.epoch
        )
        # Randomize which class receives the possible odd final draw while
        # preserving exact alternation and a maximum imbalance of one.
        first_label = int(
            torch.randint(0, 2, (1,), generator=generator).item()
        )
        event_names = {
            label: sorted(self.strata[label]) for label in (0, 1)
        }
        sampled: list[int] = []
        for draw in range(self.draws_per_epoch):
            label = (first_label + draw) % 2
            events = event_names[label]
            event_position = int(
                torch.randint(
                    0, len(events), (1,), generator=generator
                ).item()
            )
            rows = self.strata[label][events[event_position]]
            row_position = int(
                torch.randint(0, len(rows), (1,), generator=generator).item()
            )
            sampled.append(int(rows[row_position]))
        return torch.tensor(sampled, dtype=torch.long)

    def __iter__(self) -> Iterator[int]:
        return iter(self.order().tolist())

    def __len__(self) -> int:
        return self.draws_per_epoch

    def order_sha256(self) -> str:
        order = self.order().contiguous().numpy().tobytes()
        return hashlib.sha256(order).hexdigest()

    def audit(self) -> dict[str, Any]:
        event_sets = {
            label: set(self.strata[label]) for label in (0, 1)
        }
        mixed = event_sets[0] & event_sets[1]
        return {
            "algorithm": (
                "alternate-label/uniform-event-label-stratum/"
                "uniform-row-within-stratum-v1"
            ),
            "draws_per_epoch": self.draws_per_epoch,
            "seed": self.seed,
            "strata_by_label": {
                str(label): len(self.strata[label]) for label in (0, 1)
            },
            "mixed_label_events": len(mixed),
            "event_label_strata": sum(
                len(self.strata[label]) for label in (0, 1)
            ),
        }


class MatchedLastBlockArm(nn.Module):
    """One trainable block-11 branch with a t0 base and TEMPO residual."""

    def __init__(
        self,
        final_block: nn.Module,
        final_norm: nn.Module,
        *,
        feature_dim: int,
        num_roles: int,
        t0_index: int,
        head_hidden_dim: int,
        temporal_dim: int,
        dropout: float,
        periods_days: Sequence[float],
    ):
        super().__init__()
        self.final_block = copy.deepcopy(final_block)
        self.final_norm = copy.deepcopy(final_norm)
        self.t0_index = int(t0_index)
        self.base_head = tempo.T0BaseHead(
            feature_dim=int(feature_dim),
            hidden_dim=int(head_hidden_dim),
            dropout=float(dropout),
        )
        self.temporal = tempo.TEMPOGlobalResidual(
            tempo.ZeroLogitBase(),
            feature_dim=int(feature_dim),
            num_roles=int(num_roles),
            t0_index=int(t0_index),
            temporal_dim=int(temporal_dim),
            dropout=float(dropout),
            periods_days=tuple(float(value) for value in periods_days),
        )

    def encode_last_block(self, trunk_tokens: torch.Tensor) -> torch.Tensor:
        if trunk_tokens.ndim != 4:
            raise ValueError("trunk_tokens must have shape [B,T,L,D].")
        rows, roles, tokens, width = trunk_tokens.shape
        flat = trunk_tokens.reshape(rows * roles, tokens, width)
        encoded = self.final_norm(self.final_block(flat))
        return encoded[:, 0].reshape(rows, roles, width)

    def forward(
        self,
        trunk_tokens: torch.Tensor,
        unique_mask: torch.Tensor,
        delta_days: torch.Tensor,
        valid_fraction: torch.Tensor,
        role_index: torch.Tensor,
        *,
        temporal_enabled: bool,
    ) -> torch.Tensor:
        features = self.encode_last_block(trunk_tokens)
        base_logit = self.base_head(features[:, self.t0_index])
        effective_mask = unique_mask.bool()
        if not temporal_enabled:
            effective_mask = torch.zeros_like(effective_mask)
            effective_mask[:, self.t0_index] = unique_mask[:, self.t0_index]
        return self.temporal(
            features,
            effective_mask,
            delta_days,
            valid_fraction,
            role_index,
            arm="d1_gated_delta",
            base_logit_override=base_logit,
        )


class MatchedLastBlockPair(nn.Module):
    """Shared frozen blocks 0--10 and two byte-identical block-11 arms."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_roles: int,
        t0_index: int,
        head_hidden_dim: int = 256,
        temporal_dim: int = 192,
        dropout: float = 0.15,
        periods_days: Sequence[float] = (1, 3, 7, 30, 90, 365),
    ):
        super().__init__()
        blocks = getattr(backbone, "blocks", None)
        if (
            not isinstance(blocks, nn.ModuleList)
            or len(blocks) != EXPECTED_BACKBONE_BLOCKS
        ):
            raise ValueError(
                "Expected the unchunked Panopticon ViT-B/14 with exactly "
                f"{EXPECTED_BACKBONE_BLOCKS} blocks."
            )
        feature_dim = int(getattr(backbone, "embed_dim"))
        source_final_block = copy.deepcopy(blocks[-1])
        source_final_norm = copy.deepcopy(backbone.norm)

        # The original final block/norm are removed from the sole shared
        # backbone.  No dormant third copy can accidentally become trainable.
        backbone.blocks = nn.ModuleList(list(blocks[:-1]))
        backbone.norm = nn.Identity()
        backbone.requires_grad_(False)
        backbone.eval()
        self.trunk = backbone
        self.num_roles = int(num_roles)
        self.t0_index = int(t0_index)
        self.feature_dim = feature_dim

        prototype = MatchedLastBlockArm(
            source_final_block,
            source_final_norm,
            feature_dim=feature_dim,
            num_roles=self.num_roles,
            t0_index=self.t0_index,
            head_hidden_dim=int(head_hidden_dim),
            temporal_dim=int(temporal_dim),
            dropout=float(dropout),
            periods_days=periods_days,
        )
        self.current_only = copy.deepcopy(prototype)
        self.temporal_d1 = copy.deepcopy(prototype)
        if self.arm_state_sha256()["current_only"] != self.arm_state_sha256()[
            "temporal_d1"
        ]:
            raise AssertionError("Matched arms did not start byte-identically.")
        self._assert_trainability()

    def _assert_trainability(self) -> None:
        if any(parameter.requires_grad for parameter in self.trunk.parameters()):
            raise AssertionError("Shared blocks 0--10 must remain frozen.")
        for arm_name in ARM_NAMES:
            arm = getattr(self, arm_name)
            if not any(
                parameter.requires_grad
                for parameter in arm.final_block.parameters()
            ):
                raise AssertionError(f"{arm_name} final block is not trainable.")
            norm_parameters = list(arm.final_norm.parameters())
            if not norm_parameters or not all(
                parameter.requires_grad for parameter in norm_parameters
            ):
                raise AssertionError(
                    f"{arm_name} final norm must retain gradients."
                )

    def train(self, mode: bool = True) -> "MatchedLastBlockPair":
        super().train(mode)
        self.trunk.eval()
        return self

    def arm_state_sha256(self) -> dict[str, str]:
        return {
            name: state_dict_sha256(getattr(self, name).state_dict())
            for name in ARM_NAMES
        }

    def encode_shared_trunk(
        self,
        images: torch.Tensor,
        channel_ids: torch.Tensor,
        *,
        encoder_microbatch: int,
        device: torch.device,
        amp_dtype: str,
    ) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("images must have shape [B,T,C,H,W].")
        rows, roles, channels, height, width = images.shape
        if roles != self.num_roles:
            raise ValueError(
                f"Expected {self.num_roles} roles, received {roles}."
            )
        channel_ids = channel_ids.reshape(-1)
        if channel_ids.numel() != channels:
            raise ValueError("channel_ids do not match image channels.")
        flat = images.reshape(rows * roles, channels, height, width)
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(flat), int(encoder_microbatch)):
                stop = min(start + int(encoder_microbatch), len(flat))
                image_chunk = flat[start:stop].to(
                    device, non_blocking=device.type == "cuda"
                )
                id_chunk = (
                    channel_ids.view(1, -1)
                    .expand(stop - start, -1)
                    .clone()
                    .to(device, non_blocking=device.type == "cuda")
                )
                with l89.autocast_context(device, amp_dtype):
                    tokens = self.trunk.prepare_tokens_with_masks(
                        {"imgs": image_chunk, "chn_ids": id_chunk}
                    )
                    for block in self.trunk.blocks:
                        tokens = block(tokens)
                outputs.append(tokens)
        result = torch.cat(outputs, dim=0).reshape(
            rows, roles, outputs[0].shape[1], self.feature_dim
        )
        if result.requires_grad:
            raise AssertionError("Frozen shared trunk unexpectedly retained a graph.")
        return result

    def forward_from_trunk(
        self,
        trunk_tokens: torch.Tensor,
        unique_mask: torch.Tensor,
        delta_days: torch.Tensor,
        valid_fraction: torch.Tensor,
        role_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = trunk_tokens.device
        initial_rng = _rng_state(device)
        current = self.current_only(
            trunk_tokens,
            unique_mask,
            delta_days,
            valid_fraction,
            role_index,
            temporal_enabled=False,
        )
        # Replaying the RNG makes dropout/drop-path masks identical in every
        # shape-matched operation.  The history mask remains the only
        # intervention.
        _restore_rng_state(initial_rng, device)
        temporal_logit = self.temporal_d1(
            trunk_tokens,
            unique_mask,
            delta_days,
            valid_fraction,
            role_index,
            temporal_enabled=True,
        )
        return {"current_only": current, "temporal_d1": temporal_logit}

    def forward_images(
        self,
        images: torch.Tensor,
        channel_ids: torch.Tensor,
        unique_mask: torch.Tensor,
        delta_days: torch.Tensor,
        valid_fraction: torch.Tensor,
        role_index: torch.Tensor,
        *,
        encoder_microbatch: int,
        device: torch.device,
        amp_dtype: str,
    ) -> dict[str, torch.Tensor]:
        trunk_tokens = self.encode_shared_trunk(
            images,
            channel_ids,
            encoder_microbatch=encoder_microbatch,
            device=device,
            amp_dtype=amp_dtype,
        )
        with l89.autocast_context(device, amp_dtype):
            return self.forward_from_trunk(
                trunk_tokens,
                unique_mask.to(device),
                delta_days.to(device),
                valid_fraction.to(device),
                role_index.to(device),
            )


def validate_training_budget(args: argparse.Namespace) -> None:
    if not 1 <= int(args.epochs) <= MAX_EPOCHS:
        raise ValueError(f"--epochs must be in 1..{MAX_EPOCHS}.")
    if not 1 <= int(args.patience) <= MAX_PATIENCE:
        raise ValueError(f"--patience must be exactly {MAX_PATIENCE}.")
    if not 1 <= int(args.draws_per_epoch) <= MAX_DRAWS_PER_EPOCH:
        raise ValueError(
            f"--draws-per-epoch must be in 1..{MAX_DRAWS_PER_EPOCH}."
        )
    if not 1 <= int(args.batch_size) <= MAX_BATCH_SIZE:
        raise ValueError(
            f"--batch-size must be in 1..{MAX_BATCH_SIZE}."
        )
    if int(args.encoder_microbatch) < 1:
        raise ValueError("--encoder-microbatch must be positive.")
    for name in ("block_lr", "head_lr", "weight_decay", "grad_clip"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite/non-negative.")
    if float(args.block_lr) <= 0 or float(args.head_lr) <= 0:
        raise ValueError("Learning rates must be positive.")
    wall_limit = float(args.wall_limit_minutes)
    if (
        not math.isfinite(wall_limit)
        or wall_limit <= 0
        or wall_limit > MAX_WALL_LIMIT_MINUTES
    ):
        raise ValueError(
            "--wall-limit-minutes must be positive and no greater than "
            f"{MAX_WALL_LIMIT_MINUTES:g}."
        )
    cuda_limit = float(args.max_cuda_allocated_gib)
    if (
        not math.isfinite(cuda_limit)
        or cuda_limit <= 0
        or cuda_limit > MAX_CUDA_ALLOCATED_GIB
    ):
        raise ValueError(
            "--max-cuda-allocated-gib must be positive and no greater than "
            f"{MAX_CUDA_ALLOCATED_GIB:g}."
        )


def _validate_all_local_paths(
    frame: pd.DataFrame,
    path_columns: Sequence[str],
    *,
    required_root: Optional[Path],
) -> dict[str, Any]:
    paths: set[Path] = set()
    root = required_root.expanduser().resolve() if required_root else None
    for column in path_columns:
        for value in frame[column].fillna("").astype(str):
            if not value.strip():
                continue
            path = Path(os.path.abspath(os.path.expanduser(value.strip())))
            if root is not None:
                try:
                    path.relative_to(root)
                except ValueError as exc:
                    raise ValueError(
                        f"{column} escapes required local root {root}: {path}"
                    ) from exc
            paths.add(path)
    missing = [str(path) for path in sorted(paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} declared image files are missing; "
            f"examples={missing[:5]}"
        )
    return {
        "required_local_root": str(root) if root is not None else "",
        "unique_image_files": len(paths),
        "all_declared_images_exist": True,
    }


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    l89.L89FrameCacheDataset,
    l89.L89FrameCacheDataset,
    dict[str, Any],
]:
    train_csv = assert_development_path(
        Path(args.train_csv), purpose="train CSV"
    )
    dev_csv = assert_development_path(
        Path(args.dev_csv), purpose="development CSV"
    )
    train_cache_path = assert_development_path(
        Path(args.train_cache), purpose="train cache"
    )
    dev_cache_path = assert_development_path(
        Path(args.dev_cache), purpose="development cache"
    )
    weights_path = assert_development_path(
        Path(args.weights), purpose="Panopticon weights"
    )
    for path in (
        train_csv,
        dev_csv,
        train_cache_path,
        dev_cache_path,
        weights_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    train, dev, cache_audit = l89.load_cache_pair(
        train_cache_path, dev_cache_path
    )
    if int(train["features"].shape[1]) != 6:
        raise ValueError("This runner requires the audited six-role L89 cache.")
    if not train["unique_mask"][:, int(train["t0_index"])].all():
        raise ValueError("Every train row must retain unique t0 evidence.")
    if not dev["unique_mask"][:, int(dev["t0_index"])].all():
        raise ValueError("Every development row must retain unique t0 evidence.")
    weights_sha = sha256_file(weights_path)
    if weights_sha != str(train["weights_sha256"]):
        raise ValueError("Panopticon PTH SHA differs from the strict cache contract.")
    for payload, csv_path, split in (
        (train, train_csv, "train"),
        (dev, dev_csv, "val"),
    ):
        if str(Path(str(payload["csv_path"])).expanduser().resolve()) != str(
            csv_path
        ):
            raise ValueError(f"{split} cache CSV path differs from requested CSV.")
        if sha256_file(csv_path) != str(payload["csv_sha256"]):
            raise ValueError(f"{split} CSV SHA differs from its cache contract.")

    train_frame = pd.read_csv(train_csv, low_memory=False)
    dev_frame = pd.read_csv(dev_csv, low_memory=False)
    required_root = (
        Path(args.required_local_root)
        if str(args.required_local_root).strip()
        else None
    )
    path_columns = tuple(
        str(value) for value in train["input_contract"]["path_columns"]
    )
    local_audit = {
        "train": _validate_all_local_paths(
            train_frame, path_columns, required_root=required_root
        ),
        "dev": _validate_all_local_paths(
            dev_frame, path_columns, required_root=required_root
        ),
    }

    def build_dataset(
        csv_path: Path,
        frame: pd.DataFrame,
        payload: Mapping[str, Any],
    ) -> l89.L89FrameCacheDataset:
        contract = payload["input_contract"]
        return l89.L89FrameCacheDataset(
            csv_path,
            frame,
            path_columns=tuple(str(value) for value in contract["path_columns"]),
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

    train_dataset = build_dataset(train_csv, train_frame, train)
    dev_dataset = build_dataset(dev_csv, dev_frame, dev)
    audit = {
        **cache_audit,
        "train_csv": str(train_csv),
        "dev_csv": str(dev_csv),
        "weights_path": str(weights_path),
        "weights_sha256": weights_sha,
        "local_inputs": local_audit,
        "test_or_sealed_or_holdout_read": False,
    }
    return train, dev, train_dataset, dev_dataset, audit


def _loader(
    dataset: l89.L89FrameCacheDataset,
    sampler: Sampler[int],
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "sampler": sampler,
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def _metadata(
    payload: Mapping[str, Any],
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    unique = payload["unique_mask"][indices].bool().to(device)
    delta = payload["delta_days"][indices].float().to(device)
    quality_source = payload.get("valid_fraction")
    if quality_source is None:
        quality_source = payload["valid_mask"].float()
    quality = quality_source[indices].float().to(device)
    role = payload["role_index"].long().to(device)
    return unique, delta, quality, role


def _assert_online_cache_match(
    payload: Mapping[str, Any],
    indices: torch.Tensor,
    image_valid: torch.Tensor,
    valid_fraction: torch.Tensor,
) -> None:
    expected_valid = payload.get("image_valid_mask", payload["valid_mask"])[
        indices
    ].bool()
    if not torch.equal(image_valid.bool(), expected_valid):
        raise ValueError("Online image-valid mask differs from the frozen cache.")
    expected_fraction = payload.get("valid_fraction")
    if expected_fraction is not None and not torch.allclose(
        valid_fraction.float(),
        expected_fraction[indices].float(),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise ValueError("Online valid fractions differ from the frozen cache.")


def _trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_state(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    parameters = dict(model.named_parameters())
    expected = {
        name for name, parameter in parameters.items() if parameter.requires_grad
    }
    if set(state) != expected:
        raise ValueError("Trainable checkpoint keys differ from model parameters.")
    with torch.no_grad():
        for name in sorted(expected):
            parameters[name].copy_(
                state[name].to(
                    device=parameters[name].device,
                    dtype=parameters[name].dtype,
                )
            )


def optimizer_groups(
    model: MatchedLastBlockPair,
    *,
    block_lr: float,
    head_lr: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    block_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".final_block." in name or ".final_norm." in name:
            block_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    if not block_parameters or not head_parameters:
        raise RuntimeError("Optimizer groups are incomplete.")
    groups = [
        {
            "params": block_parameters,
            "lr": float(block_lr),
            "weight_decay": float(weight_decay),
            "name": "block11_and_final_norm",
        },
        {
            "params": head_parameters,
            "lr": float(head_lr),
            "weight_decay": float(weight_decay),
            "name": "t0_and_tempo_heads",
        },
    ]
    counts = {
        "block11_and_final_norm": int(
            sum(parameter.numel() for parameter in block_parameters)
        ),
        "t0_and_tempo_heads": int(
            sum(parameter.numel() for parameter in head_parameters)
        ),
        "shared_frozen_trunk": int(
            sum(parameter.numel() for parameter in model.trunk.parameters())
        ),
    }
    return groups, counts


def evaluate(
    model: MatchedLastBlockPair,
    loader: DataLoader,
    payload: Mapping[str, Any],
    channel_ids: torch.Tensor,
    *,
    device: torch.device,
    amp_dtype: str,
    encoder_microbatch: int,
    runtime_guard: Optional[RuntimeGuard] = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], np.ndarray]:
    model.eval()
    outputs: dict[str, list[torch.Tensor]] = {
        arm: [] for arm in ARM_NAMES
    }
    labels_out: list[torch.Tensor] = []
    indices_out: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            indices, images, image_valid, fractions, _status = batch
            indices = indices.long()
            _assert_online_cache_match(
                payload, indices, image_valid, fractions
            )
            unique, delta, quality, role = _metadata(
                payload, indices, device=device
            )
            logits = model.forward_images(
                images,
                channel_ids,
                unique,
                delta,
                quality,
                role,
                encoder_microbatch=int(encoder_microbatch),
                device=device,
                amp_dtype=amp_dtype,
            )
            for arm in ARM_NAMES:
                outputs[arm].append(logits[arm].float().cpu())
            labels_out.append(payload["labels"][indices].long().cpu())
            indices_out.append(indices.cpu())
    indices_array = torch.cat(indices_out).numpy().astype(np.int64)
    if not np.array_equal(indices_array, np.arange(len(payload["labels"]))):
        raise RuntimeError("Development evaluation did not preserve CSV order.")
    labels = torch.cat(labels_out).numpy().astype(np.int64)
    event_ids = [str(value) for value in payload["event_ids"]]
    probabilities: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        probability = torch.sigmoid(torch.cat(outputs[arm])).numpy()
        probabilities[arm] = probability.astype(np.float64)
        metrics[arm] = tempo.metric_bundle(
            labels, probabilities[arm], event_ids
        )
    if runtime_guard is not None:
        runtime_guard.check("development_evaluation")
    return metrics, probabilities, labels


def train_one_epoch(
    model: MatchedLastBlockPair,
    loader: DataLoader,
    payload: Mapping[str, Any],
    channel_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    amp_dtype: str,
    encoder_microbatch: int,
    grad_clip: float,
    runtime_guard: Optional[RuntimeGuard] = None,
) -> dict[str, float]:
    model.train()
    loss_sum = {arm: 0.0 for arm in ARM_NAMES}
    rows_seen = 0
    maximum_gradient = 0.0
    for batch in loader:
        indices, images, image_valid, fractions, _status = batch
        indices = indices.long()
        _assert_online_cache_match(payload, indices, image_valid, fractions)
        unique, delta, quality, role = _metadata(
            payload, indices, device=device
        )
        labels = payload["labels"][indices].float().to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_images(
            images,
            channel_ids,
            unique,
            delta,
            quality,
            role,
            encoder_microbatch=int(encoder_microbatch),
            device=device,
            amp_dtype=amp_dtype,
        )
        losses = {
            arm: F.binary_cross_entropy_with_logits(logits[arm], labels)
            for arm in ARM_NAMES
        }
        total_loss = 0.5 * (losses["current_only"] + losses["temporal_d1"])
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite supervised loss; hard stop.")
        total_loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            max_norm=float(grad_clip),
        )
        if not torch.isfinite(torch.as_tensor(gradient)):
            raise FloatingPointError("Non-finite gradient norm; hard stop.")
        maximum_gradient = max(maximum_gradient, float(gradient))
        for arm in ARM_NAMES:
            norm_parameters = list(getattr(model, arm).final_norm.parameters())
            if not any(parameter.grad is not None for parameter in norm_parameters):
                raise RuntimeError(f"{arm} final norm lost its gradient.")
        optimizer.step()
        batch_rows = int(labels.numel())
        rows_seen += batch_rows
        for arm in ARM_NAMES:
            loss_sum[arm] += float(losses[arm].detach()) * batch_rows
        if runtime_guard is not None:
            runtime_guard.check("training_batch")
    if rows_seen != len(loader.sampler):
        raise RuntimeError("Train sampler draw count changed during the epoch.")
    return {
        "rows_seen": float(rows_seen),
        "current_only_bce": loss_sum["current_only"] / rows_seen,
        "temporal_d1_bce": loss_sum["temporal_d1"] / rows_seen,
        "maximum_preclip_gradient_norm": maximum_gradient,
    }


def _prediction_frame(
    payload: Mapping[str, Any],
    labels: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "id": [str(value) for value in payload["ids"]],
            "plume_id": [str(value) for value in payload["plume_ids"]],
            "event_id": [str(value) for value in payload["event_ids"]],
            "label": labels,
        }
    )
    for arm in ARM_NAMES:
        frame[f"probability_{arm}"] = probabilities[arm]
        frame[f"prediction_{arm}"] = (
            probabilities[arm] >= float(metrics[arm]["selected_threshold"])
        ).astype(np.int64)
    return frame


def run(args: argparse.Namespace) -> None:
    validate_training_budget(args)
    set_seed(args.seed)
    output_dir = assert_development_path(
        Path(args.output_dir), purpose="exploratory output"
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is nonempty; use a new path or --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    runtime_guard = RuntimeGuard(
        device=device,
        wall_limit_minutes=float(args.wall_limit_minutes),
        max_cuda_allocated_gib=float(args.max_cuda_allocated_gib),
    )
    train, dev, train_dataset, dev_dataset, input_audit = _load_inputs(args)
    runtime_guard.check("input_validation")

    weights_path = Path(args.weights).expanduser().resolve()
    # This inherited loader applies model.load_state_dict(..., strict=True).
    backbone = load_backbone(
        str(weights_path), device=device, debug=bool(args.debug)
    )
    set_seed(args.seed)
    model = MatchedLastBlockPair(
        backbone,
        num_roles=int(train["features"].shape[1]),
        t0_index=int(train["t0_index"]),
        head_hidden_dim=int(args.head_hidden_dim),
        temporal_dim=int(args.temporal_dim),
        dropout=float(args.dropout),
        periods_days=tuple(
            float(value)
            for value in str(args.delta_periods).split(",")
            if value.strip()
        ),
    ).to(device)
    runtime_guard.check("model_initialization")
    initial_arm_sha = model.arm_state_sha256()
    if len(set(initial_arm_sha.values())) != 1:
        raise AssertionError("Epoch-0 arm state hashes differ.")

    sampler = EventLabelSampler(
        train["labels"],
        train["event_ids"],
        draws_per_epoch=int(args.draws_per_epoch),
        seed=int(args.seed),
    )
    train_loader = _loader(
        train_dataset,
        sampler,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        device=device,
    )
    dev_loader = _loader(
        dev_dataset,
        SequentialSampler(dev_dataset),
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        device=device,
    )
    channel_ids = train_dataset.channel_ids
    if not torch.equal(channel_ids, dev_dataset.channel_ids):
        raise ValueError("Train/development channel IDs differ.")

    groups, parameter_counts = optimizer_groups(
        model,
        block_lr=float(args.block_lr),
        head_lr=float(args.head_lr),
        weight_decay=float(args.weight_decay),
    )
    optimizer = torch.optim.AdamW(groups)

    started = time.monotonic()
    epoch0_metrics, epoch0_probabilities, epoch0_labels = evaluate(
        model,
        dev_loader,
        dev,
        channel_ids,
        device=device,
        amp_dtype=args.amp_dtype,
        encoder_microbatch=int(args.encoder_microbatch),
        runtime_guard=runtime_guard,
    )
    maximum_epoch0_logit_probability_error = float(
        np.max(
            np.abs(
                epoch0_probabilities["current_only"]
                - epoch0_probabilities["temporal_d1"]
            )
        )
    )
    if maximum_epoch0_logit_probability_error != 0.0:
        raise AssertionError(
            "Epoch-0 arm predictions are not bit-identical: "
            f"max probability error={maximum_epoch0_logit_probability_error:.3e}"
        )

    best = {
        "epoch": 0,
        "selection_metric": "temporal_d1.event_balanced_ap",
        "selection_value": float(
            epoch0_metrics["temporal_d1"]["event_balanced_ap"]
        ),
        "metrics": epoch0_metrics,
        "trainable_state": _trainable_state(model),
        "probabilities": epoch0_probabilities,
        "labels": epoch0_labels,
    }
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train": None,
            "validation": epoch0_metrics,
            "sampler_order_sha256": None,
        }
    ]
    l89.atomic_json_write(
        output_dir / "epoch_000.json", history[-1]
    )
    bad_epochs = 0
    stop_reason = "maximum_epochs_reached"

    for epoch in range(1, int(args.epochs) + 1):
        sampler.set_epoch(epoch)
        train_record = train_one_epoch(
            model,
            train_loader,
            train,
            channel_ids,
            optimizer,
            device=device,
            amp_dtype=args.amp_dtype,
            encoder_microbatch=int(args.encoder_microbatch),
            grad_clip=float(args.grad_clip),
            runtime_guard=runtime_guard,
        )
        metrics, probabilities, labels = evaluate(
            model,
            dev_loader,
            dev,
            channel_ids,
            device=device,
            amp_dtype=args.amp_dtype,
            encoder_microbatch=int(args.encoder_microbatch),
            runtime_guard=runtime_guard,
        )
        selection = float(metrics["temporal_d1"]["event_balanced_ap"])
        if not math.isfinite(selection):
            raise FloatingPointError("Non-finite development AP; hard stop.")
        record = {
            "epoch": epoch,
            "train": train_record,
            "validation": metrics,
            "sampler_order_sha256": sampler.order_sha256(),
        }
        history.append(record)
        l89.atomic_json_write(
            output_dir / f"epoch_{epoch:03d}.json", record
        )
        print(
            f"[epoch {epoch}] currentAP="
            f"{metrics['current_only']['event_balanced_ap']:.6f} "
            f"temporalAP={selection:.6f} "
            f"temporalMacroF1="
            f"{metrics['temporal_d1']['event_balanced_macro_f1_selected']:.6f}",
            flush=True,
        )
        if selection > float(best["selection_value"]) + float(args.min_delta):
            best = {
                "epoch": epoch,
                "selection_metric": "temporal_d1.event_balanced_ap",
                "selection_value": selection,
                "metrics": metrics,
                "trainable_state": _trainable_state(model),
                "probabilities": probabilities,
                "labels": labels,
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            stop_reason = "first_non_improving_epoch_hard_stop"
            if bad_epochs >= int(args.patience):
                break

    _load_trainable_state(model, best["trainable_state"])
    checkpoint = {
        "format_version": 1,
        "script_version": SCRIPT_VERSION,
        "epoch": int(best["epoch"]),
        "selection_metric": str(best["selection_metric"]),
        "selection_value": float(best["selection_value"]),
        "trainable_state": best["trainable_state"],
        "initial_arm_state_sha256": initial_arm_sha,
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "test_or_sealed_or_holdout_read": False,
    }
    l89.atomic_torch_save(output_dir / "checkpoint_best_dev_ap.pt", checkpoint)
    prediction_frame = _prediction_frame(
        dev,
        best["labels"],
        best["probabilities"],
        best["metrics"],
    )
    l89.atomic_csv_write(
        output_dir / "validation_best_dev_ap_predictions.csv",
        prediction_frame,
    )
    config = {
        "script_version": SCRIPT_VERSION,
        "args": vars(args),
        "input_audit": input_audit,
        "architecture": {
            "shared_frozen_blocks": list(range(11)),
            "shared_trunk_no_grad": True,
            "trainable_block_per_arm": 11,
            "trainable_final_norm_per_arm": True,
            "arms": list(ARM_NAMES),
            "epoch0_arm_state_sha256": initial_arm_sha,
            "epoch0_probability_max_abs_error": (
                maximum_epoch0_logit_probability_error
            ),
            "temporal_module": "TEMPOGlobalResidual:d1_gated_delta",
            "parameter_counts": parameter_counts,
        },
        "sampling": sampler.audit(),
        "runtime_guard": runtime_guard.audit(),
        "selection": {
            "checkpoint": "development event-balanced AP of temporal_d1",
            "threshold": "development event-balanced macro-F1 maximizer",
            "best_epoch": int(best["epoch"]),
            "best_value": float(best["selection_value"]),
            "stop_reason": stop_reason,
        },
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
        "test_or_sealed_or_holdout_read": False,
    }
    l89.atomic_json_write(output_dir / "run_summary.json", config)
    print(
        json.dumps(
            {
                "best_epoch": int(best["epoch"]),
                "best_temporal_event_balanced_ap": float(
                    best["selection_value"]
                ),
                "stop_reason": stop_reason,
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--dev-csv", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--dev-cache", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--required-local-root",
        default="",
        help="Optional root that every declared image path must remain under.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--patience", type=int, choices=(1,), default=1)
    parser.add_argument("--draws-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--encoder-microbatch", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--block-lr", type=float, default=3.0e-6)
    parser.add_argument("--head-lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--temporal-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--delta-periods", default="1,3,7,30,90,365")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--wall-limit-minutes", type=float, default=43.0)
    parser.add_argument(
        "--max-cuda-allocated-gib", type=float, default=28.0
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(handler=run)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
