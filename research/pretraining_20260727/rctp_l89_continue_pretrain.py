#!/usr/bin/env python3
"""Minimal RCTP continue-pretraining for real L89 downstream transfer.

The earlier ``rctp_l89_screen.py`` intentionally froze Panopticon and cached five
of six clean-context CLS vectors.  That is valid for a mechanism probe, but it
cannot produce a transferable encoder.  This runner closes that gap:

* P4 (response-scrambled) and P5 (correct-response) start from the same base PTH;
* every clean visit and every counterfactual is encoded online with the current
  backbone, so no stale encoder features enter the loss;
* only the final one or two Panopticon blocks and the objective probe are updated;
* the selected full backbone is exported under a ``backbone`` key, directly
  consumable by the existing L89 ``cache`` command;
* by default, cache tensors are used only for immutable row identity, temporal
  masks, and time deltas. An opt-in clean anchor may use audited base-PTH CLS as
  a detached target, never as temporal context or probe input.

This is train/dev only.  Paths with a test/sealed token are rejected.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

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

from research.pretraining_20260727 import l89_ragged_cls_experiment as cache_runner
from research.pretraining_20260727 import rctp_l89_screen as screen
from thirdparty.dinov2.data.datasets.s2_csv import S2CsvDataset
from Upgraded_dataset.dino_classifier_head_l89_temporal_satmae import load_backbone


SCRIPT_VERSION = "rctp-l89-continue-pretrain-v2-clean-anchor"
ARM_P4 = "p4_response_scrambled"
ARM_P5 = "p5_correct_response"
ARM_NAMES = (ARM_P4, ARM_P5)

# Renderer output order in rctp_l89_screen is:
# correct methane, achromatic, wavelength-shuffled.
# Reordering keeps three images, FLOPs, masks, and losses identical while making
# the requested response the first/positive example expected by assemble_probe_batch.
OBJECTIVE_VARIANT_ORDER = {
    ARM_P5: (0, 1, 2),
    ARM_P4: (2, 1, 0),
}
OBJECTIVE_VARIANT_NAMES = {
    ARM_P5: ("correct_response_positive", "achromatic", "scrambled_response"),
    ARM_P4: ("scrambled_response_positive", "achromatic", "correct_response"),
}


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
            raise RuntimeError(f"CUDA device {device} requested but CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    return device


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def objective_response(
    renderer: screen.RendererConfig, arm: str
) -> tuple[float, ...]:
    if arm == ARM_P5:
        return tuple(float(value) for value in renderer.response)
    if arm == ARM_P4:
        return tuple(
            float(renderer.response[index])
            for index in renderer.wavelength_shuffle
        )
    raise ValueError(f"Unknown RCTP arm: {arm!r}")


def reorder_objective_variants(
    variants: torch.Tensor, arm: str
) -> torch.Tensor:
    if variants.ndim != 5 or variants.shape[1] != 3:
        raise ValueError("Renderer variants must have shape (B,3,C,H,W)")
    try:
        order = OBJECTIVE_VARIANT_ORDER[arm]
    except KeyError as error:
        raise ValueError(f"Unknown RCTP arm: {arm!r}") from error
    return variants[:, list(order)]


def configure_last_blocks(
    backbone: nn.Module,
    train_last_blocks: int,
) -> dict[str, Any]:
    """Freeze the trunk and expose exactly the final N transformer blocks."""
    blocks = getattr(backbone, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not len(blocks):
        raise ValueError("Backbone must expose a non-empty nn.ModuleList at .blocks")
    if not 1 <= int(train_last_blocks) <= len(blocks):
        raise ValueError(
            f"train_last_blocks={train_last_blocks} outside 1..{len(blocks)}"
        )
    backbone.requires_grad_(False)
    backbone.eval()
    first_trainable = len(blocks) - int(train_last_blocks)
    for index, block in enumerate(blocks):
        block.requires_grad_(index >= first_trainable)
        block.eval()
    trainable_names = [
        name for name, parameter in backbone.named_parameters() if parameter.requires_grad
    ]
    frozen_names = [
        name for name, parameter in backbone.named_parameters() if not parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("No backbone parameter was made trainable")
    illegal = [
        name
        for name in trainable_names
        if not any(name.startswith(f"blocks.{index}.") for index in range(first_trainable, len(blocks)))
    ]
    if illegal:
        raise RuntimeError(f"Parameters outside final blocks became trainable: {illegal[:10]}")
    return {
        "total_blocks": len(blocks),
        "train_last_blocks": int(train_last_blocks),
        "trainable_block_indices": list(range(first_trainable, len(blocks))),
        "trainable_parameter_names": trainable_names,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad)
        ),
        "frozen_parameters": int(
            sum(parameter.numel() for parameter in backbone.parameters() if not parameter.requires_grad)
        ),
        "frozen_parameter_names_count": len(frozen_names),
    }


def set_backbone_phase(
    backbone: nn.Module,
    trainable_block_indices: Sequence[int],
    *,
    training: bool,
) -> None:
    """Keep the frozen trunk deterministic while toggling only trainable blocks."""
    backbone.eval()
    blocks = getattr(backbone, "blocks")
    for index in trainable_block_indices:
        blocks[int(index)].train(bool(training))


def trainable_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        if not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        value = parameter.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class OnlineSequenceDataset(Dataset):
    """Read every usable clean visit for a planned row from the local L89 stage."""

    def __init__(
        self,
        csv_path: Path,
        frame: pd.DataFrame,
        payload: Mapping[str, Any],
        plan: Sequence[screen.PlanEntry],
        *,
        renderer: screen.RendererConfig,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.frame = frame.reset_index(drop=True)
        self.payload = payload
        self.plan = list(plan)
        self.renderer = renderer
        contract = payload["input_contract"]
        self.path_columns = tuple(payload["path_columns"])
        self.band_indices = tuple(int(value) for value in contract["band_indices"])
        self.image_size = int(contract["image_size"])
        self.zero_invalid_pixels = bool(contract["zero_invalid_pixels"])
        self.unique_mask = payload["unique_mask"].bool()
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
            raise ValueError("Reader channels differ from cached input contract")
        renderer.validate(len(self.band_indices))

    def __len__(self) -> int:
        return len(self.plan)

    def _read_visit(
        self, row: pd.Series, column: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        path = row[column]
        if not isinstance(path, str) or not path.strip():
            raise RuntimeError(f"Usable visit has no path in column {column!r}")
        raw = self.reader._read_image_raw(path.strip())[list(self.band_indices)]
        native_valid = torch.isfinite(raw) & raw.ne(0)
        normalized = (
            torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0) - self.mean
        ) / self.std
        if self.zero_invalid_pixels:
            normalized = torch.where(
                native_valid, normalized, torch.zeros_like(normalized)
            )
        normalized = F.interpolate(
            normalized.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        valid = F.interpolate(
            native_valid.float().unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0).bool()
        return normalized, valid

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.plan[index]
        row = self.frame.iloc[entry.row_index]
        timepoints = len(self.path_columns)
        channels = len(self.band_indices)
        images = torch.zeros(
            timepoints,
            channels,
            self.image_size,
            self.image_size,
            dtype=torch.float32,
        )
        selected_valid_pixels: Optional[torch.Tensor] = None
        row_unique = self.unique_mask[entry.row_index]
        for visit_index, column in enumerate(self.path_columns):
            if not bool(row_unique[visit_index]):
                continue
            image, valid = self._read_visit(row, column)
            images[visit_index] = image
            if visit_index == entry.visit_index:
                selected_valid_pixels = valid
        if selected_valid_pixels is None:
            raise RuntimeError(
                f"Planned visit {entry.visit_index} for row {entry.row_index} "
                "was not loaded as unique evidence"
            )
        plume = screen.make_soft_plume(
            self.image_size,
            self.image_size,
            seed=entry.renderer_seed,
            config=self.renderer,
        )
        return {
            "row_index": torch.tensor(entry.row_index, dtype=torch.long),
            "visit_index": torch.tensor(entry.visit_index, dtype=torch.long),
            "peak_drop": torch.tensor(entry.peak_drop, dtype=torch.float32),
            "clean_images": images,
            "selected_valid_pixels": selected_valid_pixels,
            "plume_field": plume,
        }


def encode_online_clean_and_variants(
    backbone: nn.Module,
    clean_images: torch.Tensor,
    variants: torch.Tensor,
    channel_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One same-checkpoint forward for all clean context and three variants."""
    if clean_images.ndim != 5 or variants.ndim != 5:
        raise ValueError("clean_images/variants must both be five-dimensional")
    batch, timepoints, channels, height, width = clean_images.shape
    if variants.shape != (batch, 3, channels, height, width):
        raise ValueError("variants shape is incompatible with clean_images")
    if channel_ids.shape != (channels,):
        raise ValueError("channel_ids must have shape (C,)")
    flat_clean = clean_images.reshape(
        batch * timepoints, channels, height, width
    )
    flat_variants = variants.reshape(batch * 3, channels, height, width)
    images = torch.cat((flat_clean, flat_variants), dim=0)
    identifiers = channel_ids.view(1, -1).expand(len(images), -1).clone()
    output = backbone.forward_features({"imgs": images, "chn_ids": identifiers})
    cls = output["x_norm_clstoken"].float()
    expected = batch * (timepoints + 3)
    if cls.shape[0] != expected:
        raise RuntimeError(
            f"Backbone returned {cls.shape[0]} CLS rows; expected {expected}"
        )
    clean_cls = cls[: batch * timepoints].reshape(batch, timepoints, -1)
    variant_cls = cls[batch * timepoints :].reshape(batch, 3, -1)
    return clean_cls, variant_cls


def validity_masked_clean_anchor_loss(
    online_clean_cls: torch.Tensor,
    base_clean_target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    metric: str,
) -> torch.Tensor:
    """Anchor online clean CLS only; the audited base target is always detached."""
    if (
        online_clean_cls.ndim != 3
        or base_clean_target.shape != online_clean_cls.shape
    ):
        raise ValueError(
            "online clean CLS and base clean target must share shape (B,T,D)"
        )
    if valid_mask.shape != online_clean_cls.shape[:2]:
        raise ValueError("clean anchor valid_mask must have shape (B,T)")
    if metric not in {"cosine", "l2"}:
        raise ValueError("clean anchor metric must be 'cosine' or 'l2'")
    valid = valid_mask.bool()
    if not valid.any():
        raise ValueError("clean anchor requires at least one valid clean visit")
    online = online_clean_cls[valid].float()
    # Explicit detach is required even in CPU tests that deliberately give the
    # target requires_grad=True.
    target = base_clean_target.detach()[valid].to(
        device=online.device, dtype=torch.float32
    )
    if not torch.isfinite(online).all() or not torch.isfinite(target).all():
        raise ValueError("clean anchor inputs contain non-finite values")
    if metric == "cosine":
        return (1.0 - F.cosine_similarity(online, target, dim=-1, eps=1e-6)).mean()
    return F.mse_loss(online, target)


def combine_pretext_and_clean_anchor(
    pretext_loss: torch.Tensor,
    online_clean_cls: torch.Tensor,
    base_clean_target: Optional[torch.Tensor],
    valid_mask: torch.Tensor,
    *,
    weight: float,
    metric: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve the exact legacy loss path when weight is zero."""
    if not math.isfinite(float(weight)) or float(weight) < 0:
        raise ValueError("clean anchor weight must be finite and non-negative")
    if float(weight) == 0.0:
        # Returning the identical tensor (not pretext + 0 * anchor) guarantees
        # the first-round default has the same graph and floating-point path.
        return pretext_loss, pretext_loss.detach().new_zeros(())
    if base_clean_target is None:
        raise ValueError("Positive clean anchor weight requires base clean targets")
    anchor_loss = validity_masked_clean_anchor_loss(
        online_clean_cls,
        base_clean_target,
        valid_mask,
        metric=metric,
    )
    return pretext_loss + float(weight) * anchor_loss, anchor_loss


def objective_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    group_ids: Sequence[int],
    variant_indices: Sequence[int],
    visit_targets: Sequence[int],
    visit_predictions: Sequence[int],
    strength_targets: Sequence[float],
    strength_predictions: Sequence[float],
) -> dict[str, float | int]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    groups_array = np.asarray(group_ids, dtype=np.int64)
    variants_array = np.asarray(variant_indices, dtype=np.int64)
    if set(np.unique(labels_array)) != {0, 1}:
        raise ValueError("Objective metrics require both binary classes")
    threshold, selected_f1 = screen.best_f1_threshold(
        labels_array, probabilities_array
    )
    paired_wins: list[float] = []
    paired_margins: list[float] = []
    for group in np.unique(groups_array):
        mask = groups_array == group
        positive = probabilities_array[mask & (variants_array == 0)]
        negatives = probabilities_array[mask & (variants_array != 0)]
        if len(positive) != 1 or len(negatives) != 2:
            raise ValueError("Each matched group must have one positive and two nuisances")
        paired_wins.append(float(positive[0] > negatives.max()))
        paired_margins.append(float(positive[0] - negatives.max()))
    visit_target = np.asarray(visit_targets, dtype=np.int64)
    visit_prediction = np.asarray(visit_predictions, dtype=np.int64)
    strength_target = np.asarray(strength_targets, dtype=np.float64)
    strength_prediction = np.asarray(strength_predictions, dtype=np.float64)
    positive_mask = variants_array == 0
    return {
        "average_precision": float(
            average_precision_score(labels_array, probabilities_array)
        ),
        "roc_auc": float(roc_auc_score(labels_array, probabilities_array)),
        "selected_f1": float(selected_f1),
        "selected_threshold": float(threshold),
        "fixed_0p5_f1": float(
            f1_score(
                labels_array,
                probabilities_array >= 0.5,
                zero_division=0,
            )
        ),
        "paired_win_rate": float(np.mean(paired_wins)),
        "paired_probability_margin": float(np.mean(paired_margins)),
        "visit_accuracy": float(np.mean(visit_target == visit_prediction)),
        "positive_strength_log_mae": float(
            np.mean(
                np.abs(
                    strength_prediction[positive_mask]
                    - strength_target[positive_mask]
                )
            )
        ),
        "examples": int(len(labels_array)),
        "matched_groups": int(len(np.unique(groups_array))),
    }


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    training: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(training),
        "num_workers": int(workers),
        "pin_memory": bool(pin_memory),
        "drop_last": False,
        "generator": torch.Generator().manual_seed(int(seed)),
    }
    if workers > 0:
        arguments["prefetch_factor"] = int(prefetch_factor)
        arguments["persistent_workers"] = False
    return DataLoader(**arguments)


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(round(total_steps * float(warmup_fraction))))

    def scale(step: int) -> float:
        completed = step + 1
        if completed <= warmup_steps:
            return completed / warmup_steps
        denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, (completed - warmup_steps) / denominator)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def run_epoch(
    *,
    backbone: nn.Module,
    trainable_block_indices: Sequence[int],
    probe: screen.RCTPTemporalProbe,
    loader: DataLoader,
    payload: Mapping[str, Any],
    base_clean_anchor_features: Optional[torch.Tensor],
    clean_anchor_weight: float,
    clean_anchor_metric: str,
    renderer: screen.RendererConfig,
    response_metadata: torch.Tensor,
    arm: str,
    device: torch.device,
    amp_dtype: str,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    global_step: int,
    max_train_steps: int,
    loss_weights: tuple[float, float, float],
    grad_clip: float,
    log_interval: int,
) -> tuple[dict[str, Any], int, pd.DataFrame]:
    training = optimizer is not None
    set_backbone_phase(
        backbone, trainable_block_indices, training=training
    )
    probe.train(training)
    labels: list[int] = []
    probabilities: list[float] = []
    group_ids: list[int] = []
    row_ids: list[int] = []
    variant_indices: list[int] = []
    visit_targets: list[int] = []
    visit_predictions: list[int] = []
    strength_targets: list[float] = []
    strength_predictions: list[float] = []
    losses: list[dict[str, float]] = []
    started = time.monotonic()
    contract = payload["input_contract"]
    mean = torch.tensor(contract["normalization_mean"], dtype=torch.float32)[
        contract["band_indices"]
    ].to(device)
    std = torch.tensor(contract["normalization_std"], dtype=torch.float32)[
        contract["band_indices"]
    ].to(device)
    channel_ids = torch.tensor(
        contract["channel_ids"], dtype=torch.float32, device=device
    )
    role_index = payload["role_index"].long().to(device)
    metadata = response_metadata.to(device)

    for batch_index, source in enumerate(loader, 1):
        if training and max_train_steps > 0 and global_step >= max_train_steps:
            break
        row_index_cpu = source["row_index"].long()
        row_index = row_index_cpu.to(device)
        visit = source["visit_index"].long().to(device)
        clean_images = source["clean_images"].float().to(
            device, non_blocking=True
        )
        selected_valid_pixels = source["selected_valid_pixels"].bool().to(
            device, non_blocking=True
        )
        plume = source["plume_field"].float().to(device, non_blocking=True)
        peak_drop = source["peak_drop"].float().to(device, non_blocking=True)
        batch_rows = torch.arange(len(row_index), device=device)
        selected_clean = clean_images[batch_rows, visit]
        rendered, diagnostics = screen.render_counterfactual_variants(
            selected_clean,
            selected_valid_pixels,
            plume,
            peak_drop,
            normalization_mean=mean,
            normalization_std=std,
            config=renderer,
        )
        if float(diagnostics["max_energy_relative_error"]) > 5e-5:
            raise RuntimeError("Nuisance energy matching exceeded tolerance")
        variants = reorder_objective_variants(rendered, arm)
        if not torch.isfinite(variants).all():
            raise RuntimeError("Counterfactual renderer emitted non-finite values")

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast_context(device, amp_dtype):
            clean_cls, variant_cls = encode_online_clean_and_variants(
                backbone, clean_images, variants, channel_ids
            )
            selected_clean_cls = clean_cls[batch_rows, visit]
            unique_valid = payload["unique_mask"][row_index_cpu].bool().to(device)
            assembled = screen.assemble_probe_batch(
                clean_cls,
                unique_valid,
                payload["delta_days"][row_index_cpu].float().to(device),
                visit,
                selected_clean_cls,
                variant_cls,
            )
            repeated_metadata = metadata.view(1, -1).expand(
                len(assembled["type_target"]), -1
            )
            output = probe(
                assembled["clean_features"],
                assembled["delta_features"],
                assembled["valid_mask"],
                role_index,
                assembled["delta_days"],
                repeated_metadata,
            )
            pretext_loss, loss_values = screen._loss(
                output,
                assembled,
                peak_drop,
                type_weight=loss_weights[0],
                visit_weight=loss_weights[1],
                strength_weight=loss_weights[2],
            )
            anchor_target = None
            if clean_anchor_weight > 0:
                if base_clean_anchor_features is None:
                    raise RuntimeError(
                        "Enabled clean anchor has no audited base target tensor"
                    )
                anchor_target = base_clean_anchor_features[row_index_cpu].to(
                    device=device,
                    dtype=clean_cls.dtype,
                    non_blocking=True,
                )
            total_loss, anchor_loss = combine_pretext_and_clean_anchor(
                pretext_loss,
                clean_cls,
                anchor_target,
                unique_valid,
                weight=clean_anchor_weight,
                metric=clean_anchor_metric,
            )
            loss_values["pretext_total_loss"] = float(pretext_loss.detach())
            loss_values["clean_anchor_loss"] = float(anchor_loss.detach())
            loss_values["clean_anchor_weighted_loss"] = float(
                clean_anchor_weight * anchor_loss.detach()
            )
            loss_values["total_loss"] = float(total_loss.detach())
        if not torch.isfinite(total_loss):
            raise RuntimeError("RCTP loss became non-finite")
        if training:
            total_loss.backward()
            trainable_parameters = [
                parameter
                for parameter in list(backbone.parameters()) + list(probe.parameters())
                if parameter.requires_grad
            ]
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable_parameters, float(grad_clip))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
        losses.append(loss_values)

        probability = torch.sigmoid(output["type_logit"]).detach().float().cpu()
        visit_prediction = output["visit_logits"].argmax(dim=1).detach().cpu()
        target = assembled["type_target"].detach().long().cpu()
        repeated_rows = row_index_cpu.repeat_interleave(3).tolist()
        labels.extend(target.tolist())
        probabilities.extend(probability.tolist())
        row_ids.extend(repeated_rows)
        group_ids.extend(repeated_rows)
        variant_indices.extend(torch.arange(3).repeat(len(row_index_cpu)).tolist())
        visit_targets.extend(assembled["visit_target"].detach().cpu().tolist())
        visit_predictions.extend(visit_prediction.tolist())
        target_strength = (
            peak_drop[:, None].expand(-1, 3).reshape(-1).log().detach().cpu()
        )
        strength_targets.extend(target_strength.tolist())
        strength_predictions.extend(
            output["strength_log"].detach().float().cpu().tolist()
        )
        if batch_index % max(1, int(log_interval)) == 0:
            print(
                f"[{arm} {'train' if training else 'dev'}] "
                f"batch={batch_index}/{len(loader)} step={global_step} "
                f"loss={loss_values['total_loss']:.4f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    if not labels:
        raise RuntimeError("Epoch produced no matched examples")
    metrics = objective_metrics(
        labels,
        probabilities,
        group_ids,
        variant_indices,
        visit_targets,
        visit_predictions,
        strength_targets,
        strength_predictions,
    )
    for loss_name in (
        "type_loss",
        "visit_loss",
        "strength_loss",
        "pretext_total_loss",
        "clean_anchor_loss",
        "clean_anchor_weighted_loss",
        "total_loss",
    ):
        metrics[loss_name] = float(np.mean([item[loss_name] for item in losses]))
    metrics["clean_anchor_weight"] = float(clean_anchor_weight)
    metrics["clean_anchor_metric"] = clean_anchor_metric
    metrics["elapsed_seconds"] = float(time.monotonic() - started)
    metrics["global_optimizer_steps"] = int(global_step)
    metrics["batches_emitted"] = int(len(losses))
    frame = pd.DataFrame(
        {
            "cache_row_index": row_ids,
            "group_id": group_ids,
            "objective_variant_index": variant_indices,
            "objective_variant": [
                OBJECTIVE_VARIANT_NAMES[arm][index] for index in variant_indices
            ],
            "target_positive": labels,
            "probability_positive": probabilities,
            "injected_visit_index": visit_targets,
            "predicted_visit_index": visit_predictions,
            "target_log_strength": strength_targets,
            "predicted_log_strength": strength_predictions,
            "arm": arm,
        }
    )
    return metrics, global_step, frame


def optimizer_parameter_groups(
    backbone: nn.Module,
    probe: nn.Module,
    *,
    backbone_lr: float,
    probe_lr: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    backbone_parameters = [
        parameter for parameter in backbone.parameters() if parameter.requires_grad
    ]
    probe_parameters = [
        parameter for parameter in probe.parameters() if parameter.requires_grad
    ]
    if not backbone_parameters or not probe_parameters:
        raise ValueError("Both backbone final blocks and probe must be trainable")
    groups = [
        {
            "params": backbone_parameters,
            "lr": float(backbone_lr),
            "weight_decay": float(weight_decay),
            "group_name": "backbone_final_blocks",
        },
        {
            "params": probe_parameters,
            "lr": float(probe_lr),
            "weight_decay": float(weight_decay),
            "group_name": "objective_probe",
        },
    ]
    counts = {
        "backbone_trainable": int(
            sum(parameter.numel() for parameter in backbone_parameters)
        ),
        "probe_trainable": int(
            sum(parameter.numel() for parameter in probe_parameters)
        ),
    }
    return groups, counts


def save_transfer_checkpoint(
    path: Path,
    *,
    backbone: nn.Module,
    probe_state: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> None:
    payload = {
        "backbone": {
            name: value.detach().cpu()
            for name, value in backbone.state_dict().items()
        },
        "rctp_probe_state": {
            name: value.detach().cpu() for name, value in probe_state.items()
        },
        "rctp_metadata": dict(metadata),
    }
    cache_runner.atomic_torch_save(path, payload)


def run(args: argparse.Namespace) -> None:
    if args.arm not in ARM_NAMES:
        raise ValueError(f"--arm must be one of {ARM_NAMES}")
    train_csv = Path(args.train_csv).expanduser().resolve()
    dev_csv = Path(args.dev_csv).expanduser().resolve()
    train_cache = Path(args.train_cache).expanduser().resolve()
    dev_cache = Path(args.dev_cache).expanduser().resolve()
    base_weights = Path(args.base_weights).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path, purpose in (
        (train_csv, "train CSV"),
        (dev_csv, "dev CSV"),
        (train_cache, "train cache metadata"),
        (dev_cache, "dev cache metadata"),
        (output_dir, "continue-pretrain output"),
    ):
        cache_runner.assert_not_sealed_path(path, purpose=purpose)
    if not base_weights.is_file():
        raise FileNotFoundError(base_weights)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {output_dir}; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_runner.atomic_json_write(
        output_dir / "run_status.json",
        {
            "status": "running",
            "script_version": SCRIPT_VERSION,
            "arm": args.arm,
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "test_evaluations": 0,
        },
    )

    set_seed(args.seed)
    device = resolve_device(args.device)
    train_payload, dev_payload, cache_audit = cache_runner.load_cache_pair(
        train_cache, dev_cache
    )
    train_frame = screen.validate_frame_cache_alignment(train_csv, train_payload)
    dev_frame = screen.validate_frame_cache_alignment(dev_csv, dev_payload)
    anchor_enabled = float(args.clean_anchor_weight) > 0.0
    clean_anchor_contract = {
        "enabled": anchor_enabled,
        "weight": float(args.clean_anchor_weight),
        "metric": args.clean_anchor_metric,
        "target_encoder": "audited base Panopticon PTH frozen-CLS cache",
        "target_encoder_weights_sha256": train_payload["weights_sha256"],
        "train_target_feature_sha256": train_payload["feature_sha256"],
        "dev_target_feature_sha256": dev_payload["feature_sha256"],
        "target_tensor_detached": True,
        "validity_mask": "cache.unique_mask",
        "target_used_as_temporal_context": False,
        "target_used_as_probe_input": False,
        "anchor_applies_to": "online current-backbone clean CLS only",
        "variant_anchor": False,
        "zero_weight_behavior": (
            "target tensor discarded; identical pretext loss tensor/graph returned"
        ),
    }
    # Remove cached representation tensors from the payload used as temporal
    # context. If explicitly enabled, retain them in a separately named,
    # CPU-only target handle; otherwise discard them entirely.
    train_anchor_features = (
        train_payload.pop("features") if anchor_enabled else None
    )
    dev_anchor_features = (
        dev_payload.pop("features") if anchor_enabled else None
    )
    if not anchor_enabled:
        train_payload.pop("features")
        dev_payload.pop("features")
    base_sha = cache_runner.sha256_file(base_weights)
    if train_payload["weights_sha256"] != base_sha:
        raise ValueError(
            "Metadata/optional-anchor caches were not created by --base-weights; "
            "this is an uncontrolled input contract."
        )
    response = tuple(
        float(value.strip())
        for value in args.ch4_response.split(",")
        if value.strip()
    )
    permutation = tuple(
        int(value.strip())
        for value in args.wavelength_shuffle.split(",")
        if value.strip()
    )
    renderer = screen.RendererConfig(
        response=response,
        wavelength_shuffle=permutation,
        min_peak_drop=float(args.min_peak_drop),
        max_peak_drop=float(args.max_peak_drop),
    )
    renderer.validate(len(train_payload["input_contract"]["band_indices"]))
    arm_response = objective_response(renderer, args.arm)
    response_meta = screen.response_metadata(
        arm_response,
        train_payload["input_contract"]["channel_ids"],
        timepoints=int(train_payload["unique_mask"].shape[1]),
    )

    backbone = load_backbone(
        str(base_weights), device=device, debug=args.debug
    ).to(device)
    trainability = configure_last_blocks(backbone, args.train_last_blocks)
    trainable_initial_sha = trainable_state_sha256(backbone)
    probe_config = {
        "feature_dim": int(backbone.embed_dim),
        "response_dim": int(response_meta.numel()),
        "num_roles": int(train_payload["unique_mask"].shape[1]),
        "model_dim": int(args.model_dim),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "periods_days": tuple(
            float(value.strip())
            for value in args.delta_periods.split(",")
            if value.strip()
        ),
    }
    # Seed is reset immediately before the objective head so P4/P5 initialization
    # is byte-identical despite independently loading the same base checkpoint.
    set_seed(args.seed)
    probe = screen.RCTPTemporalProbe(**probe_config).to(device)
    probe_initial_state = {
        name: value.detach().cpu().clone()
        for name, value in probe.state_dict().items()
    }
    probe_initial_sha = cache_runner.state_dict_sha256(probe_initial_state)
    parameter_groups, parameter_counts = optimizer_parameter_groups(
        backbone,
        probe,
        backbone_lr=args.backbone_lr,
        probe_lr=args.probe_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(parameter_groups)

    dev_plan = screen.deterministic_reference_plan(
        dev_payload,
        seed=args.seed + 100_003,
        epoch=0,
        max_rows=args.max_dev_rows,
        reference_label=args.reference_label,
        renderer=renderer,
    )
    dev_dataset = OnlineSequenceDataset(
        dev_csv, dev_frame, dev_payload, dev_plan, renderer=renderer
    )
    dev_loader = make_loader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        training=False,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    planned_steps_per_epoch = math.ceil(
        min(
            args.max_train_rows if args.max_train_rows > 0 else len(train_frame),
            len(train_frame),
        )
        / args.batch_size
    )
    total_steps = int(args.epochs) * planned_steps_per_epoch
    if args.max_train_steps > 0:
        total_steps = min(total_steps, int(args.max_train_steps))
    scheduler = cosine_schedule(
        optimizer, max(1, total_steps), args.warmup_fraction
    )

    matched_contract = {
        "same_base_weights_required": True,
        "same_probe_seed_and_initialization_required": True,
        "same_renderer_draws_and_three_images": True,
        "same_online_clean_context": True,
        "same_optimizer_parameter_shapes_and_steps": True,
        "only_allowed_p4_p5_difference": (
            "positive response order and matching response metadata"
        ),
        "cached_cls_used_as_temporal_context_or_probe_input": False,
        "cached_cls_used_as_detached_clean_anchor_target": anchor_enabled,
        "clean_anchor_contract_identical_between_p4_p5": True,
        "cache_fields_used": [
            "ids",
            "event_ids",
            "unique_mask",
            "delta_days",
            "role_index",
            "input_contract",
        ]
        + (["features:detached-clean-anchor-target-only"] if anchor_enabled else []),
    }
    run_config = {
        "script_version": SCRIPT_VERSION,
        "arm": args.arm,
        "base_weights": str(base_weights),
        "base_weights_sha256": base_sha,
        "trainability": trainability,
        "trainable_initial_state_sha256": trainable_initial_sha,
        "probe_config": probe_config,
        "probe_initial_state_sha256": probe_initial_sha,
        "parameter_counts": parameter_counts,
        "clean_anchor_contract": clean_anchor_contract,
        "renderer": asdict(renderer),
        "objective_response": list(arm_response),
        "objective_variant_order": list(OBJECTIVE_VARIANT_ORDER[args.arm]),
        "train_plan_rows_per_epoch": int(
            min(args.max_train_rows, len(train_frame))
            if args.max_train_rows > 0
            else len(train_frame)
        ),
        "dev_plan_rows": len(dev_plan),
        "dev_plan_sha256": screen.plan_sha256(dev_plan),
        "planned_optimizer_steps": total_steps,
        "matched_contract": matched_contract,
        "cache_audit": cache_audit,
        "args": {
            key: value
            for key, value in vars(args).items()
            if key != "handler"
        },
        "test_evaluations": 0,
    }
    cache_runner.atomic_json_write(output_dir / "run_config.json", run_config)

    history: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    best_probe_state: Optional[dict[str, torch.Tensor]] = None
    best_dev_predictions: Optional[pd.DataFrame] = None
    global_step = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_plan = screen.deterministic_reference_plan(
            train_payload,
            seed=args.seed,
            epoch=epoch - 1,
            max_rows=args.max_train_rows,
            reference_label=args.reference_label,
            renderer=renderer,
        )
        train_dataset = OnlineSequenceDataset(
            train_csv, train_frame, train_payload, train_plan, renderer=renderer
        )
        train_loader = make_loader(
            train_dataset,
            batch_size=args.batch_size,
            workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            training=True,
            seed=args.seed + epoch,
            pin_memory=device.type == "cuda",
        )
        train_metrics, global_step, _ = run_epoch(
            backbone=backbone,
            trainable_block_indices=trainability["trainable_block_indices"],
            probe=probe,
            loader=train_loader,
            payload=train_payload,
            base_clean_anchor_features=train_anchor_features,
            clean_anchor_weight=args.clean_anchor_weight,
            clean_anchor_metric=args.clean_anchor_metric,
            renderer=renderer,
            response_metadata=response_meta,
            arm=args.arm,
            device=device,
            amp_dtype=args.amp_dtype,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            max_train_steps=args.max_train_steps,
            loss_weights=(
                args.type_loss_weight,
                args.visit_loss_weight,
                args.strength_loss_weight,
            ),
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
        )
        dev_metrics, _, dev_predictions = run_epoch(
            backbone=backbone,
            trainable_block_indices=trainability["trainable_block_indices"],
            probe=probe,
            loader=dev_loader,
            payload=dev_payload,
            base_clean_anchor_features=dev_anchor_features,
            clean_anchor_weight=args.clean_anchor_weight,
            clean_anchor_metric=args.clean_anchor_metric,
            renderer=renderer,
            response_metadata=response_meta,
            arm=args.arm,
            device=device,
            amp_dtype=args.amp_dtype,
            optimizer=None,
            scheduler=None,
            global_step=global_step,
            max_train_steps=0,
            loss_weights=(
                args.type_loss_weight,
                args.visit_loss_weight,
                args.strength_loss_weight,
            ),
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
        )
        record = {
            "epoch": epoch,
            "train_plan_sha256": screen.plan_sha256(train_plan),
            "train_rows": len(train_plan),
            "train": train_metrics,
            "dev": dev_metrics,
        }
        history.append(record)
        cache_runner.atomic_json_write(output_dir / "metrics_history.json", history)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        if best is None or dev_metrics["average_precision"] > best["dev"]["average_precision"]:
            best = copy.deepcopy(record)
            best_probe_state = {
                name: value.detach().cpu().clone()
                for name, value in probe.state_dict().items()
            }
            best_dev_predictions = dev_predictions.copy()
            transfer_metadata = {
                "script_version": SCRIPT_VERSION,
                "arm": args.arm,
                "best_epoch": epoch,
                "best_dev_metrics": copy.deepcopy(dev_metrics),
                "base_weights_path": str(base_weights),
                "base_weights_sha256": base_sha,
                "trainability": trainability,
                "probe_initial_state_sha256": probe_initial_sha,
                "trainable_initial_state_sha256": trainable_initial_sha,
                "clean_anchor_contract": clean_anchor_contract,
                "renderer": asdict(renderer),
                "objective_response": list(arm_response),
                "matched_contract": matched_contract,
                "test_evaluations": 0,
            }
            save_transfer_checkpoint(
                output_dir / "backbone_best_dev_ap.pth",
                backbone=backbone,
                probe_state=best_probe_state,
                metadata=transfer_metadata,
            )
            cache_runner.atomic_csv_write(
                output_dir / "best_dev_predictions.csv",
                best_dev_predictions,
            )
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    if best is None or best_probe_state is None or best_dev_predictions is None:
        raise RuntimeError("No continue-pretraining checkpoint was selected")
    transfer_path = output_dir / "backbone_best_dev_ap.pth"
    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "complete_train_dev_only",
        "arm": args.arm,
        "best_epoch": best["epoch"],
        "best_dev_metrics": best["dev"],
        "history": history,
        "transfer_checkpoint": str(transfer_path),
        "transfer_checkpoint_sha256": cache_runner.sha256_file(transfer_path),
        "transfer_checkpoint_load_contract": (
            "Existing load_backbone reads the top-level 'backbone' state strictly"
        ),
        "base_weights_sha256": base_sha,
        "probe_initial_state_sha256": probe_initial_sha,
        "trainable_initial_state_sha256": trainable_initial_sha,
        "trainability": trainability,
        "parameter_counts": parameter_counts,
        "clean_anchor_contract": clean_anchor_contract,
        "matched_contract": matched_contract,
        "cache_audit": cache_audit,
        "test_evaluations": 0,
    }
    cache_runner.atomic_json_write(output_dir / "summary.json", summary)
    cache_runner.atomic_json_write(
        output_dir / "run_status.json",
        {
            "status": "complete",
            "script_version": SCRIPT_VERSION,
            "arm": args.arm,
            "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "test_evaluations": 0,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    root = "/diniuvol/yuyao/methanefuse_research_20260727"
    parser = argparse.ArgumentParser(
        description="Online-context P4/P5 L89 RCTP continue-pretraining."
    )
    parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument(
        "--train-csv", default=f"{root}/manifests_staged/l89_6time/train.csv"
    )
    parser.add_argument(
        "--dev-csv", default=f"{root}/manifests_staged/l89_6time/val.csv"
    )
    parser.add_argument(
        "--train-cache", default=f"{root}/cache/l89_ragged_cls_v1/train.pt"
    )
    parser.add_argument(
        "--dev-cache", default=f"{root}/cache/l89_ragged_cls_v1/val.pt"
    )
    parser.add_argument(
        "--base-weights", default="weights/panopticon_vitb14_teacher.pth"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=2048)
    parser.add_argument("--max-dev-rows", type=int, default=512)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Global cap; zero uses all planned batches.",
    )
    parser.add_argument("--train-last-blocks", type=int, choices=(1, 2), default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--probe-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--delta-periods", default="1,3,7,30,90,365")
    parser.add_argument("--type-loss-weight", type=float, default=1.0)
    parser.add_argument("--visit-loss-weight", type=float, default=0.25)
    parser.add_argument("--strength-loss-weight", type=float, default=0.10)
    parser.add_argument(
        "--clean-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Optional detached base-PTH clean-CLS anchor. Zero preserves the "
            "exact first-round loss path and does not retain cached CLS."
        ),
    )
    parser.add_argument(
        "--clean-anchor-metric",
        choices=("cosine", "l2"),
        default="cosine",
    )
    parser.add_argument(
        "--reference-label", choices=("negative", "all"), default="negative"
    )
    parser.add_argument(
        "--ch4-response",
        default=",".join(str(value) for value in screen.DEFAULT_L89_CH4_RESPONSE),
    )
    parser.add_argument(
        "--wavelength-shuffle",
        default=",".join(
            str(value) for value in screen.DEFAULT_WAVELENGTH_SHUFFLE
        ),
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
    parser.set_defaults(handler=run)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.epochs > 3:
        parser.error("--epochs must be in 1..3 for the bounded screen")
    if args.max_train_rows < 1 or args.max_dev_rows < 1:
        parser.error("--max-train-rows/--max-dev-rows must be positive")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("batch sizes must be positive")
    if not 0 <= args.warmup_fraction < 1:
        parser.error("--warmup-fraction must lie in [0,1)")
    if not math.isfinite(args.clean_anchor_weight) or args.clean_anchor_weight < 0:
        parser.error("--clean-anchor-weight must be finite and non-negative")
    try:
        args.handler(args)
    except Exception as error:
        # Do not overwrite an unrelated pre-existing directory on an early
        # validation failure. If this run already declared itself "running",
        # make a dead process visible to the monitor/audit.
        status_path = Path(args.output_dir).expanduser().resolve() / "run_status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("status") == "running":
                    cache_runner.atomic_json_write(
                        status_path,
                        {
                            "status": "failed",
                            "script_version": SCRIPT_VERSION,
                            "arm": args.arm,
                            "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "test_evaluations": 0,
                        },
                    )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
