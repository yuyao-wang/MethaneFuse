#!/usr/bin/env python3
"""Bounded, representation-preserving Sidecar-RCTP fallback for L89.

The public Panopticon encoder is always frozen.  P4 and P5 train only an
independent rank-8 response-conditioned residual sidecar and a disposable
synthetic objective probe.  The synthetic probe receives the *unscaled
sidecar residual only*, so it cannot solve the task directly from frozen base
features.  Real downstream caches expose ``[z_base, 0.1 * residual]``; P0 uses
the identical shape with an exactly-zero residual channel.

This is an exploratory train/development-only runner.  Any path containing a
test, sealed, or holdout component is rejected.
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
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("XFORMERS_DISABLED", "1")

from research.pretraining_20260727 import l89_ragged_cls_experiment as base
from research.pretraining_20260727 import rctp_l89_continue_pretrain as continuation
from research.pretraining_20260727 import rctp_l89_screen as screen
from Upgraded_dataset.dino_classifier_head_l89_temporal_satmae import load_backbone


SCRIPT_VERSION = "rctp-l89-sidecar-fallback-v2-residual-only-pretext"
ARM_P4 = continuation.ARM_P4
ARM_P5 = continuation.ARM_P5
PRETRAIN_ARMS = (ARM_P4, ARM_P5)
CACHE_ARMS = ("p0", "p4", "p5")
ARM_TO_PRETRAIN = {"p4": ARM_P4, "p5": ARM_P5}
LOCKED_RANK = 8
LOCKED_RESIDUAL_SCALE = 0.10
LOCKED_PRETEXT_RESIDUAL_SCALE = 1.0
LOCKED_CLEAN_ANCHOR_WEIGHT = 0.10
LOCKED_EPOCHS = 2
LOCKED_TRAIN_ROWS = 2048
LOCKED_DEV_ROWS = 512
DEVELOPMENT_RE = re.compile(
    r"(^|[._/\\-])(test|sealed|holdout)([._/\\-]|$)", re.IGNORECASE
)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def assert_development_path(path: Path, *, purpose: str) -> None:
    resolved = path.expanduser().resolve()
    if DEVELOPMENT_RE.search(str(resolved)):
        raise ValueError(
            f"{purpose} path is forbidden in this train/dev-only fallback: "
            f"{resolved}"
        )


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    return device


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def module_pair_sha256(
    sidecar: nn.Module, probe: nn.Module
) -> tuple[str, str, str]:
    sidecar_state = {
        name: value.detach().cpu() for name, value in sidecar.state_dict().items()
    }
    probe_state = {
        name: value.detach().cpu() for name, value in probe.state_dict().items()
    }
    combined = {
        **{f"sidecar.{name}": value for name, value in sidecar_state.items()},
        **{f"probe.{name}": value for name, value in probe_state.items()},
    }
    return (
        base.state_dict_sha256(sidecar_state),
        base.state_dict_sha256(probe_state),
        base.state_dict_sha256(combined),
    )


class ResponseLowRankSidecar(nn.Module):
    """Bias-free low-rank residual conditioned on the sensor response.

    The up projection is exactly zero at construction, so the residual is
    exactly zero for every input.  Normalization has no affine parameters;
    consequently every trainable parameter belongs to one of the three
    rank-eight matrices.
    """

    def __init__(self, feature_dim: int, response_dim: int, rank: int = 8):
        super().__init__()
        if feature_dim < 1 or response_dim < 1 or rank < 1:
            raise ValueError("feature_dim, response_dim, and rank must be positive")
        self.feature_dim = int(feature_dim)
        self.response_dim = int(response_dim)
        self.rank = int(rank)
        self.token_down = nn.Linear(feature_dim, rank, bias=False)
        self.response_down = nn.Linear(response_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    @property
    def exact_noop(self) -> bool:
        return bool(torch.count_nonzero(self.up.weight).item() == 0)

    def forward(
        self, features: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        if features.ndim < 2 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must end in the configured feature dimension")
        if response.shape[-1] != self.response_dim:
            raise ValueError("response metadata has an incompatible final dimension")
        normalized_features = F.layer_norm(features, (self.feature_dim,))
        normalized_response = F.layer_norm(response, (self.response_dim,))
        response_latent = self.response_down(normalized_response)
        while response_latent.ndim < features.ndim:
            response_latent = response_latent.unsqueeze(-2)
        latent = F.gelu(self.token_down(normalized_features) + response_latent)
        return self.up(latent)


def concatenate_base_residual(
    base_features: torch.Tensor,
    residual: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    if base_features.shape != residual.shape:
        raise ValueError("base and residual feature shapes must match")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("residual scale must be finite and positive")
    return torch.cat((base_features, float(scale) * residual), dim=-1)


def select_pretext_residual_channel(
    base_features: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Return the unscaled residual and exclude the direct base-feature path.

    ``base_features`` is accepted only to enforce the paired tensor boundary.
    Its values are deliberately never read.  This makes the no-shortcut
    contract directly testable: permuting the direct base tensor while holding
    the learned residual fixed cannot change the pretext input.
    """

    if base_features.shape != residual.shape:
        raise ValueError("base and residual feature shapes must match")
    return residual


def clean_anchor_loss(
    residual: torch.Tensor, unique_mask: torch.Tensor
) -> torch.Tensor:
    if residual.ndim != 3 or unique_mask.shape != residual.shape[:2]:
        raise ValueError("clean residual/mask must have shapes (B,T,D)/(B,T)")
    valid = unique_mask.bool()
    if not valid.any():
        raise ValueError("clean anchor requires at least one unique clean visit")
    return residual[valid].float().square().mean()


def objective_response(
    renderer: screen.RendererConfig, arm: str
) -> tuple[float, ...]:
    return continuation.objective_response(renderer, arm)


def pretrain_variant_order(variants: torch.Tensor, arm: str) -> torch.Tensor:
    return continuation.reorder_objective_variants(variants, arm)


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    seed: int,
) -> DataLoader:
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(workers),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "generator": torch.Generator().manual_seed(int(seed)),
    }
    if workers > 0:
        options["prefetch_factor"] = int(prefetch_factor)
        options["persistent_workers"] = False
    return DataLoader(**options)


def encode_frozen_group(
    backbone: nn.Module,
    clean: torch.Tensor,
    variants: torch.Tensor,
    channel_ids: torch.Tensor,
    *,
    device: torch.device,
    amp_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clean.ndim != 4 or variants.ndim != 5 or variants.shape[1] != 3:
        raise ValueError("clean/variants must have shapes (B,C,H,W)/(B,3,C,H,W)")
    batch, channels, height, width = clean.shape
    if variants.shape != (batch, 3, channels, height, width):
        raise ValueError("variant tensor shape is incompatible with clean images")
    images = torch.cat((clean[:, None], variants), dim=1).reshape(
        batch * 4, channels, height, width
    )
    identifiers = channel_ids.view(1, -1).expand(batch * 4, -1).clone()
    with torch.inference_mode(), autocast_context(device, amp_dtype):
        output = backbone.forward_features(
            {"imgs": images, "chn_ids": identifiers}
        )
        cls = output["x_norm_clstoken"].float().reshape(batch, 4, -1)
    return cls[:, 0], cls[:, 1:]


def resource_guard(
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
            f"Sidecar allocation {allocated / 2**30:.2f} GiB exceeds "
            f"{max_allocated_gib:.2f} GiB cap"
        )
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < float(min_free_gib) * (2**30):
        raise RuntimeError(
            f"GPU free memory {free_bytes / 2**30:.2f} GiB fell below "
            f"{min_free_gib:.2f} GiB"
        )
    return allocated


def run_pretext_epoch(
    *,
    backbone: nn.Module,
    sidecar: ResponseLowRankSidecar,
    probe: screen.RCTPTemporalProbe,
    loader: DataLoader,
    payload: Mapping[str, Any],
    renderer: screen.RendererConfig,
    response_metadata: torch.Tensor,
    arm: str,
    device: torch.device,
    amp_dtype: str,
    residual_scale: float,
    clean_anchor_weight: float,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    global_step: int,
    max_train_steps: int,
    loss_weights: tuple[float, float, float],
    grad_clip: float,
    log_interval: int,
    max_cuda_allocated_gib: float,
    min_runtime_cuda_free_gib: float,
) -> tuple[dict[str, Any], int, pd.DataFrame, int]:
    training = optimizer is not None
    sidecar.train(training)
    probe.train(training)
    backbone.eval()
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
    maximum_allocated = 0
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
        clean = source["clean_image"].float().to(device, non_blocking=True)
        valid_pixels = source["valid_pixels"].bool().to(device, non_blocking=True)
        plume = source["plume_field"].float().to(device, non_blocking=True)
        peak_drop = source["peak_drop"].float().to(device, non_blocking=True)
        rendered, diagnostics = screen.render_counterfactual_variants(
            clean,
            valid_pixels,
            plume,
            peak_drop,
            normalization_mean=mean,
            normalization_std=std,
            config=renderer,
        )
        if float(diagnostics["max_energy_relative_error"]) > 5e-5:
            raise RuntimeError("Nuisance energy matching exceeded tolerance")
        variants = pretrain_variant_order(rendered, arm)
        online_clean_base, variant_base = encode_frozen_group(
            backbone,
            clean,
            variants,
            channel_ids,
            device=device,
            amp_dtype=amp_dtype,
        )
        if online_clean_base.requires_grad or variant_base.requires_grad:
            raise AssertionError("Frozen backbone output unexpectedly requires gradient")

        clean_base = payload["features"][row_index_cpu].float().to(
            device, non_blocking=True
        )
        unique = payload["unique_mask"][row_index_cpu].bool().to(device)
        delta_days = payload["delta_days"][row_index_cpu].float().to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast_context(device, amp_dtype):
            clean_residual = sidecar(clean_base, metadata)
            clean_residual = clean_residual * unique.unsqueeze(-1).to(
                clean_residual.dtype
            )
            online_clean_residual = sidecar(online_clean_base, metadata)
            variant_residual = sidecar(variant_base, metadata)
            clean_pretext = select_pretext_residual_channel(
                clean_base, clean_residual
            )
            online_clean_pretext = select_pretext_residual_channel(
                online_clean_base, online_clean_residual
            )
            variant_pretext = select_pretext_residual_channel(
                variant_base, variant_residual
            )
            assembled = screen.assemble_probe_batch(
                clean_pretext,
                unique,
                delta_days,
                visit,
                online_clean_pretext,
                variant_pretext,
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
            anchor = clean_anchor_loss(clean_residual, unique)
            total_loss = pretext_loss + float(clean_anchor_weight) * anchor
            loss_values["pretext_total_loss"] = float(pretext_loss.detach())
            loss_values["clean_anchor_loss"] = float(anchor.detach())
            loss_values["clean_anchor_weighted_loss"] = float(
                clean_anchor_weight * anchor.detach()
            )
            loss_values["total_loss"] = float(total_loss.detach())
        if not torch.isfinite(total_loss):
            raise RuntimeError("Sidecar-RCTP loss became non-finite")
        if training:
            total_loss.backward()
            trainable = list(sidecar.parameters()) + list(probe.parameters())
            if any(parameter.grad is not None for parameter in backbone.parameters()):
                raise AssertionError("Frozen Panopticon encoder received gradients")
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable, float(grad_clip))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
        losses.append(loss_values)

        probability = torch.sigmoid(output["type_logit"]).detach().float().cpu()
        predicted_visit = output["visit_logits"].argmax(dim=1).detach().cpu()
        target = assembled["type_target"].detach().long().cpu()
        repeated_rows = row_index_cpu.repeat_interleave(3).tolist()
        labels.extend(target.tolist())
        probabilities.extend(probability.tolist())
        row_ids.extend(repeated_rows)
        group_ids.extend(repeated_rows)
        variant_indices.extend(torch.arange(3).repeat(len(row_index_cpu)).tolist())
        visit_targets.extend(assembled["visit_target"].detach().cpu().tolist())
        visit_predictions.extend(predicted_visit.tolist())
        target_strength = (
            peak_drop[:, None].expand(-1, 3).reshape(-1).log().detach().cpu()
        )
        strength_targets.extend(target_strength.tolist())
        strength_predictions.extend(
            output["strength_log"].detach().float().cpu().tolist()
        )
        if (
            batch_index % max(1, int(log_interval)) == 0
            or batch_index == len(loader)
        ):
            maximum_allocated = max(
                maximum_allocated,
                resource_guard(
                    device,
                    max_allocated_gib=max_cuda_allocated_gib,
                    min_free_gib=min_runtime_cuda_free_gib,
                ),
            )
            print(
                f"[sidecar {arm} {'train' if training else 'dev'}] "
                f"batch={batch_index}/{len(loader)} step={global_step} "
                f"loss={loss_values['total_loss']:.4f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    if not labels:
        raise RuntimeError("Sidecar epoch emitted no examples")
    metrics = continuation.objective_metrics(
        labels,
        probabilities,
        group_ids,
        variant_indices,
        visit_targets,
        visit_predictions,
        strength_targets,
        strength_predictions,
    )
    for key in (
        "type_loss",
        "visit_loss",
        "strength_loss",
        "pretext_total_loss",
        "clean_anchor_loss",
        "clean_anchor_weighted_loss",
        "total_loss",
    ):
        metrics[key] = float(np.mean([entry[key] for entry in losses]))
    metrics.update(
        {
            "elapsed_seconds": float(time.monotonic() - started),
            "global_optimizer_steps": int(global_step),
            "batches_emitted": int(len(losses)),
            "clean_anchor_weight": float(clean_anchor_weight),
            "cuda_max_memory_allocated_bytes": int(maximum_allocated),
        }
    )
    predictions = pd.DataFrame(
        {
            "cache_row_index": row_ids,
            "group_id": group_ids,
            "objective_variant_index": variant_indices,
            "objective_variant": [
                continuation.OBJECTIVE_VARIANT_NAMES[arm][index]
                for index in variant_indices
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
    return metrics, global_step, predictions, maximum_allocated


def command_pretrain(args: argparse.Namespace) -> None:
    if args.arm not in PRETRAIN_ARMS:
        raise ValueError(f"Unsupported pretraining arm {args.arm!r}")
    if (
        args.rank != LOCKED_RANK
        or not math.isclose(args.residual_scale, LOCKED_RESIDUAL_SCALE)
        or not math.isclose(
            args.clean_anchor_weight, LOCKED_CLEAN_ANCHOR_WEIGHT
        )
        or args.epochs != LOCKED_EPOCHS
        or args.max_train_rows != LOCKED_TRAIN_ROWS
        or args.max_dev_rows != LOCKED_DEV_ROWS
    ):
        raise ValueError(
            "The fallback is frozen to rank=8, scale=0.1, anchor=0.1, "
            "epochs=2, train_rows=2048, and dev_rows=512"
        )
    paths = {
        "train_csv": Path(args.train_csv).expanduser().resolve(),
        "dev_csv": Path(args.dev_csv).expanduser().resolve(),
        "train_cache": Path(args.train_cache).expanduser().resolve(),
        "dev_cache": Path(args.dev_cache).expanduser().resolve(),
        "base_weights": Path(args.base_weights).expanduser().resolve(),
        "output_dir": Path(args.output_dir).expanduser().resolve(),
    }
    for name, path in paths.items():
        assert_development_path(path, purpose=name)
    for name in ("train_csv", "dev_csv", "train_cache", "dev_cache", "base_weights"):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    output_dir = paths["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base.atomic_json_write(
        output_dir / "run_status.json",
        {
            "status": "running",
            "script_version": SCRIPT_VERSION,
            "arm": args.arm,
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "test_or_sealed_or_holdout_read": False,
        },
    )

    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < float(args.min_cuda_free_gib) * (2**30):
            raise RuntimeError(
                f"Only {free_bytes / 2**30:.2f} GiB CUDA memory is free; "
                f"{args.min_cuda_free_gib:.2f} GiB is required"
            )
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    train_payload, dev_payload, cache_audit = base.load_cache_pair(
        paths["train_cache"], paths["dev_cache"]
    )
    train_frame = screen.validate_frame_cache_alignment(
        paths["train_csv"], train_payload
    )
    dev_frame = screen.validate_frame_cache_alignment(
        paths["dev_csv"], dev_payload
    )
    base_weights_sha = base.sha256_file(paths["base_weights"])
    if train_payload["weights_sha256"] != base_weights_sha:
        raise ValueError("Base CLS caches do not match the frozen base PTH")

    response = tuple(
        float(value.strip())
        for value in args.ch4_response.split(",")
        if value.strip()
    )
    wavelength_shuffle = tuple(
        int(value.strip())
        for value in args.wavelength_shuffle.split(",")
        if value.strip()
    )
    renderer = screen.RendererConfig(
        response=response,
        wavelength_shuffle=wavelength_shuffle,
        min_peak_drop=float(args.min_peak_drop),
        max_peak_drop=float(args.max_peak_drop),
    )
    renderer.validate(len(train_payload["input_contract"]["band_indices"]))
    arm_response = objective_response(renderer, args.arm)
    response_metadata = screen.response_metadata(
        arm_response,
        train_payload["input_contract"]["channel_ids"],
        timepoints=int(train_payload["features"].shape[1]),
    )

    backbone = load_backbone(
        str(paths["base_weights"]), device=device, debug=args.debug
    ).to(device)
    backbone.requires_grad_(False)
    backbone.eval()
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise AssertionError("Panopticon encoder must be fully frozen")

    feature_dim = int(train_payload["features"].shape[-1])
    set_seed(args.seed)
    sidecar = ResponseLowRankSidecar(
        feature_dim, int(response_metadata.numel()), rank=args.rank
    ).to(device)
    probe = screen.RCTPTemporalProbe(
        feature_dim=feature_dim,
        response_dim=int(response_metadata.numel()),
        num_roles=int(train_payload["features"].shape[1]),
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        periods_days=tuple(
            float(value.strip())
            for value in args.delta_periods.split(",")
            if value.strip()
        ),
    ).to(device)
    sidecar_initial_sha, probe_initial_sha, combined_initial_sha = (
        module_pair_sha256(sidecar, probe)
    )
    if not sidecar.exact_noop:
        raise AssertionError("Sidecar up projection is not exactly zero")
    with torch.no_grad():
        probe_rows = train_payload["features"][:2].float().to(device)
        epoch0_residual = sidecar(probe_rows, response_metadata.to(device))
        if torch.count_nonzero(epoch0_residual).item() != 0:
            raise AssertionError("Epoch-0 residual is not bit-exact zero")
        epoch0_representation = concatenate_base_residual(
            probe_rows, epoch0_residual, scale=args.residual_scale
        )
        if not torch.equal(epoch0_representation[..., :feature_dim], probe_rows):
            raise AssertionError("Epoch-0 base channel was modified")
        if torch.count_nonzero(epoch0_representation[..., feature_dim:]).item():
            raise AssertionError("Epoch-0 residual channel was not zero")

    parameters = list(sidecar.parameters()) + list(probe.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    planned_steps = args.epochs * math.ceil(args.max_train_rows / args.batch_size)
    scheduler = continuation.cosine_schedule(
        optimizer, planned_steps, args.warmup_fraction
    )
    trainable_parameter_count = int(
        sum(parameter.numel() for parameter in parameters)
    )
    sidecar_parameter_count = int(
        sum(parameter.numel() for parameter in sidecar.parameters())
    )
    expected_sidecar_parameters = (
        feature_dim * args.rank
        + int(response_metadata.numel()) * args.rank
        + args.rank * feature_dim
    )
    if sidecar_parameter_count != expected_sidecar_parameters:
        raise AssertionError("Sidecar has parameters outside the three low-rank matrices")

    dev_plan = screen.deterministic_reference_plan(
        dev_payload,
        seed=args.seed + 100_003,
        epoch=0,
        max_rows=args.max_dev_rows,
        reference_label=args.reference_label,
        renderer=renderer,
    )
    dev_dataset = screen.RCTPReferenceDataset(
        paths["dev_csv"], dev_frame, dev_payload, dev_plan, renderer=renderer
    )
    dev_loader = make_loader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )
    run_config = {
        "script_version": SCRIPT_VERSION,
        "scope": "exploratory train/recent-dev only",
        "arm": args.arm,
        "base_weights": str(paths["base_weights"]),
        "base_weights_sha256": base_weights_sha,
        "backbone_frozen": True,
        "backbone_trainable_parameters": 0,
        "sidecar": {
            "rank": int(args.rank),
            "residual_scale": float(args.residual_scale),
            "parameter_count": sidecar_parameter_count,
            "matrices_only": True,
            "zero_initialized_up_projection": True,
            "epoch0_exact_base_channel": True,
            "epoch0_exact_zero_residual_channel": True,
            "response_conditioned": True,
        },
        "pretext_feature_contract": {
            "probe_visual_input": "unscaled_sidecar_residual_only",
            "pretext_residual_scale": LOCKED_PRETEXT_RESIDUAL_SCALE,
            "direct_base_feature_input": False,
            "zero_sidecar_means_all_zero_clean_and_delta_features": True,
            "base_used_only_upstream_to_compute_sidecar_residual": True,
        },
        "clean_anchor": {
            "weight": float(args.clean_anchor_weight),
            "target": "zero residual on valid unique clean negative visits",
            "base_encoder_requires_anchor": False,
        },
        "renderer": asdict(renderer),
        "objective_response": list(arm_response),
        "objective_variant_order": list(
            continuation.OBJECTIVE_VARIANT_ORDER[args.arm]
        ),
        "sidecar_initial_state_sha256": sidecar_initial_sha,
        "probe_initial_state_sha256": probe_initial_sha,
        "combined_initial_state_sha256": combined_initial_sha,
        "trainable_parameter_count_including_disposable_probe": (
            trainable_parameter_count
        ),
        "planned_optimizer_steps": int(planned_steps),
        "dev_plan_sha256": screen.plan_sha256(dev_plan),
        "cache_audit": cache_audit,
        "matching_contract": {
            "same_base_encoder_and_cached_clean_cls": True,
            "same_sidecar_and_probe_initialization": True,
            "same_renderer_draws": True,
            "same_optimizer_steps_and_parameter_shapes": True,
            "only_p4_p5_difference": (
                "positive variant ordering and response metadata"
            ),
        },
        "args": {
            key: value for key, value in vars(args).items() if key != "handler"
        },
        "runner_sha256": base.sha256_file(Path(__file__).resolve()),
        "test_or_sealed_or_holdout_read": False,
    }
    base.atomic_json_write(output_dir / "run_config.json", run_config)

    history: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    best_sidecar_state: Optional[dict[str, torch.Tensor]] = None
    best_probe_state: Optional[dict[str, torch.Tensor]] = None
    best_predictions: Optional[pd.DataFrame] = None
    global_step = 0
    maximum_allocated = 0
    for epoch in range(1, args.epochs + 1):
        train_plan = screen.deterministic_reference_plan(
            train_payload,
            seed=args.seed,
            epoch=epoch - 1,
            max_rows=args.max_train_rows,
            reference_label=args.reference_label,
            renderer=renderer,
        )
        train_dataset = screen.RCTPReferenceDataset(
            paths["train_csv"],
            train_frame,
            train_payload,
            train_plan,
            renderer=renderer,
        )
        train_loader = make_loader(
            train_dataset,
            batch_size=args.batch_size,
            workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            seed=args.seed + epoch,
        )
        train_metrics, global_step, _, train_peak = run_pretext_epoch(
            backbone=backbone,
            sidecar=sidecar,
            probe=probe,
            loader=train_loader,
            payload=train_payload,
            renderer=renderer,
            response_metadata=response_metadata,
            arm=args.arm,
            device=device,
            amp_dtype=args.amp_dtype,
            residual_scale=args.residual_scale,
            clean_anchor_weight=args.clean_anchor_weight,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            max_train_steps=planned_steps,
            loss_weights=(
                args.type_loss_weight,
                args.visit_loss_weight,
                args.strength_loss_weight,
            ),
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            max_cuda_allocated_gib=args.max_cuda_allocated_gib,
            min_runtime_cuda_free_gib=args.min_runtime_cuda_free_gib,
        )
        dev_metrics, _, dev_predictions, dev_peak = run_pretext_epoch(
            backbone=backbone,
            sidecar=sidecar,
            probe=probe,
            loader=dev_loader,
            payload=dev_payload,
            renderer=renderer,
            response_metadata=response_metadata,
            arm=args.arm,
            device=device,
            amp_dtype=args.amp_dtype,
            residual_scale=args.residual_scale,
            clean_anchor_weight=args.clean_anchor_weight,
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
            max_cuda_allocated_gib=args.max_cuda_allocated_gib,
            min_runtime_cuda_free_gib=args.min_runtime_cuda_free_gib,
        )
        maximum_allocated = max(maximum_allocated, train_peak, dev_peak)
        record = {
            "epoch": epoch,
            "train_plan_sha256": screen.plan_sha256(train_plan),
            "train_rows": len(train_plan),
            "train": train_metrics,
            "dev": dev_metrics,
        }
        history.append(record)
        base.atomic_json_write(output_dir / "metrics_history.json", history)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        if best is None or dev_metrics["average_precision"] > best["dev"]["average_precision"]:
            best = copy.deepcopy(record)
            best_sidecar_state = {
                name: value.detach().cpu().clone()
                for name, value in sidecar.state_dict().items()
            }
            best_probe_state = {
                name: value.detach().cpu().clone()
                for name, value in probe.state_dict().items()
            }
            best_predictions = dev_predictions.copy()
            checkpoint = {
                "script_version": SCRIPT_VERSION,
                "arm": args.arm,
                "epoch": epoch,
                "sidecar_state": best_sidecar_state,
                "probe_state": best_probe_state,
                "sidecar_config": {
                    "feature_dim": feature_dim,
                    "response_dim": int(response_metadata.numel()),
                    "rank": int(args.rank),
                    "residual_scale": float(args.residual_scale),
                    "pretext_residual_scale": LOCKED_PRETEXT_RESIDUAL_SCALE,
                    "pretext_direct_base_feature_input": False,
                },
                "base_weights": str(paths["base_weights"]),
                "base_weights_sha256": base_weights_sha,
                "objective_response": list(arm_response),
                "renderer": asdict(renderer),
                "sidecar_initial_state_sha256": sidecar_initial_sha,
                "probe_initial_state_sha256": probe_initial_sha,
                "combined_initial_state_sha256": combined_initial_sha,
                "best_dev_metrics": copy.deepcopy(dev_metrics),
                "test_or_sealed_or_holdout_read": False,
            }
            base.atomic_torch_save(
                output_dir / "sidecar_best_dev_ap.pt", checkpoint
            )
            base.atomic_csv_write(
                output_dir / "best_dev_predictions.csv", best_predictions
            )

    if (
        best is None
        or best_sidecar_state is None
        or best_probe_state is None
        or best_predictions is None
    ):
        raise RuntimeError("No Sidecar-RCTP checkpoint was selected")
    checkpoint_path = output_dir / "sidecar_best_dev_ap.pt"
    summary = {
        "status": "complete_train_dev_only",
        "script_version": SCRIPT_VERSION,
        "arm": args.arm,
        "best_epoch": int(best["epoch"]),
        "best_dev_metrics": best["dev"],
        "sidecar_checkpoint": str(checkpoint_path),
        "sidecar_checkpoint_sha256": base.sha256_file(checkpoint_path),
        "base_weights_sha256": base_weights_sha,
        "sidecar_parameter_count": sidecar_parameter_count,
        "backbone_trainable_parameters": 0,
        "sidecar_initial_state_sha256": sidecar_initial_sha,
        "probe_initial_state_sha256": probe_initial_sha,
        "combined_initial_state_sha256": combined_initial_sha,
        "history": history,
        "cuda_max_memory_allocated_bytes": int(maximum_allocated),
        "test_or_sealed_or_holdout_read": False,
    }
    base.atomic_json_write(output_dir / "summary.json", summary)
    base.atomic_json_write(
        output_dir / "run_status.json",
        {
            "status": "complete",
            "script_version": SCRIPT_VERSION,
            "arm": args.arm,
            "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "test_or_sealed_or_holdout_read": False,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def sidecar_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> ResponseLowRankSidecar:
    config = checkpoint["sidecar_config"]
    sidecar = ResponseLowRankSidecar(
        int(config["feature_dim"]),
        int(config["response_dim"]),
        rank=int(config["rank"]),
    )
    sidecar.load_state_dict(checkpoint["sidecar_state"], strict=True)
    sidecar.requires_grad_(False)
    sidecar.eval()
    return sidecar


def derive_sidecar_features(
    base_features: torch.Tensor,
    unique_mask: torch.Tensor,
    *,
    sidecar: Optional[ResponseLowRankSidecar],
    response_metadata: Optional[torch.Tensor],
    residual_scale: float,
    batch_size: int,
) -> torch.Tensor:
    if base_features.ndim != 3 or unique_mask.shape != base_features.shape[:2]:
        raise ValueError("base features/mask must have shapes (N,T,D)/(N,T)")
    rows, roles, feature_dim = base_features.shape
    output = torch.zeros(
        rows, roles, feature_dim * 2, dtype=base_features.dtype
    )
    output[..., :feature_dim] = base_features
    if sidecar is None:
        return output
    if response_metadata is None:
        raise ValueError("A trained sidecar requires response metadata")
    with torch.inference_mode():
        for start in range(0, rows, int(batch_size)):
            stop = min(rows, start + int(batch_size))
            source = base_features[start:stop].float()
            residual = sidecar(source, response_metadata)
            residual = residual * unique_mask[start:stop].unsqueeze(-1).to(
                residual.dtype
            )
            output[start:stop, :, feature_dim:] = (
                float(residual_scale) * residual
            ).to(output.dtype)
    return output


def command_build_cache(args: argparse.Namespace) -> None:
    if args.arm not in CACHE_ARMS:
        raise ValueError(f"Unsupported cache arm {args.arm!r}")
    base_cache_path = Path(args.base_cache).expanduser().resolve()
    output_path = Path(args.output_cache).expanduser().resolve()
    checkpoint_path = (
        Path(args.sidecar_checkpoint).expanduser().resolve()
        if args.sidecar_checkpoint
        else None
    )
    for purpose, path in (
        ("base cache", base_cache_path),
        ("output cache", output_path),
    ):
        assert_development_path(path, purpose=purpose)
    if checkpoint_path is not None:
        assert_development_path(checkpoint_path, purpose="sidecar checkpoint")
    if not base_cache_path.is_file():
        raise FileNotFoundError(base_cache_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    payload = base.torch_load_trusted(base_cache_path)
    if not isinstance(payload, dict):
        raise TypeError("Base cache must be a dictionary")
    base.validate_cache_payload(
        payload, path=base_cache_path, expected_split=args.split
    )
    feature_dim = int(payload["features"].shape[-1])
    base_weights_sha = str(payload["weights_sha256"])

    checkpoint: Optional[Mapping[str, Any]] = None
    sidecar: Optional[ResponseLowRankSidecar] = None
    response_metadata: Optional[torch.Tensor] = None
    representation_weights_path: str
    representation_weights_sha: str
    objective_response_values: Optional[list[float]]
    if args.arm == "p0":
        if checkpoint_path is not None:
            raise ValueError("P0 must not receive a sidecar checkpoint")
        representation_weights_path = str(payload.get("weights_path", ""))
        representation_weights_sha = base_weights_sha
        objective_response_values = None
    else:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError(
                checkpoint_path or Path("<missing sidecar checkpoint>")
            )
        checkpoint = base.torch_load_trusted(checkpoint_path)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Sidecar checkpoint must be a mapping")
        expected_arm = ARM_TO_PRETRAIN[args.arm]
        if checkpoint.get("arm") != expected_arm:
            raise ValueError(
                f"{args.arm} cache received checkpoint arm {checkpoint.get('arm')!r}"
            )
        if checkpoint.get("base_weights_sha256") != base_weights_sha:
            raise ValueError("Sidecar checkpoint and base CLS cache PTH differ")
        config = checkpoint["sidecar_config"]
        if (
            int(config["feature_dim"]) != feature_dim
            or int(config["rank"]) != LOCKED_RANK
            or not math.isclose(
                float(config["residual_scale"]), LOCKED_RESIDUAL_SCALE
            )
        ):
            raise ValueError("Sidecar checkpoint violates the frozen fallback contract")
        sidecar = sidecar_from_checkpoint(checkpoint)
        objective_response_values = [
            float(value) for value in checkpoint["objective_response"]
        ]
        response_metadata = screen.response_metadata(
            objective_response_values,
            payload["input_contract"]["channel_ids"],
            timepoints=int(payload["features"].shape[1]),
        )
        representation_weights_path = str(checkpoint_path)
        representation_weights_sha = base.sha256_file(checkpoint_path)

    combined = derive_sidecar_features(
        payload["features"],
        payload["unique_mask"].bool(),
        sidecar=sidecar,
        response_metadata=response_metadata,
        residual_scale=LOCKED_RESIDUAL_SCALE,
        batch_size=args.batch_size,
    )
    if not torch.equal(combined[..., :feature_dim], payload["features"]):
        raise AssertionError("Derived cache changed the byte-identical base channel")
    if args.arm == "p0" and torch.count_nonzero(
        combined[..., feature_dim:]
    ).item():
        raise AssertionError("P0 residual channel is not exactly zero")

    derived = dict(payload)
    derived["features"] = combined
    derived["feature_sha256"] = base.tensor_sha256(combined)
    derived["script_version"] = SCRIPT_VERSION
    derived["created_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    derived["weights_path"] = representation_weights_path
    derived["weights_sha256"] = representation_weights_sha
    contract = copy.deepcopy(payload["input_contract"])
    contract["weights_sha256"] = representation_weights_sha
    contract["representation_contract"] = {
        "version": SCRIPT_VERSION,
        "arm": args.arm,
        "base_feature_dim": feature_dim,
        "residual_feature_dim": feature_dim,
        "output_feature_dim": feature_dim * 2,
        "base_channel_byte_identical": True,
        "residual_scale": LOCKED_RESIDUAL_SCALE,
        "rank": LOCKED_RANK,
        "p0_residual_exact_zero": args.arm == "p0",
        "base_weights_sha256": base_weights_sha,
        "sidecar_checkpoint_sha256": (
            representation_weights_sha if args.arm != "p0" else None
        ),
        "objective_response": objective_response_values,
    }
    derived["input_contract"] = contract
    derived["input_contract_sha256"] = base.sha256_bytes(
        base.canonical_json_bytes(contract)
    )
    derived["sidecar_provenance"] = {
        "arm": args.arm,
        "base_cache": str(base_cache_path),
        "base_cache_sha256": base.sha256_file(base_cache_path),
        "base_feature_sha256": str(payload["feature_sha256"]),
        "base_weights_sha256": base_weights_sha,
        "sidecar_checkpoint": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
        "sidecar_checkpoint_sha256": (
            representation_weights_sha if args.arm != "p0" else None
        ),
        "runner_sha256": base.sha256_file(Path(__file__).resolve()),
        "test_or_sealed_or_holdout_read": False,
    }
    base.atomic_torch_save(output_path, derived)
    summary = {
        "status": "complete",
        "script_version": SCRIPT_VERSION,
        "arm": args.arm,
        "split": args.split,
        "rows": int(combined.shape[0]),
        "feature_shape": list(combined.shape),
        "feature_sha256": derived["feature_sha256"],
        "cache": str(output_path),
        "cache_sha256": base.sha256_file(output_path),
        "base_cache_sha256": derived["sidecar_provenance"]["base_cache_sha256"],
        "base_channel_byte_identical": True,
        "residual_channel_exact_zero": (
            bool(torch.count_nonzero(combined[..., feature_dim:]).item() == 0)
        ),
        "sidecar_checkpoint_sha256": (
            derived["sidecar_provenance"]["sidecar_checkpoint_sha256"]
        ),
        "test_or_sealed_or_holdout_read": False,
    }
    base.atomic_json_write(output_path.with_suffix(".pt.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    root = "/diniuvol/yuyao/methanefuse_research_20260727"
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain")
    pretrain.add_argument("--arm", choices=PRETRAIN_ARMS, required=True)
    pretrain.add_argument(
        "--train-csv", default=f"{root}/manifests_staged/l89_6time/train.csv"
    )
    pretrain.add_argument(
        "--dev-csv", default=f"{root}/manifests_staged/l89_6time/val.csv"
    )
    pretrain.add_argument(
        "--train-cache", default=f"{root}/cache/l89_ragged_cls_v1/train.pt"
    )
    pretrain.add_argument(
        "--dev-cache", default=f"{root}/cache/l89_ragged_cls_v1/val.pt"
    )
    pretrain.add_argument(
        "--base-weights",
        default=str(REPO_ROOT / "weights/panopticon_vitb14_teacher.pth"),
    )
    pretrain.add_argument("--output-dir", required=True)
    pretrain.add_argument("--epochs", type=int, default=LOCKED_EPOCHS)
    pretrain.add_argument("--max-train-rows", type=int, default=LOCKED_TRAIN_ROWS)
    pretrain.add_argument("--max-dev-rows", type=int, default=LOCKED_DEV_ROWS)
    pretrain.add_argument("--rank", type=int, default=LOCKED_RANK)
    pretrain.add_argument(
        "--residual-scale", type=float, default=LOCKED_RESIDUAL_SCALE
    )
    pretrain.add_argument(
        "--clean-anchor-weight",
        type=float,
        default=LOCKED_CLEAN_ANCHOR_WEIGHT,
    )
    pretrain.add_argument("--batch-size", type=int, default=4)
    pretrain.add_argument("--eval-batch-size", type=int, default=8)
    pretrain.add_argument("--num-workers", type=int, default=4)
    pretrain.add_argument("--prefetch-factor", type=int, default=1)
    pretrain.add_argument("--learning-rate", type=float, default=1e-4)
    pretrain.add_argument("--weight-decay", type=float, default=0.05)
    pretrain.add_argument("--warmup-fraction", type=float, default=0.05)
    pretrain.add_argument("--grad-clip", type=float, default=1.0)
    pretrain.add_argument("--model-dim", type=int, default=192)
    pretrain.add_argument("--num-heads", type=int, default=6)
    pretrain.add_argument("--dropout", type=float, default=0.1)
    pretrain.add_argument("--delta-periods", default="1,3,7,30,90,365")
    pretrain.add_argument("--type-loss-weight", type=float, default=1.0)
    pretrain.add_argument("--visit-loss-weight", type=float, default=0.25)
    pretrain.add_argument("--strength-loss-weight", type=float, default=0.10)
    pretrain.add_argument(
        "--reference-label", choices=("negative",), default="negative"
    )
    pretrain.add_argument(
        "--ch4-response",
        default=",".join(
            str(value) for value in screen.DEFAULT_L89_CH4_RESPONSE
        ),
    )
    pretrain.add_argument(
        "--wavelength-shuffle",
        default=",".join(
            str(value) for value in screen.DEFAULT_WAVELENGTH_SHUFFLE
        ),
    )
    pretrain.add_argument("--min-peak-drop", type=float, default=0.01)
    pretrain.add_argument("--max-peak-drop", type=float, default=0.08)
    pretrain.add_argument("--seed", type=int, default=20260728)
    pretrain.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    pretrain.add_argument(
        "--amp-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    pretrain.add_argument("--min-cuda-free-gib", type=float, default=30.0)
    pretrain.add_argument("--min-runtime-cuda-free-gib", type=float, default=25.0)
    pretrain.add_argument("--max-cuda-allocated-gib", type=float, default=12.0)
    pretrain.add_argument("--log-interval", type=int, default=20)
    pretrain.add_argument("--debug", action="store_true")
    pretrain.set_defaults(handler=command_pretrain)

    cache = subparsers.add_parser("build-cache")
    cache.add_argument("--arm", choices=CACHE_ARMS, required=True)
    cache.add_argument("--split", choices=("train", "val"), required=True)
    cache.add_argument("--base-cache", required=True)
    cache.add_argument("--sidecar-checkpoint")
    cache.add_argument("--output-cache", required=True)
    cache.add_argument("--batch-size", type=int, default=512)
    cache.set_defaults(handler=command_build_cache)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except Exception as error:
        output = getattr(args, "output_dir", None)
        if output:
            status_path = Path(output).expanduser().resolve() / "run_status.json"
            if status_path.is_file():
                try:
                    base.atomic_json_write(
                        status_path,
                        {
                            "status": "failed",
                            "script_version": SCRIPT_VERSION,
                            "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "test_or_sealed_or_holdout_read": False,
                        },
                    )
                except Exception:
                    pass
        raise


if __name__ == "__main__":
    main()
