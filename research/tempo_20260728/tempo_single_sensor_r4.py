#!/usr/bin/env python3
"""Development-only single-sensor R4 transfer pilot.

This runner reuses an immutable t0-only RaggedCurrentQueryHead checkpoint and
per-acquisition frozen CLS caches.  It trains only a small zero-initialized
residual above the frozen base logit.  There is deliberately no test command.

Arms
----
``raw_delta``
    Project and mask-average ``t0 - history``.
``r4_motion_excitation``
    Project current appearance, signed motion, and motion magnitude, then use
    the video-inspired operator
    ``appearance * sigmoid(excitation) + signed_motion``.

Both arms instantiate the same complete parameter signature and begin from the
same seed-specific state.  Epoch zero is numerically the exact frozen P0.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as cache_runner,
)
from research.tempo_20260728 import tempo_l89_global as metric_runner  # noqa: E402


SCRIPT_VERSION = "tempo-single-sensor-r4-v1"
ARMS = ("raw_delta", "r4_motion_excitation")
FORBIDDEN_RE = re.compile(
    r"(^|[._-])(test|sealed|holdout)([._-]|$)", re.IGNORECASE
)


# Historical checkpoints serialized argparse's handler while their source was
# running as __main__.  Defining this name is necessary only for trusted local
# pickle compatibility; it is never called here.
def train_heads(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("Serialized CLI handler must never be invoked.")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def assert_development_path(path: Path, *, purpose: str) -> Path:
    resolved = path.expanduser().resolve()
    offending = [
        part for part in resolved.parts if FORBIDDEN_RE.search(part.lower())
    ]
    if offending:
        raise ValueError(
            f"{purpose} path contains held-out marker {offending}: {resolved}"
        )
    cache_runner.assert_not_sealed_path(resolved, purpose=purpose)
    return resolved


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def parameter_signature(model: nn.Module) -> dict[str, Any]:
    shapes = {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
        }
        for name, parameter in model.named_parameters()
    }
    return {
        "parameter_count": int(
            sum(item["numel"] for item in shapes.values())
        ),
        "parameter_shapes": shapes,
        "shape_sha256": hashlib.sha256(
            canonical_json_bytes(shapes)
        ).hexdigest(),
    }


def fixed_batches(
    rows: int,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> list[torch.Tensor]:
    if shuffle:
        generator = torch.Generator().manual_seed(
            int(seed) + 104_729 * int(epoch)
        )
        order = torch.randperm(int(rows), generator=generator)
    else:
        order = torch.arange(int(rows))
    return list(order.split(int(batch_size)))


class FrozenT0Base(nn.Module):
    """Exact replay wrapper around the historical t0-only checkpoint."""

    def __init__(
        self,
        checkpoint: Mapping[str, Any],
        *,
        feature_dim: int,
        num_roles: int,
        t0_index: int,
    ) -> None:
        super().__init__()
        args = checkpoint["args"]
        periods = tuple(
            float(value)
            for value in str(args["delta_periods"]).split(",")
        )
        self.head = cache_runner.RaggedCurrentQueryHead(
            feature_dim=int(feature_dim),
            num_roles=int(num_roles),
            model_dim=int(args["model_dim"]),
            num_heads=int(args["num_heads"]),
            depth=2,
            mlp_ratio=float(args["mlp_ratio"]),
            dropout=float(args["dropout"]),
            periods_days=periods,
            t0_index=int(t0_index),
        )
        self.head.load_state_dict(checkpoint["model"], strict=True)
        self.t0_index = int(t0_index)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenT0Base":
        super().train(False)
        self.head.eval()
        return self

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        unique_mask: torch.Tensor,
        role_index: torch.Tensor,
    ) -> torch.Tensor:
        effective_valid = valid_mask.bool() & unique_mask.bool()
        if not effective_valid[:, self.t0_index].all():
            raise ValueError("Frozen P0 received an invalid t0.")
        history = [
            index
            for index in range(features.shape[1])
            if index != self.t0_index
        ]
        base_features = features.float().clone()
        base_features[:, history] = 0.0
        effective_valid = effective_valid.clone()
        effective_valid[:, history] = False
        delta_days = torch.zeros(
            features.shape[:2],
            device=features.device,
            dtype=torch.float32,
        )
        return self.head(
            base_features,
            effective_valid,
            role_index,
            delta_days,
            enable_delta=False,
        )


class SingleSensorR4Residual(nn.Module):
    """Appearance-motion residual above immutable t0 evidence."""

    def __init__(
        self,
        base: FrozenT0Base,
        *,
        feature_dim: int,
        num_roles: int,
        t0_index: int,
        model_dim: int,
        residual_cap: float,
    ) -> None:
        super().__init__()
        if not 0 <= int(t0_index) < int(num_roles):
            raise ValueError("t0_index is out of range.")
        self.base = copy.deepcopy(base).requires_grad_(False)
        self.base.eval()
        self.feature_dim = int(feature_dim)
        self.num_roles = int(num_roles)
        self.t0_index = int(t0_index)
        self.model_dim = int(model_dim)
        self.residual_cap = float(residual_cap)

        # Every arm declares the complete module set.  raw_delta deliberately
        # leaves the appearance/magnitude/excitation transforms dormant.
        self.signed_motion_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.appearance_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.magnitude_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.excitation = nn.Sequential(
            nn.LayerNorm(2 * model_dim),
            nn.Linear(2 * model_dim, model_dim),
        )
        self.fused_mlp = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.classifier = nn.Linear(model_dim, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def train(self, mode: bool = True) -> "SingleSensorR4Residual":
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def masked_mean(
        values: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        weights = valid.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)

    def temporal_evidence(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        unique_mask: torch.Tensor,
        *,
        arm: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected {ARMS}.")
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must be [B,T,D].")
        if valid_mask.shape != features.shape[:2]:
            raise ValueError("valid_mask shape mismatch.")
        if unique_mask.shape != features.shape[:2]:
            raise ValueError("unique_mask shape mismatch.")

        effective_valid = valid_mask.bool() & unique_mask.bool()
        current_valid = effective_valid[:, self.t0_index]
        if not current_valid.all():
            raise ValueError("Temporal residual received an invalid t0.")
        history_indices = [
            index
            for index in range(self.num_roles)
            if index != self.t0_index
        ]
        z = F.layer_norm(features.float(), (self.feature_dim,))
        current = z[:, self.t0_index]
        history = z[:, history_indices]
        token_valid = (
            current_valid.unsqueeze(1)
            & effective_valid[:, history_indices]
        )
        delta = current.unsqueeze(1) - history
        signed_motion = self.signed_motion_projection(delta)
        if arm == "raw_delta":
            evidence = signed_motion
        else:
            appearance = self.appearance_projection(
                current.unsqueeze(1).expand_as(delta)
            )
            magnitude = self.magnitude_projection(delta.abs())
            excitation = self.excitation(
                torch.cat((signed_motion, magnitude), dim=-1)
            )
            evidence = appearance * torch.sigmoid(excitation) + signed_motion
        pooled = self.masked_mean(evidence, token_valid)
        any_history = token_valid.any(dim=1)
        pooled = torch.where(
            any_history.unsqueeze(-1), pooled, torch.zeros_like(pooled)
        )
        return pooled, any_history

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        unique_mask: torch.Tensor,
        role_index: torch.Tensor,
        *,
        arm: str,
        base_logit_override: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if base_logit_override is None:
            with torch.no_grad():
                base_logit = self.base(
                    features, valid_mask, unique_mask, role_index
                )
        else:
            base_logit = base_logit_override.float()
        evidence, any_history = self.temporal_evidence(
            features, valid_mask, unique_mask, arm=arm
        )
        raw_residual = self.classifier(self.fused_mlp(evidence)).squeeze(-1)
        residual = self.residual_cap * torch.tanh(
            raw_residual / self.residual_cap
        )
        residual = torch.where(
            any_history, residual, torch.zeros_like(residual)
        )
        output = base_logit.to(residual.dtype) + residual
        if not return_aux:
            return output
        return output, {
            "base_logit": base_logit,
            "residual": residual,
            "any_history": any_history,
        }


def load_cache_pair(
    train_path: Path, dev_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_cache, dev_cache, audit = cache_runner.load_cache_pair(
        train_path, dev_path
    )
    train_indices = cache_runner.select_usable_rows(train_cache)
    dev_indices = cache_runner.select_usable_rows(dev_cache)
    train = cache_runner.take_rows(train_cache, train_indices)
    dev = cache_runner.take_rows(dev_cache, dev_indices)
    train["valid_fraction"] = train_cache["valid_fraction"][
        train_indices
    ].float()
    dev["valid_fraction"] = dev_cache["valid_fraction"][dev_indices].float()
    return train, dev, audit


def load_frozen_base(
    checkpoint_path: Path,
    *,
    feature_dim: int,
    num_roles: int,
    t0_index: int,
) -> tuple[FrozenT0Base, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Frozen P0 checkpoint is not a mapping.")
    if checkpoint.get("arm") != "t0_masked":
        raise ValueError("Frozen P0 checkpoint must be t0_masked.")
    cache_audit = checkpoint.get("cache_audit", {})
    if int(cache_audit.get("feature_dim", -1)) != int(feature_dim):
        raise ValueError("Frozen P0 feature dimension differs from cache.")
    if int(cache_audit.get("timepoints", -1)) != int(num_roles):
        raise ValueError("Frozen P0 role count differs from cache.")
    model = FrozenT0Base(
        checkpoint,
        feature_dim=feature_dim,
        num_roles=num_roles,
        t0_index=t0_index,
    )
    audit = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": cache_runner.sha256_file(checkpoint_path),
        "checkpoint_arm": str(checkpoint["arm"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation": dict(checkpoint["validation"]),
        "checkpoint_model_state_sha256": cache_runner.state_dict_sha256(
            checkpoint["model"]
        ),
    }
    return model, audit


def infer_base_logits(
    base: FrozenT0Base,
    data: Mapping[str, Any],
    role_index: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    base = base.to(device).eval()
    output = torch.empty(len(data["labels"]), dtype=torch.float32)
    with torch.inference_mode():
        for indices in fixed_batches(
            len(output),
            batch_size=batch_size,
            seed=0,
            epoch=0,
            shuffle=False,
        ):
            output[indices] = base(
                data["features"][indices].to(device),
                data["valid_mask"][indices].to(device),
                data["unique_mask"][indices].to(device),
                role_index.to(device),
            ).cpu()
    return output


def cross_event_history_shuffle(
    data: Mapping[str, Any], *, seed: int, t0_index: int
) -> dict[str, Any]:
    donors = cache_runner.build_cross_event_donor_indices(
        data["event_ids"], seed=int(seed)
    )
    (
        features,
        valid_mask,
        unique_mask,
        delta_days,
    ) = cache_runner.apply_history_donors(
        data["features"],
        data["valid_mask"],
        data["unique_mask"],
        data["delta_days"],
        donors,
        t0_index=int(t0_index),
    )
    output = dict(data)
    output["features"] = features
    output["valid_mask"] = valid_mask
    output["unique_mask"] = unique_mask
    output["delta_days"] = delta_days
    if not torch.equal(
        output["features"][:, t0_index], data["features"][:, t0_index]
    ):
        raise RuntimeError("History shuffle changed t0.")
    return output


def predict(
    model: SingleSensorR4Residual,
    data: Mapping[str, Any],
    role_index: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    arm: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device).eval()
    probabilities = torch.empty(len(data["labels"]), dtype=torch.float32)
    residuals = torch.empty_like(probabilities)
    with torch.inference_mode():
        for indices in fixed_batches(
            len(probabilities),
            batch_size=batch_size,
            seed=0,
            epoch=0,
            shuffle=False,
        ):
            logits, aux = model(
                data["features"][indices].to(device),
                data["valid_mask"][indices].to(device),
                data["unique_mask"][indices].to(device),
                role_index.to(device),
                arm=arm,
                base_logit_override=base_logits[indices].to(device),
                return_aux=True,
            )
            probabilities[indices] = torch.sigmoid(logits).cpu()
            residuals[indices] = aux["residual"].cpu()
    return probabilities.numpy(), residuals.numpy()


def run_arm(
    *,
    arm: str,
    seed: int,
    base: FrozenT0Base,
    initial_state: Mapping[str, torch.Tensor],
    initial_state_sha256: str,
    signature: Mapping[str, Any],
    train: Mapping[str, Any],
    dev: Mapping[str, Any],
    shuffled_dev: Mapping[str, Any],
    role_index: torch.Tensor,
    train_base_logits: torch.Tensor,
    dev_base_logits: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    residual_l2: float,
    residual_cap: float,
    model_dim: int,
    patience: int,
) -> dict[str, Any]:
    arm_dir = output_dir / f"{arm}_seed{seed}"
    if arm_dir.exists():
        raise FileExistsError(f"Refusing existing arm directory: {arm_dir}")
    arm_dir.mkdir(parents=True)
    set_seed(seed)
    model = SingleSensorR4Residual(
        base,
        feature_dim=int(train["features"].shape[-1]),
        num_roles=int(train["features"].shape[1]),
        t0_index=int(base.t0_index),
        model_dim=int(model_dim),
        residual_cap=float(residual_cap),
    )
    model.load_state_dict(initial_state, strict=True)
    if parameter_signature(model) != signature:
        raise RuntimeError("Matched parameter signature changed.")
    if cache_runner.state_dict_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError("Matched initial state changed.")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    row_weight, positive_weight = metric_runner.event_training_weights(
        train["labels"], train["event_ids"]
    )
    labels = train["labels"].float()
    base_probability = torch.sigmoid(dev_base_logits).numpy()
    p0_metrics = metric_runner.metric_bundle(
        dev["labels"].numpy().astype(np.int64),
        base_probability,
        dev["event_ids"],
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "validation": p0_metrics,
            "exact_p0": True,
            "residual_abs_mean": 0.0,
        }
    ]
    best: Optional[dict[str, Any]] = None
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_probability: Optional[np.ndarray] = None
    best_residual: Optional[np.ndarray] = None
    stale = 0
    started = time.monotonic()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        loss_sum = bce_sum = l2_sum = 0.0
        seen = 0
        batches = fixed_batches(
            len(labels),
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            shuffle=True,
        )
        for indices in batches:
            target = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(
                train["features"][indices].to(device),
                train["valid_mask"][indices].to(device),
                train["unique_mask"][indices].to(device),
                role_index.to(device),
                arm=arm,
                base_logit_override=train_base_logits[indices].to(device),
                return_aux=True,
            )
            raw_bce = F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
                pos_weight=torch.tensor(
                    positive_weight, dtype=torch.float32, device=device
                ),
            )
            weights = row_weight[indices].to(device)
            bce = (raw_bce * weights).sum() / weights.sum().clamp_min(1e-8)
            l2 = aux["residual"].square().mean()
            loss = bce + float(residual_l2) * l2
            if not torch.isfinite(loss):
                raise RuntimeError(f"{arm} epoch={epoch}: non-finite loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                1.0,
            )
            optimizer.step()
            count = int(indices.numel())
            loss_sum += float(loss.detach()) * count
            bce_sum += float(bce.detach()) * count
            l2_sum += float(l2.detach()) * count
            seen += count

        probability, residual = predict(
            model,
            dev,
            role_index,
            dev_base_logits,
            arm=arm,
            batch_size=eval_batch_size,
            device=device,
        )
        metrics = metric_runner.metric_bundle(
            dev["labels"].numpy().astype(np.int64),
            probability,
            dev["event_ids"],
        )
        record = {
            "epoch": int(epoch),
            "train": {
                "loss": loss_sum / max(seen, 1),
                "event_balanced_bce": bce_sum / max(seen, 1),
                "residual_l2": l2_sum / max(seen, 1),
                "rows": int(seen),
                "steps": int(len(batches)),
            },
            "validation": metrics,
            "exact_p0": False,
            "residual_abs_mean": float(np.abs(residual).mean()),
            "residual_abs_max": float(np.abs(residual).max()),
            "elapsed_seconds": float(time.monotonic() - started),
        }
        history.append(record)
        key = (
            float(metrics["event_balanced_ap"]),
            float(metrics["event_balanced_auc"]),
            float(metrics["event_balanced_positive_f1_selected"]),
        )
        prior_key = (
            (-math.inf, -math.inf, -math.inf)
            if best is None
            else (
                float(best["validation"]["event_balanced_ap"]),
                float(best["validation"]["event_balanced_auc"]),
                float(
                    best["validation"][
                        "event_balanced_positive_f1_selected"
                    ]
                ),
            )
        )
        if key > prior_key:
            best = copy.deepcopy(record)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_probability = probability.copy()
            best_residual = residual.copy()
            stale = 0
        else:
            stale += 1
        cache_runner.atomic_json_write(arm_dir / "metrics_history.json", history)
        print(
            f"[single-R4] arm={arm} epoch={epoch}/{epochs} "
            f"F1={metrics['event_balanced_positive_f1_selected']:.6f} "
            f"macro={metrics['event_balanced_macro_f1_selected']:.6f} "
            f"AP={metrics['event_balanced_ap']:.6f} "
            f"AUC={metrics['event_balanced_auc']:.6f} "
            f"|res|={record['residual_abs_mean']:.6f}",
            flush=True,
        )
        # Fail fast after the first epoch only when both ranking metrics are
        # already below exact P0.  Otherwise ordinary patience-one applies.
        if (
            epoch == 1
            and metrics["event_balanced_ap"] < p0_metrics["event_balanced_ap"]
            and metrics["event_balanced_auc"]
            < p0_metrics["event_balanced_auc"]
        ):
            print(
                f"[single-R4] arm={arm} stop: epoch1 AP and AUC both below P0",
                flush=True,
            )
            break
        if int(patience) > 0 and stale >= int(patience):
            print(
                f"[single-R4] arm={arm} early stop epoch={epoch} "
                f"patience={patience}",
                flush=True,
            )
            break

    if (
        best is None
        or best_state is None
        or best_probability is None
        or best_residual is None
    ):
        raise RuntimeError(f"{arm} completed no epoch.")
    model.load_state_dict(best_state, strict=True)
    selected_threshold = float(
        best["validation"]["selected_threshold"]
    )
    shuffled_probability, _ = predict(
        model,
        shuffled_dev,
        role_index,
        dev_base_logits,
        arm=arm,
        batch_size=eval_batch_size,
        device=device,
    )
    shuffled_metrics = metric_runner.metric_bundle(
        dev["labels"].numpy().astype(np.int64),
        shuffled_probability,
        dev["event_ids"],
        threshold=selected_threshold,
    )
    cache_runner.atomic_torch_save(
        arm_dir / "checkpoint_best.pt",
        {
            "script_version": SCRIPT_VERSION,
            "arm": arm,
            "seed": int(seed),
            "epoch": int(best["epoch"]),
            "model": best_state,
            "model_config": {
                "feature_dim": int(train["features"].shape[-1]),
                "num_roles": int(train["features"].shape[1]),
                "t0_index": int(base.t0_index),
                "model_dim": int(model_dim),
                "residual_cap": float(residual_cap),
            },
            "parameter_signature": signature,
            "initial_state_sha256": initial_state_sha256,
            "validation": best["validation"],
            "locked_dev_threshold": selected_threshold,
            "test_or_sealed_or_holdout_read": False,
        },
    )
    cache_runner.atomic_csv_write(
        arm_dir / "dev_predictions_best.csv",
        pd.DataFrame(
            {
                "id": dev["ids"],
                "plume_id": dev["plume_ids"],
                "event_id": dev["event_ids"],
                "label": dev["labels"].long().numpy(),
                "p0_probability": base_probability,
                "probability": best_probability,
                "residual": best_residual,
                "history_shuffle_probability": shuffled_probability,
            }
        ),
    )
    result = {
        "script_version": SCRIPT_VERSION,
        "arm": arm,
        "seed": int(seed),
        "p0": p0_metrics,
        "best": best,
        "history_shuffle_fixed_threshold": shuffled_metrics,
        "history_shuffle_delta": {
            key: float(shuffled_metrics[key] - best["validation"][key])
            for key in (
                "event_balanced_positive_f1_selected",
                "event_balanced_macro_f1_selected",
                "event_balanced_ap",
                "event_balanced_auc",
            )
        },
        "parameter_signature": signature,
        "initial_state_sha256": initial_state_sha256,
        "checkpoint": str(arm_dir / "checkpoint_best.pt"),
        "predictions": str(arm_dir / "dev_predictions_best.csv"),
        "test_or_sealed_or_holdout_read": False,
    }
    cache_runner.atomic_json_write(arm_dir / "result.json", result)
    return result


def run(args: argparse.Namespace) -> None:
    train_path = assert_development_path(
        Path(args.train_cache), purpose="training cache"
    )
    dev_path = assert_development_path(
        Path(args.dev_cache), purpose="inner-development cache"
    )
    checkpoint_path = assert_development_path(
        Path(args.p0_checkpoint), purpose="frozen P0 checkpoint"
    )
    old_prediction_path = assert_development_path(
        Path(args.p0_prediction_csv), purpose="frozen P0 predictions"
    )
    output_dir = assert_development_path(
        Path(args.output_dir), purpose="experiment output"
    )
    for path in (train_path, dev_path, checkpoint_path, old_prediction_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output: {output_dir}")
    output_dir.mkdir(parents=True)

    device = torch.device(args.device)
    if device.type != "cuda" or device.index != 0:
        raise ValueError("This bounded pilot is authorized only on cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    torch.cuda.set_device(device)

    train, dev, cache_audit = load_cache_pair(train_path, dev_path)
    num_roles = int(train["features"].shape[1])
    feature_dim = int(train["features"].shape[2])
    t0_index = int(cache_audit["t0_index"])
    role_index = torch.arange(num_roles, dtype=torch.long)
    base, base_audit = load_frozen_base(
        checkpoint_path,
        feature_dim=feature_dim,
        num_roles=num_roles,
        t0_index=t0_index,
    )
    train_base_logits = infer_base_logits(
        base,
        train,
        role_index,
        batch_size=args.eval_batch_size,
        device=device,
    )
    dev_base_logits = infer_base_logits(
        base,
        dev,
        role_index,
        batch_size=args.eval_batch_size,
        device=device,
    )
    dev_base_probability = torch.sigmoid(dev_base_logits).numpy()
    old = pd.read_csv(old_prediction_path)
    if old["id"].astype(str).tolist() != list(map(str, dev["ids"])):
        raise RuntimeError("P0 prediction identities differ from dev cache.")
    replay_error = float(
        np.max(
            np.abs(
                old["probability"].to_numpy(dtype=np.float64)
                - dev_base_probability.astype(np.float64)
            )
        )
    )
    if replay_error > 1e-6:
        raise RuntimeError(
            f"Frozen P0 replay failed: max abs error={replay_error:.3e}"
        )
    base_audit.update(
        {
            "prediction_csv": str(old_prediction_path),
            "prediction_csv_sha256": cache_runner.sha256_file(
                old_prediction_path
            ),
            "replay_max_abs_probability_error": replay_error,
            "replay_tolerance": 1e-6,
            "replay_passed": True,
        }
    )
    p0_metrics = metric_runner.metric_bundle(
        dev["labels"].numpy().astype(np.int64),
        dev_base_probability,
        dev["event_ids"],
    )
    cache_audit.update(
        {
            "train_rows_usable": int(len(train["labels"])),
            "dev_rows_usable": int(len(dev["labels"])),
            "train_events": int(len(set(map(str, train["event_ids"])))),
            "dev_events": int(len(set(map(str, dev["event_ids"])))),
            "event_overlap": int(
                len(
                    set(map(str, train["event_ids"]))
                    & set(map(str, dev["event_ids"]))
                )
            ),
            "train_unique_visits": int(train["unique_mask"].sum().item()),
            "dev_unique_visits": int(dev["unique_mask"].sum().item()),
            "roles": list(
                torch.load(
                    train_path, map_location="cpu", weights_only=False
                )["role_names"]
            ),
        }
    )
    if cache_audit["event_overlap"] != 0:
        raise RuntimeError("Train/dev canonical event overlap is nonzero.")

    set_seed(args.seed)
    prototype = SingleSensorR4Residual(
        base,
        feature_dim=feature_dim,
        num_roles=num_roles,
        t0_index=t0_index,
        model_dim=args.model_dim,
        residual_cap=args.residual_cap,
    )
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in prototype.state_dict().items()
    }
    signature = parameter_signature(prototype)
    initial_state_sha = cache_runner.state_dict_sha256(initial_state)
    del prototype
    shuffled_dev = cross_event_history_shuffle(
        dev, seed=args.seed + 700_001, t0_index=t0_index
    )

    run_config = {
        "script_version": SCRIPT_VERSION,
        "args": {
            key: value
            for key, value in vars(args).items()
            if key != "handler"
        },
        "cache_audit": cache_audit,
        "base_audit": base_audit,
        "p0_exact_metrics": p0_metrics,
        "parameter_signature": signature,
        "initial_state_sha256": initial_state_sha,
        "operator": {
            "raw_delta": "mean(project(t0-history))",
            "r4_motion_excitation": (
                "mean(appearance(t0)*sigmoid(g(signed,abs))+signed_motion)"
            ),
        },
        "selection": (
            "best event-balanced AP, then AUC, then selected positive F1; "
            "maximum 3 epochs; patience 1"
        ),
        "test_or_sealed_or_holdout_read": False,
    }
    cache_runner.atomic_json_write(output_dir / "run_config.json", run_config)
    cache_runner.atomic_json_write(
        output_dir / "run_status.json",
        {
            "status": "running",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "test_or_sealed_or_holdout_read": False,
        },
    )
    print(
        f"[single-R4] P0 exact F1="
        f"{p0_metrics['event_balanced_positive_f1_selected']:.6f} "
        f"macro={p0_metrics['event_balanced_macro_f1_selected']:.6f} "
        f"AP={p0_metrics['event_balanced_ap']:.6f} "
        f"AUC={p0_metrics['event_balanced_auc']:.6f} "
        f"replay_error={replay_error:.3e}",
        flush=True,
    )

    results: dict[str, Any] = {}
    for arm in ARMS:
        results[arm] = run_arm(
            arm=arm,
            seed=args.seed,
            base=base,
            initial_state=initial_state,
            initial_state_sha256=initial_state_sha,
            signature=signature,
            train=train,
            dev=dev,
            shuffled_dev=shuffled_dev,
            role_index=role_index,
            train_base_logits=train_base_logits,
            dev_base_logits=dev_base_logits,
            output_dir=output_dir,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            residual_l2=args.residual_l2,
            residual_cap=args.residual_cap,
            model_dim=args.model_dim,
            patience=args.patience,
        )
    aggregate = {
        "script_version": SCRIPT_VERSION,
        "cache_audit": cache_audit,
        "base_audit": base_audit,
        "p0_exact_metrics": p0_metrics,
        "arms": results,
        "test_or_sealed_or_holdout_read": False,
    }
    cache_runner.atomic_json_write(output_dir / "aggregate.json", aggregate)
    cache_runner.atomic_json_write(
        output_dir / "run_status.json",
        {
            "status": "complete",
            "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "test_or_sealed_or_holdout_read": False,
        },
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    root = "/diniuvol/yuyao/methanefuse_research_20260727"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cache",
        default=f"{root}/cache/emit_ragged_cls_v1/train.pt",
    )
    parser.add_argument(
        "--dev-cache",
        default=f"{root}/cache/emit_ragged_cls_v1/val.pt",
    )
    parser.add_argument(
        "--p0-checkpoint",
        default=(
            f"{root}/results/emit_ragged_cls_v1_seed20260728/"
            "t0_masked/checkpoint_best_ap.pt"
        ),
    )
    parser.add_argument(
        "--p0-prediction-csv",
        default=(
            f"{root}/results/emit_ragged_cls_v1_seed20260728/"
            "t0_masked/validation_best_ap_predictions.csv"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--model-dim", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--residual-l2", type=float, default=1e-3)
    parser.add_argument("--residual-cap", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
