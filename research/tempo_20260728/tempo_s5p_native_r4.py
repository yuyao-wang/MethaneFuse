#!/usr/bin/env python3
"""Bounded R4 screen on the existing approximate S5P six-time grid cache.

The source is *not* exact native S5P.  Each cached 3x3 frame is a finite-mask
adaptive average of an upstream 224x224 upsampled field.  Results from this
runner are directional development evidence only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
from research.pretraining_20260727 import (  # noqa: E402
    s5p_native_grid_experiment as s5p_runner,
)
from research.tempo_20260728 import tempo_l89_global as metric_runner  # noqa: E402
from research.tempo_20260728.tempo_single_sensor_r4 import (  # noqa: E402
    ARMS,
    fixed_batches,
    parameter_signature,
    set_seed,
)


SCRIPT_VERSION = "tempo-s5p-approx-native-r4-v1"
NATIVE_DISCLAIMER = (
    "The 3x3 frames are finite-mask adaptive averages of 224x224 fields "
    "that were already upsampled upstream. They are approximate summaries, "
    "not exact original/native S5P values or learned CLS features."
)
FORBIDDEN_RE = re.compile(
    r"(^|[._-])(test|sealed|holdout)([._-]|$)", re.IGNORECASE
)


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


class FrozenS5PT0(nn.Module):
    def __init__(self, checkpoint: Mapping[str, Any]) -> None:
        super().__init__()
        config = checkpoint["model_config"]
        normalization = checkpoint["normalization"]
        self.model = s5p_runner.NativeGridTemporalClassifier(
            train_mean=float(normalization["mean"]),
            train_std=float(normalization["std"]),
            hidden_dim=int(config["hidden_dim"]),
            num_heads=int(config["num_heads"]),
            dropout=float(config["dropout"]),
        )
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenS5PT0":
        super().train(False)
        self.model.eval()
        return self

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        delta_days: torch.Tensor,
        roles: torch.Tensor,
    ) -> torch.Tensor:
        inputs = s5p_runner.apply_arm_inputs(
            "t0_masked",
            features,
            valid_mask,
            delta_days,
            roles,
            training=False,
        )
        return self.model(
            inputs.features,
            inputs.valid_mask,
            inputs.delta_days,
            inputs.roles,
            use_delta=False,
        )


class S5PNativeR4Residual(nn.Module):
    """R4 over normalized 3x3 values plus their finite-cell masks."""

    def __init__(
        self,
        *,
        train_mean: float,
        train_std: float,
        model_dim: int,
        residual_cap: float,
    ) -> None:
        super().__init__()
        if not math.isfinite(train_std) or train_std <= 0:
            raise ValueError("Invalid S5P training standard deviation.")
        self.register_buffer(
            "train_mean", torch.tensor(float(train_mean), dtype=torch.float32)
        )
        self.register_buffer(
            "train_std", torch.tensor(float(train_std), dtype=torch.float32)
        )
        self.input_dim = 18
        self.model_dim = int(model_dim)
        self.residual_cap = float(residual_cap)
        self.signed_motion_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.appearance_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.magnitude_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, model_dim),
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

    @staticmethod
    def masked_mean(
        values: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        weight = valid.to(values.dtype).unsqueeze(-1)
        return (values * weight).sum(dim=1) / weight.sum(
            dim=1
        ).clamp_min(1.0)

    def vectorize(
        self, features: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        normalized = (features.float() - self.train_mean) / self.train_std
        normalized = torch.where(
            valid_mask, normalized, torch.zeros_like(normalized)
        )
        return torch.cat(
            (
                normalized.flatten(2),
                valid_mask.to(normalized.dtype).flatten(2),
            ),
            dim=-1,
        )

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        arm: str,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}.")
        if features.ndim != 4 or features.shape[1:] != (6, 3, 3):
            raise ValueError("S5P features must be [B,6,3,3].")
        if valid_mask.shape != features.shape:
            raise ValueError("S5P valid mask shape mismatch.")
        vector = self.vectorize(features, valid_mask)
        token_valid = valid_mask.flatten(2).any(dim=-1)
        current = vector[:, 0]
        history = vector[:, 1:]
        history_valid = token_valid[:, :1] & token_valid[:, 1:]
        delta = current.unsqueeze(1) - history
        signed = self.signed_motion_projection(delta)
        if arm == "raw_delta":
            evidence = signed
        else:
            appearance = self.appearance_projection(
                current.unsqueeze(1).expand_as(delta)
            )
            magnitude = self.magnitude_projection(delta.abs())
            excitation = self.excitation(
                torch.cat((signed, magnitude), dim=-1)
            )
            evidence = appearance * torch.sigmoid(excitation) + signed
        pooled = self.masked_mean(evidence, history_valid)
        any_history = history_valid.any(dim=1)
        pooled = torch.where(
            any_history.unsqueeze(-1), pooled, torch.zeros_like(pooled)
        )
        residual = self.classifier(self.fused_mlp(pooled)).squeeze(-1)
        residual = self.residual_cap * torch.tanh(
            residual / self.residual_cap
        )
        residual = torch.where(
            any_history, residual, torch.zeros_like(residual)
        )
        logits = base_logits.float() + residual
        if not return_aux:
            return logits
        return logits, {
            "residual": residual,
            "any_history": any_history,
        }


def load_cache(path: Path, *, expected_split: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Malformed cache: {path}")
    required = {
        "features",
        "valid_mask",
        "delta_days",
        "roles",
        "labels",
        "event_ids",
        "plume_ids",
        "meta",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path} missing {missing}")
    if str(payload["meta"].get("split")) != expected_split:
        raise ValueError(
            f"{path} split={payload['meta'].get('split')} != {expected_split}"
        )
    if tuple(payload["features"].shape[1:]) != (6, 3, 3):
        raise ValueError("S5P cache does not contain six 3x3 frames.")
    return dict(payload)


def infer_base_logits(
    base: FrozenS5PT0,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    base = base.to(device).eval()
    output = torch.empty(len(cache["labels"]), dtype=torch.float32)
    with torch.inference_mode():
        for indices in fixed_batches(
            len(output),
            batch_size=batch_size,
            seed=0,
            epoch=0,
            shuffle=False,
        ):
            output[indices] = base(
                cache["features"][indices].to(device),
                cache["valid_mask"][indices].to(device),
                cache["delta_days"][indices].to(device),
                cache["roles"][indices].to(device),
            ).cpu()
    return output


def shuffled_history_cache(
    cache: Mapping[str, Any], *, seed: int
) -> dict[str, Any]:
    donors = s5p_runner.build_cross_event_donor_indices(
        cache["event_ids"], seed=int(seed)
    )
    output = dict(cache)
    features, valid = s5p_runner.replace_history_with_donors(
        cache["features"],
        cache["valid_mask"],
        cache["features"][donors],
        cache["valid_mask"][donors],
    )
    output["features"] = features
    output["valid_mask"] = valid
    if not torch.equal(features[:, 0], cache["features"][:, 0]):
        raise RuntimeError("S5P history shuffle changed t0.")
    return output


def predict(
    model: S5PNativeR4Residual,
    cache: Mapping[str, Any],
    base_logits: torch.Tensor,
    *,
    arm: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device).eval()
    probability = torch.empty(len(cache["labels"]), dtype=torch.float32)
    residual = torch.empty_like(probability)
    with torch.inference_mode():
        for indices in fixed_batches(
            len(probability),
            batch_size=batch_size,
            seed=0,
            epoch=0,
            shuffle=False,
        ):
            logits, aux = model(
                cache["features"][indices].to(device),
                cache["valid_mask"][indices].to(device),
                base_logits[indices].to(device),
                arm=arm,
                return_aux=True,
            )
            probability[indices] = torch.sigmoid(logits).cpu()
            residual[indices] = aux["residual"].cpu()
    return probability.numpy(), residual.numpy()


def run_arm(
    *,
    arm: str,
    args: argparse.Namespace,
    prototype_state: Mapping[str, torch.Tensor],
    signature: Mapping[str, Any],
    initial_state_sha: str,
    normalization: Mapping[str, Any],
    train: Mapping[str, Any],
    dev: Mapping[str, Any],
    shuffled_dev: Mapping[str, Any],
    train_base: torch.Tensor,
    dev_base: torch.Tensor,
    p0_metrics: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    arm_dir = output_dir / f"{arm}_seed{args.seed}"
    arm_dir.mkdir(parents=True)
    set_seed(args.seed)
    model = S5PNativeR4Residual(
        train_mean=float(normalization["mean"]),
        train_std=float(normalization["std"]),
        model_dim=int(args.model_dim),
        residual_cap=float(args.residual_cap),
    )
    model.load_state_dict(prototype_state, strict=True)
    if parameter_signature(model) != signature:
        raise RuntimeError("Matched parameter signature changed.")
    if s5p_runner.state_dict_sha256(model.state_dict()) != initial_state_sha:
        raise RuntimeError("Matched initialization changed.")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    labels = train["labels"].float()
    row_weight, positive_weight = metric_runner.event_training_weights(
        labels, train["event_ids"]
    )
    history: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_probability: Optional[np.ndarray] = None
    best_residual: Optional[np.ndarray] = None
    stale = 0
    started = time.monotonic()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        batches = fixed_batches(
            len(labels),
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=epoch,
            shuffle=True,
        )
        for indices in batches:
            target = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(
                train["features"][indices].to(device),
                train["valid_mask"][indices].to(device),
                train_base[indices].to(device),
                arm=arm,
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
            weight = row_weight[indices].to(device)
            bce = (raw_bce * weight).sum() / weight.sum().clamp_min(1e-8)
            l2 = aux["residual"].square().mean()
            loss = bce + float(args.residual_l2) * l2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(indices.numel())
            loss_sum += float(loss.detach()) * count
            seen += count

        probability, residual = predict(
            model,
            dev,
            dev_base,
            arm=arm,
            batch_size=args.eval_batch_size,
            device=device,
        )
        metrics = metric_runner.metric_bundle(
            dev["labels"].numpy().astype(np.int64),
            probability,
            dev["event_ids"],
        )
        record = {
            "epoch": int(epoch),
            "train_loss": loss_sum / max(seen, 1),
            "validation": metrics,
            "residual_abs_mean": float(np.abs(residual).mean()),
            "elapsed_seconds": float(time.monotonic() - started),
        }
        history.append(record)
        key = (
            float(metrics["event_balanced_ap"]),
            float(metrics["event_balanced_auc"]),
        )
        prior = (
            (-math.inf, -math.inf)
            if best is None
            else (
                float(best["validation"]["event_balanced_ap"]),
                float(best["validation"]["event_balanced_auc"]),
            )
        )
        if key > prior:
            best = copy.deepcopy(record)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_probability = probability.copy()
            best_residual = residual.copy()
            stale = 0
        else:
            stale += 1
        cache_runner.atomic_json_write(arm_dir / "metrics_history.json", history)
        print(
            f"[S5P-R4] arm={arm} epoch={epoch}/{args.epochs} "
            f"rowF1={metrics['row_positive_f1_selected']:.6f} "
            f"rowAP={metrics['row_ap']:.6f} rowAUC={metrics['row_auc']:.6f} "
            f"eventAP={metrics['event_balanced_ap']:.6f} "
            f"eventAUC={metrics['event_balanced_auc']:.6f}",
            flush=True,
        )
        if (
            epoch == 1
            and metrics["row_ap"] <= float(args.old_role_ap)
            and metrics["row_auc"] <= float(args.old_role_auc)
        ):
            print(
                f"[S5P-R4] arm={arm} stop: epoch1 AP/AUC do not exceed "
                "the frozen old role-only mean",
                flush=True,
            )
            break
        if int(args.patience) > 0 and stale >= int(args.patience):
            print(
                f"[S5P-R4] arm={arm} early stop patience={args.patience}",
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
    threshold = float(best["validation"]["selected_threshold"])
    shuffled_probability, _ = predict(
        model,
        shuffled_dev,
        dev_base,
        arm=arm,
        batch_size=args.eval_batch_size,
        device=device,
    )
    shuffle_metrics = metric_runner.metric_bundle(
        dev["labels"].numpy().astype(np.int64),
        shuffled_probability,
        dev["event_ids"],
        threshold=threshold,
    )
    cache_runner.atomic_torch_save(
        arm_dir / "checkpoint_best.pt",
        {
            "script_version": SCRIPT_VERSION,
            "arm": arm,
            "seed": int(args.seed),
            "epoch": int(best["epoch"]),
            "model": best_state,
            "model_config": {
                "input_dim": 18,
                "model_dim": int(args.model_dim),
                "residual_cap": float(args.residual_cap),
            },
            "parameter_signature": signature,
            "initial_state_sha256": initial_state_sha,
            "validation": best["validation"],
            "native_grid_disclaimer": NATIVE_DISCLAIMER,
            "test_or_sealed_or_holdout_read": False,
        },
    )
    cache_runner.atomic_csv_write(
        arm_dir / "dev_predictions_best.csv",
        pd.DataFrame(
            {
                "row_id": dev["row_ids"].numpy(),
                "plume_id": dev["plume_ids"],
                "event_id": dev["event_ids"],
                "label": dev["labels"].numpy(),
                "p0_probability": torch.sigmoid(dev_base).numpy(),
                "probability": best_probability,
                "residual": best_residual,
                "history_shuffle_probability": shuffled_probability,
            }
        ),
    )
    result = {
        "arm": arm,
        "seed": int(args.seed),
        "p0": dict(p0_metrics),
        "best": best,
        "history_shuffle_fixed_threshold": shuffle_metrics,
        "history_shuffle_delta": {
            key: float(
                shuffle_metrics[key] - best["validation"][key]
            )
            for key in (
                "event_balanced_positive_f1_selected",
                "event_balanced_macro_f1_selected",
                "event_balanced_ap",
                "event_balanced_auc",
                "row_ap",
                "row_auc",
            )
        },
        "native_grid_disclaimer": NATIVE_DISCLAIMER,
        "test_or_sealed_or_holdout_read": False,
    }
    cache_runner.atomic_json_write(arm_dir / "result.json", result)
    return result


def run(args: argparse.Namespace) -> None:
    train_path = assert_development_path(
        Path(args.train_cache), purpose="S5P train cache"
    )
    dev_path = assert_development_path(
        Path(args.dev_cache), purpose="S5P inner-dev cache"
    )
    p0_path = assert_development_path(
        Path(args.p0_checkpoint), purpose="S5P frozen P0"
    )
    old_result_path = assert_development_path(
        Path(args.old_result_json), purpose="S5P old result"
    )
    output_dir = assert_development_path(
        Path(args.output_dir), purpose="S5P R4 output"
    )
    for path in (train_path, dev_path, p0_path, old_result_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index != 0:
        raise ValueError("This pilot is authorized only on cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    torch.cuda.set_device(device)

    train = load_cache(train_path, expected_split="train")
    dev = load_cache(dev_path, expected_split="val")
    overlap = set(map(str, train["event_ids"])) & set(
        map(str, dev["event_ids"])
    )
    if overlap:
        raise RuntimeError(f"S5P event overlap={len(overlap)}")
    checkpoint = torch.load(
        p0_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("arm") != "t0_masked":
        raise ValueError("S5P P0 checkpoint is not t0_masked.")
    if (
        checkpoint.get("train_input_sha256")
        != train["meta"]["input_sha256"]
        or checkpoint.get("val_input_sha256")
        != dev["meta"]["input_sha256"]
    ):
        raise RuntimeError("S5P P0 is not bound to these caches.")
    base = FrozenS5PT0(checkpoint)
    train_base = infer_base_logits(
        base,
        train,
        batch_size=args.eval_batch_size,
        device=device,
    )
    dev_base = infer_base_logits(
        base,
        dev,
        batch_size=args.eval_batch_size,
        device=device,
    )
    p0_probability = torch.sigmoid(dev_base).numpy()
    p0_metrics = metric_runner.metric_bundle(
        dev["labels"].numpy().astype(np.int64),
        p0_probability,
        dev["event_ids"],
    )
    old_result = json.loads(old_result_path.read_text(encoding="utf-8"))
    old_p0 = old_result["arms"]["t0_masked"]["best"]["val"]
    replay = {
        "row_ap_abs_error": abs(float(old_p0["ap"]) - p0_metrics["row_ap"]),
        "row_auc_abs_error": abs(
            float(old_p0["auc"]) - p0_metrics["row_auc"]
        ),
    }
    if max(replay.values()) > 1e-6:
        raise RuntimeError(f"S5P P0 metric replay failed: {replay}")

    normalization = checkpoint["normalization"]
    set_seed(args.seed)
    prototype = S5PNativeR4Residual(
        train_mean=float(normalization["mean"]),
        train_std=float(normalization["std"]),
        model_dim=int(args.model_dim),
        residual_cap=float(args.residual_cap),
    )
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in prototype.state_dict().items()
    }
    signature = parameter_signature(prototype)
    initial_sha = s5p_runner.state_dict_sha256(initial_state)
    del prototype
    shuffled_dev = shuffled_history_cache(
        dev, seed=args.seed + 700_001
    )
    cache_audit = {
        "train_cache": str(train_path),
        "train_cache_sha256": cache_runner.sha256_file(train_path),
        "dev_cache": str(dev_path),
        "dev_cache_sha256": cache_runner.sha256_file(dev_path),
        "train_rows": int(len(train["labels"])),
        "dev_rows": int(len(dev["labels"])),
        "train_events": int(len(set(map(str, train["event_ids"])))),
        "dev_events": int(len(set(map(str, dev["event_ids"])))),
        "event_overlap": 0,
        "roles": ["t0", "prev1", "prev2", "prev3", "seasonal", "year"],
        "feature_shape_per_row": [6, 3, 3],
        "train_valid_role_visits": int(
            train["valid_mask"].flatten(2).any(dim=-1).sum().item()
        ),
        "dev_valid_role_visits": int(
            dev["valid_mask"].flatten(2).any(dim=-1).sum().item()
        ),
        "native_grid_disclaimer": NATIVE_DISCLAIMER,
    }
    config = {
        "script_version": SCRIPT_VERSION,
        "args": vars(args),
        "cache_audit": cache_audit,
        "p0_checkpoint": str(p0_path),
        "p0_checkpoint_sha256": cache_runner.sha256_file(p0_path),
        "p0_replay": replay,
        "p0_exact_metrics": p0_metrics,
        "old_role_only_gate": {
            "three_seed_mean_row_ap": float(args.old_role_ap),
            "three_seed_mean_row_auc": float(args.old_role_auc),
        },
        "parameter_signature": signature,
        "initial_state_sha256": initial_sha,
        "test_or_sealed_or_holdout_read": False,
    }
    cache_runner.atomic_json_write(output_dir / "run_config.json", config)
    results = {}
    for arm in ARMS:
        results[arm] = run_arm(
            arm=arm,
            args=args,
            prototype_state=initial_state,
            signature=signature,
            initial_state_sha=initial_sha,
            normalization=normalization,
            train=train,
            dev=dev,
            shuffled_dev=shuffled_dev,
            train_base=train_base,
            dev_base=dev_base,
            p0_metrics=p0_metrics,
            output_dir=output_dir,
            device=device,
        )
    aggregate = {
        "script_version": SCRIPT_VERSION,
        "cache_audit": cache_audit,
        "p0_exact_metrics": p0_metrics,
        "p0_replay": replay,
        "arms": results,
        "promotion_gate": {
            arm: {
                "beats_old_role_row_ap": bool(
                    result["best"]["validation"]["row_ap"]
                    > float(args.old_role_ap)
                ),
                "beats_old_role_row_auc": bool(
                    result["best"]["validation"]["row_auc"]
                    > float(args.old_role_auc)
                ),
            }
            for arm, result in results.items()
        },
        "native_grid_disclaimer": NATIVE_DISCLAIMER,
        "test_or_sealed_or_holdout_read": False,
    }
    cache_runner.atomic_json_write(output_dir / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    root = "/diniuvol/yuyao/methanefuse_research_20260727"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cache",
        default=(
            f"{root}/cache/s5p_native_grid_experiment_v2/"
            "train_cc6ce7c3292d2faf0a7c.pt"
        ),
    )
    parser.add_argument(
        "--dev-cache",
        default=(
            f"{root}/cache/s5p_native_grid_experiment_v2/"
            "val_aa59e2709f5e1746b527.pt"
        ),
    )
    parser.add_argument(
        "--p0-checkpoint",
        default=(
            f"{root}/results/s5p_native_grid_v2_seed20260728_checkpoints/"
            "t0_masked_best.pt"
        ),
    )
    parser.add_argument(
        "--old-result-json",
        default=f"{root}/results/s5p_native_grid_v2_seed20260728.json",
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
    parser.add_argument("--old-role-ap", type=float, default=0.577494)
    parser.add_argument("--old-role-auc", type=float, default=0.539372)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
