#!/usr/bin/env python3
"""CPU-only train/dev temporal-statistics upper bound for clean L89 caches.

This exploratory runner deliberately has no test/holdout/sealed input.  It
fits a small fixed set of event-balanced linear/MLP heads above frozen
Panopticon features and evaluates a strict availability-matched history
shuffle for the best development model.
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
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (
    rctp_l89_event_balanced_head_followup as head_metrics,
)
from research.tempo_20260728 import tempo_l89_global as tempo


REFERENCE_P0 = {
    "event_balanced_ap": 0.7523292775401305,
    "event_balanced_auc": 0.841389,
    "event_balanced_macro_f1_selected": 0.765197,
    "event_balanced_positive_f1_selected": 0.686078,
}
FORBIDDEN_MARKERS = ("test", "holdout", "sealed", "outer")
METRIC_KEYS = (
    "event_balanced_ap",
    "event_balanced_auc",
    "event_balanced_macro_f1_selected",
    "event_balanced_positive_f1_selected",
    "all_negative_fp_mass",
    "selected_threshold",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def assert_train_dev_path(path: Path, *, purpose: str) -> Path:
    resolved = path.expanduser().resolve()
    lowered = [part.lower() for part in resolved.parts]
    if any(marker in part for part in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"{purpose} contains a forbidden held-out marker: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def trusted_cpu_load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Cache is not a mapping: {path}")
    return payload


def validate_cache_pair(
    train: Mapping[str, Any],
    dev: Mapping[str, Any],
    *,
    train_path: Path,
    dev_path: Path,
) -> dict[str, Any]:
    required = {
        "features",
        "labels",
        "valid_mask",
        "unique_mask",
        "delta_days",
        "valid_fraction",
        "ids",
        "plume_ids",
        "event_ids",
        "role_names",
        "t0_index",
        "weights_sha256",
    }
    for name, payload in (("train", train), ("dev", dev)):
        missing = sorted(required - set(payload))
        if missing:
            raise KeyError(f"{name} cache lacks {missing}.")
        if str(payload.get("split")) not in (
            ("train",) if name == "train" else ("val", "dev", "validation")
        ):
            raise ValueError(f"Unexpected {name} split={payload.get('split')!r}.")
        features = payload["features"]
        if tuple(features.shape[1:]) != (6, 768):
            raise ValueError(f"{name} feature shape differs from [N,6,768].")
        if int(payload["t0_index"]) != 0:
            raise ValueError(f"{name} t0 index is not zero.")
        if len(payload["labels"]) != len(features):
            raise ValueError(f"{name} labels/features row mismatch.")
        if not torch.isfinite(features.float()).all():
            raise ValueError(f"{name} contains non-finite frozen features.")
        if not bool(payload["valid_mask"][:, 0].all()):
            raise ValueError(f"{name} contains invalid t0 rows.")
        load_status = payload.get("load_status")
        if isinstance(load_status, Sequence) and not isinstance(load_status, str):
            failures = [value for value in load_status if str(value).lower() != "ok"]
            if failures:
                raise ValueError(f"{name} contains {len(failures)} read failures.")
    if list(train["role_names"]) != list(dev["role_names"]):
        raise ValueError("Train/dev role names differ.")
    train_events = set(str(value) for value in train["event_ids"])
    dev_events = set(str(value) for value in dev["event_ids"])
    overlap = sorted(train_events & dev_events)
    if overlap:
        raise ValueError(f"Train/dev canonical-event overlap: {overlap[:5]}.")
    if str(train["weights_sha256"]) != str(dev["weights_sha256"]):
        raise ValueError("Train/dev encoder PTH SHA differs.")
    return {
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "train_sha256": sha256_file(train_path),
        "dev_sha256": sha256_file(dev_path),
        "train_rows": int(len(train["labels"])),
        "dev_rows": int(len(dev["labels"])),
        "train_events": int(len(train_events)),
        "dev_events": int(len(dev_events)),
        "event_overlap": 0,
        "feature_shape": [6, 768],
        "t0_index": 0,
        "role_names": list(train["role_names"]),
        "weights_sha256": str(train["weights_sha256"]),
        "test_or_holdout_or_sealed_or_outer_read": False,
    }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_std(
    values: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    variance = (
        (values - mean.unsqueeze(1)).square() * weights
    ).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return torch.sqrt(variance.clamp_min(0.0) + 1e-8)


def build_features(
    payload: Mapping[str, Any],
    feature_kind: str,
) -> torch.Tensor:
    features = payload["features"].float()
    t0_index = int(payload["t0_index"])
    history = [index for index in range(features.shape[1]) if index != t0_index]
    z0 = features[:, t0_index]
    z_history = features[:, history]
    available = (
        payload["valid_mask"][:, history].bool()
        & payload["unique_mask"][:, history].bool()
    )
    signed_delta = z0.unsqueeze(1) - z_history
    signed_mean = masked_mean(signed_delta, available)
    absolute_mean = masked_mean(signed_delta.abs(), available)
    temporal_std = masked_std(signed_delta, available, signed_mean)

    if feature_kind == "t0":
        return z0.contiguous()
    if feature_kind == "motion4":
        return torch.cat(
            (z0, signed_mean, absolute_mean, temporal_std), dim=1
        ).contiguous()
    if feature_kind == "slowfast":
        role_names = list(payload["role_names"])
        fast_roles = ("prev1", "prev2", "prev3")
        slow_roles = ("seasonal", "year")
        fast_positions = [history.index(role_names.index(role)) for role in fast_roles]
        slow_positions = [history.index(role_names.index(role)) for role in slow_roles]
        fast_delta = signed_delta[:, fast_positions]
        fast_mask = available[:, fast_positions]
        slow_delta = signed_delta[:, slow_positions]
        slow_mask = available[:, slow_positions]
        fast_signed = masked_mean(fast_delta, fast_mask)
        fast_absolute = masked_mean(fast_delta.abs(), fast_mask)
        slow_signed = masked_mean(slow_delta, slow_mask)
        slow_absolute = masked_mean(slow_delta.abs(), slow_mask)
        availability_summary = torch.stack(
            (
                fast_mask.float().mean(dim=1),
                slow_mask.float().mean(dim=1),
                available.float().mean(dim=1),
            ),
            dim=1,
        )
        return torch.cat(
            (
                z0,
                fast_signed,
                fast_absolute,
                slow_signed,
                slow_absolute,
                temporal_std,
                availability_summary,
            ),
            dim=1,
        ).contiguous()
    if feature_kind == "rolewise_delta":
        masked_delta = signed_delta * available.float().unsqueeze(-1)
        gap = torch.log1p(payload["delta_days"][:, history].abs().float())
        quality = payload["valid_fraction"][:, history].float()
        metadata = torch.cat(
            (available.float(), gap * available.float(), quality * available.float()),
            dim=1,
        )
        return torch.cat(
            (z0, masked_delta.flatten(start_dim=1), metadata), dim=1
        ).contiguous()
    raise ValueError(f"Unknown feature kind {feature_kind!r}.")


class BinaryHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        kind: str,
        width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kind == "linear":
            self.network = nn.Linear(input_dim, 1)
        elif kind == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.GELU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout),
                nn.Linear(width, 1),
            )
        else:
            raise ValueError(f"Unknown head kind {kind!r}.")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def predict(
    model: nn.Module,
    features: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in torch.arange(len(features)).split(batch_size):
            output.append(torch.sigmoid(model(features[batch])).cpu())
    probability = torch.cat(output).numpy().astype(np.float64)
    if not np.isfinite(probability).all():
        raise RuntimeError("Model produced non-finite probabilities.")
    return probability


def selected_metrics(
    labels: torch.Tensor,
    probabilities: np.ndarray,
    event_ids: Sequence[str],
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    metrics = tempo.metric_bundle(
        labels.long().numpy().astype(np.int64),
        probabilities,
        [str(value) for value in event_ids],
        threshold=threshold,
    )
    return {key: metrics[key] for key in metrics}


def train_model(
    *,
    train_x: torch.Tensor,
    dev_x: torch.Tensor,
    train: Mapping[str, Any],
    dev: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = BinaryHead(
        int(train_x.shape[1]),
        kind=str(config["head"]),
        width=int(config.get("width", 0)),
        dropout=float(config.get("dropout", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    row_weight = torch.from_numpy(
        head_metrics.mean_one_event_weights(train["event_ids"])
    ).float()
    positive_weight = torch.tensor(
        head_metrics.event_balanced_positive_weight(
            train["labels"].long().numpy(), train["event_ids"]
        ),
        dtype=torch.float32,
    )
    labels = train["labels"].float()
    best: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    no_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + 104_729 * epoch)
        batches = torch.randperm(len(labels), generator=generator).split(
            int(config["batch_size"])
        )
        loss_sum = 0.0
        row_sum = 0
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x[indices])
            per_row = F.binary_cross_entropy_with_logits(
                logits,
                labels[indices],
                pos_weight=positive_weight,
                reduction="none",
            )
            loss = (per_row * row_weight[indices]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{config['name']} produced non-finite loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss) * len(indices)
            row_sum += len(indices)
        probability = predict(
            model, dev_x, batch_size=int(config["eval_batch_size"])
        )
        metrics = selected_metrics(
            dev["labels"], probability, dev["event_ids"]
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(row_sum, 1),
            "dev": {key: metrics[key] for key in METRIC_KEYS},
        }
        history.append(record)
        if (
            best is None
            or float(metrics["event_balanced_ap"])
            > float(best["metrics"]["event_balanced_ap"]) + 1e-8
        ):
            best = {
                "epoch": epoch,
                "metrics": metrics,
                "probability": probability,
            }
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= int(config["patience"]):
            break
    if best is None or best_state is None:
        raise RuntimeError(f"{config['name']} produced no valid epoch.")
    elapsed = time.perf_counter() - started
    result = {
        "name": str(config["name"]),
        "feature_kind": str(config["feature_kind"]),
        "head": str(config["head"]),
        "width": int(config.get("width", 0)),
        "dropout": float(config.get("dropout", 0.0)),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "batch_size": int(config["batch_size"]),
        "max_epochs": int(config["epochs"]),
        "patience": int(config["patience"]),
        "seed": int(seed),
        "input_dim": int(train_x.shape[1]),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "epochs_observed": len(history),
        "best_epoch": int(best["epoch"]),
        "best_dev": {key: best["metrics"][key] for key in METRIC_KEYS},
        "delta_vs_reference_p0": {
            key: float(best["metrics"][key] - REFERENCE_P0[key])
            for key in (
                "event_balanced_ap",
                "event_balanced_auc",
                "event_balanced_macro_f1_selected",
                "event_balanced_positive_f1_selected",
            )
        },
        "history": history,
        "runtime_seconds": elapsed,
    }
    state = {
        "model": best_state,
        "probability": torch.from_numpy(best["probability"]),
    }
    return result, state


def normalized_feature_pair(
    train: Mapping[str, Any],
    dev: Mapping[str, Any],
    feature_kind: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor]:
    train_raw = build_features(train, feature_kind)
    dev_raw = build_features(dev, feature_kind)
    mean = train_raw.mean(dim=0)
    std = train_raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    train_x = ((train_raw - mean) / std).clamp(-10.0, 10.0)
    dev_x = ((dev_raw - mean) / std).clamp(-10.0, 10.0)
    if not torch.isfinite(train_x).all() or not torch.isfinite(dev_x).all():
        raise RuntimeError(f"{feature_kind} normalization produced non-finite data.")
    audit = {
        "feature_kind": feature_kind,
        "input_dim": int(train_x.shape[1]),
        "train_rows": int(train_x.shape[0]),
        "dev_rows": int(dev_x.shape[0]),
        "normalization": "train-dimension mean/std; std floor=1e-4; clip=[-10,10]",
        "train_mean_sha256": tensor_sha256(mean),
        "train_std_sha256": tensor_sha256(std),
    }
    return train_x, dev_x, audit, mean, std


def minimal_shuffle_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "features": payload["features"],
        "labels": payload["labels"],
        "valid_mask": payload["valid_mask"],
        "unique_mask": payload["unique_mask"],
        "delta_days": payload["delta_days"],
        "valid_fraction": payload["valid_fraction"],
        "event_ids": list(payload["event_ids"]),
        "ids": list(payload["ids"]),
        "plume_ids": list(payload["plume_ids"]),
        "role_names": list(payload["role_names"]),
    }


def history_shuffle_audit(
    *,
    dev: Mapping[str, Any],
    best_result: Mapping[str, Any],
    best_state: Mapping[str, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    shuffled, validity = tempo.shuffled_development_view(
        minimal_shuffle_payload(dev),
        seed=seed,
        t0_index=int(dev["t0_index"]),
        return_audit=True,
    )
    shuffled["t0_index"] = int(dev["t0_index"])
    shuffled_raw = build_features(shuffled, str(best_result["feature_kind"]))
    shuffled_x = ((shuffled_raw - mean) / std).clamp(-10.0, 10.0)
    model = BinaryHead(
        int(best_result["input_dim"]),
        kind=str(best_result["head"]),
        width=int(best_result["width"]),
        dropout=float(best_result["dropout"]),
    )
    model.load_state_dict(best_state, strict=True)
    probability = predict(
        model,
        shuffled_x,
        batch_size=int(best_result["batch_size"]) * 2,
    )
    coherent = dict(best_result["best_dev"])
    shuffled_metrics = selected_metrics(
        dev["labels"],
        probability,
        dev["event_ids"],
        threshold=float(coherent["selected_threshold"]),
    )
    metric_names = (
        "event_balanced_ap",
        "event_balanced_auc",
        "event_balanced_macro_f1_selected",
        "event_balanced_positive_f1_selected",
        "all_negative_fp_mass",
    )
    return {
        "seed": int(seed),
        "fixed_model": True,
        "fixed_coherent_threshold": True,
        "validity": validity,
        "coherent": {key: coherent[key] for key in metric_names},
        "shuffled": {key: shuffled_metrics[key] for key in metric_names},
        "shuffle_minus_coherent": {
            key: float(shuffled_metrics[key] - coherent[key])
            for key in metric_names
        },
    }


def write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "RESULT.json"
    temporary = output_dir / "RESULT.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(json_path)

    lines = [
        "# L89 clean CPU temporal upper-bound",
        "",
        "Scope: CPU-only, frozen train/dev features, no test/holdout/sealed/outer read.",
        "",
        "## Result",
        "",
        "| Model | Features | Head | Epoch | AP | ΔAP vs P0 | AUC | Macro-F1 | ΔMacro-F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["models"]:
        metrics = result["best_dev"]
        delta = result["delta_vs_reference_p0"]
        lines.append(
            "| {name} | {features} | {head} | {epoch} | {ap:.6f} | "
            "{dap:+.6f} | {auc:.6f} | {macro:.6f} | {dmacro:+.6f} |".format(
                name=result["name"],
                features=result["feature_kind"],
                head=result["head"],
                epoch=result["best_epoch"],
                ap=metrics["event_balanced_ap"],
                dap=delta["event_balanced_ap"],
                auc=metrics["event_balanced_auc"],
                macro=metrics["event_balanced_macro_f1_selected"],
                dmacro=delta["event_balanced_macro_f1_selected"],
            )
        )
    best = payload["selection"]
    shuffle = payload["history_shuffle"]
    lines.extend(
        [
            "",
            "## Best model",
            "",
            f"- Selected by development event-balanced AP: `{best['best_model']}`.",
            f"- AP: `{best['best_metrics']['event_balanced_ap']:.6f}` "
            f"(Δ vs P0 `{best['delta_vs_reference_p0']['event_balanced_ap']:+.6f}`).",
            f"- Macro-F1: `{best['best_metrics']['event_balanced_macro_f1_selected']:.6f}` "
            f"(Δ vs P0 `{best['delta_vs_reference_p0']['event_balanced_macro_f1_selected']:+.6f}`).",
            f"- Material AP upper-bound reached (ΔAP ≥ 0.005): "
            f"`{str(best['material_ap_gain']).lower()}`.",
            "",
            "## Fixed-model history shuffle",
            "",
            f"- Availability-stratified shuffle valid: "
            f"`{str(shuffle['validity']['valid']).lower()}`.",
            f"- AP change: "
            f"`{shuffle['shuffle_minus_coherent']['event_balanced_ap']:+.6f}`.",
            f"- Macro-F1 change at the coherent fixed threshold: "
            f"`{shuffle['shuffle_minus_coherent']['event_balanced_macro_f1_selected']:+.6f}`.",
            "",
            "## Interpretation",
            "",
            payload["verdict"]["summary"],
            "",
            "This is a same-development exploratory ceiling, not an outer result or a SOTA claim.",
        ]
    )
    md_path = output_dir / "RESULT.md"
    temporary_md = output_dir / "RESULT.md.tmp"
    temporary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_md.replace(md_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.perf_counter()
    train_path = assert_train_dev_path(Path(args.train_cache), purpose="train cache")
    dev_path = assert_train_dev_path(Path(args.dev_cache), purpose="dev cache")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if any(marker in part.lower() for part in output_dir.parts for marker in FORBIDDEN_MARKERS):
        raise ValueError("Output directory contains a held-out marker.")
    train = trusted_cpu_load(train_path)
    dev = trusted_cpu_load(dev_path)
    input_audit = validate_cache_pair(
        train,
        dev,
        train_path=train_path,
        dev_path=dev_path,
    )

    configs = (
        {
            "name": "t0_linear",
            "feature_kind": "t0",
            "head": "linear",
            "width": 0,
            "dropout": 0.0,
            "learning_rate": 1.5e-3,
            "weight_decay": 1e-3,
            "epochs": 6,
            "patience": 1,
            "batch_size": 512,
            "eval_batch_size": 2048,
        },
        {
            "name": "motion4_linear",
            "feature_kind": "motion4",
            "head": "linear",
            "width": 0,
            "dropout": 0.0,
            "learning_rate": 1.5e-3,
            "weight_decay": 1e-3,
            "epochs": 6,
            "patience": 1,
            "batch_size": 512,
            "eval_batch_size": 2048,
        },
        {
            "name": "motion4_mlp128",
            "feature_kind": "motion4",
            "head": "mlp",
            "width": 128,
            "dropout": 0.15,
            "learning_rate": 5e-4,
            "weight_decay": 2e-2,
            "epochs": 6,
            "patience": 1,
            "batch_size": 512,
            "eval_batch_size": 2048,
        },
        {
            "name": "slowfast_mlp96",
            "feature_kind": "slowfast",
            "head": "mlp",
            "width": 96,
            "dropout": 0.15,
            "learning_rate": 5e-4,
            "weight_decay": 2e-2,
            "epochs": 6,
            "patience": 1,
            "batch_size": 512,
            "eval_batch_size": 2048,
        },
        {
            "name": "rolewise_delta_mlp96",
            "feature_kind": "rolewise_delta",
            "head": "mlp",
            "width": 96,
            "dropout": 0.15,
            "learning_rate": 5e-4,
            "weight_decay": 2e-2,
            "epochs": 6,
            "patience": 1,
            "batch_size": 512,
            "eval_batch_size": 2048,
        },
    )
    feature_cache: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor]] = {}
    results: list[dict[str, Any]] = []
    states: dict[str, dict[str, torch.Tensor]] = {}
    normalizers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    feature_audits: dict[str, Mapping[str, Any]] = {}
    for config in configs:
        feature_kind = str(config["feature_kind"])
        if feature_kind not in feature_cache:
            feature_cache[feature_kind] = normalized_feature_pair(
                train, dev, feature_kind
            )
        train_x, dev_x, feature_audit, mean, std = feature_cache[feature_kind]
        feature_audits[feature_kind] = feature_audit
        result, state = train_model(
            train_x=train_x,
            dev_x=dev_x,
            train=train,
            dev=dev,
            config=config,
            seed=int(args.seed),
        )
        results.append(result)
        states[result["name"]] = state
        normalizers[result["name"]] = (mean, std)
        print(
            f"[cpu-upperbound] {result['name']} best_epoch={result['best_epoch']} "
            f"AP={result['best_dev']['event_balanced_ap']:.6f} "
            f"macro={result['best_dev']['event_balanced_macro_f1_selected']:.6f}",
            flush=True,
        )
        del feature_cache[feature_kind]

    best = max(
        results,
        key=lambda value: (
            float(value["best_dev"]["event_balanced_ap"]),
            float(value["best_dev"]["event_balanced_macro_f1_selected"]),
        ),
    )
    mean, std = normalizers[best["name"]]
    shuffle = history_shuffle_audit(
        dev=dev,
        best_result=best,
        best_state=states[best["name"]]["model"],
        mean=mean,
        std=std,
        seed=int(args.seed) + 8_191,
    )
    delta_ap = float(best["delta_vs_reference_p0"]["event_balanced_ap"])
    delta_macro = float(
        best["delta_vs_reference_p0"]["event_balanced_macro_f1_selected"]
    )
    material_ap = delta_ap >= 0.005
    material_macro = delta_macro >= 0.005
    if material_ap:
        summary = (
            "The explicit video-style temporal statistics form a material "
            "same-development AP upper bound over fresh P0. This warrants a "
            "controlled neural implementation, but still requires outer confirmation."
        )
    elif material_macro:
        summary = (
            "The screen did not materially improve ranking AP, but it found a "
            "material operating-point macro-F1 gain. The evidence supports a "
            "classification/F1 story rather than an AP/SOTA-ranking claim."
        )
    else:
        summary = (
            "No tested explicit video-style temporal statistic materially exceeded "
            "fresh P0. Frozen 6×768 features do not expose a strong cheap temporal "
            "upper bound on this split; further large CPU grids are not justified."
        )
    payload = {
        "schema_version": "l89-clean-cpu-temporal-upperbound-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "started_utc": started_utc,
        "scope": {
            "device": "cpu",
            "maximum_cpu_threads": int(args.threads),
            "train_dev_only": True,
            "same_event_disjoint_split": True,
            "frozen_features": True,
            "test_or_holdout_or_sealed_or_outer_read": False,
            "formal_source_modified": False,
            "exploratory_same_dev_model_selection": True,
        },
        "input_audit": input_audit,
        "reference_p0": REFERENCE_P0,
        "optimization": {
            "selection": "best epoch by development event-balanced AP",
            "early_stop_patience": 1,
            "maximum_epochs": 6,
            "loss": "event-balanced weighted BCE with event-balanced positive weight",
            "normalization": "train-only feature mean/std",
            "configuration_count": len(configs),
            "large_grid_or_sweep": False,
        },
        "feature_audits": feature_audits,
        "models": results,
        "selection": {
            "best_model": best["name"],
            "best_metrics": best["best_dev"],
            "delta_vs_reference_p0": best["delta_vs_reference_p0"],
            "material_ap_gain": material_ap,
            "material_macro_f1_gain": material_macro,
            "material_gain_threshold": 0.005,
        },
        "history_shuffle": shuffle,
        "verdict": {
            "materially_exceeds_p0_ap": material_ap,
            "materially_exceeds_p0_macro_f1": material_macro,
            "summary": summary,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    write_outputs(output_dir, payload)
    print(json.dumps(payload["selection"], indent=2, sort_keys=True), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cache",
        default="/tmp/l89_clean_exploratory_base_v1/train.pt",
    )
    parser.add_argument(
        "--dev-cache",
        default="/tmp/l89_clean_exploratory_base_v1/val.pt",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/yuyao/panopticon/research/tempo_20260728/"
            "l89_clean_cpu_temporal_upperbound_v1"
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= int(args.threads) <= 8:
        raise ValueError("--threads must be in [1,8].")
    torch.set_num_threads(int(args.threads))
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    run(args)


if __name__ == "__main__":
    main()
