#!/usr/bin/env python3
"""Development-only L89 TEMPO global-onset pilot.

This runner consumes only the frozen, event-disjoint L89 ``train`` and ``val``
CLS caches created by ``l89_ragged_cls_experiment.py``.  It has no test
subcommand and refuses paths containing test/sealed/holdout-like components.

The experiment has two stages:

1. fit one t0-only base classifier and freeze it;
2. train zero-initialized temporal residuals on top of the exact same frozen
   base logit.

The matched temporal arms are:

``p1_delta``
    current-to-history feature differences, uniformly pooled;
``p2_onset``
    current-to-history differences minus history-to-history normal variation;
``p3_gated_onset``
    P2 with learned real-gap/role/quality/relevance gating;
``p4_null_onset``
    P3 plus an all-negative-event top-k false-alarm objective.
``d1_gated_delta``
    P1 with the same learned gap/role/quality/relevance gate as P3;
``d2_null_delta``
    P1 plus the all-negative-event top-k objective;
``d3_gated_null_delta``
    D1 plus the all-negative-event top-k objective.
``d7_sparse_slowfast``
    Capacity-matched sparse SlowFast control: independently gate
    ``prev1/prev2/prev3`` and ``seasonal/year`` deltas, then combine the two
    branch aggregates with a fixed availability-normalized sum.

The script also evaluates a deterministic cross-event history shuffle without
changing t0, labels, the fitted model, or the selected threshold.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as cache_runner,
)
from research.pretraining_20260727 import (  # noqa: E402
    rctp_l89_event_balanced_head_followup as event_metrics,
)


SCRIPT_VERSION = "tempo-l89-global-v1"
ARM_NAMES = (
    "p0_base",
    "p1_delta",
    "p2_onset",
    "p3_gated_onset",
    "p4_null_onset",
    "d1_gated_delta",
    "d2_null_delta",
    "d3_gated_null_delta",
    "d6_onset_only",
    "d7_sparse_slowfast",
)
TEMPORAL_ARMS = ARM_NAMES[1:]
DEVELOPMENT_FORBIDDEN_RE = re.compile(
    r"(^|[._-])(test|sealed|holdout)([._-]|$)", re.IGNORECASE
)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return cache_runner.state_dict_sha256(state)


def atomic_json_write(path: Path, value: Any) -> None:
    cache_runner.atomic_json_write(path, value)


def atomic_torch_save(path: Path, value: Any) -> None:
    cache_runner.atomic_torch_save(path, value)


def atomic_csv_write(path: Path, frame: pd.DataFrame) -> None:
    cache_runner.atomic_csv_write(path, frame)


def assert_development_path(path: Path, *, purpose: str) -> None:
    resolved = path.expanduser().resolve()
    cache_runner.assert_not_sealed_path(resolved, purpose=purpose)
    offending = [part for part in resolved.parts if DEVELOPMENT_FORBIDDEN_RE.search(part)]
    if offending:
        raise ValueError(
            f"{purpose} path is not a development artifact: {resolved}; "
            f"offending components={offending}"
        )


def parse_names(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        names = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        names = tuple(str(part).strip() for part in value if str(part).strip())
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Expected unique non-empty names, got {names}")
    return names


def parse_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        values = tuple(int(part) for part in value)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"Expected unique integers, got {values}")
    return values


def take_development_rows(payload: Mapping[str, Any]) -> dict[str, Any]:
    indices = cache_runner.select_usable_rows(payload)
    index_list = indices.tolist()
    valid_fraction = payload.get("valid_fraction")
    if valid_fraction is None:
        valid_fraction = payload["valid_mask"].float()
    return {
        "features": payload["features"][indices].float().contiguous(),
        "labels": payload["labels"][indices].float().contiguous(),
        "valid_mask": payload["valid_mask"][indices].bool().contiguous(),
        "unique_mask": payload["unique_mask"][indices].bool().contiguous(),
        "delta_days": payload["delta_days"][indices].float().contiguous(),
        "valid_fraction": valid_fraction[indices].float().contiguous(),
        "ids": [str(payload["ids"][index]) for index in index_list],
        "plume_ids": [str(payload["plume_ids"][index]) for index in index_list],
        "event_ids": [str(payload["event_ids"][index]) for index in index_list],
    }


def load_development_pair(
    train_path: Path, dev_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], torch.Tensor]:
    assert_development_path(train_path, purpose="train cache")
    assert_development_path(dev_path, purpose="development cache")
    train_cache, dev_cache, audit = cache_runner.load_cache_pair(train_path, dev_path)
    if train_cache["features"].shape[1] != 6:
        raise ValueError("TEMPO L89 pilot requires the audited six-role cache.")
    train = take_development_rows(train_cache)
    dev = take_development_rows(dev_cache)
    role_index = train_cache["role_index"].long()
    audit = {
        **audit,
        "train_rows_usable": int(train["labels"].numel()),
        "dev_rows_usable": int(dev["labels"].numel()),
        "train_events": int(len(set(train["event_ids"]))),
        "dev_events": int(len(set(dev["event_ids"]))),
        "train_dev_event_overlap": int(
            len(set(train["event_ids"]) & set(dev["event_ids"]))
        ),
        "test_or_sealed_or_holdout_read": False,
    }
    if audit["train_dev_event_overlap"]:
        raise ValueError("Train/development canonical events overlap.")
    return train, dev, audit, role_index


def audit_reference_cache_alignment(
    sidecar_rows: Mapping[str, Any],
    reference_path: Path,
    *,
    expected_split: str,
) -> dict[str, Any]:
    """Prove a sidecar cache is a row-exact extension of the 768-D cache."""

    assert_development_path(reference_path, purpose="identity reference cache")
    reference = cache_runner.torch_load_trusted(reference_path)
    if not isinstance(reference, Mapping):
        raise ValueError(f"{reference_path} does not contain a cache mapping.")
    cache_runner.validate_cache_payload(
        reference, path=reference_path, expected_split=expected_split
    )
    indices = cache_runner.select_usable_rows(reference)
    index_list = indices.tolist()
    expected_rows = int(indices.numel())
    if len(sidecar_rows["ids"]) != expected_rows:
        raise ValueError(
            f"{expected_split} sidecar/reference usable row counts differ."
        )
    exact_sequence_fields = {
        "ids": [str(reference["ids"][index]) for index in index_list],
        "plume_ids": [
            str(reference["plume_ids"][index]) for index in index_list
        ],
        "event_ids": [
            str(reference["event_ids"][index]) for index in index_list
        ],
    }
    for field, expected in exact_sequence_fields.items():
        if list(sidecar_rows[field]) != expected:
            raise ValueError(
                f"{expected_split} sidecar/reference {field} ordering differs."
            )
    if not torch.equal(
        sidecar_rows["labels"].long(), reference["labels"][indices].long()
    ):
        raise ValueError(f"{expected_split} sidecar/reference labels differ.")
    for field in ("unique_mask", "delta_days", "valid_fraction"):
        reference_value = reference.get(field)
        if reference_value is None and field == "valid_fraction":
            reference_value = reference["valid_mask"].float()
        observed = sidecar_rows[field]
        expected = reference_value[indices].to(dtype=observed.dtype)
        if not torch.equal(observed, expected):
            raise ValueError(
                f"{expected_split} sidecar/reference {field} differs."
            )
    reference_dim = int(reference["features"].shape[-1])
    if int(sidecar_rows["features"].shape[-1]) <= reference_dim:
        raise ValueError("Sidecar cache does not extend the reference width.")
    # Chunked comparison avoids materializing another full float32 cache.
    for row_indices in torch.arange(expected_rows).split(512):
        observed = sidecar_rows["features"][row_indices, :, :reference_dim]
        expected = reference["features"][indices[row_indices]].to(
            dtype=observed.dtype
        )
        if not torch.equal(observed, expected):
            raise ValueError(
                f"{expected_split} sidecar/reference first {reference_dim} "
                "feature dimensions differ."
            )
    return {
        "split": expected_split,
        "reference_cache": str(reference_path),
        "reference_cache_sha256": cache_runner.sha256_file(reference_path),
        "reference_feature_sha256": str(reference["feature_sha256"]),
        "reference_label_sha256": str(reference.get("label_sha256", "")),
        "reference_timestamp_sha256": str(
            reference.get("timestamp_sha256", "")
        ),
        "reference_input_table_sha256": str(
            reference.get("input_table_sha256", "")
        ),
        "rows": expected_rows,
        "reference_feature_dim": reference_dim,
        "id_label_event_order_exact": True,
        "first_768_feature_block_exact": reference_dim == 768,
        "mask_gap_quality_exact": True,
        "test_or_sealed_or_holdout_read": False,
    }


def event_codes(event_ids: Sequence[str]) -> tuple[torch.Tensor, list[str]]:
    ordered = sorted(set(str(value) for value in event_ids))
    lookup = {value: index for index, value in enumerate(ordered)}
    return torch.tensor([lookup[str(value)] for value in event_ids]), ordered


def all_negative_event_mask(
    labels: torch.Tensor, codes: torch.Tensor, event_names: Sequence[str]
) -> torch.Tensor:
    if labels.shape != codes.shape:
        raise ValueError("labels and event codes must share shape.")
    positive_by_event = torch.zeros(len(event_names), dtype=torch.bool)
    positive_codes = codes[labels.long().eq(1)]
    if positive_codes.numel():
        positive_by_event[positive_codes.unique()] = True
    return ~positive_by_event


class T0BaseHead(nn.Module):
    """Small supervised classifier on the frozen t0 Panopticon CLS token."""

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def hidden(self, t0_features: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.projection(self.norm(t0_features)))

    def forward(self, t0_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.hidden(t0_features))).squeeze(-1)


class FrozenRoleOnlyBase(nn.Module):
    """Exact wrapper around an audited frozen temporal-head checkpoint."""

    def __init__(self, head: cache_runner.RaggedCurrentQueryHead):
        super().__init__()
        self.head = head.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> "FrozenRoleOnlyBase":
        super().train(mode)
        self.head.eval()
        return self

    def forward(
        self,
        features: torch.Tensor,
        unique_mask: torch.Tensor,
        delta_days: torch.Tensor,
        role_index: torch.Tensor,
    ) -> torch.Tensor:
        expected_dim = int(self.head.input_projection.in_features)
        if features.shape[-1] != expected_dim:
            if expected_dim == 2 * int(features.shape[-1]):
                # The event-balanced P0 sidecar cache is exactly
                # [Panopticon CLS, zero residual].  Construct it without
                # reading a second representation cache.
                features = torch.cat((features, torch.zeros_like(features)), dim=-1)
            else:
                raise ValueError(
                    f"Frozen base expects D={expected_dim}, got D={features.shape[-1]}."
                )
        # Keep the old role-only intervention exact: real historical content
        # and discrete roles, but the continuous-delta contribution is zero.
        return self.head(
            features,
            unique_mask,
            role_index,
            torch.zeros_like(delta_days),
            enable_delta=False,
        )


class ZeroLogitBase(nn.Module):
    """Appearance-free base for an independently trained transient expert."""

    def forward(
        self,
        features: torch.Tensor,
        unique_mask: torch.Tensor,
        delta_days: torch.Tensor,
        role_index: torch.Tensor,
    ) -> torch.Tensor:
        del unique_mask, delta_days, role_index
        return torch.zeros(
            features.shape[0], device=features.device, dtype=features.dtype
        )


def trusted_load_old_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load the local legacy checkpoint whose args pickle ``__main__.train_heads``."""

    assert_development_path(path, purpose="frozen role-only checkpoint")
    main_module = sys.modules.get("__main__")
    if main_module is None:
        raise RuntimeError("Python __main__ module is unavailable.")
    previous = getattr(main_module, "train_heads", None)
    setattr(main_module, "train_heads", cache_runner.train_heads)
    try:
        payload = cache_runner.torch_load_trusted(path)
    finally:
        if previous is None:
            with suppress(AttributeError):
                delattr(main_module, "train_heads")
        else:
            setattr(main_module, "train_heads", previous)
    if not isinstance(payload, Mapping):
        raise TypeError("Frozen role-only checkpoint must be a mapping.")
    return payload


def load_frozen_role_only_base(
    checkpoint_path: Path,
    *,
    train_cache_path: Path,
    dev_cache_path: Path,
    dev: Mapping[str, Any],
    role_index: torch.Tensor,
    device: torch.device,
    eval_batch_size: int,
) -> tuple[FrozenRoleOnlyBase, dict[str, Any], np.ndarray]:
    checkpoint = trusted_load_old_checkpoint(checkpoint_path)
    if checkpoint.get("arm") != "role_only":
        raise ValueError(
            f"Expected role_only checkpoint, got {checkpoint.get('arm')!r}."
        )
    config = checkpoint.get("args")
    if not isinstance(config, Mapping):
        raise ValueError("Frozen checkpoint is missing its training configuration.")
    expected_train = Path(str(config["train_cache"])).expanduser().resolve()
    expected_dev = Path(str(config["val_cache"])).expanduser().resolve()
    if expected_train != train_cache_path or expected_dev != dev_cache_path:
        raise ValueError(
            "Frozen role-only checkpoint does not reference the requested cache pair."
        )
    periods = tuple(float(value) for value in str(config["delta_periods"]).split(","))
    feature_dim = int(dev["features"].shape[-1])
    timepoints = int(dev["features"].shape[1])
    t0_index = int(checkpoint["cache_audit"]["t0_index"])
    head = cache_runner.RaggedCurrentQueryHead(
        feature_dim=feature_dim,
        num_roles=timepoints,
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        depth=2,
        mlp_ratio=float(config["mlp_ratio"]),
        dropout=float(config["dropout"]),
        periods_days=periods,
        t0_index=t0_index,
    )
    head.load_state_dict(checkpoint["model"], strict=True)
    base = FrozenRoleOnlyBase(head).to(device)
    wrapper = RoleBasePredictionWrapper(base).to(device)
    probabilities = predict(
        wrapper,
        dev,
        role_index,
        arm="p0_base",
        batch_size=eval_batch_size,
        device=device,
    )
    prediction_path = checkpoint_path.parent / "validation_best_ap_predictions.csv"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    old = pd.read_csv(prediction_path)
    expected_columns = {"id", "event_id", "label", "probability"}
    if not expected_columns.issubset(old.columns):
        raise ValueError("Historical prediction CSV is missing required columns.")
    if old["id"].astype(str).tolist() != list(dev["ids"]):
        raise ValueError("Historical prediction rows differ from the current dev cache.")
    old_probability = old["probability"].to_numpy(dtype=np.float64)
    maximum_error = float(np.max(np.abs(old_probability - probabilities)))
    # Historical probabilities were produced on GPU and serialized through
    # decimal CSV; CPU replay can differ by a few float32 ulps.
    replay_tolerance = 1e-6
    if maximum_error > replay_tolerance:
        raise RuntimeError(
            "Frozen role-only replay is not exact enough: "
            f"maximum probability error={maximum_error:.3e}."
        )
    audit = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": cache_runner.sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation": dict(checkpoint["validation"]),
        "prediction_csv": str(prediction_path),
        "prediction_csv_sha256": cache_runner.sha256_file(prediction_path),
        "replay_max_abs_probability_error": maximum_error,
        "replay_tolerance": replay_tolerance,
        "replay_numerically_exact": True,
        "model_state_sha256": state_dict_sha256(checkpoint["model"]),
        "test_or_sealed_or_holdout_read": False,
    }
    return base.cpu(), audit, probabilities


def load_event_balanced_base(
    checkpoint_path: Path,
    *,
    dev: Mapping[str, Any],
    role_index: torch.Tensor,
    device: torch.device,
    eval_batch_size: int,
    expected_arm: str,
    base_kind: str,
) -> tuple[FrozenRoleOnlyBase, dict[str, Any], np.ndarray]:
    """Load an exact event-balanced P0/P5 head as a frozen residual base."""

    assert_development_path(
        checkpoint_path, purpose=f"frozen event-balanced {expected_arm} checkpoint"
    )
    checkpoint = cache_runner.torch_load_trusted(checkpoint_path)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("arm") != expected_arm:
        raise ValueError(
            f"Expected a frozen event-balanced {expected_arm} checkpoint."
        )
    run_config_path = checkpoint_path.parents[1] / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(run_config_path)
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    feature_dim = int(checkpoint["cache_audit"]["feature_dim"])
    development_dim = int(dev["features"].shape[-1])
    if expected_arm == "p0":
        if feature_dim != 2 * development_dim:
            raise ValueError(
                "Event-balanced P0 does not have [base, zero] feature width."
            )
        feature_contract = "[original_Panopticon_768, exact_zero_768]"
        response_sidecar_used = False
    elif expected_arm == "p5":
        if feature_dim != development_dim:
            raise ValueError(
                "Event-balanced P5 checkpoint width differs from the sidecar cache."
            )
        feature_contract = (
            "[original_Panopticon_768, response_sidecar_768]"
        )
        response_sidecar_used = True
    else:
        raise ValueError(f"Unsupported event-balanced base arm {expected_arm!r}.")
    periods = tuple(
        float(value)
        for value in str(run_config.get("delta_periods", "1,3,7,30,90,365")).split(
            ","
        )
    )
    head = cache_runner.RaggedCurrentQueryHead(
        feature_dim=feature_dim,
        num_roles=int(dev["features"].shape[1]),
        model_dim=int(run_config["model_dim"]),
        num_heads=int(run_config["num_heads"]),
        depth=2,
        mlp_ratio=float(run_config["mlp_ratio"]),
        dropout=float(run_config["dropout"]),
        periods_days=periods,
        t0_index=int(checkpoint["cache_audit"]["t0_index"]),
    )
    head.load_state_dict(checkpoint["model"], strict=True)
    base = FrozenRoleOnlyBase(head).to(device)
    probabilities = predict(
        RoleBasePredictionWrapper(base).to(device),
        dev,
        role_index,
        arm="p0_base",
        batch_size=eval_batch_size,
        device=device,
    )
    prediction_path = (
        checkpoint_path.parent / "validation_best_event_balanced_ap_predictions.csv"
    )
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    old = pd.read_csv(prediction_path)
    if old["id"].astype(str).tolist() != list(dev["ids"]):
        raise ValueError("Event-balanced P0 prediction rows differ from dev cache.")
    old_probability = old["probability"].to_numpy(dtype=np.float64)
    maximum_error = float(np.max(np.abs(old_probability - probabilities)))
    replay_tolerance = 1e-6
    if maximum_error > replay_tolerance:
        raise RuntimeError(
            "Frozen event-balanced P0 replay failed: "
            f"maximum probability error={maximum_error:.3e}."
        )
    audit = {
        "base_kind": base_kind,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": cache_runner.sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation": dict(checkpoint["validation"]),
        "prediction_csv": str(prediction_path),
        "prediction_csv_sha256": cache_runner.sha256_file(prediction_path),
        "replay_max_abs_probability_error": maximum_error,
        "replay_tolerance": replay_tolerance,
        "replay_numerically_exact": True,
        "model_state_sha256": state_dict_sha256(checkpoint["model"]),
        "base_feature_contract": feature_contract,
        "response_sidecar_used": response_sidecar_used,
        "test_or_sealed_or_holdout_read": False,
    }
    return base.cpu(), audit, probabilities


def load_event_balanced_p0_base(
    checkpoint_path: Path,
    *,
    dev: Mapping[str, Any],
    role_index: torch.Tensor,
    device: torch.device,
    eval_batch_size: int,
) -> tuple[FrozenRoleOnlyBase, dict[str, Any], np.ndarray]:
    """Load the stronger no-sidecar P0 head trained with event-balanced loss."""

    return load_event_balanced_base(
        checkpoint_path,
        dev=dev,
        role_index=role_index,
        device=device,
        eval_batch_size=eval_batch_size,
        expected_arm="p0",
        base_kind="event_balanced_p0",
    )


def load_event_balanced_p5_base(
    checkpoint_path: Path,
    *,
    dev: Mapping[str, Any],
    role_index: torch.Tensor,
    device: torch.device,
    eval_batch_size: int,
) -> tuple[FrozenRoleOnlyBase, dict[str, Any], np.ndarray]:
    """Load the exact sidecar P5 head above its audited 1536-D cache."""

    return load_event_balanced_base(
        checkpoint_path,
        dev=dev,
        role_index=role_index,
        device=device,
        eval_batch_size=eval_batch_size,
        expected_arm="p5",
        base_kind="event_balanced_sidecar_p5",
    )


class RoleBasePredictionWrapper(nn.Module):
    def __init__(self, base: FrozenRoleOnlyBase):
        super().__init__()
        self.base = base

    def forward(
        self,
        features: torch.Tensor,
        unique_mask: torch.Tensor,
        delta_days: torch.Tensor,
        valid_fraction: torch.Tensor,
        role_index: torch.Tensor,
        *,
        arm: str,
    ) -> torch.Tensor:
        del valid_fraction, arm
        return self.base(features, unique_mask, delta_days, role_index)


class ContinuousGapEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        periods_days: Sequence[float] = (1, 3, 7, 30, 90, 365),
        maximum_days: float = 4000.0,
    ):
        super().__init__()
        periods = torch.as_tensor(tuple(float(value) for value in periods_days))
        if periods.numel() == 0 or torch.any(periods <= 0):
            raise ValueError("Gap periods must be positive.")
        self.register_buffer("periods", periods, persistent=True)
        self.maximum_days = float(maximum_days)
        input_dim = 2 + 2 * int(periods.numel())
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, delta_days: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(delta_days)
        safe = torch.where(finite, delta_days, torch.zeros_like(delta_days))
        safe = safe.clamp(-self.maximum_days, self.maximum_days)
        denominator = math.log1p(self.maximum_days)
        signed_log = torch.sign(safe) * torch.log1p(safe.abs()) / denominator
        absolute_log = torch.log1p(safe.abs()) / denominator
        angle = (
            2.0
            * math.pi
            * safe.unsqueeze(-1)
            / self.periods.to(device=safe.device, dtype=safe.dtype)
        )
        raw = torch.cat(
            (
                signed_log.unsqueeze(-1),
                absolute_log.unsqueeze(-1),
                torch.sin(angle),
                torch.cos(angle),
            ),
            dim=-1,
        )
        return self.mlp(raw) * finite.unsqueeze(-1).to(raw.dtype)


class TEMPOGlobalResidual(nn.Module):
    """Global background-conditioned temporal residual above a frozen base."""

    def __init__(
        self,
        base: nn.Module,
        *,
        feature_dim: int,
        num_roles: int,
        t0_index: int,
        temporal_dim: int,
        dropout: float,
        periods_days: Sequence[float],
    ):
        super().__init__()
        if not 0 <= int(t0_index) < int(num_roles):
            raise ValueError("t0_index is out of range.")
        self.base = copy.deepcopy(base).requires_grad_(False)
        self.base.eval()
        self.t0_index = int(t0_index)
        self.num_roles = int(num_roles)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.temporal_projection = nn.Linear(feature_dim, temporal_dim, bias=False)
        self.difference_encoder = nn.Sequential(
            nn.Linear(temporal_dim * 2, temporal_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, temporal_dim, bias=False),
        )
        self.role_embedding = nn.Embedding(num_roles, temporal_dim)
        self.gap_encoder = ContinuousGapEncoder(temporal_dim, periods_days)
        self.gate = nn.Sequential(
            nn.Linear(temporal_dim * 2 + 3, temporal_dim),
            nn.GELU(),
            nn.Linear(temporal_dim, 1),
        )
        self.evidence_norm = nn.LayerNorm(temporal_dim)
        self.residual = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, 1, bias=False),
        )
        # The first prediction is exactly the frozen base prediction.
        nn.init.zeros_(self.residual[-1].weight)

    def train(self, mode: bool = True) -> "TEMPOGlobalResidual":
        super().train(mode)
        self.base.eval()
        return self

    def _encoded_difference(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        signed = first - second
        raw = torch.cat((signed, signed.abs()), dim=-1)
        return self.difference_encoder(raw)

    def forward(
        self,
        features: torch.Tensor,
        unique_mask: torch.Tensor,
        delta_days: torch.Tensor,
        valid_fraction: torch.Tensor,
        role_index: torch.Tensor,
        *,
        arm: str,
        return_aux: bool = False,
        base_logit_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if arm not in ARM_NAMES:
            raise ValueError(f"Unknown arm {arm!r}.")
        if arm == "d7_sparse_slowfast":
            raise ValueError(
                "d7_sparse_slowfast requires SparseSlowFastResidual."
            )
        if features.ndim != 3:
            raise ValueError("features must have shape [B,T,D].")
        batch_size, timepoints, _ = features.shape
        expected = (batch_size, timepoints)
        for name, value in (
            ("unique_mask", unique_mask),
            ("delta_days", delta_days),
            ("valid_fraction", valid_fraction),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}.")
        if not unique_mask[:, self.t0_index].all():
            raise ValueError("Every row must retain unique t0 evidence.")
        if role_index.ndim == 1:
            role_index = role_index.view(1, -1).expand(batch_size, -1)
        if role_index.shape != expected:
            raise ValueError("role_index shape does not match features.")

        if base_logit_override is None:
            with torch.no_grad():
                base_logit = self.base(
                    features, unique_mask, delta_days, role_index
                )
        else:
            if base_logit_override.shape != (batch_size,):
                raise ValueError("base_logit_override must have shape [B].")
            base_logit = base_logit_override
        encoded = self.temporal_projection(self.feature_norm(features))
        history_indices = [
            index for index in range(timepoints) if index != self.t0_index
        ]
        history = encoded[:, history_indices]
        current = encoded[:, self.t0_index : self.t0_index + 1].expand_as(history)
        history_valid = unique_mask[:, history_indices]
        current_difference = self._encoded_difference(current, history)

        normal_by_history: list[torch.Tensor] = []
        for local_index in range(len(history_indices)):
            anchor = history[:, local_index : local_index + 1].expand_as(history)
            pair_difference = self._encoded_difference(anchor, history)
            pair_valid = history_valid.clone()
            pair_valid[:, local_index] = False
            denominator = pair_valid.sum(dim=1, keepdim=True).clamp_min(1)
            normal = (
                pair_difference * pair_valid.unsqueeze(-1).to(pair_difference.dtype)
            ).sum(dim=1) / denominator.to(pair_difference.dtype)
            normal_by_history.append(normal)
        normal_variation = torch.stack(normal_by_history, dim=1)
        onset = current_difference - normal_variation

        if arm in {
            "p1_delta",
            "d1_gated_delta",
            "d2_null_delta",
            "d3_gated_null_delta",
            "d6_onset_only",
        }:
            evidence_tokens = current_difference
        else:
            evidence_tokens = onset

        role = self.role_embedding(role_index[:, history_indices])
        gap = self.gap_encoder(delta_days[:, history_indices])
        quality = valid_fraction[:, history_indices].clamp(0.0, 1.0)
        cosine = F.cosine_similarity(current, history, dim=-1).unsqueeze(-1)
        distance = (current - history).square().mean(dim=-1).sqrt().unsqueeze(-1)
        gate_input = torch.cat(
            (role, gap, quality.unsqueeze(-1), cosine, distance), dim=-1
        )
        gate_logits = self.gate(gate_input).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~history_valid, -1e4)
        if arm in {
            "p3_gated_onset",
            "p4_null_onset",
            "d1_gated_delta",
            "d3_gated_null_delta",
            "d6_onset_only",
        }:
            weights = torch.softmax(gate_logits, dim=1)
        else:
            weights = history_valid.to(evidence_tokens.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        aggregate = (evidence_tokens * weights.unsqueeze(-1)).sum(dim=1)
        no_history = ~history_valid.any(dim=1)
        aggregate = aggregate.masked_fill(no_history.unsqueeze(-1), 0.0)
        if arm == "p0_base":
            aggregate = torch.zeros_like(aggregate)

        residual_logit = self.residual(self.evidence_norm(aggregate)).squeeze(-1)
        # LayerNorm maps exact zero to exact zero; bias-free residual preserves P0.
        final_logit = base_logit + residual_logit
        if not return_aux:
            return final_logit
        return final_logit, {
            "base_logit": base_logit,
            "residual_logit": residual_logit,
            "current_difference": current_difference,
            "normal_variation": normal_variation,
            "onset": onset,
            "history_weights": weights,
            "history_valid": history_valid,
        }


class SparseSlowFastResidual(TEMPOGlobalResidual):
    """Two-rate temporal control with independent acquisition gates.

    The fast branch contains ``prev1/prev2/prev3`` and the slow branch contains
    ``seasonal/year``. Both use the same signed-plus-absolute delta encoder but
    independent gate parameters. Their aggregates are fused by a fixed masked
    mean, so this control does not learn a branch liaison.
    """

    ARM = "d7_sparse_slowfast"

    def __init__(
        self,
        base: nn.Module,
        *,
        feature_dim: int,
        num_roles: int,
        t0_index: int,
        temporal_dim: int,
        dropout: float,
        periods_days: Sequence[float],
        fast_history_indices: Sequence[int] = (1, 2, 3),
        slow_history_indices: Sequence[int] = (4, 5),
    ):
        super().__init__(
            base,
            feature_dim=feature_dim,
            num_roles=num_roles,
            t0_index=t0_index,
            temporal_dim=temporal_dim,
            dropout=dropout,
            periods_days=periods_days,
        )
        self.fast_history_indices = tuple(
            int(value) for value in fast_history_indices
        )
        self.slow_history_indices = tuple(
            int(value) for value in slow_history_indices
        )
        expected_history = {
            index for index in range(int(num_roles)) if index != int(t0_index)
        }
        observed_history = set(self.fast_history_indices) | set(
            self.slow_history_indices
        )
        if (
            not self.fast_history_indices
            or not self.slow_history_indices
            or set(self.fast_history_indices) & set(self.slow_history_indices)
            or observed_history != expected_history
        ):
            raise ValueError(
                "Fast/slow role indices must be a non-empty partition of history."
            )
        self.fast_gate = self.gate
        self.slow_gate = copy.deepcopy(self.gate)
        del self.gate

    @staticmethod
    def _aggregate_branch(
        evidence: torch.Tensor,
        gate_input: torch.Tensor,
        valid: torch.Tensor,
        gate: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = gate(gate_input).squeeze(-1)
        logits = logits.masked_fill(~valid, -1e4)
        weights = torch.softmax(logits, dim=1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        aggregate = (evidence * weights.unsqueeze(-1)).sum(dim=1)
        available = valid.any(dim=1)
        aggregate = aggregate.masked_fill(~available.unsqueeze(-1), 0.0)
        return aggregate, weights, available

    def forward(
        self,
        features: torch.Tensor,
        unique_mask: torch.Tensor,
        delta_days: torch.Tensor,
        valid_fraction: torch.Tensor,
        role_index: torch.Tensor,
        *,
        arm: str,
        return_aux: bool = False,
        base_logit_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if arm != self.ARM:
            raise ValueError(f"SparseSlowFastResidual only supports {self.ARM}.")
        if features.ndim != 3:
            raise ValueError("features must have shape [B,T,D].")
        batch_size, timepoints, _ = features.shape
        expected = (batch_size, timepoints)
        for name, value in (
            ("unique_mask", unique_mask),
            ("delta_days", delta_days),
            ("valid_fraction", valid_fraction),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}.")
        if timepoints != self.num_roles:
            raise ValueError("Input role count differs from the frozen model.")
        if not unique_mask[:, self.t0_index].all():
            raise ValueError("Every row must retain unique t0 evidence.")
        if role_index.ndim == 1:
            role_index = role_index.view(1, -1).expand(batch_size, -1)
        if role_index.shape != expected:
            raise ValueError("role_index shape does not match features.")

        if base_logit_override is None:
            with torch.no_grad():
                base_logit = self.base(
                    features, unique_mask, delta_days, role_index
                )
        else:
            if base_logit_override.shape != (batch_size,):
                raise ValueError("base_logit_override must have shape [B].")
            base_logit = base_logit_override

        encoded = self.temporal_projection(self.feature_norm(features))
        current = encoded[:, self.t0_index : self.t0_index + 1]

        def branch_inputs(
            indices: Sequence[int],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            history = encoded[:, indices]
            expanded_current = current.expand_as(history)
            evidence = self._encoded_difference(expanded_current, history)
            role = self.role_embedding(role_index[:, indices])
            gap = self.gap_encoder(delta_days[:, indices])
            quality = valid_fraction[:, indices].clamp(0.0, 1.0)
            cosine = F.cosine_similarity(
                expanded_current, history, dim=-1
            ).unsqueeze(-1)
            distance = (
                (expanded_current - history)
                .square()
                .mean(dim=-1)
                .sqrt()
                .unsqueeze(-1)
            )
            gate_input = torch.cat(
                (role, gap, quality.unsqueeze(-1), cosine, distance), dim=-1
            )
            valid = unique_mask[:, indices]
            return evidence, gate_input, valid

        fast_evidence, fast_gate_input, fast_valid = branch_inputs(
            self.fast_history_indices
        )
        slow_evidence, slow_gate_input, slow_valid = branch_inputs(
            self.slow_history_indices
        )
        fast, fast_weights, fast_available = self._aggregate_branch(
            fast_evidence, fast_gate_input, fast_valid, self.fast_gate
        )
        slow, slow_weights, slow_available = self._aggregate_branch(
            slow_evidence, slow_gate_input, slow_valid, self.slow_gate
        )
        branch_count = (
            fast_available.to(fast.dtype) + slow_available.to(fast.dtype)
        ).clamp_min(1.0)
        aggregate = (fast + slow) / branch_count.unsqueeze(-1)
        no_history = ~(fast_available | slow_available)
        aggregate = aggregate.masked_fill(no_history.unsqueeze(-1), 0.0)
        residual_logit = self.residual(
            self.evidence_norm(aggregate)
        ).squeeze(-1)
        final_logit = base_logit + residual_logit
        if not return_aux:
            return final_logit
        return final_logit, {
            "base_logit": base_logit,
            "residual_logit": residual_logit,
            "fast_aggregate": fast,
            "slow_aggregate": slow,
            "fast_weights": fast_weights,
            "slow_weights": slow_weights,
            "fast_valid": fast_valid,
            "slow_valid": slow_valid,
            "fast_available": fast_available,
            "slow_available": slow_available,
            "fixed_branch_count": branch_count,
        }


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def trainable_parameter_signature(model: nn.Module) -> dict[str, Any]:
    shapes = {
        name: {
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        "parameter_count": int(sum(item["numel"] for item in shapes.values())),
        "parameter_shapes": shapes,
        "shape_sha256": hashlib.sha256(canonical_json_bytes(shapes)).hexdigest(),
    }


def fixed_batches(
    rows: int, batch_size: int, *, seed: int, epoch: int
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed) + 104_729 * int(epoch))
    return list(torch.randperm(rows, generator=generator).split(int(batch_size)))


def event_training_weights(
    labels: torch.Tensor, event_ids: Sequence[str]
) -> tuple[torch.Tensor, float]:
    target = labels.detach().cpu().numpy().astype(np.int64)
    row_weights = event_metrics.mean_one_event_weights(event_ids)
    positive_weight = event_metrics.event_balanced_positive_weight(target, event_ids)
    return torch.from_numpy(row_weights).float(), float(positive_weight)


def all_negative_topk_loss(
    logits: torch.Tensor,
    event_code: torch.Tensor,
    all_negative_by_event: torch.Tensor,
    *,
    top_k: int,
    margin: float,
) -> torch.Tensor:
    """Equal-event hard-negative loss over all-negative events in one batch."""

    selected_losses: list[torch.Tensor] = []
    present = event_code.unique()
    for code in present.tolist():
        if not bool(all_negative_by_event[int(code)]):
            continue
        values = logits[event_code.eq(int(code))]
        if values.numel() == 0:
            continue
        hard = values.topk(min(int(top_k), int(values.numel()))).values
        selected_losses.append(F.softplus(hard - float(margin)).mean())
    if not selected_losses:
        return logits.sum() * 0.0
    return torch.stack(selected_losses).mean()


def best_macro_threshold(
    labels: np.ndarray, probabilities: np.ndarray, event_ids: Sequence[str]
) -> tuple[float, float]:
    weights = cache_runner.event_balanced_row_weights(event_ids)
    return event_metrics.best_weighted_macro_f1_threshold(
        labels, probabilities, weights
    )


def metric_bundle(
    labels: np.ndarray,
    probabilities: np.ndarray,
    event_ids: Sequence[str],
    *,
    threshold: Optional[float] = None,
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    score = np.asarray(probabilities, dtype=np.float64)
    weights = cache_runner.event_balanced_row_weights(event_ids)
    if threshold is None:
        threshold, selected_macro = best_macro_threshold(target, score, event_ids)
    else:
        prediction = score >= float(threshold)
        selected_macro = f1_score(
            target,
            prediction,
            labels=[0, 1],
            average="macro",
            sample_weight=weights,
            zero_division=0,
        )
    prediction = (score >= float(threshold)).astype(np.int64)
    prediction_fixed = (score >= 0.5).astype(np.int64)

    event_frame = pd.DataFrame(
        {
            "event_id": [str(value) for value in event_ids],
            "label": target,
            "prediction": prediction,
            "prediction_fixed": prediction_fixed,
            "probability": score,
        }
    )
    grouped_label = event_frame.groupby("event_id", sort=True)["label"].max()
    all_negative_ids = set(grouped_label[grouped_label.eq(0)].index.tolist())
    positive_event_ids = set(grouped_label[grouped_label.eq(1)].index.tolist())
    null_frame = event_frame[event_frame["event_id"].isin(all_negative_ids)]
    positive_frame = event_frame[event_frame["event_id"].isin(positive_event_ids)].copy()
    if len(null_frame):
        by_event = null_frame.groupby("event_id", sort=True)
        fp_rates = by_event["prediction"].mean().to_numpy(dtype=np.float64)
        fp_rates_fixed = by_event["prediction_fixed"].mean().to_numpy(
            dtype=np.float64
        )
        any_fp = by_event["prediction"].max().to_numpy(dtype=np.float64)
        any_fp_fixed = by_event["prediction_fixed"].max().to_numpy(dtype=np.float64)
        mean_probabilities = by_event["probability"].mean().to_numpy(dtype=np.float64)
        max_probabilities = by_event["probability"].max().to_numpy(dtype=np.float64)
        fp_rows = int(null_frame["prediction"].sum())
        fp_rows_fixed = int(null_frame["prediction_fixed"].sum())
    else:
        fp_rates = np.zeros(0, dtype=np.float64)
        fp_rates_fixed = np.zeros(0, dtype=np.float64)
        any_fp = np.zeros(0, dtype=np.float64)
        any_fp_fixed = np.zeros(0, dtype=np.float64)
        mean_probabilities = np.zeros(0, dtype=np.float64)
        max_probabilities = np.zeros(0, dtype=np.float64)
        fp_rows = 0
        fp_rows_fixed = 0
    if len(positive_frame):
        positive_frame["true_detection"] = (
            positive_frame["label"] * positive_frame["prediction"]
        )
        positive_frame["true_detection_fixed"] = (
            positive_frame["label"] * positive_frame["prediction_fixed"]
        )
        positive_detection = (
            positive_frame.groupby("event_id", sort=True)["true_detection"]
            .max()
            .to_numpy(dtype=np.float64)
        )
        positive_detection_fixed = (
            positive_frame.groupby("event_id", sort=True)["true_detection_fixed"]
            .max()
            .to_numpy(dtype=np.float64)
        )
    else:
        positive_detection = np.zeros(0, dtype=np.float64)
        positive_detection_fixed = np.zeros(0, dtype=np.float64)

    return {
        "rows": int(len(target)),
        "events": int(len(set(str(value) for value in event_ids))),
        "row_ap": float(average_precision_score(target, score)),
        "row_auc": float(roc_auc_score(target, score)),
        "row_positive_f1_selected": float(
            f1_score(target, prediction, zero_division=0)
        ),
        "row_macro_f1_selected": float(
            f1_score(
                target,
                prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "event_balanced_ap": float(
            average_precision_score(target, score, sample_weight=weights)
        ),
        "event_balanced_auc": float(
            roc_auc_score(target, score, sample_weight=weights)
        ),
        "event_balanced_positive_f1_at_0_5": float(
            f1_score(
                target,
                prediction_fixed,
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "event_balanced_macro_f1_at_0_5": float(
            f1_score(
                target,
                prediction_fixed,
                labels=[0, 1],
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "selected_threshold": float(threshold),
        "event_balanced_positive_f1_selected": float(
            f1_score(
                target,
                prediction,
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "event_balanced_macro_f1_selected": float(selected_macro),
        "all_negative_event_count": int(len(all_negative_ids)),
        "all_negative_fp_mass": float(fp_rates.sum()),
        "all_negative_fp_rate_mean": float(fp_rates.mean()) if len(fp_rates) else 0.0,
        "all_negative_events_with_any_fp": int(any_fp.sum()),
        "all_negative_event_any_fp_rate": (
            float(any_fp.mean()) if len(any_fp) else 0.0
        ),
        "all_negative_fp_rows": fp_rows,
        "all_negative_fp_mass_at_0_5": float(fp_rates_fixed.sum()),
        "all_negative_events_with_any_fp_at_0_5": int(any_fp_fixed.sum()),
        "all_negative_event_any_fp_rate_at_0_5": (
            float(any_fp_fixed.mean()) if len(any_fp_fixed) else 0.0
        ),
        "all_negative_fp_rows_at_0_5": fp_rows_fixed,
        "positive_event_count": int(len(positive_event_ids)),
        "positive_event_any_detection_recall": (
            float(positive_detection.mean()) if len(positive_detection) else 0.0
        ),
        "positive_event_any_detection_recall_at_0_5": (
            float(positive_detection_fixed.mean())
            if len(positive_detection_fixed)
            else 0.0
        ),
        "all_negative_mean_probability": (
            float(mean_probabilities.mean()) if len(mean_probabilities) else 0.0
        ),
        "all_negative_max_probability_mean": (
            float(max_probabilities.mean()) if len(max_probabilities) else 0.0
        ),
        "probability_mean": float(score.mean()),
        "probability_std": float(score.std()),
    }


def batch_to_device(
    data: Mapping[str, Any], indices: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return (
        data["features"][indices].to(device, non_blocking=True),
        data["unique_mask"][indices].to(device, non_blocking=True),
        data["delta_days"][indices].to(device, non_blocking=True),
        data["valid_fraction"][indices].to(device, non_blocking=True),
        data["labels"][indices].to(device, non_blocking=True),
    )


def predict(
    model: nn.Module,
    data: Mapping[str, Any],
    role_index: torch.Tensor,
    *,
    arm: str,
    batch_size: int,
    device: torch.device,
    base_data: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    model.eval()
    result: list[torch.Tensor] = []
    with torch.inference_mode():
        for indices in torch.arange(len(data["labels"])).split(int(batch_size)):
            features, unique, delta, quality, _ = batch_to_device(
                data, indices, device
            )
            kwargs: dict[str, Any] = {}
            if base_data is not None:
                if not isinstance(model, TEMPOGlobalResidual):
                    raise TypeError("base_data override requires TEMPOGlobalResidual.")
                (
                    base_features,
                    base_unique,
                    base_delta,
                    _,
                    _,
                ) = batch_to_device(base_data, indices, device)
                base_logit = model.base(
                    base_features,
                    base_unique,
                    base_delta,
                    role_index.to(device),
                )
                kwargs["base_logit_override"] = base_logit
            logits = model(
                features,
                unique,
                delta,
                quality,
                role_index.to(device),
                arm=arm,
                **kwargs,
            )
            result.append(torch.sigmoid(logits).cpu())
    return torch.cat(result).numpy()


def build_availability_stratified_history_donors(
    event_ids: Sequence[str],
    valid_mask: torch.Tensor,
    unique_mask: torch.Tensor,
    *,
    seed: int,
    t0_index: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build cross-event donors within exact history-availability strata.

    A target row can receive history features only from a row whose complete
    non-t0 ``(valid_mask, unique_mask)`` pattern is identical.  Each stratum is
    deranged at the canonical-event level by
    :func:`build_cross_event_donor_indices`; a stratum with fewer than two
    canonical events fails closed instead of silently borrowing an
    availability-mismatched history.
    """

    if valid_mask.ndim != 2 or unique_mask.ndim != 2:
        raise ValueError("valid_mask and unique_mask must both be rank-2.")
    if valid_mask.shape != unique_mask.shape:
        raise ValueError("valid_mask and unique_mask shapes differ.")
    rows, timepoints = valid_mask.shape
    if rows != len(event_ids):
        raise ValueError("Availability masks must have one row per event id.")
    if not 0 <= int(t0_index) < int(timepoints):
        raise ValueError("t0_index is outside the availability masks.")

    history = [index for index in range(timepoints) if index != int(t0_index)]
    valid_history = valid_mask[:, history].detach().cpu().to(torch.bool)
    unique_history = unique_mask[:, history].detach().cpu().to(torch.bool)
    strata: dict[tuple[bool, ...], list[int]] = {}
    for row_index in range(rows):
        pattern = tuple(valid_history[row_index].tolist()) + tuple(
            unique_history[row_index].tolist()
        )
        strata.setdefault(pattern, []).append(row_index)

    donors = torch.empty(rows, dtype=torch.long)
    stratum_receipts: list[dict[str, Any]] = []
    failed_strata: list[dict[str, Any]] = []
    history_width = len(history)
    for stratum_index, pattern in enumerate(sorted(strata)):
        row_indices = strata[pattern]
        canonical_events = sorted({str(event_ids[index]) for index in row_indices})
        receipt = {
            "stratum_index": int(stratum_index),
            "valid_history_pattern": [
                bool(value) for value in pattern[:history_width]
            ],
            "unique_history_pattern": [
                bool(value) for value in pattern[history_width:]
            ],
            "rows": int(len(row_indices)),
            "canonical_event_count": int(len(canonical_events)),
        }
        if len(canonical_events) < 2:
            failed_strata.append(
                {
                    **receipt,
                    "canonical_events": canonical_events,
                    "reason": "fewer_than_two_canonical_events",
                }
            )
            continue
        local_event_ids = [str(event_ids[index]) for index in row_indices]
        # The sorted-stratum ordinal gives a stable independent RNG stream;
        # Python's process-randomized hash is deliberately not used.
        local_donors = cache_runner.build_cross_event_donor_indices(
            local_event_ids,
            seed=int(seed) + 1_000_003 * (stratum_index + 1),
        )
        global_donors = torch.tensor(
            [row_indices[index] for index in local_donors.tolist()],
            dtype=torch.long,
        )
        donors[torch.tensor(row_indices, dtype=torch.long)] = global_donors
        stratum_receipts.append(
            {
                **receipt,
                "donor_indices_sha256": cache_runner.tensor_sha256(
                    global_donors
                ),
                "all_cross_event": all(
                    str(event_ids[target_index])
                    != str(event_ids[donor_index])
                    for target_index, donor_index in zip(
                        row_indices, global_donors.tolist()
                    )
                ),
            }
        )

    if failed_strata:
        raise ValueError(
            "History shuffle failed closed because exact availability strata "
            "have fewer than two canonical events: "
            + json.dumps(failed_strata, sort_keys=True, separators=(",", ":"))
        )

    donor_valid_history = valid_history[donors]
    donor_unique_history = unique_history[donors]
    availability_mismatch = torch.logical_or(
        torch.any(valid_history != donor_valid_history, dim=1),
        torch.any(unique_history != donor_unique_history, dim=1),
    )
    availability_mismatch_rows = int(availability_mismatch.sum().item())
    all_cross_event = all(
        str(event_ids[index]) != str(event_ids[donor])
        for index, donor in enumerate(donors.tolist())
    )
    audit = {
        "seed": int(seed),
        "rows": int(rows),
        "history_indices": history,
        "strata_count": int(len(strata)),
        "pattern_count": int(len(strata)),
        "strata": stratum_receipts,
        "failed_strata": [],
        "availability_mismatch_rows": availability_mismatch_rows,
        "availability_pattern_mismatch_count": availability_mismatch_rows,
        "all_target_donor_strata_equal": availability_mismatch_rows == 0,
        "all_cross_event": all_cross_event,
        "all_donors_from_different_canonical_event": all_cross_event,
        "donor_indices_sha256": cache_runner.tensor_sha256(donors),
    }
    audit["valid"] = bool(
        audit["strata_count"] >= 1
        and audit["availability_mismatch_rows"] == 0
        and audit["all_target_donor_strata_equal"]
        and audit["all_cross_event"]
        and audit["failed_strata"] == []
        and all(
            record.get("all_cross_event") is True
            for record in stratum_receipts
        )
    )
    if audit["valid"] is not True:
        raise RuntimeError(
            "Exact availability-stratified history donor audit failed."
        )
    return donors, audit


def shuffled_development_view(
    data: Mapping[str, Any],
    *,
    seed: int,
    t0_index: int,
    return_audit: bool = False,
) -> Union[dict[str, Any], tuple[dict[str, Any], dict[str, Any]]]:
    donors, donor_audit = build_availability_stratified_history_donors(
        data["event_ids"],
        data["valid_mask"],
        data["unique_mask"],
        seed=int(seed),
        t0_index=int(t0_index),
    )
    output = {
        key: (value.clone() if isinstance(value, torch.Tensor) else list(value))
        for key, value in data.items()
    }
    history = [index for index in range(data["features"].shape[1]) if index != t0_index]
    # Formal mechanism shuffle replaces only the five history feature vectors.
    # Row identity, label, t0, availability, gaps, and quality/acquisition
    # metadata remain those of the target row.
    output["features"][:, history] = data["features"][donors][:, history]
    audit = {
        **donor_audit,
        "ordered_id_preserved": output["ids"] == list(data["ids"]),
        "ordered_plume_id_preserved": (
            output["plume_ids"] == list(data["plume_ids"])
        ),
        "ordered_event_id_preserved": (
            output["event_ids"] == list(data["event_ids"])
        ),
        "labels_preserved": torch.equal(output["labels"], data["labels"]),
        "shapes_preserved": all(
            output[key].shape == data[key].shape
            for key in (
                "features",
                "valid_mask",
                "unique_mask",
                "delta_days",
                "valid_fraction",
            )
        ),
        "t0_feature_preserved": torch.equal(
            output["features"][:, t0_index],
            data["features"][:, t0_index],
        ),
        "availability_pattern_preserved": (
            torch.equal(output["valid_mask"], data["valid_mask"])
            and torch.equal(output["unique_mask"], data["unique_mask"])
        ),
        "acquisition_metadata_preserved": (
            torch.equal(output["delta_days"], data["delta_days"])
            and torch.equal(
                output["valid_fraction"], data["valid_fraction"]
            )
        ),
        "only_history_features_replaced": True,
    }
    audit["valid"] = all(
        bool(audit[key])
        for key in (
            "all_target_donor_strata_equal",
            "all_cross_event",
            "all_donors_from_different_canonical_event",
            "ordered_id_preserved",
            "ordered_plume_id_preserved",
            "ordered_event_id_preserved",
            "labels_preserved",
            "shapes_preserved",
            "t0_feature_preserved",
            "availability_pattern_preserved",
            "acquisition_metadata_preserved",
            "only_history_features_replaced",
        )
    )
    if audit["valid"] is not True:
        raise RuntimeError("Cross-event history shuffle validity audit failed.")
    return (output, audit) if return_audit else output


def train_base(
    model: T0BaseHead,
    train: Mapping[str, Any],
    dev: Mapping[str, Any],
    *,
    t0_index: int,
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
) -> tuple[T0BaseHead, dict[str, Any]]:
    model = model.to(args.resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.base_learning_rate, weight_decay=args.weight_decay
    )
    row_weights, positive_weight = event_training_weights(
        train["labels"], train["event_ids"]
    )
    best: Optional[dict[str, Any]] = None
    history: list[dict[str, Any]] = []
    no_improvement = 0
    for epoch in range(1, int(args.base_epochs) + 1):
        model.train()
        losses: list[float] = []
        for indices in fixed_batches(
            len(train["labels"]), args.batch_size, seed=seed, epoch=epoch
        ):
            features = train["features"][indices, t0_index].to(args.resolved_device)
            labels = train["labels"][indices].to(args.resolved_device)
            weights = row_weights[indices].to(args.resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            per_row = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=torch.tensor(positive_weight, device=logits.device),
                reduction="none",
            )
            loss = (per_row * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        base_wrapper = BaseOnlyWrapper(model, t0_index=t0_index).to(
            args.resolved_device
        )
        probabilities = predict(
            base_wrapper,
            dev,
            torch.arange(dev["features"].shape[1]),
            arm="p0_base",
            batch_size=args.eval_batch_size,
            device=args.resolved_device,
        )
        metrics = metric_bundle(
            dev["labels"].numpy().astype(np.int64),
            probabilities,
            dev["event_ids"],
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": metrics,
        }
        history.append(record)
        improved = best is None or (
            metrics["event_balanced_ap"] > best["validation"]["event_balanced_ap"]
        )
        if improved:
            best = {
                **copy.deepcopy(record),
                "state": model_state(model),
            }
            no_improvement = 0
        else:
            no_improvement += 1
        print(
            f"[base] seed={seed} epoch={epoch} loss={np.mean(losses):.6f} "
            f"eventAP={metrics['event_balanced_ap']:.6f} "
            f"macroF1={metrics['event_balanced_macro_f1_selected']:.6f}",
            flush=True,
        )
        if no_improvement >= int(args.patience):
            break
    if best is None:
        raise RuntimeError("Base training produced no checkpoint.")
    model.load_state_dict(best.pop("state"), strict=True)
    model.requires_grad_(False).eval()
    atomic_json_write(output_dir / "base_metrics_history.json", history)
    return model.cpu(), best


class BaseOnlyWrapper(nn.Module):
    """Expose a T0BaseHead through the common TEMPO prediction signature."""

    def __init__(self, base: T0BaseHead, *, t0_index: int):
        super().__init__()
        self.base = base
        self.t0_index = int(t0_index)

    def forward(
        self,
        features: torch.Tensor,
        unique_mask: torch.Tensor,
        delta_days: torch.Tensor,
        valid_fraction: torch.Tensor,
        role_index: torch.Tensor,
        *,
        arm: str,
    ) -> torch.Tensor:
        del unique_mask, delta_days, valid_fraction, role_index, arm
        return self.base(features[:, self.t0_index])


def train_temporal_arm(
    arm: str,
    *,
    model: nn.Module,
    initial_state: Mapping[str, torch.Tensor],
    train: Mapping[str, Any],
    dev: Mapping[str, Any],
    role_index: torch.Tensor,
    t0_index: int,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
    frozen_base_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if arm not in TEMPORAL_ARMS:
        raise ValueError(f"{arm} is not a temporal arm.")
    model.load_state_dict(initial_state, strict=True)
    model = model.to(args.resolved_device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    row_weights, positive_weight = event_training_weights(
        train["labels"], train["event_ids"]
    )
    codes, names = event_codes(train["event_ids"])
    null_by_event = all_negative_event_mask(train["labels"], codes, names)
    initial_sha = state_dict_sha256(initial_state)
    history: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    no_improvement = 0
    stopped_for_patience = False
    started = time.monotonic()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_losses: list[float] = []
        cls_losses: list[float] = []
        null_losses: list[float] = []
        batches = fixed_batches(
            len(train["labels"]), args.batch_size, seed=seed, epoch=epoch
        )
        if args.max_train_steps:
            batches = batches[: int(args.max_train_steps)]
        for indices in batches:
            features, unique, delta, quality, labels = batch_to_device(
                train, indices, args.resolved_device
            )
            weights = row_weights[indices].to(args.resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                features,
                unique,
                delta,
                quality,
                role_index.to(args.resolved_device),
                arm=arm,
            )
            per_row = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=torch.tensor(positive_weight, device=logits.device),
                reduction="none",
            )
            classification_loss = (per_row * weights).sum() / weights.sum()
            null_loss = logits.sum() * 0.0
            if arm in {
                "p4_null_onset",
                "d2_null_delta",
                "d3_gated_null_delta",
            }:
                null_loss = all_negative_topk_loss(
                    logits,
                    codes[indices].to(logits.device),
                    null_by_event.to(logits.device),
                    top_k=args.null_top_k,
                    margin=args.null_margin,
                )
            loss = classification_loss + float(args.null_weight) * null_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"{arm} produced non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            total_losses.append(float(loss.detach()))
            cls_losses.append(float(classification_loss.detach()))
            null_losses.append(float(null_loss.detach()))

        probabilities = predict(
            model,
            dev,
            role_index,
            arm=arm,
            batch_size=args.eval_batch_size,
            device=args.resolved_device,
        )
        metrics = metric_bundle(
            dev["labels"].numpy().astype(np.int64),
            probabilities,
            dev["event_ids"],
        )
        record = {
            "arm": arm,
            "epoch": epoch,
            "optimizer_steps": len(batches),
            "train_loss": float(np.mean(total_losses)),
            "classification_loss": float(np.mean(cls_losses)),
            "null_loss": float(np.mean(null_losses)),
            "validation": metrics,
            "elapsed_seconds": float(time.monotonic() - started),
        }
        history.append(record)
        improved = best is None or (
            metrics["event_balanced_ap"] > best["validation"]["event_balanced_ap"]
        )
        if improved:
            best = copy.deepcopy(record)
            checkpoint = {
                "script_version": SCRIPT_VERSION,
                "arm": arm,
                "seed": int(seed),
                "epoch": int(epoch),
                "model": model_state(model),
                "initial_state_sha256": initial_sha,
                "frozen_base_checkpoint": frozen_base_audit.get(
                    "checkpoint"
                ),
                "frozen_base_checkpoint_sha256": frozen_base_audit.get(
                    "checkpoint_sha256"
                ),
                "frozen_base_model_state_sha256": frozen_base_audit.get(
                    "model_state_sha256"
                ),
                "validation": metrics,
                "test_or_sealed_or_holdout_read": False,
            }
            atomic_torch_save(output_dir / f"{arm}_best_event_ap.pt", checkpoint)
            predictions = pd.DataFrame(
                {
                    "id": dev["ids"],
                    "plume_id": dev["plume_ids"],
                    "event_id": dev["event_ids"],
                    "label": dev["labels"].long().tolist(),
                    "probability": probabilities,
                }
            )
            atomic_csv_write(
                output_dir / f"{arm}_best_event_ap_predictions.csv", predictions
            )
            no_improvement = 0
        else:
            no_improvement += 1
        atomic_json_write(output_dir / f"{arm}_metrics_history.json", history)
        print(
            f"[tempo] arm={arm} seed={seed} epoch={epoch} "
            f"loss={record['train_loss']:.6f} "
            f"eventAP={metrics['event_balanced_ap']:.6f} "
            f"eventAUC={metrics['event_balanced_auc']:.6f} "
            f"macroF1={metrics['event_balanced_macro_f1_selected']:.6f} "
            f"nullFP={metrics['all_negative_fp_mass']:.3f}",
            flush=True,
        )
        if no_improvement >= int(args.patience):
            stopped_for_patience = True
            break
    if best is None:
        raise RuntimeError(f"{arm} produced no checkpoint.")

    checkpoint = cache_runner.torch_load_trusted(
        output_dir / f"{arm}_best_event_ap.pt"
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    selected_threshold = float(best["validation"]["selected_threshold"])
    shuffled, shuffle_validity = shuffled_development_view(
        dev,
        seed=seed + 8_191,
        t0_index=t0_index,
        return_audit=True,
    )
    shuffled_probability = predict(
        model,
        shuffled,
        role_index,
        arm=arm,
        batch_size=args.eval_batch_size,
        device=args.resolved_device,
        base_data=dev,
    )
    shuffled_metrics = metric_bundle(
        dev["labels"].numpy().astype(np.int64),
        shuffled_probability,
        dev["event_ids"],
        threshold=selected_threshold,
    )
    history_path = output_dir / f"{arm}_metrics_history.json"
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    observed_epochs = [int(record["epoch"]) for record in history]
    observed_ap = [
        float(record["validation"]["event_balanced_ap"])
        for record in history
    ]
    selected_epoch = int(best["epoch"])
    selected_candidates = [
        epoch
        for epoch, value in zip(observed_epochs, observed_ap)
        if value == max(observed_ap)
    ]
    early_stop_receipt = {
        "selection_metric": "event_balanced_ap",
        "selection_tie_rule": "earliest epoch attaining maximum AP",
        "max_epochs": int(args.epochs),
        "patience": int(args.patience),
        "observed_epochs": observed_epochs,
        "observed_event_balanced_ap": observed_ap,
        "epochs_observed": len(history),
        "selected_epoch": selected_epoch,
        "selected_event_balanced_ap": float(
            best["validation"]["event_balanced_ap"]
        ),
        "stop_reason": (
            "patience_exhausted" if stopped_for_patience else "max_epochs_reached"
        ),
        "history_path": str(history_path.resolve()),
        "history_sha256": cache_runner.sha256_file(history_path),
        "valid": (
            observed_epochs == list(range(1, len(history) + 1))
            and selected_epoch == min(selected_candidates)
            and len(history) <= int(args.epochs)
            and (
                (stopped_for_patience and no_improvement >= int(args.patience))
                or (
                    not stopped_for_patience
                    and len(history) == int(args.epochs)
                )
            )
        ),
    }
    if early_stop_receipt["valid"] is not True:
        raise RuntimeError(f"{arm} early-stop receipt is internally invalid.")
    result = {
        "arm": arm,
        "seed": int(seed),
        "best": best,
        "early_stop_receipt": early_stop_receipt,
        "history_shuffle_fixed_model_and_threshold": shuffled_metrics,
        "history_shuffle_validity": shuffle_validity,
        "history_shuffle_delta": {
            key: float(shuffled_metrics[key] - best["validation"][key])
            for key in (
                "event_balanced_ap",
                "event_balanced_auc",
                "event_balanced_positive_f1_selected",
                "event_balanced_macro_f1_selected",
                "all_negative_fp_mass",
            )
        },
        "initial_state_sha256": initial_sha,
        "test_or_sealed_or_holdout_read": False,
    }
    atomic_json_write(output_dir / f"{arm}_summary.json", result)
    return result


def run_seed(
    *,
    seed: int,
    train: Mapping[str, Any],
    dev: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    role_index: torch.Tensor,
    args: argparse.Namespace,
    root: Path,
    train_cache_path: Path,
    dev_cache_path: Path,
) -> dict[str, Any]:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    t0_index = int(args.t0_index)
    feature_dim = int(train["features"].shape[-1])
    if args.base_kind == "legacy_role":
        checkpoint_path = Path(
            str(args.base_checkpoint_template).format(seed=int(seed))
        ).expanduser().resolve()
        base, base_audit, base_probability = load_frozen_role_only_base(
            checkpoint_path,
            train_cache_path=train_cache_path,
            dev_cache_path=dev_cache_path,
            dev=dev,
            role_index=role_index,
            device=args.resolved_device,
            eval_batch_size=args.eval_batch_size,
        )
        base_audit = {"base_kind": "legacy_role", **base_audit}
    elif args.base_kind == "event_balanced_p0":
        checkpoint_path = Path(
            args.event_base_checkpoint
        ).expanduser().resolve()
        base, base_audit, base_probability = load_event_balanced_p0_base(
            checkpoint_path,
            dev=dev,
            role_index=role_index,
            device=args.resolved_device,
            eval_batch_size=args.eval_batch_size,
        )
    elif args.base_kind == "event_balanced_sidecar_p5":
        checkpoint_path = Path(
            args.event_base_checkpoint
        ).expanduser().resolve()
        base, base_audit, base_probability = load_event_balanced_p5_base(
            checkpoint_path,
            dev=dev,
            role_index=role_index,
            device=args.resolved_device,
            eval_batch_size=args.eval_batch_size,
        )
    elif args.base_kind == "zero_logit":
        base = ZeroLogitBase()
        base_probability = np.full(
            len(dev["labels"]), 0.5, dtype=np.float64
        )
        empty_state: dict[str, torch.Tensor] = {}
        base_audit = {
            "base_kind": "zero_logit",
            "checkpoint": None,
            "checkpoint_sha256": None,
            "checkpoint_epoch": 0,
            "checkpoint_validation": None,
            "prediction_csv": None,
            "prediction_csv_sha256": None,
            "replay_max_abs_probability_error": 0.0,
            "replay_tolerance": 0.0,
            "replay_numerically_exact": True,
            "model_state_sha256": state_dict_sha256(empty_state),
            "base_feature_contract": "no appearance/base logit",
            "response_sidecar_used": False,
            "test_or_sealed_or_holdout_read": False,
        }
    else:
        raise ValueError(f"Unsupported base_kind={args.base_kind!r}.")
    if (
        args.base_kind != "zero_logit"
        and int(base_audit["checkpoint_epoch"]) <= 0
    ):
        raise ValueError("Frozen role-only checkpoint has an invalid selected epoch.")
    base_initial_sha = str(base_audit["model_state_sha256"])
    # ``base_probability`` was already replayed and checked against the
    # historical CSV.  The wrapper below is retained only for the shuffle.
    p0_metrics = metric_bundle(
        dev["labels"].numpy().astype(np.int64),
        base_probability,
        dev["event_ids"],
    )
    p0_predictions = pd.DataFrame(
        {
            "id": dev["ids"],
            "plume_id": dev["plume_ids"],
            "event_id": dev["event_ids"],
            "label": dev["labels"].long().tolist(),
            "probability": base_probability,
        }
    )
    atomic_csv_write(seed_dir / "p0_base_predictions.csv", p0_predictions)
    shuffled_dev = shuffled_development_view(
        dev, seed=seed + 8_191, t0_index=t0_index
    )
    base_shuffle_probability = predict(
        RoleBasePredictionWrapper(copy.deepcopy(base)).to(args.resolved_device),
        shuffled_dev,
        role_index,
        arm="p0_base",
        batch_size=args.eval_batch_size,
        device=args.resolved_device,
    )
    base_shuffle_metrics = metric_bundle(
        dev["labels"].numpy().astype(np.int64),
        base_shuffle_probability,
        dev["event_ids"],
        threshold=float(p0_metrics["selected_threshold"]),
    )

    temporal_prototype = TEMPOGlobalResidual(
        base,
        feature_dim=feature_dim,
        num_roles=int(train["features"].shape[1]),
        t0_index=t0_index,
        temporal_dim=args.temporal_dim,
        dropout=args.dropout,
        periods_days=tuple(float(value) for value in args.delta_periods.split(",")),
    )
    initial_state = model_state(temporal_prototype)
    epoch_zero_probability = predict(
        copy.deepcopy(temporal_prototype).to(args.resolved_device),
        dev,
        role_index,
        arm="d1_gated_delta",
        batch_size=args.eval_batch_size,
        device=args.resolved_device,
    )
    epoch_zero_probability_error = float(
        np.max(
            np.abs(
                np.asarray(epoch_zero_probability, dtype=np.float64)
                - np.asarray(base_probability, dtype=np.float64)
            )
        )
    )
    epoch_zero_logit_error = float(
        np.max(
            np.abs(
                safe_logit(epoch_zero_probability)
                - safe_logit(base_probability)
            )
        )
    )
    epoch_zero_probability_tolerance = 1e-7
    epoch_zero_logit_tolerance = 1e-6
    epoch_zero_temporal_replay = {
        "evaluated_before_any_temporal_optimizer_step": True,
        "arm": "d1_gated_delta",
        "zero_initialized_residual_output": True,
        "fresh_p0_checkpoint": base_audit.get("checkpoint"),
        "fresh_p0_checkpoint_sha256": base_audit.get("checkpoint_sha256"),
        "fresh_p0_model_state_sha256": base_audit.get("model_state_sha256"),
        "temporal_initial_state_sha256": state_dict_sha256(initial_state),
        "rows": int(len(base_probability)),
        "maximum_absolute_probability_error": epoch_zero_probability_error,
        "probability_tolerance": epoch_zero_probability_tolerance,
        "maximum_absolute_logit_error": epoch_zero_logit_error,
        "logit_tolerance": epoch_zero_logit_tolerance,
        "pass": (
            epoch_zero_probability_error <= epoch_zero_probability_tolerance
            and epoch_zero_logit_error <= epoch_zero_logit_tolerance
        ),
    }
    if epoch_zero_temporal_replay["pass"] is not True:
        raise RuntimeError(
            "Zero-initialized D1 prototype does not exactly replay fresh P0: "
            f"probability_error={epoch_zero_probability_error:.3e}, "
            f"logit_error={epoch_zero_logit_error:.3e}."
        )
    signature = trainable_parameter_signature(temporal_prototype)
    slowfast_prototype = SparseSlowFastResidual(
        base,
        feature_dim=feature_dim,
        num_roles=int(train["features"].shape[1]),
        t0_index=t0_index,
        temporal_dim=int(args.slowfast_temporal_dim),
        dropout=args.dropout,
        periods_days=tuple(
            float(value) for value in args.delta_periods.split(",")
        ),
    )
    slowfast_initial_state = model_state(slowfast_prototype)
    slowfast_signature = trainable_parameter_signature(slowfast_prototype)
    slowfast_capacity = {
        "d1_temporal_dim": int(args.temporal_dim),
        "d1_trainable_parameters": int(signature["parameter_count"]),
        "slowfast_temporal_dim": int(args.slowfast_temporal_dim),
        "slowfast_trainable_parameters": int(
            slowfast_signature["parameter_count"]
        ),
        "parameter_delta": int(
            slowfast_signature["parameter_count"]
            - signature["parameter_count"]
        ),
        "parameter_ratio": float(
            slowfast_signature["parameter_count"]
            / signature["parameter_count"]
        ),
        "fast_roles": ["prev1", "prev2", "prev3"],
        "slow_roles": ["seasonal", "year"],
        "independent_gate_parameters": True,
        "shared_delta_encoder": True,
        "branch_fusion": "fixed_availability_normalized_sum",
    }
    results: dict[str, Any] = {
        "p0_base": {
            "arm": "p0_base",
            "seed": int(seed),
            "best": {
                "epoch": 0,
                "validation": p0_metrics,
            },
            "epoch_zero_exact_frozen_base": {
                "source_checkpoint_epoch": int(
                    base_audit["checkpoint_epoch"]
                ),
                "replay_max_abs_probability_error": float(
                    base_audit["replay_max_abs_probability_error"]
                ),
                "replay_tolerance": float(base_audit["replay_tolerance"]),
                "exact_within_tolerance": bool(
                    base_audit["replay_numerically_exact"]
                ),
            },
            "zero_initialized_d1_prototype_exact_p0_replay": (
                epoch_zero_temporal_replay
            ),
            "frozen_role_only_base_audit": base_audit,
            "history_shuffle_fixed_model_and_threshold": base_shuffle_metrics,
            "history_shuffle_delta": {
                key: float(base_shuffle_metrics[key] - p0_metrics[key])
                for key in (
                    "event_balanced_ap",
                    "event_balanced_auc",
                    "event_balanced_positive_f1_selected",
                    "event_balanced_macro_f1_selected",
                    "all_negative_fp_mass",
                )
            },
            "test_or_sealed_or_holdout_read": False,
        }
    }
    for arm in args.resolved_arms:
        if arm == "p0_base":
            continue
        set_seed(seed)
        if arm == SparseSlowFastResidual.ARM:
            model = SparseSlowFastResidual(
                base,
                feature_dim=feature_dim,
                num_roles=int(train["features"].shape[1]),
                t0_index=t0_index,
                temporal_dim=int(args.slowfast_temporal_dim),
                dropout=args.dropout,
                periods_days=tuple(
                    float(value) for value in args.delta_periods.split(",")
                ),
            )
            arm_initial_state = slowfast_initial_state
            arm_signature = slowfast_signature
        else:
            model = TEMPOGlobalResidual(
                base,
                feature_dim=feature_dim,
                num_roles=int(train["features"].shape[1]),
                t0_index=t0_index,
                temporal_dim=args.temporal_dim,
                dropout=args.dropout,
                periods_days=tuple(
                    float(value) for value in args.delta_periods.split(",")
                ),
            )
            arm_initial_state = initial_state
            arm_signature = signature
        if trainable_parameter_signature(model) != arm_signature:
            raise RuntimeError(f"{arm} trainable parameter signature changed.")
        result = train_temporal_arm(
            arm,
            model=model,
            initial_state=arm_initial_state,
            train=train,
            dev=dev,
            role_index=role_index,
            t0_index=t0_index,
            seed=seed,
            args=args,
            output_dir=seed_dir,
            frozen_base_audit=base_audit,
        )
        if arm == "d1_gated_delta":
            result["epoch_zero_exact_p0_replay"] = (
                epoch_zero_temporal_replay
            )
        results[arm] = result
        atomic_json_write(seed_dir / "summary.json", results)

    seed_summary = {
        "script_version": SCRIPT_VERSION,
        "seed": int(seed),
        "cache_audit": dict(cache_audit),
        "frozen_role_only_base_audit": base_audit,
        "base_initial_state_sha256": base_initial_sha,
        "temporal_initial_state_sha256": state_dict_sha256(initial_state),
        "zero_initialized_d1_prototype_exact_p0_replay": (
            epoch_zero_temporal_replay
        ),
        "trainable_parameter_signature": signature,
        "trainable_parameter_signatures": {
            "d1_reference": signature,
            "d7_sparse_slowfast": slowfast_signature,
        },
        "slowfast_capacity_control": slowfast_capacity,
        "results": results,
        "test_or_sealed_or_holdout_read": False,
    }
    atomic_json_write(seed_dir / "summary.json", seed_summary)
    return seed_summary


def aggregate_runs(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms = sorted(
        set.intersection(
            *(set(summary["results"]) for summary in summaries)
        )
    )
    aggregate: dict[str, Any] = {}
    metric_names = (
        "event_balanced_ap",
        "event_balanced_auc",
        "event_balanced_positive_f1_selected",
        "event_balanced_macro_f1_selected",
        "all_negative_fp_mass",
        "all_negative_event_any_fp_rate",
        "all_negative_fp_rows",
        "all_negative_event_any_fp_rate_at_0_5",
        "all_negative_fp_rows_at_0_5",
        "positive_event_any_detection_recall",
        "positive_event_any_detection_recall_at_0_5",
    )
    for arm in arms:
        values: dict[str, list[float]] = {}
        shuffle_values: dict[str, list[float]] = {}
        for metric in metric_names:
            values[metric] = [
                float(summary["results"][arm]["best"]["validation"][metric])
                for summary in summaries
            ]
            shuffle_values[metric] = [
                float(
                    summary["results"][arm][
                        "history_shuffle_fixed_model_and_threshold"
                    ][metric]
                )
                for summary in summaries
            ]
        aggregate[arm] = {
            "seeds": [int(summary["seed"]) for summary in summaries],
            "mean": {metric: float(np.mean(value)) for metric, value in values.items()},
            "sample_sd": {
                metric: float(np.std(value, ddof=1)) if len(value) > 1 else 0.0
                for metric, value in values.items()
            },
            "per_seed": values,
            "history_shuffle_mean": {
                metric: float(np.mean(value))
                for metric, value in shuffle_values.items()
            },
            "history_shuffle_delta_mean": {
                metric: float(np.mean(shuffle_values[metric]) - np.mean(values[metric]))
                for metric in metric_names
            },
        }
    return {
        "script_version": SCRIPT_VERSION,
        "arms": aggregate,
        "existing_clean_development_guardrail": {
            "source": (
                "rctp_l89_sidecar_fallback_v1/"
                "downstream_event_balanced_seed20260728"
            ),
            "p0_event_balanced_ap": 0.749149,
            "p0_event_balanced_macro_f1_selected": 0.761262,
            "p5_event_balanced_ap": 0.765148,
            "p5_event_balanced_macro_f1_selected": 0.772047,
            "interpretation": (
                "A TEMPO arm must beat P5, not merely the older role-only "
                "checkpoint, to become the strongest clean dev head."
            ),
        },
        "selection": "best epoch by development event-balanced AP",
        "threshold": "development event-balanced macro-F1 maximizer",
        "test_or_sealed_or_holdout_read": False,
    }


def safe_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def load_prediction_table(path: Path) -> pd.DataFrame:
    assert_development_path(path, purpose="development prediction table")
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"id", "event_id", "label", "probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing prediction columns {missing}.")
    if not np.isfinite(frame["probability"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path} contains non-finite probabilities.")
    return frame


def command_blend_audit(args: argparse.Namespace) -> None:
    output_path = Path(args.output_json).expanduser().resolve()
    assert_development_path(output_path, purpose="blend audit output")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    seeds = parse_ints(args.seeds)
    weights = tuple(
        float(part.strip())
        for part in str(args.candidate_weights).split(",")
        if part.strip()
    )
    if not weights or any(not 0.0 <= value <= 1.5 for value in weights):
        raise ValueError("Candidate weights must lie in [0, 1.5].")
    metric_names = (
        "event_balanced_ap",
        "event_balanced_auc",
        "event_balanced_positive_f1_selected",
        "event_balanced_macro_f1_selected",
        "all_negative_fp_mass",
        "all_negative_event_any_fp_rate",
        "all_negative_fp_rows",
        "positive_event_any_detection_recall",
    )
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        base_path = Path(
            str(args.base_predictions).format(seed=int(seed))
        ).expanduser().resolve()
        candidate_path = Path(
            str(args.candidate_predictions).format(seed=int(seed))
        ).expanduser().resolve()
        base_frame = load_prediction_table(base_path)
        candidate_frame = load_prediction_table(candidate_path)
        for column in ("id", "event_id"):
            if (
                base_frame[column].astype(str).tolist()
                != candidate_frame[column].astype(str).tolist()
            ):
                raise ValueError(
                    f"Prediction tables differ in ordered {column}: "
                    f"{base_path} vs {candidate_path}."
                )
        base_labels = base_frame["label"].to_numpy(dtype=np.int64)
        candidate_labels = candidate_frame["label"].to_numpy(dtype=np.int64)
        if not np.array_equal(base_labels, candidate_labels):
            raise ValueError("Prediction table labels differ.")
        event_ids = base_frame["event_id"].astype(str).tolist()
        base_probability = base_frame["probability"].to_numpy(dtype=np.float64)
        candidate_probability = candidate_frame["probability"].to_numpy(
            dtype=np.float64
        )
        base_logit = safe_logit(base_probability)
        candidate_logit = safe_logit(candidate_probability)
        seed_results: dict[str, Any] = {
            "base": metric_bundle(base_labels, base_probability, event_ids),
            "candidate": metric_bundle(
                base_labels, candidate_probability, event_ids
            ),
            "blends": {},
            "provenance": {
                "base_predictions": str(base_path),
                "base_predictions_sha256": cache_runner.sha256_file(base_path),
                "candidate_predictions": str(candidate_path),
                "candidate_predictions_sha256": cache_runner.sha256_file(
                    candidate_path
                ),
            },
        }
        for weight in weights:
            blend_logit = (
                (1.0 - float(weight)) * base_logit
                + float(weight) * candidate_logit
            )
            blend_probability = 1.0 / (1.0 + np.exp(-blend_logit))
            weight_key = f"{weight:.6f}"
            blend_metrics = metric_bundle(
                base_labels, blend_probability, event_ids
            )
            seed_results["blends"][weight_key] = blend_metrics
            if args.output_predictions:
                prediction_path = Path(
                    str(args.output_predictions).format(
                        seed=int(seed), weight=weight_key
                    )
                ).expanduser().resolve()
                assert_development_path(
                    prediction_path, purpose="blend prediction output"
                )
                if prediction_path.exists() and not args.overwrite:
                    raise FileExistsError(prediction_path)
                prediction_frame = base_frame.copy()
                prediction_frame["probability"] = blend_probability
                prediction_frame["selected_threshold"] = float(
                    blend_metrics["selected_threshold"]
                )
                atomic_csv_write(prediction_path, prediction_frame)
                seed_results["provenance"][
                    f"blend_predictions_weight_{weight_key}"
                ] = str(prediction_path)
                seed_results["provenance"][
                    f"blend_predictions_weight_{weight_key}_sha256"
                ] = cache_runner.sha256_file(prediction_path)
        per_seed[str(seed)] = seed_results

    aggregate: dict[str, Any] = {}
    for weight in weights:
        key = f"{weight:.6f}"
        aggregate[key] = {}
        for metric in metric_names:
            values = [
                float(per_seed[str(seed)]["blends"][key][metric])
                for seed in seeds
            ]
            aggregate[key][metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "per_seed": values,
            }
    payload = {
        "script_version": SCRIPT_VERSION,
        "audit_type": "fixed-logit-blend-development-only",
        "base_name": args.base_name,
        "candidate_name": args.candidate_name,
        "candidate_weights": list(weights),
        "weight_provenance": args.weight_provenance,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "post_hoc_exploratory": True,
        "weights_refit_per_seed": False,
        "threshold_refit_on_each_development_prediction": True,
        "test_or_sealed_or_holdout_read": False,
    }
    atomic_json_write(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def weighted_selected_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    prediction = probability >= float(threshold)
    tp = float(sample_weights[(target == 1) & prediction].sum())
    fp = float(sample_weights[(target == 0) & prediction].sum())
    fn = float(sample_weights[(target == 1) & ~prediction].sum())
    tn = float(sample_weights[(target == 0) & ~prediction].sum())
    positive_denominator = 2.0 * tp + fp + fn
    negative_denominator = 2.0 * tn + fp + fn
    positive = (
        0.0
        if positive_denominator <= 0
        else 2.0 * tp / positive_denominator
    )
    negative = (
        0.0
        if negative_denominator <= 0
        else 2.0 * tn / negative_denominator
    )
    positive_mass = float(sample_weights[target == 1].sum())
    negative_mass = float(sample_weights[target == 0].sum())
    auc = (
        0.5
        if positive_mass <= 0 or negative_mass <= 0
        else float(
            roc_auc_score(target, probability, sample_weight=sample_weights)
        )
    )
    return {
        "event_balanced_ap": float(
            average_precision_score(target, probability, sample_weight=sample_weights)
        ),
        "event_balanced_auc": auc,
        "event_balanced_positive_f1_selected": float(positive),
        "event_balanced_macro_f1_selected": float((positive + negative) / 2.0),
    }


def paired_event_bootstrap_deltas(
    labels: np.ndarray,
    event_ids: Sequence[str],
    probabilities: Mapping[str, np.ndarray],
    point_metrics: Mapping[str, Mapping[str, Any]],
    comparisons: Sequence[tuple[str, str, str]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    names = tuple(probabilities)
    if not names:
        raise ValueError("At least one probability arm is required.")
    target = np.asarray(labels, dtype=np.int64)
    events = np.asarray([str(value) for value in event_ids], dtype=object)
    canonical, codes, sizes = np.unique(
        events, return_inverse=True, return_counts=True
    )
    if len(canonical) < 2:
        raise ValueError("Event bootstrap requires at least two events.")
    for name in names:
        if np.asarray(probabilities[name]).shape != target.shape:
            raise ValueError(f"{name} probabilities do not match labels.")
    thresholds = {
        name: float(point_metrics[name]["selected_threshold"]) for name in names
    }
    metric_names = (
        "event_balanced_ap",
        "event_balanced_auc",
        "event_balanced_positive_f1_selected",
        "event_balanced_macro_f1_selected",
    )
    distributions: dict[str, dict[str, list[float]]] = {
        comparison_name: {metric: [] for metric in metric_names}
        for comparison_name, _, _ in comparisons
    }
    rng = np.random.default_rng(int(seed))
    for replicate in range(int(replicates)):
        sampled = rng.integers(0, len(canonical), size=len(canonical))
        event_counts = np.bincount(sampled, minlength=len(canonical)).astype(
            np.float64
        )
        row_weights = event_counts[codes] / sizes[codes]
        replicate_metrics = {
            name: weighted_selected_metrics(
                target,
                probabilities[name],
                row_weights,
                thresholds[name],
            )
            for name in names
        }
        for comparison_name, left, right in comparisons:
            for metric in metric_names:
                distributions[comparison_name][metric].append(
                    replicate_metrics[left][metric]
                    - replicate_metrics[right][metric]
                )
        if (replicate + 1) % 500 == 0:
            print(
                f"[bootstrap] replicates={replicate + 1}/{replicates}",
                flush=True,
            )

    output: dict[str, Any] = {}
    for comparison_name, left, right in comparisons:
        output[comparison_name] = {}
        for metric in metric_names:
            values = np.asarray(
                distributions[comparison_name][metric], dtype=np.float64
            )
            output[comparison_name][metric] = {
                "point": float(
                    point_metrics[left][metric] - point_metrics[right][metric]
                ),
                "ci_95_low": float(np.percentile(values, 2.5)),
                "ci_95_high": float(np.percentile(values, 97.5)),
            }
    return output


def validate_aligned_prediction_frames(
    frames: Mapping[str, pd.DataFrame]
) -> None:
    names = tuple(frames)
    reference = frames[names[0]]
    for name in names[1:]:
        candidate = frames[name]
        for column in ("id", "event_id"):
            if (
                reference[column].astype(str).tolist()
                != candidate[column].astype(str).tolist()
            ):
                raise ValueError(f"{name} differs from {names[0]} in {column}.")
        if not np.array_equal(
            reference["label"].to_numpy(dtype=np.int64),
            candidate["label"].to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"{name} labels differ from {names[0]}.")


def command_bootstrap_audit(args: argparse.Namespace) -> None:
    output_path = Path(args.output_json).expanduser().resolve()
    assert_development_path(output_path, purpose="bootstrap audit output")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    seeds = parse_ints(args.seeds)
    p5_path = Path(args.p5_predictions).expanduser().resolve()
    p5_frame = load_prediction_table(p5_path)
    per_seed: dict[str, Any] = {}
    candidate_probabilities: list[np.ndarray] = []
    p0_frames: list[pd.DataFrame] = []
    for seed in seeds:
        p0_path = Path(
            str(args.p0_predictions).format(seed=int(seed))
        ).expanduser().resolve()
        candidate_path = Path(
            str(args.candidate_predictions).format(seed=int(seed))
        ).expanduser().resolve()
        frames = {
            "p0": load_prediction_table(p0_path),
            "candidate": load_prediction_table(candidate_path),
            "p5": p5_frame,
        }
        validate_aligned_prediction_frames(frames)
        labels = frames["p0"]["label"].to_numpy(dtype=np.int64)
        event_ids = frames["p0"]["event_id"].astype(str).tolist()
        probabilities = {
            name: frame["probability"].to_numpy(dtype=np.float64)
            for name, frame in frames.items()
        }
        metrics = {
            name: metric_bundle(labels, probability, event_ids)
            for name, probability in probabilities.items()
        }
        deltas = paired_event_bootstrap_deltas(
            labels,
            event_ids,
            probabilities,
            metrics,
            (
                ("candidate_minus_p0", "candidate", "p0"),
                ("candidate_minus_p5", "candidate", "p5"),
            ),
            replicates=args.replicates,
            seed=args.bootstrap_seed + int(seed),
        )
        per_seed[str(seed)] = {
            "point_metrics": metrics,
            "paired_event_bootstrap_deltas": deltas,
            "provenance": {
                "p0_predictions": str(p0_path),
                "p0_sha256": cache_runner.sha256_file(p0_path),
                "candidate_predictions": str(candidate_path),
                "candidate_sha256": cache_runner.sha256_file(candidate_path),
                "p5_predictions": str(p5_path),
                "p5_sha256": cache_runner.sha256_file(p5_path),
            },
        }
        candidate_probabilities.append(probabilities["candidate"])
        p0_frames.append(frames["p0"])

    validate_aligned_prediction_frames(
        {
            f"p0_seed_{seed}": frame
            for seed, frame in zip(seeds, p0_frames)
        }
    )
    reference = p0_frames[0]
    labels = reference["label"].to_numpy(dtype=np.int64)
    event_ids = reference["event_id"].astype(str).tolist()
    candidate_average_probability = 1.0 / (
        1.0
        + np.exp(
            -np.mean(
                np.stack(
                    [safe_logit(value) for value in candidate_probabilities],
                    axis=0,
                ),
                axis=0,
            )
        )
    )
    average_probabilities = {
        "p0": reference["probability"].to_numpy(dtype=np.float64),
        "candidate_average": candidate_average_probability,
        "p5": p5_frame["probability"].to_numpy(dtype=np.float64),
    }
    average_metrics = {
        name: metric_bundle(labels, probability, event_ids)
        for name, probability in average_probabilities.items()
    }
    average_deltas = paired_event_bootstrap_deltas(
        labels,
        event_ids,
        average_probabilities,
        average_metrics,
        (
            ("candidate_average_minus_p0", "candidate_average", "p0"),
            ("candidate_average_minus_p5", "candidate_average", "p5"),
        ),
        replicates=args.replicates,
        seed=args.bootstrap_seed + 999_983,
    )
    payload = {
        "script_version": SCRIPT_VERSION,
        "audit_type": "canonical-event-paired-bootstrap",
        "candidate_name": args.candidate_name,
        "seeds": list(seeds),
        "replicates": int(args.replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "thresholds_refit_per_replicate": False,
        "checkpoint_and_threshold_selected_on_development": True,
        "per_seed": per_seed,
        "seed_logit_average": {
            "point_metrics": average_metrics,
            "paired_event_bootstrap_deltas": average_deltas,
        },
        "p5_boundary": (
            "P5 is one fixed seed-20260728 checkpoint reused in every paired "
            "comparison; only candidate initialization/batch randomness varies."
        ),
        "post_hoc_exploratory": True,
        "test_or_sealed_or_holdout_read": False,
    }
    atomic_json_write(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def command_run(args: argparse.Namespace) -> None:
    train_path = Path(args.train_cache).expanduser().resolve()
    dev_path = Path(args.dev_cache).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    assert_development_path(output_root, purpose="TEMPO output")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Non-empty output exists: {output_root}; pass --overwrite for a new tag."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json_write(
        output_root / "run_status.json",
        {
            "status": "running",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "script_version": SCRIPT_VERSION,
            "test_or_sealed_or_holdout_read": False,
        },
    )
    try:
        train, dev, cache_audit, role_index = load_development_pair(
            train_path, dev_path
        )
        if args.base_kind == "event_balanced_sidecar_p5":
            reference_train_path = Path(
                args.identity_reference_train_cache
            ).expanduser().resolve()
            reference_dev_path = Path(
                args.identity_reference_dev_cache
            ).expanduser().resolve()
            cache_audit["sidecar_reference_alignment"] = {
                "train": audit_reference_cache_alignment(
                    train, reference_train_path, expected_split="train"
                ),
                "validation": audit_reference_cache_alignment(
                    dev, reference_dev_path, expected_split="val"
                ),
            }
        args.t0_index = int(cache_audit["t0_index"])
        arms = parse_names(args.arms)
        unknown = sorted(set(arms) - set(ARM_NAMES))
        if unknown:
            raise ValueError(f"Unknown arms {unknown}; choices={ARM_NAMES}")
        args.resolved_arms = arms
        seeds = parse_ints(args.seeds)
        device = torch.device(args.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        args.resolved_device = device
        config = {
            key: value
            for key, value in vars(args).items()
            if key not in {"resolved_device", "resolved_arms", "handler"}
        }
        config.update(
            {
                "resolved_device": str(device),
                "resolved_arms": list(arms),
                "cache_audit": cache_audit,
                "matched_contract": {
                    "one_frozen_t0_base_per_seed": True,
                    "same_temporal_initial_state_within_seed": True,
                    "same_batches_optimizer_and_updates_within_seed": True,
                    "zero_initialized_residual_output": True,
                    "p0_is_exact_frozen_base": True,
                    "p1_p2_p3_p4_same_trainable_parameter_signature": True,
                    "p4_only_loss_difference": "all-negative-event-top-k",
                    "history_shuffle": "evaluation-only-cross-canonical-event",
                    "d7_sparse_slowfast": {
                        "fast_roles": ["prev1", "prev2", "prev3"],
                        "slow_roles": ["seasonal", "year"],
                        "independent_acquisition_gates": True,
                        "shared_signed_abs_delta_encoder": True,
                        "branch_fusion": "fixed_availability_normalized_sum",
                        "capacity_reference": "d1_gated_delta",
                    },
                },
                "test_or_sealed_or_holdout_read": False,
            }
        )
        atomic_json_write(output_root / "run_config.json", config)
        summaries: list[dict[str, Any]] = []
        for seed in seeds:
            summaries.append(
                run_seed(
                    seed=seed,
                    train=train,
                    dev=dev,
                    cache_audit=cache_audit,
                    role_index=role_index,
                    args=args,
                    root=output_root,
                    train_cache_path=train_path,
                    dev_cache_path=dev_path,
                )
            )
            aggregate = aggregate_runs(summaries)
            atomic_json_write(output_root / "aggregate.json", aggregate)
        atomic_json_write(
            output_root / "run_status.json",
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "seeds": list(seeds),
                "arms": list(arms),
                "test_or_sealed_or_holdout_read": False,
            },
        )
        print(json.dumps(aggregate_runs(summaries), indent=2, sort_keys=True))
    except Exception as error:
        atomic_json_write(
            output_root / "run_status.json",
            {
                "status": "failed",
                "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "test_or_sealed_or_holdout_read": False,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    root = "/diniuvol/yuyao/methanefuse_research_20260727"
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument(
        "--train-cache", default=f"{root}/cache/l89_ragged_cls_v1/train.pt"
    )
    run.add_argument(
        "--dev-cache", default=f"{root}/cache/l89_ragged_cls_v1/val.pt"
    )
    run.add_argument(
        "--base-checkpoint-template",
        default=(
            f"{root}/results/l89_ragged_cls_v1_seed"
            "{seed}/role_only/checkpoint_best_ap.pt"
        ),
        help="Audited role_only checkpoint path; {seed} is expanded per seed.",
    )
    run.add_argument(
        "--base-kind",
        choices=(
            "legacy_role",
            "event_balanced_p0",
            "event_balanced_sidecar_p5",
            "zero_logit",
        ),
        default="legacy_role",
    )
    run.add_argument(
        "--event-base-checkpoint",
        default=(
            f"{root}/rctp_l89_sidecar_fallback_v1/"
            "downstream_event_balanced_seed20260728/p0/"
            "checkpoint_best_event_balanced_ap.pt"
        ),
        help=(
            "No-sidecar event-balanced P0 checkpoint. Its input is exactly "
            "[original Panopticon CLS, zeros]."
        ),
    )
    run.add_argument(
        "--identity-reference-train-cache",
        default=f"{root}/cache/l89_ragged_cls_v1/train.pt",
        help=(
            "Original 768-D train cache used to prove that a P5 cache is a "
            "row-exact feature extension."
        ),
    )
    run.add_argument(
        "--identity-reference-dev-cache",
        default=f"{root}/cache/l89_ragged_cls_v1/val.pt",
        help=(
            "Original 768-D development cache used for strict P5 alignment."
        ),
    )
    run.add_argument("--output-dir", required=True)
    run.add_argument("--arms", default=",".join(ARM_NAMES))
    run.add_argument("--seeds", default="20260728")
    run.add_argument("--epochs", type=int, default=4)
    run.add_argument("--patience", type=int, default=2)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--eval-batch-size", type=int, default=1024)
    run.add_argument("--temporal-dim", type=int, default=192)
    run.add_argument(
        "--slowfast-temporal-dim",
        type=int,
        default=174,
        help=(
            "Width for d7_sparse_slowfast; 174 yields 413,744 trainable "
            "parameters versus D1's 412,801 (+0.228%)."
        ),
    )
    run.add_argument("--learning-rate", type=float, default=8e-4)
    run.add_argument("--weight-decay", type=float, default=0.02)
    run.add_argument("--grad-clip", type=float, default=1.0)
    run.add_argument("--dropout", type=float, default=0.15)
    run.add_argument("--delta-periods", default="1,3,7,30,90,365")
    run.add_argument("--null-weight", type=float, default=0.10)
    run.add_argument("--null-top-k", type=int, default=4)
    run.add_argument("--null-margin", type=float, default=0.0)
    run.add_argument("--max-train-steps", type=int, default=0)
    run.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(handler=command_run)

    blend = subparsers.add_parser("blend-audit")
    blend.add_argument("--base-predictions", required=True)
    blend.add_argument("--candidate-predictions", required=True)
    blend.add_argument("--base-name", required=True)
    blend.add_argument("--candidate-name", required=True)
    blend.add_argument("--candidate-weights", required=True)
    blend.add_argument("--weight-provenance", required=True)
    blend.add_argument("--seeds", default="20260728")
    blend.add_argument("--output-json", required=True)
    blend.add_argument(
        "--output-predictions",
        help=(
            "Optional output template for blended development predictions; "
            "supports {seed} and {weight}."
        ),
    )
    blend.add_argument("--overwrite", action="store_true")
    blend.set_defaults(handler=command_blend_audit)

    bootstrap = subparsers.add_parser("bootstrap-audit")
    bootstrap.add_argument("--p0-predictions", required=True)
    bootstrap.add_argument("--candidate-predictions", required=True)
    bootstrap.add_argument("--candidate-name", required=True)
    bootstrap.add_argument("--p5-predictions", required=True)
    bootstrap.add_argument("--seeds", default="20260727,20260728,20260729")
    bootstrap.add_argument("--replicates", type=int, default=2000)
    bootstrap.add_argument("--bootstrap-seed", type=int, default=20260728)
    bootstrap.add_argument("--output-json", required=True)
    bootstrap.add_argument("--overwrite", action="store_true")
    bootstrap.set_defaults(handler=command_bootstrap_audit)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
