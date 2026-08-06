#!/usr/bin/env python3
"""Matched-compute TransientQuery heads for cached 360 m sensor features.

The module is intentionally independent of image loading and the Panopticon
backbone.  Its input is a dense feature cache with shape ``[B, S, T, D]`` and a
boolean validity mask with shape ``[B, S, T]``.  In the 360 m experiment,
``S=4`` (S2, L89, EMIT, S5P), ``T=3`` (t0, 90, 360), and ``D=768``.

All experiment arms use the same model and parameters:

``current_only``
    Keep only t0 in the attention mask.  All feature projections and both
    query blocks still execute, so it is a matched input intervention.

``transient_query``
    Use t0 as the query and all valid temporal roles as keys/values.

``scale_aware_transient_query``
    Use temporal evidence for the 360 m optical sensors, while S5P contributes
    only its current coarse-context token.  The 360 m crop pipeline stores an
    approximately 10.5 km S5P footprint, so its history is not treated as
    spatially matched plume evidence.

``history_shuffle_train``
    During training, replace each valid historical role with evidence from
    the same sensor but a different event/plume.  At evaluation time this arm
    uses the coherent TransientQuery input.

Only t0 queries are updated.  The resulting per-sensor evidence is fused with
availability-aware attention and supervised by a fused binary objective plus
masked per-sensor auxiliary objectives.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


ARM_NAMES = (
    "current_only",
    "transient_query",
    "scale_aware_transient_query",
    "history_shuffle_train",
)
DEFAULT_SENSOR_NAMES = ("s2", "l89", "emit", "s5p")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact bytes on CPU."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(list(value.shape)))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Stable initialization/checkpoint hash independent of mapping order."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json_bytes(list(value.shape)))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_parameter_signature(model: nn.Module) -> dict[str, Any]:
    """Return a serialization-friendly architecture/parameter signature."""

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
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "parameter_shapes": shapes,
        "shape_sha256": hashlib.sha256(_canonical_json_bytes(shapes)).hexdigest(),
    }


def set_deterministic_seed(seed: int) -> None:
    """Set Python, NumPy, and torch seeds for matched-arm head training."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def fixed_epoch_batches(
    rows: int,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
) -> list[torch.Tensor]:
    """Materialize a deterministic epoch order without global RNG state."""

    if rows <= 0:
        raise ValueError("rows must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if shuffle:
        generator = torch.Generator().manual_seed(int(seed) + int(epoch))
        order = torch.randperm(rows, generator=generator)
    else:
        order = torch.arange(rows)
    return list(order.split(int(batch_size)))


class CurrentQueryBlock(nn.Module):
    """Cross-attention/FFN block that updates only a one-token t0 query."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads.")
        hidden_dim = int(round(model_dim * float(mlp_ratio)))
        self.query_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(float(dropout))
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 3 or query.shape[1] != 1:
            raise ValueError("query must have shape [N,1,D].")
        if context.ndim != 3:
            raise ValueError("context must have shape [N,T,D].")
        if valid_mask.shape != context.shape[:2]:
            raise ValueError("valid_mask shape must match context [N,T].")
        if not valid_mask[:, 0].all():
            raise ValueError("Every CurrentQueryBlock row must retain valid t0.")
        attended, _ = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        query = query + self.attention_dropout(attended)
        return query + self.ffn(self.ffn_norm(query))


@dataclass(frozen=True)
class TransientQueryOutput:
    """Outputs needed by the fused and auxiliary objectives."""

    fused_logits: torch.Tensor
    sensor_logits: torch.Tensor
    sensor_valid: torch.Tensor
    sensor_evidence: torch.Tensor
    sensor_attention: torch.Tensor
    effective_time_valid: torch.Tensor


class TransientQuery360Head(nn.Module):
    """Hierarchical temporal-then-sensor TransientQuery classifier."""

    def __init__(
        self,
        feature_dim: int,
        *,
        num_sensors: int = 4,
        num_roles: int = 3,
        model_dim: int = 256,
        num_heads: int = 8,
        depth: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or num_sensors <= 0 or num_roles < 2:
            raise ValueError("feature_dim/sensors must be positive and roles >= 2.")
        if depth != 2:
            raise ValueError("The matched TransientQuery head has exactly 2 blocks.")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.feature_dim = int(feature_dim)
        self.num_sensors = int(num_sensors)
        self.num_roles = int(num_roles)
        self.model_dim = int(model_dim)

        # Projection is deliberately shared across all sensors and roles.
        self.input_projection = nn.Linear(feature_dim, model_dim)
        # All roles are sensor-specific.  In particular, role 2 need not mean
        # the same acquisition product for S2/L89/EMIT/S5P.
        self.sensor_role_embedding = nn.Embedding(
            num_sensors * num_roles, model_dim
        )
        self.temporal_input_norm = nn.LayerNorm(model_dim)
        self.blocks = nn.ModuleList(
            [
                CurrentQueryBlock(
                    model_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.temporal_output_norm = nn.LayerNorm(model_dim)

        self.sensor_embedding = nn.Embedding(num_sensors, model_dim)
        self.sensor_score = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.fusion_norm = nn.LayerNorm(model_dim)
        self.fused_classifier = nn.Linear(model_dim, 1)
        self.sensor_classifiers = nn.ModuleList(
            [nn.Linear(model_dim, 1) for _ in range(num_sensors)]
        )

    def _validate_inputs(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if features.ndim != 4:
            raise ValueError("features must have shape [B,S,T,D].")
        batch, sensors, roles, feature_dim = features.shape
        expected = (
            batch,
            self.num_sensors,
            self.num_roles,
            self.feature_dim,
        )
        if tuple(features.shape) != expected:
            raise ValueError(
                f"features shape={tuple(features.shape)}, expected={expected}."
            )
        if valid_mask.shape != (batch, sensors, roles):
            raise ValueError("valid_mask shape must match features [B,S,T].")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have boolean dtype.")
        sensor_valid = valid_mask[:, :, 0]
        if not sensor_valid.any(dim=1).all():
            bad = torch.nonzero(
                ~sensor_valid.any(dim=1), as_tuple=False
            ).flatten()
            raise ValueError(
                "Every row needs at least one valid sensor t0; "
                f"invalid rows={bad.tolist()[:20]}."
            )

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        arm: str = "transient_query",
        donor_indices: Optional[torch.Tensor] = None,
    ) -> TransientQueryOutput:
        self._validate_inputs(features, valid_mask)
        working_features, effective_valid = prepare_arm_inputs(
            features,
            valid_mask,
            arm=arm,
            training=self.training,
            donor_indices=donor_indices,
        )
        batch_size = int(features.shape[0])
        sensor_valid = effective_valid[:, :, 0]

        # Execute the same dense shared projection for every arm, including
        # current_only's masked history.
        context = self.input_projection(working_features)
        sensor_role_index = torch.arange(
            self.num_sensors * self.num_roles,
            device=features.device,
            dtype=torch.long,
        ).reshape(1, self.num_sensors, self.num_roles)
        context = context + self.sensor_role_embedding(sensor_role_index)
        context = self.temporal_input_norm(context)

        evidence = context.new_zeros(
            (batch_size, self.num_sensors, self.model_dim)
        )
        for sensor_index in range(self.num_sensors):
            active_rows = torch.nonzero(
                sensor_valid[:, sensor_index], as_tuple=False
            ).flatten()
            if active_rows.numel() == 0:
                continue
            sensor_context = context[active_rows, sensor_index]
            sensor_mask = effective_valid[active_rows, sensor_index]
            query = sensor_context[:, 0:1]
            for block in self.blocks:
                query = block(query, sensor_context, sensor_mask)
            evidence[active_rows, sensor_index] = self.temporal_output_norm(
                query[:, 0]
            )

        sensor_index = torch.arange(
            self.num_sensors, device=features.device, dtype=torch.long
        ).reshape(1, -1)
        score_input = evidence + self.sensor_embedding(sensor_index)
        scores = self.sensor_score(score_input).squeeze(-1)
        scores = scores.masked_fill(~sensor_valid, -torch.inf)
        attention = torch.softmax(scores, dim=1)
        # Explicitly clear masked positions for easy audit and protection from
        # future changes to the softmax masking implementation.
        attention = torch.where(
            sensor_valid, attention, torch.zeros_like(attention)
        )
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(attention.dtype).tiny
        )
        fused = torch.sum(evidence * attention.unsqueeze(-1), dim=1)
        fused_logits = self.fused_classifier(self.fusion_norm(fused)).squeeze(-1)

        sensor_logits = torch.stack(
            [
                classifier(evidence[:, sensor]).squeeze(-1)
                for sensor, classifier in enumerate(self.sensor_classifiers)
            ],
            dim=1,
        )
        # Invalid sensor logits must never leak bias-only predictions into a
        # downstream consumer.  The loss also masks these positions.
        sensor_logits = torch.where(
            sensor_valid, sensor_logits, torch.zeros_like(sensor_logits)
        )
        return TransientQueryOutput(
            fused_logits=fused_logits,
            sensor_logits=sensor_logits,
            sensor_valid=sensor_valid,
            sensor_evidence=evidence,
            sensor_attention=attention,
            effective_time_valid=effective_valid,
        )


def build_history_shuffle_donors(
    valid_mask: torch.Tensor,
    event_ids: Sequence[str],
    *,
    plume_ids: Optional[Sequence[str]] = None,
    seed: int,
) -> torch.Tensor:
    """Build deterministic per-row/per-sensor cross-event history donors.

    The returned tensor has shape ``[N,S]``.  ``-1`` means that the target has
    no valid history and therefore needs no donor.  A selected donor:

    * has a valid t0 for the same sensor;
    * has every historical role that is valid for the target;
    * belongs to a different event and, when supplied, a different plume.

    Preserving the target validity pattern prevents sensor/time availability
    from becoming a confound in the shuffle control.
    """

    if valid_mask.ndim != 3 or valid_mask.shape[2] < 2:
        raise ValueError("valid_mask must have shape [N,S,T] with T >= 2.")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean.")
    rows, sensors, _ = valid_mask.shape
    if len(event_ids) != rows:
        raise ValueError("event_ids must have one entry per row.")
    if plume_ids is not None and len(plume_ids) != rows:
        raise ValueError("plume_ids must have one entry per row.")

    mask = valid_mask.detach().cpu()
    events = [str(value) for value in event_ids]
    plumes = (
        [str(value) for value in plume_ids]
        if plume_ids is not None
        else None
    )
    donors = torch.full((rows, sensors), -1, dtype=torch.long)
    rng = random.Random(int(seed))

    # Index valid donor rows by their exact history bit pattern.  For the
    # 360 m three-role cache there are only four patterns per sensor, turning
    # the former target-by-candidate scan into O(N*S) indexing plus short
    # deterministic rejection walks for event/plume exclusions.
    history_roles = int(mask.shape[2] - 1)
    exact_pattern_rows: list[dict[int, list[int]]] = [
        {} for _ in range(sensors)
    ]
    for sensor in range(sensors):
        for row in range(rows):
            if not bool(mask[row, sensor, 0]):
                continue
            pattern = 0
            for role_offset in range(history_roles):
                if bool(mask[row, sensor, role_offset + 1]):
                    pattern |= 1 << role_offset
            exact_pattern_rows[sensor].setdefault(pattern, []).append(row)

    eligible_pattern_rows: list[dict[int, list[int]]] = [
        {} for _ in range(sensors)
    ]
    for sensor in range(sensors):
        observed_patterns = exact_pattern_rows[sensor]
        for required_pattern in range(1, 1 << history_roles):
            eligible: list[int] = []
            for candidate_pattern, candidate_rows in observed_patterns.items():
                if candidate_pattern & required_pattern == required_pattern:
                    eligible.extend(candidate_rows)
            eligible_pattern_rows[sensor][required_pattern] = eligible

    for sensor in range(sensors):
        for target in range(rows):
            target_history = mask[target, sensor, 1:]
            if not bool(mask[target, sensor, 0]) or not bool(target_history.any()):
                continue
            required_pattern = 0
            for role_offset, is_valid in enumerate(target_history.tolist()):
                if is_valid:
                    required_pattern |= 1 << role_offset
            candidates = eligible_pattern_rows[sensor][required_pattern]
            if not candidates:
                pattern = target_history.to(torch.int8).tolist()
                raise ValueError(
                    "No valid cross-event history donor for "
                    f"row={target}, sensor={sensor}, history_pattern={pattern}."
                )
            start = rng.randrange(len(candidates))
            selected = -1
            for offset in range(len(candidates)):
                candidate = candidates[(start + offset) % len(candidates)]
                if events[candidate] == events[target]:
                    continue
                if plumes is not None and plumes[candidate] == plumes[target]:
                    continue
                selected = candidate
                break
            if selected < 0:
                pattern = target_history.to(torch.int8).tolist()
                raise ValueError(
                    "No cross-event/cross-plume donor after identity exclusion "
                    f"for row={target}, sensor={sensor}, "
                    f"history_pattern={pattern}."
                )
            donors[target, sensor] = selected

    return donors


def apply_history_shuffle(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    donor_indices: torch.Tensor,
    *,
    event_ids: Optional[Sequence[str]] = None,
    plume_ids: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Replace valid history features while retaining t0 and validity exactly."""

    if features.ndim != 4 or valid_mask.shape != features.shape[:3]:
        raise ValueError("Expected features [N,S,T,D] and matching valid_mask.")
    rows, sensors, roles, _ = features.shape
    if donor_indices.shape != (rows, sensors):
        raise ValueError("donor_indices must have shape [N,S].")
    if event_ids is not None and len(event_ids) != rows:
        raise ValueError("event_ids must have one entry per row.")
    if plume_ids is not None and len(plume_ids) != rows:
        raise ValueError("plume_ids must have one entry per row.")

    donors = donor_indices.detach().cpu()
    device_donors = donor_indices.to(device=valid_mask.device)
    shuffled = features.clone()
    for sensor in range(sensors):
        for role in range(1, roles):
            targets = torch.nonzero(
                valid_mask[:, sensor, role], as_tuple=False
            ).flatten()
            if targets.numel() == 0:
                continue
            selected = device_donors[targets, sensor]
            if bool((selected < 0).any()) or bool((selected >= rows).any()):
                raise ValueError(
                    f"Missing/out-of-range donor for sensor={sensor}, role={role}."
                )
            if not valid_mask[selected, sensor, role].all():
                raise ValueError(
                    f"Donor lacks valid history for sensor={sensor}, role={role}."
                )
            shuffled[targets, sensor, role] = features[
                selected, sensor, role
            ]

    if event_ids is not None:
        events = [str(value) for value in event_ids]
        plumes = (
            [str(value) for value in plume_ids]
            if plume_ids is not None
            else None
        )
        for target in range(rows):
            for sensor in range(sensors):
                donor = int(donors[target, sensor])
                if donor < 0:
                    continue
                if events[target] == events[donor]:
                    raise ValueError("History donor belongs to the target event.")
                if plumes is not None and plumes[target] == plumes[donor]:
                    raise ValueError("History donor belongs to the target plume.")

    if not torch.equal(shuffled[:, :, 0], features[:, :, 0]):
        raise RuntimeError("History shuffling modified t0.")
    return shuffled


def prepare_arm_inputs(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    arm: str,
    training: bool,
    donor_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply only the input intervention associated with an experiment arm."""

    if arm not in ARM_NAMES:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARM_NAMES}.")
    if features.ndim != 4 or valid_mask.shape != features.shape[:3]:
        raise ValueError("Expected features [B,S,T,D] and valid_mask [B,S,T].")
    effective_valid = valid_mask.clone()
    working_features = features
    if arm == "current_only":
        effective_valid[:, :, 1:] = False
    elif arm == "scale_aware_transient_query":
        if effective_valid.shape[1] != 4:
            raise ValueError(
                "scale_aware_transient_query expects the fixed four-sensor "
                "order (s2,l89,emit,s5p)."
            )
        effective_valid[:, 3, 1:] = False
    elif arm == "history_shuffle_train" and training:
        if donor_indices is None:
            raise ValueError(
                "history_shuffle_train requires donor_indices while training."
            )
        working_features = apply_history_shuffle(
            features, valid_mask, donor_indices
        )
    # history_shuffle_train is deliberately coherent at validation/test time.
    return working_features, effective_valid


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    fused: torch.Tensor
    auxiliary: torch.Tensor
    valid_sensor_count: int


def transient_query_loss(
    output: TransientQueryOutput,
    labels: torch.Tensor,
    *,
    auxiliary_weight: float = 0.3,
    pos_weight: Optional[torch.Tensor | float] = None,
) -> LossBreakdown:
    """Fused BCE + weighted mean BCE over valid sensor evidence only."""

    labels = labels.to(
        device=output.fused_logits.device,
        dtype=output.fused_logits.dtype,
    ).reshape(-1)
    if labels.shape != output.fused_logits.shape:
        raise ValueError("labels shape must match fused logits [B].")
    if not output.sensor_valid.any(dim=1).all():
        raise ValueError("Every row must contain at least one valid sensor.")
    weight_tensor: Optional[torch.Tensor]
    if pos_weight is None:
        weight_tensor = None
    else:
        weight_tensor = torch.as_tensor(
            pos_weight,
            device=labels.device,
            dtype=labels.dtype,
        )
    fused = F.binary_cross_entropy_with_logits(
        output.fused_logits,
        labels,
        pos_weight=weight_tensor,
    )
    expanded_labels = labels[:, None].expand_as(output.sensor_logits)
    auxiliary_per_sensor = F.binary_cross_entropy_with_logits(
        output.sensor_logits,
        expanded_labels,
        pos_weight=weight_tensor,
        reduction="none",
    )
    valid = output.sensor_valid.to(auxiliary_per_sensor.dtype)
    valid_count = int(output.sensor_valid.sum().item())
    auxiliary = (auxiliary_per_sensor * valid).sum() / valid.sum().clamp_min(1)
    total = fused + float(auxiliary_weight) * auxiliary
    return LossBreakdown(
        total=total,
        fused=fused,
        auxiliary=auxiliary,
        valid_sensor_count=valid_count,
    )


def build_matched_models(
    model_kwargs: Mapping[str, Any],
    *,
    seed: int,
    arms: Sequence[str] = ARM_NAMES,
) -> tuple[dict[str, TransientQuery360Head], str, dict[str, Any]]:
    """Construct arm models with byte-identical initialization."""

    requested = tuple(str(arm) for arm in arms)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("arms must be non-empty and unique.")
    unknown = sorted(set(requested) - set(ARM_NAMES))
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        prototype = TransientQuery360Head(**dict(model_kwargs))
    initial_state = copy.deepcopy(prototype.state_dict())
    initialization_hash = state_dict_sha256(initial_state)
    signature = model_parameter_signature(prototype)

    models: dict[str, TransientQuery360Head] = {}
    for arm in requested:
        model = TransientQuery360Head(**dict(model_kwargs))
        model.load_state_dict(initial_state, strict=True)
        if state_dict_sha256(model.state_dict()) != initialization_hash:
            raise RuntimeError(f"{arm} failed matched initialization.")
        if model_parameter_signature(model) != signature:
            raise RuntimeError(f"{arm} parameter signature differs.")
        models[arm] = model
    return models, initialization_hash, signature


def binary_classification_metrics(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    probabilities: Sequence[float] | np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    """Compute robust binary metrics, returning ``None`` when undefined."""

    target = np.asarray(
        labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    ).reshape(-1)
    probability = np.asarray(
        probabilities.detach().cpu().numpy()
        if torch.is_tensor(probabilities)
        else probabilities
    ).reshape(-1)
    if target.shape != probability.shape:
        raise ValueError("labels and probabilities must have the same shape.")
    if not np.isfinite(probability).all():
        raise ValueError("probabilities contain non-finite values.")
    if target.size == 0:
        return {
            "rows": 0,
            "positives": 0,
            "negatives": 0,
            "ap": None,
            "auc": None,
            "macro_f1_at_0_5": None,
            "balanced_accuracy_at_0_5": None,
            "best_macro_f1": None,
            "best_macro_f1_threshold": None,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    target = target.astype(np.int64)
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("labels must be binary.")
    prediction = (probability >= 0.5).astype(np.int64)
    tn = int(np.sum((target == 0) & (prediction == 0)))
    fp = int(np.sum((target == 0) & (prediction == 1)))
    fn = int(np.sum((target == 1) & (prediction == 0)))
    tp = int(np.sum((target == 1) & (prediction == 1)))
    both_classes = np.unique(target).size == 2
    has_positive = bool(np.any(target == 1))
    best_threshold = best_macro_f1_threshold(target, probability)
    best_prediction = (probability >= best_threshold).astype(np.int64)
    best_macro_f1 = f1_score(
        target,
        best_prediction,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )
    return {
        "rows": int(target.size),
        "positives": int(np.sum(target == 1)),
        "negatives": int(np.sum(target == 0)),
        "ap": (
            float(average_precision_score(target, probability))
            if has_positive
            else None
        ),
        "auc": (
            float(roc_auc_score(target, probability))
            if both_classes
            else None
        ),
        "macro_f1_at_0_5": float(
            f1_score(
                target,
                prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy_at_0_5": (
            float(balanced_accuracy_score(target, prediction))
            if both_classes
            else None
        ),
        "best_macro_f1": float(best_macro_f1),
        "best_macro_f1_threshold": float(best_threshold),
        "pred_positive_rate_at_0_5": float(prediction.mean()),
        "probability_mean": float(probability.mean()),
        "probability_std": float(probability.std()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def best_macro_f1_threshold(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    probabilities: Sequence[float] | np.ndarray | torch.Tensor,
) -> float:
    """Select a deterministic threshold maximizing binary macro-F1 in O(N log N).

    This is a validation diagnostic only.  The preregistered fixed-threshold
    result remains ``macro_f1_at_0_5``.
    """

    target = np.asarray(
        labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels,
        dtype=np.int64,
    ).reshape(-1)
    probability = np.asarray(
        probabilities.detach().cpu().numpy()
        if torch.is_tensor(probabilities)
        else probabilities,
        dtype=np.float64,
    ).reshape(-1)
    if target.shape != probability.shape or target.size == 0:
        raise ValueError("labels/probabilities must be non-empty matching vectors.")
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("labels must be binary.")
    if not np.isfinite(probability).all():
        raise ValueError("probabilities contain non-finite values.")

    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    sorted_target = target[order]
    cumulative_tp = np.cumsum(sorted_target == 1)
    cumulative_fp = np.cumsum(sorted_target == 0)
    group_ends = np.flatnonzero(
        np.r_[sorted_probability[1:] != sorted_probability[:-1], True]
    )
    positives = int(np.sum(target == 1))
    negatives = int(np.sum(target == 0))

    no_positive_prediction = np.nextafter(sorted_probability[0], np.inf)
    thresholds = np.r_[no_positive_prediction, sorted_probability[group_ends]]
    tp = np.r_[0, cumulative_tp[group_ends]].astype(np.float64)
    fp = np.r_[0, cumulative_fp[group_ends]].astype(np.float64)
    fn = positives - tp
    tn = negatives - fp
    positive_denominator = 2 * tp + fp + fn
    negative_denominator = 2 * tn + fp + fn
    f1_positive = np.divide(
        2 * tp,
        positive_denominator,
        out=np.zeros_like(tp),
        where=positive_denominator > 0,
    )
    f1_negative = np.divide(
        2 * tn,
        negative_denominator,
        out=np.zeros_like(tn),
        where=negative_denominator > 0,
    )
    macro_f1 = (f1_positive + f1_negative) / 2.0
    tpr = np.divide(
        tp,
        positives,
        out=np.zeros_like(tp),
        where=positives > 0,
    )
    tnr = np.divide(
        tn,
        negatives,
        out=np.zeros_like(tn),
        where=negatives > 0,
    )
    balanced = (tpr + tnr) / 2.0
    best_value = float(macro_f1.max())
    candidates = np.flatnonzero(np.isclose(macro_f1, best_value))
    best_index = max(
        candidates.tolist(),
        key=lambda index: (
            float(balanced[index]),
            -abs(float(thresholds[index]) - 0.5),
            -index,
        ),
    )
    return float(thresholds[best_index])


def stratified_binary_metrics(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    probabilities: Sequence[float] | np.ndarray | torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    sensor_names: Sequence[str] = DEFAULT_SENSOR_NAMES,
) -> dict[str, Any]:
    """Overall, single/multisensor, and exact-availability metrics."""

    if valid_mask.ndim != 3:
        raise ValueError("valid_mask must have shape [N,S,T].")
    rows, sensors, _ = valid_mask.shape
    if len(sensor_names) != sensors:
        raise ValueError("sensor_names length must match valid_mask sensors.")
    target = np.asarray(
        labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    ).reshape(-1)
    probability = np.asarray(
        probabilities.detach().cpu().numpy()
        if torch.is_tensor(probabilities)
        else probabilities
    ).reshape(-1)
    if len(target) != rows or len(probability) != rows:
        raise ValueError("labels/probabilities rows must match valid_mask.")
    sensor_valid = valid_mask[:, :, 0].detach().cpu().numpy().astype(bool)
    sensor_count = sensor_valid.sum(axis=1)

    def subset(mask: np.ndarray) -> dict[str, Any]:
        return binary_classification_metrics(target[mask], probability[mask])

    availability: dict[str, dict[str, Any]] = {}
    signatures = [
        "+".join(
            str(sensor_names[index])
            for index, present in enumerate(row)
            if present
        )
        or "none"
        for row in sensor_valid
    ]
    for signature in sorted(set(signatures)):
        mask = np.asarray(
            [value == signature for value in signatures], dtype=bool
        )
        availability[signature] = subset(mask)

    return {
        "overall": binary_classification_metrics(target, probability),
        "sensor_count": {
            "single_sensor": subset(sensor_count == 1),
            "multisensor": subset(sensor_count >= 2),
        },
        "availability": availability,
    }


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model has no parameters.") from error


def _model_dtype(model: nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration as error:
        raise ValueError("model has no parameters.") from error


def train_head_epoch(
    model: TransientQuery360Head,
    optimizer: torch.optim.Optimizer,
    *,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    labels: torch.Tensor,
    arm: str,
    event_ids: Sequence[str],
    plume_ids: Optional[Sequence[str]] = None,
    batch_size: int,
    seed: int,
    epoch: int,
    auxiliary_weight: float = 0.3,
    pos_weight: Optional[torch.Tensor | float] = None,
    grad_clip: float = 1.0,
    max_steps: Optional[int] = None,
) -> dict[str, Any]:
    """Train one deterministic cached-feature epoch for one matched arm."""

    if features.ndim != 4 or valid_mask.shape != features.shape[:3]:
        raise ValueError("Expected features [N,S,T,D] and valid_mask [N,S,T].")
    rows = int(features.shape[0])
    if labels.numel() != rows or len(event_ids) != rows:
        raise ValueError("labels/event_ids must have one entry per feature row.")
    if plume_ids is not None and len(plume_ids) != rows:
        raise ValueError("plume_ids must have one entry per feature row.")

    set_deterministic_seed(int(seed) + int(epoch))
    donor_hash: Optional[str] = None
    epoch_features = features
    effective_arm = arm
    if arm == "history_shuffle_train":
        donors = build_history_shuffle_donors(
            valid_mask,
            event_ids,
            plume_ids=plume_ids,
            seed=int(seed) + 100_003 * int(epoch),
        )
        epoch_features = apply_history_shuffle(
            features,
            valid_mask,
            donors,
            event_ids=event_ids,
            plume_ids=plume_ids,
        )
        donor_hash = tensor_sha256(donors)
        # The intervention has already been applied globally, allowing donors
        # outside the current minibatch while the model sees the same TQ path.
        effective_arm = "transient_query"

    batches = fixed_epoch_batches(
        rows,
        batch_size=batch_size,
        seed=seed,
        epoch=epoch,
        shuffle=True,
    )
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive when supplied.")
        batches = batches[: int(max_steps)]
    model.train()
    device = _model_device(model)
    dtype = _model_dtype(model)
    total_sum = 0.0
    fused_sum = 0.0
    auxiliary_sum = 0.0
    seen = 0

    for indices in batches:
        batch_features = epoch_features[indices].to(device=device, dtype=dtype)
        batch_valid = valid_mask[indices].to(device=device)
        batch_labels = labels[indices].to(device=device, dtype=dtype)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch_features, batch_valid, arm=effective_arm)
        loss = transient_query_loss(
            output,
            batch_labels,
            auxiliary_weight=auxiliary_weight,
            pos_weight=pos_weight,
        )
        if not torch.isfinite(loss.total):
            raise RuntimeError("Non-finite TransientQuery training loss.")
        loss.total.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        count = int(indices.numel())
        total_sum += float(loss.total.detach()) * count
        fused_sum += float(loss.fused.detach()) * count
        auxiliary_sum += float(loss.auxiliary.detach()) * count
        seen += count

    if seen == 0:
        raise RuntimeError("No training rows were processed.")
    return {
        "loss": total_sum / seen,
        "fused_loss": fused_sum / seen,
        "auxiliary_loss": auxiliary_sum / seen,
        "rows": seen,
        "steps": len(batches),
        "history_donor_sha256": donor_hash,
    }


def evaluate_head(
    model: TransientQuery360Head,
    *,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    labels: torch.Tensor,
    arm: str,
    batch_size: int,
    sensor_names: Sequence[str] = DEFAULT_SENSOR_NAMES,
    auxiliary_weight: float = 0.3,
    pos_weight: Optional[torch.Tensor | float] = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate coherently; history_shuffle_train is never shuffled here."""

    if features.ndim != 4 or valid_mask.shape != features.shape[:3]:
        raise ValueError("Expected features [N,S,T,D] and valid_mask [N,S,T].")
    rows = int(features.shape[0])
    if labels.numel() != rows:
        raise ValueError("labels must have one entry per feature row.")
    model.eval()
    device = _model_device(model)
    dtype = _model_dtype(model)
    probabilities: list[torch.Tensor] = []
    total_sum = 0.0
    fused_sum = 0.0
    auxiliary_sum = 0.0
    batches = fixed_epoch_batches(
        rows,
        batch_size=batch_size,
        seed=0,
        epoch=0,
        shuffle=False,
    )
    with torch.inference_mode():
        for indices in batches:
            batch_features = features[indices].to(device=device, dtype=dtype)
            batch_valid = valid_mask[indices].to(device=device)
            batch_labels = labels[indices].to(device=device, dtype=dtype)
            output = model(batch_features, batch_valid, arm=arm)
            loss = transient_query_loss(
                output,
                batch_labels,
                auxiliary_weight=auxiliary_weight,
                pos_weight=pos_weight,
            )
            count = int(indices.numel())
            total_sum += float(loss.total) * count
            fused_sum += float(loss.fused) * count
            auxiliary_sum += float(loss.auxiliary) * count
            probabilities.append(torch.sigmoid(output.fused_logits).cpu())
    probability = torch.cat(probabilities).numpy()
    metrics = stratified_binary_metrics(
        labels,
        probability,
        valid_mask,
        sensor_names=sensor_names,
    )
    metrics["loss"] = {
        "total": total_sum / rows,
        "fused": fused_sum / rows,
        "auxiliary": auxiliary_sum / rows,
    }
    metrics["evaluation_intervention"] = (
        "coherent_transient_query"
        if arm == "history_shuffle_train"
        else arm
    )
    return metrics, probability


__all__ = [
    "ARM_NAMES",
    "DEFAULT_SENSOR_NAMES",
    "CurrentQueryBlock",
    "LossBreakdown",
    "TransientQuery360Head",
    "TransientQueryOutput",
    "apply_history_shuffle",
    "best_macro_f1_threshold",
    "binary_classification_metrics",
    "build_history_shuffle_donors",
    "build_matched_models",
    "evaluate_head",
    "fixed_epoch_batches",
    "model_parameter_signature",
    "prepare_arm_inputs",
    "set_deterministic_seed",
    "state_dict_sha256",
    "stratified_binary_metrics",
    "tensor_sha256",
    "train_head_epoch",
    "transient_query_loss",
]
