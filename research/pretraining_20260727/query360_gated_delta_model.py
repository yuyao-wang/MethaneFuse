#!/usr/bin/env python3
"""Low-capacity time-then-sensor residual head for legacy 360 m features.

The cached Panopticon tensor has shape ``[batch, sensor, time, feature]``.
This module keeps both acquisition axes explicit without using a large
attention stack:

1. within each sensor, masked historical features form a learned temporal
   reference and a data-dependent gate controls the current-minus-history
   delta;
2. across sensors, a second masked gate fuses the per-sensor evidence; and
3. a zero-initialized linear residual is added to the existing checkpoint
   logit.

Consequently, the untrained module reproduces the supplied checkpoint exactly.
Missing roles and sensors cannot affect either gate or the resulting logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


GATED_DELTA_ARMS = (
    "gated_delta",
    "scale_aware_gated_delta",
)


def _masked_softmax(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    """Softmax with exact zeros, including rows with no valid entries."""

    if logits.shape != valid_mask.shape:
        raise ValueError("logits and valid_mask must have identical shapes.")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean.")
    masked = logits.masked_fill(~valid_mask, -1.0e4)
    weights = torch.softmax(masked, dim=dim)
    weights = weights * valid_mask.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0)


@dataclass(frozen=True)
class GatedDeltaOutput:
    """Loss-compatible output plus auditable axis-gating diagnostics."""

    fused_logits: torch.Tensor
    sensor_logits: torch.Tensor
    sensor_valid: torch.Tensor
    sensor_evidence: torch.Tensor
    sensor_attention: torch.Tensor
    effective_time_valid: torch.Tensor
    history_attention: torch.Tensor
    history_valid: torch.Tensor
    temporal_gate: torch.Tensor
    residual_fused_logits: torch.Tensor
    residual_sensor_logits: torch.Tensor
    base_fused_logits: Optional[torch.Tensor]


class GatedDelta360Head(nn.Module):
    """Gated temporal deltas followed by masked sensor fusion.

    ``bottleneck_dim=32`` gives roughly 52k trainable parameters for a
    768-dimensional Panopticon CLS feature, small enough to fit quickly while
    retaining a nonlinear, sample-dependent decision on each axis.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        num_sensors: int = 4,
        num_roles: int = 3,
        bottleneck_dim: int = 32,
        dropout: float = 0.05,
        residual_cap: float = 4.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if num_sensors <= 0:
            raise ValueError("num_sensors must be positive.")
        if num_roles < 2:
            raise ValueError("num_roles must be at least two.")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1).")
        if residual_cap <= 0:
            raise ValueError("residual_cap must be positive.")

        self.feature_dim = int(feature_dim)
        self.num_sensors = int(num_sensors)
        self.num_roles = int(num_roles)
        self.bottleneck_dim = int(bottleneck_dim)
        self.residual_cap = float(residual_cap)

        # All feature transforms are shared across sensors. Sensor identity is
        # carried by a tiny embedding and by per-sensor gate biases.
        self.input_norm = nn.LayerNorm(feature_dim)
        self.current_projection = nn.Linear(
            feature_dim, bottleneck_dim, bias=False
        )
        self.delta_projection = nn.Linear(
            feature_dim, bottleneck_dim, bias=False
        )
        self.sensor_embedding = nn.Parameter(
            torch.zeros(num_sensors, bottleneck_dim)
        )

        # The learned logits distinguish the two historical roles for each
        # sensor. The sample-dependent scalar gate then decides how much of
        # that temporal delta to inject.
        self.history_role_logits = nn.Parameter(
            torch.zeros(num_sensors, num_roles - 1)
        )
        self.temporal_gate_network = nn.Linear(
            bottleneck_dim * 3, 1, bias=False
        )
        self.temporal_gate_bias = nn.Parameter(
            torch.full((num_sensors,), -1.0)
        )

        self.sensor_evidence_norm = nn.LayerNorm(bottleneck_dim)
        self.sensor_gate_network = nn.Linear(
            bottleneck_dim * 2, 1, bias=False
        )
        self.sensor_gate_bias = nn.Parameter(torch.zeros(num_sensors))
        self.fusion_norm = nn.LayerNorm(bottleneck_dim)
        self.dropout = nn.Dropout(float(dropout))

        # A few fixed-scale context values let the residual learn a small
        # calibration correction without requiring another MLP.
        fused_readout_dim = bottleneck_dim + 3
        self.fused_residual_classifier = nn.Linear(fused_readout_dim, 1)
        self.sensor_residual_classifiers = nn.ModuleList(
            [nn.Linear(bottleneck_dim, 1) for _ in range(num_sensors)]
        )

        # This is the central safety property: epoch zero is exactly the
        # supplied historical PTH boundary, not a randomly initialized head.
        nn.init.zeros_(self.fused_residual_classifier.weight)
        nn.init.zeros_(self.fused_residual_classifier.bias)
        for classifier in self.sensor_residual_classifiers:
            nn.init.zeros_(classifier.weight)
            nn.init.zeros_(classifier.bias)

    def _validate_inputs(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if features.ndim != 4:
            raise ValueError("features must have shape [B,S,T,D].")
        expected = (
            features.shape[0],
            self.num_sensors,
            self.num_roles,
            self.feature_dim,
        )
        if tuple(features.shape) != expected:
            raise ValueError(
                f"features shape={tuple(features.shape)}, expected={expected}."
            )
        if valid_mask.shape != features.shape[:3]:
            raise ValueError("valid_mask must match features [B,S,T].")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean.")
        if not valid_mask[:, :, 0].any(dim=1).all():
            raise ValueError("Every row needs at least one valid current sensor.")

    def _effective_mask(
        self,
        valid_mask: torch.Tensor,
        *,
        arm: str,
    ) -> torch.Tensor:
        if arm not in GATED_DELTA_ARMS:
            raise ValueError(
                f"Unknown arm {arm!r}; expected one of {GATED_DELTA_ARMS}."
            )
        effective = valid_mask.clone()
        if arm == "scale_aware_gated_delta":
            if self.num_sensors != 4:
                raise ValueError(
                    "scale-aware mode requires sensor order s2,l89,emit,s5p."
                )
            # Stored S5P history is a native ~10.5 km support, not a plume-
            # local 360 m crop. Keep its current product, but do not interpret
            # its history as a matched local temporal delta.
            effective[:, 3, 1:] = False
        return effective

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        arm: str = "scale_aware_gated_delta",
        base_fused_logits: Optional[torch.Tensor] = None,
        base_sensor_logits: Optional[torch.Tensor] = None,
    ) -> GatedDeltaOutput:
        self._validate_inputs(features, valid_mask)
        effective = self._effective_mask(valid_mask, arm=arm)
        batch_size = int(features.shape[0])
        sensor_valid = effective[:, :, 0]
        history_mask = effective[:, :, 1:] & sensor_valid.unsqueeze(-1)
        history_valid = history_mask.any(dim=-1)

        # Invalid values are removed before LayerNorm so even NaN/large
        # sentinel payloads cannot leak through multiplication by zero.
        safe = torch.where(
            effective.unsqueeze(-1), features, torch.zeros_like(features)
        )
        normalized = self.input_norm(safe)
        current = normalized[:, :, 0]

        role_logits = self.history_role_logits.reshape(
            1, self.num_sensors, self.num_roles - 1
        ).expand(batch_size, -1, -1)
        history_attention = _masked_softmax(
            role_logits, history_mask, dim=-1
        )
        history = (
            normalized[:, :, 1:]
            * history_attention.unsqueeze(-1)
        ).sum(dim=2)
        history = torch.where(
            history_valid.unsqueeze(-1), history, current
        )

        current_low = F.gelu(self.current_projection(current))
        delta_low = F.gelu(self.delta_projection(current - history))
        delta_low = torch.where(
            history_valid.unsqueeze(-1),
            delta_low,
            torch.zeros_like(delta_low),
        )
        gate_input = torch.cat(
            (current_low, delta_low, delta_low.abs()), dim=-1
        )
        temporal_gate = torch.sigmoid(
            self.temporal_gate_network(gate_input).squeeze(-1)
            + self.temporal_gate_bias.reshape(1, self.num_sensors)
        )
        temporal_gate = temporal_gate * history_valid.to(temporal_gate.dtype)

        evidence = (
            current_low
            + temporal_gate.unsqueeze(-1) * delta_low
            + self.sensor_embedding.reshape(
                1, self.num_sensors, self.bottleneck_dim
            )
        )
        evidence = self.sensor_evidence_norm(evidence)
        evidence = torch.where(
            sensor_valid.unsqueeze(-1), evidence, torch.zeros_like(evidence)
        )

        sensor_gate_input = torch.cat((evidence, delta_low.abs()), dim=-1)
        sensor_gate_logits = (
            self.sensor_gate_network(sensor_gate_input).squeeze(-1)
            + self.sensor_gate_bias.reshape(1, self.num_sensors)
        )
        sensor_attention = _masked_softmax(
            sensor_gate_logits, sensor_valid, dim=-1
        )
        fused = (
            evidence * sensor_attention.unsqueeze(-1)
        ).sum(dim=1)
        fused = self.fusion_norm(fused)

        valid_sensor_fraction = (
            sensor_valid.to(fused.dtype).mean(dim=1, keepdim=True)
        )
        valid_history_fraction = (
            history_mask.to(fused.dtype).mean(
                dim=(1, 2), keepdim=False
            ).unsqueeze(-1)
        )
        if base_fused_logits is None:
            base_context = fused.new_zeros(batch_size, 1)
        else:
            if base_fused_logits.shape != (batch_size,):
                raise ValueError("base_fused_logits must have shape [B].")
            base_context = torch.tanh(
                base_fused_logits.to(
                    device=fused.device, dtype=fused.dtype
                )
            ).unsqueeze(-1)
        fused_readout = torch.cat(
            (
                self.dropout(fused),
                base_context,
                valid_sensor_fraction,
                valid_history_fraction,
            ),
            dim=-1,
        )
        raw_residual = self.fused_residual_classifier(
            fused_readout
        ).squeeze(-1)
        residual_fused_logits = self.residual_cap * torch.tanh(
            raw_residual / self.residual_cap
        )
        if base_fused_logits is None:
            fused_logits = residual_fused_logits
        else:
            fused_logits = residual_fused_logits + base_fused_logits.to(
                device=fused.device, dtype=fused.dtype
            )

        raw_sensor_residual = torch.stack(
            [
                classifier(evidence[:, sensor]).squeeze(-1)
                for sensor, classifier in enumerate(
                    self.sensor_residual_classifiers
                )
            ],
            dim=1,
        )
        residual_sensor_logits = self.residual_cap * torch.tanh(
            raw_sensor_residual / self.residual_cap
        )
        if base_sensor_logits is None:
            sensor_logits = residual_sensor_logits
        else:
            if base_sensor_logits.shape != (batch_size, self.num_sensors):
                raise ValueError(
                    "base_sensor_logits must have shape [B,S]."
                )
            sensor_logits = residual_sensor_logits + base_sensor_logits.to(
                device=evidence.device, dtype=evidence.dtype
            )
        sensor_logits = torch.where(
            sensor_valid, sensor_logits, torch.zeros_like(sensor_logits)
        )
        residual_sensor_logits = torch.where(
            sensor_valid,
            residual_sensor_logits,
            torch.zeros_like(residual_sensor_logits),
        )

        return GatedDeltaOutput(
            fused_logits=fused_logits,
            sensor_logits=sensor_logits,
            sensor_valid=sensor_valid,
            sensor_evidence=evidence,
            sensor_attention=sensor_attention,
            effective_time_valid=effective,
            history_attention=history_attention,
            history_valid=history_valid,
            temporal_gate=temporal_gate,
            residual_fused_logits=residual_fused_logits,
            residual_sensor_logits=residual_sensor_logits,
            base_fused_logits=base_fused_logits,
        )


__all__ = [
    "GATED_DELTA_ARMS",
    "GatedDelta360Head",
    "GatedDeltaOutput",
    "_masked_softmax",
]
