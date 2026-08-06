#!/usr/bin/env python3
"""Factorized time-by-sensor classifier for cached or online 360 m features.

``TwoAxisQuery360Head`` is a small downstream module for Panopticon features
with shape ``[batch, sensor, time, feature]``.  It keeps the two acquisition
axes explicit and evaluates both non-commuting factorization orders:

``time -> sensor``
    A current token first queries the history of the same sensor.  The
    resulting per-sensor evidence is then fused across the available sensors.

``sensor -> time``
    Available sensors are first fused independently at every temporal role.
    The fused current role then queries the fused historical roles.

The two representations are combined by a sample-dependent gate.  This is
deliberately different from concatenating temporal channels before Panopticon,
and from scalar pooling of already temporally-fused sensor tokens.  Missing
sensors/times are masked on both axes.

The output contract is compatible with ``transient_query_loss`` from
``query360_model.py``; the existing fused BCE and masked per-sensor auxiliary
loss can therefore be reused without changing a trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


TWO_AXIS_ARMS = (
    "current_only",
    "two_axis_query",
    "scale_aware_two_axis_query",
)


class MaskedCurrentQueryBlock(nn.Module):
    """Pre-norm cross-attention block for one event-conditioned query."""

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
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if query.ndim != 3 or query.shape[1] != 1:
            raise ValueError("query must have shape [N,1,D].")
        if context.ndim != 3 or context.shape[0] != query.shape[0]:
            raise ValueError("context must have shape [N,K,D].")
        if valid_mask.shape != context.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape [N,K].")
        if not valid_mask.any(dim=1).all():
            raise ValueError("Every query row needs at least one valid context token.")
        attended, attention = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=~valid_mask,
            need_weights=bool(return_attention),
            average_attn_weights=True,
        )
        query = query + self.attention_dropout(attended)
        query = query + self.ffn(self.ffn_norm(query))
        if attention is not None:
            attention = attention[:, 0]
            attention = torch.where(
                valid_mask, attention, torch.zeros_like(attention)
            )
        return query, attention


@dataclass(frozen=True)
class TwoAxisQueryOutput:
    """Main output plus diagnostics for the two factorization orders."""

    fused_logits: torch.Tensor
    sensor_logits: torch.Tensor
    sensor_valid: torch.Tensor
    sensor_evidence: torch.Tensor
    sensor_attention: torch.Tensor
    effective_time_valid: torch.Tensor
    time_then_sensor_logits: torch.Tensor
    sensor_then_time_logits: torch.Tensor
    axis_gate: torch.Tensor
    role_valid: torch.Tensor
    base_fused_logits: Optional[torch.Tensor]


class TwoAxisQuery360Head(nn.Module):
    """Missingness-aware time-by-sensor axial readout for Panopticon features."""

    def __init__(
        self,
        feature_dim: int,
        *,
        num_sensors: int = 4,
        num_roles: int = 3,
        model_dim: int = 256,
        num_heads: int = 8,
        temporal_depth: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or num_sensors <= 0 or num_roles < 2:
            raise ValueError("feature_dim/sensors must be positive and roles >= 2.")
        if temporal_depth < 1:
            raise ValueError("temporal_depth must be positive.")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.feature_dim = int(feature_dim)
        self.num_sensors = int(num_sensors)
        self.num_roles = int(num_roles)
        self.model_dim = int(model_dim)

        self.input_projection = nn.Linear(feature_dim, model_dim)
        self.sensor_embedding = nn.Embedding(num_sensors, model_dim)
        self.role_embedding = nn.Embedding(num_roles, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)

        # Order A: time within sensor, then sensors within query.
        self.t2s_temporal_blocks = nn.ModuleList(
            [
                MaskedCurrentQueryBlock(
                    model_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(int(temporal_depth))
            ]
        )
        self.t2s_sensor_block = MaskedCurrentQueryBlock(
            model_dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.t2s_sensor_query = nn.Parameter(torch.zeros(1, 1, model_dim))

        # Order B: sensors within temporal role, then roles within query.
        self.s2t_role_sensor_block = MaskedCurrentQueryBlock(
            model_dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.s2t_role_queries = nn.Parameter(
            torch.zeros(1, num_roles, model_dim)
        )
        self.s2t_temporal_blocks = nn.ModuleList(
            [
                MaskedCurrentQueryBlock(
                    model_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(int(temporal_depth))
            ]
        )

        self.sensor_output_norm = nn.LayerNorm(model_dim)
        self.role_output_norm = nn.LayerNorm(model_dim)
        self.axis_output_norm = nn.LayerNorm(model_dim)
        self.axis_gate = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.fused_classifier = nn.Linear(model_dim, 1)
        self.t2s_classifier = nn.Linear(model_dim, 1)
        self.s2t_classifier = nn.Linear(model_dim, 1)
        self.sensor_classifiers = nn.ModuleList(
            [nn.Linear(model_dim, 1) for _ in range(num_sensors)]
        )

        nn.init.normal_(self.t2s_sensor_query, std=0.02)
        nn.init.normal_(self.s2t_role_queries, std=0.02)
        # When an old concat-temporal checkpoint supplies a base logit, the
        # new two-axis path must begin as an exact no-op.  Zero initialization
        # makes this a residual adapter instead of destroying the strong
        # 36-channel S2 decision boundary at step zero.
        nn.init.zeros_(self.fused_classifier.weight)
        nn.init.zeros_(self.fused_classifier.bias)
        for classifier in self.sensor_classifiers:
            nn.init.zeros_(classifier.weight)
            nn.init.zeros_(classifier.bias)

    def _validate_inputs(
        self, features: torch.Tensor, valid_mask: torch.Tensor
    ) -> None:
        expected = (
            features.shape[0],
            self.num_sensors,
            self.num_roles,
            self.feature_dim,
        )
        if features.ndim != 4 or tuple(features.shape) != expected:
            raise ValueError(
                "features must have shape "
                f"[B,{self.num_sensors},{self.num_roles},{self.feature_dim}], "
                f"got {tuple(features.shape)}."
            )
        if valid_mask.shape != features.shape[:3] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape [B,S,T].")
        if not valid_mask[:, :, 0].any(dim=1).all():
            raise ValueError("Every row needs at least one valid current sensor.")

    def _prepare_mask(
        self, valid_mask: torch.Tensor, *, arm: str
    ) -> torch.Tensor:
        if arm not in TWO_AXIS_ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected {TWO_AXIS_ARMS}.")
        effective = valid_mask.clone()
        if arm == "current_only":
            effective[:, :, 1:] = False
        elif arm == "scale_aware_two_axis_query":
            if self.num_sensors != 4:
                raise ValueError(
                    "scale-aware mode requires sensor order s2,l89,emit,s5p."
                )
            # The stored S5P context is roughly 10.5 km, not a matched 360 m
            # crop, so its history is not treated as plume-local evidence.
            effective[:, 3, 1:] = False
        return effective

    @staticmethod
    def _masked_mean(
        values: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        weight = valid_mask.to(values.dtype).unsqueeze(-1)
        return (values * weight).sum(dim=1, keepdim=True) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        arm: str = "two_axis_query",
        donor_indices: Optional[torch.Tensor] = None,
        base_fused_logits: Optional[torch.Tensor] = None,
        base_sensor_logits: Optional[torch.Tensor] = None,
    ) -> TwoAxisQueryOutput:
        del donor_indices  # Kept in the signature for drop-in trainer use.
        self._validate_inputs(features, valid_mask)
        effective = self._prepare_mask(valid_mask, arm=arm)
        batch_size = int(features.shape[0])
        sensor_valid = effective[:, :, 0]

        sensor_index = torch.arange(
            self.num_sensors, device=features.device, dtype=torch.long
        ).reshape(1, self.num_sensors, 1)
        role_index = torch.arange(
            self.num_roles, device=features.device, dtype=torch.long
        ).reshape(1, 1, self.num_roles)
        context = (
            self.input_projection(features)
            + self.sensor_embedding(sensor_index)
            + self.role_embedding(role_index)
        )
        context = self.input_norm(context)

        # ------------------------------------------------------------------
        # Order A: time -> sensor.
        # ------------------------------------------------------------------
        sensor_evidence = context.new_zeros(
            batch_size, self.num_sensors, self.model_dim
        )
        for sensor in range(self.num_sensors):
            active = torch.nonzero(
                sensor_valid[:, sensor], as_tuple=False
            ).flatten()
            if active.numel() == 0:
                continue
            sensor_context = context[active, sensor]
            sensor_mask = effective[active, sensor]
            query = sensor_context[:, 0:1]
            for block in self.t2s_temporal_blocks:
                query, _ = block(query, sensor_context, sensor_mask)
            sensor_evidence[active, sensor] = self.sensor_output_norm(query[:, 0])

        t2s_query = (
            self._masked_mean(sensor_evidence, sensor_valid)
            + self.t2s_sensor_query.expand(batch_size, -1, -1)
        )
        t2s_query, sensor_attention = self.t2s_sensor_block(
            t2s_query,
            sensor_evidence,
            sensor_valid,
            return_attention=True,
        )
        if sensor_attention is None:  # pragma: no cover - guarded by flag
            raise AssertionError("sensor attention was not returned")
        t2s = self.axis_output_norm(t2s_query[:, 0])

        # ------------------------------------------------------------------
        # Order B: sensor -> time.
        # ------------------------------------------------------------------
        role_valid = effective.any(dim=1)
        role_evidence = context.new_zeros(
            batch_size, self.num_roles, self.model_dim
        )
        for role in range(self.num_roles):
            active = torch.nonzero(role_valid[:, role], as_tuple=False).flatten()
            if active.numel() == 0:
                continue
            role_context = context[active, :, role]
            role_mask = effective[active, :, role]
            role_query = (
                self._masked_mean(role_context, role_mask)
                + self.s2t_role_queries[:, role : role + 1].expand(
                    active.numel(), -1, -1
                )
            )
            role_query, _ = self.s2t_role_sensor_block(
                role_query, role_context, role_mask
            )
            role_evidence[active, role] = self.role_output_norm(
                role_query[:, 0]
            )

        s2t_query = role_evidence[:, 0:1]
        for block in self.s2t_temporal_blocks:
            s2t_query, _ = block(s2t_query, role_evidence, role_valid)
        s2t = self.axis_output_norm(s2t_query[:, 0])

        gate = torch.sigmoid(self.axis_gate(torch.cat((t2s, s2t), dim=-1)))
        fused = self.axis_output_norm(gate * t2s + (1.0 - gate) * s2t)
        residual_fused_logits = self.fused_classifier(fused).squeeze(-1)
        if base_fused_logits is not None:
            if base_fused_logits.shape != residual_fused_logits.shape:
                raise ValueError("base_fused_logits must have shape [B].")
            fused_logits = residual_fused_logits + base_fused_logits.to(
                device=residual_fused_logits.device,
                dtype=residual_fused_logits.dtype,
            )
        else:
            fused_logits = residual_fused_logits
        sensor_logits = torch.stack(
            [
                classifier(sensor_evidence[:, sensor]).squeeze(-1)
                for sensor, classifier in enumerate(self.sensor_classifiers)
            ],
            dim=1,
        )
        if base_sensor_logits is not None:
            if base_sensor_logits.shape != sensor_logits.shape:
                raise ValueError("base_sensor_logits must have shape [B,S].")
            sensor_logits = sensor_logits + base_sensor_logits.to(
                device=sensor_logits.device,
                dtype=sensor_logits.dtype,
            )
        sensor_logits = torch.where(
            sensor_valid, sensor_logits, torch.zeros_like(sensor_logits)
        )
        return TwoAxisQueryOutput(
            fused_logits=fused_logits,
            sensor_logits=sensor_logits,
            sensor_valid=sensor_valid,
            sensor_evidence=sensor_evidence,
            sensor_attention=sensor_attention,
            effective_time_valid=effective,
            time_then_sensor_logits=self.t2s_classifier(t2s).squeeze(-1),
            sensor_then_time_logits=self.s2t_classifier(s2t).squeeze(-1),
            axis_gate=gate,
            role_valid=role_valid,
            base_fused_logits=base_fused_logits,
        )


__all__ = [
    "MaskedCurrentQueryBlock",
    "TWO_AXIS_ARMS",
    "TwoAxisQuery360Head",
    "TwoAxisQueryOutput",
]
