#!/usr/bin/env python3
"""Development-only global TEMPO pilot on the complete legacy-360 cache.

The runner freezes the promoted scale-aware two-axis checkpoint and trains a
small zero-initialized residual head on top of its logits.  It never accepts a
test cache.  Four matched heads differ only in the evidence operator/fusion:

P1
    Plain current-minus-history deltas with masked-mean sensor fusion.
P2
    History-normalized onset (current change minus the historical trend).
P3
    P2 plus nominal-gap/reliability gating and the legacy scale guard that
    prevents 10.5 km S5P history from acting as 360 m plume-local evidence.
P4
    P3 plus learned evidence-token sensor fusion.

The second, still matched, screen adds less prescriptive learned variants:

Q1
    P1 plus a learned history gap/similarity gate, retaining every sensor.
Q2
    A shared per-feature normality mixer learns from short/long deltas and
    signed/absolute historical change instead of hard-coding a linear trend.
Q3
    Q2 plus evidence-token fusion conditioned on the frozen base sensor logits.

The raw-delta optimization screen adds:

R1
    P1 trained for up to four epochs with early stopping.
R2
    Raw deltas with learned history and base-logit-conditioned sensor gates.
R3
    R1 with an additional equal-event-weight loss on all-negative train events.
R4
    TEA-style motion excitation: signed/absolute delta generates a feature-wise
    gate on current appearance, followed by a signed-delta residual.
R4-add
    Capacity-matched additive control that activates the exact same projections
    and excitation MLP but adds appearance/motion/excitation instead of
    multiplying appearance by a sigmoid gate.
R5
    NormWear-style masked CLS liaison over unchanged raw-delta sensor tokens.
R6-attn
    Direct baseline with t0 as a per-sensor query and the two histories as
    keys/values, followed by masked-mean sensor fusion.

Every learned arm instantiates the exact same parameter set and starts from
the exact same seed-specific state. The final fused and per-sensor residual
classifiers are zero initialized, so epoch zero exactly reproduces the
promoted model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_VERSION = "tempo-legacy360-global-v5"
ARMS = (
    "p1",
    "p2",
    "p3",
    "p4",
    "q1",
    "q2",
    "q3",
    "r1",
    "r2",
    "r3",
    "r4",
    "r4add",
    "r5",
    "r6attn",
)
ARM_DESCRIPTIONS = {
    "p0": "frozen promoted scale-aware two-axis checkpoint",
    "p1": "plain current-minus-history deltas; mean sensor fusion",
    "p2": "history-normalized onset; mean sensor fusion",
    "p3": "onset plus nominal-gap/reliability/scale gate; mean fusion",
    "p4": "gated onset plus learned evidence-token sensor fusion",
    "q1": "plain deltas plus learned history gate; all sensors retained",
    "q2": "learned history-normality mixer and history gate; mean fusion",
    "q3": (
        "learned normality and evidence-token fusion conditioned on frozen "
        "base sensor confidence"
    ),
    "r1": "raw deltas, longer early-stopped optimization",
    "r2": "raw deltas with learned history and base-conditioned sensor gates",
    "r3": "raw deltas plus event-balanced all-negative training penalty",
    "r4": (
        "TEA-style current appearance times delta excitation plus signed "
        "motion residual"
    ),
    "r4add": (
        "capacity-matched additive appearance, signed motion, and tanh "
        "motion-excitation control"
    ),
    "r5": "NormWear-style masked set-attention liaison over raw-delta sensors",
    "r6attn": (
        "t0-query per-sensor temporal cross-attention over history keys/values"
    ),
}
DEFAULT_FORMAL_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5"
)
DEFAULT_OUTPUT = Path(
    "/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1"
)

import sys

for search_path in (
    REPO_ROOT,
    REPO_ROOT / "research" / "pretraining_20260727",
):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from research.pretraining_20260727.query360_model import (  # noqa: E402
    fixed_epoch_batches,
    model_parameter_signature,
    set_deterministic_seed,
    state_dict_sha256,
)
from research.pretraining_20260727.query360_two_axis_full_legacy import (  # noqa: E402
    SENSOR_ORDER,
    _base_logits_for_mode,
    _features_for_mode,
    _sensor_base_for_mode,
    best_positive_f1_threshold,
    classification_metrics,
    fixed_threshold_metrics,
    load_feature_cache,
    sha256_file,
)
from research.pretraining_20260727.query360_two_axis_model import (  # noqa: E402
    TwoAxisQuery360Head,
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def guard_development_path(path: Path, role: str) -> None:
    lower_parts = [part.lower() for part in path.parts]
    blocked = {"test", "sealed", "evaluation"}
    if blocked.intersection(lower_parts):
        raise ValueError(f"{role} path looks held out and is refused: {path}")


def validate_development_caches(
    train: Mapping[str, Any],
    dev: Mapping[str, Any],
) -> None:
    if bool(train.get("sealed_test_read")) or bool(dev.get("sealed_test_read")):
        raise RuntimeError("A cache records sealed-test access.")
    if train.get("split") not in {"train", "train_core"}:
        raise ValueError(f"Unexpected train split: {train.get('split')}")
    if dev.get("split") not in {"dev", "evaluation"}:
        raise ValueError(f"Unexpected dev split: {dev.get('split')}")
    if train.get("encoder") != dev.get("encoder"):
        raise RuntimeError("Train/dev encoder provenance differs.")
    plume_overlap = set(map(str, train["plume_ids"])) & set(
        map(str, dev["plume_ids"])
    )
    event_overlap = set(map(str, train["event_ids"])) & set(
        map(str, dev["event_ids"])
    )
    if plume_overlap or event_overlap:
        raise RuntimeError(
            "Inner train/dev identity overlap: "
            f"plumes={len(plume_overlap)}, events={len(event_overlap)}"
        )


class TempoGlobalHead(nn.Module):
    """Matched global onset head with zero-init residual prediction."""

    def __init__(
        self,
        feature_dim: int,
        *,
        num_sensors: int = 4,
        model_dim: int = 48,
        residual_cap: float = 1.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or num_sensors != 4 or model_dim <= 0:
            raise ValueError("TEMPO expects positive dimensions and four sensors.")
        self.feature_dim = int(feature_dim)
        self.num_sensors = int(num_sensors)
        self.model_dim = int(model_dim)
        self.residual_cap = float(residual_cap)

        self.evidence_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.sensor_embedding = nn.Embedding(num_sensors, model_dim)
        self.normality_channel_mixer = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )
        self.gap_gate = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.sensor_gate = nn.Sequential(
            nn.LayerNorm(model_dim + 2),
            nn.Linear(model_dim + 2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.fused_mlp = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.fused_classifier = nn.Linear(model_dim, 1)
        self.sensor_classifier = nn.Linear(model_dim, 1)
        # Declared after every R1-used module so adding R4 does not perturb the
        # matched R1 initialization stream. These modules are dormant outside
        # R4/R4-add but remain in every arm's parameter signature.
        self.motion_appearance_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.motion_magnitude_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.motion_excitation = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.liaison_query = nn.Parameter(torch.empty(1, 1, model_dim))
        self.liaison_attention = nn.MultiheadAttention(
            model_dim,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )
        self.liaison_query_norm = nn.LayerNorm(model_dim)
        self.liaison_context_norm = nn.LayerNorm(model_dim)
        self.liaison_output_norm = nn.LayerNorm(model_dim)
        nn.init.normal_(self.liaison_query, std=0.02)

        nn.init.zeros_(self.fused_classifier.weight)
        nn.init.zeros_(self.fused_classifier.bias)
        nn.init.zeros_(self.sensor_classifier.weight)
        nn.init.zeros_(self.sensor_classifier.bias)
        # Direct t0-query/history-KV control. Its graph-active parameter count
        # is exactly matched to R4 at D=768, model_dim=48: three independent
        # D->d projections, bias-free attention output, one post-attention MLP,
        # and the shared sensor/fused residual heads.
        self.attention_query_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.attention_key_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.attention_value_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.attention_output_projection = nn.Linear(
            model_dim, model_dim, bias=False
        )
        self.attention_post_mlp = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )

        self.register_buffer(
            "nominal_gap_fraction",
            torch.tensor([90.0 / 360.0, 1.0], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "plume_local_sensor",
            torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32),
            persistent=True,
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
        weight = valid.to(values.dtype).unsqueeze(-1)
        return (values * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
        safe = scores.masked_fill(~valid, -torch.inf)
        any_valid = valid.any(dim=dim, keepdim=True)
        safe = torch.where(any_valid, safe, torch.zeros_like(safe))
        weight = torch.softmax(safe, dim=dim)
        return torch.where(valid & any_valid, weight, torch.zeros_like(weight))

    def evidence_operator(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        arm: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected {ARMS}.")
        if (
            features.ndim != 4
            or features.shape[1:3] != (self.num_sensors, 3)
            or features.shape[-1] != self.feature_dim
        ):
            raise ValueError("features must have shape [B,4,3,D].")
        if valid_mask.shape != features.shape[:3] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean [B,4,3].")

        # Per-observation stateless normalization makes differences comparable
        # without fitting any statistics on development.
        z = F.layer_norm(features.float(), (self.feature_dim,))
        current, history_short, history_long = z.unbind(dim=2)
        current_valid = valid_mask[:, :, 0]
        short_valid = current_valid & valid_mask[:, :, 1]
        long_valid = current_valid & valid_mask[:, :, 2]
        delta_short = current - history_short
        delta_long = current - history_long

        if arm == "r6attn":
            history = torch.stack((history_short, history_long), dim=2)
            history_valid = torch.stack((short_valid, long_valid), dim=2)
            query = self.attention_query_projection(current)
            key = self.attention_key_projection(history)
            value = self.attention_value_projection(history)
            num_heads = 4
            if self.model_dim % num_heads:
                raise RuntimeError("Attention model dimension must divide heads.")
            head_dim = self.model_dim // num_heads
            query_heads = query.reshape(
                len(query), self.num_sensors, num_heads, head_dim
            )
            key_heads = key.reshape(
                len(key), self.num_sensors, 2, num_heads, head_dim
            ).permute(0, 1, 3, 2, 4)
            value_heads = value.reshape(
                len(value), self.num_sensors, 2, num_heads, head_dim
            ).permute(0, 1, 3, 2, 4)
            attention_score = (
                query_heads.unsqueeze(3) * key_heads
            ).sum(dim=-1) / math.sqrt(float(head_dim))
            attention_valid = history_valid.unsqueeze(2).expand(
                -1, -1, num_heads, -1
            )
            attention_weight = self._masked_softmax(
                attention_score, attention_valid, dim=3
            )
            attended = (
                attention_weight.unsqueeze(-1) * value_heads
            ).sum(dim=3)
            attended = attended.reshape(
                len(attended), self.num_sensors, self.model_dim
            )
            attended = self.attention_output_projection(attended)
            attended = self.attention_post_mlp(query + attended)
            projected = attended.unsqueeze(2)
            token_valid = history_valid.any(dim=2, keepdim=True)
        elif arm in {"p1", "r1", "r2", "r3", "r4", "r4add", "r5"}:
            raw = torch.stack((delta_short, delta_long), dim=2)
            token_valid = torch.stack((short_valid, long_valid), dim=2)
        elif arm in {"p2", "p3", "p4"}:
            both = short_valid & long_valid
            historical_change = history_short - history_long
            ratio = 90.0 / (360.0 - 90.0)
            signed_innovation = delta_short - ratio * historical_change
            normal_magnitude = ratio * historical_change.abs()
            excess_magnitude = torch.sign(delta_short) * torch.relu(
                delta_short.abs() - normal_magnitude
            )
            raw = torch.stack((signed_innovation, excess_magnitude), dim=2)
            token_valid = torch.stack((both, both), dim=2)
        elif arm == "q1":
            raw = torch.stack((delta_short, delta_long), dim=2)
            token_valid = torch.stack((short_valid, long_valid), dim=2)
        else:
            both = short_valid & long_valid
            historical_change = history_short - history_long
            mixer_input = torch.stack(
                (
                    delta_short,
                    delta_long,
                    historical_change,
                    historical_change.abs(),
                ),
                dim=-1,
            )
            mixed = self.normality_channel_mixer(mixer_input)
            raw = mixed.permute(0, 1, 3, 2).contiguous()
            token_valid = torch.stack((both, both), dim=2)

        if arm == "r6attn":
            pass
        elif arm in {"r4", "r4add"}:
            current_token = current.unsqueeze(2).expand_as(raw)
            appearance = self.motion_appearance_projection(current_token)
            signed_motion = self.evidence_projection(raw)
            magnitude_motion = self.motion_magnitude_projection(raw.abs())
            excitation_logits = self.motion_excitation(
                torch.cat((signed_motion, magnitude_motion), dim=-1)
            )
            if arm == "r4":
                projected = (
                    appearance * torch.sigmoid(excitation_logits)
                    + signed_motion
                )
            else:
                projected = (
                    appearance
                    + signed_motion
                    + torch.tanh(excitation_logits)
                )
        else:
            projected = self.evidence_projection(raw)
        sensor_index = torch.arange(
            self.num_sensors, device=features.device, dtype=torch.long
        ).reshape(1, self.num_sensors, 1)
        projected = projected + self.sensor_embedding(sensor_index)

        if arm in {"p3", "p4", "q1", "q2", "q3", "r2"}:
            scale_local = self.plume_local_sensor.reshape(
                1, self.num_sensors, 1
            )
            if arm in {"p3", "p4"}:
                token_valid = token_valid & scale_local.bool()
                local = scale_local
            else:
                local = torch.ones_like(scale_local)
            cosine_short = F.cosine_similarity(current, history_short, dim=-1)
            cosine_long = F.cosine_similarity(current, history_long, dim=-1)
            similarity = torch.stack((cosine_short, cosine_long), dim=2)
            gate_input = torch.stack(
                (
                    self.nominal_gap_fraction.reshape(1, 1, 2).expand_as(similarity),
                    token_valid.to(similarity.dtype),
                    local.expand_as(similarity),
                    similarity,
                ),
                dim=-1,
            )
            gate_score = self.gap_gate(gate_input).squeeze(-1)
            history_weight = self._masked_softmax(gate_score, token_valid, dim=2)
            sensor_evidence = (projected * history_weight.unsqueeze(-1)).sum(dim=2)
        else:
            sensor_evidence = self._masked_mean(projected, token_valid, dim=2)

        evidence_valid = token_valid.any(dim=2)
        sensor_evidence = torch.where(
            evidence_valid.unsqueeze(-1),
            sensor_evidence,
            torch.zeros_like(sensor_evidence),
        )
        return sensor_evidence, evidence_valid, token_valid

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        base_fused_logits: torch.Tensor,
        base_sensor_logits: torch.Tensor,
        *,
        arm: str,
    ) -> dict[str, torch.Tensor]:
        sensor_evidence, evidence_valid, token_valid = self.evidence_operator(
            features, valid_mask, arm=arm
        )
        if arm == "r5":
            safe_valid = evidence_valid.clone()
            no_evidence = ~safe_valid.any(dim=1)
            safe_valid[no_evidence, 0] = True
            liaison_query = (
                self._masked_mean(sensor_evidence, safe_valid, dim=1).unsqueeze(1)
                + self.liaison_query.expand(len(sensor_evidence), -1, -1)
            )
            attended, sensor_weight = self.liaison_attention(
                self.liaison_query_norm(liaison_query),
                self.liaison_context_norm(sensor_evidence),
                self.liaison_context_norm(sensor_evidence),
                key_padding_mask=~safe_valid,
                need_weights=True,
                average_attn_weights=True,
            )
            fused = self.liaison_output_norm(
                liaison_query[:, 0] + attended[:, 0]
            )
            sensor_weight = sensor_weight[:, 0]
            sensor_weight = torch.where(
                evidence_valid, sensor_weight, torch.zeros_like(sensor_weight)
            )
        elif arm in {"p4", "q3", "r2"}:
            base_score = base_sensor_logits.to(sensor_evidence.dtype)
            gate_context = torch.cat(
                (
                    sensor_evidence,
                    base_score.unsqueeze(-1),
                    base_score.abs().unsqueeze(-1),
                ),
                dim=-1,
            )
            sensor_score = self.sensor_gate(gate_context).squeeze(-1)
            sensor_weight = self._masked_softmax(
                sensor_score, evidence_valid, dim=1
            )
            fused = (sensor_evidence * sensor_weight.unsqueeze(-1)).sum(dim=1)
        else:
            sensor_weight = evidence_valid.to(sensor_evidence.dtype)
            sensor_weight = sensor_weight / sensor_weight.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            fused = (sensor_evidence * sensor_weight.unsqueeze(-1)).sum(dim=1)

        any_evidence = evidence_valid.any(dim=1)
        fused_hidden = self.fused_mlp(fused)
        residual = self.fused_classifier(fused_hidden).squeeze(-1)
        residual = self.residual_cap * torch.tanh(residual / self.residual_cap)
        residual = torch.where(any_evidence, residual, torch.zeros_like(residual))
        sensor_residual = self.sensor_classifier(sensor_evidence).squeeze(-1)
        sensor_residual = self.residual_cap * torch.tanh(
            sensor_residual / self.residual_cap
        )
        sensor_residual = torch.where(
            evidence_valid, sensor_residual, torch.zeros_like(sensor_residual)
        )
        return {
            "fused_logits": base_fused_logits.to(residual.dtype) + residual,
            "sensor_logits": base_sensor_logits.to(sensor_residual.dtype)
            + sensor_residual,
            "residual": residual,
            "sensor_residual": sensor_residual,
            "sensor_valid": evidence_valid,
            "sensor_weights": sensor_weight,
            "token_valid": token_valid,
        }


def build_matched_initial_state(
    feature_dim: int,
    *,
    model_dim: int,
    residual_cap: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    set_seed(seed)
    model = TempoGlobalHead(
        feature_dim,
        model_dim=model_dim,
        residual_cap=residual_cap,
    )
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    return state, model_parameter_signature(model), state_dict_sha256(state)


def active_parameter_compute_contract(
    model: TempoGlobalHead,
    *,
    arm: str,
) -> dict[str, Any]:
    """Audit graph-active parameters and the per-row learned linear compute.

    A parameter is graph-active when autograd materializes its gradient for a
    deterministic, fully valid batch.  The gradient may be numerically zero:
    the residual classifiers are deliberately zero initialized, so requiring a
    nonzero gradient would incorrectly classify upstream modules as dormant.
    """

    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected {ARMS}.")
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    feature_count = 2 * model.num_sensors * 3 * model.feature_dim
    features = torch.linspace(
        -1.0,
        1.0,
        steps=feature_count,
        dtype=torch.float32,
    ).reshape(2, model.num_sensors, 3, model.feature_dim)
    valid = torch.ones(
        2, model.num_sensors, 3, dtype=torch.bool
    )
    base_fused = torch.zeros(2, dtype=torch.float32)
    base_sensor = torch.zeros(
        2, model.num_sensors, dtype=torch.float32
    )
    output = model(
        features,
        valid,
        base_fused,
        base_sensor,
        arm=arm,
    )
    (
        output["fused_logits"].sum()
        + output["sensor_logits"].sum()
    ).backward()
    active_parameters = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "parameter_count": int(parameter.numel()),
        }
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    active_names = [entry["name"] for entry in active_parameters]
    active_name_sha256 = hashlib.sha256(
        json.dumps(
            active_names,
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    full_parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    active_parameter_count = int(
        sum(entry["parameter_count"] for entry in active_parameters)
    )
    contract: dict[str, Any] = {
        "definition": (
            "parameter.grad is not None after one deterministic fully-valid "
            "forward/backward; zero-valued gradients remain graph-active"
        ),
        "full_parameter_count": full_parameter_count,
        "active_parameter_count": active_parameter_count,
        "dormant_parameter_count": int(
            full_parameter_count - active_parameter_count
        ),
        "active_parameter_names_sha256": active_name_sha256,
        "active_parameters": active_parameters,
        "audit_batch_shape": [
            2,
            model.num_sensors,
            3,
            model.feature_dim,
        ],
    }
    if arm in {"r4", "r4add"}:
        evidence_tokens_per_row = model.num_sensors * 2
        projection_macs = (
            3
            * evidence_tokens_per_row
            * model.feature_dim
            * model.model_dim
        )
        excitation_macs = (
            evidence_tokens_per_row
            * (2 * model.model_dim)
            * model.model_dim
        )
        head_macs = (
            model.model_dim * model.model_dim
            + model.model_dim
            + model.num_sensors * model.model_dim
        )
        contract["compute"] = {
            "learned_linear_macs_per_row": int(
                projection_macs + excitation_macs + head_macs
            ),
            "learned_linear_breakdown_per_row": {
                "three_D_to_model_dim_projections": int(projection_macs),
                "motion_excitation_2model_dim_to_model_dim": int(
                    excitation_macs
                ),
                "fused_mlp_and_classifiers": int(head_macs),
            },
            "learned_transform_sequence": [
                "appearance LayerNorm + Linear(D,model_dim) + GELU + LayerNorm",
                "signed-motion LayerNorm + Linear(D,model_dim) + GELU + LayerNorm",
                "magnitude LayerNorm + Linear(D,model_dim) + GELU + LayerNorm",
                "excitation LayerNorm(2*model_dim) + Linear(2*model_dim,model_dim)",
                "fused Linear(model_dim,model_dim) and fused/sensor classifiers",
            ],
            "projection_and_classifier_compute_matched_between_r4_r4add": True,
            "fusion_only_difference": (
                "appearance*sigmoid(excitation)+signed_motion"
                if arm == "r4"
                else "appearance+signed_motion+tanh(excitation)"
            ),
            "pointwise_fusion_note": (
                "R4 and R4-add have identical learned transforms, tensor "
                "shapes, and active parameters; their fixed pointwise "
                "nonlinearity/arithmetic is intentionally the treatment."
            ),
        }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return contract


def build_promoted_model(checkpoint: Mapping[str, Any]) -> TwoAxisQuery360Head:
    config = checkpoint["model_config"]
    model = TwoAxisQuery360Head(
        int(config["embed_dim"]),
        num_sensors=int(config["num_sensors"]),
        num_roles=int(config["num_roles"]),
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        temporal_depth=int(config["temporal_depth"]),
        mlp_ratio=float(config["mlp_ratio"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def promoted_outputs(
    checkpoint: Mapping[str, Any],
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if checkpoint.get("arm") != "scale_aware_two_axis_query":
        raise ValueError("Promoted P0 must be the scale-aware two-axis arm.")
    if checkpoint.get("base_mode") != "universal":
        raise ValueError("Promoted P0 must use the universal base mode.")
    model = build_promoted_model(checkpoint).to(device).eval()
    features = _features_for_mode(cache, "universal")
    base_fused = _base_logits_for_mode(cache, "universal")
    base_sensor = _sensor_base_for_mode(cache, "universal")
    if base_sensor is None:
        raise ValueError("Cache has no universal sensor logits.")
    fused_output = torch.empty(len(features), dtype=torch.float32)
    sensor_output = torch.empty(
        len(features), len(SENSOR_ORDER), dtype=torch.float32
    )
    with torch.inference_mode():
        for indices in fixed_epoch_batches(
            len(features),
            batch_size=batch_size,
            seed=0,
            epoch=0,
            shuffle=False,
        ):
            output = model(
                features[indices].to(device=device, dtype=torch.float32),
                cache["valid_mask"][indices].to(device),
                arm="scale_aware_two_axis_query",
                base_fused_logits=base_fused[indices].to(device),
                base_sensor_logits=base_sensor[indices].to(device),
            )
            fused_output[indices] = output.fused_logits.cpu()
            sensor_output[indices] = output.sensor_logits.cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return fused_output, sensor_output


def probability_metrics(
    labels: torch.Tensor,
    probabilities: np.ndarray,
    availability_signatures: Sequence[str],
) -> dict[str, Any]:
    output = classification_metrics(labels, probabilities)
    target = labels.cpu()
    signatures = np.asarray([str(value) for value in availability_signatures])
    by_sensor: dict[str, Any] = {}
    for sensor in SENSOR_ORDER:
        positions = np.flatnonzero(
            np.asarray(
                [
                    sensor in signature.split("+")
                    for signature in signatures.tolist()
                ],
                dtype=bool,
            )
        )
        subset = target[torch.from_numpy(positions).long()]
        if len(torch.unique(subset)) == 2:
            by_sensor[sensor] = classification_metrics(
                subset, probabilities[positions]
            )
    output["by_sensor"] = by_sensor
    return output


def fixed_probability_metrics(
    labels: torch.Tensor,
    probabilities: np.ndarray,
    availability_signatures: Sequence[str],
    *,
    threshold: float,
) -> dict[str, Any]:
    output = fixed_threshold_metrics(labels, probabilities, threshold=threshold)
    target = labels.cpu()
    signatures = np.asarray([str(value) for value in availability_signatures])
    by_sensor: dict[str, Any] = {}
    for sensor in SENSOR_ORDER:
        positions = np.flatnonzero(
            np.asarray(
                [
                    sensor in signature.split("+")
                    for signature in signatures.tolist()
                ],
                dtype=bool,
            )
        )
        subset = target[torch.from_numpy(positions).long()]
        if len(torch.unique(subset)) == 2:
            by_sensor[sensor] = fixed_threshold_metrics(
                subset, probabilities[positions], threshold=threshold
            )
    output["by_sensor"] = by_sensor
    return output


def all_negative_event_fp(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    event_ids: Sequence[str],
    *,
    threshold: float,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=np.int64),
            "probability": np.asarray(probabilities, dtype=np.float64),
            "event_id": [str(value) for value in event_ids],
        }
    )
    negative_ids = (
        frame.groupby("event_id", sort=True)["label"]
        .max()
        .loc[lambda value: value.eq(0)]
        .index
    )
    negative = frame.loc[frame["event_id"].isin(negative_ids)].copy()
    negative["hard_fp"] = (
        negative["probability"] >= float(threshold)
    ).astype(np.float64)
    event = negative.groupby("event_id", sort=True).agg(
        rows=("label", "size"),
        fp_rate=("hard_fp", "mean"),
        soft_probability=("probability", "mean"),
        max_probability=("probability", "max"),
    )
    return {
        "definition": (
            "equal canonical-event weight; all rows in each included event "
            "have label 0; FP mass is the sum of within-event hard-FP rates"
        ),
        "threshold": float(threshold),
        "events": int(len(event)),
        "rows": int(len(negative)),
        "fp_event_count": int((event["fp_rate"] > 0).sum()),
        "fp_event_fraction": (
            float((event["fp_rate"] > 0).mean()) if len(event) else 0.0
        ),
        "hard_fp_rows": int(negative["hard_fp"].sum()),
        "fp_rate_mean": float(event["fp_rate"].mean()) if len(event) else 0.0,
        "fp_mass": float(event["fp_rate"].sum()) if len(event) else 0.0,
        "max_event_fp_rate": float(event["fp_rate"].max()) if len(event) else 0.0,
        "soft_probability_mean": (
            float(event["soft_probability"].mean()) if len(event) else 0.0
        ),
        "max_probability_mean": (
            float(event["max_probability"].mean()) if len(event) else 0.0
        ),
    }


def event_operating_audit(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    event_ids: Sequence[str],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Audit event-null false alarms and any-detection on positive events."""

    frame = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=np.int64),
            "probability": np.asarray(probabilities, dtype=np.float64),
            "event_id": [str(value) for value in event_ids],
        }
    )
    frame["prediction"] = (
        frame["probability"] >= float(threshold)
    ).astype(np.int64)
    grouped = frame.groupby("event_id", sort=True).agg(
        positive=("label", "max"),
        any_detection=("prediction", "max"),
    )
    positive = grouped.loc[grouped["positive"].eq(1)]
    return {
        "canonical_events": int(len(grouped)),
        "all_negative": all_negative_event_fp(
            labels, probabilities, event_ids, threshold=threshold
        ),
        "positive_or_mixed_events": int(len(positive)),
        "positive_or_mixed_any_detection_count": int(
            positive["any_detection"].sum()
        ),
        "positive_or_mixed_any_detection_recall": (
            float(positive["any_detection"].mean()) if len(positive) else 0.0
        ),
    }


def best_event_guarded_threshold(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    event_ids: Sequence[str],
    *,
    max_all_negative_fp_rows: int,
    max_all_negative_fp_events: int,
    min_positive_event_detections: int,
) -> tuple[float, float, float] | None:
    """Exact row-F1 threshold search subject to frozen event guardrails."""

    target = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    events = np.asarray([str(value) for value in event_ids], dtype=object)
    if target.shape != probability.shape or target.shape != events.shape:
        raise ValueError("Guarded threshold inputs must share shape.")
    event_frame = pd.DataFrame({"event": events, "label": target})
    event_positive = (
        event_frame.groupby("event", sort=True)["label"].max().to_dict()
    )
    order = np.argsort(-probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_target = target[order]
    sorted_events = events[order]
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    tp = fp = null_rows = null_events = positive_events = 0
    active_events: set[str] = set()
    best: tuple[tuple[float, float, float], float, float, float] | None = None
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and sorted_probability[end] == sorted_probability[start]
        ):
            end += 1
        group_target = sorted_target[start:end]
        tp += int(group_target.sum())
        fp += int(len(group_target) - group_target.sum())
        for row_target, event in zip(
            group_target.tolist(), sorted_events[start:end].tolist()
        ):
            if not event_positive[event]:
                null_rows += 1
            if event not in active_events:
                active_events.add(event)
                if event_positive[event]:
                    positive_events += 1
                else:
                    null_events += 1
        if (
            null_rows <= int(max_all_negative_fp_rows)
            and null_events <= int(max_all_negative_fp_events)
            and positive_events >= int(min_positive_event_detections)
        ):
            fn = positives - tp
            tn = negatives - fp
            positive_f1 = (
                2.0 * tp / (2.0 * tp + fp + fn)
                if (2 * tp + fp + fn)
                else 0.0
            )
            negative_f1 = (
                2.0 * tn / (2.0 * tn + fp + fn)
                if (2 * tn + fp + fn)
                else 0.0
            )
            macro_f1 = (positive_f1 + negative_f1) / 2.0
            threshold = float(sorted_probability[start])
            key = (positive_f1, macro_f1, threshold)
            if best is None or key > best[0]:
                best = (key, threshold, positive_f1, macro_f1)
        start = end
    if best is None:
        return None
    return best[1], best[2], best[3]


def all_negative_event_row_weights(
    labels: Sequence[int] | np.ndarray,
    event_ids: Sequence[str],
) -> torch.Tensor:
    """Mean-one row weights that give every all-negative event equal mass."""

    frame = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=np.int64),
            "event_id": [str(value) for value in event_ids],
            "row": np.arange(len(event_ids), dtype=np.int64),
        }
    )
    event_max = frame.groupby("event_id", sort=True)["label"].max()
    negative_ids = set(event_max.loc[event_max.eq(0)].index.astype(str))
    negative = frame["event_id"].isin(negative_ids)
    sizes = frame.loc[negative].groupby("event_id")["row"].transform("size")
    weights = np.zeros(len(frame), dtype=np.float32)
    weights[negative.to_numpy()] = 1.0 / sizes.to_numpy(dtype=np.float32)
    nonzero = weights > 0
    if np.any(nonzero):
        weights[nonzero] /= float(weights[nonzero].mean())
    return torch.from_numpy(weights)


def shuffle_history(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Permute each valid historical sensor-role marginal; leave t0 fixed."""

    if features.ndim != 4 or valid_mask.shape != features.shape[:3]:
        raise ValueError("Malformed feature/mask tensors.")
    output = features.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for sensor in range(features.shape[1]):
        for role in range(1, features.shape[2]):
            positions = torch.nonzero(valid_mask[:, sensor, role]).flatten()
            if positions.numel() < 2:
                continue
            donors = positions[
                torch.randperm(positions.numel(), generator=generator)
            ]
            output[positions, sensor, role] = features[donors, sensor, role]
    return output


def predict_tempo(
    model: TempoGlobalHead,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    base_fused_logits: torch.Tensor,
    base_sensor_logits: torch.Tensor,
    *,
    arm: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = torch.empty(len(features), dtype=torch.float32)
    residuals = torch.empty(len(features), dtype=torch.float32)
    with torch.inference_mode():
        for indices in fixed_epoch_batches(
            len(features),
            batch_size=batch_size,
            seed=0,
            epoch=0,
            shuffle=False,
        ):
            output = model(
                features[indices].to(device=device, dtype=torch.float32),
                valid_mask[indices].to(device),
                base_fused_logits[indices].to(device),
                base_sensor_logits[indices].to(device),
                arm=arm,
            )
            probabilities[indices] = torch.sigmoid(output["fused_logits"]).cpu()
            residuals[indices] = output["residual"].cpu()
    return probabilities.numpy(), residuals.numpy()


def _selection_key(record: Mapping[str, Any]) -> tuple[float, float, float]:
    metrics = record["dev"]
    return (
        float(metrics["best_binary_f1"]),
        float(metrics["ap"]),
        float(metrics["auc"]),
    )


def train_one(
    *,
    arm: str,
    seed: int,
    initial_state: Mapping[str, torch.Tensor],
    signature: Mapping[str, Any],
    initial_sha256: str,
    train_cache: Mapping[str, Any],
    dev_cache: Mapping[str, Any],
    train_base: tuple[torch.Tensor, torch.Tensor],
    dev_base: tuple[torch.Tensor, torch.Tensor],
    shuffled_dev_features: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    sensor_aux_weight: float,
    residual_l2: float,
    null_loss_weight: float,
    null_target_probability: float,
    early_stop_patience: int,
    model_dim: int,
    residual_cap: float,
) -> dict[str, Any]:
    run_dir = output_dir / f"{arm}_seed{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    set_seed(seed)
    model = TempoGlobalHead(
        int(train_cache["features"].shape[-1]),
        model_dim=model_dim,
        residual_cap=residual_cap,
    )
    model.load_state_dict(initial_state, strict=True)
    if model_parameter_signature(model) != signature:
        raise RuntimeError("Matched model signature changed.")
    if state_dict_sha256(model.state_dict()) != initial_sha256:
        raise RuntimeError("Matched initialization changed.")
    active_contract = active_parameter_compute_contract(model, arm=arm)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_features = _features_for_mode(train_cache, "universal")
    dev_features = _features_for_mode(dev_cache, "universal")
    train_valid = train_cache["valid_mask"]
    dev_valid = dev_cache["valid_mask"]
    labels = train_cache["labels"].float()
    null_row_weights = all_negative_event_row_weights(
        train_cache["labels"].numpy(), train_cache["event_ids"]
    )
    train_fused, train_sensor = train_base
    dev_fused, dev_sensor = dev_base

    base_probability = torch.sigmoid(dev_fused).numpy()
    initial_metrics = probability_metrics(
        dev_cache["labels"],
        base_probability,
        dev_cache["availability_signatures"],
    )
    initial_event_audit = event_operating_audit(
        dev_cache["labels"].numpy(),
        base_probability,
        dev_cache["event_ids"],
        threshold=float(initial_metrics["best_binary_f1_threshold"]),
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train": None,
            "dev": initial_metrics,
            "residual_abs_mean": 0.0,
            "elapsed_seconds": 0.0,
            "exact_promoted_p0": True,
        }
    ]
    best_including_p0 = copy.deepcopy(history[0])
    best_trained: dict[str, Any] | None = None
    best_state = copy.deepcopy(initial_state)
    best_probability = base_probability.copy()
    started = time.monotonic()
    stale_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = fused_loss_sum = sensor_loss_sum = l2_sum = null_sum = 0.0
        seen = 0
        batches = fixed_epoch_batches(
            len(labels),
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            shuffle=True,
        )
        for indices in batches:
            target = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                train_features[indices].to(device=device, dtype=torch.float32),
                train_valid[indices].to(device),
                train_fused[indices].to(device),
                train_sensor[indices].to(device),
                arm=arm,
            )
            fused_loss = F.binary_cross_entropy_with_logits(
                output["fused_logits"], target
            )
            expanded_target = target[:, None].expand_as(output["sensor_logits"])
            sensor_raw = F.binary_cross_entropy_with_logits(
                output["sensor_logits"], expanded_target, reduction="none"
            )
            sensor_mask = output["sensor_valid"]
            sensor_loss = (
                (sensor_raw * sensor_mask).sum()
                / sensor_mask.sum().clamp_min(1)
            )
            l2 = output["residual"].square().mean()
            null_loss = output["fused_logits"].new_zeros(())
            if arm == "r3" and null_loss_weight > 0:
                null_weight = null_row_weights[indices].to(device)
                active_null = null_weight > 0
                if active_null.any():
                    target_logit = math.log(
                        float(null_target_probability)
                        / (1.0 - float(null_target_probability))
                    )
                    null_raw = F.softplus(
                        output["fused_logits"][active_null] - target_logit
                    )
                    active_weight = null_weight[active_null]
                    null_loss = (
                        null_raw * active_weight
                    ).sum() / active_weight.sum().clamp_min(1e-8)
            loss = (
                fused_loss
                + float(sensor_aux_weight) * sensor_loss
                + float(residual_l2) * l2
                + float(null_loss_weight) * null_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(indices.numel())
            total_loss += float(loss.detach()) * count
            fused_loss_sum += float(fused_loss.detach()) * count
            sensor_loss_sum += float(sensor_loss.detach()) * count
            l2_sum += float(l2.detach()) * count
            null_sum += float(null_loss.detach()) * count
            seen += count

        probability, residual = predict_tempo(
            model,
            dev_features,
            dev_valid,
            dev_fused,
            dev_sensor,
            arm=arm,
            batch_size=eval_batch_size,
            device=device,
        )
        metrics = probability_metrics(
            dev_cache["labels"],
            probability,
            dev_cache["availability_signatures"],
        )
        record = {
            "epoch": int(epoch),
            "train": {
                "loss": total_loss / seen,
                "fused_bce": fused_loss_sum / seen,
                "sensor_bce": sensor_loss_sum / seen,
                "residual_l2": l2_sum / seen,
                "event_null_loss": null_sum / seen,
                "rows": int(seen),
                "steps": int(len(batches)),
            },
            "dev": metrics,
            "residual_abs_mean": float(np.abs(residual).mean()),
            "residual_abs_max": float(np.abs(residual).max()),
            "elapsed_seconds": float(time.monotonic() - started),
            "exact_promoted_p0": False,
        }
        history.append(record)
        trained_improved = best_trained is None or _selection_key(
            record
        ) > _selection_key(best_trained)
        if trained_improved:
            best_trained = copy.deepcopy(record)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if _selection_key(record) > _selection_key(best_including_p0):
            best_including_p0 = copy.deepcopy(record)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_probability = probability.copy()
        atomic_json(run_dir / "metrics_history.json", history)
        print(
            f"[TEMPO] arm={arm} seed={seed} epoch={epoch}/{epochs} "
            f"loss={record['train']['loss']:.6f} "
            f"F1={metrics['best_binary_f1']:.6f} "
            f"macro={metrics['best_macro_f1_at_binary_threshold']:.6f} "
            f"AP={metrics['ap']:.6f} AUC={metrics['auc']:.6f} "
            f"|res|={record['residual_abs_mean']:.5f}",
            flush=True,
        )
        if early_stop_patience > 0 and stale_epochs >= early_stop_patience:
            print(
                f"[TEMPO] arm={arm} seed={seed} early_stop epoch={epoch} "
                f"patience={early_stop_patience}",
                flush=True,
            )
            break

    if best_trained is None:
        raise RuntimeError("No trained epoch completed.")
    # A candidate is scientifically assessed at its best trained epoch, even
    # when the deployable selection correctly falls back to exact P0.
    selected_epoch = int(best_including_p0["epoch"])
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    selected_threshold = float(
        best_including_p0["dev"]["best_binary_f1_threshold"]
    )
    if selected_epoch == 0:
        shuffled_probability = base_probability.copy()
        selected_residual = np.zeros_like(base_probability)
    else:
        selected_probability, selected_residual = predict_tempo(
            model,
            dev_features,
            dev_valid,
            dev_fused,
            dev_sensor,
            arm=arm,
            batch_size=eval_batch_size,
            device=device,
        )
        if not np.allclose(selected_probability, best_probability, atol=1e-7):
            raise RuntimeError("Reloaded selected prediction changed.")
        shuffled_probability, _ = predict_tempo(
            model,
            shuffled_dev_features,
            dev_valid,
            dev_fused,
            dev_sensor,
            arm=arm,
            batch_size=eval_batch_size,
            device=device,
        )

    shuffle_metrics = fixed_probability_metrics(
        dev_cache["labels"],
        shuffled_probability,
        dev_cache["availability_signatures"],
        threshold=selected_threshold,
    )
    selected_fp = event_operating_audit(
        dev_cache["labels"].numpy(),
        best_probability,
        dev_cache["event_ids"],
        threshold=selected_threshold,
    )
    shuffle_fp = event_operating_audit(
        dev_cache["labels"].numpy(),
        shuffled_probability,
        dev_cache["event_ids"],
        threshold=selected_threshold,
    )
    baseline_null = initial_event_audit["all_negative"]
    guarded = best_event_guarded_threshold(
        dev_cache["labels"].numpy(),
        best_probability,
        dev_cache["event_ids"],
        max_all_negative_fp_rows=int(baseline_null["hard_fp_rows"]),
        max_all_negative_fp_events=int(baseline_null["fp_event_count"]),
        min_positive_event_detections=int(
            initial_event_audit["positive_or_mixed_any_detection_count"]
        ),
    )
    guarded_operating_point: dict[str, Any] | None = None
    if guarded is not None:
        guarded_threshold, guarded_f1, guarded_macro = guarded
        guarded_operating_point = {
            "threshold": float(guarded_threshold),
            "binary_f1": float(guarded_f1),
            "macro_f1": float(guarded_macro),
            "ap": float(best_including_p0["dev"]["ap"]),
            "auc": float(best_including_p0["dev"]["auc"]),
            "event_operating_audit": event_operating_audit(
                dev_cache["labels"].numpy(),
                best_probability,
                dev_cache["event_ids"],
                threshold=guarded_threshold,
            ),
            "constraints_from_p0": {
                "max_all_negative_fp_rows": int(
                    baseline_null["hard_fp_rows"]
                ),
                "max_all_negative_fp_events": int(
                    baseline_null["fp_event_count"]
                ),
                "min_positive_event_detections": int(
                    initial_event_audit[
                        "positive_or_mixed_any_detection_count"
                    ]
                ),
            },
            "threshold_selected_on_development": True,
        }
    atomic_torch(
        run_dir / "checkpoint_best.pth",
        {
            "schema_version": "tempo-legacy360-global-head-v1",
            "script_version": SCRIPT_VERSION,
            "arm": arm,
            "seed": int(seed),
            "epoch": selected_epoch,
            "model_config": {
                "feature_dim": int(train_features.shape[-1]),
                "num_sensors": 4,
                "model_dim": int(model_dim),
                "residual_cap": float(residual_cap),
            },
            "model": best_state,
            "initial_state_sha256": initial_sha256,
            "parameter_signature": signature,
            "active_parameter_compute_contract": active_contract,
            "dev": best_including_p0["dev"],
            "locked_dev_threshold": selected_threshold,
            "test_or_sealed_read": False,
        },
    )
    predictions = pd.DataFrame(
        {
            "id": dev_cache["ids"],
            "plume_id": dev_cache["plume_ids"],
            "event_id": dev_cache["event_ids"],
            "availability_signature": dev_cache["availability_signatures"],
            "label": dev_cache["labels"].numpy(),
            "probability": best_probability,
            "residual": selected_residual,
            "history_shuffle_probability": shuffled_probability,
        }
    )
    predictions.to_csv(run_dir / "dev_predictions_best.csv", index=False)
    result = {
        "schema_version": "tempo-legacy360-global-result-v1",
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "arm": arm,
        "arm_description": ARM_DESCRIPTIONS[arm],
        "seed": int(seed),
        "matched_initial_state_sha256": initial_sha256,
        "parameter_signature": signature,
        "active_parameter_compute_contract": active_contract,
        "best_including_p0": best_including_p0,
        "best_trained_epoch": best_trained,
        "selected_is_exact_p0_fallback": bool(selected_epoch == 0),
        "selected_threshold": selected_threshold,
        "selected_event_operating_audit": selected_fp,
        "selected_all_negative_event_fp": selected_fp["all_negative"],
        "event_guarded_operating_point": guarded_operating_point,
        "history_shuffle": {
            "metrics_at_unshuffled_selected_threshold": shuffle_metrics,
            "event_operating_audit": shuffle_fp,
            "all_negative_event_fp": shuffle_fp["all_negative"],
            "delta_binary_f1": float(
                shuffle_metrics["binary_f1"]
                - best_including_p0["dev"]["best_binary_f1"]
            ),
            "delta_macro_f1": float(
                shuffle_metrics["macro_f1"]
                - best_including_p0["dev"][
                    "best_macro_f1_at_binary_threshold"
                ]
            ),
            "delta_ap": float(
                shuffle_metrics["ap"] - best_including_p0["dev"]["ap"]
            ),
            "delta_auc": float(
                shuffle_metrics["auc"] - best_including_p0["dev"]["auc"]
            ),
        },
        "history": history,
        "protocol": {
            "train_rows": int(len(train_features)),
            "dev_rows": int(len(dev_features)),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "sensor_aux_weight": float(sensor_aux_weight),
            "residual_l2": float(residual_l2),
            "null_loss_weight": (
                float(null_loss_weight) if arm == "r3" else 0.0
            ),
            "null_target_probability": float(null_target_probability),
            "early_stop_patience": int(early_stop_patience),
            "selection": (
                "positive-class dev F1, tie-broken by AP and AUC; epoch 0 "
                "exact promoted P0 remains eligible"
            ),
            "test_or_sealed_read": False,
        },
    }
    atomic_json(run_dir / "result.json", result)
    return result


def write_results_markdown(
    output_dir: Path,
    baseline: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    run_name: str,
) -> None:
    lines = [
        "# TEMPO legacy360 global development results",
        "",
        f"Run: `{run_name}`. Test/sealed artifacts read: **no**.",
        "",
        "| Arm/seed | selected epoch | binary F1 | macro F1 | AP | AUC | "
        "all-negative FP mass | shuffle ΔF1 | shuffle ΔAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| P0 | 0 | {baseline['best_binary_f1']:.5f} | "
            f"{baseline['best_macro_f1_at_binary_threshold']:.5f} | "
            f"{baseline['ap']:.5f} | {baseline['auc']:.5f} | "
            f"{baseline['all_negative_event_fp']['fp_mass']:.4f} | "
            "0.00000 | 0.00000 |"
        ),
    ]
    for result in results:
        selected = result["best_including_p0"]
        metrics = selected["dev"]
        lines.append(
            f"| {result['arm'].upper()}/{result['seed']} | "
            f"{selected['epoch']} | {metrics['best_binary_f1']:.5f} | "
            f"{metrics['best_macro_f1_at_binary_threshold']:.5f} | "
            f"{metrics['ap']:.5f} | {metrics['auc']:.5f} | "
            f"{result['selected_all_negative_event_fp']['fp_mass']:.4f} | "
            f"{result['history_shuffle']['delta_binary_f1']:+.5f} | "
            f"{result['history_shuffle']['delta_ap']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "Every learned arm uses the same parameter signature and the same "
            "seed-specific initialization. Epoch zero is an exact P0 fallback.",
            "",
            "This is a development-only engineering screen on the legacy split. "
            "No value in this file is a clean SOTA or held-out-test result.",
            "",
        ]
    )
    (output_dir / f"{run_name}_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_int_csv(value: str) -> list[int]:
    output = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return output


def parse_arm_csv(value: str) -> list[str]:
    output = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(output) - set(ARMS))
    if not output or unknown:
        raise argparse.ArgumentTypeError(
            f"Arms must be selected from {ARMS}; unknown={unknown}"
        )
    return output


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(value / (1.0 - value))


def paired_event_bootstrap(
    *,
    labels: np.ndarray,
    event_ids: Sequence[str],
    p0_probability: np.ndarray,
    p0_threshold: float,
    candidate_probability: np.ndarray,
    candidate_threshold: float,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Post-selection paired canonical-event bootstrap for metric deltas."""

    target = np.asarray(labels, dtype=np.int64)
    events = np.asarray([str(value) for value in event_ids], dtype=object)
    unique_events = np.asarray(sorted(set(events.tolist())), dtype=object)
    positions = {
        event: np.flatnonzero(events == event) for event in unique_events
    }
    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {
        "binary_f1": [],
        "macro_f1": [],
        "ap": [],
        "auc": [],
    }
    for _ in range(int(repeats)):
        drawn = rng.choice(unique_events, size=len(unique_events), replace=True)
        index = np.concatenate([positions[event] for event in drawn])
        y = target[index]
        if len(np.unique(y)) < 2:
            continue
        p0 = p0_probability[index]
        candidate = candidate_probability[index]
        p0_prediction = p0 >= float(p0_threshold)
        candidate_prediction = candidate >= float(candidate_threshold)
        samples["binary_f1"].append(
            float(f1_score(y, candidate_prediction, zero_division=0))
            - float(f1_score(y, p0_prediction, zero_division=0))
        )
        samples["macro_f1"].append(
            float(
                f1_score(
                    y,
                    candidate_prediction,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                )
            )
            - float(
                f1_score(
                    y,
                    p0_prediction,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                )
            )
        )
        samples["ap"].append(
            float(average_precision_score(y, candidate))
            - float(average_precision_score(y, p0))
        )
        samples["auc"].append(
            float(roc_auc_score(y, candidate))
            - float(roc_auc_score(y, p0))
        )
    output: dict[str, Any] = {
        "unit": "canonical event cluster",
        "repeats_requested": int(repeats),
        "repeats_valid": int(len(samples["binary_f1"])),
        "seed": int(seed),
        "post_selection_diagnostic": True,
        "thresholds_refit_inside_bootstrap": False,
    }
    for metric, values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        output[metric] = {
            "mean_delta": float(array.mean()),
            "ci95": [
                float(np.quantile(array, 0.025)),
                float(np.quantile(array, 0.975)),
            ],
            "probability_delta_gt_zero": float(np.mean(array > 0)),
        }
    return output


def _weighted_bootstrap_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    row_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized fixed-score metrics for event-bootstrap row weights."""

    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    weight = np.asarray(row_weights, dtype=np.float64)
    prediction = probability >= float(threshold)
    positive = y == 1
    negative = ~positive
    tp = weight @ (positive & prediction)
    fp = weight @ (negative & prediction)
    fn = weight @ (positive & ~prediction)
    tn = weight @ (negative & ~prediction)
    binary_f1 = np.divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=(2.0 * tp + fp + fn) > 0,
    )
    negative_f1 = np.divide(
        2.0 * tn,
        2.0 * tn + fp + fn,
        out=np.zeros_like(tn),
        where=(2.0 * tn + fp + fn) > 0,
    )
    order = np.argsort(-probability, kind="mergesort")
    sorted_y = y[order]
    sorted_weight = weight[:, order]
    cumulative_tp = np.cumsum(
        sorted_weight * sorted_y.reshape(1, -1), axis=1
    )
    cumulative_fp = np.cumsum(
        sorted_weight * (1 - sorted_y).reshape(1, -1), axis=1
    )
    precision = np.divide(
        cumulative_tp,
        cumulative_tp + cumulative_fp,
        out=np.zeros_like(cumulative_tp),
        where=(cumulative_tp + cumulative_fp) > 0,
    )
    positive_weight = sorted_weight * sorted_y.reshape(1, -1)
    positive_mass = positive_weight.sum(axis=1)
    ap = np.divide(
        (precision * positive_weight).sum(axis=1),
        positive_mass,
        out=np.zeros_like(positive_mass),
        where=positive_mass > 0,
    )
    return {
        "binary_f1": binary_f1,
        "macro_f1": (binary_f1 + negative_f1) / 2.0,
        "ap": ap,
    }


def paired_multiseed_event_bootstrap(
    *,
    labels: np.ndarray,
    event_ids: Sequence[str],
    p0_probability: np.ndarray,
    p0_threshold: float,
    candidates: Sequence[tuple[int, np.ndarray, float]],
    repeats: int,
    seed: int,
    batch_size: int = 64,
) -> dict[str, Any]:
    """10k-scale paired event bootstrap with every seed checkpoint locked."""

    target = np.asarray(labels, dtype=np.int64)
    events = np.asarray([str(value) for value in event_ids], dtype=object)
    unique_events = np.asarray(sorted(set(events.tolist())), dtype=object)
    event_lookup = {event: index for index, event in enumerate(unique_events)}
    row_event_index = np.asarray(
        [event_lookup[event] for event in events], dtype=np.int64
    )
    rng = np.random.default_rng(int(seed))
    metric_names = ("binary_f1", "macro_f1", "ap")
    per_seed_values = {
        int(candidate_seed): {
            metric: np.empty(int(repeats), dtype=np.float64)
            for metric in metric_names
        }
        for candidate_seed, _, _ in candidates
    }
    mean_values = {
        metric: np.empty(int(repeats), dtype=np.float64)
        for metric in metric_names
    }
    uniform = np.full(len(unique_events), 1.0 / len(unique_events))
    for start in range(0, int(repeats), int(batch_size)):
        stop = min(start + int(batch_size), int(repeats))
        counts = rng.multinomial(
            len(unique_events), uniform, size=stop - start
        )
        row_weights = counts[:, row_event_index].astype(np.float64)
        p0_metrics = _weighted_bootstrap_metrics(
            target, p0_probability, p0_threshold, row_weights
        )
        batch_candidate: dict[str, list[np.ndarray]] = {
            metric: [] for metric in metric_names
        }
        for candidate_seed, probability, threshold in candidates:
            candidate_metrics = _weighted_bootstrap_metrics(
                target, probability, threshold, row_weights
            )
            for metric in metric_names:
                delta = candidate_metrics[metric] - p0_metrics[metric]
                per_seed_values[int(candidate_seed)][metric][start:stop] = delta
                batch_candidate[metric].append(candidate_metrics[metric])
        for metric in metric_names:
            mean_candidate = np.mean(
                np.stack(batch_candidate[metric], axis=0), axis=0
            )
            mean_values[metric][start:stop] = (
                mean_candidate - p0_metrics[metric]
            )

    def summarize(values: np.ndarray) -> dict[str, Any]:
        return {
            "mean_delta": float(values.mean()),
            "ci95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "win_probability": float(np.mean(values > 0)),
        }

    return {
        "unit": "canonical event cluster",
        "repeats": int(repeats),
        "seed": int(seed),
        "fixed_seed_specific_checkpoints_and_thresholds": True,
        "thresholds_refit_inside_bootstrap": False,
        "post_selection_diagnostic": True,
        "per_seed": {
            str(candidate_seed): {
                metric: summarize(values)
                for metric, values in metric_values.items()
            }
            for candidate_seed, metric_values in per_seed_values.items()
        },
        "mean_across_fixed_seed_models": {
            metric: summarize(values)
            for metric, values in mean_values.items()
        },
    }


def command_aggregate(args: argparse.Namespace) -> None:
    summary_path = Path(args.summary).expanduser().absolute()
    output_dir = summary_path.parent
    guard_development_path(summary_path, "summary")
    guard_development_path(output_dir, "aggregate output")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = summary["runs"]
    if len(runs) < 2:
        raise ValueError("Aggregation requires at least two seed runs.")
    arms = {str(run["arm"]) for run in runs}
    if len(arms) != 1:
        raise ValueError(f"Aggregation requires one arm; got {sorted(arms)}")
    frames: list[pd.DataFrame] = []
    for run in runs:
        prediction_path = (
            output_dir
            / f"{run['arm']}_seed{run['seed']}"
            / "dev_predictions_best.csv"
        )
        frames.append(pd.read_csv(prediction_path))
    reference = frames[0]
    identity_columns = [
        "id",
        "plume_id",
        "event_id",
        "availability_signature",
        "label",
    ]
    for frame in frames[1:]:
        if not frame[identity_columns].equals(reference[identity_columns]):
            raise RuntimeError("Seed prediction identities differ.")
    labels = reference["label"].to_numpy(dtype=np.int64)
    event_ids = reference["event_id"].astype(str).tolist()
    signatures = reference["availability_signature"].astype(str).tolist()
    seed_logits = [
        _logit(frame["probability"].to_numpy(dtype=np.float64))
        for frame in frames
    ]
    base_logits = [
        value - frame["residual"].to_numpy(dtype=np.float64)
        for value, frame in zip(seed_logits, frames)
    ]
    maximum_base_spread = float(
        np.max(np.ptp(np.stack(base_logits, axis=0), axis=0))
    )
    # CSV round-trip plus sigmoid/logit inversion introduces about 1e-4
    # spread even though every seed used the same immutable P0 tensor.
    if maximum_base_spread > 2e-4:
        raise RuntimeError(
            f"Recovered promoted P0 logits differ across seeds: "
            f"{maximum_base_spread}"
        )
    p0_probability = 1.0 / (
        1.0 + np.exp(-np.mean(np.stack(base_logits, axis=0), axis=0))
    )
    ensemble_probability = 1.0 / (
        1.0 + np.exp(-np.mean(np.stack(seed_logits, axis=0), axis=0))
    )
    shuffle_logits = [
        _logit(
            frame["history_shuffle_probability"].to_numpy(dtype=np.float64)
        )
        for frame in frames
    ]
    shuffle_probability = 1.0 / (
        1.0 + np.exp(-np.mean(np.stack(shuffle_logits, axis=0), axis=0))
    )
    label_tensor = torch.from_numpy(labels)
    p0_metrics = probability_metrics(
        label_tensor, p0_probability, signatures
    )
    ensemble_metrics = probability_metrics(
        label_tensor, ensemble_probability, signatures
    )
    p0_threshold = float(p0_metrics["best_binary_f1_threshold"])
    ensemble_threshold = float(
        ensemble_metrics["best_binary_f1_threshold"]
    )
    p0_event = event_operating_audit(
        labels, p0_probability, event_ids, threshold=p0_threshold
    )
    ensemble_event = event_operating_audit(
        labels,
        ensemble_probability,
        event_ids,
        threshold=ensemble_threshold,
    )
    baseline_null = p0_event["all_negative"]
    guarded = best_event_guarded_threshold(
        labels,
        ensemble_probability,
        event_ids,
        max_all_negative_fp_rows=int(baseline_null["hard_fp_rows"]),
        max_all_negative_fp_events=int(baseline_null["fp_event_count"]),
        min_positive_event_detections=int(
            p0_event["positive_or_mixed_any_detection_count"]
        ),
    )
    if guarded is None:
        raise RuntimeError("No ensemble threshold satisfies P0 event guardrails.")
    guarded_threshold, guarded_f1, guarded_macro = guarded
    guarded_event = event_operating_audit(
        labels,
        ensemble_probability,
        event_ids,
        threshold=guarded_threshold,
    )
    shuffle_metrics = fixed_probability_metrics(
        label_tensor,
        shuffle_probability,
        signatures,
        threshold=ensemble_threshold,
    )
    metric_names = (
        "best_binary_f1",
        "best_macro_f1_at_binary_threshold",
        "ap",
        "auc",
    )
    seed_statistics: dict[str, Any] = {}
    for metric in metric_names:
        values = np.asarray(
            [run["best_including_p0"]["dev"][metric] for run in runs],
            dtype=np.float64,
        )
        seed_statistics[metric] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "mean_delta_from_p0": float(values.mean() - p0_metrics[metric]),
        }
    bootstrap = paired_multiseed_event_bootstrap(
        labels=labels,
        event_ids=event_ids,
        p0_probability=p0_probability,
        p0_threshold=p0_threshold,
        candidates=[
            (
                int(run["seed"]),
                1.0 / (1.0 + np.exp(-seed_logit)),
                float(
                    run["best_including_p0"]["dev"][
                        "best_binary_f1_threshold"
                    ]
                ),
            )
            for run, seed_logit in zip(runs, seed_logits)
        ],
        repeats=int(args.bootstrap_repeats),
        seed=int(args.bootstrap_seed),
    )
    aggregate = {
        "schema_version": "tempo-legacy360-global-aggregate-v1",
        "script_version": SCRIPT_VERSION,
        "source_summary": str(summary_path),
        "source_summary_sha256": sha256_file(summary_path),
        "arm": next(iter(arms)),
        "seeds": [int(run["seed"]) for run in runs],
        "p0": {
            "metrics": p0_metrics,
            "event_operating_audit": p0_event,
        },
        "seed_statistics": seed_statistics,
        "logit_ensemble": {
            "metrics": ensemble_metrics,
            "event_operating_audit": ensemble_event,
            "event_guarded_operating_point": {
                "threshold": float(guarded_threshold),
                "binary_f1": float(guarded_f1),
                "macro_f1": float(guarded_macro),
                "ap": float(ensemble_metrics["ap"]),
                "auc": float(ensemble_metrics["auc"]),
                "event_operating_audit": guarded_event,
            },
            "history_shuffle_at_unshuffled_threshold": shuffle_metrics,
            "history_shuffle_deltas": {
                "binary_f1": float(
                    shuffle_metrics["binary_f1"]
                    - ensemble_metrics["best_binary_f1"]
                ),
                "macro_f1": float(
                    shuffle_metrics["macro_f1"]
                    - ensemble_metrics[
                        "best_macro_f1_at_binary_threshold"
                    ]
                ),
                "ap": float(shuffle_metrics["ap"] - ensemble_metrics["ap"]),
                "auc": float(
                    shuffle_metrics["auc"] - ensemble_metrics["auc"]
                ),
            },
        },
        "paired_event_bootstrap_fixed_seeds": bootstrap,
        "maximum_recovered_p0_logit_spread": maximum_base_spread,
        "test_or_sealed_read": False,
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    aggregate_path = output_dir / "final_aggregate.json"
    atomic_json(aggregate_path, aggregate)
    pd.DataFrame(
        {
            **{column: reference[column] for column in identity_columns},
            "p0_probability": p0_probability,
            "ensemble_probability": ensemble_probability,
            "history_shuffle_ensemble_probability": shuffle_probability,
        }
    ).to_csv(output_dir / "final_ensemble_predictions.csv", index=False)
    metrics = ensemble_metrics
    guarded_null = guarded_event["all_negative"]
    arm_label = next(iter(arms)).upper()
    seed_count_label = {
        2: "two",
        3: "three",
        4: "four",
    }.get(len(runs), str(len(runs)))
    lines = [
        f"# Final {arm_label} {seed_count_label}-seed development aggregate",
        "",
        "No test/sealed artifact was read.",
        "",
        "| Model | Binary F1 | Macro F1 | AP | AUC | FP events | FP rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| P0 | {p0_metrics['best_binary_f1']:.5f} | "
            f"{p0_metrics['best_macro_f1_at_binary_threshold']:.5f} | "
            f"{p0_metrics['ap']:.5f} | {p0_metrics['auc']:.5f} | "
            f"{baseline_null['fp_event_count']} | "
            f"{baseline_null['hard_fp_rows']} |"
        ),
        (
            f"| {arm_label} logit ensemble | {metrics['best_binary_f1']:.5f} | "
            f"{metrics['best_macro_f1_at_binary_threshold']:.5f} | "
            f"{metrics['ap']:.5f} | {metrics['auc']:.5f} | "
            f"{ensemble_event['all_negative']['fp_event_count']} | "
            f"{ensemble_event['all_negative']['hard_fp_rows']} |"
        ),
        (
            f"| {arm_label} event-guarded | {guarded_f1:.5f} | "
            f"{guarded_macro:.5f} | {metrics['ap']:.5f} | "
            f"{metrics['auc']:.5f} | {guarded_null['fp_event_count']} | "
            f"{guarded_null['hard_fp_rows']} |"
        ),
        "",
        f"## {seed_count_label.capitalize()}-seed statistics",
        "",
        "| Metric | Mean | Sample std | Mean delta from P0 |",
        "|---|---:|---:|---:|",
    ]
    for metric, label in (
        ("best_binary_f1", "Binary F1"),
        ("best_macro_f1_at_binary_threshold", "Macro F1"),
        ("ap", "AP"),
        ("auc", "AUC"),
    ):
        value = seed_statistics[metric]
        lines.append(
            f"| {label} | {value['mean']:.6f} | "
            f"{value['sample_std']:.6f} | "
            f"{value['mean_delta_from_p0']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Sensor strata",
            "",
            f"| Sensor | P0 F1 | {arm_label} mean F1 | ΔF1 | P0 AP | "
            f"{arm_label} mean AP | ΔAP |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sensor in SENSOR_ORDER:
        p0_sensor = p0_metrics["by_sensor"][sensor]
        f1_values = [
            run["best_including_p0"]["dev"]["by_sensor"][sensor][
                "best_binary_f1"
            ]
            for run in runs
        ]
        ap_values = [
            run["best_including_p0"]["dev"]["by_sensor"][sensor]["ap"]
            for run in runs
        ]
        mean_f1 = float(np.mean(f1_values))
        mean_ap = float(np.mean(ap_values))
        lines.append(
            f"| {sensor.upper()} | {p0_sensor['best_binary_f1']:.5f} | "
            f"{mean_f1:.5f} | "
            f"{mean_f1 - p0_sensor['best_binary_f1']:+.5f} | "
            f"{p0_sensor['ap']:.5f} | {mean_ap:.5f} | "
            f"{mean_ap - p0_sensor['ap']:+.5f} |"
        )
    mean_bootstrap = bootstrap["mean_across_fixed_seed_models"]
    lines.extend(
        [
            "",
            "## Fixed-seed paired canonical-event bootstrap",
            "",
            f"`{bootstrap['repeats']}` resamples; thresholds are never refit.",
            "",
            "| Metric delta vs P0 | Mean | 95% CI | Win probability |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, label in (
        ("binary_f1", "Binary F1"),
        ("macro_f1", "Macro F1"),
        ("ap", "AP"),
    ):
        value = mean_bootstrap[metric]
        lines.append(
            f"| {label} | {value['mean_delta']:+.6f} | "
            f"[{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}] | "
            f"{value['win_probability']:.4f} |"
        )
    lines.extend(
        [
            "",
        (
            "The logit ensemble already satisfies all P0 event guardrails; "
            "its F1-selected and event-guarded thresholds are identical."
            if math.isclose(ensemble_threshold, guarded_threshold, abs_tol=1e-12)
            else "The event-guarded threshold is selected on development."
        ),
        "",
        "Paired canonical-event bootstrap is post-selection diagnostic only.",
        "",
        ]
    )
    (output_dir / "FINAL_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(aggregate["logit_ensemble"], indent=2), flush=True)


def command_combine_run_summaries(args: argparse.Namespace) -> None:
    """Combine immutable one/multi-seed summaries before aggregation."""

    source_paths = [
        Path(value).expanduser().absolute() for value in args.summaries
    ]
    output_path = Path(args.output).expanduser().absolute()
    if len(source_paths) < 2:
        raise ValueError("At least two source summaries are required.")
    guard_development_path(output_path, "combined development summary")
    if output_path.exists():
        raise FileExistsError(f"Refusing existing summary: {output_path}")
    summaries: list[dict[str, Any]] = []
    for path in source_paths:
        guard_development_path(path, "source development summary")
        with path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        if summary.get("status") != "complete":
            raise RuntimeError(f"Source summary is incomplete: {path}")
        if bool(summary["guardrails"]["test_or_sealed_read"]):
            raise RuntimeError(f"Source summary reports test/sealed read: {path}")
        summaries.append(summary)
    reference = summaries[0]
    for summary in summaries[1:]:
        if summary["inputs"] != reference["inputs"]:
            raise RuntimeError("Source summaries use different input artifacts.")
        if (
            summary["matched_parameter_signature"]
            != reference["matched_parameter_signature"]
        ):
            raise RuntimeError("Source summaries use different model signatures.")
        reference_p0 = reference["p0"]
        candidate_p0 = summary["p0"]
        for metric in ("best_binary_f1", "ap", "auc"):
            if not math.isclose(
                float(reference_p0[metric]),
                float(candidate_p0[metric]),
                abs_tol=1e-10,
            ):
                raise RuntimeError("Source summaries reproduce different P0.")
    runs = [
        copy.deepcopy(run)
        for summary in summaries
        for run in summary["runs"]
    ]
    seeds = [int(run["seed"]) for run in runs]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Duplicate seeds in combined summaries.")
    arms = sorted({str(run["arm"]) for run in runs})
    if len(arms) != 1:
        raise RuntimeError(f"Combined summary must have one arm; got {arms}.")
    runs.sort(key=lambda run: int(run["seed"]))
    combined = {
        "schema_version": "tempo-legacy360-global-combined-summary-v1",
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "run_name": str(args.run_name),
        "arms": arms,
        "seeds": [int(run["seed"]) for run in runs],
        "inputs": reference["inputs"],
        "p0": reference["p0"],
        "matched_parameter_signature": reference["matched_parameter_signature"],
        "runs": runs,
        "source_summaries": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "script_version": summary["script_version"],
                "seeds": [int(run["seed"]) for run in summary["runs"]],
            }
            for path, summary in zip(source_paths, summaries)
        ],
        "guardrails": {
            "test_or_sealed_read": False,
            "epoch_zero_exact_p0_fallback": all(
                bool(
                    summary["guardrails"][
                        "epoch_zero_exact_p0_fallback"
                    ]
                )
                for summary in summaries
            ),
            "same_parameters_steps_initialization_within_seed": all(
                bool(
                    summary["guardrails"][
                        "same_parameters_steps_initialization_within_seed"
                    ]
                )
                for summary in summaries
            ),
            "history_shuffle_threshold_reoptimized": False,
            "hyperparameters_changed_after_seed101_result": False,
        },
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output_path, combined)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "arm": arms[0],
                "seeds": combined["seeds"],
                "test_or_sealed_read": False,
            },
            indent=2,
        ),
        flush=True,
    )


def command_compare_fixed_ensembles(args: argparse.Namespace) -> None:
    r1_path = Path(args.r1_predictions).expanduser().absolute()
    r4_path = Path(args.r4_predictions).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    for path, role in (
        (r1_path, "R1 predictions"),
        (r4_path, "R4 predictions"),
        (output_dir, "comparison output"),
    ):
        guard_development_path(path, role)
    output_dir.mkdir(parents=True, exist_ok=True)
    r1 = pd.read_csv(r1_path)
    r4 = pd.read_csv(r4_path)
    identity_columns = [
        "id",
        "plume_id",
        "event_id",
        "availability_signature",
        "label",
    ]
    if not r1[identity_columns].equals(r4[identity_columns]):
        raise RuntimeError("R1/R4 prediction identities differ.")
    p0_r1 = r1["p0_probability"].to_numpy(dtype=np.float64)
    p0_r4 = r4["p0_probability"].to_numpy(dtype=np.float64)
    if float(np.max(np.abs(p0_r1 - p0_r4))) > 2e-5:
        raise RuntimeError("R1/R4 aggregates do not share exact P0.")
    p0 = (p0_r1 + p0_r4) / 2.0
    r1_probability = r1["ensemble_probability"].to_numpy(dtype=np.float64)
    r4_probability = r4["ensemble_probability"].to_numpy(dtype=np.float64)
    r1_shuffle = r1[
        "history_shuffle_ensemble_probability"
    ].to_numpy(dtype=np.float64)
    r4_shuffle = r4[
        "history_shuffle_ensemble_probability"
    ].to_numpy(dtype=np.float64)
    candidates = {
        "p0": (p0, p0),
        "r1_equal_logit": (r1_probability, r1_shuffle),
        "r4_equal_logit": (r4_probability, r4_shuffle),
        "p0_r1_half_equal_logit": (
            1.0
            / (
                1.0
                + np.exp(-0.5 * (_logit(p0) + _logit(r1_probability)))
            ),
            1.0
            / (1.0 + np.exp(-0.5 * (_logit(p0) + _logit(r1_shuffle)))),
        ),
        "r1_r4_half_equal_logit": (
            1.0
            / (
                1.0
                + np.exp(
                    -0.5
                    * (_logit(r1_probability) + _logit(r4_probability))
                )
            ),
            1.0
            / (
                1.0
                + np.exp(-0.5 * (_logit(r1_shuffle) + _logit(r4_shuffle)))
            ),
        ),
    }
    labels = r1["label"].to_numpy(dtype=np.int64)
    label_tensor = torch.from_numpy(labels)
    event_ids = r1["event_id"].astype(str).tolist()
    signatures = r1["availability_signature"].astype(str).tolist()
    p0_metrics = probability_metrics(label_tensor, p0, signatures)
    p0_threshold = float(p0_metrics["best_binary_f1_threshold"])
    p0_event = event_operating_audit(
        labels, p0, event_ids, threshold=p0_threshold
    )
    p0_null = p0_event["all_negative"]
    records: dict[str, Any] = {}
    for name, (probability, shuffle_probability) in candidates.items():
        metrics = probability_metrics(label_tensor, probability, signatures)
        threshold = float(metrics["best_binary_f1_threshold"])
        event = event_operating_audit(
            labels, probability, event_ids, threshold=threshold
        )
        guarded = best_event_guarded_threshold(
            labels,
            probability,
            event_ids,
            max_all_negative_fp_rows=int(p0_null["hard_fp_rows"]),
            max_all_negative_fp_events=int(p0_null["fp_event_count"]),
            min_positive_event_detections=int(
                p0_event["positive_or_mixed_any_detection_count"]
            ),
        )
        if guarded is None:
            raise RuntimeError(f"No guarded operating point for {name}.")
        guarded_threshold, guarded_f1, guarded_macro = guarded
        shuffle_metrics = fixed_probability_metrics(
            label_tensor,
            shuffle_probability,
            signatures,
            threshold=threshold,
        )
        records[name] = {
            "definition": (
                "fixed equal-logit arithmetic mean; no fitted ensemble weight"
                if "half" in name
                else "single fixed model/logit ensemble"
            ),
            "metrics": metrics,
            "event_operating_audit": event,
            "event_guarded_operating_point": {
                "threshold": float(guarded_threshold),
                "binary_f1": float(guarded_f1),
                "macro_f1": float(guarded_macro),
                "ap": float(metrics["ap"]),
                "auc": float(metrics["auc"]),
                "event_operating_audit": event_operating_audit(
                    labels,
                    probability,
                    event_ids,
                    threshold=guarded_threshold,
                ),
            },
            "history_shuffle_at_unshuffled_threshold": shuffle_metrics,
            "history_shuffle_deltas": {
                "binary_f1": float(
                    shuffle_metrics["binary_f1"]
                    - metrics["best_binary_f1"]
                ),
                "macro_f1": float(
                    shuffle_metrics["macro_f1"]
                    - metrics["best_macro_f1_at_binary_threshold"]
                ),
                "ap": float(shuffle_metrics["ap"] - metrics["ap"]),
                "auc": float(shuffle_metrics["auc"] - metrics["auc"]),
            },
        }
    blend_bootstrap = paired_multiseed_event_bootstrap(
        labels=labels,
        event_ids=event_ids,
        p0_probability=p0,
        p0_threshold=p0_threshold,
        candidates=[
            (
                1,
                candidates["p0_r1_half_equal_logit"][0],
                records["p0_r1_half_equal_logit"]["metrics"][
                    "best_binary_f1_threshold"
                ],
            ),
            (
                2,
                candidates["r1_r4_half_equal_logit"][0],
                records["r1_r4_half_equal_logit"]["metrics"][
                    "best_binary_f1_threshold"
                ],
            ),
        ],
        repeats=int(args.bootstrap_repeats),
        seed=int(args.bootstrap_seed),
    )
    comparison = {
        "schema_version": "tempo-fixed-equal-logit-comparison-v1",
        "script_version": SCRIPT_VERSION,
        "inputs": {
            "r1": str(r1_path),
            "r1_sha256": sha256_file(r1_path),
            "r4": str(r4_path),
            "r4_sha256": sha256_file(r4_path),
        },
        "records": records,
        "paired_event_bootstrap_vs_p0": {
            "p0_r1_half_equal_logit": blend_bootstrap["per_seed"]["1"],
            "r1_r4_half_equal_logit": blend_bootstrap["per_seed"]["2"],
            "repeats": int(args.bootstrap_repeats),
            "seed": int(args.bootstrap_seed),
            "fixed_weights_and_thresholds": True,
        },
        "test_or_sealed_read": False,
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    atomic_json(output_dir / "fixed_equal_logit_comparison.json", comparison)
    rows = [
        "# Fixed equal-logit development comparison",
        "",
        "No ensemble weight was fitted. No test/sealed artifact was read.",
        "",
        "| Model | F1 | Macro F1 | AP | AUC | FP events | FP rows | Positive events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "p0",
        "r1_equal_logit",
        "r4_equal_logit",
        "p0_r1_half_equal_logit",
        "r1_r4_half_equal_logit",
    ):
        record = records[name]
        metrics = record["metrics"]
        event = record["event_operating_audit"]
        rows.append(
            f"| {name} | {metrics['best_binary_f1']:.5f} | "
            f"{metrics['best_macro_f1_at_binary_threshold']:.5f} | "
            f"{metrics['ap']:.5f} | {metrics['auc']:.5f} | "
            f"{event['all_negative']['fp_event_count']} | "
            f"{event['all_negative']['hard_fp_rows']} | "
            f"{event['positive_or_mixed_any_detection_count']}/"
            f"{event['positive_or_mixed_events']} |"
        )
    rows.extend(
        [
            "",
            "All thresholds are selected on development and then frozen for "
            "history-shuffle/bootstrap diagnostics.",
            "",
        ]
    )
    (output_dir / "FIXED_EQUAL_LOGIT_COMPARISON.md").write_text(
        "\n".join(rows), encoding="utf-8"
    )
    print(json.dumps(records, indent=2), flush=True)


def command_compare_capacity_control(args: argparse.Namespace) -> None:
    """Compare the fixed seed-42 R4 and capacity-matched R4-add runs."""

    r4_dir = Path(args.r4_dir).expanduser().absolute()
    r4add_dir = Path(args.r4add_dir).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    for path, role in (
        (r4_dir, "R4 development run"),
        (r4add_dir, "R4-add development run"),
        (output_dir, "capacity comparison output"),
    ):
        guard_development_path(path, role)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "capacity_control_comparison.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing existing comparison: {output_path}")

    def load_run(
        run_dir: Path,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        result_path = run_dir / "result.json"
        prediction_path = run_dir / "dev_predictions_best.csv"
        guard_development_path(result_path, "development result")
        guard_development_path(prediction_path, "development predictions")
        with result_path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        prediction = pd.read_csv(prediction_path)
        if bool(result["protocol"]["test_or_sealed_read"]):
            raise RuntimeError("Input run reports a test/sealed read.")
        return result, prediction

    r4, r4_prediction = load_run(r4_dir)
    r4add, r4add_prediction = load_run(r4add_dir)
    if r4["arm"] != "r4" or r4add["arm"] != "r4add":
        raise RuntimeError("Capacity comparison expects R4 and R4-add.")
    identity_columns = ["id", "event_id", "label"]
    if not r4_prediction[identity_columns].equals(
        r4add_prediction[identity_columns]
    ):
        raise RuntimeError("R4/R4-add development rows changed.")

    r4_metrics = r4["best_including_p0"]["dev"]
    r4add_metrics = r4add["best_including_p0"]["dev"]
    bootstrap = paired_multiseed_event_bootstrap(
        labels=r4_prediction["label"].to_numpy(),
        event_ids=r4_prediction["event_id"].astype(str).tolist(),
        p0_probability=r4_prediction["probability"].to_numpy(),
        p0_threshold=float(r4["selected_threshold"]),
        candidates=[
            (
                int(r4add["seed"]),
                r4add_prediction["probability"].to_numpy(),
                float(r4add["selected_threshold"]),
            )
        ],
        repeats=int(args.bootstrap_repeats),
        seed=int(args.bootstrap_seed),
    )
    capacity_r4 = r4["active_parameter_compute_contract"]
    capacity_r4add = r4add["active_parameter_compute_contract"]
    active_capacity_matched = bool(
        capacity_r4["active_parameter_count"]
        == capacity_r4add["active_parameter_count"]
        and capacity_r4["active_parameter_names_sha256"]
        == capacity_r4add["active_parameter_names_sha256"]
        and capacity_r4["active_parameters"]
        == capacity_r4add["active_parameters"]
    )
    learned_compute_matched = bool(
        capacity_r4["compute"]["learned_linear_macs_per_row"]
        == capacity_r4add["compute"]["learned_linear_macs_per_row"]
    )
    bootstrap_metrics = bootstrap["per_seed"][str(r4add["seed"])]
    binary_ci = bootstrap_metrics["binary_f1"]["ci95"]
    macro_ci = bootstrap_metrics["macro_f1"]["ci95"]
    comparison = {
        "schema_version": "tempo-r4-capacity-control-v1",
        "script_version": SCRIPT_VERSION,
        "contrast": "R4-add minus R4",
        "inputs": {
            "r4_result": str(r4_dir / "result.json"),
            "r4_result_sha256": sha256_file(r4_dir / "result.json"),
            "r4add_result": str(r4add_dir / "result.json"),
            "r4add_result_sha256": sha256_file(r4add_dir / "result.json"),
        },
        "matched_contract": {
            "same_seed": bool(r4["seed"] == r4add["seed"]),
            "same_initial_state_sha256": bool(
                r4["matched_initial_state_sha256"]
                == r4add["matched_initial_state_sha256"]
            ),
            "same_full_parameter_signature": bool(
                r4["parameter_signature"] == r4add["parameter_signature"]
            ),
            "same_graph_active_parameters": active_capacity_matched,
            "same_learned_linear_macs_per_row": learned_compute_matched,
            "active_parameter_count": int(
                capacity_r4["active_parameter_count"]
            ),
            "full_parameter_count": int(
                capacity_r4["full_parameter_count"]
            ),
            "active_parameter_names_sha256": capacity_r4[
                "active_parameter_names_sha256"
            ],
            "learned_linear_macs_per_row": int(
                capacity_r4["compute"]["learned_linear_macs_per_row"]
            ),
            "only_treatment_difference": [
                capacity_r4["compute"]["fusion_only_difference"],
                capacity_r4add["compute"]["fusion_only_difference"],
            ],
        },
        "point_estimates": {
            "r4": {
                "selected_epoch": int(r4["best_including_p0"]["epoch"]),
                "binary_f1": float(r4_metrics["best_binary_f1"]),
                "macro_f1": float(
                    r4_metrics["best_macro_f1_at_binary_threshold"]
                ),
                "ap": float(r4_metrics["ap"]),
                "auc": float(r4_metrics["auc"]),
            },
            "r4add": {
                "selected_epoch": int(r4add["best_including_p0"]["epoch"]),
                "binary_f1": float(r4add_metrics["best_binary_f1"]),
                "macro_f1": float(
                    r4add_metrics["best_macro_f1_at_binary_threshold"]
                ),
                "ap": float(r4add_metrics["ap"]),
                "auc": float(r4add_metrics["auc"]),
            },
            "r4add_minus_r4": {
                "binary_f1": float(
                    r4add_metrics["best_binary_f1"]
                    - r4_metrics["best_binary_f1"]
                ),
                "macro_f1": float(
                    r4add_metrics["best_macro_f1_at_binary_threshold"]
                    - r4_metrics["best_macro_f1_at_binary_threshold"]
                ),
                "ap": float(r4add_metrics["ap"] - r4_metrics["ap"]),
                "auc": float(r4add_metrics["auc"] - r4_metrics["auc"]),
            },
        },
        "paired_canonical_event_bootstrap": bootstrap,
        "interpretation": {
            "primary_binary_f1_difference_is_close": bool(
                binary_ci[0] <= 0.0 <= binary_ci[1]
            ),
            "macro_f1_favors_r4_in_this_diagnostic": bool(macro_ci[1] < 0.0),
            "supported_claim": (
                "dual-stream current appearance plus signed/magnitude motion; "
                "multiplicative excitation is the best tested fusion variant "
                "but is not isolated by primary binary F1 in one seed"
            ),
        },
        "test_or_sealed_read": False,
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    atomic_json(output_path, comparison)
    point = comparison["point_estimates"]
    markdown = [
        "# R4 versus capacity-matched R4-add",
        "",
        "Development only; no test/sealed artifact was read.",
        "",
        "| Model | Epoch | Binary F1 | Macro F1 | AP | AUC |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| R4 | {point['r4']['selected_epoch']} | "
            f"{point['r4']['binary_f1']:.6f} | "
            f"{point['r4']['macro_f1']:.6f} | "
            f"{point['r4']['ap']:.6f} | {point['r4']['auc']:.6f} |"
        ),
        (
            f"| R4-add | {point['r4add']['selected_epoch']} | "
            f"{point['r4add']['binary_f1']:.6f} | "
            f"{point['r4add']['macro_f1']:.6f} | "
            f"{point['r4add']['ap']:.6f} | "
            f"{point['r4add']['auc']:.6f} |"
        ),
        "",
        (
            f"Both activate {capacity_r4['active_parameter_count']:,} of "
            f"{capacity_r4['full_parameter_count']:,} declared parameters and "
            f"use {capacity_r4['compute']['learned_linear_macs_per_row']:,} "
            "learned linear MACs per row."
        ),
        "",
        (
            "R4-add minus R4 paired event-bootstrap binary-F1 delta: "
            f"{bootstrap_metrics['binary_f1']['mean_delta']:+.6f}, 95% CI "
            f"[{binary_ci[0]:+.6f},{binary_ci[1]:+.6f}]."
        ),
        (
            "Macro-F1 delta: "
            f"{bootstrap_metrics['macro_f1']['mean_delta']:+.6f}, 95% CI "
            f"[{macro_ci[0]:+.6f},{macro_ci[1]:+.6f}]."
        ),
        "",
        comparison["interpretation"]["supported_claim"] + ".",
        "",
    ]
    (output_dir / "CAPACITY_CONTROL_COMPARISON.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2), flush=True)


def command_run(args: argparse.Namespace) -> None:
    train_path = Path(args.train_cache).expanduser().absolute()
    dev_path = Path(args.dev_cache).expanduser().absolute()
    promoted_path = Path(args.promoted_checkpoint).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    for path, role in (
        (train_path, "train"),
        (dev_path, "dev"),
        (promoted_path, "promoted checkpoint"),
        (output_dir, "output"),
    ):
        guard_development_path(path, role)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = output_dir / f"{args.run_name}_summary.json"
    if run_summary_path.exists():
        raise FileExistsError(f"Refusing existing run summary: {run_summary_path}")

    status_path = output_dir / f"{args.run_name}_status.json"
    atomic_json(
        status_path,
        {
            "status": "loading",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "test_or_sealed_read": False,
        },
    )
    train_cache = load_feature_cache(train_path, ("train_core", "train"))
    dev_cache = load_feature_cache(dev_path, ("dev", "evaluation"))
    validate_development_caches(train_cache, dev_cache)
    checkpoint = torch.load(
        promoted_path, map_location="cpu", weights_only=False
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    print(
        f"[TEMPO] loaded train={len(train_cache['labels'])} "
        f"dev={len(dev_cache['labels'])} device={device}",
        flush=True,
    )
    atomic_json(
        status_path,
        {
            "status": "computing_promoted_p0",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "test_or_sealed_read": False,
        },
    )
    train_base = promoted_outputs(
        checkpoint,
        train_cache,
        batch_size=args.eval_batch_size,
        device=device,
    )
    dev_base = promoted_outputs(
        checkpoint,
        dev_cache,
        batch_size=args.eval_batch_size,
        device=device,
    )
    baseline_probability = torch.sigmoid(dev_base[0]).numpy()
    baseline = probability_metrics(
        dev_cache["labels"],
        baseline_probability,
        dev_cache["availability_signatures"],
    )
    expected = checkpoint["dev"]
    for key in ("best_binary_f1", "ap", "auc"):
        if not math.isclose(
            float(baseline[key]),
            float(expected[key]),
            abs_tol=2e-6,
            rel_tol=0.0,
        ):
            raise RuntimeError(
                f"Promoted P0 reproduction mismatch for {key}: "
                f"{baseline[key]} vs {expected[key]}"
            )
    if not math.isclose(
        float(baseline["best_binary_f1_threshold"]),
        float(checkpoint["locked_threshold_candidate"]),
        abs_tol=2e-7,
        rel_tol=0.0,
    ):
        raise RuntimeError(
            "Promoted P0 threshold reproduction mismatch: "
            f"{baseline['best_binary_f1_threshold']} vs "
            f"{checkpoint['locked_threshold_candidate']}"
        )
    baseline_threshold = float(baseline["best_binary_f1_threshold"])
    baseline["event_operating_audit"] = event_operating_audit(
        dev_cache["labels"].numpy(),
        baseline_probability,
        dev_cache["event_ids"],
        threshold=baseline_threshold,
    )
    baseline["all_negative_event_fp"] = baseline[
        "event_operating_audit"
    ]["all_negative"]
    shuffled_dev = shuffle_history(
        _features_for_mode(dev_cache, "universal"),
        dev_cache["valid_mask"],
        seed=int(args.shuffle_seed),
    )
    print(
        f"[TEMPO] P0 reproduced F1={baseline['best_binary_f1']:.6f} "
        f"macro={baseline['best_macro_f1_at_binary_threshold']:.6f} "
        f"AP={baseline['ap']:.6f}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    common_signature: Mapping[str, Any] | None = None
    for seed in args.seeds:
        initial_state, signature, initial_sha256 = build_matched_initial_state(
            int(train_cache["features"].shape[-1]),
            model_dim=int(args.model_dim),
            residual_cap=float(args.residual_cap),
            seed=int(seed),
        )
        if common_signature is None:
            common_signature = signature
        elif common_signature != signature:
            raise RuntimeError("Parameter signature differs across seeds.")
        for arm in args.arms:
            atomic_json(
                status_path,
                {
                    "status": "training",
                    "arm": arm,
                    "seed": int(seed),
                    "completed_runs": len(results),
                    "test_or_sealed_read": False,
                },
            )
            result = train_one(
                arm=arm,
                seed=int(seed),
                initial_state=initial_state,
                signature=signature,
                initial_sha256=initial_sha256,
                train_cache=train_cache,
                dev_cache=dev_cache,
                train_base=train_base,
                dev_base=dev_base,
                shuffled_dev_features=shuffled_dev,
                output_dir=output_dir,
                device=device,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                eval_batch_size=int(args.eval_batch_size),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                sensor_aux_weight=float(args.sensor_aux_weight),
                residual_l2=float(args.residual_l2),
                null_loss_weight=float(args.null_loss_weight),
                null_target_probability=float(args.null_target_probability),
                early_stop_patience=int(args.early_stop_patience),
                model_dim=int(args.model_dim),
                residual_cap=float(args.residual_cap),
            )
            results.append(result)

    summary = {
        "schema_version": "tempo-legacy360-global-screen-summary-v1",
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "run_name": args.run_name,
        "arms": list(args.arms),
        "seeds": [int(value) for value in args.seeds],
        "p0": baseline,
        "runs": results,
        "matched_parameter_signature": common_signature,
        "inputs": {
            "train_cache": str(train_path),
            "train_manifest": train_cache["manifest"],
            "dev_cache": str(dev_path),
            "dev_manifest": dev_cache["manifest"],
            "promoted_checkpoint": str(promoted_path),
            "promoted_checkpoint_sha256": sha256_file(promoted_path),
        },
        "guardrails": {
            "test_or_sealed_read": False,
            "epoch_zero_exact_p0_fallback": True,
            "same_parameters_steps_initialization_within_seed": True,
            "history_shuffle_threshold_reoptimized": False,
        },
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    atomic_json(run_summary_path, summary)
    write_results_markdown(
        output_dir, baseline, results, run_name=args.run_name
    )
    atomic_json(
        status_path,
        {
            "status": "complete",
            "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "runs": len(results),
            "test_or_sealed_read": False,
        },
    )
    print(json.dumps(summary["guardrails"], indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run matched development-only arms.")
    run.set_defaults(function=command_run)
    run.add_argument(
        "--train-cache",
        default=str(
            DEFAULT_FORMAL_ROOT
            / "features"
            / "train_core_universal_s2hybrid.pt"
        ),
    )
    run.add_argument(
        "--dev-cache",
        default=str(
            DEFAULT_FORMAL_ROOT / "features" / "dev_universal_s2hybrid.pt"
        ),
    )
    run.add_argument(
        "--promoted-checkpoint",
        default=str(
            DEFAULT_FORMAL_ROOT
            / "heads"
            / "full_cache_dev_promotion_v1"
            / "promote_compact_scale_aware_d64_lr1e4"
            / "checkpoint_best.pth"
        ),
    )
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    run.add_argument("--run-name", required=True)
    run.add_argument("--arms", type=parse_arm_csv, default=list(ARMS))
    run.add_argument("--seeds", type=parse_int_csv, default=[42])
    run.add_argument("--epochs", type=int, default=2, choices=(1, 2, 3, 4))
    run.add_argument("--batch-size", type=int, default=1024)
    run.add_argument("--eval-batch-size", type=int, default=4096)
    run.add_argument("--learning-rate", type=float, default=3e-4)
    run.add_argument("--weight-decay", type=float, default=0.01)
    run.add_argument("--sensor-aux-weight", type=float, default=0.05)
    run.add_argument("--residual-l2", type=float, default=1e-3)
    run.add_argument("--null-loss-weight", type=float, default=0.05)
    run.add_argument("--null-target-probability", type=float, default=0.35)
    run.add_argument("--early-stop-patience", type=int, default=1)
    run.add_argument("--residual-cap", type=float, default=1.0)
    run.add_argument("--model-dim", type=int, default=48)
    run.add_argument("--shuffle-seed", type=int, default=20260728)
    run.add_argument("--device", default="cuda:0")
    aggregate = subparsers.add_parser(
        "aggregate", help="Aggregate completed one-arm multi-seed dev runs."
    )
    aggregate.set_defaults(function=command_aggregate)
    aggregate.add_argument("--summary", required=True)
    aggregate.add_argument("--bootstrap-repeats", type=int, default=10000)
    aggregate.add_argument("--bootstrap-seed", type=int, default=20260728)
    combine = subparsers.add_parser(
        "combine-run-summaries",
        help="Combine immutable development summaries for multi-seed aggregate.",
    )
    combine.set_defaults(function=command_combine_run_summaries)
    combine.add_argument("--summaries", nargs="+", required=True)
    combine.add_argument("--output", required=True)
    combine.add_argument("--run-name", required=True)
    compare = subparsers.add_parser(
        "compare-fixed-ensembles",
        help="Compare R1/R4 and fixed 0.5 equal-logit blends.",
    )
    compare.set_defaults(function=command_compare_fixed_ensembles)
    compare.add_argument("--r1-predictions", required=True)
    compare.add_argument("--r4-predictions", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--bootstrap-repeats", type=int, default=10000)
    compare.add_argument("--bootstrap-seed", type=int, default=20260728)
    capacity = subparsers.add_parser(
        "compare-capacity-control",
        help="Compare fixed R4 and graph-capacity-matched R4-add dev runs.",
    )
    capacity.set_defaults(function=command_compare_capacity_control)
    capacity.add_argument("--r4-dir", required=True)
    capacity.add_argument("--r4add-dir", required=True)
    capacity.add_argument("--output-dir", required=True)
    capacity.add_argument("--bootstrap-repeats", type=int, default=10000)
    capacity.add_argument("--bootstrap-seed", type=int, default=20260728)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
