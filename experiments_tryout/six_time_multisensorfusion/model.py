"""Two-axis Panopticon model for four sensors and six observations.

The two axes intentionally do *not* share encoder parameters:

* appearance axis: only the current image (role 0) is encoded;
* transient axis: all valid roles are independently encoded by one second
  encoder.  The same transient encoder instance is reused for all six roles,
  so ``z_0 - z_h`` is computed in one feature space.

For sensor ``s`` and historical role ``h`` the transient branch computes::

    delta_h = z_0 - z_h
    magnitude_h = abs(delta_h)
    d_h = MLP([delta_h, magnitude_h])
    alpha_h = masked_softmax(gate(day_gap, quality, role, similarity))
    v = sum_h alpha_h * d_h
    B_s = stop_gradient(A_s) + zero_initialised_residual_s(v)
    final_s = 0.5 * stop_gradient(A_s) + 0.5 * B_s

Available sensors are fused with a fixed masked mean.  This deliberately
avoids learning a sensor-fusion gate from the very small all-sensor overlap.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# Must be set before importing the DINOv2 implementation on CPU hosts.
os.environ.setdefault("XFORMERS_DISABLED", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_TIMEPOINTS = 6
TIME_ROLE_NAMES = (
    "current",
    "prev1",
    "prev2",
    "prev3",
    "seasonal",
    "year",
)
DEFAULT_SENSOR_ORDER = ("s2_cdse", "l89", "emit", "s5p")


@dataclass(frozen=True)
class SensorNormalizationStats:
    """Per-band statistics computed from the training split only."""

    mean: Sequence[float]
    std: Sequence[float]


@dataclass(frozen=True)
class SensorSixTimeInput:
    """One sensor's input for a common batch of plume/crop rows.

    Canonical tensor shapes are:

    * ``images``: ``[B, 6, C, H, W]``;
    * ``channel_ids``: ``[C]``, ``[B,C]``, ``[B,6,C]`` or the corresponding
      SRF forms ending in 1/2 values per channel;
    * ``time_valid``: boolean ``[B,6]``, where True means the image exists;
    * ``time_delta_days``: ``[B,6]`` (current should normally be zero);
    * ``quality``: optional ``[B,6,Q]``;
    * ``spectral_mask``: optional boolean channel mask, where True means that
      Panopticon must ignore the channel;
    * ``history_duplicate``: optional boolean ``[B,6]`` metadata supplied to
      the gate.  It does not silently change ``time_valid``.

    A configured sensor may be omitted entirely from the model input.  Within
    a supplied sensor, rows without a current image are also supported.  Every
    batch row must nevertheless have a current image from at least one sensor.
    """

    images: torch.Tensor
    channel_ids: torch.Tensor
    time_valid: torch.Tensor
    time_delta_days: torch.Tensor
    quality: Optional[torch.Tensor] = None
    spectral_mask: Optional[torch.Tensor] = None
    history_duplicate: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TwoAxisFusionOutput:
    """Loss-ready logits and auditable temporal diagnostics.

    ``fused_logits`` and ``sensor_logits`` are the fixed 0.5/0.5 final scores.
    Invalid sensor positions are exactly zero and must be interpreted together
    with ``sensor_valid``.
    """

    fused_logits: torch.Tensor
    sensor_logits: torch.Tensor
    sensor_valid: torch.Tensor
    appearance_fused_logits: torch.Tensor
    transient_fused_logits: torch.Tensor
    appearance_sensor_logits: torch.Tensor
    transient_sensor_logits: torch.Tensor
    residual_sensor_logits: torch.Tensor
    history_attention: torch.Tensor
    history_valid: torch.Tensor
    effective_time_valid: torch.Tensor
    appearance_features: Optional[torch.Tensor] = None
    transient_features: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class _PreparedSensorInput:
    images: torch.Tensor
    channel_ids: torch.Tensor
    time_valid: torch.Tensor
    time_delta_days: torch.Tensor
    quality: torch.Tensor
    spectral_mask: Optional[torch.Tensor]
    history_duplicate: torch.Tensor


class _SensorStandardizer(nn.Module):
    """Registered-buffer normalization for one sensor."""

    def __init__(self, stats: SensorNormalizationStats) -> None:
        super().__init__()
        mean = torch.as_tensor(stats.mean, dtype=torch.float32)
        std = torch.as_tensor(stats.std, dtype=torch.float32)
        if mean.ndim != 1 or std.ndim != 1 or mean.numel() == 0:
            raise ValueError("normalization mean/std must be non-empty 1-D arrays")
        if mean.shape != std.shape:
            raise ValueError("normalization mean and std must have identical shapes")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise ValueError("normalization mean/std must be finite")
        if not bool((std > 0).all()):
            raise ValueError("every normalization std must be positive")
        self.register_buffer("mean", mean.reshape(1, 1, -1, 1, 1))
        self.register_buffer("std", std.reshape(1, 1, -1, 1, 1))

    def forward(self, images: torch.Tensor, time_valid: torch.Tensor) -> torch.Tensor:
        if images.shape[2] != self.mean.shape[2]:
            raise ValueError(
                f"image has {images.shape[2]} bands, but train statistics have "
                f"{self.mean.shape[2]} bands"
            )
        valid = time_valid[:, :, None, None, None]
        # Remove invalid NaN/Inf payloads before arithmetic.  Valid non-finite
        # data are rejected immediately before the encoder call.
        safe = torch.where(valid, images, torch.zeros_like(images))
        normalized = (safe - self.mean.to(images)) / self.std.to(images)
        return torch.where(valid, normalized, torch.zeros_like(normalized))


class BinaryAppearanceHead(nn.Module):
    """Sensor-specific LayerNorm + scalar logit head."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(features)).squeeze(-1)


def _masked_softmax(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    """Softmax with exact zeros, including rows without a valid element."""

    if logits.shape != valid_mask.shape:
        raise ValueError("logits and valid_mask must have identical shapes")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
    masked = safe_logits.masked_fill(~valid_mask, -1.0e4)
    weights = torch.softmax(masked, dim=dim)
    weights = weights * valid_mask.to(weights.dtype)
    denominator = weights.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return weights / denominator


def _encoder_feature_dim(encoder: nn.Module) -> Optional[int]:
    value = getattr(encoder, "embed_dim", None)
    if value is None:
        value = getattr(encoder, "feature_dim", None)
    return int(value) if value is not None else None


def _assert_axes_do_not_share_parameters(
    appearance_encoder: nn.Module,
    transient_encoder: nn.Module,
) -> None:
    """Reject both direct aliases and less-obvious shared parameter storage."""

    if appearance_encoder is transient_encoder:
        raise ValueError("appearance_encoder and transient_encoder must be different modules")

    appearance_ids = {id(parameter) for parameter in appearance_encoder.parameters()}
    transient_ids = {id(parameter) for parameter in transient_encoder.parameters()}
    if appearance_ids & transient_ids:
        raise ValueError("appearance and transient encoders share Parameter objects")

    appearance_storage = {
        parameter.untyped_storage().data_ptr()
        for parameter in appearance_encoder.parameters()
        if parameter.numel() > 0
    }
    transient_storage = {
        parameter.untyped_storage().data_ptr()
        for parameter in transient_encoder.parameters()
        if parameter.numel() > 0
    }
    if appearance_storage & transient_storage:
        raise ValueError("appearance and transient encoders share parameter storage")


class SixTimeMultiSensorTwoAxisModel(nn.Module):
    """Independent appearance/transient encoders with fixed score fusion."""

    def __init__(
        self,
        appearance_encoder: nn.Module,
        transient_encoder: nn.Module,
        *,
        sensors: Sequence[str] = DEFAULT_SENSOR_ORDER,
        feature_dim: Optional[int] = None,
        quality_dim: int = 0,
        pair_hidden_dim: int = 256,
        transient_dim: int = 128,
        gate_hidden_dim: int = 64,
        role_embedding_dim: int = 16,
        sensor_embedding_dim: int = 16,
        dropout: float = 0.05,
        residual_cap: float = 4.0,
        day_gap_scale: float = 365.0,
        encode_chunk_size: Optional[int] = None,
        normalization_stats: Optional[
            Mapping[str, SensorNormalizationStats | tuple[Sequence[float], Sequence[float]]]
        ] = None,
        freeze_appearance_encoder: bool = True,
        freeze_transient_encoder: bool = True,
    ) -> None:
        super().__init__()
        _assert_axes_do_not_share_parameters(appearance_encoder, transient_encoder)

        sensor_order = tuple(str(sensor).strip().lower() for sensor in sensors)
        if not sensor_order or any(not sensor for sensor in sensor_order):
            raise ValueError("sensors must contain at least one non-empty name")
        if len(set(sensor_order)) != len(sensor_order):
            raise ValueError("sensor names must be unique")
        if quality_dim < 0:
            raise ValueError("quality_dim cannot be negative")
        for name, value in (
            ("pair_hidden_dim", pair_hidden_dim),
            ("transient_dim", transient_dim),
            ("gate_hidden_dim", gate_hidden_dim),
            ("role_embedding_dim", role_embedding_dim),
            ("sensor_embedding_dim", sensor_embedding_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if residual_cap <= 0 or day_gap_scale <= 0:
            raise ValueError("residual_cap and day_gap_scale must be positive")
        if encode_chunk_size is not None and int(encode_chunk_size) <= 0:
            raise ValueError("encode_chunk_size must be positive or None")

        inferred_a = _encoder_feature_dim(appearance_encoder)
        inferred_b = _encoder_feature_dim(transient_encoder)
        if feature_dim is None:
            if inferred_a is None or inferred_b is None:
                raise ValueError(
                    "feature_dim is required when an encoder has no embed_dim/feature_dim attribute"
                )
            if inferred_a != inferred_b:
                raise ValueError(
                    f"axis encoder dimensions differ: appearance={inferred_a}, transient={inferred_b}"
                )
            feature_dim = inferred_a
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        if inferred_a is not None and inferred_a != int(feature_dim):
            raise ValueError("appearance encoder dimension does not match feature_dim")
        if inferred_b is not None and inferred_b != int(feature_dim):
            raise ValueError("transient encoder dimension does not match feature_dim")

        self.appearance_encoder = appearance_encoder
        self.transient_encoder = transient_encoder
        self.sensor_order = sensor_order
        self.sensor_to_index = {
            sensor: index for index, sensor in enumerate(self.sensor_order)
        }
        self.num_sensors = len(self.sensor_order)
        self.feature_dim = int(feature_dim)
        self.quality_dim = int(quality_dim)
        self.transient_dim = int(transient_dim)
        self.residual_cap = float(residual_cap)
        self.day_gap_scale = float(day_gap_scale)
        self.encode_chunk_size = (
            int(encode_chunk_size) if encode_chunk_size is not None else None
        )

        self.appearance_heads = nn.ModuleDict(
            {
                sensor: BinaryAppearanceHead(self.feature_dim)
                for sensor in self.sensor_order
            }
        )
        self.transient_feature_norms = nn.ModuleDict(
            {sensor: nn.LayerNorm(self.feature_dim) for sensor in self.sensor_order}
        )
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, int(pair_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(pair_hidden_dim), self.transient_dim),
            nn.GELU(),
        )
        self.pair_sensor_embedding = nn.Embedding(
            self.num_sensors, self.transient_dim
        )
        self.gate_role_embedding = nn.Embedding(
            NUM_TIMEPOINTS - 1, int(role_embedding_dim)
        )
        self.gate_sensor_embedding = nn.Embedding(
            self.num_sensors, int(sensor_embedding_dim)
        )
        gate_input_dim = (
            1
            + 2 * self.quality_dim
            + 1
            + 1
            + int(role_embedding_dim)
            + int(sensor_embedding_dim)
        )
        self.history_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, int(gate_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_dim), 1),
        )

        # Sensor-private readouts retain a shared low-capacity delta MLP while
        # allowing each modality to calibrate its residual independently.
        self.residual_weight = nn.Parameter(
            torch.zeros(self.num_sensors, self.transient_dim)
        )
        self.residual_bias = nn.Parameter(torch.zeros(self.num_sensors))

        # A registered buffer makes the fixed contract visible in checkpoints
        # without making the 0.5/0.5 weights trainable.
        self.register_buffer("axis_score_weights", torch.tensor([0.5, 0.5]))

        self.normalizers = nn.ModuleDict()
        if normalization_stats:
            unknown = set(normalization_stats) - set(self.sensor_order)
            if unknown:
                raise ValueError(f"normalization stats contain unknown sensors: {sorted(unknown)}")
            for sensor, raw_stats in normalization_stats.items():
                if isinstance(raw_stats, SensorNormalizationStats):
                    stats = raw_stats
                else:
                    if len(raw_stats) != 2:
                        raise ValueError(f"normalization stats for {sensor} must be (mean, std)")
                    stats = SensorNormalizationStats(raw_stats[0], raw_stats[1])
                self.normalizers[sensor] = _SensorStandardizer(stats)

        self._appearance_encoder_frozen = False
        self._transient_encoder_frozen = False
        self.set_encoder_frozen("appearance", freeze_appearance_encoder)
        self.set_encoder_frozen("transient", freeze_transient_encoder)

    def set_encoder_frozen(self, axis: str, frozen: bool) -> None:
        """Freeze/unfreeze one axis encoder and preserve eval mode when frozen."""

        axis = str(axis).strip().lower()
        if axis == "appearance":
            encoder = self.appearance_encoder
            self._appearance_encoder_frozen = bool(frozen)
        elif axis == "transient":
            encoder = self.transient_encoder
            self._transient_encoder_frozen = bool(frozen)
        else:
            raise ValueError("axis must be 'appearance' or 'transient'")
        encoder.requires_grad_(not frozen)
        if frozen:
            encoder.eval()
        elif self.training:
            encoder.train()

    def configure_training_stage(
        self,
        stage: str,
        *,
        train_axis_encoder: bool = False,
    ) -> None:
        """Configure parameters for independent A or B training.

        ``appearance`` trains only the A heads unless ``train_axis_encoder`` is
        true. ``transient`` freezes all of A and trains the B delta/gate/readout
        plus, optionally, Encoder B. ``inference`` freezes everything.
        """

        stage = str(stage).strip().lower()
        transient_modules = (
            self.transient_feature_norms,
            self.pair_mlp,
            self.pair_sensor_embedding,
            self.gate_role_embedding,
            self.gate_sensor_embedding,
            self.history_gate,
        )
        if stage == "appearance":
            self.appearance_heads.requires_grad_(True)
            self.set_encoder_frozen("appearance", not train_axis_encoder)
            self.set_encoder_frozen("transient", True)
            for module in transient_modules:
                module.requires_grad_(False)
            self.residual_weight.requires_grad_(False)
            self.residual_bias.requires_grad_(False)
        elif stage == "transient":
            self.appearance_heads.requires_grad_(False)
            self.set_encoder_frozen("appearance", True)
            self.set_encoder_frozen("transient", not train_axis_encoder)
            for module in transient_modules:
                module.requires_grad_(True)
            self.residual_weight.requires_grad_(True)
            self.residual_bias.requires_grad_(True)
        elif stage == "inference":
            self.requires_grad_(False)
            self._appearance_encoder_frozen = True
            self._transient_encoder_frozen = True
            self.appearance_encoder.eval()
            self.transient_encoder.eval()
        else:
            raise ValueError("stage must be 'appearance', 'transient', or 'inference'")

    def train(self, mode: bool = True) -> "SixTimeMultiSensorTwoAxisModel":
        super().train(mode)
        # nn.Module.train() recurses, so force frozen encoders back to eval.
        if self._appearance_encoder_frozen:
            self.appearance_encoder.eval()
        if self._transient_encoder_frozen:
            self.transient_encoder.eval()
        return self

    @staticmethod
    def _expand_channel_ids(
        channel_ids: torch.Tensor,
        *,
        batch_size: int,
        num_channels: int,
    ) -> torch.Tensor:
        """Broadcast channel IDs to [B,6,C] or [B,6,C,K]."""

        ids = channel_ids
        if ids.ndim == 1 and ids.shape[0] == num_channels:
            return ids.reshape(1, 1, num_channels).expand(
                batch_size, NUM_TIMEPOINTS, -1
            )
        if ids.ndim == 2:
            if tuple(ids.shape) == (batch_size, num_channels):
                return ids[:, None, :].expand(-1, NUM_TIMEPOINTS, -1)
            if ids.shape[0] == num_channels and ids.shape[1] in (1, 2):
                return ids.reshape(1, 1, num_channels, ids.shape[1]).expand(
                    batch_size, NUM_TIMEPOINTS, -1, -1
                )
        if ids.ndim == 3:
            if tuple(ids.shape) == (batch_size, NUM_TIMEPOINTS, num_channels):
                return ids
            if (
                ids.shape[0] == batch_size
                and ids.shape[1] == num_channels
                and ids.shape[2] in (1, 2)
            ):
                return ids[:, None, :, :].expand(-1, NUM_TIMEPOINTS, -1, -1)
        if (
            ids.ndim == 4
            and ids.shape[0] == batch_size
            and ids.shape[1] == NUM_TIMEPOINTS
            and ids.shape[2] == num_channels
            and ids.shape[3] in (1, 2)
        ):
            return ids
        raise ValueError(
            "channel_ids must be [C], [B,C], [B,6,C], [C,K], [B,C,K], "
            "or [B,6,C,K] with K in {1,2}; got "
            f"{tuple(ids.shape)}"
        )

    @staticmethod
    def _expand_spectral_mask(
        spectral_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        num_channels: int,
    ) -> Optional[torch.Tensor]:
        if spectral_mask is None:
            return None
        if spectral_mask.dtype != torch.bool:
            raise TypeError("spectral_mask must be boolean (True means masked)")
        mask = spectral_mask
        if mask.ndim == 1 and mask.shape[0] == num_channels:
            return mask.reshape(1, 1, num_channels).expand(
                batch_size, NUM_TIMEPOINTS, -1
            )
        if tuple(mask.shape) == (batch_size, num_channels):
            return mask[:, None, :].expand(-1, NUM_TIMEPOINTS, -1)
        if tuple(mask.shape) == (batch_size, NUM_TIMEPOINTS, num_channels):
            return mask
        raise ValueError(
            "spectral_mask must be [C], [B,C], or [B,6,C]; got "
            f"{tuple(mask.shape)}"
        )

    def _prepare_sensor_input(
        self,
        sensor: str,
        sensor_input: SensorSixTimeInput,
        *,
        expected_batch_size: Optional[int],
    ) -> _PreparedSensorInput:
        images = sensor_input.images
        if images.ndim != 5 or images.shape[1] != NUM_TIMEPOINTS:
            raise ValueError(
                f"{sensor}.images must be [B,{NUM_TIMEPOINTS},C,H,W], got "
                f"{tuple(images.shape)}"
            )
        batch_size, _, num_channels, height, width = images.shape
        if batch_size <= 0 or num_channels <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"{sensor}.images dimensions must all be positive")
        if expected_batch_size is not None and batch_size != expected_batch_size:
            raise ValueError("all supplied sensors must have the same batch size")
        if sensor_input.time_valid.shape != (batch_size, NUM_TIMEPOINTS):
            raise ValueError(f"{sensor}.time_valid must have shape [B,6]")
        if sensor_input.time_valid.dtype != torch.bool:
            raise TypeError(f"{sensor}.time_valid must be boolean")
        if sensor_input.time_delta_days.shape != (batch_size, NUM_TIMEPOINTS):
            raise ValueError(f"{sensor}.time_delta_days must have shape [B,6]")
        if sensor_input.time_delta_days.device != images.device:
            raise ValueError(f"{sensor} metadata and images must be on the same device")
        if sensor_input.time_valid.device != images.device:
            raise ValueError(f"{sensor}.time_valid and images must be on the same device")

        if self.quality_dim:
            if sensor_input.quality is None:
                raise ValueError(
                    f"{sensor}.quality is required because quality_dim={self.quality_dim}"
                )
            if sensor_input.quality.shape != (
                batch_size,
                NUM_TIMEPOINTS,
                self.quality_dim,
            ):
                raise ValueError(
                    f"{sensor}.quality must have shape [B,6,{self.quality_dim}]"
                )
            quality = sensor_input.quality
        else:
            if sensor_input.quality is not None and sensor_input.quality.shape[-1] != 0:
                raise ValueError("quality was supplied but model quality_dim is zero")
            quality = images.new_zeros((batch_size, NUM_TIMEPOINTS, 0))
        if quality.device != images.device:
            raise ValueError(f"{sensor}.quality and images must be on the same device")

        duplicate = sensor_input.history_duplicate
        if duplicate is None:
            duplicate = torch.zeros(
                (batch_size, NUM_TIMEPOINTS),
                dtype=torch.bool,
                device=images.device,
            )
        elif duplicate.shape != (batch_size, NUM_TIMEPOINTS) or duplicate.dtype != torch.bool:
            raise ValueError(f"{sensor}.history_duplicate must be bool [B,6]")

        channel_ids = self._expand_channel_ids(
            sensor_input.channel_ids,
            batch_size=batch_size,
            num_channels=num_channels,
        )
        if channel_ids.device != images.device:
            raise ValueError(f"{sensor}.channel_ids and images must be on the same device")
        spectral_mask = self._expand_spectral_mask(
            sensor_input.spectral_mask,
            batch_size=batch_size,
            num_channels=num_channels,
        )
        if spectral_mask is not None and spectral_mask.device != images.device:
            raise ValueError(f"{sensor}.spectral_mask and images must be on the same device")

        # Histories without a current observation are not meaningful for this
        # two-axis score and are removed before any encoder/gate operation.
        effective_valid = sensor_input.time_valid & sensor_input.time_valid[:, :1]
        if not bool(torch.isfinite(sensor_input.time_delta_days[effective_valid]).all()):
            raise ValueError(f"{sensor}.time_delta_days contains NaN or Inf at a valid role")
        if self.quality_dim and not bool(torch.isfinite(quality[effective_valid]).all()):
            raise ValueError(f"{sensor}.quality contains NaN or Inf at a valid role")
        if sensor in self.normalizers:
            images = self.normalizers[sensor](images, effective_valid)
        else:
            images = torch.where(
                effective_valid[:, :, None, None, None],
                images,
                torch.zeros_like(images),
            )

        time_delta_days = torch.where(
            effective_valid,
            torch.nan_to_num(
                sensor_input.time_delta_days.to(dtype=images.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            torch.zeros_like(sensor_input.time_delta_days, dtype=images.dtype),
        )
        quality = torch.where(
            effective_valid.unsqueeze(-1),
            torch.nan_to_num(quality.to(dtype=images.dtype)),
            torch.zeros_like(quality, dtype=images.dtype),
        )
        duplicate = duplicate & effective_valid
        return _PreparedSensorInput(
            images=images,
            channel_ids=channel_ids,
            time_valid=effective_valid,
            time_delta_days=time_delta_days,
            quality=quality,
            spectral_mask=spectral_mask,
            history_duplicate=duplicate,
        )

    def _call_encoder(
        self,
        encoder: nn.Module,
        images: torch.Tensor,
        channel_ids: torch.Tensor,
        spectral_mask: Optional[torch.Tensor],
        *,
        frozen: bool,
    ) -> torch.Tensor:
        if images.shape[0] == 0:
            return images.new_zeros((0, self.feature_dim))
        if not bool(torch.isfinite(images).all()):
            raise ValueError("a valid image contains NaN or Inf")
        if not bool(torch.isfinite(channel_ids).all()):
            raise ValueError("valid channel_ids contain NaN or Inf")
        if spectral_mask is not None and not bool((~spectral_mask).any(dim=1).all()):
            raise ValueError("every valid image needs at least one unmasked channel")

        chunk_size = self.encode_chunk_size or int(images.shape[0])
        outputs = []
        grad_context = torch.no_grad() if frozen else nullcontext()
        with grad_context:
            for start in range(0, int(images.shape[0]), chunk_size):
                end = min(start + chunk_size, int(images.shape[0]))
                x_dict: dict[str, torch.Tensor] = {
                    "imgs": images[start:end],
                    # Panopticon's channel embedder mutates a view internally.
                    "chn_ids": channel_ids[start:end].clone(),
                }
                if spectral_mask is not None:
                    x_dict["spec_masks"] = spectral_mask[start:end]
                if hasattr(encoder, "forward_features"):
                    encoded = encoder.forward_features(x_dict)  # type: ignore[attr-defined]
                else:
                    encoded = encoder(x_dict)
                if isinstance(encoded, Mapping):
                    if "x_norm_clstoken" not in encoded:
                        raise KeyError("encoder output is missing x_norm_clstoken")
                    encoded = encoded["x_norm_clstoken"]
                if not isinstance(encoded, torch.Tensor):
                    raise TypeError("encoder must return a tensor or x_norm_clstoken mapping")
                if encoded.shape != (end - start, self.feature_dim):
                    raise ValueError(
                        "encoder CLS output must have shape "
                        f"[N,{self.feature_dim}], got {tuple(encoded.shape)}"
                    )
                if not bool(torch.isfinite(encoded).all()):
                    raise ValueError("encoder returned NaN or Inf CLS features")
                outputs.append(encoded)
        return torch.cat(outputs, dim=0)

    def _encode_current(
        self,
        sensor_input: _PreparedSensorInput,
        *,
        feature_dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = sensor_input.images.shape[0]
        active = torch.nonzero(sensor_input.time_valid[:, 0], as_tuple=False).flatten()
        output = torch.zeros(
            batch_size,
            self.feature_dim,
            device=sensor_input.images.device,
            dtype=feature_dtype,
        )
        if active.numel() == 0:
            return output
        features = self._call_encoder(
            self.appearance_encoder,
            sensor_input.images[:, 0].index_select(0, active),
            sensor_input.channel_ids[:, 0].index_select(0, active),
            None
            if sensor_input.spectral_mask is None
            else sensor_input.spectral_mask[:, 0].index_select(0, active),
            frozen=self._appearance_encoder_frozen,
        ).to(dtype=feature_dtype)
        return output.index_copy(0, active, features)

    def _encode_all_times(
        self,
        sensor_input: _PreparedSensorInput,
        *,
        feature_dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = sensor_input.images.shape[0]
        flat_valid = sensor_input.time_valid.reshape(-1)
        active = torch.nonzero(flat_valid, as_tuple=False).flatten()
        output = torch.zeros(
            batch_size * NUM_TIMEPOINTS,
            self.feature_dim,
            device=sensor_input.images.device,
            dtype=feature_dtype,
        )
        if active.numel() == 0:
            return output.reshape(batch_size, NUM_TIMEPOINTS, self.feature_dim)
        flat_images = sensor_input.images.flatten(0, 1)
        flat_ids = sensor_input.channel_ids.flatten(0, 1)
        flat_mask = (
            None
            if sensor_input.spectral_mask is None
            else sensor_input.spectral_mask.flatten(0, 1)
        )
        features = self._call_encoder(
            self.transient_encoder,
            flat_images.index_select(0, active),
            flat_ids.index_select(0, active),
            None if flat_mask is None else flat_mask.index_select(0, active),
            frozen=self._transient_encoder_frozen,
        ).to(dtype=feature_dtype)
        return output.index_copy(0, active, features).reshape(
            batch_size, NUM_TIMEPOINTS, self.feature_dim
        )

    @staticmethod
    def _masked_sensor_mean(
        logits: torch.Tensor,
        sensor_valid: torch.Tensor,
    ) -> torch.Tensor:
        weights = sensor_valid.to(logits.dtype)
        return (logits * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        sensor_inputs: Mapping[str, SensorSixTimeInput],
        *,
        return_features: bool = False,
        compute_transient: bool = True,
    ) -> TwoAxisFusionOutput:
        """Encode available sensors and return A/B/final binary logits.

        Set ``compute_transient=False`` while training Axis A to skip all
        Encoder-B calls.  In that mode the diagnostic B/final values equal the
        detached A values and ``transient_features`` is not returned.
        """

        if not sensor_inputs:
            raise ValueError("sensor_inputs cannot be empty")
        normalized_keys = {str(key).strip().lower() for key in sensor_inputs}
        if len(normalized_keys) != len(sensor_inputs):
            raise ValueError("sensor input keys collide after lowercase normalization")
        unknown = normalized_keys - set(self.sensor_order)
        if unknown:
            raise ValueError(f"unknown sensors: {sorted(unknown)}")
        canonical_inputs = {
            str(key).strip().lower(): value for key, value in sensor_inputs.items()
        }
        if not all(isinstance(value, SensorSixTimeInput) for value in canonical_inputs.values()):
            raise TypeError("every sensor input must be SensorSixTimeInput")

        prepared: dict[str, _PreparedSensorInput] = {}
        batch_size: Optional[int] = None
        reference_device: Optional[torch.device] = None
        reference_dtype: Optional[torch.dtype] = None
        for sensor in self.sensor_order:
            if sensor not in canonical_inputs:
                continue
            item = self._prepare_sensor_input(
                sensor,
                canonical_inputs[sensor],
                expected_batch_size=batch_size,
            )
            if batch_size is None:
                batch_size = int(item.images.shape[0])
                reference_device = item.images.device
                reference_dtype = item.images.dtype
            elif item.images.device != reference_device:
                raise ValueError("all supplied sensors must be on the same device")
            elif item.images.dtype != reference_dtype:
                raise ValueError("all supplied sensor images must use the same dtype")
            prepared[sensor] = item
        if batch_size is None or reference_device is None:
            raise AssertionError("validated sensor input unexpectedly produced no batch")

        head_parameter = next(self.appearance_heads.parameters())
        if head_parameter.device != reference_device:
            raise ValueError("model parameters and sensor inputs must be on the same device")
        feature_dtype = head_parameter.dtype

        effective_masks = []
        day_gaps = []
        qualities = []
        duplicates = []
        appearance_features_by_sensor = []
        transient_features_by_sensor = []
        for sensor in self.sensor_order:
            item = prepared.get(sensor)
            if item is None:
                effective_masks.append(
                    torch.zeros(
                        batch_size,
                        NUM_TIMEPOINTS,
                        dtype=torch.bool,
                        device=reference_device,
                    )
                )
                day_gaps.append(
                    torch.zeros(
                        batch_size,
                        NUM_TIMEPOINTS,
                        dtype=feature_dtype,
                        device=reference_device,
                    )
                )
                qualities.append(
                    torch.zeros(
                        batch_size,
                        NUM_TIMEPOINTS,
                        self.quality_dim,
                        dtype=feature_dtype,
                        device=reference_device,
                    )
                )
                duplicates.append(
                    torch.zeros(
                        batch_size,
                        NUM_TIMEPOINTS,
                        dtype=torch.bool,
                        device=reference_device,
                    )
                )
                appearance_features_by_sensor.append(
                    torch.zeros(
                        batch_size,
                        self.feature_dim,
                        dtype=feature_dtype,
                        device=reference_device,
                    )
                )
                transient_features_by_sensor.append(
                    torch.zeros(
                        batch_size,
                        NUM_TIMEPOINTS,
                        self.feature_dim,
                        dtype=feature_dtype,
                        device=reference_device,
                    )
                )
                continue
            effective_masks.append(item.time_valid)
            day_gaps.append(item.time_delta_days.to(dtype=feature_dtype))
            qualities.append(item.quality.to(dtype=feature_dtype))
            duplicates.append(item.history_duplicate)
            appearance_features_by_sensor.append(
                self._encode_current(item, feature_dtype=feature_dtype)
            )
            if compute_transient:
                transient_features_by_sensor.append(
                    self._encode_all_times(item, feature_dtype=feature_dtype)
                )
            else:
                transient_features_by_sensor.append(
                    torch.zeros(
                        batch_size,
                        NUM_TIMEPOINTS,
                        self.feature_dim,
                        dtype=feature_dtype,
                        device=reference_device,
                    )
                )

        effective_time_valid = torch.stack(effective_masks, dim=1)
        sensor_valid = effective_time_valid[:, :, 0]
        if not bool(sensor_valid.any(dim=1).all()):
            invalid_rows = torch.nonzero(~sensor_valid.any(dim=1), as_tuple=False).flatten()
            raise ValueError(
                "every batch row needs at least one valid current sensor; "
                f"invalid row indices={invalid_rows[:20].tolist()}"
            )
        gap_tensor = torch.stack(day_gaps, dim=1)
        quality_tensor = torch.stack(qualities, dim=1)
        duplicate_tensor = torch.stack(duplicates, dim=1)
        appearance_features = torch.stack(appearance_features_by_sensor, dim=1)
        transient_features = torch.stack(transient_features_by_sensor, dim=1)

        appearance_sensor_logits = torch.stack(
            [
                self.appearance_heads[sensor](appearance_features[:, sensor_index])
                for sensor_index, sensor in enumerate(self.sensor_order)
            ],
            dim=1,
        )
        appearance_sensor_logits = torch.where(
            sensor_valid,
            appearance_sensor_logits,
            torch.zeros_like(appearance_sensor_logits),
        )

        if not compute_transient:
            frozen_appearance = appearance_sensor_logits.detach()
            zero_residual = torch.zeros_like(appearance_sensor_logits)
            zero_attention = torch.zeros(
                batch_size,
                self.num_sensors,
                NUM_TIMEPOINTS - 1,
                dtype=feature_dtype,
                device=reference_device,
            )
            history_valid = (
                effective_time_valid[:, :, 1:] & sensor_valid.unsqueeze(-1)
            ).any(dim=-1)
            appearance_fused_logits = self._masked_sensor_mean(
                appearance_sensor_logits, sensor_valid
            )
            detached_fused = self._masked_sensor_mean(
                frozen_appearance, sensor_valid
            )
            return TwoAxisFusionOutput(
                fused_logits=detached_fused,
                sensor_logits=frozen_appearance,
                sensor_valid=sensor_valid,
                appearance_fused_logits=appearance_fused_logits,
                transient_fused_logits=detached_fused,
                appearance_sensor_logits=appearance_sensor_logits,
                transient_sensor_logits=frozen_appearance,
                residual_sensor_logits=zero_residual,
                history_attention=zero_attention,
                history_valid=history_valid,
                effective_time_valid=effective_time_valid,
                appearance_features=appearance_features if return_features else None,
                transient_features=None,
            )

        normalized_times = []
        for sensor_index, sensor in enumerate(self.sensor_order):
            normalized = self.transient_feature_norms[sensor](
                transient_features[:, sensor_index]
            )
            normalized_times.append(
                torch.where(
                    effective_time_valid[:, sensor_index].unsqueeze(-1),
                    normalized,
                    torch.zeros_like(normalized),
                )
            )
        normalized_transient = torch.stack(normalized_times, dim=1)
        current = normalized_transient[:, :, 0]
        history = normalized_transient[:, :, 1:]
        history_mask = (
            effective_time_valid[:, :, 1:] & sensor_valid.unsqueeze(-1)
        )
        history_valid = history_mask.any(dim=-1)

        delta = current.unsqueeze(2) - history
        delta = torch.where(history_mask.unsqueeze(-1), delta, torch.zeros_like(delta))
        pair_input = torch.cat((delta, delta.abs()), dim=-1)
        pair_features = self.pair_mlp(pair_input)
        sensor_indices = torch.arange(
            self.num_sensors, device=reference_device, dtype=torch.long
        )
        pair_features = pair_features + self.pair_sensor_embedding(
            sensor_indices
        ).reshape(1, self.num_sensors, 1, self.transient_dim)
        pair_features = torch.where(
            history_mask.unsqueeze(-1),
            pair_features,
            torch.zeros_like(pair_features),
        )

        similarity = F.cosine_similarity(current.unsqueeze(2), history, dim=-1)
        similarity = torch.where(history_mask, similarity, torch.zeros_like(similarity))
        log_gap = torch.log1p(gap_tensor[:, :, 1:].abs()) / math.log1p(
            self.day_gap_scale
        )
        current_quality = quality_tensor[:, :, :1].expand(
            -1, -1, NUM_TIMEPOINTS - 1, -1
        )
        history_quality = quality_tensor[:, :, 1:]
        role_indices = torch.arange(
            NUM_TIMEPOINTS - 1, device=reference_device, dtype=torch.long
        )
        role_features = self.gate_role_embedding(role_indices).reshape(
            1, 1, NUM_TIMEPOINTS - 1, -1
        ).expand(batch_size, self.num_sensors, -1, -1)
        sensor_gate_features = self.gate_sensor_embedding(sensor_indices).reshape(
            1, self.num_sensors, 1, -1
        ).expand(batch_size, -1, NUM_TIMEPOINTS - 1, -1)
        gate_input = torch.cat(
            (
                log_gap.unsqueeze(-1),
                current_quality,
                history_quality,
                similarity.unsqueeze(-1),
                duplicate_tensor[:, :, 1:].to(feature_dtype).unsqueeze(-1),
                role_features,
                sensor_gate_features,
            ),
            dim=-1,
        )
        gate_input = torch.where(
            history_mask.unsqueeze(-1),
            torch.nan_to_num(gate_input),
            torch.zeros_like(gate_input),
        )
        history_gate_logits = self.history_gate(gate_input).squeeze(-1)
        history_attention = _masked_softmax(
            history_gate_logits,
            history_mask,
            dim=-1,
        )
        temporal_evidence = (
            pair_features * history_attention.unsqueeze(-1)
        ).sum(dim=2)

        residual_raw = torch.einsum(
            "bsd,sd->bs", temporal_evidence, self.residual_weight
        ) + self.residual_bias.reshape(1, self.num_sensors)
        residual_sensor_logits = self.residual_cap * torch.tanh(
            residual_raw / self.residual_cap
        )
        residual_sensor_logits = torch.where(
            history_valid & sensor_valid,
            residual_sensor_logits,
            torch.zeros_like(residual_sensor_logits),
        )

        frozen_appearance = appearance_sensor_logits.detach()
        transient_sensor_logits = frozen_appearance + residual_sensor_logits
        transient_sensor_logits = torch.where(
            sensor_valid,
            transient_sensor_logits,
            torch.zeros_like(transient_sensor_logits),
        )
        score_weights = self.axis_score_weights.to(
            device=reference_device, dtype=feature_dtype
        )
        sensor_logits = (
            score_weights[0] * frozen_appearance
            + score_weights[1] * transient_sensor_logits
        )
        sensor_logits = torch.where(
            sensor_valid, sensor_logits, torch.zeros_like(sensor_logits)
        )

        appearance_fused_logits = self._masked_sensor_mean(
            appearance_sensor_logits, sensor_valid
        )
        transient_fused_logits = self._masked_sensor_mean(
            transient_sensor_logits, sensor_valid
        )
        fused_logits = self._masked_sensor_mean(sensor_logits, sensor_valid)
        return TwoAxisFusionOutput(
            fused_logits=fused_logits,
            sensor_logits=sensor_logits,
            sensor_valid=sensor_valid,
            appearance_fused_logits=appearance_fused_logits,
            transient_fused_logits=transient_fused_logits,
            appearance_sensor_logits=appearance_sensor_logits,
            transient_sensor_logits=transient_sensor_logits,
            residual_sensor_logits=residual_sensor_logits,
            history_attention=history_attention,
            history_valid=history_valid,
            effective_time_valid=effective_time_valid,
            appearance_features=appearance_features if return_features else None,
            transient_features=transient_features if return_features else None,
        )


def load_train_normalization_stats(
    path: str | Path,
) -> dict[str, SensorNormalizationStats]:
    """Load a write-once train-statistics JSON and reject non-train sources.

    Expected schema::

        {
          "source_split": "train",
          "sensors": {
            "s2_cdse": {"mean": [...], "std": [...]},
            "l89": {"mean": [...], "std": [...]}
          }
        }
    """

    stats_path = Path(path).expanduser().resolve()
    payload: Any = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("normalization JSON must contain an object")
    if str(payload.get("source_split", "")).strip().lower() != "train":
        raise ValueError("normalization JSON source_split must be exactly 'train'")
    sensor_payload = payload.get("sensors")
    if not isinstance(sensor_payload, Mapping) or not sensor_payload:
        raise ValueError("normalization JSON must contain a non-empty sensors object")
    result: dict[str, SensorNormalizationStats] = {}
    for sensor, values in sensor_payload.items():
        if not isinstance(values, Mapping) or "mean" not in values or "std" not in values:
            raise ValueError(f"normalization entry for {sensor!r} needs mean and std")
        result[str(sensor).strip().lower()] = SensorNormalizationStats(
            mean=values["mean"],
            std=values["std"],
        )
    return result


def build_two_axis_panopticon_model(
    weights_path: Optional[str | Path] = None,
    *,
    strict: bool = True,
    **model_kwargs: Any,
) -> SixTimeMultiSensorTwoAxisModel:
    """Build two independent Panopticon ViT-B/14 encoders.

    The checkpoint is loaded twice on purpose.  Do not replace the second load
    with an assignment: Encoder A and Encoder B must own independent parameters.
    """

    from src.backbones import load_panopticon_vitb14

    weights = None if weights_path is None else str(weights_path)
    appearance_encoder = load_panopticon_vitb14(weights, strict=strict)
    transient_encoder = load_panopticon_vitb14(weights, strict=strict)
    return SixTimeMultiSensorTwoAxisModel(
        appearance_encoder,
        transient_encoder,
        **model_kwargs,
    )
