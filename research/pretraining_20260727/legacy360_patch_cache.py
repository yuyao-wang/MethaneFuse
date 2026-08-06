#!/usr/bin/env python3
"""One-pass patch-token cache for legacy-preserving 360 m axial training.

This module separates expensive image/Panopticon work from the repeated
optimization loop:

``extract``
    Reads every source image once, writes local ragged shards containing
    projected per-observation patch tokens, projected legacy concat features,
    and *exact* cached legacy logits.

``train-head``
    Reads only those local shards.  A same-location temporal attention block
    operates on the patch axis, followed by masked sensor-set attention.  All
    paths that can change a logit are zero initialized, so epoch zero exactly
    reproduces the cached legacy fused and per-sensor logits.

The fixed projection is generated from a seeded Gaussian matrix followed by a
reduced QR decomposition and sign canonicalization.  The resulting 768 -> 64
matrix is stored in each cache manifest together with its SHA-256 and
orthogonality error.  Patch tensors are float16 and ragged:

``patches_{early,final}[sum(P_i),64]``
    Concatenated projected patch tokens.
``offsets[num_observations+1]``
    Slices for each valid observation.
``observation_{row,sensor,role}[num_observations]``
    Coordinates needed to rebuild ``[row,sensor,time,patch,64]`` groups.

No test or sealed manifest is opened by default.  Historical final evaluation
requires the explicit authorization mechanism in ``legacy360_patch_axial``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from legacy360_patch_axial import (  # noqa: E402
    Legacy360PatchAxial,
    MaskedSensorSetResidual,
    SameLocationTemporalResidual,
    _two_class_log_odds,
    classification_metrics,
    load_legacy_model,
    move_batch,
    select_f1_threshold,
    set_seed,
)
from query360_data import (  # noqa: E402
    CLASSIFICATION_PATH_COLUMNS,
    DEFAULT_WV3_SRF,
    SENSOR_ORDER,
    Query360Dataset,
    StrictHashedFileCache,
    collect_classification_paths,
    query360_collate,
    sha256_file,
)


SCHEMA_VERSION = "legacy360-ragged-patch-cache-v1"
HEAD_SCHEMA_VERSION = "legacy360-cached-patch-axial-head-v1"
PATCH_LAYER_NAMES = ("early", "final")


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
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
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
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def deterministic_orthogonal_projection(
    input_dim: int = 768,
    output_dim: int = 64,
    *,
    seed: int = 36064,
) -> torch.Tensor:
    """Return a reproducible Gaussian-QR projection with orthonormal columns."""

    if input_dim < output_dim or output_dim < 1:
        raise ValueError("projection needs input_dim >= output_dim >= 1")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    gaussian = torch.randn(
        input_dim,
        output_dim,
        generator=generator,
        dtype=torch.float64,
    )
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    q = q * signs.unsqueeze(0)
    return q.to(torch.float32).contiguous()


def projection_metadata(
    projection: torch.Tensor, *, seed: int
) -> dict[str, Any]:
    identity = torch.eye(projection.shape[1], dtype=torch.float32)
    error = (
        projection.float().T @ projection.float() - identity
    ).abs().max()
    return {
        "input_dim": int(projection.shape[0]),
        "output_dim": int(projection.shape[1]),
        "seed": int(seed),
        "algorithm": "cpu_float64_gaussian_reduced_qr_sign_canonicalized",
        "sha256": tensor_sha256(projection),
        "max_abs_qtq_minus_i": float(error),
    }


def _channel_ids(
    model: Legacy360PatchAxial,
    ids: torch.Tensor,
    *,
    rows: int,
    repeats: int = 1,
) -> torch.Tensor:
    return model._channel_ids(ids, batch_size=rows, repeats=repeats)


def extract_legacy_concat_features(
    model: Legacy360PatchAxial,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the old concat branch without invoking the new residuals."""

    batch_size = int(batch["index"].shape[0])
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    row_positions = model._row_positions(batch["index"])
    features = torch.zeros(
        batch_size,
        len(model.sensors),
        model.embed_dim,
        device=device,
        dtype=dtype,
    )
    valid = torch.zeros(
        batch_size, len(model.sensors), device=device, dtype=torch.bool
    )
    for sensor_index, sensor in enumerate(model.sensors):
        sensor_batch = batch["sensor_batches"][sensor]
        images = sensor_batch["images"]
        if images.shape[0] == 0:
            continue
        rows = sensor_batch["rows"]
        roles = sensor_batch["roles"]
        local_rows = [
            row_positions[int(value)]
            for value in rows.detach().cpu().tolist()
        ]
        by_sample: dict[int, dict[int, int]] = {}
        for observation, (sample, role) in enumerate(
            zip(local_rows, roles.detach().cpu().tolist())
        ):
            by_sample.setdefault(int(sample), {})[int(role)] = observation
        groups: MutableMapping[tuple[int, ...], list[int]] = defaultdict(list)
        for sample, mapping in by_sample.items():
            if 0 in mapping:
                groups[tuple(sorted(mapping))].append(sample)
        for signature, samples in groups.items():
            concatenated = torch.stack(
                [
                    torch.cat(
                        [
                            images[by_sample[sample][role]]
                            for role in signature
                        ],
                        dim=0,
                    )
                    for sample in samples
                ]
            )
            ids = _channel_ids(
                model,
                sensor_batch["channel_ids"],
                rows=len(samples),
                repeats=len(signature),
            ).to(device)
            cls, _ = model._encode(
                sensor, concatenated, ids, max_blocks=None
            )
            indices = torch.tensor(samples, device=device, dtype=torch.long)
            features[:, sensor_index] = features[:, sensor_index].index_copy(
                0, indices, cls
            )
            valid[indices, sensor_index] = True
    if not valid.any(dim=1).all():
        raise ValueError("a cache row has no valid current sensor")
    sensor_logits = torch.stack(
        [
            _two_class_log_odds(model.heads[sensor](features[:, index]))
            for index, sensor in enumerate(model.sensors)
        ],
        dim=1,
    )
    sensor_logits = torch.where(
        valid, sensor_logits, torch.zeros_like(sensor_logits)
    )
    if model.row_fusion_head is not None:
        if model.base_fusion == "max":
            fused_features = features.masked_fill(
                ~valid.unsqueeze(-1), -torch.inf
            ).max(dim=1).values
        else:
            weight = valid.to(features.dtype).unsqueeze(-1)
            fused_features = (features * weight).sum(dim=1) / weight.sum(
                dim=1
            ).clamp_min(1.0)
        fused_logits = _two_class_log_odds(
            model.row_fusion_head(fused_features)
        )
    elif model.base_fusion == "max":
        fused_logits = sensor_logits.masked_fill(
            ~valid, -torch.inf
        ).max(dim=1).values
    else:
        weight = valid.to(sensor_logits.dtype)
        fused_logits = (sensor_logits * weight).sum(dim=1) / weight.sum(
            dim=1
        ).clamp_min(1.0)
    return features, valid, sensor_logits, fused_logits


def encode_patch_layers_once(
    model: Legacy360PatchAxial,
    sensor: str,
    images: torch.Tensor,
    channel_ids: torch.Tensor,
    *,
    early_blocks: int,
    layers: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Capture early/final patches in one backbone traversal.

    Calling ``_encode(..., max_blocks=early)`` and then ``_encode(..., None)``
    repeats patch embedding and the first ``early`` blocks.  Full extraction is
    I/O-bound *and* encoder-bound, so the cache path explicitly taps the token
    stream once.
    """

    unknown = set(layers) - set(PATCH_LAYER_NAMES)
    if unknown or not layers:
        raise ValueError(f"invalid patch layers: {list(layers)}")
    backbone = model.backbone
    if not hasattr(backbone, "interpolate_pos_encoding"):
        raise TypeError("one-pass layer capture requires the Panopticon ViT")
    patch_embed = (
        model.sensor_patch_embeds[sensor]
        if sensor in model.sensor_patch_embeds
        else backbone.patch_embed
    )
    tokens, height, width = patch_embed(
        {"imgs": images, "chn_ids": channel_ids}
    )
    tokens = torch.cat(
        (backbone.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1
    )
    tokens = tokens + backbone.interpolate_pos_encoding(tokens, width, height)
    if backbone.register_tokens is not None:
        tokens = torch.cat(
            (
                tokens[:, :1],
                backbone.register_tokens.expand(tokens.shape[0], -1, -1),
                tokens[:, 1:],
            ),
            dim=1,
        )
    early_limit = min(max(int(early_blocks), 0), len(backbone.blocks))
    output: dict[str, torch.Tensor] = {}
    if "early" in layers and early_limit == 0:
        output["early"] = tokens[
            :, backbone.num_register_tokens + 1 :
        ]
    need_final = "final" in layers
    stop = len(backbone.blocks) if need_final else early_limit
    for index, block in enumerate(backbone.blocks):
        if index >= stop:
            break
        tokens = block(tokens)
        completed = index + 1
        if "early" in layers and completed == early_limit:
            output["early"] = tokens[
                :, backbone.num_register_tokens + 1 :
            ]
    if need_final:
        normalized = backbone.norm(tokens)
        output["final"] = normalized[
            :, backbone.num_register_tokens + 1 :
        ]
    if set(output) != set(layers):
        raise RuntimeError(
            f"failed to capture requested patch layers: requested={layers}, "
            f"captured={sorted(output)}"
        )
    return output


def extract_patch_shard(
    model: Legacy360PatchAxial,
    batch: Mapping[str, Any],
    projection: torch.Tensor,
    *,
    early_blocks: int,
    layers: Sequence[str],
) -> dict[str, Any]:
    """Extract one complete row batch into a self-contained ragged shard."""

    unknown = set(layers) - set(PATCH_LAYER_NAMES)
    if unknown:
        raise ValueError(f"unknown patch layers: {sorted(unknown)}")
    if not layers:
        raise ValueError("at least one patch layer is required")
    device = next(model.parameters()).device
    projection_device = projection.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        base_features, base_valid, base_sensor_logits, base_fused_logits = (
            extract_legacy_concat_features(model, batch)
        )
        projected_base = base_features.float() @ projection_device

        patch_parts: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in layers
        }
        offsets = [0]
        observation_row: list[int] = []
        observation_sensor: list[int] = []
        observation_role: list[int] = []
        row_positions = model._row_positions(batch["index"])
        per_sensor_patch_counts: dict[str, list[int]] = {
            sensor: [] for sensor in model.sensors
        }
        for sensor_index, sensor in enumerate(model.sensors):
            sensor_batch = batch["sensor_batches"][sensor]
            images = sensor_batch["images"]
            if images.shape[0] == 0:
                continue
            ids = _channel_ids(
                model,
                sensor_batch["channel_ids"],
                rows=int(images.shape[0]),
            ).to(device)
            layer_tokens = encode_patch_layers_once(
                model,
                sensor,
                images,
                ids,
                early_blocks=early_blocks,
                layers=layers,
            )
            reference_count = int(next(iter(layer_tokens.values())).shape[1])
            if any(
                int(tokens.shape[1]) != reference_count
                for tokens in layer_tokens.values()
            ):
                raise RuntimeError("early/final patch counts differ")
            rows = sensor_batch["rows"].detach().cpu().tolist()
            roles = sensor_batch["roles"].detach().cpu().tolist()
            for observation in range(int(images.shape[0])):
                for layer, tokens in layer_tokens.items():
                    projected = (
                        tokens[observation].float() @ projection_device
                    ).to(torch.float16)
                    patch_parts[layer].append(projected.cpu())
                offsets.append(offsets[-1] + reference_count)
                observation_row.append(row_positions[int(rows[observation])])
                observation_sensor.append(sensor_index)
                observation_role.append(int(roles[observation]))
                per_sensor_patch_counts[sensor].append(reference_count)

    shard: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "row_index": batch["index"].detach().cpu().to(torch.int64),
        "labels": batch["labels"].detach().cpu().to(torch.int64),
        "ids": list(batch["ids"]),
        "plume_ids": list(batch["plume_ids"]),
        "availability_signatures": list(batch["availability_signatures"]),
        "base_sensor_features_projected": projected_base.cpu().to(torch.float16),
        "base_sensor_valid": base_valid.cpu(),
        "base_sensor_logits": base_sensor_logits.float().cpu(),
        "base_fused_logits": base_fused_logits.float().cpu(),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "observation_row": torch.tensor(observation_row, dtype=torch.int32),
        "observation_sensor": torch.tensor(
            observation_sensor, dtype=torch.int8
        ),
        "observation_role": torch.tensor(observation_role, dtype=torch.int8),
        "patch_layers": list(layers),
        "patch_count_by_sensor": per_sensor_patch_counts,
    }
    for layer in layers:
        shard[f"patches_{layer}"] = (
            torch.cat(patch_parts[layer], dim=0)
            if patch_parts[layer]
            else torch.empty(0, projection.shape[1], dtype=torch.float16)
        )
    validate_shard(shard)
    return shard


def validate_shard(shard: Mapping[str, Any]) -> None:
    if shard.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported shard schema")
    rows = int(shard["row_index"].numel())
    if shard["labels"].shape != (rows,):
        raise ValueError("labels and row_index differ")
    valid = shard["base_sensor_valid"]
    if valid.shape != (rows, len(SENSOR_ORDER)) or valid.dtype != torch.bool:
        raise ValueError("invalid base_sensor_valid")
    projected = shard["base_sensor_features_projected"]
    if projected.ndim != 3 or projected.shape[:2] != valid.shape:
        raise ValueError("invalid projected base features")
    if shard["base_sensor_logits"].shape != valid.shape:
        raise ValueError("invalid base sensor logits")
    if shard["base_fused_logits"].shape != (rows,):
        raise ValueError("invalid base fused logits")
    observations = int(shard["observation_row"].numel())
    for key in ("observation_sensor", "observation_role"):
        if shard[key].numel() != observations:
            raise ValueError(f"{key} differs from observation count")
    offsets = shard["offsets"]
    if offsets.shape != (observations + 1,) or int(offsets[0]) != 0:
        raise ValueError("invalid ragged offsets")
    if not bool((offsets[1:] >= offsets[:-1]).all()):
        raise ValueError("ragged offsets are not monotonic")
    total_patches = int(offsets[-1])
    for layer in shard["patch_layers"]:
        patches = shard[f"patches_{layer}"]
        if patches.ndim != 2 or int(patches.shape[0]) != total_patches:
            raise ValueError(f"invalid {layer} ragged patches")
        if patches.dtype != torch.float16:
            raise ValueError("cached patches must be float16")
    if observations:
        if int(shard["observation_row"].min()) < 0 or int(
            shard["observation_row"].max()
        ) >= rows:
            raise ValueError("observation row is out of range")
        if int(shard["observation_sensor"].min()) < 0 or int(
            shard["observation_sensor"].max()
        ) >= len(SENSOR_ORDER):
            raise ValueError("observation sensor is out of range")
        if int(shard["observation_role"].min()) < 0 or int(
            shard["observation_role"].max()
        ) >= 3:
            raise ValueError("observation role is out of range")


class CachedPatchHeadOutput:
    def __init__(
        self,
        *,
        fused_logits: torch.Tensor,
        base_fused_logits: torch.Tensor,
        sensor_logits: torch.Tensor,
        base_sensor_logits: torch.Tensor,
        sensor_valid: torch.Tensor,
        temporal_residual_norm: torch.Tensor,
        sensor_attention: torch.Tensor,
    ) -> None:
        self.fused_logits = fused_logits
        self.base_fused_logits = base_fused_logits
        self.sensor_logits = sensor_logits
        self.base_sensor_logits = base_sensor_logits
        self.sensor_valid = sensor_valid
        self.temporal_residual_norm = temporal_residual_norm
        self.sensor_attention = sensor_attention


class CachedPatchAxialHead(nn.Module):
    """Small local-cache-only time x sensor residual head."""

    def __init__(
        self,
        projection_dim: int = 64,
        *,
        sensors: Sequence[str] = SENSOR_ORDER,
        num_roles: int = 3,
        temporal_heads: int = 4,
        sensor_heads: int = 4,
        topk_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        self.projection_dim = int(projection_dim)
        self.sensors = tuple(sensors)
        self.num_roles = int(num_roles)
        self.temporal = nn.ModuleDict(
            {
                sensor: SameLocationTemporalResidual(
                    projection_dim,
                    num_heads=temporal_heads,
                    num_roles=num_roles,
                    topk_fraction=topk_fraction,
                )
                for sensor in self.sensors
            }
        )
        self.sensor_fusion = MaskedSensorSetResidual(
            projection_dim,
            num_sensors=len(self.sensors),
            num_heads=sensor_heads,
        )
        self.sensor_delta_heads = nn.ModuleList(
            [nn.Linear(projection_dim, 1) for _ in self.sensors]
        )
        for head in self.sensor_delta_heads:
            # A random readout of an exactly-zero temporal vector still emits
            # an exact zero, while allowing the temporal projection to receive
            # an auxiliary-loss gradient on the very first optimizer step.
            nn.init.normal_(head.weight, std=projection_dim**-0.5)
            nn.init.zeros_(head.bias)

    @property
    def residual_is_zero(self) -> bool:
        parameters = [
            self.sensor_fusion.output.weight,
            self.sensor_fusion.output.bias,
        ]
        for module in self.temporal.values():
            parameters.extend(
                (
                    module.residual_projection.weight,
                    module.residual_projection.bias,
                )
            )
        for head in self.sensor_delta_heads:
            parameters.append(head.bias)
        return all(
            bool(torch.count_nonzero(parameter).item() == 0)
            for parameter in parameters
        )

    @staticmethod
    def _observation_map(
        shard: Mapping[str, Any],
    ) -> dict[tuple[int, int], dict[int, int]]:
        output: dict[tuple[int, int], dict[int, int]] = {}
        rows = shard["observation_row"].detach().cpu().tolist()
        sensors = shard["observation_sensor"].detach().cpu().tolist()
        roles = shard["observation_role"].detach().cpu().tolist()
        for observation, (row, sensor, role) in enumerate(
            zip(rows, sensors, roles)
        ):
            output.setdefault((int(row), int(sensor)), {})[
                int(role)
            ] = observation
        return output

    def forward(
        self,
        shard: Mapping[str, Any],
        *,
        patch_layer: str,
    ) -> CachedPatchHeadOutput:
        validate_shard(shard)
        if patch_layer not in shard["patch_layers"]:
            raise ValueError(
                f"patch layer {patch_layer!r} not present in shard"
            )
        patches = shard[f"patches_{patch_layer}"]
        if int(patches.shape[1]) != self.projection_dim:
            raise ValueError("projection dimension mismatch")
        base_features = shard["base_sensor_features_projected"].float()
        sensor_valid = shard["base_sensor_valid"]
        batch_size = int(base_features.shape[0])
        temporal_delta = torch.zeros_like(base_features)
        mapping = self._observation_map(shard)
        offsets = shard["offsets"].detach().cpu().tolist()

        # Group by sensor and P so every temporal call is dense while the cache
        # itself stays ragged across sensors/rows.
        groups: MutableMapping[
            tuple[int, int], list[tuple[int, dict[int, int]]]
        ] = defaultdict(list)
        for (row, sensor), role_map in mapping.items():
            if 0 not in role_map:
                continue
            current = role_map[0]
            patch_count = int(offsets[current + 1] - offsets[current])
            if patch_count < 1:
                raise ValueError("an observation has no patches")
            for observation in role_map.values():
                count = int(
                    offsets[observation + 1] - offsets[observation]
                )
                if count != patch_count:
                    raise ValueError(
                        "same row/sensor roles have unequal patch counts"
                    )
            groups[(sensor, patch_count)].append((row, role_map))

        residual_norm = torch.zeros(
            batch_size,
            len(self.sensors),
            device=base_features.device,
            dtype=torch.float32,
        )
        for (sensor, patch_count), entries in groups.items():
            dense = patches.new_zeros(
                len(entries),
                self.num_roles,
                patch_count,
                self.projection_dim,
            )
            role_valid = torch.zeros(
                len(entries),
                self.num_roles,
                device=patches.device,
                dtype=torch.bool,
            )
            row_indices = []
            for group_row, (row, role_map) in enumerate(entries):
                row_indices.append(row)
                for role, observation in role_map.items():
                    if role >= self.num_roles:
                        continue
                    start, end = offsets[observation : observation + 2]
                    dense[group_row, role] = patches[start:end]
                    role_valid[group_row, role] = True
            residual, _ = self.temporal[self.sensors[sensor]](
                dense.float(), role_valid
            )
            rows_tensor = torch.tensor(
                row_indices, device=patches.device, dtype=torch.long
            )
            temporal_delta[:, sensor] = temporal_delta[:, sensor].index_copy(
                0, rows_tensor, residual.to(temporal_delta.dtype)
            )
            residual_norm[:, sensor] = residual_norm[:, sensor].index_copy(
                0, rows_tensor, residual.float().norm(dim=-1)
            )
        evidence = base_features + temporal_delta
        sensor_delta = torch.stack(
            [
                head(temporal_delta[:, sensor]).squeeze(-1)
                for sensor, head in enumerate(self.sensor_delta_heads)
            ],
            dim=1,
        )
        sensor_logits = shard["base_sensor_logits"].float() + sensor_delta
        sensor_logits = torch.where(
            sensor_valid, sensor_logits, torch.zeros_like(sensor_logits)
        )
        fused_delta, sensor_attention = self.sensor_fusion(
            evidence, sensor_valid
        )
        base_fused = shard["base_fused_logits"].float()
        return CachedPatchHeadOutput(
            fused_logits=base_fused + fused_delta,
            base_fused_logits=base_fused,
            sensor_logits=sensor_logits,
            base_sensor_logits=shard["base_sensor_logits"].float(),
            sensor_valid=sensor_valid,
            temporal_residual_norm=residual_norm,
            sensor_attention=sensor_attention,
        )


class PatchShardDataset(Dataset):
    """Index local cache shards; no remote source path is ever consulted."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / "cache_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{manifest_path}: unsupported schema")
        self.shards = [
            self.cache_dir / item["file"] for item in self.manifest["shards"]
        ]
        if not self.shards:
            raise ValueError(f"{cache_dir}: no shards")
        missing = [str(path) for path in self.shards if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing cache shards: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        shard = torch.load(
            self.shards[int(index)],
            map_location="cpu",
            weights_only=False,
        )
        validate_shard(shard)
        return shard


def _warm_cache(
    cache: StrictHashedFileCache,
    manifest: str,
    *,
    workers: int,
) -> dict[str, Any]:
    frame = pd.read_csv(
        manifest,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        usecols=lambda name: name in CLASSIFICATION_PATH_COLUMNS,
    )
    paths = collect_classification_paths(frame)
    return asdict(cache.warm_up(paths, max_workers=workers))


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        existing = list(path.iterdir())
        if existing:
            raise FileExistsError(
                f"refusing non-empty cache output directory: {path}"
            )
    path.mkdir(parents=True, exist_ok=True)
    (path / "shards").mkdir(exist_ok=True)


def command_extract(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    _prepare_output_directory(output_dir)
    status_path = output_dir / "run_status.json"
    atomic_json(
        status_path,
        {
            "status": "running",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    )
    try:
        device = torch.device(args.device)
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = True
        local_cache = (
            StrictHashedFileCache(args.raw_cache_dir)
            if args.raw_cache_dir
            else None
        )
        cache_warmup = None
        if local_cache is not None and args.warm_cache:
            cache_warmup = _warm_cache(
                local_cache,
                args.manifest,
                workers=args.cache_workers,
            )
            atomic_json(output_dir / "raw_cache_report.json", cache_warmup)
        dataset = Query360Dataset(
            args.manifest,
            local_cache=local_cache,
            wv3_srf_csv=args.wv3_srf_csv,
            pad_to_multiple=14,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            collate_fn=query360_collate,
        )
        model, checkpoint_provenance = load_legacy_model(
            args.checkpoint,
            temporal_frame_blocks=args.early_blocks,
        )
        model.to(device).eval()
        projection = deterministic_orthogonal_projection(
            model.embed_dim,
            args.projection_dim,
            seed=args.projection_seed,
        )
        projection_info = projection_metadata(
            projection, seed=args.projection_seed
        )
        atomic_torch(
            output_dir / "projection.pth",
            {
                "projection": projection,
                "metadata": projection_info,
            },
        )
        layers = tuple(
            PATCH_LAYER_NAMES
            if args.patch_layers == "both"
            else (args.patch_layers,)
        )
        shard_records: list[dict[str, Any]] = []
        processed_rows = processed_observations = processed_patches = 0
        sensor_patch_counts: dict[str, list[int]] = {
            sensor: [] for sensor in SENSOR_ORDER
        }
        started = time.monotonic()
        for shard_index, raw_batch in enumerate(loader):
            if args.max_batches is not None and shard_index >= args.max_batches:
                break
            batch = move_batch(raw_batch, device)
            shard = extract_patch_shard(
                model,
                batch,
                projection,
                early_blocks=args.early_blocks,
                layers=layers,
            )
            shard_path = output_dir / "shards" / f"shard_{shard_index:06d}.pth"
            atomic_torch(shard_path, shard)
            size = int(shard_path.stat().st_size)
            rows = int(shard["row_index"].numel())
            observations = int(shard["observation_row"].numel())
            patches = int(shard["offsets"][-1])
            processed_rows += rows
            processed_observations += observations
            processed_patches += patches
            for sensor, values in shard["patch_count_by_sensor"].items():
                sensor_patch_counts[sensor].extend(int(value) for value in values)
            shard_records.append(
                {
                    "file": str(shard_path.relative_to(output_dir)),
                    "rows": rows,
                    "observations": observations,
                    "patches_per_layer": patches,
                    "bytes": size,
                }
            )
            print(
                f"[patch-cache] shard={shard_index} rows={rows} "
                f"observations={observations} patches/layer={patches} "
                f"size_MiB={size / 2**20:.3f}",
                flush=True,
            )
        if not processed_rows:
            raise RuntimeError("no cache rows were extracted")
        actual_shard_bytes = sum(item["bytes"] for item in shard_records)
        total_rows = len(dataset)
        scale = total_rows / processed_rows
        patch_bytes_per_layer = (
            processed_patches * args.projection_dim * 2
        )
        estimate = {
            "sampled_rows": processed_rows,
            "source_total_rows": total_rows,
            "sampled_observations": processed_observations,
            "sampled_patches_per_layer": processed_patches,
            "sampled_shard_bytes": actual_shard_bytes,
            "sampled_patch_payload_bytes_per_layer": patch_bytes_per_layer,
            "estimated_full_shard_bytes": int(round(actual_shard_bytes * scale)),
            "estimated_full_patch_payload_bytes_per_layer": int(
                round(patch_bytes_per_layer * scale)
            ),
            "estimated_full_patch_payload_bytes_all_layers": int(
                round(patch_bytes_per_layer * len(layers) * scale)
            ),
            "method": (
                "linear extrapolation from deterministic leading shards; "
                "rerun estimate on stratified/full extraction before quota allocation"
            ),
        }
        patch_distribution = {}
        for sensor, values in sensor_patch_counts.items():
            if values:
                array = np.asarray(values, dtype=np.int64)
                patch_distribution[sensor] = {
                    "observations": int(array.size),
                    "min": int(array.min()),
                    "median": float(np.median(array)),
                    "mean": float(array.mean()),
                    "max": int(array.max()),
                }
            else:
                patch_distribution[sensor] = {"observations": 0}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "partial": processed_rows != total_rows,
            "source": {
                "manifest": str(Path(args.manifest).expanduser().resolve()),
                "manifest_sha256": sha256_file(args.manifest),
                "split": args.split,
                "rows": total_rows,
            },
            "checkpoint": checkpoint_provenance,
            "projection": projection_info,
            "patch_layers": list(layers),
            "early_blocks": int(args.early_blocks),
            "storage_dtype": "float16",
            "shards": shard_records,
            "totals": {
                "processed_rows": processed_rows,
                "observations": processed_observations,
                "patches_per_layer": processed_patches,
                "shard_bytes": actual_shard_bytes,
                "elapsed_seconds": time.monotonic() - started,
            },
            "patch_distribution": patch_distribution,
            "storage_estimate": estimate,
            "raw_cache_report": cache_warmup,
        }
        atomic_json(output_dir / "cache_manifest.json", manifest)
        atomic_json(
            status_path,
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "processed_rows": processed_rows,
                "partial": processed_rows != total_rows,
            },
        )
        print(json.dumps(manifest, indent=2), flush=True)
    except Exception as exc:
        atomic_json(
            status_path,
            {
                "status": "failed",
                "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def _cache_compatibility(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> None:
    for key in ("projection", "checkpoint", "patch_layers", "early_blocks"):
        if left[key] != right[key]:
            raise ValueError(f"train/dev cache mismatch in {key}")
    if left.get("partial") or right.get("partial"):
        raise ValueError("train-head refuses partial smoke caches")


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


def evaluate_head(
    model: CachedPatchAxialHead,
    loader: DataLoader,
    *,
    device: torch.device,
    patch_layer: str,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for raw_shard in loader:
            shard = move_batch(raw_shard, device)
            with _autocast(device, amp):
                output = model(shard, patch_layer=patch_layer)
            targets.append(shard["labels"].detach().cpu())
            probabilities.append(
                torch.sigmoid(output.fused_logits.float()).detach().cpu()
            )
    return torch.cat(targets).numpy(), torch.cat(probabilities).numpy()


def command_train_head(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    train_dataset = PatchShardDataset(args.train_cache)
    dev_dataset = PatchShardDataset(args.dev_cache)
    _cache_compatibility(train_dataset.manifest, dev_dataset.manifest)
    if args.patch_layer not in train_dataset.manifest["patch_layers"]:
        raise ValueError("requested patch layer is absent")
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = CachedPatchAxialHead(
        projection_dim=int(train_dataset.manifest["projection"]["output_dim"]),
        topk_fraction=args.topk_fraction,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_labels, base_probabilities = evaluate_head(
        model,
        dev_loader,
        device=device,
        patch_layer=args.patch_layer,
        amp=args.amp,
    )
    base_threshold, base_metrics = select_f1_threshold(
        base_labels, base_probabilities
    )
    if not model.residual_is_zero:
        raise AssertionError("epoch-zero cache head must preserve the base")
    history: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_main = total_aux = 0.0
        seen = 0
        for step, raw_shard in enumerate(train_loader, start=1):
            shard = move_batch(raw_shard, device)
            target = shard["labels"].float()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, args.amp):
                output = model(shard, patch_layer=args.patch_layer)
                main = F.binary_cross_entropy_with_logits(
                    output.fused_logits, target
                )
                expanded = target[:, None].expand_as(output.sensor_logits)
                per_sensor = F.binary_cross_entropy_with_logits(
                    output.sensor_logits, expanded, reduction="none"
                )
                mask = output.sensor_valid.to(per_sensor.dtype)
                auxiliary = (per_sensor * mask).sum() / mask.sum().clamp_min(1)
                loss = main + args.sensor_aux_weight * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            rows = int(target.numel())
            total_loss += float(loss.detach()) * rows
            total_main += float(main.detach()) * rows
            total_aux += float(auxiliary.detach()) * rows
            seen += rows
            if args.max_train_shards and step >= args.max_train_shards:
                break
        labels, probabilities = evaluate_head(
            model,
            dev_loader,
            device=device,
            patch_layer=args.patch_layer,
            amp=args.amp,
        )
        threshold, metrics = select_f1_threshold(labels, probabilities)
        record = {
            "epoch": epoch,
            "train": {
                "loss": total_loss / seen,
                "main_loss": total_main / seen,
                "sensor_aux_loss": total_aux / seen,
                "rows": seen,
            },
            "development": metrics,
        }
        history.append(record)
        atomic_json(output_dir / "metrics_history.json", history)
        if best is None or metrics["binary_f1"] > best["development"]["binary_f1"]:
            best = copy.deepcopy(record)
            atomic_torch(
                output_dir / "checkpoint_best.pth",
                {
                    "schema_version": HEAD_SCHEMA_VERSION,
                    "epoch": epoch,
                    "threshold": threshold,
                    "model": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "development": metrics,
                    "cache": {
                        "train": str(Path(args.train_cache).resolve()),
                        "development": str(Path(args.dev_cache).resolve()),
                        "projection": train_dataset.manifest["projection"],
                        "checkpoint": train_dataset.manifest["checkpoint"],
                    },
                },
            )
        print(
            f"[cached-head] epoch={epoch}/{args.epochs} "
            f"loss={record['train']['loss']:.5f} "
            f"dev_F1={metrics['binary_f1']:.5f} "
            f"dev_AP={metrics['ap']:.5f}",
            flush=True,
        )
    summary = {
        "schema_version": HEAD_SCHEMA_VERSION,
        "protocol": "local_patch_cache_head_development_only",
        "patch_layer": args.patch_layer,
        "epoch_zero": {
            "threshold": base_threshold,
            "development": base_metrics,
            "exact_cached_base_by_zero_residual": True,
        },
        "best": best,
    }
    atomic_json(output_dir / "summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--split", choices=("train", "development"), required=True)
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--output_dir", required=True)
    extract.add_argument("--raw_cache_dir", default="")
    extract.add_argument(
        "--warm_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    extract.add_argument("--cache_workers", type=int, default=32)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--batch_size", type=int, default=16)
    extract.add_argument("--workers", type=int, default=8)
    extract.add_argument("--seed", type=int, default=41)
    extract.add_argument("--projection_seed", type=int, default=36064)
    extract.add_argument("--projection_dim", type=int, default=64)
    extract.add_argument("--early_blocks", type=int, default=2)
    extract.add_argument(
        "--patch_layers",
        choices=("early", "final", "both"),
        default="both",
    )
    extract.add_argument("--max_batches", type=int, default=None)
    extract.add_argument("--wv3_srf_csv", default=str(DEFAULT_WV3_SRF))
    extract.set_defaults(function=command_extract)

    train_head = subparsers.add_parser("train-head")
    train_head.add_argument("--train_cache", required=True)
    train_head.add_argument("--dev_cache", required=True)
    train_head.add_argument("--output_dir", required=True)
    train_head.add_argument("--patch_layer", choices=PATCH_LAYER_NAMES, default="early")
    train_head.add_argument("--device", default="cuda:0")
    train_head.add_argument("--epochs", type=int, default=4)
    train_head.add_argument("--workers", type=int, default=4)
    train_head.add_argument("--seed", type=int, default=41)
    train_head.add_argument("--learning_rate", type=float, default=3e-4)
    train_head.add_argument("--weight_decay", type=float, default=1e-4)
    train_head.add_argument("--sensor_aux_weight", type=float, default=0.2)
    train_head.add_argument("--topk_fraction", type=float, default=0.25)
    train_head.add_argument("--grad_clip", type=float, default=1.0)
    train_head.add_argument("--max_train_shards", type=int, default=None)
    train_head.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_head.set_defaults(function=command_train_head)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.function(arguments)
