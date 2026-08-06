#!/usr/bin/env python3
"""Full legacy-360 m time-by-sensor training from existing Panopticon weights.

This runner exists for an apples-to-apples comparison with the previous
MethaneFuse 360 m protocol.  It does not use the representative-row
macro-region screen.  Instead it:

1. consumes the complete legacy ``train_core``/``dev``/sealed-``test``
   manifests with event-disjoint inner development selection;
2. makes a durable float16 Panopticon feature cache on ``/diniuvol`` in one
   source-data pass, after which training performs no remote image I/O; and
3. selects the epoch and decision threshold on ``dev`` and only then performs
   one explicitly authorized, locked-threshold sealed-test evaluation.

The encoder loader accepts the official Panopticon state, a previous
single-sensor checkpoint containing ``backbone``, a previous universal
checkpoint containing ``model.backbone.*`` and
``model.sensor_patch_embeds.*``, or an explicit per-sensor checkpoint map.
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
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from query360_data import (  # noqa: E402
    CLASSIFICATION_PATH_COLUMNS,
    DEFAULT_WV3_SRF,
    REQUIRED_SPLIT_COLUMNS,
    SENSOR_ORDER,
    Query360Dataset,
    StrictHashedFileCache,
    _pad_image,
    query360_collate,
)
from query360_model import (  # noqa: E402
    fixed_epoch_batches,
    model_parameter_signature,
    set_deterministic_seed,
    state_dict_sha256,
    transient_query_loss,
)
from query360_two_axis_model import TwoAxisQuery360Head  # noqa: E402
from src.backbones import build_panopticon_vitb14  # noqa: E402
from src.data.sensor_transforms import (  # noqa: E402
    L89_PRECOMPUTED_STATS,
    S2_PRECOMPUTED_STATS,
)


SCRIPT_VERSION = "query360-two-axis-full-legacy-v2"
DEFAULT_TRAIN = (
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/"
    "finalDataset_query/legacy_param_360m/manifest_time_train.csv"
)
DEFAULT_EVAL = (
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/"
    "finalDataset_query/legacy_param_360m/manifest_time_test.csv"
)
DEFAULT_WEIGHTS = str(REPO_ROOT / "weights" / "panopticon_vitb14_teacher.pth")
LEGACY_S2_MEAN_UNIT = torch.tensor(
    [value / 65535.0 for value in S2_PRECOMPUTED_STATS[0]],
    dtype=torch.float32,
).view(-1, 1, 1)
LEGACY_S2_STD_UNIT = torch.tensor(
    [value / 65535.0 for value in S2_PRECOMPUTED_STATS[1]],
    dtype=torch.float32,
).clamp_min(1e-6).view(-1, 1, 1)
LEGACY_L89_MEAN_UNIT = torch.tensor(
    [value / 65535.0 for value in L89_PRECOMPUTED_STATS[0]],
    dtype=torch.float32,
).view(-1, 1, 1)
LEGACY_L89_STD_UNIT = torch.tensor(
    [value / 65535.0 for value in L89_PRECOMPUTED_STATS[1]],
    dtype=torch.float32,
).clamp_min(1e-6).view(-1, 1, 1)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def configure_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def prepare_manifest(source: Path, destination: Path, *, split: str) -> dict[str, Any]:
    frame = pd.read_csv(source, dtype=str, keep_default_na=False, low_memory=False)
    missing = [
        column
        for column in ("id", "plume_id", "label", *CLASSIFICATION_PATH_COLUMNS)
        if column not in frame
    ]
    if missing:
        raise ValueError(f"{source} misses columns: {missing}")
    labels = pd.to_numeric(frame["label"], errors="raise").astype(np.int64)
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{source}: labels are not binary")
    frame["label"] = labels
    frame["query360_index"] = np.arange(len(frame), dtype=np.int64)
    # These are metadata-only compatibility fields.  The legacy protocol is
    # already plume-disjoint and does not claim macro-region generalization.
    frame["cluster_id"] = frame["plume_id"].astype(str)
    frame["macro_region_id"] = frame["plume_id"].astype(str)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return {
        "split": split,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "prepared": str(destination),
        "prepared_sha256": sha256_file(destination),
        "rows": int(len(frame)),
        "plumes": int(frame["plume_id"].nunique()),
        "labels": {
            str(key): int(value)
            for key, value in frame["label"].value_counts().sort_index().items()
        },
    }


class LegacyAllRolesDataset(Query360Dataset):
    """Expose strict roles and a separate checkpoint-reproduction role path.

    The earlier single-sensor checkpoints were trained on concatenated
    three-role inputs and did not remove duplicate historical acquisitions.
    Their input loader also retained readable roles below the strict finite
    fraction gate.  ``observations`` therefore remains strict for the new
    temporal branch, while ``declared_observations`` preserves all three
    readable declared roles for the historical concatenated-channel branch.
    """

    preserve_declared_roles = True

    @staticmethod
    def _mask_duplicate_roles(loaded):
        t0_valid = bool(loaded[0][2])
        return [
            (image, fraction, bool(valid) and t0_valid)
            for image, fraction, valid, _finite_raw, _finite_mask in loaded
        ]

    def _declared_role_image(
        self,
        image: torch.Tensor,
        *,
        sensor: str,
        role: int,
        finite_raw: torch.Tensor,
        finite_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reproduce the old loader's readable-but-invalid role values."""

        if sensor == "s5p":
            # Historical S5P loading replaced NaN/Inf raw values with zero
            # *before* per-role normalization.  The strict path instead maps
            # invalid normalized pixels to neutral zero.
            legacy = (
                finite_raw - self._s5p_mean[role]
            ) / self._s5p_std[role]
        elif sensor == "s2":
            legacy = (
                finite_raw / 65535.0 - LEGACY_S2_MEAN_UNIT
            ) / LEGACY_S2_STD_UNIT
            legacy = torch.where(finite_mask, legacy, torch.zeros_like(legacy))
        elif sensor == "l89":
            legacy = (
                finite_raw / 65535.0 - LEGACY_L89_MEAN_UNIT
            ) / LEGACY_L89_STD_UNIT
            legacy = torch.where(finite_mask, legacy, torch.zeros_like(legacy))
        elif sensor == "emit":
            # The old WV3/EMIT path skipped its DN-like auto-scaling whenever
            # any NaN/Inf made the global absolute maximum non-finite.
            legacy = finite_raw
            if bool(finite_mask.all()):
                abs_max = torch.amax(torch.abs(legacy))
                if torch.isfinite(abs_max) and float(abs_max.item()) > 100.0:
                    legacy = legacy / 65535.0
            legacy = torch.where(finite_mask, legacy, torch.zeros_like(legacy))
        else:  # pragma: no cover - protected by the fixed sensor order
            raise AssertionError(sensor)
        legacy = torch.nan_to_num(
            legacy, nan=0.0, posinf=0.0, neginf=0.0
        )
        if sensor != "s5p":
            legacy = legacy.clamp(-50.0, 50.0)
        legacy = _pad_image(legacy, self.pad_to_multiple)
        if legacy.shape != image.shape:
            raise RuntimeError(
                "legacy and strict finalized role shapes disagree: "
                f"{tuple(legacy.shape)} != {tuple(image.shape)}"
            )
        return legacy


class ReadOnlyFallbackHashedFileCache(StrictHashedFileCache):
    """Reuse completed warm-cache entries without ever adding new ones.

    ``StrictHashedFileCache`` writes every cache miss, which is inappropriate
    for the full legacy corpus because it can exceed one terabyte.  The
    interrupted warm-up left useful, atomically completed files behind.  This
    adapter returns those files when present and otherwise streams directly
    from the source path.  It never creates, copies, replaces, or deletes a
    cache entry.
    """

    def require_cached(self, source: os.PathLike[str] | str) -> str:
        destination = self.cached_path(source)
        try:
            if destination.is_file():
                size = int(destination.stat().st_size)
                suffix = destination.suffix.casefold()
                if suffix == ".npz" and size >= 128:
                    with destination.open("rb") as stream:
                        if stream.read(4) in (
                            b"PK\x03\x04",
                            b"PK\x05\x06",
                            b"PK\x07\x08",
                        ):
                            return str(destination)
                elif suffix in {".tif", ".tiff"} and size >= 1024:
                    # Metadata-only validation catches atomically complete but
                    # structurally empty/corrupt cached TIFFs before the
                    # dataset selects them over a healthy source payload.
                    with tifffile.TiffFile(destination) as payload:
                        if len(payload.pages) > 0:
                            return str(destination)
        except (OSError, tifffile.TiffFileError):
            pass
        return str(Path(source).expanduser().absolute())


def command_prepare(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).expanduser().absolute()
    train_alias = output / "full_train.csv"
    eval_alias = output / "full_eval.csv"
    train_info = prepare_manifest(
        Path(args.train_csv), train_alias, split="train"
    )
    eval_info = prepare_manifest(
        Path(args.eval_csv), eval_alias, split="evaluation"
    )
    train_plumes = set(
        pd.read_csv(train_alias, usecols=["plume_id"])["plume_id"].astype(str)
    )
    eval_plumes = set(
        pd.read_csv(eval_alias, usecols=["plume_id"])["plume_id"].astype(str)
    )
    overlap = train_plumes & eval_plumes
    if overlap:
        raise RuntimeError(f"legacy protocol plume overlap: {sorted(overlap)[:20]}")
    payload = {
        "schema_version": "query360-full-legacy-manifests-v1",
        "script_version": SCRIPT_VERSION,
        "train": train_info,
        "evaluation": eval_info,
        "plume_overlap": 0,
        "protocol": (
            "complete legacy 360m plume-disjoint split; no representative-row "
            "or macro-region subsampling"
        ),
    }
    atomic_json(output / "protocol.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def _strip_prefix(
    state: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {
        str(key)[len(prefix) :]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }


def _checkpoint_state(path: Path) -> tuple[Mapping[str, torch.Tensor], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: checkpoint is not a mapping")
    if "model" in payload and isinstance(payload["model"], Mapping):
        return payload["model"], "universal_model"
    if "backbone" in payload and isinstance(payload["backbone"], Mapping):
        return payload["backbone"], "single_sensor_backbone"
    if all(torch.is_tensor(value) for value in payload.values()):
        return payload, "raw_backbone"
    raise TypeError(f"{path}: no supported model/backbone state")


def _load_one_backbone(
    path: Path, *, sensor: Optional[str] = None
) -> tuple[nn.Module, dict[str, Any]]:
    state, kind = _checkpoint_state(path)
    backbone = build_panopticon_vitb14()
    if kind == "universal_model":
        backbone_state = _strip_prefix(state, "backbone.")
        if not backbone_state:
            raise ValueError(f"{path}: empty model.backbone state")
        incompatible = backbone.load_state_dict(backbone_state, strict=False)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("patch_embed.")
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{path}: incompatible universal backbone; missing={missing[:10]}, "
                f"unexpected={incompatible.unexpected_keys[:10]}"
            )
        if sensor is not None:
            source_sensor = "wv3" if sensor == "emit" else sensor
            patch_state = _strip_prefix(
                state, f"sensor_patch_embeds.{source_sensor}."
            )
            if not patch_state:
                raise ValueError(f"{path}: no patch state for {source_sensor}")
            backbone.patch_embed.load_state_dict(patch_state, strict=True)
    else:
        backbone.load_state_dict(state, strict=True)
    return backbone, {
        "path": str(path),
        "sha256": sha256_file(path),
        "checkpoint_kind": kind,
    }


class BinaryCLSHead(nn.Module):
    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(features))


def _load_checkpoint_heads(
    path: Path, *, sensor: Optional[str]
) -> tuple[dict[str, Mapping[str, torch.Tensor]], Optional[Mapping[str, torch.Tensor]], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        return {}, None, "none"
    if "model" in payload and isinstance(payload["model"], Mapping):
        state = payload["model"]
        heads: dict[str, Mapping[str, torch.Tensor]] = {}
        for target_sensor, source_sensor in {
            "s2": "s2",
            "l89": "l89",
            "emit": "wv3",
            "s5p": "s5p",
        }.items():
            head_state = _strip_prefix(state, f"heads.{source_sensor}.")
            if head_state:
                heads[target_sensor] = head_state
        fusion = _strip_prefix(state, "row_fusion_head.")
        args = payload.get("args", {})
        fusion_mode = (
            str(args.get("row_fusion_mode", "max"))
            if isinstance(args, Mapping)
            else "max"
        )
        return heads, fusion or None, fusion_mode
    if (
        sensor is not None
        and "head" in payload
        and isinstance(payload["head"], Mapping)
    ):
        return {sensor: payload["head"]}, None, "sensor_logit_mean"
    return {}, None, "none"


class SensorEncoderBank(nn.Module):
    """One shared or one-per-sensor Panopticon encoder for feature caching."""

    def __init__(
        self,
        default_weights: Path,
        sensor_weights: Mapping[str, Path],
    ) -> None:
        super().__init__()
        self.sensor_specific = bool(sensor_weights)
        self.s2_override = "s2" in sensor_weights
        self.provenance: dict[str, Any] = {}
        self.heads = nn.ModuleDict()
        self.fusion_head: Optional[BinaryCLSHead] = None
        self.fusion_mode = "sensor_logit_mean"
        self.universal_s2: Optional[nn.Module] = None
        self.universal_s2_head: Optional[BinaryCLSHead] = None
        if self.sensor_specific:
            modules = {}
            for sensor in SENSOR_ORDER:
                path = sensor_weights.get(sensor, default_weights)
                modules[sensor], self.provenance[sensor] = _load_one_backbone(
                    path, sensor=sensor
                )
                head_states, _fusion, _mode = _load_checkpoint_heads(
                    path, sensor=sensor
                )
                if sensor in head_states:
                    head = BinaryCLSHead()
                    head.load_state_dict(head_states[sensor], strict=True)
                    self.heads[sensor] = head
            self.encoders = nn.ModuleDict(modules)
            self.shared = None
            self.sensor_patch_embeds = None
            _heads, fusion_state, fusion_mode = _load_checkpoint_heads(
                default_weights, sensor=None
            )
            if fusion_state is not None:
                self.fusion_head = BinaryCLSHead()
                self.fusion_head.load_state_dict(fusion_state, strict=True)
                self.fusion_mode = fusion_mode
            # A sensor override must not silently destroy the exact
            # universal checkpoint baseline.  When S2 is overridden, retain
            # one additional universal S2 encoder so the same in-memory image
            # batch produces (A) exact universal row-fusion logits and
            # (B) verified S2-head hybrid logits without a second data read.
            if self.s2_override:
                self.universal_s2, universal_s2_provenance = _load_one_backbone(
                    default_weights, sensor="s2"
                )
                self.provenance["universal_s2_base"] = universal_s2_provenance
                default_heads, _unused_fusion, _unused_mode = (
                    _load_checkpoint_heads(default_weights, sensor=None)
                )
                if "s2" not in default_heads:
                    raise ValueError(
                        "universal checkpoint has no S2 classifier head"
                    )
                self.universal_s2_head = BinaryCLSHead()
                self.universal_s2_head.load_state_dict(
                    default_heads["s2"], strict=True
                )
        else:
            shared, provenance = _load_one_backbone(default_weights)
            self.shared = shared
            self.encoders = nn.ModuleDict()
            self.provenance["shared"] = provenance
            self.sensor_patch_embeds: Optional[nn.ModuleDict] = None
            state, kind = _checkpoint_state(default_weights)
            head_states, fusion_state, fusion_mode = _load_checkpoint_heads(
                default_weights, sensor=None
            )
            for sensor, head_state in head_states.items():
                head = BinaryCLSHead()
                head.load_state_dict(head_state, strict=True)
                self.heads[sensor] = head
            if fusion_state is not None:
                self.fusion_head = BinaryCLSHead()
                self.fusion_head.load_state_dict(fusion_state, strict=True)
                self.fusion_mode = fusion_mode
            if kind == "universal_model":
                patch_modules = {}
                source_names = {
                    "s2": "s2",
                    "l89": "l89",
                    "emit": "wv3",
                    "s5p": "s5p",
                }
                for sensor, source_sensor in source_names.items():
                    patch = copy.deepcopy(shared.patch_embed)
                    patch_state = _strip_prefix(
                        state, f"sensor_patch_embeds.{source_sensor}."
                    )
                    if not patch_state:
                        raise ValueError(
                            f"{default_weights}: no patch state for {source_sensor}"
                        )
                    patch.load_state_dict(patch_state, strict=True)
                    patch_modules[sensor] = patch
                self.sensor_patch_embeds = nn.ModuleDict(patch_modules)

    @property
    def embed_dim(self) -> int:
        if self.sensor_specific:
            return int(self.encoders[SENSOR_ORDER[0]].embed_dim)
        assert self.shared is not None
        return int(self.shared.embed_dim)

    def forward_sensor(
        self, sensor: str, images: torch.Tensor, channel_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.sensor_specific:
            backbone = self.encoders[sensor]
        else:
            assert self.shared is not None
            backbone = self.shared
            if self.sensor_patch_embeds is not None:
                backbone.patch_embed = self.sensor_patch_embeds[sensor]
        output = backbone.forward_features({"imgs": images, "chn_ids": channel_ids})
        return output["x_norm_clstoken"]

    def forward_concat(
        self, sensor: str, images: torch.Tensor, channel_ids: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self.forward_sensor(sensor, images, channel_ids)
        if sensor not in self.heads:
            return features, None
        logits = self.heads[sensor](features)
        return features, logits[:, 1] - logits[:, 0]

    def forward_universal_s2_concat(
        self, images: torch.Tensor, channel_ids: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.universal_s2 is None:
            features = self.forward_sensor("s2", images, channel_ids)
            head = self.heads["s2"] if "s2" in self.heads else None
        else:
            output = self.universal_s2.forward_features(
                {"imgs": images, "chn_ids": channel_ids}
            )
            features = output["x_norm_clstoken"]
            head = self.universal_s2_head
        if head is None:
            return features, None
        logits = head(features)
        return features, logits[:, 1] - logits[:, 0]

    def fuse_universal_base(
        self,
        concat_features: torch.Tensor,
        sensor_valid: torch.Tensor,
        *,
        row_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.fusion_head is None:
            raise RuntimeError(
                "universal checkpoint does not contain row_fusion_head.*"
            )
        if self.fusion_mode != "max":
            raise RuntimeError(
                "universal checkpoint row_fusion_mode must be 'max', got "
                f"{self.fusion_mode!r}"
            )
        has_sensor = sensor_valid.any(dim=1)
        if not bool(has_sensor.all()):
            missing = torch.nonzero(~has_sensor, as_tuple=False).flatten()
            if row_indices is None:
                identifiers = missing.detach().cpu().tolist()
                label = "batch positions"
            else:
                identifiers = (
                    row_indices.detach().cpu().index_select(
                        0, missing.detach().cpu()
                    ).tolist()
                )
                label = "query360_index values"
            raise RuntimeError(
                "universal row fusion received "
                f"{int(missing.numel())} rows without a declared legacy "
                f"concat sensor; first {label}={identifiers[:20]}"
            )
        masked = concat_features.masked_fill(
            ~sensor_valid.unsqueeze(-1), -torch.inf
        )
        fused_features = masked.max(dim=1).values
        logits = self.fusion_head(fused_features)
        return logits[:, 1] - logits[:, 0]

    @staticmethod
    def fuse_hybrid_base(
        universal_logits: torch.Tensor,
        sensor_logits: torch.Tensor,
        logit_valid: torch.Tensor,
    ) -> torch.Tensor:
        # Explicit engineering base definition: use the independently
        # verified S2 checkpoint whenever S2 is present; otherwise preserve
        # the exact universal row-fusion output.  This never feeds overridden
        # S2 features into the universal fusion head.
        s2_valid = logit_valid[:, 0]
        return torch.where(s2_valid, sensor_logits[:, 0], universal_logits)


def parse_sensor_weights(value: str) -> dict[str, Path]:
    if not value.strip():
        return {}
    output: dict[str, Path] = {}
    for item in value.split(","):
        sensor, separator, path = item.partition("=")
        if not separator:
            raise ValueError(
                "--sensor-weights entries must have form sensor=/path/checkpoint"
            )
        output[sensor.strip().lower()] = Path(path.strip())
    return output


def command_extract(args: argparse.Namespace) -> None:
    if args.split == "test" and not args.sealed_test:
        raise PermissionError(
            "refusing to read sealed test without explicit --sealed-test"
        )
    if args.sealed_test and args.split != "test":
        raise ValueError("--sealed-test is valid only with --split test")
    manifest_path = Path(args.manifest).expanduser().absolute()
    output_path = Path(args.output_cache).expanduser().absolute()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    if args.raw_cache_readonly_fallback and not args.raw_cache_dir:
        raise ValueError(
            "--raw-cache-readonly-fallback requires --raw-cache-dir"
        )
    local_cache = None
    if args.raw_cache_dir:
        cache_type = (
            ReadOnlyFallbackHashedFileCache
            if args.raw_cache_readonly_fallback
            else StrictHashedFileCache
        )
        local_cache = cache_type(args.raw_cache_dir)
    dataset = LegacyAllRolesDataset(
        manifest_path,
        local_cache=local_cache,
        wv3_srf_csv=args.wv3_srf_csv,
        pad_to_multiple=14,
        allow_heldout_manifest=bool(args.sealed_test),
    )
    if int(args.max_rows) > 0:
        dataset.frame = dataset.frame.iloc[: int(args.max_rows)].copy()
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(args.row_batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": args.device.startswith("cuda"),
        "collate_fn": query360_collate,
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            prefetch_factor=int(args.prefetch_factor),
            persistent_workers=True,
        )
    loader = DataLoader(**loader_kwargs)
    # Start/fork workers before constructing the five-backbone overlay bank.
    # Otherwise Linux workers inherit the already-loaded checkpoint address
    # space and two extractors can exhaust host RAM before the first batch.
    loader_iterator = iter(loader)
    device = configure_device(args.device)
    bank = SensorEncoderBank(
        Path(args.weights), parse_sensor_weights(args.sensor_weights)
    ).to(device)
    bank.requires_grad_(False)
    bank.eval()

    rows = len(dataset)
    feature_dim = bank.embed_dim
    features = torch.zeros(
        rows, len(SENSOR_ORDER), 3, feature_dim, dtype=torch.float16
    )
    universal_features = torch.zeros_like(features)
    valid_mask = torch.zeros(rows, len(SENSOR_ORDER), 3, dtype=torch.bool)
    base_sensor_logits = torch.zeros(rows, len(SENSOR_ORDER), dtype=torch.float32)
    base_sensor_valid = torch.zeros(rows, len(SENSOR_ORDER), dtype=torch.bool)
    base_universal_sensor_logits = torch.zeros_like(base_sensor_logits)
    base_universal_sensor_valid = torch.zeros_like(base_sensor_valid)
    base_universal_logits = torch.zeros(rows, dtype=torch.float32)
    base_hybrid_logits = torch.zeros(rows, dtype=torch.float32)
    labels = torch.zeros(rows, dtype=torch.long)
    seen = torch.zeros(rows, dtype=torch.bool)
    row_ids = dataset.frame["query360_index"].astype(np.int64).tolist()
    global_to_local = {value: index for index, value in enumerate(row_ids)}
    started = time.monotonic()
    observations = 0

    with torch.inference_mode():
        for batch_number, batch in enumerate(loader_iterator, 1):
            local_rows = torch.tensor(
                [global_to_local[int(value)] for value in batch["index"].tolist()]
            )
            if seen[local_rows].any():
                raise RuntimeError("duplicate emitted row")
            seen[local_rows] = True
            valid_mask[local_rows] = batch["valid_mask"]
            labels[local_rows] = batch["labels"]
            batch_size = int(local_rows.numel())
            concat_features = torch.zeros(
                batch_size,
                len(SENSOR_ORDER),
                feature_dim,
                dtype=torch.float32,
                device=device,
            )
            universal_concat_features = torch.zeros_like(concat_features)
            concat_valid = torch.zeros(
                batch_size,
                len(SENSOR_ORDER),
                dtype=torch.bool,
                device=device,
            )
            concat_logits = torch.zeros_like(
                concat_valid, dtype=torch.float32, device=device
            )
            concat_logit_valid = torch.zeros_like(concat_valid)
            global_to_batch = {
                int(value): position
                for position, value in enumerate(batch["index"].tolist())
            }
            for sensor_index, sensor in enumerate(SENSOR_ORDER):
                sensor_batch = batch["sensor_batches"][sensor]
                images = sensor_batch["images"]
                if images.shape[0] > 0:
                    target_rows = torch.tensor(
                        [
                            global_to_local[int(value)]
                            for value in sensor_batch["rows"].tolist()
                        ]
                    )
                    roles = sensor_batch["roles"].long()
                    ids = sensor_batch["channel_ids"].reshape(1, -1)
                    for start in range(
                        0, len(images), int(args.encoder_microbatch)
                    ):
                        stop = min(
                            start + int(args.encoder_microbatch), len(images)
                        )
                        image_chunk = images[start:stop].to(
                            device, non_blocking=True
                        )
                        id_chunk = ids.expand(stop - start, -1).clone().to(
                            device, non_blocking=True
                        )
                        with autocast_context(device, args.amp_dtype):
                            encoded = bank.forward_sensor(
                                sensor, image_chunk, id_chunk
                            )
                            if sensor == "s2" and bank.s2_override:
                                encoded_universal, _unused_logit = (
                                    bank.forward_universal_s2_concat(
                                        image_chunk, id_chunk
                                    )
                                )
                            else:
                                encoded_universal = encoded
                        features[
                            target_rows[start:stop],
                            sensor_index,
                            roles[start:stop],
                        ] = encoded.float().cpu().to(torch.float16)
                        universal_features[
                            target_rows[start:stop],
                            sensor_index,
                            roles[start:stop],
                        ] = (
                            encoded_universal.float().cpu().to(torch.float16)
                        )
                        observations += stop - start

                # Preserve the old three-timepoint concatenated-band path as
                # the zero-init base branch.  This separate collate payload
                # retains declared roles below the strict finite gate and
                # globally pads C/H/W before any per-sensor slicing, exactly
                # as the historical wide-row loader did.
                legacy_batch = batch["legacy_concat_batch"]
                legacy_positions = torch.tensor(
                    [
                        position
                        for position, sensor_name in enumerate(
                            legacy_batch["sensors"]
                        )
                        if sensor_name == sensor
                    ],
                    dtype=torch.long,
                )
                if legacy_positions.numel() == 0:
                    continue
                concatenated = legacy_batch["images"].index_select(
                    0, legacy_positions
                )
                repeated_ids = legacy_batch["channel_ids"].index_select(
                    0, legacy_positions
                )
                complete_rows = (
                    legacy_batch["rows"]
                    .index_select(0, legacy_positions)
                    .tolist()
                )
                if len(complete_rows) != len(set(complete_rows)):
                    raise RuntimeError(
                        f"duplicate declared {sensor} sample within a row"
                    )
                batch_positions = torch.tensor(
                    [global_to_batch[row_value] for row_value in complete_rows],
                    dtype=torch.long,
                    device=device,
                )
                if complete_rows:
                    concat_parts: list[torch.Tensor] = []
                    logit_parts: list[Optional[torch.Tensor]] = []
                    for start in range(
                        0, len(concatenated), int(args.encoder_microbatch)
                    ):
                        stop = min(
                            start + int(args.encoder_microbatch),
                            len(concatenated),
                        )
                        image_chunk = concatenated[start:stop].to(
                            device, non_blocking=True
                        )
                        id_chunk = repeated_ids[start:stop].to(
                            device, non_blocking=True
                        )
                        with autocast_context(device, args.amp_dtype):
                            encoded_concat, old_logit = bank.forward_concat(
                                sensor, image_chunk, id_chunk
                            )
                        concat_parts.append(encoded_concat.float())
                        logit_parts.append(
                            None if old_logit is None else old_logit.float()
                        )
                    encoded_concat = torch.cat(concat_parts)
                    concat_features[
                        batch_positions, sensor_index
                    ] = encoded_concat
                    universal_encoded_concat = encoded_concat
                    universal_old_logits: Optional[torch.Tensor] = (
                        torch.cat(
                            [part for part in logit_parts if part is not None]
                        )
                        if all(part is not None for part in logit_parts)
                        else None
                    )
                    if sensor == "s2" and bank.s2_override:
                        universal_parts: list[torch.Tensor] = []
                        universal_logit_parts: list[Optional[torch.Tensor]] = []
                        for start in range(
                            0,
                            len(concatenated),
                            int(args.encoder_microbatch),
                        ):
                            stop = min(
                                start + int(args.encoder_microbatch),
                                len(concatenated),
                            )
                            image_chunk = concatenated[start:stop].to(
                                device, non_blocking=True
                            )
                            id_chunk = repeated_ids[start:stop].to(
                                device, non_blocking=True
                            )
                            with autocast_context(device, args.amp_dtype):
                                (
                                    universal_part,
                                    universal_old_logit,
                                ) = (
                                    bank.forward_universal_s2_concat(
                                        image_chunk, id_chunk
                                    )
                                )
                            universal_parts.append(universal_part.float())
                            universal_logit_parts.append(
                                None
                                if universal_old_logit is None
                                else universal_old_logit.float()
                            )
                        universal_encoded_concat = torch.cat(universal_parts)
                        universal_old_logits = (
                            torch.cat(
                                [
                                    part
                                    for part in universal_logit_parts
                                    if part is not None
                                ]
                            )
                            if all(
                                part is not None
                                for part in universal_logit_parts
                            )
                            else None
                        )
                    universal_concat_features[
                        batch_positions, sensor_index
                    ] = universal_encoded_concat
                    concat_valid[batch_positions, sensor_index] = True
                    if all(part is not None for part in logit_parts):
                        old_logits = torch.cat(
                            [part for part in logit_parts if part is not None]
                        )
                        concat_logits[
                            batch_positions, sensor_index
                        ] = old_logits
                        concat_logit_valid[
                            batch_positions, sensor_index
                        ] = True
                    if universal_old_logits is not None:
                        base_universal_sensor_logits[
                            local_rows[batch_positions.cpu()],
                            sensor_index,
                        ] = universal_old_logits.cpu()
                        base_universal_sensor_valid[
                            local_rows[batch_positions.cpu()],
                            sensor_index,
                        ] = True
            universal_base = bank.fuse_universal_base(
                universal_concat_features,
                concat_valid,
                row_indices=batch["index"],
            )
            hybrid_base = (
                bank.fuse_hybrid_base(
                    universal_base,
                    concat_logits,
                    concat_logit_valid,
                )
                if bank.s2_override
                else universal_base
            )
            base_sensor_logits[local_rows] = concat_logits.cpu()
            base_sensor_valid[local_rows] = concat_logit_valid.cpu()
            base_universal_logits[local_rows] = universal_base.cpu()
            base_hybrid_logits[local_rows] = hybrid_base.cpu()
            if (
                batch_number % max(1, int(args.log_interval)) == 0
                or batch_number == len(loader)
            ):
                print(
                    f"[extract] split={args.split} batches={batch_number}/"
                    f"{len(loader)} rows={int(seen.sum())}/{rows} "
                    f"observations={observations} "
                    f"elapsed={time.monotonic()-started:.1f}s",
                    flush=True,
                )
    if not seen.all():
        raise RuntimeError("feature extraction missed rows")
    if not valid_mask[:, :, 0].any(dim=1).all():
        bad = torch.nonzero(
            ~valid_mask[:, :, 0].any(dim=1), as_tuple=False
        ).flatten()
        raise RuntimeError(f"rows without usable current token: {bad[:20].tolist()}")
    if not torch.isfinite(features.float()).all():
        raise RuntimeError("non-finite feature cache")
    if not torch.isfinite(universal_features.float()).all():
        raise RuntimeError("non-finite universal feature cache")
    frame = dataset.frame
    payload = {
        "schema_version": "query360-two-axis-feature-cache-v1",
        "script_version": SCRIPT_VERSION,
        "split": args.split,
        # Backward-compatible alias for the engineering hybrid arm.
        "features": features,
        "features_hybrid": features,
        "features_universal": universal_features,
        "valid_mask": valid_mask,
        # Backward-compatible aliases for the engineering hybrid arm.
        "base_sensor_logits": base_sensor_logits,
        "base_sensor_valid": base_sensor_valid,
        "base_sensor_logits_hybrid": base_sensor_logits,
        "base_sensor_valid_hybrid": base_sensor_valid,
        "base_sensor_logits_universal": base_universal_sensor_logits,
        "base_sensor_valid_universal": base_universal_sensor_valid,
        # Backward-compatible alias; formal training selects one of the two
        # explicitly named bases with --base-mode.
        "base_fused_logits": base_hybrid_logits,
        "base_universal_logits": base_universal_logits,
        "base_hybrid_logits": base_hybrid_logits,
        "base_definitions": {
            "universal": (
                "exact default universal checkpoint: sensor-specific "
                "patch embeddings, max row fusion, universal row head"
            ),
            "hybrid": (
                "verified S2 concat checkpoint head when S2 is present; "
                "otherwise exact universal row-fusion log-odds"
            ),
        },
        "labels": labels,
        "ids": frame["id"].astype(str).tolist(),
        "plume_ids": frame["plume_id"].astype(str).tolist(),
        "event_ids": (
            frame["event_id"].astype(str).tolist()
            if "event_id" in frame
            else frame["plume_id"].astype(str).tolist()
        ),
        "availability_signatures": frame["availability_signature"].astype(str).tolist(),
        "sensor_names": list(SENSOR_ORDER),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "rows": rows,
        },
        "encoder": {
            "default_weights": str(args.weights),
            "sensor_weights": {
                key: str(value)
                for key, value in parse_sensor_weights(args.sensor_weights).items()
            },
            "provenance": bank.provenance,
            "state_sha256": state_dict_sha256(bank.state_dict()),
        },
        "extraction": {
            "device": str(device),
            "amp_dtype": args.amp_dtype,
            "row_batch_size": int(args.row_batch_size),
            "encoder_microbatch": int(args.encoder_microbatch),
            "num_workers": int(args.num_workers),
            "observations": int(observations),
            "elapsed_seconds": float(time.monotonic() - started),
            "durable_local_feature_cache": True,
            "training_remote_image_io": False,
            "raw_cache_dir": args.raw_cache_dir or None,
            "raw_cache_mode": (
                "readonly_fallback"
                if args.raw_cache_readonly_fallback
                else ("require_complete" if args.raw_cache_dir else "none")
            ),
            "sealed_test_authorized": bool(args.sealed_test),
        },
        "sealed_test_read": bool(args.split == "test"),
    }
    atomic_torch(output_path, payload)
    if args.split == "test":
        base_reproduction: dict[str, Any] = {
            "withheld": True,
            "reason": (
                "sealed-test extraction never computes or reports a metric; "
                "the locked fitting command evaluates this cache exactly once"
            ),
        }
    else:
        base_reproduction = {
            "withheld": False,
            "universal_binary_f1_at_0_5": float(
                f1_score(
                    labels.numpy(),
                    (torch.sigmoid(base_universal_logits).numpy() >= 0.5),
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "hybrid_binary_f1_at_0_5": float(
                f1_score(
                    labels.numpy(),
                    (torch.sigmoid(base_hybrid_logits).numpy() >= 0.5),
                    pos_label=1,
                    zero_division=0,
                )
            ),
        }
    audit = {
        "schema_version": payload["schema_version"],
        "split": args.split,
        "rows": rows,
        "observations": int(valid_mask.sum()),
        "feature_shape": list(features.shape),
        "universal_feature_shape": list(universal_features.shape),
        "feature_dtype": str(features.dtype),
        "labels": {
            "0": int((labels == 0).sum()),
            "1": int((labels == 1).sum()),
        },
        "concat_checkpoint_base": {
            "sensor_logit_rows": int(base_sensor_valid.any(dim=1).sum()),
            "all_zero_universal_rows": int(base_universal_logits.eq(0).sum()),
            "all_zero_hybrid_rows": int(base_hybrid_logits.eq(0).sum()),
            "zero_init_residual_preserves_base": True,
            "reproduction_metrics": base_reproduction,
            "definition": payload["base_definitions"],
        },
        "manifest": payload["manifest"],
        "encoder": payload["encoder"],
        "extraction": payload["extraction"],
        "cache": str(output_path),
        "cache_sha256": sha256_file(output_path),
    }
    atomic_json(output_path.with_suffix(output_path.suffix + ".audit.json"), audit)
    print(json.dumps(audit, indent=2), flush=True)


def best_positive_f1_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    order = np.argsort(-probabilities, kind="stable")
    sorted_y = labels[order]
    sorted_p = probabilities[order]
    tp = np.cumsum(sorted_y == 1)
    fp = np.cumsum(sorted_y == 0)
    ends = np.flatnonzero(np.r_[sorted_p[1:] != sorted_p[:-1], True])
    positives = int(np.sum(labels == 1))
    tp_at = tp[ends].astype(np.float64)
    fp_at = fp[ends].astype(np.float64)
    fn_at = positives - tp_at
    score = np.divide(
        2 * tp_at,
        2 * tp_at + fp_at + fn_at,
        out=np.zeros_like(tp_at),
        where=(2 * tp_at + fp_at + fn_at) > 0,
    )
    best = int(np.argmax(score))
    return float(sorted_p[ends[best]]), float(score[best])


def classification_metrics(
    labels: torch.Tensor, probabilities: np.ndarray
) -> dict[str, Any]:
    target = labels.cpu().numpy().astype(np.int64)
    fixed = fixed_threshold_metrics(labels, probabilities, threshold=0.5)
    threshold, best_binary = best_positive_f1_threshold(target, probabilities)
    best_prediction = (probabilities >= threshold).astype(np.int64)
    return {
        **fixed,
        "best_binary_f1": best_binary,
        "best_binary_f1_threshold": threshold,
        "best_macro_f1_at_binary_threshold": float(
            f1_score(
                target,
                best_prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
    }


def fixed_threshold_metrics(
    labels: torch.Tensor,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Metrics at a precommitted threshold; never searches test labels."""

    target = labels.cpu().numpy().astype(np.int64)
    prediction = (probabilities >= float(threshold)).astype(np.int64)
    suffix = "0_5" if float(threshold) == 0.5 else "locked_threshold"
    binary_f1 = float(
        f1_score(target, prediction, pos_label=1, zero_division=0)
    )
    macro_f1 = float(
        f1_score(
            target,
            prediction,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )
    balanced = float(balanced_accuracy_score(target, prediction))
    positive_rate = float(prediction.mean())
    return {
        "rows": int(len(target)),
        "positives": int(np.sum(target == 1)),
        "binary_f1": binary_f1,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced,
        "predicted_positive_rate": positive_rate,
        f"binary_f1_at_{suffix}": binary_f1,
        f"macro_f1_at_{suffix}": macro_f1,
        f"balanced_accuracy_at_{suffix}": balanced,
        f"predicted_positive_rate_at_{suffix}": positive_rate,
        "decision_threshold": float(threshold),
        "ap": float(average_precision_score(target, probabilities)),
        "auc": float(roc_auc_score(target, probabilities)),
    }


def stratified_metrics(
    labels: torch.Tensor,
    probabilities: np.ndarray,
    availability_signatures: Sequence[str],
) -> dict[str, Any]:
    output = classification_metrics(labels, probabilities)
    target = labels.cpu()
    signatures = np.asarray([str(value) for value in availability_signatures])
    availability: dict[str, Any] = {}
    for signature in sorted(set(signatures.tolist())):
        positions = np.flatnonzero(signatures == signature)
        subset_labels = target[torch.from_numpy(positions).long()]
        if len(torch.unique(subset_labels)) < 2:
            continue
        availability[signature] = classification_metrics(
            subset_labels, probabilities[positions]
        )
    output["availability"] = availability
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
        subset_labels = target[torch.from_numpy(positions).long()]
        if len(torch.unique(subset_labels)) < 2:
            continue
        by_sensor[sensor] = classification_metrics(
            subset_labels, probabilities[positions]
        )
    output["by_sensor"] = by_sensor
    return output


def stratified_locked_metrics(
    labels: torch.Tensor,
    probabilities: np.ndarray,
    availability_signatures: Sequence[str],
    *,
    locked_threshold: float,
) -> dict[str, Any]:
    output = fixed_threshold_metrics(
        labels, probabilities, threshold=locked_threshold
    )
    # Also report the conventional fixed 0.5 operating point, but never a
    # test-optimized threshold.
    output["at_0_5"] = fixed_threshold_metrics(
        labels, probabilities, threshold=0.5
    )
    target = labels.cpu()
    signatures = np.asarray([str(value) for value in availability_signatures])
    availability: dict[str, Any] = {}
    for signature in sorted(set(signatures.tolist())):
        positions = np.flatnonzero(signatures == signature)
        subset_labels = target[torch.from_numpy(positions).long()]
        if len(torch.unique(subset_labels)) < 2:
            continue
        availability[signature] = fixed_threshold_metrics(
            subset_labels,
            probabilities[positions],
            threshold=locked_threshold,
        )
    output["availability"] = availability
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
        subset_labels = target[torch.from_numpy(positions).long()]
        if len(torch.unique(subset_labels)) < 2:
            continue
        by_sensor[sensor] = fixed_threshold_metrics(
            subset_labels,
            probabilities[positions],
            threshold=locked_threshold,
        )
    output["by_sensor"] = by_sensor
    output["threshold_source"] = "locked development threshold"
    output["test_threshold_search_performed"] = False
    return output


def load_feature_cache(
    path: Path, expected_split: str | Sequence[str]
) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "query360-two-axis-feature-cache-v1":
        raise ValueError(f"{path}: unsupported cache")
    allowed = (
        {expected_split}
        if isinstance(expected_split, str)
        else {str(value) for value in expected_split}
    )
    if payload.get("split") not in allowed:
        raise ValueError(
            f"{path}: split={payload.get('split')}, expected={expected_split}"
        )
    features = payload["features"]
    valid = payload["valid_mask"]
    labels = payload["labels"]
    if features.ndim != 4 or valid.shape != features.shape[:3]:
        raise ValueError(f"{path}: malformed feature/mask tensors")
    for key in ("features_universal", "features_hybrid"):
        if key in payload and payload[key].shape != features.shape:
            raise ValueError(f"{path}: malformed {key}")
    if labels.shape != features.shape[:1]:
        raise ValueError(f"{path}: malformed labels")
    if payload.get("base_fused_logits") is not None:
        if payload["base_fused_logits"].shape != labels.shape:
            raise ValueError(f"{path}: malformed base_fused_logits")
        if payload["base_sensor_logits"].shape != valid.shape[:2]:
            raise ValueError(f"{path}: malformed base_sensor_logits")
    for key in ("base_universal_logits", "base_hybrid_logits"):
        if key in payload and payload[key].shape != labels.shape:
            raise ValueError(f"{path}: malformed {key}")
    for key in (
        "base_sensor_logits_universal",
        "base_sensor_logits_hybrid",
    ):
        if key in payload and payload[key].shape != valid.shape[:2]:
            raise ValueError(f"{path}: malformed {key}")
    return payload


def _base_logits_for_mode(
    cache: Mapping[str, Any], base_mode: str
) -> torch.Tensor:
    key = f"base_{base_mode}_logits"
    if key not in cache:
        if base_mode == "hybrid" and "base_fused_logits" in cache:
            return cache["base_fused_logits"]
        raise ValueError(f"feature cache has no {key}")
    return cache[key]


def _features_for_mode(
    cache: Mapping[str, Any], base_mode: str
) -> torch.Tensor:
    key = f"features_{base_mode}"
    if key in cache:
        return cache[key]
    if base_mode == "hybrid":
        return cache["features"]
    raise ValueError(f"feature cache has no {key}")


def _sensor_base_for_mode(
    cache: Mapping[str, Any], base_mode: str
) -> Optional[torch.Tensor]:
    key = f"base_sensor_logits_{base_mode}"
    if key in cache:
        return cache[key]
    if base_mode == "hybrid":
        return cache.get("base_sensor_logits")
    return None


def predict_probabilities(
    model: TwoAxisQuery360Head,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    arm: str,
    base_mode: str,
) -> np.ndarray:
    model.eval()
    probabilities: list[torch.Tensor] = []
    mode_features = _features_for_mode(cache, base_mode)
    rows = int(mode_features.shape[0])
    with torch.inference_mode():
        for indices in fixed_epoch_batches(
            rows, batch_size=batch_size, seed=0, epoch=0, shuffle=False
        ):
            features = mode_features[indices].to(
                device=device, dtype=torch.float32
            )
            valid = cache["valid_mask"][indices].to(device)
            base_fused = _base_logits_for_mode(cache, base_mode)
            base_sensor = _sensor_base_for_mode(cache, base_mode)
            output = model(
                features,
                valid,
                arm=arm,
                base_fused_logits=(
                    base_fused[indices].to(device)
                ),
                base_sensor_logits=(
                    None
                    if base_sensor is None
                    else base_sensor[indices].to(device)
                ),
            )
            probabilities.append(torch.sigmoid(output.fused_logits).cpu())
    probability = torch.cat(probabilities).numpy()
    return probability


def evaluate(
    model: TwoAxisQuery360Head,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    arm: str,
    base_mode: str,
) -> tuple[dict[str, Any], np.ndarray]:
    probability = predict_probabilities(
        model,
        cache,
        batch_size=batch_size,
        device=device,
        arm=arm,
        base_mode=base_mode,
    )
    return (
        stratified_metrics(
            cache["labels"],
            probability,
            cache["availability_signatures"],
        ),
        probability,
    )


def command_train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run_status.json"
    atomic_json(
        status_path,
        {
            "status": "running",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "sealed_test_read": False,
            "sealed_test_evaluations": 0,
        },
    )
    try:
        train_cache = load_feature_cache(
            Path(args.train_cache), ("train_core", "train")
        )
        dev_cache = load_feature_cache(
            Path(args.dev_cache), ("dev", "evaluation")
        )
        if train_cache["encoder"] != dev_cache["encoder"]:
            raise ValueError("train_core/dev encoder provenance differs")
        if set(train_cache["plume_ids"]) & set(dev_cache["plume_ids"]):
            raise RuntimeError("train_core/dev plume overlap")
        train_events = set(train_cache.get("event_ids", []))
        dev_events = set(dev_cache.get("event_ids", []))
        if train_events and dev_events and train_events & dev_events:
            raise RuntimeError("train_core/dev canonical event overlap")
        set_deterministic_seed(args.seed)
        model = TwoAxisQuery360Head(
            int(train_cache["features"].shape[-1]),
            num_sensors=int(train_cache["features"].shape[1]),
            num_roles=int(train_cache["features"].shape[2]),
            model_dim=int(args.model_dim),
            num_heads=int(args.num_heads),
            temporal_depth=int(args.temporal_depth),
            mlp_ratio=float(args.mlp_ratio),
            dropout=float(args.dropout),
        )
        model_config = {
            "embed_dim": int(train_cache["features"].shape[-1]),
            "num_sensors": int(train_cache["features"].shape[1]),
            "num_roles": int(train_cache["features"].shape[2]),
            "model_dim": int(args.model_dim),
            "num_heads": int(args.num_heads),
            "temporal_depth": int(args.temporal_depth),
            "mlp_ratio": float(args.mlp_ratio),
            "dropout": float(args.dropout),
        }
        initial_sha = state_dict_sha256(model.state_dict())
        signature = model_parameter_signature(model)
        device = configure_device(args.device)
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        labels = train_cache["labels"].float()
        if args.pos_weight:
            positives = float(labels.sum())
            pos_weight: Optional[float] = float(
                (labels.numel() - positives) / max(positives, 1.0)
            )
        else:
            pos_weight = None
        training_features = _features_for_mode(
            train_cache, args.base_mode
        )
        training_fused_base = _base_logits_for_mode(
            train_cache, args.base_mode
        )
        training_sensor_base = _sensor_base_for_mode(
            train_cache, args.base_mode
        )
        initial_metrics, initial_probabilities = evaluate(
            model,
            dev_cache,
            batch_size=int(args.eval_batch_size),
            device=device,
            arm=args.arm,
            base_mode=args.base_mode,
        )
        history = [
            {
                "epoch": 0,
                "train": None,
                "dev": initial_metrics,
                "elapsed_seconds": 0.0,
                "interpretation": (
                    "unchanged concat-temporal checkpoint base before the "
                    "zero-initialized two-axis residual is trained"
                ),
            }
        ]
        best: Optional[dict[str, Any]] = copy.deepcopy(history[0])
        atomic_torch(
            output_dir / "checkpoint_best.pth",
            {
                "schema_version": "query360-two-axis-head-v1",
                "epoch": 0,
                "arm": args.arm,
                "base_mode": args.base_mode,
                "model_config": model_config,
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "initial_state_sha256": initial_sha,
                "parameter_signature": signature,
                "dev": initial_metrics,
                "locked_threshold_candidate": float(
                    initial_metrics["best_binary_f1_threshold"]
                ),
            },
        )
        pd.DataFrame(
            {
                "id": dev_cache["ids"],
                "plume_id": dev_cache["plume_ids"],
                "availability_signature": dev_cache[
                    "availability_signatures"
                ],
                "label": dev_cache["labels"].tolist(),
                "probability": initial_probabilities,
            }
        ).to_csv(output_dir / "dev_predictions_best.csv", index=False)
        atomic_json(output_dir / "metrics_history.json", history)
        s2_initial = initial_metrics.get("by_sensor", {}).get("s2", {})
        print(
            "[two-axis] epoch=0 checkpoint-preserving dev base "
            f"binary_F1@.5={initial_metrics['binary_f1_at_0_5']:.6f} "
            f"macro_F1@.5={initial_metrics['macro_f1_at_0_5']:.6f} "
            f"S2_binary_F1@.5={s2_initial.get('binary_f1_at_0_5')}",
            flush=True,
        )
        started = time.monotonic()
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            total_sum = base_sum = axis_sum = 0.0
            seen = 0
            batches = fixed_epoch_batches(
                len(labels),
                batch_size=int(args.batch_size),
                seed=int(args.seed),
                epoch=epoch,
                shuffle=True,
            )
            for indices in batches:
                features = training_features[indices].to(
                    device=device, dtype=torch.float32
                )
                valid = train_cache["valid_mask"][indices].to(device)
                target = labels[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    features,
                    valid,
                    arm=args.arm,
                    base_fused_logits=(
                        training_fused_base[indices].to(device)
                    ),
                    base_sensor_logits=(
                        None
                        if training_sensor_base is None
                        else training_sensor_base[indices].to(device)
                    ),
                )
                base = transient_query_loss(
                    output,
                    target,
                    auxiliary_weight=float(args.sensor_aux_weight),
                    pos_weight=pos_weight,
                )
                weight_tensor = (
                    None
                    if pos_weight is None
                    else torch.tensor(pos_weight, device=device)
                )
                axis_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(
                        output.time_then_sensor_logits,
                        target,
                        pos_weight=weight_tensor,
                    )
                    + F.binary_cross_entropy_with_logits(
                        output.sensor_then_time_logits,
                        target,
                        pos_weight=weight_tensor,
                    )
                )
                loss = base.total + float(args.axis_aux_weight) * axis_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(args.grad_clip)
                    )
                optimizer.step()
                count = int(indices.numel())
                total_sum += float(loss.detach()) * count
                base_sum += float(base.total.detach()) * count
                axis_sum += float(axis_loss.detach()) * count
                seen += count
            metrics, probabilities = evaluate(
                model,
                dev_cache,
                batch_size=int(args.eval_batch_size),
                device=device,
                arm=args.arm,
                base_mode=args.base_mode,
            )
            record = {
                "epoch": epoch,
                "train": {
                    "loss": total_sum / seen,
                    "base_loss": base_sum / seen,
                    "axis_aux_loss": axis_sum / seen,
                    "rows": seen,
                    "steps": len(batches),
                },
                "dev": metrics,
                "elapsed_seconds": float(time.monotonic() - started),
            }
            history.append(record)
            atomic_json(output_dir / "metrics_history.json", history)
            score = float(metrics[args.selection_metric])
            if best is None or score > float(
                best["dev"][args.selection_metric]
            ):
                best = copy.deepcopy(record)
                atomic_torch(
                    output_dir / "checkpoint_best.pth",
                    {
                        "schema_version": "query360-two-axis-head-v1",
                        "epoch": epoch,
                        "arm": args.arm,
                        "base_mode": args.base_mode,
                        "model_config": model_config,
                        "model": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "initial_state_sha256": initial_sha,
                        "parameter_signature": signature,
                        "dev": metrics,
                        "locked_threshold_candidate": float(
                            metrics["best_binary_f1_threshold"]
                        ),
                    },
                )
                pd.DataFrame(
                    {
                        "id": dev_cache["ids"],
                        "plume_id": dev_cache["plume_ids"],
                        "availability_signature": dev_cache[
                            "availability_signatures"
                        ],
                        "label": dev_cache["labels"].tolist(),
                        "probability": probabilities,
                    }
                ).to_csv(
                    output_dir / "dev_predictions_best.csv", index=False
                )
            print(
                f"[two-axis] epoch={epoch}/{args.epochs} "
                f"loss={record['train']['loss']:.6f} "
                f"binary_F1@.5={metrics['binary_f1_at_0_5']:.6f} "
                f"macro_F1@.5={metrics['macro_f1_at_0_5']:.6f} "
                f"best_binary_F1={metrics['best_binary_f1']:.6f} "
                f"threshold={metrics['best_binary_f1_threshold']:.6f} "
                f"AP={metrics['ap']:.6f} AUC={metrics['auc']:.6f}",
                flush=True,
            )
        if best is None:
            raise RuntimeError("no epoch completed")
        checkpoint_path = output_dir / "checkpoint_best.pth"
        checkpoint_payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint_payload["model"], strict=True)
        model.to(device)
        locked_threshold = float(
            checkpoint_payload["locked_threshold_candidate"]
        )
        lock = {
            "schema_version": "query360-two-axis-selection-lock-v1",
            "locked_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "best_epoch": int(checkpoint_payload["epoch"]),
            "selection_metric": args.selection_metric,
            "selection_score": float(
                checkpoint_payload["dev"][args.selection_metric]
            ),
            "locked_threshold": locked_threshold,
            "arm": args.arm,
            "base_mode": args.base_mode,
            "model_config": checkpoint_payload["model_config"],
            "encoder": train_cache["encoder"],
            "split_guard": {
                "train_plume_ids": sorted(
                    {str(value) for value in train_cache["plume_ids"]}
                ),
                "dev_plume_ids": sorted(
                    {str(value) for value in dev_cache["plume_ids"]}
                ),
                "source_train_event_ids": sorted(
                    train_events | dev_events
                ),
            },
            "train_manifest": train_cache["manifest"],
            "dev_manifest": dev_cache["manifest"],
            "threshold_source": (
                "positive-class F1 maximization on development only at the "
                "development-selected epoch"
            ),
            "test_cache_read_before_lock": False,
            "protocol_caveat": (
                "engineering selection: the warm-start 360m checkpoint was "
                "trained on the old source-train pool that contains this dev "
                "subset; do not call the dev comparison leakage-free"
            ),
        }
        atomic_json(output_dir / "selection_lock.json", lock)
        summary = {
            "schema_version": "query360-two-axis-full-legacy-summary-v2",
            "script_version": SCRIPT_VERSION,
            "status": "complete",
            "protocol": (
                "complete legacy 360m train_core/dev manifests; dev selects "
                "epoch and threshold; sealed test is not read here and must "
                "be evaluated later with evaluate-locked"
            ),
            "train_rows": int(train_cache["features"].shape[0]),
            "dev_rows": int(dev_cache["features"].shape[0]),
            "encoder": train_cache["encoder"],
            "model": {
                "arm": args.arm,
                "base_mode": args.base_mode,
                "initial_state_sha256": initial_sha,
                "parameter_signature": signature,
                "model_dim": int(args.model_dim),
                "num_heads": int(args.num_heads),
                "temporal_depth": int(args.temporal_depth),
            },
            "training": {
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "sensor_aux_weight": float(args.sensor_aux_weight),
                "axis_aux_weight": float(args.axis_aux_weight),
                "selection_metric": args.selection_metric,
            },
            "best": best,
            "selection_lock": lock,
            "sealed_test": None,
            "sealed_test_read": False,
            "sealed_test_evaluations": 0,
            "history": history,
            "metric_guardrail": (
                "epoch and threshold are selected only on dev; this command "
                "cannot accept or read a sealed-test cache"
            ),
            "protocol_caveat": lock["protocol_caveat"],
        }
        atomic_json(output_dir / "summary.json", summary)
        atomic_json(
            status_path,
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "best_epoch": int(best["epoch"]),
                "best_dev_metrics": best["dev"],
                "locked_threshold": locked_threshold,
                "sealed_test_read": False,
                "sealed_test_evaluations": 0,
            },
        )
    except Exception as error:
        atomic_json(
            status_path,
            {
                "status": "failed",
                "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "sealed_test_read": False,
            },
        )
        raise


def _build_locked_model(config: Mapping[str, Any]) -> TwoAxisQuery360Head:
    required = {
        "embed_dim",
        "num_sensors",
        "num_roles",
        "model_dim",
        "num_heads",
        "temporal_depth",
        "mlp_ratio",
        "dropout",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"locked model config misses {missing}")
    return TwoAxisQuery360Head(
        int(config["embed_dim"]),
        num_sensors=int(config["num_sensors"]),
        num_roles=int(config["num_roles"]),
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        temporal_depth=int(config["temporal_depth"]),
        mlp_ratio=float(config["mlp_ratio"]),
        dropout=float(config["dropout"]),
    )


def command_evaluate_locked(args: argparse.Namespace) -> None:
    """Evaluate a prelocked checkpoint on sealed test exactly once."""

    if not args.sealed_test:
        raise PermissionError(
            "refusing sealed-test evaluation without explicit --sealed-test"
        )
    output_dir = Path(args.output_dir).expanduser().absolute()
    result_path = output_dir / "sealed_test_result.json"
    predictions_path = output_dir / "sealed_test_predictions.csv"
    status_path = output_dir / "locked_eval_status.json"
    conflicts = [
        str(path)
        for path in (result_path, predictions_path, status_path)
        if path.exists()
    ]
    if conflicts:
        raise FileExistsError(
            "refusing a second sealed-test evaluation; existing artifacts: "
            f"{conflicts}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    sealed_test_read = False
    atomic_json(
        status_path,
        {
            "status": "validating_lock",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "sealed_test_read": False,
            "sealed_test_evaluations": 0,
        },
    )
    try:
        lock_path = Path(args.selection_lock).expanduser().absolute()
        checkpoint_path = Path(args.checkpoint).expanduser().absolute()
        with lock_path.open("r", encoding="utf-8") as stream:
            lock = json.load(stream)
        if lock.get("schema_version") != (
            "query360-two-axis-selection-lock-v1"
        ):
            raise ValueError("unsupported selection lock schema")
        observed_checkpoint_sha = sha256_file(checkpoint_path)
        if observed_checkpoint_sha != lock.get("checkpoint_sha256"):
            raise ValueError("checkpoint SHA does not match selection lock")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("schema_version") != (
            "query360-two-axis-head-v1"
        ):
            raise ValueError("unsupported locked checkpoint schema")
        if int(checkpoint["epoch"]) != int(lock["best_epoch"]):
            raise ValueError("checkpoint epoch differs from selection lock")
        if checkpoint.get("arm") != lock.get("arm"):
            raise ValueError("checkpoint arm differs from selection lock")
        if checkpoint.get("base_mode") != lock.get("base_mode"):
            raise ValueError(
                "checkpoint base_mode differs from selection lock"
            )
        if checkpoint.get("model_config") != lock.get("model_config"):
            raise ValueError(
                "checkpoint model_config differs from selection lock"
            )
        locked_threshold = float(lock["locked_threshold"])
        if locked_threshold != float(
            checkpoint["locked_threshold_candidate"]
        ):
            raise ValueError(
                "locked threshold differs from checkpoint dev candidate"
            )
        model = _build_locked_model(lock["model_config"])
        if model_parameter_signature(model) != checkpoint.get(
            "parameter_signature"
        ):
            raise ValueError("locked model parameter signature differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        device = configure_device(args.device)
        model.to(device)

        # No test-cache path is touched above this point.
        sealed_test_read = True
        atomic_json(
            status_path,
            {
                "status": "sealed_test_running",
                "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "sealed_test_read": True,
                "sealed_test_evaluations": 0,
                "lock_sha256": sha256_file(lock_path),
                "checkpoint_sha256": observed_checkpoint_sha,
            },
        )
        test_cache_path = Path(args.test_cache).expanduser().absolute()
        test_cache = load_feature_cache(test_cache_path, "test")
        if test_cache["encoder"] != lock.get("encoder"):
            raise ValueError("sealed-test encoder differs from locked encoder")
        split_guard = lock.get("split_guard")
        if not isinstance(split_guard, Mapping):
            raise ValueError("selection lock has no split_guard")
        train_plumes = {
            str(value)
            for value in split_guard.get("train_plume_ids", [])
        }
        dev_plumes = {
            str(value)
            for value in split_guard.get("dev_plume_ids", [])
        }
        test_plumes = {str(value) for value in test_cache["plume_ids"]}
        overlap = (train_plumes | dev_plumes) & test_plumes
        if overlap:
            raise RuntimeError(
                f"sealed-test plume overlap: {sorted(overlap)[:20]}"
            )
        source_train_events = {
            str(value)
            for value in split_guard.get("source_train_event_ids", [])
        }
        test_events = {
            str(value) for value in test_cache.get("event_ids", [])
        }
        event_overlap = source_train_events & test_events
        test_probabilities = predict_probabilities(
            model,
            test_cache,
            batch_size=int(args.eval_batch_size),
            device=device,
            arm=str(lock["arm"]),
            base_mode=str(lock["base_mode"]),
        )
        test_metrics = stratified_locked_metrics(
            test_cache["labels"],
            test_probabilities,
            test_cache["availability_signatures"],
            locked_threshold=locked_threshold,
        )
        pd.DataFrame(
            {
                "id": test_cache["ids"],
                "plume_id": test_cache["plume_ids"],
                "availability_signature": test_cache[
                    "availability_signatures"
                ],
                "label": test_cache["labels"].tolist(),
                "probability": test_probabilities,
                "locked_prediction": (
                    test_probabilities >= locked_threshold
                ).astype(np.int64),
            }
        ).to_csv(predictions_path, index=False)
        result = {
            "artifact_type": "sealed_test_result",
            "schema_version": "query360-two-axis-locked-test-v1",
            "evaluation_count": 1,
            "evaluated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": observed_checkpoint_sha,
                "epoch": int(lock["best_epoch"]),
            },
            "selection_lock": {
                "path": str(lock_path),
                "sha256": sha256_file(lock_path),
                "locked_threshold": locked_threshold,
                "selection_metric": lock["selection_metric"],
                "selection_score": lock["selection_score"],
                "arm": lock["arm"],
                "base_mode": lock["base_mode"],
            },
            "cache": {
                "path": str(test_cache_path),
                "manifest": test_cache["manifest"],
            },
            "metrics": test_metrics,
            "plume_overlap_with_train_or_dev": 0,
            "canonical_event_overlap_with_source_train": int(
                len(event_overlap)
            ),
            "canonical_event_overlap_note": (
                "observed property of the immutable old 360m split; not used "
                "for model, epoch, or threshold selection"
            ),
            "test_threshold_search_performed": False,
            "test_cache_read_after_selection_lock": True,
        }
        atomic_json(result_path, result)
        atomic_json(
            status_path,
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "sealed_test_read": True,
                "sealed_test_evaluations": 1,
                "result": str(result_path),
            },
        )
    except Exception as error:
        atomic_json(
            status_path,
            {
                "status": "failed",
                "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "sealed_test_read": sealed_test_read,
                "sealed_test_evaluations": 0,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--train-csv", default=DEFAULT_TRAIN)
    prepare.add_argument("--eval-csv", default=DEFAULT_EVAL)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(function=command_prepare)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--manifest", required=True)
    extract.add_argument(
        "--split",
        choices=["train_core", "dev", "test", "train", "evaluation"],
        required=True,
    )
    extract.add_argument("--output-cache", required=True)
    extract.add_argument("--weights", default=DEFAULT_WEIGHTS)
    extract.add_argument(
        "--sensor-weights",
        default="",
        help="Optional exact map s2=...,l89=...,emit=...,s5p=...",
    )
    extract.add_argument("--raw-cache-dir", default="")
    extract.add_argument(
        "--raw-cache-readonly-fallback",
        action="store_true",
        help=(
            "reuse completed hashed entries and stream misses from source; "
            "never add to or delete from the raw cache"
        ),
    )
    extract.add_argument(
        "--sealed-test",
        action="store_true",
        help="explicit authorization required with --split test",
    )
    extract.add_argument("--wv3-srf-csv", default=str(DEFAULT_WV3_SRF))
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument(
        "--amp-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
    )
    extract.add_argument("--row-batch-size", type=int, default=64)
    extract.add_argument("--encoder-microbatch", type=int, default=256)
    extract.add_argument("--num-workers", type=int, default=16)
    extract.add_argument("--prefetch-factor", type=int, default=2)
    extract.add_argument("--log-interval", type=int, default=25)
    extract.add_argument("--max-rows", type=int, default=0)
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(function=command_extract)

    train = subparsers.add_parser("train")
    train.add_argument("--train-cache", required=True)
    train.add_argument("--dev-cache", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument(
        "--arm",
        choices=[
            "current_only",
            "two_axis_query",
            "scale_aware_two_axis_query",
        ],
        default="scale_aware_two_axis_query",
    )
    train.add_argument(
        "--base-mode",
        choices=["universal", "hybrid"],
        default="hybrid",
        help=(
            "universal=exact universal row fusion; hybrid=verified S2 head "
            "when S2 exists, otherwise exact universal row fusion"
        ),
    )
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--eval-batch-size", type=int, default=2048)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.02)
    train.add_argument("--sensor-aux-weight", type=float, default=0.3)
    train.add_argument("--axis-aux-weight", type=float, default=0.15)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--model-dim", type=int, default=256)
    train.add_argument("--num-heads", type=int, default=8)
    train.add_argument("--temporal-depth", type=int, default=2)
    train.add_argument("--mlp-ratio", type=float, default=2.0)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--pos-weight", action="store_true")
    train.add_argument(
        "--selection-metric",
        choices=[
            "binary_f1_at_0_5",
            "macro_f1_at_0_5",
            "best_binary_f1",
            "ap",
            "auc",
        ],
        default="best_binary_f1",
    )
    train.add_argument("--device", default="cuda:0")
    train.set_defaults(function=command_train)

    locked = subparsers.add_parser(
        "evaluate-locked",
        help=(
            "evaluate one prelocked checkpoint/threshold on sealed test "
            "without training or selection"
        ),
    )
    locked.add_argument("--checkpoint", required=True)
    locked.add_argument("--selection-lock", required=True)
    locked.add_argument("--test-cache", required=True)
    locked.add_argument("--output-dir", required=True)
    locked.add_argument(
        "--sealed-test",
        action="store_true",
        help="explicit authorization required for the one sealed evaluation",
    )
    locked.add_argument("--eval-batch-size", type=int, default=2048)
    locked.add_argument("--device", default="cuda:0")
    locked.set_defaults(function=command_evaluate_locked)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
