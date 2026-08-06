#!/usr/bin/env python3
"""Balanced multi-sensor methane classification with one shared Panopticon.

This runner deliberately reuses the deterministic loading and timestamp logic
from the four current sensor-specific Panopticon programs.  It does *not* call
their ``__getitem__`` methods, because those methods can replace an unreadable
sample with a random row.  ``StrictIndexedDataset`` calls ``_get_one`` exactly
once and fails with the original sensor/index context instead.

The model owns:

* one official wavelength-aware Panopticon ViT-B/14 backbone;
* one small temporal adapter per sensor (time projection + slot embeddings);
* one LayerNorm/linear binary classifier per sensor.

One optimization round consumes exactly one batch from every enabled sensor.
Shorter loaders are cycled, so every enabled sensor contributes the same
number of batches and the loss is averaged across sensors.  Inference calls
``model(sensor, x_dict)`` and therefore needs only one sensor's input.

The selected input plans are intentionally narrow:

* S2: current frame only;
* L89: raw t0, prev1, seasonal;
* S5P: all six raw frames from the existing NPZ;
* EMIT32: matched t0, prev1, seasonal source frames, configurable as raw or
  normalized t0-history residuals.

Use ``--self-test`` for a no-I/O CPU integration test and ``--smoke-30`` for a
frozen-official-backbone, at-most-30-round real-data smoke.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Match the sensor-specific programs and avoid xFormers discovery surprises.
os.environ.setdefault("CDIR", str(REPO_ROOT / "thirdparty" / "dinov2" / "configs"))
os.environ.setdefault("XFORMERS_DISABLED", "1")

from Upgraded_dataset import dino_classifier_head_emit32_temporal_satmae as emit_module  # noqa: E402
from Upgraded_dataset import dino_classifier_head_l89_temporal_satmae as l89_module  # noqa: E402
from Upgraded_dataset import dino_classifier_head_s2_temporal_satmae as s2_module  # noqa: E402
from Upgraded_dataset import dino_classifier_head_s5p_temporal_satmae as s5p_module  # noqa: E402
from research.pretraining_20260727.temporal_input_modes import output_timepoints  # noqa: E402
from thirdparty.dinov2.models.panopticon import (  # noqa: E402
    get_1d_sincos_pos_embed_from_grid_torch,
)


SENSOR_ORDER: Tuple[str, ...] = ("s2", "l89", "s5p", "emit")
DEFAULT_WORK_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/multisensor_panopticon"
)
DEFAULT_CACHE_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/cache/multisensor_panopticon"
)

CANONICAL_CSVS = {
    "s2": {
        "train": REPO_ROOT
        / "Upgraded_dataset/s2_v14_six_unique_dates_temporal_split/train.csv",
        "val": REPO_ROOT
        / "Upgraded_dataset/s2_v14_six_unique_dates_temporal_split/test.csv",
    },
    "l89": {
        "train": REPO_ROOT
        / (
            "Upgraded_dataset/l89_6time_temporal_hard_event_filtered_split/"
            "L89_temporal_train_hard_event_filtered.csv"
        ),
        "val": REPO_ROOT
        / (
            "Upgraded_dataset/l89_6time_temporal_hard_event_filtered_split/"
            "L89_temporal_test_hard_event_filtered.csv"
        ),
    },
    "s5p": {
        "train": REPO_ROOT
        / "Upgraded_dataset/s5p_6time_temporal_cutoff_2025_12_24/s5p_6time_train.csv",
        "val": REPO_ROOT
        / "Upgraded_dataset/s5p_6time_temporal_cutoff_2025_12_24/s5p_6time_test.csv",
    },
    "emit": {
        "train": REPO_ROOT
        / (
            "Upgraded_dataset/"
            "emit32_full_through_2026_05_strict_temporal_cutoff_2025_06_09/"
            "emit32_temporal_train.csv"
        ),
        "val": REPO_ROOT
        / (
            "Upgraded_dataset/"
            "emit32_full_through_2026_05_strict_temporal_cutoff_2025_06_09/"
            "emit32_temporal_test.csv"
        ),
    },
}

STAGED_MANIFEST_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/manifests_staged"
)
STAGED_CSVS = {
    "s2": {
        "train": STAGED_MANIFEST_ROOT / "s2/train.csv",
        "val": STAGED_MANIFEST_ROOT / "s2/val.csv",
    },
    "l89": {
        "train": STAGED_MANIFEST_ROOT / "l89_3time/train.csv",
        "val": STAGED_MANIFEST_ROOT / "l89_3time/val.csv",
    },
    "s5p": {
        "train": STAGED_MANIFEST_ROOT / "s5p/train.csv",
        "val": STAGED_MANIFEST_ROOT / "s5p/val.csv",
    },
    "emit": {
        "train": STAGED_MANIFEST_ROOT / "emit_3time/train.csv",
        "val": STAGED_MANIFEST_ROOT / "emit_3time/val.csv",
    },
}


def _default_csv(sensor: str, split: str) -> Path:
    """Prefer the already staged inner split, with a portable repo fallback."""

    staged = STAGED_CSVS[sensor][split]
    return staged if staged.is_file() else CANONICAL_CSVS[sensor][split]


DEFAULT_CSVS = {
    sensor: {
        split: _default_csv(sensor, split)
        for split in ("train", "val")
    }
    for sensor in SENSOR_ORDER
}

TIFF_TIME_COLUMNS = {
    "path_t0": "t0_image_time",
    "path_prev1": "prev1_image_time",
    "path_prev2": "prev2_image_time",
    "path_prev3": "prev3_image_time",
    "path_seasonal": "seasonal_image_time",
    "path_year": "year_image_time",
}
S5P_TIME_PATH_COLUMNS: Tuple[str, ...] = (
    "t0_path",
    "prev1_path",
    "prev2_path",
    "prev3_path",
    "seasonal_path",
    "year_path",
)

CANONICAL_EVENT_RULE = "s2-event_group-else-strip-final-hyphen-suffix-v1"
EVENT_SUFFIX_RE = re.compile(r"-[A-Za-z0-9]+$")
EVENT_PROTOCOL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SensorPlan:
    name: str
    train_csv: str
    val_csv: str
    path_columns: Tuple[str, ...]
    time_columns: Tuple[str, ...]
    input_mode: str
    residual_slots: Tuple[str, ...]
    source_timepoints: int
    model_timepoints: int
    channels: int
    resize_size: int
    stats_semantics: str


class SampleLoadError(RuntimeError):
    """A strict, non-recovering sample failure with stable row context."""


class EventProtocolError(ValueError):
    """A fail-closed event-isolation error that retains its audit payload."""

    def __init__(self, message: str, audit: Mapping[str, Any]):
        super().__init__(message)
        self.audit = dict(audit)


class LocalAwareCache:
    """Avoid duplicating files that are already on the local NVMe volume."""

    def __init__(self, delegate: Any, local_roots: Sequence[Path]):
        self.delegate = delegate
        self.local_roots = tuple(
            str(path.expanduser().resolve()).rstrip(os.sep) + os.sep
            for path in local_roots
        )
        self.async_mode = bool(getattr(delegate, "async_mode", False))

    def _already_local(self, path: str) -> bool:
        absolute = os.path.abspath(str(path))
        return any(absolute.startswith(root) for root in self.local_roots)

    def ensure_local(self, path: str) -> str:
        if self._already_local(path):
            return path
        return self.delegate.ensure_local(path)

    def schedule(self, path: str) -> None:
        if not self._already_local(path):
            self.delegate.schedule(path)


class StrictIndexedDataset(Dataset):
    """Call a sensor dataset's deterministic ``_get_one`` exactly once.

    The inherited datasets retain their existing parsing, normalization,
    wavelength IDs, and temporal transformation.  Random replacement is
    intentionally bypassed.  A failing index is never substituted.
    """

    def __init__(self, sensor: str, dataset: Dataset):
        if not hasattr(dataset, "_get_one"):
            raise TypeError(f"{type(dataset).__name__} does not expose deterministic _get_one")
        self.sensor = sensor
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def _row_context(self, index: int) -> str:
        dataframe = getattr(self.dataset, "df", None)
        if dataframe is None:
            base = getattr(self.dataset, "base_ds", None)
            dataframe = getattr(base, "df", None)
        if dataframe is None or not 0 <= index < len(dataframe):
            return f"index={index}"
        row = dataframe.iloc[index]
        fields = []
        for key in ("sample_id", "id", "plume_id", "image_path", "path_t0"):
            if key in row and pd.notna(row[key]):
                fields.append(f"{key}={row[key]}")
        return f"index={index}" + (", " + ", ".join(fields) if fields else "")

    def __getitem__(self, index: int):
        try:
            return self.dataset._get_one(index)  # type: ignore[attr-defined]
        except Exception as exc:
            raise SampleLoadError(
                f"{self.sensor} sample failed without replacement "
                f"({self._row_context(index)}): {exc}"
            ) from exc

    @property
    def labels(self) -> np.ndarray:
        dataframe = getattr(self.dataset, "df", None)
        label_column = getattr(self.dataset, "label_column", "label")
        if dataframe is None:
            base = getattr(self.dataset, "base_ds", None)
            dataframe = getattr(base, "df", None)
            label_column = getattr(base, "label_column", "label")
        if dataframe is None or label_column not in dataframe.columns:
            raise ValueError(f"Cannot resolve labels for {self.sensor}")
        return dataframe[label_column].to_numpy(dtype=np.int64, copy=True)


class SensorTemporalAdapter(nn.Module):
    """Sensor-private SatMAE-style calendar and ordered-slot embeddings."""

    def __init__(self, embed_dim: int, num_timepoints: int, time_embed_dim: int):
        super().__init__()
        if time_embed_dim % 3 != 0 or (time_embed_dim // 3) % 2 != 0:
            raise ValueError(
                "time_embed_dim must divide into three even sinusoidal components "
                "(384 is the inherited setting)"
            )
        self.embed_dim = int(embed_dim)
        self.num_timepoints = int(num_timepoints)
        self.time_embed_dim = int(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, embed_dim, bias=False)
        self.slot_embed = nn.Parameter(
            torch.zeros(1, self.num_timepoints, 1, self.embed_dim)
        )
        # Preserve the released representation at initialization.
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.slot_embed)

    def calendar_embedding(
        self, timestamps: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        if timestamps.ndim != 3 or timestamps.shape[-1] != 3:
            raise ValueError(
                f"timestamps must be (B,T,3), got {tuple(timestamps.shape)}"
            )
        batch_size, num_timepoints, _ = timestamps.shape
        if num_timepoints != self.num_timepoints:
            raise ValueError(
                f"adapter expects exactly {self.num_timepoints} timepoints, "
                f"got {num_timepoints}"
            )
        component_dim = self.time_embed_dim // 3
        flat = timestamps.reshape(-1, 3).float()
        pieces = [
            get_1d_sincos_pos_embed_from_grid_torch(component_dim, flat[:, index])
            for index in range(3)
        ]
        embedding = torch.cat(pieces, dim=1).to(
            device=timestamps.device, dtype=dtype
        )
        return self.time_proj(embedding).reshape(
            batch_size, num_timepoints, 1, self.embed_dim
        )

    def forward(
        self, patch_tokens: torch.Tensor, timestamps: torch.Tensor
    ) -> torch.Tensor:
        if patch_tokens.ndim != 4:
            raise ValueError(
                f"patch_tokens must be (B,T,L,D), got {tuple(patch_tokens.shape)}"
            )
        return (
            patch_tokens
            + self.calendar_embedding(timestamps, dtype=patch_tokens.dtype)
            + self.slot_embed.to(
                device=patch_tokens.device, dtype=patch_tokens.dtype
            )
        )


class SensorClassificationHead(nn.Module):
    """The same minimal DINO-style head used by the four current scripts."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(cls_token))


class SharedPanopticonTemporalClassifier(nn.Module):
    """One wavelength-aware backbone with sensor-private temporal modules."""

    def __init__(
        self,
        backbone: nn.Module,
        sensor_timepoints: Mapping[str, int],
        *,
        time_embed_dim: int = 384,
    ):
        super().__init__()
        if not sensor_timepoints:
            raise ValueError("At least one sensor is required")
        self.backbone = backbone
        self.embed_dim = int(backbone.embed_dim)
        self.adapters = nn.ModuleDict(
            {
                sensor: SensorTemporalAdapter(
                    self.embed_dim, int(timepoints), int(time_embed_dim)
                )
                for sensor, timepoints in sensor_timepoints.items()
            }
        )
        self.heads = nn.ModuleDict(
            {
                sensor: SensorClassificationHead(self.embed_dim)
                for sensor in sensor_timepoints
            }
        )

    def forward_features(self, sensor: str, x_dict: Mapping[str, torch.Tensor]):
        if sensor not in self.adapters:
            raise KeyError(
                f"Unknown sensor {sensor!r}; model has {tuple(self.adapters.keys())}"
            )
        imgs = x_dict["imgs"]
        channel_ids = x_dict["chn_ids"]
        timestamps = x_dict["timestamps"]
        if imgs.ndim != 5:
            raise ValueError(
                f"{sensor}: imgs must be (B,T,C,H,W), got {tuple(imgs.shape)}"
            )
        batch_size, num_timepoints, channels, height, width = imgs.shape
        expected_timepoints = self.adapters[sensor].num_timepoints
        if num_timepoints != expected_timepoints:
            raise ValueError(
                f"{sensor}: expected {expected_timepoints} model timepoints, "
                f"got {num_timepoints}"
            )
        if channel_ids.shape[:2] != (batch_size, num_timepoints):
            raise ValueError(
                f"{sensor}: chn_ids leading shape must be {(batch_size, num_timepoints)}, "
                f"got {tuple(channel_ids.shape)}"
            )
        if channel_ids.shape[-1] != channels:
            raise ValueError(
                f"{sensor}: chn_ids has {channel_ids.shape[-1]} channels but "
                f"imgs has {channels}"
            )
        if timestamps.shape != (batch_size, num_timepoints, 3):
            raise ValueError(
                f"{sensor}: timestamps must be {(batch_size, num_timepoints, 3)}, "
                f"got {tuple(timestamps.shape)}"
            )

        flat_imgs = imgs.reshape(
            batch_size * num_timepoints, channels, height, width
        )
        # Clone because the inherited wavelength embedding coarsens optical
        # IDs in-place.  Dataset tensors and checkpoint-visible inputs remain
        # unchanged.
        flat_channel_ids = channel_ids.reshape(
            batch_size * num_timepoints, channels
        ).clone()
        patch_tokens, h_in, w_in = self.backbone.patch_embed(
            {"imgs": flat_imgs, "chn_ids": flat_channel_ids}
        )

        cls_for_position = self.backbone.cls_token.to(
            device=patch_tokens.device, dtype=patch_tokens.dtype
        ).expand(batch_size * num_timepoints, -1, -1)
        position_input = torch.cat((cls_for_position, patch_tokens), dim=1)
        spatial_position = self.backbone.interpolate_pos_encoding(
            position_input, w_in, h_in
        )[:, 1:, :]
        patch_tokens = patch_tokens + spatial_position

        num_patches = patch_tokens.shape[1]
        patch_tokens = patch_tokens.reshape(
            batch_size,
            num_timepoints,
            num_patches,
            self.embed_dim,
        )
        patch_tokens = self.adapters[sensor](patch_tokens, timestamps)
        patch_tokens = patch_tokens.reshape(
            batch_size, num_timepoints * num_patches, self.embed_dim
        )

        cls_tokens = self.backbone.cls_token.to(
            device=patch_tokens.device, dtype=patch_tokens.dtype
        ).expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)
        if self.backbone.register_tokens is not None:
            tokens = torch.cat(
                (
                    tokens[:, :1],
                    self.backbone.register_tokens.to(
                        device=tokens.device, dtype=tokens.dtype
                    ).expand(batch_size, -1, -1),
                    tokens[:, 1:],
                ),
                dim=1,
            )

        for block in self.backbone.blocks:
            tokens = block(tokens)
        normalized = self.backbone.norm(tokens)
        patch_start = 1 + self.backbone.num_register_tokens
        return {
            "x_norm_clstoken": normalized[:, 0],
            "x_norm_regtokens": normalized[:, 1:patch_start],
            "x_norm_patchtokens": normalized[:, patch_start:],
            "x_prenorm": tokens,
        }

    def forward(
        self, sensor: str, x_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        features = self.forward_features(sensor, x_dict)
        return self.heads[sensor](features["x_norm_clstoken"])

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone.requires_grad_(bool(trainable))
        if trainable:
            self.backbone.train()
        else:
            self.backbone.eval()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One shared official Panopticon ViT-B with sensor-private temporal "
            "adapters/heads and balanced multi-sensor round-robin training."
        )
    )
    parser.add_argument(
        "--sensors",
        default=",".join(SENSOR_ORDER),
        help=f"Comma-separated subset/order from {SENSOR_ORDER}.",
    )
    parser.add_argument(
        "--cross-sensor-event-policy",
        choices=("error", "purge"),
        default="error",
        help=(
            "Audit union(all sensor train events) against union(all sensor "
            "validation events) before normalization or loading. The default "
            "'error' fails closed; 'purge' removes every validation event from "
            "every training manifest and then requires zero overlap."
        ),
    )
    parser.add_argument(
        "--protocol-manifest-dir",
        type=Path,
        default=None,
        help=(
            "Write-once directory for globally event-purged train manifests. "
            "Defaults to <output-dir>/protocol_manifests. Reuse the same "
            "directory across matched runs."
        ),
    )
    for sensor in SENSOR_ORDER:
        parser.add_argument(
            f"--{sensor}-train-csv",
            type=Path,
            default=DEFAULT_CSVS[sensor]["train"],
        )
        parser.add_argument(
            f"--{sensor}-val-csv",
            type=Path,
            default=DEFAULT_CSVS[sensor]["val"],
        )
        parser.add_argument(
            f"--{sensor}-batch-size",
            type=int,
            default=0,
            help="0 inherits --batch-size.",
        )
        parser.add_argument(
            f"--{sensor}-resize-size",
            type=int,
            default=0,
            help="Optional on-device square resize; 0 preserves current pipeline size.",
        )

    parser.add_argument(
        "--emit-input-mode",
        choices=("raw", "residual"),
        default="raw",
        help=(
            "Matched EMIT source frames are t0, prev1, seasonal. residual emits "
            "t0-prev1 and t0-seasonal after train-only normalization."
        ),
    )
    parser.add_argument("--residual-clip", type=float, default=5.0)
    parser.add_argument("--time-base-year", type=int, default=2000)
    parser.add_argument("--time-embed-dim", type=int, default=384)
    parser.add_argument("--pad-to-multiple", type=int, default=14)

    parser.add_argument(
        "--weights",
        default=str(REPO_ROOT / "weights/panopticon_vitb14_teacher.pth"),
        help='Official backbone state dict; "none" creates a scratch backbone.',
    )
    parser.add_argument(
        "--backbone-mode",
        choices=("frozen", "finetune"),
        default="frozen",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="For finetune mode, keep the backbone frozen for the first N epochs.",
    )
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--temporal-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--class-balanced-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use clipped train-label class weights per sensor.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--scheduler",
        choices=("none", "warmup_cosine"),
        default="warmup_cosine",
    )
    parser.add_argument("--warmup-rounds", type=int, default=100)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--rounds-per-epoch",
        type=int,
        default=0,
        help=(
            "Balanced rounds per epoch; 0 uses the shortest loader so no "
            "sensor is oversampled. Explicit larger values cycle short loaders."
        ),
    )
    parser.add_argument(
        "--max-train-rounds",
        type=int,
        default=0,
        help="Per-epoch cap after resolving --rounds-per-epoch; 0 disables.",
    )
    parser.add_argument(
        "--gradient-accumulation-rounds",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Per-sensor evaluation cap; 0 evaluates the full manifest.",
    )
    parser.add_argument(
        "--smoke-30",
        action="store_true",
        help=(
            "Force frozen backbone, one epoch, <=30 balanced train rounds and "
            "<=2 eval batches per sensor."
        ),
    )

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--cache-mode",
        choices=("off", "sync", "async"),
        default="async",
    )
    parser.add_argument("--cache-prefetch-rows", type=int, default=4)
    parser.add_argument("--cache-async-workers", type=int, default=2)
    parser.add_argument("--cache-min-free-gb", type=float, default=10.0)

    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/train_only_normalization.json.",
    )
    parser.add_argument("--stats-samples", type=int, default=200)
    parser.add_argument("--stats-workers", type=int, default=8)
    parser.add_argument("--stats-seed", type=int, default=20260727)
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument(
        "--emit-stats-source",
        choices=("compute", "verified-script-constant"),
        default="compute",
        help=(
            "The verified constant is the 2048-train-row/all-six-frame value "
            "documented in the current EMIT sensor script. compute is stricter "
            "for changed manifests and uses the selected three source frames."
        ),
    )

    parser.add_argument("--s5p-data-key", default="ch4")
    parser.add_argument("--s5p-channel-last", action="store_true")
    parser.add_argument("--s5p-allow-pickle", action="store_true")
    parser.add_argument("--s5p-channel-id", type=float, default=0.0)

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--device",
        default="auto",
        help='Examples: "auto", "cpu", "cuda", "cuda:0".',
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument("--log-interval-rounds", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Load model weights only from another runner checkpoint.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-sensor",
        choices=SENSOR_ORDER,
        default=None,
        help="Evaluate only this enabled sensor; no other sensor input is loaded.",
    )
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate CSV headers and print the resolved sensor plan; no image I/O.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a tiny synthetic CPU integration test with no real image I/O.",
    )
    return parser.parse_args(argv)


def parse_sensors(value: str) -> Tuple[str, ...]:
    sensors = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not sensors:
        raise ValueError("At least one sensor is required")
    unknown = [sensor for sensor in sensors if sensor not in SENSOR_ORDER]
    duplicates = len(set(sensors)) != len(sensors)
    if unknown or duplicates:
        raise ValueError(
            f"Invalid sensors={sensors}; unknown={unknown}, duplicates={duplicates}"
        )
    return sensors


def resolved_path(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def build_sensor_plans(
    args: argparse.Namespace, sensors: Sequence[str]
) -> Dict[str, SensorPlan]:
    plans: Dict[str, SensorPlan] = {}
    for sensor in sensors:
        train_csv = resolved_path(getattr(args, f"{sensor}_train_csv"))
        val_csv = resolved_path(getattr(args, f"{sensor}_val_csv"))
        resize_size = int(getattr(args, f"{sensor}_resize_size"))
        if resize_size < 0:
            raise ValueError(f"{sensor} resize size must be >=0")

        if sensor == "s2":
            path_columns = ("path_t0",)
            time_columns = ("t0_image_time",)
            input_mode = "current"
            residual_slots: Tuple[str, ...] = ()
            channels = 12
            stats_semantics = "train rows, t0, non-zero pixels, per band"
        elif sensor == "l89":
            path_columns = ("path_t0", "path_prev1", "path_seasonal")
            time_columns = tuple(TIFF_TIME_COLUMNS[column] for column in path_columns)
            input_mode = "raw"
            residual_slots = ()
            channels = 7
            stats_semantics = (
                "train rows, t0/prev1/seasonal, non-zero pixels, per band"
            )
        elif sensor == "s5p":
            path_columns = S5P_TIME_PATH_COLUMNS
            # These are source NC path columns used only to recover timestamps.
            time_columns = S5P_TIME_PATH_COLUMNS
            input_mode = "raw"
            residual_slots = ()
            channels = 1
            stats_semantics = (
                "train NPZ rows, all six frames, finite pixels, one shared scalar"
            )
        elif sensor == "emit":
            path_columns = ("path_t0", "path_prev1", "path_seasonal")
            time_columns = tuple(TIFF_TIME_COLUMNS[column] for column in path_columns)
            input_mode = args.emit_input_mode
            residual_slots = (
                ("path_prev1", "path_seasonal")
                if input_mode == "residual"
                else ()
            )
            channels = 32
            stats_semantics = (
                "train rows, t0/prev1/seasonal, finite non-zero pixels, per band"
            )
        else:
            raise AssertionError(sensor)

        plans[sensor] = SensorPlan(
            name=sensor,
            train_csv=train_csv,
            val_csv=val_csv,
            path_columns=tuple(path_columns),
            time_columns=tuple(time_columns),
            input_mode=input_mode,
            residual_slots=tuple(residual_slots),
            source_timepoints=len(path_columns),
            model_timepoints=output_timepoints(
                input_mode, len(path_columns), residual_slots
            ),
            channels=channels,
            resize_size=resize_size,
            stats_semantics=stats_semantics,
        )
    return plans


def required_csv_columns(sensor: str, plan: SensorPlan) -> Tuple[str, ...]:
    if sensor == "s5p":
        return (
            "label",
            "plume_time",
            "image_path",
            *S5P_TIME_PATH_COLUMNS,
        )
    return ("label", *plan.path_columns, *plan.time_columns)


def validate_csv_headers(plans: Mapping[str, SensorPlan]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for sensor, plan in plans.items():
        sensor_report = {}
        for split, path_text in (("train", plan.train_csv), ("val", plan.val_csv)):
            path = Path(path_text)
            if not path.is_file():
                raise FileNotFoundError(f"{sensor} {split} CSV not found: {path}")
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
            if not header:
                raise ValueError(f"{sensor} {split} CSV has no header: {path}")
            missing = [
                column
                for column in required_csv_columns(sensor, plan)
                if column not in header
            ]
            if missing:
                raise ValueError(
                    f"{sensor} {split} CSV is missing columns {missing}: {path}"
                )
            sensor_report[split] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "required_columns_present": True,
            }
        report[sensor] = sensor_report
    return report


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_dataframe_once_or_verify(frame: pd.DataFrame, path: Path) -> str:
    """Materialize a deterministic protocol CSV without silent replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        candidate_sha = sha256_file(temporary)
        if path.exists():
            existing_sha = sha256_file(path)
            if existing_sha != candidate_sha:
                raise FileExistsError(
                    f"Protocol manifest exists with different content: {path}; "
                    f"existing_sha256={existing_sha}, "
                    f"candidate_sha256={candidate_sha}. Use a new "
                    "--protocol-manifest-dir for the changed protocol."
                )
            temporary.unlink()
            return existing_sha
        os.replace(temporary, path)
        return candidate_sha
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def canonical_event_series(
    sensor: str, frame: pd.DataFrame, path: Path
) -> pd.Series:
    """Map all legacy sensor rows onto one physical methane-event identity."""

    if sensor == "s2" and "event_group_id" in frame.columns:
        values = frame["event_group_id"].astype("string").str.strip()
    else:
        if "plume_id" not in frame.columns:
            raise ValueError(
                f"{path} has no plume_id column; global event isolation cannot run."
            )
        values = (
            frame["plume_id"]
            .astype("string")
            .str.strip()
            .str.replace(EVENT_SUFFIX_RE, "", regex=True)
        )
    invalid = values.isna() | values.eq("")
    if invalid.any():
        raise ValueError(
            f"{path} has {int(invalid.sum())} empty canonical event identifiers."
        )
    return values.astype(str)


def _validate_manifest_labels(frame: pd.DataFrame, path: Path) -> None:
    if "label" not in frame.columns:
        raise ValueError(f"{path} has no label column.")
    numeric = pd.to_numeric(frame["label"], errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{path} has {int(numeric.isna().sum())} nonnumeric labels.")
    observed = set(numeric.unique().tolist())
    if not observed.issubset({0, 1}):
        raise ValueError(f"{path} has non-binary labels: {sorted(observed)}")


def enforce_event_protocol(
    args: argparse.Namespace,
    sensors: Sequence[str],
    plans: Mapping[str, SensorPlan],
) -> Tuple[Dict[str, SensorPlan], Dict[str, Any]]:
    """Fail closed on cross-sensor event leakage or purge it deterministically."""

    frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    events: Dict[str, Dict[str, pd.Series]] = {}
    per_sensor: Dict[str, Dict[str, Any]] = {}
    for sensor in sensors:
        frames[sensor] = {}
        events[sensor] = {}
        per_sensor[sensor] = {}
        plan = plans[sensor]
        for split, path_text in (
            ("train", plan.train_csv),
            ("val", plan.val_csv),
        ):
            path = Path(path_text)
            frame = pd.read_csv(path, low_memory=False)
            if frame.empty:
                raise ValueError(f"Empty {split} manifest: {path}")
            _validate_manifest_labels(frame, path)
            event_values = canonical_event_series(sensor, frame, path)
            frames[sensor][split] = frame
            events[sensor][split] = event_values
            per_sensor[sensor][split] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": int(len(frame)),
                "canonical_events": int(event_values.nunique()),
                "positive_rows": int(pd.to_numeric(frame["label"]).sum()),
            }

        within = set(events[sensor]["train"]) & set(events[sensor]["val"])
        per_sensor[sensor]["within_sensor_train_val_overlap"] = len(within)
        if within:
            raise ValueError(
                f"{sensor} train/val has {len(within)} canonical event "
                f"overlaps; examples={sorted(within)[:10]}"
            )

    validation_union = set().union(
        *(set(events[sensor]["val"]) for sensor in sensors)
    )
    training_union = set().union(
        *(set(events[sensor]["train"]) for sensor in sensors)
    )
    global_overlap = training_union & validation_union
    cross_pairs: Dict[str, int] = {}
    for train_sensor in sensors:
        for val_sensor in sensors:
            if train_sensor == val_sensor:
                continue
            count = len(
                set(events[train_sensor]["train"])
                & set(events[val_sensor]["val"])
            )
            if count:
                cross_pairs[f"{train_sensor}_train->{val_sensor}_val"] = count

    audit: Dict[str, Any] = {
        "schema_version": EVENT_PROTOCOL_SCHEMA_VERSION,
        "canonical_event_rule": CANONICAL_EVENT_RULE,
        "requested_policy": args.cross_sensor_event_policy,
        "global_validation_events": len(validation_union),
        "global_validation_event_sha256": json_fingerprint(
            sorted(validation_union)
        ),
        "global_training_events_before": len(training_union),
        "global_train_val_overlap_before": len(global_overlap),
        "global_train_val_overlap_examples": sorted(global_overlap)[:20],
        "cross_sensor_overlap_pairs": cross_pairs,
        "sensors": per_sensor,
    }
    updated = dict(plans)

    if global_overlap:
        if args.cross_sensor_event_policy == "error":
            raise EventProtocolError(
                "Shared Panopticon protocol leaks canonical events across "
                "sensors: union(train) intersects union(val) by "
                f"{len(global_overlap)} events; pairs={cross_pairs}; "
                f"examples={sorted(global_overlap)[:10]}. Use "
                "--cross-sensor-event-policy purge with one common "
                "--protocol-manifest-dir for every matched run.",
                audit,
            )
        protocol_dir = (
            Path(args.protocol_manifest_dir)
            if args.protocol_manifest_dir is not None
            else Path(args.output_dir) / "protocol_manifests"
        )
        for sensor in sensors:
            keep = ~events[sensor]["train"].isin(validation_union)
            purged = frames[sensor]["train"].loc[keep].copy()
            if purged.empty:
                raise ValueError(f"Global event purge emptied {sensor} training data.")
            remaining_labels = set(
                pd.to_numeric(purged["label"]).astype(int).unique().tolist()
            )
            if remaining_labels != {0, 1}:
                raise ValueError(
                    f"Global event purge leaves {sensor} "
                    f"labels={sorted(remaining_labels)}; classification would "
                    "not be a valid matched comparison."
                )
            output_path = (
                protocol_dir / f"{sensor}_train_global_val_purged.csv"
            ).expanduser().resolve()
            output_sha = write_dataframe_once_or_verify(purged, output_path)
            remaining_events = set(
                canonical_event_series(sensor, purged, output_path)
            )
            post_overlap = remaining_events & validation_union
            if post_overlap:
                raise AssertionError(
                    f"Internal error: {sensor} still overlaps global validation."
                )
            updated[sensor] = replace(
                plans[sensor],
                train_csv=str(output_path),
            )
            audit["sensors"][sensor]["purged_train"] = {
                "path": str(output_path),
                "sha256": output_sha,
                "rows": int(len(purged)),
                "canonical_events": len(remaining_events),
                "removed_rows": int((~keep).sum()),
                "removed_events": len(
                    set(events[sensor]["train"]) & validation_union
                ),
                "positive_rows": int(pd.to_numeric(purged["label"]).sum()),
            }
        audit["purge_applied"] = True
    else:
        audit["purge_applied"] = False

    post_training_union: set[str] = set()
    for sensor in sensors:
        if audit["purge_applied"]:
            train_path = Path(updated[sensor].train_csv)
            post_frame = pd.read_csv(train_path, low_memory=False)
            post_training_union.update(
                canonical_event_series(sensor, post_frame, train_path)
            )
        else:
            post_training_union.update(events[sensor]["train"])
    post_overlap = post_training_union & validation_union
    audit["global_training_events_after"] = len(post_training_union)
    audit["global_train_val_overlap_after"] = len(post_overlap)
    if post_overlap:
        raise AssertionError(
            f"Shared protocol still has {len(post_overlap)} global event overlaps."
        )
    audit["fingerprint"] = json_fingerprint(
        {
            "canonical_event_rule": CANONICAL_EVENT_RULE,
            "manifest_sha256": {
                sensor: {
                    "train": sha256_file(Path(updated[sensor].train_csv)),
                    "val": sha256_file(Path(updated[sensor].val_csv)),
                }
                for sensor in sensors
            },
            "global_validation_event_sha256": audit[
                "global_validation_event_sha256"
            ],
            "global_train_val_overlap_after": len(post_overlap),
        }
    )
    print(
        "[Protocol] "
        f"global_train_val_overlap={len(global_overlap)}->{len(post_overlap)} "
        f"purge={audit['purge_applied']} pairs={cross_pairs}",
        flush=True,
    )
    return updated, audit


def csv_identity(path_text: str) -> Dict[str, Any]:
    path = Path(path_text)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def stats_fingerprint(plan: SensorPlan, args: argparse.Namespace) -> str:
    payload = {
        "sensor": plan.name,
        "train_csv": csv_identity(plan.train_csv),
        "path_columns": plan.path_columns,
        "stats_samples": int(args.stats_samples),
        "stats_seed": int(args.stats_seed),
        "semantics": plan.stats_semantics,
        "s5p_data_key": args.s5p_data_key if plan.name == "s5p" else None,
        "s5p_channel_last": (
            bool(args.s5p_channel_last) if plan.name == "s5p" else None
        ),
        "emit_stats_source": (
            args.emit_stats_source if plan.name == "emit" else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_cache_objects(
    args: argparse.Namespace, sensors: Sequence[str]
) -> Dict[str, Optional[Any]]:
    if args.cache_mode == "off":
        return {sensor: None for sensor in sensors}
    caches: Dict[str, Optional[Any]] = {}
    for sensor in sensors:
        # L89's implementation includes a per-file lock and is otherwise
        # format-agnostic, so one implementation safely serves TIFF and NPZ.
        delegate = l89_module.StaticAnchoredCache(
            str(args.cache_root / sensor),
            min_free_gb=args.cache_min_free_gb,
            async_mode=args.cache_mode == "async",
            max_workers=args.cache_async_workers,
            prefetch_on_miss=True,
        )
        caches[sensor] = LocalAwareCache(
            delegate,
            local_roots=(Path("/diniuvol/yuyao"),),
        )
    return caches


def compute_emit_train_stats(
    plan: SensorPlan,
    *,
    n_samples: int,
    seed: int,
    num_workers: int,
    local_file_cache: Optional[Any],
) -> Tuple[Sequence[float], Sequence[float]]:
    base = emit_module.Emit32CsvDataset(
        csv_path=plan.train_csv,
        path_column=plan.path_columns[0],
        normalize_stats=None,
        scale_to_unit=False,
        compute_stats=False,
        pad_to_multiple=None,
        skip_invalid_samples=False,
        path_columns_for_validation=plan.path_columns,
        local_file_cache=local_file_cache,
    )
    if len(base) == 0:
        raise ValueError(f"Cannot compute EMIT stats from empty CSV: {plan.train_csv}")
    count = min(max(1, int(n_samples)), len(base))
    rng = random.Random(seed)
    row_indices = rng.sample(range(len(base)), count)
    jobs = [
        (row_index, column)
        for row_index in row_indices
        for column in plan.path_columns
    ]
    sums = torch.zeros(plan.channels, dtype=torch.float64)
    squared_sums = torch.zeros(plan.channels, dtype=torch.float64)
    counts = torch.zeros(plan.channels, dtype=torch.float64)

    def image_moments(job: Tuple[int, str]):
        row_index, column = job
        row = base.df.iloc[row_index]
        sample_id = (
            row[base.id_column] if base.id_column in row else row_index
        )
        image = base._load_image(
            row[column], column_name=column, sample_id=sample_id
        ).to(dtype=torch.float64)
        valid = torch.isfinite(image) & (image != 0)
        valid_float = valid.to(dtype=torch.float64)
        return (
            (torch.nan_to_num(image) * valid_float).sum(dim=(1, 2)),
            (torch.nan_to_num(image).square() * valid_float).sum(dim=(1, 2)),
            valid.sum(dim=(1, 2)).to(dtype=torch.float64),
        )

    def accumulate(result) -> None:
        image_sum, image_squared_sum, image_count = result
        sums.add_(image_sum)
        squared_sums.add_(image_squared_sum)
        counts.add_(image_count)

    workers = max(1, int(num_workers))
    max_in_flight = max(workers, workers * 4)
    if workers == 1:
        for job_index, job in enumerate(jobs, 1):
            accumulate(image_moments(job))
            if job_index % 100 == 0 or job_index == len(jobs):
                print(f"[Stats] emit TIFFs {job_index}/{len(jobs)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            iterator = iter(jobs)
            in_flight = set()

            def submit_until_full() -> None:
                while len(in_flight) < max_in_flight:
                    try:
                        job = next(iterator)
                    except StopIteration:
                        break
                    in_flight.add(pool.submit(image_moments, job))

            submit_until_full()
            completed = 0
            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    completed += 1
                    accumulate(future.result())
                    if completed % 100 == 0 or completed == len(jobs):
                        print(
                            f"[Stats] emit TIFFs {completed}/{len(jobs)}",
                            flush=True,
                        )
                submit_until_full()

    missing = counts == 0
    if missing.any():
        indices = torch.nonzero(missing, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"EMIT stats found no valid pixels in bands {indices}")
    means = sums / counts
    variances = squared_sums / counts - means.square()
    standard_deviations = torch.sqrt(torch.clamp(variances, min=1e-12))
    return means.float().tolist(), standard_deviations.float().tolist()


def compute_one_sensor_stats(
    sensor: str,
    plan: SensorPlan,
    args: argparse.Namespace,
    cache: Optional[Any],
) -> Dict[str, Any]:
    start = time.monotonic()
    if sensor == "s2":
        mean, std = s2_module.compute_s2_temporal_stats(
            plan.train_csv,
            path_columns=plan.path_columns,
            n_samples=args.stats_samples,
            seed=args.stats_seed,
            num_workers=args.stats_workers,
            local_file_cache=cache,
            resize_size=plan.resize_size,
        )
        source = "computed_from_selected_train_manifest"
    elif sensor == "l89":
        mean, std = l89_module.compute_l89_temporal_stats(
            plan.train_csv,
            path_columns=plan.path_columns,
            n_samples=args.stats_samples,
            seed=args.stats_seed,
            num_workers=args.stats_workers,
            local_file_cache=cache,
        )
        source = "computed_from_selected_train_manifest"
    elif sensor == "s5p":
        mean, std = s5p_module.compute_s5p_temporal_stats(
            plan.train_csv,
            path_column="image_path",
            n_samples=args.stats_samples,
            seed=args.stats_seed,
            num_workers=args.stats_workers,
            local_file_cache=cache,
            data_key=args.s5p_data_key,
            channel_last=args.s5p_channel_last,
            allow_pickle=args.s5p_allow_pickle,
            num_timepoints=6,
        )
        source = "computed_from_selected_train_manifest"
    elif sensor == "emit":
        if args.emit_stats_source == "verified-script-constant":
            mean, std = emit_module.PRECOMPUTED_STATS
            source = (
                "verified current EMIT script constant: deterministic 2048 "
                "train rows x six frames"
            )
        else:
            mean, std = compute_emit_train_stats(
                plan,
                n_samples=args.stats_samples,
                seed=args.stats_seed,
                num_workers=args.stats_workers,
                local_file_cache=cache,
            )
            source = "computed_from_selected_train_manifest"
    else:
        raise AssertionError(sensor)

    if len(mean) != plan.channels or len(std) != plan.channels:
        raise ValueError(
            f"{sensor} normalization length mismatch: "
            f"mean={len(mean)} std={len(std)} channels={plan.channels}"
        )
    if any(not math.isfinite(float(item)) for item in (*mean, *std)):
        raise ValueError(f"{sensor} normalization contains non-finite values")
    if any(float(item) <= 0 for item in std):
        raise ValueError(f"{sensor} normalization contains non-positive std")
    return {
        "sensor": sensor,
        "fingerprint": stats_fingerprint(plan, args),
        "source": source,
        "train_csv": csv_identity(plan.train_csv),
        "path_columns": list(plan.path_columns),
        "stats_samples_requested": int(args.stats_samples),
        "stats_seed": int(args.stats_seed),
        "semantics": plan.stats_semantics,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "elapsed_seconds": float(time.monotonic() - start),
    }


def load_or_compute_normalization(
    plans: Mapping[str, SensorPlan],
    args: argparse.Namespace,
    caches: Mapping[str, Optional[Any]],
    *,
    checkpoint_records: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    if checkpoint_records is not None:
        records: Dict[str, Dict[str, Any]] = {}
        for sensor in plans:
            if sensor not in checkpoint_records:
                raise ValueError(
                    f"Checkpoint has no normalization record for {sensor}"
                )
            record = dict(checkpoint_records[sensor])
            if len(record.get("mean", [])) != plans[sensor].channels:
                raise ValueError(
                    f"Checkpoint normalization has wrong channel count for {sensor}"
                )
            records[sensor] = record
        return records

    stats_path = args.stats_json or (
        args.output_dir / "train_only_normalization.json"
    )
    existing: Dict[str, Any] = {}
    if stats_path.is_file() and not args.recompute_stats:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        existing = dict(payload.get("sensors", {}))

    records: Dict[str, Dict[str, Any]] = {}
    for sensor, plan in plans.items():
        expected = stats_fingerprint(plan, args)
        cached = existing.get(sensor)
        if (
            cached is not None
            and cached.get("fingerprint") == expected
            and len(cached.get("mean", [])) == plan.channels
            and len(cached.get("std", [])) == plan.channels
        ):
            print(f"[Stats] {sensor}: reusing {stats_path}", flush=True)
            records[sensor] = dict(cached)
            continue
        print(
            f"[Stats] {sensor}: computing strictly from train CSV "
            f"({plan.stats_semantics})",
            flush=True,
        )
        records[sensor] = compute_one_sensor_stats(
            sensor, plan, args, caches[sensor]
        )
        atomic_json_dump(
            {
                "schema_version": 1,
                "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "sensors": records,
            },
            stats_path,
        )
    return records


def build_one_dataset(
    sensor: str,
    plan: SensorPlan,
    csv_path: str,
    normalization: Mapping[str, Any],
    args: argparse.Namespace,
    cache: Optional[Any],
) -> StrictIndexedDataset:
    stats = (normalization["mean"], normalization["std"])
    if sensor == "s2":
        dataset = s2_module.S2TemporalSequenceDataset(
            csv_path=csv_path,
            path_columns=plan.path_columns,
            time_columns=plan.time_columns,
            normalize_stats=stats,
            time_base_year=args.time_base_year,
            pad_to_multiple=(
                None if plan.resize_size > 0 else args.pad_to_multiple
            ),
            skip_invalid_samples=False,
            local_file_cache=cache,
            cache_prefetch_rows=args.cache_prefetch_rows,
            channel_indices=None,
            input_mode=plan.input_mode,
            residual_slots=(),
            residual_clip=args.residual_clip,
        )
    elif sensor == "l89":
        dataset = l89_module.L89TemporalSequenceDataset(
            csv_path=csv_path,
            path_columns=plan.path_columns,
            time_columns=plan.time_columns,
            normalize_stats=stats,
            band_indices=tuple(range(7)),
            time_base_year=args.time_base_year,
            pad_to_multiple=args.pad_to_multiple,
            skip_invalid_samples=False,
            local_file_cache=cache,
            cache_prefetch_rows=args.cache_prefetch_rows,
            input_mode="raw",
            residual_slots=(),
            residual_clip=args.residual_clip,
        )
    elif sensor == "s5p":
        dataset = s5p_module.S5PTemporalSequenceDataset(
            csv_path=csv_path,
            path_column="image_path",
            time_path_columns=S5P_TIME_PATH_COLUMNS,
            normalize_stats=stats,
            time_base_year=args.time_base_year,
            pad_to_multiple=args.pad_to_multiple,
            pad_value=0.0,
            skip_invalid_samples=False,
            local_file_cache=cache,
            cache_prefetch_rows=args.cache_prefetch_rows,
            data_key=args.s5p_data_key,
            channel_last=args.s5p_channel_last,
            allow_pickle=args.s5p_allow_pickle,
            nan_to_num=0.0,
            chn_id_value=args.s5p_channel_id,
            max_retries=1,
            input_mode="raw",
            residual_slots=(),
            residual_clip=args.residual_clip,
        )
    elif sensor == "emit":
        dataset = emit_module.Emit32TemporalSequenceDataset(
            csv_path=csv_path,
            path_columns=plan.path_columns,
            time_columns=plan.time_columns,
            normalize_stats=stats,
            time_base_year=args.time_base_year,
            pad_to_multiple=args.pad_to_multiple,
            skip_invalid_samples=False,
            skip_path_validation=True,
            local_file_cache=cache,
            cache_prefetch_rows=args.cache_prefetch_rows,
            input_mode=plan.input_mode,
            residual_slots=plan.residual_slots,
            residual_clip=args.residual_clip,
        )
    else:
        raise AssertionError(sensor)
    return StrictIndexedDataset(sensor, dataset)


def sensor_batch_size(args: argparse.Namespace, sensor: str) -> int:
    override = int(getattr(args, f"{sensor}_batch_size"))
    value = override if override > 0 else int(args.batch_size)
    if value <= 0:
        raise ValueError(f"{sensor} batch size must be positive")
    return value


def build_loaders(
    plans: Mapping[str, SensorPlan],
    normalization: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    caches: Mapping[str, Optional[Any]],
) -> Tuple[
    Dict[str, StrictIndexedDataset],
    Dict[str, StrictIndexedDataset],
    Dict[str, DataLoader],
    Dict[str, DataLoader],
]:
    train_datasets: Dict[str, StrictIndexedDataset] = {}
    val_datasets: Dict[str, StrictIndexedDataset] = {}
    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}
    for sensor_index, (sensor, plan) in enumerate(plans.items()):
        train_dataset = build_one_dataset(
            sensor,
            plan,
            plan.train_csv,
            normalization[sensor],
            args,
            caches[sensor],
        )
        val_dataset = build_one_dataset(
            sensor,
            plan,
            plan.val_csv,
            normalization[sensor],
            args,
            caches[sensor],
        )
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError(
                f"{sensor}: empty dataset train={len(train_dataset)} val={len(val_dataset)}"
            )
        batch_size = sensor_batch_size(args, sensor)
        loader_kwargs: Dict[str, Any] = {
            "num_workers": int(args.num_workers),
            "pin_memory": bool(args.pin_memory),
            "worker_init_fn": seed_worker,
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
            loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

        train_generator = torch.Generator()
        train_generator.manual_seed(args.seed + 1009 * (sensor_index + 1))
        val_generator = torch.Generator()
        val_generator.manual_seed(args.seed + 2003 * (sensor_index + 1))
        train_loaders[sensor] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=train_generator,
            **loader_kwargs,
        )
        val_loaders[sensor] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            generator=val_generator,
            **loader_kwargs,
        )
        train_datasets[sensor] = train_dataset
        val_datasets[sensor] = val_dataset
        labels = train_dataset.labels
        print(
            f"[Data] {sensor}: train={len(train_dataset)} val={len(val_dataset)} "
            f"positive_rate={float(labels.mean()):.6f} batch={batch_size} "
            f"T={plan.model_timepoints} C={plan.channels}",
            flush=True,
        )
    return train_datasets, val_datasets, train_loaders, val_loaders


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def resize_temporal_batch(
    x_dict: MutableMapping[str, torch.Tensor], size: int
) -> MutableMapping[str, torch.Tensor]:
    if int(size) <= 0:
        return x_dict
    images = x_dict["imgs"]
    if tuple(images.shape[-2:]) == (int(size), int(size)):
        return x_dict
    batch_size, timepoints, channels, height, width = images.shape
    flat = images.reshape(batch_size * timepoints, channels, height, width)
    flat = F.interpolate(
        flat,
        size=(int(size), int(size)),
        mode="bilinear",
        align_corners=True,
    )
    result = dict(x_dict)
    result["imgs"] = flat.reshape(
        batch_size, timepoints, channels, int(size), int(size)
    )
    return result


def build_backbone(weights: str) -> Tuple[nn.Module, Dict[str, Any]]:
    from src.backbones import build_panopticon_vitb14

    backbone = build_panopticon_vitb14()
    provenance: Dict[str, Any] = {
        "architecture": "official Panopticon ViT-B/14",
        "weights": weights,
    }
    if weights.lower() in {"none", "scratch", "random", ""}:
        provenance["source"] = "random_initialization"
        return backbone, provenance
    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Panopticon checkpoint not found: {path}")
    print(f"[Model] loading official backbone weights: {path}", flush=True)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, Mapping) and "backbone" in state:
        state = state["backbone"]
    result = backbone.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Strict Panopticon load was not exact: {result}"
        )
    provenance.update(
        {
            "source": "checkpoint",
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "strict": True,
        }
    )
    return backbone, provenance


def parameter_counts(
    model: SharedPanopticonTemporalClassifier,
) -> Dict[str, Any]:
    def count(module: nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    return {
        "total": count(model),
        "backbone": count(model.backbone),
        "adapters_total": count(model.adapters),
        "heads_total": count(model.heads),
        "per_sensor": {
            sensor: {
                "adapter": count(model.adapters[sensor]),
                "head": count(model.heads[sensor]),
            }
            for sensor in model.adapters
        },
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def build_losses(
    datasets: Mapping[str, StrictIndexedDataset],
    *,
    class_balanced: bool,
    label_smoothing: float,
    device: torch.device,
) -> Dict[str, nn.Module]:
    losses: Dict[str, nn.Module] = {}
    for sensor, dataset in datasets.items():
        weights = None
        if class_balanced:
            labels = dataset.labels
            positives = float((labels == 1).sum())
            negatives = float((labels == 0).sum())
            positive_weight = (
                min(max(negatives / positives, 0.25), 4.0)
                if positives > 0
                else 1.0
            )
            weights = torch.tensor(
                [1.0, positive_weight], dtype=torch.float32, device=device
            )
            print(
                f"[Loss] {sensor}: positive class weight={positive_weight:.6f}",
                flush=True,
            )
        losses[sensor] = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=float(label_smoothing),
        )
    return losses


def build_optimizer(
    model: SharedPanopticonTemporalClassifier,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    groups = [
        {
            "name": "temporal_adapters",
            "params": list(model.adapters.parameters()),
            "lr": float(args.temporal_lr),
        },
        {
            "name": "sensor_heads",
            "params": list(model.heads.parameters()),
            "lr": float(args.head_lr),
        },
    ]
    if args.backbone_mode == "finetune":
        groups.insert(
            0,
            {
                "name": "shared_backbone",
                "params": list(model.backbone.parameters()),
                "lr": float(args.backbone_lr),
            },
        )
    return torch.optim.AdamW(
        groups,
        weight_decay=float(args.weight_decay),
        betas=(float(args.adam_beta1), 0.999),
    )


def resolve_rounds(
    loaders: Mapping[str, DataLoader], args: argparse.Namespace
) -> int:
    if any(len(loader) == 0 for loader in loaders.values()):
        raise ValueError("Every enabled train loader must contain at least one batch")
    rounds = (
        int(args.rounds_per_epoch)
        if int(args.rounds_per_epoch) > 0
        else min(len(loader) for loader in loaders.values())
    )
    if int(args.max_train_rounds) > 0:
        rounds = min(rounds, int(args.max_train_rounds))
    if rounds <= 0:
        raise ValueError("Resolved train rounds must be positive")
    return rounds


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    mode: str,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    if mode == "none":
        return None
    warmup_steps = min(max(0, int(warmup_steps)), max(0, total_steps - 1))

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denominator = max(1, total_steps - warmup_steps)
        progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def amp_context(
    device: torch.device, enabled: bool, dtype_name: str
):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels_bool = labels.astype(bool)
    predictions_bool = predictions.astype(bool)
    tp = float(np.logical_and(labels_bool, predictions_bool).sum())
    fp = float(np.logical_and(~labels_bool, predictions_bool).sum())
    fn = float(np.logical_and(labels_bool, ~predictions_bool).sum())
    denominator = 2.0 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def classification_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    loss: Optional[float] = None,
) -> Dict[str, Any]:
    label_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if label_array.size == 0:
        return {
            "samples": 0,
            "loss": loss,
            "positive_rate": None,
            "accuracy_0p5": None,
            "f1_0p5": None,
            "best_f1": None,
            "best_f1_threshold": None,
            "ap": None,
            "auroc": None,
        }
    predictions = probability_array >= 0.5
    thresholds = np.unique(
        np.concatenate(
            (np.asarray([0.0, 0.5, 1.0]), probability_array)
        )
    )
    f1_values = np.asarray(
        [
            _binary_f1(label_array, probability_array >= threshold)
            for threshold in thresholds
        ]
    )
    best_index = int(np.argmax(f1_values))
    one_class = np.unique(label_array).size < 2
    return {
        "samples": int(label_array.size),
        "loss": None if loss is None else float(loss),
        "positive_rate": float(label_array.mean()),
        "accuracy_0p5": float((predictions == label_array).mean()),
        "f1_0p5": float(_binary_f1(label_array, predictions)),
        "best_f1": float(f1_values[best_index]),
        "best_f1_threshold": float(thresholds[best_index]),
        "ap": (
            None
            if one_class
            else float(average_precision_score(label_array, probability_array))
        ),
        "auroc": (
            None
            if one_class
            else float(roc_auc_score(label_array, probability_array))
        ),
    }


def macro_metrics(sensor_metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in (
        "loss",
        "accuracy_0p5",
        "f1_0p5",
        "best_f1",
        "ap",
        "auroc",
    ):
        values = [
            float(metrics[key])
            for metrics in sensor_metrics.values()
            if metrics.get(key) is not None
            and math.isfinite(float(metrics[key]))
        ]
        result[key] = None if not values else float(np.mean(values))
    return result


def cycle_next(
    sensor: str,
    loader: DataLoader,
    iterators: MutableMapping[str, Iterator[Any]],
    cycles: MutableMapping[str, int],
):
    try:
        return next(iterators[sensor])
    except StopIteration:
        cycles[sensor] += 1
        iterators[sensor] = iter(loader)
        return next(iterators[sensor])


def train_one_epoch(
    model: SharedPanopticonTemporalClassifier,
    plans: Mapping[str, SensorPlan],
    loaders: Mapping[str, DataLoader],
    losses: Mapping[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    scaler: torch.cuda.amp.GradScaler,
    *,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    global_round: int,
) -> Tuple[Dict[str, Any], int]:
    model.train()
    train_backbone = (
        args.backbone_mode == "finetune"
        and epoch > int(args.freeze_backbone_epochs)
    )
    model.set_backbone_trainable(train_backbone)

    sensors = tuple(loaders.keys())
    rounds = resolve_rounds(loaders, args)
    iterators = {sensor: iter(loader) for sensor, loader in loaders.items()}
    cycles = {sensor: 0 for sensor in sensors}
    accum_rounds = max(1, int(args.gradient_accumulation_rounds))
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    totals = {
        sensor: {"loss_sum": 0.0, "count": 0, "labels": [], "probabilities": []}
        for sensor in sensors
    }
    optimizer.zero_grad(set_to_none=True)

    for round_index in range(rounds):
        group_position = round_index % accum_rounds
        group_size = min(accum_rounds, rounds - (round_index - group_position))
        rotated = tuple(
            sensors[(index + epoch + round_index - 1) % len(sensors)]
            for index in range(len(sensors))
        )
        for sensor in rotated:
            x_dict, labels = cycle_next(
                sensor, loaders[sensor], iterators, cycles
            )
            labels = labels.to(device, non_blocking=True).long()
            x_dict = move_to_device(x_dict, device)
            x_dict = resize_temporal_batch(
                x_dict, plans[sensor].resize_size
            )
            with amp_context(device, args.amp, args.amp_dtype):
                logits = model(sensor, x_dict)
                unscaled_loss = losses[sensor](logits, labels)
                scaled_loss = unscaled_loss / (
                    len(sensors) * group_size
                )
            scaler.scale(scaled_loss).backward()

            probabilities = torch.softmax(logits.detach().float(), dim=1)[:, 1]
            batch_count = int(labels.numel())
            totals[sensor]["loss_sum"] += float(unscaled_loss.detach()) * batch_count
            totals[sensor]["count"] += batch_count
            totals[sensor]["labels"].extend(labels.detach().cpu().tolist())
            totals[sensor]["probabilities"].extend(
                probabilities.cpu().tolist()
            )

        should_step = (
            (round_index + 1) % accum_rounds == 0
            or round_index + 1 == rounds
        )
        if should_step:
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, float(args.max_grad_norm)
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        global_round += 1
        if (
            args.log_interval_rounds > 0
            and (
                (round_index + 1) % args.log_interval_rounds == 0
                or round_index + 1 == rounds
            )
        ):
            running = {
                sensor: (
                    totals[sensor]["loss_sum"] / totals[sensor]["count"]
                    if totals[sensor]["count"]
                    else None
                )
                for sensor in sensors
            }
            print(
                f"[Train] epoch={epoch} round={round_index + 1}/{rounds} "
                f"loss={running}",
                flush=True,
            )

    sensor_metrics = {}
    for sensor in sensors:
        total = totals[sensor]
        mean_loss = total["loss_sum"] / total["count"]
        sensor_metrics[sensor] = classification_metrics(
            total["labels"], total["probabilities"], loss=mean_loss
        )
        sensor_metrics[sensor]["loader_cycles"] = int(cycles[sensor])
        sensor_metrics[sensor]["batches"] = int(rounds)
    return {
        "sensors": sensor_metrics,
        "macro": macro_metrics(sensor_metrics),
        "balanced_rounds": int(rounds),
        "backbone_trainable": bool(train_backbone),
    }, global_round


@torch.no_grad()
def evaluate(
    model: SharedPanopticonTemporalClassifier,
    plans: Mapping[str, SensorPlan],
    loaders: Mapping[str, DataLoader],
    losses: Mapping[str, nn.Module],
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model.eval()
    model.backbone.eval()
    sensor_metrics: Dict[str, Any] = {}
    for sensor, loader in loaders.items():
        labels_all: list[int] = []
        probabilities_all: list[float] = []
        loss_sum = 0.0
        count = 0
        batches = 0
        for batch_index, (x_dict, labels) in enumerate(loader, 1):
            labels = labels.to(device, non_blocking=True).long()
            x_dict = move_to_device(x_dict, device)
            x_dict = resize_temporal_batch(
                x_dict, plans[sensor].resize_size
            )
            with amp_context(device, args.amp, args.amp_dtype):
                logits = model(sensor, x_dict)
                loss = losses[sensor](logits, labels)
            probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
            batch_count = int(labels.numel())
            loss_sum += float(loss) * batch_count
            count += batch_count
            labels_all.extend(labels.cpu().tolist())
            probabilities_all.extend(probabilities.cpu().tolist())
            batches = batch_index
            if (
                int(args.max_eval_batches) > 0
                and batch_index >= int(args.max_eval_batches)
            ):
                break
        sensor_metrics[sensor] = classification_metrics(
            labels_all,
            probabilities_all,
            loss=(loss_sum / count if count else None),
        )
        sensor_metrics[sensor]["batches"] = int(batches)
        print(
            f"[Eval] {sensor}: {sensor_metrics[sensor]}",
            flush=True,
        )
    return {
        "sensors": sensor_metrics,
        "macro": macro_metrics(sensor_metrics),
    }


def checkpoint_payload(
    *,
    model: SharedPanopticonTemporalClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_round: int,
    history: Sequence[Mapping[str, Any]],
    best_macro_ap: float,
    args: argparse.Namespace,
    plans: Mapping[str, SensorPlan],
    normalization: Mapping[str, Mapping[str, Any]],
    backbone_provenance: Mapping[str, Any],
    counts: Mapping[str, Any],
    event_protocol: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "runner": "multisensor_panopticon_runner.py",
        "epoch": int(epoch),
        "global_round": int(global_round),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "history": list(history),
        "best_macro_ap": float(best_macro_ap),
        "args": json_safe(vars(args)),
        "sensor_plans": {
            sensor: asdict(plan) for sensor, plan in plans.items()
        },
        "normalization": dict(normalization),
        "backbone_provenance": dict(backbone_provenance),
        "parameter_counts": dict(counts),
        "event_protocol": (
            None if event_protocol is None else dict(event_protocol)
        ),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
    }


def load_model_state_for_enabled_sensors(
    model: SharedPanopticonTemporalClassifier,
    saved_state: Mapping[str, torch.Tensor],
) -> None:
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in saved_state.items()
        if key in current
    }
    missing = [key for key in current if key not in filtered]
    shape_mismatch = [
        (key, tuple(filtered[key].shape), tuple(current[key].shape))
        for key in filtered
        if tuple(filtered[key].shape) != tuple(current[key].shape)
    ]
    if missing or shape_mismatch:
        raise RuntimeError(
            f"Checkpoint cannot initialize enabled sensors: "
            f"missing={missing[:10]} shape_mismatch={shape_mismatch[:10]}"
        )
    result = model.load_state_dict(filtered, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Unexpected subset load result: {result}")


def validate_exact_training_resume_plans(
    plans: Mapping[str, SensorPlan],
    checkpoint: Mapping[str, Any],
) -> None:
    """Reject a nominal resume that silently changes data or representation."""

    saved_plans = checkpoint.get("sensor_plans")
    if not isinstance(saved_plans, Mapping):
        raise ValueError("Resume checkpoint has no sensor_plans metadata")
    if tuple(saved_plans.keys()) != tuple(plans.keys()):
        raise ValueError(
            "Training resume requires the same sensor order: "
            f"saved={tuple(saved_plans.keys())}, current={tuple(plans.keys())}"
        )
    mismatches: Dict[str, Dict[str, Any]] = {}
    for sensor, current_plan in plans.items():
        current = json_safe(asdict(current_plan))
        saved = json_safe(saved_plans[sensor])
        differing = {
            key: {"saved": saved.get(key), "current": current.get(key)}
            for key in current
            if saved.get(key) != current.get(key)
        }
        if differing:
            mismatches[sensor] = differing
    if mismatches:
        raise ValueError(
            "Training resume changed a manifest/input plan. Use "
            "--init-checkpoint for an intentional new run instead. "
            f"Mismatches={mismatches}"
        )


def validate_checkpoint_event_protocol(
    current: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    enabled_sensors: Sequence[str],
    exact: bool,
) -> None:
    """Reject transfer/evaluation from a leaked or unaudited runner checkpoint."""

    saved = checkpoint.get("event_protocol")
    if not isinstance(saved, Mapping):
        raise ValueError(
            "Runner checkpoint has no global event-protocol audit. It predates "
            "the leakage fix and is diagnostic-only; do not resume, transfer, "
            "or evaluate it."
        )
    if int(saved.get("global_train_val_overlap_after", -1)) != 0:
        raise ValueError(
            "Runner checkpoint is not globally event-disjoint: "
            f"global_train_val_overlap_after="
            f"{saved.get('global_train_val_overlap_after')}"
        )
    if int(current.get("global_train_val_overlap_after", -1)) != 0:
        raise AssertionError("Current global event audit is not zero-overlap.")

    if exact:
        if saved.get("fingerprint") != current.get("fingerprint"):
            raise ValueError(
                "Checkpoint and current run use different global event "
                "protocols. Reuse the same purged manifests for matched "
                "resume/transfer. "
                f"saved={saved.get('fingerprint')} "
                f"current={current.get('fingerprint')}"
            )
        return

    saved_sensors = saved.get("sensors")
    current_sensors = current.get("sensors")
    if not isinstance(saved_sensors, Mapping) or not isinstance(
        current_sensors, Mapping
    ):
        raise ValueError("Checkpoint event audit has no per-sensor manifest hashes.")
    mismatches: Dict[str, Any] = {}
    for sensor in enabled_sensors:
        saved_sensor = saved_sensors.get(sensor)
        current_sensor = current_sensors.get(sensor)
        if not isinstance(saved_sensor, Mapping) or not isinstance(
            current_sensor, Mapping
        ):
            mismatches[sensor] = "missing sensor audit"
            continue
        saved_val = saved_sensor.get("val")
        current_val = current_sensor.get("val")
        if not isinstance(saved_val, Mapping) or not isinstance(
            current_val, Mapping
        ):
            mismatches[sensor] = "missing validation audit"
            continue
        if saved_val.get("sha256") != current_val.get("sha256"):
            mismatches[sensor] = {
                "saved_val_sha256": saved_val.get("sha256"),
                "current_val_sha256": current_val.get("sha256"),
            }
    if mismatches:
        raise ValueError(
            "Evaluation validation manifests differ from the checkpoint's "
            f"zero-overlap protocol: {mismatches}"
        )


def restore_rng(payload: Mapping[str, Any]) -> None:
    rng = payload.get("rng")
    if not isinstance(rng, Mapping):
        return
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])


class TinyPatchEmbed(nn.Module):
    """Channel-count-agnostic stand-in used only by ``--self-test``."""

    def __init__(self, embed_dim: int, patch_size: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Linear(2, embed_dim)

    def forward(self, x_dict: Mapping[str, torch.Tensor]):
        images = x_dict["imgs"]
        channel_ids = x_dict["chn_ids"].float()
        _, _, height, width = images.shape
        pooled = F.avg_pool2d(
            images.mean(dim=1, keepdim=True),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        spatial = pooled.flatten(2).transpose(1, 2)
        wavelength = torch.sin(channel_ids / 1000.0).mean(
            dim=1, keepdim=True
        )
        wavelength = wavelength.unsqueeze(1).expand(
            -1, spatial.shape[1], -1
        )
        features = torch.cat((spatial, wavelength), dim=-1)
        return self.proj(features), height, width


class TinyBackbone(nn.Module):
    """Minimal API-compatible backbone for integration testing."""

    def __init__(self, embed_dim: int = 48):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_register_tokens = 0
        self.patch_embed = TinyPatchEmbed(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = None
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=4,
                    dim_feedforward=embed_dim * 2,
                    batch_first=True,
                    dropout=0.0,
                )
                for _ in range(2)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def interpolate_pos_encoding(
        self, x: torch.Tensor, width: int, height: int
    ) -> torch.Tensor:
        del width, height
        return torch.zeros(
            1, x.shape[1], self.embed_dim, device=x.device, dtype=x.dtype
        )


class SyntheticSensorDataset(Dataset):
    def __init__(
        self,
        *,
        sensor: str,
        timepoints: int,
        channels: int,
        length: int,
        seed: int,
    ):
        generator = torch.Generator().manual_seed(seed)
        self.sensor = sensor
        self.images = torch.randn(
            length,
            timepoints,
            channels,
            14,
            14,
            generator=generator,
        )
        wavelengths = torch.linspace(400.0, 2400.0, channels)
        self.channel_ids = wavelengths.view(1, 1, channels).expand(
            length, timepoints, channels
        )
        timestamps = torch.tensor([26.0, 6.0, 12.0])
        self.timestamps = timestamps.view(1, 1, 3).expand(
            length, timepoints, 3
        ).clone()
        self.timestamps[:, :, 1] = torch.arange(timepoints).view(1, -1) % 12
        self._labels = np.asarray(
            [index % 2 for index in range(length)], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int):
        return {
            "imgs": self.images[index],
            "chn_ids": self.channel_ids[index],
            "timestamps": self.timestamps[index],
        }, int(self._labels[index])

    @property
    def labels(self) -> np.ndarray:
        return self._labels.copy()


class FailingSyntheticBase(Dataset):
    def __init__(self):
        self.calls: list[int] = []
        self.df = pd.DataFrame({"id": ["bad", "good"], "label": [0, 1]})

    def __len__(self) -> int:
        return 2

    def _get_one(self, index: int):
        self.calls.append(index)
        if index == 0:
            raise OSError("synthetic corrupt file")
        return {}, 1


def run_event_protocol_self_test(output_dir: Path) -> Dict[str, Any]:
    """Exercise default failure, deterministic purge, and post-purge audit."""

    synthetic_rows: Dict[str, Dict[str, Dict[str, Sequence[Any]]]] = {
        "s2": {
            "train": {
                "event_group_id": (
                    "s2-safe-zero",
                    "s2-safe-one",
                    "cross-alpha",
                ),
                "label": (0, 1, 1),
            },
            "val": {
                "event_group_id": ("cross-beta", "s2-val-only"),
                "label": (1, 0),
            },
        },
        "l89": {
            "train": {
                "plume_id": (
                    "l89-safe-zero-a",
                    "l89-safe-one-a",
                    "l89-other-a",
                ),
                "label": (0, 1, 0),
            },
            "val": {
                "plume_id": ("cross-alpha-a", "l89-val-only-a"),
                "label": (1, 0),
            },
        },
        "s5p": {
            "train": {
                "plume_id": (
                    "s5p-safe-zero-a",
                    "s5p-safe-one-a",
                    "cross-beta-a",
                ),
                "label": (0, 1, 0),
            },
            "val": {
                "plume_id": ("s5p-val-one-a", "s5p-val-zero-a"),
                "label": (1, 0),
            },
        },
        "emit": {
            "train": {
                "plume_id": ("emit-safe-zero-a", "emit-safe-one-a"),
                "label": (0, 1),
            },
            "val": {
                "plume_id": ("emit-val-one-a", "emit-val-zero-a"),
                "label": (1, 0),
            },
        },
    }
    with tempfile.TemporaryDirectory(
        prefix="event_protocol_self_test_",
        dir=str(output_dir),
    ) as temporary_name:
        root = Path(temporary_name)
        plans: Dict[str, SensorPlan] = {}
        for sensor in SENSOR_ORDER:
            paths: Dict[str, Path] = {}
            for split in ("train", "val"):
                path = root / f"{sensor}_{split}.csv"
                pd.DataFrame(synthetic_rows[sensor][split]).to_csv(
                    path,
                    index=False,
                    lineterminator="\n",
                )
                paths[split] = path
            plans[sensor] = SensorPlan(
                name=sensor,
                train_csv=str(paths["train"]),
                val_csv=str(paths["val"]),
                path_columns=(),
                time_columns=(),
                input_mode="raw",
                residual_slots=(),
                source_timepoints=1,
                model_timepoints=1,
                channels=1,
                resize_size=0,
                stats_semantics="synthetic-event-protocol",
            )

        protocol_args = argparse.Namespace(
            cross_sensor_event_policy="error",
            protocol_manifest_dir=root / "purged",
            output_dir=root / "run",
        )
        default_error_ok = False
        observed_before = None
        try:
            enforce_event_protocol(protocol_args, SENSOR_ORDER, plans)
        except EventProtocolError as exc:
            observed_before = exc.audit.get("global_train_val_overlap_before")
            default_error_ok = observed_before == 2
        if not default_error_ok:
            raise AssertionError(
                "Default cross-sensor event policy did not fail closed on the "
                f"two synthetic overlaps; observed={observed_before}"
            )

        protocol_args.cross_sensor_event_policy = "purge"
        purged_plans, audit = enforce_event_protocol(
            protocol_args,
            SENSOR_ORDER,
            plans,
        )
        if audit["global_train_val_overlap_after"] != 0:
            raise AssertionError("Synthetic purge did not reach zero overlap")
        if not audit["purge_applied"]:
            raise AssertionError("Synthetic purge was not recorded")
        if audit["sensors"]["s2"]["purged_train"]["removed_rows"] != 1:
            raise AssertionError("Synthetic S2 purge removed the wrong row count")
        if audit["sensors"]["s5p"]["purged_train"]["removed_rows"] != 1:
            raise AssertionError("Synthetic S5P purge removed the wrong row count")
        if not all(Path(plan.train_csv).is_file() for plan in purged_plans.values()):
            raise AssertionError("Synthetic purged manifests were not materialized")
        return {
            "default_error_fail_closed": default_error_ok,
            "global_overlap_before": int(
                audit["global_train_val_overlap_before"]
            ),
            "global_overlap_after": int(
                audit["global_train_val_overlap_after"]
            ),
            "purge_applied": bool(audit["purge_applied"]),
            "removed_rows": {
                sensor: int(
                    audit["sensors"][sensor]["purged_train"]["removed_rows"]
                )
                for sensor in SENSOR_ORDER
            },
        }


def run_self_test(args: argparse.Namespace) -> Dict[str, Any]:
    args.device = "cpu"
    args.amp = False
    args.backbone_mode = "frozen"
    args.freeze_backbone_epochs = 0
    args.rounds_per_epoch = 2
    args.max_train_rounds = 2
    args.max_eval_batches = 1
    args.gradient_accumulation_rounds = 1
    args.num_workers = 0
    args.pin_memory = False
    args.epochs = 1
    sensors = SENSOR_ORDER
    timepoints = {"s2": 1, "l89": 3, "s5p": 6, "emit": 2}
    channels = {"s2": 12, "l89": 7, "s5p": 1, "emit": 32}
    plans = {
        sensor: SensorPlan(
            name=sensor,
            train_csv="synthetic",
            val_csv="synthetic",
            path_columns=tuple(f"p{index}" for index in range(timepoints[sensor])),
            time_columns=tuple(f"t{index}" for index in range(timepoints[sensor])),
            input_mode="raw",
            residual_slots=(),
            source_timepoints=timepoints[sensor],
            model_timepoints=timepoints[sensor],
            channels=channels[sensor],
            resize_size=0,
            stats_semantics="synthetic",
        )
        for sensor in sensors
    }
    datasets = {
        sensor: SyntheticSensorDataset(
            sensor=sensor,
            timepoints=timepoints[sensor],
            channels=channels[sensor],
            length=4,
            seed=args.seed + index,
        )
        for index, sensor in enumerate(sensors)
    }
    loaders = {
        sensor: DataLoader(dataset, batch_size=2, shuffle=False)
        for sensor, dataset in datasets.items()
    }
    device = torch.device("cpu")
    model = SharedPanopticonTemporalClassifier(
        TinyBackbone(), timepoints, time_embed_dim=48
    ).to(device)
    model.set_backbone_trainable(False)
    losses = build_losses(
        datasets,  # type: ignore[arg-type]
        class_balanced=False,
        label_smoothing=0.0,
        device=device,
    )
    optimizer = build_optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    train_metrics, global_round = train_one_epoch(
        model,
        plans,
        loaders,
        losses,
        optimizer,
        None,
        scaler,
        device=device,
        epoch=1,
        args=args,
        global_round=0,
    )
    val_metrics = evaluate(
        model,
        plans,
        loaders,
        losses,
        device=device,
        args=args,
    )

    # Verify atomic checkpoint serialization before testing subset loading.
    counts = parameter_counts(model)
    checkpoint_path = args.output_dir / "self_test_checkpoint.pth"
    synthetic_normalization = {
        sensor: {
            "mean": [0.0] * channels[sensor],
            "std": [1.0] * channels[sensor],
            "source": "synthetic",
        }
        for sensor in sensors
    }
    synthetic_event_protocol = {
        "schema_version": EVENT_PROTOCOL_SCHEMA_VERSION,
        "global_train_val_overlap_after": 0,
        "fingerprint": "synthetic-self-test",
    }
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        epoch=1,
        global_round=global_round,
        history=[{"epoch": 1, "train": train_metrics, "val": val_metrics}],
        best_macro_ap=float(val_metrics["macro"]["ap"]),
        args=args,
        plans=plans,
        normalization=synthetic_normalization,
        backbone_provenance={"source": "synthetic_tiny_backbone"},
        counts=counts,
        event_protocol=synthetic_event_protocol,
    )
    atomic_torch_save(payload, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location="cpu")
    if reloaded.get("schema_version") != 2 or reloaded.get("epoch") != 1:
        raise AssertionError("atomic self-test checkpoint metadata did not round-trip")
    validate_checkpoint_event_protocol(
        synthetic_event_protocol,
        reloaded,
        enabled_sensors=sensors,
        exact=True,
    )
    validate_exact_training_resume_plans(plans, reloaded)

    # Verify a checkpoint containing four sensors can initialize a one-sensor
    # inference model without requiring any other sensor input.
    single_sensor = SharedPanopticonTemporalClassifier(
        TinyBackbone(), {"s2": 1}, time_embed_dim=48
    )
    load_model_state_for_enabled_sensors(single_sensor, reloaded["model"])
    single_batch = next(iter(loaders["s2"]))[0]
    single_logits = single_sensor("s2", single_batch)
    if tuple(single_logits.shape) != (2, 2):
        raise AssertionError(
            f"single-sensor inference shape failed: {tuple(single_logits.shape)}"
        )

    # Verify strict loading does not touch index 1 after index 0 fails.
    failing_base = FailingSyntheticBase()
    strict = StrictIndexedDataset("synthetic", failing_base)
    fail_fast_ok = False
    try:
        strict[0]
    except SampleLoadError:
        fail_fast_ok = failing_base.calls == [0]
    if not fail_fast_ok:
        raise AssertionError(
            f"strict sample policy unexpectedly replaced a row: {failing_base.calls}"
        )

    event_protocol_test = run_event_protocol_self_test(args.output_dir)
    result = {
        "status": "passed",
        "device": "cpu",
        "balanced_rounds": global_round,
        "train": train_metrics,
        "val": val_metrics,
        "single_sensor_inference_shape": list(single_logits.shape),
        "fail_fast_no_replacement": fail_fast_ok,
        "atomic_checkpoint_round_trip": True,
        "event_protocol": event_protocol_test,
        "checkpoint_path": str(checkpoint_path),
        "parameter_counts_tiny": counts,
    }
    output = args.output_dir / "self_test_result.json"
    atomic_json_dump(json_safe(result), output)
    print(f"[SelfTest] passed; result={output}", flush=True)
    return result


def apply_smoke_settings(args: argparse.Namespace) -> None:
    if not args.smoke_30:
        return
    args.backbone_mode = "frozen"
    args.freeze_backbone_epochs = 0
    args.epochs = 1
    args.max_train_rounds = (
        min(int(args.max_train_rounds), 30)
        if int(args.max_train_rounds) > 0
        else 30
    )
    args.max_eval_batches = (
        min(int(args.max_eval_batches), 2)
        if int(args.max_eval_batches) > 0
        else 2
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    apply_smoke_settings(args)
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if (
        args.eval_only
        and args.resume is None
        and args.init_checkpoint is None
        and not args.plan_only
    ):
        raise ValueError(
            "--eval-only requires --resume or --init-checkpoint; "
            "random sensor heads are not an evaluation target"
        )
    if args.eval_sensor is not None and not args.eval_only:
        raise ValueError("--eval-sensor is only meaningful with --eval-only")
    requested_sensors = parse_sensors(args.sensors)
    if args.eval_sensor is not None:
        if args.eval_sensor not in requested_sensors:
            raise ValueError(
                f"--eval-sensor {args.eval_sensor} is not enabled by "
                f"--sensors {requested_sensors}"
            )
    protocol_plans = build_sensor_plans(args, requested_sensors)
    header_report = validate_csv_headers(protocol_plans)
    protocol_plans, event_protocol = enforce_event_protocol(
        args,
        requested_sensors,
        protocol_plans,
    )
    header_report = validate_csv_headers(protocol_plans)

    sensors = requested_sensors
    plans = protocol_plans
    if args.eval_sensor is not None and args.eval_only:
        sensors = (args.eval_sensor,)
        plans = {args.eval_sensor: protocol_plans[args.eval_sensor]}
        header_report = {args.eval_sensor: header_report[args.eval_sensor]}
    if args.plan_only:
        payload = {
            "sensors": list(sensors),
            "plans": {
                sensor: asdict(plan) for sensor, plan in plans.items()
            },
            "csv_headers": header_report,
            "event_protocol": event_protocol,
            "note": (
                "No image, checkpoint, or normalization-stat I/O was "
                "performed. The manifest-only global event audit did run."
            ),
        }
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return payload

    if args.self_test:
        # Self-test is handled before real plan validation in cli_main.
        raise AssertionError("unreachable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    status_path = args.output_dir / "run_status.json"
    metrics_path = args.output_dir / "metrics.json"
    latest_path = args.output_dir / "checkpoint_latest.pth"
    best_path = args.output_dir / "checkpoint_best_macro_ap.pth"
    atomic_json_dump(
        json_safe(event_protocol),
        args.output_dir / "event_protocol_audit.json",
    )
    atomic_json_dump(
        {
            "status": "running",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "pid": os.getpid(),
        },
        status_path,
    )

    resume_payload = None
    init_payload = None
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f"[Resume] reading {args.resume}", flush=True)
        resume_payload = torch.load(args.resume, map_location="cpu")
        validate_checkpoint_event_protocol(
            event_protocol,
            resume_payload,
            enabled_sensors=sensors,
            exact=not args.eval_only,
        )
        if not args.eval_only:
            validate_exact_training_resume_plans(plans, resume_payload)
    elif args.init_checkpoint is not None:
        if not args.init_checkpoint.is_file():
            raise FileNotFoundError(
                f"Initialization checkpoint not found: {args.init_checkpoint}"
            )
        print(f"[Init] reading {args.init_checkpoint}", flush=True)
        init_payload = torch.load(args.init_checkpoint, map_location="cpu")
        validate_checkpoint_event_protocol(
            event_protocol,
            init_payload,
            enabled_sensors=sensors,
            exact=not args.eval_only,
        )

    caches = build_cache_objects(args, sensors)
    checkpoint_normalization = (
        resume_payload.get("normalization")
        if isinstance(resume_payload, Mapping)
        else None
    )
    normalization = load_or_compute_normalization(
        plans,
        args,
        caches,
        checkpoint_records=checkpoint_normalization,
    )
    (
        train_datasets,
        _val_datasets,
        train_loaders,
        val_loaders,
    ) = build_loaders(plans, normalization, args, caches)

    device = choose_device(args.device)
    if resume_payload is not None or init_payload is not None:
        backbone, _ = build_backbone("none")
        source_payload = (
            resume_payload if resume_payload is not None else init_payload
        )
        backbone_provenance = dict(
            source_payload.get(
                "backbone_provenance",
                {"source": "runner_checkpoint_without_embedded_provenance"},
            )
        )
        backbone_provenance["restored_from_runner_checkpoint"] = str(
            args.resume if resume_payload is not None else args.init_checkpoint
        )
    else:
        backbone, backbone_provenance = build_backbone(args.weights)
    model = SharedPanopticonTemporalClassifier(
        backbone,
        {
            sensor: plan.model_timepoints
            for sensor, plan in plans.items()
        },
        time_embed_dim=args.time_embed_dim,
    )
    model.set_backbone_trainable(args.backbone_mode == "finetune")
    model.to(device)
    counts = parameter_counts(model)
    print(f"[Model] parameter counts: {counts}", flush=True)

    losses = build_losses(
        train_datasets,
        class_balanced=args.class_balanced_loss,
        label_smoothing=args.label_smoothing,
        device=device,
    )
    optimizer = build_optimizer(model, args)
    rounds_per_epoch = resolve_rounds(train_loaders, args)
    optimizer_steps_per_epoch = math.ceil(
        rounds_per_epoch / max(1, args.gradient_accumulation_rounds)
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=max(1, args.epochs * optimizer_steps_per_epoch),
        warmup_steps=args.warmup_rounds,
        mode=args.scheduler,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            bool(args.amp)
            and device.type == "cuda"
            and args.amp_dtype == "fp16"
        )
    )

    start_epoch = 1
    global_round = 0
    history: list[Dict[str, Any]] = []
    best_macro_ap = float("-inf")
    if resume_payload is not None:
        load_model_state_for_enabled_sensors(
            model, resume_payload["model"]
        )
        exact_training_resume = not args.eval_only
        if exact_training_resume:
            optimizer.load_state_dict(resume_payload["optimizer"])
            if scheduler is not None and resume_payload.get("scheduler") is not None:
                scheduler.load_state_dict(resume_payload["scheduler"])
            if resume_payload.get("scaler") is not None:
                scaler.load_state_dict(resume_payload["scaler"])
            start_epoch = int(resume_payload["epoch"]) + 1
            global_round = int(resume_payload.get("global_round", 0))
            history = list(resume_payload.get("history", []))
            best_macro_ap = float(
                resume_payload.get("best_macro_ap", float("-inf"))
            )
            restore_rng(resume_payload)
        print(
            f"[Resume] model loaded; start_epoch={start_epoch} "
            f"global_round={global_round}",
            flush=True,
        )
    elif init_payload is not None:
        load_model_state_for_enabled_sensors(model, init_payload["model"])
        print(
            f"[Init] loaded model weights from {args.init_checkpoint}",
            flush=True,
        )

    run_config = {
        "schema_version": 2,
        "args": json_safe(vars(args)),
        "sensor_plans": {
            sensor: asdict(plan) for sensor, plan in plans.items()
        },
        "normalization": normalization,
        "backbone_provenance": backbone_provenance,
        "parameter_counts": counts,
        "balanced_rounds_per_epoch": int(rounds_per_epoch),
        "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
        "csv_headers": header_report,
        "event_protocol": event_protocol,
        "bad_sample_policy": "fail_fast_at_original_index_no_replacement",
    }
    atomic_json_dump(json_safe(run_config), config_path)

    if args.eval_only:
        evaluation = evaluate(
            model,
            plans,
            val_loaders,
            losses,
            device=device,
            args=args,
        )
        result = {
            "status": "complete",
            "mode": "eval_only",
            "checkpoint": str(args.resume or args.init_checkpoint),
            "evaluation": evaluation,
        }
        atomic_json_dump(json_safe(result), metrics_path)
        atomic_json_dump(
            {
                "status": "complete",
                "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "mode": "eval_only",
            },
            status_path,
        )
        return result

    if start_epoch > args.epochs:
        raise ValueError(
            f"Resume checkpoint epoch {start_epoch - 1} is already >= --epochs {args.epochs}"
        )

    for epoch in range(start_epoch, int(args.epochs) + 1):
        epoch_start = time.monotonic()
        train_metrics, global_round = train_one_epoch(
            model,
            plans,
            train_loaders,
            losses,
            optimizer,
            scheduler,
            scaler,
            device=device,
            epoch=epoch,
            args=args,
            global_round=global_round,
        )
        val_metrics = evaluate(
            model,
            plans,
            val_loaders,
            losses,
            device=device,
            args=args,
        )
        record = {
            "epoch": int(epoch),
            "global_round": int(global_round),
            "elapsed_seconds": float(time.monotonic() - epoch_start),
            "train": train_metrics,
            "val": val_metrics,
            "learning_rates": {
                str(group.get("name", index)): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            },
        }
        history.append(record)
        atomic_json_dump(
            {
                "schema_version": 1,
                "history": history,
                "selection_metric": "validation macro AP",
            },
            metrics_path,
        )

        macro_ap = val_metrics["macro"].get("ap")
        is_best = (
            macro_ap is not None
            and math.isfinite(float(macro_ap))
            and float(macro_ap) > best_macro_ap
        )
        if is_best:
            best_macro_ap = float(macro_ap)
        if not args.no_save_checkpoints:
            payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_round=global_round,
                history=history,
                best_macro_ap=best_macro_ap,
                args=args,
                plans=plans,
                normalization=normalization,
                backbone_provenance=backbone_provenance,
                counts=counts,
                event_protocol=event_protocol,
            )
            atomic_torch_save(payload, latest_path)
            if is_best:
                atomic_torch_save(payload, best_path)
        print(
            f"[Epoch] {epoch}: val_macro={val_metrics['macro']} "
            f"best_macro_ap={best_macro_ap}",
            flush=True,
        )

    result = {
        "status": "complete",
        "history": history,
        "best_macro_ap": (
            None if not math.isfinite(best_macro_ap) else best_macro_ap
        ),
        "latest_checkpoint": (
            None if args.no_save_checkpoints else str(latest_path)
        ),
        "best_checkpoint": (
            None
            if args.no_save_checkpoints or not best_path.is_file()
            else str(best_path)
        ),
    }
    atomic_json_dump(
        {
            "status": "complete",
            "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "epochs_completed": len(history),
            "best_macro_ap": result["best_macro_ap"],
        },
        status_path,
    )
    return result


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)
    if args.self_test:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_self_test(args)
        return 0

    status_path = args.output_dir / "run_status.json"
    try:
        run(args)
    except Exception as exc:
        if not args.plan_only:
            with contextlib.suppress(Exception):
                if isinstance(exc, EventProtocolError):
                    atomic_json_dump(
                        json_safe(exc.audit),
                        args.output_dir / "event_protocol_audit.json",
                    )
                atomic_json_dump(
                    {
                        "status": "failed",
                        "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    status_path,
                )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
