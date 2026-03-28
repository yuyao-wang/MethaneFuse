"""Multi-sensor Panopticon training script with sensor-specific embeddings and heads.

This module exposes reusable model components (``MultiSensorPanopticonClassifier``)
*and* a runnable training entry point that ingests mixed-sensor CSVs containing
Sentinel-2, Landsat 8/9, Sentinel-5P, and WV3 samples. Each sensor owns its own
Panopticon patch embedding, while the DinoViT backbone is shared and updated by
the consensus of all datasets in a batch.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import math
import os
import random
import shutil
import sys
import warnings
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# Make the repository root importable so examples work when executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid CUDA init stalls on CPU-only hosts.
os.environ.setdefault("XFORMERS_DISABLED", "1")

from dinov2.data.datasets.s2_csv import S2TemporalCsvDataset, _SkipSample
from dinov2.utils.data import extract_wavemus, load_ds_cfg
from dinov2.models.panopticon import PanopticonPE
from dinov2.models.vision_transformer import DinoVisionTransformer

S2_PRECOMPUTED_STATS = (
    [
        786.128173828125,
        1025.8876953125,
        1593.730712890625,
        2315.26123046875,
        2710.462890625,
        3115.90087890625,
        3289.0830078125,
        3465.536376953125,
        3495.579833984375,
        3517.7958984375,
        4180.28564453125,
        3567.866943359375,
    ],
    [
        435.72607421875,
        597.6113891601562,
        688.5059814453125,
        840.1614990234375,
        801.7208251953125,
        706.9466552734375,
        689.823974609375,
        727.5567626953125,
        668.30224609375,
        551.3565063476562,
        629.679931640625,
        641.590087890625,
    ],
)

L89_PRECOMPUTED_STATS = (
    [
        10729.92784546,
        11384.64407242,
        13172.77519667,
        14892.25620267,
        18149.92169893,
        20249.17615773,
        18375.0669698,
    ],
    [
        1029.18232283,
        1188.52313418,
        1552.27685613,
        1959.74400972,
        1954.80410093,
        2098.98682671,
        1895.56781996,
    ],
)

# Sentinel-5P CH4 stacked (t0, t-90, t-360) stats reused from existing examples.
S5P_PRECOMPUTED_STATS = (
    [1908.944798144908, 362.20128748570943, 666.5645739544017],
    [52.2615669211965, 743.7553740768346, 901.8399543163595],
)

DEFAULT_WV3_BANDS = [
    "Coastal (MS7)",
    "Blue (MS4)",
    "Green (MS3)",
    "Yellow (MS6)",
    "Red (MS2)",
    "Red Edge (MS5)",
    "NIR1 (MS1)",
    "NIR2 (MS8)",
    "SWIR1",
    "SWIR2",
    "SWIR3",
    "SWIR4",
    "SWIR5",
    "SWIR6",
    "SWIR7",
    "SWIR8",
]


def _parse_optional_float(text: str) -> Optional[float]:
    text = str(text).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_wv3_channel_ids_from_srf(csv_path: str, band_names: Sequence[str]) -> torch.Tensor:
    """Compute WV3 channel IDs (mu) from an SRF CSV file."""

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"WV3 SRF file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"WV3 SRF file is empty: {csv_path}") from exc

        header_idx = {name: idx for idx, name in enumerate(header) if name}
        missing = [name for name in band_names if name not in header_idx]
        if missing:
            raise ValueError(
                f"Missing WV3 band columns in {csv_path}: {missing}. Available columns include: {header[:18]}"
            )

        wavelengths = []
        responses = {name: [] for name in band_names}
        for row in reader:
            if not row:
                continue
            wavelength = _parse_optional_float(row[0] if len(row) > 0 else "")
            if wavelength is None:
                continue
            wavelengths.append(wavelength)
            for name in band_names:
                idx = header_idx[name]
                value = _parse_optional_float(row[idx] if idx < len(row) else "")
                responses[name].append(max(0.0, value if value is not None else 0.0))

    if len(wavelengths) == 0:
        raise ValueError(f"No numeric wavelength rows parsed from {csv_path}")

    wavelength_t = torch.tensor(wavelengths, dtype=torch.float64)
    mu_list = []
    for name in band_names:
        weights = torch.tensor(responses[name], dtype=torch.float64)
        total = torch.sum(weights)
        if total <= 0:
            raise ValueError(f"Band '{name}' has zero response everywhere in {csv_path}")
        mu = torch.sum(wavelength_t * weights) / total
        mu_list.append(mu)
    return torch.round(torch.stack(mu_list)).to(torch.int16)


# --------------------------------------------------------------------------------------
#  Model components
# --------------------------------------------------------------------------------------

class CLSHead(nn.Module):
    """LayerNorm + Linear classification head used per sensor."""

    def __init__(self, embed_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fc(self.norm(cls_token))

    @property
    def out_features(self) -> int:
        return self.fc.out_features


class LogitSummaryHead(nn.Module):
    """Fuse logits from all sensor heads into one final prediction."""

    def __init__(self, *, num_heads: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        in_dim = num_heads * num_classes
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class MaskedAttentionPooling(nn.Module):
    """Mask-aware attention pooling over a variable number of sensor features per row."""

    def __init__(self, *, embed_dim: int, num_sensors: int):
        super().__init__()
        self.sensor_embed = nn.Embedding(num_sensors, embed_dim)
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        feats: torch.Tensor,
        sensor_indices: torch.Tensor,
        sample_to_row: torch.Tensor,
        *,
        num_rows: int,
    ) -> torch.Tensor:
        if feats.ndim != 2:
            raise ValueError(f"feats must be [N,D], got shape={tuple(feats.shape)}")
        if sensor_indices.shape[0] != feats.shape[0] or sample_to_row.shape[0] != feats.shape[0]:
            raise ValueError(
                "MaskedAttentionPooling input length mismatch: "
                f"feats={feats.shape[0]}, sensor_indices={sensor_indices.shape[0]}, sample_to_row={sample_to_row.shape[0]}"
            )
        attn_in = feats + self.sensor_embed(sensor_indices)
        scores = self.score(attn_in).squeeze(-1)
        out = feats.new_zeros((num_rows, feats.shape[-1]))
        for row_idx in range(num_rows):
            mask = sample_to_row == row_idx
            if not torch.any(mask):
                continue
            weights = torch.softmax(scores[mask], dim=0)
            out[row_idx] = torch.sum(feats[mask] * weights.unsqueeze(-1), dim=0)
        return out


class MultiSensorPanopticonClassifier(nn.Module):
    """Shared DinoViT backbone with sensor-specific Panopticon PEs and row-level fusion."""

    def __init__(
        self,
        *,
        backbone: Optional[DinoVisionTransformer] = None,
        sensors: Sequence[str] = ("s2", "l89", "s5p", "wv3"),
        num_classes: Mapping[str, int] | int = 2,
        patch_embed_overrides: Optional[Mapping[str, PanopticonPE]] = None,
        contrastive_temperature: float = 0.1,
    ):
        super().__init__()
        if backbone is None:
            backbone = _load_backbone()
        if not isinstance(backbone, DinoVisionTransformer):
            raise TypeError("backbone must be a DinoVisionTransformer instance")

        self.backbone = backbone
        self.sensor_order = list(sensors)
        if not self.sensor_order:
            raise ValueError("At least one sensor must be specified")
        self.sensor_to_idx = {sensor: idx for idx, sensor in enumerate(self.sensor_order)}

        base_patch_embed = backbone.patch_embed
        sensor_modules = nn.ModuleDict()
        for sensor in self.sensor_order:
            if patch_embed_overrides and sensor in patch_embed_overrides:
                sensor_modules[sensor] = patch_embed_overrides[sensor]
            elif sensor == self.sensor_order[0]:
                sensor_modules[sensor] = base_patch_embed
            else:
                sensor_modules[sensor] = copy.deepcopy(base_patch_embed)
        self.sensor_patch_embeds = sensor_modules
        self.backbone.patch_embed = self.sensor_patch_embeds[self.sensor_order[0]]

        self.contrastive_temperature = float(contrastive_temperature)
        if self.contrastive_temperature <= 0.0:
            raise ValueError(f"contrastive_temperature must be > 0, got {self.contrastive_temperature}")

        embed_dim = getattr(self.backbone, "embed_dim", 768)
        if isinstance(num_classes, int):
            class_map = {sensor: num_classes for sensor in self.sensor_order}
        else:
            class_map = {sensor: num_classes[sensor] for sensor in self.sensor_order}
        out_dims = {int(class_map[sensor]) for sensor in self.sensor_order}
        if len(out_dims) != 1:
            raise ValueError("All sensors must share one num_classes value for row-level fusion")
        num_out_classes = out_dims.pop()
        self.row_fusion_pool = MaskedAttentionPooling(embed_dim=embed_dim, num_sensors=len(self.sensor_order))
        self.row_fusion_head = CLSHead(embed_dim=embed_dim, num_classes=num_out_classes)

    @contextmanager
    def _use_sensor(self, sensor: str):
        original = self.backbone.patch_embed
        self.backbone.patch_embed = self.sensor_patch_embeds[sensor]
        try:
            yield
        finally:
            self.backbone.patch_embed = original

    def encode_sensors(self, sensors: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
        idxs = [self.sensor_to_idx[s] for s in sensors]
        return torch.tensor(idxs, device=device, dtype=torch.long)

    def _normalize_sensors(self, sensors: Union[Sequence[str], torch.Tensor]) -> Sequence[str]:
        if isinstance(sensors, torch.Tensor):
            indices = sensors.detach().to("cpu").tolist()
            return [self.sensor_order[i] for i in indices]
        return list(sensors)

    def _build_sensor_batches(
        self,
        x_dict: MutableMapping[str, torch.Tensor],
        sensors: Sequence[str],
    ) -> Dict[str, "SensorBatch"]:
        device = next(iter(x_dict.values())).device
        batches: Dict[str, SensorBatch] = {}
        for sensor_name in self.sensor_order:
            idx = [i for i, s in enumerate(sensors) if s == sensor_name]
            if not idx:
                continue
            idx_tensor = torch.tensor(idx, device=device, dtype=torch.long)
            batches[sensor_name] = SensorBatch(idx_tensor, _slice_x_dict(x_dict, idx_tensor))
        return batches

    def forward(
        self,
        x_dict: MutableMapping[str, torch.Tensor],
        sensors: Sequence[str],
        *,
        return_features: bool = False,
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        sensor_labels = self._normalize_sensors(sensors)
        sensor_batches = self._build_sensor_batches(x_dict, sensor_labels)
        if not sensor_batches:
            raise ValueError("No samples matched the configured sensors")

        outputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]] = {}

        for sensor_name, sensor_batch in sensor_batches.items():
            with self._use_sensor(sensor_name):
                feats = self.backbone(sensor_batch.x_dict, is_training=True)
            cls_token = torch.nan_to_num(feats["x_norm_clstoken"], nan=0.0, posinf=1e4, neginf=-1e4)
            outputs[sensor_name] = {
                "indices": sensor_batch.indices,
                "cls_token": cls_token,
            }
            if return_features:
                outputs[sensor_name]["feats"] = feats["x_norm_patchtokens"]

        self._ensure_sensor_keys(outputs, x_dict, return_features=return_features)
        return outputs

    def compute_contrastive_loss(
        self,
        outputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        sample_to_row: torch.Tensor,
        *,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        if sample_to_row.ndim != 1:
            raise ValueError(f"sample_to_row must be rank-1, got shape={tuple(sample_to_row.shape)}")
        tau = self.contrastive_temperature if temperature is None else float(temperature)
        if tau <= 0.0:
            raise ValueError(f"contrastive temperature must be > 0, got {tau}")

        cls_list: list[torch.Tensor] = []
        row_ids_list: list[torch.Tensor] = []
        for sensor_name in self.sensor_order:
            sensor_out = outputs.get(sensor_name)
            if not isinstance(sensor_out, dict):
                continue
            idx = sensor_out.get("indices")
            cls_token = sensor_out.get("cls_token")
            if not isinstance(idx, torch.Tensor) or not isinstance(cls_token, torch.Tensor) or idx.numel() == 0:
                continue
            cls_list.append(cls_token)
            row_ids_list.append(sample_to_row.index_select(0, idx))

        if not cls_list:
            return sample_to_row.new_zeros((), dtype=torch.float32)

        z = torch.cat(cls_list, dim=0).float()
        row_ids = torch.cat(row_ids_list, dim=0)
        if z.shape[0] < 2:
            return z.new_zeros(())

        z = F.normalize(z, p=2, dim=-1)
        logits = torch.matmul(z, z.T) / tau
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        n = logits.shape[0]
        diag = torch.eye(n, dtype=torch.bool, device=logits.device)
        positive_mask = (row_ids.view(-1, 1) == row_ids.view(1, -1)) & (~diag)
        valid_anchor_mask = positive_mask.any(dim=1)
        if not torch.any(valid_anchor_mask):
            return logits.new_zeros(())

        logits = logits.masked_fill(diag, torch.finfo(logits.dtype).min)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        pos_counts = positive_mask.sum(dim=1).clamp_min(1)
        mean_log_prob_pos = (log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / pos_counts)
        return -mean_log_prob_pos[valid_anchor_mask].mean()

    def _ensure_sensor_keys(
        self,
        outputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        x_dict: MutableMapping[str, torch.Tensor],
        *,
        return_features: bool = False,
    ) -> None:
        device = next(iter(x_dict.values())).device
        embed_dim = getattr(self.backbone, "embed_dim", 768)
        for sensor_name in self.sensor_order:
            if sensor_name in outputs:
                continue
            empty_idx = torch.empty((0,), dtype=torch.long, device=device)
            empty_cls = torch.empty((0, embed_dim), device=device)
            placeholder: Dict[str, torch.Tensor] = {
                "indices": empty_idx,
                "cls_token": empty_cls,
            }
            if return_features:
                placeholder["feats"] = torch.empty((0, 0), device=device)
            outputs[sensor_name] = placeholder

    def compute_row_fused_logits(
        self,
        outputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        *,
        sample_to_row: torch.Tensor,
        num_rows: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.row_fusion_pool is None or self.row_fusion_head is None:
            raise RuntimeError("Row fusion modules are not initialized")
        if sample_to_row.ndim != 1:
            raise ValueError(f"sample_to_row must be rank-1, got shape={tuple(sample_to_row.shape)}")

        num_samples = int(sample_to_row.shape[0])
        embed_dim = getattr(self.backbone, "embed_dim", 768)
        flat_feats = torch.zeros((num_samples, embed_dim), device=device)
        flat_sensor_indices = torch.zeros((num_samples,), dtype=torch.long, device=device)

        for sensor_name in self.sensor_order:
            sensor_out = outputs.get(sensor_name)
            if not isinstance(sensor_out, dict):
                continue
            idx = sensor_out.get("indices")
            cls_token = sensor_out.get("cls_token")
            if not isinstance(idx, torch.Tensor) or not isinstance(cls_token, torch.Tensor) or idx.numel() == 0:
                continue
            flat_feats.index_copy_(0, idx, cls_token)
            flat_sensor_indices.index_fill_(0, idx, int(self.sensor_to_idx[sensor_name]))

        row_features = self.row_fusion_pool(
            flat_feats,
            flat_sensor_indices,
            sample_to_row,
            num_rows=num_rows,
        )
        return self.row_fusion_head(row_features)


@dataclass
class SensorBatch:
    indices: torch.Tensor
    x_dict: Dict[str, torch.Tensor]


# --------------------------------------------------------------------------------------
#  Dataset helpers (mixed temporal CSV + S5P NPZ support)
# --------------------------------------------------------------------------------------

class StaticAnchoredCache:
    def __init__(self, cache_dir: str, min_free_gb: float = 10.0):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_free_bytes = min_free_gb * (1024**3)
        self._warned_low_space = False
        self._warned_copy_failure = False

    def _get_free_space(self) -> int:
        return shutil.disk_usage(self.cache_dir).free

    def _hashed_path(self, original: str) -> Path:
        norm_path = os.path.abspath(original)
        digest = hashlib.sha1(norm_path.encode("utf-8")).hexdigest()
        subdir = digest[:2]
        suffix = Path(original).suffix
        return self.cache_dir / subdir / f"{digest}{suffix}"

    def ensure_local(self, original: str) -> str:
        original = str(original).strip()
        if original == "":
            return original
        dst = self._hashed_path(original)
        if dst.exists():
            return str(dst)
        if self._get_free_space() < self.min_free_bytes:
            if not self._warned_low_space:
                free_gb = self._get_free_space() / float(1024**3)
                print(
                    f"[Cache][Warn] Low free space ({free_gb:.2f} GB) under {self.cache_dir}; "
                    "falling back to remote reads.",
                    flush=True,
                )
                self._warned_low_space = True
            return original
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(original, tmp)
            os.replace(tmp, dst)
        except Exception as exc:
            with suppress(FileNotFoundError):
                tmp.unlink()
            if not self._warned_copy_failure:
                print(
                    f"[Cache][Warn] Copy failed for first seen file '{original}': {exc}. "
                    "Falling back to source path.",
                    flush=True,
                )
                self._warned_copy_failure = True
            return original
        return str(dst)

    def warm_up(self, paths: Sequence[str], max_workers: int = 8) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        unique_paths = sorted({os.path.abspath(p) for p in paths if isinstance(p, str)})
        if not unique_paths:
            return
        print(f"[Cache] Warming up (target: {len(unique_paths)})...", flush=True)

        def _copy_one(path: str):
            res = self.ensure_local(path)
            return res == path

        fallback_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_copy_one, p) for p in unique_paths]
            for fut in as_completed(futures):
                fallback_count += 1 if fut.result() else 0
        print(f"[Cache] Warmup complete. Cached: {len(unique_paths) - fallback_count}, Remote: {fallback_count}")


def _compute_mean_std(stats: Optional[Tuple[Sequence[float], Sequence[float]]]):
    if stats is None:
        return None, None
    mean, std = stats
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_tensor = torch.clamp(torch.tensor(std, dtype=torch.float32), min=1e-6).view(-1, 1, 1)
    return mean_tensor, std_tensor


class TriSensorTemporalCsvDataset(S2TemporalCsvDataset):
    """Temporal dataset that mixes Sentinel-2, Landsat 8/9, Sentinel-5P, and WV3 samples."""

    def __init__(
        self,
        *args,
        local_file_cache: Optional[StaticAnchoredCache] = None,
        s5p_data_key: Optional[str] = "ch4",
        s5p_chn_ids_key: Optional[str] = "chn_ids",
        s5p_channels_last: bool = False,
        align_l89_to_s2: bool = False,
        wv3_chn_ids: Optional[torch.Tensor] = None,
        fusion_group_column: str = "id",
        **kwargs,
    ):
        kwargs.pop("ds_cfg_name", None)
        kwargs.pop("normalize_stats", None)
        self.ds_cfg = None
        self.normalize_stats = None
        self._local_file_cache = local_file_cache
        self._s5p_data_key = s5p_data_key
        self._s5p_chn_ids_key = s5p_chn_ids_key
        self._s5p_channels_last = s5p_channels_last
        self._align_l89_to_s2 = bool(align_l89_to_s2)
        self._deprecated_fusion_group_column = str(fusion_group_column).strip() or "id"
        self._wide_sensor_columns: Dict[str, Tuple[str, ...]] = {}
        self._wv3_chn_ids = None
        if wv3_chn_ids is not None:
            self._wv3_chn_ids = torch.as_tensor(wv3_chn_ids).clone().detach()
            if self._wv3_chn_ids.ndim == 1:
                self._wv3_chn_ids = self._wv3_chn_ids.unsqueeze(-1)
        super().__init__(*args, **kwargs)

        self._validate_sensor_column()
        self.sensor_configs = self._build_sensor_configs()
        self._s5p_mean, self._s5p_std = _compute_mean_std(S5P_PRECOMPUTED_STATS)

    def _validate_sensor_column(self) -> None:
        if "sensor" in self.df.columns:
            raise ValueError(
                "Narrow/long-table CSV ('sensor' column per row) is no longer supported by this script. "
                "Please provide wide-table rows with columns like s2_0_path/l89_0_path/s5p_0_path/wv3_0_path."
            )

        def _wide_cols(prefix: str) -> Optional[Tuple[str, str, str]]:
            cols = (f"{prefix}_0_path", f"{prefix}_90_path", f"{prefix}_360_path")
            return cols if all(col in self.df.columns for col in cols) else None

        s2_cols = _wide_cols("s2")
        l89_cols = _wide_cols("l89")
        s5p_col = ("s5p_0_path",) if "s5p_0_path" in self.df.columns else None
        wv3_cols = _wide_cols("wv3")
        if wv3_cols is None:
            emit_cols = _wide_cols("emit")
            if emit_cols is not None:
                warnings.warn(
                    "Wide-table uses EMIT columns; mapping emit_*_path to sensor key 'wv3' for this script.",
                    stacklevel=2,
                )
                wv3_cols = emit_cols

        if s2_cols is not None:
            self._wide_sensor_columns["s2"] = s2_cols
        if l89_cols is not None:
            self._wide_sensor_columns["l89"] = l89_cols
        if s5p_col is not None:
            self._wide_sensor_columns["s5p"] = s5p_col
        if wv3_cols is not None:
            self._wide_sensor_columns["wv3"] = wv3_cols

        if len(self._wide_sensor_columns) == 0:
            raise ValueError(
                "CSV has no 'sensor' column and no recognized wide-table sensor path columns "
                "(expected prefixes like s2_*, l89_*, s5p_0_path, wv3_* or emit_*)."
            )

    def _build_sensor_configs(self) -> Dict[str, Dict[str, torch.Tensor]]:
        configs = {
            "l89": {
                "ds_cfg": load_ds_cfg("landsat89_7band"),
                "normalize_stats": L89_PRECOMPUTED_STATS,
                "scale_to_unit": True,
            },
            "s2": {
                "ds_cfg": load_ds_cfg("s2_12band"),
                "normalize_stats": S2_PRECOMPUTED_STATS,
                "scale_to_unit": True,
            },
        }

        if self._wv3_chn_ids is not None:
            configs["wv3"] = {
                "ds_cfg": None,
                "normalize_stats": None,
                "chn_ids": self._wv3_chn_ids,
                "mean_tensor": None,
                "std_tensor": None,
                "scale_to_unit": False,
            }

        # Inject channel ids and optional scaling.
        for name, cfg in configs.items():
            if name == "wv3":
                continue
            cfg["normalize_stats"] = self._maybe_scale_stats(cfg.get("normalize_stats"))
            ds_cfg_obj = cfg["ds_cfg"]
            chn_ids = extract_wavemus(ds_cfg_obj, return_sigmas=False).unsqueeze(-1)
            ds_cfg_obj["chn_ids"] = chn_ids
            cfg["chn_ids"] = chn_ids

        if self._align_l89_to_s2:
            configs["l89"]["chn_ids"] = configs["s2"]["chn_ids"]

        for cfg in configs.values():
            mean_tensor, std_tensor = _compute_mean_std(cfg.get("normalize_stats"))
            cfg["mean_tensor"] = mean_tensor
            cfg["std_tensor"] = std_tensor

        return configs

    def _maybe_scale_stats(self, stats):
        if stats is None or not getattr(self, "scale_to_unit", False):
            return stats
        mean, std = stats
        return ([m / 65535.0 for m in mean], [s / 65535.0 for s in std])

    @contextmanager
    def _use_sensor_cfg(self, sensor: str):
        cfg = self.sensor_configs[sensor]
        original = (self.ds_cfg, self.normalize_stats, self.chn_ids, self._mean, self._std, self.scale_to_unit)
        self.ds_cfg = cfg["ds_cfg"]
        self.normalize_stats = cfg["normalize_stats"]
        self.chn_ids = cfg["chn_ids"]
        self._mean = cfg.get("mean_tensor")
        self._std = cfg.get("std_tensor")
        self.scale_to_unit = bool(cfg.get("scale_to_unit", self.scale_to_unit))
        try:
            yield
        finally:
            self.ds_cfg, self.normalize_stats, self.chn_ids, self._mean, self._std, self.scale_to_unit = original

    def _load_image(self, path: str, *, column_name: str, sample_id: int, sensor_override: Optional[str] = None):
        row = self.df.iloc[sample_id]
        sensor = sensor_override if sensor_override is not None else row.get("sensor")
        if sensor not in self.sensor_configs:
            raise ValueError(f"Sample {sample_id} has unknown sensor '{sensor}'")
        with self._use_sensor_cfg(sensor):
            path_to_use = self._maybe_cache_path(path)
            return super(TriSensorTemporalCsvDataset, self)._load_image(
                path_to_use, column_name=column_name, sample_id=sample_id
            )

    def __getitem__(self, idx):
        attempts = 0
        last_exc: Optional[Exception] = None
        while attempts < self.max_retries:
            try:
                row = self.df.iloc[idx]
                label = int(row[self.label_column])

                sensor_samples: list[tuple[str, list[Dict[str, torch.Tensor]]]] = []
                for sensor_name in ("s2", "l89", "s5p", "wv3"):
                    cols = self._wide_sensor_columns.get(sensor_name)
                    if cols is None:
                        continue
                    if sensor_name == "s5p":
                        path_text = str(row.get(cols[0], "")).strip()
                        if path_text == "" or path_text.lower() == "nan":
                            continue
                        x_dict = self._load_s5p_sample(row, cols[0])
                        sensor_samples.append((sensor_name, [x_dict]))
                        continue

                    missing_path = False
                    for col in cols:
                        value = str(row.get(col, "")).strip()
                        if value == "" or value.lower() == "nan":
                            missing_path = True
                            break
                    if missing_path:
                        continue
                    x_list = [self._load_temporal_frame(row, col, sensor_name, idx) for col in cols]
                    sensor_samples.append((sensor_name, x_list))

                if not sensor_samples:
                    raise _SkipSample(f"No valid sensor paths found for wide-table row {idx}")
                return sensor_samples, label
            except _SkipSample as exc:
                last_exc = exc
                attempts += 1
                if attempts >= self.max_retries:
                    raise RuntimeError(f"Exceeded {self.max_retries} retries for temporal index {idx}") from exc
                idx = np.random.randint(0, len(self.df))
        raise RuntimeError("Unreachable") from last_exc

    def _load_temporal_frame(self, row, column_name: str, sensor: str, sample_id: int) -> Dict[str, torch.Tensor]:
        path = row[column_name]
        img = self._load_image(path, column_name=column_name, sample_id=sample_id, sensor_override=sensor)
        if sensor == "l89" and self._align_l89_to_s2:
            img = self._pad_l89_to_s2(img)

        # WV3 values can come in different scales across sources.
        # Auto-scale only when dynamic range is clearly raw-DN-like.
        if sensor == "wv3":
            abs_max = torch.amax(torch.abs(img))
            if torch.isfinite(abs_max) and abs_max.item() > 100.0:
                img = img / 65535.0

        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = torch.clamp(img, min=-50.0, max=50.0)
        x_dict = {
            "imgs": img,
            "chn_ids": self.sensor_configs[sensor]["chn_ids"],
        }
        if self.transform_each is not None:
            x_dict = self.transform_each(x_dict)
        return x_dict

    def _maybe_cache_path(self, path: str) -> str:
        if self._local_file_cache is not None and isinstance(path, str):
            return self._local_file_cache.ensure_local(path)
        return path

    def _load_s5p_sample(self, row, column_name: str) -> Dict[str, torch.Tensor]:
        path = row.get(column_name)
        if not isinstance(path, str) or not path:
            raise ValueError(f"S5P sample missing value in '{column_name}'")
        path = self._maybe_cache_path(path)
        chn_ids = None
        lower_path = path.lower()
        if lower_path.endswith((".tif", ".tiff")):
            raise ValueError(
                f"S5P input now expects NPZ files (e.g., .../s5p_0_path -> .npz), got TIFF: {path}"
            )
        else:
            try:
                np_obj = np.load(path, allow_pickle=False)
            except ValueError as exc:
                if "allow_pickle=False" in str(exc) or "pickled data" in str(exc):
                    np_obj = np.load(path, allow_pickle=True)
                else:
                    raise
            try:
                if isinstance(np_obj, np.lib.npyio.NpzFile):
                    img_np = self._extract_npz_array(np_obj, path)
                    if self._s5p_chn_ids_key is not None and self._s5p_chn_ids_key in np_obj:
                        chn_ids = torch.as_tensor(np_obj[self._s5p_chn_ids_key])
                else:
                    if isinstance(np_obj, np.ndarray) and np_obj.dtype == object and np_obj.size == 1:
                        obj = np_obj.item()
                        if isinstance(obj, Mapping):
                            if self._s5p_data_key is not None and self._s5p_data_key in obj:
                                img_np = np.asarray(obj[self._s5p_data_key])
                            else:
                                first_array = next(
                                    (v for v in obj.values() if isinstance(v, np.ndarray)),
                                    None,
                                )
                                if first_array is None:
                                    raise ValueError(f"Object array at {path} has no ndarray payload.")
                                img_np = np.asarray(first_array)
                            if self._s5p_chn_ids_key is not None and self._s5p_chn_ids_key in obj:
                                chn_ids = torch.as_tensor(obj[self._s5p_chn_ids_key])
                        else:
                            img_np = np.asarray(obj)
                    else:
                        img_np = np.array(np_obj)
            finally:
                if isinstance(np_obj, np.lib.npyio.NpzFile):
                    np_obj.close()

        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, 0)
        elif img_np.ndim == 3 and self._s5p_channels_last:
            img_np = np.transpose(img_np, (2, 0, 1))
        elif img_np.ndim == 3:
            c_first, c_last = img_np.shape[0], img_np.shape[-1]
            if c_first > 32 and c_last <= 16:
                img_np = np.transpose(img_np, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported S5P sample shape {img_np.shape} from {path}")

        img = torch.from_numpy(img_np).to(dtype=torch.float32)
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        expected_channels = int(self._s5p_mean.shape[0])
        if img.shape[0] != expected_channels:
            if img.shape[0] == 1 and expected_channels > 1:
                img = img.repeat(expected_channels, 1, 1)
            elif img.shape[0] > expected_channels:
                img = img[:expected_channels]
            else:
                repeat = math.ceil(expected_channels / max(1, img.shape[0]))
                img = img.repeat(repeat, 1, 1)[:expected_channels]
        img = (img - self._s5p_mean) / self._s5p_std
        if self.pad_to_multiple is not None:
            img = self._pad_to_multiple(img, self.pad_to_multiple)
        if chn_ids is None:
            chn_ids = torch.zeros((img.shape[0], 1), dtype=torch.float32)
        if chn_ids.ndim == 1:
            chn_ids = chn_ids.unsqueeze(-1)
        if chn_ids.shape[0] != img.shape[0]:
            if chn_ids.shape[0] == 1:
                chn_ids = chn_ids.repeat(img.shape[0], *([1] * (chn_ids.ndim - 1)))
            elif chn_ids.shape[0] > img.shape[0]:
                chn_ids = chn_ids[: img.shape[0]]
            else:
                repeat = math.ceil(img.shape[0] / max(1, chn_ids.shape[0]))
                chn_ids = chn_ids.repeat(repeat, *([1] * (chn_ids.ndim - 1)))[: img.shape[0]]
        x_dict = dict(imgs=img, chn_ids=chn_ids)
        if self.transform_each is not None:
            x_dict = self.transform_each(x_dict)
        return x_dict

    def _extract_npz_array(self, np_obj: np.lib.npyio.NpzFile, path: str) -> np.ndarray:
        if self._s5p_data_key is not None:
            if self._s5p_data_key not in np_obj:
                raise KeyError(f"Key '{self._s5p_data_key}' not found in NPZ file {path}")
            return np.array(np_obj[self._s5p_data_key])
        if len(np_obj.files) == 0:
            raise ValueError(f"No arrays found in NPZ file {path}")
        preferred_keys = ("ch4", "image", "imgs", "arr_0", "data")
        for key in preferred_keys:
            if key in np_obj:
                arr = np.array(np_obj[key])
                if self._looks_like_s5p_image(arr):
                    return arr

        for key in np_obj.files:
            # Skip obvious metadata keys when auto-selecting the image tensor.
            if key == self._s5p_chn_ids_key or key.lower() in {"meta", "metadata"}:
                continue
            try:
                arr = np.array(np_obj[key])
            except ValueError:
                continue
            if self._looks_like_s5p_image(arr):
                return arr

        raise ValueError(
            f"Failed to infer S5P image array from NPZ file {path}. "
            f"Available keys: {list(np_obj.files)}. Consider setting --s5p_data_key."
        )

    @staticmethod
    def _looks_like_s5p_image(arr: np.ndarray) -> bool:
        if not isinstance(arr, np.ndarray):
            return False
        if arr.dtype.kind not in {"f", "i", "u", "b"}:
            return False
        if arr.ndim not in (2, 3):
            return False
        if arr.ndim == 3 and min(arr.shape) <= 0:
            return False
        return True

    @staticmethod
    def _pad_l89_to_s2(img: torch.Tensor) -> torch.Tensor:
        if img.shape[0] != 7:
            return img
        device, dtype, h, w = img.device, img.dtype, img.shape[1], img.shape[2]
        out = torch.zeros((12, h, w), device=device, dtype=dtype)
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 7, 5: 10, 6: 11}
        for l89_idx, s2_idx in mapping.items():
            out[s2_idx] = img[l89_idx]
        return out


class ConcatTemporalDataset(Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        sensor_samples, label = sample
        flattened = []
        for sensor, x_list in sensor_samples:
            imgs = torch.cat([x["imgs"] for x in x_list], dim=0)
            chn_ids = torch.cat([x["chn_ids"] for x in x_list], dim=0)
            x_dict = dict(imgs=imgs, chn_ids=chn_ids)
            flattened.append((x_dict, label, sensor))
        return flattened


def custom_collate_fn(batch):
    x_dicts: list[Dict[str, torch.Tensor]] = []
    sensors: list[str] = []
    sample_to_row: list[int] = []
    row_labels: list[int] = []

    for row_idx, item in enumerate(batch):
        sensor_samples, label = item
        row_labels.append(int(label))
        for sensor, x_list in sensor_samples:
            imgs = torch.cat([x["imgs"] for x in x_list], dim=0)
            chn_ids = torch.cat([x["chn_ids"] for x in x_list], dim=0)
            x_dicts.append(dict(imgs=imgs, chn_ids=chn_ids))
            sensors.append(str(sensor))
            sample_to_row.append(int(row_idx))

    if len(x_dicts) == 0:
        raise ValueError("Batch has no valid sensor samples.")

    max_channels = max(x["imgs"].shape[0] for x in x_dicts)
    max_h = max(x["imgs"].shape[1] for x in x_dicts)
    max_w = max(x["imgs"].shape[2] for x in x_dicts)

    padded_imgs = []
    padded_chn_ids = []
    for x_dict in x_dicts:
        img = x_dict["imgs"]
        chn_ids = x_dict["chn_ids"]
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        img = F.pad(img, (0, pad_w, 0, pad_h))
        pad_c = max_channels - c
        if pad_c:
            img = torch.cat([img, torch.zeros((pad_c, max_h, max_w), dtype=img.dtype)], dim=0)
            chn_ids = torch.cat([chn_ids, torch.zeros((pad_c, *chn_ids.shape[1:]), dtype=chn_ids.dtype)], dim=0)
        padded_imgs.append(img)
        padded_chn_ids.append(chn_ids)
    batched_x_dict = {"imgs": torch.stack(padded_imgs), "chn_ids": torch.stack(padded_chn_ids)}
    return (
        batched_x_dict,
        torch.tensor(row_labels, dtype=torch.long),
        list(sensors),
        torch.tensor(sample_to_row, dtype=torch.long),
    )


def collect_cache_paths_from_df(df, columns: Sequence[str]) -> list[str]:
    valid_columns = [col for col in columns if col in df.columns]
    if len(valid_columns) == 0:
        return []
    out: list[str] = []
    for col in valid_columns:
        series = df[col]
        for value in series.tolist():
            if value is None:
                continue
            path = str(value).strip()
            if path == "" or path.lower() in ("nan", "none", "null"):
                continue
            out.append(path)
    return out


# --------------------------------------------------------------------------------------
#  Training utilities
# --------------------------------------------------------------------------------------

def _load_backbone(weights_path: Optional[str] = None, *, strict: bool = True) -> DinoVisionTransformer:
    from hubconf import _panopticon_vitb14

    backbone = _panopticon_vitb14()
    if weights_path in (None, "", "none", "scratch", "random"):
        return backbone
    ckpt_path = Path(weights_path)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, Mapping) and "backbone" in state:
        state = state["backbone"]
    backbone.load_state_dict(state, strict=strict)
    return backbone


def _index_batch(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, idx)


def _slice_x_dict(x_dict: MutableMapping[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: _index_batch(v, idx) for k, v in x_dict.items() if isinstance(v, torch.Tensor)}


def recursive_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: recursive_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [recursive_to_device(v, device) for v in x]
        return type(x)(t)
    return x


def set_trainable(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def init_wandb(args):
    if not args.use_wandb:
        return None
    import wandb

    return wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))


def default_run_name(args) -> str:
    train_stem = Path(args.train_csv).stem or "train"
    test_stem = Path(args.test_csv).stem or "test"
    mode = "ft" if args.train_backbone else "head"
    return f"{train_stem}__{test_stem}__{mode}"


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    best_train_acc: float,
    best_test_acc: float,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "best_train_acc": best_train_acc,
            "best_test_acc": best_test_acc,
            "args": vars(args),
        },
        path,
    )


def try_resume(path: Path, model: nn.Module, optimizer, scheduler, scaler, device):
    if not path.is_file():
        return 1, 0, 0.0, 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    global_step = ckpt.get("global_step", 0)
    best_train_acc = ckpt.get("best_train_acc", 0.0)
    best_test_acc = ckpt.get("best_test_acc", 0.0)
    print(f"Resumed from {path} at epoch {start_epoch-1}", flush=True)
    return start_epoch, global_step, best_train_acc, best_test_acc


def build_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "noam":
        warmup_steps = max(args.warmup_steps, 1)
        noam_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _binary_auroc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC for binary labels {0,1} using rank statistics (tie-aware)."""
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape[0] != scores.shape[0]:
        return float("nan")
    n = labels.shape[0]
    if n == 0:
        return float("nan")
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = float(ranks[pos_mask].sum())
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def compute_binary_metrics(labels: np.ndarray, preds: np.ndarray, pos_scores: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(np.int64, copy=False)
    preds = preds.astype(np.int64, copy=False)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {
        "fpr": _safe_ratio(fp, fp + tn),
        "recall": _safe_ratio(tp, tp + fn),
        "auroc": _binary_auroc_from_scores(labels, pos_scores),
    }


def compute_split_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
    pos_scores: Sequence[float],
) -> Dict[str, float]:
    count = int(len(labels))
    out: Dict[str, float] = {
        "count": float(count),
        "acc": float("nan"),
        "fpr": float("nan"),
        "recall": float("nan"),
        "auroc": float("nan"),
    }
    if count == 0 or len(preds) != count:
        return out
    labels_np = np.asarray(labels, dtype=np.int64)
    preds_np = np.asarray(preds, dtype=np.int64)
    out["acc"] = float((labels_np == preds_np).mean())
    if len(pos_scores) == count:
        out.update(
            compute_binary_metrics(
                labels=labels_np,
                preds=preds_np,
                pos_scores=np.asarray(pos_scores, dtype=np.float64),
            )
        )
    return out


def mean_logits_by_row(
    logits: torch.Tensor,
    sample_rows: torch.Tensor,
    *,
    num_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be [N,C], got shape={tuple(logits.shape)}")
    if sample_rows.ndim != 1 or sample_rows.shape[0] != logits.shape[0]:
        raise ValueError(
            "sample_rows must be rank-1 and match logits length: "
            f"sample_rows={tuple(sample_rows.shape)}, logits={tuple(logits.shape)}"
        )
    logits_f32 = logits.float()
    num_classes = logits_f32.shape[-1]
    row_sums = torch.zeros((num_rows, num_classes), device=logits.device, dtype=torch.float32)
    row_counts = torch.zeros((num_rows, 1), device=logits.device, dtype=torch.float32)
    row_sums.index_add_(0, sample_rows, logits_f32)
    ones = torch.ones((sample_rows.shape[0], 1), device=logits.device, dtype=torch.float32)
    row_counts.index_add_(0, sample_rows, ones)
    valid_mask = row_counts.squeeze(1) > 0
    row_ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    if row_ids.numel() == 0:
        return row_ids, row_sums.new_zeros((0, num_classes))
    row_logits = row_sums.index_select(0, row_ids) / row_counts.index_select(0, row_ids).clamp_min(1.0)
    return row_ids, row_logits


def gather_head_parameters(model: MultiSensorPanopticonClassifier) -> Sequence[nn.Parameter]:
    shared_patch = model.backbone.patch_embed
    extra_patch_params = []
    for sensor, module in model.sensor_patch_embeds.items():
        if module is shared_patch:
            continue
        extra_patch_params.extend(list(module.parameters()))
    row_fusion_params = []
    if model.row_fusion_pool is not None:
        row_fusion_params.extend(list(model.row_fusion_pool.parameters()))
    if model.row_fusion_head is not None:
        row_fusion_params.extend(list(model.row_fusion_head.parameters()))
    return row_fusion_params + extra_patch_params


# --------------------------------------------------------------------------------------
#  CLI / training loop
# --------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-sensor Panopticon finetuning (S2/L89/S5P/WV3).")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--weights", default="weights/panopticon_vitb14_teacher.pth")
    parser.add_argument(
        "--fusion_group_column",
        default="id",
        help="Deprecated and ignored. Group-based fusion is disabled; wide-table rows are treated as final units.",
    )
    parser.add_argument(
        "--s5p_data_key",
        default="ch4",
        help="NPZ key for S5P image tensor (defaults to 'ch4' for old 2025 S5P patches).",
    )
    parser.add_argument("--s5p_chn_ids_key", default="chn_ids")
    parser.add_argument("--s5p_channels_last", action="store_true")
    parser.add_argument(
        "--wv3_srf_csv",
        default=str(REPO_ROOT / "WV3_VNIR_SWIR_response.csv"),
        help="Path to WV3 SRF CSV used to derive per-band channel IDs.",
    )
    parser.add_argument(
        "--wv3_bands",
        default=",".join(DEFAULT_WV3_BANDS),
        help="Comma-separated WV3 SRF column names in channel order.",
    )
    parser.add_argument(
        "--align_l89_to_s2",
        action="store_true",
        help="Pad L89 to 12 channels and reuse S2 channel ids (legacy behavior).",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_dir", default="checkpoints/multi_sensor")
    parser.add_argument("--lr_scheduler", choices=["none", "noam"], default="noam")
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="panopticon-multisensor")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--local_cache_dir", default=None)
    parser.add_argument("--local_cache_min_free_gb", type=float, default=5.0)
    parser.add_argument("--local_cache_warmup", action="store_true")
    parser.add_argument("--local_cache_workers", type=int, default=8)
    parser.add_argument(
        "--contrastive_temperature",
        type=float,
        default=0.1,
        help="Temperature for CLS contrastive learning across active sensors.",
    )
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    wv3_band_names = [b.strip() for b in args.wv3_bands.split(",") if b.strip()]
    if len(wv3_band_names) == 0:
        raise ValueError("--wv3_bands must provide at least one WV3 band column name")
    wv3_chn_ids = load_wv3_channel_ids_from_srf(args.wv3_srf_csv, wv3_band_names).unsqueeze(-1)

    cache_obj = None
    if args.local_cache_dir:
        cache_obj = StaticAnchoredCache(args.local_cache_dir, min_free_gb=args.local_cache_min_free_gb)
        print(
            f"[Cache] Enabled local cache dir={cache_obj.cache_dir} (min_free_gb={args.local_cache_min_free_gb:.2f})",
            flush=True,
        )
        if not args.local_cache_warmup:
            print("[Cache] Warmup disabled; cache will populate lazily on first file access.", flush=True)

    base_train_ds = TriSensorTemporalCsvDataset(
        csv_path=args.train_csv,
        local_file_cache=cache_obj,
        s5p_data_key=args.s5p_data_key,
        s5p_chn_ids_key=args.s5p_chn_ids_key,
        s5p_channels_last=args.s5p_channels_last,
        align_l89_to_s2=args.align_l89_to_s2,
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )
    base_test_ds = TriSensorTemporalCsvDataset(
        csv_path=args.test_csv,
        local_file_cache=cache_obj,
        s5p_data_key=args.s5p_data_key,
        s5p_chn_ids_key=args.s5p_chn_ids_key,
        s5p_channels_last=args.s5p_channels_last,
        align_l89_to_s2=args.align_l89_to_s2,
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )

    if args.local_cache_warmup and cache_obj:
        warmup_columns: set[str] = set(col for col in base_train_ds.df.columns if str(col).endswith("_path"))
        wide_sensor_columns = getattr(base_train_ds, "_wide_sensor_columns", {})
        if isinstance(wide_sensor_columns, Mapping):
            for cols in wide_sensor_columns.values():
                warmup_columns.update(str(col) for col in cols if col in base_train_ds.df.columns)
        all_paths = collect_cache_paths_from_df(base_train_ds.df, sorted(warmup_columns))
        print(
            f"[Cache] Warmup scanning {len(warmup_columns)} path columns from train CSV, "
            f"found {len(all_paths)} path entries.",
            flush=True,
        )
        cache_obj.warm_up(all_paths, max_workers=args.local_cache_workers)

    train_ds = base_train_ds
    test_ds = base_test_ds

    core_model = MultiSensorPanopticonClassifier(
        backbone=_load_backbone(args.weights),
        contrastive_temperature=args.contrastive_temperature,
    ).to(device)
    model: nn.Module = core_model
    use_data_parallel = args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1
    if use_data_parallel:
        print(f"Enabling DataParallel across {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(core_model)
    elif args.data_parallel:
        print("DataParallel requested but insufficient CUDA devices; running single-device.", flush=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    head_params = gather_head_parameters(core_model)
    param_groups = [
        {"params": head_params, "lr": args.head_lr},
    ]
    if args.train_backbone:
        param_groups.insert(0, {"params": core_model.backbone.parameters(), "lr": args.backbone_lr})

    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay, betas=(args.momentum, 0.999))
    scheduler = build_scheduler(args, optimizer)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    run_name = args.wandb_run_name or default_run_name(args)
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    latest_path = ckpt_dir / "ckpt_latest.pth"
    best_path = ckpt_dir / "ckpt_best_test.pth"
    print(f"Checkpoints will be saved under: {ckpt_dir}", flush=True)

    start_epoch = 1
    global_step = 0
    best_train_acc = 0.0
    best_test_acc = float("-inf")

    if args.resume:
        start_epoch, global_step, best_train_acc, best_test_acc = try_resume(
            latest_path, core_model, optimizer, scheduler, scaler, device
        )

    wandb_run = init_wandb(args)

    sensors_list = core_model.sensor_order
    deprecated_opts = []
    if str(args.fusion_group_column).strip() != "id":
        deprecated_opts.append("--fusion_group_column")
    if deprecated_opts:
        warnings.warn(
            "Deprecated options are ignored in this script: "
            + ", ".join(sorted(set(deprecated_opts))),
            stacklevel=2,
        )
    if args.contrastive_temperature <= 0.0:
        raise ValueError(f"--contrastive_temperature must be > 0, got {args.contrastive_temperature}.")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
    )
    train_steps = max(1, len(train_loader))
    test_steps = max(1, len(test_loader))

    print(
        f"Using device={device}, train_samples={len(train_ds)}, test_samples={len(test_ds)}, "
        f"sensors={sensors_list}, train_backbone={args.train_backbone}, "
        f"fusion=masked_attention_pooling, "
        f"contrastive_temperature={args.contrastive_temperature:.3f}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        freeze_backbone = (not args.train_backbone) or (
            args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs
        )
        if freeze_backbone:
            set_trainable(core_model.backbone, False)
            core_model.backbone.eval()
        else:
            set_trainable(core_model.backbone, True)
            core_model.backbone.train()
        model.train()
        if core_model.row_fusion_pool is not None:
            core_model.row_fusion_pool.train()
        if core_model.row_fusion_head is not None:
            core_model.row_fusion_head.train()
        for sensor in core_model.sensor_patch_embeds.values():
            sensor.train()

        total_loss = 0.0
        total = 0
        correct = 0
        fused_loss_accum = 0.0
        fused_loss_count = 0
        contrastive_loss_accum = 0.0
        contrastive_loss_count = 0

        step_idx = 0
        for x_dict, labels, sensors, sample_to_row in train_loader:
            step_idx += 1
            labels = labels.to(device)
            sample_to_row = sample_to_row.to(device=device, dtype=torch.long)
            x_dict = recursive_to_device(x_dict, device)
            sensor_arg: Union[Sequence[str], torch.Tensor]
            if use_data_parallel:
                sensor_arg = core_model.encode_sensors(sensors, device=device)
            else:
                sensor_arg = sensors
            with autocast(enabled=use_amp):
                outputs = model(x_dict, sensors=sensor_arg)
                fused_logits = core_model.compute_row_fused_logits(
                    outputs,
                    sample_to_row=sample_to_row,
                    num_rows=labels.size(0),
                    device=device,
                )
                fused_loss = criterion(fused_logits, labels)
                contrastive_loss = core_model.compute_contrastive_loss(outputs, sample_to_row)
                loss = fused_loss + contrastive_loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(param_groups[0]["params"], args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            batch_rows = labels.size(0)
            total_loss += loss.item() * batch_rows
            total += batch_rows
            preds = fused_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            fused_loss_accum += fused_loss.item()
            fused_loss_count += 1
            contrastive_loss_accum += contrastive_loss.item()
            contrastive_loss_count += 1

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break
            if args.log_interval and step_idx % args.log_interval == 0:
                print(
                    f"Epoch {epoch} step {step_idx}/{train_steps} train_loss={total_loss/total:.4f} "
                    f"train_acc={correct/total:.4f} "
                    f"fused_loss={fused_loss.item():.4f} contrastive_loss={contrastive_loss.item():.4f}",
                    flush=True,
                )

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        test_loss_total = 0.0
        total_eval = 0
        correct_eval = 0
        eval_steps = 0
        fused_all_labels: list[int] = []
        fused_all_preds: list[int] = []
        fused_all_pos_scores: list[float] = []
        fused_overlap_labels: list[int] = []
        fused_overlap_preds: list[int] = []
        fused_overlap_pos_scores: list[float] = []
        fused_single_labels: list[int] = []
        fused_single_preds: list[int] = []
        fused_single_pos_scores: list[float] = []

        fused_all_sensor_labels: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        fused_all_sensor_preds: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        fused_all_sensor_pos_scores: Dict[str, list[float]] = {sensor: [] for sensor in sensors_list}
        fused_overlap_sensor_labels: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        fused_overlap_sensor_preds: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        fused_overlap_sensor_pos_scores: Dict[str, list[float]] = {sensor: [] for sensor in sensors_list}
        fused_single_sensor_labels: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        fused_single_sensor_preds: Dict[str, list[int]] = {sensor: [] for sensor in sensors_list}
        fused_single_sensor_pos_scores: Dict[str, list[float]] = {sensor: [] for sensor in sensors_list}

        sensor_to_idx_eval = {sensor: idx for idx, sensor in enumerate(sensors_list)}
        with torch.no_grad():
            for step, (x_dict, labels, sensors, sample_to_row) in enumerate(test_loader, 1):
                labels = labels.to(device)
                sample_to_row = sample_to_row.to(device=device, dtype=torch.long)
                x_dict = recursive_to_device(x_dict, device)
                if use_data_parallel:
                    sensor_arg = core_model.encode_sensors(sensors, device=device)
                else:
                    sensor_arg = sensors
                with autocast(enabled=use_amp):
                    outputs = model(x_dict, sensors=sensor_arg)
                    fused_logits = core_model.compute_row_fused_logits(
                        outputs,
                        sample_to_row=sample_to_row,
                        num_rows=labels.size(0),
                        device=device,
                    )
                    fused_loss = criterion(fused_logits, labels)
                    contrastive_loss = core_model.compute_contrastive_loss(outputs, sample_to_row)
                    loss = fused_loss + contrastive_loss
                batch_rows = labels.size(0)
                test_loss_total += loss.item() * batch_rows
                total_eval += batch_rows
                preds = fused_logits.argmax(dim=1)
                correct_eval += (preds == labels).sum().item()
                pos_scores: Optional[torch.Tensor] = None
                if fused_logits.shape[-1] == 2:
                    pos_scores = torch.softmax(fused_logits, dim=1)[:, 1]
                row_has_sensor = torch.zeros((batch_rows, len(sensors_list)), dtype=torch.bool, device=device)
                for sample_idx, sensor_name in enumerate(sensors):
                    sensor_idx = sensor_to_idx_eval.get(str(sensor_name))
                    if sensor_idx is None:
                        continue
                    row_idx = int(sample_to_row[sample_idx].item())
                    row_has_sensor[row_idx, sensor_idx] = True
                row_sensor_count = row_has_sensor.sum(dim=1)
                overlap_mask = row_sensor_count >= 2
                single_mask = row_sensor_count == 1

                def _append_rows(mask: torch.Tensor, label_bucket: list[int], pred_bucket: list[int], score_bucket: list[float]) -> None:
                    if not torch.any(mask):
                        return
                    row_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    label_bucket.extend(labels.index_select(0, row_ids).detach().to("cpu").tolist())
                    pred_bucket.extend(preds.index_select(0, row_ids).detach().to("cpu").tolist())
                    if pos_scores is not None:
                        score_bucket.extend(pos_scores.index_select(0, row_ids).detach().to("cpu").tolist())

                _append_rows(torch.ones((batch_rows,), dtype=torch.bool, device=device), fused_all_labels, fused_all_preds, fused_all_pos_scores)
                _append_rows(overlap_mask, fused_overlap_labels, fused_overlap_preds, fused_overlap_pos_scores)
                _append_rows(single_mask, fused_single_labels, fused_single_preds, fused_single_pos_scores)

                for sensor_idx, sensor_name in enumerate(sensors_list):
                    sensor_mask = row_has_sensor[:, sensor_idx]
                    _append_rows(
                        sensor_mask,
                        fused_all_sensor_labels[sensor_name],
                        fused_all_sensor_preds[sensor_name],
                        fused_all_sensor_pos_scores[sensor_name],
                    )
                    _append_rows(
                        sensor_mask & overlap_mask,
                        fused_overlap_sensor_labels[sensor_name],
                        fused_overlap_sensor_preds[sensor_name],
                        fused_overlap_sensor_pos_scores[sensor_name],
                    )
                    _append_rows(
                        sensor_mask & single_mask,
                        fused_single_sensor_labels[sensor_name],
                        fused_single_sensor_preds[sensor_name],
                        fused_single_sensor_pos_scores[sensor_name],
                    )
                eval_steps += 1
                if args.max_eval_steps is not None and eval_steps >= args.max_eval_steps:
                    break

        test_loss = test_loss_total / max(1, total_eval)
        test_acc = correct_eval / max(1, total_eval)
        fused_all_overall_metrics = compute_split_metrics(fused_all_labels, fused_all_preds, fused_all_pos_scores)
        fused_overlap_overall_metrics = compute_split_metrics(
            fused_overlap_labels,
            fused_overlap_preds,
            fused_overlap_pos_scores,
        )
        fused_single_overall_metrics = compute_split_metrics(
            fused_single_labels,
            fused_single_preds,
            fused_single_pos_scores,
        )
        fused_all_sensor_metrics = {
            sensor: compute_split_metrics(
                fused_all_sensor_labels[sensor],
                fused_all_sensor_preds[sensor],
                fused_all_sensor_pos_scores[sensor],
            )
            for sensor in sensors_list
        }
        fused_overlap_sensor_metrics = {
            sensor: compute_split_metrics(
                fused_overlap_sensor_labels[sensor],
                fused_overlap_sensor_preds[sensor],
                fused_overlap_sensor_pos_scores[sensor],
            )
            for sensor in sensors_list
        }
        fused_single_sensor_metrics = {
            sensor: compute_split_metrics(
                fused_single_sensor_labels[sensor],
                fused_single_sensor_preds[sensor],
                fused_single_sensor_pos_scores[sensor],
            )
            for sensor in sensors_list
        }
        test_fpr = fused_all_overall_metrics["fpr"]
        test_recall = fused_all_overall_metrics["recall"]
        test_auroc = fused_all_overall_metrics["auroc"]
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"test_fpr={test_fpr:.4f} test_recall={test_recall:.4f} test_auroc={test_auroc:.4f} "
            f"train_fused_loss={fused_loss_accum/max(1,fused_loss_count):.4f} "
            f"train_contrastive_loss={contrastive_loss_accum/max(1,contrastive_loss_count):.4f}",
            flush=True,
        )
        print(
            f"[EvalSplit][FUSED][ALL][OVERALL] count={int(fused_all_overall_metrics['count'])} "
            f"acc={fused_all_overall_metrics['acc']:.4f} fpr={fused_all_overall_metrics['fpr']:.4f} "
            f"recall={fused_all_overall_metrics['recall']:.4f} auroc={fused_all_overall_metrics['auroc']:.4f}",
            flush=True,
        )
        print(
            f"[EvalSplit][FUSED][OVERLAP][OVERALL] count={int(fused_overlap_overall_metrics['count'])} "
            f"acc={fused_overlap_overall_metrics['acc']:.4f} fpr={fused_overlap_overall_metrics['fpr']:.4f} "
            f"recall={fused_overlap_overall_metrics['recall']:.4f} auroc={fused_overlap_overall_metrics['auroc']:.4f}",
            flush=True,
        )
        print(
            f"[EvalSplit][FUSED][SINGLE][OVERALL] count={int(fused_single_overall_metrics['count'])} "
            f"acc={fused_single_overall_metrics['acc']:.4f} fpr={fused_single_overall_metrics['fpr']:.4f} "
            f"recall={fused_single_overall_metrics['recall']:.4f} auroc={fused_single_overall_metrics['auroc']:.4f}",
            flush=True,
        )
        for sensor in sensors_list:
            m_all = fused_all_sensor_metrics[sensor]
            m_overlap = fused_overlap_sensor_metrics[sensor]
            m_single = fused_single_sensor_metrics[sensor]
            print(
                f"[EvalSplit][FUSED][ALL][SENSOR_{sensor.upper()}] count={int(m_all['count'])} "
                f"acc={m_all['acc']:.4f} fpr={m_all['fpr']:.4f} "
                f"recall={m_all['recall']:.4f} auroc={m_all['auroc']:.4f}",
                flush=True,
            )
            print(
                f"[EvalSplit][FUSED][OVERLAP][SENSOR_{sensor.upper()}] count={int(m_overlap['count'])} "
                f"acc={m_overlap['acc']:.4f} fpr={m_overlap['fpr']:.4f} "
                f"recall={m_overlap['recall']:.4f} auroc={m_overlap['auroc']:.4f}",
                flush=True,
            )
            print(
                f"[EvalSplit][FUSED][SINGLE][SENSOR_{sensor.upper()}] count={int(m_single['count'])} "
                f"acc={m_single['acc']:.4f} fpr={m_single['fpr']:.4f} "
                f"recall={m_single['recall']:.4f} auroc={m_single['auroc']:.4f}",
                flush=True,
            )

        prev_best_test = best_test_acc
        best_train_acc = max(best_train_acc, train_acc)
        best_test_acc = max(best_test_acc, test_acc)

        save_checkpoint(
            latest_path,
            epoch,
            global_step,
            core_model,
            optimizer,
            scheduler,
            scaler if use_amp else None,
            best_train_acc,
            best_test_acc,
            args,
        )
        if test_acc > prev_best_test:
            save_checkpoint(
                best_path,
                epoch,
                global_step,
                core_model,
                optimizer,
                scheduler,
                scaler if use_amp else None,
                best_train_acc,
                best_test_acc,
                args,
            )
            print(f"Saved new best checkpoint: {best_path} (test_acc={test_acc:.4f})", flush=True)
        if wandb_run is not None:
            log_payload = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_fpr": test_fpr,
                "test_recall": test_recall,
                "test_auroc": test_auroc,
                "test_fused_all_overall_count": fused_all_overall_metrics["count"],
                "test_fused_all_overall_acc": fused_all_overall_metrics["acc"],
                "test_fused_all_overall_fpr": fused_all_overall_metrics["fpr"],
                "test_fused_all_overall_recall": fused_all_overall_metrics["recall"],
                "test_fused_all_overall_auroc": fused_all_overall_metrics["auroc"],
                "test_fused_overlap_overall_count": fused_overlap_overall_metrics["count"],
                "test_fused_overlap_overall_acc": fused_overlap_overall_metrics["acc"],
                "test_fused_overlap_overall_fpr": fused_overlap_overall_metrics["fpr"],
                "test_fused_overlap_overall_recall": fused_overlap_overall_metrics["recall"],
                "test_fused_overlap_overall_auroc": fused_overlap_overall_metrics["auroc"],
                "test_fused_single_overall_count": fused_single_overall_metrics["count"],
                "test_fused_single_overall_acc": fused_single_overall_metrics["acc"],
                "test_fused_single_overall_fpr": fused_single_overall_metrics["fpr"],
                "test_fused_single_overall_recall": fused_single_overall_metrics["recall"],
                "test_fused_single_overall_auroc": fused_single_overall_metrics["auroc"],
                "train_fused_loss": fused_loss_accum / max(1, fused_loss_count),
                "train_contrastive_loss": contrastive_loss_accum / max(1, contrastive_loss_count),
            }
            for sensor in sensors_list:
                fused_all_sensor = fused_all_sensor_metrics[sensor]
                fused_overlap_sensor = fused_overlap_sensor_metrics[sensor]
                fused_single_sensor = fused_single_sensor_metrics[sensor]
                log_payload[f"test_fused_all_sensor_{sensor}_count"] = fused_all_sensor["count"]
                log_payload[f"test_fused_all_sensor_{sensor}_acc"] = fused_all_sensor["acc"]
                log_payload[f"test_fused_all_sensor_{sensor}_fpr"] = fused_all_sensor["fpr"]
                log_payload[f"test_fused_all_sensor_{sensor}_recall"] = fused_all_sensor["recall"]
                log_payload[f"test_fused_all_sensor_{sensor}_auroc"] = fused_all_sensor["auroc"]
                log_payload[f"test_fused_overlap_sensor_{sensor}_count"] = fused_overlap_sensor["count"]
                log_payload[f"test_fused_overlap_sensor_{sensor}_acc"] = fused_overlap_sensor["acc"]
                log_payload[f"test_fused_overlap_sensor_{sensor}_fpr"] = fused_overlap_sensor["fpr"]
                log_payload[f"test_fused_overlap_sensor_{sensor}_recall"] = fused_overlap_sensor["recall"]
                log_payload[f"test_fused_overlap_sensor_{sensor}_auroc"] = fused_overlap_sensor["auroc"]
                log_payload[f"test_fused_single_sensor_{sensor}_count"] = fused_single_sensor["count"]
                log_payload[f"test_fused_single_sensor_{sensor}_acc"] = fused_single_sensor["acc"]
                log_payload[f"test_fused_single_sensor_{sensor}_fpr"] = fused_single_sensor["fpr"]
                log_payload[f"test_fused_single_sensor_{sensor}_recall"] = fused_single_sensor["recall"]
                log_payload[f"test_fused_single_sensor_{sensor}_auroc"] = fused_single_sensor["auroc"]
            wandb_run.log(log_payload)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
