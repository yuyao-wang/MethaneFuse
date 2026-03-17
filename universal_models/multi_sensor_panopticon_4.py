"""Multi-sensor Panopticon training script with sensor-specific embeddings and heads.

This module exposes reusable model components (``MultiSensorPanopticonClassifier``)
*and* a runnable training entry point that ingests mixed-sensor CSVs containing
Sentinel-2, Landsat 8/9, Sentinel-5P, and WV3 samples. Each sensor owns its own
Panopticon patch embedding and classifier head, while the DinoViT backbone is
shared and updated by the consensus of all datasets in a batch.
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
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset, SubsetRandomSampler

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


HeadFactory = Callable[[int, int], nn.Module]


class MultiSensorPanopticonClassifier(nn.Module):
    """Shared DinoViT backbone with sensor-specific Panopticon PEs and heads."""

    def __init__(
        self,
        *,
        backbone: Optional[DinoVisionTransformer] = None,
        sensors: Sequence[str] = ("s2", "l89", "s5p", "wv3"),
        num_classes: Mapping[str, int] | int = 2,
        patch_embed_overrides: Optional[Mapping[str, PanopticonPE]] = None,
        head_factory: Optional[Union[HeadFactory, nn.Module]] = None,
        enable_summary_head: bool = True,
        summary_hidden_dim: int = 128,
        summary_dropout: float = 0.1,
        summary_loss_weight: float = 1.0,
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

        embed_dim = getattr(self.backbone, "embed_dim", 768)
        if isinstance(num_classes, int):
            class_map = {sensor: num_classes for sensor in self.sensor_order}
        else:
            class_map = {sensor: num_classes[sensor] for sensor in self.sensor_order}

        def make_head(num_output_classes: int) -> nn.Module:
            if head_factory is None:
                return CLSHead(embed_dim=embed_dim, num_classes=num_output_classes)
            if isinstance(head_factory, nn.Module):
                return copy.deepcopy(head_factory)
            head = head_factory(embed_dim, num_output_classes)
            if not isinstance(head, nn.Module):
                raise TypeError("head_factory must create an nn.Module")
            return head

        self.heads = nn.ModuleDict()
        for sensor in self.sensor_order:
            classes = class_map[sensor]
            self.heads[sensor] = make_head(classes)
        self.summary_loss_weight = float(summary_loss_weight)
        self.summary_head: Optional[LogitSummaryHead] = None

        head_dims = [getattr(self.heads[sensor], "out_features", None) for sensor in self.sensor_order]
        can_build_summary = bool(head_dims) and None not in head_dims and len(set(head_dims)) == 1
        if enable_summary_head and can_build_summary:
            num_out_classes = int(head_dims[0])
            self.summary_head = LogitSummaryHead(
                num_heads=len(self.sensor_order),
                num_classes=num_out_classes,
                hidden_dim=summary_hidden_dim,
                dropout=summary_dropout,
            )
        elif enable_summary_head and not can_build_summary:
            warnings.warn(
                "Summary head disabled because sensor heads do not share the same out_features.",
                stacklevel=2,
            )

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
        participating = list(sensor_batches.keys())
        merge_dims = [getattr(self.heads[sensor_name], "out_features", None) for sensor_name in participating]
        allow_merge = bool(merge_dims) and None not in merge_dims and len(set(merge_dims)) == 1
        merged_logits: Optional[torch.Tensor] = None
        batch_size = len(sensors)
        summary_inputs: Optional[torch.Tensor] = None

        for sensor_name, sensor_batch in sensor_batches.items():
            with self._use_sensor(sensor_name):
                feats = self.backbone(sensor_batch.x_dict, is_training=True)
            cls_token = torch.nan_to_num(feats["x_norm_clstoken"], nan=0.0, posinf=1e4, neginf=-1e4)
            head_logits = [
                torch.nan_to_num(self.heads[name](cls_token), nan=0.0, posinf=1e4, neginf=-1e4)
                for name in self.sensor_order
            ]
            logits = head_logits[self.sensor_to_idx[sensor_name]]
            if allow_merge:
                if merged_logits is None:
                    merged_logits = logits.new_zeros((batch_size, logits.shape[-1]))
                merged_logits.index_copy_(0, sensor_batch.indices, logits)
            if self.summary_head is not None:
                if summary_inputs is None:
                    summary_inputs = logits.new_zeros((batch_size, len(self.sensor_order) * logits.shape[-1]))
                summary_inputs.index_copy_(0, sensor_batch.indices, torch.cat(head_logits, dim=-1))
            outputs[sensor_name] = {
                "indices": sensor_batch.indices,
                "cls_token": cls_token,
                "logits": logits,
            }
            if return_features:
                outputs[sensor_name]["feats"] = feats["x_norm_patchtokens"]

        if merged_logits is not None:
            outputs["merged_logits"] = merged_logits
        if summary_inputs is not None and self.summary_head is not None:
            outputs["summary_logits"] = self.summary_head(summary_inputs)
        self._ensure_sensor_keys(outputs, x_dict, return_features=return_features)
        return outputs

    def compute_loss(
        self,
        x_dict: MutableMapping[str, torch.Tensor],
        sensors: Sequence[str],
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]]:
        sensor_labels = self._normalize_sensors(sensors)
        outputs = self.forward(x_dict, sensors=sensor_labels)
        total_loss, per_sensor_losses = self._loss_from_outputs(outputs, labels, sensor_labels, criterion)
        return total_loss, per_sensor_losses, outputs

    def loss_from_outputs(
        self,
        outputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        labels: torch.Tensor,
        sensors: Union[Sequence[str], torch.Tensor],
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sensor_labels = self._normalize_sensors(sensors)
        return self._loss_from_outputs(outputs, labels, sensor_labels, criterion)

    def _loss_from_outputs(
        self,
        outputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        labels: torch.Tensor,
        sensors: Sequence[str],
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss: Optional[torch.Tensor] = None
        per_sensor_losses: Dict[str, torch.Tensor] = {}
        for sensor_name in self.sensor_order:
            sensor_out = outputs.get(sensor_name)
            if not isinstance(sensor_out, dict):
                continue
            idx = sensor_out["indices"]
            if idx.numel() == 0:
                continue
            sensor_labels = labels.index_select(0, idx)
            loss = criterion(sensor_out["logits"], sensor_labels)
            per_sensor_losses[sensor_name] = loss
            total_loss = loss if total_loss is None else total_loss + loss
        summary_logits = outputs.get("summary_logits")
        if isinstance(summary_logits, torch.Tensor):
            summary_loss = criterion(summary_logits, labels)
            per_sensor_losses["summary"] = summary_loss
            weighted_summary_loss = summary_loss * self.summary_loss_weight
            total_loss = weighted_summary_loss if total_loss is None else total_loss + weighted_summary_loss
        if total_loss is None:
            raise RuntimeError("No loss terms were computed; check the sensor labels")
        return total_loss, per_sensor_losses

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
            head = self.heads[sensor_name]
            num_classes = head.out_features
            empty_idx = torch.empty((0,), dtype=torch.long, device=device)
            empty_cls = torch.empty((0, embed_dim), device=device)
            empty_logits = torch.empty((0, num_classes), device=device)
            placeholder: Dict[str, torch.Tensor] = {
                "indices": empty_idx,
                "cls_token": empty_cls,
                "logits": empty_logits,
            }
            if return_features:
                placeholder["feats"] = torch.empty((0, 0), device=device)
            outputs[sensor_name] = placeholder


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

    def _get_free_space(self) -> int:
        return shutil.disk_usage(self.cache_dir).free

    def _hashed_path(self, original: str) -> Path:
        norm_path = os.path.abspath(original)
        digest = hashlib.sha1(norm_path.encode("utf-8")).hexdigest()
        subdir = digest[:2]
        suffix = Path(original).suffix
        return self.cache_dir / subdir / f"{digest}{suffix}"

    def ensure_local(self, original: str) -> str:
        dst = self._hashed_path(original)
        if dst.exists():
            return str(dst)
        if self._get_free_space() < self.min_free_bytes:
            return original
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(original, tmp)
            os.replace(tmp, dst)
        except Exception:
            with suppress(FileNotFoundError):
                tmp.unlink()
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
        s5p_data_key: Optional[str] = None,
        s5p_chn_ids_key: Optional[str] = "chn_ids",
        s5p_channels_last: bool = False,
        align_l89_to_s2: bool = False,
        wv3_chn_ids: Optional[torch.Tensor] = None,
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
        if "sensor" not in self.df.columns:
            raise ValueError("CSV must contain a 'sensor' column with values in {'s2','l89','s5p','wv3'}")

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

    def _load_image(self, path: str, *, column_name: str, sample_id: int):
        row = self.df.iloc[sample_id]
        sensor = row.get("sensor")
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
                sensor = row.get("sensor")
                if sensor is None:
                    raise ValueError(f"Row {idx} is missing 'sensor' value")
                label = int(row[self.label_column])

                if sensor == "s5p":
                    x_dict = self._load_s5p_sample(row, self.path_columns[0])
                    return [x_dict], label, sensor

                if sensor not in self.sensor_configs:
                    raise ValueError(f"Sample {idx} has unknown sensor '{sensor}'")

                x_list = [self._load_temporal_frame(row, col, sensor, idx) for col in self.path_columns]
                return x_list, label, sensor
            except _SkipSample as exc:
                last_exc = exc
                attempts += 1
                if attempts >= self.max_retries:
                    raise RuntimeError(f"Exceeded {self.max_retries} retries for temporal index {idx}") from exc
                idx = np.random.randint(0, len(self.df))
        raise RuntimeError("Unreachable") from last_exc

    def _load_temporal_frame(self, row, column_name: str, sensor: str, sample_id: int) -> Dict[str, torch.Tensor]:
        path = row[column_name]
        img = self._load_image(path, column_name=column_name, sample_id=sample_id)
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
        np_obj = np.load(path, allow_pickle=False)
        chn_ids = None
        try:
            if isinstance(np_obj, np.lib.npyio.NpzFile):
                img_np = self._extract_npz_array(np_obj, path)
                if self._s5p_chn_ids_key is not None and self._s5p_chn_ids_key in np_obj:
                    chn_ids = torch.as_tensor(np_obj[self._s5p_chn_ids_key])
            else:
                img_np = np.array(np_obj)
        finally:
            if isinstance(np_obj, np.lib.npyio.NpzFile):
                np_obj.close()
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, 0)
        elif img_np.ndim == 3 and self._s5p_channels_last:
            img_np = np.transpose(img_np, (2, 0, 1))
        img = torch.from_numpy(img_np).to(dtype=torch.float32)
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        if img.shape[0] != self._s5p_mean.shape[0]:
            repeat = self._s5p_mean.shape[0] // max(1, img.shape[0])
            img = img.repeat(repeat, 1, 1)
        img = (img - self._s5p_mean) / self._s5p_std
        if self.pad_to_multiple is not None:
            img = self._pad_to_multiple(img, self.pad_to_multiple)
        if chn_ids is None:
            chn_ids = torch.zeros((img.shape[0], 1), dtype=torch.float32)
        if chn_ids.ndim == 1:
            chn_ids = chn_ids.unsqueeze(-1)
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
        return np.array(np_obj[np_obj.files[0]])

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
        x_list, label, sensor = self.base_ds[idx]
        imgs = torch.cat([x["imgs"] for x in x_list], dim=0)
        chn_ids = torch.cat([x["chn_ids"] for x in x_list], dim=0)
        x_dict = dict(imgs=imgs, chn_ids=chn_ids)
        return x_dict, label, sensor


def custom_collate_fn(batch):
    x_dicts, labels, sensors = zip(*batch)
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
    return batched_x_dict, torch.tensor(labels), list(sensors)


def _resolve_base_dataset(dataset: Dataset) -> Dataset:
    base = dataset
    while hasattr(base, "base_ds"):
        base = base.base_ds  # type: ignore[attr-defined]
    return base


def _gather_sensor_indices(dataset: Dataset, sensors: Sequence[str]) -> Dict[str, Sequence[int]]:
    base_ds = _resolve_base_dataset(dataset)
    df = getattr(base_ds, "df", None)
    if df is None or "sensor" not in df.columns:
        raise ValueError("Dataset must expose a DataFrame with a 'sensor' column to split by sensor.")
    idx_map = {sensor: [] for sensor in sensors}
    for idx, sensor_name in enumerate(df["sensor"].tolist()):
        if sensor_name in idx_map:
            idx_map[sensor_name].append(idx)
    return idx_map


def build_sensor_dataloaders(
    dataset: Dataset,
    sensors: Sequence[str],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    oversample_to_max_steps: bool = False,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    idx_map = _gather_sensor_indices(dataset, sensors)
    base_steps = {
        sensor: max(1, math.ceil(len(indices) / batch_size)) if indices else 0 for sensor, indices in idx_map.items()
    }
    max_steps = max(base_steps.values(), default=0)
    loaders: Dict[str, DataLoader] = {}
    steps_per_sensor: Dict[str, int] = {}
    for sensor_name, indices in idx_map.items():
        if not indices:
            continue
        subset = Subset(dataset, indices)
        target_steps = max_steps if oversample_to_max_steps else base_steps[sensor_name]
        if target_steps <= 0:
            continue
        sampler = None
        if oversample_to_max_steps and len(indices) > 0:
            num_samples = target_steps * batch_size
            sampler = RandomSampler(subset, replacement=True, num_samples=num_samples)
        loaders[sensor_name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn,
        )
        steps_per_sensor[sensor_name] = target_steps
    return loaders, steps_per_sensor


def build_balanced_mixed_dataloader(
    dataset: Dataset,
    sensors: Sequence[str],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    oversample_to_max: bool = True,
) -> Tuple[DataLoader, int]:
    """Create a single DataLoader that mixes sensors and oversamples minority ones.

    - If ``oversample_to_max`` is True (default), each sensor is repeated until it
      reaches the size of the largest sensor, balancing class counts.
    - Otherwise, uses the natural counts.
    Returns (loader, total_steps_per_epoch).
    """

    idx_map = _gather_sensor_indices(dataset, sensors)
    counts = {s: len(v) for s, v in idx_map.items() if v}
    if not counts:
        raise ValueError("No samples found for the configured sensors.")

    target = max(counts.values()) if oversample_to_max else None
    balanced_indices: list[int] = []
    for sensor, indices in idx_map.items():
        if not indices:
            continue
        desired = target if target is not None else len(indices)
        repeat = math.ceil(desired / len(indices))
        expanded = (indices * repeat)[:desired]
        balanced_indices.extend(expanded)

    sampler = SubsetRandomSampler(balanced_indices) if shuffle else balanced_indices
    steps = math.ceil(len(balanced_indices) / batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if isinstance(sampler, SubsetRandomSampler) else shuffle,
        sampler=sampler if isinstance(sampler, SubsetRandomSampler) else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
    )
    return loader, steps


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


def gather_head_parameters(model: MultiSensorPanopticonClassifier) -> Sequence[nn.Parameter]:
    shared_patch = model.backbone.patch_embed
    extra_patch_params = []
    for sensor, module in model.sensor_patch_embeds.items():
        if module is shared_patch:
            continue
        extra_patch_params.extend(list(module.parameters()))
    summary_params = list(model.summary_head.parameters()) if model.summary_head is not None else []
    head_params = list(model.heads.parameters()) + summary_params + extra_patch_params
    return head_params


# --------------------------------------------------------------------------------------
#  CLI / training loop
# --------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-sensor Panopticon finetuning (S2/L89/S5P/WV3).")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--weights", default="weights/panopticon_vitb14_teacher.pth")
    parser.add_argument("--t0_col", default="path_t0")
    parser.add_argument("--t90_col", default="path_t90")
    parser.add_argument("--t360_col", default="path_t360")
    parser.add_argument("--s5p_data_key", default=None)
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
    parser.add_argument("--oversample_minority", action="store_true", dest="oversample_minority")
    parser.add_argument("--disable_oversample_minority", action="store_false", dest="oversample_minority")
    parser.set_defaults(oversample_minority=True)
    parser.add_argument("--sensor_switch_interval", type=int, default=100)
    parser.add_argument("--summary_head", action="store_true", dest="summary_head")
    parser.add_argument("--disable_summary_head", action="store_false", dest="summary_head")
    parser.set_defaults(summary_head=True)
    parser.add_argument("--summary_hidden_dim", type=int, default=128)
    parser.add_argument("--summary_dropout", type=float, default=0.1)
    parser.add_argument("--summary_loss_weight", type=float, default=1.0)
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    path_columns = (args.t0_col, args.t90_col, args.t360_col)
    wv3_band_names = [b.strip() for b in args.wv3_bands.split(",") if b.strip()]
    if len(wv3_band_names) == 0:
        raise ValueError("--wv3_bands must provide at least one WV3 band column name")
    wv3_chn_ids = load_wv3_channel_ids_from_srf(args.wv3_srf_csv, wv3_band_names).unsqueeze(-1)

    cache_obj = None
    if args.local_cache_dir:
        cache_obj = StaticAnchoredCache(args.local_cache_dir, min_free_gb=args.local_cache_min_free_gb)

    base_train_ds = TriSensorTemporalCsvDataset(
        csv_path=args.train_csv,
        path_columns=path_columns,
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
        path_columns=path_columns,
        local_file_cache=cache_obj,
        s5p_data_key=args.s5p_data_key,
        s5p_chn_ids_key=args.s5p_chn_ids_key,
        s5p_channels_last=args.s5p_channels_last,
        align_l89_to_s2=args.align_l89_to_s2,
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )

    if args.local_cache_warmup and cache_obj:
        all_paths = []
        for col in path_columns:
            if col in base_train_ds.df.columns:
                all_paths.extend(base_train_ds.df[col].dropna().astype(str).tolist())
        cache_obj.warm_up(all_paths, max_workers=args.local_cache_workers)

    train_ds = ConcatTemporalDataset(base_train_ds)
    test_ds = ConcatTemporalDataset(base_test_ds)

    core_model = MultiSensorPanopticonClassifier(
        backbone=_load_backbone(args.weights),
        enable_summary_head=args.summary_head,
        summary_hidden_dim=args.summary_hidden_dim,
        summary_dropout=args.summary_dropout,
        summary_loss_weight=args.summary_loss_weight,
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
    pin_memory = device.type == "cuda"
    # Single mixed loader with optional oversampling so that minority sensors are repeated
    # and batches contain mixed sensors.
    train_loader, train_steps = build_balanced_mixed_dataloader(
        train_ds,
        sensors_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        oversample_to_max=args.oversample_minority,
    )
    test_loader, test_steps = build_balanced_mixed_dataloader(
        test_ds,
        sensors_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        oversample_to_max=False,
    )

    print(
        f"Using device={device}, train_samples={len(train_ds)}, test_samples={len(test_ds)}, "
        f"sensors={sensors_list}, train_backbone={args.train_backbone}",
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
        core_model.heads.train()
        if core_model.summary_head is not None:
            core_model.summary_head.train()
        for sensor in core_model.sensor_patch_embeds.values():
            sensor.train()

        total_loss = 0.0
        total = 0
        correct = 0
        per_sensor_loss_accum = {sensor: 0.0 for sensor in sensors_list}
        per_sensor_count = {sensor: 0 for sensor in sensors_list}
        summary_loss_accum = 0.0
        summary_loss_count = 0
        train_sensor_correct = {sensor: 0 for sensor in sensors_list}
        train_sensor_total = {sensor: 0 for sensor in sensors_list}

        step_idx = 0
        for x_dict, labels, sensors in train_loader:
            step_idx += 1
            labels = labels.to(device)
            x_dict = recursive_to_device(x_dict, device)
            sensor_arg: Union[Sequence[str], torch.Tensor]
            if use_data_parallel:
                sensor_arg = core_model.encode_sensors(sensors, device=device)
            else:
                sensor_arg = sensors
            with autocast(enabled=use_amp):
                outputs = model(x_dict, sensors=sensor_arg)
                loss, per_sensor_losses = core_model.loss_from_outputs(outputs, labels, sensors, criterion)
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

            batch = labels.size(0)
            total_loss += loss.item() * batch
            total += batch
            logits = outputs.get("summary_logits")
            if logits is None:
                logits = outputs.get("merged_logits")
            if logits is None:
                logits = torch.zeros((batch, core_model.heads[sensors_list[0]].out_features), device=device)
                for sensor_name, sensor_out in outputs.items():
                    if not isinstance(sensor_out, dict):
                        continue
                    logits.index_copy_(0, sensor_out["indices"], sensor_out["logits"])
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            for i, sensor_type in enumerate(sensors):
                train_sensor_total[sensor_type] += 1
                if preds[i] == labels[i]:
                    train_sensor_correct[sensor_type] += 1
            for sensor_name, sensor_loss in per_sensor_losses.items():
                if sensor_name == "summary":
                    summary_loss_accum += sensor_loss.item()
                    summary_loss_count += 1
                    continue
                per_sensor_loss_accum[sensor_name] += sensor_loss.item()
                per_sensor_count[sensor_name] += 1

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break
            if args.log_interval and step_idx % args.log_interval == 0:
                sensor_loss_details = " ".join(
                    f"{name}_loss={per_sensor_losses[name].item():.4f}"
                    for name in sensors_list
                    if name in per_sensor_losses
                )
                if "summary" in per_sensor_losses:
                    sensor_loss_details = f"{sensor_loss_details} summary_loss={per_sensor_losses['summary'].item():.4f}".strip()
                print(
                    f"Epoch {epoch} step {step_idx}/{train_steps} train_loss={total_loss/total:.4f} "
                    f"train_acc={correct/total:.4f} {sensor_loss_details}",
                    flush=True,
                )

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)
        train_per_sensor_acc = {
            sensor: (train_sensor_correct[sensor] / train_sensor_total[sensor] if train_sensor_total[sensor] > 0 else float("nan"))
            for sensor in sensors_list
        }

        model.eval()
        sensor_correct = {sensor: 0 for sensor in sensors_list}
        sensor_total = {sensor: 0 for sensor in sensors_list}
        test_loss_total = 0.0
        total_eval = 0
        correct_eval = 0
        eval_steps = 0
        eval_labels: list[int] = []
        eval_preds: list[int] = []
        eval_pos_scores: list[float] = []
        eval_sensor_labels = {sensor: [] for sensor in sensors_list}
        eval_sensor_preds = {sensor: [] for sensor in sensors_list}
        eval_sensor_pos_scores = {sensor: [] for sensor in sensors_list}
        with torch.no_grad():
            for step, (x_dict, labels, sensors) in enumerate(test_loader, 1):
                labels = labels.to(device)
                x_dict = recursive_to_device(x_dict, device)
                if use_data_parallel:
                    sensor_arg = core_model.encode_sensors(sensors, device=device)
                else:
                    sensor_arg = sensors
                with autocast(enabled=use_amp):
                    outputs = model(x_dict, sensors=sensor_arg)
                    loss, _ = core_model.loss_from_outputs(outputs, labels, sensors, criterion)
                batch = labels.size(0)
                test_loss_total += loss.item() * batch
                total_eval += batch
                logits = outputs.get("summary_logits")
                if logits is None:
                    logits = outputs.get("merged_logits")
                if logits is None:
                    logits = torch.zeros((batch, core_model.heads[sensors_list[0]].out_features), device=device)
                    for sensor_name, sensor_out in outputs.items():
                        if not isinstance(sensor_out, dict):
                            continue
                        logits.index_copy_(0, sensor_out["indices"], sensor_out["logits"])
                preds = logits.argmax(dim=1)
                correct_eval += (preds == labels).sum().item()
                eval_labels.extend(labels.detach().to("cpu").tolist())
                eval_preds.extend(preds.detach().to("cpu").tolist())
                batch_pos_scores: Optional[list[float]] = None
                if logits.shape[-1] == 2:
                    pos_scores = torch.softmax(logits, dim=1)[:, 1]
                    batch_pos_scores = pos_scores.detach().to("cpu").tolist()
                    eval_pos_scores.extend(batch_pos_scores)
                for i, sensor_type in enumerate(sensors):
                    sensor_total[sensor_type] += 1
                    if preds[i] == labels[i]:
                        sensor_correct[sensor_type] += 1
                    eval_sensor_labels[sensor_type].append(int(labels[i].item()))
                    eval_sensor_preds[sensor_type].append(int(preds[i].item()))
                    if batch_pos_scores is not None:
                        eval_sensor_pos_scores[sensor_type].append(float(batch_pos_scores[i]))
                eval_steps += 1
                if args.max_eval_steps is not None and eval_steps >= args.max_eval_steps:
                    break

        test_loss = test_loss_total / max(1, total_eval)
        test_acc = correct_eval / max(1, total_eval)
        per_sensor_acc = {
            sensor: (sensor_correct[sensor] / sensor_total[sensor] if sensor_total[sensor] > 0 else float("nan"))
            for sensor in sensors_list
        }
        test_fpr = float("nan")
        test_recall = float("nan")
        test_auroc = float("nan")
        if eval_labels and eval_preds and eval_pos_scores and len(eval_labels) == len(eval_pos_scores):
            metrics = compute_binary_metrics(
                labels=np.asarray(eval_labels, dtype=np.int64),
                preds=np.asarray(eval_preds, dtype=np.int64),
                pos_scores=np.asarray(eval_pos_scores, dtype=np.float64),
            )
            test_fpr = metrics["fpr"]
            test_recall = metrics["recall"]
            test_auroc = metrics["auroc"]
        per_sensor_binary_metrics = {
            sensor: {"fpr": float("nan"), "recall": float("nan"), "auroc": float("nan")}
            for sensor in sensors_list
        }
        for sensor in sensors_list:
            sensor_labels = eval_sensor_labels[sensor]
            sensor_preds = eval_sensor_preds[sensor]
            sensor_scores = eval_sensor_pos_scores[sensor]
            if sensor_labels and sensor_preds and sensor_scores and len(sensor_labels) == len(sensor_scores):
                per_sensor_binary_metrics[sensor] = compute_binary_metrics(
                    labels=np.asarray(sensor_labels, dtype=np.int64),
                    preds=np.asarray(sensor_preds, dtype=np.int64),
                    pos_scores=np.asarray(sensor_scores, dtype=np.float64),
                )

        train_acc_str = " ".join(f"train_acc_{s}={train_per_sensor_acc[s]:.4f}" for s in sensors_list)
        test_acc_str = " ".join(f"test_acc_{s}={per_sensor_acc[s]:.4f}" for s in sensors_list)
        test_metric_str = " ".join(
            f"{s}(fpr={per_sensor_binary_metrics[s]['fpr']:.4f},recall={per_sensor_binary_metrics[s]['recall']:.4f},auroc={per_sensor_binary_metrics[s]['auroc']:.4f})"
            for s in sensors_list
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"test_fpr={test_fpr:.4f} test_recall={test_recall:.4f} test_auroc={test_auroc:.4f} "
            f"{train_acc_str} {test_acc_str} test_metrics_per_sensor={test_metric_str}",
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
            }
            for sensor in sensors_list:
                log_payload[f"test_acc_{sensor}"] = per_sensor_acc[sensor]
                log_payload[f"test_fpr_{sensor}"] = per_sensor_binary_metrics[sensor]["fpr"]
                log_payload[f"test_recall_{sensor}"] = per_sensor_binary_metrics[sensor]["recall"]
                log_payload[f"test_auroc_{sensor}"] = per_sensor_binary_metrics[sensor]["auroc"]
                if per_sensor_count[sensor] > 0:
                    log_payload[f"train_loss_{sensor}"] = (
                        per_sensor_loss_accum[sensor] / per_sensor_count[sensor]
                    )
                if train_sensor_total[sensor] > 0:
                    log_payload[f"train_acc_{sensor}"] = train_per_sensor_acc[sensor]
            if summary_loss_count > 0:
                log_payload["train_loss_summary"] = summary_loss_accum / summary_loss_count
            wandb_run.log(log_payload)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
