import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile as tiff
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

from thirdparty.dinov2.data.datasets.s2_csv import S2CsvDataset, _SkipSample  # noqa: E402
from thirdparty.dinov2.models.panopticon import get_1d_sincos_pos_embed_from_grid_torch  # noqa: E402
from research.pretraining_20260727.temporal_input_modes import (  # noqa: E402
    INPUT_MODES,
    output_timepoints,
    parse_residual_slots,
    transform_temporal_sample,
)


# EMIT methane-window 32-band reflectance crops.
# Mean/std computed from 2048 deterministic train rows across all six temporal
# frames (12288 TIFFs total, 0 read errors).
PRECOMPUTED_STATS = (
    [
        0.3102248617204014,
        0.3062076053137987,
        0.3000435188072249,
        0.29850084031517754,
        0.29042564517391883,
        0.28659220996284945,
        0.28383722808741113,
        0.2808488846852097,
        0.2774078032499092,
        0.27338973249027826,
        0.26911303681754767,
        0.2631630441273006,
        0.26209166857813093,
        0.258695697634416,
        0.25317290830327155,
        0.2505395171511225,
        0.24672653170982847,
        0.24207854484288904,
        0.24253591700706903,
        0.24646309523582835,
        0.24756406412800339,
        0.24479909179801757,
        0.24686169493194615,
        0.24447540560415565,
        0.24146216167485027,
        0.24188521399272622,
        0.24294280478030142,
        0.23218466215155795,
        0.22353577912529493,
        0.21181748813712248,
        0.2102313903654539,
        0.18413862995335437,
    ],
    [
        0.17151182060708098,
        0.16196419947924104,
        0.15276853376517532,
        0.1489800008515783,
        0.14882329033650493,
        0.14849481876746184,
        0.14911093695252972,
        0.1489861387081469,
        0.14804947255915735,
        0.14663433408694854,
        0.14608822599861193,
        0.1446477085487578,
        0.14480619392731428,
        0.14468403255360868,
        0.14267787728486328,
        0.14106284899703625,
        0.14085656335893254,
        0.14032345260076148,
        0.1426832285582298,
        0.14375748597513582,
        0.1468650648223444,
        0.14787482485821488,
        0.15124661873823136,
        0.15003746726830364,
        0.1465215045871904,
        0.14917709759166783,
        0.1567436237132932,
        0.14838663323624848,
        0.14923483849253444,
        0.16458661783289333,
        0.2359286547670021,
        0.24485059100995565,
    ],
)

EMIT32_WAVELENGTHS_NM = [
    2137.828857421875,
    2167.467041015625,
    2197.096923828125,
    2219.314697265625,
    2248.9296875,
    2256.332763671875,
    2263.734619140625,
    2271.136474609375,
    2278.53759765625,
    2285.938720703125,
    2293.338623046875,
    2300.73779296875,
    2308.135986328125,
    2315.5341796875,
    2322.9326171875,
    2330.329833984375,
    2337.726318359375,
    2345.12158203125,
    2352.51708984375,
    2359.91259765625,
    2367.30712890625,
    2374.70068359375,
    2382.093505859375,
    2389.486083984375,
    2396.8779296875,
    2404.26953125,
    2411.660400390625,
    2426.440185546875,
    2441.21826171875,
    2463.381591796875,
    2478.153076171875,
    2492.923828125,
]

DEFAULT_TRAIN_CSV = "/mnt/engg-niulab/Yuyao/final_crop/emit_32band/emit32_temporal_train.csv"
DEFAULT_TEST_CSV = "/mnt/engg-niulab/Yuyao/final_crop/emit_32band/emit32_temporal_test.csv"
DEFAULT_PATH_COLUMNS = "path_t0,path_prev1,path_prev2,path_prev3,path_seasonal,path_year"
DEFAULT_TIME_COLUMNS = (
    "t0_image_time,prev1_image_time,prev2_image_time,prev3_image_time,seasonal_image_time,year_image_time"
)


class StaticAnchoredCache:
    """Lazy local cache with optional background copy workers.

    In async mode, a cache miss schedules a background copy and returns the
    original path immediately. Later reads use the local file once the copy has
    completed. This lets DataLoader workers overlap remote reads/copies with GPU
    training instead of blocking on a full upfront warmup.
    """

    def __init__(
        self,
        cache_dir: str,
        min_free_gb: float = 10.0,
        *,
        async_mode: bool = False,
        max_workers: int = 2,
        prefetch_on_miss: bool = True,
    ):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_free_bytes = min_free_gb * (1024**3)
        self.async_mode = bool(async_mode)
        self.max_workers = max(1, int(max_workers))
        self.prefetch_on_miss = bool(prefetch_on_miss)
        self._executor = None
        self._executor_pid = None
        self._inflight = set()
        self._lock = None

    def _get_free_space(self) -> int:
        return shutil.disk_usage(self.cache_dir).free

    def _hashed_path(self, original: str) -> Path:
        norm_path = os.path.abspath(original)
        digest = hashlib.sha1(norm_path.encode("utf-8")).hexdigest()
        subdir = digest[:2]
        suffix = Path(original).suffix
        return self.cache_dir / subdir / f"{digest}{suffix}"

    def _copy_to_cache(self, original: str) -> str:
        dst = self._hashed_path(original)
        if dst.exists():
            return str(dst)
        if self._get_free_space() < self.min_free_bytes:
            return original
        tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(original, tmp)
            os.replace(tmp, dst)
        except Exception:
            with suppress(FileNotFoundError):
                tmp.unlink()
            return original
        return str(dst)

    def _ensure_async_state(self):
        pid = os.getpid()
        if self._executor is not None and self._executor_pid == pid:
            return
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._executor_pid = pid
        self._inflight = set()
        self._lock = threading.Lock()

    def _mark_done(self, key: str):
        if self._lock is None:
            return
        with self._lock:
            self._inflight.discard(key)

    def schedule(self, original: str) -> None:
        if not self.async_mode or not isinstance(original, str) or not original:
            return
        dst = self._hashed_path(original)
        if dst.exists():
            return
        self._ensure_async_state()
        key = str(dst)
        with self._lock:
            if key in self._inflight:
                return
            self._inflight.add(key)
        future = self._executor.submit(self._copy_to_cache, original)
        future.add_done_callback(lambda _future, _key=key: self._mark_done(_key))

    def ensure_local(self, original: str) -> str:
        dst = self._hashed_path(original)
        if dst.exists():
            return str(dst)
        if self.async_mode:
            if self.prefetch_on_miss:
                self.schedule(original)
            return original
        return self._copy_to_cache(original)

    def warm_up(self, paths: Sequence[str], max_workers: int = 8) -> None:
        unique_paths = sorted({os.path.abspath(p) for p in paths if isinstance(p, str)})
        if not unique_paths:
            return
        print(f"[Cache] Warming up (target: {len(unique_paths)})...", flush=True)

        def _copy_one(path: str):
            res = self._copy_to_cache(path)
            return res == path

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_copy_one, p) for p in unique_paths]
            fallback_count = sum(1 for fut in as_completed(futures) if fut.result())
            print(f"[Cache] Warmup complete. Cached: {len(unique_paths) - fallback_count}, Remote: {fallback_count}")

class CachedS2CsvDataset(S2CsvDataset):
    def __init__(self, *args, local_file_cache: Optional[StaticAnchoredCache] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_file_cache = local_file_cache

    def _load_image(self, path: str, *, column_name=None, sample_id=None):
        if self._local_file_cache is not None and isinstance(path, str):
            path = self._local_file_cache.ensure_local(path)
        return super()._load_image(path, column_name=column_name, sample_id=sample_id)


class Emit32CsvDataset(CachedS2CsvDataset):
    """CSV TIFF dataset for EMIT 32-band pixel-interleaved GeoTIFF crops."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, ds_cfg_name="landsat89_7band", **kwargs)
        self.chn_ids = torch.tensor(EMIT32_WAVELENGTHS_NM, dtype=torch.float32)

    def _read_image_raw(self, path: str) -> torch.Tensor:
        if not os.path.isfile(path):
            raise _SkipSample(f"Missing file: {path}")

        try:
            # EMIT crops are pixel-interleaved TIFFs.  tifffile returns them
            # as HWC and is substantially faster than opening every training
            # sample through GDAL/rasterio; transpose below preserves exactly
            # the CHW values returned by rasterio.read().
            img_np = tiff.imread(path)
        except Exception as exc:
            raise _SkipSample(f"TIFF read failed for {path}: {exc}") from exc

        expected_channels = len(EMIT32_WAVELENGTHS_NM)
        if not isinstance(img_np, np.ndarray) or img_np.ndim != 3:
            raise _SkipSample(
                f"Unexpected EMIT image shape {getattr(img_np, 'shape', None)} "
                f"for {path}; expected a 3D array."
            )
        if img_np.shape[0] == expected_channels:
            pass
        elif img_np.shape[-1] == expected_channels:
            img_np = np.transpose(img_np, (2, 0, 1))
        else:
            raise _SkipSample(
                f"Unexpected EMIT image shape {img_np.shape} for {path}; "
                f"expected ({expected_channels}, H, W)."
            )
        if img_np.size == 0 or any(dim == 0 for dim in img_np.shape):
            raise _SkipSample(f"Empty/zero-sized TIFF array for {path}: shape={img_np.shape}")

        img_np = np.ascontiguousarray(img_np, dtype=np.float32)
        return torch.from_numpy(img_np)


def parse_csv_columns(value: str) -> Tuple[str, ...]:
    columns = tuple(col.strip() for col in value.split(",") if col.strip())
    if not columns:
        raise ValueError("Expected at least one CSV column name.")
    return columns


class Emit32TemporalSequenceDataset(Dataset):
    """
    Six-timepoint EMIT 32-band dataset.

    Returns x_dict with:
      imgs: (T, 32, H, W)
      chn_ids: (T, 32)
      timestamps: (T, 3) containing year_offset, month_index, hour
    """

    def __init__(
        self,
        csv_path: str,
        *,
        path_columns: Sequence[str],
        time_columns: Sequence[str],
        normalize_stats,
        time_base_year: int = 2000,
        pad_to_multiple: int = 14,
        skip_invalid_samples: bool = False,
        skip_path_validation: bool = False,
        local_file_cache: Optional[StaticAnchoredCache] = None,
        cache_prefetch_rows: int = 0,
        input_mode: str = "raw",
        residual_slots: Sequence[str] = ("path_prev1", "path_seasonal"),
        residual_clip: float = 5.0,
    ):
        if len(path_columns) != len(time_columns):
            raise ValueError(
                f"path_columns and time_columns must have the same length, got "
                f"{len(path_columns)} and {len(time_columns)}"
            )

        dataset_cls = Emit32CsvDataset
        dataset_kwargs = {}
        if local_file_cache is not None:
            dataset_kwargs["local_file_cache"] = local_file_cache

        self.base_ds = dataset_cls(
            csv_path=csv_path,
            path_column=path_columns[0],
            normalize_stats=normalize_stats,
            scale_to_unit=False,
            compute_stats=False,
            pad_to_multiple=pad_to_multiple,
            skip_invalid_samples=skip_invalid_samples and not skip_path_validation,
            path_columns_for_validation=path_columns,
            **dataset_kwargs,
        )
        self.path_columns = tuple(path_columns)
        self.time_columns = tuple(time_columns)
        self.time_base_year = int(time_base_year)
        self.local_file_cache = local_file_cache
        self.cache_prefetch_rows = max(0, int(cache_prefetch_rows))
        self.input_mode = str(input_mode)
        self.residual_slots = tuple(residual_slots)
        self.residual_clip = float(residual_clip)
        self.timestamps = self._build_timestamp_tensor(self.base_ds.df)

    def _build_timestamp_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        time_features = []
        missing_columns = [col for col in self.time_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing timestamp columns in CSV: {missing_columns}")

        for col in self.time_columns:
            dt = pd.to_datetime(df[col], utc=True, errors="coerce")
            invalid = dt.isna()
            if invalid.any():
                bad_rows = df.loc[invalid, ["id", col] if "id" in df.columns else [col]].head(5).to_dict("records")
                raise ValueError(
                    f"Timestamp column '{col}' has {int(invalid.sum())} missing/invalid values. "
                    f"Examples: {bad_rows}"
                )

            year = torch.tensor(dt.dt.year.to_numpy(dtype="int64") - self.time_base_year, dtype=torch.float32)
            month = torch.tensor(dt.dt.month.to_numpy(dtype="int64") - 1, dtype=torch.float32)
            hour = torch.tensor(dt.dt.hour.to_numpy(dtype="int64"), dtype=torch.float32)
            time_features.append(torch.stack((year, month, hour), dim=1))

        return torch.stack(time_features, dim=1)  # (N, T, 3)

    def __len__(self):
        return len(self.base_ds)

    def _prefetch_paths(self, idx: int) -> None:
        if self.local_file_cache is None or self.cache_prefetch_rows <= 0:
            return
        if not getattr(self.local_file_cache, "async_mode", False):
            return
        stop = min(len(self.base_ds.df), idx + self.cache_prefetch_rows)
        for row_idx in range(idx, stop):
            row = self.base_ds.df.iloc[row_idx]
            for col in self.path_columns:
                self.local_file_cache.schedule(row[col])

    def _get_one(self, idx):
        self._prefetch_paths(idx)
        row = self.base_ds.df.iloc[idx]
        label = int(row[self.base_ds.label_column])
        sample_id = row[self.base_ds.id_column] if self.base_ds.id_column in row else None

        imgs = []
        chn_ids = []
        base_chn_ids = self.base_ds.chn_ids
        if not isinstance(base_chn_ids, torch.Tensor):
            base_chn_ids = torch.as_tensor(base_chn_ids)
        base_chn_ids = base_chn_ids.to(dtype=torch.float32)

        for col in self.path_columns:
            img = self.base_ds._load_image(row[col], column_name=col, sample_id=sample_id)
            img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            imgs.append(img)
            chn_ids.append(base_chn_ids.clone())

        x_dict = {
            "imgs": torch.stack(imgs, dim=0),  # (T, C, H, W)
            "chn_ids": torch.stack(chn_ids, dim=0),  # (T, C)
            "timestamps": self.timestamps[idx],
        }
        x_dict = transform_temporal_sample(
            x_dict,
            path_columns=self.path_columns,
            input_mode=self.input_mode,
            residual_slots=self.residual_slots,
            residual_clip=self.residual_clip,
        )
        return x_dict, label

    def __getitem__(self, idx):
        attempts = 0
        last_exc: Optional[Exception] = None
        while attempts < self.base_ds.max_retries:
            try:
                return self._get_one(idx)
            except _SkipSample as exc:
                last_exc = exc
                attempts += 1
                if attempts >= self.base_ds.max_retries:
                    raise RuntimeError(f"Exceeded {self.base_ds.max_retries} retries for temporal index {idx}") from exc
                idx = torch.randint(0, len(self), ()).item()
        raise RuntimeError("Unreachable: retry loop exited unexpectedly") from last_exc


class CLSHead(nn.Module):
    """Minimal DINO-style classifier head: CLS -> LayerNorm -> Linear."""

    def __init__(self, embed_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(cls))


class TemporalPanopticonBackbone(nn.Module):
    """
    Panopticon ViT-B/14 with SatMAE-style temporal embeddings.

    Each EMIT timepoint is patchified independently by the shared PanopticonPE
    patch embed. The transformer then attends over the flattened T*L token
    sequence after spatial and temporal embeddings are added.
    """

    def __init__(self, backbone: nn.Module, *, num_timepoints: int, time_embed_dim: int = 384):
        super().__init__()
        if time_embed_dim % 3 != 0 or (time_embed_dim // 3) % 2 != 0:
            raise ValueError("time_embed_dim must split into three even dimensions, e.g. 384.")

        self.backbone = backbone
        self.num_timepoints = int(num_timepoints)
        self.time_embed_dim = int(time_embed_dim)
        self.embed_dim = int(backbone.embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, self.embed_dim, bias=False)
        self.slot_embed = nn.Parameter(torch.zeros(1, self.num_timepoints, 1, self.embed_dim))

        # Start as close as possible to the pretrained Panopticon backbone.
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.slot_embed)

    def temporal_parameters(self):
        yield from self.time_proj.parameters()
        yield self.slot_embed

    def _temporal_embedding(self, timestamps: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # timestamps: (B, T, 3), ordered as year_offset, month_index, hour.
        bsz, num_timepoints, _ = timestamps.shape
        per_component_dim = self.time_embed_dim // 3
        flat = timestamps.reshape(-1, 3).float()
        parts = [
            get_1d_sincos_pos_embed_from_grid_torch(per_component_dim, flat[:, i])
            for i in range(3)
        ]
        emb = torch.cat(parts, dim=1).to(device=timestamps.device, dtype=dtype)
        emb = self.time_proj(emb).reshape(bsz, num_timepoints, 1, self.embed_dim)
        return emb

    def forward_features(self, x_dict: dict, masks=None):
        if masks is not None:
            raise NotImplementedError("TemporalPanopticonBackbone does not support patch masks yet.")

        imgs = x_dict["imgs"]
        chn_ids = x_dict["chn_ids"]
        timestamps = x_dict["timestamps"]
        if imgs.ndim != 5:
            raise ValueError(f"Expected imgs to have shape (B,T,C,H,W), got {tuple(imgs.shape)}")

        bsz, num_timepoints, channels, height, width = imgs.shape
        if num_timepoints > self.num_timepoints:
            raise ValueError(f"Got {num_timepoints} timepoints, but model was built for {self.num_timepoints}")
        if timestamps.shape[:2] != (bsz, num_timepoints):
            raise ValueError(
                f"Expected timestamps shape (B,T,3) with B,T={(bsz, num_timepoints)}, got {tuple(timestamps.shape)}"
            )

        flat_imgs = imgs.reshape(bsz * num_timepoints, channels, height, width)
        flat_chn_ids = chn_ids.reshape(bsz * num_timepoints, -1)
        patch_tokens, h_in, w_in = self.backbone.patch_embed({"imgs": flat_imgs, "chn_ids": flat_chn_ids})

        cls_for_pos = self.backbone.cls_token.to(
            device=patch_tokens.device, dtype=patch_tokens.dtype
        ).expand(bsz * num_timepoints, -1, -1)
        pos_input = torch.cat((cls_for_pos, patch_tokens), dim=1)
        spatial_pos = self.backbone.interpolate_pos_encoding(pos_input, w_in, h_in)[:, 1:, :]
        patch_tokens = patch_tokens + spatial_pos

        num_patches = patch_tokens.shape[1]
        patch_tokens = patch_tokens.reshape(bsz, num_timepoints, num_patches, self.embed_dim)
        patch_tokens = patch_tokens + self._temporal_embedding(timestamps, patch_tokens.dtype)
        patch_tokens = patch_tokens + self.slot_embed[:, :num_timepoints].to(
            device=patch_tokens.device, dtype=patch_tokens.dtype
        )
        patch_tokens = patch_tokens.reshape(bsz, num_timepoints * num_patches, self.embed_dim)

        cls_tokens = self.backbone.cls_token.to(
            device=patch_tokens.device, dtype=patch_tokens.dtype
        ).expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, patch_tokens), dim=1)

        if self.backbone.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.backbone.register_tokens.to(device=x.device, dtype=x.dtype).expand(bsz, -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        for blk in self.backbone.blocks:
            x = blk(x)

        x_norm = self.backbone.norm(x)
        patch_start = 1 + self.backbone.num_register_tokens
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1:patch_start],
            "x_norm_patchtokens": x_norm[:, patch_start:],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        return self.backbone.head(ret["x_norm_clstoken"])


def load_backbone(weights_path: str, device: torch.device, debug: bool = False):
    """
    Build the Panopticon ViT-B/14 backbone and optionally load teacher weights.
    """

    from src.backbones import build_panopticon_vitb14

    if debug:
        print("Building Panopticon ViT-B/14 backbone...", flush=True)
    model = build_panopticon_vitb14()

    if weights_path in (None, "", "none", "scratch", "random"):
        if debug:
            print("Using randomly initialized backbone.", flush=True)
        return model

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        alt_path = Path(str(weights_path) + "?download=true")
        if alt_path.is_file():
            weights_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    print(f"Loading checkpoint from {weights_path}", flush=True)
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "backbone" in state:
        state = state["backbone"]
    model.load_state_dict(state, strict=True)
    return model


def recursive_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: recursive_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [recursive_to_device(v, device) for v in x]
        return type(x)(t)
    return x


def maybe_wrap_dataparallel(module: nn.Module, device: torch.device, num_gpus: int, module_name: str) -> nn.Module:
    if device.type != "cuda" or num_gpus <= 1:
        return module

    available = torch.cuda.device_count()
    if available < num_gpus:
        raise RuntimeError(
            f"Requested num_gpus={num_gpus}, but only {available} CUDA device(s) are visible."
        )

    device_ids = list(range(num_gpus))
    print(
        f"Wrapping {module_name} with DataParallel across GPU ids {device_ids}. Batch size will be split automatically.",
        flush=True,
    )
    return nn.DataParallel(module, device_ids=device_ids)


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


def load_state_dict_flexible(module: nn.Module, state_dict):
    base_module = unwrap_module(module)
    try:
        base_module.load_state_dict(state_dict)
    except RuntimeError:
        if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
            stripped = {k[len("module."):]: v for k, v in state_dict.items()}
            base_module.load_state_dict(stripped)
        else:
            raise


def resize_imgs_to_224(x_dict):
    imgs = x_dict.get("imgs")
    if imgs is None:
        return x_dict

    if imgs.ndim == 5:
        bsz, num_timepoints, channels, _, _ = imgs.shape
        imgs = imgs.reshape(bsz * num_timepoints, channels, imgs.shape[-2], imgs.shape[-1])
        imgs = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
        x_dict["imgs"] = imgs.reshape(bsz, num_timepoints, channels, 224, 224)
    elif imgs.ndim == 4:
        x_dict["imgs"] = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
    return x_dict


def set_trainable(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def set_backbone_train_state(model: nn.Module, train_panopticon: bool):
    temporal_model = unwrap_module(model)
    set_trainable(temporal_model.backbone, train_panopticon)
    if train_panopticon:
        temporal_model.backbone.train()
    else:
        temporal_model.backbone.eval()
    set_trainable(temporal_model.time_proj, True)
    temporal_model.slot_embed.requires_grad_(True)


def init_wandb(args):
    if not args.use_wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )


def build_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "noam":
        warmup_steps = max(args.warmup_steps, 1)
        noam_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")


def default_run_name(args) -> str:
    train_stem = Path(args.train_csv).stem or "train"
    test_stem = Path(args.test_csv).stem or "test"
    mode = "ft" if args.train_backbone else "temporal_head"
    suffix = "" if args.input_mode == "raw" else f"__{args.input_mode}"
    return f"{train_stem}__{test_stem}__satmae_time__{mode}{suffix}"


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    backbone: nn.Module,
    head: nn.Module,
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
            "backbone": unwrap_module(backbone).state_dict(),
            "head": unwrap_module(head).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "best_train_acc": best_train_acc,
            "best_test_acc": best_test_acc,
            "args": vars(args),
        },
        path,
    )


def try_resume(path: Path, backbone: nn.Module, head: nn.Module, optimizer, scheduler, scaler, device):
    if not path.is_file():
        return 1, 0, 0.0, 0.0

    ckpt = torch.load(path, map_location=device)
    load_state_dict_flexible(backbone, ckpt["backbone"])
    load_state_dict_flexible(head, ckpt["head"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt.get("epoch", 0) + 1
    global_step = ckpt.get("global_step", 0)
    best_train_acc = ckpt.get("best_train_acc", 0.0)
    best_test_acc = ckpt.get("best_test_acc", 0.0)
    print(f"Resumed from {path} at epoch {start_epoch - 1}", flush=True)
    return start_epoch, global_step, best_train_acc, best_test_acc


def build_dataset(args, csv_path: str, cache_obj: Optional[StaticAnchoredCache]) -> Emit32TemporalSequenceDataset:
    return Emit32TemporalSequenceDataset(
        csv_path=csv_path,
        path_columns=parse_csv_columns(args.path_columns),
        time_columns=parse_csv_columns(args.time_columns),
        normalize_stats=PRECOMPUTED_STATS,
        time_base_year=args.time_base_year,
        pad_to_multiple=args.pad_to_multiple,
        skip_invalid_samples=args.skip_invalid_samples,
        skip_path_validation=args.skip_path_validation,
        local_file_cache=cache_obj,
        cache_prefetch_rows=args.cache_prefetch_rows,
        input_mode=args.input_mode,
        residual_slots=parse_residual_slots(args.residual_slots),
        residual_clip=args.residual_clip,
    )


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    path_columns = parse_csv_columns(args.path_columns)
    time_columns = parse_csv_columns(args.time_columns)
    residual_slots = parse_residual_slots(args.residual_slots)
    if len(path_columns) != len(time_columns):
        raise ValueError(
            f"--path_columns and --time_columns must have the same length, got {len(path_columns)} and {len(time_columns)}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    use_amp = device.type == "cuda"

    cache_obj = None
    if args.local_cache_mode != "off" and args.local_cache_dir:
        cache_obj = StaticAnchoredCache(
            args.local_cache_dir,
            min_free_gb=args.local_cache_min_free_gb,
            async_mode=args.local_cache_mode == "async",
            max_workers=args.local_cache_async_workers,
        )

    train_ds = build_dataset(args, args.train_csv, cache_obj)
    test_ds = build_dataset(args, args.test_csv, cache_obj)

    if args.local_cache_warmup and cache_obj:
        all_paths = []
        for col in train_ds.path_columns:
            all_paths.extend(train_ds.base_ds.df[col].tolist())
        cache_obj.warm_up(all_paths, max_workers=args.local_cache_workers)

    pin_memory = device.type == "cuda"
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": pin_memory}
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = not args.no_persistent_workers

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )

    cache_desc = "off" if cache_obj is None else f"{args.local_cache_mode}:{args.local_cache_dir}"
    print(
        f"Using device={device}, train_samples={len(train_ds)}, test_samples={len(test_ds)}, "
        f"source_timepoints={len(path_columns)}, "
        f"model_timepoints={output_timepoints(args.input_mode, len(path_columns), residual_slots)}, "
        f"input_mode={args.input_mode}, residual_slots={residual_slots}, "
        f"train_backbone={args.train_backbone}, cache={cache_desc}",
        flush=True,
    )

    panopticon = load_backbone(args.weights, device=device, debug=args.debug)
    backbone = TemporalPanopticonBackbone(
        panopticon,
        num_timepoints=output_timepoints(
            args.input_mode, len(path_columns), residual_slots
        ),
        time_embed_dim=args.time_embed_dim,
    ).to(device)
    head = CLSHead(embed_dim=args.embed_dim, num_classes=2).to(device)

    temporal_params = list(backbone.temporal_parameters())
    panopticon_params = list(backbone.backbone.parameters())
    backbone = maybe_wrap_dataparallel(backbone, device, args.num_gpus, "backbone")
    scaler = GradScaler(enabled=use_amp)

    param_groups = [
        {"params": temporal_params, "lr": args.temporal_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ]
    if args.train_backbone:
        param_groups.insert(0, {"params": panopticon_params, "lr": args.backbone_lr})

    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(args.momentum, 0.999),
    )
    scheduler = build_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    run_name = args.run_name or default_run_name(args)
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    latest_path = ckpt_dir / "ckpt_latest.pth"
    best_train_path = ckpt_dir / "ckpt_best_train.pth"
    best_test_path = ckpt_dir / "ckpt_best_test.pth"
    best_val_ap_path = ckpt_dir / "ckpt_best_val_ap.pth"
    metrics_path = ckpt_dir / "metrics_history.json"

    start_epoch = 1
    global_step = 0
    best_train_acc = 0.0
    best_test_acc = 0.0
    best_val_ap = float("-inf")
    metrics_history = []
    if args.resume and metrics_path.is_file():
        try:
            metrics_history = json.loads(metrics_path.read_text(encoding="utf-8"))
            best_val_ap = max(
                (
                    float(record["test_ap"])
                    for record in metrics_history
                    if record.get("test_ap") is not None
                ),
                default=float("-inf"),
            )
        except Exception:
            metrics_history = []
            best_val_ap = float("-inf")
    if args.resume:
        start_epoch, global_step, best_train_acc, best_test_acc = try_resume(
            latest_path, backbone, head, optimizer, scheduler, scaler, device
        )

    wandb_run = init_wandb(args)

    for epoch in range(start_epoch, args.epochs + 1):
        train_panopticon = args.train_backbone and not (
            args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs
        )
        backbone.train()
        set_backbone_train_state(backbone, train_panopticon)
        head.train()

        total_loss = 0.0
        correct = 0
        total = 0
        train_tp = train_fp = train_fn = train_tn = 0
        for step, (x_dict, labels) in enumerate(train_loader, 1):
            labels = labels.to(device)
            x_dict = recursive_to_device(x_dict, device)

            with autocast(enabled=use_amp):
                feats = backbone(x_dict, is_training=True)
                cls_token = feats["x_norm_clstoken"]
                logits = head(cls_token)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                trainable_params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            train_tp += ((preds == 1) & (labels == 1)).sum().item()
            train_fp += ((preds == 1) & (labels == 0)).sum().item()
            train_fn += ((preds == 0) & (labels == 1)).sum().item()
            train_tn += ((preds == 0) & (labels == 0)).sum().item()
            total += labels.size(0)

            if args.log_interval and step % args.log_interval == 0:
                running_loss = total_loss / total
                running_acc = correct / total
                running_f1_den = 2 * train_tp + train_fp + train_fn
                running_f1 = (2 * train_tp / running_f1_den) if running_f1_den > 0 else 0.0
                print(
                    f"Epoch {epoch} step {step}/{len(train_loader)} "
                    f"train_loss={running_loss:.4f} train_acc={running_acc:.4f} train_f1={running_f1:.4f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train_loss_running": running_loss,
                            "train_acc_running": running_acc,
                            "train_f1_running": running_f1,
                            "step": global_step,
                            "epoch": epoch,
                        }
                    )

            if args.max_train_steps is not None and step >= args.max_train_steps:
                break

        train_loss = total_loss / total
        train_acc = correct / total
        train_f1_den = 2 * train_tp + train_fp + train_fn
        train_f1 = (2 * train_tp / train_f1_den) if train_f1_den > 0 else 0.0

        head.eval()
        backbone.eval()
        correct = 0
        total = 0
        test_loss_total = 0.0
        tp = fp = fn = tn = 0
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for step, (x_dict, labels) in enumerate(test_loader, 1):
                labels = labels.to(device)
                x_dict = recursive_to_device(x_dict, device)
                with autocast(enabled=use_amp):
                    feats = backbone(x_dict, is_training=True)
                    cls_token = feats["x_norm_clstoken"]
                    logits = head(cls_token)
                    loss = criterion(logits, labels)

                test_loss_total += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                probs = F.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.detach().cpu())
                all_targets.append(labels.detach().cpu())
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()

                if args.max_eval_steps is not None and step >= args.max_eval_steps:
                    break

        test_acc = correct / total
        test_loss = test_loss_total / total if total > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        test_f1_den = 2 * tp + fp + fn
        test_f1 = (2 * tp / test_f1_den) if test_f1_den > 0 else 0.0
        all_probs = torch.cat(all_probs) if len(all_probs) > 0 else torch.tensor([])
        all_targets = torch.cat(all_targets) if len(all_targets) > 0 else torch.tensor([])
        test_ap = float("nan")
        test_macro_f1 = float("nan")
        test_best_f1 = float("nan")
        test_best_f1_threshold = float("nan")
        test_recall_at_fpr_01 = float("nan")
        test_recall_at_fpr_05 = float("nan")
        if all_probs.numel() > 0:
            try:
                from sklearn.metrics import (
                    average_precision_score,
                    f1_score,
                    precision_recall_curve,
                    roc_auc_score,
                    roc_curve,
                )

                target_array = all_targets.numpy()
                probability_array = all_probs.numpy()
                test_auroc = float(roc_auc_score(target_array, probability_array))
                test_ap = float(
                    average_precision_score(target_array, probability_array)
                )
                test_macro_f1 = float(
                    f1_score(
                        target_array,
                        (probability_array >= 0.5).astype("int64"),
                        average="macro",
                        zero_division=0,
                    )
                )
                precisions, recalls, thresholds = precision_recall_curve(
                    target_array, probability_array
                )
                f1_curve = np.divide(
                    2.0 * precisions * recalls,
                    precisions + recalls,
                    out=np.zeros_like(precisions),
                    where=(precisions + recalls) > 0,
                )
                best_f1_index = int(np.nanargmax(f1_curve))
                test_best_f1 = float(f1_curve[best_f1_index])
                if thresholds.size:
                    test_best_f1_threshold = float(
                        thresholds[min(best_f1_index, thresholds.size - 1)]
                    )
                roc_fpr, roc_tpr, _ = roc_curve(target_array, probability_array)
                within_01 = roc_tpr[roc_fpr <= 0.01]
                within_05 = roc_tpr[roc_fpr <= 0.05]
                test_recall_at_fpr_01 = (
                    float(within_01.max()) if within_01.size else 0.0
                )
                test_recall_at_fpr_05 = (
                    float(within_05.max()) if within_05.size else 0.0
                )
            except Exception:
                test_auroc = float("nan")
        else:
            test_auroc = float("nan")

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f} "
            f"macro_f1={test_macro_f1:.4f} recall={recall:.4f} fpr={fpr:.4f} "
            f"auroc={test_auroc:.4f} ap={test_ap:.4f} "
            f"best_f1={test_best_f1:.4f}@{test_best_f1_threshold:.4f} "
            f"recall@fpr1%={test_recall_at_fpr_01:.4f} "
            f"recall@fpr5%={test_recall_at_fpr_05:.4f}",
            flush=True,
        )
        epoch_metrics = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_f1": float(train_f1),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_f1": float(test_f1),
            "test_macro_f1": float(test_macro_f1),
            "test_best_f1": float(test_best_f1),
            "test_best_f1_threshold": float(test_best_f1_threshold),
            "test_recall": float(recall),
            "test_fpr": float(fpr),
            "test_recall_at_fpr_01": float(test_recall_at_fpr_01),
            "test_recall_at_fpr_05": float(test_recall_at_fpr_05),
            "test_auroc": float(test_auroc),
            "test_ap": float(test_ap),
        }
        metrics_history.append(epoch_metrics)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        temporary_metrics_path = metrics_path.with_suffix(
            metrics_path.suffix + f".part.{os.getpid()}"
        )
        temporary_metrics_path.write_text(
            json.dumps(metrics_history, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_metrics_path, metrics_path)

        prev_best_train = best_train_acc
        prev_best_test = best_test_acc
        prev_best_val_ap = best_val_ap
        best_train_acc = max(best_train_acc, train_acc)
        best_test_acc = max(best_test_acc, test_acc)
        if not np.isnan(test_ap):
            best_val_ap = max(best_val_ap, test_ap)

        if not args.no_save_checkpoints:
            save_checkpoint(
                latest_path,
                epoch,
                global_step,
                backbone,
                head,
                optimizer,
                scheduler,
                scaler,
                best_train_acc,
                best_test_acc,
                args,
            )

            if train_acc > prev_best_train:
                save_checkpoint(
                    best_train_path,
                    epoch,
                    global_step,
                    backbone,
                    head,
                    optimizer,
                    scheduler,
                    scaler,
                    best_train_acc,
                    best_test_acc,
                    args,
                )

            if test_acc > prev_best_test:
                save_checkpoint(
                    best_test_path,
                    epoch,
                    global_step,
                    backbone,
                    head,
                    optimizer,
                    scheduler,
                    scaler,
                    best_train_acc,
                    best_test_acc,
                    args,
                )

            if test_ap > prev_best_val_ap:
                save_checkpoint(
                    best_val_ap_path,
                    epoch,
                    global_step,
                    backbone,
                    head,
                    optimizer,
                    scheduler,
                    scaler,
                    best_train_acc,
                    best_test_acc,
                    args,
                )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "test_macro_f1": test_macro_f1,
                    "test_best_f1": test_best_f1,
                    "test_best_f1_threshold": test_best_f1_threshold,
                    "test_recall": recall,
                    "test_fpr": fpr,
                    "test_recall_at_fpr_01": test_recall_at_fpr_01,
                    "test_recall_at_fpr_05": test_recall_at_fpr_05,
                    "test_auroc": test_auroc,
                    "test_ap": test_ap,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Panopticon ViT-B/14 + SatMAE-style temporal embeddings for six-timepoint EMIT 32-band classification."
    )
    parser.add_argument("--train_csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test_csv", default=DEFAULT_TEST_CSV)
    parser.add_argument(
        "--weights",
        default="weights/panopticon_vitb14_teacher.pth",
        help='Checkpoint path. Use "none" to train the ViT backbone from scratch.',
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--temporal_lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the Panopticon backbone during finetuning.",
    )
    parser.add_argument("--lr_scheduler", choices=["none", "noam"], default="noam", help="Learning rate scheduler.")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Warmup steps for Noam LR scheduler.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (beta1) for Adam.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader batches prefetched per worker.")
    parser.add_argument("--no_persistent_workers", action="store_true", help="Disable persistent DataLoader workers.")
    parser.add_argument("--pad_to_multiple", type=int, default=14)
    parser.add_argument("--time_base_year", type=int, default=2000)
    parser.add_argument("--time_embed_dim", type=int, default=384)
    parser.add_argument("--path_columns", default=DEFAULT_PATH_COLUMNS)
    parser.add_argument("--time_columns", default=DEFAULT_TIME_COLUMNS)
    parser.add_argument(
        "--input_mode",
        choices=INPUT_MODES,
        default="raw",
        help=(
            "raw keeps all frames; current keeps t0; residual uses normalized "
            "t0-history differences; current_residual keeps t0 plus differences."
        ),
    )
    parser.add_argument(
        "--residual_slots",
        default="path_prev1,path_seasonal",
        help="Comma-separated history path-column names used to form t0-history residuals.",
    )
    parser.add_argument(
        "--residual_clip",
        type=float,
        default=5.0,
        help="Clip normalized temporal residuals to +/- this value; <=0 disables clipping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260727,
        help="Training/data-loader seed used for matched input-mode comparisons.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true", help="Print stage timings and device info.")
    parser.add_argument("--log_interval", type=int, default=100, help="Print training progress every N steps.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Optional cap on train batches per epoch.")
    parser.add_argument("--max_eval_steps", type=int, default=None, help="Optional cap on eval batches per epoch.")
    parser.add_argument(
        "--skip_invalid_samples",
        action="store_true",
        help="Drop CSV rows whose TIFFs are missing or unreadable instead of failing mid-epoch.",
    )
    parser.add_argument(
        "--skip_path_validation",
        action="store_true",
        help="Skip the startup pass that validates every TIFF path when --skip_invalid_samples is set.",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument(
        "--wandb_run_name",
        default="dino_classifier_head_emit32_temporal_satmae",
        help="Optional WandB run name.",
    )
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Base directory to store checkpoints.")
    parser.add_argument(
        "--run_name",
        default=None,
        help="Run name for checkpoint subfolder. Defaults to <train_csv_stem>__<test_csv_stem>__satmae_time__mode.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint for this run name.")
    parser.add_argument(
        "--no_save_checkpoints",
        action="store_true",
        help="Write per-epoch metrics but skip model checkpoint files.",
    )
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        help="If set, train the Panopticon ViT backbone in addition to temporal adapter and classifier head.",
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Initial epochs to freeze the Panopticon backbone when --train_backbone is set.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm; <=0 disables.")
    parser.add_argument("--embed_dim", type=int, default=768, help="Backbone output dimension.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Use DataParallel when device is CUDA.")
    parser.add_argument(
        "--local_cache_dir",
        default="/diniuvol/yuyao/emit32_temporal_cache",
        help="Directory for lazy local TIFF cache. Set empty string with --local_cache_mode off to disable.",
    )
    parser.add_argument(
        "--local_cache_mode",
        choices=["off", "sync", "async"],
        default="async",
        help="sync blocks on cache misses; async copies in background and reads remote until cached.",
    )
    parser.add_argument(
        "--cache_prefetch_rows",
        type=int,
        default=4,
        help="Rows to schedule for async cache prefetch from each sampled row.",
    )
    parser.add_argument(
        "--local_cache_async_workers",
        type=int,
        default=2,
        help="Background copy threads per DataLoader worker in async cache mode.",
    )
    parser.add_argument("--local_cache_warmup", action="store_true", help="Pre-copy training files to cache.")
    parser.add_argument("--local_cache_workers", type=int, default=12, help="Workers for full cache warmup.")
    parser.add_argument("--local_cache_min_free_gb", type=float, default=10.0)
    args = parser.parse_args()
    main(args)
