import argparse
import hashlib
import json
import os
import random
import re
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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

from thirdparty.dinov2.models.panopticon import get_1d_sincos_pos_embed_from_grid_torch  # noqa: E402
from research.pretraining_20260727.temporal_input_modes import (  # noqa: E402
    INPUT_MODES,
    output_timepoints,
    parse_residual_slots,
    transform_temporal_sample,
)


TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
DEFAULT_TRAIN_CSV = (
    "/home/yuyao/panopticon/Upgraded_dataset/s5p_6time_temporal_cutoff_2025_12_24/"
    "s5p_6time_train.csv"
)
DEFAULT_TEST_CSV = (
    "/home/yuyao/panopticon/Upgraded_dataset/s5p_6time_temporal_cutoff_2025_12_24/"
    "s5p_6time_test.csv"
)
DEFAULT_IMAGE_PATH_COLUMN = "image_path"
DEFAULT_TIME_PATH_COLUMNS = "t0_path,prev1_path,prev2_path,prev3_path,seasonal_path,year_path"
FALLBACK_TIME_OFFSETS_DAYS = {
    "t0": 0,
    "prev1": -1,
    "prev2": -2,
    "prev3": -3,
    "seasonal": -90,
    "year": -365,
}
S5P_NC_TIMESTAMP_RE = re.compile(r"(\d{8}T\d{6})")
PRECOMPUTED_STATS = None


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

def parse_csv_columns(value: str) -> Tuple[str, ...]:
    columns = tuple(col.strip() for col in value.split(",") if col.strip())
    if not columns:
        raise ValueError("Expected at least one CSV column name.")
    return columns


def valid_text(value) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def parse_s5p_nc_timestamp(path: str) -> Optional[pd.Timestamp]:
    if not valid_text(path):
        return None
    match = S5P_NC_TIMESTAMP_RE.search(Path(str(path)).name)
    if match is None:
        return None
    timestamp = pd.to_datetime(match.group(1), format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp


def parse_s5p_nc_timestamp_series(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.rsplit("/", n=1).str[-1]
    stamps = text.str.extract(S5P_NC_TIMESTAMP_RE.pattern, expand=False)
    return pd.to_datetime(stamps, format="%Y%m%dT%H%M%S", utc=True, errors="coerce")


def timestamp_features(timestamp: pd.Timestamp, *, base_year: int) -> torch.Tensor:
    return torch.tensor(
        [timestamp.year - base_year, timestamp.month - 1, timestamp.hour],
        dtype=torch.float32,
    )


def _extract_npz_array(np_obj: np.lib.npyio.NpzFile, path: str, data_key: Optional[str]) -> np.ndarray:
    if data_key is not None:
        if data_key not in np_obj:
            raise KeyError(f"Key '{data_key}' not found in NPZ file {path}")
        return np.array(np_obj[data_key])

    preferred_keys = ("ch4", "image", "imgs", "arr_0", "data")
    for key in preferred_keys:
        if key in np_obj:
            arr = np.array(np_obj[key])
            if arr.ndim in (2, 3) and arr.dtype.kind in {"f", "i", "u", "b"}:
                return arr
    for key in np_obj.files:
        if key.lower() in {"meta", "metadata"}:
            continue
        try:
            arr = np.array(np_obj[key])
        except ValueError:
            continue
        if arr.ndim in (2, 3) and arr.dtype.kind in {"f", "i", "u", "b"}:
            return arr
    raise ValueError(
        f"Failed to infer S5P image array from NPZ file {path}. "
        f"Available keys: {list(np_obj.files)}. Consider setting --data_key."
    )


def _to_time_stack(
    arr: np.ndarray,
    path: str,
    *,
    expected_timepoints: int,
    channel_last: bool,
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 0)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported S5P NPZ array shape {arr.shape} at {path}")

    if channel_last or (arr.shape[-1] == expected_timepoints and arr.shape[0] != expected_timepoints):
        arr = np.transpose(arr, (2, 0, 1))

    if arr.shape[0] != expected_timepoints:
        raise ValueError(
            f"Expected S5P NPZ array with {expected_timepoints} timepoints, got shape {arr.shape} at {path}"
        )
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def load_s5p_time_stack(
    path: str,
    *,
    data_key: Optional[str] = "ch4",
    channel_last: bool = False,
    allow_pickle: bool = False,
    expected_timepoints: int = 6,
    local_file_cache: Optional[StaticAnchoredCache] = None,
) -> np.ndarray:
    if not valid_text(path):
        raise ValueError("Encountered empty S5P NPZ path in CSV.")
    path = str(path).strip()
    if local_file_cache is not None:
        path = local_file_cache.ensure_local(path)
    if not path.lower().endswith(".npz"):
        raise ValueError(f"S5P temporal dataset expects .npz input, got path: {path}")

    np_obj = np.load(path, allow_pickle=allow_pickle)
    try:
        if isinstance(np_obj, np.lib.npyio.NpzFile):
            arr = _extract_npz_array(np_obj, path, data_key)
        else:
            arr = np.asarray(np_obj)
    finally:
        if isinstance(np_obj, np.lib.npyio.NpzFile):
            np_obj.close()
    return _to_time_stack(
        arr,
        path,
        expected_timepoints=expected_timepoints,
        channel_last=channel_last,
    )


def compute_s5p_temporal_stats(
    csv_path: str,
    *,
    path_column: str,
    n_samples: int,
    seed: int,
    num_workers: int = 1,
    local_file_cache: Optional[StaticAnchoredCache] = None,
    data_key: Optional[str] = "ch4",
    channel_last: bool = False,
    allow_pickle: bool = False,
    num_timepoints: int = 6,
) -> Tuple[Sequence[float], Sequence[float]]:
    """Compute one shared scalar mean/std over finite pixels from all six S5P timepoints."""

    df = pd.read_csv(csv_path, low_memory=False)
    if path_column not in df.columns:
        raise ValueError(f"Missing S5P image path column '{path_column}' in {csv_path}")
    if len(df) == 0:
        raise ValueError(f"Cannot compute stats from empty CSV: {csv_path}")

    sample_count = min(max(1, int(n_samples)), len(df))
    rng = random.Random(seed)
    indices = rng.sample(range(len(df)), sample_count)
    num_workers = max(1, int(num_workers))
    progress_every = max(1, 100)
    total_sum = torch.zeros((), dtype=torch.float64)
    total_sq_sum = torch.zeros((), dtype=torch.float64)
    total_count = torch.zeros((), dtype=torch.float64)

    print(
        f"[Stats] Computing S5P shared scalar normalization from {sample_count} rows "
        f"x {num_timepoints} timepoints; finite pixels only; workers={num_workers}.",
        flush=True,
    )

    def _sample_stats(row_idx: int):
        arr = load_s5p_time_stack(
            df.iloc[row_idx][path_column],
            data_key=data_key,
            channel_last=channel_last,
            allow_pickle=allow_pickle,
            expected_timepoints=num_timepoints,
            local_file_cache=local_file_cache,
        )
        valid = np.isfinite(arr)
        count = int(valid.sum())
        if count == 0:
            return 0.0, 0.0, 0
        values = arr[valid].astype(np.float64, copy=False)
        return float(values.sum()), float(np.square(values).sum()), count

    def _accumulate(result) -> None:
        sums, sq_sums, counts = result
        total_sum.add_(float(sums))
        total_sq_sum.add_(float(sq_sums))
        total_count.add_(float(counts))

    if num_workers == 1:
        for job_idx, row_idx in enumerate(indices, 1):
            _accumulate(_sample_stats(row_idx))
            if job_idx % progress_every == 0 or job_idx == len(indices):
                print(f"[Stats] processed NPZ samples {job_idx}/{len(indices)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_sample_stats, row_idx) for row_idx in indices]
            for job_idx, future in enumerate(as_completed(futures), 1):
                _accumulate(future.result())
                if job_idx % progress_every == 0 or job_idx == len(indices):
                    print(f"[Stats] processed NPZ samples {job_idx}/{len(indices)}", flush=True)

    if total_count.item() <= 0:
        raise RuntimeError("No finite S5P pixels found while computing normalization stats.")

    mean = total_sum / total_count
    variance = total_sq_sum / total_count - mean * mean
    std = torch.sqrt(torch.clamp(variance, min=1e-12))
    mean_list = [float(mean.float().item())]
    std_list = [float(std.float().item())]
    print(f"[Stats] shared_mean={mean_list}", flush=True)
    print(f"[Stats] shared_std={std_list}", flush=True)
    return mean_list, std_list


class S5PTemporalSequenceDataset(Dataset):
    """
    Six-timepoint S5P NPZ dataset.

    The input NPZ stores one array as ch4.shape == (T, H, W), where T=6.
    Returns x_dict with:
      imgs: (T, 1, H, W)
      chn_ids: (T, 1)
      timestamps: (T, 3) containing year_offset, month_index, hour
    """

    def __init__(
        self,
        csv_path: str,
        *,
        path_column: str,
        time_path_columns: Sequence[str],
        normalize_stats,
        label_column: str = "label",
        time_base_year: int = 2000,
        pad_to_multiple: Optional[int] = 14,
        pad_value: float = 0.0,
        skip_invalid_samples: bool = False,
        local_file_cache: Optional[StaticAnchoredCache] = None,
        cache_prefetch_rows: int = 0,
        data_key: Optional[str] = "ch4",
        channel_last: bool = False,
        allow_pickle: bool = False,
        nan_to_num: float = 0.0,
        chn_id_value: float = 0.0,
        max_retries: int = 5,
        input_mode: str = "raw",
        residual_slots: Sequence[str] = ("prev1_path", "year_path"),
        residual_clip: float = 5.0,
    ):
        self.df = pd.read_csv(csv_path, low_memory=False)
        if len(self.df) == 0:
            raise ValueError(f"Empty S5P CSV: {csv_path}")
        self.path_column = path_column
        self.time_path_columns = tuple(time_path_columns)
        self.num_timepoints = len(TIMEPOINTS)
        self.label_column = label_column
        self.time_base_year = int(time_base_year)
        self.pad_to_multiple = pad_to_multiple
        self.pad_value = float(pad_value)
        self.skip_invalid_samples = bool(skip_invalid_samples)
        self.local_file_cache = local_file_cache
        self.cache_prefetch_rows = max(0, int(cache_prefetch_rows))
        self.data_key = data_key
        self.channel_last = bool(channel_last)
        self.allow_pickle = bool(allow_pickle)
        self.nan_to_num = float(nan_to_num)
        self.max_retries = max(1, int(max_retries))
        self.input_mode = str(input_mode)
        self.residual_slots = tuple(residual_slots)
        self.residual_clip = float(residual_clip)
        self.chn_ids = torch.full((self.num_timepoints, 1), float(chn_id_value), dtype=torch.float32)

        if len(self.time_path_columns) != self.num_timepoints:
            raise ValueError(
                f"Expected {self.num_timepoints} timestamp path columns, got {len(self.time_path_columns)}: "
                f"{self.time_path_columns}"
            )
        self._validate_columns()
        self._mean = None
        self._std = None
        if normalize_stats is not None:
            mean, std = normalize_stats
            mean_t = torch.as_tensor(mean, dtype=torch.float32).flatten()
            std_t = torch.as_tensor(std, dtype=torch.float32).flatten()
            if mean_t.numel() != 1 or std_t.numel() != 1:
                raise ValueError(
                    "S5P temporal normalization expects one shared scalar mean/std. "
                    f"Got mean_len={mean_t.numel()} std_len={std_t.numel()}."
                )
            self._mean = mean_t.view(1, 1, 1)
            self._std = torch.clamp(std_t, min=1e-6).view(1, 1, 1)

        self.timestamps, self.timestamp_source_counts = self._build_timestamp_tensor(self.df)
        print(
            f"[Data] {csv_path}: timestamp sources "
            f"nc_filename={self.timestamp_source_counts['nc_filename']} "
            f"fallback={self.timestamp_source_counts['fallback']}",
            flush=True,
        )

    def _validate_columns(self) -> None:
        required = [self.path_column, self.label_column, "plume_time", *self.time_path_columns]
        missing = [column for column in required if column not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in S5P CSV: {missing}")
        for column in [self.path_column, self.label_column, "plume_time"]:
            invalid = self.df[column].isna()
            if invalid.any():
                examples = self.df.loc[invalid, [self.path_column, self.label_column, "plume_time"]].head(5)
                raise ValueError(f"Column '{column}' has {int(invalid.sum())} missing values. Examples: {examples}")

    def _build_timestamp_tensor(self, df: pd.DataFrame) -> Tuple[torch.Tensor, dict]:
        plume_times = pd.to_datetime(df["plume_time"], utc=True, errors="coerce")
        if plume_times.isna().any():
            bad_rows = df.loc[plume_times.isna(), ["plume_id", "plume_time"]].head(5).to_dict("records")
            raise ValueError(f"Invalid plume_time values. Examples: {bad_rows}")

        source_counts = {"nc_filename": 0, "fallback": 0}
        time_features = []
        for timepoint, column in zip(TIMEPOINTS, self.time_path_columns):
            timestamps = parse_s5p_nc_timestamp_series(df[column])
            fallback = plume_times + pd.Timedelta(days=FALLBACK_TIME_OFFSETS_DAYS[timepoint])
            fallback_mask = timestamps.isna()
            source_counts["fallback"] += int(fallback_mask.sum())
            source_counts["nc_filename"] += int((~fallback_mask).sum())
            timestamps = timestamps.fillna(fallback)

            year = torch.tensor(
                timestamps.dt.year.to_numpy(dtype="int64") - self.time_base_year,
                dtype=torch.float32,
            )
            month = torch.tensor(timestamps.dt.month.to_numpy(dtype="int64") - 1, dtype=torch.float32)
            hour = torch.tensor(timestamps.dt.hour.to_numpy(dtype="int64"), dtype=torch.float32)
            time_features.append(torch.stack((year, month, hour), dim=1))
        return torch.stack(time_features, dim=1), source_counts

    def __len__(self):
        return len(self.df)

    def _prefetch_paths(self, idx: int) -> None:
        if self.local_file_cache is None or self.cache_prefetch_rows <= 0:
            return
        if not getattr(self.local_file_cache, "async_mode", False):
            return
        stop = min(len(self.df), idx + self.cache_prefetch_rows)
        for row_idx in range(idx, stop):
            self.local_file_cache.schedule(self.df.iloc[row_idx][self.path_column])

    def _pad_to_multiple(self, img: torch.Tensor, multiple: int) -> torch.Tensor:
        _, height, width = img.shape
        target_h = int(np.ceil(height / multiple) * multiple)
        target_w = int(np.ceil(width / multiple) * multiple)
        pad_h = target_h - height
        pad_w = target_w - width
        if pad_h == 0 and pad_w == 0:
            return img
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=self.pad_value)

    def _get_one(self, idx):
        self._prefetch_paths(idx)
        row = self.df.iloc[idx]
        label = int(row[self.label_column])
        stack = load_s5p_time_stack(
            row[self.path_column],
            data_key=self.data_key,
            channel_last=self.channel_last,
            allow_pickle=self.allow_pickle,
            expected_timepoints=self.num_timepoints,
            local_file_cache=self.local_file_cache,
        )
        img = torch.from_numpy(stack).to(dtype=torch.float32)
        if self._mean is not None and self._std is not None:
            img = (img - self._mean) / self._std
        img = torch.nan_to_num(img, nan=self.nan_to_num, posinf=self.nan_to_num, neginf=self.nan_to_num)
        if self.pad_to_multiple is not None:
            img = self._pad_to_multiple(img, int(self.pad_to_multiple))

        x_dict = {
            "imgs": img.unsqueeze(1),
            "chn_ids": self.chn_ids.clone(),
            "timestamps": self.timestamps[idx],
        }
        x_dict = transform_temporal_sample(
            x_dict,
            path_columns=self.time_path_columns,
            input_mode=self.input_mode,
            residual_slots=self.residual_slots,
            residual_clip=self.residual_clip,
        )
        return x_dict, label

    def __getitem__(self, idx):
        attempts = 0
        last_exc: Optional[Exception] = None
        while attempts < self.max_retries:
            try:
                return self._get_one(idx)
            except Exception as exc:
                if not self.skip_invalid_samples:
                    raise
                last_exc = exc
                attempts += 1
                if attempts >= self.max_retries:
                    raise RuntimeError(f"Exceeded {self.max_retries} retries for temporal index {idx}") from exc
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

    Each S5P timepoint is patchified independently by the shared PanopticonPE
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
    return f"{train_stem}__{test_stem}__satmae_time__{args.input_mode}__{mode}"


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


def build_dataset(
    args,
    csv_path: str,
    cache_obj: Optional[StaticAnchoredCache],
    normalize_stats,
) -> S5PTemporalSequenceDataset:
    return S5PTemporalSequenceDataset(
        csv_path=csv_path,
        path_column=args.path_column,
        time_path_columns=parse_csv_columns(args.time_path_columns),
        normalize_stats=normalize_stats,
        time_base_year=args.time_base_year,
        pad_to_multiple=args.pad_to_multiple,
        pad_value=args.pad_value,
        skip_invalid_samples=args.skip_invalid_samples,
        local_file_cache=cache_obj,
        cache_prefetch_rows=args.cache_prefetch_rows,
        data_key=args.data_key,
        channel_last=args.channel_last,
        allow_pickle=args.allow_pickle,
        nan_to_num=args.nan_to_num,
        chn_id_value=args.chn_id_value,
        max_retries=args.max_retries,
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

    time_path_columns = parse_csv_columns(args.time_path_columns)
    residual_slots = parse_residual_slots(args.residual_slots)
    if len(time_path_columns) != len(TIMEPOINTS):
        raise ValueError(
            f"--time_path_columns must contain {len(TIMEPOINTS)} columns, got {len(time_path_columns)}"
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

    if args.use_precomputed_stats:
        if PRECOMPUTED_STATS is None:
            raise ValueError(
                "No S5P PRECOMPUTED_STATS are defined. Omit --use_precomputed_stats to compute "
                "shared scalar finite-pixel stats from the train CSV."
            )
        norm_stats = PRECOMPUTED_STATS
        print(f"[Stats] Using precomputed S5P stats: mean={norm_stats[0]} std={norm_stats[1]}", flush=True)
    else:
        stats_cache_obj = cache_obj
        if cache_obj is not None and getattr(cache_obj, "async_mode", False):
            stats_cache_obj = None
        norm_stats = compute_s5p_temporal_stats(
            args.train_csv,
            path_column=args.path_column,
            n_samples=args.stats_samples,
            seed=args.stats_seed,
            num_workers=args.stats_workers,
            local_file_cache=stats_cache_obj,
            data_key=args.data_key,
            channel_last=args.channel_last,
            allow_pickle=args.allow_pickle,
            num_timepoints=len(TIMEPOINTS),
        )

    train_ds = build_dataset(args, args.train_csv, cache_obj, norm_stats)
    test_ds = build_dataset(args, args.test_csv, cache_obj, norm_stats)

    if args.local_cache_warmup and cache_obj:
        all_paths = train_ds.df[train_ds.path_column].dropna().astype(str).tolist()
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
        f"source_timepoints={len(TIMEPOINTS)}, "
        f"model_timepoints={output_timepoints(args.input_mode, len(TIMEPOINTS), residual_slots)}, "
        f"input_mode={args.input_mode}, residual_slots={residual_slots}, "
        f"train_backbone={args.train_backbone}, cache={cache_desc}",
        flush=True,
    )

    panopticon = load_backbone(args.weights, device=device, debug=args.debug)
    backbone = TemporalPanopticonBackbone(
        panopticon,
        num_timepoints=output_timepoints(
            args.input_mode, len(TIMEPOINTS), residual_slots
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
        description="Panopticon ViT-B/14 + SatMAE-style temporal embeddings for six-timepoint S5P NPZ classification."
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
    parser.add_argument("--pad_value", type=float, default=0.0, help="Pad value when extending to pad_to_multiple.")
    parser.add_argument("--time_base_year", type=int, default=2000)
    parser.add_argument("--time_embed_dim", type=int, default=384)
    parser.add_argument(
        "--stats_samples",
        type=int,
        default=200,
        help="Rows sampled from train CSV to compute one shared S5P finite-pixel normalization scalar.",
    )
    parser.add_argument(
        "--stats_workers",
        type=int,
        default=8,
        help="Parallel NPZ readers for normalization-stat computation.",
    )
    parser.add_argument("--stats_seed", type=int, default=0, help="Random seed for normalization-stat row sampling.")
    parser.add_argument(
        "--seed",
        type=int,
        default=20260727,
        help="Training/data-loader seed used for matched input-mode comparisons.",
    )
    parser.add_argument(
        "--use_precomputed_stats",
        action="store_true",
        help="Skip runtime stat computation and use the PRECOMPUTED_STATS constants.",
    )
    parser.add_argument("--path_column", default=DEFAULT_IMAGE_PATH_COLUMN, help="CSV column containing cropped S5P NPZ paths.")
    parser.add_argument(
        "--time_path_columns",
        default=DEFAULT_TIME_PATH_COLUMNS,
        help="CSV columns containing source S5P NC paths used to parse acquisition timestamps.",
    )
    parser.add_argument(
        "--input_mode",
        choices=INPUT_MODES,
        default="raw",
        help=(
            "raw keeps all six frames; current keeps t0; residual uses normalized "
            "t0-history differences; current_residual keeps t0 plus differences."
        ),
    )
    parser.add_argument(
        "--residual_slots",
        default="prev1_path,year_path",
        help="Comma-separated time-path column names used to form t0-history residuals.",
    )
    parser.add_argument(
        "--residual_clip",
        type=float,
        default=5.0,
        help="Clip normalized temporal residuals to +/- this value; <=0 disables clipping.",
    )
    parser.add_argument("--data_key", default="ch4", help="NPZ key containing the six-time S5P array.")
    parser.add_argument("--channel_last", action="store_true", help="Set if NPZ arrays are HWT instead of THW.")
    parser.add_argument(
        "--allow_pickle",
        action="store_true",
        help="Allow pickle while loading NPZ. Not needed for the default ch4 key.",
    )
    parser.add_argument(
        "--nan_to_num",
        type=float,
        default=0.0,
        help="Value used for NaN/Inf pixels after normalization.",
    )
    parser.add_argument(
        "--chn_id_value",
        type=float,
        default=0.0,
        help="Constant Panopticon channel id for the single S5P CH4 channel.",
    )
    parser.add_argument("--max_retries", type=int, default=5, help="Retries when --skip_invalid_samples is enabled.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true", help="Print stage timings and device info.")
    parser.add_argument("--log_interval", type=int, default=100, help="Print training progress every N steps.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Optional cap on train batches per epoch.")
    parser.add_argument("--max_eval_steps", type=int, default=None, help="Optional cap on eval batches per epoch.")
    parser.add_argument(
        "--skip_invalid_samples",
        action="store_true",
        help="Retry another row when an S5P NPZ is missing or unreadable instead of failing mid-epoch.",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument(
        "--wandb_run_name",
        default="dino_classifier_head_s5p_temporal_satmae",
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
        default="/diniuvol/yuyao/s5p_temporal_cache",
        help="Directory for lazy local NPZ cache. Set empty string with --local_cache_mode off to disable.",
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
