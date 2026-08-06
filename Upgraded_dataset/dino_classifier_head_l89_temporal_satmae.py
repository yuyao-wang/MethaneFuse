import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Tuple

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

from thirdparty.dinov2.data.datasets.s2_csv import S2CsvDataset, _SkipSample  # noqa: E402
from thirdparty.dinov2.models.panopticon import get_1d_sincos_pos_embed_from_grid_torch  # noqa: E402
from research.pretraining_20260727.temporal_input_modes import (  # noqa: E402
    INPUT_MODES,
    output_timepoints,
    parse_residual_slots,
    transform_temporal_sample,
)


# Landsat 8/9 SR bands (B1-B7) mean/std sampled from 200 train rows x 6 timepoints; zero pixels ignored.
PRECOMPUTED_STATS = (
    [10331.4521484375, 11042.103515625, 12900.3623046875, 14452.6201171875, 18574.63671875, 19728.4296875, 17492.8671875],
    [2259.21142578125, 2469.471435546875, 3058.9375, 4151.76806640625, 3823.086181640625, 5154.72265625, 5008.92138671875],
)

DEFAULT_TRAIN_CSV = (
    "/mnt/engg-niulab/Yuyao/preprocessed_512/L89/l89_6time_temporal_16_resized_to_224/"
    "L89_temporal_train.csv"
)
DEFAULT_TEST_CSV = (
    "/mnt/engg-niulab/Yuyao/preprocessed_512/L89/l89_6time_temporal_16_resized_to_224/"
    "L89_temporal_test.csv"
)
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
        dst.parent.mkdir(parents=True, exist_ok=True)
        lock_path = dst.with_suffix(dst.suffix + ".lock")
        lock_fd = None
        deadline = time.monotonic() + 120.0

        while lock_fd is None:
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if dst.exists():
                    return str(dst)
                try:
                    if time.time() - lock_path.stat().st_mtime > 300.0:
                        lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    return original
                time.sleep(0.05)

        os.close(lock_fd)
        tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            if dst.exists():
                return str(dst)
            if self._get_free_space() < self.min_free_bytes:
                return original
            shutil.copy2(original, tmp)
            os.replace(tmp, dst)
        except Exception:
            with suppress(FileNotFoundError):
                tmp.unlink()
            return original
        finally:
            with suppress(FileNotFoundError):
                lock_path.unlink()
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


def parse_csv_columns(value: str) -> Tuple[str, ...]:
    columns = tuple(col.strip() for col in value.split(",") if col.strip())
    if not columns:
        raise ValueError("Expected at least one CSV column name.")
    return columns


def parse_band_indices(value: str) -> Tuple[int, ...]:
    indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not indices:
        raise ValueError("Expected at least one band index.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Band indices must be unique, got {indices}")
    if any(index < 0 for index in indices):
        raise ValueError(f"Band indices must be non-negative, got {indices}")
    return indices


class TemporalEvidenceLosses(NamedTuple):
    """Loss components for the opt-in temporal-evidence objective."""

    total: torch.Tensor
    full_ce: torch.Tensor
    current_ce: Optional[torch.Tensor]
    margin: Optional[torch.Tensor]
    margin_satisfaction: Optional[torch.Tensor]


def slice_current_temporal_view(
    x_dict: dict,
    *,
    input_mode: str,
    path_columns: Sequence[str],
) -> dict:
    """Return a batched current-only view without mutating ``x_dict``.

    ``raw`` tensors preserve source-column order, so the current index is
    resolved from ``path_t0``/``t0_path`` (falling back to the first source
    column, matching ``transform_temporal_sample``). ``current_residual`` and
    ``current`` always place the absolute current observation first. A pure
    ``residual`` input has no absolute current observation and is rejected.
    """

    if input_mode not in INPUT_MODES:
        raise ValueError(
            f"Unsupported input_mode={input_mode!r}; expected one of {INPUT_MODES}"
        )
    if input_mode == "residual":
        raise ValueError(
            "Cannot construct a current-only view from input_mode='residual': "
            "it contains only t0-history differences. Use 'raw' or "
            "'current_residual' for temporal evidence training."
        )

    required = ("imgs", "chn_ids", "timestamps")
    missing = [key for key in required if key not in x_dict]
    if missing:
        raise ValueError(f"Current-view input is missing tensor keys: {missing}")

    imgs = x_dict["imgs"]
    chn_ids = x_dict["chn_ids"]
    timestamps = x_dict["timestamps"]
    if imgs.ndim != 5 or chn_ids.ndim != 3 or timestamps.ndim != 3:
        raise ValueError(
            "Expected batched tensors imgs=(B,T,C,H,W), chn_ids=(B,T,C), "
            f"timestamps=(B,T,3); got {tuple(imgs.shape)}, "
            f"{tuple(chn_ids.shape)}, {tuple(timestamps.shape)}"
        )
    timepoints = imgs.shape[1]
    if chn_ids.shape[1] != timepoints or timestamps.shape[1] != timepoints:
        raise ValueError(
            "Temporal tensor lengths disagree: "
            f"imgs={timepoints}, chn_ids={chn_ids.shape[1]}, "
            f"timestamps={timestamps.shape[1]}"
        )

    current_index = 0
    if input_mode == "raw":
        if len(path_columns) != timepoints:
            raise ValueError(
                f"raw path column count {len(path_columns)} != tensor "
                f"timepoints {timepoints}"
            )
        slot_to_index = {
            str(column): index for index, column in enumerate(path_columns)
        }
        if "path_t0" in slot_to_index:
            current_index = slot_to_index["path_t0"]
        elif "t0_path" in slot_to_index:
            current_index = slot_to_index["t0_path"]

    if current_index >= timepoints:
        raise ValueError(
            f"Current-view index {current_index} is outside {timepoints} timepoints"
        )

    current = dict(x_dict)
    current["imgs"] = imgs[:, current_index : current_index + 1]
    current["chn_ids"] = chn_ids[:, current_index : current_index + 1]
    current["timestamps"] = timestamps[:, current_index : current_index + 1]
    return current


def label_conditioned_temporal_margin_loss(
    full_logits: torch.Tensor,
    current_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encourage temporal evidence to move the binary log-odds by label.

    Positive examples should have larger ``class1 - class0`` log-odds with the
    full temporal input; negative examples should have smaller log-odds. Using
    the logit difference makes the objective invariant to an arbitrary common
    shift of both logits. The current log-odds are a stop-gradient anchor for
    this term, while their matched CE remains independently trainable in
    ``temporal_evidence_classification_objective``.
    """

    if full_logits.ndim != 2 or full_logits.shape[1] != 2:
        raise ValueError(
            f"Expected full_logits=(B,2), got {tuple(full_logits.shape)}"
        )
    if current_logits.shape != full_logits.shape:
        raise ValueError(
            "full_logits and current_logits must have identical shape, got "
            f"{tuple(full_logits.shape)} and {tuple(current_logits.shape)}"
        )
    if labels.ndim != 1 or labels.shape[0] != full_logits.shape[0]:
        raise ValueError(
            f"Expected labels=(B,), got {tuple(labels.shape)} for "
            f"batch size {full_logits.shape[0]}"
        )
    if full_logits.shape[0] == 0:
        raise ValueError("Temporal evidence margin requires a non-empty batch")
    invalid_labels = (labels != 0) & (labels != 1)
    if bool(invalid_labels.any()):
        raise ValueError("Temporal evidence margin requires binary labels 0/1")
    if not math.isfinite(float(margin)) or margin < 0:
        raise ValueError(f"margin must be finite and non-negative, got {margin}")

    sign = labels.to(device=full_logits.device, dtype=full_logits.dtype)
    sign = sign.mul(2.0).sub(1.0)
    full_log_odds = full_logits[:, 1] - full_logits[:, 0]
    current_log_odds = current_logits[:, 1] - current_logits[:, 0]
    signed_delta = sign * (full_log_odds - current_log_odds.detach())
    margin_loss = F.softplus(float(margin) - signed_delta).mean()
    satisfaction = (
        signed_delta.detach().ge(float(margin)).to(dtype=torch.float32).mean()
    )
    return margin_loss, satisfaction, signed_delta


def temporal_evidence_classification_objective(
    full_logits: torch.Tensor,
    labels: torch.Tensor,
    criterion,
    *,
    enabled: bool,
    current_logits: Optional[torch.Tensor] = None,
    margin: float = 0.25,
    current_ce_weight: float = 1.0,
    margin_weight: float = 1.0,
) -> TemporalEvidenceLosses:
    """Compute legacy CE or the strictly opt-in temporal evidence objective."""

    full_ce = criterion(full_logits, labels)
    if not enabled:
        return TemporalEvidenceLosses(
            total=full_ce,
            full_ce=full_ce,
            current_ce=None,
            margin=None,
            margin_satisfaction=None,
        )
    if current_logits is None:
        raise ValueError(
            "current_logits are required when temporal evidence margin is enabled"
        )
    if not math.isfinite(float(current_ce_weight)) or current_ce_weight <= 0:
        raise ValueError(
            "current_ce_weight must be finite and positive when enabled, got "
            f"{current_ce_weight}"
        )
    if not math.isfinite(float(margin_weight)) or margin_weight < 0:
        raise ValueError(
            "margin_weight must be finite and non-negative when enabled, got "
            f"{margin_weight}"
        )

    current_ce = criterion(current_logits, labels)
    temporal_margin, satisfaction, _ = label_conditioned_temporal_margin_loss(
        full_logits,
        current_logits,
        labels,
        margin=margin,
    )
    total = (
        full_ce
        + float(current_ce_weight) * current_ce
        + float(margin_weight) * temporal_margin
    )
    return TemporalEvidenceLosses(
        total=total,
        full_ce=full_ce,
        current_ce=current_ce,
        margin=temporal_margin,
        margin_satisfaction=satisfaction,
    )


def validate_temporal_evidence_configuration(
    *,
    enabled: bool,
    input_mode: str,
    source_timepoints: int,
    residual_slots: Sequence[str],
    margin: float,
    current_ce_weight: float,
    margin_weight: float,
) -> None:
    """Fail early for modes that cannot provide full-vs-current evidence."""

    if not enabled:
        return
    if input_mode not in {"raw", "current_residual"}:
        reason = (
            "contains no absolute current observation"
            if input_mode == "residual"
            else "contains no additional temporal evidence beyond current"
        )
        raise ValueError(
            "Temporal evidence margin supports only input_mode='raw' or "
            f"'current_residual'; input_mode={input_mode!r} {reason}."
        )
    model_timepoints = output_timepoints(
        input_mode, source_timepoints, residual_slots
    )
    if model_timepoints < 2:
        raise ValueError(
            "Temporal evidence margin requires at least two full-view "
            f"timepoints, got {model_timepoints}"
        )
    if not math.isfinite(float(margin)) or margin < 0:
        raise ValueError(f"margin must be finite and non-negative, got {margin}")
    if not math.isfinite(float(current_ce_weight)) or current_ce_weight <= 0:
        raise ValueError(
            f"current_ce_weight must be finite and positive, got {current_ce_weight}"
        )
    if not math.isfinite(float(margin_weight)) or margin_weight < 0:
        raise ValueError(
            f"margin_weight must be finite and non-negative, got {margin_weight}"
        )


def compute_l89_temporal_stats(
    csv_path: str,
    *,
    path_columns: Sequence[str],
    n_samples: int,
    seed: int,
    num_workers: int = 1,
    local_file_cache: Optional[StaticAnchoredCache] = None,
    ds_cfg_name: str = "landsat89_7band",
) -> Tuple[Sequence[float], Sequence[float]]:
    """Compute shared 7-band L89 mean/std over sampled rows and all timepoints.

    The temporal model reuses one PanopticonPE for every timestamp, so a shared
    per-band normalization is preferable to separate per-timepoint statistics.
    Zero-valued pixels are treated as no-data and ignored.
    """

    dataset_cls = CachedS2CsvDataset if local_file_cache is not None else S2CsvDataset
    dataset_kwargs = {}
    if local_file_cache is not None:
        dataset_kwargs["local_file_cache"] = local_file_cache

    base_ds = dataset_cls(
        csv_path=csv_path,
        ds_cfg_name=ds_cfg_name,
        path_column=path_columns[0],
        normalize_stats=None,
        scale_to_unit=False,
        compute_stats=False,
        pad_to_multiple=None,
        path_columns_for_validation=path_columns,
        **dataset_kwargs,
    )
    if len(base_ds) == 0:
        raise ValueError(f"Cannot compute stats from empty CSV: {csv_path}")

    sample_count = min(max(1, int(n_samples)), len(base_ds))
    rng = random.Random(seed)
    indices = rng.sample(range(len(base_ds)), sample_count)
    num_channels = int(base_ds.chn_ids.shape[0])
    channel_sums = torch.zeros(num_channels, dtype=torch.float64)
    channel_sq_sums = torch.zeros(num_channels, dtype=torch.float64)
    channel_counts = torch.zeros(num_channels, dtype=torch.float64)
    jobs = [(row_idx, col) for row_idx in indices for col in path_columns]
    num_workers = max(1, int(num_workers))
    progress_every = max(1, 100 * len(path_columns))

    print(
        f"[Stats] Computing L89 normalization from {sample_count} rows x {len(path_columns)} timepoints "
        f"({len(jobs)} TIFFs); zeros ignored; workers={num_workers}.",
        flush=True,
    )

    def _image_stats(row_idx: int, col: str):
        row = base_ds.df.iloc[row_idx]
        sample_id = row[base_ds.id_column] if base_ds.id_column in row else row_idx
        img = base_ds._load_image(row[col], column_name=col, sample_id=sample_id).to(dtype=torch.float64)
        valid = img != 0
        valid_f = valid.to(dtype=torch.float64)
        sums = (img * valid_f).sum(dim=(1, 2))
        sq_sums = (img * img * valid_f).sum(dim=(1, 2))
        counts = valid.sum(dim=(1, 2)).to(dtype=torch.float64)
        return sums, sq_sums, counts

    def _accumulate(result) -> None:
        sums, sq_sums, counts = result
        channel_sums.add_(sums)
        channel_sq_sums.add_(sq_sums)
        channel_counts.add_(counts)

    if num_workers == 1:
        for job_idx, (row_idx, col) in enumerate(jobs, 1):
            _accumulate(_image_stats(row_idx, col))
            if job_idx % progress_every == 0 or job_idx == len(jobs):
                print(f"[Stats] processed TIFFs {job_idx}/{len(jobs)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_image_stats, row_idx, col) for row_idx, col in jobs]
            for job_idx, future in enumerate(as_completed(futures), 1):
                _accumulate(future.result())
                if job_idx % progress_every == 0 or job_idx == len(jobs):
                    print(f"[Stats] processed TIFFs {job_idx}/{len(jobs)}", flush=True)

    missing = channel_counts == 0
    if missing.any():
        missing_channels = torch.nonzero(missing, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"No valid pixels found while computing stats for channels: {missing_channels}")

    means = channel_sums / channel_counts
    variances = channel_sq_sums / channel_counts - means * means
    stds = torch.sqrt(torch.clamp(variances, min=1e-12))
    mean_list = means.float().tolist()
    std_list = stds.float().tolist()
    print(f"[Stats] mean={mean_list}", flush=True)
    print(f"[Stats] std={std_list}", flush=True)
    return mean_list, std_list


class L89TemporalSequenceDataset(Dataset):
    """
    Six-timepoint L89 dataset.

    Returns x_dict with:
      imgs: (T, 7, H, W)
      chn_ids: (T, 7)
      timestamps: (T, 3) containing year_offset, month_index, hour
    """

    def __init__(
        self,
        csv_path: str,
        *,
        path_columns: Sequence[str],
        time_columns: Sequence[str],
        normalize_stats,
        band_indices: Optional[Sequence[int]] = None,
        time_base_year: int = 2000,
        ds_cfg_name: str = "landsat89_7band",
        pad_to_multiple: int = 14,
        skip_invalid_samples: bool = False,
        local_file_cache: Optional[StaticAnchoredCache] = None,
        cache_prefetch_rows: int = 0,
        input_mode: str = "raw",
        residual_slots: Sequence[str] = ("path_prev1", "path_year"),
        residual_clip: float = 5.0,
    ):
        if len(path_columns) != len(time_columns):
            raise ValueError(
                f"path_columns and time_columns must have the same length, got "
                f"{len(path_columns)} and {len(time_columns)}"
            )

        dataset_cls = CachedS2CsvDataset if local_file_cache is not None else S2CsvDataset
        dataset_kwargs = {}
        if local_file_cache is not None:
            dataset_kwargs["local_file_cache"] = local_file_cache

        self.base_ds = dataset_cls(
            csv_path=csv_path,
            ds_cfg_name=ds_cfg_name,
            path_column=path_columns[0],
            normalize_stats=normalize_stats,
            scale_to_unit=False,
            compute_stats=False,
            pad_to_multiple=pad_to_multiple,
            skip_invalid_samples=skip_invalid_samples,
            path_columns_for_validation=path_columns,
            **dataset_kwargs,
        )
        self.path_columns = tuple(path_columns)
        self.time_columns = tuple(time_columns)
        available_bands = int(self.base_ds.chn_ids.shape[0])
        self.band_indices = tuple(range(available_bands)) if band_indices is None else tuple(band_indices)
        if any(index >= available_bands for index in self.band_indices):
            raise ValueError(
                f"Band indices {self.band_indices} exceed the available 0..{available_bands - 1} bands"
            )
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
        base_chn_ids = base_chn_ids[list(self.band_indices)].to(dtype=torch.float32)

        for col in self.path_columns:
            img = self.base_ds._load_image(row[col], column_name=col, sample_id=sample_id)
            img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            img = img[list(self.band_indices)]
            imgs.append(img)
            chn_ids.append(base_chn_ids.clone())

        x_dict = {
            "imgs": torch.stack(imgs, dim=0),  # (T, C, H, W)
            "chn_ids": torch.stack(chn_ids, dim=0),  # (T, C)
            "timestamps": self.timestamps[idx],
        }
        transform_slots = (
            self.residual_slots
            if self.input_mode in {"residual", "current_residual"}
            else ()
        )
        x_dict = transform_temporal_sample(
            x_dict,
            path_columns=self.path_columns,
            input_mode=self.input_mode,
            residual_slots=transform_slots,
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

    Each L89 timepoint is patchified independently by the shared PanopticonPE
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
    input_mode = getattr(args, "input_mode", "raw")
    input_suffix = "" if input_mode == "raw" else f"__{input_mode}"
    evidence_suffix = (
        "__temporal_evidence_margin"
        if getattr(args, "use_temporal_evidence_margin", False)
        else ""
    )
    return (
        f"{train_stem}__{test_stem}__satmae_time__{mode}"
        f"{input_suffix}{evidence_suffix}"
    )


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
    best_test_f1: float,
    best_val_ap: float,
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
            "best_test_f1": best_test_f1,
            "best_val_ap": best_val_ap,
            "args": vars(args),
        },
        path,
    )


def try_resume(path: Path, backbone: nn.Module, head: nn.Module, optimizer, scheduler, scaler, device):
    if not path.is_file():
        return 1, 0, 0.0, 0.0, 0.0, float("-inf")

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
    best_test_f1 = ckpt.get("best_test_f1", 0.0)
    best_val_ap = ckpt.get("best_val_ap", float("-inf"))
    print(f"Resumed from {path} at epoch {start_epoch - 1}", flush=True)
    return (
        start_epoch,
        global_step,
        best_train_acc,
        best_test_acc,
        best_test_f1,
        best_val_ap,
    )


def build_dataset(
    args,
    csv_path: str,
    cache_obj: Optional[StaticAnchoredCache],
    normalize_stats,
) -> L89TemporalSequenceDataset:
    return L89TemporalSequenceDataset(
        csv_path=csv_path,
        path_columns=parse_csv_columns(args.path_columns),
        time_columns=parse_csv_columns(args.time_columns),
        normalize_stats=normalize_stats,
        band_indices=parse_band_indices(args.band_indices),
        time_base_year=args.time_base_year,
        pad_to_multiple=args.pad_to_multiple,
        skip_invalid_samples=args.skip_invalid_samples,
        local_file_cache=cache_obj,
        cache_prefetch_rows=args.cache_prefetch_rows,
        input_mode=args.input_mode,
        residual_slots=parse_residual_slots(args.residual_slots),
        residual_clip=args.residual_clip,
    )


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    path_columns = parse_csv_columns(args.path_columns)
    time_columns = parse_csv_columns(args.time_columns)
    band_indices = parse_band_indices(args.band_indices)
    residual_slots = parse_residual_slots(args.residual_slots)
    use_temporal_evidence_margin = getattr(
        args, "use_temporal_evidence_margin", False
    )
    temporal_evidence_margin = getattr(args, "temporal_evidence_margin", 0.25)
    temporal_evidence_current_ce_weight = getattr(
        args, "temporal_evidence_current_ce_weight", 1.0
    )
    temporal_evidence_margin_weight = getattr(
        args, "temporal_evidence_margin_weight", 1.0
    )
    if len(path_columns) != len(time_columns):
        raise ValueError(
            f"--path_columns and --time_columns must have the same length, got {len(path_columns)} and {len(time_columns)}"
        )
    if args.input_mode in {"residual", "current_residual"}:
        missing_residual_slots = [
            slot for slot in residual_slots if slot not in path_columns
        ]
        if missing_residual_slots:
            raise ValueError(
                f"--residual_slots {missing_residual_slots} are absent from "
                f"--path_columns {path_columns}"
            )
    validate_temporal_evidence_configuration(
        enabled=use_temporal_evidence_margin,
        input_mode=args.input_mode,
        source_timepoints=len(path_columns),
        residual_slots=residual_slots,
        margin=temporal_evidence_margin,
        current_ce_weight=temporal_evidence_current_ce_weight,
        margin_weight=temporal_evidence_margin_weight,
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
        norm_stats = PRECOMPUTED_STATS
        print(f"[Stats] Using precomputed L89 stats: mean={norm_stats[0]} std={norm_stats[1]}", flush=True)
    else:
        stats_cache_obj = cache_obj
        if cache_obj is not None and getattr(cache_obj, "async_mode", False):
            stats_cache_obj = None
        norm_stats = compute_l89_temporal_stats(
            args.train_csv,
            path_columns=path_columns,
            n_samples=args.stats_samples,
            seed=args.stats_seed,
            num_workers=args.stats_workers,
            local_file_cache=stats_cache_obj,
        )

    train_ds = build_dataset(args, args.train_csv, cache_obj, norm_stats)
    test_ds = build_dataset(args, args.test_csv, cache_obj, norm_stats)

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

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
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
        f"band_indices={band_indices}, "
        f"train_backbone={args.train_backbone}, cache={cache_desc}",
        flush=True,
    )
    if use_temporal_evidence_margin:
        print(
            "[TemporalEvidence] enabled: "
            f"margin={temporal_evidence_margin}, "
            f"current_ce_weight={temporal_evidence_current_ce_weight}, "
            f"margin_weight={temporal_evidence_margin_weight}",
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
    best_test_f1_path = ckpt_dir / "ckpt_best_test_f1.pth"
    best_val_ap_path = ckpt_dir / "ckpt_best_val_ap.pth"
    metrics_path = ckpt_dir / "metrics_history.json"

    start_epoch = 1
    global_step = 0
    best_train_acc = 0.0
    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_val_ap = float("-inf")
    metrics_history = []
    if args.resume and metrics_path.is_file():
        try:
            loaded_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(loaded_metrics, list):
                metrics_history = loaded_metrics
                best_val_ap = max(
                    (
                        float(record["test_ap"])
                        for record in metrics_history
                        if record.get("test_ap") is not None
                        and math.isfinite(float(record["test_ap"]))
                    ),
                    default=float("-inf"),
                )
        except Exception as exc:
            print(f"[Resume] Could not read {metrics_path}: {exc}", flush=True)
            metrics_history = []
    if args.resume:
        (
            start_epoch,
            global_step,
            best_train_acc,
            best_test_acc,
            best_test_f1,
            checkpoint_best_val_ap,
        ) = try_resume(latest_path, backbone, head, optimizer, scheduler, scaler, device)
        best_val_ap = max(best_val_ap, checkpoint_best_val_ap)

    wandb_run = init_wandb(args)

    for epoch in range(start_epoch, args.epochs + 1):
        train_panopticon = args.train_backbone and not (
            args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs
        )
        backbone.train()
        set_backbone_train_state(backbone, train_panopticon)
        head.train()

        total_loss = 0.0
        train_full_ce_total = 0.0
        train_current_ce_total = 0.0
        train_temporal_margin_total = 0.0
        train_margin_satisfaction_total = 0.0
        correct = 0
        total = 0
        train_tp = train_fp = train_fn = train_tn = 0
        for step, (x_dict, labels) in enumerate(train_loader, 1):
            labels = labels.to(device)
            x_dict = recursive_to_device(x_dict, device)
            current_x_dict = None
            if use_temporal_evidence_margin:
                current_x_dict = slice_current_temporal_view(
                    x_dict,
                    input_mode=args.input_mode,
                    path_columns=path_columns,
                )

            with autocast(enabled=use_amp):
                feats = backbone(x_dict, is_training=True)
                cls_token = feats["x_norm_clstoken"]
                logits = head(cls_token)
                temporal_losses = None
                if use_temporal_evidence_margin:
                    current_feats = backbone(current_x_dict, is_training=True)
                    current_logits = head(current_feats["x_norm_clstoken"])
                    temporal_losses = temporal_evidence_classification_objective(
                        logits,
                        labels,
                        criterion,
                        enabled=True,
                        current_logits=current_logits,
                        margin=temporal_evidence_margin,
                        current_ce_weight=temporal_evidence_current_ce_weight,
                        margin_weight=temporal_evidence_margin_weight,
                    )
                    loss = temporal_losses.total
                else:
                    # Keep the legacy flag-off path exactly as a single CE.
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
            if temporal_losses is not None:
                train_full_ce_total += (
                    temporal_losses.full_ce.item() * labels.size(0)
                )
                train_current_ce_total += (
                    temporal_losses.current_ce.item() * labels.size(0)
                )
                train_temporal_margin_total += (
                    temporal_losses.margin.item() * labels.size(0)
                )
                train_margin_satisfaction_total += (
                    temporal_losses.margin_satisfaction.item()
                    * labels.size(0)
                )
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
                running_message = (
                    f"Epoch {epoch} step {step}/{len(train_loader)} "
                    f"train_loss={running_loss:.4f} "
                    f"train_acc={running_acc:.4f} "
                    f"train_f1={running_f1:.4f}"
                )
                if use_temporal_evidence_margin:
                    running_message += (
                        f" full_ce={train_full_ce_total / total:.4f}"
                        f" current_ce={train_current_ce_total / total:.4f}"
                        f" temporal_margin="
                        f"{train_temporal_margin_total / total:.4f}"
                        f" margin_satisfaction="
                        f"{train_margin_satisfaction_total / total:.4f}"
                    )
                print(
                    running_message,
                    flush=True,
                )
                if wandb_run is not None:
                    running_metrics = {
                        "train_loss_running": running_loss,
                        "train_acc_running": running_acc,
                        "train_f1_running": running_f1,
                        "step": global_step,
                        "epoch": epoch,
                    }
                    if use_temporal_evidence_margin:
                        running_metrics.update(
                            {
                                "train_full_ce_loss_running": (
                                    train_full_ce_total / total
                                ),
                                "train_current_ce_loss_running": (
                                    train_current_ce_total / total
                                ),
                                "train_temporal_margin_loss_running": (
                                    train_temporal_margin_total / total
                                ),
                                "train_temporal_margin_satisfaction_running": (
                                    train_margin_satisfaction_total / total
                                ),
                            }
                        )
                    wandb_run.log(running_metrics)

            if args.max_train_steps is not None and step >= args.max_train_steps:
                break

        train_loss = total_loss / total
        train_acc = correct / total
        train_f1_den = 2 * train_tp + train_fp + train_fn
        train_f1 = (2 * train_tp / train_f1_den) if train_f1_den > 0 else 0.0
        if use_temporal_evidence_margin:
            train_full_ce_loss = train_full_ce_total / total
            train_current_ce_loss = train_current_ce_total / total
            train_temporal_margin_loss = train_temporal_margin_total / total
            train_temporal_margin_satisfaction = (
                train_margin_satisfaction_total / total
            )

        head.eval()
        backbone.eval()
        correct = 0
        total = 0
        test_loss_total = 0.0
        test_full_ce_total = 0.0
        test_current_ce_total = 0.0
        test_temporal_margin_total = 0.0
        test_margin_satisfaction_total = 0.0
        tp = fp = fn = tn = 0
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for step, (x_dict, labels) in enumerate(test_loader, 1):
                labels = labels.to(device)
                x_dict = recursive_to_device(x_dict, device)
                current_x_dict = None
                if use_temporal_evidence_margin:
                    current_x_dict = slice_current_temporal_view(
                        x_dict,
                        input_mode=args.input_mode,
                        path_columns=path_columns,
                    )
                with autocast(enabled=use_amp):
                    feats = backbone(x_dict, is_training=True)
                    cls_token = feats["x_norm_clstoken"]
                    logits = head(cls_token)
                    temporal_losses = None
                    if use_temporal_evidence_margin:
                        current_feats = backbone(
                            current_x_dict, is_training=True
                        )
                        current_logits = head(
                            current_feats["x_norm_clstoken"]
                        )
                        temporal_losses = (
                            temporal_evidence_classification_objective(
                                logits,
                                labels,
                                criterion,
                                enabled=True,
                                current_logits=current_logits,
                                margin=temporal_evidence_margin,
                                current_ce_weight=(
                                    temporal_evidence_current_ce_weight
                                ),
                                margin_weight=temporal_evidence_margin_weight,
                            )
                        )
                        loss = temporal_losses.total
                    else:
                        loss = criterion(logits, labels)

                test_loss_total += loss.item() * labels.size(0)
                if temporal_losses is not None:
                    test_full_ce_total += (
                        temporal_losses.full_ce.item() * labels.size(0)
                    )
                    test_current_ce_total += (
                        temporal_losses.current_ce.item() * labels.size(0)
                    )
                    test_temporal_margin_total += (
                        temporal_losses.margin.item() * labels.size(0)
                    )
                    test_margin_satisfaction_total += (
                        temporal_losses.margin_satisfaction.item()
                        * labels.size(0)
                    )
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
        if use_temporal_evidence_margin:
            test_full_ce_loss = (
                test_full_ce_total / total if total > 0 else float("nan")
            )
            test_current_ce_loss = (
                test_current_ce_total / total
                if total > 0
                else float("nan")
            )
            test_temporal_margin_loss = (
                test_temporal_margin_total / total
                if total > 0
                else float("nan")
            )
            test_temporal_margin_satisfaction = (
                test_margin_satisfaction_total / total
                if total > 0
                else float("nan")
            )
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        test_f1_den = 2 * tp + fp + fn
        test_f1 = (2 * tp / test_f1_den) if test_f1_den > 0 else 0.0
        all_probs = torch.cat(all_probs) if len(all_probs) > 0 else torch.tensor([])
        all_targets = torch.cat(all_targets) if len(all_targets) > 0 else torch.tensor([])
        test_ap = float("nan")
        test_macro_f1 = float("nan")
        test_auroc = float("nan")
        if all_probs.numel() > 0:
            try:
                from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

                targets_np = all_targets.numpy()
                probs_np = all_probs.numpy()
                try:
                    test_auroc = float(roc_auc_score(targets_np, probs_np))
                except ValueError:
                    test_auroc = float("nan")
                test_ap = float(average_precision_score(targets_np, probs_np))
                test_macro_f1 = float(
                    f1_score(
                        targets_np,
                        (probs_np >= 0.5).astype("int64"),
                        average="macro",
                        zero_division=0,
                    )
                )
            except Exception:
                test_ap = float("nan")
                test_macro_f1 = float("nan")
                test_auroc = float("nan")

        epoch_message = (
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f} "
            f"macro_f1={test_macro_f1:.4f} recall={recall:.4f} fpr={fpr:.4f} "
            f"auroc={test_auroc:.4f} ap={test_ap:.4f}"
        )
        if use_temporal_evidence_margin:
            epoch_message += (
                f" train_full_ce={train_full_ce_loss:.4f}"
                f" train_current_ce={train_current_ce_loss:.4f}"
                f" train_temporal_margin={train_temporal_margin_loss:.4f}"
                f" train_margin_satisfaction="
                f"{train_temporal_margin_satisfaction:.4f}"
                f" test_full_ce={test_full_ce_loss:.4f}"
                f" test_current_ce={test_current_ce_loss:.4f}"
                f" test_temporal_margin={test_temporal_margin_loss:.4f}"
                f" test_margin_satisfaction="
                f"{test_temporal_margin_satisfaction:.4f}"
            )
        print(epoch_message, flush=True)
        epoch_metrics = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "input_mode": args.input_mode,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_f1": float(train_f1),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_f1": float(test_f1),
            "test_macro_f1": float(test_macro_f1),
            "test_recall": float(recall),
            "test_fpr": float(fpr),
            "test_auroc": float(test_auroc),
            "test_ap": float(test_ap),
        }
        if use_temporal_evidence_margin:
            epoch_metrics.update(
                {
                    "train_full_ce_loss": float(train_full_ce_loss),
                    "train_current_ce_loss": float(train_current_ce_loss),
                    "train_temporal_margin_loss": float(
                        train_temporal_margin_loss
                    ),
                    "train_temporal_margin_satisfaction": float(
                        train_temporal_margin_satisfaction
                    ),
                    "test_full_ce_loss": float(test_full_ce_loss),
                    "test_current_ce_loss": float(test_current_ce_loss),
                    "test_temporal_margin_loss": float(
                        test_temporal_margin_loss
                    ),
                    "test_temporal_margin_satisfaction": float(
                        test_temporal_margin_satisfaction
                    ),
                }
            )
        metrics_history.append(epoch_metrics)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        temporary_metrics_path = metrics_path.with_suffix(
            metrics_path.suffix + f".part.{os.getpid()}"
        )
        temporary_metrics_path.write_text(
            json.dumps(metrics_history, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metrics_path, metrics_path)

        prev_best_train = best_train_acc
        prev_best_test = best_test_acc
        prev_best_test_f1 = best_test_f1
        prev_best_val_ap = best_val_ap
        best_train_acc = max(best_train_acc, train_acc)
        best_test_acc = max(best_test_acc, test_acc)
        best_test_f1 = max(best_test_f1, test_f1)
        if math.isfinite(test_ap):
            best_val_ap = max(best_val_ap, test_ap)

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
            best_test_f1,
            best_val_ap,
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
                best_test_f1,
                best_val_ap,
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
                best_test_f1,
                best_val_ap,
                args,
            )

        if test_f1 > prev_best_test_f1:
            save_checkpoint(
                best_test_f1_path,
                epoch,
                global_step,
                backbone,
                head,
                optimizer,
                scheduler,
                scaler,
                best_train_acc,
                best_test_acc,
                best_test_f1,
                best_val_ap,
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
                best_test_f1,
                best_val_ap,
                args,
            )

        if wandb_run is not None:
            wandb_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_macro_f1": test_macro_f1,
                "test_recall": recall,
                "test_fpr": fpr,
                "test_auroc": test_auroc,
                "test_ap": test_ap,
            }
            if use_temporal_evidence_margin:
                wandb_metrics.update(
                    {
                        "train_full_ce_loss": train_full_ce_loss,
                        "train_current_ce_loss": train_current_ce_loss,
                        "train_temporal_margin_loss": (
                            train_temporal_margin_loss
                        ),
                        "train_temporal_margin_satisfaction": (
                            train_temporal_margin_satisfaction
                        ),
                        "test_full_ce_loss": test_full_ce_loss,
                        "test_current_ce_loss": test_current_ce_loss,
                        "test_temporal_margin_loss": (
                            test_temporal_margin_loss
                        ),
                        "test_temporal_margin_satisfaction": (
                            test_temporal_margin_satisfaction
                        ),
                    }
                )
            wandb_run.log(wandb_metrics)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Panopticon ViT-B/14 + SatMAE-style temporal embeddings for six-timepoint L89 classification."
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
    parser.add_argument(
        "--stats_samples",
        type=int,
        default=200,
        help="Rows sampled from train CSV to compute shared 7-band L89 normalization stats across all timepoints.",
    )
    parser.add_argument(
        "--stats_workers",
        type=int,
        default=8,
        help="Parallel TIFF readers for normalization-stat computation.",
    )
    parser.add_argument("--stats_seed", type=int, default=0, help="Random seed for normalization-stat row sampling.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for model initialization and train/data-loader shuffling.",
    )
    parser.add_argument(
        "--use_precomputed_stats",
        action="store_true",
        help="Skip runtime stat computation and use the PRECOMPUTED_STATS constants.",
    )
    parser.add_argument("--path_columns", default=DEFAULT_PATH_COLUMNS)
    parser.add_argument("--time_columns", default=DEFAULT_TIME_COLUMNS)
    parser.add_argument(
        "--input_mode",
        choices=INPUT_MODES,
        default="raw",
        help=(
            "raw keeps all source frames; current keeps t0; residual uses normalized "
            "t0-history differences; current_residual concatenates t0 and differences."
        ),
    )
    parser.add_argument(
        "--residual_slots",
        default="path_prev1,path_year",
        help="Comma-separated history path-column names used to form t0-history residuals.",
    )
    parser.add_argument(
        "--residual_clip",
        type=float,
        default=5.0,
        help="Clip normalized temporal residuals to +/- this value; <=0 disables clipping.",
    )
    parser.add_argument(
        "--use_temporal_evidence_margin",
        action="store_true",
        help=(
            "Opt in to full/current matched CE plus a label-conditioned "
            "full-vs-current binary-log-odds margin. Supported only for raw and "
            "current_residual inputs."
        ),
    )
    parser.add_argument(
        "--temporal_evidence_margin",
        type=float,
        default=0.25,
        help="Required signed binary-log-odds improvement from full temporal evidence.",
    )
    parser.add_argument(
        "--temporal_evidence_current_ce_weight",
        type=float,
        default=1.0,
        help="Weight on matched current-only cross-entropy in the opt-in objective.",
    )
    parser.add_argument(
        "--temporal_evidence_margin_weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the label-conditioned temporal softplus margin loss; "
            "0 provides the matched full-CE + current-CE compute control."
        ),
    )
    parser.add_argument(
        "--band_indices",
        default="0,1,2,3,4,5,6",
        help="Zero-based Landsat band indices to use: B1..B7 map to 0..6.",
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
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument(
        "--wandb_run_name",
        default="dino_classifier_head_l89_temporal_satmae",
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
        default="/diniuvol/yuyao/l89_temporal_cache",
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
