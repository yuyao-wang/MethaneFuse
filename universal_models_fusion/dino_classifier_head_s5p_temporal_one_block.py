import argparse
import csv
import hashlib
import math
import os
import shutil
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

# Per-timeframe normalization stats for S5P CH4 stacked (t0, t-90, t-360).
PRECOMPUTED_STATS = (
    [1908.944798144908, 362.20128748570943, 666.5645739544017],
    [52.2615669211965, 743.7553740768346, 901.8399543163595],
)


def parse_comma_separated_floats(val: Optional[str]) -> Optional[Sequence[float]]:
    if val is None:
        return None
    parts = [p.strip() for p in val.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Could not parse floats from '{val}'") from exc


def parse_subset_value(val: Optional[str]) -> Optional[Union[int, float]]:
    if val is None:
        return None
    try:
        text = str(val)
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("compute_stats_subset must be an int or float") from exc


def load_normalization_stats(args) -> Optional[Tuple[Sequence[float], Sequence[float]]]:
    if args.normalize_stats_npz is not None:
        with np.load(args.normalize_stats_npz) as npz:
            if "mean" not in npz or "std" not in npz:
                raise KeyError(f"{args.normalize_stats_npz} must contain 'mean' and 'std'")
            mean = npz["mean"].tolist()
            std = npz["std"].tolist()
            return mean, std

    mean_override = parse_comma_separated_floats(args.normalize_mean)
    std_override = parse_comma_separated_floats(args.normalize_std)
    if mean_override is not None and std_override is not None:
        if len(mean_override) != len(std_override):
            raise ValueError("normalize_mean and normalize_std must have the same length")
        return mean_override, std_override
    if (mean_override is None) != (std_override is None):
        raise ValueError("normalize_mean and normalize_std must be provided together")

    return PRECOMPUTED_STATS


class StaticAnchoredCache:
    def __init__(self, cache_dir: str, min_free_gb: float = 30.0):
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
        unique_paths = sorted({os.path.abspath(p) for p in paths if isinstance(p, str)})
        if not unique_paths:
            return
        print(f"[Cache] Warming up (target: {len(unique_paths)})...", flush=True)

        def _copy_one(path: str):
            res = self.ensure_local(path)
            return res == path

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_copy_one, p) for p in unique_paths]
            fallback_count = sum(1 for fut in as_completed(futures) if fut.result())
            print(f"[Cache] Warmup complete. Cached: {len(unique_paths) - fallback_count}, Remote: {fallback_count}")


class CachedS5pNpzDataset(Dataset):
    """
    S5P dataset wrapper that routes file paths through StaticAnchoredCache.
    Defined at module scope so it remains picklable for DataLoader workers.
    """

    def __init__(self, base_ds, local_file_cache: Optional[StaticAnchoredCache] = None):
        self.base_ds = base_ds
        self._local_file_cache = local_file_cache

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        row = self.base_ds.df.iloc[idx]
        label = int(row[self.base_ds.label_column])
        img_path = row[self.base_ds.path_column]

        if self._local_file_cache is not None and isinstance(img_path, str):
            img_path = self._local_file_cache.ensure_local(img_path)

        img, chn_ids = self.base_ds._load_image(img_path)
        x_dict = dict(imgs=img, chn_ids=chn_ids)
        if self.base_ds.transform is not None:
            x_dict = self.base_ds.transform(x_dict)

        return x_dict, label


class CLSHead(nn.Module):
    """Minimal DINO-style classifier head: CLS -> LayerNorm -> Linear."""

    def __init__(self, embed_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(cls))


class S5pSimpleNpzDataset(Dataset):
    """Wraps S5pNpzDataset to return a single x_dict for the model."""

    def __init__(self, base_ds, resize_to: Optional[int] = None):
        self.base_ds = base_ds
        self.resize_to = resize_to

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x_dict, label = self.base_ds[idx]
        # The base dataset already returns the x_dict we need.
        if self.resize_to is not None:
            x_dict = resize_imgs(x_dict, self.resize_to)
        return x_dict, label


class S5pTemporalTiffDataset(Dataset):
    """Temporal S5P NPZ dataset that concatenates three timepoints into 3 channels."""

    def __init__(
        self,
        csv_path: str,
        *,
        path_columns: Sequence[str],
        label_column: str = "label",
        normalize_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        compute_stats: bool = False,
        compute_stats_subset: Optional[Union[int, float]] = None,
        pad_to_multiple: Optional[int] = None,
        pad_value: float = 0.0,
        chn_ids: Optional[Sequence[float]] = None,
        default_chn_id_value: float = 0.0,
        local_file_cache: Optional[StaticAnchoredCache] = None,
        nan_to_num: Optional[float] = 0.0,
        max_retries: int = 5,
        data_key: Optional[str] = "ch4",
        chn_ids_key: Optional[str] = "chn_ids",
        channel_last: bool = False,
        allow_pickle: bool = False,
        scale_to_unit: bool = False,
        scale_value: float = 65535.0,
    ):
        if len(path_columns) not in (1, 3):
            raise ValueError(
                f"Expected either 1 (single-frame wide-table) or 3 temporal columns, got {len(path_columns)}: {path_columns}"
            )

        self.df = pd.read_csv(csv_path)
        self.path_columns = tuple(path_columns)
        self.label_column = label_column
        self.pad_to_multiple = pad_to_multiple
        self.pad_value = float(pad_value)
        self.default_chn_id_value = float(default_chn_id_value)
        self._local_file_cache = local_file_cache
        self.nan_to_num = nan_to_num
        self.max_retries = max(1, int(max_retries))
        self._stats_source: Optional[str] = None
        self.data_key = data_key
        self.chn_ids_key = chn_ids_key
        self.channel_last = bool(channel_last)
        self.allow_pickle = bool(allow_pickle)
        self.scale_to_unit = bool(scale_to_unit)
        self.scale_value = float(scale_value)
        if self.scale_to_unit and self.scale_value <= 0:
            raise ValueError(f"scale_value must be > 0 when scale_to_unit is enabled, got {self.scale_value}")

        if chn_ids is not None:
            ids = torch.as_tensor(chn_ids, dtype=torch.float32).flatten()
            if ids.numel() == 1:
                ids = ids.repeat(3)
            if ids.numel() != 3:
                raise ValueError(f"chn_ids must have length 1 or 3 for temporal S5P, got {ids.numel()}")
            self._chn_ids = ids
        else:
            self._chn_ids = torch.full((3,), fill_value=self.default_chn_id_value, dtype=torch.float32)

        if normalize_stats is not None:
            mean, std = normalize_stats
            if len(mean) != 3 or len(std) != 3:
                raise ValueError("S5P temporal normalization stats must each have length 3 (t0/t90/t360).")
            self._mean = torch.as_tensor(mean, dtype=torch.float32).view(3, 1, 1)
            self._std = torch.clamp(torch.as_tensor(std, dtype=torch.float32), min=1e-6).view(3, 1, 1)
            self._stats_source = "provided"
        else:
            self._mean = None
            self._std = None
            if compute_stats:
                mean_t, std_t = self._compute_dataset_stats(subset=compute_stats_subset)
                self._mean = mean_t.view(3, 1, 1)
                self._std = torch.clamp(std_t, min=1e-6).view(3, 1, 1)
                self._stats_source = "computed"

    def _read_temporal_sample(self, row) -> Tuple[list[torch.Tensor], torch.Tensor]:
        if len(self.path_columns) == 1:
            path = row[self.path_columns[0]]
            arr, sample_chn_ids = self._load_npz_array(path)
            arr_chw = self._to_chw(arr, path)
            c = int(arr_chw.shape[0])
            if c >= 3:
                arr_chw = arr_chw[:3]
            else:
                repeat = math.ceil(3 / max(1, c))
                arr_chw = np.repeat(arr_chw, repeats=repeat, axis=0)[:3]
            frames = [torch.from_numpy(arr_chw[i]).to(dtype=torch.float32) for i in range(3)]
            chn_ids = self._normalize_chn_ids(sample_chn_ids, expected_channels=3)
            return frames, chn_ids

        frames = []
        for col in self.path_columns:
            arr, _ = self._load_npz_array(row[col])
            arr_chw = self._to_chw(arr, row[col])
            if arr_chw.shape[0] != 1:
                raise ValueError(
                    f"Temporal-column S5P mode expects per-file single-channel arrays, got shape {arr.shape} at {row[col]}"
                )
            frames.append(torch.from_numpy(arr_chw[0]).to(dtype=torch.float32))
        return frames, self._chn_ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        attempts = 0
        while attempts < self.max_retries:
            try:
                row = self.df.iloc[idx]
                label = int(row[self.label_column])
                frames, sample_chn_ids = self._read_temporal_sample(row)
                img = torch.stack(frames, dim=0)  # (3,H,W)
                if self.nan_to_num is not None:
                    img = torch.nan_to_num(img, nan=self.nan_to_num, posinf=self.nan_to_num, neginf=self.nan_to_num)
                if self._mean is not None and self._std is not None:
                    img = (img - self._mean) / self._std
                if self.pad_to_multiple is not None:
                    img = self._pad_to_multiple(img, int(self.pad_to_multiple))
                x_dict = {"imgs": img, "chn_ids": sample_chn_ids}
                return x_dict, label
            except Exception as exc:
                attempts += 1
                if attempts >= self.max_retries:
                    raise RuntimeError(f"Failed to load temporal NPZ sample at index {idx}") from exc
                idx = np.random.randint(0, len(self.df))

    def _resolve_path(self, path: str) -> str:
        if self._local_file_cache is not None and isinstance(path, str):
            return self._local_file_cache.ensure_local(path)
        return path

    def _load_npz_array(self, path: str) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Encountered empty NPZ path in CSV.")
        path = self._resolve_path(path.strip())
        if not path.lower().endswith(".npz"):
            raise ValueError(f"S5P dataset now expects .npz inputs, got path: {path}")

        try:
            np_obj = np.load(path, allow_pickle=self.allow_pickle)
        except ValueError as exc:
            if ("allow_pickle=False" in str(exc) or "pickled data" in str(exc)) and not self.allow_pickle:
                raise ValueError(
                    f"NPZ at {path} requires pickle support. Re-run with --allow_pickle if the source is trusted."
                ) from exc
            raise

        chn_ids = None
        try:
            if isinstance(np_obj, np.lib.npyio.NpzFile):
                arr = self._extract_npz_array(np_obj, path)
                if self.chn_ids_key is not None and self.chn_ids_key in np_obj:
                    chn_ids = torch.as_tensor(np_obj[self.chn_ids_key], dtype=torch.float32)
            else:
                if isinstance(np_obj, np.ndarray) and np_obj.dtype == object and np_obj.size == 1:
                    obj = np_obj.item()
                    if isinstance(obj, dict):
                        if self.data_key is not None and self.data_key in obj:
                            arr = np.asarray(obj[self.data_key])
                        else:
                            first_array = next((v for v in obj.values() if isinstance(v, np.ndarray)), None)
                            if first_array is None:
                                raise ValueError(f"Object array at {path} has no ndarray payload.")
                            arr = np.asarray(first_array)
                        if self.chn_ids_key is not None and self.chn_ids_key in obj:
                            chn_ids = torch.as_tensor(obj[self.chn_ids_key], dtype=torch.float32)
                    else:
                        arr = np.asarray(obj)
                else:
                    arr = np.asarray(np_obj)
        finally:
            if isinstance(np_obj, np.lib.npyio.NpzFile):
                np_obj.close()

        if self.scale_to_unit:
            arr = arr.astype(np.float32, copy=False) / self.scale_value
        return np.asarray(arr), chn_ids

    def _extract_npz_array(self, np_obj: np.lib.npyio.NpzFile, path: str) -> np.ndarray:
        if self.data_key is not None:
            if self.data_key not in np_obj:
                raise KeyError(f"Key '{self.data_key}' not found in NPZ file {path}")
            return np.array(np_obj[self.data_key])
        preferred_keys = ("ch4", "image", "imgs", "arr_0", "data")
        for key in preferred_keys:
            if key in np_obj:
                arr = np.array(np_obj[key])
                if self._looks_like_image(arr):
                    return arr
        for key in np_obj.files:
            if key == self.chn_ids_key or key.lower() in {"meta", "metadata"}:
                continue
            arr = np.array(np_obj[key])
            if self._looks_like_image(arr):
                return arr
        raise ValueError(
            f"Failed to infer S5P image array from NPZ file {path}. "
            f"Available keys: {list(np_obj.files)}. Consider setting --data_key."
        )

    @staticmethod
    def _looks_like_image(arr: np.ndarray) -> bool:
        if not isinstance(arr, np.ndarray):
            return False
        if arr.ndim not in (2, 3):
            return False
        if arr.dtype.kind not in {"f", "i", "u", "b"}:
            return False
        return True

    def _to_chw(self, arr: np.ndarray, path: str) -> np.ndarray:
        if arr.ndim == 2:
            return np.expand_dims(arr, 0)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported S5P NPZ array shape {arr.shape} at {path}")
        if self.channel_last:
            return np.transpose(arr, (2, 0, 1))
        c_first, c_last = arr.shape[0], arr.shape[-1]
        if c_first <= 16:
            return arr
        if c_last <= 16:
            return np.transpose(arr, (2, 0, 1))
        return arr

    def _normalize_chn_ids(self, sample_chn_ids: Optional[torch.Tensor], expected_channels: int) -> torch.Tensor:
        if sample_chn_ids is None:
            return self._chn_ids
        ids = torch.as_tensor(sample_chn_ids, dtype=torch.float32).flatten()
        if ids.numel() == 0:
            return self._chn_ids
        if ids.numel() == 1:
            ids = ids.repeat(expected_channels)
        elif ids.numel() < expected_channels:
            repeat = math.ceil(expected_channels / ids.numel())
            ids = ids.repeat(repeat)[:expected_channels]
        elif ids.numel() > expected_channels:
            ids = ids[:expected_channels]
        return ids

    def _compute_dataset_stats(self, subset: Optional[Union[int, float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs = list(range(len(self.df)))
        if subset is not None:
            if isinstance(subset, float):
                subset = int(len(idxs) * subset)
            subset = max(1, min(len(idxs), int(subset)))
            idxs = idxs[:subset]
        sum_c = torch.zeros(3, dtype=torch.double)
        sumsq_c = torch.zeros(3, dtype=torch.double)
        count = 0
        for i in idxs:
            row = self.df.iloc[i]
            frames, _ = self._read_temporal_sample(row)
            frames = [frame.double() for frame in frames]
            img = torch.stack(frames, dim=0)
            count += int(img.shape[1] * img.shape[2])
            sum_c += img.sum(dim=(1, 2))
            sumsq_c += (img * img).sum(dim=(1, 2))
        mean = (sum_c / count).float()
        std = torch.sqrt((sumsq_c / count) - (mean.double() ** 2)).float()
        return mean, std

    def _pad_to_multiple(self, img: torch.Tensor, multiple: int) -> torch.Tensor:
        _, h, w = img.shape
        target_h = int(np.ceil(h / multiple) * multiple)
        target_w = int(np.ceil(w / multiple) * multiple)
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h == 0 and pad_w == 0:
            return img
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=self.pad_value)

    def get_normalize_stats(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._mean is None or self._std is None:
            return None
        return self._mean.squeeze().cpu(), self._std.squeeze().cpu()

    def get_stats_source(self) -> Optional[str]:
        return self._stats_source


def resolve_s5p_path_columns(csv_path: str, requested_cols: Tuple[str, str, str]) -> Tuple[str, ...]:
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
    except StopIteration as exc:
        raise ValueError(f"CSV file has no header row: {csv_path}") from exc
    header_set = set(header)

    if all(col in header_set for col in requested_cols):
        return requested_cols

    wide_cols = ("s5p_0_path", "s5p_90_path", "s5p_360_path")
    if all(col in header_set for col in wide_cols):
        print(
            f"[Data] {csv_path}: requested columns {requested_cols} not found; using wide-table S5P columns {wide_cols}.",
            flush=True,
        )
        return wide_cols

    single_wide_col = ("s5p_0_path",)
    if single_wide_col[0] in header_set:
        print(
            f"[Data] {csv_path}: found only {single_wide_col[0]} in wide-table input; using single-frame mode (internally repeated to 3 channels).",
            flush=True,
        )
        return single_wide_col

    legacy_cols = ("image_path", "s5p_pre_path", "s5p_pre_pre_path")
    if all(col in header_set for col in legacy_cols):
        print(
            f"[Data] {csv_path}: requested columns {requested_cols} not found; using legacy S5P columns {legacy_cols}.",
            flush=True,
        )
        return legacy_cols

    raise ValueError(
        f"Could not resolve temporal S5P path columns for {csv_path}. "
        f"Requested={requested_cols}. Available columns={header}"
    )


def _valid_path_series(path_series):
    text = path_series.astype("string").str.strip().str.lower()
    return (~path_series.isna()) & (text != "") & (text != "nan") & (text != "none") & (text != "null")


def filter_dataset_to_s5p_only(base_ds: S5pTemporalTiffDataset, *, split_name: str, sensor_column: str = "sensor") -> None:
    before = len(base_ds.df)
    df = base_ds.df
    if sensor_column in df.columns:
        sensor_vals = df[sensor_column].astype("string").str.strip().str.lower()
        df = df[sensor_vals == "s5p"]

    path_cols = tuple(dict.fromkeys(base_ds.path_columns))
    valid_masks = {}
    for col in path_cols:
        if col not in df.columns:
            raise ValueError(f"{split_name}: missing required path column '{col}' after S5P filtering.")
        valid_masks[col] = _valid_path_series(df[col])

    filtered_df = df
    for col in path_cols:
        filtered_df = filtered_df[valid_masks[col]]

    # Wide-table fallback: if 3-column temporal intersection is empty but s5p_0_path is valid,
    # switch to single-frame mode and internally repeat to 3 channels.
    if len(filtered_df) == 0 and len(path_cols) >= 2:
        single_col = "s5p_0_path" if "s5p_0_path" in path_cols else path_cols[0]
        single_mask = valid_masks.get(single_col)
        single_valid = int(single_mask.sum()) if single_mask is not None else 0
        if single_valid > 0:
            print(
                f"[Data] {split_name}: no rows with all temporal columns valid for {path_cols}; "
                f"falling back to single-column S5P mode using '{single_col}' ({single_valid} rows).",
                flush=True,
            )
            filtered_df = df[single_mask]
            base_ds.path_columns = (single_col,)

    base_ds.df = filtered_df.reset_index(drop=True)
    after = len(base_ds.df)
    print(
        f"[Data] {split_name}: kept {after}/{before} rows for S5P using path columns {base_ds.path_columns}.",
        flush=True,
    )
    if after == 0:
        raise ValueError(f"{split_name}: no valid S5P rows available after filtering.")


def load_backbone(weights_path: str, device: torch.device, debug: bool = False):
    """Build the Panopticon ViT backbone and optionally load pretrained weights."""

    from hubconf import _panopticon_vitb14

    if weights_path in (None, "", "none", "scratch", "random"):
        if debug:
            print("Building backbone from scratch (no pretrained weights)", flush=True)
        return _panopticon_vitb14()

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        alt_path = Path(str(weights_path) + "?download=true")
        if alt_path.is_file():
            weights_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    print(f"Loading checkpoint from {weights_path}", flush=True)
    if debug:
        print("Building backbone...", flush=True)
    model = _panopticon_vitb14()
    if debug:
        print("Loading state dict to CPU...", flush=True)
    state = torch.load(weights_path, map_location="cpu")
    if debug:
        print("Applying state dict...", flush=True)
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


def resize_imgs(x_dict, size: int):
    """Resize x_dict["imgs"] to (size, size) spatial resolution via bilinear upsampling."""

    imgs = x_dict.get("imgs")
    if imgs is None:
        return x_dict

    squeeze_back = False
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
        squeeze_back = True

    if imgs.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor for imgs, got shape {tuple(imgs.shape)}")

    imgs = F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)
    if squeeze_back:
        imgs = imgs.squeeze(0)
    x_dict["imgs"] = imgs
    return x_dict


def set_trainable(module: nn.Module, requires_grad: bool):
    if isinstance(module, nn.DataParallel):
        module = module.module
    for p in module.parameters():
        p.requires_grad = requires_grad


def maybe_wrap_dataparallel(module: nn.Module, use_dp: bool, device: torch.device, name: str) -> nn.Module:
    if not use_dp:
        return module
    if device.type != "cuda":
        print(f"[DataParallel:{name}] Requested but device={device} is not CUDA; running single-device.", flush=True)
        return module
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        print(
            f"[DataParallel:{name}] Requested but only {gpu_count} CUDA device detected; running single-device.",
            flush=True,
        )
        return module
    print(f"Enabling DataParallel for {name} across {gpu_count} GPUs", flush=True)
    return nn.DataParallel(module)


def init_wandb(args):
    if not args.use_wandb:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    return run


def build_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "noam":
        warmup_steps = max(args.warmup_steps, 1)
        noam_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    norm_stats = load_normalization_stats(args)
    chn_ids = parse_comma_separated_floats(args.chn_ids)
    stats_subset = parse_subset_value(args.compute_stats_subset)
    requested_cols = (args.t0_col, args.t90_col, args.t360_col)
    train_path_columns = resolve_s5p_path_columns(args.train_csv, requested_cols)
    test_path_columns = resolve_s5p_path_columns(args.test_csv, requested_cols)

    cache_obj = None
    if args.local_cache_dir:
        cache_obj = StaticAnchoredCache(
            args.local_cache_dir, min_free_gb=args.local_cache_min_free_gb
        )

    base_train_ds = S5pTemporalTiffDataset(
        csv_path=args.train_csv,
        path_columns=train_path_columns,
        normalize_stats=norm_stats,
        chn_ids=chn_ids,
        compute_stats=args.compute_stats and norm_stats is None,
        compute_stats_subset=stats_subset,
        pad_to_multiple=args.pad_to_multiple,
        pad_value=args.pad_value,
        default_chn_id_value=args.default_chn_id_value,
        local_file_cache=cache_obj,
        nan_to_num=args.nan_to_num,
        data_key=args.data_key,
        chn_ids_key=args.chn_ids_key,
        channel_last=args.channel_last,
        allow_pickle=args.allow_pickle,
        scale_to_unit=args.scale_to_unit,
        scale_value=args.scale_value,
    )
    filter_dataset_to_s5p_only(base_train_ds, split_name="train")

    resolved_stats = base_train_ds.get_normalize_stats()
    if resolved_stats is None:
        resolved_stats = norm_stats
    elif norm_stats is None:
        mean, std = resolved_stats
        print(
            f"Computed normalization stats from training set (mean len={len(mean)}, std len={len(std)})",
            flush=True,
        )

    base_test_ds = S5pTemporalTiffDataset(
        csv_path=args.test_csv,
        path_columns=test_path_columns,
        normalize_stats=resolved_stats,
        chn_ids=chn_ids,
        compute_stats=False,
        pad_to_multiple=args.pad_to_multiple,
        pad_value=args.pad_value,
        default_chn_id_value=args.default_chn_id_value,
        local_file_cache=cache_obj,
        nan_to_num=args.nan_to_num,
        data_key=args.data_key,
        chn_ids_key=args.chn_ids_key,
        channel_last=args.channel_last,
        allow_pickle=args.allow_pickle,
        scale_to_unit=args.scale_to_unit,
        scale_value=args.scale_value,
    )
    filter_dataset_to_s5p_only(base_test_ds, split_name="test")

    if args.local_cache_warmup and cache_obj is not None:
        all_paths = []
        for ds in (base_train_ds, base_test_ds):
            for col in dict.fromkeys(ds.path_columns):
                if col in ds.df.columns:
                    all_paths.extend(ds.df[col].dropna().astype(str).tolist())
        cache_obj.warm_up(all_paths, max_workers=args.local_cache_workers)

    train_ds = S5pSimpleNpzDataset(base_train_ds, resize_to=args.resize_size)
    test_ds = S5pSimpleNpzDataset(base_test_ds, resize_to=args.resize_size)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory
    )

    stats_status = base_train_ds.get_stats_source()
    if stats_status is None:
        stats_status = "provided" if resolved_stats is not None else "none"

    print(
        f"Using device={device}, resize={args.resize_size}, train_samples={len(train_ds)}, test_samples={len(test_ds)}, "
        f"norm_stats={stats_status}, nan_to_num={args.nan_to_num}, train_backbone={args.train_backbone}",
        flush=True,
    )

    backbone = load_backbone(args.weights, device=device, debug=args.debug).to(device)
    head = CLSHead(embed_dim=args.embed_dim, num_classes=2).to(device)

    data_parallel = args.data_parallel
    backbone = maybe_wrap_dataparallel(backbone, data_parallel, device, "backbone")
    head = maybe_wrap_dataparallel(head, data_parallel, device, "head")

    will_train_backbone = args.train_backbone
    param_groups = [{"params": head.parameters(), "lr": args.head_lr}]
    if will_train_backbone:
        param_groups.insert(0, {"params": backbone.parameters(), "lr": args.backbone_lr})

    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(args.momentum, 0.999),
    )
    scheduler = build_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    wandb_run = init_wandb(args)

    metric_mode = args.best_mode
    if metric_mode is None:
        metric_mode = "min" if args.best_metric in {"test_loss", "test_fpr"} else "max"
    best_metric_value = float("inf") if metric_mode == "min" else float("-inf")
    best_epoch = 0
    best_ckpt_path = args.best_ckpt_path

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        freeze_backbone = (not will_train_backbone) or (
            args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs
        )
        if freeze_backbone:
            set_trainable(backbone, False)
            backbone.eval()
        else:
            set_trainable(backbone, True)
            backbone.train()
        head.train()

        total_loss = 0.0
        correct = 0
        total = 0
        for step, (x_dict, labels) in enumerate(train_loader, 1):
            labels = labels.to(device)
            x_dict = recursive_to_device(x_dict, device)
            imgs = x_dict.get("imgs")
            if isinstance(imgs, torch.Tensor) and not torch.isfinite(imgs).all():
                x_dict["imgs"] = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)

            feats = backbone(x_dict, is_training=True)
            cls_token = feats["x_norm_clstoken"]
            logits = head(cls_token)
            if not torch.isfinite(logits).all():
                print(f"[Warn] Non-finite logits at train step {step}; skipping batch.", flush=True)
                continue
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                print(f"[Warn] Non-finite loss at train step {step}; skipping batch.", flush=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                trainable_params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if args.log_interval and step % args.log_interval == 0:
                running_loss = total_loss / total
                running_acc = correct / total
                print(
                    f"Epoch {epoch} step {step}/{len(train_loader)} "
                    f"train_loss={running_loss:.4f} train_acc={running_acc:.4f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train_loss_running": running_loss,
                            "train_acc_running": running_acc,
                            "step": global_step,
                            "epoch": epoch,
                        }
                    )

            if args.max_train_steps is not None and step >= args.max_train_steps:
                break

        train_loss = total_loss / total if total > 0 else float("nan")
        train_acc = correct / total if total > 0 else 0.0

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
                imgs = x_dict.get("imgs")
                if isinstance(imgs, torch.Tensor) and not torch.isfinite(imgs).all():
                    x_dict["imgs"] = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)
                feats = backbone(x_dict, is_training=True)
                cls_token = feats["x_norm_clstoken"]
                logits = head(cls_token)
                if not torch.isfinite(logits).all():
                    print(f"[Warn] Non-finite logits at eval step {step}; skipping batch.", flush=True)
                    continue
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    print(f"[Warn] Non-finite loss at eval step {step}; skipping batch.", flush=True)
                    continue

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

        test_acc = correct / total if total > 0 else 0.0
        test_loss = test_loss_total / total if total > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        all_probs = torch.cat(all_probs) if len(all_probs) > 0 else torch.tensor([])
        all_targets = torch.cat(all_targets) if len(all_targets) > 0 else torch.tensor([])
        if all_probs.numel() > 0:
            try:
                from sklearn.metrics import roc_auc_score

                test_auroc = float(roc_auc_score(all_targets.numpy(), all_probs.numpy()))
            except Exception:
                test_auroc = float("nan")
        else:
            test_auroc = float("nan")

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"recall={recall:.4f} fpr={fpr:.4f} auroc={test_auroc:.4f}",
            flush=True,
        )

        metric_values = {
            "test_acc": test_acc,
            "test_loss": test_loss,
            "test_recall": recall,
            "test_fpr": fpr,
            "test_auroc": test_auroc,
        }
        current_metric = float(metric_values[args.best_metric])
        metric_is_finite = bool(np.isfinite(current_metric))
        improved = False
        if metric_is_finite:
            if metric_mode == "max":
                improved = current_metric > best_metric_value
            else:
                improved = current_metric < best_metric_value
        if best_ckpt_path and improved:
            best_metric_value = current_metric
            best_epoch = epoch
            ckpt_path = Path(best_ckpt_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "best_metric_name": args.best_metric,
                "best_metric_mode": metric_mode,
                "best_metric_value": best_metric_value,
                "metrics": {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_recall": recall,
                    "test_fpr": fpr,
                    "test_auroc": test_auroc,
                },
                "backbone_state_dict": unwrap_module(backbone).state_dict(),
                "head_state_dict": unwrap_module(head).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "args": vars(args),
            }
            torch.save(checkpoint, ckpt_path)
            print(
                f"[Checkpoint] Saved best model to {ckpt_path} "
                f"(epoch={epoch}, {args.best_metric}={current_metric:.6f}, mode={metric_mode}).",
                flush=True,
            )
        elif best_ckpt_path and not metric_is_finite:
            print(
                f"[Checkpoint] Skipped save at epoch {epoch}: {args.best_metric} is non-finite ({current_metric}).",
                flush=True,
            )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_recall": recall,
                    "test_fpr": fpr,
                    "test_auroc": test_auroc,
                    "best_metric_value": best_metric_value if np.isfinite(best_metric_value) else float("nan"),
                    "best_metric_epoch": best_epoch,
                }
            )

    if best_ckpt_path:
        if best_epoch > 0:
            print(
                f"[Checkpoint] Best checkpoint summary: epoch={best_epoch}, "
                f"{args.best_metric}={best_metric_value:.6f}, path={best_ckpt_path}",
                flush=True,
            )
        else:
            print(
                f"[Checkpoint] No checkpoint was saved. Metric '{args.best_metric}' never produced a finite value.",
                flush=True,
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Panopticon ViT + CLS head for Sentinel-5P temporal NPZ inputs concatenated along channels."
    )
    parser.add_argument("--train_csv", default="data_csv/hongxuan_temporal_32/train.csv")
    parser.add_argument("--test_csv", default="data_csv/hongxuan_temporal_32/test.csv")
    parser.add_argument(
        "--weights",
        default="weights/panopticon_vitb14_teacher.pth",
        help='Checkpoint path. Use "none" to train the ViT backbone from scratch.',
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the backbone during finetuning (typically smaller than head lr).",
    )
    parser.add_argument("--lr_scheduler", choices=["none", "noam"], default="noam", help="Learning rate scheduler.")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=4000,
        help="Warmup steps for Noam LR scheduler.",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (beta1) for Adam.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pad_to_multiple", type=int, default=None, help="Pad H/W to this multiple (set None to skip).")
    parser.add_argument("--pad_value", type=float, default=0.0, help="Pad value when extending to pad_to_multiple.")
    parser.add_argument("--scale_to_unit", action="store_true", help="Divide NPZ values by scale_value before normalization.")
    parser.add_argument("--scale_value", type=float, default=65535.0, help="Divisor used when scale_to_unit is enabled.")
    parser.add_argument(
        "--data_key",
        default="ch4",
        help="NPZ key that stores the image array (default: 'ch4').",
    )
    parser.add_argument("--chn_ids_key", default="chn_ids", help="NPZ key containing per-sample channel IDs if available.")
    parser.add_argument("--channel_last", action="store_true", help="Set if NPZ arrays are stored as HWC instead of CHW.")
    parser.add_argument("--ds_cfg_name", default=None, help="Optional dataset config name for channel IDs.")
    parser.add_argument("--chn_ids", default=None, help="Comma-separated channel IDs to override ds_cfg_name/NPZ.")
    parser.add_argument(
        "--default_chn_id_value",
        type=float,
        default=0.0,
        help="Constant channel id used when none are provided (keeps model happy for single-channel grids).",
    )
    parser.add_argument(
        "--allow_pickle",
        action="store_true",
        help="Allow loading NPZ files that contain pickled data (enable only if you trust the source).",
    )
    parser.add_argument(
        "--nan_to_num",
        type=float,
        default=0.0,
        help="Replace NaN/Inf in NPZ arrays with this value before normalization (default: 0.0).",
    )
    parser.add_argument(
        "--normalize_stats_npz",
        default=None,
        help="Optional .npz file with 'mean' and 'std' arrays to use for normalization.",
    )
    parser.add_argument("--normalize_mean", default=None, help="Override normalization mean as comma-separated floats.")
    parser.add_argument("--normalize_std", default=None, help="Override normalization std as comma-separated floats.")
    parser.add_argument(
        "--compute_stats_subset",
        default=None,
        help="Limit samples when computing normalization stats (int count or float fraction).",
    )
    parser.add_argument(
        "--compute_stats",
        dest="compute_stats",
        action="store_true",
        help="Compute per-channel mean/std from the training set when stats are not provided.",
    )
    parser.add_argument("--no_compute_stats", dest="compute_stats", action="store_false", help="Skip computing stats.")
    parser.set_defaults(compute_stats=True)
    parser.add_argument("--resize_size", type=int, default=224, help="Resize spatial resolution before feeding the backbone.")
    parser.add_argument("--device", default="cpu", help='PyTorch device string, e.g. "cpu" or "cuda".')
    parser.add_argument("--debug", action="store_true", help="Print stage timings and device info.")
    parser.add_argument("--log_interval", type=int, default=100, help="Print training progress every N steps.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Optional cap on number of training batches per epoch (for smoke tests).",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=None,
        help="Optional cap on number of eval batches per epoch (for smoke tests).",
    )
    parser.add_argument(
        "--t0_col",
        default="path_t0",
        help="CSV column for t0 path. Wide-table fallback uses s5p_0_path/s5p_90_path/s5p_360_path; if only s5p_0_path exists it is reused for all 3 timepoints.",
    )
    parser.add_argument(
        "--t90_col",
        default="path_t90",
        help="CSV column for t-90 path.",
    )
    parser.add_argument(
        "--t360_col",
        default="path_t360",
        help="CSV column for t-360 path.",
    )
    parser.add_argument("--local_cache_dir", default=None, help="Directory for caching NPZ files locally.")
    parser.add_argument("--local_cache_warmup", action="store_true", help="Pre-copy training/testing files to the local cache.")
    parser.add_argument("--local_cache_workers", type=int, default=8, help="Number of workers for cache warmup.")
    parser.add_argument(
        "--local_cache_min_free_gb",
        type=float,
        default=30.0,
        help="Stop caching new files when free disk space goes below this threshold (GB).",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument("--wandb_run_name", default=None, help="Optional WandB run name.")
    parser.add_argument(
        "--data_parallel",
        action="store_true",
        help="Wrap backbone and head with torch.nn.DataParallel when multiple CUDA GPUs are available.",
    )
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        help="If set, train the entire ViT; otherwise only the classifier head is trained.",
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Number of initial epochs to freeze the backbone (only used when train_backbone is True).",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm; set <=0 to disable.",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=768,
        help="Backbone output dimension (Panopticon teacher is 768).",
    )
    parser.add_argument(
        "--best_ckpt_path",
        default="checkpoints/dino_classifier_head_s5p_temporal_one_block_best.pt",
        help="Save path for the best checkpoint. Set empty string to disable checkpoint saving.",
    )
    parser.add_argument(
        "--best_metric",
        choices=["test_acc", "test_loss", "test_recall", "test_fpr", "test_auroc"],
        default="test_acc",
        help="Validation metric used to decide the best checkpoint.",
    )
    parser.add_argument(
        "--best_mode",
        choices=["max", "min"],
        default=None,
        help="Optimization direction for best_metric. Default is automatic (min for test_loss/test_fpr, else max).",
    )
    args = parser.parse_args()
    if isinstance(args.best_ckpt_path, str):
        args.best_ckpt_path = args.best_ckpt_path.strip()
        if args.best_ckpt_path == "":
            args.best_ckpt_path = None
    main(args)
