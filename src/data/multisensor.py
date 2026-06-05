"""Dataset and collation utilities for MethaneFuse multi-sensor CSVs."""

from __future__ import annotations

import hashlib
import math
import os
import random
import shutil
import warnings
import zipfile
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from thirdparty.dinov2.data.datasets.s2_csv import S2TemporalCsvDataset, _SkipSample
from thirdparty.dinov2.utils.data import extract_wavemus, load_ds_cfg

from src.data.sensor_transforms import (
    L89_PRECOMPUTED_STATS,
    S2_PRECOMPUTED_STATS,
    S5P_PRECOMPUTED_STATS,
    _compute_mean_std,
)

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
        s5p_col = None
        if "s5p_0_path" in self.df.columns:
            s5p_col = ("s5p_0_path",)
        elif "s5p_npz_path" in self.df.columns:
            s5p_col = ("s5p_npz_path",)
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
            self._validate_s5p_npz_paths(s5p_col[0])
        if wv3_cols is not None:
            self._wide_sensor_columns["wv3"] = wv3_cols

        if len(self._wide_sensor_columns) == 0:
            raise ValueError(
                "CSV has no 'sensor' column and no recognized wide-table sensor path columns "
                "(expected prefixes like s2_*, l89_*, s5p_0_path, wv3_* or emit_*)."
            )

    def _validate_s5p_npz_paths(self, column_name: str) -> None:
        bad_examples: list[str] = []
        bad_count = 0
        s5p_series = self.df[column_name]
        for row_pos, value in enumerate(s5p_series):
            text = str(value).strip()
            if text == "" or text.lower() == "nan":
                continue
            if text.lower().endswith((".tif", ".tiff")):
                bad_count += 1
                if len(bad_examples) < 5:
                    row_id = s5p_series.index[row_pos]
                    bad_examples.append(f"row={row_id}, path={text}")
        if bad_count > 0:
            raise ValueError(
                f"S5P column '{column_name}' expects NPZ files, but found {bad_count} TIFF entries. "
                f"Examples: {'; '.join(bad_examples)}"
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

    @staticmethod
    def _is_skippable_tiff_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            token in msg
            for token in (
                "deflateerror",
                "libdeflate",
                "decompress",
                "truncated stream",
                "unknown compression method",
                "tiff",
            )
        )

    @staticmethod
    def _is_skippable_s5p_error(exc: Exception) -> bool:
        if isinstance(exc, (zipfile.BadZipFile, FileNotFoundError, EOFError, OSError)):
            return True
        if not isinstance(exc, ValueError):
            return False
        msg = str(exc).lower()
        return any(
            token in msg
            for token in (
                "not a zip file",
                "failed to interpret file",
                "cannot load file",
                "truncated",
                "eof",
                "bad magic",
                "decompress",
                "unknown compression method",
            )
        )

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
            try:
                return super(TriSensorTemporalCsvDataset, self)._load_image(
                    path_to_use, column_name=column_name, sample_id=sample_id
                )
            except _SkipSample:
                raise
            except RuntimeError as exc:
                if self._is_skippable_tiff_error(exc):
                    raise _SkipSample(
                        f"Corrupt/undecodable TIFF for sample {sample_id}, column={column_name}, "
                        f"path={path_to_use}: {exc}"
                    ) from exc
                raise

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
            raise _SkipSample(f"S5P sample missing value in '{column_name}'")
        path = self._maybe_cache_path(path)
        chn_ids = None
        lower_path = path.lower()
        if lower_path.endswith((".tif", ".tiff")):
            raise RuntimeError(
                f"S5P input now expects NPZ files (e.g., .../s5p_0_path -> .npz), got TIFF: {path}"
            )
        try:
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
        except _SkipSample:
            raise
        except Exception as exc:
            if self._is_skippable_s5p_error(exc):
                raise _SkipSample(f"Corrupt/undecodable S5P NPZ at {path}: {exc}") from exc
            raise

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

