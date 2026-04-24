import argparse
import csv
import faulthandler
import json
import math
import os
import random
import shutil
import threading
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rasterio

try:
    from netCDF4 import Dataset
except Exception:
    Dataset = None

from common import has_value, read_csv_rows


GSD = {
    "s2": 10.0,
    "l89": 30.0,
    "emit": 60.0,
    "s5p": 3500.0,
}

RASTER_SENSORS = ("s2", "l89", "emit")
IMAGE_SIZE_512 = 512
CENTER = IMAGE_SIZE_512 // 2
S5P_PATCH = 3
S5P_MISSING_THRESH = 0.50
CH4_CANDIDATES = (
    "methane_mixing_ratio_bias_corrected",
    "methane_mixing_ratio",
    "xch4",
)
CHECKPOINT_SUFFIX = ".resume.json"
S5P_IO_LOCK = threading.Lock()
TIFF_WRITE_RETRIES = 3
NPZ_WRITE_RETRIES = 2

SENSOR_COLS = {
    "s2": {
        "image": ["s2_0_512_path", "s2_-90_512_path", "s2_-360_512_path"],
        "mask": "s2_mask_512_path",
        "time_names": ["0", "90", "360"],
    },
    "l89": {
        "image": ["l89_0_512_path", "l89_-90_512_path", "l89_-360_512_path"],
        "mask": "l89_mask_512_path",
        "time_names": ["0", "90", "360"],
    },
    "emit": {
        "image": ["emit_0_512_path", "emit_-90_512_path", "emit_-180_512_path"],
        "mask": "emit_mask_512_path",
        "time_names": ["0", "90", "180"],
    },
}


def is_path(v: object) -> bool:
    return has_value(v) and Path(str(v)).exists()


def parse_valid_sensors(row: Dict[str, str]) -> List[str]:
    raw = str(row.get("valid_sensors", "") or "")
    sensors = [s.strip().lower() for s in raw.split(";") if s.strip()]
    if sensors:
        return sensors
    out = []
    for s in RASTER_SENSORS:
        spec = SENSOR_COLS[s]
        if all(is_path(row.get(c, "")) for c in spec["image"]) and is_path(row.get(spec["mask"], "")):
            out.append(s)
    if all(is_path(row.get(c, "")) for c in ["S5p_path", "s5p_minus90_path", "s5p_minus360_path"]):
        out.append("s5p")
    return out


def query_patch_size_px(query_size_m: float, sensor: str) -> int:
    return max(1, int(round(query_size_m / GSD[sensor])))


def target_size_from_query(query_size_m: float, cap: int = 518) -> int:
    s2_px = query_patch_size_px(query_size_m, "s2")
    multiplier = max(18, s2_px)
    return min(int(multiplier * 14), int(cap))


def to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported raster dims: {arr.shape}")
    return arr


def read_raster_chw(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read().astype(np.float32)


def read_raster_hw(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1)


def crop_chw(arr: np.ndarray, x: int, y: int, size: int) -> Optional[np.ndarray]:
    _, h, w = arr.shape
    if x < 0 or y < 0 or x + size > w or y + size > h:
        return None
    return arr[:, y : y + size, x : x + size]


def crop_hw(arr: np.ndarray, x: int, y: int, size: int) -> Optional[np.ndarray]:
    h, w = arr.shape
    if x < 0 or y < 0 or x + size > w or y + size > h:
        return None
    return arr[y : y + size, x : x + size]


def resize_chw_linear(img: np.ndarray, out_size: int) -> np.ndarray:
    c, h, w = img.shape
    if h == out_size and w == out_size:
        return img.astype(np.float32, copy=False)
    ys = np.linspace(0, h - 1, out_size)
    xs = np.linspace(0, w - 1, out_size)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    out = np.empty((c, out_size, out_size), dtype=np.float32)
    for i in range(c):
        band = img[i].astype(np.float32, copy=False)
        a = band[y0[:, None], x0[None, :]]
        b = band[y0[:, None], x1[None, :]]
        c0 = band[y1[:, None], x0[None, :]]
        d = band[y1[:, None], x1[None, :]]
        out[i] = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c0 * (1 - wx) * wy + d * wx * wy
    return out


def resize_hw_nearest(mask: np.ndarray, out_size: int) -> np.ndarray:
    h, w = mask.shape
    if h == out_size and w == out_size:
        return mask.astype(np.uint8, copy=False)
    ys = np.clip(np.round(np.linspace(0, h - 1, out_size)).astype(np.int64), 0, h - 1)
    xs = np.clip(np.round(np.linspace(0, w - 1, out_size)).astype(np.int64), 0, w - 1)
    return mask[ys[:, None], xs[None, :]].astype(np.uint8)


def resize_nan_aware_2d(src: np.ndarray, out_size: int) -> np.ndarray:
    fin = np.isfinite(src)
    v = np.where(fin, src, 0.0).astype(np.float32)
    w = fin.astype(np.float32)
    vr = resize_chw_linear(v[None, :, :], out_size)[0]
    wr = resize_chw_linear(w[None, :, :], out_size)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(wr > 1e-6, vr / wr, np.nan).astype(np.float32)


def safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def tif_tmp_path(path: Path, attempt: int) -> Path:
    return path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{attempt}"
    )


def npz_tmp_path(path: Path, attempt: int) -> Path:
    return path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{attempt}"
    )


def backup_path_for(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.bak.{os.getpid()}.{threading.get_ident()}"
    )


def verify_tif_readback(path: Path, expected_shape: Tuple[int, int, int], expected_dtype: np.dtype) -> None:
    with rasterio.open(path) as ds:
        got = ds.read()
    if tuple(got.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"shape mismatch for {path}: expected {expected_shape}, got {tuple(got.shape)}"
        )
    if np.dtype(got.dtype) != np.dtype(expected_dtype):
        raise RuntimeError(
            f"dtype mismatch for {path}: expected {np.dtype(expected_dtype)}, got {np.dtype(got.dtype)}"
        )


def write_chw_tif(path: Path, arr: np.ndarray, max_retries: int = TIFF_WRITE_RETRIES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c, h, w = arr.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": str(arr.dtype),
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    backup: Optional[Path] = None
    if path.exists():
        backup = backup_path_for(path)
        safe_unlink(backup)
        path.replace(backup)

    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, int(max_retries)) + 1):
        tmp = tif_tmp_path(path, attempt)
        safe_unlink(tmp)
        try:
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(arr)
            verify_tif_readback(tmp, (c, h, w), arr.dtype)
            tmp.replace(path)
            verify_tif_readback(path, (c, h, w), arr.dtype)
            if backup is not None:
                safe_unlink(backup)
            return
        except Exception as exc:
            last_exc = exc
            safe_unlink(tmp)
            safe_unlink(path)

    if backup is not None and backup.exists():
        backup.replace(path)
    raise RuntimeError(
        f"TIFF write/verify failed for {path} after {max(1, int(max_retries))} attempts: {last_exc}"
    )


def write_hw_tif(path: Path, arr: np.ndarray) -> None:
    write_chw_tif(path, arr[None, :, :])


def verify_npz_readback(path: Path, expected_shape: Tuple[int, int, int]) -> None:
    with np.load(path) as data:
        if "ch4" not in data:
            raise RuntimeError(f"missing ch4 key in {path}")
        got = data["ch4"]
    if tuple(got.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"shape mismatch for {path}: expected {expected_shape}, got {tuple(got.shape)}"
        )
    if np.dtype(got.dtype) != np.dtype(np.float32):
        raise RuntimeError(f"dtype mismatch for {path}: expected float32, got {np.dtype(got.dtype)}")


def write_s5p_npz(path: Path, crops: Dict[str, np.ndarray], max_retries: int = NPZ_WRITE_RETRIES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack(
        [
            crops["0"][0].astype(np.float32, copy=False),
            crops["90"][0].astype(np.float32, copy=False),
            crops["360"][0].astype(np.float32, copy=False),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    backup: Optional[Path] = None
    if path.exists():
        backup = backup_path_for(path)
        safe_unlink(backup)
        path.replace(backup)

    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, int(max_retries)) + 1):
        tmp = npz_tmp_path(path, attempt)
        safe_unlink(tmp)
        try:
            with tmp.open("wb") as f:
                np.savez_compressed(f, ch4=arr)
            verify_npz_readback(tmp, tuple(arr.shape))
            tmp.replace(path)
            verify_npz_readback(path, tuple(arr.shape))
            if backup is not None:
                safe_unlink(backup)
            return
        except Exception as exc:
            last_exc = exc
            safe_unlink(tmp)
            safe_unlink(path)

    if backup is not None and backup.exists():
        backup.replace(path)
    raise RuntimeError(
        f"NPZ write/verify failed for {path} after {max(1, int(max_retries))} attempts: {last_exc}"
    )


def missing_ratio_valid(crops: Iterable[np.ndarray], thresh: float) -> bool:
    for crop in crops:
        if crop.size == 0:
            return False
        band0 = crop[0]
        miss = np.isnan(band0) | (band0 == 0)
        if float(miss.mean()) > thresh:
            return False
    return True


def s2_valid(crops: Iterable[np.ndarray], band_index: int = 11, zero_thresh: float = 0.20) -> bool:
    for crop in crops:
        if crop.shape[0] <= band_index:
            return False
        if float((crop[band_index] == 0).mean()) >= zero_thresh:
            return False
    return True


def center_contained(x: int, y: int, size: int, center_box_px: int) -> bool:
    cx1 = CENTER - center_box_px // 2
    cy1 = CENTER - center_box_px // 2
    cx2 = CENTER + center_box_px // 2
    cy2 = CENTER + center_box_px // 2
    return x <= cx1 and y <= cy1 and x + size >= cx2 and y + size >= cy2


def sensor_top_left(dx_m: float, dy_m: float, sensor: str, patch_size: int) -> Tuple[int, int]:
    dx = int(round(dx_m / GSD[sensor]))
    dy = int(round(dy_m / GSD[sensor]))
    cx = CENTER + dx
    cy = CENTER + dy
    half = patch_size // 2
    return cx - half, cy - half


def pick_anchor(sensors: List[str]) -> str:
    for s in ("s2", "l89", "emit", "s5p"):
        if s in sensors:
            return s
    return sensors[0]


def offset_bounds_for_anchor(anchor: str, patch_px: int) -> Tuple[int, int, int, int]:
    half = patch_px // 2
    lo = -(CENTER - half)
    hi = (IMAGE_SIZE_512 - 1 - half) - CENTER
    return lo, hi, lo, hi


def sample_positive_offset(anchor: str, patch_px: int) -> Tuple[int, int]:
    half = patch_px // 2
    pos_min = -(half - 1)
    pos_max = half - 1
    if pos_min > pos_max:
        pos_min = pos_max = 0
    return random.randint(pos_min, pos_max), random.randint(pos_min, pos_max)


def sample_negative_offset(anchor: str, patch_px: int, center_box_px: int, far_pool: int) -> Tuple[int, int]:
    lo_x, hi_x, lo_y, hi_y = offset_bounds_for_anchor(anchor, patch_px)
    candidates = []
    attempts = max(1, far_pool)
    half = patch_px // 2
    for _ in range(attempts):
        dx = random.randint(lo_x, hi_x)
        dy = random.randint(lo_y, hi_y)
        x = CENTER + dx - half
        y = CENTER + dy - half
        if not center_contained(x, y, patch_px, center_box_px):
            candidates.append((dx * dx + dy * dy, dx, dy))
    if not candidates:
        # Fall back to old random loop. This should be rare unless the query is
        # almost as large as the 512 image.
        while True:
            dx = random.randint(lo_x, hi_x)
            dy = random.randint(lo_y, hi_y)
            x = CENTER + dx - half
            y = CENTER + dy - half
            if not center_contained(x, y, patch_px, center_box_px):
                return dx, dy
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2]


def pick_ch4_var(prod) -> Optional[str]:
    for k in CH4_CANDIDATES:
        if k in prod.variables:
            return k
    return None


def to_nan_invalid(arr, attrs=None) -> np.ndarray:
    a = np.array(arr, dtype=np.float32, copy=False)
    attrs = attrs or {}
    fv = attrs.get("_FillValue", None)
    mv = attrs.get("missing_value", None)
    if fv is not None:
        a = np.where(a == np.float32(fv), np.nan, a)
    if mv is not None:
        a = np.where(a == np.float32(mv), np.nan, a)
    a = np.where(np.abs(a) > 1e20, np.nan, a)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"Unexpected S5P array dims: {a.shape}")
    return a


@contextmanager
def silence_fd2():
    """Suppress noisy stderr from some netCDF/HDF5 backends on CIFS."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(devnull)
        os.close(old)


def read_s5p_ch4(path_nc: str, ch4_hint: Optional[str] = None) -> Tuple[np.ndarray, str]:
    if Dataset is None:
        raise RuntimeError("netCDF4 is not available")
    p = Path(str(path_nc))
    if not p.exists():
        raise RuntimeError(f"S5P file does not exist: {path_nc}")
    # Serialize netCDF/HDF5 reads plus stderr redirection in multi-thread runs.
    with S5P_IO_LOCK:
        with silence_fd2():
            ds = Dataset(str(p), "r")
        try:
            prod = ds.groups["PRODUCT"]
            ch4 = ch4_hint if ch4_hint and ch4_hint in prod.variables else pick_ch4_var(prod)
            if ch4 is None:
                raise RuntimeError(f"No CH4 variable found in {path_nc}")
            v = prod.variables[ch4]
            return to_nan_invalid(v[:], getattr(v, "__dict__", {})), ch4
        finally:
            ds.close()


def s5p_crop(arr: np.ndarray, cy: int, cx: int, target_size: int) -> Optional[np.ndarray]:
    half = S5P_PATCH // 2
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    if y0 < 0 or x0 < 0 or y1 > arr.shape[0] or x1 > arr.shape[1]:
        return None
    small = arr[y0:y1, x0:x1]
    if 1.0 - (np.isfinite(small).sum() / small.size) > S5P_MISSING_THRESH:
        return None
    return resize_nan_aware_2d(small, target_size)[None, :, :]


@dataclass
class RasterBundle:
    sensor: str
    t0: np.ndarray
    t90: np.ndarray
    t360: np.ndarray
    mask: np.ndarray
    patch_px: int


@dataclass
class S5PBundle:
    t0: np.ndarray
    t90: np.ndarray
    t360: np.ndarray
    nearest_iy: int
    nearest_ix: int
    has_90: bool
    has_360: bool


def load_raster_bundle(row: Dict[str, str], sensor: str, query_size_m: float) -> Optional[RasterBundle]:
    spec = SENSOR_COLS[sensor]
    try:
        paths = [Path(str(row[c])) for c in spec["image"]]
        mpath = Path(str(row[spec["mask"]]))
        if not all(p.exists() for p in paths) or not mpath.exists():
            return None
        return RasterBundle(
            sensor=sensor,
            t0=read_raster_chw(paths[0]),
            t90=read_raster_chw(paths[1]),
            t360=read_raster_chw(paths[2]),
            mask=(read_raster_hw(mpath) > 0).astype(np.uint8),
            patch_px=query_patch_size_px(query_size_m, sensor),
        )
    except Exception:
        return None


def load_s5p_bundle(row: Dict[str, str]) -> Optional[S5PBundle]:
    try:
        if not is_path(row.get("S5p_path", "")):
            return None
        if not has_value(row.get("nearest_iy", "")) or not has_value(row.get("nearest_ix", "")):
            return None
        ch4_hint = row.get("ch4_var", None) if has_value(row.get("ch4_var", "")) else None
        t0, ch4_used = read_s5p_ch4(str(row["S5p_path"]), ch4_hint)

        has_90 = False
        has_360 = False
        t90 = np.full_like(t0, np.nan, dtype=np.float32)
        t360 = np.full_like(t0, np.nan, dtype=np.float32)

        if is_path(row.get("s5p_minus90_path", "")):
            try:
                t90, _ = read_s5p_ch4(str(row["s5p_minus90_path"]), ch4_used)
                has_90 = True
            except Exception:
                has_90 = False

        if is_path(row.get("s5p_minus360_path", "")):
            try:
                t360, _ = read_s5p_ch4(str(row["s5p_minus360_path"]), ch4_used)
                has_360 = True
            except Exception:
                has_360 = False

        return S5PBundle(
            t0=t0,
            t90=t90,
            t360=t360,
            nearest_iy=int(float(row["nearest_iy"])),
            nearest_ix=int(float(row["nearest_ix"])),
            has_90=has_90,
            has_360=has_360,
        )
    except Exception:
        return None


def check_and_crop_raster(
    bundle: RasterBundle,
    dx_m: float,
    dy_m: float,
    label: int,
    target_size: int,
) -> Tuple[bool, Dict[str, np.ndarray], str, int]:
    x, y = sensor_top_left(dx_m, dy_m, bundle.sensor, bundle.patch_px)
    c0 = crop_chw(bundle.t0, x, y, bundle.patch_px)
    c90 = crop_chw(bundle.t90, x, y, bundle.patch_px)
    c360 = crop_chw(bundle.t360, x, y, bundle.patch_px)
    cm = crop_hw(bundle.mask, x, y, bundle.patch_px)
    if c0 is None or c90 is None or c360 is None or cm is None:
        return False, {}, "out_of_bounds", 0

    if bundle.sensor == "s2" and not s2_valid([c0, c90, c360]):
        return False, {}, "quality_fail", int(cm.sum())
    if bundle.sensor == "l89" and not missing_ratio_valid([c0, c90, c360], 0.25):
        return False, {}, "quality_fail", int(cm.sum())
    if bundle.sensor == "emit" and not missing_ratio_valid([c0, c90, c360], 0.25):
        return False, {}, "quality_fail", int(cm.sum())

    pos_pixels = int((cm > 0).sum())
    if label == 1 and pos_pixels <= 0:
        return False, {}, "positive_mask_empty", pos_pixels
    if label == 0 and pos_pixels > 0:
        return False, {}, "negative_mask_nonzero", pos_pixels

    return True, {
        "0": resize_chw_linear(c0, target_size),
        "90": resize_chw_linear(c90, target_size),
        "360": resize_chw_linear(c360, target_size),
        "mask": resize_hw_nearest((cm > 0).astype(np.uint8), target_size),
    }, "ok", pos_pixels


def nan_chw(target_size: int) -> np.ndarray:
    return np.full((1, target_size, target_size), np.nan, dtype=np.float32)


def crop_s5p(bundle: S5PBundle, dx_m: float, dy_m: float, target_size: int) -> Tuple[bool, Dict[str, np.ndarray], str]:
    dx = int(round(dx_m / GSD["s5p"]))
    dy = int(round(dy_m / GSD["s5p"]))
    cx = bundle.nearest_ix + dx
    cy = bundle.nearest_iy + dy
    c0 = s5p_crop(bundle.t0, cy, cx, target_size)
    if c0 is None:
        return False, {}, "crop_fail_t0"
    c90 = s5p_crop(bundle.t90, cy, cx, target_size)
    c360 = s5p_crop(bundle.t360, cy, cx, target_size)
    if c90 is None:
        c90 = nan_chw(target_size)
    if c360 is None:
        c360 = nan_chw(target_size)
    return True, {"0": c0, "90": c90, "360": c360}, "ok"


class IdCounter:
    def __init__(self, start: int = 0):
        self.n = start
        self.lock = threading.Lock()

    def next(self) -> int:
        with self.lock:
            self.n += 1
            return self.n


def checkpoint_path_for(out_csv: Path) -> Path:
    return out_csv.with_name(f"{out_csv.name}{CHECKPOINT_SUFFIX}")


def save_checkpoint(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, sort_keys=True)
    tmp.replace(path)


def load_checkpoint(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid checkpoint format: {path}")
    return data


def dir_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


class StageSpaceLimiter:
    def __init__(self, max_bytes: int, existing_bytes: int, wait_sec: float):
        self.max_bytes = max(0, int(max_bytes))
        self.existing_bytes = max(0, int(existing_bytes))
        self.wait_sec = max(0.1, float(wait_sec))
        self._reserved_bytes = 0
        self._warned_oversize = False
        self._cond = threading.Condition()

    def acquire(self, need_bytes: int, row_idx: int) -> Tuple[int, int]:
        need = max(0, int(need_bytes))
        if self.max_bytes <= 0 or need == 0:
            return need, 0

        waits = 0
        with self._cond:
            while self.existing_bytes + self._reserved_bytes + need > self.max_bytes:
                # If one row is larger than cap, allow one row to proceed to avoid deadlock.
                if need > self.max_bytes and self._reserved_bytes == 0:
                    if not self._warned_oversize:
                        print(
                            f"[stage_guard] warning: single-row staged bytes ({need}) exceed cap ({self.max_bytes}); "
                            "allowing one row at a time.",
                            flush=True,
                        )
                        self._warned_oversize = True
                    break
                waits += 1
                if waits == 1:
                    used = self.existing_bytes + self._reserved_bytes
                    print(
                        f"[stage_guard] row={row_idx} waiting: used={used} need={need} cap={self.max_bytes}",
                        flush=True,
                    )
                self._cond.wait(timeout=self.wait_sec)
            self._reserved_bytes += need
        return need, waits

    def release(self, token_bytes: int) -> None:
        tok = max(0, int(token_bytes))
        if tok == 0 or self.max_bytes <= 0:
            return
        with self._cond:
            self._reserved_bytes = max(0, self._reserved_bytes - tok)
            self._cond.notify_all()


def stageable_columns() -> List[str]:
    cols: List[str] = []
    for sensor in RASTER_SENSORS:
        spec = SENSOR_COLS[sensor]
        cols.extend(spec["image"])
        cols.append(spec["mask"])
    cols.extend(["S5p_path", "s5p_minus90_path", "s5p_minus360_path"])
    return cols


def estimate_stage_bytes(row: Dict[str, str]) -> int:
    total = 0
    seen: set = set()
    for col in stageable_columns():
        value = row.get(col, "")
        if not has_value(value):
            continue
        src = Path(str(value))
        if not src.exists() or not src.is_file():
            continue
        key = str(src)
        if key in seen:
            continue
        seen.add(key)
        try:
            total += src.stat().st_size
        except OSError:
            continue
    return int(total)


def stage_row_inputs(
    row: Dict[str, str],
    stage_root: Path,
    source_row_idx: int,
    limiter: Optional[StageSpaceLimiter] = None,
) -> Tuple[Dict[str, str], Path, int, int]:
    need_bytes = estimate_stage_bytes(row)
    reserved_bytes = 0
    wait_loops = 0
    if limiter is not None:
        reserved_bytes, wait_loops = limiter.acquire(need_bytes, source_row_idx)

    stage_dir = stage_root / f"row_{source_row_idx:08d}"
    try:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True, exist_ok=True)

        staged = dict(row)
        for col in stageable_columns():
            value = row.get(col, "")
            if not has_value(value):
                continue
            src = Path(str(value))
            if not src.exists():
                continue
            suffix = "".join(src.suffixes) or ".bin"
            dst = stage_dir / f"{col}{suffix}"
            shutil.copy2(src, dst)
            staged[col] = str(dst)
        return staged, stage_dir, reserved_bytes, wait_loops
    except Exception:
        if limiter is not None and reserved_bytes > 0:
            limiter.release(reserved_bytes)
        raise


def infer_resume_from_csv(path: Path) -> Tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not fields:
            return 0, 0, 0
        if "source_row_idx" not in fields:
            raise RuntimeError(
                f"{path} has no source_row_idx column; cannot infer resume point. "
                "Use a fresh out_csv path or regenerate with the updated script."
            )
        max_row_idx = -1
        max_id = 0
        samples = 0
        for row in reader:
            samples += 1
            rid = row.get("id", "")
            if has_value(rid):
                try:
                    max_id = max(max_id, int(float(rid)))
                except Exception:
                    pass
            rix = row.get("source_row_idx", "")
            if has_value(rix):
                try:
                    max_row_idx = max(max_row_idx, int(float(rix)))
                except Exception:
                    pass
    return max_row_idx + 1, max_id, samples


def write_sample_files(sample_dir: Path, sensor: str, crops: Dict[str, np.ndarray]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if sensor == "s5p":
        path = sample_dir / "s5p_triplet.npz"
        write_s5p_npz(path, crops)
        out["s5p_npz_path"] = str(path)
        return out

    for key in ("0", "90", "360"):
        if key not in crops:
            continue
        path = sample_dir / f"{sensor}_{key}.tif"
        write_chw_tif(path, crops[key].astype(np.float32))
        out[f"{sensor}_{key}_path"] = str(path)
    if "mask" in crops:
        path = sample_dir / f"{sensor}_mask.tif"
        write_hw_tif(path, crops["mask"].astype(np.uint8))
        out[f"{sensor}_mask_path"] = str(path)
    return out


def try_make_sample(
    row: Dict[str, str],
    raster_bundles: Dict[str, RasterBundle],
    s5p_bundle: Optional[S5PBundle],
    label: int,
    dx_anchor_px: int,
    dy_anchor_px: int,
    anchor: str,
    target_size: int,
    query_size_m: float,
    out_root: Path,
    counter: IdCounter,
    debug_stats: Counter,
) -> Optional[Dict[str, object]]:
    dx_m = dx_anchor_px * GSD[anchor]
    dy_m = dy_anchor_px * GSD[anchor]
    sensor_crops: Dict[str, Dict[str, np.ndarray]] = {}
    mask_counts: Dict[str, int] = {}

    for sensor, bundle in raster_bundles.items():
        ok, crops, reason, pos_pixels = check_and_crop_raster(bundle, dx_m, dy_m, label, target_size)
        if not ok:
            debug_stats[f"{sensor}_{reason}_label{label}"] += 1
            return None
        sensor_crops[sensor] = crops
        mask_counts[sensor] = pos_pixels

    if s5p_bundle is not None:
        ok, crops, reason = crop_s5p(s5p_bundle, dx_m, dy_m, target_size)
        if ok:
            sensor_crops["s5p"] = crops
        else:
            debug_stats[f"s5p_{reason}_label{label}"] += 1

    if not sensor_crops:
        debug_stats[f"no_sensor_written_label{label}"] += 1
        return None

    sid = counter.next()
    sample_dir = out_root / f"query_{sid:08d}"
    rec: Dict[str, object] = {
        "id": sid,
        "plume_id": row.get("plume_id", ""),
        "label": int(label),
        "query_size_m": float(query_size_m),
        "target_size": int(target_size),
        "anchor_sensor": anchor,
        "dx_anchor_px": int(dx_anchor_px),
        "dy_anchor_px": int(dy_anchor_px),
        "dx_m": float(dx_m),
        "dy_m": float(dy_m),
        "latitude": row.get("latitude", ""),
        "longitude": row.get("longitude", ""),
        "datetime": row.get("datetime", ""),
    }
    written = []
    try:
        for sensor, crops in sensor_crops.items():
            rec.update(write_sample_files(sample_dir, sensor, crops))
            written.append(sensor)
    except Exception:
        shutil.rmtree(sample_dir, ignore_errors=True)
        debug_stats[f"sample_write_fail_label{label}"] += 1
        return None
    rec["written_sensors"] = ";".join(written)
    for sensor in RASTER_SENSORS:
        rec[f"{sensor}_mask_positive_pixels"] = mask_counts.get(sensor, "")
    return rec


def process_row(row: Dict[str, str], args, counter: IdCounter, source_row_idx: int) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    local_stats: Counter = Counter()
    work_row = row
    stage_dir: Optional[Path] = None
    staged_reserved_bytes = 0
    stage_limiter: Optional[StageSpaceLimiter] = getattr(args, "_stage_limiter", None)
    if args.local_stage_root is not None:
        try:
            work_row, stage_dir, staged_reserved_bytes, wait_loops = stage_row_inputs(
                row,
                args.local_stage_root,
                source_row_idx,
                limiter=stage_limiter,
            )
            if wait_loops > 0:
                local_stats["stage_wait_rows"] += 1
                local_stats["stage_wait_loops"] += wait_loops
        except Exception:
            local_stats["stage_copy_fail"] += 1
            return [], dict(local_stats)

    try:
        valid_sensors = parse_valid_sensors(work_row)
        raster_sensors = [s for s in RASTER_SENSORS if s in valid_sensors]
        raster_bundles = {
            s: b
            for s in raster_sensors
            if (b := load_raster_bundle(work_row, s, args.query_size_m)) is not None
        }
        s5p_bundle = load_s5p_bundle(work_row) if "s5p" in valid_sensors else None
        loaded = list(raster_bundles.keys()) + (["s5p"] if s5p_bundle is not None else [])
        if not loaded:
            local_stats["skip_no_loaded_sensor"] += 1
            return [], dict(local_stats)

        anchor = pick_anchor(loaded)
        anchor_patch = S5P_PATCH if anchor == "s5p" else query_patch_size_px(args.query_size_m, anchor)
        center_box_px = max(1, int(round(args.center_box_m / GSD[anchor])))
        target_size = args.target_size if args.target_size > 0 else target_size_from_query(args.query_size_m, args.target_cap)

        out_rows: List[Dict[str, object]] = []
        for label, target_n, max_attempts in [(1, args.n_pos, args.max_attempts_pos), (0, args.n_neg, args.max_attempts_neg)]:
            made = 0
            attempts = 0
            while made < target_n and attempts < max_attempts:
                attempts += 1
                if label == 1:
                    dx_anchor, dy_anchor = sample_positive_offset(anchor, anchor_patch)
                else:
                    dx_anchor, dy_anchor = sample_negative_offset(anchor, anchor_patch, center_box_px, args.negative_far_pool)
                rec = try_make_sample(
                    row=work_row,
                    raster_bundles=raster_bundles,
                    s5p_bundle=s5p_bundle,
                    label=label,
                    dx_anchor_px=dx_anchor,
                    dy_anchor_px=dy_anchor,
                    anchor=anchor,
                    target_size=target_size,
                    query_size_m=args.query_size_m,
                    out_root=args.out_root,
                    counter=counter,
                    debug_stats=local_stats,
                )
                if rec is None:
                    continue
                rec["source_row_idx"] = int(source_row_idx)
                out_rows.append(rec)
                made += 1
            local_stats[f"made_label{label}"] += made
            if made < target_n:
                local_stats[f"short_label{label}"] += 1
        return out_rows, dict(local_stats)
    finally:
        if stage_dir is not None and not args.keep_staged_inputs:
            shutil.rmtree(stage_dir, ignore_errors=True)
        if stage_limiter is not None and staged_reserved_bytes > 0 and not args.keep_staged_inputs:
            stage_limiter.release(staged_reserved_bytes)


def output_fieldnames() -> List[str]:
    fields = [
        "id",
        "source_row_idx",
        "plume_id",
        "label",
        "query_size_m",
        "target_size",
        "anchor_sensor",
        "dx_anchor_px",
        "dy_anchor_px",
        "dx_m",
        "dy_m",
        "latitude",
        "longitude",
        "datetime",
        "written_sensors",
    ]
    for sensor in ("s2", "l89", "emit"):
        for key in ("0", "90", "360"):
            fields.append(f"{sensor}_{key}_path")
        fields.append(f"{sensor}_mask_path")
    fields.append("s5p_npz_path")
    for sensor in RASTER_SENSORS:
        fields.append(f"{sensor}_mask_positive_pixels")
    return fields


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create query-size multisensor crops from raw512 manifest.")
    p.add_argument("--manifest_csv", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--query_size_m", type=float, required=True)
    p.add_argument("--target_size", type=int, default=0, help="0 means auto from S2 crop size.")
    p.add_argument("--target_cap", type=int, default=518)
    p.add_argument("--center_box_m", type=float, default=100.0, help="Old-pipeline center box, converted to anchor pixels.")
    p.add_argument("--n_pos", type=int, default=16)
    p.add_argument("--n_neg", type=int, default=16)
    p.add_argument("--max_attempts_pos", type=int, default=800)
    p.add_argument("--max_attempts_neg", type=int, default=5000)
    p.add_argument("--negative_far_pool", type=int, default=8, help="Sample old-valid negatives and keep the farthest.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=50, help="Checkpoint every N processed manifest rows.")
    p.add_argument("--resume", action="store_true", help="Resume from existing out_csv/checkpoint.")
    p.add_argument("--local_stage_root", type=Path, default=None, help="Stage each row's inputs to local disk before processing.")
    p.add_argument("--keep_staged_inputs", action="store_true", help="Keep staged per-row files under local_stage_root.")
    p.add_argument("--stage_max_gb", type=float, default=0.0, help="Max local staging bytes in GB; workers wait when cap is reached.")
    p.add_argument("--stage_wait_sec", type=float, default=2.0, help="Wait interval seconds when local staging cap is reached.")
    return p.parse_args()


def main() -> None:
    faulthandler.enable(all_threads=True)
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.stage_max_gb < 0:
        raise RuntimeError("--stage_max_gb must be >= 0")
    if args.stage_wait_sec <= 0:
        raise RuntimeError("--stage_wait_sec must be > 0")
    if args.stage_max_gb > 0 and args.local_stage_root is None:
        raise RuntimeError("--stage_max_gb requires --local_stage_root")
    if args.stage_max_gb > 0 and args.keep_staged_inputs:
        raise RuntimeError("--stage_max_gb cannot be used with --keep_staged_inputs")

    stage_limiter: Optional[StageSpaceLimiter] = None
    if args.local_stage_root is not None:
        args.local_stage_root.mkdir(parents=True, exist_ok=True)
        if args.stage_max_gb > 0:
            max_bytes = int(args.stage_max_gb * (1024 ** 3))
            existing_bytes = dir_size_bytes(args.local_stage_root)
            stage_limiter = StageSpaceLimiter(
                max_bytes=max_bytes,
                existing_bytes=existing_bytes,
                wait_sec=args.stage_wait_sec,
            )
            print(
                f"[stage_guard] root={args.local_stage_root} cap_bytes={max_bytes} existing_bytes={existing_bytes}",
                flush=True,
            )
    setattr(args, "_stage_limiter", stage_limiter)

    _, rows = read_csv_rows(args.manifest_csv)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    total_rows = len(rows)
    fields = output_fieldnames()
    ckpt_path = checkpoint_path_for(args.out_csv)
    start_row_idx = 0
    counter_start = 0
    samples_written = 0
    stats: Counter = Counter()

    if args.resume:
        if ckpt_path.exists():
            ckpt = load_checkpoint(ckpt_path)
            ckpt_manifest = str(ckpt.get("manifest_csv", ""))
            if ckpt_manifest and ckpt_manifest != str(args.manifest_csv):
                raise RuntimeError(
                    f"Checkpoint manifest mismatch: {ckpt_manifest} != {args.manifest_csv}"
                )
            start_row_idx = int(ckpt.get("next_row_idx", 0))
            counter_start = int(ckpt.get("counter_n", 0))
            samples_written = int(ckpt.get("samples_written", 0))
            for k, v in dict(ckpt.get("stats", {})).items():
                try:
                    stats[str(k)] = int(v)
                except Exception:
                    continue
            print(
                f"[resume] checkpoint loaded: next_row_idx={start_row_idx} "
                f"counter_start={counter_start} samples={samples_written}",
                flush=True,
            )
        elif args.out_csv.exists():
            start_row_idx, counter_start, samples_written = infer_resume_from_csv(args.out_csv)
            print(
                f"[resume] inferred from csv: next_row_idx={start_row_idx} "
                f"counter_start={counter_start} samples={samples_written}",
                flush=True,
            )
    else:
        if args.out_csv.exists():
            args.out_csv.unlink()
        if ckpt_path.exists():
            ckpt_path.unlink()

    if start_row_idx < 0 or start_row_idx > total_rows:
        raise RuntimeError(
            f"Invalid resume row index {start_row_idx}, expected in [0, {total_rows}]"
        )

    counter = IdCounter(start=counter_start)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    append_mode = args.resume and args.out_csv.exists() and start_row_idx > 0

    if start_row_idx >= total_rows:
        if not args.out_csv.exists():
            with args.out_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
        if ckpt_path.exists():
            ckpt_path.unlink()
        print("[resume] all rows already processed.", flush=True)
        print(f"saved: {args.out_csv}")
        print(f"rows_in: {total_rows}")
        print(f"samples: {samples_written}")
        for key in sorted(stats):
            print(f"{key}: {stats[key]}")
        return

    def _work(item):
        row_idx, row = item
        recs, local = process_row(row, args, counter, row_idx)
        return row_idx, recs, local

    with args.out_csv.open("a" if append_mode else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not append_mode:
            writer.writeheader()

        work_items = enumerate(rows[start_row_idx:], start=start_row_idx)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            for processed, (row_idx, recs, local) in enumerate(ex.map(_work, work_items), start=1):
                for rec in recs:
                    writer.writerow(rec)
                samples_written += len(recs)
                stats.update(local)
                done_rows = row_idx + 1

                if args.debug_every > 0 and (processed % args.debug_every == 0 or done_rows == total_rows):
                    print(f"[progress] rows={done_rows}/{total_rows} samples={samples_written}", flush=True)

                if args.save_every > 0 and (processed % args.save_every == 0 or done_rows == total_rows):
                    f.flush()
                    save_checkpoint(
                        ckpt_path,
                        {
                            "manifest_csv": str(args.manifest_csv),
                            "next_row_idx": done_rows,
                            "counter_n": counter.n,
                            "samples_written": samples_written,
                            "stats": dict(stats),
                        },
                    )

    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"saved: {args.out_csv}")
    print(f"rows_in: {total_rows}")
    print(f"samples: {samples_written}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
