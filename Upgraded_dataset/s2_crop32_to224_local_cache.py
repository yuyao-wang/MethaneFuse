#!/usr/bin/env python3
"""Create Sentinel-2 six-time 32-patch dataset and 224-resized copy.

This wraps cell 7 of the old notebook
``preprocess_dataset_s2/carbon_mapper_sentinel2_plume_-7_download.ipynb``,
but keeps all six current S2 timepoints. For every accepted sample, the same
32x32 crop window is applied to t0, prev1(-7), prev2, prev3, seasonal(-90),
and year(-360). The 224 step resizes those six image tensors and the plume
mask.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "Upgrade_data_pipeline" / "csv").exists():
            return parent
    fallback = Path("/home/yuyao/methane_train")
    if (fallback / "Upgrade_data_pipeline" / "csv").exists():
        return fallback
    raise RuntimeError(f"Could not find repo root from {here}")


REPO_ROOT = find_repo_root()
DEFAULT_INPUT_CSV = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_all6_available_paths_std512_complete.csv"
DEFAULT_OUT_ROOT_32 = Path("/mnt/engg-niulab/yuyao/final_crop/s2_6time_32")
DEFAULT_OUT_ROOT_224 = Path("/mnt/engg-niulab/yuyao/final_crop/s2_6time_32_to_224")
DEFAULT_MASK_ROOTS = [
    Path("/mnt/engg-niulab/yuyao/sensors_raw_data/CM"),
    Path("/mnt/engg-niulab/Yuyao/sensors_raw_data/CM"),
    Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_masks_wv3_512"),
    Path("/data2/yuyao/methane_emission/carbon_mapper_data_masks"),
]


@dataclass(frozen=True)
class Timepoint:
    name: str
    std512_col: str
    patch_name: str
    image_time_col: str


TIMEPOINTS = [
    Timepoint("t0", "s2_0_std_512", "s2_0.tif", "t0_image_time"),
    Timepoint("prev1", "s2_-7_std_512", "s2_prev1.tif", "prev1_image_time"),
    Timepoint("prev2", "s2_prev2_std_512", "s2_prev2.tif", "prev2_image_time"),
    Timepoint("prev3", "s2_prev3_std_512", "s2_prev3.tif", "prev3_image_time"),
    Timepoint("seasonal", "s2_-90_std_512", "s2_seasonal.tif", "seasonal_image_time"),
    Timepoint("year", "s2_-360_std_512", "s2_year.tif", "year_image_time"),
]
REQUIRED_NONZERO_BANDS = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)

STD512_COLS = [tp.std512_col for tp in TIMEPOINTS]
SIX_TIME_PATH_COLS = [f"path_{tp.name}" for tp in TIMEPOINTS]
LEGACY_ALIAS_COLS = {
    "image_path": "path_t0",
    "s2_path": "path_t0",
    "s2_-7_path": "path_prev1",
    "s2_pre_path": "path_seasonal",
    "s2_pre_pre_path": "path_year",
}
MASK_512_COLUMNS = [
    "resized_512x512_path",
    "s2_plume_mask_512_path",
    "s2_mask_512_path",
    "mask_512_path",
    "mask_path_512",
    "mask_path",
]
CM_MASK_COLUMNS = [
    "cm_plume_tif_local_path",
    "local_plume_tif",
    "plume_tif_path",
    "plume_path",
    "plume_tif",
]
PREALIGNED_MASK_NAMES = [
    "resized_512x512.tif",
    "mask_512.tif",
    "mask_60m_512.tif",
    "mask_30m_512.tif",
    "plume_512.tif",
]


@dataclass(frozen=True)
class PatchArrays:
    kind: str
    index: int
    x: int
    y: int
    crops: dict[str, np.ndarray]
    plume: np.ndarray
    label: int
    source: str


def debug(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][pid:{os.getpid()}][tid:{threading.get_ident()}] {msg}", flush=True)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def existing_file(value: Any) -> bool:
    if not has_value(value):
        return False
    path = Path(str(value).strip())
    return path.exists() and path.stat().st_size > 0


def ensure_abs(value: Any) -> str:
    if not has_value(value):
        return ""
    text = str(value).strip()
    return text if os.path.isabs(text) else os.path.abspath(text)


def sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha1("|".join(parts).encode("utf-8", "ignore")).hexdigest()
    return int(seed) + int(digest[:8], 16)


def to_chw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected ndim={arr.ndim}, shape={arr.shape}")
    if arr.shape[0] in (1, 3, 4, 12, 13):
        return arr
    if arr.shape[-1] in (1, 3, 4, 12, 13):
        return np.transpose(arr, (2, 0, 1))
    if arr.shape[0] <= 20 and arr.shape[0] < arr.shape[-1]:
        return arr
    return np.transpose(arr, (2, 0, 1))


def read_chw_512(path: str, tag: str, bug: list[str]) -> np.ndarray | None:
    path = ensure_abs(path)
    if not existing_file(path):
        bug.append(f"{tag}_missing:{path}")
        return None
    try:
        chw = to_chw(tifffile.imread(path)).astype(np.float32, copy=False)
        if chw.shape[-2:] != (512, 512):
            bug.append(f"{tag}_not512:{chw.shape}")
            return None
        return chw
    except Exception as exc:
        bug.append(f"{tag}_read_err:{type(exc).__name__}:{exc}")
        return None


def stage_source_file(
    source: str,
    destination: Path,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}."
        f"{threading.get_ident()}.{uuid.uuid4().hex}.part"
    )
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(destination)


def to_hw_mask(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        mask = arr
    elif arr.ndim == 3:
        if arr.shape[0] <= 4 and arr.shape[1] > 4 and arr.shape[2] > 4:
            chw = arr
        elif arr.shape[-1] <= 4:
            chw = np.moveaxis(arr, -1, 0)
        elif arr.shape[0] <= 16 and arr.shape[0] < arr.shape[-1]:
            chw = arr
        else:
            chw = np.moveaxis(arr, -1, 0)
        candidate = chw[-1]
        if np.count_nonzero(candidate) == 0:
            candidate = np.max(chw, axis=0)
        mask = candidate
    else:
        raise ValueError(f"Unexpected mask ndim={arr.ndim}, shape={arr.shape}")
    return (np.nan_to_num(mask) > 0).astype(np.uint8)


def resize_hw_nearest(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape == (out_h, out_w):
        return mask
    src_h, src_w = mask.shape
    if src_h <= 0 or src_w <= 0 or out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid nearest resize {mask.shape} -> {(out_h, out_w)}")
    ys = np.linspace(0, src_h - 1, out_h)
    xs = np.linspace(0, src_w - 1, out_w)
    ys = np.clip(np.round(ys).astype(np.int64), 0, src_h - 1)
    xs = np.clip(np.round(xs).astype(np.int64), 0, src_w - 1)
    return mask[ys[:, None], xs[None, :]]


def write_tif_atomic(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp.tif")
    try:
        tifffile.imwrite(str(tmp), arr)
        tmp.replace(path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except FileNotFoundError:
            pass


def read_existing_512_mask(path: Path) -> np.ndarray | None:
    try:
        mask = to_hw_mask(tifffile.imread(str(path)))
        if mask.shape == (512, 512):
            return mask
    except Exception:
        return None
    return None


def candidate_prealigned_masks(row: dict[str, Any], roots: list[Path], plume_id: str) -> list[Path]:
    out: list[Path] = []
    for col in MASK_512_COLUMNS:
        value = row.get(col, "")
        if existing_file(value):
            out.append(Path(str(value).strip()))

    for root in roots:
        plume_dir = root / plume_id
        for name in PREALIGNED_MASK_NAMES:
            out.append(plume_dir / name)
        if plume_dir.exists():
            out.extend(sorted(plume_dir.glob("*512*.tif")))

    seen: set[str] = set()
    unique: list[Path] = []
    for path in out:
        text = str(path)
        if text not in seen:
            seen.add(text)
            unique.append(path)
    return unique


def candidate_cm_mask(row: dict[str, Any], roots: list[Path], plume_id: str) -> Path | None:
    for col in CM_MASK_COLUMNS:
        value = row.get(col, "")
        if existing_file(value):
            return Path(str(value).strip())

    for root in roots:
        candidates = [
            root / plume_id / "plume.tif",
            root / plume_id / f"{plume_id}_plume.tif",
            root / f"{plume_id}.tif",
            root / f"{plume_id}_plume.tif",
        ]
        for path in candidates:
            if path.exists() and path.stat().st_size > 0:
                return path
        plume_dir = root / plume_id
        if plume_dir.exists():
            for path in sorted(plume_dir.glob("*plume*.tif")):
                if path.exists() and path.stat().st_size > 0:
                    return path
    return None


def read_or_build_mask_512(
    row: dict[str, Any],
    mask_roots: list[Path],
    mask_out_root: Path,
    build_from_cm: bool,
    overwrite: bool,
) -> tuple[np.ndarray | None, str, str]:
    plume_id = str(row.get("plume_id", "")).strip()
    if not plume_id:
        return None, "", "missing_plume_id"

    for path in candidate_prealigned_masks(row, mask_roots, plume_id):
        if path.exists() and path.stat().st_size > 0:
            mask = read_existing_512_mask(path)
            if mask is not None:
                return mask, str(path), "existing_512_mask"

    if not build_from_cm:
        return None, "", "missing_512_mask"

    out_path = mask_out_root / sanitize_component(plume_id) / "resized_512x512.tif"
    if out_path.exists() and not overwrite:
        mask = read_existing_512_mask(out_path)
        if mask is not None:
            return mask, str(out_path), "existing_built_mask"

    src = candidate_cm_mask(row, mask_roots, plume_id)
    if src is None:
        return None, "", "missing_source_cm_mask"

    try:
        source_mask = to_hw_mask(tifffile.imread(str(src)))
        mask512 = resize_hw_nearest(source_mask, 512, 512).astype(np.uint8, copy=False)
        write_tif_atomic(out_path, mask512)
        return mask512, str(out_path), f"built_from:{src}"
    except Exception as exc:
        return None, "", f"build_mask_err:{type(exc).__name__}:{exc}"


def band_zero_too_much(chw: np.ndarray, band_index: int, threshold: float) -> bool:
    if chw is None or chw.ndim != 3:
        return True
    if chw.shape[0] <= band_index:
        return False
    total = chw[band_index].size
    if total == 0:
        return True
    return float((chw[band_index] == 0).sum()) / float(total) >= threshold


def crop_chw(chw: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    return chw[:, y : y + size, x : x + size]


def crop_hw(hw: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    return hw[y : y + size, x : x + size]


def sample_center_xy(rng: random.Random, width: int, height: int, patch_size: int, center_box: int) -> tuple[int, int]:
    cx = width // 2
    cy = height // 2
    half_box = center_box // 2
    min_center_x = cx - half_box
    max_center_x = cx + half_box
    min_center_y = cy - half_box
    max_center_y = cy + half_box
    min_x = max(0, min_center_x - patch_size // 2)
    max_x = min(width - patch_size, max_center_x - patch_size // 2)
    min_y = max(0, min_center_y - patch_size // 2)
    max_y = min(height - patch_size, max_center_y - patch_size // 2)
    if min_x > max_x or min_y > max_y:
        raise ValueError(
            f"center_box_invalid:min_x={min_x},max_x={max_x},min_y={min_y},max_y={max_y},"
            f"patch={patch_size},center_box={center_box}"
        )
    return rng.randint(int(min_x), int(max_x)), rng.randint(int(min_y), int(max_y))


def legacy_center_contained(x: int, y: int, size: int, center_box: int) -> bool:
    """Copied from the old legacy_param crop: crop contains the center anchor box."""
    center = 512 // 2
    cx1 = center - center_box // 2
    cy1 = center - center_box // 2
    cx2 = center + center_box // 2
    cy2 = center + center_box // 2
    x1, y1 = x, y
    x2, y2 = x + size, y + size
    return x1 <= cx1 and y1 <= cy1 and x2 >= cx2 and y2 >= cy2


def legacy_center_offsets(
    rng: random.Random,
    patch_size: int,
    n_pos: int,
    n_neg: int,
    center_box: int,
) -> list[tuple[int, int, int]]:
    """Copied from preprocess_dataset_query_multi/crop_legacy_param.py.

    Labels are center-anchor labels, not mask-content labels:
    positives keep the plume center point inside the crop; negatives are sampled
    from locations whose crop does not contain the legacy center anchor box.
    """
    center = 512 // 2
    half = patch_size // 2
    out: list[tuple[int, int, int]] = []

    pos_min = -(half - 1)
    pos_max = half - 1
    for _ in range(n_pos):
        out.append((1, rng.randint(pos_min, pos_max), rng.randint(pos_min, pos_max)))

    while len(out) < n_pos + n_neg:
        dx = rng.randint(-(center - half), (512 - 1 - half) - center)
        dy = rng.randint(-(center - half), (512 - 1 - half) - center)
        x = center + dx - half
        y = center + dy - half
        if not legacy_center_contained(x, y, patch_size, center_box):
            out.append((0, dx, dy))
    return out


def notebook_cell7_samples(
    rng: random.Random,
    patch_size: int,
    n_pos: int,
    n_neg: int,
    positive_center_size: int,
    label_center_size: int,
) -> list[tuple[int, int, int, str]]:
    """Copy the sampling and labeling geometry from notebook cell 7."""
    width = 512
    height = 512
    center_x = width // 2
    center_y = height // 2
    center_left = center_x - positive_center_size // 2
    center_right = center_x + positive_center_size // 2
    center_top = center_y - positive_center_size // 2
    center_bottom = center_y + positive_center_size // 2

    left_min = max(0, center_right - patch_size)
    left_max = min(center_left, width - patch_size)
    top_min = max(0, center_bottom - patch_size)
    top_max = min(center_top, height - patch_size)
    if left_min > left_max or top_min > top_max:
        raise ValueError(
            "notebook_cell7_positive_geometry_invalid:"
            f"x={left_min}:{left_max},y={top_min}:{top_max},"
            f"patch={patch_size},center={positive_center_size}"
        )

    samples: list[tuple[int, int, int, str]] = []
    for _ in range(n_pos):
        samples.append(
            (
                1,
                rng.randint(left_min, left_max),
                rng.randint(top_min, top_max),
                "notebook_cell7_positive",
            )
        )

    for _ in range(n_neg):
        x = rng.randint(0, width - patch_size)
        y = rng.randint(0, height - patch_size)
        label = int(
            legacy_center_contained(
                x,
                y,
                patch_size,
                label_center_size,
            )
        )
        samples.append((label, x, y, "notebook_cell7_random"))
    return samples


def sample_legacy_positive_xy(
    rng: random.Random,
    patch_size: int,
) -> tuple[int, int]:
    center = 512 // 2
    half = patch_size // 2
    jitter = half - 1
    dx = rng.randint(-jitter, jitter)
    dy = rng.randint(-jitter, jitter)
    return center + dx - half, center + dy - half


def sample_legacy_negative_xy(
    rng: random.Random,
    patch_size: int,
    center_box: int,
) -> tuple[int, int]:
    center = 512 // 2
    half = patch_size // 2
    while True:
        dx = rng.randint(-(center - half), (512 - 1 - half) - center)
        dy = rng.randint(-(center - half), (512 - 1 - half) - center)
        x = center + dx - half
        y = center + dy - half
        if not legacy_center_contained(x, y, patch_size, center_box):
            return x, y


def crop_all_timepoints(images: dict[str, np.ndarray], x: int, y: int, size: int) -> dict[str, np.ndarray]:
    return {name: crop_chw(chw, x, y, size) for name, chw in images.items()}


def sample_is_clean(crops: dict[str, np.ndarray], args: argparse.Namespace) -> bool:
    size = int(args.patch_size)
    quality_timepoints = [
        value.strip()
        for value in str(args.quality_timepoints).split(",")
        if value.strip()
    ]
    for timepoint in quality_timepoints:
        crop = crops[timepoint]
        if crop.shape[-2:] != (size, size):
            return False
        if (
            str(args.label_rule) != "legacy_center_anchor"
            and any(
                band >= crop.shape[0] or not np.any(crop[band])
                for band in REQUIRED_NONZERO_BANDS
            )
        ):
            return False
        if band_zero_too_much(crop, int(args.band_index), float(args.zero_ratio_thresh)):
            return False
    return True


def write_patch_sample(
    sample: PatchArrays,
    row: dict[str, Any],
    out_root: Path,
    split_name: str,
    mask_path_512: str,
    overwrite: bool,
    output_cache_root: str,
) -> dict[str, Any]:
    pid = str(row.get("plume_id", "")).strip()
    plume_dir = out_root / split_name / sanitize_component(pid)
    sample_dir = plume_dir / f"{sample.kind}_{sample.index:02d}_x{sample.x}_y{sample.y}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    cache_sample_dir = (
        Path(output_cache_root)
        / sanitize_component(split_name)
        / sanitize_component(pid)
        / sample_dir.name
        if has_value(output_cache_root)
        else None
    )

    def write_output(destination: Path, array: np.ndarray) -> None:
        if cache_sample_dir is None:
            write_tif_atomic(destination, array)
            return
        cached = cache_sample_dir / destination.name
        write_tif_atomic(cached, array)
        stage_source_file(str(cached), destination)
        cached.unlink(missing_ok=True)

    paths: dict[str, str] = {}
    for tp in TIMEPOINTS:
        dst = sample_dir / tp.patch_name
        if overwrite or not (dst.exists() and dst.stat().st_size > 0):
            write_output(dst, sample.crops[tp.name])
        paths[f"path_{tp.name}"] = str(dst)

    plume_path = sample_dir / "plume.tif"
    if overwrite or not (plume_path.exists() and plume_path.stat().st_size > 0):
        write_output(
            plume_path,
            sample.plume.astype(np.uint8, copy=False),
        )
    if cache_sample_dir is not None:
        shutil.rmtree(cache_sample_dir, ignore_errors=True)

    rec = {
        "plume_id": pid,
        "split": split_name,
        "label": int(sample.label),
        "path": str(sample_dir),
        "data_path": str(sample_dir),
        "path_plume": str(plume_path),
        "plume_mask_path": str(plume_path),
        "mask_path": str(plume_path),
        "crop_x": int(sample.x),
        "crop_y": int(sample.y),
        "source_x": int(sample.x),
        "source_y": int(sample.y),
        "source": sample.source,
        "mask_path_512": mask_path_512,
        "event_group_id": row.get("event_group_id", ""),
        "plume_latitude": row.get("plume_latitude", ""),
        "plume_longitude": row.get("plume_longitude", ""),
        "latitude": row.get("plume_latitude", ""),
        "longitude": row.get("plume_longitude", ""),
        "event_time": row.get("event_time", row.get("datetime", "")),
    }
    for tp in TIMEPOINTS:
        rec[tp.image_time_col] = row.get(tp.image_time_col, "")
    rec.update(paths)
    for alias, path_col in LEGACY_ALIAS_COLS.items():
        rec[alias] = rec[path_col]
    return rec


def crop_one_plume(row: dict[str, Any], split_name: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    pid = str(row.get("plume_id", "")).strip()
    if not pid:
        return [], "missing_plume_id"

    bug: list[str] = []
    images: dict[str, np.ndarray] = {}
    raw_parts: list[str] = []
    source_paths: dict[str, str] = {}
    for tp in TIMEPOINTS:
        path = ensure_abs(row.get(tp.std512_col, ""))
        raw_parts.append(f"{tp.name}={path}")
        source_paths[tp.name] = path
    rawvals = "rawvals:" + "|".join(raw_parts)
    cache_dir: Path | None = None
    if has_value(args.source_cache_root):
        cache_dir = (
            Path(args.source_cache_root)
            / sanitize_component(split_name)
            / sanitize_component(pid)
        )
        shutil.rmtree(cache_dir, ignore_errors=True)
        for tp in TIMEPOINTS:
            source = source_paths[tp.name]
            if not existing_file(source):
                bug.append(f"{tp.name}_missing:{source}")
                continue
            try:
                source_paths[tp.name] = stage_source_file(
                    source,
                    cache_dir / f"{tp.name}.tif",
                )
            except Exception as exc:
                bug.append(
                    f"{tp.name}_cache_err:"
                    f"{type(exc).__name__}:{exc}"
                )
    try:
        for tp in TIMEPOINTS:
            image = read_chw_512(
                source_paths[tp.name],
                tp.name,
                bug,
            )
            if image is not None:
                images[tp.name] = image
    finally:
        if cache_dir is not None:
            shutil.rmtree(cache_dir, ignore_errors=True)
    if len(images) != len(TIMEPOINTS):
        return [], ";".join(bug + [rawvals])

    mask_roots = [Path(p) for p in args.mask_roots]
    mask, mask_path, mask_msg = read_or_build_mask_512(
        row,
        mask_roots,
        Path(args.mask_out_root),
        bool(args.build_mask512_from_cm),
        bool(args.mask_overwrite),
    )
    if mask is None:
        if str(args.mask_policy) == "legacy_zero_if_missing" and str(args.label_rule) == "legacy_center_anchor":
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask_path = ""
            mask_msg = "legacy_zero_if_missing"
        else:
            return [], ";".join(bug + [mask_msg, rawvals])
    if bool(args.forbid_mask512_fallback):
        mask_text = str(mask_path)
        if "_mask512" in mask_text or mask_msg in {"existing_built_mask"} or str(mask_msg).startswith("built_from:"):
            return [], ";".join(bug + ["forbidden_mask512_fallback", f"mask512={mask_path}", f"mask_msg={mask_msg}", rawvals])

    patch_size = int(args.patch_size)
    rng = random.Random(stable_seed(int(args.seed), split_name, pid))
    samples: list[PatchArrays] = []

    if str(args.label_rule) == "legacy_center_anchor":
        notebook_samples = notebook_cell7_samples(
            rng,
            patch_size,
            int(args.n_pos),
            int(args.n_neg),
            int(args.legacy_positive_center_size),
            int(args.legacy_center_box),
        )
        source_indices = {
            "notebook_cell7_positive": 0,
            "notebook_cell7_random": 0,
        }
        for label, x, y, source in notebook_samples:
            plume_crop = crop_hw(mask, x, y, patch_size)
            if plume_crop.shape != (patch_size, patch_size):
                bug.append(f"legacy_crop_bad_mask_shape:{plume_crop.shape}")
                continue
            crops = crop_all_timepoints(images, x, y, patch_size)
            if not sample_is_clean(crops, args):
                bug.append("legacy_quality_fail")
                continue
            index = source_indices[source]
            source_indices[source] += 1
            kind = "pos" if source == "notebook_cell7_positive" else "random"
            output_mask = (
                plume_crop
                if int(label) == 1
                else np.zeros((patch_size, patch_size), dtype=mask.dtype)
            )
            samples.append(
                PatchArrays(
                    kind,
                    index,
                    x,
                    y,
                    crops,
                    output_mask,
                    int(label),
                    source,
                )
            )

        expected = int(args.n_pos) + int(args.n_neg)
        if not samples:
            return [], ";".join(
                bug
                + [
                    f"legacy_no_valid_samples:0/{expected}",
                    rawvals,
                    f"mask512={mask_path}",
                    f"mask_msg={mask_msg}",
                ]
            )
        if len(samples) < expected:
            bug.append(f"legacy_partial:{len(samples)}/{expected}")
        records = [
            write_patch_sample(
                sample,
                row,
                Path(args.out_root_32),
                split_name,
                mask_path,
                bool(args.overwrite),
                str(args.output_cache_root),
            )
            for sample in samples
        ]
        return records, ";".join(bug)

    if str(args.label_rule) == "legacy_center_mask_gated":
        pos_written = 0
        pos_tries = 0
        while pos_written < int(args.n_pos) and pos_tries < int(args.max_tries_pos):
            pos_tries += 1
            x, y = sample_legacy_positive_xy(rng, patch_size)
            plume_crop = crop_hw(mask, x, y, patch_size)
            if (
                plume_crop.shape != (patch_size, patch_size)
                or float(plume_crop.sum()) < float(args.pos_mask_min_sum)
            ):
                continue
            crops = crop_all_timepoints(images, x, y, patch_size)
            if not sample_is_clean(crops, args):
                continue
            samples.append(
                PatchArrays(
                    "pos",
                    pos_written,
                    x,
                    y,
                    crops,
                    plume_crop,
                    1,
                    "legacy_center_mask_gated",
                )
            )
            pos_written += 1

        if pos_written < int(args.n_pos):
            bug.append(f"pos_insufficient:{pos_written}/{args.n_pos}")

        neg_written = 0
        neg_tries = 0
        while neg_written < int(args.n_neg) and neg_tries < int(args.max_tries_neg):
            neg_tries += 1
            x, y = sample_legacy_negative_xy(
                rng,
                patch_size,
                int(args.legacy_center_box),
            )
            plume_crop = crop_hw(mask, x, y, patch_size)
            if plume_crop.shape != (patch_size, patch_size):
                continue
            if float(plume_crop.sum()) > float(args.neg_mask_max_sum):
                continue
            crops = crop_all_timepoints(images, x, y, patch_size)
            if not sample_is_clean(crops, args):
                continue
            samples.append(
                PatchArrays(
                    "neg",
                    neg_written,
                    x,
                    y,
                    crops,
                    np.zeros((patch_size, patch_size), dtype=mask.dtype),
                    0,
                    "legacy_center_mask_gated",
                )
            )
            neg_written += 1

        if neg_written < int(args.n_neg):
            bug.append(f"neg_insufficient:{neg_written}/{args.n_neg}")

        expected = int(args.n_pos) + int(args.n_neg)
        if len(samples) < expected:
            return [], ";".join(
                bug
                + [
                    rawvals,
                    f"mask512={mask_path}",
                    f"mask_msg={mask_msg}",
                ]
            )
        records = [
            write_patch_sample(
                sample,
                row,
                Path(args.out_root_32),
                split_name,
                mask_path,
                bool(args.overwrite),
                str(args.output_cache_root),
            )
            for sample in samples
        ]
        return records, ""

    pos_written = 0
    pos_tries = 0
    seen_pos: set[tuple[int, int]] = set()
    while pos_written < int(args.n_pos) and pos_tries < int(args.max_tries_pos):
        pos_tries += 1
        try:
            x, y = sample_center_xy(rng, 512, 512, patch_size, int(args.center_box))
        except Exception as exc:
            return [], f"center_box_err:{exc};{rawvals}"
        if (x, y) in seen_pos:
            continue
        seen_pos.add((x, y))

        plume_crop = crop_hw(mask, x, y, patch_size)
        if plume_crop.shape != (patch_size, patch_size) or float(plume_crop.sum()) < float(args.pos_mask_min_sum):
            continue
        crops = crop_all_timepoints(images, x, y, patch_size)
        if not sample_is_clean(crops, args):
            continue
        samples.append(PatchArrays("pos", pos_written, x, y, crops, plume_crop, 1, "all6_t0_mask"))
        pos_written += 1

    if pos_written < int(args.n_pos):
        bug.append(f"pos_insufficient:{pos_written}/{args.n_pos}")

    neg_written = 0
    neg_tries = 0
    seen_neg: set[tuple[int, int]] = set()
    while neg_written < int(args.n_neg) and neg_tries < int(args.max_tries_neg):
        neg_tries += 1
        x, y = sample_center_xy(rng, 512, 512, patch_size, int(args.center_box))
        if (x, y) in seen_neg:
            continue
        seen_neg.add((x, y))

        plume_crop = crop_hw(mask, x, y, patch_size)
        if plume_crop.shape != (patch_size, patch_size):
            continue
        if bool(args.neg_require_mask_empty) and float(plume_crop.sum()) > float(args.neg_mask_max_sum):
            continue
        crops = crop_all_timepoints(images, x, y, patch_size)
        if not sample_is_clean(crops, args):
            continue
        zero_mask = np.zeros((patch_size, patch_size), dtype=mask.dtype)
        samples.append(PatchArrays("neg", neg_written, x, y, crops, zero_mask, 0, "all6_no_plume"))
        neg_written += 1

    if neg_written < int(args.n_neg):
        bug.append(f"neg_insufficient:{neg_written}/{args.n_neg}")

    expected = int(args.n_pos) + int(args.n_neg)
    if len(samples) < expected:
        return [], ";".join(bug + [rawvals, f"mask512={mask_path}", f"mask_msg={mask_msg}"])

    records = [
        write_patch_sample(
            sample,
            row,
            Path(args.out_root_32),
            split_name,
            mask_path,
            bool(args.overwrite),
            str(args.output_cache_root),
        )
        for sample in samples
    ]
    return records, ""


def required_columns_present(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [col for col in ["plume_id", *STD512_COLS] if col not in df.columns]
    if missing:
        raise RuntimeError(f"{csv_path} missing required columns: {missing}")


def format_split_path(template: str, split_name: str) -> Path:
    return Path(template.format(split=split_name))


def flush_records(records: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".part")
    pd.DataFrame(records).to_csv(tmp, index=False)
    tmp.replace(out_csv)


def done_plumes_from_manifest(records: list[dict[str, Any]], expected_per_plume: int) -> set[str]:
    if not records:
        return set()
    df = pd.DataFrame(records)
    if "plume_id" not in df.columns:
        return set()
    counts = df.groupby("plume_id").size()
    return set(counts[counts >= expected_per_plume].index.astype(str))


def run_crop_csv(input_csv: Path, split_name: str, out_csv: Path, args: argparse.Namespace) -> Path:
    df = pd.read_csv(input_csv, low_memory=False)
    required_columns_present(df, input_csv)
    if args.limit:
        df = df.head(int(args.limit)).copy()

    if args.only_complete:
        mask = df[STD512_COLS].apply(lambda row: all(existing_file(v) for v in row), axis=1)
        df = df[mask].copy()

    expected = int(args.n_pos) + int(args.n_neg)
    minimum_records = (
        1 if str(args.label_rule) == "legacy_center_anchor" else expected
    )
    out_records: list[dict[str, Any]] = []
    if args.resume and out_csv.exists():
        try:
            old = pd.read_csv(out_csv, low_memory=False)
            out_records = old.to_dict("records")
            debug(f"[{split_name}] resume rows={len(out_records)} from {out_csv}")
        except Exception as exc:
            debug(f"[{split_name}] resume ignored, failed to read {out_csv}: {exc}")

    done_plumes = done_plumes_from_manifest(out_records, minimum_records)
    rows = []
    for _, row in df.iterrows():
        pid = str(row.get("plume_id", "")).strip()
        if pid and pid in done_plumes:
            continue
        rows.append(row.to_dict())
    if args.limit:
        rows = rows[: int(args.limit)]

    debug(
        f"[{split_name}] crop32 input={input_csv} rows={len(df)} "
        f"to_process={len(rows)} done_plumes={len(done_plumes)} out={out_csv} timepoints={len(TIMEPOINTS)}"
    )

    ok = 0
    partial = 0
    fail = 0
    failures: list[dict[str, str]] = []
    issues: list[dict[str, str | int]] = []
    last_flush = time.time()

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(crop_one_plume, row, split_name, args): row.get("plume_id", "") for row in rows}
        for idx, fut in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc=f"{split_name}: crop32", ncols=120),
            start=1,
        ):
            pid = str(futures[fut])
            try:
                records, bug = fut.result()
            except Exception as exc:
                records = []
                bug = f"worker_exception:{type(exc).__name__}:{exc}\n{traceback.format_exc()}"

            if len(records) >= minimum_records:
                ok += 1
                out_records.extend(records)
                if len(records) < expected:
                    partial += 1
                    issues.append(
                        {
                            "plume_id": pid,
                            "records": len(records),
                            "expected": expected,
                            "bug": bug,
                        }
                    )
            else:
                fail += 1
                failures.append({"plume_id": pid, "bug": bug})
                if args.verbose_failures:
                    debug(f"[{split_name}][{pid}] FAIL {bug}")

            now = time.time()
            if idx % int(args.progress_every) == 0 or now - last_flush >= float(args.flush_every_sec):
                flush_records(out_records, out_csv)
                last_flush = now
                debug(
                    f"[{split_name}] crop32 progress={idx}/{len(futures)} "
                    f"ok={ok} partial={partial} fail={fail} samples={len(out_records)}"
                )

    flush_records(out_records, out_csv)
    if failures:
        fail_csv = out_csv.with_suffix(".failures.csv")
        pd.DataFrame(failures).to_csv(fail_csv, index=False)
        debug(f"[{split_name}] failures={len(failures)} wrote {fail_csv}")
    if issues:
        issue_csv = out_csv.with_suffix(".issues.csv")
        pd.DataFrame(issues).to_csv(issue_csv, index=False)
        debug(f"[{split_name}] partial={len(issues)} wrote {issue_csv}")
    debug(
        f"[{split_name}] crop32 done ok={ok} partial={partial} fail={fail} "
        f"total_samples={len(out_records)}"
    )
    return out_csv


def ensure_chw_for_resize(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"Expected 3D CHW/HWC, got {img.shape}")
    if img.shape[0] in (1, 3, 4, 12, 13):
        return img
    if img.shape[-1] in (1, 3, 4, 12, 13):
        return np.transpose(img, (2, 0, 1))
    if img.shape[0] <= 20 and img.shape[0] < img.shape[-1]:
        return img
    return np.transpose(img, (2, 0, 1))


def resize_bilinear_chw(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    img = ensure_chw_for_resize(img)
    _, h, w = img.shape
    if h == out_h and w == out_w:
        return img

    y = np.linspace(0, h - 1, out_h, dtype=np.float32)
    x = np.linspace(0, w - 1, out_w, dtype=np.float32)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (x - x0.astype(np.float32))[None, :]
    wy = (y - y0.astype(np.float32))[:, None]

    ia = img[:, y0[:, None], x0[None, :]]
    ib = img[:, y0[:, None], x1[None, :]]
    ic = img[:, y1[:, None], x0[None, :]]
    id_ = img[:, y1[:, None], x1[None, :]]
    out = ia * (1 - wx) * (1 - wy) + ib * wx * (1 - wy) + ic * (1 - wx) * wy + id_ * wx * wy
    return out.astype(img.dtype, copy=False)


def resized_path(src_path: str, old_root: Path, new_root: Path) -> str:
    src = Path(src_path)
    try:
        rel = src.relative_to(old_root)
    except ValueError:
        rel = Path(sanitize_component(src.name))
    return str(new_root / rel)


def is_resized_done(dst_path: str, out_size: int) -> bool:
    path = Path(dst_path)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
        if len(shape) == 2:
            return shape == (out_size, out_size)
        if len(shape) != 3:
            return False
        if shape[0] in (1, 3, 4, 12, 13) and shape[1:] == (out_size, out_size):
            return True
        if shape[-1] in (1, 3, 4, 12, 13) and shape[:2] == (out_size, out_size):
            return True
    except Exception:
        return False
    return False


def resize_one_tif(src_path: str, dst_path: str, out_size: int, overwrite: bool, mode: str = "image") -> tuple[str, str, str]:
    try:
        if not overwrite and is_resized_done(dst_path, out_size):
            return src_path, "skip", ""
        img = tifffile.imread(src_path)
        if mode == "mask" or img.ndim == 2:
            out = resize_hw_nearest(to_hw_mask(img), out_size, out_size).astype(np.float32, copy=False)
        else:
            out = resize_bilinear_chw(img, out_size, out_size)
        write_tif_atomic(Path(dst_path), out)
        return src_path, "ok", ""
    except Exception as exc:
        return src_path, "fail", f"{type(exc).__name__}:{exc}\n{traceback.format_exc()}"


def run_resize_tasks(tasks: list[tuple[str, str, str]], args: argparse.Namespace) -> None:
    total = len(tasks)
    if total == 0:
        return

    max_workers = max(1, int(args.resize_workers))
    timeout_sec = float(args.resize_task_timeout_sec)
    timeout_enabled = timeout_sec > 0
    bad = 0
    skipped = 0
    finished = 0
    submitted = 0
    pending: set[Any] = set()
    fut_to_src: dict[Any, str] = {}
    fut_start: dict[Any, float] = {}
    timeout_sources: list[str] = []
    task_iter = iter(tasks)

    def submit_next(pool: ThreadPoolExecutor) -> bool:
        nonlocal submitted
        try:
            src, dst, mode = next(task_iter)
        except StopIteration:
            return False
        fut = pool.submit(resize_one_tif, src, dst, int(args.target_size), bool(args.overwrite_224), mode)
        pending.add(fut)
        fut_to_src[fut] = src
        fut_start[fut] = time.time()
        submitted += 1
        return True

    def submit_until_full(pool: ThreadPoolExecutor) -> None:
        while len(pending) < max_workers and submitted < total:
            if not submit_next(pool):
                break

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        submit_until_full(pool)

        while pending:
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    src, status, info = fut.result()
                except Exception as exc:
                    src = fut_to_src.get(fut, "<unknown>")
                    status = "fail"
                    info = repr(exc)

                finished += 1
                if status == "skip":
                    skipped += 1
                elif status == "fail":
                    bad += 1
                    debug(f"[resize224] FAIL {src}\n{info}")

                fut_to_src.pop(fut, None)
                fut_start.pop(fut, None)
                if finished % int(args.progress_every) == 0 or finished == total:
                    debug(f"[resize224] done={finished}/{total} ok_or_skip={finished - bad} skip={skipped} bad={bad}")
                submit_until_full(pool)

            if not timeout_enabled:
                continue

            now = time.time()
            timed_out = [
                fut
                for fut in list(pending)
                if now - fut_start.get(fut, now) > timeout_sec
            ]
            for fut in timed_out:
                src = fut_to_src.get(fut, "<unknown>")
                if not fut.cancel():
                    continue
                timeout_sources.append(src)
                bad += 1
                finished += 1
                pending.remove(fut)
                fut_to_src.pop(fut, None)
                fut_start.pop(fut, None)
                debug(f"[resize224] TIMEOUT {src} >{timeout_sec}s")
                submit_until_full(pool)

        if submitted != total:
            raise RuntimeError(f"resize queue ended early: submitted={submitted} total={total}")

    if timeout_sources:
        timeout_path = Path(args.out_root_224) / "resize_timeouts.txt"
        timeout_path.parent.mkdir(parents=True, exist_ok=True)
        timeout_path.write_text("\n".join(timeout_sources) + "\n", encoding="utf-8")
        debug(f"[resize224] wrote timeout list {timeout_path}")
    debug(f"[resize224] done total={total} skip={skipped} bad={bad} timeouts={len(timeout_sources)}")

def set_legacy_aliases(df: pd.DataFrame) -> pd.DataFrame:
    for alias, path_col in LEGACY_ALIAS_COLS.items():
        if path_col in df.columns:
            df[alias] = df[path_col]
    return df


def resize_patch_csv(input_csv: Path, output_csv: Path, old_root: Path, new_root: Path, args: argparse.Namespace) -> Path:
    df = pd.read_csv(input_csv, low_memory=False)
    missing = [col for col in SIX_TIME_PATH_COLS if col not in df.columns]
    if missing:
        raise RuntimeError(f"{input_csv} missing six-time resize columns: {missing}")
    if "path_plume" not in df.columns:
        raise RuntimeError(f"{input_csv} missing plume resize column: path_plume")

    src_to_dst: dict[str, str] = {}
    for col in SIX_TIME_PATH_COLS:
        new_paths = []
        for src in df[col].astype(str).tolist():
            dst = resized_path(src, old_root, new_root)
            src_to_dst[src] = dst
            new_paths.append(dst)
        df[col] = new_paths
    plume_src_to_dst: dict[str, str] = {}
    plume_new_paths = []
    for src in df["path_plume"].astype(str).tolist():
        dst = resized_path(src, old_root, new_root)
        plume_src_to_dst[src] = dst
        plume_new_paths.append(dst)
    df["path_plume"] = plume_new_paths
    for col in ("plume_mask_path", "mask_path"):
        if col in df.columns:
            df[col] = df["path_plume"]
    df = set_legacy_aliases(df)

    debug(
        f"[resize224] prepare input={input_csv} rows={len(df)} "
        f"unique_images={len(src_to_dst)} unique_masks={len(plume_src_to_dst)}"
    )
    if args.trust_source_paths:
        image_tasks = [(src, dst, "image") for src, dst in sorted(src_to_dst.items())]
        mask_tasks = [(src, dst, "mask") for src, dst in sorted(plume_src_to_dst.items())]
    else:
        image_tasks = [(src, dst, "image") for src, dst in sorted(src_to_dst.items()) if existing_file(src)]
        mask_tasks = [(src, dst, "mask") for src, dst in sorted(plume_src_to_dst.items()) if existing_file(src)]
    tasks = image_tasks + mask_tasks
    debug(
        f"[resize224] input={input_csv} images={len(image_tasks)} masks={len(mask_tasks)} "
        f"timepoints={len(TIMEPOINTS)} out_root={new_root}"
    )
    run_resize_tasks(tasks, args)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    debug(f"[resize224] wrote {output_csv} rows={len(df)}")
    return output_csv


def output_csv_for_split(template: str, default_root: Path, split_name: str, is_224: bool) -> Path:
    if template:
        return format_split_path(template, split_name)
    suffix = "224.csv" if is_224 else "32.csv"
    return default_root / f"{split_name}_patches_{suffix}"


def run_pipeline(args: argparse.Namespace) -> int:
    out_root_32 = Path(args.out_root_32)
    out_root_224 = Path(args.out_root_224) if args.out_root_224 else DEFAULT_OUT_ROOT_224
    args.out_root_32 = str(out_root_32)
    args.out_root_224 = str(out_root_224)
    if not args.mask_out_root:
        args.mask_out_root = str(out_root_32 / "_mask512")

    if args.only_resize:
        patch_csv_value = args.patch_csv_32 or args.out_csv_32
        if not patch_csv_value:
            raise RuntimeError("--only-resize requires --patch-csv-32 or --out-csv-32")
        patch_csv = Path(patch_csv_value)
        out_csv_224 = Path(args.out_csv_224) if args.out_csv_224 else out_root_224 / f"{patch_csv.stem}_224.csv"
        resize_patch_csv(patch_csv, out_csv_224, out_root_32, out_root_224, args)
        return 0

    split_inputs: list[tuple[str, Path]] = []
    if args.train_csv or args.test_csv:
        if not args.train_csv or not args.test_csv:
            raise RuntimeError("Provide both --train-csv and --test-csv, or use --input-csv.")
        split_inputs = [("train", Path(args.train_csv)), ("test", Path(args.test_csv))]
    else:
        split_inputs = [(str(args.split_name), Path(args.input_csv))]

    for split_name, input_csv in split_inputs:
        out_csv_32 = output_csv_for_split(str(args.out_csv_32), out_root_32, split_name, is_224=False)
        crop_csv = run_crop_csv(input_csv, split_name, out_csv_32, args)
        if args.resize_to_224:
            out_csv_224 = output_csv_for_split(str(args.out_csv_224), out_root_224, split_name, is_224=True)
            resize_patch_csv(crop_csv, out_csv_224, out_root_32, out_root_224, args)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build six-time S2 32x32 patches and resize six S2 images per sample to 224."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--train-csv", default="")
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--split-name", default="all")
    parser.add_argument("--out-root-32", default=str(DEFAULT_OUT_ROOT_32))
    parser.add_argument("--out-root-224", default=str(DEFAULT_OUT_ROOT_224))
    parser.add_argument("--out-csv-32", default="", help="May include {split} when using train/test.")
    parser.add_argument("--out-csv-224", default="", help="May include {split} when using train/test.")
    parser.add_argument("--patch-csv-32", default="", help="Existing 32 manifest for --only-resize.")
    parser.add_argument("--only-resize", action="store_true")
    parser.add_argument("--resize-to-224", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--trust-source-paths",
        action="store_true",
        help="Build resize tasks from the CSV without a slow pre-resize exists() scan over source TIFFs.",
    )

    parser.add_argument("--mask-roots", nargs="+", default=[str(p) for p in DEFAULT_MASK_ROOTS])
    parser.add_argument("--mask-out-root", default="")
    parser.add_argument(
        "--build-mask512-from-cm",
        action="store_true",
        help="If no aligned 512 mask exists, build one by binarizing and nearest-resizing CM plume.tif.",
    )
    parser.add_argument("--mask-overwrite", action="store_true")
    parser.add_argument(
        "--forbid-mask512-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject masks created under _mask512 or built from raw CM resize fallback.",
    )
    parser.add_argument(
        "--mask-policy",
        choices=["require_aligned", "legacy_zero_if_missing"],
        default="require_aligned",
        help=(
            "require_aligned drops rows without an existing aligned mask. "
            "legacy_zero_if_missing matches legacy crop_legacy_param.py: if no aligned mask exists, "
            "use an all-zero plume mask while labels still come from center-anchor sampling."
        ),
    )

    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--center-box", type=int, default=256)
    parser.add_argument(
        "--label-rule",
        choices=[
            "mask_gated",
            "legacy_center_anchor",
            "legacy_center_mask_gated",
        ],
        default="mask_gated",
        help=(
            "legacy_center_anchor copies cell 7 of "
            "preprocess_dataset_s2/carbon_mapper_sentinel2_plume_-7_download.ipynb. "
            "legacy_center_mask_gated keeps those offsets while requiring non-empty positive "
            "masks and empty negative masks."
        ),
    )
    parser.add_argument("--legacy-center-box", type=int, default=10)
    parser.add_argument("--legacy-positive-center-size", type=int, default=20)
    parser.add_argument("--n-pos", type=int, default=16)
    parser.add_argument("--n-neg", type=int, default=16)
    parser.add_argument("--max-tries-pos", type=int, default=800)
    parser.add_argument("--max-tries-neg", type=int, default=800)
    parser.add_argument("--pos-mask-min-sum", type=float, default=1.0)
    parser.add_argument("--neg-mask-max-sum", type=float, default=0.0)
    parser.add_argument("--neg-require-mask-empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--band-index", type=int, default=11)
    parser.add_argument("--zero-ratio-thresh", type=float, default=0.20)
    parser.add_argument(
        "--quality-timepoints",
        default="t0,seasonal,year",
        help="Comma-separated timepoints used by the band-zero quality filter.",
    )

    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--source-cache-root",
        default="",
        help=(
            "Optional local scratch root. Each plume's six 512 TIFFs "
            "are staged here before decoding and removed immediately."
        ),
    )
    parser.add_argument(
        "--output-cache-root",
        default="",
        help=(
            "Optional local staging root. Each 32-pixel TIFF is written "
            "locally, atomically pushed to the output filesystem, then removed."
        ),
    )
    parser.add_argument("--resize-workers", type=int, default=4)
    parser.add_argument(
        "--resize-task-timeout-sec",
        type=float,
        default=0.0,
        help="Per submitted resize task timeout. 0 disables timeout; recommended for large queued runs.",
    )
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--flush-every-sec", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite 32 patch TIFFs.")
    parser.add_argument("--overwrite-224", action="store_true", help="Overwrite resized 224 TIFFs.")
    parser.add_argument("--only-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose-failures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
