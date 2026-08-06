#!/usr/bin/env python3
"""Build S2-aligned 512x512 plume masks for the six-time S2 table.

This intentionally reuses the old query pipeline's mask rule:
``preprocess_dataset_query_multi/prepare_raw512_masks.py`` reprojects the raw
Carbon Mapper plume mask onto a georeferenced sensor reference with rasterio.

It does not implement the old temporary ``_mask512`` fallback. If no existing
aligned mask or georeferenced S2 reference is available, the row is marked as
failed so it cannot silently enter a formal crop manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio import windows
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject


REPO_ROOT = Path("/home/yuyao/methane_train")
DEFAULT_INPUT_CSV = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_all6_available_paths_std512_complete.csv"
DEFAULT_GEE_LOCAL_CSV = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_gee_local_paths.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_all6_available_paths_std512_complete_with_s2_masks.csv"
DEFAULT_QA_CSV = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_aligned_mask_build_qa.csv"
DEFAULT_MASK_ROOT = Path("/mnt/engg-niulab/yuyao/preprocessed_512/S2_masks_aligned")
DEFAULT_CM_ROOT = Path("/mnt/engg-niulab/yuyao/sensors_raw_data/CM")

EXISTING_MASK_ROOTS = [
    Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_masks_wv3_512"),
    Path("/data2/yuyao/methane_emission/carbon_mapper_data_masks"),
]
EXISTING_MASK_NAMES = [
    "s2_mask_512.tif",
    "resized_512x512.tif",
    "mask_512.tif",
    "mask_60m_512.tif",
]
MASK_COLUMNS = [
    "s2_mask_512_path",
    "s2_plume_mask_512_path",
    "resized_512x512_path",
    "mask_path_512",
    "mask_512_path",
]
GEE_PATH_COLUMNS = [
    "gee_t0_raw_path",
    "gee_prev1_raw_path",
    "gee_prev2_raw_path",
    "gee_prev3_raw_path",
    "gee_seasonal_raw_path",
    "gee_year_raw_path",
]
WINDOW_SIZE = 512
STD512_ALIASES = {
    "s2_0_std_512": "t0_raw_path",
    "s2_-7_std_512": "prev1_raw_path",
    "s2_prev2_std_512": "prev2_raw_path",
    "s2_prev3_std_512": "prev3_raw_path",
    "s2_-90_std_512": "seasonal_raw_path",
    "s2_-360_std_512": "year_raw_path",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def existing_file(value: Any) -> Path | None:
    text = clean(value)
    if not text:
        return None
    path = Path(text)
    try:
        if path.exists() and path.stat().st_size > 0:
            return path
    except OSError:
        return None
    return None


def binary_array(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    return ((arr > 0) & finite).astype(np.uint8)


def write_binary_tif(path: Path, mask: np.ndarray, profile_ref: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if profile_ref is None:
        profile: dict[str, Any] = {
            "driver": "GTiff",
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
            "count": 1,
            "dtype": "uint8",
            "compress": "deflate",
            "tiled": True,
            "blockxsize": min(256, int(mask.shape[1])),
            "blockysize": min(256, int(mask.shape[0])),
            "BIGTIFF": "IF_SAFER",
        }
    else:
        profile = dict(profile_ref)
        profile.update(
            driver="GTiff",
            height=int(mask.shape[0]),
            width=int(mask.shape[1]),
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            tiled=True,
            blockxsize=min(256, int(mask.shape[1])),
            blockysize=min(256, int(mask.shape[0])),
            BIGTIFF="IF_SAFER",
        )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def read_mask_512(path: Path) -> np.ndarray | None:
    try:
        with rasterio.open(path) as ds:
            arr = ds.read(1)
        mask = binary_array(arr)
        if mask.shape == (WINDOW_SIZE, WINDOW_SIZE):
            return mask
    except Exception:
        return None
    return None


def sanitize(value: Any) -> str:
    text = clean(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text) or "missing"


def fallback_event_group_id(plume_id: str) -> str:
    parts = plume_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1] and len(parts[1]) <= 4:
        return parts[0]
    return plume_id


def center_window_for_dataset(ds: rasterio.io.DatasetReader) -> windows.Window:
    if ds.width < WINDOW_SIZE or ds.height < WINDOW_SIZE:
        return windows.Window(0, 0, ds.width, ds.height)
    col0 = (ds.width - WINDOW_SIZE) // 2
    row0 = (ds.height - WINDOW_SIZE) // 2
    return windows.Window(col0, row0, WINDOW_SIZE, WINDOW_SIZE)


def has_real_georef(ds: rasterio.io.DatasetReader) -> bool:
    if ds.crs is None:
        return False
    identity = Affine.identity()
    try:
        if ds.transform == identity:
            return False
    except Exception:
        pass
    return True


def reference_profile_512(ref_path: Path) -> tuple[dict[str, Any], tuple[int, int]] | None:
    with rasterio.open(ref_path) as ref:
        if not has_real_georef(ref):
            return None
        win = center_window_for_dataset(ref)
        out_h = int(min(WINDOW_SIZE, win.height))
        out_w = int(min(WINDOW_SIZE, win.width))
        transform = windows.transform(win, ref.transform)
        profile = ref.profile.copy()
        profile.update(
            height=out_h,
            width=out_w,
            count=1,
            dtype="uint8",
            transform=transform,
            crs=ref.crs,
        )
    return profile, (out_h, out_w)


def pad_center(mask: np.ndarray, profile: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = mask.shape
    if h == WINDOW_SIZE and w == WINDOW_SIZE:
        return mask, profile
    out = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=mask.dtype)
    y0 = max(0, (WINDOW_SIZE - h) // 2)
    x0 = max(0, (WINDOW_SIZE - w) // 2)
    out[y0 : y0 + h, x0 : x0 + w] = mask
    # For padded rare cases, keep a valid profile but do not pretend the padded
    # pixels have a precise geotransform. Downstream treats this as an ML mask.
    profile = dict(profile)
    profile.update(height=WINDOW_SIZE, width=WINDOW_SIZE)
    return out, profile


def reproject_raw_plume_to_ref(src_mask: Path, ref_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    ref_info = reference_profile_512(ref_path)
    if ref_info is None:
        raise RuntimeError("reference_has_no_real_georef")
    profile, dst_shape = ref_info

    with rasterio.open(src_mask) as src:
        if src.crs is None:
            raise RuntimeError("source_mask_has_no_crs")
        src_arr = src.read(1).astype(np.float32)
        dst = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=Resampling.max,
        )
    mask = binary_array(dst)
    mask, profile = pad_center(mask, profile)
    return mask, profile


def georef_sidecar_for_t0(row: dict[str, Any]) -> Path | None:
    t0_path = existing_file(row.get("t0_raw_path", ""))
    if t0_path is None:
        return None
    sidecar = t0_path.with_name(t0_path.name + ".georef.json")
    if sidecar.is_file() and sidecar.stat().st_size > 0:
        return sidecar
    return None


def reproject_raw_plume_to_sidecar(
    src_mask: Path,
    sidecar_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = json.loads(sidecar_path.read_text())
    transform_values = metadata.get("transform", [])
    crs_wkt = clean(metadata.get("crs_wkt", ""))
    height = int(metadata.get("height", WINDOW_SIZE))
    width = int(metadata.get("width", WINDOW_SIZE))
    if len(transform_values) < 6 or not crs_wkt:
        raise RuntimeError("invalid_georef_sidecar")
    if (height, width) != (WINDOW_SIZE, WINDOW_SIZE):
        raise RuntimeError(f"sidecar_not512:{height}x{width}")
    dst_transform = Affine(*[float(value) for value in transform_values[:6]])
    dst_crs = CRS.from_wkt(crs_wkt)
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "transform": dst_transform,
        "crs": dst_crs,
    }

    with rasterio.open(src_mask) as src:
        if src.crs is None:
            raise RuntimeError("source_mask_has_no_crs")
        src_arr = src.read(1).astype(np.float32)
        dst = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.max,
        )
    return binary_array(dst), profile


def existing_aligned_mask(row: dict[str, Any], plume_id: str, roots: list[Path]) -> tuple[Path | None, str]:
    for col in MASK_COLUMNS:
        path = existing_file(row.get(col, ""))
        if path is None:
            continue
        if "_mask512" in str(path):
            continue
        if read_mask_512(path) is not None:
            return path, f"column:{col}"

    for root in roots:
        for name in EXISTING_MASK_NAMES:
            path = root / plume_id / name
            if "_mask512" in str(path):
                continue
            if path.exists() and path.stat().st_size > 0 and read_mask_512(path) is not None:
                return path, f"root:{root.name}/{name}"
    return None, ""


def georef_reference(row: dict[str, Any]) -> tuple[Path | None, str]:
    for col in GEE_PATH_COLUMNS:
        path = existing_file(row.get(col, ""))
        if path is None:
            continue
        try:
            with rasterio.open(path) as ds:
                if has_real_georef(ds):
                    return path, col
        except Exception:
            continue
    return None, ""


def process_row(row: dict[str, Any], args: argparse.Namespace, roots: list[Path]) -> dict[str, Any]:
    plume_id = clean(row.get("plume_id", ""))
    out_path = Path(args.out_root) / sanitize(plume_id) / "s2_mask_512.tif"
    qa: dict[str, Any] = {
        "plume_id": plume_id,
        "status": "fail",
        "reason": "",
        "source_mask": "",
        "reference": "",
        "reference_kind": "",
        "out_mask": str(out_path),
        "positive_pixels": "",
    }
    if not plume_id:
        qa["reason"] = "missing_plume_id"
        return qa
    if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
        mask = read_mask_512(out_path)
        if mask is not None:
            qa.update(status="ok", reason="exists", positive_pixels=int(mask.sum()))
            return qa

    existing, existing_kind = existing_aligned_mask(row, plume_id, roots)
    if existing is not None:
        mask = read_mask_512(existing)
        if mask is None:
            qa["reason"] = "existing_mask_read_failed"
            return qa
        write_binary_tif(out_path, mask)
        qa.update(
            status="ok",
            reason="copied_existing_aligned_mask",
            source_mask=str(existing),
            reference_kind=existing_kind,
            positive_pixels=int(mask.sum()),
        )
        return qa

    src_mask = Path(args.cm_root) / plume_id / "plume.tif"
    if not (src_mask.exists() and src_mask.stat().st_size > 0):
        qa["reason"] = "missing_raw_cm_plume"
        return qa

    sidecar = georef_sidecar_for_t0(row)
    ref, ref_kind = georef_reference(row)
    if sidecar is None and ref is None:
        qa["reason"] = "missing_georef_s2_reference"
        qa["source_mask"] = str(src_mask)
        return qa

    try:
        if sidecar is not None:
            mask, profile = reproject_raw_plume_to_sidecar(src_mask, sidecar)
            reference = sidecar
            reference_kind = "t0_georef_sidecar"
        else:
            mask, profile = reproject_raw_plume_to_ref(src_mask, ref)
            reference = ref
            reference_kind = ref_kind
        write_binary_tif(out_path, mask, profile_ref=profile)
        qa.update(
            status="ok",
            reason="reprojected_raw_plume",
            source_mask=str(src_mask),
            reference=str(reference),
            reference_kind=reference_kind,
            positive_pixels=int(mask.sum()),
        )
        return qa
    except Exception as exc:
        qa.update(
            reason=f"reproject_failed:{type(exc).__name__}:{str(exc)[:220]}",
            source_mask=str(src_mask),
            reference=str(sidecar or ref),
            reference_kind="t0_georef_sidecar" if sidecar is not None else ref_kind,
        )
        return qa


def merge_gee_columns(df: pd.DataFrame, gee_csv: Path) -> pd.DataFrame:
    if not gee_csv.exists():
        return df
    gee_cols = ["plume_id", *GEE_PATH_COLUMNS]
    gee = pd.read_csv(gee_csv, usecols=lambda c: c in set(gee_cols), low_memory=False)
    keep = [c for c in gee_cols if c in gee.columns]
    gee = gee[keep].drop_duplicates("plume_id")
    overlap = [c for c in keep if c != "plume_id" and c in df.columns]
    if overlap:
        df = df.drop(columns=overlap)
    return df.merge(gee, on="plume_id", how="left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build formal S2-aligned 512 plume masks.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--gee-local-csv", default=str(DEFAULT_GEE_LOCAL_CSV))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--qa-csv", default=str(DEFAULT_QA_CSV))
    parser.add_argument("--out-root", default=str(DEFAULT_MASK_ROOT))
    parser.add_argument("--cm-root", default=str(DEFAULT_CM_ROOT))
    parser.add_argument("--existing-mask-roots", nargs="+", default=[str(p) for p in EXISTING_MASK_ROOTS])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input_csv, low_memory=False)
    df = merge_gee_columns(df, Path(args.gee_local_csv))
    if args.limit:
        df = df.head(int(args.limit)).copy()

    roots = [Path(p) for p in args.existing_mask_roots]
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    rows = df.to_dict("records")
    print(
        f"[mask512] input={args.input_csv} rows={len(rows)} out_root={args.out_root} "
        f"gee_csv={args.gee_local_csv}",
        flush=True,
    )

    qa_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for i, qa in enumerate(pool.map(lambda r: process_row(r, args, roots), rows), start=1):
            qa_rows.append(qa)
            if i % 250 == 0 or i == len(rows):
                print(f"[mask512] progress={i}/{len(rows)}", flush=True)

    qa_df = pd.DataFrame(qa_rows)
    ok_map = {
        str(r["plume_id"]): str(r["out_mask"])
        for r in qa_rows
        if r.get("status") == "ok" and Path(str(r.get("out_mask", ""))).exists()
    }
    df["s2_mask_512_path"] = df["plume_id"].astype(str).map(ok_map).fillna("")
    df["has_s2_mask_512"] = df["s2_mask_512_path"].astype(str).str.len() > 0
    if "event_group_id" not in df.columns:
        df["event_group_id"] = df["plume_id"].astype(str).map(
            fallback_event_group_id
        )
    else:
        missing_group = df["event_group_id"].map(clean).eq("")
        df.loc[missing_group, "event_group_id"] = (
            df.loc[missing_group, "plume_id"].astype(str).map(
                fallback_event_group_id
            )
        )
    for alias, raw_column in STD512_ALIASES.items():
        if raw_column not in df.columns:
            raise RuntimeError(f"input table is missing {raw_column}")
        df[alias] = df[raw_column]
    df["has_all6_512"] = df[list(STD512_ALIASES)].apply(
        lambda row: all(existing_file(value) is not None for value in row),
        axis=1,
    )

    out_csv = Path(args.out_csv)
    qa_csv = Path(args.qa_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    qa_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    tmp.replace(out_csv)
    qa_df.to_csv(qa_csv, index=False)

    counts = Counter(qa_df["reason"].astype(str))
    print(f"[mask512] wrote {out_csv}", flush=True)
    print(f"[mask512] wrote {qa_csv}", flush=True)
    print(f"[mask512] ok={int((qa_df['status'] == 'ok').sum())} fail={int((qa_df['status'] != 'ok').sum())}", flush=True)
    for key, value in sorted(counts.items()):
        print(f"[mask512] reason {key}: {value}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
