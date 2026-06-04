#!/usr/bin/env python3
"""Prepare HF CSV manifests and a source-to-archive file manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "hf_dataset_release"
TABLE_DIR = OUT / "tables"

QUERY_TABLES = {
    "120m": ROOT / "manifest_time_test_120m_actual_overlap_all_samples_with_pred_metrics.csv",
    "360m": ROOT / "manifest_time_test_360m_actual_overlap_all_samples_with_pred_metrics.csv",
    "480m": ROOT / "manifest_time_test_480m_actual_overlap_all_samples_with_pred_metrics.csv",
    "960m": ROOT / "manifest_time_test_960m_actual_overlap_all_samples_with_pred_metrics.csv",
}

ORIGINAL_GEO = {
    "train": ROOT / "manifest_multisensor_crop_scheme2_train_geo_resplit.csv",
    "test": ROOT / "manifest_multisensor_crop_scheme2_test_geo_resplit.csv",
}

ORIGINAL_PROVIDED_SOURCES = {
    "l89": {
        "train": Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/data_dir_l89_L2SR/l89_temporal_16_resized_to_224_CRSfixed/train_2025_balanced.csv"),
        "test": Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/data_dir_l89_L2SR/l89_temporal_16_resized_to_224_CRSfixed/test_filtered_2025.csv"),
    },
    "s2": {
        "train": Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2025_16/train.csv"),
        "test": Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2025_16/test.csv"),
    },
    "s5p": {
        "train": Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s5p_patches_3x3_to_224_offl_triplet/train.csv"),
        "test": Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s5p_patches_3x3_to_224_offl_triplet/test.csv"),
    },
    "emit": {
        "train": Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/emit_wv3_temporal_-90_-180_16_to_224/train_balanced.csv"),
        "test": Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/emit_wv3_temporal_-90_-180_16_to_224/test_balanced.csv"),
    },
}

FINAL_COLS = [
    "id",
    "label",
    "latitude",
    "longitude",
    "available_sensor",
    "S2_t0_path",
    "S2_pre_path",
    "S2_pre_pre_path",
    "S2_plume_label_path",
    "L89_t0_path",
    "L89_pre_path",
    "L89_pre_pre_path",
    "L89_plume_label_path",
    "EMIT_t0_path",
    "EMIT_pre_path",
    "EMIT_pre_pre_path",
    "EMIT_plume_label_path",
    "S5p_temporal_path",
]

SENSOR_FILES = {
    "S2": {
        "t0": ("S2_t0_path", "s2.tif"),
        "pre": ("S2_pre_path", "s2_pre.tif"),
        "pre_pre": ("S2_pre_pre_path", "s2_pre_pre.tif"),
        "plume": ("S2_plume_label_path", "s2_plume.tif"),
    },
    "L89": {
        "t0": ("L89_t0_path", "l89.tif"),
        "pre": ("L89_pre_path", "l89_pre.tif"),
        "pre_pre": ("L89_pre_pre_path", "l89_pre_pre.tif"),
        "plume": ("L89_plume_label_path", "l89_plume.tif"),
    },
    "EMIT": {
        "t0": ("EMIT_t0_path", "emit.tif"),
        "pre": ("EMIT_pre_path", "emit_pre.tif"),
        "pre_pre": ("EMIT_pre_pre_path", "emit_pre_pre.tif"),
        "plume": ("EMIT_plume_label_path", "emit_plume.tif"),
    },
    "S5p": {
        "temporal": ("S5p_temporal_path", "s5p_temporal.npz"),
    },
}


def plume_hash_split(plume_id: object, test_frac: float = 0.2) -> str:
    key = "" if pd.isna(plume_id) else str(plume_id)
    value = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if value < test_frac else "train"


def clean_source(value: object) -> str:
    if pd.isna(value) or value == "":
        return ""
    return str(value)


def sample_key(value: object) -> str:
    if pd.isna(value) or value == "":
        return "missing"
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("/", "_").replace(" ", "_")


def init_output(df: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["id"] = df[id_col].map(sample_key) if id_col in df.columns else [str(i) for i in range(len(df))]
    out["label"] = df["label"].astype(int)
    out["latitude"] = df.get("latitude", df.get("lat", df.get("plume_latitude", "")))
    out["longitude"] = df.get("longitude", df.get("lon", df.get("plume_longitude", "")))
    out["available_sensor"] = ""
    for col in FINAL_COLS[5:]:
        out[col] = ""
    out["_prefix"] = prefix
    return out


def target_path(prefix: str, sensor_dir: str, row_id: object, filename: str) -> str:
    return f"data/{prefix}/{sensor_dir}/{sample_key(row_id)}/{filename}"


def add_path(
    out: pd.DataFrame,
    row_idx: int,
    source: object,
    prefix: str,
    sensor_dir: str,
    row_id: object,
    final_col: str,
    filename: str,
    file_rows: list[dict[str, str]],
    errors: list[dict[str, str]],
    require_npz: bool = False,
) -> None:
    src = clean_source(source)
    if not src:
        return
    if require_npz and Path(src).suffix.lower() != ".npz":
        errors.append({"id": sample_key(row_id), "column": final_col, "source_path": src, "error": "S5p_temporal_path must be .npz"})
        return
    tgt = target_path(prefix, sensor_dir, row_id, filename)
    out.at[row_idx, final_col] = tgt
    file_rows.append({"source_path": src, "target_path": tgt})


def finalize_available_sensor(out: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = {
        "S2": "S2_t0_path",
        "L89": "L89_t0_path",
        "EMIT": "EMIT_t0_path",
        "S5p": "S5p_temporal_path",
    }
    values = []
    for _, row in out.iterrows():
        sensors = [name for name, col in sensor_cols.items() if str(row[col])]
        values.append(",".join(sensors))
    out["available_sensor"] = values
    return out[FINAL_COLS]


def convert_multisensor_manifest(df: pd.DataFrame, prefix: str, file_rows: list[dict[str, str]], errors: list[dict[str, str]]) -> pd.DataFrame:
    if "id" in df.columns and "sample_id" not in df.columns:
        df = df.rename(columns={"id": "sample_id"})
    out = init_output(df, "sample_id", prefix)
    mapping = [
        ("S2", "s2", "s2_0_path", "t0"),
        ("S2", "s2", "s2_90_path", "pre"),
        ("S2", "s2", "s2_360_path", "pre_pre"),
        ("S2", "s2", "s2_plume_path", "plume"),
        ("L89", "l89", "l89_0_path", "t0"),
        ("L89", "l89", "l89_90_path", "pre"),
        ("L89", "l89", "l89_360_path", "pre_pre"),
        ("L89", "l89", "l89_plume_path", "plume"),
        ("EMIT", "emit", "emit_0_path", "t0"),
        ("EMIT", "emit", "emit_90_path", "pre"),
        ("EMIT", "emit", "emit_360_path", "pre_pre"),
        ("EMIT", "emit", "emit_plume_path", "plume"),
        ("S5p", "s5p", "s5p_0_path", "temporal"),
    ]
    for sensor, sensor_dir, src_col, role in mapping:
        if src_col not in df.columns:
            continue
        final_col, filename = SENSOR_FILES[sensor][role]
        for idx, src in df[src_col].items():
            add_path(out, idx, src, prefix, sensor_dir, df.at[idx, "sample_id"], final_col, filename, file_rows, errors, require_npz=(sensor == "S5p"))
    return finalize_available_sensor(out)


def convert_provided_source(df: pd.DataFrame, sensor: str, prefix: str, file_rows: list[dict[str, str]], errors: list[dict[str, str]]) -> pd.DataFrame:
    sensor_upper = {"s2": "S2", "l89": "L89", "emit": "EMIT", "s5p": "S5p"}[sensor]
    id_col = "sample_id" if "sample_id" in df.columns else "id"
    if id_col not in df.columns:
        df = df.copy()
        df["id"] = range(len(df))
        id_col = "id"
    out = init_output(df, id_col, prefix)
    if sensor == "s2":
        source_cols = {"t0": "image_path", "pre": "s2_pre_path", "pre_pre": "s2_pre_pre_path", "plume": "plume_mask_path"}
    elif sensor == "l89":
        source_cols = {"t0": "path_t0", "pre": "path_t90", "pre_pre": "path_t360"}
    elif sensor == "emit":
        source_cols = {"t0": "path_t0", "pre": "path_t90", "pre_pre": "path_t360", "plume": "mask_path"}
    else:
        source_cols = {"temporal": "path_t0"}
    for role, src_col in source_cols.items():
        if src_col not in df.columns:
            continue
        final_col, filename = SENSOR_FILES[sensor_upper][role]
        for idx, src in df[src_col].items():
            add_path(out, idx, src, prefix, sensor, df.at[idx, id_col], final_col, filename, file_rows, errors, require_npz=(sensor == "s5p"))
    return finalize_available_sensor(out)


def write_split_tables(df: pd.DataFrame, out_dir: Path) -> dict[str, dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}
    for split in ["train", "test"]:
        part = df[df["_split"] == split].drop(columns=["_split"], errors="ignore")
        part.to_csv(out_dir / f"{split}.csv", index=False)
        stats[split] = {
            "rows": int(len(part)),
            "positive": int((part["label"] == 1).sum()),
            "negative": int((part["label"] == 0).sum()),
        }
    return stats


def with_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    df["_split"] = split
    return df


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    file_rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    summary: dict[str, object] = {"tables": {}, "notes": []}

    # Original provided split from mounted source CSVs.
    provided_parts = []
    for sensor, split_paths in ORIGINAL_PROVIDED_SOURCES.items():
        for split, path in split_paths.items():
            raw = pd.read_csv(path, low_memory=False)
            table = convert_provided_source(raw, sensor, f"original/provided_split/{split}", file_rows, errors)
            provided_parts.append(with_split(table, split))
    original_provided = pd.concat(provided_parts, ignore_index=True)
    summary["tables"]["original_provided_split"] = write_split_tables(original_provided, TABLE_DIR / "original" / "provided_split")

    # Original geo split from multisensor manifests.
    geo_parts = []
    for split, path in ORIGINAL_GEO.items():
        raw = pd.read_csv(path, low_memory=False)
        table = convert_multisensor_manifest(raw, f"original/geo/{split}", file_rows, errors)
        geo_parts.append(with_split(table, split))
    original_geo = pd.concat(geo_parts, ignore_index=True)
    summary["tables"]["original_geo"] = write_split_tables(original_geo, TABLE_DIR / "original" / "geo")

    # Original plume hash split derived from original geo rows.
    raw_geo = pd.concat([pd.read_csv(path, low_memory=False) for path in ORIGINAL_GEO.values()], ignore_index=True)
    if "id" in raw_geo.columns and "sample_id" not in raw_geo.columns:
        raw_geo = raw_geo.rename(columns={"id": "sample_id"})
    hash_split = raw_geo["plume_id"].map(plume_hash_split) if "plume_id" in raw_geo.columns else pd.Series(["train"] * len(raw_geo))
    original_hash = convert_multisensor_manifest(raw_geo, "original/plume_hash", file_rows, errors)
    original_hash = original_hash.assign(_split=hash_split.values)
    summary["tables"]["original_plume_hash"] = write_split_tables(original_hash, TABLE_DIR / "original" / "plume_hash")

    # Query actual-overlap splits.
    query_raw_parts = []
    for scale, path in QUERY_TABLES.items():
        raw = pd.read_csv(path, low_memory=False)
        raw["_scale"] = scale
        query_raw_parts.append(raw)
    query_raw = pd.concat(query_raw_parts, ignore_index=True)
    if "id" in query_raw.columns and "sample_id" not in query_raw.columns:
        query_raw = query_raw.rename(columns={"id": "sample_id"})

    query_hash_split = query_raw["plume_id"].map(plume_hash_split)
    query_hash = convert_multisensor_manifest(query_raw, "query/actual_overlap/plume_hash", file_rows, errors)
    query_hash = query_hash.assign(_split=query_hash_split.values)
    summary["tables"]["query_actual_overlap_plume_hash"] = write_split_tables(query_hash, TABLE_DIR / "query" / "actual_overlap" / "plume_hash")

    train_geo = pd.read_csv(ORIGINAL_GEO["train"], usecols=["plume_id"], low_memory=False)
    test_geo = pd.read_csv(ORIGINAL_GEO["test"], usecols=["plume_id"], low_memory=False)
    train_plumes = set(train_geo["plume_id"].dropna().astype(str))
    test_plumes = set(test_geo["plume_id"].dropna().astype(str))
    def geo_split(pid: object) -> str:
        key = "" if pd.isna(pid) else str(pid)
        if key in train_plumes:
            return "train"
        if key in test_plumes:
            return "test"
        return plume_hash_split(key)
    query_geo_split = query_raw["plume_id"].map(geo_split)
    query_geo = convert_multisensor_manifest(query_raw, "query/actual_overlap/geo_plume", file_rows, errors)
    query_geo = query_geo.assign(_split=query_geo_split.values)
    summary["tables"]["query_actual_overlap_geo_plume"] = write_split_tables(query_geo, TABLE_DIR / "query" / "actual_overlap" / "geo_plume")

    # Paper/table inspection CSVs mirror the main split tables.
    sheet_dir = OUT / "excel_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for src, name in [
        (TABLE_DIR / "original" / "provided_split" / "train.csv", "orig_provided_train.csv"),
        (TABLE_DIR / "original" / "provided_split" / "test.csv", "orig_provided_test.csv"),
        (TABLE_DIR / "original" / "geo" / "train.csv", "orig_geo_train.csv"),
        (TABLE_DIR / "original" / "geo" / "test.csv", "orig_geo_test.csv"),
        (TABLE_DIR / "query" / "actual_overlap" / "plume_hash" / "train.csv", "query_actual_hash_train.csv"),
        (TABLE_DIR / "query" / "actual_overlap" / "plume_hash" / "test.csv", "query_actual_hash_test.csv"),
        (TABLE_DIR / "query" / "actual_overlap" / "geo_plume" / "train.csv", "query_actual_geo_train.csv"),
        (TABLE_DIR / "query" / "actual_overlap" / "geo_plume" / "test.csv", "query_actual_geo_test.csv"),
    ]:
        shutil.copyfile(src, sheet_dir / name)

    file_manifest = pd.DataFrame(file_rows).drop_duplicates()
    file_manifest.to_csv(OUT / "file_manifest.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(OUT / "s5p_path_errors.csv", index=False)
    else:
        (OUT / "s5p_path_errors.csv").write_text("id,column,source_path,error\n", encoding="utf-8")

    summary["file_manifest"] = {
        "rows": int(len(file_manifest)),
        "unique_source_paths": int(file_manifest["source_path"].nunique()),
        "unique_target_paths": int(file_manifest["target_path"].nunique()),
        "s5p_path_errors": int(len(errors)),
    }
    summary["notes"] = [
        "All release CSV tables are restricted to the requested 18 columns.",
        "Path values are target archive paths, not local absolute paths.",
        "S5p_temporal_path is required to point to .npz files; violations are written to s5p_path_errors.csv.",
        "Use file_manifest.csv with package_hf_dataset_parts.py to create dataset_part_XXX.tar.gz without first copying files.",
    ]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "README.md").write_text("""---
license: cc-by-nc-4.0
task_categories:
- image-classification
- image-segmentation
tags:
- methane
- remote-sensing
- multi-sensor
pretty_name: Multi-scale Methane Plume Dataset
---

# Multi-scale Methane Plume Dataset

CSV tables under `tables/` use exactly these columns: `id`, `label`, `latitude`, `longitude`, `available_sensor`, `S2_t0_path`, `S2_pre_path`, `S2_pre_pre_path`, `S2_plume_label_path`, `L89_t0_path`, `L89_pre_path`, `L89_pre_pre_path`, `L89_plume_label_path`, `EMIT_t0_path`, `EMIT_pre_path`, `EMIT_pre_pre_path`, `EMIT_plume_label_path`, `S5p_temporal_path`.

The path columns refer to files inside the `dataset_part_XXX.tar.gz` archives. S5P temporal files are `.npz`; plume masks are renamed to sensor-specific `*_plume.tif` names in the archive manifest.
""", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
