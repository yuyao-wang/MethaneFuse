#!/usr/bin/env python3
"""Build a six-time S2 source manifest from completed CDSE and legacy 512 data."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import rasterio
import tifffile


METHANE_ROOT = Path("/home/yuyao/methane_train")
CSV_ROOT = METHANE_ROOT / "Upgrade_data_pipeline" / "csv"
DEFAULT_MAIN_CSV = CSV_ROOT / "carbon_mapper_plumes_20160101_20260530_with_t0_flags.csv"
DEFAULT_CDSE_512_CSV = CSV_ROOT / "s2_6time_all6_available_paths_std512_complete.csv"
DEFAULT_DOWNLOAD_CSV = CSV_ROOT / "s2_download_manifest.csv"
DEFAULT_OLD3_CSV = METHANE_ROOT / "preprocess_dataset_s2" / "raw_s2_90360_cleaned_fixed.csv"
DEFAULT_OLD4_CSV = METHANE_ROOT / "preprocess_dataset_s2" / "CM_S2_L2A_-7_gee90360_std512.csv"
DEFAULT_OUT_CSV = CSV_ROOT / "s2_6510_existing_source_census.csv"
DEFAULT_SUMMARY_JSON = CSV_ROOT / "s2_6510_existing_source_census_summary.json"

NIULAB_512_ROOT = Path("/mnt/engg-niulab/yuyao/preprocessed_512/S2")
NIULAB_RAW_ROOT = Path("/mnt/engg-niulab/yuyao/sensors_raw_data/S2")
NIULAB_CM_ROOT = Path("/mnt/engg-niulab/yuyao/sensors_raw_data/CM")
LEUNG_DATASET_ROOT = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset")


@dataclass(frozen=True)
class Timepoint:
    name: str
    std_column: str
    std_filename: str
    raw_filename: str
    image_time_column: str
    nominal_days: int


TIMEPOINTS = (
    Timepoint("t0", "s2_0_std_512", "s2_0_std_512.tif", "s2.tif", "t0_image_time", 0),
    Timepoint("prev1", "s2_-7_std_512", "s2_-7_std_512.tif", "s2_-7.tif", "prev1_image_time", -7),
    Timepoint("prev2", "s2_prev2_std_512", "s2_prev2_std_512.tif", "s2_prev2.tif", "prev2_image_time", -14),
    Timepoint("prev3", "s2_prev3_std_512", "s2_prev3_std_512.tif", "s2_prev3.tif", "prev3_image_time", -21),
    Timepoint("seasonal", "s2_-90_std_512", "s2_-90_std_512.tif", "s2_-90.tif", "seasonal_image_time", -90),
    Timepoint("year", "s2_-360_std_512", "s2_-360_std_512.tif", "s2_-360.tif", "year_image_time", -360),
)

CURRENT_GEE_MARKERS = ("/S2_GEE_6time/", "CM_S2_L2A_6TIME_GEE")


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def file_ok(path: str) -> bool:
    if not path:
        return False
    try:
        result = os.stat(path)
        return stat.S_ISREG(result.st_mode) and result.st_size > 0
    except OSError:
        return False


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", mode="w", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def indexed_csv(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, low_memory=False)
    if "plume_id" not in frame.columns:
        return {}
    frame = frame[frame["plume_id"].notna()].drop_duplicates("plume_id", keep="last")
    return {clean(row["plume_id"]): row for row in frame.to_dict("records")}


def nominal_time(event_time: Any, days: int) -> str:
    timestamp = pd.to_datetime(event_time, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return (timestamp + pd.Timedelta(days=days)).isoformat()


def add_candidate(
    candidates: dict[tuple[str, str], list[tuple[int, int, str, str, str]]],
    seen: dict[tuple[str, str], set[str]],
    plume_id: str,
    timepoint: str,
    priority: int,
    source: str,
    kind: str,
    path: Any,
    order: int,
) -> int:
    value = clean(path)
    if not value or any(marker in value for marker in CURRENT_GEE_MARKERS):
        return order
    key = (plume_id, timepoint)
    if value in seen[key]:
        return order
    seen[key].add(value)
    candidates[key].append((priority, order, source, kind, value))
    return order + 1


def legacy_patterns(plume_id: str, timepoint: str) -> Iterable[tuple[str, Path]]:
    if timepoint == "t0":
        yield "engg_leung_old3_fixed512", LEUNG_DATASET_ROOT / "plume_raw_s2_90360_fixed_512" / plume_id / "s2_std_512.tif"
        yield "engg_leung_old_t0_512", LEUNG_DATASET_ROOT / "plume_s2_CDSE0_gee90360_original_512" / plume_id / "s2_std.tif"
    elif timepoint == "seasonal":
        yield "engg_leung_old3_fixed512", LEUNG_DATASET_ROOT / "plume_raw_s2_90360_fixed_512" / plume_id / "s2_90_std_512.tif"
        yield "engg_leung_old_seasonal_512", LEUNG_DATASET_ROOT / "plume_raw_s2_gee90360_512" / plume_id / "s2_90_std_512.tif"
    elif timepoint == "year":
        yield "engg_leung_old3_fixed512", LEUNG_DATASET_ROOT / "plume_raw_s2_90360_fixed_512" / plume_id / "s2_360_std_512.tif"
        yield "engg_leung_old_year_512", LEUNG_DATASET_ROOT / "plume_raw_s2_gee90360_512" / plume_id / "s2_360_std_512.tif"


def raster_shape(path: str) -> tuple[int, int, int] | None:
    try:
        with rasterio.open(path) as dataset:
            shape = (int(dataset.count), int(dataset.height), int(dataset.width))
        if shape[0] > 1:
            return shape
    except Exception:
        pass
    try:
        with tifffile.TiffFile(path) as dataset:
            shape = tuple(int(value) for value in dataset.series[0].shape)
        if len(shape) != 3:
            return None
        if shape[0] in {12, 13}:
            return shape
        if shape[-1] in {12, 13}:
            return (shape[-1], shape[0], shape[1])
    except Exception:
        return None
    return None


def validate_selected(item: tuple[str, str]) -> tuple[str, bool, str]:
    path, kind = item
    shape = raster_shape(path)
    if shape is None:
        return path, False, "unreadable_or_non_3d"
    bands, height, width = shape
    if bands not in {12, 13}:
        return path, False, f"bands={bands}"
    if kind == "std512" and (height, width) != (512, 512):
        return path, False, f"shape={shape}"
    return path, True, f"shape={shape}"


def build(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    main = pd.read_csv(args.main_csv, low_memory=False)
    target = main[main["s2_has_t0"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    target = target[target["plume_id"].notna()].drop_duplicates("plume_id", keep="last")
    target["plume_id"] = target["plume_id"].astype(str)
    target_ids = set(target["plume_id"])

    cdse = indexed_csv(Path(args.cdse_512_csv))
    old3 = indexed_csv(Path(args.old3_csv))
    old4 = indexed_csv(Path(args.old4_csv))
    download = pd.read_csv(args.download_csv, low_memory=False)
    download = download[download["plume_id"].notna()].copy()
    download["plume_id"] = download["plume_id"].astype(str)
    download = download[download["plume_id"].isin(target_ids)]

    download_paths: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    download_times: dict[tuple[str, str, str], str] = {}
    for row in download.to_dict("records"):
        plume_id = clean(row.get("plume_id"))
        timepoint = clean(row.get("timepoint"))
        raw_path = clean(row.get("raw_path"))
        if plume_id and timepoint and raw_path:
            download_paths[(plume_id, timepoint)].append((raw_path, clean(row.get("status"))))
            acquisition = clean(row.get("acquisition_time"))
            if acquisition:
                download_times[(plume_id, timepoint, raw_path)] = acquisition

    candidates: dict[tuple[str, str], list[tuple[int, int, str, str, str]]] = defaultdict(list)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    order = 0
    for row in target.to_dict("records"):
        plume_id = clean(row["plume_id"])
        cdse_row = cdse.get(plume_id, {})
        old3_row = old3.get(plume_id, {})
        old4_row = old4.get(plume_id, {})
        for tp in TIMEPOINTS:
            order = add_candidate(candidates, seen, plume_id, tp.name, 0, "cdse_completed_std512", "std512", cdse_row.get(tp.std_column), order)

            if tp.name in {"t0", "prev1", "seasonal", "year"} and old4_row:
                order = add_candidate(
                    candidates,
                    seen,
                    plume_id,
                    tp.name,
                    2,
                    "legacy_old4_niulab_std512",
                    "std512",
                    NIULAB_512_ROOT / plume_id / tp.std_filename,
                    order,
                )

            main_column = {"t0": "s2_0_std_512", "seasonal": "s2_-90_std_512", "year": "s2_-360_std_512"}.get(tp.name)
            if main_column:
                order = add_candidate(candidates, seen, plume_id, tp.name, 3, "main_engg_leung_legacy512", "std512", row.get(main_column), order)

            old3_column = {"t0": "s2_path_std", "seasonal": "s2_90_path_std", "year": "s2_360_path_std"}.get(tp.name)
            if old3_column:
                order = add_candidate(candidates, seen, plume_id, tp.name, 4, "engg_leung_old3_csv_512", "std512", old3_row.get(old3_column), order)

            old4_column = {
                "t0": "s2_0_std_512",
                "prev1": "s2_-7_std_512",
                "seasonal": "s2_-90_std_512",
                "year": "s2_-360_std_512",
            }.get(tp.name)
            if old4_column:
                order = add_candidate(candidates, seen, plume_id, tp.name, 5, "legacy_old4_csv_512", "std512", old4_row.get(old4_column), order)

            order = add_candidate(
                candidates,
                seen,
                plume_id,
                tp.name,
                6,
                "niulab_existing_std512",
                "std512",
                NIULAB_512_ROOT / plume_id / tp.std_filename,
                order,
            )
            for source, path in legacy_patterns(plume_id, tp.name):
                order = add_candidate(candidates, seen, plume_id, tp.name, 7, source, "std512", path, order)

            for raw_path, status in download_paths.get((plume_id, tp.name), []):
                source = f"cdse_download_manifest:{status or 'unknown'}"
                order = add_candidate(candidates, seen, plume_id, tp.name, 20, source, "raw", raw_path, order)
            order = add_candidate(
                candidates,
                seen,
                plume_id,
                tp.name,
                21,
                "cdse_canonical_raw",
                "raw",
                NIULAB_RAW_ROOT / tp.name / plume_id / tp.raw_filename,
                order,
            )

    mask_candidates: dict[str, tuple[str, str]] = {}
    for plume_id in target_ids:
        old3_mask = clean(old3.get(plume_id, {}).get("resized_512x512_path"))
        raw_mask = str(NIULAB_CM_ROOT / plume_id / "plume.tif")
        mask_candidates[plume_id] = (old3_mask, raw_mask)

    unique_paths = sorted(
        {candidate[4] for values in candidates.values() for candidate in values}
        | {path for values in mask_candidates.values() for path in values if path}
    )
    with ThreadPoolExecutor(max_workers=max(1, int(args.stat_workers))) as pool:
        availability = dict(zip(unique_paths, pool.map(file_ok, unique_paths)))

    selected: dict[tuple[str, str], tuple[str, str, str]] = {}
    for key, values in candidates.items():
        for _, _, source, kind, path in sorted(values):
            if availability.get(path, False):
                selected[key] = (source, kind, path)
                break

    validation: dict[str, tuple[bool, str]] = {}
    selected_items = sorted({(value[2], value[1]) for value in selected.values()})
    if args.validate_metadata:
        with ThreadPoolExecutor(max_workers=max(1, int(args.metadata_workers))) as pool:
            for path, ok, note in pool.map(validate_selected, selected_items):
                validation[path] = (ok, note)
    else:
        validation = {path: (True, "not_checked") for path, _ in selected_items}

    rows: list[dict[str, Any]] = []
    for row in target.sort_values(["datetime", "plume_id"], kind="stable").to_dict("records"):
        plume_id = clean(row["plume_id"])
        cdse_row = cdse.get(plume_id, {})
        old3_row = old3.get(plume_id, {})
        record: dict[str, Any] = {
            "plume_id": plume_id,
            "event_group_id": clean(row.get("event_group_id")) or plume_id.rsplit("-", 1)[0],
            "event_time": clean(row.get("datetime")),
            "plume_latitude": row.get("plume_latitude", ""),
            "plume_longitude": row.get("plume_longitude", ""),
            "plume_bounds": clean(row.get("plume_bounds")),
            "is_cdse_all6_completed": int(plume_id in cdse),
        }
        missing: list[str] = []
        invalid: list[str] = []
        kinds: list[str] = []
        for tp in TIMEPOINTS:
            value = selected.get((plume_id, tp.name))
            if value is None:
                source, kind, path = "", "", ""
                missing.append(tp.name)
                valid, note = False, "missing"
            else:
                source, kind, path = value
                valid, note = validation.get(path, (False, "not_validated"))
                if not valid:
                    invalid.append(tp.name)
                kinds.append(kind)
            record[f"{tp.name}_source"] = source
            record[f"{tp.name}_input_kind"] = kind
            record[f"{tp.name}_input_path"] = path
            record[f"{tp.name}_metadata_ok"] = int(valid)
            record[f"{tp.name}_metadata_note"] = note
            image_time = clean(cdse_row.get(tp.image_time_column))
            if not image_time and path:
                image_time = download_times.get((plume_id, tp.name, path), "")
            if not image_time and tp.name == "t0":
                image_time = clean(row.get("s2_t0_time"))
            record[tp.image_time_column] = image_time or nominal_time(row.get("datetime"), tp.nominal_days)

        legacy_mask, raw_mask = mask_candidates[plume_id]
        if availability.get(legacy_mask, False):
            record["mask_source"] = "engg_leung_legacy_512_mask"
            record["mask_input_kind"] = "mask512"
            record["mask_input_path"] = legacy_mask
        elif availability.get(raw_mask, False):
            record["mask_source"] = "niulab_raw_cm_mask"
            record["mask_input_kind"] = "raw_mask"
            record["mask_input_path"] = raw_mask
        else:
            record["mask_source"] = ""
            record["mask_input_kind"] = ""
            record["mask_input_path"] = ""
        record["missing_timepoints"] = "+".join(missing)
        record["invalid_timepoints"] = "+".join(invalid)
        record["input_kind_pattern"] = "+".join(kinds)
        record["has_all6_sources"] = int(not missing)
        record["has_all6_valid_metadata"] = int(not missing and not invalid)
        record["has_mask_source"] = int(bool(record["mask_input_path"]))
        rows.append(record)

    output = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "target_rows": int(len(output)),
        "target_unique_plumes": int(output["plume_id"].nunique()),
        "cdse_all6_completed_rows": int(output["is_cdse_all6_completed"].sum()),
        "needs_source_merge_rows": int((output["is_cdse_all6_completed"] == 0).sum()),
        "has_all6_sources": int(output["has_all6_sources"].sum()),
        "has_all6_valid_metadata": int(output["has_all6_valid_metadata"].sum()),
        "has_mask_source": int(output["has_mask_source"].sum()),
        "missing_pattern_counts": dict(Counter(output.loc[output["missing_timepoints"].ne(""), "missing_timepoints"])),
        "invalid_pattern_counts": dict(Counter(output.loc[output["invalid_timepoints"].ne(""), "invalid_timepoints"])),
        "timepoints": {},
    }
    for tp in TIMEPOINTS:
        summary["timepoints"][tp.name] = {
            "selected": int(output[f"{tp.name}_input_path"].ne("").sum()),
            "valid_metadata": int(output[f"{tp.name}_metadata_ok"].sum()),
            "source_counts": dict(Counter(output.loc[output[f"{tp.name}_source"].ne(""), f"{tp.name}_source"])),
            "kind_counts": dict(Counter(output.loc[output[f"{tp.name}_input_kind"].ne(""), f"{tp.name}_input_kind"])),
        }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-csv", default=str(DEFAULT_MAIN_CSV))
    parser.add_argument("--cdse-512-csv", default=str(DEFAULT_CDSE_512_CSV))
    parser.add_argument("--download-csv", default=str(DEFAULT_DOWNLOAD_CSV))
    parser.add_argument("--old3-csv", default=str(DEFAULT_OLD3_CSV))
    parser.add_argument("--old4-csv", default=str(DEFAULT_OLD4_CSV))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--stat-workers", type=int, default=48)
    parser.add_argument("--metadata-workers", type=int, default=24)
    parser.add_argument("--validate-metadata", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, summary = build(args)
    atomic_csv(output, Path(args.out_csv))
    atomic_json(summary, Path(args.summary_json))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
