#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
RECROP_TIMEPOINTS = ("prev1", "prev2", "prev3")
LEGACY_FILENAMES = {
    "t0": "s2_std_512.tif",
    "seasonal": "s2_90_std_512.tif",
    "year": "s2_360_std_512.tif",
}
TARGET_FILENAMES = {
    "prev1": "s2_-7.tif",
    "prev2": "s2_prev2.tif",
    "prev3": "s2_prev3.tif",
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


def existing_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the legacy-cohort manifest using the known-good GEE 512 "
            "timepoints and locally available CDSE SAFE products."
        )
    )
    parser.add_argument(
        "--source-table",
        default=(
            "/home/yuyao/methane_train/Upgrade_data_pipeline/csv/"
            "s2_6time_legacy_rebuild/s2_6time_legacy_exact_sources.csv"
        ),
    )
    parser.add_argument(
        "--download-manifest",
        default=(
            "/home/yuyao/methane_train/Upgrade_data_pipeline/csv/"
            "s2_download_manifest.csv"
        ),
    )
    parser.add_argument(
        "--legacy-512-root",
        default=(
            "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/"
            "Dataset/plume_raw_s2_90360_fixed_512"
        ),
    )
    parser.add_argument(
        "--recrop-root",
        default=(
            "/mnt/engg-niulab/yuyao/sensors_raw_data/"
            "S2_legacy_prev123_point_center_exact_v4"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_legacy_recovery_v4.csv"
        ),
    )
    parser.add_argument(
        "--complete-output",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_legacy_recovery_v4_complete.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_legacy_recovery_v4_report.json"
        ),
    )
    return parser.parse_args()


def select_product_rows(
    download_manifest: pd.DataFrame,
    plume_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    manifest = download_manifest.copy()
    manifest["plume_id"] = manifest["plume_id"].map(clean)
    manifest["timepoint"] = manifest["timepoint"].map(clean)
    manifest = manifest[
        manifest["plume_id"].isin(plume_ids)
        & manifest["timepoint"].isin(RECROP_TIMEPOINTS)
    ].copy()
    for column in (
        "status",
        "product_name",
        "product_id",
        "acquisition_time",
        "selection_source",
        "raw_path",
        "target_512_path",
        "existing_512_path",
    ):
        manifest[column] = manifest[column].map(clean)

    manifest["has_product"] = (
        manifest["product_name"].ne("") & manifest["product_id"].ne("")
    )
    manifest["status_rank"] = manifest["status"].map(
        {
            "downloaded": 5,
            "available": 4,
            "existing": 4,
            "target_exists": 4,
            "planned": 3,
            "missing": 1,
            "failed": 0,
        }
    ).fillna(2)
    manifest["has_acquisition"] = manifest["acquisition_time"].ne("").astype(int)
    manifest["has_recorded_path"] = manifest[
        ["raw_path", "target_512_path", "existing_512_path"]
    ].apply(lambda row: int(any(clean(value) for value in row)), axis=1)
    manifest = manifest.sort_values(
        [
            "plume_id",
            "timepoint",
            "has_product",
            "has_recorded_path",
            "status_rank",
            "has_acquisition",
        ],
        ascending=[True, True, False, False, False, False],
        kind="stable",
    )

    conflicts = (
        manifest[manifest["has_product"]]
        .groupby(["plume_id", "timepoint"])["product_name"]
        .nunique()
    )
    conflict_groups = int((conflicts > 1).sum())
    selected = manifest.drop_duplicates(["plume_id", "timepoint"], keep="first")
    selected = selected[selected["has_product"]].copy()
    return selected, {
        "candidate_rows": int(len(manifest)),
        "selected_product_rows": int(len(selected)),
        "conflicting_product_groups": conflict_groups,
    }


def main() -> int:
    args = parse_args()
    source = pd.read_csv(args.source_table, low_memory=False)
    legacy = source[
        source["cohort"].astype(str)
        == "legacy_existing3_plus_gee_prev123"
    ].copy()
    legacy["plume_id"] = legacy["plume_id"].map(clean)
    legacy = legacy.drop_duplicates("plume_id", keep="first").reset_index(drop=True)

    legacy_root = Path(args.legacy_512_root)
    recrop_root = Path(args.recrop_root)
    for timepoint, filename in LEGACY_FILENAMES.items():
        path_column = f"{timepoint}_input_path"
        kind_column = f"{timepoint}_input_kind"
        legacy[path_column] = legacy["plume_id"].map(
            lambda plume_id: str(legacy_root / plume_id / filename)
        )
        legacy[kind_column] = "legacy_gee_512"
        legacy[f"has_{timepoint}_legacy_file"] = legacy[path_column].map(
            lambda value: int(existing_file(Path(value)))
        )

    manifest = pd.read_csv(args.download_manifest, low_memory=False)
    selected, selection_report = select_product_rows(
        manifest,
        set(legacy["plume_id"]),
    )
    selected = selected.set_index(["plume_id", "timepoint"])

    for timepoint in RECROP_TIMEPOINTS:
        product_name_column = f"{timepoint}_product_name"
        product_id_column = f"{timepoint}_product_id"
        image_time_column = f"{timepoint}_image_time"
        input_path_column = f"{timepoint}_input_path"
        input_kind_column = f"{timepoint}_input_kind"
        raw_path_column = f"{timepoint}_raw_path"
        target_column = f"{timepoint}_download_target_raw_path"

        product_names: list[str] = []
        product_ids: list[str] = []
        acquisition_times: list[str] = []
        targets: list[str] = []
        for plume_id in legacy["plume_id"]:
            key = (plume_id, timepoint)
            if key in selected.index:
                record = selected.loc[key]
                product_names.append(clean(record["product_name"]))
                product_ids.append(clean(record["product_id"]))
                acquisition_times.append(clean(record["acquisition_time"]))
            else:
                product_names.append("")
                product_ids.append("")
                acquisition_times.append("")
            targets.append(
                str(
                    recrop_root
                    / timepoint
                    / plume_id
                    / TARGET_FILENAMES[timepoint]
                )
            )

        legacy[product_name_column] = product_names
        legacy[product_id_column] = product_ids
        legacy[image_time_column] = acquisition_times
        legacy[target_column] = targets
        legacy[raw_path_column] = targets
        legacy[input_path_column] = targets
        legacy[input_kind_column] = "local_safe_point_recrop"
        legacy[f"has_{timepoint}_product"] = (
            legacy[product_name_column].ne("")
            & legacy[product_id_column].ne("")
        ).astype(int)
        legacy[f"has_{timepoint}_recrop"] = legacy[target_column].map(
            lambda value: int(existing_file(Path(value)))
        )
        legacy[f"{timepoint}_download_needed"] = (
            1 - legacy[f"has_{timepoint}_recrop"]
        )
        legacy[f"{timepoint}_local_status"] = legacy[
            f"has_{timepoint}_recrop"
        ].map({1: "available", 0: "missing"})
        legacy[f"{timepoint}_path_source"] = "legacy_recovery_v4"

    for timepoint in TIMEPOINTS:
        if f"{timepoint}_raw_path" not in legacy:
            legacy[f"{timepoint}_raw_path"] = legacy.get(
                f"{timepoint}_input_path",
                "",
            )
        if f"{timepoint}_download_needed" not in legacy:
            legacy[f"{timepoint}_download_needed"] = 0
        if f"{timepoint}_local_status" not in legacy:
            legacy[f"{timepoint}_local_status"] = "available"
        if f"{timepoint}_path_source" not in legacy:
            legacy[f"{timepoint}_path_source"] = "legacy_gee_512"

    legacy["has_legacy3"] = legacy[
        [f"has_{timepoint}_legacy_file" for timepoint in LEGACY_FILENAMES]
    ].min(axis=1)
    legacy["has_prev123_products"] = legacy[
        [f"has_{timepoint}_product" for timepoint in RECROP_TIMEPOINTS]
    ].min(axis=1)
    legacy["has_prev123_recrops"] = legacy[
        [f"has_{timepoint}_recrop" for timepoint in RECROP_TIMEPOINTS]
    ].min(axis=1)
    legacy["source_complete"] = (
        (legacy["has_legacy3"] == 1)
        & (legacy["has_prev123_products"] == 1)
    ).astype(int)
    legacy["all6_complete"] = (
        (legacy["has_legacy3"] == 1)
        & (legacy["has_prev123_recrops"] == 1)
    ).astype(int)

    output_path = Path(args.output)
    complete_output_path = Path(args.complete_output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    legacy.to_csv(output_path, index=False)
    legacy[legacy["source_complete"] == 1].to_csv(
        complete_output_path,
        index=False,
    )

    report = {
        "source_table": args.source_table,
        "download_manifest": args.download_manifest,
        "legacy_512_root": args.legacy_512_root,
        "recrop_root": args.recrop_root,
        "legacy_rows": int(len(legacy)),
        "legacy_unique_ids": int(legacy["plume_id"].nunique()),
        "has_legacy3": int(legacy["has_legacy3"].sum()),
        "has_prev123_products": int(legacy["has_prev123_products"].sum()),
        "source_complete": int(legacy["source_complete"].sum()),
        "all6_complete_before_recrop": int(legacy["all6_complete"].sum()),
        "missing_by_timepoint": {
            timepoint: {
                "legacy_file_missing": int(
                    len(legacy)
                    - legacy.get(
                        f"has_{timepoint}_legacy_file",
                        pd.Series([1] * len(legacy)),
                    ).sum()
                ),
                "product_missing": int(
                    len(legacy)
                    - legacy.get(
                        f"has_{timepoint}_product",
                        pd.Series([1] * len(legacy)),
                    ).sum()
                ),
                "recrop_missing": int(
                    len(legacy)
                    - legacy.get(
                        f"has_{timepoint}_recrop",
                        pd.Series([1] * len(legacy)),
                    ).sum()
                ),
            }
            for timepoint in TIMEPOINTS
        },
        **selection_report,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
