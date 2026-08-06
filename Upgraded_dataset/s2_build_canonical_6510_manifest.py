#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd


TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
EXACT_FILENAMES = {
    "t0": "s2.tif",
    "prev1": "s2_-7.tif",
    "prev2": "s2_prev2.tif",
    "prev3": "s2_prev3.tif",
    "seasonal": "s2_-90.tif",
    "year": "s2_-360.tif",
}
LEGACY_FILENAMES = {
    "t0": "s2_std_512.tif",
    "seasonal": "s2_90_std_512.tif",
    "year": "s2_360_std_512.tif",
}
NOMINAL_DAY_OFFSETS = {
    "t0": 0,
    "prev1": -7,
    "prev2": -14,
    "prev3": -21,
    "seasonal": -90,
    "year": -360,
}
STD512_COLUMNS = {
    "t0": "s2_0_std_512",
    "prev1": "s2_-7_std_512",
    "prev2": "s2_prev2_std_512",
    "prev3": "s2_prev3_std_512",
    "seasonal": "s2_-90_std_512",
    "year": "s2_-360_std_512",
}
DEFAULT_PRODUCT_ROOTS = ",".join(
    [
        "/diniuvol/yuyao/s2_cdse_point_repair_cache",
        "/diniuvol/yuyao/s2_early_boundary_products",
        "/mnt/engg-niulab/yuyao/sensors_raw_data/S2/raw_data_dir_s2",
        "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_S2",
        "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_s2_90360",
        "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2",
        "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2_-7",
    ]
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def file_ok(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def event_group_id(plume_id: str) -> str:
    parts = plume_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1] and len(parts[1]) <= 4:
        return parts[0]
    return plume_id


def iso_time(value: Any) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.isoformat()


def nominal_time(event_time: str, days: int) -> str:
    timestamp = pd.to_datetime(event_time, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return (timestamp + timedelta(days=days)).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a canonical six-timepoint S2 manifest from the 6510-ID "
            "catalogue using corrected exact crops, known-good legacy GEE "
            "512 files, and only locally available SAFE products."
        )
    )
    parser.add_argument(
        "--catalog",
        default=(
            "/home/yuyao/methane_train/Upgrade_data_pipeline/csv/"
            "s2_6510_missing_by_plume.csv"
        ),
    )
    parser.add_argument(
        "--metadata",
        default=(
            "/home/yuyao/methane_train/Upgrade_data_pipeline/csv/"
            "carbon_mapper_plumes_20160101_20260530_with_plume_tif.csv"
        ),
    )
    parser.add_argument(
        "--current-table",
        default=(
            "/home/yuyao/methane_train/Upgrade_data_pipeline/csv/"
            "s2_6time_point_covering_tiles_v3.csv"
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
        "--minus7-manifest",
        default=(
            "/home/yuyao/methane_train/preprocess_dataset_s2/"
            "manifest_minus7_plume_to_safe.csv"
        ),
    )
    parser.add_argument(
        "--exact-root",
        default=(
            "/mnt/engg-niulab/yuyao/sensors_raw_data/"
            "S2_point_center_exact_v3"
        ),
    )
    parser.add_argument(
        "--legacy-root",
        default=(
            "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/"
            "Dataset/plume_raw_s2_90360_fixed_512"
        ),
    )
    parser.add_argument(
        "--recrop-root",
        default=(
            "/mnt/engg-niulab/yuyao/sensors_raw_data/"
            "S2_canonical_6510_local_safe_recrop_v4"
        ),
    )
    parser.add_argument(
        "--product-roots",
        default=DEFAULT_PRODUCT_ROOTS,
    )
    parser.add_argument(
        "--output",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_canonical_6510_sources_v4.csv"
        ),
    )
    parser.add_argument(
        "--recrop-output",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_canonical_6510_recrop_tasks_v4.csv"
        ),
    )
    parser.add_argument(
        "--complete-output",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_canonical_6510_complete_v4.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_canonical_6510_report_v4.json"
        ),
    )
    return parser.parse_args()


def load_product_locations(
    roots: list[Path],
) -> tuple[dict[str, str], dict[str, int]]:
    locations: dict[str, str] = {}
    counts: dict[str, int] = {}
    for root in roots:
        try:
            names = os.listdir(root)
        except OSError:
            counts[str(root)] = 0
            continue
        product_names = [name for name in names if name.endswith(".SAFE")]
        counts[str(root)] = len(product_names)
        for name in product_names:
            locations.setdefault(name, str(root / name))
    return locations, counts


def directory_ids(root: Path) -> set[str]:
    try:
        return set(os.listdir(root))
    except OSError:
        return set()


def add_candidate(
    candidates: dict[tuple[str, str], list[dict[str, Any]]],
    plume_id: Any,
    timepoint: Any,
    product_name: Any,
    product_id: Any,
    acquisition_time: Any,
    source: str,
    rank: int,
) -> None:
    plume_id = clean(plume_id)
    timepoint = clean(timepoint)
    product_name = clean(product_name)
    product_id = clean(product_id)
    if (
        not plume_id
        or timepoint not in TIMEPOINTS
        or not product_name
        or not product_id
    ):
        return
    candidates[(plume_id, timepoint)].append(
        {
            "product_name": product_name,
            "product_id": product_id,
            "acquisition_time": iso_time(acquisition_time),
            "candidate_source": source,
            "candidate_rank": int(rank),
        }
    )


def build_candidates(
    current: pd.DataFrame,
    download: pd.DataFrame,
    minus7: pd.DataFrame,
    catalog_ids: set[str],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in current.to_dict("records"):
        plume_id = clean(row.get("plume_id"))
        if plume_id not in catalog_ids:
            continue
        for timepoint in TIMEPOINTS:
            add_candidate(
                candidates,
                plume_id,
                timepoint,
                row.get(f"{timepoint}_product_name"),
                row.get(f"{timepoint}_product_id"),
                row.get(f"{timepoint}_image_time"),
                "current_table",
                100,
            )

    status_rank = {
        "downloaded": 90,
        "available": 85,
        "target_exists": 85,
        "resume_skip_completed": 80,
        "planned": 60,
        "failed": 10,
    }
    for row in download.to_dict("records"):
        plume_id = clean(row.get("plume_id"))
        if plume_id not in catalog_ids:
            continue
        add_candidate(
            candidates,
            plume_id,
            row.get("timepoint"),
            row.get("product_name"),
            row.get("product_id"),
            row.get("acquisition_time"),
            "download_manifest",
            status_rank.get(clean(row.get("status")), 50),
        )

    for row in minus7.to_dict("records"):
        plume_id = clean(row.get("plume_id"))
        if plume_id not in catalog_ids:
            continue
        add_candidate(
            candidates,
            plume_id,
            "prev1",
            row.get("s2_minus7_product_name"),
            row.get("s2_minus7_id"),
            row.get("s2_minus7_datetime"),
            "legacy_minus7_manifest",
            95,
        )
    return candidates


def select_local_candidate(
    values: list[dict[str, Any]],
    product_locations: dict[str, str],
) -> dict[str, Any] | None:
    available = [
        value
        for value in values
        if value["product_name"] in product_locations
    ]
    if not available:
        return None
    available.sort(
        key=lambda value: (
            value["candidate_rank"],
            bool(value["acquisition_time"]),
        ),
        reverse=True,
    )
    selected = dict(available[0])
    selected["product_path"] = product_locations[selected["product_name"]]
    return selected


def main() -> int:
    args = parse_args()
    catalog = pd.read_csv(args.catalog, low_memory=False)
    catalog["plume_id"] = catalog["plume_id"].map(clean)
    catalog = catalog.drop_duplicates("plume_id", keep="first")
    catalog_ids = set(catalog["plume_id"])

    metadata = pd.read_csv(args.metadata, low_memory=False)
    metadata["plume_id"] = metadata["plume_id"].map(clean)
    metadata = metadata[metadata["plume_id"].isin(catalog_ids)]
    metadata = metadata.drop_duplicates("plume_id", keep="first")
    metadata_map = metadata.set_index("plume_id").to_dict("index")

    current = pd.read_csv(args.current_table, low_memory=False)
    current["plume_id"] = current["plume_id"].map(clean)
    current = current.drop_duplicates("plume_id", keep="first")
    current_map = current.set_index("plume_id").to_dict("index")
    download = pd.read_csv(args.download_manifest, low_memory=False)
    minus7 = pd.read_csv(args.minus7_manifest, low_memory=False)

    product_roots = [
        Path(value.strip())
        for value in args.product_roots.split(",")
        if value.strip()
    ]
    product_locations, product_root_counts = load_product_locations(
        product_roots
    )
    candidates = build_candidates(
        current,
        download,
        minus7,
        catalog_ids,
    )

    exact_root = Path(args.exact_root)
    legacy_root = Path(args.legacy_root)
    recrop_root = Path(args.recrop_root)
    current_ids = set(current["plume_id"])
    exact_ids = {
        timepoint: directory_ids(exact_root / timepoint) & current_ids
        for timepoint in TIMEPOINTS
    }
    legacy_ids = directory_ids(legacy_root)
    recrop_ids = {
        timepoint: directory_ids(recrop_root / timepoint)
        for timepoint in TIMEPOINTS
    }
    records: list[dict[str, Any]] = []
    recrop_records: list[dict[str, Any]] = []

    for plume_id in sorted(catalog_ids):
        metadata_row = metadata_map.get(plume_id, {})
        current_row = current_map.get(plume_id, {})
        event_time = iso_time(
            current_row.get("event_time")
            or metadata_row.get("datetime")
        )
        record: dict[str, Any] = {
            "plume_id": plume_id,
            "event_group_id": clean(
                current_row.get("event_group_id")
            )
            or event_group_id(plume_id),
            "event_time": event_time,
            "datetime": event_time,
            "plume_latitude": current_row.get(
                "plume_latitude",
                metadata_row.get("plume_latitude", ""),
            ),
            "plume_longitude": current_row.get(
                "plume_longitude",
                metadata_row.get("plume_longitude", ""),
            ),
            "plume_bounds": clean(
                current_row.get("plume_bounds")
            )
            or clean(metadata_row.get("plume_bounds")),
            "plume_tif": clean(metadata_row.get("plume_tif")),
        }
        recrop_record = dict(record)
        planned_recrops: list[str] = []
        unavailable: list[str] = []

        for timepoint in TIMEPOINTS:
            exact_path = (
                exact_root
                / timepoint
                / plume_id
                / EXACT_FILENAMES[timepoint]
            )
            legacy_path = (
                legacy_root
                / plume_id
                / LEGACY_FILENAMES.get(timepoint, "")
            )
            recrop_path = (
                recrop_root
                / timepoint
                / plume_id
                / EXACT_FILENAMES[timepoint]
            )
            selected = select_local_candidate(
                candidates.get((plume_id, timepoint), []),
                product_locations,
            )

            input_path = ""
            input_kind = ""
            if plume_id in exact_ids[timepoint]:
                input_path = str(exact_path)
                input_kind = "point_exact_v3_harmonized"
            elif (
                timepoint in LEGACY_FILENAMES
                and plume_id in legacy_ids
            ):
                input_path = str(legacy_path)
                input_kind = "legacy_gee_correct_512"
            elif plume_id in recrop_ids[timepoint]:
                input_path = str(recrop_path)
                input_kind = "local_safe_point_recrop_v4"
            elif selected is not None:
                planned_recrops.append(timepoint)
                recrop_record[f"{timepoint}_product_name"] = selected[
                    "product_name"
                ]
                recrop_record[f"{timepoint}_product_id"] = selected[
                    "product_id"
                ]
                recrop_record[f"{timepoint}_download_target_raw_path"] = str(
                    recrop_path
                )
                recrop_record[f"{timepoint}_raw_path"] = ""
                recrop_record[f"{timepoint}_download_needed"] = 1
                recrop_record[f"{timepoint}_local_status"] = "missing"
                recrop_record[f"{timepoint}_path_source"] = (
                    "canonical_local_safe"
                )
                recrop_record[f"{timepoint}_product_local_path"] = selected[
                    "product_path"
                ]
                recrop_record[f"{timepoint}_candidate_source"] = selected[
                    "candidate_source"
                ]
            else:
                unavailable.append(timepoint)

            image_time = iso_time(
                current_row.get(f"{timepoint}_image_time")
            )
            if not image_time and selected is not None:
                image_time = selected["acquisition_time"]
            if not image_time:
                image_time = nominal_time(
                    event_time,
                    NOMINAL_DAY_OFFSETS[timepoint],
                )
            record[f"{timepoint}_input_path"] = input_path
            record[f"{timepoint}_input_kind"] = input_kind
            record[f"{timepoint}_raw_path"] = input_path
            record[f"{timepoint}_image_time"] = image_time
            record[f"{timepoint}_source_available"] = int(bool(input_path))
            record[STD512_COLUMNS[timepoint]] = input_path
            recrop_record[f"{timepoint}_image_time"] = image_time
            recrop_record.setdefault(f"{timepoint}_product_name", "")
            recrop_record.setdefault(f"{timepoint}_product_id", "")
            recrop_record.setdefault(f"{timepoint}_download_needed", 0)
            recrop_record.setdefault(
                f"{timepoint}_download_target_raw_path",
                "",
            )
            recrop_record.setdefault(f"{timepoint}_raw_path", input_path)
            recrop_record.setdefault(
                f"{timepoint}_local_status",
                "available" if input_path else "missing",
            )
            recrop_record.setdefault(
                f"{timepoint}_path_source",
                input_kind,
            )

        record["missing_timepoints"] = ",".join(
            timepoint
            for timepoint in TIMEPOINTS
            if not record[f"{timepoint}_input_path"]
        )
        record["all6_complete"] = int(not record["missing_timepoints"])
        record["projected_complete_after_recrop"] = int(not unavailable)
        record["unavailable_timepoints"] = ",".join(unavailable)
        record["input_kind_pattern"] = "|".join(
            record[f"{timepoint}_input_kind"] for timepoint in TIMEPOINTS
        )
        recrop_record["planned_recrop_timepoints"] = ",".join(
            planned_recrops
        )
        recrop_record["projected_complete_after_recrop"] = int(
            not unavailable
        )
        recrop_record["already_complete"] = record["all6_complete"]
        records.append(record)
        recrop_records.append(recrop_record)

    output = pd.DataFrame(records)
    recrop_output = pd.DataFrame(recrop_records)
    recrop_output = recrop_output[
        (recrop_output["projected_complete_after_recrop"] == 1)
        & (recrop_output["already_complete"] == 0)
    ].copy()
    complete = output[output["all6_complete"] == 1].copy()
    output_path = Path(args.output)
    recrop_output_path = Path(args.recrop_output)
    complete_output_path = Path(args.complete_output)
    report_path = Path(args.report)
    for path in (
        output_path,
        recrop_output_path,
        complete_output_path,
        report_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    recrop_output.to_csv(recrop_output_path, index=False)
    complete.to_csv(complete_output_path, index=False)

    report = {
        "catalog_rows": int(len(catalog)),
        "metadata_rows": int(len(metadata)),
        "current_rows": int(len(current)),
        "available_product_names": int(len(product_locations)),
        "product_root_counts": product_root_counts,
        "complete_rows": int(len(complete)),
        "projected_complete_rows": int(
            output["projected_complete_after_recrop"].sum()
        ),
        "recrop_rows": int(len(recrop_output)),
        "incomplete_rows": int(len(output) - len(complete)),
        "source_counts": {
            timepoint: output[f"{timepoint}_input_kind"]
            .replace("", "missing")
            .value_counts()
            .to_dict()
            for timepoint in TIMEPOINTS
        },
        "missing_counts": {
            timepoint: int(
                (output[f"{timepoint}_source_available"] == 0).sum()
            )
            for timepoint in TIMEPOINTS
        },
        "planned_local_recrops": {
            timepoint: int(
                recrop_output[f"{timepoint}_product_name"].ne("").sum()
            )
            for timepoint in TIMEPOINTS
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
