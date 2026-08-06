#!/usr/bin/env python3
"""Build the leak-free, partial-sensor six-timepoint 512-source manifest.

The Carbon Mapper catalogue is authoritative for ``event_group_id`` and event
time.  Sensor manifests are joined by ``plume_id`` after an explicit duplicate
consistency audit.  The script never rewrites or normalizes source paths:
existence checks use the exact, case-sensitive strings found in each input.

This script only creates metadata.  It does not crop imagery or modify any
legacy pipeline/data product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import pandas as pd


TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
SENSOR_ORDER = ("s2", "l89", "emit", "s5p")
SCHEMA_VERSION = "methanefuse_unified_512_manifest_v1"

DEFAULT_CATALOGUE = Path(
    "/home/yuyao/methane_train/Upgrade_data_pipeline/csv/"
    "carbon_mapper_plumes_20160101_20260530_with_t0_flags.csv"
)
DEFAULT_S2 = Path(
    "/home/yuyao/panopticon/Upgraded_dataset/"
    "s2_historical_point_center_v14/s2_v14_all_512.csv"
)
DEFAULT_L89 = Path(
    "/home/yuyao/methane_train/Upgrade_data_pipeline/csv/"
    "l89_6time_complete_paths.csv"
)
DEFAULT_EMIT = Path(
    "/mnt/engg-niulab/Yuyao/preprocessed_512/emit_32band/"
    "emit_6time_512_manifest.csv"
)
DEFAULT_S5P = Path(
    "/home/yuyao/methane_train/Upgrade_data_pipeline/csv/"
    "s5p_6time_with_centers.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/manifests"
)


@dataclass(frozen=True)
class SensorSpec:
    name: str
    source_path_columns: Mapping[str, str]
    mask_source_column: str | None
    metadata_columns: Mapping[str, str]
    source_event_group_column: str | None = None
    source_event_time_column: str = "event_time"
    source_latitude_column: str = "plume_latitude"
    source_longitude_column: str = "plume_longitude"
    source_bounds_column: str = "plume_bounds"
    source_plume_tif_column: str | None = "plume_tif"
    declared_all6_column: str | None = None
    source_kind: str = "512_raster"


def _time_metadata(
    *,
    image_time: bool = True,
    product_id: bool = False,
    overpass_key: bool = False,
    cloud_cover: bool = False,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for timepoint in TIMEPOINTS:
        if image_time:
            result[f"{timepoint}_image_time"] = f"{timepoint}_image_time"
        if product_id:
            result[f"{timepoint}_product_id"] = f"{timepoint}_product_id"
        if overpass_key:
            result[f"{timepoint}_overpass_key"] = f"{timepoint}_overpass_key"
        if cloud_cover:
            result[f"{timepoint}_cloud_cover"] = f"{timepoint}_cloud_cover"
    return result


def _s5p_metadata() -> dict[str, str]:
    result: dict[str, str] = {}
    for timepoint in TIMEPOINTS:
        result[f"s5p_{timepoint}_image_time"] = f"{timepoint}_image_time"
        result[f"s5p_{timepoint}_product_id"] = f"{timepoint}_product_id"
        for suffix in (
            "target_time",
            "time_delta_hours",
            "qc_reason",
            "qc_center_iy",
            "qc_center_ix",
            "qc_center_distance_km",
            "qc_patch_missing_ratio",
            "qc_patch_finite_count",
            "qc_patch_total",
        ):
            source = f"s5p_{timepoint}_{suffix}"
            result[source] = f"{timepoint}_{suffix}"
    result.update(
        {
            "center_status": "center_status",
            "center_note": "center_note",
            "nearest_iy": "nearest_iy",
            "nearest_ix": "nearest_ix",
            "pos_centers": "pos_centers",
        }
    )
    return result


SENSOR_SPECS = {
    "s2": SensorSpec(
        name="s2",
        source_path_columns={
            "t0": "s2_0_std_512",
            "prev1": "s2_-7_std_512",
            "prev2": "s2_prev2_std_512",
            "prev3": "s2_prev3_std_512",
            "seasonal": "s2_-90_std_512",
            "year": "s2_-360_std_512",
        },
        mask_source_column="s2_mask_512_path",
        metadata_columns=_time_metadata(product_id=True),
        source_event_group_column="event_group_id",
        declared_all6_column="has_all6_512",
    ),
    "l89": SensorSpec(
        name="l89",
        source_path_columns={
            "t0": "l89_0_std_512",
            "prev1": "l89_-7_std_512",
            "prev2": "l89_prev2_std_512",
            "prev3": "l89_prev3_std_512",
            "seasonal": "l89_-90_std_512",
            "year": "l89_-360_std_512",
        },
        mask_source_column="l89_512_mask_path",
        metadata_columns=_time_metadata(
            overpass_key=True,
            cloud_cover=True,
        ),
        declared_all6_column="std_ok",
    ),
    "emit": SensorSpec(
        name="emit",
        source_path_columns={
            "t0": "t0_512_path",
            "prev1": "prev1_512_path",
            "prev2": "prev2_512_path",
            "prev3": "prev3_512_path",
            "seasonal": "seasonal_512_path",
            "year": "year_512_path",
        },
        mask_source_column="emit_mask_512_path",
        metadata_columns=_time_metadata(
            product_id=True,
            overpass_key=True,
            cloud_cover=True,
        ),
        source_event_group_column="event_group_id",
        declared_all6_column="has_all6_512",
    ),
    "s5p": SensorSpec(
        name="s5p",
        source_path_columns={
            "t0": "t0_raw_path",
            "prev1": "prev1_raw_path",
            "prev2": "prev2_raw_path",
            "prev3": "prev3_raw_path",
            "seasonal": "seasonal_raw_path",
            "year": "year_raw_path",
        },
        mask_source_column=None,
        metadata_columns=_s5p_metadata(),
        source_plume_tif_column=None,
        source_kind="raw_netcdf",
    ),
}

CATALOGUE_COLUMNS = (
    "plume_id",
    "event_group_id",
    "datetime",
    "plume_latitude",
    "plume_longitude",
    "plume_bounds",
    "plume_tif",
    "country",
    "region",
    "place",
    "ipcc_sector",
    "gas",
    "instrument",
    "platform",
    "provider",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--s2-manifest", type=Path, default=DEFAULT_S2)
    parser.add_argument("--l89-manifest", type=Path, default=DEFAULT_L89)
    parser.add_argument("--emit-manifest", type=Path, default=DEFAULT_EMIT)
    parser.add_argument("--s5p-manifest", type=Path, default=DEFAULT_S5P)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "multisensor_6time_512_wide.csv",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "multisensor_6time_512_wide.audit.json",
    )
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--val-end", default="2025-06-09")
    parser.add_argument(
        "--min-sensors",
        type=int,
        default=1,
        help=(
            "Keep rows with at least this many complete six-timepoint sensor "
            "sources. Use 0 to retain catalogue rows with no complete sensor."
        ),
    )
    parser.add_argument(
        "--stat-workers",
        type=int,
        default=32,
        help="Threads used for exact-case os.path.isfile checks.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "first"),
        default="error",
        help=(
            "Conflicting plume_id duplicates fail by default. 'first' is an "
            "explicit deterministic escape hatch and remains fully audited."
        ),
    )
    parser.add_argument(
        "--allow-key-conflicts",
        action="store_true",
        help=(
            "Do not fail if a non-empty sensor event_group_id or UTC event date "
            "disagrees with the authoritative catalogue."
        ),
    )
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="Add SHA-256 content hashes for all input CSVs to the audit.",
    )
    parser.add_argument(
        "--plume-id",
        action="append",
        default=[],
        help=(
            "Debug/smoke-test filter; repeat for multiple plume IDs. All input "
            "tables are filtered to this exact set before joining."
        ),
    )
    parser.add_argument(
        "--max-rows-per-input",
        type=int,
        default=None,
        help="Debug-only read limit applied independently to every input CSV.",
    )
    return parser.parse_args()


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path, *, include_hash: bool) -> dict:
    stat = path.stat()
    result = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        result["sha256"] = _sha256(path)
    return result


def _read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _read_selected(
    path: Path,
    *,
    required: Iterable[str],
    optional: Iterable[str],
    plume_filter: set[str] | None,
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict]:
    header = _read_header(path)
    required_set = set(required)
    missing = sorted(required_set - set(header))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    selected_set = required_set | (set(optional) & set(header))
    selected = [column for column in header if column in selected_set]
    frame = pd.read_csv(
        path,
        usecols=selected,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        nrows=max_rows,
    )
    before_filter = len(frame)
    if plume_filter is not None:
        frame = frame[frame["plume_id"].isin(plume_filter)].copy()
    read_audit = {
        "columns_read": selected,
        "rows_read_before_plume_filter": int(before_filter),
        "rows_after_plume_filter": int(len(frame)),
    }
    return frame, read_audit


def _display_values(values: Sequence[object], limit: int = 5) -> list[str]:
    return sorted(str(value) for value in values)[:limit]


def _deduplicate(
    frame: pd.DataFrame,
    *,
    input_name: str,
    policy: str,
    sample_limit: int = 20,
) -> tuple[pd.DataFrame, dict]:
    empty_id = frame["plume_id"].eq("")
    if empty_id.any():
        raise ValueError(f"{input_name} contains {int(empty_id.sum())} empty plume_id")

    duplicate_mask = frame["plume_id"].duplicated(keep=False)
    duplicate_rows = frame.loc[duplicate_mask]
    conflict_samples: list[dict] = []
    exact_groups = 0
    conflicting_groups = 0
    if not duplicate_rows.empty:
        compare_columns = [
            column for column in frame.columns if column != "plume_id"
        ]
        for plume_id, group in duplicate_rows.groupby(
            "plume_id", sort=True, observed=True
        ):
            conflicts = [
                column
                for column in compare_columns
                if group[column].nunique(dropna=False) > 1
            ]
            if conflicts:
                conflicting_groups += 1
                if len(conflict_samples) < sample_limit:
                    conflict_samples.append(
                        {
                            "plume_id": str(plume_id),
                            "conflicting_columns": conflicts,
                            "values": {
                                column: _display_values(
                                    group[column].drop_duplicates().tolist()
                                )
                                for column in conflicts
                            },
                        }
                    )
            else:
                exact_groups += 1

    duplicate_group_count = int(duplicate_rows["plume_id"].nunique())
    audit = {
        "input_rows": int(len(frame)),
        "unique_plume_ids": int(frame["plume_id"].nunique()),
        "rows_in_duplicate_groups": int(len(duplicate_rows)),
        "duplicate_plume_ids": duplicate_group_count,
        "excess_rows": int(len(frame) - frame["plume_id"].nunique()),
        "exact_duplicate_plume_ids": int(exact_groups),
        "conflicting_duplicate_plume_ids": int(conflicting_groups),
        "conflict_samples": conflict_samples,
        "policy": policy,
    }
    if conflicting_groups and policy == "error":
        raise ValueError(
            f"{input_name} has {conflicting_groups} conflicting plume_id "
            f"duplicate groups; sample={conflict_samples[:3]}"
        )
    result = frame.drop_duplicates("plume_id", keep="first").copy()
    return result, audit


def _source_base_columns(spec: SensorSpec) -> list[str]:
    columns = [
        spec.source_event_time_column,
        spec.source_latitude_column,
        spec.source_longitude_column,
        spec.source_bounds_column,
    ]
    for optional in (
        spec.source_event_group_column,
        spec.source_plume_tif_column,
        spec.declared_all6_column,
    ):
        if optional:
            columns.append(optional)
    return list(dict.fromkeys(columns))


def _read_catalogue(
    path: Path,
    *,
    plume_filter: set[str] | None,
    max_rows: int | None,
    duplicate_policy: str,
) -> tuple[pd.DataFrame, dict]:
    required = (
        "plume_id",
        "event_group_id",
        "datetime",
        "plume_latitude",
        "plume_longitude",
    )
    frame, read_audit = _read_selected(
        path,
        required=required,
        optional=CATALOGUE_COLUMNS,
        plume_filter=plume_filter,
        max_rows=max_rows,
    )
    frame, duplicate_audit = _deduplicate(
        frame,
        input_name="catalogue",
        policy=duplicate_policy,
    )
    frame = frame.rename(columns={"datetime": "event_time"})
    for column in CATALOGUE_COLUMNS:
        target = "event_time" if column == "datetime" else column
        if target not in frame:
            frame[target] = ""
    frame["in_catalogue"] = True
    audit = {
        **read_audit,
        "duplicate_consistency": duplicate_audit,
    }
    return frame, audit


def _read_sensor(
    path: Path,
    spec: SensorSpec,
    *,
    plume_filter: set[str] | None,
    max_rows: int | None,
    duplicate_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = ["plume_id", *spec.source_path_columns.values()]
    optional = [
        *_source_base_columns(spec),
        *spec.metadata_columns.keys(),
    ]
    if spec.mask_source_column:
        optional.append(spec.mask_source_column)
    raw, read_audit = _read_selected(
        path,
        required=required,
        optional=optional,
        plume_filter=plume_filter,
        max_rows=max_rows,
    )
    raw, duplicate_audit = _deduplicate(
        raw,
        input_name=spec.name,
        policy=duplicate_policy,
    )

    output = pd.DataFrame({"plume_id": raw["plume_id"]})
    output[f"in_{spec.name}_manifest"] = True
    output[f"{spec.name}_source_kind"] = spec.source_kind
    for timepoint, source_column in spec.source_path_columns.items():
        output[f"{spec.name}_{timepoint}_path"] = raw[source_column]
    if spec.mask_source_column:
        output[f"{spec.name}_mask_path"] = raw.get(
            spec.mask_source_column, ""
        )

    base_mapping = {
        spec.source_event_time_column: "event_time",
        spec.source_latitude_column: "plume_latitude",
        spec.source_longitude_column: "plume_longitude",
        spec.source_bounds_column: "plume_bounds",
    }
    if spec.source_event_group_column:
        base_mapping[spec.source_event_group_column] = "event_group_id"
    if spec.source_plume_tif_column:
        base_mapping[spec.source_plume_tif_column] = "plume_tif"
    for source_column, common_name in base_mapping.items():
        output[f"{spec.name}_source_{common_name}"] = raw.get(source_column, "")

    if spec.declared_all6_column:
        output[f"{spec.name}_declared_all6"] = raw.get(
            spec.declared_all6_column, ""
        )
    for source_column, destination_suffix in spec.metadata_columns.items():
        if source_column in raw:
            output[f"{spec.name}_{destination_suffix}"] = raw[source_column]

    audit = {
        **read_audit,
        "source_kind": spec.source_kind,
        "source_path_columns": dict(spec.source_path_columns),
        "mask_source_column": spec.mask_source_column,
        "duplicate_consistency": duplicate_audit,
    }
    return raw, output, audit


def _utc_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _cross_source_consistency(
    catalogue: pd.DataFrame,
    source: pd.DataFrame,
    spec: SensorSpec,
    *,
    coordinate_tolerance: float = 1e-6,
    sample_limit: int = 20,
) -> dict:
    catalogue_compare = catalogue[
        [
            "plume_id",
            "event_group_id",
            "event_time",
            "plume_latitude",
            "plume_longitude",
        ]
    ].rename(
        columns={
            "event_group_id": "_catalogue_event_group_id",
            "event_time": "_catalogue_event_time",
            "plume_latitude": "_catalogue_plume_latitude",
            "plume_longitude": "_catalogue_plume_longitude",
        }
    )
    source_mapping = {
        spec.source_event_time_column: "_source_event_time",
        spec.source_latitude_column: "_source_plume_latitude",
        spec.source_longitude_column: "_source_plume_longitude",
    }
    if spec.source_event_group_column:
        source_mapping[
            spec.source_event_group_column
        ] = "_source_event_group_id"
    source_columns = [
        "plume_id",
        *(column for column in source_mapping if column in source),
    ]
    source_compare = source[source_columns].rename(columns=source_mapping)
    joined = source_compare.merge(
        catalogue_compare,
        on="plume_id",
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    matched = joined[joined["_merge"].eq("both")].copy()
    orphan_ids = joined.loc[joined["_merge"].ne("both"), "plume_id"].astype(str)

    result: dict[str, object] = {
        "sensor_rows": int(len(source)),
        "matched_catalogue_rows": int(len(matched)),
        "not_in_catalogue_rows": int(len(orphan_ids)),
        "not_in_catalogue_sample": sorted(orphan_ids.tolist())[:sample_limit],
    }

    group_mismatch_ids: list[str] = []
    if spec.source_event_group_column and spec.source_event_group_column in source:
        source_group = matched["_source_event_group_id"]
        catalogue_group = matched["_catalogue_event_group_id"]
        nonempty = source_group.ne("")
        mismatch = nonempty & source_group.ne(catalogue_group)
        group_mismatch_ids = sorted(
            matched.loc[mismatch, "plume_id"].astype(str).tolist()
        )
        result["event_group_id"] = {
            "source_nonempty_rows": int(nonempty.sum()),
            "source_empty_rows": int((~nonempty).sum()),
            "mismatch_rows": int(mismatch.sum()),
            "mismatch_sample": group_mismatch_ids[:sample_limit],
        }
    else:
        result["event_group_id"] = {
            "source_column_present": False,
            "catalogue_used_as_authority": True,
        }

    source_time = _utc_series(matched["_source_event_time"])
    catalogue_time = _utc_series(matched["_catalogue_event_time"])
    comparable = source_time.notna() & catalogue_time.notna()
    timestamp_mismatch = comparable & source_time.ne(catalogue_time)
    date_mismatch = comparable & source_time.dt.date.ne(catalogue_time.dt.date)
    date_mismatch_ids = sorted(
        matched.loc[date_mismatch, "plume_id"].astype(str).tolist()
    )
    result["event_time"] = {
        "source_invalid_or_empty_rows": int(source_time.isna().sum()),
        "catalogue_invalid_or_empty_rows": int(catalogue_time.isna().sum()),
        "exact_timestamp_mismatch_rows": int(timestamp_mismatch.sum()),
        "utc_date_mismatch_rows": int(date_mismatch.sum()),
        "utc_date_mismatch_sample": date_mismatch_ids[:sample_limit],
    }

    for coordinate in ("plume_latitude", "plume_longitude"):
        source_coordinate = pd.to_numeric(
            matched[f"_source_{coordinate}"], errors="coerce"
        )
        catalogue_coordinate = pd.to_numeric(
            matched[f"_catalogue_{coordinate}"], errors="coerce"
        )
        delta = (source_coordinate - catalogue_coordinate).abs()
        mismatch = delta.gt(coordinate_tolerance)
        result[coordinate] = {
            "tolerance": coordinate_tolerance,
            "mismatch_rows": int(mismatch.sum()),
            "max_abs_delta": (
                None if delta.dropna().empty else float(delta.max())
            ),
            "mismatch_sample": sorted(
                matched.loc[mismatch, "plume_id"].astype(str).tolist()
            )[:sample_limit],
        }

    result["split_key_conflict_rows"] = int(
        len(set(group_mismatch_ids) | set(date_mismatch_ids))
    )
    return result


def _path_prefix(path: str, components: int = 4) -> str:
    parts = PurePosixPath(path).parts
    if not parts:
        return ""
    return str(PurePosixPath(*parts[:components]))


class ExactPathChecker:
    """Case-preserving, exact-path file checks shared across all sensors."""

    def __init__(self, workers: int):
        self.workers = workers
        self.exists: dict[str, bool] = {}
        self.errors: dict[str, str] = {}

    @staticmethod
    def _check(path: str) -> tuple[str, bool, str | None]:
        try:
            return path, os.path.isfile(path), None
        except OSError as exc:
            return path, False, f"{type(exc).__name__}: {exc}"

    def populate(self, paths: Iterable[str]) -> None:
        unique = sorted(
            {
                path
                for path in paths
                if _nonempty(path) and path not in self.exists
            }
        )
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            for path, exists, error in executor.map(self._check, unique):
                self.exists[path] = exists
                if error is not None:
                    self.errors[path] = error

    def flags(self, values: pd.Series) -> pd.Series:
        return values.map(
            lambda value: bool(self.exists.get(value, False))
            if _nonempty(value)
            else False
        ).astype(bool)


def _path_column_audit(
    values: pd.Series,
    flags: pd.Series,
    *,
    missing_sample_limit: int = 10,
) -> dict:
    nonempty = values.map(_nonempty)
    exact_path_flags: dict[str, bool] = {}
    for value, exists in zip(values.tolist(), flags.tolist()):
        if _nonempty(value):
            exact_path_flags[value] = bool(exists)
    missing_paths = sorted(
        path for path, exists in exact_path_flags.items() if not exists
    )
    return {
        "rows_nonempty": int(nonempty.sum()),
        "rows_existing": int((nonempty & flags).sum()),
        "rows_missing": int((nonempty & ~flags).sum()),
        "unique_paths": int(len(exact_path_flags)),
        "unique_paths_existing": int(sum(exact_path_flags.values())),
        "unique_paths_missing": int(len(missing_paths)),
        "missing_path_sample_exact_case": missing_paths[:missing_sample_limit],
    }


def _add_path_flags(
    sensor_frames: Mapping[str, pd.DataFrame],
    *,
    workers: int,
) -> tuple[dict[str, pd.DataFrame], dict, ExactPathChecker]:
    all_paths: list[str] = []
    for sensor, frame in sensor_frames.items():
        spec = SENSOR_SPECS[sensor]
        for timepoint in TIMEPOINTS:
            all_paths.extend(frame[f"{sensor}_{timepoint}_path"].tolist())
        if spec.mask_source_column:
            all_paths.extend(frame[f"{sensor}_mask_path"].tolist())

    checker = ExactPathChecker(workers)
    checker.populate(all_paths)
    output_frames: dict[str, pd.DataFrame] = {}
    audit: dict[str, dict] = {}
    for sensor, original in sensor_frames.items():
        spec = SENSOR_SPECS[sensor]
        frame = original.copy()
        path_audit: dict[str, dict] = {}
        existence_columns: list[str] = []
        nonempty_columns: list[pd.Series] = []
        for timepoint in TIMEPOINTS:
            path_column = f"{sensor}_{timepoint}_path"
            exists_column = f"{sensor}_{timepoint}_exists"
            frame[exists_column] = checker.flags(frame[path_column])
            existence_columns.append(exists_column)
            nonempty_columns.append(frame[path_column].map(_nonempty))
            path_audit[timepoint] = _path_column_audit(
                frame[path_column],
                frame[exists_column],
            )
        frame[f"{sensor}_six_paths_nonempty"] = pd.concat(
            nonempty_columns, axis=1
        ).all(axis=1)
        frame[f"has_{sensor}"] = frame[existence_columns].all(axis=1)
        frame[f"{sensor}_existing_timepoints"] = frame[
            existence_columns
        ].sum(axis=1)

        if spec.mask_source_column:
            frame[f"{sensor}_mask_exists"] = checker.flags(
                frame[f"{sensor}_mask_path"]
            )
            frame[f"{sensor}_crop_ready"] = (
                frame[f"has_{sensor}"] & frame[f"{sensor}_mask_exists"]
            )
            path_audit["mask"] = _path_column_audit(
                frame[f"{sensor}_mask_path"],
                frame[f"{sensor}_mask_exists"],
            )
        else:
            frame[f"{sensor}_crop_ready"] = frame[f"has_{sensor}"]

        paths = [
            value
            for timepoint in TIMEPOINTS
            for value in frame[f"{sensor}_{timepoint}_path"].tolist()
            if _nonempty(value)
        ]
        if spec.mask_source_column:
            paths.extend(
                value
                for value in frame[f"{sensor}_mask_path"].tolist()
                if _nonempty(value)
            )
        prefix_counts = Counter(_path_prefix(path) for path in paths)
        audit[sensor] = {
            "rows_in_manifest": int(len(frame)),
            "source_kind": spec.source_kind,
            "path_columns": path_audit,
            "rows_with_six_nonempty_paths": int(
                frame[f"{sensor}_six_paths_nonempty"].sum()
            ),
            "rows_with_six_existing_paths": int(frame[f"has_{sensor}"].sum()),
            "rows_crop_ready": int(frame[f"{sensor}_crop_ready"].sum()),
            "exact_case_prefix_counts": {
                key: int(prefix_counts[key]) for key in sorted(prefix_counts)
            },
        }
        output_frames[sensor] = frame

    casefold_variants: defaultdict[str, set[str]] = defaultdict(set)
    for path in checker.exists:
        casefold_variants[path.casefold()].add(path)
    collisions = [
        sorted(variants)
        for variants in casefold_variants.values()
        if len(variants) > 1
    ]
    collisions.sort()
    audit["_global"] = {
        "unique_exact_path_strings_checked": int(len(checker.exists)),
        "existing_exact_path_strings": int(sum(checker.exists.values())),
        "missing_exact_path_strings": int(
            len(checker.exists) - sum(checker.exists.values())
        ),
        "stat_error_count": int(len(checker.errors)),
        "stat_error_sample": [
            {"path": path, "error": checker.errors[path]}
            for path in sorted(checker.errors)[:10]
        ],
        "casefold_collision_groups": int(len(collisions)),
        "casefold_collision_sample": collisions[:10],
        "path_strings_preserved_without_case_or_path_normalization": True,
        "existence_primitive": "os.path.isfile(exact_input_string)",
    }
    return output_frames, audit, checker


def _coalesce_nonempty(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.Series, pd.Series]:
    values = pd.Series("", index=frame.index, dtype=object)
    sources = pd.Series("", index=frame.index, dtype=object)
    for column in columns:
        if column not in frame:
            continue
        candidate = frame[column].fillna("").astype(str)
        take = values.eq("") & candidate.ne("")
        values.loc[take] = candidate.loc[take]
        sources.loc[take] = column
    return values, sources


def _derive_event_group(plume_ids: pd.Series) -> pd.Series:
    return plume_ids.astype(str).str.replace(
        r"-[A-Za-z0-9]+$", "", regex=True
    )


def _assign_global_split(
    frame: pd.DataFrame,
    *,
    train_end: date,
    val_end: date,
) -> tuple[pd.DataFrame, dict]:
    result = frame.copy()
    event_timestamp = _utc_series(result["event_time"])
    invalid = event_timestamp.isna()
    if invalid.any():
        sample = sorted(result.loc[invalid, "plume_id"].astype(str).tolist())[:20]
        raise ValueError(
            f"{int(invalid.sum())} rows have invalid event_time; sample={sample}"
        )
    result["_event_timestamp"] = event_timestamp
    result["event_time"] = event_timestamp.map(lambda value: value.isoformat())
    result["event_date"] = event_timestamp.dt.strftime("%Y-%m-%d")

    group_dates = result.groupby("event_group_id", observed=True)[
        "event_date"
    ].agg(["min", "max", "nunique"])
    conflicting = group_dates[group_dates["nunique"].gt(1)]
    if not conflicting.empty:
        raise ValueError(
            f"{len(conflicting)} event_group_id values span multiple UTC dates; "
            f"sample={conflicting.head(20).index.astype(str).tolist()}"
        )

    group_date = pd.to_datetime(group_dates["min"], errors="raise").dt.date
    group_split = pd.Series("test", index=group_dates.index, dtype=object)
    group_split.loc[group_date <= train_end] = "train"
    group_split.loc[
        (group_date > train_end) & (group_date <= val_end)
    ] = "val"
    result["split"] = result["event_group_id"].map(group_split)

    timestamp_counts = result.groupby(
        "event_group_id", observed=True
    )["_event_timestamp"].nunique()
    split_event_sets = {
        split: set(result.loc[result["split"].eq(split), "event_group_id"])
        for split in ("train", "val", "test")
    }
    audit = {
        "policy": (
            f"train <= {train_end.isoformat()}; "
            f"val <= {val_end.isoformat()}; else test"
        ),
        "train_end_inclusive": train_end.isoformat(),
        "val_end_inclusive": val_end.isoformat(),
        "timezone": "UTC",
        "assignment_unit": "event_group_id",
        "event_groups": int(len(group_dates)),
        "event_groups_spanning_multiple_utc_dates": int(len(conflicting)),
        "event_groups_with_multiple_exact_timestamps_same_date": int(
            timestamp_counts.gt(1).sum()
        ),
        "rows_by_split": {
            split: int(result["split"].eq(split).sum())
            for split in ("train", "val", "test")
        },
        "event_groups_by_split": {
            split: int(result.loc[result["split"].eq(split), "event_group_id"].nunique())
            for split in ("train", "val", "test")
        },
        "pairwise_event_overlap": {
            "train_val": int(
                len(split_event_sets["train"] & split_event_sets["val"])
            ),
            "train_test": int(
                len(split_event_sets["train"] & split_event_sets["test"])
            ),
            "val_test": int(
                len(split_event_sets["val"] & split_event_sets["test"])
            ),
        },
    }
    return result, audit


def _counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def _summarize_output(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "plume_ids": int(frame["plume_id"].nunique()),
        "event_groups": int(frame["event_group_id"].nunique()),
        "rows_by_split": _counts(frame["split"]),
        "event_groups_by_split": {
            split: int(
                frame.loc[frame["split"].eq(split), "event_group_id"].nunique()
            )
            for split in ("train", "val", "test")
        },
        "available_sensor_combination_rows": _counts(
            frame["available_sensors"].replace("", "none")
        ),
        "crop_ready_sensor_combination_rows": _counts(
            frame["crop_ready_sensors"].replace("", "none")
        ),
        "complete_six_timepoint_rows_by_sensor": {
            sensor: int(frame[f"has_{sensor}"].sum())
            for sensor in SENSOR_ORDER
        },
        "crop_ready_rows_by_sensor": {
            sensor: int(frame[f"{sensor}_crop_ready"].sum())
            for sensor in SENSOR_ORDER
        },
    }


def _ordered_columns(frame: pd.DataFrame) -> list[str]:
    base = [
        "plume_id",
        "event_group_id",
        "event_time",
        "event_date",
        "split",
        "plume_latitude",
        "plume_longitude",
        "plume_bounds",
        "plume_tif",
        "country",
        "region",
        "place",
        "ipcc_sector",
        "gas",
        "instrument",
        "platform",
        "provider",
        "in_catalogue",
        "event_group_id_source",
        "event_time_source",
        "num_available_sensors",
        "available_sensors",
        "has_any_sensor",
        "num_crop_ready_sensors",
        "crop_ready_sensors",
    ]
    summary: list[str] = []
    detail: list[str] = []
    for sensor in SENSOR_ORDER:
        summary.extend(
            [
                f"in_{sensor}_manifest",
                f"has_{sensor}",
                f"{sensor}_existing_timepoints",
                f"{sensor}_six_paths_nonempty",
                f"{sensor}_crop_ready",
                f"{sensor}_source_kind",
            ]
        )
        optional_summary = [
            f"{sensor}_mask_path",
            f"{sensor}_mask_exists",
            f"{sensor}_declared_all6",
        ]
        summary.extend(column for column in optional_summary if column in frame)
        for timepoint in TIMEPOINTS:
            detail.extend(
                [
                    f"{sensor}_{timepoint}_path",
                    f"{sensor}_{timepoint}_exists",
                ]
            )
            metadata_prefix = f"{sensor}_{timepoint}_"
            detail.extend(
                sorted(
                    column
                    for column in frame.columns
                    if column.startswith(metadata_prefix)
                    and column
                    not in {
                        f"{sensor}_{timepoint}_path",
                        f"{sensor}_{timepoint}_exists",
                    }
                )
            )
        detail.extend(
            sorted(
                column
                for column in frame.columns
                if column.startswith(f"{sensor}_source_")
            )
        )
        detail.extend(
            sorted(
                column
                for column in frame.columns
                if column
                in {
                    f"{sensor}_center_status",
                    f"{sensor}_center_note",
                    f"{sensor}_nearest_iy",
                    f"{sensor}_nearest_ix",
                    f"{sensor}_pos_centers",
                }
            )
        )
    ordered = []
    seen: set[str] = set()
    for column in [*base, *summary, *detail, *sorted(frame.columns)]:
        if column in frame and column not in seen and not column.startswith("_"):
            ordered.append(column)
            seen.add(column)
    return ordered


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build(args: argparse.Namespace) -> dict:
    train_end = date.fromisoformat(args.train_end)
    val_end = date.fromisoformat(args.val_end)
    if train_end >= val_end:
        raise ValueError("--train-end must be earlier than --val-end")
    if not 0 <= args.min_sensors <= len(SENSOR_ORDER):
        raise ValueError("--min-sensors must be between 0 and 4")
    if args.stat_workers < 1:
        raise ValueError("--stat-workers must be positive")
    if args.max_rows_per_input is not None and args.max_rows_per_input < 1:
        raise ValueError("--max-rows-per-input must be positive")

    input_paths = {
        "catalogue": args.catalogue,
        "s2": args.s2_manifest,
        "l89": args.l89_manifest,
        "emit": args.emit_manifest,
        "s5p": args.s5p_manifest,
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} input does not exist: {path}")

    plume_filter = set(args.plume_id) if args.plume_id else None
    input_audit = {
        name: _file_fingerprint(path, include_hash=args.hash_inputs)
        for name, path in input_paths.items()
    }

    catalogue, catalogue_audit = _read_catalogue(
        args.catalogue,
        plume_filter=plume_filter,
        max_rows=args.max_rows_per_input,
        duplicate_policy=args.duplicate_policy,
    )
    input_audit["catalogue"].update(catalogue_audit)

    raw_sensors: dict[str, pd.DataFrame] = {}
    sensor_frames: dict[str, pd.DataFrame] = {}
    cross_source: dict[str, dict] = {}
    for sensor in SENSOR_ORDER:
        spec = SENSOR_SPECS[sensor]
        raw, prepared, sensor_audit = _read_sensor(
            input_paths[sensor],
            spec,
            plume_filter=plume_filter,
            max_rows=args.max_rows_per_input,
            duplicate_policy=args.duplicate_policy,
        )
        raw_sensors[sensor] = raw
        sensor_frames[sensor] = prepared
        input_audit[sensor].update(sensor_audit)
        cross_source[sensor] = _cross_source_consistency(
            catalogue,
            raw,
            spec,
        )

    conflict_sensors = {
        sensor: audit["split_key_conflict_rows"]
        for sensor, audit in cross_source.items()
        if audit["split_key_conflict_rows"]
    }
    if conflict_sensors and not args.allow_key_conflicts:
        raise ValueError(
            "sensor/catalogue split-key conflicts detected; "
            f"use --allow-key-conflicts only after review: {conflict_sensors}"
        )

    sensor_frames, path_audit, _ = _add_path_flags(
        sensor_frames,
        workers=args.stat_workers,
    )

    wide = catalogue.copy()
    for sensor in SENSOR_ORDER:
        wide = wide.merge(
            sensor_frames[sensor],
            on="plume_id",
            how="outer",
            sort=False,
            validate="one_to_one",
        )

    membership_columns = ["in_catalogue"] + [
        f"in_{sensor}_manifest" for sensor in SENSOR_ORDER
    ]
    for column in membership_columns:
        wide[column] = wide[column].fillna(False).astype(bool)

    group_candidates = ["event_group_id"] + [
        f"{sensor}_source_event_group_id" for sensor in SENSOR_ORDER
    ]
    event_group, event_group_source = _coalesce_nonempty(
        wide, group_candidates
    )
    derived = _derive_event_group(wide["plume_id"])
    use_derived = event_group.eq("")
    event_group.loc[use_derived] = derived.loc[use_derived]
    event_group_source.loc[use_derived] = "derived_from_plume_id"
    if event_group.eq("").any():
        raise ValueError("unable to determine event_group_id for every row")
    wide["event_group_id"] = event_group
    wide["event_group_id_source"] = event_group_source

    time_candidates = ["event_time"] + [
        f"{sensor}_source_event_time" for sensor in SENSOR_ORDER
    ]
    wide["event_time"], wide["event_time_source"] = _coalesce_nonempty(
        wide, time_candidates
    )

    for base_column in (
        "plume_latitude",
        "plume_longitude",
        "plume_bounds",
        "plume_tif",
    ):
        candidates = [base_column] + [
            f"{sensor}_source_{base_column}" for sensor in SENSOR_ORDER
        ]
        wide[base_column], _ = _coalesce_nonempty(wide, candidates)
    for column in CATALOGUE_COLUMNS:
        target = "event_time" if column == "datetime" else column
        if target not in wide:
            wide[target] = ""

    wide, temporal_audit_before_filter = _assign_global_split(
        wide,
        train_end=train_end,
        val_end=val_end,
    )

    for sensor in SENSOR_ORDER:
        boolean_columns = [
            f"has_{sensor}",
            f"{sensor}_six_paths_nonempty",
            f"{sensor}_crop_ready",
        ]
        if f"{sensor}_mask_exists" in wide:
            boolean_columns.append(f"{sensor}_mask_exists")
        for column in boolean_columns:
            wide[column] = wide[column].fillna(False).astype(bool)
        wide[f"{sensor}_existing_timepoints"] = (
            pd.to_numeric(
                wide[f"{sensor}_existing_timepoints"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    wide["num_available_sensors"] = sum(
        wide[f"has_{sensor}"].astype(int) for sensor in SENSOR_ORDER
    )
    wide["has_any_sensor"] = wide["num_available_sensors"].gt(0)
    wide["available_sensors"] = wide.apply(
        lambda row: "|".join(
            sensor for sensor in SENSOR_ORDER if bool(row[f"has_{sensor}"])
        ),
        axis=1,
    )
    wide["num_crop_ready_sensors"] = sum(
        wide[f"{sensor}_crop_ready"].astype(int) for sensor in SENSOR_ORDER
    )
    wide["crop_ready_sensors"] = wide.apply(
        lambda row: "|".join(
            sensor
            for sensor in SENSOR_ORDER
            if bool(row[f"{sensor}_crop_ready"])
        ),
        axis=1,
    )

    pre_filter_summary = _summarize_output(wide)
    output = wide[
        wide["num_available_sensors"].ge(args.min_sensors)
    ].copy()
    if output.empty:
        raise ValueError(
            f"no rows remain after --min-sensors={args.min_sensors}"
        )
    output = output.sort_values(
        ["event_time", "event_group_id", "plume_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    output = output[_ordered_columns(output)]
    output_summary = _summarize_output(output)

    _atomic_write_csv(output, args.output_csv)
    output_sha256 = _sha256(args.output_csv)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "inputs": input_audit,
        "debug_limits": {
            "plume_id_filter": sorted(plume_filter) if plume_filter else [],
            "max_rows_per_input": args.max_rows_per_input,
            "is_debug_limited": bool(
                plume_filter is not None or args.max_rows_per_input is not None
            ),
        },
        "duplicate_policy": args.duplicate_policy,
        "catalogue_authority": {
            "event_group_id": True,
            "event_time": True,
            "orphan_fallback": (
                "non-empty sensor value, then event_group_id derived by "
                "removing the final plume suffix"
            ),
        },
        "cross_source_consistency": cross_source,
        "path_existence": path_audit,
        "temporal_split_before_sensor_filter": temporal_audit_before_filter,
        "output_filter": {
            "min_complete_six_timepoint_sensors": args.min_sensors,
            "sensor_availability_definition": (
                "all six exact source path strings pass os.path.isfile"
            ),
            "raster_crop_ready_definition": (
                "six image paths exist and sensor-aligned mask exists"
            ),
            "s5p_crop_ready_definition": "all six raw NetCDF paths exist",
        },
        "outer_join_before_filter": pre_filter_summary,
        "output": {
            **output_summary,
            "csv": str(args.output_csv),
            "audit_json": str(args.audit_json),
            "csv_sha256": output_sha256,
            "columns": list(output.columns),
        },
    }
    _atomic_write_json(audit, args.audit_json)

    return {
        "schema_version": SCHEMA_VERSION,
        "output_csv": str(args.output_csv),
        "audit_json": str(args.audit_json),
        "rows": output_summary["rows"],
        "event_groups": output_summary["event_groups"],
        "rows_by_split": output_summary["rows_by_split"],
        "complete_six_timepoint_rows_by_sensor": output_summary[
            "complete_six_timepoint_rows_by_sensor"
        ],
        "crop_ready_rows_by_sensor": output_summary[
            "crop_ready_rows_by_sensor"
        ],
        "csv_sha256": output_sha256,
    }


def main() -> None:
    result = build(_parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
