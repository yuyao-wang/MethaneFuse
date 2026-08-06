#!/usr/bin/env python3
"""Read-only manifest-v2 provenance and missing-state audit.

This tool deliberately separates three metadata layers:

1. the full v1 wide source manifest (CSV fields only);
2. optional crop-task/smoke sidecars (already-computed query diagnostics);
3. an explicitly requested, tiny EMIT embedded-provenance probe.

The EMIT probe opens GeoTIFF headers/tags but never calls ``read`` on a raster.
It then reads only the small ``granule_id.npy`` and ``source_nc.npy`` members
from the tagged NPZ archive.  It never loads reflectance, latitude, longitude,
or NetCDF arrays.  Inputs and legacy pipeline files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
SENSORS = ("s2", "l89", "emit", "s5p")
MISSING_TOKENS = frozenset({"nan", "none", "null", "<na>", "nat"})
SCHEMA_VERSION = "methanefuse_manifest_v2_provenance_audit_v1"

HERE = Path(__file__).resolve().parent
DEFAULT_WIDE_MANIFEST = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/manifests/"
    "multisensor_6time_512_wide.csv"
)
DEFAULT_OLD_EMIT_PIPELINE = Path(
    "/home/yuyao/methane_train/Upgrade_data_pipeline/code/"
    "EMIT_preprocess/emit_6time_preprocess.py"
)
DEFAULT_OUTPUT_JSON = HERE / "manifest_v2_provenance_pilot.audit.json"
DEFAULT_OUTPUT_CSV = HERE / "manifest_v2_provenance_pilot.emit_probe.csv"

TASK_V2_FIELDS = (
    "task_id",
    "plume_id",
    "event_group_id",
    "split",
    "label",
    "query_center_longitude",
    "query_center_latitude",
    "source_sensor_presence_mask",
    "query_usable_anytime_sensor_mask",
    "query_current_sensor_mask",
    "eligible_fused_current_classification",
    "execution_status",
    "missingness_policy_version",
    "coverage_config_sha256",
)
SENSOR_V2_SUFFIXES = (
    "source_path_time_mask",
    "query_any_valid_time_mask",
    "query_usable_time_mask",
    "query_unique_time_mask",
    "num_usable_times",
    "num_unique_usable_times",
    "current_usable",
    "ssl_current_eligible",
    "ssl_temporal_eligible",
)
SLOT_V2_SUFFIXES = (
    "source_path",
    "source_exists",
    "acquisition_time",
    "product_id",
    "observation_key",
    "alias_of",
    "crop_status",
    "valid_reference_pixels",
    "total_reference_pixels",
    "valid_reference_fraction",
    "valid_elements_by_channel",
    "crop_path",
    "validity_path",
    "crop_sha256",
    "validity_sha256",
)
PROBE_COLUMNS = (
    "plume_id",
    "timepoint",
    "manifest_path",
    "manifest_exists",
    "manifest_product_id",
    "manifest_acquisition_time",
    "tiff_header_status",
    "tiff_width",
    "tiff_height",
    "tiff_count",
    "source_npz",
    "source_npz_slot_token",
    "source_npz_slot_matches_manifest_slot",
    "npz_granule_id",
    "npz_source_nc",
    "manifest_product_matches_npz_granule",
    "npz_source_nc_matches_granule",
    "granule_manifest_time_delta_seconds",
    "provenance_status",
    "reason",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wide-manifest", type=Path, default=DEFAULT_WIDE_MANIFEST
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        action="append",
        default=[],
        help="Optional v1 task CSV used only for schema/header checks.",
    )
    parser.add_argument(
        "--validation-sidecar",
        type=Path,
        action="append",
        default=[],
        help="Optional existing smoke-validation CSV; no imagery is reread.",
    )
    parser.add_argument(
        "--emit-probe-plume-id",
        action="append",
        default=[],
        help=(
            "Explicit plume_id whose six EMIT TIFF headers and tagged NPZ "
            "scalar provenance members may be inspected."
        ),
    )
    parser.add_argument(
        "--old-emit-pipeline",
        type=Path,
        default=DEFAULT_OLD_EMIT_PIPELINE,
        help="Read-only legacy-code identity recorded in the audit.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--output-probe-csv", type=Path, default=DEFAULT_OUTPUT_CSV
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument(
        "--time-tolerance-seconds",
        type=float,
        default=300.0,
        help="Tolerance for EMIT product timestamp versus manifest time.",
    )
    parser.add_argument(
        "--max-scalar-member-bytes",
        type=int,
        default=1_048_576,
        help="Refuse to read an NPZ provenance member above this size.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _value_state(value: Any) -> str:
    text = _clean(value)
    if not text:
        return "empty"
    if text.casefold() in MISSING_TOKENS:
        return "literal_missing_token"
    return "value"


def _has_value(value: Any) -> bool:
    return _value_state(value) == "value"


def _bool_state(value: Any) -> str:
    state = _value_state(value)
    if state != "value":
        return state
    text = _clean(value).casefold()
    if text in {"1", "1.0", "true", "yes"}:
        return "true"
    if text in {"0", "0.0", "false", "no"}:
        return "false"
    return "invalid"


def _normalize_time(value: Any) -> str:
    text = _clean(value)
    if not _has_value(text):
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
        for form in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
            try:
                parsed = datetime.strptime(text, form)
                break
            except ValueError:
                continue
        if parsed is None:
            return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _manifest_identity(
    row: Mapping[str, str], sensor: str, timepoint: str
) -> tuple[str, str]:
    prefix = f"{sensor}_{timepoint}_"
    product = _clean(row.get(prefix + "product_id", ""))
    overpass = _clean(row.get(prefix + "overpass_key", ""))
    acquisition = _normalize_time(row.get(prefix + "image_time", ""))
    path = _clean(row.get(prefix + "path", ""))
    if _has_value(product):
        return "product_id", product
    if _has_value(overpass) and _has_value(acquisition):
        return "overpass_and_time", f"{overpass}|{acquisition}"
    if _has_value(overpass):
        return "overpass_key", overpass
    if _has_value(acquisition) and _has_value(path):
        return "time_and_exact_path", f"{acquisition}|{path}"
    if _has_value(path):
        return "exact_path_fallback", path
    return "missing", ""


def _primary_identity_metadata(
    row: Mapping[str, str], sensor: str, timepoint: str
) -> str:
    prefix = f"{sensor}_{timepoint}_"
    product = _clean(row.get(prefix + "product_id", ""))
    overpass = _clean(row.get(prefix + "overpass_key", ""))
    return product if _has_value(product) else overpass if _has_value(overpass) else ""


def _add_sample(
    samples: list[dict[str, Any]],
    value: dict[str, Any],
    limit: int,
) -> None:
    if len(samples) < limit:
        samples.append(value)


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def _compatibility_alias(field: str, available: set[str]) -> str:
    direct_aliases = {
        "source_sensor_presence_mask": "sensor_presence_mask",
    }
    if field in direct_aliases and direct_aliases[field] in available:
        return direct_aliases[field]
    for sensor in SENSORS:
        if field == f"{sensor}_source_path_time_mask":
            for candidate in (
                f"{sensor}_source_time_presence_mask",
                f"{sensor}_time_presence_mask",
            ):
                if candidate in available:
                    return candidate
        for timepoint in TIMEPOINTS:
            prefix = f"{sensor}_{timepoint}_"
            if not field.startswith(prefix):
                continue
            suffix = field[len(prefix) :]
            candidates = {
                "source_path": (prefix + "path",),
                "source_exists": (prefix + "exists",),
                "acquisition_time": (prefix + "image_time",),
                "product_id": (
                    prefix + "product_id",
                    prefix + "overpass_key",
                ),
            }.get(suffix, ())
            for candidate in candidates:
                if candidate in available:
                    return candidate
    return ""


def _schema_readiness(
    wide_header: Sequence[str], task_headers: Sequence[Sequence[str]]
) -> dict[str, Any]:
    available = set(wide_header)
    for header in task_headers:
        available.update(header)
    required = list(TASK_V2_FIELDS)
    required.extend(
        f"{sensor}_{suffix}"
        for sensor in SENSORS
        for suffix in SENSOR_V2_SUFFIXES
    )
    required.extend(
        f"{sensor}_{timepoint}_{suffix}"
        for sensor in SENSORS
        for timepoint in TIMEPOINTS
        for suffix in SLOT_V2_SUFFIXES
    )
    status: dict[str, dict[str, str]] = {}
    counts = Counter()
    for field in required:
        if field in available:
            item = {"status": "direct", "column": field}
        else:
            alias = _compatibility_alias(field, available)
            if alias:
                item = {"status": "compatibility_alias", "column": alias}
            else:
                item = {"status": "missing", "column": ""}
        status[field] = item
        counts[item["status"]] += 1
    missing = [field for field, item in status.items() if item["status"] == "missing"]
    return {
        "required_field_count": len(required),
        "status_counts": {key: int(counts[key]) for key in sorted(counts)},
        "ready_for_v2_execution": not missing,
        "fields": status,
        "missing_fields": missing,
        "interpretation": (
            "Compatibility aliases are source evidence, not permission to "
            "overload v1 presence masks with query usability."
        ),
    }


def _audit_wide_manifest(
    path: Path,
    *,
    max_rows: int,
    sample_limit: int,
    requested_probe_ids: set[str],
    time_tolerance_seconds: float,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], list[str]]:
    sensor_audits: dict[str, dict[str, Any]] = {}
    for sensor in SENSORS:
        sensor_audits[sensor] = {
            "represented_rows": 0,
            "slot_states": {
                timepoint: {
                    "path": Counter(),
                    "exists": Counter(),
                    "acquisition_time": Counter(),
                    "product_or_overpass_identity": Counter(),
                }
                for timepoint in TIMEPOINTS
            },
            "represented_slot_states": {
                timepoint: {
                    "path": Counter(),
                    "exists": Counter(),
                    "acquisition_time": Counter(),
                    "product_or_overpass_identity": Counter(),
                }
                for timepoint in TIMEPOINTS
            },
            "rows_with_manifest_identity_alias": 0,
            "manifest_identity_alias_groups": 0,
            "manifest_identity_extra_slots": 0,
            "rows_with_exact_path_alias": 0,
            "exact_path_alias_groups": 0,
            "exact_path_extra_slots": 0,
            "path_identity_conflict_groups": 0,
            "same_identity_multiple_time_groups": 0,
            "rows_with_product_time_mismatch": 0,
            "product_time_mismatch_slots": 0,
            "product_time_mismatch_by_slot": Counter(),
            "manifest_identity_alias_samples": [],
            "exact_path_alias_samples": [],
            "path_identity_conflict_samples": [],
            "same_identity_multiple_time_samples": [],
            "product_time_mismatch_samples": [],
        }

    rows = 0
    plume_ids: set[str] = set()
    event_groups: set[str] = set()
    split_groups: defaultdict[str, set[str]] = defaultdict(set)
    probe_rows: dict[str, dict[str, str]] = {}
    header: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        for row in reader:
            if max_rows > 0 and rows >= max_rows:
                break
            rows += 1
            plume_id = _clean(row.get("plume_id", ""))
            event_group = _clean(row.get("event_group_id", ""))
            split = _clean(row.get("split", ""))
            if plume_id:
                plume_ids.add(plume_id)
            if event_group:
                event_groups.add(event_group)
                split_groups[split].add(event_group)
            if plume_id in requested_probe_ids:
                probe_rows[plume_id] = dict(row)

            for sensor in SENSORS:
                audit = sensor_audits[sensor]
                represented = (
                    _bool_state(row.get(f"in_{sensor}_manifest", "")) == "true"
                    or any(
                        _has_value(row.get(f"{sensor}_{timepoint}_path", ""))
                        for timepoint in TIMEPOINTS
                    )
                )
                if represented:
                    audit["represented_rows"] += 1

                by_identity: defaultdict[str, list[str]] = defaultdict(list)
                by_path: defaultdict[str, list[str]] = defaultdict(list)
                identity_details: dict[str, tuple[str, str]] = {}
                acquisition_by_primary: defaultdict[str, set[str]] = defaultdict(set)
                identity_by_path: defaultdict[str, set[str]] = defaultdict(set)
                row_product_time_mismatch = False

                for timepoint in TIMEPOINTS:
                    prefix = f"{sensor}_{timepoint}_"
                    path_value = row.get(prefix + "path", "")
                    acquisition_value = row.get(prefix + "image_time", "")
                    product_or_overpass = _primary_identity_metadata(
                        row, sensor, timepoint
                    )
                    audit["slot_states"][timepoint]["path"][
                        _value_state(path_value)
                    ] += 1
                    audit["slot_states"][timepoint]["exists"][
                        _bool_state(row.get(prefix + "exists", ""))
                    ] += 1
                    audit["slot_states"][timepoint]["acquisition_time"][
                        _value_state(acquisition_value)
                    ] += 1
                    audit["slot_states"][timepoint][
                        "product_or_overpass_identity"
                    ][_value_state(product_or_overpass)] += 1
                    if represented:
                        audit["represented_slot_states"][timepoint]["path"][
                            _value_state(path_value)
                        ] += 1
                        audit["represented_slot_states"][timepoint]["exists"][
                            _bool_state(row.get(prefix + "exists", ""))
                        ] += 1
                        audit["represented_slot_states"][timepoint][
                            "acquisition_time"
                        ][_value_state(acquisition_value)] += 1
                        audit["represented_slot_states"][timepoint][
                            "product_or_overpass_identity"
                        ][_value_state(product_or_overpass)] += 1

                    method, identity = _manifest_identity(
                        row, sensor, timepoint
                    )
                    if identity:
                        by_identity[identity].append(timepoint)
                        identity_details[timepoint] = (method, identity)
                    path_text = _clean(path_value)
                    if _has_value(path_text):
                        by_path[path_text].append(timepoint)
                        if product_or_overpass:
                            identity_by_path[path_text].add(product_or_overpass)
                    if product_or_overpass:
                        normalized_time = _normalize_time(acquisition_value)
                        if normalized_time:
                            acquisition_by_primary[product_or_overpass].add(
                                normalized_time
                            )
                    if sensor == "emit":
                        product = _clean(row.get(prefix + "product_id", ""))
                        product_time = _emit_granule_time(product)
                        manifest_time = _parsed_time(
                            _clean(acquisition_value)
                        )
                        if product_time is not None and manifest_time is not None:
                            delta_seconds = abs(
                                (product_time - manifest_time).total_seconds()
                            )
                            if delta_seconds > time_tolerance_seconds:
                                row_product_time_mismatch = True
                                audit["product_time_mismatch_slots"] += 1
                                audit["product_time_mismatch_by_slot"][
                                    timepoint
                                ] += 1
                                _add_sample(
                                    audit["product_time_mismatch_samples"],
                                    {
                                        "plume_id": plume_id,
                                        "timepoint": timepoint,
                                        "product_id": product,
                                        "manifest_acquisition_time": _clean(
                                            acquisition_value
                                        ),
                                        "delta_seconds": delta_seconds,
                                    },
                                    sample_limit,
                                )

                identity_aliases = [
                    slots for slots in by_identity.values() if len(slots) > 1
                ]
                if identity_aliases:
                    audit["rows_with_manifest_identity_alias"] += 1
                    audit["manifest_identity_alias_groups"] += len(
                        identity_aliases
                    )
                    audit["manifest_identity_extra_slots"] += sum(
                        len(slots) - 1 for slots in identity_aliases
                    )
                    _add_sample(
                        audit["manifest_identity_alias_samples"],
                        {
                            "plume_id": plume_id,
                            "groups": identity_aliases,
                            "identity_details": {
                                slot: list(identity_details[slot])
                                for slots in identity_aliases
                                for slot in slots
                            },
                        },
                        sample_limit,
                    )

                path_aliases = [
                    slots for slots in by_path.values() if len(slots) > 1
                ]
                if path_aliases:
                    audit["rows_with_exact_path_alias"] += 1
                    audit["exact_path_alias_groups"] += len(path_aliases)
                    audit["exact_path_extra_slots"] += sum(
                        len(slots) - 1 for slots in path_aliases
                    )
                    _add_sample(
                        audit["exact_path_alias_samples"],
                        {"plume_id": plume_id, "groups": path_aliases},
                        sample_limit,
                    )

                path_conflicts = [
                    {
                        "path": source_path,
                        "identities": sorted(identities),
                        "slots": by_path[source_path],
                    }
                    for source_path, identities in identity_by_path.items()
                    if len(identities) > 1
                ]
                audit["path_identity_conflict_groups"] += len(path_conflicts)
                for conflict in path_conflicts:
                    _add_sample(
                        audit["path_identity_conflict_samples"],
                        {"plume_id": plume_id, **conflict},
                        sample_limit,
                    )

                time_conflicts = [
                    {
                        "identity": identity,
                        "acquisition_times": sorted(times),
                    }
                    for identity, times in acquisition_by_primary.items()
                    if len(times) > 1
                ]
                audit["same_identity_multiple_time_groups"] += len(
                    time_conflicts
                )
                for conflict in time_conflicts:
                    _add_sample(
                        audit["same_identity_multiple_time_samples"],
                        {"plume_id": plume_id, **conflict},
                        sample_limit,
                    )
                if row_product_time_mismatch:
                    audit["rows_with_product_time_mismatch"] += 1

    for sensor, audit in sensor_audits.items():
        for state_scope in ("slot_states", "represented_slot_states"):
            for timepoint in TIMEPOINTS:
                for field in (
                    "path",
                    "exists",
                    "acquisition_time",
                    "product_or_overpass_identity",
                ):
                    counter = audit[state_scope][timepoint][field]
                    audit[state_scope][timepoint][field] = {
                        key: int(counter[key]) for key in sorted(counter)
                    }
        mismatch_counter = audit["product_time_mismatch_by_slot"]
        audit["product_time_mismatch_by_slot"] = {
            key: int(mismatch_counter[key])
            for key in sorted(mismatch_counter)
        }

    overlaps = {
        "train_val": len(split_groups["train"] & split_groups["val"]),
        "train_test": len(split_groups["train"] & split_groups["test"]),
        "val_test": len(split_groups["val"] & split_groups["test"]),
    }
    result = {
        "input": _file_identity(path),
        "rows_read": rows,
        "max_rows": max_rows,
        "debug_limited": max_rows > 0,
        "emit_product_time_tolerance_seconds": time_tolerance_seconds,
        "unique_plume_ids": len(plume_ids),
        "unique_event_group_ids": len(event_groups),
        "event_groups_by_split": {
            key: len(split_groups[key]) for key in sorted(split_groups)
        },
        "pairwise_event_group_overlap": overlaps,
        "event_split_gate_pass": all(value == 0 for value in overlaps.values()),
        "sensors": sensor_audits,
    }
    return result, probe_rows, header


def _audit_task_manifests(
    paths: Sequence[Path],
) -> tuple[list[dict], list[list[str]]]:
    audits: list[dict] = []
    headers: list[list[str]] = []
    for path in paths:
        header = _header(path)
        headers.append(header)
        rows = 0
        task_ids: list[str] = []
        labels: Counter[str] = Counter()
        splits: Counter[str] = Counter()
        sensor_patterns: Counter[str] = Counter()
        split_groups: defaultdict[str, set[str]] = defaultdict(set)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                task_id = _clean(row.get("task_id", ""))
                if task_id:
                    task_ids.append(task_id)
                labels[_clean(row.get("label", "")) or "empty"] += 1
                split = _clean(row.get("split", "")) or "empty"
                splits[split] += 1
                sensor_patterns[
                    _clean(row.get("available_sensors", "")) or "none"
                ] += 1
                event_group = _clean(row.get("event_group_id", ""))
                if event_group:
                    split_groups[split].add(event_group)
        overlaps = {
            "train_val": len(split_groups["train"] & split_groups["val"]),
            "train_test": len(split_groups["train"] & split_groups["test"]),
            "val_test": len(split_groups["val"] & split_groups["test"]),
        }
        audits.append(
            {
                "input": _file_identity(path),
                "column_count": len(header),
                "rows": rows,
                "unique_task_ids": len(set(task_ids)),
                "duplicate_task_id_rows": len(task_ids) - len(set(task_ids)),
                "label_counts": {
                    key: int(labels[key]) for key in sorted(labels)
                },
                "split_counts": {
                    key: int(splits[key]) for key in sorted(splits)
                },
                "available_sensor_pattern_counts": {
                    key: int(sensor_patterns[key])
                    for key in sorted(sensor_patterns)
                },
                "pairwise_event_group_overlap": overlaps,
                "schema_version_column_present": "schema_version" in header,
                "task_id_present": "task_id" in header,
                "label_present": "label" in header,
                "legacy_sensor_presence_mask_present": (
                    "sensor_presence_mask" in header
                ),
                "explicit_v2_query_current_sensor_mask_present": (
                    "query_current_sensor_mask" in header
                ),
            }
        )
    return audits, headers


def _parse_json_object(value: Any) -> dict[str, Any]:
    text = _clean(value)
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed


def _mask_from_shapes(shapes: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for sensor, time_data in shapes.items():
        if not isinstance(time_data, Mapping):
            continue
        bits: list[str] = []
        for timepoint in TIMEPOINTS:
            item = time_data.get(timepoint, {})
            finite = item.get("finite_values", 0) if isinstance(item, Mapping) else 0
            try:
                bits.append("1" if int(finite) > 0 else "0")
            except (TypeError, ValueError):
                bits.append("0")
        result[str(sensor)] = "".join(bits)
    return result


def _audit_validation_sidecars(paths: Sequence[Path]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for path in paths:
        status_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        mask_patterns: defaultdict[str, Counter[str]] = defaultdict(Counter)
        mask_sources: Counter[str] = Counter()
        examples: list[dict[str, Any]] = []
        rows = 0
        known_regression: dict[str, Any] | None = None
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            for row in reader:
                rows += 1
                status_counts[_clean(row.get("status", "")) or "empty"] += 1
                label_counts[_clean(row.get("label", "")) or "empty"] += 1
                masks: dict[str, str] = {}
                explicit = _clean(row.get("raster_valid_time_masks", ""))
                if explicit:
                    masks = {
                        str(key): str(value)
                        for key, value in _parse_json_object(explicit).items()
                    }
                    mask_sources["raster_valid_time_masks"] += 1
                else:
                    shapes = _parse_json_object(
                        row.get("raster_time_shapes", "")
                    )
                    masks = _mask_from_shapes(shapes)
                    mask_sources["derived_from_finite_value_counts"] += 1
                for sensor, mask in masks.items():
                    mask_patterns[sensor][mask] += 1
                item = {
                    "task_id": _clean(row.get("task_id", "")),
                    "plume_id": _clean(row.get("plume_id", "")),
                    "label": _clean(row.get("label", "")),
                    "status": _clean(row.get("status", "")),
                    "reason": _clean(row.get("reason", "")),
                    "query_any_valid_masks": masks,
                }
                if len(examples) < 20:
                    examples.append(item)
                if item["plume_id"] == "emi20230405t190335p13004-A":
                    known_regression = {
                        **item,
                        "emit_expected": "010011",
                        "emit_matches_expected": masks.get("emit") == "010011",
                    }
        audits.append(
            {
                "input": _file_identity(path),
                "rows": rows,
                "columns": header,
                "status_counts": {
                    key: int(status_counts[key]) for key in sorted(status_counts)
                },
                "label_counts": {
                    key: int(label_counts[key]) for key in sorted(label_counts)
                },
                "mask_source_counts": {
                    key: int(mask_sources[key]) for key in sorted(mask_sources)
                },
                "query_any_valid_mask_patterns": {
                    sensor: {
                        key: int(mask_patterns[sensor][key])
                        for key in sorted(mask_patterns[sensor])
                    }
                    for sensor in sorted(mask_patterns)
                },
                "known_emit_regression": known_regression,
                "examples": examples,
                "limitations": [
                    "finite_values is an aggregate, not per-channel validity",
                    "no frozen usable-threshold decision is materialized",
                    "no alias/provenance quarantine state is materialized",
                ],
            }
        )
    return audits


def _read_npz_scalar_member(
    npz_path: Path, key: str, *, max_member_bytes: int
) -> tuple[str, int]:
    import numpy as np

    with zipfile.ZipFile(npz_path, "r") as archive:
        candidates = [
            item
            for item in archive.infolist()
            if Path(item.filename).name == f"{key}.npy"
        ]
        if len(candidates) != 1:
            raise KeyError(f"{key}.npy member_count={len(candidates)}")
        member = candidates[0]
        if member.file_size > max_member_bytes:
            raise ValueError(
                f"{member.filename} uncompressed_size={member.file_size} "
                f"exceeds limit={max_member_bytes}"
            )
        with archive.open(member, "r") as handle:
            payload = handle.read(max_member_bytes + 1)
        if len(payload) > max_member_bytes:
            raise ValueError(f"{member.filename} read exceeded limit")
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    if array.size != 1:
        raise ValueError(f"{key} must be scalar, shape={array.shape}")
    return _clean(array.reshape(-1)[0].item()), len(payload)


EMIT_TIME_PATTERN = re.compile(r"_(\d{8}T\d{6})_")


def _emit_granule_time(granule_id: str) -> datetime | None:
    match = EMIT_TIME_PATTERN.search(granule_id)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc
    )


def _parsed_time(value: str) -> datetime | None:
    normalized = _normalize_time(value)
    if not normalized:
        return None
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_slot_token(source_npz: str, plume_id: str) -> str:
    parts = Path(source_npz).parts
    try:
        index = parts.index(plume_id)
    except ValueError:
        return ""
    return parts[index - 1] if index > 0 else ""


def _probe_emit_slot(
    row: Mapping[str, str],
    plume_id: str,
    timepoint: str,
    *,
    time_tolerance_seconds: float,
    max_scalar_member_bytes: int,
) -> tuple[dict[str, Any], int]:
    # Importing/opening rasterio is intentionally isolated to explicit probes.
    import rasterio

    prefix = f"emit_{timepoint}_"
    path_text = _clean(row.get(prefix + "path", ""))
    manifest_exists = _bool_state(row.get(prefix + "exists", ""))
    manifest_product = _clean(row.get(prefix + "product_id", ""))
    manifest_time = _clean(row.get(prefix + "image_time", ""))
    result: dict[str, Any] = {
        "plume_id": plume_id,
        "timepoint": timepoint,
        "manifest_path": path_text,
        "manifest_exists": manifest_exists,
        "manifest_product_id": manifest_product,
        "manifest_acquisition_time": manifest_time,
        "tiff_header_status": "",
        "tiff_width": "",
        "tiff_height": "",
        "tiff_count": "",
        "source_npz": "",
        "source_npz_slot_token": "",
        "source_npz_slot_matches_manifest_slot": "",
        "npz_granule_id": "",
        "npz_source_nc": "",
        "manifest_product_matches_npz_granule": "",
        "npz_source_nc_matches_granule": "",
        "granule_manifest_time_delta_seconds": "",
        "provenance_status": "",
        "reason": "",
    }
    scalar_bytes = 0
    if not _has_value(path_text):
        result["provenance_status"] = "source_missing"
        result["reason"] = "manifest_path_empty"
        return result, scalar_bytes
    path = Path(path_text)
    if not path.is_file():
        result["provenance_status"] = "source_missing"
        result["reason"] = "exact_manifest_path_not_file"
        return result, scalar_bytes

    try:
        with rasterio.open(path, "r") as dataset:
            tags = dataset.tags()
            result["tiff_width"] = int(dataset.width)
            result["tiff_height"] = int(dataset.height)
            result["tiff_count"] = int(dataset.count)
        result["tiff_header_status"] = "ok"
    except Exception as exc:
        result["tiff_header_status"] = "read_error"
        result["provenance_status"] = "read_error"
        result["reason"] = f"{type(exc).__name__}:{str(exc)[:300]}"
        return result, scalar_bytes

    source_npz = _clean(tags.get("source_npz", ""))
    result["source_npz"] = source_npz
    slot_token = _source_slot_token(source_npz, plume_id)
    result["source_npz_slot_token"] = slot_token
    result["source_npz_slot_matches_manifest_slot"] = (
        slot_token == timepoint if slot_token else ""
    )
    if not source_npz:
        result["provenance_status"] = "provenance_unavailable"
        result["reason"] = "tiff_source_npz_tag_missing"
        return result, scalar_bytes
    npz_path = Path(source_npz)
    if not npz_path.is_file():
        result["provenance_status"] = "read_error"
        result["reason"] = "tagged_source_npz_not_file"
        return result, scalar_bytes

    try:
        granule, read_bytes = _read_npz_scalar_member(
            npz_path,
            "granule_id",
            max_member_bytes=max_scalar_member_bytes,
        )
        scalar_bytes += read_bytes
        source_nc, read_bytes = _read_npz_scalar_member(
            npz_path,
            "source_nc",
            max_member_bytes=max_scalar_member_bytes,
        )
        scalar_bytes += read_bytes
    except Exception as exc:
        result["provenance_status"] = "read_error"
        result["reason"] = f"{type(exc).__name__}:{str(exc)[:300]}"
        return result, scalar_bytes

    result["npz_granule_id"] = granule
    result["npz_source_nc"] = source_nc
    product_match = bool(manifest_product and granule == manifest_product)
    source_nc_match = bool(granule and granule in Path(source_nc).name)
    result["manifest_product_matches_npz_granule"] = product_match
    result["npz_source_nc_matches_granule"] = source_nc_match

    granule_time = _emit_granule_time(granule)
    recorded_time = _parsed_time(manifest_time)
    time_delta: float | None = None
    if granule_time is not None and recorded_time is not None:
        time_delta = abs((granule_time - recorded_time).total_seconds())
        result["granule_manifest_time_delta_seconds"] = time_delta

    reasons: list[str] = []
    if not manifest_product:
        reasons.append("manifest_product_id_missing")
    elif not product_match:
        reasons.append("manifest_product_id_vs_npz_granule_mismatch")
    if not source_nc:
        reasons.append("npz_source_nc_missing")
    elif not source_nc_match:
        reasons.append("npz_source_nc_vs_granule_mismatch")
    if time_delta is not None and time_delta > time_tolerance_seconds:
        reasons.append(
            "manifest_acquisition_time_vs_granule_time_exceeds_tolerance"
        )

    if reasons:
        result["provenance_status"] = "provenance_conflict"
        result["reason"] = ";".join(reasons)
    else:
        result["provenance_status"] = "verified"
        if slot_token and slot_token != timepoint:
            result["reason"] = (
                "verified_identity_but_source_npz_directory_slot_differs"
            )
        else:
            result["reason"] = "verified"
    return result, scalar_bytes


def _audit_emit_probes(
    probe_rows: Mapping[str, Mapping[str, str]],
    requested_ids: Sequence[str],
    *,
    time_tolerance_seconds: float,
    max_scalar_member_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    scalar_bytes = 0
    missing_rows: list[str] = []
    for plume_id in sorted(set(requested_ids)):
        row = probe_rows.get(plume_id)
        if row is None:
            missing_rows.append(plume_id)
            continue
        for timepoint in TIMEPOINTS:
            record, read_bytes = _probe_emit_slot(
                row,
                plume_id,
                timepoint,
                time_tolerance_seconds=time_tolerance_seconds,
                max_scalar_member_bytes=max_scalar_member_bytes,
            )
            scalar_bytes += read_bytes
            records.append(record)

    aliases: list[dict[str, Any]] = []
    path_duplicates: list[dict[str, Any]] = []
    for plume_id in sorted(set(record["plume_id"] for record in records)):
        plume_records = [
            record for record in records if record["plume_id"] == plume_id
        ]
        by_granule: defaultdict[str, list[str]] = defaultdict(list)
        by_source_npz: defaultdict[str, list[str]] = defaultdict(list)
        for record in plume_records:
            granule = _clean(record["npz_granule_id"])
            if granule:
                by_granule[granule].append(str(record["timepoint"]))
            source_npz = _clean(record["source_npz"])
            if source_npz:
                by_source_npz[source_npz].append(str(record["timepoint"]))
        aliases.extend(
            {
                "plume_id": plume_id,
                "observation_key": granule,
                "slots": slots,
                "stable_first_slot_for_audit_only": slots[0],
                "extra_slots": slots[1:],
                "all_slots_verified": all(
                    record["provenance_status"] == "verified"
                    for record in plume_records
                    if record["timepoint"] in slots
                ),
            }
            for granule, slots in by_granule.items()
            if len(slots) > 1
        )
        path_duplicates.extend(
            {
                "plume_id": plume_id,
                "source_npz": source_npz,
                "slots": slots,
            }
            for source_npz, slots in by_source_npz.items()
            if len(slots) > 1
        )

    status_counts = Counter(
        str(record["provenance_status"]) for record in records
    )
    result = {
        "requested_plume_ids": sorted(set(requested_ids)),
        "requested_ids_missing_from_wide_manifest": missing_rows,
        "slot_records": len(records),
        "status_counts": {
            key: int(status_counts[key]) for key in sorted(status_counts)
        },
        "provenance_conflict_count": int(
            status_counts["provenance_conflict"]
        ),
        "verified_observation_alias_groups": aliases,
        "duplicate_source_npz_groups": path_duplicates,
        "npz_scalar_uncompressed_bytes_read": scalar_bytes,
        "raster_pixel_read_calls": 0,
        "large_npz_array_read_calls": 0,
        "netcdf_array_read_calls": 0,
        "selection_is_explicit_not_random": True,
        "alias_assignment_warning": (
            "stable_first_slot_for_audit_only is not a production alias_of "
            "decision; intended target-time metadata must be frozen first"
        ),
    }
    return result, records


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
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


def _atomic_csv(
    records: Sequence[Mapping[str, Any]], path: Path
) -> None:
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
            writer = csv.DictWriter(handle, fieldnames=PROBE_COLUMNS)
            writer.writeheader()
            for record in records:
                writer.writerow({key: record.get(key, "") for key in PROBE_COLUMNS})
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _self_test() -> None:
    assert _value_state("") == "empty"
    assert _value_state(" NaN ") == "literal_missing_token"
    assert _value_state("/a/b") == "value"
    assert _bool_state("True") == "true"
    assert _bool_state("0.0") == "false"
    assert _bool_state("maybe") == "invalid"
    assert (
        _normalize_time("20230129T130333Z")
        == "2023-01-29T13:03:33+00:00"
    )
    row = {
        "emit_t0_product_id": "G",
        "emit_t0_image_time": "2023-01-01T00:00:00Z",
        "emit_t0_path": "/x",
    }
    assert _manifest_identity(row, "emit", "t0") == ("product_id", "G")
    parsed = _emit_granule_time(
        "EMIT_L2A_RFL_001_20230405T190323_2309513_003"
    )
    assert parsed is not None
    assert parsed.isoformat() == "2023-04-05T19:03:23+00:00"
    assert _source_slot_token("/a/prev2/plume/emit.npz", "plume") == "prev2"
    print(json.dumps({"self_test": "pass"}, sort_keys=True))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_rows < 0:
        raise ValueError("--max-rows must be >= 0")
    if args.sample_limit < 0:
        raise ValueError("--sample-limit must be >= 0")
    if args.max_scalar_member_bytes <= 0:
        raise ValueError("--max-scalar-member-bytes must be > 0")
    requested_ids = sorted(set(args.emit_probe_plume_id))
    wide, probe_rows, wide_header = _audit_wide_manifest(
        args.wide_manifest,
        max_rows=args.max_rows,
        sample_limit=args.sample_limit,
        requested_probe_ids=set(requested_ids),
        time_tolerance_seconds=args.time_tolerance_seconds,
    )
    task_audits, task_headers = _audit_task_manifests(args.task_manifest)
    validation_audits = _audit_validation_sidecars(args.validation_sidecar)
    emit_probe, probe_records = _audit_emit_probes(
        probe_rows,
        requested_ids,
        time_tolerance_seconds=args.time_tolerance_seconds,
        max_scalar_member_bytes=args.max_scalar_member_bytes,
    )
    schema_readiness = _schema_readiness(wide_header, task_headers)

    blockers = [
        "v2 execution/query-coverage columns are not materialized",
        (
            "full-table embedded provenance is unaudited; this pilot probes "
            "only explicitly named EMIT plume IDs"
        ),
        (
            "query any-valid smoke metadata lacks per-channel validity and "
            "frozen usable-threshold decisions"
        ),
        (
            "canonical alias_of selection needs explicit intended target "
            "times for every sensor/time slot"
        ),
    ]
    if emit_probe["provenance_conflict_count"]:
        blockers.append(
            "at least one embedded/manifest provenance conflict must be "
            "repaired or quarantined"
        )
    emit_time_mismatches = wide["sensors"]["emit"][
        "product_time_mismatch_slots"
    ]
    if emit_time_mismatches:
        blockers.append(
            f"{emit_time_mismatches} EMIT slots have manifest acquisition "
            "times inconsistent with their product IDs"
        )
    if any(
        set(audit["label_counts"]) != {"0", "1"}
        or not {"train", "val", "test"}.issubset(audit["split_counts"])
        for audit in task_audits
    ):
        blockers.append(
            "current task pilot is not stratified across both labels and all "
            "train/val/test splits"
        )
    if not wide["event_split_gate_pass"]:
        blockers.append("event_group split overlap is nonzero")

    result = {
        "schema_version": SCHEMA_VERSION,
        "mode": "read_only_metadata_audit",
        "no_materialization_or_training": True,
        "io_contract": {
            "wide_manifest": "sequential CSV metadata read",
            "task_and_validation_inputs": "CSV metadata read",
            "emit_tiff": "header/default tags only; dataset.read never called",
            "emit_npz": (
                "granule_id.npy and source_nc.npy only, each size-capped"
            ),
            "raster_pixels_read": False,
            "netcdf_arrays_read": False,
        },
        "script": _file_identity(Path(__file__).resolve()),
        "legacy_emit_pipeline": _file_identity(args.old_emit_pipeline),
        "wide_manifest": wide,
        "task_manifests": task_audits,
        "validation_sidecars": validation_audits,
        "v2_schema_readiness": schema_readiness,
        "embedded_emit_probe": emit_probe,
        "production_crop_authorized": False,
        "remaining_blockers": blockers,
        "output_probe_csv": str(args.output_probe_csv),
    }
    _atomic_csv(probe_records, args.output_probe_csv)
    result["output_probe_csv_sha256"] = _sha256(args.output_probe_csv)
    _atomic_json(result, args.output_json)
    return result


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return
    result = run(args)
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "rows_read": result["wide_manifest"]["rows_read"],
                "event_split_gate_pass": result["wide_manifest"][
                    "event_split_gate_pass"
                ],
                "v2_ready": result["v2_schema_readiness"][
                    "ready_for_v2_execution"
                ],
                "emit_probe_status_counts": result["embedded_emit_probe"][
                    "status_counts"
                ],
                "production_crop_authorized": result[
                    "production_crop_authorized"
                ],
                "output_json": str(args.output_json),
                "output_probe_csv": str(args.output_probe_csv),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
