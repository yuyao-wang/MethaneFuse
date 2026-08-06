#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
FILENAMES = {
    "t0": "s2.tif",
    "prev1": "s2_-7.tif",
    "prev2": "s2_prev2.tif",
    "prev3": "s2_prev3.tif",
    "seasonal": "s2_-90.tif",
    "year": "s2_-360.tif",
}
STD_COLUMNS = {
    "t0": "s2_0_std_512",
    "prev1": "s2_-7_std_512",
    "prev2": "s2_prev2_std_512",
    "prev3": "s2_prev3_std_512",
    "seasonal": "s2_-90_std_512",
    "year": "s2_-360_std_512",
}
MODEL_BANDS = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)
BASE_BANDS = (0, 1, 2, 3, 4, 5, 6, 10, 11)
CURRENT_EXACT_MARKER = "/S2_point_center_exact_v3/"
LOCAL_SAFE_MARKER = "/S2_canonical_6510_local_safe_recrop_v4/"
LEGACY_GEE_MARKER = "/plume_raw_s2_90360_fixed_512/"

DEFAULT_CANONICAL_TABLE = Path(
    "/home/yuyao/panopticon/Upgraded_dataset/"
    "s2_canonical_temporal_split_v4/"
    "s2_cdse_legacy512_all_cutoff_2025-11-24.csv"
)
DEFAULT_EXACT_TABLE = Path(
    "/home/yuyao/methane_train/Upgrade_data_pipeline/csv/"
    "s2_6time_point_center_exact_v3_paths.csv"
)
DEFAULT_RECROP_METADATA_TABLE = Path(
    "/home/yuyao/panopticon/Upgraded_dataset/"
    "s2_canonical_6510_recrop_retry_progress_v4.csv"
)
DEFAULT_CORRECT_OLD_T0_ROOT = Path(
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/"
    "plume_s2_CDSE0_gee90360_original_512"
)
DEFAULT_TARGET_ROOT = Path(
    "/mnt/engg-niulab/yuyao/preprocessed_512/"
    "S2_6time_historical_exact_v8"
)
DEFAULT_WORK_ROOT = Path(
    "/home/yuyao/panopticon/Upgraded_dataset/"
    "s2_historical_exact_v8"
)
DEFAULT_RECROP_PROGRESS_TABLE = (
    DEFAULT_WORK_ROOT / "s2_v8_recrop_progress.csv"
)
DEFAULT_CACHE_ROOT = Path(
    "/diniuvol/yuyao/s2_historical_exact_v8_restore_cache"
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def file_ok(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def read_json(path: Path) -> dict[str, Any]:
    if not file_ok(path):
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json_atomic(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sidecar_path(path: Path) -> Path:
    return Path(str(path) + ".georef.json")


def read_bhw(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim != 3:
        raise ValueError(f"expected a 3-D TIFF, got {image.shape}: {path}")
    if image.shape[0] == 12:
        output = image
    elif image.shape[-1] == 12:
        output = np.moveaxis(image, -1, 0)
    else:
        raise ValueError(f"expected 12 bands, got {image.shape}: {path}")
    if output.shape[1:] != (512, 512):
        raise ValueError(f"expected 512x512, got {output.shape}: {path}")
    if output.dtype != np.uint16:
        if np.issubdtype(output.dtype, np.floating):
            output = np.clip(output, 0, 65535).astype(np.uint16)
        else:
            output = output.astype(np.uint16)
    return output


def target_path(target_root: Path, timepoint: str, plume_id: str) -> Path:
    return target_root / timepoint / plume_id / FILENAMES[timepoint]


def exact_legacy_source(
    exact_lookup: dict[str, pd.Series],
    plume_id: str,
    timepoint: str,
) -> str:
    row = exact_lookup.get(plume_id)
    if row is None:
        return ""
    input_kind = clean(row.get(f"{timepoint}_input_kind", ""))
    path = clean(row.get(f"{timepoint}_input_path", ""))
    if input_kind == "512" and LEGACY_GEE_MARKER in path and file_ok(Path(path)):
        return path
    return ""


def source_metadata(path: Path) -> dict[str, Any]:
    return read_json(sidecar_path(path))


def product_fields(metadata: dict[str, Any]) -> tuple[str, str]:
    product_id = clean(
        metadata.get(
            "product_id",
            metadata.get("canonical_s2_band_source_product_id", ""),
        )
    )
    product_name = clean(
        metadata.get(
            "product_name",
            metadata.get("canonical_s2_band_source_product_name", ""),
        )
    )
    return product_id, product_name


def prepare(args: argparse.Namespace) -> int:
    canonical = pd.read_csv(args.canonical_table, low_memory=False)
    exact = pd.read_csv(args.exact_table, low_memory=False)
    recrop_metadata = pd.read_csv(args.recrop_metadata_table, low_memory=False)
    exact_lookup = {
        clean(row["plume_id"]): row
        for _, row in exact.iterrows()
        if clean(row.get("plume_id", ""))
    }
    recrop_lookup = {
        clean(row["plume_id"]): row
        for _, row in recrop_metadata.iterrows()
        if clean(row.get("plume_id", ""))
    }
    correct_old_root = Path(args.correct_old_t0_root)
    target_root = Path(args.target_root)
    metadata_paths = sorted(
        {
            clean(row.get(f"{timepoint}_input_path", ""))
            for _, row in canonical.iterrows()
            for timepoint in TIMEPOINTS
            if CURRENT_EXACT_MARKER
            in clean(row.get(f"{timepoint}_input_path", ""))
            or LOCAL_SAFE_MARKER
            in clean(row.get(f"{timepoint}_input_path", ""))
        }
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        metadata_values = executor.map(
            lambda value: source_metadata(Path(value)),
            metadata_paths,
        )
        metadata_lookup = dict(zip(metadata_paths, metadata_values))
    rows: list[pd.Series] = []
    dropped: list[dict[str, str]] = []
    policy_counts: Counter[str] = Counter()
    product_counts: Counter[str] = Counter()

    for _, source_row in canonical.iterrows():
        row = source_row.copy()
        plume_id = clean(row.get("plume_id", ""))
        old_t0 = correct_old_root / plume_id / "s2_std.tif"
        legacy_t0 = clean(row.get("t0_input_kind", "")) == "legacy_gee_correct_512"
        if legacy_t0 and not file_ok(old_t0):
            dropped.append(
                {
                    "plume_id": plume_id,
                    "event_group_id": clean(row.get("event_group_id", "")),
                    "reason": "historical_cdse0_t0_missing",
                    "wrong_t0_path": clean(row.get("t0_input_path", "")),
                }
            )
            continue

        for timepoint in TIMEPOINTS:
            source_path = clean(row.get(f"{timepoint}_input_path", ""))
            source_kind = clean(row.get(f"{timepoint}_input_kind", ""))
            policy = "recrop_or_restore"
            direct_path = ""

            if timepoint == "t0" and legacy_t0:
                policy = "historical_cdse0_t0"
                direct_path = str(old_t0)
            elif (
                timepoint in {"seasonal", "year"}
                and source_kind == "legacy_gee_correct_512"
                and file_ok(Path(source_path))
            ):
                policy = "historical_gee_direct"
                direct_path = source_path
            elif CURRENT_EXACT_MARKER in source_path:
                legacy_source = exact_legacy_source(
                    exact_lookup,
                    plume_id,
                    timepoint,
                )
                if legacy_source:
                    policy = "historical_gee_direct"
                    direct_path = legacy_source

            row[f"{timepoint}_v8_source_path"] = source_path
            row[f"{timepoint}_v8_policy"] = policy
            row[f"{timepoint}_v8_direct_path"] = direct_path
            row[f"{timepoint}_v8_target_path"] = str(
                target_path(target_root, timepoint, plume_id)
            )
            policy_counts[f"{timepoint}:{policy}"] += 1

            product_id = ""
            product_name = ""
            if policy == "recrop_or_restore":
                metadata = metadata_lookup.get(source_path, {})
                product_id, product_name = product_fields(metadata)
                manifest_row = exact_lookup.get(plume_id)
                if manifest_row is None:
                    manifest_row = recrop_lookup.get(plume_id)
                if manifest_row is not None:
                    product_id = clean(
                        manifest_row.get(f"{timepoint}_product_id", "")
                    ) or product_id
                    product_name = clean(
                        manifest_row.get(f"{timepoint}_product_name", "")
                    ) or product_name
                    for suffix in (
                        "product_local_path",
                        "stac_item_id",
                        "stac_base_url",
                        "stac_cog_base_url",
                        "stac_dn_add",
                        "stac_source_product_name",
                        "stac_mosaic_json",
                        "local_mosaic_json",
                        "cdse_mosaic_json",
                    ):
                        column = f"{timepoint}_{suffix}"
                        row[column] = clean(manifest_row.get(column, ""))
                product_counts[
                    f"{timepoint}:{'has_product' if product_id and product_name else 'no_product'}"
                ] += 1
            row[f"{timepoint}_product_id"] = product_id
            row[f"{timepoint}_product_name"] = product_name
            row[f"{timepoint}_raw_path"] = source_path
            row[f"{timepoint}_download_needed"] = 0
            row[f"{timepoint}_download_target_raw_path"] = ""
        rows.append(row)

    prepared = pd.DataFrame(rows).reset_index(drop=True)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    output_table = Path(args.prepared_table)
    write_csv_atomic(prepared, output_table)
    write_csv_atomic(pd.DataFrame(dropped), Path(args.dropped_table))
    report = {
        "created_at": utc_now(),
        "canonical_rows": int(len(canonical)),
        "prepared_rows": int(len(prepared)),
        "dropped_rows": int(len(dropped)),
        "drop_reasons": dict(Counter(item["reason"] for item in dropped)),
        "policy_counts": dict(sorted(policy_counts.items())),
        "product_counts": dict(sorted(product_counts.items())),
        "prepared_table": str(output_table),
        "dropped_table": str(args.dropped_table),
        "target_root": str(target_root),
    }
    write_json_atomic(report, Path(args.prepare_report))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def classify_fill_source(metadata: dict[str, Any]) -> tuple[int, int, str]:
    raw_current_add = metadata.get(
        "canonical_s2_band_dn_add",
        metadata.get("b08_b09_dn_add"),
    )
    try:
        current_add = int(raw_current_add)
    except (TypeError, ValueError):
        raise ValueError("missing canonical B8A DN metadata")

    source = clean(
        metadata.get(
            "canonical_s2_band_source",
            metadata.get("b08_b09_source", ""),
        )
    )
    if source == "cdse_nodes_exact":
        desired_add = 0
        source_class = "cdse_nodes_exact"
    elif metadata.get("canonical_s2_band_cog_base_url") or metadata.get(
        "b08_b09_cog_base_url"
    ):
        desired_add = 1000
        source_class = "stac_cog"
    else:
        raise ValueError("unknown canonical B8A source")
    return current_add, desired_add, source_class


def validate_legacy_cdse(image: np.ndarray, path: Path) -> None:
    if image.shape != (12, 512, 512):
        raise ValueError(f"invalid output shape {image.shape}: {path}")
    if np.any(image[8]) or np.any(image[9]):
        raise ValueError(f"bands 8 and 9 are not zero: {path}")
    empty = [band for band in MODEL_BANDS if not np.any(image[band])]
    if empty:
        raise ValueError(f"empty model bands {empty}: {path}")


def restore_one(
    task: dict[str, str],
    target_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    plume_id = task["plume_id"]
    timepoint = task["timepoint"]
    source = Path(task["source_path"])
    target = target_path(target_root, timepoint, plume_id)
    marker = target.with_name(target.name + ".legacy_v8.json")
    if file_ok(target) and file_ok(marker):
        marker_data = read_json(marker)
        if marker_data.get("state") == "complete":
            return {
                **task,
                "status": "target_exists",
                "target_path": str(target),
                "source_class": clean(marker_data.get("source_class", "")),
                "bytes": int(target.stat().st_size),
            }

    if not file_ok(source):
        raise FileNotFoundError(source)
    relative = Path(timepoint) / plume_id / FILENAMES[timepoint]
    cache_dir = cache_root / relative.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_cache = cache_dir / (relative.name + ".source")
    output_cache = cache_dir / (relative.name + ".output")
    shutil.copyfile(source, source_cache)
    image = read_bhw(source_cache)
    metadata = source_metadata(source)
    current_add, desired_add, source_class = classify_fill_source(metadata)
    delta = desired_add - current_add

    output = np.zeros((12, 512, 512), dtype=np.uint16)
    output[list(BASE_BANDS)] = image[list(BASE_BANDS)]
    b8a = image[8].astype(np.int32)
    b8a[b8a > 0] += delta
    output[7] = np.clip(b8a, 0, 65535).astype(np.uint16)
    validate_legacy_cdse(output, source)
    tifffile.imwrite(output_cache, output)
    validate_legacy_cdse(read_bhw(output_cache), output_cache)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    shutil.copyfile(output_cache, temporary)
    os.replace(temporary, target)
    marker_data = {
        "state": "complete",
        "created_at": utc_now(),
        "source_path": str(source),
        "source_class": source_class,
        "current_fill_dn_add": current_add,
        "desired_native_dn_add": desired_add,
        "applied_b8a_delta": delta,
        "layout": "B01,B02,B03,B04,B05,B06,B07,B8A,0,0,B11,B12",
    }
    write_json_atomic(marker_data, marker)
    source_cache.unlink(missing_ok=True)
    output_cache.unlink(missing_ok=True)
    try:
        cache_dir.rmdir()
    except OSError:
        pass
    return {
        **task,
        "status": "restored",
        "target_path": str(target),
        "source_class": source_class,
        "bytes": int(target.stat().st_size),
    }


def valid_recrop(path: Path) -> bool:
    if not file_ok(path):
        return False
    try:
        with tifffile.TiffFile(path) as tif:
            shape = tuple(tif.series[0].shape)
            dtype = np.dtype(tif.series[0].dtype)
        if shape not in {(12, 512, 512), (512, 512, 12)}:
            return False
        if dtype != np.dtype(np.uint16):
            return False
    except (OSError, ValueError, tifffile.TiffFileError):
        return False
    return True


def finalize(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.prepared_table, low_memory=False)
    progress = pd.read_csv(args.recrop_progress_table, low_memory=False)
    progress_lookup = {
        clean(row["plume_id"]): row
        for _, row in progress.iterrows()
        if clean(row.get("plume_id", ""))
    }
    target_root = Path(args.target_root)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, str]] = []
    final_paths: dict[tuple[int, str], str] = {}
    direct_counts: Counter[str] = Counter()
    recrop_counts: Counter[str] = Counter()
    direct_candidates: dict[tuple[int, str], Path] = {}

    for row_index, row in frame.iterrows():
        plume_id = clean(row["plume_id"])
        for timepoint in TIMEPOINTS:
            policy = clean(row[f"{timepoint}_v8_policy"])
            direct = Path(clean(row[f"{timepoint}_v8_direct_path"]))
            target = target_path(target_root, timepoint, plume_id)
            if policy != "recrop_or_restore":
                direct_candidates[(row_index, timepoint)] = direct
                continue
            progress_row = progress_lookup.get(plume_id)
            status = (
                clean(progress_row.get(f"{timepoint}_recrop_status", ""))
                if progress_row is not None
                else ""
            )
            if status in {"downloaded", "target_exists"}:
                final_paths[(row_index, timepoint)] = str(target)
                recrop_counts[f"{timepoint}:safe_recrop"] += 1
                continue
            tasks.append(
                {
                    "row_index": str(row_index),
                    "plume_id": plume_id,
                    "timepoint": timepoint,
                    "source_path": clean(row[f"{timepoint}_v8_source_path"]),
                }
            )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        direct_validity = dict(
            zip(
                direct_candidates,
                executor.map(file_ok, direct_candidates.values()),
            )
        )
    invalid_direct = [
        str(direct_candidates[key])
        for key, valid in direct_validity.items()
        if not valid
    ]
    if invalid_direct:
        raise FileNotFoundError(
            f"missing direct historical sources: {invalid_direct[:20]}"
        )
    for key, direct in direct_candidates.items():
        row_index, timepoint = key
        policy = clean(frame.at[row_index, f"{timepoint}_v8_policy"])
        final_paths[key] = str(direct)
        direct_counts[f"{timepoint}:{policy}"] += 1

    print(
        f"rows={len(frame)} direct={sum(direct_counts.values())} "
        f"safe_recrops={sum(recrop_counts.values())} "
        f"fallback_restore_tasks={len(tasks)} workers={args.workers}",
        flush=True,
    )
    started = time.monotonic()
    completed = 0
    restored_bytes = 0
    failures: list[dict[str, str]] = []
    dropped_quality_groups: list[str] = []
    dropped_quality_plumes: list[str] = []
    source_counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(restore_one, task, target_root, cache_root): task
            for task in tasks
        }
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                result = future.result()
                row_index = int(result["row_index"])
                timepoint = result["timepoint"]
                final_paths[(row_index, timepoint)] = result["target_path"]
                source_counts[result["source_class"]] += 1
                restored_bytes += int(result["bytes"])
            except Exception as exc:
                failures.append(
                    {
                        **task,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            completed += 1
            if completed % max(1, args.progress_every) == 0:
                elapsed = time.monotonic() - started
                rate = completed / max(elapsed, 1e-6)
                eta = (len(tasks) - completed) / max(rate, 1e-6)
                print(
                    f"restored={completed}/{len(tasks)} failures={len(failures)} "
                    f"rate={rate:.2f}/s eta={eta / 60:.1f}m",
                    flush=True,
                )

    if failures:
        non_quality_failures = [
            failure
            for failure in failures
            if "empty model bands" not in failure["error"]
        ]
        if non_quality_failures:
            write_json_atomic(
                {
                    "created_at": utc_now(),
                    "task_count": len(tasks),
                    "failure_count": len(failures),
                    "non_quality_failure_count": len(non_quality_failures),
                    "failures": failures,
                },
                Path(args.failure_report),
            )
            raise RuntimeError(
                f"fallback restoration has {len(non_quality_failures)} "
                f"non-quality failures; see {args.failure_report}"
            )
        failed_row_indices = {
            int(failure["row_index"]) for failure in failures
        }
        dropped_quality_groups = sorted(
            {
                clean(frame.at[row_index, "event_group_id"])
                for row_index in failed_row_indices
            }
        )
        drop_mask = frame["event_group_id"].astype(str).isin(
            dropped_quality_groups
        )
        dropped_quality_plumes = sorted(
            frame.loc[drop_mask, "plume_id"].astype(str)
        )
        if int(drop_mask.sum()) > int(args.max_quality_drop_rows):
            raise RuntimeError(
                f"quality failures require dropping {int(drop_mask.sum())} "
                f"rows, above --max-quality-drop-rows="
                f"{args.max_quality_drop_rows}"
            )
        write_json_atomic(
            {
                "created_at": utc_now(),
                "task_count": len(tasks),
                "failure_count": len(failures),
                "dropped_event_groups": dropped_quality_groups,
                "dropped_plume_ids": dropped_quality_plumes,
                "failures": failures,
            },
            Path(args.failure_report),
        )
        frame = frame.loc[~drop_mask].copy()
        print(
            f"dropped_quality_rows={int(drop_mask.sum())} "
            f"event_groups={len(dropped_quality_groups)} "
            f"failed_files={len(failures)}",
            flush=True,
        )

    for row_index, row in frame.iterrows():
        for timepoint in TIMEPOINTS:
            final_path = final_paths[(row_index, timepoint)]
            frame.at[row_index, f"{timepoint}_input_path"] = final_path
            frame.at[row_index, f"{timepoint}_raw_path"] = final_path
            frame.at[row_index, STD_COLUMNS[timepoint]] = final_path
            frame.at[row_index, f"{timepoint}_input_kind"] = clean(
                row[f"{timepoint}_v8_policy"]
            )
    if "resized_512x512_path" not in frame.columns:
        frame["resized_512x512_path"] = frame["s2_mask_512_path"]
    else:
        frame["resized_512x512_path"] = frame[
            "resized_512x512_path"
        ].fillna(frame["s2_mask_512_path"])
    frame["has_all6_512"] = True
    write_csv_atomic(frame, Path(args.final_table))
    report = {
        "finished_at": utc_now(),
        "rows": int(len(frame)),
        "files": int(len(frame) * len(TIMEPOINTS)),
        "direct_counts": dict(sorted(direct_counts.items())),
        "safe_recrop_counts": dict(sorted(recrop_counts.items())),
        "fallback_tasks": int(len(tasks)),
        "fallback_source_counts": dict(sorted(source_counts.items())),
        "dropped_quality_event_groups": dropped_quality_groups,
        "dropped_quality_plumes": dropped_quality_plumes,
        "restored_bytes": int(restored_bytes),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "final_table": str(args.final_table),
        "target_root": str(target_root),
    }
    write_json_atomic(report, Path(args.finalize_report))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def split_final(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.final_table, low_memory=False)
    if "split" not in frame.columns:
        raise ValueError("final table has no preserved temporal split column")
    if "resized_512x512_path" not in frame.columns:
        frame["resized_512x512_path"] = frame["s2_mask_512_path"]
    frame["split"] = frame["split"].map(clean)
    bad_splits = sorted(set(frame["split"]) - {"train", "test"})
    if bad_splits:
        raise ValueError(f"unexpected split values: {bad_splits}")

    train = frame[frame["split"] == "train"].copy()
    test = frame[frame["split"] == "test"].copy()
    train_groups = set(train["event_group_id"].astype(str))
    test_groups = set(test["event_group_id"].astype(str))
    leakage = sorted(train_groups & test_groups)
    ratio = len(train) / max(1, len(frame))
    train_time = pd.to_datetime(train["event_time"], utc=True, errors="coerce")
    test_time = pd.to_datetime(test["event_time"], utc=True, errors="coerce")
    if leakage:
        raise ValueError(f"event leakage across split: {leakage[:10]}")
    if not 0.80 <= ratio <= 0.90:
        raise ValueError(f"train ratio is outside [0.80, 0.90]: {ratio:.6f}")
    if train_time.isna().any() or test_time.isna().any():
        raise ValueError("invalid event timestamps in temporal split")
    if train_time.max() >= test_time.min():
        raise ValueError(
            "temporal split is not strictly ordered: "
            f"train_max={train_time.max()} test_min={test_time.min()}"
        )

    split_root = Path(args.split_root)
    split_root.mkdir(parents=True, exist_ok=True)
    train_path = split_root / "s2_v8_train.csv"
    test_path = split_root / "s2_v8_test.csv"
    write_csv_atomic(train, train_path)
    write_csv_atomic(test, test_path)
    report = {
        "created_at": utc_now(),
        "all_rows": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_ratio": ratio,
        "train_event_groups": int(len(train_groups)),
        "test_event_groups": int(len(test_groups)),
        "event_group_overlap": int(len(leakage)),
        "train_time_min": train_time.min().isoformat(),
        "train_time_max": train_time.max().isoformat(),
        "test_time_min": test_time.min().isoformat(),
        "test_time_max": test_time.max().isoformat(),
        "train_csv": str(train_path),
        "test_csv": str(test_path),
    }
    write_json_atomic(report, split_root / "split_audit.json")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the S2 six-time 512 dataset in the exact historical "
            "CDSE/GEE layout, with a bounded local cache."
        )
    )
    parser.add_argument("action", choices=("prepare", "finalize", "split"))
    parser.add_argument("--canonical-table", default=str(DEFAULT_CANONICAL_TABLE))
    parser.add_argument("--exact-table", default=str(DEFAULT_EXACT_TABLE))
    parser.add_argument(
        "--recrop-metadata-table",
        default=str(DEFAULT_RECROP_METADATA_TABLE),
    )
    parser.add_argument(
        "--correct-old-t0-root",
        default=str(DEFAULT_CORRECT_OLD_T0_ROOT),
    )
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    parser.add_argument(
        "--prepared-table",
        default=str(DEFAULT_WORK_ROOT / "s2_v8_recrop_sources.csv"),
    )
    parser.add_argument(
        "--recrop-progress-table",
        default=str(DEFAULT_RECROP_PROGRESS_TABLE),
    )
    parser.add_argument(
        "--dropped-table",
        default=str(DEFAULT_WORK_ROOT / "s2_v8_dropped_wrong_t0.csv"),
    )
    parser.add_argument(
        "--prepare-report",
        default=str(DEFAULT_WORK_ROOT / "s2_v8_prepare_report.json"),
    )
    parser.add_argument(
        "--final-table",
        default=str(DEFAULT_WORK_ROOT / "s2_v8_all_512.csv"),
    )
    parser.add_argument(
        "--finalize-report",
        default=str(DEFAULT_WORK_ROOT / "s2_v8_finalize_report.json"),
    )
    parser.add_argument(
        "--failure-report",
        default=str(DEFAULT_WORK_ROOT / "s2_v8_restore_failures.json"),
    )
    parser.add_argument(
        "--split-root",
        default=str(DEFAULT_WORK_ROOT / "temporal_split"),
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--max-quality-drop-rows", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "prepare":
        return prepare(args)
    if args.action == "finalize":
        return finalize(args)
    return split_final(args)


if __name__ == "__main__":
    raise SystemExit(main())
