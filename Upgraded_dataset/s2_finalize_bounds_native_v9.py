#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

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
FINAL_COLUMNS = {
    "t0": "s2_0_std_512",
    "prev1": "s2_-7_std_512",
    "prev2": "s2_prev2_std_512",
    "prev3": "s2_prev3_std_512",
    "seasonal": "s2_-90_std_512",
    "year": "s2_-360_std_512",
}
MODEL_BANDS = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_json_atomic(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def target_path(root: Path, timepoint: str, plume_id: str) -> Path:
    return root / timepoint / plume_id / FILENAMES[timepoint]


def read_bhw(path: Path) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    if image.ndim != 3:
        raise ValueError(f"expected 3-D TIFF, got {image.shape}: {path}")
    if image.shape[0] == 12:
        output = image
    elif image.shape[-1] == 12:
        output = np.moveaxis(image, -1, 0)
    else:
        raise ValueError(f"expected 12 bands, got {image.shape}: {path}")
    if output.shape != (12, 512, 512):
        raise ValueError(f"expected 12x512x512, got {output.shape}: {path}")
    return output.astype(np.uint16, copy=False)


def normalize_direct_one(
    source: Path,
    target: Path,
    cache_root: Path,
    plume_id: str,
    timepoint: str,
) -> str:
    marker = target.with_name(target.name + ".legacy_v9.json")
    if marker.is_file():
        try:
            metadata = json.loads(marker.read_text())
            if metadata.get("state") == "complete":
                return "reused"
        except (OSError, json.JSONDecodeError):
            pass
    cache = cache_root / timepoint / plume_id / source.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.part"
    )
    try:
        shutil.copyfile(source, cache)
        image = read_bhw(cache)
        output = np.zeros((12, 512, 512), dtype=np.uint16)
        output[:7] = image[:7]
        if np.any(image[8]) or np.any(image[9]):
            output[7] = image[8]
            source_layout = "full_s2_b8a_at_8"
        else:
            output[7] = image[7]
            source_layout = "legacy_b8a_at_7"
        output[10] = image[10]
        output[11] = image[11]
        if any(not np.any(output[band]) for band in MODEL_BANDS):
            raise ValueError(f"empty model band after normalization: {source}")
        tifffile.imwrite(temporary_target, output)
        os.replace(temporary_target, target)
        write_json_atomic(
            {
                "state": "complete",
                "created_at": utc_now(),
                "source_path": str(source),
                "source_layout": source_layout,
                "target_layout": "B01,B02,B03,B04,B05,B06,B07,B8A,0,0,B11,B12",
            },
            marker,
        )
    finally:
        cache.unlink(missing_ok=True)
        temporary_target.unlink(missing_ok=True)
    return "normalized"


def header_ok(path: Path) -> tuple[bool, str]:
    try:
        with tifffile.TiffFile(path) as tif:
            shape = tuple(tif.series[0].shape)
            dtype = np.dtype(tif.series[0].dtype)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if shape not in {(12, 512, 512), (512, 512, 12)}:
        return False, f"shape={shape}"
    if dtype != np.dtype(np.uint16):
        return False, f"dtype={dtype}"
    return True, ""


def content_ok(path: Path) -> tuple[bool, str]:
    try:
        image = read_bhw(path)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if np.any(image[8]) or np.any(image[9]):
        return False, "bands 8/9 are not muted"
    empty = [band for band in MODEL_BANDS if not np.any(image[band])]
    if empty:
        return False, f"empty model bands={empty}"
    return True, ""


def main(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.table, low_memory=False)
    target_root = Path(args.target_root)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    normalize_tasks = []
    direct_policies = {
        "t0": {"historical_cdse0_t0"},
        "seasonal": {"historical_gee_direct"},
        "year": {"historical_gee_direct"},
    }
    for row in frame.itertuples(index=False):
        plume_id = str(row.plume_id)
        for timepoint, policies in direct_policies.items():
            policy = str(getattr(row, f"{timepoint}_v8_policy"))
            if policy not in policies:
                continue
            target = target_path(target_root, timepoint, plume_id)
            normalize_tasks.append(
                (target, target, cache_root, plume_id, timepoint)
            )

    normalized = 0
    reused = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(normalize_direct_one, *task)
            for task in normalize_tasks
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            normalized += int(result == "normalized")
            reused += int(result == "reused")
            if completed % 500 == 0 or completed == len(futures):
                print(
                    f"[Normalize] {completed}/{len(futures)} "
                    f"normalized={normalized} reused={reused}",
                    flush=True,
                )

    row_paths = {}
    all_paths = []
    for row_index, row in frame.iterrows():
        paths = {
            timepoint: target_path(
                target_root, timepoint, str(row["plume_id"])
            )
            for timepoint in TIMEPOINTS
        }
        row_paths[row_index] = paths
        all_paths.extend(paths.values())

    invalid_headers = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(header_ok, path): path for path in all_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            valid, message = future.result()
            if not valid:
                invalid_headers[str(path)] = message
            if completed % 5000 == 0 or completed == len(futures):
                print(
                    f"[Headers] {completed}/{len(futures)} "
                    f"invalid={len(invalid_headers)}",
                    flush=True,
                )

    bad_rows = {
        row_index
        for row_index, paths in row_paths.items()
        if any(str(path) in invalid_headers for path in paths.values())
    }
    bad_event_groups = sorted(
        frame.loc[list(bad_rows), "event_group_id"].astype(str).unique()
    )
    drop_mask = frame["event_group_id"].astype(str).isin(bad_event_groups)
    if int(drop_mask.sum()) > args.max_drop_rows:
        raise RuntimeError(
            f"missing/invalid files require dropping {int(drop_mask.sum())} rows, "
            f"above --max-drop-rows={args.max_drop_rows}"
        )
    complete = frame.loc[~drop_mask].copy()

    sample_count = min(args.content_samples, len(complete) * len(TIMEPOINTS))
    sample_paths = [
        row_paths[row_index][timepoint]
        for row_index in complete.index
        for timepoint in TIMEPOINTS
    ]
    rng = np.random.default_rng(args.seed)
    sampled_paths = [
        sample_paths[index]
        for index in rng.choice(
            len(sample_paths), size=sample_count, replace=False
        )
    ]
    invalid_content = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(content_ok, path): path for path in sampled_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            valid, message = future.result()
            if not valid:
                invalid_content[str(path)] = message
            if completed % 500 == 0 or completed == len(futures):
                print(
                    f"[Content] {completed}/{len(futures)} "
                    f"invalid={len(invalid_content)}",
                    flush=True,
                )
    if invalid_content:
        raise RuntimeError(
            f"content audit failed for {len(invalid_content)} sampled files: "
            f"{list(invalid_content.items())[:10]}"
        )

    for row_index, row in complete.iterrows():
        for timepoint in TIMEPOINTS:
            path = str(row_paths[row_index][timepoint])
            complete.at[row_index, FINAL_COLUMNS[timepoint]] = path
            complete.at[row_index, f"{timepoint}_input_path"] = path
            complete.at[row_index, f"{timepoint}_raw_path"] = path
            complete.at[row_index, f"{timepoint}_input_kind"] = (
                "historical_bounds_native_v9"
            )
    complete["resized_512x512_path"] = complete["s2_mask_512_path"]
    complete["has_all6_512"] = True
    complete = complete.reset_index(drop=True)

    train = complete[complete["split"].astype(str).eq("train")].copy()
    test = complete[complete["split"].astype(str).eq("test")].copy()
    train_events = set(train["event_group_id"].astype(str))
    test_events = set(test["event_group_id"].astype(str))
    event_overlap = sorted(train_events & test_events)
    train_times = pd.to_datetime(train["event_time"], utc=True, errors="coerce")
    test_times = pd.to_datetime(test["event_time"], utc=True, errors="coerce")
    train_ratio = len(train) / max(1, len(complete))
    if event_overlap:
        raise RuntimeError(f"event leakage: {event_overlap[:10]}")
    if not 0.80 <= train_ratio <= 0.90:
        raise RuntimeError(f"train ratio outside [0.80, 0.90]: {train_ratio}")
    if train_times.isna().any() or test_times.isna().any():
        raise RuntimeError("invalid event timestamps")
    if train_times.max() >= test_times.min():
        raise RuntimeError(
            f"temporal order violated: {train_times.max()} >= {test_times.min()}"
        )

    output_root = Path(args.output_root)
    all_csv = output_root / "s2_v9_all_512.csv"
    train_csv = output_root / "temporal_split" / "s2_v9_train.csv"
    test_csv = output_root / "temporal_split" / "s2_v9_test.csv"
    write_csv_atomic(complete, all_csv)
    write_csv_atomic(train, train_csv)
    write_csv_atomic(test, test_csv)
    report = {
        "finished_at": utc_now(),
        "input_rows": len(frame),
        "complete_rows": len(complete),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_ratio": train_ratio,
        "train_event_max": str(train_times.max()),
        "test_event_min": str(test_times.min()),
        "event_overlap": len(event_overlap),
        "invalid_header_files": len(invalid_headers),
        "dropped_event_groups": bad_event_groups,
        "dropped_rows": int(drop_mask.sum()),
        "content_samples": sample_count,
        "normalized_direct_files": normalized,
        "reused_direct_files": reused,
        "all_csv": str(all_csv),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
    }
    write_json_atomic(report, output_root / "finalize_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_historical_bounds_native_v9/s2_v9_recrop_progress.csv"
        ),
    )
    parser.add_argument(
        "--target-root",
        default=(
            "/mnt/engg-niulab/yuyao/preprocessed_512/"
            "S2_6time_historical_bounds_native_v9"
        ),
    )
    parser.add_argument(
        "--cache-root",
        default="/diniuvol/yuyao/s2_v9_finalize_cache",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_historical_bounds_native_v9"
        ),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--content-samples", type=int, default=3000)
    parser.add_argument("--max-drop-rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260724)
    main(parser.parse_args())
