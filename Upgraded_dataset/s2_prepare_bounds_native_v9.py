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

import pandas as pd


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
DIRECT_POLICIES = {
    ("t0", "historical_cdse0_t0"),
    ("seasonal", "historical_gee_direct"),
    ("year", "historical_gee_direct"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def target_path(root: Path, timepoint: str, plume_id: str) -> Path:
    return root / timepoint / plume_id / FILENAMES[timepoint]


def stage_copy(
    source: Path,
    target: Path,
    cache_root: Path,
    plume_id: str,
    timepoint: str,
) -> tuple[str, int]:
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return "reused", target.stat().st_size
    cache = cache_root / timepoint / plume_id / source.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache.with_name(
        f".{cache.name}.{os.getpid()}.{threading.get_ident()}.part"
    )
    temporary_target = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.part"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(source, temporary_cache)
        os.replace(temporary_cache, cache)
        shutil.copyfile(cache, temporary_target)
        os.replace(temporary_target, target)
    finally:
        temporary_cache.unlink(missing_ok=True)
        temporary_target.unlink(missing_ok=True)
        cache.unlink(missing_ok=True)
    if timepoint == "t0":
        sidecar = target.with_name(target.name + ".georef.json")
        sidecar.write_text(
            json.dumps(
                {
                    "state": "complete",
                    "created_at": utc_now(),
                    "source_kind": "historical_cdse0_exact_512",
                    "source_path": str(source),
                    "center_mode": "historical_plume_bounds",
                    "dn_add": 0,
                },
                indent=2,
            )
            + "\n"
        )
    return "copied", target.stat().st_size


def main(args: argparse.Namespace) -> None:
    source_table = Path(args.source_table)
    target_root = Path(args.target_root)
    cache_root = Path(args.cache_root)
    frame = pd.read_csv(source_table, low_memory=False)
    tasks = []
    direct_counts = {}
    for timepoint in TIMEPOINTS:
        policy_column = f"{timepoint}_v8_policy"
        final_column = FINAL_COLUMNS[timepoint]
        target_column = f"{timepoint}_v9_target_path"
        frame[target_column] = [
            str(target_path(target_root, timepoint, str(plume_id)))
            for plume_id in frame["plume_id"]
        ]
        direct_mask = frame[policy_column].map(
            lambda policy: (timepoint, str(policy)) in DIRECT_POLICIES
        )
        direct_counts[timepoint] = int(direct_mask.sum())
        for plume_id, source_value, target_value in frame.loc[
            direct_mask, ["plume_id", final_column, target_column]
        ].itertuples(index=False, name=None):
            source = Path(str(source_value))
            target = Path(str(target_value))
            if not source.is_file():
                raise FileNotFoundError(source)
            tasks.append(
                {
                    "plume_id": str(plume_id),
                    "timepoint": timepoint,
                    "source": source,
                    "target": target,
                }
            )

    report = {
        "created_at": utc_now(),
        "source_table": str(source_table),
        "output_table": args.output_table,
        "target_root": str(target_root),
        "cache_root": str(cache_root),
        "rows": len(frame),
        "direct_counts": direct_counts,
        "direct_tasks": len(tasks),
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return

    cache_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    reused = 0
    copied_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                stage_copy,
                task["source"],
                task["target"],
                cache_root,
                task["plume_id"],
                task["timepoint"],
            )
            for task in tasks
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            status, size = future.result()
            copied += int(status == "copied")
            reused += int(status == "reused")
            copied_bytes += int(size)
            if completed % 500 == 0 or completed == len(futures):
                print(
                    f"[Direct] {completed}/{len(futures)} "
                    f"copied={copied} reused={reused}",
                    flush=True,
                )

    write_csv_atomic(frame, Path(args.output_table))
    report.update(
        finished_at=utc_now(),
        copied=copied,
        reused=reused,
        copied_bytes=copied_bytes,
    )
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-table",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_historical_exact_v8/s2_v8_all_512.csv"
        ),
    )
    parser.add_argument(
        "--output-table",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_historical_bounds_native_v9/s2_v9_prepared.csv"
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
        default="/diniuvol/yuyao/s2_v9_direct_stage",
    )
    parser.add_argument(
        "--report",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_historical_bounds_native_v9/prepare_report.json"
        ),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
