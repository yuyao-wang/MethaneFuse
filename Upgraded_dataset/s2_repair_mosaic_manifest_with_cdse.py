#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

import s2_resolve_point_covering_tiles as resolver


TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
MOSAIC_STATUSES = {"local_mosaic_repaired", "stac_mosaic_repaired"}
CLEAR_SUFFIXES = (
    "stac_item_id",
    "stac_base_url",
    "stac_cog_base_url",
    "stac_dn_add",
    "stac_source_product_name",
    "stac_mosaic_json",
    "local_mosaic_json",
    "cdse_mosaic_json",
)
COMPLETE_STATUSES = {"downloaded", "target_exists"}


def clean(value: Any) -> str:
    return resolver.clean(value)


def save_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(
        path.name + f".tmp.{os.getpid()}"
    )
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def repair_task(task: dict[str, Any], retries: int) -> dict[str, Any]:
    wanted = resolver.parse_product(task["original_product_name"])
    try:
        record = resolver.cdse_point_record(
            wanted,
            task,
            retries,
            force_mosaic=True,
        )
    except Exception as exc:
        return {
            **task,
            "repair": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if record is None:
        return {
            **task,
            "repair": None,
            "error": (
                "no alternate CDSE product or same-acquisition tile "
                "mosaic fully covers the 512 crop"
            ),
        }
    margin = float(record["tile_margin_pixels"])
    if (
        margin < 0
        and record["tile_repair_status"] != "cdse_mosaic_repaired"
    ):
        return {
            **task,
            "repair": None,
            "error": f"CDSE footprint margin is negative: {margin:.3f}",
        }
    return {**task, "repair": record, "error": ""}


def apply_repair(
    frame: pd.DataFrame,
    index: int,
    timepoint: str,
    repair: dict[str, Any],
) -> None:
    frame.at[index, f"{timepoint}_product_name"] = repair[
        "selected_product_name"
    ]
    frame.at[index, f"{timepoint}_product_id"] = repair[
        "selected_product_id"
    ]
    for suffix in CLEAR_SUFFIXES:
        column = f"{timepoint}_{suffix}"
        if column not in frame.columns:
            frame[column] = ""
        frame.at[index, column] = ""
    cdse_mosaic = clean(repair.get("cdse_mosaic_json"))
    if cdse_mosaic:
        frame.at[index, f"{timepoint}_cdse_mosaic_json"] = (
            cdse_mosaic
        )
    for suffix in (
        "tile_repair_status",
        "tile_repair_note",
        "tile_margin_pixels",
        "center_offset_pixels",
    ):
        column = f"{timepoint}_{suffix}"
        if column not in frame.columns:
            frame[column] = ""
        frame.at[index, column] = repair.get(suffix, "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--resume-csv", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--request-retries", type=int, default=4)
    parser.add_argument("--min-request-interval", type=float, default=0.1)
    parser.add_argument(
        "--selection",
        choices=("mosaic", "failed"),
        default="mosaic",
    )
    parser.add_argument(
        "--http-cache",
        default="/diniuvol/yuyao/s2_stac_resolver_cache",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source_csv)
    resume_path = Path(args.resume_csv)
    source = pd.read_csv(source_path, low_memory=False)
    resume = pd.read_csv(resume_path, low_memory=False)
    if len(source) != len(resume):
        raise RuntimeError(
            f"row mismatch source={len(source)} resume={len(resume)}"
        )
    mutable_suffixes = (
        *CLEAR_SUFFIXES,
        "product_name",
        "product_id",
        "tile_repair_status",
        "tile_repair_note",
        "tile_margin_pixels",
        "center_offset_pixels",
        "recrop_status",
        "recrop_message",
    )
    for frame in (source, resume):
        for timepoint in TIMEPOINTS:
            for suffix in mutable_suffixes:
                column = f"{timepoint}_{suffix}"
                if column in frame.columns:
                    frame[column] = frame[column].astype(object)

    resolver._request_interval = max(
        0.0,
        float(args.min_request_interval),
    )
    resolver._http_cache_root = Path(args.http_cache)
    resolver._http_cache_root.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for index, row in source.iterrows():
        for timepoint in TIMEPOINTS:
            status = clean(row.get(f"{timepoint}_tile_repair_status"))
            resume_status = clean(
                resume.at[index, f"{timepoint}_recrop_status"]
            )
            if args.selection == "mosaic":
                if status not in MOSAIC_STATUSES:
                    continue
                original_name = clean(
                    row.get(
                        f"{timepoint}_tile_repair_original_product_name"
                    )
                ) or clean(row.get(f"{timepoint}_product_name"))
            else:
                if (
                    not resume_status
                    or resume_status in COMPLETE_STATUSES
                ):
                    continue
                original_name = clean(
                    row.get(f"{timepoint}_product_name")
                )
            tasks.append(
                {
                    "row_index": int(index),
                    "plume_id": clean(row.get("plume_id")),
                    "timepoint": timepoint,
                    "product_name": original_name,
                    "product_id": clean(
                        row.get(
                            f"{timepoint}_tile_repair_original_product_id"
                        )
                    ),
                    "original_product_name": original_name,
                    "longitude": float(row["plume_longitude"]),
                    "latitude": float(row["plume_latitude"]),
                }
            )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                repair_task,
                task,
                args.request_retries,
            ): task
            for task in tasks
        }
        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            results.append(future.result())
            if completed % 50 == 0 or completed == len(futures):
                failures = sum(
                    result["repair"] is None for result in results
                )
                print(
                    f"mosaics={completed}/{len(futures)} "
                    f"unresolved={failures}",
                    flush=True,
                )

    repaired_source = 0
    repaired_resume = 0
    kept_completed = 0
    unresolved: list[dict[str, Any]] = []
    for result in results:
        repair = result["repair"]
        if repair is None:
            unresolved.append(
                {
                    "plume_id": result["plume_id"],
                    "timepoint": result["timepoint"],
                    "product_name": result["original_product_name"],
                    "error": result["error"],
                }
            )
            continue
        index = int(result["row_index"])
        timepoint = result["timepoint"]
        apply_repair(source, index, timepoint, repair)
        repaired_source += 1

        status = clean(
            resume.at[index, f"{timepoint}_recrop_status"]
        )
        raw_path = Path(
            clean(resume.at[index, f"{timepoint}_raw_path"])
        )
        if (
            status in COMPLETE_STATUSES
            and raw_path.is_file()
            and raw_path.stat().st_size > 0
        ):
            kept_completed += 1
            continue
        apply_repair(resume, index, timepoint, repair)
        resume.at[index, f"{timepoint}_recrop_status"] = ""
        resume.at[index, f"{timepoint}_recrop_message"] = ""
        repaired_resume += 1

    save_atomic(source, source_path)
    save_atomic(resume, resume_path)
    report = {
        "selection": args.selection,
        "selected_tasks": len(tasks),
        "repaired_source": repaired_source,
        "repaired_resume": repaired_resume,
        "kept_completed": kept_completed,
        "unresolved": unresolved,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
