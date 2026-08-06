#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
RAW_FILENAMES = {
    "t0": "s2.tif",
    "prev1": "s2_-7.tif",
    "prev2": "s2_prev2.tif",
    "prev3": "s2_prev3.tif",
    "seasonal": "s2_-90.tif",
    "year": "s2_-360.tif",
}


def save_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def combine(
    current: pd.DataFrame,
    restored: pd.DataFrame,
    plume_order: dict[str, int],
) -> pd.DataFrame:
    current_ids = set(current["plume_id"].astype(str))
    additions = restored.loc[
        ~restored["plume_id"].astype(str).isin(current_ids)
    ]
    combined = pd.concat(
        [current, additions],
        ignore_index=True,
        sort=False,
    )
    combined["_plume_order"] = (
        combined["plume_id"].astype(str).map(plume_order)
    )
    if combined["_plume_order"].isna().any():
        raise RuntimeError("combined table contains unknown plume IDs")
    combined = combined.sort_values(
        "_plume_order",
        kind="stable",
    ).drop(columns="_plume_order")
    return combined.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-csv", required=True)
    parser.add_argument("--resolved-restored-csv", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--resume-csv", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--backup-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original = pd.read_csv(args.original_csv, low_memory=False)
    restored = pd.read_csv(
        args.resolved_restored_csv,
        low_memory=False,
    )
    source_path = Path(args.source_csv)
    resume_path = Path(args.resume_csv)
    source = pd.read_csv(source_path, low_memory=False)
    resume = pd.read_csv(resume_path, low_memory=False)
    if set(source["plume_id"].astype(str)) != set(
        resume["plume_id"].astype(str)
    ):
        raise RuntimeError("source and resume plume sets differ")

    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, backup_dir / source_path.name)
    shutil.copy2(resume_path, backup_dir / resume_path.name)

    plume_order = {
        plume_id: index
        for index, plume_id in enumerate(
            original["plume_id"].astype(str)
        )
    }
    source = combine(source, restored, plume_order)
    resume = combine(resume, restored, plume_order)
    restored_ids = set(restored["plume_id"].astype(str))
    target_root = Path(args.target_root)

    reset_failures = 0
    restored_existing = 0
    for timepoint in TIMEPOINTS:
        status_column = f"{timepoint}_recrop_status"
        message_column = f"{timepoint}_recrop_message"
        for column in (status_column, message_column):
            if column not in resume.columns:
                resume[column] = ""
            resume[column] = resume[column].astype(object)
        failed = (
            resume[status_column]
            .fillna("")
            .astype(str)
            .eq("failed")
        )
        reset_failures += int(failed.sum())
        resume.loc[failed, status_column] = ""
        resume.loc[failed, message_column] = ""

        for index in resume.index[
            resume["plume_id"].astype(str).isin(restored_ids)
        ]:
            plume_id = str(resume.at[index, "plume_id"])
            target = (
                target_root
                / timepoint
                / plume_id
                / RAW_FILENAMES[timepoint]
            )
            complete = target.is_file() and target.stat().st_size > 0
            if timepoint == "t0":
                sidecar = target.with_suffix(target.suffix + ".georef.json")
                complete = complete and sidecar.is_file()
            if not complete:
                resume.at[index, status_column] = ""
                resume.at[index, message_column] = ""
                continue
            resume.at[index, status_column] = "target_exists"
            resume.at[index, message_column] = ""
            resume.at[index, f"{timepoint}_raw_path"] = str(target)
            resume.at[index, f"{timepoint}_path_source"] = (
                "download_s2_missing_from_clean_table:"
                "target_exists_on_restore"
            )
            resume.at[index, f"{timepoint}_local_status"] = "available"
            resume.at[index, f"{timepoint}_download_needed"] = 0
            resume.at[
                index,
                f"{timepoint}_download_target_raw_path",
            ] = ""
            resume.at[
                index,
                f"{timepoint}_matched_old_timepoint",
            ] = "product_crops"
            restored_existing += 1

    if len(source) != len(original) or len(resume) != len(original):
        raise RuntimeError(
            f"restored row mismatch source={len(source)} "
            f"resume={len(resume)} expected={len(original)}"
        )
    save_atomic(source, source_path)
    save_atomic(resume, resume_path)
    print(
        f"rows={len(source)} restored_rows={len(restored_ids)} "
        f"restored_existing_timepoints={restored_existing} "
        f"reset_failures={reset_failures}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
