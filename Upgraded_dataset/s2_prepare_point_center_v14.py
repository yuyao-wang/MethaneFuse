#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


TIMEPOINTS = {
    "t0": ("t0_raw_path", "s2_0_std_512"),
    "prev1": ("prev1_raw_path", "s2_-7_std_512"),
    "prev2": ("prev2_raw_path", "s2_prev2_std_512"),
    "prev3": ("prev3_raw_path", "s2_prev3_std_512"),
    "seasonal": ("seasonal_raw_path", "s2_-90_std_512"),
    "year": ("year_raw_path", "s2_-360_std_512"),
}
RAW_FILENAMES = {
    "t0": "s2.tif",
    "prev1": "s2_-7.tif",
    "prev2": "s2_prev2.tif",
    "prev3": "s2_prev3.tif",
    "seasonal": "s2_-90.tif",
    "year": "s2_-360.tif",
}
DAMAGED_SOURCE_CLASS = "S2_point_center_exact_v3"
FINAL_SOURCE_CLASS = "S2_point_center_plus1000_v14"


def file_ok(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def event_report(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    train_events = set(train["event_group_id"].astype(str))
    test_events = set(test["event_group_id"].astype(str))
    overlap = sorted(train_events & test_events)
    if overlap:
        raise ValueError(f"event leakage detected: {overlap[:20]}")
    return {
        "train": len(train_events),
        "test": len(test_events),
        "overlap": 0,
    }


def prepare(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False)
    train["split"] = "train"
    test["split"] = "test"
    events = event_report(train, test)
    combined = pd.concat([train, test], ignore_index=True)

    class_columns = [
        f"{canonical_column}_source_class"
        for _, canonical_column in TIMEPOINTS.values()
    ]
    missing = [column for column in class_columns if column not in combined]
    if missing:
        raise ValueError(f"missing source-class columns: {missing}")
    damaged_by_timepoint = {
        column: int(combined[column].eq(DAMAGED_SOURCE_CLASS).sum())
        for column in class_columns
    }
    damaged_mask = combined[class_columns].eq(DAMAGED_SOURCE_CLASS).any(axis=1)
    mixed_mask = (
        combined[class_columns].eq(DAMAGED_SOURCE_CLASS).sum(axis=1)
        .between(1, len(class_columns) - 1)
    )
    if mixed_mask.any():
        examples = combined.loc[mixed_mask, ["plume_id", *class_columns]].head(10)
        raise ValueError(
            "rows mix damaged and trusted 512 sources:\n"
            + examples.to_string(index=False)
        )

    damaged = combined.loc[damaged_mask].copy()
    trusted = combined.loc[~damaged_mask].copy()
    damaged_path = output_root / "point_rows_to_recrop.csv"
    seed_path = output_root / "all_rows_seed.csv"
    damaged.to_csv(damaged_path, index=False)
    combined.to_csv(seed_path, index=False)

    report = {
        "mode": "prepare",
        "rows": {
            "all": len(combined),
            "damaged_point_rows": len(damaged),
            "trusted_legacy_rows": len(trusted),
        },
        "split_rows": combined["split"].value_counts().to_dict(),
        "damaged_by_timepoint": damaged_by_timepoint,
        "events": events,
        "outputs": {
            "point_rows_to_recrop": str(damaged_path),
            "all_rows_seed": str(seed_path),
        },
    }
    (output_root / "prepare_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


def finalize(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    seed = pd.read_csv(args.seed_csv, low_memory=False)
    recropped = pd.read_csv(args.recropped_csv, low_memory=False)
    if seed["plume_id"].astype(str).duplicated().any():
        raise ValueError("seed table contains duplicate plume_id values")
    if recropped["plume_id"].astype(str).duplicated().any():
        raise ValueError("recrop table contains duplicate plume_id values")

    recropped = recropped.set_index(recropped["plume_id"].astype(str))
    if args.legacy_recovery_report:
        recovery = json.loads(Path(args.legacy_recovery_report).read_text())
        if recovery.get("failed"):
            raise ValueError(
                f"legacy recovery contains {recovery['failed']} failures"
            )
        for record in recovery.get("records", []):
            plume_id = str(record["plume_id"])
            timepoint = str(record["timepoint"])
            raw_column, _ = TIMEPOINTS[timepoint]
            recropped.at[plume_id, raw_column] = record["target_path"]
            recropped.at[
                plume_id, f"{timepoint}_recrop_status"
            ] = "legacy_512_recentered"
            recropped.at[
                plume_id, f"{timepoint}_recrop_message"
            ] = (
                f"legacy 512 recentered shift_x={record['shift_x']} "
                f"shift_y={record['shift_y']}"
            )
    if args.repair_tables:
        for repair_path in (
            value.strip()
            for value in args.repair_tables.split(",")
            if value.strip()
        ):
            repair = pd.read_csv(repair_path, low_memory=False)
            for row in repair.to_dict("records"):
                plume_id = str(row["plume_id"])
                if plume_id not in recropped.index:
                    continue
                for timepoint, (raw_column, _) in TIMEPOINTS.items():
                    status = str(
                        row.get(f"{timepoint}_recrop_status", "")
                    ).strip()
                    path = str(row.get(raw_column, "")).strip()
                    if status not in {"downloaded", "target_exists"}:
                        continue
                    if not path or not file_ok(path):
                        continue
                    recropped.at[plume_id, raw_column] = path
                    recropped.at[
                        plume_id, f"{timepoint}_recrop_status"
                    ] = status
                    recropped.at[
                        plume_id, f"{timepoint}_recrop_message"
                    ] = str(
                        row.get(f"{timepoint}_recrop_message", "")
                    )
    if args.expected_recrop_root:
        physical_root = Path(args.expected_recrop_root).resolve()
        plume_ids = recropped["plume_id"].astype(str)
        for timepoint, (raw_column, _) in TIMEPOINTS.items():
            recropped[raw_column] = plume_ids.map(
                lambda plume_id: str(
                    physical_root
                    / timepoint
                    / plume_id
                    / RAW_FILENAMES[timepoint]
                )
            )
            recropped[f"{timepoint}_recrop_status"] = "target_exists"
            recropped[
                f"{timepoint}_recrop_message"
            ] = "reconciled from physical V14 target"
    point_mask = seed["plume_id"].astype(str).isin(recropped.index)
    expected_point_rows = int(
        seed[f"s2_0_std_512_source_class"].eq(DAMAGED_SOURCE_CLASS).sum()
    )
    if int(point_mask.sum()) != expected_point_rows:
        raise ValueError(
            f"recropped rows={int(point_mask.sum())}, expected={expected_point_rows}"
        )

    failures: list[str] = []
    expected_root = (
        str(Path(args.expected_recrop_root).resolve())
        if args.expected_recrop_root
        else ""
    )
    for timepoint, (raw_column, canonical_column) in TIMEPOINTS.items():
        status_column = f"{timepoint}_recrop_status"
        if status_column not in recropped:
            failures.append(f"missing status column {status_column}")
            continue
        bad_status = ~recropped[status_column].isin(
            ["downloaded", "target_exists", "legacy_512_recentered"]
        )
        if bad_status.any():
            examples = recropped.loc[
                bad_status, ["plume_id", status_column]
            ].head(5)
            failures.append(
                f"{timepoint} bad statuses: {examples.to_dict('records')}"
            )
        if expected_root:
            outside_root = ~recropped[raw_column].astype(str).map(
                lambda value: value == expected_root
                or value.startswith(expected_root + os.sep)
            )
            if outside_root.any():
                examples = recropped.loc[
                    outside_root, ["plume_id", raw_column]
                ].head(5)
                failures.append(
                    f"{timepoint} paths outside final root: "
                    f"{examples.to_dict('records')}"
                )
        recrop_paths = recropped[raw_column].astype(str).to_dict()
        seed.loc[point_mask, canonical_column] = (
            seed.loc[point_mask, "plume_id"].astype(str).map(recrop_paths)
        )
        seed.loc[
            point_mask, f"{canonical_column}_source_class"
        ] = FINAL_SOURCE_CLASS
    if failures:
        raise ValueError("; ".join(failures))

    reconciled_path = output_root / "point_rows_reconciled.csv"
    recropped.to_csv(reconciled_path, index=False)

    all_paths = [
        str(value)
        for _, canonical_column in TIMEPOINTS.values()
        for value in seed[canonical_column]
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        statuses = list(executor.map(file_ok, all_paths))
    missing_paths = [
        path for path, status in zip(all_paths, statuses) if not status
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} final 512 paths are missing: "
            f"{missing_paths[:10]}"
        )

    train = seed[seed["split"].eq("train")].copy()
    test = seed[seed["split"].eq("test")].copy()
    events = event_report(train, test)
    all_path = output_root / "s2_v14_all_512.csv"
    train_path = output_root / "s2_v14_train_512.csv"
    test_path = output_root / "s2_v14_test_512.csv"
    seed.to_csv(all_path, index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    report = {
        "mode": "finalize",
        "rows": {
            "all": len(seed),
            "train": len(train),
            "test": len(test),
            "recropped": int(point_mask.sum()),
            "trusted_legacy": int((~point_mask).sum()),
        },
        "events": events,
        "checked_files": len(all_paths),
        "missing_files": 0,
        "outputs": {
            "all": str(all_path),
            "train": str(train_path),
            "test": str(test_path),
            "point_rows_reconciled": str(reconciled_path),
        },
    }
    (output_root / "finalize_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--train-csv", required=True)
    prepare_parser.add_argument("--test-csv", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.set_defaults(function=prepare)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--seed-csv", required=True)
    finalize_parser.add_argument("--recropped-csv", required=True)
    finalize_parser.add_argument("--legacy-recovery-report", default="")
    finalize_parser.add_argument("--repair-tables", default="")
    finalize_parser.add_argument("--expected-recrop-root", default="")
    finalize_parser.add_argument("--output-root", required=True)
    finalize_parser.add_argument("--workers", type=int, default=64)
    finalize_parser.set_defaults(function=finalize)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
