#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--maximum-event-time", default="2025-10-31T23:59:59Z")
    parser.add_argument("--cutoff", default="2025-07-27T00:00:00Z")
    args = parser.parse_args()

    source = pd.read_csv(args.input_csv, low_memory=False)
    required_columns = {"event_group_id", "plume_id", "event_time", "label"}
    missing_columns = sorted(required_columns - set(source.columns))
    if missing_columns:
        raise ValueError(f"Input CSV is missing columns: {missing_columns}")

    event_times = pd.to_datetime(source["event_time"], utc=True, errors="coerce")
    if event_times.isna().any():
        raise ValueError(f"event_time has {int(event_times.isna().sum())} invalid values")

    source = source.copy()
    source["_event_time"] = event_times
    group_time_counts = source.groupby("event_group_id")["_event_time"].nunique()
    if group_time_counts.gt(1).any():
        invalid_groups = group_time_counts[group_time_counts.gt(1)].index[:10].tolist()
        raise ValueError(f"Events contain inconsistent event_time values: {invalid_groups}")

    maximum_event_time = pd.Timestamp(args.maximum_event_time)
    if maximum_event_time.tzinfo is None:
        maximum_event_time = maximum_event_time.tz_localize("UTC")
    else:
        maximum_event_time = maximum_event_time.tz_convert("UTC")

    cutoff = pd.Timestamp(args.cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    filtered = source[source["_event_time"].le(maximum_event_time)].copy()
    train = filtered[filtered["_event_time"].lt(cutoff)].copy()
    test = filtered[filtered["_event_time"].ge(cutoff)].copy()
    train["split"] = "train"
    test["split"] = "test"
    all_rows = pd.concat([train, test], ignore_index=True)

    train_events = set(train["event_group_id"].astype(str))
    test_events = set(test["event_group_id"].astype(str))
    train_plumes = set(train["plume_id"].astype(str))
    test_plumes = set(test["plume_id"].astype(str))
    event_overlap = train_events & test_events
    plume_overlap = train_plumes & test_plumes
    if event_overlap:
        raise ValueError(f"Event leakage detected: {sorted(event_overlap)[:10]}")
    if plume_overlap:
        raise ValueError(f"Plume leakage detected: {sorted(plume_overlap)[:10]}")

    for frame in (train, test, all_rows):
        frame.drop(columns="_event_time", inplace=True)

    output_root = Path(args.output_root)
    all_path = output_root / "all.csv"
    train_path = output_root / "train.csv"
    test_path = output_root / "test.csv"
    write_csv_atomic(all_rows, all_path)
    write_csv_atomic(train, train_path)
    write_csv_atomic(test, test_path)

    total_rows = len(all_rows)
    total_plumes = len(train_plumes) + len(test_plumes)
    total_events = len(train_events) + len(test_events)
    audit = {
        "source_csv": str(Path(args.input_csv).resolve()),
        "rules": {
            "maximum_event_time_inclusive": maximum_event_time.isoformat(),
            "train": f"event_time < {cutoff.isoformat()}",
            "test": (
                f"{cutoff.isoformat()} <= event_time <= "
                f"{maximum_event_time.isoformat()}"
            ),
            "grouping": "event_group_id",
        },
        "rows": {"all": total_rows, "train": len(train), "test": len(test)},
        "plumes": {
            "all": total_plumes,
            "train": len(train_plumes),
            "test": len(test_plumes),
            "overlap": len(plume_overlap),
        },
        "events": {
            "all": total_events,
            "train": len(train_events),
            "test": len(test_events),
            "overlap": len(event_overlap),
        },
        "labels": {
            "train": train["label"].value_counts().sort_index().to_dict(),
            "test": test["label"].value_counts().sort_index().to_dict(),
        },
        "test_ratios": {
            "patch": len(test) / total_rows,
            "plume": len(test_plumes) / total_plumes,
            "event": len(test_events) / total_events,
        },
        "event_time_ranges": {
            "train": {
                "minimum": train["event_time"].min(),
                "maximum": train["event_time"].max(),
            },
            "test": {
                "minimum": test["event_time"].min(),
                "maximum": test["event_time"].max(),
            },
        },
        "outputs": {
            "all": str(all_path.resolve()),
            "train": str(train_path.resolve()),
            "test": str(test_path.resolve()),
        },
    }
    audit_path = output_root / "split_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
