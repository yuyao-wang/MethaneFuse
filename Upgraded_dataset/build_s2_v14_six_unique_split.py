#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


TIME_COLUMNS = [
    "t0_image_time",
    "prev1_image_time",
    "prev2_image_time",
    "prev3_image_time",
    "seasonal_image_time",
    "year_image_time",
]


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--minimum-test-ratio", type=float, default=0.10)
    parser.add_argument("--maximum-test-ratio", type=float, default=0.20)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_csv, low_memory=False)
    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False)

    missing_columns = [
        column for column in ["plume_id", "event_group_id", *TIME_COLUMNS]
        if column not in metadata
    ]
    if missing_columns:
        raise ValueError(f"metadata missing columns: {missing_columns}")

    normalized_times = pd.DataFrame(index=metadata.index)
    for column in TIME_COLUMNS:
        parsed = pd.to_datetime(metadata[column], utc=True, errors="coerce")
        if parsed.isna().any():
            raise ValueError(
                f"{column} contains {int(parsed.isna().sum())} invalid timestamps"
            )
        normalized_times[column] = parsed.dt.round("s")

    metadata = metadata[["plume_id", "event_group_id", *TIME_COLUMNS]].copy()
    metadata["unique_acquisition_dates"] = normalized_times.nunique(axis=1)
    eligible = set(
        metadata.loc[
            metadata["unique_acquisition_dates"].eq(len(TIME_COLUMNS)),
            "plume_id",
        ].astype(str)
    )

    filtered_train = train[
        train["plume_id"].astype(str).isin(eligible)
    ].copy()
    filtered_test = test[
        test["plume_id"].astype(str).isin(eligible)
    ].copy()
    filtered_train["unique_acquisition_dates"] = len(TIME_COLUMNS)
    filtered_test["unique_acquisition_dates"] = len(TIME_COLUMNS)

    train_plumes = set(filtered_train["plume_id"].astype(str))
    test_plumes = set(filtered_test["plume_id"].astype(str))
    train_events = set(filtered_train["event_group_id"].astype(str))
    test_events = set(filtered_test["event_group_id"].astype(str))
    if train_plumes & test_plumes:
        raise ValueError("plume leakage detected")
    if train_events & test_events:
        raise ValueError("event leakage detected")

    all_rows = pd.concat(
        [filtered_train, filtered_test],
        ignore_index=True,
    )
    ratios = {
        "patch": len(filtered_test) / len(all_rows),
        "plume": len(test_plumes) / (len(train_plumes) + len(test_plumes)),
        "event": len(test_events) / (len(train_events) + len(test_events)),
    }
    invalid_ratios = {
        name: ratio
        for name, ratio in ratios.items()
        if not args.minimum_test_ratio <= ratio <= args.maximum_test_ratio
    }
    if invalid_ratios:
        raise ValueError(
            f"test ratios outside [{args.minimum_test_ratio}, "
            f"{args.maximum_test_ratio}]: {invalid_ratios}"
        )

    output_root = Path(args.output_root)
    train_path = output_root / "train.csv"
    test_path = output_root / "test.csv"
    all_path = output_root / "all.csv"
    write_csv_atomic(filtered_train, train_path)
    write_csv_atomic(filtered_test, test_path)
    write_csv_atomic(all_rows, all_path)

    report = {
        "rule": "retain plumes with six distinct acquisition timestamps",
        "time_columns": TIME_COLUMNS,
        "rows": {
            "all": len(all_rows),
            "train": len(filtered_train),
            "test": len(filtered_test),
        },
        "plumes": {
            "all": len(train_plumes) + len(test_plumes),
            "train": len(train_plumes),
            "test": len(test_plumes),
            "overlap": 0,
        },
        "events": {
            "all": len(train_events) + len(test_events),
            "train": len(train_events),
            "test": len(test_events),
            "overlap": 0,
        },
        "labels": {
            "train": filtered_train["label"].value_counts().sort_index().to_dict(),
            "test": filtered_test["label"].value_counts().sort_index().to_dict(),
        },
        "test_ratios": ratios,
        "outputs": {
            "all": str(all_path.resolve()),
            "train": str(train_path.resolve()),
            "test": str(test_path.resolve()),
        },
    }
    report_path = output_root / "split_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
