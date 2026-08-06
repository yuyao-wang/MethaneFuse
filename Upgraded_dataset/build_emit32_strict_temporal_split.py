#!/usr/bin/env python3
"""Build a strict event-level temporal split for the complete EMIT32 dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUTS = (
    Path(
        "Upgraded_dataset/emit32_event_clean/"
        "emit32_temporal_train_event_clean.csv"
    ),
    Path(
        "Upgraded_dataset/emit32_event_clean/"
        "emit32_temporal_test_event_clean.csv"
    ),
)


def base_event_id(value: object) -> str:
    return re.sub(r"-[A-Za-z]+$", "", str(value).strip())


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def time_range(frame: pd.DataFrame, column: str) -> dict[str, str]:
    values = pd.to_datetime(frame[column], utc=True, errors="raise")
    return {"min": values.min().isoformat(), "max": values.max().isoformat()}


def label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in frame["label"]
        .astype(int)
        .value_counts()
        .sort_index()
        .items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        action="append",
        default=None,
        help="Input CSV; repeat for multiple existing splits.",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--target-test-ratio", type=float, default=0.15)
    parser.add_argument("--minimum-test-ratio", type=float, default=0.10)
    parser.add_argument("--maximum-test-ratio", type=float, default=0.20)
    args = parser.parse_args()

    input_paths = (
        tuple(Path(path) for path in args.input_csv)
        if args.input_csv
        else DEFAULT_INPUTS
    )
    frames = [pd.read_csv(path, low_memory=False) for path in input_paths]
    complete = pd.concat(frames, ignore_index=True)

    required = {"sample_id", "plume_id", "event_time", "label"}
    missing = sorted(required - set(complete.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if complete["sample_id"].duplicated().any():
        examples = complete.loc[
            complete["sample_id"].duplicated(False), "sample_id"
        ].head(10)
        raise ValueError(f"Duplicate sample_id values: {examples.tolist()}")

    complete = complete.copy()
    complete["_event_time"] = pd.to_datetime(
        complete["event_time"], utc=True, errors="raise"
    )
    complete["_base_event"] = complete["plume_id"].map(base_event_id)
    if complete["_base_event"].eq("").any():
        raise ValueError("At least one row has an empty base event ID")

    events = (
        complete.groupby("_base_event", observed=True)
        .agg(
            event_time=("_event_time", "min"),
            event_time_max=("_event_time", "max"),
            rows=("sample_id", "size"),
        )
        .reset_index()
    )
    inconsistent = events["event_time"].ne(events["event_time_max"])
    if inconsistent.any():
        examples = events.loc[
            inconsistent, ["_base_event", "event_time", "event_time_max"]
        ].head(10)
        raise ValueError(
            "A base event spans multiple event timestamps: "
            f"{examples.to_dict('records')}"
        )

    events["cutoff"] = events["event_time"].dt.floor("D")
    rows_by_cutoff = (
        events.groupby("cutoff", observed=True)["rows"].sum().sort_index()
    )
    test_rows = rows_by_cutoff.iloc[::-1].cumsum().iloc[::-1]
    candidates = pd.DataFrame({"test_rows": test_rows})
    candidates["test_ratio"] = candidates["test_rows"] / len(complete)
    candidates = candidates[
        candidates["test_ratio"].between(
            args.minimum_test_ratio,
            args.maximum_test_ratio,
            inclusive="both",
        )
    ].copy()
    if candidates.empty:
        raise RuntimeError(
            "No UTC date cutoff gives a test ratio in "
            f"[{args.minimum_test_ratio}, {args.maximum_test_ratio}]"
        )
    candidates["target_distance"] = (
        candidates["test_ratio"] - args.target_test_ratio
    ).abs()
    cutoff = candidates.sort_values(
        ["target_distance", "cutoff"], kind="stable"
    ).index[0]

    train = complete.loc[complete["_event_time"] < cutoff].copy()
    test = complete.loc[complete["_event_time"] >= cutoff].copy()
    train_events = set(train["_base_event"])
    test_events = set(test["_base_event"])
    event_overlap = train_events & test_events
    sample_overlap = set(train["sample_id"]) & set(test["sample_id"])
    if event_overlap or sample_overlap:
        raise RuntimeError(
            f"Leakage detected: events={len(event_overlap)}, "
            f"samples={len(sample_overlap)}"
        )
    if train["_event_time"].max() >= test["_event_time"].min():
        raise RuntimeError("Train and test event times are not strictly ordered")

    test_ratio = len(test) / len(complete)
    if not args.minimum_test_ratio <= test_ratio <= args.maximum_test_ratio:
        raise RuntimeError(f"Final test ratio is invalid: {test_ratio}")

    cutoff_tag = cutoff.strftime("%Y_%m_%d")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else Path(
            f"Upgraded_dataset/emit32_strict_temporal_cutoff_{cutoff_tag}"
        )
    )
    train_path = output_root / "emit32_temporal_train.csv"
    test_path = output_root / "emit32_temporal_test.csv"
    helper_columns = ["_event_time", "_base_event"]
    write_csv_atomic(train.drop(columns=helper_columns), train_path)
    write_csv_atomic(test.drop(columns=helper_columns), test_path)

    audit = {
        "source_csvs": [str(path.resolve()) for path in input_paths],
        "source_rows": int(len(complete)),
        "source_unique_samples": int(complete["sample_id"].nunique()),
        "source_unique_base_events": int(complete["_base_event"].nunique()),
        "selection": (
            "Choose the UTC date cutoff whose row-level test ratio is closest "
            "to the target while remaining within the allowed range."
        ),
        "target_test_ratio": args.target_test_ratio,
        "minimum_test_ratio": args.minimum_test_ratio,
        "maximum_test_ratio": args.maximum_test_ratio,
        "eligible_cutoff_dates": int(len(candidates)),
        "cutoff_utc": cutoff.isoformat(),
        "split_rule": "train event_time < cutoff; test event_time >= cutoff",
        "train_csv": str(train_path.resolve()),
        "test_csv": str(test_path.resolve()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_ratio": float(len(train) / len(complete)),
        "test_ratio": float(test_ratio),
        "train_unique_base_events": int(len(train_events)),
        "test_unique_base_events": int(len(test_events)),
        "base_event_leakage_count": int(len(event_overlap)),
        "sample_id_leakage_count": int(len(sample_overlap)),
        "train_label_counts": label_counts(train),
        "test_label_counts": label_counts(test),
        "train_event_time": time_range(train, "event_time"),
        "test_event_time": time_range(test, "event_time"),
        "train_t0_image_time": time_range(train, "t0_image_time"),
        "test_t0_image_time": time_range(test, "t0_image_time"),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "split_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
