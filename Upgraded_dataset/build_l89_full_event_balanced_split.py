#!/usr/bin/env python3
"""Balance L89 patch repetition while preserving the original temporal split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def add_event_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["event_id_split"] = (
        result["plume_id"].astype(str).str.replace(
            r"-[A-Za-z0-9]+$", "", regex=True
        )
        + "|"
        + result["event_time"].astype(str)
    )
    event_time = pd.to_datetime(result["event_time"], utc=True)
    result["event_month_split"] = event_time.dt.strftime("%Y-%m")
    return result


def cap_event_label_rows(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if cap <= 0:
        return frame.copy()
    ranked = frame.copy()
    hashed = pd.util.hash_pandas_object(
        ranked[["id", "event_id_split", "label"]],
        index=False,
        hash_key="0123456789abcdef",
    ).astype(np.uint64)
    ranked["_sample_rank"] = hashed ^ np.uint64(seed)
    ranked = ranked.sort_values(
        ["event_id_split", "label", "_sample_rank"], kind="stable"
    )
    ranked = ranked.groupby(["event_id_split", "label"], observed=True).head(cap)
    return ranked.drop(columns="_sample_rank").sort_values("id", kind="stable")


def clean_output(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["event_id_split", "event_month_split"])


def counts(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "events": int(frame["event_id_split"].nunique()),
        "labels": {
            str(key): int(value)
            for key, value in frame["label"].value_counts().sort_index().items()
        },
        "months": {
            str(key): int(value)
            for key, value in frame["event_month_split"]
            .value_counts()
            .sort_index()
            .items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_cap_per_event_label", type=int, default=64)
    parser.add_argument("--test_cap_per_event_label", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20251031)
    args = parser.parse_args()

    original_train = add_event_fields(pd.read_csv(args.train_csv, low_memory=False))
    original_test = add_event_fields(pd.read_csv(args.test_csv, low_memory=False))
    train_events = set(original_train["event_id_split"])
    test_events_before = set(original_test["event_id_split"])
    overlap = train_events & test_events_before

    raw_train = original_train.copy()
    raw_test = original_test[
        ~original_test["event_id_split"].isin(overlap)
    ].copy()
    test_events = set(raw_test["event_id_split"])
    if train_events & test_events:
        raise RuntimeError("Event leakage remains after filtering")

    train = cap_event_label_rows(
        raw_train, args.train_cap_per_event_label, args.seed
    )
    balanced_test = cap_event_label_rows(
        raw_test, args.test_cap_per_event_label, args.seed + 1
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "L89_temporal_train_full_event_balanced.csv"
    raw_test_path = output_dir / "L89_temporal_test_full_event_raw.csv"
    balanced_test_path = output_dir / "L89_temporal_test_full_event_balanced.csv"
    clean_output(train).to_csv(train_path, index=False)
    clean_output(raw_test).to_csv(raw_test_path, index=False)
    clean_output(balanced_test).to_csv(balanced_test_path, index=False)

    unique_events = train_events | test_events
    summary = {
        "method": (
            "Preserve the original temporal assignment, remove any overlapping "
            "test event, and cap repeated patches per event and label."
        ),
        "seed": args.seed,
        "event_key": "plume_id without final site suffix + exact event_time",
        "overlap_events_removed_from_test": sorted(overlap),
        "event_leakage_count": 0,
        "event_train_fraction": len(train_events) / len(unique_events),
        "row_train_fraction_before_caps": len(raw_train)
        / (len(raw_train) + len(raw_test)),
        "original_train": counts(original_train),
        "original_test": counts(original_test),
        "train": counts(train),
        "test_raw": counts(raw_test),
        "test_balanced": counts(balanced_test),
        "train_cap_per_event_label": args.train_cap_per_event_label,
        "test_cap_per_event_label": args.test_cap_per_event_label,
        "paths": {
            "train": str(train_path),
            "test_raw": str(raw_test_path),
            "test_balanced": str(balanced_test_path),
        },
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
