#!/usr/bin/env python3
"""Split S5P six-time samples by plume_id and plume_time cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


DEFAULT_INPUT_CSV = Path("/home/yuyao/methane_train/Upgrade_data_pipeline/csv/s5p_6time_samples.csv")
DEFAULT_OUTPUT_DIR = Path("Upgraded_dataset/s5p_6time_temporal_cutoff_2025_12_24")


def _label_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df["label"].astype(int).value_counts().sort_index().items()}


def _require_nonempty_text(df: pd.DataFrame, column: str) -> None:
    text = df[column].astype("string").str.strip().str.lower()
    invalid = df[column].isna() | text.isin(["", "nan", "none", "null", "<na>"])
    if invalid.any():
        examples = df.loc[invalid, ["plume_id", column]].head(5).to_dict("records")
        raise ValueError(f"Column '{column}' has {int(invalid.sum())} empty values. Examples: {examples}")


def split_s5p_6time(
    input_csv: Path,
    output_dir: Path,
    cutoff_date: str,
    *,
    train_name: str = "s5p_6time_train.csv",
    test_name: str = "s5p_6time_test.csv",
) -> dict:
    df = pd.read_csv(input_csv, low_memory=False)
    required_columns = ["plume_id", "plume_time", "image_path", "label"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column in ["plume_id", "image_path"]:
        _require_nonempty_text(df, column)

    df = df.copy()
    df["_plume_id"] = df["plume_id"].astype(str)
    df["_plume_time_utc"] = pd.to_datetime(df["plume_time"], utc=True, errors="coerce")
    invalid_time = df["_plume_time_utc"].isna()
    if invalid_time.any():
        examples = df.loc[invalid_time, ["plume_id", "plume_time"]].head(5).to_dict("records")
        raise ValueError(f"Invalid plume_time rows: {int(invalid_time.sum())}. Examples: {examples}")

    cutoff = pd.Timestamp(cutoff_date, tz="UTC").date()
    group_times = df.groupby("_plume_id", sort=False)["_plume_time_utc"].agg(["min", "max"])
    cross_cutoff = group_times[(group_times["min"].dt.date <= cutoff) & (group_times["max"].dt.date > cutoff)]
    if not cross_cutoff.empty:
        examples = cross_cutoff.head(5).reset_index().to_dict("records")
        raise ValueError(f"{len(cross_cutoff)} plume_id groups cross the cutoff. Examples: {examples}")

    train_ids = set(group_times[group_times["max"].dt.date <= cutoff].index)
    test_ids = set(group_times[group_times["min"].dt.date > cutoff].index)
    if train_ids & test_ids:
        raise RuntimeError("Internal split error: train/test plume_id overlap.")

    train = df[df["_plume_id"].isin(train_ids)].drop(columns=["_plume_id", "_plume_time_utc"])
    test = df[df["_plume_id"].isin(test_ids)].drop(columns=["_plume_id", "_plume_time_utc"])
    leakage = set(train["plume_id"].astype(str)) & set(test["plume_id"].astype(str))
    if leakage:
        raise RuntimeError(f"event leakage_count={len(leakage)} examples={sorted(leakage)[:5]}")
    if train.empty or test.empty:
        raise ValueError(f"Empty split produced: train={len(train)} test={len(test)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / train_name
    test_csv = output_dir / test_name
    audit_json = output_dir / "split_audit.json"
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)

    audit = {
        "input_csv": str(input_csv),
        "cutoff_inclusive_utc_date": str(cutoff),
        "split_rule": (
            "train UTC plume_time date <= cutoff_date; test UTC plume_time date > cutoff_date; "
            "split grouped by plume_id"
        ),
        "source_rows": int(len(df)),
        "source_plume_ids": int(df["_plume_id"].nunique()),
        "source_label_counts": _label_counts(df),
        "train_csv": str(train_csv.resolve()),
        "test_csv": str(test_csv.resolve()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_plume_ids": int(train["plume_id"].astype(str).nunique()),
        "test_plume_ids": int(test["plume_id"].astype(str).nunique()),
        "train_label_counts": _label_counts(train),
        "test_label_counts": _label_counts(test),
        "event_leakage_count": int(len(leakage)),
        "train_plume_time_min": str(pd.to_datetime(train["plume_time"], utc=True).min()),
        "train_plume_time_max": str(pd.to_datetime(train["plume_time"], utc=True).max()),
        "test_plume_time_min": str(pd.to_datetime(test["plume_time"], utc=True).min()),
        "test_plume_time_max": str(pd.to_datetime(test["plume_time"], utc=True).max()),
    }
    audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoff-date", default="2025-12-24")
    parser.add_argument("--train-name", default="s5p_6time_train.csv")
    parser.add_argument("--test-name", default="s5p_6time_test.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = split_s5p_6time(
        args.input_csv,
        args.output_dir,
        args.cutoff_date,
        train_name=args.train_name,
        test_name=args.test_name,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
