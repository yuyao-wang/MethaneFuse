#!/usr/bin/env python3
"""Build S5P and EMIT cutoff-2025-10 train/test CSVs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


S5P_INPUT = Path("/home/yuyao/methane_train/Upgrade_data_pipeline/csv/s5p_6time_samples.csv")
EMIT_TRAIN_INPUT = Path("Upgraded_dataset/emit32_event_clean/emit32_temporal_train_event_clean.csv")
EMIT_TEST_INPUT = Path("Upgraded_dataset/emit32_event_clean/emit32_temporal_test_event_clean.csv")

S5P_OUT_DIR = Path("Upgraded_dataset/s5p_6time_temporal_cutoff_2025_10_31")
EMIT_OUT_DIR = Path("Upgraded_dataset/emit32_temporal_cutoff_2025_10_31")

S5P_TRAIN_MAX = pd.Timestamp("2025-07-16", tz="UTC")
S5P_TEST_MIN = pd.Timestamp("2025-07-21", tz="UTC")
CUTOFF_MAX = pd.Timestamp("2025-10-31", tz="UTC")


def _base_plume_id(value: object) -> str:
    return re.sub(r"-[A-Za-z]+$", "", str(value).strip())


def _event_key(df: pd.DataFrame, time_col: str) -> pd.Series:
    times = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df["plume_id"].map(_base_plume_id) + "|" + times.fillna("BAD_TIME")


def _label_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df["label"].astype(int).value_counts().sort_index().items()}


def _time_summary(df: pd.DataFrame, time_col: str) -> Dict[str, str]:
    times = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    return {
        "min": str(times.min()),
        "max": str(times.max()),
        "bad": int(times.isna().sum()),
    }


def _split_audit(
    *,
    name: str,
    source: Iterable[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    time_col: str,
    split_rule: str,
) -> Dict[str, object]:
    train_keys = set(_event_key(train, time_col))
    test_keys = set(_event_key(test, time_col))
    leakage = sorted(train_keys & test_keys)
    total = len(train) + len(test)
    return {
        "name": name,
        "source": list(source),
        "split_rule": split_rule,
        "event_key": 'plume_id stripped with r"-[A-Za-z]+$" plus normalized event time',
        "train_csv": "",
        "test_csv": "",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_ratio": float(len(train) / total) if total else 0.0,
        "test_ratio": float(len(test) / total) if total else 0.0,
        "train_label_counts": _label_counts(train),
        "test_label_counts": _label_counts(test),
        "train_unique_events": int(len(train_keys)),
        "test_unique_events": int(len(test_keys)),
        "event_leakage_count": int(len(leakage)),
        "event_leakage_examples": leakage[:10],
        "train_time": _time_summary(train, time_col),
        "test_time": _time_summary(test, time_col),
    }


def build_s5p() -> Dict[str, object]:
    df = pd.read_csv(S5P_INPUT, low_memory=False)
    required = {"plume_id", "plume_time", "image_path", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"S5P missing required columns: {missing}")

    out = df.copy()
    out["_event_time"] = pd.to_datetime(out["plume_time"], utc=True, errors="coerce")
    bad_time = out["_event_time"].isna()
    if bad_time.any():
        raise ValueError(f"S5P has bad plume_time rows: {int(bad_time.sum())}")

    dates = out["_event_time"].dt.date
    train_mask = dates <= S5P_TRAIN_MAX.date()
    test_mask = (dates >= S5P_TEST_MIN.date()) & (dates <= CUTOFF_MAX.date())
    dropped_gap = int(((dates > S5P_TRAIN_MAX.date()) & (dates < S5P_TEST_MIN.date())).sum())
    dropped_after_cutoff = int((dates > CUTOFF_MAX.date()).sum())

    train = out.loc[train_mask].drop(columns=["_event_time"])
    test = out.loc[test_mask].drop(columns=["_event_time"])

    S5P_OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_csv = S5P_OUT_DIR / "s5p_6time_train_cutoff_2025_10_31.csv"
    test_csv = S5P_OUT_DIR / "s5p_6time_test_cutoff_2025_10_31.csv"
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)

    audit = _split_audit(
        name="s5p_6time_cutoff_2025_10_31",
        source=[str(S5P_INPUT)],
        train=train,
        test=test,
        time_col="plume_time",
        split_rule=(
            "S5P old pipeline date window: train UTC plume_time date <= 2025-07-16; "
            "test UTC plume_time date >= 2025-07-21 and <= 2025-10-31; "
            "2025-07-17..2025-07-20 and >2025-10-31 dropped"
        ),
    )
    audit["train_csv"] = str(train_csv.resolve())
    audit["test_csv"] = str(test_csv.resolve())
    audit["dropped_gap_rows"] = dropped_gap
    audit["dropped_after_cutoff_rows"] = dropped_after_cutoff
    (S5P_OUT_DIR / "split_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    return audit


def build_emit() -> Dict[str, object]:
    train = pd.read_csv(EMIT_TRAIN_INPUT, low_memory=False)
    test = pd.read_csv(EMIT_TEST_INPUT, low_memory=False)
    required = {"plume_id", "event_time", "label"}
    for name, df in [("train", train), ("test", test)]:
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"EMIT {name} missing required columns: {missing}")

    train_time = pd.to_datetime(train["event_time"], utc=True, errors="coerce")
    test_time = pd.to_datetime(test["event_time"], utc=True, errors="coerce")
    if train_time.isna().any() or test_time.isna().any():
        raise ValueError(
            f"EMIT bad event_time rows: train={int(train_time.isna().sum())} test={int(test_time.isna().sum())}"
        )

    train_filtered = train.loc[train_time.dt.date <= CUTOFF_MAX.date()].copy()
    test_filtered = test.loc[test_time.dt.date <= CUTOFF_MAX.date()].copy()

    EMIT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_csv = EMIT_OUT_DIR / "emit32_temporal_train_cutoff_2025_10_31.csv"
    test_csv = EMIT_OUT_DIR / "emit32_temporal_test_cutoff_2025_10_31.csv"
    train_filtered.to_csv(train_csv, index=False)
    test_filtered.to_csv(test_csv, index=False)

    audit = _split_audit(
        name="emit32_temporal_cutoff_2025_10_31",
        source=[str(EMIT_TRAIN_INPUT), str(EMIT_TEST_INPUT)],
        train=train_filtered,
        test=test_filtered,
        time_col="event_time",
        split_rule=(
            "Keep old EMIT event-clean train/test split; filter both splits to UTC event_time date <= 2025-10-31"
        ),
    )
    audit["train_csv"] = str(train_csv.resolve())
    audit["test_csv"] = str(test_csv.resolve())
    audit["dropped_train_after_cutoff_rows"] = int(len(train) - len(train_filtered))
    audit["dropped_test_after_cutoff_rows"] = int(len(test) - len(test_filtered))
    (EMIT_OUT_DIR / "split_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    return audit


def main() -> int:
    for audit in [build_s5p(), build_emit()]:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
