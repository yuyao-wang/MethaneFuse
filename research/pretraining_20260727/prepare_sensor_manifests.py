#!/usr/bin/env python3
"""Prepare event-held-out validation and event-balanced exploration manifests.

The source test files are treated as immutable final-test sets.  This script
only partitions each source training CSV into an older training core and a
recent validation tail, then caps training rows per canonical event and label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SensorSpec:
    train_csv: Path
    test_csv: Path
    time_column: str
    event_key: Callable[[pd.DataFrame], pd.Series]
    id_candidates: tuple[str, ...]


def _base_event(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(
        r"-[A-Za-z0-9]+$", "", regex=True
    )


def _s2_event(frame: pd.DataFrame) -> pd.Series:
    return frame["event_group_id"].astype(str).str.strip()


def _plume_event(frame: pd.DataFrame) -> pd.Series:
    return _base_event(frame["plume_id"])


SPECS = {
    "s2": SensorSpec(
        REPO_ROOT
        / "Upgraded_dataset/s2_v14_six_unique_dates_temporal_split/train.csv",
        REPO_ROOT
        / "Upgraded_dataset/s2_v14_six_unique_dates_temporal_split/test.csv",
        "event_time",
        _s2_event,
        ("sample_id", "id"),
    ),
    "l89": SensorSpec(
        REPO_ROOT
        / (
            "Upgraded_dataset/l89_6time_temporal_hard_event_filtered_split/"
            "L89_temporal_train_hard_event_filtered.csv"
        ),
        REPO_ROOT
        / (
            "Upgraded_dataset/l89_6time_temporal_hard_event_filtered_split/"
            "L89_temporal_test_hard_event_filtered.csv"
        ),
        "event_time",
        _plume_event,
        ("id",),
    ),
    "emit": SensorSpec(
        REPO_ROOT
        / (
            "Upgraded_dataset/"
            "emit32_full_through_2026_05_strict_temporal_cutoff_2025_06_09/"
            "emit32_temporal_train.csv"
        ),
        REPO_ROOT
        / (
            "Upgraded_dataset/"
            "emit32_full_through_2026_05_strict_temporal_cutoff_2025_06_09/"
            "emit32_temporal_test.csv"
        ),
        "event_time",
        _plume_event,
        ("sample_id",),
    ),
    "s5p": SensorSpec(
        REPO_ROOT
        / (
            "Upgraded_dataset/s5p_6time_temporal_cutoff_2025_12_24/"
            "s5p_6time_train.csv"
        ),
        REPO_ROOT
        / (
            "Upgraded_dataset/s5p_6time_temporal_cutoff_2025_12_24/"
            "s5p_6time_test.csv"
        ),
        "plume_time",
        _plume_event,
        ("image_path", "plume_id"),
    ),
}


def _stable_rank(frame: pd.DataFrame, id_column: str, seed: int) -> pd.Series:
    def digest(value: object) -> int:
        payload = f"{seed}|{value}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

    return frame[id_column].map(digest)


def _choose_validation_events(
    frame: pd.DataFrame, val_fraction: float
) -> tuple[set[str], pd.Timestamp]:
    event_summary = (
        frame.groupby("_event", observed=True)
        .agg(_event_time=("_time", "max"), _rows=("_event", "size"))
        .sort_values(["_event_time", "_rows"], kind="mergesort")
    )
    target_rows = max(1, round(len(frame) * val_fraction))
    chosen: list[str] = []
    selected_rows = 0
    for event, row in event_summary.iloc[::-1].iterrows():
        chosen.append(str(event))
        selected_rows += int(row["_rows"])
        if selected_rows >= target_rows:
            break
    val_events = set(chosen)
    cutoff = event_summary.loc[list(val_events), "_event_time"].min()
    return val_events, cutoff


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(int(label)): int(count)
        for label, count in frame["label"].value_counts().sort_index().items()
    }


def _time_range(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {"min": None, "max": None}
    return {
        "min": frame["_time"].min().isoformat(),
        "max": frame["_time"].max().isoformat(),
    }


def _summarize(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "events": int(frame["_event"].nunique()),
        "labels": _label_counts(frame),
        "time": _time_range(frame),
    }


def prepare_sensor(
    sensor: str,
    spec: SensorSpec,
    output_dir: Path,
    *,
    val_fraction: float,
    cap_per_event_label: int,
    seed: int,
) -> dict:
    source_train = pd.read_csv(spec.train_csv, low_memory=False)
    source_test = pd.read_csv(spec.test_csv, low_memory=False)

    for name, frame in (("train", source_train), ("test", source_test)):
        if "label" not in frame:
            raise ValueError(f"{sensor} {name} CSV has no label column")
        frame["_event"] = spec.event_key(frame)
        frame["_time"] = pd.to_datetime(
            frame[spec.time_column], utc=True, errors="raise"
        )
        if frame["_event"].eq("").any():
            raise ValueError(f"{sensor} {name} contains empty canonical events")

    original_overlap = set(source_train["_event"]) & set(source_test["_event"])
    if original_overlap:
        raise ValueError(
            f"{sensor} source train/test overlap in {len(original_overlap)} events"
        )

    val_events, val_cutoff = _choose_validation_events(source_train, val_fraction)
    val = source_train[source_train["_event"].isin(val_events)].copy()
    train_core = source_train[~source_train["_event"].isin(val_events)].copy()
    if set(train_core["_event"]) & set(val["_event"]):
        raise RuntimeError(f"{sensor} internal train/val event leakage")

    id_column = next(
        (column for column in spec.id_candidates if column in train_core.columns),
        None,
    )
    if id_column is None:
        raise ValueError(
            f"{sensor} has none of the ID columns {spec.id_candidates}"
        )
    train_core["_rank"] = _stable_rank(train_core, id_column, seed)
    capped_train = (
        train_core.sort_values(
            ["_event", "label", "_rank"], kind="mergesort"
        )
        .groupby(["_event", "label"], observed=True, sort=False)
        .head(cap_per_event_label)
        .copy()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    helper_columns = ["_event", "_time", "_rank"]
    outputs = {
        "train_core_full": output_dir / f"{sensor}_train_core_full.csv",
        "train_core_cap": output_dir
        / f"{sensor}_train_core_cap{cap_per_event_label}.csv",
        "val": output_dir / f"{sensor}_val_recent.csv",
    }
    train_core.drop(columns=helper_columns, errors="ignore").to_csv(
        outputs["train_core_full"], index=False
    )
    capped_train.drop(columns=helper_columns, errors="ignore").to_csv(
        outputs["train_core_cap"], index=False
    )
    val.drop(columns=helper_columns, errors="ignore").to_csv(
        outputs["val"], index=False
    )

    audit = {
        "sensor": sensor,
        "source_train": str(spec.train_csv),
        "source_test": str(spec.test_csv),
        "validation_policy": "most recent events until target row fraction",
        "validation_target_fraction": val_fraction,
        "validation_first_time": val_cutoff.isoformat(),
        "cap_per_event_label": cap_per_event_label,
        "seed": seed,
        "source": {
            "train": _summarize(source_train),
            "test": _summarize(source_test),
            "train_test_event_overlap": len(original_overlap),
        },
        "prepared": {
            "train_core_full": _summarize(train_core),
            "train_core_cap": _summarize(capped_train),
            "val": _summarize(val),
            "train_val_event_overlap": int(
                len(set(train_core["_event"]) & set(val["_event"]))
            ),
            "val_test_event_overlap": int(
                len(set(val["_event"]) & set(source_test["_event"]))
            ),
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    audit_path = output_dir / f"{sensor}_split_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/diniuvol/yuyao/methanefuse_research_20260727/manifests"
        ),
    )
    parser.add_argument(
        "--sensors", nargs="+", choices=tuple(SPECS), default=tuple(SPECS)
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--cap-per-event-label", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must lie between 0 and 0.5")
    if args.cap_per_event_label <= 0:
        raise ValueError("--cap-per-event-label must be positive")

    combined = {}
    for sensor in args.sensors:
        combined[sensor] = prepare_sensor(
            sensor,
            SPECS[sensor],
            args.output_dir,
            val_fraction=args.val_fraction,
            cap_per_event_label=args.cap_per_event_label,
            seed=args.seed,
        )
        print(
            f"[{sensor}] train_cap={combined[sensor]['prepared']['train_core_cap']['rows']} "
            f"val={combined[sensor]['prepared']['val']['rows']}",
            flush=True,
        )
    (args.output_dir / "all_sensor_split_audits.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

