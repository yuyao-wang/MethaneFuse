#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


EMPTY_TOKENS = {"", "nan", "none", "null"}
EVENT_SUFFIX_RE = re.compile(r"-[A-Za-z0-9]+$")
S2_PATH_COLUMNS = ("s2_0_path", "s2_90_path", "s2_360_path")


def clean_paths(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    return values.mask(values.str.casefold().isin(EMPTY_TOKENS), "")


def canonical_events(plume_ids: pd.Series) -> pd.Series:
    events = plume_ids.astype(str).str.strip().str.replace(
        EVENT_SUFFIX_RE, "", regex=True
    )
    if events.eq("").any():
        raise ValueError("canonical event derivation produced empty values")
    return events


def sha256_rows(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[["id", "plume_id", "label"]].itertuples(index=False):
        digest.update(f"{row.id}\0{row.plume_id}\0{row.label}\n".encode("utf-8"))
    return digest.hexdigest()


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    dates = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    return {
        "rows": int(len(frame)),
        "labels": {
            str(key): int(value)
            for key, value in frame["label"].value_counts().sort_index().items()
        },
        "positive_fraction": float(frame["label"].mean()),
        "plumes": int(frame["plume_id"].nunique()),
        "events": int(frame["event_id"].nunique()),
        "date_min": dates.min().isoformat(),
        "date_max": dates.max().isoformat(),
        "row_sha256": sha256_rows(frame),
    }


def overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    return {
        "ids": int(len(set(train["id"]) & set(test["id"]))),
        "plumes": int(len(set(train["plume_id"]) & set(test["plume_id"]))),
        "events": int(len(set(train["event_id"]) & set(test["event_id"]))),
    }


def event_split(
    frame: pd.DataFrame,
    test_fraction: float,
    seed: int,
    trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    event_stats = (
        frame.groupby("event_id", observed=True)
        .agg(rows=("label", "size"), positives=("label", "sum"))
        .reset_index()
    )
    event_ids = event_stats["event_id"].to_numpy()
    rows = event_stats["rows"].to_numpy(dtype=np.int64)
    positives = event_stats["positives"].to_numpy(dtype=np.int64)
    target_rows = len(frame) * test_fraction
    target_positive_fraction = float(frame["label"].mean())
    target_events = max(1, int(round(len(event_ids) * test_fraction)))

    best_indices: np.ndarray | None = None
    best_score: tuple[float, float, float] | None = None
    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        selected = rng.choice(
            len(event_ids), size=target_events, replace=False
        )
        selected_rows = int(rows[selected].sum())
        selected_positives = int(positives[selected].sum())
        selected_positive_fraction = selected_positives / max(selected_rows, 1)
        score = (
            abs(selected_rows - target_rows) / max(target_rows, 1),
            abs(selected_positive_fraction - target_positive_fraction),
            abs(len(selected) / len(event_ids) - test_fraction),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_indices = selected

    if best_indices is None or best_score is None:
        raise RuntimeError("event split search failed")
    test_events = set(event_ids[best_indices].tolist())
    test = frame[frame["event_id"].isin(test_events)].copy()
    train = frame[~frame["event_id"].isin(test_events)].copy()
    diagnostics = {
        "trials": int(trials),
        "target_test_fraction": float(test_fraction),
        "achieved_test_fraction": float(len(test) / len(frame)),
        "target_positive_fraction": target_positive_fraction,
        "test_positive_fraction": float(test["label"].mean()),
        "score_row_fraction": float(best_score[0]),
        "score_positive_fraction": float(best_score[1]),
    }
    return train, test, diagnostics


def write_split(
    output_dir: Path,
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    split_dir = output_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)
    train = train.sample(frac=1.0, random_state=19).reset_index(drop=True)
    test = test.sample(frac=1.0, random_state=23).reset_index(drop=True)
    train_path = split_dir / "train.csv"
    test_path = split_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    report = {
        "name": name,
        "train_csv": str(train_path),
        "test_csv": str(test_path),
        "train": summarize(train),
        "test": summarize(test),
        "overlap": overlap(train, test),
        "diagnostics": diagnostics,
    }
    (split_dir / "split_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def write_gee_export_input(
    source_csv: str,
    plume_ids: set[str],
    output_dir: Path,
) -> dict[str, object]:
    source = pd.read_csv(source_csv, low_memory=False)
    if "plume_id" not in source.columns:
        raise ValueError(f"{source_csv} has no plume_id column")
    source["plume_id"] = source["plume_id"].astype(str).str.strip()
    filtered = source[source["plume_id"].isin(plume_ids)].copy()
    duplicate_count = int(filtered["plume_id"].duplicated().sum())
    matched_ids = set(filtered["plume_id"])
    missing_ids = sorted(plume_ids - matched_ids)
    extra_ids = sorted(matched_ids - plume_ids)
    if duplicate_count or missing_ids or extra_ids:
        raise ValueError(
            "GEE export input does not match the historical cohort: "
            f"duplicates={duplicate_count}, missing={len(missing_ids)}, "
            f"extra={len(extra_ids)}"
        )
    filtered = filtered.sort_values("plume_id").reset_index(drop=True)
    output_path = output_dir / "gee_export_input.csv"
    filtered.to_csv(output_path, index=False)
    return {
        "source_csv": source_csv,
        "output_csv": str(output_path),
        "rows": int(len(filtered)),
        "plumes": int(filtered["plume_id"].nunique()),
        "duplicate_plumes": duplicate_count,
        "missing_plumes": missing_ids,
        "extra_plumes": extra_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-train-csv", required=True)
    parser.add_argument("--legacy-test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--event-search-trials", type=int, default=5000)
    parser.add_argument("--gee-export-source")
    args = parser.parse_args()

    if not 0 < args.test_fraction < 0.5:
        raise ValueError("--test-fraction must be in (0, 0.5)")

    pieces = [
        pd.read_csv(args.legacy_train_csv, low_memory=False),
        pd.read_csv(args.legacy_test_csv, low_memory=False),
    ]
    frame = pd.concat(pieces, ignore_index=True)
    for column in S2_PATH_COLUMNS:
        frame[column] = clean_paths(frame[column])
    frame = frame[frame[list(S2_PATH_COLUMNS)].ne("").all(axis=1)].copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int64)
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("labels must be binary")
    frame["plume_id"] = frame["plume_id"].astype(str).str.strip()
    frame["event_id"] = canonical_events(frame["plume_id"])

    random_train, random_test = train_test_split(
        frame,
        test_size=args.test_fraction,
        random_state=args.seed,
        shuffle=True,
        stratify=frame["label"],
    )
    event_train, event_test, event_diagnostics = event_split(
        frame,
        test_fraction=args.test_fraction,
        seed=args.seed,
        trials=args.event_search_trials,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_dir / "cohort_all.csv"
    frame.sort_values("id").reset_index(drop=True).to_csv(cohort_path, index=False)
    reports = {
        "source": {
            "legacy_train_csv": args.legacy_train_csv,
            "legacy_test_csv": args.legacy_test_csv,
            "cohort_csv": str(cohort_path),
            **summarize(frame),
        },
        "row_random": write_split(
            output_dir,
            "row_random_80_20",
            random_train,
            random_test,
            {"seed": args.seed, "stratified_by": "label"},
        ),
        "event_disjoint": write_split(
            output_dir,
            "event_disjoint_80_20",
            event_train,
            event_test,
            event_diagnostics,
        ),
    }
    if args.gee_export_source:
        reports["gee_export_input"] = write_gee_export_input(
            args.gee_export_source,
            set(frame["plume_id"]),
            output_dir,
        )
    (output_dir / "split_comparison.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
