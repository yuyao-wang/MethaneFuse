#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sample_complete_plumes(
    csv_path: str,
    maximum_rows: int,
    seed: int,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, low_memory=False)
    required = {"plume_id", "event_group_id", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    label_counts = (
        frame.groupby(["plume_id", "label"]).size().unstack(fill_value=0)
    )
    label_counts = label_counts.reindex(columns=[0, 1], fill_value=0)
    eligible = label_counts[
        (label_counts[0] == 16)
        & (label_counts[1] == 16)
    ].index
    frame = frame[frame["plume_id"].isin(eligible)].copy()
    plume_ids = pd.Series(sorted(frame["plume_id"].astype(str).unique()))
    maximum_plumes = max(1, maximum_rows // 32)
    if len(plume_ids) > maximum_plumes:
        plume_ids = plume_ids.sample(
            n=maximum_plumes,
            random_state=seed,
        )
    subset = frame[frame["plume_id"].astype(str).isin(set(plume_ids))].copy()
    return subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-train-rows", type=int, default=16000)
    parser.add_argument("--max-test-rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    train = sample_complete_plumes(
        args.train_csv,
        args.max_train_rows,
        args.seed,
    )
    test = sample_complete_plumes(
        args.test_csv,
        args.max_test_rows,
        args.seed + 1,
    )
    event_overlap = set(train["event_group_id"].astype(str)) & set(
        test["event_group_id"].astype(str)
    )
    plume_overlap = set(train["plume_id"].astype(str)) & set(
        test["plume_id"].astype(str)
    )
    if event_overlap or plume_overlap:
        raise RuntimeError(
            f"gate subset leakage: events={len(event_overlap)} "
            f"plumes={len(plume_overlap)}"
        )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    train_path = out_root / "train_gate.csv"
    test_path = out_root / "test_gate.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(
        {
            "train_csv": str(train_path),
            "test_csv": str(test_path),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_plumes": int(train["plume_id"].nunique()),
            "test_plumes": int(test["plume_id"].nunique()),
            "train_labels": train["label"].value_counts().sort_index().to_dict(),
            "test_labels": test["label"].value_counts().sort_index().to_dict(),
            "event_overlap": len(event_overlap),
            "plume_overlap": len(plume_overlap),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
