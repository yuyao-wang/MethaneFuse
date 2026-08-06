#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_unique_plumes(csv_path: Path, split_name: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, low_memory=False)
    required = {"plume_id", "event_group_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    plume_events = frame[["plume_id", "event_group_id"]].drop_duplicates()
    duplicate_plumes = plume_events["plume_id"].duplicated(keep=False)
    if duplicate_plumes.any():
        examples = plume_events.loc[duplicate_plumes].head(10).to_dict("records")
        raise ValueError(f"{split_name} maps plume IDs to multiple events: {examples}")
    plume_events["target_split"] = split_name
    return plume_events


def main(args: argparse.Namespace) -> None:
    source_frames = [
        pd.read_csv(Path(args.local_root) / "train.csv", low_memory=False),
        pd.read_csv(Path(args.local_root) / "test.csv", low_memory=False),
    ]
    source = pd.concat(source_frames, ignore_index=True)
    source["plume_id"] = source["plume_id"].astype(str)
    source["event_group_id"] = source["event_group_id"].astype(str)

    train_plumes = load_unique_plumes(Path(args.target_train_512_csv), "train")
    test_plumes = load_unique_plumes(Path(args.target_test_512_csv), "test")
    target = pd.concat([train_plumes, test_plumes], ignore_index=True)
    target["plume_id"] = target["plume_id"].astype(str)
    target["event_group_id"] = target["event_group_id"].astype(str)

    duplicate_targets = target["plume_id"].duplicated(keep=False)
    if duplicate_targets.any():
        examples = target.loc[duplicate_targets].head(10).to_dict("records")
        raise ValueError(f"plume IDs occur in both target splits: {examples}")

    source_plume_events = source[["plume_id", "event_group_id"]].drop_duplicates()
    event_mismatch = source_plume_events.merge(
        target[["plume_id", "event_group_id"]],
        on="plume_id",
        suffixes=("_source", "_target"),
    )
    event_mismatch = event_mismatch[
        event_mismatch["event_group_id_source"] != event_mismatch["event_group_id_target"]
    ]
    if len(event_mismatch):
        raise ValueError(
            "event IDs changed between local crops and target split: "
            f"{event_mismatch.head(10).to_dict('records')}"
        )

    selected = source.merge(
        target[["plume_id", "target_split"]],
        on="plume_id",
        how="inner",
        validate="many_to_one",
    )
    selected["split"] = selected["target_split"]
    selected = selected.drop(columns=["target_split"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        split_name: output_dir / f"{split_name}.csv"
        for split_name in ("train", "test")
    }
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name, output_path in output_paths.items():
        split_frame = selected[selected["split"].eq(split_name)].copy()
        split_frame = split_frame.sort_values(
            ["event_time", "event_group_id", "plume_id", "sample_id"],
            kind="stable",
        )
        split_frame.to_csv(output_path, index=False)
        split_frames[split_name] = split_frame

    train_events = set(split_frames["train"]["event_group_id"].astype(str))
    test_events = set(split_frames["test"]["event_group_id"].astype(str))
    event_overlap = sorted(train_events & test_events)
    if event_overlap:
        raise ValueError(f"event leakage detected: {event_overlap[:20]}")

    source_plumes = set(source["plume_id"])
    report = {
        "local_root": str(Path(args.local_root)),
        "target_train_512_csv": str(Path(args.target_train_512_csv)),
        "target_test_512_csv": str(Path(args.target_test_512_csv)),
        "source_rows": int(len(source)),
        "source_plumes": int(len(source_plumes)),
        "selected_rows": int(len(selected)),
        "selected_plumes": int(selected["plume_id"].nunique()),
        "target_plumes": int(len(target)),
        "missing_target_plumes": int(len(set(target["plume_id"]) - source_plumes)),
        "missing_target_plume_examples": sorted(set(target["plume_id"]) - source_plumes)[:20],
        "event_group_overlap": len(event_overlap),
        "splits": {},
    }
    for split_name, split_frame in split_frames.items():
        report["splits"][split_name] = {
            "csv": str(output_paths[split_name]),
            "rows": int(len(split_frame)),
            "plumes": int(split_frame["plume_id"].nunique()),
            "events": int(split_frame["event_group_id"].nunique()),
            "labels": {
                str(label): int(count)
                for label, count in split_frame["label"].value_counts().sort_index().items()
            },
        }

    report_path = output_dir / "repartition_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-root",
        default="/diniuvol/yuyao/s2_6time_point_center_corrected_32",
    )
    parser.add_argument(
        "--target-train-512-csv",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_historical_point_center_v11/s2_v11_train_512.csv"
        ),
    )
    parser.add_argument(
        "--target-test-512-csv",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_historical_point_center_v11/s2_v11_test_512.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_historical_point_center_v11/pre_harmonize_local_split"
        ),
    )
    main(parser.parse_args())
