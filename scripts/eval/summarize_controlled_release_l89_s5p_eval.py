#!/usr/bin/env python3
"""Summarize Universal 360 m L89/S5P controlled-release predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUSTED_KEYS = {
    "20210803_32p053111_m102p300687",
    "20211019_33p630634_m114p489143",
    "20211103_33p630634_m114p489143",
    "20221029_33p630634_m114p489143",
    "20221108_32p821842_m111p785754",
    "20221115_32p821842_m111p785754",
    "20221118_32p821842_m111p785754",
    "20230627_32p245193_m93p286508",
    "20230719_32p245193_m93p286508",
}
PREDICTION_FILES = {
    "L89": REPO_ROOT / "results/eval/controlled_release_l89_legacy360_all_available_predictions.csv",
    "S5P": REPO_ROOT / "results/eval/controlled_release_s5p_all_available_predictions.csv",
    "L89+S5P": REPO_ROOT
    / "results/eval/controlled_release_l89_s5p_legacy360_all_available_predictions.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=REPO_ROOT
        / "results/eval/controlled_release_l89_s5p_legacy360_summary.json",
    )
    parser.add_argument(
        "--output_event_csv",
        type=Path,
        default=REPO_ROOT
        / "results/eval/controlled_release_l89_s5p_legacy360_event_summary.csv",
    )
    return parser.parse_args()


def key_from_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"^(cr_|s5p_)", "", regex=True)


def scalar_summary(frame: pd.DataFrame) -> dict:
    scores = frame["positive_probability"]
    return {
        "rows": int(len(frame)),
        "predicted_positive": int(frame["prediction"].sum()),
        "prediction_rate": float(frame["prediction"].mean()) if len(frame) else None,
        "mean_positive_probability": float(scores.mean()) if len(frame) else None,
        "median_positive_probability": float(scores.median()) if len(frame) else None,
        "min_positive_probability": float(scores.min()) if len(frame) else None,
        "max_positive_probability": float(scores.max()) if len(frame) else None,
    }


def main() -> None:
    args = parse_args()
    l89_status = pd.read_csv(
        REPO_ROOT / "data/controlled_release_test/l89_gee_download.csv", low_memory=False
    )
    l89_status = l89_status.loc[l89_status["timepoint"].eq("t0")].copy()
    l89_status["event_key"] = key_from_series(l89_status["plume_id"])
    l89_status["event_time_parsed"] = pd.to_datetime(l89_status["event_time"], utc=True)
    l89_status["acquisition_time_parsed"] = pd.to_datetime(
        l89_status["acquisition_time_utc"], utc=True
    )
    l89_status["l89_delta_minutes"] = (
        l89_status["acquisition_time_parsed"] - l89_status["event_time_parsed"]
    ).dt.total_seconds() / 60.0
    l89_by_key = l89_status.set_index("event_key")

    s5p_status = pd.read_csv(
        REPO_ROOT
        / "data/controlled_release_test/model_ready_480m/s5p/s5p_local_cache_selection.csv",
        low_memory=False,
    )
    s5p_status = s5p_status.loc[s5p_status["status"].eq("complete")].copy()
    s5p_status["event_key"] = key_from_series(s5p_status["group_id"])
    s5p_by_key = s5p_status.set_index("event_key")

    overall: dict[str, dict] = {}
    event_rows: list[dict] = []
    for sensor, path in PREDICTION_FILES.items():
        frame = pd.read_csv(path, low_memory=False)
        identity_column = "id" if sensor == "S5P" else "plume_id"
        frame["event_key"] = key_from_series(frame[identity_column])
        trusted = frame.loc[frame["event_key"].isin(TRUSTED_KEYS)]
        overall[sensor] = {
            "all_available": scalar_summary(frame),
            "trusted9_overlap": scalar_summary(trusted),
        }
        for event_key, group in frame.groupby("event_key", sort=True):
            row = {
                "sensor": sensor,
                "event_key": event_key,
                "in_trusted9": event_key in TRUSTED_KEYS,
                **scalar_summary(group),
            }
            if event_key in l89_by_key.index:
                source = l89_by_key.loc[event_key]
                row.update(
                    {
                        "l89_event_time_utc": source["event_time"],
                        "l89_acquisition_time_utc": source["acquisition_time_utc"],
                        "l89_delta_minutes": float(source["l89_delta_minutes"]),
                    }
                )
            if event_key in s5p_by_key.index:
                source = s5p_by_key.loc[event_key]
                row.update(
                    {
                        "s5p_event_time_utc": source["event_time_utc"],
                        "s5p_acquisition_time_utc": source["t0_acquisition_time_utc"],
                        "s5p_abs_delta_minutes": float(
                            source["t0_ground_truth_delta_seconds"] / 60.0
                        ),
                    }
                )
            event_rows.append(row)

    result = {
        "decision_rule": "checkpoint-native two-class argmax (positive iff p(class 1) > 0.5)",
        "label_scope": (
            "All available rows carry source label 1. Prediction rate equals recall only if "
            "the source positive label is accepted for the sensor acquisition time."
        ),
        "metric_limitations": (
            "No negative examples are present, so FPR, specificity, AUROC, AP, balanced "
            "accuracy, and F1 are not identifiable from this evaluation."
        ),
        "overall": overall,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_event_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(event_rows).to_csv(args.output_event_csv, index=False)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(f"Wrote event summary: {args.output_event_csv}")


if __name__ == "__main__":
    main()
