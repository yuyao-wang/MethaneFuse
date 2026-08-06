#!/usr/bin/env python3
"""Build L89 full-test CSVs with geographically outlying test events removed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TRAIN = (
    "/mnt/engg-niulab/Yuyao/preprocessed_512/L89/"
    "l89_6time_temporal_16_resized_to_224/L89_temporal_train.csv"
)
DEFAULT_TEST = (
    "/mnt/engg-niulab/Yuyao/preprocessed_512/L89/"
    "l89_6time_temporal_16_resized_to_224/L89_temporal_test.csv"
)
DEFAULT_OUT_DIR = "Upgraded_dataset/l89_6time_geo_outlier_test_filters"
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1 = np.radians(lat1)[:, None]
    lon1 = np.radians(lon1)[:, None]
    lat2 = np.radians(lat2)[None, :]
    lon2 = np.radians(lon2)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def event_key(df: pd.DataFrame) -> pd.Series:
    if "plume_id" in df.columns:
        base = df["plume_id"].astype(str).str.replace(r"-[A-Za-z]+$", "", regex=True)
    else:
        base = df["id"].astype(str).str.replace(r"_\d+$", "", regex=True)
    return base + "|" + df["event_time"].astype(str)


def event_centroids(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_key"] = event_key(work)
    return (
        work.dropna(subset=["latitude", "longitude"])
        .groupby("_key", as_index=False)
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
    )


def nearest_train_distance(test_events: pd.DataFrame, train_events: pd.DataFrame) -> np.ndarray:
    train_lat = train_events["latitude"].to_numpy(float)
    train_lon = train_events["longitude"].to_numpy(float)
    test_lat = test_events["latitude"].to_numpy(float)
    test_lon = test_events["longitude"].to_numpy(float)
    nearest = np.empty(len(test_events), dtype=float)
    for start in range(0, len(test_events), 512):
        end = start + 512
        dist = haversine_km(test_lat[start:end], test_lon[start:end], train_lat, train_lon)
        nearest[start:end] = dist.min(axis=1)
    return nearest


def train_leave_one_out_nn(train_events: pd.DataFrame) -> np.ndarray:
    lat = train_events["latitude"].to_numpy(float)
    lon = train_events["longitude"].to_numpy(float)
    nearest = np.empty(len(train_events), dtype=float)
    for start in range(0, len(train_events), 512):
        end = min(start + 512, len(train_events))
        dist = haversine_km(lat[start:end], lon[start:end], lat, lon)
        rows = np.arange(start, end)
        dist[np.arange(end - start), rows] = np.inf
        nearest[start:end] = dist.min(axis=1)
    return nearest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", default=DEFAULT_TRAIN)
    parser.add_argument("--test_csv", default=DEFAULT_TEST)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    train_events = event_centroids(train)
    test_events = event_centroids(test)
    train_nn = train_leave_one_out_nn(train_events)

    test_events["nearest_train_km"] = nearest_train_distance(test_events, train_events)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_events.to_csv(out_dir / "L89_full_test_event_nearest_train_distance.csv", index=False)

    thresholds = {
        "p95_train_nn_km": float(np.percentile(train_nn, 95)),
        "p99_train_nn_km": float(np.percentile(train_nn, 99)),
        "fixed_500km": 500.0,
        "fixed_1000km": 1000.0,
    }
    test_with_key = test.copy()
    test_with_key["_key"] = event_key(test_with_key)
    dist_by_event = test_events.set_index("_key")["nearest_train_km"]

    audit = {
        "train_events": int(len(train_events)),
        "test_events": int(len(test_events)),
        "test_rows": int(len(test)),
        "train_nn_km": {
            "median": float(np.percentile(train_nn, 50)),
            "p75": float(np.percentile(train_nn, 75)),
            "p90": float(np.percentile(train_nn, 90)),
            "p95": float(np.percentile(train_nn, 95)),
            "p99": float(np.percentile(train_nn, 99)),
            "max": float(np.max(train_nn)),
        },
        "filters": {},
    }

    for name, threshold in thresholds.items():
        keep_events = set(test_events.loc[test_events["nearest_train_km"] <= threshold, "_key"])
        kept = test_with_key[test_with_key["_key"].isin(keep_events)].drop(columns=["_key"])
        removed_events = test_events.loc[test_events["nearest_train_km"] > threshold, "_key"].tolist()
        csv_path = out_dir / f"L89_temporal_test_geo_inlier_{name}.csv"
        kept.to_csv(csv_path, index=False)
        audit["filters"][name] = {
            "threshold_km": threshold,
            "kept_rows": int(len(kept)),
            "removed_rows": int(len(test) - len(kept)),
            "kept_events": int(len(keep_events)),
            "removed_events": int(len(removed_events)),
            "removed_event_examples": removed_events[:10],
            "csv": str(csv_path.resolve()),
            "label_counts_kept": {str(k): int(v) for k, v in kept["label"].value_counts().sort_index().items()},
        }

    (out_dir / "geo_outlier_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
