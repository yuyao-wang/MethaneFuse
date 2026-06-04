#!/usr/bin/env python3
"""
Merge one train CSV + one test CSV, then re-split by geo clusters.

Key points:
1) Merge original train/test first.
2) Hard lock exact same lat/lon into one atomic key.
3) DBSCAN(haversine) clusters on coordinates, and split by cluster to reduce geo leakage.
4) Dynamic epsilon search: prefer smaller epsilon when test-ratio tolerance is satisfied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_M = 6371008.8
LAT_CANDIDATES = ("latitude", "lat", "plume_latitude")
LON_CANDIDATES = ("longitude", "lon", "plume_longitude")
DEFAULT_EPS_FACTORS = (0.70, 0.85, 1.00, 1.15, 1.30)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geo-cluster re-split for one train/test CSV pair")
    p.add_argument("--train_csv", required=True, help="Path to input train CSV")
    p.add_argument("--test_csv", required=True, help="Path to input test CSV")
    p.add_argument("--out_dir", default=".", help="Output directory")
    p.add_argument("--train_out", default="train_geo_resplit.csv", help="Output train CSV filename")
    p.add_argument("--test_out", default="test_geo_resplit.csv", help="Output test CSV filename")
    p.add_argument("--summary_out", default="geo_resplit_summary.json", help="Output summary JSON filename")
    p.add_argument("--lat_col", default=None, help="Optional latitude column name")
    p.add_argument("--lon_col", default=None, help="Optional longitude column name")
    p.add_argument("--label_col", default="label", help="Label column for split quality scoring")
    p.add_argument("--test_ratio", type=float, default=0.20, help="Target test ratio")
    p.add_argument("--tile_edge_m", type=float, default=21000.0, help="Tile edge length in meters")
    p.add_argument("--blind_gap_m", type=float, default=20.0, help="Safety gap in meters")
    p.add_argument(
        "--eps_factors",
        type=float,
        nargs="+",
        default=list(DEFAULT_EPS_FACTORS),
        help="Candidate multipliers for base epsilon",
    )
    p.add_argument("--ratio_tol", type=float, default=0.03, help="Tolerance for test ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def pick_first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def normalize_latlon_columns(
    df: pd.DataFrame, lat_col: Optional[str], lon_col: Optional[str]
) -> Tuple[pd.DataFrame, str, str]:
    lat = lat_col or pick_first_existing(df.columns, LAT_CANDIDATES)
    lon = lon_col or pick_first_existing(df.columns, LON_CANDIDATES)
    if lat is None or lon is None:
        raise ValueError(
            "Cannot find lat/lon columns. "
            f"Columns={list(df.columns)}, lat candidates={LAT_CANDIDATES}, lon candidates={LON_CANDIDATES}"
        )

    out = df.copy()
    out["_lat"] = pd.to_numeric(out[lat], errors="coerce")
    out["_lon"] = pd.to_numeric(out[lon], errors="coerce")
    out = out.dropna(subset=["_lat", "_lon"])
    return out, lat, lon


def dbscan_cluster_haversine(lat_lon_deg: np.ndarray, eps_m: float) -> np.ndarray:
    radians = np.radians(lat_lon_deg)
    eps_rad = eps_m / EARTH_RADIUS_M
    return DBSCAN(eps=eps_rad, min_samples=1, metric="haversine").fit_predict(radians)


def greedy_split_by_cluster(
    df: pd.DataFrame, cluster_col: str, test_ratio: float, seed: int, label_col: str
) -> pd.Series:
    rng = np.random.default_rng(seed)

    if label_col in df.columns:
        pos_agg = (label_col, "sum")
    else:
        pos_agg = (cluster_col, "size")

    cluster_stats = df.groupby(cluster_col).agg(n=(cluster_col, "size"), pos=pos_agg).reset_index()
    cluster_stats = cluster_stats.sample(frac=1.0, random_state=seed).sort_values("n", ascending=False)

    total_n = int(cluster_stats["n"].sum())
    total_pos = float(cluster_stats["pos"].sum())
    target_test_n = total_n * test_ratio
    target_test_pos = total_pos * test_ratio

    test_clusters = set()
    test_n = 0
    test_pos = 0.0
    for _, row in cluster_stats.iterrows():
        cid = row[cluster_col]
        n = int(row["n"])
        pos = float(row["pos"])

        keep_cost = abs(test_n - target_test_n) + 1.2 * abs(test_pos - target_test_pos)
        take_cost = abs((test_n + n) - target_test_n) + 1.2 * abs((test_pos + pos) - target_test_pos)
        jitter = rng.uniform(-1e-6, 1e-6)
        if take_cost + jitter < keep_cost:
            test_clusters.add(cid)
            test_n += n
            test_pos += pos

    if len(test_clusters) == 0 and len(cluster_stats) > 0:
        test_clusters.add(cluster_stats.iloc[0][cluster_col])
    if len(test_clusters) == len(cluster_stats) and len(cluster_stats) > 1:
        test_clusters.remove(cluster_stats.iloc[-1][cluster_col])

    return df[cluster_col].isin(test_clusters)


def evaluate_split(df: pd.DataFrame, is_test: pd.Series, test_ratio: float, label_col: str) -> float:
    total = len(df)
    if total == 0:
        return float("inf")

    test_frac = float(is_test.mean())
    ratio_err = abs(test_frac - test_ratio)

    if label_col not in df.columns:
        return ratio_err

    global_pos = float(df[label_col].mean())
    test_pos = float(df.loc[is_test, label_col].mean()) if is_test.any() else global_pos
    label_err = abs(test_pos - global_pos)
    return ratio_err + 0.5 * label_err


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv, low_memory=False)
    test_df = pd.read_csv(args.test_csv, low_memory=False)
    train_df["_orig_split"] = "train"
    test_df["_orig_split"] = "test"
    df = pd.concat([train_df, test_df], ignore_index=True, sort=False)

    df, lat_col, lon_col = normalize_latlon_columns(df, lat_col=args.lat_col, lon_col=args.lon_col)

    df["_coord_key"] = df[lat_col].astype(str).str.strip() + "|" + df[lon_col].astype(str).str.strip()
    coord_df = df.groupby("_coord_key", as_index=False).agg({"_lat": "first", "_lon": "first"})

    base_eps = 2.0 * float(args.tile_edge_m) + float(args.blind_gap_m)
    candidates = []
    for factor in sorted(args.eps_factors):
        eps_m = base_eps * float(factor)
        coord_labels = dbscan_cluster_haversine(coord_df[["_lat", "_lon"]].to_numpy(), eps_m=eps_m)
        key2cluster = dict(zip(coord_df["_coord_key"], coord_labels))
        df_tmp = df.copy()
        df_tmp["_cluster_id"] = df_tmp["_coord_key"].map(key2cluster)
        is_test = greedy_split_by_cluster(
            df_tmp, "_cluster_id", test_ratio=args.test_ratio, seed=args.seed, label_col=args.label_col
        )
        quality = evaluate_split(df_tmp, is_test, test_ratio=args.test_ratio, label_col=args.label_col)
        candidates.append((eps_m, quality, is_test, coord_labels))

    feasible = []
    for eps_m, quality, is_test, coord_labels in candidates:
        if abs(float(is_test.mean()) - args.test_ratio) <= args.ratio_tol:
            feasible.append((eps_m, quality, is_test, coord_labels))

    if feasible:
        feasible.sort(key=lambda x: (x[0], x[1]))
        best_eps_m, best_quality, best_is_test, best_coord_labels = feasible[0]
    else:
        candidates.sort(key=lambda x: (x[1], x[0]))
        best_eps_m, best_quality, best_is_test, best_coord_labels = candidates[0]

    key2cluster = dict(zip(coord_df["_coord_key"], best_coord_labels))
    df["cluster_id"] = df["_coord_key"].map(key2cluster).astype(int)
    df["split_geo"] = np.where(best_is_test, "test", "train")

    out_train = df[df["split_geo"] == "train"].copy()
    out_test = df[df["split_geo"] == "test"].copy()

    drop_cols = ["_lat", "_lon", "_coord_key", "_orig_split", "split_geo"]
    out_train = out_train.drop(columns=[c for c in drop_cols if c in out_train.columns])
    out_test = out_test.drop(columns=[c for c in drop_cols if c in out_test.columns])

    train_out_path = out_dir / args.train_out
    test_out_path = out_dir / args.test_out
    out_train.to_csv(train_out_path, index=False)
    out_test.to_csv(test_out_path, index=False)

    summary = {
        "input_train_csv": str(args.train_csv),
        "input_test_csv": str(args.test_csv),
        "output_train_csv": str(train_out_path),
        "output_test_csv": str(test_out_path),
        "total_rows_after_latlon_filter": int(len(df)),
        "train_rows": int(len(out_train)),
        "test_rows": int(len(out_test)),
        "test_ratio_actual": float(len(out_test) / len(df)) if len(df) > 0 else None,
        "test_ratio_target": float(args.test_ratio),
        "selected_eps_m": float(best_eps_m),
        "selected_quality": float(best_quality),
        "n_clusters": int(df["cluster_id"].nunique()),
        "lat_col": lat_col,
        "lon_col": lon_col,
        "label_col_used": args.label_col if args.label_col in df.columns else None,
    }

    summary_out_path = out_dir / args.summary_out
    with summary_out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 90)
    print("Geo cluster re-split done")
    print(f"Input train: {args.train_csv}")
    print(f"Input test : {args.test_csv}")
    print(f"Lat/Lon    : {lat_col}, {lon_col}")
    print(f"Selected eps(m): {best_eps_m:.2f}")
    print(f"Clusters      : {summary['n_clusters']}")
    print(f"Output train  : {train_out_path} ({len(out_train)} rows)")
    print(f"Output test   : {test_out_path} ({len(out_test)} rows)")
    print(f"Summary JSON  : {summary_out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
