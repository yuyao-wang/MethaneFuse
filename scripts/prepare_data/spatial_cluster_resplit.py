#!/usr/bin/env python3
"""
Build sensor-wise spatial clusters and resplit train/test.

Core rules:
1) Merge each sensor's original train/test files first.
2) Hard deduplication: rows with identical lat/lon are collapsed into atomic groups so they stay in the same split.
3) Cluster coordinates with DBSCAN (haversine), then split by cluster to avoid spatial leakage.
4) Dynamic epsilon: use 2 * tile_edge_m as the baseline, try smaller radii first, and prefer the smallest epsilon that satisfies test-ratio tolerance.
5) Plot each sensor's cluster distribution and final train/test split.

Dependencies:
    pip install pandas numpy scikit-learn matplotlib

Run:
    python spatial_cluster_resplit.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_M = 6371008.8

# -----------------------------
# 1) Edit input paths here
# -----------------------------
SENSOR_CONFIG: Dict[str, Dict[str, object]] = {
    "l89": {
        "train_csv": "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/data_dir_l89_L2SR/l89_temporal_16_resized_to_224_CRSfixed/train_2025_balanced.csv",
        "test_csv": "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/data_dir_l89_L2SR/l89_temporal_16_resized_to_224_CRSfixed/test_filtered_2025.csv",
        "tile_edge_m": 480.0,  # 16 * 30m
    },
    "s2": {
        "train_csv": "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16/train.csv",
        "test_csv": "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16/test.csv",
        "tile_edge_m": 160.0,  # 16 * 10m
    },
    "s5p": {
        "train_csv": "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s5p_patches_3x3_to_224_offl_triplet/train.csv",
        "test_csv": "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s5p_patches_3x3_to_224_offl_triplet/test.csv",
        "tile_edge_m": 21000.0,  # approximately 3 * 7 km
    },
    "wv3": {
        "train_csv": "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/emit_wv3_temporal_-90_-180_16_to_224/train_balanced.csv",
        "test_csv": "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/emit_wv3_temporal_-90_-180_16_to_224/test_balanced.csv",
        "tile_edge_m": 960.0,  # for 16 * 60 m
    },
}

# Output directory
OUT_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training")
# Combined output filenames
MIXED_TRAIN_OUT = "train_4_geo.csv"
MIXED_TEST_OUT = "test_4_geo.csv"

# Split parameters
TEST_RATIO = 0.20
RANDOM_SEED = 42
BLIND_GAP_M = 20.0  # visual blind gap; >0 is more conservative
EPS_FACTORS = (0.70, 0.85, 1.00, 1.15, 1.30)  # dynamic radius candidates; smaller radii are preferred
RATIO_TOL = 0.03


LAT_CANDIDATES = ("latitude", "lat", "plume_latitude")
LON_CANDIDATES = ("longitude", "lon", "plume_longitude")
@dataclass
class SplitResult:
    sensor: str
    eps_m: float
    n_clusters: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def pick_first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def ensure_sensor_column(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    if "sensor" not in df.columns:
        df = df.copy()
        df["sensor"] = sensor
    return df


def normalize_latlon_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    lat_col = pick_first_existing(df.columns, LAT_CANDIDATES)
    lon_col = pick_first_existing(df.columns, LON_CANDIDATES)
    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Could not find latitude/longitude columns. Existing columns: {list(df.columns)}；supported candidates lat={LAT_CANDIDATES}, lon={LON_CANDIDATES}"
        )

    out = df.copy()
    out["_lat"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["_lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=["_lat", "_lon"])
    return out, lat_col, lon_col


def dbscan_cluster_haversine(lat_lon_deg: np.ndarray, eps_m: float) -> np.ndarray:
    # lat_lon_deg shape: (N,2), columns [lat, lon]
    radians = np.radians(lat_lon_deg)
    eps_rad = eps_m / EARTH_RADIUS_M
    # min_samples=1: no noise points; every point is assigned to a cluster
    labels = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine").fit_predict(radians)
    return labels


def greedy_split_by_cluster(df: pd.DataFrame, cluster_col: str, test_ratio: float, seed: int) -> pd.Series:
    """Return a boolean series where True means test split."""
    rng = np.random.default_rng(seed)

    # Cluster-level statistics
    cluster_stats = (
        df.groupby(cluster_col)
        .agg(n=(cluster_col, "size"), pos=("label", "sum"))
        .reset_index()
    )

    # Shuffle, then assign larger clusters first to avoid being blocked by large groups at the end
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

        # Assignment cost: sample-ratio error plus positive-ratio error
        keep_cost = abs(test_n - target_test_n) + 1.2 * abs(test_pos - target_test_pos)
        take_cost = abs((test_n + n) - target_test_n) + 1.2 * abs((test_pos + pos) - target_test_pos)

        # Add a small random jitter to avoid deterministic ties
        jitter = rng.uniform(-1e-6, 1e-6)
        if take_cost + jitter < keep_cost:
            test_clusters.add(cid)
            test_n += n
            test_pos += pos

    # Guard against assigning everything to train or test
    if len(test_clusters) == 0 and len(cluster_stats) > 0:
        test_clusters.add(cluster_stats.iloc[0][cluster_col])
    if len(test_clusters) == len(cluster_stats) and len(cluster_stats) > 1:
        test_clusters.remove(cluster_stats.iloc[-1][cluster_col])

    return df[cluster_col].isin(test_clusters)


def evaluate_split(df: pd.DataFrame, is_test: pd.Series, test_ratio: float) -> float:
    total = len(df)
    if total == 0:
        return float("inf")

    global_pos = float(df["label"].mean()) if "label" in df.columns else 0.0
    test_df = df[is_test]
    test_frac = len(test_df) / total
    test_pos = float(test_df["label"].mean()) if len(test_df) > 0 else global_pos

    ratio_err = abs(test_frac - test_ratio)
    label_err = abs(test_pos - global_pos)
    return ratio_err + 0.5 * label_err


def split_one_sensor(
    sensor: str,
    train_csv: str,
    test_csv: str,
    tile_edge_m: float,
    test_ratio: float,
    blind_gap_m: float,
    eps_factors: Sequence[float],
    ratio_tol: float,
    seed: int,
) -> SplitResult:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df = ensure_sensor_column(train_df, sensor)
    test_df = ensure_sensor_column(test_df, sensor)

    train_df["_orig_split"] = "train"
    test_df["_orig_split"] = "test"
    df = pd.concat([train_df, test_df], ignore_index=True)

    df, lat_col, lon_col = normalize_latlon_columns(df)

    if "label" not in df.columns:
        raise ValueError(f"{sensor}: missing label column; cannot evaluate stratification quality.")

    # -----------------------------
    # Hard deduplication: merge points with identical coordinates first
    # -----------------------------
    # Use string keys for identical coordinates so temporal/duplicate samples stay together
    df["_coord_key"] = df[lat_col].astype(str).str.strip() + "|" + df[lon_col].astype(str).str.strip()
    coord_df = df.groupby("_coord_key", as_index=False).agg({"_lat": "first", "_lon": "first"})

    base_eps = 2.0 * float(tile_edge_m) + float(blind_gap_m)

    best = None
    candidate_records: List[Tuple[float, float, pd.Series, np.ndarray]] = []

    for factor in sorted(eps_factors):
        eps_m = base_eps * float(factor)

        coord_labels = dbscan_cluster_haversine(coord_df[["_lat", "_lon"]].to_numpy(), eps_m=eps_m)
        coord_df_tmp = coord_df.copy()
        coord_df_tmp["_cluster_id"] = coord_labels

        # Map assignments back to rows
        key2cluster = dict(zip(coord_df_tmp["_coord_key"], coord_df_tmp["_cluster_id"]))
        df_tmp = df.copy()
        df_tmp["_cluster_id"] = df_tmp["_coord_key"].map(key2cluster)

        is_test = greedy_split_by_cluster(df_tmp, "_cluster_id", test_ratio=test_ratio, seed=seed)
        quality = evaluate_split(df_tmp, is_test, test_ratio=test_ratio)
        candidate_records.append((eps_m, quality, is_test, coord_labels))

    # Prefer the smallest eps that satisfies test-ratio error; otherwise choose the best-quality smaller eps
    feasible = []
    for eps_m, quality, is_test, coord_labels in candidate_records:
        frac = float(is_test.mean())
        if abs(frac - test_ratio) <= ratio_tol:
            feasible.append((eps_m, quality, is_test, coord_labels))

    if feasible:
        feasible.sort(key=lambda x: (x[0], x[1]))  # Prefer smaller eps
        best = feasible[0]
    else:
        candidate_records.sort(key=lambda x: (x[1], x[0]))
        best = candidate_records[0]

    best_eps_m, _, best_is_test, best_coord_labels = best

    # Build final cluster_id values for the selected eps
    final_coord_df = coord_df.copy()
    final_coord_df["_cluster_id"] = best_coord_labels
    key2cluster = dict(zip(final_coord_df["_coord_key"], final_coord_df["_cluster_id"]))
    df["cluster_id"] = df["_coord_key"].map(key2cluster)
    df["split"] = np.where(best_is_test, "test", "train")

    out_train = df[df["split"] == "train"].copy()
    out_test = df[df["split"] == "test"].copy()

    # Drop temporary columns
    drop_cols = ["_lat", "_lon", "_coord_key", "_orig_split", "split"]
    out_train = out_train.drop(columns=[c for c in drop_cols if c in out_train.columns])
    out_test = out_test.drop(columns=[c for c in drop_cols if c in out_test.columns])

    n_clusters = int(final_coord_df["_cluster_id"].nunique())
    return SplitResult(sensor=sensor, eps_m=best_eps_m, n_clusters=n_clusters, train_df=out_train, test_df=out_test)


def plot_sensor_clusters(
    sensor: str,
    full_df: pd.DataFrame,
    out_png: Path,
    lat_col: str = "_lat",
    lon_col: str = "_lon",
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Left plot: cluster distribution
    axes[0].scatter(
        full_df[lon_col],
        full_df[lat_col],
        c=full_df["cluster_id"].astype(int),
        s=6,
        cmap="tab20",
        alpha=0.75,
        linewidths=0,
    )
    axes[0].set_title(f"{sensor}: Spatial Clusters")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    # Right plot: train/test distribution
    split_color = np.where(full_df["split"] == "test", "#D62728", "#1F77B4")
    axes[1].scatter(
        full_df[lon_col],
        full_df[lat_col],
        c=split_color,
        s=6,
        alpha=0.75,
        linewidths=0,
    )
    axes[1].set_title(f"{sensor}: Train/Test by Cluster")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    train_n = int((full_df["split"] == "train").sum())
    test_n = int((full_df["split"] == "test").sum())
    axes[1].text(
        0.01,
        0.99,
        f"train={train_n}, test={test_n}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    viz_dir = OUT_DIR / "cluster_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    all_train: List[pd.DataFrame] = []
    all_test: List[pd.DataFrame] = []

    print("=" * 90)
    print("Spatial cluster split started")
    print(f"Output dir: {OUT_DIR}")
    print("=" * 90)

    for sensor, cfg in SENSOR_CONFIG.items():
        train_csv = str(cfg["train_csv"])
        test_csv = str(cfg["test_csv"])
        tile_edge_m = float(cfg["tile_edge_m"])

        print(f"\n[{sensor}] loading")
        print(f"  train: {train_csv}")
        print(f"  test : {test_csv}")

        split_res = split_one_sensor(
            sensor=sensor,
            train_csv=train_csv,
            test_csv=test_csv,
            tile_edge_m=tile_edge_m,
            test_ratio=TEST_RATIO,
            blind_gap_m=BLIND_GAP_M,
            eps_factors=EPS_FACTORS,
            ratio_tol=RATIO_TOL,
            seed=RANDOM_SEED,
        )

        sensor_train_out = OUT_DIR / f"train_{sensor}_geo.csv"
        sensor_test_out = OUT_DIR / f"test_{sensor}_geo.csv"
        split_res.train_df.to_csv(sensor_train_out, index=False)
        split_res.test_df.to_csv(sensor_test_out, index=False)

        print(
            f"  eps={split_res.eps_m:.1f}m, clusters={split_res.n_clusters}, "
            f"train={len(split_res.train_df)}, test={len(split_res.test_df)}"
        )
        print(f"  -> {sensor_train_out}")
        print(f"  -> {sensor_test_out}")

        all_train.append(split_res.train_df)
        all_test.append(split_res.test_df)

        # Rebuild full_df for visualization with intermediate columns
        # Recompute here to avoid returning a large temporary DataFrame.
        merged = pd.concat([
            ensure_sensor_column(pd.read_csv(train_csv), sensor),
            ensure_sensor_column(pd.read_csv(test_csv), sensor),
        ], ignore_index=True)
        merged, lat_col, lon_col = normalize_latlon_columns(merged)
        merged["_coord_key"] = merged[lat_col].astype(str).str.strip() + "|" + merged[lon_col].astype(str).str.strip()
        coord_df = merged.groupby("_coord_key", as_index=False).agg({"_lat": "first", "_lon": "first"})
        coord_labels = dbscan_cluster_haversine(coord_df[["_lat", "_lon"]].to_numpy(), split_res.eps_m)
        key2cluster = dict(zip(coord_df["_coord_key"], coord_labels))
        merged["cluster_id"] = merged["_coord_key"].map(key2cluster)

        # Fill split labels using final train/test index keys
        # Prefer common ID columns; fall back to path_t0 if needed
        key_col = None
        for cand in ("id", "sample_id", "path_t0"):
            if cand in merged.columns and cand in split_res.train_df.columns and cand in split_res.test_df.columns:
                key_col = cand
                break
        if key_col is None:
            warnings.warn(f"{sensor}: could not find a stable key to fill split labels; visualization will show a cluster-level approximation.")
            mask_test = greedy_split_by_cluster(merged, "cluster_id", test_ratio=TEST_RATIO, seed=RANDOM_SEED)
            merged["split"] = np.where(mask_test, "test", "train")
        else:
            train_keys = set(split_res.train_df[key_col].astype(str))
            merged["split"] = np.where(merged[key_col].astype(str).isin(train_keys), "train", "test")

        plot_path = viz_dir / f"{sensor}_cluster_split.png"
        plot_sensor_clusters(sensor=sensor, full_df=merged, out_png=plot_path)
        print(f"  -> {plot_path}")

    mixed_train = pd.concat(all_train, ignore_index=True)
    mixed_test = pd.concat(all_test, ignore_index=True)

    mixed_train_out = OUT_DIR / MIXED_TRAIN_OUT
    mixed_test_out = OUT_DIR / MIXED_TEST_OUT
    mixed_train.to_csv(mixed_train_out, index=False)
    mixed_test.to_csv(mixed_test_out, index=False)

    print("\n" + "=" * 90)
    print(f"Mixed train -> {mixed_train_out} ({len(mixed_train)} rows)")
    print(f"Mixed test  -> {mixed_test_out} ({len(mixed_test)} rows)")
    print("Done.")


if __name__ == "__main__":
    run()
