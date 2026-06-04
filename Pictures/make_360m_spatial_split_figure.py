#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


TRAIN_CSV = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_360m/manifest_time_train.csv"
TEST_CSV = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_360m/manifest_time_test.csv"

OUT_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_360m_cluster_split")
TRAIN_OUT = "manifest_time_train_360m_macroregion_by_plume.csv"
TEST_OUT = "manifest_time_test_360m_macroregion_by_plume.csv"
FIG_PNG = Path("/home/yuyao/panopticon/Pictures/360m_macroregion_split_map.png")
FIG_PDF = Path("/home/yuyao/panopticon/Pictures/360m_macroregion_split_map.pdf")

TILE_EDGE_M = 360.0
BLIND_GAP_M = 20.0
SMALL_EPS_FACTOR = 0.70
MACRO_EPS_M = 275_000.0
TEST_RATIO = 0.20
RANDOM_SEED = 42

# Keep one high-density basin from becoming the whole test set.
MAX_REGION_FRACTION_OF_TARGET = 0.28
MIN_REGION_SIZE = 20
MIN_TEST_REGIONS = 8
MAX_TEST_REGIONS = 12
LOWER_TARGET_FRAC = 0.95
UPPER_TARGET_FRAC = 1.08
SEARCH_ITERS = 30_000

EARTH_RADIUS_M = 6371008.8
LAT_CANDIDATES = ("latitude", "lat", "plume_latitude", "center_lat", "centroid_lat")
LON_CANDIDATES = ("longitude", "lon", "plume_longitude", "center_lon", "centroid_lon")
PLUME_ID_CANDIDATES = ("plume_id", "plumeid", "plume_name", "source_id", "id")


def pick_first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def normalize_latlon_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    lat_col = pick_first_existing(df.columns, LAT_CANDIDATES)
    lon_col = pick_first_existing(df.columns, LON_CANDIDATES)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not find latitude/longitude columns in {list(df.columns)}")

    out = df.copy()
    out["_lat"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["_lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=["_lat", "_lon"]).reset_index(drop=True)
    return out, lat_col, lon_col


def make_sample_id(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    plume_col = pick_first_existing(df.columns, PLUME_ID_CANDIDATES)
    out = df.copy()
    if plume_col is None:
        out["_sample_id"] = "row_" + out.index.astype(str)
        return out, "_sample_id"

    raw = out[plume_col].astype("string")
    missing = raw.isna() | raw.str.strip().eq("")
    out["_sample_id"] = raw.fillna("").str.strip()
    out.loc[missing, "_sample_id"] = "row_" + out.index[missing].astype(str)
    return out, "_sample_id"


def dbscan_haversine(lat_lon_deg: np.ndarray, eps_m: float) -> np.ndarray:
    return DBSCAN(
        eps=eps_m / EARTH_RADIUS_M,
        min_samples=1,
        metric="haversine",
    ).fit_predict(np.radians(lat_lon_deg))


def build_sample_table(df: pd.DataFrame, sample_col: str) -> pd.DataFrame:
    sample_df = (
        df.groupby(sample_col, as_index=False)
        .agg(_lat=("_lat", "mean"), _lon=("_lon", "mean"), label=("label", "max"))
        .rename(columns={sample_col: "sample_id"})
    )
    return sample_df


def build_macro_regions(sample_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    small_eps_m = (2.0 * TILE_EDGE_M + BLIND_GAP_M) * SMALL_EPS_FACTOR
    sample_df = sample_df.copy()
    sample_df["cluster_id"] = dbscan_haversine(sample_df[["_lat", "_lon"]].to_numpy(), small_eps_m)

    cluster_df = (
        sample_df.groupby("cluster_id")
        .agg(
            center_lat=("_lat", "mean"),
            center_lon=("_lon", "mean"),
            n_samples=("sample_id", "size"),
            pos=("label", "sum"),
        )
        .reset_index()
    )
    cluster_df["macro_region_id"] = dbscan_haversine(
        cluster_df[["center_lat", "center_lon"]].to_numpy(), MACRO_EPS_M
    )

    cluster_to_macro = dict(zip(cluster_df["cluster_id"], cluster_df["macro_region_id"]))
    sample_df["macro_region_id"] = sample_df["cluster_id"].map(cluster_to_macro)
    return sample_df, cluster_df


def select_test_macro_regions(sample_df: pd.DataFrame, region_df: pd.DataFrame) -> set[int]:
    target_n = len(sample_df) * TEST_RATIO
    max_region_n = target_n * MAX_REGION_FRACTION_OF_TARGET
    global_pos_rate = float(sample_df["label"].mean())

    region_stats = (
        region_df.groupby("macro_region_id")
        .agg(
            n=("n_samples", "sum"),
            pos=("pos", "sum"),
            center_lat=("center_lat", "mean"),
            center_lon=("center_lon", "mean"),
            n_clusters=("cluster_id", "size"),
        )
        .reset_index()
    )

    candidates = region_stats[
        (region_stats["n"] >= MIN_REGION_SIZE) & (region_stats["n"] <= max_region_n)
    ].copy()
    if len(candidates) < MIN_TEST_REGIONS:
        candidates = region_stats[region_stats["n"] <= max_region_n].copy()
    if len(candidates) < MIN_TEST_REGIONS:
        candidates = region_stats.copy()

    candidates = candidates.reset_index(drop=True)
    weights = np.sqrt(candidates["n"].to_numpy(dtype=float))
    weights = weights / weights.sum()
    rng = np.random.default_rng(RANDOM_SEED)

    best: tuple[float, list[int]] | None = None
    for _ in range(SEARCH_ITERS):
        order = rng.choice(len(candidates), size=len(candidates), replace=False, p=weights)
        selected: list[int] = []
        n = 0.0
        pos = 0.0

        for idx in order:
            row = candidates.iloc[int(idx)]
            if n + row.n > target_n * UPPER_TARGET_FRAC:
                continue
            selected.append(int(idx))
            n += float(row.n)
            pos += float(row.pos)
            if n >= target_n * LOWER_TARGET_FRAC and len(selected) >= MIN_TEST_REGIONS:
                break

        if not (target_n * LOWER_TARGET_FRAC <= n <= target_n * UPPER_TARGET_FRAC):
            continue
        if not (MIN_TEST_REGIONS <= len(selected) <= MAX_TEST_REGIONS):
            continue

        chosen = candidates.iloc[selected]
        pos_rate = pos / n
        lon_bins = len(np.unique(np.floor((chosen["center_lon"].to_numpy() + 180.0) / 40.0)))
        lat_bins = len(np.unique(np.floor((chosen["center_lat"].to_numpy() + 90.0) / 25.0)))
        spread_bonus = min(lon_bins, 6) * 0.012 + min(lat_bins, 4) * 0.006

        cost = (
            abs(n - target_n) / target_n
            + 0.30 * abs(pos_rate - global_pos_rate)
            + 0.004 * max(0, len(selected) - 10)
            - spread_bonus
        )

        if best is None or cost < best[0]:
            best = (float(cost), selected)

    if best is None:
        raise RuntimeError("Could not find a macro-region split satisfying the constraints.")

    chosen = candidates.iloc[best[1]]
    return set(chosen["macro_region_id"].astype(int))


def save_split(df: pd.DataFrame, sample_df: pd.DataFrame, sample_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_to_cluster = dict(zip(sample_df["sample_id"], sample_df["cluster_id"]))
    sample_to_macro = dict(zip(sample_df["sample_id"], sample_df["macro_region_id"]))
    sample_to_split = dict(zip(sample_df["sample_id"], sample_df["split"]))

    df = df.copy()
    df["cluster_id"] = df[sample_col].map(sample_to_cluster)
    df["macro_region_id"] = df[sample_col].map(sample_to_macro)
    df["split"] = df[sample_col].map(sample_to_split)

    train_out = df[df["split"] == "train"].copy()
    test_out = df[df["split"] == "test"].copy()
    drop_cols = ["_lat", "_lon", "_orig_split", "split"]
    train_out = train_out.drop(columns=[c for c in drop_cols if c in train_out.columns])
    test_out = test_out.drop(columns=[c for c in drop_cols if c in test_out.columns])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(OUT_DIR / TRAIN_OUT, index=False)
    test_out.to_csv(OUT_DIR / TEST_OUT, index=False)
    return train_out, test_out


def plot_map(sample_df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    train = sample_df[sample_df["split"] == "train"]
    test = sample_df[sample_df["split"] == "test"]

    region_centers = (
        sample_df.groupby("macro_region_id")
        .agg(lat=("_lat", "mean"), lon=("_lon", "mean"), n=("sample_id", "size"))
        .reset_index()
        .sort_values(["lon", "lat"])
    )
    macro_regions = [int(rid) for rid in region_centers["macro_region_id"]]

    base_palette = [
        "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
        "#00A6A6", "#A65628", "#F781BF", "#7FC97F", "#BEAED4",
        "#FDC086", "#386CB0", "#F0027F", "#BF5B17", "#1B9E77",
        "#D95F02", "#7570B3", "#66A61E", "#E7298A", "#A6761D",
    ]
    region_color = {rid: base_palette[i % len(base_palette)] for i, rid in enumerate(macro_regions)}
    train_colors = [region_color[int(rid)] for rid in train["macro_region_id"]]
    test_colors = [region_color[int(rid)] for rid in test["macro_region_id"]]

    fig, ax = plt.subplots(figsize=(7.1, 3.7), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for rid in macro_regions:
        part = sample_df[sample_df["macro_region_id"] == rid]
        if len(part) < 3:
            continue
        lon = part["_lon"].to_numpy()
        lat = part["_lat"].to_numpy()
        lon_span = float(np.ptp(lon))
        lat_span = float(np.ptp(lat))
        width = max(lon_span * 1.9 + 2.4, np.std(lon) * 9.0, 2.8)
        height = max(lat_span * 1.9 + 1.8, np.std(lat) * 9.0, 2.2)
        ax.add_patch(
            Ellipse(
                (float(np.mean(lon)), float(np.mean(lat))),
                width=width,
                height=height,
                facecolor=region_color[rid],
                edgecolor=region_color[rid],
                linewidth=0.95,
                alpha=0.14,
                zorder=0,
            )
        )

    ax.scatter(
        train["_lon"],
        train["_lat"],
        marker="o",
        s=9.5,
        c=train_colors,
        alpha=0.74,
        linewidths=0.08,
        edgecolors="white",
        label="Train plumes",
        zorder=2,
    )

    ax.scatter(
        test["_lon"],
        test["_lat"],
        marker="^",
        s=42,
        c=test_colors,
        alpha=0.98,
        linewidths=0.55,
        edgecolors="#111827",
        label="Test plumes",
        zorder=4,
    )

    ax.set_xlim(-130, 155)
    ax.set_ylim(-52, 75)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    # ax.set_title("Spatially blocked train/test split for 360 m plume samples", pad=6)
    ax.set_xticks(np.arange(-120, 181, 40))
    ax.set_yticks(np.arange(-40, 81, 20))
    ax.grid(False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#3A3A3A")

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        framealpha=0.94,
        borderpad=0.45,
        handletextpad=0.6,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    ax.text(
        0.985,
        0.965,
        f"Color = macro-region ({sample_df['macro_region_id'].nunique()})\nMarker = split",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.94},
        zorder=10,
    )

    fig_png = Path(FIG_PNG)
    fig_pdf = Path(FIG_PDF)
    if not fig_png.is_absolute():
        fig_png = OUT_DIR / "cluster_viz" / fig_png
    if not fig_pdf.is_absolute():
        fig_pdf = OUT_DIR / "cluster_viz" / fig_pdf

    fig_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_png, bbox_inches="tight", dpi=600)
    fig.savefig(fig_pdf, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    train_df["_orig_split"] = "train"
    test_df["_orig_split"] = "test"
    df = pd.concat([train_df, test_df], ignore_index=True)

    df, _, _ = normalize_latlon_columns(df)
    df, sample_col = make_sample_id(df)
    sample_df = build_sample_table(df, sample_col)
    sample_df, region_df = build_macro_regions(sample_df)

    test_regions = select_test_macro_regions(sample_df, region_df)
    sample_df["split"] = np.where(sample_df["macro_region_id"].isin(test_regions), "test", "train")

    train_out, test_out = save_split(df, sample_df, sample_col)
    plot_map(sample_df)

    test_sample = sample_df[sample_df["split"] == "test"]
    train_sample = sample_df[sample_df["split"] == "train"]
    print(f"sample-level train={len(train_sample)}, test={len(test_sample)}, test_frac={len(test_sample)/len(sample_df):.3f}")
    print(f"row-level train={len(train_out)}, test={len(test_out)}")
    print(f"small DBSCAN clusters={sample_df['cluster_id'].nunique()}")
    print(f"macro regions total={sample_df['macro_region_id'].nunique()}, test={test_sample['macro_region_id'].nunique()}")
    print(f"train csv: {OUT_DIR / TRAIN_OUT}")
    print(f"test csv : {OUT_DIR / TEST_OUT}")
    print(f"figure png: {FIG_PNG}")
    print(f"figure pdf: {FIG_PDF}")
    print("\nSelected test macro-regions:")
    print(
        test_sample.groupby("macro_region_id")
        .agg(n=("sample_id", "size"), pos_rate=("label", "mean"), lat=("_lat", "mean"), lon=("_lon", "mean"))
        .sort_values("lon")
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
