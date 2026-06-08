#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


rows_480 = [
    ("S2", 11735, 84.87, 84.55, 17.03, 86.11, 92.63, 88.42, 88.47, 10.51, 87.46, 97.22),
    ("L8/9", 7315, 76.81, 78.20, 16.65, 72.94, 86.77, 78.53, 80.04, 13.76, 73.72, 93.97),
    ("EMIT", 5737, 73.25, 65.17, 55.87, 78.98, 69.71, 83.91, 80.23, 27.58, 85.36, 87.60),
    ("S5P", 631, 62.27, 64.66, 21.68, 53.33, 65.69, 63.14, 60.22, 42.31, 62.32, 78.86),
    ("S2+L8/9", 1334, 83.82, 83.36, 16.61, 83.33, 93.10, 91.78, 92.13, 0.16, 84.93, 99.85),
    ("S2+EMIT", 677, 82.72, 82.72, 17.89, 83.33, 89.00, 83.64, 84.34, 12.02, 80.65, 84.36),
    ("S2+S5P", 565, 77.55, 78.58, 15.16, 72.57, 89.92, 91.76, 92.21, 0.36, 85.07, 99.84),
    ("L8/9+EMIT", 724, 79.73, 79.14, 18.82, 77.34, 82.66, 72.21, 71.82, 25.00, 69.01, 96.02),
    ("L8/9+S5P", 535, 74.95, 74.77, 14.29, 66.45, 87.75, 96.26, 95.89, 0.43, 93.09, 94.95),
    ("EMIT+S5P", 1538, 75.60, 72.43, 20.10, 68.01, 78.54, 90.21, 87.84, 14.51, 89.23, 95.52),
    ("S2+L8/9+EMIT", 60, 83.58, 81.67, 25.00, 87.50, 88.95, 95.52, 95.00, 10.71, 100.00, 96.88),
    ("S2+L8/9+S5P", 350, 85.92, 82.86, 15.08, 81.70, 89.28, 98.46, 98.00, 5.56, 100.00, 98.86),
    ("S2+EMIT+S5P", 224, 82.79, 81.25, 15.62, 78.91, 88.27, 79.01, 69.64, 70.83, 100.00, 98.86),
    ("L8/9+EMIT+S5P", 162, 78.31, 77.78, 17.57, 73.86, 81.56, 80.37, 73.46, 58.11, 100.00, 96.71),
]

cols = [
    "availability",
    "count",
    "base_f1",
    "base_acc",
    "base_fpr",
    "base_rec",
    "base_auroc",
    "mf_f1",
    "mf_acc",
    "mf_fpr",
    "mf_rec",
    "mf_auroc",
]

group_defs = {
    "S2": ["S2"],
    "L8/9": ["L8/9"],
    "EMIT": ["EMIT"],
    "S5P": ["S5P"],
    "S2+{L8/9, EMIT}": ["S2+L8/9", "S2+EMIT", "S2+L8/9+EMIT"],
    "S2+S5P": ["S2+S5P"],
    "S2+S5P+{L8/9, EMIT}": ["S2+L8/9+S5P", "S2+EMIT+S5P"],
    "L8/9+{EMIT, S5P}": ["L8/9+EMIT", "L8/9+S5P", "L8/9+EMIT+S5P"],
    "EMIT+{L8/9, S5P}": ["L8/9+EMIT", "EMIT+S5P", "L8/9+EMIT+S5P"],
    "S5P+{L8/9, EMIT}": ["L8/9+S5P", "EMIT+S5P", "L8/9+EMIT+S5P"],
}

group_order = [
    "S2",
    "L8/9",
    "EMIT",
    "S5P",
    "S2+{L8/9, EMIT}",
    "S2+S5P",
    "S2+S5P+{L8/9, EMIT}",
    "L8/9+{EMIT, S5P}",
    "EMIT+{L8/9, S5P}",
    "S5P+{L8/9, EMIT}",
]

short_labels = {
    "S2": "S2",
    "L8/9": "L8/9",
    "EMIT": "EMIT",
    "S5P": "S5P",
    "S2+{L8/9, EMIT}": "S2 + {L, E}",
    "S2+S5P": "S2 + S",
    "S2+S5P+{L8/9, EMIT}": "S2 + S + {L, E}",
    "L8/9+{EMIT, S5P}": "L + {E, S}",
    "EMIT+{L8/9, S5P}": "E + {L, S}",
    "S5P+{L8/9, EMIT}": "S + {L, E}",
}

BASE_COLOR = "#1f77b4"
MF_COLOR = "#d95f02"
LINE_COLOR = "#9aa3ad"
SEP_COLOR = "#c8d0d8"
TEXT_COLOR = "#222222"


def aggregate_group(df_sub):
    weights = df_sub["count"].to_numpy()
    return {
        "count": int(df_sub["count"].sum()),
        "base_f1": np.average(df_sub["base_f1"], weights=weights),
        "mf_f1": np.average(df_sub["mf_f1"], weights=weights),
    }


def build_plot_df():
    df = pd.DataFrame(rows_480, columns=cols)
    plot_rows = []
    for group in group_order:
        row = aggregate_group(df[df["availability"].isin(group_defs[group])])
        row["category"] = group
        row["label"] = short_labels[group]
        plot_rows.append(row)
    return pd.DataFrame(plot_rows)


def add_section_guides(ax):
    ax.axhline(3.5, color=SEP_COLOR, lw=1.8)
    ax.axhline(6.5, color=SEP_COLOR, lw=1.8)

    label_kwargs = {
        "ha": "left",
        "va": "top",
        "fontsize": 23,
        "fontweight": "bold",
        "color": "#333333",
        "bbox": {"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.9},
        "zorder": 4,
    }
    ax.text(53.6, -0.18, "Single-sensor", **label_kwargs)
    ax.text(53.6, 3.63, "With S2", **label_kwargs)
    ax.text(53.6, 6.63, "Without S2", **label_kwargs)


def annotate_pair(ax, x1, x2, y):
    box = {"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.92}
    fs = 18
    if abs(x2 - x1) < 2.4:
        ax.annotate(
            f"{x1:.1f}",
            (x1, y),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fs,
            fontweight="bold",
            color=BASE_COLOR,
            bbox=box,
        )
        ax.annotate(
            f"{x2:.1f}",
            (x2, y),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=fs,
            fontweight="bold",
            color=MF_COLOR,
            bbox=box,
        )
    else:
        ax.annotate(
            f"{x1:.1f}",
            (x1, y),
            xytext=(-11, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=fs,
            fontweight="bold",
            color=BASE_COLOR,
            bbox=box,
        )
        ax.annotate(
            f"{x2:.1f}",
            (x2, y),
            xytext=(11, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=fs,
            fontweight="bold",
            color=MF_COLOR,
            bbox=box,
        )


def main():
    out_dir = Path(__file__).resolve().parent
    plot_df = build_plot_df()

    labels = plot_df["label"].tolist()
    base_vals = plot_df["base_f1"].to_numpy()
    mf_vals = plot_df["mf_f1"].to_numpy()
    y = np.arange(len(labels))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 22,
            "axes.labelsize": 24,
            "xtick.labelsize": 21,
            "ytick.labelsize": 23,
            "legend.fontsize": 22,
            "axes.linewidth": 1.2,
        }
    )

    fig, ax = plt.subplots(figsize=(14.2, 8.3))

    for yi, base, mf in zip(y, base_vals, mf_vals):
        ax.plot([base, mf], [yi, yi], color=LINE_COLOR, lw=3.0, zorder=1)

    ax.scatter(base_vals, y, s=120, color=BASE_COLOR, label="ViT-Avg", zorder=3)
    ax.scatter(mf_vals, y, s=120, color=MF_COLOR, label="MethaneFuse", zorder=3)

    for yi, base, mf in zip(y, base_vals, mf_vals):
        annotate_pair(ax, base, mf, yi)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(52.5, 102.5)
    ax.set_xlabel("F1 (%)", labelpad=8)
    ax.tick_params(axis="both", length=5, width=1.1, colors=TEXT_COLOR)
    ax.grid(axis="x", color="#d9dee3", alpha=0.75, lw=1.0)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_section_guides(ax)

    ax.legend(
        loc="lower left",
        frameon=False,
        ncol=2,
        handletextpad=0.45,
        columnspacing=1.2,
        borderaxespad=0.2,
    )

    fig.subplots_adjust(left=0.19, right=0.985, top=0.975, bottom=0.13)
    fig.savefig(out_dir / "sensor_set_transfer_f1_480m_readable.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "sensor_set_transfer_f1_480m_readable.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
