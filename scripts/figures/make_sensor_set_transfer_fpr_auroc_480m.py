#!/usr/bin/env python3

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
    "availability", "count",
    "base_f1", "base_acc", "base_fpr", "base_rec", "base_auroc",
    "mf_f1", "mf_acc", "mf_fpr", "mf_rec", "mf_auroc",
]

df = pd.DataFrame(rows_480, columns=cols)

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
    "S2", "L8/9", "EMIT", "S5P",
    "S2+{L8/9, EMIT}", "S2+S5P", "S2+S5P+{L8/9, EMIT}",
    "L8/9+{EMIT, S5P}", "EMIT+{L8/9, S5P}", "S5P+{L8/9, EMIT}",
]

short_labels = {
    "S2": "S2",
    "L8/9": "L8/9",
    "EMIT": "EMIT",
    "S5P": "S5P",
    "S2+{L8/9, EMIT}": "S2+{L,E}",
    "S2+S5P": "S2+S",
    "S2+S5P+{L8/9, EMIT}": "S2+S+{L,E}",
    "L8/9+{EMIT, S5P}": "L+{E,S}",
    "EMIT+{L8/9, S5P}": "E+{L,S}",
    "S5P+{L8/9, EMIT}": "S+{L,E}",
}



def aggregate_group(df_sub):
    weights = df_sub["count"].to_numpy()
    out = {"count": int(df_sub["count"].sum())}
    for col in ["base_fpr", "mf_fpr", "base_auroc", "mf_auroc"]:
        out[col] = np.average(df_sub[col], weights=weights)
    return out


plot_rows = []
for group in group_order:
    row = aggregate_group(df[df["availability"].isin(group_defs[group])])
    row["category"] = group
    row["label"] = short_labels[group]
    plot_rows.append(row)

plot_df = pd.DataFrame(plot_rows)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "pdf.use14corefonts": True,
    "ps.useafm": True,
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
})

BASE_BAR = "#C8D8EB"
MF_BAR = "#5E84AF"
BASE_EDGE = "#9EB7D3"
MF_EDGE = "#476B93"
SEP_COLOR = "#D7D7D7"
GRID_COLOR = "#CFCFCF"


def add_section_guides(ax):
    ax.axvline(3.5, color=SEP_COLOR, lw=1.0)
    ax.axvline(6.5, color=SEP_COLOR, lw=1.0)

    label_kwargs = {
        "ha": "center",
        "va": "top",
        "fontsize": 11,
        "fontweight": "bold",
        "color": "#444444",
        "transform": ax.get_xaxis_transform(),
        "bbox": {"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.86},
        "zorder": 5,
    }
    ax.text(1.5, 0.985, "Single-sensor", **label_kwargs)
    ax.text(5.0, 0.985, "With S2", **label_kwargs)
    ax.text(8.0, 0.985, "Without S2", **label_kwargs)


def draw_grouped_bars(ax, labels, base_vals, mf_vals, title, ylabel, ylim):
    x = np.arange(len(labels))
    width = 0.34

    bars_base = ax.bar(
        x - width / 2,
        base_vals,
        width=width,
        color=BASE_BAR,
        edgecolor=BASE_EDGE,
        linewidth=0.8,
        label="ViT-Avg",
        zorder=3,
    )
    bars_mf = ax.bar(
        x + width / 2,
        mf_vals,
        width=width,
        color=MF_BAR,
        edgecolor=MF_EDGE,
        linewidth=0.8,
        label="MethaneFuse",
        zorder=3,
    )

    ax.set_title(title, pad=11)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right", rotation_mode="anchor")
    ax.margins(x=0.025)

    ax.grid(axis="y", color=GRID_COLOR, alpha=0.35, lw=0.8)
    ax.set_axisbelow(True)
    add_section_guides(ax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return bars_base, bars_mf


def main():
    labels = plot_df["label"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.8), sharex=False)

    bars_base, bars_mf = draw_grouped_bars(
        ax=axes[0],
        labels=labels,
        base_vals=plot_df["base_fpr"].to_numpy(),
        mf_vals=plot_df["mf_fpr"].to_numpy(),
        title="FPR",
        ylabel="FPR (%)",
        ylim=(0, 88),
    )

    draw_grouped_bars(
        ax=axes[1],
        labels=labels,
        base_vals=plot_df["base_auroc"].to_numpy(),
        mf_vals=plot_df["mf_auroc"].to_numpy(),
        title="AUROC",
        ylabel="AUROC (%)",
        ylim=(58, 108),
    )

    fig.legend(
        handles=[bars_base[0], bars_mf[0]],
        labels=["ViT-Avg", "MethaneFuse"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        handlelength=1.7,
        columnspacing=2.0,
    )

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.22, top=0.84, wspace=0.20)
    fig.savefig("sensor_set_transfer_fpr_auroc_480m.pdf", bbox_inches="tight")
    fig.savefig("sensor_set_transfer_fpr_auroc_480m.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
