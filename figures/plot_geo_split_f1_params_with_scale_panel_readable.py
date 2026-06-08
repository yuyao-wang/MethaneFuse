import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


baseline_data = [
    # Method, Label, Group, Params(M), Params_plot(M), F1
    ["Majority", "Maj.", "Per-sensor ViT fusion", 346.40, 338.0, 67.85],
    ["Logical OR", "OR", "Per-sensor ViT fusion", 346.40, 346.4, 66.90],
    ["Average", "Avg.", "Per-sensor ViT fusion", 346.40, 354.8, 69.30],
    ["SatMAE-FT", "SatMAE", "FM fine-tuning", 342.52, 322.0, 59.40],
    ["Panopticon-FT", "Panopticon", "FM fine-tuning", 395.92, 395.92, 67.20],
    ["AnySat-FT", "AnySat", "FM fine-tuning", 503.64, 503.64, 51.80],
    ["MethaneFuse", "MethaneFuse", "Ours", 98.98, 98.98, 70.24],
]

df = pd.DataFrame(
    baseline_data,
    columns=["Method", "Label", "Group", "Params_M", "Params_plot", "F1"],
)

scale_data = [
    [120, 66.70, 81.20],
    [360, 70.40, 83.40],
    [480, 70.20, 83.10],
    [960, 68.30, 82.00],
]

df_scale = pd.DataFrame(scale_data, columns=["Scale", "F1", "AUROC"])

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9.8,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 8.8,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.8, 3.2),
    gridspec_kw={"width_ratios": [1.85, 0.8], "wspace": 0.38},
)

ax = axes[0]
ax2 = axes[1]

style_map = {
    "Per-sensor ViT fusion": {
        "marker": "o",
        "facecolor": "white",
        "edgecolor": "#1F77B4",
        "size": 78,
        "label": "Per-sensor ViT fusion",
        "zorder": 4,
    },
    "FM fine-tuning": {
        "marker": "s",
        "facecolor": "#B0B0B0",
        "edgecolor": "black",
        "size": 78,
        "label": "FM fine-tuning",
        "zorder": 4,
    },
    "Ours": {
        "marker": "*",
        "facecolor": "#D62728",
        "edgecolor": "black",
        "size": 220,
        "label": "MethaneFuse",
        "zorder": 6,
    },
}

for group, sub in df.groupby("Group"):
    st = style_map[group]
    ax.scatter(
        sub["Params_plot"],
        sub["F1"],
        s=st["size"],
        marker=st["marker"],
        facecolors=st["facecolor"],
        edgecolors=st["edgecolor"],
        linewidths=0.95,
        label=st["label"],
        zorder=st["zorder"],
    )

# Offsets are in screen points, so labels stay separated after resizing/exporting.
label_offsets = {
    "MethaneFuse": (14, 9, "left", "bottom"),
    "Avg.": (7, 13, "left", "bottom"),
    "Maj.": (-13, 3, "right", "center"),
    "OR": (10, -11, "left", "top"),
    "SatMAE": (10, 8, "left", "bottom"),
    "Panopticon": (11, -5, "left", "top"),
    "AnySat": (-10, 8, "right", "bottom"),
}

for _, row in df.iterrows():
    dx, dy, ha, va = label_offsets[row["Label"]]
    ax.annotate(
        row["Label"],
        xy=(row["Params_plot"], row["F1"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=10.8,
        ha=ha,
        va=va,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82),
        zorder=8,
    )

best_baseline = df[df["Method"] != "MethaneFuse"]["F1"].max()
ax.axhline(
    best_baseline,
    linestyle="--",
    linewidth=0.85,
    color="black",
    alpha=0.55,
    zorder=1,
)

ax.annotate(
    "Best baseline",
    xy=(455, best_baseline),
    xytext=(0, 7),
    textcoords="offset points",
    fontsize=10.6,
    ha="left",
    va="bottom",
    alpha=0.78,
    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82),
)

ax.set_xlabel("Inference-time parameters (M) ↓")
ax.set_ylabel("F1 (%) ↑")
ax.set_xlim(70, 535)
ax.set_ylim(50, 72)
ax.set_xticks([100, 200, 300, 400, 500])
ax.set_yticks([50, 55, 60, 65, 70])
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor="#1F77B4",
        markeredgewidth=0.95,
        markersize=7.2,
        label="Per-sensor ViT fusion",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        markerfacecolor="#B0B0B0",
        markeredgecolor="black",
        markeredgewidth=0.95,
        markersize=7.2,
        label="FM fine-tuning",
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="none",
        markerfacecolor="#D62728",
        markeredgecolor="black",
        markeredgewidth=0.95,
        markersize=11.4,
        label="MethaneFuse",
    ),
]

ax.legend(
    handles=handles,
    loc="lower left",
    fontsize=10.2,
    frameon=True,
    borderpad=0.48,
    handletextpad=0.62,
    labelspacing=0.52,
)

xpos = range(len(df_scale))

ax2.plot(
    xpos,
    df_scale["F1"],
    marker="o",
    linewidth=1.45,
    markersize=5.0,
    label="F1",
)

ax2.plot(
    xpos,
    df_scale["AUROC"],
    marker="s",
    linewidth=1.45,
    markersize=4.8,
    label="AUROC",
)

idx_480 = df_scale.index[df_scale["Scale"] == 480][0]
row_480 = df_scale.loc[idx_480]

ax2.scatter(
    [idx_480],
    [row_480["F1"]],
    marker="*",
    s=130,
    facecolors="#D62728",
    edgecolors="black",
    linewidths=0.8,
    zorder=5,
)

ax2.annotate(
    "480 m",
    xy=(idx_480, row_480["F1"]),
    xytext=(18, -18),
    textcoords="offset points",
    fontsize=9.0,
    ha="left",
    va="top",
    arrowprops=dict(arrowstyle="-", lw=0.65, color="0.25", shrinkA=3, shrinkB=5),
    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.86),
    zorder=8,
)

ax2.set_xlabel("Query scale (m)")
ax2.set_ylabel("Score (%) ↑")
ax2.set_xlim(-0.35, len(df_scale) - 0.55)
ax2.set_ylim(64, 85)
ax2.set_xticks(list(xpos))
ax2.set_xticklabels(df_scale["Scale"].astype(str))
ax2.set_yticks([65, 70, 75, 80, 85])
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2.legend(
    loc="lower right",
    frameon=True,
    borderpad=0.42,
    handletextpad=0.55,
    labelspacing=0.45,
)

fig.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.965, wspace=0.38)
plt.savefig("figures/geo_split_f1_params_with_scale_panel_readable.pdf", bbox_inches="tight")
plt.savefig(
    "figures/geo_split_f1_params_with_scale_panel_readable.png",
    bbox_inches="tight",
    dpi=600,
)
plt.show()
