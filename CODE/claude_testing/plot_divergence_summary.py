#!/usr/bin/env python3
"""Grouped bar chart of classifier two-sample AUC across feature spaces."""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/data/patrik/NPG/DATA/ANALYSIS"
IN_TSV = os.path.join(OUT_DIR, "group_divergence_summary.tsv")
OUT_PNG = os.path.join(OUT_DIR, "group_divergence_auc.png")

FEATURE_SPACE_LABELS = {
    "morphometrics": "Morphometrics",
    "disconnectome_node_weights": "Disconnectome\n(weighted)",
    "disconnectome_binary_node_degrees": "Disconnectome\n(binary)",
}
COMPARISON_COLORS = {
    "Real Stroke vs Real Meningioma": "#2a78d6",
    "Synthetic Stroke vs Synthetic Meningioma": "#eb6834",
    "Real Meningioma vs Synthetic Meningioma": "#1baf7a",
    "Real Stroke vs Synthetic Stroke": "#eda100",
}

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BASELINE = "#c3c2b7"


def main():
    df = pd.read_csv(IN_TSV, sep="\t")
    df["comparison"] = df["group_a"] + " vs " + df["group_b"]
    feature_spaces = list(FEATURE_SPACE_LABELS.keys())
    comparisons = list(COMPARISON_COLORS.keys())

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)

    n_comp = len(comparisons)
    bar_width = 0.8 / n_comp
    x = np.arange(len(feature_spaces))

    for i, comp in enumerate(comparisons):
        sub = df[df["comparison"] == comp].set_index("feature_space")
        means = [sub.loc[fs, "auc_mean"] if fs in sub.index else np.nan for fs in feature_spaces]
        stds = [sub.loc[fs, "auc_std"] if fs in sub.index else 0 for fs in feature_spaces]
        offset = (i - (n_comp - 1) / 2) * bar_width
        ax.bar(
            x + offset, means, width=bar_width * 0.9, color=COMPARISON_COLORS[comp],
            edgecolor=INK, linewidth=0.3, yerr=stds, capsize=2,
            error_kw={"ecolor": INK, "elinewidth": 0.8}, label=comp, zorder=2,
        )

    ax.axhline(0.5, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(len(feature_spaces) - 0.5, 0.505, "chance (0.5)", color=MUTED, fontsize=9, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_SPACE_LABELS[fs] for fs in feature_spaces], color=INK, fontsize=10)
    ax.set_ylabel("Classifier two-sample AUC\n(5-fold CV, random forest)", color=MUTED, fontsize=10)
    ax.set_ylim(0.45, 1.02)
    ax.tick_params(colors=MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_title("How separable are these groups?", color=INK, fontsize=14, pad=14)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5, labelcolor=INK, ncol=1)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
