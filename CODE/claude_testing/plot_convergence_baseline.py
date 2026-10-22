#!/usr/bin/env python3
"""Bar chart of PERMANOVA R2 (variance explained by pathology) across feature spaces."""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/data/patrik/NPG/DATA/ANALYSIS"
IN_TSV = os.path.join(OUT_DIR, "convergence_baseline.tsv")
OUT_PNG = os.path.join(OUT_DIR, "convergence_baseline_r2.png")

LABELS = {
    "morphometrics": "Morphometrics\n(lesion masks)",
    "disconnectome_node_weights": "Disconnectome\n(weighted)",
    "disconnectome_binary_node_degrees": "Disconnectome\n(binary)",
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def main():
    df = pd.read_csv(IN_TSV, sep="\t").set_index("feature_space").loc[list(LABELS.keys())]

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)

    x = range(len(LABELS))
    ax.bar(x, df["permanova_r2"], color=COLORS, edgecolor=INK, linewidth=0.3, width=0.6, zorder=2)
    for i, (r2, p) in enumerate(zip(df["permanova_r2"], df["permanova_p"])):
        ax.text(i, r2 + 0.01, f"R²={r2:.3f}\np={p:.3f}", ha="center", va="bottom", fontsize=9, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[k] for k in LABELS], color=INK, fontsize=10)
    ax.set_ylabel("PERMANOVA R² (variance explained by pathology)", color=MUTED, fontsize=10)
    ax.set_ylim(0, 0.4)
    ax.tick_params(colors=MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_title("How much does pathology explain the data?", color=INK, fontsize=14, pad=14)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
