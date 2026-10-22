#!/usr/bin/env python3
"""Heatmap of hub-independent satellite enrichment, network x pathology."""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUT_DIR = "/data/patrik/NPG/DATA/ANALYSIS"
IN_TSV = os.path.join(OUT_DIR, "satellite_enrichment_per_network.tsv")
OUT_PNG = os.path.join(OUT_DIR, "satellite_enrichment_heatmap.png")

PATHOLOGIES = ["Glioma", "Ischaemic", "Meningioma"]
COL_LABELS = ["Glioma", "Stroke", "Meningioma"]

INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
RED = "#e34948"
NEUTRAL = "#f0efec"


def main():
    df = pd.read_csv(IN_TSV, sep="\t").sort_values("min_enrichment_across_pathologies", ascending=True)
    cols = [f"{p}_enrichment_mean" for p in PATHOLOGIES]
    mat = df[cols].values
    networks = df["network"].tolist()

    fig, ax = plt.subplots(figsize=(6.5, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    norm = TwoSlopeNorm(vmin=max(0.5, mat.min() - 0.02), vcenter=1.0, vmax=mat.max() + 0.02)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=INK)

    ax.set_xticks(range(len(COL_LABELS)))
    ax.set_xticklabels(COL_LABELS, color=INK, fontsize=10)
    ax.set_yticks(range(len(networks)))
    ax.set_yticklabels(networks, color=INK, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Enrichment (1.0 = chance)", color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED)
    cbar.outline.set_visible(False)

    ax.set_title(
        "Hub-independent satellite enrichment by network\n(above 1 = co-occurs with the hub's community more than chance)",
        color=INK, fontsize=12, pad=14,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
