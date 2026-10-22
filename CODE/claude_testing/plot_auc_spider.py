#!/usr/bin/env python3
"""Spider/radar plot: pairwise classification AUC, lesion (morphometrics) vs disconnectome (network).

Shows the other half of the convergence story: network representations
converge (lower AUC than morphometrics, consistent with the similarity
spider plot) but pathology is still highly separable in network space
(AUC far above 0.5 chance) -- specificity isn't lost, just reduced.
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/data/patrik/NPG/DATA/ANALYSIS"
OUT_PNG = os.path.join(OUT_DIR, "pairwise_auc_spider.png")

AXES = ["Glioma vs Stroke", "Glioma vs Meningioma", "Stroke vs Meningioma"]
LESION = [0.977, 0.933, 0.976]
NETWORK = [0.957, 0.912, 0.889]
CHANCE = 0.5

LESION_COLOR = "#E5446D"
NETWORK_COLOR = "#388697"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
RMIN = 0.5


def main():
    n = len(AXES)
    angles = [i / n * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    lesion = LESION + LESION[:1]
    network = NETWORK + NETWORK[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, color=INK, fontsize=11)

    ax.set_ylim(RMIN, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], color=MUTED, fontsize=8)
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)

    ax.plot(angles, lesion, color=LESION_COLOR, linewidth=2, marker="o", markersize=5)
    ax.fill(angles, lesion, color=LESION_COLOR, alpha=0.15)

    ax.plot(angles, network, color=NETWORK_COLOR, linewidth=2, marker="o", markersize=5)
    ax.fill(angles, network, color=NETWORK_COLOR, alpha=0.15)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
