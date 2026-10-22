#!/usr/bin/env python3
"""Spider/radar plot: cross-pathology similarity (Spearman rho), lesion-mask-only vs disconnectome."""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/data/patrik/NPG/DATA/ANALYSIS"
OUT_PNG = os.path.join(OUT_DIR, "cross_pathology_similarity_spider.png")

AXES = ["Glioma vs Stroke", "Glioma vs Meningioma", "Stroke vs Meningioma"]
LESION = [0.678, 0.747, 0.588]
NETWORK = [0.885, 0.905, 0.820]

LESION_COLOR = "#E5446D"
NETWORK_COLOR = "#388697"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


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

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color=MUTED, fontsize=8)
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
