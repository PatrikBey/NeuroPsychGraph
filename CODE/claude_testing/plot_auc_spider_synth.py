#!/usr/bin/env python3
"""Spider/radar plot: pairwise classification AUC, lesion (morphometrics) vs disconnectome
(network) -- SYNTHETIC data. Mirrors plot_auc_spider.py (real data)."""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/data/patrik/NPG/DATA/ANALYSIS"
OUT_PNG = os.path.join(OUT_DIR, "pairwise_auc_spider_synth.png")

AXES = ["Glioma vs Stroke", "Glioma vs Meningioma", "Stroke vs Meningioma"]
LESION = [0.985, 0.834, 0.861]
NETWORK = [0.955, 0.824, 0.788]

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

    ax.plot(angles, lesion, color=LESION_COLOR, linewidth=2, marker="o", markersize=5, label="Lesion mask (morphometrics)")
    ax.fill(angles, lesion, color=LESION_COLOR, alpha=0.15)

    ax.plot(angles, network, color=NETWORK_COLOR, linewidth=2, marker="o", markersize=5, label="Disconnectome (network)")
    ax.fill(angles, network, color=NETWORK_COLOR, alpha=0.15)

    for ang, lv, nv in zip(angles[:-1], LESION, NETWORK):
        ax.annotate(f"{lv:.2f}", (ang, lv), textcoords="offset points", xytext=(-20, 6), color=LESION_COLOR, fontsize=9)
        ax.annotate(f"{nv:.2f}", (ang, nv), textcoords="offset points", xytext=(6, -12), color=NETWORK_COLOR, fontsize=9)

    ax.set_title(
        "Pathology separability in network space (SYNTHETIC)\n"
        "(pairwise classifier AUC; axis starts at 0.5 = chance, not 0)",
        color=INK, fontsize=13, pad=30,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False, labelcolor=INK, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
