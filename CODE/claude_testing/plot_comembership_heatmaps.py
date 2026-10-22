#!/usr/bin/env python3
"""ROI x ROI co-membership heatmaps, one panel per aetiology, ROIs sorted
by Schaefer 17-network so block (community) structure is visible."""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
ANALYSIS_DIR = "/data/patrik/NPG/DATA/ANALYSIS"
OUT_PNG = os.path.join(ANALYSIS_DIR, "comembership_heatmaps.png")
PATHOLOGIES = ["Glioma", "Ischaemic", "Meningioma"]
LABELS = {"Glioma": "Glioma", "Ischaemic": "Stroke", "Meningioma": "Meningioma"}

INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"


def roi_order():
    df = pd.read_csv(
        os.path.join(DISCO_DIR, "Glioma", "Glioma_00001_Schaefer2018-100.tsv"),
        sep="\t", header=0, index_col=0,
    )
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    rois = sorted(df.columns.tolist())  # matches the order comembership_*.npy was built in
    network = [r.split("_")[2] for r in rois]
    hemi = [r.split("_")[1] for r in rois]
    order = sorted(range(len(rois)), key=lambda i: (network[i], hemi[i], rois[i]))
    sorted_rois = [rois[i] for i in order]
    sorted_networks = [network[i] for i in order]
    return order, sorted_rois, sorted_networks


def network_boundaries(sorted_networks):
    bounds, labels = [], []
    start = 0
    for i in range(1, len(sorted_networks) + 1):
        if i == len(sorted_networks) or sorted_networks[i] != sorted_networks[start]:
            bounds.append((start, i))
            labels.append(sorted_networks[start])
            start = i
    return bounds, labels


def main():
    order, sorted_rois, sorted_networks = roi_order()
    bounds, net_labels = network_boundaries(sorted_networks)
    tick_pos = [(a + b) / 2 - 0.5 for a, b in bounds]

    mats = {}
    for p in PATHOLOGIES:
        m = np.load(os.path.join(ANALYSIS_DIR, f"comembership_{p}.npy"))
        m = m[np.ix_(order, order)]
        mats[p] = m

    fig, axes = plt.subplots(1, 3, figsize=(19, 7), facecolor=SURFACE)
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad(color="#dddad2")

    im = None
    for ax, p in zip(axes, PATHOLOGIES):
        m = np.ma.masked_invalid(mats[p])
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        for a, b in bounds:
            ax.axhline(a - 0.5, color=SURFACE, linewidth=0.4)
            ax.axvline(a - 0.5, color=SURFACE, linewidth=0.4)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(net_labels, rotation=90, fontsize=6.5, color=INK)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(net_labels, fontsize=6.5, color=INK)
        ax.set_title(LABELS[p], color=INK, fontsize=13, pad=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("P(same community | both affected)", color=MUTED, fontsize=10)
    cbar.ax.tick_params(colors=MUTED)
    cbar.outline.set_visible(False)

    fig.suptitle(
        "ROI co-membership: how often are two regions in the same disconnection community?\n"
        "(ROIs sorted by Schaefer 17-network; gray = never both affected in the same subject)",
        color=INK, fontsize=13, y=1.03,
    )
    fig.savefig(OUT_PNG, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
