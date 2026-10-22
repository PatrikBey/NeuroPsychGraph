#!/usr/bin/env python3
"""Side-by-side disconnectome heatmaps for three specific subjects (one per
pathology) whose lesions share almost the same MNI centroid -- same ROI
ordering/style as the co-membership figures for visual consistency."""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
OUT_PNG = "/data/patrik/NPG/DATA/ANALYSIS/subject_disconnectome_heatmaps.png"

SUBJECTS = [
    ("Glioma", "Glioma_00299_Schaefer2018-100.tsv", "Glioma_00299"),
    ("Ischaemic", "sub-K1093_Schaefer2018-100.tsv", "sub-K1093 (Stroke)"),
    ("Meningioma", "B_Meningioma_00746_Schaefer2018-100.tsv", "B_Meningioma_00746"),
]

INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"


def roi_order():
    df = pd.read_csv(
        os.path.join(DISCO_DIR, "Glioma", "Glioma_00001_Schaefer2018-100.tsv"),
        sep="\t", header=0, index_col=0,
    )
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    rois = sorted(df.columns.tolist())
    network = [r.split("_")[2] for r in rois]
    hemi = [r.split("_")[1] for r in rois]
    order = sorted(range(len(rois)), key=lambda i: (network[i], hemi[i], rois[i]))
    return order, [network[i] for i in order]


def network_boundaries(sorted_networks):
    bounds, labels = [], []
    start = 0
    for i in range(1, len(sorted_networks) + 1):
        if i == len(sorted_networks) or sorted_networks[i] != sorted_networks[start]:
            bounds.append((start, i))
            labels.append(sorted_networks[start])
            start = i
    return bounds, labels


def load_matrix(path, order):
    df = pd.read_csv(path, sep="\t", header=0, index_col=0)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    rois = df.columns.tolist()
    name_to_idx = {r: i for i, r in enumerate(rois)}
    all_rois_sorted_ref = sorted(rois)
    mat = df.values.astype(float)
    reorder = [name_to_idx[all_rois_sorted_ref[i]] for i in order]
    return mat[np.ix_(reorder, reorder)]


def main():
    order, sorted_networks = roi_order()
    bounds, net_labels = network_boundaries(sorted_networks)
    tick_pos = [(a + b) / 2 - 0.5 for a, b in bounds]

    mats = []
    vmax = 0
    for pathology, fname, label in SUBJECTS:
        m = load_matrix(os.path.join(DISCO_DIR, pathology, fname), order)
        mats.append(m)
        vmax = max(vmax, np.percentile(m[m > 0], 99) if (m > 0).any() else 0)

    fig, axes = plt.subplots(1, 3, figsize=(19, 7), facecolor=SURFACE)
    cmap = matplotlib.colormaps["magma"].copy()

    im = None
    for ax, m, (pathology, fname, label) in zip(axes, mats, SUBJECTS):
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
        for a, b in bounds:
            ax.axhline(a - 0.5, color=SURFACE, linewidth=0.4)
            ax.axvline(a - 0.5, color=SURFACE, linewidth=0.4)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(net_labels, rotation=90, fontsize=6.5, color=INK)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(net_labels, fontsize=6.5, color=INK)
        ax.set_title(label, color=INK, fontsize=13, pad=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("disconnection weight (clipped at 99th pct)", color=MUTED, fontsize=10)
    cbar.ax.tick_params(colors=MUTED)
    cbar.outline.set_visible(False)

    fig.suptitle(
        "Disconnectomes of three lesions sharing ~the same MNI centroid ([-31,-40,41], left parietal)\n"
        "(ROIs sorted by Schaefer 17-network, shared color scale)",
        color=INK, fontsize=13, y=1.03,
    )
    fig.savefig(OUT_PNG, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
