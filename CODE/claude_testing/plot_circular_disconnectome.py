#!/usr/bin/env python3
"""Circular (chord-diagram style) disconnectome plots.

Nodes (ROIs) are arranged on a circle, grouped into contiguous arcs by
Yeo-7 network membership (the standard collapse of each ROI's 17-network
Schaefer label -- VisCent/VisPeri->Visual, SomMotA/B->Somatomotor,
DorsAttnA/B->DorsAttn, SalVentAttnA/B->SalVentAttn, LimbicA/B->Limbic,
ContA/B/C->Cont, DefaultA/B/C + TempPar->Default). Right-hemisphere
networks occupy the right half, left mirrors it on the left; network
groups are ordered top-to-bottom by their members' mean MNI y-coordinate
(anterior first), and ROIs within a group are sub-sorted the same way.
Grouping by network (not raw anatomical position) is deliberate: it
guarantees each network's label spans one truly contiguous arc, since
position-based lobes are not contiguous for networks like Default or
Limbic that mix anterior and posterior regions. Edges are quadratic
Bezier curves through the circle's center (standard chord-diagram
construction), colored and made more opaque/thicker by edge weight
using plasma.
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches

DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
ANALYSIS_DIR = "/data/patrik/NPG/DATA/ANALYSIS"
CENTROIDS_TSV = os.path.join(ANALYSIS_DIR, "roi_centroids.tsv")
OUT_PNG = os.path.join(ANALYSIS_DIR, "subject_circular_disconnectomes.png")

SUBJECTS = [
    ("Glioma", "Glioma_00299_Schaefer2018-100.tsv", "Glioma_00299"),
    ("Ischaemic", "sub-K1093_Schaefer2018-100.tsv", "sub-K1093 (Stroke)"),
    ("Meningioma", "B_Meningioma_00746_Schaefer2018-100.tsv", "B_Meningioma_00746"),
]

INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GAP = np.deg2rad(4)  # gap at top/bottom between hemispheres

# Standard Yeo-7 <- Schaefer-17Networks collapse (CBIG/Yeo2011 correspondence).
YEO7_MAP = {
    "VisCent": "Visual", "VisPeri": "Visual",
    "SomMotA": "Somatomotor", "SomMotB": "Somatomotor",
    "DorsAttnA": "DorsAttn", "DorsAttnB": "DorsAttn",
    "SalVentAttnA": "SalVentAttn", "SalVentAttnB": "SalVentAttn",
    "LimbicA": "Limbic", "LimbicB": "Limbic",
    "ContA": "Cont", "ContB": "Cont", "ContC": "Cont",
    "DefaultA": "Default", "DefaultB": "Default", "DefaultC": "Default",
    "TempPar": "Default",
}


def yeo7_of(network):
    return YEO7_MAP[network]


def layout():
    df = pd.read_csv(CENTROIDS_TSV, sep="\t")
    df["network"] = df["roi"].str.split("_").str[2]
    df["hemi"] = df["roi"].str.split("_").str[1]
    df["yeo7"] = df["network"].map(yeo7_of)

    theta = {}
    for hemi in ["RH", "LH"]:
        sub = df[df.hemi == hemi].copy()
        group_order = sub.groupby("yeo7")["y"].mean().sort_values(ascending=False).index.tolist()
        sub["group_rank"] = sub["yeo7"].map({g: i for i, g in enumerate(group_order)})
        sub = sub.sort_values(["group_rank", "y"], ascending=[True, False]).reset_index(drop=True)
        n = len(sub)
        for rank, roi in enumerate(sub["roi"]):
            if hemi == "RH":
                theta[roi] = GAP + rank / (n - 1) * (np.pi - 2 * GAP)
            else:
                theta[roi] = (2 * np.pi - GAP) - rank / (n - 1) * (np.pi - 2 * GAP)

    pos = {roi: (np.sin(t), np.cos(t)) for roi, t in theta.items()}
    networks = dict(zip(df["roi"], df["network"]))
    yeo7 = dict(zip(df["roi"], df["yeo7"]))
    hemis = dict(zip(df["roi"], df["hemi"]))
    return pos, theta, networks, yeo7, hemis, df["network"].unique().tolist()


PERCENTILE = 90  # None = show all nonzero edges, no thresholding
EDGE_LW_MIN = 0.4
EDGE_LW_MAX = 2.2
EDGE_ALPHA = 0.5
NODE_COLOR = "#4C1D95"


def load_matrix_and_order(path):
    df = pd.read_csv(path, sep="\t", header=0, index_col=0)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df.values.astype(float), df.columns.tolist()


def edge_binary_mask(weights):
    """Top (100-PERCENTILE)% of nonzero edges -> boolean mask, same rule used everywhere else."""
    nonzero = weights[weights > 0]
    if PERCENTILE is None:
        return weights > 0
    threshold = np.percentile(nonzero, PERCENTILE) if nonzero.size else 0
    return weights > threshold


def draw_edges(ax, rois, pos, iu, keep, color_values, log_scale=False):
    cmap = matplotlib.colormaps["plasma"]
    vals = color_values[keep]
    if log_scale:
        vals = np.log10(vals)
    vmin = vals.min() if vals.size else 0.0
    vmax = vals.max() if vals.size else 1.0

    idxs = np.where(keep)[0]
    idxs = idxs[np.argsort(color_values[idxs])]  # weakest first, strongest drawn on top
    for idx in idxs:
        v = np.log10(color_values[idx]) if log_scale else color_values[idx]
        i, j = iu[0][idx], iu[1][idx]
        p1, p2 = pos[rois[i]], pos[rois[j]]
        verts = [p1, (0.0, 0.0), p2]
        path_obj = Path(verts, [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        frac = min((v - vmin) / (vmax - vmin), 1.0) if vmax > vmin else 1.0
        color = cmap(frac)
        lw = EDGE_LW_MIN + (EDGE_LW_MAX - EDGE_LW_MIN) * frac
        patch = mpatches.PathPatch(path_obj, facecolor="none", edgecolor=color, alpha=EDGE_ALPHA, linewidth=lw, zorder=2)
        ax.add_patch(patch)
    return vmin, vmax


LABEL_RADIUS = 1.1
LABEL_CHAR_STEP_MAX = np.deg2rad(3.4)  # cap so a short label in a wide group doesn't stretch absurdly


def draw_curved_label(ax, t_min, t_max, text, fontsize):
    """Place `text` one character at a time along the circle's arc, spanning
    this group's own [t_min, t_max] node range (now guaranteed contiguous by
    the network-grouped layout, so this no longer collides with neighbors)
    -- each glyph individually rotated tangent to the circle, not the whole
    word rotated as one radial block."""
    t_mid = (t_min + t_max) / 2
    flipped = t_mid > np.pi
    chars = text[::-1] if flipped else text
    n = len(chars)
    span = max(t_max - t_min, 1e-6)
    char_step = min(span / max(n - 1, 1), LABEL_CHAR_STEP_MAX)
    start = t_mid - (n - 1) / 2 * char_step
    for i, ch in enumerate(chars):
        t_i = start + i * char_step
        x, y = LABEL_RADIUS * np.sin(t_i), LABEL_RADIUS * np.cos(t_i)
        rot = -np.degrees(t_i)  # tangent to the circle at t_i, not radial
        if flipped:
            rot += 180
        ax.text(x, y, ch, rotation=rot, rotation_mode="anchor", ha="center", va="center",
                 fontsize=fontsize, color=INK, fontweight="bold")


def draw_nodes_labels(ax, rois, pos, theta, yeo7, hemis):
    for roi in rois:
        x, y = pos[roi]
        ax.scatter([x], [y], s=14, color=NODE_COLOR, edgecolor=INK, linewidth=0.2, zorder=3)

    # group labels disabled for now -- see draw_curved_label for the curved-text machinery

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_panel(ax, path, pos, theta, yeo7, hemis):
    mat, rois = load_matrix_and_order(path)
    n = len(rois)
    iu = np.triu_indices(n, k=1)
    weights = mat[iu]
    keep = edge_binary_mask(weights)

    vmin, vmax = draw_edges(ax, rois, pos, iu, keep, weights, log_scale=False)
    draw_nodes_labels(ax, rois, pos, theta, yeo7, hemis)
    return vmin, vmax


def draw_agreement_panel(ax, pos, theta, yeo7, hemis):
    """Edges present in ALL THREE subjects' thresholded binary graphs (binarize each,
    multiply elementwise). Color = log10 of the product of the three RAW weights at
    those agreeing positions (not the binary values themselves)."""
    rois_ref = None
    binaries = []
    raw_weights = []
    for pathology, fname, _ in SUBJECTS:
        mat, rois = load_matrix_and_order(os.path.join(DISCO_DIR, pathology, fname))
        if rois_ref is None:
            rois_ref = rois
            iu = np.triu_indices(len(rois), k=1)
        assert rois == rois_ref, "ROI order mismatch between subjects"
        weights = mat[iu]
        binaries.append(edge_binary_mask(weights).astype(float))
        raw_weights.append(weights)

    agreement = binaries[0] * binaries[1] * binaries[2]  # 1 only where all three survive threshold
    keep = agreement > 0
    product_weight = raw_weights[0] * raw_weights[1] * raw_weights[2]

    vmin, vmax = draw_edges(ax, rois_ref, pos, iu, keep, product_weight, log_scale=True)
    draw_nodes_labels(ax, rois_ref, pos, theta, yeo7, hemis)
    print(f"agreement edges (all 3 survive their own top-{100-PERCENTILE}% threshold): {keep.sum()}")
    return vmin, vmax


def main():
    pos, theta, networks, yeo7, hemis, network_list = layout()

    fig, axes = plt.subplots(1, 4, figsize=(28, 8), facecolor=SURFACE)
    for ax, (pathology, fname, label) in zip(axes[:3], SUBJECTS):
        path = os.path.join(DISCO_DIR, pathology, fname)
        vmin, vmax = draw_panel(ax, path, pos, theta, yeo7, hemis)
        ax.set_title(label, color=INK, fontsize=14, pad=4)

        sm = matplotlib.cm.ScalarMappable(cmap="plasma", norm=matplotlib.colors.Normalize(vmin, vmax))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, orientation="horizontal", location="bottom")
        cbar.set_label("disconnection weight", color=MUTED, fontsize=8)
        cbar.ax.tick_params(colors=MUTED, labelsize=7)
        cbar.outline.set_visible(False)

    ax4 = axes[3]
    vmin4, vmax4 = draw_agreement_panel(ax4, pos, theta, yeo7, hemis)
    ax4.set_title("Agreement (all 3)", color=INK, fontsize=14, pad=4)
    sm4 = matplotlib.cm.ScalarMappable(cmap="plasma", norm=matplotlib.colors.Normalize(vmin4, vmax4))
    cbar4 = fig.colorbar(sm4, ax=ax4, fraction=0.04, pad=0.02, orientation="horizontal", location="bottom")
    cbar4.set_label("log10(w₁·w₂·w₃)", color=MUTED, fontsize=8)
    cbar4.ax.tick_params(colors=MUTED, labelsize=7)
    cbar4.outline.set_visible(False)

    edge_desc = "all nonzero edges" if PERCENTILE is None else f"top {100 - PERCENTILE}% of nonzero edges"
    fig.suptitle(
        f"Circular disconnectome view — {edge_desc} per subject, individual color ranges\n"
        "right hemisphere on the right (frontal top → visual bottom, clockwise), left hemisphere mirrored. "
        "4th panel: edges surviving the threshold in all three (binarized, multiplied), colored by log product weight.",
        color=INK, fontsize=13, y=1.04,
    )
    fig.savefig(OUT_PNG, dpi=220, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
