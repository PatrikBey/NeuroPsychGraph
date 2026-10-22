#!/usr/bin/env python3
"""t-SNE scatter plots of lesion morphometric measures across pathology classes.

Fits a single t-SNE embedding on the combined, standardized feature matrix
for Glioma, Ischaemic, and Meningioma so all plots share one coordinate
system, then renders:
  - all three classes together
  - Ischaemic vs Glioma only
  - Ischaemic vs Meningioma only
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

MORPH_DIR = "/data/patrik/NPG/DATA/MORPHOMETRICS"
FILES = {
    "Glioma": os.path.join(MORPH_DIR, "Glioma_morphometrics.csv"),
    "Ischaemic": os.path.join(MORPH_DIR, "Ischaemic_morphometrics.csv"),
    "Meningioma": os.path.join(MORPH_DIR, "Meningioma_morphometrics.csv"),
}
COLORS = {"Glioma": "#2a78d6", "Ischaemic": "#eb6834", "Meningioma": "#1baf7a"}
FEATURE_COLUMNS = [
    "volume_mm3",
    "equiv_diameter_mm",
    "surface_area_mm2",
    "surface_to_volume_ratio",
    "n_components",
    "laterality_index",
    "largest_component_volume_fraction",
    "sphericity",
    "axis1_mm",
    "axis2_mm",
    "axis3_mm",
    "elongation_ratio",
    "flatness_ratio",
    "solidity",
]

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_combined():
    frames = []
    for pathology, path in FILES.items():
        df = pd.read_csv(path)
        df = df[df["error"].fillna("") == ""]
        df = df.dropna(subset=FEATURE_COLUMNS)
        df["pathology"] = pathology
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelsize=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_xlabel("t-SNE dim 1", color=MUTED, fontsize=10)
    ax.set_ylabel("t-SNE dim 2", color=MUTED, fontsize=10)


def scatter_classes(ax, df, classes):
    for cls in classes:
        sub = df[df["pathology"] == cls]
        ax.scatter(
            sub["tsne_x"],
            sub["tsne_y"],
            s=22,
            c=COLORS[cls],
            edgecolors=INK,
            linewidths=0.3,
            alpha=0.75,
            label=f"{cls} (n={len(sub)})",
            zorder=2,
        )


def save_plot(df, classes, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    style_axes(ax)
    scatter_classes(ax, df, classes)
    ax.set_title(title, color=INK, fontsize=13, pad=12)
    legend = ax.legend(frameon=False, loc="best", fontsize=10, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    df = load_combined()
    print(f"combined rows: {len(df)} ({df['pathology'].value_counts().to_dict()})")

    X = StandardScaler().fit_transform(df[FEATURE_COLUMNS].values)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    embedding = tsne.fit_transform(X)
    df["tsne_x"] = embedding[:, 0]
    df["tsne_y"] = embedding[:, 1]

    save_plot(
        df,
        ["Glioma", "Ischaemic", "Meningioma"],
        "t-SNE of lesion morphometrics — all three classes",
        os.path.join(MORPH_DIR, "tsne_all_classes.png"),
    )
    save_plot(
        df,
        ["Glioma", "Ischaemic"],
        "t-SNE of lesion morphometrics — Ischaemic vs Glioma",
        os.path.join(MORPH_DIR, "tsne_ischaemic_glioma.png"),
    )
    save_plot(
        df,
        ["Meningioma", "Ischaemic"],
        "t-SNE of lesion morphometrics — Ischaemic vs Meningioma",
        os.path.join(MORPH_DIR, "tsne_ischaemic_meningioma.png"),
    )


if __name__ == "__main__":
    main()
