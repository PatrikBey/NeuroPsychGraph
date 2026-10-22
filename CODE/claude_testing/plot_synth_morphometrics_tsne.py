#!/usr/bin/env python3
"""t-SNE of lesion morphometrics: synthetic-only, and real+synthetic combined.

Plot 1: the two synthetic classes (Ischaemic/Stroke, Meningioma) only,
fit and colored on their own.

Plot 2: real and synthetic Meningioma and Stroke (Ischaemic) combined
in a single t-SNE fit -- Glioma is excluded, and Meningioma/Stroke are
kept as separate pathologies (not merged into a tumour/stroke split).
Four explicit colors per user instruction: real Meningioma = crimson,
synthetic Meningioma = gold, real Stroke = purple, synthetic Stroke =
teal.
"""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

MORPH_DIR = "/data/patrik/NPG/DATA/MORPHOMETRICS"
REAL_FILES = {
    "Ischaemic": os.path.join(MORPH_DIR, "Ischaemic_morphometrics.csv"),
    "Meningioma": os.path.join(MORPH_DIR, "Meningioma_morphometrics.csv"),
}
SYNTH_FILES = {
    "Ischaemic": os.path.join(MORPH_DIR, "SYNTH_Ischaemic_morphometrics.csv"),
    "Meningioma": os.path.join(MORPH_DIR, "SYNTH_Meningioma_morphometrics.csv"),
}
DISPLAY_NAME = {"Ischaemic": "Stroke", "Meningioma": "Meningioma"}

PATHOLOGY_COLORS = {"Ischaemic": "#008080", "Meningioma": "#FFD700"}  # synthetic-only plot: teal / gold
REAL_COLORS = {"Ischaemic": "#800080", "Meningioma": "#DC143C"}  # real-only plot: purple / crimson

COMBO_COLORS = {
    ("Real", "Meningioma"): "#DC143C",  # crimson
    ("Synthetic", "Meningioma"): "#FFD700",  # gold
    ("Real", "Ischaemic"): "#800080",  # purple
    ("Synthetic", "Ischaemic"): "#008080",  # teal
}

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


def load(files, origin):
    frames = []
    for pathology, path in files.items():
        df = pd.read_csv(path)
        df = df[df["error"].fillna("") == ""]
        df = df.dropna(subset=FEATURE_COLUMNS)
        df["pathology"] = pathology
        df["origin"] = origin
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_xlabel("t-SNE dim 1", color=MUTED, fontsize=10)
    ax.set_ylabel("t-SNE dim 2", color=MUTED, fontsize=10)


def fit_tsne(df):
    X = StandardScaler().fit_transform(df[FEATURE_COLUMNS].values)
    embedding = TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(X)
    df = df.copy()
    df["tsne_x"], df["tsne_y"] = embedding[:, 0], embedding[:, 1]
    return df


def plot_single_origin(df, colors, title, out_name):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    style_axes(ax)
    for pathology in ["Ischaemic", "Meningioma"]:
        sub = df[df["pathology"] == pathology]
        ax.scatter(
            sub["tsne_x"],
            sub["tsne_y"],
            s=22,
            c=colors[pathology],
            edgecolors=INK,
            linewidths=0.3,
            alpha=0.75,
            label=f"{DISPLAY_NAME[pathology]} (n={len(sub)})",
            zorder=2,
        )
    ax.set_title(title, color=INK, fontsize=13, pad=12)
    ax.legend(frameon=False, loc="best", fontsize=10, labelcolor=INK)
    fig.tight_layout()
    out = os.path.join(MORPH_DIR, out_name)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def plot_combined(df):
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=SURFACE)
    style_axes(ax)
    for (origin, pathology), color in COMBO_COLORS.items():
        sub = df[(df["origin"] == origin) & (df["pathology"] == pathology)]
        if sub.empty:
            continue
        ax.scatter(
            sub["tsne_x"],
            sub["tsne_y"],
            s=22,
            c=color,
            edgecolors=INK,
            linewidths=0.3,
            alpha=0.8,
            label=f"{origin} {DISPLAY_NAME[pathology]} (n={len(sub)})",
            zorder=2,
        )
    ax.set_title(
        "t-SNE of lesion morphometrics — real vs synthetic, Stroke vs Meningioma", color=INK, fontsize=13, pad=12
    )
    ax.legend(frameon=False, loc="best", fontsize=9, labelcolor=INK)

    fig.tight_layout()
    out = os.path.join(MORPH_DIR, "tsne_real_vs_synthetic_combined.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    synth = load(SYNTH_FILES, "Synthetic")
    print(f"synthetic rows: {len(synth)} ({synth['pathology'].value_counts().to_dict()})")
    synth_embedded = fit_tsne(synth)
    plot_single_origin(
        synth_embedded, PATHOLOGY_COLORS,
        "t-SNE of synthetic lesion morphometrics\nStroke vs Meningioma", "tsne_synthetic_only.png",
    )

    real = load(REAL_FILES, "Real")
    print(f"real rows: {len(real)} ({real['pathology'].value_counts().to_dict()})")
    real_embedded = fit_tsne(real)
    plot_single_origin(
        real_embedded, REAL_COLORS,
        "t-SNE of real lesion morphometrics\nStroke vs Meningioma", "tsne_real_only.png",
    )

    combined = pd.concat([real, synth], ignore_index=True)
    print(f"combined rows: {len(combined)} ({combined.groupby(['origin','pathology']).size().to_dict()})")
    combined_embedded = fit_tsne(combined)
    plot_combined(combined_embedded)


if __name__ == "__main__":
    main()
