#!/usr/bin/env python3
"""t-SNE of real disconnectome node degrees, Stroke (Ischaemic) vs Meningioma only.

Glioma excluded, own dedicated t-SNE fit (not a subset of the 3-class or
real+synthetic embeddings). Colors match the real/synthetic combined
plot's real-side colors: Meningioma = crimson, Stroke = purple.
"""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
REAL_WEIGHTED = os.path.join(DISCO_DIR, "node_degree_summary.tsv")
REAL_BINARY = os.path.join(DISCO_DIR, "node_degree_binary_summary.tsv")

DISPLAY_NAME = {"Ischaemic": "Stroke", "Meningioma": "Meningioma"}
COLORS = {"Meningioma": "#DC143C", "Ischaemic": "#800080"}  # crimson / purple

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_xlabel("t-SNE dim 1", color=MUTED, fontsize=10)
    ax.set_ylabel("t-SNE dim 2", color=MUTED, fontsize=10)


def load_weighted(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df[df["pathology"].isin(["Ischaemic", "Meningioma"])]
    feat = [c for c in df.columns if c != "pathology"]
    df = df[df[feat].sum(axis=1) > 0]
    return df, feat


def load_binary(path, empty_files):
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df[df["pathology"].isin(["Ischaemic", "Meningioma"])]
    df = df[~df.index.isin(empty_files)]
    feat = [c for c in df.columns if c != "pathology"]
    return df, feat


def fit_tsne(df, feat):
    embedding = TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(df[feat].values)
    df = df.copy()
    df["tsne_x"], df["tsne_y"] = embedding[:, 0], embedding[:, 1]
    return df


def plot(df, tag, title):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    style_axes(ax)
    for pathology in ["Ischaemic", "Meningioma"]:
        sub = df[df["pathology"] == pathology]
        ax.scatter(
            sub["tsne_x"], sub["tsne_y"], s=22, c=COLORS[pathology],
            edgecolors=INK, linewidths=0.3, alpha=0.8,
            label=f"{DISPLAY_NAME[pathology]} (n={len(sub)})", zorder=2,
        )
    ax.set_title(title, color=INK, fontsize=12, pad=12)
    ax.legend(frameon=False, loc="best", fontsize=10, labelcolor=INK)
    fig.tight_layout()
    out = os.path.join(DISCO_DIR, f"tsne_real_stroke_meningioma_{tag}.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    weighted_df, feat = load_weighted(REAL_WEIGHTED)
    print(f"weighted rows: {len(weighted_df)} ({weighted_df['pathology'].value_counts().to_dict()})")
    plot(fit_tsne(weighted_df, feat), "node_degrees", "t-SNE of real disconnectome node weights\nStroke vs Meningioma")

    raw_weighted = pd.read_csv(REAL_WEIGHTED, sep="\t", index_col=0)
    raw_weighted = raw_weighted[raw_weighted["pathology"].isin(["Ischaemic", "Meningioma"])]
    raw_feat = [c for c in raw_weighted.columns if c != "pathology"]
    empty_files = set(raw_weighted.index[raw_weighted[raw_feat].sum(axis=1) == 0])

    binary_df, bfeat = load_binary(REAL_BINARY, empty_files)
    print(f"binary rows: {len(binary_df)} ({binary_df['pathology'].value_counts().to_dict()})")
    plot(
        fit_tsne(binary_df, bfeat), "binary_node_degrees",
        "t-SNE of real disconnectome binary node degrees\nStroke vs Meningioma",
    )


if __name__ == "__main__":
    main()
