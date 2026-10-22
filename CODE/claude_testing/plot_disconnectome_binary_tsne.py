#!/usr/bin/env python3
"""t-SNE scatter plot of binarized (nonzero-edge 50th-percentile) disconnectome node degrees, all classes together."""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
SUMMARY = os.path.join(DISCO_DIR, "node_degree_binary_summary.tsv")
WEIGHTED_SUMMARY = os.path.join(DISCO_DIR, "node_degree_summary.tsv")
OUT_PNG = os.path.join(DISCO_DIR, "tsne_all_classes_binary_node_degrees.png")
OUT_COORDS = os.path.join(DISCO_DIR, "tsne_all_classes_binary_node_degrees_coords.tsv")
COLORS = {"Glioma": "#2a78d6", "Ischaemic": "#eb6834", "Meningioma": "#1baf7a"}

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


def main():
    df = pd.read_csv(SUMMARY, sep="\t", index_col=0)
    feature_cols = [c for c in df.columns if c != "pathology"]

    # binarizing an all-tied-at-zero matrix still yields a (spurious, arbitrary)
    # 50%-edge pattern, so emptiness must be detected from the weighted matrix.
    weighted = pd.read_csv(WEIGHTED_SUMMARY, sep="\t", index_col=0)
    weighted_feat_cols = [c for c in weighted.columns if c != "pathology"]
    empty_files = set(weighted.index[weighted[weighted_feat_cols].sum(axis=1) == 0])

    n_total = len(df)
    df = df[~df.index.isin(empty_files)]
    print(f"excluded {n_total - len(df)} empty (all-zero weighted) disconnectomes, {len(df)} remain")

    X = df[feature_cols].values
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    embedding = tsne.fit_transform(X)
    df["tsne_x"], df["tsne_y"] = embedding[:, 0], embedding[:, 1]

    classes = sorted(df["pathology"].unique())

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    style_axes(ax)
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
    ax.set_title(
        f"t-SNE of binarized node degrees (>50th pct of nonzero edges)\n{' vs '.join(classes)}",
        color=INK,
        fontsize=13,
        pad=12,
    )
    ax.legend(frameon=False, loc="best", fontsize=10, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")

    df[["pathology", "tsne_x", "tsne_y"]].to_csv(OUT_COORDS, sep="\t", index_label="filename")
    print(f"wrote {OUT_COORDS}")


if __name__ == "__main__":
    main()
