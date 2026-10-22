#!/usr/bin/env python3
"""Pairwise disconnectome t-SNE scatter plots, reusing the all-classes embedding.

Subsets the already-fit 3-class t-SNE coordinates (weighted and binary)
rather than refitting t-SNE per pair, so coordinates stay consistent
with the all-classes plots.
"""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
COLORS = {"Glioma": "#2a78d6", "Ischaemic": "#eb6834", "Meningioma": "#1baf7a"}

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

PAIRS = [("Glioma", "Ischaemic"), ("Meningioma", "Ischaemic")]
VARIANTS = [
    ("tsne_all_classes_node_degrees_coords.tsv", "t-SNE of disconnectome node degrees", "node_degrees"),
    (
        "tsne_all_classes_binary_node_degrees_coords.tsv",
        "t-SNE of binarized node degrees (>50th pct of nonzero edges)",
        "binary_node_degrees",
    ),
]


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
    for coords_file, title_prefix, tag in VARIANTS:
        df = pd.read_csv(os.path.join(DISCO_DIR, coords_file), sep="\t", index_col=0)
        for cls_a, cls_b in PAIRS:
            fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
            style_axes(ax)
            for cls in [cls_a, cls_b]:
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
            ax.set_title(f"{title_prefix}\n{cls_b} vs {cls_a}", color=INK, fontsize=13, pad=12)
            ax.legend(frameon=False, loc="best", fontsize=10, labelcolor=INK)
            fig.tight_layout()
            out_path = os.path.join(DISCO_DIR, f"tsne_{cls_b.lower()}_{cls_a.lower()}_{tag}.png")
            fig.savefig(out_path, dpi=200, facecolor=SURFACE)
            plt.close(fig)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
