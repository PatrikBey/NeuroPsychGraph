#!/usr/bin/env python3
"""Node-degree t-SNE plots for synthetic disconnectomes, and combined with real.

For each variant (weighted node-weight, binary node-degree):
  - synthetic-only plot (Ischaemic/Stroke vs Meningioma)
  - combined real+synthetic plot, real Glioma excluded, Meningioma/Stroke
    kept separate (not merged), four explicit colors matching the
    morphometrics combined plot: real Meningioma=crimson, synthetic
    Meningioma=gold, real Stroke=purple, synthetic Stroke=teal.

No standardization across subjects for either variant, consistent with
the real-data node-degree analysis (disconnectomes share one underlying
atlas/space, so raw magnitude is meaningful, not noise to remove).
Empty (all-zero weighted) disconnectomes are excluded in both variants,
detected from the weighted matrix (binarizing an all-zero-tied matrix
does not itself look empty).
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
SYNTH_WEIGHTED = os.path.join(DISCO_DIR, "node_degree_summary_synth.tsv")
SYNTH_BINARY = os.path.join(DISCO_DIR, "node_degree_binary_summary_synth.tsv")

DISPLAY_NAME = {"Ischaemic": "Stroke", "Meningioma": "Meningioma"}
PATHOLOGY_COLORS = {"Ischaemic": "#008080", "Meningioma": "#FFD700"}  # synthetic-only plot: teal / gold
COMBO_COLORS = {
    ("Real", "Meningioma"): "#DC143C",  # crimson
    ("Synthetic", "Meningioma"): "#FFD700",  # gold
    ("Real", "Ischaemic"): "#800080",  # purple
    ("Synthetic", "Ischaemic"): "#008080",  # teal
}

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


def load_weighted(path, keep_pathologies=None):
    df = pd.read_csv(path, sep="\t", index_col=0)
    if keep_pathologies is not None:
        df = df[df["pathology"].isin(keep_pathologies)]
    feat = [c for c in df.columns if c != "pathology"]
    df = df[df[feat].sum(axis=1) > 0]
    return df, feat


def load_binary(path, empty_files, keep_pathologies=None):
    df = pd.read_csv(path, sep="\t", index_col=0)
    if keep_pathologies is not None:
        df = df[df["pathology"].isin(keep_pathologies)]
    df = df[~df.index.isin(empty_files)]
    feat = [c for c in df.columns if c != "pathology"]
    return df, feat


def fit_tsne(df, feat):
    embedding = TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(df[feat].values)
    df = df.copy()
    df["tsne_x"], df["tsne_y"] = embedding[:, 0], embedding[:, 1]
    return df


def plot_synthetic_only(df, tag, title):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    style_axes(ax)
    for pathology in ["Ischaemic", "Meningioma"]:
        sub = df[df["pathology"] == pathology]
        ax.scatter(
            sub["tsne_x"], sub["tsne_y"], s=22, c=PATHOLOGY_COLORS[pathology],
            edgecolors=INK, linewidths=0.3, alpha=0.75,
            label=f"{DISPLAY_NAME[pathology]} (n={len(sub)})", zorder=2,
        )
    ax.set_title(title, color=INK, fontsize=12, pad=12)
    ax.legend(frameon=False, loc="best", fontsize=10, labelcolor=INK)
    fig.tight_layout()
    out = os.path.join(DISCO_DIR, f"tsne_synthetic_only_{tag}.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def plot_combined(df, tag, title):
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=SURFACE)
    style_axes(ax)
    for (origin, pathology), color in COMBO_COLORS.items():
        sub = df[(df["origin"] == origin) & (df["pathology"] == pathology)]
        if sub.empty:
            continue
        ax.scatter(
            sub["tsne_x"], sub["tsne_y"], s=22, c=color,
            edgecolors=INK, linewidths=0.3, alpha=0.8,
            label=f"{origin} {DISPLAY_NAME[pathology]} (n={len(sub)})", zorder=2,
        )
    ax.set_title(title, color=INK, fontsize=12, pad=12)
    ax.legend(frameon=False, loc="best", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    out = os.path.join(DISCO_DIR, f"tsne_real_vs_synthetic_{tag}.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def run_variant(real_path, synth_path, tag, label, loader):
    synth_df, feat = loader(synth_path)
    print(f"[{label}] synthetic rows: {len(synth_df)} ({synth_df['pathology'].value_counts().to_dict()})")
    synth_embedded = fit_tsne(synth_df, feat)
    plot_synthetic_only(synth_embedded, tag, f"t-SNE of synthetic disconnectome {label}\nStroke vs Meningioma")

    real_df, real_feat = loader(real_path, keep_pathologies=["Ischaemic", "Meningioma"])
    assert real_feat == feat
    real_df = real_df.copy()
    real_df["origin"] = "Real"
    synth_df = synth_df.copy()
    synth_df["origin"] = "Synthetic"
    combined = pd.concat([real_df, synth_df], ignore_index=False)
    print(f"[{label}] combined rows: {len(combined)} ({combined.groupby(['origin','pathology']).size().to_dict()})")
    combined_embedded = fit_tsne(combined, feat)
    plot_combined(
        combined_embedded, tag,
        f"t-SNE of disconnectome {label}\nreal vs synthetic, Stroke vs Meningioma",
    )


def main():
    run_variant(
        REAL_WEIGHTED, SYNTH_WEIGHTED, "node_degrees", "node weights",
        loader=lambda p, keep_pathologies=None: load_weighted(p, keep_pathologies),
    )

    weighted_real = pd.read_csv(REAL_WEIGHTED, sep="\t", index_col=0)
    weighted_synth = pd.read_csv(SYNTH_WEIGHTED, sep="\t", index_col=0)
    feat_cols = [c for c in weighted_real.columns if c != "pathology"]
    empty_files = set(weighted_real.index[weighted_real[feat_cols].sum(axis=1) == 0]) | set(
        weighted_synth.index[weighted_synth[feat_cols].sum(axis=1) == 0]
    )

    run_variant(
        REAL_BINARY, SYNTH_BINARY, "binary_node_degrees", "binary node degrees",
        loader=lambda p, keep_pathologies=None: load_binary(p, empty_files, keep_pathologies),
    )


if __name__ == "__main__":
    main()
