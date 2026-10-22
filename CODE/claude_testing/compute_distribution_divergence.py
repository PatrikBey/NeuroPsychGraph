#!/usr/bin/env python3
"""Quantify how different pairs of groups are, across three feature spaces:
morphometrics (14-dim), weighted disconnectome node weights (100-dim),
and binarized node degrees (100-dim) -- and do it in a way that's
comparable ACROSS feature spaces, which is the actual point: whether
the disconnectome view separates Stroke from Meningioma more or less
than morphometry does.

Raw KL is not comparable across spaces of different dimensionality: the
k-NN KL estimator has an explicit d/n scaling term, so a 100-dim space
inflates KL relative to a 14-dim space purely from dimensionality, not
from any real difference in separation. Fix: standardize every feature
space the same way, then PCA-reduce ALL of them to the same fixed
number of components (N_COMPONENTS) before computing anything. This
deliberately departs from the t-SNE plots' "don't standardize
disconnectome features" convention -- that convention was about not
distorting node-level magnitude in a visualization; here the goal is a
cross-space-comparable number, which requires putting every space on
the same scale and dimensionality.

Two complementary measures per comparison, computed on the shared
N_COMPONENTS-dim projection (never on a t-SNE projection -- t-SNE
coordinates are a stochastic, nonlinear embedding with no reliable
probabilistic meaning):

  - KL divergence (both directions), via the Wang/Kulkarni/Verdu k-NN
    estimator -- works directly on continuous multivariate samples
    without needing a parametric density model.
  - 5-fold cross-validated ROC-AUC of a random forest trained to tell
    the two groups apart -- a model-free, easy-to-interpret sanity
    check (0.5 = indistinguishable, 1.0 = perfectly separable) that
    doesn't depend on the KL estimator's assumptions.

Comparisons: Real Stroke vs Real Meningioma, Synthetic Stroke vs
Synthetic Meningioma, Real Meningioma vs Synthetic Meningioma, Real
Stroke vs Synthetic Stroke -- each repeated across the three feature
spaces (12 rows total).
"""
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

N_COMPONENTS = 10

MORPH_DIR = "/data/patrik/NPG/DATA/MORPHOMETRICS"
DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
OUT_DIR = "/data/patrik/NPG/DATA/ANALYSIS"
OUT_TSV = os.path.join(OUT_DIR, "group_divergence_summary.tsv")

FEATURE_COLUMNS = [
    "volume_mm3", "equiv_diameter_mm", "surface_area_mm2", "surface_to_volume_ratio",
    "n_components", "laterality_index", "largest_component_volume_fraction", "sphericity",
    "axis1_mm", "axis2_mm", "axis3_mm", "elongation_ratio", "flatness_ratio", "solidity",
]

GROUPS = ["Real Stroke", "Real Meningioma", "Synthetic Stroke", "Synthetic Meningioma"]
COMPARISONS = [
    ("Real Stroke", "Real Meningioma"),
    ("Synthetic Stroke", "Synthetic Meningioma"),
    ("Real Meningioma", "Synthetic Meningioma"),
    ("Real Stroke", "Synthetic Stroke"),
]


def knn_kl_divergence(X, Y, k=5):
    """Wang/Kulkarni/Verdu kNN estimator of KL(P||Q), X ~ P (n,d), Y ~ Q (m,d)."""
    n, d = X.shape
    m = Y.shape[0]
    rho = cKDTree(X).query(X, k=k + 1)[0][:, k]  # k-th NN within X, excluding self
    nu = cKDTree(Y).query(X, k=k)[0][:, k - 1]  # k-th NN in Y
    eps = 1e-12
    return float((d / n) * np.sum(np.log((nu + eps) / (rho + eps))) + np.log(m / (n - 1)))


def classifier_auc(X, y, n_splits=5):
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def standardize_and_reduce(raw, group_a, group_b, n_components=N_COMPONENTS):
    """Fit scaler+PCA on the pooled two-group sample (no leakage across comparisons),
    project both groups into the same N_COMPONENTS-dim space, every feature space,
    every comparison -- this is what makes KL numbers comparable across spaces."""
    pooled = np.vstack([raw[group_a], raw[group_b]])
    k = min(n_components, pooled.shape[1])
    scaler = StandardScaler().fit(pooled)
    pca = PCA(n_components=k, random_state=42).fit(scaler.transform(pooled))
    Xa = pca.transform(scaler.transform(raw[group_a]))
    Xb = pca.transform(scaler.transform(raw[group_b]))
    return Xa, Xb, float(pca.explained_variance_ratio_.sum()), k


def summarize_pair(raw, group_a, group_b, label):
    Xa, Xb, explained_var, k = standardize_and_reduce(raw, group_a, group_b)
    kl_ab = knn_kl_divergence(Xa, Xb)
    kl_ba = knn_kl_divergence(Xb, Xa)
    X = np.vstack([Xa, Xb])
    y = np.array([0] * len(Xa) + [1] * len(Xb))
    auc_mean, auc_std = classifier_auc(X, y)
    return {
        "feature_space": label,
        "group_a": group_a,
        "n_a": len(raw[group_a]),
        "group_b": group_b,
        "n_b": len(raw[group_b]),
        "n_components": k,
        "explained_variance": explained_var,
        "kl_a_given_b": kl_ab,
        "kl_b_given_a": kl_ba,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
    }


def load_morphometrics():
    files = {
        "Real Stroke": os.path.join(MORPH_DIR, "Ischaemic_morphometrics.csv"),
        "Real Meningioma": os.path.join(MORPH_DIR, "Meningioma_morphometrics.csv"),
        "Synthetic Stroke": os.path.join(MORPH_DIR, "SYNTH_Ischaemic_morphometrics.csv"),
        "Synthetic Meningioma": os.path.join(MORPH_DIR, "SYNTH_Meningioma_morphometrics.csv"),
    }
    raw = {}
    for group, path in files.items():
        df = pd.read_csv(path)
        df = df[df["error"].fillna("") == ""].dropna(subset=FEATURE_COLUMNS)
        raw[group] = df[FEATURE_COLUMNS].values
    return raw


def load_disconnectome(weighted_path, synth_weighted_path, group_map, feature_filter=None):
    """group_map: {group_label: (path, pathology, is_synth)}"""
    raw = {}
    for group, (path, pathology) in group_map.items():
        df = pd.read_csv(path, sep="\t", index_col=0)
        df = df[df["pathology"] == pathology]
        feat = [c for c in df.columns if c != "pathology"]
        df = df[df[feat].sum(axis=1) > 0]  # exclude empty disconnectomes
        raw[group] = df[feat].values
    return raw


def run_feature_space(raw, label):
    rows = []
    for group_a, group_b in COMPARISONS:
        print(f"[{label}] {group_a} (n={len(raw[group_a])}) vs {group_b} (n={len(raw[group_b])})")
        rows.append(summarize_pair(raw, group_a, group_b, label))
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []

    morph_raw = load_morphometrics()
    all_rows += run_feature_space(morph_raw, "morphometrics")

    weighted_map = {
        "Real Stroke": (os.path.join(DISCO_DIR, "node_degree_summary.tsv"), "Ischaemic"),
        "Real Meningioma": (os.path.join(DISCO_DIR, "node_degree_summary.tsv"), "Meningioma"),
        "Synthetic Stroke": (os.path.join(DISCO_DIR, "node_degree_summary_synth.tsv"), "Ischaemic"),
        "Synthetic Meningioma": (os.path.join(DISCO_DIR, "node_degree_summary_synth.tsv"), "Meningioma"),
    }
    weighted_raw = load_disconnectome(None, None, weighted_map)
    all_rows += run_feature_space(weighted_raw, "disconnectome_node_weights")

    binary_map = {
        "Real Stroke": (os.path.join(DISCO_DIR, "node_degree_binary_summary.tsv"), "Ischaemic"),
        "Real Meningioma": (os.path.join(DISCO_DIR, "node_degree_binary_summary.tsv"), "Meningioma"),
        "Synthetic Stroke": (os.path.join(DISCO_DIR, "node_degree_binary_summary_synth.tsv"), "Ischaemic"),
        "Synthetic Meningioma": (os.path.join(DISCO_DIR, "node_degree_binary_summary_synth.tsv"), "Meningioma"),
    }
    # binary tables don't zero out empty-disconnectome subjects themselves, but weighted
    # tables do (via sum>0 filtering inside load_disconnectome against weighted data);
    # filter binary rows down to the same index as the matching weighted group.
    binary_raw_full = {}
    for group, (path, pathology) in binary_map.items():
        df = pd.read_csv(path, sep="\t", index_col=0)
        df = df[df["pathology"] == pathology]
        binary_raw_full[group] = df

    weighted_index = {}
    for group, (path, pathology) in weighted_map.items():
        df = pd.read_csv(path, sep="\t", index_col=0)
        df = df[df["pathology"] == pathology]
        feat = [c for c in df.columns if c != "pathology"]
        df = df[df[feat].sum(axis=1) > 0]
        weighted_index[group] = set(df.index)

    binary_raw = {}
    for group, df in binary_raw_full.items():
        df = df[df.index.isin(weighted_index[group])]
        feat = [c for c in df.columns if c != "pathology"]
        binary_raw[group] = df[feat].values

    all_rows += run_feature_space(binary_raw, "disconnectome_binary_node_degrees")

    result = pd.DataFrame(all_rows)
    result.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
