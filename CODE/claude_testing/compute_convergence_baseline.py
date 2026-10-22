#!/usr/bin/env python3
"""Baseline for the lesion-mask-to-disconnectome convergence framework.

Three real pathology groups (Glioma, Ischaemic/Stroke, Meningioma),
three feature spaces (morphometrics, disconnectome node weights,
disconnectome binary node degrees), standardized and PCA-reduced to a
common N_COMPONENTS dimensionality for cross-space comparability (same
method/rationale as compute_distribution_divergence.py).

Four convergence metrics per feature space, each handling all three
groups jointly rather than pairwise:
  - multiclass_auc: 5-fold CV macro one-vs-rest ROC-AUC, random forest.
  - silhouette: silhouette score using true pathology as the label.
  - permanova_r2: fraction of total SS explained by group membership,
    with a permutation p-value (999 permutations).
  - mean_pairwise_kl: mean of the three pairwise symmetrized k-NN KL
    divergences (Glioma-Ischaemic, Glioma-Meningioma, Ischaemic-Meningioma).

Lower values across the board = more convergence when moving from
morphometrics to disconnectome space.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from compute_distribution_divergence import knn_kl_divergence

MORPH_DIR = "/data/patrik/NPG/DATA/MORPHOMETRICS"
DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
OUT_DIR = "/data/patrik/NPG/DATA/ANALYSIS"
OUT_TSV = os.path.join(OUT_DIR, "convergence_baseline.tsv")

N_COMPONENTS = 10
GROUPS = ["Glioma", "Ischaemic", "Meningioma"]
PAIRS = [("Glioma", "Ischaemic"), ("Glioma", "Meningioma"), ("Ischaemic", "Meningioma")]

FEATURE_COLUMNS = [
    "volume_mm3", "equiv_diameter_mm", "surface_area_mm2", "surface_to_volume_ratio",
    "n_components", "laterality_index", "largest_component_volume_fraction", "sphericity",
    "axis1_mm", "axis2_mm", "axis3_mm", "elongation_ratio", "flatness_ratio", "solidity",
]


def load_morphometrics():
    files = {
        "Glioma": os.path.join(MORPH_DIR, "Glioma_morphometrics.csv"),
        "Ischaemic": os.path.join(MORPH_DIR, "Ischaemic_morphometrics.csv"),
        "Meningioma": os.path.join(MORPH_DIR, "Meningioma_morphometrics.csv"),
    }
    raw = {}
    for group, path in files.items():
        df = pd.read_csv(path)
        df = df[df["error"].fillna("") == ""].dropna(subset=FEATURE_COLUMNS)
        raw[group] = df[FEATURE_COLUMNS].values
    return raw


def load_disconnectome_weighted():
    df = pd.read_csv(os.path.join(DISCO_DIR, "node_degree_summary.tsv"), sep="\t", index_col=0)
    feat = [c for c in df.columns if c != "pathology"]
    raw = {}
    for group in GROUPS:
        sub = df[df["pathology"] == group]
        sub = sub[sub[feat].sum(axis=1) > 0]
        raw[group] = sub[feat].values
    return raw


def load_disconnectome_binary():
    weighted = pd.read_csv(os.path.join(DISCO_DIR, "node_degree_summary.tsv"), sep="\t", index_col=0)
    wfeat = [c for c in weighted.columns if c != "pathology"]
    binary = pd.read_csv(os.path.join(DISCO_DIR, "node_degree_binary_summary.tsv"), sep="\t", index_col=0)
    bfeat = [c for c in binary.columns if c != "pathology"]
    raw = {}
    for group in GROUPS:
        wsub = weighted[weighted["pathology"] == group]
        keep = wsub[wsub[wfeat].sum(axis=1) > 0].index
        bsub = binary[(binary["pathology"] == group) & (binary.index.isin(keep))]
        raw[group] = bsub[bfeat].values
    return raw


def standardize_and_reduce(raw, n_components=N_COMPONENTS):
    pooled = np.vstack([raw[g] for g in GROUPS])
    k = min(n_components, pooled.shape[1])
    scaler = StandardScaler().fit(pooled)
    pca = PCA(n_components=k, random_state=42).fit(scaler.transform(pooled))
    reduced = {g: pca.transform(scaler.transform(raw[g])) for g in GROUPS}
    return reduced, float(pca.explained_variance_ratio_.sum()), k


def multiclass_auc(reduced):
    X = np.vstack([reduced[g] for g in GROUPS])
    y = np.concatenate([[g] * len(reduced[g]) for g in GROUPS])
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc_ovr")
    return float(scores.mean()), float(scores.std())


def silhouette(reduced):
    X = np.vstack([reduced[g] for g in GROUPS])
    y = np.concatenate([[g] * len(reduced[g]) for g in GROUPS])
    return float(silhouette_score(X, y))


def permanova_r2(reduced, n_perm=999, seed=42):
    X = np.vstack([reduced[g] for g in GROUPS])
    y = np.concatenate([[g] * len(reduced[g]) for g in GROUPS])
    grand = X.mean(axis=0)
    ss_total = np.sum((X - grand) ** 2)

    def ss_within(labels):
        ss = 0.0
        for g in GROUPS:
            Xg = X[labels == g]
            ss += np.sum((Xg - Xg.mean(axis=0)) ** 2)
        return ss

    r2_obs = 1 - ss_within(y) / ss_total
    rng = np.random.default_rng(seed)
    perm_r2 = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        perm_r2[i] = 1 - ss_within(y_perm) / ss_total
    p_value = (np.sum(perm_r2 >= r2_obs) + 1) / (n_perm + 1)
    return float(r2_obs), float(p_value)


def mean_pairwise_kl(reduced):
    vals = []
    for a, b in PAIRS:
        kl_ab = knn_kl_divergence(reduced[a], reduced[b])
        kl_ba = knn_kl_divergence(reduced[b], reduced[a])
        vals.append((kl_ab + kl_ba) / 2)
    return float(np.mean(vals)), dict(zip(["_".join(p) for p in PAIRS], vals))


def run_feature_space(raw, label):
    reduced, explained_var, k = standardize_and_reduce(raw)
    auc_mean, auc_std = multiclass_auc(reduced)
    sil = silhouette(reduced)
    r2, p = permanova_r2(reduced)
    mean_kl, pair_kls = mean_pairwise_kl(reduced)
    print(f"[{label}] n={[len(raw[g]) for g in GROUPS]} k={k} explained_var={explained_var:.3f}")
    print(f"  multiclass AUC = {auc_mean:.3f} +/- {auc_std:.3f}")
    print(f"  silhouette     = {sil:.3f}")
    print(f"  PERMANOVA R2   = {r2:.3f} (p={p:.4f})")
    print(f"  mean pairwise KL = {mean_kl:.3f}  {pair_kls}")
    return {
        "feature_space": label,
        "n_glioma": len(raw["Glioma"]), "n_ischaemic": len(raw["Ischaemic"]), "n_meningioma": len(raw["Meningioma"]),
        "n_components": k, "explained_variance": explained_var,
        "multiclass_auc_mean": auc_mean, "multiclass_auc_std": auc_std,
        "silhouette": sil, "permanova_r2": r2, "permanova_p": p,
        "mean_pairwise_kl": mean_kl,
        **pair_kls,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    rows.append(run_feature_space(load_morphometrics(), "morphometrics"))
    rows.append(run_feature_space(load_disconnectome_weighted(), "disconnectome_node_weights"))
    rows.append(run_feature_space(load_disconnectome_binary(), "disconnectome_binary_node_degrees"))

    result = pd.DataFrame(rows)
    result.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")


if __name__ == "__main__":
    main()
