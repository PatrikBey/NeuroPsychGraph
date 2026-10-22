#!/usr/bin/env python3
"""Variant 1: residualize disconnectome node features on breadth (n_nonzero_nodes)
before recomputing KL/AUC -- within-group linear regression per node, so only the
within-pathology volume/breadth-driven variance is removed, not the real between-
group difference. Morphometrics is untouched (breadth isn't a morphometric concept).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import compute_distribution_divergence as cdd

OUT_TSV = os.path.join(cdd.OUT_DIR, "group_divergence_summary_breadth_corrected.tsv")


def residualize_within_group(raw, breadth):
    """raw[group]: (n, d) feature matrix. breadth[group]: (n,) covariate.
    Fits y = a + b*breadth per node, per group, returns residuals."""
    out = {}
    for group, Y in raw.items():
        b = breadth[group]
        X = np.column_stack([np.ones_like(b), b])
        coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        predicted = X @ coeffs
        out[group] = Y - predicted
    return out


def load_with_breadth(group_map):
    raw, breadth = {}, {}
    for group, (path, pathology) in group_map.items():
        df = pd.read_csv(path, sep="\t", index_col=0)
        df = df[df["pathology"] == pathology]
        feat = [c for c in df.columns if c != "pathology"]
        df = df[df[feat].sum(axis=1) > 0]  # exclude empty disconnectomes
        raw[group] = df[feat].values
        breadth[group] = (df[feat] > 1e-9).sum(axis=1).values.astype(float)
    return raw, breadth


def main():
    all_rows = []

    weighted_map = {
        "Real Stroke": (os.path.join(cdd.DISCO_DIR, "node_degree_summary.tsv"), "Ischaemic"),
        "Real Meningioma": (os.path.join(cdd.DISCO_DIR, "node_degree_summary.tsv"), "Meningioma"),
        "Synthetic Stroke": (os.path.join(cdd.DISCO_DIR, "node_degree_summary_synth.tsv"), "Ischaemic"),
        "Synthetic Meningioma": (os.path.join(cdd.DISCO_DIR, "node_degree_summary_synth.tsv"), "Meningioma"),
    }
    weighted_raw, weighted_breadth = load_with_breadth(weighted_map)
    weighted_resid = residualize_within_group(weighted_raw, weighted_breadth)
    all_rows += cdd.run_feature_space(weighted_resid, "disconnectome_node_weights_breadth_corrected")

    binary_map = {
        "Real Stroke": (os.path.join(cdd.DISCO_DIR, "node_degree_binary_summary.tsv"), "Ischaemic"),
        "Real Meningioma": (os.path.join(cdd.DISCO_DIR, "node_degree_binary_summary.tsv"), "Meningioma"),
        "Synthetic Stroke": (os.path.join(cdd.DISCO_DIR, "node_degree_binary_summary_synth.tsv"), "Ischaemic"),
        "Synthetic Meningioma": (os.path.join(cdd.DISCO_DIR, "node_degree_binary_summary_synth.tsv"), "Meningioma"),
    }
    binary_raw_full, _ = load_with_breadth(binary_map)
    # align binary rows to the same subjects as the weighted (non-empty) set, reuse weighted breadth
    binary_raw = {}
    for group, (path, pathology) in binary_map.items():
        df = pd.read_csv(path, sep="\t", index_col=0)
        df = df[df["pathology"] == pathology]
        feat = [c for c in df.columns if c != "pathology"]
        wdf = pd.read_csv(weighted_map[group][0], sep="\t", index_col=0)
        wdf = wdf[wdf["pathology"] == weighted_map[group][1]]
        wfeat = [c for c in wdf.columns if c != "pathology"]
        keep_index = wdf[wdf[wfeat].sum(axis=1) > 0].index
        df = df[df.index.isin(keep_index)]
        assert list(df.index) == list(keep_index), f"row order mismatch for {group}"
        binary_raw[group] = df[feat].values
    binary_resid = residualize_within_group(binary_raw, weighted_breadth)
    all_rows += cdd.run_feature_space(binary_resid, "disconnectome_binary_node_degrees_breadth_corrected")

    result = pd.DataFrame(all_rows)
    result.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
