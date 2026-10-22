#!/usr/bin/env python3
"""Graph-theoretical measures of the disconnectome networks, real pathologies.

Each subject's disconnectome is binarized the same way as elsewhere in
this project (edges above the 50th percentile of that subject's own
nonzero edges -- see compute_disconnectome_binary_degrees.py), then
every measure below is computed on the induced subgraph of affected
nodes only (nodes with at least one surviving edge). This matters
because these are sparse, partial subgraphs of a potential full
connectome -- most of the 100 ROIs are untouched by any given lesion,
so measures are scoped to "the part of the network this lesion
actually reaches," not the full 100-node graph (which would make
density/clustering/etc. meaningless, dominated by disconnected zeros).

Ten measures, covering spread, cohesion, clustering, assortativity and
community structure:
  - n_affected_nodes, n_edges: raw spread (breadth of the affected subgraph)
  - density: how interconnected the affected region is, independent of its size
  - n_components, largest_component_fraction: fragmentation / cohesion
  - global_efficiency: integration (handles disconnected components natively,
    unlike characteristic path length, which is essential here)
  - avg_clustering: local triangle density (binary clustering coefficient)
  - assortativity: do high-degree affected nodes connect to other high-degree
    ones, or to low-degree ones?
  - modularity: strength of community structure (greedy modularity maximization)
  - degree_centralization: Freeman centralization -- hub-dominated vs even spread
"""
import glob
import os
import sys
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd

DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
CLASSES = ["Glioma", "Ischaemic", "Meningioma"]
OUTPUT = os.path.join(DISCO_DIR, "graph_measures_summary.tsv")
PERCENTILE = 50


def binarize(path):
    df = pd.read_csv(path, sep="\t", header=0, index_col=0)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    mat = df.values.astype(float)
    n = mat.shape[0]

    iu = np.triu_indices(n, k=1)
    weights = mat[iu]
    nonzero = weights[weights > 0]

    binary = np.zeros_like(mat, dtype=np.uint8)
    if nonzero.size:
        threshold = np.percentile(nonzero, PERCENTILE)
        keep = weights > threshold
        rows, cols = iu[0][keep], iu[1][keep]
        binary[rows, cols] = 1
        binary[cols, rows] = 1
    return binary, float(mat.sum())


MEASURE_KEYS = [
    "n_affected_nodes", "n_edges", "density", "n_components", "largest_component_fraction",
    "global_efficiency", "avg_clustering", "assortativity", "modularity", "degree_centralization",
]


def graph_measures(binary):
    degrees = binary.sum(axis=1)
    affected = np.where(degrees > 0)[0]
    n_affected = len(affected)
    n_edges = int(binary.sum() / 2)

    row = {k: np.nan for k in MEASURE_KEYS}
    row["n_affected_nodes"] = n_affected
    row["n_edges"] = n_edges

    if n_affected < 3 or n_edges < 2:
        return row

    sub = binary[np.ix_(affected, affected)]
    G = nx.from_numpy_array(sub)

    row["density"] = nx.density(G)
    comps = list(nx.connected_components(G))
    row["n_components"] = len(comps)
    row["largest_component_fraction"] = max(len(c) for c in comps) / n_affected
    row["global_efficiency"] = nx.global_efficiency(G)
    row["avg_clustering"] = nx.average_clustering(G)

    try:
        row["assortativity"] = nx.degree_assortativity_coefficient(G)
    except Exception:
        pass

    try:
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        row["modularity"] = nx.algorithms.community.quality.modularity(G, communities)
    except Exception:
        pass

    degs = np.array([d for _, d in G.degree()])
    n = len(degs)
    if n > 2:
        row["degree_centralization"] = float(np.sum(degs.max() - degs) / ((n - 1) * (n - 2)))

    return row


def _worker(args):
    pathology, path = args
    filename = os.path.basename(path)
    try:
        binary, total_weight = binarize(path)
        if total_weight == 0:
            return filename, pathology, None, "empty disconnectome"
        measures = graph_measures(binary)
        return filename, pathology, measures, None
    except Exception as e:
        return filename, pathology, None, str(e)


def main():
    jobs = []
    for pathology in CLASSES:
        for path in sorted(glob.glob(os.path.join(DISCO_DIR, pathology, "*.tsv"))):
            jobs.append((pathology, path))

    print(f"{len(jobs)} disconnectome files found across {CLASSES}")
    rows = {}
    errors = []
    with Pool(processes=8) as pool:
        for i, (filename, pathology, measures, err) in enumerate(pool.imap_unordered(_worker, jobs), 1):
            if err:
                errors.append((filename, err))
                continue
            measures["pathology"] = pathology
            rows[filename] = measures
            if i % 200 == 0:
                print(f"progress: {i}/{len(jobs)}")

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df[["pathology"] + MEASURE_KEYS].sort_index()
    df.to_csv(OUTPUT, sep="\t", index_label="filename")
    print(f"wrote {OUTPUT} ({len(df)} rows, {len(errors)} excluded as empty/error)")
    print(df.groupby("pathology").size())


if __name__ == "__main__":
    main()
