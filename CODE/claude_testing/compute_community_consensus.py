#!/usr/bin/env python3
"""Do the three aetiologies organize disconnection into similar communities?

Per subject: binarize (same 50th-percentile-of-nonzero-edges method as
elsewhere), restrict to the affected-node subgraph, run greedy modularity
community detection, and record (a) which community each affected ROI
falls into, and (b) the "hub" = the highest-degree affected node, and
its community's other members ("satellites").

Two analyses, matching the two questions:

1. Cross-pathology community similarity. Build a 100x100 ROI-pair
   co-membership matrix per pathology: P(i,j in the same community |
   both i and j are affected), estimated across all subjects in that
   pathology. Spearman-correlate the three pathologies' matrices
   (upper triangle) against each other -- high correlation means the
   three aetiologies tend to group the same regions together even
   though the individual lesions differ.

2. Hub-independent satellite enrichment. For each subject, compute the
   chance rate that a random other affected node would land in the
   hub's community under uniform random assignment with the same
   community-size distribution: (hub_community_size - 1) / (n_affected - 1).
   For each ROI, average the observed hub-community membership rate
   and the chance rate across all subject-instances where that ROI is
   affected (and isn't the hub itself); enrichment = observed/expected.
   ROIs/networks with enrichment > 1 consistently across all three
   pathologies are candidates for a shared "satellite" community that
   gets swept in regardless of where the lesion/hub actually is.

Both analyses are also summarized at the Schaefer 17-network level
(ROI name's 3rd underscore-separated token) since raw 100-ROI output
isn't very interpretable on its own.
"""
import glob
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
OUT_DIR = "/data/patrik/NPG/DATA/ANALYSIS"
CLASSES = ["Glioma", "Ischaemic", "Meningioma"]
PERCENTILE = 50


def network_of(roi_name):
    return roi_name.split("_")[2]


def binarize_and_partition(path):
    df = pd.read_csv(path, sep="\t", header=0, index_col=0)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    roi_names = df.columns.tolist()
    mat = df.values.astype(float)
    n = mat.shape[0]

    iu = np.triu_indices(n, k=1)
    weights = mat[iu]
    nonzero = weights[weights > 0]
    if nonzero.size == 0:
        return None

    binary = np.zeros_like(mat, dtype=np.uint8)
    threshold = np.percentile(nonzero, PERCENTILE)
    keep = weights > threshold
    rows, cols = iu[0][keep], iu[1][keep]
    binary[rows, cols] = 1
    binary[cols, rows] = 1

    degrees = binary.sum(axis=1)
    affected = np.where(degrees > 0)[0]
    if len(affected) < 4:
        return None

    sub = binary[np.ix_(affected, affected)]
    G = nx.from_numpy_array(sub)
    try:
        communities = nx.algorithms.community.greedy_modularity_communities(G)
    except Exception:
        return None

    local_to_roi = {local: roi_names[global_idx] for local, global_idx in enumerate(affected)}
    membership = {}
    for cid, comm in enumerate(communities):
        for local in comm:
            membership[local_to_roi[local]] = cid

    sub_degrees = dict(G.degree())
    hub_local = max(sub_degrees, key=sub_degrees.get)
    hub_roi = local_to_roi[hub_local]
    hub_cid = membership[hub_roi]
    hub_community_size = sum(1 for v in membership.values() if v == hub_cid)
    satellites = [roi for roi, cid in membership.items() if cid == hub_cid and roi != hub_roi]

    affected_rois = [local_to_roi[i] for i in range(len(affected))]
    return {
        "affected_rois": affected_rois,
        "membership": membership,
        "hub_roi": hub_roi,
        "hub_community_size": hub_community_size,
        "satellites": satellites,
        "n_affected": len(affected),
    }


def _worker(args):
    pathology, path = args
    try:
        result = binarize_and_partition(path)
    except Exception:
        return pathology, None
    return pathology, result


def main():
    jobs = []
    for pathology in CLASSES:
        for path in sorted(glob.glob(os.path.join(DISCO_DIR, pathology, "*.tsv"))):
            jobs.append((pathology, path))
    print(f"{len(jobs)} files")

    per_pathology_results = {p: [] for p in CLASSES}
    with Pool(processes=8) as pool:
        for i, (pathology, result) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=4), 1):
            if result is not None:
                per_pathology_results[pathology].append(result)
            if i % 500 == 0:
                print(f"progress: {i}/{len(jobs)}")

    for p in CLASSES:
        print(f"{p}: {len(per_pathology_results[p])} usable subjects")

    all_rois = sorted({roi for results in per_pathology_results.values() for r in results for roi in r["affected_rois"]})
    roi_idx = {roi: i for i, roi in enumerate(all_rois)}
    n_roi = len(all_rois)

    # --- Analysis 1: co-membership matrices per pathology ---
    comembership = {}
    for p in CLASSES:
        both_count = np.zeros((n_roi, n_roi))
        same_comm_count = np.zeros((n_roi, n_roi))
        for r in per_pathology_results[p]:
            rois = r["affected_rois"]
            membership = r["membership"]
            idxs = [roi_idx[roi] for roi in rois]
            comms = np.array([membership[roi] for roi in rois])
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    ia, ib = idxs[a], idxs[b]
                    both_count[ia, ib] += 1
                    both_count[ib, ia] += 1
                    if comms[a] == comms[b]:
                        same_comm_count[ia, ib] += 1
                        same_comm_count[ib, ia] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.divide(same_comm_count, both_count, out=np.full_like(both_count, np.nan), where=both_count > 0)
        comembership[p] = frac
        np.save(os.path.join(OUT_DIR, f"comembership_{p}.npy"), frac)

    pairs = [("Glioma", "Ischaemic"), ("Glioma", "Meningioma"), ("Ischaemic", "Meningioma")]
    iu = np.triu_indices(n_roi, k=1)
    corr_rows = []
    for a, b in pairs:
        va, vb = comembership[a][iu], comembership[b][iu]
        valid = ~np.isnan(va) & ~np.isnan(vb)
        rho, p_val = stats.spearmanr(va[valid], vb[valid])
        corr_rows.append({"pair": f"{a}_vs_{b}", "n_pairs": int(valid.sum()), "spearman_rho": rho, "p_value": p_val})
        print(f"co-membership similarity {a} vs {b}: rho={rho:.3f} (n={valid.sum()})")
    pd.DataFrame(corr_rows).to_csv(os.path.join(OUT_DIR, "comembership_cross_pathology_correlation.tsv"), sep="\t", index=False)

    # --- Analysis 2: hub-independent satellite enrichment, per ROI and per network ---
    roi_obs = defaultdict(lambda: defaultdict(float))
    roi_exp = defaultdict(lambda: defaultdict(float))
    roi_n = defaultdict(lambda: defaultdict(int))
    for p in CLASSES:
        for r in per_pathology_results[p]:
            n_affected = r["n_affected"]
            if n_affected < 2:
                continue
            chance = (r["hub_community_size"] - 1) / (n_affected - 1)
            satellite_set = set(r["satellites"])
            for roi in r["affected_rois"]:
                if roi == r["hub_roi"]:
                    continue
                roi_obs[p][roi] += 1.0 if roi in satellite_set else 0.0
                roi_exp[p][roi] += chance
                roi_n[p][roi] += 1

    roi_rows = []
    for roi in all_rois:
        row = {"roi": roi, "network": network_of(roi)}
        for p in CLASSES:
            n = roi_n[p].get(roi, 0)
            obs = roi_obs[p].get(roi, 0.0) / n if n > 0 else np.nan
            exp = roi_exp[p].get(roi, 0.0) / n if n > 0 else np.nan
            row[f"{p}_n"] = n
            row[f"{p}_observed_rate"] = obs
            row[f"{p}_expected_rate"] = exp
            row[f"{p}_enrichment"] = obs / exp if exp and exp > 0 else np.nan
        roi_rows.append(row)
    roi_df = pd.DataFrame(roi_rows)
    roi_df.to_csv(os.path.join(OUT_DIR, "satellite_enrichment_per_roi.tsv"), sep="\t", index=False)

    network_df = roi_df.groupby("network").agg(
        n_rois=("roi", "count"),
        **{f"{p}_enrichment_mean": (f"{p}_enrichment", "mean") for p in CLASSES},
        **{f"{p}_n_total": (f"{p}_n", "sum") for p in CLASSES},
    ).reset_index()
    network_df["min_enrichment_across_pathologies"] = network_df[[f"{p}_enrichment_mean" for p in CLASSES]].min(axis=1)
    network_df = network_df.sort_values("min_enrichment_across_pathologies", ascending=False)
    network_df.to_csv(os.path.join(OUT_DIR, "satellite_enrichment_per_network.tsv"), sep="\t", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.3g}")
    print("\n=== satellite enrichment by network (sorted by min across pathologies) ===")
    print(network_df.to_string(index=False))


if __name__ == "__main__":
    main()
