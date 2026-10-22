#!/usr/bin/env python3
"""Binarized disconnectome node-degree table.

Each subject's 100x100 weighted matrix is thresholded individually at
the 50th percentile of that subject's *nonzero* edges (upper triangle,
excluding the zero diagonal): edges with weight above the threshold
are kept as 1, the rest (including all exact-zero edges) set to 0,
then mirrored to stay symmetric. Binary node degree is the row sum of
the resulting 0/1 matrix (count of surviving connections per ROI).

Note on method: computing the 50th percentile over *all* possible
edges (including the exact-zero ones) degenerates for these matrices,
since most subjects have well over 50% exact-zero edges -- the
percentile value is 0 for nearly everyone, and naive rank-based
50%-of-all-edges thresholding forces padding with arbitrary zero-tied
edges for every subject (worse for sparse subjects, who end up almost
entirely defined by identical filler -- a clustering artifact, not
signal). Restricting the percentile to the nonzero edges only avoids
both problems: the threshold is computed from -- and applied to --
only the edges that actually carry data.
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
CLASSES = ["Glioma", "Ischaemic", "Meningioma"]
OUTPUT = os.path.join(DISCO_DIR, "node_degree_binary_summary.tsv")
PERCENTILE = 50


def parse_binary_degrees(path):
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

    return df.columns.tolist(), binary.sum(axis=0)


def _worker(args):
    pathology, path = args
    filename = os.path.basename(path)
    try:
        roi_names, degrees = parse_binary_degrees(path)
        return filename, pathology, roi_names, degrees, None
    except Exception as e:
        return filename, pathology, None, None, str(e)


def main():
    jobs = []
    for pathology in CLASSES:
        for path in sorted(glob.glob(os.path.join(DISCO_DIR, pathology, "*.tsv"))):
            jobs.append((pathology, path))

    print(f"{len(jobs)} disconnectome files found across {CLASSES}")
    rows = {}
    roi_columns = None
    errors = []
    with Pool(processes=8) as pool:
        for i, (filename, pathology, roi_names, degrees, err) in enumerate(
            pool.imap_unordered(_worker, jobs), 1
        ):
            if err:
                errors.append((filename, err))
                print(f"WARN {filename}: {err}", file=sys.stderr)
                continue
            if roi_columns is None:
                roi_columns = roi_names
            elif roi_names != roi_columns:
                errors.append((filename, "ROI order mismatch"))
                print(f"WARN {filename}: ROI order mismatch, skipped", file=sys.stderr)
                continue
            row = dict(zip(roi_columns, degrees))
            row["pathology"] = pathology
            rows[filename] = row
            if i % 200 == 0:
                print(f"progress: {i}/{len(jobs)}")

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df[["pathology"] + roi_columns].sort_index()
    df.to_csv(OUTPUT, sep="\t", index_label="filename")
    print(f"wrote {OUTPUT} ({len(df)} rows, {len(roi_columns)} ROI columns, {len(errors)} errors)")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
