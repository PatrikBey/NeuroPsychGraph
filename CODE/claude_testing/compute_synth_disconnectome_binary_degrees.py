#!/usr/bin/env python3
"""Node-degree summary table for synthetic disconnectomes, binarized at the
50th percentile of each subject's nonzero edges (same method as the real data)."""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
CLASSES = [("SYNTH_Ischaemic", "Ischaemic"), ("SYNTH_Meningioma", "Meningioma")]
OUTPUT = os.path.join(DISCO_DIR, "node_degree_binary_summary_synth.tsv")
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
    for folder, pathology in CLASSES:
        for path in sorted(glob.glob(os.path.join(DISCO_DIR, folder, "*.tsv"))):
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
