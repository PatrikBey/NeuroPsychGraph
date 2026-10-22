#!/usr/bin/env python3
"""Summarize disconnectome node-weight vectors into one TSV table.

Each disconnectome file is a 100x100 weighted (ROI x ROI) connectivity
matrix. Node weight for each ROI is the column sum (the matrix is
symmetric, so this equals the row sum / weighted degree). One row per
disconnectome file, one column per ROI weight, plus a `pathology` label
column.
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

DISCO_DIR = "/data/patrik/NPG/DATA/DISCONNECTOMES"
CLASSES = ["Glioma", "Ischaemic", "Meningioma"]
OUTPUT = os.path.join(DISCO_DIR, "node_degree_summary.tsv")


def parse_weights(path):
    df = pd.read_csv(path, sep="\t", header=0, index_col=0)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    mat = df.values.astype(float)
    return df.columns.tolist(), mat.sum(axis=0)


def _worker(args):
    pathology, path = args
    filename = os.path.basename(path)
    try:
        roi_names, weights = parse_weights(path)
        return filename, pathology, roi_names, weights, None
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
        for i, (filename, pathology, roi_names, weights, err) in enumerate(
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
            row = dict(zip(roi_columns, weights))
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
