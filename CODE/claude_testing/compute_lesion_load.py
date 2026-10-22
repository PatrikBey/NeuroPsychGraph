#!/usr/bin/env python3
"""Lesion-load maps: for each lesion mask, overlap with the Schaefer-100
atlas and compute, per ROI, the percentage of that ROI's voxels covered
by the lesion. One row per subject, 100 ROI columns -- directly
comparable in structure to the disconnectome node-weight tables, but
purely spatial (no tractography), for use as the "lesion mask only"
baseline against the disconnectome-derived co-occurrence analysis.
"""
import glob
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd

LESIONS_DIR = "/data/patrik/NPG/DATA/LESIONS"
ATLAS_PATH = "/data/patrik/NPG/DATA/ATLAS/Schaefer2018-100.nii.gz"
ROI_NAMES_PATH = "/data/patrik/NPG/DATA/ATLAS/Schaefer2018-100_ROIs.txt"
OUT_DIR = "/data/patrik/NPG/DATA/ANALYSIS"
CLASSES = ["Glioma", "Ischaemic", "Meningioma"]

atlas_data = None
roi_voxel_counts = None
roi_names = None


def init_worker():
    global atlas_data, roi_voxel_counts, roi_names
    atlas_data = nib.load(ATLAS_PATH).get_fdata().astype(np.int16)
    roi_voxel_counts = np.bincount(atlas_data.ravel(), minlength=101)[1:101].astype(float)
    with open(ROI_NAMES_PATH) as f:
        roi_names = [line.strip() for line in f if line.strip()]


def process_one(path):
    mask = nib.load(path).get_fdata() > 0
    overlap_counts = np.bincount(atlas_data[mask], minlength=101)[1:101].astype(float)
    load_pct = np.divide(overlap_counts, roi_voxel_counts, out=np.zeros(100), where=roi_voxel_counts > 0) * 100
    return os.path.basename(path), load_pct


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    init_worker()  # populate roi_names in the main process for the header

    for cls in CLASSES:
        files = sorted(
            f for f in glob.glob(os.path.join(LESIONS_DIR, cls, "*.nii.gz"))
            if not os.path.basename(f).startswith("Schaefer2018-100")
        )
        print(f"{cls}: {len(files)} files")

        rows = {}
        with Pool(processes=8, initializer=init_worker) as pool:
            for i, (filename, load_pct) in enumerate(pool.imap_unordered(process_one, files, chunksize=8), 1):
                rows[filename] = load_pct
                if i % 200 == 0:
                    print(f"  progress: {i}/{len(files)}")

        df = pd.DataFrame.from_dict(rows, orient="index", columns=roi_names).sort_index()
        out = os.path.join(OUT_DIR, f"lesion_load_{cls}.tsv")
        df.to_csv(out, sep="\t", index_label="filename")
        print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
