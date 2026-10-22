#!/usr/bin/env python3
"""Resample synthetic lesion masks to 1mm atlas space, same method as the real masks.

Selects the first N_PER_CLASS files (sorted by filename) per class,
nearest-neighbor resamples each to the atlas grid, and verifies the
result is still a strict binary mask.
"""
import glob
import os
import sys
import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

SRC_ROOT = "/data/patrik/NPG/DATA/LESIONS/SYNTH"
DST_ROOT = "/data/patrik/NPG/DATA/LESIONS/SYNTH_1mm"
ATLAS_PATH = "/data/patrik/NPG/DATA/ATLAS/Schaefer2018-100.nii.gz"
N_PER_CLASS = 1000
CLASSES = ["Ischaemic", "Meningioma"]

atlas_img = None


def init_worker():
    global atlas_img
    atlas_img = nib.load(ATLAS_PATH)


def process_one(args):
    src_path, dst_path = args
    try:
        src = nib.load(src_path)
        resampled = resample_to_img(
            src, atlas_img, interpolation="nearest", force_resample=True, copy_header=True
        )
        data = resampled.get_fdata()
        rounded = np.rint(data)
        max_err = np.max(np.abs(data - rounded)) if data.size else 0.0
        uniq = set(rounded.tolist()) if rounded.size < 10 else set(np.unique(rounded).tolist())
        if max_err > 1e-3 or not uniq <= {0.0, 1.0}:
            return (src_path, False, f"non-binary values after resample: max_err={max_err}, uniq={uniq}")
        out = nib.Nifti1Image(rounded.astype(np.uint8), resampled.affine, resampled.header)
        out.header.set_data_dtype(np.uint8)
        nib.save(out, dst_path)
        return (src_path, True, None)
    except Exception as e:
        return (src_path, False, str(e))


def main():
    os.makedirs(DST_ROOT, exist_ok=True)
    jobs = []
    for cls in CLASSES:
        src_dir = os.path.join(SRC_ROOT, cls, "nifti")
        dst_dir = os.path.join(DST_ROOT, cls)
        os.makedirs(dst_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(src_dir, "*.nii.gz")))[:N_PER_CLASS]
        print(f"{cls}: selected {len(files)} of {len(glob.glob(os.path.join(src_dir, '*.nii*')))} files")
        for f in files:
            dst_path = os.path.join(dst_dir, os.path.basename(f))
            jobs.append((f, dst_path))

    print(f"total jobs: {len(jobs)}")
    t0 = time.time()
    failures = []
    done = 0
    with Pool(processes=24, initializer=init_worker) as pool:
        for src_path, ok, err in pool.imap_unordered(process_one, jobs, chunksize=4):
            done += 1
            if not ok:
                failures.append((src_path, err))
                print(f"FAIL {src_path}: {err}", flush=True)
            if done % 200 == 0:
                elapsed = time.time() - t0
                print(f"progress: {done}/{len(jobs)} ({elapsed:.1f}s elapsed)", flush=True)

    elapsed = time.time() - t0
    print(f"done: {done}/{len(jobs)} in {elapsed:.1f}s, failures: {len(failures)}")
    if failures:
        print("FAILURES:")
        for f, e in failures:
            print(f"  {f}: {e}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
