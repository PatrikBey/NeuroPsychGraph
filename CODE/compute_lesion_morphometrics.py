#!/usr/bin/env python3
"""Compute shape/location morphometrics for a directory of binary lesion masks.

Two-tier design:
  - whole-mask tier (volume, surface area, surface:volume ratio, fragment
    count, hemispheric laterality): computed over all foreground voxels,
    well-defined regardless of how many disconnected pieces a lesion has.
  - component tier (sphericity, principal axis lengths, elongation,
    flatness, solidity): computed on the single largest connected
    component only, so irregular-shape measures describe one lesion body
    rather than being confounded by multifocality.

Assumes inputs are binary masks already aligned to a standard NIfTI RAS+
world space (e.g. MNI152), where +x = right hemisphere, used for the
laterality index.
"""
import argparse
import glob
import os
import sys
import traceback
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes, mesh_surface_area

CONNECTIVITY = ndimage.generate_binary_structure(3, 3)  # 26-connected

COLUMNS = [
    "volume_mm3",
    "equiv_diameter_mm",
    "surface_area_mm2",
    "surface_to_volume_ratio",
    "n_components",
    "laterality_index",
    "largest_component_volume_fraction",
    "sphericity",
    "axis1_mm",
    "axis2_mm",
    "axis3_mm",
    "elongation_ratio",
    "flatness_ratio",
    "solidity",
    "error",
]


def _surface_area(mask, spacing):
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing)
    return mesh_surface_area(verts, faces)


def _principal_axis_lengths(world_coords):
    cov = np.cov(world_coords, rowvar=False)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)
    return np.sqrt(eigvals)[::-1]  # descending: major, mid, minor


def _solidity(world_coords, volume_mm3):
    try:
        hull = ConvexHull(world_coords)
        return volume_mm3 / hull.volume if hull.volume > 0 else np.nan
    except Exception:
        return np.nan


def compute_measures(path):
    row = {k: np.nan for k in COLUMNS}
    row["error"] = ""
    try:
        img = nib.load(path)
        mask = np.asarray(img.dataobj) > 0
        zooms = img.header.get_zooms()[:3]
        voxel_volume = float(np.prod(zooms))

        n_voxels = int(mask.sum())
        if n_voxels == 0:
            row["error"] = "empty mask"
            return row

        # --- whole-mask tier ---
        volume_mm3 = n_voxels * voxel_volume
        surface_area_mm2 = _surface_area(mask, zooms)

        labeled, n_components = ndimage.label(mask, structure=CONNECTIVITY)
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, n_components + 1))
        largest_label = int(np.argmax(sizes)) + 1

        all_x = nib.affines.apply_affine(img.affine, np.argwhere(mask))[:, 0]
        n_right, n_left = int((all_x > 0).sum()), int((all_x < 0).sum())
        laterality_index = (
            (n_right - n_left) / (n_right + n_left) if (n_right + n_left) else np.nan
        )

        row.update(
            volume_mm3=volume_mm3,
            equiv_diameter_mm=(6 * volume_mm3 / np.pi) ** (1 / 3),
            surface_area_mm2=surface_area_mm2,
            surface_to_volume_ratio=surface_area_mm2 / volume_mm3,
            n_components=n_components,
            laterality_index=laterality_index,
        )

        # --- component tier: largest connected component only ---
        comp_slice = ndimage.find_objects(labeled)[largest_label - 1]
        comp_mask = labeled[comp_slice] == largest_label
        comp_n_voxels = int(comp_mask.sum())
        comp_volume_mm3 = comp_n_voxels * voxel_volume
        comp_surface_area_mm2 = _surface_area(comp_mask, zooms)

        comp_ijk = np.argwhere(labeled == largest_label)
        comp_world = nib.affines.apply_affine(img.affine, comp_ijk)
        axis1, axis2, axis3 = _principal_axis_lengths(comp_world)

        row.update(
            largest_component_volume_fraction=comp_volume_mm3 / volume_mm3,
            sphericity=(
                (np.pi ** (1 / 3) * (6 * comp_volume_mm3) ** (2 / 3)) / comp_surface_area_mm2
                if comp_surface_area_mm2 > 0
                else np.nan
            ),
            axis1_mm=axis1,
            axis2_mm=axis2,
            axis3_mm=axis3,
            elongation_ratio=axis1 / axis2 if axis2 > 0 else np.nan,
            flatness_ratio=axis2 / axis3 if axis3 > 0 else np.nan,
            solidity=_solidity(comp_world, comp_volume_mm3),
        )
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc(file=sys.stderr)
    return row


def _worker(path):
    return path, compute_measures(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="directory of binary NIfTI lesion masks")
    parser.add_argument(
        "-o", "--output", default=None, help="output CSV path (default: <input_dir_name>_morphometrics.csv in cwd)"
    )
    parser.add_argument("-j", "--jobs", type=int, default=8, help="parallel workers (default: 8)")
    args = parser.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.input_dir, "*.nii.gz")) + glob.glob(os.path.join(args.input_dir, "*.nii"))
    )
    if not files:
        print(f"no .nii/.nii.gz files found in {args.input_dir}", file=sys.stderr)
        return 1

    output = args.output or f"{os.path.basename(os.path.normpath(args.input_dir))}_morphometrics.csv"

    print(f"{len(files)} files found, computing with {args.jobs} worker(s)...")
    rows = {}
    with Pool(processes=args.jobs) as pool:
        for i, (path, row) in enumerate(pool.imap_unordered(_worker, files), 1):
            rows[os.path.basename(path)] = row
            if row["error"]:
                print(f"WARN {path}: {row['error']}", file=sys.stderr)
            if i % 100 == 0:
                print(f"progress: {i}/{len(files)}")

    df = pd.DataFrame.from_dict(rows, orient="index")[COLUMNS]
    df = df.sort_index()
    df.to_csv(output, index_label="filename")
    n_errors = int((df["error"] != "").sum())
    print(f"wrote {output} ({len(df)} rows, {n_errors} with errors)")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
