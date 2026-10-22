#!/usr/bin/env python3
"""Glass-brain views of the three centroid-matched lesion masks, plus a
fourth panel showing their triple intersection (voxels inside all three)."""
import os

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import plotting

LESIONS_DIR = "/data/patrik/NPG/DATA/LESIONS"
OUT_PNG = "/data/patrik/NPG/DATA/ANALYSIS/glass_brain_lesions.png"

SUBJECTS = [
    ("Glioma", "Glioma_00299.nii.gz", "Glioma_00299", "#2a78d6"),
    ("Ischaemic", "sub-K1093.nii.gz", "sub-K1093 (Stroke)", "#eb6834"),
    ("Meningioma", "B_Meningioma_00746.nii.gz", "B_Meningioma_00746", "#DC143C"),
]

INK = "#0b0b0b"
SURFACE = "#fcfcfb"
OVERLAP_COLOR = "#4C1D95"


def solid_cmap(hex_color):
    return matplotlib.colors.ListedColormap([hex_color])


def main():
    imgs = {}
    for pathology, fname, label, color in SUBJECTS:
        img = nib.load(os.path.join(LESIONS_DIR, pathology, fname))
        imgs[label] = (img, color)

    masks = [nib.load(os.path.join(LESIONS_DIR, p, f)).get_fdata() > 0 for p, f, _, _ in SUBJECTS]
    ref_img = nib.load(os.path.join(LESIONS_DIR, SUBJECTS[0][0], SUBJECTS[0][1]))
    overlap = masks[0] & masks[1] & masks[2]
    overlap_img = nib.Nifti1Image(overlap.astype(np.uint8), ref_img.affine, ref_img.header)
    print(f"triple-overlap voxels: {overlap.sum()}")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5), facecolor=SURFACE)

    for ax, (pathology, fname, label, color) in zip(axes[:3], SUBJECTS):
        img, _ = imgs[label]
        plotting.plot_glass_brain(
            img, display_mode="ortho", colorbar=False, cmap=solid_cmap(color),
            threshold=0.5, alpha=0.9, plot_abs=False, axes=ax, figure=fig,
            title=label, black_bg=False,
        )

    plotting.plot_glass_brain(
        overlap_img, display_mode="ortho", colorbar=False, cmap=solid_cmap(OVERLAP_COLOR),
        threshold=0.5, alpha=0.9, plot_abs=False, axes=axes[3], figure=fig,
        title=f"Overlap of all three (n={int(overlap.sum())} voxels)", black_bg=False,
    )

    fig.suptitle(
        "Glass-brain view of the three centroid-matched lesions, and their voxel-wise triple overlap",
        color=INK, fontsize=14, y=1.03,
    )
    fig.savefig(OUT_PNG, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
