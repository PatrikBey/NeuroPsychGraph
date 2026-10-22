#!/usr/bin/env python3
"""3D transparent brain-surface mesh with lesion surface meshes embedded.

Brain shell: the fsaverage5 pial cortical surface (both hemispheres),
rendered translucent. Lesions: a rough (coarse step-size) marching-cubes
isosurface of each binary lesion mask, rendered as a faceted dark-purple
mesh (visible triangle edges) in the same approximate world (mm)
coordinate space. Fourth panel: isosurface of the voxel-wise triple
intersection.
"""
import os

import matplotlib
import nibabel as nib
import numpy as np
from nilearn import datasets, surface
from skimage.measure import marching_cubes

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

LESIONS_DIR = "/data/patrik/NPG/DATA/LESIONS"
OUT_PNG = "/data/patrik/NPG/DATA/ANALYSIS/surface_mesh_lesions.png"

SUBJECTS = [
    ("Glioma", "Glioma_00299.nii.gz", "Glioma_00299"),
    ("Ischaemic", "sub-K1093.nii.gz", "sub-K1093 (Stroke)"),
    ("Meningioma", "B_Meningioma_00746.nii.gz", "B_Meningioma_00746"),
]

LESION_COLOR = "#4C1D95"
LESION_EDGE_COLOR = "#2a1052"
BRAIN_COLOR = "#b9b7ae"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"
VIEW = dict(elev=90, azim=-90)  # axial (top-down)
LESION_STEP_SIZE = 2  # coarser marching-cubes mesh -- visible facets, not a smooth blob
LIGHT_DIR = np.array([0.3, -0.4, 1.0])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


def mesh_world(volume, affine, level=0.5, step_size=1):
    verts, faces, _, _ = marching_cubes(volume.astype(float), level=level, step_size=step_size)
    world = nib.affines.apply_affine(affine, verts)
    return world, faces


def shade_facecolors(world, faces, base_hex, alpha):
    base_rgb = np.array(matplotlib.colors.to_rgb(base_hex))
    tris = world[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms
    intensity = np.clip(np.abs(normals @ LIGHT_DIR), 0.35, 1.0)  # abs so either-facing triangle still shades
    rgb = base_rgb[None, :] * intensity[:, None]
    rgb = np.clip(rgb, 0, 1)
    return np.concatenate([rgb, np.full((len(faces), 1), alpha)], axis=1)


def brain_shell_mesh():
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    left = surface.load_surf_mesh(fs["pial_left"])
    right = surface.load_surf_mesh(fs["pial_right"])
    coords = np.vstack([left.coordinates, right.coordinates])
    faces = np.vstack([left.faces, right.faces + len(left.coordinates)])
    return coords, faces


def add_mesh(ax, world, faces, color, alpha, edgecolor="none", linewidth=0.0, shaded=False):
    tris = world[faces]
    if shaded:
        poly = Poly3DCollection(tris, edgecolor=edgecolor, linewidth=linewidth)
        poly.set_facecolor(shade_facecolors(world, faces, color, alpha))
    else:
        poly = Poly3DCollection(tris, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
    ax.add_collection3d(poly)


def set_bounds(ax, world):
    mins, maxs = world.min(axis=0), world.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)
    ax.set_axis_off()
    ax.view_init(**VIEW)


def main():
    brain_world, brain_faces = brain_shell_mesh()

    masks = {}
    ref_affine = None
    for pathology, fname, label in SUBJECTS:
        img = nib.load(os.path.join(LESIONS_DIR, pathology, fname))
        if ref_affine is None:
            ref_affine = img.affine
        masks[label] = img.get_fdata() > 0

    overlap = masks[SUBJECTS[0][2]] & masks[SUBJECTS[1][2]] & masks[SUBJECTS[2][2]]
    print(f"triple-overlap voxels: {overlap.sum()}")

    fig = plt.figure(figsize=(24, 6.5), facecolor=SURFACE)
    axes = [fig.add_subplot(1, 4, i + 1, projection="3d", facecolor=SURFACE) for i in range(4)]

    for ax, (pathology, fname, label) in zip(axes[:3], SUBJECTS):
        add_mesh(ax, brain_world, brain_faces, BRAIN_COLOR, alpha=0.08)
        lesion_world, lesion_faces = mesh_world(masks[label].astype(float), ref_affine, step_size=LESION_STEP_SIZE)
        add_mesh(ax, lesion_world, lesion_faces, LESION_COLOR, alpha=0.8, edgecolor=LESION_EDGE_COLOR, linewidth=0.15, shaded=True)
        set_bounds(ax, brain_world)
        ax.set_title(label, color=INK, fontsize=13, pad=0)

    ax4 = axes[3]
    add_mesh(ax4, brain_world, brain_faces, BRAIN_COLOR, alpha=0.08)
    overlap_world, overlap_faces = mesh_world(overlap.astype(float), ref_affine, step_size=LESION_STEP_SIZE)
    add_mesh(ax4, overlap_world, overlap_faces, LESION_COLOR, alpha=0.8, edgecolor=LESION_EDGE_COLOR, linewidth=0.15, shaded=True)
    set_bounds(ax4, brain_world)
    ax4.set_title(f"Overlap of all three (n={int(overlap.sum())} voxels)", color=INK, fontsize=13, pad=0)

    fig.suptitle(
        "3D transparent brain-surface mesh with lesion surface meshes embedded (dark purple)",
        color=INK, fontsize=15, y=0.98,
    )
    fig.savefig(OUT_PNG, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
