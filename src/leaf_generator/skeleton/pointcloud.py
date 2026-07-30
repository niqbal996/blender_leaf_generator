"""Point cloud extraction and cleanup (pure numpy/scipy, no pycolmap needed
here -- `extract_xyz_rgb` just reads attributes off an already-built
pycolmap.Reconstruction object).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def extract_xyz_rgb(reconstruction) -> Tuple[np.ndarray, np.ndarray]:
    """Pull (xyz, rgb) arrays out of a pycolmap.Reconstruction's points3D."""
    xyz = np.array([p.xyz for p in reconstruction.points3D.values()], dtype=np.float64)
    rgb = np.array([p.color for p in reconstruction.points3D.values()], dtype=np.uint8)
    return xyz, rgb


def remove_statistical_outliers(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    k: int = 12,
    std_ratio: float = 2.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Drop points whose mean distance to their k nearest neighbors is more
    than `std_ratio` standard deviations above the dataset's mean -- a
    standard point-cloud denoising step (isolated mismatched SfM points
    tend to sit far from everything else).
    """
    if len(xyz) <= k:
        return xyz, rgb

    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1)  # includes the point itself at k=0
    mean_neighbor_dist = dists[:, 1:].mean(axis=1)

    global_mean = mean_neighbor_dist.mean()
    global_std = mean_neighbor_dist.std()
    keep = mean_neighbor_dist <= global_mean + std_ratio * global_std

    return xyz[keep], (rgb[keep] if rgb is not None else None)


def filter_by_vegetation_color(
    xyz: np.ndarray, rgb: np.ndarray, exg_threshold: float = 0.12, min_brightness: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only points whose color looks like vegetation (Excess Green
    Index > threshold), same idea as `masking.vegetation_mask` but applied
    post-hoc to already-triangulated 3D points instead of to 2D pixels.

    Prefer this over masking *before* SIFT matching when the background is
    cluttered/textured (e.g. a hand holding the plant): masking away
    everything but the plant *before* matching starves COLMAP of the
    texture it needs for camera pose estimation in the first place. Here,
    pose estimation gets the benefit of every textured pixel (fingers
    included), and only the resulting points get filtered by color.

    `min_brightness` (sum of R+G+B, out of 765) guards against near-black
    pixels (e.g. dark soil): ExG is a *ratio*, so for a near-black color
    like (1, 2, 0) a single count of sensor/JPEG noise swings the ratio
    wildly, letting soil-colored points randomly pass the threshold. Real
    vegetation is never this dark, so points below `min_brightness` are
    dropped before the ratio is even trusted.
    """
    r, g, b = rgb[:, 0].astype(np.float32), rgb[:, 1].astype(np.float32), rgb[:, 2].astype(np.float32)
    raw_total = r + g + b
    total = raw_total + 1e-6
    exg = 2 * (g / total) - (r / total) - (b / total)
    keep = (exg > exg_threshold) & (raw_total > min_brightness)
    return xyz[keep], rgb[keep]


def keep_largest_cluster(xyz: np.ndarray, rgb: Optional[np.ndarray] = None, radius: Optional[float] = None):
    """Keep only the largest spatially-connected cluster of points (radius
    graph connected components) -- drops small floating groups of
    mismatched points that survive outlier removal.
    """
    if len(xyz) == 0:
        return xyz, rgb

    if radius is None:
        tree = cKDTree(xyz)
        nn_dists, _ = tree.query(xyz, k=2)
        radius = float(np.median(nn_dists[:, 1]) * 4.0)

    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")

    n = len(xyz)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in pairs:
        union(i, j)

    roots = np.array([find(i) for i in range(n)])
    labels, counts = np.unique(roots, return_counts=True)
    largest_label = labels[np.argmax(counts)]
    keep = roots == largest_label

    return xyz[keep], (rgb[keep] if rgb is not None else None)
