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


def vegetation_color_mask(
    rgb: np.ndarray, exg_threshold: float = 0.12, min_brightness: float = 30.0
) -> np.ndarray:
    """Boolean mask of points whose color looks like vegetation (Excess
    Green Index > threshold). Split out from `filter_by_vegetation_color`
    so callers that need *both* halves of the split (e.g. root detection,
    which uses the non-vegetation/soil points too) don't have to
    recompute it.

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
    return (exg > exg_threshold) & (raw_total > min_brightness)


def filter_by_vegetation_color(
    xyz: np.ndarray, rgb: np.ndarray, exg_threshold: float = 0.12, min_brightness: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only points whose color looks like vegetation, same idea as
    `masking.vegetation_mask` but applied post-hoc to already-triangulated
    3D points instead of to 2D pixels.

    Prefer this over masking *before* SIFT matching when the background is
    cluttered/textured (e.g. a hand holding the plant): masking away
    everything but the plant *before* matching starves COLMAP of the
    texture it needs for camera pose estimation in the first place. Here,
    pose estimation gets the benefit of every textured pixel (fingers
    included), and only the resulting points get filtered by color.
    """
    keep = vegetation_color_mask(rgb, exg_threshold=exg_threshold, min_brightness=min_brightness)
    return xyz[keep], rgb[keep]


def voxel_downsample(
    xyz: np.ndarray, rgb: Optional[np.ndarray] = None, voxel_size: float = 0.01
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Keep one point per occupied voxel of size `voxel_size` -- a standard
    point-cloud thinning step, used by `skeletonize.build_skeleton_graph`
    to bring a densely-sampled cloud (many points spread over each leaf's
    full surface) down to a sparser one before skeletonization: an MST
    over every raw surface point zigzags across that surface and reports
    the zigzags as spurious branches, no matter how the spur-pruning
    threshold is tuned, because at that density the noise and the real
    structure are the same physical size. Thinning removes the surface
    detail an MST-based skeleton was never meant to resolve, instead of
    fighting it with a different pruning threshold per point count.
    """
    if len(xyz) == 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    _, first_index = np.unique(keys, axis=0, return_index=True)
    first_index.sort()
    return xyz[first_index], (rgb[first_index] if rgb is not None else None)


def remove_sparse_points(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    radius_factor: float = 4.0,
    density_fraction: float = 0.30,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Drop points whose local neighbor count is far below the cloud's own
    typical density -- grit, root hairs and stray specks scattered over the
    substrate, which a color threshold cannot reject because they genuinely
    are slightly green.

    This matters more than "a few stray points" suggests, because of *what
    the skeletonizer does with them*. Geodesic distance is measured over a
    k-NN graph, and a thin trail of debris across the substrate is a
    perfectly good path through that graph -- so a leaf whose base is not
    directly connected to the stem gets reached by routing out across the
    ground and back, and the resulting branch crawls along the substrate
    for a third of its length before climbing. Rejecting such branches
    afterwards does not work: the crawl is the *shared prefix* of an
    otherwise real leaf, so its support density averages out to something
    respectable. The trail has to be gone before paths are computed.

    `radius_factor` is in units of the cloud's median nearest-neighbor
    distance, and the threshold is `density_fraction` of the *median*
    neighbor count, so both adapt to how densely a given capture
    reconstructed -- the two test captures differ by a factor of ~2.3 in
    typical neighbor count, and any absolute cutoff that cleaned one
    stripped real leaves off the other.
    """
    if len(xyz) < 4:
        return xyz, rgb

    tree = cKDTree(xyz)
    spacing = float(np.median(tree.query(xyz, k=2)[0][:, 1]))
    if spacing <= 0:
        return xyz, rgb

    counts = np.array([len(n) for n in tree.query_ball_point(xyz, r=radius_factor * spacing)])
    threshold = max(int(round(density_fraction * float(np.median(counts)))), 2)
    keep = counts >= threshold
    if keep.sum() < 3:
        return xyz, rgb

    return xyz[keep], (rgb[keep] if rgb is not None else None)


def find_root_point(
    xyz_plant: np.ndarray, xyz_soil: np.ndarray, max_soil_distance: Optional[float] = None
) -> Optional[int]:
    """Index into `xyz_plant` of the point closest to the growing medium
    (soil/pot) -- i.e. the base of the stem, where the plant "starts".

    Works in any orientation/frame (no up-axis needed): `xyz_soil` is
    whatever 3D points got reconstructed but filtered out as non-vegetation
    (soil, pot, background) -- COLMAP already triangulates them, they're
    just discarded by `filter_by_vegetation_color`. The plant point nearest
    to *any* soil point is, by definition, wherever the plant touches the
    ground it's growing out of.

    `max_soil_distance` restricts candidate soil points to ones actually
    near the plant (default: 3x the plant cloud's own bounding-box
    diagonal), so a far-away wall/table point that happened to survive
    upstream filtering doesn't get treated as "the ground right under this
    plant". Returns None if no soil points are within range (falls back to
    an unrooted skeleton upstream).
    """
    if len(xyz_plant) == 0 or len(xyz_soil) == 0:
        return None

    if max_soil_distance is None:
        plant_extent = np.linalg.norm(xyz_plant.max(axis=0) - xyz_plant.min(axis=0))
        max_soil_distance = max(float(plant_extent) * 3.0, 1e-6)

    soil_tree = cKDTree(xyz_soil)
    dists, _ = soil_tree.query(xyz_plant, k=1)

    if dists.min() > max_soil_distance:
        return None
    return int(np.argmin(dists))


def keep_plant_clusters(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    merge_distance_factor: float = 1.5,
    min_cluster_size: int = 2,
):
    """Keep the largest spatially-connected cluster of points, *plus* any
    other cluster close enough to plausibly be another part of the same
    plant (a separate leaf that just didn't triangulate densely enough to
    stay radius-connected to the rest) -- drops only clusters that are both
    small and far away, which is what real background/mismatch noise looks
    like.

    A plant with several leaves often doesn't reconstruct as one single
    blob: thin, sparsely-textured leaf surfaces can end up as separate
    connected components at a tight radius even though they're all the
    same plant. The previous version of this function (`keep_largest_
    cluster`) kept only the single largest component, which silently
    discarded every other leaf as if it were noise.

    `radius` (default: 4x median nearest-neighbor distance) defines the
    *fine* clustering used to find individual components. `merge_distance_
    factor`, relative to the largest cluster's own bounding-box diagonal,
    then decides which of those components are "close enough to the main
    plant body" to keep -- using the plant's own size as the distance
    scale (not the fine radius, which reflects point *density*, not plant
    *size*, and is usually too tight to bridge a real gap between separate
    leaves).
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
    largest_points = xyz[roots == largest_label]

    if len(labels) == 1:
        return xyz, rgb

    largest_extent = np.linalg.norm(largest_points.max(axis=0) - largest_points.min(axis=0))
    merge_distance = max(float(largest_extent) * merge_distance_factor, radius)

    largest_tree = cKDTree(largest_points)
    keep = roots == largest_label
    for label, count in zip(labels, counts):
        if label == largest_label or count < min_cluster_size:
            continue
        component_points = xyz[roots == label]
        dists, _ = largest_tree.query(component_points, k=1)
        if dists.min() <= merge_distance:
            keep |= roots == label

    return xyz[keep], (rgb[keep] if rgb is not None else None)
