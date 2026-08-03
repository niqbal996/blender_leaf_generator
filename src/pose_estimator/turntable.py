"""Recover the turntable's geometry -- rotation axis, soil plane, and the
plant's footprint -- from a COLMAP reconstruction, so the skeleton stage
knows which way is up and where the ground is.

Without this, nothing downstream has an up-axis at all: `pointcloud.
find_root_point` used to guess the plant's base as "the vegetation point
closest to any non-vegetation point", which lands wherever soil, pot rim,
table or wall points happen to sit nearest the plant in 3D -- measured at
84% of plant height (i.e. near the top of the canopy) on one of the two
turntable test captures. A turntable capture makes the real answer
recoverable directly: the cameras orbit a fixed vertical axis, the plant
stands on that axis, and the soil is the densest horizontal band of points
around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .alignment import estimate_up_axis, orient_up_axis


@dataclass
class TurntableFrame:
    """The capture rig's geometry, in the raw COLMAP frame.

    Two different surfaces matter here and they are not parallel. `up` is
    the turntable's rotation axis, which is what radial distance is
    measured against. `substrate_normal`/`substrate_point` describe the top
    of the mud clump the plant actually grows out of -- a clump sits on the
    turntable at whatever angle it was set down at, and its top face is an
    irregular, debris-covered surface several centimeters *above* the
    turntable. Heights are measured against that face, because "where does
    the plant leave the soil" is a question about the clump, not the rig.
    """

    up: np.ndarray  # (3,) unit vector, soil -> canopy; the rotation axis
    axis_point: np.ndarray  # (3,) a point on the rotation axis
    substrate_normal: np.ndarray  # (3,) unit normal of the clump's top face
    substrate_point: np.ndarray  # (3,) a point on that face
    radius: float  # plant footprint radius around the axis

    def heights(self, xyz: np.ndarray) -> np.ndarray:
        """Height of each point above the clump's top face (negative = buried)."""
        return (np.asarray(xyz) - self.substrate_point) @ self.substrate_normal

    def radii(self, xyz: np.ndarray) -> np.ndarray:
        """Perpendicular distance of each point from the rotation axis."""
        offsets = np.asarray(xyz) - self.axis_point
        along = offsets @ self.up
        return np.linalg.norm(offsets - np.outer(along, self.up), axis=1)

    @property
    def substrate_tilt_degrees(self) -> float:
        """Angle between the clump's top face and the turntable's plane."""
        cosine = float(np.clip(abs(self.substrate_normal @ self.up), 0.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))


def solve_axis_point(camera_centers: np.ndarray, viewing_dirs: np.ndarray) -> np.ndarray:
    """The point minimizing squared distance to every camera's viewing ray --
    i.e. where the rig is aimed, which for a turntable capture is a point on
    the rotation axis, at roughly the subject's mid-height.

    The camera *centroid* is not a usable substitute: it lies on the
    rotation axis only if the orbit is complete and evenly sampled, which a
    hand-triggered capture (or one where some images fail to register)
    isn't. Convergence of the viewing rays is what actually pins the axis
    down laterally.
    """
    centers = np.asarray(camera_centers, dtype=np.float64)
    dirs = np.asarray(viewing_dirs, dtype=np.float64)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    # sum_i (I - d_i d_i^T) p = sum_i (I - d_i d_i^T) c_i
    lhs = len(dirs) * np.eye(3) - dirs.T @ dirs
    rhs = (centers - np.sum(centers * dirs, axis=1)[:, None] * dirs).sum(axis=0)
    return np.linalg.solve(lhs, rhs)


def find_ground_height(
    xyz: np.ndarray,
    up: np.ndarray,
    axis_point: np.ndarray,
    radius: float,
    num_bins: int = 50,
) -> float:
    """Height (along `up`, relative to `axis_point`) of the densest
    horizontal band of points within `radius` of the rotation axis -- the
    soil surface / pot rim.

    Everything in the capture is reconstructed sparsely except the soil,
    which is a broad, flat, heavily-textured horizontal disc facing every
    camera in the orbit -- so it dominates the height histogram. Restricting
    to a cylinder around the axis first is what keeps a textured wall or
    table edge elsewhere in the scene from outvoting it.
    """
    offsets = np.asarray(xyz) - axis_point
    along = offsets @ up
    perpendicular = np.linalg.norm(offsets - np.outer(along, up), axis=1)

    near_axis = along[perpendicular < radius]
    if len(near_axis) < num_bins:
        near_axis = along
    if len(near_axis) == 0:
        return 0.0

    counts, edges = np.histogram(near_axis, bins=num_bins)
    peak = int(np.argmax(counts))
    return float((edges[peak] + edges[peak + 1]) / 2)


def _axis_basis(up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to `up`."""
    seed = np.array([1.0, 0.0, 0.0])
    if abs(seed @ up) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = np.cross(up, seed)
    first /= np.linalg.norm(first)
    return first, np.cross(up, first)


def fit_substrate_plane(
    xyz: np.ndarray,
    is_plant: np.ndarray,
    up: np.ndarray,
    axis_point: np.ndarray,
    radius: float,
    cells_across: int = 25,
    band_fraction: float = 0.12,
    iterations: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the plane of the mud clump's *top face* -- the surface the plant
    emerges from -- returning its (normal, point).

    Fitting the top face means fitting an upper envelope, not a point
    cloud: the non-plant points near the axis include the clump's top, its
    crumbling sides, the paper under it and the turntable itself, all
    stacked vertically. So the points are binned into cells across the
    turntable plane and only the highest point in each cell is kept, which
    reduces that stack to a single height field; the clump top is then the
    elevated plateau in it.

    The plane's *normal* is fitted, not assumed parallel to the turntable,
    because a clump is set down at whatever angle it lands at -- measured
    ~10 degrees of tilt on one of the two test captures and under 1 on the
    other. But an unconstrained plane fit over the raw soil points does not
    recover it: tried that way, RANSAC locked onto the clump's near-vertical
    *side* wall and reported a normal 18 degrees off with only 31% inliers.
    Restricting to the upper envelope is what makes the top face the
    dominant structure rather than one surface among several.

    Iteratively reweighted least squares (Cauchy weights) handles the
    debris -- bark chips and perlite stand proud of the surface by a
    centimeter or two and would otherwise tilt a plain least-squares fit.
    """
    plant_free = ~np.asarray(is_plant)
    offsets = np.asarray(xyz) - axis_point
    along = offsets @ up
    perpendicular = np.linalg.norm(offsets - np.outer(along, up), axis=1)

    candidates = plant_free & (perpendicular < radius * 1.5)
    if candidates.sum() < 12:
        return np.asarray(up, dtype=np.float64), np.asarray(axis_point, dtype=np.float64)

    points = np.asarray(xyz)[candidates]
    heights = along[candidates]
    radial = perpendicular[candidates]

    first, second = _axis_basis(up)
    local = points - axis_point
    cell = max(radius / cells_across, 1e-9)
    keys = np.stack(
        [np.floor((local @ first) / cell), np.floor((local @ second) / cell)], axis=1
    ).astype(np.int64)

    # Highest point per cell == the upper envelope.
    order = np.lexsort((-heights, keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    is_first = np.ones(len(order), dtype=bool)
    is_first[1:] = (sorted_keys[1:] != sorted_keys[:-1]).any(axis=1)
    envelope = order[is_first]

    envelope_heights = heights[envelope]
    near_axis = radial[envelope] < radius * 0.8
    if near_axis.sum() < 12:
        near_axis = np.ones(len(envelope), dtype=bool)

    counts, edges = np.histogram(envelope_heights[near_axis], bins=40)
    peak = int(np.argmax(counts))
    plateau = float((edges[peak] + edges[peak + 1]) / 2)

    band = max(band_fraction * radius, 1e-9)
    on_top = (np.abs(envelope_heights - plateau) < band) & (radial[envelope] < radius * 0.9)
    if on_top.sum() < 12:
        on_top = np.abs(envelope_heights - plateau) < band
    if on_top.sum() < 12:
        return np.asarray(up, dtype=np.float64), points[envelope][np.argmax(envelope_heights)]

    top = points[envelope][on_top]

    weights = np.ones(len(top))
    normal = np.asarray(up, dtype=np.float64)
    centroid = top.mean(axis=0)
    for _ in range(iterations):
        centroid = (weights[:, None] * top).sum(axis=0) / weights.sum()
        centered = top - centroid
        _, _, right = np.linalg.svd((centered * weights[:, None]).T @ centered)
        normal = right[2]
        if normal @ up < 0:
            normal = -normal
        residuals = np.abs(centered @ normal)
        scale = 1.4826 * np.median(residuals) + 1e-12
        weights = 1.0 / (1.0 + (residuals / scale) ** 2)

    return normal, centroid


def solve_turntable_frame(
    xyz: np.ndarray,
    camera_centers: np.ndarray,
    viewing_dirs: np.ndarray,
    is_plant: Optional[np.ndarray] = None,
    radius_percentile: float = 95.0,
    radius_margin: float = 1.15,
) -> TurntableFrame:
    """Recover the full rig geometry from the reconstruction.

    `up` reuses the same camera-orbit-plane-normal estimate that
    `pose-align-skeleton` already solves for (see `alignment.
    estimate_up_axis`/`orient_up_axis`), so the skeleton is built in the
    same orientation it will later be aligned into, rather than in an
    orientation-free frame that has to be reconciled afterwards.

    `is_plant` is a boolean mask of points to exclude when fitting the
    clump's top face (the vegetation-colored ones -- leaves are not part of
    the surface the plant grows out of). Without it every point is treated
    as potential substrate, which is usually still fine since the clump
    outnumbers the seedling heavily.

    `radius` is set from how far the points near the axis actually spread,
    not from a fixed constant, since the COLMAP frame is unitless -- a
    literal radius would mean a different physical size on every capture.
    """
    up = orient_up_axis(estimate_up_axis(camera_centers), viewing_dirs)
    axis_point = solve_axis_point(camera_centers, viewing_dirs)

    offsets = np.asarray(xyz) - axis_point
    along = offsets @ up
    perpendicular = np.linalg.norm(offsets - np.outer(along, up), axis=1)

    # Seeded from the bulk of the reconstruction (the subject dominates it;
    # far-off background points are the tail), then used to re-measure the
    # footprint with a cylinder that actually matches the subject.
    seed_radius = float(np.percentile(perpendicular, radius_percentile))
    seed_ground = find_ground_height(xyz, up, axis_point, seed_radius)

    above_ground = along > seed_ground
    if above_ground.sum() >= 10:
        radius = float(np.percentile(perpendicular[above_ground], radius_percentile))
    else:
        radius = seed_radius
    radius = max(radius * radius_margin, 1e-6)

    if is_plant is None:
        is_plant = np.zeros(len(xyz), dtype=bool)
    substrate_normal, substrate_point = fit_substrate_plane(
        xyz, is_plant, up, axis_point, radius
    )

    return TurntableFrame(
        up=up,
        axis_point=axis_point,
        substrate_normal=substrate_normal,
        substrate_point=substrate_point,
        radius=radius,
    )


def crop_to_plant(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray],
    frame: TurntableFrame,
    min_height: float = 0.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Keep only points standing above the soil, inside the turntable's
    footprint cylinder.

    This is a *geometric* plant/background split, and it does the job the
    Excess-Green color threshold was being asked to do alone and failing at:
    on the two turntable captures, roughly half the points passing ExG sat
    on the wrong side of the soil plane -- background foliage, green-cast
    highlights, and dark-pixel ratio noise. Those survive color filtering
    and statistical outlier removal (there are too many of them to look
    like outliers), and they are what the skeleton was being built through.
    """
    heights = frame.heights(xyz)
    radii = frame.radii(xyz)
    keep = (heights > min_height) & (radii < frame.radius)
    return np.asarray(xyz)[keep], (np.asarray(rgb)[keep] if rgb is not None else None)


def find_root_point_on_ground(
    xyz_plant: np.ndarray,
    frame: TurntableFrame,
    quantile: float = 0.05,
    min_support: int = 6,
    support_radius_factor: float = 4.0,
) -> Optional[int]:
    """Index into `xyz_plant` of where the plant emerges from the clump:
    among the lowest `quantile` of well-supported points, the one nearest
    the plant's own vertical centerline.

    Three conditions, each earning its place:

    *Low*, measured above the clump's top face rather than the turntable.
    A clump is several centimeters tall with crumbling sides, so height
    above the *rig* counts mud as plant -- on one test capture the clump top
    sat 0.109 above the old turntable-relative zero while the supposed plant
    cloud started at 0.054, meaning the bottom of it was all substrate.

    *Well-supported*: at least `min_support` neighbors within
    `support_radius_factor` times the cloud's median nearest-neighbor
    spacing. Loose grit and root hairs on the clump surface reconstruct as
    isolated specks that pass a vegetation-color test, and being both low
    and near the axis they are otherwise ideal root candidates -- the
    speck previously chosen on that capture had 1 neighbor where the cloud's
    median point had 8. Real stem base sits in dense cloud.

    *Central*, against the vertical line through the plant cloud's own
    horizontal centroid rather than the rig's rotation axis. The plant is
    never perfectly centered on the turntable and `solve_axis_point` finds
    where the cameras are *aimed* (roughly canopy mid-height), so the rig
    axis sat ~0.27 units off the cloud center against a plant radius of
    ~0.2 -- enough to pick a canopy-edge point over the stem.
    """
    if len(xyz_plant) == 0:
        return None

    heights = frame.heights(xyz_plant)

    supported = np.ones(len(xyz_plant), dtype=bool)
    if len(xyz_plant) > min_support:
        tree = cKDTree(xyz_plant)
        spacing = float(np.median(tree.query(xyz_plant, k=2)[0][:, 1]))
        if spacing > 0:
            counts = np.array(
                [len(n) for n in tree.query_ball_point(xyz_plant, r=support_radius_factor * spacing)]
            )
            if (counts >= min_support).sum() >= 3:
                supported = counts >= min_support

    candidates = np.nonzero(supported)[0]
    cutoff = float(np.quantile(heights[candidates], quantile))
    lowest = candidates[heights[candidates] <= cutoff]
    if len(lowest) == 0:
        return None

    offsets = xyz_plant[lowest] - np.asarray(xyz_plant).mean(axis=0)
    along = offsets @ frame.up
    radii = np.linalg.norm(offsets - np.outer(along, frame.up), axis=1)
    return int(lowest[int(np.argmin(radii))])
