"""P4c stage 2 -- carry the 2D class maps back onto the 3D points, by voting.

Stage 1 (`classify2d.py`) decided what each *pixel* is. This stage decides what
each *point* is, by rendering the cloud from every solved camera and letting
the pixels vote for the points that produced them.

From Jia et al., *The Turning Point of 3D Plant Phenotyping* (arXiv 2607.01753),
section 2.5. Only the labelling idea is adopted; the paper's erosion-based
organ *separation* is deliberately not, because erosion destroys exactly the
small leaves this stage exists to recover.

The problem it solves: a small apex or basal leaf has too few 3D points to be
distinguishable from noise by any method that decides leaf-versus-stem from 3D
geometry alone. In a photograph that same leaf is still obviously a leaf. So
the decision is made in 2D, where the evidence is, and carried back to 3D here.

Three design points, all load-bearing:

- **An explicit pixel-to-point index map.** Rendering the point cloud directly
  means each pixel knows precisely which point produced it, so back-projection
  is exact. The alternative -- rendering a surface and finding the nearest
  cloud point afterwards -- reintroduces an association error precisely in the
  sparse regions where small leaves live.

- **Uncertainty is carried forward, not thresholded away.** Every point keeps
  its vote counts and how many views actually saw it. A point seen twice is
  weaker evidence than one seen thirty times, and P5 needs to know the
  difference. Points no view saw stay unlabeled rather than being filled in.

- **Views are weighted by obliquity, not required to agree.** A 2D segmenter
  recognises a leaf when its blade faces the camera and loses it when the same
  leaf turns edge-on. Over a 360 degree orbit that is guaranteed to happen to
  every leaf, so any scheme needing all views to concur is asking for
  something the capture cannot provide. Each view's vote is instead scaled by
  the cosine between the point's surface normal and the ray to that camera --
  the foreshortening factor, i.e. how much of that surface the camera actually
  sees. Broad-side views carry full weight, grazing views approach zero and
  abstain on their own. See `view_weights`.

Naming: the plan calls this stage "P4b", but that name is already taken in this
pipeline by the 2DGS surface stage. Since the plan slots this between dense
geometry and structure, and the surfel stage *is* dense geometry, it lands here
as P4c.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

UNLABELED = -1
LABEL_NONE = UNLABELED  # retained name; the tests and P5 both read it


@dataclass
class ViewCamera:
    """A calibrated view the cloud can be rendered from.

    In practice always a real solved COLMAP view -- see `camera_from_colmap`.
    Kept as a plain dataclass rather than taking a pycolmap object directly so
    the renderer and the vote weighting stay testable without a reconstruction.
    """

    K: np.ndarray
    world_to_camera: np.ndarray
    width: int
    height: int


def _look_at(eye: np.ndarray, target: np.ndarray, up_hint=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World-to-camera matrix for a camera at `eye` looking at `target`."""
    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-12)
    up_hint = np.asarray(up_hint, dtype=float)
    if abs(float(forward @ up_hint)) > 0.98:  # looking straight up/down
        up_hint = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up_hint)
    right = right / max(np.linalg.norm(right), 1e-12)
    down = np.cross(forward, right)

    rotation = np.stack([right, down, forward])
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = -rotation @ eye
    return world_to_camera


def render_points(
    points: np.ndarray,
    colors: np.ndarray,
    camera: ViewCamera,
    splat_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Z-buffered point-cloud render. Returns (rgb uint8, index map int32).

    The index map is the whole point: entry (y, x) holds the index of the cloud
    point that produced that pixel, or -1. Votes travel back along it exactly,
    with no nearest-neighbour guesswork.

    Implemented as a painter's algorithm -- points sorted far-to-near and
    written in that order, so nearer points overwrite farther ones. That is a
    correct depth test for opaque splats and avoids a scatter-min pass followed
    by a second lookup to recover which point won.
    """
    R = camera.world_to_camera[:3, :3]
    t = camera.world_to_camera[:3, 3]
    cam = points @ R.T + t
    depth = cam[:, 2]

    in_front = depth > 1e-6
    safe = np.where(in_front, depth, 1.0)
    pixels = (cam[:, :2] / safe[:, None]) @ camera.K[:2, :2].T + camera.K[:2, 2]

    x = np.round(pixels[:, 0]).astype(np.int64)
    y = np.round(pixels[:, 1]).astype(np.int64)
    visible = in_front & (x >= -splat_radius) & (x < camera.width + splat_radius) \
        & (y >= -splat_radius) & (y < camera.height + splat_radius)

    order = np.argsort(-depth[visible])  # far first
    idx = np.nonzero(visible)[0][order]
    xs, ys = x[idx], y[idx]

    index_map = np.full((camera.height, camera.width), -1, np.int32)
    rgb = np.zeros((camera.height, camera.width, 3), np.uint8)

    for dy in range(-splat_radius, splat_radius + 1):
        for dx in range(-splat_radius, splat_radius + 1):
            if dx * dx + dy * dy > splat_radius * splat_radius:
                continue
            px, py = xs + dx, ys + dy
            keep = (px >= 0) & (px < camera.width) & (py >= 0) & (py < camera.height)
            index_map[py[keep], px[keep]] = idx[keep]
            rgb[py[keep], px[keep]] = colors[idx[keep]]

    return rgb, index_map


def camera_from_colmap(image, camera) -> ViewCamera:
    """Wrap a solved COLMAP view so the same renderer can produce its index map."""
    world_to_camera = np.eye(4)
    world_to_camera[:3, :] = image.cam_from_world().matrix()
    return ViewCamera(
        K=np.asarray(camera.calibration_matrix(), dtype=np.float64),
        world_to_camera=world_to_camera,
        width=int(camera.width),
        height=int(camera.height),
    )


# --------------------------------------------------------------------------
# Voting
# --------------------------------------------------------------------------


@dataclass
class VoteResult:
    labels: np.ndarray        # int8 (N,), -1 where no view saw the point
    n_views_seen: np.ndarray  # int32 (N,) labelled *pixel* observations, not distinct views:
                              # a splat covers several pixels, uniformly per view, so ratios
                              # against it stay meaningful while the raw number does not
    weight: np.ndarray        # float32 (N, C) normal-weighted votes
    count: np.ndarray         # int32   (N, C) raw unweighted votes
    confidence: np.ndarray    # float32 (N,) winning weight / total weight
    mean_cosine: np.ndarray   # float32 (N,) average obliquity of the views that saw it
    unweighted_labels: np.ndarray  # int8 (N,) what plain vote counting would have said


def estimate_normals(points: np.ndarray, k: int = 24) -> np.ndarray:
    """Per-point normals by local PCA -- the eigenvector of least variance.

    Only needed when the cloud carries none. P4b's carved surface already
    stores `nx, ny, nz`, and those are better: they come from surfels fitted
    against the photographs rather than from the sampled geometry.

    Sign is left arbitrary on purpose; see `view_weights`.
    """
    from scipy.spatial import cKDTree

    k = min(k, len(points))
    _, neighbours = cKDTree(points).query(points, k=k)
    patches = points[neighbours]
    patches = patches - patches.mean(axis=1, keepdims=True)
    # Batched covariance eigen-decomposition; smallest eigenvector is the normal.
    cov = np.einsum("nki,nkj->nij", patches, patches) / max(k - 1, 1)
    return np.linalg.eigh(cov)[1][:, :, 0].astype(np.float32)


def view_weights(points: np.ndarray, normals: np.ndarray, camera: ViewCamera) -> np.ndarray:
    """How broad-side each point's surface is to this camera, in [0, 1].

    This is the fix for the observation that a 2D segmenter only recognises a
    leaf when its blade faces the camera. Over a 360 degree orbit every leaf is
    broad-side to some views and edge-on to others, so demanding that every
    view agree is the wrong requirement -- the edge-on views are not wrong
    about the leaf, they simply cannot see it as one. Weighting by obliquity
    lets them **abstain** rather than outvote the views that can.

    The weight is the cosine between the point's normal and the ray to the
    camera, which is not a tuning constant: it is the foreshortening factor,
    the fraction of the point's surface area that this camera actually sees. A
    blade square to the camera contributes its full area; the same blade turned
    edge-on projects to a sliver and contributes almost nothing. No threshold
    and no exponent, because the physical quantity already has the right shape.

    The ray to the camera is used rather than the camera's optical axis. They
    agree only at the principal point, and a leaf at the edge of a wide frame
    is exactly where the difference bites.

    **The sign is discarded.** A leaf is a two-sided sheet and which face is
    turned toward the camera flips halfway round the orbit, so a signed cosine
    would reward one half of the capture and punish the other for the same
    geometry. Absolute value asks the only question that matters -- broad-side
    or edge-on -- and sidesteps normal orientation entirely. Self-occlusion is
    already handled: the z-buffer means a point that reaches the index map is
    the one facing the camera.
    """
    rotation = camera.world_to_camera[:3, :3]
    translation = camera.world_to_camera[:3, 3]
    centre = -rotation.T @ translation

    rays = centre[None, :] - points
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    unit = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return np.abs(np.einsum("ij,ij->i", unit, rays)).astype(np.float32)


def accumulate_votes(n_points: int, n_classes: int = 3) -> Dict[str, np.ndarray]:
    """Empty tally: weighted votes, plus raw counts kept alongside.

    The raw counts are not used to decide anything. They are carried so a run
    can report how many points the weighting actually moved, which is the only
    honest way to show the fusion is doing something rather than being
    decoration.
    """
    return {
        "weight": np.zeros((n_points, n_classes), np.float64),
        "count": np.zeros((n_points, n_classes), np.int32),
        "n_views_seen": np.zeros(n_points, np.int32),
        "cosine_sum": np.zeros(n_points, np.float64),
    }


def cast_votes(
    tally: Dict[str, np.ndarray],
    index_map: np.ndarray,
    class_map: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> None:
    """Credit one weighted vote per labelled pixel to the point that produced it.

    `weights` is per-point for this view (see `view_weights`); omitting it
    falls back to plain counting. A point is counted as *seen* only where the
    render placed it and the segmenter produced a class -- counting a point as
    seen but unlabelled would silently depress its confidence.
    """
    valid = index_map >= 0
    indices = index_map[valid]
    classes = class_map[valid].astype(np.int64)

    labelled = classes != LABEL_NONE
    indices, classes = indices[labelled], classes[labelled]
    if not len(indices):
        return

    within = classes < tally["weight"].shape[1]
    indices, classes = indices[within], classes[within]

    per_pixel = np.ones(len(indices)) if weights is None else weights[indices]
    np.add.at(tally["weight"], (indices, classes), per_pixel)
    np.add.at(tally["count"], (indices, classes), 1)
    np.add.at(tally["n_views_seen"], indices, 1)
    np.add.at(tally["cosine_sum"], indices, per_pixel)


def finalise_votes(tally: Dict[str, np.ndarray]) -> VoteResult:
    """Weighted argmax, with confidence and an honest unlabeled class.

    No smoothing, no nearest-neighbour fill. A point that no view saw stays -1
    so P5 can decide what to do about it; filling it in here would manufacture
    evidence that does not exist.
    """
    weight, count = tally["weight"], tally["count"]
    seen = tally["n_views_seen"]
    total = weight.sum(axis=1)

    decided = (seen > 0) & (total > 0)
    labels = np.where(decided, weight.argmax(axis=1), LABEL_NONE).astype(np.int8)
    unweighted = np.where(seen > 0, count.argmax(axis=1), LABEL_NONE).astype(np.int8)
    confidence = np.where(decided, weight.max(axis=1) / np.maximum(total, 1e-12), 0.0)

    return VoteResult(
        labels=labels,
        n_views_seen=seen,
        weight=weight.astype(np.float32),
        count=count,
        confidence=confidence.astype(np.float32),
        mean_cosine=(tally["cosine_sum"] / np.maximum(seen, 1)).astype(np.float32),
        unweighted_labels=unweighted,
    )


def colors_from_surfels(points: np.ndarray, surfel_means: np.ndarray, surfel_colors: np.ndarray) -> np.ndarray:
    """Colour the surface cloud from the trained surfels' nearest neighbour.

    The carved cloud carries positions and normals but no colour, and SAM needs
    something to look at. The surfels were fitted against the real photographs,
    so their colours are the closest thing to observed appearance available
    without re-projecting into every frame.
    """
    from scipy.spatial import cKDTree

    _, nearest = cKDTree(surfel_means).query(points)
    rgb = 1.0 / (1.0 + np.exp(-surfel_colors[nearest]))  # stored as logits
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
