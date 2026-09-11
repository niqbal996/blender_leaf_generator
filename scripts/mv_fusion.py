"""Multi-view depth agreement, shared by the exporters that need it.

A depth error moves a point *along the ray it was seen on*, which leaves its
projection in the view it came from exactly where it was. So a per-view cloud
always looks right in its own view and can only be judged by the others: the
test is whether another view's depth map puts a surface at the same distance.

Measured on thistle3, which is why this exists: with the fusion below, VGGT-
Omega's points are corroborated by a median of 25 of 27 views. MapAnything's
export, which applies no geometric fusion at all, manages 16 -- so a third of
the views disagree about where any given point is, and the cloud reads as a
thick noisy shell rather than a plant.
"""

from __future__ import annotations

import numpy as np


def agreement_matrix(
    candidates: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    masks: np.ndarray | None,
    tolerance: float,
) -> np.ndarray:
    """Which views corroborate each candidate point.

    Args:
        candidates: (N, 3) world points.
        depths: (V, H, W) per-view depth, NaN where there is none.
        intrinsics: (V, 3, 3).
        extrinsics: (V, 3, 4) camera-from-world.
        masks: (V, H, W) bool, or None to accept any pixel.
        tolerance: agreement band, as a fraction of the candidate's depth.

    Returns:
        (N, V) bool: view v places a surface where candidate n claims one.
    """
    num_views, height, width = depths.shape
    agrees = np.zeros((len(candidates), num_views), dtype=bool)
    for view in range(num_views):
        rotation, translation = extrinsics[view][:3, :3], extrinsics[view][:3, 3]
        camera_points = candidates @ rotation.T + translation
        z = camera_points[:, 2]
        in_front = z > 1e-6
        safe = np.where(in_front, z, 1.0)
        u = camera_points[:, 0] / safe * intrinsics[view][0, 0] + intrinsics[view][0, 2]
        v = camera_points[:, 1] / safe * intrinsics[view][1, 1] + intrinsics[view][1, 2]
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        inside = in_front & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        ui = np.clip(ui, 0, width - 1)
        vi = np.clip(vi, 0, height - 1)
        observed = depths[view][vi, ui]
        if masks is not None:
            inside &= masks[view][vi, ui]
        agrees[:, view] = (inside & np.isfinite(observed)
                           & (np.abs(observed - z) <= tolerance * np.maximum(z, 1e-9)))
    return agrees


def reprojected_pixels(
    candidates: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    shape_hw: tuple,
) -> tuple:
    """Candidate points projected into one view: (u, v, inside-the-image)."""
    height, width = shape_hw
    rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
    camera_points = candidates @ rotation.T + translation
    z = camera_points[:, 2]
    in_front = z > 1e-6
    safe = np.where(in_front, z, 1.0)
    u = np.round(camera_points[:, 0] / safe * intrinsic[0, 0] + intrinsic[0, 2]).astype(np.int64)
    v = np.round(camera_points[:, 1] / safe * intrinsic[1, 1] + intrinsic[1, 2]).astype(np.int64)
    inside = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.clip(u, 0, width - 1), np.clip(v, 0, height - 1), inside
