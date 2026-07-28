"""Solve the similarity transform from a COLMAP reconstruction's arbitrary,
unitless SfM frame into real-world meters, Z-up -- so the skeleton, point
cloud, and trained Gaussian Splat (all reconstructed in that same raw COLMAP
frame) can be overlaid with real-world-scaled leaf assets in Blender.

Rotation is solved automatically from the registered camera centers: for a
turntable capture, the centers lie approximately on a circle in a plane, so
PCA's smallest-eigenvalue eigenvector is that plane's normal -- the rotation
("up") axis. Scale is NOT recoverable from SfM alone (no metric anchor
in-shot) -- it must come from one real-world reference measurement.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Alignment:
    rotation: np.ndarray  # (3, 3), COLMAP frame -> Blender Z-up
    scale: float  # meters per COLMAP unit
    translation: np.ndarray  # (3,), applied after rotation+scale

    def apply(self, points_colmap: np.ndarray) -> np.ndarray:
        """Transform (N, 3) points from the raw COLMAP frame into aligned,
        real-world-meter, Z-up coordinates.
        """
        return (np.asarray(points_colmap) @ self.rotation.T) * self.scale + self.translation

    def apply_to_gaussians(
        self, means: np.ndarray, quats_wxyz: np.ndarray, log_scales: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform a trained Gaussian Splat's per-Gaussian means,
        orientation quaternions (w, x, y, z), and *log*-scales (the standard
        3DGS ply pre-activation convention) into the aligned frame.

        Rotating a Gaussian composes its stored orientation with the
        alignment's rotation; a uniform scale factor shifts log-scales
        additively (log(s * scale) = log(s) + log(scale)), not
        multiplicatively -- getting this wrong silently produces
        wrong-sized ellipsoids with no error raised.
        """
        aligned_means = self.apply(means)

        align_rot = Rotation.from_matrix(self.rotation)
        gaussian_rot = Rotation.from_quat(_wxyz_to_xyzw(np.asarray(quats_wxyz)))
        aligned_rot = align_rot * gaussian_rot
        aligned_quats_wxyz = _xyzw_to_wxyz(aligned_rot.as_quat())

        aligned_log_scales = np.asarray(log_scales) + np.log(self.scale)

        return aligned_means, aligned_quats_wxyz, aligned_log_scales

    def to_dict(self) -> dict:
        return {
            "rotation_matrix": self.rotation.tolist(),
            "rotation_quaternion_wxyz": _xyzw_to_wxyz(Rotation.from_matrix(self.rotation).as_quat()).tolist(),
            "scale": self.scale,
            "translation": self.translation.tolist(),
        }


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def estimate_up_axis(camera_centers: np.ndarray) -> np.ndarray:
    """Smallest-eigenvalue eigenvector of the centered camera centers'
    covariance -- the normal of the plane the (approximately circular,
    turntable) camera path lies in. Sign is arbitrary; see `orient_up_axis`.
    """
    centered = np.asarray(camera_centers) - np.mean(camera_centers, axis=0)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalue order
    up = eigvecs[:, 0]
    return up / np.linalg.norm(up)


def orient_up_axis(up: np.ndarray, viewing_dirs: np.ndarray) -> np.ndarray:
    """Flip `up` if needed so it points away from the mean camera viewing
    direction -- for a rig where the camera looks roughly level or slightly
    down at the subject (the common turntable setup), true "up" should
    oppose where the cameras are looking.
    """
    mean_view_dir = np.mean(viewing_dirs, axis=0)
    if np.dot(up, mean_view_dir) > 0:
        return -up
    return up


def rotation_to_z_up(up: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping `up` to [0, 0, 1]. Rotation about the Z axis
    itself is unconstrained (the turntable's start angle is arbitrary and
    doesn't matter for alignment) -- `Rotation.align_vectors` picks the
    minimal such rotation.
    """
    with warnings.catch_warnings():
        # A single vector pair under-determines rotation about that axis --
        # expected here (the turntable's start angle is arbitrary and
        # doesn't matter), not a real problem, so silence scipy's warning.
        warnings.filterwarnings("ignore", message="Optimal rotation is not uniquely")
        rot, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [up])
    return rot.as_matrix()


def solve_scale_from_reference(point_a: np.ndarray, point_b: np.ndarray, real_world_distance_m: float) -> float:
    """Meters-per-COLMAP-unit scale factor from one measured real-world
    distance between two points expressed in the raw COLMAP frame.
    """
    colmap_distance = float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))
    if colmap_distance <= 0:
        raise ValueError("reference points must not coincide")
    return real_world_distance_m / colmap_distance


def solve_alignment(
    camera_centers: np.ndarray,
    viewing_dirs: np.ndarray,
    scale: float,
    recenter_point: Optional[np.ndarray] = None,
) -> Alignment:
    """Solve the full COLMAP-frame -> real-world-meters, Z-up `Alignment`.

    `recenter_point` (raw COLMAP frame, e.g. the cleaned point cloud's
    centroid) maps to the world origin after alignment, if given.
    """
    if len(camera_centers) < 3:
        raise ValueError(f"need at least 3 registered cameras to solve alignment, got {len(camera_centers)}")

    up = orient_up_axis(estimate_up_axis(camera_centers), viewing_dirs)
    rotation = rotation_to_z_up(up)

    translation = np.zeros(3)
    if recenter_point is not None:
        translation = -(rotation @ np.asarray(recenter_point)) * scale

    return Alignment(rotation=rotation, scale=scale, translation=translation)
