"""Silhouette carving against a synthetic object with known ground truth.

This exists because the projection/visibility conventions here are easy to
get subtly wrong in a way that still produces a plausible-looking hull. The
first implementation counted an off-image projection as tacit agreement,
which quietly kept every voxel outside all the frustums: it recovered a
radius of 1.14 for a true 0.30 while looking entirely reasonable in the
aggregate statistics. Only a known-answer test catches that.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pose_estimator.hull import CarveCamera, carve  # noqa: E402

SPHERE_RADIUS = 0.30
IMAGE_SIZE = 480
FOCAL = 480.0


def _sphere_views(n_views=24, orbit_radius=2.0, height=0.35):
    """Cameras on a circle, each rendering the silhouette of a known sphere."""
    K = np.array([[FOCAL, 0, IMAGE_SIZE / 2], [0, FOCAL, IMAGE_SIZE / 2], [0, 0, 1]], float)
    rng = np.random.default_rng(0)
    surface = rng.normal(size=(60000, 3))
    surface /= np.linalg.norm(surface, axis=1)[:, None]
    surface *= SPHERE_RADIUS

    cameras = []
    for angle in np.linspace(0, 2 * np.pi, n_views, endpoint=False):
        center = np.array([orbit_radius * np.cos(angle), orbit_radius * np.sin(angle), height])
        forward = -center / np.linalg.norm(center)
        right = np.cross(forward, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])

        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = -rotation @ center

        camera = CarveCamera(
            K=K,
            world_to_camera=world_to_camera,
            mask=np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool),
            name=f"view_{angle:.3f}",
        )
        pixels, in_front = camera.project(surface)
        x = np.round(pixels[:, 0]).astype(int)
        y = np.round(pixels[:, 1]).astype(int)
        good = in_front & (x >= 0) & (x < IMAGE_SIZE) & (y >= 0) & (y < IMAGE_SIZE)

        rendered = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        rendered[y[good], x[good]] = 1
        camera.mask = cv2.morphologyEx(
            rendered, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
        ).astype(bool)
        cameras.append(camera)
    return cameras


def test_carve_recovers_a_known_sphere():
    cameras = _sphere_views()
    points, _voxel, _bounds = carve(
        cameras,
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        resolution=96,
        min_inside_fraction=1.0,
    )

    radii = np.linalg.norm(points, axis=1)

    # The hull is an outward bound, so it may exceed the true radius slightly
    # (voxel quantisation + silhouette closing) but must never undercut it.
    assert radii.max() == pytest.approx(SPHERE_RADIUS, rel=0.10)
    assert radii.max() >= SPHERE_RADIUS * 0.98, "hull cut inside the true surface"
    assert np.abs(points.mean(axis=0)).max() < 0.02, "hull is off-centre"


def test_carve_rejects_voxels_outside_the_frustums():
    """The specific regression: geometry no camera can see must not survive."""
    cameras = _sphere_views()
    points, _voxel, _bounds = carve(
        cameras,
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        resolution=96,
        min_inside_fraction=1.0,
    )

    # Nothing should survive out near the corners of the initial volume.
    assert np.linalg.norm(points, axis=1).max() < 0.5
