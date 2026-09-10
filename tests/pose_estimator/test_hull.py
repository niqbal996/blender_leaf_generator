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
    rng = np.random.default_rng(0)
    surface = rng.normal(size=(60000, 3))
    surface /= np.linalg.norm(surface, axis=1)[:, None]
    surface *= SPHERE_RADIUS
    return _object_views(surface, n_views=n_views, orbit_radius=orbit_radius, height=height)


def _object_views(surface, n_views=24, orbit_radius=2.0, height=0.35):
    """Cameras on a circle, each rendering one object's true silhouette."""
    K = np.array([[FOCAL, 0, IMAGE_SIZE / 2], [0, FOCAL, IMAGE_SIZE / 2], [0, 0, 1]], float)
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


def _misplace_cameras(cameras, distance):
    """Displace each camera centre by `distance`, leaving its mask truthful.

    This is what a feed-forward P3 backend produces: every view registered,
    each pose individually plausible, none of them mutually consistent to
    the precision an intersection needs. Displacing the centre is also the
    error `compare.json` reports, so the magnitudes here are comparable to
    the ones a real run prints.

    Rotating the cameras instead would not do: a rotation about an axis
    through the object leaves the object's projection near the optical axis
    put, so the hull shrinks toward that axis rather than emptying.
    """
    rng = np.random.default_rng(1)
    moved = []
    for camera in cameras:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        world_to_camera = camera.world_to_camera.copy()
        # X_cam = R(X - C): moving C by s takes t to t - R s.
        world_to_camera[:3, 3] -= world_to_camera[:3, :3] @ (direction * distance)
        moved.append(CarveCamera(K=camera.K, world_to_camera=world_to_camera,
                                 mask=camera.mask, name=camera.name))
    return moved


def _lopsided_surface():
    """Two unequal lobes off the orbit axis.

    A sphere is the wrong object for testing pose sensitivity: it is
    rotationally symmetric, so misplacing the cameras about the orbit axis
    leaves every silhouette perfectly consistent with the same sphere and
    the hull survives. A plant is not symmetric, and neither is this.
    """
    rng = np.random.default_rng(2)
    big = rng.normal(size=(40000, 3))
    big /= np.linalg.norm(big, axis=1)[:, None]
    big = big * 0.20 + np.array([0.22, 0.0, 0.0])
    small = rng.normal(size=(20000, 3))
    small /= np.linalg.norm(small, axis=1)[:, None]
    small = small * 0.10 + np.array([-0.18, 0.10, 0.12])
    return np.vstack([big, small])


def test_an_empty_hull_reports_how_far_the_poses_missed_by():
    """A near miss must be distinguishable from unrelated masks."""
    cameras = _misplace_cameras(_object_views(_lopsided_surface()), 0.25)
    bounds = np.array([-0.6, -0.6, -0.6]), np.array([0.6, 0.6, 0.6])

    with pytest.raises(RuntimeError) as failure:
        carve(cameras, *bounds, resolution=48, min_inside_fraction=0.86)
    message = str(failure.value)

    assert "in-silhouette in only" in message
    assert "86%" in message                       # the threshold it fell short of
    assert "--min-inside-fraction" in message     # the knob that would carve one
    assert "camera-pose error" in message         # the likely cause at this distance
    assert "different objects" not in message     # which this is not


def test_the_suggested_threshold_actually_carves_a_hull():
    """The number in the message has to be usable, not decorative."""
    import re

    cameras = _misplace_cameras(_object_views(_lopsided_surface()), 0.25)
    bounds = np.array([-0.6, -0.6, -0.6]), np.array([0.6, 0.6, 0.6])
    with pytest.raises(RuntimeError) as failure:
        carve(cameras, *bounds, resolution=48, min_inside_fraction=0.86)

    suggested = float(re.search(r"--min-inside-fraction ([0-9.]+)", str(failure.value)).group(1))
    points, _, _ = carve(cameras, *bounds, resolution=48, min_inside_fraction=suggested)
    assert len(points) > 0


def test_masks_from_another_capture_are_called_out_as_such():
    """Total disagreement gets the opposite diagnosis, not a threshold to lower."""
    cameras = _misplace_cameras(_object_views(_lopsided_surface()), 0.60)
    bounds = np.array([-0.6, -0.6, -0.6]), np.array([0.6, 0.6, 0.6])

    with pytest.raises(RuntimeError) as failure:
        carve(cameras, *bounds, resolution=48, min_inside_fraction=0.86)
    message = str(failure.value)
    assert "different objects" in message
    assert "--min-inside-fraction" not in message   # lowering it would not help


def test_good_poses_still_carve_the_sphere_after_the_diagnosis_change():
    points, _, _ = carve(_sphere_views(), np.array([-0.6, -0.6, -0.6]),
                         np.array([0.6, 0.6, 0.6]), resolution=48, min_inside_fraction=0.86)
    radius = np.linalg.norm(points - points.mean(axis=0), axis=1).max()
    assert 0.25 < radius < 0.40      # the known 0.30, within voxel resolution
