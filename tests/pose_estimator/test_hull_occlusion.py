"""Something the holder hides for most of the orbit must survive the carve.

The exposed root sits behind the pliers for most of a turntable rotation, so
it appeared in only 27% of thistle1's masks. A plain silhouette intersection
asks every view to agree, which deleted it -- and the fix is not to lower the
bar (that inflates the whole hull) but to stop counting views that could not
see it. Behind the tool there is no evidence either way.
"""

import numpy as np

from pose_estimator.hull import CarveCamera, carve


def ring_of_cameras(count, radius=6.0, size=200, focal=200.0):
    """Cameras on a circle in the z=0 plane, all looking at the origin."""
    K = np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], float)
    cameras = []
    for i in range(count):
        angle = 2 * np.pi * i / count
        eye = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
        forward = -eye / np.linalg.norm(eye)
        right = np.cross(forward, [0, 0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, forward])
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = R
        world_to_camera[:3, 3] = -R @ eye
        cameras.append((K, world_to_camera, size))
    return cameras


def render(camera, points, radius_px):
    """Silhouette of a blob of points, as seen by one camera."""
    K, w2c, size = camera
    mask = np.zeros((size, size), bool)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    good = cam[:, 2] > 1e-6
    px = (cam[good, :2] / cam[good, 2:3]) @ K[:2, :2].T + K[:2, 2]
    ys, xs = np.mgrid[0:size, 0:size]
    for x, y in px:
        mask |= (xs - x) ** 2 + (ys - y) ** 2 <= radius_px ** 2
    return mask


def build(hide_from, body, thin):
    """Cameras whose silhouette holds the body always and the thin part never
    when hidden; `hide_from` says which views the occluder covers."""
    rigs = ring_of_cameras(24)
    cameras = []
    for i, rig in enumerate(rigs):
        K, w2c, size = rig
        body_mask = render(rig, body, 14)
        thin_mask = render(rig, thin, 5)
        hidden = i in hide_from
        mask = body_mask if hidden else (body_mask | thin_mask)
        occluder = thin_mask & ~body_mask if hidden else np.zeros_like(mask)
        cameras.append(CarveCamera(K=K, world_to_camera=w2c, mask=mask,
                                   name=f"v{i:02d}", occluder=occluder))
    return cameras


BODY = np.array([[0.0, 0.0, 0.6]])
THIN = np.array([[0.0, 0.0, -0.6]])
LOW = np.array([-1.5, -1.5, -1.5])
HIGH = np.array([1.5, 1.5, 1.5])


def _kept_near(points, target, tol=0.45):
    return bool(len(points)) and bool(
        (np.linalg.norm(points - target, axis=1) < tol).any())


def test_occluded_part_is_deleted_without_the_holder_mask():
    """The failure as it stood: hidden reads as absent.

    Hidden in 6 of 24 views, matching what was measured on thistle1's root
    column -- 42 of 192 views holder-hidden, the rest judged. That is enough
    to fail an 86% vote when the hidden views count against.
    """
    cameras = build(set(range(6)), BODY, THIN)      # hidden in 6 of 24
    for camera in cameras:
        camera.occluder = None                      # pretend we have no holder mask
    points, _, _ = carve(cameras, LOW, HIGH, resolution=64, coarse_resolution=32,
                         min_judged_views=8)
    assert _kept_near(points, BODY[0]), "the always-visible body must survive"
    assert not _kept_near(points, THIN[0]), "this is the bug being reproduced"


def test_occluded_part_survives_when_the_holder_mask_is_known():
    """Same geometry, same views -- only the occluder is declared."""
    cameras = build(set(range(6)), BODY, THIN)
    points, _, _ = carve(cameras, LOW, HIGH, resolution=64, coarse_resolution=32,
                         min_judged_views=4)
    assert _kept_near(points, BODY[0]), "the body must still survive"
    assert _kept_near(points, THIN[0]), "the occluded part should now be kept"


def test_too_few_unoccluded_views_is_not_enough():
    """Excusing views must not let two agreeing voters invent geometry."""
    cameras = build(set(range(23)), BODY, THIN)     # only 1 view ever sees it
    points, _, _ = carve(cameras, LOW, HIGH, resolution=64, coarse_resolution=32,
                         min_judged_views=8)
    assert not _kept_near(points, THIN[0])


def test_the_holders_own_shadow_is_not_kept():
    """Excusing occluded views must not build a solid inside the tool.

    The volume within the pliers is hidden in nearly every frame, so almost
    nothing votes against it. On thistle1 that produced 221,484 phantom
    voxels -- 32% of the hull, sitting in the tool -- while the root it was
    meant to rescue stayed missing. A voxel whose evidence is almost entirely
    excused has not been observed at all.
    """
    cameras = build(set(range(23)), BODY, THIN)     # hidden in 23 of 24 views
    points, _, _ = carve(cameras, LOW, HIGH, resolution=64, coarse_resolution=32,
                         min_judged_views=1, min_judged_fraction=0.5)
    assert _kept_near(points, BODY[0]), "the visible body must survive"
    assert not _kept_near(points, THIN[0]), "a shape hidden in 96% of views is not evidence"
