"""The parts of the VGGT-Omega adapter that can be checked without the model.

The checkpoints are gated and the inference needs a GPU, so what is tested
here is everything downstream of `model(images)`: unprojection, multi-view
fusion, and the COLMAP text the pipeline actually consumes. That is also
where the bugs measured on the other backends lived -- an unfused union of
per-view clouds, and intrinsics left in the model's working resolution.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "vggt_omega_colmap.py"
spec = importlib.util.spec_from_file_location("vggt_omega_colmap", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def _two_view_scene(offset=0.0):
    """Two cameras looking at a plane of points from either side.

    `offset` displaces the second camera's *depth map* only, so the two views
    disagree about where the surface is while both remain individually
    plausible -- the failure the fusion step exists to catch.
    """
    height, width = 24, 32
    K = np.array([[40.0, 0, width / 2], [0, 40.0, height / 2], [0, 0, 1]])
    extrinsics, depths = [], []
    for angle in (0.0, np.pi / 6):
        centre = np.array([2.0 * np.sin(angle), 0.0, -2.0 * np.cos(angle)])
        forward = -centre / np.linalg.norm(centre)
        right = np.cross([0, 1.0, 0], forward)
        right /= np.linalg.norm(right)
        rotation = np.stack([right, np.cross(forward, right), forward])
        extrinsic = np.zeros((3, 4))
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = -rotation @ centre
        extrinsics.append(extrinsic)
        depths.append(np.full((height, width), 2.0 + (offset if angle else 0.0)))
    return np.stack(depths), np.stack([K, K]), np.stack(extrinsics)


def test_unprojection_puts_points_where_the_camera_says_they_are():
    depths, intrinsics, extrinsics = _two_view_scene()
    world = exporter.unproject(depths[0], intrinsics[0], extrinsics[0])
    # The principal ray at depth d must land d in front of the camera centre.
    centre = -extrinsics[0][:3, :3].T @ extrinsics[0][:3, 3]
    middle = world[depths.shape[1] // 2, depths.shape[2] // 2]
    assert np.isclose(np.linalg.norm(middle - centre), 2.0, atol=1e-6)


def test_fusion_keeps_points_two_views_agree_about():
    depths, intrinsics, extrinsics = _two_view_scene()
    points = np.stack([exporter.unproject(depths[i], intrinsics[i], extrinsics[i])
                       for i in range(2)])
    xyz, tracks = exporter.fuse_by_consistency(
        points, depths, intrinsics, extrinsics, masks=None,
        min_views=1, tolerance=0.05, max_points=10_000)
    assert len(xyz) > 0
    # Agreement means real multi-view tracks, which is what BA needs and what
    # the older VGGT export never produced.
    assert max(len(track) for track in tracks) >= 2


def test_fusion_rejects_views_that_disagree_about_the_surface():
    """A second view that puts the surface elsewhere must not corroborate."""
    depths, intrinsics, extrinsics = _two_view_scene(offset=0.8)
    points = np.stack([exporter.unproject(depths[i], intrinsics[i], extrinsics[i])
                       for i in range(2)])
    xyz, _ = exporter.fuse_by_consistency(
        points, depths, intrinsics, extrinsics, masks=None,
        min_views=1, tolerance=0.01, max_points=10_000)
    assert len(xyz) == 0


def test_masks_keep_fusion_inside_the_plant():
    depths, intrinsics, extrinsics = _two_view_scene()
    points = np.stack([exporter.unproject(depths[i], intrinsics[i], extrinsics[i])
                       for i in range(2)])
    masks = np.zeros(depths.shape, bool)
    masks[:, 8:16, 12:20] = True
    xyz, tracks = exporter.fuse_by_consistency(
        points, depths, intrinsics, extrinsics, masks=masks,
        min_views=1, tolerance=0.05, max_points=10_000)
    assert 0 < len(xyz) <= masks[0].sum() * 2
    for track in tracks:
        for view, u, v in track:
            assert masks[view][v, u]


def test_quaternion_round_trips_through_colmap_convention():
    for angle in (0.0, 0.5, 2.0, 3.0):
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]])
        qw, qx, qy, qz = exporter.quaternion_from_matrix(rotation)
        assert np.isclose(qw**2 + qx**2 + qy**2 + qz**2, 1.0, atol=1e-9)
        back = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]])
        np.testing.assert_allclose(back, rotation, atol=1e-9)


def test_written_model_is_readable_and_in_frame_pixels(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")

    depths, intrinsics, extrinsics = _two_view_scene()
    points = np.stack([exporter.unproject(depths[i], intrinsics[i], extrinsics[i])
                       for i in range(2)])
    xyz, tracks = exporter.fuse_by_consistency(
        points, depths, intrinsics, extrinsics, None, 1, 0.05, 2000)

    # What the adapter does before writing: model pixels -> frame pixels.
    frame_sizes = [(320, 240), (320, 240)]
    scale = 320 / 32
    scaled = []
    for K in intrinsics:
        K = K.copy()
        K[0, 0] *= scale; K[0, 2] *= scale
        K[1, 1] *= scale; K[1, 2] *= scale
        scaled.append(K)
    tracks_scaled = [[(v, u * scale, w * scale) for v, u, w in track] for track in tracks]

    out = tmp_path / "sparse"
    exporter.write_colmap_text(out, ["frame_0000.jpg", "frame_0001.jpg"], frame_sizes,
                               scaled, extrinsics, xyz, tracks_scaled,
                               [np.array([10, 20, 30])] * len(xyz))
    reconstruction = pycolmap.Reconstruction(str(out))
    assert reconstruction.num_reg_images() == 2
    assert len(reconstruction.points3D) == len(xyz)
    camera = list(reconstruction.cameras.values())[0]
    assert (camera.width, camera.height) == (320, 240)
    # The principal point must sit at the frame centre, not the model's.
    assert np.isclose(camera.params[2], 160.0) and np.isclose(camera.params[3], 120.0)
    assert {image.name for image in reconstruction.images.values()} == {
        "frame_0000.jpg", "frame_0001.jpg"}
