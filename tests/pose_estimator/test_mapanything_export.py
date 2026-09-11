"""The MapAnything exporter's geometry fix, which is pure arithmetic.

Upstream writes points from the model's world-pointmap head and cameras from
its pose and intrinsics heads. Those are separate predictions and need not
agree: measured on thistle3, only 60% of the exported points fell inside the
P2 silhouette they had been masked to, in the model's own coordinate frame.
Unprojecting depth through the intrinsics that get written makes reprojection
exact instead, which is what these tests pin down.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "mapanything_colmap.py"
spec = importlib.util.spec_from_file_location("mapanything_colmap", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def _prediction(height=12, width=16, focal=30.0, angle=0.4):
    """One view with a known pose, known intrinsics and a sloping depth map."""
    K = torch.tensor([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
    rotation = torch.tensor([[np.cos(angle), 0.0, np.sin(angle)],
                             [0.0, 1.0, 0.0],
                             [-np.sin(angle), 0.0, np.cos(angle)]], dtype=torch.float32)
    cam_to_world = torch.eye(4)
    cam_to_world[:3, :3] = rotation
    cam_to_world[:3, 3] = torch.tensor([0.3, -0.2, 1.5])
    depth = 2.0 + 0.01 * torch.arange(height * width, dtype=torch.float32).reshape(height, width)
    return {
        "depth_z": depth[None, ..., None],
        "intrinsics": K[None],
        "camera_poses": cam_to_world[None],
        # Deliberately wrong, standing in for a pointmap head that disagrees
        # with the camera heads.
        "pts3d": torch.zeros(1, height, width, 3),
    }


def test_rebuilt_points_reproject_onto_the_pixels_they_came_from():
    prediction = _prediction()
    assert exporter.rebuild_points_from_depth([prediction]) == 1

    world = prediction["pts3d"][0].numpy()
    K = prediction["intrinsics"][0].numpy()
    cam_to_world = prediction["camera_poses"][0].numpy()
    world_to_cam = np.linalg.inv(cam_to_world)

    height, width = world.shape[:2]
    points = world.reshape(-1, 3)
    camera_points = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    u = camera_points[:, 0] / camera_points[:, 2] * K[0, 0] + K[0, 2]
    v = camera_points[:, 1] / camera_points[:, 2] * K[1, 1] + K[1, 2]

    expected_v, expected_u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    np.testing.assert_allclose(u, expected_u.reshape(-1), atol=1e-3)
    np.testing.assert_allclose(v, expected_v.reshape(-1), atol=1e-3)
    # ...and at the depth the model predicted, not merely along the right ray.
    np.testing.assert_allclose(camera_points[:, 2],
                               prediction["depth_z"][0].squeeze(-1).numpy().reshape(-1), atol=1e-4)


def test_a_masked_pixel_stays_inside_its_own_mask_after_the_rebuild():
    """The property the whole fix exists for: masking pixels masks points."""
    prediction = _prediction()
    exporter.rebuild_points_from_depth([prediction])

    height, width = prediction["depth_z"][0].shape[:2]
    mask = np.zeros((height, width), bool)
    mask[3:9, 4:12] = True
    world = prediction["pts3d"][0].numpy()[mask]

    K = prediction["intrinsics"][0].numpy()
    world_to_cam = np.linalg.inv(prediction["camera_poses"][0].numpy())
    camera_points = world @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    u = np.round(camera_points[:, 0] / camera_points[:, 2] * K[0, 0] + K[0, 2]).astype(int)
    v = np.round(camera_points[:, 1] / camera_points[:, 2] * K[1, 1] + K[1, 2]).astype(int)
    assert mask[v, u].all()


def test_predictions_without_depth_are_left_alone():
    prediction = _prediction()
    del prediction["depth_z"]
    before = prediction["pts3d"].clone()
    assert exporter.rebuild_points_from_depth([prediction]) == 0
    assert torch.equal(prediction["pts3d"], before)
