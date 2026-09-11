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


def test_known_intrinsics_land_in_the_models_working_resolution():
    """The mapping must be the exact inverse of the model's own preprocessing.

    MapAnything resizes by the larger ratio and centre-crops, so a focal
    given in frame pixels has to be scaled and the principal point shifted by
    the crop -- getting this wrong is how a correct focal still produces a
    wrong reconstruction.
    """
    # 1920x1280 -> the 3:2 bucket MapAnything picks, which is 518x336.
    K = exporter.processed_intrinsics((3020.48, 3020.48, 960.0, 640.0), (1920, 1280), (336, 518))
    scale = 518 / 1920
    assert np.isclose(K[0, 0], 3020.48 * scale)
    # The crop is 4.5px top and bottom, so a centred principal point stays
    # centred in the cropped image.
    assert np.isclose(K[0, 2], 259.0, atol=0.5)
    assert np.isclose(K[1, 2], 168.0, atol=0.5)


def test_a_focal_only_source_puts_the_principal_point_at_the_frame_centre():
    """EXIF gives a focal and nothing else, which implies a centred pinhole."""
    exif = exporter.processed_intrinsics((2880.0, 2880.0, None, None), (1920, 1280), (336, 518))
    explicit = exporter.processed_intrinsics((2880.0, 2880.0, 960.0, 640.0), (1920, 1280), (336, 518))
    np.testing.assert_allclose(exif, explicit)


def test_the_mapping_inverts_what_the_exporter_undoes_afterwards():
    """Round-trip against the code that maps an exported model back to frames."""
    pytest.importorskip("pycolmap")
    from pose_estimator.geometry import rescale_model_to_frames

    frame = (3020.48, 3020.48, 960.0, 640.0)
    K = exporter.processed_intrinsics(frame, (1920, 1280), (336, 518))

    model = Path(pytest.importorskip("tempfile").mkdtemp()) / "sparse"
    model.mkdir(parents=True)
    (model / "cameras.txt").write_text(
        f"1 PINHOLE 518 336 {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
    (model / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame_0000.jpg\n\n")
    (model / "points3D.txt").write_text("1 0 0 1 128 128 128 0\n")
    rescale_model_to_frames(model, (1920, 1280))

    import pycolmap

    camera = list(pycolmap.Reconstruction(str(model)).cameras.values())[0]
    np.testing.assert_allclose(camera.params[0], frame[0], rtol=1e-6)
    np.testing.assert_allclose(camera.params[2], frame[2], atol=0.5)
    np.testing.assert_allclose(camera.params[3], frame[3], atol=0.5)


def test_intrinsics_sources_resolve_to_real_files(tmp_path):
    import pytest as _pytest

    from pose_estimator.geometry import resolve_intrinsics_source

    assert resolve_intrinsics_source(tmp_path, None) is None
    with _pytest.raises(FileNotFoundError, match="intrinsics.json"):
        resolve_intrinsics_source(tmp_path, "exif")
    (tmp_path / "p1").mkdir()
    (tmp_path / "p1" / "intrinsics.json").write_text('{"frame_0000": 2880.0}')
    assert resolve_intrinsics_source(tmp_path, "exif").endswith("p1/intrinsics.json")
    with _pytest.raises(FileNotFoundError):
        resolve_intrinsics_source(tmp_path, "/nope/missing.json")
