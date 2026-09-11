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


def _two_view_outputs(height=16, width=20, disagree=0.0, radius=0.5):
    """Two views of one sphere, so both depth maps describe the same surface.

    Two planes at constant depth would not do: they meet only along a line, so
    even perfect cameras would agree nowhere. `disagree` slides view 1's depth
    along its own rays, which leaves its points where they were in its own
    image and is therefore invisible to it -- only the other view can notice.
    """
    focal = 25.0
    K = torch.tensor([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
    outputs = []
    for index, angle in enumerate((0.0, 0.35)):
        rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                             [0.0, 1.0, 0.0],
                             [-np.sin(angle), 0.0, np.cos(angle)]])
        # centre = -1.5 * forward, so both cameras actually look at the sphere.
        centre = np.array([-1.5 * np.sin(angle), 0.0, -1.5 * np.cos(angle)])
        vs, us = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        rays = np.stack([(us - width / 2) / focal, (vs - height / 2) / focal,
                         np.ones_like(us, dtype=float)], axis=-1)
        world_rays = rays @ rotation.T
        # |centre + s * ray| = radius, nearest root; depth_z is s because the
        # ray's camera-frame z component is 1.
        a = (world_rays ** 2).sum(-1)
        b = 2 * (world_rays * centre).sum(-1)
        c = float(centre @ centre) - radius ** 2
        disc = b ** 2 - 4 * a * c
        s = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.nan)
        depth = s + (disagree if index else 0.0)
        hit = np.isfinite(depth) & (depth > 0)

        cam_to_world = torch.eye(4)
        cam_to_world[:3, :3] = torch.from_numpy(rotation).float()
        cam_to_world[:3, 3] = torch.from_numpy(centre).float()
        outputs.append({
            "depth_z": torch.from_numpy(np.nan_to_num(depth, nan=0.0)).float()[None, ..., None],
            "intrinsics": K[None],
            "camera_poses": cam_to_world,
            "pts3d": torch.zeros(1, height, width, 3),
            "mask": torch.from_numpy(hit)[None, ..., None],
        })
        outputs[-1]["camera_poses"] = cam_to_world[None]
    exporter.rebuild_points_from_depth(outputs)
    return outputs


def test_fusion_keeps_pixels_the_other_view_corroborates():
    outputs = _two_view_outputs()
    kept, before = exporter.fuse_masks_by_consistency(outputs, min_views=1, tolerance=0.02)
    assert before > 0
    assert kept > 0.5 * before          # the sphere is seen by both views


def test_fusion_drops_pixels_when_the_views_disagree_about_depth():
    """The failure this exists for: shells that each look right alone."""
    agreeing = exporter.fuse_masks_by_consistency(_two_view_outputs(), 1, 0.02)[0]
    disagreeing = exporter.fuse_masks_by_consistency(
        _two_view_outputs(disagree=0.5), 1, 0.02)[0]
    assert disagreeing == 0
    assert agreeing > 0


def test_fusion_only_shrinks_the_mask_so_upstream_still_does_the_export():
    outputs = _two_view_outputs()
    original = [prediction["mask"].clone() for prediction in outputs]
    exporter.fuse_masks_by_consistency(outputs, min_views=1, tolerance=0.02)
    for prediction, before in zip(outputs, original):
        assert prediction["mask"].shape == before.shape
        # A mask may only lose pixels, never gain them.
        assert not (prediction["mask"] & ~before).any()


def _colmap_text_model(tmp_path, poses):
    """A minimal COLMAP text model with the given (name, qvec, tvec) images."""
    model = tmp_path / "sparse"
    model.mkdir(parents=True, exist_ok=True)
    (model / "cameras.txt").write_text("1 PINHOLE 64 48 50 50 32 24\n")
    lines = []
    for index, (name, q, t) in enumerate(poses, start=1):
        lines.append(f"{index} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")
    (model / "images.txt").write_text("".join(lines))
    (model / "points3D.txt").write_text("1 0 0 1 128 128 128 0\n")
    return model


def test_colmap_poses_are_inverted_into_camera_to_world():
    """COLMAP stores world-to-camera; MapAnything wants the inverse."""
    pytest.importorskip("pycolmap")
    import tempfile

    tmp_path = Path(tempfile.mkdtemp())
    # Identity rotation, camera at world (0, 0, -3): world2cam translation is
    # -R @ C = (0, 0, 3).
    model = _colmap_text_model(tmp_path, [("frame_0000.jpg", (1, 0, 0, 0), (0, 0, 3))])

    poses = exporter.colmap_cam_to_world(str(model), ["frame_0000.jpg"])
    cam_to_world = poses["frame_0000"]
    assert cam_to_world.shape == (4, 4)
    np.testing.assert_allclose(cam_to_world[:3, 3], [0, 0, -3], atol=1e-9)
    np.testing.assert_allclose(cam_to_world[:3, :3], np.eye(3), atol=1e-9)


def test_a_missing_first_pose_is_refused_with_the_reason():
    """Partial poses are fine, except for the one view the model insists on."""
    names = ["frame_0000.jpg", "frame_0001.jpg"]
    exporter.check_first_view_has_a_pose({"frame_0000": np.eye(4)}, names, "somewhere")

    with pytest.raises(SystemExit) as failure:
        exporter.check_first_view_has_a_pose({"frame_0001": np.eye(4)}, names, "somewhere")
    message = str(failure.value)
    assert "frame_0000" in message
    assert "1 of 2 frames" in message
    assert "--poses-from" in message
