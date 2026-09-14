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


def test_corroborated_copies_collapse_into_one_point():
    """Every view that vouches for a point used to write its own copy of it.

    Corroboration is the whole test the fusion applies, so the better a point
    was the more times it appeared: on thistle3, 357,352 points with a mean
    track length of 10.3, collapsing to 35,261 at half a percent of the plant's
    extent -- 10.1x redundancy, which is the track length. The copies landed
    wherever each view's own depth put them, anywhere inside the agreement
    band, so the "dense" cloud was a slab rather than a surface: local flatness
    0.726 against 0.390 for the COLMAP baseline, past the 0.577 an isotropic
    ball of points would score.
    """
    depths, intrinsics, extrinsics = _two_view_scene()
    points = np.stack([exporter.unproject(depths[i], intrinsics[i], extrinsics[i])
                       for i in range(2)])
    common = dict(masks=None, min_views=1, tolerance=0.05, max_points=10_000)

    kept, _ = exporter.fuse_by_consistency(points, depths, intrinsics, extrinsics, **common)
    extent = float(np.linalg.norm(kept.max(axis=0) - kept.min(axis=0)))
    merged, tracks = exporter.fuse_by_consistency(
        points, depths, intrinsics, extrinsics, merge_radius=0.01 * extent, **common)

    assert len(merged) < len(kept), "the copies must collapse, not merely be reordered"
    # No two survivors may still be describing the same bit of surface.
    from scipy.spatial import cKDTree
    nearest = cKDTree(merged).query(merged, k=2)[0][:, 1]
    assert nearest.min() > 0.0
    # A merged point keeps every view that saw it, so BA still has tracks.
    assert max(len(track) for track in tracks) >= 2


def test_agreement_band_can_be_set_in_scene_units():
    """`tolerance` is a fraction of the distance to the camera, which is a
    property of where the tripod stood and not of the plant. The two exporters
    defaulted to 0.01 and 0.02 on scenes whose extents differed by 4x, so they
    were holding their backends to different standards. `absolute` is the band
    that means the same thing on both."""
    import sys
    sys.path.insert(0, str(SCRIPT.parent))
    import mv_fusion

    depths, intrinsics, extrinsics = _two_view_scene(offset=0.4)
    points = np.stack([exporter.unproject(depths[i], intrinsics[i], extrinsics[i])
                       for i in range(2)])
    candidates = points[0].reshape(-1, 3)

    wide = mv_fusion.agreement_matrix(candidates, depths, intrinsics, extrinsics,
                                      None, 0.0, absolute=1.0)
    narrow = mv_fusion.agreement_matrix(candidates, depths, intrinsics, extrinsics,
                                        None, 0.0, absolute=0.01)
    # Not zero: the second view is rotated, so a 0.4 shift in its depth map is
    # not a 0.4 disagreement everywhere in the image -- points near the edge
    # sit at grazing geometry and move much less. The claim is that the band
    # dominates the outcome, which is the whole reason it has to be set in
    # units that mean something about the plant.
    assert narrow[:, 1].sum() < 0.1 * wide[:, 1].sum(), (
        f"narrow band kept {narrow[:, 1].sum()} of {wide[:, 1].sum()}")


def _noisy_plane_scene(noise, views=10, size=40, seed=0):
    """One world plane, `views` cameras round it, independent per-view depth noise."""
    rng = np.random.default_rng(seed)
    K = np.array([[50.0, 0, size / 2], [0, 50.0, size / 2], [0, 0, 1]])
    extrinsics = []
    for angle in np.linspace(0, 2 * np.pi, views, endpoint=False):
        centre = np.array([3 * np.cos(angle), 3 * np.sin(angle), 2.5])
        forward = -centre / np.linalg.norm(centre)
        right = np.cross(forward, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        rotation = np.stack([right, np.cross(forward, right), forward])
        extrinsics.append(np.hstack([rotation, (-rotation @ centre).reshape(3, 1)]))
    extrinsics = np.stack(extrinsics)

    rows, cols = np.mgrid[0:size, 0:size]
    dirs = np.stack([(cols - K[0, 2]) / K[0, 0], (rows - K[1, 2]) / K[1, 1],
                     np.ones_like(cols, float)], -1)
    depths, masks, points = [], [], []
    for extrinsic in extrinsics:
        rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
        centre = -rotation.T @ translation
        world_dirs = dirs @ rotation
        scale = -centre[2] / world_dirs[..., 2]
        hit = centre + world_dirs * scale[..., None]
        ok = (np.isfinite(scale) & (scale > 0)
              & (np.abs(hit[..., 0]) < 1.0) & (np.abs(hit[..., 1]) < 1.0))
        depth = np.where(ok, scale + rng.normal(0, noise, scale.shape), np.nan)
        camera = np.stack([(cols - K[0, 2]) / K[0, 0] * depth,
                           (rows - K[1, 2]) / K[1, 1] * depth, depth], -1)
        depths.append(depth)
        masks.append(ok)
        points.append((camera - translation) @ rotation)
    return (np.stack(points), np.stack(depths), np.stack([K] * views),
            extrinsics, np.stack(masks))


def test_the_agreement_band_is_measured_rather_than_guessed():
    """A fixed band has to be wrong for one backend to be right for another.

    0.005 of the plant's extent was 3.5x tighter than VGGT-Omega's old setting
    and 4.6x tighter than MapAnything's, and MapAnything's thistle3 export fell
    from 188,586 points to 12,397 -- taking every root point with it, because
    root tissue is gripped by the jaws and so corroborated by the fewest views.
    Measured from the data, the band tracks whatever noise the backend has.
    """
    import sys
    sys.path.insert(0, str(SCRIPT.parent))
    import mv_fusion

    bands = {}
    for noise in (0.01, 0.04):
        points, depths, intrinsics, extrinsics, masks = _noisy_plane_scene(noise)
        bands[noise] = mv_fusion.auto_agreement_band(points, depths, intrinsics,
                                                     extrinsics, masks, min_views=3)
    assert bands[0.04] > bands[0.01] * 1.5, f"band must follow the noise: {bands}"
    # ... and land near it rather than at an arbitrary multiple of it.
    for noise, band in bands.items():
        assert 0.5 * noise < band < 4.0 * noise, f"band {band} unrelated to noise {noise}"


def test_a_noisy_backend_keeps_its_tissue_under_the_measured_band():
    """The regression this exists to stop: a band chosen for a clean backend
    silently discarding most of a noisier one's reconstruction."""
    import sys
    sys.path.insert(0, str(SCRIPT.parent))
    import mv_fusion

    points, depths, intrinsics, extrinsics, masks = _noisy_plane_scene(0.06)
    extent = mv_fusion.plant_extent(points, masks, depths)
    measured = mv_fusion.auto_agreement_band(points, depths, intrinsics, extrinsics,
                                             masks, min_views=3)

    common = dict(masks=masks, min_views=3, tolerance=0.01, max_points=100_000)
    kept_auto, _ = exporter.fuse_by_consistency(points, depths, intrinsics, extrinsics,
                                                absolute=measured, **common)
    kept_fixed, _ = exporter.fuse_by_consistency(points, depths, intrinsics, extrinsics,
                                                 absolute=0.005 * extent, **common)
    assert len(kept_auto) > 3 * len(kept_fixed), (
        f"measured band kept {len(kept_auto)}, fixed kept {len(kept_fixed)}")
