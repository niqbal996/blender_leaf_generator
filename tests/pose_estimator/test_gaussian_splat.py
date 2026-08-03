import numpy as np
import pytest

from pose_estimator.gaussian_splat import init_gaussians_from_pointcloud


def test_init_gaussians_from_pointcloud_shapes_and_means():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(20, 3))
    rgb = rng.integers(0, 256, size=(20, 3)).astype(np.uint8)

    gaussians = init_gaussians_from_pointcloud(xyz, rgb, sh_degree=3)

    np.testing.assert_array_equal(gaussians.means, xyz)
    assert gaussians.log_scales.shape == (20, 3)
    assert gaussians.quats_wxyz.shape == (20, 4)
    assert gaussians.opacity_logits.shape == (20,)
    assert gaussians.sh0.shape == (20, 1, 3)
    assert gaussians.shN.shape == (20, 15, 3)  # (3+1)^2 - 1 = 15


def test_init_gaussians_identity_orientation():
    xyz = np.zeros((5, 3))
    rgb = np.zeros((5, 3), dtype=np.uint8)

    gaussians = init_gaussians_from_pointcloud(xyz, rgb)

    np.testing.assert_array_equal(gaussians.quats_wxyz, np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)))


def test_init_gaussians_higher_order_sh_starts_zero():
    xyz = np.random.default_rng(1).normal(size=(10, 3))
    rgb = np.full((10, 3), 128, dtype=np.uint8)

    gaussians = init_gaussians_from_pointcloud(xyz, rgb, sh_degree=2)

    assert gaussians.shN.shape == (10, 8, 3)  # (2+1)^2 - 1 = 8
    np.testing.assert_array_equal(gaussians.shN, np.zeros((10, 8, 3)))


def test_write_read_gaussian_ply_round_trip(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gsplat")

    from pose_estimator.gaussian_splat import read_gaussian_ply, write_gaussian_ply

    rng = np.random.default_rng(2)
    xyz = rng.normal(size=(8, 3))
    rgb = rng.integers(0, 256, size=(8, 3)).astype(np.uint8)
    gaussians = init_gaussians_from_pointcloud(xyz, rgb, sh_degree=1)

    path = tmp_path / "splat.ply"
    write_gaussian_ply(path, gaussians)
    result = read_gaussian_ply(path)

    np.testing.assert_allclose(result.means, gaussians.means, atol=1e-5)
    np.testing.assert_allclose(result.log_scales, gaussians.log_scales, atol=1e-5)
    np.testing.assert_allclose(result.quats_wxyz, gaussians.quats_wxyz, atol=1e-5)
    np.testing.assert_allclose(result.opacity_logits, gaussians.opacity_logits, atol=1e-5)
    np.testing.assert_allclose(result.sh0, gaussians.sh0, atol=1e-5)
    np.testing.assert_allclose(result.shN, gaussians.shN, atol=1e-5)
