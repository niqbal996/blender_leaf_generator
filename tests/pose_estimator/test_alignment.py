import numpy as np
from scipy.spatial.transform import Rotation

from pose_estimator.alignment import (
    Alignment,
    estimate_up_axis,
    orient_up_axis,
    rotation_to_z_up,
    solve_alignment,
    solve_scale_from_reference,
)


def _turntable_cameras(true_up, rng, n=24, radius=3.0, height=1.5):
    """Camera centers on a circle around `true_up`, each looking at the
    origin -- mimics a turntable rig with the object rotating below/around
    a roughly fixed camera.
    """
    true_up = true_up / np.linalg.norm(true_up)
    # any vector not parallel to true_up, to build an orthonormal basis
    seed = np.array([1.0, 0.0, 0.0]) if abs(true_up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    basis_a = np.cross(true_up, seed)
    basis_a /= np.linalg.norm(basis_a)
    basis_b = np.cross(true_up, basis_a)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centers = np.array(
        [radius * np.cos(a) * basis_a + radius * np.sin(a) * basis_b + height * true_up for a in angles]
    )
    centers += rng.normal(scale=0.01, size=centers.shape)
    viewing_dirs = -centers / np.linalg.norm(centers, axis=1, keepdims=True)
    return centers, viewing_dirs


def test_estimate_up_axis_recovers_known_axis_under_rotation():
    rng = np.random.default_rng(0)
    true_rotation = Rotation.from_euler("xyz", [37, -52, 19], degrees=True).as_matrix()
    true_up = true_rotation @ np.array([0.0, 0.0, 1.0])

    centers, _ = _turntable_cameras(true_up, rng)
    estimated = estimate_up_axis(centers)

    assert abs(abs(np.dot(estimated, true_up)) - 1.0) < 1e-2


def test_orient_up_axis_resolves_sign():
    rng = np.random.default_rng(1)
    true_up = np.array([0.0, 0.0, 1.0])
    centers, viewing_dirs = _turntable_cameras(true_up, rng)

    for candidate in (estimate_up_axis(centers), -estimate_up_axis(centers)):
        oriented = orient_up_axis(candidate, viewing_dirs)
        assert np.dot(oriented, true_up) > 0


def test_rotation_to_z_up_maps_up_to_z():
    up = np.array([0.3, -0.6, 0.74])
    up /= np.linalg.norm(up)

    rotation = rotation_to_z_up(up)

    np.testing.assert_allclose(rotation @ up, [0, 0, 1], atol=1e-6)


def test_solve_scale_from_reference():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([2.0, 0.0, 0.0])

    scale = solve_scale_from_reference(a, b, real_world_distance_m=0.5)

    assert scale == 0.25


def test_alignment_apply_matches_manual_formula():
    rotation = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    alignment = Alignment(rotation=rotation, scale=2.0, translation=np.array([1.0, 2.0, 3.0]))
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 5.0]])

    result = alignment.apply(points)
    expected = (points @ rotation.T) * 2.0 + np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(result, expected)


def test_alignment_apply_to_gaussians_scale_and_rotation():
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    alignment = Alignment(rotation=rotation, scale=2.0, translation=np.zeros(3))

    means = np.array([[1.0, 0.0, 0.0]])
    quats_wxyz = np.array([[1.0, 0.0, 0.0, 0.0]])  # identity orientation
    log_scales = np.array([[0.0, 0.0, 0.0]])

    aligned_means, aligned_quats, aligned_log_scales = alignment.apply_to_gaussians(means, quats_wxyz, log_scales)

    np.testing.assert_allclose(aligned_means, [[0.0, 2.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(aligned_log_scales, np.log(2.0) * np.ones((1, 3)), atol=1e-6)

    # A vector rotated by the (identity) Gaussian orientation then the
    # alignment's rotation should match a vector rotated by the aligned
    # quaternion directly.
    v = np.array([1.0, 0.3, -0.2])
    direct = Rotation.from_quat(aligned_quats[0][[1, 2, 3, 0]]).apply(v)
    expected = Rotation.from_matrix(rotation).apply(v)
    np.testing.assert_allclose(direct, expected, atol=1e-6)


def test_solve_alignment_recenters_to_origin():
    rng = np.random.default_rng(2)
    true_up = np.array([0.0, 0.0, 1.0])
    centers, viewing_dirs = _turntable_cameras(true_up, rng)
    recenter_point = np.array([0.5, -0.3, 0.7])

    alignment = solve_alignment(centers, viewing_dirs, scale=1.5, recenter_point=recenter_point)

    np.testing.assert_allclose(alignment.apply(recenter_point), [0, 0, 0], atol=1e-6)
