import numpy as np
import pytest

from pose_estimator.skeletonize import build_skeleton_graph


def _sample_segment(p0, p1, n, rng, noise=0.02):
    p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
    t = np.linspace(0, 1, n)[:, None]
    pts = p0 + t * (p1 - p0)
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts


def _y_shaped_plant(rng):
    """One stem splitting into two branches -- 1 branch point, 3 tips."""
    stem = _sample_segment([0, 0, 0], [0, 0, 5], 40, rng)
    branch_a = _sample_segment([0, 0, 5], [-2, 0, 7], 20, rng)
    branch_b = _sample_segment([0, 0, 5], [2, 0, 7], 20, rng)
    return np.vstack([stem, branch_a, branch_b])


def test_y_shaped_plant_has_one_branch_and_three_endpoints():
    """3 degree-1 endpoints, but only 2 are "tips": with no `root_xyz` given,
    the skeletonizer still picks one endpoint as the root and labels it
    "root", so `num_tips` is endpoints-minus-one by design.
    """
    rng = np.random.default_rng(0)
    xyz = _y_shaped_plant(rng)

    skeleton = build_skeleton_graph(xyz, k_neighbors=6)

    assert skeleton.num_branch_points == 1
    assert skeleton.num_tips == 2
    assert sum(1 for k in skeleton.keypoint_kinds.values() if k == "root") == 1


def test_y_shaped_plant_branch_point_near_junction():
    rng = np.random.default_rng(1)
    xyz = _y_shaped_plant(rng)

    skeleton = build_skeleton_graph(xyz, k_neighbors=6)

    branch_indices = [i for i, kind in skeleton.keypoint_kinds.items() if kind == "branch"]
    branch_point = skeleton.points[branch_indices[0]]

    assert np.linalg.norm(branch_point - np.array([0, 0, 5])) < 0.5


def test_straight_line_has_root_and_one_tip_no_branch():
    rng = np.random.default_rng(2)
    xyz = _sample_segment([0, 0, 0], [0, 0, 10], 50, rng)

    skeleton = build_skeleton_graph(xyz, k_neighbors=6)

    assert skeleton.num_branch_points == 0
    assert skeleton.num_tips == 1
    assert sum(1 for k in skeleton.keypoint_kinds.values() if k == "root") == 1


def test_short_spur_gets_pruned():
    """A tiny spurious branch (much shorter than the main structure) should
    be pruned away rather than reported as a real branch point.
    """
    rng = np.random.default_rng(3)
    stem = _sample_segment([0, 0, 0], [0, 0, 10], 60, rng)
    tiny_spur = _sample_segment([0, 0, 5], [0.15, 0, 5.05], 4, rng, noise=0.005)
    xyz = np.vstack([stem, tiny_spur])

    skeleton = build_skeleton_graph(xyz, k_neighbors=6, min_branch_fraction=0.03)

    assert skeleton.num_branch_points == 0
    assert skeleton.num_tips == 1


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        build_skeleton_graph(np.zeros((2, 3)))


def test_branch_polylines_cover_every_simplified_edge():
    rng = np.random.default_rng(0)
    xyz = _y_shaped_plant(rng)

    skeleton = build_skeleton_graph(xyz, k_neighbors=6)

    assert set(skeleton.branch_polylines) == set(skeleton.simplified_edges)

    mst_edge_set = {frozenset(e) for e in skeleton.mst_edges}
    for edge, path in skeleton.branch_polylines.items():
        assert path[0] == edge[0]
        assert path[-1] == edge[1]
        for a, b in zip(path, path[1:]):
            assert frozenset((a, b)) in mst_edge_set
