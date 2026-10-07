"""The stem as a branched tree, and petioles taken from it.

Measured on vogelmeere (chickweed, 2026-10-07): the stem was drawn as one
crown-to-top path, 92 mm of a 340 mm stem system, and P5x's stem radius --
measured around the lower trunk, where this plant already branches -- came
out at 51 mm, so every leaf base sat "inside the stem" and all 48 petioles
were empty.
"""

import numpy as np

from pose_estimator import stem_tree
from pose_estimator.leaf_skeleton import SKELETON, trace

VOXEL = 0.01
REACH = 3.0 * VOXEL


def _tube(start, end, radius, n, rng):
    t = rng.uniform(0.0, 1.0, n)
    direction = np.asarray(end, float) - np.asarray(start, float)
    length = np.linalg.norm(direction)
    direction /= length
    helper = np.array([1.0, 0, 0]) if abs(direction[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    return (np.asarray(start, float) + np.outer(t * length, direction)
            + np.outer(r * np.cos(theta), u) + np.outer(r * np.sin(theta), v))


def _blade(centre, normal_axis, size, n, rng):
    """A flat disc of leaf tissue lying across `normal_axis`."""
    pts = rng.uniform(-size, size, (n, 3))
    pts = pts[np.linalg.norm(pts, axis=1) < size]
    pts[:, normal_axis] *= 0.05
    return pts + np.asarray(centre, float)


def branching_plant(seed=0):
    """A main stem; low on it a side branch ending in two leaves; higher up a
    leaf on its own stalk. Leaf ids: 0 = the stalked leaf, 1 and 2 = the
    branch's pair."""
    rng = np.random.default_rng(seed)
    stem = _tube((0, 0, 0), (0, 0, 1.0), 0.012, 5000, rng)
    branch = _tube((0, 0, 0.25), (0.35, 0, 0.55), 0.01, 2500, rng)
    stalk = _tube((0, 0, 0.7), (-0.28, 0, 0.75), 0.008, 1500, rng)
    leaf0 = _blade((-0.37, 0, 0.76), 2, 0.09, 4000, rng)
    leaf1 = _blade((0.38, 0.075, 0.55), 2, 0.07, 3000, rng)
    leaf2 = _blade((0.38, -0.075, 0.55), 2, 0.07, 3000, rng)
    xyz = np.vstack([stem, branch, stalk, leaf0, leaf1, leaf2])
    labels = np.concatenate([np.full(len(stem) + len(branch) + len(stalk), SKELETON),
                             np.full(len(leaf0), 0), np.full(len(leaf1), 1),
                             np.full(len(leaf2), 2)]).astype(np.int32)
    return xyz, labels


def test_the_tree_finds_the_main_stem_the_branch_and_the_stalk():
    xyz, labels = branching_plant()
    tree = stem_tree.build(xyz[labels == SKELETON], (0, 0, 0), REACH)
    found = stem_tree.classify_petioles(tree, xyz[labels >= 0], labels[labels >= 0], REACH)
    kinds = sorted(a.kind for a in tree.axes)
    assert kinds == ["branch", "petiole", "stem"], kinds
    assert found == 1
    petiole = next(a for a in tree.axes if a.kind == "petiole")
    assert petiole.leaf == 0, "the stalk carries leaf 0"
    lines = tree.polylines()
    main = lines[0]
    assert main[0][2] < 0.05 and main[-1][2] > 0.9, "the main stem runs foot to top"


def test_the_radius_is_the_stems_own_not_the_spread_of_its_branches():
    xyz, labels = branching_plant()
    tree = stem_tree.build(xyz[labels == SKELETON], (0, 0, 0), REACH)
    assert tree.radius < 0.03, f"radius {tree.radius:.3f} for a 0.012 stem"


def test_children_start_on_their_parents_smoothed_curve():
    xyz, labels = branching_plant()
    tree = stem_tree.build(xyz[labels == SKELETON], (0, 0, 0), REACH)
    lines = tree.polylines()
    for axis, line in zip(tree.axes, lines):
        if axis.parent >= 0:
            gap = np.linalg.norm(lines[axis.parent] - line[0], axis=1).min()
            assert gap < 1e-9, f"axis starts {gap:.3f} off its parent"


def test_a_branch_leaves_the_stem_along_its_own_direction():
    """A slip road, not a T-junction. Just above a fork the slices still hold
    stem and branch as one piece, so the branch's first node of its own sits
    to the side; joined straight to the fork node, vogelmeere's branches left
    the stem steeply and then turned onto their course."""
    xyz, labels = branching_plant()
    tree = stem_tree.build(xyz[labels == SKELETON], (0, 0, 0), REACH)
    lines = tree.polylines()
    branch = next(line for axis, line in zip(tree.axes, lines)
                  if axis.parent >= 0 and line[-1][0] > 0.2)       # the one heading +x
    first, overall = branch[1] - branch[0], branch[-1] - branch[0]
    angle = np.degrees(np.arccos(first @ overall / np.linalg.norm(first) / np.linalg.norm(overall)))
    # Joined at the fork node it started 0.046 too high and turned 8.4 deg.
    assert angle < 4, f"the branch leaves the stem {angle:.1f} deg off its own course"
    # The fixture's branch leaves the stem at (0, 0, 0.25).
    assert np.linalg.norm(branch[0] - [0, 0, 0.25]) < 0.025, branch[0]


def test_petioles_come_from_the_tree_not_from_the_main_stem():
    xyz, labels = branching_plant()
    out = trace(xyz, labels, voxel=VOXEL)
    by_id = {leaf["id"]: leaf for leaf in out["leaves"]}
    assert set(by_id) == {0, 1, 2}
    # The stalked leaf's petiole is its stalk.
    assert by_id[0]["petiole_from"] == "stalk"
    assert 0.2 < by_id[0]["petiole_length"] < 0.45, by_id[0]["petiole_length"]
    # The branch's pair hang off the branch: no petiole runs back down it to
    # the main stem (the branch alone is 0.46 long).
    for i in (1, 2):
        assert by_id[i]["petiole_length"] < 0.2, by_id[i]["petiole_length"]
    kinds = sorted(axis["kind"] for axis in out["axes"])
    assert kinds == ["branch", "petiole", "stem"], kinds


def test_a_leaf_reached_through_a_neighbour_does_not_copy_its_petiole():
    """vogelmeere: the shortest paths to leaves 8, 9 and 37 climbed leaf 6's
    stalk and crossed its blade, so leaf 6 showed four petioles. A leaf with no
    stalk of its own, touching another leaf's blade, gets no petiole -- not a
    copy of the neighbour's."""
    xyz, labels = branching_plant()
    rng = np.random.default_rng(5)
    beyond = _blade((-0.52, 0, 0.76), 2, 0.07, 3000, rng)       # touches leaf 0's far edge
    xyz = np.vstack([xyz, beyond])
    labels = np.concatenate([labels, np.full(len(beyond), 3)]).astype(np.int32)
    out = trace(xyz, labels, voxel=VOXEL)
    by_id = {leaf["id"]: leaf for leaf in out["leaves"]}
    assert by_id[0]["petiole_from"] == "stalk"
    assert by_id[3]["petiole_length"] < 0.05, \
        f"leaf 3 drew a {by_id[3]['petiole_length']:.2f} petiole through leaf 0"
