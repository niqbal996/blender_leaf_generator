"""The skeleton is traced through tissue whose identity is already known.

P5 has to *find* the leaves first -- crown, geodesic depth field, local maxima
as candidate tips, grouping, ownership -- because a SAM2 P2 gives it one
silhouette. P5x is handed the leaf each point belongs to, so a tip is not
discovered: it is the point of that leaf furthest from the crown along the
plant, and the path back splits itself into petiole (skeleton tissue) and
midrib (the leaf's own points) by reading the labels.

These pin the properties that make that true, on a plant whose answer is known
by construction.
"""

import numpy as np
import pytest

from pose_estimator.leaf_skeleton import SKELETON, UNSEEN, find_crown, trace


def synthetic_plant(heights=(0.3, 0.6, 0.9), angles=(0.0, 2.1, 4.2),
                    reach=0.5, rise=0.3, noise=0.002, seed=0):
    """A vertical stem with one straight leaf off it at each height."""
    rng = np.random.default_rng(seed)
    points, labels = [], []
    z = np.linspace(0.0, 1.0, 200)
    points.append(np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1))
    labels.append(np.full(200, SKELETON))
    for index, (height, angle) in enumerate(zip(heights, angles)):
        t = np.linspace(0.0, reach, 120)
        points.append(np.stack([t * np.cos(angle), t * np.sin(angle),
                                height + t * rise], axis=1))
        labels.append(np.full(120, index))
    xyz = np.vstack(points)
    return xyz + rng.normal(0, noise, xyz.shape), np.concatenate(labels).astype(np.int32)


def test_every_leaf_gets_a_midrib_and_a_tip():
    points, assignment = synthetic_plant()
    out = trace(points, assignment, voxel=0.01)

    assert len(out["leaves"]) == 3
    assert not out["unreachable"]
    for leaf in out["leaves"]:
        assert len(leaf["midrib"]) >= 2
        assert leaf["midrib_length"] > 0


def test_the_tip_is_the_far_end_of_its_own_leaf():
    """Not of whichever leaf the depth field happened to run furthest along.
    Two leaves lying against each other are still two leaves here, because
    they were two instances in 2D."""
    heights = (0.3, 0.6, 0.9)
    points, assignment = synthetic_plant(heights=heights)
    out = trace(points, assignment, voxel=0.01)

    for leaf in sorted(out["leaves"], key=lambda leaf: leaf["id"]):
        expected = heights[leaf["id"]] + 0.5 * 0.3      # reach * rise
        assert abs(leaf["tip"][2] - expected) < 0.05, leaf


def test_the_petiole_lengthens_with_height_up_the_stem():
    """The petiole is the crown-to-leaf part of the path, so a leaf higher up
    the stem must be carried by a longer one. This is what shows the split is
    being read off the labels rather than cut at a fixed distance."""
    out = trace(*synthetic_plant(), voxel=0.01)
    by_id = {leaf["id"]: leaf for leaf in out["leaves"]}
    assert by_id[0]["petiole_length"] < by_id[1]["petiole_length"] < by_id[2]["petiole_length"]


def test_the_midrib_is_the_leaf_and_not_the_stem():
    """A midrib of the whole path would be as long as the petiole plus the
    blade; each leaf here is the same length whatever height it sits at."""
    out = trace(*synthetic_plant(), voxel=0.01)
    lengths = [leaf["midrib_length"] for leaf in out["leaves"]]
    assert max(lengths) - min(lengths) < 0.05, lengths
    assert all(0.45 < length < 0.60 for length in lengths), lengths


def test_a_leaf_cut_off_from_the_plant_is_reported_not_invented():
    """Its bridge to the crown was carved away. A straight line to it would
    look exactly like a measurement."""
    points, assignment = synthetic_plant()
    far = points[:, 2].max() + 5.0
    points = np.vstack([points, np.array([[9.0, 9.0, far], [9.01, 9.0, far]])])
    assignment = np.concatenate([assignment, np.array([7, 7], np.int32)])

    out = trace(points, assignment, voxel=0.01)
    assert 7 in out["unreachable"]
    assert 7 not in [leaf["id"] for leaf in out["leaves"]]


def test_the_crown_sits_at_the_foot_of_the_stem():
    points, assignment = synthetic_plant()
    crown = find_crown(points, assignment)
    assert abs(crown[2]) < 0.05, crown
    assert np.linalg.norm(crown[:2]) < 0.05, "the crown drifted off the stem"


def test_crown_ignores_leaf_tissue():
    """A rosette's lowest points are leaves. Taking the crown from them would
    put the origin under a drooping blade rather than at the holder."""
    points, assignment = synthetic_plant()
    below = np.array([[0.4, 0.4, -1.0], [0.41, 0.4, -1.0]])
    points = np.vstack([points, below])
    assignment = np.concatenate([assignment, np.array([0, 0], np.int32)])

    crown = find_crown(points, assignment)
    assert crown[2] > -0.5, "a low leaf was mistaken for the crown"


def test_a_cloud_with_nothing_labelled_returns_nothing():
    points = np.random.default_rng(0).random((50, 3))
    out = trace(points, np.full(50, UNSEEN, np.int32), voxel=0.01)
    assert out["leaves"] == [] and out["crown"] is None


# --------------------------------------------------------------------------
# The root is below the clamp and cut off from the plant. It must not be
# mistaken for the crown, and a crown that lands on an island must not
# quietly report every leaf as unreachable.
# --------------------------------------------------------------------------


def plant_with_a_severed_root(gap=0.25):
    """The real gaensefuss_1 geometry: a root blob hanging below the jaws,
    with no cloud bridging the gap the clamp occupies."""
    points, assignment = synthetic_plant()
    rng = np.random.default_rng(1)
    root = rng.normal(0, 0.02, (300, 3)) + np.array([0.0, 0.0, -gap])
    return (np.vstack([points, root]),
            np.concatenate([assignment, np.full(300, ROOT, np.int32)]))


ROOT = -3


def test_the_root_is_not_mistaken_for_the_crown():
    """It is the lowest tissue on the plant, so "lowest of stem and root"
    finds it every time -- and it is a separate component, so tracing from it
    reaches nothing."""
    points, assignment = plant_with_a_severed_root()
    crown = find_crown(points, assignment)
    assert crown[2] > -0.1, f"the crown landed in the root blob: {crown}"


def test_every_leaf_is_still_reachable_with_a_severed_root():
    """The failure this pins reported 17 of 17 leaves unreachable on real data
    while the graph was perfectly healthy -- the start was simply stranded."""
    points, assignment = plant_with_a_severed_root()
    out = trace(points, assignment, voxel=0.01)

    assert not out["unreachable"], out["unreachable"]
    assert len(out["leaves"]) == 3


def test_a_stranded_crown_is_moved_and_reported():
    """Forced by giving the plant no stem label at all, so the crown falls
    back to the lowest of everything -- the root island."""
    points, assignment = plant_with_a_severed_root()
    assignment = np.where(assignment == SKELETON, ROOT, assignment)

    out = trace(points, assignment, voxel=0.01)
    assert out["crown_moved_to_largest_component"] is not None, \
        "the crown was left on the island and the leaves reported unreachable"
    assert not out["unreachable"]


# --------------------------------------------------------------------------
# A free shortest path from the crown to a tip is not the leaf's midrib. Where
# stem tissue runs alongside a blade -- which is what a petiole mask looks
# like, and what the stem residual carries -- the cheapest route to the tip
# goes *beside* the blade and steps onto it only at the last node. The leaf
# then contributes one point, `len(midrib_nodes) < 2` fires, and a leaf with
# thousands of points is reported unreachable.
#
# Measured on gaensefuss_1 at 512^3: 10 of 21 leaves failed exactly this way,
# every one of them with precisely 1 own-label node on its path, one of them
# holding 14,991 points. It got *worse* with better geometry, because a denser
# cloud offers more skeleton points to route around the blade with.
# --------------------------------------------------------------------------


def plant_with_tissue_alongside_the_blade(offset=0.03):
    """One leaf, with a line of skeleton points running parallel to it.

    `offset` is inside the graph's edge cap, so both routes to the tip exist
    and the solver is free to prefer the wrong one.
    """
    rng = np.random.default_rng(2)
    points, labels = [], []
    z = np.linspace(0.0, 1.0, 200)
    points.append(np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1))
    labels.append(np.full(200, SKELETON))

    t = np.linspace(0.0, 0.5, 120)
    blade = np.stack([t, np.zeros_like(t), 0.5 + t * 0.3], axis=1)
    points.append(blade)
    labels.append(np.full(120, 0))

    # the same run, one offset below: stem tissue escorting the blade
    points.append(blade + np.array([0.0, 0.0, -offset]))
    labels.append(np.full(120, SKELETON))

    xyz = np.vstack(points)
    return xyz + rng.normal(0, 0.001, xyz.shape), np.concatenate(labels).astype(np.int32)


def test_the_midrib_runs_through_the_blade_not_alongside_it():
    points, assignment = plant_with_tissue_alongside_the_blade()
    out = trace(points, assignment, voxel=0.01)

    assert not out["unreachable"], (
        "the leaf was dropped: the path took the skeleton route beside the blade")
    leaf = out["leaves"][0]
    assert len(leaf["midrib"]) >= 2
    # the blade is 0.5 long and rises 0.15, so its own length is ~0.52. A
    # midrib that ran alongside and hopped on at the end would be far shorter.
    assert leaf["midrib_length"] > 0.4, leaf["midrib_length"]


def test_the_tip_is_still_the_far_end_of_the_blade():
    points, assignment = plant_with_tissue_alongside_the_blade()
    out = trace(points, assignment, voxel=0.01)
    tip = np.asarray(out["leaves"][0]["tip"])
    assert abs(tip[0] - 0.5) < 0.05 and abs(tip[2] - 0.65) < 0.05, tip
