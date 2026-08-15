"""Tip-driven leaf instancing: the three failures it was built to stop.

Each test builds a tiny synthetic plant where the right answer is known by
construction, so a regression names the failure rather than just moving a
count. The three are, in order: a second high point on one blade splitting it
in two, a leaf's territory running through the stem into a neighbour, and a
bud with no tip of its own being absorbed instead of standing alone.
"""

import numpy as np
import pytest

from pose_estimator.structure_labels import (
    claim_orphans,
    depth_from_stem,
    grow_from_tips,
    instance_by_tips,
    leaf_graph,
    leaf_midrib,
    select_tips,
    tip_persistence,
)

SPACING = 0.01
CONTACT = SPACING * 1.5


def blade(start, direction, n, spacing=SPACING, width=0):
    """A strip of points running from `start` along `direction`."""
    direction = np.asarray(direction, float)
    direction = direction / np.linalg.norm(direction)
    side = np.cross(direction, [0, 0, 1.0])
    if np.linalg.norm(side) < 1e-9:
        side = np.array([1.0, 0, 0])
    side /= np.linalg.norm(side)
    pts = []
    for i in range(n):
        for w in range(-width, width + 1):
            pts.append(np.asarray(start, float) + direction * i * spacing + side * w * spacing)
    return np.array(pts)


K = 6


def two_leaf_plant():
    """A vertical stem with one leaf low on the left and one high on the right.

    The two insertions are deliberately far apart in height. Attached at the
    same height they would sit a couple of point-spacings from each other and
    the kNN graph would join them *directly*, so a front could pass from one to
    the other without ever going through the stem -- which is not the situation
    the stem block is for, and made this fixture test nothing.
    """
    stem = blade([0, 0, 0], [0, 0, 1], 14)
    left = blade([-SPACING, 0, 0.02], [-1, 0, 0], 20)
    right = blade([SPACING, 0, 0.10], [1, 0, 0], 20)
    return np.vstack([left, right]), stem, len(left)


def test_two_separate_leaves_are_two_instances():
    leaf, stem, split = two_leaf_plant()
    inst = instance_by_tips(leaf, stem, CONTACT, min_points=5, min_tip_depth=SPACING * 5, k_neighbors=K)
    assert len(inst.accepted_tips) == 2
    assert len({int(o) for o in inst.owner if o >= 0}) == 2
    # each side keeps its own points
    left_ids = set(inst.owner[:split].tolist()) - {-1}
    right_ids = set(inst.owner[split:].tolist()) - {-1}
    assert left_ids.isdisjoint(right_ids)


def test_a_second_bump_on_one_blade_does_not_split_it():
    """The failure that split single leaves in two: one blade offering two
    high points. Persistence must see the second as part of the first,
    because the tissue between them never descends to the stem."""
    stem = blade([0, 0, 0], [0, 0, 1], 12)
    main = blade([SPACING, 0, 0.05], [1, 0, 0], 24)
    # a stub hanging off the middle of that blade -- a second local maximum,
    # but one whose path to the main tip stays far out on the leaf
    stub = blade([SPACING * 12, 0, 0.05], [0, 1, 0], 6)[1:]
    leaf = np.vstack([main, stub])

    graph = leaf_graph(leaf, K)
    depth, _ = depth_from_stem(leaf, stem, graph, CONTACT)
    records = tip_persistence(graph, depth)
    _, accepted = select_tips(records, min_depth=SPACING * 3, min_persistence_ratio=0.5)

    assert len(accepted) == 1, "a bump on a blade must not become its own leaf"
    # and the saddle between the two maxima sits well above the stem
    saddles = sorted(s for _i, _p, s in records if s > 0)
    assert saddles and max(saddles) > depth.max() * 0.2


def bridged_chain(n=18, bridge=8):
    """A line of nodes joined end to end, with one nominated as the bridge.

    Crafted rather than grown from kNN on purpose. `leaf_graph` holds *leaf*
    points only -- the stem is not a node in it -- so two leaves are joined in
    that graph by their petiole bases being neighbours, and those bases are
    exactly the contact points. This is that situation reduced to the one edge
    the block has to cut, which a geometric fixture kept obscuring: place the
    two blades close enough to connect at all and they also connect a point or
    two above the base, bypassing the block for reasons that have nothing to
    do with the rule being tested.
    """
    from scipy.sparse import csr_matrix

    rows = list(range(n - 1))
    cols = list(range(1, n))
    data = [1.0] * (n - 1)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)
    # depth falls to zero at the bridge and rises again on the far side, the
    # shape a stem junction between two leaves actually has
    depth = np.abs(np.arange(n) - bridge).astype(float) * 0.01
    return graph, depth, np.array([bridge], np.int64)


def test_growth_stops_at_the_stem():
    """A front that reaches the stem must not continue up the far side --
    that is what put a bud on the end of an unrelated leaf's midrib."""
    graph, _depth, contact = bridged_chain()
    tip = np.array([0])

    unblocked, _ = grow_from_tips(graph, tip, graph.shape[0])
    blocked, _ = grow_from_tips(graph, tip, graph.shape[0], blocked_at=contact)

    assert (unblocked >= 0).all(), "without the block it runs the whole chain"
    assert (blocked[:9] >= 0).all(), "it still claims its own side, up to the junction"
    assert (blocked[9:] < 0).all(), "and nothing past the junction"


def test_orphan_tissue_becomes_its_own_leaf_not_someone_elses_tail():
    graph, depth, contact = bridged_chain()
    tip = np.array([0])
    owner, distance = grow_from_tips(graph, tip, graph.shape[0], blocked_at=contact)
    assert (owner < 0).any(), "the far side should start out unclaimed"

    owner, distance, new_tips = claim_orphans(
        graph, owner, distance, depth, min_points=3, min_depth=0.02)

    assert len(new_tips) == 1
    assert (owner[9:] == 1).all(), "the far side becomes its own instance"
    assert (owner[:9] == 0).all(), "the original leaf is untouched"
    assert new_tips[0] == graph.shape[0] - 1, "seeded from its own deepest point"


def test_orphan_scraps_at_the_stem_are_not_promoted():
    """Tissue lying against the stem has no depth of its own; calling it a leaf
    produced instances whose 'tip' was zero voxels from the stem."""
    graph, depth, contact = bridged_chain()
    owner = np.full(graph.shape[0], -1, np.int64)
    distance = np.full(graph.shape[0], np.inf)

    # demand more depth than any of this tissue has
    owner, _d, new_tips = claim_orphans(graph, owner, distance, depth,
                                        min_points=3, min_depth=depth.max() * 2)
    assert new_tips == []
    assert (owner < 0).all()


def test_midrib_runs_base_to_tip_and_follows_a_bend():
    """Geodesic shells, not Euclidean: an L-shaped leaf's midrib must follow
    the bend rather than cutting the corner."""
    leaf = np.vstack([blade([0, 0, 0], [1, 0, 0], 15),
                      blade([SPACING * 14, 0, 0], [0, 1, 0], 15)[1:]])
    graph = leaf_graph(leaf, K)
    from scipy.sparse.csgraph import dijkstra

    tip = len(leaf) - 1
    distance = dijkstra(graph, directed=False, indices=[tip], min_only=True)
    curve, tip_xyz, attachment = leaf_midrib(leaf, distance, num_stations=10, min_bin=1)

    assert len(curve) >= 4
    assert np.allclose(curve[-1], tip_xyz, atol=SPACING * 2), "curve must end at the tip"
    assert np.allclose(curve[0], attachment)
    # the corner is occupied, so a midrib that cut it would leave the tissue
    corner = np.array([SPACING * 14, 0, 0])
    assert np.min(np.linalg.norm(curve - corner, axis=1)) < SPACING * 3


def test_no_leaves_when_there_is_no_stem():
    leaf = blade([0, 0, 0], [1, 0, 0], 10)
    inst = instance_by_tips(leaf, np.zeros((0, 3)), CONTACT, min_points=2)
    assert len(inst.contact_index) == 0
    assert not np.isfinite(inst.depth).any()


def test_empty_input_is_handled():
    inst = instance_by_tips(np.zeros((0, 3)), np.zeros((0, 3)), CONTACT)
    assert len(inst.owner) == 0
    assert inst.to_dict()["leaf_points"] == 0


def test_tip_is_the_point_furthest_from_the_stem():
    """Geodesic depth picks the point furthest *through the tissue*, which on a
    sheet is often a lateral corner rather than the pointy end. A leaf's tip is
    whatever part of it gets furthest from the stem, so that is what is used."""
    from pose_estimator.structure_labels import (
        _tips_furthest_from_stem,
        distance_to_polyline,
        tip_reach_shortfall,
    )

    stem_line = blade([0, 0, 0], [0, 0, 1], 10)
    # one leaf running out sideways; the far end is unambiguous
    leaf = blade([SPACING, 0, 0.05], [1, 0, 0], 25)
    owner = np.zeros(len(leaf), np.int64)

    tips = _tips_furthest_from_stem(leaf, owner, stem_line)
    assert len(tips) == 1
    radial = distance_to_polyline(leaf, stem_line)
    assert tips[0] == int(np.argmax(radial)), "must be the furthest point"
    assert tip_reach_shortfall(leaf, owner, [tips[0]], stem_line)[0] == pytest.approx(0.0)

    # a tip put anywhere else is reported as short, which is the whole check
    wrong = int(np.argmin(radial))
    assert tip_reach_shortfall(leaf, owner, [wrong], stem_line)[0] > SPACING * 10


def test_shortfall_is_per_instance():
    """Leaf A's reach must not excuse a bad tip on leaf B."""
    from pose_estimator.structure_labels import tip_reach_shortfall

    stem_line = blade([0, 0, 0], [0, 0, 1], 10)
    long_leaf = blade([SPACING, 0, 0.02], [1, 0, 0], 30)
    short_leaf = blade([-SPACING, 0, 0.08], [-1, 0, 0], 10)
    leaf = np.vstack([long_leaf, short_leaf])
    owner = np.concatenate([np.zeros(len(long_leaf), np.int64),
                            np.ones(len(short_leaf), np.int64)])

    good = [len(long_leaf) - 1, len(leaf) - 1]        # each leaf's own far end
    assert all(s == pytest.approx(0.0) for s in tip_reach_shortfall(leaf, owner, good, stem_line))

    bad = [len(long_leaf) - 1, len(long_leaf)]        # leaf 1's tip at its base
    shortfall = tip_reach_shortfall(leaf, owner, bad, stem_line)
    assert shortfall[0] == pytest.approx(0.0)
    assert shortfall[1] > SPACING * 5


def test_centreline_stations_in_empty_space_are_dropped():
    """A station is the mean of its bin, which lands on the axis when the bin
    surrounds it and in mid-air when it does not. The latter is what made the
    stem swing away from the cloud in an L and pulled a midrib base out to meet
    it."""
    from pose_estimator.structure_labels import drop_stations_in_empty_space

    tissue = blade([0, 0, 0], [0, 0, 1], 40)
    good = np.array([[0, 0, z] for z in np.linspace(0, 0.39, 12)])
    kept = drop_stations_in_empty_space(good, tissue)
    assert len(kept) == len(good), "a centreline lying in the tissue is untouched"

    strayed = good.copy()
    strayed[1] = [0.5, 0.5, 0.02]          # far outside the plant
    kept = drop_stations_in_empty_space(strayed, tissue)
    assert len(kept) == len(good) - 1
    assert not any(np.allclose(k, [0.5, 0.5, 0.02]) for k in kept)


def test_empty_space_test_does_not_punish_a_fat_stem():
    """The gap from an axis to the nearest surface point *is* the radius, so a
    thick stem must not be mistaken for a centreline gone astray."""
    from pose_estimator.structure_labels import drop_stations_in_empty_space

    rng = np.random.default_rng(0)
    z = np.linspace(0, 0.4, 60)
    theta = rng.uniform(0, 2 * np.pi, 60)
    radius = 0.05                                   # deliberately wide
    tube = np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)
    axis = np.array([[0, 0, zz] for zz in np.linspace(0, 0.4, 12)])

    kept = drop_stations_in_empty_space(axis, tube)
    assert len(kept) == len(axis)


def test_root_anchor_is_a_real_shoot_point_at_the_junction():
    """Not the centroid of the root's top slice: a root branches, so that
    centroid falls between its branches -- 33 voxels from any root point on
    plant_9, which is the mid-air failure being avoided."""
    from pose_estimator.structure_labels import root_anchor

    shoot = blade([0, 0, 0], [0, 0, 1], 30)
    # two root branches splaying below, with nothing down the middle
    root = np.vstack([blade([0, 0, -0.01], [-1, 0, -1], 15),
                      blade([0, 0, -0.01], [1, 0, -1], 15)])

    anchor = root_anchor(root, shoot)
    assert anchor is not None
    assert any(np.allclose(anchor, p) for p in shoot), "must be an actual shoot point"
    assert anchor[2] == pytest.approx(shoot[:, 2].min(), abs=SPACING * 2)


def test_root_anchor_absent_without_a_root():
    from pose_estimator.structure_labels import root_anchor
    assert root_anchor(np.zeros((0, 3)), blade([0, 0, 0], [0, 0, 1], 5)) is None
    assert root_anchor(None, blade([0, 0, 0], [0, 0, 1], 5)) is None
