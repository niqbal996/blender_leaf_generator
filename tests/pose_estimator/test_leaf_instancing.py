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


def test_orphan_tissue_becomes_its_own_leaf_not_someone_elses_tail():
    graph, depth, contact = bridged_chain()
    tip = np.array([0])
    # One leaf holding its own side, the far side unclaimed. Built directly
    # rather than grown, so this tests `claim_orphans` and nothing else.
    owner = np.full(graph.shape[0], -1, np.int64)
    owner[:9] = 0
    distance = np.where(owner >= 0, np.arange(graph.shape[0]) * 0.01, np.inf)
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


def test_caulescent_needs_a_stem():
    """Asked for the stem path explicitly, no stem still means no leaves.

    This used to be the behaviour in every case, which is what made a rosette
    unprocessable: a thistle has no stem tissue to seed the depth field from,
    so it returned zero leaves and no amount of clicking could fix it.
    """
    leaf = blade([0, 0, 0], [1, 0, 0], 10)
    inst = instance_by_tips(leaf, np.zeros((0, 3)), CONTACT, min_points=2,
                            architecture="caulescent")
    assert len(inst.contact_index) == 0
    assert not np.isfinite(inst.depth).any()


def test_rosette_finds_a_crown_without_any_stem():
    """Blades radiating from one point get a base, and it lands on that point."""
    centre = np.array([0.0, 0.0, 0.0])
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    leaf = np.vstack([blade(centre, d, 12) for d in directions])

    inst = instance_by_tips(leaf, np.zeros((0, 3)), CONTACT, min_points=5,
                            architecture="rosette")

    assert inst.base is not None, "no stem and no base means nothing downstream can run"
    assert inst.architecture == "rosette"
    # The crown is where the blades meet, not somewhere out along one of them.
    assert np.linalg.norm(inst.base.center - centre) < SPACING * 4
    assert np.isfinite(inst.depth).all(), "every blade must be reachable from the crown"
    assert len({int(o) for o in inst.owner if o >= 0}) == len(directions)


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


def test_caulescent_is_the_default_and_uses_the_stem():
    """The stemmed path is what you get unless you ask for a rosette."""
    from pose_estimator.structure_labels import instance_by_tips

    stem = np.array([[0, 0, z * SPACING] for z in range(12)], float)
    leaf = np.vstack([blade([0, 0, SPACING * 3], [1, 0, 0], 10),
                      blade([0, 0, SPACING * 9], [-1, 0, 0], 10)])

    inst = instance_by_tips(leaf, stem, CONTACT, min_points=3)
    assert inst.architecture == "caulescent"
    assert inst.base is None, "no crown search happens on the stemmed path"
    assert len(inst.contact_index) > 0
    from pose_estimator.structure_labels import depth_from_stem, leaf_graph
    expected = depth_from_stem(leaf, stem, leaf_graph(leaf, 10), CONTACT)[1]
    assert set(inst.contact_index.tolist()) == set(expected.tolist()), \
        "depth must start from the stem, not from a derived crown"


def test_rosette_trunk_is_a_point_not_a_curve():
    """A rosette must not emit a stem polyline.

    The trunk tracer returns whatever nodes the leaf paths happen to share,
    which inside a crown is a few voxels of zigzag -- 3 nodes spanning 4% of
    the plant on runs/thistle1. Rendered as a tube that is an elbow of pipe no
    thistle has, so the base collapses to the crown instead.
    """
    from pose_estimator.structure_labels import build_from_labels

    centre = np.array([0.0, 0.0, 0.0])
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    points = np.vstack([blade(centre, d, 12) for d in directions])
    labels = np.zeros(len(points), np.int64)          # every point is leaf

    structure = build_from_labels(points, labels, ["leaf"], voxel=SPACING,
                                  min_leaf_points=5, min_tip_depth_voxels=2.0,
                                  architecture="rosette")

    assert len(structure.stem_path) == 1, "a rosette has no centreline to draw"
    assert np.linalg.norm(structure.stem_path[0] - centre) < SPACING * 4
    # Blade separation is asserted in test_rosette_finds_a_crown_without_any_stem;
    # these toy blades are too short for a midrib fit, which num_leaves counts.
    assert len({int(i) for i in structure.leaf_ids if i >= 0}) == len(directions)


def test_rosette_ignores_stem_labels():
    """--architecture rosette must not seed depth from stem tissue.

    P4c calls a rosette's crown "stem" -- it is thick and not lamina -- so
    the labels are there and are wrong for this purpose. Told it is a rosette,
    P5 has to use the crown it locates, not the stem contact ring.
    """
    from pose_estimator.structure_labels import instance_by_tips

    centre = np.array([0.0, 0.0, 0.0])
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    leaf = np.vstack([blade(centre, d, 12) for d in directions])
    stem = centre + SPACING * np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])

    inst = instance_by_tips(leaf, stem, CONTACT, min_points=5, architecture="rosette")
    assert inst.architecture == "rosette"
    assert inst.base is not None, "a crown must have been located"
    assert len({int(o) for o in inst.owner if o >= 0}) == len(directions)


def test_every_midrib_starts_at_the_crown_and_goes_no_further():
    """One curve per tip, beginning at the base, with no overshoot.

    A rosette's base is a single crown node, and landing the midrib on it
    used to be guarded on there being a stem *line* -- so on a thistle every
    midrib stopped short of the crown. Worse, the path that extends a midrib
    past its blade can run through the crown and up a neighbour, drawing a
    curve across a leaf that has no tip of its own. Missing a leaf is
    acceptable; inventing one over it is not.
    """
    from pose_estimator.structure_labels import build_from_labels

    centre = np.array([0.0, 0.0, 0.0])
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    # wide enough that the midrib binning has real shells to average
    points = np.vstack([blade(centre, d, 40, width=2) for d in directions])
    labels = np.zeros(len(points), np.int64)

    structure = build_from_labels(points, labels, ["leaf"], voxel=SPACING,
                                  min_leaf_points=20, min_tip_depth_voxels=2.0,
                                  architecture="rosette")

    crown = structure.stem_path[0]
    assert len(structure.stem_path) == 1, "a rosette base is a point"
    assert structure.axes, "no midribs fitted"
    assert len(structure.axes) == structure.num_leaves, "one curve per leaf, no more"

    for axis in structure.axes:
        start = np.linalg.norm(axis[0] - crown)
        assert start < SPACING * 2, f"midrib starts {start:.3f} from the crown"
        # The far end must be further from the crown than the near end, and
        # no interior station may double back through the crown.
        radial = np.linalg.norm(axis - crown, axis=1)
        assert radial[-1] > radial[0], "curve runs the wrong way"
        assert radial[1:].min() >= radial[0] - SPACING, "curve doubles back past the crown"


def test_clip_to_base_removes_overshoot_past_the_crown():
    """A midrib that runs through the crown and up a neighbour is trimmed.

    This is the case the structural test above cannot reach: it needs an
    instance whose extension path crosses the base, which only happens when
    two leaves were merged. Tested directly on the polyline instead.
    """
    from pose_estimator.structure_labels import clip_to_base

    crown = np.array([0.0, 0.0, 0.0])
    # starts up a neighbouring leaf, comes down through the crown, then runs
    # out along its own leaf
    curve = np.array([[-0.30, 0, 0], [-0.15, 0, 0], [0.02, 0, 0],
                      [0.20, 0, 0], [0.40, 0, 0]])
    clipped = clip_to_base(curve, crown)

    assert np.allclose(clipped[0], crown), "must start at the crown"
    assert len(clipped) == 4, f"overshoot not removed: {clipped}"
    assert (np.linalg.norm(clipped - crown, axis=1)[1:] >= 0).all()
    # nothing on the neighbour's side of the crown survives
    assert (clipped[1:, 0] >= 0).all(), "kept points past the crown"


def test_clip_to_base_is_a_no_op_without_a_base():
    from pose_estimator.structure_labels import clip_to_base
    curve = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    assert np.allclose(clip_to_base(curve, None), curve)


def test_crown_is_located_even_when_it_is_labelled_stem():
    """Searching leaf tissue alone looks for the crown in a cloud without one.

    A rosette's crown is thick, so P4c labels it stem and it leaves the leaf
    set -- which turns four blades meeting at a point into four disconnected
    strips. The crown search has to include stem tissue.
    """
    from pose_estimator.structure_labels import _crown_base, leaf_graph

    centre = np.array([0.0, 0.0, 0.0])
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    gap = SPACING * 5
    leaf = np.vstack([blade(centre + gap * np.array(d, float), d, 14) for d in directions])
    step = np.arange(-6, 7) * (SPACING / 2)
    grid = np.array([[x, y, z] for x in step for y in step for z in step])
    stem = centre + grid[np.linalg.norm(grid, axis=1) <= SPACING * 5.5]

    base = _crown_base(leaf, stem, leaf_graph(leaf, 10), 10)
    assert base is not None, "no crown found"
    assert np.linalg.norm(base.center - centre) < SPACING * 3, \
        f"crown at {base.center}, expected near the origin"
    assert len(base.nodes) and (base.nodes < len(leaf)).all(), \
        "nodes must index the leaf array, which is what depth runs over"


def test_every_midrib_actually_touches_the_crown():
    """Smoothing must not detach a curve from the base it was attached to.

    clip_to_base puts the crown on the front of the curve; fit_smooth_curve
    then runs, and a smoothing spline does not interpolate its endpoints. On
    thistle1 that left midribs starting 3.3 to 11.9 voxels from the crown, so
    the skeleton was not connected.
    """
    from pose_estimator.structure_labels import build_from_labels

    centre = np.array([0.0, 0.0, 0.0])
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    points = np.vstack([blade(centre, d, 40, width=2) for d in directions])
    labels = np.zeros(len(points), np.int64)

    structure = build_from_labels(points, labels, ["leaf"], voxel=SPACING,
                                  min_leaf_points=20, min_tip_depth_voxels=2.0,
                                  architecture="rosette")
    crown = structure.stem_path[0]
    assert structure.axes
    for i, axis in enumerate(structure.axes):
        gap = float(np.linalg.norm(axis[0] - crown))
        assert gap < 1e-9, f"leaf {i} starts {gap / SPACING:.1f} voxels off the crown"


def test_chord_midrib_cannot_loop_or_double_back():
    """Progress from base to tip is structural, not hoped for.

    Geodesic-shell centroids can sit anywhere when an instance holds tissue
    from two blades, which is how midribs ended up looping and crossing into
    neighbours. Projecting onto the chord makes reversal impossible.
    """
    from pose_estimator.structure_labels import chord_midrib

    base, tip = np.zeros(3), np.array([1.0, 0.0, 0.0])
    t = np.linspace(0, 1, 300)
    blade_pts = np.stack([t, 0.2 * np.sin(np.pi * t), np.zeros_like(t)], 1)
    # a clump of foreign tissue off to one side, as a merged instance carries
    stray = np.stack([np.full(80, 0.5), np.full(80, -1.5), np.linspace(-.2, .2, 80)], 1)

    curve = chord_midrib(np.vstack([blade_pts, stray]), base, tip)

    assert np.allclose(curve[0], base) and np.allclose(curve[-1], tip)
    along = (curve - base) @ np.array([1.0, 0, 0])
    assert (np.diff(along) > 0).all(), "curve reversed along its own chord"
    arc = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
    assert arc < 2.0 * np.linalg.norm(tip - base), f"curve wanders: arc/chord {arc:.2f}"


def test_chord_midrib_follows_a_bowed_leaf():
    """It must still bend to the tissue, not collapse onto the straight line."""
    from pose_estimator.structure_labels import chord_midrib

    base, tip = np.zeros(3), np.array([1.0, 0.0, 0.0])
    t = np.linspace(0, 1, 200)
    pts = np.stack([t, 0.15 * np.sin(np.pi * t), np.zeros_like(t)], 1)
    curve = chord_midrib(pts, base, tip)
    assert 0.10 < np.abs(curve[:, 1]).max() < 0.20, "did not follow the bow"
