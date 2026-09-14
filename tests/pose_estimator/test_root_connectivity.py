"""The root found as a detached body rather than as a label.

P4c labels organs by appearance, and a root is the one organ appearance is
worst at: a shadowed leaf underside is dark and strap-shaped in exactly the
same way. Measured on thistle3, 10,601 points carried the root label and only
571 of them were in the root -- the rest sat in the middle of the canopy and
pulled the crown, and so the whole skeleton, up to 42% of the plant's height.

What the rig guarantees instead is geometric: the jaws grip where the root
meets the shoot and hide a band all the way round, so the root below is a
connected component of its own, sitting under the plant.
"""

import numpy as np

from pose_estimator.structure_labels import root_by_connectivity

VOXEL = 0.01
SPACING = VOXEL          # points a voxel apart, so a body is connected at 2+ voxels


def body(centre, shape, spacing=SPACING):
    """A filled box of points on a regular lattice, centred on `centre`.

    A lattice rather than a gaussian blob: the test is about which bodies are
    connected to which, so the point spacing has to be known rather than
    sampled -- a blob whose own points fall further apart than the search
    radius fragments, and the fixture stops testing what it says it does.
    """
    axes = [np.arange(n) * spacing - (n - 1) * spacing / 2 for n in shape]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid + np.asarray(centre, float)


def clamped_plant():
    """Foliage above, a detached root below, a clamp-sized gap between them."""
    canopy = body([0, 0, 1.0], (12, 12, 12))          # 1728 points
    root = body([0, 0, 0.2], (6, 6, 6))               # 216 points, well clear
    return np.vstack([canopy, root]), len(canopy)


def test_detached_body_below_the_plant_is_the_root():
    points, split = clamped_plant()
    found = root_by_connectivity(points, VOXEL)

    assert found is not None, "a plant with a clamp gap must yield a root"
    assert found[split:].all(), "every root point should be found"
    assert not found[:split].any(), "no canopy point should be called root"


def test_a_severed_leaf_tip_up_in_the_canopy_is_not_the_root():
    """Detachment alone is not enough. A fragment the carve cut off sits at
    canopy height, and calling it root would put the crown up there with it."""
    canopy = body([0, 0, 1.0], (12, 12, 12))
    fragment = body([1.2, 0, 1.1], (6, 6, 6))          # detached, but high
    points = np.vstack([canopy, fragment])

    found = root_by_connectivity(points, VOXEL)
    assert found is None or not found[len(canopy):].any()


def test_nothing_detached_returns_none_rather_than_guessing():
    """A specimen the jaws never fully hid has no gap, so there is no geometric
    answer -- the caller's labels are all there is and must stay in charge."""
    assert root_by_connectivity(body([0, 0, 1.0], (12, 12, 12)), VOXEL) is None


def test_the_split_does_not_move_with_the_radius():
    """The reason to trust this over a tuned label threshold: thistle3 returns
    the same two bodies at 2, 3, 4 and 6 voxels, a factor of three."""
    points, split = clamped_plant()
    counts = {r: int(root_by_connectivity(points, VOXEL, radius_voxels=r).sum())
              for r in (2.0, 3.0, 4.0, 6.0)}
    assert len(set(counts.values())) == 1, f"root size moved with the radius: {counts}"


def test_the_heart_stays_a_point_after_midribs_are_fitted():
    """A rosette has no heart, and must still report None rather than a flag.

    `use_straight_chord` returns a per-leaf boolean and was assigned to a
    variable called `heart`, which is also the name of the anatomical point at
    the top of an upright plant's stem. The last leaf's boolean was what
    reached stem_graph.json, so Blender read `"heart_xyz": true` as a
    coordinate and stopped with `cannot reshape array of size 1 into shape
    (3)`. It stayed hidden while P5 was finding no leaves: the loop that does
    the overwriting never ran.
    """
    from pose_estimator.structure_labels import build_from_labels

    # Four blades radiating from a crown, each a strip rather than a line so
    # the foliage outweighs the root -- the largest component is taken as the
    # plant, and a root bigger than the shoot would be read as the plant.
    spacing = 0.01
    blades = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        step = np.arange(1, 31) * spacing
        for across in (-1, 0, 1):
            shift = np.array([-dy, dx, 0]) * across * spacing
            blades.append(np.stack([step * dx, step * dy, np.zeros_like(step)], 1) + shift)
    leaf = np.vstack(blades)
    root = body([0, 0, -0.30], (4, 4, 4), spacing)
    points = np.vstack([leaf, root])
    labels = np.array([0] * len(leaf) + [1] * len(root))
    assert len(leaf) > len(root), "the shoot has to be the larger body"

    structure = build_from_labels(points, labels, ["leaf", "root"], voxel=spacing,
                                  min_leaf_points=5, architecture="rosette")

    assert structure.num_leaves > 0, "fixture must actually fit some midribs"
    assert structure.heart is None or np.asarray(structure.heart).shape == (3,), (
        f"heart must be a point or None, got {structure.heart!r}")
