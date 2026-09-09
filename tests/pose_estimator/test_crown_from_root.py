"""The crown is where the shoot leaves the root, and the jaws hide that point.

Both specimens that broke this did so for a different reason, and a fix for
either one alone re-breaks the other -- so both are here as tests.
"""

import numpy as np
import pytest

from pose_estimator.structure_labels import crown_from_root

VOXEL = 0.0075


def blob(centre, n=400, spread=0.02, seed=0):
    return np.random.default_rng(seed).normal(centre, spread, size=(n, 3))


def column(x, y, z0, z1, n=400, seed=1):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(x, 0.01, n), rng.normal(y, 0.01, n),
                     rng.uniform(z0, z1, n)], axis=1)


def test_crown_sits_just_above_the_root():
    root = blob((0, 0, -0.10))
    foliage = column(0, 0, 0.05, 1.0)
    crown = crown_from_root(root, foliage, VOXEL)
    assert crown is not None
    assert 0.04 < crown[2] < 0.12


def test_a_crown_offset_sideways_from_the_root_is_still_found():
    # sugarbeet_4: the shoot does not come up above the middle of the root --
    # its crown tissue sits 0.045-0.15 off the root's axis. A fixed column
    # about that axis contained no crown at all and the search walked up it
    # into the canopy, reporting z=0.973 against foliage starting at z=0.068.
    root = blob((0, 0, -0.10))
    foliage = column(0.20, 0.10, 0.05, 1.0)
    crown = crown_from_root(root, foliage, VOXEL)
    assert crown is not None
    assert crown[2] < 0.15, f"crown climbed to z={crown[2]:.3f}"


def test_a_leaf_drooping_below_the_root_is_not_the_crown():
    # thistle3: outer leaves hang below the crown, and the lowest tissue in the
    # cloud is a blade tip. Without the height guard the crown lands on it.
    root = blob((0, 0, 0.30))
    upright = column(0, 0, 0.35, 1.0)
    drooping = column(0.05, 0.0, 0.05, 0.28, n=200, seed=2)
    crown = crown_from_root(root, np.vstack([upright, drooping]), VOXEL)
    assert crown is not None
    assert crown[2] > 0.30, f"crown fell onto the drooping blade at z={crown[2]:.3f}"


def test_unassigned_tissue_counts_the_same_as_assigned():
    # The caller must not filter on leaf ownership: own_by_subtree leaves crown
    # tissue unowned by design, so filtering discards what is being searched
    # for. Whatever the instancing did, the crown must not move.
    root = blob((0, 0, -0.10))
    low, high = column(0, 0, 0.05, 0.3, seed=3), column(0, 0, 0.3, 1.0, seed=4)
    everything = crown_from_root(root, np.vstack([low, high]), VOXEL)
    only_high = crown_from_root(root, high, VOXEL)
    assert everything[2] < 0.12
    assert only_high[2] > 0.28
    assert everything[2] < only_high[2]


def test_stray_root_points_in_the_canopy_do_not_raise_the_floor():
    # Dirt on the turntable classifies as root; on thistle1 that put 189 stray
    # root points up among the leaves. The largest cluster is the root.
    root = np.vstack([blob((0, 0, -0.10)), blob((0.1, 0.1, 0.90), n=40, seed=5)])
    crown = crown_from_root(root, column(0, 0, 0.05, 1.0), VOXEL)
    assert crown is not None and crown[2] < 0.12


@pytest.mark.parametrize("top_fraction", [0.02, 0.05, 0.10, 0.20, 0.35])
def test_the_root_top_quantile_does_not_matter(top_fraction):
    # Measured on both specimens: a factor of 17 in this knob moves the crown
    # not at all (thistle3 z=0.3070, sugarbeet_4 z=0.0784 throughout).
    root = blob((0, 0, -0.10))
    crown = crown_from_root(root, column(0, 0, 0.05, 1.0), VOXEL, top_fraction=top_fraction)
    assert crown is not None and 0.04 < crown[2] < 0.12


def test_no_root_means_no_answer_rather_than_a_guess():
    assert crown_from_root(None, column(0, 0, 0.05, 1.0), VOXEL) is None
    assert crown_from_root(np.zeros((0, 3)), column(0, 0, 0.05, 1.0), VOXEL) is None


def test_nothing_above_the_root_means_no_answer():
    root = blob((0, 0, 0.90))
    assert crown_from_root(root, column(0, 0, 0.05, 0.5), VOXEL) is None
