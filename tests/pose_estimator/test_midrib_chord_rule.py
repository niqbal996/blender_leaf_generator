"""When a leaf is drawn as a straight line instead of from its own points.

The straight crown-to-tip chord exists for one situation: a rosette's central
leaves, which the cloud fails to reconstruct near the crown and which survive
as one-sided slivers. Selecting them on steepness alone conflates how a leaf
is *posed* with how well it was *reconstructed*, and drew sugarbeet_4's six
large upright blades as sticks.
"""

import numpy as np
import pytest

from pose_estimator.structure_labels import (
    HEART_LEAF_ELEVATION,
    HEART_LEAF_SUPPORT,
    chord_midrib,
    chord_station_support,
    use_straight_chord,
)


def blade(base, tip, n=900, bow=0.0, seed=0):
    """Points spread along base->tip, optionally bowed sideways."""
    rng = np.random.default_rng(seed)
    base, tip = np.asarray(base, float), np.asarray(tip, float)
    t = rng.uniform(0, 1, n)
    axis = tip - base
    # A helper that is never parallel to the axis, or the bow vanishes.
    helper = [1.0, 0, 0] if abs(axis[2]) > abs(axis[0]) else [0, 0, 1.0]
    side = np.cross(axis, helper)
    side = side / max(np.linalg.norm(side), 1e-12)
    return (base + np.outer(t, axis)
            + np.outer(bow * np.sin(np.pi * t), side)
            + rng.normal(0, 0.002, (n, 3)))


def test_support_is_one_when_the_leaf_covers_its_own_chord():
    pts = blade((0, 0, 0), (0, 0, 1.0))
    assert chord_station_support(pts, (0, 0, 0), (0, 0, 1.0)) == pytest.approx(1.0)


def test_support_is_low_for_a_sliver_near_the_tip():
    # What a heart leaf actually leaves behind: nothing over the lower half.
    pts = blade((0, 0, 0.6), (0, 0, 1.0), n=200)
    assert chord_station_support(pts, (0, 0, 0), (0, 0, 1.0)) < 0.5


def test_support_is_zero_with_no_points():
    assert chord_station_support(np.zeros((0, 3)), (0, 0, 0), (0, 0, 1.0)) == 0.0


def test_a_steep_well_reconstructed_leaf_follows_its_points():
    # sugarbeet_4: 8k-36k points, up to 1.33 long, 100% coverage -- and it was
    # being drawn as a straight stick.
    assert not use_straight_chord(60.0, 1.00)
    assert not use_straight_chord(68.0, 0.93)


def test_a_steep_poorly_reconstructed_leaf_keeps_the_chord():
    # thistle3's steep leaves: 79%, 57%, 50%.
    assert use_straight_chord(56.9, 0.79)
    assert use_straight_chord(57.4, 0.57)
    assert use_straight_chord(78.6, 0.50)


def test_a_shallow_leaf_follows_its_points_however_sparse():
    # Steepness is a necessary condition: the chord is not a repair for
    # sparse tissue in general, only for the upright centre of a rosette.
    assert not use_straight_chord(10.0, 0.20)
    assert not use_straight_chord(HEART_LEAF_ELEVATION - 0.1, 0.0)


def test_strict_refuses_the_chord_even_for_a_steep_sliver():
    assert use_straight_chord(80.0, 0.30)
    assert not use_straight_chord(80.0, 0.30, strict=True)


def test_the_threshold_sits_inside_the_measured_gap():
    # thistle3's steep leaves need the chord and score 0.50, 0.57, 0.79.
    # sugarbeet_4's must not have it and score 0.93, 1.00. The threshold
    # belongs between those groups, with margin on both sides rather than
    # hugging either edge.
    needs_chord, must_not = 0.79, 0.93
    assert needs_chord < HEART_LEAF_SUPPORT < must_not
    assert min(HEART_LEAF_SUPPORT - needs_chord, must_not - HEART_LEAF_SUPPORT) >= 0.05


def test_a_chorded_midrib_is_straight_and_a_fitted_one_is_not():
    base, tip = np.array([0.0, 0, 0]), np.array([0.0, 0, 1.0])
    pts = blade(base, tip, bow=0.08)
    straight = chord_midrib(pts, base, tip, heart=True)
    fitted = chord_midrib(pts, base, tip, heart=False)

    def bow(curve):
        return float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()
                     / np.linalg.norm(curve[-1] - curve[0]))

    assert bow(straight) == pytest.approx(1.0, abs=1e-6)
    assert bow(fitted) > 1.005


def test_both_constructions_still_start_and_end_where_told():
    base, tip = np.array([0.1, 0.2, 0.3]), np.array([0.4, -0.2, 1.1])
    pts = blade(base, tip, bow=0.05)
    for heart in (True, False):
        curve = chord_midrib(pts, base, tip, heart=heart)
        assert np.allclose(curve[0], base, atol=1e-6)
        assert np.allclose(curve[-1], tip, atol=1e-6)
