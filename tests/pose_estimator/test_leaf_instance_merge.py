"""P5x: a leaf must not be counted once per capture pass.

Each capture pass is its own SAM3 session and its object ids restart at zero,
so the id of a leaf in pass 0 says nothing about its id in pass 1. Measured on
gaensefuss_1 (three passes of 45, 40 and 50 frames) the first version of this
wrote every pass into one masks/leaf_instances/<id>/, and id 16 accumulated
130 frames -- three unrelated leaves under one name. Their votes spread over
the whole plant and 43% of the cloud fell through to skeleton.

The fix has two halves and both are pinned here: the folders are namespaced
per pass, and the passes are matched to each other in 3D afterwards, because
the cloud is the one frame of reference all of them share.
"""

import numpy as np
import pytest

from pose_estimator.cli.leaf_instances import (
    ROOT,
    SKELETON,
    UNSEEN,
    _combine_passes,
    group_by_pass,
    leaf_palette,
    merge_across_passes,
)


class _Folder:
    """Stands in for a Path with only the attribute under test."""

    def __init__(self, name):
        self.name = name


def test_folders_are_grouped_by_their_pass_prefix():
    groups = group_by_pass([_Folder("pass0_3"), _Folder("pass0_7"), _Folder("pass1_3")])
    assert sorted(groups) == ["pass0_", "pass1_"]
    assert len(groups["pass0_"]) == 2
    assert len(groups["pass1_"]) == 1


def test_unprefixed_folders_group_together():
    """Pre-fix output, and a single-pass capture, both land in one group."""
    groups = group_by_pass([_Folder("0"), _Folder("1")])
    assert list(groups) == [""]


def test_the_same_leaf_in_two_passes_is_merged():
    """Pass 0 calls it 2, pass 1 calls it 5, and they claim the same points."""
    left = np.full(100, UNSEEN, np.int32)
    right = np.full(100, UNSEEN, np.int32)
    left[:40] = 2
    right[:40] = 5
    counts = {("pass0_", 2): 40, ("pass1_", 5): 40}

    merged = merge_across_passes({"pass0_": left, "pass1_": right}, counts, 0.30)
    assert merged[("pass0_", 2)] == merged[("pass1_", 5)], "one leaf became two"


def test_two_different_leaves_are_not_merged():
    left = np.full(100, UNSEEN, np.int32)
    right = np.full(100, UNSEEN, np.int32)
    left[:40] = 0            # one leaf
    right[60:] = 0           # a different leaf that happens to share an id
    counts = {("pass0_", 0): 40, ("pass1_", 0): 40}

    merged = merge_across_passes({"pass0_": left, "pass1_": right}, counts, 0.30)
    assert merged[("pass0_", 0)] != merged[("pass1_", 0)], "two leaves became one"


def test_partial_overlap_below_the_threshold_does_not_merge():
    """Leaves that touch are not the same leaf."""
    left = np.full(100, UNSEEN, np.int32)
    right = np.full(100, UNSEEN, np.int32)
    left[:40] = 0
    right[36:76] = 0                       # 4 points of 40 in common
    counts = {("pass0_", 0): 40, ("pass1_", 0): 40}

    merged = merge_across_passes({"pass0_": left, "pass1_": right}, counts, 0.30)
    assert merged[("pass0_", 0)] != merged[("pass1_", 0)]


def test_merging_is_transitive_across_three_passes():
    """A leaf seen in all three passes is one leaf, not three or two."""
    a = np.full(60, UNSEEN, np.int32)
    b = np.full(60, UNSEEN, np.int32)
    c = np.full(60, UNSEEN, np.int32)
    a[:30] = 1
    b[:30] = 4
    c[:30] = 9
    counts = {("pass0_", 1): 30, ("pass1_", 4): 30, ("pass2_", 9): 30}

    merged = merge_across_passes({"pass0_": a, "pass1_": b, "pass2_": c}, counts, 0.30)
    assert len(set(merged.values())) == 1


def test_combine_takes_the_majority_across_passes():
    merged = {("pass0_", 0): 0, ("pass1_", 0): 0, ("pass2_", 1): 1}
    per_pass = {
        "pass0_": np.array([0, SKELETON, UNSEEN], np.int32),
        "pass1_": np.array([0, SKELETON, SKELETON], np.int32),
        "pass2_": np.array([1, 1, UNSEEN], np.int32),
    }
    out = _combine_passes(per_pass, merged, 3)
    assert out[0] == 0, "two passes said leaf 0, one said leaf 1"
    assert out[2] == SKELETON, "the only pass that saw it said skeleton"


def test_a_point_no_pass_saw_stays_unseen():
    """Never manufactured into skeleton: 'nothing saw it' is not 'it is stem'."""
    per_pass = {"pass0_": np.array([UNSEEN], np.int32),
                "pass1_": np.array([UNSEEN], np.int32)}
    assert _combine_passes(per_pass, {}, 1)[0] == UNSEEN


def test_a_tie_prefers_the_more_specific_claim():
    """One pass says leaf, one says skeleton: the leaf is the real observation,
    the stem residual is what is left when no leaf was recognised."""
    merged = {("pass0_", 0): 0}
    per_pass = {"pass0_": np.array([0], np.int32),
                "pass1_": np.array([SKELETON], np.int32)}
    assert _combine_passes(per_pass, merged, 1)[0] == 0

    per_pass = {"pass0_": np.array([ROOT], np.int32),
                "pass1_": np.array([SKELETON], np.int32)}
    assert _combine_passes(per_pass, merged, 1)[0] == ROOT


def test_single_pass_is_unchanged_by_the_merge():
    """The one-pass case must behave exactly as it did before any of this."""
    labels = np.array([0, 1, SKELETON, UNSEEN, 1], np.int32)
    counts = {("", 0): 1, ("", 1): 2}
    merged = merge_across_passes({"": labels}, counts, 0.30)
    out = _combine_passes({"": labels}, merged or {("", 0): 0, ("", 1): 1}, len(labels))
    assert out[2] == SKELETON and out[3] == UNSEEN
    assert out[0] != out[1] and out[1] == out[4]


@pytest.mark.parametrize("n", [1, 12, 53, 120])
def test_every_leaf_gets_its_own_colour(n):
    """Two leaves sharing a colour read as one leaf in a 3D view."""
    palette = leaf_palette(n)
    assert len(palette) == n
    assert len({tuple(c) for c in palette}) == n
