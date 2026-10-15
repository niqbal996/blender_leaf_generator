"""P5x on a two-skinned cloud: a leaf's two faces are one leaf, and two passes must not tie.

P4g draws a leaf as a closed shell -- a top and a bottom skin 1-4 mm apart that meet at
the margin -- and a view votes only for the skin facing it. Measured on the 06-10-2026
session (2026-10-14): SAM3 renamed a thistle leaf when it turned its underside to the
camera, and the two ids, one per skin, became two leaves; a weed_x leaf came out as two
leaves on its two skins; and with two capture passes every disagreement was a 1:1 tie
handed to the lower leaf id (37% of weed_x's leaf points), so leaves spread over their
neighbours and lost their tips. These pin the three fixes.
"""

import numpy as np

from pose_estimator.cli.leaf_instances import (
    ROOT,
    SKELETON,
    UNSEEN,
    _pool_passes,
    face_links,
    merge_across_passes,
    skin_partners,
)


def _sheet(z, n=24, step=0.1):
    x, y = np.meshgrid(np.arange(n) * step, np.arange(n) * step)
    return np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], axis=1)


def _shell_and_a_far_leaf():
    """Two skins 1.0 apart (one leaf), and a third sheet 4.0 above the top skin."""
    bottom, top, far = _sheet(0.0), _sheet(1.0), _sheet(5.0)
    return np.vstack([bottom, top, far]), len(bottom)


def test_a_skin_pairs_with_the_other_skin_of_its_leaf():
    points, n = _shell_and_a_far_leaf()
    partners = skin_partners(points, np.ones(len(points), bool))
    bottom, top = partners[:n], partners[n:2 * n]
    inner = np.arange(n)[(points[:n, 0] > 0.3) & (points[:n, 0] < 2.0)
                         & (points[:n, 1] > 0.3) & (points[:n, 1] < 2.0)]
    assert (bottom[inner] >= n).all() and (bottom[inner] < 2 * n).all(), "bottom skin -> top skin"
    assert (top[inner] >= 0).all() and (top[inner] < n).all(), "top skin -> bottom skin, not the far leaf"


def test_a_skin_facing_another_leaf_across_air_keeps_no_partner():
    """The far sheet's first hit is the top skin, whose own first hit is the bottom skin."""
    points, n = _shell_and_a_far_leaf()
    partners = skin_partners(points, np.ones(len(points), bool))
    assert (partners[2 * n:] == -1).all()


def test_unseen_points_take_no_part():
    """Points no view saw (inside the shell) neither stop the search nor get a partner."""
    points, n = _shell_and_a_far_leaf()
    inside = _sheet(0.5)
    cloud = np.vstack([points, inside])
    seen = np.r_[np.ones(len(points), bool), np.zeros(len(inside), bool)]
    partners = skin_partners(cloud, seen)
    assert (partners[len(points):] == -1).all()
    assert (partners[:n][(cloud[:n, 0] > 0.3) & (cloud[:n, 1] > 0.3)
                         & (cloud[:n, 0] < 2.0) & (cloud[:n, 1] < 2.0)] >= n).all()


def test_the_two_faces_of_one_leaf_are_linked():
    """thistle_x pass 0: id 4 on the top skin, id 8 on the bottom -- one leaf."""
    labels = np.array([4, 4, 4, 8, 8, 8], np.int32)
    partners = np.array([3, 4, 5, 0, 1, 2])
    counts = {("pass0_", 4): 3, ("pass0_", 8): 3}
    links = face_links({"pass0_": labels}, partners, counts, 0.30)
    assert [(a, b) for _, _, a, b in links] == [(("pass0_", 4), ("pass0_", 8))]
    assert links[0][0] == 1.0


def test_the_share_is_of_one_id_not_both_ways_added():
    """A small id on part of a big one's other face: its own share, not an inflated sum."""
    labels = np.array([0] * 10 + [1] * 4, np.int32)
    partners = np.full(14, -1)
    partners[10:12], partners[0:2] = [0, 1], [10, 11]       # 2 of id 1's 4 points face id 0
    counts = {("p", 0): 10, ("p", 1): 4}
    (share, _, _, _), = face_links({"p": labels}, partners, counts, 0.30)
    assert share == 0.5


def test_a_face_link_joins_two_ids_sam3_showed_apart():
    """With the cannot-link pair lifted, the face link merges them."""
    a = np.array([0, 0, 1, 1], np.int32)
    counts = {("p", 0): 2, ("p", 1): 2}
    merged = merge_across_passes({"p": a}, counts, 0.30,
                                 extra_links=[(1.0, 2, ("p", 0), ("p", 1))])
    assert merged[("p", 0)] == merged[("p", 1)]


def test_a_face_link_still_yields_to_a_stronger_cross_pass_link():
    """weed_x: pass 1's id 13 faces leaf 5 (38%) but is leaf 2's in pass 0 (52%)."""
    p0 = np.full(20, UNSEEN, np.int32)
    p1 = np.full(20, UNSEEN, np.int32)
    p0[:10], p0[10:20] = 2, 7                     # two pass-0 leaves SAM3 showed apart
    p1[:10], p1[10:20] = 13, 0
    counts = {("pass0_", 2): 10, ("pass0_", 7): 10, ("pass1_", 13): 10, ("pass1_", 0): 10}
    forbid = {frozenset((("pass0_", 2), ("pass0_", 7)))}
    merged = merge_across_passes({"pass0_": p0, "pass1_": p1}, counts, 0.30, cannot_link=forbid,
                                 extra_links=[(0.38, 4, ("pass1_", 0), ("pass1_", 13))])
    assert merged[("pass1_", 13)] == merged[("pass0_", 2)]
    assert merged[("pass1_", 0)] == merged[("pass0_", 7)]
    assert merged[("pass0_", 2)] != merged[("pass0_", 7)]


def test_pooled_votes_do_not_hand_a_disagreement_to_the_lower_id():
    """Pass 0 is sure it is leaf 5 (weight 0.9), pass 1 weakly says leaf 0 (0.2)."""
    merged = {("pass0_", 0): 5, ("pass1_", 0): 0, ("pass0_", 1): 0}
    # columns: the pass's ids, then stem, then root
    w0 = np.array([[0.9, 0.0, 0.0, 0.0]], np.float32)       # pass0: ids 0, 1 -> leaves 5, 0
    w1 = np.array([[0.2, 0.0, 0.0]], np.float32)            # pass1: id 0 -> leaf 0
    out = _pool_passes({"pass0_": w0, "pass1_": w1}, merged, 1)
    assert out[0] == 5


def test_pooled_votes_keep_stem_root_and_unseen():
    merged = {("p", 0): 0}
    w = np.array([[0.1, 0.5, 0.0], [0.1, 0.0, 0.7], [0.0, 0.0, 0.0]], np.float32)
    out = _pool_passes({"p": w}, merged, 3)
    assert out.tolist() == [SKELETON, ROOT, UNSEEN]
