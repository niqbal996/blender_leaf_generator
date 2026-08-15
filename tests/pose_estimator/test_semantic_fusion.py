"""Normal-weighted multi-view fusion (P4c).

The scenario these tests encode is the observed one: a 2D segmenter finds a
leaf when its blade faces the camera and mistakes the same leaf for a stem
when it turns edge-on. Over a full orbit that happens to every leaf, so most
views of any given leaf are wrong about it and plain majority voting loses.
"""

import numpy as np
import pytest

from pose_estimator.semantic import (
    LABEL_NONE,
    ViewCamera,
    _look_at,
    accumulate_votes,
    cast_votes,
    estimate_normals,
    finalise_votes,
    view_weights,
)

LEAF, STEM = 0, 1


def orbit_cameras(n=36, radius=6.0, resolution=256, focal=260.0):
    """Cameras on a horizontal ring looking at the origin -- the real rig."""
    cameras = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        eye = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
        K = np.array([[focal, 0, resolution / 2], [0, focal, resolution / 2], [0, 0, 1.0]])
        cameras.append(ViewCamera(K=K, world_to_camera=_look_at(eye, np.zeros(3)),
                                     width=resolution, height=resolution))
    return cameras


def leaf_patch(n=400, spread=0.8):
    """A flat blade in the y-z plane, so its normal points along +x."""
    rng = np.random.default_rng(0)
    points = np.zeros((n, 3))
    points[:, 1:] = rng.uniform(-spread, spread, size=(n, 2))
    normals = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    return points, normals


def test_weight_is_one_face_on_and_zero_edge_on():
    points, normals = leaf_patch(n=1)
    points[:] = 0.0

    face_on = _look_at(np.array([6.0, 0.0, 0.0]), np.zeros(3))
    edge_on = _look_at(np.array([0.0, 6.0, 0.0]), np.zeros(3))
    K = np.eye(3)

    def weight(w2c):
        cam = ViewCamera(K=K, world_to_camera=w2c, width=8, height=8)
        return float(view_weights(points, normals, cam)[0])

    assert weight(face_on) == pytest.approx(1.0, abs=1e-6)
    assert weight(edge_on) == pytest.approx(0.0, abs=1e-6)


def test_weight_ignores_which_face_is_turned_toward_the_camera():
    """Halfway round the orbit the blade presents its other side. That is the
    same geometry and must earn the same weight, which is why the cosine's
    sign is discarded."""
    points, normals = leaf_patch(n=1)
    points[:] = 0.0
    K = np.eye(3)

    front = ViewCamera(K=K, world_to_camera=_look_at(np.array([6.0, 0, 0]), np.zeros(3)),
                          width=8, height=8)
    back = ViewCamera(K=K, world_to_camera=_look_at(np.array([-6.0, 0, 0]), np.zeros(3)),
                         width=8, height=8)
    assert view_weights(points, normals, front)[0] == pytest.approx(
        view_weights(points, normals, back)[0], abs=1e-6)


def _run_orbit(weighted: bool, recognises_above=0.8):
    """Vote over a full orbit with a segmenter that only sees the blade when it
    is more than `recognises_above` broad-side, and calls it stem otherwise."""
    points, normals = leaf_patch()
    cameras = orbit_cameras()
    tally = accumulate_votes(len(points), 2)

    for camera in cameras:
        cosines = view_weights(points, normals, camera)
        # One pixel per point, so the index map is trivially exact.
        index_map = np.arange(len(points), dtype=np.int32)[None, :]
        class_map = np.where(cosines > recognises_above, LEAF, STEM).astype(np.int8)[None, :]
        cast_votes(tally, index_map, class_map, cosines if weighted else None)

    return finalise_votes(tally)


def test_plain_counting_loses_the_leaf_but_weighting_keeps_it():
    """At this tolerance the segmenter finds the blade in 14 of 36 views -- a
    clear minority -- so majority voting calls the leaf a stem. Weighting by
    obliquity gives those 14 views 13.2 of the 22.9 total weight, because they
    are the only ones looking at any appreciable amount of blade."""
    unweighted = _run_orbit(weighted=False)
    weighted = _run_orbit(weighted=True)

    assert (unweighted.labels == STEM).all(), "the premise -- most views say stem"
    assert (weighted.labels == LEAF).all(), "obliquity weighting recovers the blade"

    # Both readings come from one pass over the views, not two runs.
    assert (weighted.unweighted_labels == STEM).all()


def test_the_recovery_has_a_limit_and_it_is_where_it_should_be():
    """Weighting is not unlimited rescue. Majority voting needs the blade
    recognised in over half the views; weighting drops that to roughly a
    third, because a broad-side view outweighs a grazing one but does not
    outweigh arbitrarily many. Below that the evidence really is gone, and the
    fusion is expected to say so rather than invent a leaf.

    Asserted as fractions, not per point: the blade has extent, so points near
    its edge see the camera at a slightly different angle than points at its
    centre and cross the tolerance at slightly different moments. The boundary
    is genuinely soft and pretending otherwise would make the test lie. The
    measured crossover sits at a tolerance of about 0.9, where the blade is
    recognised in 10 of 36 views: leaf 100% at 0.85, 16% at 0.9, 0% at 0.95."""
    assert (_run_orbit(weighted=True, recognises_above=0.85).labels == LEAF).mean() > 0.95
    assert (_run_orbit(weighted=True, recognises_above=0.95).labels == STEM).mean() > 0.95


def test_edge_on_views_abstain_rather_than_being_discarded():
    """A grazing view still contributes -- just negligibly. Nothing is dropped,
    so a point seen only edge-on is still labelled, only with low confidence."""
    result = _run_orbit(weighted=True)
    assert (result.n_views_seen == 36).all(), "every view was counted as having seen the point"
    assert (result.weight.sum(axis=1) < result.count.sum(axis=1)).all(), "but weighted down"
    assert (result.labels != LABEL_NONE).all()


def test_confidence_reflects_agreement_among_views_that_could_see():
    weighted = _run_orbit(weighted=True)
    unweighted = _run_orbit(weighted=False)
    assert weighted.confidence.mean() > unweighted.confidence.mean()


def test_estimated_normals_recover_a_known_plane():
    points, normals = leaf_patch(n=600)
    estimated = estimate_normals(points, k=12)
    alignment = np.abs(np.einsum("ij,ij->i", estimated, normals))
    assert np.median(alignment) > 0.99


def test_class_indices_beyond_the_tally_are_ignored_not_miscounted():
    """A four-class seed set against a three-class tally must drop the extra
    rather than silently fold it into class 0."""
    tally = accumulate_votes(4, 3)
    index_map = np.array([[0, 1, 2, 3]], np.int32)
    class_map = np.array([[0, 1, 2, 3]], np.int8)
    cast_votes(tally, index_map, class_map)
    assert tally["count"][3].sum() == 0
    assert tally["count"][0][0] == 1 and tally["count"][2][2] == 1
