"""A rosette has no stem: P5x writes its stem votes as unassigned leaf tissue, and the tracer still
stands the crown on that tissue (cli/leaf_instances.no_stem_on_rosette, leaf_skeleton.find_crown)."""
import numpy as np

from pose_estimator.cli.leaf_instances import SKELETON, UNSEEN, no_stem_on_rosette
from pose_estimator.leaf_skeleton import UNASSIGNED, find_crown


def test_a_rosettes_stem_votes_become_unassigned_leaf_tissue():
    a = np.array([0, 1, SKELETON, SKELETON, UNSEEN])
    assert no_stem_on_rosette(a, "rosette") == 2
    assert (a == SKELETON).sum() == 0 and (a == UNASSIGNED).sum() == 2
    assert list(a[:2]) == [0, 1] and a[4] == UNSEEN


def test_an_upright_plant_keeps_its_stem():
    a = np.array([0, SKELETON, SKELETON])
    assert no_stem_on_rosette(a, "upright") == 0
    assert (a == SKELETON).sum() == 2


def test_the_crown_stands_on_unassigned_tissue_as_on_stem():
    rng = np.random.default_rng(0)
    base = rng.normal([0, 0, 1], 0.1, (200, 3))          # the tissue at the foot
    leaf = rng.normal([0, 0, 5], 0.5, (200, 3))
    points = np.vstack([base, leaf])
    stem = np.r_[np.full(200, SKELETON), np.zeros(200, int)]
    unassigned = np.r_[np.full(200, UNASSIGNED), np.zeros(200, int)]
    np.testing.assert_allclose(find_crown(points, stem), find_crown(points, unassigned))
