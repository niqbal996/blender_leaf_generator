"""The two pure pieces of P4g that fail silently: the backdrop test and consolidation."""

import numpy as np

from pose_estimator.gs_surface import backdrop_pixels, consolidate


def _scene(rng):
    """A noisy black backdrop with a green blade split by a 4 px backdrop gap."""
    img = rng.integers(10, 35, size=(200, 200, 3)).astype(np.uint8)       # dark, desaturated
    img[60:140, 60:140] = (60, 150, 50)                                     # the blade (RGB)
    img[60:140, 98:102] = rng.integers(10, 35, size=(80, 4, 3))             # a gap SAM3 filled
    maybe = np.zeros((200, 200), np.uint8)
    maybe[50:150, 50:150] = 1                                               # the (dilated) mask
    return img, maybe, np.ones_like(maybe)


def test_backdrop_opens_the_gap_and_spares_the_blade():
    img, maybe, evidence = _scene(np.random.default_rng(0))
    found = backdrop_pixels(img, maybe, evidence, max_share=0.5)
    assert found[60:140, 98:102].mean() > 0.95          # the gap is backdrop
    assert found[60:140, 60:95].mean() == 0             # the blade is not


def test_bright_objects_on_the_backdrop_do_not_widen_it():
    rng = np.random.default_rng(1)
    img, maybe, evidence = _scene(rng)
    img[0:30, 0:40] = (250, 250, 250)                   # an AprilTag
    img[160:200, 150:200] = (240, 200, 20)              # the plier handle
    found = backdrop_pixels(img, maybe, evidence, max_share=0.5)
    assert found[60:140, 60:95].mean() == 0


def test_a_view_where_the_test_would_take_too_much_is_not_trusted():
    img, maybe, evidence = _scene(np.random.default_rng(2))
    assert backdrop_pixels(img, maybe, evidence, max_share=0.01).sum() == 0


def test_consolidate_averages_per_voxel():
    pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.5, 0.1, 0.1]])
    nrm = np.array([[0, 0, 1.0], [0, 0, 1.0], [1.0, 0, 0]])
    col = np.array([[0, 0, 0], [200, 100, 50], [10, 10, 10]], np.uint8)
    p, n, c = consolidate(pts, nrm, col, voxel=1.0)
    order = np.argsort(p[:, 0])
    assert len(p) == 2
    assert np.allclose(p[order[0]], [0.15, 0.15, 0.15])
    assert c[order[0]].tolist() == [100, 50, 25]
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
