"""The apex stage on a rosette: the leaves' origin, and a leaf whose blade starts at its base.

The user's look at the 06-10-2026 rosettes (2026-10-14): the fitted midribs floated above the
plant instead of meeting at its origin, and the fit drew chickweed's oval blade on a stalk where
sugar beet and thistle leaves run from the origin, narrow there and broader away from it.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import apex_regions                                                      # noqa: E402
import apex_select                                                       # noqa: E402


class _Run:
    """The two things rosette_origin reads: P5x's leaves, and its plant -> world transform."""

    def __init__(self, midribs):
        self.skeleton = {"leaves": [{"id": i, "midrib": m.tolist()} for i, m in enumerate(midribs)]}

    @staticmethod
    def to_world(points):
        return np.asarray(points, float)


def _leaf(origin, direction, gap, length=10.0, n=12):
    """A straight midrib pointing away from `origin`, its base `gap` out along `direction`."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    return np.asarray(origin) + d * (gap + np.linspace(0, length, n))[:, None]


def test_the_origin_is_where_the_leaves_base_lines_meet():
    """Bases 3-8 mm short of the origin (P5x traces a leaf from where its graph reaches it)."""
    o = np.array([1.0, 2.0, 3.0])
    dirs = [(1, 0, 0.3), (0, 1, 0.4), (-1, 0, 0.2), (0, -1, 0.5), (0.7, 0.7, 0.3)]
    run = _Run([_leaf(o, d, gap) for d, gap in zip(dirs, (3, 5, 8, 4, 6))])
    found, dist = apex_regions.rosette_origin(run)
    assert np.linalg.norm(found - o) < 1e-6
    assert dist.max() < 1e-6


def test_a_leaf_off_its_line_does_not_pull_the_origin():
    """One leaf held aside by the pliers: its base line misses the origin by 5 units."""
    o = np.zeros(3)
    dirs = [(1, 0, 0.3), (0, 1, 0.4), (-1, 0, 0.2), (0, -1, 0.5), (0.7, 0.7, 0.3), (-0.7, 0.7, 0.2)]
    mids = [_leaf(o, d, 4) for d in dirs]
    mids.append(_leaf(np.array([0.0, 0.0, -5.0]), (1, -1, 0.0), 4))   # aside, below the plant
    found, _ = apex_regions.rosette_origin(_Run(mids))
    assert np.linalg.norm(found - o) < 0.5, "a sum of distances, not of squares"


def test_too_few_leaves_give_no_origin():
    assert apex_regions.rosette_origin(_Run([_leaf(np.zeros(3), (1, 0, 0), 2)])) is None


def test_the_outline_starts_widest_where_the_plant_is():
    for w in (0.35, 0.5, 0.65):
        a, b = apex_select.outline_tapers(w)
        assert abs(a / (a + b) - w) < 1e-6
    a, b = apex_select.outline_tapers(0.05)                  # beyond the fit's bounds: clipped
    assert 0.5 <= a <= 2.0 and 0.5 <= b <= 2.0


def test_a_from_base_strip_has_no_stalk_and_its_widest_where_measured():
    curve = np.stack([np.linspace(0, 1, 40), np.zeros(40), np.zeros(40)], axis=1)
    old = apex_select.SHAPE
    try:
        apex_select.SHAPE = {"half_share": 0.15, "widest_at": 0.65}
        X = apex_select.strip(curve, 0.0).reshape(apex_select.ALONG, apex_select.ACROSS, 3)
    finally:
        apex_select.SHAPE = old
    half = np.abs(X[:, :, 1]).max(axis=1)
    s = np.linspace(0, 1, apex_select.ALONG)
    assert abs(s[np.argmax(half)] - 0.65) < 0.05
    assert half[3] > 0.1 * half.max(), "blade from the base: widening from the start, no stalk"
    assert abs(half.max() - 0.15 * 1.06) < 0.01


def _model(prior, fixed_origin):
    import torch                                                        # noqa: F401
    import apex_fit
    frame = {"node": np.zeros(3), "axis": np.array([0.0, 0.0, 1.0]), "e1": np.array([1.0, 0.0, 0.0]),
             "reach_mm": 20.0, "fixed_origin": fixed_origin}
    curve = np.stack([np.linspace(0, 10, 32), np.zeros(32), np.linspace(1, 4, 32)], axis=1)
    row = apex_fit.to_row(curve, frame, 1.0, 0.3, active=True)
    return apex_fit.Apex(frame, [row], 1.0, 0.05, "cpu", prior=prior, fixed_origin=fixed_origin), row


def test_a_rosette_leaf_starts_at_the_origin_with_no_stalk():
    model, row = _model({"half_share": 0.14, "widest_at": 0.65}, True)
    sh = model.shape()
    assert row["h_mm"] == 0.0, "the start row joins at O"
    assert float(sh["attach_mm"][0]) == 0.0 and float(sh["blade"][0]) == 0.0
    a, b = float(sh["a"][0]), float(sh["b"][0])
    assert abs(a / (a + b) - 0.65) < 1e-3, "widest where the plant's leaves are"
    assert abs(float(sh["half_share"][0]) - 0.14) < 1e-3


def test_the_origin_does_not_move_in_the_fit():
    import apex_fit
    model, _ = _model({"half_share": 0.14, "widest_at": 0.65}, True)
    opt_params = []
    real = apex_fit.torch.optim.Adam

    class Spy(real):
        def __init__(self, groups, *a, **k):
            opt_params.extend(p for g in groups for p in g["params"])
            super().__init__(groups, *a, **k)
    apex_fit.torch.optim.Adam = Spy
    try:
        apex_fit.fit(model, None, None, 0, None, "t")
    except Exception:
        pass                                                            # no views: only the setup is under test
    finally:
        apex_fit.torch.optim.Adam = real
    assert all(p is not model.dO for p in opt_params)


def test_a_chickweed_leaf_keeps_its_stalk_and_attach():
    model, _ = _model(None, False)
    sh = model.shape()
    assert float(sh["blade"][0]) > 0 and float(sh["attach_mm"][0]) > 0
