"""Where the midrib runs, and which end of it was joined to the stem.

Both are measured against a synthetic leaf whose answer is known by
construction: an elliptical blade with a stalk at one end, and -- for the
tests that need image evidence -- a bright rib painted down a known column.
"""

import numpy as np
import pytest

from leaf_pose import keypoints, midrib


def blade_with_stalk(height=400, width=200, stalk_at_top=False):
    mask = np.zeros((height, width), bool)
    ys, xs = np.mgrid[0:height, 0:width]
    mask |= ((xs - 100) ** 2 / 60.0 ** 2 + (ys - 140) ** 2 / 120.0 ** 2) < 1
    mask[250:380, 96:104] = True
    return mask[::-1].copy() if stalk_at_top else mask


def painted(mask, rib_column=100, rib_halfwidth=3):
    image = np.zeros(mask.shape + (3,), np.float32)
    image[mask] = 0.30
    ribbon = np.zeros(mask.shape, bool)
    ribbon[:, rib_column - rib_halfwidth:rib_column + rib_halfwidth] = True
    image[mask & ribbon] = 0.60
    return image


def test_midrib_spans_the_leaf_end_to_end():
    fitted = midrib.fit_midrib(blade_with_stalk(), num_samples=48)

    assert fitted is not None
    # One end at the far tip of the stalk (y~379), the other at the top of
    # the blade (y~21). The arclength is the leaf's full extent, not a chord.
    ends = sorted([fitted.path[0][1], fitted.path[-1][1]])
    assert ends[0] < 30 and ends[1] > 370
    assert fitted.arclength == pytest.approx(358, abs=15)


def test_width_along_recovers_the_blade_width():
    fitted = midrib.fit_midrib(blade_with_stalk(), num_samples=48)
    assert fitted.width.max() == pytest.approx(120, abs=8)  # 2 * semi-axis
    assert fitted.width.min() < 12                          # the stalk


def test_ridge_evidence_pulls_the_path_onto_a_painted_rib():
    """A rib painted off-centre must move the midrib, not be ignored.

    The mask's own medial axis runs down the middle of the blade; the rib is
    the thing the photograph knows about and the mask does not. If the ridge
    term were inert this test would still pass at column 100.
    """
    mask = blade_with_stalk()
    off_centre = painted(mask, rib_column=120)

    with_image = midrib.fit_midrib(mask, image=off_centre, ridge_weight=0.9,
                                   num_samples=48)
    without = midrib.fit_midrib(mask, num_samples=48)

    middle = slice(12, 36)  # the blade, away from the stalk that anchors both
    assert with_image.path[middle, 0].mean() > without.path[middle, 0].mean() + 4


def test_geodesic_endpoints_are_the_two_ends_not_a_lobe():
    """A big basal lobe must not out-rank the tip as an extremity."""
    mask = blade_with_stalk()
    ys, xs = np.mgrid[0:400, 0:200]
    mask |= ((xs - 40) ** 2 / 34.0 ** 2 + (ys - 215) ** 2 / 22.0 ** 2) < 1

    first, second = midrib.geodesic_endpoints(mask)
    rows = sorted([first[0], second[0]])
    assert rows[0] < 40 and rows[1] > 360


def test_petiole_end_is_found_whichever_way_the_leaf_lies():
    for stalk_at_top in (False, True):
        mask = blade_with_stalk(stalk_at_top=stalk_at_top)
        oriented, found = keypoints.locate(midrib.fit_midrib(mask, num_samples=48))

        # Station 0 is the cut end, always.
        assert oriented.path[0][1] == pytest.approx(379 if not stalk_at_top else 20,
                                                    abs=25)
        assert found.flipped is stalk_at_top
        assert found.petiole_length == pytest.approx(130, abs=40)
        assert found.blade_length > found.petiole_length
        assert found.confidence > 0.2


def test_orientation_is_reported_as_drawn():
    """A leaf pointing up the frame reads as +90, not -90."""
    _, found = keypoints.locate(midrib.fit_midrib(blade_with_stalk(),
                                                  num_samples=48))
    assert found.orientation_deg == pytest.approx(90, abs=12)


def test_colour_breaks_a_tie_the_width_profile_cannot():
    """A blade with no stalk at all: only the pale base says which end it is."""
    mask = np.zeros((400, 200), bool)
    ys, xs = np.mgrid[0:400, 0:200]
    mask |= ((xs - 100) ** 2 / 55.0 ** 2 + (ys - 200) ** 2 / 190.0 ** 2) < 1

    fitted = midrib.fit_midrib(mask, num_samples=48)
    greenness = np.where(mask, 0.9, 0.0).astype(np.float32)
    greenness[330:, :] = np.where(mask[330:, :], 0.2, 0.0)  # pale at high y

    _, blind = keypoints.locate(fitted)
    _, told = keypoints.locate(fitted, greenness=greenness)

    assert blind.confidence < 0.1                 # genuinely ambiguous on shape
    assert told.petiole_origin[1] > 300           # the pale end
    assert told.confidence > blind.confidence


def test_stalk_run_is_zero_when_the_profile_starts_thick():
    assert keypoints.stalk_run(np.array([10.0, 10.0, 10.0])) == 0.0
    assert keypoints.stalk_run(np.array([1.0, 1.0, 10.0, 10.0])) == 0.5


def ovate_with_petiole(height=420, width=200, petiole_rows=(250, 400),
                       petiole_half=3):
    """An entire-margined ovate blade on a long thin petiole.

    Stellaria, roughly: no teeth anywhere, an acute apex, and a stalk that is
    a small fraction of the blade's width. This is the shape a margin-asymmetry
    statistic gets wrong, because the only strong asymmetry on it is the step
    where the blade narrows into the stalk.
    """
    mask = np.zeros((height, width), bool)
    ys, xs = np.mgrid[0:height, 0:width]
    mask |= ((xs - 100) ** 2 / 62.0 ** 2 + (ys - 150) ** 2 / 120.0 ** 2) < 1
    mask[petiole_rows[0]:petiole_rows[1],
         100 - petiole_half:100 + petiole_half] = True
    return mask


def test_an_entire_margined_leaf_on_a_long_petiole_is_read_from_its_stalk():
    """The vogelmeere regression.

    A margin-teeth vote was added here and removed: on a leaf with no teeth
    it reads the blade-to-petiole step instead, and that always points the
    "tip" at the stalk. It inverted six of fifteen Stellaria leaves whose
    petioles were unmistakable. The stalk must win on this shape, every time.
    """
    mask = ovate_with_petiole()
    fitted = midrib.fit_midrib(mask, num_samples=48)
    oriented, found = keypoints.locate(fitted)

    # The stalk runs to high rows, so the petiole origin belongs there.
    assert found.petiole_origin[1] > found.tip[1]
    assert found.petiole_origin[1] > 380
    assert found.petiole_length > 100
    assert found.confidence > 0.2
    assert oriented.width[0] < oriented.width.max() * keypoints.STALK_WIDTH_FRACTION


def test_locate_takes_no_margin_argument():
    """The tooth cue is gone, not merely disabled. A caller still passing a
    contour would otherwise silently get the old, wrong behaviour back."""
    import inspect

    assert "contour" not in inspect.signature(keypoints.locate).parameters
    assert not hasattr(keypoints, "tooth_direction")


# --------------------------------------------------------------------------
# The ridge filter must not cost more because the camera has more pixels.
# --------------------------------------------------------------------------


def _synthetic_leaf(height, width, rib_sigma=14.0):
    yy, xx = np.mgrid[0:height, 0:width]
    mask = (((yy - height / 2) / (height * 0.42)) ** 2
            + ((xx - width / 2) / (width * 0.47)) ** 2) < 1.0
    grey = np.where(mask, 60.0, 8.0)
    grey = grey + 90.0 * np.exp(-((yy - height / 2) ** 2) / (2 * rib_sigma ** 2)) * mask
    return mask, np.repeat(grey[:, :, None], 3, axis=2).astype(np.float32)


def test_ridge_response_is_unchanged_by_the_downsampling():
    """`sato` costs pixels x sigma, and sigma is a fraction of the leaf's
    half-width -- so a high-resolution flat-lay drives both up at once and the
    stage takes hours. Filtering at sigma/k on a k-times smaller image is the
    same scale-space location, and this pins that it really is."""
    from skimage.morphology import medial_axis

    from leaf_pose import midrib as rib

    mask, image = _synthetic_leaf(600, 900)
    _, distance = medial_axis(mask, return_distance=True)
    assert max(0.40 * distance.max(), 1.0) > rib.MAX_FILTER_SIGMA, \
        "this leaf is too small to exercise the downsampling path"

    fast = rib.ridge_response(image, distance, mask)

    original = rib.MAX_FILTER_SIGMA
    try:
        rib.MAX_FILTER_SIGMA = 1e9          # force the full-resolution path
        exact = rib.ridge_response(image, distance, mask)
    finally:
        rib.MAX_FILTER_SIGMA = original

    correlation = np.corrcoef(fast[mask], exact[mask])[0, 1]
    assert correlation > 0.98, f"downsampled response diverged (r={correlation:.3f})"
    # And both must still put the ridge where the ridge actually is.
    middle = image.shape[1] // 2
    assert abs(int(fast[:, middle].argmax()) - image.shape[0] // 2) <= 8
    assert abs(int(exact[:, middle].argmax()) - image.shape[0] // 2) <= 8


def test_small_leaves_take_the_exact_path_unchanged():
    """A leaf whose sigmas are already small must not be resampled at all."""
    from skimage.morphology import medial_axis

    from leaf_pose import midrib as rib

    mask, image = _synthetic_leaf(90, 140, rib_sigma=3.0)
    _, distance = medial_axis(mask, return_distance=True)
    assert 0.40 * distance.max() <= rib.MAX_FILTER_SIGMA

    response = rib.ridge_response(image, distance, mask)
    assert response.shape == mask.shape
    assert response.max() > 0
