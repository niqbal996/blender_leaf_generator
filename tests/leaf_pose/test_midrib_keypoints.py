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


LONG_FLANK, SHORT_FLANK = 26.0, 9.0


def serrated_blade(height=400, width=200, teeth=9, depth=14, apex_toward_high_row=True):
    """An elliptical blade with teeth pointing toward one end. No petiole.

    Built by carving triangular sinuses out of the margin rather than by
    adding triangles to it, so the blade stays one simple region. The sign
    convention is the easy thing to get backwards, so it is spelled out:

    A tooth points toward the apex means that, walking that way, the margin
    climbs slowly along the tooth's long proximal flank and then drops
    sharply down its short distal flank into the next sinus. The margin is
    `base - reach`, so `reach` must do the opposite: fall slowly away from a
    sinus in the direction the teeth point, and rise sharply back into the
    next one. That puts the sinus's **long** taper on the side the teeth
    point toward.
    """
    mask = np.zeros((height, width), bool)
    ys, xs = np.mgrid[0:height, 0:width]
    mask |= ((xs - 100) ** 2 / 55.0 ** 2 + (ys - 200) ** 2 / 185.0 ** 2) < 1

    for centre in np.linspace(60, 340, teeth):
        for row in range(int(centre) - 30, int(centre) + 30):
            if not 0 <= row < height:
                continue
            delta = row - centre
            toward = (delta >= 0) if apex_toward_high_row else (delta < 0)
            reach = depth * max(0.0, 1.0 - abs(delta)
                                / (LONG_FLANK if toward else SHORT_FLANK))
            if reach <= 0:
                continue
            columns = np.flatnonzero(mask[row])
            if len(columns) < 4:
                continue
            mask[row, columns[0]:columns[0] + int(reach)] = False
            mask[row, columns[-1] - int(reach):columns[-1] + 1] = False
    return mask


def keypoints_contour(mask):
    from leaf_pose.instances import subpixel_contour
    return subpixel_contour(mask.astype(np.float32))


def test_tooth_direction_reads_which_way_the_teeth_point():
    high = serrated_blade(apex_toward_high_row=True)
    low = serrated_blade(apex_toward_high_row=False)

    high_fit = midrib.fit_midrib(high, num_samples=48)
    low_fit = midrib.fit_midrib(low, num_samples=48)

    first = keypoints.tooth_direction(keypoints_contour(high), high_fit.path)
    second = keypoints.tooth_direction(keypoints_contour(low), low_fit.path)
    assert np.sign(first) == -np.sign(second)
    assert abs(first) > 0.3 and abs(second) > 0.3


def test_tooth_direction_is_antisymmetric_under_reversal():
    """Reversing the leaf must negate the statistic exactly, or the vote
    would favour one orientation by construction rather than by evidence."""
    mask = serrated_blade()
    fitted = midrib.fit_midrib(mask, num_samples=48)
    contour = keypoints_contour(mask)

    forward = keypoints.tooth_direction(contour, fitted.path)
    backward = keypoints.tooth_direction(contour, fitted.path[::-1])
    assert forward == pytest.approx(-backward, abs=0.05)


def test_an_entire_margin_abstains_rather_than_voting_on_noise():
    mask = np.zeros((400, 200), bool)
    ys, xs = np.mgrid[0:400, 0:200]
    mask |= ((xs - 100) ** 2 / 55.0 ** 2 + (ys - 200) ** 2 / 185.0 ** 2) < 1

    fitted = midrib.fit_midrib(mask, num_samples=48)
    assert abs(keypoints.tooth_direction(keypoints_contour(mask), fitted.path)) < 0.5


def test_teeth_decide_a_leaf_with_no_petiole_at_all():
    """The case this vote exists for: nothing else has anything to read."""
    mask = serrated_blade(apex_toward_high_row=True)
    fitted = midrib.fit_midrib(mask, num_samples=48)
    contour = keypoints_contour(mask)

    _, blind = keypoints.locate(fitted)
    _, told = keypoints.locate(fitted, contour=contour)

    assert blind.confidence < 0.12          # no stalk, no colour: a coin toss
    assert told.confidence > blind.confidence
    # Teeth point toward increasing row here, so the tip is the high-row end.
    assert told.tip[1] > told.petiole_origin[1]
    assert told.tooth_skew < 0              # reported in the final orientation
