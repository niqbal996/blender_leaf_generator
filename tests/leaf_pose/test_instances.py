"""Detection and the full-resolution boundary.

The synthetic leaf here is deliberately crude -- a blob with a stalk on a
dark background -- because what these tests pin down is not how well the
matte follows a real margin, but the handful of places where the geometry
was wrong in a way that still produced a plausible-looking mask.
"""

import numpy as np
import pytest

from leaf_pose import instances


def leaf_mask(height=400, width=300, stalk_rows=(225, 295)):
    """An elliptical blade with a thin stalk running off the bottom.

    Set in a frame several times its own size, as a capture is: a leaf that
    fills its frame exercises neither the area rules nor the crop margins the
    way a flat lay does.
    """
    mask = np.zeros((height, width), bool)
    ys, xs = np.mgrid[0:height, 0:width]
    mask |= ((xs - width // 2) ** 2 / 34.0 ** 2
             + (ys - 170) ** 2 / 62.0 ** 2) < 1
    mask[stalk_rows[0]:stalk_rows[1], width // 2 - 3:width // 2 + 3] = True
    return mask


def scene(mask, leaf=(0.05, 0.30, 0.05), backing=0.004, seed=0):
    """The mask painted as a green leaf on a dark backing, linear RGB.

    With noise, and not as decoration: a perfectly uniform backing has a
    single-valued histogram, and Otsu's split of it lands exactly on that
    value -- so a noiseless fixture tests a case no photograph produces and
    fails on the boundary. The noise is multiplicative, like a sensor's.
    """
    rng = np.random.default_rng(seed)
    image = np.full(mask.shape + (3,), backing, np.float32)
    image[mask] = leaf
    return (image * rng.normal(1.0, 0.06, image.shape)).astype(np.float32)


def test_largest_filled_keeps_a_mask_that_touches_the_corner():
    """The hole fill must not read the whole crop as one enclosed hole.

    A leaf whose crop was clipped against the frame edge reaches (0, 0). The
    flood seed then starts inside the leaf, nothing is flooded, and every
    background pixel looks enclosed -- which returned the entire crop.
    """
    mask = np.zeros((40, 30), bool)
    mask[0:20, 0:12] = True
    assert instances.largest_filled(mask).sum() == mask.sum()


def test_largest_filled_closes_an_interior_hole():
    mask = np.zeros((40, 30), bool)
    mask[10:30, 8:20] = True
    mask[15:18, 12:15] = False
    assert instances.largest_filled(mask).sum() == mask.sum() + 9


def test_largest_filled_drops_a_detached_speck():
    mask = np.zeros((40, 30), bool)
    mask[10:30, 0:12] = True
    mask[2:5, 20:24] = True
    assert instances.largest_filled(mask).sum() == 240


def test_solidity_separates_a_blade_from_a_fibrous_spray():
    blade = leaf_mask()
    spray = np.zeros((200, 200), bool)
    for angle in np.linspace(0, np.pi, 9):
        for step in range(90):
            spray[100 + int(step * np.sin(angle)), 100 + int(step * np.cos(angle))] = True

    assert instances.solidity_of(blade) > instances.MIN_SOLIDITY
    assert instances.solidity_of(spray) < instances.MIN_SOLIDITY


def test_foreground_index_separates_leaf_from_backing():
    """Log luminance, not colour: the index must not care about the hue.

    The petiole case in one line -- a pale, barely green stalk has to land on
    the leaf side of the split, because the colour index put it on the other
    one and the stalk was cut off.
    """
    image = scene(leaf_mask())
    image[260:295, 147:153] = (0.10, 0.11, 0.09)  # a pale, nearly grey petiole

    index = instances.foreground_index(image)
    threshold = instances.foreground_threshold(index)
    assert index[170, 150] > threshold    # blade
    assert index[275, 150] > threshold    # petiole
    assert index[5, 5] < threshold       # backing


def test_detection_finds_one_leaf_and_rejects_a_grey_marker():
    image = scene(leaf_mask(), leaf=(0.05, 0.30, 0.05))
    image[20:60, 230:270] = 0.6  # a white fiducial marker

    report, scale = instances.detect_colour(image, max_side=400,
                                            min_area_fraction=1e-4)
    assert scale == 1.0
    assert len(report.accepted) == 1
    assert [d.rejected for d in report.rejected] == ["not green"]


def test_refinement_does_not_inflate_the_mask_by_the_crop_margin():
    """The coarse mask is placed in the crop, not stretched to fill it.

    Stretching it inflates the mask by the margin on every side -- which is
    ~15% for a big leaf and 2.4x for a small one, so the bug hid in exactly
    the leaves nobody checks first.
    """
    truth = leaf_mask()
    image = scene(truth)

    report, scale = instances.detect_colour(image, max_side=400,
                                            min_area_fraction=1e-4)
    refined = instances.refine_instance(image, report.accepted[0], scale, 1,
                                        margin=40)

    assert refined is not None
    assert refined.mask.sum() == pytest.approx(truth.sum(), rel=0.12)


def test_subpixel_contour_is_not_quantised_to_the_pixel_grid():
    alpha = np.zeros((40, 40), np.float32)
    alpha[10:30, 10:30] = 1.0
    alpha[9, 10:30] = alpha[30, 10:30] = 0.5  # a half-covered boundary row

    contour = instances.subpixel_contour(alpha)
    assert len(contour) > 4
    assert np.any(np.abs(contour - np.round(contour)) > 1e-6)
