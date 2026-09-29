"""P3's rotating-region mask keeps the camera-mounted backdrop out.

Measured on maize_1 (2026-09-28), side pass: room visible past the backdrop's
left edge sweeps with the orbit, so the top-left pixel reads as moving. The
old flood fill seeded at (0, 0) then filled nothing, every backdrop pixel
became a "hole", the mask covered 100% of the frame, and ~27% of that pass's
triangulated observations landed on the backdrop -- the pass's cameras
collapsed toward one point.
"""

import numpy as np

from pose_estimator.pose import _fill_holes


def test_backdrop_stays_out_when_corner_is_moving():
    mask = np.zeros((60, 80), np.uint8)
    mask[:, :4] = 255          # room past the backdrop edge, touching (0, 0)
    mask[20:50, 30:50] = 255   # the plant

    filled = _fill_holes(mask)

    assert filled[10, 60] == 0, "backdrop was swallowed into the mask"
    assert (filled > 0).mean() < 0.5


def test_interior_hole_is_still_filled():
    mask = np.zeros((60, 80), np.uint8)
    mask[10:50, 20:60] = 255
    mask[25:35, 35:45] = 0     # untextured disc patch inside the swept region

    filled = _fill_holes(mask)

    assert filled[30, 40] == 255
    assert filled[5, 5] == 0


def test_background_cut_off_from_corner_is_not_a_hole():
    # A moving band splits the frame: background on both sides touches the
    # border, so neither side is an interior hole.
    mask = np.zeros((60, 80), np.uint8)
    mask[:, 38:42] = 255

    filled = _fill_holes(mask)

    assert filled[30, 10] == 0
    assert filled[30, 70] == 0
