"""The root check must survive a holder mask that is the whole tool.

`root_tracked_below_the_jaws` locates the jaws as a row in the image and
counts plant pixels below it. It read that row from the mean of the *entire*
holder mask, which is the jaw tip under the SAM2 backend and the whole pair of
pliers -- handles included, lying across the turntable -- under SAM3. The
pliers' mean row sits below the root, so every root pixel measured as being
above the jaws and a root tracked in 11 of 20 frames was reported as tracked
in 0, on masks that were on disk and correct.

The fix reads the row from the holder pixels touching the plant, which is the
grip under either backend. These tests pin both directions: the whole-tool
mask must now find the root, and the jaw-tip mask must keep the answer it
already gave.
"""

import cv2
import numpy as np
import pytest

from pose_estimator.segmentation_qc import run_qc


FRAME_H, FRAME_W = 400, 300
CROWN_ROW = 200          # where the tool grips
ROOT_ROWS = slice(210, 260)
FOLIAGE_ROWS = slice(40, 200)


def _write(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def _capture(tmp_path, holder_mask, frames=6):
    """A plant with foliage above the crown and a root below it."""
    frames_dir, p2 = tmp_path / "p1" / "frames", tmp_path / "p2"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(frames):
        stem = f"frame_{i:04d}"
        cv2.imwrite(str(frames_dir / f"{stem}.jpg"),
                    np.zeros((FRAME_H, FRAME_W, 3), np.uint8))

        plant = np.zeros((FRAME_H, FRAME_W), bool)
        plant[FOLIAGE_ROWS, 100:200] = True      # foliage, above the grip
        plant[ROOT_ROWS, 140:160] = True         # root, below the grip
        _write(p2 / "masks" / "plant" / f"{stem}.png", plant)
        _write(p2 / "masks" / "holder" / f"{stem}.png", holder_mask)
    return frames_dir, p2, {f"frame_{i:04d}": 0 for i in range(frames)}


def _jaw_tip_mask():
    """SAM2's holder: a small blob at the crown, touching the plant."""
    holder = np.zeros((FRAME_H, FRAME_W), bool)
    holder[CROWN_ROW - 5:CROWN_ROW + 5, 130:170] = True
    return holder


def _whole_tool_mask():
    """SAM3's holder: the same jaws plus handles sprawling down the frame."""
    holder = _jaw_tip_mask()
    holder[300:380, 20:280] = True               # handles, well below the root
    return holder


def test_whole_tool_holder_still_finds_the_root(tmp_path):
    frames_dir, p2, sources = _capture(tmp_path, _whole_tool_mask())
    report = run_qc(frames_dir, p2, sources=sources)

    summary = report["root_below_jaws_per_pass"]["0"]
    assert summary["frames_with_root"] == summary["num_frames"], (
        "the handles dragged the jaw row below the root, hiding tissue that is "
        "in the mask")
    assert report["checks"]["root_tracked_below_the_jaws"]["pass"]


def test_jaw_tip_holder_is_unchanged(tmp_path):
    """The SAM2 reading must not move: this fix may refine it, never flip it."""
    frames_dir, p2, sources = _capture(tmp_path, _jaw_tip_mask())
    report = run_qc(frames_dir, p2, sources=sources)

    summary = report["root_below_jaws_per_pass"]["0"]
    assert summary["frames_with_root"] == summary["num_frames"]
    assert report["checks"]["root_tracked_below_the_jaws"]["pass"]


def test_holder_far_from_the_plant_falls_back_to_the_whole_mask(tmp_path):
    """Nothing touches the plant, so there is no grip to measure -- the old
    reading is the only one available and must still be produced."""
    holder = np.zeros((FRAME_H, FRAME_W), bool)
    holder[300:380, 20:280] = True               # a tool lying apart from the plant
    frames_dir, p2, sources = _capture(tmp_path, holder)
    report = run_qc(frames_dir, p2, sources=sources)

    # The jaw row is now the handles at ~row 340, so the root at 210-260 reads
    # as above it. That is the honest answer for this geometry, and the point
    # of the test is that it is produced rather than crashing on an empty grip.
    assert "0" in report["root_below_jaws_per_pass"]


def test_no_holder_mask_reports_that_it_cannot_locate_the_jaws(tmp_path):
    frames_dir, p2, sources = _capture(tmp_path, np.zeros((FRAME_H, FRAME_W), bool))
    report = run_qc(frames_dir, p2, sources=sources)
    check = report["checks"]["root_tracked_below_the_jaws"]
    assert check["pass"], "an absent holder is not evidence of a lost root"
