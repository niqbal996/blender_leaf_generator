"""Tissue that is segmented but never reconstructed must fail a check.

Both runs in this repo lost the exposed root, and every acceptance check
passed it through. P2 reported clean masks, P4a a plausible hull, and P4c
"all_classes_present: leaf 57151, root 4, stem 1720" -- because 4 > 0. The
root was in 727,000 classified pixels and 4 points of 58,875.

The two checks here close that gap at the two places the evidence exists:
P2 knows whether the root is in the mask often enough to survive carving,
and P4c knows whether a class the 2D classifier found in quantity arrived
in 3D at all.
"""

import numpy as np
import pytest

from pose_estimator.cli.fuse import _survival_check


CLASSES = ["leaf", "root", "stem"]


def test_class_segmented_in_2d_but_absent_in_3d_fails():
    """thistle1's actual numbers: 6.8% of pixels, 4 points of 58,875."""
    check = _survival_check(
        CLASSES,
        {"leaf": 57151, "root": 4, "stem": 1720},
        58875,
        {"leaf": 9723508, "root": 727000, "stem": 286215},
    )
    assert not check["pass"]
    assert "root" in check["detail"]


def test_classes_that_transferred_pass():
    """3D shares close to the 2D shares is the healthy case."""
    check = _survival_check(
        CLASSES,
        {"leaf": 5700, "root": 400, "stem": 200},
        6300,
        {"leaf": 900000, "root": 60000, "stem": 40000},
    )
    assert check["pass"]


def test_a_class_the_classifier_barely_saw_is_not_held_against_3d():
    """Below the 2D share floor there is not enough evidence to judge."""
    check = _survival_check(
        CLASSES,
        {"leaf": 6000, "root": 0, "stem": 300},
        6300,
        {"leaf": 900000, "root": 200, "stem": 40000},
    )
    assert check["pass"]


def test_survival_check_is_inert_without_2d_counts():
    """An older run has no pixels_per_class; that is not a failure."""
    assert _survival_check(CLASSES, {"leaf": 1, "root": 0, "stem": 0}, 1, None)["pass"]


# --------------------------------------------------------------------------
# P2: is the root in the mask often enough for P4a to keep it?
# --------------------------------------------------------------------------

def write_run(tmp_path, root_fraction_per_frame, num_frames=20):
    """A synthetic P2 run: holder across the middle, root below it."""
    import cv2

    frames = tmp_path / "p1" / "frames"
    plant_dir = tmp_path / "p2" / "masks" / "plant"
    holder_dir = tmp_path / "p2" / "masks" / "holder"
    for d in (frames, plant_dir, holder_dir):
        d.mkdir(parents=True, exist_ok=True)

    sources = {}
    for i in range(num_frames):
        stem = f"frame_{i:04d}"
        cv2.imwrite(str(frames / f"{stem}.jpg"), np.zeros((100, 100, 3), np.uint8))

        holder = np.zeros((100, 100), np.uint8)
        holder[48:52, :] = 255            # jaws across the middle

        plant = np.zeros((100, 100), np.uint8)
        plant[10:45, 40:60] = 255         # foliage above the jaws
        want = root_fraction_per_frame[i]
        if want > 0:                      # root below the jaws
            rows = int(round(want * 35 * 20 / (20 * (1 - want)))) if want < 1 else 20
            plant[55:55 + max(rows, 1), 45:55] = 255

        cv2.imwrite(str(plant_dir / f"{stem}.png"), plant)
        cv2.imwrite(str(holder_dir / f"{stem}.png"), holder)
        sources[stem] = 0
    return sources


def test_root_tracked_in_most_frames_passes(tmp_path):
    from pose_estimator.segmentation_qc import run_qc

    sources = write_run(tmp_path, [0.3] * 20)
    report = run_qc(tmp_path / "p1" / "frames", tmp_path / "p2", sources=sources)
    assert report["checks"]["root_tracked_below_the_jaws"]["pass"]


def test_root_missing_from_every_frame_fails(tmp_path):
    """thistle2's registered pass: root in 0 of 96 frames, nothing else wrong."""
    from pose_estimator.segmentation_qc import run_qc

    sources = write_run(tmp_path, [0.0] * 20)
    report = run_qc(tmp_path / "p1" / "frames", tmp_path / "p2", sources=sources)
    check = report["checks"]["root_tracked_below_the_jaws"]
    assert not check["pass"]
    assert "pose-pick-prompts" in check["detail"]


def test_intermittent_root_fails_because_carving_needs_86_percent(tmp_path):
    """Present in 60% of frames is not enough: P4a asks for 86%."""
    from pose_estimator.segmentation_qc import run_qc

    sources = write_run(tmp_path, [0.3] * 12 + [0.0] * 8)
    report = run_qc(tmp_path / "p1" / "frames", tmp_path / "p2", sources=sources)
    assert not report["checks"]["root_tracked_below_the_jaws"]["pass"]


def test_a_good_pass_does_not_hide_a_bad_one(tmp_path):
    """Pooled, 50% of frames have the root and it looks survivable."""
    from pose_estimator.segmentation_qc import run_qc

    sources = write_run(tmp_path, [0.3] * 10 + [0.0] * 10)
    for i in range(10, 20):
        sources[f"frame_{i:04d}"] = 1     # the second pass is the starved one

    report = run_qc(tmp_path / "p1" / "frames", tmp_path / "p2", sources=sources)
    check = report["checks"]["root_tracked_below_the_jaws"]
    assert not check["pass"]
    assert "pass 1" in check["detail"]
    per_pass = report["root_below_jaws_per_pass"]
    assert per_pass["0"]["frames_with_root"] == 10
    assert per_pass["1"]["frames_with_root"] == 0
