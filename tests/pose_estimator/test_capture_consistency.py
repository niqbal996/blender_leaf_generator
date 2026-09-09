"""P1 is the last phase that can tell two shoots apart.

After it, every pass is `frame_XXXX.jpg` in one directory and a pass belonging
to a different specimen is indistinguishable from a second elevation of this
one. It masks cleanly in P2, reaches P3, and shows up only as a solve that
will not settle -- with no phase reporting a failure.
"""

from datetime import datetime, timedelta

import pytest

from pose_estimator.frames import CaptureMismatch, check_capture_consistency

START = datetime(2026, 9, 1, 16, 44, 0)


def pass_records(start, count=10, step_s=6, camera="NIKON D5600", source_dir="/data/pass1"):
    return [
        {
            "frame": f"frame_{i:04d}",
            "source_dir": source_dir,
            "source_file": f"DSC_{i:04d}.JPG",
            "shot_at": (start + timedelta(seconds=i * step_s)).isoformat(),
            "camera": camera,
            "focal_px": 2400.0,
        }
        for i in range(count)
    ]


def test_consecutive_passes_of_one_shoot_are_accepted():
    # Measured spacing on sugarbeet_3 and sugarbeet_4: 1.0-1.3 min between passes.
    first = pass_records(START, count=40)
    second = pass_records(START + timedelta(minutes=4, seconds=60), count=45)
    assert check_capture_consistency([first, second]) == []


def test_a_pass_shot_before_the_previous_one_is_rejected():
    # sugarbeet_4/pass1 + sugarbeet_3/pass2: the second was shot an hour earlier.
    first = pass_records(START, count=40)
    earlier = pass_records(START - timedelta(minutes=57), count=13, source_dir="/data/other/pass2")
    with pytest.raises(CaptureMismatch, match="BEFORE"):
        check_capture_consistency([first, earlier])


def test_a_long_gap_between_passes_is_rejected():
    first = pass_records(START, count=20)
    late = pass_records(START + timedelta(minutes=90), count=20)
    with pytest.raises(CaptureMismatch, match="between passes"):
        check_capture_consistency([first, late])


def test_the_gap_limit_is_adjustable():
    first = pass_records(START, count=20)
    late = pass_records(START + timedelta(minutes=90), count=20)
    assert check_capture_consistency([first, late], max_pass_gap_minutes=180.0) == []


def test_two_camera_bodies_are_rejected():
    first = pass_records(START, count=10)
    second = pass_records(START + timedelta(minutes=2), count=10, camera="Canon EOS R5")
    with pytest.raises(CaptureMismatch, match="camera bodies"):
        check_capture_consistency([first, second])


def test_filename_order_that_is_not_capture_order_is_rejected():
    # The README has always required this and nothing checked it: a folder
    # holding two shoots, or renamed files, breaks SAM2 propagation and P3's
    # circle fit without any phase noticing.
    first = pass_records(START, count=10)
    first[7]["shot_at"] = (START - timedelta(minutes=5)).isoformat()
    second = pass_records(START + timedelta(minutes=2), count=10)
    with pytest.raises(CaptureMismatch, match="not capture order"):
        check_capture_consistency([first, second])


def test_passes_without_exif_timestamps_are_left_alone():
    # Video frames carry no EXIF; refusing to guess is better than a false alarm.
    first = [dict(r, shot_at=None) for r in pass_records(START, count=10)]
    second = [dict(r, shot_at=None) for r in pass_records(START, count=10)]
    assert check_capture_consistency([first, second]) == []
