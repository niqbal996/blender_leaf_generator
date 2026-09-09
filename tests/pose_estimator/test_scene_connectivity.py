"""P3 reports when COLMAP built more than one scene, instead of hiding it.

The failure guarded against here is silent by construction. When a group of
images cannot be tied to the rest, COLMAP does not fail -- it starts a second
scene and carries on. Keeping the largest then discards a complete capture
pass, and the only outward sign is a registered count that looks exactly like
a handful of soft frames.

That is the shape a handheld top-down pass fails in, which is why the check
asks a structural question -- did every pass reach the winning scene -- rather
than measuring anything about the plant.
"""

import pytest

from pose_estimator.pose import evaluate_poses, _connectivity_checks
from pose_estimator.reconstruction import (
    capture_guidance,
    describe_connectivity,
)


class FakeImage:
    def __init__(self, name):
        self.name = name


class FakeModel:
    """Just enough of a pycolmap.Reconstruction for the connectivity report."""

    def __init__(self, names):
        self._images = {i: FakeImage(n) for i, n in enumerate(names)}

    def reg_image_ids(self):
        return list(self._images)

    @property
    def images(self):
        return self._images

    def num_reg_images(self):
        return len(self._images)


ORBIT = [f"frame_{i:04d}.jpg" for i in range(96)]
TOPDOWN = [f"frame_{i:04d}.jpg" for i in range(96, 126)]
SOURCES = {f"frame_{i:04d}": (0 if i < 96 else 1) for i in range(126)}


def test_a_split_names_the_pass_that_was_left_out():
    orbit, topdown = FakeModel(ORBIT), FakeModel(TOPDOWN)
    info = describe_connectivity({0: orbit, 1: topdown}, orbit, ORBIT + TOPDOWN, SOURCES)

    assert info["num_models"] == 2
    assert info["winner"]["passes"] == {0: 96}
    assert info["passes_absent_from_winner"] == [1]
    assert info["discarded_models"] == [
        {"model": 1, "num_images": 30, "passes": {1: 30}}
    ]
    # Every image is accounted for by some scene, so none are "unregistered".
    assert info["unregistered"] == []


def test_a_lost_pass_fails_the_check():
    orbit, topdown = FakeModel(ORBIT), FakeModel(TOPDOWN)
    info = describe_connectivity({0: orbit, 1: topdown}, orbit, ORBIT + TOPDOWN, SOURCES)

    checks = _connectivity_checks({"final": info, "recovered_images": 0})
    check = checks["all_passes_joined_one_scene"]

    assert check["pass"] is False
    # The detail has to say which pass vanished -- a count alone reads like blur.
    assert "pass 1" in check["detail"]
    assert "NOTHING" in check["detail"]


def test_one_scene_holding_every_pass_passes():
    everything = FakeModel(ORBIT + TOPDOWN)
    info = describe_connectivity({0: everything}, everything, ORBIT + TOPDOWN, SOURCES)

    assert info["passes_absent_from_winner"] == []
    checks = _connectivity_checks({"final": info, "recovered_images": 0})
    assert checks["all_passes_joined_one_scene"]["pass"] is True


def test_a_stray_scene_is_reported_but_does_not_fail_a_single_pass_run():
    """A disconnected clump on one capture is worth seeing, not worth stopping
    for: the main scene can be complete without it, and gating here would halt
    good unattended runs."""
    names = [f"frame_{i:04d}.jpg" for i in range(96)]
    main, stray = FakeModel(names[:84]), FakeModel(names[84:])
    info = describe_connectivity({0: main, 1: stray}, main, names, None)

    assert info["passes_absent_from_winner"] == []
    check = _connectivity_checks({"final": info, "recovered_images": 0})[
        "all_passes_joined_one_scene"]
    assert check["pass"] is True
    assert "disconnected" in check["detail"]


def test_unregistered_images_belong_to_no_scene():
    names = ORBIT + TOPDOWN
    orbit = FakeModel(ORBIT)
    info = describe_connectivity({0: orbit}, orbit, names, SOURCES)

    assert len(info["unregistered"]) == 30
    assert info["unregistered_passes"] == {1: 30}
    assert info["passes_absent_from_winner"] == [1]


def test_guidance_is_silent_when_nothing_is_missing():
    everything = FakeModel(ORBIT + TOPDOWN)
    info = describe_connectivity({0: everything}, everything, ORBIT + TOPDOWN, SOURCES)
    assert capture_guidance(info, low_texture=False) == []


def test_guidance_names_the_capture_fix_and_only_offers_low_texture_when_it_is_off():
    orbit, topdown = FakeModel(ORBIT), FakeModel(TOPDOWN)
    info = describe_connectivity({0: orbit, 1: topdown}, orbit, ORBIT + TOPDOWN, SOURCES)

    off = "\n".join(capture_guidance(info, low_texture=False))
    assert "--low-texture" in off
    assert "CONTINUOUS climb" in off
    # Exhaustive matching already tried every pair, so the advice must not
    # suggest the solver has anything left to try.
    assert "Exhaustive matching" in off

    on = "\n".join(capture_guidance(info, low_texture=True))
    assert "--low-texture" not in on


def test_check_is_absent_rather_than_guessed_for_an_older_workdir():
    assert _connectivity_checks(None) == {}
    assert _connectivity_checks({}) == {}


def test_json_round_trip_keeps_the_check_working():
    """`solve.json` turns integer pass keys into strings; the check must still
    name the right pass afterwards."""
    import json

    orbit, topdown = FakeModel(ORBIT), FakeModel(TOPDOWN)
    info = describe_connectivity({0: orbit, 1: topdown}, orbit, ORBIT + TOPDOWN, SOURCES)
    reloaded = json.loads(json.dumps({"final": info, "recovered_images": 0}))

    check = _connectivity_checks(reloaded)["all_passes_joined_one_scene"]
    assert check["pass"] is False
    assert "pass 1 (30 frames)" in check["detail"]
