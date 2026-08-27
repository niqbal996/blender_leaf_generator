"""P2 prompt picking: the rules that decide which object SAM2 tracks.

None of this needs a window. The window only drives `PromptSession`, and the
rules worth testing -- one seed set per pass, refusing to save a pass with no
plant point, the full-frame to crop conversion, and the continuity tracker
that stops the holder stealing the crop -- live in the session, in `Prompts`,
or in the file format.

The failure these exist to prevent, from thistle1: the pliers' amber grip
reads as green-dominant, out-grew the plant in the second pass, and took both
the SAM2 seed and the tracking crop with it. P4a then intersected two
silhouette sets describing different objects and carved nothing.
"""

import json

import numpy as np
import pytest

from pose_estimator.prompt_picker import (
    CapturePass,
    PromptSession,
    load_prompts,
    save_prompts,
)
from pose_estimator.segmentation import Prompts, _component_nearest


def two_passes(tmp_path=None):
    return [
        CapturePass(index=0, frame="frame_0000", path=tmp_path or "frame_0000.jpg"),
        CapturePass(index=1, frame="frame_0096", path=tmp_path or "frame_0096.jpg"),
    ]


def subject_mask(h=40, w=60):
    """What rotates with the table; the rest is camera-mounted backdrop."""
    mask = np.zeros((h, w), bool)
    mask[10:35, 10:50] = True
    return mask


def test_points_are_kept_per_pass():
    """One point cannot serve two passes: each is a separate propagation
    seeded on its own first frame."""
    session = PromptSession(passes=two_passes())
    session.add(20, 20)
    session.step_pass(1)
    session.add(30, 25)

    assert session.points[0]["plant"] == [(20, 20)]
    assert session.points[1]["plant"] == [(30, 25)]


def test_backdrop_click_warns_but_is_kept():
    """The rotating-region mask is derived by Otsu, not ground truth, so it
    must not be able to lock the user out of seeding."""
    session = PromptSession(passes=two_passes())
    ok, message = session.add(2, 2, subject_mask())

    assert ok
    assert message.startswith("?")
    assert "backdrop" in message
    assert session.points[0]["plant"] == [(2, 2)]


def test_click_on_the_subject_is_not_flagged():
    session = PromptSession(passes=two_passes())
    ok, message = session.add(20, 20, subject_mask())

    assert ok
    assert not message.startswith(("!", "?"))


def test_click_outside_the_image_is_refused():
    session = PromptSession(passes=two_passes())
    ok, _ = session.add(500, 500, subject_mask())

    assert not ok
    assert session.points[0]["plant"] == []


def test_a_pass_without_a_plant_point_blocks_saving():
    """Saving a half-filled file would fail in the middle of the next run
    rather than here, after the frames are already extracted."""
    session = PromptSession(passes=two_passes())
    session.add(20, 20)

    assert session.passes_missing_plant() == [1]

    session.step_pass(1)
    session.add(30, 25)
    assert session.passes_missing_plant() == []


def test_holder_alone_does_not_satisfy_a_pass():
    session = PromptSession(passes=two_passes())
    session.set_class(1)
    session.add(20, 20)

    assert session.counts()["holder"] == 1
    assert session.passes_missing_plant() == [0, 1]


def test_undo_only_touches_the_current_pass():
    session = PromptSession(passes=two_passes())
    session.add(20, 20)
    session.step_pass(1)
    session.add(30, 25)
    session.undo()

    assert session.points[0]["plant"] == [(20, 20)]
    assert session.points[1]["plant"] == []


def test_round_trip_through_the_file(tmp_path):
    session = PromptSession(passes=two_passes())
    session.add(20, 20)
    session.set_class(1)
    session.add(40, 30)
    session.set_class(2)
    session.add(22, 38)
    session.step_pass(1)
    session.set_class(0)
    session.add(31, 26)

    path = save_prompts(tmp_path / "prompts_clicked.json", session)
    loaded = load_prompts(path)

    assert sorted(loaded) == [0, 1]
    assert loaded[0].plant == [(20, 20)]
    assert loaded[0].holder == [(40, 30)]
    assert loaded[0].root == [(22, 38)]
    assert loaded[1].plant == [(31, 26)]
    assert loaded[1].root == []
    # Full-frame is what makes the file convertible once a crop exists; a file
    # in crop coordinates could not be reinterpreted if the crop changed.
    assert loaded[0].space == "full_frame"


def test_a_file_from_before_the_root_category_still_loads(tmp_path):
    """Root points are additive: prompt files clicked before [3] existed have
    no "root" key and must keep loading, with the old two-object behaviour."""
    path = tmp_path / "prompts_clicked.json"
    path.write_text(json.dumps({
        "version": 1, "space": "full_frame",
        "passes": {"0": {"frame": "frame_0000", "plant": [[20, 20]], "holder": [[40, 30]]}},
    }))

    loaded = load_prompts(path)
    assert loaded[0].plant == [(20, 20)]
    assert loaded[0].root == []


def test_a_pass_without_a_plant_point_is_rejected_on_load(tmp_path):
    path = tmp_path / "prompts_clicked.json"
    path.write_text(json.dumps({
        "version": 1, "space": "full_frame",
        "passes": {"0": {"frame": "frame_0000", "plant": [], "holder": [[5, 5]]}},
    }))

    with pytest.raises(SystemExit, match="no plant point"):
        load_prompts(path)


def test_a_stale_file_version_is_rejected(tmp_path):
    path = tmp_path / "prompts_clicked.json"
    path.write_text(json.dumps({"version": 99, "space": "full_frame", "passes": {}}))

    with pytest.raises(SystemExit, match="version"):
        load_prompts(path)


# --------------------------------------------------------------------------
# Full-frame -> crop conversion
# --------------------------------------------------------------------------


def test_prompts_convert_into_crop_space():
    prompts = Prompts(plant=[(952, 476)], holder=[(1100, 500)], space="full_frame")
    converted = prompts.to_crop((408, 0, 1488, 1080))

    assert converted.plant == [(544, 476)]
    assert converted.holder == [(692, 500)]
    assert converted.space == "crop"


def test_a_plant_point_outside_the_crop_is_fatal():
    """The crop is solved *from* the plant point, so a plant point outside it
    means the window is not where it was asked to be."""
    prompts = Prompts(plant=[(1900, 500)], space="full_frame")

    with pytest.raises(ValueError, match="plant prompt"):
        prompts.to_crop((408, 0, 1488, 1080))


def test_a_holder_point_outside_the_crop_is_dropped():
    """Expected, not fatal: the crop is sized to the plant and the pliers
    extend well past it, so a click on the far end of the handle is out."""
    prompts = Prompts(plant=[(952, 476)], holder=[(1900, 900)], space="full_frame")
    converted = prompts.to_crop((408, 0, 1488, 1080))

    assert converted.plant == [(544, 476)]
    assert converted.holder == []


def test_root_points_convert_like_plant_points():
    prompts = Prompts(plant=[(952, 476)], root=[(970, 900)], space="full_frame")
    converted = prompts.to_crop((408, 0, 1488, 1080))

    assert converted.root == [(562, 900)]


def test_a_root_point_outside_the_crop_is_fatal():
    """Unlike the holder, a dropped root point silently re-creates the root
    loss this category exists to fix -- the crop solver is told to contain
    it, so falling outside means the window is wrong, not the click."""
    prompts = Prompts(plant=[(952, 476)], root=[(952, 1500)], space="full_frame")

    with pytest.raises(ValueError, match="root prompt"):
        prompts.to_crop((408, 0, 1488, 1080))


# --------------------------------------------------------------------------
# The continuity tracker
# --------------------------------------------------------------------------


def test_nearest_component_prefers_continuity_over_area():
    """The whole point: the wrong blob is allowed to be the bigger one.

    Mirrors thistle1's second pass, where the handle was a 62k-px blob
    against the plant's 44k and every area-based rule picked the tool.
    """
    mask = np.zeros((100, 200), bool)
    mask[40:60, 20:40] = True     # the plant: 400 px
    mask[20:80, 120:180] = True   # the holder: 3600 px

    tracked = _component_nearest(mask, np.array([30.0, 50.0]))

    assert tracked is not None
    assert tracked[50, 30]        # on the small, near blob
    assert not tracked[50, 150]   # not on the large, far one


def test_tracking_crop_grows_to_contain_clicked_points(tmp_path):
    """The crop is sized from the colour prepass's foliage blob, and the
    exposed root hangs below the jaws outside that blob. Without growth the
    root click lands outside the window and `to_crop` (correctly) refuses to
    run -- so the solver must widen the window to what was clicked."""
    import cv2

    from pose_estimator.segmentation import solve_tracking_crop

    frame = np.zeros((300, 300, 3), np.uint8)
    frame[40:80, 60:100] = (0, 200, 0)  # the foliage blob the prepass sees
    paths = []
    for i in range(3):
        path = tmp_path / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)

    root_click = (80, 250)  # far below the blob, like a root below the jaws
    without = solve_tracking_crop(paths)
    with_root = solve_tracking_crop(paths, include_points=[root_click])

    def contains(box, point):
        x0, y0, x1, y1 = box
        return x0 <= point[0] < x1 and y0 <= point[1] < y1

    assert not contains(without.boxes[0], root_click)
    assert contains(with_root.boxes[0], root_click)


def test_nearest_component_returns_none_on_an_empty_mask():
    assert _component_nearest(np.zeros((50, 50), bool), np.array([10.0, 10.0])) is None


def test_specks_below_the_area_floor_are_ignored():
    """A single noise pixel next to the seed must not beat the real blob."""
    mask = np.zeros((200, 200), bool)
    mask[100:140, 100:140] = True  # the plant
    mask[10, 10] = True            # a speck, closer to the seed

    tracked = _component_nearest(mask, np.array([12.0, 12.0]))

    assert tracked is not None
    assert tracked[120, 120]
