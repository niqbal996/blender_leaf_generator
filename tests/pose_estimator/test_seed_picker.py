"""Seed picking: the rules that can silently produce useless seeds.

None of this needs a window. The window only drives `SeedSession`, and every
rule worth testing -- refusing off-plant clicks, undo, class switching, the
saved-file round trip -- lives in the session or the file format.
"""

import numpy as np
import pytest

from pose_estimator.seed_picker import (
    SeedSession,
    class_color,
    load_seeds,
    save_seeds,
)


def plant_mask(h=40, w=60):
    """A blob in the middle; everything else is background."""
    mask = np.zeros((h, w), bool)
    mask[10:30, 15:45] = True
    return mask


def test_click_on_the_plant_is_recorded():
    session = SeedSession(classes=["leaf", "stem"])
    ok, _ = session.add("frame_0000", 0, 20, 15, plant_mask())
    assert ok
    assert len(session.seeds) == 1
    assert session.seeds[0].label == "leaf"
    assert (session.seeds[0].x, session.seeds[0].y) == (20, 15)


def test_click_off_the_plant_is_refused():
    """The classifier blanks the background before extracting features, so a
    seed there would describe a black patch rather than an organ."""
    session = SeedSession(classes=["leaf", "stem"])
    ok, message = session.add("frame_0000", 0, 2, 2, plant_mask())
    assert not ok
    assert "not on the plant" in message
    assert session.seeds == []


def test_click_outside_the_image_is_refused():
    session = SeedSession(classes=["leaf"])
    ok, message = session.add("frame_0000", 0, 500, 500, plant_mask())
    assert not ok
    assert "outside" in message


def test_class_switching_and_counts():
    session = SeedSession(classes=["leaf", "stem", "root"])
    mask = plant_mask()
    session.add("frame_0000", 0, 20, 15, mask)
    assert session.set_class(1)
    session.add("frame_0000", 0, 22, 16, mask)
    session.add("frame_0000", 0, 24, 17, mask)
    assert session.counts() == {"leaf": 1, "stem": 2, "root": 0}
    assert session.missing_classes() == ["root"]
    assert not session.set_class(9)


def test_undo_and_clear():
    session = SeedSession(classes=["leaf"])
    mask = plant_mask()
    session.add("frame_0000", 0, 20, 15, mask)
    session.add("frame_0000", 0, 21, 16, mask)
    removed = session.undo()
    assert removed.x == 21
    assert len(session.seeds) == 1
    assert session.clear() == 1
    assert session.undo() is None


def test_seeds_from_several_frames_group_by_frame():
    """Pooling angles is the point of multi-frame seeding; the grouping is
    what `build_seed_vectors_multi` consumes."""
    session = SeedSession(classes=["leaf"])
    mask = plant_mask()
    session.add("frame_0000", 0, 20, 15, mask)
    session.add("frame_0040", 40, 25, 20, mask)
    session.add("frame_0040", 40, 26, 21, mask)
    grouped = session.by_frame()
    assert set(grouped) == {"frame_0000", "frame_0040"}
    assert len(grouped["frame_0040"]) == 2


def test_save_load_round_trip(tmp_path):
    session = SeedSession(classes=["leaf", "stem", "root"])
    mask = plant_mask()
    session.add("frame_0000", 0, 20, 15, mask)
    session.set_class(1)
    session.add("frame_0040", 40, 25, 20, mask)

    path = save_seeds(tmp_path / "seeds.json", session, pad=60)
    seeds, class_order, pad = load_seeds(path)

    assert pad == 60
    assert len(seeds) == 2
    assert [s.label for s in seeds] == ["leaf", "stem"]
    assert [(s.x, s.y) for s in seeds] == [(20, 15), (25, 20)]
    assert [s.frame for s in seeds] == ["frame_0000", "frame_0040"]
    # A class nobody clicked must not become a column the classifier cannot fill.
    assert class_order == ["leaf", "stem"]
    assert "root" not in class_order


def test_load_rejects_an_empty_file(tmp_path):
    session = SeedSession(classes=["leaf"])
    path = save_seeds(tmp_path / "empty.json", session)
    with pytest.raises(SystemExit):
        load_seeds(path)


def test_cli_string_matches_the_typed_form():
    session = SeedSession(classes=["leaf", "stem"])
    mask = plant_mask()
    session.add("frame_0000", 0, 20, 15, mask)
    session.set_class(1)
    session.add("frame_0000", 0, 30, 25, mask)
    assert session.as_cli_string() == '"leaf:20,15" "stem:30,25"'


def test_display_scaling_maps_clicks_back_to_crop_coordinates():
    """The window may be shrunk to fit the screen; what gets stored must be a
    coordinate in the full-size crop, not in the shrunken view."""
    session = SeedSession(classes=["leaf"])
    mask = np.ones((400, 600), bool)
    scale = 1100 / 2200  # a 2200px-wide crop shown at 1100px
    click_x, click_y = 150, 100                     # as clicked in the window
    session.add("frame_0000", 0, int(round(click_x / scale)),
                int(round(click_y / scale)), mask)
    assert (session.seeds[0].x, session.seeds[0].y) == (300, 200)


def test_known_organs_keep_stable_colors():
    assert class_color("leaf", 0) == class_color("leaf", 5)
    assert class_color("leaf", 0) != class_color("stem", 0)
