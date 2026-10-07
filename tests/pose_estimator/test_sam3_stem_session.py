"""P2 (SAM3): the stem gets a session of its own, and trims the leaf masks.

vogelmeere: SAM3's leaf track pass0_24 slid onto leaf 6's petiole partway
through pass 0. In the plant session "stem" cannot claim tissue "leaf" holds,
so the petiole was missing from masks/stem and voted "leaf" in P5x. A separate
stem session claimed 85-100% of such stalk-bound leaf masks and none of any
blade. Fake sessions here -- no GPU, no weights.
"""

import json

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
from pose_estimator import segmentation_sam3 as sam3  # noqa: E402

H = W = 80


def _rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


STEM = _rect(10, 75, 38, 42)                  # the main stem
STALK = _rect(30, 33, 42, 60)                 # a petiole off it
BLADE = _rect(20, 45, 60, 78)                 # the blade it carries
STALK2 = _rect(55, 58, 10, 38)                # another petiole, its leaf off frame
PLANT = STEM | STALK | BLADE | STALK2


def fake_session(processor, model, torch, video, phrases, device, want_soft=False, label=""):
    frames = {}
    for i in range(len(video)):
        if "plant" in phrases:
            frames[i] = {"plant": {0: PLANT},
                         # leaf 1 took its petiole in; leaf 2 is a track sitting on a stalk
                         "leaf": {1: BLADE | STALK, 2: STALK2}}
        elif phrases == ["stem"]:
            frames[i] = {"stem": {0: STEM | STALK | STALK2}}
        else:
            frames[i] = {}
    soft = {i: PLANT.astype(np.float32) for i in frames} if want_soft else None
    return frames, soft


@pytest.fixture
def p2(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(2):
        cv2.imwrite(str(frames / f"frame_{i:04d}.jpg"), np.full((H, W, 3), 90, np.uint8))
    monkeypatch.setattr(sam3, "_run_session", fake_session)
    out = tmp_path / "p2"
    sam3.segment_sequence_sam3(frames, out, prompts=sam3.Sam3Prompts(root=[]), use_roi=False,
                               device="cpu", session=(None, None), instance_prefix="pass0_")
    return out


def read(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127


def test_stalks_a_leaf_track_held_are_stem(p2):
    stem = read(p2 / "masks" / "stem" / "frame_0000.png")
    assert stem[STALK].all(), "the petiole leaf 1 took in is stem"
    assert stem[STALK2].all(), "the stalk leaf 2's track sat on is stem"
    assert not stem[BLADE].any(), "a blade is never stem"


def test_each_leaf_mask_ends_where_its_blade_begins(p2):
    leaf1 = read(p2 / "masks" / "leaf_instances" / "pass0_1" / "frame_0000.png")
    assert leaf1[BLADE].all() and not leaf1[STALK].any()


def test_a_leaf_mask_lying_on_a_stalk_is_not_written(p2):
    assert not (p2 / "masks" / "leaf_instances" / "pass0_2").exists()
    record = json.loads((p2 / "prompts.json").read_text())["stem_residual"]
    assert record["leaf_masks_dropped_as_stalk"] == {"pass0_2": ["frame_0000", "frame_0001"]}
