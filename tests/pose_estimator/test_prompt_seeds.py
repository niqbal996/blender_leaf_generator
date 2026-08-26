"""The plant/holder prompts have to survive a change of video.

That is the whole reason this exists: --plant-point is a pixel coordinate,
which means nothing in the next capture, and the colour rule describes one
pair of pliers. These tests use a stub backbone so they say something about
the matching logic rather than about DINO.
"""

import numpy as np
import pytest

from pose_estimator.prompt_seeds import (
    build_prompt_bank,
    load_prompt_bank,
    locate_prompts,
    save_prompt_bank,
)


class FakeBackbone:
    """Patch features from a painted map: each pixel value is a class id."""

    grid = 16

    def __init__(self, dim=8):
        self.dim = dim

    def features(self, bgr):
        # One orthogonal vector per class id, read at each patch centre.
        h, w = bgr.shape[:2]
        out = np.zeros((self.grid * self.grid, self.dim), np.float32)
        for gy in range(self.grid):
            for gx in range(self.grid):
                y = int((gy + 0.5) / self.grid * h)
                x = int((gx + 0.5) / self.grid * w)
                out[gy * self.grid + gx, int(bgr[y, x, 0])] = 1.0
        return out


def painted(width=320, height=240, boxes=()):
    """Frame whose blue channel carries a class id per region."""
    frame = np.zeros((height, width, 3), np.uint8)
    for value, (x0, y0, x1, y1) in boxes:
        frame[y0:y1, x0:x1, 0] = value
    return frame


def test_prompts_move_with_the_object():
    """A bank built on one frame finds the object where the next frame put it."""
    backbone = FakeBackbone()
    # plant = 1 on the left, holder = 2 on the right
    first = painted(boxes=[(1, (40, 40, 120, 200)), (2, (200, 40, 280, 200))])
    vectors, labels = build_prompt_bank(backbone, first, [("plant", 80, 120),
                                                          ("holder", 240, 120)])

    # Same objects, swapped sides and different sizes -- a new capture.
    later = painted(boxes=[(2, (30, 60, 90, 180)), (1, (150, 30, 290, 210))])
    found = locate_prompts(backbone, later, vectors, labels)

    px, py = found["plant"][0]
    hx, hy = found["holder"][0]
    assert later[py, px, 0] == 1
    assert later[hy, hx, 0] == 2


def test_several_examples_widen_a_class():
    """Two different-looking plants both match, because vectors are kept apart.

    Averaging them would give a direction resembling neither, which is the
    same reason the organ seeds are stored individually.
    """
    backbone = FakeBackbone()
    a = painted(boxes=[(1, (40, 40, 120, 200))])
    b = painted(boxes=[(3, (40, 40, 120, 200))])
    v1, l1 = build_prompt_bank(backbone, a, [("plant", 80, 120)])
    v2, l2 = build_prompt_bank(backbone, b, [("plant", 80, 120)])
    vectors = np.concatenate([v1, v2], axis=1)
    labels = l1 + l2

    for value in (1, 3):
        frame = painted(boxes=[(value, (180, 60, 260, 180))])
        x, y = locate_prompts(backbone, frame, vectors, labels)["plant"][0]
        assert frame[y, x, 0] == value


def test_frame_edge_is_not_a_prompt():
    """Edge patches are half background, and a prompt there makes SAM2 leak."""
    backbone = FakeBackbone()
    ref = painted(boxes=[(1, (40, 40, 120, 200))])
    vectors, labels = build_prompt_bank(backbone, ref, [("plant", 80, 120)])

    # The object touches the top-left corner; the prompt must land inside it,
    # not on the corner patch.
    frame = painted(boxes=[(1, (0, 0, 140, 140))])
    x, y = locate_prompts(backbone, frame, vectors, labels)["plant"][0]
    assert x > 320 * 0.02 and y > 240 * 0.02


def test_bank_from_another_model_is_refused(tmp_path):
    """768 dims from dinov2 and from dinov3 are not the same 768 dims."""
    path = save_prompt_bank(tmp_path / "bank.npz", np.eye(4, 2, dtype=np.float32),
                            ["plant", "holder"], "facebook/dinov3-vitb16-pretrain-lvd1689m", 896)

    vectors, labels = load_prompt_bank(path, "facebook/dinov3-vitb16-pretrain-lvd1689m")
    assert labels == ["plant", "holder"]
    assert vectors.shape == (4, 2)

    with pytest.raises(SystemExit, match="dinov2-base"):
        load_prompt_bank(path, "facebook/dinov2-base")


def test_several_prompts_spread_across_the_object():
    """count>1 must place prompts on different parts, not neighbours.

    On thistle2 pass 0 one correct prompt seeded one leaf and SAM2 tracked
    only that leaf -- 12,255 px against 41,040 px for the pass where the
    plant filled more of the frame. Prompts clustered in one spot would be
    no better than a single prompt.
    """
    backbone = FakeBackbone()
    ref = painted(boxes=[(1, (40, 40, 120, 200))])
    vectors, labels = build_prompt_bank(backbone, ref, [("plant", 80, 120)])

    # one plant in two well-separated patches of the frame
    frame = painted(boxes=[(1, (20, 30, 90, 90)), (1, (220, 150, 300, 210))])
    points = locate_prompts(backbone, frame, vectors, labels, count=3)["plant"]

    assert len(points) >= 2, "a single prompt is the bug this exists to fix"
    for x, y in points:
        assert frame[y, x, 0] == 1, "a prompt drifted off the object"
    spread = max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in points for b in points)
    assert spread > 100, f"prompts clustered together: {points}"


def test_holder_is_searched_only_near_the_plant():
    """The holder prompt is useless outside the plant-sized tracking crop.

    P2 tracks inside a crop sized to the plant and drops prompts outside it.
    The most plier-like patch in the frame is the big handle, far from the
    plant -- on thistle2 both passes reported the holder prompt ignored, so
    the tool was never tracked and could not be subtracted.
    """
    backbone = FakeBackbone()
    ref = painted(boxes=[(1, (40, 40, 120, 200)), (2, (200, 40, 280, 200))])
    vectors, labels = build_prompt_bank(backbone, ref, [("plant", 80, 120),
                                                        ("holder", 240, 120)])

    # plant far left; holder tissue both beside it (jaws) and far right (handle)
    frame = painted(boxes=[(1, (20, 90, 70, 150)),
                           (2, (75, 100, 105, 140)),
                           (2, (250, 60, 310, 190))])
    found = locate_prompts(backbone, frame, vectors, labels, crop_side=120)

    assert found["plant"], "no plant prompt"
    hx, hy = found["holder"][0]
    assert frame[hy, hx, 0] == 2, "holder prompt is not on the holder"
    px = found["plant"][0][0]
    assert abs(hx - px) <= 120, f"holder at x={hx} is outside the crop around x={px}"


class ConfusableBackbone(FakeBackbone):
    """Like FakeBackbone, but one class is partly similar to another.

    Orthogonal one-hot features cannot reproduce the thistle2 failure: there
    the board was *somewhat* leaf-like and clearly not plier-like, so its
    plant-minus-holder margin was positive and prompts landed on it.
    """

    def __init__(self, dim=8, mix=None):
        super().__init__(dim)
        self.mix = mix or {}

    def features(self, bgr):
        out = super().features(bgr)
        for value, (onto, weight) in self.mix.items():
            rows = np.nonzero(out[:, value] > 0)[0]
            v = np.zeros(self.dim, np.float32)
            v[value] = np.sqrt(1 - weight ** 2)
            v[onto] = weight
            out[rows] = v
        return out


def test_extra_prompts_do_not_wander_onto_the_background():
    """The thistle2 failure, as a test.

    A small plant on a large surface that reads as more leaf-like than
    plier-like. The first prompt lands on the plant and suppresses it, so
    prompts 2 and 3 have nowhere to go but the surface -- which had a
    positive plant-minus-holder margin because background was not a class.
    SAM2 was then told the surface was the object and returned the crop
    minus the plant, inverted.
    """
    backbone = ConfusableBackbone(mix={5: (1, 0.7)})   # class 5 is 0.7-similar to plant
    ref = painted(boxes=[(1, (40, 40, 120, 200))])
    vectors, labels = build_prompt_bank(backbone, ref, [("plant", 80, 120),
                                                        ("holder", 300, 20)])

    frame = painted(boxes=[(5, (0, 0, 320, 240)), (1, (140, 100, 180, 140))])
    points = locate_prompts(backbone, frame, vectors, labels, count=3)["plant"]

    assert points, "the anchor prompt must always be returned"
    for x, y in points:
        assert frame[y, x, 0] == 1, f"prompt at ({x},{y}) landed on the background"
