"""P3 does not let a degenerate scene win, and does not keep unsupported poses.

Both cases were measured, not imagined. On maize_1 (2026-09-29, EXIF cameras)
the larger scene had its shared focal run to 2698px on a 1653px lens, and it
beat a smaller scene whose orbits were 0.12% and 0.22% circles. On sugarbeet_1
one frame kept its pose on 3 triangulated points, sat 62 units off an orbit of
radius 3.6, and alone failed three P3 checks.
"""

from pose_estimator.reconstruction import (
    MAX_FOCAL_DRIFT,
    drop_unsupported_images,
    focal_drift,
    pick_winner,
)


class FakeCamera:
    def __init__(self, focal):
        self.focal = focal

    def mean_focal_length(self):
        return self.focal


class FakeImage:
    def __init__(self, image_id, name, support):
        self.image_id = image_id
        self.frame_id = image_id
        self.camera_id = 1
        self.name = name
        self.num_points3D = support


class FakeScene:
    """Just enough of a pycolmap.Reconstruction: one shared camera."""

    def __init__(self, names, focal, supports=None):
        supports = supports or [500] * len(names)
        self.images = {i: FakeImage(i, n, s) for i, (n, s) in enumerate(zip(names, supports))}
        self.cameras = {1: FakeCamera(focal)}
        self._registered = set(self.images)

    def reg_image_ids(self):
        return sorted(self._registered)

    def num_reg_images(self):
        return len(self._registered)

    def deregister_frame(self, frame_id):
        self._registered.discard(frame_id)


def names(start, count):
    return [f"frame_{i:04d}.jpg" for i in range(start, start + count)]


EXIF = {f"frame_{i:04d}": 1653.3 for i in range(200)}


def test_a_larger_scene_with_a_runaway_focal_does_not_win():
    degenerate = FakeScene(names(41, 72), focal=2698.0)
    sound = FakeScene(names(0, 61), focal=1695.0)

    winner, rejected = pick_winner({0: sound, 1: degenerate}, EXIF)

    assert winner is sound
    assert [r["model"] for r in rejected] == [1]
    assert rejected[0]["focal_drift"] > MAX_FOCAL_DRIFT


def test_ordinary_focal_refinement_is_not_mistaken_for_degeneracy():
    # Sound scenes measured up to +11.5% per image, median at most +7.2%.
    scene = FakeScene(names(0, 60), focal=1653.3 * 1.115)
    assert abs(focal_drift(scene, EXIF)) < MAX_FOCAL_DRIFT


def test_without_exif_the_largest_scene_still_wins():
    small, large = FakeScene(names(0, 10), 400.0), FakeScene(names(10, 50), 2698.0)
    winner, rejected = pick_winner({0: small, 1: large}, {})
    assert winner is large and rejected == []


def test_when_every_scene_is_implausible_the_largest_still_wins():
    small, large = FakeScene(names(0, 10), 400.0), FakeScene(names(10, 50), 2698.0)
    winner, rejected = pick_winner({0: small, 1: large}, EXIF)
    assert winner is large and len(rejected) == 2


def test_an_image_placed_by_nothing_is_dropped_and_named():
    scene = FakeScene(names(87, 3), 2959.0, supports=[500, 3, 37])

    dropped = drop_unsupported_images(scene)

    assert dropped == [("frame_0088.jpg", 3)]
    assert scene.num_reg_images() == 2
