"""Cropping to the plant before the learned backends see it.

The learned models work at a few hundred pixels square. Measured on thistle3,
whose plant fills 6.8% of the frame, that leaves the subject spanning 303 px
for VGGT-Omega and 251 for MapAnything against COLMAP's 931 -- so the branch
comparison was, in large part, a resolution comparison. Cropping to the plant
and filling the crop from the original 24 MP photograph is what closes it.

The arithmetic has to be exact: everything after P3 -- the P2 masks P4a carves
with, the class maps P4c votes on -- lives in full-frame pixels, so a point the
model places in a crop has to come back out into the frame it came from. These
tests are that round trip.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from pose_estimator.geometry import (plant_crop_boxes, rescale_model_to_frames,
                                     stage_masked_images)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import mv_fusion  # noqa: E402

FRAME_W, FRAME_H = 1920, 1280


def workdir_with_plant(tmp_path, box=(700, 400, 500, 450)):
    """A P1 frame and a P2 mask holding one rectangular 'plant'."""
    frames = tmp_path / "p1" / "frames"
    masks = tmp_path / "p2" / "masks" / "plant"
    frames.mkdir(parents=True)
    masks.mkdir(parents=True)
    x, y, w, h = box
    for i in range(3):
        image = np.full((FRAME_H, FRAME_W, 3), 30, np.uint8)
        image[y:y + h, x:x + w] = 200
        mask = np.zeros((FRAME_H, FRAME_W), np.uint8)
        mask[y:y + h, x:x + w] = 255
        cv2.imwrite(str(frames / f"frame_{i:04d}.jpg"), image)
        cv2.imwrite(str(masks / f"frame_{i:04d}.png"), mask)
    return tmp_path


def test_the_crop_is_centred_on_the_principal_point(tmp_path):
    """Not a preference. VGGT-Omega's camera head predicts a field of view and
    writes `intrinsics[0, 2] = W / 2` literally, so it cannot represent a
    principal point anywhere but the middle of what it is given. An off-centre
    crop moves the true one and leaves the model's assumption behind."""
    work = workdir_with_plant(tmp_path)
    frames = sorted((work / "p1" / "frames").glob("*.jpg"))
    for x0, y0, width, height in plant_crop_boxes(work, frames).values():
        assert np.isclose(x0 + width / 2, FRAME_W / 2, atol=1.0)
        assert np.isclose(y0 + height / 2, FRAME_H / 2, atol=1.0)


def test_the_crop_contains_the_whole_plant(tmp_path):
    work = workdir_with_plant(tmp_path)
    frames = sorted((work / "p1" / "frames").glob("*.jpg"))
    px, py, pw, ph = 700, 400, 500, 450
    for x0, y0, width, height in plant_crop_boxes(work, frames).values():
        assert x0 <= px and y0 <= py
        assert x0 + width >= px + pw and y0 + height >= py + ph
        assert width < FRAME_W, "a crop that keeps the whole frame buys nothing"


def test_a_plant_filling_the_frame_is_not_cropped(tmp_path):
    """Nothing to gain, and the code should say so by declining rather than
    returning a box that is the frame."""
    work = workdir_with_plant(tmp_path, box=(0, 0, FRAME_W, FRAME_H))
    frames = sorted((work / "p1" / "frames").glob("*.jpg"))
    assert plant_crop_boxes(work, frames) is None


def test_staging_writes_the_crop_and_a_manifest(tmp_path):
    work = workdir_with_plant(tmp_path)
    staged, crop = stage_masked_images(work, tmp_path / "out")

    assert crop is not None
    assert crop["frame_size"] == [FRAME_W, FRAME_H]
    written = cv2.imread(str(staged[0]))
    assert written.shape[1] == crop["staged_size"][0]
    assert written.shape[1] < FRAME_W, "the staged image should be the crop"
    assert len(crop["boxes_in_frame"]) == len(staged), "one box per frame"


def test_a_model_pixel_maps_back_to_the_frame_pixel_it_came_from(tmp_path):
    """The round trip, at the two points whose answer is known independently.

    The principal point is the strong one: the crop is centred on it, so
    whatever the model's working resolution, the centre of the model image has
    to land exactly on the centre of the frame. If the crop offset and the
    scale disagree, this is where it shows.
    """
    work = workdir_with_plant(tmp_path)
    staged, crop = stage_masked_images(work, tmp_path / "out")

    for name in crop["boxes_in_frame"]:
        for model_size in [(512, 512), (624, 416), (518, 336)]:
            sx, sy, x0, y0 = mv_fusion.to_frame_pixels(
                crop, crop["staged_size"], model_size, name=name)
            centre_x = (model_size[0] / 2) * sx + x0
            centre_y = (model_size[1] / 2) * sy + y0
            assert np.isclose(centre_x, FRAME_W / 2, atol=1e-6), (name, model_size)
            assert np.isclose(centre_y, FRAME_H / 2, atol=1e-6), (name, model_size)


def test_no_crop_leaves_the_old_mapping_alone(tmp_path):
    """A run without a crop must behave exactly as it did before."""
    sx, sy, x0, y0 = mv_fusion.to_frame_pixels(None, (1920, 1280), (624, 416))
    assert (x0, y0) == (0.0, 0.0)
    assert np.isclose(sx, 1920 / 624) and np.isclose(sy, 1280 / 416)


def test_rescaling_a_model_undoes_the_crop(tmp_path):
    """MapAnything writes its model at its own resolution and P3 maps it back.
    With a crop that mapping gains a translation, and the principal point is
    again where the answer is known: the middle of the frame."""
    pycolmap = pytest.importorskip("pycolmap")

    box = (285, 30, 1350, 1220)
    model_w, model_h = 518, 336
    model = tmp_path / "sparse"
    model.mkdir()
    reconstruction = pycolmap.Reconstruction()
    # What the model writes: a centred principal point in its own pixels.
    reconstruction.add_camera(pycolmap.Camera(
        camera_id=1, model="PINHOLE", width=model_w, height=model_h,
        params=[468.0, 468.0, model_w / 2, model_h / 2]))
    reconstruction.write(str(model))

    crop = {"boxes_in_frame": {"frame_0000": list(box)}, "frame_size": [FRAME_W, FRAME_H]}
    changed = rescale_model_to_frames(model, (FRAME_W, FRAME_H), crop=crop)
    assert changed and "skipped" not in changed

    camera = list(pycolmap.Reconstruction(str(model)).cameras.values())[0]
    assert (camera.width, camera.height) == (FRAME_W, FRAME_H)
    assert np.isclose(camera.principal_point_x, FRAME_W / 2, atol=2.0)
    assert np.isclose(camera.principal_point_y, FRAME_H / 2, atol=2.0)
