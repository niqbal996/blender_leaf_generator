import cv2
import numpy as np
import pytest

from leaf_generator.calibration import SessionCalibration
from leaf_generator.sizing import measure_leaf_bbox_px, measure_leaf_size_mm


def _save_rect_mask(path, canvas=200, rect_w=100, rect_h=40):
    img = np.zeros((canvas, canvas), dtype=np.uint8)
    x0 = (canvas - rect_w) // 2
    y0 = (canvas - rect_h) // 2
    img[y0:y0 + rect_h, x0:x0 + rect_w] = 255
    cv2.imwrite(str(path), img)


def test_measure_leaf_bbox_px_matches_known_rect(tmp_path):
    mask_path = tmp_path / "mask.png"
    _save_rect_mask(mask_path, rect_w=100, rect_h=40)

    bbox = measure_leaf_bbox_px(mask_path)

    assert bbox["width_px"] == pytest.approx(100, abs=1)
    assert bbox["height_px"] == pytest.approx(40, abs=1)


def test_measure_leaf_size_mm_applies_calibration(tmp_path):
    mask_path = tmp_path / "mask.png"
    _save_rect_mask(mask_path, rect_w=100, rect_h=40)
    calibration = SessionCalibration(pixel_size_mm=0.5)

    size = measure_leaf_size_mm(mask_path, calibration)

    assert size["width_mm"] == pytest.approx(50, abs=1)
    assert size["height_mm"] == pytest.approx(20, abs=1)
