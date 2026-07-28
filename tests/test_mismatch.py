import cv2
import numpy as np

from leaf_generator.mismatch import mask_shape_iou


def _save_circle(path, size=200, radius=80, center=None):
    img = np.zeros((size, size), dtype=np.uint8)
    center = center or (size // 2, size // 2)
    cv2.circle(img, center, radius, 255, -1)
    cv2.imwrite(str(path), img)


def _save_teardrop(path, size=200):
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, 255, -1)
    triangle = np.array([[0, size // 2], [size // 2, 0], [size // 2, size]], dtype=np.int32)
    cv2.fillConvexPoly(img, triangle, 255)
    cv2.imwrite(str(path), img)


def test_identical_shapes_have_high_iou(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _save_circle(a)
    _save_circle(b, center=(90, 110))  # translated, should still match (translation invariant)

    assert mask_shape_iou(a, b) > 0.95


def test_different_shapes_have_low_iou(tmp_path):
    a = tmp_path / "circle.png"
    b = tmp_path / "teardrop.png"
    _save_circle(a)
    _save_teardrop(b)

    assert mask_shape_iou(a, b) < 0.8


def test_empty_mask_returns_zero(tmp_path):
    a = tmp_path / "empty.png"
    b = tmp_path / "circle.png"
    cv2.imwrite(str(a), np.zeros((100, 100), dtype=np.uint8))
    _save_circle(b)

    assert mask_shape_iou(a, b) == 0.0
