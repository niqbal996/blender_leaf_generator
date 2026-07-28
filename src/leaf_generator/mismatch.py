"""Sanity-check how well two masks of the "same" leaf (oberseite/unterseite)
agree in shape, so a double-sided mesh built from one side's silhouette can
warn the user if the other side looks like a different outline (misaligned
scan, wrong leaf, badly cropped ROI, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


def mask_shape_iou(
    mask_a_path: Union[str, Path],
    mask_b_path: Union[str, Path],
    canvas: int = 256,
) -> float:
    """Intersection-over-union of two masks' silhouettes, after cropping each
    to its bounding box and resizing (aspect-preserving) onto a common
    canvas. Scale/translation invariant, so it isolates shape differences
    rather than the leaf being e.g. slightly recentered between scans.

    Returns 0.0 if either mask has no foreground.
    """
    a = _binary_bbox_crop(mask_a_path)
    b = _binary_bbox_crop(mask_b_path)
    if a is None or b is None:
        return 0.0

    a_canvas = _center_on_canvas(a, canvas)
    b_canvas = _center_on_canvas(b, canvas)

    intersection = np.logical_and(a_canvas, b_canvas).sum()
    union = np.logical_or(a_canvas, b_canvas).sum()
    return float(intersection / union) if union else 0.0


def _binary_bbox_crop(mask_path: Union[str, Path]):
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return binary[y0:y1, x0:x1]


def _center_on_canvas(binary_crop: np.ndarray, canvas: int) -> np.ndarray:
    h, w = binary_crop.shape
    scale = canvas / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(binary_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    out = np.zeros((canvas, canvas), dtype=np.uint8)
    y_off = (canvas - new_h) // 2
    x_off = (canvas - new_w) // 2
    out[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return out > 0
