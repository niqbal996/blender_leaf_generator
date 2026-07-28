"""Measure a leaf's physical size from its mask + capture calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import cv2

from .calibration import SessionCalibration


def measure_leaf_bbox_px(mask_path: Union[str, Path]) -> Dict[str, float]:
    """Tight bounding box of the leaf silhouette in the mask, in pixels.

    This is deliberately the silhouette's own bounding box, not the full
    (padded) image canvas -- ROI crops typically include a margin of
    background around the leaf, which would otherwise inflate the size
    estimate.
    """
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")

    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No foreground contour found in mask: {mask_path}")

    contour = max(contours, key=cv2.contourArea)
    _, _, w, h = cv2.boundingRect(contour)
    area_px = cv2.contourArea(contour)

    return {"width_px": float(w), "height_px": float(h), "area_px": float(area_px)}


def measure_leaf_size_mm(
    mask_path: Union[str, Path], calibration: SessionCalibration
) -> Dict[str, float]:
    """Leaf silhouette's bounding-box width/height and area, converted to mm
    (mm^2 for area) using the session's pixel_size_mm calibration.
    """
    bbox_px = measure_leaf_bbox_px(mask_path)
    return {
        "width_mm": calibration.px_to_mm(bbox_px["width_px"]),
        "height_mm": calibration.px_to_mm(bbox_px["height_px"]),
        "area_mm2": bbox_px["area_px"] * (calibration.pixel_size_mm ** 2),
    }
