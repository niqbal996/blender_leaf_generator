"""Estimate the stem-attachment keypoint (petiole cut point) for a leaf mask.

Leaves in this rig are photographed as isolated blades: one end is broad
(the tip of the leaf), the other tapers to where it was cut from the stem.
We estimate that narrow end from the mask alone:

1. Fit the leaf's principal axis (PCA over the contour points).
2. Measure the blade's width in bins along that axis.
3. Whichever end has the smaller average width is the narrow / attachment
   end; the keypoint is the centroid of the contour points near that end.

This is a heuristic, not a measurement of an actual petiole (there usually
isn't one left in the crop), so it also reports a `confidence` (how
strongly tapered the two ends are) so obviously ambiguous leaves (e.g.
near-symmetric blades) can be flagged for manual review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np


class NoContourError(ValueError):
    """Raised when no foreground contour can be found in a mask image."""


def estimate_attachment_point(
    mask_path: Union[str, Path],
    tip_fraction: float = 0.12,
    sample_bins: int = 40,
) -> Dict:
    """Estimate the narrow-end attachment keypoint from a binary leaf mask.

    Returns a dict with:
        pixel: (x, y) attachment point in source-image pixel coordinates
        image_size: (width, height) of the mask image
        narrow_width / broad_width: mean blade width near each end (px)
        confidence: 0 (symmetric, ambiguous) .. 1 (strongly tapered)
    """
    mask_path = Path(mask_path)
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")

    height, width = img.shape
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise NoContourError(f"No foreground contour found in mask: {mask_path}")

    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2).astype(np.float64)

    centroid = points.mean(axis=0)
    centered = points - centroid

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, int(np.argmax(eigvals))]
    minor_axis = np.array([-major_axis[1], major_axis[0]])

    proj_major = centered @ major_axis
    proj_minor = centered @ minor_axis

    proj_min, proj_max = proj_major.min(), proj_major.max()
    axis_length = proj_max - proj_min
    if axis_length <= 0:
        raise NoContourError(f"Degenerate contour (zero extent) in mask: {mask_path}")

    bin_edges = np.linspace(proj_min, proj_max, sample_bins + 1)
    widths = np.zeros(sample_bins)
    for i in range(sample_bins):
        in_bin = (proj_major >= bin_edges[i]) & (proj_major < bin_edges[i + 1])
        if np.any(in_bin):
            widths[i] = proj_minor[in_bin].max() - proj_minor[in_bin].min()

    tip_bins = max(1, int(round(sample_bins * tip_fraction)))
    start_width = float(widths[:tip_bins].mean())
    end_width = float(widths[-tip_bins:].mean())

    tip_span = tip_fraction * axis_length
    if start_width <= end_width:
        tip_selector = proj_major <= (proj_min + tip_span)
        narrow_width, broad_width = start_width, end_width
    else:
        tip_selector = proj_major >= (proj_max - tip_span)
        narrow_width, broad_width = end_width, start_width

    tip_pixel = points[tip_selector].mean(axis=0)
    max_width = float(widths.max()) if widths.max() > 0 else 1.0
    confidence = abs(end_width - start_width) / max_width

    return {
        "pixel": (float(tip_pixel[0]), float(tip_pixel[1])),
        "image_size": (int(width), int(height)),
        "narrow_width": narrow_width,
        "broad_width": broad_width,
        "confidence": float(confidence),
    }


def pixel_to_local(
    pixel: Tuple[float, float],
    image_size: Tuple[int, int],
    scale_x: float,
    scale_y: float,
) -> Tuple[float, float, float]:
    """Map a mask-image pixel coordinate to the same local mesh space used
    by `leaf_generator.blender.mesh.create_contour_based_mesh`.
    """
    x_pixel, y_pixel = pixel
    width, height = image_size
    x_local = (x_pixel / width - 0.5) * scale_x
    y_local = (y_pixel / height - 0.5) * scale_y
    return (x_local, y_local, 0.0)
