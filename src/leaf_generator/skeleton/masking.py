"""Per-frame foreground masking, so COLMAP only matches plant pixels.

Two strategies:
- `black_background_mask`: for a plant on a turntable against a black/dark
  background (the planned real capture setup) -- simple brightness
  thresholding, robust.
- `vegetation_mask`: best-effort fallback for messy backgrounds (e.g. a hand
  holding the plant in frame) -- an Excess Green Index (ExG = 2G - R - B)
  threshold, a standard vegetation-segmentation heuristic from agricultural
  computer vision. It works because skin and blurred indoor backgrounds
  aren't particularly green, but it is not a precise hand/plant segmenter
  and will sometimes include green-ish background clutter or exclude
  reddish/pale plant parts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

# COLMAP's mask convention: mask file for `images/<name>` lives at
# `<mask_dir>/<name>.png`; pixels with value 0 are ignored, nonzero pixels
# are used for feature extraction.
COLMAP_MASK_SUFFIX = ".png"


def vegetation_mask(
    image_bgr: np.ndarray,
    exg_threshold: float = 0.08,
    open_close_kernel: int = 5,
) -> np.ndarray:
    """Binary mask (255=keep, 0=ignore) of the "greenest" connected region."""
    img = image_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    total = r + g + b + 1e-6
    rn, gn, bn = r / total, g / total, b / total
    exg = 2 * gn - rn - bn  # roughly in [-2, 2]

    mask = (exg > exg_threshold).astype(np.uint8) * 255

    kernel = np.ones((open_close_kernel, open_close_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return _keep_largest_component(mask)


def black_background_mask(
    image_bgr: np.ndarray,
    brightness_threshold: int = 30,
    open_close_kernel: int = 5,
) -> np.ndarray:
    """Binary mask (255=keep, 0=ignore) for a subject on a near-black
    background: anything brighter than `brightness_threshold` is foreground.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((open_close_kernel, open_close_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return _keep_largest_component(mask)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def write_colmap_masks(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    mode: str = "vegetation",
    **mask_kwargs,
) -> int:
    """Compute and write a COLMAP-convention mask for every image in
    `image_dir` into `mask_dir`. Returns the number of masks written.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if mode == "vegetation":
        mask_fn = vegetation_mask
    elif mode == "black_background":
        mask_fn = black_background_mask
    else:
        raise ValueError(f"Unknown mask mode: {mode!r} (expected 'vegetation' or 'black_background')")

    count = 0
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        mask = mask_fn(img, **mask_kwargs)
        out_path = mask_dir / f"{image_path.name}{COLMAP_MASK_SUFFIX}"
        cv2.imwrite(str(out_path), mask)
        count += 1

    return count
