"""Extract frames from a rotation video."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np


def extract_frames(
    video_path: Union[str, Path],
    out_dir: Union[str, Path],
    target_frame_count: int = 60,
    jpeg_quality: int = 95,
) -> List[Path]:
    """Save `target_frame_count` evenly-spaced frames from `video_path` into
    `out_dir` as `frame_0000.jpg`, `frame_0001.jpg`, ...

    Returns the list of written frame paths.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video reports zero frames (corrupt file?): {video_path}")

    step = max(1, total // target_frame_count)
    written: List[Path] = []

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            out_path = out_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            written.append(out_path)
            saved += 1
        idx += 1
    cap.release()

    return written


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian -- the standard cheap focus measure. Higher
    is sharper; it collapses toward zero as an image blurs, because blurring
    is exactly what removes the high-frequency content the Laplacian responds
    to.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_sharpest_frames(
    video_path: Union[str, Path],
    out_dir: Union[str, Path],
    target_frame_count: int = 96,
    jpeg_quality: int = 95,
    roi: Union[Tuple[int, int, int, int], None] = None,
) -> List[Path]:
    """Split the video into `target_frame_count` consecutive bins and save the
    single sharpest frame from each, as `frame_0000.jpg`, `frame_0001.jpg`, ...

    Even sampling (`extract_frames`) takes whatever frame happens to land on
    the step boundary, which on a continuously-moving turntable is a coin
    flip between a crisp frame and a motion-blurred one. Since a turntable
    bin spans a small rotation either way, picking the sharpest frame per bin
    costs nothing angularly and removes the single largest source of bad
    input to SfM and silhouette carving.

    `roi` (x0, y0, x1, y1) restricts the focus measure to a region -- worth
    setting to the subject's bounding box, since a large out-of-focus
    background can otherwise dominate the variance and make the ranking
    meaningless.

    Returns the list of written frame paths, in capture order.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video reports zero frames (corrupt file?): {video_path}")

    bin_size = max(1, total // target_frame_count)

    # Keep only the current bin's best frame in memory, not the whole video.
    best_per_bin: dict = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bin_index = idx // bin_size
        if bin_index < target_frame_count:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if roi is not None:
                x0, y0, x1, y1 = roi
                gray = gray[y0:y1, x0:x1]
            score = sharpness(gray)
            if bin_index not in best_per_bin or score > best_per_bin[bin_index][0]:
                best_per_bin[bin_index] = (score, frame.copy())
        idx += 1
    cap.release()

    written: List[Path] = []
    for saved, bin_index in enumerate(sorted(best_per_bin)):
        out_path = out_dir / f"frame_{saved:04d}.jpg"
        cv2.imwrite(str(out_path), best_per_bin[bin_index][1], [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        written.append(out_path)

    return written
