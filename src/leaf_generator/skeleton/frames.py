"""Extract evenly-spaced frames from a rotation video."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import cv2


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
