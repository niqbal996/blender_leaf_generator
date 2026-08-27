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
    start_index: int = 0,
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
        # Offset so several capture passes can share one frames directory
        # without colliding, which is what lets P3 solve them together.
        out_path = out_dir / f"frame_{start_index + saved:04d}.jpg"
        cv2.imwrite(str(out_path), best_per_bin[bin_index][1], [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        written.append(out_path)

    return written


def ingest_photos(
    photo_dir: Union[str, Path],
    out_dir: Union[str, Path],
    start_index: int = 0,
    max_edge: int = 1920,
    jpeg_quality: int = 95,
) -> List[Path]:
    """Bring a directory of still photos into P1's frame contract.

    Filename order must be capture order around the turntable: SAM2's video
    propagation and P3's orbit checks both assume consecutive frames are
    neighbouring angles, and stills carry no other ordering.

    `max_edge` resizes each photo so its long edge is at most this many
    pixels; 0 keeps the original. The default matches the 1920px video path,
    which is what every downstream default was fitted against, and at current
    settings costs no detail: SAM2 resizes its input to 1024 regardless, P3
    caps SIFT at `--max-image-size` (1920), and P4b trains at `--downsample`
    (2). What a 24 MP frame *does* change is cost -- 11x the pixels per view
    through P4a's carve and P4b's rasteriser, which is a VRAM wall rather
    than a slow run on an 8 GB card. Raise it in step with those settings if
    you want the extra resolution to reach anything.

    Intrinsics stay consistent either way: COLMAP records the camera at the
    image's real dimensions even when `max_image_size` makes it extract on a
    smaller copy (verified on these photos: 6000x4000 in, 6000x4000 camera),
    so full-size frames are not silently mismatched against their masks.

    Unlike a video there is no redundancy to pick the sharpest frame from, so
    each photo's sharpness is measured (same metric P1 uses on video) and the
    softest are named in the summary rather than silently kept: culling a
    hopeless photo is the operator's call, because dropping it also widens
    the angular gap that `full_rotation_covered` checks.
    """
    import shutil

    photo_dir, out_dir = Path(photo_dir), Path(out_dir)
    photos = sorted(p for p in photo_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg"))
    if not photos:
        raise FileNotFoundError(f"no .jpg photos in {photo_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    scores: List[Tuple[float, str]] = []
    resized_to = None
    for i, src in enumerate(photos):
        dest = out_dir / f"frame_{start_index + i:04d}.jpg"
        bgr = cv2.imread(str(src))
        if bgr is None:
            raise ValueError(f"could not read {src}")
        height, width = bgr.shape[:2]

        scale = 1.0 if not max_edge else min(1.0, max_edge / max(width, height))
        if scale < 1.0:
            bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)),
                             interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dest), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            resized_to = f"{bgr.shape[1]}x{bgr.shape[0]}"
        else:
            # Copy the bytes rather than re-encode: no generation loss, and
            # nothing to gain from a second JPEG pass.
            shutil.copyfile(src, dest)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Scored at one common size so the numbers mean the same thing across
        # capture sessions regardless of megapixels.
        common = 1600.0 / max(gray.shape[1], gray.shape[0])
        if common < 1.0:
            gray = cv2.resize(gray, None, fx=common, fy=common, interpolation=cv2.INTER_AREA)
        scores.append((sharpness(gray), src.name))
        written.append(dest)

    if resized_to:
        print(f"    resized {width}x{height} -> {resized_to} (--photo-max-edge 0 keeps full size)")

    values = sorted(scores)
    median = values[len(values) // 2][0]
    print(f"    {len(written)} photos ingested; sharpness median {median:.1f}, "
          f"range {values[0][0]:.1f}..{values[-1][0]:.1f}")
    soft = [f"{name} ({score:.0f})" for score, name in values[:5] if score < 0.5 * median]
    if soft:
        print(f"    much softer than the rest (worth culling and re-ingesting): {', '.join(soft)}")
    return written
