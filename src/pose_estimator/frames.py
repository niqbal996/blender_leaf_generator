"""Extract frames from a rotation video."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


def _exif(photo: Union[str, Path]) -> dict:
    """EXIF as a tag-name dict, empty when there is none or the file is bad."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        with Image.open(photo) as img:
            raw = img._getexif() or {}
    except Exception:
        return {}
    return {TAGS.get(tag, tag): value for tag, value in raw.items()}


def exif_capture_info(photo: Union[str, Path]) -> Tuple[Optional[datetime], Optional[str]]:
    """(when it was shot, which body shot it), from EXIF. None where unknown.

    These two facts are what make a capture pass identifiable after the fact.
    P1 flattens every pass into one `frame_XXXX.jpg` sequence, so once ingested
    there is nothing in the frames themselves to say a pass came from a
    different shoot -- and a pass from a different shoot is not a second view
    of this plant, it is a second plant.
    """
    exif = _exif(photo)
    when = None
    for tag in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        text = exif.get(tag)
        if text:
            try:
                when = datetime.strptime(str(text), "%Y:%m:%d %H:%M:%S")
                break
            except ValueError:
                continue
    body = " ".join(str(exif[t]).strip() for t in ("Make", "Model") if exif.get(t)) or None
    return when, body


@dataclass
class IngestResult:
    """What `ingest_photos` learned, beyond the frames it wrote."""

    frames: List[Path] = field(default_factory=list)
    focals: Dict[str, float] = field(default_factory=dict)
    records: List[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)


def exif_focal_px(photo: Union[str, Path], width: int, height: int) -> Optional[float]:
    """Focal length in pixels for `photo` rendered at `width` x `height`.

    Without a prior COLMAP guesses 1.2x the long edge. On this rig that is
    wrong by up to 40% and, worse, wrong by a *different* amount per capture
    pass, because the passes are shot at different zooms. One shared camera
    then absorbs the spread into the camera positions instead of the
    intrinsics: measured on sugarbeet_3 (passes at 48/32/22mm, all solved as
    one camera at COLMAP's 2304px guess), the three orbits came back with
    rotation axes 7.7 degrees apart and P4a over-carved the hull to 0.53 IoU
    against masks that were themselves clean.

    `FocalLengthIn35mmFilm` is preferred because it needs no sensor-size
    table: 35mm film is 36mm across by definition, so the focal in pixels is
    f35 * long_edge / 36. It is a rounded integer, which on the D5600 lands
    about 2% from the focal computed from the true 23.5mm sensor -- immaterial
    for a value bundle adjustment refines, and a rounding error is a different
    thing from a 40% guess. `FocalPlaneXResolution` is the fallback for bodies
    that omit f35.

    Returns None when EXIF offers neither, which is the honest answer for
    video frames and stripped JPEGs; the caller leaves COLMAP to its default.
    """
    exif = _exif(photo)
    if not exif:
        return None

    def number(value) -> Optional[float]:
        # Pillow >=8 hands back an IFDRational, older versions a (num, den).
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1]) if value[1] else None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    long_edge = float(max(width, height))

    f35 = number(exif.get("FocalLengthIn35mmFilm"))
    if f35:
        return f35 * long_edge / 36.0

    focal_mm = number(exif.get("FocalLength"))
    x_res = number(exif.get("FocalPlaneXResolution"))
    sensor_px = number(exif.get("ExifImageWidth"))
    if focal_mm and x_res and sensor_px:
        # EXIF unit codes: 2 = inch, 3 = centimetre.
        unit = exif.get("FocalPlaneResolutionUnit", 2)
        per_mm = {2: x_res / 25.4, 3: x_res / 10.0}.get(unit)
        if per_mm:
            sensor_mm = sensor_px / per_mm
            if sensor_mm > 0:
                return focal_mm * long_edge / sensor_mm

    return None


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
) -> "IngestResult":
    """Bring a directory of still photos into P1's frame contract.

    Returns the frames written, the EXIF focal length in pixels per frame stem
    for photos that carried one, and a provenance record per frame. P3 groups
    frames by focal and seeds one COLMAP camera per group, so a capture shot
    at several zooms still solves as several honest cameras rather than one
    averaged wrong one. Frames whose EXIF is silent are simply absent from the
    focal mapping.

    The records exist because P1 is where a capture stops being traceable:
    every pass is flattened into one `frame_XXXX.jpg` sequence, and from P2
    onward nothing can tell that two frames came from different shoots. They
    carry the source path, the shutter time and the camera body forward so
    `check_capture_consistency` can still ask that question.

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
    so full-size frames are not silently mismatched against their masks. The
    focal returned alongside is scaled to whatever size was written here, so
    it stays correct across `max_edge` changes.

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
    focals: Dict[str, float] = {}
    records: List[dict] = []
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

        out_height, out_width = bgr.shape[:2]
        focal = exif_focal_px(src, out_width, out_height)
        if focal:
            focals[dest.stem] = focal
        when, body = exif_capture_info(src)
        records.append({
            "frame": dest.stem,
            "source_dir": str(photo_dir.resolve()),
            "source_file": src.name,
            "shot_at": when.isoformat() if when else None,
            "camera": body,
            "focal_px": focal,
        })

    if resized_to:
        print(f"    resized {width}x{height} -> {resized_to} (--photo-max-edge 0 keeps full size)")

    values = sorted(scores)
    median = values[len(values) // 2][0]
    print(f"    {len(written)} photos ingested; sharpness median {median:.1f}, "
          f"range {values[0][0]:.1f}..{values[-1][0]:.1f}")
    soft = [f"{name} ({score:.0f})" for score, name in values[:5] if score < 0.5 * median]
    if soft:
        print(f"    much softer than the rest (worth culling and re-ingesting): {', '.join(soft)}")

    if focals:
        distinct = sorted(set(round(f, 1) for f in focals.values()))
        shown = ", ".join(f"{f:.0f}px" for f in distinct[:4])
        more = "" if len(distinct) <= 4 else f" (+{len(distinct) - 4} more)"
        print(f"    EXIF focal: {shown}{more} at {resized_to or f'{width}x{height}'}"
              f"{'' if len(focals) == len(written) else f' -- {len(written) - len(focals)} photo(s) without EXIF'}")
    else:
        print("    no EXIF focal in these photos -- P3 will fall back to COLMAP's own guess")

    return IngestResult(frames=written, focals=focals, records=records)


class CaptureMismatch(ValueError):
    """The passes handed to one workdir do not look like one capture."""


def check_capture_consistency(
    records_per_pass: List[List[dict]], max_pass_gap_minutes: float = 30.0
) -> List[str]:
    """Fail when several passes cannot plausibly be one shoot of one plant.

    P1 is the last place this is answerable. Afterwards every pass is just
    `frame_XXXX.jpg` in one directory, and a pass belonging to a *different
    specimen* is indistinguishable from a second elevation of this one -- so
    it flows through P2 (which masks it happily), into P3, and shows up only
    as a solve that will not settle.

    That is not hypothetical. Measured on sugarbeet_4: 40 of its own frames
    plus 13 frames of sugarbeet_3, shot an hour earlier and passed in by a
    mistyped `--photos`. COLMAP refused to register the 13 -- and still had
    its orbit wrecked by them, because the two shoots share a turntable, a
    pair of pliers and a backdrop, so the spurious cross-matches are on real
    repeated structure. Camera-centre RMS off the fitted circle went from
    0.05% of radius on the 40 alone to 7.01% with the 13 present, and the
    hull fell to 0.471 IoU. Every phase reported success.

    The checks, in order of how certain they are:

    - **Out-of-order passes.** A pass that starts before its predecessor ends
      cannot be a later pass of the same shoot. No threshold, no judgement.
    - **A gap between passes.** Consecutive passes of one plant follow each
      other immediately: measured 1.0-1.3 min on sugarbeet_3 and sugarbeet_4,
      against 57 min for the foreign pass. The default limit sits far above
      the former and far below the latter, and a long gap is worth stopping
      for anyway -- a plant wilts and droops, and P3 has no way to model it.
    - **Two camera bodies.** One rig, one body.
    - **Filename order that is not capture order.** Everything downstream
      assumes consecutive frames are neighbouring angles; the README says so
      and nothing checked it until now. A folder holding two shoots, or
      renamed files, breaks SAM2 propagation and P3's circle fit silently.

    Returns the human-readable findings; raises `CaptureMismatch` on any.
    Passes with no EXIF timestamp are skipped rather than guessed at.
    """
    problems: List[str] = []

    def when(record):
        return datetime.fromisoformat(record["shot_at"]) if record.get("shot_at") else None

    bodies = {r["camera"] for pass_records in records_per_pass for r in pass_records
              if r.get("camera")}
    if len(bodies) > 1:
        problems.append(f"passes were shot on {len(bodies)} different camera bodies: "
                        + ", ".join(sorted(bodies)))

    for index, pass_records in enumerate(records_per_pass):
        times = [(when(r), r) for r in pass_records]
        times = [(t, r) for t, r in times if t is not None]
        out_of_order = [(a, b) for (ta, a), (tb, b) in zip(times, times[1:]) if tb < ta]
        if out_of_order:
            first, second = out_of_order[0]
            problems.append(
                f"pass {index}: filename order is not capture order -- "
                f"{second['source_file']} was shot before {first['source_file']} "
                f"({len(out_of_order)} such step(s)). Everything downstream assumes "
                f"consecutive frames are neighbouring angles.")

    spans = []
    for index, pass_records in enumerate(records_per_pass):
        times = sorted(t for t in (when(r) for r in pass_records) if t is not None)
        spans.append((index, times[0], times[-1], pass_records) if times else (index, None, None, pass_records))

    dated = [s for s in spans if s[1] is not None]
    for (i, _, end, before), (j, start, _, after) in zip(dated, dated[1:]):
        gap = (start - end).total_seconds() / 60.0
        where = (f"pass {i} ({Path(before[0]['source_dir']).name}) -> "
                 f"pass {j} ({Path(after[0]['source_dir']).name})")
        if gap < 0:
            problems.append(
                f"{where}: pass {j} was shot {-gap:.0f} min BEFORE pass {i} ended. "
                f"A later pass cannot precede an earlier one -- these are different "
                f"shoots, so one of them is a different specimen.")
        elif gap > max_pass_gap_minutes:
            problems.append(
                f"{where}: {gap:.0f} min between passes (limit {max_pass_gap_minutes:.0f}). "
                f"Passes of one plant follow each other immediately; this long a gap "
                f"means either a different shoot or a plant that has moved since.")

    if problems:
        raise CaptureMismatch(
            "these passes do not look like one capture of one plant:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n  Check the --photos/--video paths. Pass --allow-mixed-capture to "
              "proceed anyway if you know this is right.")
    return problems
