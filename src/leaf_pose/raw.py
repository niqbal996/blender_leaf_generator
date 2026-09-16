"""Ingest a capture folder: RAW frames in, two linear images out.

A capture is 2N photographs of one static scene -- N lit by each LED in turn
through *parallel* polarisers, then the same N through *crossed* ones. The
rig writes them in that order, which is also the order `rigdef_cam.xml`
declares (`sidx` 0..N-1 for the specular set, `didx` N..2N-1 for the diffuse
one), so the split is positional rather than something to read out of EXIF.

Two images come out of that, and the rest of the package is written against
them rather than against the frames:

``flat``
    the mean of the crossed-polariser set. Crossed polarisers delete the
    surface reflection, so this is the leaf's own colour, and averaging the
    12 light directions removes the directional shading along with it. It is
    the image to threshold, to matte against, and to show a human.
``specular``
    what the parallel set has that the crossed set does not, which is the
    surface reflection alone. A leaf's midrib is a raised ridge, so it
    catches that reflection differently from the lamina either side of it --
    this is evidence about *shape* in a way ``flat`` deliberately is not.

Measured on gaensefuss_31 (24 x 61 MP): the crossed set's mean frame
brightness is 2.0-4.4, the parallel set's 9.2-21.1, a ratio of about 4. That
gap is what `split_polarisation` checks itself against, because a capture
saved in the other order would otherwise invert both images silently and
every later stage would still run.

Memory is the constraint that shapes the API. One decoded 61 MP frame is
732 MB as float32 RGB, so nothing here ever holds the stack: frames are
decoded one at a time into running sums. Per-light data, which photometric
stereo does need all of at once, is cropped to the leaf ROIs first -- see
`photometric.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# Sony, Nikon, Canon, Adobe. Lower case is checked too; a card straight out
# of a camera is upper case and one that has been through a file manager is
# whatever that manager felt like.
RAW_SUFFIXES = (".arw", ".nef", ".cr2", ".cr3", ".dng", ".raf", ".rw2", ".orf")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")


class CaptureError(RuntimeError):
    """The folder does not hold a capture this package can read."""


@dataclass
class Capture:
    """Which files were found, and which polarisation each belongs to."""

    parallel: List[Path]
    crossed: List[Path]
    is_raw: bool

    @property
    def files(self) -> List[Path]:
        return list(self.parallel) + list(self.crossed)

    @property
    def num_lights(self) -> int:
        return len(self.crossed)

    @property
    def has_polarisation(self) -> bool:
        """Both sets present, so a specular residual can be formed."""
        return bool(self.parallel) and bool(self.crossed)


@dataclass
class FlatField:
    """The two images every later stage reads, plus where they came from."""

    flat: np.ndarray  # (H, W, 3) float32, linear, crossed-polariser mean
    specular: Optional[np.ndarray]  # (H, W) float32, linear, >= 0
    capture: Capture

    @property
    def shape(self) -> Tuple[int, int]:
        return self.flat.shape[0], self.flat.shape[1]


def find_frames(path: Path) -> Tuple[List[Path], bool]:
    """Every frame in `path`, sorted, and whether they are RAW.

    A single file is a capture of one. Sorting is by name, which is the
    capture order for every camera that names frames with a counter -- the
    polarisation split depends on it, so a folder holding two sessions'
    frames is a user error this cannot detect.
    """
    path = Path(path)
    if path.is_file():
        return [path], path.suffix.lower() in RAW_SUFFIXES

    if not path.is_dir():
        raise CaptureError(f"no such capture: {path}")

    raw = sorted(p for p in path.iterdir()
                 if p.is_file() and p.suffix.lower() in RAW_SUFFIXES)
    if raw:
        return raw, True

    plain = sorted(p for p in path.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if plain:
        return plain, False

    raise CaptureError(
        f"{path} holds no RAW ({', '.join(RAW_SUFFIXES)}) or image files")


def split_polarisation(
    frames: Sequence[Path],
    order: str = "auto",
    brightness: Optional[Sequence[float]] = None,
) -> Capture:
    """Split frames into the parallel set and the crossed set.

    `order` is "parallel-first" (the rig's own order), "crossed-first", or
    "auto", which uses `brightness` -- one mean per frame -- to decide.
    Crossed polarisers cost both the reflection and about half the
    transmission, so the crossed half is unambiguously the darker one; on
    gaensefuss_31 the two halves differ by 4x, far outside any variation
    between LEDs within a half.

    An odd number of frames, or a single frame, is not a polarised pair set:
    it is treated as crossed, since that is the set `flat` is built from and
    a plain photograph behaves like one (no reflection to remove is the same
    as a reflection already removed, as far as everything downstream reads).
    """
    frames = list(frames)
    if len(frames) < 2 or len(frames) % 2 == 1:
        return Capture(parallel=[], crossed=frames, is_raw=False)

    half = len(frames) // 2
    first, second = frames[:half], frames[half:]

    if order == "parallel-first":
        return Capture(parallel=first, crossed=second, is_raw=False)
    if order == "crossed-first":
        return Capture(parallel=second, crossed=first, is_raw=False)
    if order != "auto":
        raise ValueError(f"unknown polarisation order: {order!r}")

    if brightness is None:
        # No evidence to judge with: take the rig's documented order rather
        # than guess. Wrong only for a capture saved back to front, which
        # --polarisation-order exists to fix.
        return Capture(parallel=first, crossed=second, is_raw=False)

    brightness = np.asarray(brightness, dtype=np.float64)
    if len(brightness) != len(frames):
        raise ValueError("one brightness value per frame is required")
    if brightness[:half].mean() >= brightness[half:].mean():
        return Capture(parallel=first, crossed=second, is_raw=False)
    return Capture(parallel=second, crossed=first, is_raw=False)


def decode(path: Path, half_size: bool = False) -> np.ndarray:
    """One frame as linear float32 RGB in 0..1.

    Linear, not display-ready: `gamma=(1, 1)` and `no_auto_bright` are what
    make two frames of the same scene under different lights *comparable*,
    which every difference taken in this package depends on. A tone curve
    applied per frame -- which is what a default `postprocess()` does -- is a
    different, brightness-dependent function per frame, so the specular
    residual below it would be measuring the curve rather than the surface.

    The camera's own white balance is kept: it is one fixed scaling of the
    three channels, so it cancels out of every ratio taken later, and it
    makes the green of a leaf actually read as green to the colour index.
    """
    import rawpy

    path = Path(path)
    if path.suffix.lower() not in RAW_SUFFIXES:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise CaptureError(f"could not read {path}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image[:, :, :3][:, :, ::-1].astype(np.float32)
        # sRGB-encoded 8/16-bit files are linearised approximately; nothing
        # here needs colorimetric accuracy, only that a difference of two
        # frames means the same thing at both ends of the range.
        scale = 65535.0 if image.dtype == np.uint16 or image.max() > 255 else 255.0
        return np.clip(image / scale, 0.0, 1.0) ** 2.2

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            half_size=half_size,
            no_auto_bright=True,
            use_camera_wb=True,
            output_bps=16,
            gamma=(1, 1),
        )
    return rgb.astype(np.float32) / 65535.0


def frame_brightness(frames: Sequence[Path], half_size: bool = True) -> List[float]:
    """Mean level of each frame, decoded at half resolution.

    Half resolution because this only has to rank two groups that differ by a
    factor of four, and a full decode of a 61 MP frame costs ~10 seconds
    against ~3.
    """
    return [float(decode(f, half_size=half_size).mean()) for f in frames]


def probe_order(frames: Sequence[Path], samples: int = 3) -> str:
    """Which half is the parallel set, decided from a few frames of each.

    A few rather than all of them: deciding this by decoding all 24 frames
    costs as much as the run that follows, and the question is which of two
    groups is four times brighter -- three evenly spaced frames from each
    half settle that with room to spare. Evenly spaced rather than the first
    three, because the LEDs are not equally bright and three neighbours on
    the ring are a biased sample of the twelve.
    """
    half = len(frames) // 2
    if half < 2:
        return "parallel-first"

    picks = np.unique(np.linspace(0, half - 1, min(samples, half)).astype(int))
    first = np.mean([decode(frames[i], half_size=True).mean() for i in picks])
    second = np.mean([decode(frames[half + i], half_size=True).mean() for i in picks])
    return "parallel-first" if first >= second else "crossed-first"


def specular_residual(parallel: np.ndarray, crossed: np.ndarray) -> np.ndarray:
    """What the parallel set sees and the crossed set does not.

    The two sets are not on the same scale -- the crossed pair of filters
    passes less light than the parallel pair, 57.9% on this rig's own
    measurement -- so the crossed image is first scaled to match. The gain is
    fitted rather than taken from the calibration file: it is a single number
    recovered from the median of both images over the pixels that are
    actually lit, which is available whether or not a calibration was saved
    beside the capture.

    The median, not the mean: the mean is pulled by the very highlights this
    is trying to isolate, which would scale them away.

    Fitted on the brightest few per cent, not on most of the frame. On a flat
    lay the leaves are about 5% of the pixels and the rest is unlit backing,
    where both images are near zero and their ratio is the ratio of two noise
    floors -- so a gain fitted over the whole frame is a gain fitted to the
    backing. The comparison is `>=` so that a uniform image, where no pixel
    is above the percentile, still selects all of them rather than none.
    """
    lit = crossed >= np.percentile(crossed, 95)
    if not lit.any():
        return np.zeros(crossed.shape[:2], np.float32)

    numerator = float(np.median(parallel[lit]))
    denominator = float(np.median(crossed[lit]))
    gain = numerator / denominator if denominator > 1e-8 else 1.0

    residual = parallel - gain * crossed
    if residual.ndim == 3:
        residual = residual.mean(axis=2)
    return np.clip(residual, 0.0, None).astype(np.float32)


def load_flat_field(
    capture: Capture,
    half_size: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> FlatField:
    """Decode the capture into `flat` and `specular`.

    One decode per frame, accumulated in place. Peak memory is three
    full-resolution float32 images regardless of how many LEDs the rig has.
    """
    say = progress or (lambda _message: None)

    def mean_of(paths: Sequence[Path], label: str) -> Optional[np.ndarray]:
        total = None
        for index, path in enumerate(paths, start=1):
            say(f"  {label} {index}/{len(paths)}  {path.name}")
            frame = decode(path, half_size=half_size)
            total = frame if total is None else total + frame
        return None if total is None else (total / max(len(paths), 1))

    flat = mean_of(capture.crossed, "crossed")
    if flat is None:
        raise CaptureError("capture has no frames to build a flat image from")

    specular = None
    if capture.parallel:
        parallel = mean_of(capture.parallel, "parallel")
        specular = specular_residual(parallel, flat)
        del parallel

    return FlatField(flat=flat, specular=specular, capture=capture)


def tone_map(image: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """A linear image as 8-bit sRGB-ish, for looking at.

    The white point is a high percentile rather than the maximum: a single
    blown highlight -- a fibre on the backing, a marker's white quiet zone --
    otherwise sets the exposure for the whole frame and the subject goes
    black.
    """
    white = float(np.percentile(image, percentile))
    if white <= 0:
        white = float(image.max()) or 1.0
    shown = np.clip(image / white, 0.0, 1.0) ** (1.0 / 2.2)
    return (shown * 255.0).astype(np.uint8)
