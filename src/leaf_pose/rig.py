"""The light rig: where the LEDs are, and how far away the subject is.

`rigdef_cam.xml` sits beside a capture session and lists each LED's position
in millimetres in a frame centred on the camera, with all of them at z = 0 --
they are mounted in the lens's own plane, on a ring of roughly 255 mm. It
also records the pairing this package's polarisation split relies on: `sidx`
0..N-1 are the specular (parallel) frames and `didx` N..2N-1 the diffuse
(crossed) ones, in that order.

None of this is needed to mask a leaf or to find its midrib. It is needed to
turn 12 photographs into surface normals, because a normal is only recoverable
if the direction each photograph was lit from is known.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Sensor width in millimetres. Full frame; `--sensor-width-mm` overrides it
# for a crop body. Only used to turn a measured scale into a working
# distance, and only when the capture does not say what the distance was.
FULL_FRAME_WIDTH_MM = 35.7


@dataclass
class Rig:
    """LED positions in millimetres, camera at the origin, lights at z = 0."""

    positions: np.ndarray  # (N, 3)
    source: Optional[Path] = None

    @property
    def num_lights(self) -> int:
        return len(self.positions)

    @property
    def radius_mm(self) -> float:
        return float(np.linalg.norm(self.positions[:, :2], axis=1).mean())


def find_rig(start: Path) -> Optional[Path]:
    """Nearest `rigdef_cam.xml` at or above `start`.

    Searched upward because it describes the *session*: one rig definition
    sits beside the calibration folder and serves every specimen shot that
    day, so requiring a copy in each capture folder would mean maintaining
    copies.
    """
    start = Path(start).resolve()
    for folder in [start if start.is_dir() else start.parent, *start.parents]:
        candidate = folder / "rigdef_cam.xml"
        if candidate.is_file():
            return candidate
        nested = folder / "calibration" / "rigdef_cam.xml"
        if nested.is_file():
            return nested
    return None


# Matched rather than parsed, because the file the rig writes is not
# well-formed XML: `framelabel="<cam>_<frame>"` and `spath="<frame>/lit/68.jpg"`
# put raw angle brackets inside attribute values, which every conforming
# parser rejects -- and the file is the rig's output, so it is not ours to
# correct. Those same stray brackets rule out matching a whole `<light .../>`
# element too, so the text is cut at each `<light` and the attributes are
# read from the span up to the next one. Only `sidx` and `pos` are looked at,
# which is the other reason matching is enough here.
_ATTRIBUTE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def load_rig(path: Path) -> Rig:
    """The `<light pos="x y z" />` entries, in `sidx` order."""
    entries = []
    for chunk in re.split(r"<light\b", Path(path).read_text())[1:]:
        attributes = dict(_ATTRIBUTE.findall(chunk))
        if "pos" not in attributes:
            continue
        position = [float(v) for v in attributes["pos"].split()]
        entries.append((int(attributes.get("sidx", len(entries))), position))
    if not entries:
        raise ValueError(f"{path} declares no <light> positions")
    entries.sort(key=lambda item: item[0])
    return Rig(positions=np.array([p for _, p in entries], dtype=float), source=path)


def pixel_pitch_mm(image_width_px: int,
                   sensor_width_mm: float = FULL_FRAME_WIDTH_MM) -> float:
    """Millimetres per pixel *at the sensor*, for an image this wide.

    Derived from the image rather than hard-coded from the camera's native
    pixel count, because the decode does not have to be native: `--half-size`
    halves the width, which halves the number of pixels the same sensor is
    divided into and therefore doubles the pitch. Using the native figure
    against a half-size image inflates the magnification twofold, and the
    working distance with it -- it read 2036 mm for a copy stand about
    600 mm tall.
    """
    return sensor_width_mm / float(image_width_px)


def working_distance_mm(
    mm_per_pixel: float,
    focal_length_mm: float,
    image_width_px: int,
    sensor_width_mm: float = FULL_FRAME_WIDTH_MM,
) -> float:
    """How far the subject was from the lens, from the measured scale.

    The markers in the frame give millimetres per pixel at the subject; the
    sensor's width over the image's gives millimetres per pixel at the
    sensor. Their ratio is the magnification, and the thin-lens relation
    turns that into a distance:  m = f / (z - f),  so  z = f (1/m + 1).

    Worth deriving rather than asking for, because the number wanted is the
    distance to the *subject* from the lens's rear principal plane, and a tape
    measure to the front of the barrel is not that.
    """
    magnification = pixel_pitch_mm(image_width_px, sensor_width_mm) / mm_per_pixel
    if magnification <= 0:
        raise ValueError("scale must be positive")
    return float(focal_length_mm * (1.0 / magnification + 1.0))


def scale_from_optics(
    focal_length_mm: float,
    distance_mm: float,
    image_width_px: int,
    sensor_width_mm: float = FULL_FRAME_WIDTH_MM,
) -> float:
    """Millimetres per pixel at the subject, from the optics alone.

    The inverse of `working_distance_mm`, and the way to get real units out
    of a capture with no fiducial markers in it -- a copy stand's height is
    something you can measure once and reuse for every capture on it.

    Less trustworthy than the markers, and for a specific reason: `z` here is
    to the lens's rear principal plane, which is somewhere inside the barrel,
    so a tape measure to the front of the lens is short by an unknown amount.
    On a 50 mm lens at 600 mm, being 30 mm out is a 5% error in every length
    reported. The markers lie in the subject's own plane and have no such
    term, so they win wherever they are available.
    """
    if distance_mm <= focal_length_mm:
        raise ValueError("the subject cannot be closer than the focal length")
    magnification = focal_length_mm / (distance_mm - focal_length_mm)
    return pixel_pitch_mm(image_width_px, sensor_width_mm) / magnification


def light_directions(
    rig: Rig,
    shape: tuple,
    mm_per_pixel: float,
    distance_mm: float,
    origin_xy: tuple = (0.0, 0.0),
    centre_xy: Optional[tuple] = None,
) -> np.ndarray:
    """Unit vector from each pixel to each light. Returns (N, H, W, 3).

    Near-field, not the usual distant-light approximation, because on this
    rig the approximation does not hold: the LEDs sit 255 mm off-axis at a
    working distance of roughly 600 mm, and the subject spans some 400 mm of
    the frame. A leaf at the left edge and one at the right therefore see the
    same LED from directions tens of degrees apart, and a single direction
    per light would bake that error straight into every normal.

    `origin_xy` is where this crop sits in the full frame, and `centre_xy` the
    frame's principal point, so a crop is treated in the geometry of the whole
    capture rather than as if it were centred.
    """
    height, width = shape[:2]
    if centre_xy is None:
        centre_xy = (width / 2.0, height / 2.0)

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    x_mm = (xs + origin_xy[0] - centre_xy[0]) * mm_per_pixel
    y_mm = (ys + origin_xy[1] - centre_xy[1]) * mm_per_pixel

    directions = np.empty((rig.num_lights, height, width, 3), np.float32)
    for index, (lx, ly, _lz) in enumerate(rig.positions):
        vector = np.stack([lx - x_mm, ly - y_mm,
                           np.full_like(x_mm, distance_mm)], axis=-1)
        norm = np.linalg.norm(vector, axis=-1, keepdims=True)
        directions[index] = vector / np.maximum(norm, 1e-9)
    return directions
