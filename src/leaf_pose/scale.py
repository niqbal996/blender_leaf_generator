"""Millimetres per pixel, from the fiducial markers lying in the frame.

The markers are in the photograph, in the same plane as the leaves, so they
measure the thing that actually matters: the scale *at the subject*. A focal
length and a tape measure to the tripod do not -- the distance wanted is to
the lens's rear principal plane, which is somewhere inside the barrel.

gaensefuss_31 carries two AprilTag 36h11 markers, one at each top corner.
Both are read and their side lengths averaged, which also cross-checks the
setup: markers at opposite corners of a flat lay should measure the same,
and a disagreement beyond a per cent or so means the backing is not flat or
the camera is not square to it.

The marker's printed size is not in the photograph and cannot be inferred, so
`--marker-mm` is asked for. Without it everything downstream is still
correct, just reported in pixels: a guessed millimetre is worse than an
honest pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

# 36h11 first: it is what this rig prints, and it is the family with the
# largest Hamming distance between codes, so a false detection on leaf
# speckle is very unlikely. The ArUco dictionaries follow for other rigs.
MARKER_DICTIONARIES = (
    "DICT_APRILTAG_36h11",
    "DICT_APRILTAG_25h9",
    "DICT_4X4_50",
    "DICT_5X5_100",
    "DICT_6X6_250",
    "DICT_ARUCO_ORIGINAL",
)


@dataclass
class Marker:
    id: int
    corners: np.ndarray  # (4, 2) x/y in full-resolution pixels
    side_px: float


@dataclass
class Scale:
    mm_per_pixel: Optional[float]
    markers: List[Marker]
    dictionary: Optional[str]
    side_spread: float  # largest relative disagreement between markers, 0..1

    @property
    def pixels_per_mm(self) -> Optional[float]:
        return None if not self.mm_per_pixel else 1.0 / self.mm_per_pixel

    def to_dict(self) -> dict:
        return {
            "mm_per_pixel": self.mm_per_pixel,
            "dictionary": self.dictionary,
            "side_spread": round(self.side_spread, 5),
            "markers": [{"id": m.id, "side_px": round(m.side_px, 2),
                         "corners": m.corners.round(1).tolist()} for m in self.markers],
        }

    def mm(self, pixels: float) -> Optional[float]:
        return None if self.mm_per_pixel is None else pixels * self.mm_per_pixel


def _side_length(corners: np.ndarray) -> float:
    """Mean of the quad's four edges.

    All four rather than one: the markers sit near the corners of a wide
    frame, where even a well-aligned camera leaves a little perspective, and
    averaging the edges cancels its first-order effect on the quad.
    """
    edges = np.linalg.norm(np.diff(np.vstack([corners, corners[:1]]), axis=0), axis=1)
    return float(edges.mean())


def detect_markers(grey: np.ndarray, dictionaries: Sequence[str] = MARKER_DICTIONARIES):
    """First dictionary that finds anything, and what it found.

    Trying families in turn rather than requiring one to be named: a capture
    session changes its printed markers about as often as it changes its
    camera, and having to remember which is a step that can be wrong.
    """
    parameters = cv2.aruco.DetectorParameters()
    for name in dictionaries:
        if not hasattr(cv2.aruco, name):
            continue
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)), parameters)
        corners, ids, _rejected = detector.detectMarkers(grey)
        if ids is not None and len(ids):
            found = [Marker(id=int(i), corners=c.reshape(4, 2).astype(float),
                            side_px=_side_length(c.reshape(4, 2)))
                     for c, i in zip(corners, ids.ravel())]
            return name, found
    return None, []


def measure_scale(
    flat: np.ndarray,
    marker_mm: Optional[float] = None,
    max_side: int = 4000,
) -> Scale:
    """Detect the markers and, given their printed size, the scale.

    Detection runs on a downscaled copy and the corners are scaled back up.
    A 61 megapixel frame is not more detectable than a 4000-pixel one -- the
    marker is hundreds of pixels across either way -- and the detector's
    adaptive threshold takes seconds at full size for no gain.
    """
    from .raw import tone_map

    longest = max(flat.shape[:2])
    scale = 1.0 if longest <= max_side else max_side / float(longest)
    work = (cv2.resize(flat, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0 else flat)
    grey = cv2.cvtColor(tone_map(work), cv2.COLOR_RGB2GRAY)

    name, markers = detect_markers(grey)
    for marker in markers:
        marker.corners = marker.corners / scale
        marker.side_px = marker.side_px / scale

    if not markers:
        return Scale(mm_per_pixel=None, markers=[], dictionary=None, side_spread=0.0)

    sides = np.array([m.side_px for m in markers])
    spread = float((sides.max() - sides.min()) / sides.mean()) if len(sides) > 1 else 0.0
    mm_per_pixel = float(marker_mm / sides.mean()) if marker_mm else None
    return Scale(mm_per_pixel=mm_per_pixel, markers=markers,
                 dictionary=name, side_spread=spread)
