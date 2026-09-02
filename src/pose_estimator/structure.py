"""P5 -- the plant's own coordinate frame: which way is up, and where is zero.

Everything here runs before structure extraction and is shared by whatever
backend does it. Two questions have to be answered first, and both are
answered from data already solved rather than from an assumption about how
the rig was set up:

1. **Get upright.** The orbit axis from P3 fixes the plant's axis but not its
   sign -- it comes from an SVD, so it points up on one specimen and down on
   the next. Getting it wrong would invert every insertion angle downstream
   while leaving the geometry looking perfectly plausible. Resolved against
   the table plane, which is the modal height of the P3 sparse cloud.

2. **Find the clamp line.** The pliers occlude a band of stem, so the carve
   leaves a gap there. That gap is not damage to route around: it is a direct
   observation of where the holder grips, which is exactly the origin the
   output schema calls for ("stem base at substrate/clamp line"). When no gap
   is detectable the caller is told, so it can say so rather than quietly
   presenting a guess as a measurement.

Stem tracing and leaf instancing live in `structure_labels.py`, driven by the
P4c organ labels. The earlier geometry-only skeletoniser was removed: it
inferred which branches were leaves through a chain of thresholds
(`min_branch_fraction` and friends) that had to be re-calibrated per specimen
and never showed a stable plateau, which is the failure mode the labelled
path exists to eliminate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PlantFrame:
    """Transform from the raw COLMAP frame into an upright plant frame."""

    origin: np.ndarray  # world point mapped to (0,0,0): the clamp line
    rotation: np.ndarray  # (3,3), rows are the plant frame's basis vectors
    table_height: float  # table plane height in the plant frame (negative = below origin)
    clamp_height: float  # clamp line height in the raw orbit frame, for audit

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin) @ self.rotation.T

    def to_dict(self) -> dict:
        return {
            "origin": self.origin.tolist(),
            "rotation": self.rotation.tolist(),
            "table_height": float(self.table_height),
            "clamp_height_orbit_frame": float(self.clamp_height),
        }


def up_from_holder(hull_points: np.ndarray, heights: np.ndarray, cameras) -> Optional[int]:
    """Which way is up, from where the tool grips: +1, -1, or None.

    The rig grips the plant at its base, so the gripped end is the bottom.
    That is a property of how a specimen is mounted rather than of what it
    looks like, which makes it a far steadier cue than the one below.

    Each point is scored by the fraction of views in which it projects onto
    the holder mask -- gripped tissue lands on the tool from every angle,
    tissue further up only occasionally -- and the score-weighted mean height
    is compared against the plant's own. Up is whichever direction puts the
    grip underneath.

    Returns None when there are no holder masks to read, leaving the caller's
    fallback in charge.
    """
    occluders = [c for c in cameras or [] if getattr(c, "occluder", None) is not None]
    if not occluders or len(hull_points) == 0:
        return None

    hits = np.zeros(len(hull_points))
    seen = np.zeros(len(hull_points))
    for camera in occluders:
        pixels, valid = camera.project(hull_points)
        height_px, width_px = camera.occluder.shape
        x = np.round(pixels[:, 0]).astype(int)
        y = np.round(pixels[:, 1]).astype(int)
        inside = valid & (x >= 0) & (x < width_px) & (y >= 0) & (y < height_px)
        if not inside.any():
            continue
        seen[inside] += 1.0
        hits[inside] += camera.occluder[y[inside], x[inside]].astype(np.float64)

    observed = seen > 0
    if not observed.any():
        return None
    score = np.zeros(len(hull_points))
    score[observed] = hits[observed] / seen[observed]
    if score.sum() <= 0:
        return None

    gripped = float((heights * score).sum() / score.sum())
    middle = float(heights[observed].mean())
    if abs(gripped - middle) < 1e-9:
        return None
    return -1 if gripped > middle else 1


def solve_up_direction(hull_points: np.ndarray, sparse_points: np.ndarray) -> int:
    """+1 or -1: which way along the orbit axis is up, in the orbit frame.

    The orbit axis comes from an SVD of the camera centres, so its sign is
    arbitrary and does vary between specimens (observed: +Z on DSC_0009, -Z on
    DSC_0010). Getting it wrong would invert every insertion angle downstream
    while leaving the geometry looking perfectly plausible.

    The table is the one large plane in the scene and the plant necessarily
    stands on it, so the modal height of the sparse cloud gives the table and
    the hull's centre of mass gives the side the plant is on. Both come from
    already-solved data; nothing here needs a marker or an assumption about
    how the rig was set up.
    """
    counts, edges = np.histogram(sparse_points[:, 2], bins=80)
    table_height = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])
    return 1 if hull_points[:, 2].mean() > table_height else -1


def find_clamp_height(
    heights: np.ndarray, min_gap_voxels: float = 4.0, voxel: float = 1.0
) -> Optional[float]:
    """Height of the widest vertical gap in the hull, if there is a real one.

    The holder occludes a band of stem in every view, so the carve removes it
    and leaves a clean break. Returning the gap's midpoint gives the clamp
    line directly.

    Returns None when no gap exceeds `min_gap_voxels` voxels, which is the
    honest answer for a specimen whose holder never fully hid the stem --
    DSC_0010 is one, with a largest gap of 0.0001 against a 0.0025 voxel.
    Callers must handle that rather than fabricating a clamp.
    """
    if len(heights) < 2:
        return None
    ordered = np.sort(heights)
    gaps = np.diff(ordered)
    widest = int(np.argmax(gaps))
    if gaps[widest] < min_gap_voxels * voxel:
        return None
    return float(0.5 * (ordered[widest] + ordered[widest + 1]))


def solve_plant_frame(
    hull_points: np.ndarray,
    sparse_points: np.ndarray,
    orbit_origin: np.ndarray,
    orbit_rotation: np.ndarray,
    voxel: float,
    cameras=None,
) -> Tuple[PlantFrame, Optional[float]]:
    """Build the upright plant frame, with its origin on the clamp line.

    Falls back to the lowest hull point when no clamp gap is detectable, and
    reports which happened so the caller can say so rather than quietly
    presenting a guess as a measurement.
    """
    hull_orbit = (hull_points - orbit_origin) @ orbit_rotation.T
    sparse_orbit = (sparse_points - orbit_origin) @ orbit_rotation.T

    # Where the tool grips first, the table plane only as a fallback. The
    # table is estimated as the commonest height in the sparse cloud, and on
    # a leafy specimen the foliage outnumbers the table: measured on
    # sugarbeet_4's video capture, that put the "table" at z=0.279 inside the
    # plant, read the foliage as being above it, and stood the whole plant on
    # its head -- root points at 1.21-1.48 with the leaves at 0.44.
    up = up_from_holder(hull_points, hull_orbit[:, 2], cameras)
    if up is None:
        up = solve_up_direction(hull_orbit, sparse_orbit)

    # Flip the frame so +Z is up, keeping a right-handed basis.
    flip = np.diag([1.0, float(up), float(up)])
    rotation = flip @ orbit_rotation

    heights = (hull_points - orbit_origin) @ rotation.T
    clamp = find_clamp_height(heights[:, 2], voxel=voxel)

    counts, edges = np.histogram(sparse_orbit[:, 2] * up, bins=80)
    table_height = float(0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1]))

    origin_height = clamp if clamp is not None else float(heights[:, 2].min())
    origin = orbit_origin + (origin_height * rotation[2])

    frame = PlantFrame(
        origin=origin,
        rotation=rotation,
        table_height=table_height - origin_height,
        clamp_height=origin_height,
    )
    return frame, clamp
