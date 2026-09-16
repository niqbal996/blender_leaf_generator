"""Which end of the midrib is the tip, and which was joined to the stem.

The midrib path has two ends and no inherent direction. Getting them the
right way round is not cosmetic: every quantity measured from the base
outward -- the width profile, the insertion angle, the arclength parameter a
mesh is built against -- inverts silently if the path runs the wrong way, and
the result still looks like a leaf.

The discriminator is the *shape of the width profile at each end*, not the
width itself. Both ends are narrow, so a single width comparison is close to
a coin toss on a young leaf. What differs is how the narrowness is arranged:

- the petiole is a stalk. Its width is small and stays small for a long run,
  then steps up sharply where the lamina begins.
- the tip is a taper. Its width grows from zero more or less steadily, and
  there is no run at all where it is both narrow and constant.

So the rule is "which end has a longer thin run", with mean end width as a
second, weaker vote for the leaf where the petiole was cut off flush and
there is no run to find.

A third vote comes from colour, and it is worth having because it is
*independent* of the first two: a petiole is chlorophyll-poor next to the
lamina it carries -- paler, yellower, sometimes frankly red -- so the end
that is less green is the base. Checked against the leaves whose shape is
unambiguous on gaensefuss_31, colour agreed on all 11 of them, with the base
reading a quarter of the tip's excess green on the largest.

All three degrade together on the leaf that has no petiole at all -- cut off,
or never attached when the leaf was laid out -- because two of them describe
the stalk and the third describes the tissue beside it. That leaf is reported
with a low `confidence` and is genuinely ambiguous to this code.

**A fourth vote was tried here and removed.** A leaf's marginal teeth point
toward its apex, so the asymmetry of the margin profile -- slow rise along
each tooth's long proximal flank, sharp fall down its short distal one --
looks like a direction cue that needs no petiole at all. It validated 8/8 on
gaensefuss_31's unambiguous leaves and was shipped.

It is wrong, and the way it is wrong is instructive. What the statistic
actually measures on most leaves is the **step where the blade narrows into
the petiole**: one large asymmetric slope event, which dominates a cubed
moment and always points the "tip" at the stalk. On gaensefuss that error
happened to agree with the real teeth. On vogelmeere_1 -- Stellaria, an
entire margin with no teeth to read -- it was right on 3 of 15 leaves, which
is worse than abstaining, and it overturned six stalks that were
unmistakable. Its magnitude gave no warning either: a median of 1.74 on the
untoothed species against 2.81 on the toothed one, so no threshold separates
"reading teeth" from "reading the petiole step". Trimming the petiole out
first made it 0/15; a quartile-based statistic immune to the single step
event dropped the toothed case to 1/8.

Against the leaves where the stalk settles the answer independently, the
three votes below score 15/15 on vogelmeere and 8/8 on gaensefuss; adding the
teeth made that 9/15 and 8/8. The cue was contributing nothing where the
answer was already known and inverting it where it was not.

A leaf with none of the three signals is genuinely ambiguous, and `confidence`
says so rather than the answer being presented as certain -- which is the
same contract `leaf_generator.keypoints` reports under, for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .midrib import Midrib

# Fraction of the maximum blade width below which the midrib is running
# through a stalk rather than a lamina. 0.25 rather than something tighter
# because a petiole thickens towards the blade and a hard cut at 0.1 stops
# the run early, understating exactly the signal being measured.
STALK_WIDTH_FRACTION = 0.25
# How much of each end to average for the fallback votes.
END_FRACTION = 0.12
# How much the colour vote counts against the two shape votes. Half, so that
# it cannot overturn a clear stalk -- which is the more direct evidence, and
# which a red-petioled or a variegated leaf would otherwise lose to colour.
COLOUR_WEIGHT = 0.5


@dataclass
class Keypoints:
    """The two ends, and the leaf's pose in the image plane.

    Coordinates are bbox-local x/y at full resolution, matching `Midrib`.
    """

    petiole_origin: np.ndarray  # (2,) the cut end that met the stem
    blade_base: np.ndarray      # (2,) where the petiole gives way to lamina
    tip: np.ndarray             # (2,) the far end of the blade
    petiole_length: float       # pixels along the midrib
    blade_length: float         # pixels along the midrib
    blade_width_max: float      # pixels
    orientation_deg: float      # petiole origin -> tip, CCW from +x, y up
    petiole_angle_deg: float    # petiole direction against the blade's chord
    confidence: float           # 0 ambiguous .. 1 unmistakable
    flipped: bool               # whether the input path had to be reversed

    def to_dict(self) -> dict:
        return {
            "petiole_origin": [float(v) for v in self.petiole_origin],
            "blade_base": [float(v) for v in self.blade_base],
            "tip": [float(v) for v in self.tip],
            "petiole_length_px": round(float(self.petiole_length), 2),
            "blade_length_px": round(float(self.blade_length), 2),
            "blade_width_max_px": round(float(self.blade_width_max), 2),
            "orientation_deg": round(float(self.orientation_deg), 2),
            "petiole_angle_deg": round(float(self.petiole_angle_deg), 2),
            "confidence": round(float(self.confidence), 4),
            "flipped": bool(self.flipped),
        }


def stalk_run(width: np.ndarray, fraction: float = STALK_WIDTH_FRACTION) -> float:
    """How far from the start of the profile the blade stays stalk-thin, 0..1.

    Read as a fraction of the profile's length so leaves of different sizes
    are comparable, which is what lets one threshold serve a 1400-pixel leaf
    and a 60-pixel one.
    """
    if len(width) == 0:
        return 0.0
    peak = float(width.max())
    if peak <= 0:
        return 0.0
    thin = width < fraction * peak
    if not thin[0]:
        return 0.0
    # First station that is not thin; everything before it is the run.
    thick = np.flatnonzero(~thin)
    length = int(thick[0]) if len(thick) else len(width)
    return length / float(len(width))


def end_score(width: np.ndarray, greenness: Optional[np.ndarray] = None) -> float:
    """How petiole-like the start of these profiles looks, higher is more so.

    Votes on one 0..1 scale so they can simply be added: the length of the
    thin run, how thin the end is on average, and -- when a colour profile is
    given -- how far from the leaf's own green the end is. The run dominates
    when there is one, and the other two decide when there is not.
    """
    run = stalk_run(width)
    peak = float(width.max()) or 1.0
    span = max(1, int(round(END_FRACTION * len(width))))
    score = run + (1.0 - float(width[:span].mean()) / peak)

    if greenness is not None and len(greenness) == len(width):
        greenest = float(greenness.max())
        if greenest > 1e-6:
            pallor = 1.0 - float(greenness[:span].mean()) / greenest
            score += COLOUR_WEIGHT * np.clip(pallor, 0.0, 1.0)
    return score


def sample_along(path: np.ndarray, field: np.ndarray) -> np.ndarray:
    """A 2D map read off at each station of a polyline."""
    rows = np.clip(np.round(path[:, 1]).astype(int), 0, field.shape[0] - 1)
    cols = np.clip(np.round(path[:, 0]).astype(int), 0, field.shape[1] - 1)
    return field[rows, cols]


def locate(midrib: Midrib, greenness: Optional[np.ndarray] = None
           ) -> Tuple[Midrib, Keypoints]:
    """Orient the midrib base-to-tip and measure the leaf's pose.

    `greenness` is the leaf crop's colour index, the same array shape as the
    mask, read along the midrib for the colour vote; it may be left out, and
    then that vote simply does not happen.

    Returns the midrib as it should be read from now on -- station 0 at the
    petiole's cut end -- together with the keypoints.
    """
    profile = None if greenness is None else sample_along(midrib.path, greenness)

    forward = end_score(midrib.width, profile)
    backward = end_score(midrib.width[::-1],
                         None if profile is None else profile[::-1])
    flipped = backward > forward

    path = midrib.path[::-1].copy() if flipped else midrib.path.copy()
    width = midrib.width[::-1].copy() if flipped else midrib.width.copy()
    oriented = Midrib(path=path, width=width, arclength=midrib.arclength,
                      ridge_support=midrib.ridge_support)

    best, worst = max(forward, backward), min(forward, backward)
    confidence = (best - worst) / best if best > 1e-9 else 0.0

    # Where the stalk ends is where the blade begins. With no stalk the
    # blade begins at the cut, which is the literal truth for a leaf whose
    # petiole was removed and keeps petiole_length at 0 rather than inventing
    # one.
    run = stalk_run(width)
    junction = min(int(round(run * len(width))), len(width) - 1)

    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    petiole_length = float(cumulative[junction])
    blade_length = float(cumulative[-1] - cumulative[junction])

    petiole_origin, blade_base, tip = path[0], path[junction], path[-1]

    # Image rows increase downward; negating y reports the angle the way it
    # is drawn, so a leaf pointing up the frame reads as +90 rather than -90.
    axis = tip - petiole_origin
    orientation = float(np.degrees(np.arctan2(-axis[1], axis[0])))

    stalk = blade_base - petiole_origin
    chord = tip - blade_base
    if np.linalg.norm(stalk) < 1e-6 or np.linalg.norm(chord) < 1e-6:
        petiole_angle = 0.0
    else:
        cosine = float(np.dot(stalk, chord)
                       / (np.linalg.norm(stalk) * np.linalg.norm(chord)))
        petiole_angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    return oriented, Keypoints(
        petiole_origin=petiole_origin,
        blade_base=blade_base,
        tip=tip,
        petiole_length=petiole_length,
        blade_length=blade_length,
        blade_width_max=float(width.max()),
        orientation_deg=orientation,
        petiole_angle_deg=petiole_angle,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        flipped=flipped,
    )
