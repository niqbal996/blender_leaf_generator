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
the stalk and the third describes the tissue beside it. The fourth vote is
for exactly that leaf, and it reads the **margin teeth**: see
`tooth_direction`.

A leaf with none of the four signals is genuinely ambiguous, and `confidence`
says so rather than the answer being presented as certain -- which is the
same contract `leaf_generator.keypoints` reports under, for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from pose_estimator.leaf import resample_by_arclength

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
# The tooth vote's weight, and the skew magnitude at which it saturates. The
# cap matters: measured skews run from about 0.3 on a nearly entire margin to
# 11 on a strongly toothed one, and uncapped, one emphatic leaf would make
# this vote unanswerable by any amount of contrary evidence.
TOOTH_WEIGHT = 0.35
TOOTH_SATURATION = 4.0
# Resolution the margin is sampled at, and the window separating a tooth from
# the blade's own outline. Both in bins along the midrib; the answer was the
# same for every window from 9 to 61.
TOOTH_BINS = 300
TOOTH_ENVELOPE = 21


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
    # Margin-tooth asymmetry, reported in the *final* orientation, so a
    # negative value always means "the teeth agree that the tip is at the far
    # end". Near zero means an entire margin with no teeth to read.
    tooth_skew: float = 0.0

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
            "tooth_skew": round(float(self.tooth_skew), 3),
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


def margin_profiles(
    contour: np.ndarray, midrib: np.ndarray, bins: int = TOOTH_BINS,
) -> List[np.ndarray]:
    """How far the margin lies from the midrib, per side, along the leaf.

    Each contour point is assigned to its nearest point on a densely
    resampled midrib, which gives it a position *along* the leaf and a
    distance *across* it, and the side it is on comes from the sign of the
    cross product with the local tangent. Binning the across-distance by the
    along-position turns each margin into a 1-D signal that teeth appear in
    as bumps.

    Parameterising by the midrib rather than by the contour's own arclength
    is what makes the two sides comparable: teeth point toward the apex on
    both margins, but walking the contour traverses one side base-to-tip and
    the other tip-to-base, so a contour-order statistic would see them as
    opposite and cancel them out.

    The midrib is resampled to 1500 points first. Using it as stored -- 48
    stations -- gives every contour point one of 48 positions, so 300 bins
    have 48 filled and the profile is thrown away as too sparse to use. The
    symptom was a tooth statistic of exactly 0.000 for every leaf.
    """
    dense, length = resample_by_arclength(midrib, 1500)
    if length <= 0:
        return []
    along = np.linspace(0.0, length, len(dense))

    tangents = np.gradient(dense, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)

    nearest = np.linalg.norm(contour[:, None, :] - dense[None, :, :],
                             axis=2).argmin(axis=1)
    offset = contour - dense[nearest]
    across = np.linalg.norm(offset, axis=1)
    side = (tangents[nearest, 0] * offset[:, 1]
            - tangents[nearest, 1] * offset[:, 0])
    position = along[nearest]

    edges = np.linspace(0.0, length, bins + 1)
    out = []
    for sign in (+1.0, -1.0):
        keep = np.sign(side) == sign
        if keep.sum() < 60:
            continue
        index = np.clip(np.digitize(position[keep], edges) - 1, 0, bins - 1)
        profile = np.full(bins, np.nan)
        distances = across[keep]
        for b in range(bins):
            members = index == b
            if members.any():
                profile[b] = distances[members].max()
        filled = ~np.isnan(profile)
        if filled.sum() < bins * 0.5:
            continue
        out.append(np.interp(np.arange(bins), np.flatnonzero(filled),
                             profile[filled]))
    return out


def tooth_direction(contour: np.ndarray, midrib: np.ndarray) -> float:
    """Which way the margin teeth point. Negative means "toward the end".

    A leaf's marginal teeth point toward its apex. Walk a toothed margin from
    base to tip and the distance from the midrib therefore rises slowly along
    each tooth's long proximal edge and drops sharply down its short distal
    edge into the sinus -- a sawtooth with a slow rise and a fast fall. Walk
    it the other way and the asymmetry reverses.

    That asymmetry is exactly what the **skewness of the slope** measures:
    many small positive steps and a few large negative ones cube out to a
    negative skew. So a negative value means the midrib's own direction, base
    index to last index, points at the tip.

    The blade's outline is removed first -- subtracting a smoothed version of
    the profile leaves only what is finer than the leaf's own silhouette --
    because the leaf widening and then narrowing is a far larger signal than
    any tooth and is not what this is asking about.

    Two properties make it safe to add to the other votes without special
    cases. It is exactly antisymmetric: reversing the leaf negates the
    statistic, so it cannot favour an orientation by construction. And a leaf
    with an entire margin has nothing to be skewed, so it returns ~0 and
    abstains rather than voting on noise.

    Measured on gaensefuss_31 against the eight leaves whose petiole is long
    enough to settle the question independently: 8/8, with skews from -0.6 to
    -9.3. The control that matters is that re-running it with the petiole
    excluded from the profile *strengthened* every one of those eight
    (-1.3 to -10.8), which is what rules out its having simply re-detected
    the stalk -- a large asymmetric feature at one end that would have
    produced the same 8/8 for the wrong reason.
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.stats import skew

    values = []
    for profile in margin_profiles(contour, midrib):
        residual = profile - uniform_filter1d(profile, size=TOOTH_ENVELOPE,
                                              mode="nearest")
        slope = np.diff(residual)
        if slope.std() < 1e-9:
            continue
        values.append(float(skew(slope)))
    return float(np.mean(values)) if values else 0.0


def sample_along(path: np.ndarray, field: np.ndarray) -> np.ndarray:
    """A 2D map read off at each station of a polyline."""
    rows = np.clip(np.round(path[:, 1]).astype(int), 0, field.shape[0] - 1)
    cols = np.clip(np.round(path[:, 0]).astype(int), 0, field.shape[1] - 1)
    return field[rows, cols]


def locate(midrib: Midrib, greenness: Optional[np.ndarray] = None,
           contour: Optional[np.ndarray] = None) -> Tuple[Midrib, Keypoints]:
    """Orient the midrib base-to-tip and measure the leaf's pose.

    `greenness` is the leaf crop's colour index, the same array shape as the
    mask, read along the midrib for the colour vote. `contour` is the leaf's
    outline in the same crop-local coordinates as `midrib.path`, for the
    tooth vote. Both may be left out; each vote simply does not happen.

    Returns the midrib as it should be read from now on -- station 0 at the
    petiole's cut end -- together with the keypoints.
    """
    profile = None if greenness is None else sample_along(midrib.path, greenness)

    forward = end_score(midrib.width, profile)
    backward = end_score(midrib.width[::-1],
                         None if profile is None else profile[::-1])

    # The tooth statistic is antisymmetric under reversal, so one evaluation
    # settles both orientations: it is added to one score and subtracted from
    # the other. Saturated first, so an emphatically toothed leaf contributes
    # a bounded amount rather than an overwhelming one.
    tooth = 0.0 if contour is None else tooth_direction(contour, midrib.path)
    vote = TOOTH_WEIGHT * float(np.clip(-tooth / TOOTH_SATURATION, -1.0, 1.0))
    forward += vote
    backward -= vote

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
        tooth_skew=-tooth if flipped else tooth,
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
