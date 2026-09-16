"""The strong primary veins -- and a clear account of what is out of reach.

**Read this before using `--veins`.** It finds the few large veins that leave
the midrib near the base and run out into the basal lobes. It does *not*
recover a leaf's secondary and tertiary venation, and on the capture it was
built against it cannot be made to. That was measured rather than assumed,
and the measurement is worth stating because it is the difference between a
result and a plausible-looking artefact:

- **The flat image is the wrong picture to look for a vein in.** It is the
  mean of twelve light directions, so it cancels shading by construction --
  and a vein is visible precisely *because* it is raised. Run on leaf 3 of
  gaensefuss_31 the flat image yielded nothing at all.
- **Chenopodium's surface competes directly with its veins.** The leaf is
  farinose: its upper face is covered in mealy bladder cells, which are a
  shape signal at a finer scale than a vein but a comparable amplitude. In a
  synthetic leaf carrying both, a ridge filter thresholded at any level
  returned *more* candidate curves when the veins were removed than when they
  were present -- the threshold was tracking the texture, not the veins.
- **What survives is the strong stuff.** Reading the crease in the
  photometric normals, smoothed past the granules, leaf 3 gives two genuine
  basal veins running into its two basal lobes, plus one false positive along
  the margin rim -- which is what `MARGIN_BAND` and `MIN_INSERTION_ANGLE`
  below exist to remove.

So: primary veins, yes, when `--photometric` ran. A venation network, no.
Two capture-side changes would make the network easy, and neither is
something this code can do for you: photograph the leaves **underside up**,
where the secondaries stand proud instead of being buried under the mealy
adaxial surface, or photograph them in **transmitted light** on a backlit
panel, which is the classical way venation is imaged and renders the whole
network as dark lines through a translucent lamina.

The insertion angle is reported against the *local* midrib tangent rather
than against the leaf's long axis, because that is the quantity that stays
meaningful on a curved leaf and the one a growth model asks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .midrib import Midrib

# Vein widths as a fraction of the leaf's half-width. Below the midrib's own
# scales, which keeps the midrib from being re-detected as its own widest
# vein, but not as far below as they were: at (0.02, 0.04, 0.07) the finest
# filter was matched to the leaf's surface granules rather than to a vein.
VEIN_SCALES = (0.05, 0.09, 0.14)
# How much the normal field is smoothed before its crease is taken, as a
# fraction of the leaf's half-width, with a floor in pixels.
#
# Much heavier than the midrib uses, and this is the single setting that
# decides whether any vein is found at all. A Chenopodium leaf is farinose --
# its upper surface is covered in mealy bladder cells -- and those granules
# are a *shape* signal of their own, at a finer scale than a vein but a
# comparable amplitude. Smoothed at the midrib's 5 pixels the crease map is
# granules; at 20 the granules are gone and the basal secondaries survive.
# Measured on leaf 3 of gaensefuss_31: 0 veins found, then 3.
VEIN_NORMAL_BLUR = 0.13
VEIN_NORMAL_BLUR_FLOOR = 6.0
# How the two evidence maps are mixed when both exist. Weighted hard toward
# the normals, because they measure shape: the granular texture that defeats
# the flat image is albedo, and averaging twelve light directions removes a
# vein's shading while leaving that texture untouched. The flat image found
# none of leaf 3's three veins on its own.
CREASE_WEIGHT = 0.75
# Ridge strength a pixel must reach to be considered vein at all, as a
# percentile of the response inside the leaf. High, because most of a lamina
# is not vein, and a low cut turns leaf texture into a mat of false branches.
VEIN_PERCENTILE = 88.0
# How far a candidate must reach, as a fraction of the leaf's half-width --
# that is, of the distance from the midrib to the margin. Measured against
# the midrib's *length* before, which is the wrong ruler: a vein runs across
# the blade, not along it, so a long thin leaf and a round one demanded
# wildly different things of the same feature.
#
# The separation is clean at 0.6. Over four noise seeds of a synthetic leaf
# with no veins at all, the longest texture streak reached 0.51 half-widths;
# on the same leaf with two real basal veins the candidates ran to 1.32, and
# on leaf 3 of gaensefuss_31 the two genuine basal veins reached 0.85 and
# 2.35. Nothing false survives 0.6 and nothing true is lost to it.
MIN_VEIN_REACH = 0.6
# Band inside the leaf margin that is not searched, as a fraction of the
# leaf's half-width. The margin is a raised rim of its own, and at 0.08 the
# rim was still being reported as a vein down the whole upper-left side of
# leaf 3. Widening past about 0.2 starts deleting the real basal veins with
# it, so this alone does not settle it -- MIN_INSERTION_ANGLE finishes the job.
MARGIN_BAND = 0.15
# Angle to the local midrib, in degrees, below which a candidate is not a
# vein departing the midrib but something running alongside it -- the margin
# rim, or the midrib's own flank where the corridor was too narrow to cover
# it. On leaf 3 the rim artefact came in at 10 degrees against 24 and 39 for
# the two real basal veins, which is a wide enough gap to cut in.
MIN_INSERTION_ANGLE = 20.0


@dataclass
class Vein:
    path: np.ndarray          # (N, 2) x/y, bbox-local, ordered from the midrib out
    station: int              # index into the midrib it departs from
    insertion_angle_deg: float  # against the midrib's local tangent
    length_px: float
    strength: float           # mean ridge response along it, 0..1

    def to_dict(self) -> dict:
        return {
            "station": int(self.station),
            "insertion_angle_deg": round(float(self.insertion_angle_deg), 2),
            "length_px": round(float(self.length_px), 2),
            "strength": round(float(self.strength), 4),
            "path": self.path.round(2).tolist(),
        }


def _branches(skeleton: np.ndarray) -> List[np.ndarray]:
    """Cut a thinned skeleton at its junctions into simple curves.

    Junctions are removed rather than resolved: a vein that forks is reported
    as the two pieces it forks into, which is the truthful reading of what
    the image showed, and joining them would require deciding which fork is
    the continuation -- a decision the pixels do not support.
    """
    neighbours = cv2.filter2D(skeleton.astype(np.uint8), -1,
                              np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8))
    simple = skeleton & (neighbours <= 2)

    count, labels = cv2.connectedComponents(simple.astype(np.uint8), connectivity=8)
    out = []
    for number in range(1, count):
        ys, xs = np.nonzero(labels == number)
        if len(xs) < 4:
            continue
        out.append(np.stack([xs, ys], axis=1).astype(float))
    return out


def _order_from(points: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Chain a branch's pixels into a path starting at the end nearest `anchor`.

    Nearest-neighbour walk rather than a sort along any axis: a vein that
    curves back on itself has two pixels at the same x, and sorting puts them
    in an order the curve never takes.
    """
    remaining = list(range(len(points)))
    start = int(np.argmin(np.linalg.norm(points - anchor, axis=1)))
    remaining.remove(start)
    path = [points[start]]
    while remaining:
        here = path[-1]
        step = min(remaining, key=lambda i: float(np.sum((points[i] - here) ** 2)))
        if np.linalg.norm(points[step] - here) > 3.0:
            break  # a gap this size is a different curve, not the next pixel
        remaining.remove(step)
        path.append(points[step])
    return np.array(path)


def find_veins(
    mask: np.ndarray,
    image: np.ndarray,
    midrib: Midrib,
    normals: Optional[np.ndarray] = None,
    percentile: float = VEIN_PERCENTILE,
    max_veins: int = 40,
) -> List[Vein]:
    """Vein candidates for one leaf, strongest first.

    `normals` is the photometric normal field for this crop, when one was
    solved. It is worth far more than the photograph here and the crease is
    recomputed from it at this function's own smoothing rather than reusing
    the midrib's -- see `VEIN_NORMAL_BLUR`.
    """
    from skimage.filters import sato
    from skimage.morphology import skeletonize

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    half_width = float(distance.max())
    if half_width <= 4 or len(midrib.path) < 3:
        return []

    from .midrib import flatten_background

    def normalise(field: np.ndarray) -> np.ndarray:
        inside = field[mask]
        top = float(np.percentile(inside, 99.5)) if inside.size else 0.0
        return np.clip(field / top, 0.0, 1.0) if top > 1e-9 else np.zeros_like(field)

    sigmas = [max(1.0, s * half_width) for s in VEIN_SCALES]
    response = normalise(sato(flatten_background(image, mask), sigmas=sigmas,
                              black_ridges=False).astype(np.float32))

    if normals is not None:
        from .photometric import ridge_from_normals

        blur = max(VEIN_NORMAL_BLUR_FLOOR, VEIN_NORMAL_BLUR * half_width)
        crease = ridge_from_normals(normals, mask, blur=blur)
        crease = normalise(sato(crease.astype(np.float32), sigmas=sigmas,
                                black_ridges=False).astype(np.float32))
        response = (1.0 - CREASE_WEIGHT) * response + CREASE_WEIGHT * crease

    # The midrib's own corridor is excluded so the threshold is set by vein
    # contrast rather than by the one structure already accounted for.
    corridor = np.zeros(mask.shape, np.uint8)
    polyline = np.round(midrib.path).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(corridor, [polyline], False, 1,
                  thickness=max(3, int(0.25 * half_width)))

    # ...and so is a band inside the margin. Flattening the backing removes
    # the step edge, but the margin is also a real raised rim, and a "vein"
    # that runs along it at a constant tiny distance from the outline is the
    # rim rather than a vein -- there is no way to tell the two apart there,
    # so neither is claimed.
    rim = max(2, int(round(MARGIN_BAND * half_width)))
    interior = cv2.erode(mask.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (2 * rim + 1, 2 * rim + 1)))
    lamina = interior.astype(bool) & (corridor == 0)
    if lamina.sum() < 64:
        return []

    cut = float(np.percentile(response[lamina], percentile))
    strong = skeletonize((response >= cut) & lamina)

    scale = np.clip(response / max(float(response[lamina].max()), 1e-9), 0, 1)
    tangents = np.gradient(midrib.path, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)

    minimum_length = MIN_VEIN_REACH * half_width
    veins: List[Vein] = []
    for branch in _branches(strong):
        distances = np.linalg.norm(branch[:, None, :] - midrib.path[None, :, :], axis=2)
        pixel, station = np.unravel_index(int(np.argmin(distances)), distances.shape)
        if distances[pixel, station] > 0.35 * half_width:
            continue  # reaches no midrib: a hair, a fold, or backing texture

        # Screened on pixel count before being ordered. `_order_from` walks
        # nearest-neighbour, which is quadratic in the branch's length, and
        # most branches a threshold produces are too short to keep -- so
        # paying that cost and then discarding the result is the wrong way
        # round. A thinned branch's pixel count is within a factor of root
        # two of its length, hence the margin.
        if len(branch) * 1.42 < minimum_length:
            continue

        path = _order_from(branch, midrib.path[station])
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        length = float(steps.sum())
        if length < minimum_length or len(path) < 4:
            continue

        direction = path[-1] - path[0]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        cosine = float(np.dot(direction / norm, tangents[station]))
        angle = float(np.degrees(np.arccos(np.clip(abs(cosine), -1.0, 1.0))))
        if angle < MIN_INSERTION_ANGLE:
            continue

        rows = np.clip(path[:, 1].astype(int), 0, scale.shape[0] - 1)
        cols = np.clip(path[:, 0].astype(int), 0, scale.shape[1] - 1)
        veins.append(Vein(path=path, station=int(station), insertion_angle_deg=angle,
                          length_px=length, strength=float(scale[rows, cols].mean())))

    veins.sort(key=lambda v: -v.strength * v.length_px)
    return veins[:max_veins]
