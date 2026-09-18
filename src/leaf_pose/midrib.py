"""The midrib: where it runs, and how wide the blade is along it.

The shape of the mask and the content of the photograph each know something
the other does not, and the midrib is found by making them agree.

*The mask* knows the topology. Its medial axis runs down the middle of the
blade and out along the petiole, and the two ends of the longest path through
the leaf's interior are the petiole's cut end and the leaf's tip -- not
because anything was assumed about which is which, but because those are the
two points of the blade furthest apart *through* it. Measuring the distance
through the shape rather than across it is what keeps this right for a leaf
that curves: the straight line between the ends of a bent leaf leaves the
leaf, and a principal axis fitted to it is a chord rather than an axis.

*The photograph* knows where the rib actually is. The midrib is a raised
ridge, so it is brighter than the lamina on either side and it catches the
surface reflection the crossed-polariser image threw away. A ridge filter
responds to precisely that -- a bright line of a given width, at any
orientation -- and where it responds strongly, the mask's medial axis is
being told to move.

So: endpoints from the shape, then a minimum-cost path between them through
evidence that prefers both the middle of the blade and the visible ridge. For
an unlobed leaf the two agree and the ridge term changes nothing. For a lobed
one the medial axis alone branches into every lobe, and the ridge term is
what keeps the path on the rib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from pose_estimator.leaf import resample_by_arclength, smooth_polyline

# Cost charged to a pixel outside the leaf. Large rather than infinite: an
# infinite cost makes the whole array non-finite for the router, while a
# large one simply means no path ever takes that step.
OUTSIDE_COST = 1e6
# The largest sigma the ridge filter is ever run at, in pixels of whatever
# image it is handed. Above this the crop is downsampled and the sigmas scaled
# to match: `sato` costs pixels x sigma, and a high-resolution flat-lay drives
# both up at once. 32 keeps the filter kernel small while staying far above
# the few-pixel scale where resampling would blur the rib itself.
MAX_FILTER_SIGMA = 32.0


@dataclass
class Midrib:
    """A midrib polyline and what was measured along it.

    Coordinates are bbox-local x/y at full resolution; `Instance.to_full`
    converts them to frame coordinates.
    """

    path: np.ndarray        # (N, 2) arclength-sampled, x/y
    width: np.ndarray       # (N,) blade width in pixels at each station
    arclength: float        # pixels, along the curve
    ridge_support: float    # 0..1, how much of the path the image agrees with


def flatten_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace the backing with the leaf's own level, before any filtering.

    The strongest edge in a leaf crop by a wide margin is the leaf's own
    outline -- lit tissue against near-black felt, a step of thirty to one --
    and a ridge filter responds to it accordingly. That costs twice. The
    response along the outline sets the scale the rest of the crop is
    normalised against, so real interior structure is compressed into the
    bottom of the range; and because the filter has width, the response leaks
    a few pixels inward, where it is indistinguishable from a vein running
    along the margin. Left in, it is what `find_veins` returns: on leaf 3 of
    gaensefuss_31 all three "veins" found were the two margins.

    Filling the backing with the interior median removes the step and touches
    nothing inside the leaf.
    """
    grey = image.mean(axis=2) if image.ndim == 3 else image
    grey = grey.astype(np.float32)
    if not mask.any():
        return grey
    return np.where(mask, grey, float(np.median(grey[mask]))).astype(np.float32)


def ridge_response(
    image: np.ndarray, distance: np.ndarray, mask: np.ndarray,
    scales: Sequence[float] = (0.15, 0.25, 0.40),
) -> np.ndarray:
    """Where a bright ridge runs, normalised to 0..1 inside the mask.

    The filter scales are fractions of the leaf's own maximum half-width
    rather than pixel counts, because a capture holds leaves from 60 to 1400
    pixels across and one fixed sigma cannot serve both. A rib is a small
    fraction of the blade's width at every size, so the fraction transfers
    where the pixel count does not.
    """
    from skimage.filters import sato

    half_width = float(distance.max())
    if half_width <= 2:
        return np.zeros(image.shape[:2], np.float32)

    grey = flatten_background(image, mask)
    sigmas = [max(1.0, s * half_width) for s in scales]

    # Run the filter at a scale where the sigmas are small, not at whatever
    # resolution the leaf happened to be photographed at.
    #
    # `sato` costs roughly pixels x sigma, and both grow with the capture:
    # sigma is a fraction of the leaf's half-width, so a scan at 35 px/mm
    # asks for sigma ~360 on a 8 MP crop, which measured ~5 minutes per call
    # and is called twice per leaf. On a 61 MP flat-lay of seven leaves that
    # is over an hour in a stage that looks like it has hung.
    #
    # Downsampling is not an approximation of the answer, it is the same
    # answer computed sensibly: a ridge at sigma 360 is by construction a
    # low-frequency feature, and filtering at sigma/k on a k-times smaller
    # image is the same scale-space location. Only the upsampled response's
    # edges are softer, and this response is used as a routing cost and a
    # mean support value -- neither is sub-pixel.
    factor = max(1, int(np.ceil(max(sigmas) / MAX_FILTER_SIGMA)))
    if factor > 1:
        height, width = grey.shape[:2]
        small = cv2.resize(grey, (max(width // factor, 8), max(height // factor, 8)),
                           interpolation=cv2.INTER_AREA)
        small_sigmas = [max(1.0, s / factor) for s in sigmas]
        small_response = sato(small, sigmas=small_sigmas, black_ridges=False)
        response = cv2.resize(small_response.astype(np.float32), (width, height),
                              interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        response = sato(grey, sigmas=sigmas, black_ridges=False).astype(np.float32)

    # Normalised against its own distribution *inside the leaf*: the backing
    # carries ridge-like fibres, and including them sets the scale by
    # something that is not the subject.
    inside = response[mask]
    if inside.size == 0:
        return np.zeros_like(response)
    low, high = np.percentile(inside, [50.0, 99.0])
    if high - low < 1e-9:
        return np.zeros_like(response)
    return np.clip((response - low) / (high - low), 0.0, 1.0) * mask


def geodesic_endpoints(mask: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """The two points of the mask furthest apart through its own interior.

    Two sweeps: from an arbitrary seed to the furthest point, then from there
    to the furthest point again. The second sweep starts from a point already
    known to be extremal, which is what makes the pair it returns the
    diameter of the shape rather than an accident of where the first seed
    was. Returned as (row, col), the order the router works in.
    """
    from skimage.graph import MCP_Geometric

    inside = mask.astype(bool)
    cost = np.where(inside, 1.0, OUTSIDE_COST)

    def furthest(start: Tuple[int, int]) -> Tuple[int, int]:
        distances = MCP_Geometric(cost).find_costs([start])[0]
        reachable = np.where(inside, distances, -np.inf)
        best = float(reachable.max())

        # Every point within half a percent of the maximum is a tie, and
        # there are usually many: the far end of a blunt leaf is an arc, not
        # a point, and the router's 8-connected metric orders points along it
        # by how diagonal the path to them was rather than by how far away
        # they are. Ties are broken on straight-line distance, which is the
        # question actually being asked.
        candidates = np.argwhere(reachable >= best * 0.995)
        spread = candidates - np.array(start)
        return tuple(candidates[int(np.argmax((spread ** 2).sum(axis=1)))])

    rows, cols = np.nonzero(inside)
    seed = (int(rows[0]), int(cols[0]))
    first = furthest(seed)
    return first, furthest(first)


def route(cost: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> np.ndarray:
    """Minimum-cost 8-connected path, as an (N, 2) array of x/y."""
    from skimage.graph import route_through_array

    indices, _ = route_through_array(cost, start, end, fully_connected=True,
                                     geometric=True)
    path = np.asarray(indices, dtype=float)
    return np.stack([path[:, 1], path[:, 0]], axis=1)  # (row, col) -> (x, y)


def midrib_cost(
    distance: np.ndarray, ridge: Optional[np.ndarray], mask: np.ndarray,
    ridge_weight: float = 0.5,
) -> np.ndarray:
    """Per-pixel cost for the path: cheap down the middle and along the rib.

    Both terms are reciprocals of a 0..1 preference, so a pixel the evidence
    likes costs a little over 1 and one it does not costs a lot. Centrality
    is normalised by the leaf's own half-width so that a narrow leaf's
    centre is as attractive as a broad one's, and so that the weighting
    between the two terms means the same thing for both.
    """
    half_width = float(distance.max())
    centrality = distance / half_width if half_width > 0 else np.zeros_like(distance)
    preference = centrality
    if ridge is not None:
        preference = (1.0 - ridge_weight) * centrality + ridge_weight * ridge

    cost = 1.0 / (preference + 0.05)
    return np.where(mask, cost, OUTSIDE_COST).astype(np.float64)


def width_along(path: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Blade width in pixels at each station of the path.

    Twice the distance transform, which is the radius of the largest disc
    that fits inside the leaf centred there. On the midrib that disc is
    bounded by the two margins, so twice it is the width across the blade --
    and unlike measuring along a fixed perpendicular it needs no assumption
    that the two margins are opposite each other, which a lobed leaf breaks.
    """
    rows = np.clip(np.round(path[:, 1]).astype(int), 0, distance.shape[0] - 1)
    cols = np.clip(np.round(path[:, 0]).astype(int), 0, distance.shape[1] - 1)
    return 2.0 * distance[rows, cols]


def fit_midrib(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    specular: Optional[np.ndarray] = None,
    normal_ridge: Optional[np.ndarray] = None,
    num_samples: int = 48,
    ridge_weight: float = 0.5,
    smooth_iterations: int = 40,
) -> Optional[Midrib]:
    """Fit one leaf's midrib. Every array passed in shares one shape.

    Three optional kinds of evidence, averaged into one map rather than
    ranked, because each fails on a different leaf and none fails on all of
    them:

    `image`
        the flat photograph. Always available, weakest: it sees the rib only
        where the rib is a different colour from the lamina.
    `specular`
        the parallel-minus-crossed residual, when the capture was polarised.
        A raised rib reflects where the flat lamina beside it does not, with
        the leaf's own colour variation subtracted out. Blank for a leaf
        lying with its rib turned away from every LED.
    `normal_ridge`
        the crease in the photometric normals, when those were computed.
        The strongest of the three and the only one that is a statement
        about shape rather than about brightness -- see `photometric.py`.
        Already a 0..1 ridge measure, so unlike the other two it is used as
        it is rather than filtered again.
    """
    from skimage.morphology import medial_axis

    if mask.sum() < 64:
        return None

    _, distance = medial_axis(mask, return_distance=True)
    if float(distance.max()) <= 1:
        return None

    evidence = []
    if image is not None:
        evidence.append(ridge_response(image, distance, mask))
    if specular is not None:
        evidence.append(ridge_response(specular, distance, mask))
    if normal_ridge is not None:
        evidence.append(np.clip(normal_ridge, 0.0, 1.0).astype(np.float32) * mask)
    ridge = np.mean(evidence, axis=0) if evidence else None

    start, end = geodesic_endpoints(mask)
    if start == end:
        return None

    raw = route(midrib_cost(distance, ridge, mask, ridge_weight), start, end)
    if len(raw) < 4:
        return None

    # Smooth the dense 8-connected chain before decimating, not after. The
    # chain is a staircase -- every step is one pixel, so its direction only
    # ever takes eight values -- and decimating first samples that staircase
    # rather than the curve underneath it.
    smoothed = smooth_polyline(raw, iterations=smooth_iterations, strength=0.5)
    path, arclength = resample_by_arclength(smoothed, num_samples)
    if arclength <= 0:
        return None

    support = 0.0
    if ridge is not None:
        rows = np.clip(np.round(path[:, 1]).astype(int), 0, ridge.shape[0] - 1)
        cols = np.clip(np.round(path[:, 0]).astype(int), 0, ridge.shape[1] - 1)
        support = float(np.mean(ridge[rows, cols]))

    return Midrib(
        path=path,
        width=width_along(path, distance),
        arclength=float(arclength),
        ridge_support=support,
    )
