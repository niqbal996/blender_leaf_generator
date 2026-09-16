"""Find each leaf, then find its edge properly.

These are two different problems and they want two different tools, which is
the one structural decision in this package worth arguing for.

**Which pixels belong to which leaf** is a grouping problem. On a flat lay
the leaves are separated by backing, so a colour index answers it exactly and
in a second -- there is nothing for a learned model to contribute. It earns
its place only when leaves touch or overlap, and `--instances sam` is there
for that capture.

**Where the edge actually is** is a resolution problem, and SAM is the wrong
instrument for it at this scale. Its mask decoder emits a 256x256 logit map
that is upsampled to the image; against a 9568x6376 frame that is ~37 frame
pixels per logit pixel, so a leaf's marginal teeth and a 30-pixel-wide
petiole are below the decoder's own grid before any thresholding happens.
Whatever produced the coarse mask, the boundary is therefore re-decided here
at native resolution, from the image, by `refine_instance`.

The output of that is a matte and a sub-pixel contour rather than only a
bitmap. The contour is what a mesh builder wants -- `leaf_generator`'s
Blender path builds from a contour -- and marching squares on the matte
locates the 0.5 crossing between two pixels instead of picking one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# The same chromatic index P2 seeds SAM2 with. Chromatic normalisation is
# what makes it survive the strong centre-to-corner falloff of a light rig:
# a leaf at the edge of the frame is darker than one in the middle but not
# less green.
from pose_estimator.segmentation import excess_green

# Defaults fitted on gaensefuss_31 (2026-09-15), 30 leaves on black felt.
# Each is a rejection rule, and each rejects something that was actually
# there rather than something imagined -- see `reject_reason`.
MIN_AREA_FRACTION = 2e-5      # backing fibres and dust specks
# The backing itself, if a threshold ever inverts. Generous, because the
# thing on the other side of this rule is a legitimate capture: one leaf
# photographed on its own fills a large part of the frame, and a cap set to
# what a 30-leaf flat lay needs would silently reject it. A backing is 90% of
# a frame or more, so 0.6 still separates the two cases with room to spare.
MAX_AREA_FRACTION = 0.6
MIN_SOLIDITY = 0.45           # the root system
# Horizontal bands the frame is cut into to put accepted leaves in reading
# order; see `_report_from_labels`.
READING_BANDS = 24
# The fiducial markers, and anything else in frame that is not plant tissue.
# Set from the measured distribution on gaensefuss_31 rather than from what a
# neutral surface "should" read: the markers came out at 0.097 and 0.120 --
# not the ~0 white paper implies -- and the leaves at 0.70 to 0.96, with the
# inflorescence lowest of the plant material at 0.565. 0.35 sits in the wide
# empty band between the two groups. A floor of 0.04, chosen on the theory
# rather than the histogram, accepted both markers as leaves.
#
# This asks "is this plant tissue", not "is this a leaf". The inflorescence
# passes, and should: dropping it silently would hide something that is
# really there, and it is obvious in the diagram and in the report.
MIN_GREENNESS = 0.35


@dataclass
class Instance:
    """One accepted blob, before and after the boundary is re-decided."""

    index: int
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1 in full-resolution pixels
    mask: np.ndarray                 # bool, bbox-sized, full resolution
    alpha: np.ndarray                # float32 0..1, bbox-sized, the matte
    contour: np.ndarray              # (N, 2) float, sub-pixel, full-frame x/y
    area_px: float
    solidity: float
    greenness: float

    @property
    def offset(self) -> Tuple[int, int]:
        return self.bbox[0], self.bbox[1]

    def to_full(self, points: np.ndarray) -> np.ndarray:
        """Bbox-local x/y to full-frame x/y."""
        return np.asarray(points, dtype=float) + np.array(self.offset, dtype=float)


@dataclass
class Detection:
    """A candidate blob and the measurements a rejection rule reads."""

    label: int
    bbox: Tuple[int, int, int, int]  # in work-resolution pixels
    mask: np.ndarray                 # bool, work resolution, bbox-sized
    area_fraction: float
    solidity: float
    greenness: float
    rejected: Optional[str] = None


@dataclass
class DetectionReport:
    accepted: List[Detection] = field(default_factory=list)
    rejected: List[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        def row(d: Detection) -> dict:
            return {
                "bbox": list(d.bbox),
                "area_fraction": round(d.area_fraction, 8),
                "solidity": round(d.solidity, 4),
                "greenness": round(d.greenness, 4),
                "rejected": d.rejected,
            }

        return {
            "accepted": [row(d) for d in self.accepted],
            "rejected": [row(d) for d in self.rejected],
        }


# --------------------------------------------------------------------------
# Coarse detection
# --------------------------------------------------------------------------


def work_scale(shape: Tuple[int, int], max_side: int) -> float:
    """Factor taking a full frame down to the detection resolution.

    Detection does not need 61 megapixels -- a leaf is thousands of pixels
    across at that size and the grouping is settled long before. Everything
    that *is* resolution-sensitive happens in `refine_instance`, back at
    full resolution.
    """
    longest = max(shape[:2])
    return 1.0 if longest <= max_side else max_side / float(longest)


def greenness_map(rgb: np.ndarray) -> np.ndarray:
    """Excess green on an RGB float image, as `excess_green` wants BGR.

    Used to say *what* a blob is, never where its edge is. On a near-black
    backing the index is unreliable at exactly the level the edge lives at:
    chromatic normalisation divides by R+G+B, so where the backing is almost
    unlit the divisor approaches zero and sensor noise is amplified into
    apparent colour. Measured on the felt in gaensefuss_31, its median excess
    green is 0.12-0.18 -- not the ~0 a neutral surface should give -- while a
    leaf reaches 0.9. The blob-level median is still a clean signal, because
    averaging over thousands of pixels is what that noise cannot survive.
    """
    return excess_green(np.ascontiguousarray(rgb[:, :, ::-1]))


def foreground_index(rgb: np.ndarray) -> np.ndarray:
    """Log luminance: how far above the backing a pixel is.

    This is the index every *boundary* decision is made on, and it replaced
    excess green there for two measured reasons.

    It separates further. On gaensefuss_31 the felt sits at a luminance of
    0.0055 and lit plant tissue at 0.16 -- a ratio of 30 -- where the colour
    index has the backing at a sixth of a leaf's value.

    It does not care what colour the tissue is, which is the reason that
    matters. A petiole is paler and yellower than the blade it carries, and
    on some species frankly brown; under the colour index it fell below
    threshold and was cut off. Leaf 5's stalk was truncated by 210
    full-resolution pixels that way, and the leaf then measured as having
    almost no petiole -- a wrong number rather than a missing one.

    Log rather than linear because the quantity that matters is the ratio to
    the backing, not the difference: it makes the two modes comparable in
    width so Otsu's assumption holds, and it keeps a dim structure well
    clear of 0.5 in the matte instead of marginal.
    """
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.log10(np.maximum(luminance, 1e-6)).astype(np.float32)


def foreground_threshold(index: np.ndarray) -> float:
    """Otsu's split of the colour index.

    Otsu rather than a fixed cut because the index's absolute value depends
    on the backing: black felt sits near 0, a grey card near 0.02, and a
    fixed cut that works on one silently erodes leaves on the other. What is
    stable across backings is that the histogram is strongly bimodal, which
    is exactly the assumption Otsu makes.
    """
    finite = index[np.isfinite(index)]
    low, high = np.percentile(finite, [1.0, 99.9])
    if high - low < 1e-6:
        return float(high)
    scaled = np.clip((finite - low) / (high - low), 0, 1)
    level, _ = cv2.threshold((scaled * 255).astype(np.uint8), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(low + (level / 255.0) * (high - low))


def solidity_of(mask: np.ndarray) -> float:
    """Blob area over its convex hull's area.

    This is the rule that separates a leaf from the root system, and it is
    a shape statement rather than a size one: a leaf, lobed margin and all,
    fills most of its hull, while a root's spray of fibres encloses mostly
    air. Measured on gaensefuss_31 the leaves ran 0.63-0.95 and the
    root+stem 0.08, which is not a borderline call.
    """
    filled = mask.astype(np.uint8)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    hull = float(cv2.contourArea(cv2.convexHull(contour)))
    return area / hull if hull > 0 else 0.0


def reject_reason(
    area_fraction: float,
    solidity: float,
    greenness: float,
    min_area_fraction: float = MIN_AREA_FRACTION,
    max_area_fraction: float = MAX_AREA_FRACTION,
    min_solidity: float = MIN_SOLIDITY,
    min_greenness: float = MIN_GREENNESS,
) -> Optional[str]:
    """Why this blob is not a leaf, or None if it is.

    Returned as a string, and carried into the report, because a silently
    dropped leaf and a correctly dropped root look identical in a count.
    """
    if area_fraction < min_area_fraction:
        return "too small"
    if area_fraction > max_area_fraction:
        return "too large"
    if greenness < min_greenness:
        return "not green"
    if solidity < min_solidity:
        return "too sparse"
    return None


def detect_colour(
    flat: np.ndarray,
    max_side: int = 2000,
    open_radius: int = 1,
    **limits,
) -> Tuple[DetectionReport, float]:
    """Group the frame into blobs standing clear of the backing.

    Returns (report, scale); the blobs are in work-resolution coordinates and
    `scale` converts them back to the frame's.

    The opening is one pixel, not the three a noisy threshold would want. A
    petiole is only a handful of pixels across at the detection resolution,
    and an ellipse of radius 2 erodes it into segments -- which then fail the
    area test individually and disappear, taking the leaf's stalk with them.
    A radius of one still clears the backing's lint, because the luminance
    index leaves so little to clear.
    """
    scale = work_scale(flat.shape, max_side)
    work = (cv2.resize(flat, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0 else flat)

    index = foreground_index(work)
    binary = (index > foreground_threshold(index)).astype(np.uint8)
    if open_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * open_radius + 1, 2 * open_radius + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return _report_from_labels(binary, greenness_map(work), **limits), scale


def detect_sam(
    flat: np.ndarray,
    checkpoint: Path,
    max_side: int = 2000,
    device: str = "cuda",
    points_per_side: int = 32,
    **limits,
) -> Tuple[DetectionReport, float]:
    """Group the frame with SAM2's automatic mask generator instead.

    For captures the colour route cannot do: leaves touching, leaves
    overlapping, a backing that is not a colour apart from the subject. The
    masks it returns are still only the *grouping* -- every one of them is
    handed to `refine_instance` like any other, because at this working
    resolution its boundary is no better than the colour route's.
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    from pose_estimator.segmentation import _resolve_model_cfg

    scale = work_scale(flat.shape, max_side)
    work = (cv2.resize(flat, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0 else flat)
    index = greenness_map(work)

    from .raw import tone_map

    model = build_sam2(_resolve_model_cfg(checkpoint), str(checkpoint), device=device)
    generator = SAM2AutomaticMaskGenerator(model, points_per_side=points_per_side)
    proposals = generator.generate(tone_map(work))

    # Painted into one label image rather than kept as overlapping proposals:
    # SAM2 happily returns a leaf, its blade without the petiole, and the pair
    # of leaves next to it, and a later stage has no way to know which of the
    # three is the object. Largest first means a big proposal cannot overwrite
    # the more specific one drawn after it.
    labels = np.zeros(work.shape[:2], np.int32)
    for number, proposal in enumerate(
            sorted(proposals, key=lambda p: -p["area"]), start=1):
        labels[proposal["segmentation"].astype(bool)] = number

    return _report_from_labels(labels, index, prelabelled=True, **limits), scale


def _report_from_labels(
    source: np.ndarray, index: np.ndarray, prelabelled: bool = False, **limits,
) -> DetectionReport:
    """Measure and judge every blob in a binary or label image."""
    if prelabelled:
        labels = source
        count = int(labels.max()) + 1
    else:
        count, labels = cv2.connectedComponents(source.astype(np.uint8), connectivity=8)

    frame_area = float(labels.shape[0] * labels.shape[1])
    report = DetectionReport()
    for number in range(1, count):
        blob = labels == number
        area = float(blob.sum())
        if area <= 0:
            continue
        ys, xs = np.nonzero(blob)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        local = blob[y0:y1, x0:x1]

        detection = Detection(
            label=number,
            bbox=(x0, y0, x1, y1),
            mask=local,
            area_fraction=area / frame_area,
            solidity=solidity_of(local),
            greenness=float(np.median(index[blob])),
        )
        detection.rejected = reject_reason(
            detection.area_fraction, detection.solidity, detection.greenness, **limits)
        (report.rejected if detection.rejected else report.accepted).append(detection)

    # Reading order: down the rows, then across. Leaves on a flat lay are put
    # out in rows, so this makes the printed ids match the way a person
    # scanning the photograph would number them. The frame is cut into bands
    # and blobs are sorted by band first; the band count only has to be
    # coarser than the leaves are tall and finer than the rows are apart,
    # which anything from about 8 to 40 satisfies for a two-row lay.
    band = max(1, labels.shape[0] // READING_BANDS)
    report.accepted.sort(key=lambda d: (d.bbox[1] // band, d.bbox[0]))
    return report


# --------------------------------------------------------------------------
# Full-resolution boundary
# --------------------------------------------------------------------------


def guided_matte(
    crop: np.ndarray, coarse: np.ndarray, radius: int = 4, eps: float = 1e-4,
) -> np.ndarray:
    """A soft 0..1 matte for the leaf, at the crop's own resolution.

    Three steps, and the middle one is the one doing the work:

    1. Score every pixel by the same colour index, and turn it into a rough
       alpha by asking where it falls between the blob's own interior level
       and the surrounding backing's. Both levels are measured from *this*
       crop, so a leaf in a dim corner is judged against its own
       neighbourhood rather than the frame's average.
    2. Guided-filter that alpha with the photograph as the guide. This is
       what makes the edge sharp: the filter's output is a locally linear
       function of the guide, so the alpha transition is forced to coincide
       with the image's own intensity transition -- the real leaf margin --
       instead of with the upsampled coarse mask's idea of it.
    3. Re-centre on 0.5 so a plain threshold and the marching-squares contour
       agree about where the boundary is.

    The radius is deliberately small. The filter cannot preserve a structure
    thinner than its own window, and the thinnest thing that must survive is
    a petiole -- about 16 pixels across at full resolution on this rig -- so a
    radius of 8 was already half of it and visibly thinned the stalk.
    """
    index = foreground_index(crop)
    inner = cv2.erode(coarse.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    outer = ~cv2.dilate(coarse.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    if not inner.any() or not outer.any():
        inner, outer = coarse, ~coarse
    if not inner.any() or not outer.any():
        return coarse.astype(np.float32)

    leaf_level = float(np.median(index[inner]))
    backing_level = float(np.median(index[outer]))
    span = leaf_level - backing_level
    if span <= 1e-6:
        return coarse.astype(np.float32)

    rough = np.clip((index - backing_level) / span, 0.0, 1.0).astype(np.float32)

    guide = np.ascontiguousarray(crop[:, :, ::-1])
    guide = (guide / max(float(guide.max()), 1e-6)).astype(np.float32)
    try:
        alpha = cv2.ximgproc.guidedFilter(guide, rough, radius, eps)
    except (AttributeError, cv2.error):
        # opencv-contrib absent: a joint bilateral filter is the same idea --
        # smooth alpha only across pixels the photograph agrees are alike --
        # and ships in the base build.
        alpha = cv2.bilateralFilter(rough, d=2 * radius + 1,
                                    sigmaColor=0.1, sigmaSpace=float(radius))
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def largest_filled(mask: np.ndarray) -> np.ndarray:
    """Largest connected component, holes closed.

    Holes are filled because a leaf is a simple sheet: an insect bite is
    real, but so is a specular flare in the middle of the lamina, and only
    the second is common enough at this scale to matter. The alternative --
    keeping every interior hole -- put a ragged ring of them down the midrib
    of every leaf, which is where the highlight sits.
    """
    solid = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if count > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        solid = (labels == biggest).astype(np.uint8)

    # Flooded on a copy with a one-pixel border of guaranteed background, so
    # the seed is background whatever the mask does. Seeding at (0, 0) of the
    # crop itself is the usual idiom and it is wrong here: a leaf whose crop
    # was clipped against the frame edge reaches that corner, the flood then
    # starts *inside* the leaf and spreads nowhere, and every background
    # pixel is left looking like an enclosed hole -- so the "hole fill"
    # returns the entire crop as leaf. Observed on leaf 15 of gaensefuss_31,
    # where an 8,999-pixel matte became a 35,083-pixel mask and took the
    # midrib out onto the backing with it.
    bordered = np.zeros((solid.shape[0] + 2, solid.shape[1] + 2), np.uint8)
    bordered[1:-1, 1:-1] = solid
    scratch = np.zeros((bordered.shape[0] + 2, bordered.shape[1] + 2), np.uint8)
    cv2.floodFill(bordered, scratch, (0, 0), 1)
    holes = bordered[1:-1, 1:-1] == 0
    return (solid.astype(bool) | holes)


def subpixel_contour(alpha: np.ndarray, level: float = 0.5) -> np.ndarray:
    """The matte's 0.5 crossing as an (N, 2) x/y polygon.

    Marching squares interpolates between the two pixels the crossing lies
    between, so the contour is not quantised to the pixel grid -- which is
    the point of having refined at full resolution in the first place.
    """
    from skimage.measure import find_contours

    contours = find_contours(alpha, level)
    if not contours:
        return np.empty((0, 2), float)
    longest = max(contours, key=len)
    return np.stack([longest[:, 1], longest[:, 0]], axis=1)  # (row, col) -> (x, y)


def refine_instance(
    flat: np.ndarray,
    detection: Detection,
    scale: float,
    index: int,
    margin: int = 48,
    radius: int = 4,
    eps: float = 1e-4,
) -> Optional[Instance]:
    """Re-decide one blob's boundary at the frame's own resolution."""
    height, width = flat.shape[:2]
    x0, y0, x1, y1 = detection.bbox
    full = [int(round(v / scale)) for v in (x0, y0, x1, y1)]
    x0 = max(0, full[0] - margin)
    y0 = max(0, full[1] - margin)
    x1 = min(width, full[2] + margin)
    y1 = min(height, full[3] + margin)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    crop = flat[y0:y1, x0:x1]

    # The coarse mask is placed into the crop, not resized to fill it. It
    # covers the detection's own bounding box; the crop is that box plus a
    # margin on every side. Stretching one onto the other inflates the mask
    # by the margin -- about 15% for a big leaf, where the matte absorbs it,
    # but 2.4x for a 69-pixel-wide one, where it does not: the level
    # estimates are then read off the wrong pixels and the band constrains
    # nothing. One affine does the scaling and the offset together, and
    # clips to the crop for free.
    placement = np.array([[1.0 / scale, 0.0, detection.bbox[0] / scale - x0],
                          [0.0, 1.0 / scale, detection.bbox[1] / scale - y0]])
    coarse = cv2.warpAffine(detection.mask.astype(np.uint8), placement,
                            (x1 - x0, y1 - y0),
                            flags=cv2.INTER_NEAREST).astype(bool)
    if not coarse.any():
        return None

    alpha = guided_matte(crop, coarse, radius=radius, eps=eps)

    # Refinement may move the boundary; it may not annex new territory. The
    # backing carries lint fibres that are genuinely bright -- the luminance
    # index is right about them -- and at full resolution a fibre touching a
    # leaf is connected to it, so the largest component swallows it. That
    # cost more than a ragged outline: a 200-pixel fibre is then the furthest
    # point of the shape, so `geodesic_endpoints` put the leaf's "tip" out on
    # the backing and the midrib followed it there.
    #
    # The coarse mask does not have this problem, because at detection
    # resolution a two-pixel fibre is below one pixel and averages away. So
    # the coarse mask is allowed to constrain *where* the boundary may be,
    # while the full-resolution matte decides where within that band it is.
    # The band is three detection pixels wide, which is the coarse mask's own
    # quantisation -- enough for the edge to be anywhere the coarse pass could
    # not have resolved, and far short of a fibre's length.
    band = max(8, int(round(3.0 / max(scale, 1e-6))))
    reachable = cv2.dilate(coarse.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (2 * band + 1, 2 * band + 1)))
    alpha = alpha * reachable

    mask = largest_filled(alpha > 0.5)
    if mask.sum() < 64:
        return None
    # The matte is trimmed to the component that survived, so the contour and
    # the bitmap cannot disagree about which blob they describe.
    alpha = alpha * mask

    contour = subpixel_contour(alpha)
    if len(contour) == 0:
        return None

    return Instance(
        index=index,
        bbox=(x0, y0, x1, y1),
        mask=mask,
        alpha=alpha,
        contour=contour + np.array([x0, y0], dtype=float),
        area_px=float(mask.sum()),
        solidity=solidity_of(mask),
        greenness=detection.greenness,
    )
