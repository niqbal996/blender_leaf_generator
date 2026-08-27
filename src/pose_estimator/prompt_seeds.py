"""Reusable "this is the plant, that is the holder" seeds for P2.

P2 tracks two objects through the video and has to be told which is which.
Two ways existed and neither carries to a new capture:

- a colour rule that encodes *red and yellow insulated pliers* specifically,
  so a pot, a clamp or differently-coloured pliers all defeat it;
- `--plant-point X,Y`, which is a pixel coordinate in one video and means
  nothing in the next.

This is the third way, and it is the same trick P4c already uses for organ
classes: click the plant and the holder once with `pose-pick-prompts`, store
the DINO **feature vectors** at those points, and on a new video find the patch that most
resembles each. Vectors describe what a plier looks like rather than where
this video's plier happens to sit, so one bank serves a whole batch shot at
different poses.

The prompts only have to land somewhere inside the right object -- SAM2 finds
the real boundary itself -- so being approximately right is enough, which is
exactly what feature matching is good at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


def build_prompt_bank(backbone, bgr: np.ndarray,
                      clicks: Sequence[Tuple[str, int, int]]) -> Tuple[np.ndarray, List[str]]:
    """Feature vector per clicked point on a *full* frame.

    Full frame, not cropped to the plant: at P2 time no plant mask exists yet,
    which is the whole reason P2 needs telling which object is which.
    """
    features = backbone.features(bgr)
    grid = backbone.grid
    height, width = bgr.shape[:2]

    vectors, labels = [], []
    for label, x, y in clicks:
        gx = min(int(x / width * grid), grid - 1)
        gy = min(int(y / height * grid), grid - 1)
        vectors.append(features[gy * grid + gx])
        labels.append(label)
    return np.stack(vectors, axis=1), labels


def _border_background(features: np.ndarray, similarity: np.ndarray,
                       border: np.ndarray, sample: int = 256) -> Optional[np.ndarray]:
    """Similarity of every patch to the most similar background patch."""
    index = np.nonzero(border)[0]
    if len(index) < 4:
        return None
    # Drop the quartile that most resembles a labelled class: those are the
    # object itself running off the edge of the frame, not background.
    looks_labelled = similarity[index].max(axis=1)
    index = index[looks_labelled <= np.percentile(looks_labelled, 75)]
    if len(index) < 4:
        return None
    if len(index) > sample:
        index = index[np.linspace(0, len(index) - 1, sample).astype(int)]
    return (features @ features[index].T).max(axis=1)


def _spread_picks(score: np.ndarray, allowed: np.ndarray, grid: int,
                  count: int, radius: int) -> List[Tuple[int, int, float]]:
    """Greedy top-`count` patches, suppressing a disc around each pick.

    Without the suppression every pick lands in the same leaf -- the patches
    around the best one are its neighbours and score almost as well -- which
    is no better than a single prompt.
    """
    field = np.where(allowed, score, -np.inf).reshape(grid, grid).astype(float).copy()
    yy, xx = np.mgrid[0:grid, 0:grid]
    picks: List[Tuple[int, int, float]] = []
    for _ in range(max(count, 1)):
        flat = int(np.argmax(field))
        value = float(field.ravel()[flat])
        if not np.isfinite(value):
            break
        gy, gx = divmod(flat, grid)
        picks.append((gx, gy, value))
        field[(xx - gx) ** 2 + (yy - gy) ** 2 <= radius ** 2] = -np.inf
    return picks


def _box_allowed(grid: int, width: int, height: int,
                 centre: Tuple[int, int], side: float) -> np.ndarray:
    """Patches inside a square of `side` pixels centred on `centre`."""
    cx, cy = centre
    x = (np.arange(grid) + 0.5) / grid * width
    y = (np.arange(grid) + 0.5) / grid * height
    inside_x = np.abs(x - cx) <= side / 2
    inside_y = np.abs(y - cy) <= side / 2
    return (inside_y[:, None] & inside_x[None, :]).ravel()


def locate_prompts(backbone, bgr: np.ndarray, vectors: np.ndarray,
                   labels: Sequence[str],
                   exclude_border: float = 0.02,
                   count: int = 1,
                   anchor: str = "plant",
                   crop_side: Optional[float] = None) -> Dict[str, List[Tuple[int, int]]]:
    """Best-matching pixels in this frame for each labelled class.

    Returns `{label: [(x, y), ...]}` -- a list, because one prompt is not
    enough on a small or distant plant. Measured on thistle2 pass 0: a single
    correct prompt landed on one leaf and SAM2 tracked only that leaf, 12,255
    px against 41,040 px for the pass where the plant filled more of the
    frame. Extra prompts on different leaves are how SAM2 is told the object
    is the whole rosette.

    Patches are ranked by **margin** -- similarity to this class minus the
    best similarity to any other class -- not by raw similarity. The extra
    prompts are the ones at risk of drifting onto the tool or the background,
    and a relative comparison is what keeps them honest. Only the anchor's
    first prompt is returned regardless of margin, so there is always a plant
    point to seed with; every additional prompt must actually out-score the
    other classes. Note that background is not a class unless the bank has
    one, so clicking a few background examples makes this test stronger.

    `anchor` is searched over the whole frame; every other class is searched
    only within a `crop_side`-wide square around the anchor, defaulting to the
    frame height. That is not cosmetic. P2 tracks inside a crop sized to the
    plant, and a prompt outside it is dropped -- which is what happened to the
    holder on thistle2, because the most plier-like patch in the frame is the
    big colourful handle, metres of pixels away from a plant-sized crop. The
    picker tells a human to click the jaws for exactly this reason; this is
    the same instruction expressed as a search region.

    A margin at the frame edge is skipped. A patch there is half background,
    and a prompt on the edge of an object is what makes SAM2 latch onto its
    neighbour.
    """
    features = backbone.features(bgr)          # (grid*grid, dim), L2-normalised
    grid = backbone.grid
    height, width = bgr.shape[:2]
    similarity = features @ vectors            # (patches, seeds)

    classes = list(dict.fromkeys(labels))
    best_of: Dict[str, np.ndarray] = {}
    for name in classes:
        columns = [i for i, other in enumerate(labels) if other == name]
        best_of[name] = similarity[:, columns].max(axis=1)

    edge = max(int(grid * exclude_border), 1)
    keep = np.zeros((grid, grid), bool)
    keep[edge:grid - edge, edge:grid - edge] = True
    allowed_anywhere = keep.ravel()

    # Background, taken from the frame's border ring rather than clicked.
    #
    # Without it the margin only ever compares plant against holder, so a
    # patch of bare table out-scores nothing and extra prompts wander onto it.
    # Measured on thistle2: prompts 2 and 3 landed on the checkerboard, SAM2
    # was told the board was the object, and the returned mask was the crop
    # minus the plant -- inverted.
    #
    # The border is background by construction: the subject is never at the
    # frame edge, which is the same assumption that already excludes those
    # patches from being prompts. The top quartile by labelled-class
    # similarity is dropped first, so a leaf that does touch the edge cannot
    # smuggle itself in as its own competitor -- a rank rule, with no
    # magnitude to tune.
    background = _border_background(features, similarity, ~allowed_anywhere)

    to_pixels = lambda gx, gy: (int((gx + 0.5) / grid * width),
                                int((gy + 0.5) / grid * height))

    order = ([anchor] if anchor in best_of else []) + [c for c in classes if c != anchor]
    out: Dict[str, List[Tuple[int, int]]] = {}
    anchor_point: Optional[Tuple[int, int]] = None

    for name in order:
        rivals = [best_of[o] for o in classes if o != name]
        if background is not None:
            rivals.append(background)
        margin = best_of[name] - (np.max(np.stack(rivals), axis=0) if rivals else 0.0)

        allowed = allowed_anywhere.copy()
        if name != anchor and anchor_point is not None:
            allowed &= _box_allowed(grid, width, height, anchor_point,
                                    crop_side if crop_side is not None else height)
            if not allowed.any():          # nothing of this class near the plant
                out[name] = []
                continue

        # Suppression radius from the object's own size, not the frame's.
        # A fraction of the frame is meaningless here: the plant covered 3%
        # of thistle2 pass 0 and 20% of pass 1, so one constant either
        # clusters every prompt in a single leaf or pushes them off the
        # plant. Partition the class's footprint into `count` square cells
        # and suppress one cell side -- which needs no constant at all.
        support = int((allowed & (margin > 0)).sum())
        radius = max(int(np.sqrt(max(support, 1) / max(count, 1))), 1)
        picks = _spread_picks(margin, allowed, grid, count, radius)
        points = [to_pixels(gx, gy) for rank, (gx, gy, value) in enumerate(picks)
                  if value > 0 or (rank == 0 and name == anchor)]
        out[name] = points
        if name == anchor and points:
            anchor_point = points[0]
    return out


def best_patch(backbone, bgr: np.ndarray, vectors: np.ndarray, labels: Sequence[str],
               name: str,
               within: Optional[Tuple[int, int, int, int]] = None,
               exclude_border: float = 0.02) -> Optional[Tuple[int, int, float]]:
    """The single best patch for class `name` by margin, or None.

    Same rules as `locate_prompts` -- margin against every other bank class
    plus the frame-border background, frame edge excluded -- reduced to one
    class and one answer. P2 uses it to re-acquire the root object when SAM2
    loses it mid-sequence: candidate frames are scanned and the best positive
    margin wins, which is a rank rule with nothing to tune. Returns None when
    no patch out-scores the rivals, which is the honest answer on a frame
    where the class is occluded.

    `within` restricts the search to a full-frame pixel box -- the tracking
    crop, because a conditioning point outside the crop cannot be handed to
    SAM2 at all.
    """
    features = backbone.features(bgr)
    grid = backbone.grid
    height, width = bgr.shape[:2]
    similarity = features @ vectors

    classes = list(dict.fromkeys(labels))
    if name not in classes:
        return None
    best_of: Dict[str, np.ndarray] = {}
    for cls in classes:
        columns = [i for i, other in enumerate(labels) if other == cls]
        best_of[cls] = similarity[:, columns].max(axis=1)

    edge = max(int(grid * exclude_border), 1)
    keep = np.zeros((grid, grid), bool)
    keep[edge:grid - edge, edge:grid - edge] = True
    allowed = keep.ravel()
    background = _border_background(features, similarity, ~allowed)

    rivals = [best_of[o] for o in classes if o != name]
    if background is not None:
        rivals.append(background)
    margin = best_of[name] - (np.max(np.stack(rivals), axis=0) if rivals else 0.0)

    if within is not None:
        x0, y0, x1, y1 = within
        cx = (np.arange(grid) + 0.5) / grid * width
        cy = (np.arange(grid) + 0.5) / grid * height
        inside = ((cy[:, None] >= y0) & (cy[:, None] < y1)
                  & (cx[None, :] >= x0) & (cx[None, :] < x1))
        allowed = allowed & inside.ravel()
    if not allowed.any():
        return None

    field = np.where(allowed, margin, -np.inf)
    flat = int(np.argmax(field))
    value = float(field[flat])
    if not np.isfinite(value) or value <= 0:
        return None
    gy, gx = divmod(flat, grid)
    return (int((gx + 0.5) / grid * width), int((gy + 0.5) / grid * height), value)


def save_prompt_bank(path: Union[str, Path], vectors: np.ndarray, labels: Sequence[str],
                     model: str, size: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vectors=vectors, labels=np.array(list(labels)), model=model, size=size)
    return path


def load_prompt_bank(path: Union[str, Path], model: str) -> Tuple[np.ndarray, List[str]]:
    """Vectors and labels, refusing a bank built by a different model.

    Matching width is not enough: dinov2-base and dinov3-vitb16 both emit
    768-dim vectors, so a mismatched bank passes a shape check while the
    numbers mean nothing in the other model's space.
    """
    bank = np.load(Path(path), allow_pickle=True)
    stored = str(bank["model"]) if "model" in bank else ""
    if stored and stored != model:
        raise SystemExit(
            f"prompt bank {path} was built with {stored} but this run uses {model}. "
            f"Their feature spaces are unrelated. Re-run with --dino-model {stored}.")
    return bank["vectors"], [str(x) for x in bank["labels"]]
