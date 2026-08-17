"""Find leaf tips in the photographs and vote them onto the 3D cloud.

Two leaves lying on top of each other merge into one sheet in the carved
surface, and no amount of graph work on that sheet recovers them -- the
information is not in it. In a photograph they are still two objects, with an
occlusion edge and a shading step between them, and SAM2 separates them
readily. So tips are found where that evidence exists and carried into 3D,
the same trade P4c already makes for organ class.

The method, per frame:

1. SAM2 proposes object masks. Keep the ones lying inside the plant and made
   mostly of leaf-labelled pixels, judged against the P4c class map that is
   already on disk -- so this reuses the semantic work rather than repeating
   it, and never has to decide "is this a leaf" itself.
2. A leaf's tip in that view is the pixel of its mask furthest from the stem.
   The stem is taken from the same class map, so no 3D structure is needed
   and this can run before P5 rather than after it.
3. The z-buffered index map says which 3D point produced that pixel, so the
   tip lands on the cloud exactly, with no nearest-neighbour guesswork.

Then across all frames: a real tip is seen from many directions and its votes
pile up in one place, while a mask boundary that happened to look like a tip
in one view is contradicted by the rest. Clustering the votes and keeping the
clusters with real support is what separates them.

What this does *not* need is for the two leaves to be separable in 3D, which
is the entire point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


@dataclass
class TipVote:
    """One frame's opinion about where a leaf ends."""

    frame: str
    point_index: int          # into the 3D cloud
    pixel: Tuple[int, int]
    mask_area: int
    distance_from_stem_px: float
    facing: float = 1.0       # how square-on this leaf was to this camera, 0..1


@dataclass
class TipCluster:
    position: np.ndarray
    point_index: int
    votes: int
    weight: float = 0.0         # votes summed by how face-on each view was
    visible: int = 0            # views that rendered this patch of surface at all
    frames: List[str] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        """Votes as a share of the views that could have voted.

        The number that matters for a leaf seen from only a few directions. A
        blade visible in six views and voted in five of them is as certain as
        one visible in sixty and voted in fifty, but the raw counts differ by
        an order of magnitude -- so ranking on raw votes buries exactly the
        leaves that are hardest to see and most worth recovering.
        """
        return self.votes / max(self.visible, 1)

    def to_dict(self) -> dict:
        return {"xyz": self.position.tolist(), "point_index": int(self.point_index),
                "votes": int(self.votes), "weight": round(self.weight, 2),
                "visible": int(self.visible), "hit_rate": round(self.hit_rate, 3),
                "frames": self.frames[:12]}


def leaf_masks_in_frame(
    generator, bgr: np.ndarray, plant: np.ndarray, class_map: np.ndarray,
    leaf_class: int, min_leaf_fraction: float = 0.6,
    min_area_fraction: float = 0.004, whole_plant_fraction: float = 0.7,
) -> List[np.ndarray]:
    """SAM2 masks that are plausibly one leaf blade.

    The class map does the deciding. A mask is a leaf when most of its pixels
    were already labelled leaf, which means this never re-litigates organ
    identity -- it only proposes *boundaries*, which is the one thing the
    per-patch classifier cannot give and SAM2 can.
    """
    ys, xs = np.nonzero(plant)
    if len(xs) == 0:
        return []
    pad = 60
    y0, y1 = max(0, ys.min() - pad), min(bgr.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(bgr.shape[1], xs.max() + pad)

    crop = bgr[y0:y1, x0:x1].copy()
    crop_plant = plant[y0:y1, x0:x1]
    crop_class = class_map[y0:y1, x0:x1]
    crop[~crop_plant] = 0

    plant_area = int(crop_plant.sum())
    out: List[np.ndarray] = []
    for entry in generator.generate(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)):
        inside = entry["segmentation"] & crop_plant
        area = int(inside.sum())
        if area < min_area_fraction * plant_area:
            continue
        if area > whole_plant_fraction * plant_area:
            continue                      # SAM2 always offers the whole plant
        if (crop_class[inside] == leaf_class).mean() < min_leaf_fraction:
            continue                      # stem, holder, or a mixed blob
        full = np.zeros(plant.shape, bool)
        full[y0:y1, x0:x1] = inside
        out.append(full)
    return out


def tip_pixel(
    mask: np.ndarray, stem_pixels: np.ndarray, tip_prior: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, int, float, bool]]:
    """The masked pixel furthest from the stem: (x, y, distance, used_prior).

    The same rule the 3D side uses, applied where overlapping leaves are still
    distinct. Distance is measured to the nearest stem *pixel* rather than
    along the blade, because in one view a leaf is a flat region and its
    furthest extent from the stem is exactly its visible tip.

    `tip_prior` -- pixels a DINO seed class called "leaf tip" claimed -- fixes
    the failure mode of the rule on its own. SAM2 sometimes returns half a
    leaf, split along a shading boundary, and the furthest point of half a
    leaf is a point on its cut edge rather than a tip. Restricting the search
    to tip-labelled pixels rejects those, because an edge mid-blade is not
    tip-like to DINO even though it is far from the stem.

    Each model is used for what it is good at and nothing else: SAM2 for the
    boundary (which leaf), DINO for tipness (is this end a tip at all), and
    the geometry for the extremity (where exactly). The prior is a filter, not
    the answer -- it is broad, so the furthest point *within* it is still what
    gets voted.
    """
    from scipy.spatial import cKDTree

    if len(stem_pixels) == 0:
        return None
    tree = cKDTree(stem_pixels)

    used_prior = False
    search = mask
    if tip_prior is not None:
        gated = mask & tip_prior
        if gated.sum() >= 8:
            search, used_prior = gated, True

    ys, xs = np.nonzero(search)
    if len(xs) == 0:
        return None
    d = tree.query(np.stack([xs, ys], axis=1))[0]
    best = int(np.argmax(d))
    return int(xs[best]), int(ys[best]), float(d[best]), used_prior


def mask_facing(index_map: np.ndarray, mask: np.ndarray, points: np.ndarray,
                normals: np.ndarray, camera_centre: np.ndarray) -> float:
    """How square-on this leaf was to this camera, from 0 to 1.

    "Furthest point of the mask from the stem" is the tip only when the camera
    sees the leaf's true outline. Seen at an angle the outline is foreshortened
    and its extremity slides to a side corner -- which is where false tips down
    the sides of wide leaves come from. Scoring each vote by how face-on the
    blade was lets the oblique views abstain in proportion, so no range of
    viewing angles has to be chosen or re-fitted per plant.

    The averaging has to be done on the *directions*, and an earlier version
    got this wrong in a way worth recording: it averaged the cosine per point
    instead. Cosines of scattered normals average to about 0.5 whichever way
    the blade actually points, so every leaf scored 0.5 and the measure
    carried no information at all.

    Direction averaging also has to survive the normals' arbitrary sign -- P4b
    does not orient them, deliberately. The top eigenvector of the orientation
    tensor is the sign-free answer: it finds the axis the normals cluster
    around without ever needing to know which way along it they point. On this
    plant the normals of one blade cluster at 0.67-0.89 by that measure, so
    the blade orientation is real and worth measuring.
    """
    inside = index_map[mask]
    inside = inside[inside >= 0]
    if len(inside) < 8:
        return 0.0

    unit = normals[inside]
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-9)
    tensor = unit.T @ unit / len(unit)
    axis = np.linalg.eigh(tensor)[1][:, -1]          # dominant normal direction

    rays = camera_centre[None, :] - points[inside]
    rays = rays / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-9)
    return float(np.abs(rays.mean(axis=0) @ axis))


def point_at_pixel(index_map: np.ndarray, x: int, y: int, search: int = 16) -> int:
    """The cloud point at a pixel, or the nearest one within `search` pixels.

    An exact lookup is not enough and rejecting the miss is worse. A tip pixel
    sits on the outer edge of a mask, where the reconstruction has no points
    to render -- the carve erodes thin margins and the splat does not quite
    reach the silhouette. Measured on one frame that threw away four of twelve
    leaves, the long ones included, purely for ending where the cloud does not.

    Widening to the nearest rendered point keeps the vote and moves it a few
    pixels inward along the same blade, which is the right compromise: the
    tip of the *reconstruction* is what the 3D stage can act on anyway.
    """
    height, width = index_map.shape
    if 0 <= y < height and 0 <= x < width and index_map[y, x] >= 0:
        return int(index_map[y, x])

    y0, y1 = max(0, y - search), min(height, y + search + 1)
    x0, x1 = max(0, x - search), min(width, x + search + 1)
    window = index_map[y0:y1, x0:x1]
    hit = np.nonzero(window >= 0)
    if len(hit[0]) == 0:
        return -1
    dy = hit[0] + y0 - y
    dx = hit[1] + x0 - x
    nearest = int(np.argmin(dy * dy + dx * dx))
    return int(window[hit[0][nearest], hit[1][nearest]])


def cluster_votes(
    votes: Sequence[TipVote], points: np.ndarray, radius: float, min_votes: int,
    visibility: Optional[np.ndarray] = None, min_hit_rate: float = 0.0,
) -> List[TipCluster]:
    """Group votes that landed in the same place; keep the well-supported ones.

    A tip seen from thirty directions produces thirty votes within a few
    voxels of each other. A mask edge mistaken for a tip in one awkward view
    produces one, somewhere nobody else agrees with.

    `visibility` -- how many views rendered each point -- turns the raw count
    into a rate, which is what a leaf hidden in most frames needs. Judged on
    votes alone such a leaf loses to any well-lit edge; judged on the share of
    the views that could see it, it wins. Both numbers are kept and reported,
    because a high rate from three views is a weaker claim than the same rate
    from thirty and the caller should be able to see which it has.
    """
    if not votes:
        return []
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    index = np.array([v.point_index for v in votes])
    xyz = points[index]

    pairs = cKDTree(xyz).query_pairs(r=radius, output_type="ndarray")
    if len(pairs):
        graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                           shape=(len(xyz),) * 2)
        count, label = connected_components(graph, directed=False)
    else:
        count, label = len(xyz), np.arange(len(xyz))

    clusters: List[TipCluster] = []
    for c in range(count):
        member = np.nonzero(label == c)[0]
        if len(member) < min_votes:
            continue
        weight = float(sum(votes[m].facing for m in member))
        centre = xyz[member].mean(axis=0)
        # Report an actual cloud point, not the mean of a few: downstream this
        # seeds a graph traversal, which needs a node rather than a position.
        nearest = member[int(np.argmin(np.linalg.norm(xyz[member] - centre, axis=1)))]
        seen = int(visibility[index[member]].max()) if visibility is not None else 0
        cluster = TipCluster(
            position=points[index[nearest]], point_index=int(index[nearest]),
            votes=len(member), weight=weight, visible=seen,
            frames=sorted({votes[m].frame for m in member}))
        if visibility is not None and cluster.hit_rate < min_hit_rate:
            continue
        clusters.append(cluster)
    # Ranked by weight, not by count: a tip seen properly a few times beats
    # one glimpsed edge-on many times, which is the entire point.
    clusters.sort(key=lambda c: -c.weight)
    return clusters


def write_tips(path: Union[str, Path], clusters: Sequence[TipCluster],
               settings: Optional[dict] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"num_tips": len(clusters),
                   "settings": settings or {},
                   "tips": [c.to_dict() for c in clusters]}, f, indent=2)
    return path


def read_tips(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (positions, point indices) for tips voted by pose-tips."""
    with open(path) as f:
        payload = json.load(f)
    tips = payload.get("tips", [])
    if not tips:
        return np.zeros((0, 3)), np.zeros(0, np.int64)
    return (np.array([t["xyz"] for t in tips], float),
            np.array([t["point_index"] for t in tips], np.int64))
