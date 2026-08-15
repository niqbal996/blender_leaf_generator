"""P5 driven by P4c organ labels rather than by geometry alone.

The geometry-only path had to *infer* which branches were leaves, and every
step of that inference needed a threshold: `min_branch_fraction` to decide how
short an organ could be, a climb-rate rule to decide where the stem ended, a
tip-termination rule to reject internodes. Each was calibrated on one
specimen and would drift on the next.

With per-point labels none of those decisions remain:

- **The stem is given**, so its centreline is fitted through stem points
  instead of traced by guessing which branch keeps climbing.
- **Leaves are given**, so no minimum-length threshold decides what counts.
- **Attachment points are given**, because a leaf point adjacent to a stem
  point *is* the insertion. That was the single worst defect in the previous
  P6: leaf bases landed mid-blade, which hooked the midribs and produced
  impossible width profiles.

Instancing uses the stem too. Connectivity alone cannot split leaves whose
blades touch -- measured on DSC_0009, no connectivity radius separated them
(2.5 voxels gave one 49k-point blob, 1.1 fragmented small leaves before
splitting the blob cleanly). But two touching leaves still meet the stem at
*different* places, so leaf points are grouped by which attachment site they
reach first through the leaf-only graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


@dataclass
class LabelledStructure:
    stem_path: np.ndarray                      # (M, 3) base -> apex
    leaf_ids: np.ndarray                       # per leaf-point instance id, -1 = dropped
    leaf_points: np.ndarray                    # (K, 3) the leaf points leaf_ids indexes
    attachments: List[np.ndarray] = field(default_factory=list)   # per instance
    tips: List[np.ndarray] = field(default_factory=list)
    axes: List[np.ndarray] = field(default_factory=list)          # per instance polyline
    root_points: Optional[np.ndarray] = None

    @property
    def num_leaves(self) -> int:
        return len(self.axes)


def fit_stem_path(stem_points: np.ndarray, num_stations: int = 24,
                  min_bin: int = 8) -> np.ndarray:
    """Centreline through the stem points, from lowest to highest.

    Height-binned centroids rather than a skeletonisation: the points are
    already known to be stem, so the only question left is where its middle
    runs, and a cross-section's centroid answers that directly.
    """
    if len(stem_points) < min_bin * 3:
        return stem_points[np.argsort(stem_points[:, 2])]

    z = stem_points[:, 2]
    edges = np.linspace(z.min(), z.max(), num_stations + 1)
    station = np.clip(np.digitize(z, edges) - 1, 0, num_stations - 1)

    path = [stem_points[station == i].mean(axis=0)
            for i in range(num_stations) if (station == i).sum() >= min_bin]
    if len(path) < 2:
        return stem_points[np.argsort(z)]
    return _smooth(np.array(path))


def _smooth(points: np.ndarray, iterations: int = 6, strength: float = 0.35) -> np.ndarray:
    out = points.astype(float).copy()
    for _ in range(iterations):
        out[1:-1] += strength * (0.5 * (out[:-2] + out[2:]) - out[1:-1])
    return out


def _knn_graph(points: np.ndarray, k: int) -> csr_matrix:
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(k + 1, len(points)))
    rows = np.repeat(np.arange(len(points)), indices.shape[1] - 1)
    cols = indices[:, 1:].ravel()
    data = distances[:, 1:].ravel()
    graph = csr_matrix((data, (rows, cols)), shape=(len(points), len(points)))
    return graph.maximum(graph.T)


def instance_by_attachment(
    leaf_points: np.ndarray,
    stem_points: np.ndarray,
    contact_radius: float,
    k_neighbors: int = 10,
    min_points: int = 150,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Group leaf points by which stem contact they reach first.

    Returns (instance id per leaf point, attachment position per instance).

    Two leaves whose blades overlap are one connected component and cannot be
    split by proximity. They are still distinct organs because they meet the
    stem at different heights and azimuths, so the split is made at the
    *attachment* and propagated outward along the leaf tissue by geodesic
    distance -- which follows the lamina rather than cutting across the gap
    between two leaves that merely touch.
    """
    if len(leaf_points) == 0 or len(stem_points) == 0:
        return np.full(len(leaf_points), -1, np.int32), []

    # Leaf points touching the stem are the candidate insertions.
    stem_tree = cKDTree(stem_points)
    contact = stem_tree.query(leaf_points)[0] <= contact_radius
    if not contact.any():
        return np.full(len(leaf_points), -1, np.int32), []

    # Cluster the contacts into distinct sites; one leaf touches the stem over
    # a small patch, not a single point.
    contact_idx = np.nonzero(contact)[0]
    sites = _cluster(leaf_points[contact_idx], contact_radius * 2.0)
    num_sites = int(sites.max()) + 1 if len(sites) else 0
    if num_sites == 0:
        return np.full(len(leaf_points), -1, np.int32), []

    # Multi-source geodesic over leaf tissue only: every leaf point is claimed
    # by the site it can reach through leaf points with the least travel.
    graph = _knn_graph(leaf_points, k_neighbors)
    best_distance = np.full(len(leaf_points), np.inf)
    owner = np.full(len(leaf_points), -1, np.int32)

    for site in range(num_sites):
        seeds = contact_idx[sites == site]
        distances = dijkstra(graph, directed=False, indices=seeds, min_only=True)
        closer = distances < best_distance
        best_distance[closer] = distances[closer]
        owner[closer] = site

    # Drop sites too small to be an organ, then renumber compactly.
    counts = np.bincount(owner[owner >= 0], minlength=num_sites)
    keep = [s for s in np.argsort(-counts) if counts[s] >= min_points]
    remap = np.full(num_sites, -1, np.int32)
    for new_id, site in enumerate(keep):
        remap[site] = new_id

    instances = np.where(owner >= 0, remap[owner], -1)
    attachments = [leaf_points[contact_idx[sites == site]].mean(axis=0) for site in keep]
    return instances, attachments


def _cluster(points: np.ndarray, radius: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, np.int32)
    pairs = cKDTree(points).query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(len(points), dtype=np.int32)
    graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int32)


def leaf_axis(points: np.ndarray, attachment: np.ndarray,
              num_stations: int = 20, min_bin: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Midrib polyline and tip for one leaf, ordered from its attachment.

    Stations are shells of distance from the attachment, and the tip is simply
    the farthest point. Because the attachment is now *known* rather than
    inferred, s=0 is genuinely the petiole -- the previous P6 resolved it by
    guessing which end of a skeleton branch was inner, got it wrong on a
    drooping leaf, and produced a blade that measured widest at its own base.
    """
    distance = np.linalg.norm(points - attachment, axis=1)
    tip = points[int(np.argmax(distance))]

    edges = np.linspace(0.0, distance.max(), num_stations + 1)
    station = np.clip(np.digitize(distance, edges) - 1, 0, num_stations - 1)

    axis = [attachment]
    for i in range(num_stations):
        members = points[station == i]
        if len(members) >= min_bin:
            axis.append(members.mean(axis=0))
    axis.append(tip)
    return _smooth(np.array(axis)), tip


def build_from_labels(
    points: np.ndarray,
    labels: np.ndarray,
    class_order: Sequence[str],
    voxel: float,
    contact_voxels: float = 3.0,
    min_leaf_points: int = 150,
) -> LabelledStructure:
    """Full structure from a labelled cloud, in the plant frame."""
    leaf_ids_in_order = [i for i, n in enumerate(class_order) if "leaf" in n]
    stem_ids = [i for i, n in enumerate(class_order) if n in ("stem", "petiole", "branch")]
    root_ids = [i for i, n in enumerate(class_order) if n == "root"]

    leaf_points = points[np.isin(labels, leaf_ids_in_order)]
    stem_points = points[np.isin(labels, stem_ids)]
    root_points = points[np.isin(labels, root_ids)] if root_ids else None

    stem_path = fit_stem_path(stem_points) if len(stem_points) else np.zeros((0, 3))

    instances, attachments = instance_by_attachment(
        leaf_points, stem_points, contact_radius=voxel * contact_voxels,
        min_points=min_leaf_points)

    axes, tips = [], []
    for instance in range(len(attachments)):
        subset = leaf_points[instances == instance]
        if len(subset) < min_leaf_points:
            continue
        axis, tip = leaf_axis(subset, attachments[instance])
        axes.append(axis)
        tips.append(tip)

    return LabelledStructure(
        stem_path=stem_path, leaf_ids=instances, leaf_points=leaf_points,
        attachments=attachments[: len(axes)], tips=tips, axes=axes,
        root_points=root_points,
    )
