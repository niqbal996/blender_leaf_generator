"""A skeleton built on leaves that are already known, not inferred.

P5 has to find the leaves before it can draw them. It locates the crown, runs
a geodesic depth field out over the leaf tissue, takes local maxima as
candidate tips, groups them, and assigns ownership outward -- a chain of
inference standing in for the fact that a SAM2 P2 hands it one silhouette and
no idea what is inside it.

P5x already knows. Every point carries the id of the leaf it belongs to,
voted from SAM3's own 2D instances, and the stem-and-petiole tissue carries
its own label. So the skeleton does not have to be discovered; it only has to
be traced, in two steps that use two different graphs:

    one Dijkstra from the crown over the whole plant finds where each leaf
    joins it -- the leaf's own point the crown reaches first. That approach
    is the petiole.

    a second Dijkstra from that attachment, over *only that leaf's points*,
    gives the blade. Its far end is the tip and the path to it is the midrib.

The second graph is the part that has to be restricted, and the reason is
worth keeping. A single shortest path from the crown to a tip is the cheapest
route to that point, which is not the route along the leaf: wherever stem
tissue runs beside a blade -- which is what a petiole is -- the cheap route
goes alongside and steps onto the blade only at the last node. The blade then
contributes one point and the leaf is thrown away. Measured on gaensefuss_1
at 512^3, that discarded 10 of 21 leaves, one of them holding 14,991 points,
and it got *worse* as the geometry got better, because a denser cloud offers
more tissue to route around the blade with.

Restricting the second search to the leaf's own points removes the short cut
instead of pricing it, so nothing has to be weighted or tuned.

This is what makes the tip reliable here in a way it was not before. A tip
found as a maximum of a depth field is the end of *something*; two leaves
lying against each other give one maximum and one of them is lost. A tip
found as the far end of a known leaf is the end of that leaf, and two leaves
touching are still two leaves because they were two instances in 2D.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pose_estimator.leaf import resample_by_arclength, smooth_polyline

# Sentinels shared with cli/leaf_instances.py.
UNSEEN = -1
SKELETON = -2
ROOT = -3


def _knn_graph(points: np.ndarray, k: int, max_edge: float):
    """Sparse symmetric kNN graph, edges longer than `max_edge` cut.

    The cap matters more than k. Without it the graph bridges the gap between
    two leaves that merely pass close to each other, and a path from the crown
    then reaches a tip by stepping through a neighbouring blade -- which makes
    the midrib jump between organs and the "tip" the far end of a different
    leaf.
    """
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    distances, neighbours = tree.query(points, k=min(k + 1, len(points)))
    distances, neighbours = distances[:, 1:], neighbours[:, 1:]   # drop self

    rows = np.repeat(np.arange(len(points)), neighbours.shape[1])
    cols = neighbours.ravel()
    values = distances.ravel()
    keep = np.isfinite(values) & (values <= max_edge)

    graph = csr_matrix((values[keep], (rows[keep], cols[keep])),
                       shape=(len(points), len(points)))
    return graph.maximum(graph.T)          # symmetric: an edge is an edge


def find_crown(points: np.ndarray, assignment: np.ndarray,
               fraction: float = 0.02) -> np.ndarray:
    """Where the plant meets its holder, as a point to trace from.

    **The root is deliberately excluded.** It is the lowest tissue on the
    plant, so "the bottom of the stem and root" finds it every time -- and the
    clamp jaws cut it off from the foliage, so in the cloud it is a separate
    connected component. Measured on gaensefuss_1: the crown landed in a
    4,552-point root blob, 9% of the plant, and every one of the 17 leaves was
    then correctly reported as unreachable from it.

    The stem tissue is the right support. In the plant frame the origin is
    already the clamp line, with the stem climbing from z=0 and the root
    hanging below it, so taking the foot of the stem is a refinement of a
    known answer rather than a search.
    """
    for support in (assignment == SKELETON, np.isin(assignment, (SKELETON, ROOT)),
                    assignment != UNSEEN):
        if support.any():
            break
    candidates = points[support]
    cut = np.quantile(candidates[:, 2], max(fraction, 1e-3))
    lowest = candidates[candidates[:, 2] <= cut]
    return lowest.mean(axis=0) if len(lowest) else candidates.mean(axis=0)


def trace(points: np.ndarray, assignment: np.ndarray, voxel: float,
          k: int = 12, max_edge_voxels: float = 3.0,
          samples: int = 32, smooth_iterations: int = 24) -> dict:
    """Crown, and per leaf a tip, a midrib and the petiole that carries it.

    `voxel` sets the only length scale: the longest edge the graph may use is
    `max_edge_voxels` of it. Everything else is read off the labels.
    """
    from scipy.sparse.csgraph import dijkstra

    drawn = assignment != UNSEEN
    if not drawn.any():
        return {"crown": None, "leaves": [], "unreachable": []}

    index = np.flatnonzero(drawn)
    local = points[index]
    labels = assignment[index]

    crown = find_crown(points, assignment)
    graph = _knn_graph(local, k, voxel * max_edge_voxels)

    # Trace from the graph node nearest the crown rather than from the crown
    # itself, which is a centroid and need not be a point of the cloud.
    start = int(np.argmin(np.linalg.norm(local - crown, axis=1)))

    # And the start has to be on the body being traced. A carved cloud is not
    # one connected piece -- the clamp cuts the root off, a thin petiole can
    # be carved through -- and a start stranded on a small island reports
    # every leaf as unreachable while the graph is perfectly healthy. Moving
    # it to the foot of the largest component is the honest repair, and it is
    # said out loud because it means the crown is not where the labels put it.
    from scipy.sparse.csgraph import connected_components

    n_parts, part = connected_components(graph, directed=False)
    moved = None
    if n_parts > 1:
        sizes = np.bincount(part)
        biggest = int(np.argmax(sizes))
        if part[start] != biggest:
            body = np.flatnonzero(part == biggest)
            foot = body[np.argmin(local[body][:, 2])]
            moved = {"from_component_size": int(sizes[part[start]]),
                     "to_component_size": int(sizes[biggest])}
            start = int(foot)
            crown = local[start]

    distance, predecessor = dijkstra(graph, indices=start, return_predecessors=True)

    leaves, unreachable, dropped_because = [], [], {}
    for leaf_id in sorted({int(v) for v in np.unique(labels) if v >= 0}):
        members = np.flatnonzero(labels == leaf_id)
        reach = distance[members]
        if not np.isfinite(reach).any():
            # Disconnected from the crown: real when a leaf's only bridge to
            # the plant was carved away, and reported rather than papered over
            # with a straight line that would look like a measurement.
            unreachable.append(leaf_id)
            dropped_because[leaf_id] = "no path from the crown"
            continue

        # Where this leaf joins the plant: its own point that the crown
        # reaches first. Everything before it is the petiole, everything
        # after it is the blade, and the labels -- not a distance -- say
        # which is which.
        attachment = int(members[np.nanargmin(np.where(np.isfinite(reach), reach, np.inf))])

        inside = _within_leaf(graph, members, attachment)
        if inside is None:
            unreachable.append(leaf_id)
            dropped_because[leaf_id] = "the leaf is a single isolated point"
            continue
        midrib_nodes, tip_local = inside

        # The petiole is what carried us to the attachment. Nodes of this same
        # leaf are dropped from it so a blade the path grazed on the way does
        # not get counted twice.
        approach = _walk_back(predecessor, attachment)
        petiole_nodes = approach[labels[approach] != leaf_id]

        midrib = _tidy(local[midrib_nodes], samples, smooth_iterations)
        petiole = (_tidy(local[petiole_nodes], max(samples // 2, 4), smooth_iterations)
                   if len(petiole_nodes) >= 2 else np.zeros((0, 3)))
        base = midrib[0]
        tip = local[tip_local]
        leaves.append({
            "id": leaf_id,
            "points": int(len(members)),
            "tip": [float(v) for v in tip],
            "base": [float(v) for v in base],
            "midrib": midrib.round(6).tolist(),
            "petiole": petiole.round(6).tolist(),
            "midrib_length": float(_length(midrib)),
            "petiole_length": float(_length(petiole)),
            "midrib_points": int(len(midrib_nodes)),
            "height": float(tip[2]),
        })

    return {"crown": [float(v) for v in crown], "leaves": leaves,
            "unreachable": unreachable, "crown_moved_to_largest_component": moved,
            "graph_components": int(n_parts), "dropped_because": dropped_because}


def _within_leaf(graph, members: np.ndarray, attachment: int):
    """(midrib node indices, tip) traced *inside* one leaf's own tissue.

    This is the whole correction. Taking the tip as the member furthest from
    the crown and walking the global predecessor tree back gives the cheapest
    route to that point, which is not the same thing as the route along the
    leaf: wherever stem tissue runs beside a blade -- which is exactly what a
    petiole is, and what the stem residual carries -- the cheap route runs
    alongside and steps onto the blade at the last node. The blade then
    contributes one point and the leaf is discarded.

    Restricting the search to the leaf's own points removes the short cut
    rather than penalising it, so no weight or margin has to be chosen. The
    tip becomes the far end of the blade *measured along the blade*, which is
    what a tip is, and the midrib cannot leave the leaf because no edge out of
    it exists in this subgraph.
    """
    from scipy.sparse.csgraph import connected_components, dijkstra

    if len(members) < 2:
        return None
    sub = graph[members][:, members]
    start = int(np.searchsorted(members, attachment))

    # A leaf carved into pieces can attach by a fragment. Tracing from a
    # one-point island reports a leaf with thousands of points as untraceable,
    # so the blade is taken from the largest piece and re-entered at whichever
    # of its points the crown reaches first -- the same repair the crown
    # itself gets, for the same reason.
    n_parts, part = connected_components(sub, directed=False)
    if n_parts > 1 and int((part == part[start]).sum()) < 2:
        sizes = np.bincount(part)
        biggest = np.flatnonzero(part == int(np.argmax(sizes)))
        if len(biggest) < 2:
            return None
        start = int(biggest[0])

    inner, predecessor = dijkstra(sub, indices=start, return_predecessors=True)
    if not np.isfinite(inner).any():
        return None
    tip = int(np.nanargmax(np.where(np.isfinite(inner), inner, -np.inf)))
    nodes = _walk_back(predecessor, tip)
    if len(nodes) < 2:
        return None
    return members[nodes], int(members[tip])


def _walk_back(predecessor: np.ndarray, node: int) -> np.ndarray:
    """Crown-to-node path, as node indices in crown-first order."""
    chain = []
    while node >= 0:
        chain.append(node)
        node = int(predecessor[node])
    return np.array(chain[::-1], dtype=int)


def _tidy(path: np.ndarray, samples: int, iterations: int) -> np.ndarray:
    """Smooth the graph walk, then sample it evenly along its length.

    A Dijkstra path is a staircase between neighbouring points, so its
    direction only takes as many values as the graph has edges per node.
    Smoothing before resampling is deliberate and the same order
    `leaf_pose.midrib` uses: decimating first samples the staircase rather
    than the curve underneath it.
    """
    if len(path) < 2:
        return path
    smoothed = smooth_polyline(path, iterations=iterations, strength=0.5)
    resampled, _length_of = resample_by_arclength(smoothed, samples)
    return np.asarray(resampled, float)


def _length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
