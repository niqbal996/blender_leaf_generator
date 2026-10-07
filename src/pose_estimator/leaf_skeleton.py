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
from scipy.spatial import cKDTree

from pose_estimator import stem_tree
from pose_estimator.leaf import resample_by_arclength, smooth_polyline

# A petiole path that runs through this many consecutive nodes of another leaf
# has crossed that leaf's blade (see `trace`).
CROSSING_NODES = 3

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
          samples: int = 32, smooth_iterations: int = 24,
          architecture: Optional[str] = None) -> dict:
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

    # The stem, once, and each petiole only from where its leaf's path leaves
    # it. Drawing each leaf's whole crown-to-leaf path as its "petiole", as
    # this used to, drew the stem once per leaf: a bundle of lines from the
    # crown to every leaf. Which plants *have* a stem is P5's --architecture,
    # not something to infer: a rosette's leaves meet at the crown and keep
    # their whole paths as petioles.
    #
    # Otherwise the stem is the branched tree of the stem cloud (`stem_tree`):
    # a main stem, its branches, and the stalks that end in one leaf, which
    # are that leaf's petiole. The single crown-to-highest-stem path this
    # replaced was one curve through a tree -- 92 mm of vogelmeere's 340 --
    # and every leaf on a side branch measured its petiole from it.
    tree = None
    if architecture != "rosette":
        tree = stem_tree.build(local[labels == SKELETON], crown, voxel * max_edge_voxels)
    axes_out, axis_lines, petiole_axis = [], [], {}
    if tree is not None and tree.axes:
        stem_tree.classify_petioles(tree, local[labels >= 0], labels[labels >= 0],
                                    voxel * max_edge_voxels)
        axis_lines = tree.polylines()
        axes_out = [axis.to_dict(line, i) for i, (axis, line) in enumerate(zip(tree.axes, axis_lines))]
        for axis, line in zip(tree.axes, axis_lines):
            if axis.kind == "petiole":
                # A leaf whose stalk split in two keeps the longer.
                if _length(line) > _length(petiole_axis.get(axis.leaf, np.zeros((0, 3)))):
                    petiole_axis[axis.leaf] = line
        stem = _tidy(axis_lines[0], samples, 0) if len(axis_lines[0]) >= 2 else axis_lines[0]
    else:
        stem = np.zeros((0, 3))

    leaves, unreachable, dropped_because = [], [], {}
    traced = []                              # (leaf_id, members, attachment, midrib nodes, tip)
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

        inside = _within_leaf(graph, members, attachment, distance)
        if inside is None:
            unreachable.append(leaf_id)
            dropped_because[leaf_id] = "the leaf is a single isolated point"
            continue
        midrib_nodes, tip_local = inside
        traced.append((leaf_id, attachment, midrib_nodes, tip_local, len(members)))


    # A carved stem is thick, and inside it the shortest-path tree splits
    # into side-by-side lanes: a leaf reached through a neighbouring lane
    # leaves the trunk low and climbs *inside* the stem, which drew as
    # parallel lines up the stem. So a petiole without a stalk of its own
    # starts at the last point of its path still inside a stem or branch --
    # within 1.5x the stem's measured radius of their centre lines. That
    # radius used to be measured around the lower trunk, which on a plant
    # branching low takes in the branches: 51 mm on vogelmeere, every leaf
    # base "inside the stem", and every petiole empty.
    trunk_lines = [line for axis, line in zip(tree.axes, axis_lines) if axis.kind != "petiole"] \
        if tree is not None else []
    stem_finder = cKDTree(np.vstack([_densify(line) for line in trunk_lines])) if trunk_lines else None
    stem_radius = 1.5 * tree.radius if tree is not None else 0.0

    for leaf_id, attachment, midrib_nodes, tip_local, n_members in traced:
        midrib = _tidy(local[midrib_nodes], samples, smooth_iterations)
        stalk, cut = petiole_axis.get(leaf_id), None
        if stalk is not None:
            # Where the stalk ends on this leaf's midrib. SAM3's leaf mask often
            # takes in the petiole, so the traced midrib can start back at the
            # stem and run out along the stalk; the blade starts where the
            # stalk ends. A stalk ending nowhere near the midrib is not this
            # leaf's after all, and neither is one ending near the tip: that is
            # stem tissue escorting the blade along its length, not a petiole
            # ending where the blade begins. The path rule below decides those.
            gaps = np.linalg.norm(midrib - stalk[-1], axis=1)
            at = int(np.argmin(gaps))
            if gaps[at] <= 2.0 * tree.shell and \
                    _length(midrib[at:]) >= 0.3 * _length(midrib):
                cut = at
        if cut is not None:
            if len(midrib) - cut >= 2 and _length(midrib[cut:]) > 0:
                midrib = _tidy(midrib[cut:], samples, 0)
            petiole = _tidy(np.vstack([stalk, midrib[:1]]), max(samples // 2, 4), 0)
        else:
            # No stalk of its own: the stretch of this leaf's path that leads
            # into it. Walking back from the leaf, it ends at a stem or branch
            # -- or where the path crossed another leaf's blade. Leaves touching
            # a neighbour are often reached *through* it: on vogelmeere the
            # shortest paths to leaves 8, 9 and 37 climbed leaf 6's stalk and
            # crossed 11-12 nodes of its blade, and each drew a copy of leaf
            # 6's petiole. A leaf reached only through a neighbour gets no
            # petiole rather than its neighbour's.
            #
            # A crossing, not a graze: a run of CROSSING_NODES. Leaf clouds
            # carry stray specks, and the good petioles there met none or 2.
            # Another leaf's *stalk* is no evidence either way -- stalk ends
            # sit in crowds of blades, 2-3 leaves within 1 mm, and stopping at
            # them cut 6 good petioles. Nodes of this same leaf are dropped so
            # a blade the path grazed on the way does not count twice.
            approach = _walk_back(predecessor, attachment)
            start_at = 0
            on_trunk = np.zeros(len(approach), bool)
            if stem_finder is not None and stem_radius > 0:
                on_trunk = stem_finder.query(local[approach])[0] <= stem_radius
            on_path = labels[approach]
            stop = on_trunk | _runs(((on_path >= 0) & (on_path != leaf_id)), CROSSING_NODES)
            last = np.flatnonzero(stop)
            if len(last):
                # From the stem's surface, or from just past the neighbour.
                start_at = int(last[-1]) + (0 if on_trunk[last[-1]] else 1)
            branch = approach[start_at:]
            petiole_nodes = branch[labels[branch] != leaf_id]
            petiole = (_tidy(local[petiole_nodes], max(samples // 2, 4), smooth_iterations)
                       if len(petiole_nodes) >= 2 else np.zeros((0, 3)))
        base = midrib[0]
        tip = local[tip_local]
        leaves.append({
            "id": leaf_id,
            "points": int(n_members),
            "tip": [float(v) for v in tip],
            "base": [float(v) for v in base],
            "midrib": midrib.round(6).tolist(),
            "petiole": petiole.round(6).tolist(),
            "midrib_length": float(_length(midrib)),
            "petiole_length": float(_length(petiole)),
            "petiole_from": "stalk" if cut is not None else "path",
            "midrib_points": int(len(midrib_nodes)),
            "height": float(tip[2]),
        })

    return {"crown": [float(v) for v in crown], "stem": stem.round(6).tolist(), "leaves": leaves,
            "axes": axes_out,
            "stem_tree": tree.summary() if tree is not None else None,
            "unreachable": unreachable, "crown_moved_to_largest_component": moved,
            "graph_components": int(n_parts), "dropped_because": dropped_because}


def _densify(line: np.ndarray, per_segment: int = 8) -> np.ndarray:
    line = np.asarray(line, float)
    if len(line) < 2:
        return line
    t = np.linspace(0.0, 1.0, per_segment, endpoint=False)
    pieces = [a + (b - a) * t[:, None] for a, b in zip(line[:-1], line[1:])]
    return np.vstack(pieces + [line[-1:]])


def _within_leaf(graph, members: np.ndarray, attachment: int,
                 distance: Optional[np.ndarray] = None):
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
    #
    # It applies whenever the attachment is not on the blade's largest piece,
    # not only when it is a one-point island: a two-point fragment at the
    # petiole confined the search to itself and put the "tip" at the stem end
    # (10 of 37 leaves on gaensefuss_1). And the re-entry point used to be the
    # piece's first point in index order -- arbitrary -- not the crown-nearest
    # one this comment always promised.
    n_parts, part = connected_components(sub, directed=False)
    if n_parts > 1:
        sizes = np.bincount(part)
        largest = int(np.argmax(sizes))
        if part[start] != largest and sizes[largest] > sizes[part[start]]:
            biggest = np.flatnonzero(part == largest)
            if len(biggest) < 2:
                return None
            if distance is not None and np.isfinite(distance[members[biggest]]).any():
                d = np.where(np.isfinite(distance[members[biggest]]),
                             distance[members[biggest]], np.inf)
                start = int(biggest[int(np.argmin(d))])
            else:
                start = int(biggest[0])

    inner, predecessor = dijkstra(sub, indices=start, return_predecessors=True)
    if not np.isfinite(inner).any():
        return None
    tip = int(np.nanargmax(np.where(np.isfinite(inner), inner, -np.inf)))
    nodes = _walk_back(predecessor, tip)
    if len(nodes) < 2:
        return None
    return members[nodes], int(members[tip])


def _runs(flags: np.ndarray, length: int) -> np.ndarray:
    """`flags`, keeping only the runs of at least `length` consecutive Trues."""
    out = np.zeros(len(flags), bool)
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run >= length:
            out[i - length + 1:i + 1] = True
    return out


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
