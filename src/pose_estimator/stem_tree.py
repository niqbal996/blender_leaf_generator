"""The stem as a branched tree of curves, read off the stem cloud.

P5x used to draw the stem as one path, from the crown to the highest stem
tissue. On a branching plant that is one curve through a tree: on vogelmeere
(chickweed, 2026-10-07) it was 92 mm of a stem system 356 mm long, and every
leaf on a side branch then measured its petiole from the main stem.

This reads the whole tree, by the level-set method:

    1. geodesic distance from the foot, through the stem cloud only;
    2. cut into shells one stem diameter thick;
    3. each connected piece of a shell is a node, at its centroid -- one
       piece per branch the shell crosses, so a fork shows up as one piece
       becoming two;
    4. each node's parent is the piece one shell lower it touches most;
    5. spurs shorter than a few stem diameters are bumps, and are pruned.

Every length in it is measured, not set from the grid: the shell is the
stem's own diameter, read off provisional shells, and the prune length is a
multiple of it. The graph reach is the only grid-tied number, and it only
decides what counts as connected.

The tree is then split into axes, Gravelius-style: the main stem is the path
from the foot to the farthest end, and at every fork the child leading
farthest continues its parent's axis while the others start new axes one
order higher. Whether a terminal axis is a branch or a petiole is decided
against the leaves, in `classify_petioles`:

    a petiole is the stalk of one leaf: it ends in one blade and nothing
    else attaches along it. Anything else is a branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from pose_estimator.leaf import smooth_polyline

# A spur shorter than this many shells -- each one stem diameter thick -- is a
# bump on the stem, not an organ. The probe on vogelmeere pruned at 2.5 shells
# (5 mm at 2 mm shells) and kept every lateral and every petiole of the
# lowest leaf pair.
PRUNE_DIAMETERS = 3.0


@dataclass
class Axis:
    nodes: List[int]                 # tree nodes, starting at the fork it leaves its parent from
    order: int                       # 0 = main stem, 1 = off it, ...
    parent: int                      # parent axis, -1 for the main stem
    kind: str = "stem"               # "stem" (order 0), "branch" or "petiole"
    leaf: Optional[int] = None       # the leaf a petiole carries

    def to_dict(self, points: np.ndarray, axis_id: int) -> dict:
        return {"id": axis_id, "order": self.order, "parent": self.parent,
                "kind": self.kind, "leaf": self.leaf,
                "points": np.round(points, 6).tolist()}


@dataclass
class StemTree:
    centres: np.ndarray              # (K, 3) node positions
    parent: np.ndarray               # (K,) parent node, -1 for the root
    radius: float                    # measured stem radius
    shell: float                     # shell thickness used
    axes: List[Axis] = field(default_factory=list)
    foot: Optional[np.ndarray] = None

    def polylines(self, iterations: int = 3) -> List[np.ndarray]:
        """Every axis as a curve: node centres, lightly smoothed, ends fixed.

        A child axis starts on its parent's *smoothed* curve -- smoothing the
        parent moves the fork node, and a child starting from where the fork
        was before smoothing floats a shell's width off the stem -- at the
        point where the child really leaves it (`_junction`).
        """
        lines: List[np.ndarray] = []
        for axis in self.axes:                 # parents come before their children
            line = self.centres[axis.nodes].copy()
            if axis.parent >= 0:
                parent_line = lines[axis.parent]
                fork = self.axes[axis.parent].nodes.index(axis.nodes[0])
                line[0] = parent_line[_junction(parent_line, fork, line[1:])]
            if len(line) >= 3:
                line = smooth_polyline(line, iterations=iterations, strength=0.5)
            lines.append(line)
        return lines

    @property
    def forks(self) -> int:
        return int(sum(1 for k in range(len(self.parent)) if self._children(k) > 1))

    @property
    def ends(self) -> int:
        return int(sum(1 for k in range(len(self.parent)) if self._children(k) == 0))

    def _children(self, node: int) -> int:
        return int((self.parent == node).sum())

    def length(self) -> float:
        child = np.flatnonzero(self.parent >= 0)
        return float(np.linalg.norm(self.centres[child] - self.centres[self.parent[child]],
                                    axis=1).sum())

    def summary(self) -> dict:
        kinds: Dict[str, int] = {}
        for a in self.axes:
            kinds[a.kind] = kinds.get(a.kind, 0) + 1
        return {"radius": self.radius, "shell": self.shell, "nodes": int(len(self.centres)),
                "forks": self.forks, "ends": self.ends, "length": self.length(),
                "axes": kinds}


def _junction(parent_line: np.ndarray, fork: int, child: np.ndarray, window: int = 3) -> int:
    """Index on the parent's curve where a branch really leaves it.

    Just above a fork the shell slices still hold stem and branch as one
    piece, so the branch's first node of its own is already a stem's width or
    more off to the side. Joined straight to the fork node, the branch jumped
    sideways off the stem and then turned onto its course -- a T-junction where
    the plant has a slip road (vogelmeere, 2026-10-07).

    A branch leaves along its own direction. So its first nodes are extended
    back down that direction, and it joins the parent where that line passes
    closest -- searched only `window` nodes either side of the fork, so a
    branch running nearly parallel to the stem is not carried far along it.
    """
    if len(child) < 2:
        return fork
    back = child[0] - child[min(len(child), 4) - 1]      # from the branch toward the stem
    norm = np.linalg.norm(back)
    if norm == 0:
        return fork
    back /= norm
    lo, hi = max(fork - window, 0), min(fork + window + 1, len(parent_line))
    offset = parent_line[lo:hi] - child[0]
    along = offset @ back
    off = np.linalg.norm(offset - np.outer(along, back), axis=1)
    off[along < 0] = np.inf                               # only behind the branch, not past it
    return lo + int(np.argmin(off)) if np.isfinite(off).any() else fork


def _radius_graph(points: np.ndarray, reach: float) -> csr_matrix:
    pairs = cKDTree(points).query_pairs(reach, output_type="ndarray")
    if not len(pairs):
        return csr_matrix((len(points), len(points)))
    w = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    w = np.maximum(w, 1e-12)                  # coincident points still connect
    graph = coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2).tocsr()
    return graph.maximum(graph.T)


def _pieces(graph: csr_matrix, level: np.ndarray):
    """Connected pieces of each shell: (piece per point, piece count)."""
    g = graph.tocoo()
    same = level[g.row] == level[g.col]
    within = coo_matrix((np.ones(int(same.sum())), (g.row[same], g.col[same])),
                        shape=graph.shape)
    return connected_components(within, directed=False)[::-1]


def _thickness(points: np.ndarray, graph: csr_matrix, dist: np.ndarray, shell: float) -> float:
    """The stem's radius: how far its tissue sits from its own centre line.

    Each shell piece is a slice across one stem or branch. Its axis is the
    way distance grows through it, so the radial part of each point's offset
    from the slice centroid is what is left once the along-axis part is
    removed -- the slice's own thickness would otherwise count as width. The
    90th percentile per slice reaches the stem's surface whether the cloud is
    a shell or solid; the median over slices ignores the fat ones at forks
    and in the apical clusters.
    """
    piece, count = _pieces(graph, np.floor(dist / shell).astype(np.int64))
    order = np.argsort(piece, kind="stable")
    bounds = np.searchsorted(piece[order], np.arange(count + 1))
    per_piece = []
    for k in range(count):
        members = order[bounds[k]:bounds[k + 1]]
        if len(members) < 8:
            continue
        pts, d = points[members], dist[members]
        offset = pts - pts.mean(axis=0)
        high = d > np.median(d)
        axis = pts[high].mean(axis=0) - pts[~high].mean(axis=0) if high.any() and (~high).any() \
            else np.zeros(3)
        norm = np.linalg.norm(axis)
        if norm > 0:
            axis /= norm
            offset = offset - np.outer(offset @ axis, axis)
        per_piece.append(np.percentile(np.linalg.norm(offset, axis=1), 90))
    return float(np.median(per_piece)) if per_piece else shell / 2.0


def build(points: np.ndarray, foot: np.ndarray, reach: float,
          prune_diameters: float = PRUNE_DIAMETERS) -> Optional[StemTree]:
    """The stem tree of a stem cloud, grown from the point nearest `foot`.

    `reach` is the longest edge that counts as connected tissue. Returns None
    when there is too little stem to build anything from.
    """
    points = np.asarray(points, float)
    if len(points) < 10:
        return None
    graph = _radius_graph(points, reach)
    _, part = connected_components(graph, directed=False)
    start = int(np.argmin(np.linalg.norm(points - np.asarray(foot, float), axis=1)))
    sizes = np.bincount(part)
    if sizes[part[start]] < 0.1 * len(points):
        # The foot fell on a speck: start from the foot of the main body instead.
        body = np.flatnonzero(part == int(np.argmax(sizes)))
        start = int(body[np.argmin(points[body, 2])])
    keep = np.flatnonzero(part == part[start])
    start = int(np.searchsorted(keep, start))
    points, graph = points[keep], graph[keep][:, keep]
    dist = dijkstra(graph, directed=False, indices=start)

    # One stem diameter per shell, measured on provisional shells first.
    radius = _thickness(points, graph, dist, shell=2.0 * reach)
    shell = max(2.0 * radius, 1.5 * reach)
    level = np.floor(dist / shell).astype(np.int64)
    piece, count = _pieces(graph, level)
    sizes = np.bincount(piece, minlength=count)
    centres = np.zeros((count, 3))
    np.add.at(centres, piece, points)
    centres /= np.maximum(sizes, 1)[:, None]
    piece_level = np.zeros(count, np.int64)
    piece_level[piece] = level

    # Each piece hangs from the piece one shell lower that it touches most.
    g = graph.tocoo()
    up = level[g.row] == level[g.col] + 1
    pairs, touches = np.unique(np.stack([piece[g.row[up]], piece[g.col[up]]], 1),
                               axis=0, return_counts=True)
    parent = np.full(count, -1, np.int64)
    best = np.zeros(count)
    for (child, low), n in zip(pairs, touches):
        if n > best[child]:
            parent[child], best[child] = low, n
    root = int(piece[start])
    parent[root] = -1

    alive = _prune(centres, parent, root, prune_diameters * shell)
    # Keep only what is still attached to the root, renumbered densely.
    reached = np.zeros(count, bool)
    reached[root] = True
    for k in np.argsort(piece_level):
        if alive[k] and parent[k] >= 0 and reached[parent[k]]:
            reached[k] = True
    index = np.full(count, -1, np.int64)
    index[reached] = np.arange(int(reached.sum()))
    tree = StemTree(centres=centres[reached],
                    parent=np.where(parent[reached] >= 0, index[parent[reached]], -1),
                    radius=radius, shell=shell, foot=points[start])
    tree.axes = _axes(tree, int(index[root]))
    return tree


def _kids(parent: np.ndarray, alive: np.ndarray) -> List[List[int]]:
    kids: List[List[int]] = [[] for _ in range(len(parent))]
    for k in np.flatnonzero(alive & (parent >= 0)):
        if alive[parent[k]]:
            kids[int(parent[k])].append(int(k))
    return kids


def _prune(centres: np.ndarray, parent: np.ndarray, root: int, shortest: float) -> np.ndarray:
    """Drop spurs shorter than `shortest`, the shortest first.

    A spur is a terminal chain hanging off a fork. One at a time, re-reading
    the tree after each, so a fork whose children are all short keeps the
    longest of them as its continuation rather than losing every one.
    """
    alive = np.ones(len(parent), bool)
    while True:
        kids = _kids(parent, alive)
        spurs = []
        for end in np.flatnonzero(alive):
            if kids[end] or end == root:
                continue
            chain, node = [int(end)], int(end)
            while parent[node] >= 0 and len(kids[int(parent[node])]) == 1:
                node = int(parent[node])
                chain.append(node)
            fork = int(parent[node])
            if fork < 0:
                continue                      # the trunk itself, down to the root
            pts = centres[chain + [fork]]
            spurs.append((float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()), chain))
        if not spurs:
            return alive
        length, chain = min(spurs, key=lambda s: s[0])
        if length >= shortest:
            return alive
        alive[chain] = False


def _axes(tree: StemTree, root: int) -> List[Axis]:
    """Split the tree into axes: at each fork the farthest-reaching child carries on."""
    n = len(tree.parent)
    kids = _kids(tree.parent, np.ones(n, bool))
    order_bfs, depth = [root], np.zeros(n)
    for node in order_bfs:
        for c in kids[node]:
            depth[c] = depth[node] + np.linalg.norm(tree.centres[c] - tree.centres[node])
            order_bfs.append(c)
    farthest = depth.copy()
    for node in reversed(order_bfs):
        for c in kids[node]:
            farthest[node] = max(farthest[node], farthest[c])

    axes: List[Axis] = []
    stack = [(root, -1, 0, None)]
    while stack:
        node, parent_axis, order, attach = stack.pop()
        axis_id, nodes = len(axes), ([attach] if attach is not None else [])
        while True:
            nodes.append(node)
            ranked = sorted(kids[node], key=lambda c: -farthest[c])
            if not ranked:
                break
            for other in ranked[1:]:
                stack.append((other, axis_id, order + 1, node))
            node = ranked[0]
        axes.append(Axis(nodes=nodes, order=order, parent=parent_axis,
                         kind="stem" if order == 0 else "branch"))
    return axes


def classify_petioles(tree: StemTree, leaf_points: np.ndarray, leaf_ids: np.ndarray,
                      reach: float, dominance: float = 0.6, min_points: int = 5) -> int:
    """Mark the terminal axes that are one leaf's stalk. Returns how many.

    A petiole ends in one blade -- the leaf points around its far end belong
    (mostly) to one leaf -- and no other leaf attaches along it. A terminal
    axis that fails either test carries more than one organ, so it is a
    branch: a shoot ending in a bud between two leaves, say.
    """
    if not len(leaf_points):
        return 0
    finder = cKDTree(leaf_points)
    has_children = {a.parent for a in tree.axes}
    lines = tree.polylines()
    found = 0
    for i, axis in enumerate(tree.axes):
        if axis.order == 0 or i in has_children:
            continue
        line = lines[i]
        near_end = finder.query_ball_point(line[-1], 2.0 * tree.shell)
        ids, counts = np.unique(leaf_ids[near_end], return_counts=True)
        if not len(ids) or counts.max() < min_points or counts.max() < dominance * counts.sum():
            continue
        leaf = int(ids[np.argmax(counts)])
        # Along the stalk, past the fork it leaves from: nothing but this leaf.
        middle = line[2:-2]
        if len(middle):
            touching = np.concatenate([np.asarray(finder.query_ball_point(p, tree.radius + reach),
                                                  dtype=np.int64) for p in middle])
            other, n_other = np.unique(leaf_ids[touching], return_counts=True)
            if ((other != leaf) & (n_other >= min_points)).any():
                continue
        axis.kind, axis.leaf = "petiole", leaf
        found += 1
    return found
