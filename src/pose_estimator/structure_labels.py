"""P5 structure from the P4c organ labels: stem centreline and leaf instances.

The labels answer "what is this point". This module answers "which leaf is it",
and then "where does that leaf's midrib run".

**Leaves are separated by their tips, not by their attachments.** The earlier
version grew each leaf outward from where it touched the stem, and that fails
on exactly the plants worth measuring: two leaves fused near the apex share one
contact patch, so they were seen as one organ. Measured on plant_9, one
instance ended up holding 45% of all leaf tissue while the rest split into ten
fragments -- eleven instances for seven leaves.

A tip survives what an attachment does not. Two fused blades still have two
extremities, so tips are counted first and each leaf is then grown *inward*
from its own tip. The same measurement: the largest claim drops from 45% to
11%, the claims come out evenly sized, and the leaf count holds at 7 across a
range of settings where the old method swept 178 -> 73 -> 28 -> 14 -> 4 with no
stable region anywhere.

Deciding how many tips there are needs no tuned radius either. Sweep a level
downward through the depth field: each local maximum starts its own component,
and when two components touch, the shallower one stops being separate. Two
bumps on one blade join high up on that blade; two real leaves can only join
by descending to where they both meet the stem. So `peak - saddle` -- the
persistence -- is near the peak's own depth for a real leaf and small for a
second high point on a leaf already counted. Measured on plant_9 the five
large leaves all have a saddle of exactly zero, while a competing maximum at
depth 26 saddles at 19 and is correctly rejected.

Growth also stops at the stem. A front that runs down its own blade, reaches
the stem and keeps going will hang a tiny bud off the end of an unrelated
midrib -- so the stem contact points can be reached but not travelled
through, and tissue left over with no tip of its own becomes its own leaf
rather than someone else's tail.

Every intermediate is kept on the `Instancing` record rather than being
consumed in place: the depth field, the candidate tips, which candidates were
merged, and what each surviving tip claimed. A leaf count is not a diagnosis,
and when this stage is wrong the answer is always in one of those steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


@dataclass
class Instancing:
    """Every step of the tip-driven split, kept for inspection."""

    depth: np.ndarray                  # (N,) geodesic distance from the stem, per leaf point
    contact_index: np.ndarray          # indices of leaf points touching the stem
    candidate_tips: np.ndarray         # indices into the leaf points
    tip_group: np.ndarray              # (n_maxima, 3): peak index, peak depth, saddle depth
    accepted_tips: np.ndarray          # one representative index per merged group
    owner: np.ndarray                  # (N,) instance id per leaf point, -1 = unassigned
    distance_from_tip: np.ndarray      # (N,) geodesic distance to the winning tip
    dropped: List[int] = field(default_factory=list)   # groups cut by min_points
    base: object = None                # BaseRegion, whenever the geometry was consulted
    architecture: str = "caulescent"   # what the depth field was actually started from

    def to_dict(self) -> dict:
        finite = np.isfinite(self.depth)
        return {
            "architecture": self.architecture,
            "base": self.base.to_dict() if self.base is not None else None,
            "leaf_points": int(len(self.depth)),
            "reachable_from_stem": int(finite.sum()),
            "contact_points": int(len(self.contact_index)),
            "candidate_tips": int(len(self.candidate_tips)),
            "tips_after_merge": int(len(self.accepted_tips)),
            "instances_kept": int(len({int(o) for o in self.owner if o >= 0})),
            "instances_dropped_as_too_small": len(self.dropped),
            "unassigned_points": int((self.owner < 0).sum()),
            "max_depth": float(self.depth[finite].max()) if finite.any() else 0.0,
        }


@dataclass
class LabelledStructure:
    stem_path: np.ndarray                      # (M, 3) base -> apex
    leaf_ids: np.ndarray                       # per leaf-point instance id, -1 = dropped
    leaf_points: np.ndarray                    # (K, 3) the leaf points leaf_ids indexes
    attachments: List[np.ndarray] = field(default_factory=list)   # per instance
    tips: List[np.ndarray] = field(default_factory=list)
    axes: List[np.ndarray] = field(default_factory=list)          # per instance, base -> tip
    root_points: Optional[np.ndarray] = None
    instancing: Optional[Instancing] = None

    @property
    def num_leaves(self) -> int:
        return len(self.axes)


# --------------------------------------------------------------------------
# Stem
# --------------------------------------------------------------------------


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


def fit_smooth_curve(
    points: np.ndarray, tolerance: float, num_out: int = 0, degree: int = 3,
) -> np.ndarray:
    """A smoothing cubic spline through a polyline, resampled evenly.

    Laplacian smoothing was doing this job and does it badly. It only ever
    averages a point with its two neighbours, so it cannot tell a real bend
    from a station that landed a voxel off: enough passes to remove the
    zigzag also drag genuine curvature out, and too few leave the wobble.

    A spline separates the two properly. `tolerance` is how far the curve may
    sit from the input *per point*, so setting it to a voxel or two says
    "ignore deviations at the scale of the sampling, keep everything larger" --
    which is the actual distinction wanted, and it is expressed in the
    plant's own units rather than as an iteration count.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    if len(points) < 4:
        return points

    from scipy.interpolate import splev, splprep

    # Duplicate stations make the parameterisation singular, and binned
    # centroids produce them whenever two bins land on the same tissue.
    keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9])
    points = points[keep]
    if len(points) < 4:
        return points

    count = num_out or max(len(points), 12)
    try:
        # s is a total squared-error budget, so it scales with how many points
        # are being fitted; tolerance stays a per-point distance.
        tck, _u = splprep(points.T, s=len(points) * tolerance ** 2,
                          k=min(degree, len(points) - 1))
        return np.stack(splev(np.linspace(0, 1, count), tck), axis=1)
    except (TypeError, ValueError):
        # Degenerate input (collinear, or too few distinct knots) -- the
        # unsmoothed polyline is still better than nothing.
        return _smooth(points)


# --------------------------------------------------------------------------
# Leaf instancing, step by step
# --------------------------------------------------------------------------


def leaf_graph(leaf_points: np.ndarray, k: int = 10) -> csr_matrix:
    """Symmetric kNN graph over leaf tissue, edges weighted by length.

    Distances along this graph are what "geodesic" means everywhere below:
    travel through leaf tissue rather than straight through the air, which is
    what keeps a curved blade's far end from looking near its own base.
    """
    tree = cKDTree(leaf_points)
    distances, indices = tree.query(leaf_points, k=min(k + 1, len(leaf_points)))
    rows = np.repeat(np.arange(len(leaf_points)), indices.shape[1] - 1)
    graph = csr_matrix((distances[:, 1:].ravel(), (rows, indices[:, 1:].ravel())),
                       shape=(len(leaf_points),) * 2)
    return graph.maximum(graph.T)


def depth_from_stem(
    leaf_points: np.ndarray, stem_points: np.ndarray, graph: csr_matrix,
    contact_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Geodesic distance from the stem to every leaf point.

    Returns (depth, indices of the leaf points touching the stem). Points no
    path reaches get infinity rather than a fabricated value -- an island of
    leaf tissue with no route to the stem is a real thing to know about.
    """
    if len(stem_points) == 0 or len(leaf_points) == 0:
        return np.full(len(leaf_points), np.inf), np.zeros(0, np.int64)

    contact = np.nonzero(cKDTree(stem_points).query(leaf_points)[0] <= contact_radius)[0]
    if len(contact) == 0:
        return np.full(len(leaf_points), np.inf), contact
    return dijkstra(graph, directed=False, indices=contact, min_only=True), contact


def tip_persistence(graph: csr_matrix, depth: np.ndarray) -> List[Tuple[int, float, float]]:
    """Every local maximum of depth, with the level at which it stops being one.

    Returns (peak index, peak depth, saddle depth) per maximum.

    Sweep a level downward from the deepest tissue. Each local maximum starts
    its own component; when two components touch, the shallower one stops
    being a separate thing and the level where that happened is its *saddle*.
    Two bumps on one blade join high up on that blade, because the tissue
    between them is itself high. Two genuinely different leaves can only join
    by descending to where they meet the stem, so their saddle is near zero.

    That makes `peak - saddle` -- the persistence -- the honest measure of "is
    this its own leaf", and it needs no distance threshold, because the answer
    is written in the shape of the depth field rather than in how far apart
    two points happen to be. It also fixes the case a separation radius cannot:
    a long leaf whose blade offers a second high point was being split in two.
    """
    n = len(depth)
    finite = np.isfinite(depth)
    order = np.argsort(-np.where(finite, depth, -np.inf))

    parent = np.full(n, -1, np.int64)
    peak_of: Dict[int, float] = {}
    records: List[Tuple[int, float, float]] = []
    peak_index: Dict[int, int] = {}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    for i in order:
        i = int(i)
        if not finite[i]:
            break
        parent[i] = i
        roots = {find(int(j)) for j in graph.indices[graph.indptr[i]:graph.indptr[i + 1]]
                 if parent[int(j)] != -1}
        if not roots:
            peak_of[i] = float(depth[i])          # a new maximum is born here
            peak_index[i] = i
            continue
        alive = sorted(roots, key=lambda r: -peak_of[r])
        survivor = alive[0]
        for other in alive[1:]:
            records.append((peak_index[other], peak_of[other], float(depth[i])))
            parent[other] = survivor
        parent[i] = survivor
        if depth[i] > peak_of[survivor]:
            peak_of[survivor] = float(depth[i])

    for root in {find(i) for i in range(n) if parent[i] != -1}:
        # Never merged into anything: it reaches all the way to the stem.
        records.append((peak_index[root], peak_of[root], 0.0))
    return records


def select_tips(
    records: Sequence[Tuple[int, float, float]],
    min_depth: float,
    min_persistence_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the maxima into real leaf tips and bumps on a blade.

    Returns (all candidates above `min_depth`, the accepted subset).

    A maximum is its own leaf when it survives down through most of its own
    depth -- `(peak - saddle) / peak` above the ratio. Expressed as a fraction
    of the peak rather than in voxels so a small leaf is judged by the same
    rule as a large one.
    """
    candidates, accepted = [], []
    for index, peak, saddle in sorted(records, key=lambda r: -r[1]):
        if peak < min_depth:
            continue
        candidates.append(int(index))
        if (peak - saddle) / max(peak, 1e-12) > min_persistence_ratio:
            accepted.append(int(index))
    return np.array(candidates, np.int64), np.array(accepted, np.int64)


def grow_from_tips(
    graph: csr_matrix, tips: np.ndarray, n_points: int,
    blocked_at: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Claim leaf tissue for the tip that reaches it with least travel.

    Growing inward from the tips rather than outward from the stem is what
    separates two blades fused at the apex: their fronts meet at the fusion
    instead of being decided by a contact patch they share.

    `blocked_at` -- the leaf points touching the stem -- may be *reached* but
    not travelled through. Without that, a front runs down its own blade,
    arrives at the stem and simply keeps going up whatever it meets next, so a
    tiny bud on the far side ends up on the end of another leaf's midrib. A
    leaf's territory has to stop where the leaf does.
    """
    if blocked_at is not None and len(blocked_at):
        graph = graph.tolil(copy=True)
        for point in blocked_at:
            graph.rows[int(point)] = []
            graph.data[int(point)] = []
        graph = graph.tocsr()
        directed = True
    else:
        directed = False

    best = np.full(n_points, np.inf)
    owner = np.full(n_points, -1, np.int64)
    for instance, tip in enumerate(tips):
        distances = dijkstra(graph, directed=directed, indices=[int(tip)], min_only=True)
        closer = distances < best
        best[closer] = distances[closer]
        owner[closer] = instance
    return owner, best


def claim_orphans(
    graph: csr_matrix, owner: np.ndarray, distance: np.ndarray,
    depth: np.ndarray, min_points: int, min_depth: float,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Give leftover tissue its own instance rather than someone else's.

    Blocking growth at the stem leaves anything with no tip of its own
    unclaimed -- typically a bud too shallow to register as a maximum. The old
    behaviour absorbed it into whichever leaf's front arrived first, which put
    it on the end of an unrelated midrib.

    A leftover piece becomes its own leaf only if it both holds enough points
    *and* actually stands away from the stem: it must reach `min_depth`, the
    same bar a tip has to clear. Without that second test this promotes tissue
    lying against the stem into "leaves" whose deepest point is zero voxels
    from it -- which is not a leaf, it is the bit of blade base the contact
    block just cut off. Anything failing either test stays honestly
    unassigned rather than being attached to a leaf it does not belong to.
    """
    free = owner < 0
    if not free.any():
        return owner, distance, []

    index = np.nonzero(free)[0]
    sub = graph[index][:, index]
    count, component = connected_components(sub, directed=False)

    new_tips: List[int] = []
    next_id = int(owner.max()) + 1 if (owner >= 0).any() else 0
    for c in range(count):
        member = index[component == c]
        if len(member) < min_points:
            continue
        reach = np.where(np.isfinite(depth[member]), depth[member], -np.inf)
        if reach.max() < min_depth:
            continue
        seed = member[int(np.argmax(reach))]
        local = dijkstra(sub, directed=False,
                         indices=[int(np.nonzero(index == seed)[0][0])], min_only=True)
        owner[member] = next_id
        distance[member] = local[component == c]
        new_tips.append(int(seed))
        next_id += 1
    return owner, distance, new_tips


def instance_by_tips(
    leaf_points: np.ndarray,
    stem_points: np.ndarray,
    contact_radius: float,
    k_neighbors: int = 10,
    min_points: int = 150,
    min_tip_depth: float = 0.0,
    min_persistence_ratio: float = 0.5,
    architecture: str = "caulescent",
) -> Instancing:
    """Split leaf tissue into leaves, keeping every intermediate.

    `architecture` decides where the depth field starts, and is told rather
    than inferred. "caulescent" seeds from stem tissue, as always. "rosette"
    ignores stem labels and locates the crown geometrically -- a thistle or a
    sugar beet has no stem to seed from, and the old behaviour there was 0
    contact points, 0 tips, 0 leaves.
    """
    n = len(leaf_points)
    if n == 0:
        empty = np.zeros(0)
        return Instancing(depth=empty, contact_index=np.zeros(0, np.int64),
                          candidate_tips=np.zeros(0, np.int64), tip_group=np.zeros(0, np.int64),
                          accepted_tips=np.zeros(0, np.int64),
                          owner=np.zeros(0, np.int64), distance_from_tip=empty)

    graph = leaf_graph(leaf_points, k_neighbors)

    # Which architecture this is comes from --architecture, not from a guess.
    # It is known when the capture is made, and an earlier version that
    # inferred it flipped thistle1's verdict purely because P4c had started
    # labelling the crown "stem".
    base = None
    if architecture == "rosette":
        base = _crown_base(leaf_points, stem_points, graph, k_neighbors)
        if base is None or not len(base.nodes):
            raise ValueError(
                "--architecture rosette, but no crown could be located: the leaf tissue "
                "has fewer than two geodesic extremities, so there are no paths to "
                "intersect. Check p4c/labels_vis.ply -- this usually means the labels "
                "are wrong rather than the plant is.")
        contact = base.nodes
        depth = dijkstra(graph, directed=False, indices=contact, min_only=True)
    else:
        depth, contact = depth_from_stem(leaf_points, stem_points, graph, contact_radius)

    records = tip_persistence(graph, depth)
    candidates, tips = select_tips(records, min_tip_depth, min_persistence_ratio)
    owner, distance_from_tip = grow_from_tips(graph, tips, n, blocked_at=contact)
    owner, distance_from_tip, orphan_tips = claim_orphans(
        graph, owner, distance_from_tip, depth, min_points, min_tip_depth)
    if orphan_tips:
        tips = np.concatenate([tips, np.array(orphan_tips, np.int64)])

    # Drop instances too small to be an organ, then renumber compactly so the
    # ids stay dense for everything downstream.
    counts = np.bincount(owner[owner >= 0], minlength=max(len(tips), 1))
    keep = [g for g in np.argsort(-counts) if counts[g] >= min_points]
    dropped = [int(g) for g in range(len(counts)) if g not in keep and counts[g] > 0]
    remap = np.full(max(len(counts), 1), -1, np.int64)
    for new_id, g in enumerate(keep):
        remap[g] = new_id

    persistence = np.array([[idx, peak, saddle] for idx, peak, saddle in records], float)

    return Instancing(
        base=base,
        architecture=architecture,
        depth=depth,
        contact_index=contact,
        candidate_tips=candidates,
        tip_group=persistence,
        accepted_tips=tips[keep] if len(tips) else tips,
        owner=np.where(owner >= 0, remap[owner], -1),
        distance_from_tip=distance_from_tip,
        dropped=dropped,
    )


# --------------------------------------------------------------------------
# Trunk and attachments, from the shoot's own tree structure
# --------------------------------------------------------------------------


def root_anchor(root_points: np.ndarray, shoot_points: np.ndarray):
    """Where the root meets the shoot: the shoot point closest to the root.

    The root is the stem continued below the clamp, so the junction between
    them is the one place the stem certainly passes through -- a far better
    anchor than the obvious alternatives, both of which are wrong here. The
    plant frame's origin lies on the *orbit axis* rather than on the plant,
    and the lowest shoot point is whatever hangs down furthest, which on
    plant_9 was a drooping leaf spread over 92 voxels of radius.

    Returned as an actual shoot point rather than a centroid of the root's
    upper slice. A root is a branching thing, so that centroid falls in the
    gap *between* its branches -- measured at 33 voxels from any root point,
    which is the same mid-air failure this is meant to prevent.
    """
    if root_points is None or len(root_points) == 0 or len(shoot_points) == 0:
        return None
    nearest = cKDTree(root_points).query(shoot_points)[0]
    return shoot_points[int(np.argmin(nearest))]


def shoot_paths(
    shoot_points: np.ndarray, base_radius: float, k_neighbors: int = 10,
    anchor: Optional[np.ndarray] = None,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Geodesic distance and a shortest-path tree over the whole shoot.

    Returns (graph, distance from the base, predecessor array).

    Over *all* shoot tissue -- leaf and stem together -- rather than each
    class on its own. Where the stem ends and a leaf begins is exactly the
    judgement the labels are worst at: measured on plant_9 the stem class is a
    tube of radius 4 voxels low down and a 15-voxel blob at the apex, because
    leaf bases and petioles land in it. Anything that trusts that boundary
    inherits the blob. The tree structure does not need it -- a plant is a
    trunk with branches whichever way its pixels were labelled.

    `anchor` is where the stem actually starts, normally the top of the root.
    Every path is traced back to there, so the trunk ends up running down the
    stem. Seeding on height instead gathered 287 points spread across 92
    voxels of radius -- a low leaf as much as the stem -- and the trunk left
    the cloud entirely on its way to them.
    """
    graph = leaf_graph(shoot_points, k_neighbors)

    if anchor is not None:
        near = np.nonzero(np.linalg.norm(shoot_points - anchor, axis=1) <= base_radius)[0]
        base = near if len(near) else np.array(
            [int(np.argmin(np.linalg.norm(shoot_points - anchor, axis=1)))])
    else:
        lowest = shoot_points[:, 2].min()
        base = np.nonzero(shoot_points[:, 2] <= lowest + base_radius)[0]
        if len(base) == 0:
            base = np.array([int(np.argmin(shoot_points[:, 2]))])

    # With min_only the call also reports which source won each node; the
    # predecessor chain is all we need, so the third return is discarded.
    distance, predecessor, _sources = dijkstra(
        graph, directed=False, indices=base, min_only=True, return_predecessors=True)
    return graph, distance, predecessor


def trace_to_base(predecessor: np.ndarray, node: int, limit: int = 1_000_000) -> List[int]:
    """The chain of nodes from `node` back down to the base."""
    path = [int(node)]
    seen = {int(node)}
    for _ in range(limit):
        nxt = int(predecessor[path[-1]])
        if nxt < 0 or nxt in seen:
            break
        path.append(nxt)
        seen.add(nxt)
    return path


def trunk_and_attachments(
    shoot_points: np.ndarray, predecessor: np.ndarray, distance: np.ndarray,
    tips: Sequence[int], merge_radius: float, smooth_tolerance: float = 0.0,
    num_stations: int = 40,
) -> Tuple[np.ndarray, List[int], List[List[int]]]:
    """Split the shoot into the trunk every leaf shares and each leaf's own tail.

    Returns (trunk centreline, one attachment node per tip, the path per tip).

    Trace every tip back to the base. Where a leaf's path comes within
    `merge_radius` of another leaf's path, the two have met: that is the fork
    where this leaf leaves the stem, so it is the attachment. Everything below
    the highest fork is trunk.

    Proximity rather than a shared node, which is what an earlier version
    tested. The stem is a *surface* in this cloud, near enough a tube, and two
    paths can run down opposite sides of it without ever using the same point
    -- measured on plant_9 that put one leaf's attachment at the plant base,
    105 voxels from the stem, because its descent never once coincided with
    another's.

    For the same reason the centreline is binned and averaged rather than
    taken from the nodes directly: the mean of two paths on opposite walls of
    a tube is its axis, while either path alone is its surface.

    Where the stem stops needs no threshold and no label. Above the highest
    fork there is nothing left for two leaves to share, so the trunk ends
    there by construction -- which holds for a plant with one leaf or forty.
    """
    paths = [trace_to_base(predecessor, int(t)) for t in tips]
    if not paths:
        return np.zeros((0, 3)), [], []

    attachments: List[int] = []
    for i, path in enumerate(paths):
        others = np.concatenate([p for j, p in enumerate(paths) if j != i]) \
            if len(paths) > 1 else np.zeros(0, np.int64)
        if len(others) == 0:
            attachments.append(int(path[-1]))
            continue
        tree = cKDTree(shoot_points[others])
        close = tree.query(shoot_points[path])[0] <= merge_radius
        first = int(np.argmax(close)) if close.any() else len(path) - 1
        attachments.append(int(path[first]))

    top = max(distance[a] for a in attachments)
    trunk_nodes = np.unique(np.concatenate(
        [np.array([n for n in path if distance[n] <= top], np.int64) for path in paths]))
    if len(trunk_nodes) < 2:
        return shoot_points[trunk_nodes], attachments, paths

    # Bin along the path from the base and average, so opposite walls of the
    # stem collapse onto its axis instead of zig-zagging between them.
    d = distance[trunk_nodes]
    edges = np.linspace(d.min(), d.max(), num_stations + 1)
    station = np.clip(np.digitize(d, edges) - 1, 0, num_stations - 1)
    centre = [shoot_points[trunk_nodes[station == s]].mean(axis=0)
              for s in range(num_stations) if (station == s).sum() >= 2]
    if len(centre) < 2:
        return shoot_points[trunk_nodes[np.argsort(d)]], attachments, paths

    # Smooth first, then reject: smoothing pulls each station toward its
    # neighbours, so a station that was on the cloud can be nudged off it, and
    # testing beforehand would pass exactly the nodes that end up in mid-air.
    centre = drop_stations_in_empty_space(np.array(centre), shoot_points)
    if len(centre) < 2:
        return shoot_points[trunk_nodes[np.argsort(d)]], attachments, paths

    # How far the curve may move is the stem's own radius, measured rather
    # than chosen. A station is the mean of the trunk nodes in its bin, and a
    # bin that caught more of one wall than the other is displaced by up to a
    # radius -- so deviations at that scale are sampling noise and anything
    # larger is a real bend. On the axis of a tube the distance to the nearest
    # surface point *is* the radius, which makes it free to measure.
    # Reject first, then fit: a spline pulled toward a station sitting in
    # mid-air bends the whole neighbourhood toward it.
    radius = float(np.median(cKDTree(shoot_points).query(centre)[0]))
    return (fit_smooth_curve(centre, tolerance=max(2.0 * radius, smooth_tolerance)),
            attachments, paths)


def drop_stations_in_empty_space(
    centreline: np.ndarray, tissue: np.ndarray, tolerance: float = 3.0,
) -> np.ndarray:
    """Remove centreline stations that sit where the plant is not.

    A station is the mean of the trunk nodes in its bin, which is the axis of
    the stem when those nodes surround it -- and somewhere in mid-air when
    they do not, as happens wherever paths have not converged yet. The result
    is a centreline that leaves the cloud, and a leaf whose midrib then reaches
    out to meet it across empty space.

    The test needs no fixed length: a station on the axis of a tube is about
    one stem radius from the nearest surface point, so stations are compared
    against the *median* gap along this plant's own stem and dropped when they
    exceed it several times over. Scale-free, and it cannot flag a fat stem
    merely for being fat.
    """
    if len(centreline) < 3:
        return centreline
    gap = cKDTree(tissue).query(centreline)[0]
    limit = max(np.median(gap) * tolerance, 1e-12)
    keep = gap <= limit
    return centreline[keep] if keep.sum() >= 2 else centreline


# --------------------------------------------------------------------------
# Midrib
# --------------------------------------------------------------------------


def distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest *node* of a polyline.

    Node-nearest rather than exact segment distance: the stem centreline is
    already sampled far more finely than a leaf is long, so the difference is
    below the voxel size and the KD-tree is a great deal faster.
    """
    return cKDTree(polyline).query(points)[0]


def _crown_base(leaf_points, stem_points, graph, k_neighbors):
    """Locate a rosette's crown, searching leaf AND stem tissue.

    Stem tissue has to be included. A rosette's crown is exactly what P4c
    labels "stem" -- it is thick and not lamina -- so searching leaf tissue
    alone looks for the one place the blades meet in a cloud where that place
    has been removed, leaving four disconnected strips.

    Nodes come back indexed into `leaf_points`, since that is what the depth
    field runs over.
    """
    from pose_estimator.plant_base import BaseRegion, find_base

    if len(stem_points) == 0:
        return find_base(leaf_points, graph)

    shoot = np.vstack([leaf_points, stem_points])
    base = find_base(shoot, leaf_graph(shoot, k_neighbors))
    if base is None:
        return None

    nodes = base.nodes[base.nodes < len(leaf_points)]
    if not len(nodes):
        # The crown is entirely stem-labelled -- which is the normal case, the
        # crown being the thick part -- so seed from the leaf tissue nearest
        # to it: the inner ends of the blades. Measured from the closest leaf
        # point rather than from the crown centre, because the blades start
        # wherever the crown stops and that distance is not known in advance.
        distance = np.linalg.norm(leaf_points - base.center, axis=1)
        radius = base.evidence["ball_fraction"] * base.evidence["plant_extent"]
        nodes = np.nonzero(distance <= distance.min() + radius)[0]
    if not len(nodes):
        return None
    return BaseRegion(nodes=nodes, center=base.center,
                      extremities=base.extremities, evidence=base.evidence)


def chord_midrib(points: np.ndarray, base: np.ndarray, tip: np.ndarray,
                 num_stations: int = 14, degree: int = 3,
                 min_bin: int = 3) -> np.ndarray:
    """A midrib built as a bend applied to the straight base-to-tip line.

    The previous construction chained centroids of geodesic shells. On a
    merged or cupped instance those centroids can sit anywhere -- shells far
    from the tip contain tissue from two different blades, so their centroid
    lands between them -- and the resulting polyline loops, doubles back, or
    crosses a neighbouring leaf. Nothing in it forces progress from base to
    tip.

    Here the chord *is* the parameter. Every point is projected onto it, so
    stations advance from base to tip by construction and cannot reverse. The
    only freedom is lateral: two offset functions of t, each forced to vanish
    at both ends, so the curve begins exactly at the base and ends exactly at
    the tip whatever the data does. Fitting them as low-order polynomials
    caps how much the curve can wriggle -- a leaf midrib is a gentle arc, and
    a cubic cannot tie a knot.

    Returns a polyline from `base` to `tip`.
    """
    base = np.asarray(base, float).reshape(3)
    tip = np.asarray(tip, float).reshape(3)
    axis = tip - base
    length = float(np.linalg.norm(axis))
    if length < 1e-12 or len(points) == 0:
        return np.vstack([base, tip])
    u = axis / length

    # two directions across the chord, so lateral offset has two components
    helper = np.array([0.0, 0.0, 1.0])
    if abs(u @ helper) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, helper); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)

    rel = points - base
    t = (rel @ u) / length
    inside = (t >= 0.0) & (t <= 1.0)
    if inside.sum() < min_bin:
        return np.vstack([base, tip])
    t, rel = t[inside], rel[inside]

    edges = np.linspace(0.0, 1.0, num_stations + 1)
    ts, o1, o2 = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b) if b < 1.0 else (t >= a) & (t <= b)
        if m.sum() < min_bin:
            continue
        ts.append(float(t[m].mean()))
        ts_rel = rel[m].mean(axis=0)
        o1.append(float(ts_rel @ e1))
        o2.append(float(ts_rel @ e2))
    if len(ts) < 2:
        return np.vstack([base, tip])

    # basis t^k (1 - t): every term is zero at t=0 and t=1, so the endpoints
    # are exact rather than fitted.
    ts = np.asarray(ts)
    design = np.stack([ts ** k * (1.0 - ts) for k in range(1, degree + 1)], axis=1)
    c1, *_ = np.linalg.lstsq(design, np.asarray(o1), rcond=None)
    c2, *_ = np.linalg.lstsq(design, np.asarray(o2), rcond=None)

    grid = np.linspace(0.0, 1.0, 40)
    basis = np.stack([grid ** k * (1.0 - grid) for k in range(1, degree + 1)], axis=1)
    return (base + np.outer(grid * length, u)
            + np.outer(basis @ c1, e1) + np.outer(basis @ c2, e2))


def clip_to_base(curve: np.ndarray, base_point: Optional[np.ndarray]) -> np.ndarray:
    """Trim a midrib so it starts at the base and runs nowhere past it.

    A midrib is extended past its blade along the shortest path back to the
    trunk, and that path can carry on through the base and up a neighbouring
    leaf -- which draws a curve across a leaf that has no tip of its own.
    Missing a leaf is acceptable; inventing one over it is not.

    The curve runs base -> tip, so its closest approach to the base is where
    it should begin and everything before that is overshoot. The base point
    is then prepended: the path descends the *surface* of the stem or crown,
    so it stops one radius short of the axis that insertion angle is measured
    against.
    """
    if base_point is None or not len(curve):
        return curve
    cut = int(np.argmin(np.linalg.norm(curve - base_point, axis=1)))
    return np.vstack([np.asarray(base_point, float).reshape(1, 3), curve[cut:]])


def _tips_furthest_from_stem(
    leaf_points: np.ndarray, owner: np.ndarray, stem_path: np.ndarray,
) -> Dict[int, int]:
    """Per instance, the index of the point that reaches furthest from the stem."""
    radial = distance_to_polyline(leaf_points, stem_path)
    tips: Dict[int, int] = {}
    for instance in range(int(owner.max()) + 1):
        member = np.nonzero(owner == instance)[0]
        if len(member):
            tips[instance] = int(member[int(np.argmax(radial[member]))])
    return tips


def tip_reach_shortfall(
    leaf_points: np.ndarray, owner: np.ndarray, tips: Sequence[int], stem_path: np.ndarray,
) -> List[float]:
    """How far short of its own leaf's reach each tip falls, per instance.

    The check this stage was missing. A leaf's tip should be the furthest part
    of it from the stem, so a positive shortfall means the tip was put
    somewhere the leaf does not end -- which is visible in Blender as a marker
    on a blade edge with point cloud carrying on past it.
    """
    if len(stem_path) < 1 or not len(owner):
        return []
    radial = distance_to_polyline(leaf_points, stem_path)
    out: List[float] = []
    for instance, tip in enumerate(tips):
        member = np.nonzero(owner == instance)[0]
        if len(member) and 0 <= int(tip) < len(radial):
            out.append(float(radial[member].max() - radial[int(tip)]))
    return out


def leaf_midrib(
    points: np.ndarray, distance_from_tip: np.ndarray,
    num_stations: int = 20, min_bin: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midrib for one leaf, from its base to its tip.

    Returns (polyline base->tip, tip position, attachment position).

    Stations are shells of *geodesic* distance from the tip, and each shell's
    centroid is the midrib estimate for that station -- the lamina is roughly
    symmetric about the midrib, so the centroid of a cross-section lies on the
    vein. Geodesic, not straight-line: on a curved blade a Euclidean shell cuts
    across the curve and merges parts of the leaf that are far apart along it,
    which bends the midrib through air the leaf does not occupy.

    The attachment is the far end of this curve rather than an input. That is
    the inversion that matters -- the old code was handed an attachment derived
    from stem adjacency, and any error in it hooked the whole midrib.
    """
    finite = np.isfinite(distance_from_tip)
    if finite.sum() < min_bin * 2:
        return np.zeros((0, 3)), points[0], points[0]

    points, distance = points[finite], distance_from_tip[finite]
    tip = points[int(np.argmin(distance))]

    edges = np.linspace(0.0, float(distance.max()), num_stations + 1)
    station = np.clip(np.digitize(distance, edges) - 1, 0, num_stations - 1)

    curve = [points[station == i].mean(axis=0)
             for i in range(num_stations) if (station == i).sum() >= min_bin]
    if len(curve) < 2:
        return np.zeros((0, 3)), tip, points[int(np.argmax(distance))]

    # Built tip-first; reverse so index 0 is the attachment, which is the
    # contract P6 reads (s = 0 at the petiole).
    curve = _smooth(np.array(curve))[::-1]
    return curve, tip, curve[0]


def build_from_labels(
    points: np.ndarray,
    labels: np.ndarray,
    class_order: Sequence[str],
    voxel: float,
    contact_voxels: float = 3.0,
    min_leaf_points: int = 150,
    min_tip_depth_voxels: float = 8.0,
    min_persistence_ratio: float = 0.5,
    k_neighbors: int = 10,
    architecture: str = "caulescent",
) -> LabelledStructure:
    """Full structure from a labelled cloud, in the plant frame.

    `architecture` is passed through to the instancing: see `instance_by_tips`.
    A rosette reaches this function with no stem tissue and no root tissue --
    its roots are below the clamp and outside the P2 plant mask -- so the
    anchor for the shoot tree comes from the derived crown instead.
    """
    leaf_ids_in_order = [i for i, n in enumerate(class_order) if "leaf" in n]
    stem_ids = [i for i, n in enumerate(class_order) if n in ("stem", "petiole", "branch")]
    root_ids = [i for i, n in enumerate(class_order) if n == "root"]

    leaf_points = points[np.isin(labels, leaf_ids_in_order)]
    stem_points = points[np.isin(labels, stem_ids)]
    root_points = points[np.isin(labels, root_ids)] if root_ids else None

    instancing = instance_by_tips(
        leaf_points, stem_points,
        contact_radius=voxel * contact_voxels,
        k_neighbors=k_neighbors,
        min_points=min_leaf_points,
        min_tip_depth=voxel * min_tip_depth_voxels,
        min_persistence_ratio=min_persistence_ratio,
        architecture=architecture,
    )

    # --- the shoot as one tree, so a midrib can run past the blade to the stem ---
    leaf_index = np.nonzero(np.isin(labels, leaf_ids_in_order))[0]
    shoot_index = np.nonzero(np.isin(labels, leaf_ids_in_order + stem_ids))[0]
    shoot = points[shoot_index]
    to_shoot = np.full(len(points), -1, np.int64)
    to_shoot[shoot_index] = np.arange(len(shoot_index))

    # The root is the stem carried on below the clamp, so its top is the one
    # place the stem is certain to pass through -- a far better anchor than
    # "lowest point", which is whatever hangs down furthest.
    anchor = root_anchor(root_points, shoot)
    if anchor is None and instancing.base is not None:
        # No root and no stem: a rosette. The crown is where the leaves meet,
        # which is the same thing the root junction gives on a stemmed plant.
        anchor = shoot[int(cKDTree(shoot).query(instancing.base.center)[1])]
    graph, base_distance, predecessor = shoot_paths(
        shoot, base_radius=voxel * 8.0, k_neighbors=k_neighbors, anchor=anchor)

    shoot_tips = [int(to_shoot[leaf_index[int(t)]]) for t in instancing.accepted_tips]
    stem_path, attach_nodes, paths = trunk_and_attachments(
        shoot, predecessor, base_distance, shoot_tips, merge_radius=voxel * 4.0,
        smooth_tolerance=voxel * 1.5)
    if len(stem_path) < 2 and len(stem_points):
        stem_path = fit_stem_path(stem_points)

    # A rosette's trunk is a point, not a curve. Left alone, the trunk tracer
    # returns whatever few nodes the paths happened to share inside the crown
    # -- on runs/thistle1 a 3-node zigzag spanning 4% of the plant, which is
    # noise, and drawn as a tube in Blender it reads as an elbow of pipe that
    # no thistle has.
    rosette = architecture == "rosette"
    if rosette and instancing.base is not None:
        stem_path = np.asarray(instancing.base.center, float).reshape(1, 3)

    # Persistence says how many leaves there are, and does that well. Where the
    # tip *is* is a different question, and geodesic depth answers it badly: on
    # a sheet the point furthest from the contact region measured through the
    # tissue is often a lateral corner rather than the pointy end, especially
    # when the petiole carries the stem label so the leaf touches the stem at a
    # corner of the blade. Measured on plant_9 that left five of seven tips
    # short of their own leaf's reach, one by 45 voxels.
    #
    # A leaf's tip is simply the part of it that gets furthest from the stem,
    # so once the trunk is known each instance re-picks its own tip that way.
    if len(stem_path) >= 1 and (instancing.owner >= 0).any():
        refined = _tips_furthest_from_stem(leaf_points, instancing.owner, stem_path)
        if refined:
            instancing.accepted_tips = np.array(
                [refined.get(i, int(instancing.accepted_tips[i]))
                 for i in range(len(instancing.accepted_tips))], np.int64)
            shoot_tips = [int(to_shoot[leaf_index[int(t)]])
                          for t in instancing.accepted_tips]
            # Re-solve the topology from the corrected tips, so each leaf's
            # path down to its fork starts from the right end of the leaf.
            new_path, attach_nodes, paths = trunk_and_attachments(
                shoot, predecessor, base_distance, shoot_tips, merge_radius=voxel * 4.0,
                smooth_tolerance=voxel * 1.5)
            if not rosette:
                stem_path = new_path

    # Carry the stem down to where the root begins, so the two organs meet
    # instead of stopping a gap apart.
    if anchor is not None and len(stem_path) > 1:
        if np.linalg.norm(stem_path[0] - anchor) > voxel:
            stem_path = np.vstack([anchor, stem_path])

    axes, tips, attachments = [], [], []
    num = int(instancing.owner.max()) + 1 if (instancing.owner >= 0).any() else 0
    for instance in range(num):
        member = np.nonzero(instancing.owner == instance)[0]
        if instance >= len(paths):
            continue
        # Geodesic distance measured from *this* instance's tip, over this
        # instance only. The field on `instancing` was measured from the tip
        # persistence proposed, which the refinement above may have moved.
        member_shoot = to_shoot[leaf_index[member]]
        member_shoot = member_shoot[member_shoot >= 0]
        local = np.nonzero(member_shoot == shoot_tips[instance])[0]
        if len(local):
            sub = graph[member_shoot][:, member_shoot]
            from_tip = dijkstra(sub, directed=False, indices=[int(local[0])], min_only=True)
        else:
            from_tip = instancing.distance_from_tip[member]

        blade, tip, _base = leaf_midrib(leaf_points[member], from_tip)
        if len(blade) < 2:
            continue

        # Carry the curve on past the lamina to the fork. The petiole is thin
        # and usually carries the stem label, so it is not in the instance at
        # all, and binning cannot recover it either -- a one-point-wide chain
        # puts one or two points in a shell and `min_bin` throws it away. That
        # is why midribs were stopping on the edge of the blade. The path is
        # already a centreline, so it is appended rather than re-averaged.
        path = paths[instance]
        stop = path.index(attach_nodes[instance]) + 1
        own = set(to_shoot[leaf_index[member]].tolist())
        extension = [shoot[n] for n in path[:stop] if n not in own]
        curve = np.array(list(reversed(extension)) + list(blade)) if extension else blade

        # End at the base, and nowhere past it. A rosette's base is a single
        # crown node, and this step used to be guarded on there being a stem
        # *line*, so on a thistle no midrib ever met the crown.
        if len(stem_path) == 1:
            base_point = stem_path[0]
        elif len(stem_path) > 1:
            base_point = stem_path[int(np.argmin(np.linalg.norm(stem_path - curve[0], axis=1)))]
        else:
            base_point = None
        curve = clip_to_base(curve, base_point)

        # Same rule for the midrib: its stations are shell centroids, so they
        # wander by about the blade's own half-thickness, and the sharpest
        # kink sits where the blade centroids meet the raw petiole path.
        # Rebuild the curve as a bend on the base-to-tip chord. The station
        # chain above finds the right *ends*; what it cannot guarantee is that
        # everything between them advances, and on a merged instance it does
        # not. Projecting onto the chord makes progress structural.
        if base_point is not None and len(curve) > 1:
            chorded = chord_midrib(leaf_points[member], base_point, curve[-1])
            if len(chorded) > 1:
                curve = chorded

        if len(curve) > 3:
            thickness = float(np.median(cKDTree(leaf_points[member]).query(curve)[0]))
            curve = fit_smooth_curve(curve, tolerance=max(2.0 * thickness, voxel))

        # Re-attach after smoothing. `clip_to_base` put the base on the front,
        # but a smoothing spline does not interpolate its endpoints, so it
        # then pulled the curve off again -- measured on thistle1, midribs
        # starting 3.3 to 11.9 voxels away from a crown they were supposed to
        # begin at. Every tip has to reach the base or the skeleton is not
        # connected.
        if base_point is not None and len(curve):
            curve = np.vstack([np.asarray(base_point, float).reshape(1, 3), curve[1:]])
        axes.append(curve)
        tips.append(tip)
        attachments.append(shoot[attach_nodes[instance]])

    return LabelledStructure(
        stem_path=stem_path, leaf_ids=instancing.owner, leaf_points=leaf_points,
        attachments=attachments, tips=tips, axes=axes,
        root_points=root_points, instancing=instancing,
    )
