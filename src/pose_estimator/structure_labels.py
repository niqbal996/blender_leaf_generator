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


# Above this elevation of the crown-to-tip line, a leaf is treated as one of
# the upright "heart" leaves at the centre of a rosette. Measured on
# thistle3, whose leaves sit at -22, -18, -18, -14, +9, +55, +56 and +80
# degrees: the split falls in a 46-degree empty band, so the exact value is
# not delicate. Halfway between flat and vertical is the natural place for it.
HEART_LEAF_ELEVATION = 45.0

# Persistence above which a maximum is a leaf beyond argument, so the
# leaf/noise cut is never made among them. This was the whole rule once, as a
# fixed cut; it survives as the ceiling on where the adaptive cut may fall.
# Measured: thistle3's leaves score 1.00-0.37 and thistle1's 1.00-0.35, while
# the noise on both begins at 0.17 and below.
CERTAIN_TIP_PERSISTENCE = 0.5


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
    heart: Optional[np.ndarray] = None  # top of the stem, where an upright plant's leaves start
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
    crown: Optional[np.ndarray] = None   # foot of the stem, just above the root
    heart: Optional[np.ndarray] = None   # top of the stem, where the leaves start

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
    min_persistence_ratio: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the maxima into real leaf tips and bumps on a blade.

    Returns (all candidates above `min_depth`, the accepted subset).

    A maximum is its own leaf when it survives down through most of its own
    depth -- `(peak - saddle) / peak`, a fraction of the peak so a small leaf
    is judged by the same rule as a large one.

    **Where the line falls is read off the plant, not fixed.** A constant was
    tried first and is a knife-edge. Ranking thistle3's candidates by that
    ratio gives

        1.00 0.96 0.90 0.88 0.73 0.57 | 0.48 0.37 | 0.10 0.09 0.08 0.07 ...

    -- real leaves down to 0.37, then nothing until the bumps at 0.10 and
    below. The old fixed 0.5 cut lands *inside* the run of real leaves and
    discards two of them, one by 0.02: a leaf fused to its neighbour partway
    along scores lower than a free-standing one, and how much lower depends
    on the specimen, not on anything universal.

    So cut at the widest gap in the sorted ratios instead. Leaves and bumps
    separate by a chasm (0.27 here, against 0.02-0.16 between neighbouring
    leaves), and the gap is a property of this plant's own distribution, so
    it needs no refitting across sizes or species. Pass
    `min_persistence_ratio` to override with a fixed cut.
    """
    ranked = sorted((r for r in records if r[1] >= min_depth), key=lambda r: -r[1])
    candidates = [int(index) for index, _, _ in ranked]
    if not candidates:
        return np.array([], np.int64), np.array([], np.int64)

    ratios = np.array([(peak - saddle) / max(peak, 1e-12) for _, peak, saddle in ranked])
    index = np.array([int(i) for i, _, _ in ranked], np.int64)

    if min_persistence_ratio is not None:
        return np.array(candidates, np.int64), index[ratios > min_persistence_ratio]

    order = np.argsort(-ratios)
    sorted_ratios = ratios[order]

    # Anything this persistent is a leaf, full stop, and the cut is never
    # made among them. Below it, the widest step down the ranking separates
    # the remaining leaves from bumps on a leaf.
    #
    # Cutting at the widest gap over the *whole* ranking was tried and does
    # not generalise. It reads thistle3 correctly (8 leaves; the leaves run
    # 1.00-0.37 and the noise starts at 0.10, a 0.27 chasm) and misreads
    # thistle1 badly (2 leaves): that plant's leaves touch each other more,
    # which raises their saddles and spreads their scores to 1.00 0.93 0.69
    # 0.55 0.52 0.35, so the widest gap in the ranking falls *between two
    # real leaves* (0.93 -> 0.69) rather than at the leaf/noise boundary
    # (0.35 -> 0.17). The old fixed 0.5 read thistle1 better than the gap
    # rule did, which is what this keeps.
    #
    # The virtual zero at the end lets "they are all leaves" be expressed:
    # four identical leaves have no gap between them, and without a final
    # step down to nothing the widest gap would be a tie and only the first
    # would survive.
    certain = int((sorted_ratios > CERTAIN_TIP_PERSISTENCE).sum())
    remainder = sorted_ratios[certain:]
    if len(remainder) < 2:
        cut = max(certain, 1)
    else:
        gaps = np.diff(np.concatenate([remainder, [0.0]]).astype(float)) * -1.0
        cut = certain + int(np.argmax(gaps)) + 1
    return np.array(candidates, np.int64), index[order[:cut]]


def own_by_subtree(graph: csr_matrix, tips: np.ndarray, contact: np.ndarray,
                  n_points: int) -> np.ndarray:
    """Which leaf each point is on, following the plant's own branching.

    The rule this replaces was "whichever tip is nearest across the surface",
    and it splits two leaves at the *midpoint between their tips* rather than
    where they separate. That is only the same place when both leaves are the
    same length. Measured on thistle3, whose tips sit 53 to 163 voxels from
    the crown: the shortest leaf's territory reached 55 voxels up every
    neighbour's stalk, so one instance held 29% of all leaf tissue and wrapped
    357 degrees around the crown, against 24-43 degrees for a real leaf. The
    midribs fitted to those instances inherited the borrowed stalks.

    Blocking travel through the contact points was meant to stop exactly this
    and cannot: on thistle3 that is 198 nodes out of 54,670, and the tissue
    around a crown is a continuous sheet, so a front simply walks around them.

    So ask a structural question instead of a metric one. Grow a
    shortest-path forest out from the base and look at what lies *beyond*
    each point:

      exactly one tip beyond it   -> it is on that leaf
      two or more tips beyond it  -> it is crown or trunk, shared, no leaf
      no tip beyond it            -> a dead end; it belongs where its parent does

    Returns (owner, shared). `shared` is the crown/trunk tissue -- the points
    with two or more leaves beyond them, plus the base itself. It is reported
    separately because "no leaf owns this" and "no leaf reaches this" need
    different handling downstream: the first is an anatomical fact about a
    crown and must stay unclaimed, the second is a fragment that may deserve
    an instance of its own.

    Boundaries then land where the stalks actually diverge, whatever the leaf
    lengths, and nothing here carries a length, a count or a tuned constant --
    it is the branching itself doing the deciding.
    """
    distance, predecessor, _ = dijkstra(graph, directed=False, indices=contact,
                                        min_only=True, return_predecessors=True)
    reachable = np.isfinite(distance)
    outward = np.argsort(np.where(reachable, distance, -np.inf))

    trunk = np.zeros(n_points, bool)          # two or more leaves genuinely beyond
    beyond = np.zeros(n_points, np.int64)     # accepted tips past this point
    which = np.full(n_points, -1, np.int64)   # the one tip, while there is one
    for rank, tip in enumerate(tips):
        beyond[int(tip)] += 1
        which[int(tip)] = rank

    for node in outward[::-1]:                # deepest first: gather toward the base
        parent = predecessor[node]
        if parent < 0 or not reachable[node] or beyond[node] == 0:
            continue
        if beyond[parent] == 0:
            which[parent] = which[node]
        elif which[parent] != which[node]:
            which[parent] = -1
        beyond[parent] += beyond[node]

    owner = np.full(n_points, -1, np.int64)
    shared = np.zeros(n_points, bool)
    for node in outward:                      # base first: hand ownership outward
        if not reachable[node]:
            continue
        if beyond[node] >= 2:
            shared[node] = True
            trunk[node] = True
        elif beyond[node] == 1:
            owner[node] = which[node]
        elif predecessor[node] >= 0:
            owner[node] = owner[predecessor[node]]
            shared[node] = shared[predecessor[node]]
        else:
            shared[node] = True               # a base node with nothing beyond it
    # `trunk` is the tissue that is shared in its own right -- two or more
    # leaves genuinely lie beyond it. The rest of `shared` merely inherited the
    # label from a parent, which is a different thing: it is a dead end hanging
    # off the trunk, and how much of the plant it amounts to depends on how
    # large the seed region was.
    return owner, shared, trunk


def distance_to_own_tip(graph: csr_matrix, tips: np.ndarray, owner: np.ndarray,
                        n_points: int) -> np.ndarray:
    """Geodesic distance from each point to the tip of the leaf it is on.

    Separate from ownership now that the two are decided differently: the
    midrib fit needs how far a point is along *its own* leaf, which is no
    longer the distance to the nearest tip.
    """
    out = np.full(n_points, np.inf)
    for rank, tip in enumerate(tips):
        mine = owner == rank
        if not mine.any():
            continue
        reach = dijkstra(graph, directed=False, indices=[int(tip)], min_only=True)
        out[mine] = reach[mine]
    return out


def keep_largest_blob(graph: csr_matrix, owner: np.ndarray,
                      spacing_factor: float = 3.0) -> Tuple[np.ndarray, List[dict]]:
    """Reduce each leaf to its own connected blob, releasing the rest.

    Ownership is decided by walking the plant's branching, and where two
    leaves touch, the branching itself is wrong: the surface joins them, so a
    stretch of one leaf can hang off the other's subtree. What that produces
    is recognisable -- an instance made of one large blob, its real leaf, plus
    a small detached blob sitting over a neighbour, with a clear gap between
    the two because the tissue joining them near the crown is shared and owned
    by neither.

    Being detached is the evidence. A leaf is one connected piece of plant, so
    a component that does not touch the instance's main body is not part of
    that leaf whatever the graph said. The minority is released rather than
    reassigned: it is nearly always the neighbour's, but "nearly always" is
    not a thing to encode, and released tissue is picked up by the midrib fit
    of whichever leaf actually runs through it.

    Connectivity has to be judged by distance, not by the kNN graph: that
    graph always joins the ten nearest neighbours however far away they are,
    so it bridges the very gap this is looking for, and components taken
    straight off it never split. Edges longer than `spacing_factor` times the
    instance's own median edge are cut first -- a ratio against the plant's
    own sampling, so it carries no length and does not care how big the
    specimen is. Swept on thistle3 the split is identical from 1.5 to 4.0 and
    stops firing at 4.5, so the default sits in the middle of a wide plateau
    rather than on an edge.
    """
    released: List[dict] = []
    for instance in range(int(owner.max()) + 1 if (owner >= 0).any() else 0):
        member = np.nonzero(owner == instance)[0]
        if len(member) < 2:
            continue
        sub = graph[member][:, member].tocoo()
        lengths = sub.data[sub.data > 0]
        if not len(lengths):
            continue
        limit = spacing_factor * float(np.median(lengths))
        near = sub.data <= limit
        pruned = csr_matrix((sub.data[near], (sub.row[near], sub.col[near])),
                            shape=sub.shape)
        count, component = connected_components(pruned, directed=False)
        if count <= 1:
            continue
        sizes = np.bincount(component)
        main = int(np.argmax(sizes))
        stray = member[component != main]
        owner[stray] = -1
        released.append({"instance": instance, "kept": int(sizes[main]),
                         "released": int(len(stray)), "blobs": int(count)})
    return owner, released


def claim_orphans(
    graph: csr_matrix, owner: np.ndarray, distance: np.ndarray,
    depth: np.ndarray, min_points: int, min_depth: float,
    exclude: Optional[np.ndarray] = None,
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
    # `exclude` is the crown/trunk: tissue that genuinely belongs to no single
    # leaf because several pass through it. Left in, it is one big connected
    # component that clears both bars and becomes a "leaf" wrapping the whole
    # plant -- measured on thistle3 as a 5,952-point instance spanning 357
    # degrees, which is the crown wearing a leaf's colour.
    free = owner < 0
    if exclude is not None:
        free &= ~exclude
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
    min_persistence_ratio: Optional[float] = None,
    architecture: str = "caulescent",
    heart: Optional[np.ndarray] = None,
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
    # Which architecture this is comes from --architecture, not from a guess.
    # It is known when the capture is made, and an earlier version that
    # inferred it flipped thistle1's verdict purely because P4c had started
    # labelling the crown "stem".
    #
    # Seeding the upright path from the crown as well was tried and is wrong:
    # for an upright plant "depth" has to mean distance *out from the stem*,
    # not distance from the base, or a leaf low on the stem measures as
    # shallow and is rejected as a bump. On the two-leaf test plant that
    # dropped the lower blade and returned one tip for two leaves.
    base = None
    if architecture == "upright" and heart is not None:
        # An upright plant: crown at the foot of the stem, heart at its top,
        # leaves radiating from the heart. Depth is measured from the heart,
        # which is the same statement a rosette makes about its crown -- and
        # the reason the old seeding failed here was that it started from
        # every leaf point touching the stem instead, 2,930 of them on weed_3,
        # which leaves the tree too shallow for ownership to mean anything.
        reach = np.linalg.norm(leaf_points - np.asarray(heart, float), axis=1)
        contact = np.nonzero(reach <= contact_radius)[0]
        if not len(contact):
            # No blade within touching distance: take the nearest one, so the
            # depth field still starts at the top of the stem, not nowhere.
            contact = np.array([int(np.argmin(reach))], np.int64)
        depth = dijkstra(graph, directed=False, indices=contact, min_only=True)
    else:
        heart = None

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
    elif heart is None:
        # "caulescent": leaves distributed along the stem, so depth is measured
        # out from the stem itself. Distinct from "upright", where they all
        # start at the top of it -- seeding an along-the-stem plant from the
        # heart makes its lower leaves look shallow and merges them away.
        depth, contact = depth_from_stem(leaf_points, stem_points, graph, contact_radius)

    records = tip_persistence(graph, depth)
    candidates, tips = select_tips(records, min_tip_depth, min_persistence_ratio)
    owner, shared, trunk = own_by_subtree(graph, tips, contact, n)

    # Whatever the tree could not attribute goes to its nearest tip. Dead-end
    # tissue hanging straight off a seed node has no tip beyond it and no
    # owning parent to inherit from, and how much of the plant that is depends
    # entirely on how large the seed region is: a rosette seeds from a crown of
    # a dozen nodes and hardly any tissue lands here, an upright plant seeds
    # from every leaf point touching the stem -- 2,930 of them on weed_3 --
    # which makes the tree shallow and leaves 93% of the plant unowned, four
    # instances holding 2,008 points of 27,595 and every midrib of length zero.
    #
    # Crown and trunk tissue is *not* filled in: it has two or more leaves
    # beyond it and genuinely belongs to none of them, which is the
    # distinction the subtree walk exists to make.
    # Only on the upright path. A rosette seeds from a crown of a dozen nodes,
    # so its tree is deep, almost nothing lands here, and what does is crown
    # tissue that should stay unowned -- filling it in moved 1,100 points into
    # thistle3's leaves that the subtree walk had correctly left out.
    # Only where the seed region is large enough to flatten the tree, which is
    # the along-the-stem case. A rosette seeds from a crown and an upright
    # plant from its heart; both are small, and their trees are deep enough
    # that what lands here is trunk tissue that should stay unowned.
    unresolved = (owner < 0) & ~trunk if architecture == "caulescent" else np.zeros(n, bool)
    if unresolved.any() and len(tips):
        reach = np.full((len(tips), n), np.inf)
        for rank, tip in enumerate(tips):
            reach[rank] = dijkstra(graph, directed=False, indices=[int(tip)], min_only=True)
        nearest = np.argmin(reach, axis=0)
        finite = np.isfinite(reach[nearest, np.arange(n)])
        owner[unresolved & finite] = nearest[unresolved & finite]

    distance_from_tip = distance_to_own_tip(graph, tips, owner, n)
    owner, distance_from_tip, orphan_tips = claim_orphans(
        graph, owner, distance_from_tip, depth, min_points, min_tip_depth,
        exclude=shared)

    # Second pass, after the instances exist: a leaf is one connected piece,
    # so anything of an instance that is detached from its main body was
    # taken from a neighbour across a contact.
    owner, _released = keep_largest_blob(graph, owner)
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
        heart=None if heart is None else np.asarray(heart, float),
    )


# --------------------------------------------------------------------------
# Trunk and attachments, from the shoot's own tree structure
# --------------------------------------------------------------------------


def _largest_cluster(points: np.ndarray, radius: float) -> Optional[np.ndarray]:
    """The biggest blob of `points`, joining anything within `radius`.

    A plain connected-components pass over a radius graph. Used to separate a
    plant's real root from tissue that merely carries the root label -- dirt
    on the table, or shadow under the jaws -- which is common enough that the
    crown cannot be derived from the label alone.
    """
    if points is None or len(points) == 0:
        return points
    if len(points) == 1:
        return points
    from scipy.sparse import coo_matrix

    tree = cKDTree(points)
    pairs = np.array(list(tree.query_pairs(radius)), dtype=np.int64)
    if not len(pairs):
        return points
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(points), len(points))).tocsr()
    count, label = connected_components(graph, directed=False)
    if count <= 1:
        return points
    return points[label == int(np.argmax(np.bincount(label)))]


def crown_from_root(root_points, foliage_points, voxel: float,
                    top_fraction: float = 0.05, bin_voxels: float = 6.0):
    """The crown: the lowest foliage sitting directly above the root.

    A rosette has no stem to trace, but it does have a root, and the jaws grip
    exactly where the root ends and the shoot begins. That band is *occluded*
    -- the tool is in front of it from every angle -- so the junction itself is
    never reconstructed and cannot be measured directly. What can be measured
    is the tissue on either side of the hole, and the first foliage above it is
    the closest thing to the crown the data contains.

    Measured on thistle3, in a 6-voxel column about the root's own axis:

        z 0.150-0.330   root                <- root body
        z 0.330-0.360   nothing at all      <- the pliers
        z 0.360-0.420   stem, then leaf     <- foliage bottom

    Three details, each of which the data forced:

    * **The column matters.** A rosette's outer leaves droop well below the
      crown -- thistle3 has leaf tissue down at z=0.087, far under the root's
      top -- so "the lowest foliage" without a horizontal restriction returns a
      blade tip off to one side. Confining the search to a column about the
      root's axis is what makes "above the root" mean above *the root*.

    * **Unassigned tissue is excluded** by the caller passing only foliage that
      P5 actually attached to a leaf, plus stem. The tissue P4c labels leaf but
      P5 attaches to nothing sits *inside* the root band on thistle3 (z
      0.150-0.330, interleaved with root), so leaving it in lets it spoof a
      foliage bottom 0.2 units too low.

    * **The root's top is a quantile**, not its highest point: thistle3 has a
      stray root point at z=0.494, inside the band the jaws occlude, and a max
      would put the floor above the foliage it is meant to sit under.

    Neither knob is delicate. Across bin radii of 4-12 voxels and
    `top_fraction` of 0.05-0.10 the crown moved from z=0.386 to z=0.368 and
    settled on the same point, against a plant 1.2 units tall.

    Returns None when there is no root to stand on, or nothing above it in the
    column, leaving the caller's own fallback in charge.
    """
    if root_points is None or len(root_points) == 0 or len(foliage_points) == 0:
        return None

    # The root's *main mass*, not every point wearing the root label. Dirt on
    # the turntable classifies as root, and on thistle1 that put 189 stray
    # root points up at z 0.85-0.94 among the leaves, against a real root
    # body of 0.00-0.43. This function walks to the top of the root by
    # design, so those strays became the top and the crown followed them into
    # the foliage. A quantile cannot save it -- they were 2% of the label but
    # a third of everything above the cut.
    #
    # Keeping the largest cluster is enough, and needs no threshold beyond
    # the sampling the cloud already has: real root tissue is contiguous, a
    # cloud of misread dirt in the canopy is not connected to it.
    root_points = _largest_cluster(root_points, voxel * 4.0)
    if root_points is None or len(root_points) == 0:
        return None

    heights = root_points[:, 2]
    root_top = float(np.quantile(heights, 1.0 - top_fraction))
    upper = root_points[heights >= root_top]
    if not len(upper):
        return None
    centre_x, centre_y = upper[:, 0].mean(), upper[:, 1].mean()

    across = np.hypot(foliage_points[:, 0] - centre_x, foliage_points[:, 1] - centre_y)
    above = (across <= voxel * bin_voxels) & (foliage_points[:, 2] > root_top)
    if not above.any():
        return None
    candidates = foliage_points[above]
    return candidates[int(np.argmin(candidates[:, 2]))]


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


def _base_in_main_body(points, graph, find_base):
    """`find_base`, restricted to the largest connected piece of tissue.

    A reconstructed plant is usually one connected surface but not always:
    stray specks survive the carve, and organ labels can cut a blade off from
    the rest. `find_base` walks geodesics, and on a disconnected graph those
    are infinite between pieces -- the extremity search then has no way to
    prefer the plant over a speck, and the betweenness it computes is over
    whatever piece it happened to land in.

    That is not hypothetical. On weed_3 the leaf tissue came out as three
    components of 27,542, 37 and 16 points, the crown was located inside the
    37-point fragment, and the depth field seeded from it reached 37 points of
    27,595. Every leaf was then unassigned and P5 reported no leaves at all,
    with nothing upstream failing.

    The largest piece is the plant. Deciding that needs no threshold -- it is
    a comparison -- and the smaller pieces keep their labels, they simply do
    not get to say where the plant is rooted.
    """
    count, label = connected_components(graph, directed=False)
    if count <= 1:
        return find_base(points, graph)

    main = int(np.argmax(np.bincount(label)))
    index = np.nonzero(label == main)[0]
    base = find_base(points[index], graph[index][:, index])
    if base is None:
        return None
    # Re-index back into the full point array the caller knows about.
    base.nodes = index[base.nodes]
    base.extremities = index[base.extremities]
    return base


def stem_line_from_root(root_points, stem_points, voxel: float):
    """The stem line of an upright plant: crown at its foot, heart at its top.

    Stated plainly, and this is the whole rule:

    * Walk up out of the root cloud until you reach the stem cloud. The first
      stem tissue you meet is the **crown** -- the start of the stem line.
      That walk is the same one a rosette makes (`crown_from_root`); the only
      difference is that here it lands on stem rather than on a blade.
    * The highest stem point along z is the **heart** -- the end of the stem
      line, and where an upright plant's leaves start.

    Both ends are *measured off the labelled cloud*, so neither can double
    back the way a traced trunk did on weed_3 (z -0.055 -> 0.160 -> 0.088).

    Outliers are dropped by keeping the largest connected blob of stem, for
    the same reason `crown_from_root` does it to the root: stray specks
    wearing the stem label sit up among the leaves, and a bare `max(z)` would
    follow them there.

    Returns (crown, heart), or None when there is no stem cloud to walk.
    """
    if stem_points is None or len(stem_points) == 0:
        return None
    body = _largest_cluster(stem_points, voxel * 4.0)
    if body is None or len(body) == 0:
        return None

    heart = body[int(np.argmax(body[:, 2]))]

    # The crown, by the walk up out of the root. Without a root there is
    # nothing to walk out of, so the bottom of the stem stands in for it --
    # the same point, just without the evidence that it sits above a root.
    crown = crown_from_root(root_points, body, voxel)
    if crown is None:
        crown = body[int(np.argmin(body[:, 2]))]
    return np.asarray(crown, float), np.asarray(heart, float)


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
        return _base_in_main_body(leaf_points, graph, find_base)

    shoot = np.vstack([leaf_points, stem_points])
    base = _base_in_main_body(shoot, leaf_graph(shoot, k_neighbors), find_base)
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
                 spare: Optional[np.ndarray] = None, heart: bool = False,
                 num_stations: int = 14, degree: int = 3,
                 min_bin: int = 3) -> np.ndarray:
    """A midrib built as a bend applied to the straight base-to-tip line.

    Every point is projected onto the chord, so stations advance from base to
    tip and cannot reverse. The only freedom is lateral, and it is forced to
    vanish at both ends, so the curve begins exactly at the base and ends
    exactly at the tip whatever the data does.

    `heart` switches which of two situations this leaf is in, and it matters
    because the right answer is opposite in each:

    * **A normal leaf** has tissue along most of its length. Its own points
      describe its shape, so the bend is fitted across the whole chord, with
      shared tissue standing in for the few stations it lacks.
    * **A heart leaf** -- one of the small upright leaves at the centre of a
      rosette -- is the straight chord, end to end, and none of its points
      are consulted. Seen from above the cloud closes over the middle of the
      plant slightly higher than those leaves attach, so about half their
      length is not reconstructed at all; and what does survive is a
      one-sided sliver rather than a blade seen from both edges. A station's
      centre is the midpoint of the tissue in it, so lopsided tissue puts the
      station off the vein, and the curve waves from station to station. The
      leaves this applies to are short and near-upright, so the straight line
      between crown and tip is already a good midrib for them -- better than
      one bent toward whichever side happened to be reconstructed.

    Applying the heart-leaf rule to every leaf was tried and is wrong: a
    well-supported leaf loses the part of its shape that its own points were
    perfectly able to describe.
    """
    base = np.asarray(base, float).reshape(3)
    tip = np.asarray(tip, float).reshape(3)
    axis = tip - base
    length = float(np.linalg.norm(axis))
    if length < 1e-12 or len(points) == 0:
        return np.vstack([base, tip])
    u = axis / length

    if heart:
        # Straight, and deliberately without looking at the points: see above.
        return base + np.outer(np.linspace(0.0, 1.0, 40), axis)

    def project(cloud):
        rel = np.asarray(cloud, float) - base
        t = (rel @ u) / length
        inside = (t >= 0.0) & (t <= 1.0)
        return t[inside], rel[inside]

    t, rel = project(points)
    if len(t) < min_bin:
        return np.vstack([base, tip])
    t_spare, rel_spare = project(spare) if spare is not None and len(spare) else (
        np.zeros(0), np.zeros((0, 3)))

    # A heart leaf bends only over the stretch it actually occupies; a normal
    # leaf over the whole chord. `lift` is the power the offset basis starts
    # at: 2 makes the curve leave the chord tangentially at the join, which
    # only matters when there is a join.
    origin_t, lift, span = 0.0, 1, 1.0

    helper = np.array([0.0, 0.0, 1.0])
    if abs(u @ helper) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, helper); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)

    edges = np.linspace(origin_t, 1.0, num_stations + 1)
    ts, o1, o2 = [], [], []
    own_stations = 0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b) if b < 1.0 else (t >= a) & (t <= b)
        if m.sum() >= min_bin:
            station_t, centre = float(t[m].mean()), rel[m].mean(axis=0)
            own_stations += 1
        elif len(t_spare):
            # Shared tissue nearest this leaf's own line -- its petiole --
            # standing in where the instance has nothing. Never for a heart
            # leaf: there the shared tissue is what pulls the curve off.
            n = (t_spare >= a) & (t_spare < b) if b < 1.0 else (t_spare >= a) & (t_spare <= b)
            if n.sum() < min_bin:
                continue
            here = rel_spare[n]
            lateral = np.linalg.norm(here - np.outer((here @ u), u), axis=1)
            closest = np.argsort(lateral)[:max(min_bin, len(here) // 8)]
            station_t = float(t_spare[n][closest].mean())
            centre = np.median(here[closest], axis=0)
        else:
            continue
        ts.append((station_t - origin_t) / span)
        o1.append(float(centre @ e1))
        o2.append(float(centre @ e2))
    if len(ts) < 2:
        return np.vstack([base, tip])

    # One polynomial term per three stations of this leaf's own tissue, so a
    # curve never carries more bend than the evidence behind it.
    degree = int(np.clip(max(own_stations, len(ts)) // 3, 1, degree))
    ts = np.asarray(ts)
    design = np.stack([ts ** k * (1.0 - ts) for k in range(lift, degree + lift)], axis=1)
    c1, *_ = np.linalg.lstsq(design, np.asarray(o1), rcond=None)
    c2, *_ = np.linalg.lstsq(design, np.asarray(o2), rcond=None)

    grid = np.linspace(0.0, 1.0, 40)
    tau = np.clip((grid - origin_t) / span, 0.0, 1.0).reshape(-1, 1)
    basis = np.concatenate([tau ** k * (1.0 - tau) for k in range(lift, degree + lift)], axis=1)
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
    min_reach: float = 0.0,
) -> Dict[int, int]:
    """Per instance, the index of the point that reaches furthest from the stem.

    An instance whose furthest point is still within `min_reach` of the stem
    is left out, and the caller keeps the tip persistence proposed. Refining
    against a stem line only helps while that line is a stem: on weed_3 the
    traced trunk doubled back on itself, so for two instances the "furthest"
    point sat on the trunk itself, tip and base landed on the same coordinate
    and the midrib came out zero-length. A tip that has not left the stem is
    not a tip.
    """
    radial = distance_to_polyline(leaf_points, stem_path)
    tips: Dict[int, int] = {}
    for instance in range(int(owner.max()) + 1):
        member = np.nonzero(owner == instance)[0]
        if not len(member):
            continue
        best = int(member[int(np.argmax(radial[member]))])
        if radial[best] > min_reach:
            tips[instance] = best
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
    min_persistence_ratio: Optional[float] = None,
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

    # An upright plant's stem line is measured off the labelled cloud before
    # anything else, because the heart is where its leaf depth is measured
    # from. See `stem_line_from_root`.
    crown = heart = None
    if architecture == "upright":
        line = stem_line_from_root(root_points, stem_points, voxel)
        if line is not None:
            crown, heart = line

    instancing = instance_by_tips(
        leaf_points, stem_points,
        contact_radius=voxel * contact_voxels,
        k_neighbors=k_neighbors,
        min_points=min_leaf_points,
        min_tip_depth=voxel * min_tip_depth_voxels,
        min_persistence_ratio=min_persistence_ratio,
        architecture=architecture,
        heart=heart,
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
    # An upright plant's stem is exactly the crown-to-heart segment measured
    # above. `fixed_stem` keeps the tip refinement below from replacing it
    # with a re-traced trunk -- that trace is what produced weed_3's zigzag,
    # and re-running it after the tips move produces the same zigzag again.
    fixed_stem = crown is not None and heart is not None
    if fixed_stem:
        stem_path = np.vstack([crown.reshape(1, 3), heart.reshape(1, 3)])

    rosette = architecture == "rosette"
    if rosette and instancing.base is not None:
        # Prefer the root->shoot walk: it names the anatomical base directly.
        # The betweenness centre is where the *leaves* meet, which stays the
        # depth field's zero but is a poorer answer for "where does this plant
        # come out of the ground" -- and it is all there is when the specimen
        # was clamped above its root, or the root was never labelled.
        # Foliage means tissue P5 actually attached to a leaf, plus stem --
        # not every leaf-labelled point. The unattached remainder lies in the
        # root band and would masquerade as the foliage bottom.
        attached = leaf_points[instancing.owner >= 0] if len(leaf_points) else leaf_points
        foliage = np.vstack([attached, stem_points]) if len(stem_points) else attached
        crown = crown_from_root(root_points, foliage, voxel)
        if crown is None:
            crown = instancing.base.center
        stem_path = np.asarray(crown, float).reshape(1, 3)

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
        refined = _tips_furthest_from_stem(leaf_points, instancing.owner, stem_path,
                                           min_reach=voxel * 2.0)
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
            if not rosette and not fixed_stem:
                stem_path = new_path

    # Carry the stem down to where the root begins, so the two organs meet
    # instead of stopping a gap apart. Not on the upright path: there the
    # stem line is defined as starting at the crown, and the crown was
    # already found by walking up out of the root.
    if anchor is not None and len(stem_path) > 1 and not fixed_stem:
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
            # To the tip itself, not to the blade chain's last station. That
            # station is a shell *centroid*, so it stops a shell short of the
            # leaf's actual end -- which is why the midrib and its chord
            # agreed with each other and both fell short of the tip.
            # A heart leaf: one of the small upright ones at the centre of
            # the rosette, told apart by how steeply the crown-to-tip line
            # rises out of the ground plane. They are the leaves the cloud
            # fails to reconstruct near the crown, and the only ones the
            # straight-chord rule belongs to.
            rise = np.asarray(tip, float) - np.asarray(base_point, float)
            reach = float(np.linalg.norm(rise))
            elevation = np.degrees(np.arcsin(np.clip(rise[2] / max(reach, 1e-9), -1.0, 1.0)))
            chorded = chord_midrib(leaf_points[member], base_point, tip,
                                   spare=leaf_points[instancing.owner < 0],
                                   heart=elevation >= HEART_LEAF_ELEVATION)
            # Always the chord construction now. Selecting between it and
            # the geodesic station chain was tried and is subtly wrong: the
            # chain is built from real points, so "which curve sits closer to
            # this leaf's tissue" always picks it -- including across the
            # attachment gap, where the chain gets there by detouring around
            # the hole. That detour is the outward bow. The chord fit crosses
            # the gap on the shared tissue lying along this leaf's own line,
            # and follows the blade wherever the blade exists.
            if len(chorded) > 1:
                curve = chorded
                chorded_fit = True

        # Only the station chain needs smoothing. `chord_midrib` already
        # returns an analytic cubic that meets the base and the tip exactly,
        # so running a smoothing spline over it does nothing but drag those
        # endpoints off -- which is what left the midrib stopping short and
        # then taking an elbow to reach its tip.
        if not chorded_fit and len(curve) > 3:
            thickness = float(np.median(cKDTree(leaf_points[member]).query(curve)[0]))
            curve = fit_smooth_curve(curve, tolerance=max(2.0 * thickness, voxel))

        # Re-attach after smoothing. `clip_to_base` put the base on the front,
        # but a smoothing spline does not interpolate its endpoints, so it
        # then pulled the curve off again -- measured on thistle1, midribs
        # starting 3.3 to 11.9 voxels away from a crown they were supposed to
        # begin at. Every tip has to reach the base or the skeleton is not
        # connected.
        # Only the smoothed chain needs its base put back; the chord fit
        # already starts there.
        if not chorded_fit and base_point is not None and len(curve):
            curve = np.vstack([np.asarray(base_point, float).reshape(1, 3), curve[1:]])
        axes.append(curve)
        tips.append(tip)
        attachments.append(shoot[attach_nodes[instance]])

    return LabelledStructure(
        stem_path=stem_path, leaf_ids=instancing.owner, leaf_points=leaf_points,
        attachments=attachments, tips=tips, axes=axes,
        root_points=root_points, instancing=instancing,
        crown=crown, heart=heart,
    )
