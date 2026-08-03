"""Estimate a stem/branch/tip curve skeleton from a 3D point cloud, by
iteratively tracing organs outward from the plant's base.

Geodesic (along-the-plant) distance is computed from the root over a
k-nearest-neighbor graph. The point farthest from the root is a leaf tip,
so the shortest path back from it is that leaf's midline; that path is
recorded as a branch and the blade around it is marked as claimed. Repeat
on the farthest *unclaimed* point until what remains is too short to be an
organ. Each node is finally re-centered onto the centroid of the cloud
points nearest it, which puts the curve on the blade's central vein rather
than on whichever surface points happened to reconstruct.

Claiming each blade as it is extracted is the crux. Purely local
constructions -- a minimum spanning tree, or geodesic level sets -- have no
way to tell "this blade's point coverage is momentarily ragged" from "the
stem forks here", because at leaf scale the two look identical. Both were
tried on the turntable test captures and both fragmented single rosette
leaves into many strands, reporting 23 and 81 tips for plants with 3 and 8
real leaves, with fragment lengths forming a smooth continuum with the real
leaves' -- so no pruning threshold could separate them. Extracting whole
organs greedily, longest first, sidesteps that: a leaf is consumed in one
pass and cannot reappear as its own fragments.

Still a heuristic, not a guaranteed-correct curve skeleton. It assumes
"geodesically near" means "adjacent along the plant", so two leaves
touching in space can be bridged into one branch, and it cannot distinguish
a genuinely tiny leaf from a fragment of a large one by length alone.
Inspect the point cloud alongside the graph rather than trusting it
blindly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .pointcloud import voxel_downsample


@dataclass
class SkeletonGraph:
    points: np.ndarray  # (N, 3) points retained in the skeleton
    mst_edges: List[Tuple[int, int]]  # every node-to-node edge, indices into `points`, for detailed rendering
    keypoint_indices: List[int]  # indices of root/tip/branch nodes
    keypoint_kinds: Dict[int, str] = field(default_factory=dict)  # index -> "root" | "tip" | "branch"
    simplified_edges: List[Tuple[int, int]] = field(default_factory=list)  # keypoint-to-keypoint
    branch_polylines: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    # simplified_edges key -> ordered point indices from one keypoint to the
    # other (inclusive of both endpoints) along the collapsed chain -- for
    # rendering a branch as a curve that follows the stem, not a straight
    # chord between its two keypoints. When `root_index` is set, both
    # simplified_edges and branch_polylines are oriented (parent, child)
    # outward from the root, so the graph reads as "from the ground,
    # up the stem, out to each leaf tip" instead of an arbitrary order.
    root_index: Optional[int] = None

    @property
    def num_tips(self) -> int:
        return sum(1 for kind in self.keypoint_kinds.values() if kind == "tip")

    @property
    def num_branch_points(self) -> int:
        return sum(1 for kind in self.keypoint_kinds.values() if kind == "branch")


def build_skeleton_graph(
    xyz: np.ndarray,
    k_neighbors: int = 8,
    root_xyz: Optional[np.ndarray] = None,
    voxel_downsample_fraction: Optional[float] = None,
    min_branch_fraction: float = 0.25,
    cover_radius_fraction: float = 0.08,
    min_support_density: float = 0.8,
) -> SkeletonGraph:
    """`root_xyz` (see `turntable.find_root_point_on_ground`) is the (3,)
    coordinate where the plant emerges from the soil, and every branch is
    traced back to it. It is close to mandatory: without it there is no
    "distance from the base" to order organs by, so the geodesically most
    extreme point is used instead -- that is some leaf tip, and the whole
    skeleton then reads as growing out of that leaf.

    The two tuning knobs are both fractions of the plant's own size, so
    they carry between captures whose COLMAP units differ:

    `min_branch_fraction` -- shortest organ to accept, as a fraction of the
    longest root-to-tip distance. This is the main sensitivity trade: raise
    it if leaf blades are being split into extra branches, lower it if
    genuinely small leaves are missed. A real but tiny leaf and a fragment
    of a big one are not distinguishable by length alone, so this cannot be
    pushed arbitrarily low.

    `cover_radius_fraction` -- how wide a swath around each traced path is
    claimed as that organ's, as a fraction of the bounding-box diagonal.
    Roughly a leaf's half-width. Too small and one blade is re-extracted as
    several parallel branches; too large and a real leaf is swallowed by
    its neighbor.

    `min_support_density` -- how many cloud points, within the cover
    radius, a branch must have per unit of point spacing along its length.
    This is what rejects branches that are a handful of stray points
    bridged across empty space (substrate grit, stray green-cast specks)
    rather than a real organ. Lower it if sparsely-reconstructed real
    leaves vanish; raise it if debris trails still get traced.

    `voxel_downsample_fraction` (a fraction of the bounding-box diagonal)
    thins the cloud first. Off by default: it existed to fight the older
    MST construction's wandering across leaf surfaces, and this one wants
    the opposite -- dense surface coverage is what lets each node re-center
    onto the blade's true midline. Still available for very large clouds.
    """
    if len(xyz) < 3:
        raise ValueError(f"Need at least 3 points to build a skeleton, got {len(xyz)}")

    if voxel_downsample_fraction is not None:
        bbox_diagonal = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
        voxel_size = bbox_diagonal * voxel_downsample_fraction
        if voxel_size > 0:
            thinned, _ = voxel_downsample(xyz, voxel_size=voxel_size)
            if len(thinned) >= 3:
                xyz = thinned

    graph = _connected_knn_graph(xyz, k_neighbors)

    source = _nearest_point_index(xyz, root_xyz) if root_xyz is not None else None
    geodesic, predecessors = _geodesic_from_root(graph, source)
    if source is None:
        source = int(np.argmin(geodesic))

    extent = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    cover_radius = max(extent * cover_radius_fraction, 1e-9)

    paths = _trace_branches(
        xyz, geodesic, predecessors, min_branch_fraction, cover_radius, min_support_density
    )
    nodes, adjacency, root_index = _branches_to_nodes(xyz, paths, source, cover_radius)
    xyz = nodes

    degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
    keypoint_kinds: Dict[int, str] = {}
    for node, degree in degrees.items():
        if node == root_index:
            keypoint_kinds[node] = "root"
        elif degree == 1 or degree >= 3:
            keypoint_kinds[node] = "tip" if degree == 1 else "branch"
    if root_index is not None and root_index not in keypoint_kinds and root_index in adjacency:
        keypoint_kinds[root_index] = "root"

    simplified_edges, branch_polylines = _contract_chains(adjacency, keypoint_kinds)
    rooted = root_index is not None and root_index in keypoint_kinds
    if rooted:
        simplified_edges, branch_polylines = _orient_from_root(simplified_edges, branch_polylines, root_index)

    mst_edges = [(i, j) for i in adjacency for j in adjacency[i] if i < j]

    return SkeletonGraph(
        points=xyz,
        mst_edges=mst_edges,
        keypoint_indices=sorted(keypoint_kinds.keys()),
        keypoint_kinds=keypoint_kinds,
        simplified_edges=simplified_edges,
        branch_polylines=branch_polylines,
        root_index=root_index if rooted else None,
    )


def _geodesic_from_root(
    graph: csr_matrix, source: Optional[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Geodesic distance (and the shortest-path predecessor tree) from
    `source` to every point, measured *along the plant* rather than through
    the air -- two points on either side of the stem are Euclidean-near but
    geodesically far, and it is the geodesic ordering that makes distance
    shells cut across organs instead of slicing through the whole plant.

    With `source=None` (no root supplied), the geodesically most distant
    point from an arbitrary start is used -- the standard "double sweep"
    for an extremity. On a plant that is a leaf tip, not the base, so the
    caller should supply a real root whenever one is available.
    """
    if source is None:
        first_pass = dijkstra(graph, directed=False, indices=0)
        finite = np.isfinite(first_pass)
        source = int(np.argmax(np.where(finite, first_pass, -np.inf)))

    distances, predecessors = dijkstra(
        graph, directed=False, indices=source, return_predecessors=True
    )
    # A disconnected straggler would otherwise land in no shell at all and
    # silently disappear; parking it in the outermost shell keeps it
    # visible as a spur for the pruning pass to judge on its merits.
    finite = np.isfinite(distances)
    if not finite.all():
        distances = np.where(finite, distances, distances[finite].max() if finite.any() else 0.0)
    return distances, predecessors


def _trace_branches(
    xyz: np.ndarray,
    geodesic: np.ndarray,
    predecessors: np.ndarray,
    min_branch_fraction: float,
    cover_radius: float,
    min_support_density: float = 0.8,
) -> List[List[int]]:
    """Extract one path per organ, farthest point first: take the point
    geodesically farthest from the root, trace the shortest-path tree back
    until it meets an already-extracted branch, record that as a branch,
    then mark every point within `cover_radius` of it as claimed. Repeat.

    Claiming the whole blade around each path is the part that matters. It
    means an organ is consumed in one go and cannot come back as a second,
    third and fourth branch -- which is what a purely local construction
    (level sets, or an MST) cannot prevent: any place a blade's point
    coverage is momentarily ragged looks locally identical to a real fork,
    so a rosette leaf fragments into several strands and every fragment
    boundary becomes a spurious branch point. Measured on the two test
    captures, that produced 23 and 81 raw tips for plants with 3 and 8 real
    leaves, and the fragments' lengths formed a smooth continuum with the
    real leaves' -- no pruning threshold could separate them, because
    fragmentation splits a real leaf into pieces that are individually as
    short as the noise.

    Tracing farthest-first instead compares each candidate against the
    plant's *whole* extent (`min_branch_fraction` of the longest root-to-tip
    distance) rather than against local point spacing, which is a question
    about anatomy -- "is this long enough to be an organ?" -- and not about
    sampling density.
    """
    max_distance = float(geodesic.max())
    covered = np.zeros(len(xyz), dtype=bool)
    tree = cKDTree(xyz)
    spacing = float(np.median(tree.query(xyz, k=min(2, len(xyz)))[0][:, -1])) or 1.0
    paths: List[List[int]] = []

    for seed in np.argsort(-geodesic):
        if covered[seed]:
            continue

        # Traced all the way back to the root, not stopped at the first
        # claimed point. Two seeds' shortest paths merge exactly at their
        # common ancestor, so full paths share a prefix node-for-node and
        # the branch points fall out of that sharing for free. Stopping
        # early instead ends each branch on whichever *claimed* point the
        # walk happened to touch first -- and since claiming covers a whole
        # blade, not just its midline, that point usually lies on no other
        # path at all, leaving every branch a disconnected floating chain.
        path: List[int] = []
        current = int(seed)
        while current >= 0:
            path.append(current)
            current = int(predecessors[current])

        # The new part is everything out beyond the already-claimed region.
        claimed = [i for i, point in enumerate(path) if covered[point]]
        junction_at = claimed[0] if claimed else len(path) - 1
        junction = path[junction_at]
        new_length = geodesic[seed] - geodesic[junction]
        if new_length < min_branch_fraction * max_distance:
            continue

        # A real organ is *made of* points along its whole length; a
        # spurious branch is a few stray points bridged across empty space.
        # Measured on the sparse test capture, the branch that crawled
        # sideways along the substrate and the two others that looked wrong
        # in Blender carried 0.36-0.40 supporting points per point-spacing
        # step, against 1.23-1.89 for the three real leaves -- while on the
        # dense capture every real branch scored above 5.9. Neither height
        # above the substrate nor branch length separates those cases (that
        # crawler was one of the *longest* branches, and real leaves splay
        # down to substrate level), but support density does.
        segment = path[: junction_at + 1]
        supported = set()
        for neighbors in tree.query_ball_point(xyz[segment], r=cover_radius):
            supported.update(neighbors)
        steps = max(new_length / spacing, 1e-9)
        if len(supported) / steps < min_support_density:
            continue

        # Trim the branch at its true tip. The seed is the geodesically
        # farthest point, but geodesic distance keeps growing as a path
        # wraps around a blade's edge, so on a wide leaf the farthest point
        # is out on the margin *past* the tip -- the traced curve then runs
        # up the midrib, reaches the tip, and carries on hooking around the
        # edge and back down the side. The anatomical tip is instead where
        # straight-line distance from the leaf's attachment stops growing,
        # since wrapping around the margin no longer takes you further from
        # the base. Measured on the one leaf that showed this, the path ran
        # 2.27x its own chord and reached 0.397 from its start against a
        # 0.265 chord; every well-behaved branch sat at 1.02-1.37.
        # Trimming drops only the overshoot beyond the tip; the path still
        # runs back to the root, because sibling branches sharing a prefix
        # node-for-node is what creates the branch points at all.
        distances = np.linalg.norm(xyz[segment] - xyz[junction], axis=1)
        trimmed = path[int(np.argmax(distances)) :]

        # Claiming still uses the full traced path, not the trimmed one, so
        # the wrapped-around margin stays claimed and cannot come back as a
        # branch of its own.
        paths.append(trimmed[::-1])  # root -> tip
        for neighbors in tree.query_ball_point(xyz[path], r=cover_radius):
            covered[neighbors] = True

    return paths


def _branches_to_nodes(
    xyz: np.ndarray,
    paths: List[List[int]],
    root_point: int,
    cover_radius: float,
) -> Tuple[np.ndarray, Dict[int, Dict[int, float]], Optional[int]]:
    """Turn traced point paths into skeleton nodes and adjacency, then pull
    each node onto its organ's midline.

    Re-centering is what turns a traced path into a *midrib*. The traced
    path runs through actual reconstructed points, and those sit on the
    blade's surface -- often nearer one edge than the center, wherever that
    edge happened to reconstruct more densely. Averaging each node against
    the cloud points it is nearest to recovers the center of the blade's
    local cross-section, which is where the central vein runs.
    """
    node_of_point: Dict[int, int] = {}
    adjacency: Dict[int, Dict[int, float]] = {}

    for path in paths:
        for point in path:
            if point not in node_of_point:
                node_of_point[point] = len(node_of_point)
                adjacency[node_of_point[point]] = {}
        for a, b in zip(path, path[1:]):
            i, j = node_of_point[a], node_of_point[b]
            if i != j:
                weight = float(np.linalg.norm(xyz[a] - xyz[b]))
                adjacency[i][j] = weight
                adjacency[j][i] = weight

    if not node_of_point:
        raise ValueError("no branch survived tracing -- try lowering min_branch_fraction")

    ordered_points = sorted(node_of_point, key=lambda p: node_of_point[p])
    nodes = xyz[ordered_points].copy()

    # Assign every cloud point to its nearest node, then move each node to
    # the centroid of what it captured.
    node_tree = cKDTree(nodes)
    distances, nearest = node_tree.query(xyz, k=1)
    within = distances <= cover_radius
    if within.any():
        sums = np.zeros_like(nodes)
        counts = np.zeros(len(nodes))
        np.add.at(sums, nearest[within], xyz[within])
        np.add.at(counts, nearest[within], 1)
        moved = counts > 0
        nodes[moved] = sums[moved] / counts[moved, None]

    return nodes, adjacency, node_of_point.get(root_point)


def _connected_knn_graph(xyz: np.ndarray, k_neighbors: int) -> csr_matrix:
    """Build a k-NN graph over `xyz`, then bridge any disconnected
    components via each pair's single nearest cross-component point pair
    (cheapest-edge-first, union-find -- a component-level minimum spanning
    tree), so the graph is guaranteed connected before computing a
    spanning tree over it.

    Separate plant parts (e.g. two leaves that survived `pointcloud.
    keep_plant_clusters` but aren't close enough to be mutual k-nearest-
    neighbors) need to actually end up in the same skeleton graph instead
    of one silently vanishing here -- which is what discarding non-largest
    components used to do.
    """
    n = len(xyz)
    tree = cKDTree(xyz)
    k = min(k_neighbors + 1, n)
    dists, idx = tree.query(xyz, k=k)

    rows = np.repeat(np.arange(n), k - 1)
    cols = idx[:, 1:].ravel()
    data = dists[:, 1:].ravel()
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)  # symmetrize

    n_components, labels = connected_components(graph, directed=False)
    if n_components == 1:
        return graph

    graph = graph.tolil()
    component_indices = [np.nonzero(labels == c)[0] for c in range(n_components)]
    component_trees = [cKDTree(xyz[indices]) for indices in component_indices]

    candidates = []  # (distance, comp_a, comp_b, global_i, global_j)
    for a in range(n_components):
        for b in range(a + 1, n_components):
            dists_to_b, nearest_in_b = component_trees[b].query(xyz[component_indices[a]], k=1)
            best_a_local = int(np.argmin(dists_to_b))
            global_i = int(component_indices[a][best_a_local])
            global_j = int(component_indices[b][nearest_in_b[best_a_local]])
            candidates.append((float(dists_to_b[best_a_local]), a, b, global_i, global_j))
    candidates.sort(key=lambda c: c[0])

    parent = list(range(n_components))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    merged = 0
    for dist, a, b, gi, gj in candidates:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        graph[gi, gj] = dist
        graph[gj, gi] = dist
        parent[ra] = rb
        merged += 1
        if merged == n_components - 1:
            break

    return graph.tocsr()



def _nearest_point_index(xyz: np.ndarray, target_xyz: np.ndarray) -> int:
    tree = cKDTree(xyz)
    _, index = tree.query(target_xyz, k=1)
    return int(index)





def smooth_polyline(points: np.ndarray, iterations: int = 8, strength: float = 0.4) -> np.ndarray:
    """Take the local jitter out of a branch polyline via a few passes of
    Laplacian smoothing: each interior point is pulled toward the midpoint
    of its two neighbors along the path. The two endpoints (the branch's
    actual keypoints -- root/branch/tip) are never moved.

    Iteration count matters more than it looks. With the endpoints pinned,
    this converges to the straight chord between them, so running it to
    convergence returns a straight line no matter what shape went in -- it
    used to default to 200 iterations, which is well past that point. That
    was defensible when branch polylines came from an MST threading back
    and forth across a leaf blade, where essentially all of the wiggle was
    noise; it is not defensible now that they come from geodesic level-set
    centroids, where the wiggle is mostly the organ's real curvature. A
    curved leaf is exactly the signal this pipeline is trying to capture,
    so run just enough passes to suppress centroid jitter and stop.
    """
    points = points.copy()
    n = len(points)
    if n <= 2:
        return points
    for _ in range(iterations):
        midpoints = (points[:-2] + points[2:]) / 2
        points[1:-1] = points[1:-1] + strength * (midpoints - points[1:-1])
    return points



def _contract_chains(
    adjacency: Dict[int, Dict[int, float]], keypoint_kinds: Dict[int, str]
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], List[int]]]:
    """Collapse chains of degree-2 nodes between keypoints into direct
    keypoint-to-keypoint edges, also returning each edge's full node path
    (inclusive of both endpoints) in order from `edge[0]` to `edge[1]`.
    """
    simplified_edges = []
    branch_polylines: Dict[Tuple[int, int], List[int]] = {}
    visited_directed_edges = set()

    for keypoint in keypoint_kinds:
        for neighbor in adjacency.get(keypoint, {}):
            if (keypoint, neighbor) in visited_directed_edges:
                continue

            path = [keypoint, neighbor]
            prev, current = keypoint, neighbor
            visited_directed_edges.add((prev, current))
            while current not in keypoint_kinds:
                next_candidates = [n for n in adjacency.get(current, {}) if n != prev]
                if not next_candidates:
                    break
                nxt = next_candidates[0]
                visited_directed_edges.add((current, nxt))
                path.append(nxt)
                prev, current = current, nxt

            if current in keypoint_kinds:
                visited_directed_edges.add((current, prev))
                edge = tuple(sorted((keypoint, current)))
                if edge not in branch_polylines and edge[0] != edge[1]:
                    simplified_edges.append(edge)
                    branch_polylines[edge] = path if edge[0] == keypoint else list(reversed(path))

    return simplified_edges, branch_polylines


def _orient_from_root(
    simplified_edges: List[Tuple[int, int]],
    branch_polylines: Dict[Tuple[int, int], List[int]],
    root_index: int,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], List[int]]]:
    """Reorient every simplified edge/polyline as (parent, child) outward
    from `root_index`, via BFS over the (small) keypoint graph -- so the
    skeleton reads as "starting from the ground, following the stem,
    branching into leaves" instead of an arbitrarily-ordered undirected
    tree.
    """
    keypoint_adjacency: Dict[int, List[int]] = {}
    for a, b in simplified_edges:
        keypoint_adjacency.setdefault(a, []).append(b)
        keypoint_adjacency.setdefault(b, []).append(a)

    directed_edges: List[Tuple[int, int]] = []
    directed_polylines: Dict[Tuple[int, int], List[int]] = {}
    visited = {root_index}
    queue = deque([root_index])

    while queue:
        node = queue.popleft()
        for neighbor in keypoint_adjacency.get(node, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            edge = (node, neighbor)
            directed_edges.append(edge)
            original = tuple(sorted((node, neighbor)))
            path = branch_polylines[original]
            directed_polylines[edge] = path if path[0] == node else list(reversed(path))
            queue.append(neighbor)

    return directed_edges, directed_polylines
