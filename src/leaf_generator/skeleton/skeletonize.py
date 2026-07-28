"""Estimate a stem/branch/tip topology graph from a 3D point cloud, via a
minimum-spanning-tree over a k-nearest-neighbor graph.

This is a lightweight heuristic, not a proper curve-skeleton algorithm
(e.g. L1-medial skeleton or mesh contraction): it works reasonably for
tree-like structures (a plant's stem/branch/leaf architecture genuinely is
a tree) as long as the input point cloud is dense and clean enough that
"nearest neighbors" mostly means "adjacent along the plant", not "adjacent
because of unrelated noise nearby". Sparse or noisy point clouds will
produce a noisy or wrong-looking skeleton -- inspect the point cloud
alongside the graph, don't trust the graph blindly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree


@dataclass
class SkeletonGraph:
    points: np.ndarray  # (N, 3) points retained in the skeleton (post outlier/component filtering)
    mst_edges: List[Tuple[int, int]]  # raw MST edges, indices into `points`, for detailed rendering
    keypoint_indices: List[int]  # indices of tip/branch nodes, post-pruning
    keypoint_kinds: Dict[int, str] = field(default_factory=dict)  # index -> "tip" | "branch"
    simplified_edges: List[Tuple[int, int]] = field(default_factory=list)  # keypoint-to-keypoint
    branch_polylines: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    # simplified_edges key -> ordered point indices from one keypoint to the
    # other (inclusive of both endpoints) along the collapsed chain -- for
    # rendering a branch as a curve that follows the stem, not a straight
    # chord between its two keypoints.

    @property
    def num_tips(self) -> int:
        return sum(1 for kind in self.keypoint_kinds.values() if kind == "tip")

    @property
    def num_branch_points(self) -> int:
        return sum(1 for kind in self.keypoint_kinds.values() if kind == "branch")


def build_skeleton_graph(
    xyz: np.ndarray,
    k_neighbors: int = 8,
    min_branch_length_fraction: float = 0.03,
    max_prune_passes: int = 5,
) -> SkeletonGraph:
    if len(xyz) < 3:
        raise ValueError(f"Need at least 3 points to build a skeleton, got {len(xyz)}")

    xyz, index_map = _largest_knn_component(xyz, k_neighbors)
    mst = _minimum_spanning_tree_over_knn(xyz, k_neighbors)
    adjacency = _mst_to_adjacency(mst)

    total_length = sum(w for neighbors in adjacency.values() for w in neighbors.values()) / 2
    min_branch_length = min_branch_length_fraction * total_length

    adjacency = _prune_short_spurs(adjacency, xyz, min_branch_length, max_prune_passes)

    degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
    keypoint_kinds = {
        node: ("tip" if degree == 1 else "branch")
        for node, degree in degrees.items()
        if degree == 1 or degree >= 3
    }

    simplified_edges, branch_polylines = _contract_chains(adjacency, keypoint_kinds)
    mst_edges = [(i, j) for i in adjacency for j in adjacency[i] if i < j]

    return SkeletonGraph(
        points=xyz,
        mst_edges=mst_edges,
        keypoint_indices=sorted(keypoint_kinds.keys()),
        keypoint_kinds=keypoint_kinds,
        simplified_edges=simplified_edges,
        branch_polylines=branch_polylines,
    )


def _largest_knn_component(xyz: np.ndarray, k_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only points that fall in the largest connected component of the
    k-NN graph, so a disjoint straggler point can't break the MST step.
    """
    tree = cKDTree(xyz)
    k = min(k_neighbors + 1, len(xyz))
    dists, idx = tree.query(xyz, k=k)

    n = len(xyz)
    rows = np.repeat(np.arange(n), k - 1)
    cols = idx[:, 1:].ravel()
    data = dists[:, 1:].ravel()
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)  # symmetrize

    n_components, labels = connected_components(graph, directed=False)
    if n_components == 1:
        return xyz, np.arange(n)

    largest_label = np.argmax(np.bincount(labels))
    keep = labels == largest_label
    return xyz[keep], np.nonzero(keep)[0]


def _minimum_spanning_tree_over_knn(xyz: np.ndarray, k_neighbors: int) -> csr_matrix:
    tree = cKDTree(xyz)
    n = len(xyz)
    k = min(k_neighbors + 1, n)
    dists, idx = tree.query(xyz, k=k)

    rows = np.repeat(np.arange(n), k - 1)
    cols = idx[:, 1:].ravel()
    data = dists[:, 1:].ravel()
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)

    mst = minimum_spanning_tree(graph)
    return mst.maximum(mst.T)  # symmetrize for easy adjacency traversal


def _mst_to_adjacency(mst: csr_matrix) -> Dict[int, Dict[int, float]]:
    mst_coo = mst.tocoo()
    adjacency: Dict[int, Dict[int, float]] = {}
    for i, j, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        if i == j:
            continue
        adjacency.setdefault(int(i), {})[int(j)] = float(w)
        adjacency.setdefault(int(j), {})[int(i)] = float(w)
    return adjacency


def _prune_short_spurs(
    adjacency: Dict[int, Dict[int, float]],
    xyz: np.ndarray,
    min_branch_length: float,
    max_passes: int,
) -> Dict[int, Dict[int, float]]:
    """Repeatedly remove tip-terminated spurs shorter than `min_branch_length`
    -- noise typically shows up as short dead-end branches off the main
    structure.
    """
    adjacency = {node: dict(neighbors) for node, neighbors in adjacency.items()}

    for _ in range(max_passes):
        degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
        tips = [node for node, degree in degrees.items() if degree == 1]
        if not tips:
            break

        removed_any = False
        for tip in tips:
            if tip not in adjacency or not adjacency[tip]:
                continue
            path, length = _walk_to_next_keypoint(adjacency, tip)
            if length < min_branch_length and len(adjacency) - len(path) >= 3:
                for node in path:
                    _remove_node(adjacency, node)
                removed_any = True

        if not removed_any:
            break

    return adjacency


def _walk_to_next_keypoint(adjacency: Dict[int, Dict[int, float]], start: int) -> Tuple[List[int], float]:
    """Walk from a degree-1 tip along the chain of degree-2 nodes until
    hitting a branch point (degree>=3) or another tip; returns the nodes
    strictly before that junction (exclusive) and the accumulated length.
    """
    path = [start]
    length = 0.0
    prev, current = None, start

    while True:
        neighbors = adjacency.get(current, {})
        next_candidates = [n for n in neighbors if n != prev]
        if len(neighbors) != (1 if current == start else 2) or not next_candidates:
            break
        nxt = next_candidates[0]
        length += neighbors[nxt]
        if len(adjacency.get(nxt, {})) != 2:
            break
        path.append(nxt)
        prev, current = current, nxt

    return path, length


def _remove_node(adjacency: Dict[int, Dict[int, float]], node: int) -> None:
    for neighbor in list(adjacency.get(node, {})):
        adjacency[neighbor].pop(node, None)
    adjacency.pop(node, None)


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
