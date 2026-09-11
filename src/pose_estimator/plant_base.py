"""Where leaves attach, found from geometry instead of from a stem label.

P5 never actually needed a stem. It needed a *base*: the set of points to
start the geodesic depth field from, so that "how far into a leaf is this
point" has a zero. On an upright plant the stem plays that role. On a rosette
-- thistle, sugar beet -- there is no stem to play it, the leaves radiate
from a compressed crown at ground level, and the roots fall outside the P2
plant mask so they cannot be clicked either. Asking the operator to seed a
`stem` class there is asking for something the specimen does not have, and
what came back was 0 contact points, 0 tips and 0 leaves.

So the base is derived rather than labelled:

1. take the extremities by geodesic farthest-point sampling -- no stem needed
   to find the far ends of a plant, only the graph;
2. every leaf-to-leaf path must cross the attachment region, whatever shape
   it is, so count how many of those paths pass near each point;
3. the hot set is the base.

This locates a crown; it does not decide whether the plant has one. Which
architecture a capture is gets passed in with --architecture, because it is
known at capture time and a classifier for it is a liability: an earlier
version read the hot set's spread and flipped thistle1 from crown (0.016 of
plant extent) to stem (0.148) purely because P4c had started labelling the
crown "stem", removing it from the tissue the test ran on. Same plant, same
geometry, opposite verdict.

Measured on runs/thistle1: the hot set localises to 3.1% of plant extent at a
4% ball and 3.9% at 6%, a stable plateau rather than a knife-edge. Those
numbers are still reported in p5/instancing.json as evidence that the crown
was actually found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

ROSETTE = "rosette"
CAULESCENT = "caulescent"


@dataclass
class BaseRegion:
    """The attachment region: where the depth field starts."""

    nodes: np.ndarray  # indices into the point array: the depth field's zero
    center: np.ndarray  # (3,) centroid of those nodes
    extremities: np.ndarray  # the geodesic extremities used to find it
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "num_base_nodes": int(len(self.nodes)),
            "center": [float(x) for x in self.center],
            "num_extremities": int(len(self.extremities)),
            **self.evidence,
        }


def geodesic_extremities(graph: csr_matrix, max_count: int = 16,
                         stop_ratio: float = 0.25) -> np.ndarray:
    """Far ends of the plant, by geodesic farthest-point sampling.

    Start anywhere, take the geodesically furthest point, then repeatedly take
    the point furthest from everything chosen so far. The first pick is
    arbitrary but the second is not: the furthest point from anywhere is
    always an extremity, so the sampling is seeded correctly after one step.

    Stops when a new extremity is closer than `stop_ratio` of the first
    covering radius -- a ratio, so it carries no length scale and does not
    need refitting per specimen. This deliberately over-collects: extra
    extremities on one blade are harmless here (persistence decides the real
    leaf count later), whereas a missed blade would leave its paths out of the
    betweenness count.
    """
    n = graph.shape[0]
    if n == 0:
        return np.zeros(0, np.int64)

    first = int(np.argmax(np.nan_to_num(dijkstra(graph, directed=False, indices=0),
                                        posinf=0.0)))
    picked = [first]
    covering = np.nan_to_num(dijkstra(graph, directed=False, indices=first), posinf=0.0)

    span0 = None
    while len(picked) < max_count:
        nxt = int(np.argmax(covering))
        span = float(covering[nxt])
        if span0 is None:
            span0 = span
        elif span0 > 0 and span / span0 < stop_ratio:
            break
        picked.append(nxt)
        covering = np.minimum(
            covering, np.nan_to_num(dijkstra(graph, directed=False, indices=nxt), posinf=0.0))
    return np.array(picked, np.int64)


def path_betweenness(graph: csr_matrix, extremities: Sequence[int]) -> np.ndarray:
    """How many extremity-to-extremity geodesics pass through each node.

    Every path between two leaves has to cross wherever those leaves are
    joined, so the attachment region is whatever the paths have in common.
    """
    count = np.zeros(graph.shape[0], np.int64)
    tips = list(extremities)
    for i, a in enumerate(tips):
        _, predecessor = dijkstra(graph, directed=False, indices=a,
                                  return_predecessors=True)
        for b in tips[i + 1:]:
            node = int(b)
            while node != a and node >= 0:
                count[node] += 1
                node = int(predecessor[node])
    return count


def find_base(
    points: np.ndarray,
    graph: csr_matrix,
    ball_fraction: float = 0.04,
    hot_quantile: float = 0.9,
    max_extremities: int = 16,
) -> Optional[BaseRegion]:
    """The attachment region, with no stem label anywhere in the derivation.

    `ball_fraction` is a fraction of the plant's own extent, so it carries no
    length scale. The spread and elongation of the hot set are reported as
    evidence but decide nothing: which architecture a plant has is told to the
    pipeline with --architecture, not guessed from it.

    Betweenness is aggregated over a ball before the hot set is taken. Raw
    per-node betweenness does not work on this data and it is worth recording
    why: the cloud is a *surface*, not a curve skeleton, so geodesics spread
    across many parallel routes through a wide crown instead of funnelling
    through one node. On runs/thistle1 the busiest single node carried 6 of 21
    paths. Aggregated over a ball the same region carries all of them.
    """
    if len(points) == 0 or graph.shape[0] != len(points):
        return None

    extremities = geodesic_extremities(graph, max_count=max_extremities)
    if len(extremities) < 2:
        return None

    count = path_betweenness(graph, extremities)
    if count.sum() == 0:
        return None

    extent = float(np.ptp(points, axis=0).max())
    radius = ball_fraction * extent
    tree = cKDTree(points)
    # Scattered from the path nodes rather than gathered around every point.
    # The two are identical -- a ball sum is symmetric, and `count` is zero
    # except on the geodesics between at most `max_extremities` tips, so only
    # those nodes can contribute anything. Gathering asked cKDTree for one
    # Python list per point, which is fine at the 57k points of a carved
    # surface and fatal at the 190k of a learned cloud: that form reached 6GB
    # and was killed by the OOM killer, on a plant no bigger than before.
    aggregated = np.zeros(len(points), float)
    carriers = np.nonzero(count > 0)[0]
    for start in range(0, len(carriers), 512):
        chunk = carriers[start:start + 512]
        for node, neighbours in zip(chunk, tree.query_ball_point(points[chunk], radius)):
            aggregated[neighbours] += count[node]
    if aggregated.max() <= 0:
        return None

    hot = np.nonzero(aggregated >= hot_quantile * aggregated.max())[0]
    center = points[hot].mean(axis=0)

    offset = points[hot] - center
    spread = float(np.percentile(np.linalg.norm(offset, axis=1), 90)) / max(extent, 1e-12)

    # Shape of the hot set: a crown is a blob, a stem is a curve. Reported
    # either way so a disagreement between the two is visible rather than
    # silently resolved.
    if len(hot) >= 3:
        eigenvalues = np.linalg.eigvalsh(np.cov(offset.T))[::-1]
        elongation = float(np.sqrt(max(eigenvalues[0], 0) / max(eigenvalues[1], 1e-12)))
    else:
        elongation = float("inf")

    return BaseRegion(
        nodes=hot,
        center=center,
        extremities=extremities,
        evidence={
            "base_spread_fraction_of_extent": round(spread, 4),
            "base_elongation": round(elongation, 3),
            "ball_fraction": ball_fraction,
            "plant_extent": round(extent, 5),
        },
    )
