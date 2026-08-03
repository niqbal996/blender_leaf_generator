"""Render a point cloud + its estimated skeleton graph for visual inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .skeletonize import SkeletonGraph, smooth_polyline

DEFAULT_VIEWS = [(15, 20), (15, 110), (80, 20)]


def plot_skeleton(
    skeleton: SkeletonGraph,
    out_path: Union[str, Path],
    background_xyz: Optional[np.ndarray] = None,
    background_rgb: Optional[np.ndarray] = None,
    views: Sequence[Tuple[float, float]] = DEFAULT_VIEWS,
    point_size: float = 3.0,
) -> None:
    """Save a multi-angle render: optional raw point cloud in the background
    (faint, true colors), the underlying node graph in light gray, and each
    branch drawn in bold along its actual curve (root=green, tips=blue,
    branch points=red).
    """
    fig = plt.figure(figsize=(6 * len(views), 6))

    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")

        if background_xyz is not None and len(background_xyz):
            colors = (background_rgb / 255.0) if background_rgb is not None else "lightgray"
            ax.scatter(*background_xyz.T, c=colors, s=point_size * 0.5, alpha=0.25, linewidths=0)

        pts = skeleton.points
        for i0, j0 in skeleton.mst_edges:
            ax.plot(*zip(pts[i0], pts[j0]), c="gray", linewidth=0.5, alpha=0.6)

        # Each branch is drawn along its polyline, not as a straight chord
        # between its two keypoints. The chord version made every branch
        # look straight and arbitrarily-directed no matter what shape the
        # skeleton actually had -- which, for a pipeline whose whole point
        # is recovering *curved* leaf midribs, hid the signal being
        # inspected and made correct and broken skeletons render alike.
        for edge in skeleton.simplified_edges:
            path = skeleton.branch_polylines.get(edge) or skeleton.branch_polylines.get(
                tuple(sorted(edge))
            )
            curve = smooth_polyline(pts[path]) if path else np.array([pts[edge[0]], pts[edge[1]]])
            ax.plot(*curve.T, c="black", linewidth=2.0)

        tip_idx = [idx for idx, kind in skeleton.keypoint_kinds.items() if kind == "tip"]
        branch_idx = [idx for idx, kind in skeleton.keypoint_kinds.items() if kind == "branch"]
        root_idx = [idx for idx, kind in skeleton.keypoint_kinds.items() if kind == "root"]
        if tip_idx:
            ax.scatter(*pts[tip_idx].T, c="royalblue", s=60, label="tip", depthshade=False)
        if branch_idx:
            ax.scatter(*pts[branch_idx].T, c="crimson", s=80, label="branch point", depthshade=False)
        if root_idx:
            ax.scatter(*pts[root_idx].T, c="limegreen", s=100, label="root", depthshade=False)

        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(f"elev={elev} azim={azim}")
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
