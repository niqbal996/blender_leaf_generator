"""Render a point cloud + its estimated skeleton graph for visual inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .skeletonize import SkeletonGraph

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
    (faint, true colors), the MST in light gray, and the simplified
    keypoint graph in bold (tips=blue, branch points=red).
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

        for i0, j0 in skeleton.simplified_edges:
            ax.plot(*zip(pts[i0], pts[j0]), c="black", linewidth=2.0)

        tip_idx = [idx for idx, kind in skeleton.keypoint_kinds.items() if kind == "tip"]
        branch_idx = [idx for idx, kind in skeleton.keypoint_kinds.items() if kind == "branch"]
        if tip_idx:
            ax.scatter(*pts[tip_idx].T, c="royalblue", s=60, label="tip", depthshade=False)
        if branch_idx:
            ax.scatter(*pts[branch_idx].T, c="crimson", s=80, label="branch point", depthshade=False)

        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(f"elev={elev} azim={azim}")
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
