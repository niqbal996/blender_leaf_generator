"""P5 diagnostics: is this actually the plant's structure, or just a graph?

A skeleton is the easiest thing in the pipeline to get plausibly wrong. It
will always return *some* tree, and its numbers -- node counts, branch counts
-- look equally reasonable whether the branches follow real leaves or wander
across a leaf surface. So the two views that matter are the structure in 3D
with each leaf coloured separately, and the skeleton drawn back over the
photographs it came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np

LEAF_PALETTE = np.array(
    [
        [80, 80, 230], [120, 200, 80], [240, 140, 90], [70, 190, 240],
        [220, 100, 200], [210, 210, 90], [90, 140, 240], [150, 220, 90],
    ],
    dtype=np.uint8,
)  # BGR


def write_structure_3d_plot(
    out_path: Union[str, Path],
    points: np.ndarray,
    leaf_ids: Optional[np.ndarray],
    num_leaves: int,
    stem_path: Optional[np.ndarray] = None,
    root_points: Optional[np.ndarray] = None,
    max_points: int = 40000,
) -> Path:
    """Leaf-coloured point cloud plus the traced stem, from three angles.

    Takes plain arrays rather than a structure object, so any backend that can
    produce "a leaf id per point and a stem centreline" can be inspected with
    it without adapting its own types to this function's.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    assignment = np.full(len(points), -1) if leaf_ids is None else np.asarray(leaf_ids)

    if len(points) > max_points:
        pick = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, assignment = points[pick], assignment[pick]

    colors = np.full((len(points), 3), 0.82)
    for leaf_id in range(max(num_leaves, 1)):
        hit = assignment == leaf_id
        if hit.any():
            colors[hit] = LEAF_PALETTE[leaf_id % len(LEAF_PALETTE)][::-1] / 255.0

    stem = None if stem_path is None else np.asarray(stem_path, dtype=float).reshape(-1, 3)

    fig = plt.figure(figsize=(16, 5.4))
    for index, (elev, azim, title) in enumerate(
        [(16, -60, "perspective"), (2, -90, "front"), (89, -90, "from above")]
    ):
        ax = fig.add_subplot(1, 3, index + 1, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1.0, alpha=0.7, linewidths=0)
        if root_points is not None and len(root_points):
            ax.scatter(root_points[:, 0], root_points[:, 1], root_points[:, 2],
                       c="#8C6446", s=0.7, alpha=0.5, linewidths=0)
        if stem is not None and len(stem) > 1:
            ax.plot(stem[:, 0], stem[:, 1], stem[:, 2], color="k", lw=2.4, alpha=0.9)
        # The origin sits on the clamp line, so marking it shows at a glance
        # whether the stem was traced from the right end.
        ax.scatter([0], [0], [0], c="k", marker="x", s=70)

        allp = np.vstack([points, root_points]) if root_points is not None and len(root_points) else points
        span = (allp.max(axis=0) - allp.min(axis=0)).max() / 2.0
        mid = (allp.max(axis=0) + allp.min(axis=0)) / 2.0
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)
        for setter in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
            setter([])

    fig.suptitle(
        f"P5: {num_leaves} leaf instances (colour), stem (black), "
        f"roots (brown), origin at the clamp line (x)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def write_reprojected_skeleton(
    out_dir: Union[str, Path],
    stem_path: np.ndarray,
    leaf_axes: Sequence[np.ndarray],
    frame,
    reconstruction,
    frames_dir: Union[str, Path],
    n_samples: int = 4,
) -> List[Path]:
    """Draw the stem centreline and leaf axes back onto the source photographs.

    The decisive check. In 3D a structure that has wandered off the plant
    still looks like a tidy tree; over the original image it either follows
    the stem and runs down the middle of each leaf, or it visibly does not.

    Takes the polylines directly, in the plant frame, rather than a structure
    object -- so it works for whatever backend produced them.
    """
    out_dir, frames_dir = Path(out_dir), Path(frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_path = np.asarray(stem_path, dtype=float).reshape(-1, 3)
    inverse_rotation = frame.rotation.T

    def to_world(points_plant: np.ndarray) -> np.ndarray:
        return points_plant @ inverse_rotation.T + frame.origin

    image_ids = sorted(reconstruction.reg_image_ids())
    picks = np.linspace(0, len(image_ids) - 1, min(n_samples, len(image_ids))).astype(int)

    written: List[Path] = []
    for i in picks:
        image = reconstruction.images[image_ids[i]]
        camera = reconstruction.cameras[image.camera_id]
        canvas = cv2.imread(str(frames_dir / image.name))
        if canvas is None:
            continue

        world_to_camera = np.eye(4)
        world_to_camera[:3, :] = image.cam_from_world().matrix()
        K = np.asarray(camera.calibration_matrix())

        def project(points_plant: np.ndarray) -> np.ndarray:
            cam = to_world(points_plant) @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
            depth = np.where(cam[:, 2] > 1e-6, cam[:, 2], 1.0)
            return (cam[:, :2] / depth[:, None]) @ K[:2, :2].T + K[:2, 2]

        for leaf_id, axis in enumerate(leaf_axes):
            axis = np.asarray(axis, dtype=float).reshape(-1, 3)
            if len(axis) < 2:
                continue
            pts = project(axis).astype(np.int32)
            color = tuple(int(c) for c in LEAF_PALETTE[leaf_id % len(LEAF_PALETTE)])
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, 4, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts[-1]), 7, color, -1, cv2.LINE_AA)

        if len(stem_path) > 1:
            stem = project(stem_path).astype(np.int32)
            cv2.polylines(canvas, [stem.reshape(-1, 1, 2)], False, (255, 255, 255), 6, cv2.LINE_AA)
            cv2.polylines(canvas, [stem.reshape(-1, 1, 2)], False, (20, 20, 20), 3, cv2.LINE_AA)

        origin = project(np.zeros((1, 3))).astype(np.int32)[0]
        cv2.drawMarker(canvas, tuple(origin), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 26, 4)
        cv2.drawMarker(canvas, tuple(origin), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 26, 2)

        label = f"{image.name}: stem (black/white), {len(leaf_axes)} leaf axes, origin (x)"
        cv2.putText(canvas, label, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 5)
        cv2.putText(canvas, label, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

        path = out_dir / f"skeleton_{Path(image.name).stem}.jpg"
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(path)
    return written
