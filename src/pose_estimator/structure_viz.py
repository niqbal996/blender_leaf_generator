"""P5 diagnostics: is this actually the plant's structure, or just a graph?

A skeleton is the easiest thing in the pipeline to get plausibly wrong. It
will always return *some* tree, and its numbers -- node counts, branch counts
-- look equally reasonable whether the branches follow real leaves or wander
across a leaf surface. So the two views that matter are the structure in 3D
with each leaf coloured separately, and the skeleton drawn back over the
photographs it came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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


@dataclass
class StructureView:
    """Everything an inspector needs, read back from a finished P5 run.

    Loading from the artifacts rather than recomputing is deliberate: a viewer
    that re-derived the instancing could disagree with what the pipeline
    actually wrote, which is precisely the thing you would be trying to check.
    """

    leaf_points: np.ndarray            # (N, 3) in the plant frame
    leaf_ids: np.ndarray               # (N,) instance id, -1 = unassigned
    depth: np.ndarray                  # (N,) geodesic distance from the stem
    candidate_tips: np.ndarray         # indices into leaf_points
    accepted_tips: np.ndarray
    axes: List[np.ndarray] = field(default_factory=list)      # midribs, base -> tip
    stem_path: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    stem_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    root_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    voxel: float = 1.0

    @property
    def num_leaves(self) -> int:
        return len(self.axes)


def load_structure_view(workdir: Union[str, Path]) -> StructureView:
    """Read a finished P5 run back into memory for inspection."""
    from .ply_io import read_ply_vertices

    workdir = Path(workdir)
    p5, p4c = workdir / "p5", workdir / "p4c"
    missing = [p for p in (p5 / "leaf_points_xyz.npy", p5 / "leaf_points.npy",
                           p5 / "stem_graph.json") if not p.exists()]
    if missing:
        raise SystemExit(
            "missing " + ", ".join(str(m) for m in missing)
            + "\nrun pose-structure on this workdir first")

    leaf_points = np.load(p5 / "leaf_points_xyz.npy")
    leaf_ids = np.load(p5 / "leaf_points.npy")
    depth = (np.load(p5 / "leaf_depth.npy") if (p5 / "leaf_depth.npy").exists()
             else np.full(len(leaf_points), np.nan))

    candidate = accepted = np.zeros(0, np.int64)
    if (p5 / "tips.npz").exists():
        tips = np.load(p5 / "tips.npz")
        candidate, accepted = tips["candidate"], tips["accepted"]

    with open(p5 / "stem_graph.json") as f:
        graph = json.load(f)
    axes = [np.array(leaf["axis_xyz"]) for leaf in graph.get("leaves", [])]
    stem_path = np.array(graph.get("stem_path_xyz") or []).reshape(-1, 3)

    voxel = 1.0
    if (workdir / "p4" / "hull.json").exists():
        with open(workdir / "p4" / "hull.json") as f:
            voxel = json.load(f)["voxel_size"]

    # Stem and root are not stored separately -- rebuild them from the labelled
    # cloud through the plant frame P5 recorded, so the viewer shows the same
    # transform the structure was solved in.
    stem_points = root_points = np.zeros((0, 3))
    surface = workdir / "p4b" / "surface.ply"
    hull = workdir / "p4" / "hull_points.ply"
    cloud_path = surface if surface.exists() else hull
    if cloud_path.exists() and (p4c / "labels.npy").exists() and "plant_frame" in graph:
        fields = read_ply_vertices(cloud_path)
        cloud = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(float)
        labels = np.load(p4c / "labels.npy")
        if len(labels) == len(cloud):
            order = [str(x) for x in
                     np.load(p4c / "votes.npz", allow_pickle=True)["class_order"]]
            frame = graph["plant_frame"]
            origin = np.array(frame["origin"])
            rotation = np.array(frame["rotation"])
            upright = (cloud - origin) @ rotation.T
            stem_ids = [i for i, n in enumerate(order) if n in ("stem", "petiole", "branch")]
            root_ids = [i for i, n in enumerate(order) if n == "root"]
            stem_points = upright[np.isin(labels, stem_ids)]
            root_points = upright[np.isin(labels, root_ids)]

    return StructureView(
        leaf_points=leaf_points, leaf_ids=leaf_ids, depth=depth,
        candidate_tips=candidate, accepted_tips=accepted, axes=axes,
        stem_path=stem_path, stem_points=stem_points, root_points=root_points,
        voxel=voxel,
    )


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


def write_instancing_plot(
    out_path: Union[str, Path], structure, voxel: float, max_points: int = 40000,
) -> Optional[Path]:
    """The three steps of the leaf split, side by side.

    Left to right: what the split is computed from, what it decided, and what
    came out. A wrong leaf count is almost always visible in the first two --
    either the depth field is wrong (islands, or the stem seed in the wrong
    place) or the tips are, and the final panel alone cannot tell you which.
    """
    inst = getattr(structure, "instancing", None)
    if inst is None or len(structure.leaf_points) == 0:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pts = structure.leaf_points
    show = np.arange(len(pts))
    if len(pts) > max_points:
        show = np.random.default_rng(0).choice(len(pts), max_points, replace=False)

    depth = inst.depth
    finite = np.isfinite(depth)
    fig = plt.figure(figsize=(17, 5.6))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ok = show[finite[show]]
    bad = show[~finite[show]]
    if len(ok):
        ax.scatter(pts[ok, 0], pts[ok, 1], pts[ok, 2], c=depth[ok] / voxel,
                   cmap="viridis", s=0.8, alpha=0.85, linewidths=0)
    if len(bad):
        ax.scatter(pts[bad, 0], pts[bad, 1], pts[bad, 2], c="magenta", s=1.6, linewidths=0)
    ax.set_title(f"1. geodesic depth from the stem\n"
                 f"{'magenta = unreachable island' if len(bad) else 'all tissue reachable'}",
                 fontsize=9.5)

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.scatter(pts[show, 0], pts[show, 1], pts[show, 2], c="#D3DAD5", s=0.5,
               alpha=0.35, linewidths=0)
    accepted = set(int(t) for t in inst.accepted_tips)
    rejected = [int(t) for t in inst.candidate_tips if int(t) not in accepted]
    if rejected:
        ax.scatter(pts[rejected, 0], pts[rejected, 1], pts[rejected, 2],
                   c="#9AA3A8", s=26, marker="x", linewidths=1.2)
    if len(inst.accepted_tips):
        keep = inst.accepted_tips.astype(int)
        ax.scatter(pts[keep, 0], pts[keep, 1], pts[keep, 2],
                   c="#1F9E4B", s=90, marker="o", edgecolors="k", linewidths=0.8)
    ax.set_title(f"2. tips: {len(inst.candidate_tips)} candidates -> "
                 f"{len(inst.accepted_tips)} leaves\n(grey x = merged into another tip)",
                 fontsize=9.5)

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    colors = np.full((len(pts), 3), 0.82)
    for leaf_id in range(max(structure.num_leaves, 1)):
        hit = inst.owner == leaf_id
        if hit.any():
            colors[hit] = LEAF_PALETTE[leaf_id % len(LEAF_PALETTE)][::-1] / 255.0
    ax.scatter(pts[show, 0], pts[show, 1], pts[show, 2], c=colors[show], s=0.8,
               alpha=0.8, linewidths=0)
    for axis in structure.axes:
        ax.plot(axis[:, 0], axis[:, 1], axis[:, 2], color="k", lw=2.0)
    ax.set_title(f"3. {structure.num_leaves} instances, grown inward from their tips\n"
                 f"(black = midrib, base to tip)", fontsize=9.5)

    for ax in fig.axes:
        _equalise(ax, pts[show])
        ax.view_init(elev=16, azim=-60)
        for setter in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
            setter([])

    fig.suptitle("P5 leaf instancing: tips separate what attachments cannot", fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _equalise(ax, points: np.ndarray) -> None:
    span = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    mid = (points.max(axis=0) + points.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)


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
