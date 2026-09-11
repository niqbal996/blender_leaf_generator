#!/usr/bin/env python
"""Look at the labelled plant in matplotlib, interactively or as PNGs.

    # rotatable window (needs a display; WSLg provides one)
    python scripts/view_plant.py --workdir runs/plant_9

    # write stills instead, when there is no display
    python scripts/view_plant.py --workdir runs/plant_9 --save runs/plant_9/view

    # colour by organ rather than by leaf instance
    python scripts/view_plant.py --workdir runs/plant_9 --mode organs

Reads P5's per-leaf assignment when it exists (leaves split at their stem
attachments, which is the better separation) and falls back to P4c's labels
otherwise. The stem centreline and per-leaf axes are drawn over the cloud when
P5 has produced them.

Point clouds this size are slow to rotate in matplotlib, so the display is
decimated by default -- `--max-points` controls it. The decimation is for
drawing only and never touches the data on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ORGAN_COLORS = {"leaf": "#E03C3C", "tiny leaf": "#F05A3C", "stem": "#A03CC8",
                "petiole": "#A03CC8", "root": "#F08C28"}
def distinct_leaf_colors(n, avoid=(0.79, 0.083), margin=0.055):
    """`n` visually separable hues, keeping clear of the stem and root colours.

    A fixed palette runs out of separation quickly -- at 13 leaves the previous
    one repeated three near-identical greens and put one leaf on almost the
    same purple as the stem. Hues are instead spread evenly over the circle
    with the stem (purple) and root (orange) bands excluded, and lightness is
    alternated so neighbouring hues stay distinguishable even when adjacent in
    space.
    """
    import colorsys

    allowed = []
    steps = max(n * 6, 360)
    for i in range(steps):
        hue = i / steps
        if all(min(abs(hue - a), 1.0 - abs(hue - a)) > margin for a in avoid):
            allowed.append(hue)
    if not allowed:
        allowed = [i / max(n, 1) for i in range(max(n, 1))]

    colors = []
    for i in range(n):
        hue = allowed[int(i * len(allowed) / max(n, 1))]
        value = 0.95 if i % 2 == 0 else 0.72
        saturation = 0.85 if i % 3 != 2 else 0.62
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((r, g, b))
    return colors


def leaf_hexes(n):
    return ["#%02X%02X%02X" % tuple(int(255 * c) for c in rgb)
            for rgb in distinct_leaf_colors(n)]


def load(workdir: Path, source: str, geometry_backend: str = "colmap"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from pose_estimator import cloud_source
    from pose_estimator.ply_io import read_ply_vertices

    chosen = cloud_source.resolve(workdir, geometry_backend)
    p4c, p5 = chosen.labels_dir, chosen.structure_dir
    labels = np.load(p4c / "labels.npy")
    votes = np.load(p4c / "votes.npz", allow_pickle=True)
    class_order = [str(x) for x in votes["class_order"]]

    cloud_path = chosen.path
    fields = read_ply_vertices(cloud_path)
    points = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(float)

    data = {"points": points, "labels": labels, "class_order": class_order,
            "stem_path": None, "axes": [], "leaf_points": None, "leaf_ids": None}

    use_p5 = source in ("auto", "p5") and (p5 / "stem_graph.json").exists()
    if use_p5:
        with open(p5 / "stem_graph.json") as f:
            graph = json.load(f)
        if graph.get("stem_path_xyz"):
            data["stem_path"] = np.array(graph["stem_path_xyz"])
        data["axes"] = [np.array(leaf["axis_xyz"]) for leaf in graph.get("leaves", [])]
        if (p5 / "leaf_points.npy").exists():
            data["leaf_ids"] = np.load(p5 / "leaf_points.npy")
            data["leaf_points"] = np.load(p5 / "leaf_points_xyz.npy")

        # P5 works in the upright plant frame; P4c's cloud is in the raw
        # frame. Bring the raw cloud across so both draw in the same space.
        frame = graph.get("plant_frame")
        if frame:
            origin = np.array(frame["origin"])
            rotation = np.array(frame["rotation"])
            data["points"] = (points - origin) @ rotation.T
    return data


def decimate(points, extra, limit):
    if len(points) <= limit:
        return points, extra
    pick = np.random.default_rng(0).choice(len(points), limit, replace=False)
    return points[pick], (None if extra is None else extra[pick])


def draw(ax, data, mode, limit):
    if mode == "instances" and data["leaf_points"] is not None:
        pts, ids = decimate(data["leaf_points"], data["leaf_ids"], limit)
        count = int(ids.max()) + 1 if len(ids) else 0
        palette = leaf_hexes(count)
        for instance in range(count):
            hit = ids == instance
            if hit.any():
                ax.scatter(*pts[hit].T, s=1.0, alpha=0.75, linewidths=0,
                           c=palette[instance],
                           label=f"leaf {instance} ({int(hit.sum())})")
        # Stem and root come from the organ labels; P5 keeps only leaves.
        others = [i for i, n in enumerate(data["class_order"]) if "leaf" not in n]
        for index in others:
            hit = data["labels"] == index
            if not hit.any():
                continue
            sub, _ = decimate(data["points"][hit], None, limit // 3)
            name = data["class_order"][index]
            ax.scatter(*sub.T, s=1.0, alpha=0.6, linewidths=0,
                       c=ORGAN_COLORS.get(name, "#999999"), label=name)
    else:
        pts, labs = decimate(data["points"], data["labels"], limit)
        for index, name in enumerate(data["class_order"]):
            hit = labs == index
            if hit.any():
                ax.scatter(*pts[hit].T, s=1.0, alpha=0.7, linewidths=0,
                           c=ORGAN_COLORS.get(name, "#999999"),
                           label=f"{name} ({int((data['labels'] == index).sum())})")

    if data["stem_path"] is not None and len(data["stem_path"]) > 1:
        ax.plot(*data["stem_path"].T, color="k", lw=3.0, label="stem centreline")
    for axis in data["axes"]:
        if len(axis) > 1:
            ax.plot(*axis.T, color="k", lw=1.0, alpha=0.55)

    allp = data["points"]
    span = (allp.max(axis=0) - allp.min(axis=0)).max() / 2.0
    mid = (allp.max(axis=0) + allp.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    for setter in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
        setter([])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--mode", choices=["instances", "organs"], default="instances")
    p.add_argument("--source", choices=["auto", "p4c", "p5"], default="auto")
    p.add_argument("--geometry-backend", default="colmap",
                   help="Which P3 branch to show: colmap, vggt_omega or mapanything. "
                        "A learned branch reads p4c/experiments/<backend> and "
                        "p5/experiments/<backend>")
    p.add_argument("--max-points", type=int, default=60000,
                   help="points drawn; decimation affects the display only")
    p.add_argument("--save", type=Path, help="write PNGs here instead of opening a window")
    args = p.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    data = load(args.workdir, args.source, args.geometry_backend)
    n_instances = int(data["leaf_ids"].max()) + 1 if data["leaf_ids"] is not None else 0
    print(f"  {len(data['points'])} points, classes {data['class_order']}, "
          f"{n_instances} leaf instance(s)")

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        views = [(18, -60, "perspective"), (2, -90, "front"), (2, 0, "side"), (89, -90, "top")]
        fig = plt.figure(figsize=(19, 5.2))
        for i, (elev, azim, title) in enumerate(views):
            ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
            draw(ax, data, args.mode, args.max_points)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, fontsize=10)
            if i == 0:
                ax.legend(fontsize=6.5, loc="upper left", markerscale=6, framealpha=0.8)
        fig.suptitle(f"{args.workdir.name}  --  {args.mode}", fontsize=13)
        fig.tight_layout()
        out = args.save / f"plant_{args.mode}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  wrote {out}")
        return

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    draw(ax, data, args.mode, args.max_points)
    ax.set_title(f"{args.workdir.name} -- {args.mode}   (drag to rotate, scroll to zoom)")
    ax.legend(fontsize=7, loc="upper left", markerscale=6)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
