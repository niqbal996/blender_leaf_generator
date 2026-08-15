"""Rotate the P5 structure in 3D, to judge whether the tips are real.

    pose-view-structure --workdir runs/plant_9/

A tip is a claim about anatomy, and a fixed 2D render cannot support or refute
it: a marker that looks like it sits at the end of a blade from one angle is
often floating in front of a different leaf entirely. Turning the cloud settles
it in seconds.

Keys:
    1   colour by leaf instance          2   colour by geodesic depth from stem
    t   accepted tips on/off             r   rejected candidate tips on/off
    m   midribs on/off                   s   stem and root on/off
    [ ] step through leaves one at a time (the rest fade back)
    a   show every leaf again            h   print this list
    q   close

What to look for. An accepted tip should sit at the far end of a blade, not in
the middle of one and not in mid-air. Two accepted tips on what is visibly a
single leaf means `--merge-cut` is too low; one tip serving two obviously
separate blades means it is too high. Rejected candidates (grey) clustered
around an accepted tip are the normal case -- one ragged blade end offers
several, and merging them is the point.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from pose_estimator.structure_viz import LEAF_PALETTE, load_structure_view

HELP = """
  1 instances   2 depth      t tips      r rejected tips
  m midribs     s stem/root  [ ] leaf    a all leaves   h help   q quit
"""


def run(workdir: Path, max_points: int = 30000, point_size: float = 1.2) -> None:
    import matplotlib

    backend = matplotlib.get_backend()
    if backend.lower() in ("agg", "pdf", "ps", "svg", "template"):
        raise SystemExit(
            f"matplotlib is using the non-interactive '{backend}' backend, so no window "
            "can open.\nInstall a GUI backend (tkinter ships with most Python builds) or "
            "check that $DISPLAY is set.\nThe static panels are still written to "
            f"{workdir / 'p5' / 'diag'}.")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    view = load_structure_view(workdir)
    print(f"  {len(view.leaf_points)} leaf points, {view.num_leaves} instance(s), "
          f"{len(view.accepted_tips)} accepted tip(s) of {len(view.candidate_tips)} candidates")
    print(HELP)

    rng = np.random.default_rng(0)
    show = np.arange(len(view.leaf_points))
    if len(show) > max_points:
        show = rng.choice(len(show), max_points, replace=False)

    state = {"colour": "instance", "tips": True, "rejected": True,
             "midribs": True, "stem": True, "isolate": -1}

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    def draw():
        elev, azim = ax.elev, ax.azim
        ax.clear()
        pts = view.leaf_points
        sel = show
        if state["isolate"] >= 0:
            member = view.leaf_ids == state["isolate"]
            other = show[~member[show]]
            if len(other):
                ax.scatter(pts[other, 0], pts[other, 1], pts[other, 2],
                           c="#DDE2DE", s=point_size * 0.5, alpha=0.12, linewidths=0)
            sel = np.nonzero(member)[0]
            if len(sel) > max_points:
                sel = rng.choice(sel, max_points, replace=False)

        if state["colour"] == "depth":
            finite = np.isfinite(view.depth[sel])
            ok, bad = sel[finite], sel[~finite]
            if len(ok):
                ax.scatter(pts[ok, 0], pts[ok, 1], pts[ok, 2],
                           c=view.depth[ok] / view.voxel, cmap="viridis",
                           s=point_size, alpha=0.9, linewidths=0)
            if len(bad):
                ax.scatter(pts[bad, 0], pts[bad, 1], pts[bad, 2], c="magenta",
                           s=point_size * 3, linewidths=0)
        else:
            colours = np.full((len(pts), 3), 0.82)
            for leaf_id in range(max(view.num_leaves, 1)):
                hit = view.leaf_ids == leaf_id
                if hit.any():
                    colours[hit] = LEAF_PALETTE[leaf_id % len(LEAF_PALETTE)][::-1] / 255.0
            ax.scatter(pts[sel, 0], pts[sel, 1], pts[sel, 2], c=colours[sel],
                       s=point_size, alpha=0.85, linewidths=0)

        if state["stem"]:
            if len(view.stem_points):
                s = view.stem_points
                ax.scatter(s[:, 0], s[:, 1], s[:, 2], c="#A03CB4", s=point_size * 0.8,
                           alpha=0.5, linewidths=0)
            if len(view.root_points):
                r = view.root_points
                ax.scatter(r[:, 0], r[:, 1], r[:, 2], c="#C9740F", s=point_size * 0.8,
                           alpha=0.5, linewidths=0)
            if len(view.stem_path) > 1:
                ax.plot(view.stem_path[:, 0], view.stem_path[:, 1], view.stem_path[:, 2],
                        color="k", lw=3.0)

        if state["midribs"]:
            for i, axis in enumerate(view.axes):
                if state["isolate"] >= 0 and i != state["isolate"]:
                    continue
                ax.plot(axis[:, 0], axis[:, 1], axis[:, 2], color="k", lw=2.4)
                ax.scatter(*axis[0], color="k", s=34, marker="s")   # attachment

        if state["rejected"] and len(view.candidate_tips):
            accepted = set(int(t) for t in view.accepted_tips)
            rej = [int(t) for t in view.candidate_tips if int(t) not in accepted]
            if rej:
                ax.scatter(pts[rej, 0], pts[rej, 1], pts[rej, 2], c="#8A938C",
                           s=40, marker="x", linewidths=1.4)

        if state["tips"]:
            for i, t in enumerate(view.accepted_tips):
                t = int(t)
                if state["isolate"] >= 0 and view.leaf_ids[t] != state["isolate"]:
                    continue
                ax.scatter(*pts[t], color="#0F8A3C", s=150, marker="o",
                           edgecolors="k", linewidths=1.0, depthshade=False)
                ax.text(*pts[t], f"  {i} ({view.depth[t] / view.voxel:.0f}v)", fontsize=8)

        ax.scatter([0], [0], [0], c="k", marker="x", s=80)  # origin, at the clamp line
        span = (pts.max(axis=0) - pts.min(axis=0)).max() / 2.0
        mid = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.view_init(elev=elev, azim=azim)
        for setter in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
            setter([])

        which = ("all leaves" if state["isolate"] < 0
                 else f"leaf {state['isolate']} only "
                      f"({int((view.leaf_ids == state['isolate']).sum())} pts)")
        ax.set_title(f"{workdir.name}   colour: {state['colour']}   {which}\n"
                     f"{view.num_leaves} instances from {len(view.candidate_tips)} "
                     f"candidate tips   (h for keys)", fontsize=10)
        fig.canvas.draw_idle()

    def on_key(event):
        key = (event.key or "").lower()
        if key == "1":
            state["colour"] = "instance"
        elif key == "2":
            state["colour"] = "depth"
        elif key == "t":
            state["tips"] = not state["tips"]
        elif key == "r":
            state["rejected"] = not state["rejected"]
        elif key == "m":
            state["midribs"] = not state["midribs"]
        elif key == "s":
            state["stem"] = not state["stem"]
        elif key == "a":
            state["isolate"] = -1
        elif key == "]":
            state["isolate"] = min(state["isolate"] + 1, view.num_leaves - 1)
        elif key == "[":
            state["isolate"] = max(state["isolate"] - 1, -1)
        elif key == "h":
            print(HELP)
            return
        else:
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path,
                        help="Specimen run directory (needs p5/ from pose-structure)")
    parser.add_argument("--max-points", type=int, default=30000,
                        help="Points drawn at once. Matplotlib's 3D scatter gets sluggish to "
                             "rotate well before the full cloud, and thinning it costs nothing "
                             "for judging where a tip sits.")
    parser.add_argument("--point-size", type=float, default=1.2)
    args = parser.parse_args(argv)
    run(workdir=args.workdir, max_points=args.max_points, point_size=args.point_size)


if __name__ == "__main__":
    main()
