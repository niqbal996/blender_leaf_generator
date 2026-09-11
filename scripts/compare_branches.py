#!/usr/bin/env python3
"""Put the geometry branches side by side on the deliverable, not on proxies.

    ./scripts/compare_branches.py runs/thistle3/plant
    ./scripts/compare_branches.py runs/thistle3/plant --branches colmap,mapanything

Every earlier comparison between COLMAP, VGGT-Omega and MapAnything was made
on geometry proxies -- silhouette IoU, pose residuals, how the cloud looks.
Those say whether a reconstruction is self-consistent, not whether it yields
leaves. This reads the end of each branch instead: how many leaves P5 found,
how many P6 could fit a midrib to, and how the fitted midribs behave.

The one trap here is scale. The backends reconstruct at unrelated scales -- on
thistle3 the orbit radius is 3.69 for COLMAP, 2.85 for MapAnything and 0.86
for VGGT-Omega -- and none of them is metric. So an arclength of 0.66 in one
branch and 0.18 in another says nothing at all. Every length below is
therefore divided by its own branch's cloud extent, which makes it a shape
measurement and comparable; the raw extent is printed alongside so the scale
difference stays visible rather than being quietly normalised away.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEFAULT_BRANCHES = ("colmap", "vggt_omega", "mapanything")


def read_json(path: Path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def extent_of(cloud_path: Path):
    """The branch's own length unit: the longest side of its cloud's bounds."""
    from pose_estimator.ply_io import read_ply_vertices
    import numpy as np

    if not cloud_path.exists():
        return None
    fields = read_ply_vertices(cloud_path)
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(float)
    if not len(xyz):
        return None
    return float((xyz.max(axis=0) - xyz.min(axis=0)).max()), len(xyz)


def failed_checks(report) -> list:
    if not report:
        return []
    return [name for name, check in (report.get("checks") or {}).items()
            if not check.get("pass", True)]


def collect(workdir: Path, backend: str) -> dict:
    from pose_estimator import cloud_source

    chosen = cloud_source.resolve(workdir, backend)
    p5_dir = chosen.structure_dir
    p6_dir = (workdir / "p6" if backend == cloud_source.BASELINE
              else workdir / "p6" / "experiments" / backend)

    # A branch counts as run when it produced a P5 result, not when its P3
    # cloud exists: P5's structure.ply is what the deliverable is built from
    # and what the Blender comparison draws, so measuring it here keeps the
    # table and the picture talking about the same points.
    structure = p5_dir / "structure.ply"
    row = {"backend": backend, "cloud": chosen.path, "origin": chosen.origin,
           "structure": structure, "ran": structure.exists()}
    measured = extent_of(structure)
    if measured:
        row["extent"], row["points"] = measured

    row["p5"] = read_json(p5_dir / "p5.json")
    row["p6"] = read_json(p6_dir / "p6.json")
    row["leaves"] = read_json(p6_dir / "leaves.json")
    return row


def value(row, *path, default=None):
    node = row
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def table(rows: list) -> str:
    scale = {r["backend"]: r.get("extent") for r in rows}

    def relative(row, raw):
        unit = scale.get(row["backend"])
        return None if raw is None or not unit else raw / unit

    lines = [
        ("cloud points", lambda r: r.get("points")),
        ("cloud extent (own units)", lambda r: r.get("extent")),
        ("P5 leaf instances", lambda r: value(r, "p5", "num_leaves")),
        ("P6 midribs fitted", lambda r: value(r, "p6", "num_leaves")),
        ("mean arclength / extent", lambda r: relative(r, value(r, "p6", "arclength", "mean"))),
        ("longest leaf / extent", lambda r: relative(r, value(r, "p6", "arclength", "max"))),
        ("mean insertion angle deg", lambda r: value(r, "p6", "insertion_angle_deg", "mean")),
        ("max frame step deg", lambda r: value(r, "p6", "max_frame_step_deg")),
        ("worst tip shortfall vox", lambda r: max(value(r, "p5", "tip_shortfall_voxels",
                                                        default=[0]) or [0])),
        ("P5 checks failed", lambda r: len(failed_checks(r.get("p5")))),
        ("P6 checks failed", lambda r: len(failed_checks(r.get("p6")))),
    ]

    width = max(len(name) for name, _ in lines) + 2
    header = "".ljust(width) + "".join(r["backend"].rjust(16) for r in rows)
    out = [header, "-" * len(header)]
    for name, getter in lines:
        cells = []
        for row in rows:
            try:
                measurement = getter(row) if row["ran"] else None
            except Exception:
                measurement = None
            if measurement is None:
                cells.append("-".rjust(16))
            elif isinstance(measurement, float):
                cells.append(f"{measurement:.4g}".rjust(16))
            else:
                cells.append(str(measurement).rjust(16))
        out.append(name.ljust(width) + "".join(cells))
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--branches", default=",".join(DEFAULT_BRANCHES),
                        help="comma-separated geometry backends to compare")
    parser.add_argument("--json", type=Path, help="also write the rows here")
    args = parser.parse_args()

    backends = [b.strip() for b in args.branches.split(",") if b.strip()]
    rows = [collect(args.workdir, backend) for backend in backends]

    print(f"\nBranch comparison -- {args.workdir}\n")
    for row in rows:
        state = "ok" if row["ran"] else "NOT RUN"
        print(f"  {row['backend']:<12} {state:<8} {row['origin']}")
    print()
    print(table(rows))
    print("""
Lengths are divided by each branch's own cloud extent, because the backends
reconstruct at unrelated and non-metric scales -- the raw extent row is what
those scales actually are. Leaf counts, angles and check failures need no
such treatment: they are already dimensionless.""")

    missing = [r["backend"] for r in rows if not r["ran"]]
    if missing:
        print(f"\n  not on disk: {', '.join(missing)} -- run "
              f"./run_pipeline.sh <dataset> --compare {args.branches}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump([{k: (str(v) if isinstance(v, Path) else v)
                        for k, v in row.items()} for row in rows], handle, indent=2)
        print(f"\n  rows -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
