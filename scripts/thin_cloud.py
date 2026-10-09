#!/usr/bin/env python
"""Thin a point cloud to one point per voxel (averaged position, normal, colour).

    python scripts/thin_cloud.py <in.ply> <out.ply> --mm 0.15 --mm-per-unit 93.08

For P5x on the fine clouds: it renders the cloud into every view and builds a
graph over every point, so P4m's 3.6 M / P4g's 12.7 M points are thinned first.
At 0.15 mm they stay ~2x denser than P4b (0.26 mm).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pose_estimator.gs_surface import consolidate                  # noqa: E402
from pose_estimator.ply_io import read_ply_vertices, write_ply_vertices  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src", type=Path)
ap.add_argument("dst", type=Path)
ap.add_argument("--mm", type=float, default=0.15)
ap.add_argument("--mm-per-unit", type=float, required=True, help="the run's scale (tag_scale.json)")
a = ap.parse_args()
f = read_ply_vertices(a.src)
pts = np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float64)
nrm = np.stack([f["nx"], f["ny"], f["nz"]], 1).astype(np.float64) if "nx" in f else np.zeros_like(pts)
col = np.stack([f["red"], f["green"], f["blue"]], 1) if "red" in f else np.full((len(pts), 3), 128, np.uint8)
p, n, c = consolidate(pts, nrm, col, a.mm / a.mm_per_unit)
write_ply_vertices(a.dst, {"x": p[:, 0].astype(np.float32), "y": p[:, 1].astype(np.float32),
                           "z": p[:, 2].astype(np.float32), "nx": n[:, 0].astype(np.float32),
                           "ny": n[:, 1].astype(np.float32), "nz": n[:, 2].astype(np.float32),
                           "red": c[:, 0], "green": c[:, 1], "blue": c[:, 2]}, binary=True)
print(f"{a.src}: {len(pts):,} -> {len(p):,} points at {a.mm} mm -> {a.dst}")
