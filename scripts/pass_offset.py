#!/usr/bin/env python
"""Do the two capture passes fuse the same surface? From P4m's pre-merge sample.

    python scripts/pass_offset.py <workdir> [--mm-per-unit 93.08] [--heart x,y,z --heart-mm 15]

For points from pass A: distance to the nearest point from pass B (A != B), against the same
distance within one pass (its even views vs its odd views) -- the noise floor. A cross-pass
distance well above the floor means the plant moved between passes and fusion stacks two
offset sheets.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("workdir", type=Path)
ap.add_argument("--mm-per-unit", type=float)
ap.add_argument("--heart", help="x,y,z in COLMAP world units (compare_clouds scores.json has it in "
                                "the plant frame; omit for the whole plant)")
ap.add_argument("--heart-mm", type=float, default=15.0)
args = ap.parse_args()
z = np.load(args.workdir / "p4m" / "fused_raw_sample.npz")
pts, view, vpass = z["points"].astype(np.float64), z["view"], z["view_pass"]
mm = args.mm_per_unit
if mm is None:
    for f in ("tag_scale.json", "scores.json"):
        try:
            mm = float(json.loads((args.workdir / f).read_text())["mm_per_unit"])
            break
        except (OSError, KeyError, TypeError):
            pass
mm = mm or 1.0
p = vpass[view]
sel = np.ones(len(pts), bool)
if args.heart:
    h = np.array([float(v) for v in args.heart.split(",")])
    sel = np.linalg.norm(pts - h, axis=1) * mm <= args.heart_mm
passes = sorted(set(p[sel].tolist()) - {-1})
print(f"{int(sel.sum()):,} sample points, passes {passes}, {mm:.2f} mm/unit")
rng = np.random.default_rng(0)
for a in passes:
    A = sel & (p == a)
    within = A & (view % 2 == 0), A & (view % 2 == 1)
    q = pts[within[0]][rng.choice(within[0].sum(), min(200_000, within[0].sum()), replace=False)]
    floor = cKDTree(pts[within[1]]).query(q)[0] * mm
    print(f"  pass {a}: within the pass (even vs odd views) median {np.median(floor):.3f} mm, "
          f"p90 {np.percentile(floor, 90):.3f} mm")
    for b in passes:
        if b == a:
            continue
        B = sel & (p == b)
        qa = pts[A][rng.choice(A.sum(), min(200_000, A.sum()), replace=False)]
        d = cKDTree(pts[B]).query(qa)[0] * mm
        print(f"  pass {a} -> pass {b}: median {np.median(d):.3f} mm, p90 {np.percentile(d, 90):.3f} mm")
