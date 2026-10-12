#!/usr/bin/env python
"""Apex leaves read in one key view and lifted to 3D -- no leaf identity across views.

    python scripts/apex_lift.py --workdir <specimen>/plant --out <specimen>/plant/debug/apex_lift \
        --regions <specimen>/plant/debug/apex_labels/apex_regions.npz \
        [--evidence <s2d dir>/evidence_2d.json] [--base-skeleton <s2d dir>]

Step 1 of docs/apex-plan.md, on hand-drawn key-view lines (LabelMe, see
apex_labels.py) -- the oracle: if lines drawn by a person lift to the right
3D, the 3D step works and what is left is finding the lines automatically.

Per labelled frame k (each in turn is the key view) and per leaf line in an
apex box there:

  v1  the line is a fan of camera rays; the leaf was drawn because it was
      fully visible, so along each ray it is the front surface: the depth is
      the nearest cloud depth under that pixel (FRONT_PCT of the cloud
      points within FRONT_PX), filled along the line where the cloud has a
      hole
  v2  that depth bent by 3 numbers per leaf (offset, slope, curvature along
      the leaf) so that in every *other* view the leaf stays inside the
      plant mask and lies on *some* leaf midrib found there (--evidence:
      skeleton_2d.py's per-mask midribs) -- whichever one, no ids

Scores, neither needing identity:

  held out    lifted leaves projected into the *other* labelled frames, paired
              with the lines drawn there (apex_labels.score)
  agreement   the same apex lifted from two key frames: share of leaves with
              a partner within AGREE_MM in 3D, and the median distance

Writes into --out: lift.json (scores), lift_<v>_<frame>/skeleton.json (one per
key frame), lift_<v>_all/skeleton.json (every key frame's leaves, duplicates
merged, plus --base-skeleton's leaves outside the apices: the whole plant),
score_<v>_<frame>.jpg. Every skeleton.json is skeleton_2d's schema, so
`view_in_blender.sh <workdir> --skeleton2d <dir>,<dir>` shows them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apex_labels import apex_of, draw_score, load, score                # noqa: E402
from compare_runs import track_colour                                   # noqa: E402
from p5x_views import Run                                               # noqa: E402
from pose_estimator.ply_io import read_ply_vertices                     # noqa: E402
from skeleton_2d import resample                                        # noqa: E402

SAMPLES = 12        # points along a lifted leaf
FRONT_PX = 4.0      # cloud points within this of a line pixel are under it
FRONT_PCT = 10      # ... and the front surface is this percentile of their depths
PRIOR_MM = 1.5      # v2 may move a leaf this far from its v1 depth before it costs much
VIEW_PX = 5.0       # soft_l1 scale of the per-view terms: occluded views stop pulling past it
AGREE_MM = 1.0      # two lifts of one apex from different key views agree within this
DEDUP_MM = 1.0      # lift_*_all: leaves this close from different key views are one leaf


class View:
    """One frame's camera and, for v2, its identity-free evidence."""

    def __init__(self, run: Run, frame: str):
        im = run.image_of[frame]
        self.cam = run.rec.cameras[im.camera_id]
        pose = im.cam_from_world()
        self.R, self.t = pose.rotation.matrix(), np.asarray(pose.translation)
        self.C = -self.R.T @ self.t
        self.frame = frame
        self.ribs = None
        self.outside = None

    def project(self, X):
        c = np.atleast_2d(X) @ self.R.T + self.t
        return self.cam.img_from_cam(c), c[:, 2]

    def rays(self, uv):
        n = self.cam.cam_from_img(np.asarray(uv, float))
        d = np.c_[n, np.ones(len(n))] @ self.R                    # = R.T @ (x, y, 1)
        return d                                                   # unit depth along the axis

    def point(self, uv, depth):
        return self.C + self.rays(uv) * np.asarray(depth)[:, None]


def front_depth(view: View, cloud: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Per line pixel, the cloud's front-surface depth under it, NaN where none."""
    puv, z = view.project(cloud)
    ok = z > 0
    tree = cKDTree(puv[ok])
    zs = z[ok]
    out = np.full(len(uv), np.nan)
    for i, hits in enumerate(tree.query_ball_point(uv, FRONT_PX)):
        if hits:
            out[i] = np.percentile(zs[hits], FRONT_PCT)
    if np.isnan(out).all():
        return out
    s = np.arange(len(uv))
    good = ~np.isnan(out)
    return np.interp(s, s[good], out[good])


def lift_v1(view: View, cloud: np.ndarray, line: np.ndarray):
    uv = resample(line, SAMPLES)
    depth = front_depth(view, cloud, uv)
    if np.isnan(depth).any():
        return None
    return view.point(uv, depth), uv, depth


def lift_v2(key: View, uv, depth0, others, mpu: float):
    """v1's depths bent by a + b s + c s^2 to agree with the other views, identity-free."""
    s = np.linspace(0.0, 1.0, len(uv))
    rays = key.rays(uv)
    prior = PRIOR_MM / mpu

    def points(x):
        return key.C + rays * (depth0 + x[0] + x[1] * s + x[2] * s * s)[:, None]

    def residuals(x):
        X = points(x)
        out = [x / prior * 3.0]
        for v in others:
            p, z = v.project(X)
            if (z <= 0).any():
                continue
            if v.ribs is not None:
                out.append(v.ribs.query(p)[0])
            if v.outside is not None:
                h, w = v.outside.shape
                xi = np.clip(p[:, 0].round().astype(int), 0, w - 1)
                yi = np.clip(p[:, 1].round().astype(int), 0, h - 1)
                out.append(v.outside[yi, xi])
        return np.concatenate(out)

    fit = least_squares(residuals, np.zeros(3), loss="soft_l1", f_scale=VIEW_PX, max_nfev=60)
    return points(fit.x)


def agreement(a, b, mpu: float):
    """Share of a's leaves with a leaf of b within AGREE_MM (mean 3D curve distance), and
    the median of those nearest distances, mm."""
    if not a or not b:
        return None, None
    trees = [cKDTree(resample(c, 40)) for c in b]
    best = [min(t.query(resample(c, 40))[0].mean() for t in trees) * mpu for c in a]
    return float(np.mean(np.asarray(best) < AGREE_MM)), float(np.median(best))


def as_skeleton(run: Run, leaves, base: dict | None, regions: dict, mpu: float) -> dict:
    """skeleton_2d's schema: lifted leaves (plant frame), plus base's leaves outside the apices."""
    to_plant = lambda X: ((np.atleast_2d(X) - run.origin) @ run.rotation.T)
    doc = {"crown": base["crown"] if base else [0, 0, 0],
           "stem": base["stem"] if base else [], "axes": base.get("axes", []) if base else [],
           "leaves": []}
    if base:
        centres = {n: p.mean(axis=0) for n, p in regions.items()}
        reach = {n: np.percentile(np.linalg.norm(p - centres[n], axis=1), 95) for n, p in regions.items()}
        for leaf in base["leaves"]:
            mid = run.to_world(np.asarray(leaf["midrib"])[len(leaf["midrib"]) // 2])[0]
            if all(np.linalg.norm(mid - centres[n]) > reach[n] for n in regions):
                doc["leaves"].append(dict(leaf, id=len(doc["leaves"])))
    for i, (apex, X) in enumerate(leaves):
        P = to_plant(X)
        doc["leaves"].append({
            "id": len(doc["leaves"]), "track": f"lift_{apex}_{i}", "sam3_ids": [f"lift_{apex}_{i}"],
            "tip": P[-1].tolist(), "base": P[0].tolist(), "midrib": P.tolist(), "petiole": [],
            "petiole_rule": "none (lifted apex leaf: base -> tip as drawn)",
            "colour_bgr": list(track_colour(i)),
            "midrib_mm": float(np.linalg.norm(np.diff(X, axis=0), axis=1).sum() * mpu),
            "tip_inliers": 0, "observations": 1, "fit_px": 0.0})
    return doc


def dedup(leaves, mpu: float):
    """Leaves from several key views: one per physical leaf, the longest of each group."""
    order = sorted(range(len(leaves)), key=lambda i: -np.linalg.norm(np.diff(leaves[i][1], axis=0),
                                                                      axis=1).sum())
    kept = []
    for i in order:
        apex, X = leaves[i]
        dense = resample(X, 40)
        if any(a == apex and cKDTree(resample(Y, 40)).query(dense)[0].mean() * mpu < DEDUP_MM
               for a, Y in kept):
            continue
        kept.append((apex, X))
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--labels", type=Path, help="LabelMe JSONs (default: <workdir>/p1/frames)")
    ap.add_argument("--regions", type=Path, required=True, help="apex_labels.py's apex_regions.npz")
    ap.add_argument("--cloud", type=Path, help="depth cloud (default: <workdir>/p4g/surface_p5x.ply)")
    ap.add_argument("--evidence", type=Path,
                    help="skeleton_2d.py evidence_2d.json whose per-mask midribs v2 checks against "
                         "(default: none -> v2 uses the plant masks only)")
    ap.add_argument("--base-skeleton", type=Path,
                    help="s2d output whose leaves outside the apices complete lift_*_all")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mpu = json.loads((args.workdir / "tag_scale.json").read_text())["mm_per_unit"]
    run = Run(args.workdir)
    labels = load(args.labels or args.workdir / "p1" / "frames")
    regions = dict(np.load(args.regions))

    f = read_ply_vertices(args.cloud or args.workdir / "p4g" / "surface_p5x.ply")
    cloud = np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float64)
    near = np.zeros(len(cloud), bool)
    for pts in regions.values():
        c = pts.mean(axis=0)
        r = np.percentile(np.linalg.norm(pts - c, axis=1), 95)
        near |= np.linalg.norm(cloud - c, axis=1) < 1.5 * r
    cloud = cloud[near]
    print(f"  {len(cloud)} cloud points round the {len(regions)} apices; {len(labels)} labelled frames")

    views = {fr: View(run, fr) for fr in sorted(run.image_of)}
    if args.evidence:
        rows = json.loads(args.evidence.read_text())["leaves"]
        by = defaultdict(list)
        for r in rows:
            by[r["frame"]].append(resample(np.asarray(r["midrib"]), 60))
        for fr, ribs in by.items():
            if fr in views:
                views[fr].ribs = cKDTree(np.vstack(ribs))
        print(f"  evidence: {len(rows)} per-mask midribs over {len(by)} frames from {args.evidence}")
    for fr, v in views.items():
        plant = cv2.imread(str(args.workdir / "p2" / "masks" / "plant" / f"{fr}.png"), 0) > 127
        v.outside = cv2.distanceTransform((~plant).astype(np.uint8), cv2.DIST_L2, 5)

    base = json.loads((args.base_skeleton / "skeleton.json").read_text()) if args.base_skeleton else None
    report = {"settings": {"front_px": FRONT_PX, "front_pct": FRONT_PCT, "prior_mm": PRIOR_MM,
                           "evidence": str(args.evidence) if args.evidence else None},
              "v1": {}, "v2": {}}
    lifted = {"v1": {}, "v2": {}}
    for key, lab in labels.items():
        kv = views[key]
        others = [v for fr, v in views.items() if fr != key]
        for ver in ("v1", "v2"):
            lifted[ver][key] = []
        for _, line in lab["leaves"]:
            apex = apex_of(lab["apices"], line)
            if apex is None:
                continue
            got = lift_v1(kv, cloud, line)
            if got is None:
                continue
            X1, uv, depth = got
            lifted["v1"][key].append((apex, X1))
            lifted["v2"][key].append((apex, lift_v2(kv, uv, depth, others, mpu)))
        for ver in ("v1", "v2"):
            sk = as_skeleton(run, lifted[ver][key], None, regions, mpu)
            held = {fr: l for fr, l in labels.items() if fr != key}
            s, pairs = score(run, held, sk)
            report[ver][key] = {"leaves": len(lifted[ver][key]), "held_out": s}
            d = args.out / f"lift_{ver}_{key}"
            d.mkdir(exist_ok=True)
            (d / "skeleton.json").write_text(json.dumps(as_skeleton(run, lifted[ver][key], base,
                                                                    regions, mpu), indent=1))
            draw_score(run, held, pairs, args.out / f"score_{ver}_{key}.jpg", f"{ver} lifted from {key}")
        print(f"  {key}: {len(lifted['v1'][key])} leaves lifted; held-out found "
              f"v1 {report['v1'][key]['held_out']['all']['found']:.0%} "
              f"(prec {report['v1'][key]['held_out']['all']['precision']:.0%}), "
              f"v2 {report['v2'][key]['held_out']['all']['found']:.0%} "
              f"(prec {report['v2'][key]['held_out']['all']['precision']:.0%})")

    for ver in ("v1", "v2"):
        agree = {}
        keys = sorted(lifted[ver])
        for apex in sorted(regions):
            name = apex
            shares, dists = [], []
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    A = [X for n, X in lifted[ver][a] if n == name]
                    B = [X for n, X in lifted[ver][b] if n == name]
                    sh, md = agreement(A, B, mpu)
                    if sh is not None:
                        shares.append(sh)
                        dists.append(md)
            agree[name] = {"share_within_mm": float(np.mean(shares)) if shares else None,
                           "median_mm": float(np.median(dists)) if dists else None,
                           "pairs_of_key_views": len(shares)}
        report[ver]["agreement"] = agree
        everything = [x for k in keys for x in lifted[ver][k]]
        merged = dedup(everything, mpu)
        d = args.out / f"lift_{ver}_all"
        d.mkdir(exist_ok=True)
        (d / "skeleton.json").write_text(json.dumps(as_skeleton(run, merged, base, regions, mpu),
                                                    indent=1))
        report[ver]["all"] = {"lifted": len(everything), "after_merging": len(merged)}
        print(f"  {ver}: key views agree (share of leaves with a partner within {AGREE_MM} mm, median mm): "
              + ", ".join(f"{a} {v['share_within_mm']:.0%}/{v['median_mm']:.2f}"
                          for a, v in agree.items() if v["share_within_mm"] is not None)
              + f"; {len(everything)} lifts -> {len(merged)} leaves after merging")
    (args.out / "lift.json").write_text(json.dumps(report, indent=1))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
