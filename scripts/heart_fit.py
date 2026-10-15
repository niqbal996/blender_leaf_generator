#!/usr/bin/env python
"""A rosette's heart: its young leaves fitted to the P4g points standing at its origin.

    PYTHONPATH=src:scripts python scripts/heart_fit.py --workdir <rosette>/plant \
        [--fit <workdir>/apex/fit] [--out <workdir>/debug/heart_fit]

The heart is the cluster of young leaves at a rosette's origin (sugarbeet, thistle, weed). The
crown fit (apex_fit.py) misses them: it scores leaves against SAM3's leaf masks, and SAM3 draws
almost none there -- P2's *stem* mask covers the heart in every view instead (young folded leaves
look like stems; heart_evidence.py, 2026-10-15), so P5x labels the heart's P4g points stem.
Those points are this fit's evidence.

Heart points: P4g points (p4g/surface_p5x.ply) P5x labelled stem (p5x_p4g/instances.npy), off
the outer leaves -- farther than one skin gap from every leaf-labelled point (a point within it
belongs to that leaf's shell) -- and off P5x's traced petioles (farther than the petiole's own
radius, measured on its stem points), in the connected pieces that reach the origin.

The skin gap: P4g draws a leaf as two skins (docs/apex-plan.md 0.0d); the median distance
between a leaf point and its partner on the other skin (leaf_instances.skin_partners), measured
on this plant's leaves. It is also the scale of the fit: a point within half of it lies on a
leaf's mid-sheet.

Leaf: apex_fit.py's leaf (12 numbers in the shoot's frame), held at the origin (attach 0), blade
from the base, its midrib turning only towards the axis (a young leaf curls in over the bud,
never out), started as the plant's own leaf (apex_select.leaf_shape_prior) folded like the bud
(apex_fit.BUD_FOLD): upright, folded, elongated.

Objective, in mm, robust (Geman-McClure, scale = half the skin gap):
  coverage  each heart point's distance to the nearest heart-leaf surface point
  support   each leaf's surface points' distance to the nearest heart point (mean per leaf)

Count, greedy: the next leaf starts in the azimuth (about the axis) holding most heart points
no leaf covers yet, all leaves are refitted, and it stays if most of its surface lies on heart
points and most of the heart points it covers are covered by no other leaf (not a twin).
Growing stops at the first leaf that does not stay.

Writes <out>/skeleton.json (the crown fit's, its bud replaced by the heart leaves when there are
any), heart.json, heart_ortho.jpg, heart_photos.jpg.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apex_fit                                                          # noqa: E402
from apex_fit import ALONG, ACROSS, BUD_FOLD, PER_LEAF, START_INCURVE, Apex   # noqa: E402
from pose_estimator import cloud_source                                  # noqa: E402
from pose_estimator.cli.leaf_instances import SKELETON, skin_partners    # noqa: E402
from pose_estimator.ply_io import read_ply_vertices                     # noqa: E402
from pose_estimator.semantic import camera_from_colmap                  # noqa: E402

ITERS = 300


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


class Plant:
    def __init__(self, workdir: Path, fit_dir: Path | None = None):
        """`fit_dir`: a rosette's crown fit (its origin and axis); without one -- or on a plant that is
        no rosette -- the origin is P5x's crown and the axis plant up."""
        self.workdir, self.fit_dir = workdir, fit_dir
        f = read_ply_vertices(workdir / "p4g" / "surface_p5x.ply")
        self.W = np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float64)
        self.lab = np.load(workdir / "p5x_p4g" / "instances.npy").astype(int)
        assert len(self.lab) == len(self.W), "p5x_p4g/instances.npy does not match p4g/surface_p5x.ply"
        g = json.loads((workdir / "p5" / "stem_graph.json").read_text())["plant_frame"]
        self.o, self.R = np.asarray(g["origin"], float), np.asarray(g["rotation"], float)
        self.mpu = json.loads((workdir / "tag_scale.json").read_text())["mm_per_unit"]
        self.p5x = json.loads((workdir / "p5x_p4g" / "skeleton.json").read_text())
        have = fit_dir is not None and (fit_dir / "skeleton.json").exists()
        self.skeleton = json.loads((fit_dir / "skeleton.json").read_text()) if have else {}
        self.fit_report = json.loads((fit_dir / "fit.json").read_text()) \
            if have and (fit_dir / "fit.json").exists() else {}
        if "origin" in self.skeleton:                                        # a rosette's crown fit
            self.O = np.asarray(self.skeleton["origin"], float)              # plant frame
            apex = next(iter(self.skeleton["apices"].values()))
            a = np.asarray(apex["axis"], float)
        else:
            self.O = np.asarray(self.p5x["crown"], float)
            a = np.array([0.0, 0.0, 1.0])
        self.axis = a / np.linalg.norm(a)
        self.P = (self.W - self.o) @ np.linalg.inv(self.R)                   # plant frame
        self.Q = (self.P - self.O) * self.mpu                                # mm about the origin
        self.prior = self.fit_report.get("settings", {}).get("leaf_shape_prior")

    def to_mm(self, plant_xyz):
        return (np.asarray(plant_xyz, float) - self.O) * self.mpu

    def to_plant(self, mm):
        return np.asarray(mm, float) / self.mpu + self.O


def skin_gap(plant: Plant, near_mm: float) -> tuple[float, float]:
    """(median skin gap, point spacing) in mm, on the leaf points within near_mm of the origin."""
    r = np.linalg.norm(plant.Q, axis=1)
    sel = np.flatnonzero((r < near_mm) & (plant.lab >= 0))
    Q, lab = plant.Q[sel], plant.lab[sel]
    partner = skin_partners(Q, np.ones(len(Q), bool))
    ok = (partner >= 0)
    ok[ok] &= lab[partner[ok]] == lab[ok]
    gap = float(np.median(np.linalg.norm(Q[ok] - Q[partner[ok]], axis=1)))
    spacing = float(np.median(cKDTree(Q).query(Q, k=2)[0][:, 1]))
    return gap, spacing


def petiole_tubes(plant: Plant, stem_idx: np.ndarray):
    """Each P5x petiole as (curve mm, radius mm): the radius is the median distance to it of the
    stem points nearer it than to anything else of the skeleton, doubled (a tube's points lie at
    up to its radius; their median, about 0.7 of it, is robust to the heart's points beside it)."""
    tubes = []
    S = plant.Q[stem_idx]
    for leaf in plant.p5x.get("leaves", []):
        pet = np.asarray(leaf.get("petiole") or [], float)
        if len(pet) < 2:
            continue
        C = apex_fit.resample(plant.to_mm(pet), 32)
        if np.linalg.norm(np.diff(C, axis=0), axis=1).sum() <= 0:
            continue
        d, _ = cKDTree(C).query(S)
        near = d[d < np.percentile(d, 5)]
        tubes.append((C, float(2 * np.median(near)) if len(near) else 0.0, int(leaf["id"])))
    return tubes


def heart_points(plant: Plant, gap: float, spacing: float, reach_mm: float):
    """Indices of the heart's points (see the module doc) and a report."""
    r = np.linalg.norm(plant.Q, axis=1)
    near = r < reach_mm
    stem = np.flatnonzero(near & (plant.lab == SKELETON))
    leaf = np.flatnonzero(near & (plant.lab >= 0))
    d_leaf, _ = cKDTree(plant.Q[leaf]).query(plant.Q[stem]) if len(leaf) else (np.full(len(stem), np.inf), None)
    off_leaf = stem[d_leaf > gap]
    tubes = petiole_tubes(plant, off_leaf)
    keep = np.ones(len(off_leaf), bool)
    for C, rad, _ in tubes:
        # only the petiole beyond the heart: its part within one tube radius of the origin is the heart's
        Cb = C[np.linalg.norm(C, axis=1) > rad + gap]
        if len(Cb) >= 2:
            d, _ = cKDTree(Cb).query(plant.Q[off_leaf])
            keep &= d > rad
    cand = off_leaf[keep]
    # connected pieces (edges up to three point spacings), those reaching the origin
    X = plant.Q[cand]
    tree = cKDTree(X)
    pairs = tree.query_pairs(3 * spacing, output_type="ndarray")
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(X), len(X)))
    n, comp = connected_components(A, directed=False)
    d0 = np.linalg.norm(X, axis=1)
    touch = np.unique(comp[d0 <= max(gap, np.sort(d0)[min(len(d0) - 1, 20)])])
    heart = cand[np.isin(comp, touch)]
    rep = {"stem_points_within_reach": int(len(stem)), "off_leaves": int(len(off_leaf)),
           "off_petioles": int(len(cand)), "pieces": int(n), "pieces_at_origin": int(len(touch)),
           "heart_points": int(len(heart)),
           "petiole_radii_mm": [round(t[1], 2) for t in tubes]}
    return heart, rep, tubes


# --------------------------------------------------------------------------
# pictures
# --------------------------------------------------------------------------


def ortho(views, half, px=420, title=""):
    """views: list of (points mm (n,3), BGR colour or (n,3) colours, size); panels top, side x, side y."""
    panels = []
    for axes in [(0, 1), (0, 2), (1, 2)]:
        img = np.zeros((px, px, 3), np.uint8)
        s = px / (2 * half)
        depth = ({0, 1, 2} - set(axes)).pop()
        for X, col, size in views:
            if not len(X):
                continue
            col = np.broadcast_to(np.asarray(col, np.uint8), (len(X), 3))
            o = np.argsort(X[:, depth] * (1 if axes == (0, 1) else -1))
            x = ((X[o, axes[0]] + half) * s).astype(int)
            y = (px - 1 - (X[o, axes[1]] + half) * s).astype(int)
            ok = (x >= 0) & (x < px) & (y >= 0) & (y < px)
            for k in np.flatnonzero(ok):
                if size <= 1:
                    img[y[k], x[k]] = col[o][k]
                else:
                    cv2.circle(img, (int(x[k]), int(y[k])), size, tuple(int(c) for c in col[o][k]), -1)
        cv2.circle(img, (px // 2, px // 2), 4, (0, 255, 255), 1)
        panels.append(img)
    strip = np.concatenate(panels, 1)
    for i, t in enumerate(["top", "side (x, z)", "side (y, z)"]):
        cv2.putText(strip, t, (6 + px * i, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(strip, title, (6, px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return strip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--fit", type=Path, help="the crown fit (default <workdir>/apex/fit)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--stage", default="all", choices=["points", "all"])
    args = ap.parse_args()
    fit_dir = args.fit or args.workdir / "apex" / "fit"
    out = args.out or args.workdir / "debug" / "heart_fit"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    plant = Plant(args.workdir, fit_dir)
    lengths = [np.linalg.norm(np.diff(plant.to_mm(l["midrib"]), axis=0), axis=1).sum()
               for l in plant.p5x["leaves"] if len(l.get("midrib") or []) > 3]
    reach = float(np.median(lengths))                     # apex_regions' crown: one median leaf
    gap, spacing = skin_gap(plant, reach)
    log(f"{args.workdir.parent.name}: skin gap {gap:.2f} mm (median, this plant's leaves), spacing {spacing:.3f} mm; "
        f"reach {reach:.0f} mm")
    heart, rep, tubes = heart_points(plant, gap, spacing, reach)
    log(f"  stem points within reach {rep['stem_points_within_reach']}, off the leaves {rep['off_leaves']}, "
        f"off the petioles {rep['off_petioles']} (radii {rep['petiole_radii_mm']} mm), in {rep['pieces']} pieces, "
        f"{rep['pieces_at_origin']} at the origin: {rep['heart_points']} heart points")
    H = plant.Q[heart]
    hr = np.linalg.norm(H, axis=1)
    half = float(np.percentile(hr, 98)) * 1.2 if len(H) else 10.0
    near = np.linalg.norm(plant.Q, axis=1) < 1.5 * half
    leafy = near & (plant.lab >= 0)
    stem_other = near & (plant.lab == SKELETON)
    views = [(plant.Q[leafy][::3], (90, 60, 60), 1), (plant.Q[stem_other][::2], (130, 130, 130), 1),
             (H, (0, 220, 255), 1)]
    for C, rad, _ in tubes:
        views.append((C, (255, 0, 255), 2))
    for l in plant.p5x["leaves"]:
        views.append((plant.to_mm(l.get("midrib") or np.zeros((0, 3))), (0, 255, 0), 1))
    cv2.imwrite(str(out / "heart_points.jpg"),
                ortho(views, half, title=f"+-{half:.0f} mm; orange heart points, grey other stem points, blue "
                                         f"leaf points, magenta petioles, green midribs"))
    (out / "heart_points.json").write_text(json.dumps(dict(rep, gap_mm=gap, spacing_mm=spacing, reach_mm=reach), indent=1))
    np.save(out / "heart_idx.npy", heart)
    log(f"  {time.time() - t0:.0f}s; wrote {out}/heart_points.jpg")


if __name__ == "__main__":
    main()
