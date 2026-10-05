#!/usr/bin/env python
"""Leaf tips and bases found in 2D, where the leaves are clean, then triangulated.

    python scripts/leaf_tips_2d.py --workdir <dataset>/plant --out runs/<specimen>_p5x_views

Diagnostic only. Needs `p5x_views.py` to have run into the same --out (it
reuses the mask cache and the 2D-id -> 3D-leaf association).

Why: P5x traces a tip through the leaf's own 3D points, and those points are
speckled -- on gaensefuss_1 a leaf breaks into 6-66 pieces under the tracer's
3-voxel edge limit -- so 10 of 37 tips came out at the stem end. SAM3's 2D
masks of the same leaves are clean. In 2D:

    base  the mask pixels nearest the stem residual (p2/masks/stem): where
          the blade meets the petiole/stem
    tip   the mask pixel furthest from the base *along the mask* (geodesic),
          so a curved or lobed blade still ends at its apex

Each 2D tip is a ray. A leaf's tip in 3D is where the rays from many views
meet, found by RANSAC so the views where the tip was hidden, clipped or seen
end-on simply fail to agree and drop out. Weights, not rules, carry the
per-view doubt: a tip touching another leaf's mask may be occluded, and a leaf
that looks short in a view is foreshortened there.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

from p5x_views import Run, cache_masks, track_to_leaf, unpack

BORDER_PX = 4           # a tip this close to the frame/crop edge may be clipped
TOUCH_PX = 4            # a tip this close to another leaf's mask may be occluded
TOUCH_WEIGHT = 0.3
INLIER_PX = 8.0         # SAM3 masks come from a 288px head: ~4.4 px quantisation
MIN_RAY_ANGLE = np.deg2rad(5.0)


# --------------------------------------------------------------------------
# 2D
# --------------------------------------------------------------------------


def frame_boxes(workdir: Path) -> dict:
    """{frame: SAM3 tracking crop box}, so a tip on the crop edge is known clipped."""
    crops = json.loads((workdir / "p2" / "crops_per_pass.json").read_text())
    sources = json.loads((workdir / "p1" / "sources.json").read_text())
    by_pass = defaultdict(list)
    for frame, p in sorted(sources.items()):
        by_pass[str(p)].append(frame)
    out = {}
    for p, frames in by_pass.items():
        for frame, box in zip(frames, crops[p]["boxes"]):
            out[frame] = box
    return out


def tips_2d(run: Run, masks, out: Path) -> list:
    cache = out / "tips2d.json"
    if cache.exists():
        return json.loads(cache.read_text())
    boxes = frame_boxes(run.workdir)
    by_frame = defaultdict(list)
    for k, f in enumerate(masks["frame"]):
        by_frame[str(f)].append(k)
    rows = []
    t0 = time.time()
    for frame, ks in sorted(by_frame.items()):
        stem = cv2.imread(str(run.workdir / "p2" / "masks" / "stem" / f"{frame}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if stem is None:
            continue
        h, w = stem.shape
        to_stem = cv2.distanceTransform((stem <= 127).astype(np.uint8), cv2.DIST_L2, 5)
        owner = np.full((h, w), -1, np.int32)          # which mask claims each pixel
        for k in ks:
            x0, y0, x1, y1 = masks["box"][k]
            owner[y0:y1, x0:x1][unpack(masks, k)] = k
        bx0, by0, bx1, by1 = boxes.get(frame, (0, 0, w, h))
        for k in ks:
            x0, y0, x1, y1 = masks["box"][k]
            crop = unpack(masks, k)
            if crop.sum() < 20:
                continue
            d_stem = to_stem[y0:y1, x0:x1]
            inside = np.where(crop, d_stem, np.inf)
            near = inside <= inside.min() + 2.0
            starts = np.argwhere(near)
            cost = np.where(crop, 1.0, np.inf)
            mcp = MCP_Geometric(cost)
            dist, _ = mcp.find_costs(starts.tolist())
            dist = np.where(crop & np.isfinite(dist), dist, -1.0)
            ty, tx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            by, bx = starts.mean(axis=0)
            tip = (float(x0 + tx), float(y0 + ty))
            base = (float(x0 + bx), float(y0 + by))
            # what is around the tip, in the full frame
            r = TOUCH_PX
            ys, xs = slice(max(0, int(tip[1]) - r), int(tip[1]) + r + 1), \
                slice(max(0, int(tip[0]) - r), int(tip[0]) + r + 1)
            around = owner[ys, xs]
            touches = bool(((around >= 0) & (around != k)).any())
            clipped = (tip[0] - bx0 < BORDER_PX or bx1 - 1 - tip[0] < BORDER_PX
                       or tip[1] - by0 < BORDER_PX or by1 - 1 - tip[1] < BORDER_PX)
            rows.append({"k": int(k), "track": str(masks["track"][k]), "frame": frame,
                         "tip": tip, "base": base, "length_px": float(dist.max()),
                         "base_gap_px": float(inside.min()), "touches": touches,
                         "clipped": bool(clipped)})
    # foreshortening: apparent length against this track's longest view
    longest = defaultdict(float)
    for row in rows:
        longest[row["track"]] = max(longest[row["track"]], row["length_px"])
    for row in rows:
        ratio = row["length_px"] / max(longest[row["track"]], 1e-6)
        weight = 0.0 if row["clipped"] else ratio ** 2 * (TOUCH_WEIGHT if row["touches"] else 1.0)
        row["ratio"] = round(ratio, 3)
        row["weight"] = round(weight, 4)
    cache.write_text(json.dumps(rows))
    print(f"  2D tips/bases for {len(rows)} masks ({time.time() - t0:.0f}s); "
          f"{sum(r['clipped'] for r in rows)} clipped, {sum(r['touches'] for r in rows)} touch "
          "another leaf at the tip")
    return rows


def _farthest(mask: np.ndarray, starts) -> tuple:
    """(pixel, distance field) of the mask pixel geodesically farthest from `starts`."""
    dist, _ = MCP_Geometric(np.where(mask, 1.0, np.inf)).find_costs(list(starts))
    dist = np.where(mask & np.isfinite(dist), dist, -1.0)
    return np.unravel_index(int(np.argmax(dist)), dist.shape), dist


def tips_2d_foot(run: Run, masks, out: Path) -> list:
    """Tips by "the leaf end you reach last walking up the plant from its foot".

    No upward/outward rule: the two ends of a leaf are the endpoints of its
    longest geodesic path, and the tip is the one farther from the plant's
    foot *along the plant mask*. An upright leaf lying beside the stem ends
    at its top; an outward or drooping one at its outer end, because the walk
    climbs the stem to the junction first. The foot is P5x's crown projected
    into the frame.
    """
    from p5x_views import project

    cache = out / "tips2d_foot.json"
    if cache.exists():
        return json.loads(cache.read_text())
    boxes = frame_boxes(run.workdir)
    crown = run.to_world(run.skeleton["crown"])
    by_frame = defaultdict(list)
    for k, f in enumerate(masks["frame"]):
        by_frame[str(f)].append(k)
    rows = []
    t0 = time.time()
    for frame, ks in sorted(by_frame.items()):
        if frame not in run.image_of:
            continue
        plant = cv2.imread(str(run.workdir / "p2" / "masks" / "plant" / f"{frame}.png"),
                           cv2.IMREAD_GRAYSCALE) > 127
        h, w = plant.shape
        ys, xs = np.nonzero(plant)
        px0, py0, px1, py1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        sub = plant[py0:py1, px0:px1]
        uv, _ = project(run.camera(frame), crown)
        fy, fx = np.argwhere(sub)[np.argmin(np.hypot(*(np.argwhere(sub) -
                                                        [uv[0, 1] - py0, uv[0, 0] - px0]).T))]
        _, foot_dist = _farthest(sub, [(fy, fx)])
        owner = np.full((h, w), -1, np.int32)
        for k in ks:
            x0, y0, x1, y1 = masks["box"][k]
            owner[y0:y1, x0:x1][unpack(masks, k)] = k
        bx0, by0, bx1, by1 = boxes.get(frame, (0, 0, w, h))
        for k in ks:
            x0, y0, x1, y1 = masks["box"][k]
            crop = unpack(masks, k)
            if crop.sum() < 20:
                continue
            seed = tuple(np.argwhere(crop)[0])
            a, _ = _farthest(crop, [seed])
            b, along = _farthest(crop, [a])
            length = float(along.max())
            # which end is farther from the foot, walking through the plant
            fa = foot_dist[y0 + a[0] - py0, x0 + a[1] - px0]
            fb = foot_dist[y0 + b[0] - py0, x0 + b[1] - px0]
            tip_rc, base_rc = (a, b) if fa > fb else (b, a)
            tip = (float(x0 + tip_rc[1]), float(y0 + tip_rc[0]))
            base = (float(x0 + base_rc[1]), float(y0 + base_rc[0]))
            r = TOUCH_PX
            around = owner[max(0, int(tip[1]) - r):int(tip[1]) + r + 1,
                           max(0, int(tip[0]) - r):int(tip[0]) + r + 1]
            rows.append({"k": int(k), "track": str(masks["track"][k]), "frame": frame,
                         "tip": tip, "base": base, "length_px": length,
                         "foot_margin_px": float(abs(fa - fb)),
                         "touches": bool(((around >= 0) & (around != k)).any()),
                         "clipped": bool(tip[0] - bx0 < BORDER_PX or bx1 - 1 - tip[0] < BORDER_PX
                                         or tip[1] - by0 < BORDER_PX
                                         or by1 - 1 - tip[1] < BORDER_PX)})
    longest = defaultdict(float)
    for row in rows:
        longest[row["track"]] = max(longest[row["track"]], row["length_px"])
    for row in rows:
        ratio = row["length_px"] / max(longest[row["track"]], 1e-6)
        row["ratio"] = round(ratio, 3)
        row["weight"] = round(0.0 if row["clipped"] else
                              ratio ** 2 * (TOUCH_WEIGHT if row["touches"] else 1.0), 4)
    cache.write_text(json.dumps(rows))
    print(f"  foot-rule 2D tips for {len(rows)} masks ({time.time() - t0:.0f}s)")
    return rows


# --------------------------------------------------------------------------
# 3D
# --------------------------------------------------------------------------


class Rays:
    """World-space rays for a set of 2D observations, distortion included."""

    def __init__(self, run: Run, frames, uvs):
        self.R, self.t, self.C, self.d, self.cam_id = [], [], [], [], []
        self.cameras = {}
        for frame, uv in zip(frames, uvs):
            im = run.image_of[frame]
            cam = run.rec.cameras[im.camera_id]
            pose = im.cam_from_world()
            R = pose.rotation.matrix()
            t = np.asarray(pose.translation)
            n = cam.cam_from_img(np.asarray([uv], float))[0]
            d = R.T @ np.array([n[0], n[1], 1.0])
            self.R.append(R)
            self.t.append(t)
            self.C.append(-R.T @ t)
            self.d.append(d / np.linalg.norm(d))
            self.cam_id.append(im.camera_id)
            self.cameras[im.camera_id] = cam
        self.cam_id = np.asarray(self.cam_id)
        self.R, self.t = np.array(self.R), np.array(self.t)
        self.C, self.d = np.array(self.C), np.array(self.d)
        self.uv = np.asarray(uvs, float)

    def project(self, X: np.ndarray) -> np.ndarray:
        cam_xyz = np.einsum("nij,j->ni", self.R, X) + self.t
        out = np.full((len(cam_xyz), 2), np.inf)
        front = cam_xyz[:, 2] > 1e-9
        # one call per camera, not per view: a capture usually has one camera,
        # and RANSAC projects every view for every candidate pair
        for cid, cam in self.cameras.items():
            sel = front & (self.cam_id == cid)
            if sel.any():
                out[sel] = cam.img_from_cam(cam_xyz[sel])
        return out

    def errors(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.project(X) - self.uv, axis=1)


def intersect(C: np.ndarray, d: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares point nearest a bundle of rays."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, v, wi in zip(C, d, w):
        P = np.eye(3) - np.outer(v, v)
        A += wi * P
        b += wi * P @ c
    return np.linalg.lstsq(A, b, rcond=None)[0]


def ransac_point(rays: Rays, weights: np.ndarray, rng, max_pairs: int = 1500):
    n = len(weights)
    if n < 2:
        return None
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        pick = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[p] for p in pick]
    best, best_score = None, -1.0
    for i, j in pairs:
        if np.arccos(np.clip(abs(rays.d[i] @ rays.d[j]), -1, 1)) < MIN_RAY_ANGLE:
            continue
        X = intersect(rays.C[[i, j]], rays.d[[i, j]], np.ones(2))
        err = rays.errors(X)
        inl = err < INLIER_PX
        score = float(weights[inl].sum())
        if score > best_score:
            best, best_score = inl, score
    if best is None or best.sum() < 2:
        return None
    for _ in range(3):                                   # refine on the inliers
        X = intersect(rays.C[best], rays.d[best], np.maximum(weights[best], 1e-3))
        err = rays.errors(X)
        best = err < INLIER_PX
        if best.sum() < 2:
            return None
    return {"xyz": X.tolist(), "inliers": int(best.sum()), "observations": n,
            "rms_px": float(np.sqrt(np.mean(err[best] ** 2))), "inlier_mask": best,
            "errors": err}


def triangulate(run: Run, rows: list, key: str, group_of, rng, passes=None):
    """{group: result} triangulating `key` ('tip' or 'base') of every row in a group.

    One observation per frame per group: when two 2D ids of one frame land in
    the same group, the longer-looking one speaks for it.
    """
    grouped = defaultdict(dict)
    for row in rows:
        g = group_of(row)
        if g is None or row["weight"] <= 0:
            continue
        if passes is not None and run.sources[row["frame"]] not in passes:
            continue
        prev = grouped[g].get(row["frame"])
        if prev is None or row["length_px"] > prev["length_px"]:
            grouped[g][row["frame"]] = row
    out = {}
    for g, by_frame in grouped.items():
        obs = list(by_frame.values())
        rays = Rays(run, [o["frame"] for o in obs], [o[key] for o in obs])
        res = ransac_point(rays, np.array([o["weight"] for o in obs]), rng)
        if res is not None:
            res["frames"] = [o["frame"] for o in obs]
            out[g] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    run = Run(args.workdir)
    masks = cache_masks(args.workdir, args.out)
    assoc = track_to_leaf(run, masks, args.out)
    rows = tips_2d(run, masks, args.out)
    leaf_of = lambda row: assoc.get(row["track"], {}).get("leaf")

    tips = triangulate(run, rows, "tip", leaf_of, rng)
    bases = triangulate(run, rows, "base", leaf_of, rng)

    # Hold one pass out: triangulate from the other two, score the third's 2D tips.
    held = defaultdict(list)
    for p in sorted(set(run.sources.values())):
        others = {q for q in set(run.sources.values()) if q != p}
        part = triangulate(run, rows, "tip", leaf_of, rng, passes=others)
        for leaf, res in part.items():
            mine = [r for r in rows if leaf_of(r) == leaf and run.sources[r["frame"]] == p
                    and r["weight"] >= 0.25]
            if not mine:
                continue
            rays = Rays(run, [r["frame"] for r in mine], [r["tip"] for r in mine])
            err = rays.errors(np.asarray(res["xyz"]))
            held[leaf].append(float(np.median(err)))

    scale = 116.0       # mm per unit, camera-derived and flat-lay rank-matched (see notes)
    gt = json.loads((Path(__file__).resolve().parents[1] / "runs" / "gaensefuss_1_leaves"
                     / "leaves.json").read_text())["leaves"]
    gt_blades = sorted((l["blade_length"] for l in gt if l["area"] > 5), reverse=True)

    sk = {l["id"]: l for l in run.skeleton["leaves"]}
    report, chords = [], []
    print(f"\n  {'leaf':>4} {'obs':>4} {'tip inl':>7} {'rms':>5} {'held-out':>8} "
          f"{'base inl':>8} {'chord mm':>8} {'p5x midrib mm':>13} {'p5x tip off mm':>14}")
    for leaf in sorted(set(tips) | set(bases)):
        t, b = tips.get(leaf), bases.get(leaf)
        chord = (np.linalg.norm(np.subtract(t["xyz"], b["xyz"])) * scale
                 if t is not None and b is not None else float("nan"))
        p5x = sk.get(leaf)
        off = (np.linalg.norm(run.to_world(p5x["tip"]) - np.asarray(t["xyz"])) * scale
               if p5x is not None and t is not None else float("nan"))
        mid = p5x["midrib_length"] * scale if p5x is not None else float("nan")
        h = np.median(held[leaf]) if held.get(leaf) else float("nan")
        print(f"  {leaf:>4} {t['observations'] if t else 0:>4} "
              f"{(str(t['inliers']) if t else '-'):>7} {t['rms_px'] if t else float('nan'):5.1f} "
              f"{h:8.1f} {(str(b['inliers']) if b else '-'):>8} {chord:8.1f} {mid:13.1f} {off:14.1f}")
        chords.append(chord)
        report.append({"leaf": leaf, "tip": t and {k: v for k, v in t.items()
                                                   if k not in ("inlier_mask", "errors")},
                       "base": b and {k: v for k, v in b.items()
                                      if k not in ("inlier_mask", "errors")},
                       "held_out_median_px": h, "chord_mm": chord, "p5x_tip_offset_mm": off})
    valid = sorted((c for c in chords if np.isfinite(c)), reverse=True)
    print(f"\n  held-out-pass tip error, median over leaves: "
          f"{np.nanmedian([np.median(v) for v in held.values()]):.1f} px "
          f"({len(held)} leaves scored)")
    print("  chord lengths (mm) ranked vs flat-lay blades:")
    for i in range(0, len(valid), 8):
        print("    3D " + " ".join(f"{c:5.1f}" for c in valid[i:i + 8]))
        print("    GT " + " ".join(f"{c:5.1f}" for c in gt_blades[i:i + 8]))
    (args.out / "tips3d.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
