#!/usr/bin/env python
"""A plant skeleton built from SAM3's 2D masks and fused in 3D: tips, bases, midribs, stem.

    python scripts/skeleton_2d.py --workdir <ds>/plant_pass0 \
        --masks runs/<specimen>_vis/masks.npz --other-passes <ds>/plant \
        --out runs/<specimen>_skeleton2d

Diagnostic prototype -- reads a finished run, writes into --out only.

Why 2D first: SAM3's masks are clean, while the 3D labels are not (thick
carved geometry, labels bleeding at leaf borders). So every part of the
skeleton is found where the evidence is clean and only *positioned* in 3D:

  per mask   the leaf's two ends (its longest path through the mask); the tip is
             the end farther from the plant's foot walking through the plant
             mask, the base the other; the midrib is the path between them
             through the middle of the mask.
  per leaf   tip and base triangulated by RANSAC over its frames; the midrib is
             a 3D curve whose projections lie on the 2D midribs of the views
             where tip and base agree.
  stem       the same curve fit, from the crown up to the highest stem-residual
             point; petioles join each leaf base to it.

Leaf identity is SAM3's own track id within one pass -- no 3D voting at all.

Outputs (in --out):
  skeleton.json            P5x's schema, in P5's upright plant frame, + "stem"
  reprojection.jpg         the skeleton drawn on photos it was fitted on, and
                           on other passes' photos it was not
  p5x_vs_2d.jpg            P5x's traced skeleton next to this one, same views
  scores.json              held-out reprojection error, lengths vs flat lay
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric, route_through_array

from compare_runs import junk_masks, track_colour
from leaf_tips_2d import BORDER_PX, TOUCH_PX, TOUCH_WEIGHT, Rays, frame_boxes, ransac_point
from p5x_views import Run, crop_box, unpack

SAMPLES_2D = 24          # points along each 2D midrib
SAMPLES_3D = 40          # points along each 3D curve when it is compared with 2D
MIN_INLIERS = 3          # views that must agree on a tip (and on a base)
MAX_FIT_VIEWS = 12
MM_PER_UNIT = 112.8      # camera-derived for this P3 solve (6.8 px/mm at the plant)
# One leaf's tip estimates spread a few mm with the view; SAM3 id swaps
# measured 13-40 mm. Pieces closer than this are one leaf.
SAME_LEAF_TIP = 10.0 / MM_PER_UNIT


# --------------------------------------------------------------------------
# 2D
# --------------------------------------------------------------------------


def _farthest(mask: np.ndarray, starts) -> tuple:
    dist, _ = MCP_Geometric(np.where(mask, 1.0, np.inf)).find_costs([tuple(s) for s in starts])
    dist = np.where(mask & np.isfinite(dist), dist, -1.0)
    return np.unravel_index(int(np.argmax(dist)), dist.shape), dist


def resample(poly: np.ndarray, n: int) -> np.ndarray:
    """A polyline resampled to n points evenly spaced along its length."""
    poly = np.asarray(poly, float)
    if len(poly) < 2:
        return np.repeat(poly[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return np.repeat(poly[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, poly[:, d]) for d in range(poly.shape[1])], axis=1)


def medial_path(mask: np.ndarray, start, end) -> np.ndarray:
    """(row, col) path from start to end that keeps to the middle of the mask."""
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    cost = np.where(mask, 1.0 / (dt + 0.5), 1e3)
    path, _ = route_through_array(cost, tuple(start), tuple(end), fully_connected=True,
                                  geometric=True)
    return np.asarray(path, float)


def petiole_node(plant, stem, stem_line, base_rc, offset, window=260):
    """Where the leaf's petiole meets the stem line, walking from the blade base.

    The walk is cheap on stem-residual pixels (petioles live there) and dear on
    other plant pixels, so it follows the petiole rather than cutting across a
    neighbouring blade. The node is the first stem-line pixel it reaches.
    """
    if stem_line is None:
        return None
    r0, c0 = int(base_rc[0]), int(base_rc[1])
    h, w = plant.shape
    ys, ye = max(0, r0 - window), min(h, r0 + window)
    xs, xe = max(0, c0 - window), min(w, c0 + window)
    on_line = (stem_line[:, 0] >= ys) & (stem_line[:, 0] < ye) & \
              (stem_line[:, 1] >= xs) & (stem_line[:, 1] < xe)
    if not on_line.any():
        return None
    cost = np.where(plant[ys:ye, xs:xe], np.where(stem[ys:ye, xs:xe], 1.0, 4.0), np.inf)
    start = (min(max(r0 - ys, 0), ye - ys - 1), min(max(c0 - xs, 0), xe - xs - 1))
    if not np.isfinite(cost[start]):
        cost[start] = 4.0
    dist, _ = MCP_Geometric(cost).find_costs([start])
    line = stem_line[on_line] - [ys, xs]
    d = dist[line[:, 0], line[:, 1]]
    if not np.isfinite(d).any():
        return None
    r, c = line[int(np.argmin(d))]
    return [float(c + xs + offset[0]), float(r + ys + offset[1])]


def extract_2d(run: Run, masks, frames, junk, cache: Path):
    """Per mask: tip, base, midrib (full-frame x,y). Per frame: stem path and apex."""
    if cache.exists():
        data = json.loads(cache.read_text())
        return data["leaves"], data["stems"]
    boxes = frame_boxes(run.workdir)
    crown = run.to_world(run.skeleton["crown"])
    by_frame = defaultdict(list)
    for k, f in enumerate(masks["frame"]):
        if str(f) in frames and k not in junk:
            by_frame[str(f)].append(k)
    leaves, stems = [], {}
    t0 = time.time()
    for frame in sorted(by_frame):
        ks = by_frame[frame]
        plant = cv2.imread(str(run.workdir / "p2" / "masks" / "plant" / f"{frame}.png"), 0) > 127
        stem = cv2.imread(str(run.workdir / "p2" / "masks" / "stem" / f"{frame}.png"), 0) > 127
        h, w = plant.shape
        ys, xs = np.nonzero(plant)
        px0, py0, px1, py1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        sub, stem_sub = plant[py0:py1, px0:px1], stem[py0:py1, px0:px1]
        im = run.image_of[frame]
        cam_xyz = im.cam_from_world() * crown
        fuv = run.rec.cameras[im.camera_id].img_from_cam(np.asarray([cam_xyz]))[0]
        inside = np.argwhere(sub)
        foot = inside[np.argmin(np.hypot(inside[:, 0] - (fuv[1] - py0),
                                         inside[:, 1] - (fuv[0] - px0)))]
        _, foot_dist = _farthest(sub, [foot])

        # the stem: foot -> highest stem-residual pixel the foot can reach
        stem_line = None
        reach = stem_sub & (foot_dist >= 0)
        if reach.any():
            rows_ = np.argwhere(reach)
            apex = rows_[np.argmin(rows_[:, 0])]
            dt = cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 5)
            cost = np.where(sub, (1.0 / (dt + 0.5)) * np.where(stem_sub, 1.0, 3.0), 1e3)
            path, _ = route_through_array(cost, tuple(foot), tuple(apex), fully_connected=True,
                                          geometric=True)
            stem_line = np.asarray(path, int)                              # rows, cols in sub
            path = np.asarray(path, float)[:, ::-1] + [px0, py0]           # -> x, y
            stems[frame] = {"path": resample(path, 60).tolist(),
                            "apex": [float(apex[1] + px0), float(apex[0] + py0)],
                            "foot": [float(foot[1] + px0), float(foot[0] + py0)]}

        owner = np.full((h, w), -1, np.int32)
        for k in ks:
            x0, y0, x1, y1 = masks["box"][k]
            owner[y0:y1, x0:x1][unpack(masks, k)] = k
        bx0, by0, bx1, by1 = boxes.get(frame, (0, 0, w, h))
        for k in ks:
            x0, y0, x1, y1 = masks["box"][k]
            crop = unpack(masks, k)
            if crop.sum() < 30:
                continue
            a, _ = _farthest(crop, [np.argwhere(crop)[0]])
            b, along = _farthest(crop, [a])
            fa = foot_dist[y0 + a[0] - py0, x0 + a[1] - px0]
            fb = foot_dist[y0 + b[0] - py0, x0 + b[1] - px0]
            tip_rc, base_rc = (a, b) if fa > fb else (b, a)
            mid = medial_path(crop, base_rc, tip_rc)[:, ::-1] + [x0, y0]    # base -> tip, x,y
            node = petiole_node(sub, stem_sub, stem_line,
                                (y0 + base_rc[0] - py0, x0 + base_rc[1] - px0), [px0, py0])
            tip = (float(x0 + tip_rc[1]), float(y0 + tip_rc[0]))
            r = TOUCH_PX
            around = owner[max(0, int(tip[1]) - r):int(tip[1]) + r + 1,
                           max(0, int(tip[0]) - r):int(tip[0]) + r + 1]
            leaves.append({
                "k": int(k), "track": str(masks["track"][k]), "frame": frame, "tip": tip,
                "base": (float(x0 + base_rc[1]), float(y0 + base_rc[0])),
                "midrib": resample(mid, SAMPLES_2D).tolist(), "length_px": float(along.max()),
                "node": node,
                "touches": bool(((around >= 0) & (around != k)).any()),
                "clipped": bool(tip[0] - bx0 < BORDER_PX or bx1 - 1 - tip[0] < BORDER_PX
                                or tip[1] - by0 < BORDER_PX or by1 - 1 - tip[1] < BORDER_PX)})
    longest = defaultdict(float)
    for row in leaves:
        longest[row["track"]] = max(longest[row["track"]], row["length_px"])
    for row in leaves:
        ratio = row["length_px"] / max(longest[row["track"]], 1e-6)
        row["weight"] = 0.0 if row["clipped"] else ratio ** 2 * (TOUCH_WEIGHT if row["touches"]
                                                                 else 1.0)
    cache.write_text(json.dumps({"leaves": leaves, "stems": stems}))
    print(f"  2D: {len(leaves)} leaf masks and {len(stems)} stem paths over {len(by_frame)} "
          f"frames ({time.time() - t0:.0f}s)")
    return leaves, stems


# --------------------------------------------------------------------------
# 3D
# --------------------------------------------------------------------------


class CurveView:
    """One view of a 2D curve, in normalised camera coordinates (distortion removed)."""

    def __init__(self, run: Run, frame: str, curve_xy, weight: float = 1.0):
        im = run.image_of[frame]
        cam = run.rec.cameras[im.camera_id]
        pose = im.cam_from_world()
        self.R, self.t = pose.rotation.matrix(), np.asarray(pose.translation)
        self.focal = float(cam.mean_focal_length())
        self.curve = cam.cam_from_img(np.asarray(curve_xy, float))
        self.tree = cKDTree(self.curve)
        self.weight = weight
        self.frame = frame

    def project(self, X: np.ndarray) -> np.ndarray:
        c = X @ self.R.T + self.t
        return c[:, :2] / np.maximum(c[:, 2:3], 1e-9)

    def chamfer_px(self, X: np.ndarray) -> np.ndarray:
        """Symmetric distances (px) between the projected 3D curve and the 2D curve."""
        uv = self.project(resample(X, SAMPLES_3D))
        forward = self.tree.query(uv)[0]
        backward = cKDTree(uv).query(self.curve)[0]
        return np.r_[forward, backward] * self.focal


def fit_curve(p0: np.ndarray, p1: np.ndarray, views, n_ctrl: int = 5,
              smooth: float = 2.0, fixed_end: bool = True) -> np.ndarray:
    """3D polyline from p0 to p1 whose projections lie on the views' 2D curves."""
    if len(views) < 2:
        return np.linspace(p0, p1, n_ctrl + 2)
    init = np.linspace(p0, p1, n_ctrl + 2)
    free = init[1:-1] if fixed_end else init[1:]
    scale = np.linalg.norm(p1 - p0) / (n_ctrl + 1)

    def unpack_ctrl(x):
        inner = x.reshape(-1, 3)
        return np.vstack([p0, inner, p1]) if fixed_end else np.vstack([p0, inner])

    def residuals(x):
        ctrl = unpack_ctrl(x)
        out = [np.sqrt(v.weight) * v.chamfer_px(ctrl) for v in views]
        bend = np.diff(ctrl, 2, axis=0).ravel() / max(scale, 1e-9)
        out.append(smooth * bend * 10.0)
        return np.concatenate(out)

    fit = least_squares(residuals, free.ravel(), loss="soft_l1", f_scale=4.0, max_nfev=200)
    return unpack_ctrl(fit.x)


def _spread(views, k: int):
    """Up to k views, best weight first, skipping ones too close in angle to a kept one."""
    kept = []
    for v in sorted(views, key=lambda v: -v.weight):
        axis = v.R[2]
        if all(np.degrees(np.arccos(np.clip(axis @ u.R[2], -1, 1))) > 8.0 for u in kept):
            kept.append(v)
        if len(kept) == k:
            break
    return kept


def _segments(run: Run, obs, rng, max_models: int = 3):
    """A SAM3 id split into the pieces that agree on one tip.

    SAM3 swaps ids between neighbouring leaves mid-pass: on gaensefuss_1 pass 0,
    pass0_0 follows one leaf for frames 0-8 and another 40 mm away for most of
    the rest. One RANSAC per id keeps the majority leaf and loses the other, and
    two swapped ids land on the same leaf as duplicates. So RANSAC is repeated
    on what each model leaves over: every piece that agrees on a tip is a leaf
    candidate of its own.
    """
    remaining, models, tips = list(obs), [], []
    while len(remaining) >= MIN_INLIERS and len(models) < max_models:
        rays = Rays(run, [o["frame"] for o in remaining], [o["tip"] for o in remaining])
        res = ransac_point(rays, np.array([o["weight"] for o in remaining]), rng)
        need = MIN_INLIERS if not models else MIN_INLIERS + 1
        if res is None or res["inliers"] < need:
            break
        inl = [o for o, i in zip(remaining, res["inlier_mask"]) if i]
        remaining = [o for o, i in zip(remaining, res["inlier_mask"]) if not i]
        tip = np.asarray(res["xyz"])
        # A second "tip" a few mm from the first is the same leaf seen from
        # another side -- the farthest point of a silhouette moves with the
        # view. Swaps measured 13-40 mm; a different leaf is far, not near.
        near = [i for i, t in enumerate(tips) if np.linalg.norm(t - tip) < SAME_LEAF_TIP]
        if near:
            models[near[0]] += inl
            continue
        models.append(inl)
        tips.append(tip)
    return models


def _fit_leaf(run: Run, obs, rng, stem):
    """Tip, base, node and midrib of one leaf from the observations that show it."""
    weights = np.array([o["weight"] for o in obs])
    res = {}
    for key in ("tip", "base"):
        rays = Rays(run, [o["frame"] for o in obs], [o[key] for o in obs])
        res[key] = ransac_point(rays, weights, rng)
        if res[key] is None or res[key]["inliers"] < MIN_INLIERS:
            return None
    both = res["tip"]["inlier_mask"] & res["base"]["inlier_mask"]
    views = _spread([CurveView(run, o["frame"], o["midrib"], o["weight"])
                     for o, ok in zip(obs, both) if ok and o["weight"] >= 0.2], MAX_FIT_VIEWS)
    T, B = np.asarray(res["tip"]["xyz"]), np.asarray(res["base"]["xyz"])
    midrib = fit_curve(B, T, views)
    err = np.concatenate([v.chamfer_px(midrib) for v in views]) if views else np.array([])

    # the node: where the petiole meets the stem, triangulated, then put on the stem
    node, node_inliers = None, 0
    with_node = [o for o, ok in zip(obs, both) if ok and o.get("node") is not None]
    if stem is not None and len(with_node) >= MIN_INLIERS:
        rays = Rays(run, [o["frame"] for o in with_node], [o["node"] for o in with_node])
        nres = ransac_point(rays, np.array([o["weight"] for o in with_node]), rng)
        if nres is not None and nres["inliers"] >= MIN_INLIERS:
            dense = resample(stem, 400)
            node = dense[np.argmin(np.linalg.norm(dense - np.asarray(nres["xyz"]), axis=1))]
            node_inliers = nres["inliers"]
    return {"tip": T, "base": B, "midrib": midrib, "node": node, "node_inliers": node_inliers,
            "frames": {o["frame"] for o, ok in zip(obs, both) if ok},
            "views": [v.frame for v in views], "tip_inliers": res["tip"]["inliers"],
            "base_inliers": res["base"]["inliers"], "observations": len(obs),
            "fit_px": float(np.median(err)) if len(err) else float("nan")}


def reconstruct(run: Run, leaves2d, stems2d, frames, rng, merge_tol: float = 6.0 / MM_PER_UNIT,
                has_stem: bool = True):
    """Leaves and the stem, from the 2D evidence of `frames` only.

    1. the stem first, so each leaf's petiole node can be put on it;
    2. every SAM3 id split into the pieces that agree on a tip (`_segments`);
    3. pieces merged when they never claim the same frame and their tips are
       within 6 mm -- and, within one pass, their bases within 8 mm too. One
       leaf carried by swapped ids, or seen by two passes. Across passes the
       base is not compared: whether SAM3's leaf mask takes in part of the
       petiole changes with elevation, so one leaf's base moves between passes
       while its tip does not (sugarbeet_1, 28-09). Within a pass the frame
       test guards leaves seen side by side; across passes nothing can, so two
       leaves whose tips sit within 6 mm in different passes would merge;
    4. every leaf refitted on all of its observations.
    """
    crown = run.to_world(run.skeleton["crown"])
    st = [(f, s) for f, s in stems2d.items() if f in frames]
    stem = None
    if has_stem and len(st) >= MIN_INLIERS:
        rays = Rays(run, [f for f, _ in st], [s["apex"] for _, s in st])
        apex = ransac_point(rays, np.ones(len(st)), rng)
        if apex is not None:
            ok = [(f, s) for (f, s), m in zip(st, apex["inlier_mask"]) if m]
            views = _spread([CurveView(run, f, s["path"]) for f, s in ok], MAX_FIT_VIEWS)
            stem = fit_curve(crown, np.asarray(apex["xyz"]), views, n_ctrl=8, smooth=4.0)

    by_track = defaultdict(dict)
    for row in leaves2d:
        if row["frame"] in frames and row["weight"] > 0:
            prev = by_track[row["track"]].get(row["frame"])
            if prev is None or row["length_px"] > prev["length_px"]:
                by_track[row["track"]][row["frame"]] = row
    pieces, failed = [], {}
    for track, rows in sorted(by_track.items()):
        models = _segments(run, list(rows.values()), rng) if len(rows) >= MIN_INLIERS else []
        fits = [(m, _fit_leaf(run, m, rng, stem)) for m in models]
        fits = [(m, f) for m, f in fits if f is not None]
        if not fits:
            failed[track] = (f"only {len(rows)} usable views" if len(rows) < MIN_INLIERS
                             else "its views do not agree on a tip and a base")
        for i, (m, f) in enumerate(fits):
            pieces.append({"name": f"{track}" + (f"#{i}" if i else ""), "tracks": {track},
                           "obs": m, "fit": f})

    # merge pieces that are the same leaf
    order = sorted(range(len(pieces)), key=lambda i: -pieces[i]["fit"]["tip_inliers"])
    owner = list(range(len(pieces)))
    for a_i, a in enumerate(order):
        for b in order[a_i + 1:]:
            if owner[b] != b or owner[a] != a:
                continue
            fa, fb = pieces[a]["fit"], pieces[b]["fit"]
            same_pass = _pass_of(pieces[a]) == _pass_of(pieces[b])
            if (np.linalg.norm(fa["tip"] - fb["tip"]) < merge_tol
                    and len({o["frame"] for o in pieces[a]["obs"]}
                            & {o["frame"] for o in pieces[b]["obs"]}) <= 1
                    and (not same_pass
                         or np.linalg.norm(fa["base"] - fb["base"]) < 1.33 * merge_tol)):
                owner[b] = a
    leaves = {}
    for i, p in enumerate(pieces):
        if owner[i] != i:
            continue
        group = [pieces[j] for j in range(len(pieces)) if owner[j] == i]
        if len(group) == 1:
            fit = p["fit"]
        else:
            obs = {}
            for g in group:
                for o in g["obs"]:
                    if o["frame"] not in obs or o["weight"] > obs[o["frame"]]["weight"]:
                        obs[o["frame"]] = o
            fit = _fit_leaf(run, list(obs.values()), rng, stem) or p["fit"]
        fit["tracks"] = sorted(set().union(*(g["tracks"] for g in group)))
        fit["pieces"] = [g["name"] for g in group]
        leaves["+".join(g["name"] for g in group)] = fit
    return leaves, failed, stem, crown, len(pieces)


def _pass_of(piece: dict) -> str:
    return sorted(piece["tracks"])[0].split("_", 1)[0]


def petiole(leaf: dict, stem, crown=None):
    """(petiole polyline, which rule placed it).

    A rosette (P5's --architecture) has no stem: its petioles meet at the
    crown, so each runs from the blade base to the crown.

    The petiole continues the leaf's own axis: from the tip through the
    blade-petiole joint (the base) and on until it meets the stem. Snapping the
    base to the nearest stem point instead gave right-angled petioles near the
    crown, where the stem runs beside the leaves.

    "Meets" is the stem point nearest that extended line, accepted when it lies
    ahead of the base, within 0.8 blade lengths (the flat lay's longest petiole
    is 0.57 of its blade) and within a ~19 degree cone of the axis. A leaf whose
    axis never comes near the stem -- a strongly curved petiole -- falls back
    to the node triangulated from the photos, then to the nearest stem point.
    """
    if stem is None:
        if crown is not None:
            return np.vstack([np.asarray(crown), np.asarray(leaf["base"])]), "to the crown (rosette)"
        return np.zeros((0, 3)), "none"
    tip, base = np.asarray(leaf["tip"]), np.asarray(leaf["base"])
    chord = base - tip
    length = float(np.linalg.norm(chord))
    dense = resample(stem, 400)
    if length > 0:
        d = chord / length
        v = dense - base
        ahead = v @ d
        off = np.linalg.norm(v - ahead[:, None] * d, axis=1)
        ok = (ahead > 0) & (ahead <= 0.8 * length) & (off <= np.maximum(0.35 * ahead,
                                                                        3.0 / MM_PER_UNIT))
        if ok.any():
            j = np.flatnonzero(ok)[int(np.argmin(off[ok]))]
            return np.vstack([dense[j], base]), "leaf axis"
    if leaf.get("node") is not None:
        return np.vstack([leaf["node"], base]), "node from photos"
    return np.vstack([dense[np.argmin(np.linalg.norm(dense - base, axis=1))], base]), "nearest"


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------


def _uv(cam, pose, X):
    c = np.asarray(X) @ pose.rotation.matrix().T + np.asarray(pose.translation)
    return cam.img_from_cam(c)


def draw(img, cam, pose, stem, leaves, colours, thin=False):
    w = 1 if thin else 2
    if stem is not None:
        line = _uv(cam, pose, resample(stem, 80)).astype(np.int32)
        cv2.polylines(img, [line], False, (0, 0, 0), 4 + w, cv2.LINE_AA)
        cv2.polylines(img, [line], False, (255, 255, 255), 2 + w, cv2.LINE_AA)
    for key, leaf in leaves.items():
        col = colours[key]
        if len(leaf.get("petiole", [])) >= 2:
            pet = _uv(cam, pose, leaf["petiole"]).astype(np.int32)
            cv2.polylines(img, [pet], False, (210, 210, 210), 1 + w, cv2.LINE_AA)
        rib = _uv(cam, pose, resample(leaf["midrib"], 40)).astype(np.int32)
        cv2.polylines(img, [rib], False, (0, 0, 0), 3 + w, cv2.LINE_AA)
        cv2.polylines(img, [rib], False, col, 1 + w, cv2.LINE_AA)
        tip = tuple(_uv(cam, pose, [leaf["tip"]])[0].astype(int))
        cv2.circle(img, tip, 5 + w, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, tip, 3 + w, col, -1, cv2.LINE_AA)


def tile(run_or_dir, rec, frame, title, stem, leaves, colours, thin=False):
    workdir = run_or_dir
    photo = cv2.imread(str(workdir / "p1" / "frames" / f"{frame}.jpg"))
    img = (photo * 0.55).astype(np.uint8)
    im = next(i for i in rec.images.values() if Path(i.name).stem == frame)
    draw(img, rec.cameras[im.camera_id], im.cam_from_world(), stem, leaves, colours, thin)
    x0, y0, x1, y1 = crop_box(workdir, frame)
    t = img[y0:y1, x0:x1].copy()
    t = cv2.resize(t, (int(t.shape[1] * 900 / t.shape[0]), 900))
    cv2.rectangle(t, (0, 0), (t.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(t, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def grid(tiles, cols):
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    padded = [np.pad(t, ((0, h - t.shape[0]), (0, w - t.shape[1]), (0, 0))) for t in tiles]
    while len(padded) % cols:
        padded.append(np.zeros_like(padded[0]))
    return np.vstack([np.hstack(padded[i:i + cols]) for i in range(0, len(padded), cols)])


# --------------------------------------------------------------------------


def main():
    import pycolmap

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True, help="a single-pass run, e.g. plant_pass0")
    ap.add_argument("--masks", type=Path,
                    help="masks.npz (p5x_views.py's cache); default: built from the workdir's "
                         "p2/masks/leaf_instances into --out")
    ap.add_argument("--passes", type=int, nargs="+",
                    help="use only these capture passes' frames -- the skeleton needs P2 masks, "
                         "P3 poses and P5's crown/frame, not a per-pass 3D run")
    ap.add_argument("--architecture", choices=["upright", "caulescent", "rosette"],
                    help="override P5's (p5/instancing.json); rosette = no stem, petioles to "
                         "the crown")
    ap.add_argument("--other-passes", type=Path, help="full run, to draw on frames not used")
    ap.add_argument("--flat-lay", type=Path, help="leaf-pose output dir with leaves.json")
    ap.add_argument("--retrace-p5x", action="store_true",
                    help="re-trace P5x's skeleton with the current leaf_skeleton.trace for the "
                         "comparison figure, instead of reading p5x/skeleton.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    from pose_estimator import cloud_source
    from pose_estimator.cli.leaf_instances import _architecture
    from p5x_views import cache_masks

    run = Run(args.workdir)
    if args.masks:
        z = np.load(args.masks, allow_pickle=True)
        masks = {k: z[k] for k in z.files}
    else:
        masks = cache_masks(args.workdir, args.out)
    frames = sorted(f for f in run.image_of
                    if args.passes is None or int(run.sources[f]) in set(args.passes))
    architecture = args.architecture or _architecture(args.workdir, cloud_source.BASELINE)
    has_stem = architecture != "rosette"
    print(f"  {len(frames)} frames" + (f" of passes {args.passes}" if args.passes else "")
          + f"; architecture {architecture or 'unknown'} -> "
          + ("stem + petioles off it" if has_stem else "no stem, petioles to the crown"))
    junk = junk_masks(masks, args.workdir, set(frames))
    life = defaultdict(set)
    for t, f in zip(masks["track"], masks["frame"]):
        life[str(t)].add(str(f))
    leaves2d, stems2d = extract_2d(run, masks, set(frames), junk, args.out / "evidence_2d.json")

    # 1. honesty first: fit on even frames, measure on the odd ones it never saw
    even = {f for i, f in enumerate(frames) if i % 2 == 0}
    odd = set(frames) - even
    part, _, _, _, _ = reconstruct(run, leaves2d, stems2d, even, rng, has_stem=has_stem)
    by_track = defaultdict(list)
    for leaf in part.values():
        for t in leaf["tracks"]:
            by_track[t].append(leaf)
    held, fitted = [], []
    for row in leaves2d:
        if row["weight"] < 0.2 or not by_track.get(row["track"]):
            continue
        # an id can carry more than one leaf (SAM3 swaps); score against the one it shows here
        view = CurveView(run, row["frame"], row["midrib"])
        err = min(float(np.median(view.chamfer_px(l["midrib"]))) for l in by_track[row["track"]])
        (held if row["frame"] in odd else fitted).append(err)
    print(f"\n  midrib reprojection error, fitted on even frames only:")
    print(f"    on the even frames it was fitted to:  median {np.median(fitted):.1f} px")
    print(f"    on the odd frames it never saw:       median {np.median(held):.1f} px "
          f"({len(held)} leaf views)")

    # 2. the skeleton itself, from every frame of the pass
    leaves, failed, stem, crown, n_pieces = reconstruct(run, leaves2d, stems2d, set(frames), rng,
                                                        has_stem=has_stem)
    rules = defaultdict(int)
    for leaf in leaves.values():
        leaf["petiole"], leaf["petiole_rule"] = petiole(leaf, stem, crown)
        rules[leaf["petiole_rule"]] += 1
    n_ids = len({r["track"] for r in leaves2d})
    print(f"\n  {n_ids} SAM3 leaf ids -> {n_pieces} consistent pieces (ids split where SAM3 "
          f"swapped leaves) -> {len(leaves)} leaves in 3D after merging pieces of one leaf; "
          f"{len(failed)} ids not reconstructed")
    print("  petioles placed by: " + ", ".join(f"{k} {v}" for k, v in sorted(rules.items())))
    for track, why in sorted(failed.items()):
        print(f"    {track}: {why}")

    lengths = sorted((float(np.linalg.norm(np.diff(l["midrib"], axis=0), axis=1).sum())
                      * MM_PER_UNIT for l in leaves.values()), reverse=True)
    scores = {"held_out_px": float(np.median(held)), "fitted_px": float(np.median(fitted)),
              "leaves": len(leaves), "failed": failed, "midrib_mm": lengths}
    if args.flat_lay:
        gt = json.loads((args.flat_lay / "leaves.json").read_text())["leaves"]
        blades = sorted((l["blade_length"] for l in gt if l["area"] > 5), reverse=True)
        n = min(len(blades), len(lengths))
        print("\n  midrib lengths (mm) ranked against the flat lay's blades:")
        for i in range(0, n, 10):
            print("    3D " + " ".join(f"{v:5.1f}" for v in lengths[i:i + 10]))
            print("    GT " + " ".join(f"{v:5.1f}" for v in blades[i:i + 10]))
        scores["flat_lay_blades_mm"] = blades
    (args.out / "scores.json").write_text(json.dumps(scores, indent=1))

    # 3. P5x's schema, in P5's upright frame, so the Blender viewer can draw it
    to_plant = lambda X: ((np.atleast_2d(X) - run.origin) @ run.rotation.T)
    tracks = sorted(leaves)
    by_id = {t: track_colour(i) for i, t in enumerate(sorted({str(t) for t in masks["track"]}))}
    colours = {name: by_id[leaves[name]["tracks"][0]] for name in leaves}
    doc = {"crown": to_plant(crown)[0].tolist(),
           "stem": to_plant(stem).tolist() if stem is not None else [],
           "leaves": [{"id": i, "track": t, "sam3_ids": leaves[t]["tracks"],
                       "tip": to_plant(leaves[t]["tip"])[0].tolist(),
                       "base": to_plant(leaves[t]["base"])[0].tolist(),
                       "midrib": to_plant(leaves[t]["midrib"]).tolist(),
                       "petiole": to_plant(leaves[t]["petiole"]).tolist(),
                       "colour_bgr": list(colours[t]),
                       "midrib_mm": float(np.linalg.norm(np.diff(leaves[t]["midrib"], axis=0),
                                                         axis=1).sum() * MM_PER_UNIT),
                       "tip_inliers": leaves[t]["tip_inliers"],
                       "observations": leaves[t]["observations"],
                       "fit_px": leaves[t]["fit_px"]} for i, t in enumerate(tracks)]}
    (args.out / "skeleton.json").write_text(json.dumps(doc, indent=1))

    # 4. pictures
    rec0 = run.rec
    shots = [frames[i] for i in np.linspace(2, len(frames) - 3, 4).round().astype(int)]
    tiles = [tile(args.workdir, rec0, f, f"{f}  pass {run.sources[f]}  (fitted)", stem, leaves,
                  colours) for f in shots]
    if args.passes and not args.other_passes:
        # frames of this same workdir that the fit did not use
        rest = sorted(f for f in run.image_of if int(run.sources[f]) not in set(args.passes))
        for p in sorted({int(run.sources[f]) for f in rest})[:2]:
            fs = [f for f in rest if int(run.sources[f]) == p]
            f = fs[len(fs) // 3]
            tiles.append(tile(args.workdir, rec0, f, f"{f}  pass {p}  (NOT used for the fit)",
                              stem, leaves, colours))
    if args.other_passes:
        rec_all = pycolmap.Reconstruction(str(args.other_passes / "p3" / "sparse" / "best"))
        src = json.loads((args.other_passes / "p1" / "sources.json").read_text())
        for p in (1, 2):
            fs = sorted(f for f, q in src.items() if int(q) == p)
            f = fs[len(fs) // 3]
            tiles.append(tile(args.other_passes, rec_all, f,
                              f"{f}  pass {p}  (NOT used; plant drooped since)", stem, leaves,
                              colours))
    cv2.imwrite(str(args.out / "reprojection.jpg"), grid(tiles, 3), [cv2.IMWRITE_JPEG_QUALITY, 88])

    # P5x's traced skeleton next to this one, same views
    p5x = run.skeleton
    if args.retrace_p5x:
        from pose_estimator.leaf_skeleton import trace
        upright = (run.points - run.origin) @ run.rotation.T
        voxel, _ = cloud_source.voxel_size(args.workdir, cloud_source.BASELINE, upright)
        from pose_estimator.cli.leaf_instances import _architecture
        p5x = trace(upright, run.assignment, voxel,
                    architecture=_architecture(args.workdir, cloud_source.BASELINE))
        print(f"  P5x skeleton re-traced with the current tracer: {len(p5x['leaves'])} midribs, "
              f"stem of {len(p5x.get('stem', []))} samples")
    p5x_stem = run.to_world(p5x["stem"]) if len(p5x.get("stem") or []) >= 2 else None
    p5x_leaves = {}
    for leaf in p5x["leaves"]:
        p5x_leaves[leaf["id"]] = {"tip": run.to_world(leaf["tip"]),
                                  "midrib": run.to_world(leaf["midrib"]),
                                  "petiole": run.to_world(leaf["petiole"])
                                  if len(leaf["petiole"]) >= 2 else np.zeros((0, 3))}
    p5x_colours = {k: tuple(int(c) for c in run.colour(k)[::-1]) for k in p5x_leaves}
    pairs = []
    for f in shots[:3]:
        pairs.append(tile(args.workdir, rec0, f, f"{f}  P5x: traced through 3D labels", p5x_stem,
                          {k: v for k, v in p5x_leaves.items() if len(v["midrib"]) >= 2},
                          p5x_colours))
        pairs.append(tile(args.workdir, rec0, f, f"{f}  new: built in 2D, fused in 3D", stem,
                          leaves, colours))
    cv2.imwrite(str(args.out / "p5x_vs_2d.jpg"), grid(pairs, 2), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\n  wrote {args.out / 'reprojection.jpg'}, {args.out / 'p5x_vs_2d.jpg'}, "
          f"{args.out / 'skeleton.json'}")


if __name__ == "__main__":
    main()
