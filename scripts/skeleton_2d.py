#!/usr/bin/env python
"""A plant skeleton built from SAM3's 2D masks and fused in 3D: tips, bases, midribs, stem.

    python scripts/skeleton_2d.py --workdir <ds>/plant_pass0 \
        --masks runs/<specimen>_vis/masks.npz --other-passes <ds>/plant \
        --flat-lay runs/<specimen>_leaves --out runs/<specimen>_skeleton2d

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
from pose_estimator import plant_profiles
from leaf_tips_2d import BORDER_PX, TOUCH_PX, TOUCH_WEIGHT, Rays, frame_boxes, ransac_point
from p5x_views import Run, crop_box, unpack

SAMPLES_2D = 24          # points along each 2D midrib
SAMPLES_3D = 40          # points along each 3D curve when it is compared with 2D
MIN_INLIERS = 3          # views that must agree on a tip (and on a base)
MAX_FIT_VIEWS = 12
# mm per reconstruction unit. Structure from motion has no scale of its own and
# every P3 solve has a different one, so main() sets this per run (set_scale):
# --mm-per-unit, the AprilTags in the scene (--tag-mm), or fitted against the
# flat lay. 112.8 was gaensefuss_1's solve; nothing uses it once main() runs.
MM_PER_UNIT = 112.8
# Where the flat-lay fit starts: camera-to-crown distance on the boom rig. From
# sugarbeet_x_1 pass 2: 8 mm tags at ~110 px in 6000 px frames, f ~8200 px.
# A seed only -- it sets the first fit's merge thresholds, not the answer.
RIG_CAMERA_MM = 600.0
# A leaf's identity is the centre of its blade, not its tip. Where a mask's
# ends are unsure -- a broad blade whose longest path runs corner to corner, or
# tip and base swapped -- every trace of the blade still crosses its middle.
# On vogelmeere_x_1, 22 of 35 pairs of s2d leaves on one blade had midrib
# centres < 3.3 mm apart; the closest two distinct blades were 3.3 mm apart.
# Their tips were 6-25 mm apart, overlapping distinct leaves from 3.2 mm.
#   SAME_BLADE    within one SAM3 id, two centre models this close are one
#                 blade (SAM3 id swaps measured 13-40 mm)
#   CENTRE_TOL    pieces whose 3D midrib centres -- or tips -- are this close,
#                 and which never claim the same frame, are one leaf. The tip
#                 catches what the centre misses when SAM3's mask takes in
#                 part of the stalk in one pass or id and not the other (it
#                 moves the centre, not the tip): 3 of the 4 extra merges on
#                 vogelmeere_x_1 had tips 0.0-1.0 mm apart.
#   REL_TOL       ... or this share of the shorter blade, if larger: one
#                 blade's centre estimates spread with its size. sugarbeet_x_1
#                 split 65-72 mm blades into pieces with centres 6.5 mm apart.
SAME_BLADE = 6.0 / MM_PER_UNIT
CENTRE_TOL = 3.0 / MM_PER_UNIT
REL_TOL = 0.15
SCALE_TOL = 0.02         # flat-lay scale fit has converged when it moves less than this


def set_scale(mm_per_unit: float) -> None:
    """Every millimetre threshold in this file follows MM_PER_UNIT."""
    global MM_PER_UNIT, SAME_BLADE, CENTRE_TOL
    MM_PER_UNIT = float(mm_per_unit)
    SAME_BLADE = 6.0 / MM_PER_UNIT
    CENTRE_TOL = 3.0 / MM_PER_UNIT


def rig_scale(run: Run) -> float:
    """mm per unit if the cameras sit RIG_CAMERA_MM from the crown."""
    crown = run.to_world(run.skeleton["crown"])
    dist = []
    for im in run.image_of.values():
        pose = im.cam_from_world()
        R, t = pose.rotation.matrix(), np.asarray(pose.translation)
        dist.append(np.linalg.norm(-R.T @ t - crown))
    return RIG_CAMERA_MM / float(np.median(dist))


def flat_lay_scale(lengths_units, blades_mm) -> float:
    """mm per unit that best maps the n longest 3D midribs onto the n longest
    flat-lay blades, by rank (score_against_flatlay.py's fit)."""
    n = min(len(lengths_units), len(blades_mm))
    if n == 0:
        raise ValueError("no leaves to fit a scale against the flat lay")
    three = np.sort(np.asarray(lengths_units, float))[::-1][:n]
    truth = np.sort(np.asarray(blades_mm, float))[::-1][:n]
    return float((truth * three).sum() / (three * three).sum())


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


def stalk_end(crop: np.ndarray, stem: np.ndarray, x0: int, y0: int, touch: int = 5):
    """(row, col) in `crop` where the stalk goes into this blade, or None.

    The blade-edge pixels touching the stem mask, and of those the one farthest
    (through the blade) from the blade's centre: a stalk goes in at an end of
    the blade, while a stem passing alongside touches its middle. P2's stem
    session trims the stalk off every leaf mask, so this edge is where it was.
    """
    h, w = stem.shape
    ch, cw = crop.shape
    pad = 2 * touch
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(w, x0 + cw + pad), min(h, y0 + ch + pad)
    m = np.zeros((Y1 - Y0, X1 - X0), bool)
    m[y0 - Y0:y0 - Y0 + ch, x0 - X0:x0 - X0 + cw] = crop
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch + 1, 2 * touch + 1))
    ring = (cv2.dilate(m.astype(np.uint8), disc) > 0) & stem[Y0:Y1, X0:X1] & ~m
    edge = m & (cv2.dilate(ring.astype(np.uint8), disc) > 0)
    if edge.sum() < 5:
        return None
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    _, from_centre = _farthest(m, [np.unravel_index(int(np.argmax(dt)), dt.shape)])
    cand = np.argwhere(edge)
    r, c = cand[int(np.argmax(from_centre[cand[:, 0], cand[:, 1]]))]
    return np.array([r - (y0 - Y0), c - (x0 - X0)])


def stem_contact(stem: np.ndarray, rc, radius: float) -> int:
    """Stem-mask pixels within `radius` of one end of a leaf mask (full-frame row, col)."""
    r0, c0, rad = int(rc[0]), int(rc[1]), int(np.ceil(radius))
    h, w = stem.shape
    ys, ye, xs, xe = max(0, r0 - rad), min(h, r0 + rad + 1), max(0, c0 - rad), min(w, c0 + rad + 1)
    window = stem[ys:ye, xs:xe]
    yy, xx = np.mgrid[ys:ye, xs:xe]
    return int((window & ((yy - r0) ** 2 + (xx - c0) ** 2 <= radius ** 2)).sum())


def extract_2d(run: Run, masks, frames, junk, cache: Path, base_rule: str = "foot"):
    """Per mask: tip, base, midrib (full-frame x,y). Per frame: stem path and apex.

    `base_rule` (the plant profile's) decides which end of a mask is the base:
      "foot"          the end nearer the plant's foot, walking through the plant
                      mask (gaensefuss)
      "stem_contact"  the end with stem-mask pixels around it -- where the
                      petiole goes in. On a bushy plant overlapping blades are
                      shortcuts through the plant mask and the foot rule flips:
                      on vogelmeere 6 of 10 duplicate leaves were one blade with
                      tip and base swapped. Falls back to "foot" where neither
                      end clearly touches stem.
      "stalk"         the base is not an end of the longest path at all but
                      where the trimmed stalk meets the blade (`stalk_end`), and
                      the tip the blade pixel farthest from it. On a broad blade
                      with its stalk trimmed off the longest path runs corner to
                      corner and its ends move with the view: vogelmeere_x_1
                      split 13 blades into 36 "leaves" that way. Falls back to
                      "foot" on a mask with no stem contact.
    The cache records the rule it was built with and is rebuilt for another.
    """
    p2_written = run.workdir / "p2" / "prompts.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        stale = p2_written.exists() and p2_written.stat().st_mtime > cache.stat().st_mtime
        if data.get("base_rule", "foot") == base_rule and not stale:
            return data["leaves"], data["stems"]
        print(f"  {cache.name} " + (f"predates this P2 ({p2_written})" if stale else
              f"was built with base rule {data.get('base_rule', 'foot')}, not {base_rule}")
              + " -- rebuilding it")
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
            base_by = "foot"
            anchored = stalk_end(crop, stem, x0, y0) if base_rule == "stalk" else None
            if anchored is not None:
                base_rc = anchored
                tip_rc, along = _farthest(crop, [base_rc])
                base_by = "stalk"
            elif base_rule == "stem_contact":
                # Around each end, a sixth of the leaf's own length.
                reach = max(8.0, along.max() / 6.0)
                ca = stem_contact(stem, (y0 + a[0], x0 + a[1]), reach)
                cb = stem_contact(stem, (y0 + b[0], x0 + b[1]), reach)
                if max(ca, cb) >= 5 and max(ca, cb) >= 1.5 * min(ca, cb):
                    base_rc, tip_rc = (a, b) if ca > cb else (b, a)
                    base_by = "stem"
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
                "node": node, "base_by": base_by,
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
    cache.write_text(json.dumps({"leaves": leaves, "stems": stems, "base_rule": base_rule}))
    print(f"  2D: {len(leaves)} leaf masks and {len(stems)} stem paths over {len(by_frame)} "
          f"frames ({time.time() - t0:.0f}s)")
    if base_rule in ("stem_contact", "stalk"):
        by_stem = sum(1 for r in leaves if r["base_by"] != "foot")
        print(f"    base found by {base_rule} on {by_stem} of {len(leaves)} masks, by the "
              "foot rule on the rest")
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
    """A SAM3 id split into the pieces that agree on one blade centre.

    SAM3 swaps ids between neighbouring leaves mid-pass: on gaensefuss_1 pass 0,
    pass0_0 follows one leaf for frames 0-8 and another 40 mm away for most of
    the rest. One RANSAC per id keeps the majority leaf and loses the other, and
    two swapped ids land on the same leaf as duplicates. So RANSAC is repeated
    on what each model leaves over: every piece that agrees on a centre is a
    leaf candidate of its own.

    The centre (the 2D midrib's midpoint), not the tip: on a broad blade the
    tip found in 2D wanders round the margin with the view and splits one
    blade into two or three pieces, the centre does not move.
    """
    remaining, models, centres = list(obs), [], []
    while len(remaining) >= MIN_INLIERS and len(models) < max_models:
        rays = Rays(run, [o["frame"] for o in remaining], [o["centre"] for o in remaining])
        res = ransac_point(rays, np.array([o["weight"] for o in remaining]), rng)
        need = MIN_INLIERS if not models else MIN_INLIERS + 1
        if res is None or res["inliers"] < need:
            break
        inl = [o for o, i in zip(remaining, res["inlier_mask"]) if i]
        remaining = [o for o, i in zip(remaining, res["inlier_mask"]) if not i]
        centre = np.asarray(res["xyz"])
        near = [i for i, c in enumerate(centres) if np.linalg.norm(c - centre) < SAME_BLADE]
        if near:
            models[near[0]] += inl
            continue
        models.append(inl)
        centres.append(centre)
    return models


def _oriented_ends(run: Run, obs, rng):
    """Tip and base of one blade from 2D ends whose order per frame is unsure.

    Each frame names two ends of the mask, and which is the base is a rule's
    guess that flips from view to view (10 of vogelmeere_x_1's duplicate pairs
    were one blade traced both ways). So the two ends are found without the
    order: the point most frames put an end at, then the point their other
    ends agree on. Which of the two is the base is then a vote of the frames,
    stalk or stem contact counting double the foot rule. Frames whose ends
    are neither (a corner-to-corner trace) drop out as outliers.

    When the frames agree on one end only -- most often the base, where the
    stalk goes in behind the blade and its contact jumps from view to view
    (vogelmeere_x_1 pass0_27: 12 frames agree on the centre, 5 on the tip, 2 on
    the base) -- the leaf is kept with that end fixed and the other one left
    free ("free": its start, the agreed end mirrored through the blade centre),
    for the midrib fit to place.
    """
    n = len(obs)
    frames = [o["frame"] for o in obs]
    w = np.array([o["weight"] for o in obs])
    ends = Rays(run, frames + frames, [o["tip"] for o in obs] + [o["base"] for o in obs])
    first = ransac_point(ends, np.r_[w, w], rng)
    if first is None:
        return None
    at_tip, at_base = first["inlier_mask"][:n], first["inlier_mask"][n:]
    agree = at_tip | at_base
    if agree.sum() < MIN_INLIERS:
        return None
    idx = np.flatnonzero(agree)
    other = Rays(run, [frames[i] for i in idx],
                 [obs[i]["base"] if at_tip[i] else obs[i]["tip"] for i in idx])
    second = ransac_point(other, w[idx], rng)
    strength = np.array([1.0 if o.get("base_by", "foot") != "foot" else 0.5 for o in obs])
    if second is None or second["inliers"] < MIN_INLIERS:
        centre = ransac_point(Rays(run, [frames[i] for i in idx], [obs[i]["centre"] for i in idx]),
                              w[idx], rng)
        if centre is None or centre["inliers"] < MIN_INLIERS:
            return None
        X = np.asarray(first["xyz"])
        x_is_base = float((w * strength)[at_base & ~at_tip].sum()) > \
            float((w * strength)[at_tip].sum())
        return {"fixed": X, "free": 2.0 * np.asarray(centre["xyz"]) - X,
                "fixed_is_tip": not x_is_base, "both": agree,
                "flipped": (at_tip if x_is_base else at_base & ~at_tip),
                "tip_inliers": 0 if x_is_base else int(agree.sum()),
                "base_inliers": int(agree.sum()) if x_is_base else 0}
    both = np.zeros(n, bool)
    both[idx[second["inlier_mask"]]] = True
    # flipped[i]: frame i called the first point its base
    flipped = at_base & ~at_tip
    first_is_base = float((w * strength)[both & flipped].sum())
    second_is_base = float((w * strength)[both & ~flipped].sum())
    X, Y = np.asarray(first["xyz"]), np.asarray(second["xyz"])
    n_first, n_second = int(agree.sum()), int(second["inliers"])
    if first_is_base > second_is_base:
        return {"tip": Y, "base": X, "tip_inliers": n_second, "base_inliers": n_first,
                "both": both, "flipped": both & ~flipped}
    return {"tip": X, "base": Y, "tip_inliers": n_first, "base_inliers": n_second,
            "both": both, "flipped": both & flipped}


def _fit_leaf(run: Run, obs, rng, stem):
    """Tip, base, node and midrib of one leaf from the observations that show it."""
    ends = _oriented_ends(run, obs, rng)
    if ends is None:
        return None
    both = ends["both"]
    views = _spread([CurveView(run, o["frame"], o["midrib"], o["weight"])
                     for o, ok in zip(obs, both) if ok and o["weight"] >= 0.2], MAX_FIT_VIEWS)
    if "free" in ends:
        # one end agreed on; the midrib fit places the other
        if len(views) < 2:
            return None
        midrib = fit_curve(ends["fixed"], ends["free"], views, fixed_end=False)
        if ends["fixed_is_tip"]:
            midrib = midrib[::-1]
        T, B = midrib[-1], midrib[0]
    else:
        T, B = np.asarray(ends["tip"]), np.asarray(ends["base"])
        midrib = fit_curve(B, T, views)
    res = {"tip": {"inliers": ends["tip_inliers"]}, "base": {"inliers": ends["base_inliers"]}}
    err = np.concatenate([v.chamfer_px(midrib) for v in views]) if views else np.array([])

    # the node: where the petiole meets the stem, triangulated, then put on the
    # stem -- from frames that had the base at the right end, since each 2D node
    # was walked from that frame's own base
    node, node_inliers = None, 0
    with_node = [o for o, ok, f in zip(obs, both, ends["flipped"])
                 if ok and not f and o.get("node") is not None]
    if stem is not None and len(with_node) >= MIN_INLIERS:
        rays = Rays(run, [o["frame"] for o in with_node], [o["node"] for o in with_node])
        nres = ransac_point(rays, np.array([o["weight"] for o in with_node]), rng)
        if nres is not None and nres["inliers"] >= MIN_INLIERS:
            dense = resample(stem, 400)
            node = dense[np.argmin(np.linalg.norm(dense - np.asarray(nres["xyz"]), axis=1))]
            node_inliers = nres["inliers"]
    return {"tip": T, "base": B, "midrib": midrib, "node": node, "node_inliers": node_inliers,
            "frames": {o["frame"] for o, ok in zip(obs, both) if ok},
            "flipped_views": int(ends["flipped"].sum()),
            "free_end": ("base" if ends.get("fixed_is_tip") else "tip") if "free" in ends else None,
            "views": [v.frame for v in views], "tip_inliers": res["tip"]["inliers"],
            "base_inliers": res["base"]["inliers"], "observations": len(obs),
            "fit_px": float(np.median(err)) if len(err) else float("nan")}


def reconstruct(run: Run, leaves2d, stems2d, frames, rng, merge_tol: float | None = None,
                has_stem: bool = True):
    """Leaves and the stem, from the 2D evidence of `frames` only.

    1. the stem first, so each leaf's petiole node can be put on it;
    2. every SAM3 id split into the pieces that agree on a blade centre
       (`_segments`), each piece's ends found without trusting their order
       (`_oriented_ends`);
    3. pieces merged when their 3D midrib centres or tips are within
       CENTRE_TOL (or REL_TOL of the shorter blade, if larger) and the merged
       group never claims a frame twice: one leaf
       carried by swapped ids, or seen by two passes. The tolerance is tight
       (3 mm, where the old tip rule needed 6) because the centre does not
       wander with the view the way the old 2D tip did. Bases are not
       compared: whether SAM3's mask takes in part of the petiole changes with
       elevation (sugarbeet_1, 28-09). Within a pass the frame test guards
       leaves seen side by side; across passes nothing can, so two leaves
       whose centres or tips sit within CENTRE_TOL in different passes would
       merge;
    4. every leaf refitted on all of its observations.
    """
    if merge_tol is None:
        merge_tol = CENTRE_TOL
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
                by_track[row["track"]][row["frame"]] = dict(
                    row, centre=row["midrib"][len(row["midrib"]) // 2])
    pieces, failed = [], {}
    for track, rows in sorted(by_track.items()):
        models = _segments(run, list(rows.values()), rng) if len(rows) >= MIN_INLIERS else []
        fits = [(m, _fit_leaf(run, m, rng, stem)) for m in models]
        fits = [(m, f) for m, f in fits if f is not None]
        if not fits:
            failed[track] = (f"only {len(rows)} usable views" if len(rows) < MIN_INLIERS
                             else "its views do not agree on a blade, or on its two ends")
        for i, (m, f) in enumerate(fits):
            pieces.append({"name": f"{track}" + (f"#{i}" if i else ""), "tracks": {track},
                           "obs": m, "fit": f})

    # merge pieces that are the same leaf: centres or tips close, no frame claimed twice
    centre = [resample(p["fit"]["midrib"], 41)[20] for p in pieces]
    blade = [float(np.linalg.norm(np.diff(p["fit"]["midrib"], axis=0), axis=1).sum()) for p in pieces]
    claimed = [{o["frame"] for o in p["obs"]} for p in pieces]
    order = sorted(range(len(pieces)), key=lambda i: -pieces[i]["fit"]["tip_inliers"])
    owner = list(range(len(pieces)))
    for a_i, a in enumerate(order):
        for b in order[a_i + 1:]:
            if owner[b] != b or owner[a] != a:
                continue
            tol = max(merge_tol, REL_TOL * min(blade[a], blade[b]))
            close = (np.linalg.norm(centre[a] - centre[b]) < tol
                     or np.linalg.norm(pieces[a]["fit"]["tip"] - pieces[b]["fit"]["tip"]) < tol)
            if close and len(claimed[a] & claimed[b]) <= 1:
                owner[b] = a
                claimed[a] |= claimed[b]
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


def petiole(leaf: dict, stem, crown=None):
    """(petiole polyline, which rule placed it).

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


def petiole_on_tree(leaf: dict, trunk, stalks, max_ratio: float = 1.5):
    """(petiole polyline, which rule placed it), from the 3D stem tree.

    For a plant whose leaves sit on side branches (plant profile
    petiole_rule "stem_tree", vogelmeere). The single-stem rule above sent
    them across open air to the one stem it knew: 45 of 56 petioles longer
    than their own blade, 18 attached at the clamp.

    `trunk` are the stem and branch curves, `stalks` the tree's stalks that
    end in one blade. A stalk ending at this blade's base is its petiole;
    otherwise the petiole joins the base to the nearest stem or branch point.
    None is drawn when nothing is within `max_ratio` blade lengths -- a
    missing petiole is honest, a 40 mm line to the wrong branch is not.
    """
    tip, base = np.asarray(leaf["tip"]), np.asarray(leaf["base"])
    blade = float(np.linalg.norm(np.diff(leaf["midrib"], axis=0), axis=1).sum()) \
        or float(np.linalg.norm(tip - base))
    if stalks:
        gaps = [float(np.linalg.norm(st[-1] - base)) for st in stalks]
        k = int(np.argmin(gaps))
        if gaps[k] <= max(0.25 * blade, 3.0 / MM_PER_UNIT):
            return np.vstack([stalks[k], base]), "stem-tree stalk"
    if trunk:
        dense = np.vstack([resample(t, max(int(len(t)) * 8, 2)) for t in trunk])
        d = np.linalg.norm(dense - base, axis=1)
        j = int(np.argmin(d))
        if d[j] <= max_ratio * blade:
            return np.vstack([dense[j], base]), "nearest stem or branch"
    return np.zeros((0, 3)), "none within reach"


def petioles_2d(run: Run, leaves: dict, parents, stalks) -> dict:
    """{leaf key: (petiole polyline, which rule placed it)}, each petiole traced in the photos.

    The straight rules above draw a petiole as a line from the blade base to
    a stem point. P5x follows tissue instead -- the stem tree's stalk, or the
    shortest path through the cloud -- but the cloud is the weak link where
    leaves crowd: on vogelmeere_x_1, 43% of the midribs of leaves with a close
    neighbour lie off its surface. The stalk is clean in 2D, though: P2's stem
    session draws it in every frame. So, in each view of the leaf:

      the 3D base and the parent axes (stem and branches) are projected;
      a walk from the base, cheap on stem-mask pixels and dear on other plant
      pixels (petiole_node's costs), runs to the first axis pixel it reaches;

    the views vote on where in 3D the petiole joins (hits within CENTRE_TOL
    agree), and the petiole is the 3D curve whose projections lie on the
    agreeing views' 2D walks (fit_curve, as for the midribs), from that point
    to the base. A stem-tree stalk that ends at the blade base is used as it
    is -- it is tissue too. A leaf whose views agree on no joint keeps the
    straight line to the nearest axis point, and says so.

    `parents` are 3D polylines: the stem and its branches, or the one stem.
    A rosette has no petioles at all (main()).
    """
    out = {}
    lines = [np.atleast_2d(np.asarray(p, float)) for p in parents]
    dense = np.vstack([resample(p, max(len(p) * 8, 2)) for p in lines])
    todo = defaultdict(list)
    for key, leaf in leaves.items():
        base = np.asarray(leaf["base"])
        blade = float(np.linalg.norm(np.diff(leaf["midrib"], axis=0), axis=1).sum())
        if stalks:
            gaps = [float(np.linalg.norm(st[-1] - base)) for st in stalks]
            k = int(np.argmin(gaps))
            if gaps[k] <= max(0.25 * blade, 3.0 / MM_PER_UNIT):
                out[key] = (np.vstack([stalks[k], base]), "stem-tree stalk")
                continue
        for frame in (leaf["views"] if len(leaf["views"]) >= 2 else sorted(leaf["frames"])):
            todo[frame].append(key)

    hits = defaultdict(list)                     # key -> [(frame, index into dense, 2D walk)]
    for frame, keys in sorted(todo.items()):
        plant = cv2.imread(str(run.workdir / "p2" / "masks" / "plant" / f"{frame}.png"), 0) > 127
        stem = cv2.imread(str(run.workdir / "p2" / "masks" / "stem" / f"{frame}.png"), 0) > 127
        h, w = plant.shape
        im = run.image_of[frame]
        cam, pose = run.rec.cameras[im.camera_id], im.cam_from_world()
        on_axis = _uv(cam, pose, dense)
        for key in keys:
            leaf = leaves[key]
            b = _uv(cam, pose, [leaf["base"]])[0]
            rib = _uv(cam, pose, leaf["midrib"])
            reach = int(min(500, 2.0 * np.linalg.norm(np.diff(rib, axis=0), axis=1).sum() + 40))
            r0, c0 = int(round(b[1])), int(round(b[0]))
            ys, ye, xs, xe = max(0, r0 - reach), min(h, r0 + reach), max(0, c0 - reach), min(w, c0 + reach)
            if not (ys <= r0 < ye and xs <= c0 < xe):
                continue
            sel = np.flatnonzero((on_axis[:, 0] >= xs) & (on_axis[:, 0] < xe - 0.5)
                                 & (on_axis[:, 1] >= ys) & (on_axis[:, 1] < ye - 0.5))
            if not len(sel):
                continue
            cost = np.where(plant[ys:ye, xs:xe], np.where(stem[ys:ye, xs:xe], 1.0, 4.0), np.inf)
            start = (r0 - ys, c0 - xs)
            if not np.isfinite(cost[start]):
                cost[start] = 4.0
            mcp = MCP_Geometric(cost)
            dist, _ = mcp.find_costs([start])
            rr = on_axis[sel, 1].round().astype(int) - ys
            cc = on_axis[sel, 0].round().astype(int) - xs
            d = dist[rr, cc]
            if not np.isfinite(d).any():
                continue
            j = int(np.argmin(d))
            walk = np.asarray(mcp.traceback((rr[j], cc[j])), float)[:, ::-1] + [xs, ys]
            hits[key].append((frame, sel[j], walk))

    for key, leaf in leaves.items():
        if key in out:
            continue
        base = np.asarray(leaf["base"])
        hs = hits.get(key, [])
        if len(hs) >= 2:
            P = dense[[hh[1] for hh in hs]]
            D = np.linalg.norm(P[:, None] - P[None], axis=2)
            i = int(np.argmax((D < CENTRE_TOL).sum(axis=1)))
            agree = np.flatnonzero(D[i] < CENTRE_TOL)
            if len(agree) >= 2:
                joint = dense[np.argmin(np.linalg.norm(dense - P[agree].mean(axis=0), axis=1))]
                views = [CurveView(run, hs[k][0], resample(hs[k][2], SAMPLES_2D)) for k in agree]
                out[key] = (fit_curve(joint, base, views, n_ctrl=3, smooth=2.0), "2D stalk walk")
                continue
        out[key] = (np.vstack([dense[np.argmin(np.linalg.norm(dense - base, axis=1))], base]),
                    "nearest axis point (no 2D walk agreed)")
    return out


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------


def _uv(cam, pose, X):
    c = np.asarray(X) @ pose.rotation.matrix().T + np.asarray(pose.translation)
    return cam.img_from_cam(c)


def draw(img, cam, pose, stem, leaves, colours, thin=False):
    w = 1 if thin else 2
    # The stem, or the stem and its branches when it is a tree.
    for curve in (stem if isinstance(stem, list) else [stem] if stem is not None else []):
        line = _uv(cam, pose, resample(curve, 80)).astype(np.int32)
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
                    help="override P5's (p5/instancing.json); rosette = no stem, no petioles: "
                         "each leaf runs from the crown")
    ap.add_argument("--other-passes", type=Path, help="full run, to draw on frames not used")
    ap.add_argument("--flat-lay", type=Path,
                    help="leaf-pose output dir with leaves.json (run with --marker-mm so blades "
                         "are in mm); without --mm-per-unit the scale is fitted against it")
    ap.add_argument("--mm-per-unit", type=float,
                    help="mm per reconstruction unit of this P3 solve, from an independent "
                         "measurement. Overrides --tag-mm and the flat-lay fit")
    ap.add_argument("--tag-mm", type=float,
                    help="printed side of the AprilTags in the turntable scene, mm: the scale "
                         "from them (tag_scale.py, over every frame of the solve). Falls back "
                         "to the flat-lay fit when no tag passes its checks")
    ap.add_argument("--retrace-p5x", action="store_true",
                    help="re-trace P5x's skeleton with the current leaf_skeleton.trace for the "
                         "comparison figure, instead of reading p5x/skeleton.json")
    ap.add_argument("--profile", choices=sorted(plant_profiles.PROFILES),
                    help="use this plant's rules (pose_estimator/plant_profiles.py) instead of "
                         "the one the workdir's folder name picks")
    ap.add_argument("--petiole-rule", choices=["leaf_axis", "stem_tree", "stalk_2d"],
                    help="override the profile's petiole rule (see petioles_2d, petiole_on_tree, "
                         "petiole). A rosette has no petioles whatever this says")
    ap.add_argument("--base-rule", choices=["foot", "stem_contact", "stalk"],
                    help="override the profile's rule for telling a mask's base from its tip "
                         "(see extract_2d)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.mm_per_unit is None and args.tag_mm is None and args.flat_lay is None:
        ap.error("no scale: pass --mm-per-unit, --tag-mm, or --flat-lay to fit one. Every P3 "
                 "solve has its own units, so a constant from another run gives wrong mm "
                 "and merges")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    blades = None
    if args.flat_lay:
        gt = json.loads((args.flat_lay / "leaves.json").read_text())["leaves"]
        blades = sorted((l["blade_length"] for l in gt if l["area"] > 5), reverse=True)

    from pose_estimator import cloud_source
    from pose_estimator.cli.leaf_instances import _architecture
    from p5x_views import cache_masks

    run = Run(args.workdir)
    profile, folder = plant_profiles.profile_for(args.workdir)
    if args.profile:
        profile, folder = plant_profiles.PROFILES[args.profile], f"--profile {args.profile}"
    print(f"  {plant_profiles.describe(profile, folder)}")
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
          + ("stem + petioles off it" if has_stem else "no stem, leaves run from the crown"))
    junk = junk_masks(masks, args.workdir, set(frames))
    life = defaultdict(set)
    for t, f in zip(masks["track"], masks["frame"]):
        life[str(t)].add(str(f))
    base_rule = args.base_rule or profile.base_rule
    leaves2d, stems2d = extract_2d(run, masks, set(frames), junk, args.out / "evidence_2d.json",
                                   base_rule=base_rule)

    # 0. the scale, before anything that thresholds in mm. Fitted against the
    # flat lay it is circular -- the merges it sets decide which midribs the fit
    # sees -- so it is iterated to a fixed point from the rig's camera distance.
    midrib_units = lambda ls: [float(np.linalg.norm(np.diff(l["midrib"], axis=0), axis=1).sum())
                               for l in ls.values()]
    full = None
    tags = None
    if args.mm_per_unit is None and args.tag_mm is not None:
        from tag_scale import tag_scale
        tags = tag_scale(run, args.tag_mm)
        (args.out / "tag_scale.json").write_text(json.dumps(tags, indent=1))
        for tag, row in tags["tags"].items():
            print(f"  tag {tag}: in {row['frames_detected']} frames"
                  + (f", rejected: {row['rejected']}" if "rejected" in row
                     else f", square {row['square']:.3f}"))
        if tags["mm_per_unit"] is None and args.flat_lay is None:
            raise SystemExit("  no tag gave a scale and there is no --flat-lay to fall back on")
    if args.mm_per_unit is not None:
        set_scale(args.mm_per_unit)
        scale_source = "--mm-per-unit"
        print(f"  scale {MM_PER_UNIT:.2f} mm/unit (given)")
    elif tags is not None and tags["mm_per_unit"] is not None:
        set_scale(tags["mm_per_unit"])
        scale_source = f"AprilTags, {tags['edges_used']} edges, spread {tags['spread']:.1%}"
        print(f"  scale {MM_PER_UNIT:.2f} mm/unit ({scale_source})")
    else:
        scale_source = "flat lay, rank-matched" + (" (no tag passed)" if tags else "")
        set_scale(rig_scale(run))
        print(f"  scale seed {MM_PER_UNIT:.2f} mm/unit (cameras ~{RIG_CAMERA_MM:.0f} mm "
              f"from the crown)")
        for it in range(6):
            full = reconstruct(run, leaves2d, stems2d, set(frames), rng, has_stem=has_stem)
            fitted_scale = flat_lay_scale(midrib_units(full[0]), blades)
            moved = abs(fitted_scale / MM_PER_UNIT - 1)
            print(f"  scale fit {it}: {MM_PER_UNIT:7.2f} -> {fitted_scale:7.2f} mm/unit "
                  f"({len(full[0])} leaves vs {len(blades)} on the flat lay)")
            set_scale(fitted_scale)
            if moved < SCALE_TOL:
                break
        else:
            scale_source += " (did not converge)"
            full = None
            print("  WARNING: the scale fit did not settle; lengths in mm are unreliable")

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
    # (the converged fit's last reconstruction, when there is one: its merges
    # were made within SCALE_TOL of the final scale)
    leaves, failed, stem, crown, n_pieces = full or reconstruct(run, leaves2d, stems2d,
                                                                set(frames), rng,
                                                                has_stem=has_stem)
    # The stem tree P5x traced from the stem cloud: what petioles join on a
    # plant whose leaves sit on side branches. A skeleton.json written before
    # the tree existed has no axes, so it is re-traced here.
    p5x = run.skeleton
    if args.retrace_p5x or ((args.petiole_rule or profile.petiole_rule) in ("stem_tree", "stalk_2d")
                              and not p5x.get("axes")):
        from pose_estimator.leaf_skeleton import trace
        upright = (run.points - run.origin) @ run.rotation.T
        voxel, _ = cloud_source.voxel_size(args.workdir, cloud_source.BASELINE, upright)
        from pose_estimator.cli.leaf_instances import _architecture
        p5x = trace(upright, run.assignment, voxel,
                    architecture=_architecture(args.workdir, cloud_source.BASELINE))
        print(f"  P5x skeleton re-traced with the current tracer: {len(p5x['leaves'])} midribs, "
              f"{len(p5x.get('axes') or [])} stem-tree axes")
    trunk, stalks = [], []
    for axis in p5x.get("axes") or []:
        (stalks if axis["kind"] == "petiole" else trunk).append(run.to_world(axis["points"]))
    petiole_rule = args.petiole_rule or profile.petiole_rule
    # the stem tree is drawn, and petioles join it, for both tree rules
    use_tree = petiole_rule in ("stem_tree", "stalk_2d") and bool(trunk)
    if petiole_rule == "stem_tree" and not trunk:
        print("  WARNING: the profile asks for petioles on the stem tree, but P5x has no stem "
              "tree (no stem tissue?) -- falling back to the leaf-axis rule")

    rules = defaultdict(int)
    if petiole_rule == "stalk_2d" and has_stem:
        # walk to the stem tree, else to the one stem
        placed = petioles_2d(run, leaves, trunk if trunk else [stem], stalks)
    for key, leaf in leaves.items():
        if not has_stem:
            # A rosette leaf has no petiole: it runs from the crown, widening
            # into its blade, all the way to the tip. Whatever part of it the
            # 2D masks did not show is joined straight to the crown.
            leaf["midrib"] = np.vstack([crown, leaf["midrib"]])
            leaf["base"] = np.asarray(crown)
            leaf["petiole"], leaf["petiole_rule"] = np.zeros((0, 3)), "none (rosette: leaf from the crown)"
        elif petiole_rule == "stalk_2d":
            leaf["petiole"], leaf["petiole_rule"] = placed[key]
        elif petiole_rule == "stem_tree" and use_tree:
            leaf["petiole"], leaf["petiole_rule"] = petiole_on_tree(leaf, trunk, stalks)
        else:
            leaf["petiole"], leaf["petiole_rule"] = petiole(leaf, stem, crown)
        rules[leaf["petiole_rule"]] += 1
    blade_of = lambda l: float(np.linalg.norm(np.diff(l["midrib"], axis=0), axis=1).sum())
    pet_of = lambda l: (float(np.linalg.norm(np.diff(l["petiole"], axis=0), axis=1).sum())
                        if len(l["petiole"]) >= 2 else 0.0)
    too_long = sum(1 for l in leaves.values() if pet_of(l) > blade_of(l))
    n_ids = len({r["track"] for r in leaves2d})
    print(f"\n  {n_ids} SAM3 leaf ids -> {n_pieces} consistent pieces (ids split where SAM3 "
          f"swapped leaves) -> {len(leaves)} leaves in 3D after merging pieces of one leaf; "
          f"{len(failed)} ids not reconstructed")
    print("  petioles placed by: " + ", ".join(f"{k} {v}" for k, v in sorted(rules.items())))
    print(f"  petioles longer than their own blade: {too_long} of {len(leaves)}")
    for track, why in sorted(failed.items()):
        print(f"    {track}: {why}")

    lengths = sorted((float(np.linalg.norm(np.diff(l["midrib"], axis=0), axis=1).sum())
                      * MM_PER_UNIT for l in leaves.values()), reverse=True)
    scores = {"held_out_px": float(np.median(held)), "fitted_px": float(np.median(fitted)),
              "leaves": len(leaves), "failed": failed, "midrib_mm": lengths,
              "mm_per_unit": MM_PER_UNIT, "scale_source": scale_source,
              "profile": profile.name, "base_rule": base_rule, "petiole_rule": petiole_rule,
              "petiole_rules": dict(rules),
              "petioles_longer_than_blade": too_long}
    if blades is not None and lengths and not scale_source.startswith("flat lay"):
        # what the flat lay would have said: agreement with an independent scale
        # is evidence that both the tags and the rank matching are right
        check = flat_lay_scale([v / MM_PER_UNIT for v in lengths], blades)
        scores["flat_lay_fitted_mm_per_unit"] = check
        print(f"\n  scale cross-check: flat-lay fit says {check:.2f} mm/unit, "
              f"used {MM_PER_UNIT:.2f} ({check / MM_PER_UNIT - 1:+.1%})")
    if blades is not None:
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
           # On the stem tree the stem is P5x's main axis, already in the plant
           # frame, and the branches come with it.
           "stem": ((np.asarray(p5x["axes"][0]["points"]).tolist()) if use_tree
                    else to_plant(stem).tolist() if stem is not None else []),
           "axes": p5x.get("axes") or [] if use_tree else [],
           "leaves": [{"id": i, "track": t, "sam3_ids": leaves[t]["tracks"],
                       "tip": to_plant(leaves[t]["tip"])[0].tolist(),
                       "base": to_plant(leaves[t]["base"])[0].tolist(),
                       "midrib": to_plant(leaves[t]["midrib"]).tolist(),
                       "petiole": to_plant(leaves[t]["petiole"]).tolist(),
                       "petiole_rule": leaves[t]["petiole_rule"],
                       "colour_bgr": list(colours[t]),
                       "midrib_mm": float(np.linalg.norm(np.diff(leaves[t]["midrib"], axis=0),
                                                         axis=1).sum() * MM_PER_UNIT),
                       "tip_inliers": leaves[t]["tip_inliers"],
                       "observations": leaves[t]["observations"],
                       "fit_px": leaves[t]["fit_px"]} for i, t in enumerate(tracks)]}
    (args.out / "skeleton.json").write_text(json.dumps(doc, indent=1))

    # 4. pictures -- the stem tree, branches and all, where it was used
    drawn_stem = trunk if use_tree else stem
    rec0 = run.rec
    shots = [frames[i] for i in np.linspace(2, len(frames) - 3, 4).round().astype(int)]
    tiles = [tile(args.workdir, rec0, f, f"{f}  pass {run.sources[f]}  (fitted)", drawn_stem,
                  leaves, colours) for f in shots]
    if args.passes and not args.other_passes:
        # frames of this same workdir that the fit did not use
        rest = sorted(f for f in run.image_of if int(run.sources[f]) not in set(args.passes))
        for p in sorted({int(run.sources[f]) for f in rest})[:2]:
            fs = [f for f in rest if int(run.sources[f]) == p]
            f = fs[len(fs) // 3]
            tiles.append(tile(args.workdir, rec0, f, f"{f}  pass {p}  (NOT used for the fit)",
                              drawn_stem, leaves, colours))
    if args.other_passes:
        rec_all = pycolmap.Reconstruction(str(args.other_passes / "p3" / "sparse" / "best"))
        src = json.loads((args.other_passes / "p1" / "sources.json").read_text())
        for p in (1, 2):
            fs = sorted(f for f, q in src.items() if int(q) == p)
            f = fs[len(fs) // 3]
            tiles.append(tile(args.other_passes, rec_all, f,
                              f"{f}  pass {p}  (NOT used; plant drooped since)", drawn_stem,
                              leaves, colours))
    cv2.imwrite(str(args.out / "reprojection.jpg"), grid(tiles, 3), [cv2.IMWRITE_JPEG_QUALITY, 88])

    # P5x's traced skeleton next to this one, same views (`p5x` from step 2)
    p5x_stem = ([run.to_world(a["points"]) for a in p5x["axes"] if a["kind"] != "petiole"]
                if p5x.get("axes") else
                run.to_world(p5x["stem"]) if len(p5x.get("stem") or []) >= 2 else None)
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
        pairs.append(tile(args.workdir, rec0, f, f"{f}  new: built in 2D, fused in 3D", drawn_stem,
                          leaves, colours))
    cv2.imwrite(str(args.out / "p5x_vs_2d.jpg"), grid(pairs, 2), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\n  wrote {args.out / 'reprojection.jpg'}, {args.out / 'p5x_vs_2d.jpg'}, "
          f"{args.out / 'skeleton.json'}")


if __name__ == "__main__":
    main()
