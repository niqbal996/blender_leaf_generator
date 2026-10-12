#!/usr/bin/env python
"""Apex leaves in 3D from per-frame leaf lines, with which-is-which solved by geometry.

    python scripts/apex_match.py --workdir <specimen>/plant --out <specimen>/plant/debug/apex_match \
        --regions <specimen>/plant/debug/apex_labels/apex_regions.npz [--base-skeleton <s2d dir>]
    python scripts/apex_match.py ... --evidence <s2d dir>/evidence_2d.json   # SAM3's lines, 74 frames

Nobody can keep one name on one apex leaf across views -- not SAM3, not a
person labelling. But every line drawn base -> tip in a frame with a known
camera says where that leaf's base and tip can be in 3D, and two lines of the
same leaf in two frames must agree: the bases on each other's epipolar
lines, and the tips too. A third frame then either shows a line where that
3D leaf projects, or it does not. So identity is not labelled, it is found:

  1. every pair of lines in two different frames (FAN_DEG apart at least) of
     one apex: triangulate base from the two bases, tip from the two tips;
     a pair is a candidate leaf if all four reproject within PAIR_PX and the
     leaf lies in the apex region
  2. support: the lines of other frames whose base and tip both land within
     SUPPORT_PX of the candidate's projections
  3. greedy: the best-supported candidate whose lines are all still free
     becomes a leaf, takes its lines (one per frame) and its midrib is fitted
     to all of them (skeleton_2d.fit_curve); repeat

Scored honestly: leave one labelled frame out, match the rest, score the 3D
leaves on the left-out frame (apex_labels.score); every frame in turn.

--evidence: the same on the lines SAM3 found instead of hand-drawn ones --
skeleton_2d.py's per-mask base, tip and midrib, every frame (zoomed masks
from apex_zoom.py are best at the apex). Many more frames, so a leaf must be
backed by MIN_SUPPORT of them; SAM3's base/tip call can be swapped, so both
orientations are tried; and the hand labels are not used at all, so every
labelled frame is held out.

Writes into --out: match.json, match_all/skeleton.json (every frame used;
plus --base-skeleton's leaves outside the apices: the whole plant, for
view_in_blender.sh --skeleton2d), score_without_<frame>.jpg.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apex_labels import apex_of, draw_score, load, score                # noqa: E402
from apex_lift import as_skeleton                                       # noqa: E402
from leaf_tips_2d import Rays, intersect                                # noqa: E402
from p5x_views import Run                                               # noqa: E402
from skeleton_2d import CurveView, fit_curve, resample                  # noqa: E402

PAIR_PX = 10.0      # base and tip of two lines of one leaf reproject this close (hand clicks ~3-5 px)
SUPPORT_PX = 12.0   # a third frame's line backs a leaf if its base and tip land this close
FAN_DEG = 15.0      # two frames closer than this in viewing direction triangulate too poorly
MIN_MM, MAX_MM = 1.0, 40.0
MAX_FAN_DEG = 75.0  # --evidence: pair frames up to this far apart (further: the leaf turns away)
MIN_SUPPORT = 4     # --evidence: frames that must back a leaf
SAME_MM = 1.0       # --evidence: a candidate whose base and tip are this close to a kept leaf's is it


def observations(labels: dict, frames):
    """Per apex: [(frame, line)] over the given frames."""
    out = {}
    for frame in frames:
        lab = labels[frame]
        for _, line in lab["leaves"]:
            apex = apex_of(lab["apices"], line)
            if apex is not None:
                out.setdefault(apex, []).append((frame, resample(line, 24)))
    return out


def view_dir(run: Run, frame: str) -> np.ndarray:
    return run.image_of[frame].cam_from_world().rotation.matrix()[2]


def match(run: Run, obs, region: np.ndarray, mpu: float):
    """[(3D curve, [obs indices])] for one apex."""
    centre = region.mean(axis=0)
    reach = 1.5 * np.percentile(np.linalg.norm(region - centre, axis=1), 95)
    frames = [f for f, _ in obs]
    dirs = {f: view_dir(run, f) for f in set(frames)}
    ends = [(line[0], line[-1]) for _, line in obs]
    cands = []
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            fi, fj = frames[i], frames[j]
            if fi == fj or np.degrees(np.arccos(np.clip(dirs[fi] @ dirs[fj], -1, 1))) < FAN_DEG:
                continue
            rb = Rays(run, [fi, fj], [ends[i][0], ends[j][0]])
            rt = Rays(run, [fi, fj], [ends[i][1], ends[j][1]])
            B = intersect(rb.C, rb.d, np.ones(2))
            T = intersect(rt.C, rt.d, np.ones(2))
            err = max(rb.errors(B).max(), rt.errors(T).max())
            length = np.linalg.norm(T - B) * mpu
            if (err > PAIR_PX or not MIN_MM <= length <= MAX_MM
                    or np.linalg.norm(B - centre) > reach or np.linalg.norm(T - centre) > reach):
                continue
            # support: per other frame, the closest line whose both ends land near
            members, best = {fi: i, fj: j}, {}
            for k in range(len(obs)):
                fk = frames[k]
                if fk in (fi, fj):
                    continue
                rk = Rays(run, [fk], [ends[k][0]])
                e = max(np.linalg.norm(rk.project(B)[0] - ends[k][0]),
                        np.linalg.norm(rk.project(T)[0] - ends[k][1]))
                if e < SUPPORT_PX and e < best.get(fk, np.inf):
                    members[fk], best[fk] = k, e
            cands.append((len(members), -err, B, T, list(members.values())))
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
    used, leaves = set(), []
    for n, _, B, T, idx in cands:
        if used & set(idx):
            continue
        used |= set(idx)
        views = [CurveView(run, frames[k], obs[k][1]) for k in idx]
        leaves.append((fit_curve(B, T, views, n_ctrl=5), idx))
    return leaves


def run_match(run: Run, labels: dict, frames, regions: dict, mpu: float):
    obs = observations(labels, frames)
    out, stats = [], {}
    for apex, o in sorted(obs.items()):
        if f"{apex}" not in regions:
            continue
        got = match(run, o, regions[apex], mpu)
        out += [(apex, X) for X, _ in got]
        support = [len(idx) for _, idx in got]
        stats[apex] = {"lines": len(o), "leaves": len(got),
                       "frames_per_leaf": {str(k): support.count(k) for k in sorted(set(support))},
                       "lines_unmatched": len(o) - sum(support)}
    return out, stats


# --------------------------------------------------------------------------
# --evidence: the same idea on SAM3's lines over every frame, vectorised
# --------------------------------------------------------------------------


class Cam:
    def __init__(self, run: Run, frame: str):
        im = run.image_of[frame]
        self.cam = run.rec.cameras[im.camera_id]
        pose = im.cam_from_world()
        self.R, self.t = pose.rotation.matrix(), np.asarray(pose.translation)
        self.C = -self.R.T @ self.t
        self.axis = self.R[2]
        self.size = (self.cam.width, self.cam.height)

    def rays(self, uv):
        n = self.cam.cam_from_img(np.asarray(uv, float).reshape(-1, 2))
        return np.c_[n, np.ones(len(n))] @ self.R

    def project(self, X):
        c = X @ self.R.T + self.t
        uv = np.full((len(X), 2), np.nan)
        front = c[:, 2] > 1e-9
        if front.any():
            uv[front] = self.cam.img_from_cam(c[front])
        return uv


def two_rays(p1, d1, p2, d2):
    """Midpoints of the closest points of ray pairs (broadcast)."""
    w0 = p1 - p2
    a, b, c = (d1 * d1).sum(-1), (d1 * d2).sum(-1), (d2 * d2).sum(-1)
    d, e = (d1 * w0).sum(-1), (d2 * w0).sum(-1)
    den = np.maximum(a * c - b * b, 1e-12)
    s_, u_ = (b * e - c * d) / den, (a * e - b * d) / den
    return 0.5 * (p1 + s_[..., None] * d1 + p2 + u_[..., None] * d2)


def evidence_obs(run: Run, rows, regions: dict, cams: dict):
    """Per apex: arrays of SAM3 lines whose middle lies in the apex's projected region."""
    out = {}
    for apex, pts in regions.items():
        sub = pts[:: max(1, len(pts) // 800)]
        box = {}
        for f, cam in cams.items():
            uv = cam.project(sub)
            uv = uv[~np.isnan(uv).any(axis=1)]
            if len(uv):
                lo, hi = uv.min(axis=0), uv.max(axis=0)
                pad = 0.15 * (hi - lo)
                box[f] = (lo - pad, hi + pad)
        keep = []
        for r in rows:
            if r["frame"] not in box:
                continue
            lo, hi = box[r["frame"]]
            mid = np.asarray(r["midrib"])[len(r["midrib"]) // 2]
            if ((mid >= lo) & (mid <= hi)).all():
                keep.append(r)
        out[apex] = {"frame": [r["frame"] for r in keep],
                     "base": np.array([r["base"] for r in keep], float).reshape(-1, 2),
                     "tip": np.array([r["tip"] for r in keep], float).reshape(-1, 2),
                     "line": [np.asarray(r["midrib"], float) for r in keep]}
    return out


def match_dense(run: Run, o: dict, region: np.ndarray, cams: dict, mpu: float):
    t0 = time.time()
    centre = region.mean(axis=0)
    reach = 1.5 * np.percentile(np.linalg.norm(region - centre, axis=1), 95)
    frames = sorted(set(o["frame"]))
    fi = np.array([frames.index(f) for f in o["frame"]])
    ends = np.stack([o["base"], o["tip"]], axis=1)                          # (n, 2, 2)
    rays = np.zeros((len(fi), 2, 3))
    for k, f in enumerate(frames):
        sel = fi == k
        rays[sel] = cams[f].rays(ends[sel].reshape(-1, 2)).reshape(-1, 2, 3)
    C = np.array([cams[f].C for f in frames])

    # 1. candidates from frame pairs
    B_all, T_all, err_all = [], [], []
    for a in range(len(frames)):
        for b in range(a + 1, len(frames)):
            ang = np.degrees(np.arccos(np.clip(cams[frames[a]].axis @ cams[frames[b]].axis, -1, 1)))
            if not FAN_DEG <= ang <= MAX_FAN_DEG:
                continue
            ia, ib = np.nonzero(fi == a)[0], np.nonzero(fi == b)[0]
            if not len(ia) or not len(ib):
                continue
            for flip in (0, 1):
                rb = rays[ib][:, ::-1] if flip else rays[ib]
                eb = ends[ib][:, ::-1] if flip else ends[ib]
                B = two_rays(C[a], rays[ia][:, None, 0], C[b], rb[None, :, 0]).reshape(-1, 3)
                T = two_rays(C[a], rays[ia][:, None, 1], C[b], rb[None, :, 1]).reshape(-1, 3)
                ea = np.repeat(ends[ia], len(ib), axis=0)
                eb = np.tile(eb, (len(ia), 1, 1))
                err = np.maximum.reduce([
                    np.linalg.norm(cams[frames[a]].project(B) - ea[:, 0], axis=1),
                    np.linalg.norm(cams[frames[a]].project(T) - ea[:, 1], axis=1),
                    np.linalg.norm(cams[frames[b]].project(B) - eb[:, 0], axis=1),
                    np.linalg.norm(cams[frames[b]].project(T) - eb[:, 1], axis=1)])
                length = np.linalg.norm(T - B, axis=1) * mpu
                ok = ((err < PAIR_PX) & (length >= MIN_MM) & (length <= MAX_MM)
                      & (np.linalg.norm(B - centre, axis=1) < reach)
                      & (np.linalg.norm(T - centre, axis=1) < reach))
                B_all.append(B[ok]); T_all.append(T[ok]); err_all.append(err[ok])
    if not B_all or not sum(len(b) for b in B_all):
        return []
    B, T, err = np.vstack(B_all), np.vstack(T_all), np.concatenate(err_all)
    print(f"    {len(fi)} lines in {len(frames)} frames -> {len(B)} two-frame candidates "
          f"({time.time() - t0:.0f}s)")

    # 2. support: per frame, the line (either way round) whose ends both land near
    member = np.full((len(B), len(frames)), -1)
    flipped = np.zeros((len(B), len(frames)), bool)
    for k, f in enumerate(frames):
        idx = np.nonzero(fi == k)[0]
        if not len(idx):
            continue
        pb, pt = cams[f].project(B), cams[f].project(T)
        d_b = np.linalg.norm(pb[:, None] - ends[idx][None, :, 0], axis=2)
        d_t = np.linalg.norm(pt[:, None] - ends[idx][None, :, 1], axis=2)
        f_b = np.linalg.norm(pb[:, None] - ends[idx][None, :, 1], axis=2)
        f_t = np.linalg.norm(pt[:, None] - ends[idx][None, :, 0], axis=2)
        e_dir, e_flip = np.maximum(d_b, d_t), np.maximum(f_b, f_t)
        e = np.fmin(e_dir, e_flip)
        best = np.nanargmin(np.where(np.isnan(e), np.inf, e), axis=1)
        be = e[np.arange(len(B)), best]
        hit = be < SUPPORT_PX
        member[hit, k] = idx[best[hit]]
        flipped[hit, k] = (e_flip < e_dir)[np.arange(len(B)), best][hit]
    support = (member >= 0).sum(axis=1)
    print(f"    support counted ({time.time() - t0:.0f}s): {int((support >= MIN_SUPPORT).sum())} "
          f"candidates backed by >= {MIN_SUPPORT} frames")

    # 3. greedy, best backed first. Each SAM3 line belongs to one leaf: a
    # candidate most of whose lines a kept leaf already took is that leaf seen
    # from another pair of frames (or the same leaf with base and tip swapped)
    order = np.lexsort((err, -support))
    leaves, kept, taken = [], [], set()
    same = SAME_MM / mpu
    for c in order:
        if support[c] < MIN_SUPPORT:
            break
        mine = set(member[c][member[c] >= 0].tolist())
        free = mine - taken
        if len(free) < MIN_SUPPORT or len(free) < 0.5 * len(mine):
            continue
        if any((np.linalg.norm(B[c] - b) < same and np.linalg.norm(T[c] - t) < same)
               or (np.linalg.norm(B[c] - t) < same and np.linalg.norm(T[c] - b) < same)
               for b, t in kept):
            continue
        kept.append((B[c], T[c]))
        taken |= mine
        ks = [k for k in np.nonzero(member[c] >= 0)[0] if member[c, k] in free]
        ks = sorted(ks, key=lambda k: -np.ptp(o["line"][member[c, k]], axis=0).max())[:12]
        views = [CurveView(run, frames[k], o["line"][member[c, k]]) for k in ks]
        leaves.append((fit_curve(B[c], T[c], views, n_ctrl=5), len(free)))
    print(f"    {len(leaves)} leaves kept ({time.time() - t0:.0f}s)")
    return leaves


def run_dense(args, run: Run, labels: dict, regions: dict, mpu: float):
    rows = json.loads(args.evidence.read_text())["leaves"]
    cams = {f: Cam(run, f) for f in run.image_of}
    obs = evidence_obs(run, rows, regions, cams)
    leaves, stats = [], {}
    for apex in sorted(obs):
        got = match_dense(run, obs[apex], regions[apex], cams, mpu)
        leaves += [(apex, X) for X, _ in got]
        sup = [n for _, n in got]
        stats[apex] = {"lines": len(obs[apex]["frame"]), "leaves": len(got),
                       "median_frames_per_leaf": float(np.median(sup)) if sup else None}
        print(f"  {apex}: {stats[apex]['lines']} SAM3 lines -> {len(got)} leaves "
              f"(median {stats[apex]['median_frames_per_leaf']} frames backing each)")
    sk = as_skeleton(run, leaves, None, regions, mpu)
    sc, pairs = score(run, labels, sk)
    draw_score(run, labels, pairs, args.out / "score_dense.jpg", f"SAM3 lines matched by geometry")
    print(f"  on your {len(labels)} labelled frames (none used): found {sc['all']['matched']}/"
          f"{sc['all']['labelled']} ({sc['all']['found']:.0%}), precision {sc['all']['precision']:.0%}")
    for a in sorted(k for k in sc if k != "all"):
        print(f"    {a}: found {sc[a]['matched']}/{sc[a]['labelled']}, precision "
              f"{(sc[a]['precision'] or 0):.0%}, tip {sc[a]['tip_px'] or float('nan'):.1f} px")
    base = json.loads((args.base_skeleton / "skeleton.json").read_text()) if args.base_skeleton else None
    d = args.out / "dense_all"
    d.mkdir(exist_ok=True)
    (d / "skeleton.json").write_text(json.dumps(as_skeleton(run, leaves, base, regions, mpu), indent=1))
    (args.out / "dense.json").write_text(json.dumps({"evidence": str(args.evidence), "stats": stats,
                                                     "score": sc}, indent=1))


def main():
    global MIN_SUPPORT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--labels", type=Path, help="LabelMe JSONs (default: <workdir>/p1/frames)")
    ap.add_argument("--regions", type=Path, required=True, help="apex_labels.py's apex_regions.npz")
    ap.add_argument("--base-skeleton", type=Path,
                    help="s2d output whose leaves outside the apices complete match_all")
    ap.add_argument("--evidence", type=Path,
                    help="skeleton_2d.py evidence_2d.json: match SAM3's lines over every frame "
                         "instead of the hand-drawn ones (which then only score)")
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT,
                    help=f"--evidence: frames that must back a leaf (default {MIN_SUPPORT}); higher "
                         "keeps fewer, surer leaves")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    MIN_SUPPORT = args.min_support
    args.out.mkdir(parents=True, exist_ok=True)
    mpu = json.loads((args.workdir / "tag_scale.json").read_text())["mm_per_unit"]
    run = Run(args.workdir)
    labels = load(args.labels or args.workdir / "p1" / "frames")
    regions = dict(np.load(args.regions))
    frames = sorted(labels)
    if args.evidence:
        run_dense(args, run, labels, regions, mpu)
        return

    report = {"settings": {"pair_px": PAIR_PX, "support_px": SUPPORT_PX, "fan_deg": FAN_DEG},
              "leave_one_out": {}}
    m = l = s_ = 0
    for out_frame in frames:
        leaves, stats = run_match(run, labels, [f for f in frames if f != out_frame], regions, mpu)
        sk = as_skeleton(run, leaves, None, regions, mpu)
        sc, pairs = score(run, {out_frame: labels[out_frame]}, sk)
        draw_score(run, {out_frame: labels[out_frame]}, pairs,
                   args.out / f"score_without_{out_frame}.jpg", f"matched without {out_frame}")
        report["leave_one_out"][out_frame] = {"leaves": len(leaves), "stats": stats, "score": sc}
        m += sc["all"]["matched"]
        l += sc["all"]["labelled"]
        s_ += sc["all"]["skeleton_in_box"]
        print(f"  without {out_frame}: {len(leaves)} leaves from the other {len(frames) - 1} frames; "
              f"on {out_frame}: found {sc['all']['matched']}/{sc['all']['labelled']}, "
              f"precision {sc['all']['precision']:.0%}")
    report["pooled"] = {"found": m / max(l, 1), "precision": m / max(s_, 1), "matched": m,
                        "labelled": l, "skeleton_in_box": s_}
    print(f"  pooled over the left-out frames: found {m}/{l} ({m / max(l, 1):.0%}), "
          f"precision {m / max(s_, 1):.0%}")

    leaves, stats = run_match(run, labels, frames, regions, mpu)
    report["all_frames"] = stats
    base = json.loads((args.base_skeleton / "skeleton.json").read_text()) if args.base_skeleton else None
    d = args.out / "match_all"
    d.mkdir(exist_ok=True)
    (d / "skeleton.json").write_text(json.dumps(as_skeleton(run, leaves, base, regions, mpu), indent=1))
    for apex, st in stats.items():
        print(f"  all frames, {apex}: {st['lines']} lines -> {st['leaves']} leaves "
              f"(frames per leaf {st['frames_per_leaf']}), {st['lines_unmatched']} lines unmatched")
    (args.out / "match.json").write_text(json.dumps(report, indent=1))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
