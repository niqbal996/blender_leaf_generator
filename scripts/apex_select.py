#!/usr/bin/env python
"""Apex leaves chosen by drawing candidates into every view: the set that explains SAM3's masks.

    python scripts/apex_select.py --workdir <specimen>/plant --out <specimen>/plant/debug/apex_select \
        --regions <dir>/apex_regions.npz --zoom <dir>/apex_zoom_fair \
        --evidence <dir>/apex_zoom_fair/s2d_zoom_fair/evidence_2d.json \
        --candidates <specimen>/plant/p5x <specimen>/plant/p5x_p4g <s2d dir> ... \
        [--base <specimen>/plant/p5x]

Steps 1-2 of docs/apex-plan.md (option 3, from proposals). Diagnostic --
writes into --out only.

Candidates per apex (labelled region, apex_labels.py): the leaves geometry
matches out of SAM3's lines (apex_match.py, generously: >= CAND_SUPPORT
frames), and every --candidates skeleton's leaves whose midrib middle lies
in the region. Each becomes a leaf-shaped strip -- the curve, WIDTH of its
length at the widest, narrow over its first STALK -- at each of ROLLS (a
curve alone does not say which way its blade faces).

Step 1, the renderer: every strip drawn into the apex's square in every
view (apex_zoom.py's squares, GRID px), depth-tested, on the GPU, with
--base's leaves outside every apex drawn too as fixed occluders.

Step 2, the selection: greedily add the candidate that most raises

  J = sum over views and SAM3 masks of the visible pixels of the one leaf
      that mask is the best mask of (each mask pays one leaf, each leaf is
      paid by its best mask only)
    - ALPHA * every visible leaf pixel
    - LEAF_COST per leaf

so a leaf earns what it covers of one zoomed SAM3 mask, pays for pixels off
it, and a duplicate -- hidden behind its twin -- earns nothing. Stop when
nothing raises J. Masks are SAM3's on the zoomed squares, so no leaf ids
across views are involved anywhere.

Writes: skeleton.json (--base with its apex leaves replaced by the selected
ones, each one curve from the stem to its tip -- its base is the end at the
stem, as SAM3's base/tip call can be swapped -- and the stem tree ended at
the leaves, stem_tree.end_axes), select.json (per
apex: candidates and kept per source, J), apex_<X>.jpg (photo | SAM3 masks
| kept leaves as drawn, 8 views).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apex_match                                                        # noqa: E402
from apex_zoom import cut                                               # noqa: E402
from compare_runs import track_colour                                   # noqa: E402
from p5x_views import Run                                               # noqa: E402
from pose_estimator.fine_views import source_image                     # noqa: E402
from skeleton_2d import resample                                        # noqa: E402

GRID = 128          # px of an apex square in the renderer
ROLLS = (0, 45, 90, 135)
WIDTH = 0.5         # widest blade / curve length (vogelmeere: ovate, ~0.5-0.7)
STALK = 0.15        # share of the curve, from its base, that is stalk
ALONG, ACROSS = 64, 21
ALPHA = 0.5         # cost of a visible leaf pixel off its best mask, per pixel
LEAF_COST = 0.0     # cost per leaf, px summed over views: 0, 150, 300 kept 26, 24, 23 leaves on
                    # vogelmeere -- not what decides; ALPHA (majority on one mask) does
MIN_MASK = 40       # SAM3 masks smaller than this at GRID are specks
CAND_SUPPORT = 4    # geometry candidates: frames that must back a leaf
SAME_MM = 0.7       # a candidate this close (mean curve distance) to a kept leaf is it
BATCH = 48


def region_reach(regions):
    out = {}
    for name, pts in regions.items():
        c = pts.mean(axis=0)
        out[name] = (c, float(np.percentile(np.linalg.norm(pts - c, axis=1), 95)))
    return out


def in_region(curve_w, reach):
    c, r = reach
    return np.linalg.norm(curve_w[len(curve_w) // 2] - c) < r


def strip(curve_w: np.ndarray, roll_deg: float) -> np.ndarray:
    """(ALONG * ACROSS, 3) points of a leaf-shaped strip along the curve."""
    C = resample(curve_w, ALONG)
    length = np.linalg.norm(np.diff(C, axis=0), axis=1).sum()
    T = np.gradient(C, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    a = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n = np.cross(T[0], a)
    n /= np.linalg.norm(n)
    N = np.zeros_like(C)
    for i in range(len(C)):                          # parallel transport along the curve
        n = n - (n @ T[i]) * T[i]
        n /= max(np.linalg.norm(n), 1e-12)
        N[i] = n
    th = np.radians(roll_deg)
    W = np.cos(th) * N + np.sin(th) * np.cross(T, N)
    s = np.linspace(0.0, 1.0, ALONG)
    blade = np.clip((s - STALK) / (1 - STALK), 0, 1)
    half = 0.5 * WIDTH * length * np.where(s < STALK, 0.06, np.sin(np.pi * blade) ** 0.7 + 0.06)
    u = np.linspace(-1.0, 1.0, ACROSS)
    return (C[:, None] + (half[:, None] * u[None])[..., None] * W[:, None]).reshape(-1, 3)


class Views:
    """One apex's squares in every frame, and SAM3's zoomed masks there as label maps."""

    def __init__(self, run: Run, zoom: Path, apex: str, dev):
        names, crops, ups, labels = [], [], [], []
        for path in sorted((zoom / "sam3").glob(f"apex{apex}_pass*.npz")):
            z = np.load(path)
            frames = [str(f) for f in z["frames"]]
            masks = defaultdict(list)
            for key in z.files:
                if key.startswith("m_"):
                    _, i, obj = key.split("_")
                    masks[int(i)].append(z[key])
            for i, frame in enumerate(frames):
                lab = np.zeros((GRID, GRID), np.int16)
                ms = []
                for packed in masks.get(i, []):
                    m = np.unpackbits(packed)[:1008 * 1008].reshape(1008, 1008)
                    small = cv2.resize(m.astype(np.float32), (GRID, GRID),
                                       interpolation=cv2.INTER_AREA) > 0.5
                    if small.sum() >= MIN_MASK:
                        ms.append(small)
                for k, m in enumerate(sorted(ms, key=lambda m: -m.sum())):   # small on top
                    lab[m] = k + 1
                names.append(frame)
                crops.append([float(v) for v in z[f"crop_{i}"]])
                ups.append(float(z[f"up_{i}"]))
                labels.append(lab)
        self.frames = names
        cams = [run.rec.cameras[run.image_of[f].camera_id] for f in names]
        poses = [run.image_of[f].cam_from_world() for f in names]
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64, device=dev)
        self.R = t([p.rotation.matrix() for p in poses])
        self.t = t([np.asarray(p.translation) for p in poses])
        self.f = t([c.params[0] for c in cams])
        self.cx = t([c.params[1] for c in cams])
        self.cy = t([c.params[2] for c in cams])
        self.k = t([c.params[3] for c in cams])
        self.up = t(ups)
        crop = t(crops)
        self.x0, self.y0, self.side = crop[:, 0], crop[:, 1], crop[:, 2]
        self.labels = torch.tensor(np.stack(labels), dtype=torch.long, device=dev).reshape(len(names), -1)
        self.K = int(self.labels.max().item()) + 1

    def depth(self, pts: torch.Tensor) -> torch.Tensor:
        """(N, M, 3) world points of N leaves -> (N, V, GRID*GRID) nearest depth, inf where none."""
        N, M, _ = pts.shape
        V = len(self.frames)
        c = torch.einsum("vij,nmj->nvmi", self.R, pts) + self.t[None, :, None]
        z = c[..., 2]
        x, y = c[..., 0] / z, c[..., 1] / z
        fac = 1 + self.k[None, :, None] * (x * x + y * y)
        u = (self.f[None, :, None] * x * fac + self.cx[None, :, None]) * self.up[None, :, None]
        v = (self.f[None, :, None] * y * fac + self.cy[None, :, None]) * self.up[None, :, None]
        gu = ((u - self.x0[None, :, None]) * GRID / self.side[None, :, None]).floor().long()
        gv = ((v - self.y0[None, :, None]) * GRID / self.side[None, :, None]).floor().long()
        ok = (z > 0) & (gu >= 0) & (gu < GRID) & (gv >= 0) & (gv < GRID)
        pix = torch.where(ok, gv * GRID + gu, torch.full_like(gu, GRID * GRID))
        out = torch.full((N, V, GRID * GRID + 1), float("inf"), dtype=torch.float32, device=pts.device)
        out.scatter_reduce_(2, pix, z.float(), reduce="amin")
        return out[..., :-1]


def credit(slots: torch.Tensor, labels: torch.Tensor, n_slots: int, K: int) -> torch.Tensor:
    """slots (B, V, P): leaf slot per pixel (-1 none / occluder). -> J without leaf costs, (B,)."""
    B, V, P = slots.shape
    valid = slots >= 0
    lab = labels[None].expand(B, V, P)
    bi = torch.arange(B, device=slots.device)[:, None, None].expand(B, V, P)
    vi = torch.arange(V, device=slots.device)[None, :, None].expand(B, V, P)
    idx = ((bi * V + vi) * n_slots + slots.clamp(min=0)) * K + lab
    counts = torch.zeros(B * V * n_slots * K, device=slots.device)
    counts.scatter_add_(0, idx[valid], torch.ones(int(valid.sum()), device=slots.device))
    counts = counts.view(B, V, n_slots, K)
    total = counts.sum(-1)                                              # visible px per leaf
    best_val, best_k = counts[..., 1:].max(-1)                          # its best mask
    paid = torch.zeros(B, V, K - 1, device=slots.device)
    paid.scatter_reduce_(2, best_k, best_val, reduce="amax")            # each mask pays one leaf
    return paid.sum((1, 2)) - ALPHA * total.sum((1, 2))


def select(views: Views, cands, occluders, dev, mpu: float, log):
    """Greedy: indices of kept candidates, J history."""
    V = len(views.frames)
    P = GRID * GRID
    D = torch.cat([views.depth(torch.tensor(np.stack([strip(c["curve"], c["roll"])
                                                     for c in cands[i:i + BATCH]]),
                                           dtype=torch.float64, device=dev))
                   for i in range(0, len(cands), BATCH)])
    cur_depth = torch.full((V, P), float("inf"), device=dev)
    cur_slot = torch.full((V, P), -1, dtype=torch.long, device=dev)
    if occluders:
        Do = torch.cat([views.depth(torch.tensor(np.stack([strip(c, 0) for c in occluders[i:i + BATCH]]),
                                                 dtype=torch.float64, device=dev))
                        for i in range(0, len(occluders), BATCH)]).amin(0)
        cur_depth = torch.minimum(cur_depth, Do)
        cur_slot[torch.isfinite(Do)] = -2
    alive = torch.ones(len(cands), dtype=torch.bool, device=dev)
    kept, J_cur, history = [], 0.0, []
    dense = [resample(c["curve"], 30) for c in cands]
    same = SAME_MM / mpu
    while alive.any():
        m = len(kept)
        best_J, best_c = -np.inf, None
        ids = torch.nonzero(alive).flatten()
        for i in range(0, len(ids), BATCH):
            cb = ids[i:i + BATCH]
            front = D[cb] < cur_depth[None]
            slots = torch.where(front, torch.full_like(cur_slot, m)[None], cur_slot[None])
            J = credit(slots, views.labels, m + 1, views.K) - LEAF_COST * (m + 1)
            j = int(torch.argmax(J))
            if J[j].item() > best_J:
                best_J, best_c = J[j].item(), int(cb[j])
        if best_c is None or best_J <= J_cur:
            break
        kept.append(best_c)
        J_cur = best_J
        history.append(round(best_J, 1))
        front = D[best_c] < cur_depth
        cur_depth = torch.minimum(cur_depth, D[best_c])
        cur_slot = torch.where(front, torch.full_like(cur_slot, m), cur_slot)
        g = cands[best_c]["group"]
        for i, c in enumerate(cands):                 # its other rolls, and copies of it
            if c["group"] == g or np.mean(np.linalg.norm(dense[i] - dense[best_c], axis=1)) < same:
                alive[i] = False
        log(f"      + {cands[best_c]['source']:<12s} roll {cands[best_c]['roll']:3d}  J {best_J:9.0f}")
    return kept, history, cur_slot


def montage(path: Path, run: Run, views: Views, slots: torch.Tensor, kept_colours, n=8):
    tiles = []
    for i in np.linspace(0, len(views.frames) - 1, n).round().astype(int):
        frame = views.frames[i]
        photo = cv2.imread(str(source_image(run.workdir, frame)))
        x0, y0, side = (int(v) for v in (views.x0[i].item(), views.y0[i].item(), views.side[i].item()))
        img = cv2.cvtColor(cut(photo, x0, y0, side), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
        lab = views.labels[i].view(GRID, GRID).cpu().numpy()
        sl = slots[i].view(GRID, GRID).cpu().numpy()
        lab_rgb = np.zeros((GRID, GRID, 3), np.uint8)
        for k in range(1, lab.max() + 1):
            lab_rgb[lab == k] = track_colour(k)
        sel_rgb = np.zeros((GRID, GRID, 3), np.uint8)
        sel_rgb[sl == -2] = (70, 70, 70)
        for s, col in enumerate(kept_colours):
            sel_rgb[sl == s] = col
        up = lambda a: cv2.resize(a, (384, 384), interpolation=cv2.INTER_NEAREST)
        row = np.hstack([img, cv2.addWeighted(img, 0.4, up(lab_rgb), 0.6, 0),
                         cv2.addWeighted(img, 0.4, up(sel_rgb), 0.6, 0)])
        cv2.putText(row, f"{frame}: photo | SAM3 masks | kept leaves (grey: other leaves)", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(row)
    cv2.imwrite(str(path), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])


def main():
    global ALPHA, LEAF_COST
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--zoom", type=Path, required=True, help="apex_zoom.py --regions output dir")
    ap.add_argument("--evidence", type=Path, help="evidence_2d.json for geometry candidates")
    ap.add_argument("--candidates", type=Path, nargs="*", default=[],
                    help="skeleton dirs whose apex leaves are candidates too")
    ap.add_argument("--base", type=Path, help="P5x skeleton dir the result replaces apex leaves in "
                                              "(default: <workdir>/p5x); its stem tree is ended at "
                                              "the leaves (stem_tree.end_axes)")
    ap.add_argument("--alpha", type=float, default=ALPHA, help=f"default {ALPHA}")
    ap.add_argument("--leaf-cost", type=float, default=LEAF_COST, help=f"default {LEAF_COST}")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ALPHA, LEAF_COST = args.alpha, args.leaf_cost
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    mpu = json.loads((args.workdir / "tag_scale.json").read_text())["mm_per_unit"]
    run = Run(args.workdir)
    regions = dict(np.load(args.regions))
    reach = region_reach(regions)
    base_dir = args.base or args.workdir / "p5x"
    base = json.loads((base_dir / "skeleton.json").read_text())

    def leaf_curve(leaf):
        pet = np.asarray(leaf.get("petiole") or [], float).reshape(-1, 3)
        rib = np.asarray(leaf["midrib"], float)
        return run.to_world(np.vstack([pet, rib]) if len(pet) >= 2 else rib)

    # candidates per apex
    per = defaultdict(list)
    group = 0

    def add(apex, source, curve):
        nonlocal group
        for r in ROLLS:
            per[apex].append({"source": source, "curve": curve, "roll": r, "group": group})
        group += 1

    if args.evidence:
        rows = json.loads(args.evidence.read_text())["leaves"]
        cams = {f: apex_match.Cam(run, f) for f in run.image_of}
        obs = apex_match.evidence_obs(run, rows, regions, cams)
        apex_match.MIN_SUPPORT = CAND_SUPPORT
        for name in sorted(obs):
            with contextlib.redirect_stdout(io.StringIO()):
                got = apex_match.match_dense(run, obs[name], regions[name], cams, mpu)
            for X, _ in got:
                add(name, "geometry", X)
    for d in args.candidates:
        sk = json.loads((d / "skeleton.json").read_text())
        for leaf in sk["leaves"]:
            X = leaf_curve(leaf)
            for name in regions:
                if in_region(X, reach[name]):
                    add(name, d.name, X)
    occluders = [leaf_curve(l) for l in base["leaves"]
                 if not any(in_region(leaf_curve(l), reach[n]) for n in regions)]

    report, kept_all = {}, []
    log = lambda s: print(s, flush=True)
    for name in sorted(regions):
        cands = per[name]
        apex = name.replace("apex_", "")
        t0 = time.time()
        views = Views(run, args.zoom, apex, dev)
        by_src = defaultdict(int)
        for c in cands:
            by_src[c["source"]] += 1
        log(f"  {name}: {len(cands) // len(ROLLS)} candidate leaves x {len(ROLLS)} rolls "
            f"({', '.join(f'{k} {v // len(ROLLS)}' for k, v in sorted(by_src.items()))}), "
            f"{len(views.frames)} views, {views.K - 1} SAM3 masks at most per view")
        kept, history, slots = select(views, cands, occluders, dev, mpu, log)
        kept_src = defaultdict(int)
        for i in kept:
            kept_src[cands[i]["source"]] += 1
        log(f"    kept {len(kept)} ({', '.join(f'{k} {v}' for k, v in sorted(kept_src.items()))}) "
            f"in {time.time() - t0:.0f}s")
        colours = [track_colour(len(kept_all) + j) for j in range(len(kept))]
        montage(args.out / f"{name}.jpg", run, views, slots, colours)
        kept_all += [(name, cands[i]) for i in kept]
        report[name] = {"candidates": {k: v // len(ROLLS) for k, v in by_src.items()},
                        "kept": dict(kept_src), "J": history}
        del views, slots
        torch.cuda.empty_cache()

    # the result: --base with its apex leaves replaced, one curve per leaf
    to_plant = lambda X: ((np.atleast_2d(X) - run.origin) @ run.rotation.T)
    leaves = [dict(l) for l in base["leaves"]
              if not any(in_region(leaf_curve(l), reach[n]) for n in regions)]
    replaced = len(base["leaves"]) - len(leaves)
    from pose_estimator import stem_tree
    from pose_estimator.leaf_skeleton import smooth_curve
    length = lambda P: float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    # Apex leaves radiate from their apex: a leaf's base is its end nearer the
    # apex node (SAM3's base/tip call can be swapped), and the node is where
    # the bases meet, their median. The two settle each other: start at the
    # region's centre, orient, take the median of the bases, orient again,
    # until nothing turns. (Orienting by the nearest stem instead turned
    # leaves at C and D, where several branches pass close by.)
    picked = defaultdict(list)
    for j, (name, c) in enumerate(kept_all):
        picked[name].append([j, c, to_plant(resample(c["curve"], 24))])
    nodes = {}
    for name, rows in picked.items():
        node = to_plant(reach[name][0])[0]
        for _ in range(20):
            for r in rows:
                if np.linalg.norm(r[2][-1] - node) < np.linalg.norm(r[2][0] - node):
                    r[2] = r[2][::-1]
            moved = np.median(np.stack([r[2][0] for r in rows]), axis=0)
            if np.allclose(moved, node):
                break
            node = moved
        nodes[name] = node

    # Stems and branches end at base's leaves outside the apices (end_axes).
    # Then each apex's shoot -- the stem or branch nearest its node -- runs on
    # to the node, or a short branch leaves the shoot for it when the node is
    # beside it, and every apex leaf begins on that shoot.
    starts = [l["curve"][0] for l in leaves if l.get("curve") and l.get("starts_on") == "stem"]
    axes, _ = stem_tree.end_axes(base.get("axes") or [], starts)
    trunk_ids = [i for i, a in enumerate(axes) if a["kind"] != "petiole" and len(a["points"]) >= 2]
    shoot = {}
    for name, node in nodes.items():
        if not trunk_ids:
            break
        hits = [(stem_tree._on_polyline(np.asarray(axes[i]["points"]), node), i) for i in trunk_ids]
        (s_at, near, _), i = min(hits, key=lambda h: h[0][2])
        line = np.asarray(axes[i]["points"], float)
        if s_at >= length(line) - 1e-9:
            axes[i]["points"] = np.vstack([line, node]).round(6).tolist()      # the shoot runs on
            shoot[name] = i
        else:
            axes.append({"id": len(axes), "order": axes[i]["order"] + 1, "parent": i,
                         "kind": "branch", "leaf": None,                       # beside the axis
                         "points": np.vstack([near, node]).round(6).tolist()})
            trunk_ids.append(len(axes) - 1)
            shoot[name] = len(axes) - 1

    def apex_leaf(j, name, c, curve, blade_at):
        """One curve from the shoot to the tip, smoothed as the tracer smooths."""
        curve = smooth_curve(curve, run.voxel)
        b = int(np.argmin(np.linalg.norm(curve - blade_at, axis=1)))
        return {"id": None, "track": f"sel_{name}_{j}_{c['source']}", "sam3_ids": [f"sel_{name}_{j}"],
                "tip": curve[-1].tolist(), "base": curve[b].tolist(), "curve": curve.tolist(),
                "blade_start": b, "midrib": curve[b:].tolist(),
                "petiole": curve[:b + 1].tolist() if b > 0 else [],
                "starts_on": "stem", "source": c["source"], "roll": c["roll"],
                "colour_bgr": list(track_colour(j)), "midrib_mm": float(length(curve[b:]) * mpu)}

    # Every apex leaf joins the nearest stem or branch -- its own apex's shoot,
    # or a neighbouring one where shoots crowd (apex C is the main stem's top,
    # with branches through it) -- and leads away from where it joins: its tip
    # is farther from that point than its base. One that would fold back
    # (vogelmeere C: blades turning 132-139 deg back past the stem) is the other
    # way round, and joins at its other end.
    trunk_lines = [np.asarray(axes[i]["points"], float) for i in trunk_ids]

    def join(P):
        _, q, d = min((stem_tree._on_polyline(t, P[0]) for t in trunk_lines), key=lambda h: h[2])
        return q, d, np.linalg.norm(P[-1] - q) - np.linalg.norm(P[0] - q)

    joined, flipped = [], 0
    for name, rows in picked.items():
        for j, c, P in rows:
            if not trunk_lines:
                leaves.append(apex_leaf(j, name, c, P, P[0]))
                continue
            q, d, away = join(P)
            if away <= 0:
                qf, df, away_f = join(P[::-1])
                if away_f > away:
                    P, q, d = P[::-1], qf, df
                    flipped += 1
            joined.append(d * mpu)
            leaves.append(apex_leaf(j, name, c, np.vstack([q, P]) if d > 0 else P, P[0]))
    if flipped:
        log(f"  {flipped} apex leaves turned to lead away from the stem they join")
    if joined:
        log(f"  every apex leaf joins its shoot: {np.median(joined):.1f} mm median "
            f"(max {np.max(joined):.1f})")

    # once more, now that every leaf is on the tree: an axis left beside a shoot
    # that was run on to its apex node ends at its last leaf too (no stubs)
    on_tree = [l for l in leaves if l.get("curve") and l.get("starts_on") == "stem"]
    axes, attach = stem_tree.end_axes(axes, [l["curve"][0] for l in on_tree])
    for l, q in zip(on_tree, attach):
        if np.linalg.norm(q - np.asarray(l["curve"][0])) > 0:
            whole = np.vstack([q, np.asarray(l["curve"], float)])
            blade_at = np.asarray(l["curve"], float)[int(l.get("blade_start") or 0)]
            curve = smooth_curve(whole, run.voxel)
            b = int(np.argmin(np.linalg.norm(curve - blade_at, axis=1)))
            l.update(curve=curve.tolist(), blade_start=b, base=curve[b].tolist(),
                     midrib=curve[b:].tolist(), petiole=curve[:b + 1].tolist() if b > 0 else [])
    for i, l in enumerate(leaves):
        l["id"] = i
    main_axis = next((a for a in axes if a["kind"] == "stem"), None)
    doc = dict(base, leaves=leaves, axes=axes,
               stem=main_axis["points"] if main_axis else base.get("stem", []),
               built={"by": "scripts/apex_select.py", "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "base": str(base_dir)})
    (args.out / "skeleton.json").write_text(json.dumps(doc, indent=1))
    report["base"] = {"dir": str(base_dir), "apex_leaves_replaced": replaced, "kept": len(kept_all),
                      "settings": {"grid": GRID, "rolls": ROLLS, "width": WIDTH, "alpha": ALPHA,
                                   "leaf_cost": LEAF_COST, "cand_support": CAND_SUPPORT}}
    (args.out / "select.json").write_text(json.dumps(report, indent=1))
    log(f"  {base_dir.name}: {replaced} apex leaves replaced by {len(kept_all)} selected -> "
        f"{args.out / 'skeleton.json'}")


if __name__ == "__main__":
    main()
