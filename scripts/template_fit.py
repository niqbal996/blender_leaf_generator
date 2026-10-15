#!/usr/bin/env python
"""Every leaf of the plant placed from its flat-lay twin: shape known, pose fitted to P4g.

    PYTHONPATH=src:scripts python scripts/template_fit.py --workdir <specimen>/plant \
        [--flat <specimen>/leaf_pose] [--match <workdir>/apex/flatlay/flatlay_match.json] \
        [--templates <flat-lay ids>] [--base slide|origin] [--out <workdir>/apex/templates]

(run_apex.sh runs it when <specimen>/leaf_pose/leaves.json exists.) On a plant that is no rosette the
leaves start along the stem: --base slide always, no heart, the stretch to a leaf drawn as its petiole.

After the capture the plant was taken apart and its leaves shot flat (src/leaf_pose): every leaf's
true outline and length, in mm, and how many leaves it has. The heart's young leaves leave no
per-leaf trace in 3D or in SAM3's masks (docs/apex-plan.md 0.0f), so each flat-lay leaf is placed
as it is and only its pose is fitted. Nothing says which leaves are the heart's: every flat-lay
leaf is a template (--templates narrows it, for tests).

Which points each template takes (flatlay_match.py: flat-lay outlines against the flattened P5x-on-
P4g leaves, paired one to one above IoU 1/2):
  matched    its P5x leaf's points
  unmatched  either a P5x leaf left unpaired (a small leaf's outline match can fail) or the heart:
             the tissue no leaf mask claimed (P2's stem mask covers the heart; a rosette's P5x
             calls it unassigned leaf tissue) within its own length of its base; every assignment
             is fitted and the one whose leaves sit best kept
Stage 1 fits the matched templates, each from its P5x leaf's curve; stage 2 the unmatched ones,
with the matched ones fixed; a heart leaf from several starts (azimuths round the axis x two
elevations).

The leaf: its flat-lay outline (half-width per side along its midrib, from the petiole origin)
on a midrib that is a quadratic Bezier from its base in the plane through the plant's up axis --
azimuth, elevation (from the axis), incurve (towards or away from the axis, at most 90 deg: a
rosette's leaves arch out), the split of its two legs -- scaled to the flat-lay length, the blade rolled about it
within +-90 deg (upper face towards the axis). Every leaf starts at the origin, with no petiole: a
rosette's leaves join it directly (the user, 2026-10-15: bases slid along P5x's curves put the
leaves up off the origin, past their tips, with a petiole drawn to them).
The blade stays flat across, as laid flat: the plant's own leaves measure nearly flat across (cup
rise 6-8 % of the half-width, leaf_curvature.py), and a free fold closed the young heart leaf shut to
fit the crowded heart (2026-10-15, the user: its wings must spread).

The origin: P5x's crown (p5x_p4g/skeleton.json: where the user sees the leaves join, 2026-10-15);
the axis: plant up.

Cost, mm, robust (Geman-McClure at half the plant's skin gap, leaf_instances.skin_partners):
  on     each template's surface to its own points (mean); for a leaf placed on a P5x leaf, half
         that and half the leaf's points to its surface (it must cover the leaf it is)
  off    its surface on points that are not its own -- other P5x leaves', placed templates' (share)
  apart  its surface on another template being fitted (share)
  past   a leaf placed on a P5x leaf: its surface past that leaf's tip -- beyond the plane through
         the tip across the curve's last stretch (the user: P5x's tips are the leaves' extremities)

Writes <out>/skeleton.json (one leaf per flat-lay leaf, with its blade_mesh; `origin`/`crown` = P5x's
crown), template_fit.json (per leaf: matched to, pose, support = share of its surface on its own
points, off, apart) and template_fit.jpg (the views that look down into the heart: photo | the leaves
where P4g does not hide them).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from math import comb
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flatlay_match import flat_leaf                                     # noqa: E402
from heart_fit import Plant, skin_gap                                   # noqa: E402
from pose_estimator import cloud_source                                 # noqa: E402
from pose_estimator.cli.leaf_instances import SKELETON, leaf_palette    # noqa: E402
from pose_estimator.leaf_skeleton import UNASSIGNED                     # noqa: E402
from pose_estimator.semantic import camera_from_colmap                  # noqa: E402

ALONG, ACROSS = 48, 13
ITERS = 400
LR = 0.03
SAMPLE = 20000          # points per evidence set, at most (a uniform subsample)


def template(l, n=ALONG):
    """A flat-lay leaf: its length (petiole origin to tip, mm) and half-width per side at n points
    along it (left = +v walking base to tip)."""
    s, v, L = flat_leaf(l)
    S = np.linspace(0, L, n)
    left, right = np.zeros(n), np.zeros(n)
    w = L / (n - 1)
    for i, x in enumerate(S):
        m = np.abs(s - x) <= w
        if m.any():
            left[i] = max(0.0, v[m].max())
            right[i] = max(0.0, -v[m].min())
    return {"id": l["id"], "length": float(L), "left": left, "right": right}


def bernstein(t):
    return torch.stack([comb(2, j) * t ** j * (1 - t) ** (2 - j) for j in range(3)], -1)


class Templates(torch.nn.Module):
    """Template leaves in a frame with up axis a, mm; each leaf's base slides along its own polyline
    (`routes`: its P5x leaf's curve from the crown, or a single point)."""

    INCURVE = (-0.5 * np.pi, 0.5 * np.pi)   # at most a right angle either way: away from the axis as apex_fit's
    #   rosette rule allows, and towards it no more -- thistle_x's placed leaves turn -85..+7 deg; only
    #   the heart leaf, fitted to the vase's wall, hooked back 113 deg (the user: a U-turn)

    def __init__(self, temps, axis, starts, dev, routes):
        super().__init__()
        self.routes = []
        for R in routes:
            R = np.asarray(R, float).reshape(-1, 3)
            if len(R) == 1:
                R = np.vstack([R, R])
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(R, axis=0), axis=1))]
            self.routes.append((torch.tensor(R, dtype=torch.float32, device=dev),
                                torch.tensor(cum, dtype=torch.float32, device=dev)))
        t = lambda x: torch.tensor(np.asarray(x, np.float32), device=dev)
        p = lambda x: torch.nn.Parameter(t(x))
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(a, ref)
        e1 /= np.linalg.norm(e1)
        self.a, self.e1, self.e2 = t(a), t(e1), t(np.cross(a, e1))
        self.L = t([x["length"] for x in temps])
        self.left = t(np.stack([x["left"] for x in temps]))
        self.right = t(np.stack([x["right"] for x in temps]))
        lg = lambda q: float(np.log(q / (1 - q)))
        self.phi = p([s["phi"] for s in starts])
        self.p_psi = p([lg(np.clip(s["psi"] / np.pi, 0.02, 0.98)) for s in starts])
        k0, k1 = self.INCURVE
        self.p_kappa = p([lg(np.clip((s["kappa"] - k0) / (k1 - k0), 0.02, 0.98)) for s in starts])
        self.p_split = p([0.0] * len(temps))
        self.p_roll = p([0.0] * len(temps))
        self.p_base = p([lg(np.clip(s.get("base_share", 0.5), 0.02, 0.98)) for s in starts])

    def pose(self):
        return {"azimuth": self.phi, "elevation": np.pi * torch.sigmoid(self.p_psi),
                "incurve": self.INCURVE[0] + (self.INCURVE[1] - self.INCURVE[0]) * torch.sigmoid(self.p_kappa),
                "split": 0.2 + 0.6 * torch.sigmoid(self.p_split), "roll": 0.5 * np.pi * torch.tanh(self.p_roll),
                "base_share": torch.sigmoid(self.p_base)}

    def bases(self):
        """Each leaf's base: at its share of the way along its route."""
        out = []
        for (R, cum), q in zip(self.routes, torch.sigmoid(self.p_base)):
            s = q * cum[-1]
            i = torch.searchsorted(cum.contiguous(), s.detach().reshape(1)).clamp(1, len(cum) - 1)[0]
            w = ((s - cum[i - 1]) / (cum[i] - cum[i - 1]).clamp(min=1e-9)).clamp(0, 1)
            out.append(R[i - 1] * (1 - w) + R[i] * w)
        return torch.stack(out)

    def surface(self):
        """(n, ALONG, ACROSS, 3) blade points, (n, ALONG, 3) midrib."""
        po = self.pose()
        r = torch.cos(po["azimuth"])[:, None] * self.e1 + torch.sin(po["azimuth"])[:, None] * self.e2
        d = lambda th: torch.sin(th)[:, None] * r + torch.cos(th)[:, None] * self.a
        P1 = po["split"][:, None] * d(po["elevation"])
        P2 = P1 + (1 - po["split"])[:, None] * d(po["elevation"] - po["incurve"])
        P = torch.stack([torch.zeros_like(P1), P1, P2], 1)                    # about the base
        tt = torch.linspace(0, 1, 200, device=P.device)
        D = torch.einsum("tj,njc->ntc", bernstein(tt), P)
        seg = (D[:, 1:] - D[:, :-1]).norm(dim=-1)
        cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(1)], 1)
        D = D * (self.L / cum[:, -1])[:, None, None]                          # the flat-lay length
        cum = cum * (self.L / cum[:, -1])[:, None]
        target = torch.linspace(0, 1, ALONG, device=P.device)[None] * self.L[:, None]
        idx = torch.searchsorted(cum.detach().contiguous(), target.detach().contiguous()).clamp(1, 199)
        lo, hi = cum.gather(1, idx - 1), cum.gather(1, idx)
        w = ((target - lo) / (hi - lo).clamp(min=1e-9)).clamp(0, 1)[..., None]
        g = lambda k: D.gather(1, k[..., None].expand(-1, -1, 3))
        C = g(idx - 1) * (1 - w) + g(idx) * w
        T = torch.gradient(C, dim=1)[0]
        T = T / T.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        across = torch.linalg.cross(self.a.expand_as(r), r)
        upper = torch.linalg.cross(T, across[:, None].expand_as(T))
        th = po["roll"][:, None].expand(-1, ALONG)
        W = torch.cos(th)[..., None] * across[:, None] + torch.sin(th)[..., None] * upper
        u = torch.linspace(-1, 1, ACROSS, device=P.device)
        hw = torch.where(u[None, None] < 0, self.right[:, :, None], self.left[:, :, None])    # (n, ALONG, ACROSS)
        B = self.bases()
        C = C + B[:, None]
        X = C[:, :, None] + (hw * u[None, None])[..., None] * W[:, :, None]                 # flat across
        return X, C


def rho(d, g):
    return d * d / (d * d + g * g)


def nearest(A, B):
    return torch.cdist(A, B).min(1).values if len(B) else torch.full((len(A),), float("inf"), device=A.device)


def objective(model, on_sets, off_sets, g, cover=None, tips=None):
    """J and its parts per leaf (see the module doc). `cover[k]` (a leaf placed on a P5x leaf): its
    on-cost is two-sided -- half its surface to the points, half the points to its surface: a template
    that IS that leaf must also cover it (one-sided, a 12 mm leaf lay wholly on a 23 mm cotyledon's
    points and won it, thistle_x)."""
    X, _ = model.surface()
    n = X.shape[0]
    flat = X.reshape(n, -1, 3)
    cover = cover or [False] * n
    tips = tips or [None] * n
    on, off, apart, support, past = [], [], [], [], []
    for k in range(n):
        d_on = nearest(flat[k], on_sets[k])
        c = rho(d_on, g).mean()
        if cover[k]:
            c = 0.5 * (c + rho(nearest(on_sets[k], flat[k]), g).mean())
        on.append(c)
        support.append((d_on < g).float().mean())
        if tips[k] is not None:                   # its P5x leaf's tip: no blade past it (the user, 2026-10-15)
            T_, t_ = tips[k]
            past.append(rho(torch.relu((flat[k] - T_) @ t_), g).mean())
        else:
            past.append(torch.zeros((), device=X.device))
        off.append((1 - rho(nearest(flat[k], off_sets[k]), g)).mean())
        others = [flat[j] for j in range(n) if j != k]
        apart.append((1 - rho(nearest(flat[k], torch.cat(others)), g)).mean() if others
                     else torch.zeros((), device=X.device))
    on, off, apart, past = torch.stack(on), torch.stack(off), torch.stack(apart), torch.stack(past)
    J = -(on + off + apart + past).sum()
    return J, {"on": on.detach().cpu().numpy(), "off": off.detach().cpu().numpy(),
               "apart": apart.detach().cpu().numpy(), "support": torch.stack(support).cpu().numpy(),
               "past_tip": past.detach().cpu().numpy()}


def fit(model, on_sets, off_sets, g, cover=None, iters=ITERS, tips=None):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=LR * 0.05)
    for _ in range(iters):
        opt.zero_grad()
        J, _ = objective(model, on_sets, off_sets, g, cover, tips)
        (-J).backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return objective(model, on_sets, off_sets, g, cover, tips)


def route_start(C, length, axis, e1, e2):
    """A template's start on its P5x curve C (mm, from the crown): its base where the rest of the curve is
    the flat-lay length, its pose from the curve beyond. On thistle_x every P5x tip lies 3-9 mm further
    from the crown in a straight line than the whole flat-lay leaf is long: the leaves start above it."""
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    s0 = max(0.0, cum[-1] - length)
    i = int(np.searchsorted(cum, s0))
    st = start_from_curve(C[min(i, len(C) - 2):], axis, e1, e2)
    st["base_share"] = float(np.clip(s0 / max(cum[-1], 1e-9), 0.02, 0.98))
    return st


def tip_plane(curve_mm, dev):
    """A P5x leaf's tip and the outward direction of its curve's last stretch (a tenth of it)."""
    C = np.asarray(curve_mm, float)
    q = max(1, len(C) // 10)
    d = C[-1] - C[-1 - q]
    d = d / max(np.linalg.norm(d), 1e-12)
    return (torch.tensor(C[-1], dtype=torch.float32, device=dev), torch.tensor(d, dtype=torch.float32, device=dev))


def start_from_curve(C, axis, e1, e2):
    """Azimuth, elevation and incurve of a curve from its first point (mm)."""
    a = axis / np.linalg.norm(axis)
    v = C[-1] - (C[-1] @ a) * a
    phi = float(np.arctan2(v @ e2, v @ e1))
    r = np.cos(phi) * e1 + np.sin(phi) * e2
    xz = np.stack([(C - C[0]) @ r, (C - C[0]) @ a], 1)
    q = max(1, len(xz) // 4)
    ang = lambda d: float(np.arctan2(d[0], d[1]))
    th0, th1 = ang(xz[q] - xz[0]), ang(xz[-1] - xz[-1 - q])
    return {"phi": phi, "psi": float(np.clip(th0, 0.05, np.pi - 0.05)),
            "kappa": float(np.clip(th0 - th1, Templates.INCURVE[0] + 0.05, Templates.INCURVE[1] - 0.05))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--flat", type=Path)
    ap.add_argument("--match", type=Path)
    ap.add_argument("--templates", type=int, nargs="*", help="flat-lay leaf ids (default: all)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--p5x", type=Path, help="the P5x-on-P4g folder (default <workdir>/p5x_p4g)")
    ap.add_argument("--azimuths", type=int, default=12)
    ap.add_argument("--base", choices=["origin", "slide"], default="slide",
                    help="origin: every leaf from P5x's crown; slide: a leaf placed on a P5x leaf starts along "
                         "that leaf's curve, the stretch from the crown drawn as its own narrow base")
    ap.add_argument("--draw-only", action="store_true", help="redraw template_fit.jpg from <out>/skeleton.json")
    args = ap.parse_args()
    flat_dir = args.flat or args.workdir.parent / "leaf_pose"
    match_path = args.match or args.workdir / "apex" / "flatlay" / "flatlay_match.json"
    out = args.out or args.workdir / "apex" / "templates"
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    plant = Plant(args.workdir)
    if args.p5x:
        plant.lab = np.load(args.p5x / "instances.npy").astype(int)
        plant.p5x = json.loads((args.p5x / "skeleton.json").read_text())
    # the base: P5x's crown; the axis: plant up
    plant.O = np.asarray(plant.p5x["crown"], float)
    plant.Q = (plant.P - plant.O) * plant.mpu
    plant.axis = np.array([0.0, 0.0, 1.0])
    rosette = plant.p5x.get("architecture") == "rosette"
    if not rosette and args.base == "origin":
        print("  not a rosette: its leaves start along the stem, not at the crown -- --base slide", flush=True)
        args.base = "slide"
    flat = {l["id"]: l for l in json.loads((flat_dir / "leaves.json").read_text())["leaves"]}
    ids = args.templates or sorted(flat)
    temps = [template(flat[i]) for i in ids]
    if args.draw_only:
        sk = json.loads((out / "skeleton.json").read_text())
        by = {l.get("flat_leaf"): l for l in sk["leaves"]}
        X = [plant.to_mm(np.asarray(by[i]["blade_mesh"]["points"])).reshape(by[i]["blade_mesh"]["rows"], ACROSS, 3)
             for i in ids]
        draw(plant, args.workdir, X, temps, out / "template_fit.jpg",
             json.loads((out / "template_fit.json").read_text())["skin_gap_mm"])
        return
    match = json.loads(match_path.read_text())
    pair_of = {p["flat"]: p["p5x"] for p in match["pairs"]}
    Q, lab = plant.Q, plant.lab
    r = np.linalg.norm(Q, axis=1)
    reach = max(t["length"] for t in temps)
    gap, _ = skin_gap(plant, reach)
    g = gap / 2
    a = plant.axis
    e1 = np.cross(a, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    near = r < 1.5 * reach
    leaves = sorted({int(x) for x in np.unique(lab) if x >= 0})
    matched_p5x = {pair_of[i] for i in ids if i in pair_of}
    unpaired_p5x = [x for x in leaves if x not in matched_p5x]
    T = lambda P: torch.tensor(P[::max(1, len(P) // SAMPLE)], dtype=torch.float32, device=dev)
    sk5 = {l["id"]: l for l in plant.p5x["leaves"]}
    print(f"{args.workdir.parent.name}: {len(temps)} flat-lay leaves, P5x leaves {leaves}; matched "
          f"{ {i: pair_of[i] for i in ids if i in pair_of} }, unmatched flat-lay "
          f"{[i for i in ids if i not in pair_of]}, unpaired P5x {unpaired_p5x}; skin gap {gap:.2f} mm "
          f"(scale {g:.2f}); base = P5x's crown", flush=True)

    # stage 1: the matched templates, each on its P5x leaf, off every other leaf
    m_idx = [k for k, i in enumerate(ids) if i in pair_of]
    u_idx = [k for k, i in enumerate(ids) if i not in pair_of]
    X_all, C_all, rows = [None] * len(ids), [None] * len(ids), [None] * len(ids)

    def record(idxs, model, parts):
        with torch.no_grad():
            X, C = model.surface()
            po = {k_: v.detach().cpu().numpy() for k_, v in model.pose().items()}
        for j, k in enumerate(idxs):
            X_all[k], C_all[k] = X[j].cpu().numpy(), C[j].cpu().numpy()
            rows[k] = {"flat_leaf": ids[k], "p5x_match": pair_of.get(ids[k]), "length_mm": round(temps[k]["length"], 2),
                       "base_from_crown_mm": round(float(np.linalg.norm(C_all[k][0])), 2),
                       **{f"{n_}_deg": round(float(np.degrees(po[n_][j])), 1)
                          for n_ in ("azimuth", "elevation", "incurve", "roll")},
                       "support": round(float(parts["support"][j]), 3), "off": round(float(parts["off"][j]), 3),
                       "apart": round(float(parts["apart"][j]), 3),
                       "past_tip": round(float(parts["past_tip"][j]), 3)}

    if m_idx:
        on = [T(Q[lab == pair_of[ids[k]]]) for k in m_idx]
        off = [T(Q[near & (lab >= 0) & (lab != pair_of[ids[k]])]) for k in m_idx]
        curves = [plant.to_mm(sk5[pair_of[ids[k]]]["curve"]) for k in m_idx]
        if args.base == "slide":
            st = [route_start(C, temps[k]["length"], a, e1, e2) for C, k in zip(curves, m_idx)]
            routes = curves
        else:
            st = [start_from_curve(C, a, e1, e2) for C in curves]
            routes = [np.zeros(3)] * len(m_idx)
        model = Templates([temps[k] for k in m_idx], a, st, dev, routes).to(dev)
        J1, parts = fit(model, on, off, g, [True] * len(m_idx), tips=[tip_plane(C, dev) for C in curves])
        record(m_idx, model, parts)
        print(f"  stage 1: {len(m_idx)} matched templates fitted ({time.time() - t0:.0f}s)", flush=True)

    # stage 2: each unmatched template either on a P5x leaf no template matched (its outline match can
    # fail on a small leaf: thistle_x's cotyledon, IoU 0.38) -- base sliding along that leaf's curve --
    # or, on a rosette, in the heart: base at the middle of the ring where the matched leaves start (the
    # mean of their fitted bases), on the tissue no leaf mask claimed within its length of there. Each
    # template is fitted alone on each candidate; they are assigned one to one by those fits (the heart
    # takes any number), then refitted together, off the placed leaves and every P5x leaf not their own.
    # (Every combination fitted, as first done, does not scale past a few leaves.)
    assigned = {ids[k]: pair_of[ids[k]] for k in m_idx}
    if u_idx:
        from scipy.optimize import linear_sum_assignment
        placed = np.concatenate([X_all[k].reshape(-1, 3) for k in m_idx]) if m_idx else np.zeros((0, 3))
        heart_base = np.mean([C_all[k][0] for k in m_idx], 0) if (m_idx and args.base == "slide") else np.zeros(3)
        r_h = np.linalg.norm(Q - heart_base, axis=1)
        cands = list(unpaired_p5x) + (["heart"] if rosette else [])
        grid = [(phi, psi) for phi in np.linspace(-np.pi, np.pi, args.azimuths, endpoint=False)
                for psi in (np.radians(30), np.radians(70))]

        def setup(k, c):
            """(on, off, route, starts, tip, cover) of template k on candidate c."""
            if c == "heart":
                own = np.isin(lab, (SKELETON, UNASSIGNED)) & (r_h < temps[k]["length"])
                off_ = near & (lab >= 0)
                starts = [{"phi": float(phi), "psi": float(psi), "kappa": np.radians(10)} for phi, psi in grid]
                return T(Q[own]), T(np.concatenate([Q[off_], placed])), heart_base, starts, None, False
            Cc = plant.to_mm(sk5[c]["curve"])
            off_ = near & (lab >= 0) & (lab != c)
            if args.base == "slide":
                route, st = Cc, route_start(Cc, temps[k]["length"], a, e1, e2)
            else:
                route, st = np.zeros(3), start_from_curve(Cc, a, e1, e2)
            return T(Q[lab == c]), T(np.concatenate([Q[off_], placed])), route, [st], tip_plane(Cc, dev), True

        single, score = {}, np.full((len(u_idx), len(cands)), -1e9)
        for i, k in enumerate(u_idx):
            for j, c in enumerate(cands):
                on, off, route, starts, tip, cover = setup(k, c)
                for st in starts:
                    model = Templates([temps[k]], a, [st], dev, [route]).to(dev)
                    J, _ = fit(model, [on], [off], g, [cover], tips=[tip])
                    if J.item() > score[i, j]:
                        score[i, j], single[i, j] = J.item(), model
        cols = [j for j, c in enumerate(cands) for _ in range(len(u_idx) if c == "heart" else 1)]
        r_, c_ = linear_sum_assignment(-score[:, cols])
        choice = {i: cands[cols[cc]] for i, cc in zip(r_, c_)}
        done = [i for i in range(len(u_idx)) if i in choice]
        for i in range(len(u_idx)):
            if i not in choice:
                print(f"  flat-lay {ids[u_idx[i]]}: no P5x leaf left for it, not placed", flush=True)
        if done:
            ks = [u_idx[i] for i in done]
            setups = [setup(u_idx[i], choice[i]) for i in done]
            model = Templates([temps[k] for k in ks], a, [s[3][0] for s in setups], dev, [s[2] for s in setups]).to(dev)
            with torch.no_grad():                   # from the single fits
                for jj, i in enumerate(done):
                    src = single[i, cands.index(choice[i])]
                    for name in ("phi", "p_psi", "p_kappa", "p_split", "p_roll", "p_base"):
                        getattr(model, name)[jj] = getattr(src, name)[0]
            J2, parts = fit(model, [s[0] for s in setups], [s[1] for s in setups], g, [s[5] for s in setups],
                            tips=[s[4] for s in setups])
            record(ks, model, parts)
            for i, k in zip(done, ks):
                rows[k]["placed_on"] = "heart" if choice[i] == "heart" else f"P5x {choice[i]}"
                if choice[i] != "heart":
                    assigned[ids[k]] = choice[i]
        print(f"  stage 2: {len(u_idx)} unmatched templates on {len(cands)} candidates; placed "
              f"{ {ids[u_idx[i]]: rows[u_idx[i]]['placed_on'] for i in done} } ({time.time() - t0:.0f}s)", flush=True)
    for k, row in enumerate(rows):
        if row is None:
            continue
        print(f"  flat-lay {row['flat_leaf']} ({row['length_mm']:.0f} mm, "
              f"{'P5x ' + str(row['p5x_match']) if row['p5x_match'] is not None else 'unmatched'}): azimuth "
              f"{row['azimuth_deg']}, elevation {row['elevation_deg']}, incurve {row['incurve_deg']}, roll "
              f"{row['roll_deg']}; on its points {row['support']:.0%}, on others {row['off']:.0%}, "
              f"on fellow templates {row['apart']:.0%}", flush=True)

    crown = plant.to_plant(np.zeros(3))
    out_leaves = []
    for k, t in enumerate(temps):
        if rows[k] is None:                       # not placed
            continue
        rib, Xk = C_all[k], X_all[k]
        b = 0
        if np.linalg.norm(rib[0]) > 1e-6:          # (slide) from the crown to where the template starts:
            if ids[k] in assigned:                 # along its P5x curve, else straight
                R = plant.to_mm(sk5[assigned[ids[k]]]["curve"])
                j = int(np.argmin(np.linalg.norm(R - rib[0], axis=1)))
                lead = R[:j] if j > 0 else np.zeros((1, 3))
            else:
                lead = np.linspace(np.zeros(3), rib[0], 8)[:-1]
            lead = lead if np.linalg.norm(lead[0]) < 1e-9 else np.vstack([np.zeros(3), lead])
            if rosette:                            # no petiole: the leaf's own narrow base, down to the origin
                strip = lead[:, None] + (Xk[0] - rib[0])[None]         # the template's first row, carried down
                rib, Xk = np.vstack([lead, rib]), np.concatenate([strip, Xk], 0)
            else:                                  # an upright plant's leaf: the stretch is its petiole
                b = len(lead)
                rib = np.vstack([lead, rib])
        curve, mesh = plant.to_plant(rib), plant.to_plant(Xk.reshape(-1, 3))
        if ids[k] in assigned:
            Cp = plant.to_mm(sk5[assigned[ids[k]]]["curve"])
            T_, t_ = (x.cpu().numpy() for x in tip_plane(Cp, "cpu"))
            rows[k]["tip_to_p5x_tip_mm"] = round(float(np.linalg.norm(C_all[k][-1] - T_)), 2)
            rows[k]["past_p5x_tip_mm"] = round(float(max(0.0, ((X_all[k].reshape(-1, 3) - T_) @ t_).max())), 2)
        out_leaves.append({"id": k, "track": f"flat_{t['id']}", "flat_leaf": t["id"], "bud": False,
                           "source": f"flat-lay leaf {t['id']}", "tip": curve[-1].tolist(), "base": curve[0].tolist(),
                           "curve": curve.round(6).tolist(), "blade_start": b, "midrib": curve[b:].round(6).tolist(),
                           "petiole": curve[:b + 1].round(6).tolist() if b else [],
                           "starts_on": "origin" if rosette else "stem",
                           "midrib_mm": t["length"],
                           "blade": dict(rows[k], frame="angles about plant up, from P5x's crown"),
                           "blade_mesh": {"rows": len(Xk), "cols": ACROSS, "points": mesh.round(6).tolist()}})
    doc = {"leaves": out_leaves, "origin": crown.round(6).tolist(), "crown": crown.round(6).tolist(),
           "stem": [] if rosette else plant.p5x.get("stem", []), "axes": [] if rosette else plant.p5x.get("axes", []),
           "architecture": plant.p5x.get("architecture"),
           "built": {"by": "scripts/template_fit.py", "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "flat": str(flat_dir), "match": str(match_path)}}
    (out / "skeleton.json").write_text(json.dumps(doc))
    (out / "template_fit.json").write_text(json.dumps({"leaves": rows, "skin_gap_mm": gap,
                                                       "seconds": round(time.time() - t0, 1)}, indent=1))
    drawn = [k for k in range(len(temps)) if X_all[k] is not None]
    draw(plant, args.workdir, [X_all[k] for k in drawn], [temps[k] for k in drawn], out / "template_fit.jpg", gap)
    print(f"  wrote {out}/skeleton.json, template_fit.json, template_fit.jpg ({time.time() - t0:.0f}s)")


def draw(plant, workdir, X, temps, path, gap_mm, n=6):
    """The template leaves drawn into the views that look down into the heart most (plant up):
    photo | each leaf's surface (densely sampled) where P4g's cloud does not hide it -- no deeper than
    the cloud's depth there plus the skin gap (a leaf between P4g's two skins is not hidden by them)."""
    import pycolmap
    from pose_estimator.semantic import render_points
    sparse, _ = cloud_source.geometry(workdir, cloud_source.BASELINE)
    rec = pycolmap.Reconstruction(str(sparse))
    up = np.array([0.0, 0.0, 1.0]) @ plant.R
    Ow = plant.O @ plant.R + plant.o
    pal = leaf_palette(max(len(X), 2))
    dense = []
    for x in X:
        x = np.asarray(x).reshape(-1, ACROSS, 3)
        nr = len(x)
        i = np.linspace(0, nr - 1, 4 * nr)
        j = np.linspace(0, ACROSS - 1, 4 * ACROSS)
        xi = np.stack([np.stack([np.interp(i, np.arange(nr), x[:, c, d]) for d in range(3)], 1)
                       for c in range(ACROSS)], 1)                               # (4A, ACROSS, 3)
        xij = np.stack([np.stack([np.interp(j, np.arange(ACROSS), xi[a_, :, d]) for d in range(3)], 1)
                        for a_ in range(len(i))], 0)                             # (4A, 4C, 3)
        dense.append(plant.to_plant(xij.reshape(-1, 3)) @ plant.R + plant.o)
    reach = max(t_["length"] for t_ in temps)
    cloud = plant.W[np.linalg.norm(plant.Q, axis=1) < 1.5 * reach]
    views = []
    for im in rec.images.values():
        cam = camera_from_colmap(im, rec.cameras[im.camera_id])
        Rc, t = cam.world_to_camera[:3, :3], cam.world_to_camera[:3, 3]
        d = Ow - (-Rc.T @ t)
        views.append((float(d @ up / np.linalg.norm(d)), im, cam))
    views.sort(key=lambda v: v[0])
    skin = gap_mm / plant.mpu
    tiles = []
    for down, im, cam in views[:n]:
        photo = cv2.imread(str(workdir / "p1" / "frames" / Path(im.name).name))
        H, Wd = photo.shape[:2]
        Rc, t = cam.world_to_camera[:3, :3], cam.world_to_camera[:3, 3]
        _, index = render_points(cloud, np.zeros((len(cloud), 3), np.uint8), cam, splat_radius=2)
        zc = np.where(index >= 0, (cloud[np.maximum(index, 0)] @ Rc.T + t)[..., 2], np.inf)
        over = photo.copy()
        for k, Pw in enumerate(dense):
            Pc = Pw @ Rc.T + t
            uv = np.round((Pc[:, :2] / Pc[:, 2:3]) @ cam.K[:2, :2].T + cam.K[:2, 2]).astype(int)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < Wd) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            u_, v_, z_ = uv[ok, 0], uv[ok, 1], Pc[ok, 2]
            seen = z_ <= zc[v_, u_] + skin
            over[v_[seen], u_[seen]] = pal[k % len(pal)][::-1]
        over = cv2.dilate(over, np.ones((2, 2), np.uint8)) if False else over
        over = cv2.addWeighted(photo, 0.4, over, 0.6, 0)
        oc = Ow @ Rc.T + t
        ou = (oc[:2] / oc[2]) @ cam.K[:2, :2].T + cam.K[:2, 2]
        hp = int(0.9 * reach / plant.mpu * cam.K[0, 0] / oc[2])
        x0, y0 = max(0, int(ou[0]) - hp), max(0, int(ou[1]) - hp)
        a_, b_ = photo[y0:int(ou[1]) + hp, x0:int(ou[0]) + hp], over[y0:int(ou[1]) + hp, x0:int(ou[0]) + hp]
        tile = np.concatenate([cv2.resize(a_, (480, 480)), cv2.resize(b_, (480, 480))], 1)
        cv2.putText(tile, f"{Path(im.name).stem} ({np.degrees(np.arcsin(-down)):.0f} deg down): flat-lay leaves, "
                          f"unhidden parts", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        for k, t_ in enumerate(temps):
            c = tuple(int(v) for v in pal[k % len(pal)][::-1])
            cv2.putText(tile, f"{t_['id']}", (488 + 22 * k, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        tiles.append(tile)
    cv2.imwrite(str(path), np.concatenate([np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles) - 1, 2)], 0))


if __name__ == "__main__":
    main()
