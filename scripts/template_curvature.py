#!/usr/bin/env python
"""The placed flat-lay leaves (template_fit.py) given their real leaf's curvature: its cup and wavy edge.

    PYTHONPATH=src:scripts python scripts/template_curvature.py --workdir <workdir> \
        [--templates <workdir>/apex/templates] [--curvature <workdir>/apex/curvature] [--out DIR]

A template is the flat-lay leaf, laid on the plant as a flat sheet. leaf_curvature.py reads, per P5x
leaf, how the real leaf stands off its own flat self -- its cup across and its wavy edge -- on a
size-free grid (u along, r across: -1 one margin, 0 the midrib, 1 the other). A template's mesh is
already that grid (rows along it, columns r = -1..1). So each template vertex moves along its blade's
normal by:
  - its P5x leaf's own cup and wave (the leaf it was placed on), heights times the template's half-
    width over that leaf's; r mirrored when the two leaves' across directions point opposite ways;
  - a leaf placed with no P5x twin (the heart's): the plant's model -- its wave (margin spectrum,
    envelope from midrib to margin) and its cup, as blender_leaf_curvature.py builds them.
The normal is the mesh's own, turned to plant up on average (leaf_curvature.py's sign: + = the upper
face). The midrib's bend is the template's own pose; only the blade across it moves.

Writes <out>/skeleton.json: template_fit.py's, every blade_mesh bent (<out> defaults to
<workdir>/apex/templates_curved).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heart_fit import Plant                                             # noqa: E402

ACROSS = 13


def bilinear(G, U, R, u, r):
    i = np.clip(np.interp(u, U, np.arange(len(U))), 0, len(U) - 1.001)
    j = np.clip(np.interp(r, R, np.arange(len(R))), 0, len(R) - 1.001)
    i0, j0 = i.astype(int), j.astype(int)
    di, dj = i - i0, j - j0
    return (G[i0, j0] * (1 - di) * (1 - dj) + G[i0 + 1, j0] * di * (1 - dj)
            + G[i0, j0 + 1] * (1 - di) * dj + G[i0 + 1, j0 + 1] * di * dj)


def model_lift(model, u, r, hw_local, hw_max, length, rng):
    """The plant's model (as blender_leaf_curvature.Curvature.heights)."""
    env = lambda rho: np.interp(np.abs(rho), [0.0] + model["envelope_rho"] + [1.0],
                                [model["envelope_share_of_margin"][0]] + model["envelope_share_of_margin"] + [1.0])
    F = np.asarray(model["spectrum_cycles_per_half_width"]) / hw_max
    P = np.asarray(model["spectrum_power"])
    keep = (P > 0) & (F > 0)
    n = 512
    s = np.linspace(0, length, n)
    sides = []
    for _ in range(2):
        x = np.zeros(n)
        for f, p in zip(F[keep], P[keep]):
            x += math.sqrt(p) * np.cos(2 * np.pi * f * s + rng.uniform(0, 2 * np.pi))
        sides.append(x / x.std() if x.std() > 0 else x)
    k = np.clip((u * (n - 1)).round().astype(int), 0, n - 1)
    X = np.where(r < 0, sides[0][k], sides[1][k])
    a_edge = model["margin_amplitude_per_half_width"] / float(np.mean(env(np.linspace(0.5, 1.0, 11))))
    return a_edge * hw_local * env(r) * X + model["cup_rise_per_half_width"] * hw_max * r ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--templates", type=Path)
    ap.add_argument("--curvature", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tdir = args.templates or args.workdir / "apex" / "templates"
    cdir = args.curvature or args.workdir / "apex" / "curvature"
    out = args.out or args.workdir / "apex" / "templates_curved"
    out.mkdir(parents=True, exist_ok=True)
    plant = Plant(args.workdir)
    sk = json.loads((tdir / "skeleton.json").read_text())
    model = json.loads((cdir / "curvature.json").read_text())["model"]
    rng = np.random.default_rng(args.seed)
    for leaf in sk["leaves"]:
        m = leaf["blade_mesh"]
        X = plant.to_mm(np.asarray(m["points"], float)).reshape(m["rows"], m["cols"], 3)
        rows, cols = X.shape[:2]
        rib = X[:, cols // 2]
        seg = np.linalg.norm(np.diff(rib, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        u = np.broadcast_to((cum / max(cum[-1], 1e-9))[:, None], (rows, cols))
        r = np.broadcast_to(np.linspace(-1, 1, cols)[None], (rows, cols))
        hw = np.linalg.norm(X - rib[:, None], axis=2)                     # each vertex's |v|
        hw_side = np.where(r < 0, hw[:, :1], hw[:, -1:]) * np.ones((1, cols))
        hw_max = float(np.max((hw[:, 0] + hw[:, -1]) / 2))
        dr = np.gradient(X, axis=0)
        dc = np.gradient(X, axis=1)
        n = np.cross(dr, dc)
        n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-12)
        if n[..., 2].mean() < 0:                                           # + = the upper face, to plant up
            n = -n
        placed = (leaf.get("blade") or {}).get("placed_on") or (
            f"P5x {leaf['blade']['p5x_match']}" if (leaf.get("blade") or {}).get("p5x_match") is not None else None)
        src = None
        if placed and placed.startswith("P5x"):
            pid = int(placed.split()[1])
            f = cdir / f"curvature_{pid}.json"
            if f.exists():
                src = (pid, json.loads(f.read_text()))
        if src is not None:
            pid, mj = src
            W5 = np.load(cdir / f"curvature_{pid}.npz")["W"]
            Wt = X[rows // 2, -1] - X[rows // 2, 0]                        # the template's across, r -1 -> 1
            mirror = float(np.dot(Wt, W5[len(W5) // 2])) < 0
            rr = -r if mirror else r
            ratio = hw_max / max(mj["half_width_mm"], 1e-9)
            U, R = np.asarray(mj["u"]), np.asarray(mj["r"])
            lift = ratio * (bilinear(np.asarray(mj["wave_mm"]), U, R, u, rr)
                            + bilinear(np.asarray(mj["cup_mm"]), U, R, u, rr))
            how = f"P5x {pid}'s own{' (mirrored)' if mirror else ''}, x{ratio:.2f}"
        else:
            lift = model_lift(model, u, r, hw_side, hw_max, cum[-1], rng)
            how = "the plant's model"
        X2 = X + lift[..., None] * n
        m["points"] = plant.to_plant(X2.reshape(-1, 3)).round(6).tolist()
        leaf["curvature"] = {"from": how, "lift_mm": [round(float(lift.min()), 2), round(float(lift.max()), 2)],
                             "lift_sd_mm": round(float(lift.std()), 3)}
        print(f"  flat-lay {leaf.get('flat_leaf')}: {how}; lift {lift.min():+.2f}..{lift.max():+.2f} mm "
              f"(sd {lift.std():.2f})", flush=True)
    sk["built"] = dict(sk.get("built", {}), curvature=str(cdir), by_curvature="scripts/template_curvature.py")
    (out / "skeleton.json").write_text(json.dumps(sk))
    print(f"  wrote {out}/skeleton.json")


if __name__ == "__main__":
    main()
