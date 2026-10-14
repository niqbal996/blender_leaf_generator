#!/usr/bin/env python
"""How good are a skeleton's fitted apex leaves: count, shape, per-leaf support, instance scores, twins, tips.

    python scripts/apex_fit_check.py --workdir <specimen>/plant --skeleton <plant>/debug/apex_fit_v3 \
        [--out <dir>]          # default: the skeleton's folder

Diagnostic for scripts/apex_fit.py output (leaves carrying `apex` and
`blade_mesh`); docs/apex-plan.md section 0. Any version's skeleton is scored
the same way -- its blades drawn with apex_fit's renderer, beside the plant's
other leaves as instances (apex_fit.context), against its zoomed SAM3 masks
-- so two versions compare. Writes into --out:

  leaf_support.txt/.json  per apex:
                          - leaves against the user's count
                            (p1/frames/apex_counts.json; the bud not counted),
                          - shape about the apex's shoot axis (its fitted one,
                            `apices`, else the axis its leaves radiate round,
                            apex_fit.apex_frame): per leaf the midrib's turn
                            from base to tip, how much of it is towards the axis
                            (+, incurved) or away (-), and out of the leaf's
                            median plane (sideways: the spiral look); roll,
                            fold;
                          - the instance scores apex_fit optimises (one-to-one
                            matching, TP above IoU 0.5; PQ, F1, SQ, TP / FP /
                            FN, TP by context leaves), on all frames;
                          - per leaf: views it shows in (>= MIN_MASK px), and
                            of those, alone on a SAM3 mask, sharing one with
                            another leaf, or on no mask. Split blades share;
                            ghost leaves sit on no mask;
                          - twins: pairs of leaves whose blade midribs lie within
                            half a blade width of each other.
                          Tips against the hand labels (p1/frames/*.json,
                          apex_labels.py's pairing): error along the labelled
                          line (+ = beyond your tip) and across it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apex_fit as af                                                   # noqa: E402
import apex_labels as al                                                # noqa: E402
from p5x_views import Run                                               # noqa: E402
from skeleton_2d import resample                                        # noqa: E402


def blade_cells(run, leaves, dev):
    """Each leaf's `blade_mesh` as Gaussians at its cell centres: (n, rows-1, cols-1, 3), covariances."""
    G = [run.to_world(np.asarray(l["blade_mesh"]["points"])).reshape(
        l["blade_mesh"]["rows"], l["blade_mesh"]["cols"], 3) for l in leaves]
    X = torch.tensor(np.stack([(g[1:, 1:] + g[:-1, 1:] + g[1:, :-1] + g[:-1, :-1]) / 4 for g in G]),
                     dtype=torch.float32, device=dev)
    a, b = torch.gradient(X, dim=1)[0], torch.gradient(X, dim=2)[0]
    cov = af.KAPPA ** 2 * (a[..., :, None] * a[..., None, :] + b[..., :, None] * b[..., None, :]) \
        + 1e-10 * torch.eye(3, device=dev)
    return X, cov


def support(Pm, views, leaves):
    """Per leaf: visible views, alone on a mask, sharing one, on none."""
    O = torch.einsum("vxn,vkx->vnk", Pm, views.T)
    vis = Pm.sum(1)
    best, k = O.max(-1)
    visible = vis >= af.MIN_MASK
    on_mask = visible & (best > 0.5 * vis)                 # most of its visible pixels on one mask
    rows = []
    for i, l in enumerate(leaves):
        share = alone = 0
        for v in torch.nonzero(on_mask[:, i]).flatten().tolist():
            other = any(j != i and on_mask[v, j] and k[v, j] == k[v, i] for j in range(len(leaves)))
            share += other
            alone += not other
        nv = int(visible[:, i].sum())
        rows.append({"id": l["id"], "track": l["track"], "found": bool(l.get("gt_found")), "visible_views": nv,
                     "alone_on_a_mask": alone, "sharing_a_mask": share, "on_no_mask": nv - alone - share})
    return rows


def twins(leaves, mpu):
    B = [resample(np.asarray(l["midrib"]), 20) for l in leaves]
    out = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            d = float(np.mean(np.linalg.norm(B[i] - B[j], axis=1)) * mpu)
            if d < min(leaves[i]["blade"]["width_mm"], leaves[j]["blade"]["width_mm"]) / 2:
                out.append([leaves[i]["track"], leaves[j]["track"], round(d, 2)])
    return out


def bend(leaves, axis, mpu):
    """Per leaf, about the shoot axis (plant frame): the midrib's turn from base to tip (deg), and
    the change of its unit tangent split into towards the axis (+) / away (-) in its median plane,
    and out of that plane (sideways, signed by the sense of rotation about the axis)."""
    u = lambda v: v / max(np.linalg.norm(v), 1e-12)
    out = []
    for l in leaves:
        c = np.asarray(l["curve"], float)
        b = int(l.get("blade_start") or 0)
        rib = resample(c[b:] if len(c) - b >= 4 else c, 24)
        chord = rib[-1] - rib[0]
        r = u(chord - (chord @ axis) * axis)
        t0, t1 = u(rib[2] - rib[0]), u(rib[-1] - rib[-3])
        n_in = u(axis - (axis @ u(chord)) * u(chord))
        dt = t1 - t0
        bl = l.get("blade") or {}
        out.append({"track": l["track"], "length_mm": float(np.linalg.norm(np.diff(rib, axis=0), axis=1).sum() * mpu),
                    "turn_deg": float(np.degrees(np.arccos(np.clip(t0 @ t1, -1, 1)))),
                    "inward": float(dt @ n_in), "sideways": float(dt @ np.cross(axis, r)),
                    "roll_deg": bl.get("roll_deg"), "roll_tip_deg": bl.get("roll_tip_deg"),
                    "twist_deg": bl.get("twist_deg"), "fold_deg": bl.get("fold_deg")})
    return out


def tips(run, labels, skeleton):
    """Tip error of label-paired leaves, along the labelled line and across it (px)."""
    _, pairs = al.score(run, labels, skeleton)
    along, across = [], []
    for (gts, near, ok) in pairs.values():
        for r, c in ok:
            g, s = gts[r], near[c]
            d = g[-1] - resample(g, 10)[-3]
            d /= np.linalg.norm(d)
            off = s[3] - g[-1]
            along.append(float(off @ d))
            across.append(float(abs(off @ np.array([-d[1], d[0]]))))
    return {"pairs": len(along), "along_median_px": float(np.median(along)) if along else None,
            "short_share": float(np.mean(np.array(along) < 0)) if along else None,
            "across_median_px": float(np.median(across)) if across else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--skeleton", type=Path, required=True, help="apex_fit.py output dir")
    ap.add_argument("--regions", type=Path, help="default: <workdir>/debug/apex_labels/apex_regions.npz")
    ap.add_argument("--zoom", type=Path, help="default: <workdir>/debug/apex_zoom_fair")
    ap.add_argument("--out", type=Path, help="default: --skeleton")
    args = ap.parse_args()
    P = args.workdir
    out = args.out or args.skeleton
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    run = Run(P)
    mpu = json.loads((P / "tag_scale.json").read_text())["mm_per_unit"]
    regions = dict(np.load(args.regions or P / "debug" / "apex_labels" / "apex_regions.npz"))
    zoom = args.zoom or P / "debug" / "apex_zoom_fair"
    sk = json.loads((args.skeleton / "skeleton.json").read_text())
    counted = P / "p1" / "frames" / "apex_counts.json"
    truth = json.loads(counted.read_text())["leaves"] if counted.exists() else {}
    lines = af.trunk_lines(run, run.skeleton)                              # P5x's own tree
    report, text = {"skeleton": str(args.skeleton), "built": sk.get("built"), "apices": {}}, []
    say = lambda s: (print(s, flush=True), text.append(s))
    total = {"visible_views": 0, "alone_on_a_mask": 0, "sharing_a_mask": 0, "on_no_mask": 0}
    shape_all = []
    for name in sorted(regions):
        leaves = [l for l in sk["leaves"] if l.get("apex") == name and l.get("blade_mesh")]
        if not leaves:
            continue
        pts = regions[name]
        c = pts.mean(axis=0)
        reach = float(np.linalg.norm(pts - c, axis=1).max())
        views = af.Views(run, zoom, name.replace("apex_", ""), None, dev)
        ctx = af.context(run, views, af.rows_in(run.points, pts), dev)
        with torch.no_grad():
            X, cov = blade_cells(run, leaves, dev)
            Pa, Pc = af.render_with(X, cov, len(leaves), views, ctx)
            bud = torch.tensor([bool(l.get("bud")) for l in leaves], device=dev)
            sc, _ = af.scores(Pa, Pc, views, X, bud)                    # the bud: never a false positive
            rows = support(Pa, views, leaves)
        n_leaves = sum(not l.get("bud") for l in leaves)
        if (sk.get("apices") or {}).get(name):
            axis, which = np.asarray(sk["apices"][name]["axis"], float), "its fitted axis"
        else:
            node = run.to_world(np.median(np.stack([np.asarray(l["curve"][0]) for l in leaves]), axis=0))
            fr = af.apex_frame(node, [run.to_world(np.asarray(l["curve"][-1])) for l in leaves if not l.get("bud")],
                               lines, reach, 1 / mpu, run.rotation[2])
            axis, which = fr["axis"] @ run.rotation.T, "the axis its leaves radiate round"
        bent = iter(bend([l for l in leaves if not l.get("bud")], axis / np.linalg.norm(axis), mpu))
        per = [None if l.get("bud") else next(bent) for l in leaves]          # aligned with `leaves`
        sh = [s for s in per if s is not None]
        shape_all += sh
        tw = twins(leaves, mpu)
        gt = truth.get(name)
        say(f"{name}: {n_leaves} leaves{f' (you count {gt})' if gt else ''}"
            f"{' + bud' if any(l.get('bud') for l in leaves) else ''}  PQ {sc['pq']:.3f}  F1 {sc['f1']:.3f}  "
            f"SQ {sc['sq']:.3f}  TP {sc['tp']} (context leaves {sc['tp_context']}) FP {sc['fp']} FN {sc['fn']}  "
            f"({len(views.frames)} views)  twins {len(tw)}")
        say(f"  shape about {which}: turn median {np.median([s['turn_deg'] for s in sh]):.0f} deg "
            f"(max {max(s['turn_deg'] for s in sh):.0f}), incurved {sum(s['inward'] > 0 for s in sh)} of {len(sh)}, "
            f"sideways median {np.median([abs(s['sideways']) for s in sh]):.2f}")
        for r, s in zip(rows, per):
            shape = "" if s is None else (
                f"  turn {s['turn_deg']:4.0f}  in {s['inward']:+.2f}  side {s['sideways']:+.2f}"
                + (f"  roll {s['roll_deg']:6.1f}" if s["roll_deg"] is not None else "")
                + (f" -> {s['roll_tip_deg']:6.1f}" if s.get("roll_tip_deg") is not None else "")
                + (f"  twist {s['twist_deg']:6.1f}" if s.get("twist_deg") is not None else "")
                + (f"  fold {s['fold_deg']:4.1f}" if s["fold_deg"] is not None else ""))
            say(f"  {r['track']:30s} {'GT' if r['found'] else '  '} visible {r['visible_views']:2d}  "
                f"alone {r['alone_on_a_mask']:2d}  shared {r['sharing_a_mask']:2d}  "
                f"on no mask {r['on_no_mask']:2d}{shape}")
            for k in total:
                total[k] += r[k]
        for t in tw:
            say(f"  twin: {t[0]} ~ {t[1]}  {t[2]} mm apart")
        report["apices"][name] = dict(sc, leaves=n_leaves, user_count=gt, twins=tw, per_leaf=rows, shape=sh,
                                      shape_axis=which)
        del views, ctx
        torch.cuda.empty_cache()
    v = max(1, total["visible_views"])
    report["support_share"] = {k: total[k] / v for k in total if k != "visible_views"}
    report["tips_vs_labels"] = tips(run, al.load(P / "p1" / "frames"), sk)
    if truth:
        diff = {n: report["apices"][n]["leaves"] - truth[n] for n in truth if n in report["apices"]}
        report["count_error"] = diff
        say("leaves - your count: " + ", ".join(f"{n[-1]} {d:+d}" for n, d in diff.items())
            + f"  (sum of |errors| {sum(abs(d) for d in diff.values())})")
    say(f"all: turn median {np.median([s['turn_deg'] for s in shape_all]):.0f} deg, incurved "
        f"{sum(s['inward'] > 0 for s in shape_all)} of {len(shape_all)}, sideways median "
        f"{np.median([abs(s['sideways']) for s in shape_all]):.2f}; of the views a leaf shows in, alone on a mask "
        f"{report['support_share']['alone_on_a_mask']:.0%}, sharing {report['support_share']['sharing_a_mask']:.0%}, "
        f"on none {report['support_share']['on_no_mask']:.0%}")
    t = report["tips_vs_labels"]
    if t["pairs"]:
        say(f"tips vs your lines ({t['pairs']} pairs): along {t['along_median_px']:+.1f} px "
            f"(short {t['short_share']:.0%}), across {t['across_median_px']:.1f} px")
    (out / "leaf_support.json").write_text(json.dumps(report, indent=1))
    (out / "leaf_support.txt").write_text("\n".join(text) + "\n")
    print(f"wrote {out / 'leaf_support.txt'}")


if __name__ == "__main__":
    main()
