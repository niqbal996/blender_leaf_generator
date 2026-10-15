#!/usr/bin/env python
"""Where a plant's apices are, as point sets of its cloud: the input of apex_zoom/select/fit.

    python scripts/apex_regions.py --workdir <specimen>/plant --out <workdir>/apex/regions [--auto]

Writes apex_regions.npz (name -> world points, apex_labels.py's format),
apex_regions.json (per apex: rule, centre, radius, points, P5x leaves) and
apex_regions.jpg (the regions drawn on 6 photos).

Two rules, none with a tuned number:

  labelled   the user drew apex boxes (p1/frames/frame_*.json, LabelMe): the
             cloud points inside an apex's box in every view that boxes it
             (apex_labels.carve_apices -- vogelmeere's regions). --auto skips it.
  automatic  an apex is the end of a shoot: every stem or branch end of P5x's
             tree (stem_tree.end_axes ends an axis where the cloud closes over
             the young leaves), or, on a rosette, its crown. Its region is the
             cloud points within one leaf length of the end -- the plant's own
             median traced leaf -- and nearer that end than any other, so two
             apices never share a point. An end with no P5x leaf mostly inside
             is no apex. Measured against the hand-carved regions of
             vogelmeere (2026-10-13): the four ends are the four labelled apices,
             4-11 mm from their centres; IoU 0.85 / 0.74 / 0.55 / 0.25 for
             A / B / C / D (D's branch ends 11 mm short of its cluster, beside C).
             (A first rule, the points beyond each end -- nearest the tree at
             its very end -- held 0-32 % of the hand regions: a cluster sits
             round its shoot's end, not past it.)
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p5x_views import Run                                                # noqa: E402
from pose_estimator.fine_views import source_image                      # noqa: E402


def labelled(run: Run, workdir: Path, mpu: float):
    import apex_labels
    labels = apex_labels.load(workdir / "p1" / "frames")
    if not any(l["apices"] for l in labels.values()):
        return None
    _, regions = apex_labels.carve_apices(run, labels, mpu)
    return {name: (pts, {"rule": "labelled"}) for name, pts in regions.items()} or None


def leaf_length(run: Run) -> float:
    """The median length of P5x's traced leaves, in cloud units."""
    L = [np.linalg.norm(np.diff(np.asarray(l["curve"], float), axis=0), axis=1).sum()
         for l in run.skeleton.get("leaves", []) if len(l.get("curve") or []) >= 2]
    return float(np.median(L)) if L else 0.0


def rosette_origin(run: Run):
    """Where a rosette's leaves start: the point nearest every P5x leaf's base line -- its blade's
    base end, and the direction from a sixth of the way up the blade back to it -- in the sum
    of distances (Weiszfeld's iteration, so a leaf held off its line by the pliers does not pull
    it), or None with fewer than two leaves. Measured on 06-10-2026 (2026-10-14): weed_x's 11
    base lines pass within 1.2 mm of it (median; 2.0 max), thistle_x 1.7, sugarbeet_x 3.9 --
    4-7 mm from P5x's crown, which is the foot of the stem tissue, not where the leaves meet.
    Returns (origin, each line's distance from it) in world units."""
    P, D = [], []
    for leaf in run.skeleton.get("leaves", []):
        m = np.asarray(leaf.get("midrib") or [], float)
        if len(m) < 4:
            continue
        m = run.to_world(m)
        base, up = m[0], m[min(len(m) - 1, max(2, len(m) // 6))]
        d = base - up
        if np.linalg.norm(d) > 0:
            P.append(base)
            D.append(d / np.linalg.norm(d))
    if len(P) < 2:
        return None
    P, D = np.asarray(P), np.asarray(D)
    proj = np.eye(3)[None] - D[:, :, None] * D[:, None, :]          # onto each line's normal plane
    dist = lambda x: np.linalg.norm(np.einsum("nij,nj->ni", proj, x - P), axis=1)
    x = np.linalg.solve(proj.sum(0), np.einsum("nij,nj->i", proj, P))   # least squares to start
    for _ in range(50):
        w = 1.0 / np.maximum(dist(x), 1e-9)
        x_new = np.linalg.solve((w[:, None, None] * proj).sum(0), np.einsum("n,nij,nj->i", w, proj, P))
        if np.linalg.norm(x_new - x) < 1e-12:
            break
        x = x_new
    return x, dist(x)


def automatic(run: Run, arch: str, mpu: float):
    sk = run.skeleton
    if arch == "rosette":
        found = rosette_origin(run)
        if found is not None:
            o, d = found
            print(f"  the rosette's origin: where its {len(d)} leaves' base lines meet, each within "
                  f"{np.median(d) * mpu:.1f} mm of it (median; {d.max() * mpu:.1f} max)")
            ends = [("origin", o)]
        else:
            ends = [("crown", run.to_world(np.asarray(sk["crown"], float).reshape(1, 3))[0])] if sk.get("crown") else []
    else:
        ends = [(f"{a['kind']} {a['id']}", run.to_world(a["points"])[-1]) for a in sk.get("axes") or []
                if a["kind"] in ("stem", "branch") and len(a["points"]) >= 2]
    R = leaf_length(run)
    if not ends or R <= 0:
        return {}
    E = np.stack([e for _, e in ends])
    D = np.linalg.norm(run.points[:, None] - E[None], axis=2)
    own = D.argmin(1)
    lab = run.assignment
    found = []
    for i, (what, e) in enumerate(ends):
        region = (own == i) & (D[:, i] <= R)
        leaves = [int(l) for l in np.unique(lab[region & (lab >= 0)]) if region[lab == l].mean() > 0.5]
        if leaves:
            found.append((run.points[region], {"rule": f"automatic: the {what} end, within one median leaf "
                                                       f"({R * mpu:.1f} mm)", "end_world": e.tolist(),
                                               "end": what, "p5x_leaves": leaves}))
        else:
            print(f"  the {what} end: no P5x leaf mostly within {R * mpu:.1f} mm -- not an apex")
    return {f"apex_{string.ascii_uppercase[k]}": v for k, v in enumerate(found)}


def draw(run: Run, regions: dict, path: Path, n=6):
    from apex_labels import project_world
    frames = sorted(run.image_of)
    tiles = []
    cols = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 255, 0)]
    for f in [frames[i] for i in np.linspace(0, len(frames) - 1, n).round().astype(int)]:
        img = cv2.imread(str(source_image(run.workdir, f)))
        for k, (name, (pts, _)) in enumerate(sorted(regions.items())):
            uv = project_world(run, f, pts[::max(1, len(pts) // 3000)])
            for u, v in uv.astype(int):
                if 0 <= v < img.shape[0] and 0 <= u < img.shape[1]:
                    cv2.circle(img, (int(u), int(v)), 6, cols[k % len(cols)], -1)
            c = project_world(run, f, pts.mean(0, keepdims=True))[0].astype(int)
            cv2.putText(img, name, (int(c[0]) + 20, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 3, cols[k % len(cols)], 6)
        tiles.append(cv2.resize(img, (900, 600)))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--auto", action="store_true", help="ignore hand-drawn apex boxes")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mpu = json.loads((args.workdir / "tag_scale.json").read_text())["mm_per_unit"]
    run = Run(args.workdir)
    arch = run.skeleton.get("architecture") or "caulescent"
    regions = None if args.auto else labelled(run, args.workdir, mpu)
    if regions is None:
        regions = automatic(run, arch, mpu)
    if not regions:
        print(f"  no apex found ({arch}): nothing for the apex fit to do")
    report = {}
    for name, (pts, info) in sorted(regions.items()):
        c = pts.mean(0)
        r = float(np.linalg.norm(pts - c, axis=1).max())
        report[name] = dict(info, points=int(len(pts)), centre_world=c.tolist(), radius_mm=round(r * mpu, 2))
        print(f"  {name}: {info['rule']}; {len(pts)} points, radius {r * mpu:.1f} mm")
    np.savez_compressed(args.out / "apex_regions.npz", **{k: v[0] for k, v in regions.items()})
    (args.out / "apex_regions.json").write_text(json.dumps({"architecture": arch, "apices": report}, indent=1))
    if regions:
        draw(run, regions, args.out / "apex_regions.jpg")
    print(f"  wrote {args.out / 'apex_regions.npz'} ({len(regions)} apices)")


if __name__ == "__main__":
    main()
