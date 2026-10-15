#!/usr/bin/env python
"""The plant's own leaves, laid flat after the capture (leaf_pose), matched to its 3D leaves flattened.

    PYTHONPATH=src:scripts python scripts/flatlay_match.py --workdir <specimen>/plant \
        [--flat <specimen>/leaf_pose] [--curvature <workdir>/apex/curvature] [--out <workdir>/apex/flatlay]

After the turntable capture the plant was taken apart and its leaves shot flat from above
(src/leaf_pose: per leaf its contour, midrib from petiole origin to tip, mm per pixel). Those are
the true outlines, in mm. leaf_curvature.py flattens each P5x-on-P4g leaf into the same kind of
outline (along its midline, across it, in mm). Matching the two:

  - which real leaf each 3D leaf is -- and which real leaves no 3D leaf is (the heart's young
    leaves, the ones the 3D pipeline misses: candidates for the heart fit's templates);
  - how well the flattening reproduces the real outline (IoU of matched outlines, lengths, widths).

Both outlines are rasterised on a 0.5 mm grid (along x across), the flat leaf's base at its
petiole origin; the 3D leaf is slid along the flat one's midrib (its midline starts wherever P5x's
midrib did) and mirrored (a leaf lies face up or down), and the best IoU kept. Leaves are paired
one to one by the Hungarian method on IoU, a pair kept above IoU 1/2.

Writes <out>/flatlay_match.json and flatlay_match.jpg; <out> defaults to <workdir>/apex/flatlay.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from skeleton_2d import resample

STEP = 0.5   # mm, the leaf_curvature grid


def flat_leaf(l):
    """A flat-lay leaf in its own frame: (s, v) of its contour, in mm, s from its petiole origin
    along its midrib, v across (+ = left of the midrib walking base to tip); its midrib length."""
    mpp = l["mm_per_pixel"]
    M = resample(np.asarray(l["midrib"], float), 200) * mpp
    C = np.asarray(l["contour"], float).reshape(-1, 2) * mpp
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(M, axis=0), axis=1))]
    T = np.gradient(M, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    d = np.linalg.norm(C[:, None] - M[None], axis=2)
    j = d.argmin(1)
    s = cum[j] + np.einsum("ij,ij->i", C - M[j], T[j])
    v = (C - M[j])[:, 0] * -T[j, 1] + (C - M[j])[:, 1] * T[j, 0]
    return s, v, float(cum[-1])


def raster_flat(s, v, L, vmax):
    """The contour's inside on the (s, v) grid: per row of s, from its lowest to its highest v."""
    ns, nv = int(np.ceil((L + 2) / STEP)), int(np.ceil(2 * vmax / STEP)) + 1
    img = np.zeros((ns, nv), np.uint8)
    pts = np.stack([(v + vmax) / STEP, (s + 1) / STEP], 1).round().astype(np.int32)
    cv2.fillPoly(img, [pts], 1)
    return img


def raster_curvature(z, vmax):
    """leaf_curvature's flattened outline (edges per s row) on the same kind of grid."""
    e = z["edges"]
    ns, nv = len(e), int(np.ceil(2 * vmax / STEP)) + 1
    img = np.zeros((ns, nv), np.uint8)
    V = np.arange(nv) * STEP - vmax
    for r in range(ns):
        if np.isfinite(e[r]).all():
            img[r] = (V >= e[r, 0]) & (V <= e[r, 1])
    return img


def best_iou(A, B):
    """Max IoU of B over A, B slid along s (rows) and mirrored across; (iou, shift rows, mirrored)."""
    best = (0.0, 0, False)
    for mir in (False, True):
        Bm = B[:, ::-1] if mir else B
        for sh in range(-len(B) + 1, len(A)):
            a0, a1 = max(0, sh), min(len(A), sh + len(Bm))
            if a1 <= a0:
                continue
            inter = np.logical_and(A[a0:a1], Bm[a0 - sh:a1 - sh]).sum()
            union = A.sum() + Bm.sum() - inter
            iou = inter / union if union else 0.0
            if iou > best[0]:
                best = (float(iou), sh, mir)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--flat", type=Path)
    ap.add_argument("--curvature", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    flat_dir = args.flat or args.workdir.parent / "leaf_pose"
    curv_dir = args.curvature or args.workdir / "apex" / "curvature"
    out = args.out or args.workdir / "apex" / "flatlay"
    out.mkdir(parents=True, exist_ok=True)
    flat = json.loads((flat_dir / "leaves.json").read_text())["leaves"]
    rel = json.loads((curv_dir / "curvature.json").read_text())["leaves"]
    F = [(l["id"], *flat_leaf(l), l) for l in flat]
    R = [(r["leaf"], np.load(curv_dir / f"curvature_{r['leaf']}.npz"), r) for r in rel]
    vmax = max([np.abs(v).max() for _, s, v, L, _ in F] +
               [np.nanmax(np.abs(z["edges"])) for _, z, _ in R if np.isfinite(z["edges"]).any()]) + 1
    FA = [raster_flat(s, v, L, vmax) for _, s, v, L, _ in F]
    RA = [raster_curvature(z, vmax) for _, z, _ in R]
    I = np.zeros((len(F), len(R)))
    how = {}
    for i in range(len(F)):
        for k in range(len(R)):
            iou, sh, mir = best_iou(FA[i], RA[k])
            I[i, k] = iou
            how[i, k] = (sh, mir)
    rows, cols = linear_sum_assignment(-I)
    # a pair is one leaf only above IoU 1/2 (panoptic quality's rule, apex_fit.TP_IOU): below it the
    # outlines differ more than they agree (thistle_x: heart leaf 1, 12 mm, went to a 22 mm cotyledon
    # at 0.39 -- the photos show it is not)
    keep = I[rows, cols] > 0.5
    rows, cols = rows[keep], cols[keep]
    pairs = []
    for i, k in zip(rows, cols):
        fid, s, v, L, l = F[i]
        rid, z, r = R[k]
        pairs.append({"flat": fid, "p5x": rid, "iou": round(float(I[i, k]), 3),
                      "flat_length_mm": round(L, 1), "p5x_length_mm": r["midrib_mm"],
                      "flat_width_mm": l["blade_width_max"], "p5x_flat_width_mm": round(2 * r["half_width_mm"], 1),
                      "shift_mm": how[i, k][0] * STEP, "mirrored": bool(how[i, k][1])})
    matched_flat = {p["flat"] for p in pairs}
    print(f"{args.workdir.parent.name}: {len(F)} flat-lay leaves, {len(R)} P5x leaves")
    print("  IoU (rows flat-lay, columns P5x " + " ".join(f"{rid:>5}" for rid, _, _ in R) + ")")
    for i, (fid, s, v, L, l) in enumerate(F):
        print(f"  flat {fid:2d} ({L:5.1f} mm, {l['blade_width_max']:4.1f} wide): " + " ".join(f"{x:5.2f}" for x in I[i]))
    for p in sorted(pairs, key=lambda p: p["flat"]):
        print(f"  flat {p['flat']} <-> P5x {p['p5x']}: IoU {p['iou']:.2f}; length {p['flat_length_mm']} vs "
              f"{p['p5x_length_mm']} mm, width {p['flat_width_mm']} vs {p['p5x_flat_width_mm']} mm"
              f"{', mirrored' if p['mirrored'] else ''}")
    unmatched = [fid for fid, *_ in F if fid not in matched_flat]
    print(f"  flat-lay leaves with no P5x leaf: {unmatched}")
    (out / "flatlay_match.json").write_text(json.dumps({"flat": str(flat_dir), "curvature": str(curv_dir),
                                                        "iou": I.round(3).tolist(),
                                                        "flat_ids": [f[0] for f in F], "p5x_ids": [r[0] for r in R],
                                                        "pairs": pairs, "unmatched_flat": unmatched}, indent=1))
    # picture: each pair overlaid (flat green, P5x magenta), unmatched flat leaves alone
    tiles = []
    for i, (fid, *_rest) in enumerate(F):
        A = FA[i]
        k = next((kk for ii, kk in zip(rows, cols) if ii == i), None)
        img = np.full(A.shape + (3,), 255, np.uint8)
        img[A > 0] = (120, 220, 120)
        title = f"flat {fid}"
        if k is not None:
            sh, mir = how[i, k]
            B = RA[k][:, ::-1] if mir else RA[k]
            for r in range(len(B)):
                rr = r + sh
                if 0 <= rr < len(A):
                    both = B[r] > 0
                    img[rr, both & (A[rr] > 0)] = (90, 90, 90)
                    img[rr, both & (A[rr] == 0)] = (200, 80, 200)
            title += f" / P5x {R[k][0]}: IoU {I[i, k]:.2f}"
        nz = np.flatnonzero(A.any(1) | (img[:, :, 1] != 255).any(1))
        img = img[max(0, nz.min() - 2):nz.max() + 3] if len(nz) else img
        img = cv2.resize(np.rot90(img, 1), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        img = cv2.copyMakeBorder(img, 18, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(img, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        tiles.append(img)
    Wd = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, Wd - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    cv2.imwrite(str(out / "flatlay_match.jpg"), np.concatenate(tiles, 0))
    print(f"  wrote {out}/flatlay_match.json, flatlay_match.jpg (green flat-lay, magenta P5x outside it, grey both)")


if __name__ == "__main__":
    main()
