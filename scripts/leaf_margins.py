"""Each leaf's margin, read off P4g's leaf shell along its midrib: width profile, teeth and waves.

    PYTHONPATH=src:scripts python scripts/leaf_margins.py --workdir <workdir> [--cloud p5x_p4g] [--out DIR]

What a flat Blender leaf asset needs from the real one, per leaf of <workdir>/<cloud>/ (P5x on
P4g: its points in segmented.ply, its midrib in skeleton.json), in mm (tag_scale.json):

  width     the blade's width along its midrib (20 samples, base to tip), its widest and where
  outline   the margin's in-plane wave about its own trend -- teeth, lobes: amplitude (robust
            standard deviation) and dominant wavelength, both sides
  height    the margin's out-of-plane wave -- a wavy edge: amplitude and dominant wavelength

How. P4g draws a leaf as a closed shell, top and bottom skins meeting at the margin
(docs/apex-plan.md 0.0d). Along the midrib, in steps of STEP_MM, the leaf's points nearest that
stretch are taken in the plane across the midrib: their widest spread is the across direction,
the next the normal, pointed up (the skins' gap lies along it, the width across it). The two extreme
points across are the margins -- where the skins meet -- and their median normal coordinate is
the margin's height. A wave is what is left after the margin's running median over one
half-width of the leaf: the leaf's own scale, shorter than the blade's own curve (sugarbeet_x,
2026-10-14: margins wave 0.1-1 mm at 6-10 mm). Points with fewer than BODY_N of their leaf's
points within BODY_MM are strays leaked onto other leaves and dropped first.

Writes <out>/margins.json (per leaf) and <out>/margins.jpg (per leaf: the width profile, and the
outline and height waves of both margins). <out> defaults to <workdir>/apex/margins.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from pose_estimator.ply_io import read_ply_vertices
from skeleton_2d import resample

STEP_MM = 0.5                # along the midrib: one margin sample per step
BODY_MM, BODY_N = 0.5, 8     # a stray: fewer than BODY_N of its leaf's points within BODY_MM


def body(P):
    n = np.array([len(v) for v in cKDTree(P).query_ball_point(P, BODY_MM)])
    return P[n > BODY_N]


def wave(x, y, window_mm):
    """y minus its running median over `window_mm` of x: (residual, amplitude, dominant
    wavelength). The amplitude is the residual's robust standard deviation (1.4826 x its median
    absolute value), not its RMS: where part of a margin is missing -- hidden, or claimed by a
    leaf crossing it -- the trace jumps for a step or two, and those jumps are gaps, not waves
    (sugarbeet_x, 2026-10-14: spikes at the width profile's dips). The spectrum is taken with
    them clipped to 3 of those deviations."""
    if len(y) < 8:
        return np.zeros_like(y), None, None
    step = float(np.median(np.diff(x))) or STEP_MM
    k = max(3, int(round(window_mm / step)) | 1)
    pad = np.pad(y, k // 2, mode="edge")
    r = y - np.array([np.median(pad[i:i + k]) for i in range(len(y))])
    sigma = 1.4826 * float(np.median(np.abs(r)))
    rc = np.clip(r, -3 * sigma, 3 * sigma) if sigma > 0 else r
    f = np.fft.rfftfreq(len(rc), step)
    a = np.abs(np.fft.rfft(rc - rc.mean()))
    keep = f > 1.0 / window_mm
    lam = float(1.0 / f[keep][np.argmax(a[keep])]) if keep.any() else None
    return r, sigma, lam


def read_leaf(P, midrib):
    """Along the midrib: (s mm, left u, right u, left h, right h), the leaf's length.

    A point's place along the leaf is its nearest midrib sample's arc length plus its offset
    along the midrib there (the nearest sample alone throws a wide blade's margin to the wrong
    place where the midrib curves); a stretch's across and normal directions come from the
    points of the stretch and its two neighbours either side (one step's points alone wobble)."""
    M = resample(midrib, max(16, int(np.linalg.norm(np.diff(midrib, axis=0), axis=1).sum() / STEP_MM) + 1))
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(M, axis=0), axis=1))]
    T = np.gradient(M, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    _, j = cKDTree(M).query(P)
    s_pt = cum[j] + np.einsum("ij,ij->i", P - M[j], T[j])
    inside = (s_pt > 0) & (s_pt < cum[-1])
    P, s_pt, j = P[inside], s_pt[inside], j[inside]
    edges = np.arange(0, cum[-1] + STEP_MM, STEP_MM)
    k = np.digitize(s_pt, edges) - 1
    rows, W_prev = [], None
    for i in range(len(edges) - 1):
        mine = k == i
        if mine.sum() < 12:
            continue
        c = 0.5 * (edges[i] + edges[i + 1])
        m = int(np.clip(np.searchsorted(cum, c), 0, len(M) - 1))
        t = T[m]
        wide = (k >= i - 2) & (k <= i + 2)
        qw = P[wide] - M[m]
        qw = qw - (qw @ t)[:, None] * t
        _, _, V = np.linalg.svd(qw - qw.mean(0), full_matrices=False)
        W, N = V[0], V[1]                          # across, normal (V[2] is the midrib itself)
        if W_prev is not None and W @ W_prev < 0:  # one side stays one side along the leaf
            W = -W
        if N[2] < 0:                               # height up, in the plant frame
            N = -N
        W_prev = W
        q = P[mine] - M[m]
        q = q - (q @ t)[:, None] * t
        u, h = q @ W, q @ N
        lo, hi = np.percentile(u, [1, 99])
        band = max(STEP_MM / 2, 2 * float(np.median(np.abs(np.diff(np.sort(u))))))
        hl, hr = h[np.abs(u - lo) <= band], h[np.abs(u - hi) <= band]
        rows.append((c, lo, hi, np.median(hl) if len(hl) else np.nan, np.median(hr) if len(hr) else np.nan))
    return np.asarray(rows), float(cum[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--cloud", default="p5x_p4g", help="the P5x folder whose leaves to read")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.workdir / "apex" / "margins"
    out.mkdir(parents=True, exist_ok=True)
    mpu = json.loads((args.workdir / "tag_scale.json").read_text())["mm_per_unit"]
    folder = args.workdir / args.cloud
    f = read_ply_vertices(folder / "segmented.ply")
    X = np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float64) * mpu
    lab = np.asarray(f["label"])
    sk = json.loads((folder / "skeleton.json").read_text())
    leaves, panels = [], []
    for leaf in sk.get("leaves", []):
        rib = np.asarray(leaf.get("midrib") or [], float) * mpu
        P = X[lab == int(leaf["id"])]
        if len(rib) < 4 or len(P) < 500:
            continue
        P = body(P)
        R, length = read_leaf(P, rib)
        if len(R) < 8:
            continue
        s, lo, hi, hl, hr = R.T
        width = hi - lo
        k = int(np.argmax(width))
        win = max(float(width[k]) / 2, 4 * STEP_MM)            # one half-width: the leaf's own scale
        ok_l, ok_r = np.isfinite(hl), np.isfinite(hr)
        ol, ol_rms, ol_lam = wave(s, lo, win)
        orr, or_rms, or_lam = wave(s, hi, win)
        hl_r, hl_rms, hl_lam = wave(s[ok_l], hl[ok_l], win)
        hr_r, hr_rms, hr_lam = wave(s[ok_r], hr[ok_r], win)
        mean = lambda *v: (float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None)
        prof = np.interp(np.linspace(s[0], s[-1], 20), s, width)
        row = {"leaf": int(leaf["id"]), "points": int(len(P)), "midrib_mm": round(length, 2),
               "read_mm": [round(float(s[0]), 2), round(float(s[-1]), 2)],
               "width_max_mm": round(float(width[k]), 2), "widest_at": round(float(s[k] / max(length, 1e-9)), 3),
               "width_profile_mm": [round(float(v), 2) for v in prof],
               "outline_wave": {"amplitude_mm": round(mean(ol_rms, or_rms), 3), "wavelength_mm": mean(ol_lam, or_lam),
                                "left": [ol_rms, ol_lam], "right": [or_rms, or_lam]},
               "height_wave": {"amplitude_mm": mean(hl_rms, hr_rms), "wavelength_mm": mean(hl_lam, hr_lam),
                               "left": [hl_rms, hl_lam], "right": [hr_rms, hr_lam]},
               "trend_window_mm": round(win, 2)}
        leaves.append(row)
        panels.append((row, s, width, (s, ol, orr), ((s[ok_l], hl_r), (s[ok_r], hr_r))))
        fmt = lambda v: "-" if v is None else f"{v:.2f}"
        print(f"  leaf {row['leaf']:3d}: midrib {length:5.1f} mm, widest {row['width_max_mm']:5.1f} mm at "
              f"{row['widest_at']:.2f}; outline wave {fmt(row['outline_wave']['amplitude_mm'])} mm @ "
              f"{fmt(row['outline_wave']['wavelength_mm'])} mm, edge height wave "
              f"{fmt(row['height_wave']['amplitude_mm'])} mm @ {fmt(row['height_wave']['wavelength_mm'])} mm")
    (out / "margins.json").write_text(json.dumps({
        "cloud": str(folder), "mm_per_unit": mpu, "step_mm": STEP_MM, "leaves": leaves,
        "how": "along each leaf's midrib on P4g's shell: width, and the margin's in-plane (outline) and "
               "out-of-plane (height) wave about its running median over one half-width; amplitude = robust "
               "standard deviation (gaps in a margin do not count), dominant wavelength "
               "(scripts/leaf_margins.py)"}, indent=1))
    if panels:
        H, Wd = 120, 900
        img = np.full((H * len(panels), Wd, 3), 255, np.uint8)
        for i, (row, s, width, (so, ol, orr), heights) in enumerate(panels):
            y0 = i * H
            x = lambda v: (10 + (np.asarray(v) - s[0]) / max(s[-1] - s[0], 1e-9) * (Wd - 20)).astype(np.int32)
            wpts = np.stack([x(s), y0 + H - 8 - width / max(width.max(), 1e-9) * (H // 2 - 10)], 1).astype(np.int32)
            cv2.polylines(img, [wpts], False, (150, 150, 150), 1)
            lim = max(0.25, np.nanpercentile(np.abs(np.concatenate([ol, orr] + [h for _, h in heights])), 98))
            for (sx, r), colour in (((so, ol), (0, 140, 0)), ((so, orr), (0, 200, 120)),
                                    (heights[0], (190, 80, 0)), (heights[1], (240, 140, 60))):
                if len(sx) > 1:
                    pts = np.stack([x(sx), y0 + H // 3 - r / lim * (H // 3 - 8)], 1).astype(np.int32)
                    cv2.polylines(img, [pts], False, colour, 1)
            cv2.line(img, (0, y0 + H // 3), (Wd, y0 + H // 3), (225, 225, 225), 1)
            cv2.putText(img, f"leaf {row['leaf']}: midrib {row['midrib_mm']:.0f} mm, widest {row['width_max_mm']:.0f} mm "
                             f"at {row['widest_at']:.2f}; waves +-{lim:.2f} mm: green outline, blue edge height; "
                             f"grey width profile", (6, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
        cv2.imwrite(str(out / "margins.jpg"), img)
    print(f"  wrote {out / 'margins.json'} ({len(leaves)} leaves), margins.jpg")


if __name__ == "__main__":
    main()
