"""A leaf's margin line, read off P4g's two-skinned leaf shell, and how wavy it is.

    PYTHONPATH=src python scripts/leaf_margin.py <workdir> <leaf label> [<leaf label> ...] [--cloud p5x_p4g]

P4g draws a leaf as a closed shell (top and bottom skins 1-4 mm apart, meeting at the margin):
every slab across the leaf is a loop whose two ends are the margins. Per 0.5 mm slab along the
leaf's long axis, each margin is the loop's extreme across (robust: 0.5th/99.5th percentile) and
its height the median normal coordinate of the points within 0.3 mm of it. Along the leaf, the
margin's height minus its own 10 mm running median is the out-of-plane wave, its across position
minus the same the in-plane one (teeth, lobes). Prints RMS and the dominant wavelength (FFT peak)
of each; writes <workdir>/debug/leaf_relief/margin_<leaf>.png (both traces, both margins) and
sections_<leaf>.png (four 0.5 mm slabs across the leaf: the shell's loops).

A first look (2026-10-14): fine on sugarbeet_x's smooth-edged blades; on thistle's lobed ones
the extreme across jumps from lobe to lobe -- a margin read along the midrib is needed there.
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from pose_estimator.ply_io import read_ply_vertices

SLAB, NEAR, TREND = 0.5, 0.3, 10.0      # mm: slab step, margin neighbourhood, running-median window
BODY = 0.5                              # mm: a point with no 8 leaf neighbours this close is a stray


def body(P):
    """The leaf without its strays: points leaked onto other leaves sit alone, a skin never does."""
    from scipy.spatial import cKDTree
    n = np.array([len(v) for v in cKDTree(P).query_ball_point(P, BODY)])
    return P[n > 8]


def margins(P):
    c = P.mean(0); _, _, V = np.linalg.svd(P - c, full_matrices=False); q = (P - c) @ V.T
    xs = np.arange(*np.percentile(q[:, 0], [2, 98]), SLAB)
    out = []
    for x in xs:
        s = q[np.abs(q[:, 0] - x) < SLAB / 2]
        if len(s) < 30:
            continue
        row = [x]
        for side in (0.5, 99.5):
            u = np.percentile(s[:, 1], side); near = s[np.abs(s[:, 1] - u) < NEAR, 2]
            row += [u, np.median(near) if len(near) else np.nan]
        if np.isfinite(row).all():
            out.append(row)
    return np.asarray(out)          # x, u_left, w_left, u_right, w_right


def sections(P, leaf, path):
    """Four slabs across the leaf, each drawn in its own (across, normal) plane."""
    c = P.mean(0); _, _, V = np.linalg.svd(P - c, full_matrices=False); q = (P - c) @ V.T
    tiles = []
    for x0 in np.linspace(*np.percentile(q[:, 0], [3, 97]), 6)[1:-1]:
        sl = q[np.abs(q[:, 0] - x0) < SLAB / 2]; img = np.full((260, 520, 3), 255, np.uint8)
        if len(sl):
            u0, u1, w0, w1 = sl[:, 1].min(), sl[:, 1].max(), sl[:, 2].min(), sl[:, 2].max()
            sc = min(500 / max(u1 - u0, 1e-6), 230 / max(w1 - w0, 1e-6), 40)
            for u, w in sl[:, 1:]:
                cv2.circle(img, (int(10 + (u - u0) * sc), int(250 - (w - w0) * sc)), 1, (40, 40, 40), -1)
            cv2.putText(img, f"leaf {leaf}: slab at {x0:+.0f} mm along; across {u1-u0:.1f} mm, normal span {w1-w0:.1f} mm, "
                             f"1 px = {1/sc:.2f} mm", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1)
        tiles.append(img)
    cv2.imwrite(str(path), np.vstack(tiles))


def detrend(x, y):
    k = max(3, int(round(TREND / SLAB)) | 1)
    pad = np.pad(y, k // 2, mode="edge")
    return y - np.array([np.median(pad[i:i + k]) for i in range(len(y))])


def wave(x, r):
    f = np.fft.rfftfreq(len(r), SLAB); a = np.abs(np.fft.rfft(r - r.mean()))
    keep = f > 1 / TREND                       # shorter than the trend window
    return float(np.sqrt(np.mean(r ** 2))), (float(1 / f[keep][np.argmax(a[keep])]) if keep.any() else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("workdir", type=Path); ap.add_argument("leaves", type=int, nargs="+")
    ap.add_argument("--cloud", default="p5x_p4g"); a = ap.parse_args()
    mpu = json.loads((a.workdir / "tag_scale.json").read_text())["mm_per_unit"]
    f = read_ply_vertices(a.workdir / a.cloud / "segmented.ply")
    X = np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float64) * mpu; lab = np.asarray(f["label"])
    out = a.workdir / "debug" / "leaf_relief"; out.mkdir(parents=True, exist_ok=True)
    for leaf in a.leaves:
        P = X[lab == leaf]; B = body(P); M = margins(B); x = M[:, 0]
        traces = {"left height": detrend(x, M[:, 2]), "right height": detrend(x, M[:, 4]),
                  "left outline": detrend(x, M[:, 1]), "right outline": detrend(x, M[:, 3])}
        print(f"{a.workdir.parent.name} leaf {leaf}: {len(P) - len(B)} strays of {len(P)} points dropped; {len(x)} slabs over {x[-1]-x[0]:.0f} mm")
        for k, r in traces.items():
            rms, lam = wave(x, r)
            print(f"  {k:13s}: RMS {rms:.2f} mm, dominant wavelength {lam:.1f} mm" if lam else f"  {k}: -")
        H, Wd = 180, 900; img = np.full((H * 4, Wd, 3), 255, np.uint8)
        lim = max(np.percentile(np.abs(np.concatenate(list(traces.values()))), 98), 0.2)
        for i, (k, r) in enumerate(traces.items()):
            y0 = H * i + H // 2
            cv2.line(img, (0, y0), (Wd, y0), (200, 200, 200), 1)
            pts = np.stack([(x - x[0]) / (x[-1] - x[0]) * (Wd - 20) + 10, y0 - r / lim * (H // 2 - 15)], 1).astype(np.int32)
            cv2.polylines(img, [pts], False, (180, 60, 0) if "height" in k else (0, 120, 0), 1)
            cv2.putText(img, f"leaf {leaf} {k} (running {TREND:g} mm median removed): +-{lim:.2f} mm full scale, {x[-1]-x[0]:.0f} mm long",
                        (8, H * i + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.imwrite(str(out / f"margin_{leaf}.png"), img)
        sections(B, leaf, out / f"sections_{leaf}.png")
    print(f"  wrote {out}/margin_<leaf>.png, sections_<leaf>.png")


if __name__ == "__main__":
    main()
