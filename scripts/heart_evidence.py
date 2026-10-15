#!/usr/bin/env python
"""What the P4g cloud holds at a rosette's heart, and who P5x gave it to (diagnostic).

    PYTHONPATH=src:scripts python scripts/heart_evidence.py --workdir <rosette>/plant [--out DIR]

The heart: the young leaves standing at a rosette's origin (apex/fit/skeleton.json 'origin',
where the P5x leaves' base lines meet). SAM3 rarely masks them, so P5x labels few of their
points; they stay UNSEEN (never under a leaf, stem or root pixel) or go to an outer leaf whose
base they touch. This counts, around the origin, P4g's points (p4g/surface_p5x.ply) by P5x's
label (p5x_p4g/instances.npy), in shells of distance from the origin and by height above it,
and draws them: top and two side views in the plant frame (z up), and the steepest-down photos
with the points drawn over them (colour = P5x leaf, white = stem, magenta = unseen).

Writes <out>/heart_evidence.json and heart_<view>.jpg; <out> defaults to <workdir>/debug/heart_evidence.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pose_estimator import cloud_source
from pose_estimator.cli.leaf_instances import ROOT, SKELETON, UNSEEN, leaf_palette
from pose_estimator.ply_io import read_ply_vertices
from pose_estimator.semantic import camera_from_colmap

UNSEEN_RGB = (255, 0, 255)
STEM_RGB = (255, 255, 255)


def load(workdir: Path):
    f = read_ply_vertices(workdir / "p4g" / "surface_p5x.ply")
    W = np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float64)
    lab = np.load(workdir / "p5x_p4g" / "instances.npy").astype(int)
    assert len(lab) == len(W), (len(lab), len(W))
    g = json.loads((workdir / "p5" / "stem_graph.json").read_text())["plant_frame"]
    o, R = np.asarray(g["origin"], float), np.asarray(g["rotation"], float)
    P = (W - o) @ np.linalg.inv(R)                       # to_world(p) = p @ R + o
    mpu = json.loads((workdir / "tag_scale.json").read_text())["mm_per_unit"]
    sk = json.loads((workdir / "apex" / "fit" / "skeleton.json").read_text())
    O = np.asarray(sk["origin"], float)
    n2d = json.loads((workdir / "p5x_p4g" / "instances.json").read_text())["num_leaves_2d"]
    return W, P, lab, mpu, O, (o, R), sk, leaf_palette(max(n2d, 1))


def colours(lab, pal):
    c = np.zeros((len(lab), 3), np.uint8)
    leaf = lab >= 0
    c[leaf] = pal[lab[leaf] % len(pal)]
    c[lab == SKELETON] = STEM_RGB
    c[lab == ROOT] = (240, 140, 40)
    c[lab == UNSEEN] = UNSEEN_RGB
    return c


def ortho(Q, c, axes, half, px=420):
    """Q in mm about the origin; axes picks the two plotted coordinates; nearer drawn last."""
    img = np.zeros((px, px, 3), np.uint8)
    depth_ax = ({0, 1, 2} - set(axes)).pop()
    order = np.argsort(Q[:, depth_ax] * (1 if axes == (0, 1) else -1))
    s = px / (2 * half)
    x = ((Q[order, axes[0]] + half) * s).astype(int)
    y = (px - 1 - (Q[order, axes[1]] + half) * s).astype(int)
    ok = (x >= 0) & (x < px) & (y >= 0) & (y < px)
    img[y[ok], x[ok]] = c[order][ok][:, ::-1]
    cv2.circle(img, (px // 2, px // 2), 4, (0, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--radius-mm", type=float, help="half-size of the views (default: the shortest P5x leaf)")
    ap.add_argument("--photos", type=int, default=4)
    args = ap.parse_args()
    out = args.out or args.workdir / "debug" / "heart_evidence"
    out.mkdir(parents=True, exist_ok=True)
    W, P, lab, mpu, O, (o, R), sk, pal = load(args.workdir)
    Q = (P - O) * mpu                                     # mm, plant frame, origin at 0
    rib_len = [np.linalg.norm(np.diff(np.asarray(l["midrib"], float), axis=0), axis=1).sum() * mpu
               for l in json.loads((args.workdir / "p5x_p4g" / "skeleton.json").read_text())["leaves"]
               if len(l.get("midrib") or []) > 3]
    half = args.radius_mm or float(min(rib_len))
    r = np.linalg.norm(Q, axis=1)
    rh = np.linalg.norm(Q[:, :2], axis=1)
    rep = {"workdir": str(args.workdir), "mm_per_unit": mpu, "origin": O.tolist(), "half_mm": half,
           "leaf_lengths_mm": sorted(round(v, 1) for v in rib_len), "shells": []}
    print(f"{args.workdir.parent.name}: P5x leaves {sorted(round(v) for v in rib_len)} mm; views +-{half:.0f} mm")
    for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 15), (15, 25), (25, 40)]:
        m = (r >= lo) & (r < hi)
        n = int(m.sum())
        if not n:
            continue
        row = {"r_mm": [lo, hi], "points": n, "unseen": int((lab[m] == UNSEEN).sum()),
               "stem": int((lab[m] == SKELETON).sum()),
               "leaves": {int(k): int(v) for k, v in zip(*np.unique(lab[m][lab[m] >= 0], return_counts=True))},
               "above_origin": int((Q[m, 2] > 0).sum())}
        rep["shells"].append(row)
        print(f"  r {lo:2d}-{hi:2d} mm: {n:6d} pts, unseen {row['unseen'] / n:4.0%}, stem {row['stem'] / n:4.0%}, "
              f"above origin {row['above_origin'] / n:4.0%}, leaves {row['leaves']}")
    # the column over the origin: points within one third of the view horizontally, by height
    col = rh < half / 3
    for lo, hi in [(-10, 0), (0, 5), (5, 10), (10, 20), (20, 40)]:
        m = col & (Q[:, 2] >= lo) & (Q[:, 2] < hi)
        if m.any():
            print(f"    column |xy|<{half / 3:.0f} mm, z {lo:3d}..{hi:3d}: {int(m.sum()):6d} pts, unseen "
                  f"{(lab[m] == UNSEEN).mean():4.0%}, leaves {dict(zip(*[a.tolist() for a in np.unique(lab[m][lab[m] >= 0], return_counts=True)]))}")
    near = r < 1.5 * half
    c = colours(lab[near], pal)
    panels = [ortho(Q[near], c, (0, 1), half), ortho(Q[near], c, (0, 2), half), ortho(Q[near], c, (1, 2), half)]
    unseen = lab[near] == UNSEEN
    panels += [ortho(Q[near][unseen], c[unseen], (0, 1), half), ortho(Q[near][unseen], c[unseen], (0, 2), half)]
    strip = np.concatenate(panels, 1)
    for i, t in enumerate(["top (x,y)", "side (x,z)", "side (y,z)", "unseen only, top", "unseen only, side"]):
        cv2.putText(strip, t, (6 + 420 * i, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(strip, f"+-{half:.0f} mm about the origin; colour = P5x leaf, white = stem, magenta = unseen",
                (6, 412), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(str(out / "heart_ortho.jpg"), strip)

    # the steepest-down photos, the heart's points drawn over them
    import pycolmap
    sparse, _ = cloud_source.geometry(args.workdir, cloud_source.BASELINE)
    rec = pycolmap.Reconstruction(str(sparse))
    up_world = np.array([0, 0, 1.0]) @ R                  # plant z in the world
    Ow = O @ R + o
    views = []
    for im in rec.images.values():
        cam = camera_from_colmap(im, rec.cameras[im.camera_id])
        Rc, t = cam.world_to_camera[:3, :3], cam.world_to_camera[:3, 3]
        centre = -Rc.T @ t
        d = Ow - centre
        views.append((float(d @ up_world / np.linalg.norm(d)), im, cam))
    views.sort(key=lambda v: v[0])
    frames_dir = args.workdir / "p1" / "frames"
    masks_tiles = []
    by_name = sorted(views, key=lambda v: v[1].name)
    for down, im, cam in by_name[:: max(1, len(by_name) // 8)][:8]:
        stem_png = args.workdir / "p2" / "masks" / "stem" / Path(im.name).with_suffix(".png").name
        photo = cv2.imread(str(frames_dir / Path(im.name).name))
        stem = cv2.imread(str(stem_png), cv2.IMREAD_GRAYSCALE)
        if photo is None or stem is None:
            continue
        Rc, t = cam.world_to_camera[:3, :3], cam.world_to_camera[:3, 3]
        oc = Ow @ Rc.T + t
        ou = (oc[:2] / oc[2]) @ cam.K[:2, :2].T + cam.K[:2, 2]
        hp = int(1.5 * half / mpu * cam.K[0, 0] / oc[2])
        x0, y0 = max(0, int(ou[0]) - hp), max(0, int(ou[1]) - hp)
        x1, y1 = int(ou[0]) + hp, int(ou[1]) + hp
        leafm = np.zeros(stem.shape, bool)
        for d in (args.workdir / "p2" / "masks" / "leaf_instances").iterdir():
            m = cv2.imread(str(d / stem_png.name), cv2.IMREAD_GRAYSCALE) if (d / stem_png.name).exists() else None
            if m is not None:
                leafm |= m > 127
        over = (photo * 0.5).astype(np.uint8)
        over[leafm] = (0.5 * over[leafm] + (0, 110, 0)).astype(np.uint8)
        over[stem > 127] = (0.4 * over[stem > 127] + (0, 0, 150)).astype(np.uint8)
        cv2.circle(over, (int(ou[0]), int(ou[1])), 5, (0, 255, 255), 1)
        a, b = photo[y0:y1, x0:x1], over[y0:y1, x0:x1]
        tile = np.concatenate([cv2.resize(a, (360, 360)), cv2.resize(b, (360, 360))], 1)
        cv2.putText(tile, f"{Path(im.name).stem} ({np.degrees(np.arcsin(-down)):.0f} deg down): red stem mask, "
                          f"green SAM3 leaves", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        masks_tiles.append(tile)
    if masks_tiles:
        pairs = [np.concatenate(masks_tiles[i:i + 2], 1) for i in range(0, len(masks_tiles) - 1, 2)]
        cv2.imwrite(str(out / "heart_masks.jpg"), np.concatenate(pairs, 0))
    tiles = []
    for down, im, cam in views[: args.photos]:
        photo = cv2.imread(str(frames_dir / Path(im.name).name))
        if photo is None:
            continue
        Rc, t = cam.world_to_camera[:3, :3], cam.world_to_camera[:3, 3]
        X = W[near] @ Rc.T + t
        uv = (X[:, :2] / X[:, 2:3]) @ cam.K[:2, :2].T + cam.K[:2, 2]
        oc = Ow @ Rc.T + t
        ou = (oc[:2] / oc[2]) @ cam.K[:2, :2].T + cam.K[:2, 2]
        f = cam.K[0, 0]
        hp = int(half / mpu * f / oc[2])                   # the view's half-size in pixels
        x0, y0 = int(ou[0]) - hp, int(ou[1]) - hp
        crop = photo[max(0, y0):y0 + 2 * hp, max(0, x0):x0 + 2 * hp].copy()
        over = crop.copy()
        order = np.argsort(-X[:, 2])
        for k in order[:: max(1, len(order) // 60000)]:
            px, py = int(uv[k, 0]) - max(0, x0), int(uv[k, 1]) - max(0, y0)
            if 0 <= px < crop.shape[1] and 0 <= py < crop.shape[0]:
                over[py, px] = c[k][::-1]
        tile = np.concatenate([cv2.resize(crop, (420, 420)), cv2.resize(over, (420, 420))], 1)
        cv2.putText(tile, f"{Path(im.name).stem}: looks {np.degrees(np.arcsin(-down)):.0f} deg down",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(tile)
    if tiles:
        cv2.imwrite(str(out / "heart_photos.jpg"), np.concatenate(tiles, 0))
    (out / "heart_evidence.json").write_text(json.dumps(rep, indent=1))
    print(f"  wrote {out}/heart_ortho.jpg, heart_photos.jpg, heart_evidence.json")


if __name__ == "__main__":
    main()
