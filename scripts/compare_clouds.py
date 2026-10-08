#!/usr/bin/env python
"""Point-cloud variants of one plant, scored on the same yardstick and shown side by side.

    python scripts/compare_clouds.py --workdir <ds>/plant --skeleton2d <s2d out dir> \
        [--clouds p4a,p4b,p4m,p4g] [--source-root <photos>] [--out <dir>]

The question: where leaves crowd -- the heart, where new leaves emerge
intertwined -- does a cloud keep the real blades, or wrap the cluster in a
skin? The yardstick does not come from any cloud: s2d's midribs are
triangulated from SAM3's 2D masks alone (scripts/skeleton_2d.py), and on
isolated leaves they lie 0.24 mm from P4b's surface, so where they leave a
cloud's surface, the cloud is what lost the blade.

Per cloud (scores.json, printed):
  spacing        median distance to the nearest neighbour, mm
  on surface     share of s2d midrib samples with a cloud point within 0.5 mm
  buried         share inside the P4a hull with no cloud point within 1 mm --
                 blade the cloud does not have, where the hull says plant
  webbing        over pairs of distinct leaves 1.5-8 mm apart, the share of
                 the straight gap between their midribs that has a point
                 within 0.4 mm (low = the gap stayed open)
  each for isolated leaves, crowded ones (a distinct neighbour <= 8 mm) and
  the heart (leaves whose centre is within --heart-mm of the densest cluster)

The P4a hull is a solid, so its surface is taken as its boundary voxels.

Figures (in --out):
  heart_views.jpg      the heart in three photos, each next to every cloud
                       rendered from that camera, coloured by surface
                       orientation (geometry only, not the photo colour)
  cross_sections.jpg   1 mm slabs through the closest leaf pairs, one column
                       per cloud; red/blue crosses: the two s2d midribs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pose_estimator.ply_io import read_ply_vertices    # noqa: E402

VARIANTS = {"p4a": "p4/hull_points.ply", "p4b": "p4b/surface.ply",
            "p4m": "p4m/surface.ply", "p4g": "p4g/surface.ply"}
LABELS = {"p4a": "P4a hull", "p4b": "P4b 2DGS (960px)", "p4m": "P4m MVS", "p4g": "P4g 2DGS (full res)"}


def resample(poly, n):
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if len(poly) < 2 or s[-1] <= 0:
        return np.repeat(poly[:1], n, axis=0)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, poly[:, d]) for d in range(3)], axis=1)


def boundary_voxels(points, voxel):
    """Voxels of a solid with an empty face neighbour -- the solid's skin."""
    keys = np.round(points / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 3
    lin = lambda k: ((k[:, 0] + 1) * dims[1] + (k[:, 1] + 1)) * dims[2] + (k[:, 2] + 1)
    occupied = np.sort(lin(keys))
    skin = np.zeros(len(points), bool)
    for d in np.vstack([np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)]):
        q = lin(keys + d)
        at = np.clip(np.searchsorted(occupied, q), 0, len(occupied) - 1)
        skin |= occupied[at] != q
    return skin


def pca_normals(points, k=12):
    tree = cKDTree(points)
    _, nb = tree.query(points, k=min(k, len(points)))
    c = points[nb] - points[nb].mean(axis=1, keepdims=True)
    return np.linalg.eigh(np.einsum("nki,nkj->nij", c, c))[1][:, :, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--skeleton2d", required=True, type=Path,
                    help="skeleton_2d.py output dir (skeleton.json, scores.json, evidence_2d.json)")
    ap.add_argument("--clouds", default=",".join(VARIANTS),
                    help="comma list of p4a,p4b,p4m,p4g or name=path pairs")
    ap.add_argument("--source-root", type=Path, help="original photos, as for pose-gs-surface")
    ap.add_argument("--heart-mm", type=float, default=15.0, help="radius of the heart region")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    wd, sd = args.workdir, args.skeleton2d
    out = args.out or wd / "compare_clouds"
    out.mkdir(parents=True, exist_ok=True)

    pf = json.loads((wd / "p5" / "stem_graph.json").read_text())["plant_frame"]
    origin, R = np.asarray(pf["origin"], float), np.asarray(pf["rotation"], float)
    to_plant = lambda X: (np.asarray(X, float) - origin) @ R.T
    to_world = lambda X: np.asarray(X, float) @ R + origin
    mm = float(json.loads((sd / "scores.json").read_text())["mm_per_unit"])
    hull_voxel = float(json.loads((wd / "p4" / "hull.json").read_text())["voxel_size"])

    # ---- clouds, in P5's plant frame ----
    clouds = {}
    for item in args.clouds.split(","):
        name, _, rel = item.partition("=")
        path = Path(rel) if rel else wd / VARIANTS[name]
        if not path.exists():
            print(f"  {name}: {path} not found -- skipped")
            continue
        f = read_ply_vertices(path)
        pts = to_plant(np.stack([f["x"], f["y"], f["z"]], 1))
        nrm = np.stack([f["nx"], f["ny"], f["nz"]], 1) @ R.T if "nx" in f else None
        if name == "p4a":
            skin = boundary_voxels(pts, hull_voxel)
            pts, nrm = pts[skin], None
            print(f"  p4a: {len(skin):,} hull voxels, {int(skin.sum()):,} on its skin (scored as its surface)")
        if nrm is None:
            nrm = pca_normals(pts)
        clouds[name] = {"points": pts, "normals": nrm, "tree": cKDTree(pts), "path": str(path)}
    hf = read_ply_vertices(wd / "p4" / "hull_points.ply")
    hull_tree = cKDTree(to_plant(np.stack([hf["x"], hf["y"], hf["z"]], 1)))

    # ---- the yardstick: s2d midribs, crowded vs isolated, the heart ----
    sk = json.loads((sd / "skeleton.json").read_text())
    L = sk["leaves"]
    frames_of = defaultdict(set)
    for r in json.loads((sd / "evidence_2d.json").read_text())["leaves"]:
        frames_of[r["track"]].add(r["frame"])
    ribs = [resample(l["midrib"], 40)[10:] for l in L]           # outer 3/4: away from the node
    centres = np.array([resample(l["midrib"], 41)[20] for l in L])

    def covisible(a, b):
        return max([len(frames_of[x] & frames_of[y]) for x in a["sam3_ids"] for y in b["sam3_ids"]
                    if x.split("_")[0] == y.split("_")[0]] or [0])

    pairs, crowded = [], np.zeros(len(L), bool)
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            if covisible(L[i], L[j]) < 3:
                continue
            d = np.linalg.norm(ribs[i][:, None] - ribs[j][None], axis=2)
            a, b = np.unravel_index(int(np.argmin(d)), d.shape)
            if d[a, b] * mm <= 8.0:
                crowded[i] = crowded[j] = True
            if 1.5 <= d[a, b] * mm <= 8.0:
                pairs.append((d[a, b] * mm, i, j, ribs[i][a], ribs[j][b], ribs[i], ribs[j]))
    pairs.sort(key=lambda p: p[0])
    near = (np.linalg.norm(centres[:, None] - centres[None], axis=2) * mm <= 12.0).sum(1)
    heart = centres[int(np.argmax(near))]
    in_heart = np.linalg.norm(centres - heart, axis=1) * mm <= args.heart_mm
    groups = {"isolated": ~crowded, "crowded": crowded, "heart": in_heart}
    print(f"  yardstick: {len(L)} s2d leaves ({int(crowded.sum())} crowded, {int(in_heart.sum())} in "
          f"the heart, {len(pairs)} close pairs); {mm:.2f} mm/unit")

    scores, rows = {}, []
    rng = np.random.default_rng(0)
    for name, c in clouds.items():
        sample = c["points"][rng.choice(len(c["points"]), min(20000, len(c["points"])), replace=False)]
        spacing = float(np.median(c["tree"].query(sample, k=2)[0][:, 1]) * mm)
        row = {"points": int(len(c["points"])), "spacing_mm": spacing}
        for g, sel in groups.items():
            if not sel.any():
                continue
            P = np.vstack([ribs[i] for i in np.flatnonzero(sel)])
            ds = c["tree"].query(P)[0] * mm
            dh = hull_tree.query(P)[0] * mm
            row[g] = {"leaves": int(sel.sum()), "on_surface": float((ds <= 0.5).mean()),
                      "buried": float(((dh <= 0.5) & (ds > 1.0)).mean()),
                      "median_mm": float(np.median(ds))}
        web = []
        for gap, i, j, p, q, A, B in pairs:
            seg = p + np.linspace(0.3, 0.7, 15)[:, None] * (q - p)
            web.append(float((c["tree"].query(seg)[0] * mm <= 0.4).mean()))
        row["webbing_median"] = float(np.median(web)) if web else None
        row["pairs_half_filled"] = int(np.sum(np.asarray(web) > 0.5)) if web else None
        scores[name] = row
    scores["_yardstick"] = {"leaves": len(L), "crowded": int(crowded.sum()), "heart": int(in_heart.sum()),
                            "close_pairs": len(pairs), "mm_per_unit": mm, "heart_point": heart.tolist(),
                            "skeleton2d": str(sd)}
    (out / "scores.json").write_text(json.dumps(scores, indent=1))

    head = f"{'cloud':22s} {'points':>10s} {'spacing':>8s} | " + " | ".join(
        f"{g + ' on/buried':^19s}" for g in groups) + f" | {'webbing':>7s} {'>50%':>5s}"
    print("\n" + head + "\n" + "-" * len(head))
    for name, r in scores.items():
        if name.startswith("_"):
            continue
        cells = " | ".join(f"{r[g]['on_surface']:6.0%} / {r[g]['buried']:4.0%}     " if g in r else " " * 19
                           for g in groups)
        print(f"{LABELS.get(name, name):22s} {r['points']:>10,} {r['spacing_mm']:6.3f}mm | {cells} | "
              f"{r['webbing_median']:7.0%} {r['pairs_half_filled']:>2d}/{len(pairs)}")

    # ---- figure 1: the heart, photo vs every cloud from the same camera ----
    try:
        heart_views(wd, args.source_root, clouds, heart, args.heart_mm / mm, to_world, mm, out)
    except FileNotFoundError as e:
        print(f"  heart_views.jpg skipped: {e}")
    # ---- figure 2: cross-sections through the closest pairs ----
    cross_sections(clouds, pairs[:6], L, mm, out)
    print(f"\n  wrote {out / 'scores.json'}, heart_views.jpg, cross_sections.jpg")


def heart_views(wd, source_root, clouds, heart, radius, to_world, mm, out, n_views=3, scale=0.5):
    """Photo crop of the heart, then each cloud rendered from that camera, shaded by orientation."""
    from pose_estimator.fine_views import load_fine_views

    hf = read_ply_vertices(wd / "p4" / "hull_points.ply")
    hull = np.stack([hf["x"], hf["y"], hf["z"]], 1)
    manifest = [m["frame"] for m in json.loads((wd / "p1" / "manifest.json").read_text())]
    pick = [manifest[int(i)] for i in np.linspace(0, len(manifest) - 1, n_views + 1)[:-1].round()]
    views = load_fine_views(wd, hull, scale=scale, source_root=source_root, frames=pick, verbose=False)
    H = 420
    rows = []
    centre_w = to_world(heart[None])[0]
    for v in views:
        R, t, K = v.world_to_camera[:3, :3], v.world_to_camera[:3, 3], v.K
        c = R @ centre_w + t
        u0 = (c[:2] / c[2]) @ K[:2, :2].T + K[:2, 2]
        r_px = int(radius * K[0, 0] / c[2])
        x0, y0 = int(u0[0]) - r_px, int(u0[1]) - r_px
        size = 2 * r_px
        photo = np.zeros((size, size, 3), np.uint8)
        ys, xs = slice(max(0, y0), max(0, y0 + size)), slice(max(0, x0), max(0, x0 + size))
        patch = cv2.cvtColor(v.image[ys, xs], cv2.COLOR_RGB2BGR)
        photo[ys.start - y0:ys.start - y0 + patch.shape[0], xs.start - x0:xs.start - x0 + patch.shape[1]] = patch
        tiles = [_label(cv2.resize(photo, (H, H)), f"photo {v.name}")]
        for name, cl in clouds.items():
            P = to_world(cl["points"])
            cam = P @ R.T + t
            uv = (cam[:, :2] / cam[:, 2:3]) @ K[:2, :2].T + K[:2, 2] - [x0, y0]
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size) & (cam[:, 2] > 0)
            spacing_px = np.median(cl["tree"].query(cl["points"][::max(1, len(cl["points"]) // 5000)], k=2)[0][:, 1]) \
                * K[0, 0] / c[2]
            rad = int(np.clip(np.ceil(spacing_px * 0.75), 1, 6))
            zbuf = np.full((size, size), np.inf)
            colour = np.zeros((size, size, 3))
            # colour = surface orientation in this camera's frame, turned to
            # face the camera: overlapping leaves at different angles show
            # as different colours, a skin wrapping them as one smooth one
            n_cam = _to_world_dirs(cl["normals"][ok], to_world) @ R.T
            n_cam *= -np.sign(np.einsum("ij,ij->i", n_cam, cam[ok] / np.linalg.norm(cam[ok], axis=1,
                                                                                  keepdims=True)))[:, None]
            rgb = np.clip(n_cam * 0.5 + 0.5, 0, 1)[:, ::-1]      # BGR for cv2
            px, py, pz = uv[ok, 0].astype(int), uv[ok, 1].astype(int), cam[ok, 2]
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    if dx * dx + dy * dy > rad * rad:
                        continue
                    qx, qy = px + dx, py + dy
                    m = (qx >= 0) & (qx < size) & (qy >= 0) & (qy < size)
                    idx = np.flatnonzero(m)
                    flat = qy[idx] * size + qx[idx]
                    best = np.full(size * size, np.inf)
                    np.minimum.at(best, flat, pz[idx])
                    win = pz[idx] <= best[flat]
                    closer = win & (pz[idx] < zbuf.ravel()[flat])
                    sel = idx[closer]
                    zbuf[qy[sel], qx[sel]] = pz[sel]
                    colour[qy[sel], qx[sel]] = rgb[sel]
            img = (colour * 255).astype(np.uint8)
            img[~np.isfinite(zbuf)] = 0
            tiles.append(_label(cv2.resize(img, (H, H), interpolation=cv2.INTER_AREA),
                                LABELS.get(name, name)))
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out / "heart_views.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def _to_world_dirs(n_plant, to_world):
    origin = to_world(np.zeros((1, 3)))[0]
    return to_world(n_plant) - origin


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def cross_sections(clouds, pairs, L, mm, out, S=300):
    half = 9.0 / mm
    rows = []
    for gap, i, j, p, q, A, B in pairs:
        c = (p + q) / 2
        u = (q - p) / np.linalg.norm(q - p)
        tangent = lambda C, x: C[min(np.argmin(np.linalg.norm(C - x, axis=1)) + 1, len(C) - 1)] - \
            C[max(np.argmin(np.linalg.norm(C - x, axis=1)) - 1, 0)]
        ta, tb = tangent(A, p), tangent(B, q)
        n = ta / np.linalg.norm(ta) + np.sign(ta @ tb) * tb / np.linalg.norm(tb)
        n -= (n @ u) * u
        n /= np.linalg.norm(n)
        v = np.cross(n, u)
        tiles = []
        for name, cl in clouds.items():
            img = np.full((S, S, 3), 255, np.uint8)
            P = cl["points"][cl["tree"].query_ball_point(c, 1.5 * half)]
            P = P[np.abs((P - c) @ n) <= 0.5 / mm]
            uv = (np.stack([(P - c) @ u, (P - c) @ v], 1) / half * (S / 2) + S / 2).astype(int)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < S) & (uv[:, 1] >= 0) & (uv[:, 1] < S)
            img[S - 1 - uv[ok, 1], uv[ok, 0]] = (60, 60, 60)
            for X, col in ((p, (0, 0, 255)), (q, (255, 0, 0))):
                x, y = (np.array([(X - c) @ u, (X - c) @ v]) / half * (S / 2) + S / 2).astype(int)
                cv2.drawMarker(img, (x, S - 1 - y), col, cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img, LABELS.get(name, name), (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f"[{L[i]['id']}]/[{L[j]['id']}] gap {gap:.1f} mm", (5, S - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (0, 0), (S - 1, S - 1), (200, 200, 200), 1)
            tiles.append(img)
        rows.append(np.hstack(tiles))
    if rows:
        cv2.imwrite(str(out / "cross_sections.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
