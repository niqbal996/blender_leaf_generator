#!/usr/bin/env python
"""Hand labels of the apices (LabelMe) as a yardstick: apex positions in 3D, 3D leaves, scores.

    python scripts/apex_labels.py --workdir <specimen>/plant --out <specimen>/plant/debug/apex_labels \
        [--labels <specimen>/plant/p1/frames] [--score <s2d dir> ...]

Step 0b of docs/apex-plan.md. A measuring tool for one plant, not part of
the method. Reads LabelMe JSONs next to the P1 frames (1920 px, the
pipeline's own pixels):

  apex_<X>    rectangle round a whole apex cluster, same name in every frame
  leaf*       linestrip base -> tip of a leaf fully visible there. Any
              number after "leaf" is ignored: which leaf is which across
              frames is not labelled (nobody can tell, at the apex)

Writes into --out:

  apices.json        per apex: the cloud points inside its box in every view
                     that boxes it -- centre, radius, and how many points
  apex_regions.npz   those points per apex (world frame), for apex_zoom.py --regions
  labels.jpg         every labelled frame, labels drawn, each apex's carved
                     points projected back (red)
  scores.json        (--score) per s2d skeleton, per apex: your leaves found,
                     its leaves that are yours, tip and midrib errors
  score_<name>.jpg   (--score) per apex and frame: your lines white, the
                     skeleton's leaves coloured, matched ones solid

Scoring needs no identity across frames: in each frame every skeleton leaf
is projected, and the labelled lines and projected leaves are paired one to
one by how well they overlap (MATCH_PX). A labelled leaf with no partner is
missed; a skeleton leaf in the apex box with no partner is extra.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p5x_views import Run                                             # noqa: E402
from skeleton_2d import resample                                      # noqa: E402

MATCH_PX = 15.0     # a pairing costs more than this (~1 mm at the apex): not the same leaf
BOX_GROW = 0.15     # an apex box grown by this share of its size, for "in the apex"
CARVE_GROW = 0.05   # ... and for carving: hand-drawn boxes are a little tight here and there
SAMPLES = 30


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def load(labels: Path) -> dict:
    """frame -> {"apices": {name: (x0, y0, x1, y1)}, "leaves": [(None, (n, 2) base->tip)]}."""
    out = {}
    for path in sorted(labels.glob("frame_*.json")):
        doc = json.loads(path.read_text())
        apices, leaves = {}, []
        for s in doc["shapes"]:
            pts = np.asarray(s["points"], float)
            if s["shape_type"] == "rectangle" and s["label"].startswith("apex"):
                (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
                apices[s["label"]] = (x0, y0, x1, y1)
            elif s["shape_type"] in ("linestrip", "line") and s["label"].startswith("leaf"):
                if len(pts) >= 2:
                    leaves.append((None, pts))
        out[path.stem] = {"apices": apices, "leaves": leaves}
    return out


def grown(box, k=BOX_GROW):
    x0, y0, x1, y1 = box
    dx, dy = k * (x1 - x0), k * (y1 - y0)
    return x0 - dx, y0 - dy, x1 + dx, y1 + dy


def inside(box, pts) -> np.ndarray:
    x0, y0, x1, y1 = box
    pts = np.atleast_2d(pts)
    return (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)


def apex_of(boxes: dict, line: np.ndarray):
    """The apex whose (grown) box holds most of the line, or None."""
    best, share = None, 0.5
    for name, box in boxes.items():
        s = inside(grown(box), resample(line, SAMPLES)).mean()
        if s > share:
            best, share = name, s
    return best


# --------------------------------------------------------------------------
# 3D
# --------------------------------------------------------------------------


def carve_apices(run: Run, labels: dict, mpu: float):
    """Each apex: the cloud points that project inside its box (grown by
    CARVE_GROW) in every view that boxes it -- the boxes' visual hull, cut
    out of the plant. A box centre is not one 3D point when the cluster is
    lopsided (A and C reprojected 22-24 px off as box-centre rays)."""
    views = defaultdict(list)
    for frame, lab in labels.items():
        for name, box in lab["apices"].items():
            views[name].append((frame, box))
    report, regions = {}, {}
    for name, obs in sorted(views.items()):
        if len(obs) < 2:
            print(f"  {name}: boxed in {len(obs)} frame -- needs 2 to place it in 3D")
            continue
        keep = np.ones(len(run.points), bool)
        for frame, box in obs:
            keep &= inside(grown(box, CARVE_GROW), project_world(run, frame, run.points))
        pts = run.points[keep]
        if len(pts) < 10:
            print(f"  {name}: only {len(pts)} cloud points inside all its boxes -- boxes disagree?")
            continue
        centre = pts.mean(axis=0)
        radius = float(np.percentile(np.linalg.norm(pts - centre, axis=1), 95))
        regions[name] = pts
        report[name] = {"views": [f for f, _ in obs], "points": int(len(pts)),
                        "centre_world": centre.tolist(),
                        "centre_plant": ((centre - run.origin) @ run.rotation.T).tolist(),
                        "radius_mm": radius * mpu}
        print(f"  {name}: {len(obs)} boxes carve {len(pts)} cloud points, radius {radius * mpu:.1f} mm")
    return report, regions


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def project_world(run: Run, frame: str, world_xyz) -> np.ndarray:
    im = run.image_of[frame]
    cam = run.rec.cameras[im.camera_id]
    pose = im.cam_from_world()
    c = np.asarray(world_xyz) @ pose.rotation.matrix().T + np.asarray(pose.translation)
    return cam.img_from_cam(c)


def project(run: Run, frame: str, plant_xyz) -> np.ndarray:
    im = run.image_of[frame]
    cam = run.rec.cameras[im.camera_id]
    pose = im.cam_from_world()
    c = run.to_world(plant_xyz) @ pose.rotation.matrix().T + np.asarray(pose.translation)
    return cam.img_from_cam(c)


def skeleton_curves(run: Run, frame: str, skeleton: dict):
    """Per skeleton leaf: (name, 2D whole curve petiole+midrib, 2D midrib, 2D tip)."""
    out = []
    for leaf in skeleton["leaves"]:
        rib = np.asarray(leaf["midrib"], float)
        pet = np.asarray(leaf.get("petiole") or [], float)
        whole = np.vstack([pet, rib]) if len(pet) >= 2 else rib
        out.append((leaf.get("track", str(leaf["id"])), project(run, frame, resample(whole, SAMPLES)),
                    project(run, frame, resample(rib, SAMPLES)), project(run, frame, [leaf["tip"]])[0]))
    return out


def pair_cost(gt: np.ndarray, rib: np.ndarray, tip: np.ndarray) -> float:
    """How far a projected skeleton leaf is from a labelled one, px: its blade
    lying on the drawn line, and its tip at the drawn tip."""
    on_line = cKDTree(resample(gt, 4 * SAMPLES)).query(rib)[0].mean()
    return 0.5 * (on_line + np.linalg.norm(tip - gt[-1]))


def score(run: Run, labels: dict, skeleton: dict) -> dict:
    per_apex = defaultdict(lambda: defaultdict(list))
    pairs = {}
    for frame, lab in labels.items():
        sk = skeleton_curves(run, frame, skeleton)
        # labelled lines and skeleton leaves go to an apex by one rule: the box
        # holding most of them. Boxes overlap (C holds D in frame_0000), and
        # "any box holding it" counted a neighbour's leaves as extras of both.
        sk_apex = [apex_of(lab["apices"], s[2]) for s in sk]
        for apex, box in lab["apices"].items():
            gts = [line for _, line in lab["leaves"] if apex_of(lab["apices"], line) == apex]
            near = [s for s, a in zip(sk, sk_apex) if a == apex]
            cost = np.array([[pair_cost(g, s[2], s[3]) for s in near] for g in gts]).reshape(
                len(gts), len(near))
            rows, cols = linear_sum_assignment(cost) if cost.size else ([], [])
            ok = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < MATCH_PX]
            d = per_apex[apex]
            d["labelled"].append(len(gts))
            d["skeleton"].append(len(near))
            d["matched"].append(len(ok))
            for r, c in ok:
                g, s = gts[r], near[c]
                d["tip_px"].append(float(np.linalg.norm(s[3] - g[-1])))
                d["blade_on_line_px"].append(float(cKDTree(resample(g, 4 * SAMPLES)).query(s[2])[0].mean()))
                d["start_px"].append(float(np.linalg.norm(s[1][0] - g[0])))
            pairs[frame, apex] = (gts, near, ok)
    out = {}
    for apex, d in sorted(per_apex.items()):
        lab, sk, m = sum(d["labelled"]), sum(d["skeleton"]), sum(d["matched"])
        med = lambda k: float(np.median(d[k])) if d[k] else None
        out[apex] = {"labelled": lab, "skeleton_in_box": sk, "matched": m,
                     "found": m / lab if lab else None, "precision": m / sk if sk else None,
                     "tip_px": med("tip_px"), "blade_on_line_px": med("blade_on_line_px"),
                     "start_px": med("start_px")}
    total = {k: sum(out[a][k] for a in out) for k in ("labelled", "skeleton_in_box", "matched")}
    total["found"] = total["matched"] / max(total["labelled"], 1)
    total["precision"] = total["matched"] / max(total["skeleton_in_box"], 1)
    out["all"] = total
    return out, pairs


# --------------------------------------------------------------------------
# pictures
# --------------------------------------------------------------------------


def colour(i: int) -> tuple:
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[(i * 47) % 180, 220, 255]]]),
                                              cv2.COLOR_HSV2BGR)[0, 0])


def draw_labels(run: Run, labels: dict, regions: dict, out: Path):
    tiles = []
    for frame, lab in labels.items():
        img = cv2.imread(str(run.workdir / "p1" / "frames" / f"{frame}.jpg"))
        for name, box in lab["apices"].items():
            x0, y0, x1, y1 = (int(v) for v in box)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.putText(img, name, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if name in regions:
                for u, v in project_world(run, frame, regions[name][::7]).astype(int):
                    if 0 <= v < img.shape[0] and 0 <= u < img.shape[1]:
                        img[v, u] = (0, 0, 255)
        for _, line in lab["leaves"]:
            cv2.polylines(img, [line.astype(np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, tuple(line[0].astype(int)), 5, (0, 160, 0), -1)      # base: green
            cv2.circle(img, tuple(line[-1].astype(int)), 5, (0, 0, 255), -1)     # tip: red
        ys, xs = np.nonzero(cv2.imread(str(run.workdir / "p2" / "masks" / "plant" / f"{frame}.png"), 0))
        crop = img[max(ys.min() - 30, 0):ys.max() + 30, max(xs.min() - 30, 0):xs.max() + 30]
        crop = cv2.resize(crop, (int(crop.shape[1] * 900 / crop.shape[0]), 900))
        cv2.putText(crop, frame, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        tiles.append(crop)
    w = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in tiles]
    rows = [np.hstack(tiles[i:i + 3]) if len(tiles[i:i + 3]) == 3 else
            np.hstack(tiles[i:i + 3] + [np.zeros_like(tiles[0])] * (3 - len(tiles[i:i + 3])))
            for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(out / "labels.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def draw_score(run: Run, labels: dict, pairs: dict, path: Path, title: str):
    rows = []
    for apex in sorted({a for _, a in pairs}):
        tiles = []
        for frame in labels:
            if (frame, apex) not in pairs:
                continue
            gts, near, ok = pairs[frame, apex]
            x0, y0, x1, y1 = (int(v) for v in grown(labels[frame]["apices"][apex], 0.4))
            img = cv2.imread(str(run.workdir / "p1" / "frames" / f"{frame}.jpg"))
            img = (img * 0.6).astype(np.uint8)
            matched_g = {r for r, _ in ok}
            matched_s = {c for _, c in ok}
            for r, g in enumerate(gts):
                cv2.polylines(img, [g.astype(np.int32)], False, (255, 255, 255),
                              2 if r in matched_g else 1, cv2.LINE_AA)
            for c, s in enumerate(near):
                col = colour(c)
                cv2.polylines(img, [s[1].astype(np.int32)], False, col,
                              2 if c in matched_s else 1, cv2.LINE_AA)
                cv2.circle(img, tuple(s[3].astype(int)), 3, col, -1)
            h, w = img.shape[:2]
            t = img[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]
            t = cv2.resize(t, (360, int(360 * t.shape[0] / max(t.shape[1], 1))))[:360]
            t = np.pad(t, ((0, 360 - t.shape[0]), (0, 0), (0, 0)))
            cv2.putText(t, f"{frame} {apex}: {len(ok)}/{len(gts)} found, {len(near)} drawn",
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(t)
        if tiles:
            rows.append(np.hstack(tiles + [np.zeros_like(tiles[0])] * (len(labels) - len(tiles))))
    img = np.vstack(rows)
    cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, title + "   white: your lines (thick = found)   colour: skeleton leaves",
                (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--labels", type=Path, help="folder of LabelMe JSONs (default: <workdir>/p1/frames)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--score", type=Path, nargs="*", default=[],
                    help="s2d output folders (with skeleton.json) to score against the labels")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mpu = json.loads((args.workdir / "tag_scale.json").read_text())["mm_per_unit"]
    run = Run(args.workdir)
    labels = load(args.labels or args.workdir / "p1" / "frames")
    n_lines = sum(len(l["leaves"]) for l in labels.values())
    print(f"  {len(labels)} labelled frames, {n_lines} leaf lines")

    apices, regions = carve_apices(run, labels, mpu)
    (args.out / "apices.json").write_text(json.dumps(apices, indent=1))
    np.savez_compressed(args.out / "apex_regions.npz", **regions)
    draw_labels(run, labels, regions, args.out)

    scores = {}
    for d in args.score:
        sk = json.loads((d / "skeleton.json").read_text())
        scores[d.name], pairs = score(run, labels, sk)
        draw_score(run, labels, pairs, args.out / f"score_{d.name}.jpg", d.name)
    if scores:
        (args.out / "scores.json").write_text(json.dumps(scores, indent=1))
        apex_names = sorted(a for a in next(iter(scores.values())) if a != "all")
        print(f"\n  your leaves found (matched / labelled), share of skeleton leaves in the box that "
              f"are yours, median tip error px")
        print("  " + "".ljust(10) + "".join(n[:26].ljust(28) for n in scores))
        for a in apex_names + ["all"]:
            cells = []
            for n in scores:
                s = scores[n][a]
                tip = f", tip {s['tip_px']:.0f}px" if s.get("tip_px") is not None else ""
                cells.append(f"{s['matched']}/{s['labelled']}, prec {s['precision'] or 0:.0%}{tip}"
                             [:26].ljust(28))
            print("  " + a.ljust(10) + "".join(cells))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
