#!/usr/bin/env python
"""Do the 3D leaves look like the 2D leaves? Several P5x runs, scored on the same photos.

    python scripts/compare_runs.py --runs <ds>/plant <ds>/plant_pass0 <ds>/plant_pass0_1 \
        --names full pass0 pass0+1 --masks runs/<specimen>_p5x_views/masks.npz \
        --out runs/<specimen>_compare

Diagnostic only. Every run is rendered from the cameras of --eval-pass (default
0, which every run here contains) and compared with SAM3's own masks of those
frames -- the masks are the clean signal, so they are the reference:

  coverage   share of a SAM3 leaf's pixels that its 3D leaf reproduces. Low
             means a truncated or missing leaf. A 2D leaf that lands on no 3D
             leaf scores 0, so a run cannot look good by dropping leaves.
  purity     share of the rendered 3D leaf that falls on that SAM3 leaf. Low
             means the 3D leaf bleeds -- speckle, or two leaves fused.
  fused      pairs of SAM3 leaves shown side by side (>= 2 frames) that ended
             up as one 3D leaf.

The figure colours each 3D leaf by the SAM3 leaf that lands on it most, so a
physical leaf has one colour in the photo and in every run.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from p5x_views import Run, crop_box, unpack
from pose_estimator.cli.leaf_instances import ROOT, SKELETON
from pose_estimator.semantic import render_points, surface_scale, visible_only

LOST_SHARE = 0.15
SPLAT = 3
# A SAM3 "leaf" mask over this share of the frame's plant mask is the whole
# plant, not a leaf (gaensefuss_1 has ids that do this on some frames). It is
# left out of scoring and drawing for every run alike.
JUNK_SHARE = 0.30


def junk_masks(masks, workdir: Path, frames) -> set:
    plant_area = {}
    for f in frames:
        m = cv2.imread(str(workdir / "p2" / "masks" / "plant" / f"{f}.png"), cv2.IMREAD_GRAYSCALE)
        plant_area[f] = int((m > 127).sum()) if m is not None else 0
    out = set()
    for k, f in enumerate(masks["frame"]):
        f = str(f)
        if f in plant_area and unpack(masks, k).sum() > JUNK_SHARE * max(plant_area[f], 1):
            out.add(k)
    return out


def track_colour(i: int) -> tuple:
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.85, 1.0 if i % 2 == 0 else 0.75)
    return int(b * 255), int(g * 255), int(r * 255)


def label_image(run: Run, frame: str) -> np.ndarray:
    """3D label per pixel as the run's cloud renders from this camera; -9 = nothing.

    Rendered with the occlusion test, so what is scored is what a camera would
    actually see of the 3D model -- not points showing through gaps.
    """
    if not hasattr(run, "_scale"):
        run._scale = surface_scale(run.points)
    spacing, thickness = run._scale
    camera = run.camera(frame)
    _rgb, index = render_points(run.points, np.zeros((len(run.points), 3), np.uint8),
                                camera, splat_radius=SPLAT)
    index = visible_only(run.points, camera, index, spacing, 2.0 * thickness)
    out = np.full(index.shape, -9, np.int32)
    hit = index >= 0
    out[hit] = run.assignment[index[hit]]
    return out


def score(run: Run, masks, frames, tracks, junk=frozenset()):
    by_frame = defaultdict(list)
    for k, (t, f) in enumerate(zip(masks["track"], masks["frame"])):
        if str(f) in frames and str(t) in tracks and k not in junk:
            by_frame[str(f)].append(k)
    labels = {f: label_image(run, f) for f in frames}

    # which 3D leaf each SAM3 leaf lands on, over all its frames. The share is
    # of everything the mask's pixels hit (stem and root included), as in
    # p5x_views: a leaf whose pixels mostly land on stem is lost, not found.
    tally = defaultdict(lambda: defaultdict(int))
    for f, ks in by_frame.items():
        for k in ks:
            x0, y0, x1, y1 = masks["box"][k]
            under = labels[f][y0:y1, x0:x1][unpack(masks, k)]
            for lab, n in zip(*np.unique(under[under != -9], return_counts=True)):
                tally[str(masks["track"][k])][int(lab)] += int(n)
    leaf_of, pixels = {}, {}
    for t in tracks:
        votes = tally.get(t, {})
        leaves = {lab: n for lab, n in votes.items() if lab >= 0}
        best = max(leaves, key=leaves.get) if leaves else None
        total = sum(votes.values())
        leaf_of[t] = best if best is not None and leaves[best] / total >= LOST_SHARE else None
        pixels[t] = leaves

    # per (track, frame): coverage and purity against that frame's render of its leaf
    cover, purity = defaultdict(list), defaultdict(list)
    kernel = np.ones((5, 5), np.uint8)
    for f, ks in by_frame.items():
        lab = labels[f]
        rendered = {}
        for k in ks:
            t = str(masks["track"][k])
            x0, y0, x1, y1 = masks["box"][k]
            m = unpack(masks, k)
            leaf = leaf_of.get(t)
            if leaf is None:
                cover[t].append(0.0)
                continue
            if leaf not in rendered:
                rendered[leaf] = cv2.morphologyEx((lab == leaf).astype(np.uint8), cv2.MORPH_CLOSE,
                                                  kernel).astype(bool)
            r = rendered[leaf]
            inter = int((r[y0:y1, x0:x1] & m).sum())
            cover[t].append(inter / max(int(m.sum()), 1))
            if r.any():
                purity[t].append(inter / int(r.sum()))
    return leaf_of, pixels, cover, purity, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--masks", type=Path, required=True, help="masks.npz from p5x_views.py")
    ap.add_argument("--eval-pass", type=int, default=0)
    ap.add_argument("--figure-frames", nargs="*", default=["frame_0022", "frame_0033"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    z = np.load(args.masks, allow_pickle=True)
    masks = {k: z[k] for k in z.files}
    sources = json.loads((args.runs[0] / "p1" / "sources.json").read_text())
    frames = {f for f, p in sources.items() if int(p) == args.eval_pass}
    prefix = f"pass{args.eval_pass}_"
    life = defaultdict(set)
    for t, f in zip(masks["track"], masks["frame"]):
        if str(t).startswith(prefix):
            life[str(t)].add(str(f))
    tracks = sorted(t for t, fs in life.items() if len(fs) >= 3)
    colour = {t: track_colour(i) for i, t in enumerate(tracks)}
    junk = junk_masks(masks, args.runs[0], frames)
    print(f"  {len(junk)} whole-plant SAM3 'leaf' masks left out (> {JUNK_SHARE:.0%} of the plant)")

    rows, rendered = [], {}
    for path, name in zip(args.runs, args.names):
        run = Run(path)
        leaf_of, pixels, cover, purity, labels = score(run, masks, frames, tracks, junk)
        landed = defaultdict(list)
        for t, leaf in leaf_of.items():
            if leaf is not None:
                landed[leaf].append(t)
        fused = sum(1 for ts in landed.values() for i in range(len(ts)) for j in range(i + 1, len(ts))
                    if len(life[ts[i]] & life[ts[j]]) >= 2)
        report = json.loads((path / "p5x" / "instances.json").read_text())
        hull = json.loads((path / "p4" / "hull.json").read_text())
        p4b = json.loads((path / "p4b" / "p4b.json").read_text())
        p4b_iou = p4b.get("silhouette_iou")
        rows.append({
            "run": name, "views": hull["num_views"], "hull_iou": hull["reprojection_iou"]["mean"],
            "p4b_iou": p4b_iou, "leaves_3d": report["num_leaves_3d"],
            "leaves_with_a_2d_leaf": len(landed),
            "tracks": len(tracks), "lost": sum(1 for v in leaf_of.values() if v is None),
            "fused_pairs": fused,
            "coverage": float(np.median([np.mean(v) for v in cover.values() if v])),
            "purity": float(np.median([np.mean(v) for v in purity.values() if v])),
        })
        # colour each 3D leaf by the SAM3 leaf that put the most pixels on it
        owner = {}
        for leaf in landed:
            owner[leaf] = max(tracks, key=lambda t: pixels[t].get(leaf, 0))
        rendered[name] = (labels, owner)

    print(f"\n  scored on the {len(frames)} pass-{args.eval_pass} photos, {len(tracks)} SAM3 leaves "
          f"(ids living >= 3 frames)\n")
    head = (f"  {'run':<9}{'views':>6}{'hull IoU':>10}{'P4b IoU':>9}{'3D leaves':>11}"
            f"{'lost 2D':>9}{'fused':>7}{'coverage':>10}{'purity':>8}")
    print(head)
    for r in rows:
        print(f"  {r['run']:<9}{r['views']:>6}{r['hull_iou']:>10.3f}"
              f"{(r['p4b_iou'] or float('nan')):>9.3f}{r['leaves_3d']:>11}{r['lost']:>9}"
              f"{r['fused_pairs']:>7}{r['coverage']:>10.1%}{r['purity']:>8.1%}")
    (args.out / "scores.json").write_text(json.dumps(rows, indent=1))

    # --- one figure: the photo, then each run, same colour per physical leaf ---
    sheets = []
    for frame in args.figure_frames:
        photo = cv2.imread(str(args.runs[0] / "p1" / "frames" / f"{frame}.jpg"))
        base = (photo * 0.5).astype(np.uint8)
        drawn = [k for k, (t, f) in enumerate(zip(masks["track"], masks["frame"]))
                 if str(f) == frame and str(t) in colour and k not in junk]
        for k in sorted(drawn, key=lambda k: -int(unpack(masks, k).sum())):   # small on top
            t = masks["track"][k]
            if True:
                x0, y0, x1, y1 = masks["box"][k]
                m = unpack(masks, k)
                region = base[y0:y1, x0:x1]
                region[m] = (0.25 * region[m] + 0.75 * np.array(colour[str(t)])).astype(np.uint8)
        tiles = [base]
        for name in args.names:
            labels, owner = rendered[name]
            lab = labels[frame]
            img = np.zeros_like(photo)
            img[lab == SKELETON] = (150, 150, 150)
            img[lab == ROOT] = (40, 90, 140)
            for leaf in np.unique(lab[lab >= 0]):
                t = owner.get(int(leaf))
                img[lab == leaf] = colour[t] if t is not None else (255, 255, 255)
            tiles.append(img)
        x0, y0, x1, y1 = crop_box(args.runs[0], frame)
        tiles = [t[y0:y1, x0:x1].copy() for t in tiles]
        for t, title in zip(tiles, ["SAM3 (2D)"] + [f"3D: {n}" for n in args.names]):
            cv2.putText(t, title, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                        cv2.LINE_AA)
        sheets.append(np.hstack(tiles))
    width = max(sh.shape[1] for sh in sheets)
    sheet = np.vstack([np.pad(sh, ((0, 0), (0, width - sh.shape[1]), (0, 0))) for sh in sheets])
    sheet = cv2.resize(sheet, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(args.out / "runs_side_by_side.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\n  figure: {args.out / 'runs_side_by_side.jpg'}  "
          "(same colour = same SAM3 leaf; white = 3D leaf no SAM3 leaf lands on)")


if __name__ == "__main__":
    main()
