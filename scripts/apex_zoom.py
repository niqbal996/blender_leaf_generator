#!/usr/bin/env python
"""SAM3 re-run zoomed in on every branch apex, its leaf masks merged into P2's.

    python scripts/apex_zoom.py --workdir <specimen>/plant --out <specimen>/plant/debug/apex_zoom
    python scripts/apex_zoom.py --workdir <specimen>/plant --out <dir> \
        --compare <dir>/s2d_baseline <dir>/s2d_zoom      # after skeleton_2d.py on each
    python scripts/apex_zoom.py ... --regions <dir>/apex_regions.npz   # crops from labelled apices

Step 0 of docs/apex-plan.md: do finer masks alone fix the apex? Diagnostic --
reads a finished run and its s2d skeleton.json (for the stem tree), writes into
--out only:

  sam3/apex<id>_pass<p>.npz   zoomed masks at crop resolution: crop_<i> (x0, y0, side in
                              original px) and up_<i> per frame, m_<i>_<obj> per mask
  sam3/apex<id>_pass<p>.jpg   photo | P2 masks | zoomed masks kept (dropped ones grey)
  masks_zoom.npz              p5x_views.cache_masks' format, for skeleton_2d.py --masks:
                              P2's leaf masks with every apex leaf replaced by its zoomed mask
  apex_zoom.json              apices, crop sizes, masks added / trimmed / dropped per apex
                              (--merge-only redoes masks_zoom.npz from sam3/ without SAM3)
  compare/apex<id>.jpg        (--compare) photo | each skeleton, on the apex squares
  compare/summary.json        (--compare) per skeleton and apex: leaves there, lengths, fit

Where the squares go: by default each is CROP_MM wide, centred on the end of
a stem-tree branch. That end stops where the cloud closes over the cluster,
short of the real apex, and a fixed size cuts big clusters (vogelmeere's
labelled ones are 9-15 mm in radius). With --regions (apex_labels.py: the
cloud carved by hand-drawn apex boxes) each square is instead the
projection of the apex's own points, plus REGION_MARGIN, in every frame.

Why: P2 ran SAM3 on the whole plant, so its 288 px mask head lands ~13 px of
the 6000 px original per step -- about 0.3 mm, a tenth of a young leaf -- and
one mask often covers two or three apex leaves. A square of CROP_MM around
the apex fills SAM3's 1008 px input instead, ~5x finer per step.

What counts as an apex leaf: a zoomed mask that stays inside its square. One
that reaches the square's edge is a larger leaf passing through, and P2's
mask of it stays. Every P2 mask loses the pixels the apex leaves now claim,
and is dropped if less than KEEP_SHARE of it is left: those were the masks
drawn over several apex leaves at once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import track_colour                                 # noqa: E402
from p5x_views import Run, unpack                                     # noqa: E402
from pose_estimator.fine_views import source_image                   # noqa: E402

CROP_MM = 18.0       # side of the square cut around each apex: the cluster plus its first pair
SAM_PX = 1008        # SAM3's input size; the crop is resampled to it
EDGE_PX = 6          # a zoomed mask within this of its square's edge leaves the square
MIN_AREA = 1500      # zoomed mask pixels (at SAM_PX) below which it is speckle
DUPLICATE_IOU = 0.5  # two apices' squares overlap: a mask this close to one kept is that one
KEEP_SHARE = 0.4     # a P2 mask keeps what the apex leaves left of it only if this much is left
REGION_MARGIN = 0.15 # --regions: a square is the region's projected extent plus this share each side


def apices(run: Run, skeleton: dict, regions: dict | None = None):
    """(id, kind, world point): each labelled region's centre with --regions,
    else the end of the stem and of every branch of the stem tree."""
    if regions:
        return [(name.replace("apex_", ""), "labelled", pts.mean(axis=0))
                for name, pts in sorted(regions.items())]
    return [(int(a["id"]), a["kind"], run.to_world(a["points"][-1]))
            for a in skeleton.get("axes") or [] if a["kind"] in ("stem", "branch")]


def tag(aid) -> str:
    return f"{aid:02d}" if isinstance(aid, int) else str(aid)


def square(run: Run, frame: str, X: np.ndarray, mm_per_unit: float, photo_w: int, pts=None):
    """(x0, y0, side) in the original photo's pixels, or None if the apex is behind the
    camera: round the projection of the region's points `pts` if given, else CROP_MM
    centred on X."""
    im = run.image_of[frame]
    cam = run.rec.cameras[im.camera_id]
    up = photo_w / cam.width
    pose = im.cam_from_world()
    if pts is not None:
        c = pts @ pose.rotation.matrix().T + np.asarray(pose.translation)
        if (c[:, 2] <= 0).any():
            return None
        uv = cam.img_from_cam(c) * up
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        side = int(round((hi - lo).max() * (1 + 2 * REGION_MARGIN)))
        u, v = (lo + hi) / 2
        return int(round(u - side / 2)), int(round(v - side / 2)), side
    c = pose * X
    if c[2] <= 0:
        return None
    u, v = cam.img_from_cam(np.asarray([c]))[0] * up
    side = int(round(CROP_MM / mm_per_unit * cam.mean_focal_length() * up / c[2]))
    return int(round(u - side / 2)), int(round(v - side / 2)), side


def cut(photo: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """The square, black where it leaves the photo, resampled to SAM_PX (RGB)."""
    out = np.zeros((side, side, 3), np.uint8)
    h, w = photo.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = photo[sy0:sy1, sx0:sx1]
    out = cv2.resize(out, (SAM_PX, SAM_PX), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def largest(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def to_frame(mask: np.ndarray, x0: int, y0: int, side: int, up: float, shape):
    """A SAM_PX mask of the square -> (box, crop) in P2 frame pixels, or None."""
    n = max(1, int(round(side / up)))
    small = cv2.resize(mask.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA) > 0.5
    fx, fy = int(round(x0 / up)), int(round(y0 / up))
    h, w = shape
    full = np.zeros((h, w), bool)
    ax0, ay0, ax1, ay1 = max(fx, 0), max(fy, 0), min(fx + n, w), min(fy + n, h)
    if ax1 <= ax0 or ay1 <= ay0:
        return None
    full[ay0:ay1, ax0:ax1] = small[ay0 - fy:ay1 - fy, ax0 - fx:ax1 - fx]
    return full


def packed(tracks, frames, fulls):
    """cache_masks' npz layout from full-frame boolean masks."""
    boxes, shapes, blobs = [], [], []
    for m in fulls:
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        crop = m[y0:y1, x0:x1]
        boxes.append((x0, y0, x1, y1))
        shapes.append(crop.shape)
        blobs.append(np.packbits(crop.ravel()))
    offsets = np.cumsum([0] + [len(b) for b in blobs])
    return dict(track=np.array(tracks), frame=np.array(frames), box=np.array(boxes),
                shape=np.array(shapes), offsets=offsets, packed=np.concatenate(blobs))


def colour(i: int) -> tuple:
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[(i * 37) % 180, 230, 255]]]),
                                              cv2.COLOR_HSV2BGR)[0, 0])


def outline(img, mask, col, thick=3, label=None):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, col, thick)
    if label is not None and mask.any():
        ys, xs = np.nonzero(mask)
        cv2.putText(img, label, (int(xs.mean()) - 10, int(ys.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)


# --------------------------------------------------------------------------
# zoomed SAM3 and the merge
# --------------------------------------------------------------------------


def zoom(args, run: Run, skeleton: dict, mpu: float):
    import torch
    from pose_estimator.segmentation_sam3 import _run_session, load_sam3

    out = args.out
    (out / "sam3").mkdir(parents=True, exist_ok=True)
    p2 = dict(np.load(args.masks, allow_pickle=True))
    tips = apices(run, skeleton, args.regions)
    print(f"  {len(tips)} apices " + ("from the labelled regions: " if args.regions else
                                      "from the stem tree: ")
          + ", ".join(f"{a} ({k})" for a, k, _ in tips))
    frames = sorted(run.image_of)
    passes = sorted({int(run.sources[f]) for f in frames})

    # every square of every apex from one read of each photo
    t0 = time.time()
    squares, videos = {}, defaultdict(list)
    for frame in frames:
        photo = cv2.imread(str(source_image(run.workdir, frame)))
        for aid, _, X in tips:
            sq = square(run, frame, X, mpu, photo.shape[1], (args.regions or {}).get(f"apex_{aid}"))
            if sq is None:
                continue
            squares[aid, frame] = sq + (photo.shape[1] / run.rec.cameras[
                run.image_of[frame].camera_id].width,)
            videos[aid, int(run.sources[frame])].append((frame, cut(photo, *sq)))
    print(f"  cut {len(squares)} squares from {len(frames)} originals ({time.time() - t0:.0f}s)")

    processor, model = load_sam3()
    zoomed = defaultdict(list)          # frame -> [(apex, pass, obj, full-frame mask)]
    report = {"crop_mm": None if args.regions else CROP_MM,
              "regions": str(args.regions_path) if args.regions else None, "apices": {}}
    plant_box = plant_boxes(run, frames)
    for aid, kind, X in tips:
        rep = report["apices"][str(aid)] = {"kind": kind, "point_plant": (
            (X - run.origin) @ run.rotation.T).tolist(), "passes": {}}
        for p in passes:
            clip = videos.get((aid, p), [])
            if len(clip) < 3:
                continue
            names = [f for f, _ in clip]
            t0 = time.time()
            per_frame, _ = _run_session(processor, model, torch, [v for _, v in clip], ["leaf"],
                                        "cuda", label=f"apex {aid} pass {p}")
            kept_n, dropped = 0, defaultdict(int)
            store = {}
            for i, frame in enumerate(names):
                x0, y0, side, up = squares[aid, frame]
                store[f"crop_{i}"] = np.array([x0, y0, side], np.int64)
                store[f"up_{i}"] = np.float64(up)
                for obj, m in per_frame.get(i, {}).get("leaf", {}).items():
                    m = largest(m)
                    store[f"m_{i}_{obj}"] = np.packbits(m)
                    full, why = accept(m, x0, y0, side, up, plant_box[frame])
                    if full is None:
                        dropped[why] += 1
                        continue
                    zoomed[frame].append((aid, p, obj, full))
                    kept_n += 1
            np.savez_compressed(out / "sam3" / f"apex{aid}_pass{p}.npz", frames=np.array(names),
                                sam_px=SAM_PX, **store)
            ids = {o for i in per_frame for o in per_frame[i].get("leaf", {})}
            rep["passes"][str(p)] = {"frames": len(names), "sam3_ids": len(ids),
                                     "masks_kept": kept_n, "masks_dropped": dict(dropped),
                                     "seconds": round(time.time() - t0, 1)}
            print(f"    {len(ids)} ids, {kept_n} masks kept as apex leaves, dropped "
                  + ", ".join(f"{v} {k}" for k, v in dropped.items()))
            montage(out / "sam3" / f"apex{aid}_pass{p}.jpg", run, p2, clip, per_frame,
                    [squares[aid, f] for f in names], zoomed, aid, p)
    del model
    torch.cuda.empty_cache()

    merge(out, p2, zoomed, frames, plant_box, report)



def merge(out: Path, p2, zoomed, frames, plant_box, report: dict):
    """masks_zoom.npz: the zoomed apex leaves, then P2's masks less what they claim."""
    tracks, frames_out, fulls = [], [], []
    added = trimmed = dropped_p2 = dup = 0
    by_frame = defaultdict(list)
    for k, f in enumerate(p2["frame"]):
        by_frame[str(f)].append(k)
    for frame in frames:
        h, w = plant_box[frame][0]
        claimed = np.zeros((h, w), bool)
        kept = []
        for aid, p, obj, full in zoomed.get(frame, []):
            if any((full & k).sum() > DUPLICATE_IOU * (full | k).sum() for k in kept):
                dup += 1
                continue
            kept.append(full)
            claimed |= full
            tracks.append(f"z{aid}p{p}_{obj}")
            frames_out.append(frame)
            fulls.append(full)
            added += 1
        for k in by_frame.get(frame, []):
            x0, y0, x1, y1 = p2["box"][k]
            m = np.zeros((h, w), bool)
            m[y0:y1, x0:x1] = unpack(p2, k)
            left = m & ~claimed
            if left.sum() == m.sum():
                keep = m
            else:
                keep = largest(left) if left.any() else left
                if keep.sum() < max(30, KEEP_SHARE * m.sum()):
                    dropped_p2 += 1
                    continue
                trimmed += 1
            tracks.append(str(p2["track"][k]))
            frames_out.append(frame)
            fulls.append(keep)
    np.savez_compressed(out / "masks_zoom.npz", **packed(tracks, frames_out, fulls))
    report["merge"] = {"p2_masks": int(len(p2["track"])), "zoomed_added": added,
                       "zoomed_duplicates_skipped": dup, "p2_trimmed": trimmed,
                       "p2_dropped": dropped_p2, "masks_out": len(tracks)}
    (out / "apex_zoom.json").write_text(json.dumps(report, indent=1))
    print(f"\n  merge: {added} zoomed apex-leaf masks added ({dup} duplicates of an overlapping "
          f"apex skipped); of {len(p2['track'])} P2 masks {trimmed} trimmed and {dropped_p2} "
          f"dropped -> {out / 'masks_zoom.npz'} ({len(tracks)} masks)")

def plant_boxes(run: Run, frames):
    """frame -> ((h, w), (x0, y0, x1, y1)): the P2 plant mask's bounding box. skeleton_2d
    measures every leaf mask inside it, so a zoomed mask must not leave it."""
    out = {}
    for f in frames:
        plant = cv2.imread(str(run.workdir / "p2" / "masks" / "plant" / f"{f}.png"), 0) > 127
        ys, xs = np.nonzero(plant)
        out[f] = (plant.shape, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    return out


def accept(m: np.ndarray, x0: int, y0: int, side: int, up: float, box):
    """(full-frame mask, None) for an apex leaf, or (None, why not)."""
    if m.sum() < MIN_AREA:
        return None, "speck"
    ys, xs = np.nonzero(m)
    if (xs.min() < EDGE_PX or ys.min() < EDGE_PX or xs.max() >= SAM_PX - EDGE_PX
            or ys.max() >= SAM_PX - EDGE_PX):
        return None, "leaves the square"
    shape, (bx0, by0, bx1, by1) = box
    full = to_frame(m, x0, y0, side, up, shape)
    if full is not None:
        inside = np.zeros(shape, bool)
        inside[by0:by1, bx0:bx1] = True
        full &= inside
    if full is None or full.sum() < 30:
        return None, "outside the frame"
    return full, None


def photo_scale(run: Run, frame: str) -> float:
    """Original photo width over the P3 camera's (3.125 for 6000 px D5600 originals)."""
    photo = cv2.imread(str(source_image(run.workdir, frame)), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    return photo.shape[1] * 2 / run.rec.cameras[run.image_of[frame].camera_id].width


def merge_only(args, run: Run, skeleton: dict, mpu: float):
    """Redo the merge from sam3/*.npz -- new thresholds without running SAM3 again."""
    out = args.out
    p2 = dict(np.load(args.masks, allow_pickle=True))
    report = json.loads((out / "apex_zoom.json").read_text())
    frames = sorted(run.image_of)
    plant_box = plant_boxes(run, frames)
    zoomed = defaultdict(list)
    scale = {}                          # frame -> photo_scale, for files written without up_<i>
    for aid, _, _ in apices(run, skeleton, args.regions):
        for path in sorted((out / "sam3").glob(f"apex{aid}_pass*.npz")):
            p = int(path.stem.split("pass")[1])
            z = np.load(path)
            names = [str(f) for f in z["frames"]]
            kept_n, dropped = 0, defaultdict(int)
            for key in z.files:
                if not key.startswith("m_"):
                    continue
                _, i, obj = key.split("_")
                frame = names[int(i)]
                x0, y0, side = (int(v) for v in z[f"crop_{i}"])
                if f"up_{i}" in z.files:
                    up = float(z[f"up_{i}"])
                else:
                    if frame not in scale:
                        scale[frame] = photo_scale(run, frame)
                    up = scale[frame]
                m = np.unpackbits(z[key])[:SAM_PX * SAM_PX].reshape(SAM_PX, SAM_PX).astype(bool)
                full, why = accept(m, x0, y0, side, up, plant_box[frame])
                if full is None:
                    dropped[why] += 1
                    continue
                zoomed[frame].append((aid, p, int(obj), full))
                kept_n += 1
            report["apices"][str(aid)]["passes"][str(p)].update(
                masks_kept=kept_n, masks_dropped=dict(dropped))
            print(f"  apex {aid} pass {p}: {kept_n} kept, dropped "
                  + ", ".join(f"{v} {k}" for k, v in dropped.items()))
    merge(out, p2, zoomed, frames, plant_box, report)


def montage(path: Path, run: Run, p2, clip, per_frame, squares, zoomed, aid, p, n=8):
    tiles = []
    for i in np.linspace(0, len(clip) - 1, min(n, len(clip))).round().astype(int):
        frame, rgb = clip[i]
        x0, y0, side, up = squares[i]
        photo = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        old, new = photo.copy(), photo.copy()
        cam = run.rec.cameras[run.image_of[frame].camera_id]
        for k in np.nonzero(p2["frame"] == frame)[0]:
            bx0, by0, bx1, by1 = p2["box"][k]
            m = np.zeros((cam.height, cam.width), np.uint8)
            m[by0:by1, bx0:bx1] = unpack(p2, k) * 255
            # frame -> this square at SAM_PX
            s = SAM_PX / side * up
            M = np.float32([[s, 0, -x0 / side * SAM_PX], [0, s, -y0 / side * SAM_PX]])
            sq = cv2.warpAffine(m, M, (SAM_PX, SAM_PX), flags=cv2.INTER_LINEAR) > 127
            if sq.any():
                outline(old, sq, colour(hash(str(p2["track"][k])) % 97))
        kept = {obj for a, q, obj, _ in zoomed.get(frame, []) if a == aid and q == p}
        for obj, m in per_frame.get(i, {}).get("leaf", {}).items():
            if obj in kept:
                outline(new, m, colour(obj), 3, str(obj))
            else:
                outline(new, m, (120, 120, 120), 1)
        t = np.hstack([cv2.resize(x, (480, 480)) for x in (photo, old, new)])
        cv2.putText(t, f"{frame}  photo | P2 masks | zoomed SAM3 (kept coloured)", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(t)
    cv2.imwrite(str(path), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])


# --------------------------------------------------------------------------
# two s2d skeletons on the apex squares
# --------------------------------------------------------------------------


def compare(args, run: Run, skeleton: dict, mpu: float):
    (args.out / "compare").mkdir(parents=True, exist_ok=True)
    sk = [json.loads((d / "skeleton.json").read_text()) for d in args.compare]
    names = [d.name for d in args.compare]
    frames = sorted(run.image_of)
    for aid, kind, X in apices(run, skeleton, args.regions):
        rows = []
        for frame in [frames[i] for i in np.linspace(0, len(frames) - 1, 8).round().astype(int)]:
            im = run.image_of[frame]
            cam = run.rec.cameras[im.camera_id]
            photo = cv2.imread(str(source_image(run.workdir, frame)))
            sq = square(run, frame, X, mpu, photo.shape[1], (args.regions or {}).get(f"apex_{aid}"))
            if sq is None:
                continue
            x0, y0, side = sq
            up = photo.shape[1] / cam.width
            base = cv2.cvtColor(cut(photo, x0, y0, side), cv2.COLOR_RGB2BGR)
            tiles = [base]
            R, t = im.cam_from_world().rotation.matrix(), np.asarray(im.cam_from_world().translation)

            def uv(P):
                c = run.to_world(P) @ R.T + t
                return ((cam.img_from_cam(c) * up - [x0, y0]) * SAM_PX / side).astype(np.int32)

            for s in sk:
                img = base.copy()
                for a in s.get("axes") or []:
                    if a["kind"] != "petiole" and len(a["points"]) >= 2:
                        cv2.polylines(img, [uv(a["points"])], False, (255, 255, 255), 4)
                for leaf in s["leaves"]:
                    col = tuple(int(c) for c in leaf.get("colour_bgr") or track_colour(int(leaf["id"])))
                    if len(leaf.get("curve") or []) >= 2:       # one curve, stem to tip
                        cv2.polylines(img, [uv(leaf["curve"])], False, col, 5)
                    else:                                       # petiole (light green, as in Blender)
                        if len(leaf["petiole"]) >= 2:
                            cv2.polylines(img, [uv(leaf["petiole"])], False, (128, 242, 158), 3)
                        cv2.polylines(img, [uv(leaf["midrib"])], False, col, 5)
                    cv2.circle(img, tuple(int(v) for v in uv([leaf["tip"]])[0]), 9, col, -1)
                tiles.append(img)
            row = np.hstack([cv2.resize(x, (480, 480)) for x in tiles])
            cv2.putText(row, f"{frame}  photo | " + " | ".join(names), (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            rows.append(row)
        if rows:
            cv2.imwrite(str(args.out / "compare" / f"apex{tag(aid)}_{kind}.jpg"), np.vstack(rows),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
    summary = apex_table(args, run, skeleton, sk, names, mpu)
    (args.out / "compare" / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"  wrote {args.out / 'compare'}")


def apex_table(args, run: Run, skeleton: dict, sk, names, mpu: float) -> dict:
    """Per skeleton: its scores, and per apex the leaves whose midrib middle lies within
    its reach (CROP_MM / 2, or a labelled region's 95th-percentile radius)."""
    out = {}
    tips, reach = {}, {}
    for aid, kind, X in apices(run, skeleton, args.regions):
        tips[aid] = (kind, (X - run.origin) @ run.rotation.T)
        pts = (args.regions or {}).get(f"apex_{aid}")
        reach[aid] = (float(np.percentile(np.linalg.norm(pts - X, axis=1), 95)) * mpu
                      if pts is not None else CROP_MM / 2)
    print("\n  leaves with their midrib middle within reach of each apex "
          "(n: blade lengths mm / median fit px)")
    print("  " + "apex".ljust(12) + "".join(n[:28].ljust(30) for n in names))
    for name, s, d in zip(names, sk, args.compare):
        f = d / "scores.json"                     # s2d writes one; P5x and the others do not
        scores = json.loads(f.read_text()) if f.exists() else {}
        out[name] = {"leaves": len(s["leaves"]), "failed_ids": len(scores.get("failed", {})),
                     "held_out_px": scores.get("held_out_px", float("nan")), "apices": {}}
        for aid, (kind, P) in tips.items():
            near = [l for l in s["leaves"]
                    if np.linalg.norm(np.asarray(l["midrib"])[len(l["midrib"]) // 2] - P) * mpu
                    < reach[aid]]
            out[name]["apices"][str(aid)] = {
                "kind": kind, "leaves": len(near),
                "midrib_mm": sorted(round(float(np.linalg.norm(np.diff(np.asarray(l["midrib"]), axis=0),
                                                              axis=1).sum()) * mpu, 1) for l in near),
                "median_fit_px": (float(np.median([l["fit_px"] for l in near]))
                                  if near and all("fit_px" in l for l in near) else None)}
    for aid, (kind, _) in tips.items():
        cells = []
        for name in names:
            a = out[name]["apices"][str(aid)]
            mm = ",".join(f"{v:.0f}" for v in a["midrib_mm"])
            px = f"{a['median_fit_px']:.1f}" if a["median_fit_px"] is not None else "-"
            cells.append(f"{a['leaves']}: [{mm}] / {px}"[:28].ljust(30))
        print("  " + f"{aid} {kind}".ljust(12) + "".join(cells))
    print("  " + "whole plant".ljust(12) + "".join(
        f"{out[n]['leaves']} leaves, {out[n]['failed_ids']} failed, {out[n]['held_out_px']:.1f} px"
        [:28].ljust(30) for n in names))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--skeleton", type=Path,
                    help="s2d skeleton.json with the stem tree (default: <workdir>/skeleton.json)")
    ap.add_argument("--masks", type=Path,
                    help="P2 leaf masks in cache_masks' format (default: <workdir>/masks.npz)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path, nargs="+", metavar="S2D_DIR",
                    help="draw these s2d outputs on the apex squares, and tabulate their apex "
                         "leaves, instead of running SAM3")
    ap.add_argument("--regions", type=Path,
                    help="apex_labels.py's apex_regions.npz: crop each labelled apex round its own "
                         "points instead of a fixed square on a stem-tree branch end")
    ap.add_argument("--merge-only", action="store_true",
                    help="redo masks_zoom.npz from the zoomed masks already in --out/sam3")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.masks = args.masks or args.workdir / "masks.npz"
    args.regions_path = args.regions
    args.regions = dict(np.load(args.regions)) if args.regions else None
    skeleton = json.loads((args.skeleton or args.workdir / "skeleton.json").read_text())
    mpu = json.loads((args.workdir / "tag_scale.json").read_text())["mm_per_unit"]
    run = Run(args.workdir)
    (compare if args.compare else merge_only if args.merge_only else zoom)(args, run, skeleton, mpu)


if __name__ == "__main__":
    main()
