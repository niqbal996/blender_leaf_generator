#!/usr/bin/env python
"""2D next to 3D, per view, one colour per leaf in both: where did each SAM3 leaf go?

    python scripts/p5x_views.py --workdir <dataset>/plant --out runs/<specimen>_p5x_views

Diagnostic only -- reads a finished P5x and writes pictures into --out.

For each chosen view, three panels on the same crop:

  2D     SAM3's leaf masks for that frame, each filled with the colour of the 3D
         leaf it ended up in. A 2D id whose pixels reach no 3D leaf is drawn
         grey with a red outline and marked "lost" -- a leaf SAM3 saw and P5x
         did not keep.
  3D     the P5x cloud rendered from that camera in the same colours, with the
         skeleton (midrib, petiole, tip, crown) projected on.
  photo  the photograph with the same skeleton drawn over it, so a 3D tip can
         be checked against where the real tip is.

Colours are P5x's own `leaf_palette`, indexed exactly as `leaves.ply` is, so the
pictures match the Blender scene.

"Ended up in" is measured, not looked up: every view's z-buffered render says
which 3D point sits under each mask pixel, and a 2D id's leaf is the one its
pixels land on most, over all its frames. That is the 2D-to-3D association as
it actually came out, which is the thing being inspected.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pose_estimator import cloud_source
from pose_estimator.cli.leaf_instances import ROOT, SKELETON, UNSEEN, leaf_palette
from pose_estimator.ply_io import read_ply_vertices
from pose_estimator.semantic import camera_from_colmap, render_points

LOST_SHARE = 0.15       # under this share of its pixels on its best leaf, a 2D id is "lost"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


class Run:
    def __init__(self, workdir: Path):
        import pycolmap

        self.workdir = workdir
        chosen = cloud_source.resolve(workdir, cloud_source.BASELINE, None, "auto")
        f = read_ply_vertices(chosen.path)
        self.points = np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float64)
        self.assignment = np.load(workdir / "p5x" / "instances.npy")
        if len(self.assignment) != len(self.points):
            raise SystemExit(f"p5x/instances.npy has {len(self.assignment)} rows but "
                             f"{chosen.path} has {len(self.points)} -- P5x is stale")
        report = json.loads((workdir / "p5x" / "instances.json").read_text())
        self.palette = leaf_palette(max(report["num_leaves_2d"], 1))
        self.skeleton = json.loads((workdir / "p5x" / "skeleton.json").read_text())
        graph = json.loads((workdir / "p5" / "stem_graph.json").read_text())["plant_frame"]
        self.origin = np.asarray(graph["origin"], float)
        self.rotation = np.asarray(graph["rotation"], float)
        sparse, _ = cloud_source.geometry(workdir, cloud_source.BASELINE)
        self.rec = pycolmap.Reconstruction(str(sparse))
        self.image_of = {Path(im.name).stem: im for im in self.rec.images.values()}
        self.sources = json.loads((workdir / "p1" / "sources.json").read_text())
        self.voxel = float(json.loads((workdir / "p4" / "hull.json").read_text())["voxel_size"])

    def to_world(self, plant_xyz) -> np.ndarray:
        """P5x writes the skeleton in its upright plant frame; cameras live in the world."""
        return np.asarray(plant_xyz, float) @ self.rotation + self.origin

    def camera(self, frame: str):
        im = self.image_of[frame]
        return camera_from_colmap(im, self.rec.cameras[im.camera_id])

    def colour(self, leaf: int) -> np.ndarray:
        """RGB of a 3D leaf id, exactly as leaves.ply has it."""
        return self.palette[leaf % len(self.palette)]


def cache_masks(workdir: Path, out: Path):
    """Every leaf-instance mask once, as a full-resolution crop to its bbox."""
    cache = out / "masks.npz"
    if cache.exists() and cache.stat().st_mtime > (workdir / "p2" / "prompts.json").stat().st_mtime:
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    root = workdir / "p2" / "masks" / "leaf_instances"
    tracks, frames, boxes, shapes, blobs = [], [], [], [], []
    t0 = time.time()
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for png in sorted(d.glob("*.png")):
            m = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            if m is None or not (m > 127).any():
                continue
            b = m > 127
            ys, xs = np.nonzero(b)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            crop = b[y0:y1, x0:x1]
            tracks.append(d.name)
            frames.append(png.stem)
            boxes.append((x0, y0, x1, y1))
            shapes.append(crop.shape)
            blobs.append(np.packbits(crop.ravel()))
    offsets = np.cumsum([0] + [len(b) for b in blobs])
    data = dict(track=np.array(tracks), frame=np.array(frames), box=np.array(boxes),
                shape=np.array(shapes), offsets=offsets, packed=np.concatenate(blobs))
    np.savez_compressed(cache, **data)
    print(f"  cached {len(tracks)} masks from {len(set(tracks))} ids ({time.time() - t0:.0f}s)")
    return data


def unpack(masks, k: int) -> np.ndarray:
    h, w = masks["shape"][k]
    raw = masks["packed"][masks["offsets"][k]:masks["offsets"][k + 1]]
    return np.unpackbits(raw)[: h * w].reshape(h, w).astype(bool)


# --------------------------------------------------------------------------
# 2D -> 3D association, as it came out
# --------------------------------------------------------------------------


def track_to_leaf(run: Run, masks, out: Path) -> dict:
    """{2D id: (3D leaf or None, share of its pixels on that leaf, share on skeleton)}."""
    cache = out / "track_to_leaf.json"
    if cache.exists() and cache.stat().st_mtime > (run.workdir / "p5x" / "instances.npy").stat().st_mtime:
        return json.loads(cache.read_text())
    by_frame = defaultdict(list)
    for k, f in enumerate(masks["frame"]):
        by_frame[str(f)].append(k)
    tally = defaultdict(lambda: defaultdict(int))
    colours = np.zeros((len(run.points), 3), np.uint8)
    for frame, ks in sorted(by_frame.items()):
        if frame not in run.image_of:
            continue
        _rgb, index = render_points(run.points, colours, run.camera(frame))
        for k in ks:
            x0, y0, x1, y1 = masks["box"][k]
            under = index[y0:y1, x0:x1][unpack(masks, k)]
            under = run.assignment[under[under >= 0]]
            labels, counts = np.unique(under, return_counts=True)
            for lab, n in zip(labels, counts):
                tally[str(masks["track"][k])][int(lab)] += int(n)
    result = {}
    for track, t in tally.items():
        total = sum(t.values())
        leaves = {lab: n for lab, n in t.items() if lab >= 0}
        best = max(leaves, key=leaves.get) if leaves else None
        share = leaves[best] / total if best is not None else 0.0
        result[track] = {"leaf": best if share >= LOST_SHARE else None,
                         "best_leaf": best, "share": round(share, 3),
                         "skeleton_share": round(t.get(SKELETON, 0) / total, 3),
                         "pixels": total}
    cache.write_text(json.dumps(result, indent=1))
    return result


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------


def _bgr(rgb) -> tuple:
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def project(camera, xyz: np.ndarray):
    R, t = camera.world_to_camera[:3, :3], camera.world_to_camera[:3, 3]
    cam = np.atleast_2d(xyz) @ R.T + t
    uv = (cam[:, :2] / np.maximum(cam[:, 2:3], 1e-9)) @ camera.K[:2, :2].T + camera.K[:2, 2]
    return uv, cam[:, 2]


def depth_buffer(run: Run, camera, index: np.ndarray) -> np.ndarray:
    """Per-pixel depth of the point the render put there, inf where empty."""
    R, t = camera.world_to_camera[:3, :3], camera.world_to_camera[:3, 3]
    depth = (run.points @ R.T + t)[:, 2]
    out = np.full(index.shape, np.inf)
    hit = index >= 0
    out[hit] = depth[index[hit]]
    # Close the gaps between splats so a skeleton behind a leaf reads as hidden.
    return cv2.erode(out.astype(np.float32), np.ones((7, 7), np.uint8))


def visible(zbuf: np.ndarray, uv: np.ndarray, depth: np.ndarray, tol: float) -> np.ndarray:
    h, w = zbuf.shape
    x = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    return depth <= zbuf[y, x] + tol


def draw_skeleton(img, run: Run, camera, zbuf, extra_tips=None):
    """Midribs in the leaf colour, petioles white, tips as rings; hidden parts thin."""
    tol = 3 * run.voxel
    for leaf in run.skeleton["leaves"]:
        col = _bgr(run.colour(leaf["id"]))
        for key, colour, width in (("petiole", (235, 235, 235), 2), ("midrib", col, 3)):
            line = run.to_world(leaf[key]) if len(leaf[key]) >= 2 else None
            if line is None:
                continue
            uv, d = project(camera, line)
            vis = visible(zbuf, uv, d, tol)
            for i in range(len(uv) - 1):
                p, q = tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[i + 1]).astype(int))
                if vis[i] and vis[i + 1]:
                    cv2.line(img, p, q, (0, 0, 0), width + 2, cv2.LINE_AA)
                    cv2.line(img, p, q, colour, width, cv2.LINE_AA)
                else:
                    cv2.line(img, p, q, colour, 1, cv2.LINE_AA)
        uv, d = project(camera, run.to_world(leaf["tip"]))
        p = tuple(np.round(uv[0]).astype(int))
        shown = visible(zbuf, uv, d, tol)[0]
        cv2.circle(img, p, 9, (255, 255, 255), 2 if shown else 1, cv2.LINE_AA)
        cv2.circle(img, p, 6, col, -1 if shown else 2, cv2.LINE_AA)
        cv2.putText(img, str(leaf["id"]), (p[0] + 10, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, str(leaf["id"]), (p[0] + 10, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1, cv2.LINE_AA)
    if run.skeleton.get("crown") is not None:
        uv, _ = project(camera, run.to_world(run.skeleton["crown"]))
        cv2.drawMarker(img, tuple(np.round(uv[0]).astype(int)), (0, 255, 255),
                       cv2.MARKER_STAR, 22, 2, cv2.LINE_AA)
    for (uv, colour) in (extra_tips or []):
        cv2.drawMarker(img, tuple(np.round(uv).astype(int)), colour, cv2.MARKER_TILTED_CROSS,
                       14, 2, cv2.LINE_AA)


def panel_2d(run: Run, masks, frame: str, photo: np.ndarray, assoc: dict,
             tips2d=None) -> np.ndarray:
    img = (photo * 0.45).astype(np.uint8)
    ks = [k for k, f in enumerate(masks["frame"]) if f == frame]
    labels = []
    for k in ks:
        x0, y0, x1, y1 = masks["box"][k]
        m = np.zeros(photo.shape[:2], bool)
        m[y0:y1, x0:x1] = unpack(masks, k)
        info = assoc.get(str(masks["track"][k]), {})
        leaf = info.get("leaf")
        if leaf is None:
            img[m] = (0.5 * img[m] + 0.5 * np.array((128, 128, 128))).astype(np.uint8)
            outline, text = (0, 0, 255), "lost"
        else:
            col = np.array(_bgr(run.colour(leaf)))
            img[m] = (0.3 * img[m] + 0.7 * col).astype(np.uint8)
            outline, text = (255, 255, 255), str(leaf)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, outline, 1, cv2.LINE_AA)
        ys, xs = np.nonzero(m)
        labels.append(((int(xs.mean()), int(ys.mean())), text, outline))
    for row in (tips2d or {}).get(frame, []):
        leaf = assoc.get(row["track"], {}).get("leaf")
        col = _bgr(run.colour(leaf)) if leaf is not None else (0, 0, 255)
        tip = tuple(int(v) for v in row["tip"])
        base = tuple(int(v) for v in row["base"])
        cv2.drawMarker(img, tip, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 15, 4, cv2.LINE_AA)
        cv2.drawMarker(img, tip, col, cv2.MARKER_TILTED_CROSS, 13, 2, cv2.LINE_AA)
        cv2.rectangle(img, (base[0] - 3, base[1] - 3), (base[0] + 3, base[1] + 3), (255, 255, 255), 1)
    for (x, y), text, colour in labels:
        cv2.putText(img, text, (x - 8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(img, text, (x - 8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
                    cv2.LINE_AA)
    return img


def cloud_colours(run: Run) -> np.ndarray:
    rgb = np.full((len(run.points), 3), 60, np.uint8)
    rgb[run.assignment == SKELETON] = (175, 175, 175)
    rgb[run.assignment == ROOT] = (240, 140, 40)
    leaf = run.assignment >= 0
    rgb[leaf] = run.palette[run.assignment[leaf] % len(run.palette)]
    return rgb


def crop_box(workdir: Path, frame: str, pad: int = 40):
    plant = cv2.imread(str(workdir / "p2" / "masks" / "plant" / f"{frame}.png"),
                       cv2.IMREAD_GRAYSCALE) > 127
    ys, xs = np.nonzero(plant)
    h, w = plant.shape
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(w, xs.max() + pad), min(h, ys.max() + pad))


def render_view(run: Run, masks, assoc: dict, frame: str, extra=None,
                tips2d=None, tips3d=None) -> np.ndarray:
    camera = run.camera(frame)
    photo = cv2.imread(str(run.workdir / "p1" / "frames" / f"{frame}.jpg"))
    rgb, index = render_points(run.points, cloud_colours(run), camera, splat_radius=3)
    zbuf = depth_buffer(run, camera, index)

    a = panel_2d(run, masks, frame, photo, assoc, tips2d)
    b = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    draw_skeleton(b, run, camera, zbuf)
    c = (photo * 0.6).astype(np.uint8)
    draw_skeleton(c, run, camera, zbuf, extra_tips=extra)
    im = run.image_of[frame]
    pose, cam = im.cam_from_world(), run.rec.cameras[im.camera_id]
    for leaf, xyz in (tips3d or {}).items():
        X = pose.rotation.matrix() @ np.asarray(xyz) + np.asarray(pose.translation)
        if X[2] <= 0:
            continue
        uv = tuple(int(v) for v in np.round(cam.img_from_cam(X[None])[0]))
        cv2.drawMarker(c, uv, (255, 255, 255), cv2.MARKER_DIAMOND, 16, 4, cv2.LINE_AA)
        cv2.drawMarker(c, uv, _bgr(run.colour(leaf)), cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)

    x0, y0, x1, y1 = crop_box(run.workdir, frame)
    tiles = [t[y0:y1, x0:x1].copy() for t in (a, b, c)]
    names = ["2D: SAM3 ids by 3D leaf; x = 2D tip", "3D: P5x leaves + skeleton",
             "photo: o P5x tip, <> 2D-triangulated tip"]
    p = run.sources.get(frame, "?")
    for t, name in zip(tiles, names):
        cv2.rectangle(t, (0, 0), (t.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(t, f"{frame} (pass {p})  {name}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack(tiles)


def pick_frames(run: Run, per_pass: int):
    by_pass = defaultdict(list)
    for frame, p in sorted(run.sources.items()):
        if frame in run.image_of:
            by_pass[p].append(frame)
    out = []
    for p, frames in sorted(by_pass.items()):
        idx = np.linspace(0, len(frames) - 1, per_pass + 2)[1:-1].round().astype(int)
        out += [frames[i] for i in idx]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-pass", type=int, default=3, help="views per capture pass")
    ap.add_argument("--frames", nargs="*", help="explicit frame stems instead")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    run = Run(args.workdir)
    masks = cache_masks(args.workdir, args.out)
    assoc = track_to_leaf(run, masks, args.out)

    leaves3d = sorted({int(v) for v in run.assignment if v >= 0})
    reached = defaultdict(list)
    for track, info in assoc.items():
        if info["leaf"] is not None:
            reached[info["leaf"]].append(track)
    lost = sorted(t for t, i in assoc.items() if i["leaf"] is None)
    print(f"  {len(assoc)} 2D ids: {len(assoc) - len(lost)} land on a 3D leaf, {len(lost)} lost "
          f"(< {LOST_SHARE:.0%} of their pixels on any leaf)")
    print(f"  {len(leaves3d)} 3D leaves; ids per leaf: "
          + ", ".join(f"{leaf}:{len(reached.get(leaf, []))}" for leaf in leaves3d))
    print(f"  3D leaves no 2D id lands on: {[l for l in leaves3d if l not in reached]}")

    tips2d, tips3d = defaultdict(list), {}
    if (args.out / "tips2d.json").exists():
        for row in json.loads((args.out / "tips2d.json").read_text()):
            tips2d[row["frame"]].append(row)
    if (args.out / "tips3d.json").exists():
        tips3d = {r["leaf"]: r["tip"]["xyz"] for r in json.loads((args.out / "tips3d.json").read_text())
                  if r.get("tip")}
    frames = args.frames or pick_frames(run, args.per_pass)
    for frame in frames:
        img = render_view(run, masks, assoc, frame, tips2d=tips2d, tips3d=tips3d)
        cv2.imwrite(str(args.out / f"view_{frame}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  wrote view_{frame}.jpg")


if __name__ == "__main__":
    main()
