#!/usr/bin/env python
"""Where do the leaves go? A per-stage funnel from P2's 2D ids to P5x's 3D leaves.

    python scripts/leafcount_funnel.py --workdir <dataset>/plant --out runs/<specimen>_leafcount

Diagnostic only: it reads a finished run and writes numbers and pictures into
--out. Nothing in the workdir is touched.

Stage 1 (2D): every leaf-instance mask P2 wrote is read once and summarised
per (track, frame) -- area, bbox, centroid -- and cached as `tracks2d.npz` so
the later stages do not re-read ~4000 PNGs from a slow mount.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def stage1_tracks(workdir: Path, out: Path, scale: int = 2) -> dict:
    """Per-track, per-frame mask stats, plus a downsampled label stack per frame.

    The label stack keeps every instance separately (a frame can have
    overlapping instances), stored as a list of (track, frame, packed mask at
    1/scale resolution) so co-existence and overlap can be measured later.
    """
    root = workdir / "p2" / "masks" / "leaf_instances"
    cache = out / "tracks2d.npz"
    if cache.exists():
        print(f"  stage 1: cached {cache}")
        return dict(np.load(cache, allow_pickle=True))

    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    rows, packed = [], []
    t0 = time.time()
    shape_small = None
    for k, d in enumerate(dirs):
        for png in sorted(d.glob("*.png")):
            m = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            b = m > 127
            area = int(b.sum())
            if area == 0:
                continue
            ys, xs = np.nonzero(b)
            n_comp = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)[0] - 1
            small = cv2.resize(b.astype(np.uint8), (b.shape[1] // scale, b.shape[0] // scale),
                               interpolation=cv2.INTER_AREA) > 0
            shape_small = small.shape
            rows.append((d.name, png.stem, area, xs.min(), ys.min(), xs.max(), ys.max(),
                         xs.mean(), ys.mean(), n_comp))
            packed.append(np.packbits(small.ravel()))
        if k % 10 == 0:
            print(f"    read {k + 1}/{len(dirs)} tracks, {len(rows)} masks, {time.time() - t0:.0f}s")

    names = np.array([r[0] for r in rows])
    frames = np.array([r[1] for r in rows])
    stats = np.array([r[2:] for r in rows], dtype=np.float64)
    np.savez_compressed(cache, track=names, frame=frames, stats=stats,
                        packed=np.stack(packed), shape_small=np.array(shape_small),
                        scale=np.array(scale))
    print(f"  stage 1: {len(dirs)} tracks, {len(rows)} masks -> {cache} ({time.time() - t0:.0f}s)")
    return dict(np.load(cache, allow_pickle=True))


def stage2_evidence(workdir: Path, out: Path) -> Path:
    """Which cloud points sit under which 2D mask, in every view -- before any painting order.

    P5x collapses each view to one label per pixel (later instance folders
    overwrite earlier ones, then stem and root overwrite all) and then to one
    label per point. Both collapses throw away exactly the evidence needed to
    ask where a leaf was lost, so this keeps the raw relation instead:

        per view: the pixels the render covers on the plant, the point that
        drew each, its normal weight, and for every instance mask / stem /
        root, which of those pixels it claims.

    Saved as evidence.npz; every P5x variant (Run A/B/C) is recomputed from it
    in seconds.
    """
    import pycolmap
    from pose_estimator import cloud_source
    from pose_estimator.ply_io import read_ply_vertices
    from pose_estimator.semantic import camera_from_colmap, render_points, view_weights

    cache = out / "evidence.npz"
    if cache.exists():
        print(f"  stage 2: cached {cache}")
        return cache

    p2 = workdir / "p2" / "masks"
    chosen = cloud_source.resolve(workdir, cloud_source.BASELINE, None, "auto")
    fields = read_ply_vertices(chosen.path)
    points = np.stack([fields["x"], fields["y"], fields["z"]], 1).astype(np.float64)
    normals = np.stack([fields["nx"], fields["ny"], fields["nz"]], 1).astype(np.float64)
    sparse, _ = cloud_source.geometry(workdir, cloud_source.BASELINE)
    rec = pycolmap.Reconstruction(str(sparse))
    print(f"  stage 2: {len(points)} points from {chosen.path}, {rec.num_reg_images()} views")

    tracks = sorted(d.name for d in (p2 / "leaf_instances").iterdir() if d.is_dir())
    track_index = {t: i for i, t in enumerate(tracks)}

    view_names, pix_view, pix_point, pix_w, pix_flat = [], [], [], [], []
    claim_pix, claim_label = [], []          # label: track index, or -2 stem, -3 root
    offset = 0
    t0 = time.time()
    for k, image_id in enumerate(sorted(rec.reg_image_ids())):
        image = rec.images[image_id]
        stem_name = Path(image.name).stem
        camera = camera_from_colmap(image, rec.cameras[image.camera_id])
        _rgb, index_map = render_points(points, np.zeros((len(points), 3), np.uint8), camera)
        weights = view_weights(points, normals, camera)

        flat = np.flatnonzero(index_map.ravel() >= 0)
        pts = index_map.ravel()[flat]
        lookup = np.full(index_map.size, -1, np.int64)
        lookup[flat] = np.arange(len(flat)) + offset

        def claim(path: Path, label: int):
            if not path.exists():
                return
            m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return
            hit = lookup[np.flatnonzero(m.ravel() > 127)]
            hit = hit[hit >= 0]
            claim_pix.append(hit)
            claim_label.append(np.full(len(hit), label, np.int16))

        for t in tracks:
            claim(p2 / "leaf_instances" / t / f"{stem_name}.png", track_index[t])
        claim(p2 / "stem" / f"{stem_name}.png", -2)
        claim(p2 / "root" / f"{stem_name}.png", -3)

        view_names.append(stem_name)
        pix_view.append(np.full(len(flat), k, np.int16))
        pix_point.append(pts.astype(np.int32))
        pix_w.append(weights[pts].astype(np.float32))
        pix_flat.append(flat.astype(np.int32))
        offset += len(flat)
        if k % 15 == 0:
            print(f"    view {k + 1}: {offset} rendered pixels, {time.time() - t0:.0f}s")

    np.savez_compressed(
        cache, points=points, normals=normals, tracks=np.array(tracks),
        views=np.array(view_names), pix_view=np.concatenate(pix_view),
        pix_point=np.concatenate(pix_point), pix_w=np.concatenate(pix_w),
        pix_flat=np.concatenate(pix_flat),
        claim_pix=np.concatenate(claim_pix), claim_label=np.concatenate(claim_label))
    print(f"  stage 2: {offset} rendered pixels, {sum(map(len, claim_pix))} claims "
          f"-> {cache} ({time.time() - t0:.0f}s)")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stage1_tracks(args.workdir, args.out)
    stage2_evidence(args.workdir, args.out)


if __name__ == "__main__":
    main()
