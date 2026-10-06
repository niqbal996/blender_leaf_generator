#!/usr/bin/env python
"""mm per reconstruction unit, from the AprilTags in the turntable scene.

    python scripts/tag_scale.py --workdir <ds>/plant --tag-mm 8

Structure from motion has no scale, and every P3 solve has its own. The holder
carries printed AprilTag 36h11 stickers -- the same sheet as the flat lay's
markers (sugarbeet_x_1: ids 36/543/565/578 on the holder, 558/569 on the flat
lay) -- so their printed side is known. Each tag's four corners are detected in
every P1 frame, triangulated with P3's poses, and the scale is the printed side
over the triangulated one.

Independent of the flat lay, which is the point: a scale fitted to the flat lay
absorbs any global length error of the reconstruction, and scoring blade lengths
against that same flat lay then flatters the run.

Two checks travel with the number, because a wrong corner gives a wrong scale
without failing: the spread of the side estimates across edges and tags, and
how square each triangulated tag is (diagonal / side, sqrt(2) for a square).

Writes <workdir>/tag_scale.json, or --out.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from leaf_tips_2d import Rays, intersect

DICTIONARY = "DICT_APRILTAG_36h11"
CORNER_PX = 2.0          # corner reprojection inlier, frame pixels (subpixel-refined)
MIN_VIEWS = 3            # views a corner needs to be triangulated
MIN_BASELINE = np.deg2rad(10.0)   # widest ray pair a corner needs, for depth
MAX_SQUARE_DEV = 0.05    # |diagonal/side / sqrt(2) - 1| a tag may show


class Solve:
    """Just P3's cameras -- what Rays needs -- without the rest of a finished run."""

    def __init__(self, workdir: Path):
        import pycolmap
        from pose_estimator import cloud_source

        sparse, _ = cloud_source.geometry(workdir, cloud_source.BASELINE)
        self.workdir = workdir
        self.rec = pycolmap.Reconstruction(str(sparse))
        self.image_of = {Path(im.name).stem: im for im in self.rec.images.values()}


def detect(workdir: Path, frames) -> dict:
    """{tag id: {frame: 4x2 corners}} over the P1 frames."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY)), params)
    seen = defaultdict(dict)
    for frame in frames:
        grey = cv2.imread(str(workdir / "p1" / "frames" / f"{frame}.jpg"), cv2.IMREAD_GRAYSCALE)
        if grey is None:
            continue
        corners, ids, _ = detector.detectMarkers(grey)
        for c, i in zip(corners if ids is not None else [], ids.ravel() if ids is not None else []):
            seen[int(i)][frame] = c.reshape(4, 2).astype(float)
    return seen


def triangulate(solve, frames, uvs):
    """A corner from its views, dropping views that disagree; None if too few."""
    rays = Rays(solve, frames, uvs)
    keep = np.ones(len(frames), bool)
    for _ in range(6):
        if keep.sum() < MIN_VIEWS:
            return None
        X = intersect(rays.C[keep], rays.d[keep], np.ones(int(keep.sum())))
        err = rays.errors(X)
        new = err < max(CORNER_PX, 3.0 * float(np.median(err[keep])))
        if (new == keep).all():
            break
        keep = new
    keep &= err < CORNER_PX
    if keep.sum() < MIN_VIEWS:
        return None
    d = rays.d[keep]
    if np.arccos(np.clip((d @ d.T).min(), -1, 1)) < MIN_BASELINE:
        return None
    X = intersect(rays.C[keep], rays.d[keep], np.ones(int(keep.sum())))
    return {"xyz": X, "views": int(keep.sum()),
            "rms_px": float(np.sqrt(np.mean(rays.errors(X)[keep] ** 2)))}


def tag_scale(solve, tag_mm: float, frames=None) -> dict:
    """The scale and the evidence for it. mm_per_unit is None when no tag passes."""
    frames = sorted(frames if frames is not None else solve.image_of)
    seen = detect(solve.workdir, [f for f in frames if f in solve.image_of])
    tags, sides = {}, []
    for tag, views in sorted(seen.items()):
        fs = sorted(views)
        corners = [triangulate(solve, fs, [views[f][k] for f in fs]) for k in range(4)]
        row = {"frames_detected": len(fs)}
        if any(c is None for c in corners):
            row["rejected"] = "a corner did not triangulate (too few agreeing views or baseline)"
            tags[tag] = row
            continue
        P = np.array([c["xyz"] for c in corners])
        edge = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)
        diag = np.linalg.norm(P[[0, 1]] - P[[2, 3]], axis=1)
        square = float(diag.mean() / edge.mean() / np.sqrt(2))
        row.update({"edges_units": edge.tolist(), "square": square,
                    "views": [c["views"] for c in corners],
                    "rms_px": [c["rms_px"] for c in corners]})
        if abs(square - 1) > MAX_SQUARE_DEV:
            row["rejected"] = f"not square: diagonal/side is {square:.3f} x sqrt(2)"
        else:
            sides.extend(edge.tolist())
        tags[tag] = row
    out = {"tag_mm": tag_mm, "dictionary": DICTIONARY, "tags": tags, "mm_per_unit": None}
    if sides:
        side = float(np.median(sides))
        out.update({"mm_per_unit": tag_mm / side, "side_units": side, "edges_used": len(sides),
                    "spread": float(np.std(sides) / side)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True, help="a run with P1 frames and P3")
    ap.add_argument("--tag-mm", type=float, required=True,
                    help="printed side of the tags' black square, mm")
    ap.add_argument("--out", type=Path, help="default <workdir>/tag_scale.json")
    args = ap.parse_args()
    res = tag_scale(Solve(args.workdir), args.tag_mm)
    for tag, row in res["tags"].items():
        what = row.get("rejected") or (f"edges {np.mean(row['edges_units']):.5f} units, "
                                       f"square {row['square']:.3f}, views {row['views']}")
        print(f"  tag {tag:4d}  in {row['frames_detected']:3d} frames  {what}")
    if res["mm_per_unit"] is None:
        print("  no tag gave a scale")
    else:
        print(f"  {res['mm_per_unit']:.2f} mm/unit from {res['edges_used']} edges, "
              f"spread {res['spread']:.1%}")
    out = args.out or args.workdir / "tag_scale.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
