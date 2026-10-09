"""P4m CLI: COLMAP PatchMatch MVS on the full-resolution originals, fused on the plant.

    pose-mvs --workdir <ds>/plant                          # full resolution (cluster, CUDA)
    pose-mvs --workdir <ds>/plant --max-image-size 2400    # ~0.6x, a quicker first look
    pose-mvs --workdir <ds>/plant --stop-after prepare     # no CUDA needed: check the workspace

Reads p1/manifest.json (to find the original photos), p2/masks/plant,
p3/sparse/best and p4/hull_points.ply. Writes only into <workdir>/p4m:
    input/, dense/   the COLMAP workspace (dense/stereo holds the depth maps)
    fused.ply        COLMAP's fusion output
    surface.ply      the fused points inside the hull bound (x,y,z,n,rgb)
    p4m.json         settings and counts
A re-run reuses the depth maps already in dense/ unless --redo-patch-match.

P4a and P4b are untouched, so they stay the baselines this is compared to
(scripts/compare_clouds.py). See pose_estimator.mvs for the method.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from pose_estimator.ply_io import read_ply_vertices, write_ply_vertices


def main(argv: Optional[list] = None) -> None:
    import pycolmap

    from pose_estimator import mvs

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--source-root", type=Path,
                    help="where the original photos are on this machine: <root>/<pass>/<file>; "
                         "default: the paths in p1/manifest.json")
    ap.add_argument("--max-image-size", type=int, default=-1,
                    help="cap on the undistorted, plant-cropped images' longer side (-1: full)")
    ap.add_argument("--num-src", type=int, default=15, help="source views per image")
    ap.add_argument("--window-radius", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--min-num-pixels", type=int, default=3,
                    help="fusion: views a point must be seen consistently in")
    ap.add_argument("--gpu-index", default="-1")
    ap.add_argument("--plain-fusion", action="store_true",
                    help="fuse without the plant masks and bounding box (the hull bound still applies)")
    ap.add_argument("--stop-after", choices=["prepare", "patch_match", "fuse"], default="fuse")
    ap.add_argument("--redo-patch-match", action="store_true")
    args = ap.parse_args(argv)

    wd, out = args.workdir, args.workdir / "p4m"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = read_ply_vertices(wd / "p4" / "hull_points.ply")
    hull = np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float64)
    rec = pycolmap.Reconstruction(str(wd / "p3" / "sparse" / "best"))
    p3_focal = float(np.mean([rec.cameras[c].mean_focal_length() for c in rec.cameras]))

    maps = out / "dense" / "stereo" / "depth_maps"
    have_maps = maps.is_dir() and (any(maps.glob("*.geometric.bin")) or any(maps.glob("*.photometric.bin")))
    if have_maps and not args.redo_patch_match:
        print(f"P4m: reusing the depth maps in {maps}")
        info = json.loads((out / "prepare.json").read_text())
    else:
        print("P4m: preparing the workspace")
        info = mvs.prepare(wd, out, hull, source_root=args.source_root,
                           max_image_size=args.max_image_size, num_src=args.num_src)
        if args.stop_after == "prepare":
            print(f"  stopped after prepare; workspace in {out / 'dense'} ({time.time() - t0:.0f} s)")
            return
        print("P4m: PatchMatch stereo")
        mvs.patch_match(out, info, window_radius=args.window_radius, iterations=args.iterations,
                        gpu_index=args.gpu_index)
    t_pm = time.time() - t0
    if args.stop_after == "patch_match":
        return
    print("P4m: fusion on the plant")
    pts, nrm, col, stats = mvs.fuse(out, info, hull, p3_focal, min_num_pixels=args.min_num_pixels,
                                    plain=args.plain_fusion)
    write_ply_vertices(out / "surface.ply", {
        "x": pts[:, 0].astype(np.float32), "y": pts[:, 1].astype(np.float32),
        "z": pts[:, 2].astype(np.float32),
        "nx": nrm[:, 0].astype(np.float32), "ny": nrm[:, 1].astype(np.float32),
        "nz": nrm[:, 2].astype(np.float32),
        "red": col[:, 0], "green": col[:, 1], "blue": col[:, 2]}, binary=True)
    report = {**info, **stats, "settings": vars(args) | {"workdir": str(wd), "source_root": str(args.source_root)},
              "seconds": {"prepare_and_patch_match": round(t_pm), "total": round(time.time() - t0)}}
    (out / "p4m.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"  wrote {out / 'surface.ply'} ({len(pts):,} points) and p4m.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
