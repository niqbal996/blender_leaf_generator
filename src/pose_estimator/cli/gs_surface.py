"""P4g CLI: a full-resolution 2DGS surface, for leaves P4a/P4b merge at the heart.

    pose-gs-surface --workdir <ds>/plant                    # full resolution, 30k steps
    pose-gs-surface --workdir <ds>/plant --scale 0.5        # half resolution, faster

Run P4m (pose-mvs) first when possible: the Gaussians then start on its
photo-consistent points instead of the hull (--init).

Reads p1/manifest.json (to find the original photos), p2/masks/{plant,holder},
p3/sparse/best and p4/hull_points.ply. Writes only into <workdir>/p4g:
    surface.ply     surface points with normals and colour
    gaussians.npz   the trained 2D Gaussians
    p4g.json        settings, training history, extraction counts
    diag/           photo | render | normals, whole crop and a 1:1 detail

P4a and P4b are untouched, so they stay the baselines this is compared to
(scripts/compare_clouds.py). See pose_estimator.gs_surface for the method.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pose_estimator.ply_io import read_ply_vertices, write_ply_vertices


def run(workdir: Path, scale: float = 1.0, iterations: int = 30_000, seeds: int = 400_000,
        source_root: Optional[Path] = None, frames=None, stride: int = 2, sh_degree: int = 3,
        dist_from: int = 3000, normal_from: int = 7000, init: str = "auto",
        device: str = "cuda") -> dict:
    import pycolmap
    import torch

    from pose_estimator import gs_surface as gs
    from pose_estimator.fine_views import load_fine_views

    out = workdir / "p4g"
    (out / "diag").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    hull_fields = read_ply_vertices(workdir / "p4" / "hull_points.ply")
    hull = np.stack([hull_fields["x"], hull_fields["y"], hull_fields["z"]], 1).astype(np.float64)
    hull_voxel = float(json.loads((workdir / "p4" / "hull.json").read_text())["voxel_size"])
    rec = pycolmap.Reconstruction(str(workdir / "p3" / "sparse" / "best"))
    p3_focal = float(np.mean([rec.cameras[c].mean_focal_length() for c in rec.cameras]))
    scene_scale = float(np.linalg.norm(hull.max(0) - hull.min(0)) / 2)

    print(f"P4g: loading views at x{scale:g} of the original photos")
    views = load_fine_views(workdir, hull, scale=scale, source_root=source_root, frames=frames)
    mvs_cloud = workdir / "p4m" / "surface.ply"
    if init == "auto":
        init = "p4m" if mvs_cloud.exists() else "hull"
    if init == "p4m":
        m = read_ply_vertices(mvs_cloud)
        pts = np.stack([m["x"], m["y"], m["z"]], 1).astype(np.float64)
        print(f"  seeding {min(seeds, len(pts)):,} Gaussians on P4m's {len(pts):,} MVS points")
        params = gs.init_from_points(pts, np.stack([m["nx"], m["ny"], m["nz"]], 1),
                                     np.stack([m["red"], m["green"], m["blue"]], 1).astype(np.float64),
                                     seeds, sh_degree, device)
    else:
        print(f"  hull: {len(hull):,} voxels of {hull_voxel:.5f}; seeding {min(seeds, len(hull)):,} "
              f"Gaussians inside and on it (no P4m cloud to start from)")
        params = gs.init_from_hull(hull, hull_voxel, views, seeds, sh_degree, device)

    print(f"P4g: training {iterations} steps")
    result = gs.train(params, views, iterations=iterations, scene_scale=scene_scale,
                      sh_degree=sh_degree, dist_from=dist_from, normal_from=normal_from,
                      device=device)
    t_train = time.time() - t_start

    print("P4g: surface from rendered median depth, kept where other views agree")
    points, normals, colours, stats = gs.extract(params, views, hull, p3_focal,
                                                 sh_degree=sh_degree, stride=stride, device=device)
    print(f"  {stats['surface_points']:,} surface points at {stats['consolidation_voxel']:.6f} "
          f"units spacing")
    write_ply_vertices(out / "surface.ply", {
        "x": points[:, 0].astype(np.float32), "y": points[:, 1].astype(np.float32),
        "z": points[:, 2].astype(np.float32),
        "nx": normals[:, 0].astype(np.float32), "ny": normals[:, 1].astype(np.float32),
        "nz": normals[:, 2].astype(np.float32),
        "red": colours[:, 0], "green": colours[:, 1], "blue": colours[:, 2]})
    np.savez_compressed(out / "gaussians.npz",
                        **{k: v.detach().cpu().numpy() for k, v in params.items()})

    silhouette = _diagnostics(params, views, out / "diag", sh_degree, device)
    report = {"scale": scale, "iterations": iterations, "seeds": seeds, "init": init, "stride": stride,
              "views": len(views), "crop_px_median": [int(np.median([v.image.shape[1] for v in views])),
                                                      int(np.median([v.image.shape[0] for v in views]))],
              "focal_px": float(views[0].K[0, 0]), "gaussians": int(len(params["means"])),
              "silhouette_agreement": silhouette, "extraction": stats,
              "seconds": {"train": round(t_train), "total": round(time.time() - t_start)},
              "training": result["history"]}
    (out / "p4g.json").write_text(json.dumps(report, indent=1))
    print(f"  silhouette agreement on pixels the mask is sure of: {silhouette:.3f}")
    print(f"  wrote {out / 'surface.ply'}, gaussians.npz, p4g.json, diag/  "
          f"({report['seconds']['total']} s)")
    return report


def _diagnostics(params, views, diag: Path, sh_degree: int, device: str) -> float:
    """photo | render | normals for a few views; mean silhouette agreement over all."""
    import torch

    from pose_estimator import gs_surface as gs

    agree = []
    show = set(np.linspace(0, len(views) - 1, 4).round().astype(int).tolist())
    with torch.no_grad():
        for i, v in enumerate(views):
            h, w = v.mask.shape
            K = torch.tensor(v.K, dtype=torch.float32, device=device)
            w2c = torch.tensor(v.world_to_camera, dtype=torch.float32, device=device)
            rgb, alpha, normal, *_ = gs._render(params, K, w2c, w, h, sh_degree)
            sure, maybe, evidence = gs.loss_masks(v)
            a = alpha[0, ..., 0].cpu().numpy() > 0.5
            judged = ((sure > 0) | (maybe == 0)) & (evidence > 0)
            agree.append(float((a == (sure > 0))[judged].mean()))
            if i not in show:
                continue
            photo = cv2.cvtColor(v.image, cv2.COLOR_RGB2BGR)
            render = cv2.cvtColor((rgb[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                                  cv2.COLOR_RGB2BGR)
            nrm = ((normal[0].cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)[..., ::-1]
            strip = np.hstack([photo, render, nrm])
            scale = 1800 / strip.shape[1]
            cv2.imwrite(str(diag / f"{v.name}_whole.jpg"), cv2.resize(strip, None, fx=scale, fy=scale),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            ys, xs = np.nonzero(v.mask > 127)
            cy, cx = int(np.median(ys)), int(np.median(xs))
            r = 300
            crop = lambda im: im[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            cv2.imwrite(str(diag / f"{v.name}_detail.jpg"),
                        np.hstack([crop(photo), crop(render), crop(nrm)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return float(np.mean(agree))


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resolution as a share of the original photos (1.0 = full, 6000 px on the "
                         "D5600; 0.32 ~ P1's frames)")
    ap.add_argument("--iterations", type=int, default=30_000)
    ap.add_argument("--seeds", type=int, default=400_000, help="hull voxels to seed Gaussians on")
    ap.add_argument("--stride", type=int, default=2,
                    help="back-project every n-th pixel of each rendered depth map")
    ap.add_argument("--source-root", type=Path,
                    help="where the original photos are on this machine: <root>/<pass>/<file>; "
                         "default: the paths in p1/manifest.json")
    ap.add_argument("--frames", nargs="+", help="only these frame stems (smoke tests)")
    ap.add_argument("--dist-from", type=int, default=3000, help="step the distortion loss starts")
    ap.add_argument("--normal-from", type=int, default=7000, help="step the normal loss starts")
    ap.add_argument("--init", choices=["auto", "p4m", "hull"], default="auto",
                    help="where the Gaussians start: P4m's MVS points (auto, when p4m/surface.ply "
                         "exists) or the P4a hull's voxels")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    run(args.workdir, scale=args.scale, iterations=args.iterations, seeds=args.seeds,
        source_root=args.source_root, frames=args.frames, stride=args.stride,
        dist_from=args.dist_from, normal_from=args.normal_from, init=args.init, device=args.device)


if __name__ == "__main__":
    main()
