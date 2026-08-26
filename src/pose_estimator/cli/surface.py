"""P4b CLI: train surface-aligned 2D Gaussians and extract a carved surface.

    pose-surface --workdir runs/plant_9/ --iterations 4000

Reads <workdir>/p1/frames, p2/masks/plant, p3/sparse/best and p4/hull_points.ply.
Writes into <workdir>/p4b:
    surface.ply       carved surface point cloud with normals (the P5 input)
    surfels.npz       trained 2D Gaussian parameters
    p4b.json          training history + acceptance checks
    diag/             rendered-vs-input comparisons, 3D surface views

Why this phase exists: P4a's hull is a solid bound, and P5 skeletonisation
needs a thin surface. See `pose_estimator.surfels` for the full argument.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pose_estimator.ply_io import read_ply_vertices, write_ply_vertices
from pose_estimator.surfels import (
    consolidate,
    init_surfels_from_hull,
    load_views,
    render_surface_points,
    train,
)


def run(
    workdir: Path,
    iterations: int = 4000,
    downsample: int = 2,
    target_surfels: int = 150_000,
    alpha_threshold: float = 0.5,
    device: str = "cuda",
) -> dict:
    import pycolmap
    import torch

    p4b = workdir / "p4b"
    (p4b / "diag").mkdir(parents=True, exist_ok=True)

    hull_fields = read_ply_vertices(workdir / "p4" / "hull_points.ply")
    hull = np.stack([hull_fields["x"], hull_fields["y"], hull_fields["z"]], axis=1).astype(np.float64)
    with open(workdir / "p4" / "hull.json") as f:
        hull_voxel = json.load(f)["voxel_size"]

    reconstruction = pycolmap.Reconstruction(str(workdir / "p3" / "sparse" / "best"))
    sparse = np.array([p.xyz for p in reconstruction.points3D.values()])
    sparse_rgb = np.array([p.color for p in reconstruction.points3D.values()])

    print(f"Loading views (downsample x{downsample})...")
    holder_masks = workdir / "p2" / "masks" / "holder"
    views = load_views(reconstruction, workdir / "p1" / "frames", workdir / "p2" / "masks" / "plant",
                       downsample=downsample,
                       occluder_dir=holder_masks if holder_masks.is_dir() else None)
    print(f"  {len(views)} views at {views[0].image.shape[1]}x{views[0].image.shape[0]}")
    if views[0].occluder is None:
        print("  no holder masks -- anything the tool hides for most of the orbit will be")
        print("  trained away, the same way it used to be carved away in P4a")
    else:
        hidden = float(np.mean([v.occluder.mean() for v in views if v.occluder is not None]))
        print(f"  holder masks used as occluders: {hidden:.1%} of an average frame carries "
              "no evidence and is excluded from the silhouette loss")

    print(f"Initialising surfels from the hull ({len(hull)} voxels)...")
    params = init_surfels_from_hull(hull, sparse, sparse_rgb, hull_voxel,
                                    target_count=target_surfels, device=device)
    print(f"  {len(params['means'])} surfels")

    result = train(params, views, iterations=iterations, device=device)
    for entry in result["history"][-1:]:
        print(f"  final loss {entry['loss']:.4f} (rgb L1 {entry['l1']:.4f}, "
              f"mask L1 {entry['mask_l1']:.4f}, normal {entry['normal']:.4f})")

    print("Extracting geometry from rendered depth and carving against the hull...")
    points, normals, carve_stats = render_surface_points(
        params, views, hull, hull_voxel, alpha_threshold=alpha_threshold, device=device
    )
    print(f"  back-projected {carve_stats['raw_backprojected']} points, "
          f"{carve_stats['rejected_fraction']:.1%} rejected by the hull")

    points, normals = consolidate(points, normals, voxel=hull_voxel)
    print(f"  consolidated to {len(points)} surface points")

    write_ply_vertices(
        p4b / "surface.ply",
        {
            "x": points[:, 0].astype(np.float32), "y": points[:, 1].astype(np.float32),
            "z": points[:, 2].astype(np.float32),
            "nx": normals[:, 0].astype(np.float32), "ny": normals[:, 1].astype(np.float32),
            "nz": normals[:, 2].astype(np.float32),
        },
    )
    np.savez(
        p4b / "surfels.npz",
        **{k: v.detach().cpu().numpy() for k, v in params.items()},
    )

    report = _evaluate(params, views, points, normals, hull_voxel, p4b, device)
    report.update({"training": result["history"], "carve": carve_stats,
                   "num_surface_points": int(len(points)), "num_surfels": int(len(params["means"]))})
    with open(p4b / "p4b.json", "w") as f:
        json.dump(report, f, indent=2)

    _write_surface_3d_plot(p4b / "diag" / "surface_3d.png", points, normals, hull)

    print(f"\n  P4b checks ({'ALL PASSED' if report['all_passed'] else 'FAILURES PRESENT'}):")
    for name, check in report["checks"].items():
        print(f"    [{'PASS' if check['pass'] else 'FAIL'}] {name}: {check['detail']}")
    print(f"\n  artifacts + diagnostics in {p4b}")
    return report


def _evaluate(params, views, points, normals, voxel, p4b, device) -> dict:
    """Score the splat against held-out silhouettes and measure surface thinness."""
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization_2dgs
    from scipy.spatial import cKDTree

    ious = []
    sample_indices = set(np.linspace(0, len(views) - 1, 5).astype(int).tolist())

    with torch.no_grad():
        for index, view in enumerate(views):
            height, width = view.image.shape[:2]
            K = torch.tensor(view.K, dtype=torch.float32, device=device)[None]
            viewmat = torch.tensor(view.world_to_camera, dtype=torch.float32, device=device)[None]
            rendered, alphas, *_ = rasterization_2dgs(
                params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
                torch.sigmoid(params["opacities"]), torch.sigmoid(params["colors"])[None],
                viewmat, K, width, height, render_mode="RGB+ED",
            )
            alpha = alphas[0, ..., 0].cpu().numpy()
            rendered_mask = alpha > 0.5
            target = view.mask > 0.5
            union = (rendered_mask | target).sum()
            ious.append(float((rendered_mask & target).sum() / union) if union else 0.0)

            if index in sample_indices:
                rgb = (rendered[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                side = np.hstack([
                    cv2.cvtColor((view.image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                ])
                cv2.putText(side, f"{view.name}  input | rendered   IoU={ious[-1]:.3f}",
                            (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imwrite(str(p4b / "diag" / f"render_{Path(view.name).stem}.jpg"), side,
                            [cv2.IMWRITE_JPEG_QUALITY, 88])

    # Thinness: the whole point of this phase. Measure how planar a local
    # neighbourhood is -- the hull scored 0.798 (isotropic blobs), a real
    # surface should be far lower.
    rng = np.random.default_rng(0)
    tree = cKDTree(points)
    flatness = []
    for i in rng.choice(len(points), min(3000, len(points)), replace=False):
        neighbourhood = points[tree.query_ball_point(points[i], voxel * 6)]
        if len(neighbourhood) < 12:
            continue
        centred = neighbourhood - neighbourhood.mean(0)
        singular = np.linalg.svd(centred, compute_uv=False) / np.sqrt(len(neighbourhood))
        flatness.append(singular[2] / max(singular[0], 1e-9))
    median_flatness = float(np.median(flatness)) if flatness else 1.0
    mean_iou = float(np.mean(ious))

    checks = {
        "renders_match_silhouettes": {
            "pass": mean_iou >= 0.90,
            "detail": f"mean rendered-alpha vs input-mask IoU {mean_iou:.3f} over {len(views)} views (target 0.90)",
        },
        "surface_is_thin": {
            "pass": median_flatness <= 0.25,
            "detail": f"median local flatness {median_flatness:.3f} (hull was 0.798; target <=0.25 "
            f"so P5 sees sheets, not blobs)",
        },
    }
    return {"silhouette_iou": mean_iou, "median_flatness": median_flatness,
            "checks": checks, "all_passed": all(c["pass"] for c in checks.values())}


def _write_surface_3d_plot(out_path: Path, points, normals, hull):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    rng = np.random.default_rng(0)
    show = points[rng.choice(len(points), min(40000, len(points)), replace=False)] if len(points) else points
    shade = (np.abs(normals[: len(show)]) if len(normals) >= len(show) else None)

    fig = plt.figure(figsize=(16, 5.2))
    for i, (elev, azim, title) in enumerate(
        [(16, -60, "surface, perspective"), (2, -90, "surface, front"), (89, -90, "surface, from above")]
    ):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(show[:, 0], show[:, 1], show[:, 2],
                   c=shade if shade is not None else "#2F7A4A", s=0.6, alpha=0.7, linewidths=0)
        span = (show.max(axis=0) - show.min(axis=0)).max() / 2.0
        mid = (show.max(axis=0) + show.min(axis=0)) / 2.0
        ax.set_xlim(mid[0] - span, mid[0] + span); ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.view_init(elev=elev, azim=azim); ax.set_title(title, fontsize=10)
        for setter in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
            setter([])
    fig.suptitle("P4b: surface from rendered depth, carved against the hull (colour = normal direction)",
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--downsample", type=int, default=2,
                        help="Train against images downsampled by this factor")
    parser.add_argument("--target-surfels", type=int, default=150_000,
                        help="How many hull voxels to seed surfels from")
    parser.add_argument("--alpha-threshold", type=float, default=0.5,
                        help="Minimum rendered alpha for a depth sample to be back-projected")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run(workdir=args.workdir, iterations=args.iterations, downsample=args.downsample,
        target_surfels=args.target_surfels, alpha_threshold=args.alpha_threshold, device=args.device)


if __name__ == "__main__":
    main()
