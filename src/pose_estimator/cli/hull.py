"""P4 (part 1) CLI: carve the visual hull from the P2 silhouettes and P3 poses.

    pose-hull --workdir runs/plant_9/ --resolution 256

Reads <workdir>/p2/masks/plant and <workdir>/p3/sparse/best, writes into
<workdir>/p4:
    hull.ply            watertight-ish hull mesh
    hull_points.ply     occupied voxel centres
    hull.json           carve settings + acceptance checks
    diag/               hull reprojected over input frames

The hull is a *bound*, not the final surface -- silhouettes cannot see
concavities, so a cupped leaf carves flat. Its job is to be the deterministic
constraint that later stages get rejected against, replacing the old
pipeline's hand-tuned per-plant opacity/density thresholds.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pose_estimator.hull import (
    bounds_from_points,
    carve,
    load_carve_cameras,
    to_mesh,
    write_hull_3d_plot,
    write_ply_mesh,
)


def run(
    workdir: Path,
    resolution: int = 256,
    min_inside_fraction: float = 0.86,
    dilate_px: int = 2,
    min_judged_views: int = 8,
    min_judged_fraction: float = 0.5,
) -> dict:
    import pycolmap

    p3_sparse = workdir / "p3" / "sparse" / "best"
    if not p3_sparse.is_dir():
        raise FileNotFoundError(f"{p3_sparse} not found -- run pose-solve on this workdir first")

    plant_masks = workdir / "p2" / "masks" / "plant"
    p4_dir = workdir / "p4"
    p4_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading poses from {p3_sparse}...")
    reconstruction = pycolmap.Reconstruction(str(p3_sparse))
    holder_masks = workdir / "p2" / "masks" / "holder"
    occluder_dir = holder_masks if holder_masks.is_dir() else None
    cameras = load_carve_cameras(reconstruction, plant_masks, dilate_px=dilate_px,
                                 occluder_dir=occluder_dir)
    print(f"  {len(cameras)} views with silhouettes (masks dilated {dilate_px}px)")
    if occluder_dir is None:
        print("  no holder masks -- carving without occlusion handling; anything the")
        print("  tool hides for most of the orbit will be carved away")
    else:
        hidden = np.mean([c.occluder.mean() for c in cameras if c.occluder is not None])
        print(f"  holder masks used as occluders: {hidden:.1%} of an average frame is "
              "hidden and does not vote")
    if len(cameras) < 8:
        raise RuntimeError(f"Only {len(cameras)} usable views -- carving needs many more than that")

    sparse_xyz = np.array([p.xyz for p in reconstruction.points3D.values()])
    bounds_min, bounds_max = bounds_from_points(sparse_xyz)
    print(f"  initial volume {np.round(bounds_max - bounds_min, 3).tolist()} (COLMAP units)")

    print(f"Carving to {resolution}^3 (voxel must be in-silhouette in >={min_inside_fraction:.0%} of views)...")
    points, voxel, _ = carve(
        cameras,
        bounds_min,
        bounds_max,
        resolution=resolution,
        min_inside_fraction=min_inside_fraction,
        min_judged_views=min_judged_views,
        min_judged_fraction=min_judged_fraction,
    )
    extent = points.max(axis=0) - points.min(axis=0)
    print(f"  {len(points)} occupied voxels, voxel size {voxel:.5f}, extent {np.round(extent, 3).tolist()}")

    verts, faces = to_mesh(points, voxel)
    write_ply_mesh(p4_dir / "hull.ply", verts, faces)
    print(f"  hull mesh: {len(verts)} vertices, {len(faces)} faces -> {p4_dir / 'hull.ply'}")

    _write_points_ply(p4_dir / "hull_points.ply", points)

    orbit = None
    poses_path = workdir / "p3" / "poses.json"
    if poses_path.exists():
        with open(poses_path) as f:
            orbit = json.load(f).get("orbit")
    write_hull_3d_plot(p4_dir / "diag" / "hull_3d.png", points, sparse_xyz, orbit)

    report = _evaluate(cameras, points, voxel, p4_dir)
    report.update(
        {
            "resolution": resolution,
            "voxel_size": voxel,
            "num_voxels": int(len(points)),
            "num_views": len(cameras),
            "min_inside_fraction": min_inside_fraction,
            "mask_dilation_px": dilate_px,
            # Recorded so a hull on disk says which carve produced it. These
            # two are what separates the root from the solid block the pliers
            # would otherwise leave in their own shadow, and a run predating
            # them is otherwise indistinguishable from a current one.
            "min_judged_views": min_judged_views,
            "min_judged_fraction": min_judged_fraction,
            "occlusion_aware": occluder_dir is not None,
            "extent": extent.tolist(),
        }
    )
    with open(p4_dir / "hull.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  P4 hull checks ({'ALL PASSED' if report['all_passed'] else 'FAILURES PRESENT'}):")
    for name, check in report["checks"].items():
        print(f"    [{'PASS' if check['pass'] else 'FAIL'}] {name}: {check['detail']}")
    print(f"\n  artifacts + diagnostics in {p4_dir}")
    return report


def _write_points_ply(path: Path, points: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(
            (
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(points)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "end_header\n"
            ).encode("ascii")
        )
        f.write(points.astype("<f4").tobytes())


def _evaluate(cameras, points: np.ndarray, voxel: float, p4_dir: Path) -> dict:
    """Reproject the hull into every view and compare with the input silhouette.

    This is the check the plan asks for in P4, and it is the first genuinely
    independent cross-validation in the pipeline: the carve enforces that the
    hull lies *inside* every silhouette, but nothing forces it to *fill* them.
    A hull that covers only part of each mask means the poses and the masks
    disagree -- so a high IoU is real evidence that P2 and P3 are consistent
    with each other, which neither phase could establish alone.
    """
    diag_dir = p4_dir / "diag"
    diag_dir.mkdir(exist_ok=True)

    ious, recalls = [], []
    sample_indices = set(np.linspace(0, len(cameras) - 1, 6).astype(int).tolist())

    for index, camera in enumerate(cameras):
        pixels, in_front = camera.project(points)
        height, width = camera.mask.shape
        x = np.round(pixels[:, 0]).astype(np.int64)
        y = np.round(pixels[:, 1]).astype(np.int64)
        keep = in_front & (x >= 0) & (x < width) & (y >= 0) & (y < height)

        rendered = np.zeros_like(camera.mask)
        rendered[y[keep], x[keep]] = True

        # Voxel centres are point samples, so they project to a stipple with
        # gaps between them; comparing that raw would measure how finely the
        # hull was sampled rather than what shape it is. Close by the voxel's
        # own projected size -- derived per view from the geometry, not picked
        # -- so neighbouring samples merge exactly when they represent
        # neighbouring voxels and no further.
        depth = (points @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3])[:, 2]
        median_depth = float(np.median(depth[depth > 1e-6])) if (depth > 1e-6).any() else 1.0
        projected_voxel_px = float(camera.K[0, 0]) * voxel / max(median_depth, 1e-6)
        radius = int(np.clip(np.ceil(projected_voxel_px), 1, 9))
        rendered = cv2.morphologyEx(
            rendered.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        ).astype(bool)

        union = (rendered | camera.mask).sum()
        ious.append(float((rendered & camera.mask).sum() / union) if union else 0.0)
        recalls.append(
            float((rendered & camera.mask).sum() / camera.mask.sum()) if camera.mask.any() else 0.0
        )

        if index in sample_indices:
            overlay = np.zeros((height, width, 3), np.uint8)
            overlay[camera.mask] = (0, 90, 0)
            overlay[rendered] = (0, 0, 200)
            overlay[rendered & camera.mask] = (0, 220, 220)
            cv2.putText(
                overlay,
                f"{camera.name}  IoU={ious[-1]:.3f}  (cyan=agree, green=mask only, red=hull only)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
            )
            cv2.imwrite(str(diag_dir / f"hull_vs_mask_{Path(camera.name).stem}.jpg"), overlay,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

    mean_iou = float(np.mean(ious))
    mean_recall = float(np.mean(recalls))
    checks = {
        "hull_reprojects_onto_masks": {
            "pass": mean_iou >= 0.75,
            "detail": f"mean hull-vs-mask IoU {mean_iou:.3f} over {len(cameras)} views (target 0.75)",
        },
        "hull_fills_the_silhouettes": {
            "pass": mean_recall >= 0.90,
            "detail": f"hull covers {mean_recall:.1%} of mask area on average (target 90%)",
        },
    }
    return {
        "reprojection_iou": {"mean": mean_iou, "min": float(np.min(ious))},
        "silhouette_recall": {"mean": mean_recall, "min": float(np.min(recalls))},
        "checks": checks,
        "all_passed": all(c["pass"] for c in checks.values()),
    }


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path, help="Specimen run directory (needs p2/ and p3/)")
    parser.add_argument("--resolution", type=int, default=256, help="Final carve resolution (per axis)")
    parser.add_argument(
        "--min-inside-fraction",
        type=float,
        default=0.86,
        help="Share of observing views that must place a voxel inside the silhouette. 1.0 is the "
        "textbook visual hull and is brittle -- it deletes the exposed root, which is thin and "
        "hidden behind the pliers for part of the turn. The default was fit by sweeping both "
        "specimens; reprojection IoU peaks there for each.",
    )
    parser.add_argument(
        "--dilate-px",
        type=int,
        default=2,
        help="Dilate silhouettes before carving. Biases the hull outward, which is the right "
        "direction of error for an upper bound.",
    )
    args = parser.parse_args(argv)

    run(
        workdir=args.workdir,
        resolution=args.resolution,
        min_inside_fraction=args.min_inside_fraction,
        dilate_px=args.dilate_px,
    )


if __name__ == "__main__":
    main()
