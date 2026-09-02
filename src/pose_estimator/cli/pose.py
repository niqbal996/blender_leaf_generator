"""P3 CLI: solve camera poses for a turntable capture, using the P2 masks.

    pose-solve --workdir runs/plant_9/

Reads <workdir>/p1/frames and <workdir>/p2/masks, writes into <workdir>/p3:
    masks/frame_XXXX.jpg.png   COLMAP masks (rotating rig only)
    sparse/best/               the winning COLMAP model
    poses.json                 acceptance checks + fitted orbit
    diag/                      rotating-region mask + orbit plot

Why the masks matter: the camera is static and the subject rotates, so
unmasked COLMAP locks onto the backdrop and concludes nothing moved. See
`pose_estimator.pose` for the full argument.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pose_estimator.pose import (
    evaluate_poses,
    export_scene_ply,
    rotating_masks_per_source,
    write_colmap_masks,
    write_orbit_plot,
    write_reprojection_overlays,
    write_scene_3d_plot,
)
from pose_estimator.reconstruction import build_sparse_reconstruction, get_registered_camera_poses


def run(
    workdir: Path,
    num_threads: int = 8,
    max_image_size: int = 1920,
    use_gpu: bool = False,
    low_texture: bool = False,
    cameras: str = "single",
    reuse_sparse: bool = False,
    single_camera: bool = True,
) -> dict:
    frames_dir = workdir / "p1" / "frames"
    p2_dir = workdir / "p2"
    p3_dir = workdir / "p3"
    p3_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames in {frames_dir} -- run pose-segment first")

    sources_file = workdir / "p1" / "sources.json"
    source_of = json.loads(sources_file.read_text()) if sources_file.exists() else {}
    sources = [int(source_of.get(p.stem, 0)) for p in frame_paths]
    num_passes = len(set(sources))

    print(f"Separating the rotating rig from the static backdrop "
          f"({len(frame_paths)} frames, {num_passes} capture pass(es))...")
    masks, thresholds = rotating_masks_per_source(frame_paths, sources)

    diag_dir = p3_dir / "diag"
    diag_dir.mkdir(exist_ok=True)
    for pass_index, rotating in sorted(masks.items()):
        coverage = float((rotating > 0).mean())
        print(f"  pass {pass_index}: rotating region covers {coverage:.1%} of the frame "
              f"(Otsu on temporal variance, t={thresholds[pass_index]:.0f})")
        if coverage > 0.9:
            print("    WARNING: nearly the whole frame is 'moving' -- is the backdrop really "
                  "rigid with the camera in this pass?")
        if coverage < 0.05:
            print("    WARNING: almost nothing is moving -- did the turntable turn in this clip?")

        first = next(p for p, g in zip(frame_paths, sources) if g == pass_index)
        overlay = cv2.imread(str(first))
        overlay[rotating > 0] = (
            0.65 * overlay[rotating > 0] + 0.35 * np.array([255, 120, 0])).astype(np.uint8)
        suffix = "" if num_passes == 1 else f"_pass{pass_index}"
        cv2.imwrite(str(diag_dir / f"rotating_region{suffix}.jpg"), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    mask_dir = p3_dir / "masks"
    n = write_colmap_masks(
        frame_paths,
        mask_dir,
        masks,
        plant_mask_dir=p2_dir / "masks" / "plant",
        holder_mask_dir=p2_dir / "masks" / "holder",
        sources=sources,
    )
    print(f"  wrote {n} COLMAP masks to {mask_dir}")

    sparse_best = p3_dir / "sparse" / "best"
    if reuse_sparse and sparse_best.is_dir():
        import pycolmap

        print(f"Reusing existing reconstruction at {sparse_best}...")
        reconstruction = pycolmap.Reconstruction(str(sparse_best))
    else:
        print("Running COLMAP (masked SIFT + exhaustive matching + incremental mapping)...")
        reconstruction = build_sparse_reconstruction(
            frames_dir,
            p3_dir,
            mask_dir=mask_dir,
            num_threads=num_threads,
            max_image_size=max_image_size,
            use_gpu=use_gpu,
            low_texture=low_texture,
            cameras=cameras,
            single_camera=single_camera,
        )

    report = evaluate_poses(reconstruction, num_input_frames=len(frame_paths),
                            sources=source_of if num_passes > 1 else None)

    centers, _, _ = get_registered_camera_poses(reconstruction)
    if report["orbit"] is not None:
        write_orbit_plot(diag_dir / "camera_orbit.png", centers, report["orbit"])

    sparse_xyz = np.array([p.xyz for p in reconstruction.points3D.values()])
    sparse_rgb = np.array([p.color for p in reconstruction.points3D.values()])
    write_scene_3d_plot(diag_dir / "scene_3d.png", centers, sparse_xyz, sparse_rgb, report["orbit"])
    write_reprojection_overlays(diag_dir, reconstruction, frames_dir)
    export_scene_ply(
        p3_dir / "sparse_points.ply", p3_dir / "camera_centers.ply", sparse_xyz, sparse_rgb, centers
    )
    print(f"  sparse cloud + camera path exported as PLY for Blender ({p3_dir})")

    with open(p3_dir / "poses.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  P3 checks ({'ALL PASSED' if report['all_passed'] else 'FAILURES PRESENT'}):")
    for name, check in report["checks"].items():
        print(f"    [{'PASS' if check['pass'] else 'FAIL'}] {name}: {check['detail']}")
    print(f"\n  {report['num_points3D']} sparse 3D points")
    print(f"  artifacts + diagnostics in {p3_dir}")

    return report


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path, help="Specimen run directory (must already have p1/ and p2/)")
    parser.add_argument("--num-threads", type=int, default=8, help="Threads for SIFT extraction")
    parser.add_argument(
        "--cameras", choices=["single", "per-image", "auto"], default="single",
        help="How COLMAP groups intrinsics. 'single' (default) is right when every "
             "frame came from one camera at one zoom -- the normal rig. Use "
             "'per-image' if the lens was zoomed or swapped between passes: one "
             "shared focal cannot fit two, and the passes come back interleaved.")
    parser.add_argument(
        "--low-texture",
        action="store_true",
        help="Spend more time on features so more frames register. Turns on COLMAP's "
             "viewpoint- and scale-robust descriptors (estimate_affine_shape, "
             "domain_size_pooling), keeps weaker maxima and runs guided matching. "
             "Use when frames fail to register on a small, smooth or softly-focused "
             "subject; costs roughly 3-5x the extraction time.")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1920,
        help="Downscale images to this max dimension before SIFT. Default keeps these 1920x1080 "
        "frames at native resolution -- the subject is small in frame, so downscaling costs "
        "exactly the features that matter.",
    )
    parser.add_argument("--use-gpu", action="store_true", help="GPU SIFT (needs a CUDA pycolmap build)")
    parser.add_argument(
        "--per-image-cameras",
        action="store_true",
        help="Let every frame solve its own intrinsics. Off by default because it is physically "
        "wrong for a locked-off rig, and COLMAP will use the freedom to absorb drift into the "
        "focal length -- measured at a 27.5%% focal spread on one capture. Only for footage where "
        "the lens genuinely changed mid-sequence.",
    )
    parser.add_argument(
        "--reuse-sparse",
        action="store_true",
        help="Skip COLMAP and re-score the model already at <workdir>/p3/sparse/best",
    )
    args = parser.parse_args(argv)

    run(
        workdir=args.workdir,
        num_threads=args.num_threads,
        max_image_size=args.max_image_size,
        low_texture=args.low_texture,
        cameras=args.cameras,
        use_gpu=args.use_gpu,
        reuse_sparse=args.reuse_sparse,
        single_camera=not args.per_image_cameras,
    )


if __name__ == "__main__":
    main()
