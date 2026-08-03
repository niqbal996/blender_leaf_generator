"""CLI: estimate a plant's stem/branch/tip skeleton from a rotation video or
an already-captured folder of still images.

    pose-estimate-skeleton --video path/to/video.MOV --workdir out/
    pose-estimate-skeleton --images path/to/stills/ --workdir out/

Pipeline: extract frames from video, or use a still-image folder as-is ->
COLMAP sparse reconstruction (unmasked, by default -- see --mask-mode) ->
solve the turntable's axis/soil plane and crop the plant out geometrically
-> filter 3D points by color -> point-cloud cleanup -> trace organs outward
from the stem base -> render + JSON summary.

This is a research prototype (see src/pose_estimator/__init__.py
and the README's "Plant skeleton from video" section) -- inspect the point
cloud render alongside the skeleton, don't trust the graph blindly, since
the whole pipeline lives or dies on how clean the source images are.

Requires the "skeleton" extra: pip install -e ".[skeleton]"
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from pose_estimator.frames import extract_frames
from pose_estimator.masking import write_colmap_masks
from pose_estimator.pointcloud import (
    extract_xyz_rgb,
    keep_plant_clusters,
    remove_sparse_points,
    remove_statistical_outliers,
    vegetation_color_mask,
)
from pose_estimator.ply_io import write_ply_vertices
from pose_estimator.reconstruction import (
    build_sparse_reconstruction,
    get_registered_camera_poses,
)
from pose_estimator.turntable import (
    crop_to_plant,
    find_root_point_on_ground,
    solve_turntable_frame,
)
from pose_estimator.skeletonize import build_skeleton_graph, smooth_polyline
from pose_estimator.visualize import plot_skeleton


def run(
    workdir: Path,
    video_path: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    num_frames: int = 60,
    mask_mode: str = "none",
    color_filter: str = "vegetation",
    color_filter_threshold: float = 0.12,
    k_neighbors: int = 8,
    min_branch_fraction: float = 0.25,
    cover_radius_fraction: float = 0.08,
    min_point_density: float = 0.30,
    voxel_downsample_fraction: Optional[float] = None,
    num_threads: int = 4,
    max_image_size: int = 2000,
    use_gpu: bool = False,
    reuse_sparse: bool = False,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    masks_dir = workdir / "masks"

    if video_path is not None:
        images_dir = workdir / "images"
        print(f"Extracting {num_frames} frames from {video_path.name}...")
        frame_paths = extract_frames(video_path, images_dir, target_frame_count=num_frames)
        print(f"  wrote {len(frame_paths)} frames to {images_dir}")
        num_images = len(frame_paths)
    else:
        num_images = sum(
            1 for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        print(f"Using {num_images} pre-captured images from {images_dir}")

    mask_dir_arg = None
    if mask_mode != "none":
        print(f"Masking frames before matching (mode={mask_mode})...")
        n_masks = write_colmap_masks(images_dir, masks_dir, mode=mask_mode)
        print(f"  wrote {n_masks} masks to {masks_dir}")
        mask_dir_arg = masks_dir

    # Everything after this point is seconds of work, while the
    # reconstruction is minutes-to-hours -- and the skeleton's two tuning
    # knobs usually need a few passes to settle on a given plant. Reloading
    # the model the last run already saved makes that loop practical.
    sparse_best = workdir / "sparse" / "best"
    if reuse_sparse and sparse_best.is_dir():
        import pycolmap

        print(f"Reusing existing sparse reconstruction from {sparse_best}...")
        reconstruction = pycolmap.Reconstruction(str(sparse_best))
    else:
        if reuse_sparse:
            print(f"(--reuse-sparse given, but {sparse_best} does not exist yet)")
        print("Running COLMAP sparse reconstruction (this can take a couple minutes)...")
        reconstruction = build_sparse_reconstruction(
            images_dir,
            workdir,
            mask_dir=mask_dir_arg,
            num_threads=num_threads,
            max_image_size=max_image_size,
            use_gpu=use_gpu,
        )
    print(
        f"  registered {reconstruction.num_reg_images()} / {num_images} images, "
        f"{reconstruction.num_points3D()} 3D points, "
        f"mean reprojection error {reconstruction.compute_mean_reprojection_error():.2f}px"
    )

    xyz, rgb = extract_xyz_rgb(reconstruction)

    # Geometry before color: the turntable's own axis and soil plane say
    # which points are "the plant standing on the pot" far more reliably
    # than greenness does. On both turntable test captures roughly half the
    # points passing the ExG threshold sat *below* the soil plane --
    # background foliage and green-cast highlights that no color threshold
    # or outlier filter separates, but a cylinder crop removes outright.
    camera_centers, viewing_dirs, _ = get_registered_camera_poses(reconstruction)
    # The vegetation mask is computed here only to keep leaves out of the
    # substrate fit; the actual color filtering still happens after cropping.
    frame = solve_turntable_frame(
        xyz,
        camera_centers,
        viewing_dirs,
        is_plant=vegetation_color_mask(rgb, exg_threshold=color_filter_threshold),
    )
    print(
        f"  turntable axis {frame.up.round(3).tolist()}, footprint radius "
        f"{frame.radius:.3f} (COLMAP units)"
    )
    print(
        f"  substrate top face tilted {frame.substrate_tilt_degrees:.1f} deg "
        "from the turntable plane"
    )

    xyz, rgb = crop_to_plant(xyz, rgb, frame)
    print(f"  {len(xyz)} points above the substrate and inside the turntable footprint")

    if color_filter == "vegetation":
        print(f"Filtering the cropped points by color (ExG > {color_filter_threshold})...")
        veg_mask = vegetation_color_mask(rgb, exg_threshold=color_filter_threshold)
        xyz, rgb = xyz[veg_mask], rgb[veg_mask]
        print(f"  {len(xyz)} vegetation-colored points kept")

    xyz_clean, rgb_clean = remove_statistical_outliers(xyz, rgb)
    xyz_clean, rgb_clean = keep_plant_clusters(xyz_clean, rgb_clean)
    before_density = len(xyz_clean)
    xyz_clean, rgb_clean = remove_sparse_points(
        xyz_clean, rgb_clean, density_fraction=min_point_density
    )
    print(
        f"  point cloud after cleanup: {len(xyz_clean)} / {len(xyz)} points kept "
        f"({before_density - len(xyz_clean)} dropped as substrate debris)"
    )

    root_index = find_root_point_on_ground(xyz_clean, frame) if len(xyz_clean) else None
    root_xyz = xyz_clean[root_index] if root_index is not None else None
    if root_xyz is not None:
        print(f"  root at {frame.heights(root_xyz[None])[0]:.4f} above the substrate top face")
    else:
        print("  no root point found -- skeleton will be unrooted (and will start at a leaf tip)")

    pointcloud_path = workdir / "pointcloud.ply"
    write_ply_vertices(
        pointcloud_path,
        {
            "x": xyz_clean[:, 0],
            "y": xyz_clean[:, 1],
            "z": xyz_clean[:, 2],
            "red": rgb_clean[:, 0],
            "green": rgb_clean[:, 1],
            "blue": rgb_clean[:, 2],
        },
    )
    print(f"  point cloud saved to {pointcloud_path} (raw COLMAP frame -- not yet aligned/scaled)")

    print("Estimating skeleton graph...")
    skeleton = build_skeleton_graph(
        xyz_clean,
        k_neighbors=k_neighbors,
        root_xyz=root_xyz,
        voxel_downsample_fraction=voxel_downsample_fraction,
        min_branch_fraction=min_branch_fraction,
        cover_radius_fraction=cover_radius_fraction,
    )
    root_note = "rooted" if skeleton.root_index is not None else "unrooted"
    print(f"  {skeleton.num_tips} tip(s), {skeleton.num_branch_points} branch point(s), {root_note}")

    render_path = workdir / "skeleton.png"
    plot_skeleton(skeleton, render_path, background_xyz=xyz_clean, background_rgb=rgb_clean)
    print(f"  render saved to {render_path}")

    summary = {
        "video": str(video_path) if video_path is not None else None,
        "images_dir": str(images_dir),
        "num_source_images": num_images,
        "num_images_registered": reconstruction.num_reg_images(),
        "num_points3D_raw": int(len(xyz)),
        "num_points3D_cleaned": int(len(xyz_clean)),
        "mean_reprojection_error_px": reconstruction.compute_mean_reprojection_error(),
        "num_tips": skeleton.num_tips,
        "num_branch_points": skeleton.num_branch_points,
        "root_index": skeleton.root_index,
        "keypoints": [
            {"index": int(idx), "kind": kind, "xyz": skeleton.points[idx].tolist()}
            for idx, kind in skeleton.keypoint_kinds.items()
        ],
        # Raw COLMAP frame (unitless, arbitrary orientation) -- see
        # pose-align-skeleton for solving+baking real-world alignment.
        "edges": [list(edge) for edge in skeleton.simplified_edges],
        "branch_polylines": [
            {
                "from_index": edge[0],
                "to_index": edge[1],
                "point_indices": path,
                # Smoothed (see smooth_polyline) -- the raw MST path zigzags
                # across the real points' scatter width instead of running
                # down the middle; point_indices above still names exactly
                # which raw points contributed, if needed.
                "points_xyz": smooth_polyline(skeleton.points[path]).tolist(),
            }
            for edge, path in skeleton.branch_polylines.items()
        ],
        "pointcloud_centroid_colmap": xyz_clean.mean(axis=0).tolist(),
        # Rig geometry, raw COLMAP frame -- `up` here is the same axis
        # pose-align-skeleton solves for, recorded so the crop and root
        # choice can be audited without re-deriving them.
        "turntable": {
            "up": frame.up.tolist(),
            "axis_point": frame.axis_point.tolist(),
            "substrate_normal": frame.substrate_normal.tolist(),
            "substrate_point": frame.substrate_point.tolist(),
            "substrate_tilt_degrees": frame.substrate_tilt_degrees,
            "footprint_radius": frame.radius,
            "root_height_above_substrate": (
                float(frame.heights(root_xyz[None])[0]) if root_xyz is not None else None
            ),
        },
    }
    summary_path = workdir / "skeleton.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary saved to {summary_path}")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Path to the rotation video")
    parser.add_argument(
        "--images", type=Path, help="Path to a folder of pre-captured still images (alternative to --video)"
    )
    parser.add_argument("--workdir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--num-frames", type=int, default=60, help="Frames to extract from the video (ignored with --images)"
    )
    parser.add_argument(
        "--mask-mode",
        choices=["vegetation", "black_background", "none"],
        default="none",
        help="Mask applied BEFORE matching, at the image level. Default 'none': for a "
        "cluttered scene (e.g. a hand holding the plant), masking away everything but the "
        "plant starves COLMAP of texture needed for pose estimation -- use --color-filter "
        "instead. 'black_background' is safe (and recommended) for a turntable + dark "
        "backdrop capture, since the background contributes no useful features anyway.",
    )
    parser.add_argument(
        "--color-filter",
        choices=["vegetation", "none"],
        default="vegetation",
        help="Filter applied AFTER reconstruction, to the 3D points by color (post-hoc, "
        "doesn't affect pose estimation). Default 'vegetation' keeps only green-ish points.",
    )
    parser.add_argument(
        "--color-filter-threshold",
        type=float,
        default=0.12,
        help="Excess Green Index cutoff for --color-filter=vegetation. Higher = stricter "
        "(fewer, more confidently-green points). Worth sweeping per-video.",
    )
    parser.add_argument("--k-neighbors", type=int, default=8, help="k for the skeleton's k-NN graph")
    parser.add_argument(
        "--min-branch-fraction",
        type=float,
        default=0.25,
        help="Shortest organ to accept, as a fraction of the longest root-to-tip distance "
        "along the plant. Main sensitivity knob: raise it if leaf blades split into extra "
        "branches, lower it if small leaves are missed. Because a tiny real leaf and a "
        "fragment of a big one are the same length, this cannot go arbitrarily low.",
    )
    parser.add_argument(
        "--cover-radius-fraction",
        type=float,
        default=0.08,
        help="How wide a swath around each traced branch is claimed as that organ's, as a "
        "fraction of the point cloud's bounding-box diagonal -- roughly a leaf half-width. "
        "Too small and one blade is re-extracted as several parallel branches; too large "
        "and a real leaf gets swallowed by its neighbor.",
    )
    parser.add_argument(
        "--min-point-density",
        type=float,
        default=0.30,
        help="Drop points whose neighbor count is below this fraction of the cloud's median "
        "neighbor count -- substrate grit and root hairs that pass the color filter. Worth "
        "raising if branches still crawl along the substrate before climbing a leaf, lowering "
        "if sparsely-reconstructed real leaves disappear.",
    )
    parser.add_argument(
        "--voxel-downsample-fraction",
        type=float,
        default=None,
        help="Before skeletonizing, thin the point cloud to one point per voxel of this "
        "fraction of the cloud's own bounding-box diagonal. Off by default: it existed to "
        "stop the old MST skeletonizer wandering across leaf surfaces, but the level-set "
        "skeletonizer wants dense surface coverage -- that is what puts each distance "
        "shell's centroid on the organ's real midline. Only worth enabling for very large "
        "clouds, where it trades midline accuracy for speed.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Threads for COLMAP SIFT feature extraction. Lower this (or --max-image-size) if "
        "extraction gets OOM-killed -- each thread holds a full decoded image in memory.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=2000,
        help="Downscale images to this max dimension before SIFT extraction (COLMAP's own "
        "default is unbounded, which is a common OOM cause on phone-camera-sized photos).",
    )
    parser.add_argument(
        "--reuse-sparse",
        action="store_true",
        help="Skip COLMAP and reload the model saved at <workdir>/sparse/best by a previous "
        "run. The reconstruction is the slow part while everything after it takes seconds, so "
        "use this when re-tuning --min-branch-fraction/--cover-radius-fraction on a capture "
        "you have already reconstructed. Ignored (with a note) if that folder is absent.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Run SIFT feature extraction on the GPU instead of CPU threads (--num-threads is "
        "then ignored). Requires a CUDA-enabled pycolmap build, e.g. `pip install pycolmap-cuda` "
        "plus `pip install nvidia-cuda-runtime-cu12` if `import pycolmap` complains about "
        "libcudart.so.12.",
    )
    args = parser.parse_args(argv)
    if (args.video is None) == (args.images is None):
        parser.error("Provide exactly one of --video or --images")

    run(
        video_path=args.video,
        images_dir=args.images,
        workdir=args.workdir,
        num_frames=args.num_frames,
        mask_mode=args.mask_mode,
        color_filter=args.color_filter,
        color_filter_threshold=args.color_filter_threshold,
        k_neighbors=args.k_neighbors,
        min_branch_fraction=args.min_branch_fraction,
        cover_radius_fraction=args.cover_radius_fraction,
        min_point_density=args.min_point_density,
        voxel_downsample_fraction=args.voxel_downsample_fraction,
        num_threads=args.num_threads,
        max_image_size=args.max_image_size,
        use_gpu=args.use_gpu,
        reuse_sparse=args.reuse_sparse,
    )


if __name__ == "__main__":
    main()
