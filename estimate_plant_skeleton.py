"""CLI: estimate a plant's stem/branch/tip skeleton from a rotation video or
an already-captured folder of still images.

    python estimate_plant_skeleton.py --video path/to/video.MOV --workdir out/
    python estimate_plant_skeleton.py --images path/to/stills/ --workdir out/

Pipeline: extract frames from video, or use a still-image folder as-is ->
COLMAP sparse reconstruction (unmasked, by default -- see --mask-mode) ->
filter 3D points by color -> point-cloud cleanup -> MST-based skeleton graph
-> render + JSON summary.

This is a research prototype (see src/leaf_generator/skeleton/__init__.py
and the README's "Plant skeleton from video" section) -- inspect the point
cloud render alongside the skeleton, don't trust the graph blindly, since
the whole pipeline lives or dies on how clean the source images are.

Requires the "skeleton" extra: pip install -e ".[skeleton]"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from leaf_generator.skeleton.frames import extract_frames  # noqa: E402
from leaf_generator.skeleton.masking import write_colmap_masks  # noqa: E402
from leaf_generator.skeleton.pointcloud import (  # noqa: E402
    extract_xyz_rgb,
    filter_by_vegetation_color,
    keep_largest_cluster,
    remove_statistical_outliers,
)
from leaf_generator.skeleton.ply_io import write_ply_vertices  # noqa: E402
from leaf_generator.skeleton.reconstruction import build_sparse_reconstruction  # noqa: E402
from leaf_generator.skeleton.skeletonize import build_skeleton_graph  # noqa: E402
from leaf_generator.skeleton.visualize import plot_skeleton  # noqa: E402


def run(
    workdir: Path,
    video_path: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    num_frames: int = 60,
    mask_mode: str = "none",
    color_filter: str = "vegetation",
    color_filter_threshold: float = 0.12,
    k_neighbors: int = 8,
    min_branch_fraction: float = 0.03,
    num_threads: int = 4,
    max_image_size: int = 2000,
    use_gpu: bool = False,
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

    if color_filter == "vegetation":
        print(f"Filtering points by color (ExG > {color_filter_threshold})...")
        xyz, rgb = filter_by_vegetation_color(xyz, rgb, exg_threshold=color_filter_threshold)
        print(f"  {len(xyz)} vegetation-colored points kept")

    xyz_clean, rgb_clean = remove_statistical_outliers(xyz, rgb)
    xyz_clean, rgb_clean = keep_largest_cluster(xyz_clean, rgb_clean)
    print(f"  point cloud after cleanup: {len(xyz_clean)} / {len(xyz)} points kept")

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
        xyz_clean, k_neighbors=k_neighbors, min_branch_length_fraction=min_branch_fraction
    )
    print(f"  {skeleton.num_tips} tip(s), {skeleton.num_branch_points} branch point(s)")

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
        "keypoints": [
            {"index": int(idx), "kind": kind, "xyz": skeleton.points[idx].tolist()}
            for idx, kind in skeleton.keypoint_kinds.items()
        ],
        # Raw COLMAP frame (unitless, arbitrary orientation) -- see
        # align_plant_skeleton.py for solving+baking real-world alignment.
        "edges": [list(edge) for edge in skeleton.simplified_edges],
        "branch_polylines": [
            {
                "from_index": edge[0],
                "to_index": edge[1],
                "point_indices": path,
                "points_xyz": skeleton.points[path].tolist(),
            }
            for edge, path in skeleton.branch_polylines.items()
        ],
        "pointcloud_centroid_colmap": xyz_clean.mean(axis=0).tolist(),
    }
    summary_path = workdir / "skeleton.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary saved to {summary_path}")


if __name__ == "__main__":
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
        "--min-branch-fraction", type=float, default=0.03, help="Prune spurs shorter than this fraction of total skeleton length"
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
        "--use-gpu",
        action="store_true",
        help="Run SIFT feature extraction on the GPU instead of CPU threads (--num-threads is "
        "then ignored). Requires a CUDA-enabled pycolmap build, e.g. `pip install pycolmap-cuda` "
        "plus `pip install nvidia-cuda-runtime-cu12` if `import pycolmap` complains about "
        "libcudart.so.12.",
    )
    args = parser.parse_args()
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
        num_threads=args.num_threads,
        max_image_size=args.max_image_size,
        use_gpu=args.use_gpu,
    )
