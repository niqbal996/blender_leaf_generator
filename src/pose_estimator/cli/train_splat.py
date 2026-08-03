"""CLI: train a 3D Gaussian Splat from a plant capture already processed by
`pose-estimate-skeleton` (needs that script's `sparse/best/` + `images/`
in the same --workdir).

    pose-train-splat --workdir out/plant1/ --iterations 30000

Writes `splat.ply` into --workdir, in the same raw COLMAP frame as
`skeleton.json`/`pointcloud.ply` -- see `pose-align-skeleton` to bake in
real-world scale/orientation before viewing it in Blender.

Requires the "skeleton" extra (for the COLMAP reconstruction) and the
"splat" extra (torch + gsplat): pip install -e ".[skeleton,splat]", plus a
CUDA-matching torch build -- see the README.
"""

import argparse
from pathlib import Path
from typing import Optional

from pose_estimator.gaussian_splat import (
    init_gaussians_from_pointcloud,
    load_training_views,
    train,
    write_gaussian_ply,
)
from pose_estimator.pointcloud import (
    extract_xyz_rgb,
    keep_plant_clusters,
    remove_statistical_outliers,
)


def run(
    workdir: Path,
    iterations: int = 30000,
    strategy: str = "default",
    sh_degree: int = 3,
    image_downsample_factor: int = 1,
) -> None:
    import pycolmap

    sparse_best = workdir / "sparse" / "best"
    if not sparse_best.exists():
        raise FileNotFoundError(
            f"{sparse_best} not found -- run pose-estimate-skeleton on this "
            "--workdir first (this trainer reuses its COLMAP reconstruction)."
        )

    print(f"Loading COLMAP reconstruction from {sparse_best}...")
    reconstruction = pycolmap.Reconstruction(str(sparse_best))

    print("Extracting + cleaning the point cloud for Gaussian initialization...")
    # Deliberately skip the skeleton pipeline's vegetation color filter here --
    # a splat should represent the whole visible plant (stems, soil-colored
    # bits, etc.), not the skeleton-optimized green-only subset.
    xyz, rgb = extract_xyz_rgb(reconstruction)
    xyz, rgb = remove_statistical_outliers(xyz, rgb)
    xyz, rgb = keep_plant_clusters(xyz, rgb)
    print(f"  seeding {len(xyz)} Gaussians from the cleaned point cloud")

    gaussians = init_gaussians_from_pointcloud(xyz, rgb, sh_degree=sh_degree)

    print("Undistorting images for training (cached under workdir/undistorted/)...")
    views = load_training_views(workdir, downsample_factor=image_downsample_factor)
    print(f"  {len(views)} training views")

    print(f"Training ({strategy} strategy, {iterations} iterations)...")
    trained = train(gaussians, views, iterations=iterations, strategy=strategy, sh_degree=sh_degree)

    splat_path = workdir / "splat.ply"
    write_gaussian_ply(splat_path, trained)
    print(f"  splat saved to {splat_path} (raw COLMAP frame -- not yet aligned/scaled)")
    print("  run pose-align-skeleton next to bake in real-world scale/orientation.")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", required=True, type=Path, help="Same --workdir passed to pose-estimate-skeleton"
    )
    parser.add_argument("--iterations", type=int, default=30000, help="Training iterations")
    parser.add_argument(
        "--strategy",
        choices=["default", "mcmc"],
        default="default",
        help="Densification/pruning strategy (gsplat.DefaultStrategy or gsplat.MCMCStrategy)",
    )
    parser.add_argument("--sh-degree", type=int, default=3, help="Max spherical harmonics degree")
    parser.add_argument(
        "--image-downsample-factor",
        type=int,
        default=1,
        help="Train against images downsampled by this factor (2-4 speeds up training substantially "
        "with modest quality loss -- mirrors gsplat's own examples' --data_factor)",
    )
    args = parser.parse_args(argv)

    run(
        workdir=args.workdir,
        iterations=args.iterations,
        strategy=args.strategy,
        sh_degree=args.sh_degree,
        image_downsample_factor=args.image_downsample_factor,
    )


if __name__ == "__main__":
    main()
