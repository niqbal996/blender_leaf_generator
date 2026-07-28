"""CLI: train a 3D Gaussian Splat from a plant capture already processed by
`estimate_plant_skeleton.py` (needs that script's `sparse/best/` + `images/`
in the same --workdir).

    python train_gaussian_splat.py --workdir out/plant1/ --iterations 30000

Writes `splat.ply` into --workdir, in the same raw COLMAP frame as
`skeleton.json`/`pointcloud.ply` -- see `align_plant_skeleton.py` to bake in
real-world scale/orientation before viewing it in Blender.

Requires the "skeleton" extra (for the COLMAP reconstruction) and the
"splat" extra (torch + gsplat): pip install -e ".[skeleton,splat]", plus a
CUDA-matching torch build -- see the README.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from leaf_generator.skeleton.gaussian_splat import (  # noqa: E402
    init_gaussians_from_pointcloud,
    load_training_views,
    train,
    write_gaussian_ply,
)
from leaf_generator.skeleton.pointcloud import (  # noqa: E402
    extract_xyz_rgb,
    keep_largest_cluster,
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
            f"{sparse_best} not found -- run estimate_plant_skeleton.py on this "
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
    xyz, rgb = keep_largest_cluster(xyz, rgb)
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
    print("  run align_plant_skeleton.py next to bake in real-world scale/orientation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", required=True, type=Path, help="Same --workdir passed to estimate_plant_skeleton.py"
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
    args = parser.parse_args()

    run(
        workdir=args.workdir,
        iterations=args.iterations,
        strategy=args.strategy,
        sh_degree=args.sh_degree,
        image_downsample_factor=args.image_downsample_factor,
    )
