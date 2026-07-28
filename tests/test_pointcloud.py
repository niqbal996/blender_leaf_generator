import numpy as np

from leaf_generator.skeleton.pointcloud import (
    filter_by_vegetation_color,
    keep_largest_cluster,
    remove_statistical_outliers,
)


def test_filter_by_vegetation_color_keeps_green_drops_skin():
    xyz = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
    rgb = np.array(
        [
            [40, 160, 60],  # green, G-dominant -> vegetation
            [200, 120, 80],  # skin-tone, clearly R-dominant -> not vegetation
            [80, 80, 80],  # neutral gray -> not vegetation
        ],
        dtype=np.uint8,
    )

    xyz_kept, rgb_kept = filter_by_vegetation_color(xyz, rgb)

    assert len(xyz_kept) == 1
    assert np.array_equal(xyz_kept[0], xyz[0])


def test_remove_statistical_outliers_drops_far_point():
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.1, size=(50, 3))
    outlier = np.array([[100.0, 100.0, 100.0]])
    xyz = np.vstack([cluster, outlier])

    xyz_kept, _ = remove_statistical_outliers(xyz, k=8, std_ratio=2.0)

    assert len(xyz_kept) == 50
    assert not np.any(np.all(xyz_kept == outlier, axis=1))


def test_keep_largest_cluster_drops_small_floating_group():
    rng = np.random.default_rng(1)
    main = rng.normal(scale=0.1, size=(40, 3))
    floater = rng.normal(loc=[20, 20, 20], scale=0.1, size=(5, 3))
    xyz = np.vstack([main, floater])

    xyz_kept, _ = keep_largest_cluster(xyz)

    # Allow a rare Gaussian-tail point to also fall outside the auto radius;
    # what matters is the floater group is gone and most of main is kept.
    assert 38 <= len(xyz_kept) <= 40
