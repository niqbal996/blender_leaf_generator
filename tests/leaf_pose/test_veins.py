"""What the vein finder can do, and -- as importantly -- what it cannot.

`veins.py`'s docstring makes three claims that were measured rather than
assumed. These pin all three, including the negative one: a test suite that
only asserts successes would let the fine-vein case quietly start returning
texture again, which is exactly the failure this feature was tightened to
avoid.
"""

import cv2
import numpy as np

from leaf_pose import midrib, veins


def blade(height=420, width=260):
    ys, xs = np.mgrid[0:height, 0:width]
    return ((xs - 130) ** 2 / 90.0 ** 2 + (ys - 210) ** 2 / 195.0 ** 2) < 1


def surface(mask, rib_sigma=6.0, basal_veins=False, fine_veins=False,
            granules=0.0, seed=0):
    """A height field: a midrib, optionally veins, optionally mealy granules."""
    height, width = mask.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    field = np.exp(-((xs - 130) ** 2) / (2 * rib_sigma ** 2)) * 6.0

    if basal_veins:
        # Two strong veins leaving the midrib near the base at about 45
        # degrees -- the pair that runs out into a lobed leaf's basal lobes.
        for side in (-1, 1):
            along = (ys - 330) - side * (xs - 130)
            field += np.exp(-(along ** 2) / (2 * 9.0 ** 2)) * 5.0 * (ys > 180)

    if fine_veins:
        for row in range(70, 360, 34):
            for side in (-1, 1):
                along = (ys - row) * 0.9 - side * (xs - 130)
                field += np.exp(-(along ** 2) / (2 * 7.0 ** 2)) * 2.2 * (
                    np.abs(xs - 130) < 80)

    if granules:
        rng = np.random.default_rng(seed)
        speckle = rng.normal(0.0, 1.0, mask.shape).astype(np.float32)
        field += cv2.GaussianBlur(speckle, (0, 0), 2.0) * granules
    return field * mask


def normals_of(field):
    dy, dx = np.gradient(field.astype(np.float32))
    stacked = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
    return stacked / np.linalg.norm(stacked, axis=-1, keepdims=True)


def photograph(mask, granules=0.0, seed=0):
    """A flat image with no vein contrast -- only granular albedo, if any."""
    image = np.zeros(mask.shape + (3,), np.float32)
    image[mask] = 0.30
    if granules:
        rng = np.random.default_rng(seed + 1)
        speckle = cv2.GaussianBlur(
            rng.normal(0.0, 1.0, mask.shape).astype(np.float32), (0, 0), 2.0)
        image[mask] += (speckle * granules)[mask][:, None]
    return image


def test_strong_basal_veins_are_found_from_the_normals():
    mask = blade()
    field = normals_of(surface(mask, basal_veins=True, granules=0.15))
    found = veins.find_veins(mask, photograph(mask, granules=0.03),
                             midrib.fit_midrib(mask, num_samples=48),
                             normals=field)

    assert found, "the two basal veins should be found"
    # They leave at about 45 degrees; anything near 0 is the midrib's flank
    # or the margin rim, which MIN_INSERTION_ANGLE is there to exclude.
    assert all(v.insertion_angle_deg >= veins.MIN_INSERTION_ANGLE for v in found)
    assert 25.0 < float(np.median([v.insertion_angle_deg for v in found])) < 70.0


def test_the_photograph_alone_finds_nothing_when_veins_are_only_relief():
    """The flat image averages twelve lights, so a vein's shading is gone."""
    mask = blade()
    found = veins.find_veins(mask, photograph(mask, granules=0.03),
                             midrib.fit_midrib(mask, num_samples=48))
    assert len(found) <= 1


def test_nothing_is_reported_for_a_leaf_that_has_no_veins():
    """The case that fails silently: a midrib flank or the margin rim read as
    a vein. Both run alongside the midrib, so both are cut on angle."""
    mask = blade()
    for seed in range(3):
        field = normals_of(surface(mask, rib_sigma=14.0, granules=0.2, seed=seed))
        found = veins.find_veins(mask, photograph(mask), 
                                 midrib.fit_midrib(mask, num_samples=48),
                                 normals=field)
        assert found == [], f"invented {len(found)} veins on a leaf with none"


def test_fine_venation_under_surface_texture_is_not_claimed():
    """The documented limit. A herringbone of fine veins buried in granular
    texture is not recoverable here, and the honest behaviour is to return
    almost nothing rather than a plausible-looking network."""
    mask = blade()
    field = normals_of(surface(mask, fine_veins=True, granules=0.2))
    found = veins.find_veins(mask, photograph(mask, granules=0.04),
                             midrib.fit_midrib(mask, num_samples=48),
                             normals=field)
    # Not asserting zero -- asserting it does not manufacture a network.
    assert len(found) <= 3


def test_a_candidate_reaching_no_midrib_is_not_reported():
    mask = blade()
    field = normals_of(surface(mask))
    found = veins.find_veins(mask, photograph(mask),
                             midrib.fit_midrib(mask, num_samples=48),
                             normals=field)
    assert found == []
