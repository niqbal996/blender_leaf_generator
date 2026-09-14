"""Multi-view depth agreement, shared by the exporters that need it.

A depth error moves a point *along the ray it was seen on*, which leaves its
projection in the view it came from exactly where it was. So a per-view cloud
always looks right in its own view and can only be judged by the others: the
test is whether another view's depth map puts a surface at the same distance.

Measured on thistle3, which is why this exists: with the fusion below, VGGT-
Omega's points are corroborated by a median of 25 of 27 views. MapAnything's
export, which applies no geometric fusion at all, manages 16 -- so a third of
the views disagree about where any given point is, and the cloud reads as a
thick noisy shell rather than a plant.
"""

from __future__ import annotations

import numpy as np


def agreement_matrix(
    candidates: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    masks: np.ndarray | None,
    tolerance: float,
    absolute: float | None = None,
) -> np.ndarray:
    """Which views corroborate each candidate point.

    Args:
        candidates: (N, 3) world points.
        depths: (V, H, W) per-view depth, NaN where there is none.
        intrinsics: (V, 3, 3).
        extrinsics: (V, 3, 4) camera-from-world.
        masks: (V, H, W) bool, or None to accept any pixel.
        tolerance: agreement band, as a fraction of the candidate's depth.
        absolute: agreement band in scene units. Overrides `tolerance` when
            given, and is what callers should prefer -- see below.

    Returns:
        (N, V) bool: view v places a surface where candidate n claims one.

    `tolerance` is a fraction of the *distance to the camera*, which is not a
    property of the plant: move the tripod back and the band widens while the
    leaf it is meant to resolve does not. Measured on thistle3, where VGGT put
    the plant 1.0 units away and 0.53 units across, the 0.01 default worked out
    at 1.9% of the plant -- about ten leaf thicknesses -- so views half a leaf
    apart counted as agreeing. The two exporters also defaulted differently
    (0.01 and 0.02) on scenes of different scale, which left MapAnything held
    to a roughly 2x stricter standard than VGGT-Omega for no stated reason.
    `absolute` is a band in scene units, so a caller can set it from the
    plant's own size and have it mean the same thing on every capture.
    """
    num_views, height, width = depths.shape
    agrees = np.zeros((len(candidates), num_views), dtype=bool)
    for view in range(num_views):
        rotation, translation = extrinsics[view][:3, :3], extrinsics[view][:3, 3]
        camera_points = candidates @ rotation.T + translation
        z = camera_points[:, 2]
        in_front = z > 1e-6
        safe = np.where(in_front, z, 1.0)
        u = camera_points[:, 0] / safe * intrinsics[view][0, 0] + intrinsics[view][0, 2]
        v = camera_points[:, 1] / safe * intrinsics[view][1, 1] + intrinsics[view][1, 2]
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        inside = in_front & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        ui = np.clip(ui, 0, width - 1)
        vi = np.clip(vi, 0, height - 1)
        observed = depths[view][vi, ui]
        if masks is not None:
            inside &= masks[view][vi, ui]
        band = absolute if absolute is not None else tolerance * np.maximum(z, 1e-9)
        agrees[:, view] = inside & np.isfinite(observed) & (np.abs(observed - z) <= band)
    return agrees


def reprojected_pixels(
    candidates: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    shape_hw: tuple,
) -> tuple:
    """Candidate points projected into one view: (u, v, inside-the-image)."""
    height, width = shape_hw
    rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
    camera_points = candidates @ rotation.T + translation
    z = camera_points[:, 2]
    in_front = z > 1e-6
    safe = np.where(in_front, z, 1.0)
    u = np.round(camera_points[:, 0] / safe * intrinsic[0, 0] + intrinsic[0, 2]).astype(np.int64)
    v = np.round(camera_points[:, 1] / safe * intrinsic[1, 1] + intrinsic[1, 2]).astype(np.int64)
    inside = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.clip(u, 0, width - 1), np.clip(v, 0, height - 1), inside


def plant_extent(points: np.ndarray, masks: np.ndarray | None,
                 depths: np.ndarray | None = None) -> float:
    """How big the subject is, in scene units, before anything is fused.

    The bounding diagonal of the masked-in points, taken between the 1st and
    99th percentile per axis so a handful of flyers cannot inflate it. This is
    the length every tolerance below should be expressed as a fraction of: it
    is a property of the plant, where a distance to the camera is a property of
    where the tripod stood.
    """
    if masks is None:
        sample = points.reshape(-1, 3)
    else:
        sample = points[masks]
    if depths is not None:
        finite = np.isfinite(depths[masks] if masks is not None else depths.reshape(-1))
        sample = sample[finite]
    sample = sample[np.isfinite(sample).all(axis=1)]
    if len(sample) < 2:
        return 0.0
    low = np.percentile(sample, 1, axis=0)
    high = np.percentile(sample, 99, axis=0)
    return float(np.linalg.norm(high - low))


def merge_duplicates(xyz: np.ndarray, tracks: list, radius: float) -> tuple:
    """Collapse points that describe the same bit of surface into one.

    Every exporter here walks the views and writes a point for each pixel a
    view is confident about and the others corroborate. Corroboration is the
    whole test, so a point the other views *do* confirm is written once per
    view that saw it -- and the copies land wherever each view's own depth put
    them, which is anywhere inside the agreement band.

    Measured on thistle3's VGGT-Omega export before this existed: 357,352
    points, mean track length 10.3, and collapsing at 0.5% of the plant's
    extent left 35,261 -- a redundancy of 10.1x, which is the track length. The
    cloud was ~35k real surface samples with ten copies each, smeared across
    the band. Local flatness came out at 0.726 against 0.390 for the COLMAP
    baseline: past 0.577, the isotropic value, so the leaves were not surfaces
    at all but tubes of points.

    Averaging the copies is not only deduplication. Each is an independent
    estimate of the same surface, so their mean is a better estimate than any
    one of them -- the per-view depth noise is what the spread is made of.

    Points are binned on a grid of `radius`, which merges within a cell but not
    across a boundary. That is the cheap approximation and it is the right one
    here: the alternative, clustering, costs a neighbour search over hundreds
    of thousands of points to move a small fraction of them one cell over.
    Tracks are unioned, so a merged point keeps every observation of it and
    bundle adjustment still has something to work with.
    """
    if len(xyz) == 0:
        return xyz, tracks
    keys = np.floor(xyz / max(radius, 1e-12)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    merged_xyz, merged_tracks = [], []
    for group in np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1):
        merged_xyz.append(xyz[group].mean(axis=0))
        seen, track = set(), []
        for member in group:
            for view, u, v in tracks[member]:
                if view not in seen:
                    seen.add(view)
                    track.append((view, u, v))
        merged_tracks.append(track)
    return np.asarray(merged_xyz), merged_tracks


def sheet_flatness(points: np.ndarray, radius: float, sample: int = 2000,
                   seed: int = 0) -> dict:
    """Is this cloud made of surfaces, or of fog? Lower is flatter.

    For a sample of points, fit an ellipsoid to the neighbours within `radius`
    and report sigma_min / sigma_max. A leaf is a sheet, so its neighbourhoods
    should be wide in two directions and near-zero in the third. Read against
    two fixed marks:

      0.577  isotropic -- the neighbourhood is a ball, so there is no surface
             here at all, only a cloud of points
      0.25   P4b's `surface_is_thin` target, the point at which P5 sees sheets

    Measured on thistle3 at radius = 1% of extent: the COLMAP P4b surface 0.390,
    VGGT-Omega 0.726 and MapAnything 0.721. Both learned clouds sat *past*
    isotropic, which is what "dense but thick" means numerically and is the
    reason this is now reported on every run rather than measured by hand
    afterwards. `radius` matters: at a fixed neighbour *count* a denser cloud
    is measured over a smaller ball, and the branches stop being comparable.
    """
    from scipy.spatial import cKDTree

    if len(points) < 16:
        return {"flatness": None, "thickness": None, "neighbours": 0, "radius": radius}
    tree = cKDTree(points)
    rng = np.random.default_rng(seed)
    index = rng.choice(len(points), min(sample, len(points)), replace=False)
    flat, thick, counts = [], [], []
    for i in index:
        neighbours = tree.query_ball_point(points[i], radius)
        if len(neighbours) < 12:
            continue
        centred = points[neighbours] - points[neighbours].mean(axis=0)
        sigma = np.sqrt(np.clip(np.linalg.eigvalsh(centred.T @ centred / len(neighbours)), 0, None))
        flat.append(sigma[0] / max(sigma[2], 1e-12))
        thick.append(sigma[0])
        counts.append(len(neighbours))
    if not flat:
        return {"flatness": None, "thickness": None, "neighbours": 0, "radius": radius}
    return {
        "flatness": float(np.median(flat)),
        "thickness": float(np.median(thick)),
        "neighbours": float(np.median(counts)),
        "radius": float(radius),
        "sampled": len(flat),
    }


def report_flatness(points: np.ndarray, extent: float, label: str = "") -> dict:
    """`sheet_flatness` at 1% of extent, printed the way a QC line reads."""
    stats = sheet_flatness(points, radius=0.01 * extent)
    if stats["flatness"] is None:
        print(f"  flatness: too few points to measure{label}")
        return stats
    verdict = ("sheets" if stats["flatness"] <= 0.25 else
               "thick, but still anisotropic" if stats["flatness"] <= 0.577 else
               "NOT A SURFACE -- rounder than isotropic")
    print(f"  flatness {stats['flatness']:.3f} at r=1% of extent "
          f"({stats['neighbours']:.0f} neighbours): {verdict}")
    print(f"    sheet thickness {stats['thickness'] / max(extent, 1e-12) * 100:.3f}% of extent "
          f"(P4b target <=0.25 flatness; 0.577 = isotropic)")
    return stats
