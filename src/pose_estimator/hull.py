"""P4 (part 1) -- visual hull by silhouette carving.

The hull is built first, before any splatting, because it is the piece that
removes the old pipeline's dependence on hand-tuned per-plant thresholds. It
is deterministic, needs no training, and is strongest exactly where
photometric methods are weakest: thin stems and petioles, which carve
perfectly well from silhouettes but reconstruct poorly from matching.

What it is *not* is the final surface. Silhouettes cannot see concavities --
a cupped leaf carves as if it were flat -- and the hull systematically
over-estimates leaf thickness. It is a bound, and it is used as a bound: the
splatting stage supplies the concave detail, and the hull rejects anything
that strays outside what every view agrees is occupied.

Carving is coarse-to-fine. A dense 512^3 grid is 134M voxels and projecting
all of them into ~96 cameras is both slow and memory-hostile, but the hull
occupies a tiny fraction of its bounding box, so subdividing only surviving
voxels reaches the same resolution for a small fraction of the work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


@dataclass
class CarveCamera:
    """One view's projection, plus its silhouette."""

    K: np.ndarray  # (3, 3)
    world_to_camera: np.ndarray  # (4, 4)
    mask: np.ndarray  # bool (H, W), True = subject
    name: str
    occluder: Optional[np.ndarray] = None  # bool (H, W), True = something in front

    def project(self, points_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """World points -> (pixel coords Nx2, in-front-of-camera flag)."""
        R = self.world_to_camera[:3, :3]
        t = self.world_to_camera[:3, 3]
        cam = points_world @ R.T + t
        depth = cam[:, 2]
        valid = depth > 1e-6
        safe_depth = np.where(valid, depth, 1.0)
        pixels = (cam[:, :2] / safe_depth[:, None]) @ self.K[:2, :2].T + self.K[:2, 2]
        return pixels, valid


def load_carve_cameras(
    reconstruction,
    mask_dir: Union[str, Path],
    dilate_px: int = 2,
    occluder_dir: Optional[Union[str, Path]] = None,
) -> List[CarveCamera]:
    """Build carve cameras from a COLMAP reconstruction + P2 plant masks.

    Masks are dilated slightly. Carving is an intersection, so it is
    *unforgiving*: a silhouette that is one pixel too tight in a single view
    slices real surface off the hull permanently, and there are ~96 chances
    for that to happen. Dilation biases the hull outward, which is the
    correct direction of error for something used as an upper bound.
    """
    mask_dir = Path(mask_dir)
    occluder_dir = Path(occluder_dir) if occluder_dir is not None else None
    kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8) if dilate_px > 0 else None

    cameras: List[CarveCamera] = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        cam = reconstruction.cameras[image.camera_id]

        mask_path = mask_dir / f"{Path(image.name).stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        binary = mask > 127
        if kernel is not None:
            binary = cv2.dilate(binary.astype(np.uint8), kernel).astype(bool)

        occluder = None
        if occluder_dir is not None:
            raw = cv2.imread(str(occluder_dir / f"{Path(image.name).stem}.png"),
                             cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                occluder = raw > 127
                if kernel is not None:
                    # Dilated by the same amount as the silhouette, so the rim
                    # where the two masks disagree is excused rather than
                    # counted as evidence against the voxel behind it.
                    occluder = cv2.dilate(occluder.astype(np.uint8), kernel).astype(bool)
                occluder &= ~binary      # plant in front of the tool is still plant

        world_to_camera = np.eye(4)
        world_to_camera[:3, :] = image.cam_from_world().matrix()

        cameras.append(
            CarveCamera(
                K=np.asarray(cam.calibration_matrix(), dtype=np.float64),
                world_to_camera=world_to_camera,
                mask=binary,
                name=image.name,
                occluder=occluder,
            )
        )
    return cameras


def _vote(
    points: np.ndarray, cameras: Sequence[CarveCamera], chunk: int = 500_000
) -> Tuple[np.ndarray, np.ndarray]:
    """Per point: (views that could observe it, views that place it in-silhouette).

    Both counts are needed, and conflating them is a real trap. Treating an
    off-image projection as tacit agreement seems safe -- it avoids clipping
    the hull to the least generous frame -- but it silently keeps every voxel
    that falls outside *all* the frustums, since those agree with everyone by
    default. Verified against a synthetic sphere: that rule recovered a radius
    of 1.14 for a true 0.30.

    So "was it observable" and "did it look occupied" are counted separately,
    and the caller requires a voxel to be observable in nearly every view
    *and* in-silhouette in nearly every view that observed it.
    """
    observed = np.zeros(len(points), dtype=np.int32)
    judged = np.zeros(len(points), dtype=np.int32)
    inside = np.zeros(len(points), dtype=np.int32)

    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        block_observed = np.zeros(len(block), dtype=np.int32)
        block_judged = np.zeros(len(block), dtype=np.int32)
        block_inside = np.zeros(len(block), dtype=np.int32)

        for camera in cameras:
            pixels, in_front = camera.project(block)
            height, width = camera.mask.shape
            x = np.round(pixels[:, 0]).astype(np.int64)
            y = np.round(pixels[:, 1]).astype(np.int64)

            testable = (x >= 0) & (x < width) & (y >= 0) & (y < height) & in_front
            xi = np.clip(x, 0, width - 1)
            yi = np.clip(y, 0, height - 1)

            block_observed += testable
            # Behind the holder we learn nothing. Counting a hidden voxel as
            # "not in the silhouette" is what deleted the root: the pliers
            # cross in front of it for most of the orbit, so it was absent
            # from 73% of the masks and lost a 86%-of-views vote it was never
            # given a fair chance at. Occluded views leave the denominator
            # instead of voting against.
            seen = testable
            if camera.occluder is not None:
                seen = seen & ~camera.occluder[yi, xi]
            block_judged += seen
            block_inside += seen & camera.mask[yi, xi]

        observed[start : start + chunk] = block_observed
        judged[start : start + chunk] = block_judged
        inside[start : start + chunk] = block_inside

    return observed, judged, inside


def carve(
    cameras: Sequence[CarveCamera],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    resolution: int = 256,
    coarse_resolution: int = 64,
    min_inside_fraction: float = 0.86,
    min_observed_fraction: float = 0.9,
    min_judged_views: int = 8,
    min_judged_fraction: float = 0.5,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Coarse-to-fine silhouette carve.

    Returns (occupied voxel centres, voxel size, refined bounds_min).

    `min_inside_fraction` is the share of observing views that must place a
    voxel inside the silhouette. 1.0 is the textbook visual hull and is far
    too brittle here: any structure that is occluded, or that a single frame's
    mask clipped, is deleted with no recourse. On these captures the exposed
    root is thin *and* hidden behind the pliers for part of the rotation, so a
    strict intersection removes it entirely even though it is plainly present
    in most masks.

    The default was fit by sweeping the threshold on both specimens and
    scoring reprojected hull against input silhouette. Mean IoU peaks at
    0.86 for both (0.785 on DSC_0009, 0.838 on DSC_0010) and falls away on
    either side -- tighter loses the root, looser inflates the whole hull.
    Expressed as a fraction rather than a view count so it stays meaningful
    when the number of frames changes.

    `min_observed_fraction` requires a voxel to actually fall inside most
    cameras' images before it can be kept at all, which is what confines the
    hull to the intersection of the frustums. It counts *frustum* membership,
    deliberately including views where the voxel was hidden -- being behind
    the holder is not the same as being outside the camera, and conflating
    them would delete anything the tool covers for most of the orbit, which
    is the failure this occlusion handling exists to fix.

    `min_judged_fraction` is the floor on how much of a voxel's evidence may
    be excused. Without it, excusing occluded views builds a solid block in
    the holder's shadow: the volume inside the pliers is hidden in almost
    every frame, so almost nothing votes against it and the little that does
    is outvoted. Measured on thistle1, that was 221,484 voxels -- 32% of the
    hull -- sitting inside the tool, while the actual root stayed missing.
    Requiring half the frustum views to be unoccluded separates the two
    cleanly there: the plier shadow is judged in 9% of its views, the root
    column in 78%.

    `min_judged_views` is the floor on unoccluded views. Excusing occluded
    views makes the in-silhouette fraction a vote among fewer voters, and two
    voters agreeing means very little; a silhouette intersection needs views
    spread around the orbit before it constrains a shape at all. Absolute
    rather than a fraction because it expresses "enough evidence to say
    anything", which does not scale with how long the capture was.
    """
    min_observed = max(int(np.ceil(min_observed_fraction * len(cameras))), 1)

    def survives(points: np.ndarray) -> np.ndarray:
        observed, judged, inside = _vote(points, cameras)
        enough = ((observed >= min_observed)
                  & (judged >= min_judged_views)
                  & (judged >= min_judged_fraction * np.maximum(observed, 1)))
        return enough & (inside >= min_inside_fraction * np.maximum(judged, 1))

    # --- coarse pass: find where the hull actually lives ---
    grid = _grid_points(bounds_min, bounds_max, coarse_resolution)
    keep = survives(grid)
    if not keep.any():
        raise RuntimeError(
            "Silhouette carving produced an empty hull. The masks and the poses probably "
            "disagree -- check p3/diag/camera_orbit.png and that the P2 masks belong to these frames."
        )

    occupied = grid[keep]
    span = (bounds_max - bounds_min) / coarse_resolution
    bounds_min = occupied.min(axis=0) - 2 * span
    bounds_max = occupied.max(axis=0) + 2 * span

    # --- refine by repeated subdivision of survivors only ---
    level_resolution = coarse_resolution
    points = _grid_points(bounds_min, bounds_max, level_resolution)
    points = points[survives(points)]
    voxel = float(np.max((bounds_max - bounds_min) / level_resolution))

    while level_resolution < resolution:
        level_resolution *= 2
        voxel /= 2.0
        points = _subdivide(points, voxel)
        points = points[survives(points)]
        if len(points) == 0:
            raise RuntimeError(f"Hull vanished while refining to {level_resolution}^3")

    return points, voxel, bounds_min


def _grid_points(bounds_min: np.ndarray, bounds_max: np.ndarray, resolution: int) -> np.ndarray:
    axes = [
        np.linspace(bounds_min[i], bounds_max[i], resolution, dtype=np.float32) for i in range(3)
    ]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def _subdivide(points: np.ndarray, new_voxel: float) -> np.ndarray:
    """Replace each voxel centre with its 8 children's centres."""
    offset = new_voxel / 2.0
    shifts = np.array(
        [[dx, dy, dz] for dx in (-offset, offset) for dy in (-offset, offset) for dz in (-offset, offset)],
        dtype=np.float32,
    )
    return (points[:, None, :] + shifts[None, :, :]).reshape(-1, 3)


def bounds_from_points(points: np.ndarray, percentile: float = 2.0, margin: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Robust bounding box of a sparse cloud, with margin.

    Percentile-clipped because COLMAP sparse clouds reliably contain a few
    wild triangulations; letting those set the box would spend most of the
    voxel budget on empty space.
    """
    low = np.percentile(points, percentile, axis=0)
    high = np.percentile(points, 100 - percentile, axis=0)
    pad = (high - low) * margin
    return low - pad, high + pad


def to_mesh(points: np.ndarray, voxel: float) -> Tuple[np.ndarray, np.ndarray]:
    """Marching-cubes surface from occupied voxel centres."""
    from skimage import measure

    origin = points.min(axis=0) - 2 * voxel
    indices = np.round((points - origin) / voxel).astype(np.int32)
    dims = indices.max(axis=0) + 3

    from scipy.ndimage import gaussian_filter

    volume = np.zeros(dims, dtype=np.float32)
    volume[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    # One genuinely 3D smoothing pass, so marching cubes returns a surface
    # rather than a staircase of voxel faces. Must be a volumetric filter --
    # a 2D blur over a flattened volume mixes unrelated slices and is worse
    # than no smoothing at all.
    volume = gaussian_filter(volume, sigma=0.7)

    verts, faces, _normals, _values = measure.marching_cubes(volume, level=0.35)
    return verts * voxel + origin, faces


def write_hull_3d_plot(
    out_path: Union[str, Path],
    hull_points: np.ndarray,
    sparse_points: np.ndarray,
    circle: Optional[dict] = None,
    max_points: int = 40000,
) -> Path:
    """Render the carved hull, and what the carve threw away.

    The first panel is the one that answers "why is the pipeline reconstructing
    the pliers?": it draws the P3 sparse cloud and the hull together, so the
    table top and the holder are visibly present as input and visibly absent
    from the result. Those points earn their place in P3 by supplying the
    parallax a thin seedling cannot, and the carve removes them for free,
    because they fall outside the plant silhouette in nearly every view.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from .pose import _equalise_3d, orbit_frame

    if circle is not None:
        origin, rotation = orbit_frame(circle)
        hull_points = (hull_points - origin) @ rotation.T
        sparse_points = (sparse_points - origin) @ rotation.T

    rng = np.random.default_rng(0)
    if len(hull_points) > max_points:
        hull_points = hull_points[rng.choice(len(hull_points), max_points, replace=False)]
    if len(sparse_points) > max_points:
        sparse_points = sparse_points[rng.choice(len(sparse_points), max_points, replace=False)]

    # Height above the carve's own base reads as a natural plant axis.
    height = hull_points[:, 2]

    fig = plt.figure(figsize=(19, 5.2))

    # Frame this panel on a neighbourhood of the hull rather than on the full
    # sparse cloud. The cloud spans the whole table top, which is an order of
    # magnitude larger than the seedling, so equalising on it shrinks the
    # subject to an illegible speck and the comparison shows nothing.
    hull_centre = (hull_points.max(axis=0) + hull_points.min(axis=0)) / 2.0
    hull_span = (hull_points.max(axis=0) - hull_points.min(axis=0)).max()
    context_radius = hull_span * 1.1
    near = np.all(np.abs(sparse_points - hull_centre) <= context_radius, axis=1)
    context = sparse_points[near]

    ax = fig.add_subplot(1, 4, 1, projection="3d")
    if len(context):
        ax.scatter(context[:, 0], context[:, 1], context[:, 2],
                   c="#B0453D", s=1.4, alpha=0.5, linewidths=0, label="P3 points, carved away")
    ax.scatter(hull_points[:, 0], hull_points[:, 1], hull_points[:, 2],
               c="#2F7A4A", s=0.8, alpha=0.8, linewidths=0, label="kept: carved hull")
    _equalise_3d(ax, np.vstack([context, hull_points]) if len(context) else hull_points)
    ax.view_init(elev=16, azim=-60)
    ax.set_title("carve vs input: table + holder removed", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=7.5, markerscale=6, framealpha=0.75)
    for setter in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
        setter([])

    for index, (elev, azim, title) in enumerate(
        [(16, -60, "hull, perspective"), (2, 0, "hull, edge-on"), (89, -90, "hull, from above")]
    ):
        ax = fig.add_subplot(1, 4, index + 2, projection="3d")
        ax.scatter(hull_points[:, 0], hull_points[:, 1], hull_points[:, 2],
                   c=height, cmap="viridis", s=1.0, alpha=0.85, linewidths=0)
        _equalise_3d(ax, hull_points)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=9.5)
        for setter in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
            setter([])

    fig.suptitle("P4: visual hull carved from the plant silhouettes (colour = height)", fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def write_ply_mesh(path: Union[str, Path], verts: np.ndarray, faces: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(verts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"element face {len(faces)}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        f.write(verts.astype("<f4").tobytes())
        face_bytes = bytearray()
        for face in faces.astype("<i4"):
            face_bytes.append(3)
            face_bytes.extend(face.tobytes())
        f.write(bytes(face_bytes))
