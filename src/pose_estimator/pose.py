"""P3 -- camera poses for a turntable capture, without fiducial markers.

The rig: the plant sits on a fixed table and does not move. The **camera
orbits it**, and the backdrop travels with the camera rig -- which is why the
footage looks like a locked-off camera watching a spinning plant, and why
that appearance is misleading in a way that matters.

The subject and table are the true static world, so what COLMAP recovers is
the camera's actual physical path, not a relative-motion stand-in. But the
backdrop, being rigid with the camera, is the one thing in frame that really
is motionless in image space. Match features on it and SfM draws the only
conclusion available: nothing moved. Every camera collapses to a single
point, and the plant -- the one thing with genuine parallax -- is rejected as
an outlier.

So matching is restricted to what is rigidly attached to the *table*: the
disc, the holder, and the plant. Those carry the parallax; the backdrop
carries an actively wrong signal, not merely a useless one.

Separating them needs no color threshold and no marker, only the observation
that they move differently in image space. Over a full orbit the
camera-mounted backdrop has near-zero temporal variance while the table
sweeps past. Measured on DSC_0009: backdrop ~5 DN standard deviation, disc
30-60 -- bimodal enough for Otsu to split without a hand-tuned constant.

Caveat worth carrying forward: the plant is only *approximately* static.
Residual vibration from the moving rig perturbs leaves between frames, which
shows up later as slight silhouette disagreement during P4 carving. That is
what `carve(min_inside_fraction=...)` absorbs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# COLMAP's mask convention: the mask for `images/<name>` lives at
# `<mask_dir>/<name>.png`; zero pixels are ignored, nonzero are used.
COLMAP_MASK_SUFFIX = ".png"


def temporal_std(frame_paths: Sequence[Path], stride: int = 2, max_frames: int = 64) -> np.ndarray:
    """Per-pixel standard deviation of intensity across the sequence.

    This is the motion signal the whole phase rests on, and it works here
    only because the backdrop rides along with the camera. That makes the
    footage behave, in image space, like a locked-off camera: the
    camera-mounted backdrop holds still while the fixed table sweeps past.
    On a rig whose backdrop stayed put, everything would move and this map
    would say nothing.
    """
    selected = list(frame_paths[::stride])[:max_frames]
    if len(selected) < 3:
        raise ValueError(f"Need at least 3 frames to measure temporal variance, got {len(selected)}")

    stack = np.stack(
        [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32) for p in selected]
    )
    return stack.std(axis=0)


def rotating_region_mask(
    frame_paths: Sequence[Path],
    close_radius: int = 15,
    min_area_fraction: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """Mask of what sweeps through the frame -- the table, holder and plant
    (255) -- versus the camera-mounted backdrop (0). Returns (mask, threshold).

    Note the inversion: the 255 region is what is *physically stationary*,
    and the 0 region is what physically travels with the camera. The mask is
    built from apparent motion because that is what is measurable, but what
    it selects is the true static world -- exactly the features SfM needs.

    One mask serves every frame, since the backdrop's fixed relationship to
    the camera keeps the swept region fixed in image space too.

    Holes are closed and filled deliberately: the disc has dark, untextured
    patches whose intensity barely changes as they sweep past, so a raw
    threshold perforates it. Those patches belong to the table, and punching
    them out would discard the parallax that conditions the reconstruction.
    """
    std = temporal_std(frame_paths)

    # Otsu on the variance map, not on intensity: the split being sought is
    # "moved" vs "did not move", and the histogram is strongly bimodal.
    std_u8 = np.clip(std / max(std.max(), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    threshold, mask = cv2.threshold(std_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((close_radius, close_radius), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if count > 1:
        min_area = min_area_fraction * mask.size
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        if not keep.any():
            keep[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))] = True
        mask = keep[labels].astype(np.uint8) * 255

    mask = _fill_holes(mask)
    return mask, float(threshold)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes of a binary mask via border flood fill."""
    mask = (mask > 0).astype(np.uint8) * 255
    h, w = mask.shape
    flood = mask.copy()
    scratch = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, scratch, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def rotating_masks_per_source(
    frame_paths: Sequence[Path], sources: Sequence[int]
) -> Tuple[dict, dict]:
    """One rotating-region mask per capture pass.

    A single mask over all frames is wrong the moment there is more than one
    pass. The backdrop is rigid with the *camera*, so at a different elevation
    it sits at a different place in the image -- pooling both passes makes
    almost everything look like it moved, and the mask stops separating
    anything. Each pass gets its own temporal-variance map instead.
    """
    masks, thresholds = {}, {}
    for source in sorted(set(sources)):
        subset = [p for p, g in zip(frame_paths, sources) if g == source]
        if len(subset) < 3:
            continue
        masks[source], thresholds[source] = rotating_region_mask(subset)
    return masks, thresholds


def write_colmap_masks(
    frame_paths: Sequence[Path],
    mask_dir: Union[str, Path],
    rotating_mask,
    plant_mask_dir: Optional[Union[str, Path]] = None,
    holder_mask_dir: Optional[Union[str, Path]] = None,
    sources: Optional[Sequence[int]] = None,
) -> int:
    """Write one COLMAP mask per frame: the rotating region, unioned with that
    frame's P2 plant/holder masks.

    The union is insurance rather than necessity. A leaf tip that only ever
    reaches one extreme of its arc contributes little temporal variance there
    and can fall outside the rotating region, and losing plant pixels is the
    one error worth spending a dilation to avoid.
    """
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for index, path in enumerate(frame_paths):
        # `rotating_mask` is either one array, or a {source: array} mapping
        # when several capture passes share this workdir.
        if isinstance(rotating_mask, dict):
            source = sources[index] if sources is not None else 0
            combined = rotating_mask[source].copy()
        else:
            combined = rotating_mask.copy()
        for extra_dir in (plant_mask_dir, holder_mask_dir):
            if extra_dir is None:
                continue
            extra = cv2.imread(str(Path(extra_dir) / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
            if extra is not None:
                combined = np.maximum(combined, (extra > 0).astype(np.uint8) * 255)
        cv2.imwrite(str(mask_dir / f"{path.name}{COLMAP_MASK_SUFFIX}"), combined)
        written += 1
    return written


# --------------------------------------------------------------------------
# Pose quality
# --------------------------------------------------------------------------


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares plane through points. Returns (centroid, unit normal, RMS)."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[2]
    rms = float(np.sqrt(np.mean(((points - centroid) @ normal) ** 2)))
    return centroid, normal, rms


def fit_circle_3d(points: np.ndarray) -> dict:
    """Fit a circle to 3D points assumed to lie near a common plane.

    This is the acceptance test that actually validates a turntable solve.
    The camera is fixed and the subject spins, so in the subject's frame the
    camera *must* trace a circle. Nothing in the SfM pipeline enforces that,
    which is what makes it genuine evidence rather than a restatement of the
    optimiser's own objective: a reconstruction that has collapsed, drifted,
    or mirrored will not produce a clean circle by accident.
    """
    centroid, normal, plane_rms = fit_plane(points)

    # Project into the plane's own 2D basis, where a circle fit is linear.
    basis_a = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(basis_a) < 1e-6:
        basis_a = np.cross(normal, [0.0, 1.0, 0.0])
    basis_a /= np.linalg.norm(basis_a)
    basis_b = np.cross(normal, basis_a)

    local = points - centroid
    u, v = local @ basis_a, local @ basis_b

    # Algebraic (Kasa) fit: u^2+v^2 = 2*cu*u + 2*cv*v + c
    A = np.stack([2 * u, 2 * v, np.ones_like(u)], axis=1)
    solution, *_ = np.linalg.lstsq(A, u**2 + v**2, rcond=None)
    cu, cv_, c = solution
    radius = float(np.sqrt(max(c + cu**2 + cv_**2, 0.0)))

    residuals = np.abs(np.hypot(u - cu, v - cv_) - radius)
    center_3d = centroid + cu * basis_a + cv_ * basis_b

    # Angular coverage: a full turn should span ~360 deg with no large gap.
    angles = np.sort(np.degrees(np.arctan2(v - cv_, u - cu)) % 360.0)
    gaps = np.diff(np.concatenate([angles, angles[:1] + 360.0]))

    return {
        "center": center_3d.tolist(),
        "axis": normal.tolist(),
        "radius": radius,
        "circle_rms": float(residuals.mean()),
        "circle_rms_relative": float(residuals.mean() / radius) if radius > 0 else float("inf"),
        "circle_max_deviation_relative": float(residuals.max() / radius) if radius > 0 else float("inf"),
        "plane_rms_relative": float(plane_rms / radius) if radius > 0 else float("inf"),
        "largest_angular_gap_deg": float(gaps.max()),
    }


def evaluate_poses(reconstruction, num_input_frames: int,
                   sources: Optional[dict] = None) -> dict:
    """Score a finished reconstruction against turntable-specific expectations.

    With more than one capture pass the single-circle test is wrong by
    construction -- two elevations trace two circles, and fitting one through
    both reports a large residual for a perfectly good solve. Each pass is
    fitted separately, and the *agreement between their axes* becomes a new
    and stronger check: two independently-solved orbits sharing a rotation
    axis is hard to achieve by accident, so it is real evidence the passes
    were merged into one consistent frame.
    """
    from .reconstruction import get_registered_camera_poses

    centers, _directions, names = get_registered_camera_poses(reconstruction)
    registered_fraction = reconstruction.num_reg_images() / max(num_input_frames, 1)
    reprojection_error = reconstruction.compute_mean_reprojection_error()

    circle = fit_circle_3d(centers) if len(centers) >= 5 else None

    # Group the registered cameras by which pass their frame came from.
    groups = {}
    if sources:
        for name, centre in zip(names, centers):
            groups.setdefault(sources.get(Path(name).stem, 0), []).append(centre)
    per_pass = {g: fit_circle_3d(np.array(c)) for g, c in groups.items() if len(c) >= 5}

    checks = {
        "most_frames_registered": {
            "pass": registered_fraction >= 0.95,
            "detail": f"{reconstruction.num_reg_images()}/{num_input_frames} frames registered "
            f"({registered_fraction:.1%}, target 95%)",
        },
        "low_reprojection_error": {
            "pass": reprojection_error < 1.5,
            "detail": f"mean reprojection error {reprojection_error:.3f} px (limit 1.5)",
        },
    }
    if len(per_pass) > 1:
        worst_rms = max(c["circle_rms_relative"] for c in per_pass.values())
        worst_gap = max(c["largest_angular_gap_deg"] for c in per_pass.values())
        checks["each_pass_lies_on_a_circle"] = {
            "pass": worst_rms < 0.02,
            "detail": "; ".join(
                f"pass {g}: {c['circle_rms_relative']:.2%} RMS, {len(groups[g])} views"
                for g, c in sorted(per_pass.items())),
        }
        checks["full_rotation_covered"] = {
            "pass": worst_gap < 30.0,
            "detail": f"largest angular gap across passes {worst_gap:.1f} deg (limit 30)",
        }

        axes = [np.array(c["axis"]) / np.linalg.norm(c["axis"]) for c in per_pass.values()]
        angles = [float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))
                  for i, a in enumerate(axes) for b in axes[i + 1:]]
        checks["passes_share_a_rotation_axis"] = {
            "pass": bool(angles) and max(angles) < 2.0,
            "detail": f"axes differ by at most {max(angles):.2f} deg (limit 2). This is the "
                      f"check that the passes really merged: nothing in the solve enforces it",
        }

        elevations = sorted(float(np.array(c["center"]) @ axes[0]) for c in per_pass.values())
        spread = max(elevations) - min(elevations)
        radius = float(np.mean([c["radius"] for c in per_pass.values()]))
        checks["passes_are_at_different_elevations"] = {
            "pass": spread > 0.05 * radius,
            "detail": f"orbit centres separated by {spread / max(radius, 1e-9):.1%} of orbit "
                      f"radius along the axis -- a second elevation only helps if it *is* one",
        }
    elif circle is not None:
        checks["cameras_lie_on_a_circle"] = {
            "pass": circle["circle_rms_relative"] < 0.02,
            "detail": f"camera centres deviate {circle['circle_rms_relative']:.2%} of the orbit "
            f"radius RMS (limit 2%), worst {circle['circle_max_deviation_relative']:.2%}",
        }
        checks["cameras_coplanar"] = {
            "pass": circle["plane_rms_relative"] < 0.02,
            "detail": f"out-of-plane scatter {circle['plane_rms_relative']:.2%} of radius (limit 2%)",
        }
        checks["full_rotation_covered"] = {
            "pass": circle["largest_angular_gap_deg"] < 30.0,
            "detail": f"largest angular gap {circle['largest_angular_gap_deg']:.1f} deg (limit 30)",
        }

    return {
        "num_input_frames": num_input_frames,
        "num_registered": reconstruction.num_reg_images(),
        "registered_fraction": registered_fraction,
        "mean_reprojection_error_px": reprojection_error,
        "num_points3D": reconstruction.num_points3D(),
        "orbit": consensus_orbit(per_pass) if len(per_pass) > 1 else circle,
        "orbits_per_pass": {str(g): c for g, c in per_pass.items()},
        "num_passes": len(per_pass) if per_pass else 1,
        "registered_image_names": sorted(names),
        "checks": checks,
        "all_passed": all(c["pass"] for c in checks.values()),
    }


def write_scene_3d_plot(
    out_path: Union[str, Path],
    centers: np.ndarray,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    circle: Optional[dict] = None,
    max_points: int = 20000,
) -> Path:
    """Render the solved scene: sparse cloud plus the camera path around it.

    Poses on their own are invisible -- a list of matrices looks the same
    whether it is right or wrong. Drawn together with the points they
    triangulated, correctness becomes obvious at a glance: the cloud should
    look like a plant standing on a disc, and the cameras should ring it at a
    constant radius and height.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if len(points) > max_points:
        pick = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, colors = points[pick], (colors[pick] if colors is not None else None)

    # Rotate into the orbit's own frame (axis -> +Z, centre -> origin).
    # Without this the "top-down" and "side-on" panels look down COLMAP's
    # arbitrary world axes instead of the turntable axis, and the views do not
    # show what their titles claim -- which is worse than not drawing them.
    if circle is not None:
        origin, rotation = orbit_frame(circle)
        points = (points - origin) @ rotation.T
        centers = (centers - origin) @ rotation.T

    point_colors = (colors / 255.0) if colors is not None else "tab:green"

    panels = [
        (20, -60, "perspective", False),
        (89, -90, "down the orbit axis (should be a circle)", False),
        (0, 0, "edge-on (orbit should be a flat line)", False),
        (18, -60, "zoom: sparse cloud only", True),
    ]

    fig = plt.figure(figsize=(19, 5.2))
    for index, (elev, azim, title, zoom) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), index + 1, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors, s=1.2 if zoom else 0.6,
                   alpha=0.6, linewidths=0)
        if not zoom:
            ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="tab:red", lw=1.0, alpha=0.7)
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="tab:red", s=10, depthshade=False)
            _equalise_3d(ax, np.vstack([points, centers]))
        else:
            _equalise_3d(ax, points)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    fig.suptitle(
        "P3: sparse reconstruction (point colours from the images) + solved camera path (red)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def consensus_orbit(per_pass: dict) -> dict:
    """One orbit description shared by several coaxial passes.

    Everything downstream -- the upright plant frame P5 works in, every 3D
    diagnostic -- is derived from a single centre and axis. Fitting one circle
    through the pooled centres of two elevations would describe neither: the
    result sits between the rings with a radius belonging to no pass. What is
    actually shared is the *rotation axis*, and every pass centre lies on it,
    so the axis is averaged and the centre taken along it.

    Quality numbers are reported as the worst pass rather than an average --
    a frame is only as trustworthy as its weakest contributor.
    """
    fits = [per_pass[g] for g in sorted(per_pass)]
    reference = np.array(fits[0]["axis"], dtype=float)
    axes = []
    for fit in fits:
        axis = np.array(fit["axis"], dtype=float)
        # The sign of a fitted plane normal is arbitrary; averaging without
        # aligning them can cancel two perfectly consistent axes to zero.
        axes.append(-axis if axis @ reference < 0 else axis)

    axis = np.mean(axes, axis=0)
    axis /= np.linalg.norm(axis)
    center = np.mean([fit["center"] for fit in fits], axis=0)

    return {
        "center": center.tolist(),
        "axis": axis.tolist(),
        "radius": float(np.mean([fit["radius"] for fit in fits])),
        "circle_rms": float(max(fit["circle_rms"] for fit in fits)),
        "circle_rms_relative": float(max(fit["circle_rms_relative"] for fit in fits)),
        "circle_max_deviation_relative": float(
            max(fit["circle_max_deviation_relative"] for fit in fits)),
        "plane_rms_relative": float(max(fit["plane_rms_relative"] for fit in fits)),
        "largest_angular_gap_deg": float(max(fit["largest_angular_gap_deg"] for fit in fits)),
        "from_passes": sorted(per_pass),
    }


def orbit_frame(circle: dict) -> Tuple[np.ndarray, np.ndarray]:
    """(origin, rotation) taking world points into the orbit's own frame.

    Shared by every 3D diagnostic so they all agree on which way is up. In
    the returned frame the orbit axis is +Z and its centre is the origin,
    which is what makes "down the axis" and "edge-on" views meaningful --
    COLMAP's world axes are arbitrary and have no relation to the rig.
    """
    origin = np.array(circle["center"], dtype=float)
    axis = np.array(circle["axis"], dtype=float)
    axis /= np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    basis_a = np.cross(axis, helper)
    basis_a /= np.linalg.norm(basis_a)
    basis_b = np.cross(axis, basis_a)
    return origin, np.stack([basis_a, basis_b, axis])


def _equalise_3d(ax, points: np.ndarray) -> None:
    """Equal aspect for a 3D axis -- without it a circle renders as an ellipse
    and the diagnostic lies about the thing it exists to show.
    """
    span = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    mid = (points.max(axis=0) + points.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)


def write_reprojection_overlays(
    out_dir: Union[str, Path],
    reconstruction,
    frames_dir: Union[str, Path],
    n_samples: int = 4,
) -> List[Path]:
    """Draw each frame's own triangulated points back onto it.

    The end-to-end check that poses, intrinsics and images actually agree:
    every dot is a 3D point re-projected through the solved camera, so if the
    dots land on the plant and the table texture that produced them, the
    solve is consistent with the pixels. If the poses were wrong the dots
    would drift off the features entirely.
    """
    out_dir, frames_dir = Path(out_dir), Path(frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_ids = sorted(reconstruction.reg_image_ids())
    picks = np.linspace(0, len(image_ids) - 1, min(n_samples, len(image_ids))).astype(int)

    written = []
    for i in picks:
        image = reconstruction.images[image_ids[i]]
        frame = cv2.imread(str(frames_dir / image.name))
        if frame is None:
            continue

        drawn = 0
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            x, y = int(round(point2D.xy[0])), int(round(point2D.xy[1]))
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            drawn += 1

        label = f"{image.name}: {drawn} triangulated points reprojected"
        cv2.putText(frame, label, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(frame, label, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        path = out_dir / f"reprojection_{Path(image.name).stem}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(path)
    return written


def export_scene_ply(
    points_path: Union[str, Path],
    cameras_path: Union[str, Path],
    points: np.ndarray,
    colors: np.ndarray,
    centers: np.ndarray,
) -> None:
    """Dump the sparse cloud and camera centres as PLY, for Blender/MeshLab.

    Two files rather than one so the camera path can be styled separately;
    both are in the raw COLMAP frame (unscaled, arbitrary orientation).
    """
    from .ply_io import write_ply_vertices

    write_ply_vertices(
        Path(points_path),
        {
            "x": points[:, 0].astype(np.float32),
            "y": points[:, 1].astype(np.float32),
            "z": points[:, 2].astype(np.float32),
            "red": colors[:, 0].astype(np.uint8),
            "green": colors[:, 1].astype(np.uint8),
            "blue": colors[:, 2].astype(np.uint8),
        },
    )
    write_ply_vertices(
        Path(cameras_path),
        {
            "x": centers[:, 0].astype(np.float32),
            "y": centers[:, 1].astype(np.float32),
            "z": centers[:, 2].astype(np.float32),
            "red": np.full(len(centers), 255, np.uint8),
            "green": np.zeros(len(centers), np.uint8),
            "blue": np.zeros(len(centers), np.uint8),
        },
    )


def write_orbit_plot(out_path: Union[str, Path], centers: np.ndarray, circle: dict) -> Path:
    """Plot the solved camera centres against their fitted circle.

    A turntable solve either looks obviously right here or obviously wrong,
    which no scalar summary conveys as quickly.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    center = np.array(circle["center"])
    normal = np.array(circle["axis"])
    basis_a = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(basis_a) < 1e-6:
        basis_a = np.cross(normal, [0.0, 1.0, 0.0])
    basis_a /= np.linalg.norm(basis_a)
    basis_b = np.cross(normal, basis_a)

    local = centers - center
    u, v, w = local @ basis_a, local @ basis_b, local @ normal

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    theta = np.linspace(0, 2 * np.pi, 256)
    ax1.plot(circle["radius"] * np.cos(theta), circle["radius"] * np.sin(theta),
             color="tab:gray", ls="--", lw=1.2, label="fitted circle")
    ax1.scatter(u, v, c=np.arange(len(u)), cmap="viridis", s=26, zorder=3, label="camera centres")
    ax1.set_aspect("equal")
    ax1.set_title(f"Orbit in the turntable plane\nRMS {circle['circle_rms_relative']:.2%} of radius")
    ax1.set_xlabel("in-plane u"); ax1.set_ylabel("in-plane v")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

    ax2.scatter(np.degrees(np.arctan2(v, u)) % 360.0, w, c=np.arange(len(u)), cmap="viridis", s=26)
    ax2.axhline(0, color="tab:gray", ls="--", lw=1.2)
    ax2.set_title(f"Out-of-plane deviation\nRMS {circle['plane_rms_relative']:.2%} of radius")
    ax2.set_xlabel("orbit angle (deg)"); ax2.set_ylabel("height off the fitted plane")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
