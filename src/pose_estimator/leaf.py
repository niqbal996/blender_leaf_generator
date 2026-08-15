"""P6 -- per-leaf midrib, frame, curvature and width.

Path B of the plan: an explicit geometric fit, which is both the fallback for
the neural parametric model and the sanity check on it.

The one detail worth getting right before anything else is the frame carried
along the midrib. It must be a **rotation-minimising frame** (double
reflection, Wang et al. 2008), not a Frenet frame. A Frenet frame is defined
by the curve's second derivative, so its normal flips through 180 degrees at
every inflection point -- and leaf midribs inflect constantly, since that is
what "the leaf curves down then back up" means. Every measurement taken in
that frame (width, curl, asymmetry) would silently invert at each flip, and
the result looks plausible rather than obviously broken. The plan flags this
as a silent corruption risk, and it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LeafModel:
    leaf_id: int
    midrib: np.ndarray  # (N, 3) arclength-sampled centreline
    tangents: np.ndarray  # (N, 3)
    normals: np.ndarray  # (N, 3) rotation-minimising
    binormals: np.ndarray  # (N, 3)
    curvature: np.ndarray  # (N,) 1/length units
    half_width: np.ndarray  # (N,) mean of the two sides
    width_left: np.ndarray  # (N,)
    width_right: np.ndarray  # (N,)
    arclength: float
    insertion_angle_deg: float
    azimuth_deg: float
    num_points: int

    def to_dict(self) -> dict:
        return {
            "id": self.leaf_id,
            "arclength": float(self.arclength),
            "insertion_angle_deg": float(self.insertion_angle_deg),
            "azimuth_deg": float(self.azimuth_deg),
            "num_points": int(self.num_points),
            "samples": [
                {
                    "s": float(i / max(len(self.midrib) - 1, 1)),
                    "p": self.midrib[i].tolist(),
                    "tangent": self.tangents[i].tolist(),
                    "normal": self.normals[i].tolist(),
                    "binormal": self.binormals[i].tolist(),
                    "kappa": float(self.curvature[i]),
                    "half_width": float(self.half_width[i]),
                    "width_asymmetry": float(
                        (self.width_left[i] - self.width_right[i])
                        / max(self.width_left[i] + self.width_right[i], 1e-9)
                    ),
                }
                for i in range(len(self.midrib))
            ],
        }


def resolve_leaf_base(
    axis: np.ndarray, points: np.ndarray, stem_axis_point: np.ndarray
) -> np.ndarray:
    """Where the leaf actually attaches, resolved from geometry not topology.

    Two failures this exists to prevent, both observed on DSC_0009:

    - **A reversed axis.** The skeleton's edges are meant to be oriented
      outward from the root, but not all of them are: leaf 0's polyline ran
      tip-to-base, so treating `axis[0]` as the attachment put s=0 at the
      tip and inverted the whole width profile. Choosing the endpoint nearer
      the stem is orientation-independent.

    - **A base that is not on the leaf.** A branch node can sit in mid-air
      between organs -- leaf 1's was 0.018 away from its own nearest point,
      about a fifth of the leaf's length. Distance-from-base binning then
      slices the blade at an angle and the resulting midrib hooks. Snapping
      onto the leaf's own points guarantees the origin of the arclength
      parameter lies on the thing being measured.

    Once P5 only emits edges that genuinely end at a tip, the axis
    orientation is reliable: those edges run (branch point -> tip), so
    `axis[0]` *is* the insertion. Two cleverer rules were tried before that
    was understood -- picking the endpoint nearest the plant origin, then the
    endpoint nearest the stem axis -- and both made the result worse, because
    they were compensating for a defect that was not in this function.
    """
    if len(axis) == 0:
        radii = np.hypot(*(points - stem_axis_point)[:, :2].T)
        return points[int(np.argmin(radii))]

    return points[np.argmin(np.linalg.norm(points - axis[0], axis=1))]


def extract_midrib(
    points: np.ndarray, base_xyz: np.ndarray, num_stations: int = 24, min_bin_points: int = 4
) -> Optional[np.ndarray]:
    """Centreline of a leaf's point set, from its base outward.

    Points are ordered by distance from the leaf's attachment rather than by
    a principal axis: a curved leaf's principal axis is a chord, and binning
    along it collapses the two ends of a strongly bent blade into the same
    station. Geodesic-from-base ordering follows the leaf even when it curls.

    Each station's centroid is the midrib estimate for that station -- the
    lamina is roughly symmetric about the midrib, so its cross-sectional
    centroid lies on it.
    """
    if len(points) < num_stations * min_bin_points:
        return None

    distance = np.linalg.norm(points - base_xyz, axis=1)
    edges = np.linspace(distance.min(), distance.max(), num_stations + 1)
    station_index = np.clip(np.digitize(distance, edges) - 1, 0, num_stations - 1)

    midrib = []
    for i in range(num_stations):
        members = points[station_index == i]
        if len(members) < min_bin_points:
            continue
        midrib.append(members.mean(axis=0))

    if len(midrib) < 4:
        return None
    return smooth_polyline(np.array(midrib))


def smooth_polyline(points: np.ndarray, iterations: int = 6, strength: float = 0.35) -> np.ndarray:
    """Laplacian smoothing with fixed endpoints.

    Station centroids inherit the noise of however many points landed in each
    bin; without smoothing that noise becomes curvature, and curvature is one
    of the outputs.
    """
    smoothed = points.astype(float).copy()
    for _ in range(iterations):
        interior = smoothed[1:-1] + strength * (
            0.5 * (smoothed[:-2] + smoothed[2:]) - smoothed[1:-1]
        )
        smoothed[1:-1] = interior
    return smoothed


def resample_by_arclength(points: np.ndarray, count: int) -> Tuple[np.ndarray, float]:
    """Evenly re-space a polyline along its own arclength."""
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    total = float(cumulative[-1])
    if total <= 0:
        return points, 0.0
    target = np.linspace(0.0, total, count)
    resampled = np.stack([np.interp(target, cumulative, points[:, i]) for i in range(3)], axis=1)
    return resampled, total


def rotation_minimising_frame(
    curve: np.ndarray, initial_normal: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation-minimising frame by double reflection (Wang et al. 2008).

    Returns (tangents, normals, binormals).

    The frame is propagated by reflecting the previous normal twice -- once
    into the plane bisecting consecutive positions, once into the plane
    bisecting consecutive tangents. The result is the frame that rotates as
    little as possible about the tangent between stations, so it stays
    continuous through inflection points where a Frenet frame would flip.
    """
    tangents = np.gradient(curve, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(lengths, 1e-12)

    if initial_normal is None:
        seed = np.array([0.0, 0.0, 1.0])
        if abs(float(tangents[0] @ seed)) > 0.9:
            seed = np.array([1.0, 0.0, 0.0])
        initial_normal = np.cross(tangents[0], seed)
    normal = initial_normal - tangents[0] * float(initial_normal @ tangents[0])
    normal /= max(np.linalg.norm(normal), 1e-12)

    normals = [normal]
    for i in range(len(curve) - 1):
        # First reflection: in the plane bisecting x_i and x_{i+1}.
        v1 = curve[i + 1] - curve[i]
        c1 = float(v1 @ v1)
        if c1 < 1e-24:
            normals.append(normals[-1])
            continue
        n_l = normals[-1] - (2.0 / c1) * float(v1 @ normals[-1]) * v1
        t_l = tangents[i] - (2.0 / c1) * float(v1 @ tangents[i]) * v1

        # Second reflection: in the plane bisecting t_l and t_{i+1}.
        v2 = tangents[i + 1] - t_l
        c2 = float(v2 @ v2)
        if c2 < 1e-24:
            normals.append(n_l)
            continue
        next_normal = n_l - (2.0 / c2) * float(v2 @ n_l) * v2
        next_normal -= tangents[i + 1] * float(next_normal @ tangents[i + 1])
        next_normal /= max(np.linalg.norm(next_normal), 1e-12)
        normals.append(next_normal)

    normals = np.array(normals)
    binormals = np.cross(tangents, normals)
    return tangents, normals, binormals


def curvature_along(curve: np.ndarray) -> np.ndarray:
    """Curvature magnitude |r' x r''| / |r'|^3 at each station."""
    first = np.gradient(curve, axis=0)
    second = np.gradient(first, axis=0)
    cross = np.cross(first, second)
    denominator = np.maximum(np.linalg.norm(first, axis=1) ** 3, 1e-12)
    return np.linalg.norm(cross, axis=1) / denominator


def measure_widths(
    points: np.ndarray,
    curve: np.ndarray,
    tangents: np.ndarray,
    binormals: np.ndarray,
    slab_fraction: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-station lamina half-width, measured separately on each side.

    Each station takes the points lying within a thin slab perpendicular to
    the midrib, and measures how far they reach along the binormal. Left and
    right are kept apart rather than averaged because their difference is the
    asymmetry the schema asks for, and averaging would erase it.
    """
    spacing = float(np.median(np.linalg.norm(np.diff(curve, axis=0), axis=1)))
    half_slab = max(spacing * slab_fraction, 1e-9)

    left = np.zeros(len(curve))
    right = np.zeros(len(curve))
    for i in range(len(curve)):
        offset = points - curve[i]
        along = offset @ tangents[i]
        in_slab = np.abs(along) <= half_slab
        if not in_slab.any():
            continue
        lateral = offset[in_slab] @ binormals[i]
        positive = lateral[lateral > 0]
        negative = lateral[lateral < 0]
        left[i] = float(np.percentile(positive, 95)) if len(positive) else 0.0
        right[i] = float(-np.percentile(negative, 5)) if len(negative) else 0.0
    return left, right


def fit_leaf(
    leaf_id: int,
    points: np.ndarray,
    base_xyz: np.ndarray,
    stem_tangent: np.ndarray,
    stem_axis: np.ndarray,
    num_samples: int = 20,
) -> Optional[LeafModel]:
    """Full geometric fit for one leaf's point subset."""
    midrib = extract_midrib(points, base_xyz)
    if midrib is None:
        return None

    # Resample finely, smooth, *then* decimate to the reported stations.
    # Smoothing at the output resolution is not enough: with ~20 stations
    # spanning a curved leaf the tangent can swing 25 degrees from one station
    # to the next, and since the frame is carried along the tangent it swings
    # with it -- which reads as a frame discontinuity when the frame is in
    # fact correct and the curve is simply under-sampled. Fitting the shape at
    # high resolution first separates the two.
    dense, arclength = resample_by_arclength(midrib, max(num_samples * 6, 60))
    if arclength <= 0:
        return None
    dense = smooth_polyline(dense, iterations=24, strength=0.5)
    curve, arclength = resample_by_arclength(dense, num_samples)

    tangents, normals, binormals = rotation_minimising_frame(curve)
    curvature = curvature_along(curve)
    left, right = measure_widths(points, curve, tangents, binormals)

    # Insertion angle: midrib tangent where it leaves the stem, against the
    # stem's own direction there.
    cosine = float(np.clip(tangents[0] @ stem_tangent, -1.0, 1.0))
    insertion_angle = float(np.degrees(np.arccos(cosine)))

    # Azimuth: where around the stem the leaf departs. Measured in the plane
    # perpendicular to the stem axis, so it is the phyllotactic angle rather
    # than an artefact of the world frame's arbitrary orientation.
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(stem_axis @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_a = np.cross(stem_axis, reference)
    basis_a /= max(np.linalg.norm(basis_a), 1e-12)
    basis_b = np.cross(stem_axis, basis_a)
    radial = curve[-1] - curve[0]
    radial -= stem_axis * float(radial @ stem_axis)
    azimuth = float(np.degrees(np.arctan2(radial @ basis_b, radial @ basis_a)) % 360.0)

    return LeafModel(
        leaf_id=leaf_id,
        midrib=curve,
        tangents=tangents,
        normals=normals,
        binormals=binormals,
        curvature=curvature,
        half_width=0.5 * (left + right),
        width_left=left,
        width_right=right,
        arclength=arclength,
        insertion_angle_deg=insertion_angle,
        azimuth_deg=azimuth,
        num_points=len(points),
    )


def frame_continuity_degrees(normals: np.ndarray) -> np.ndarray:
    """Angle between consecutive frame normals, in degrees.

    The plan's acceptance criterion: no discontinuity greater than 5 degrees
    between adjacent stations. This is what would catch a Frenet-style flip
    if one ever crept back in -- a flip shows up here as a ~180 degree jump.
    """
    dots = np.clip((normals[:-1] * normals[1:]).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))
