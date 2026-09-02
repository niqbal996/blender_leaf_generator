"""Turn a hand-marked pixel into the plant's 3D origin.

Where a plant meets its growing medium is trivial for a person to point at
in one photo and genuinely hard to infer geometrically: the pipeline's
automatic answer works by finding the substrate the plant stands on, which
presumes there *is* one. A cutting held in a pair of pliers for a clear
view has no pot, no soil plane, and no ground contact at all, so there is
nothing to detect -- and no amount of tuning invents it.

One click in one frame resolves it. The marked pixel defines a ray through
the scene, and the reconstructed cloud says how far along that ray the
plant actually is, so a single 2D annotation recovers a 3D point. Every
other frame then agrees with it for free, because they all share one
reconstruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

MARKER_FILENAME = "root_marker.json"


@dataclass
class RootMarker:
    """Either a pixel in one frame, or a point already known in 3D.

    The 3D form exists because the 2D form has an irreducible weakness:
    one pixel gives a ray, and depth along it has to be guessed from the
    cloud. Against a thin stem that guess lands on whatever else the ray
    grazes, producing a point that reprojects *consistently* -- and so
    looks right -- while floating beside the plant. Picking in 3D has no
    such ambiguity.
    """

    image_name: Optional[str] = None  # e.g. "DSC_0009_0000.jpg", for the 2D form
    xy: Optional[Tuple[float, float]] = None  # pixel coords, origin top-left
    xyz_colmap: Optional[Tuple[float, float, float]] = None  # raw COLMAP frame

    @property
    def is_3d(self) -> bool:
        return self.xyz_colmap is not None


def load_root_marker(workdir: Union[str, Path]) -> Optional[RootMarker]:
    """Read `<workdir>/root_marker.json`, or None if absent."""
    path = Path(workdir) / MARKER_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if "xyz_colmap" in data:
        return RootMarker(xyz_colmap=tuple(float(v) for v in data["xyz_colmap"]))
    return RootMarker(image_name=data["image"], xy=(float(data["xy"][0]), float(data["xy"][1])))


def save_root_marker(workdir: Union[str, Path], marker: RootMarker) -> Path:
    path = Path(workdir) / MARKER_FILENAME
    if marker.is_3d:
        payload = {"xyz_colmap": list(marker.xyz_colmap)}
    else:
        payload = {"image": marker.image_name, "xy": list(marker.xy)}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def camera_ray(reconstruction, image_name: str, xy: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """(origin, unit direction) of the viewing ray through pixel `xy` of
    `image_name`, in the reconstruction's world frame.

    `cam_from_img` is what makes this correct on a real lens: it inverts the
    camera model, so a pixel measured on the *distorted* frame the user
    actually looked at maps to the right direction. Treating the pixel as
    pinhole would misplace the ray by more the further the mark sits from
    the image center.
    """
    image = None
    for candidate in reconstruction.images.values():
        if candidate.name == image_name:
            image = candidate
            break
    if image is None:
        available = sorted(c.name for c in reconstruction.images.values())[:5]
        raise KeyError(
            f"{image_name!r} is not a registered image. First few registered: {available}"
        )

    camera = reconstruction.cameras[image.camera_id]
    normalized = np.asarray(camera.cam_from_img(np.asarray(xy, dtype=np.float64)), dtype=np.float64)
    direction_camera = np.array([normalized[0], normalized[1], 1.0])

    pose = image.cam_from_world()
    rotation = pose.rotation.matrix()
    origin = -rotation.T @ pose.translation
    direction_world = rotation.T @ direction_camera
    return origin, direction_world / np.linalg.norm(direction_world)


def point_nearest_ray(
    xyz: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    max_distance: Optional[float] = None,
    min_support: int = 3,
    min_depth_fraction: float = 0.3,
) -> Optional[int]:
    """Index of the first *real surface* the ray meets: walking outward from
    the camera, the nearest point that is not an isolated speck.

    Depth along the ray is what a single click cannot give, so it is taken
    from the cloud -- but neither obvious rule works alone, and both fail
    quietly rather than loudly:

    - Nearest to the *ray* picks whatever is best reconstructed along that
      line. A thin dark stem in front of a broad dusty turntable loses to
      the turntable, and the result reprojects a consistent few centimeters
      off the stem in every view -- consistent enough to look right.
    - Nearest to the *camera* is correct ray-casting, but SfM clouds contain
      floaters, and one stray point between lens and plant wins outright.

    Requiring `min_support` neighbours within `max_distance` separates the
    two: a floater has no neighbours, a real surface does. So the rule is
    ray-casting, restricted to points that are part of something.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0:
        return None

    offsets = xyz - origin
    along = offsets @ direction
    perpendicular = np.linalg.norm(offsets - np.outer(along, direction), axis=1)

    # Scale everything by how far the camera actually is from the scene, not
    # by the cloud's bounding box. The box is inflated by whatever distant
    # background survived reconstruction -- 30 units across on a capture
    # where the camera sits 4 from the subject -- so a "2% of the box"
    # tolerance is half the working distance, and admits everything.
    scene_distance = float(np.median(np.linalg.norm(offsets, axis=1)))
    if max_distance is None:
        max_distance = 0.01 * scene_distance

    # Points a few thousandths of the working distance from the lens are
    # mis-triangulations, not the subject; this capture had a knot of them
    # at depth 0.005 against a subject at 3.4, and taking the first hit
    # walked straight into it.
    valid = (along > min_depth_fraction * scene_distance) & (perpendicular <= max_distance)
    if not valid.any():
        return None

    candidates = np.nonzero(valid)[0]
    candidates = candidates[np.argsort(along[candidates])]

    tree = cKDTree(xyz)
    for index in candidates:
        if len(tree.query_ball_point(xyz[index], r=max_distance)) >= min_support:
            return int(index)

    # Everything along the ray is isolated; fall back to the best-supported
    # candidate rather than failing outright.
    supports = [len(tree.query_ball_point(xyz[i], r=max_distance)) for i in candidates]
    return int(candidates[int(np.argmax(supports))])


def resolve_root_xyz(
    reconstruction, marker: RootMarker, xyz: np.ndarray, max_distance: Optional[float] = None
) -> Optional[np.ndarray]:
    """The 3D point the marker refers to, or None if the ray missed the cloud.

    A 3D marker is already the answer and is returned unchanged -- no ray,
    no depth guess, nothing to go wrong.
    """
    if marker.is_3d:
        return np.asarray(marker.xyz_colmap, dtype=np.float64)

    origin, direction = camera_ray(reconstruction, marker.image_name, marker.xy)
    index = point_nearest_ray(xyz, origin, direction, max_distance)
    return None if index is None else np.asarray(xyz)[index]
