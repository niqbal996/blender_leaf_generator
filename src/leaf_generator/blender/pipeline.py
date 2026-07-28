"""Orchestrates loading leaf map sets into Blender: one mesh + material +
attachment-point Empty per leaf, laid out in a row per session ("maps"
folder), grouped under collections.

Entry point: `run(base_path)`.
"""

from __future__ import annotations

import json
from math import radians
from pathlib import Path
from typing import Dict, List, Optional, Union

import bpy
from mathutils import Vector

from .. import calibration, discovery, sizing
from ..calibration import SessionCalibration
from ..keypoints import NoContourError, estimate_attachment_point, pixel_to_local
from ..mismatch import mask_shape_iou
from .keypoint_empty import create_attachment_empty
from .materials import build_leaf_material
from .mesh import create_contour_based_mesh


def run(
    base_path: Union[str, Path],
    fallback_scale: float = 0.08,
    row_spacing: float = 0.02,
    mismatch_warn_threshold: float = 0.5,
    root_collection_name: str = "LeafAssets",
) -> None:
    """Load every leaf found under `base_path` into the scene.

    `base_path` can either be a single "maps" folder (like the
    `weed1/maps` example), or a parent directory containing multiple such
    folders anywhere below it (e.g. one per plant) -- each is discovered
    and laid out as its own row, in its own sub-collection.

    Each session's leaves are scaled to true real-world size (1 Blender
    unit = 1 meter) using the `pixelsize_mm` calibration from that
    session's `oberseite_log.json` / `unterseite_log.json`, when present.
    Sessions without a usable log fall back to normalizing every leaf's
    longer side to `fallback_scale` meters -- those leaves will *not* be
    correctly sized relative to each other or to calibrated sessions.
    """
    base_path = Path(base_path)

    if not base_path.is_dir():
        raise FileNotFoundError(
            f"[leaf_generator] MAPS_FOLDER does not exist or isn't visible to Blender: {base_path}\n"
            "If Blender runs on Windows, this must be a path Windows can resolve "
            "(e.g. 'E:\\...'), not a WSL '/mnt/e/...' path."
        )

    sessions = _discover_sessions(base_path)
    if not sessions:
        raise FileNotFoundError(
            f"[leaf_generator] No leaf map files (e.g. '1_ALBEDO_oberseite.png') found "
            f"under {base_path} or any 'maps' subfolder beneath it."
        )

    bpy.context.scene.render.engine = 'CYCLES'
    root_collection = _get_or_create_collection(bpy.context.scene.collection, root_collection_name)

    total_leaves = 0
    for session_path in sessions:
        session_name = session_path.parent.name
        session_collection = _get_or_create_collection(root_collection, session_name)
        total_leaves += _process_session(
            session_path,
            session_name,
            session_collection,
            fallback_scale=fallback_scale,
            row_spacing=row_spacing,
            mismatch_warn_threshold=mismatch_warn_threshold,
        )

    print(
        f"[leaf_generator] Done. Built {total_leaves} leaf object(s) across "
        f"{len(sessions)} session(s) under collection '{root_collection_name}'."
    )


def _discover_sessions(base_path: Path) -> List[Path]:
    if discovery.is_maps_folder(base_path):
        return [base_path]
    return sorted(p for p in base_path.rglob("maps") if discovery.is_maps_folder(p))


def _get_or_create_collection(parent: bpy.types.Collection, name: str) -> bpy.types.Collection:
    existing = parent.children.get(name)
    if existing is not None:
        return existing
    new_collection = bpy.data.collections.new(name)
    parent.children.link(new_collection)
    return new_collection


def _process_session(
    session_path: Path,
    session_name: str,
    collection: bpy.types.Collection,
    fallback_scale: float,
    row_spacing: float,
    mismatch_warn_threshold: float,
) -> int:
    leaves = discovery.find_leaf_groups(session_path)
    if not leaves:
        return 0

    session_calibration = calibration.load_session_calibration(session_path, side=discovery.PRIMARY_SIDE)
    if session_calibration is not None:
        print(
            f"[leaf_generator] {session_name}: using calibration from "
            f"'{session_calibration.source_log.name}' ({session_calibration.pixel_size_mm} mm/px "
            f"@ {session_calibration.distance_z_mm}mm) -- leaves scaled to real-world size."
        )
    else:
        print(
            f"[leaf_generator] {session_name}: WARNING no calibration log found (expected "
            f"'oberseite_log.json' or similar next to the maps) -- leaves will use a normalized "
            f"{fallback_scale}m placeholder scale, NOT real-world size."
        )

    keypoints_dir = session_path / "keypoints"
    keypoints_dir.mkdir(exist_ok=True)

    y_offset = 0.0
    count = 0
    for leaf in leaves.values():
        next_offset = _process_leaf(
            leaf,
            session_name=session_name,
            collection=collection,
            session_calibration=session_calibration,
            fallback_scale=fallback_scale,
            y_offset=y_offset,
            row_spacing=row_spacing,
            mismatch_warn_threshold=mismatch_warn_threshold,
            keypoints_dir=keypoints_dir,
        )
        if next_offset is None:
            continue
        y_offset = next_offset
        count += 1

    return count


def _process_leaf(
    leaf: discovery.LeafMapSet,
    session_name: str,
    collection: bpy.types.Collection,
    session_calibration: Optional[SessionCalibration],
    fallback_scale: float,
    y_offset: float,
    row_spacing: float,
    mismatch_warn_threshold: float,
    keypoints_dir: Path,
) -> Optional[float]:
    primary_side = leaf.primary_side
    front_maps = leaf.maps_for(primary_side)
    mask_path = front_maps.get("mask")
    if mask_path is None:
        print(f"[leaf_generator] Skipping leaf {leaf.leaf_id} ({session_name}): no mask for side '{primary_side}'.")
        return None

    mesh_name = f"{session_name}_leaf_{leaf.leaf_id}"
    mesh_result = create_contour_based_mesh(
        mask_path,
        mesh_name,
        collection=collection,
        pixel_size_m=session_calibration.pixel_size_m if session_calibration else None,
        fallback_scale=fallback_scale,
    )
    if mesh_result is None:
        print(f"[leaf_generator] Skipping leaf {leaf.leaf_id} ({session_name}): could not build mesh from mask.")
        return None
    obj = mesh_result.obj

    back_maps = None
    mask_iou = None
    if leaf.has_both_sides:
        secondary_side = discovery.SECONDARY_SIDE if primary_side == discovery.PRIMARY_SIDE else discovery.PRIMARY_SIDE
        back_maps = leaf.maps_for(secondary_side)
        mask_iou = mask_shape_iou(front_maps["mask"], back_maps["mask"])
        if mask_iou < mismatch_warn_threshold:
            print(
                f"[leaf_generator] WARNING: leaf {leaf.leaf_id} ({session_name}): "
                f"{primary_side}/{secondary_side} mask shapes only {mask_iou:.2f} IoU -- "
                f"sides may not correspond well. Built as a double-sided mesh using "
                f"'{primary_side}' geometry anyway; inspect this leaf manually."
            )

    material = build_leaf_material(f"{mesh_name}_material", front_maps, back_maps)
    obj.data.materials.clear()
    obj.data.materials.append(material)

    obj.rotation_euler[0] = radians(180)
    bpy.context.view_layer.update()

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    bbox = [eval_obj.matrix_world @ Vector(corner) for corner in eval_obj.bound_box]
    y_min = min(v.y for v in bbox)
    y_max = max(v.y for v in bbox)
    z_min = min(v.z for v in bbox)
    leaf_height = y_max - y_min

    obj.location.x = 0.0
    obj.location.y = y_offset + leaf_height / 2
    obj.location.z = -z_min
    bpy.context.view_layer.update()

    attachment_record = _place_attachment_empty(mesh_name, obj, mask_path, mesh_result, collection, leaf.leaf_id)
    size_record = _measure_leaf_size(mask_path, session_calibration, leaf.leaf_id)

    _write_keypoint_json(
        keypoints_dir / f"leaf_{leaf.leaf_id}.json",
        leaf_id=leaf.leaf_id,
        mesh_name=mesh_name,
        primary_side=primary_side,
        sides_available=sorted(leaf.sides.keys()),
        mask_shape_iou=mask_iou,
        attachment_point=attachment_record,
        real_world_size_mm=size_record,
        calibration=_calibration_record(session_calibration),
    )

    print(f"[leaf_generator] {mesh_name}: positioned at Y={obj.location.y:.3f}, height={leaf_height:.3f}m")
    return y_offset + leaf_height + row_spacing


def _place_attachment_empty(mesh_name, obj, mask_path, mesh_result, collection, leaf_id) -> Optional[Dict]:
    try:
        keypoint = estimate_attachment_point(mask_path)
    except (NoContourError, FileNotFoundError) as exc:
        print(f"[leaf_generator] Warning: could not estimate attachment point for leaf {leaf_id}: {exc}")
        return None

    local = pixel_to_local(keypoint["pixel"], keypoint["image_size"], mesh_result.scale_x, mesh_result.scale_y)
    empty = create_attachment_empty(f"{mesh_name}_attachment", obj, local, collection=collection)
    bpy.context.view_layer.update()
    world = empty.matrix_world.translation

    return {
        "pixel": list(keypoint["pixel"]),
        "image_size": list(keypoint["image_size"]),
        "local": list(local),
        "world": [world.x, world.y, world.z],
        "narrow_width_px": keypoint["narrow_width"],
        "broad_width_px": keypoint["broad_width"],
        "confidence": keypoint["confidence"],
    }


def _measure_leaf_size(mask_path, session_calibration: Optional[SessionCalibration], leaf_id) -> Optional[Dict]:
    if session_calibration is None:
        return None
    try:
        return sizing.measure_leaf_size_mm(mask_path, session_calibration)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[leaf_generator] Warning: could not measure size for leaf {leaf_id}: {exc}")
        return None


def _calibration_record(session_calibration: Optional[SessionCalibration]) -> Optional[Dict]:
    if session_calibration is None:
        return None
    return {
        "pixel_size_mm": session_calibration.pixel_size_mm,
        "distance_z_mm": session_calibration.distance_z_mm,
        "camera_model": session_calibration.camera_model,
        "lens_model": session_calibration.lens_model,
        "focal_length_mm": session_calibration.focal_length_mm,
        "source_log": str(session_calibration.source_log) if session_calibration.source_log else None,
    }


def _write_keypoint_json(path: Path, **data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
