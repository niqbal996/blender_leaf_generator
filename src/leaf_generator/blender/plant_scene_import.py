"""Build a Blender scene for one plant capture: skeleton curves + keypoint
Empties from `align_plant_skeleton.py`'s baked `skeleton_blender.json`,
optionally a sanity-check point-cloud mesh, and (if the KIRI 3DGS Render
add-on is installed) the trained splat -- all already aligned to real-world
meters, Z-up by `align_plant_skeleton.py`, so nothing needs a live transform
object in Blender; everything just drops in at the right place/scale.

Entry point: `run(workdir)`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import bpy

_TIP_DISPLAY_TYPE = "SPHERE"
_BRANCH_DISPLAY_TYPE = "CUBE"


def run(
    workdir: Union[str, Path],
    collection_name: str = "Plant",
    import_pointcloud: bool = True,
    import_splat: bool = True,
    curve_bevel_depth: float = 0.002,
) -> bpy.types.Collection:
    workdir = Path(workdir)
    skeleton_path = workdir / "skeleton_blender.json"
    if not skeleton_path.exists():
        raise FileNotFoundError(
            f"[leaf_generator] {skeleton_path} not found -- run align_plant_skeleton.py on this "
            "workdir first (this needs its aligned/scaled output, not estimate_plant_skeleton.py's "
            "raw skeleton.json)."
        )

    with open(skeleton_path) as f:
        skeleton = json.load(f)

    collection = _get_or_create_collection(bpy.context.scene.collection, collection_name)

    keypoint_empties = _create_keypoint_empties(skeleton, collection)
    _create_branch_curves(skeleton, collection, curve_bevel_depth)

    if import_pointcloud:
        _import_pointcloud(workdir / "pointcloud_blender.ply", collection)

    if import_splat:
        _import_splat(workdir / "splat_blender.ply", collection)

    print(
        f"[leaf_generator] Plant scene built under collection '{collection_name}': "
        f"{len(keypoint_empties)} keypoint(s), {len(skeleton['branch_polylines'])} branch curve(s)."
    )
    return collection


def _get_or_create_collection(parent: bpy.types.Collection, name: str) -> bpy.types.Collection:
    existing = parent.children.get(name)
    if existing is not None:
        return existing
    new_collection = bpy.data.collections.new(name)
    parent.children.link(new_collection)
    return new_collection


def _create_keypoint_empties(skeleton: dict, collection: bpy.types.Collection) -> List[bpy.types.Object]:
    empties = []
    for kp in skeleton["keypoints"]:
        empty = bpy.data.objects.new(f"{kp['kind']}_{kp['index']}", None)
        empty.empty_display_type = _TIP_DISPLAY_TYPE if kp["kind"] == "tip" else _BRANCH_DISPLAY_TYPE
        empty.empty_display_size = 0.01
        empty.location = kp["xyz"]
        collection.objects.link(empty)
        empties.append(empty)
    return empties


def _create_branch_curves(skeleton: dict, collection: bpy.types.Collection, bevel_depth: float) -> None:
    for branch in skeleton["branch_polylines"]:
        points = branch["points_xyz"]
        if len(points) < 2:
            continue

        curve_data = bpy.data.curves.new(f"branch_{branch['from_index']}_{branch['to_index']}", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.bevel_depth = bevel_depth

        spline = curve_data.splines.new("POLY")
        spline.points.add(len(points) - 1)
        for i, xyz in enumerate(points):
            spline.points[i].co = (xyz[0], xyz[1], xyz[2], 1.0)

        curve_obj = bpy.data.objects.new(curve_data.name, curve_data)
        collection.objects.link(curve_obj)


def _import_pointcloud(ply_path: Path, collection: bpy.types.Collection) -> None:
    if not ply_path.exists():
        print(f"[leaf_generator] (skipping point cloud import -- {ply_path} not found)")
        return

    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(ply_path))
    elif hasattr(bpy.ops.import_mesh, "ply"):
        bpy.ops.import_mesh.ply(filepath=str(ply_path))
    else:
        print("[leaf_generator] No PLY importer found in this Blender version -- skipping point cloud import.")
        return
    _reparent_new_objects(before, collection)


def _find_kiri_import_ply_op():
    """The KIRI 3DGS Render add-on's import operator lives under the `sna`
    namespace with an auto-generated hash suffix on its bl_idname (e.g.
    `sna.dgs_render_import_ply_e0a3a`) that has changed across the add-on's
    own releases -- so match by prefix instead of hardcoding one hash.
    """
    if not hasattr(bpy.ops, "sna"):
        return None
    for name in dir(bpy.ops.sna):
        if name.startswith("dgs_render_import_ply"):
            return getattr(bpy.ops.sna, name)
    return None


def _import_splat(splat_ply_path: Path, collection: bpy.types.Collection) -> None:
    if not splat_ply_path.exists():
        print(
            f"[leaf_generator] (skipping splat import -- {splat_ply_path} not found; run "
            "train_gaussian_splat.py then align_plant_skeleton.py first if you want one)"
        )
        return

    import_op = _find_kiri_import_ply_op()
    if import_op is None:
        print(
            "[leaf_generator] KIRI 3DGS Render add-on not installed -- skipping splat import. "
            f"Install it (https://github.com/Kiri-Innovation/3dgs-render-blender-addon), then "
            f"re-run, or import {splat_ply_path} manually."
        )
        return

    # The add-on's import operator defaults to "Verts" mode, which imports
    # the raw mesh points only (looks like a plain point cloud). "Faces"
    # mode is what actually appends its KIRI_3DGS_Render_GN geometry-nodes
    # setup that turns each point into a shaded, oriented Gaussian splat.
    if hasattr(bpy.context.scene, "sna_dgs_scene_properties"):
        bpy.context.scene.sna_dgs_scene_properties.import_face_vert = "Faces"

    before = set(bpy.data.objects)
    # No auto-center option on this operator (unlike some older 3DGS Blender
    # importers) -- splat_blender.ply is already aligned/scaled by
    # align_plant_skeleton.py, so there's nothing to disable here.
    import_op(filepath=str(splat_ply_path))
    _reparent_new_objects(before, collection)


def _reparent_new_objects(before: set, collection: bpy.types.Collection) -> None:
    new_objects = [obj for obj in bpy.data.objects if obj not in before]
    for obj in new_objects:
        for existing_collection in list(obj.users_collection):
            existing_collection.objects.unlink(obj)
        collection.objects.link(obj)
