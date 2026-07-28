"""Build a leaf mesh from a mask's silhouette contour."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import bmesh
import bpy
import cv2


@dataclass
class ContourMeshResult:
    obj: bpy.types.Object
    scale_x: float
    scale_y: float
    image_width: int
    image_height: int


def create_contour_based_mesh(
    mask_path: Union[str, Path],
    mesh_name: str,
    collection: Optional[bpy.types.Collection] = None,
    pixel_size_m: Optional[float] = None,
    fallback_scale: float = 0.08,
) -> Optional[ContourMeshResult]:
    """Build a flat, contour-shaped mesh (with UVs) from a binary mask image.

    If `pixel_size_m` (meters per pixel, from capture calibration) is given,
    the mesh is built at true real-world size -- the full image canvas maps
    to `image_width_px * pixel_size_m` x `image_height_px * pixel_size_m`
    meters, so leaves of different physical sizes end up correctly sized
    relative to each other.

    Without calibration, falls back to normalizing every leaf's longer side
    to `fallback_scale` meters (preserving aspect ratio) -- leaves will
    *not* be relatively correctly sized in this case.
    """
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load mask image: {mask_path}")
        return None

    img_height, img_width = img.shape

    if pixel_size_m is not None:
        scale_x = img_width * pixel_size_m
        scale_y = img_height * pixel_size_m
    else:
        aspect_ratio = img_width / img_height
        scale_x = fallback_scale if img_width > img_height else fallback_scale * aspect_ratio
        scale_y = fallback_scale / aspect_ratio if img_width > img_height else fallback_scale

    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in mask: {mask_path}")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    mesh = bpy.data.meshes.new(mesh_name)
    obj = bpy.data.objects.new(mesh_name, mesh)

    if collection:
        collection.objects.link(obj)
    else:
        bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    contour_verts = []
    for point in simplified_contour:
        x_pixel, y_pixel = point[0]
        x_world = (x_pixel / img_width - 0.5) * scale_x
        y_world = (y_pixel / img_height - 0.5) * scale_y
        contour_verts.append(bm.verts.new((x_world, y_world, 0.0)))

    bm.verts.ensure_lookup_table()

    if len(contour_verts) >= 3:
        try:
            bm.faces.new(contour_verts)
        except ValueError:
            print(f"Face creation failed for {mesh_name}; likely non-manifold or intersecting edges.")

    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=3, use_grid_fill=True)

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    obj.data.update()
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")

    uv_layer = obj.data.uv_layers.active.data
    for poly in obj.data.polygons:
        for loop_index in poly.loop_indices:
            vert_index = obj.data.loops[loop_index].vertex_index
            vert = obj.data.vertices[vert_index].co

            x_pixel = ((vert.x / scale_x) + 0.5) * img_width
            y_pixel = ((vert.y / scale_y) + 0.5) * img_height

            u = x_pixel / img_width
            v = 1.0 - (y_pixel / img_height)
            uv_layer[loop_index].uv = (u, v)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    return ContourMeshResult(
        obj=obj,
        scale_x=scale_x,
        scale_y=scale_y,
        image_width=img_width,
        image_height=img_height,
    )
