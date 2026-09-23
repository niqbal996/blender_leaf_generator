"""Make the built leaves actually inspectable: near clip, camera, key light.

These leaves are 1-6 cm across, and Blender's defaults are all sized for
objects around a metre. Three of those defaults get in the way at this
scale, and this module fixes each one:

- The viewport's 0.1 m near clip plane sits *further away* than the leaf is
  wide, so orbiting in to look at a leaf makes it vanish. This is the "zoom
  stops" symptom: the view is not refusing to move, the geometry is being
  clipped away in front of the camera.
- Blender also slows zoom as `view_distance` shrinks, so a view still
  parked at metre scale crawls the last few centimetres.
- The startup point light is a 1000 W lamp 4 m up, which at 5 cm is both
  far away and aimed at nothing in particular.

Everything here is idempotent: re-running the pipeline reuses the camera
and light it made last time rather than littering the scene with copies.
"""

from __future__ import annotations

from math import tan
from typing import Iterable, Optional, Sequence, Tuple

import bpy
from mathutils import Vector

CAMERA_NAME = "LeafInspectCam"
LIGHT_NAME = "LeafInspectKey"

# Straight down would be the natural view for flat leaves lying in the XY
# plane, but it flattens the normal and roughness maps into nothing. A small
# tilt keeps the silhouette readable while letting the key light rake across
# the surface, which is the whole point of looking at a PBR scan.
VIEW_DIRECTION = Vector((0.0, -0.45, 1.0))


def world_bounds(objects: Iterable[bpy.types.Object]) -> Optional[Tuple[Vector, Vector]]:
    """(min_corner, max_corner) over every object's world-space bounding box."""
    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        if obj.type == 'MESH'
        for corner in obj.bound_box
    ]
    if not corners:
        return None
    return (
        Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners))),
        Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners))),
    )


def frame_viewports(centre: Vector, extent: float, shading: str = 'MATERIAL') -> int:
    """Point every 3D viewport at `centre` and fix its clipping for `extent`.

    Returns how many viewports were adjusted -- zero in background mode,
    where there are no screens, which is why this never uses an operator or
    touches `bpy.context.area`.
    """
    adjusted = 0
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                if space.type != 'VIEW_3D':
                    continue
                space.region_3d.view_location = centre
                space.region_3d.view_distance = extent * 2.0
                # Scaled to the subject, not a fixed constant: the same
                # pipeline builds 1 cm seedling leaves and 1.5 m rows of them.
                space.clip_start = max(extent * 1e-3, 1e-5)
                space.clip_end = max(extent * 1000.0, 100.0)
                space.shading.type = shading
                adjusted += 1
    return adjusted


def fit_distance(size_x: float, size_y: float, camera: bpy.types.Camera, margin: float) -> float:
    """How far a perspective camera must sit back to fit size_x by size_y."""
    scene = bpy.context.scene
    render = scene.render
    aspect = (render.resolution_x * render.pixel_aspect_x) / (
        render.resolution_y * render.pixel_aspect_y
    )

    # `camera.angle` is the FOV across whichever sensor dimension is larger
    # under the default AUTO sensor fit, so which axis it describes flips
    # with the render aspect. Getting this backwards crops a portrait subject.
    half = tan(camera.angle / 2.0)
    if aspect >= 1.0:
        tan_h, tan_v = half, half / aspect
    else:
        tan_h, tan_v = half * aspect, half

    return max(size_x / 2.0 / tan_h, size_y / 2.0 / tan_v) * margin


def place_camera(bounds: Tuple[Vector, Vector], margin: float = 1.15) -> bpy.types.Object:
    """Create (or re-aim) a camera that frames `bounds`, and make it active."""
    low, high = bounds
    centre = (low + high) / 2.0
    size = high - low

    camera_data = bpy.data.cameras.get(CAMERA_NAME) or bpy.data.cameras.new(CAMERA_NAME)
    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None or camera.type != 'CAMERA':
        camera = bpy.data.objects.new(CAMERA_NAME, camera_data)
    if camera.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(camera)

    direction = VIEW_DIRECTION.normalized()
    # The subject is viewed down `direction`, so what the camera has to fit
    # is the bounding box measured across that view, not the raw XY size.
    up = Vector((0.0, 0.0, 1.0))
    right = direction.cross(up)
    if right.length < 1e-9:  # looking straight down; any horizontal axis will do
        right = Vector((1.0, 0.0, 0.0))
    right.normalize()
    screen_up = right.cross(direction).normalized()

    half = size / 2.0
    width = 2.0 * sum(abs(half[i] * right[i]) for i in range(3))
    height = 2.0 * sum(abs(half[i] * screen_up[i]) for i in range(3))
    depth = 2.0 * sum(abs(half[i] * direction[i]) for i in range(3))

    distance = fit_distance(width, height, camera_data, margin) + depth / 2.0
    camera.location = centre + direction * distance
    camera.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()

    # Near/far clip is the same trap as the viewport's, and a camera that
    # clips its subject renders an empty frame with no obvious cause.
    camera_data.clip_start = max(distance * 1e-3, 1e-5)
    camera_data.clip_end = max(distance * 100.0, 100.0)

    bpy.context.scene.camera = camera
    return camera


def place_key_light(
    bounds: Tuple[Vector, Vector],
    power_at_1m: float = 100.0,
    radius_ratio: float = 0.1,
) -> bpy.types.Object:
    """Create (or re-place) a point light above the subject.

    Power follows the inverse-square law off `power_at_1m` so the exposure
    holds whether the subject is one 2 cm leaf or a 1.5 m row of them -- a
    fixed wattage that looks right on the row blows out a single leaf.
    """
    low, high = bounds
    centre = (low + high) / 2.0
    extent = max(max(high - low), 1e-4)

    light_data = bpy.data.lights.get(LIGHT_NAME) or bpy.data.lights.new(LIGHT_NAME, type='POINT')
    light_data.type = 'POINT'
    light = bpy.data.objects.get(LIGHT_NAME)
    if light is None or light.type != 'LIGHT':
        light = bpy.data.objects.new(LIGHT_NAME, light_data)
    if light.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(light)

    distance = extent * 0.8
    light.location = Vector((centre.x, centre.y - extent * 0.3, high.z + distance))
    light_data.energy = power_at_1m * (distance ** 2)
    # A perfect point source gives razor-edged shadows that read as noise on
    # a 2 cm leaf; a little radius softens them without washing out the
    # normal map's relief.
    light_data.shadow_soft_size = extent * radius_ratio

    return light


def stage(
    objects: Sequence[bpy.types.Object],
    margin: float = 1.15,
    power_at_1m: float = 100.0,
) -> None:
    """Frame `objects` in the viewports and set up a camera and key light."""
    bounds = world_bounds(objects)
    if bounds is None:
        print("[leaf_generator] Nothing to frame -- skipping camera/light setup.")
        return

    low, high = bounds
    centre = (low + high) / 2.0
    extent = max(max(high - low), 1e-4)

    viewports = frame_viewports(centre, extent)
    camera = place_camera(bounds, margin=margin)
    light = place_key_light(bounds, power_at_1m=power_at_1m)

    print(
        f"[leaf_generator] Staged view: camera '{camera.name}' and light '{light.name}' "
        f"on a {extent:.3f}m subject; {viewports} viewport(s) re-clipped "
        f"(near clip now {max(extent * 1e-3, 1e-5):.5f}m, so you can zoom right in)."
    )


def focus_on(pattern: str, margin: float = 1.15, power_at_1m: float = 100.0) -> int:
    """Re-aim the camera, light and viewports at the leaves matching `pattern`.

    The leaves are laid out in a row that can run to well over a metre, so a
    camera framing all of them makes each leaf a few pixels tall. This is the
    "look at one leaf properly" counterpart, meant to be called straight from
    Blender's Python console after a run::

        from leaf_generator.blender import staging
        staging.focus_on("leaf_17")

    Matching is a case-insensitive substring of the object name, so
    "gaensefuss_1" frames one session out of several. Returns the number of
    objects matched.
    """
    needle = pattern.lower()
    matched = [
        obj for obj in bpy.context.scene.objects
        if obj.type == 'MESH' and needle in obj.name.lower()
    ]
    if not matched:
        print(f"[leaf_generator] No mesh objects matching {pattern!r} -- nothing to focus.")
        return 0

    stage(matched, margin=margin, power_at_1m=power_at_1m)
    print(f"[leaf_generator] Focused on {len(matched)} object(s) matching {pattern!r}.")
    return len(matched)
