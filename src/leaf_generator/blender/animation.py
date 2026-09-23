"""Animate the camera and key light over the built leaves, for a video.

Two moves, both aimed at showing how a scanned leaf's material behaves as
the lighting angle changes -- which is the thing a still frame can't show,
because a normal and roughness map only reveal themselves when the specular
highlight travels across the surface:

- "dolly" tracks along the row of leaves from above, close enough that each
  leaf roughly fills the frame. This doubles as the answer to the row being
  1.5 m long: rather than framing all 35 leaves at once and getting a green
  stripe, the camera visits them in turn.
- "orbit" circles one subject. Use it on a single leaf (`focus=`) when you
  want to study one material properly.

In both modes the key light sweeps on its own path rather than riding along
with the camera. A light locked to the camera gives a flat, unchanging
highlight -- the whole point here is that the highlight moves.

Keyframes are baked one per frame and set to LINEAR. Dense keys plus the
default Bezier interpolation makes the motion surge between keys, and for an
orbit it also lets the Euler angles wind the wrong way round between
samples; sampling every frame sidesteps both.
"""

from __future__ import annotations

from math import cos, pi, sin
from typing import Optional, Sequence, Tuple

import bpy
from mathutils import Vector

from . import staging

MODES = ("dolly", "orbit")

# Preferred render engines, best-effort in order. EEVEE renders a 240-frame
# sweep in a sensible amount of time where Cycles would not, and the name
# changed in 4.2, so this probes what the running Blender actually offers
# instead of hard-coding one.
_EEVEE_NAMES = ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")


def _set_engine(scene: bpy.types.Scene, names: Sequence[str]) -> Optional[str]:
    available = scene.render.bl_rna.properties["engine"].enum_items.keys()
    for name in names:
        if name in available:
            scene.render.engine = name
            return name
    return None


def _leaf_objects(focus: Optional[str]) -> list:
    needle = (focus or "").lower()
    return [
        obj for obj in bpy.context.scene.objects
        if obj.type == 'MESH' and (not needle or needle in obj.name.lower())
    ]


def _clear_keys(obj: bpy.types.Object) -> None:
    """Drop any previous flythrough so re-running doesn't layer moves."""
    obj.animation_data_clear()


def _key(obj: bpy.types.Object, frame: int) -> None:
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_euler", frame=frame)


def _linearize(obj: bpy.types.Object) -> None:
    if not (obj.animation_data and obj.animation_data.action):
        return
    for fcurve in obj.animation_data.action.fcurves:
        for point in fcurve.keyframe_points:
            point.interpolation = 'LINEAR'


def _aim(obj: bpy.types.Object, position: Vector, target: Vector) -> None:
    obj.location = position
    # target - position, not the reverse: to_track_quat aligns the named axis
    # *along the vector given*, so handing it target->camera aims the lens at
    # empty space behind the camera and renders an empty frame.
    obj.rotation_euler = (target - position).to_track_quat('-Z', 'Y').to_euler()


def flythrough(
    mode: str = "dolly",
    focus: Optional[str] = None,
    frames: int = 240,
    fps: int = 24,
    margin: float = 1.25,
    power_at_1m: float = 100.0,
    light_sweeps: float = 2.0,
    orbit_turns: float = 1.0,
    elevation_deg: float = 55.0,
    frames_per_leaf: Optional[float] = None,
    height_scale: float = 1.0,
) -> Tuple[bpy.types.Object, bpy.types.Object]:
    """Keyframe the staged camera and light across `frames`.

    `focus` narrows to leaves whose object name contains it (e.g. "leaf_17").
    `light_sweeps` is how many times the key light crosses the subject over
    the whole clip -- that count is the knob for how often the highlight
    travels, and so how much material behaviour the clip actually shows.

    Dolly speed is set by `frames_per_leaf` in preference to `frames`: the
    camera covers one leaf per that many frames, so the pace stays the same
    whether a session holds 8 leaves or 35, where a fixed `frames` silently
    speeds up as leaf count grows. `frames` still sets the length when
    `frames_per_leaf` is None, and `fps` trades clip length against pace.

    `height_scale` lifts (>1) or drops (<1) the dolly camera from the height
    that framing the median leaf asks for. Lower gets you closer and more
    raking light; higher shows more leaves at once. Note it is a framing
    control as much as a position one -- moving up makes each leaf smaller.

    Returns the (camera, light) it animated. Nothing is rendered here; call
    `render_video` for that, or just press Spacebar in the viewport.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    objects = _leaf_objects(focus)
    bounds = staging.world_bounds(objects)
    if bounds is None:
        raise RuntimeError(
            f"[leaf_generator] No leaf meshes{f' matching {focus!r}' if focus else ''} "
            "to animate -- run the pipeline first."
        )

    low, high = bounds
    centre = (low + high) / 2.0
    size = high - low

    if mode == "dolly" and frames_per_leaf:
        frames = max(2, int(round(frames_per_leaf * max(len(objects), 1))))

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames
    scene.render.fps = fps

    # Build the camera and light first so they exist and are framed sanely;
    # the keyframes below then overwrite their static placement.
    camera = staging.place_camera(bounds, margin=margin)
    light = staging.place_key_light(bounds, power_at_1m=power_at_1m)
    _clear_keys(camera)
    _clear_keys(light)

    if mode == "dolly":
        _dolly(camera, light, objects, size, frames, margin, light_sweeps,
               power_at_1m, height_scale)
    else:
        _orbit(camera, light, centre, size, frames, margin, orbit_turns,
               elevation_deg, light_sweeps, power_at_1m)

    _linearize(camera)
    _linearize(light)

    scene.frame_set(1)
    print(
        f"[leaf_generator] Flythrough ready: {mode}, {frames} frames @ {fps}fps "
        f"({frames / fps:.1f}s) over {len(objects)} leaf object(s)"
        + (f", {frames / max(len(objects), 1):.1f} frames per leaf" if mode == "dolly" else "")
        + f". Light crosses the subject {light_sweeps:g}x."
    )
    return camera, light


def _dolly(camera, light, objects, size, frames, margin, light_sweeps,
           power_at_1m, height_scale=1.0):
    """Track along the row from above, framing roughly one leaf at a time."""
    # The row is whichever horizontal axis is longer -- laid out along +Y
    # today, but that is the pipeline's choice, not a fact to hard-code.
    row_axis = 0 if size.x > size.y else 1
    cross_axis = 1 - row_axis

    # Per-leaf, not the overall bounding box. One leaf with a bad mask can be
    # an order of magnitude wider than the rest (gaensefuss_1's leaf_3 spans
    # 21 cm against a 1-3 cm typical), and framing the union pulls the camera
    # so far back that every good leaf becomes a speck.
    leaves = []
    for obj in objects:
        bounds = staging.world_bounds([obj])
        if bounds is None:
            continue
        low, high = bounds
        leaves.append(((low + high) / 2.0, max(high.x - low.x, high.y - low.y)))
    if not leaves:
        raise RuntimeError("[leaf_generator] No leaf bounds to dolly along.")

    leaves.sort(key=lambda item: item[0][row_axis])
    centres = [centre for centre, _ in leaves]

    # The median leaf, so a single outlier can't set the shot. An unusually
    # large leaf overflows the frame rather than shrinking all the others.
    typical = sorted(extent for _, extent in leaves)[len(leaves) // 2]
    height = (staging.fit_distance(typical, typical, camera.data, margin) + size.z) * height_scale

    for frame in range(1, frames + 1):
        t = (frame - 1) / max(frames - 1, 1)

        # Walk the leaf centres themselves rather than a straight line down
        # the row, so each leaf passes through the middle of frame even if it
        # sits off-axis in its scan.
        position_in_row = t * (len(centres) - 1)
        index = min(int(position_in_row), len(centres) - 2) if len(centres) > 1 else 0
        target = (centres[index].lerp(centres[index + 1], position_in_row - index)
                  if len(centres) > 1 else Vector(centres[0]))

        position = Vector(target)
        position.z = target.z + height
        # Trail slightly behind the subject so the shot has some depth
        # rather than reading as a flatbed scan.
        position[row_axis] -= height * 0.35
        _aim(camera, position, target)
        _key(camera, frame)

        # The light crosses the row as the camera advances, so the highlight
        # rakes over each leaf instead of sitting in the middle of it.
        phase = sin(2.0 * pi * light_sweeps * t)
        light_distance = height * 0.6
        light_pos = Vector(target)
        light_pos[cross_axis] += phase * typical * 1.5
        light_pos.z = target.z + light_distance
        light.location = light_pos
        light.data.energy = power_at_1m * (light_distance ** 2)
        _key(light, frame)


def _orbit(camera, light, centre, size, frames, margin, turns, elevation_deg, light_sweeps, power_at_1m):
    """Circle the subject, with the light turning at a different rate."""
    extent = max(max(size), 1e-4)
    radius = staging.fit_distance(extent, extent, camera.data, margin)
    elevation = elevation_deg * pi / 180.0

    for frame in range(1, frames + 1):
        t = (frame - 1) / max(frames - 1, 1)
        angle = 2.0 * pi * turns * t
        position = centre + Vector((
            radius * cos(angle) * cos(elevation),
            radius * sin(angle) * cos(elevation),
            radius * sin(elevation),
        ))
        _aim(camera, position, centre)
        _key(camera, frame)

        # Counter-rotating: the highlight sweeps the surface faster than the
        # camera moves, so one orbit shows several passes of the material.
        light_angle = -2.0 * pi * light_sweeps * t
        light_distance = radius * 0.8
        light.location = centre + Vector((
            light_distance * cos(light_angle) * 0.6,
            light_distance * sin(light_angle) * 0.6,
            light_distance * 0.8,
        ))
        light.data.energy = power_at_1m * (light_distance ** 2)
        _key(light, frame)


def render_video(
    filepath: str,
    resolution: Tuple[int, int] = (1280, 720),
    samples: int = 64,
    engine: Optional[str] = None,
    blocking: Optional[bool] = None,
) -> str:
    """Point the scene at an H.264 mp4 and render the current frame range.

    Blender appends the frame range to the name for movie output, so the
    file you get back is the one this returns, not exactly `filepath`.

    By default this hands the render to Blender's modal operator when a UI
    is present, which is what keeps the window usable: it opens the Render
    window with a progress bar and Esc cancels. The blocking form freezes
    Blender solid for the whole render -- minutes, for a few hundred frames
    -- with no progress and no way out, while FFMPEG creates the output file
    immediately, so it looks like a hang with a half-written file. Set
    `blocking=True` only when you need the file to exist by the time this
    returns; in background mode it is always blocking, since there is no
    event loop to be modal in.
    """
    scene = bpy.context.scene
    chosen = _set_engine(scene, [engine] if engine else _EEVEE_NAMES)
    if chosen is None:
        print(f"[leaf_generator] Engine {engine!r} unavailable; leaving {scene.render.engine}.")

    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    if scene.render.engine == 'CYCLES':
        scene.cycles.samples = samples
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = samples

    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'HIGH'
    scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
    # Without this Blender writes a silent audio track some players choke on.
    scene.render.ffmpeg.audio_codec = 'NONE'
    scene.render.filepath = filepath

    frames = scene.frame_end - scene.frame_start + 1
    if blocking is None:
        blocking = bpy.app.background

    print(
        f"[leaf_generator] Rendering {frames} frames ({scene.frame_start}-{scene.frame_end}) "
        f"at {resolution[0]}x{resolution[1]} with {scene.render.engine} to {filepath} ..."
    )
    if blocking:
        if not bpy.app.background:
            print(
                "[leaf_generator] Blocking render: Blender will be unresponsive until "
                "it finishes. Watch the console, not the window."
            )
        bpy.ops.render.render(animation=True)
    else:
        # Modal: returns immediately and renders in Blender's own window, so
        # the UI stays alive and Esc aborts. The file is not complete when
        # this returns.
        bpy.ops.render.render('INVOKE_DEFAULT', animation=True)
        print(
            "[leaf_generator] Render started in Blender's Render window -- "
            "Esc cancels. The mp4 is only complete once it reaches the last frame."
        )
    return filepath
