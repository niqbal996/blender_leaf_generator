"""Render a leaf-curvature demo scene (blender_leaf_curvature.py --demo) to a PNG, headless (Cycles, CPU).

    blender -b curvature_demo.blend --python scripts/blender_curvature_render.py -- out.png [elevation_deg] [name pattern]

Objects not matching the pattern are hidden; the camera frames the rest.
"""
import fnmatch
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
elev = math.radians(float(argv[1]) if len(argv) > 1 else 25)
pattern = argv[2] if len(argv) > 2 else "*"
obs = []
for o in bpy.data.objects:
    if o.type == "MESH":
        keep = fnmatch.fnmatch(o.name, pattern)
        o.hide_render = not keep
        if keep:
            obs.append(o)
lo = Vector((min(min((o.matrix_world @ Vector(c))[k] for c in o.bound_box) for o in obs) for k in range(3)))
hi = Vector((max(max((o.matrix_world @ Vector(c))[k] for c in o.bound_box) for o in obs) for k in range(3)))
centre, size = (lo + hi) / 2, (hi - lo).length
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
bpy.context.scene.collection.objects.link(cam)
d = Vector((-0.35, -math.cos(elev), math.sin(elev))).normalized()
cam.location = centre + d * size * 1.05
cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
cam.data.lens = 40
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 4.0
sun.rotation_euler = (math.radians(35), math.radians(-25), 0)
bpy.context.scene.collection.objects.link(sun)
sc = bpy.context.scene
sc.camera = cam
sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = 48
sc.render.resolution_x, sc.render.resolution_y = 1800, 1100
sc.world = sc.world or bpy.data.worlds.new("w")
sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.9, 0.9, 1)
sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
print("rendered", out)
