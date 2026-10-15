"""Check a Blender bundle headless: build its scene with the viewer, list what was drawn.

Run after blender_view_plant.py in the same Blender, from the unpacked bundle's folder,
with the arguments its VERSION.txt gives:

    blender -b --factory-startup --python scripts/blender_view_plant.py \
        --python <repo>/scripts/viewer_test.py -- --workdir gf_1/plant --p5x p5x_p4g \
        --skeleton2d gf_1/plant/p5x,gf_1/plant/s2d

Any Linux Blender works headless (download.blender.org, e.g. Blender4.2/blender-4.2.3-linux-x64.tar.xz).
Prints one line per collection (objects, points/vertices, curves) and VIEWER_TEST OK, or
VIEWER_TEST EMPTY when the scene holds no geometry.
"""
import bpy


def size(ob):
    d = ob.data
    if ob.type == "MESH":
        return len(d.vertices)
    if ob.type == "CURVE":
        return sum(len(s.points) + len(s.bezier_points) for s in d.splines)
    if ob.type == "POINTCLOUD":
        return len(d.points)
    return 0


total = 0
for col in sorted(bpy.data.collections, key=lambda c: c.name):
    obs = list(col.objects)
    if not obs:
        continue
    n = sum(size(o) for o in obs)
    total += n
    kinds = sorted({o.type for o in obs})
    print(f"  {col.name:40s} {len(obs):4d} objects {n:9d} points  {','.join(kinds)}")
print("VIEWER_TEST", "OK" if total else "EMPTY", f"({total} points/vertices in all)")
