#!/usr/bin/env python
"""Split the P4c labelled cloud into one PLY per organ, for Blender.

    python scripts/export_organ_plys.py --workdir runs/plant_9

Reads p4c/labels.npy + p4c/votes.npz and the cloud P4c labelled. Writes
p4c/objects/:

    stem.ply            all stem points
    root.ply            all root points
    leaf_00.ply ...     one file per leaf instance
    _import.py          Blender script that loads them all, coloured

Why separate files rather than one cloud with vertex colours: Blender imports
per-vertex colour into a `Col` attribute, but nothing in the default viewport
*shows* it -- the object renders flat grey until a material is built that
reads the attribute, which is why the combined PLYs looked uncoloured. One
object per organ sidesteps that entirely: each gets its own material, and
they can be soloed, hidden and measured individually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# RGB, matched to the P4c diagnostics so the 3D view and the images agree.
ORGAN_COLORS = {"stem": (160, 60, 200), "root": (240, 140, 40)}
def distinct_leaf_colors(n, avoid=(0.79, 0.083), margin=0.055):
    """`n` visually separable hues, keeping clear of the stem and root colours.

    A fixed palette runs out of separation quickly -- at 13 leaves the previous
    one repeated three near-identical greens and put one leaf on almost the
    same purple as the stem. Hues are instead spread evenly over the circle
    with the stem (purple) and root (orange) bands excluded, and lightness is
    alternated so neighbouring hues stay distinguishable even when adjacent in
    space.
    """
    import colorsys

    allowed = []
    steps = max(n * 6, 360)
    for i in range(steps):
        hue = i / steps
        if all(min(abs(hue - a), 1.0 - abs(hue - a)) > margin for a in avoid):
            allowed.append(hue)
    if not allowed:
        allowed = [i / max(n, 1) for i in range(max(n, 1))]

    colors = []
    for i in range(n):
        hue = allowed[int(i * len(allowed) / max(n, 1))]
        value = 0.95 if i % 2 == 0 else 0.72
        saturation = 0.85 if i % 3 != 2 else 0.62
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((r, g, b))
    return colors

BLENDER_TEMPLATE = '''"""Auto-generated. Run inside Blender: Text Editor > Open > Alt+P.

Each organ is already a coloured mesh, so this script only imports the files
and sets the viewport to show object colours. No Geometry Nodes, no materials.
"""

import bpy
from pathlib import Path

OBJECTS = {objects!r}

FALLBACK_DIR = Path(r"{objects_dir}")
try:
    DIR = Path(__file__).resolve().parent
    if not (DIR / (OBJECTS[0][0] + ".ply")).exists():
        DIR = FALLBACK_DIR
except NameError:
    DIR = FALLBACK_DIR

if not (DIR / (OBJECTS[0][0] + ".ply")).exists():
    raise SystemExit("No .ply files beside this script, nor at " + str(FALLBACK_DIR))

collection = bpy.data.collections.get("Plant")
if collection is None:
    collection = bpy.data.collections.new("Plant")
    bpy.context.scene.collection.children.link(collection)

for name, rgb in OBJECTS:
    path = DIR / (name + ".ply")
    if not path.exists():
        print("missing: " + str(path))
        continue
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.ply(filepath=str(path))
    for obj in [o for o in bpy.data.objects if o not in before]:
        obj.name = name
        for other in list(obj.users_collection):
            other.objects.unlink(obj)
        collection.objects.link(obj)
        obj.color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0)
        print("imported " + name + "  " + str(len(obj.data.polygons)) + " faces")

# Solid shading coloured by object, and a near-clip small enough for a plant
# well under one unit tall.
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.shading.type = "SOLID"
                space.shading.color_type = "OBJECT"
                space.clip_start = 0.0001

print("[pose_estimator] done. If nothing is visible: select the Plant collection "
      "and press the . key on the numpad (View > Frame Selected).")
'''


def write_ply(path: Path, xyz: np.ndarray, rgb, size: float) -> None:
    """Write each point as a small camera-facing triangle, with vertex colours.

    Bare vertices are the natural encoding for a point cloud and the wrong one
    for Blender: a mesh with no faces has nothing for Solid shading to draw, so
    the object imports correctly and is simply invisible. Turning every point
    into one tiny triangle costs 3x the vertices and makes the cloud render
    everywhere, with no Geometry Nodes, no materials and nothing to configure.
    """
    n = len(xyz)
    # Three offsets forming a small equilateral triangle in the XY plane.
    angles = np.array([0.0, 2.0944, 4.1888])
    offsets = np.stack([np.cos(angles) * size, np.sin(angles) * size, np.zeros(3)], axis=1)

    verts = (xyz[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    colors = np.repeat(np.tile(np.array(rgb, np.uint8), (n, 1)), 3, axis=0)
    faces = np.arange(n * 3, dtype=np.int32).reshape(n, 3)

    with open(path, "wb") as f:
        f.write((
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(verts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(faces)}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        ).encode("ascii"))

        vertex_data = np.empty(len(verts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                                  ("r", "u1"), ("g", "u1"), ("b", "u1")])
        vertex_data["x"], vertex_data["y"], vertex_data["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
        vertex_data["r"], vertex_data["g"], vertex_data["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
        f.write(vertex_data.tobytes())

        face_data = np.empty(len(faces), dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
        face_data["n"] = 3
        face_data["a"], face_data["b"], face_data["c"] = faces[:, 0], faces[:, 1], faces[:, 2]
        f.write(face_data.tobytes())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--instance-radius-voxels", type=float, default=2.5)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from pose_estimator.dino import leaf_instances
    from pose_estimator.ply_io import read_ply_vertices

    p4c = args.workdir / "p4c"
    labels = np.load(p4c / "labels.npy")
    votes = np.load(p4c / "votes.npz", allow_pickle=True)
    class_order = [str(x) for x in votes["class_order"]]

    surface = args.workdir / "p4b" / "surface.ply"
    cloud_path = surface if surface.exists() else args.workdir / "p4" / "hull_points.ply"
    fields = read_ply_vertices(cloud_path)
    points = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)
    if len(points) != len(labels):
        raise SystemExit(f"{cloud_path} has {len(points)} points but labels has {len(labels)}")

    with open(args.workdir / "p4" / "hull.json") as f:
        voxel = json.load(f)["voxel_size"]

    from scipy.spatial import cKDTree
    sample = points[np.random.default_rng(0).choice(len(points), min(4000, len(points)), replace=False)]
    spacing = float(np.median(cKDTree(sample).query(sample, k=2)[0][:, 1]))
    triangle = max(spacing * 0.8, 1e-5)
    print(f"  point spacing {spacing:.5f} -> triangle size {triangle:.5f}")

    out = p4c / "objects"
    out.mkdir(exist_ok=True)
    for stale in out.glob("*.ply"):
        stale.unlink()

    manifest = []
    for index, name in enumerate(class_order):
        if "leaf" in name:
            continue
        hit = labels == index
        if not hit.any():
            continue
        colour = ORGAN_COLORS.get(name, (200, 200, 200))
        write_ply(out / f"{name}.ply", points[hit], colour, triangle)
        manifest.append((name, colour))
        print(f"  {name + '.ply':<20} {int(hit.sum()):>7} points")

    leaf_ids = [i for i, n in enumerate(class_order) if "leaf" in n]
    is_leaf = np.isin(labels, leaf_ids)
    instances = leaf_instances(points[is_leaf], voxel * args.instance_radius_voxels)
    leaf_points = points[is_leaf]

    n_instances = int(instances.max()) + 1 if len(instances) else 0
    leaf_colors = [tuple(int(255 * c) for c in rgb)
                   for rgb in distinct_leaf_colors(n_instances)]
    for instance in range(n_instances):
        hit = instances == instance
        if not hit.any():
            continue
        colour = leaf_colors[instance]
        name = f"leaf_{instance:02d}"
        write_ply(out / f"{name}.ply", leaf_points[hit], colour, triangle)
        manifest.append((name, colour))
        print(f"  {name + '.ply':<20} {int(hit.sum()):>7} points")

    script = out / "_import.py"
    script.write_text(BLENDER_TEMPLATE.format(objects=manifest, objects_dir=str(out.resolve())))
    print(f"\n  {len(manifest)} objects -> {out}")
    print(f"  Blender:  blender --python {script}")
    print("  or paste that file into Blender's Text Editor and press Alt+P")


if __name__ == "__main__":
    main()
