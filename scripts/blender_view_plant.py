"""Build a P5 result into a Blender scene: cloud, stem, midribs, tips.

Three ways to run it.

1. From WSL, letting the wrapper translate paths::

       ./scripts/view_in_blender.sh runs/plant_9

2. From a terminal, by hand::

       blender --python scripts/blender_view_plant.py -- --workdir runs/plant_9

3. From Blender's Text Editor: set WORKDIR just below, then Run Script. On a
   Windows Blender reading a WSL checkout, give the path Windows can resolve
   (\\\\wsl.localhost\\<distro>\\home\\...), not the /home/... form.

What you get, as four collections:

    plant_cloud    the leaf (and unassigned) points, in their own colours
    plant_root     the P4c root points, split out and drawn larger -- they are
                   a couple of percent of the cloud and vanish when merged
    plant_stem_cloud  the P4c stem points (the tissue, not the centreline)
    plant_stem     the stem centreline -- thick, white, unmistakable.
                   On a rosette (thistle, sugar beet) there is no stem: this
                   holds a single sphere at the crown where the leaves meet.
    plant_chords   straight tip-to-base lines, to read the midribs against
    plant_midribs  one curve per leaf, tip to stem, in the P5 leaf colours
    plant_tips     a sphere on each detected leaf tip

Everything is emission-shaded, so it reads the same in solid and rendered
view and needs no lighting set up. Self-contained on purpose: it only uses
`bpy`, `numpy` and the standard library, so Blender's bundled Python can run
it without the repo being importable.

Two rules this file has to follow, both learned the hard way:

- **Never raise SystemExit.** Inside Blender that quits the application, so a
  missing argument closed the whole program instead of printing a complaint.
  Everything here reports and returns.
- **Never call `bpy.ops`.** Operators need a window context that does not
  exist yet while `--python` is running at startup, so they fail there while
  working fine in `--background`. All geometry below is built from data.

Console output is the only feedback: on Windows open it with
Window > Toggle System Console.
"""

import json
import os
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np

# Set this to run from Blender's Text Editor, e.g.
# WORKDIR = r"\\wsl.localhost\Ubuntu-20.04\home\me\repo\runs\plant_9"
WORKDIR = ""

STEM_RGBA = (1.0, 1.0, 1.0, 1.0)          # deliberately not a leaf colour

# The organ colours P5 writes into structure.ply (cli/structure.py's ROOT_RGB
# and STEM_RGB), normalised the way read_ply returns them. Used only to split
# the cloud into separate objects -- the colours themselves come from the file.
ROOT_RGB01 = (240 / 255.0, 140 / 255.0, 40 / 255.0)
STEM_RGB01 = (160 / 255.0, 60 / 255.0, 200 / 255.0)
TIP_RGBA = (0.06, 0.85, 0.30, 1.0)
VOTED_TIP_RGBA = (1.0, 0.35, 0.85, 1.0)   # magenta: found in 2D, not by P5
LEAF_RGBA = [
    (0.90, 0.24, 0.24, 1.0), (0.24, 0.78, 0.39, 1.0), (0.27, 0.51, 0.94, 1.0),
    (0.94, 0.75, 0.24, 1.0), (0.78, 0.35, 0.86, 1.0), (0.27, 0.82, 0.82, 1.0),
    (0.94, 0.55, 0.31, 1.0), (0.59, 0.86, 0.31, 1.0),
]


# --------------------------------------------------------------------------
# Reading the artifacts
# --------------------------------------------------------------------------


def read_ply(path):
    """Minimal PLY reader for the clouds this pipeline writes.

    Blender's own importer builds a mesh and drops the vertex colours, which
    are the whole point here, and the repo's `ply_io` is not importable from
    Blender's bundled Python -- so the header is parsed directly.

    Both encodings are handled because the repo writes both: `ply_io` emits
    ASCII while `cli/hull.py` writes binary little-endian.
    """
    codes = {b"float": "<f4", b"float32": "<f4", b"double": "<f8",
             b"uchar": "u1", b"uint8": "u1", b"int": "<i4", b"int32": "<i4"}

    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")

        encoding, count, fields, seen_vertex = None, 0, [], False
        while True:
            parts = f.readline().split()
            if not parts:
                continue
            if parts[0] == b"end_header":
                break
            if parts[0] == b"format":
                encoding = parts[1]
            elif parts[0] == b"element":
                seen_vertex = parts[1] == b"vertex"
                if seen_vertex:
                    count = int(parts[2])
            elif parts[0] == b"property" and parts[1] != b"list" and seen_vertex:
                fields.append((parts[2].decode(), codes[parts[1]]))

        dtype = np.dtype(fields)
        if encoding == b"binary_little_endian":
            data = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype, count=count)
        elif encoding == b"ascii":
            flat = np.array(f.read().split(), dtype=float)
            flat = flat[:count * len(fields)].reshape(count, len(fields))
            data = {name: flat[:, i] for i, (name, _) in enumerate(fields)}
        else:
            raise ValueError(f"{path}: unsupported PLY encoding {encoding!r}")

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    names = data.dtype.names if hasattr(data, "dtype") else data.keys()
    if "red" in names:
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(float) / 255.0
    else:
        rgb = np.full((len(xyz), 3), 0.7)
    return xyz, rgb


def load(workdir):
    """Returns (xyz, rgb, stem, leaves, graph), or None if the run is not there."""
    workdir = Path(workdir)
    graph_path = workdir / "p5" / "stem_graph.json"
    if not graph_path.exists():
        print(f"[plant] ERROR: {graph_path} not found.")
        print("[plant] Run `pose-structure --workdir <dir>` first, and check the path is")
        print("[plant] one this Blender can resolve (\\\\wsl.localhost\\... on Windows).")
        return None
    with open(graph_path) as f:
        graph = json.load(f)

    cloud_path = workdir / "p5" / "structure.ply"
    xyz, rgb = read_ply(cloud_path) if cloud_path.exists() else (np.zeros((0, 3)), None)

    stem = np.array(graph.get("stem_path_xyz") or []).reshape(-1, 3)
    return xyz, rgb, stem, graph.get("leaves", []), graph, tip_evidence(workdir)


def tip_evidence(workdir):
    """Tip evidence that does not come from P5's own instancing.

    Two independent sources, both written to disk and neither reaching the
    scene -- which is why the apex looked empty even though both had
    something to say about it.

    p4c/tips3d.ply    tips voted from the photographs, SAM2 across many views.
                      Finds leaves that merge into one sheet in 3D, so this is
                      where apex leaves show up at all.
    p5/tip_class.ply  points a DINO seed class named "leaf tip" claimed. Broad
                      regions rather than points, so a prior, not a location.
    """
    out = {}
    voted = workdir / "p4c" / "tips3d.ply"
    if voted.exists():
        xyz, rgb = read_ply(voted)
        # Green carries how well supported each tip is. Drawing every cluster
        # the same size made twelve noise votes look exactly like the seven
        # real tips -- the detection was right and the picture was not.
        out["voted_support"] = rgb[:, 1] if len(rgb) else np.zeros(0)
        graph_path = workdir / "p5" / "stem_graph.json"
        if len(xyz) and graph_path.stat().st_mtime > voted.stat().st_mtime:
            print("[plant] NOTE: p4c/tips3d.ply predates this P5 run -- "
                  "re-run pose-tips to refresh it.")
        out["voted"] = xyz
    prior = workdir / "p5" / "tip_class.ply"
    if prior.exists():
        out["prior"], _ = read_ply(prior)

    # Per-leaf membership, so each instance can be shown, hidden or checked on
    # its own. One merged cloud shows the colours but gives no way to ask
    # "which points does leaf 3 actually own".
    ids_path = workdir / "p5" / "leaf_points.npy"
    xyz_path = workdir / "p5" / "leaf_points_xyz.npy"
    if ids_path.exists() and xyz_path.exists():
        out["leaf_ids"] = np.load(ids_path)
        out["leaf_xyz"] = np.load(xyz_path)
    return out


# --------------------------------------------------------------------------
# Scene building -- data API only, no operators
# --------------------------------------------------------------------------


def emission_material(name, rgba, strength=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = rgba
    emit.inputs["Strength"].default_value = strength
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = rgba          # so solid view matches rendered view
    return mat


def vertex_colour_material(name):
    """Emission driven by the mesh's own Color attribute."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Color"
    links.new(attr.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def collection(name):
    if name in bpy.data.collections:
        existing = bpy.data.collections[name]
        for obj in list(existing.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        if not existing.users_scene:
            bpy.context.scene.collection.children.link(existing)
        return existing
    made = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(made)
    return made


def _matches_colour(rgb, target, tolerance=1.5 / 255.0):
    """Rows of `rgb` equal to `target`, within one 8-bit step.

    P5 writes each organ as a flat colour, so an exact class test is a colour
    test. The tolerance is there because the values round-trip through uint8
    and back to float, not because the classes are fuzzy.
    """
    return np.all(np.abs(np.asarray(rgb) - np.asarray(target)) <= tolerance, axis=1)


def add_point_cloud(xyz, rgb, radius, into, name="plant_cloud", material=None):
    mesh = bpy.data.meshes.new(f"{name}_points")
    mesh.from_pydata([tuple(p) for p in xyz], [], [])
    mesh.update()

    layer = mesh.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="POINT")
    alpha = np.ones((len(xyz), 1))
    layer.data.foreach_set("color", np.hstack([rgb, alpha]).ravel())

    obj = bpy.data.objects.new(name, mesh)
    material = material or vertex_colour_material(f"{name}_mat")
    obj.data.materials.append(material)
    into.objects.link(obj)

    # Geometry Nodes turns the vertices into visible points; bare vertices
    # render as nothing at all.
    modifier = obj.modifiers.new("points", "NODES")
    tree = bpy.data.node_groups.new(f"{name}_nodes", "GeometryNodeTree")
    modifier.node_group = tree
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    group_in = tree.nodes.new("NodeGroupInput")
    group_out = tree.nodes.new("NodeGroupOutput")
    to_points = tree.nodes.new("GeometryNodeMeshToPoints")
    to_points.inputs["Radius"].default_value = radius
    # Points produced by Mesh to Points do NOT inherit the object's material
    # slots -- without this the whole cloud renders untextured, which is why
    # the per-instance colours were on disk, on the mesh, and still invisible.
    set_material = tree.nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = material
    tree.links.new(group_in.outputs[0], to_points.inputs["Mesh"])
    tree.links.new(to_points.outputs["Points"], set_material.inputs["Geometry"])
    tree.links.new(set_material.outputs["Geometry"], group_out.inputs[0])
    return obj


def add_curve(points, name, rgba, radius, into, resolution=3):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = resolution
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for i, p in enumerate(points):
        spline.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)

    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(emission_material(f"{name}_mat", rgba))
    into.objects.link(obj)
    return obj


def add_sphere(centre, name, rgba, radius, into, segments=14, rings=7):
    """A UV sphere built from data.

    Deliberately not `bpy.ops.mesh.primitive_uv_sphere_add`: operators need a
    window context that does not exist while `--python` runs at startup, so
    that call worked in `--background` and silently failed in the GUI.
    """
    u = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    v = np.linspace(0, np.pi, rings + 2)[1:-1]
    verts = [(0.0, 0.0, radius), (0.0, 0.0, -radius)]
    for theta in v:
        for phi in u:
            verts.append((radius * np.sin(theta) * np.cos(phi),
                          radius * np.sin(theta) * np.sin(phi),
                          radius * np.cos(theta)))

    faces = []
    ring0 = 2
    for j in range(segments):                      # north cap
        faces.append((0, ring0 + (j + 1) % segments, ring0 + j))
    for i in range(len(v) - 1):                    # bands
        a, b = 2 + i * segments, 2 + (i + 1) * segments
        for j in range(segments):
            k = (j + 1) % segments
            faces.append((a + j, a + k, b + k, b + j))
    last = 2 + (len(v) - 1) * segments
    for j in range(segments):                      # south cap
        faces.append((1, last + j, last + (j + 1) % segments))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = tuple(float(c) for c in centre)
    obj.data.materials.append(emission_material(f"{name}_mat", rgba))
    into.objects.link(obj)
    return obj


def frame_view(centre, extent):
    """Point every 3D viewport at the plant, without using an operator."""
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                space.region_3d.view_location = tuple(float(c) for c in centre)
                space.region_3d.view_distance = float(extent * 2.2)
                # The plant is well under a metre, so the default 0.1 m near
                # clip would slice straight through it.
                space.clip_start = max(extent * 1e-3, 1e-5)
                space.clip_end = max(extent * 1000.0, 100.0)


def clear_startup_scene():
    """Remove Blender's default Cube, Camera and Light before building.

    This script is always launched into a fresh Blender, so what is in the
    scene at this point is the startup file and nothing of the user's. The
    cube in particular is not harmless decoration: it sits at the origin, at
    a size comparable to a plant, and the framing step below sizes the view
    to everything visible.

    Only the three startup objects are touched, and only when they still look
    like the defaults -- a mesh named "Cube" with 8 vertices. Anything the
    user has since made or renamed is left alone.
    """
    # `scene.objects`, not `scene.collection.objects`: the startup file puts
    # its three objects inside a child collection named "Collection", so the
    # scene's own collection is empty and a direct loop finds nothing.
    scene = bpy.context.scene
    removed = []
    for obj in list(scene.objects):
        default_cube = (obj.type == "MESH" and obj.name == "Cube"
                        and len(obj.data.vertices) == 8)
        if default_cube or (obj.type in {"CAMERA", "LIGHT"}
                            and obj.name in {"Camera", "Light"}):
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
    if removed:
        print("[plant] cleared startup objects: " + ", ".join(removed))


def build(workdir, point_radius=None, stem_radius=None, frame=True):
    clear_startup_scene()
    loaded = load(workdir)
    if loaded is None:
        return None
    xyz, rgb, stem, leaves, graph, extra = loaded

    reference = xyz if len(xyz) else stem
    if not len(reference):
        print("[plant] ERROR: the run contains no points and no stem to draw.")
        return None
    low, high = reference.min(axis=0), reference.max(axis=0)
    extent = float((high - low).max())

    point_radius = point_radius or extent * 0.0016
    stem_radius = stem_radius or extent * 0.010     # bold on purpose
    leaf_radius = stem_radius * 0.45
    tip_radius = stem_radius * 1.6

    if len(xyz):
        # Root and stem go into their own objects rather than being mixed into
        # one cloud. They are a small share of the points -- 956 root of 56,607
        # on thistle3, 1.7% -- so merged in they are invisible in practice and
        # cannot be isolated, soloed or hidden. Root is drawn larger for the
        # same reason.
        if rgb is None:
            add_point_cloud(xyz, rgb, point_radius, collection("plant_cloud"))
        else:
            is_root = _matches_colour(rgb, ROOT_RGB01)
            is_stem = _matches_colour(rgb, STEM_RGB01)
            rest = ~(is_root | is_stem)
            if rest.any():
                add_point_cloud(xyz[rest], rgb[rest], point_radius,
                                collection("plant_cloud"))
            if is_stem.any():
                add_point_cloud(xyz[is_stem], rgb[is_stem], point_radius,
                                collection("plant_stem_cloud"), name="plant_stem_cloud")
            if is_root.any():
                add_point_cloud(xyz[is_root], rgb[is_root], point_radius * 1.8,
                                collection("plant_root"), name="plant_root")
            print(f"[plant] cloud split: {int(rest.sum())} leaf/unassigned, "
                  f"{int(is_stem.sum())} stem, {int(is_root.sum())} root")

    if len(stem) > 1:
        add_curve(stem, "plant_stem_line", STEM_RGBA, stem_radius, collection("plant_stem"))
    elif len(stem) == 1:
        # A rosette: P5 reports its base as one node because the leaves meet at
        # a crown rather than along a stem. Drawn as a curve that would be an
        # elbow of pipe no thistle has, so it gets a sphere instead.
        add_sphere(stem[0], "plant_crown", STEM_RGBA, stem_radius * 2.5,
                   collection("plant_stem"))

    chords = np.array(graph.get("chords_xyz") or []).reshape(-1, 24, 3) \
        if graph.get("chords_xyz") else np.zeros((0, 24, 3))
    if len(chords):
        # Straight tip-to-base reference lines. A midrib that wanders is
        # obvious beside one; alone it just looks like a curve.
        chord_group = collection("plant_chords")
        for i, chord in enumerate(chords):
            add_curve(chord, f"chord_{i:02d}", (0.75, 0.75, 0.75, 1.0),
                      leaf_radius * 0.35, chord_group)

    midribs = collection("plant_midribs")
    tips = collection("plant_tips")
    for leaf in leaves:
        axis = np.array(leaf["axis_xyz"]).reshape(-1, 3)
        colour = LEAF_RGBA[leaf["id"] % len(LEAF_RGBA)]
        if len(axis) > 1:
            add_curve(axis, f"midrib_{leaf['id']:02d}", colour, leaf_radius, midribs)
        if "tip_xyz" in leaf:
            add_sphere(leaf["tip_xyz"], f"tip_{leaf['id']:02d}", TIP_RGBA, tip_radius, tips)

    # Evidence P5's instancing did not produce. Drawn distinctly on purpose:
    # where these disagree with plant_tips is exactly where to look.
    voted = extra.get("voted", np.zeros((0, 3)))
    if len(voted):
        support = extra.get("voted_support", np.ones(len(voted)))
        coll = collection("plant_voted_tips")
        for i, position in enumerate(voted):
            # Radius tracks support, floored so a weak tip stays findable
            # rather than invisible -- it is evidence, just weak evidence.
            # `support` is already 0..1: read_ply normalises colours, and
            # dividing by 255 a second time pinned every sphere to the floor.
            scale = 0.25 + 0.75 * float(np.clip(support[i], 0.0, 1.0))
            add_sphere(position, f"voted_tip_{i:02d}_{int(support[i]):03d}",
                       VOTED_TIP_RGBA, tip_radius * 1.3 * scale, coll)

    ids = extra.get("leaf_ids")
    leaf_xyz = extra.get("leaf_xyz")
    if ids is not None and leaf_xyz is not None and len(ids) == len(leaf_xyz):
        coll = collection("plant_instances")
        for leaf_id in range(int(ids.max()) + 1):
            member = ids == leaf_id
            if not member.any():
                continue
            rgba = LEAF_RGBA[leaf_id % len(LEAF_RGBA)]
            add_point_cloud(leaf_xyz[member],
                            np.tile(np.array([rgba[:3]]), (int(member.sum()), 1)),
                            point_radius * 1.4, coll, name=f"leaf_{leaf_id:02d}",
                            material=emission_material(f"leaf_{leaf_id:02d}_mat", rgba))
        unassigned = ids < 0
        if unassigned.any():
            add_point_cloud(leaf_xyz[unassigned],
                            np.tile(np.array([[0.45, 0.45, 0.45]]), (int(unassigned.sum()), 1)),
                            point_radius, coll, name="leaf_unassigned",
                            material=emission_material("leaf_unassigned_mat",
                                                       (0.45, 0.45, 0.45, 1.0)))

    prior = extra.get("prior", np.zeros((0, 3)))
    if len(prior):
        add_point_cloud(prior, np.tile(np.array([[0.15, 0.9, 0.45]]), (len(prior), 1)),
                        point_radius * 2.2, collection("plant_tip_class"))

    if frame:
        frame_view((low + high) / 2.0, extent)

    print(f"[plant] {len(xyz)} points, stem {len(stem)} nodes, {len(leaves)} leaves")
    print(f"[plant] origin: {graph.get('origin_definition', 'unknown')}")
    print(f"[plant] extent {extent:.3f}, stem radius {stem_radius:.4f}, "
          f"points {point_radius:.4f}")
    print("[plant] collections: plant_cloud, plant_root, plant_stem_cloud, "
          "plant_stem, plant_midribs, plant_tips")
    return {"points": len(xyz), "leaves": len(leaves), "stem_nodes": len(stem)}


def resolve_workdir(argv):
    if WORKDIR:
        return WORKDIR
    if "--" in argv:
        rest = argv[argv.index("--") + 1:]
        if "--workdir" in rest:
            return rest[rest.index("--workdir") + 1]
        if rest:
            return rest[0]
    return os.environ.get("PLANT_WORKDIR", "")


def hide_splash():
    """Stop the splash screen ("what kind of new file?") from opening.

    Startup `--python` scripts run before the first window draw, so clearing
    the preference here takes effect for this launch. It is set on the
    running session only and never saved, so the user's own preference file
    is left as they set it.
    """
    try:
        bpy.context.preferences.view.show_splash = False
    except Exception:
        pass          # a Blender build without the preference is not worth failing over


def main():
    hide_splash()
    target = resolve_workdir(list(sys.argv))
    if not target:
        # Printed, never raised: SystemExit here would close Blender, which is
        # exactly what a missing argument used to do.
        print("[plant] No run directory given, so there is nothing to build.")
        print("[plant] Either edit WORKDIR at the top of this file and run again,")
        print("[plant] or launch as:")
        print("[plant]   blender --python blender_view_plant.py -- --workdir runs/plant_9")
        return
    try:
        build(target)
    except Exception:
        print("[plant] failed while building the scene:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
