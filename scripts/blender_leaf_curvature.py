"""Give a flat leaf the curvature of a real one: the wavy edge (and the cup) read off P4g by leaf_curvature.py.

Run inside Blender (4.x; numpy ships with it):

  apply to flat leaves already in a scene (src/leaf_generator's: a scanned leaf on a flat mesh, its
  base at the "<leaf>_attachment" Empty):
    blender leaves.blend --python scripts/blender_leaf_curvature.py -- \\
        --curvature <plant>/debug/curvature/curvature.json [--map curvature_2.json] \\
        [--objects "session_leaf_*"] [--parts wave,cup] [--scale 1] [--seed 0] [--save out.blend]

  the demo (each leaf read off the plant: flat, with its own curvature, and with the plant's model):
    blender -b --python scripts/blender_leaf_curvature.py -- \\
        --curvature <plant>/debug/curvature/curvature.json --demo --save curvature_demo.blend

How a flat leaf is read: its mesh's plane (least-variance axis = its normal), its base (the
attachment Empty, else the --base-end of its long axis) and its tip (the outline point farthest
from the base); s runs base to tip (u = s / length), v across; its half-width per side along s
from the outline. The leaf is rebuilt as a dense grid over its own footprint (GRID cells along
its length; the scanned mask's alpha draws the true edge), its UVs carried over by the affine map
the flat mesh's own UVs follow, then lifted along its normal by:

  wave  --map: that leaf's own wave on the size-free grid (u, r = v / half-width), heights
        scaled by this leaf's half-width over that leaf's. Else the plant's model: per side a
        wave along s with the measured margin spectrum (random phases, --seed), unit deviation,
        times the margin's amplitude (its share of the half-width, at this leaf's local half-width),
        times the envelope's share at r (measured in five bands from midrib to margin).
  cup   --map: that leaf's cup; else the model's rise at the margin (share of the half-width) r^2.

Nothing is tuned here: every number comes from curvature.json (scripts/leaf_curvature.py, the plant's
own leaves); --scale only exaggerates for viewing (default 1 = as measured).
"""
import argparse
import fnmatch
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np

GRID = 200     # grid cells along a rebuilt leaf


def args_after_dashes():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--curvature", required=True, help="leaf_curvature.py's curvature.json")
    ap.add_argument("--map", help="one leaf's curvature_<id>.json: its own curvature instead of the model")
    ap.add_argument("--objects", default="*", help="name pattern of the flat leaves to lift")
    ap.add_argument("--parts", default="wave,cup")
    ap.add_argument("--scale", type=float, default=1.0, help="exaggeration, for viewing only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-end", choices=["min", "max"], default="min",
                    help="without an attachment Empty: which end of the long axis is the base")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--save")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------
# the curvature
# --------------------------------------------------------------------------


class Curvature:
    def __init__(self, curvature_json, map_json=None, parts=("wave", "cup"), scale=1.0, seed=0):
        d = json.loads(Path(curvature_json).read_text())
        self.model = d["model"]
        self.parts, self.scale = set(parts), scale
        self.rng = np.random.default_rng(seed)
        self.map = json.loads(Path(map_json).read_text()) if map_json else None

    def env(self, rho):
        """The wave's envelope at |r|, as a share of the margin's (bands at 0.1 .. 0.9)."""
        m = self.model
        return np.interp(np.abs(rho), [0.0] + m["envelope_rho"] + [1.0],
                         [m["envelope_share_of_margin"][0]] + m["envelope_share_of_margin"] + [1.0])

    def side_wave(self, n, length, half):
        """A unit-deviation wave along s (n samples over `length`), with the measured margin spectrum
        (cycles per half-width, scaled by this leaf's half-width), random phases."""
        m = self.model
        F = np.asarray(m["spectrum_cycles_per_half_width"]) / half      # cycles per unit length
        P = np.asarray(m["spectrum_power"])
        keep = (P > 0) & (F > 0)
        s = np.linspace(0, length, n)
        x = np.zeros(n)
        for f, p in zip(F[keep], P[keep]):
            x += math.sqrt(p) * np.cos(2 * np.pi * f * s + self.rng.uniform(0, 2 * np.pi))
        sd = x.std()
        return x / sd if sd > 0 else x

    def heights(self, u, r, hw_local, hw_max, length):
        """Lift (same units as the leaf) at size-free (u, r); hw_local: this point's side's
        half-width at its s; hw_max: the leaf's half-width."""
        z = np.zeros_like(u)
        if self.map is not None:
            ratio = hw_max / self.map["half_width_mm"]
            U, R = np.asarray(self.map["u"]), np.asarray(self.map["r"])
            for part in ("wave", "cup"):
                if part in self.parts:
                    z += ratio * bilinear(np.asarray(self.map[f"{part}_mm"]), U, R, u, r)
            return self.scale * z
        m = self.model
        if "wave" in self.parts:
            n = 512
            # the margin's measured amplitude is the trace's over r in [1/2, 1]: per unit envelope there
            mean_env = float(np.mean(self.env(np.linspace(0.5, 1.0, 11))))
            a_edge = m["margin_amplitude_per_half_width"] / mean_env
            sides = [self.side_wave(n, length, hw_max) for _ in range(2)]
            k = np.clip((u * (n - 1)).round().astype(int), 0, n - 1)
            X = np.where(r < 0, sides[0][k], sides[1][k])
            z += a_edge * hw_local * self.env(r) * X
        if "cup" in self.parts:
            z += m["cup_rise_per_half_width"] * hw_max * r ** 2
        return self.scale * z


def bilinear(G, U, R, u, r):
    i = np.clip(np.interp(u, U, np.arange(len(U))), 0, len(U) - 1.001)
    j = np.clip(np.interp(r, R, np.arange(len(R))), 0, len(R) - 1.001)
    i0, j0 = i.astype(int), j.astype(int)
    di, dj = i - i0, j - j0
    return (G[i0, j0] * (1 - di) * (1 - dj) + G[i0 + 1, j0] * di * (1 - dj)
            + G[i0, j0 + 1] * (1 - di) * dj + G[i0 + 1, j0 + 1] * di * dj)


# --------------------------------------------------------------------------
# a flat leaf, rebuilt and lifted
# --------------------------------------------------------------------------


def leaf_frame(obj, base_end="min"):
    P = np.array([v.co[:] for v in obj.data.vertices], float)
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    normal = Vt[2]
    base = None
    for ch in obj.children:
        if ch.name.endswith("_attachment"):
            base = np.array((obj.matrix_world.inverted() @ ch.matrix_world.translation)[:])
    if base is None:
        t = (P - c) @ Vt[0]
        base = P[np.argmin(t) if base_end == "min" else np.argmax(t)]
    base = base - ((base - c) @ normal) * normal                      # onto the leaf's plane
    tip = P[np.argmax(np.linalg.norm(P - base, axis=1))]
    axis = tip - base
    axis -= (axis @ normal) * normal
    length = float(np.linalg.norm(axis))
    axis /= length
    across = np.cross(normal, axis)
    return P, base, axis, across, normal, length


def uv_map(obj):
    """The affine map (x, y, z, 1) -> (u, v) the flat mesh's UVs follow, or None."""
    me = obj.data
    if not me.uv_layers:
        return None
    uv = me.uv_layers.active.data
    X, Y = [], []
    for loop in me.loops:
        X.append(list(me.vertices[loop.vertex_index].co) + [1.0])
        Y.append(list(uv[loop.index].uv))
    return np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0]


def outline(P, base, axis, across, length, nb=64):
    """Half-width per side along s (nb bins): from the flat mesh's own vertices."""
    s = (P - base) @ axis
    v = (P - base) @ across
    edges = np.linspace(0, length, nb + 1)
    L, R = np.zeros(nb), np.zeros(nb)
    for i in range(nb):
        sel = (s >= edges[i] - 1e-9) & (s <= edges[i + 1] + 1e-9)
        if sel.any():
            L[i], R[i] = max(0.0, -v[sel].min()), max(0.0, v[sel].max())
    mids = 0.5 * (edges[:-1] + edges[1:])
    for H in (L, R):                                                   # empty bins: from neighbours
        ok = H > 0
        if ok.any():
            H[:] = np.interp(mids, mids[ok], H[ok])
    return mids, L, R


def lift(obj, curvature: Curvature, base_end="min", name=None):
    P, base, axis, across, normal, length = leaf_frame(obj, base_end)
    A = uv_map(obj)
    mids, hl, hr = outline(P, base, axis, across, length)
    hw_max = float(((hl + hr) / 2).max())                            # half the full width, as leaf_curvature.py reads it
    step = length / GRID
    S = np.arange(0, length + step / 2, step)
    V = np.arange(-hl.max() - step, hr.max() + step + step / 2, step)
    sg, vg = np.meshgrid(S, V, indexing="ij")
    left, right = np.interp(sg, mids, hl), np.interp(sg, mids, hr)
    inside = (vg >= -left - step) & (vg <= right + step)
    hw_local = np.where(vg < 0, left, right)
    u = sg / length
    r = np.clip(vg / np.maximum(hw_local, 1e-12), -1, 1)
    z = curvature.heights(u.ravel(), r.ravel(), hw_local.ravel(), hw_max, length).reshape(sg.shape)
    X = base[None, None] + sg[..., None] * axis + vg[..., None] * across + z[..., None] * normal
    flat = base[None, None] + sg[..., None] * axis + vg[..., None] * across
    bm = bmesh.new()
    idx = -np.ones(sg.shape, int)
    verts = []
    for (i, j) in zip(*np.nonzero(inside)):
        idx[i, j] = len(verts)
        verts.append(bm.verts.new(X[i, j]))
    bm.verts.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for i in range(sg.shape[0] - 1):
        for j in range(sg.shape[1] - 1):
            q = [idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]]
            if min(q) < 0:
                continue
            f = bm.faces.new([verts[k] for k in q])
            if A is not None:
                for loop, (a, b) in zip(f.loops, [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]):
                    loop[uv_layer].uv = tuple(np.r_[flat[a, b], 1.0] @ A)
    me = bpy.data.meshes.new(name or f"{obj.name}_curvature")
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    out = bpy.data.objects.new(me.name, me)
    out.matrix_world = obj.matrix_world.copy()
    for mat in obj.data.materials:
        out.data.materials.append(mat)
    for poly in out.data.polygons:
        poly.use_smooth = True
    (obj.users_collection[0] if obj.users_collection else bpy.context.scene.collection).objects.link(out)
    zr = z[inside]
    print(f"[curvature] {obj.name} -> {out.name}: length {length:.4g}, half-width {hw_max:.4g}, "
          f"lift {zr.min():.3g}..{zr.max():.3g} (sd {zr.std():.3g}), {len(verts)} vertices")
    return out


# --------------------------------------------------------------------------
# the demo
# --------------------------------------------------------------------------


def collection(name, parent=None):
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if c.name not in (parent or bpy.context.scene.collection).children:
        (parent or bpy.context.scene.collection).children.link(c)
    return c


def leaf_material():
    m = bpy.data.materials.get("curvature_leaf") or bpy.data.materials.new("curvature_leaf")
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (0.18, 0.42, 0.12, 1)
    b.inputs["Roughness"].default_value = 0.55
    return m


def flat_leaf(name, mj, coll, mm=0.001):
    """A flat grid mesh with a leaf's flattened outline as read off the plant (mm -> m), base at the origin,
    the midline along x; vertices on the midline at both ends so its base and tip read back."""
    U = np.asarray(mj["u"])
    L = mj["midrib_mm"]
    hl, hr = np.asarray(mj["half_width_left_mm"]), np.asarray(mj["half_width_right_mm"])
    step = L / GRID
    S = np.arange(0, L + step / 2, step)
    left, right = np.interp(S / L, U, hl), np.interp(S / L, U, hr)
    V = np.arange(-hl.max() - step, hr.max() + step, step)
    bm = bmesh.new()
    idx = {}
    for i, s in enumerate(S):
        for j, v in enumerate(V):
            if -left[i] <= v <= right[i] or abs(v) < step / 2:
                idx[i, j] = bm.verts.new((s * mm, v * mm, 0.0))
    for (i, j), a in idx.items():
        q = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        if all(k in idx for k in q):
            bm.faces.new([idx[k] for k in q])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(leaf_material())
    coll.objects.link(ob)
    return ob


def demo(args):
    rdir = Path(args.curvature).parent
    d = json.loads(Path(args.curvature).read_text())
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    root = collection("leaf_curvature_demo")
    cols = {k: collection(f"curvature_{k}", root) for k in ("flat", "own", "model", "model_x3")}
    y = 0.0
    for row in d["leaves"]:
        mpath = rdir / f"curvature_{row['leaf']}.json"
        if not mpath.exists() or not row.get("half_width_mm"):
            continue
        mj = json.loads(mpath.read_text())
        flat = flat_leaf(f"leaf_{row['leaf']}_flat", mj, cols["flat"])
        hw = row["half_width_mm"] * 0.001
        y += hw + 0.004
        for k, rel in (("own", Curvature(args.curvature, mpath, args.parts.split(","))),
                       ("model", Curvature(args.curvature, None, args.parts.split(","), 1.0, args.seed + row["leaf"])),
                       ("model_x3", Curvature(args.curvature, None, args.parts.split(","), 3.0, args.seed + row["leaf"]))):
            ob = lift(flat, rel, name=f"leaf_{row['leaf']}_{k}")
            for c in ob.users_collection:
                c.objects.unlink(ob)
            cols[k].objects.link(ob)
            ob.location = ({"own": 1, "model": 2, "model_x3": 3}[k] * 0.08, y, 0)
        flat.location = (0, y, 0)
        y += hw + 0.004
    print(f"[curvature] demo: columns flat | own | model | model x3, one row per leaf of {d['cloud']}")


def main():
    args = args_after_dashes()
    if args.demo:
        demo(args)
    else:
        rel = Curvature(args.curvature, args.map, args.parts.split(","), args.scale, args.seed)
        todo = [o for o in bpy.data.objects if o.type == "MESH" and fnmatch.fnmatch(o.name, args.objects)
                and not o.name.endswith("_curvature")]
        for o in todo:
            lift(o, rel, args.base_end)
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.save).resolve()))
        print(f"[curvature] saved {args.save}")


main()
