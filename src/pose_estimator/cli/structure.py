"""P5 CLI: stem centreline and leaf instances from the P4c organ labels.

    pose-structure --workdir runs/plant_9/

Reads <workdir>/p4b/surface.ply (or p4/hull_points.ply), p4c/labels.npy and
<workdir>/p3, writes into <workdir>/p5:
    stem_graph.json      plant frame, stem centreline, per-leaf axes
    leaf_points.npy      per-point leaf id (-1 = not assigned to a leaf)
    leaf_points_xyz.npy  the points those ids index, in the plant frame
    structure.ply        leaf + stem + root points coloured, for Blender
    p5.json              acceptance checks
    diag/                3D structure plot + axes reprojected on the frames

This phase needs P4c labels. The earlier geometry-only path -- kNN graph, MST,
and a `min_branch_fraction` deciding how short an organ could be -- was
removed: it had to be re-calibrated per specimen and never showed a stable
plateau (sweeping it gave leaf counts of 26, 8, 5, 1 with no settled region),
which is precisely the class of per-plant threshold this pipeline exists to
get rid of.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from pose_estimator import cloud_source
from pose_estimator.ply_io import read_ply_vertices, write_ply_vertices
from pose_estimator.pose import orbit_frame
from pose_estimator.structure import solve_plant_frame
from pose_estimator.structure_labels import build_from_labels, tip_reach_shortfall
from pose_estimator.structure_viz import (
    write_instancing_plot,
    write_reprojected_skeleton,
    write_structure_3d_plot,
)

STEM_RGB = (160, 60, 200)
# A rosette has no stem. P4c still labels its crown "stem" -- the crown is
# thick and not lamina, so that is a reasonable thing for a patch classifier
# to say -- but exporting it as a stem draws a organ the plant does not have,
# sitting in among the leaves. On a rosette it is written in the crown's own
# grey instead, which is what it anatomically is.
CROWN_RGB = (200, 200, 200)
ROOT_RGB = (240, 140, 40)
PALETTE = np.array([[230, 60, 60], [60, 200, 100], [70, 130, 240], [240, 190, 60],
                    [200, 90, 220], [70, 210, 210], [240, 140, 80], [150, 220, 80]], np.uint8)


def run(
    workdir: Path,
    source: str = "auto",
    contact_voxels: float = 3.0,
    min_leaf_points: int = 150,
    strict_midribs: bool = False,
    min_tip_depth_voxels: float = 8.0,
    min_persistence_ratio: Optional[float] = None,
    architecture: str = "caulescent",
    geometry_backend: str = cloud_source.BASELINE,
    cloud: Optional[Path] = None,
) -> dict:
    import pycolmap

    # Prefer the P4b surface over the P4a hull. The hull is a solid bound --
    # 71.5% enclosed interior voxels, near-isotropic local neighbourhoods --
    # so organ labels on it describe volume rather than surface.
    chosen = cloud_source.resolve(workdir, geometry_backend, cloud, source)
    cloud_path = chosen.path
    if not cloud_path.exists():
        raise FileNotFoundError(
            f"{cloud_path} not found ({chosen.origin}) -- "
            + ("run pose-hull / pose-surface first" if chosen.is_baseline
               else f"run pose-geometry --backends {geometry_backend} first"))
    print(f"  structure source: {cloud_path} -- {chosen.origin}")

    labels_file = chosen.labels_dir / "labels.npy"
    if not labels_file.exists():
        raise SystemExit(
            f"{labels_file} not found. P5 is driven by the P4c organ labels -- "
            f"run pose-classify + pose-fuse (or pose-semantic)"
            + ("" if chosen.is_baseline
               else f" --geometry-backend {geometry_backend}") + " first.")

    p5_dir = chosen.structure_dir
    (p5_dir / "diag").mkdir(parents=True, exist_ok=True)

    fields = read_ply_vertices(cloud_path)
    cloud_points = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)

    labels = np.load(labels_file)
    votes = np.load(chosen.labels_dir / "votes.npz", allow_pickle=True)
    class_order = [str(x) for x in votes["class_order"]]
    if len(labels) != len(cloud_points):
        raise SystemExit(
            f"{labels_file} has {len(labels)} entries but {cloud_path.name} has {len(cloud_points)} "
            "points -- they were produced from different clouds. Re-run pose-fuse.")

    voxel, voxel_origin = cloud_source.voxel_size(workdir, geometry_backend, cloud_points)
    print(f"  voxel {voxel:.5f} ({voxel_origin})")
    sparse_model, poses_json = cloud_source.geometry(workdir, geometry_backend)
    with open(poses_json) as f:
        orbit = json.load(f)["orbit"]
    reconstruction = pycolmap.Reconstruction(str(sparse_model))
    sparse = np.array([p.xyz for p in reconstruction.points3D.values()])

    orbit_origin, orbit_rotation = orbit_frame(orbit)
    # Holder masks, so which way is up can be read off where the tool grips
    # rather than guessed from a table plane the foliage outvotes.
    cameras = None
    holder_dir = workdir / "p2" / "masks" / "holder"
    if holder_dir.is_dir() and any(holder_dir.iterdir()):
        from pose_estimator.hull import load_carve_cameras

        cameras = load_carve_cameras(reconstruction, workdir / "p2" / "masks" / "plant",
                                     occluder_dir=holder_dir)
    frame, clamp = solve_plant_frame(cloud_points, sparse, orbit_origin, orbit_rotation, voxel,
                                     cameras=cameras)
    upright = frame.apply(cloud_points)

    if clamp is not None:
        print(f"  clamp line detected as a {voxel * 4:.4f}+ gap in the cloud; origin placed there")
    else:
        print("  NO clamp gap found -- the holder never fully hid the stem on this specimen.")
        print("  Origin falls back to the lowest point, which is NOT the anatomical stem base.")

    print(f"  organ classes: {', '.join(class_order)}")
    structure = build_from_labels(upright, labels, class_order, voxel,
                                  contact_voxels=contact_voxels,
                                  min_leaf_points=min_leaf_points,
                                  min_tip_depth_voxels=min_tip_depth_voxels,
                                  min_persistence_ratio=min_persistence_ratio,
                                  architecture=architecture,
                                  strict_midribs=strict_midribs)

    base = structure.instancing.base if structure.instancing is not None else None
    if base is not None:
        ev = base.evidence
        print(f"\n  --architecture rosette: crown located from the geometry")
        print(f"    extremities used  {len(base.extremities)}")
        print(f"    base spread       {ev['base_spread_fraction_of_extent']:.1%} of plant "
              "extent (small = the leaves really do meet at a point)")
        print(f"    base elongation   {ev['base_elongation']:.2f}  (1 = ball, large = curve)")
        print(f"    crown at          {np.round(base.center, 4).tolist()}")

    _report_instancing(structure, voxel)

    print(f"\n  stem path {len(structure.stem_path)} nodes, "
          f"{structure.num_leaves} leaf instance(s), "
          f"{0 if structure.root_points is None else len(structure.root_points)} root points")
    for i, axis in enumerate(structure.axes):
        n = int((structure.leaf_ids == i).sum())
        length = float(np.linalg.norm(np.diff(axis, axis=0), axis=1).sum())
        depth = structure.instancing.depth[structure.leaf_ids == i]
        finite = np.isfinite(depth)
        print(f"    leaf {i}: {n:>6} points, midrib {length:.4f}, "
              f"reaches {depth[finite].max() / voxel:>5.1f} voxels from the stem"
              if finite.any() else f"    leaf {i}: {n:>6} points, midrib {length:.4f}")

    stem_ids = [i for i, n in enumerate(class_order) if n in ("stem", "petiole", "branch")]
    stem_points = upright[np.isin(labels, stem_ids)]

    _write_outputs(p5_dir, structure, stem_points, frame, clamp,
                   stem_rgb=CROWN_RGB if architecture == "rosette" else STEM_RGB)
    _write_instancing_artifacts(p5_dir, structure, voxel)

    # A "leaf tip" seed class, if the seeds defined one, written as its own
    # cloud. It is part of leaf tissue everywhere else (the substring rule
    # folds "leaf tip" into leaf), so this is the only place it is visible.
    tip_ids = [i for i, n in enumerate(class_order) if "tip" in n]
    if tip_ids:
        tip_xyz = upright[np.isin(labels, tip_ids)]
        print(f"  {len(tip_xyz)} points in the tip class -> p5/tip_class.ply")
        if len(tip_xyz):
            _ply(p5_dir / "tip_class.ply", tip_xyz,
                 np.tile(np.array([[40, 220, 90]], np.uint8), (len(tip_xyz), 1)))
    write_structure_3d_plot(p5_dir / "diag" / "structure_3d.png",
                            structure.leaf_points, structure.leaf_ids,
                            structure.num_leaves, structure.stem_path,
                            structure.root_points)
    write_instancing_plot(p5_dir / "diag" / "instancing.png", structure, voxel)
    write_reprojected_skeleton(p5_dir / "diag", structure.stem_path, structure.axes,
                               frame, reconstruction, workdir / "p1" / "frames")

    report = _evaluate(structure, clamp, frame, voxel)
    with open(p5_dir / "p5.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  P5 checks ({'ALL PASSED' if report['all_passed'] else 'FAILURES PRESENT'}):")
    for name, check in report["checks"].items():
        print(f"    [{'PASS' if check['pass'] else 'FAIL'}] {name}: {check['detail']}")
    print(f"\n  artifacts + diagnostics in {p5_dir}")
    return report


def _report_instancing(structure, voxel: float) -> None:
    """Print the split one step at a time, because the leaf count alone never
    says which step went wrong."""
    inst = structure.instancing
    if inst is None:
        return
    stats = inst.to_dict()
    unreachable = stats["leaf_points"] - stats["reachable_from_stem"]

    print("\n  leaf instancing, step by step:")
    print(f"    1. leaf tissue                 {stats['leaf_points']:>7} points")
    print(f"    2. touching the stem           {stats['contact_points']:>7} points "
          f"({100 * stats['contact_points'] / max(stats['leaf_points'], 1):.1f}%) -- the seed for depth")
    if unreachable:
        print(f"       no path back to the stem   {unreachable:>7} points "
              f"({100 * unreachable / max(stats['leaf_points'], 1):.1f}%) -- islands in the surface")
    print(f"    3. deepest point is            {stats['max_depth'] / voxel:>7.0f} voxels of "
          f"tissue from the stem")
    print(f"    4. candidate tips              {stats['candidate_tips']:>7}")
    print(f"    5. survive persistence         {stats['tips_after_merge']:>7}  "
          f"<- a maximum on a blade already counted is dropped here")
    print(f"    6. instances kept              {stats['instances_kept']:>7}"
          + (f"  ({stats['instances_dropped_as_too_small']} dropped as too small)"
             if stats["instances_dropped_as_too_small"] else ""))
    if stats["unassigned_points"]:
        print(f"       unassigned                 {stats['unassigned_points']:>7} points")


def _write_instancing_artifacts(p5_dir: Path, structure, voxel: float) -> None:
    """Dump each intermediate as something openable, not just a number.

    The clouds are the point: `depth.ply` shows what the split is computed
    from, `tips.ply` shows what it decided, and comparing the two explains a
    wrong leaf count faster than any log line.
    """
    inst = structure.instancing
    if inst is None or len(structure.leaf_points) == 0:
        return
    pts = structure.leaf_points

    with open(p5_dir / "instancing.json", "w") as f:
        json.dump({**inst.to_dict(), "voxel_size": voxel,
                   "max_depth_voxels": inst.to_dict()["max_depth"] / voxel,
                   "tips": [{"index": int(t),
                             "xyz": pts[int(t)].tolist(),
                             "depth_voxels": float(inst.depth[int(t)] / voxel)}
                            for t in inst.accepted_tips]}, f, indent=2)

    # geodesic depth from the stem, as a heat map over the leaf tissue
    finite = np.isfinite(inst.depth)
    scaled = np.zeros(len(pts))
    if finite.any():
        top = inst.depth[finite].max()
        scaled[finite] = inst.depth[finite] / max(top, 1e-9)
    heat = (np.clip(scaled, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([heat, np.full_like(heat, 60), 255 - heat], axis=1)
    rgb[~finite] = (255, 0, 255)  # unreachable islands, impossible to miss
    _ply(p5_dir / "depth.ply", pts, rgb)

    # candidate tips vs the ones that survived merging
    if len(inst.candidate_tips):
        accepted = set(int(t) for t in inst.accepted_tips)
        tip_rgb = np.array([[0, 220, 0] if int(t) in accepted else [120, 120, 120]
                            for t in inst.candidate_tips], np.uint8)
        _ply(p5_dir / "tips.ply", pts[inst.candidate_tips], tip_rgb)

    # Indices as well as positions, so a viewer can tell accepted from merged
    # without having to read colours back out of a PLY.
    # Whatever the seeds called a tip class, kept as its own cloud so the
    # DINO tip prior can be looked at directly rather than inferred from the
    # instance colours it is buried in.
    np.savez(p5_dir / "tips.npz",
             candidate=inst.candidate_tips, accepted=inst.accepted_tips,
             group=inst.tip_group, depth=inst.depth[inst.candidate_tips]
             if len(inst.candidate_tips) else np.zeros(0))
    np.save(p5_dir / "leaf_depth.npy", inst.depth)


def _ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    write_ply_vertices(path, {
        "x": xyz[:, 0].astype(np.float32), "y": xyz[:, 1].astype(np.float32),
        "z": xyz[:, 2].astype(np.float32),
        "red": rgb[:, 0].astype(np.uint8), "green": rgb[:, 1].astype(np.uint8),
        "blue": rgb[:, 2].astype(np.uint8)})


def _write_outputs(p5_dir: Path, structure, stem_points: np.ndarray, frame, clamp,
                   stem_rgb=STEM_RGB) -> None:
    graph = {
        "plant_frame": frame.to_dict(),
        "clamp_detected": clamp is not None,
        "origin_definition": ("clamp line (gap under the holder)" if clamp is not None
                              else "lowest point -- clamp not detectable"),
        "source": "p4c labels",
        "stem_path_xyz": structure.stem_path.tolist(),
        # The two ends of an upright plant's stem line, named, so they can be
        # drawn and argued with rather than inferred from the polyline.
        "crown_xyz": None if structure.crown is None else np.asarray(structure.crown).tolist(),
        "heart_xyz": None if structure.heart is None else np.asarray(structure.heart).tolist(),
        "leaves": [
            {
                "id": i,
                "axis_xyz": structure.axes[i].tolist(),
                "attachment_xyz": structure.attachments[i].tolist(),
                "tip_xyz": structure.tips[i].tolist(),
                "num_points": int((structure.leaf_ids == i).sum()),
            }
            for i in range(structure.num_leaves)
        ],
    }
    with open(p5_dir / "stem_graph.json", "w") as f:
        json.dump(graph, f, indent=2)

    # The straight chord from each tip back to the base, as a reference the
    # midrib can be read against. A midrib should bow away from its chord by
    # roughly the leaf's own curvature and no more; one that loops, doubles
    # back or crosses another leaf stands out immediately next to a straight
    # line, where on its own it just looks like a curve.
    if structure.axes and len(structure.stem_path):
        chords, gaps = [], []
        base = structure.stem_path[0]
        for axis in structure.axes:
            axis = np.asarray(axis).reshape(-1, 3)
            if len(axis) < 2:
                continue
            chord = np.linspace(axis[0], axis[-1], 24)
            chords.append(chord)
            arc = float(np.linalg.norm(np.diff(axis, axis=0), axis=1).sum())
            straight = float(np.linalg.norm(axis[-1] - axis[0]))
            gaps.append(arc / max(straight, 1e-9))
        if chords:
            pts = np.vstack(chords)
            _ply(p5_dir / "chords.ply", pts,
                 np.tile(np.array([[210, 210, 210]], np.uint8), (len(pts), 1)))
            graph["chords_xyz"] = [c.tolist() for c in chords]
            graph["arc_over_chord"] = [round(g, 3) for g in gaps]
            with open(p5_dir / "stem_graph.json", "w") as f:
                json.dump(graph, f, indent=2)
            print("  arc/chord per leaf (1.0 = straight; a wandering midrib is >> 1): "
                  + ", ".join(f"{g:.2f}" for g in gaps))

    np.save(p5_dir / "leaf_points.npy", structure.leaf_ids)
    np.save(p5_dir / "leaf_points_xyz.npy", structure.leaf_points)

    # Every organ, not just leaves. Writing only the leaf points left the stem
    # and roots absent from the exported cloud, visible in the plot solely as
    # the fitted centreline -- which is a curve, not the tissue it came from.
    leaf_colors = np.tile(np.array([[200, 200, 200]], np.uint8), (len(structure.leaf_points), 1))
    for i in range(structure.num_leaves):
        leaf_colors[structure.leaf_ids == i] = PALETTE[i % len(PALETTE)]

    parts = [(structure.leaf_points, leaf_colors)]
    if len(stem_points):
        parts.append((stem_points, np.tile(np.array([stem_rgb], np.uint8), (len(stem_points), 1))))
    if structure.root_points is not None and len(structure.root_points):
        parts.append((structure.root_points,
                      np.tile(np.array([ROOT_RGB], np.uint8), (len(structure.root_points), 1))))

    xyz = np.vstack([a for a, _ in parts])
    rgb = np.vstack([c for _, c in parts])
    print(f"  structure.ply: {len(structure.leaf_points)} leaf + {len(stem_points)} stem + "
          f"{0 if structure.root_points is None else len(structure.root_points)} root points")
    write_ply_vertices(p5_dir / "structure.ply", {
        "x": xyz[:, 0].astype(np.float32), "y": xyz[:, 1].astype(np.float32),
        "z": xyz[:, 2].astype(np.float32),
        "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2]})


def _stem_check(structure) -> dict:
    """A rosette has no stem, so demanding a centreline is demanding a fiction.

    Left as a real check for anything that does have stem tissue, because
    there a missing centreline is a genuine failure rather than anatomy.
    """
    inst = structure.instancing
    if inst is not None and inst.architecture == "rosette":
        spread = inst.base.evidence["base_spread_fraction_of_extent"] if inst.base else 0.0
        return {
            "pass": True,
            "detail": (f"not applicable: --architecture rosette, leaves meet at a crown "
                       f"({spread:.1%} of extent) rather than along a stem"),
        }
    if structure.crown is not None and structure.heart is not None:
        # The upright path measures the stem line rather than tracing it, so
        # it is two nodes by construction and a node count says nothing. What
        # can fail is the two ends landing on top of each other, which is what
        # this asks instead.
        length = float(np.linalg.norm(np.asarray(structure.heart) - np.asarray(structure.crown)))
        return {
            "pass": length > 0.0,
            "detail": (f"crown to heart, measured: {length:.4f} long "
                       f"(z {structure.crown[2]:.4f} -> {structure.heart[2]:.4f})"),
        }
    return {
        "pass": len(structure.stem_path) >= 3,
        "detail": f"{len(structure.stem_path)} stem centreline nodes",
    }


def _evaluate(structure, clamp, frame, voxel: float) -> dict:
    lengths = [float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()) for a in structure.axes]
    per_leaf = [int((structure.leaf_ids == i).sum()) for i in range(structure.num_leaves)]
    shortfall = tip_reach_shortfall(
        structure.leaf_points, structure.leaf_ids,
        structure.instancing.accepted_tips if structure.instancing is not None else [],
        structure.stem_path)
    worst = max(shortfall) / voxel if shortfall else 0.0

    checks = {
        "stem_traced": _stem_check(structure),
        "leaves_found": {
            "pass": structure.num_leaves >= 2,
            "detail": (f"{structure.num_leaves} leaf instance(s) from "
                       f"{len(structure.instancing.candidate_tips)} candidate tips"
                       if structure.instancing is not None
                       else f"{structure.num_leaves} leaf instance(s)"),
        },
        "leaves_have_points": {
            "pass": bool(per_leaf) and min(per_leaf) >= 150,
            "detail": f"smallest leaf holds {min(per_leaf) if per_leaf else 0} points",
        },
        "no_degenerate_axes": {
            # A length *ratio* check was here and had to go: it assumed every
            # leaf is roughly the same size, which is exactly false once the
            # small apex and basal leaves are being found on purpose. What
            # still matters is that no axis is degenerate.
            "pass": bool(lengths) and min(lengths) > 0.01 * max(lengths),
            "detail": (f"axis lengths {min(lengths):.4f}..{max(lengths):.4f} "
                       f"({max(lengths) / max(min(lengths), 1e-9):.1f}x spread, expected when "
                       f"small leaves are resolved)" if lengths else "no leaves"),
        },
        # A leaf's tip is the part of it that reaches furthest from the stem.
        # If a tip falls short, the cloud visibly carries on past the marker --
        # which is what a tip sitting on a blade edge looks like in Blender.
        "tips_reach_the_end_of_their_leaf": {
            "pass": worst <= 3.0,
            "detail": f"worst tip falls {worst:.1f} voxels short of its own leaf's "
                      f"furthest point (limit 3)",
        },
        "origin_is_the_clamp_line": {
            "pass": clamp is not None,
            "detail": "clamp gap found" if clamp is not None
            else "no clamp gap; origin is the lowest point, not the anatomical base",
        },
    }
    return {
        "source": "p4c labels",
        "num_leaves": structure.num_leaves,
        "points_per_leaf": per_leaf,
        "axis_lengths": lengths,
        # Per leaf: how steeply it rises, how much of its own chord its points
        # cover, and which midrib construction that earned it. A leaf reading
        # "chord" with high coverage is one --strict-midribs would draw from
        # its points instead.
        "midrib_support": structure.midrib_support,
        "tip_shortfall_voxels": [round(v / voxel, 2) for v in shortfall],
        "plant_frame": frame.to_dict(),
        "checks": checks,
        "all_passed": all(c["pass"] for c in checks.values()),
    }


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path,
                        help="Specimen run directory (needs p3/, p4/ and p4c/)")
    parser.add_argument("--source", choices=["auto", "surface", "hull"], default="auto",
                        help="Which cloud to read. 'auto' prefers the P4b surface when present.")
    parser.add_argument("--geometry-backend", default=cloud_source.BASELINE,
                        help="Skeletonise this P3 backend's branch, reading its labels from "
                             "p4c/experiments/<backend> and writing to p5/experiments/<backend>")
    parser.add_argument("--cloud", type=Path,
                        help="Skeletonise this PLY instead, whatever produced it")
    parser.add_argument("--contact-voxels", type=float, default=3.0,
                        help="How close a leaf point must be to a stem point to count as "
                             "touching it. Only seeds the depth field now -- it no longer "
                             "decides the leaf count, which is what made it critical before.")
    parser.add_argument("--min-leaf-points", type=int, default=150,
                        help="Drop leaf instances smaller than this. Known to remove the tiny "
                             "basal and apex leaves that ground truth says are present.")
    parser.add_argument("--min-tip-depth-voxels", type=float, default=8.0,
                        help="Ignore maxima shallower than this. Low on purpose -- persistence "
                             "does the rejecting, so this only screens out surface noise.")
    parser.add_argument(
        "--strict-midribs", action="store_true",
        help="Fit every midrib from its own leaf's points, however steep or sparsely "
             "reconstructed the leaf is. By default a leaf that is both steep and "
             "poorly covered by its own tissue is drawn as the straight crown-to-tip "
             "chord instead, because a curve fitted to a one-sided sliver waves off "
             "the vein. Use this when the leaves are genuinely upright and well "
             "reconstructed and you would rather have their real curvature -- and "
             "check p5.json's midrib_support, which reports the coverage each leaf "
             "was judged on.")
    parser.add_argument("--architecture", choices=["upright", "rosette", "caulescent"],
                        default="caulescent",
                        help="What kind of plant this is. caulescent (default): an upright "
                             "plant with a central stem; leaf depth is measured from the "
                             "stem tissue. rosette: leaves radiate from a crown at ground "
                             "level with no stem at all (thistle, sugar beet); the crown is "
                             "located geometrically and stem labels are ignored. Not "
                             "inferred -- you know which it is when you shoot it.")
    parser.add_argument("--min-persistence-ratio", type=float, default=None,
                        help="Override the automatic tip cut with a fixed persistence "
                             "ratio. By default the cut is read off this plant: the "
                             "candidates are ranked by persistence and split at the widest "
                             "gap, which separates leaves from bumps without a constant. "
                             "A fixed value is a knife-edge -- on thistle3 the old 0.5 "
                             "default discarded a real leaf scoring 0.48.")
    args = parser.parse_args(argv)

    run(workdir=args.workdir, source=args.source,
        contact_voxels=args.contact_voxels, min_leaf_points=args.min_leaf_points,
        min_tip_depth_voxels=args.min_tip_depth_voxels,
        min_persistence_ratio=args.min_persistence_ratio,
        # "caulescent" is the old name for "upright", kept as an alias so
        # existing commands and scripts keep working.
        architecture=args.architecture,
        strict_midribs=args.strict_midribs,
        geometry_backend=args.geometry_backend, cloud=args.cloud)


if __name__ == "__main__":
    main()
