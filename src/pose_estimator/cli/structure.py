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

from pose_estimator.ply_io import read_ply_vertices, write_ply_vertices
from pose_estimator.pose import orbit_frame
from pose_estimator.structure import solve_plant_frame
from pose_estimator.structure_labels import build_from_labels
from pose_estimator.structure_viz import write_reprojected_skeleton, write_structure_3d_plot

STEM_RGB = (160, 60, 200)
ROOT_RGB = (240, 140, 40)
PALETTE = np.array([[230, 60, 60], [60, 200, 100], [70, 130, 240], [240, 190, 60],
                    [200, 90, 220], [70, 210, 210], [240, 140, 80], [150, 220, 80]], np.uint8)


def run(
    workdir: Path,
    source: str = "auto",
    contact_voxels: float = 3.0,
    min_leaf_points: int = 150,
) -> dict:
    import pycolmap

    # Prefer the P4b surface over the P4a hull. The hull is a solid bound --
    # 71.5% enclosed interior voxels, near-isotropic local neighbourhoods --
    # so organ labels on it describe volume rather than surface.
    surface_path = workdir / "p4b" / "surface.ply"
    hull_path = workdir / "p4" / "hull_points.ply"
    cloud_path = surface_path if source == "surface" or (
        source == "auto" and surface_path.exists()) else hull_path
    if not cloud_path.exists():
        raise FileNotFoundError(f"{cloud_path} not found -- run pose-hull / pose-surface first")
    print(f"  structure source: {cloud_path}")

    labels_file = workdir / "p4c" / "labels.npy"
    if not labels_file.exists():
        raise SystemExit(
            f"{labels_file} not found. P5 is driven by the P4c organ labels -- "
            "run pose-classify + pose-fuse (or pose-semantic) first.")

    p5_dir = workdir / "p5"
    (p5_dir / "diag").mkdir(parents=True, exist_ok=True)

    fields = read_ply_vertices(cloud_path)
    cloud = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)

    labels = np.load(labels_file)
    votes = np.load(workdir / "p4c" / "votes.npz", allow_pickle=True)
    class_order = [str(x) for x in votes["class_order"]]
    if len(labels) != len(cloud):
        raise SystemExit(
            f"p4c/labels.npy has {len(labels)} entries but {cloud_path.name} has {len(cloud)} "
            "points -- they were produced from different clouds. Re-run pose-fuse.")

    with open(workdir / "p4" / "hull.json") as f:
        voxel = json.load(f)["voxel_size"]
    with open(workdir / "p3" / "poses.json") as f:
        orbit = json.load(f)["orbit"]
    reconstruction = pycolmap.Reconstruction(str(workdir / "p3" / "sparse" / "best"))
    sparse = np.array([p.xyz for p in reconstruction.points3D.values()])

    orbit_origin, orbit_rotation = orbit_frame(orbit)
    frame, clamp = solve_plant_frame(cloud, sparse, orbit_origin, orbit_rotation, voxel)
    upright = frame.apply(cloud)

    if clamp is not None:
        print(f"  clamp line detected as a {voxel * 4:.4f}+ gap in the cloud; origin placed there")
    else:
        print("  NO clamp gap found -- the holder never fully hid the stem on this specimen.")
        print("  Origin falls back to the lowest point, which is NOT the anatomical stem base.")

    print(f"  organ classes: {', '.join(class_order)}")
    structure = build_from_labels(upright, labels, class_order, voxel,
                                  contact_voxels=contact_voxels,
                                  min_leaf_points=min_leaf_points)

    print(f"  stem path {len(structure.stem_path)} nodes, "
          f"{structure.num_leaves} leaf instance(s), "
          f"{0 if structure.root_points is None else len(structure.root_points)} root points")
    for i, axis in enumerate(structure.axes):
        n = int((structure.leaf_ids == i).sum())
        length = float(np.linalg.norm(np.diff(axis, axis=0), axis=1).sum())
        print(f"    leaf {i}: {n:>6} points, axis length {length:.4f}")

    stem_ids = [i for i, n in enumerate(class_order) if n in ("stem", "petiole", "branch")]
    stem_points = upright[np.isin(labels, stem_ids)]

    _write_outputs(p5_dir, structure, stem_points, frame, clamp)
    write_structure_3d_plot(p5_dir / "diag" / "structure_3d.png",
                            structure.leaf_points, structure.leaf_ids,
                            structure.num_leaves, structure.stem_path,
                            structure.root_points)
    write_reprojected_skeleton(p5_dir / "diag", structure.stem_path, structure.axes,
                               frame, reconstruction, workdir / "p1" / "frames")

    report = _evaluate(structure, clamp, frame)
    with open(p5_dir / "p5.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  P5 checks ({'ALL PASSED' if report['all_passed'] else 'FAILURES PRESENT'}):")
    for name, check in report["checks"].items():
        print(f"    [{'PASS' if check['pass'] else 'FAIL'}] {name}: {check['detail']}")
    print(f"\n  artifacts + diagnostics in {p5_dir}")
    return report


def _write_outputs(p5_dir: Path, structure, stem_points: np.ndarray, frame, clamp) -> None:
    graph = {
        "plant_frame": frame.to_dict(),
        "clamp_detected": clamp is not None,
        "origin_definition": ("clamp line (gap under the holder)" if clamp is not None
                              else "lowest point -- clamp not detectable"),
        "source": "p4c labels",
        "stem_path_xyz": structure.stem_path.tolist(),
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
        parts.append((stem_points, np.tile(np.array([STEM_RGB], np.uint8), (len(stem_points), 1))))
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


def _evaluate(structure, clamp, frame) -> dict:
    lengths = [float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()) for a in structure.axes]
    per_leaf = [int((structure.leaf_ids == i).sum()) for i in range(structure.num_leaves)]

    checks = {
        "stem_traced": {
            "pass": len(structure.stem_path) >= 3,
            "detail": f"{len(structure.stem_path)} stem centreline nodes",
        },
        "leaves_found": {
            "pass": structure.num_leaves >= 2,
            "detail": f"{structure.num_leaves} leaf instance(s) split at their stem attachments",
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
    parser.add_argument("--contact-voxels", type=float, default=3.0,
                        help="How close a leaf point must be to a stem point to count as an "
                             "insertion, in hull voxels. Sweeping this on plant_9 gave 178, 73, "
                             "28, 14 and 4 candidate sites at 1.5, 2, 3, 4 and 6 -- no plateau, "
                             "which means the organ labels are still too intermixed for "
                             "adjacency to identify an insertion. Fix the labels, not this.")
    parser.add_argument("--min-leaf-points", type=int, default=150,
                        help="Drop leaf instances smaller than this. Known to remove the tiny "
                             "basal and apex leaves that ground truth says are present.")
    args = parser.parse_args(argv)

    run(workdir=args.workdir, source=args.source,
        contact_voxels=args.contact_voxels, min_leaf_points=args.min_leaf_points)


if __name__ == "__main__":
    main()
