"""P4c stage 2 CLI: fuse the per-frame class maps onto the 3D points.

    pose-fuse --workdir runs/plant_9

Reads <workdir>/p4c/class_maps (from pose-classify), p4b/surface.ply (or the
P4a hull), p3/sparse/best and p4/hull.json. Writes into <workdir>/p4c:
    labels.npy          int8 (N,) aligned to the cloud's row order
    votes.npz           per-point weighted votes, raw counts, confidence,
                        mean obliquity, and what plain counting would have said
    labels_vis.ply      cloud coloured by organ class     <- open in Blender
    leaf_instances.ply  leaf points, one colour per leaf   <- open in Blender
    confidence.ply      cloud coloured by vote confidence
    qc.json             acceptance checks

Every view's vote is scaled by how broad-side the surface is to it, so a
camera seeing a leaf edge-on abstains instead of outvoting the cameras that
can actually see the blade. See `pose_estimator.semantic` for why.

This stage does no image understanding at all -- it only moves labels from
pixels to points. That is deliberate: it means it can be validated on its own
by feeding it synthetic class maps whose correct answer is known.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from pose_estimator.classify2d import load_class_map, read_manifest
from pose_estimator.dino import leaf_instances
from pose_estimator.ply_io import read_ply_vertices, write_ply_vertices
from pose_estimator.semantic import (
    accumulate_votes,
    camera_from_colmap,
    cast_votes,
    colors_from_surfels,
    estimate_normals,
    finalise_votes,
    render_points,
    view_weights,
)

CLASS_COLORS = {
    "leaf": (220, 40, 40),        # RGB
    "tiny leaf": (240, 90, 60),
    "stem": (160, 60, 200),
    "petiole": (160, 60, 200),
    "root": (240, 140, 40),
}
SPARE = [(90, 200, 110), (60, 190, 210), (220, 200, 70), (200, 200, 200)]
UNLABELED_RGB = (60, 60, 60)

INSTANCE_PALETTE = np.array([
    [230, 60, 60], [60, 200, 100], [70, 130, 240], [240, 190, 60],
    [200, 90, 220], [70, 210, 210], [240, 140, 80], [150, 220, 80],
    [220, 120, 170], [120, 160, 90], [180, 180, 240], [110, 220, 160],
], np.uint8)


def run(
    workdir: Path,
    source: str = "auto",
    instance_radius_voxels: float = 2.5,
    normal_weighting: bool = True,
) -> dict:
    import pycolmap

    p4c = workdir / "p4c"
    class_map_dir = p4c / "class_maps"
    manifest = read_manifest(p4c)
    class_order = manifest["class_order"]
    print(f"  class maps from the {manifest['backend']} backend: {', '.join(class_order)}")

    # P4b's thin surface if it exists, else P4a's hull. Labelling only needs 3D
    # points and poses, so the hull works -- it is simply a solid, so the
    # coloured cloud is blobbier and its interior points have no surface to be
    # normal to, which makes the obliquity weighting far less meaningful.
    surface = workdir / "p4b" / "surface.ply"
    hull = workdir / "p4" / "hull_points.ply"
    cloud_path = surface if source == "surface" or (
        source == "auto" and surface.exists()) else hull
    if not cloud_path.exists():
        raise SystemExit(f"{cloud_path} not found -- run pose-hull (and ideally pose-surface) first")
    print(f"  labelling {cloud_path}")

    fields = read_ply_vertices(cloud_path)
    points = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)

    normals = None
    if normal_weighting:
        if all(k in fields for k in ("nx", "ny", "nz")):
            normals = np.stack([fields["nx"], fields["ny"], fields["nz"]], axis=1).astype(np.float64)
            print("  per-point normals read from the cloud (fitted by P4b against the photographs)")
        else:
            normals = estimate_normals(points).astype(np.float64)
            print(f"  no normals in {cloud_path.name}; estimated by local PCA")
            if cloud_path == hull:
                print("    WARNING: this is the P4a hull, a solid. Its interior points have no "
                      "surface to be normal to, so the weighting is far less meaningful here "
                      "than on P4b's carved surface.")

    surfel_file = workdir / "p4b" / "surfels.npz"
    if surfel_file.exists():
        surfels = np.load(surfel_file)
        colors = colors_from_surfels(points, surfels["means"], surfels["colors"])
    else:
        # Colours only feed the index-map render, which uses positions alone,
        # so grey is fine when P4b was skipped.
        colors = np.full((len(points), 3), 160, np.uint8)
    with open(workdir / "p4" / "hull.json") as f:
        voxel = json.load(f)["voxel_size"]

    reconstruction = pycolmap.Reconstruction(str(workdir / "p3" / "sparse" / "best"))
    image_ids = sorted(reconstruction.reg_image_ids())
    print(f"  {len(points)} points, {len(image_ids)} registered views")

    tally = accumulate_votes(len(points), len(class_order))
    used = 0
    for n, image_id in enumerate(image_ids):
        image = reconstruction.images[image_id]
        class_map = load_class_map(class_map_dir, Path(image.name).stem)
        if class_map is None:
            continue  # frame was strided out at classify time
        camera = camera_from_colmap(image, reconstruction.cameras[image.camera_id])
        _rgb, index_map = render_points(points, colors, camera)
        weights = view_weights(points, normals, camera) if normals is not None else None
        cast_votes(tally, index_map, class_map, weights)
        used += 1
        if (n + 1) % 12 == 0:
            print(f"    {n + 1}/{len(image_ids)} views")

    if used == 0:
        raise SystemExit(
            f"none of the registered views had a class map in {class_map_dir} -- "
            "run pose-classify on this workdir first")

    result = finalise_votes(tally)
    report = _write_outputs(p4c, points, result, class_order,
                            voxel * instance_radius_voxels, normals is not None)
    report["views_fused"] = used
    report["backend"] = manifest["backend"]

    with open(p4c / "qc.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  P4c checks ({'ALL PASSED' if report['all_passed'] else 'FAILURES PRESENT'}):")
    for name, check in report["checks"].items():
        print(f"    [{'PASS' if check['pass'] else 'FAIL'}] {name}: {check['detail']}")
    print("\n  per-class point counts:")
    for name, count in report["class_counts"].items():
        print(f"    {name:<12} {count:>8}  ({100 * count / max(len(points), 1):.1f}%)")
    fusion = report["fusion"]
    if fusion["normal_weighted"]:
        print(f"\n  normal-weighted fusion moved {fusion['points_relabelled']} points "
              f"({fusion['fraction_relabelled']:.2%}) off what plain vote counting decided")
        print(f"    mean obliquity of the views that saw a point: {fusion['mean_cosine']:.3f}")
        print(f"    points only ever seen edge-on (mean cosine < 0.2): "
              f"{fusion['fraction_grazing_only']:.2%}")
    print(f"\n  {report['num_leaf_instances']} leaf instance(s) by 3D connectivity")
    print(f"  artifacts in {p4c}")
    return report


def _class_color(name: str, index: int):
    return CLASS_COLORS.get(name, SPARE[index % len(SPARE)])


def _write_outputs(p4c: Path, points, result, class_order, instance_radius,
                   normal_weighted: bool) -> dict:
    labels = result.labels
    counts = {name: int((labels == i).sum()) for i, name in enumerate(class_order)}
    counts["unlabeled"] = int((labels < 0).sum())

    rgb = np.tile(np.array(UNLABELED_RGB, np.uint8), (len(points), 1))
    for index, name in enumerate(class_order):
        rgb[labels == index] = _class_color(name, index)
    _ply(p4c / "labels_vis.ply", points, rgb)

    leaf_ids = [i for i, n in enumerate(class_order) if "leaf" in n]
    is_leaf = np.isin(labels, leaf_ids)
    instances = leaf_instances(points[is_leaf], instance_radius)
    num_instances = int(instances.max() + 1) if len(instances) else 0

    leaf_points = points[is_leaf]
    keep = instances >= 0
    instance_rgb = INSTANCE_PALETTE[instances[keep] % len(INSTANCE_PALETTE)]
    _ply(p4c / "leaf_instances.ply", leaf_points[keep], instance_rgb)

    heat = (np.clip(result.confidence, 0, 1) * 255).astype(np.uint8)
    _ply(p4c / "confidence.ply", points,
         np.stack([255 - heat, heat, np.zeros_like(heat)], axis=1))

    np.save(p4c / "labels.npy", labels)
    np.savez(p4c / "votes.npz", point_index=np.arange(len(points), dtype=np.int32),
             n_views_seen=result.n_views_seen, weight=result.weight, count=result.count,
             confidence=result.confidence, mean_cosine=result.mean_cosine,
             labels=labels, unweighted_labels=result.unweighted_labels,
             class_order=np.array(class_order))

    labelled = labels >= 0
    unlabeled_fraction = counts["unlabeled"] / max(len(points), 1)

    # How much the obliquity weighting actually changed. Reported rather than
    # asserted: if it moves nothing the weighting is inert on this specimen,
    # and that is worth seeing rather than hiding behind a passing check.
    moved = labelled & (labels != result.unweighted_labels)
    seen = result.n_views_seen > 0
    grazing_only = seen & (result.mean_cosine < 0.2)
    fusion = {
        "normal_weighted": bool(normal_weighted),
        "points_relabelled": int(moved.sum()),
        "fraction_relabelled": float(moved.sum() / max(labelled.sum(), 1)),
        "mean_cosine": float(result.mean_cosine[seen].mean()) if seen.any() else 0.0,
        "fraction_grazing_only": float(grazing_only.sum() / max(seen.sum(), 1)),
    }

    checks = {
        "most_points_labelled": {
            "pass": unlabeled_fraction < 0.02,
            "detail": f"{unlabeled_fraction:.2%} unlabeled (limit 2%)",
        },
        "all_classes_present": {
            "pass": all(counts[n] > 0 for n in class_order),
            "detail": ", ".join(f"{n} {counts[n]}" for n in class_order),
        },
        "leaf_instances_found": {
            "pass": num_instances >= 2,
            "detail": f"{num_instances} leaf instance(s) by 3D connectivity "
                      f"(radius {instance_radius:.4f})",
        },
        # A point every camera grazed was labelled from evidence no view was in
        # a position to give. It is not an error, but it is the population to
        # distrust, so it is surfaced rather than buried in the cloud.
        "few_points_decided_edge_on": {
            "pass": fusion["fraction_grazing_only"] < 0.10,
            "detail": f"{fusion['fraction_grazing_only']:.2%} of seen points were only ever "
                      f"viewed near edge-on (mean cosine < 0.2, limit 10%)",
        },
    }
    return {
        "num_points": int(len(points)),
        "class_order": class_order,
        "class_counts": counts,
        "unlabeled_fraction": float(unlabeled_fraction),
        "num_leaf_instances": num_instances,
        "fusion": fusion,
        "mean_confidence": float(result.confidence[labelled].mean()) if labelled.any() else 0.0,
        "checks": checks,
        "all_passed": all(c["pass"] for c in checks.values()),
    }


def _ply(path: Path, xyz, rgb) -> None:
    write_ply_vertices(path, {
        "x": xyz[:, 0].astype(np.float32), "y": xyz[:, 1].astype(np.float32),
        "z": xyz[:, 2].astype(np.float32),
        "red": rgb[:, 0].astype(np.uint8), "green": rgb[:, 1].astype(np.uint8),
        "blue": rgb[:, 2].astype(np.uint8),
    })


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared with pose-semantic, which runs the classify stage and this one."""
    parser.add_argument("--source", choices=["auto", "surface", "hull"], default="auto",
                        help="label the P4b thin surface or the P4a hull. 'auto' prefers the "
                             "surface when present")
    parser.add_argument("--instance-radius-voxels", type=float, default=2.5,
                        help="3D connectivity radius for splitting leaves into instances, "
                             "in hull voxels. Connectivity alone cannot separate leaves whose "
                             "blades touch -- P5 splits them at their stem attachments instead")
    parser.add_argument("--no-normal-weighting", action="store_true",
                        help="fall back to plain vote counting, every view equal. The default "
                             "scales each view's vote by how broad-side the surface is to it. "
                             "Use this to A/B the effect")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path)
    add_arguments(parser)
    args = parser.parse_args(argv)

    run(workdir=args.workdir, source=args.source,
        instance_radius_voxels=args.instance_radius_voxels,
        normal_weighting=not args.no_normal_weighting)


if __name__ == "__main__":
    main()
