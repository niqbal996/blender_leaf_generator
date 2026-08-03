"""CLI: solve the COLMAP-frame -> real-world-meters, Z-up alignment for a
plant capture, and bake it into Blender-ready copies of the skeleton, point
cloud, and (if trained) Gaussian Splat.

    pose-align-skeleton --workdir out/plant1/ \\
        --scale-ref-a 3 --scale-ref-b 9 --scale-ref-distance-m 0.084

`--scale-ref-a`/`--scale-ref-b` are `skeleton.json` keypoint indices (the
labeled dots in `skeleton.png`) -- measure the real-world distance between
those same two plant features (calipers, ruler) and pass it as
`--scale-ref-distance-m`. Alternatively pass a precomputed
`--scale-factor` (meters per COLMAP unit) directly. Omitting both still
solves rotation/recentering but leaves scale at 1.0 (not metric) with a
loud warning.

This step reuses `pose-estimate-skeleton`'s `sparse/best/` (no COLMAP
rerun) and is cheap -- safe to re-run any time a better scale reference is
found, without retraining the splat.

Writes into --workdir: `alignment.json`, `skeleton_blender.json`,
`pointcloud_blender.ply`, and (if `splat.ply` exists) `splat_blender.ply`.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from pose_estimator.alignment import solve_alignment, solve_scale_from_reference
from pose_estimator.ply_io import read_ply_vertices, write_ply_vertices
from pose_estimator.reconstruction import get_registered_camera_poses


def run(
    workdir: Path,
    scale_factor: Optional[float] = None,
    scale_ref: Optional[tuple] = None,  # (index_a, index_b, distance_m)
    recenter: bool = True,
) -> None:
    import pycolmap

    skeleton_json_path = workdir / "skeleton.json"
    with open(skeleton_json_path) as f:
        skeleton_summary = json.load(f)

    keypoints_by_index = {kp["index"]: np.array(kp["xyz"]) for kp in skeleton_summary["keypoints"]}

    if scale_factor is not None and scale_ref is not None:
        raise ValueError("provide at most one of --scale-factor or --scale-ref-*")

    if scale_ref is not None:
        idx_a, idx_b, distance_m = scale_ref
        scale = solve_scale_from_reference(keypoints_by_index[idx_a], keypoints_by_index[idx_b], distance_m)
        scale_source = "reference_points"
        scale_reference = {"index_a": idx_a, "index_b": idx_b, "distance_m": distance_m, "scale": scale}
        print(f"Solved scale from reference points {idx_a}/{idx_b}: {scale:.6g} meters per COLMAP unit")
    elif scale_factor is not None:
        scale = scale_factor
        scale_source = "manual"
        scale_reference = None
        print(f"Using manually provided scale: {scale:.6g} meters per COLMAP unit")
    else:
        scale = 1.0
        scale_source = "unset"
        scale_reference = None
        print(
            "WARNING: no --scale-factor or --scale-ref-* given -- rotation/recentering will be "
            "solved, but output is NOT metric (scale=1.0, i.e. still in raw COLMAP units)."
        )

    sparse_best = workdir / "sparse" / "best"
    print(f"Loading camera poses from {sparse_best}...")
    reconstruction = pycolmap.Reconstruction(str(sparse_best))
    camera_centers, viewing_dirs, _names = get_registered_camera_poses(reconstruction)
    print(f"  {len(camera_centers)} registered cameras")

    recenter_point = np.array(skeleton_summary["pointcloud_centroid_colmap"]) if recenter else None
    alignment = solve_alignment(camera_centers, viewing_dirs, scale=scale, recenter_point=recenter_point)

    alignment_record = alignment.to_dict()
    alignment_record["scale_source"] = scale_source
    alignment_record["scale_reference"] = scale_reference
    alignment_record["num_cameras_used_for_pca"] = len(camera_centers)

    alignment_path = workdir / "alignment.json"
    with open(alignment_path, "w") as f:
        json.dump(alignment_record, f, indent=2)
    print(f"  alignment saved to {alignment_path}")

    _bake_skeleton(skeleton_summary, alignment, workdir)
    _bake_pointcloud(workdir, alignment)
    _bake_splat(workdir, alignment)


def _bake_skeleton(skeleton_summary: dict, alignment, workdir: Path) -> None:
    keypoints = [
        {"index": kp["index"], "kind": kp["kind"], "xyz": alignment.apply(np.array(kp["xyz"])).tolist()}
        for kp in skeleton_summary["keypoints"]
    ]
    branch_polylines = [
        {
            "from_index": bp["from_index"],
            "to_index": bp["to_index"],
            "point_indices": bp["point_indices"],
            "points_xyz": alignment.apply(np.array(bp["points_xyz"])).tolist(),
        }
        for bp in skeleton_summary["branch_polylines"]
    ]

    aligned = {"keypoints": keypoints, "edges": skeleton_summary["edges"], "branch_polylines": branch_polylines}
    aligned_path = workdir / "skeleton_blender.json"
    with open(aligned_path, "w") as f:
        json.dump(aligned, f, indent=2)
    print(f"  aligned skeleton saved to {aligned_path}")


def _bake_pointcloud(workdir: Path, alignment) -> None:
    pointcloud_path = workdir / "pointcloud.ply"
    if not pointcloud_path.exists():
        print(f"  (skipping point cloud bake -- {pointcloud_path} not found)")
        return

    fields = read_ply_vertices(pointcloud_path)
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1)
    aligned_xyz = alignment.apply(xyz)

    aligned_fields = dict(fields)
    aligned_fields["x"] = aligned_xyz[:, 0].astype(np.float32)
    aligned_fields["y"] = aligned_xyz[:, 1].astype(np.float32)
    aligned_fields["z"] = aligned_xyz[:, 2].astype(np.float32)

    out_path = workdir / "pointcloud_blender.ply"
    write_ply_vertices(out_path, aligned_fields)
    print(f"  aligned point cloud saved to {out_path}")


def _bake_splat(workdir: Path, alignment) -> None:
    splat_path = workdir / "splat.ply"
    if not splat_path.exists():
        print(f"  (skipping splat bake -- {splat_path} not found; run pose-train-splat first if you want one)")
        return

    fields = read_ply_vertices(splat_path)
    n = len(fields["x"])

    means = np.stack([fields["x"], fields["y"], fields["z"]], axis=1)
    quats_wxyz = np.stack([fields[f"rot_{i}"] for i in range(4)], axis=1)
    log_scales = np.stack([fields[f"scale_{i}"] for i in range(3)], axis=1)

    aligned_means, aligned_quats, aligned_log_scales = alignment.apply_to_gaussians(means, quats_wxyz, log_scales)

    aligned_fields = dict(fields)
    aligned_fields["x"] = aligned_means[:, 0].astype(np.float32)
    aligned_fields["y"] = aligned_means[:, 1].astype(np.float32)
    aligned_fields["z"] = aligned_means[:, 2].astype(np.float32)
    for i in range(4):
        aligned_fields[f"rot_{i}"] = aligned_quats[:, i].astype(np.float32)
    for i in range(3):
        aligned_fields[f"scale_{i}"] = aligned_log_scales[:, i].astype(np.float32)
    # f_dc_*/f_rest_*/opacity are untouched by a rigid+uniform-scale alignment.

    out_path = workdir / "splat_blender.ply"
    write_ply_vertices(out_path, aligned_fields, binary=True)
    print(f"  aligned splat ({n} Gaussians) saved to {out_path}")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path, help="Same --workdir used by pose-estimate-skeleton")
    parser.add_argument("--scale-factor", type=float, help="Precomputed meters-per-COLMAP-unit scale")
    parser.add_argument("--scale-ref-a", type=int, help="First skeleton.json keypoint index for a scale reference")
    parser.add_argument("--scale-ref-b", type=int, help="Second skeleton.json keypoint index for a scale reference")
    parser.add_argument("--scale-ref-distance-m", type=float, help="Measured real-world distance between the two reference points, in meters")
    parser.add_argument("--no-recenter", action="store_true", help="Don't translate the point cloud centroid to the origin")
    args = parser.parse_args(argv)

    scale_ref_args = [args.scale_ref_a, args.scale_ref_b, args.scale_ref_distance_m]
    if any(a is not None for a in scale_ref_args) and not all(a is not None for a in scale_ref_args):
        parser.error("--scale-ref-a, --scale-ref-b, and --scale-ref-distance-m must be given together")
    scale_ref = tuple(scale_ref_args) if scale_ref_args[0] is not None else None
    if args.scale_factor is not None and scale_ref is not None:
        parser.error("provide at most one of --scale-factor or --scale-ref-*")

    run(
        workdir=args.workdir,
        scale_factor=args.scale_factor,
        scale_ref=scale_ref,
        recenter=not args.no_recenter,
    )


if __name__ == "__main__":
    main()
