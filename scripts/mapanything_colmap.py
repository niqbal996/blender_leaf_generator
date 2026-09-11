"""COLMAP export for MapAnything, with its quality controls actually turned on.

MapAnything's own `scripts/demo_colmap.py` calls `model.infer(...)` with
`apply_mask=True, mask_edges=True` and nothing else, which leaves two of its
noise controls at their permissive defaults:

* `apply_confidence_mask` (default False) -- no confidence filtering at all.
* `use_multiview_confidence` (default False) -- per-pixel learned confidence
  instead of confidence derived from *agreement between views*.

On thistle3 the default export put only 48% of its points inside the P2 plant
silhouette while covering 99.8% of the mask area: it reconstructs the whole
plant and a haze of everything else. The second flag is the one that speaks
to that, since a point no other view corroborates is what the haze is made
of. This script is the upstream exporter with those exposed, reusing their
own `export_predictions_to_colmap` so the model format stays theirs.

`--plant-masks` goes further: P2 already knows which pixels are the subject,
so there is no reason to reconstruct a masked-out background at all.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mv_fusion  # noqa: E402  - a sibling module, not an installed package


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plant-masks", default=None,
                        help="P2 plant masks; geometry outside them is discarded before export")
    parser.add_argument("--apache", action="store_true",
                        help="Use the Apache-2.0 checkpoint instead of the CC-BY-NC one")
    parser.add_argument("--resize-mode", default="fixed_mapping",
                        choices=["fixed_mapping", "longest_side", "square", "fixed_size"],
                        help="fixed_mapping is what the model was trained with; longest_side "
                             "with --size raises resolution at the cost of leaving that regime")
    parser.add_argument("--size", type=int, default=None,
                        help="Required by longest_side/square; ignored by fixed_mapping")
    parser.add_argument("--confidence-percentile", type=float, default=10,
                        help="Drop this bottom percentile of confidence")
    parser.add_argument("--no-confidence-mask", action="store_true",
                        help="Turn confidence filtering off again (the upstream default)")
    parser.add_argument("--no-multiview-confidence", action="store_true",
                        help="Use learned per-pixel confidence instead of cross-view agreement")
    parser.add_argument("--memory-efficient", action="store_true",
                        help="Trade speed for peak memory; unnecessary on a large GPU")
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--intrinsics-from", default=None,
                        help="Give the model the intrinsics instead of letting it guess: a COLMAP "
                             "sparse directory, or a JSON of {frame stem: focal in pixels}. "
                             "MapAnything converts them to ray directions internally")
    parser.add_argument("--poses-from", default=None,
                        help="Give the model the camera poses too: a COLMAP sparse directory. "
                             "It then solves geometry on known cameras instead of predicting "
                             "them, which takes its pose head out of the comparison entirely")
    parser.add_argument("--use-pose-scale", action="store_true",
                        help="Trust the translation magnitudes of --poses-from as metric. A "
                             "COLMAP world is scale-free, so by default only their relative "
                             "geometry is used and the model keeps its own metric scale")
    parser.add_argument("--fuse-views", type=int, default=4,
                        help="Keep only points this many OTHER views corroborate, by their own "
                             "depth maps. 0 exports every masked pixel, which is upstream's "
                             "behaviour and leaves the per-view shells superimposed")
    parser.add_argument("--fuse-tolerance", type=float, default=0.02,
                        help="Depth agreement band, as a fraction of the point's depth")
    parser.add_argument("--points-from", default="depth", choices=["depth", "pointmap"],
                        help="depth: unproject depth_z through the intrinsics and pose that are "
                             "written to the model, so points reproject where they came from. "
                             "pointmap: upstream's behaviour, which exports the separately "
                             "predicted world pointmap instead")
    parser.add_argument("--voxel-fraction", type=float, default=0.002,
                        help="Export voxel size as a fraction of scene extent. Upstream uses "
                             "0.01; a plant's petioles and leaf tips need finer than that")
    return parser.parse_args()


def colmap_cam_to_world(source: str, names) -> dict:
    """Per-frame 4x4 camera-to-world poses from a COLMAP sparse model.

    COLMAP stores world-to-camera; MapAnything wants the inverse, in "any
    world frame", so no alignment or rescaling is needed -- only the
    convention flip.
    """
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(Path(source)))
    poses = {}
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        # A property in pycolmap 3.x, a method in 4.x.
        pose = image.cam_from_world
        if callable(pose):
            pose = pose()
        world_to_cam = np.eye(4)
        world_to_cam[:3, :] = np.asarray(pose.matrix(), dtype=float)
        poses[Path(image.name).stem] = np.linalg.inv(world_to_cam)
    return poses


def check_first_view_has_a_pose(poses: dict, names, source: str) -> None:
    """MapAnything requires view 0 to have a pose if any view does.

    Partial poses are otherwise fine, which matters because SfM routinely
    fails to register a frame or two -- but if the one it dropped happens to
    be the first, the whole pose input has to go.
    """
    first = Path(names[0]).stem
    if first in poses:
        return
    raise SystemExit(
        f"{first} has no pose in {source}, and MapAnything requires the first view to have "
        f"one if any view does.\n"
        f"  {len(poses)} of {len(names)} frames are registered there. Re-solve P3 so the first "
        f"frame registers, or drop --poses-from."
    )


def frame_pixel_intrinsics(source: str, names) -> dict:
    """Per-frame (fx, fy, cx, cy) in original frame pixels, from what is known.

    Two sources, and the difference matters for what a comparison means. P1's
    `intrinsics.json` holds focals the camera itself reported, so using it
    leaves the reconstruction independent of COLMAP. A COLMAP sparse model
    holds focals solved from these very images -- more accurate, but it makes
    the result a COLMAP-calibrated one rather than an independent method.
    """
    path = Path(source)
    stems = [Path(name).stem for name in names]
    if path.is_dir():
        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(path))
        by_stem = {}
        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.images[image_id]
            camera = reconstruction.cameras[image.camera_id]
            K = np.asarray(camera.calibration_matrix(), dtype=float)
            by_stem[Path(image.name).stem] = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        missing = [stem for stem in stems if stem not in by_stem]
        if missing:
            print(f"  {len(missing)} frames are not in {path}; their intrinsics stay predicted "
                  f"(first: {missing[0]})")
        return by_stem
    focals = json.loads(path.read_text())
    return {stem: (float(focals[stem]), float(focals[stem]), None, None)
            for stem in stems if stem in focals}


def processed_intrinsics(frame_intrinsics, frame_size, processed_hw):
    """Frame-pixel intrinsics expressed in the model's working resolution.

    `load_images` scales by the larger of the two ratios and centre-crops, so
    undoing it is exact. A principal point of None means "the frame centre",
    which is what a focal-only source such as EXIF implies.
    """
    fx, fy, cx, cy = frame_intrinsics
    frame_width, frame_height = frame_size
    height, width = processed_hw
    if cx is None:
        cx, cy = frame_width / 2.0, frame_height / 2.0
    scale = max(width / frame_width, height / frame_height)
    crop_x = (round(frame_width * scale) - width) / 2.0
    crop_y = (round(frame_height * scale) - height) / 2.0
    return np.array([[fx * scale, 0.0, cx * scale - crop_x],
                     [0.0, fy * scale, cy * scale - crop_y],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def rebuild_points_from_depth(outputs) -> int:
    """Replace the world pointmap with one the exported cameras agree with.

    MapAnything predicts `pts3d` (world points), `intrinsics` and
    `camera_poses` as separate heads, and upstream's exporter writes points
    from the first and cameras from the other two. Nothing constrains those
    heads to agree exactly, and on thistle3 they did not: only 60% of the
    exported points landed inside the P2 silhouette they were masked to, in
    the model's own coordinate frame -- a cloud that looks unmasked and noisy
    however carefully the pixels were masked.

    Unprojecting `depth_z` through the very intrinsics and pose that get
    written makes reprojection exact by construction, which is the property
    P4 carving and P4c label fusion depend on.
    """
    import torch

    rebuilt = 0
    for prediction in outputs:
        if not all(key in prediction for key in ("depth_z", "intrinsics", "camera_poses")):
            continue
        depth = prediction["depth_z"][0].squeeze(-1)                  # (H, W)
        K = prediction["intrinsics"][0].to(depth.dtype)               # (3, 3)
        cam_to_world = prediction["camera_poses"][0].to(depth.dtype)  # (4, 4)
        height, width = depth.shape
        vs, us = torch.meshgrid(
            torch.arange(height, device=depth.device, dtype=depth.dtype),
            torch.arange(width, device=depth.device, dtype=depth.dtype),
            indexing="ij")
        x = (us - K[0, 2]) / K[0, 0] * depth
        y = (vs - K[1, 2]) / K[1, 1] * depth
        camera_points = torch.stack([x, y, depth], dim=-1)            # (H, W, 3)
        world = camera_points @ cam_to_world[:3, :3].transpose(0, 1) + cam_to_world[:3, 3]
        prediction["pts3d"] = world[None].to(prediction["pts3d"].dtype)
        rebuilt += 1
    return rebuilt


def fuse_masks_by_consistency(outputs, min_views: int, tolerance: float):
    """Drop masked pixels no other view's depth map corroborates.

    A depth error slides a point along the ray it was seen on, so it stays put
    in its own view's image and is only visible to the others. Upstream's
    export writes every masked pixel of every view, which superimposes as many
    slightly-disagreeing shells of the subject as there are views -- on
    thistle3 the median point was inside the silhouette in 16 of 27 views,
    against 25 for an export that fuses.

    Implemented as a mask shrink so upstream's own exporter still does the
    writing: it builds points from `pts3d[mask]`.
    """
    import torch

    depths, intrinsics, extrinsics, masks, points = [], [], [], [], []
    for prediction in outputs:
        depth = prediction["depth_z"][0].squeeze(-1).float().cpu().numpy()
        mask = prediction["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        cam_to_world = prediction["camera_poses"][0].float().cpu().numpy()
        world_to_cam = np.linalg.inv(cam_to_world)[:3, :4]
        depths.append(np.where(mask & (depth > 0), depth, np.nan))
        masks.append(mask)
        intrinsics.append(prediction["intrinsics"][0].float().cpu().numpy())
        extrinsics.append(world_to_cam)
        points.append(prediction["pts3d"][0].float().cpu().numpy())
    depths = np.stack(depths); masks = np.stack(masks)
    intrinsics = np.stack(intrinsics); extrinsics = np.stack(extrinsics)

    before = int(masks.sum())
    kept = 0
    for index, prediction in enumerate(outputs):
        rows, cols = np.nonzero(masks[index])
        if len(rows) == 0:
            continue
        candidates = points[index][rows, cols]
        agrees = mv_fusion.agreement_matrix(candidates, depths, intrinsics, extrinsics,
                                            masks, tolerance)
        agrees[:, index] = False                 # a view cannot corroborate itself
        survives = agrees.sum(axis=1) >= min_views
        fused = np.zeros_like(masks[index])
        fused[rows[survives], cols[survives]] = True
        kept += int(fused.sum())
        current = prediction["mask"]
        keep_t = torch.from_numpy(fused).to(current.device)[None, ..., None]
        prediction["mask"] = current & keep_t
    return kept, before


def main():
    args = parse_args()
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    paths = sorted(glob.glob(os.path.join(args.images_dir, "*")))
    if not paths:
        raise SystemExit(f"no images in {args.images_dir}")
    names = [os.path.basename(path) for path in paths]

    load_kwargs = {"resize_mode": args.resize_mode}
    if args.size is not None:
        load_kwargs["size"] = args.size
    views = load_images(paths, **load_kwargs)
    print(f"Loaded {len(views)} views ({args.resize_mode}"
          f"{f', size {args.size}' if args.size else ''})")

    if args.intrinsics_from:
        from PIL import Image

        known = frame_pixel_intrinsics(args.intrinsics_from, names)
        attached = 0
        for view, path, name_ in zip(views, paths, names):
            entry = known.get(Path(name_).stem)
            if entry is None:
                continue
            processed_hw = tuple(view["img"].shape[-2:])
            K = processed_intrinsics(entry, Image.open(path).size, processed_hw)
            view["intrinsics"] = torch.from_numpy(K).to(view["img"].dtype)[None]
            attached += 1
        example = next((v["intrinsics"][0, 0, 0].item() for v in views if "intrinsics" in v), None)
        print(f"Known intrinsics attached to {attached}/{len(views)} views from "
              f"{args.intrinsics_from}"
              + (f" (focal {example:.1f} in the model's {tuple(views[0]['img'].shape[-2:])} frame)"
                 if example else ""))

    if args.poses_from:
        poses = colmap_cam_to_world(args.poses_from, names)
        check_first_view_has_a_pose(poses, names, args.poses_from)
        attached = 0
        for view, name_ in zip(views, names):
            pose = poses.get(Path(name_).stem)
            if pose is None:
                continue
            view["camera_poses"] = torch.from_numpy(pose).to(view["img"].dtype)[None]
            attached += 1
        print(f"Known poses attached to {attached}/{len(views)} views from {args.poses_from}"
              + ("" if args.use_pose_scale else "; their scale is ignored (a COLMAP world is "
                                                "scale-free, so only relative geometry is used)"))

    name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained(name).to(device)
    print(f"Loaded {name} on {device}")

    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=args.memory_efficient,
            minibatch_size=args.minibatch_size,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            # The two the upstream exporter leaves off.
            apply_confidence_mask=not args.no_confidence_mask,
            confidence_percentile=args.confidence_percentile,
            use_multiview_confidence=not args.no_multiview_confidence,
            ignore_pose_scale_inputs=not args.use_pose_scale,
        )
    print("Inference complete")

    if args.points_from == "depth":
        rebuilt = rebuild_points_from_depth(outputs)
        print(f"Rebuilt pts3d from depth_z for {rebuilt} views, so points and cameras agree")

    if args.plant_masks:
        import cv2

        # export_predictions_to_colmap reads pred["mask"] as (B, H, W, 1), so
        # intersecting the P2 silhouette into it is all that is needed to keep
        # background geometry out of the exported model.
        kept = dropped = 0
        for view_index, prediction in enumerate(outputs):
            if "mask" not in prediction:
                print("  predictions carry no 'mask' field; skipping plant masking")
                break
            current = prediction["mask"]
            height, width = int(current.shape[1]), int(current.shape[2])
            raw = cv2.imread(str(Path(args.plant_masks) / f"{Path(names[view_index]).stem}.png"),
                             cv2.IMREAD_GRAYSCALE)
            if raw is None:
                continue
            plant = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST) > 127
            plant_t = torch.from_numpy(plant).to(current.device)[None, ..., None]
            before = int(current.sum())
            prediction["mask"] = current & plant_t
            kept += int(prediction["mask"].sum())
            dropped += before - int(prediction["mask"].sum())
        print(f"P2 plant masks applied: kept {kept} pixels, dropped {dropped} outside the plant")

    if args.fuse_views > 0:
        kept, before = fuse_masks_by_consistency(outputs, args.fuse_views, args.fuse_tolerance)
        print(f"Multi-view fusion: kept {kept} of {before} masked pixels "
              f"({kept / max(before, 1):.1%}) corroborated by >= {args.fuse_views} other views")

    from mapanything.utils.colmap_export import export_predictions_to_colmap

    os.makedirs(args.output_dir, exist_ok=True)
    export_predictions_to_colmap(outputs=outputs, processed_views=views,
                                 image_names=names, output_dir=args.output_dir,
                                 voxel_fraction=args.voxel_fraction)
    Path(args.output_dir, "mapanything_export.json").write_text(json.dumps({
        "model": name, "resize_mode": args.resize_mode, "size": args.size,
        "confidence_mask": not args.no_confidence_mask,
        "confidence_percentile": args.confidence_percentile,
        "multiview_confidence": not args.no_multiview_confidence,
        "voxel_fraction": args.voxel_fraction,
        "plant_masks_applied": bool(args.plant_masks),
        "points_from": args.points_from,
        "intrinsics_from": args.intrinsics_from,
        "poses_from": args.poses_from,
        "pose_scale_used": args.use_pose_scale,
        "fuse_views": args.fuse_views,
        "fuse_tolerance": args.fuse_tolerance,
    }, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
