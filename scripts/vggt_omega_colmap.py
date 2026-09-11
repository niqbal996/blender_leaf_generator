"""COLMAP export for VGGT-Omega, which ships no exporter of its own.

VGGT (the 2025 model) provides `demo_colmap.py`; VGGT-Omega provides only a
Gradio demo, so this is the adapter that turns its predictions into the
sparse model the rest of P3--P6 consumes.

Three things here are deliberate, and each fixes something measured on
thistle3 with the older VGGT backend:

* **Points are fused across views, not concatenated.** VGGT's no-BA path
  unprojects every view's depth map with that view's pose and writes the
  union: on thistle3 all 34,719 points carried a single-view track and the
  per-view centroids scattered over 99% of the cloud extent -- 22 overlapping
  copies of one plant. Here a point survives only where independent views
  agree about it, which is also what gives the points real multi-view tracks
  for bundle adjustment to work on.
* **Only plant pixels are unprojected.** P2 already knows which pixels are
  the subject. Unprojecting a masked-out background wastes the budget on
  points that are then filtered by confidence at best, and left as noise at
  worst.
* **Intrinsics are written in original frame pixels.** The model works at a
  reduced resolution (512 -> 624x416 for a 3:2 frame); a model exported in
  that space projects into the top-left corner of a full-resolution P2 mask
  and scores exactly 0.000 in-silhouette. MapAnything shipped precisely that
  bug, so it is undone here at the source.

Written as COLMAP text so no pycolmap version is required to produce a model;
pycolmap is imported only for the optional bundle adjustment.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mv_fusion  # noqa: E402  - a sibling module, not an installed package


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", required=True,
                        help="Directory holding images/; sparse/ is written beside it")
    parser.add_argument("--checkpoint", default="vggt_omega_1b_512.pt",
                        help="A local .pt path, or a filename in the gated facebook/VGGT-Omega "
                             "repository to fetch with HF_TOKEN. The 512 checkpoint is the one "
                             "its model card recommends for real captures")
    parser.add_argument("--masks_dir", default=None,
                        help="Optional P2 plant masks; only masked-in pixels are unprojected")
    parser.add_argument("--image-resolution", type=int, default=512,
                        help="Token budget per image. Use the checkpoint's own resolution: "
                             "the 512 checkpoint was trained at 512")
    parser.add_argument("--mode", default="balanced", choices=["balanced", "max_size"],
                        help="balanced keeps the token count near image-resolution^2 and gives "
                             "more pixels than max_size at the same setting")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="Drop pixels below this depth-confidence percentile rank (0-1)")
    parser.add_argument("--consistency-views", type=int, default=3,
                        help="How many other views must agree about a point before it is kept")
    parser.add_argument("--consistency-tolerance", type=float, default=0.01,
                        help="Relative depth agreement, as a fraction of the point's depth")
    parser.add_argument("--max-points-per-view", type=int, default=20000,
                        help="Uniform subsample before fusion, to bound the pairwise cost")
    parser.add_argument("--bundle-adjust", action="store_true",
                        help="Refine poses and points with pycolmap after fusion")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> COLMAP's (qw, qx, qy, qz)."""
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2.0 * np.sqrt(1.0 + trace)
        return np.array([0.25 * scale,
                         (rotation[2, 1] - rotation[1, 2]) / scale,
                         (rotation[0, 2] - rotation[2, 0]) / scale,
                         (rotation[1, 0] - rotation[0, 1]) / scale])
    axis = int(np.argmax(np.diag(rotation)))
    other = [(axis + 1) % 3, (axis + 2) % 3]
    scale = 2.0 * np.sqrt(1.0 + rotation[axis, axis] - sum(rotation[i, i] for i in other))
    quaternion = np.zeros(4)
    quaternion[0] = (rotation[other[1], other[0]] - rotation[other[0], other[1]]) / scale
    quaternion[axis + 1] = 0.25 * scale
    quaternion[other[0] + 1] = (rotation[other[0], axis] + rotation[axis, other[0]]) / scale
    quaternion[other[1] + 1] = (rotation[other[1], axis] + rotation[axis, other[1]]) / scale
    return quaternion


def resolve_checkpoint(checkpoint: str) -> str:
    """A local path, or a file fetched from the gated model repository.

    Access is granted per account, so this needs HF_TOKEN in the environment
    and an approved request; the failure is reported here rather than as a
    404 from inside the loader.
    """
    if Path(checkpoint).is_file():
        return checkpoint
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download("facebook/VGGT-Omega", checkpoint)
    except Exception as exc:                       # noqa: BLE001 - reported verbatim
        raise SystemExit(
            f"could not fetch {checkpoint} from facebook/VGGT-Omega: {exc}\n"
            "  The VGGT-Omega checkpoints are gated: request access on the model page and\n"
            "  export HF_TOKEN, or pass a downloaded file with --checkpoint /path/to.pt"
        )


def load_plant_masks(masks_dir, names, shape_hw):
    """P2 masks resampled to the model's working resolution, or None."""
    if masks_dir is None:
        return None
    import cv2

    height, width = shape_hw
    masks = []
    for name in names:
        path = Path(masks_dir) / f"{Path(name).stem}.png"
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"  no mask for {name}; that view will unproject every pixel")
            masks.append(np.ones((height, width), bool))
            continue
        masks.append(cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST) > 127)
    return np.stack(masks)


def unproject(depth, intrinsic, extrinsic):
    """Per-pixel depth -> world points, and the world->camera pose it came from.

    `extrinsic` is 3x4 camera-from-world, so the camera centre is -R^T t.
    """
    height, width = depth.shape
    us, vs = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    x = (us - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (vs - intrinsic[1, 2]) / intrinsic[1, 1]
    camera_points = np.stack([x * depth, y * depth, depth], axis=-1)
    rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
    world = (camera_points.reshape(-1, 3) - translation) @ rotation
    return world.reshape(height, width, 3)


def fuse_by_consistency(points, depths, intrinsics, extrinsics, masks,
                        min_views, tolerance, max_points):
    """Keep points several views independently agree about, with their tracks.

    The test is the standard multi-view geometric one: push a candidate point
    into another view, and ask whether that view's own depth map puts a
    surface at the same distance. A point that only one view believes in is
    exactly what produced the duplicate-plant artifact, and it is dropped
    here rather than written out.
    """
    num_views, height, width = depths.shape
    kept_xyz, kept_tracks = [], []
    for index in range(num_views):
        valid = np.isfinite(depths[index]) & (depths[index] > 0)
        if masks is not None:
            valid &= masks[index]
        rows, cols = np.nonzero(valid)
        if len(rows) == 0:
            continue
        if len(rows) > max_points:
            pick = np.linspace(0, len(rows) - 1, max_points).astype(int)
            rows, cols = rows[pick], cols[pick]
        candidates = points[index][rows, cols]

        agrees = mv_fusion.agreement_matrix(candidates, depths, intrinsics, extrinsics,
                                            masks, tolerance)
        agrees[:, index] = False              # a view cannot corroborate itself
        survivors = np.nonzero(agrees.sum(axis=1) >= min_views)[0]
        for position in survivors:
            track = [(index, int(cols[position]), int(rows[position]))]
            for other in np.nonzero(agrees[position])[0]:
                u, v, _ = mv_fusion.reprojected_pixels(
                    candidates[position][None], intrinsics[other], extrinsics[other],
                    (height, width))
                track.append((int(other), int(u[0]), int(v[0])))
            kept_xyz.append(candidates[position])
            kept_tracks.append(track)
    if not kept_xyz:
        return np.zeros((0, 3)), []
    return np.asarray(kept_xyz), kept_tracks


def write_colmap_text(out_dir, names, sizes, intrinsics, extrinsics, xyz, tracks, colors):
    """A COLMAP text model in original-frame pixels, with real frame names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cameras.txt", "w") as handle:
        for index, (width, height) in enumerate(sizes):
            K = intrinsics[index]
            handle.write(f"{index + 1} PINHOLE {width} {height} "
                         f"{K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")

    # A track element is (image, index *within that image's POINTS2D list*),
    # and COLMAP verifies the two sides agree on load -- so the indices have
    # to be recorded while the per-image lists are being built.
    observations = {index: [] for index in range(len(names))}
    track_elements = []
    for point_id, track in enumerate(tracks, start=1):
        elements = []
        for view, u, v in track:
            elements.append((view, len(observations[view])))
            observations[view].append((u, v, point_id))
        track_elements.append(elements)
    with open(out_dir / "images.txt", "w") as handle:
        for index, name in enumerate(names):
            qw, qx, qy, qz = quaternion_from_matrix(extrinsics[index][:3, :3])
            tx, ty, tz = extrinsics[index][:3, 3]
            handle.write(f"{index + 1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {index + 1} {name}\n")
            handle.write(" ".join(f"{u} {v} {pid}" for u, v, pid in observations[index]) + "\n")

    with open(out_dir / "points3D.txt", "w") as handle:
        for point_id, (point, elements, color) in enumerate(zip(xyz, track_elements, colors), start=1):
            track_text = " ".join(f"{view + 1} {index}" for view, index in elements)
            handle.write(f"{point_id} {point[0]} {point[1]} {point[2]} "
                         f"{color[0]} {color[1]} {color[2]} 0 {track_text}\n")


def main():
    args = parse_args()
    import torch
    from PIL import Image

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    scene = Path(args.scene_dir)
    paths = sorted(glob.glob(str(scene / "images" / "*")))
    if not paths:
        raise SystemExit(f"no images in {scene / 'images'}")
    names = [os.path.basename(path) for path in paths]
    frame_sizes = [Image.open(path).size for path in paths]     # (width, height)
    print(f"Loading {len(paths)} images at resolution {args.image_resolution} ({args.mode})")

    images = load_and_preprocess_images(paths, mode=args.mode,
                                        image_resolution=args.image_resolution).to(args.device)
    print(f"  model input {tuple(images.shape)}")

    checkpoint = resolve_checkpoint(args.checkpoint)
    model = VGGTOmega().to(args.device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    print(f"  loaded {checkpoint}")

    with torch.inference_mode():
        predictions = model(images[None] if images.dim() == 4 else images)
    processed_hw = predictions["images"].shape[-2:]
    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], processed_hw)
    extrinsics = extrinsics.squeeze(0).float().cpu().numpy()
    intrinsics = intrinsics.squeeze(0).float().cpu().numpy()
    depth = predictions["depth"].squeeze(0).float().cpu().numpy()
    depth_conf = predictions["depth_conf"].squeeze(0).float().cpu().numpy()
    depth = depth.squeeze(-1) if depth.ndim == 4 else depth
    depth_conf = depth_conf.squeeze(-1) if depth_conf.ndim == 4 else depth_conf
    height, width = int(processed_hw[0]), int(processed_hw[1])
    print(f"  predicted {depth.shape} depth at {width}x{height}")

    # Confidence gate first: it is per-view and cheap, and it keeps the
    # pairwise consistency pass working on plausible candidates only.
    if 0.0 < args.conf_threshold < 1.0:
        floor = np.quantile(depth_conf, args.conf_threshold)
        depth = np.where(depth_conf >= floor, depth, np.nan)
        print(f"  confidence gate at rank {args.conf_threshold}: value {floor:.3f}")

    masks = load_plant_masks(args.masks_dir, names, (height, width))
    points = np.stack([unproject(depth[i], intrinsics[i], extrinsics[i]) for i in range(len(paths))])
    xyz, tracks = fuse_by_consistency(points, depth, intrinsics, extrinsics, masks,
                                      args.consistency_views, args.consistency_tolerance,
                                      args.max_points_per_view)
    print(f"  fused {len(xyz)} points agreed by >= {args.consistency_views} other views")
    if len(xyz) == 0:
        raise SystemExit("no point survived multi-view consistency -- the views disagree entirely")

    colours = []
    rgb = (predictions["images"].squeeze(0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    rgb = rgb.transpose(0, 2, 3, 1) if rgb.shape[1] == 3 else rgb
    for track in tracks:
        view, u, v = track[0]
        colours.append(rgb[view, v, u])

    # Intrinsics out of the model's working resolution and into frame pixels,
    # which is the coordinate system every later phase assumes.
    scaled = []
    for index, (frame_width, frame_height) in enumerate(frame_sizes):
        sx, sy = frame_width / width, frame_height / height
        K = intrinsics[index].copy()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        scaled.append(K)
    tracks_scaled = [[(view, u * frame_sizes[view][0] / width, v * frame_sizes[view][1] / height)
                      for view, u, v in track] for track in tracks]

    sparse = scene / "sparse"
    write_colmap_text(sparse, names, frame_sizes, scaled, extrinsics, xyz, tracks_scaled, colours)
    print(f"  wrote {sparse}")
    (scene / "omega_export.json").write_text(json.dumps({
        "checkpoint": checkpoint, "image_resolution": args.image_resolution,
        "mode": args.mode, "processed_hw": [height, width], "frame_size": list(frame_sizes[0]),
        "points": int(len(xyz)), "consistency_views": args.consistency_views,
        "consistency_tolerance": args.consistency_tolerance,
        "mean_track_length": float(np.mean([len(t) for t in tracks])),
    }, indent=2))

    if args.bundle_adjust:
        import pycolmap

        print("  bundle adjustment...")
        reconstruction = pycolmap.Reconstruction(str(sparse))
        pycolmap.bundle_adjustment(reconstruction, pycolmap.BundleAdjustmentOptions())
        reconstruction.write_text(str(sparse))
        print(f"  refined: mean reprojection error "
              f"{reconstruction.compute_mean_reprojection_error():.3f} px")


if __name__ == "__main__":
    sys.exit(main())
