#!/usr/bin/env python
"""Depth Anything V2 -> surface normals, for the SAM organ lab bench.

    python scripts/depth_to_normals.py --images runs/plant_9/p1/frames \
        --frames 57,76 --out runs/plant_9/normals_da

Writes per frame:
    <stem>.npy        HxWx3 float32 unit normals in camera space
    <stem>_vis.jpg    depth | normals, for eyeballing
    run.log

Why normals rather than depth. Leaf-versus-stem cannot be decided from a 2D
outline: a grass blade seen flat-on is a ribbon and a leaf seen edge-on is a
ribbon, so any elongation threshold is measuring the viewpoint, not the organ.
Surface orientation does not have that problem -- a lamina is a flat sheet
from every direction and a stem is a tube from every direction.

On the scale/shift ambiguity. Monocular depth is only correct up to
`z = a*z_pred + b`, so these normals are *not* metrically right, and nothing
here should be used as geometry. But the consumer is a *flatness* measure --
does orientation agree across a mask -- and an unknown global `a` applies
uniformly, so a flat region still reads as flat. That is why this is usable
for classification while being unusable for reconstruction.

Intrinsics matter and are used when available: normals come from back-projected
3D points, not from raw image-space depth gradients, so the result does not
warp toward the frame edges.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def load_frames(images: str, frames: Optional[str], stride: int) -> List[Tuple[str, np.ndarray]]:
    paths = sorted(Path(images).glob("*.jpg")) + sorted(Path(images).glob("*.png"))
    if frames:
        picks = [int(x) for x in frames.split(",")]
        paths = [paths[i] for i in picks if i < len(paths)]
    else:
        paths = paths[::stride]
    return [(p.stem, cv2.imread(str(p))) for p in paths]


def colmap_intrinsics(sparse_dir: Path) -> Optional[Tuple[float, float, float, float]]:
    """(fx, fy, cx, cy) from a solved reconstruction, if one is available."""
    try:
        import pycolmap
    except ImportError:
        return None
    if not sparse_dir.is_dir():
        return None
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    camera = list(reconstruction.cameras.values())[0]
    K = np.asarray(camera.calibration_matrix())
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def normals_from_depth(
    depth: np.ndarray, intrinsics: Optional[Tuple[float, float, float, float]], smooth: int = 5
) -> np.ndarray:
    """Unit surface normals from a depth map, via back-projected 3D points.

    Cross product of the local surface tangents. Depth is smoothed first
    because normals are a derivative, and differentiating a noisy depth map
    produces normals that fan out even on a genuinely flat leaf -- which would
    defeat the whole point of measuring flatness.
    """
    height, width = depth.shape
    if intrinsics is None:
        # Fall back to a plausible focal length; only the ratio to image size
        # matters for the shape of the resulting normals.
        fx = fy = float(max(height, width))
        cx, cy = width / 2.0, height / 2.0
    else:
        fx, fy, cx, cy = intrinsics

    if smooth > 1:
        depth = cv2.bilateralFilter(depth.astype(np.float32), smooth, 0.1, smooth)

    us, vs = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (us - cx) * depth / fx
    y = (vs - cy) * depth / fy
    points = np.stack([x, y, depth], axis=2)

    # Central differences along each image axis give two surface tangents.
    du = np.gradient(points, axis=1)
    dv = np.gradient(points, axis=0)
    normals = np.cross(du, dv)

    lengths = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.divide(normals, np.maximum(lengths, 1e-9))
    # Face the camera, so a sheet seen from either side reads consistently.
    flip = normals[:, :, 2] > 0
    normals[flip] *= -1.0
    return normals.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--frames", help="comma-separated indices, e.g. 57,76")
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Large-hf",
                   help="HF model id; -Small-hf / -Base-hf are lighter")
    p.add_argument("--sparse", default="", help="COLMAP sparse dir, for real intrinsics")
    p.add_argument("--smooth", type=int, default=5, help="bilateral filter size on depth")
    p.add_argument("--plant-mask-dir",
                   help="crop to the plant before inference. Strongly recommended: the plant is "
                        "~2.5%% of these frames and the model works at ~518px internally, so "
                        "uncropped it predicts the plant as background and the normals are noise")
    p.add_argument("--pad", type=int, default=60)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "run.log")
    print(json.dumps(vars(args), indent=2))

    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    print(f"\nloading {args.model} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(args.device).eval()

    intrinsics = colmap_intrinsics(Path(args.sparse)) if args.sparse else None
    print(f"intrinsics: {intrinsics if intrinsics else 'none -- using a fallback focal length'}")

    frames = load_frames(args.images, args.frames, args.stride)
    print(f"{len(frames)} frames\n")

    for stem, bgr in frames:
        if bgr is None:
            continue
        # Crop to the subject so the model spends its resolution on the plant.
        y0, y1, x0, x1 = 0, bgr.shape[0], 0, bgr.shape[1]
        if args.plant_mask_dir:
            m = cv2.imread(str(Path(args.plant_mask_dir) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None and (m > 127).any():
                ys, xs = np.nonzero(m > 127)
                y0 = max(0, ys.min() - args.pad); y1 = min(bgr.shape[0], ys.max() + args.pad)
                x0 = max(0, xs.min() - args.pad); x1 = min(bgr.shape[1], xs.max() + args.pad)
        view = bgr[y0:y1, x0:x1]

        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt").to(args.device)
        with torch.no_grad():
            predicted = model(**inputs).predicted_depth

        predicted = torch.nn.functional.interpolate(
            predicted.unsqueeze(1), size=view.shape[:2], mode="bicubic", align_corners=False
        )[0, 0].cpu().numpy()

        # Depth Anything emits inverse depth (near = large). Invert it, or a
        # flat surface comes out curved and every flatness measure is wrong.
        disparity = predicted - predicted.min()
        disparity = disparity / max(disparity.max(), 1e-9)
        depth = 1.0 / (disparity * 0.9 + 0.1)
        depth = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-9) + 0.5

        # The principal point moves with the crop; the focal length does not.
        shifted = None
        if intrinsics is not None:
            fx, fy, cx, cy = intrinsics
            shifted = (fx, fy, cx - x0, cy - y0)
        crop_normals = normals_from_depth(depth, shifted, smooth=args.smooth)

        # Paste back so the .npy is aligned to the full frame, which is what
        # the lab bench indexes into.
        normals = np.zeros((bgr.shape[0], bgr.shape[1], 3), np.float32)
        normals[y0:y1, x0:x1] = crop_normals
        np.save(out_dir / f"{stem}.npy", normals)

        depth_vis = cv2.applyColorMap(
            (255 * (1.0 - (depth - depth.min()) / max(depth.max() - depth.min(), 1e-9))
             ).astype(np.uint8), cv2.COLORMAP_TURBO)
        normal_vis = ((crop_normals * 0.5 + 0.5) * 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(str(out_dir / f"{stem}_vis.jpg"), np.hstack([view, depth_vis, normal_vis]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

        print(f"{stem}: crop {x1 - x0}x{y1 - y0}  depth {depth.min():.2f}..{depth.max():.2f}  "
              f"-> {stem}.npy")

    print(f"\nwrote {out_dir}/*.npy and *_vis.jpg")


if __name__ == "__main__":
    main()
