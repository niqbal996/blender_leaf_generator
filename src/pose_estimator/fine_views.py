"""Full-resolution, undistorted, plant-cropped views -- the input of the fine-cloud phases.

P3 solved on P1's 1920 px frames, and P4b trained at half of that (960 px):
about 0.46 mm per pixel on the boom rig, so a 2-4 mm heart leaf was 4-9
pixels wide before any geometry was fitted to it. The originals the frames
were made from are 3.125x larger (6000x4000 on the D5600) and are still on
disk -- P1's manifest names each one -- so the same COLMAP poses serve them
with the camera rescaled.

Each view here is:
  - the original photo, resampled once onto an exact pinhole camera (the
    COLMAP model's radial distortion removed: P4b projected pinhole-only
    onto distorted images, a few px off at the frame edges at 960 px and
    ~20 px at full resolution),
  - cropped to the plant (the P4a hull's projection plus a margin), which is
    a few percent of a 24 MP frame,
  - with the P2 plant mask and holder mask resampled onto the same pixels.

The masks stay coarse: SAM3's masks come from a 288 px head, ~4 px steps at
1920, ~13 px at full resolution. `mask_band_px` says how wide that
uncertainty is at the view's own resolution, so a consumer can trust the
mask's inside and outside and leave the edge to the photograph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# SAM3's leaf/plant masks are upsampled from a 288 px head: on the 1920 px P2
# frames that is ~4.4 px quantisation (leaf_tips_2d.INLIER_PX's note).
MASK_STEP_PX_AT_P2 = 4.4


@dataclass
class FineView:
    name: str                      # frame stem, e.g. frame_0007
    image: np.ndarray              # (H, W, 3) uint8 RGB, undistorted, cropped
    mask: np.ndarray               # (H, W) uint8 0..255, P2 plant mask on the same pixels
    occluder: Optional[np.ndarray] # (H, W) uint8 0..255, P2 holder mask, or None
    K: np.ndarray                  # (3, 3) pinhole intrinsics of the crop
    world_to_camera: np.ndarray    # (4, 4)
    crop: tuple                    # (x0, y0) of the crop in the undistorted full image
    mask_band_px: float            # SAM3 mask edge uncertainty at this resolution


def source_image(workdir: Path, frame: str, source_root: Optional[Path] = None) -> Path:
    """The original photo P1 made `frame` from.

    The manifest records the cluster path. `source_root` re-roots it on
    another machine: <source_root>/<pass folder>/<file>, which is how a
    local mirror lays the session out (vogelmeere_x_1/pass1/DSC_0671.JPG).
    """
    manifest = {m["frame"]: m for m in json.loads((workdir / "p1" / "manifest.json").read_text())}
    entry = manifest[frame]
    path = Path(entry["source_dir"]) / entry["source_file"]
    if source_root is not None:
        path = Path(source_root) / Path(entry["source_dir"]).name / entry["source_file"]
    return path


def _pose_matrix(image) -> np.ndarray:
    pose = image.cam_from_world()
    m = np.eye(4)
    m[:3, :3] = pose.rotation.matrix()
    m[:3, 3] = np.asarray(pose.translation)
    return m


def load_fine_views(workdir: Path, hull_points: np.ndarray, scale: float = 1.0,
                    source_root: Optional[Path] = None, margin: float = 0.08,
                    frames: Optional[List[str]] = None, verbose: bool = True) -> List[FineView]:
    """Every registered P3 view at `scale` x the original photo's resolution.

    `scale` 1.0 is the original (6000 px wide on the D5600); 0.32 is about
    P1's 1920 px. `frames` restricts to those frame stems (smoke tests).
    """
    import pycolmap

    workdir = Path(workdir)
    rec = pycolmap.Reconstruction(str(workdir / "p3" / "sparse" / "best"))
    views = []
    for image_id in sorted(rec.reg_image_ids()):
        im = rec.images[image_id]
        name = Path(im.name).stem
        if frames is not None and name not in frames:
            continue
        path = source_image(workdir, name, source_root)
        photo = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if photo is None:
            raise FileNotFoundError(f"{path} (original of {name}) -- pass --source-root if the "
                                    "photos live somewhere else on this machine")
        full_h, full_w = photo.shape[:2]
        cam = rec.cameras[im.camera_id]
        up = full_w / cam.width                       # P3 camera -> original photo
        if abs(full_h / cam.height - up) > 1e-3:
            raise ValueError(f"{path}: {full_w}x{full_h} is not a uniform scale of the "
                             f"{cam.width}x{cam.height} P3 camera")
        # the original camera, distortion and all, at the photo's resolution
        params = np.asarray(cam.params, float).copy()
        for i in list(cam.focal_length_idxs()) + list(cam.principal_point_idxs()):
            params[i] *= up
        orig = pycolmap.Camera(model=cam.model, width=full_w, height=full_h, params=params)

        # the pinhole camera we resample onto, at `scale`
        f = float(np.mean([params[i] for i in cam.focal_length_idxs()])) * scale
        cx, cy = (np.asarray([params[i] for i in cam.principal_point_idxs()]) * scale)
        W, H = int(round(full_w * scale)), int(round(full_h * scale))
        w2c = _pose_matrix(im)

        # crop: the hull's projection plus a margin
        c = hull_points @ w2c[:3, :3].T + w2c[:3, 3]
        uv = c[:, :2] / c[:, 2:3] * f + [cx, cy]
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        pad = margin * (hi - lo).max() + 16
        x0, y0 = (np.floor(np.maximum(lo - pad, 0))).astype(int)
        x1, y1 = (np.ceil(np.minimum(hi + pad, [W, H]))).astype(int)
        K = np.array([[f, 0, cx - x0], [0, f, cy - y0], [0, 0, 1.0]])

        # where each crop pixel comes from in the original (distorted) photo
        xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64))
        norm = np.stack([(xs - cx) / f, (ys - cy) / f, np.ones_like(xs)], axis=-1).reshape(-1, 3)
        src = orig.img_from_cam(norm).reshape(ys.shape + (2,)).astype(np.float32)
        image = cv2.remap(photo, src[..., 0], src[..., 1], cv2.INTER_AREA if scale < 0.5
                          else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        def resample_mask(folder: str) -> Optional[np.ndarray]:
            m = cv2.imread(str(workdir / "p2" / "masks" / folder / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return None
            k = m.shape[1] / full_w                  # original photo -> P2 mask pixels
            return cv2.remap(m, src[..., 0] * k, src[..., 1] * k, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)

        mask = resample_mask("plant")
        if mask is None:
            continue
        holder = resample_mask("holder")
        p2_width = cv2.imread(str(workdir / "p2" / "masks" / "plant" / f"{name}.png"),
                              cv2.IMREAD_GRAYSCALE).shape[1]
        views.append(FineView(name=name, image=image, mask=mask,
                              occluder=holder if holder is not None and holder.any() else None,
                              K=K, world_to_camera=w2c, crop=(int(x0), int(y0)),
                              mask_band_px=MASK_STEP_PX_AT_P2 * W / p2_width))
        if verbose and len(views) == 1:
            print(f"  {name}: original {full_w}x{full_h} from {path}, x{scale:g} -> crop "
                  f"{x1 - x0}x{y1 - y0}, f {f:.0f} px, mask edge band {views[-1].mask_band_px:.1f} px")
    if verbose:
        sizes = np.array([v.image.shape[:2] for v in views])
        print(f"  {len(views)} views, crops {int(np.median(sizes[:, 1]))}x{int(np.median(sizes[:, 0]))} "
              f"px median, {sum(v.occluder is not None for v in views)} with holder masks")
    return views
