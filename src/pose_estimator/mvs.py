"""P4m -- COLMAP PatchMatch multi-view stereo on the full-resolution originals.

The other route to the heart (see gs_surface for the measurement that asked
for both): photometric depth per view, so every point is a surface some photo
actually saw, and a gap between two stacked leaves stays open wherever any
view saw through it -- the opposite of a visual hull, which fills every gap
no silhouette happened to show.

Steps, all on P3's poses:
  1. a COLMAP model whose cameras are rescaled to the original photos
     (P1's frames are exact 3.125x downscales of them), images linked to the
     originals the manifest names;
  2. COLMAP's undistorter, cropped to the plant (the P4a hull's projection
     over all views, a region of interest shared by every image);
  3. each image's source views picked by viewing direction (3-40 degrees
     away), written as an explicit patch-match.cfg -- the rescaled model
     carries no keypoints, so COLMAP's shared-point selection has nothing to
     go on; and a depth range from the hull, not from the sparse cloud,
     which spans the room;
  4. PatchMatch with geometric consistency (CUDA: run it on the cluster);
  5. fusion restricted to the plant: P2's plant masks, resampled onto the
     undistorted images and dilated by their own edge band, and the hull's
     bounding box;
  6. a floater bound against the hull in P3 pixels, as for P4g.

Writes nothing outside its own folder (p4m/).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pose_estimator.fine_views import MASK_STEP_PX_AT_P2, source_image


def _w2c(image) -> np.ndarray:
    pose = image.cam_from_world()
    m = np.eye(4)
    m[:3, :3] = pose.rotation.matrix()
    m[:3, 3] = np.asarray(pose.translation)
    return m


def prepare(workdir: Path, out: Path, hull: np.ndarray, source_root: Optional[Path] = None,
            max_image_size: int = -1, num_src: int = 10, margin: float = 0.06, log=print) -> dict:
    """Steps 1-3 and the fusion masks of step 5. Returns the settings PatchMatch needs."""
    import pycolmap

    workdir, out = Path(workdir), Path(out)
    src_dir, model_dir, dense = out / "input" / "images", out / "input" / "sparse", out / "dense"
    for d in (src_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    rec = pycolmap.Reconstruction(str(workdir / "p3" / "sparse" / "best"))
    ups = {}
    for image_id in rec.reg_image_ids():
        im = rec.images[image_id]
        path = source_image(workdir, Path(im.name).stem, source_root)
        if not path.exists():
            raise FileNotFoundError(f"{path} -- pass --source-root if the originals live elsewhere")
        link = src_dir / im.name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(path, link)
        if im.camera_id not in ups:
            h, w = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8).shape
            ups[im.camera_id] = (w * 8) / rec.cameras[im.camera_id].width
    originals = {}
    for cid, up in ups.items():
        cam = rec.cameras[cid]
        originals[cid] = (cam.width, cam.height)
        cam.rescale(int(round(cam.width * up)), int(round(cam.height * up)))
    # pycolmap-cuda 3.13.0.dev2 (the cluster's) lacks delete_all_points2D_and_points3D
    if hasattr(rec, "delete_all_points2D_and_points3D"):
        rec.delete_all_points2D_and_points3D()
    else:
        for pid in list(rec.points3D.keys()):
            rec.delete_point3D(pid)
    rec.write(str(model_dir))
    cam0 = rec.cameras[next(iter(ups))]
    log(f"  model rescaled x{next(iter(ups.values())):.3f} to the originals "
        f"({cam0.width}x{cam0.height}); images linked from {src_dir}")

    # region of interest: the hull's projection in every view, plus a margin
    lo, hi = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
    for image_id in rec.reg_image_ids():
        im = rec.images[image_id]
        cam = rec.cameras[im.camera_id]
        m = _w2c(im)
        uv = cam.img_from_cam(hull @ m[:3, :3].T + m[:3, 3])
        rel = uv / [cam.width, cam.height]
        lo, hi = np.minimum(lo, rel.min(0)), np.maximum(hi, rel.max(0))
    pad = margin * (hi - lo).max()
    lo, hi = np.clip(lo - pad, 0, 1), np.clip(hi + pad, 0, 1)
    opts = pycolmap.UndistortCameraOptions()
    opts.roi_min_x, opts.roi_min_y = float(lo[0]), float(lo[1])
    opts.roi_max_x, opts.roi_max_y = float(hi[0]), float(hi[1])
    opts.max_image_size = int(max_image_size)
    if dense.exists():
        shutil.rmtree(dense)
    pycolmap.undistort_images(str(dense), str(model_dir), str(src_dir), undistort_options=opts,
                              num_patch_match_src_images=num_src)
    und = pycolmap.Reconstruction(str(dense / "sparse"))
    ucam = und.cameras[next(iter(und.cameras))]
    log(f"  undistorted and cropped to the plant: {ucam.width}x{ucam.height} px "
        f"(region {lo.round(3).tolist()}..{hi.round(3).tolist()} of the frame)")

    # source views by viewing direction; depth range from the hull
    ims = [und.images[i] for i in sorted(und.reg_image_ids())]
    axes = np.array([_w2c(im)[2, :3] for im in ims])
    lines, dmin, dmax = [], np.inf, 0.0
    for k, im in enumerate(ims):
        angle = np.degrees(np.arccos(np.clip(axes @ axes[k], -1, 1)))
        order = [j for j in np.argsort(angle) if j != k and 3.0 <= angle[j] <= 40.0][:num_src]
        if not order:
            order = [j for j in np.argsort(angle) if j != k][:num_src]
        lines += [im.name, ", ".join(ims[j].name for j in order)]
        z = (hull @ _w2c(im)[:3, :3].T + _w2c(im)[:3, 3])[:, 2]
        dmin, dmax = min(dmin, float(z.min())), max(dmax, float(z.max()))
    (dense / "stereo" / "patch-match.cfg").write_text("\n".join(lines) + "\n")
    span = dmax - dmin
    depth_min, depth_max = dmin - 0.1 * span, dmax + 0.1 * span

    # fusion masks: P2's plant mask on each undistorted image, dilated by its edge band
    masks = dense / "masks"
    masks.mkdir(exist_ok=True)
    band = None
    for im in ims:
        orig = rec.cameras[rec.find_image_with_name(im.name).camera_id]
        u = und.cameras[im.camera_id]
        m = cv2.imread(str(workdir / "p2" / "masks" / "plant" / f"{Path(im.name).stem}.png"),
                       cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        xs, ys = np.meshgrid(np.arange(u.width, dtype=np.float64), np.arange(u.height, dtype=np.float64))
        norm = u.cam_from_img(np.stack([xs.ravel(), ys.ravel()], 1))
        src = orig.img_from_cam(np.c_[norm, np.ones(len(norm))]) * (m.shape[1] / orig.width)
        src = src.reshape(ys.shape + (2,)).astype(np.float32)
        warped = cv2.remap(m, src[..., 0], src[..., 1], cv2.INTER_LINEAR) > 127
        band = int(np.ceil(MASK_STEP_PX_AT_P2 * u.width / ((hi[0] - lo[0]) * m.shape[1])))
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
        cv2.imwrite(str(masks / f"{im.name}.png"), cv2.dilate(warped.astype(np.uint8) * 255, disc))
    log(f"  {len(ims)} images, {num_src} source views each (3-40 deg away); depth range "
        f"{depth_min:.3f}..{depth_max:.3f}; fusion masks dilated {band} px")
    lo3, hi3 = hull.min(0), hull.max(0)
    pad3 = 0.05 * (hi3 - lo3)
    info = {"depth_min": depth_min, "depth_max": depth_max, "image_size": [ucam.width, ucam.height],
            "roi": [lo.tolist(), hi.tolist()], "num_src": num_src, "mask_band_px": band,
            "bbox": [(lo3 - pad3).tolist(), (hi3 + pad3).tolist()]}
    (out / "prepare.json").write_text(json.dumps(info, indent=1))
    return info


def patch_match(out: Path, info: dict, window_radius: int = 5, iterations: int = 5,
                gpu_index: str = "-1", log=print) -> None:
    """Step 4 (CUDA)."""
    import pycolmap

    if not getattr(pycolmap, "has_cuda", False):
        raise SystemExit("PatchMatch needs pycolmap built with CUDA (pycolmap-cuda); this one is not. "
                         "Run P4m on the cluster, or with --stop-after prepare here.")
    opts = pycolmap.PatchMatchOptions()
    opts.depth_min, opts.depth_max = float(info["depth_min"]), float(info["depth_max"])
    opts.window_radius, opts.num_iterations = int(window_radius), int(iterations)
    opts.geom_consistency, opts.filter = True, True
    opts.gpu_index = str(gpu_index)
    log(f"  PatchMatch: window {2 * window_radius + 1} px, {iterations} iterations, geometric "
        f"consistency on, GPU {gpu_index}")
    pycolmap.patch_match_stereo(str(out / "dense"), options=opts)


def _pca_normals(points: np.ndarray, k: int = 12) -> np.ndarray:
    from scipy.spatial import cKDTree

    if len(points) < 3:
        return np.zeros_like(points)
    _, nb = cKDTree(points).query(points, k=min(k, len(points)))
    c = points[nb] - points[nb].mean(axis=1, keepdims=True)
    return np.linalg.eigh(np.einsum("nki,nkj->nij", c, c))[1][:, :, 0]


def fuse(out: Path, info: dict, hull: np.ndarray, p3_focal: float, min_num_pixels: int = 3,
         hull_slack_px: float = 6.0, log=print):
    """Steps 5-6: fused points restricted to the plant. Returns (points, normals, colours, stats)."""
    import pycolmap
    from scipy.spatial import cKDTree

    from pose_estimator.ply_io import read_ply_vertices

    opts = pycolmap.StereoFusionOptions()
    opts.mask_path = str(out / "dense" / "masks")
    opts.min_num_pixels = int(min_num_pixels)
    opts.bounding_box = (np.asarray(info["bbox"][0], np.float32), np.asarray(info["bbox"][1], np.float32))
    fused = out / "fused.ply"
    if fused.exists():
        fused.unlink()
    try:                                  # pycolmap 4.x writes the PLY itself
        result = pycolmap.stereo_fusion(str(fused), str(out / "dense"), input_type="geometric",
                                        options=opts, output_type="PLY")
    except TypeError:                     # 3.13 has no output_type and returns a model
        result = pycolmap.stereo_fusion(str(fused), str(out / "dense"), input_type="geometric",
                                        options=opts)
    if fused.exists() and fused.is_file():
        f = read_ply_vertices(fused)
        pts = np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float64)
        nrm = np.stack([f["nx"], f["ny"], f["nz"]], 1).astype(np.float64)
        col = np.stack([f["red"], f["green"], f["blue"]], 1).astype(np.uint8)
    else:
        # the returned model carries points and colours but no normals:
        # local PCA, flipped to face the nearest camera below
        ps = list(result.points3D.values())
        pts = np.array([q.xyz for q in ps], np.float64).reshape(-1, 3)
        col = np.array([q.color for q in ps], np.uint8).reshape(-1, 3)
        nrm = _pca_normals(pts)
    und = pycolmap.Reconstruction(str(out / "dense" / "sparse"))
    centres = np.array([-_w2c(im)[:3, :3].T @ _w2c(im)[:3, 3] for im in und.images.values()])
    # the plant is small next to the camera distance: one depth serves the bound
    depth = float(np.median(np.linalg.norm(centres - hull.mean(0), axis=1)))
    if len(pts):
        # orient normals towards the nearest camera (PCA ones have no sign)
        near = centres[cKDTree(centres).query(pts)[1]]
        nrm *= np.where(np.einsum("ij,ij->i", nrm, near - pts) < 0, -1.0, 1.0)[:, None]
    d_hull, _ = cKDTree(hull).query(pts)
    keep = d_hull <= hull_slack_px * depth / p3_focal
    stats = {"fused": int(len(pts)), "inside_hull_bound": int(keep.sum())}
    log(f"  fused {len(pts):,} points, {int(keep.sum()):,} inside the hull bound "
        f"({hull_slack_px:g} P3 px at their depth)")
    return pts[keep], nrm[keep], col[keep], stats
