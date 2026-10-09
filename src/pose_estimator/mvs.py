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


def read_colmap_array(path: Path) -> np.ndarray:
    """A COLMAP depth/normal map (.bin): 'w&h&c&' then float32, channel-major."""
    with open(path, "rb") as f:
        head = b""
        while head.count(b"&") < 3:
            head += f.read(1)
        w, h, c = (int(x) for x in head.split(b"&")[:3])
        data = np.fromfile(f, np.float32, w * h * c)
    return data.reshape(c, h, w).transpose(1, 2, 0)


def depth_coverage(out: Path, n: int = 4, log=print) -> dict:
    """{map type: median share of plant-mask pixels with a depth} over a few images.

    What fusion has to work with: if the geometric pass filtered (almost)
    everything, fusion makes nothing, which is what an empty P4m looks like.
    """
    stereo = out / "dense" / "stereo"
    names = [l.strip() for l in (stereo / "fusion.cfg").read_text().splitlines() if l.strip()]
    pick = [names[int(i)] for i in np.linspace(0, len(names) - 1, n).round()]
    report = {}
    for kind in ("photometric", "geometric"):
        have = sorted((stereo / "depth_maps").glob(f"*.{kind}.bin"))
        shares, ranges = [], []
        for name in pick:
            path = stereo / "depth_maps" / f"{name}.{kind}.bin"
            mask = cv2.imread(str(out / "dense" / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            if not path.exists() or mask is None:
                continue
            d = read_colmap_array(path)[..., 0]
            plant = mask > 0
            if d.shape != plant.shape:
                log(f"  {name}.{kind}: depth map {d.shape} but mask {plant.shape}")
                continue
            valid = plant & (d > 0)
            shares.append(float(valid.sum() / max(plant.sum(), 1)))
            if valid.any():
                ranges.append((float(np.percentile(d[valid], 1)), float(np.percentile(d[valid], 99))))
        report[kind] = {"maps": len(have), "plant_covered": float(np.median(shares)) if shares else 0.0,
                        "depth_p1_p99": ranges[:1]}
        log(f"  {kind:11s} depth maps: {len(have)}; plant pixels with a depth (median over "
            f"{len(shares)} images): {report[kind]['plant_covered']:.0%}"
            + (f"; depths {ranges[0][0]:.3f}..{ranges[0][1]:.3f}" if ranges else ""))
    return report


def _pca_normals(points: np.ndarray, k: int = 12) -> np.ndarray:
    from scipy.spatial import cKDTree

    if len(points) < 3:
        return np.zeros_like(points)
    _, nb = cKDTree(points).query(points, k=min(k, len(points)))
    c = points[nb] - points[nb].mean(axis=1, keepdims=True)
    return np.linalg.eigh(np.einsum("nki,nkj->nij", c, c))[1][:, :, 0]


def _read_cfg_lists(path: Path) -> dict:
    """patch-match.cfg: {image: [source images]} (pairs of lines)."""
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    return {lines[i]: [x.strip() for x in lines[i + 1].split(",") if x.strip()]
            for i in range(0, len(lines) - 1, 2)}


def own_fuse(out: Path, input_type: str = "geometric", min_consistent: int = 2, neighbours: int = 8,
             consistency_px: float = 2.0, stride: int = 2, log=print):
    """Depth maps fused without COLMAP's fuser: back-projected, kept where neighbours agree.

    Each image's depth map, inside its plant mask, is back-projected; a point
    is kept when at least `min_consistent` of its source views (from
    patch-match.cfg) have a depth within `consistency_px` of it, in their own
    pixels at its depth -- the test P4g uses on rendered depth. Normals
    come from COLMAP's normal maps, colour from the undistorted photo.

    Exists because pycolmap-cuda 3.13.0.dev2's fuser returned 0 points per
    image on vogelmeere's maps, which covered 35% of the plant (geometric) at
    the plant's depths, with settings the released 3.13.0 fuses with.
    Returns (points, normals, colours uint8, voxel) -- not yet hull-bounded
    or consolidated.
    """
    import pycolmap

    dense = out / "dense"
    und = pycolmap.Reconstruction(str(dense / "sparse"))
    by_name = {im.name: im for im in und.images.values()}
    names = [l.strip() for l in (dense / "stereo" / "fusion.cfg").read_text().splitlines() if l.strip()]
    sources = _read_cfg_lists(dense / "stereo" / "patch-match.cfg")
    cams, depth, masks = {}, {}, {}
    for n in names:
        im = by_name[n]
        cam = und.cameras[im.camera_id]
        K = np.asarray(cam.calibration_matrix(), float)
        cams[n] = (K, _w2c(im))
        path = dense / "stereo" / "depth_maps" / f"{n}.{input_type}.bin"
        if not path.exists():
            continue
        depth[n] = read_colmap_array(path)[..., 0]
        m = cv2.imread(str(dense / "masks" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        masks[n] = (m > 0) if m is not None else np.ones(depth[n].shape, bool)
    log(f"  own fusion: {len(depth)} {input_type} depth maps, {neighbours} source views each, "
        f"agreement within {consistency_px:g} px, at least {min_consistent} views")
    P, N, C = [], [], []
    raw = kept = 0
    for i, n in enumerate(names):
        if n not in depth:
            continue
        K, m = cams[n]
        d = depth[n]
        valid = (d > 0) & masks[n]
        sub = np.zeros_like(valid)
        sub[::stride, ::stride] = True
        ys, xs = np.nonzero(valid & sub)
        if not len(ys):
            continue
        z = d[ys, xs].astype(np.float64)
        cam = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], 1)
        X = (cam - m[:3, 3]) @ m[:3, :3]
        raw += len(X)
        agree = np.zeros(len(X), np.int32)
        for src in [s_ for s_ in sources.get(n, []) if s_ in depth][:neighbours]:
            Kj, mj = cams[src]
            dj = depth[src]
            cj = X @ mj[:3, :3].T + mj[:3, 3]
            u = cj[:, 0] / cj[:, 2] * Kj[0, 0] + Kj[0, 2]
            v = cj[:, 1] / cj[:, 2] * Kj[1, 1] + Kj[1, 2]
            xi, yi = np.round(u).astype(int), np.round(v).astype(int)
            ok = (xi >= 0) & (xi < dj.shape[1]) & (yi >= 0) & (yi < dj.shape[0]) & (cj[:, 2] > 0)
            seen = np.zeros(len(X))
            seen[ok] = dj[yi[ok], xi[ok]]
            tol = consistency_px * cj[:, 2] / Kj[0, 0]
            agree += ((seen > 0) & (np.abs(seen - cj[:, 2]) <= tol)).astype(np.int32)
        keep = agree >= min_consistent
        kept += int(keep.sum())
        normal_path = dense / "stereo" / "normal_maps" / f"{n}.{input_type}.bin"
        if normal_path.exists():
            nc = read_colmap_array(normal_path)[ys[keep], xs[keep]].astype(np.float64)
            N.append(nc @ m[:3, :3])                      # camera -> world
        else:
            N.append(np.zeros((int(keep.sum()), 3)))
        photo = cv2.imread(str(dense / "images" / n), cv2.IMREAD_COLOR)
        C.append(photo[ys[keep], xs[keep]][:, ::-1] if photo is not None
                 else np.full((int(keep.sum()), 3), 128, np.uint8))
        P.append(X[keep])
        if (i + 1) % 10 == 0 or i == len(names) - 1:
            log(f"  own fusion {i + 1}/{len(names)}: {raw:,} depths back-projected, {kept:,} agreed")
    f_med = float(np.median([cams[n][0][0, 0] for n in depth]))
    z_med = float(np.median([np.median(depth[n][depth[n] > 0]) for n in depth if (depth[n] > 0).any()]))
    voxel = stride * z_med / f_med
    cat = lambda L, w, t: np.vstack(L).astype(t) if L else np.zeros((0, w), t)
    return cat(P, 3, np.float64), cat(N, 3, np.float64), cat(C, 3, np.uint8), voxel


def _colmap_fusion(pycolmap, fused: Path, out: Path, input_type: str, opts, read_ply_vertices):
    """COLMAP's own fuser: (points, normals, colours), on pycolmap 4.x or 3.13."""
    try:                                  # pycolmap 4.x writes the PLY itself
        result = pycolmap.stereo_fusion(str(fused), str(out / "dense"), input_type=input_type,
                                        options=opts, output_type="PLY")
    except TypeError:                     # 3.13 has no output_type and returns a model
        result = pycolmap.stereo_fusion(str(fused), str(out / "dense"), input_type=input_type,
                                        options=opts)
    if fused.exists() and fused.is_file():
        f = read_ply_vertices(fused)
        return (np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float64),
                np.stack([f["nx"], f["ny"], f["nz"]], 1).astype(np.float64),
                np.stack([f["red"], f["green"], f["blue"]], 1).astype(np.uint8))
    # the returned model carries points and colours but no normals: local
    # PCA, flipped to face the nearest camera by the caller
    ps = list(result.points3D.values())
    pts = np.array([q.xyz for q in ps], np.float64).reshape(-1, 3)
    return pts, _pca_normals(pts), np.array([q.color for q in ps], np.uint8).reshape(-1, 3)


def fuse(out: Path, info: dict, hull: np.ndarray, p3_focal: float, min_num_pixels: int = 3,
         hull_slack_px: float = 6.0, plain: bool = False, method: str = "auto", log=print):
    """Steps 5-6: fused points restricted to the plant. Returns (points, normals, colours, stats)."""
    import pycolmap
    from scipy.spatial import cKDTree

    from pose_estimator.ply_io import read_ply_vertices

    coverage = depth_coverage(out, log=log)
    input_type = "geometric"
    if coverage["geometric"]["plant_covered"] < 0.02 <= coverage["photometric"]["plant_covered"]:
        input_type = "photometric"
        log("  the geometric maps are (almost) empty -- fusing the photometric ones instead")
    opts = pycolmap.StereoFusionOptions()
    opts.min_num_pixels = int(min_num_pixels)
    if plain:
        # no masks, no box: the hull bound below still removes the background
        log("  plain fusion: no plant masks, no bounding box")
    else:
        opts.mask_path = str(out / "dense" / "masks")
        opts.bounding_box = (np.asarray(info["bbox"][0], np.float32), np.asarray(info["bbox"][1], np.float32))
    fused = out / "fused.ply"
    if fused.exists():
        fused.unlink()
    if method == "own":
        pts = nrm = np.zeros((0, 3))
        col = np.zeros((0, 3), np.uint8)
    else:
        pts, nrm, col = _colmap_fusion(pycolmap, fused, out, input_type, opts, read_ply_vertices)
    voxel = None
    if method == "own" or (method == "auto" and not len(pts)):
        if method == "auto":
            log("  COLMAP's fusion kept nothing -- fusing the depth maps directly instead")
        from pose_estimator.gs_surface import consolidate

        pts, nrm, col, voxel = own_fuse(out, input_type=input_type, min_consistent=max(1, min_num_pixels - 1),
                                        log=log)
        pts, nrm, col = consolidate(pts, nrm, col, voxel)
        log(f"  merged per {voxel:.6f}-unit voxel (2 px at the plant's depth): {len(pts):,} points")
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
    stats = {"fused": int(len(pts)), "inside_hull_bound": int(keep.sum()), "input_type": input_type,
             "fusion": "own" if voxel is not None else "colmap", "coverage": coverage}
    if not len(pts):
        log("  WARNING: fusion produced no points. The coverage lines above say whether the depth "
            "maps hold any plant; send them with p4m/run.log")
    log(f"  fused {len(pts):,} points, {int(keep.sum()):,} inside the hull bound "
        f"({hull_slack_px:g} P3 px at their depth)")
    return pts[keep], nrm[keep], col[keep], stats
