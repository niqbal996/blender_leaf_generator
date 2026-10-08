"""P4g -- a 2D Gaussian splatting surface at full resolution, built to resolve the heart.

Measured on vogelmeere_x_1 (2026-10-08): where leaves crowd, P4b's surface is
a hollow shell round the cluster. Only 57% of the midribs of leaves with a
neighbour within 8 mm lie on it, against 91% for isolated leaves, and 13% are
buried inside the hull with no surface within 1 mm. P4b cannot do better by
construction: its surfels start on the hull's *outer* voxels, are never
added to (no densification), train 4000 steps on 960 px images (~0.46 mm per
pixel), and are carved back to the hull. It can polish the skin and nothing
else.

This phase changes exactly those things and keeps the rest of P4b's ideas
(2D Gaussians, silhouette supervision, geometry from rendered median depth):

  - **Resolution.** Full-resolution originals, undistorted onto an exact
    pinhole camera and cropped to the plant (`fine_views`): ~0.07 mm per
    pixel at scale 1.
  - **Seeds inside the hull, not only on it.** Every hull voxel can seed a
    Gaussian, so a leaf hidden inside a cluster has primitives near it from
    the start; unsupported ones fade and are pruned.
  - **Densification and pruning** (gsplat's DefaultStrategy on the 2DGS
    gradient), so detail is added where the photographs ask for it.
  - **The mask's edge is left to the photograph.** SAM3 masks step ~13 px at
    full resolution, so the silhouette term only trusts the mask's inside
    (alpha 1) and outside (alpha 0); the band between, and anything behind
    the holder, is judged by colour alone. The backdrop is black, so the
    photograph itself carries a sharp silhouette.
  - 2DGS's own geometry terms: depth distortion and normal consistency.
  - **The hull is a floater bound, not a carve**: a point is dropped only
    when it lies further outside the hull than `hull_slack_px` P3 pixels at
    its depth. Points are kept where `min_consistent` other views' rendered
    depth agrees with them (multi-view consistency), which is what removes
    floaters inside the bound.

Writes nothing outside its own folder (p4g/); P4a and P4b stay as baselines.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional

import cv2
import numpy as np

from pose_estimator.fine_views import FineView

SH_C0 = 0.28209479177387814


def _logit(x: float) -> float:
    return float(np.log(x / (1.0 - x)))


def loss_masks(view: FineView):
    """(sure, maybe, evidence) uint8 maps for one view.

    sure      plant for certain: the mask eroded by its edge band, minus
              pixels the photograph shows to be backdrop
    maybe     could be plant: the mask dilated by its edge band, minus the
              same backdrop pixels
    evidence  1 where the pixel says anything: not behind the holder

    SAM3's mask fills small gaps between stems and leaves (on vogelmeere_x_1's
    heart, 1.5-4.5% of mask pixels are backdrop in the full-resolution
    photo), and a silhouette term that trusts it builds surface across them.
    The backdrop is black and plain, so a pixel inside the mask whose
    colour lies within the backdrop's own spread -- sampled outside the
    dilated mask in the same crop -- is taken as backdrop.
    """
    band = max(1, int(math.ceil(view.mask_band_px)))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    plant = (view.mask > 127).astype(np.uint8)
    sure = cv2.erode(plant, disc)
    maybe = cv2.dilate(plant, disc)
    evidence = np.ones_like(plant)
    if view.occluder is not None:
        evidence = ((view.occluder <= 127) | (plant > 0)).astype(np.uint8)
    backdrop = backdrop_pixels(view.image, maybe, evidence)
    sure &= ~backdrop
    maybe &= ~backdrop
    return sure, maybe, evidence


def backdrop_pixels(image: np.ndarray, maybe: np.ndarray, evidence: np.ndarray,
                    max_share: float = 0.08) -> np.ndarray:
    """uint8 1 where the photo looks like the backdrop, judged against the backdrop itself.

    Value and saturation of pixels well outside the mask set the backdrop's
    range -- median + 3 MAD, robust to the AprilTags and the plier handle
    that share the backdrop (their 99th percentile reached the plant's
    colours on vogelmeere_x_1 frame 49 and took 79% of the mask). A pixel
    within both is backdrop: on that plant the backdrop sits at saturation
    ~20-30 and plant tissue above 64 even in its darkest 1%. Small specks
    are dropped, and a view where the test would take more than `max_share`
    of the mask is not trusted at all.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    outside = (cv2.dilate(maybe, np.ones((15, 15), np.uint8)) == 0) & (evidence > 0)
    if outside.sum() < 500:
        return np.zeros_like(maybe)
    limit = []
    for ch in (2, 1):                                     # value, saturation
        x = hsv[..., ch][outside].astype(np.float64)
        med = np.median(x)
        limit.append(med + max(3.0 * 1.4826 * np.median(np.abs(x - med)), 10.0))
    dark = ((hsv[..., 2] <= limit[0]) & (hsv[..., 1] <= limit[1])).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 9
    found = keep[lab]
    if (found & maybe).sum() > max_share * max(maybe.sum(), 1):
        return np.zeros_like(maybe)
    return found


def init_from_points(points: np.ndarray, normals: np.ndarray, colours: np.ndarray, count: int,
                     sh_degree: int, device: str = "cuda"):
    """Gaussians seeded on another cloud's points -- P4m's photo-consistent surface.

    A hull's skin sits in front of the real surface wherever the plant is
    concave, and Gaussians started there can explain the photos well enough
    that nothing pulls them inward (the smoke run's cross-sections at the
    heart kept P4b's shell). MVS points are surfaces the photographs agreed
    on, so the discs start on them, oriented by their normals.
    """
    import torch
    from scipy.spatial import cKDTree

    from pose_estimator.surfels import _quats_from_normals

    rng = np.random.default_rng(0)
    pick = rng.choice(len(points), min(count, len(points)), replace=False)
    pts, nrm, col = points[pick], normals[pick], colours[pick] / 255.0
    spacing = cKDTree(pts).query(pts, k=4)[0][:, 1:].mean(axis=1)
    radius = np.clip(spacing, 1e-6, None)
    n_rest = (sh_degree + 1) ** 2 - 1
    as_param = lambda a: torch.nn.Parameter(torch.tensor(np.asarray(a), dtype=torch.float32, device=device))
    return torch.nn.ParameterDict({
        "means": as_param(pts),
        "scales": as_param(np.log(np.stack([radius, radius, 0.1 * radius], 1))),
        "quats": as_param(_quats_from_normals(nrm)),
        "opacities": as_param(np.full(len(pts), _logit(0.5))),
        "sh0": as_param(((col - 0.5) / SH_C0)[:, None, :]),
        "shN": as_param(np.zeros((len(pts), n_rest, 3))),
    })


def init_from_hull(hull_points: np.ndarray, hull_voxel: float, views: List[FineView],
                   count: int, sh_degree: int, device: str = "cuda"):
    """Gaussians seeded on hull voxels -- interior ones included -- coloured from the photos."""
    import torch

    rng = np.random.default_rng(0)
    pick = rng.choice(len(hull_points), min(count, len(hull_points)), replace=False)
    pts = hull_points[pick] + rng.uniform(-0.5, 0.5, (len(pick), 3)) * hull_voxel

    # colour: the median over a few views where the seed lands on the plant
    colours = np.full((len(pts), 3), np.nan)
    samples = []
    for v in views[:: max(1, len(views) // 8)]:
        c = pts @ v.world_to_camera[:3, :3].T + v.world_to_camera[:3, 3]
        uv = (c[:, :2] / c[:, 2:3]) @ v.K[:2, :2].T + v.K[:2, 2]
        x, y = uv[:, 0].round().astype(int), uv[:, 1].round().astype(int)
        h, w = v.mask.shape
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        ok[ok] &= v.mask[y[ok], x[ok]] > 127
        s = np.full((len(pts), 3), np.nan)
        s[ok] = v.image[y[ok], x[ok]] / 255.0
        samples.append(s)
    colours = np.nanmedian(np.stack(samples), axis=0)
    colours = np.where(np.isnan(colours), 0.3, colours)

    q = rng.normal(size=(len(pts), 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    scales = np.log(np.tile([0.7 * hull_voxel, 0.7 * hull_voxel, 0.07 * hull_voxel], (len(pts), 1)))
    n_rest = (sh_degree + 1) ** 2 - 1
    as_param = lambda a: torch.nn.Parameter(torch.tensor(np.asarray(a), dtype=torch.float32, device=device))
    return torch.nn.ParameterDict({
        "means": as_param(pts),
        "scales": as_param(scales),
        "quats": as_param(q),
        "opacities": as_param(np.full(len(pts), _logit(0.1))),
        "sh0": as_param(((colours - 0.5) / SH_C0)[:, None, :]),
        "shN": as_param(np.zeros((len(pts), n_rest, 3))),
    })


def _render(params, view_K, view_w2c, width, height, sh_degree, distloss=False):
    import torch
    from gsplat import rasterization_2dgs

    colours = torch.cat([params["sh0"], params["shN"]], dim=1)
    return rasterization_2dgs(
        params["means"], params["quats"], torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colours,
        view_w2c[None], view_K[None], width, height,
        sh_degree=sh_degree, render_mode="RGB+ED", distloss=distloss,
        backgrounds=torch.zeros(1, 3, device=view_K.device), packed=False)


def train(params, views: List[FineView], iterations: int = 30_000, scene_scale: float = 1.0,
          sh_degree: int = 3, lambda_ssim: float = 0.2, lambda_mask: float = 0.5,
          lambda_dist: float = 0.1, dist_from: int = 3000,
          lambda_normal: float = 0.05, normal_from: int = 7000,
          device: str = "cuda", log_every: int = 500, log=print) -> dict:
    import torch
    import torch.nn.functional as F
    from gsplat import DefaultStrategy
    from pytorch_msssim import ssim as ssim_fn

    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    means_decay = 0.01 ** (1.0 / max(iterations, 1))      # lr -> 1% over the run (3DGS's schedule)
    strategy = DefaultStrategy(key_for_gradient="gradient_2dgs", refine_start_iter=500,
                               refine_stop_iter=iterations // 2, reset_every=3000,
                               refine_every=100, prune_opa=0.05, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    masks = [loss_masks(v) for v in views]
    Ks = [torch.tensor(v.K, dtype=torch.float32, device=device) for v in views]
    w2cs = [torch.tensor(v.world_to_camera, dtype=torch.float32, device=device) for v in views]
    rng = np.random.default_rng(0)
    history, t0 = [], time.time()
    for step in range(iterations):
        i = int(rng.integers(len(views)))
        v = views[i]
        h, w = v.mask.shape
        sure, maybe, evidence = (torch.from_numpy(m).to(device, non_blocking=True).float()
                                 for m in masks[i])
        target = torch.from_numpy(v.image).to(device, non_blocking=True).float() / 255.0
        target = target * maybe[..., None]

        degree = min(step // 1000, sh_degree)
        rgb, alpha, normals, surf_normals, distort, _median, meta = _render(
            params, Ks[i], w2cs[i], w, h, degree, distloss=step >= dist_from)
        strategy.step_pre_backward(params, optimizers, state, step, meta)

        rgb, alpha = rgb[0, ..., :3], alpha[0, ..., 0]
        weight = evidence[..., None]
        l1 = ((rgb - target).abs() * weight).sum() / (weight.sum() * 3).clamp(min=1)
        ssim = 1.0 - ssim_fn((rgb * weight).permute(2, 0, 1)[None],
                             (target * weight).permute(2, 0, 1)[None], data_range=1.0)
        # silhouette: alpha 1 where the mask is surely plant, 0 where surely
        # not; the edge band and the holder's shadow are left to colour
        judged = ((sure > 0) | (maybe == 0)) & (evidence > 0)
        mask_term = ((alpha - sure).abs() * judged).sum() / judged.sum().clamp(min=1)
        loss = (1 - lambda_ssim) * l1 + lambda_ssim * ssim + lambda_mask * mask_term
        dist_term = normal_term = torch.zeros((), device=device)
        if step >= dist_from and (sure > 0).any():
            # gsplat's distortion is in scene depth units (spread of the
            # surfaces along a ray); dividing by the plant's size makes the
            # weight independent of how far the cameras stood
            dist_term = distort[0, ..., 0][sure > 0].mean() / scene_scale
            loss = loss + lambda_dist * dist_term
        if step >= normal_from and (sure > 0).any():
            # surf_normals has no camera axis in gsplat 1.5 (H, W, 3)
            depth_normals = surf_normals[0] if surf_normals.dim() == 4 else surf_normals
            agree = (normals[0] * depth_normals).sum(-1)
            normal_term = (1.0 - agree[sure > 0]).mean()
            loss = loss + lambda_normal * normal_term

        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for group in optimizers["means"].param_groups:
            group["lr"] *= means_decay
        strategy.step_post_backward(params, optimizers, state, step, meta, packed=False)

        if step % log_every == 0 or step == iterations - 1:
            row = {"step": step, "loss": float(loss), "l1": float(l1), "ssim": float(ssim),
                   "mask": float(mask_term), "distort": float(dist_term),
                   "normal": float(normal_term), "gaussians": int(len(params["means"])),
                   "seconds": round(time.time() - t0, 1)}
            history.append(row)
            log(f"  step {step:6d}  loss {row['loss']:.4f}  l1 {row['l1']:.4f}  mask "
                f"{row['mask']:.4f}  dist {row['distort']:.5f}  normal {row['normal']:.3f}  "
                f"{row['gaussians']:,} Gaussians  {row['seconds']:.0f}s")
    return {"history": history, "sh_degree": sh_degree}


def extract(params, views: List[FineView], hull_points: np.ndarray, p3_focal: float,
            sh_degree: int = 3, alpha_threshold: float = 0.5, consistency_px: float = 4.0,
            min_consistent: int = 2, neighbours: int = 8, stride: int = 2,
            hull_slack_px: float = 6.0, device: str = "cuda", log=print):
    """Surface points from rendered median depth, kept where other views agree.

    Returns (points, normals, colours uint8, stats). A point from view i is
    kept when, among its `neighbours` nearest views (by viewing direction),
    at least `min_consistent` render a surface within `consistency_px` of
    its depth (in that view's pixels at the point's depth) where they see
    it -- and when it is no further outside the P4a hull than
    `hull_slack_px` P3 pixels at its depth (the hull was carved on P3's
    masks, so that is the unit its error comes in).
    """
    import torch
    from scipy.spatial import cKDTree

    # every view's depth and coverage, kept for the consistency test; depth as
    # a float16 offset from the view's own median (a few 0.01 mm of precision
    # at boom-rig distances, half the memory of float32 at full resolution)
    depth_of, ref_of, seen_of = [], [], []
    with torch.no_grad():
        for v in views:
            h, w = v.mask.shape
            K = torch.tensor(v.K, dtype=torch.float32, device=device)
            w2c = torch.tensor(v.world_to_camera, dtype=torch.float32, device=device)
            _rgb, alpha, _n, _sn, _d, median, _m = _render(params, K, w2c, w, h, sh_degree)
            d, a = median[0, ..., 0], alpha[0, ..., 0] > alpha_threshold
            ref = float(d[a].median()) if a.any() else 0.0
            depth_of.append((d - ref).half().cpu().numpy())
            ref_of.append(ref)
            seen_of.append(a.cpu().numpy())
    centres = np.array([-v.world_to_camera[:3, :3].T @ v.world_to_camera[:3, 3] for v in views])
    axes = np.array([v.world_to_camera[2, :3] for v in views])
    hull_tree = cKDTree(hull_points)

    pts_all, nrm_all, col_all = [], [], []
    raw = kept_consistent = kept_hull = 0
    for i, v in enumerate(views):
        h, w = v.mask.shape
        with torch.no_grad():
            K = torch.tensor(v.K, dtype=torch.float32, device=device)
            w2c = torch.tensor(v.world_to_camera, dtype=torch.float32, device=device)
            rgb, _a, normal, _sn, _d, _md, _m = _render(params, K, w2c, w, h, sh_degree)
        _sure, maybe, evidence = loss_masks(v)
        valid = seen_of[i] & (maybe > 0) & (evidence > 0)
        sub = np.zeros_like(valid)
        sub[::stride, ::stride] = True
        ys, xs = np.nonzero(valid & sub)
        if not len(ys):
            continue
        z = depth_of[i][ys, xs].astype(np.float64) + ref_of[i]
        cam = np.stack([(xs - v.K[0, 2]) / v.K[0, 0] * z, (ys - v.K[1, 2]) / v.K[1, 1] * z, z], 1)
        R, t = v.world_to_camera[:3, :3], v.world_to_camera[:3, 3]
        X = (cam - t) @ R
        raw += len(X)

        order = np.argsort(-(axes @ axes[i]))
        near = [j for j in order if j != i][:neighbours]
        agree = np.zeros(len(X), np.int32)
        for j in near:
            u = views[j]
            cj = X @ u.world_to_camera[:3, :3].T + u.world_to_camera[:3, 3]
            uv = (cj[:, :2] / cj[:, 2:3]) @ u.K[:2, :2].T + u.K[:2, 2]
            hj, wj = u.mask.shape
            x, y = uv[:, 0].round().astype(int), uv[:, 1].round().astype(int)
            ok = (x >= 0) & (x < wj) & (y >= 0) & (y < hj) & (cj[:, 2] > 0)
            dj = np.full(len(X), np.nan)
            aj = np.zeros(len(X), bool)
            dj[ok] = depth_of[j][y[ok], x[ok]].astype(np.float64) + ref_of[j]
            aj[ok] = seen_of[j][y[ok], x[ok]]
            tol = consistency_px * cj[:, 2] / u.K[0, 0]
            agree += (aj & (np.abs(dj - cj[:, 2]) <= tol)).astype(np.int32)
        keep = agree >= min_consistent
        kept_consistent += int(keep.sum())

        slack = np.maximum(hull_slack_px * z / p3_focal, 1e-9)
        d_hull, _ = hull_tree.query(X[keep])
        inside = d_hull <= slack[keep]
        kept_hull += int(inside.sum())
        sel = np.flatnonzero(keep)[inside]
        pts_all.append(X[sel])
        nrm_all.append(normal[0].cpu().numpy()[ys[sel], xs[sel]])
        col_all.append((rgb[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()[ys[sel], xs[sel]])
        if (i + 1) % 10 == 0 or i == len(views) - 1:
            log(f"  {i + 1}/{len(views)} views: {raw:,} back-projected, {kept_consistent:,} "
                f"multi-view consistent, {kept_hull:,} inside the hull bound")

    points = np.vstack(pts_all) if pts_all else np.zeros((0, 3))
    normals = np.vstack(nrm_all) if nrm_all else np.zeros((0, 3))
    colours = np.vstack(col_all) if col_all else np.zeros((0, 3), np.uint8)
    # one point per voxel the size of a pixel's footprint at the typical depth
    depth = float(np.median(np.linalg.norm(centres - hull_points.mean(axis=0), axis=1)))
    voxel = stride * depth / float(np.median([v.K[0, 0] for v in views]))
    points, normals, colours = consolidate(points, normals, colours, voxel)
    stats = {"back_projected": raw, "multi_view_consistent": kept_consistent,
             "inside_hull_bound": kept_hull, "consolidation_voxel": voxel,
             "surface_points": int(len(points))}
    return points, normals, colours, stats


def consolidate(points, normals, colours, voxel):
    """Average points, normals and colours per voxel."""
    if not len(points):
        return points, normals, colours
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    out_p = np.zeros((len(counts), 3))
    out_n = np.zeros((len(counts), 3))
    out_c = np.zeros((len(counts), 3))
    np.add.at(out_p, inverse, points)
    np.add.at(out_n, inverse, normals)
    np.add.at(out_c, inverse, colours.astype(np.float64))
    out_p /= counts[:, None]
    out_c = (out_c / counts[:, None]).round().clip(0, 255).astype(np.uint8)
    out_n /= np.maximum(np.linalg.norm(out_n, axis=1, keepdims=True), 1e-9)
    return out_p, out_n, out_c
