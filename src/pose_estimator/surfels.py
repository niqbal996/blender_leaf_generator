"""P4b -- surface-aligned splatting, to turn the hull bound into a surface.

P4a's visual hull is a *solid*: 71.5% of its voxels are fully enclosed, and
local neighbourhoods measure as near-isotropic blobs rather than sheets. That
is correct behaviour for a bound, but it is the wrong input for structure
extraction, and P5 failed on it for exactly that reason -- kNN/MST
skeletonisation assumes samples on a surface and traces arbitrary interior
paths through a solid.

This phase does what plan §P4 steps 2-4 specify:

- Train **2D Gaussians** (surfels -- flat, oriented discs) rather than
  isotropic 3DGS, so the primitives lie *on* surfaces instead of filling
  volume.
- Supervise with masked images plus an **alpha loss against the P2 mattes**,
  so the splat is shaped by the silhouettes rather than merely coloured by
  them, and a **normal-consistency** term, which is what stops a surfel from
  satisfying the photometric loss while facing the wrong way.
- Extract geometry from **rendered depth**, never from primitive centres.
  Splat centres are optimised for photometric loss and are not surface
  samples; the plan names this as the root cause of the old pipeline's need
  for hand-tuned thresholds.
- **Carve** the result against the P4a hull, which is where the hull earns its
  keep: it is the deterministic bound that rejects floaters without any
  per-plant opacity or density threshold.

Initialisation comes from the hull rather than the sparse cloud. The hull
already covers the plant densely and correctly in the aggregate, so the
optimiser starts near the answer and needs far less densification -- which is
why no densification strategy runs here at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


@dataclass
class TrainView:
    """One supervised view: image, silhouette, and calibrated pose."""

    image: np.ndarray  # (H, W, 3) float32 in 0..1, background already zeroed
    mask: np.ndarray  # (H, W) float32 in 0..1
    K: np.ndarray  # (3, 3)
    world_to_camera: np.ndarray  # (4, 4)
    name: str
    occluder: Optional[np.ndarray] = None  # (H, W) float32, 1 = hidden by the holder


def load_views(
    reconstruction,
    frames_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    downsample: int = 2,
    occluder_dir: Optional[Union[str, Path]] = None,
) -> List[TrainView]:
    """Build training views from P3 poses, P1 frames and P2 plant masks.

    Images are multiplied by their mask. The rig already shoots against a
    black backdrop, so zeroing the background costs no real signal and stops
    the optimiser spending capacity reconstructing a backdrop that is rigid
    with the camera and therefore geometrically meaningless.
    """
    frames_dir, mask_dir = Path(frames_dir), Path(mask_dir)
    occluder_dir = Path(occluder_dir) if occluder_dir is not None else None
    views: List[TrainView] = []

    for image_id in sorted(reconstruction.reg_image_ids()):
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]

        bgr = cv2.imread(str(frames_dir / image.name))
        mask = cv2.imread(str(mask_dir / f"{Path(image.name).stem}.png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            continue

        hidden = None
        if occluder_dir is not None:
            raw = cv2.imread(str(occluder_dir / f"{Path(image.name).stem}.png"),
                             cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                hidden = np.where((raw > 127) & (mask <= 127), 255, 0).astype(np.uint8)

        if downsample > 1:
            size = (bgr.shape[1] // downsample, bgr.shape[0] // downsample)
            bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
            if hidden is not None:
                hidden = cv2.resize(hidden, size, interpolation=cv2.INTER_AREA)

        K = np.asarray(camera.calibration_matrix(), dtype=np.float64).copy()
        K[:2] /= downsample

        world_to_camera = np.eye(4)
        world_to_camera[:3, :] = image.cam_from_world().matrix()

        soft_mask = (mask.astype(np.float32) / 255.0)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        views.append(
            TrainView(
                image=rgb * soft_mask[..., None],
                mask=soft_mask,
                K=K,
                world_to_camera=world_to_camera,
                name=image.name,
                occluder=None if hidden is None else hidden.astype(np.float32) / 255.0,
            )
        )
    return views


def hull_surface_voxels(hull_points: np.ndarray, voxel: float) -> np.ndarray:
    """Boolean mask selecting only the hull's boundary voxels.

    The carved hull is a solid -- measured at 71.5% fully-enclosed interior
    voxels on DSC_0009 -- and interior voxels are actively unhelpful here.
    They have no surface to align to, they cannot be seen from any view, and
    seeding surfels there spends capacity on primitives that will be hidden
    behind the ones that matter.
    """
    keys = np.round(hull_points / voxel).astype(np.int64)
    occupied = {tuple(k) for k in keys}
    face_neighbours = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    return np.array(
        [
            any((k[0] + d[0], k[1] + d[1], k[2] + d[2]) not in occupied for d in face_neighbours)
            for k in map(tuple, keys)
        ]
    )


def _quats_from_normals(normals: np.ndarray) -> np.ndarray:
    """Quaternions (w,x,y,z) whose rotation maps +Z onto each normal.

    gsplat's 2D Gaussians lie in the plane spanned by the first two axes of
    their local frame, so the third axis is the disc normal. Seeding that
    from the hull's own surface orientation is what lets the optimiser start
    with discs already lying flat against the object.
    """
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    helper = np.where(
        (np.abs(normals[:, 0]) < 0.9)[:, None], np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    tangent = np.cross(normals, helper)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    bitangent = np.cross(normals, tangent)

    # Rotation matrices with columns (tangent, bitangent, normal).
    R = np.stack([tangent, bitangent, normals], axis=2)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    w = np.sqrt(np.maximum(1.0 + trace, 1e-12)) / 2.0
    safe = np.maximum(4.0 * w, 1e-9)
    x = (R[:, 2, 1] - R[:, 1, 2]) / safe
    y = (R[:, 0, 2] - R[:, 2, 0]) / safe
    z = (R[:, 1, 0] - R[:, 0, 1]) / safe
    quats = np.stack([w, x, y, z], axis=1)
    return quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-9)


def init_surfels_from_hull(
    hull_points: np.ndarray,
    sparse_points: np.ndarray,
    sparse_colors: np.ndarray,
    hull_voxel: float,
    target_count: int = 150_000,
    device: str = "cuda",
):
    """Initialise 2D Gaussians on the hull's *surface*, oriented to it.

    Starting from the hull rather than the sparse cloud matters: the sparse
    cloud has a few thousand points concentrated on whatever happened to
    match, while the hull covers the whole plant including the thin root that
    photometric matching never recovered.

    Orientation is seeded from the local surface normal rather than left at
    identity. With identity quaternions every disc starts facing the same
    arbitrary direction, the normal-consistency loss opens at ~0.99 (normals
    essentially orthogonal to the surface), and the optimiser has to rotate
    150k primitives from scratch. Seeding it makes the first iteration
    already approximately right.
    """
    import torch
    from scipy.spatial import cKDTree

    boundary = hull_surface_voxels(hull_points, hull_voxel)
    points = hull_points[boundary]

    if len(points) > target_count:
        pick = np.random.default_rng(0).choice(len(points), target_count, replace=False)
        points = points[pick]

    # Local PCA normal: on a boundary shell the smallest principal direction
    # is the surface normal, which is exactly what interior voxels lack.
    tree = cKDTree(points)
    neighbourhoods = tree.query(points, k=min(16, len(points)))[1]
    centred = points[neighbourhoods] - points[neighbourhoods].mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centred, centred)
    normals = np.linalg.eigh(covariance)[1][:, :, 0]

    if len(sparse_points):
        _, nearest = cKDTree(sparse_points).query(points)
        colors = sparse_colors[nearest].astype(np.float32) / 255.0
    else:
        colors = np.full((len(points), 3), 0.5, np.float32)

    spacing, _ = tree.query(points, k=2)
    radius = np.clip(spacing[:, 1], 1e-6, None)

    def logit(x):
        return float(np.log(x / (1.0 - x)))

    params = {
        "means": torch.tensor(points, dtype=torch.float32, device=device),
        # 2DGS uses only the first two scale components; the third is ignored
        # by the rasterizer, which is what makes these discs rather than
        # ellipsoids.
        "scales": torch.tensor(np.log(np.stack([radius, radius, radius], 1)),
                               dtype=torch.float32, device=device),
        "quats": torch.tensor(_quats_from_normals(normals), dtype=torch.float32, device=device),
        "opacities": torch.full((len(points),), logit(0.35), dtype=torch.float32, device=device),
        "colors": torch.tensor(colors, dtype=torch.float32, device=device),
    }
    for tensor in params.values():
        tensor.requires_grad_(True)
    return params


def train(
    params: dict,
    views: Sequence[TrainView],
    iterations: int = 4000,
    lambda_mask: float = 1.0,
    lambda_normal: float = 0.05,
    lambda_ssim: float = 0.2,
    device: str = "cuda",
    log_every: int = 500,
) -> dict:
    """Optimise surfels against masked images, silhouettes and normals."""
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization_2dgs
    from pytorch_msssim import ssim as ssim_fn
    from tqdm import trange

    optimiser = torch.optim.Adam(
        [
            {"params": [params["means"]], "lr": 1.6e-4},
            {"params": [params["scales"]], "lr": 5e-3},
            {"params": [params["quats"]], "lr": 1e-3},
            {"params": [params["opacities"]], "lr": 5e-2},
            {"params": [params["colors"]], "lr": 2.5e-3},
        ],
        eps=1e-15,
    )

    images = [torch.tensor(v.image, device=device) for v in views]
    masks = [torch.tensor(v.mask, device=device) for v in views]
    # 1 where the pixel carries evidence, 0 where the holder hides whatever is
    # behind it. Without this the silhouette term trains the root away: the
    # pliers cross in front of it for most of the orbit, so in ~73% of frames
    # the mask says "empty" at pixels that are merely hidden, and opacity
    # there is driven to zero. Absence of evidence, taught as evidence of
    # absence.
    visible = [None if v.occluder is None
               else torch.tensor(1.0 - v.occluder, device=device) for v in views]
    Ks = [torch.tensor(v.K, dtype=torch.float32, device=device)[None] for v in views]
    viewmats = [torch.tensor(v.world_to_camera, dtype=torch.float32, device=device)[None] for v in views]

    rng = np.random.default_rng(0)
    history = []

    for step in trange(iterations, desc="training surfels"):
        i = int(rng.integers(len(views)))
        height, width = images[i].shape[:2]

        rendered, alphas, normals, normals_from_depth, _distort, _median, _meta = rasterization_2dgs(
            params["means"],
            F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]),
            torch.sigmoid(params["colors"])[None],
            torch.cat(viewmats[i : i + 1]),
            torch.cat(Ks[i : i + 1]),
            width,
            height,
            render_mode="RGB+ED",
        )

        rgb = rendered[0, ..., :3]
        alpha = alphas[0, ..., 0]

        photometric = F.l1_loss(rgb, images[i])
        ssim_term = 1.0 - ssim_fn(
            rgb.permute(2, 0, 1)[None], images[i].permute(2, 0, 1)[None], data_range=1.0
        )
        # The silhouette term is what makes this a geometry fit rather than a
        # texture fit: it penalises opacity anywhere the mask says empty --
        # except where the holder is in the way, which is not the same thing.
        if visible[i] is None:
            mask_term = F.l1_loss(alpha, masks[i])
        else:
            weight = visible[i]
            mask_term = ((alpha - masks[i]).abs() * weight).sum() / weight.sum().clamp(min=1.0)
        # Normal consistency: the rendered surfel normals should agree with
        # the normals implied by the rendered depth. Without it a surfel can
        # satisfy the photometric loss while oriented arbitrarily, and the
        # depth map it produces is then unusable for surface extraction.
        #
        # Restricted to the silhouette. gsplat returns depth-derived normals
        # of zero wherever nothing was rendered, so averaging over the whole
        # frame mixes in a large background region that scores a constant 1.0
        # and cannot be improved -- the loss then sits near 0.98 no matter how
        # well the surfels are actually oriented, and supervises nothing.
        # (Observed: 0.9895 at init, 0.9791 after 5000 iterations.)
        foreground = masks[i] > 0.5
        if foreground.any():
            agreement = (normals[0] * normals_from_depth).sum(-1)
            normal_term = (1.0 - agreement[foreground]).mean()
        else:
            normal_term = torch.zeros((), device=device)

        loss = (
            (1.0 - lambda_ssim) * photometric
            + lambda_ssim * ssim_term
            + lambda_mask * mask_term
            + lambda_normal * normal_term
        )

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        if step % log_every == 0 or step == iterations - 1:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "l1": float(photometric.detach()),
                    "mask_l1": float(mask_term.detach()),
                    "normal": float(normal_term.detach()),
                }
            )

    return {"params": params, "history": history}


def render_surface_points(
    params: dict,
    views: Sequence[TrainView],
    hull_points: np.ndarray,
    hull_voxel: float,
    alpha_threshold: float = 0.5,
    hull_dilation_voxels: float = 2.0,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Back-project rendered depth into a carved surface point cloud.

    Geometry comes from the rendered depth map, not the primitive centres --
    the distinction the plan insists on, because centres are wherever the
    photometric loss put them and need not lie on any surface.

    Every back-projected point is then rejected unless it falls within a
    dilated P4a hull. This is the step that replaces the old pipeline's
    manual opacity/scale thresholds: floaters are removed because an
    independent, deterministic bound says they are outside the object, not
    because someone picked a cutoff that happened to work on one plant.
    """
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization_2dgs
    from scipy.spatial import cKDTree

    hull_tree = cKDTree(hull_points)
    reject_radius = hull_dilation_voxels * hull_voxel

    all_points, all_normals = [], []
    kept_total = raw_total = 0

    with torch.no_grad():
        for view in views:
            height, width = view.image.shape[:2]
            K = torch.tensor(view.K, dtype=torch.float32, device=device)[None]
            viewmat = torch.tensor(view.world_to_camera, dtype=torch.float32, device=device)[None]

            rendered, alphas, normals, _nfd, _distort, median_depth, _meta = rasterization_2dgs(
                params["means"],
                F.normalize(params["quats"], dim=-1),
                torch.exp(params["scales"]),
                torch.sigmoid(params["opacities"]),
                torch.sigmoid(params["colors"])[None],
                viewmat,
                K,
                width,
                height,
                render_mode="RGB+ED",
            )

            # Median depth, not expected depth. Expected depth is the
            # opacity-weighted mean along each ray, so wherever the ray passes
            # through more than one semi-transparent surfel it lands *between*
            # them -- which smears a thin leaf into a slab. The median picks an
            # actual surface crossing, which is what back-projection needs.
            depth = median_depth[0, ..., 0]
            alpha = alphas[0, ..., 0]
            mask = torch.tensor(view.mask, device=device)

            # A pixel the holder covers tells us nothing, so it must not veto
            # a surfel behind it either. Anything admitted this way still has
            # to survive the hull carve below, which is what keeps floaters
            # out.
            seen = mask > 0.5
            if view.occluder is not None:
                seen = seen | (torch.tensor(view.occluder, device=device) > 0.5)
            valid = (alpha > alpha_threshold) & (depth > 1e-6) & seen
            if not valid.any():
                continue

            ys, xs = torch.nonzero(valid, as_tuple=True)
            z = depth[ys, xs]
            fx, fy = K[0, 0, 0], K[0, 1, 1]
            cx, cy = K[0, 0, 2], K[0, 1, 2]
            cam = torch.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], dim=1)

            R = viewmat[0, :3, :3]
            t = viewmat[0, :3, 3]
            world = (cam - t) @ R  # R^T (x - t), since R is orthonormal

            # gsplat already returns render_normals in world space (it applies
            # inv(viewmat) internally), so rotating again here would be a
            # second, spurious transform.
            n = normals[0][ys, xs]

            world_np = world.cpu().numpy()
            raw_total += len(world_np)

            distance, _ = hull_tree.query(world_np)
            inside = distance <= reject_radius
            kept_total += int(inside.sum())

            all_points.append(world_np[inside])
            all_normals.append(n.cpu().numpy()[inside])

    points = np.vstack(all_points) if all_points else np.zeros((0, 3))
    normals_out = np.vstack(all_normals) if all_normals else np.zeros((0, 3))
    stats = {
        "raw_backprojected": int(raw_total),
        "kept_after_hull_carve": int(kept_total),
        "rejected_fraction": float(1.0 - kept_total / max(raw_total, 1)),
    }
    return points, normals_out, stats


def consolidate(
    points: np.ndarray, normals: np.ndarray, voxel: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge the per-view depth clouds into one, averaging normals per voxel.

    96 views each contribute their own depth map, so the raw union is roughly
    96x redundant along every visible surface. Averaging within a voxel also
    suppresses the per-view depth noise that would otherwise survive as
    thickness.
    """
    if len(points) == 0:
        return points, normals

    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)

    merged = np.zeros((len(counts), 3))
    merged_normals = np.zeros((len(counts), 3))
    np.add.at(merged, inverse, points)
    np.add.at(merged_normals, inverse, normals)
    merged /= counts[:, None]

    lengths = np.linalg.norm(merged_normals, axis=1, keepdims=True)
    merged_normals = np.divide(merged_normals, np.maximum(lengths, 1e-9))
    return merged, merged_normals
