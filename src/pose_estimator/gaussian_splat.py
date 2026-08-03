"""Train a 3D Gaussian Splat from a "skeleton" capture's COLMAP
reconstruction, via gsplat (https://github.com/nerfstudio-project/gsplat).

`torch`/`gsplat` are optional, heavy (GPU) dependencies (`pip install -e
".[splat]"`, plus a CUDA-matching torch build -- see the README) and are
imported lazily so the rest of this subpackage stays usable without them,
matching this subpackage's existing lazy-import convention for `pycolmap`.

Output is a standard 3DGS ply (the schema written by nearly every Gaussian
Splat tool: x,y,z, f_dc_*/f_rest_* spherical harmonics, opacity, scale_*,
rot_* -- all stored *pre-activation* (log-scale, logit-opacity), the de
facto convention this whole ecosystem shares) via gsplat's own
`export_splats`, so it drops straight into any standard splat viewer/importer
(including the Blender add-on `plant_scene_import.py` drives).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np


@dataclass
class GaussianParams:
    """A Gaussian Splat's parameters, in the standard 3DGS *pre-activation*
    convention: `log_scales`/`opacity_logits` need exp()/sigmoid() before
    use, matching what `gsplat.rasterization` and every ply-based splat
    viewer expects the *stored* (not rendered) values to be.
    """

    means: np.ndarray  # (N, 3)
    log_scales: np.ndarray  # (N, 3)
    quats_wxyz: np.ndarray  # (N, 4), not required to be normalized
    opacity_logits: np.ndarray  # (N,)
    sh0: np.ndarray  # (N, 1, 3) -- DC term
    shN: np.ndarray  # (N, K, 3) -- higher-order SH, K = (sh_degree + 1)**2 - 1


@dataclass
class TrainingView:
    image_path: Path
    width: int
    height: int
    K: np.ndarray  # (3, 3)
    world_to_camera: np.ndarray  # (4, 4)


def init_gaussians_from_pointcloud(
    xyz: np.ndarray, rgb: np.ndarray, sh_degree: int = 3, init_scale_factor: float = 1.0
) -> GaussianParams:
    """Seed one Gaussian per input point: mean = point position, color = its
    observed RGB (as the SH DC term, all higher-order SH at zero), a
    conservative starting opacity, and scale from the local point spacing
    (mean nearest-neighbor distance) so Gaussians start roughly touching
    their neighbors rather than leaving gaps or wildly overlapping.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)

    tree = cKDTree(xyz)
    k = min(4, n)
    nn_dists, _ = tree.query(xyz, k=k)
    mean_nn_dist = nn_dists[:, 1:].mean(axis=1) if k > 1 else np.ones(n)
    mean_nn_dist = np.clip(mean_nn_dist, 1e-6, None) * init_scale_factor
    log_scales = np.log(mean_nn_dist)[:, None].repeat(3, axis=1)

    rgb_unit = np.asarray(rgb, dtype=np.float64) / 255.0
    sh_c0 = 0.28209479177387814  # Y_0^0 -- converts flat RGB <-> SH DC term
    sh0 = ((rgb_unit - 0.5) / sh_c0)[:, None, :]  # (N, 1, 3)

    num_sh_rest = (sh_degree + 1) ** 2 - 1
    shN = np.zeros((n, num_sh_rest, 3))

    quats_wxyz = np.zeros((n, 4))
    quats_wxyz[:, 0] = 1.0  # identity orientation

    opacity_logits = np.full(n, _logit(0.1))

    return GaussianParams(
        means=xyz,
        log_scales=log_scales,
        quats_wxyz=quats_wxyz,
        opacity_logits=opacity_logits,
        sh0=sh0,
        shN=shN,
    )


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def load_training_views(workdir: Union[str, Path], downsample_factor: int = 1) -> List[TrainingView]:
    """Undistort `workdir`'s reconstruction once (cached under
    `workdir/undistorted/`) and return one `TrainingView` per registered
    image. gsplat's rasterizer expects pinhole (undistorted) cameras, but a
    capture reconstructed with a distortion-aware COLMAP camera model (the
    common case for `main.py`'s capture pipeline) has radial/tangential
    distortion baked into both the images and the camera parameters.
    """
    import pycolmap

    from .reconstruction import get_camera_data

    workdir = Path(workdir)
    sparse_best = workdir / "sparse" / "best"
    images_dir = workdir / "images"
    undistorted_dir = workdir / "undistorted"
    undistorted_sparse = undistorted_dir / "sparse"

    # Cache is only valid for the `sparse_best` it was built from -- if
    # pose-estimate-skeleton was re-run (new mask mode, more images,
    # etc.), sparse_best is newer than the stale undistorted/ cache, which
    # would otherwise silently pair new camera poses with old undistorted
    # images (or vice versa).
    cache_is_stale = undistorted_sparse.exists() and any(
        sparse_best.stat().st_mtime > f.stat().st_mtime for f in undistorted_sparse.iterdir()
    )
    if cache_is_stale:
        import shutil

        print(f"[gaussian_splat] {undistorted_dir} is stale (sparse_best changed) -- regenerating")
        shutil.rmtree(undistorted_dir)

    if not undistorted_sparse.exists():
        undistorted_dir.mkdir(parents=True, exist_ok=True)
        pycolmap.undistort_images(
            str(undistorted_dir), str(sparse_best), str(images_dir), output_type="COLMAP"
        )

    reconstruction = pycolmap.Reconstruction(str(undistorted_dir / "sparse"))
    camera_views = get_camera_data(reconstruction, undistorted_dir / "images")

    training_views = []
    for cv in camera_views:
        width, height, K = cv.width, cv.height, cv.K.copy()
        if downsample_factor > 1:
            width = max(1, width // downsample_factor)
            height = max(1, height // downsample_factor)
            K[0, 0] /= downsample_factor
            K[1, 1] /= downsample_factor
            K[0, 2] /= downsample_factor
            K[1, 2] /= downsample_factor
        training_views.append(
            TrainingView(image_path=cv.image_path, width=width, height=height, K=K, world_to_camera=cv.world_to_camera)
        )
    return training_views


def _load_training_image(view: TrainingView, device):
    import cv2
    import torch

    img_bgr = cv2.imread(str(view.image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"could not read training image: {view.image_path}")
    if (img_bgr.shape[1], img_bgr.shape[0]) != (view.width, view.height):
        img_bgr = cv2.resize(img_bgr, (view.width, view.height), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img_rgb).to(device)


def train(
    gaussians: GaussianParams,
    views: List[TrainingView],
    iterations: int = 30000,
    strategy: str = "default",
    sh_degree: int = 3,
    sh_degree_interval: int = 1000,
    device: str = "cuda",
    seed: int = 0,
) -> GaussianParams:
    """Optimize `gaussians` against `views` via gsplat's differentiable
    rasterizer, using gsplat's own `DefaultStrategy`/`MCMCStrategy` for
    densification/pruning (see gsplat.strategy) rather than reimplementing
    the 3DGS paper's heuristics from scratch.

    Loss is L1 + SSIM against the ground-truth training image, following
    the original 3DGS paper's weighting. SH degree ramps from 0 up to
    `sh_degree` over training (one step every `sh_degree_interval`
    iterations) for optimization stability, also standard practice.
    """
    import torch
    import torch.nn.functional as F
    from gsplat import DefaultStrategy, MCMCStrategy, rasterization
    from pytorch_msssim import ssim as ssim_fn
    from tqdm import trange

    if strategy not in ("default", "mcmc"):
        raise ValueError(f"unknown strategy: {strategy!r} (expected 'default' or 'mcmc')")
    if not views:
        raise ValueError("need at least one training view")

    torch_device = torch.device(device)

    def _make_param(array: np.ndarray, lr: float):
        param = torch.nn.Parameter(torch.as_tensor(array, dtype=torch.float32, device=torch_device))
        optimizer = torch.optim.Adam([param], lr=lr, eps=1e-15)
        return param, optimizer

    # Learning rates follow the original 3DGS paper's per-parameter schedule.
    means_p, means_opt = _make_param(gaussians.means, 1.6e-4)
    scales_p, scales_opt = _make_param(gaussians.log_scales, 5e-3)
    quats_p, quats_opt = _make_param(gaussians.quats_wxyz, 1e-3)
    opacities_p, opacities_opt = _make_param(gaussians.opacity_logits, 5e-2)
    sh0_p, sh0_opt = _make_param(gaussians.sh0, 2.5e-3)
    shN_p, shN_opt = _make_param(gaussians.shN, 2.5e-3 / 20)

    params = {
        "means": means_p,
        "scales": scales_p,
        "quats": quats_p,
        "opacities": opacities_p,
        "sh0": sh0_p,
        "shN": shN_p,
    }
    optimizers = {
        "means": means_opt,
        "scales": scales_opt,
        "quats": quats_opt,
        "opacities": opacities_opt,
        "sh0": sh0_opt,
        "shN": shN_opt,
    }

    if strategy == "default":
        strat = DefaultStrategy()
        scene_scale = float(np.linalg.norm(gaussians.means - gaussians.means.mean(axis=0), axis=1).max())
        strat_state = strat.initialize_state(scene_scale=max(scene_scale, 1e-6))
    else:
        strat = MCMCStrategy(cap_max=max(len(gaussians.means) * 4, 1_000_000))
        strat_state = strat.initialize_state()
    strat.check_sanity(params, optimizers)

    rng = np.random.default_rng(seed)
    for step in trange(iterations, desc=f"training splat ({strategy})"):
        view = views[int(rng.integers(len(views)))]
        target = _load_training_image(view, torch_device)

        current_sh_degree = min(step // sh_degree_interval, sh_degree)
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)  # (N, K+1, 3)

        rendered, _alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=torch.as_tensor(view.world_to_camera, dtype=torch.float32, device=torch_device)[None],
            Ks=torch.as_tensor(view.K, dtype=torch.float32, device=torch_device)[None],
            width=view.width,
            height=view.height,
            sh_degree=current_sh_degree,
            packed=False,  # must match the (also-default) packed=False the strategy below assumes
        )
        rendered = rendered[0]  # (H, W, 3)

        l1 = F.l1_loss(rendered, target)
        ssim_value = ssim_fn(
            rendered.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None], data_range=1.0
        )
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim_value)

        strat.step_pre_backward(params, optimizers, strat_state, step, info)
        loss.backward()
        if strategy == "mcmc":
            current_means_lr = optimizers["means"].param_groups[0]["lr"]
            strat.step_post_backward(params, optimizers, strat_state, step, info, lr=current_means_lr)
        else:
            strat.step_post_backward(params, optimizers, strat_state, step, info)

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return GaussianParams(
        means=params["means"].detach().cpu().numpy(),
        log_scales=params["scales"].detach().cpu().numpy(),
        quats_wxyz=params["quats"].detach().cpu().numpy(),
        opacity_logits=params["opacities"].detach().cpu().numpy(),
        sh0=params["sh0"].detach().cpu().numpy(),
        shN=params["shN"].detach().cpu().numpy(),
    )


def write_gaussian_ply(path: Union[str, Path], gaussians: GaussianParams) -> None:
    """Write via gsplat's own `export_splats` -- the standard binary 3DGS
    ply schema (x,y,z,f_dc_*,f_rest_*,opacity,scale_*,rot_*), understood by
    virtually every splat viewer/importer, including the Blender add-on
    this pipeline drives.
    """
    import torch
    from gsplat import export_splats

    export_splats(
        means=torch.as_tensor(gaussians.means, dtype=torch.float32),
        scales=torch.as_tensor(gaussians.log_scales, dtype=torch.float32),
        quats=torch.as_tensor(gaussians.quats_wxyz, dtype=torch.float32),
        opacities=torch.as_tensor(gaussians.opacity_logits, dtype=torch.float32),
        sh0=torch.as_tensor(gaussians.sh0, dtype=torch.float32),
        shN=torch.as_tensor(gaussians.shN, dtype=torch.float32),
        format="ply",
        save_to=str(path),
    )


def read_gaussian_ply(path: Union[str, Path]) -> GaussianParams:
    """Inverse of `write_gaussian_ply` -- reads the standard 3DGS ply schema
    back into a `GaussianParams`. Pure numpy (no `torch`/`gsplat` needed).
    """
    from .ply_io import read_ply_vertices

    fields = read_ply_vertices(path)
    n = len(fields["x"])

    means = np.stack([fields["x"], fields["y"], fields["z"]], axis=1)

    def _sorted_by_suffix(prefix: str):
        names = [k for k in fields if k.startswith(prefix)]
        return sorted(names, key=lambda k: int(k.rsplit("_", 1)[-1]))

    f_dc_names = _sorted_by_suffix("f_dc_")
    sh0 = np.stack([fields[k] for k in f_dc_names], axis=1)[:, None, :]  # (N, 1, 3)

    f_rest_names = _sorted_by_suffix("f_rest_")
    if f_rest_names:
        num_sh_bases = len(f_rest_names) // 3
        # export_splats packs shN as shN.permute(0, 2, 1).reshape(N, -1) from
        # (N, K, 3) -- i.e. channel-major, coefficient-minor; undo that here.
        shN_flat = np.stack([fields[k] for k in f_rest_names], axis=1)
        shN = shN_flat.reshape(n, 3, num_sh_bases).transpose(0, 2, 1)
    else:
        shN = np.zeros((n, 0, 3))

    scale_names = _sorted_by_suffix("scale_")
    log_scales = np.stack([fields[k] for k in scale_names], axis=1)

    rot_names = _sorted_by_suffix("rot_")
    quats_wxyz = np.stack([fields[k] for k in rot_names], axis=1)

    return GaussianParams(
        means=means, log_scales=log_scales, quats_wxyz=quats_wxyz, opacity_logits=fields["opacity"], sh0=sh0, shN=shN
    )
