"""Adapters and common reports for learned P3 geometry experiments.

The normal P3 solver remains the COLMAP baseline.  Learned systems are kept
in ``p3/experiments/<backend>/`` so a bad prediction can never silently
replace the reconstruction used by P4--P6.  Each adapter stages the same
plant-masked RGB frames, asks the model's *official* COLMAP exporter to run,
and copies the exported sparse model to a predictable location.

Keeping the interface at COLMAP's sparse-model boundary is intentional:
VGGT and MapAnything both maintain that exporter themselves, while their
Python prediction APIs change more often.  It also means an accepted
experiment can be used by the existing hull/surfel stages without a lossy
pose conversion.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pose_estimator.pose import evaluate_poses, export_scene_ply
from pose_estimator.reconstruction import get_registered_camera_poses


BACKENDS = ("colmap", "vggt", "mapanything")
_OFFICIAL_REPOS = {
    "vggt": "https://github.com/facebookresearch/vggt.git",
    "mapanything": "https://github.com/facebookresearch/map-anything.git",
}
_HF_MODELS = {"vggt": "facebook/VGGT-1B", "mapanything": "facebook/map-anything"}


def geometry_dir(workdir: Path, backend: str = "colmap") -> Path:
    """Directory containing a standardized geometry result.

    ``colmap`` deliberately resolves to the historical P3 location.  All
    experimental backends stay isolated beneath it.
    """
    if backend not in BACKENDS:
        raise ValueError("backend must be one of " + ", ".join(BACKENDS))
    return workdir / "p3" if backend == "colmap" else workdir / "p3" / "experiments" / backend


def sparse_model_dir(workdir: Path, backend: str = "colmap") -> Path:
    return geometry_dir(workdir, backend) / "sparse" / "best"


def require_sparse_model(workdir: Path, backend: str = "colmap") -> Path:
    path = sparse_model_dir(workdir, backend)
    if not path.is_dir() or not _is_colmap_model(path):
        hint = "run pose-solve first" if backend == "colmap" else f"run pose-geometry --backends {backend} first"
        raise FileNotFoundError(f"{path} is not a COLMAP sparse model -- {hint}")
    return path


def stage_masked_images(workdir: Path, destination: Path, max_images: int = 0) -> List[Path]:
    """Write RGB frames with non-plant pixels blacked out for learned models.

    Unlike COLMAP's feature masks, the model sees pixels, so masking is
    applied to RGB.  The original frame names are retained: COLMAP exporters
    use those names in ``images.bin``, which lets P4 re-use the P2 masks.
    """
    frames = sorted((workdir / "p1" / "frames").glob("frame_*.jpg"))
    if not frames:
        raise FileNotFoundError("no P1 frames -- run pose-segment first")
    mask_dir = workdir / "p2" / "masks" / "plant"
    if not mask_dir.is_dir():
        raise FileNotFoundError("no P2 plant masks -- run pose-segment first")
    selected = uniformly_sample(frames, max_images)
    destination.mkdir(parents=True, exist_ok=True)
    staged = []
    from tqdm import tqdm

    for frame in tqdm(selected, desc="  masking model inputs", unit="frame", dynamic_ncols=True):
        mask = cv2.imread(str(mask_dir / f"{frame.stem}.png"), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if mask is None or image is None:
            raise FileNotFoundError(f"need readable frame and plant mask for {frame.name}")
        if mask.shape != image.shape[:2]:
            raise ValueError(f"P2 mask shape differs from its frame: {frame.name}")
        # A small feather avoids a high-contrast hard edge becoming a feature
        # track.  This is still restricted to the P2 silhouette and therefore
        # cannot reintroduce the fixed backdrop.
        alpha = cv2.GaussianBlur((mask > 127).astype(np.float32), (0, 0), 1.0)[..., None]
        masked = np.round(image.astype(np.float32) * alpha).astype(np.uint8)
        out = destination / frame.name
        if not cv2.imwrite(str(out), masked, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"could not write {out}")
        staged.append(out)
    return staged


def uniformly_sample(items: Sequence[Path], maximum: int = 0) -> List[Path]:
    """Keep an ordered, evenly spaced subset; zero means all images."""
    if maximum <= 0 or len(items) <= maximum:
        return list(items)
    indices = np.linspace(0, len(items) - 1, maximum).round().astype(int)
    return [items[i] for i in np.unique(indices)]


def run_learned_backend(
    workdir: Path,
    backend: str,
    repo_root: Optional[Path] = None,
    max_images: int = 0,
    bundle_adjust: bool = False,
    device: Optional[str] = None,
    model_python: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_home: Optional[Path] = None,
    code_cache: Optional[Path] = None,
    auto_fetch_code: bool = True,
    dry_run: bool = False,
) -> Dict:
    """Run an official exporter and standardize its output under P3.

    Model weights are fetched by the official ``from_pretrained`` calls from
    Hugging Face.  When no ``repo_root`` is given, the small public exporter
    repository is cloned automatically into P3; users therefore never need
    to fetch either model weights or exporter code by hand.
    """
    if backend not in ("vggt", "mapanything"):
        raise ValueError("run_learned_backend supports vggt or mapanything")
    # A dry run is genuinely offline: show the future cache path rather than
    # cloning code just to prove a command can be assembled.
    if dry_run and repo_root is None:
        repo_root = backend_code_cache(code_cache) / backend
    else:
        repo_root = resolve_backend_repo(workdir, backend, repo_root, auto_fetch_code, code_cache=code_cache)

    experiment = geometry_dir(workdir, backend)
    input_images = experiment / "input" / "images"
    runner = experiment / "runner"
    # These two folders are generated exclusively by this adapter.  Clear a
    # prior attempt before staging: otherwise a second run with --max-images
    # would quietly feed the old, larger image set to the exporter.
    if input_images.exists():
        shutil.rmtree(input_images)
    if runner.exists():
        shutil.rmtree(runner)
    staged = stage_masked_images(workdir, input_images, max_images=max_images)
    runner_images = runner / "images"
    _copy_images(staged, runner_images)

    if backend == "vggt":
        script = repo_root / "demo_colmap.py"
        command = [model_python or sys.executable, "-u", str(script), f"--scene_dir={runner}"]
        if bundle_adjust:
            command.append("--use_ba")
    else:
        script = repo_root / "scripts" / "demo_colmap.py"
        output = runner / "output"
        command = [model_python or sys.executable, "-u", str(script),
                   f"--images_dir={runner_images}", f"--output_dir={output}"]

    if not dry_run and not script.is_file():
        raise FileNotFoundError(f"official {backend} exporter not found at {script}")
    # The libraries honour HF_TOKEN.  It lives only in this child process's
    # environment: never in command.json, stdout/stderr names, or reports.
    # HUGGINGFACE_HUB_TOKEN is included for older hub clients.
    resolved_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    env = os.environ.copy()
    if resolved_token:
        env["HF_TOKEN"] = resolved_token
        env["HUGGINGFACE_HUB_TOKEN"] = resolved_token
    if hf_home:
        env["HF_HOME"] = str(hf_home)
    if device:
        # Both projects honour CUDA_VISIBLE_DEVICES via PyTorch; recording it
        # and constraining this subprocess is less surprising than inventing
        # model-specific device flags.
        env["CUDA_VISIBLE_DEVICES"] = device
    (experiment / "command.json").write_text(json.dumps({
        "command": command, "dry_run": dry_run, "huggingface_model": _HF_MODELS[backend],
        "hf_token_supplied": bool(resolved_token), "code_repository": str(repo_root),
    }, indent=2))
    if dry_run:
        return {"backend": backend, "command": command, "staged_images": len(staged), "dry_run": True}

    print(f"  {backend}: starting official exporter (live output below; model download may take time)...")
    # Combine streams so a verbose stderr warning cannot block a quiet stdout
    # reader. -u above makes Python-level progress appear as it is emitted.
    process = subprocess.Popen(
        command, cwd=str(repo_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, env=env,
    )
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        print(f"    [{backend}] {line}", end="", flush=True)
    returncode = process.wait()
    output = "".join(output_lines)
    (experiment / "stdout.log").write_text(output)
    # Keep the established failure path valid. The combined stream contains
    # Python tracebacks and dependency errors regardless of originating fd.
    (experiment / "stderr.log").write_text(output)
    if returncode:
        raise RuntimeError(
            f"{backend} exporter exited {returncode}; see {experiment / 'stderr.log'}"
        )
    source = find_colmap_model(runner)
    standardized = sparse_model_dir(workdir, backend)
    if standardized.exists():
        shutil.rmtree(standardized)
    standardized.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, standardized)
    report = score_sparse_model(workdir, backend, len(staged))
    report.update({"backend": backend, "input": "P2 plant-masked RGB", "num_staged_images": len(staged),
                   "official_exporter": str(script), "bundle_adjust": bundle_adjust})
    (experiment / "poses.json").write_text(json.dumps(report, indent=2))
    return report


def resolve_backend_repo(
    workdir: Path,
    backend: str,
    repo_root: Optional[Path],
    auto_fetch_code: bool = True,
    code_cache: Optional[Path] = None,
) -> Path:
    """Return official exporter code, optionally fetching its public source.

    This intentionally fetches code separately from model weights. The former
    is a small, public Git checkout; the latter is authenticated through the
    Hugging Face client and cached according to ``HF_HOME``/its normal cache.
    """
    if repo_root is not None:
        resolved = Path(repo_root)
        if not resolved.is_dir():
            raise FileNotFoundError(f"{resolved} does not exist")
        return resolved
    if not auto_fetch_code:
        raise ValueError(f"no {backend} exporter code: give --{backend}-root or allow automatic code fetch")
    # Never put a Git checkout below the specimen: datasets commonly live on
    # /mnt/<drive> under WSL, where Git cannot chmod .git/config.lock. The
    # Linux-side cache also lets every specimen share one exporter checkout.
    resolved = backend_code_cache(code_cache) / backend
    if _exporter_script(resolved, backend).is_file():
        return resolved
    # A clone can have failed after creating its directory. It is generated
    # code only, so removing that incomplete checkout is safe and makes the
    # next attempt recover instead of treating it as a valid repository.
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", _OFFICIAL_REPOS[backend], str(resolved)],
        text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"could not fetch official {backend} exporter code (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return resolved


def backend_code_cache(code_cache: Optional[Path] = None) -> Path:
    """Linux-local cache for the small public exporter repositories.

    ``XDG_CACHE_HOME`` is respected; it is deliberately independent of the
    workdir because WSL-mounted drives do not support Git's POSIX metadata.
    """
    if code_cache is not None:
        return Path(code_cache)
    root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return root / "blender_leaf_generator" / "model_code"


def _exporter_script(repo_root: Path, backend: str) -> Path:
    return repo_root / ("demo_colmap.py" if backend == "vggt" else "scripts/demo_colmap.py")


def find_colmap_model(root: Path) -> Path:
    """Find an exporter-produced COLMAP model without assuming its layout."""
    candidates = [path.parent for path in root.rglob("cameras.bin")] + [path.parent for path in root.rglob("cameras.txt")]
    candidates = [path for path in candidates if _is_colmap_model(path)]
    if not candidates:
        raise FileNotFoundError(f"no COLMAP cameras/images model below {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _is_colmap_model(path: Path) -> bool:
    binary = all((path / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin"))
    text = all((path / name).is_file() for name in ("cameras.txt", "images.txt", "points3D.txt"))
    return binary or text


def _copy_images(paths: Iterable[Path], destination: Path) -> None:
    """Copy model inputs without POSIX metadata.

    ``copy2`` also attempts to copy timestamps and mode bits.  That fails on
    Windows-mounted WSL paths (``/mnt/c`` etc.) even when ordinary byte writes
    work, and image metadata is irrelevant to the exporter.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copyfile(path, destination / path.name)


def score_sparse_model(workdir: Path, backend: str, num_input_frames: int) -> Dict:
    """Create the same basic report/PLY artifacts for every backend."""
    import pycolmap

    destination = geometry_dir(workdir, backend)
    reconstruction = pycolmap.Reconstruction(str(require_sparse_model(workdir, backend)))
    report = evaluate_poses(reconstruction, num_input_frames=num_input_frames)
    centers, _, _ = get_registered_camera_poses(reconstruction)
    xyz, rgb = _points_and_colors(reconstruction)
    export_scene_ply(destination / "sparse_points.ply", destination / "camera_centers.ply", xyz, rgb, centers)
    return report


def compare_backends(workdir: Path, backends: Sequence[str]) -> Dict:
    """Compare learned runs to COLMAP after a similarity alignment.

    Absolute scale and coordinate axes are not common between SfM and learned
    models.  Comparing raw coordinates would therefore say nothing.  Camera
    centres with matching frame names are aligned with a 7-DoF similarity,
    then residuals measure disagreement in the actual camera trajectory.
    """
    if "colmap" not in backends:
        backends = ["colmap", *backends]
    models = {name: _load_model_summary(workdir, name) for name in backends}
    baseline = models["colmap"]
    comparisons = {}
    for name, model in models.items():
        if name == "colmap":
            continue
        shared = sorted(set(baseline["centers"]) & set(model["centers"]))
        entry = {
            "shared_registered_frames": len(shared),
            "baseline_registered_frames": baseline["num_images"],
            "candidate_registered_frames": model["num_images"],
            "candidate_registration_fraction": model["num_images"] / max(baseline["num_input_frames"], 1),
            "candidate_sparse_points": model["num_points"],
        }
        if len(shared) >= 3:
            source = np.stack([model["centers"][frame] for frame in shared])
            target = np.stack([baseline["centers"][frame] for frame in shared])
            scale, rotation, translation = similarity_transform(source, target)
            aligned = scale * source @ rotation.T + translation
            errors = np.linalg.norm(aligned - target, axis=1)
            entry["camera_center_alignment"] = {
                "scale_to_colmap": float(scale), "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "median": float(np.median(errors)), "max": float(errors.max()),
            }
        else:
            entry["camera_center_alignment"] = None
            entry["note"] = "fewer than three shared registered frames; trajectory cannot be aligned"
        comparisons[name] = entry
    report = {"reference": "colmap", "models": {name: _serializable_summary(m) for name, m in models.items()},
              "comparisons": comparisons,
              "interpretation": "Camera-centre errors are after best-fit similarity alignment; lower is better. "
                                "They assess agreement with COLMAP, not ground-truth accuracy."}
    out = workdir / "p3" / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(report, indent=2))
    _write_comparison_plot(out / "diag" / "camera_compare.png", models)
    return report


def similarity_transform(source: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity mapping ``source`` to ``target`` (no reflection)."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("source and target must both be N-by-3 with N >= 3")
    src_mean, dst_mean = source.mean(0), target.mean(0)
    src0, dst0 = source - src_mean, target - dst_mean
    covariance = dst0.T @ src0 / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = float((src0 ** 2).sum() / len(source))
    scale = float((singular * np.diag(correction)).sum() / max(variance, 1e-12))
    translation = dst_mean - scale * src_mean @ rotation.T
    return scale, rotation, translation


def _load_model_summary(workdir: Path, backend: str) -> Dict:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(require_sparse_model(workdir, backend)))
    centers = {}
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        centers[image.name] = np.asarray(image.projection_center(), dtype=float).reshape(3)
    frames = sorted((workdir / "p1" / "frames").glob("frame_*.jpg"))
    xyz, _ = _points_and_colors(reconstruction)
    return {"num_images": len(centers), "num_points": len(xyz), "num_input_frames": len(frames),
            "centers": centers, "points": xyz}


def _points_and_colors(reconstruction) -> Tuple[np.ndarray, np.ndarray]:
    points = list(reconstruction.points3D.values())
    if not points:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
    return np.stack([p.xyz for p in points]), np.stack([p.color for p in points]).astype(np.uint8)


def _serializable_summary(summary: Dict) -> Dict:
    return {key: summary[key] for key in ("num_images", "num_points", "num_input_frames")}


def _write_comparison_plot(path: Path, models: Dict[str, Dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 8))
    axis = fig.add_subplot(111, projection="3d")
    for name, summary in models.items():
        centers = np.asarray(list(summary["centers"].values()))
        if len(centers):
            axis.plot(centers[:, 0], centers[:, 1], centers[:, 2], marker="o", ms=2, label=f"{name} cameras")
        points = summary["points"]
        if len(points):
            take = np.linspace(0, len(points) - 1, min(12_000, len(points))).astype(int)
            axis.scatter(points[take, 0], points[take, 1], points[take, 2], s=0.3, alpha=0.12)
    axis.set_title("P3 learned-geometry comparison (raw model coordinate frames)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
