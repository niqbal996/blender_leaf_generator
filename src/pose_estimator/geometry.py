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
import re
import shutil
import subprocess
import sys
import threading
import time
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
    heartbeat_seconds: float = 30.0,
    skip_env_check: bool = False,
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

    # Before staging: staging clears a previous attempt's outputs, and the
    # failures this catches are certain, not probabilistic.
    environment = None
    if not skip_env_check:
        print(f"  {backend}: checking the exporter environment ({model_python or sys.executable})...")
        environment = require_model_environment(backend, model_python, repo_root)
        cuda = environment.get("torch_cuda") or {}
        print(f"    Python {'.'.join(str(part) for part in environment['python'])}, torch CUDA "
              f"{cuda.get('build') or 'n/a'}, devices: {', '.join(cuda.get('devices') or ['none'])}")

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
    env = exporter_pythonpath(repo_root)
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

    log_path = experiment / "stdout.log"
    preflight = report_run_environment(backend, len(staged), bundle_adjust, log_path, device=device)
    print(f"  {backend}: starting official exporter (live output below; model download may take time)...")
    # Combine streams so a verbose stderr warning cannot block a quiet stdout
    # reader. -u above makes Python-level progress appear as it is emitted.
    process = subprocess.Popen(
        command, cwd=str(repo_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, env=env,
    )
    output, usage = stream_exporter_output(process, backend, log_path, heartbeat_seconds)
    returncode = process.wait()
    # Keep the established failure path valid. The combined stream contains
    # Python tracebacks and dependency errors regardless of originating fd.
    (experiment / "stderr.log").write_text(output)
    (experiment / "resources.json").write_text(json.dumps(
        {"preflight": preflight, "environment": environment, "observed": usage,
         "exit_code": returncode}, indent=2))
    if returncode:
        raise RuntimeError(diagnose_exporter_failure(
            backend, returncode, output, len(staged), usage, log_path, bundle_adjust))
    print(f"  {backend}: exporter finished in {format_elapsed(usage['elapsed_seconds'])}"
          + (f", peak GPU {usage['peak_gpu_used_gb']:.1f}/{usage['gpu_total_gb']:.1f} GB"
             if usage.get("gpu_total_gb") else ""))
    source = find_colmap_model(runner)
    standardized = sparse_model_dir(workdir, backend)
    if standardized.exists():
        shutil.rmtree(standardized)
    standardized.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, standardized)
    frame_size = frame_resolution(workdir)
    if frame_size:
        rescaled = rescale_model_to_frames(standardized, frame_size)
        if rescaled and "skipped" in rescaled:
            print(f"  WARNING: {backend} model is not at frame resolution and {rescaled['skipped']}")
        elif rescaled:
            print(f"  {backend}: model was exported at the model's own resolution; intrinsics "
                  f"mapped back to {frame_size[0]}x{frame_size[1]} frame pixels "
                  f"(scale {rescaled['scale']}, focal {rescaled['focal_in_frame_pixels']})")
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


_SAMPLE_SECONDS = 5.0
# Coarse constants for the VGGT advisory estimate only; see the docstring.
_VGGT_WEIGHTS_GB = 2.6
_VGGT_GB_PER_FRAME = 0.55
# Substrings of the official exporters' own progress prints, mapped to what
# the model is about to do next.  A stage is what makes a silent stretch
# interpretable, and the exporters emit nothing during their longest one.
_STAGE_HINTS = {
    "vggt": (
        ("Using dtype", "loading VGGT-1B weights (the first run also downloads them)"),
        ("Model loaded", "loading and squaring the input images"),
        ("images from", "aggregator + camera/depth heads over all frames at once -- the peak-VRAM stage"),
        ("Predicting Tracks", "LightGlue track prediction for bundle adjustment"),
        ("Converting to COLMAP", "building the COLMAP sparse model (CPU)"),
    ),
    "mapanything": (
        ("Loading", "loading model weights (the first run also downloads them)"),
        ("images", "inference over the staged frames"),
        ("COLMAP", "building the COLMAP sparse model (CPU)"),
    ),
}
_OOM_MARKERS = (
    "CUDA out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDA error: out of memory", "MemoryError", "std::bad_alloc",
    "cannot allocate memory", "DefaultCPUAllocator: not enough memory",
)


# VGGT's exporter and its vendored dependencies annotate with PEP 604 unions
# (``np.ndarray | None``) in evaluated positions, so they raise TypeError on
# import below 3.10 rather than failing gracefully.
_MIN_MODEL_PYTHON = {"vggt": (3, 10), "mapanything": (3, 10)}
# What each official exporter imports before it does any work.  ``pycolmap``
# is in this list but deliberately absent from the ``vggt`` extra: see
# _PYCOLMAP_HELP.
# The dotted entries are the exporters' own top imports, checked with the
# checkout on PYTHONPATH.  Importing those is what catches a missing
# transitive dependency (mapanything needs uniception, natsort, python-box...)
# without reimplementing either project's requirements here.
_REQUIRED_MODULES = {
    "vggt": ("torch", "torchvision", "numpy", "PIL", "pycolmap", "trimesh",
             "lightglue", "einops", "safetensors", "huggingface_hub",
             "vggt.models.vggt", "vggt.dependency.np_to_pycolmap"),
    "mapanything": ("torch", "numpy", "PIL", "pycolmap", "huggingface_hub",
                    "mapanything.models", "mapanything.utils.colmap_export"),
}
_PYCOLMAP_HELP = (
    "pip install pycolmap  (or pycolmap-cuda for GPU SIFT). The [vggt] extra "
    "deliberately pins neither: both packages provide the same `pycolmap` module, "
    "so naming one would clobber the build this project's own P3 solver uses."
)
_MODULE_HELP = {
    "pycolmap": _PYCOLMAP_HELP,
    "lightglue": 'pip install "lightglue @ git+https://github.com/jytime/LightGlue.git"',
    "torch": "install a CUDA-matching torch build first: https://pytorch.org/get-started/locally/",
}
_ENVIRONMENT_MARKERS = (
    ("unsupported operand type(s) for |",
     "a vendored file uses PEP 604 `X | None` annotations, which need Python 3.10 or newer -- "
     "point --model-python at a newer interpreter"),
    ("ModuleNotFoundError: No module named 'pycolmap'", _PYCOLMAP_HELP),
    ("No module named 'mapanything'",
     "the MapAnything checkout is not installed; its exporter lives in scripts/, so Python "
     'does not put the package beside it on sys.path -- pip install -e "<checkout>[colmap]"'),
    ("No module named 'uniception'",
     'MapAnything\'s dependencies are not installed: pip install -e "<checkout>[colmap]"'),
    ("object has no attribute 'id'",
     "the installed pycolmap is newer than the API this exporter is written against "
     "(pycolmap 4.x renamed Image.id); install the version its requirements_demo.txt pins"),
    ("Cannot import the C++ backend",
     "the installed pycolmap wheel cannot load its compiled core; the exporter only writes a "
     "model, so plain pycolmap at the pinned version is the simplest fix"),
    ("libcudart", "an installed CUDA extension cannot find its CUDA runtime; put it on LD_LIBRARY_PATH"),
    ("ModuleNotFoundError", "a dependency of the official exporter is missing from that environment"),
    ("ImportError", "a dependency of the official exporter is present but unusable"),
)
# Printed as one line by the probe so surrounding warnings cannot confuse the
# parse -- importing torch is noisy on many installs.
_PROBE_MARKER = "POSE_GEOMETRY_PROBE "
_PROBE_SCRIPT = f'''
import json, sys
report = {{"python": list(sys.version_info[:3]), "executable": sys.executable, "modules": {{}}}}
for name in sys.argv[1:]:
    try:
        __import__(name)
        report["modules"][name] = getattr(sys.modules[name], "__version__", "present")
    except BaseException as exc:            # a bad wheel can raise anything
        report["modules"][name] = f"MISSING: {{type(exc).__name__}}: {{exc}}"
try:
    import pycolmap
    report["pycolmap_version"] = getattr(pycolmap, "__version__", "unknown")
    try:
        # The exporter's own call. pycolmap 4.x renamed `id` and made
        # cam_from_world read-only, so importability proves nothing.
        pycolmap.Image(id=1, name="probe", camera_id=1, cam_from_world=pycolmap.Rigid3d())
        report["pycolmap_exporter_api"] = "ok"
    except BaseException as exc:
        report["pycolmap_exporter_api"] = f"INCOMPATIBLE: {{type(exc).__name__}}: {{exc}}"
except BaseException:
    pass
try:
    import torch
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    report["torch_cuda"] = {{
        "available": torch.cuda.is_available(), "build": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(count)],
    }}
except BaseException:
    pass
print({_PROBE_MARKER!r} + json.dumps(report))
'''


def exporter_pythonpath(repo_root: Optional[Path], base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment with the checkout importable by the exporter's own imports.

    Python puts the *script's* directory on ``sys.path``, not the working
    directory, so MapAnything's ``scripts/demo_colmap.py`` cannot see the
    ``mapanything`` package beside it.  VGGT only works without this because
    its exporter happens to sit at its repository root.
    """
    env = dict(base if base is not None else os.environ)
    if repo_root is None:
        return env
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing}" if existing else str(repo_root)
    return env


def exporter_pinned_pycolmap(repo_root: Optional[Path], default: str = "3.10.0") -> str:
    """The pycolmap version the checked-out exporter pins for itself.

    Reading it from the checkout means this stays correct when upstream bumps
    the pin, rather than encoding today's answer in this file.
    """
    if repo_root is None:
        return default
    # VGGT pins in requirements_demo.txt, MapAnything in its pyproject's
    # "colmap" extra; both are plain `pycolmap==x.y.z` text.
    for name in ("requirements_demo.txt", "requirements.txt", "pyproject.toml"):
        try:
            content = (Path(repo_root) / name).read_text()
        except OSError:
            continue
        match = re.search(r"pycolmap\s*==\s*([\w.]+)", content)
        if match:
            return match.group(1)
    return default


def probe_model_environment(
    backend: str,
    model_python: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Dict:
    """Ask the model's interpreter what it actually has, without loading a model.

    ``--model-python`` means the exporter may run in a different environment
    from this CLI, so the only trustworthy answer comes from that interpreter.
    Importing torch takes a few seconds; spending them here is far better than
    failing after staging frames and downloading weights.
    """
    executable = model_python or sys.executable
    required = _REQUIRED_MODULES.get(backend, ("torch", "numpy", "pycolmap"))
    try:
        result = subprocess.run([executable, "-c", _PROBE_SCRIPT, *required],
                                text=True, capture_output=True, timeout=300,
                                env=exporter_pythonpath(repo_root))
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not run {executable} to check the {backend} environment: {exc}")
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_PROBE_MARKER):
            report = json.loads(line[len(_PROBE_MARKER):])
            report["required"] = list(required)
            report["repo_root"] = str(repo_root) if repo_root else None
            return report
    raise RuntimeError(
        f"could not read the {backend} environment report from {executable} "
        f"(exit {result.returncode}). Output was:\n{(result.stdout + result.stderr).strip()[:2000]}"
    )


def require_model_environment(
    backend: str,
    model_python: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Dict:
    """Fail with instructions when the exporter's interpreter cannot run it.

    Both failures this catches are environment mistakes, not model problems,
    and both otherwise surface as an opaque traceback from inside a vendored
    file several minutes into a run.
    """
    report = probe_model_environment(backend, model_python, repo_root)
    version = tuple(report["python"])
    problems = []
    minimum = _MIN_MODEL_PYTHON.get(backend)
    if minimum and version < minimum:
        problems.append(
            f"{report['executable']} is Python {'.'.join(map(str, version))}, but the official "
            f"{backend} exporter needs {'.'.join(map(str, minimum))} or newer: it annotates with "
            f"`X | None` in positions Python evaluates at import time, which raises "
            f"\"unsupported operand type(s) for |\" on older versions.\n"
            f"      Create a newer environment and point --model-python at its python; the rest "
            f"of this pipeline can stay where it is."
        )
    pinned = exporter_pinned_pycolmap(repo_root)
    missing = {name: detail for name, detail in report["modules"].items() if detail.startswith("MISSING:")}
    for name, detail in missing.items():
        hint = _MODULE_HELP.get(name)
        # A pycolmap wheel that is installed but broken is a separate case from
        # an absent one, and the CUDA build is the usual reason: this exporter
        # only writes a model, so it gains nothing from GPU SIFT.
        if name == "pycolmap" and ("libcudart" in detail or "_core" in detail):
            hint = (f'pip install "pycolmap=={pinned}" -- the exporter only writes a COLMAP '
                    f"model and never extracts features, so the pycolmap-cuda build buys it "
                    f"nothing and only has to load")
        elif name == "pycolmap":
            hint = f'pip install "pycolmap=={pinned}"  ({_PYCOLMAP_HELP})'
        elif name.startswith(backend + "."):
            # The checkout is already on PYTHONPATH by here, so this is a real
            # missing dependency of the project rather than a path problem.
            extra = "[colmap]" if backend == "mapanything" else ""
            hint = (f'pip install -e "{report.get("repo_root") or "<checkout>"}{extra}" '
                    f"-- the official {backend} project and its dependencies, in an environment "
                    f"of its own (the two backends pin different lightglue forks)")
        problems.append(f"{name} cannot be imported in {report['executable']}: "
                        f"{detail[len('MISSING: '):]}" + (f"\n      Fix: {hint}" if hint else ""))
    api = report.get("pycolmap_exporter_api", "")
    if api.startswith("INCOMPATIBLE:"):
        problems.append(
            f"pycolmap {report.get('pycolmap_version', '?')} in {report['executable']} cannot build "
            f"the model this exporter writes: {api[len('INCOMPATIBLE: '):]}\n"
            f"      The exporter is written against the version it pins for itself, and pycolmap 4.x "
            f"renamed `Image.id` and made `cam_from_world` read-only.\n"
            f'      Fix: pip install "pycolmap=={pinned}" in the exporter environment.'
        )
    if problems:
        raise RuntimeError(
            f"the {backend} exporter cannot run in this environment:\n    - "
            + "\n    - ".join(problems)
            + "\n  Nothing was staged or downloaded. Re-run with --skip-env-check to try anyway."
        )
    return report


def read_host_memory() -> Optional[Dict[str, float]]:
    """Host RAM totals in GB, or None where /proc/meminfo is unavailable."""
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            fields[key] = float(rest.split()[0]) / 1024 / 1024
    except (OSError, IndexError, ValueError):
        return None
    if "MemTotal" not in fields:
        return None
    return {"total_gb": fields["MemTotal"], "available_gb": fields.get("MemAvailable", fields["MemTotal"])}


def read_gpu_memory(device: Optional[str] = None) -> Optional[List[Dict]]:
    """Per-GPU memory via nvidia-smi, or None when it cannot be queried.

    ``device`` is the same string as ``CUDA_VISIBLE_DEVICES``; when it selects
    specific indices the report is narrowed to those, so the numbers describe
    the GPU the subprocess will actually use.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    wanted = {part.strip() for part in device.split(",")} if device else None
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index, name, used, total = parts[0], parts[1], float(parts[2]) / 1024, float(parts[3]) / 1024
        except ValueError:
            continue
        if wanted and index not in wanted:
            continue
        gpus.append({"index": index, "name": name, "used_gb": used, "total_gb": total,
                     "free_gb": max(total - used, 0.0)})
    return gpus or None


def read_gpu_processes() -> Optional[List[Dict]]:
    """Compute processes on the GPUs, with per-process VRAM where offered.

    ``used_memory`` is reported as ``[N/A]`` under WSL, so only the pids are
    dependable there.  Knowing *that* another run still holds the card is
    already the answer to a mysteriously tiny free figure.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    processes = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        try:
            used = float(parts[2]) / 1024
        except ValueError:
            used = None  # "[N/A]" under WSL
        processes.append({"pid": int(parts[0]), "name": parts[1], "used_gb": used})
    return processes


def read_process_memory(pid: int) -> Optional[Dict[str, float]]:
    """Current and peak resident set size of one process, in GB.

    ``VmHWM`` is the kernel's own high-water mark, so it survives the spike
    that a host out-of-memory kill ends on.
    """
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    values = {}
    for key, field in (("rss_gb", "VmRSS:"), ("peak_rss_gb", "VmHWM:")):
        match = re.search(rf"^{field}\s+(\d+) kB", status, re.MULTILINE)
        if match:
            values[key] = float(match.group(1)) / 1024 / 1024
    return values or None


def estimate_vggt_vram_gb(num_frames: int, bundle_adjust: bool = False) -> float:
    """Advisory estimate of VGGT-1B's peak VRAM at its fixed 518x518 input.

    VGGT's aggregator alternates frame-wise attention with *global* attention
    over the tokens of every frame at once, so cost grows with frame count
    instead of staying per-image constant -- which is why a run that works at
    8 frames can die at 30.  The two constants are a coarse fit to observed
    usage, not a model of the network: they exist only to warn before a long
    silent run and to propose a frame count, never to block one.
    """
    per_frame = _VGGT_GB_PER_FRAME * (1.6 if bundle_adjust else 1.0)
    return _VGGT_WEIGHTS_GB + per_frame * max(num_frames, 1)


def suggest_max_images(free_gb: float, bundle_adjust: bool = False) -> int:
    """Frame count whose estimate fits in ``free_gb``, keeping a 10% margin."""
    per_frame = _VGGT_GB_PER_FRAME * (1.6 if bundle_adjust else 1.0)
    budget = free_gb * 0.9 - _VGGT_WEIGHTS_GB
    return max(4, int(budget // per_frame))


def format_elapsed(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def stage_for_line(backend: str, line: str) -> Optional[str]:
    for marker, stage in _STAGE_HINTS.get(backend, ()):  # first match wins
        if marker in line:
            return stage
    return None


def report_run_environment(
    backend: str,
    num_images: int,
    bundle_adjust: bool,
    log_path: Path,
    device: Optional[str] = None,
) -> Dict:
    """Print what the run is about to cost, before the model goes quiet.

    The expensive stage produces no output for many minutes and is where
    these models run out of memory, so the memory situation is far more
    useful in front of it than in a post-mortem.
    """
    host, gpus = read_host_memory(), read_gpu_memory(device)
    others = [proc for proc in (read_gpu_processes() or ()) if proc["pid"] != os.getpid()]
    snapshot = {"num_images": num_images, "bundle_adjust": bundle_adjust, "host": host, "gpus": gpus,
                "other_gpu_processes": others}
    print(f"  {backend}: {num_images} masked frames"
          f"{', bundle adjustment on' if bundle_adjust else ''}; live log {log_path}")
    if host:
        print(f"  host RAM: {host['available_gb']:.1f} GB available of {host['total_gb']:.1f} GB")
    for gpu in gpus or ():
        print(f"  GPU {gpu['index']} ({gpu['name']}): {gpu['free_gb']:.1f} GB free "
              f"of {gpu['total_gb']:.1f} GB")
    if gpus and others:
        pids = ", ".join(f"pid {proc['pid']}" for proc in others)
        print(f"  WARNING: {len(others)} other compute process(es) already hold this GPU ({pids}); "
              f"only the free memory above is available. Stop them for the full card.")
    if gpus is None:
        print("  GPU memory could not be queried (no nvidia-smi); running on CPU is far slower")
    elif backend == "vggt":
        # Size against the free memory, but suggest against the whole card:
        # a frame count worth retyping is one that works once the GPU is idle.
        free = max(gpu["free_gb"] for gpu in gpus)
        capacity = max(gpu["total_gb"] for gpu in gpus)
        needed = estimate_vggt_vram_gb(num_images, bundle_adjust)
        snapshot["estimated_vram_gb"] = round(needed, 1)
        if needed > free:
            print(f"  WARNING: VGGT needs roughly {needed:.1f} GB for {num_images} frames but "
                  f"{free:.1f} GB is free. It attends over all frames at once, so expect an "
                  f"out-of-memory failure or heavy swapping.")
            print(f"  WARNING: consider --max-images {suggest_max_images(capacity, bundle_adjust)}"
                  f"{' and dropping --bundle-adjust' if bundle_adjust else ''} "
                  f"(an estimate from this card's {capacity:.0f} GB, not a hard limit).")
    return snapshot


def stream_exporter_output(
    process: subprocess.Popen,
    backend: str,
    log_path: Path,
    heartbeat_seconds: float = 30.0,
) -> Tuple[str, Dict]:
    """Relay exporter output with elapsed times, and speak up while it is silent.

    Both exporters spend their longest stretch inside a single silent CUDA
    call, which is indistinguishable from a hang and is exactly where memory
    runs out.  A sampling thread therefore reports the last stage reached
    together with live GPU/host/process memory, and returns the peaks so a
    failure can say how close the run came to the limit.  Output is written
    to ``log_path`` as it arrives, so an interrupt or a kill still leaves one.
    """
    started = time.monotonic()
    lock = threading.Lock()
    finished = threading.Event()
    state = {
        "stage": "starting the exporter process", "last_line": "", "last_output": started,
        "peak_gpu_used_gb": 0.0, "gpu_total_gb": 0.0, "peak_process_rss_gb": 0.0,
        "min_host_available_gb": None,
    }

    def sample() -> str:
        gpus, host, proc = read_gpu_memory(), read_host_memory(), read_process_memory(process.pid)
        parts = []
        with lock:
            if gpus:
                busiest = max(gpus, key=lambda gpu: gpu["used_gb"])
                state["peak_gpu_used_gb"] = max(state["peak_gpu_used_gb"], busiest["used_gb"])
                state["gpu_total_gb"] = busiest["total_gb"]
                parts.append(f"GPU {busiest['used_gb']:.1f}/{busiest['total_gb']:.1f} GB used")
            if host:
                previous = state["min_host_available_gb"]
                state["min_host_available_gb"] = (host["available_gb"] if previous is None
                                                  else min(previous, host["available_gb"]))
                parts.append(f"host {host['available_gb']:.1f} GB free")
            if proc:
                peak = proc.get("peak_rss_gb", proc.get("rss_gb", 0.0))
                state["peak_process_rss_gb"] = max(state["peak_process_rss_gb"], peak)
                parts.append(f"RSS {proc.get('rss_gb', peak):.1f} GB (peak {peak:.1f})")
        return "".join(f" | {part}" for part in parts)

    def monitor() -> None:
        last_beat = started
        while not finished.wait(_SAMPLE_SECONDS):
            usage = sample()  # sampled more often than reported, to catch peaks
            now = time.monotonic()
            with lock:
                silent_for, stage = now - state["last_output"], state["stage"]
            # Zero means "stay quiet", but keep sampling: the peak figures are
            # what make a later out-of-memory failure explicable.
            if heartbeat_seconds > 0 and silent_for >= heartbeat_seconds and now - last_beat >= heartbeat_seconds:
                last_beat = now
                print(f"    [{backend} {format_elapsed(now - started)}] still running, no output "
                      f"for {int(silent_for)}s | {stage}{usage}", flush=True)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    lines: List[str] = []
    assert process.stdout is not None
    try:
        with open(log_path, "w") as log:
            for line in process.stdout:
                now = time.monotonic()
                lines.append(line)
                log.write(line)
                log.flush()
                with lock:
                    state["last_output"], state["last_line"] = now, line.strip()
                    state["stage"] = stage_for_line(backend, line) or state["stage"]
                print(f"    [{backend} {format_elapsed(now - started)}] {line}", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n  interrupted -- stopping the {backend} exporter so it releases the GPU", flush=True)
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    finally:
        finished.set()
        thread.join(timeout=_SAMPLE_SECONDS + 1)
    with lock:
        usage = {key: state[key] for key in
                 ("stage", "last_line", "peak_gpu_used_gb", "gpu_total_gb",
                  "peak_process_rss_gb", "min_host_available_gb")}
    usage["elapsed_seconds"] = round(time.monotonic() - started, 1)
    return "".join(lines), usage


def diagnose_exporter_failure(
    backend: str,
    returncode: int,
    output: str,
    num_staged: int,
    usage: Dict,
    log_path: Path,
    bundle_adjust: bool = False,
) -> str:
    """Explain a failed exporter run, naming out-of-memory when it fits.

    A bare exit code is the least useful thing to report here: the two common
    outcomes are a CUDA allocation failure in the child and a host
    out-of-memory kill, which arrives as SIGKILL with no traceback at all.
    """
    elapsed = format_elapsed(usage.get("elapsed_seconds", 0))
    detail = [f"{backend} exporter exited {returncode} after {elapsed} while: {usage.get('stage', 'unknown')}"]
    killed_by_host = returncode in (-9, 137)
    if killed_by_host:
        detail.append("Killed by SIGKILL with no traceback, which is normally the host "
                      "out-of-memory killer (common under WSL, whose RAM is capped) rather than a crash.")
    cuda_oom = any(marker in output for marker in _OOM_MARKERS)
    peak_gpu, gpu_total = usage.get("peak_gpu_used_gb") or 0.0, usage.get("gpu_total_gb") or 0.0
    if gpu_total:
        detail.append(f"Peak GPU memory in use on the device, all processes together: "
                      f"{peak_gpu:.1f} of {gpu_total:.1f} GB.")
    if usage.get("peak_process_rss_gb"):
        detail.append(f"Peak exporter host memory: {usage['peak_process_rss_gb']:.1f} GB "
                      f"(host free fell to {usage.get('min_host_available_gb') or 0.0:.1f} GB).")
    environment = next((help_text for marker, help_text in _ENVIRONMENT_MARKERS if marker in output), None)
    if environment and not cuda_oom:
        detail.append(f"This is an environment problem, not a model failure: {environment}.")
        detail.append("Re-running with --dry-run checks the exporter environment without loading a model.")
    elif cuda_oom or killed_by_host:
        suggestion = suggest_max_images(gpu_total or 8.0, bundle_adjust) if backend == "vggt" else max(4, num_staged // 2)
        detail.append(f"This is an out-of-memory failure. {num_staged} frames were staged; retry with "
                      f"--max-images {min(suggestion, max(num_staged - 1, 4))}"
                      f"{' and without --bundle-adjust' if bundle_adjust else ''}.")
    elif usage.get("last_line"):
        detail.append(f"Last output: {usage['last_line']}")
    detail.append(f"Full exporter output: {log_path}")
    return "\n  ".join(detail)


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


def read_capture_passes(workdir: Path) -> Optional[Dict[str, int]]:
    """P1's frame -> capture-pass map, when the capture had more than one.

    ``evaluate_poses`` fits one circle per pass and then checks that the
    passes share a rotation axis.  Without this, a two-elevation capture is
    scored against a single circle it cannot lie on, which reports a large
    residual for a perfectly good solve -- and it loses the strongest check
    there is.  pose-solve already passes this, so a learned backend scored
    without it is not comparable to the baseline.
    """
    try:
        sources = json.loads((workdir / "p1" / "sources.json").read_text())
    except (OSError, ValueError):
        return None
    passes = {str(name): int(index) for name, index in sources.items()}
    return passes if len(set(passes.values())) > 1 else None


def frame_resolution(workdir: Path) -> Optional[Tuple[int, int]]:
    """(width, height) of the P1 frames, which the P2 masks share."""
    frames = sorted((workdir / "p1" / "frames").glob("frame_*.jpg"))
    if not frames:
        return None
    image = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return int(image.shape[1]), int(image.shape[0])


def rescale_model_to_frames(model_dir: Path, frame_size: Tuple[int, int]) -> Optional[Dict]:
    """Express an exported model's cameras in original-frame pixels.

    Every consumer downstream -- silhouette carving, organ-label fusion,
    the silhouette metric -- projects into a full-resolution P2 mask, so a
    model is only usable if its intrinsics speak that coordinate system.
    VGGT's exporter rescales back to the input resolution itself; MapAnything's
    writes the model in its own working resolution (518x336 for a 3:2 frame)
    beside a folder of resized images, which is self-consistent but silently
    projects every point into the top-left corner of a 1920x1280 mask.  That
    scored exactly 0.000 in-silhouette -- a units bug wearing the costume of
    a bad model.

    Both projects reach that resolution the same way: scale by the larger of
    the two ratios, then centre-crop to the target.  Inverting that is exact,
    so nothing here is fitted.  Returns None when the model already matches.
    """
    import pycolmap

    width, height = frame_size
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    cameras = list(reconstruction.cameras.values())
    if not cameras or all(int(c.width) == width and int(c.height) == height for c in cameras):
        return None

    changed = {}
    for camera in reconstruction.cameras.values():
        scale = max(int(camera.width) / width, int(camera.height) / height)
        crop_x = (round(width * scale) - int(camera.width)) / 2.0
        crop_y = (round(height * scale) - int(camera.height)) / 2.0
        params = np.asarray(camera.params, dtype=float).copy()
        # PINHOLE (fx, fy, cx, cy) and SIMPLE_PINHOLE (f, cx, cy) are what
        # these exporters write; anything else is left alone rather than
        # guessed at.
        if len(params) == 4:
            params[0] /= scale
            params[1] /= scale
            params[2] = (params[2] + crop_x) / scale
            params[3] = (params[3] + crop_y) / scale
        elif len(params) == 3:
            params[0] /= scale
            params[1] = (params[1] + crop_x) / scale
            params[2] = (params[2] + crop_y) / scale
        else:
            return {"skipped": f"camera model with {len(params)} params left unscaled"}
        camera.params = params
        camera.width, camera.height = width, height
        changed = {"scale": round(scale, 5), "crop_px": [crop_x, crop_y],
                   "focal_in_frame_pixels": round(float(params[0]), 1)}
    reconstruction.write(str(model_dir))
    return changed


def score_sparse_model(workdir: Path, backend: str, num_input_frames: int) -> Dict:
    """Create the same basic report/PLY artifacts for every backend."""
    import pycolmap

    destination = geometry_dir(workdir, backend)
    reconstruction = pycolmap.Reconstruction(str(require_sparse_model(workdir, backend)))
    report = evaluate_poses(reconstruction, num_input_frames=num_input_frames,
                            sources=read_capture_passes(workdir))
    centers, _, _ = get_registered_camera_poses(reconstruction)
    xyz, rgb = _points_and_colors(reconstruction)
    export_scene_ply(destination / "sparse_points.ply", destination / "camera_centers.ply", xyz, rgb, centers)
    return report


def silhouette_agreement(
    workdir: Path,
    backend: str,
    dilate_px: int = 2,
    coverage_radius_px: int = 6,
    max_points: int = 200_000,
) -> Dict:
    """Score a backend's cloud against the P2 silhouettes it should explain.

    Camera-centre agreement says whether two backends *agree*; it cannot say
    which is right.  The masks can: they are the one piece of geometry in this
    pipeline that is independently trusted, and P4 carving already treats them
    as truth.  Two numbers, deliberately kept apart:

    ``points_in_silhouette``  of every (point, view) pair the camera can see,
        the fraction landing inside the plant mask.  Wrong poses or wrong
        depths scatter points outside the silhouette, so this is the honest
        precision measure and it is comparable across backends.
    ``silhouette_coverage``  the fraction of mask area with a projected point
        within ``coverage_radius_px``.  This rewards dense pointmaps by
        construction, so it is only comparable between clouds of similar
        size -- read it as "does the cloud explain the whole plant, including
        the thin structures", never on its own.
    """
    import pycolmap

    from pose_estimator.hull import load_carve_cameras

    reconstruction = pycolmap.Reconstruction(str(require_sparse_model(workdir, backend)))
    cameras = load_carve_cameras(reconstruction, workdir / "p2" / "masks" / "plant",
                                 dilate_px=dilate_px)
    xyz, _ = _points_and_colors(reconstruction)
    if not cameras or not len(xyz):
        return {"views_scored": len(cameras), "num_points": int(len(xyz)),
                "note": "need both a cloud and P2 masks to score silhouette agreement"}
    points = xyz if len(xyz) <= max_points else xyz[
        np.linspace(0, len(xyz) - 1, max_points).astype(int)]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * coverage_radius_px + 1, 2 * coverage_radius_px + 1))
    observed = inside = 0
    coverages, per_view = [], {}
    for camera in cameras:
        height, width = camera.mask.shape
        pixels, in_front = camera.project(points)
        columns = np.round(pixels[:, 0]).astype(np.int64)
        rows = np.round(pixels[:, 1]).astype(np.int64)
        # Bounds come from the mask, not the model's stored width/height: the
        # exporters rescale intrinsics to the original frames but do not
        # always update that metadata.
        visible = in_front & (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
        columns, rows = columns[visible], rows[visible]
        hit = camera.mask[rows, columns]
        observed += int(visible.sum())
        inside += int(hit.sum())

        raster = np.zeros(camera.mask.shape, np.uint8)
        raster[rows, columns] = 1
        explained = cv2.dilate(raster, kernel).astype(bool) & camera.mask
        mask_area = int(camera.mask.sum())
        coverage = explained.sum() / mask_area if mask_area else 0.0
        coverages.append(coverage)
        per_view[camera.name] = {
            "points_visible": int(visible.sum()),
            "points_in_silhouette": float(hit.mean()) if visible.any() else 0.0,
            "silhouette_coverage": float(coverage),
        }

    return {
        "views_scored": len(cameras),
        "num_points": int(len(xyz)),
        "points_scored": int(len(points)),
        "point_view_pairs_visible": observed,
        "points_in_silhouette": inside / observed if observed else 0.0,
        "silhouette_coverage": float(np.mean(coverages)),
        "worst_view_coverage": float(np.min(coverages)),
        "coverage_radius_px": coverage_radius_px,
        "mask_dilate_px": dilate_px,
        "per_view": per_view,
    }


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
    # Scored against the masks, so every backend gets a number that does not
    # depend on agreeing with COLMAP.
    silhouettes = {name: silhouette_agreement(workdir, name) for name in backends}
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
    for name, entry in silhouettes.items():
        summary = {key: value for key, value in entry.items() if key != "per_view"}
        report_target = comparisons.get(name)
        if report_target is not None:
            report_target["silhouette"] = summary
    report = {"reference": "colmap",
              "models": {name: dict(_serializable_summary(models[name]),
                                    silhouette={key: value for key, value in silhouettes[name].items()
                                                if key != "per_view"})
                         for name in models},
              "comparisons": comparisons,
              "interpretation": {
                  "camera_center_alignment": "Errors after best-fit similarity alignment; lower is better. "
                                             "Measures agreement with COLMAP, not ground-truth accuracy.",
                  "points_in_silhouette": "Fraction of visible (point, view) pairs landing inside the P2 plant "
                                          "mask. Independent of COLMAP and comparable across backends: higher "
                                          "is better geometry.",
                  "silhouette_coverage": "Fraction of mask area explained by a nearby projected point. Rewards "
                                         "dense pointmaps by construction, so compare it only between clouds of "
                                         "similar size, alongside num_points.",
              }}
    out = workdir / "p3" / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(report, indent=2))
    # The numbers are the deliverable; the plot is a convenience. An exporter
    # environment need not have matplotlib, and losing the figure must not
    # discard a comparison that is already written.
    try:
        _write_comparison_plot(out / "diag" / "camera_compare.png", models)
    except ImportError as exc:
        print(f"  (no comparison plot: {exc}; compare.json is written)")
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
