"""Run VGGT/MapAnything geometry experiments and compare them with COLMAP.

Examples:
  pose-geometry --workdir runs/plant_9 --backends vggt \
      --vggt-root ~/src/vggt --bundle-adjust
  pose-geometry --workdir runs/plant_9 --backends vggt mapanything \
      --vggt-root ~/src/vggt --mapanything-root ~/src/map-anything --max-images 80

P3 COLMAP must be run first. Learned results live under
``p3/experiments/<backend>``; they do not overwrite the baseline. The public
exporter code is fetched automatically and model weights load from Hugging
Face using ``HF_TOKEN`` (or ``--hf-token``).
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from pose_estimator.geometry import BACKENDS, compare_backends, require_sparse_model, run_learned_backend


def run(
    workdir: Path,
    backends,
    vggt_root: Optional[Path] = None,
    vggt_omega_root: Optional[Path] = None,
    mapanything_root: Optional[Path] = None,
    max_images: int = 0,
    bundle_adjust: bool = False,
    device: Optional[str] = None,
    model_python: Optional[str] = None,
    vggt_python: Optional[str] = None,
    vggt_omega_python: Optional[str] = None,
    mapanything_python: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_home: Optional[Path] = None,
    code_cache: Optional[Path] = None,
    auto_fetch_code: bool = True,
    dry_run: bool = False,
    heartbeat_seconds: float = 30.0,
    skip_env_check: bool = False,
    image_resolution: Optional[int] = None,
    omega_checkpoint: Optional[str] = None,
    no_plant_masks: bool = False,
) -> dict:
    require_sparse_model(workdir, "colmap")
    selected = list(dict.fromkeys(backends))
    for backend in selected:
        if backend == "colmap":
            continue
        root = {"vggt": vggt_root, "vggt_omega": vggt_omega_root,
                "mapanything": mapanything_root}[backend]
        # VGGT and MapAnything pin different forks of lightglue, which install
        # under the same module name, so one environment cannot serve both.
        backend_python = {"vggt": vggt_python, "vggt_omega": vggt_omega_python,
                          "mapanything": mapanything_python}[backend] or model_python
        print(f"Running {backend} on P2 plant-masked RGB frames...")
        report = run_learned_backend(workdir, backend, root, max_images=max_images,
                                     bundle_adjust=bundle_adjust, device=device, model_python=backend_python,
                                     hf_token=hf_token, hf_home=hf_home,
                                     code_cache=code_cache,
                                     auto_fetch_code=auto_fetch_code, dry_run=dry_run,
                                     heartbeat_seconds=heartbeat_seconds,
                                     skip_env_check=skip_env_check,
                                     image_resolution=image_resolution,
                                     checkpoint=omega_checkpoint if backend == "vggt_omega" else None,
                                     use_plant_masks=not no_plant_masks)
        if dry_run:
            print("  would run: " + " ".join(report["command"]))
        else:
            print(f"  {report['num_registered']}/{report['num_input_frames']} frames registered; "
                  f"{report['num_points3D']} sparse points")
    if dry_run:
        return {"dry_run": True}
    available = [name for name in selected if name == "colmap" or require_sparse_model(workdir, name).is_dir()]
    if "colmap" not in available:
        available.insert(0, "colmap")
    comparison = compare_backends(workdir, available)
    print(f"Comparison: {workdir / 'p3' / 'experiments' / 'compare.json'}")
    for name, model in comparison["models"].items():
        silhouette = model.get("silhouette") or {}
        if "points_in_silhouette" in silhouette:
            print(f"  {name}: {silhouette['points_in_silhouette']:.1%} of projected points inside the P2 "
                  f"masks, {silhouette['silhouette_coverage']:.1%} mask coverage "
                  f"({silhouette['num_points']} points)")
    for name, entry in comparison["comparisons"].items():
        alignment = entry["camera_center_alignment"]
        detail = "not alignable" if alignment is None else f"aligned camera RMSE {alignment['rmse']:.4f}"
        print(f"  {name}: {entry['candidate_registered_frames']} registered, {detail}")
    return comparison


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS,
                        default=["vggt_omega", "mapanything"],
                        help="Models to run; COLMAP is always included as the comparison reference")
    parser.add_argument("--vggt-root", type=Path,
                        help="Optional existing VGGT checkout; fetched automatically when omitted")
    parser.add_argument("--vggt-omega-root", type=Path,
                        help="Optional existing VGGT-Omega checkout; fetched automatically when omitted")
    parser.add_argument("--mapanything-root", type=Path,
                        help="Optional existing MapAnything checkout; fetched automatically when omitted")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Uniformly sample this many P1 frames (0 = all); lower for VRAM-limited trials")
    parser.add_argument("--bundle-adjust", action="store_true",
                        help="Ask VGGT's official exporter for bundle adjustment (ignored by MapAnything)")
    parser.add_argument("--device", help="CUDA_VISIBLE_DEVICES value for the model subprocess, e.g. 0")
    parser.add_argument("--model-python",
                        help="Python executable in the VGGT/MapAnything environment; default is this command's Python")
    parser.add_argument("--vggt-python",
                        help="Python for the VGGT environment specifically, overriding --model-python")
    parser.add_argument("--vggt-omega-python",
                        help="Python for the VGGT-Omega environment specifically, overriding --model-python")
    parser.add_argument("--image-resolution", type=int,
                        help="Model input resolution. Default is each model's own trained "
                             "resolution, which is also its best: VGGT-Omega 512, MapAnything's "
                             "aspect-matched bucket. Raising it leaves the training regime")
    parser.add_argument("--omega-checkpoint",
                        help="VGGT-Omega checkpoint: a local .pt, or a filename in the gated "
                             "facebook/VGGT-Omega repo (default vggt_omega_1b_512.pt)")
    parser.add_argument("--no-plant-masks", action="store_true",
                        help="Do not tell the exporters which pixels P2 called plant")
    parser.add_argument("--mapanything-python",
                        help="Python for the MapAnything environment specifically, overriding --model-python. "
                             "The two backends pin different lightglue forks, so comparing both in one run "
                             "needs an environment for each.")
    parser.add_argument("--hf-token", help="Hugging Face token; defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN. "
                        "Prefer the environment variable so the token is not in shell history/process arguments.")
    parser.add_argument("--hf-home", type=Path,
                        help="Optional Hugging Face cache directory for downloaded weights")
    parser.add_argument("--code-cache", type=Path,
                        help="Linux-local cache for public exporter code; defaults to $XDG_CACHE_HOME or ~/.cache")
    parser.add_argument("--no-auto-fetch-code", action="store_true",
                        help="Require supplied --vggt-root/--mapanything-root rather than cloning public exporter code")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0,
                        help="While the exporter prints nothing, report its stage and GPU/host memory "
                             "this often. The peak-VRAM stage is silent for minutes; 0 disables it.")
    parser.add_argument("--skip-env-check", action="store_true",
                        help="Do not verify the exporter's Python version and imports before running")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check the exporter environment and stage inputs, but do not start models")
    args = parser.parse_args(argv)
    run(args.workdir, args.backends, vggt_root=args.vggt_root,
        vggt_omega_root=args.vggt_omega_root, mapanything_root=args.mapanything_root,
        max_images=args.max_images, bundle_adjust=args.bundle_adjust, device=args.device,
        model_python=args.model_python, vggt_python=args.vggt_python,
        vggt_omega_python=args.vggt_omega_python,
        mapanything_python=args.mapanything_python, hf_token=args.hf_token, hf_home=args.hf_home,
        code_cache=args.code_cache,
        auto_fetch_code=not args.no_auto_fetch_code, dry_run=args.dry_run,
        heartbeat_seconds=args.heartbeat_seconds, skip_env_check=args.skip_env_check,
        image_resolution=args.image_resolution, omega_checkpoint=args.omega_checkpoint,
        no_plant_masks=args.no_plant_masks)


if __name__ == "__main__":
    main()
