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
    mapanything_root: Optional[Path] = None,
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
) -> dict:
    require_sparse_model(workdir, "colmap")
    selected = list(dict.fromkeys(backends))
    for backend in selected:
        if backend == "colmap":
            continue
        root = vggt_root if backend == "vggt" else mapanything_root
        print(f"Running {backend} on P2 plant-masked RGB frames...")
        report = run_learned_backend(workdir, backend, root, max_images=max_images,
                                     bundle_adjust=bundle_adjust, device=device, model_python=model_python,
                                     hf_token=hf_token, hf_home=hf_home,
                                     code_cache=code_cache,
                                     auto_fetch_code=auto_fetch_code, dry_run=dry_run,
                                     heartbeat_seconds=heartbeat_seconds)
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
    for name, entry in comparison["comparisons"].items():
        alignment = entry["camera_center_alignment"]
        detail = "not alignable" if alignment is None else f"aligned camera RMSE {alignment['rmse']:.4f}"
        print(f"  {name}: {entry['candidate_registered_frames']} registered, {detail}")
    return comparison


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["vggt", "mapanything"],
                        help="Models to run; COLMAP is always included as the comparison reference")
    parser.add_argument("--vggt-root", type=Path,
                        help="Optional existing VGGT checkout; fetched automatically when omitted")
    parser.add_argument("--mapanything-root", type=Path,
                        help="Optional existing MapAnything checkout; fetched automatically when omitted")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Uniformly sample this many P1 frames (0 = all); lower for VRAM-limited trials")
    parser.add_argument("--bundle-adjust", action="store_true",
                        help="Ask VGGT's official exporter for bundle adjustment (ignored by MapAnything)")
    parser.add_argument("--device", help="CUDA_VISIBLE_DEVICES value for the model subprocess, e.g. 0")
    parser.add_argument("--model-python",
                        help="Python executable in the VGGT/MapAnything environment; default is this command's Python")
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
    parser.add_argument("--dry-run", action="store_true", help="Stage inputs and record, but do not start models")
    args = parser.parse_args(argv)
    run(args.workdir, args.backends, vggt_root=args.vggt_root, mapanything_root=args.mapanything_root,
        max_images=args.max_images, bundle_adjust=args.bundle_adjust, device=args.device,
        model_python=args.model_python, hf_token=args.hf_token, hf_home=args.hf_home,
        code_cache=args.code_cache,
        auto_fetch_code=not args.no_auto_fetch_code, dry_run=args.dry_run,
        heartbeat_seconds=args.heartbeat_seconds)


if __name__ == "__main__":
    main()
