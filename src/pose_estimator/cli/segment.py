"""P2 CLI: extract frames from a turntable video and segment the plant and
its holder in every frame, via SAM2 video propagation.

    pose-segment --video /path/DSC_0009.MOV --workdir runs/plant_9/ \\
        --checkpoint checkpoints/sam2.1_hiera_large.pt

Writes into <workdir>:
    p1/frames/frame_XXXX.jpg     sharpest frame per angular bin
    p2/masks/plant/*.png         binary silhouettes, full-frame coords
    p2/masks/holder/*.png        binary holder masks (pliers/clamp/pot)
    p2/alpha/*.png               soft plant matte
    p2/qc.json                   acceptance checks -- read this before trusting the run
    p2/diag/                     overlays + mask-area plot

The masks are the input to the P4 visual hull, which is what replaces the
old per-plant color/density thresholds. Their quality caps everything
downstream, so the QC report is not decoration: a run that fails
`agrees_with_luminance_matte` or `area_temporally_smooth` should be fixed
here rather than compensated for later.

Requires the "segment" extra plus a SAM2 checkpoint -- see the README.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

from pose_estimator.frames import extract_sharpest_frames
from pose_estimator.segmentation import Prompts, segment_sequence
from pose_estimator.segmentation_qc import run_qc, write_area_plot, write_overlays


def _parse_point(text: Optional[str]) -> Optional[Tuple[int, int]]:
    if text is None:
        return None
    x, y = text.split(",")
    return (int(x), int(y))


def run(
    workdir: Path,
    checkpoint: Path,
    video_path: Optional[Path] = None,
    num_frames: int = 96,
    use_roi: bool = True,
    roi_padding: float = 0.45,
    plant_point: Optional[Tuple[int, int]] = None,
    holder_point: Optional[Tuple[int, int]] = None,
    device: str = "cuda",
    reuse_frames: bool = False,
) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    frames_dir = workdir / "p1" / "frames"
    p2_dir = workdir / "p2"

    existing = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.is_dir() else []
    if reuse_frames and existing:
        print(f"Reusing {len(existing)} frames already in {frames_dir}")
    elif video_path is not None:
        print(f"Extracting the sharpest of {num_frames} angular bins from {video_path.name}...")
        written = extract_sharpest_frames(video_path, frames_dir, target_frame_count=num_frames)
        print(f"  wrote {len(written)} frames to {frames_dir}")
    elif existing:
        print(f"No --video given; using the {len(existing)} frames already in {frames_dir}")
    else:
        raise FileNotFoundError(f"No --video given and no frames found in {frames_dir}")

    prompts = None
    if plant_point or holder_point:
        prompts = Prompts(
            plant=[plant_point] if plant_point else [],
            holder=[holder_point] if holder_point else [],
        )
        if not prompts.plant:
            raise ValueError("--holder-point given without --plant-point; the plant prompt is required")

    print("Segmenting with SAM2 video propagation...")
    result = segment_sequence(
        frames_dir=frames_dir,
        out_dir=p2_dir,
        checkpoint=checkpoint,
        prompts=prompts,
        use_roi=use_roi,
        roi_padding=roi_padding,
        device=device,
    )

    with open(p2_dir / "frame_stats.json", "w") as f:
        json.dump(result["per_frame"], f, indent=2)

    print("Scoring the segmentation...")
    report = run_qc(frames_dir, p2_dir)

    write_overlays(frames_dir, p2_dir, crop_boxes=result["crop"]["boxes"] if result["crop"] else None)
    write_area_plot(p2_dir, report)

    print(f"\n  P2 checks ({'ALL PASSED' if report['all_passed'] else 'FAILURES PRESENT'}):")
    for name, check in report["checks"].items():
        print(f"    [{'PASS' if check['pass'] else 'FAIL'}] {name}: {check['detail']}")
    print(f"\n  median plant mask area: {report['plant_area_px']['median']:.0f} px")
    print(f"  artifacts + diagnostics in {p2_dir}")

    return report


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path, help="Run directory for this specimen")
    parser.add_argument("--video", type=Path, help="Turntable video. Omit to reuse <workdir>/p1/frames/")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/sam2.1_hiera_large.pt"),
        help="SAM2 checkpoint. The config is inferred from the filename.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=96,
        help="Angular bins to split the video into; the sharpest frame of each is kept.",
    )
    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Segment full frames instead of tracking the plant with a moving crop. SAM2 "
        "resizes its input to 1024x1024 regardless, so on a wide shot of a small seedling this "
        "spends most of the resolution on backdrop and loses thin petioles -- only use it if "
        "the color prepass is tracking the wrong object.",
    )
    parser.add_argument(
        "--roi-padding",
        type=float,
        default=0.45,
        help="Pad the tracking crop by this fraction of the plant's size on each side. Generous "
        "by default so the crop also contains the clamp jaws gripping the stem.",
    )
    parser.add_argument(
        "--plant-point",
        type=str,
        help="Override the auto-derived plant prompt, as X,Y in the first frame's crop (or "
        "full-frame coordinates with --no-roi). Check p2/diag/ overlays to see where the auto "
        "seed landed.",
    )
    parser.add_argument("--holder-point", type=str, help="Override the auto-derived holder prompt, as X,Y")
    parser.add_argument("--device", default="cuda", help="torch device (cuda or cpu)")
    parser.add_argument(
        "--reuse-frames",
        action="store_true",
        help="Skip frame extraction and reuse <workdir>/p1/frames/ -- use when re-running "
        "segmentation with different prompts on a capture already extracted.",
    )
    args = parser.parse_args(argv)

    run(
        workdir=args.workdir,
        checkpoint=args.checkpoint,
        video_path=args.video,
        num_frames=args.num_frames,
        use_roi=not args.no_roi,
        roi_padding=args.roi_padding,
        plant_point=_parse_point(args.plant_point),
        holder_point=_parse_point(args.holder_point),
        device=args.device,
        reuse_frames=args.reuse_frames,
    )


if __name__ == "__main__":
    main()
