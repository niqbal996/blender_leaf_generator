"""P2 CLI: extract frames from a turntable video and segment the plant and
its holder in every frame, via SAM2 video propagation.

    pose-segment --video /path/DSC_0009.MOV --workdir runs/plant_9/ \\
        --checkpoint checkpoints/sam2.1_hiera_large.pt

Still photos work too -- one directory per capture pass, filenames in
capture order around the turntable:

    pose-segment --photos /path/plant_9_shots/ --workdir runs/plant_9/

Everything downstream reads p1/frames + p1/sources.json and cannot tell
which kind of capture produced them.

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

import cv2
from pathlib import Path
from typing import Optional, Tuple

from pose_estimator.frames import (
    check_capture_consistency,
    extract_sharpest_frames,
    ingest_photos,
)
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
    video_paths: Optional[list] = None,
    photo_dirs: Optional[list] = None,
    photo_max_edge: int = 1920,
    num_frames: int = 96,
    use_roi: bool = True,
    roi_padding: float = 0.45,
    plant_point: Optional[Tuple[int, int]] = None,
    holder_point: Optional[Tuple[int, int]] = None,
    root_point: Optional[Tuple[int, int]] = None,
    device: str = "cuda",
    reuse_frames: bool = False,
    prompts_file: Optional[Path] = None,
    prompt_bank: Optional[Path] = None,
    dino_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    dino_size: int = 896,
    prompt_points: int = 3,
    allow_mixed_capture: bool = False,
) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    frames_dir = workdir / "p1" / "frames"
    p2_dir = workdir / "p2"

    existing = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.is_dir() else []
    sources_file = workdir / "p1" / "sources.json"
    # frame stem -> which capture pass it came from. P3 needs this: each pass
    # has the backdrop in a different place, so they cannot share one
    # rotating-region mask, and two elevations trace two orbits not one.
    sources: dict = {}
    intrinsics_file = workdir / "p1" / "intrinsics.json"
    # frame stem -> EXIF focal length in pixels, for frames that carried one.
    # P3 groups by this so passes shot at different zooms get one camera each
    # instead of one averaged camera that fits none of them.
    intrinsics: dict = {}
    manifest_file = workdir / "p1" / "manifest.json"
    # One provenance record per ingested frame: where it came from, when it was
    # shot, on what. P1 is the last phase that can still tell two shoots apart.
    records_per_pass: list = []

    if reuse_frames and existing:
        print(f"Reusing {len(existing)} frames already in {frames_dir}")
        if sources_file.exists():
            sources = json.loads(sources_file.read_text())
    elif video_paths or photo_dirs:
        # One pass per source, whichever kind it is: a video is sampled down
        # to its sharpest frame per angular bin, a photo directory is already
        # one shot per angle and is taken as it stands. Everything after this
        # point sees only frame_XXXX.jpg + sources.json and cannot tell which
        # was which.
        # A previous ingest into this workdir may have written more frames
        # than this one will, and the surplus is a *different* capture: it
        # survives in p1/frames, gets no sources.json entry, and P3 then files
        # it under pass 0. Measured on sugarbeet_3, 39 orphans left by an
        # aborted run turned a 46-frame solve into an 85-frame one, broke the
        # pass-0 circle fit and collapsed the P4a hull to 0.23 IoU. Clearing
        # first keeps p1/frames describing this run and nothing else.
        stale = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.is_dir() else []
        if stale:
            print(f"  clearing {len(stale)} frame(s) left by an earlier ingest")
            for path in stale:
                path.unlink()

        groups = []
        for index, source in enumerate(list(video_paths or []) + list(photo_dirs or [])):
            start = sum(len(g) for g in groups)
            if source.is_dir():
                print(f"Pass {index}: still photos from {source.name}/...")
                ingested = ingest_photos(source, frames_dir, start_index=start,
                                         max_edge=photo_max_edge)
                written = ingested.frames
                intrinsics.update(ingested.focals)
                records_per_pass.append(ingested.records)
            else:
                print(f"Pass {index}: sharpest of {num_frames} angular bins from {source.name}...")
                written = extract_sharpest_frames(source, frames_dir, target_frame_count=num_frames,
                                                  start_index=start)
            print(f"  wrote {len(written)} frames (frame_{start:04d} onward)")
            groups.append(written)
            for path in written:
                sources[path.stem] = index
        sources_file.parent.mkdir(parents=True, exist_ok=True)
        sources_file.write_text(json.dumps(sources, indent=2))
        # Rewritten even when empty, so a stale file from an earlier ingest
        # into the same workdir cannot outlive the frames it described.
        intrinsics_file.write_text(json.dumps(intrinsics, indent=2))
        manifest_file.write_text(json.dumps(
            [r for pass_records in records_per_pass for r in pass_records], indent=2))

        # Written first, so the manifest is on disk to inspect when this raises.
        if len(records_per_pass) > 1 and not allow_mixed_capture:
            check_capture_consistency(records_per_pass)
    elif existing:
        print(f"No --video/--photos given; using the {len(existing)} frames already in {frames_dir}")
        if sources_file.exists():
            sources = json.loads(sources_file.read_text())
    else:
        raise FileNotFoundError(
            f"No --video or --photos given and no frames found in {frames_dir}")

    if not sources:
        sources = {p.stem: 0 for p in sorted(frames_dir.glob("frame_*.jpg"))}
        sources_file.parent.mkdir(parents=True, exist_ok=True)
        sources_file.write_text(json.dumps(sources, indent=2))

    # Clicked prompts, one set per pass, in full-frame coordinates. Found
    # automatically at the default path so a re-run after pose-pick-prompts
    # needs no extra argument.
    clicked: dict = {}
    default_prompts_file = p2_dir / "prompts_clicked.json"
    if prompts_file is None and default_prompts_file.exists():
        prompts_file = default_prompts_file
        print(f"found clicked prompts at {prompts_file} -- using them instead of colour")
    if prompts_file is not None:
        from pose_estimator.prompt_picker import load_prompts

        clicked = load_prompts(prompts_file)

    prompts = None
    if plant_point or holder_point or root_point:
        if clicked:
            raise ValueError(
                f"--plant-point/--holder-point/--root-point conflict with the clicked prompts in "
                f"{prompts_file}. Use one or the other.")
        prompts = Prompts(
            plant=[plant_point] if plant_point else [],
            holder=[holder_point] if holder_point else [],
            root=[root_point] if root_point else [],
        )
        if not prompts.plant:
            raise ValueError("--holder-point/--root-point given without --plant-point; "
                             "the plant prompt is required")

    # One SAM2 session per pass. Propagation carries temporal memory between
    # consecutive frames, so running it across a cut between two videos would
    # ask it to track through a discontinuity it has no reason to survive.
    all_frames = sorted(frames_dir.glob("frame_*.jpg"))
    per_pass = {}
    for path in all_frames:
        per_pass.setdefault(sources.get(path.stem, 0), []).append(path)

    combined_stats, crops = [], {}
    boxes_by_stem = {}
    if clicked:
        unknown = sorted(set(clicked) - set(per_pass))
        missing = sorted(set(per_pass) - set(clicked))
        if unknown or missing:
            raise ValueError(
                f"{prompts_file} does not match this workdir: it has "
                f"{sorted(clicked)} but the frames have passes {sorted(per_pass)}. "
                "Re-run pose-pick-prompts.")

    # A prompt bank locates the plant and the holder by appearance, so one set
    # of clicks serves every later capture -- pixel coordinates only ever
    # described one video. Done per pass, on the frame that pass is actually
    # seeded on: two elevations show the plant in different places, so one
    # located point cannot serve both.
    if prompt_bank is not None:
        from pose_estimator.banks import resolve_bank

        # Resolved even when it will be ignored below: a path that names the
        # wrong thing is worth hearing about now rather than on the next run
        # where nothing overrides it.
        prompt_bank = resolve_bank(prompt_bank, "--prompt-bank")
        if clicked:
            print(f"  ignoring {prompt_bank}: this workdir has its own clicked prompts")
        elif plant_point or holder_point or root_point:
            print(f"  ignoring {prompt_bank}: --plant-point/--holder-point/--root-point given")
        else:
            clicked = _locate_per_pass(prompt_bank, per_pass, dino_model, dino_size,
                                       device, prompt_points)

    # The bank that re-acquires a lost root object mid-sequence. Explicit
    # --prompt-bank wins; otherwise the bank pose-pick-prompts wrote next to
    # the clicked coordinates. Loaded lazily -- the DINO weights only load if
    # a root actually goes missing.
    reseed_bank = prompt_bank if prompt_bank is not None else (p2_dir / "prompt_bank.npz")
    reacquire = (_root_reacquirer(reseed_bank, dino_model, dino_size, device)
                 if Path(reseed_bank).exists() else None)

    for pass_index in sorted(per_pass):
        paths = per_pass[pass_index]
        print(f"Segmenting pass {pass_index} ({len(paths)} frames) with SAM2...")
        result = segment_sequence(
            frames_dir=frames_dir,
            out_dir=p2_dir,
            checkpoint=checkpoint,
            prompts=clicked.get(pass_index, prompts),
            use_roi=use_roi,
            roi_padding=roi_padding,
            device=device,
            frame_paths=paths,
            reacquire_root=reacquire,
        )
        combined_stats.extend(result["per_frame"])
        crops[str(pass_index)] = result["crop"]
        if result["crop"]:
            for path, box in zip(paths, result["crop"]["boxes"]):
                boxes_by_stem[path.stem] = box

    with open(p2_dir / "frame_stats.json", "w") as f:
        json.dump(combined_stats, f, indent=2)
    with open(p2_dir / "crops_per_pass.json", "w") as f:
        json.dump(crops, f, indent=2)

    # Each pass solves its own tracking crop, so the boxes must be re-indexed
    # into global frame order before the overlays can draw them.
    all_boxes = [boxes_by_stem[p.stem] for p in all_frames] if boxes_by_stem else None

    print("Scoring the segmentation...")
    report = run_qc(frames_dir, p2_dir, sources=sources)
    _check_prompts_are_covered(report, clicked or {}, prompts, per_pass, p2_dir)
    # run_qc has already written qc.json, so anything added afterwards has to
    # be written again or it exists only on the console -- which is where the
    # prompt-coverage check spent its first two runs.
    with open(p2_dir / "qc.json", "w") as f:
        json.dump(report, f, indent=2)

    write_overlays(frames_dir, p2_dir, crop_boxes=all_boxes)
    write_area_plot(p2_dir, report)

    print(f"\n  P2 checks ({'ALL PASSED' if report['all_passed'] else 'FAILURES PRESENT'}):")
    for name, check in report["checks"].items():
        print(f"    [{'PASS' if check['pass'] else 'FAIL'}] {name}: {check['detail']}")
    print(f"\n  median plant mask area: {report['plant_area_px']['median']:.0f} px")
    print(f"  artifacts + diagnostics in {p2_dir}")

    return report



def _check_prompts_are_covered(report, clicked, fallback, per_pass, p2_dir) -> None:
    """Did the mask actually grow to cover every point we prompted with?

    None of the other checks can see undersegmentation. `plant_mask_free_of
    _holder` only measures contamination *by the tool*, so a mask holding one
    leaf of six passes it cleanly -- which is exactly what happened on
    thistle2 pass 0, and it was read as success. A prompt that ends up outside
    the very mask it seeded is direct evidence SAM2 kept only part of the
    object.
    """
    outside, total = [], 0
    for pass_index, paths in sorted(per_pass.items()):
        prompts = clicked.get(pass_index, fallback)
        if prompts is None or not getattr(prompts, "plant", None):
            continue
        if getattr(prompts, "space", "crop") != "full_frame":
            continue           # crop-space points cannot be checked in full-frame masks
        mask_path = p2_dir / "masks" / "plant" / f"{paths[0].stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        height, width = mask.shape
        # Root prompts are checked against the same plant mask: the tracked
        # root object is unioned into it, so a root prompt outside the plant
        # mask means the root object failed to attach.
        for x, y in list(prompts.plant) + list(getattr(prompts, "root", [])):
            total += 1
            if not (0 <= x < width and 0 <= y < height and mask[y, x] > 127):
                outside.append((pass_index, int(x), int(y)))
    if not total:
        return
    report["checks"]["mask_covers_its_own_prompts"] = {
        "pass": not outside,
        "detail": (f"{len(outside)} of {total} plant prompt(s) fall outside the mask they "
                   f"seeded{': ' + str(outside) if outside else ''} -- a prompt outside its "
                   "own mask means SAM2 kept only part of the plant"),
    }
    report["all_passed"] = all(c["pass"] for c in report["checks"].values())


def _root_reacquirer(prompt_bank, dino_model, dino_size, device):
    """A lazy `(bgr, crop_box) -> (x, y, margin) or None` root locator.

    Everything heavy -- the bank file, the DINO weights -- loads on the first
    call, which only happens when the root object actually loses a stretch of
    frames. A bank without root examples, or DINO weights that cannot load
    (gated repo, no token), degrade to "no re-acquisition" with a printed
    note rather than failing the run: the segmentation itself never needed
    DINO.
    """
    holder = {}

    def locate(bgr, box):
        if "fn" not in holder:
            holder["fn"] = None
            try:
                from pose_estimator.dino import DinoBackbone
                from pose_estimator.prompt_seeds import best_patch, load_prompt_bank

                vectors, labels = load_prompt_bank(prompt_bank, dino_model)
                if "root" not in labels:
                    print(f"  {prompt_bank} has no root examples -- a lost root cannot "
                          "be re-acquired. Re-click with pose-pick-prompts ([3] = root).")
                else:
                    backbone = DinoBackbone(dino_model, device=device, size=dino_size)
                    holder["fn"] = lambda b, bx: best_patch(
                        backbone, b, vectors, labels, "root", within=bx)
            except (Exception, SystemExit) as exc:
                print(f"  root re-acquisition unavailable ({exc}) -- continuing without it")
        return holder["fn"](bgr, box) if holder["fn"] else None

    return locate


def _locate_per_pass(prompt_bank, per_pass, dino_model, dino_size, device,
                     prompt_points: int = 3):
    """Plant/holder prompts per capture pass, found by appearance.

    Several plant points, not one: a single prompt on a small plant seeds one
    leaf and SAM2 tracks only that leaf.
    """
    from pose_estimator.dino import DinoBackbone
    from pose_estimator.prompt_seeds import load_prompt_bank, locate_prompts

    vectors, labels = load_prompt_bank(prompt_bank, dino_model)
    backbone = DinoBackbone(dino_model, device=device, size=dino_size)

    print(f"  plant/holder prompts located by appearance from {prompt_bank}:")
    located_by_pass = {}
    for pass_index in sorted(per_pass):
        first = per_pass[pass_index][0]
        found = locate_prompts(backbone, cv2.imread(str(first)), vectors, labels,
                               count=prompt_points)
        if not found.get("plant"):
            raise ValueError(
                f"{prompt_bank} has no 'plant' examples, so there is nothing to seed "
                "SAM2 with. Re-click it with pose-pick-prompts.")
        located_by_pass[pass_index] = Prompts(
            plant=list(found["plant"]),
            holder=list(found.get("holder", [])),
            # Root examples in the bank place a root prompt the same way. A
            # bank whose plant examples are all foliage never seeds the root
            # -- measured on thistle2, all 3 located prompts landed at
            # y 472-549, none below the jaws, and the root left every phase.
            root=list(found.get("root", [])),
            # locate_prompts reads the FULL frame, so these are full-frame
            # pixels. Leaving the default ("crop") is not a labelling detail:
            # the coordinates get re-read as crop-relative and shifted by the
            # crop origin, and -- worse -- the tracking crop then falls back to
            # the colour prepass, which is the thing the bank exists to
            # replace. Measured on thistle2: a correct prompt at (771, 607) on
            # the plant became (1563, 607) on the plier handle.
            space="full_frame",
        )
        detail = "  ".join(f"{k}={len(v)}x {v}" for k, v in found.items())
        print(f"    pass {pass_index} ({first.stem})   {detail}")
    return located_by_pass


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path, help="Run directory for this specimen")
    parser.add_argument("--video", type=Path, nargs="+",
                        help="Turntable video(s). Give several to merge capture passes at "
                             "different elevations into one workdir; each is tracked separately "
                             "and they are solved together in P3. Omit to reuse existing frames.")
    parser.add_argument("--photos", type=Path, nargs="+", metavar="DIR",
                        help="Directory of still photos instead of (or alongside) --video, one "
                             "directory per capture pass. Filename order must be capture order "
                             "around the turntable. Photos are used as they are -- there is no "
                             "redundancy to pick a sharpest frame from, so cull hopeless shots "
                             "yourself; each one's sharpness is printed on ingest.")
    parser.add_argument("--photo-max-edge", type=int, default=1920,
                        help="Resize ingested photos so the long edge is at most this many "
                             "pixels (0 = keep original). The default matches the video path, "
                             "which every downstream default was fitted against; a 24MP frame "
                             "is 11x the pixels through P4a/P4b for detail that SAM2 (1024px) "
                             "and P3 (--max-image-size) discard anyway.")
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
    parser.add_argument(
        "--root-point",
        type=str,
        help="Seed the exposed root as its own tracked SAM2 object, as X,Y (same pixel space "
        "as --plant-point). The root's mask is unioned into the plant mask; without a root "
        "seed a root cut off by the jaws drops out of the mask and everything downstream.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Clicked SAM2 seeds from pose-pick-prompts, one set per capture pass, in "
        "full-frame coordinates. Defaults to <workdir>/p2/prompts_clicked.json when that "
        "exists. Unlike --plant-point this works on multi-pass captures, where a single "
        "point cannot serve two passes with different first frames.",
    )
    parser.add_argument("--prompt-bank", type=Path,
                        help="p2/prompt_bank.npz from an earlier specimen's pose-pick-prompts. "
                             "Locates the plant and the holder by appearance rather than "
                             "by pixel coordinate, so one bank works across videos and "
                             "poses. Use this whenever the pliers get tracked as the "
                             "plant. Ignored if this workdir has its own clicked prompts.")
    parser.add_argument("--dino-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                        help="model for --prompt-bank; must match the one that built it")
    parser.add_argument("--prompt-points", type=int, default=3,
                        help="Plant prompts to place per pass from --prompt-bank. One is "
                             "enough only when the plant fills the frame; on a small or "
                             "distant plant a single prompt seeds one leaf and SAM2 "
                             "tracks just that leaf.")
    parser.add_argument(
        "--allow-mixed-capture", action="store_true",
        help="Skip the check that the --photos/--video passes are one shoot of one "
             "plant (same camera body, consecutive in time, filenames in capture "
             "order). Only for a capture you know is right but whose EXIF says "
             "otherwise -- the check exists because a foreign pass is invisible "
             "after P1 and wrecks P3 without failing any phase.")
    parser.add_argument("--dino-size", type=int, default=896)
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
        video_paths=args.video,
        photo_dirs=args.photos,
        photo_max_edge=args.photo_max_edge,
        num_frames=args.num_frames,
        use_roi=not args.no_roi,
        roi_padding=args.roi_padding,
        plant_point=_parse_point(args.plant_point),
        holder_point=_parse_point(args.holder_point),
        root_point=_parse_point(args.root_point),
        device=args.device,
        reuse_frames=args.reuse_frames,
        prompts_file=args.prompts_file,
        prompt_bank=args.prompt_bank,
        dino_model=args.dino_model,
        dino_size=args.dino_size,
        prompt_points=args.prompt_points,
        allow_mixed_capture=args.allow_mixed_capture,
    )


if __name__ == "__main__":
    main()
