"""Click organ seeds on a frame, instead of typing grid coordinates.

    pose-pick-seeds --workdir runs/plant_9/

Opens a window on the plant, cropped exactly as the classifier will crop it.
Click to place a seed, press 1-9 to switch class, n/p to change frame, s to
save. Writes <workdir>/p4c/seeds.json, which pose-classify reads directly:

    pose-classify --workdir runs/plant_9/ --seeds-file runs/plant_9/p4c/seeds.json

Seeding from several frames is worth doing. A leaf photographed from one
angle is one example of "leaf"; from three angles it is three, and the seed
vectors are kept individually rather than averaged, so extra examples widen
the class instead of blurring it. Use n/p to move between frames and keep
clicking -- seeds from every frame you visit go into the same file.

Needs a display. On Windows 11 WSLg provides one and this just works; on an
older WSL without it, fall back to the printed coordinate grid:

    python scripts/dinov3_organ_lab.py --mode reference \\
        --images <workdir>/p1/frames --plant-mask-dir <workdir>/p2/masks/plant \\
        --frames 57 --out /tmp/ref
"""

import argparse
from pathlib import Path
from typing import Optional

from pose_estimator.seed_picker import display_available, pick, save_seeds


def run(
    workdir: Path,
    classes: Optional[list] = None,
    frame: int = 0,
    out: Optional[Path] = None,
    pad: int = 60,
    max_display: int = 1100,
) -> Optional[Path]:
    frames_dir = workdir / "p1" / "frames"
    mask_dir = workdir / "p2" / "masks" / "plant"
    if not frames_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(
            f"need {frames_dir} and {mask_dir} -- run pose-segment on this workdir first")

    ok, why = display_available()
    if not ok:
        raise SystemExit(
            f"OpenCV cannot open a window here ({why}).\n"
            "On Windows 11, WSLg should provide a display -- check that $DISPLAY is set.\n"
            "Otherwise use the printed coordinate grid instead:\n"
            "  python scripts/dinov3_organ_lab.py --mode reference \\\n"
            f"      --images {frames_dir} --plant-mask-dir {mask_dir} \\\n"
            "      --frames 57 --out /tmp/ref")

    classes = list(classes or ["leaf", "stem", "root"])
    print(f"  classes: {', '.join(f'[{i + 1}] {c}' for i, c in enumerate(classes))}")
    print("  click = add seed    1-9 = class    u = undo    c = clear")
    print("  n / p = next / previous frame      s = save and quit    q = quit")
    print("  Clicks off the plant are refused: the classifier blanks that area before")
    print("  reading features, so a seed there would describe nothing.\n")

    session = pick(frames_dir, mask_dir, classes, start_index=frame,
                   pad=pad, max_display=max_display)
    if session is None:
        print("  quit without saving")
        return None
    if not session.seeds:
        print("  no seeds placed, nothing written")
        return None

    out = out or (workdir / "p4c" / "seeds.json")
    save_seeds(out, session, pad=pad)

    counts = session.counts()
    frames_used = sorted({s.frame for s in session.seeds})
    print(f"\n  {len(session.seeds)} seed(s) over {len(frames_used)} frame(s) -> {out}")
    for name, n in counts.items():
        print(f"    {name:<12} {n}")
    missing = session.missing_classes()
    if missing:
        print(f"  NOTE: no seeds for {', '.join(missing)} -- "
              "those classes cannot be predicted and are dropped.")
    if len(frames_used) == 1:
        print(f"\n  equivalent to:  --seed-frame {session.seeds[0].frame_index} "
              f"--seeds {session.as_cli_string()}")
    print(f"\n  next:  pose-classify --workdir {workdir} --seeds-file {out}")
    return out


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path,
                        help="Specimen run directory (needs p1/frames and p2/masks/plant)")
    parser.add_argument("--classes", nargs="+", default=["leaf", "stem", "root"],
                        help="Organ classes to label, in key order 1..9")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to open first; n/p move from there")
    parser.add_argument("--out", type=Path, help="default <workdir>/p4c/seeds.json")
    parser.add_argument("--pad", type=int, default=60,
                        help="Crop padding. Must match what the classifier uses (60).")
    parser.add_argument("--max-display", type=int, default=1100,
                        help="Longest window edge in pixels; clicks are mapped back to "
                             "full crop coordinates automatically")
    args = parser.parse_args(argv)

    run(workdir=args.workdir, classes=args.classes, frame=args.frame,
        out=args.out, pad=args.pad, max_display=args.max_display)


if __name__ == "__main__":
    main()
