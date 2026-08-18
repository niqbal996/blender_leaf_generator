"""Click the SAM2 plant/holder seeds for P2, one set per capture pass.

    pose-pick-prompts --workdir runs/plant_9/

Opens the first frame of each capture pass -- the frame SAM2 is actually
seeded on. Click the plant, press 2 and click the holder, press n for the
next pass, press s to save. Writes <workdir>/p2/prompts_clicked.json, which
pose-segment picks up automatically:

    pose-segment --workdir runs/plant_9/ --reuse-frames

Faint grey crosses show where the colour heuristic *would* have seeded. When
those are sitting on the holder, that is the run you are here to fix.

Why per pass: each pass is a separate SAM2 propagation with its own first
frame, so one point cannot serve both. Why the first frame only: propagation
is seeded at frame 0 of the pass and carries memory forward, so a point
clicked anywhere else has nothing to attach to.

Needs a display. On Windows 11 WSLg provides one and this just works. Without
one, read the coordinates off a frame in any image viewer and write the file
by hand -- it is three lines per pass:

    {"version": 1, "space": "full_frame",
     "passes": {"0": {"frame": "frame_0000", "plant": [[952, 470]], "holder": [[1541, 840]]}}}

Coordinates are full-frame pixels (1920x1080 here), NOT the cropped frame.
"""

import argparse
from pathlib import Path
from typing import Optional

from pose_estimator.prompt_picker import capture_passes, frames_per_pass, pick, save_prompts
from pose_estimator.seed_picker import display_available


def run(
    workdir: Path,
    out: Optional[Path] = None,
    max_display: int = 1400,
    show_auto: bool = True,
) -> Optional[Path]:
    passes = capture_passes(workdir)

    ok, why = display_available()
    if not ok:
        raise SystemExit(
            f"OpenCV cannot open a window here ({why}).\n"
            "On Windows 11, WSLg should provide a display -- check that $DISPLAY is set.\n"
            f"Otherwise write {workdir / 'p2' / 'prompts_clicked.json'} by hand; see\n"
            "  pose-pick-prompts --help  for the format.")

    print(f"  {len(passes)} capture pass(es): "
          + ", ".join(f"pass {p.index} -> {p.frame}" for p in passes))
    print("  click = add point    1 = plant    2 = holder    u = undo    c = clear")
    print("  n / p = next / previous pass       s = save and quit    q = quit")
    print("  Faint grey crosses are what the colour heuristic picked -- if they are on")
    print("  the holder, that is the bug you are fixing.")
    print("  Every pass needs at least one plant point; the holder is optional -- but")
    print("  click the JAWS where they grip the stem, not the far end of the handle:")
    print("  the tracking crop is sized to the plant, and a point outside it is dropped.\n")

    session = pick(passes, max_display=max_display, show_auto=show_auto,
                   pass_frames=frames_per_pass(workdir))
    if session is None:
        print("  quit without saving")
        return None

    out = out or (workdir / "p2" / "prompts_clicked.json")
    save_prompts(out, session)

    print(f"\n  prompts for {len(passes)} pass(es) -> {out}")
    for spec in session.passes:
        counts = session.counts(spec.index)
        detail = "  ".join(f"{name}={n}" for name, n in counts.items())
        print(f"    pass {spec.index} ({spec.frame})   {detail}")
        if not counts["holder"]:
            print("      no holder point -- P2 will not track the holder separately for this pass,"
                  " so it cannot subtract it from the plant mask")
    print(f"\n  next:  pose-segment --workdir {workdir} --reuse-frames")
    return out


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path,
                        help="Specimen run directory (needs p1/frames)")
    parser.add_argument("--out", type=Path, help="default <workdir>/p2/prompts_clicked.json")
    parser.add_argument("--max-display", type=int, default=1400,
                        help="Longest window edge in pixels; clicks are mapped back to "
                             "full-frame coordinates automatically")
    parser.add_argument("--no-auto", action="store_true",
                        help="Do not draw the colour heuristic's own seed points")
    args = parser.parse_args(argv)

    run(workdir=args.workdir, out=args.out, max_display=args.max_display,
        show_auto=not args.no_auto)


if __name__ == "__main__":
    main()
