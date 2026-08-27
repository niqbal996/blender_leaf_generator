"""Click the SAM2 plant/holder seeds for P2, one set per capture pass.

    pose-pick-prompts --workdir runs/plant_9/

Opens the first frame of each capture pass -- the frame SAM2 is actually
seeded on. Click the plant, press 2 and click the holder, press 3 and click
the exposed root, press n for the next pass, press s to save.

[3] root is a *tracking* category, not a third output class: the root is part
of the plant, and P2's masks stay plant/holder. It gets its own key because
the jaws cut the root into a disconnected blob, and a blob sharing the
foliage object's SAM2 memory drops out of the mask intermittently -- measured
on thistle1, root points clicked as plant held the root in only 51-77% of
frames, under the ~86% silhouette agreement P4a needs, so the carve deleted
it anyway. Seeded as its own SAM2 object the root is one connected region
with its own memory, and its mask is unioned into the plant mask at write
time. Which parts of the plant are leaf, stem or root is a separate question
that pose-pick-seeds answers at P4c. Writes <workdir>/p2/prompts_clicked.json, which
pose-segment picks up automatically:

    pose-segment --workdir runs/plant_9/ --reuse-frames

It also writes <workdir>/p2/prompt_bank.npz, and that one is for the *next*
video. The clicked coordinates only mean something in this capture; the bank
stores what the plant and the holder look like, so a later specimen shot at a
slightly different pose can be searched for whatever most resembles them:

    pose-segment --workdir runs/plant_10/ --prompt-bank runs/plant_9/p2/prompt_bank.npz

Click a few passes' worth into one bank and it covers a batch. Building it
needs the DINO weights (--dino-model, --hf-token); --no-bank skips it.

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
     "passes": {"0": {"frame": "frame_0000", "plant": [[952, 470]],
                      "holder": [[1541, 840]], "root": [[970, 905]]}}}

Coordinates are full-frame pixels (1920x1080 here), NOT the cropped frame.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np

from pose_estimator.prompt_picker import capture_passes, frames_per_pass, pick, save_prompts
from pose_estimator.seed_picker import display_available


def run(
    workdir: Path,
    out: Optional[Path] = None,
    max_display: int = 1400,
    show_auto: bool = True,
    bank: bool = True,
    bank_out: Optional[Path] = None,
    dino_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    dino_size: int = 896,
    device: str = "cuda",
    hf_token: Optional[str] = None,
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
    print("  click = add point    1 = plant    2 = holder    3 = root    u = undo    c = clear")
    print("  n / p = next / previous pass       s = save and quit    q = quit")
    print("  Faint grey crosses are what the colour heuristic picked -- if they are on")
    print("  the holder, that is the bug you are fixing.")
    print("  Every pass needs at least one plant point; the holder is optional -- but")
    print("  click the JAWS where they grip the stem, not the far end of the handle:")
    print("  the tracking crop is sized to the plant, and a point outside it is dropped.")
    print("  For the reusable bank, place THREE OR MORE plant points per pass, on")
    print("  different parts of the plant. One example transfers to a new video badly")
    print("  (1 frame in 12 in a measured test); three per frame got 11 in 12.")
    print("  ROOTS: click the exposed root with [3] root, once per pass. The jaws cut")
    print("  it into a separate blob, and as extra plant points it flickered out of the")
    print("  mask in a quarter to half of the frames -- [3] gives it its own SAM2 object,")
    print("  and P2 folds that mask back into the plant mask on write. Without a root")
    print("  point the root is absent from the mask, the hull, the cloud, and every")
    print("  phase after; P4c cannot put it back.")
    print("  leaf/stem/root is a different question, answered later by pose-pick-seeds.\n")

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
        if not counts.get("root"):
            print("      no root point -- if this specimen has an exposed root below the jaws,"
                  " it will be missing from the mask and everything downstream ([3] = root)")
    if bank:
        _write_bank(workdir, session, bank_out, dino_model, dino_size, device, hf_token)
    print(f"\n  next:  pose-segment --workdir {workdir} --reuse-frames")
    return out


def _write_bank(workdir, session, bank_out, dino_model, dino_size, device, hf_token):
    """Store the clicks as feature vectors, which is what makes them portable.

    Same trick P4c uses for organ classes. The pixel coordinates already saved
    describe this video only; a vector describes what a plier looks like, so
    the next capture can be searched for the patch that most resembles it
    however the rig was posed.
    """
    import cv2

    from pose_estimator.dino import DinoBackbone
    from pose_estimator.prompt_seeds import build_prompt_bank, locate_prompts, save_prompt_bank

    try:
        backbone = DinoBackbone(dino_model, device=device, size=dino_size, token=hf_token)
    except (Exception, SystemExit) as exc:
        # SystemExit and not just Exception: DinoBackbone raises SystemExit for
        # a gated repo, and the clicked coordinates are already saved by now --
        # losing the bank must not look like losing the clicks.
        print(f"\n  no reusable prompt bank written: {exc}")
        print("  The clicked coordinates above are saved and this video will segment fine.")
        print("  Re-run with --dino-model facebook/dinov2-base (ungated) to get a bank.")
        return None

    vectors, labels = [], []
    for spec in session.passes:
        bgr = cv2.imread(str(spec.path))
        clicks = [(name, x, y)
                  for name, points in session.points[spec.index].items()
                  for (x, y) in points]
        if not clicks:
            continue
        v, l = build_prompt_bank(backbone, bgr, clicks)
        vectors.append(v)
        labels.extend(l)
    if not labels:
        return None

    vectors = np.concatenate(vectors, axis=1)
    bank_out = bank_out or (workdir / "p2" / "prompt_bank.npz")
    save_prompt_bank(bank_out, vectors, labels, dino_model, dino_size)
    print(f"\n  reusable prompt bank -> {bank_out}")
    for name in dict.fromkeys(labels):
        print(f"    {name}: {sum(1 for l in labels if l == name)} example(s)")

    # Where it would place the prompts on the last pass's first frame. Only a
    # sanity check, but it costs one forward pass and catches a bank built
    # from clicks that all landed on background.
    located = locate_prompts(backbone, cv2.imread(str(session.passes[-1].path)), vectors, labels)
    print("    self-check on " + session.passes[-1].frame + ": "
          + ", ".join(f"{k} at {v}" for k, v in located.items() if v))
    print(f"  use it on the next specimen:  "
          f"pose-segment --workdir <new> --prompt-bank {bank_out}")
    return bank_out


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
    parser.add_argument("--no-bank", action="store_true",
                        help="Skip the reusable prompt bank (needs the DINO weights). "
                             "The clicked coordinates are still written.")
    parser.add_argument("--bank-out", type=Path, help="default <workdir>/p2/prompt_bank.npz")
    parser.add_argument("--dino-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                        help="Backbone the bank is built with. --prompt-bank later must "
                             "use the same one; the vectors mean nothing across models.")
    parser.add_argument("--dino-size", type=int, default=896)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="DINOv3 is gated on HuggingFace; or export HF_TOKEN")
    args = parser.parse_args(argv)

    run(workdir=args.workdir, out=args.out, max_display=args.max_display,
        show_auto=not args.no_auto, bank=not args.no_bank, bank_out=args.bank_out,
        dino_model=args.dino_model, dino_size=args.dino_size, device=args.device,
        hf_token=args.hf_token)


if __name__ == "__main__":
    main()
