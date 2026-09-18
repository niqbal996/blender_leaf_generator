#!/usr/bin/env python
"""Can SAM3's text prompts produce P2's three classes, without the clicker?

    # the measurement, against a P2 run that already exists
    python scripts/sam3_p2_prompt_lab.py \
        --images <workdir>/p1/frames --p2 <workdir>/p2 \
        --frames 0-14 --out /tmp/sam3_p2_lab

    # try other wording -- that is the whole experiment
    python scripts/sam3_p2_prompt_lab.py --images ... --p2 ... \
        --group plant "leaf" "stem" --group holder "pliers" "metal clamp" "pot" \
        --out /tmp/sam3_p2_lab2

    # no P2 at all -- SAM3 finds its own crop, on a bare folder of frames
    python scripts/sam3_p2_prompt_lab.py --images /any/folder/of/frames \
        --group plant "leaf" "stem" --out /tmp/sam3_bare

    # also check the localisation pass the real backend would need
    python scripts/sam3_p2_prompt_lab.py --images ... --p2 ... --localize

P2 today seeds SAM2 with clicked points and tracks three objects: `plant`,
`holder` (the pliers/clamp/pot) and `root`. Placing those clicks is the step
that fails -- on sugarbeet_4 it lost the root in 85 of 85 frames and left 3 of
12 plant prompts outside the mask they seeded. SAM3 asks for a noun phrase
instead of a coordinate, which would remove the clicker entirely.

`sam3_leaf_track.py` already showed SAM3 finds and tracks leaves well. It did
not answer the question that decides whether P2 can be rebuilt on it, which is
about the *other two* classes: `masks/holder` is read by P4a, P4b, P3 and P5,
and `masks/root` is what P4c uses to place the root class. A SAM3 P2 that
produces a beautiful plant mask and no holder mask is not a P2.

`--p2` is optional, and only turns scoring on. Without it SAM3 locates its own
crop on a subsample of full frames (`--crop-from sam3`, the default in that
case) and the run reports what each phrase found with no reference to compare
it to -- which is all a bare folder of frames can support, and enough to judge
wording by eye from the overlays.

With `--p2`, this script scores wording against the P2 masks already on disk:

  * `iou_plant` / `iou_holder` -- agreement with that reference class.
  * `inside_plant` -- what fraction of the phrase's own pixels are plant. A
    phrase meant for the holder wants this near 0; a phrase meant for the
    plant wants it near 1. This is the number that catches a holder prompt
    quietly masking the stem base.
  * `covers_plant` -- what fraction of the reference plant this phrase found.
  * `frames` / `instances` -- how often the phrase fired at all. A phrase that
    scores well on the two frames it appears in has not replaced anything.

The reference is a *previous SAM2 run*, not ground truth, and on this capture
it is a run that failed its own QC. Read the numbers accordingly: a high
`iou_plant` means "agrees with what P2 does today", and the overlay panels --
photo | SAM3 by phrase | the P2 reference -- are what say whether disagreeing
was SAM3 being right.

Each `--group` is its own SAM3 session, on purpose. One session assigns an
instance to a single prompt, so asking about "plant" and "leaf" together
measures which word won a competition rather than what either word finds, and
every extra concept costs GPU memory on a sequence this long.

Writes into --out:
    frame_XXXX.jpg     photo | SAM3, one colour per phrase | P2 reference
    lab.json           every metric below, per phrase and per combination
    localize.json      the full-frame localisation check (--localize)
    run.log
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sam3_leaf_track import crop_from_masks, load_cropped_video  # noqa: E402
from text_organ_lab import Tee  # noqa: E402

# The wording actually worth trying, grouped by the P2 class it is a candidate
# for. `plant` is the one already known to work; the run this script exists to
# judge is the other two.
DEFAULT_GROUPS = [
    ("plant", ["leaf", "stem", "plant"]),
    ("holder", ["pliers", "metal clamp", "plant pot"]),
    ("root", ["root", "taproot"]),
]
# Phrase unions worth scoring as a whole, since P2 wants one mask per class
# and not one per word. Any phrase not present in a run is skipped.
DEFAULT_COMBOS = ["leaf+stem", "leaf+stem+root", "plant"]

# BGR, one per phrase in the order the group lists them.
PHRASE_COLORS = [
    (60, 60, 220), (90, 210, 100), (230, 140, 60), (200, 60, 200),
    (60, 240, 240), (230, 80, 120), (180, 130, 240), (200, 200, 80),
]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def fraction_of(part: np.ndarray, whole: np.ndarray) -> float:
    """|part & whole| / |part| -- how much of `part` lands inside `whole`."""
    total = int(part.sum())
    return float((part & whole).sum()) / total if total else 0.0


def load_reference(p2_dir: Optional[Path], stem: str, box, shape) -> Dict[str, np.ndarray]:
    """P2's own masks for one frame, cropped to the window SAM3 saw.

    A class P2 never wrote -- `root` on a run whose clicks lost it -- comes
    back as an empty mask rather than missing, so the metrics stay comparable
    across frames and the zero is visible in the table instead of absent.

    With no `p2_dir` at all every class is empty and the IoU columns are
    meaningless; `--p2` is what turns scoring on, and the caller drops those
    columns from the table when it is absent.
    """
    height, width = shape
    out = {}
    for name in ("plant", "holder", "root"):
        path = (p2_dir / "masks" / name / f"{stem}.png") if p2_dir else None
        full = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if (path and path.exists()) else None
        if full is None:
            out[name] = np.zeros((height, width), bool)
            continue
        x0, y0, x1, y1 = box
        out[name] = (full[y0:y1, x0:x1] > 127)
    return out


def crop_from_sam3(processor, model, torch, frame_paths, phrase: str, stride: int,
                   padding: float, device: str):
    """A plant-following crop solved by SAM3 itself, with no P2 in the loop.

    This is what makes the lab runnable on a bare folder of frames, and it is
    also the piece a real P2 backend needs: inside P2 there are no masks to
    crop from, and the colour prepass that would otherwise supply the window
    is the thing that fails when a pot or a pair of pliers is in shot.

    Cheap on purpose. It runs on every `stride`th frame at full resolution --
    enough to say where the plant is, which is all a crop window needs -- and
    the frames in between take an interpolated centre. The window is one fixed
    size for the whole sequence, as in `solve_tracking_crop` and
    `crop_from_masks`: the frames are stacked into one tensor so they must
    agree, and a per-frame size would rescale the plant frame to frame, which
    is exactly the apparent motion that makes tracking harder.
    """
    from sam3_leaf_track import TrackingCrop

    sub = list(range(0, len(frame_paths), stride))
    if sub[-1] != len(frame_paths) - 1:
        sub.append(len(frame_paths) - 1)   # both ends anchored, so nothing extrapolates
    print(f"  locating the plant with {phrase!r} on {len(sub)} of {len(frame_paths)} full frames")

    video = load_cropped_video([frame_paths[i] for i in sub], None)
    full_h, full_w = video[0].shape[:2]
    unions, _ = run_group(processor, model, torch, video, [phrase], device)

    centres = np.full((len(frame_paths), 2), np.nan)
    spans = []
    for k, frame_index in enumerate(sub):
        mask = unions.get(k, {}).get(phrase)
        if mask is None or not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        centres[frame_index] = ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0)
        spans.append(max(xs.max() - xs.min(), ys.max() - ys.min()))
    if not spans:
        raise SystemExit(
            f"SAM3 did not find {phrase!r} in any full frame, so there is nothing to crop to. "
            "Try --crop-phrase with different wording, or pass --p2 to crop from existing masks.")
    print(f"    found it in {len(spans)}/{len(sub)} of them; largest span {max(spans)} px")

    for axis in (0, 1):
        col = centres[:, axis]
        bad = np.isnan(col)
        if bad.all():
            col[:] = (full_w if axis == 0 else full_h) / 2.0
        elif bad.any():
            col[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), col[~bad])

    side = int(min(min(full_w, full_h), max(spans) * (1.0 + 2.0 * padding)))
    side = max(side, 64)
    boxes = []
    for cx, cy in centres:
        x0 = max(0, min(int(round(cx - side / 2)), full_w - side))
        y0 = max(0, min(int(round(cy - side / 2)), full_h - side))
        boxes.append((x0, y0, x0 + side, y0 + side))
    return TrackingCrop(boxes=boxes, width=side, height=side,
                        frame_width=full_w, frame_height=full_h)


def draw_panel(view_bgr, by_phrase: Dict[str, np.ndarray], phrases: Sequence[str],
               reference: Optional[Dict[str, np.ndarray]], label: str):
    """photo | SAM3 coloured by phrase | P2's plant and holder, for the eye.

    The numbers cannot distinguish "SAM3 disagreed with P2" from "SAM3 was
    right and P2 was wrong", and on this capture P2 is known to be wrong
    somewhere. That judgement is made here.
    """
    sam3 = view_bgr.copy()
    for index, phrase in enumerate(phrases):
        mask = by_phrase.get(phrase)
        if mask is None or not mask.any():
            continue
        colour = np.array(PHRASE_COLORS[index % len(PHRASE_COLORS)])
        sam3[mask] = (0.45 * sam3[mask] + 0.55 * colour).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(sam3, contours, -1, colour.tolist(), 2)

    columns = [view_bgr, sam3]
    lines = [label, "  ".join(f"[{p}]" for p in phrases)]
    if reference is not None:
        ref = view_bgr.copy()
        for name, colour in (("plant", (90, 210, 100)), ("holder", (200, 60, 200)),
                             ("root", (40, 140, 240))):
            mask = reference[name]
            if mask.any():
                ref[mask] = (0.45 * ref[mask] + 0.55 * np.array(colour)).astype(np.uint8)
        columns.append(ref)
        lines.append("reference: P2 plant/holder/root")

    panel = np.hstack(columns)
    for row, text in enumerate(lines):
        for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(panel, text, (14, 30 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, colour, thick)
    return panel


def score(per_frame: List[dict], n_frames: int) -> dict:
    """Average one phrase's per-frame numbers into the row that gets printed.

    Averaged over the frames the phrase *fired* on, with the count reported
    beside it: a phrase present in 3 of 15 frames and perfect in those three
    has not replaced a clicked prompt, and one mean would hide that.
    """
    fired = [f for f in per_frame if f["area"] > 0]
    if not fired:
        return {"frames": 0, "of": n_frames, "instances": 0.0, "area": 0,
                "iou_plant": 0.0, "iou_holder": 0.0, "inside_plant": 0.0,
                "inside_holder": 0.0, "covers_plant": 0.0, "covers_holder": 0.0}
    mean = lambda key: round(float(np.mean([f[key] for f in fired])), 3)  # noqa: E731
    return {
        "frames": len(fired),
        "of": n_frames,
        "instances": round(float(np.mean([f["instances"] for f in fired])), 1),
        "area": int(np.mean([f["area"] for f in fired])),
        "iou_plant": mean("iou_plant"),
        "iou_holder": mean("iou_holder"),
        "inside_plant": mean("inside_plant"),
        "inside_holder": mean("inside_holder"),
        "covers_plant": mean("covers_plant"),
        "covers_holder": mean("covers_holder"),
    }


def measure(mask: np.ndarray, reference: Dict[str, np.ndarray], instances: int) -> dict:
    return {
        "area": int(mask.sum()),
        "instances": instances,
        "iou_plant": iou(mask, reference["plant"]),
        "iou_holder": iou(mask, reference["holder"]),
        "inside_plant": fraction_of(mask, reference["plant"]),
        "inside_holder": fraction_of(mask, reference["holder"]),
        "covers_plant": fraction_of(reference["plant"], mask),
        "covers_holder": fraction_of(reference["holder"], mask),
    }


def print_table(title: str, rows: Dict[str, dict], scored: bool = True) -> None:
    """Without a reference every IoU column is 0.000 by construction, which
    reads as a measurement rather than as the absence of one -- so drop them."""
    print(f"\n{title}")
    if not scored:
        print(f"    {'phrase':<18} {'frames':>8} {'inst':>5} {'area':>8}")
        for phrase, s in rows.items():
            print(f"    {phrase:<18} {s['frames']:>3}/{s['of']:<4} {s['instances']:>5} "
                  f"{s['area']:>8}")
        return
    print(f"    {'phrase':<18} {'frames':>8} {'inst':>5} {'area':>8} {'IoU pl':>7} "
          f"{'IoU ho':>7} {'in pl':>6} {'in ho':>6} {'cov pl':>7} {'cov ho':>7}")
    for phrase, s in rows.items():
        print(f"    {phrase:<18} {s['frames']:>3}/{s['of']:<4} {s['instances']:>5} "
              f"{s['area']:>8} {s['iou_plant']:>7.3f} {s['iou_holder']:>7.3f} "
              f"{s['inside_plant']:>6.2f} {s['inside_holder']:>6.2f} "
              f"{s['covers_plant']:>7.2f} {s['covers_holder']:>7.2f}")


# --------------------------------------------------------------------------
# SAM3
# --------------------------------------------------------------------------


def load_model(model_name: str, device: str):
    try:
        from transformers import Sam3VideoModel, Sam3VideoProcessor
    except ImportError as exc:
        raise SystemExit(
            f"\nSAM3's video classes are not in this transformers ({exc}).\n"
            "They landed in transformers 5.0:  pip install -e \".[sam3]\"\n")
    print(f"loading {model_name} ...")
    try:
        processor = Sam3VideoProcessor.from_pretrained(model_name)
        model = Sam3VideoModel.from_pretrained(model_name).to(device).eval()
    except OSError as exc:
        if "gated" not in str(exc).lower() and "401" not in str(exc):
            raise
        raise SystemExit(
            f"\n{model_name} is a gated HuggingFace repo. To use it:\n"
            f"  1. accept the licence at https://huggingface.co/{model_name}\n"
            f"  2. hf auth login       (or: export HF_TOKEN=<your token>)\n")
    return processor, model


def decode(processor, session, model_outputs):
    """One frame's instances as {obj_id: bool mask} plus {obj_id: phrase}."""
    result = processor.postprocess_outputs(session, model_outputs)
    masks = result["masks"]
    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4:                       # (n, 1, H, W)
        masks = masks[:, 0]
    if masks.dtype != bool:
        masks = masks > 0.5

    by_id = {int(o): masks[k].astype(bool) for k, o in enumerate(result["object_ids"])}
    id_to_phrase = {}
    for phrase, ids in (result.get("prompt_to_obj_ids") or {}).items():
        for o in ids:
            id_to_phrase[int(o)] = phrase
    return by_id, id_to_phrase


def run_group(processor, model, torch, video, phrases: Sequence[str], device: str):
    """Track one group of phrases over the whole clip.

    Returns per frame: {phrase: union mask} and {phrase: instance count}.
    """
    session = processor.init_video_session(
        video=video, inference_device=device,
        processing_device="cpu", video_storage_device="cpu",
    )
    processor.add_text_prompt(session, list(phrases))

    unions: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)
    counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with torch.inference_mode():
        for model_outputs in model.propagate_in_video_iterator(
                inference_session=session, show_progress_bar=True):
            i = model_outputs.frame_idx
            by_id, id_to_phrase = decode(processor, session, model_outputs)
            for obj_id, mask in by_id.items():
                if not mask.any():
                    continue
                # An instance with no phrase attributed to it is charged to
                # the first prompt, matching sam3_leaf_track.py -- with one
                # phrase in the group that is exact, and with several it is
                # visible as a phrase scoring on a mask that is not its own.
                phrase = id_to_phrase.get(obj_id, phrases[0])
                unions[i][phrase] = (mask if phrase not in unions[i]
                                     else unions[i][phrase] | mask)
                counts[i][phrase] += 1

    del session
    if device == "cuda":
        torch.cuda.empty_cache()
    return unions, counts


def localize(processor, model, torch, frame_paths, phrase: str, stride: int,
             device: str, p2_dir: Path) -> dict:
    """Does SAM3 find the plant in a *full* frame, with no crop to help it?

    This is the question the real P2 backend turns on and the one the
    experiment above cannot ask, because it borrows P2's masks for its crop.
    Inside P2 there are no masks yet: the crop has to come from either the
    colour prepass -- the thing that fails when the pliers and the pot are in
    shot, which is why this idea exists -- or from SAM3 itself, run once at
    full frame to find the subject before the real pass runs on the crop.

    Scored as box IoU against the P2 plant mask's own bounding box, on every
    --localize-stride'th frame. A pass that holds the subject in a box is
    enough; the mask quality at this scale is not the point.
    """
    sub = frame_paths[::stride]
    print(f"\nlocalisation check: {phrase!r} on {len(sub)} full frames "
          f"(every {stride}{'st' if stride == 1 else 'th'})")
    video = load_cropped_video(sub, None)
    unions, counts = run_group(processor, model, torch, video, [phrase], device)

    rows = []
    for i, path in enumerate(sub):
        found = unions.get(i, {}).get(phrase)
        reference = cv2.imread(str(p2_dir / "masks" / "plant" / f"{path.stem}.png"),
                               cv2.IMREAD_GRAYSCALE)
        row = {"frame": path.stem, "instances": counts.get(i, {}).get(phrase, 0)}
        if found is None or not found.any() or reference is None or not (reference > 127).any():
            row.update({"box_iou": 0.0, "found": bool(found is not None and found.any())})
            rows.append(row)
            continue
        ys, xs = np.nonzero(found)
        ry, rx = np.nonzero(reference > 127)
        a = np.zeros(found.shape, bool)
        b = np.zeros(found.shape, bool)
        a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        b[ry.min():ry.max() + 1, rx.min():rx.max() + 1] = True
        row.update({"found": True, "box_iou": round(iou(a, b), 3),
                    "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "reference_box": [int(rx.min()), int(ry.min()),
                                      int(rx.max()), int(ry.max())]})
        rows.append(row)

    hits = [r for r in rows if r["found"]]
    ious = [r["box_iou"] for r in hits]
    print(f"    found the plant in {len(hits)}/{len(rows)} full frames; "
          f"median box IoU vs P2 {np.median(ious) if ious else 0:.3f}")
    for r in rows:
        print(f"    {r['frame']:<12} instances={r['instances']:<3} box IoU {r['box_iou']:.3f}")
    return {"phrase": phrase, "stride": stride, "frames": rows,
            "found_in": len(hits), "of": len(rows),
            "median_box_iou": round(float(np.median(ious)), 3) if ious else 0.0}


# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="<workdir>/p1/frames, or any folder of frames")
    p.add_argument("--p2", type=Path, default=None,
                   help="<workdir>/p2 from an existing SAM2 run, to score against. OPTIONAL: "
                        "without it the lab still runs and reports what each phrase found, it "
                        "just has no reference to compute IoU against")
    p.add_argument("--crop-from", choices=("p2", "sam3", "none"), default=None,
                   help="where the tracking crop comes from. Default: 'p2' when --p2 is given, "
                        "'sam3' otherwise -- which is what a real P2 backend would have to do")
    p.add_argument("--crop-phrase", default="plant",
                   help="the noun phrase --crop-from sam3 locates the subject with")
    p.add_argument("--crop-stride", type=int, default=8,
                   help="locate the plant on every Nth full frame; the rest interpolate. "
                        "A crop window does not need every frame measured")
    p.add_argument("--out", required=True)
    p.add_argument("--group", nargs="+", action="append", metavar=("NAME PHRASE", ""),
                   help="a P2 class name followed by the phrases to try for it. One SAM3 "
                        "session per group. Repeatable; replaces the defaults entirely")
    p.add_argument("--combo", action="append",
                   help="phrases to also score as one union, joined by '+', e.g. 'leaf+stem'. "
                        "Repeatable; replaces the defaults")
    p.add_argument("--frames", help="'a,b,c' or an inclusive range 'a-b'. ONE capture pass: "
                                    "SAM3 reads a jump between passes as motion")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--pad", type=float, default=0.18, help="crop margin, fraction of plant size")
    p.add_argument("--save-masks", action="store_true",
                   help="write each phrase's union mask per frame, in CROP coordinates, so the "
                        "disagreement with P2 can be analysed offline. The reference masks are "
                        "written beside them in the same space")
    p.add_argument("--localize", action="store_true",
                   help="also check whether SAM3 finds the plant in uncropped frames")
    p.add_argument("--localize-phrase", default="plant")
    p.add_argument("--localize-stride", type=int, default=4)
    p.add_argument("--model", default="facebook/sam3")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "run.log")
    print(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    groups = ([(g[0], list(g[1:])) for g in args.group] if args.group else DEFAULT_GROUPS)
    for name, phrases in groups:
        if not phrases:
            raise SystemExit(f"--group {name} was given no phrases")
    combos = [c.split("+") for c in (args.combo or DEFAULT_COMBOS)]

    # --- frames ---
    images = Path(args.images)
    frame_paths = sorted(images.glob("*.jpg")) + sorted(images.glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"no frames in {images}")
    if args.frames:
        if "-" in args.frames and "," not in args.frames:
            lo, hi = (int(x) for x in args.frames.split("-"))
            frame_paths = frame_paths[lo:hi + 1]
        else:
            frame_paths = [frame_paths[int(x)] for x in args.frames.split(",")
                           if int(x) < len(frame_paths)]
    frame_paths = frame_paths[::args.stride]
    print(f"\n{len(frame_paths)} frames: {frame_paths[0].name} .. {frame_paths[-1].name}")

    import torch
    processor, model = load_model(args.model, args.device)

    # --- the crop window ---
    source = args.crop_from or ("p2" if args.p2 else "sam3")
    if source == "p2":
        if not args.p2:
            raise SystemExit("--crop-from p2 needs --p2")
        crop = crop_from_masks(frame_paths, args.p2 / "masks" / "plant", args.pad)
        print(f"crop {crop.width}x{crop.height} of {crop.frame_width}x{crop.frame_height} "
              f"(from the reference P2 masks)")
    elif source == "sam3":
        crop = crop_from_sam3(processor, model, torch, frame_paths, args.crop_phrase,
                              args.crop_stride, args.pad, args.device)
        print(f"crop {crop.width}x{crop.height} of {crop.frame_width}x{crop.frame_height} "
              f"(located by SAM3 -- no P2 involved)")
    else:
        crop = None
        print("no crop -- full frames. SAM3 works at 1008px, so a small plant in a wide "
              "shot has very little resolution left to segment")

    video = load_cropped_video(frame_paths, crop)
    shape = (crop.height, crop.width) if crop else video[0].shape[:2]
    boxes = crop.boxes if crop else [(0, 0, shape[1], shape[0])] * len(frame_paths)

    scored = args.p2 is not None
    references = [load_reference(args.p2, path.stem, boxes[i], shape)
                  for i, path in enumerate(frame_paths)]
    if scored:
        for name in ("plant", "holder", "root"):
            present = sum(1 for r in references if r[name].any())
            print(f"  reference {name:<7} present in {present}/{len(references)} frames")
    else:
        print("  no --p2: reporting what each phrase found, with nothing to score it against")

    # --- one session per group ---
    all_masks: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)
    all_counts: Dict[int, Dict[str, int]] = defaultdict(dict)
    per_phrase: Dict[str, List[dict]] = {}
    results = {}
    for name, phrases in groups:
        print(f"\n=== group {name!r}: {phrases} over {len(video)} frames ===")
        unions, counts = run_group(processor, model, torch, video, phrases, args.device)
        rows = {}
        for phrase in phrases:
            per_frame = []
            for i in range(len(video)):
                mask = unions.get(i, {}).get(phrase)
                if mask is None:
                    mask = np.zeros(shape, bool)
                all_masks[i][phrase] = mask
                all_counts[i][phrase] = counts.get(i, {}).get(phrase, 0)
                per_frame.append(measure(mask, references[i], all_counts[i][phrase]))
            per_phrase[phrase] = per_frame
            rows[phrase] = score(per_frame, len(video))
        print_table(f"group {name!r} -- per phrase, averaged over the frames it fired on",
                    rows, scored)
        results[name] = {"phrases": phrases, "scores": rows}

    # --- combinations, because P2 wants one mask per class ---
    combo_rows = {}
    for parts in combos:
        usable = [p for p in parts if p in per_phrase]
        if not usable:
            continue
        per_frame = []
        for i in range(len(video)):
            union = np.zeros(shape, bool)
            instances = 0
            for phrase in usable:
                union |= all_masks[i][phrase]
                instances += all_counts[i][phrase]
            per_frame.append(measure(union, references[i], instances))
        combo_rows["+".join(usable)] = score(per_frame, len(video))
    if combo_rows:
        print_table("unions -- what a P2 plant mask built from these phrases would score",
                    combo_rows, scored)
    results["combinations"] = combo_rows

    # --- overlays ---
    ordered = [p for _, phrases in groups for p in phrases]
    if args.save_masks:
        # Crop coordinates, not full-frame: the question these answer is where
        # SAM3 and P2 disagree, and both are already in this space here.
        for i, path in enumerate(frame_paths):
            for phrase in ordered:
                mask = all_masks[i][phrase]
                if not mask.any():
                    continue
                d = out_dir / "masks" / phrase.replace(" ", "_")
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{path.stem}.png"), mask.astype(np.uint8) * 255)
            for name, mask in (references[i].items() if scored else ()):
                d = out_dir / "masks" / f"reference_{name}"
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{path.stem}.png"), mask.astype(np.uint8) * 255)
    for i, path in enumerate(frame_paths):
        view = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
        label = f"{path.stem}   " + "  ".join(
            f"{p}:{all_counts[i][p]}" for p in ordered if all_counts[i][p])
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"),
                    draw_panel(view, all_masks[i], ordered,
                               references[i] if scored else None, label),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    payload = {"frames": len(video),
               "frame_names": [p.name for p in frame_paths],
               "crop": crop.to_dict() if crop else None,
               "groups": results,
               "per_phrase_per_frame": per_phrase,
               "settings": {k: str(v) for k, v in vars(args).items()}}

    if args.localize:
        payload["localize"] = localize(processor, model, torch, frame_paths,
                                       args.localize_phrase, args.localize_stride,
                                       args.device, args.p2)
        with open(out_dir / "localize.json", "w") as f:
            json.dump(payload["localize"], f, indent=2)

    with open(out_dir / "lab.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_dir}/frame_*.jpg, lab.json, run.log")


if __name__ == "__main__":
    main()
