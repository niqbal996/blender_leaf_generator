#!/usr/bin/env python
"""Zero-shot text-prompted organ segmentation: SAM3, or Grounding DINO + SAM2.

    # SAM3, text straight to masks
    python scripts/text_organ_lab.py --images runs/plant_9/p1/frames \
        --frames 57,76 --plant-mask-dir runs/plant_9/p2/masks/plant \
        --out /tmp/text1

    # try different wording -- this is the whole point
    python scripts/text_organ_lab.py --images runs/plant_9/p1/frames --frames 57 \
        --prompts "leaf blade" "stem and petiole" --out /tmp/text2

    # Grounding DINO boxes prompting SAM2 for the masks, for comparison
    python scripts/text_organ_lab.py --images runs/plant_9/p1/frames --frames 57 \
        --backend groundingdino --out /tmp/text3

Why this exists. Deciding leaf-versus-stem from the *shape* of a SAM mask does
not generalise, and measurement says so: on two frames of DSC_0009, 2D
elongation put leaves at 1.1-5.7 and stems at 3.8-8.3 (wide overlap), and
Depth-Anything surface flatness put leaves at 0.73-0.91 and stems at 0.46-0.73
(touching, no gap). Every such statistic describes how an organ *looks from
one viewpoint*, and both organs can produce any value of it.

A text-prompted model asserts the class instead of leaving it to be inferred,
so there is no threshold to tune per species. That is the property being
tested here, and the reason the prompts are a command-line argument: the
experiment is which wording generalises.

Writes per frame:
    <stem>_<backend>.jpg   photo | class overlay
    results.json           per-detection class, score, box, area
    run.log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# BGR. Keyed by name so a class keeps its colour no matter what order the
# prompts are given in, or how many there are.
NAMED_COLORS = {
    "red": (40, 40, 220), "purple": (200, 60, 160), "orange": (40, 140, 240),
    "green": (90, 210, 100), "blue": (230, 130, 60), "cyan": (200, 200, 60),
    "yellow": (60, 200, 230), "magenta": (200, 60, 220), "white": (230, 230, 230),
}
# Defaults for the organs this pipeline cares about; anything else falls back
# to the spare palette below, in order.
DEFAULT_CLASS_COLORS = {
    "leaf": "red", "tiny leaf": "red", "small leaf": "red", "leaf blade": "red",
    "stem": "purple", "petiole": "purple", "stem and petiole": "purple", "branch": "purple",
    "root": "orange",
}
SPARE = ["green", "cyan", "yellow", "magenta", "blue", "white"]


def resolve_colors(prompts, overrides):
    """One BGR colour per prompt, from --colors, then the organ defaults, then
    the spare palette."""
    explicit = {}
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--colors wants name=colour, got {item!r}")
        key, value = item.split("=", 1)
        explicit[key.strip().lower()] = value.strip().lower()

    out, spare = [], list(SPARE)
    for phrase in prompts:
        key = phrase.strip().lower()
        name = explicit.get(key) or DEFAULT_CLASS_COLORS.get(key)
        if name is None:
            name = spare.pop(0) if spare else "white"
        if name not in NAMED_COLORS:
            raise SystemExit(f"unknown colour {name!r}; pick from {sorted(NAMED_COLORS)}")
        out.append((name, NAMED_COLORS[name]))
    return out


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def load_frames(images: Optional[str], video: Optional[str], frames: Optional[str],
                stride: int) -> List[Tuple[str, np.ndarray]]:
    if video:
        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = ([int(x) for x in frames.split(",")] if frames else list(range(0, total, stride)))
        out = []
        for i in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, im = cap.read()
            if ok:
                out.append((f"frame_{i:05d}", im))
        cap.release()
        return out
    paths = sorted(Path(images).glob("*.jpg")) + sorted(Path(images).glob("*.png"))
    if frames:
        picks = [int(x) for x in frames.split(",")]
        paths = [paths[i] for i in picks if i < len(paths)]
    else:
        paths = paths[::stride]
    return [(p.stem, cv2.imread(str(p))) for p in paths]


def crop_to_plant(bgr, mask_dir: Optional[str], stem: str, pad: int):
    """Crop to the subject. Necessary, not cosmetic: the plant is ~2.5% of these
    frames, and every model here works at a few hundred pixels internally, so
    uncropped the plant is a smudge and nothing detects organs in it."""
    if not mask_dir:
        return bgr, (0, bgr.shape[0], 0, bgr.shape[1])
    m = cv2.imread(str(Path(mask_dir) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if m is None or not (m > 127).any():
        return bgr, (0, bgr.shape[0], 0, bgr.shape[1])
    ys, xs = np.nonzero(m > 127)
    y0, y1 = max(0, ys.min() - pad), min(bgr.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(bgr.shape[1], xs.max() + pad)
    return bgr[y0:y1, x0:x1], (y0, y1, x0, x1)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


def run_sam3(view, prompts, model, processor, threshold, device):
    """SAM3: text in, instance masks out, one prompt at a time.

    Each phrase is run separately rather than concatenated. SAM3 accepts a
    single noun phrase per pass, and running them separately also keeps the
    per-class scores comparable instead of competing inside one query.
    """
    import torch

    rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
    found = []
    for class_id, phrase in enumerate(prompts):
        inputs = processor(images=rgb, text=phrase, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=0.5,
            target_sizes=[(view.shape[0], view.shape[1])],
        )[0]
        for mask, score in zip(results["masks"], results["scores"]):
            found.append({
                "class_id": class_id, "phrase": phrase,
                "score": float(score),
                "mask": mask.cpu().numpy().astype(bool),
            })
    return found


def run_grounding_dino(view, prompts, model, processor, sam_predictor, threshold, device):
    """Grounding DINO proposes boxes from text; SAM2 turns them into masks.

    One phrase per forward pass, deliberately. Concatenating them into a single
    query ("leaf. tiny leaf. stem. root.") makes the model ground against
    *tokens*, and overlapping phrases then come back merged -- prompting with
    "leaf" and "tiny leaf" returned labels like "leaf tiny leaf", which cannot
    be attributed to either class. Separate passes cost more time and give an
    unambiguous class per detection.
    """
    import torch

    rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(rgb)

    found = []
    for class_id, phrase in enumerate(prompts):
        query = phrase.strip().lower().rstrip(".") + "."
        inputs = processor(images=rgb, text=query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=threshold, text_threshold=0.25,
            target_sizes=[(view.shape[0], view.shape[1])],
        )[0]
        for box, score in zip(results["boxes"], results["scores"]):
            masks, _iou, _ = sam_predictor.predict(
                box=box.cpu().numpy()[None, :], multimask_output=False)
            found.append({
                "class_id": class_id, "phrase": phrase, "score": float(score),
                "mask": masks[0].astype(bool),
            })
    return found


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images")
    p.add_argument("--video")
    p.add_argument("--frames")
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--plant-mask-dir")
    p.add_argument("--pad", type=int, default=60)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", choices=["sam3", "groundingdino"], default="sam3")
    p.add_argument("--prompts", nargs="+", default=["leaf", "stem"],
                   help="one noun phrase per class, in order. Try: 'leaf blade' 'stem and petiole'")
    p.add_argument("--threshold", type=float, default=0.3, help="detection score cutoff")
    p.add_argument("--max-class-fraction", type=float, default=0.85,
                   help="drop a detection covering more than this share of the plant. Grounding "
                        "DINO cannot localise thin parts, so asking it for 'stem' or 'root' "
                        "returns a box around the whole plant; without this the whole plant gets "
                        "painted with whichever such class scored highest")
    p.add_argument("--colors", nargs="*", default=None,
                   help="override per class, e.g. --colors leaf=red stem=purple root=orange. "
                        "Defaults already map leaf/tiny leaf->red, stem/petiole->purple, root->orange")
    p.add_argument("--sam3-model", default="facebook/sam3")
    p.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--sam2-checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--keep-background", action="store_true",
                   help="do not zero pixels outside the plant mask")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if not args.images and not args.video:
        p.error("give --images or --video")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "run.log")
    print(json.dumps(vars(args), indent=2))

    import torch

    model = processor = sam_predictor = None
    if args.backend == "sam3":
        from transformers import Sam3Model, Sam3Processor
        print(f"\nloading {args.sam3_model} ...")
        try:
            processor = Sam3Processor.from_pretrained(args.sam3_model)
            model = Sam3Model.from_pretrained(args.sam3_model).to(args.device).eval()
        except OSError as exc:
            if "gated" not in str(exc).lower() and "401" not in str(exc):
                raise
            raise SystemExit(
                f"\n{args.sam3_model} is a gated HuggingFace repo. To use it:\n"
                f"  1. accept the licence at https://huggingface.co/{args.sam3_model}\n"
                f"  2. hf auth login       (or: export HF_TOKEN=<your token>)\n"
                f"\nUntil then use the ungated backend:  --backend groundingdino\n"
            )
    else:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from pose_estimator.segmentation import _resolve_model_cfg

        print(f"\nloading {args.gdino_model} + SAM2 ...")
        processor = AutoProcessor.from_pretrained(args.gdino_model)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model).to(args.device).eval()
        ckpt = Path(args.sam2_checkpoint)
        sam_predictor = SAM2ImagePredictor(
            build_sam2(_resolve_model_cfg(ckpt), str(ckpt), device=args.device))

    colors = resolve_colors(args.prompts, args.colors)
    print("prompts and colours:")
    for phrase, (name, _bgr) in zip(args.prompts, colors):
        print(f"    {phrase:<20} -> {name}")
    print()
    frames = load_frames(args.images, args.video, args.frames, args.stride)
    everything = {}

    for stem, bgr in frames:
        if bgr is None:
            continue
        view, (y0, y1, x0, x1) = crop_to_plant(bgr, args.plant_mask_dir, stem, args.pad)
        plant = None
        if args.plant_mask_dir:
            m = cv2.imread(str(Path(args.plant_mask_dir) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                plant = (m > 127)[y0:y1, x0:x1]
                if not args.keep_background:
                    view = view.copy()
                    view[~plant] = 0

        with torch.inference_mode():
            if args.backend == "sam3":
                found = run_sam3(view, args.prompts, model, processor, args.threshold, args.device)
            else:
                found = run_grounding_dino(view, args.prompts, model, processor,
                                           sam_predictor, args.threshold, args.device)

        overlay = view.copy()
        records = []
        plant_area = int(plant.sum()) if plant is not None else view.shape[0] * view.shape[1]

        # Largest first, so a small organ paints over the big one it sits on.
        # Score order would be wrong here: a confident whole-plant detection
        # would bury every organ inside it.
        for det in sorted(found, key=lambda d: -int(d["mask"].sum())):
            mask = det["mask"]
            if plant is not None:
                mask = mask & plant
            if mask.sum() < 64:
                continue
            fraction = mask.sum() / max(plant_area, 1)
            if fraction > args.max_class_fraction:
                records.append({"class": args.prompts[det["class_id"]],
                                "score": round(det["score"], 3), "area": int(mask.sum()),
                                "rejected": "covers the whole plant"})
                continue
            colour = colors[det["class_id"]][1]
            overlay[mask] = (0.42 * overlay[mask] + 0.58 * np.array(colour)).astype(np.uint8)
            records.append({"class": args.prompts[det["class_id"]],
                            "score": round(det["score"], 3), "area": int(mask.sum())})

        legend = "  ".join(f"{p}={colors[i][0]}" for i, p in enumerate(args.prompts))
        panel = np.hstack([view, overlay])
        for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(panel, f"{stem}  {args.backend}   {legend}", (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, thick)
        cv2.imwrite(str(out_dir / f"{stem}_{args.backend}.jpg"), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        print(f"{stem}: {len(records)} detections")
        dropped = [r for r in records if r.get("rejected")]
        if dropped:
            names = ", ".join(sorted({r["class"] for r in dropped}))
            print(f"    (dropped {len(dropped)} whole-plant detection(s) from: {names})")
        for phrase in args.prompts:
            group = [r for r in records if r["class"] == phrase and not r.get("rejected")]
            if not group:
                print(f"    {phrase:<22}   0 masks  -- nothing matched this prompt")
                continue
            areas = sum(g["area"] for g in group)
            best = max(g["score"] for g in group)
            print(f"    {phrase:<22} {len(group):>3} masks  {areas:>7} px  best score {best:.3f}")
        everything[stem] = records

    with open(out_dir / "results.json", "w") as f:
        json.dump({"settings": vars(args), "frames": everything}, f, indent=2)
    print(f"\nwrote {out_dir}/*_{args.backend}.jpg, results.json, run.log")


if __name__ == "__main__":
    main()
