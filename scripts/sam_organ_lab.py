#!/usr/bin/env python
"""Standalone lab bench for SAM-based leaf/stem segmentation on real frames.

Deliberately independent of the pipeline: it needs images and a SAM2
checkpoint, nothing else. No COLMAP, no reconstruction, no run directory
layout. Point it at any folder of frames (or a video) and sweep parameters
until the masks look right, then port the settings into P4c.

    # simplest: a few frames from an existing run
    python scripts/sam_organ_lab.py --images runs/plant_9/p1/frames \
        --frames 0,31,57,76 --out /tmp/lab1

    # straight from a video, every 40th frame
    python scripts/sam_organ_lab.py --video /mnt/e/.../DSC_0009.MOV \
        --stride 40 --out /tmp/lab2 --auto-plant-mask

    # sweep the knob that matters most
    for e in 2.5 3.5 5.0; do
      python scripts/sam_organ_lab.py --images runs/plant_9/p1/frames \
          --frames 57 --out /tmp/elong_$e --max-leaf-elongation $e
    done

Writes per frame:
    <stem>_masks.jpg     photo | every SAM mask in its own colour
    <stem>_organs.jpg    photo | leaf/stem classification
    masks.json           per-mask geometry, so thresholds can be chosen from numbers
    run.log              everything printed here

What the classifier does, and where to intervene:
  1. SAM2 automatic mask generation over the (optionally masked) frame.
  2. Drop masks mostly outside the plant, near-duplicates, and the one mask
     that covers essentially the whole plant -- SAM reliably emits that and it
     is the plant, not an organ.
  3. Call a mask *stem* if it is a thin ribbon (elongation above
     --max-leaf-elongation), else *leaf*.
  4. Anything inside the plant silhouette that no mask claimed defaults to
     stem, so pixels are never silently dropped.

Step 3 is the crude part and the obvious thing to replace -- with a normal-map
cue, a LoRA-tuned SAM, or SAM3 text prompts. The numbers in masks.json are
there to decide whether it needs replacing for a given specimen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

LEAF, STEM, NONE = 1, 2, -1
LEAF_BGR = (90, 210, 100)
STEM_BGR = (60, 130, 230)  # BGR -> renders orange


class Tee:
    """Mirror stdout into a log file, because a run you cannot read later did
    not really happen."""

    def __init__(self, path: Path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------


def load_frames(args) -> List[Tuple[str, np.ndarray]]:
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"could not open {args.video}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = ([int(x) for x in args.frames.split(",")] if args.frames
                  else list(range(0, total, args.stride)))
        out = []
        for index in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if ok:
                out.append((f"frame_{index:05d}", image))
        cap.release()
        return out

    paths = sorted(Path(args.images).glob("*.jpg")) + sorted(Path(args.images).glob("*.png"))
    if args.frames:
        picks = [int(x) for x in args.frames.split(",")]
        paths = [paths[i] for i in picks if i < len(paths)]
    else:
        paths = paths[:: args.stride]
    return [(p.stem, cv2.imread(str(p))) for p in paths]


def auto_plant_mask(bgr: np.ndarray) -> np.ndarray:
    """Rough plant silhouette for a dark-backdrop rig, so the lab bench works
    without the pipeline's P2 masks.

    Brightness plus greenness, largest component. Crude on purpose -- it only
    has to keep SAM off the backdrop; pass --plant-mask-dir for the real thing.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b, g, r = cv2.split(bgr.astype(np.float32))
    total = np.maximum(b + g + r, 1.0)
    excess_green = (2 * g - r - b) / total

    mask = ((gray > otsu * 0.5) | (excess_green > 0.06)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask > 0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


def mask_shape(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    coords = np.stack([xs, ys], axis=1).astype(float)
    coords -= coords.mean(axis=0)
    sv = np.linalg.svd(coords, compute_uv=False) / np.sqrt(max(len(coords), 1))
    major, minor = float(sv[0]), float(max(sv[1], 1e-9))
    return {
        "area": int(mask.sum()),
        "major": round(major, 2),
        "minor": round(minor, 2),
        "elongation": round(major / minor, 2),
    }


def normal_flatness(normals: Optional[np.ndarray], mask: np.ndarray) -> Optional[float]:
    """How consistently the surface under this mask faces one direction.

    A lamina is a flat sheet, so its normals cluster tightly; a stem is a tube
    and its normals fan out around the axis. This is the geometric cue that
    should eventually replace 2D elongation -- it is a property of the organ
    rather than of how the organ happens to project into this view.

    Returned as the mean resultant length of the normals (1 = perfectly flat,
    toward 0 = fanning). Requires a rendered normal map; None without one.
    """
    if normals is None:
        return None
    selected = normals[mask]
    selected = selected[np.linalg.norm(selected, axis=1) > 0.1]
    if len(selected) < 20:
        return None
    selected = selected / np.linalg.norm(selected, axis=1, keepdims=True)
    # Normals are sign-ambiguous for a thin sheet seen from either side, so
    # average the outer products rather than the vectors themselves.
    tensor = (selected[:, :, None] * selected[:, None, :]).mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(tensor)
    return float(eigenvalues[-1])


def segment_frame(bgr, plant, generator, args, normals=None):
    height, width = plant.shape
    ys, xs = np.nonzero(plant)
    if len(xs) == 0:
        return np.full((height, width), NONE, np.int8), [], None

    if args.no_crop:
        y0, y1, x0, x1 = 0, height, 0, width
    else:
        pad = args.pad
        y0, y1 = max(0, ys.min() - pad), min(height, ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(width, xs.max() + pad)

    crop = bgr[y0:y1, x0:x1].copy()
    crop_plant = plant[y0:y1, x0:x1]
    crop_normals = normals[y0:y1, x0:x1] if normals is not None else None
    if not args.keep_background:
        crop[~crop_plant] = 0

    entries = generator.generate(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    plant_area = max(int(crop_plant.sum()), 1)

    kept, records, all_masks = [], [], []
    for entry in sorted(entries, key=lambda m: -m["area"]):
        segment = entry["segmentation"]
        inside = segment & crop_plant
        if inside.sum() < 16:
            continue
        all_masks.append(inside)

        shape = mask_shape(inside)
        fraction = inside.sum() / plant_area
        record = {"fraction": round(float(fraction), 4), **shape,
                  "flatness": normal_flatness(crop_normals, inside)}

        if inside.sum() < 0.5 * segment.sum():
            record["verdict"] = "rejected: mostly outside plant"
        elif fraction >= args.whole_plant_fraction:
            record["verdict"] = "rejected: whole plant"
        elif fraction < args.min_leaf_fraction:
            record["verdict"] = "rejected: too small"
        elif any((inside & prev).sum() > args.dedup_iou * min(inside.sum(), prev.sum())
                 for prev in kept):
            record["verdict"] = "rejected: duplicate"
        else:
            is_stem = shape["elongation"] > args.max_leaf_elongation
            if args.use_normals and record["flatness"] is not None:
                # A flat sheet outvotes the 2D shape: a leaf seen edge-on
                # projects as a ribbon and would otherwise be called stem.
                is_stem = record["flatness"] < args.min_leaf_flatness
            record["verdict"] = "stem" if is_stem else "leaf"
            kept.append(inside)
        records.append(record)

    classes = np.full(crop_plant.shape, NONE, np.int8)
    classes[crop_plant] = STEM
    for mask, record in zip(kept, [r for r in records if r["verdict"] in ("leaf", "stem")]):
        classes[mask] = LEAF if record["verdict"] == "leaf" else STEM

    full = np.full((height, width), NONE, np.int8)
    full[y0:y1, x0:x1] = classes
    return full, records, (crop, crop_plant, all_masks, (y0, y1, x0, x1))


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def write_visuals(out_dir: Path, stem: str, crop, all_masks, classes_crop):
    rng = np.random.default_rng(0)
    every = crop.copy()
    for mask in all_masks:
        colour = rng.integers(60, 255, 3)
        every[mask] = (0.45 * every[mask] + 0.55 * colour).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{stem}_masks.jpg"), np.hstack([crop, every]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    organs = crop.copy()
    for label, colour in ((LEAF, LEAF_BGR), (STEM, STEM_BGR)):
        hit = classes_crop == label
        if hit.any():
            organs[hit] = (0.42 * organs[hit] + 0.58 * np.array(colour)).astype(np.uint8)
    panel = np.hstack([crop, organs])
    cv2.putText(panel, "photo | leaf=green  stem=orange", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4)
    cv2.putText(panel, "photo | leaf=green  stem=orange", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1)
    cv2.imwrite(str(out_dir / f"{stem}_organs.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("input")
    src.add_argument("--images", help="folder of frames")
    src.add_argument("--video", help="video file (alternative to --images)")
    src.add_argument("--frames", help="comma-separated indices, e.g. 0,31,57")
    src.add_argument("--stride", type=int, default=24, help="use every Nth frame if --frames absent")
    src.add_argument("--plant-mask-dir", help="P2 masks; <stem>.png per frame")
    src.add_argument("--auto-plant-mask", action="store_true",
                     help="derive a rough plant mask instead (dark-backdrop rigs)")
    src.add_argument("--normals-dir", help="optional rendered normal maps (<stem>.npy, HxWx3)")
    src.add_argument("--out", required=True, help="output directory")

    sam = p.add_argument_group("SAM2")
    sam.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    sam.add_argument("--points-per-side", type=int, default=48,
                     help="sampling grid; higher finds smaller organs and costs time")
    sam.add_argument("--pred-iou", type=float, default=0.72)
    sam.add_argument("--stability", type=float, default=0.88)
    sam.add_argument("--min-region-area", type=int, default=200)
    sam.add_argument("--device", default="cuda")

    cls = p.add_argument_group("organ classification")
    cls.add_argument("--max-leaf-elongation", type=float, default=3.5,
                     help="above this a mask is called stem. Measured on DSC_0009: leaf blades "
                          "1.1-2.2, stem/root/petiole 4.4-8.3")
    cls.add_argument("--min-leaf-fraction", type=float, default=0.012,
                     help="masks smaller than this share of the plant are ignored")
    cls.add_argument("--whole-plant-fraction", type=float, default=0.85,
                     help="masks larger than this share are the plant itself, not an organ")
    cls.add_argument("--dedup-iou", type=float, default=0.8)
    cls.add_argument("--use-normals", action="store_true",
                     help="classify by surface flatness instead of 2D elongation (needs --normals-dir)")
    cls.add_argument("--min-leaf-flatness", type=float, default=0.75,
                     help="normals-based cutoff: above = flat sheet = leaf")

    view = p.add_argument_group("framing")
    view.add_argument("--no-crop", action="store_true", help="segment the full frame")
    view.add_argument("--keep-background", action="store_true",
                      help="do not zero pixels outside the plant mask")
    view.add_argument("--pad", type=int, default=60)

    args = p.parse_args()
    if not args.images and not args.video:
        p.error("give --images or --video")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "run.log")

    print(f"settings: {json.dumps(vars(args), indent=2)}")

    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from pose_estimator.segmentation import _resolve_model_cfg

    checkpoint = Path(args.checkpoint)
    model = build_sam2(_resolve_model_cfg(checkpoint), str(checkpoint), device=args.device)
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou,
        stability_score_thresh=args.stability,
        min_mask_region_area=args.min_region_area,
    )

    frames = load_frames(args)
    print(f"\n{len(frames)} frames to process\n")

    everything = {}
    for stem, bgr in frames:
        if bgr is None:
            continue
        if args.plant_mask_dir:
            m = cv2.imread(str(Path(args.plant_mask_dir) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            plant = m > 127 if m is not None else auto_plant_mask(bgr)
        elif args.auto_plant_mask:
            plant = auto_plant_mask(bgr)
        else:
            plant = np.ones(bgr.shape[:2], bool)

        normals = None
        if args.normals_dir:
            path = Path(args.normals_dir) / f"{stem}.npy"
            if path.exists():
                normals = np.load(path)

        with torch.inference_mode():
            classes, records, extra = segment_frame(bgr, plant, generator, args, normals)
        if extra is None:
            continue
        crop, _crop_plant, all_masks, (y0, y1, x0, x1) = extra
        write_visuals(out_dir, stem, crop, all_masks, classes[y0:y1, x0:x1])

        leaves = [r for r in records if r["verdict"] == "leaf"]
        stems = [r for r in records if r["verdict"] == "stem"]
        pixels = classes[classes != NONE]
        leaf_px = int((pixels == LEAF).sum())
        print(f"{stem}: {len(records)} masks -> {len(leaves)} leaf, {len(stems)} stem, "
              f"{len(records) - len(leaves) - len(stems)} rejected | "
              f"pixels {100 * leaf_px / max(len(pixels), 1):.1f}% leaf")
        for r in records:
            flat = f" flat {r['flatness']:.2f}" if r["flatness"] is not None else ""
            print(f"    {r['verdict']:<28} area {r['area']:>6} "
                  f"({100 * r['fraction']:>4.1f}%) elong {r['elongation']:>5.1f}{flat}")
        everything[stem] = records

    with open(out_dir / "masks.json", "w") as f:
        json.dump({"settings": vars(args), "frames": everything}, f, indent=2)

    print(f"\nwrote {out_dir}/*_masks.jpg, *_organs.jpg, masks.json, run.log")


if __name__ == "__main__":
    main()
