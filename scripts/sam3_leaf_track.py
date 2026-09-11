#!/usr/bin/env python
"""SAM3 concept tracking: does one leaf keep its identity through a rotation?

    # the question, asked directly
    python scripts/sam3_leaf_track.py --images runs/plant_9/p1/frames \
        --plant-mask-dir runs/plant_9/p2/masks/plant --out /tmp/sam3_leaves

    # wording is the experiment, same as text_organ_lab.py
    python scripts/sam3_leaf_track.py --images runs/plant_9/p1/frames \
        --prompts "leaf" "stem" --plant-mask-dir runs/plant_9/p2/masks/plant \
        --out /tmp/sam3_organs

    # masks on disk, in full-frame coordinates, for a downstream experiment
    python scripts/sam3_leaf_track.py --images runs/plant_9/p1/frames \
        --plant-mask-dir runs/plant_9/p2/masks/plant --save-masks --out /tmp/sam3_leaves

Why this exists, and how it differs from the two labs next to it.

`text_organ_lab.py --backend sam3` already runs SAM3 on single frames. That
answers "can it find leaves in this photo", which it can. It cannot answer the
question that actually blocks P4c/P5, because each frame is segmented
independently and the masks then have to be associated afterwards -- and
associating leaf masks across viewpoints by overlap or appearance is the step
that has never worked reliably here on a rotating plant.

SAM3's video path does the association inside the model: Promptable Concept
Segmentation returns "segmentation masks and unique identities for all
matching object instances", and the identities persist across frames. So the
measurement this script reports is not mask quality -- it is *track
persistence*: how many leaves get an id at all, how many hold that id for the
whole rotation, and how badly the rest fragment. That number decides whether
leaf correspondence can be delegated to SAM3 instead of being reconstructed
from 3D, and it is printed as a table at the end and written to tracks.json.

Two things to know before reading the output:

  * A turntable pass is not a video of a moving object -- it is a moving
    camera around a still one, and a leaf goes edge-on and disappears
    entirely at some viewpoints. A track that breaks there is the model
    behaving correctly on a hard sequence, not a bug; what matters is whether
    it *resumes under the same id* afterwards, which "gaps" below counts.
  * Each capture pass must be tracked on its own. Frames from two passes in
    one directory are temporally discontinuous at the join, and the tracker
    will treat the jump as motion. Pass --frames for the range of one pass.

Prompts are text only. SAM3 also accepts image exemplars, but the video
concept API exposed by transformers takes noun phrases
(`processor.add_text_prompt`) and has no entry point for "this leaf, here";
per-instance visual prompting is the Sam3Tracker* models, which track one
prompted object rather than detecting every instance of a concept.

Writes into --out:
    frame_XXXX.jpg     photo | tracked instances, one stable colour per id
    tracks.json        per-id: prompt, frames present, gaps, area, centroid, score
    masks/<id>/...     per-instance PNGs in full-frame coordinates (--save-masks)
    run.log
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pose_estimator.segmentation import TrackingCrop, solve_tracking_crop  # noqa: E402
from text_organ_lab import Tee  # noqa: E402

# Distinct at a glance and stable per id, so the same leaf is the same colour
# in every frame -- the whole point of the overlay is spotting the frame where
# a colour changes. BGR.
INSTANCE_COLORS = [
    (60, 60, 220), (60, 200, 240), (90, 210, 100), (230, 140, 60),
    (200, 60, 200), (60, 240, 240), (230, 80, 120), (120, 230, 180),
    (180, 130, 240), (40, 160, 250), (200, 200, 80), (140, 90, 230),
]


def color_for(obj_id: int):
    return INSTANCE_COLORS[obj_id % len(INSTANCE_COLORS)]


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------


def crop_from_masks(frame_paths, mask_dir: Path, padding: float) -> TrackingCrop:
    """A plant-following crop window solved from P2's masks.

    Same shape as `solve_tracking_crop` -- one window size for the whole
    sequence, moving per frame -- and preferred over it whenever P2 has run,
    because a segmented plant locates the subject better than the colour
    prepass that has to guess at it. The fixed size is not optional: the
    frames are stacked into one tensor, so they must agree, and a per-frame
    size would rescale the plant frame to frame, which is exactly the
    apparent motion that makes tracking harder.
    """
    first = cv2.imread(str(frame_paths[0]))
    full_h, full_w = first.shape[:2]

    centres, spans = [], []
    for path in frame_paths:
        m = cv2.imread(str(mask_dir / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None or not (m > 127).any():
            centres.append((np.nan, np.nan))
            continue
        ys, xs = np.nonzero(m > 127)
        centres.append(((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0))
        spans.append(max(xs.max() - xs.min(), ys.max() - ys.min()))
    if not spans:
        raise SystemExit(f"no usable masks in {mask_dir}")

    centres = np.asarray(centres, float)
    # Frames P2 lost the plant on inherit a neighbour's window rather than
    # dropping out: the sequence has to stay contiguous for the tracker.
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


def load_cropped_video(frame_paths, crop: Optional[TrackingCrop]) -> List[np.ndarray]:
    """RGB frames, all the same size, as SAM3's session wants them."""
    frames = []
    for i, path in enumerate(frame_paths):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise SystemExit(f"could not read {path}")
        if crop is not None:
            x0, y0, x1, y1 = crop.boxes[i]
            bgr = bgr[y0:y1, x0:x1]
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


# --------------------------------------------------------------------------
# tracks
# --------------------------------------------------------------------------


@dataclass
class Track:
    obj_id: int
    prompt: str = "?"
    frames: List[int] = None          # indices this id was present on
    areas: List[int] = None
    scores: List[float] = None
    centroids: List[List[float]] = None   # in cropped coordinates

    def __post_init__(self):
        self.frames = self.frames if self.frames is not None else []
        self.areas = self.areas if self.areas is not None else []
        self.scores = self.scores if self.scores is not None else []
        self.centroids = self.centroids if self.centroids is not None else []

    def summary(self, n_frames: int) -> dict:
        first, last = self.frames[0], self.frames[-1]
        span = last - first + 1
        return {
            "id": self.obj_id,
            "prompt": self.prompt,
            "first_frame": first,
            "last_frame": last,
            "frames_present": len(self.frames),
            # Frames inside the track's own span where it went missing. A
            # leaf turning edge-on and coming back is a gap; the id surviving
            # that is the behaviour being tested.
            "gaps": span - len(self.frames),
            "covers_sequence": len(self.frames) == n_frames,
            "mean_area": int(np.mean(self.areas)),
            "mean_score": round(float(np.mean(self.scores)), 3),
            "areas": self.areas,
            "centroids": [[round(c[0], 1), round(c[1], 1)] for c in self.centroids],
        }


def draw_overlay(view_bgr, masks_by_id: Dict[int, np.ndarray], frame_label: str):
    overlay = view_bgr.copy()
    # Largest first so a small leaf stays visible on top of the big one
    # behind it, matching text_organ_lab.py's painting order.
    for obj_id, mask in sorted(masks_by_id.items(), key=lambda kv: -int(kv[1].sum())):
        if not mask.any():
            continue
        colour = color_for(obj_id)
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array(colour)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, colour, 2)
        ys, xs = np.nonzero(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        for c, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(overlay, str(obj_id), (cx - 8, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, c, thick)

    panel = np.hstack([view_bgr, overlay])
    for c, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(panel, frame_label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, thick)
    return panel


def report(tracks: Dict[int, Track], n_frames: int, prompts: List[str]) -> dict:
    """The measurement this script exists for, printed and returned."""
    summaries = [t.summary(n_frames) for t in tracks.values()]
    summaries.sort(key=lambda s: (-s["frames_present"], s["id"]))

    per_prompt = {}
    for phrase in prompts:
        group = [s for s in summaries if s["prompt"] == phrase]
        if not group:
            print(f"\n{phrase!r}: nothing matched this prompt.")
            per_prompt[phrase] = {"tracks": 0}
            continue

        lengths = [s["frames_present"] for s in group]
        full = [s for s in group if s["covers_sequence"]]
        late = [s for s in group if s["first_frame"] > 0]
        broken = [s for s in group if s["gaps"] > 0]

        print(f"\n{phrase!r}: {len(group)} tracks over {n_frames} frames")
        print(f"    {'id':>4} {'frames':>8} {'span':>10} {'gaps':>6} {'mean area':>10} {'score':>6}")
        for s in group:
            print(f"    {s['id']:>4} {s['frames_present']:>4}/{n_frames:<3} "
                  f"{s['first_frame']:>4}-{s['last_frame']:<5} {s['gaps']:>6} "
                  f"{s['mean_area']:>10} {s['mean_score']:>6.3f}")

        stats = {
            "tracks": len(group),
            "covering_whole_sequence": len(full),
            "median_length": int(np.median(lengths)),
            "appearing_after_frame_0": len(late),
            "with_gaps": len(broken),
        }
        print(f"    -- {len(full)}/{len(group)} hold their id for all {n_frames} frames; "
              f"median track length {stats['median_length']}")
        # A track starting mid-sequence is either a leaf rotating into view or
        # the same leaf coming back as a new id. The two are indistinguishable
        # from these numbers alone, which is what the overlays are for: an id
        # that changes colour on a leaf already visible is a switch.
        print(f"    -- {len(late)} appear after frame 0, {len(broken)} have gaps "
              f"(check the overlays: a new id on an already-visible leaf is a switch)")
        per_prompt[phrase] = stats

    return {"frames": n_frames, "per_prompt": per_prompt, "tracks": summaries}


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True,
                   help="a directory of frames from ONE capture pass, e.g. <workdir>/p1/frames")
    p.add_argument("--out", required=True)
    p.add_argument("--prompts", nargs="+", default=["leaf"],
                   help="noun phrases. Every instance of each is detected and tracked")
    p.add_argument("--plant-mask-dir",
                   help="P2 plant masks, e.g. <workdir>/p2/masks/plant. Strongly recommended: "
                        "the plant is a few percent of the frame and SAM3 works at 1008px, so "
                        "uncropped there is nothing to segment")
    p.add_argument("--frames", help="explicit indices 'a,b,c' or an inclusive range 'a-b'. "
                                    "Use a range to isolate one capture pass")
    p.add_argument("--stride", type=int, default=1,
                   help="every Nth frame. Above 1 this stops being a continuity test")
    p.add_argument("--pad", type=float, default=0.18, help="crop margin, fraction of plant size")
    p.add_argument("--no-crop", action="store_true", help="feed full frames")
    p.add_argument("--save-masks", action="store_true",
                   help="write per-instance PNGs in full-frame coordinates")
    p.add_argument("--model", default="facebook/sam3")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "run.log")
    print(json.dumps(vars(args), indent=2))

    # --- frames ---
    frame_paths = sorted(Path(args.images).glob("*.jpg")) + sorted(Path(args.images).glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"no frames in {args.images}")
    if args.frames:
        if "-" in args.frames and "," not in args.frames:
            lo, hi = (int(x) for x in args.frames.split("-"))
            frame_paths = frame_paths[lo:hi + 1]
        else:
            picks = [int(x) for x in args.frames.split(",")]
            frame_paths = [frame_paths[i] for i in picks if i < len(frame_paths)]
    frame_paths = frame_paths[::args.stride]
    print(f"\n{len(frame_paths)} frames: {frame_paths[0].name} .. {frame_paths[-1].name}")

    # --- framing ---
    crop = None
    if not args.no_crop:
        if args.plant_mask_dir:
            crop = crop_from_masks(frame_paths, Path(args.plant_mask_dir), args.pad)
            print(f"crop {crop.width}x{crop.height} of {crop.frame_width}x{crop.frame_height} "
                  f"(from P2 masks)")
        else:
            print("no --plant-mask-dir; falling back to the colour prepass for the crop")
            crop = solve_tracking_crop(frame_paths, padding_fraction=args.pad)
            print(f"crop {crop.width}x{crop.height} of {crop.frame_width}x{crop.frame_height}")

    video = load_cropped_video(frame_paths, crop)

    # --- model ---
    import torch
    try:
        from transformers import Sam3VideoModel, Sam3VideoProcessor
    except ImportError as exc:
        raise SystemExit(
            f"\nSAM3's video classes are not in this transformers ({exc}).\n"
            "They landed in transformers 5.0:  pip install -e \".[sam3]\"\n"
            "Note that is a major-version bump for an environment that also runs P4c;\n"
            "see README 'SAM3 (experimental)' before installing it beside the pipeline.\n"
        )

    print(f"\nloading {args.model} ...")
    try:
        processor = Sam3VideoProcessor.from_pretrained(args.model)
        model = Sam3VideoModel.from_pretrained(args.model).to(args.device).eval()
    except OSError as exc:
        if "gated" not in str(exc).lower() and "401" not in str(exc):
            raise
        raise SystemExit(
            f"\n{args.model} is a gated HuggingFace repo. To use it:\n"
            f"  1. accept the licence at https://huggingface.co/{args.model}\n"
            f"  2. hf auth login       (or: export HF_TOKEN=<your token>)\n"
        )

    session = processor.init_video_session(
        video=video,
        inference_device=args.device,
        processing_device="cpu",
        video_storage_device="cpu",
    )
    # One pass detects every instance of every prompt; the vision features are
    # shared, so extra prompts are much cheaper than extra runs.
    processor.add_text_prompt(session, list(args.prompts))
    print(f"prompts: {args.prompts}")

    # --- propagate ---
    print(f"\ntracking {len(video)} frames ...")
    tracks: Dict[int, Track] = {}
    if args.save_masks:
        (out_dir / "masks").mkdir(exist_ok=True)

    with torch.inference_mode():
        for model_outputs in model.propagate_in_video_iterator(
                inference_session=session, show_progress_bar=True):
            i = model_outputs.frame_idx
            result = processor.postprocess_outputs(session, model_outputs)

            obj_ids = [int(o) for o in result["object_ids"]]
            masks = result["masks"]
            scores = result["scores"]
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
            masks = np.asarray(masks)
            if masks.ndim == 4:              # (n, 1, H, W)
                masks = masks[:, 0]
            masks = masks > 0.5 if masks.dtype != bool else masks

            # Which prompt found each id. Constant for an id's lifetime, but
            # only reported per frame, so it is recorded the first time seen.
            id_to_prompt = {}
            for phrase, ids in (result.get("prompt_to_obj_ids") or {}).items():
                for o in ids:
                    id_to_prompt[int(o)] = phrase

            masks_by_id = {}
            for k, obj_id in enumerate(obj_ids):
                mask = masks[k].astype(bool)
                if not mask.any():
                    continue
                masks_by_id[obj_id] = mask
                track = tracks.setdefault(obj_id, Track(obj_id))
                if track.prompt == "?":
                    track.prompt = id_to_prompt.get(obj_id, args.prompts[0])
                ys, xs = np.nonzero(mask)
                track.frames.append(i)
                track.areas.append(int(mask.sum()))
                track.scores.append(float(scores[k]))
                track.centroids.append([float(xs.mean()), float(ys.mean())])

            view = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
            label = f"{frame_paths[i].stem}  {len(masks_by_id)} instances"
            cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"),
                        draw_overlay(view, masks_by_id, label),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            if args.save_masks:
                box = crop.boxes[i] if crop is not None else None
                h = crop.frame_height if crop else view.shape[0]
                w = crop.frame_width if crop else view.shape[1]
                for obj_id, mask in masks_by_id.items():
                    d = out_dir / "masks" / str(obj_id)
                    d.mkdir(parents=True, exist_ok=True)
                    # Pasted back to full-frame coordinates, like P2's masks,
                    # so anything downstream can read both the same way.
                    full = np.zeros((h, w), np.uint8)
                    if box is None:
                        full[:] = mask.astype(np.uint8) * 255
                    else:
                        x0, y0, x1, y1 = box
                        full[y0:y1, x0:x1] = mask.astype(np.uint8) * 255
                    cv2.imwrite(str(d / f"{frame_paths[i].stem}.png"), full)

    # --- report ---
    if not tracks:
        print("\nno instances tracked at all -- try a different --prompts wording, "
              "or check the overlays to see what the crop actually contains.")
    summary = report(tracks, len(video), list(args.prompts))
    summary["settings"] = vars(args)
    summary["frame_names"] = [p.name for p in frame_paths]
    with open(out_dir / "tracks.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_dir}/frame_*.jpg, tracks.json, run.log")


if __name__ == "__main__":
    main()
