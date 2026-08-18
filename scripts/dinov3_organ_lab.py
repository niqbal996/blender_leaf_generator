#!/usr/bin/env python
"""DINOv3 patch features for plant organs: probe, part-label, and track.

DINOv3 does not return leaves, depths or classes. It returns a **feature
vector per image patch** -- a grid of roughly H/16 x W/16 embeddings. Those
vectors are semantically organised without any training: patches on the same
organ land close together, patches on different organs land further apart.
Everything else people do with it (segmentation, depth, classification) is a
small head trained on top of those features; the backbone alone gives you
similarity, and nothing more.

That is exactly what makes it worth trying here. Three modes, in the order
worth running them:

  --mode similarity   Pick one point. See every patch in that frame, and in
                      later frames, coloured by how similar it is. This is the
                      direct answer to "what does DINOv3 think this leaf is
                      like?" -- and the honest way to find out whether a leaf
                      stays recognisable when it turns edge-on.

  --mode parts        Give a handful of labelled points (leaf, stem, root).
                      Every patch in every frame is assigned to whichever
                      label it most resembles. A nearest-prototype classifier
                      in feature space -- no training, no thresholds, and it
                      generalises across species as well as the features do.

  --mode track        Seed one point on one leaf. Follow the peak of the
                      similarity map through the rotation. Answers the actual
                      question: is the leaf still matched when it rotates from
                      broad face to narrow face, or does the track jump?

WORKFLOW

  1. Write a reference frame with a coordinate grid, so points can be read off:

       python scripts/dinov3_organ_lab.py --mode reference \\
           --images runs/plant_9/p1/frames --frames 57 \\
           --plant-mask-dir runs/plant_9/p2/masks/plant --out /tmp/dino

     Open /tmp/dino/reference_frame_0057.jpg and read coordinates off the grid.
     All --seeds and --track-point values are in THAT image's pixel space.

  2. Probe a single point:

       python scripts/dinov3_organ_lab.py --mode similarity \\
           --images runs/plant_9/p1/frames --frames 57,63,70,76 \\
           --plant-mask-dir runs/plant_9/p2/masks/plant \\
           --seed-frame 57 --track-point 250,300 --out /tmp/dino_sim

  3. Label parts from a few clicks:

       python scripts/dinov3_organ_lab.py --mode parts \\
           --images runs/plant_9/p1/frames --stride 8 \\
           --plant-mask-dir runs/plant_9/p2/masks/plant --seed-frame 57 \\
           --seeds "leaf:250,300" "leaf:700,250" "stem:430,330" "root:400,600" \\
           --out /tmp/dino_parts

  4. Track one leaf through the turn:

       python scripts/dinov3_organ_lab.py --mode track \\
           --images runs/plant_9/p1/frames --stride 4 \\
           --plant-mask-dir runs/plant_9/p2/masks/plant \\
           --seed-frame 57 --track-point 250,300 --out /tmp/dino_track

MODEL ACCESS
  facebook/dinov3-* are gated. Accept the licence at
  https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m then run
  `hf auth login`. Until then use the ungated DINOv2 backbone, which behaves
  the same way for these purposes:
      --model facebook/dinov2-base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CLASS_COLORS = {
    "leaf": (40, 40, 220),      # red   (BGR)
    "tiny leaf": (40, 40, 220),
    "stem": (200, 60, 160),     # purple
    "petiole": (200, 60, 160),
    "root": (40, 140, 240),     # orange
}
SPARE = [(90, 210, 100), (200, 200, 60), (200, 60, 220), (230, 230, 230)]


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


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------


def load_frames(args) -> List[Tuple[str, np.ndarray]]:
    if args.video:
        cap = cv2.VideoCapture(args.video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = ([int(x) for x in args.frames.split(",")] if args.frames
                  else list(range(0, total, args.stride)))
        out = []
        for i in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, im = cap.read()
            if ok:
                out.append((f"frame_{i:04d}", im))
        cap.release()
        return out

    paths = sorted(Path(args.images).glob("*.jpg")) + sorted(Path(args.images).glob("*.png"))
    if args.frames:
        picks = [int(x) for x in args.frames.split(",")]
        paths = [paths[i] for i in picks if i < len(paths)]
    else:
        paths = paths[:: args.stride]
    return [(p.stem, cv2.imread(str(p))) for p in paths]


def crop_to_plant(bgr, mask_dir: Optional[str], stem: str, pad: int):
    """Crop to the subject before feature extraction.

    Not cosmetic. DINOv3 tiles the image into 16px patches, and the plant is
    about 2.5% of these frames -- uncropped it occupies a handful of patches
    and every organ falls inside one of them, so no amount of feature quality
    recovers organ-level detail.
    """
    if not mask_dir:
        return bgr, None
    m = cv2.imread(str(Path(mask_dir) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if m is None or not (m > 127).any():
        return bgr, None
    ys, xs = np.nonzero(m > 127)
    y0, y1 = max(0, ys.min() - pad), min(bgr.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(bgr.shape[1], xs.max() + pad)
    return bgr[y0:y1, x0:x1], (m > 127)[y0:y1, x0:x1]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


class Backbone:
    """DINOv3 (or DINOv2) patch features, L2-normalised, on a fixed grid."""

    def __init__(self, model_id: str, device: str, size: int):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.size = size
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        except OSError as exc:
            if "gated" not in str(exc).lower() and "401" not in str(exc):
                raise
            raise SystemExit(
                f"\n{model_id} is a gated HuggingFace repo.\n"
                f"  1. accept the licence at https://huggingface.co/{model_id}\n"
                f"  2. hf auth login      (or: export HF_TOKEN=<token>)\n"
                f"\nUngated alternative that works the same way here:\n"
                f"  --model facebook/dinov2-base\n"
            )

    def features(self, bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """(num_patches, dim) L2-normalised features, plus the patch grid shape.

        The image is resized to a fixed square so every frame yields the same
        grid, which is what lets features be compared across frames by index.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        inputs = self.processor(images=rgb, return_tensors="pt", do_resize=False,
                                do_center_crop=False).to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs).last_hidden_state[0]

        # Strip whatever prefix tokens this backbone uses (CLS, and DINOv3's
        # register tokens) by inferring how many are not patches.
        patch = getattr(self.model.config, "patch_size", 16)
        grid = self.size // patch
        prefix = out.shape[0] - grid * grid
        if prefix < 0:
            raise SystemExit(f"unexpected token count {out.shape[0]} for a {grid}x{grid} grid")
        tokens = out[prefix:]
        tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return tokens.cpu().numpy(), (grid, grid)


def patch_index(point_xy, image_shape, grid) -> int:
    """Image pixel -> flat patch index on the feature grid."""
    x, y = point_xy
    height, width = image_shape[:2]
    gx = min(int(x / width * grid[1]), grid[1] - 1)
    gy = min(int(y / height * grid[0]), grid[0] - 1)
    return gy * grid[1] + gx


def similarity_map(features: np.ndarray, prototype: np.ndarray, grid, out_shape) -> np.ndarray:
    """Cosine similarity of every patch to one prototype, upsampled to the image."""
    sim = features @ prototype
    sim = sim.reshape(grid)
    return cv2.resize(sim.astype(np.float32), (out_shape[1], out_shape[0]),
                      interpolation=cv2.INTER_CUBIC)


def heat_overlay(bgr, sim, mask=None, low=0.3) -> np.ndarray:
    norm = (sim - sim.min()) / max(sim.max() - sim.min(), 1e-9)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    blended = (0.45 * bgr + 0.55 * heat).astype(np.uint8)
    if mask is not None:
        blended[~mask] = (bgr[~mask] * 0.25).astype(np.uint8)
    faint = norm < low
    blended[faint] = (0.75 * bgr[faint] + 0.25 * blended[faint]).astype(np.uint8)
    return blended


# --------------------------------------------------------------------------


def parse_seeds(items) -> List[Tuple[str, Tuple[int, int]]]:
    out = []
    for item in items or []:
        if ":" not in item or "," not in item:
            raise SystemExit(f"--seeds wants label:x,y  (got {item!r})")
        label, coords = item.split(":", 1)
        x, y = coords.split(",")
        out.append((label.strip().lower(), (int(x), int(y))))
    return out


def color_for(label: str, assigned: Dict[str, tuple]) -> tuple:
    if label in assigned:
        return assigned[label]
    colour = CLASS_COLORS.get(label) or SPARE[len(assigned) % len(SPARE)]
    assigned[label] = colour
    return colour


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True,
                   choices=["reference", "similarity", "parts", "track"])
    p.add_argument("--images")
    p.add_argument("--video")
    p.add_argument("--frames", help="comma-separated indices")
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--plant-mask-dir")
    p.add_argument("--pad", type=int, default=60)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                   help="gated; use facebook/dinov2-base for an ungated equivalent")
    p.add_argument("--size", type=int, default=896,
                   help="square side the crop is resized to. Larger = finer patch grid "
                        "(896/16 = 56x56 patches) and more GPU memory")
    p.add_argument("--seed-frame", type=int,
                   help="frame index the seed coordinates were read off")
    p.add_argument("--seeds", nargs="*", help='parts mode: "leaf:250,300" "stem:430,330" ...')
    p.add_argument("--track-point", help="similarity/track mode: x,y")
    p.add_argument("--keep-background", action="store_true",
                   help="compute features on the whole crop. Off by default: the pliers and "
                        "table are strong, distinctive texture and they otherwise contribute "
                        "prototypes that compete with the plant's own")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if not args.images and not args.video:
        p.error("give --images or --video")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "run.log")
    print(json.dumps(vars(args), indent=2))

    frames = load_frames(args)
    print(f"\n{len(frames)} frames")

    # ---- reference: just dump a coordinate grid, no model needed ----
    if args.mode == "reference":
        for stem, bgr in frames:
            view, _ = crop_to_plant(bgr, args.plant_mask_dir, stem, args.pad)
            grid_img = view.copy()
            step = 50
            for x in range(0, view.shape[1], step):
                cv2.line(grid_img, (x, 0), (x, view.shape[0]), (70, 70, 70), 1)
                cv2.putText(grid_img, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (200, 200, 200), 1)
            for y in range(0, view.shape[0], step):
                cv2.line(grid_img, (0, y), (view.shape[1], y), (70, 70, 70), 1)
                cv2.putText(grid_img, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (200, 200, 200), 1)
            cv2.imwrite(str(out_dir / f"reference_{stem}.jpg"), grid_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"{stem}: crop {view.shape[1]}x{view.shape[0]} -> reference_{stem}.jpg")
        print("\nread seed coordinates off these images; they are the space "
              "--seeds and --track-point use")
        print("\nfor picking seeds for the pipeline, prefer clicking them:")
        print("    pose-pick-seeds --workdir <workdir>")
        print("which writes p4c/seeds.json for `pose-classify --seeds-file`. "
              "This grid\nis the fallback for machines with no display.")
        return

    print(f"loading {args.model} ...")
    backbone = Backbone(args.model, args.device, args.size)

    # ---- locate the seed frame ----
    # Loaded straight from disk rather than picked out of the selected subset.
    # With --stride 8 the subset is 0000, 0008, ... and a --seed-frame of 57 is
    # simply not in it; silently falling back to the first frame put every seed
    # coordinate on the wrong part of a different image, and the labels then
    # collapsed to one class on every frame but that one.
    seed_stem, seed_bgr = None, None
    if args.seed_frame is not None:
        if args.images:
            all_paths = sorted(Path(args.images).glob("*.jpg")) + sorted(Path(args.images).glob("*.png"))
            if args.seed_frame < len(all_paths):
                seed_stem = all_paths[args.seed_frame].stem
                seed_bgr = cv2.imread(str(all_paths[args.seed_frame]))
        else:
            cap = cv2.VideoCapture(args.video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.seed_frame)
            ok, seed_bgr = cap.read()
            cap.release()
            seed_stem = f"frame_{args.seed_frame:04d}" if ok else None
        if seed_bgr is None:
            raise SystemExit(f"--seed-frame {args.seed_frame} could not be loaded")
    if seed_stem is None:
        seed_stem, seed_bgr = frames[0]
    print(f"  seed frame resolved to {seed_stem}")
    seed_view, seed_plant = crop_to_plant(seed_bgr, args.plant_mask_dir, seed_stem, args.pad)
    if seed_plant is not None and not args.keep_background:
        seed_view = seed_view.copy()
        seed_view[~seed_plant] = 0
    seed_feats, grid = backbone.features(seed_view)
    print(f"seed frame {seed_stem}, crop {seed_view.shape[1]}x{seed_view.shape[0]}, "
          f"patch grid {grid[0]}x{grid[1]}, dim {seed_feats.shape[1]}")

    # ---- build prototypes ----
    prototypes: Dict[str, np.ndarray] = {}
    if args.mode == "parts":
        seeds = parse_seeds(args.seeds)
        if not seeds:
            raise SystemExit("--mode parts needs --seeds label:x,y ...")
        # Keep every seed vector separately and match to the nearest one, rather
        # than averaging each class into a single prototype. Averaging is what
        # the first version did and it is actively harmful here: a class given
        # several *different-looking* examples (three leaves at three angles)
        # averages toward a generic direction that resembles no individual leaf,
        # while a class with one tight example keeps its full similarity and
        # wins patches it should not.
        seed_vectors, seed_labels = [], []
        for label, point in seeds:
            seed_vectors.append(seed_feats[patch_index(point, seed_view.shape, grid)])
            seed_labels.append(label)
            print(f"  seed {label:<12} at {point}")
        prototypes = {"__vectors__": np.stack(seed_vectors, axis=1),
                      "__labels__": seed_labels}
    else:
        if not args.track_point:
            raise SystemExit(f"--mode {args.mode} needs --track-point x,y")
        x, y = (int(v) for v in args.track_point.split(","))
        prototypes["seed"] = seed_feats[patch_index((x, y), seed_view.shape, grid)]
        print(f"  seed point at ({x}, {y})")

    preview = seed_view.copy()
    if args.mode == "parts":
        for label, point in parse_seeds(args.seeds):
            colour = CLASS_COLORS.get(label, (230, 230, 230))
            cv2.drawMarker(preview, point, (255, 255, 255), cv2.MARKER_CROSS, 24, 4)
            cv2.drawMarker(preview, point, colour, cv2.MARKER_CROSS, 24, 2)
            cv2.putText(preview, label, (point[0] + 10, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(preview, label, (point[0] + 10, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    elif args.track_point:
        x, y = (int(v) for v in args.track_point.split(","))
        cv2.drawMarker(preview, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 26, 4)
        cv2.drawMarker(preview, (x, y), (40, 40, 220), cv2.MARKER_CROSS, 26, 2)
    cv2.imwrite(str(out_dir / "seeds_preview.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  seed placement -> {out_dir}/seeds_preview.jpg")

    assigned: Dict[str, tuple] = {}
    trajectory = []

    for stem, bgr in frames:
        view, plant = crop_to_plant(bgr, args.plant_mask_dir, stem, args.pad)
        if plant is not None and not args.keep_background:
            view = view.copy()
            view[~plant] = 0
        feats, this_grid = backbone.features(view)
        if this_grid != grid:
            print(f"  {stem}: grid mismatch {this_grid} -- skipped")
            continue

        if args.mode == "parts":
            # Nearest labelled example per patch. No threshold: every patch goes
            # to whichever individual seed it most resembles.
            seed_stack = prototypes["__vectors__"]
            seed_names = prototypes["__labels__"]
            labels = list(dict.fromkeys(seed_names))
            nearest = (feats @ seed_stack).argmax(axis=1)
            best = np.array([labels.index(seed_names[i]) for i in nearest]).reshape(grid)
            best = cv2.resize(best.astype(np.uint8), (view.shape[1], view.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

            overlay = view.copy()
            counts = {}
            for i, label in enumerate(labels):
                hit = best == i
                if plant is not None:
                    hit &= plant
                counts[label] = int(hit.sum())
                if hit.any():
                    colour = color_for(label, assigned)
                    overlay[hit] = (0.42 * overlay[hit] + 0.58 * np.array(colour)).astype(np.uint8)

            legend = "  ".join(f"{k}" for k in labels)
            panel = np.hstack([view, overlay])
            for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(panel, f"{stem}  DINO parts:  {legend}", (14, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.68, colour, thick)
            cv2.imwrite(str(out_dir / f"{stem}_parts.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
            total = max(sum(counts.values()), 1)
            print(f"{stem}: " + "  ".join(f"{k} {100 * v / total:.0f}%" for k, v in counts.items()))

        else:
            sim = similarity_map(feats, prototypes["seed"], grid, view.shape)
            search = sim.copy()
            if plant is not None:
                search[~plant] = -1.0
            peak = np.unravel_index(int(search.argmax()), search.shape)
            peak_value = float(search[peak])

            overlay = heat_overlay(view, sim, plant)
            cv2.drawMarker(overlay, (peak[1], peak[0]), (255, 255, 255),
                           cv2.MARKER_CROSS, 26, 3)
            cv2.drawMarker(overlay, (peak[1], peak[0]), (0, 0, 0), cv2.MARKER_CROSS, 26, 1)
            panel = np.hstack([view, overlay])
            for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(panel, f"{stem}  peak sim {peak_value:.3f} at "
                                   f"({peak[1]},{peak[0]})", (14, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.68, colour, thick)
            cv2.imwrite(str(out_dir / f"{stem}_{args.mode}.jpg"), panel,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            trajectory.append({"frame": stem, "peak_xy": [int(peak[1]), int(peak[0])],
                               "peak_similarity": round(peak_value, 4)})
            print(f"{stem}: peak similarity {peak_value:.3f} at ({peak[1]}, {peak[0]})")

    if trajectory:
        with open(out_dir / "trajectory.json", "w") as f:
            json.dump({"settings": vars(args), "seed_frame": seed_stem,
                       "trajectory": trajectory}, f, indent=2)
        values = [t["peak_similarity"] for t in trajectory]
        print(f"\npeak similarity over {len(values)} frames: "
              f"min {min(values):.3f}  median {np.median(values):.3f}  max {max(values):.3f}")
        print("A track that survives the turn holds a high peak throughout. A collapse "
              "marks the viewpoint where the leaf stopped being recognisable -- which is "
              "the number this experiment exists to produce.")

    print(f"\nwrote {out_dir}/")


if __name__ == "__main__":
    main()
