"""P2 -- multi-view segmentation of the plant and its holder, via SAM2 video
propagation.

Produces, for every extracted frame, three mutually-exclusive classes:

- ``plant``  -- the specimen itself (leaves, stem, petioles, exposed roots)
- ``holder`` -- the pliers/clamp/pot gripping it
- background -- everything else (implicitly, whatever neither mask claims)

The holder class is not optional bookkeeping. A clamp touching the stem base
is spatially continuous with the plant, so any downstream skeletonizer will
happily absorb it as a fake basal branch. Carving it out here, once, in 2D,
is far easier than trying to recognise and delete it in 3D later.

Why SAM2 rather than a color threshold: the eventual consumer is the visual
hull (P4), and a hull is only as good as its silhouettes. Color thresholds
fail exactly where it matters -- dark stems against a dark backdrop, thin
petioles, specular leaf highlights -- and each failure carves real geometry
away permanently. SAM2 propagates one set of clicks through the whole
sequence with temporal memory, so a petiole that is ambiguous in one frame is
still resolved by its neighbours.

Requires the "segment" extra plus a SAM2 checkpoint -- see the README.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Object ids used throughout the SAM2 session. 0 is avoided deliberately --
# SAM2 accepts it, but it reads as "no object" in enough places downstream
# (and in saved label images) that keeping it free is worth one integer.
PLANT_ID = 1
HOLDER_ID = 2
CLASS_NAMES = {PLANT_ID: "plant", HOLDER_ID: "holder"}


# --------------------------------------------------------------------------
# Prompt derivation
# --------------------------------------------------------------------------
#
# SAM2 needs a few click points to know what it is tracking. Rather than
# require a GUI, we derive them from color -- which is safe here in a way a
# color *threshold* would not be, because these points only have to land
# somewhere inside the right object. SAM2 then finds the actual boundary
# itself, so a sloppy seed still yields a correct mask, and a seed that lands
# on the wrong object is obvious in the diagnostic overlay rather than
# silently biasing geometry.


def excess_green(bgr: np.ndarray) -> np.ndarray:
    """Excess Green Index (2G - R - B) on 0..1 chromatic coordinates.

    Chromatic normalisation (dividing by R+G+B) is what makes this robust to
    the strong brightness falloff across a turntable: a shadowed leaf and a
    lit leaf have very different RGB values but similar chromaticity.
    """
    bgr = bgr.astype(np.float32)
    total = bgr.sum(axis=2, keepdims=True)
    total[total == 0] = 1.0
    b, g, r = np.split(bgr / total, 3, axis=2)
    return (2.0 * g - r - b)[:, :, 0]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Binary mask reduced to its single largest connected component."""
    mask = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _significant_components(mask: np.ndarray, min_area_fraction: float = 2e-4) -> np.ndarray:
    """Drop connected components smaller than `min_area_fraction` of the frame.

    The ROI is a union over every frame, so it is only as tight as the single
    worst speck anywhere in the sequence -- one surviving noise pixel in a
    corner drags the crop out to the full image. Unlike `_largest_component`
    this keeps *all* real blobs, because the plant and the holder are
    genuinely separate objects and the ROI must contain both.
    """
    mask = (mask > 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask.astype(bool)

    min_area = min_area_fraction * mask.size
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    if not keep.any():  # everything is small -- fall back to the largest blob
        keep[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))] = True
    return keep[labels]


def _interior_points(mask: np.ndarray, n: int = 3) -> List[Tuple[int, int]]:
    """Up to `n` points well inside `mask`, ranked by distance from its edge.

    Prompting with edge-adjacent pixels is what makes SAM2 latch onto a
    neighbouring object; the distance transform's peaks are the safest
    interior seeds available, and for a thin structure they are the midline.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    if not mask_u8.any():
        return []
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)

    points: List[Tuple[int, int]] = []
    working = dist.copy()
    for _ in range(n):
        _, max_val, _, max_loc = cv2.minMaxLoc(working)
        if max_val <= 0:
            break
        points.append((int(max_loc[0]), int(max_loc[1])))
        # Suppress a neighbourhood so the next pick is not the same blob.
        cv2.circle(working, max_loc, max(8, int(max_val * 2)), 0, -1)
    return points


def foreground_gate(bgr: np.ndarray, relative: float = 0.25) -> np.ndarray:
    """Everything brighter than a fraction of the image's own Otsu threshold.

    Chromatic indices like ExG divide by (R+G+B), which is exactly what makes
    them robust to shading -- and exactly what makes them explode on
    near-black pixels, where sensor and JPEG noise of a couple of DN turns
    into a "strongly green" chromaticity. Against this rig's black backdrop
    that produces green confetti across the whole frame, which is enough to
    blow the subject ROI out to the full image.

    The threshold is derived per-frame from the image's own histogram rather
    than fixed, so it follows exposure; `relative` sits well below Otsu on
    purpose, since the job here is only to exclude the near-black backdrop,
    not to find the subject.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray > (otsu_threshold * relative)


def holder_color_mask(bgr: np.ndarray) -> np.ndarray:
    """Rough "this is the holder" mask, for prompt derivation only.

    Two disjoint rules, because the rig's pliers have two differently-coloured
    grips and a single hue window cannot separate either from foliage alone:

    - **red**: a narrow true-red hue window plus high saturation. Narrow on
      purpose -- widening it toward orange starts eating yellow-green leaves,
      which sit at hue ~30-45 and are a large fraction of the plant.
    - **yellow**: hue overlaps yellow-green foliage badly, so brightness does
      the separating instead. The plastic is near-white-hot (V>=215) while
      leaves, even lit ones, stay below that.

    Measured on DSC_0009: this claims under 1% of the true plant mask while
    still finding 40k-135k holder pixels per frame. An earlier single rule
    (any warm hue, S>=120) took 20% of the plant with it.

    The grey metal jaws are deliberately not matched -- they have no chromatic
    signature at all. SAM2 picks them up from the handle seed, because the
    pliers are one connected object.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # OpenCV hue is 0..179; red wraps, so it needs both ends of the range.
    red = ((hue <= 15) | (hue >= 165)) & (sat >= 120)
    yellow = (hue <= 45) & (sat >= 80) & (val >= 215)
    return (red | yellow) & foreground_gate(bgr)


def plant_color_mask(bgr: np.ndarray, exg_threshold: float = 0.06) -> np.ndarray:
    """Rough "this is foliage" mask, for prompt derivation only.

    Subtracting the holder is not optional: the pliers' *yellow* handle is
    green-dominant in RGB, so it clears any Excess-Green threshold just as
    easily as a leaf does. Without this the auto-derived plant prompt lands
    on the pliers -- observed on both DSC_0009 and DSC_0010 -- and SAM2 then
    faithfully tracks the wrong object for the entire sequence.
    """
    return (excess_green(bgr) > exg_threshold) & foreground_gate(bgr) & ~holder_color_mask(bgr)


@dataclass
class Prompts:
    plant: List[Tuple[int, int]] = field(default_factory=list)
    holder: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"plant": [list(p) for p in self.plant], "holder": [list(p) for p in self.holder]}


def derive_prompts(bgr: np.ndarray, n_points: int = 3) -> Prompts:
    """Auto-derive plant and holder click points from one frame."""
    plant = _largest_component(plant_color_mask(bgr))
    holder = _largest_component(holder_color_mask(bgr))
    return Prompts(plant=_interior_points(plant, n_points), holder=_interior_points(holder, n_points))


# --------------------------------------------------------------------------
# Region of interest
# --------------------------------------------------------------------------


@dataclass
class TrackingCrop:
    """A per-frame crop window that follows the plant around the turntable.

    All windows share one size, so the cropped sequence is a well-formed
    video for SAM2; only the offset changes per frame.
    """

    boxes: List[Tuple[int, int, int, int]]  # (x0, y0, x1, y1) per frame
    width: int
    height: int
    frame_width: int
    frame_height: int

    def to_dict(self) -> dict:
        return {
            "boxes": [list(b) for b in self.boxes],
            "crop_width": self.width,
            "crop_height": self.height,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }


def plant_centroids(frame_paths: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Per-frame plant centroid and bbox size, from the color prepass.

    Returns (centroids Nx2, sizes Nx2, (frame_w, frame_h)); frames where the
    plant could not be found get NaN and are interpolated by the caller.
    """
    centroids, sizes = [], []
    frame_w = frame_h = 0

    for path in frame_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            centroids.append([np.nan, np.nan])
            sizes.append([np.nan, np.nan])
            continue
        frame_h, frame_w = bgr.shape[:2]
        mask = _significant_components(plant_color_mask(bgr))
        if not mask.any():
            centroids.append([np.nan, np.nan])
            sizes.append([np.nan, np.nan])
            continue
        ys, xs = np.nonzero(mask)
        centroids.append([xs.mean(), ys.mean()])
        sizes.append([xs.max() - xs.min(), ys.max() - ys.min()])

    return np.array(centroids, float), np.array(sizes, float), (frame_w, frame_h)


def _interpolate_nans(values: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation over the frame index."""
    out = values.copy()
    for col in range(out.shape[1]):
        column = out[:, col]
        bad = np.isnan(column)
        if bad.all():
            column[:] = 0.0
        elif bad.any():
            idx = np.arange(len(column))
            column[bad] = np.interp(idx[bad], idx[~bad], column[~bad])
    return out


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Circular moving average -- the sequence is one full turntable rotation,
    so frame 0 and frame N-1 are neighbours and wrap-around smoothing avoids
    an artificial discontinuity at the seam.
    """
    if window < 3 or len(values) < window:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window) / window
    padded = np.concatenate([values[-pad:], values, values[:pad]], axis=0)
    return np.stack(
        [np.convolve(padded[:, c], kernel, mode="valid") for c in range(values.shape[1])],
        axis=1,
    )


def solve_tracking_crop(
    frame_paths: Sequence[Path],
    padding_fraction: float = 0.45,
    smooth_window: int = 9,
) -> TrackingCrop:
    """Solve a plant-following crop window for every frame.

    SAM2 resizes whatever it is given to a square 1024x1024, so on a
    1920x1080 frame a seedling spanning ~500 px arrives at ~266 px and its
    petioles at 2-4 px -- below what any segmenter reliably holds onto.

    A *fixed* crop does not help here: the specimen is clamped off-axis, so
    it orbits the turntable centre and its union bounding box over a full
    rotation covers 1639 of 1920 px (measured on DSC_0009) -- essentially the
    whole frame. Tracking the plant instead keeps the crop tight, which both
    recovers resolution and stabilises the subject in frame, making SAM2's
    temporal memory attention an easier problem rather than a harder one.

    The centroid trajectory is smoothed before use: an orbit is smooth by
    construction, so smoothing suppresses the occasional bad color-prepass
    frame that would otherwise jerk the crop and break tracking.

    `padding_fraction` is generous by default so the crop also catches the
    clamp jaws gripping the stem -- P2 has to segment the holder where it
    touches the plant, which is precisely where confusing the two matters.
    """
    centroids, sizes, (frame_w, frame_h) = plant_centroids(frame_paths)
    if np.isnan(centroids).all():
        raise RuntimeError(
            "Could not locate the plant in any frame -- the color prepass is failing. "
            "Inspect a frame and consider --no-roi with an explicit --plant-point."
        )

    centroids = _smooth(_interpolate_nans(centroids), smooth_window)
    sizes = _interpolate_nans(sizes)

    # One window size for the whole sequence, from the largest the plant ever
    # appears -- a per-frame size would rescale the subject frame to frame,
    # which is exactly the kind of apparent motion that confuses tracking.
    span = float(np.nanmax(sizes))
    side = int(min(min(frame_w, frame_h), span * (1.0 + 2.0 * padding_fraction)))
    side = max(side, 64)

    boxes: List[Tuple[int, int, int, int]] = []
    for cx, cy in centroids:
        x0 = int(round(cx - side / 2))
        y0 = int(round(cy - side / 2))
        # Clamp while preserving the window size, so every crop is identical
        # in shape; a shrunken edge crop would change the subject's scale.
        x0 = max(0, min(x0, frame_w - side))
        y0 = max(0, min(y0, frame_h - side))
        boxes.append((x0, y0, x0 + side, y0 + side))

    return TrackingCrop(boxes=boxes, width=side, height=side, frame_width=frame_w, frame_height=frame_h)


# --------------------------------------------------------------------------
# SAM2 propagation
# --------------------------------------------------------------------------


def _resolve_model_cfg(checkpoint: Path) -> str:
    """Map a checkpoint filename to its packaged hydra config name."""
    stem = checkpoint.stem
    known = {
        "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    if stem not in known:
        raise ValueError(
            f"Unrecognised SAM2 checkpoint name '{stem}'. Expected one of {sorted(known)}."
        )
    return known[stem]


def _write_sam2_inputs(
    frame_paths: Sequence[Path],
    dest: Path,
    crop: Optional[TrackingCrop],
) -> List[Path]:
    """Write cropped, integer-named JPEGs for SAM2's video loader.

    SAM2's `load_video_frames` sorts a directory by `int(filename_stem)`, so
    the frames must literally be named 0.jpg, 1.jpg, ... -- our own
    `frame_0000.jpg` naming raises inside that sort.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob("*.jpg"):
        stale.unlink()

    written: List[Path] = []
    for i, path in enumerate(frame_paths):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read frame {path}")
        if crop is not None:
            x0, y0, x1, y1 = crop.boxes[i]
            bgr = bgr[y0:y1, x0:x1]
        out = dest / f"{i}.jpg"
        cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(out)
    return written


def segment_sequence(
    frames_dir: Union[str, Path],
    out_dir: Union[str, Path],
    checkpoint: Union[str, Path],
    prompts: Optional[Prompts] = None,
    use_roi: bool = True,
    roi_padding: float = 0.18,
    device: str = "cuda",
    offload_to_cpu: bool = True,
    frame_paths: Optional[Sequence[Path]] = None,
) -> dict:
    """Run SAM2 video propagation over `frames_dir` and write P2 artifacts.

    Writes into `out_dir`:
      masks/plant/frame_XXXX.png   binary 0/255, full-frame coordinates
      masks/holder/frame_XXXX.png  binary 0/255, full-frame coordinates
      alpha/frame_XXXX.png         soft plant matte (sigmoid of the logits)
      roi.json                     crop window + frame size
      prompts.json                 the seed points actually used
    """
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    frames_dir, out_dir, checkpoint = Path(frames_dir), Path(out_dir), Path(checkpoint)
    # An explicit list lets one workdir hold several capture passes: SAM2's
    # video propagation needs temporal continuity, so each pass is tracked
    # separately even though the frames share a directory.
    if frame_paths is None:
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    frame_paths = list(frame_paths)
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.jpg found in {frames_dir}")

    first = cv2.imread(str(frame_paths[0]))
    full_h, full_w = first.shape[:2]

    crop = solve_tracking_crop(frame_paths, padding_fraction=roi_padding) if use_roi else None
    if crop is not None:
        print(
            f"  tracking crop {crop.width}x{crop.height} of {full_w}x{full_h} "
            f"-- {1024 / crop.width:.2f}x the effective resolution of full-frame input"
        )

    sam2_dir = out_dir / "sam2_input"
    _write_sam2_inputs(frame_paths, sam2_dir, crop)

    # Prompts are derived on the cropped first frame, so their coordinates
    # already live in the same space SAM2 will see.
    first_cropped = cv2.imread(str(sam2_dir / "0.jpg"))
    if prompts is None:
        prompts = derive_prompts(first_cropped)
    if not prompts.plant:
        raise RuntimeError(
            "No plant prompt point could be derived from the first frame. Pass --plant-point X,Y "
            "explicitly (coordinates are in the cropped ROI unless --no-roi)."
        )
    print(f"  prompts: plant={prompts.plant} holder={prompts.holder or 'none found'}")

    predictor = build_sam2_video_predictor(_resolve_model_cfg(checkpoint), str(checkpoint), device=device)

    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype, enabled=(device == "cuda")):
        state = predictor.init_state(
            video_path=str(sam2_dir),
            offload_video_to_cpu=offload_to_cpu,
            offload_state_to_cpu=offload_to_cpu,
        )
        predictor.reset_state(state)

        for obj_id, points in ((PLANT_ID, prompts.plant), (HOLDER_ID, prompts.holder)):
            if not points:
                continue
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                points=np.array(points, dtype=np.float32),
                labels=np.ones(len(points), dtype=np.int32),  # all positive clicks
            )

        logits_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            logits_by_frame[frame_idx] = {
                int(obj_id): mask_logits[i, 0].float().cpu().numpy()
                for i, obj_id in enumerate(obj_ids)
            }

    # --- write artifacts, pasted back into full-frame coordinates ---
    for sub in ("masks/plant", "masks/holder", "alpha"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    per_frame_stats = []
    for i, path in enumerate(frame_paths):
        logits = logits_by_frame.get(i, {})
        box = crop.boxes[i] if crop is not None else None
        stats = {"frame": path.name, "index": i, "crop_box": list(box) if box else None}

        for obj_id, name in CLASS_NAMES.items():
            crop_logit = logits.get(obj_id)
            if crop_logit is None:
                shape = (crop.height, crop.width) if crop else (full_h, full_w)
                crop_logit = np.full(shape, -1e3, np.float32)
            binary = _paste(crop_logit > 0, box, full_h, full_w).astype(np.uint8) * 255
            cv2.imwrite(str(out_dir / "masks" / name / f"{path.stem}.png"), binary)
            stats[f"{name}_area_px"] = int((binary > 0).sum())

            if obj_id == PLANT_ID:
                soft = _paste(_sigmoid(crop_logit), box, full_h, full_w, fill=0.0)
                cv2.imwrite(str(out_dir / "alpha" / f"{path.stem}.png"), (soft * 255).astype(np.uint8))

        per_frame_stats.append(stats)

    with open(out_dir / "crop.json", "w") as f:
        json.dump(
            crop.to_dict() if crop else {"boxes": None, "frame_width": full_w, "frame_height": full_h},
            f,
            indent=2,
        )
    with open(out_dir / "prompts.json", "w") as f:
        json.dump(prompts.to_dict(), f, indent=2)

    return {
        "per_frame": per_frame_stats,
        "crop": crop.to_dict() if crop else None,
        "frame_width": full_w,
        "frame_height": full_h,
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _paste(
    crop_array: np.ndarray,
    box: Optional[Tuple[int, int, int, int]],
    full_h: int,
    full_w: int,
    fill: float = 0.0,
) -> np.ndarray:
    """Place a crop-space array back into a full-frame canvas.

    Everything downstream (P3 poses, P4 carving) works in full-frame pixel
    coordinates because that is what the camera intrinsics describe, so the
    per-frame crop must never leak past this function.
    """
    if box is None:
        return crop_array
    canvas = np.full((full_h, full_w), fill, dtype=np.float32)
    x0, y0, x1, y1 = box
    canvas[y0:y1, x0:x1] = crop_array.astype(np.float32)
    return canvas
