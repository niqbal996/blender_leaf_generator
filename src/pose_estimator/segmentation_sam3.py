"""P2 backend: the same masks as `segmentation.py`, from noun phrases.

SAM2 needs to be told *where* the plant is, in pixels, on the first frame of
every capture pass. That click is the step that fails: on sugarbeet_4 it left
3 of 12 plant prompts outside the mask they seeded and lost the root in 85 of
85 frames, and the colour prepass that places the tracking crop when nobody
clicks picks the largest green-dominant blob -- which an amber plier grip is.

SAM3 is told *what* the plant is instead, as text, and finds it itself. This
module produces byte-identical artifacts to `segment_sequence`, so P3, P4a,
P4b, P4c and P5 cannot tell which backend ran:

    masks/plant/frame_XXXX.png   binary 0/255, full-frame coordinates
    masks/holder/frame_XXXX.png  binary 0/255, full-frame coordinates
    masks/root/frame_XXXX.png    binary 0/255, when a root prompt matched
    masks/stem/frame_XXXX.png    plant minus leaf, holder and root -- see below
    alpha/frame_XXXX.png         soft plant matte
    crop.json                    the tracking window
    prompts.json                 the phrases used, and what each one found

Three measured facts shape the design, all from `scripts/sam3_p2_prompt_lab.py`
on sugarbeet_4 and vogelmeere_1:

* **The phrases must be unioned, not chosen between.** Each drops out on its
  own schedule -- `plant` held 0.94-0.97 coverage but its instance vanished on
  5 of 15 frames, and `leaf` climbed to 0.87-0.96 on exactly those frames.
  Their union scored IoU 0.916 against a SAM2 run where the best single phrase
  scored 0.713. Nothing here picks a winner; every phrase for a class is
  unioned into it.

* **The holder needs the full frame, the plant needs the crop.** `pliers`
  found the tool in 3 of 15 frames when it was shown the plant-tracking crop
  and 15 of 15 when shown whole frames, because the crop clips the handles.
  The plant wants the opposite -- SAM3 works at 1008px, so a seedling in a
  wide shot has no resolution left for petioles. They are separate sessions
  anyway, so each runs at the scale that suits it.

* **One session per class.** A session assigns each instance to exactly one
  prompt, so putting "plant" and "pliers" in one session measures which word
  won rather than what either found.

masks/stem is reached by subtraction rather than by prompting, because the
vocabulary runs out before the tissue does: "petiole" and "branch" matched
nothing at all on gaensefuss_1, while the same pixels sit reliably inside what
the broad phrase `plant` returns (which covered 0.88 of the reference there
against leaf+stem's 0.81). So the stem class is what `plant` claimed and the
leaf instances, the holder and the root did not -- the stem, the petioles and
the crown, which is the tissue a skeleton is actually built along.

It doubles as a check on the leaf prompt. A petiole is thin, so a *large*
connected component in that residual is not a petiole: it is a leaf blade the
leaf prompt lost on that frame, and the run says so rather than quietly
filing it as stem.

The leaf instance ids SAM3 carries across frames are written out too, as
masks/leaf_instances/<id>/. Nothing in P1-P6 reads them yet; they are the 2D
half of the leaf correspondence that P4c and P5 currently reconstruct in 3D
(see `dino.leaf_instances`, which merges any two leaves that touch).

Needs transformers >= 5.0 for the SAM3 video classes, and the gated
`facebook/sam3` weights -- see the README.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from pose_estimator.segmentation import ROOT_MASK_DIR, TrackingCrop, _paste

# Wording that measured well on this rig. Every phrase for a class is unioned
# into that class, so adding one can only ever add pixels -- which is why the
# defaults are broad rather than careful.
DEFAULT_PLANT_PROMPTS = ("leaf", "stem", "plant")
# "metal clamp", "metal gripper" and "plant pot" were each tried and returned
# nothing at all on two captures; they are left out rather than left in as
# dead weight.
DEFAULT_HOLDER_PROMPTS = ("pliers", "tool")
DEFAULT_ROOT_PROMPTS = ("root",)
# What the crop window is solved from. Deliberately the broad term: this pass
# only has to say where the subject is, and "plant" held a single instance on
# 4 of 4 full frames at a median box IoU of 0.853 against P2's own bounding
# box, which is far more than a padded crop needs.
DEFAULT_CROP_PROMPT = "plant"
# Every frame, not a subsample. Interpolating the window between located
# frames is a real approximation -- the orbit is smooth but the plant's
# apparent size is not, and a window solved on frame 0 and frame 8 sits
# slightly wrong on the six between. Locating on all of them costs one extra
# full-resolution pass and removes the approximation; raise it only if that
# pass is what does not fit.
DEFAULT_CROP_STRIDE = 1
# Tighter than the SAM2 path's 0.45, deliberately. That margin exists so the
# clamp jaws land inside the same window as the plant, and here they do not
# have to: the holder is prompted on full frames precisely because the crop
# clips the tool. What is left to pay for is the error in SAM3's own
# localisation box, which measured a median IoU of 0.853 against P2's plant
# bounding box -- a fifth of the subject's size covers that comfortably.
#
# It is worth being tight. SAM3 resizes its input to 1008px and its mask head
# emits 288x288, so on a 1280px crop one mask pixel is 4.4 real pixels: the
# padding is paid for in the resolution of every petiole in the frame.
DEFAULT_ROI_PADDING = 0.20

# The instance ids of this prompt are kept per-instance on disk. Only "leaf"
# earns that: it is the one class whose instances are the thing downstream
# cannot currently recover.
INSTANCE_PROMPT = "leaf"

# Plant tissue that the leaf prompt did not claim -- the stem, the petioles
# and the crown -- written as its own class. It is a *subtraction*, not a
# prompt: "petiole" and "branch" match nothing on these plants, while the
# same pixels are reliably inside what `plant` returns.
STEM_MASK_DIR = "stem"
# Specks to drop from that subtraction. A boundary disagreement of one or two
# pixels between the broad phrase and the leaf instances leaves a rind of
# pepper noise along every blade edge, and it is not tissue. Whichever of the
# two thresholds is larger wins, so a big plant is cleaned proportionally and
# a small one is not scrubbed away.
MIN_RESIDUAL_AREA = 40
RESIDUAL_SPECK_FRACTION = 0.0008
# A residual component this large a share of the plant mask is not a petiole.
# Petioles are thin; on the captures measured here the whole residual runs a
# few percent of the plant. One component at a fifth of it is a leaf blade the
# leaf prompt lost on that frame -- the sugarbeet_4 frame-4 failure exactly.
MISSED_LEAF_FRACTION = 0.20


@dataclass
class Sam3Prompts:
    """The phrases for one capture pass, by P2 class."""

    plant: List[str] = field(default_factory=lambda: list(DEFAULT_PLANT_PROMPTS))
    holder: List[str] = field(default_factory=lambda: list(DEFAULT_HOLDER_PROMPTS))
    root: List[str] = field(default_factory=lambda: list(DEFAULT_ROOT_PROMPTS))
    crop: str = DEFAULT_CROP_PROMPT

    def to_dict(self) -> dict:
        return {"backend": "sam3", "plant": self.plant, "holder": self.holder,
                "root": self.root, "crop": self.crop}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def load_sam3(model_name: str = "facebook/sam3", device: str = "cuda"):
    """The processor and model, with the two failures that actually happen
    reported as instructions rather than tracebacks."""
    try:
        from transformers import Sam3VideoModel, Sam3VideoProcessor
    except ImportError as exc:
        raise SystemExit(
            f"\nSAM3's video classes are not in this transformers ({exc}).\n"
            "They landed in transformers 5.0:  pip install -e \".[sam3]\"\n"
            "See the README section 'SAM3' before installing it beside the pipeline.\n")
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


def _frames_as_rgb(frame_paths: Sequence[Path],
                   crop: Optional[TrackingCrop]) -> List[np.ndarray]:
    """Frames in the one size a SAM3 session accepts, cropped if asked."""
    video = []
    for i, path in enumerate(frame_paths):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"could not read {path}")
        if crop is not None:
            x0, y0, x1, y1 = crop.boxes[i]
            bgr = bgr[y0:y1, x0:x1]
        video.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return video


def _run_session(processor, model, torch, video, phrases: Sequence[str], device: str,
                 want_soft: bool = False, label: str = "tracking"):
    """Track every instance of every phrase over one clip.

    Returns `(per_frame, soft)` where `per_frame[i][phrase]` is
    `{obj_id: bool mask}` and `soft[i]` is the per-pixel maximum probability
    over every instance of every phrase on that frame, or None.

    The soft map is why this does not simply call `postprocess_outputs`: that
    interpolates the low-res masks and then thresholds them at `> 0`, handing
    back `torch.bool`. P6 trains a 2D-Gaussian alpha against p2/alpha, so the
    matte has to be the probability, and the logits only exist one level down
    in `model_outputs["obj_id_to_mask"]`.
    """
    import torch.nn.functional as F

    session = processor.init_video_session(
        video=video, inference_device=device,
        processing_device="cpu", video_storage_device="cpu",
    )
    processor.add_text_prompt(session, list(phrases))

    height, width = video[0].shape[:2]
    per_frame: Dict[int, Dict[str, Dict[int, np.ndarray]]] = {}
    soft: Dict[int, np.ndarray] = {}

    # Our own bar rather than the model's, so it can say *which* of the four
    # sessions a pass runs is on screen. Without that the run prints a header
    # and then nothing for minutes, which is indistinguishable from a hang --
    # and the per-frame cost climbs through a session, so the early frames do
    # not predict the late ones either.
    from tqdm.auto import tqdm

    with torch.inference_mode():
        frames = model.propagate_in_video_iterator(
            inference_session=session, show_progress_bar=False)
        for model_outputs in tqdm(frames, total=len(video), desc=f"    {label}",
                                  unit="frame", leave=True, dynamic_ncols=True):
            index = model_outputs.frame_idx
            result = processor.postprocess_outputs(session, model_outputs)

            masks = result["masks"]
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            masks = np.asarray(masks)
            if masks.ndim == 4:                       # (n, 1, H, W)
                masks = masks[:, 0]
            if masks.dtype != bool:
                masks = masks > 0.5

            phrase_of = {}
            for phrase, ids in (result.get("prompt_to_obj_ids") or {}).items():
                for obj_id in ids:
                    phrase_of[int(obj_id)] = phrase

            frame: Dict[str, Dict[int, np.ndarray]] = {}
            for k, obj_id in enumerate(int(o) for o in result["object_ids"]):
                mask = masks[k].astype(bool)
                if not mask.any():
                    continue
                # An instance the model did not attribute is charged to the
                # first phrase: with one phrase that is exact, and with
                # several it keeps the pixels rather than dropping them, which
                # is what matters for a union.
                frame.setdefault(phrase_of.get(obj_id, phrases[0]), {})[obj_id] = mask
            per_frame[index] = frame

            if want_soft:
                low = model_outputs["obj_id_to_mask"]
                if low:
                    stacked = torch.cat([m.reshape(1, 1, *m.shape[-2:]).float()
                                         for m in low.values()], dim=0)
                    probs = torch.sigmoid(F.interpolate(
                        stacked, size=(height, width), mode="bilinear", align_corners=False))
                    soft[index] = probs.max(dim=0).values[0].cpu().numpy()
                else:
                    soft[index] = np.zeros((height, width), np.float32)

    del session
    if device == "cuda":
        torch.cuda.empty_cache()
    return per_frame, (soft if want_soft else None)


def _drop_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Components below `min_area` removed, in one pass."""
    if not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def _largest_component_area(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0


def _union(frame: Dict[str, Dict[int, np.ndarray]], phrases: Sequence[str],
           shape) -> np.ndarray:
    out = np.zeros(shape, bool)
    for phrase in phrases:
        for mask in frame.get(phrase, {}).values():
            out |= mask
    return out


# --------------------------------------------------------------------------
# the crop
# --------------------------------------------------------------------------


def solve_crop_with_sam3(processor, model, torch, frame_paths: Sequence[Path],
                         phrase: str, stride: int, padding: float,
                         device: str) -> TrackingCrop:
    """The plant-following window, located by SAM3 on full frames.

    The colour prepass this replaces is the reason the backend exists: it
    picks the largest green-dominant blob and an amber plier grip qualifies.
    Running on every `stride`th frame is enough -- an orbit is smooth, so the
    frames between interpolate -- and the window is one fixed size for the
    sequence, as in `solve_tracking_crop`: the frames are stacked into a
    single tensor so they must agree, and a per-frame size would rescale the
    subject frame to frame, which is the apparent motion that makes tracking
    harder rather than easier.
    """
    sampled = list(range(0, len(frame_paths), max(1, stride)))
    if sampled[-1] != len(frame_paths) - 1:
        sampled.append(len(frame_paths) - 1)     # anchor both ends, never extrapolate

    video = _frames_as_rgb([frame_paths[i] for i in sampled], None)
    full_h, full_w = video[0].shape[:2]
    per_frame, _ = _run_session(processor, model, torch, video, [phrase], device,
                                label=f"locating {phrase!r}")

    centres = np.full((len(frame_paths), 2), np.nan)
    spans: List[int] = []
    for k, frame_index in enumerate(sampled):
        found = _union(per_frame.get(k, {}), [phrase], (full_h, full_w))
        if not found.any():
            continue
        ys, xs = np.nonzero(found)
        centres[frame_index] = ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0)
        spans.append(int(max(xs.max() - xs.min(), ys.max() - ys.min())))

    if not spans:
        raise RuntimeError(
            f"SAM3 found no {phrase!r} in any of {len(sampled)} full frames, so there is "
            "nothing to centre a crop on. Try --sam3-crop-prompt with different wording, "
            "or --no-roi to segment full frames.")
    print(f"    located {phrase!r} in {len(spans)}/{len(sampled)} sampled frames")

    for axis in (0, 1):
        column = centres[:, axis]
        missing = np.isnan(column)
        if missing.all():
            column[:] = (full_w if axis == 0 else full_h) / 2.0
        elif missing.any():
            column[missing] = np.interp(np.flatnonzero(missing),
                                        np.flatnonzero(~missing), column[~missing])

    side = int(min(min(full_w, full_h), max(spans) * (1.0 + 2.0 * padding)))
    side = max(side, 64)
    boxes = []
    for cx, cy in centres:
        x0 = max(0, min(int(round(cx - side / 2)), full_w - side))
        y0 = max(0, min(int(round(cy - side / 2)), full_h - side))
        boxes.append((x0, y0, x0 + side, y0 + side))
    return TrackingCrop(boxes=boxes, width=side, height=side,
                        frame_width=full_w, frame_height=full_h)


# --------------------------------------------------------------------------


def segment_sequence_sam3(
    frames_dir: Union[str, Path],
    out_dir: Union[str, Path],
    prompts: Optional[Sam3Prompts] = None,
    use_roi: bool = True,
    roi_padding: float = DEFAULT_ROI_PADDING,
    device: str = "cuda",
    frame_paths: Optional[Sequence[Path]] = None,
    model_name: str = "facebook/sam3",
    crop_stride: int = DEFAULT_CROP_STRIDE,
    save_instances: bool = True,
    instance_prefix: str = "",
    session: Optional[Tuple] = None,
) -> dict:
    """Segment one capture pass with SAM3. Same artifacts as `segment_sequence`.

    `session` is an already-loaded `(processor, model)`; P2 runs one session
    per capture pass and the weights are 3.3 GB, so the caller loads them once
    and passes them in.

    `instance_prefix` namespaces the leaf-instance folders, and a multi-pass
    capture must set it. Each pass is its own SAM3 session and its object ids
    restart from zero, so writing them all to masks/leaf_instances/<id>/ files
    three unrelated leaves under one id -- measured on gaensefuss_1, whose
    three passes are 45, 40 and 50 frames and where id 16 accumulated 130.
    """
    import torch

    frames_dir, out_dir = Path(frames_dir), Path(out_dir)
    if frame_paths is None:
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    frame_paths = list(frame_paths)
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.jpg found in {frames_dir}")
    prompts = prompts or Sam3Prompts()

    processor, model = session if session is not None else load_sam3(model_name, device)

    first = cv2.imread(str(frame_paths[0]))
    full_h, full_w = first.shape[:2]

    # Said before any of it starts. A pass is several full propagations, not
    # one, and the count is the difference between a long wait that is
    # understood and a long wait that looks like a failure.
    sessions = ["plant"] + (["root"] if prompts.root else []) \
        + (["holder"] if prompts.holder else [])
    localising = (f"1 localisation over {len(frame_paths)} full frames + "
                  if use_roi and crop_stride == 1 else
                  f"1 localisation over ~{len(frame_paths) // max(crop_stride, 1) + 1} "
                  f"full frames + " if use_roi else "")
    print(f"  {localising}{len(sessions)} tracking session(s) "
          f"({', '.join(sessions)}) over {len(frame_paths)} frames each")

    crop = None
    if use_roi:
        where = "every frame" if crop_stride == 1 else f"every {crop_stride}th frame"
        print(f"  locating the subject with {prompts.crop!r} on {where}...")
        crop = solve_crop_with_sam3(processor, model, torch, frame_paths, prompts.crop,
                                    crop_stride, roi_padding, device)
        # Both paths end up resized to SAM3's 1008px square, so the gain from
        # cropping is how much less of the frame is being thrown away:
        # full_w/crop.width, not 1008/crop.width. (segmentation.py prints the
        # latter for SAM2 and understates the crop for it.)
        print(f"  tracking crop {crop.width}x{crop.height} of {full_w}x{full_h} "
              f"-- {full_w / crop.width:.2f}x the effective resolution of full-frame input")
    crop_shape = (crop.height, crop.width) if crop else (full_h, full_w)

    # --- the plant, on the crop, where the petioles still have pixels ---
    print(f"  plant   {list(prompts.plant)}")
    cropped = _frames_as_rgb(frame_paths, crop)
    plant_frames, plant_soft = _run_session(processor, model, torch, cropped,
                                            prompts.plant, device, want_soft=True,
                                            label="plant")

    root_frames = {}
    if prompts.root:
        print(f"  root    {list(prompts.root)}")
        root_frames, _ = _run_session(processor, model, torch, cropped, prompts.root,
                                      device, label="root")
    del cropped

    # --- the holder, on full frames, because the crop clips the handles ---
    holder_frames = {}
    if prompts.holder:
        print(f"  holder  {list(prompts.holder)}  (full frames -- the crop clips the tool)")
        full = _frames_as_rgb(frame_paths, None)
        holder_frames, _ = _run_session(processor, model, torch, full, prompts.holder,
                                        device, label="holder (full frames)")
        del full

    # --- write, in full-frame coordinates, exactly as the SAM2 path does ---
    for sub in ("masks/plant", "masks/holder", "alpha"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    per_frame_stats, found_counts = [], {p: 0 for p in
                                         list(prompts.plant) + list(prompts.holder) + list(prompts.root)}
    instance_frames: Dict[int, List[str]] = {}
    missed_leaf_frames: List[str] = []

    for i, path in enumerate(frame_paths):
        box = crop.boxes[i] if crop is not None else None
        stats = {"frame": path.name, "index": i, "crop_box": list(box) if box else None}

        plant_frame = plant_frames.get(i, {})
        for phrase in prompts.plant:
            found_counts[phrase] += 1 if plant_frame.get(phrase) else 0
        plant_crop = _union(plant_frame, prompts.plant, crop_shape)

        # The holder is already full-frame; the plant and root are not.
        holder_full = _union(holder_frames.get(i, {}), prompts.holder, (full_h, full_w))
        for phrase in prompts.holder:
            found_counts[phrase] += 1 if holder_frames.get(i, {}).get(phrase) else 0

        root_crop = None
        if prompts.root:
            root_crop = _union(root_frames.get(i, {}), prompts.root, crop_shape)
            for phrase in prompts.root:
                found_counts[phrase] += 1 if root_frames.get(i, {}).get(phrase) else 0
            if not root_crop.any():
                root_crop = None

        # The root is plant, and is unioned in here exactly as the SAM2 path
        # does it -- it is written separately only because P4c has to decide
        # which tissue is root. Holder pixels win the overlap: the jaws grip
        # the root, so that boundary is where both models are least sure, and
        # a holder pixel reads as occlusion downstream rather than as absence.
        holder_in_crop = (holder_full[box[1]:box[3], box[0]:box[2]] if box is not None
                          else holder_full)
        if root_crop is not None:
            root_crop = root_crop & ~holder_in_crop
            stats["root_area_px"] = int(root_crop.sum())
            plant_crop = plant_crop | root_crop

        plant_full = _paste(plant_crop, box, full_h, full_w).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / "masks" / "plant" / f"{path.stem}.png"), plant_full)
        stats["plant_area_px"] = int((plant_full > 0).sum())

        # What the broad phrase claimed and the organ phrases did not. SAM3
        # segments petioles perfectly well inside "plant" but will not return
        # them for the word "petiole" -- measured on gaensefuss_1, where
        # "petiole" and "branch" matched nothing while `plant` covered 0.88 of
        # the reference against leaf+stem's 0.81. Subtracting is the way to
        # reach tissue the vocabulary cannot name.
        leaf_union = np.zeros(crop_shape, bool)
        for mask in plant_frame.get(INSTANCE_PROMPT, {}).values():
            leaf_union |= mask
        residual = plant_crop & ~leaf_union & ~holder_in_crop
        if root_crop is not None:
            residual = residual & ~root_crop
        residual = _drop_specks(residual, max(MIN_RESIDUAL_AREA,
                                              int(RESIDUAL_SPECK_FRACTION * plant_crop.sum())))
        (out_dir / "masks" / STEM_MASK_DIR).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "masks" / STEM_MASK_DIR / f"{path.stem}.png"),
                    _paste(residual, box, full_h, full_w).astype(np.uint8) * 255)
        stats["stem_area_px"] = int(residual.sum())
        # A petiole is thin. A *large* residual blob is not a petiole, it is a
        # leaf the leaf prompt missed on this frame -- which makes this number
        # the cheapest detector there is for exactly that failure.
        biggest = _largest_component_area(residual)
        stats["stem_largest_component_px"] = biggest
        if plant_crop.any() and biggest > MISSED_LEAF_FRACTION * plant_crop.sum():
            missed_leaf_frames.append(path.stem)

        holder_png = holder_full.astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / "masks" / "holder" / f"{path.stem}.png"), holder_png)
        stats["holder_area_px"] = int((holder_png > 0).sum())

        if root_crop is not None:
            root_out = out_dir / "masks" / ROOT_MASK_DIR
            root_out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root_out / f"{path.stem}.png"),
                        _paste(root_crop, box, full_h, full_w).astype(np.uint8) * 255)

        # The matte, from the probabilities rather than the thresholded mask,
        # and zeroed outside the union so a confident background pixel cannot
        # leak into the class the mask says it is not in.
        soft_crop = plant_soft.get(i)
        if soft_crop is None:
            soft_crop = plant_crop.astype(np.float32)
        soft_crop = np.where(plant_crop, np.maximum(soft_crop, 0.5), 0.0).astype(np.float32)
        soft = _paste(soft_crop, box, full_h, full_w, fill=0.0)
        cv2.imwrite(str(out_dir / "alpha" / f"{path.stem}.png"), (soft * 255).astype(np.uint8))

        if save_instances:
            for obj_id, mask in plant_frame.get(INSTANCE_PROMPT, {}).items():
                folder = out_dir / "masks" / "leaf_instances" / f"{instance_prefix}{obj_id}"
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / f"{path.stem}.png"),
                            _paste(mask, box, full_h, full_w).astype(np.uint8) * 255)
                instance_frames.setdefault(f"{instance_prefix}{obj_id}", []).append(path.stem)

        per_frame_stats.append(stats)

    n = len(frame_paths)
    print("  phrases that matched, by frame count:")
    for phrase, count in found_counts.items():
        note = "" if count else "   <- never matched; drop it or reword it"
        print(f"    {phrase:<14} {count:>4}/{n}{note}")
    if save_instances and instance_frames:
        held = sum(1 for frames in instance_frames.values() if len(frames) == n)
        print(f"  {len(instance_frames)} leaf instances tracked, {held} present in all {n} frames")

    stem_areas = [s["stem_area_px"] for s in per_frame_stats]
    plant_areas = [s["plant_area_px"] for s in per_frame_stats]
    share = (100.0 * sum(stem_areas) / sum(plant_areas)) if sum(plant_areas) else 0.0
    print(f"  masks/{STEM_MASK_DIR}: plant minus leaf/holder/root, median "
          f"{int(np.median(stem_areas))} px, {share:.1f}% of the plant mask")
    if missed_leaf_frames:
        print(f"  WARNING: on {len(missed_leaf_frames)} frame(s) the residual holds one "
              f"component larger than {MISSED_LEAF_FRACTION:.0%} of the plant mask.")
        print(f"    A petiole is not that big -- this is a leaf the {INSTANCE_PROMPT!r} prompt "
              "lost on that frame,")
        print(f"    landing in masks/{STEM_MASK_DIR} as if it were stem. Check "
              f"{', '.join(missed_leaf_frames[:6])}"
              + (" ..." if len(missed_leaf_frames) > 6 else "") + " in p2/diag/.")

    with open(out_dir / "crop.json", "w") as f:
        json.dump(crop.to_dict() if crop else
                  {"boxes": None, "frame_width": full_w, "frame_height": full_h}, f, indent=2)
    payload = prompts.to_dict()
    payload["matched_frames"] = found_counts
    payload["frames"] = n
    payload["stem_residual"] = {
        "share_of_plant_percent": round(share, 2),
        "frames_with_a_leaf_sized_component": missed_leaf_frames,
    }
    if instance_frames:
        payload["leaf_instances"] = {str(k): v for k, v in sorted(instance_frames.items())}
        payload.setdefault("instance_prefix", instance_prefix)
    with open(out_dir / "prompts.json", "w") as f:
        json.dump(payload, f, indent=2)

    return {
        "per_frame": per_frame_stats,
        "crop": crop.to_dict() if crop else None,
        "frame_width": full_w,
        "frame_height": full_h,
    }
