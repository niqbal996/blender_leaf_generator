"""P4c stage 1 -- decide what each pixel is, one photograph at a time.

Splitting this out from the 3D fusion is the point. Before, a disappointing
organ label could have come from the 2D classifier or from the multi-view
voting, and there was no artifact in between to look at, so neither half could
be measured on its own. Now stage 1 writes a class map per frame to disk, and:

- the classifier can be scored directly against a few hand-labelled frames,
- the fusion can be fed synthetic perfect maps to check it returns them,
- a new classifier is a drop-in that never touches the voting code.

**Output contract**, identical for every backend: one PNG per frame, uint8,
same size as the frame. 0 means "not the plant"; any other value v means
`class_order[v - 1]`. The class order is recorded once in `p4c/classify.json`
so the maps stay readable without guessing.

Two backends ship, and they disagree in useful ways:

- ``dino`` -- every image patch is assigned to whichever hand-clicked example
  it most resembles in DINOv3 feature space. Open-vocabulary: the classes are
  whatever the seeds were labelled. Weak on thin structures, because a patch
  is ~11 source pixels wide and a petiole is narrower than that.
- ``sam`` -- SAM2 proposes object masks and each is assigned to leaf or stem
  by shape. Fixed two-class vocabulary, but the boundaries are real object
  boundaries rather than a patch grid, so it is the better of the two wherever
  an organ edge matters.

Neither is right everywhere, which is why both are kept.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

NOT_PLANT = -1


# --------------------------------------------------------------------------
# Class-map artifacts
# --------------------------------------------------------------------------


def save_class_map(out_dir: Union[str, Path], stem: str, class_map: np.ndarray) -> Path:
    """Write one frame's class map, shifting -1 (not plant) to 0."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    cv2.imwrite(str(path), (np.asarray(class_map, np.int16) + 1).astype(np.uint8))
    return path


def load_class_map(out_dir: Union[str, Path], stem: str) -> Optional[np.ndarray]:
    """Read one frame's class map back as int8, -1 outside the plant."""
    path = Path(out_dir) / f"{stem}.png"
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None
    return raw.astype(np.int16) - 1


def write_manifest(p4c_dir: Union[str, Path], backend: str, class_order: Sequence[str],
                   extra: Optional[dict] = None) -> Path:
    """Record what produced the class maps, so stage 2 need not be told again."""
    p4c_dir = Path(p4c_dir)
    p4c_dir.mkdir(parents=True, exist_ok=True)
    path = p4c_dir / "classify.json"
    with open(path, "w") as f:
        json.dump({"backend": backend, "class_order": list(class_order), **(extra or {})}, f, indent=2)
    return path


def read_manifest(p4c_dir: Union[str, Path]) -> dict:
    path = Path(p4c_dir) / "classify.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run pose-classify before pose-fuse "
            "(or run pose-semantic, which does both).")
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class DinoClassifier:
    """Nearest hand-clicked example in DINOv3 patch-feature space.

    Seeds come either from clicks on one frame or from a saved bank. A bank is
    only valid for the model that built it: dinov2-base and dinov3-vitb16 both
    emit 768-dim vectors, so a mismatched bank passes a width check while the
    vectors mean nothing in the other model's space. The model name is stored
    and compared for that reason.
    """

    name = "dino"

    def __init__(self, model_id: str, seed_vectors: np.ndarray, seed_labels: Sequence[str],
                 class_order: Sequence[str], size: int = 896, device: str = "cuda",
                 token: Optional[str] = None):
        from .dino import DinoBackbone

        self.backbone = DinoBackbone(model_id, device=device, size=size, token=token)
        self.seed_vectors = seed_vectors
        self.seed_labels = list(seed_labels)
        self.class_order = list(class_order)
        self.model_id = model_id
        self.size = size

    @property
    def grid(self) -> int:
        return self.backbone.grid

    def classify(self, bgr: np.ndarray, plant_mask: np.ndarray) -> Optional[np.ndarray]:
        from .dino import classify_frame

        return classify_frame(self.backbone, bgr, plant_mask, self.seed_vectors,
                              self.seed_labels, self.class_order)


class SamClassifier:
    """SAM2 automatic masks on the photograph, assigned to leaf or stem by shape.

    Run on the actual frame rather than a render of the point cloud. SAM is
    trained on photographs, and a splatted cloud is nothing like one --
    stippled, unshaded, full of holes. Fed renders it returned a single blob
    and called the whole plant leaf; fed the same plant photographed, it
    separates individual laminae cleanly, including the small ones.

    Shape does the assignment because colour is known to fail on this rig (the
    holder's yellow grip is green-dominant in RGB, and 47% of true plant pixels
    share its hue window). Measured across prototype frames, leaf blades came
    out at 1.1-2.2 major/minor while stem, petiole and root masks ran 4.4-8.3,
    so the threshold sits in the empty gap between two populations rather than
    being tuned to hit a target.
    """

    name = "sam"
    class_order = ["leaf", "stem"]

    def __init__(self, checkpoint: Union[str, Path], device: str = "cuda",
                 points_per_side: int = 48, pred_iou: float = 0.7,
                 stability: float = 0.85, min_region_area: int = 40,
                 whole_plant_fraction: float = 0.85, min_leaf_fraction: float = 0.012,
                 max_leaf_elongation: float = 3.5, pad: int = 60):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        from .segmentation import _resolve_model_cfg

        checkpoint = Path(checkpoint)
        model = build_sam2(_resolve_model_cfg(checkpoint), str(checkpoint), device=device)
        self.generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou,
            stability_score_thresh=stability,
            min_mask_region_area=min_region_area,
        )
        self.whole_plant_fraction = whole_plant_fraction
        self.min_leaf_fraction = min_leaf_fraction
        self.max_leaf_elongation = max_leaf_elongation
        self.pad = pad

    def classify(self, bgr: np.ndarray, plant_mask: np.ndarray) -> Optional[np.ndarray]:
        leaf_id = self.class_order.index("leaf")
        stem_id = self.class_order.index("stem")

        height, width = plant_mask.shape
        ys, xs = np.nonzero(plant_mask)
        if len(xs) == 0:
            return None

        y0, y1 = max(0, ys.min() - self.pad), min(height, ys.max() + self.pad)
        x0, x1 = max(0, xs.min() - self.pad), min(width, xs.max() + self.pad)

        crop = bgr[y0:y1, x0:x1].copy()
        crop_plant = plant_mask[y0:y1, x0:x1]
        # Zero the background so SAM cannot latch onto the holder or the table.
        # The rig already shoots against near-black, so this stays photographic.
        crop[~crop_plant] = 0

        masks = self.generator.generate(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plant_area = int(crop_plant.sum())

        leaves, stems = [], []
        for entry in sorted(masks, key=lambda m: -m["area"]):
            segment = entry["segmentation"]
            inside = segment & crop_plant
            if inside.sum() < 0.5 * segment.sum():
                continue  # mostly outside the plant -- background or holder
            fraction = inside.sum() / max(plant_area, 1)
            # SAM reliably emits one mask covering nearly the whole plant; that
            # is the plant, not an organ. Slivers are too small to be either.
            if fraction >= self.whole_plant_fraction or fraction < self.min_leaf_fraction:
                continue
            # Skip a mask that merely repeats one already taken -- the mask
            # generator emits the same organ at several scales.
            if any((inside & prev).sum() > 0.8 * min(inside.sum(), prev.sum()) for prev in leaves):
                continue
            ys_m, xs_m = np.nonzero(inside)
            centred = np.stack([xs_m, ys_m], axis=1).astype(float)
            centred -= centred.mean(axis=0)
            sv = np.linalg.svd(centred, compute_uv=False) / np.sqrt(len(centred))
            elongation = float(sv[0] / max(sv[1], 1e-9))
            (stems if elongation > self.max_leaf_elongation else leaves).append(inside)

        # Everything inside the silhouette defaults to stem, so pixels no mask
        # claimed are not silently lost; explicit stem masks and then leaves
        # paint over it. Leaf last: where a leaf overlaps the stem mask it is
        # the leaf that is in front.
        crop_classes = np.full(crop_plant.shape, NOT_PLANT, np.int16)
        crop_classes[crop_plant] = stem_id
        for stem in stems:
            crop_classes[stem] = stem_id
        for leaf in leaves:
            crop_classes[leaf] = leaf_id

        class_map = np.full((height, width), NOT_PLANT, np.int16)
        class_map[y0:y1, x0:x1] = crop_classes
        return class_map


def classify_sequence(
    classifier,
    frames_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    out_dir: Union[str, Path],
    frame_stems: Sequence[str],
    progress_every: int = 12,
) -> dict:
    """Run a classifier over every frame and write the class maps.

    Returns per-class pixel totals, which are the cheapest early warning that
    something is off -- a seedling whose 'stem' outweighs its 'leaf' is telling
    you so here, before three more phases build on it.
    """
    frames_dir, mask_dir, out_dir = Path(frames_dir), Path(mask_dir), Path(out_dir)
    totals = np.zeros(len(classifier.class_order), np.int64)
    written = 0

    for n, stem in enumerate(frame_stems):
        bgr = cv2.imread(str(frames_dir / f"{stem}.jpg"))
        plant = cv2.imread(str(mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or plant is None:
            continue
        class_map = classifier.classify(bgr, plant > 127)
        if class_map is None:
            continue
        save_class_map(out_dir, stem, class_map)
        written += 1
        for i in range(len(classifier.class_order)):
            totals[i] += int((class_map == i).sum())
        if (n + 1) % progress_every == 0:
            print(f"    {n + 1}/{len(frame_stems)} frames")

    return {
        "frames_written": written,
        "pixels_per_class": {name: int(totals[i]) for i, name in enumerate(classifier.class_order)},
    }
