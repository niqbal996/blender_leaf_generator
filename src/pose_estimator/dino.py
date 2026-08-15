"""DINOv3 patch features as the P4c organ classifier.

DINOv3 returns a feature vector per image patch, not a label. Those vectors
are semantically organised without training, so a handful of clicked examples
is enough: every patch is assigned to whichever labelled example it most
resembles.

Why this replaced the earlier classifiers, measured on DSC_0009:

- **2D mask elongation** put leaves at 1.1-5.7 and stems at 3.8-8.3. Wide
  overlap, because elongation describes how an organ *projects* into one view
  -- a leaf seen edge-on is a ribbon, and so is a stem.
- **Depth-Anything surface flatness** put leaves at 0.73-0.91 and stems at
  0.46-0.73. Touching, no gap: a strongly curved lamina fans its normals just
  as a tube does.
- **Text-prompted detection** localised `leaf` well but returned the *whole
  plant* for `stem` and `root`, because a detector grounds a thin part to its
  parent object.
- **DINO features** separate all three cleanly from six clicks, and hold
  across the whole rotation -- verified on frames 57 turns away from the seed.

What they do *not* do is distinguish one leaf from another: every leaf lights
up against a leaf query, so these features carry organ semantics and no
instance identity. Identity comes from the 3D correspondence instead, which
P3 already provides exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Seed:
    label: str
    x: int
    y: int


def parse_seeds(items: Sequence[str]) -> List[Seed]:
    """Parse 'leaf:140,150' strings into seeds."""
    out = []
    for item in items:
        if ":" not in item or "," not in item:
            raise ValueError(f"seed must look like label:x,y -- got {item!r}")
        label, coords = item.split(":", 1)
        x, y = coords.split(",")
        out.append(Seed(label.strip().lower(), int(x), int(y)))
    return out


class DinoBackbone:
    """Patch features from a DINOv3 (or DINOv2) backbone, L2-normalised."""

    def __init__(self, model_id: str, device: str = "cuda", size: int = 896,
                 token: Optional[str] = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.size = size

        kwargs = {"token": token} if token else {}
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
            self.model = AutoModel.from_pretrained(model_id, **kwargs).to(device).eval()
        except OSError as exc:
            if "gated" not in str(exc).lower() and "401" not in str(exc):
                raise
            raise SystemExit(
                f"\n{model_id} is a gated HuggingFace repo.\n"
                f"  1. accept the licence at https://huggingface.co/{model_id}\n"
                f"  2. pass --hf-token <token>, or export HF_TOKEN=<token>\n"
                f"\nUngated alternative that behaves the same way here:\n"
                f"  --dino-model facebook/dinov2-base\n"
            )
        self.patch = int(getattr(self.model.config, "patch_size", 16))
        self.grid = self.size // self.patch

    def features(self, bgr: np.ndarray) -> np.ndarray:
        """(grid*grid, dim) L2-normalised patch features.

        The crop is resized to a fixed square so every frame yields the same
        grid, which is what lets a seed vector from one frame be compared
        against patches in another.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        inputs = self.processor(images=rgb, return_tensors="pt",
                                do_resize=False, do_center_crop=False).to(self.device)
        with self.torch.no_grad():
            tokens = self.model(**inputs).last_hidden_state[0]

        # Drop CLS and any register tokens by inferring how many are not patches.
        prefix = tokens.shape[0] - self.grid * self.grid
        if prefix < 0:
            raise RuntimeError(
                f"{tokens.shape[0]} tokens for a {self.grid}x{self.grid} grid -- "
                "is --dino-size a multiple of the patch size?")
        tokens = tokens[prefix:]
        tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return tokens.cpu().numpy()


def patch_index(x: int, y: int, shape, grid: int) -> int:
    height, width = shape[:2]
    gx = min(int(x / width * grid), grid - 1)
    gy = min(int(y / height * grid), grid - 1)
    return gy * grid + gx


def crop_to_plant(bgr: np.ndarray, plant: np.ndarray, pad: int = 60):
    """Crop to the subject and zero the background.

    Both parts matter. The plant is about 2.5% of a full frame, so uncropped
    it spans two or three patches and organ detail is gone before the model
    sees it. Zeroing the background keeps the pliers and table from
    contributing features that compete with the plant's own.
    """
    ys, xs = np.nonzero(plant)
    if len(xs) == 0:
        return None, None, None
    y0, y1 = max(0, ys.min() - pad), min(bgr.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(bgr.shape[1], xs.max() + pad)

    view = bgr[y0:y1, x0:x1].copy()
    crop_plant = plant[y0:y1, x0:x1]
    view[~crop_plant] = 0
    return view, crop_plant, (y0, y1, x0, x1)


def build_seed_vectors(
    backbone: DinoBackbone, seed_bgr: np.ndarray, seed_plant: np.ndarray,
    seeds: Sequence[Seed], pad: int = 60,
) -> Tuple[np.ndarray, List[str]]:
    """Feature vector per seed click, kept individually.

    Individually, not averaged into one prototype per class. Averaging was
    tried first and is actively harmful: a class given several
    different-looking examples (three leaves at three angles) averages toward
    a generic direction resembling no individual leaf, while a class with one
    tight example keeps its full similarity and wins patches it should not.
    Nearest-individual-seed keeps each example's specificity.
    """
    view, plant, _ = crop_to_plant(seed_bgr, seed_plant, pad)
    if view is None:
        raise RuntimeError("seed frame has an empty plant mask")

    features = backbone.features(view)
    vectors, labels = [], []
    for seed in seeds:
        vectors.append(features[patch_index(seed.x, seed.y, view.shape, backbone.grid)])
        labels.append(seed.label)
    return np.stack(vectors, axis=1), labels


def build_seed_vectors_multi(
    backbone: DinoBackbone,
    frames_dir,
    mask_dir,
    seeds_by_frame: Dict[str, Sequence[Seed]],
    pad: int = 60,
) -> Tuple[np.ndarray, List[str]]:
    """Seed vectors pooled from several frames.

    One frame shows each organ from one angle, and a leaf turned edge-on
    barely resembles the same leaf face-on. Seeding from a few frames around
    the orbit gives each class several genuinely different examples, which
    costs nothing here precisely because the vectors are kept individually --
    pooling is concatenation, not averaging, so a new example can only add
    coverage and never dilute an existing one.
    """
    from pathlib import Path

    frames_dir, mask_dir = Path(frames_dir), Path(mask_dir)
    all_vectors, all_labels = [], []

    for stem in sorted(seeds_by_frame):
        seeds = seeds_by_frame[stem]
        if not seeds:
            continue
        bgr = cv2.imread(str(frames_dir / f"{stem}.jpg"))
        mask = cv2.imread(str(mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise SystemExit(f"seed frame {stem} or its mask is missing")
        vectors, labels = build_seed_vectors(backbone, bgr, mask > 127, seeds, pad)
        all_vectors.append(vectors)
        all_labels.extend(labels)

    if not all_vectors:
        raise SystemExit("no seeds to build vectors from")
    return np.concatenate(all_vectors, axis=1), all_labels


def classify_frame(
    backbone: DinoBackbone,
    bgr: np.ndarray,
    plant: np.ndarray,
    seed_vectors: np.ndarray,
    seed_labels: Sequence[str],
    class_order: Sequence[str],
    pad: int = 60,
) -> Optional[np.ndarray]:
    """Per-pixel class map over a full frame; -1 outside the plant.

    Nearest labelled example per patch, with no confidence threshold. A
    threshold here would be a per-plant knob, and the point of this stage is
    to remove those; a patch's evidence is instead carried forward as the
    multi-view vote count.
    """
    view, crop_plant, box = crop_to_plant(bgr, plant, pad)
    if view is None:
        return None

    features = backbone.features(view)
    nearest = (features @ seed_vectors).argmax(axis=1)
    class_ids = np.array([class_order.index(seed_labels[i]) for i in nearest], np.int8)
    grid_map = class_ids.reshape(backbone.grid, backbone.grid)

    resized = cv2.resize(grid_map, (view.shape[1], view.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    resized[~crop_plant] = -1

    full = np.full(bgr.shape[:2], -1, np.int8)
    y0, y1, x0, x1 = box
    full[y0:y1, x0:x1] = resized
    return full


def leaf_instances(points: np.ndarray, radius: float, min_points: int = 50) -> np.ndarray:
    """Split leaf-labelled points into instances by 3D connectivity.

    Cheap stand-in for a structure backend, and honest about what it can do:
    two leaves that physically touch, or overlap within `radius`, merge into
    one instance. It exists so the point cloud can be coloured per leaf for
    inspection, not as a replacement for P5.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if len(points) == 0:
        return np.zeros(0, np.int32)

    pairs = cKDTree(points).query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(len(points), dtype=np.int32)

    graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(points), len(points)))
    count, labels = connected_components(graph, directed=False)

    sizes = np.bincount(labels, minlength=count)
    keep = np.argsort(-sizes)
    remap = np.full(count, -1, np.int32)
    next_id = 0
    for component in keep:
        if sizes[component] >= min_points:
            remap[component] = next_id
            next_id += 1
    return remap[labels]
