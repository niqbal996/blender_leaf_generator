"""COLMAP export for MapAnything, with its quality controls actually turned on.

MapAnything's own `scripts/demo_colmap.py` calls `model.infer(...)` with
`apply_mask=True, mask_edges=True` and nothing else, which leaves two of its
noise controls at their permissive defaults:

* `apply_confidence_mask` (default False) -- no confidence filtering at all.
* `use_multiview_confidence` (default False) -- per-pixel learned confidence
  instead of confidence derived from *agreement between views*.

On thistle3 the default export put only 48% of its points inside the P2 plant
silhouette while covering 99.8% of the mask area: it reconstructs the whole
plant and a haze of everything else. The second flag is the one that speaks
to that, since a point no other view corroborates is what the haze is made
of. This script is the upstream exporter with those exposed, reusing their
own `export_predictions_to_colmap` so the model format stays theirs.

`--plant-masks` goes further: P2 already knows which pixels are the subject,
so there is no reason to reconstruct a masked-out background at all.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plant-masks", default=None,
                        help="P2 plant masks; geometry outside them is discarded before export")
    parser.add_argument("--apache", action="store_true",
                        help="Use the Apache-2.0 checkpoint instead of the CC-BY-NC one")
    parser.add_argument("--resize-mode", default="fixed_mapping",
                        choices=["fixed_mapping", "longest_side", "square", "fixed_size"],
                        help="fixed_mapping is what the model was trained with; longest_side "
                             "with --size raises resolution at the cost of leaving that regime")
    parser.add_argument("--size", type=int, default=None,
                        help="Required by longest_side/square; ignored by fixed_mapping")
    parser.add_argument("--confidence-percentile", type=float, default=10,
                        help="Drop this bottom percentile of confidence")
    parser.add_argument("--no-confidence-mask", action="store_true",
                        help="Turn confidence filtering off again (the upstream default)")
    parser.add_argument("--no-multiview-confidence", action="store_true",
                        help="Use learned per-pixel confidence instead of cross-view agreement")
    parser.add_argument("--memory-efficient", action="store_true",
                        help="Trade speed for peak memory; unnecessary on a large GPU")
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--voxel-fraction", type=float, default=0.002,
                        help="Export voxel size as a fraction of scene extent. Upstream uses "
                             "0.01; a plant's petioles and leaf tips need finer than that")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    paths = sorted(glob.glob(os.path.join(args.images_dir, "*")))
    if not paths:
        raise SystemExit(f"no images in {args.images_dir}")
    names = [os.path.basename(path) for path in paths]

    load_kwargs = {"resize_mode": args.resize_mode}
    if args.size is not None:
        load_kwargs["size"] = args.size
    views = load_images(paths, **load_kwargs)
    print(f"Loaded {len(views)} views ({args.resize_mode}"
          f"{f', size {args.size}' if args.size else ''})")

    name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained(name).to(device)
    print(f"Loaded {name} on {device}")

    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=args.memory_efficient,
            minibatch_size=args.minibatch_size,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            # The two the upstream exporter leaves off.
            apply_confidence_mask=not args.no_confidence_mask,
            confidence_percentile=args.confidence_percentile,
            use_multiview_confidence=not args.no_multiview_confidence,
        )
    print("Inference complete")

    if args.plant_masks:
        import cv2

        # export_predictions_to_colmap reads pred["mask"] as (B, H, W, 1), so
        # intersecting the P2 silhouette into it is all that is needed to keep
        # background geometry out of the exported model.
        kept = dropped = 0
        for view_index, prediction in enumerate(outputs):
            if "mask" not in prediction:
                print("  predictions carry no 'mask' field; skipping plant masking")
                break
            current = prediction["mask"]
            height, width = int(current.shape[1]), int(current.shape[2])
            raw = cv2.imread(str(Path(args.plant_masks) / f"{Path(names[view_index]).stem}.png"),
                             cv2.IMREAD_GRAYSCALE)
            if raw is None:
                continue
            plant = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST) > 127
            plant_t = torch.from_numpy(plant).to(current.device)[None, ..., None]
            before = int(current.sum())
            prediction["mask"] = current & plant_t
            kept += int(prediction["mask"].sum())
            dropped += before - int(prediction["mask"].sum())
        print(f"P2 plant masks applied: kept {kept} pixels, dropped {dropped} outside the plant")

    from mapanything.utils.colmap_export import export_predictions_to_colmap

    os.makedirs(args.output_dir, exist_ok=True)
    export_predictions_to_colmap(outputs=outputs, processed_views=views,
                                 image_names=names, output_dir=args.output_dir,
                                 voxel_fraction=args.voxel_fraction)
    Path(args.output_dir, "mapanything_export.json").write_text(json.dumps({
        "model": name, "resize_mode": args.resize_mode, "size": args.size,
        "confidence_mask": not args.no_confidence_mask,
        "confidence_percentile": args.confidence_percentile,
        "multiview_confidence": not args.no_multiview_confidence,
        "voxel_fraction": args.voxel_fraction,
        "plant_masks_applied": bool(args.plant_masks),
    }, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
