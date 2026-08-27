"""P4c stage 1 CLI: classify every frame's pixels into organ classes.

    pose-pick-seeds --workdir runs/plant_9              # click the seeds
    pose-classify   --workdir runs/plant_9 --seeds-file runs/plant_9/p4c/seeds.json

    pose-classify --workdir runs/plant_9 --backend sam \\
        --checkpoint checkpoints/sam2.1_hiera_large.pt      # no seeds at all

Reads <workdir>/p1/frames and p2/masks/plant. Writes into <workdir>/p4c:
    class_maps/frame_XXXX.png   uint8, 0 = not plant, i+1 = class_order[i]
    classify.json               backend, class order, settings
    seed_bank.npz               (dino) the seed vectors, reusable elsewhere
    diag/parts_*.jpg            photograph | classification, side by side

Why this is its own stage: the organ labels that come out of P4c can be wrong
either because the 2D classifier was wrong or because the multi-view fusion
was, and with one combined stage there was no way to tell which. The class
maps written here are the artifact that makes each half measurable -- score
these against a few hand-labelled frames, and separately feed the fusion
synthetic perfect maps to check it returns them.

Three ways to give the dino backend its seeds, in order of preference:

    --seeds-file p4c/seeds.json     clicked with pose-pick-seeds. Coordinates
                                    come from the same crop this stage builds,
                                    and seeds from several frames are pooled.
    --seed-bank <path>.npz          reuse an earlier specimen's vectors, so a
                                    50-plant batch needs no clicking per plant.
    --seeds "leaf:140,150" ...      typed by hand, in the *cropped* frame's
                                    pixel space. Needs --seed-frame too.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pose_estimator.classify2d import (
    DinoClassifier,
    SamClassifier,
    classify_sequence,
    load_class_map,
    write_manifest,
)

CLASS_COLORS = {
    "leaf": (220, 40, 40),        # RGB
    "tiny leaf": (240, 90, 60),
    "stem": (160, 60, 200),
    "petiole": (160, 60, 200),
    "root": (240, 140, 40),
}
SPARE = [(90, 200, 110), (60, 190, 210), (220, 200, 70), (200, 200, 200)]


def build_dino_classifier(
    workdir: Path,
    seeds: Optional[list],
    seed_frame: Optional[int],
    seed_bank: Optional[Path],
    save_seed_bank: Optional[Path],
    dino_model: str,
    dino_size: int,
    hf_token: Optional[str],
    device: str,
    seeds_file: Optional[Path] = None,
):
    from pose_estimator.dino import (
        DinoBackbone,
        build_seed_vectors,
        build_seed_vectors_multi,
        parse_seeds,
    )

    frames_dir = workdir / "p1" / "frames"
    mask_dir = workdir / "p2" / "masks" / "plant"

    # Before the weights, not after: loading DINO takes long enough that a
    # bad path discovered afterwards wastes the wait, and this one used to
    # surface as IsADirectoryError from inside numpy.load.
    if seed_bank is not None:
        from pose_estimator.banks import resolve_bank

        seed_bank = resolve_bank(seed_bank, "--seed-bank")

    print(f"Loading {dino_model} ...")
    backbone = DinoBackbone(dino_model, device=device, size=dino_size, token=hf_token)
    print(f"  patch grid {backbone.grid}x{backbone.grid} "
          f"(~{dino_size / backbone.grid:.0f} px per patch at the crop's scale)")

    if seed_bank is not None:
        # Feature vectors transfer between specimens: they encode "what a leaf
        # looks like", not "where this plant's leaves are". Clicking once and
        # reusing the vectors is the automation -- no per-sample coordinates.
        bank = np.load(seed_bank, allow_pickle=True)
        seed_vectors = bank["vectors"]
        seed_labels = [str(x) for x in bank["labels"]]
        if seed_vectors.shape[0] != backbone.model.config.hidden_size:
            raise SystemExit(
                f"seed bank has {seed_vectors.shape[0]}-dim vectors but {dino_model} "
                f"produces {backbone.model.config.hidden_size}. A bank is only valid "
                "for the model that built it.")
        # Matching width is not enough, and assuming it was silently corrupted a
        # real comparison: dinov2-base and dinov3-vitb16 both emit 768-dim
        # vectors, so a v2 bank loaded into v3 passed the check while the
        # vectors meant nothing in v3's feature space.
        bank_model = str(bank["model"]) if "model" in bank else ""
        if bank_model and bank_model != dino_model:
            raise SystemExit(
                f"seed bank was built with {bank_model} but this run uses {dino_model}. "
                f"Their feature spaces are unrelated even when the widths match. "
                f"Re-run with --dino-model {bank_model}, or rebuild the bank with --seeds.")
        print(f"  {len(seed_labels)} seed vectors loaded from {seed_bank}")
    elif seeds_file is not None:
        # Written by pose-pick-seeds, which clicked them on the same crop this
        # classifier builds -- so there is no coordinate space to get wrong and
        # nothing to retype.
        from pose_estimator.seed_picker import load_seeds

        picked, _class_order, file_pad = load_seeds(seeds_file)
        by_frame: dict = {}
        for seed in picked:
            by_frame.setdefault(seed.frame, []).append(seed)
        seed_vectors, seed_labels = build_seed_vectors_multi(
            backbone, frames_dir, mask_dir, by_frame, pad=file_pad)
        print(f"  {len(picked)} seeds from {len(by_frame)} frame(s) in {seeds_file}")
        for stem in sorted(by_frame):
            kinds = ", ".join(f"{n}x {lbl}" for lbl, n in
                              sorted({s.label: sum(1 for t in by_frame[stem] if t.label == s.label)
                                      for s in by_frame[stem]}.items()))
            print(f"    {stem}: {kinds}")
    else:
        if not seeds or seed_frame is None:
            raise SystemExit(
                "give seeds one of three ways:\n"
                "  --seeds-file p4c/seeds.json   (from pose-pick-seeds -- click, don't type)\n"
                "  --seeds \"leaf:140,150\" ... --seed-frame 57\n"
                "  --seed-bank <path>/seed_bank.npz   (reuse an earlier specimen's)")
        parsed = parse_seeds(seeds)
        seed_paths = sorted(frames_dir.glob("frame_*.jpg"))
        if seed_frame >= len(seed_paths):
            raise SystemExit(f"--seed-frame {seed_frame} but only {len(seed_paths)} frames exist")
        seed_path = seed_paths[seed_frame]
        seed_bgr = cv2.imread(str(seed_path))
        seed_plant = cv2.imread(str(mask_dir / f"{seed_path.stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
        seed_vectors, seed_labels = build_seed_vectors(backbone, seed_bgr, seed_plant, parsed)
        print(f"  {len(parsed)} seeds taken from {seed_path.stem}")

    class_order = list(dict.fromkeys(seed_labels))
    if len(class_order) < 2:
        # Not an error. A rosette (thistle, sugar beet) has no stem and its
        # roots fall outside the P2 plant mask, so "leaf" is the only class
        # there is anything to click -- demanding a second one would be
        # demanding something that does not exist on the specimen.
        print(f"\n  NOTE: one class only ({class_order[0]!r}, {len(seed_labels)} examples).")
        print("  Every patch on the plant becomes that class, so this pass adds no")
        print("  information -- which is correct for an all-leaf rosette and wrong")
        print("  for a plant that does have a stem you meant to seed.")
        print("  P5 cannot grow leaves from a stem that was never labelled; it needs")
        print("  a base found geometrically instead. See 'Plant architecture' in the README.\n")

    classifier = DinoClassifier.__new__(DinoClassifier)
    classifier.backbone = backbone
    classifier.seed_vectors = seed_vectors
    classifier.seed_labels = seed_labels
    classifier.class_order = class_order
    classifier.model_id = dino_model
    classifier.size = dino_size

    bank_out = save_seed_bank or (workdir / "p4c" / "seed_bank.npz")
    bank_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(bank_out, vectors=seed_vectors, labels=np.array(seed_labels),
             model=dino_model, size=dino_size)
    print(f"  seed vectors saved to {bank_out} -- reuse with --seed-bank")
    return classifier


def run(
    workdir: Path,
    backend: str = "dino",
    seeds: Optional[list] = None,
    seed_frame: Optional[int] = None,
    seed_bank: Optional[Path] = None,
    save_seed_bank: Optional[Path] = None,
    seeds_file: Optional[Path] = None,
    dino_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    dino_size: int = 896,
    hf_token: Optional[str] = None,
    checkpoint: Optional[Path] = None,
    stride: int = 1,
    device: str = "cuda",
) -> dict:
    frames_dir = workdir / "p1" / "frames"
    mask_dir = workdir / "p2" / "masks" / "plant"
    p4c = workdir / "p4c"
    (p4c / "diag").mkdir(parents=True, exist_ok=True)

    frame_stems = [p.stem for p in sorted(frames_dir.glob("frame_*.jpg"))][::stride]
    if not frame_stems:
        raise SystemExit(f"no frames in {frames_dir} -- run pose-segment first")

    if backend == "dino":
        classifier = build_dino_classifier(workdir, seeds, seed_frame, seed_bank,
                                           save_seed_bank, dino_model, dino_size,
                                           hf_token, device, seeds_file)
        settings = {"model": dino_model, "size": dino_size, "stride": stride}
    elif backend == "sam":
        if checkpoint is None:
            raise SystemExit("--backend sam needs --checkpoint <sam2 checkpoint>")
        print(f"Loading SAM2 from {checkpoint} ...")
        classifier = SamClassifier(checkpoint, device=device)
        settings = {"checkpoint": str(checkpoint), "stride": stride}
    else:
        raise SystemExit(f"unknown backend {backend!r}")

    print(f"  classes: {classifier.class_order}")
    print(f"Classifying {len(frame_stems)} frames with the {backend} backend...")
    stats = classify_sequence(classifier, frames_dir, mask_dir,
                              p4c / "class_maps", frame_stems)

    write_manifest(p4c, backend, classifier.class_order, {**settings, **stats})

    total = max(sum(stats["pixels_per_class"].values()), 1)
    print(f"\n  {stats['frames_written']} class maps -> {p4c / 'class_maps'}")
    print("  pixel share per class (across all frames):")
    for name, count in stats["pixels_per_class"].items():
        print(f"    {name:<12} {count:>12}  ({100 * count / total:5.1f}%)")
    print("\n  A seedling whose stem outweighs its leaves is a warning: the classes are")
    print("  intermixed, and P5's insertion detection cannot survive that. Check")
    print(f"  {p4c / 'diag'} before running pose-fuse.")

    _write_diags(p4c, frames_dir, mask_dir, frame_stems, classifier.class_order)
    return stats


def _class_color(name: str, index: int):
    return CLASS_COLORS.get(name, SPARE[index % len(SPARE)])


def _write_diags(p4c: Path, frames_dir: Path, mask_dir: Path, frame_stems, class_order,
                 n: int = 6) -> None:
    """Photograph beside its classification, for a handful of frames."""
    picks = np.linspace(0, len(frame_stems) - 1, min(n, len(frame_stems))).astype(int)
    for i in picks:
        stem = frame_stems[i]
        bgr = cv2.imread(str(frames_dir / f"{stem}.jpg"))
        plant = cv2.imread(str(mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        class_map = load_class_map(p4c / "class_maps", stem)
        if bgr is None or plant is None or class_map is None:
            continue

        ys, xs = np.nonzero(plant > 127)
        if len(xs) == 0:
            continue
        pad = 60
        y0, y1 = max(0, ys.min() - pad), min(bgr.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(bgr.shape[1], xs.max() + pad)
        crop = bgr[y0:y1, x0:x1].copy()
        cls = class_map[y0:y1, x0:x1]

        overlay = crop.copy()
        for index, label in enumerate(class_order):
            hit = cls == index
            if hit.any():
                rgb = _class_color(label, index)
                overlay[hit] = (0.42 * overlay[hit] + 0.58 * np.array(rgb[::-1])).astype(np.uint8)

        panel = np.hstack([crop, overlay])
        legend = "  ".join(class_order)
        for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(panel, f"{stem}   {legend}", (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, thick)
        cv2.imwrite(str(p4c / "diag" / f"parts_{stem}.jpg"), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared with pose-semantic, which runs this stage and the next together."""
    parser.add_argument("--backend", choices=["dino", "sam"], default="dino",
                        help="dino: nearest clicked example in DINOv3 feature space, "
                             "open-vocabulary. sam: SAM2 masks assigned to leaf/stem by shape, "
                             "fixed vocabulary but real object boundaries.")
    parser.add_argument("--seeds-file", type=Path,
                        help="(dino) seeds clicked with pose-pick-seeds, normally "
                             "<workdir>/p4c/seeds.json. Preferred over --seeds: the "
                             "coordinates were placed on the same crop the classifier "
                             "builds, and seeds from several frames are supported")
    parser.add_argument("--seeds", nargs="+",
                        help='(dino) labelled points typed by hand, e.g. "leaf:140,150". '
                             "Coordinates are in the *cropped* frame")
    parser.add_argument("--seed-frame", type=int,
                        help="(dino) index into p1/frames the seed coordinates were read off")
    parser.add_argument("--seed-bank", type=Path,
                        help="(dino) reuse seed vectors saved by an earlier run. Feature vectors "
                             "describe what a leaf looks like, not where this plant's leaves "
                             "are, so one bank can serve many specimens")
    parser.add_argument("--save-seed-bank", type=Path,
                        help="(dino) where to write the seed vectors "
                             "(default <workdir>/p4c/seed_bank.npz)")
    parser.add_argument("--dino-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                        help="gated; facebook/dinov2-base is an ungated equivalent")
    parser.add_argument("--dino-size", type=int, default=896,
                        help="square side the crop is resized to; must be a multiple of the "
                             "patch size. Larger means a finer patch grid, which is the most "
                             "direct lever on thin structures being lost")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token for gated models (or set HF_TOKEN)")
    parser.add_argument("--checkpoint", type=Path,
                        help="(sam) SAM2 checkpoint, e.g. checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--stride", type=int, default=1, help="classify every Nth frame")
    parser.add_argument("--device", default="cuda")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path)
    add_arguments(parser)
    args = parser.parse_args(argv)

    run(workdir=args.workdir, backend=args.backend, seeds=args.seeds,
        seed_frame=args.seed_frame, seed_bank=args.seed_bank,
        save_seed_bank=args.save_seed_bank, seeds_file=args.seeds_file,
        dino_model=args.dino_model,
        dino_size=args.dino_size, hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        checkpoint=args.checkpoint, stride=args.stride, device=args.device)


if __name__ == "__main__":
    main()
