"""P4c: organ labels on the 3D cloud. Runs both stages back to back.

    pose-semantic --workdir runs/plant_9 --seed-frame 57 \\
        --seeds "leaf:140,150" "leaf:400,250" "stem:300,300" "root:370,640"

This is a convenience wrapper, kept so the pipeline script has one command to
call. The work happens in two stages that are also runnable on their own, and
running them separately is the better move whenever something looks wrong:

    pose-classify --workdir ...    frames -> p4c/class_maps/*.png
    pose-fuse     --workdir ...    maps   -> p4c/labels.npy + coloured clouds

Splitting them is what makes P4c debuggable. A bad organ label comes either
from the 2D classifier or from the multi-view fusion, and the class maps on
disk are what let you tell which -- score them against a few hand-labelled
frames, or feed the fusion synthetic perfect maps and check it returns them.
Inspect `p4c/diag/parts_*.jpg` after classify, before spending time on fusion.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from pose_estimator.cli import classify as classify_stage
from pose_estimator.cli import fuse as fuse_stage

CLASSIFY_KEYS = ("backend", "seeds", "seed_frame", "seed_bank", "save_seed_bank",
                 "dino_model", "dino_size", "hf_token", "checkpoint", "stride", "device")
FUSE_KEYS = ("source", "instance_radius_voxels", "normal_weighting")


def run(workdir: Path, **kwargs) -> dict:
    print("=== P4c stage 1 of 2: classify (frames -> class maps) ===")
    classify_stage.run(workdir, **{k: v for k, v in kwargs.items() if k in CLASSIFY_KEYS})
    print("\n=== P4c stage 2 of 2: fuse (class maps -> point labels) ===")
    return fuse_stage.run(workdir, **{k: v for k, v in kwargs.items() if k in FUSE_KEYS})


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path)
    classify_stage.add_arguments(parser)
    fuse_stage.add_arguments(parser)
    args = parser.parse_args(argv)

    run(
        workdir=args.workdir,
        backend=args.backend,
        seeds=args.seeds,
        seed_frame=args.seed_frame,
        seed_bank=args.seed_bank,
        save_seed_bank=args.save_seed_bank,
        dino_model=args.dino_model,
        dino_size=args.dino_size,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        checkpoint=args.checkpoint,
        stride=args.stride,
        device=args.device,
        source=args.source,
        instance_radius_voxels=args.instance_radius_voxels,
        normal_weighting=not args.no_normal_weighting,
    )


if __name__ == "__main__":
    main()
