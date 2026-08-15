"""Command-line entry points for the plant pose pipeline.

Each module here is a standalone phase that reads from and writes to a
plant's `--workdir` on disk -- no phase holds another's state in memory, so
any one of them can be re-run (or swapped for a different backend) without
touching the others.

Run them either as installed console scripts, in pipeline order::

    pose-segment   --video capture.MOV --workdir runs/plant_9/   # P1 + P2
    pose-solve     --workdir runs/plant_9/                       # P3
    pose-hull      --workdir runs/plant_9/                       # P4a
    pose-surface   --workdir runs/plant_9/                       # P4b
    pose-classify  --workdir runs/plant_9/ --seeds ...           # P4c stage 1
    pose-fuse      --workdir runs/plant_9/                       # P4c stage 2
    pose-structure --workdir runs/plant_9/                       # P5
    pose-leaf      --workdir runs/plant_9/                       # P6

or, without installing, as modules from the repo root::

    PYTHONPATH=src python -m pose_estimator.cli.hull --help

`pose-semantic` runs both P4c stages in one call; `run_pipeline.sh` drives
P1 through P4c end to end.
"""
