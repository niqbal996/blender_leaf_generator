"""Command-line entry points for the plant pose pipeline.

Each module here is a standalone phase that reads from and writes to a
plant's `--workdir` on disk -- no phase holds another's state in memory, so
any one of them can be re-run (or swapped for a different backend) without
touching the others.

Run them either as installed console scripts::

    pose-estimate-skeleton --images stills/ --workdir out/plant1/
    pose-train-splat       --workdir out/plant1/
    pose-align-skeleton    --workdir out/plant1/

or, without installing, as modules from the repo root::

    PYTHONPATH=src python -m pose_estimator.cli.estimate_skeleton --help
"""
