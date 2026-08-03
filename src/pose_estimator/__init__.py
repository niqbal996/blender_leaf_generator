"""pose_estimator: recover a plant's 3D structure -- point cloud, stem/branch
skeleton, and (eventually) per-leaf midrib/curvature/curl -- from a turntable
capture of a single potted or clamp-held plant.

Split from `leaf_generator` (which assembles procedural leaf *assets* for
Blender out of PBR map sets); the two packages share no code. Everything here
is about measuring a real plant, not authoring a synthetic one.

Layout:
- Pure, bpy-free library modules (`frames`, `masking`, `reconstruction`,
  `pointcloud`, `turntable`, `skeletonize`, `alignment`, `gaussian_splat`,
  `ply_io`, `visualize`) -- importable with a regular Python interpreter.
- `pose_estimator.cli.*` -- the end-to-end CLIs, also exposed as the
  `pose-estimate-skeleton` / `pose-train-splat` / `pose-align-skeleton`
  console scripts.
- `pose_estimator.blender.*` -- requires `bpy`, run only inside Blender.

`reconstruction.py` and `gaussian_splat.py` need optional dependencies
(`pycolmap`, and `torch`/`gsplat` respectively); both are imported lazily so
the rest of the package stays usable without them -- install via
`pip install -e ".[skeleton]"` / `".[skeleton,splat]"`.

Status: the skeleton/splat path is a research prototype whose output should
be inspected, not trusted blindly -- see `plant_pose_pipeline_PLAN.md` for
the phased rebuild that replaces its per-plant appearance thresholds with
geometry-derived constraints.
"""

__version__ = "0.1.0"
