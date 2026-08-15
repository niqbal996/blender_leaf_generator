"""pose_estimator: measure a real plant from a turntable capture.

Video in, per-leaf midribs out. Split from `leaf_generator` (which assembles
procedural leaf *assets* for Blender out of PBR map sets); the two packages
share no code. Everything here is about measuring a real plant, not authoring
a synthetic one.

Every phase is a standalone CLI that reads its predecessor's files out of a
run directory and writes its own, so any one can be re-run, inspected or
swapped without touching the others:

    P1+P2  pose-segment    sharpest frames + SAM2 plant/holder masks
    P3     pose-solve      camera poses, masked COLMAP
    P4a    pose-hull       visual hull by silhouette carving
    P4b    pose-surface    2DGS surfels -> carved thin surface
    P4c    pose-pick-seeds click organ seeds on a frame (DINOv3 backend)
           pose-classify   per-frame organ class maps  (DINOv3 or SAM2)
           pose-fuse       class maps -> per-point organ labels
           pose-semantic   both P4c stages in one call
    P5     pose-structure  stem centreline + leaf instances
    P6     pose-leaf       per-leaf midrib, frame, curvature, width

    pose-view-structure    rotate P5's tips and instances in 3D

Each phase also writes a QC report (`pN/*.json`) with explicit pass/fail
acceptance checks and diagnostics under `pN/diag/`. Read those before trusting
a run -- the checks exist to catch the failures that look plausible, not the
ones that look broken.

Library modules are pure and bpy-free. `reconstruction.py` needs `pycolmap`
and `surfels.py` needs `torch`/`gsplat`; both are imported lazily so the rest
of the package stays usable without them. Install via
`pip install -e ".[dev,skeleton,segment]"`.

`alignment.py` is not yet wired into the P1-P6 chain. It is kept because it is
the only metric-scale machinery in the repo -- it solves the similarity
transform from two points whose real-world separation was measured -- and
scale is the largest outstanding gap in the pipeline.

Status: research code. See `plant_pose_pipeline_PLAN.md` for the target and
`DECISIONS.md` for what was measured and why each choice was made.
"""

__version__ = "0.2.0"
