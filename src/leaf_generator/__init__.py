"""leaf_generator: assemble procedural leaf assets from per-leaf PBR map sets.

This package is split into:
- Pure, bpy-free modules (`discovery`, `keypoints`, `mismatch`) that can be
  imported and unit-tested with a regular Python interpreter.
- `leaf_generator.blender.*`, which requires `bpy` and must be run from
  inside Blender.
"""

__version__ = "0.1.0"
