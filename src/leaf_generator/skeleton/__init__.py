"""Experimental: estimate a whole-plant skeleton (stem/branch/leaf graph)
from a rotation video, via structure-from-motion + point-cloud skeletonization.

This is a research prototype, not a validated pipeline -- see
`estimate_plant_skeleton.py` at the repo root for the end-to-end CLI, and
the README's "Plant skeleton from video" section for known limitations.

`reconstruction.py` requires `pycolmap` (optional dependency, install via
`pip install -e ".[skeleton]"`); the rest of this subpackage only needs
numpy/opencv/scipy/matplotlib.
"""
