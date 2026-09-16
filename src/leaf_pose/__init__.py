"""leaf_pose -- per-leaf shape and pose from a flat-lay polarised light scan.

The input is one capture folder of a *dissected* plant: the leaves have been
detached and laid out on a dark backing, petioles still attached, and
photographed from directly above while 12 LEDs fire one at a time, twice --
once through parallel polarisers and once through crossed ones.

The output, per leaf, is a mask with a sub-pixel contour, a midrib polyline,
the tip, and the petiole origin (the cut end that was joined to the stem).

This package is `bpy`-free and does not need a reconstruction: everything is
measured in the one image plane, which is the whole point. `pose_estimator`
recovers the same quantities for a plant that is still assembled, and pays
for it with a multi-view solve; here the plant has been taken apart, so the
geometry is already flat and the measurement is direct.
"""

__all__ = [
    "instances",
    "keypoints",
    "midrib",
    "photometric",
    "polyline",
    "raw",
    "rig",
    "scale",
    "veins",
    "viz",
]
