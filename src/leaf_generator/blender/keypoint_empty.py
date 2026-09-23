"""Place a Blender Empty marking a leaf's estimated stem-attachment point."""

from __future__ import annotations

from typing import Tuple

import bpy
from mathutils import Matrix

# How the attachment marker is drawn. 'point' is the default because these
# leaves are 1-6 cm across and the old PLAIN_AXES cross, at a fixed 3 cm,
# was drawn larger than the leaf it marked -- a row of crosses with the
# leaves lost inside them. 'hidden' still builds the Empty (so the keypoint
# JSON and any parenting stay intact) and just stops drawing it.
DISPLAY_STYLES = ("point", "axes", "hidden")


def create_attachment_empty(
    name: str,
    obj: bpy.types.Object,
    local_coords: Tuple[float, float, float],
    collection: bpy.types.Collection,
    display_size: float = 0.03,
    style: str = "point",
) -> bpy.types.Object:
    """Create an Empty at `local_coords` (in `obj`'s local mesh space, i.e.
    the same space as its vertices) and rigidly parent it to `obj`, so it
    tracks any later position/rotation applied to the leaf.

    `display_size` is in metres and is only how large the marker is *drawn*;
    it has no effect on the recorded attachment position.
    """
    if style not in DISPLAY_STYLES:
        raise ValueError(f"unknown attachment display style {style!r}; expected one of {DISPLAY_STYLES}")

    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = 'PLAIN_AXES' if style == "axes" else 'SPHERE'
    # Blender clamps to a positive display size; a leaf whose bbox came out
    # degenerate would otherwise ask for 0.
    empty.empty_display_size = max(float(display_size), 1e-5)
    collection.objects.link(empty)

    if style == "hidden":
        empty.hide_viewport = True
        empty.hide_render = True

    empty.parent = obj
    empty.matrix_parent_inverse = Matrix.Identity(4)
    empty.location = local_coords

    return empty
