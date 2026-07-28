"""Place a Blender Empty marking a leaf's estimated stem-attachment point."""

from __future__ import annotations

from typing import Tuple

import bpy
from mathutils import Matrix


def create_attachment_empty(
    name: str,
    obj: bpy.types.Object,
    local_coords: Tuple[float, float, float],
    collection: bpy.types.Collection,
    display_size: float = 0.03,
) -> bpy.types.Object:
    """Create an Empty at `local_coords` (in `obj`'s local mesh space, i.e.
    the same space as its vertices) and rigidly parent it to `obj`, so it
    tracks any later position/rotation applied to the leaf.
    """
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = 'PLAIN_AXES'
    empty.empty_display_size = display_size
    collection.objects.link(empty)

    empty.parent = obj
    empty.matrix_parent_inverse = Matrix.Identity(4)
    empty.location = local_coords

    return empty
