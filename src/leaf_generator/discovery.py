"""Discovery of per-leaf PBR map sets on disk.

Expects filenames of the form ``<leaf_id>_<TYPE>_<side>.png`` inside a flat
"maps" folder, e.g.::

    1_ALBEDO_oberseite.png
    1_HEIGHT_oberseite.png
    1_mask_oberseite.png
    1_NORMAL_GL_oberseite.png
    1_ROUGHNESS_oberseite.png
    1_ALBEDO_unterseite.png
    ...

``unterseite`` (underside) is optional per leaf; ``oberseite`` (topside) is
treated as the primary side whenever it exists.

The trailing token doesn't have to be a side name. Sessions captured from
2026-07-31 on put the plant name there instead (``1_ALBEDO_plant_1.png``,
``10_mask_gaensefuss_1.png``) and scan a single face per leaf; those load as
one unnamed side, which `primary_side` picks up and `has_both_sides` reports
False for, so they build as single-sided leaves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union

# The trailing token is whatever the capture wrote there. Older sessions put
# the side (`oberseite`/`unterseite`) in that slot and scanned both faces of
# each leaf; since 2026-07-31 the capture writes the plant name instead
# (`1_ALBEDO_plant_1.png`) and scans one face only. Anything that isn't a
# recognized side name is therefore treated as a single unnamed side -- see
# `LeafMapSet.primary_side` / `has_both_sides`, which key off the two real
# side names and so degrade to single-sided on their own.
_FILENAME_RE = re.compile(
    r"^(?P<id>[A-Za-z0-9]+)_(?P<type>ALBEDO|HEIGHT|MASK|NORMAL_GL|NORMAL_DX|ROUGHNESS)"
    r"_(?P<side>[A-Za-z0-9_-]+)\.(?:png|jpe?g|tiff?)$",
    re.IGNORECASE,
)

# Normalize the various filename type tokens onto a small, stable set of
# map-type keys used everywhere else in the pipeline.
_TYPE_ALIASES = {
    "albedo": "albedo",
    "height": "height",
    "mask": "mask",
    "normal_gl": "normal",
    "normal_dx": "normal",
    "roughness": "roughness",
}

PRIMARY_SIDE = "oberseite"
SECONDARY_SIDE = "unterseite"


@dataclass
class LeafMapSet:
    leaf_id: str
    sides: Dict[str, Dict[str, Path]] = field(default_factory=dict)

    @property
    def primary_side(self) -> str:
        """oberseite if present, otherwise whichever side was found."""
        if PRIMARY_SIDE in self.sides:
            return PRIMARY_SIDE
        return next(iter(self.sides))

    @property
    def has_both_sides(self) -> bool:
        return PRIMARY_SIDE in self.sides and SECONDARY_SIDE in self.sides

    def maps_for(self, side: str) -> Dict[str, Path]:
        return self.sides.get(side, {})


def find_leaf_groups(maps_dir: Union[str, Path]) -> Dict[str, LeafMapSet]:
    """Group all recognized map files in ``maps_dir`` by leaf id.

    Files that don't match the naming convention are silently skipped so
    stray files (the *_log.json sidecars, thumbnails, etc.) don't break
    loading.
    """
    maps_dir = Path(maps_dir)
    groups: Dict[str, LeafMapSet] = {}

    for path in sorted(maps_dir.iterdir()):
        if not path.is_file():
            continue
        match = _FILENAME_RE.match(path.name)
        if not match:
            continue

        leaf_id = match.group("id")
        map_type = _TYPE_ALIASES[match.group("type").lower()]
        side = match.group("side").lower()

        leaf = groups.setdefault(leaf_id, LeafMapSet(leaf_id=leaf_id))
        leaf.sides.setdefault(side, {})[map_type] = path

    return dict(sorted(groups.items(), key=_leaf_sort_key))


def is_maps_folder(path: Union[str, Path]) -> bool:
    """True if `path` directly contains at least one recognized leaf map file."""
    path = Path(path)
    if not path.is_dir():
        return False
    return any(_FILENAME_RE.match(p.name) for p in path.iterdir() if p.is_file())


def _leaf_sort_key(item):
    leaf_id, _leaf = item
    try:
        return (0, int(leaf_id))
    except ValueError:
        return (1, leaf_id)
