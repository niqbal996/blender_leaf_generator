"""One leaf's finished measurements: what is written out, and what is drawn.

Everything upstream works in a leaf's own crop, because that is the only way
to hold a 61 megapixel frame's worth of leaves in memory at once. A record is
where those crop-local results are put back into frame coordinates and, where
a scale was measured, into millimetres -- so the JSON, the diagram and any
later consumer all read the same numbers in the same frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .instances import Instance
from .keypoints import Keypoints
from .midrib import Midrib
from .veins import Vein


@dataclass
class LeafRecord:
    leaf_id: int
    instance: Instance
    midrib: Midrib
    keypoints: Keypoints
    veins: List[Vein] = field(default_factory=list)
    mm_per_pixel: Optional[float] = None

    # ---- frame coordinates ------------------------------------------------
    @property
    def contour(self) -> np.ndarray:
        return self.instance.contour  # already full-frame

    @property
    def midrib_xy(self) -> np.ndarray:
        return self.instance.to_full(self.midrib.path)

    @property
    def tip(self) -> np.ndarray:
        return self.instance.to_full(self.keypoints.tip[None, :])[0]

    @property
    def petiole_origin(self) -> np.ndarray:
        return self.instance.to_full(self.keypoints.petiole_origin[None, :])[0]

    @property
    def blade_base(self) -> np.ndarray:
        return self.instance.to_full(self.keypoints.blade_base[None, :])[0]

    # ---- measurements -----------------------------------------------------
    def length(self, key: str) -> float:
        """A stored pixel length, in millimetres when a scale was measured."""
        pixels = {
            "blade": self.keypoints.blade_length,
            "petiole": self.keypoints.petiole_length,
            "width": self.keypoints.blade_width_max,
            "total": self.keypoints.blade_length + self.keypoints.petiole_length,
        }[key]
        return pixels * self.mm_per_pixel if self.mm_per_pixel else pixels

    @property
    def area(self) -> float:
        """Leaf area, in mm^2 when a scale was measured, else in pixels."""
        if self.mm_per_pixel:
            return float(self.instance.area_px * self.mm_per_pixel ** 2)
        return float(self.instance.area_px)

    @property
    def units(self) -> str:
        return "mm" if self.mm_per_pixel else "px"

    def to_dict(self) -> dict:
        # The keypoints come back in the leaf's own crop; the report is in
        # frame coordinates throughout, so they are re-read off this record's
        # properties rather than copied from the crop-local dict.
        keypoints = self.keypoints.to_dict()
        for name in ("petiole_origin", "blade_base", "tip"):
            keypoints[name] = [round(float(v), 2) for v in getattr(self, name)]
        return {
            "id": self.leaf_id,
            "units": self.units,
            "mm_per_pixel": self.mm_per_pixel,
            "bbox": list(self.instance.bbox),
            "area": round(self.area, 3),
            "blade_length": round(self.length("blade"), 3),
            "petiole_length": round(self.length("petiole"), 3),
            "blade_width_max": round(self.length("width"), 3),
            "midrib_ridge_support": round(float(self.midrib.ridge_support), 4),
            "keypoints": keypoints,
            "midrib": self.midrib_xy.round(2).tolist(),
            "contour": self.contour.round(2).tolist(),
            "veins": [v.to_dict() for v in self.veins],
        }
