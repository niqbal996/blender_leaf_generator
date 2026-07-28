"""Read per-session capture calibration from the processing log JSON that
sits alongside a "maps" folder (e.g. `oberseite_log.json`).

The log's `parameters.pixelsize_mm` is the mm-per-pixel resolution *at the
object plane* (already resolved through the lens/distance calibration --
not the camera's raw sensor pixel pitch), so real-world leaf size is just
`pixels * pixel_size_mm`, no further lens-magnification math needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class SessionCalibration:
    pixel_size_mm: float  # mm per pixel at the object (leaf) plane
    distance_z_mm: Optional[float] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length_mm: Optional[float] = None
    source_log: Optional[Path] = None

    def px_to_mm(self, pixels: float) -> float:
        return pixels * self.pixel_size_mm

    def px_to_m(self, pixels: float) -> float:
        return self.px_to_mm(pixels) / 1000.0

    @property
    def pixel_size_m(self) -> float:
        return self.pixel_size_mm / 1000.0


def find_calibration_log(maps_dir: Union[str, Path], side: str = "oberseite") -> Optional[Path]:
    """Prefer `<side>_log.json`; fall back to any `*_log.json` in the folder."""
    maps_dir = Path(maps_dir)
    preferred = maps_dir / f"{side}_log.json"
    if preferred.is_file():
        return preferred

    candidates = sorted(maps_dir.glob("*_log.json"))
    return candidates[0] if candidates else None


def load_session_calibration(
    maps_dir: Union[str, Path], side: str = "oberseite"
) -> Optional[SessionCalibration]:
    """Load calibration for a session's maps folder, or None if no usable
    log file / pixelsize_mm field is found (caller should fall back to an
    arbitrary, non-physical scale and warn).
    """
    log_path = find_calibration_log(maps_dir, side=side)
    if log_path is None:
        return None

    with open(log_path) as f:
        data = json.load(f)

    pixel_size_mm = data.get("parameters", {}).get("pixelsize_mm")
    if pixel_size_mm is None:
        return None

    params = data.get("parameters", {})
    camera_info = data.get("metadata", {}).get("camera_info", {})
    focal_length_mm = None
    try:
        focal_length_mm = float(camera_info["focal_length"])
    except (KeyError, TypeError, ValueError):
        pass

    return SessionCalibration(
        pixel_size_mm=float(pixel_size_mm),
        distance_z_mm=params.get("distance_z_mm"),
        camera_model=camera_info.get("camera_model"),
        lens_model=camera_info.get("lens_model"),
        focal_length_mm=focal_length_mm,
        source_log=log_path,
    )
