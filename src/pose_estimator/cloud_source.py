"""Which point cloud a phase labels, where its results go, and at what scale.

P4c and P5 never wanted a hull. They want *a point cloud with cameras*, and
they already chose between two producers -- P4b's carved surface and P4a's
hull. A learned P3 backend is a third producer, and a better-shaped one: a
pointmap is a surface sample, which is the kind of input P5 already prefers.

Keeping that choice here rather than in each phase means the three branches
differ by one flag instead of by a fork in every stage:

    colmap      -> P4a hull -> P4b surfels -> P4c -> P5      (the baseline)
    vggt_omega  -> P4c -> P5                                 (no carving)
    mapanything -> P4c -> P5                                 (no carving)

What a learned branch gives up is worth saying plainly. The hull is an upper
bound built by occlusion-aware voting: it can be too big, but it cannot invent
surface. A learned cloud has no such guarantee. And P4b's normals are fitted
against the photographs, where a bare pointmap's have to be estimated by local
PCA, which weakens the obliquity weighting in label fusion. Carving the hull
anyway and treating it as a *check* on the learned cloud -- rather than as the
geometry -- keeps that safeguard without making P4a a dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

BASELINE = "colmap"


@dataclass
class CloudSource:
    """The cloud a branch runs on, and where that branch's results belong."""

    path: Path
    labels_dir: Path
    structure_dir: Path
    backend: str
    origin: str
    is_hull: bool = False

    @property
    def is_baseline(self) -> bool:
        return self.backend == BASELINE


def resolve(
    workdir: Path,
    backend: str = BASELINE,
    cloud: Optional[Path] = None,
    source: str = "auto",
) -> CloudSource:
    """Pick the cloud for this branch and the directories it writes to.

    ``source`` keeps its P4a/P4b meaning for the baseline. For a learned
    backend it selects between that backend's P3 cloud and a hull carved from
    it, if one was carved as a check.
    """
    workdir = Path(workdir)
    labels_dir, structure_dir = phase_dirs(workdir, backend)
    if cloud is not None:
        return CloudSource(Path(cloud), labels_dir, structure_dir, backend,
                           f"{cloud} (given explicitly)")

    if backend == BASELINE:
        surface = workdir / "p4b" / "surface.ply"
        hull = workdir / "p4" / "hull_points.ply"
        chosen = surface if source == "surface" or (
            source == "auto" and surface.exists()) else hull
        origin = ("P4b carved surface" if chosen == surface
                  else "P4a hull (no P4b surface on disk)")
        return CloudSource(chosen, labels_dir, structure_dir, backend, origin,
                           is_hull=chosen == hull)

    experiment_hull = workdir / "p4" / "experiments" / backend / "hull_points.ply"
    learned = workdir / "p3" / "experiments" / backend / "sparse_points.ply"
    if source == "hull":
        return CloudSource(experiment_hull, labels_dir, structure_dir, backend,
                           f"hull carved from {backend} poses", is_hull=True)
    return CloudSource(learned, labels_dir, structure_dir, backend,
                       f"{backend} P3 cloud, uncarved")


def phase_dirs(workdir: Path, backend: str):
    """(p4c, p5) for this branch, keeping experiments beside the baseline."""
    workdir = Path(workdir)
    if backend == BASELINE:
        return workdir / "p4c", workdir / "p5"
    return (workdir / "p4c" / "experiments" / backend,
            workdir / "p5" / "experiments" / backend)


def class_map_dir(workdir: Path) -> Path:
    """Where P4c's 2D class maps live -- shared by every branch.

    Classification reads frames and P2 masks and knows nothing about 3D, so
    the maps are identical whichever cloud is being labelled. Classifying once
    and fusing three times is both faster and a fairer comparison: the
    branches then differ only in geometry.
    """
    return Path(workdir) / "p4c" / "class_maps"


def voxel_size(workdir: Path, backend: str, points: np.ndarray,
               resolution: int = 256) -> tuple:
    """The length unit the tuned ``*_voxels`` parameters are counted in.

    P4a writes one into `hull.json`, and for the baseline that is the answer.
    A learned branch has no hull, and the baseline's voxel is useless to it:
    the backends reconstruct at unrelated scales (on thistle3 the orbit radius
    is 3.69 for COLMAP, 2.85 for MapAnything and 0.86 for VGGT-Omega), so a
    COLMAP-sized voxel would be off by a factor of four.

    So it is derived from the cloud instead, as extent/resolution -- the same
    definition P4a uses at its default resolution, which keeps parameters like
    ``instance_radius_voxels=2.5`` meaning what they were tuned to mean.
    Returns (voxel, where it came from).
    """
    hull_json = (Path(workdir) / "p4" / "hull.json" if backend == BASELINE
                 else Path(workdir) / "p4" / "experiments" / backend / "hull.json")
    if hull_json.exists():
        with open(hull_json) as handle:
            return float(json.load(handle)["voxel_size"]), str(hull_json)
    if len(points) == 0:
        raise ValueError("cannot derive a voxel size from an empty cloud")
    extent = np.percentile(points, 98, axis=0) - np.percentile(points, 2, axis=0)
    return float(max(extent.max(), 1e-9) / resolution), f"cloud extent / {resolution}"


def geometry(workdir: Path, backend: str = BASELINE) -> tuple:
    """The P3 model and pose report whose cameras this branch is labelled with.

    P4c projects the cloud into cameras to collect votes, and P5 reads the
    orbit from the same report. Both have to come from the reconstruction that
    produced the cloud. A learned backend solves its own frame at its own
    scale -- on thistle3 the orbit radius is 3.69 for COLMAP and 0.86 for
    VGGT-Omega -- so projecting its points through the COLMAP cameras would
    label them by where a different reconstruction thought they were, and the
    votes would be nonsense rather than merely noisy.
    """
    from pose_estimator.geometry import geometry_dir

    directory = geometry_dir(Path(workdir), backend)
    return directory / "sparse" / "best", directory / "poses.json"
