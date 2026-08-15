"""Thin wrapper around pycolmap for sparse structure-from-motion.

`pycolmap` is an optional dependency (`pip install -e ".[skeleton]"`) and is
imported lazily so the rest of this subpackage stays usable without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class CameraView:
    """One registered image's camera, for feeding a Gaussian Splat trainer."""

    image_path: Path
    width: int
    height: int
    K: np.ndarray  # (3, 3) intrinsics
    world_to_camera: np.ndarray  # (4, 4) extrinsics


def build_sparse_reconstruction(
    image_dir: Union[str, Path],
    output_dir: Union[str, Path],
    mask_dir: Optional[Union[str, Path]] = None,
    database_path: Optional[Union[str, Path]] = None,
    num_threads: int = 4,
    max_image_size: int = 2000,
    use_gpu: bool = False,
    single_camera: bool = True,
):
    """Run SIFT extraction + exhaustive matching + incremental mapping.

    Returns the largest `pycolmap.Reconstruction` model found (COLMAP may
    split a scene into multiple disconnected models if matching fails to
    fully connect it -- for a single rotating subject, one model spanning
    most/all frames is what you want).

    `num_threads`/`max_image_size` default to conservative values -- COLMAP's
    own defaults extract at full resolution with one thread per CPU core,
    which for phone-camera-sized images (4000px+) is enough to OOM-kill the
    process on a 32GB machine. `use_gpu` needs a CUDA-enabled pycolmap build
    (e.g. `pip install pycolmap-cuda`, plus `nvidia-cuda-runtime-cu12` and
    that package's lib/ dir on LD_LIBRARY_PATH if `import pycolmap` raises
    `libcudart.so.12: cannot open shared object file`); `num_threads` is
    ignored on the GPU path.

    `single_camera` forces every frame to share one set of intrinsics, and
    defaults to True because it is simply true of these rigs: one locked-off
    body and lens for the whole sequence. COLMAP's own default is AUTO, which
    on frames extracted from video has no EXIF to group by and silently falls
    back to a *separate camera per image* -- handing the solver ~96 extra
    free focal lengths that it will happily use to absorb reconstruction
    drift. Measured on DSC_0010: per-image cameras produced a 27.5% spread in
    focal length across frames from a fixed lens, and camera centres that
    missed their own fitted circle by 5.2% of orbit radius.
    """
    import pycolmap

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(database_path) if database_path else output_dir / "database.db"
    if db_path.exists():
        db_path.unlink()

    reader_options = pycolmap.ImageReaderOptions()
    if mask_dir is not None:
        reader_options.mask_path = str(mask_dir)

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.num_threads = num_threads
    extraction_options.max_image_size = max_image_size
    extraction_options.use_gpu = use_gpu

    pycolmap.extract_features(
        db_path,
        image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu,
    )
    pycolmap.match_exhaustive(db_path)

    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    reconstructions = pycolmap.incremental_mapping(db_path, image_dir, sparse_dir)

    if not reconstructions:
        raise RuntimeError(
            "COLMAP could not reconstruct any model from these frames -- "
            "likely too few/blurry matches. Try more frames, better masking, "
            "or steadier rotation."
        )

    best = max(reconstructions.values(), key=lambda rec: rec.num_reg_images())

    # Persist the winning model under a stable name -- `incremental_mapping`
    # numbers its candidate models (0, 1, 2, ...) and the winner's number is
    # otherwise lost once we return just the in-memory object, so nothing
    # downstream (splat training, re-alignment) could cheaply reload just
    # this model without rerunning feature extraction/matching from scratch.
    best_dir = sparse_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best.write(str(best_dir))

    return best


def get_registered_camera_poses(reconstruction) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(N, 3) projection centers, (N, 3) unit viewing directions, and image
    names, for every registered image -- feeds `alignment.solve_alignment`.
    """
    centers, directions, names = [], [], []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        centers.append(image.projection_center().reshape(3))
        directions.append(image.viewing_direction().reshape(3))
        names.append(image.name)
    return np.array(centers), np.array(directions), names


def get_camera_data(reconstruction, images_dir: Union[str, Path]) -> List[CameraView]:
    """Per-registered-image intrinsics/extrinsics/image path, for the
    Gaussian Splat trainer's training views.
    """
    images_dir = Path(images_dir)
    views = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]

        world_to_camera = np.eye(4)
        world_to_camera[:3, :] = image.cam_from_world().matrix()

        views.append(
            CameraView(
                image_path=images_dir / image.name,
                width=camera.width,
                height=camera.height,
                K=camera.calibration_matrix(),
                world_to_camera=world_to_camera,
            )
        )
    return views
