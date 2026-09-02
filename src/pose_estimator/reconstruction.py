"""Thin wrapper around pycolmap for sparse structure-from-motion.

`pycolmap` is an optional dependency (`pip install -e ".[skeleton]"`) and is
imported lazily so the rest of this subpackage stays usable without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


def _camera_mode(cameras: str, single_camera: bool):
    """COLMAP's camera grouping, defaulting to one shared camera.

    One camera for the whole capture is right for this rig and wrong the
    moment the lens is touched. Measured on sugarbeet_3, whose three passes
    were shot at 48mm, 32mm and 22mm: forcing a single focal across a 2.2x
    zoom range gives a compromise that fits none of them, and the passes come
    back interleaved rather than as three orbits.

    EXIF cannot rescue it here -- `ingest_photos` re-encodes when it resizes,
    which drops the tags -- and the passes share one frames directory, so
    per-folder grouping does not apply either. `per-image` is the escape
    hatch: it costs free parameters but makes no assumption about the lens.
    """
    import pycolmap

    if cameras == "per-image":
        return pycolmap.CameraMode.PER_IMAGE
    if cameras == "auto":
        return pycolmap.CameraMode.AUTO
    return pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO


def build_sparse_reconstruction(
    image_dir: Union[str, Path],
    output_dir: Union[str, Path],
    mask_dir: Optional[Union[str, Path]] = None,
    database_path: Optional[Union[str, Path]] = None,
    num_threads: int = 4,
    max_image_size: int = 2000,
    use_gpu: bool = False,
    single_camera: bool = True,
    low_texture: bool = False,
    cameras: str = "single",
):
    """Run SIFT extraction + exhaustive matching + incremental mapping.

    Returns the largest `pycolmap.Reconstruction` model found (COLMAP may
    split a scene into multiple disconnected models if matching fails to
    fully connect it -- for a single rotating subject, one model spanning
    most/all frames is what you want).

    `num_threads`/`max_image_size` default to conservative values -- COLMAP's
    own defaults extract at full resolution with one thread per CPU core,
    which for phone-camera-sized images (4000px+) is enough to OOM-kill the
    process on a 32GB machine.

    `use_gpu` moves SIFT extraction onto the GPU and needs a pycolmap that was
    *built* with CUDA. The wheels on PyPI are not: `pycolmap.has_cuda` is the
    authoritative test, and it is False for them, so GPU SIFT means building
    COLMAP and pycolmap from source with `-DCUDA_ENABLED=ON`. Asking for it
    without that support fails inside COLMAP's own option validation with
    `Check failed: extraction_options.Check()`, which names neither the option
    nor the cause, so it is checked here first. `num_threads` is ignored on
    the GPU path. Only extraction is affected -- matching and mapping are CPU
    either way, and on a 96-frame sequence they dominate the runtime.

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

    if use_gpu and not getattr(pycolmap, "has_cuda", False):
        raise RuntimeError(
            "use_gpu was requested but this pycolmap has no CUDA support "
            f"(pycolmap {getattr(pycolmap, '__version__', '?')} at {pycolmap.__file__}).\n"
            "  pycolmap.has_cuda is the authoritative test and it is False for the\n"
            "  wheels published on PyPI -- GPU SIFT needs a pycolmap built against\n"
            "  COLMAP with -DCUDA_ENABLED=ON.\n"
            "  Simplest fix: drop --use-gpu. CPU SIFT is the default, and extraction\n"
            "  is roughly a quarter of P3 -- matching and mapping stay on the CPU\n"
            "  regardless, so the GPU path saves less than it sounds like.")

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.num_threads = num_threads
    extraction_options.max_image_size = max_image_size
    extraction_options.use_gpu = use_gpu

    matching_options = pycolmap.FeatureMatchingOptions()
    if low_texture:
        # COLMAP's own settings for hard material, left at their defaults
        # until now. A plant is a small, smooth, self-similar subject and the
        # captures here are soft, so features are found in quantity and then
        # fail to *match*: measured on weed_1, a median of 1,984 keypoints per
        # image but only 29 matches per pair, and 13 of 27 images registered.
        #
        #   peak_threshold   keeps weaker maxima, which is most of what a
        #                    slightly soft image has left
        #   estimate_affine_shape / domain_size_pooling
        #                    descriptors that survive a viewpoint or scale
        #                    change, which is exactly what fails between two
        #                    photographs taken 13 degrees apart
        #   guided_matching  a second matching pass using the geometry the
        #                    first one found
        #
        # All deterministic and applied identically to every image, so unlike
        # a generative "enhancement" they cannot invent detail that differs
        # between views. The cost is roughly 3-5x in extraction time.
        extraction_options.sift.peak_threshold = 0.004
        extraction_options.sift.max_num_features = 16384
        extraction_options.sift.estimate_affine_shape = True
        extraction_options.sift.domain_size_pooling = True
        matching_options.guided_matching = True

    pycolmap.extract_features(
        db_path,
        image_dir,
        camera_mode=_camera_mode(cameras, single_camera),
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu,
    )
    pycolmap.match_exhaustive(db_path, matching_options=matching_options)

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


