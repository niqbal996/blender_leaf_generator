"""Thin wrapper around pycolmap for sparse structure-from-motion.

`pycolmap` is an optional dependency (`pip install -e ".[skeleton]"`) and is
imported lazily so the rest of this subpackage stays usable without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def _group_by_focal(
    image_names: List[str], focal_priors: Dict[str, float]
) -> List[Tuple[Optional[float], List[str]]]:
    """Split image names into one group per distinct EXIF focal length.

    Frames whose EXIF was silent land together under None and are left to
    COLMAP's own guess -- a mixed video+photo capture still solves, with the
    photo passes correctly seeded and the video pass no worse off than before.

    Focals are keyed to 0.1px because they come from an integer EXIF field
    times a constant, so frames from one lens setting agree exactly; the
    rounding only guards against float formatting noise, not a real spread.
    """
    groups: Dict[Optional[float], List[str]] = {}
    for name in image_names:
        focal = focal_priors.get(Path(name).stem)
        key = round(float(focal), 1) if focal else None
        groups.setdefault(key, []).append(name)
    # Named groups first, largest first; the unknown bucket last so its images
    # cannot claim camera_id 1 and mislead anyone reading the database.
    known = sorted((k for k in groups if k is not None), key=lambda k: (-len(groups[k]), k))
    order = known + ([None] if None in groups else [])
    return [(key, sorted(groups[key])) for key in order]


def _camera_mode(cameras: str, single_camera: bool):
    """COLMAP's camera grouping, defaulting to one shared camera.

    One camera for the whole capture is right for this rig and wrong the
    moment the lens is touched. Measured on sugarbeet_3, whose three passes
    were shot at 48mm, 32mm and 22mm: forcing a single focal across a 2.2x
    zoom range gives a compromise that fits none of them, and the passes come
    back interleaved rather than as three orbits.

    This is the fallback ladder, not the answer. The answer is `exif` (the
    default), which reads the focal off each photo in P1 and gives each
    distinct lens setting its own camera -- the truth, and one free focal per
    *lens* rather than per image. It is handled before this function is
    reached and only falls through to here when P1 recorded no focal at all.

    `per-image` remains the escape hatch for that case: EXIF-stripped photos
    at mixed zooms, where nothing can be inferred and one shared camera is
    known to be wrong. It costs a free focal per frame -- enough rope for the
    solver to absorb real drift into intrinsics -- so prefer `exif` whenever
    the photos still carry their tags. Per-folder grouping never applies here
    because the passes share one frames directory.
    """
    import pycolmap

    if cameras == "per-image":
        return pycolmap.CameraMode.PER_IMAGE
    if cameras == "auto":
        return pycolmap.CameraMode.AUTO
    return pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO


def _extract_with_focal_priors(
    db_path, image_dir, focal_priors, mask_dir, extraction_options, device
) -> List[Tuple[Optional[float], List[str]]]:
    """Extract features once per focal group, each into its own SINGLE camera.

    `extract_features` appends to the database it is given and takes an
    explicit `image_names`, so calling it once per group leaves one shared
    camera per group -- the honest model for a capture whose passes were shot
    at different zooms, and the one COLMAP cannot infer for itself once P1 has
    re-encoded the EXIF away.

    Seeding `camera_params` also marks the focal as a prior, so bundle
    adjustment starts from the lens that was actually on the body and refines
    from there, rather than starting 40% out and refining into whatever fits.
    """
    import cv2
    import pycolmap

    image_dir = Path(image_dir)
    names = sorted(
        path.name for path in image_dir.iterdir()
        if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    groups = _group_by_focal(names, focal_priors)

    for focal, group_names in groups:
        options = pycolmap.ImageReaderOptions()
        if mask_dir is not None:
            options.mask_path = str(mask_dir)
        if focal is not None:
            image = cv2.imread(str(image_dir / group_names[0]))
            if image is None:
                raise ValueError(f"could not read {image_dir / group_names[0]}")
            height, width = image.shape[:2]
            options.camera_model = "SIMPLE_RADIAL"
            options.camera_params = (
                f"{focal:.6f},{width / 2.0:.6f},{height / 2.0:.6f},0.0")
        pycolmap.extract_features(
            db_path,
            image_dir,
            image_names=group_names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=options,
            extraction_options=extraction_options,
            device=device,
        )

    return groups


# --------------------------------------------------------------------------
# Scene connectivity: did every capture pass end up in ONE scene?
# --------------------------------------------------------------------------
#
# COLMAP builds a scene by adding one image at a time. When a group of images
# cannot be tied to the rest -- too little overlap, too big a viewpoint jump --
# it does not fail. It quietly starts a *second* scene and carries on, and
# `incremental_mapping` hands back every scene it built.
#
# Taking the largest and returning it (which is all this module used to do)
# therefore discards the others in silence. A complete, internally perfect
# top-down pass can vanish with nothing to show for it but a lower registered
# count, which reads exactly like a few blurry frames.
#
# Nothing below is tuned to a specimen. "Did pass 1 end up in a scene of its
# own" is a fact about the capture, not a measurement of the plant, so it
# behaves identically on every run.


def _pass_of(name: str, sources: Optional[Dict[str, int]]) -> int:
    return int((sources or {}).get(Path(name).stem, 0))


def _pass_counts(names, sources: Optional[Dict[str, int]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for name in names:
        key = _pass_of(name, sources)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _fmt_passes(counts: Dict[int, int]) -> str:
    return ", ".join(f"pass {p}: {n}" for p, n in counts.items()) if counts else "none"


def _registered_names(reconstruction) -> set:
    return {reconstruction.images[i].name for i in reconstruction.reg_image_ids()}


def describe_connectivity(models: dict, best, all_names, sources=None) -> dict:
    """Which capture pass landed in which scene -- structural, no thresholds.

    `models` is what `incremental_mapping` returned, `best` the one that would
    be kept. The report names any scene that was built and would be thrown
    away, any image in no scene at all, and -- the condition that actually
    matters -- any pass with *zero* images in the winning scene.
    """
    won = _registered_names(best)
    claimed = set(won)
    discarded = []
    for model_id, model in sorted(models.items()):
        names = _registered_names(model)
        if names == won:
            continue
        claimed |= names
        discarded.append({
            "model": int(model_id),
            "num_images": len(names),
            "passes": _pass_counts(names, sources),
        })

    nowhere = [n for n in all_names if n not in claimed]
    totals = _pass_counts(all_names, sources)
    kept = _pass_counts([n for n in all_names if n in won], sources)

    return {
        "num_models": len(models),
        "winner": {"num_images": len(won), "passes": kept},
        "discarded_models": discarded,
        "unregistered": sorted(nowhere),
        "unregistered_passes": _pass_counts(nowhere, sources),
        "passes_total": totals,
        # The catastrophic case: a whole pass contributed nothing to the scene
        # everything downstream will be built from.
        "passes_absent_from_winner": [p for p in totals if kept.get(p, 0) == 0],
    }


def log_connectivity(stage: str, info: dict) -> None:
    """Print the connectivity report. Explicit by design -- this is the failure
    that used to leave no trace at all."""
    print(f"  [{stage}] scenes built: {info['num_models']}")
    print(f"  [{stage}] winning scene: {info['winner']['num_images']} image(s) -- "
          f"{_fmt_passes(info['winner']['passes'])}")
    for other in info["discarded_models"]:
        print(f"  [{stage}] SEPARATE scene {other['model']}: {other['num_images']} image(s) -- "
              f"{_fmt_passes(other['passes'])}")
        print(f"  [{stage}]   ^ built fine, but shares no images with the winning scene, "
              f"so keeping the winner throws it away")
    if info["unregistered"]:
        print(f"  [{stage}] in no scene at all: {len(info['unregistered'])} image(s) -- "
              f"{_fmt_passes(info['unregistered_passes'])}")
    for index in info["passes_absent_from_winner"]:
        print(f"  [{stage}] *** pass {index} contributed ZERO images to the winning scene "
              f"({info['passes_total'].get(index, 0)} frames shot) ***")


def capture_guidance(info: dict, low_texture: bool) -> List[str]:
    """What to change when the solver has genuinely run out of moves.

    Matching here is *exhaustive*: every image has already been compared
    against every other image, so there is no "try harder" left in the code.
    What remains are descriptor robustness and the capture itself, and both
    are specimen-independent -- which is why this is printed advice rather
    than another parameter to fit.
    """
    if not info["discarded_models"] and not info["unregistered"]:
        return []

    lines = ["why this happens and what fixes it:"]
    split_passes = sorted({p for other in info["discarded_models"] for p in other["passes"]})
    if split_passes:
        lines.append(
            f"  - frames from {_fmt_passes({p: info['passes_total'].get(p, 0) for p in split_passes})} "
            f"match each other but not the winning scene. Exhaustive matching already "
            f"compared every image pair, so this is missing overlap in the capture, "
            f"not a setting that needs loosening.")
        lines.append(
            "  - shoot the joining pass as a CONTINUOUS climb in elevation starting from "
            "the height of the pass it must join, not as a separate cluster at the top. "
            "Every frame then has a near neighbour and the chain carries it in.")
    if info["unregistered"] and not split_passes:
        lines.append(
            "  - these frames matched nothing well enough to place. Usually motion blur or "
            "too large a viewpoint step from their neighbours.")
    if not low_texture:
        lines.append(
            "  - re-run with --low-texture. It turns on viewpoint- and scale-robust "
            "descriptors (estimate_affine_shape, domain_size_pooling), which survive a "
            "much larger viewpoint change. Costs roughly 3-5x the extraction time.")
    lines.append(
        "  - the holder bridges elevations best: rigid, three-dimensional and "
        "non-repeating. The turntable disc is weaker (it foreshortens sharply between a "
        "low and a high camera, and a repeating checkerboard aliases); the plant is "
        "weakest of all -- smooth, self-occluding and slightly mobile.")
    return lines


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
    cameras: str = "exif",
    focal_priors: Optional[Dict[str, float]] = None,
    sources: Optional[Dict[str, int]] = None,
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

    `focal_priors` maps frame stem to focal length in pixels, as recorded by
    P1 from EXIF (`p1/intrinsics.json`). With `cameras="exif"` -- the default
    -- frames are grouped by that focal and each group gets one camera seeded
    with it, so a capture shot at several zooms solves as several honest
    cameras. With no priors the behaviour is exactly `single`, which is what
    video frames and EXIF-stripped photos get.

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

    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

    if cameras == "exif" and focal_priors:
        groups = _extract_with_focal_priors(
            db_path, image_dir, focal_priors, mask_dir, extraction_options, device)
        known = [(focal, names) for focal, names in groups if focal is not None]
        summary = ", ".join(f"{focal:.0f}px x{len(names)}" for focal, names in known)
        unknown = next((names for focal, names in groups if focal is None), [])
        print(f"  EXIF intrinsics: {len(known)} camera(s) seeded from the lens -- {summary}"
              + (f"; {len(unknown)} frame(s) without EXIF left to COLMAP's guess" if unknown else ""))
    else:
        if cameras == "exif":
            print("  no EXIF focal lengths recorded for these frames "
                  "(video, or photos with stripped EXIF) -- using one shared camera")
        pycolmap.extract_features(
            db_path,
            image_dir,
            camera_mode=_camera_mode(cameras, single_camera),
            reader_options=reader_options,
            extraction_options=extraction_options,
            device=device,
        )

    pycolmap.match_exhaustive(db_path, matching_options=matching_options)

    all_names = sorted(
        path.name for path in image_dir.iterdir()
        if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    print(f"  mapping: placing {len(all_names)} image(s) into a scene...")
    reconstructions = pycolmap.incremental_mapping(db_path, image_dir, sparse_dir)

    if not reconstructions:
        raise RuntimeError(
            "COLMAP could not reconstruct any model from these frames -- "
            "likely too few/blurry matches. Try more frames, better masking, "
            "or steadier rotation."
        )

    best = max(reconstructions.values(), key=lambda rec: rec.num_reg_images())

    # --- Layer 1: say what happened -------------------------------------
    first = describe_connectivity(reconstructions, best, all_names, sources)
    log_connectivity("mapping", first)

    # --- Layer 2: give the leftovers a second, fairer attempt -----------
    #
    # COLMAP allows each image three tries to join the scene, and those tries
    # are spent whenever the mapper happens to reach it. An image is therefore
    # judged against whatever the scene contained at that moment -- so frames
    # of a viewpoint the scene has barely covered yet (a top-down pass, early
    # on) can exhaust their attempts against a half-built scene and never be
    # reconsidered once it has grown enough to accept them.
    #
    # `incremental_mapping` takes an `input_path`, which loads a finished scene
    # and continues from it with a fresh mapper -- and therefore fresh attempt
    # counters. So we hand back the scene we just built and let the leftovers
    # try again against the complete thing. Nothing is loosened: no threshold
    # moves, the images simply get their attempt under the best conditions
    # available instead of the accidental ones. `multiple_models=False` keeps
    # it from wandering off and starting yet another scene.
    recovery = None
    if best.num_reg_images() < len(all_names):
        missing = len(all_names) - best.num_reg_images()
        print(f"  {missing} image(s) outside the winning scene -- retrying them "
              f"against the finished scene (no thresholds changed)...")
        seed_dir = sparse_dir / "stage1_best"
        seed_dir.mkdir(parents=True, exist_ok=True)
        best.write(str(seed_dir))

        options = pycolmap.IncrementalPipelineOptions()
        options.multiple_models = False
        continued_dir = sparse_dir / "continued"
        continued_dir.mkdir(parents=True, exist_ok=True)
        try:
            again = pycolmap.incremental_mapping(
                db_path, image_dir, continued_dir,
                options=options, input_path=str(seed_dir))
        except Exception as exc:  # pragma: no cover - depends on pycolmap build
            again = {}
            print(f"  second attempt could not run ({type(exc).__name__}: {exc}); "
                  f"keeping the first result")

        if again:
            candidate = max(again.values(), key=lambda rec: rec.num_reg_images())
            gained = candidate.num_reg_images() - best.num_reg_images()
            if gained > 0:
                print(f"  recovered {gained} image(s): "
                      f"{best.num_reg_images()} -> {candidate.num_reg_images()} registered")
                best = candidate
                reconstructions = again
            else:
                print(f"  no images recovered (still {best.num_reg_images()}) -- "
                      f"the missing frames genuinely do not overlap this scene")
        recovery = describe_connectivity(reconstructions, best, all_names, sources)
        log_connectivity("after retry", recovery)

    final = recovery or first

    # --- Layer 3: when the solver is out of moves, say what to change ----
    guidance = capture_guidance(final, low_texture)
    for line in guidance:
        print(f"  {line}")

    # Persist the winning model under a stable name -- `incremental_mapping`
    # numbers its candidate models (0, 1, 2, ...) and the winner's number is
    # otherwise lost once we return just the in-memory object, so nothing
    # downstream (splat training, re-alignment) could cheaply reload just
    # this model without rerunning feature extraction/matching from scratch.
    best_dir = sparse_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best.write(str(best_dir))

    # On disk so `evaluate_poses` can check it, `--reuse-sparse` can re-read it
    # without re-solving, and a finished run still carries the evidence.
    report = {
        "num_input_images": len(all_names),
        "mapping": first,
        "after_retry": recovery,
        "final": final,
        "recovered_images": (
            (recovery["winner"]["num_images"] - first["winner"]["num_images"])
            if recovery else 0),
        "guidance": guidance,
    }
    with open(output_dir / "solve.json", "w") as handle:
        json.dump(report, handle, indent=2)

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


