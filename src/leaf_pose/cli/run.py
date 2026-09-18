"""Measure every leaf in a flat-lay capture.

    leaf-pose --input /mnt/d/PBR_Scans/2026-09-15-Naeem/gaensefuss_31 \
              --workdir runs/gaensefuss_31 --marker-mm 20 --visualize

Reads a folder of RAW frames (or a single photograph) and writes, into
`--workdir`:

    flat.jpg            the crossed-polariser mean, tone-mapped -- what was measured
    leaves.json         per leaf: contour, midrib, keypoints, sizes, veins
    detections.json     every blob found, accepted or rejected, with the reason
    masks/leaf_NNN.png  the full-resolution binary mask, on the leaf's own crop
    alpha/leaf_NNN.png  the matte behind it, for compositing
    normals/leaf_NNN.png  --photometric only: a measured OpenGL normal map
    leaves/leaf_NNN.png   --veins only: one leaf large, with its veins drawn
    leaf_poses.png      --visualize only: the whole result in one diagram

The stages are deliberately one command rather than six. Unlike the plant
pipeline next door, nothing here takes long enough to want resuming: the run
is dominated by decoding the RAW files, and re-deciding a midrib afterwards
costs less than re-reading them would.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from leaf_pose import instances as detect
from leaf_pose import keypoints as keypoint
from leaf_pose import midrib as rib
from leaf_pose import photometric as ps
from leaf_pose import raw as ingest
from leaf_pose import rig as riglib
from leaf_pose import scale as scalelib
from leaf_pose import veins as veinlib
from leaf_pose.record import LeafRecord


def say(message: str) -> None:
    print(message, flush=True)


def run(
    source: Path,
    workdir: Path,
    marker_mm: Optional[float] = None,
    backend: str = "colour",
    checkpoint: Optional[Path] = None,
    polarisation_order: str = "auto",
    detect_max_side: int = 2000,
    photometric: bool = False,
    working_distance_mm: Optional[float] = None,
    focal_length_mm: float = 50.0,
    sensor_width_mm: float = riglib.FULL_FRAME_WIDTH_MM,
    want_veins: bool = False,
    visualize: bool = False,
    show: bool = False,
    half_size: bool = False,
    limit: Optional[int] = None,
    min_solidity: float = detect.MIN_SOLIDITY,
    min_greenness: float = detect.MIN_GREENNESS,
    min_area_fraction: float = detect.MIN_AREA_FRACTION,
    device: str = "cuda",
) -> dict:
    started = time.time()
    workdir.mkdir(parents=True, exist_ok=True)

    # ---- the capture ------------------------------------------------------
    frames, is_raw = ingest.find_frames(source)
    say(f"==> {len(frames)} frame(s) in {source}")
    if polarisation_order == "auto" and len(frames) >= 4:
        say("  probing a few frames to tell the polarisation sets apart")
        polarisation_order = ingest.probe_order(frames)
        say(f"  {polarisation_order}")
    capture = ingest.split_polarisation(frames, polarisation_order)
    capture.is_raw = is_raw
    if capture.has_polarisation:
        say(f"  {len(capture.parallel)} parallel + {len(capture.crossed)} crossed "
            f"({capture.parallel[0].name}.. / {capture.crossed[0].name}..)")
    else:
        say(f"  unpolarised: {len(capture.crossed)} frame(s)")

    say("==> decoding")
    field = ingest.load_flat_field(capture, half_size=half_size, progress=say)
    height, width = field.shape
    say(f"  flat image {width}x{height}"
        + ("  + specular residual" if field.specular is not None else ""))
    cv2.imwrite(str(workdir / "flat.jpg"),
                cv2.cvtColor(ingest.tone_map(field.flat), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    # ---- scale ------------------------------------------------------------
    measured = scalelib.measure_scale(field.flat, marker_mm=marker_mm)
    if measured.mm_per_pixel is None and working_distance_mm is not None:
        # No marker size, but the stand was measured: the optics give the
        # scale on their own. Less accurate than the markers -- see
        # `rig.scale_from_optics` -- so it is only used when they cannot.
        measured.mm_per_pixel = riglib.scale_from_optics(
            focal_length_mm, working_distance_mm, width, sensor_width_mm)
        say(f"==> scale {measured.mm_per_pixel:.5f} mm/px, from "
            f"{focal_length_mm:.0f} mm at {working_distance_mm:.0f} mm "
            f"(no --marker-mm; the markers would be more accurate)")

    if measured.markers:
        say(f"==> {len(measured.markers)} {measured.dictionary} marker(s), "
            f"side {measured.markers[0].side_px:.1f} px"
            + (f", spread {measured.side_spread * 100:.2f}%"
               if len(measured.markers) > 1 else ""))
        if measured.mm_per_pixel:
            say(f"  scale {measured.mm_per_pixel:.5f} mm/px "
                f"({measured.pixels_per_mm:.1f} px/mm)")
        else:
            say("  no --marker-mm given, so sizes stay in pixels")
    else:
        say("==> no fiducial markers found; sizes stay in pixels")

    # ---- instances --------------------------------------------------------
    say(f"==> finding leaves ({backend})")
    limits = {"min_solidity": min_solidity, "min_greenness": min_greenness,
              "min_area_fraction": min_area_fraction}
    if backend == "sam":
        from pose_estimator.checkpoints import resolve_checkpoint
        report, work = detect.detect_sam(
            field.flat, resolve_checkpoint(checkpoint), max_side=detect_max_side,
            device=device, **limits)
    else:
        report, work = detect.detect_colour(
            field.flat, max_side=detect_max_side, **limits)

    say(f"  {len(report.accepted)} accepted, {len(report.rejected)} rejected")
    for reason in sorted({d.rejected for d in report.rejected if d.rejected}):
        count = sum(1 for d in report.rejected if d.rejected == reason)
        say(f"    {count:4d}  {reason}")
    (workdir / "detections.json").write_text(
        json.dumps({"work_scale": work, **report.to_dict()}, indent=2))

    accepted = report.accepted[:limit] if limit else report.accepted
    say("==> re-deciding each boundary at full resolution")
    found = []
    for number, detection in enumerate(accepted, start=1):
        instance = detect.refine_instance(field.flat, detection, work, number)
        if instance is not None:
            found.append(instance)
    say(f"  {len(found)} leaves matted")
    if not found:
        raise SystemExit("no leaves survived refinement -- see detections.json")

    # ---- photometric normals (optional) ------------------------------------
    # Keyed by leaf id, not by position. A leaf whose midrib cannot be fitted
    # is dropped from `records` but not from `found`, so any list indexed by
    # position drifts by one from that leaf onward -- and the symptom is one
    # leaf's normal map written under another leaf's name, which looks
    # entirely plausible in the output folder.
    ridges: Dict[int, np.ndarray] = {}
    normals: Dict[int, np.ndarray] = {}
    # The normal field itself is kept, not only the crease taken from it:
    # `find_veins` needs a crease at its own, much heavier smoothing.
    normal_fields: Dict[int, np.ndarray] = {}
    if photometric:
        rig_path = riglib.find_rig(source)
        if rig_path is None:
            say("==> no rigdef_cam.xml found; skipping photometric normals")
        elif capture.num_lights < 3:
            say("==> fewer than 3 lights; skipping photometric normals")
        else:
            rig = riglib.load_rig(rig_path)
            say(f"==> photometric stereo from {rig.num_lights} lights ({rig_path})")
            mm_per_pixel = measured.mm_per_pixel
            distance = working_distance_mm
            if distance is None and mm_per_pixel:
                distance = riglib.working_distance_mm(
                    mm_per_pixel, focal_length_mm, width, sensor_width_mm)
                say(f"  working distance {distance:.0f} mm, from the measured scale")
            if distance is None:
                # Only the relative directions matter for the crease this is
                # used for, so a plausible geometry beats refusing to run --
                # but say so, because the normals are then not metric.
                distance = 2.5 * rig.radius_mm
                mm_per_pixel = mm_per_pixel or (4.0 * rig.radius_mm / max(width, 1))
                say(f"  no scale and no --working-distance-mm: assuming "
                    f"{distance:.0f} mm. Creases stay usable; the normals are "
                    f"not metric.")

            boxes = [i.bbox for i in found]
            stacks = ps.crop_light_stack(capture.crossed, boxes, progress=say)
            for instance, stack in zip(found, stacks):
                x0, y0, x1, y1 = instance.bbox
                directions = riglib.light_directions(
                    rig, (y1 - y0, x1 - x0), mm_per_pixel, distance,
                    origin_xy=(x0, y0), centre_xy=(width / 2.0, height / 2.0))
                solved = ps.solve(stack, directions, instance.mask)
                ridges[instance.index] = solved.ridge
                normals[instance.index] = solved.normal_map_rgb()
                if want_veins:
                    normal_fields[instance.index] = solved.normals
                del directions, stack
            say(f"  normals for {len(found)} leaves")

    # ---- midribs, keypoints, veins ----------------------------------------
    say("==> midribs and keypoints")
    mm_per_pixel = measured.mm_per_pixel
    records: List[LeafRecord] = []
    for number, instance in enumerate(found, start=1):
        x0, y0, x1, y1 = instance.bbox
        crop = field.flat[y0:y1, x0:x1]
        # Said before the work, not after. This stage is the slowest in the
        # run on a high-resolution flat-lay and it used to print only on
        # failure, so a capture whose leaves are a few thousand pixels across
        # looked indistinguishable from a hang for as long as it took.
        say(f"  leaf {instance.index} ({number}/{len(found)}) "
            f"{x1 - x0}x{y1 - y0} px")
        specular = field.specular[y0:y1, x0:x1] if field.specular is not None else None

        fitted = rib.fit_midrib(instance.mask, image=crop, specular=specular,
                                normal_ridge=ridges.get(instance.index))
        if fitted is None:
            say(f"  leaf {instance.index}: no midrib -- skipped")
            continue
        oriented, points = keypoint.locate(
            fitted, greenness=detect.greenness_map(crop) * instance.mask)

        found_veins = []
        if want_veins:
            found_veins = veinlib.find_veins(instance.mask, crop, oriented,
                                             normals=normal_fields.get(instance.index))

        records.append(LeafRecord(leaf_id=instance.index, instance=instance,
                                  midrib=oriented, keypoints=points,
                                  veins=found_veins, mm_per_pixel=mm_per_pixel))

    say(f"  {len(records)} leaves measured")
    low = [r.leaf_id for r in records if r.keypoints.confidence < 0.15]
    if low:
        say(f"  which end is the petiole is uncertain for leaf(s) {low} -- "
            f"check them in the diagram")

    # ---- write out --------------------------------------------------------
    say("==> writing")
    for folder in ("masks", "alpha"):
        (workdir / folder).mkdir(exist_ok=True)
    for record in records:
        stem = f"leaf_{record.leaf_id:03d}.png"
        cv2.imwrite(str(workdir / "masks" / stem),
                    (record.instance.mask * 255).astype(np.uint8))
        cv2.imwrite(str(workdir / "alpha" / stem),
                    (record.instance.alpha * 255).astype(np.uint8))
        if record.leaf_id in normals:
            (workdir / "normals").mkdir(exist_ok=True)
            cv2.imwrite(str(workdir / "normals" / stem),
                        cv2.cvtColor(normals[record.leaf_id], cv2.COLOR_RGB2BGR))

    summary = {
        "source": str(source),
        "frames": len(frames),
        "polarised": capture.has_polarisation,
        "image_size": [width, height],
        "scale": measured.to_dict(),
        "units": records[0].units if records else "px",
        "photometric": bool(ridges),
        "leaves": [r.to_dict() for r in records],
    }
    (workdir / "leaves.json").write_text(json.dumps(summary, indent=2))

    if visualize:
        say("==> drawing")
        from leaf_pose import viz
        viz.plot_overview(records, field.flat, workdir / "leaf_poses.png",
                          title=source.name, show=show)
        if want_veins:
            for record in records:
                x0, y0, x1, y1 = record.instance.bbox
                viz.plot_leaf_detail(record, field.flat[y0:y1, x0:x1],
                                     workdir / "leaves" /
                                     f"leaf_{record.leaf_id:03d}.png")
        say(f"  {workdir / 'leaf_poses.png'}")

    say(f"==> done in {time.time() - started:.1f}s -- {workdir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.split("\n\n")[1:]))
    parser.add_argument("--input", type=Path, required=True,
                        help="capture folder of RAW frames, or one image file")
    parser.add_argument("--workdir", type=Path, required=True,
                        help="where results are written")
    parser.add_argument("--marker-mm", type=float, default=None,
                        help="printed side length of the fiducial markers, in mm. "
                             "Without it every size is reported in pixels.")
    parser.add_argument("--instances", dest="backend", default="colour",
                        choices=("colour", "sam"),
                        help="how leaves are told apart: colour (default, exact on "
                             "a flat lay) or sam (for touching or overlapping leaves)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="SAM2 weights, for --instances sam")
    parser.add_argument("--polarisation-order", default="auto",
                        choices=("auto", "parallel-first", "crossed-first"),
                        help="which half of the frames is which (default: decide "
                             "from brightness)")
    parser.add_argument("--photometric", action="store_true",
                        help="also solve surface normals from the individual "
                             "lights, and use the crease in them to place the "
                             "midrib. Costs one more pass over the RAW files.")
    parser.add_argument("--working-distance-mm", type=float, default=None,
                        help="lens-to-subject distance, for --photometric. "
                             "Derived from --marker-mm when not given.")
    parser.add_argument("--focal-length-mm", type=float, default=50.0,
                        help="lens focal length, used to derive the working "
                             "distance (default %(default)s)")
    parser.add_argument("--sensor-width-mm", type=float,
                        default=riglib.FULL_FRAME_WIDTH_MM,
                        help="sensor width, used to derive the working "
                             "distance (default %(default)s, full frame)")
    parser.add_argument("--veins", dest="want_veins", action="store_true",
                        help="also look for secondary veins, and write a per-leaf "
                             "detail image for each")
    parser.add_argument("--visualize", action="store_true",
                        help="write leaf_poses.png, the whole result in one diagram")
    parser.add_argument("--show", action="store_true",
                        help="open the diagram in a window as well as saving it")
    parser.add_argument("--half-size", action="store_true",
                        help="decode the RAW files at half resolution -- four times "
                             "faster, for a first look")
    parser.add_argument("--detect-max-side", type=int, default=2000,
                        help="resolution leaves are told apart at (default 2000); "
                             "boundaries are always re-decided at full resolution")
    parser.add_argument("--limit", type=int, default=None,
                        help="only measure the first N leaves")
    parser.add_argument("--min-solidity", type=float, default=detect.MIN_SOLIDITY,
                        help="blob area over convex-hull area below which a blob is "
                             "not a leaf (default %(default)s; the root system "
                             "scores about 0.08)")
    parser.add_argument("--min-greenness", type=float, default=detect.MIN_GREENNESS,
                        help="excess-green level below which a blob is not a leaf "
                             "(default %(default)s)")
    parser.add_argument("--min-area-fraction", type=float,
                        default=detect.MIN_AREA_FRACTION,
                        help="fraction of the frame below which a blob is debris "
                             "(default %(default)s)")
    parser.add_argument("--device", default="cuda", help="torch device for --instances sam")
    args = parser.parse_args()

    run(source=args.input, workdir=args.workdir, marker_mm=args.marker_mm,
        backend=args.backend, checkpoint=args.checkpoint,
        polarisation_order=args.polarisation_order,
        detect_max_side=args.detect_max_side, photometric=args.photometric,
        working_distance_mm=args.working_distance_mm,
        focal_length_mm=args.focal_length_mm, sensor_width_mm=args.sensor_width_mm,
        want_veins=args.want_veins,
        visualize=args.visualize, show=args.show, half_size=args.half_size,
        limit=args.limit, min_solidity=args.min_solidity,
        min_greenness=args.min_greenness, min_area_fraction=args.min_area_fraction,
        device=args.device)


if __name__ == "__main__":
    main()
