#!/usr/bin/env python
"""Draw a finished leaf-pose run, from the files it already wrote.

    python scripts/view_leaf_poses.py --workdir /path/to/leaf_poses

`leaf-pose --visualize` writes leaf_poses.png at the end of a run. Without
that flag the run still writes everything the diagram is made of -- the flat
image, and leaves.json with every contour, midrib and keypoint in frame
coordinates -- so the picture can be drawn afterwards without going near the
RAW files again.

That matters because the flag is the only part that is expensive to have
missed: a run is dominated by decoding 24 RAW frames, and re-running it purely
to redraw would pay that again for a picture whose inputs are already on disk.

Writes leaf_poses.png next to leaves.json (or --out elsewhere), and prints the
measurement table, which is the part of a run you would otherwise read out of
JSON by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# BGR. The midrib is the measurement; the contour is context, so it is drawn
# thinner and cooler. Keypoints are the three the pipeline actually commits to.
CONTOUR = (90, 200, 90)
MIDRIB = (60, 60, 240)
TIP = (60, 220, 255)
BLADE_BASE = (240, 200, 60)
PETIOLE_ORIGIN = (230, 100, 220)
UNCERTAIN = (60, 160, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(canvas, text, xy, colour, scale, thick=2):
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(canvas, text, (x, y), FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def draw(flat, leaves, low_confidence: float):
    canvas = flat.copy()
    # Line widths in pixels of a 9568px-wide scan would be invisible at the
    # size anyone actually looks at this, so everything scales with the image.
    unit = max(2, int(round(max(canvas.shape[:2]) / 900)))
    scale = max(0.8, max(canvas.shape[:2]) / 1800.0)

    for leaf in leaves:
        keypoints = leaf["keypoints"]
        uncertain = keypoints.get("confidence", 1.0) < low_confidence

        contour = np.asarray(leaf["contour"], np.int32)
        if len(contour):
            cv2.polylines(canvas, [contour.reshape(-1, 1, 2)], True,
                          UNCERTAIN if uncertain else CONTOUR, unit, cv2.LINE_AA)

        midrib = np.asarray(leaf["midrib"], np.int32)
        if len(midrib) > 1:
            cv2.polylines(canvas, [midrib.reshape(-1, 1, 2)], False, MIDRIB,
                          unit * 2, cv2.LINE_AA)

        for name, colour in (("tip", TIP), ("blade_base", BLADE_BASE),
                             ("petiole_origin", PETIOLE_ORIGIN)):
            point = keypoints.get(name)
            if point is None:
                continue
            centre = (int(point[0]), int(point[1]))
            cv2.circle(canvas, centre, unit * 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, centre, unit * 4, colour, -1, cv2.LINE_AA)

        # The id goes on the blade, not the centroid of the bounding box: on a
        # long curved leaf the box centre often falls outside the leaf.
        anchor = midrib[len(midrib) // 2] if len(midrib) else contour[0]
        text = f"{leaf['id']}" + ("  ?" if uncertain else "")
        label(canvas, text, (anchor[0] + unit * 4, anchor[1]),
              UNCERTAIN if uncertain else (255, 255, 255), scale * 1.4, unit)
    return canvas, unit, scale


def legend(canvas, units, low_confidence, unit, scale):
    entries = [("midrib", MIDRIB), ("outline", CONTOUR), ("tip", TIP),
               ("blade base", BLADE_BASE), ("petiole origin", PETIOLE_ORIGIN),
               (f"'?' = petiole end uncertain (confidence < {low_confidence:g})", UNCERTAIN)]
    pad = int(30 * scale)
    line = int(46 * scale)
    band = np.full((pad * 2 + line * len(entries), canvas.shape[1], 3), 32, np.uint8)
    for row, (text, colour) in enumerate(entries):
        y = pad + row * line + int(line * 0.7)
        cv2.rectangle(band, (pad, y - int(line * 0.5)),
                      (pad + int(line * 0.8), y), colour, -1)
        cv2.putText(band, text, (pad * 2 + int(line * 0.8), y), FONT, scale,
                    (235, 235, 235), max(1, unit // 2), cv2.LINE_AA)
    cv2.putText(band, f"sizes in {units}", (canvas.shape[1] - int(260 * scale), pad + line),
                FONT, scale, (235, 235, 235), max(1, unit // 2), cv2.LINE_AA)
    return np.vstack([band, canvas])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path,
                        help="a finished leaf-pose --workdir (holds leaves.json + flat.jpg)")
    parser.add_argument("--out", type=Path, help="default: <workdir>/leaf_poses.png")
    parser.add_argument("--max-side", type=int, default=2600,
                        help="downscale the saved picture to this long edge. The scan is "
                             "60MP and nothing displays that; 0 keeps full resolution")
    parser.add_argument("--low-confidence", type=float, default=0.65,
                        help="mark a leaf whose petiole end was decided below this")
    args = parser.parse_args()

    report_path = args.workdir / "leaves.json"
    if not report_path.exists():
        raise SystemExit(f"{report_path} not found -- is that a finished leaf-pose workdir?")
    report = json.loads(report_path.read_text())
    leaves = report.get("leaves", [])
    if not leaves:
        raise SystemExit(f"{report_path} lists no leaves")

    flat_path = args.workdir / "flat.jpg"
    if not flat_path.exists():
        raise SystemExit(f"{flat_path} not found -- it is what the diagram is drawn on")
    flat = cv2.imread(str(flat_path))
    if flat is None:
        raise SystemExit(f"could not read {flat_path}")

    units = report.get("units", "px")
    # `scale` is the whole calibration record -- dictionary, spread, every
    # marker's corners -- not a number.
    mm_per_px = (report.get("scale") or {}).get("mm_per_pixel")
    scale_note = (f"{mm_per_px:.5f} mm/px ({1 / mm_per_px:.1f} px/mm)" if mm_per_px
                  else "no scale -- sizes in pixels")
    print(f"{len(leaves)} leaves from {report.get('source', '?')}  ({scale_note})")
    print(f"\n  {'id':>3} {'blade len':>10} {'petiole':>9} {'width':>8} {'area':>10} "
          f"{'rib support':>12} {'petiole end':>12}")
    for leaf in leaves:
        confidence = leaf["keypoints"].get("confidence", 1.0)
        flag = "uncertain" if confidence < args.low_confidence else "ok"
        print(f"  {leaf['id']:>3} {leaf['blade_length']:>10.1f} {leaf['petiole_length']:>9.1f} "
              f"{leaf['blade_width_max']:>8.1f} {leaf['area']:>10.1f} "
              f"{leaf['midrib_ridge_support']:>12.3f} {flag:>12} ({confidence:.2f})")
    print(f"  sizes in {units}")

    canvas, unit, scale = draw(flat, leaves, args.low_confidence)
    canvas = legend(canvas, units, args.low_confidence, unit, scale)
    if args.max_side and max(canvas.shape[:2]) > args.max_side:
        factor = args.max_side / max(canvas.shape[:2])
        canvas = cv2.resize(canvas, (int(canvas.shape[1] * factor),
                                     int(canvas.shape[0] * factor)),
                            interpolation=cv2.INTER_AREA)

    out = args.out or (args.workdir / "leaf_poses.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"\nwrote {out}  ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
