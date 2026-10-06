#!/usr/bin/env python
"""Score a P5x skeleton against the same plant's flat-lay measurement.

    python scripts/score_against_flatlay.py \
        --workdir /mnt/e/.../gaensefuss_1/plant \
        --flat-lay runs/gaensefuss_1_leaves

The flat lay is the only external truth this pipeline has. The plant has been
taken apart and photographed in one plane against a marker of known size, so
`leaf-pose` returns a leaf count and a blade length in millimetres for every
leaf -- quantities the turntable reconstruction has to guess at.

**This script reports. It does not choose.** No threshold is searched and no
parameter is fitted to make a number look better; the one quantity solved for
is the single global millimetre-per-unit scale, because structure from motion
genuinely cannot recover it and it has to come from somewhere. Everything else
is measurement against a fixed answer. Tuning a pipeline constant until this
script prints a good number is how you get something that works on one plant,
which is the failure this file exists to make visible rather than to enable.

Three numbers matter and they are reported separately, because they fail
separately:

  recall    how many of the flat lay's leaves reached 3D at all. This is the
            pipeline's real weakness and it is not improved by making the
            leaves it *does* find more accurate.
  accuracy  RMS of blade length, in mm, over the leaves it did find.
  scale     mm per reconstruction unit. Reported because its *stability*
            across runs is evidence the leaf matching is real: a fitted scale
            that swings between runs means the correspondence is noise.

The matching is by rank -- the n longest 3D midribs against the n longest
flat-lay blades -- which assumes the reconstruction finds the biggest leaves
first. That assumption is stated in the output rather than hidden, because it
is the weakest link here: where recall is low the small leaves are matched
against the wrong partners and the RMS is pessimistic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# A midrib shorter than this many voxels is not a leaf that was measured
# badly, it is a leaf that was not traced: the path never entered the blade.
# Reported on its own rather than folded into the accuracy number, which it
# would otherwise quietly poison.
STUB_VOXELS = 2.0


def load_flat_lay(path: Path) -> list:
    data = json.loads((path / "leaves.json").read_text())
    if data.get("units") != "mm":
        raise SystemExit(
            f"{path}/leaves.json is in '{data.get('units')}', not mm -- re-run leaf-pose "
            "with --marker-mm so the blades carry a real length")
    return sorted((leaf["blade_length"] for leaf in data["leaves"]), reverse=True)


def load_skeleton(path: Path) -> tuple:
    data = json.loads((path / "skeleton.json").read_text())
    lengths = sorted((leaf["midrib_length"] for leaf in data["leaves"]), reverse=True)
    return lengths, data.get("unreachable", []), data.get("dropped_because", {})


def voxel_of(workdir: Path) -> float:
    hull = json.loads((workdir / "p4" / "hull.json").read_text())
    return float(hull["voxel_size"])


def score(workdir: Path, flat_lay: Path) -> dict:
    blades = load_flat_lay(flat_lay)
    lengths, unreachable, why = load_skeleton(workdir / "p5x")
    voxel = voxel_of(workdir)

    usable = [x for x in lengths if x >= STUB_VOXELS * voxel]
    stubs = len(lengths) - len(usable)
    n = min(len(usable), len(blades))
    if n == 0:
        raise SystemExit("no usable midribs to score")

    three = np.array(usable[:n])
    truth = np.array(blades[:n])
    # One global scale, least squares. This is the only fitted quantity.
    mm_per_unit = float((truth * three).sum() / (three * three).sum())
    residual = three * mm_per_unit - truth
    rms = float(np.sqrt((residual ** 2).mean()))

    return {
        "workdir": str(workdir),
        "voxel_mm": voxel * mm_per_unit,
        "leaves_flat_lay": len(blades),
        "midribs_traced": len(lengths),
        "midribs_usable": len(usable),
        "stubs": stubs,
        "unreachable": len(unreachable),
        "dropped_because": why,
        "recall": len(usable) / len(blades),
        "mm_per_unit": mm_per_unit,
        "rms_mm": rms,
        "rms_relative": rms / float(truth.mean()),
        "worst_mm": float(np.abs(residual).max()),
        "pairs": [(float(a * mm_per_unit), float(b)) for a, b in zip(three, truth)],
    }


def report(s: dict, verbose: bool = False) -> None:
    print(f"  {s['workdir']}")
    print(f"    voxel                 {s['voxel_mm']:.2f} mm")
    print(f"    leaves (flat lay)     {s['leaves_flat_lay']}")
    print(f"    midribs traced        {s['midribs_traced']}"
          f"  ({s['stubs']} stubs, {s['unreachable']} unreachable)")
    print(f"    RECALL                {s['recall']:.0%}"
          f"   ({s['midribs_usable']}/{s['leaves_flat_lay']} leaves reached 3D)")
    print(f"    ACCURACY              {s['rms_mm']:.2f} mm RMS"
          f"  ({s['rms_relative']:.1%}), worst {s['worst_mm']:.1f} mm")
    print(f"    scale                 {s['mm_per_unit']:.1f} mm/unit")
    if s["dropped_because"]:
        print(f"    dropped: {s['dropped_because']}")
    if verbose:
        print("       3D (mm)   flat lay (mm)   error")
        for a, b in s["pairs"]:
            print(f"      {a:8.1f}   {b:11.1f}   {a-b:+6.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", required=True, type=Path, nargs="+",
                   help="plant workdir(s) holding p5x/ and p4/. Several may be given, "
                        "and are reported side by side -- which is how a change is judged")
    p.add_argument("--flat-lay", required=True, type=Path,
                   help="the leaf-pose run for the SAME plant")
    p.add_argument("--verbose", action="store_true", help="per-leaf table too")
    p.add_argument("--json", type=Path, help="also write the scores here")
    args = p.parse_args()

    out = []
    for w in args.workdir:
        s = score(w, args.flat_lay)
        report(s, args.verbose)
        print()
        out.append(s)

    if len(out) > 1:
        print("  side by side:")
        print(f"    {'workdir':<28} {'voxel':>7} {'recall':>7} {'RMS mm':>8} {'scale':>8}")
        for s in out:
            print(f"    {Path(s['workdir']).name:<28} {s['voxel_mm']:6.2f}  "
                  f"{s['recall']:6.0%} {s['rms_mm']:8.2f} {s['mm_per_unit']:8.1f}")
        scales = [s["mm_per_unit"] for s in out]
        spread = (max(scales) - min(scales)) / float(np.mean(scales))
        print(f"\n    fitted scale varies by {spread:.1%} across these runs "
              f"-- a large spread means the leaf matching is not stable, "
              f"so believe the recall before the RMS")

    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
