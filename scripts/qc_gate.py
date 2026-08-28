"""Decide whether a phase's QC is catastrophically bad: `qc_gate.py <phase> <qc.json>`.

Prints one line per fatal condition (empty output = proceed). Only failures
that poison every later phase count; advisory failures happen on good runs
and never gate. Each signature below was measured in a real disaster that
previously ran to completion and produced an empty result -- see
DECISIONS.md 2026-08-26.
"""

import json
import sys


def fatal_conditions(phase: str, report: dict) -> list:
    checks = report.get("checks", {})

    def failed(name):
        c = checks.get(name)
        return c is not None and not c.get("pass", True)

    fatal = []
    if phase == "p1p2":
        # The "mask IS the pliers" signature. Good runs measured 0.9-19%
        # colour contamination (a root ball in front of the yellow handle);
        # the colour-rule disasters measured 99%.
        worst = report.get("holder_contamination", {}).get("max", 0.0)
        if worst >= 0.5:
            fatal.append(f"{worst:.0%} of the plant mask sits on holder plastic -- "
                         "P2 tracked the tool, not the plant (re-click: pose-pick-prompts)")
    elif phase == "p3":
        # README: if the circle fit fails, nothing downstream can be right.
        for name in ("cameras_lie_on_a_circle", "each_pass_lies_on_a_circle",
                     "full_rotation_covered"):
            if failed(name):
                fatal.append(f"{name}: {checks[name].get('detail', '')}")
    elif phase == "p4a":
        # Hull and masks describing different objects. Good runs sit at
        # 0.74-0.81 against the 0.75 target; the mid-run mask rewrite
        # disaster measured 0.026.
        iou = report.get("reprojection_iou")
        if isinstance(iou, dict):
            iou = iou.get("mean")
        if iou is not None and iou < 0.3:
            fatal.append(f"hull-vs-mask IoU {iou:.3f} -- the hull describes a different "
                         "object than the masks on disk (were the masks rewritten mid-run?)")
    return fatal


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <phase> <qc.json>")
    try:
        report = json.load(open(sys.argv[2]))
    except Exception:
        return  # no QC file, nothing to judge
    for line in fatal_conditions(sys.argv[1], report):
        print(line)


if __name__ == "__main__":
    main()
