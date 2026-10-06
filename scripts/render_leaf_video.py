"""Render a leaf flythrough headlessly, so the GUI stays free.

Run via `scripts/render_leaf_video.sh`, or directly:

    blender -b --python scripts/render_leaf_video.py -- \
        --maps 'D:\\PBR_Scans\\...\\maps' --out 'D:\\clips\\gaensefuss'

A few hundred frames is minutes of rendering. Doing that from the Text
Editor freezes Blender for the duration with no progress and no way to
cancel, which reads as a hang -- FFMPEG creates the output file up front, so
there is even a plausible-looking mp4 sitting there the whole time. This
runs it in a separate background Blender instead, where the frame-by-frame
progress goes to the terminal and Ctrl+C stops it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="repo root, as Blender can resolve it")
    parser.add_argument("--maps", required=True, help="a 'maps' folder, or a parent of several")
    parser.add_argument("--out", required=True, help="output path, without extension")
    parser.add_argument("--mode", default="dolly", choices=("dolly", "orbit"))
    parser.add_argument("--focus", default=None, help="substring of a leaf name, e.g. leaf_17")
    parser.add_argument("--frames-per-leaf", type=float, default=12.0)
    parser.add_argument("--height-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--resolution", default="1280x720")
    return parser.parse_args(argv)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)

    sys.path.insert(0, str(Path(args.repo) / "src"))
    from leaf_generator.blender import animation
    from leaf_generator.blender.pipeline import run

    # No point staging a viewport that does not exist, and the flythrough
    # places its own camera and light anyway.
    run(args.maps, stage_view=False)

    animation.flythrough(
        mode=args.mode,
        focus=args.focus,
        fps=args.fps,
        frames_per_leaf=args.frames_per_leaf if args.mode == "dolly" else None,
        height_scale=args.height_scale,
    )

    width, height = (int(part) for part in args.resolution.lower().split("x"))
    animation.render_video(
        args.out,
        resolution=(width, height),
        samples=args.samples,
        blocking=True,
    )


if __name__ == "__main__":
    main()
