"""Entry point script to run inside Blender's Text Editor (or via
`blender --python blender_pipeline.py`).

It loads every leaf found under MAPS_FOLDER -- each leaf's Albedo, Normal,
Height, Roughness and mask maps -- as a mesh + material + attachment-point
Empty, laid out in a row in the viewport.

Update MAPS_FOLDER below to point at a "maps" folder (or a parent directory
containing several), then run this script inside Blender.
"""

import os
import sys
from pathlib import Path

# EDIT THIS if you move the repo. Blender's Text Editor doesn't reliably
# expose __file__ when a script is run via Alt+P/Run Script (e.g. for
# unsaved or pasted text blocks), so the repo location is set explicitly
# here rather than guessed from __file__.
REPO_ROOT = Path(r"\\wsl.localhost\Ubuntu-22.04\home\niqbal\git\blender_leaf_generator")

SRC_DIR = REPO_ROOT / "src"
if not (SRC_DIR / "leaf_generator").is_dir():
    raise RuntimeError(
        f"leaf_generator package not found at {SRC_DIR}. "
        f"Update REPO_ROOT at the top of this script to the actual repo path."
    )
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Force a fresh import of leaf_generator every run. Blender keeps this
# process alive across re-runs (Alt+P / Run Script), and Python caches
# imported modules in sys.modules -- so without this, edits to the package
# source would silently keep running the *first* version ever imported in
# this Blender session, no matter how many times you re-run the script.
for _module_name in list(sys.modules):
    if _module_name == "leaf_generator" or _module_name.startswith("leaf_generator."):
        del sys.modules[_module_name]

# --- Optional: attach the VS Code debugger before running (see README's
# "Blender + VS Code" workflow section). Requires `debugpy` installed into
# Blender's bundled Python, and a "Attach to Blender" debugpy config in
# .vscode/launch.json listening on the same port. ---
DEBUG = False
if DEBUG:
    import debugpy
    if not debugpy.is_client_connected():
        debugpy.listen(("0.0.0.0", 5678))
        print("[leaf_generator] Waiting for VS Code debugger to attach on port 5678...")
        debugpy.wait_for_client()

from leaf_generator.blender.pipeline import run  # noqa: E402

# UPDATE THIS PATH (or set the LEAF_MAPS_PATH environment variable) to point
# at a "maps" folder, e.g. the example dataset. Blender itself runs on
# Windows here, so this must be a path Windows can resolve directly.
#
# If E:\... isn't visible to Blender (e.g. it's a mapped network drive not
# visible in Blender's session), go through the same WSL UNC path Blender is
# already using to read this script -- WSL's /mnt/e is just the E: drive
# viewed from inside WSL, so this reaches the same files:
MAPS_FOLDER = os.environ.get(
    "LEAF_MAPS_PATH",
    r"D:\PBR_Scans\2026-09-16-Naeem\gaensefuss_1\maps",
)

run(MAPS_FOLDER)

# --- Optional: animate the camera and key light over the leaves, to see how
# the materials behave as the lighting angle travels across them. Leave
# ANIMATE as None for a still scene.
#
#   "dolly" tracks along the row from above, close enough that each leaf
#           roughly fills the frame -- the way to actually look at a row
#           that's over a metre long.
#   "orbit" circles one leaf; pair it with FOCUS.
#
# Set FOCUS to part of an object name ("leaf_17") to animate just that leaf.
# After running, press Spacebar in the viewport to play it back. Set
# VIDEO_OUTPUT to a path to also write an mp4 (slow -- it renders every
# frame), or leave it None and use Render > Render Animation when ready. ---
ANIMATE = None        # None | "dolly" | "orbit"
FOCUS = None          # e.g. "leaf_17"
VIDEO_OUTPUT = None   # e.g. r"C:\Users\niqbal\Desktop\gaensefuss"

# Dolly pace: how many frames the camera spends covering each leaf. Higher is
# slower. At 24fps, 8 frames/leaf is a brisk pass and 24 is a leaf per second.
# This is preferred over a fixed total frame count because it holds the pace
# steady whether a session has 8 leaves or 35.
FRAMES_PER_LEAF = 12

# Dolly height, as a multiple of what framing the median leaf asks for.
# Below 1 moves the camera down (closer, bigger leaves, more raking light);
# above 1 moves it up (smaller leaves, more of the row in shot).
HEIGHT_SCALE = 1.0

if ANIMATE:
    from leaf_generator.blender import animation  # noqa: E402

    animation.flythrough(
        mode=ANIMATE,
        focus=FOCUS,
        frames_per_leaf=FRAMES_PER_LEAF,
        height_scale=HEIGHT_SCALE,
    )
    if VIDEO_OUTPUT:
        animation.render_video(VIDEO_OUTPUT)
