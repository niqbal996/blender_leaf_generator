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
    r"E:\Camera_rig_data\2026-07-20-Naeem\weed1\maps",
)

run(MAPS_FOLDER)
