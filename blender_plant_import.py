"""Entry point script to run inside Blender's Text Editor (or via
`blender --python blender_plant_import.py`).

Builds a "Plant" collection from one plant capture's aligned output: skeleton
curves + tip/branch Empties, a sanity-check point-cloud mesh, and (if the
KIRI 3DGS Render add-on is installed) the trained Gaussian Splat -- all
already in real-world meters, Z-up, courtesy of `pose-align-skeleton`.

Update PLANT_WORKDIR below (or set the PLANT_WORKDIR environment variable) to
point at a plant's --workdir (the same one passed to
`pose-estimate-skeleton` / `pose-train-splat` / `pose-align-skeleton`), then
run this script inside Blender. Needs `pose-align-skeleton` to have already
been run on that workdir (it reads `skeleton_blender.json`, not the raw
`skeleton.json`).

This stays a loose script at the repo root, rather than moving into
`src/pose_estimator/`, because Blender needs a real file path to open in its
Text Editor or pass to `blender --python`.
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
if not (SRC_DIR / "pose_estimator").is_dir():
    raise RuntimeError(
        f"pose_estimator package not found at {SRC_DIR}. "
        f"Update REPO_ROOT at the top of this script to the actual repo path."
    )
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Force a fresh import of pose_estimator every run -- see blender_pipeline.py
# for why (Blender caches sys.modules across Alt+P re-runs within a session).
for _module_name in list(sys.modules):
    if _module_name == "pose_estimator" or _module_name.startswith("pose_estimator."):
        del sys.modules[_module_name]

# --- Optional: attach the VS Code debugger before running -- see README's
# "Blender + VS Code" workflow section. ---
DEBUG = False
if DEBUG:
    import debugpy
    if not debugpy.is_client_connected():
        debugpy.listen(("0.0.0.0", 5678))
        print("[pose_estimator] Waiting for VS Code debugger to attach on port 5678...")
        debugpy.wait_for_client()

from pose_estimator.blender.plant_scene_import import run  # noqa: E402

# UPDATE THIS PATH (or set the PLANT_WORKDIR environment variable) to point
# at a plant's --workdir. Blender itself runs on Windows here, so this must
# be a path Windows can resolve directly (see blender_pipeline.py's notes on
# \\wsl.localhost UNC paths if a drive letter isn't visible to Blender).
PLANT_WORKDIR = os.environ.get(
    "PLANT_WORKDIR",
    r"E:\Camera_rig_data\turn_table_datasets\plant_1",
)

collection = run(PLANT_WORKDIR)
print(f"[pose_estimator] Now snap leaf assets (from blender_pipeline.py) onto the keypoint Empties under '{collection.name}'.")
