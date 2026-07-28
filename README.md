# blender_leaf_generator

Turns per-leaf PBR scans (Albedo, Normal, Height, Roughness, mask) into
Blender leaf assets: one true-to-real-world-size mesh + material per leaf,
laid out in a row, each with an estimated stem-attachment keypoint.

## Repo layout

- `src/leaf_generator/` -- the leaf-assembly package.
  - `discovery.py`, `keypoints.py`, `mismatch.py`, `calibration.py`,
    `sizing.py` -- pure Python (no `bpy`), unit-tested with pytest. Handle
    finding/grouping map files per leaf, estimating the stem-attachment
    point from a mask, scoring how well a leaf's two sides agree in shape,
    reading the capture calibration log, and measuring real-world leaf size.
  - `blender/` -- `bpy`-dependent code (mesh building, materials, the
    attachment-point Empty, and the `run()` orchestration). Only importable
    from inside Blender.
  - `skeleton/` -- experimental, `bpy`-free: estimate a whole-plant
    stem/branch/leaf skeleton from a rotation video, train a Gaussian Splat
    of the same capture, and solve the real-world alignment between them.
    See "Plant skeleton from video" and "Gaussian Splat + Blender overlay"
    below.
- `blender_pipeline.py` -- thin entry-point script, run inside Blender, for
  the leaf-assembly pipeline above.
- `estimate_plant_skeleton.py` -- CLI for the video/images -> skeleton
  pipeline.
- `train_gaussian_splat.py` -- CLI: trains a Gaussian Splat from an
  `estimate_plant_skeleton.py` workdir, via gsplat.
- `align_plant_skeleton.py` -- CLI: solves + bakes the real-world alignment
  (rotation/scale/recenter) for a workdir's skeleton, point cloud, and splat.
- `blender_plant_import.py` -- thin entry-point script, run inside Blender,
  building one plant's aligned skeleton + point cloud + splat into a scene.
- `main.py`, `get_mask.py`, `preprocess_NEF.py`, `leaf_size_in_cm.py` --
  the capture-side pipeline (RAW processing, masking, size calibration).
  Unchanged; still run standalone.
- `Uni-MS-PS/` -- git submodule for normal-map estimation. Ignored for now.

## Install

```bash
pip install -e ".[dev]"
# only if you also run the capture pipeline (get_mask.py / Uni-MS-PS):
pip install -e ".[capture]"
# only if you also run the plant-skeleton / Gaussian Splat pipeline:
pip install -e ".[skeleton]"
pip install torch --index-url https://download.pytorch.org/whl/cu118  # match your CUDA version
pip install -e ".[splat]"
```

This installs `src/leaf_generator` so it's importable, plus pytest,
debugpy, and `fake-bpy-module` (IDE stubs for `bpy`/`bmesh`/`mathutils` --
autocomplete only, not a runtime module).

Run the pure-module tests any time with:

```bash
pytest
```

## Leaf map data format

Each leaf is a set of files in a flat "maps" folder, named
`<leaf_id>_<TYPE>_<side>.png`:

```
1_ALBEDO_oberseite.png
1_HEIGHT_oberseite.png
1_mask_oberseite.png
1_NORMAL_GL_oberseite.png
1_ROUGHNESS_oberseite.png
1_ALBEDO_unterseite.png
...
```

- `oberseite` (topside) is treated as the primary/geometry side. `unterseite`
  (underside) is optional -- if it's missing, the leaf just loads
  single-sided using oberseite; nothing breaks.
- **Albedo is the diffuse/base-color map** -- it's just the PBR-workflow name
  for it. There's no missing map here; Albedo plugs straight into Base Color.
- When both sides exist, the leaf becomes **one double-sided mesh**: the
  mesh geometry comes from the primary side's mask contour, and the
  material switches between the oberseite and unterseite texture sets by
  face orientation (`Geometry > Backfacing`), so it reads correctly from
  either side.
- Because only one side can define the geometry, `mismatch.mask_shape_iou()`
  checks how well the two sides' silhouettes agree (translation/scale
  invariant). If they diverge too much (default threshold: IoU < 0.5), a
  warning is printed with the leaf id so you can inspect it manually -- the
  double-sided mesh is still built either way.

## Real-world scaling

If a `<side>_log.json` (e.g. `oberseite_log.json`) sits next to the maps
folder, `calibration.py` reads its `parameters.pixelsize_mm` -- the
mm-per-pixel resolution *at the leaf plane*, already resolved through the
lens/distance calibration (not the camera's raw sensor pixel pitch, which
would be ~10x smaller). Every leaf in that session is then built at true
scale: `image_width_px * pixelsize_mm` meters wide, and so on, so leaves of
different physical sizes come out correctly sized relative to each other
(1 Blender unit = 1 meter, Blender's default). `sizing.py` additionally
measures each leaf's own tight silhouette bounding box (not the padded ROI
crop) and records it in mm in that leaf's `keypoints/leaf_<id>.json`
sidecar, alongside the calibration values used (camera model, focal
length, distance, pixel size, source log path).

**If no log file is found**, that session's leaves fall back to a
normalized placeholder scale (`fallback_scale`, default 8cm on the longer
side) -- a clearly printed warning names the session, and those leaves will
*not* be sized correctly relative to each other or to calibrated sessions.

## Running in Blender

Open `blender_pipeline.py` in Blender's Text Editor (or run
`blender --python blender_pipeline.py`), update `MAPS_FOLDER` at the bottom
(or set the `LEAF_MAPS_PATH` env var) to point at a maps folder -- or a
parent directory containing several -- and run it.

For each leaf it creates:
- A contour-shaped mesh with pixel-accurate UVs (built directly from the
  mask, so textures line up 1:1 -- no `smart_project` re-unwrap, which
  would have discarded that alignment).
- A material (single- or double-sided, see above).
- An Empty (`<leaf>_attachment`) at the estimated stem-attachment point,
  parented to the leaf so it follows any later rigging/positioning.
- A `keypoints/leaf_<id>.json` sidecar next to the maps folder, with the
  attachment point in pixel/local/world coordinates, the estimated narrow
  and broad blade widths, a confidence score (0 = ambiguous/near-symmetric
  leaf, 1 = clearly tapered), and the oberseite/unterseite mask IoU when
  applicable.

The attachment-point estimate is a shape heuristic (PCA major axis + a
width profile along it -- the narrower tapered end is picked as the
attachment point), not a detected petiole, since the scans are cropped to
the blade only. Treat low-confidence leaves (near-round blades) as
approximate.

## Blender + VS Code workflow (WSL editing, Blender on Windows)

Recommended: install the **Blender Development** VS Code extension
(`JacquesLucke.blender-development`) in this Remote-WSL window. It launches
Blender for you, gives a "Run Script" command that executes the file
you're editing directly inside the already-running Blender (no more
copy-paste), and wires up breakpoint debugging automatically. WSL can
launch a Windows `.exe` directly (e.g.
`/mnt/c/Program Files/Blender Foundation/Blender 4.x/blender.exe`), and
WSL2's localhost forwarding lets the debugger connect across the
WSL/Windows boundary -- but this cross-boundary spawn is the one part
that's worth testing first, since behavior can vary by Windows/WSL build.

Fallback if the extension doesn't cooperate: manual `debugpy` attach, which
is already wired up in this repo:

1. One-time: install `debugpy` into Blender's bundled Python (from a
   Windows terminal): `"C:\Program Files\Blender Foundation\Blender
   4.x\4.x\python\bin\python.exe" -m pip install debugpy`.
2. Set `DEBUG = True` at the top of `blender_pipeline.py` and run it in
   Blender -- it'll print "Waiting for VS Code debugger..." and block.
3. In VS Code, run the **"Attach to Blender (debugpy)"** launch config
   (`.vscode/launch.json`). Breakpoints in `leaf_generator` modules should
   bind immediately since Blender loads the script straight from this repo.

Either way, iterating on `src/leaf_generator/**` doesn't require restarting
Blender for the pure modules -- only `blender/*.py` changes need a re-run
(or Blender Development's "Reload Addons"/"Run Script") since `bpy` state
(materials, meshes already in the scene) doesn't get reset automatically.

Note: Blender doesn't bundle `opencv-python`/`Pillow` -- install both into
Blender's bundled Python the same way as `debugpy` above if you haven't
already (the original script relied on this too).

## Troubleshooting

- **Where's the console output?** `print()` from a script run via the Text
  Editor's Run Script doesn't show in the Info editor or the interactive
  Python console -- on Windows it goes to **Window > Toggle System
  Console**, a separate window. That's where `[leaf_generator] ...`
  progress/warning lines and full tracebacks show up.
- **Edited the script, but behavior didn't change?** Two separate caches to
  know about:
  - Blender's **Text Editor** loads a file into an in-memory text
    data-block on open and does *not* auto-reload it when the file changes
    on disk. Use **Text menu > Reload** (or the reload icon Blender shows
    in the header when the file changed externally) before re-running.
  - Python's **module cache** (`sys.modules`): once `leaf_generator.*` is
    imported in this Blender session, a plain re-run reuses the cached
    module even if the source changed. `blender_pipeline.py` already
    force-clears `leaf_generator.*` from `sys.modules` before importing, so
    this one is handled for you -- but it's specific to that entry script;
    keep that clearing block if you split things up further.
  - Running `blender --python blender_pipeline.py` from a terminal instead
    of the Text Editor sidesteps both, since it's a fresh process every
    time.
- **`MAPS_FOLDER` "doesn't exist" but you can see it in Explorer?** Since
  Blender runs on Windows here, the path must be one Windows can resolve
  directly -- a WSL-style `/mnt/e/...` path means nothing to it. Use the
  native drive letter (`E:\...`), or if that drive isn't visible in
  Blender's session (e.g. a network drive mapped only in your interactive
  login), go through the WSL UNC path instead:
  `\\wsl.localhost\<distro>\mnt\e\...` (find `<distro>` via `echo
  $WSL_DISTRO_NAME` in a WSL shell).

## Plant skeleton from video (experimental)

`estimate_plant_skeleton.py` estimates a whole-plant stem/branch/leaf-tip
topology from a rotation video (e.g. rotating a plant by hand, or on a
turntable), via structure-from-motion + point-cloud skeletonization. This
is a research prototype, not a validated pipeline -- treat every run's
output as something to inspect, not trust blindly.

```bash
pip install -e ".[skeleton]"   # pulls in pycolmap + matplotlib
python estimate_plant_skeleton.py --video path/to/video.MOV --workdir out/
```

Outputs land in `--workdir`: `images/` (extracted frames), `sparse/`
(COLMAP's reconstruction), `skeleton.png` (multi-angle render: point cloud
+ MST in gray, simplified graph in black, tips in blue, branch points in
red), and `skeleton.json` (registration stats + keypoint coordinates).

**Pipeline**: extract ~60 evenly-spaced frames -> COLMAP sparse
reconstruction (via `pycolmap`) -> filter the resulting 3D points by color
-> denoise (statistical outlier removal + largest-cluster filter) -> a
minimum-spanning-tree over a k-NN graph, classifying degree-1 nodes as
tips and degree>=3 nodes as branch points, with short spurious spurs
pruned -- see `src/leaf_generator/skeleton/skeletonize.py` for the exact
algorithm and its docstring's caveats.

### What we learned testing this on a real (messy) capture

Tested against a ~15s handheld video of a tiny seedling rotated between
finger and thumb (no turntable, cluttered background):

- **Camera pose estimation worked well**: 56-57/64 frames registered,
  ~1.1px mean reprojection error. So COLMAP itself handles a wobbly
  handheld rotation fine -- this isn't the bottleneck.
- **Masking the hand out *before* matching backfired badly**: registration
  collapsed to 4/64 images. The plant alone doesn't have enough texture
  for SIFT to find correspondences -- ironically, the fingers you want to
  exclude are also the main thing giving COLMAP enough texture to solve
  camera poses at all. Because of this, `--mask-mode` defaults to `none`;
  only use `vegetation`/`black_background` masking-before-matching when
  the background is genuinely low-texture (a real black backdrop), never
  to mask out a hand.
- **Filtering points by color *after* reconstruction works much better**:
  pose estimation gets the benefit of all that finger texture, and only
  the final point cloud gets filtered (`--color-filter vegetation`, an
  Excess Green Index threshold -- see `pointcloud.filter_by_vegetation_color`).
  This needed hand-tuning per video (`--color-filter-threshold`): too low
  and skin/background edge pixels leak through as noise; too high and you
  lose real plant points. 0.12 worked reasonably for this video; expect to
  sweep it for others.
- **End result**: from ~3800 raw points down to ~330 after filtering, the
  skeleton graph found 5 tips and 3 branch points in a plausible
  hub-and-spoke arrangement -- but this is a sparse, noisy result from a
  genuinely hard capture (tiny subject, shallow depth of field, tight
  crop dominated by fingers). Treat it as "the pipeline runs and produces
  something structurally sane," not "this is an accurate reconstruction."
- We were not able to source a clean rotating-plant video from the open
  web to validate the pipeline against an easy case (stock-video sites are
  JS-gated and not scrapable with the tools available) -- the strongest
  remaining validation is the synthetic point-cloud tests in
  `tests/test_skeletonize.py`, which confirm the graph algorithm itself is
  correct on clean data.

For your planned turntable + black-background capture, use
`--mask-mode black_background --color-filter none` -- masking before
matching should be safe there (the background has nothing worth matching
either way), and a real black backdrop needs no color-based cleanup.

## Gaussian Splat training + Blender overlay (experimental)

Trains a real 3D Gaussian Splat of the same turntable capture (via
[gsplat](https://github.com/nerfstudio-project/gsplat)), solves the
real-world alignment between it and the estimated skeleton, and builds both
into one Blender scene -- so individual leaf assets (from
`blender_pipeline.py`) can be manually snapped onto the skeleton for a
side-by-side morphological comparison against the real plant. Training runs
entirely in this repo (no external GUI tool), so it's scriptable across many
plant captures.

### Prerequisites

- An NVIDIA GPU with CUDA. Install a torch build matching your CUDA version
  *before* the `splat` extra, e.g. for CUDA 11.8:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  pip install -e ".[skeleton,splat]"
  ```
  `gsplat` JIT-compiles its CUDA kernels against whatever torch/CUDA it finds
  at first use, so this needs a working `nvcc` too (part of the CUDA
  toolkit, not just the driver).
- The free [KIRI Engine 3DGS Render Blender
  add-on](https://github.com/Kiri-Innovation/3dgs-render-blender-addon) (or
  any add-on exposing a scriptable splat-ply-import operator), installed
  once in Blender. `blender_plant_import.py` drives its import operator
  automatically per plant; without it, splat import is skipped with a
  printed reminder (the rest of the scene -- skeleton curves, keypoints,
  point cloud -- still builds fine).

### Pipeline

```bash
# 1. Reconstruct + estimate the skeleton (as above), writing sparse/best/,
#    skeleton.json, and pointcloud.ply -- all in COLMAP's raw, unitless,
#    arbitrarily-oriented frame.
python estimate_plant_skeleton.py --images stills/ --workdir out/plant1/ \
    --mask-mode black_background --color-filter none

# 2. Train the splat against that same reconstruction. Writes splat.ply,
#    also in the raw COLMAP frame.
python train_gaussian_splat.py --workdir out/plant1/ --iterations 30000

# 3. Open out/plant1/skeleton.png + skeleton.json, pick two keypoint indices
#    (the labeled dots) whose real-world distance you can measure (calipers,
#    ruler, pot rim -- anything on the plant itself). Solve + bake the
#    alignment -- cheap: no COLMAP rerun, no retraining.
python align_plant_skeleton.py --workdir out/plant1/ \
    --scale-ref-a 3 --scale-ref-b 9 --scale-ref-distance-m 0.084

# 4. Build the Blender scene (skeleton curves/Empties + point cloud +
#    splat, all pre-aligned to real-world meters, Z-up):
PLANT_WORKDIR="<path>/out/plant1" blender --python blender_plant_import.py

# 5. Load leaf assets and manually snap them onto the skeleton (see below):
LEAF_MAPS_PATH="<path>/weed1/maps" blender --python blender_pipeline.py
```

For many plants, steps 1-3 are just a shell loop -- the whole point of
training in-repo instead of through a GUI tool:

```bash
for d in out/*/; do
    python estimate_plant_skeleton.py --images "$d/stills" --workdir "$d" --mask-mode black_background --color-filter none
    python train_gaussian_splat.py --workdir "$d"
done
```

### How alignment works

COLMAP's reconstruction (and therefore the skeleton, point cloud, and
trained splat, all reconstructed in that same frame) has an arbitrary,
unitless scale and orientation -- nothing about "up" or "how big" is known
without extra information. `align_plant_skeleton.py` solves this in two
parts:

- **Rotation**: automatic. A turntable capture's registered camera centers
  lie approximately on a circle in a plane, so PCA over those centers finds
  the plane's normal -- the rotation axis -- with no manual picking needed.
- **Scale**: not recoverable from SfM alone (there's no metric reference in
  the shot). Give it two `skeleton.json` keypoint indices (visible as
  labeled dots in `skeleton.png`) plus a real-world distance you measured
  between those same two plant features.

The result is baked directly into `skeleton_blender.json`,
`pointcloud_blender.ply`, and `splat_blender.ply` (all real-world meters,
Z-up) -- `alignment.json` records the raw transform too, for provenance and
for cheaply re-baking after refining the scale reference. Because the splat
is trained by this same repo (not an external tool with its own coordinate
conventions), its `.ply` never gets silently re-oriented/re-scaled by
something else -- the exact same alignment applies to it as to the skeleton.

One ply convention worth knowing if you ever touch this code: the standard
3DGS ply schema stores `scale_0..2` as **log-scale** and `opacity` as
**logit(alpha)**, not the raw values -- `Alignment.apply_to_gaussians` (in
`src/leaf_generator/skeleton/alignment.py`) shifts log-scales additively by
`log(scale)` rather than multiplying, and composes orientation quaternions
rather than touching them independently. Getting this wrong produces
wrong-sized, wrong-oriented Gaussians with no error raised.

### Manual leaf-snapping workflow

No new code here -- standard Blender operations, using the skeleton's
keypoint Empties (from `blender_plant_import.py`) and each leaf's existing
`<mesh_name>_attachment` Empty (from `blender_pipeline.py`) as guides. Leaf
attachment Empties are *children* of their leaf mesh, so moving one alone
doesn't move the leaf:

1. One-time per leaf: select its attachment Empty -> `Shift+S` -> *Cursor to
   Selected*; select the leaf mesh alone -> *Object > Set Origin > Origin to
   3D Cursor* (the mesh's own origin now coincides with its attachment
   point).
2. To place a leaf: select the target skeleton keypoint Empty -> *Cursor to
   Selected* -> select the leaf mesh -> *Selection to Cursor*, then
   eyeball-rotate (`R`) against the local branch curve's direction.

No orientation/placement algorithm yet -- this is a first pass to get real
comparisons in front of you before investing in automating placement.
