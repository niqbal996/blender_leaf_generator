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
- `src/pose_estimator/` -- the plant-measurement package (`bpy`-free).
  Turntable video in, per-leaf midribs out, as the phased P1-P6 pipeline
  described under "Plant pose pipeline" below.
  - `frames.py`, `segmentation.py`, `reconstruction.py`, `pose.py`,
    `hull.py`, `surfels.py`, `classify2d.py`, `dino.py`, `semantic.py`,
    `structure.py`, `structure_labels.py`, `leaf.py`, `ply_io.py` -- the
    library modules, one cluster per phase.
  - `cli/` -- the phase CLIs, installed as the `pose-*` console scripts.
  - `alignment.py` -- solves a real-world similarity transform from two
    points whose separation you measured. Not yet wired into P1-P6; kept
    because metric scale is the pipeline's largest outstanding gap.
- `blender_pipeline.py` -- thin entry-point script, run inside Blender, for
  the leaf-assembly pipeline above.
- `plant_pose_pipeline_PLAN.md` -- the target spec for the pose pipeline
  (visual hull + surface-aligned splatting + per-leaf midrib/curl). Read
  before extending `pose_estimator`.
- `DECISIONS.md` -- what was measured, what was chosen, and why. Read this
  before changing a threshold; most of them were fit rather than guessed.
- `HOW_IT_WORKS.md` -- the mechanism at the centre of the pipeline: how camera
  poses are recovered, what the three point clouds are, how a 3D point is
  projected into a photograph, and how a label drawn in 2D ends up attached to
  a point in 3D. Start here if the 2D-to-3D step is unclear.
- `main.py`, `get_mask.py`, `preprocess_NEF.py`, `leaf_size_in_cm.py` --
  the capture-side pipeline (RAW processing, masking, size calibration).
  Unchanged; still run standalone.
- `Uni-MS-PS/` -- git submodule for normal-map estimation. Ignored for now.

## Install

```bash
pip install -e ".[dev]"
# only if you also run the capture pipeline (get_mask.py / Uni-MS-PS):
pip install -e ".[capture]"
# only if you also run the plant pose pipeline:
pip install -e ".[skeleton,segment]"
pip install torch --index-url https://download.pytorch.org/whl/cu118  # match your CUDA version
pip install -e ".[splat]"
```

This installs both `src/leaf_generator` and `src/pose_estimator` so they're
importable, plus pytest, debugpy, and `fake-bpy-module` (IDE stubs for
`bpy`/`bmesh`/`mathutils` -- autocomplete only, not a runtime module). The
`[skeleton]` extra additionally puts the `pose-*` phase commands on your PATH
(`pose-segment`, `pose-solve`, `pose-hull`, `pose-surface`, `pose-classify`,
`pose-fuse`, `pose-semantic`, `pose-structure`, `pose-leaf`).

Without installing, run the CLIs straight from the source tree:

```bash
PYTHONPATH=src python -m pose_estimator.cli.hull --help
```

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

## Plant pose pipeline (P1-P6)

The phased pipeline described in `plant_pose_pipeline_PLAN.md`. Every phase is
a standalone CLI that reads from and writes to a specimen's run directory on
disk, so any one can be re-run or swapped without touching the others.

| phase | command | does | writes |
|---|---|---|---|
| P1+P2 | `pose-segment` | sharpest frame per angular bin, then SAM2 plant/holder masks | `p1/`, `p2/` |
| P3 | `pose-solve` | camera poses, masked COLMAP, one shared camera | `p3/` |
| P4a | `pose-hull` | visual hull by silhouette carving | `p4/` |
| P4b | `pose-surface` | 2DGS surfels, then a carved thin surface | `p4b/` |
| P4c | `pose-classify` | per-frame organ class maps (DINOv3 or SAM2) | `p4c/class_maps/` |
| P4c | `pose-fuse` | class maps voted onto the 3D points | `p4c/labels.npy` |
| P5 | `pose-structure` | stem centreline + leaf instances | `p5/` |
| P6 | `pose-leaf` | per-leaf midrib, frame, curvature, width | `p6/` |

`run_pipeline.sh` drives P1 through P4c in one call. P5 and P6 are run by hand.

```bash
conda create -n pose_estimator python=3.11 -y && conda activate pose_estimator
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,skeleton,segment]"

# SAM2 -- from source, NOT `pip install sam2` (that PyPI name is a
# third-party upload, not facebookresearch's). See DECISIONS.md.
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
pip install -e third_party/sam2
mkdir -p checkpoints && wget -P checkpoints \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

```bash
pose-segment  --video /path/DSC_0009.MOV --workdir runs/plant_9/
pose-solve    --workdir runs/plant_9/
pose-hull     --workdir runs/plant_9/ --resolution 256
pose-surface  --workdir runs/plant_9/ --iterations 5000

# P4c, as two stages. Click the seeds rather than typing coordinates:
pose-pick-seeds --workdir runs/plant_9/
pose-classify   --workdir runs/plant_9/ --seeds-file runs/plant_9/p4c/seeds.json
pose-fuse       --workdir runs/plant_9/

pose-structure --workdir runs/plant_9/
pose-leaf      --workdir runs/plant_9/

# rotate the result to check the tips are where leaves actually end
pose-view-structure --workdir runs/plant_9/
```

### Judging P5: leaves are separated by their tips

Leaf instances are split at their **tips**, not at where they touch the stem.
Two leaves fused near the apex share one contact patch, so attachment-based
splitting saw them as one organ -- on plant_9 that put 45% of all leaf tissue
into a single instance and returned 11 instances for 7 leaves. A tip survives
the fusion that an attachment does not.

How many tips there are needs no tuned radius. Two tips on different leaves
can only reach each other by travelling down one midrib to the base and back
up the other, so their geodesic separation approaches the sum of their depths
from the stem; two spurious tips on one ragged blade cut straight across it.
`--merge-cut` is that ratio, and it carries no length scale, so it does not
need refitting per specimen -- on plant_9 the leaf count holds at 7 across
0.4-0.5, where the old method swept 178 -> 73 -> 28 -> 14 -> 4 with no stable
region at all.

Every step is inspectable, because a leaf count is not a diagnosis:

```
p5/diag/instancing.png   depth field -> tips -> instances, side by side
p5/depth.ply             leaf tissue by geodesic depth; magenta = unreachable
p5/tips.ply              green = accepted tip, grey = merged away
p5/instancing.json       counts per step, and every tip's depth and position
```

`pose-view-structure` opens the same data in a rotatable window, which is the
only way to tell a tip at the end of a blade from one floating in front of a
different leaf. Keys: `1` instances, `2` depth, `t` tips, `r` rejected
candidates, `m` midribs, `s` stem/root, `[` `]` step through leaves one at a
time, `a` all, `h` help.

Each phase writes a QC report (`p2/qc.json`, `p3/poses.json`) with explicit
pass/fail acceptance checks, plus diagnostic images under `pN/diag/`. **Read
those before trusting a run** -- the checks exist to catch the failures that
look plausible, not the ones that look broken.

### Why P4c is two commands

Organ labels can be wrong because the 2D classifier was wrong, or because the
multi-view fusion was. As one stage there was no way to tell which. Splitting
it puts the per-frame class maps on disk, so each half can be checked alone:
score `p4c/class_maps/` against a few hand-labelled frames to test the
classifier, or feed the fusion synthetic maps whose answer you already know.

Both backends are kept because they fail differently. `--backend dino`
assigns each image patch to the nearest hand-clicked example, so the class
vocabulary is whatever you labelled -- but a patch is ~11 source pixels wide,
so thin petioles are lost. `--backend sam` takes SAM2's object masks and
sorts them into leaf/stem by shape: a fixed two-class vocabulary, but real
object boundaries rather than a patch grid.

`pose-semantic` runs both stages in one call, which is what `run_pipeline.sh`
uses.

### Seeds, and how to avoid clicking 50 times

DINOv3 gives every image patch a feature vector but no names, so a few
labelled examples are needed to say which vectors mean "leaf". `pose-pick-seeds`
opens the frame cropped exactly as the classifier will crop it; click to place
a seed, `1`-`9` to switch class, `n`/`p` to change frame, `s` to save. It
writes `p4c/seeds.json`, which `pose-classify --seeds-file` reads directly.

Two rules it enforces that hand-typed coordinates do not: the crop matches the
one the classifier builds, and clicks off the plant are refused (the
background is zeroed before features are extracted, so a seed there describes
a blank patch).

Seeding from several frames is worth doing -- a leaf edge-on barely resembles
the same leaf face-on. Move with `n`/`p` and keep clicking; seeds from every
frame you visit are pooled. That is safe because the vectors are stored
individually rather than averaged, so an extra example only adds coverage.

For a batch of specimens there are two ways to avoid per-plant clicking:

```bash
# no seeds at all -- SAM2 masks sorted by shape. Leaf/stem only, no root.
pose-classify --workdir runs/plant_N/ --backend sam \
    --checkpoint checkpoints/sam2.1_hiera_large.pt

# or click once, reuse the vectors everywhere
pose-classify --workdir runs/plant_1/ --seeds-file runs/plant_1/p4c/seeds.json
pose-classify --workdir runs/plant_2/ --seed-bank runs/plant_1/p4c/seed_bank.npz
```

**The seed bank's cross-specimen transfer is untested** -- only one specimen
has ever been run. Before committing a large batch to it, seed three or four
plants that differ and check whether one plant's bank labels the others
sensibly. If it drifts, pool seeds from those few plants into one bank.

Requires a display. WSLg on Windows 11 provides one; without it, fall back to
the printed coordinate grid (`scripts/dinov3_organ_lab.py --mode reference`)
and pass `--seeds "leaf:x,y" ... --seed-frame N` by hand.

### The one thing to know about this rig

The camera is locked off and the *subject* rotates. Structure-from-motion
assumes the opposite, so unmasked COLMAP latches onto the backdrop, correctly
concludes it never moved, and returns every camera at the same point. P3
therefore masks matching down to what is rigidly attached to the turntable
(disc + holder + plant), which it identifies from per-pixel temporal variance
over the sequence -- static backdrop varies by ~5 DN, the rotating disc by
30-60. No color threshold, no fiducial marker.

That also means the acceptance test for P3 is real evidence rather than a
restatement of COLMAP's own objective: in the subject's frame the camera
*must* trace a circle, and nothing in the solver enforces that.

### Shooting more than one orbit

```bash
pose-segment --video /path/pass_low.MOV /path/pass_high.MOV --workdir runs/plant_9/
```

Several videos of the **same plant at different camera elevations** go into
one workdir and are solved together, ending up in a single coordinate frame.
Use this when leaves merge where they attach: a single waist-height orbit
never looks down into an apex whorl, so leaves inserted at nearly the same
height are never separated by any two silhouettes, and that is a limit of the
capture rather than of the carving.

Each pass keeps its own tracking session, its own temporal-variance mask and
its own fitted circle -- they differ enough between elevations that sharing
them degrades the solve. Two extra checks appear in `p3/poses.json`:
`passes_share_a_rotation_axis` (the evidence the passes actually merged --
nothing in the solve enforces it) and `passes_are_at_different_elevations`
(the extra footage only buys anything if it was shot from somewhere new).

### Removed: the older single-shot prototype

An earlier `pose-estimate-skeleton` / `pose-train-splat` /
`pose-align-skeleton` path estimated a skeleton straight from a COLMAP cloud
with a kNN graph and an MST, deciding which branches were leaves via a
`min_branch_fraction` threshold. It has been deleted, along with the Blender
scene-import script that consumed its output.

The reason is worth keeping: that threshold had to be re-tuned per specimen
and never showed a stable plateau. Sweeping it gave leaf counts of 26, 8, 5
and 1 with no settled region anywhere, and at settings producing a plausible
count the smallest "leaf" held 2-9 points -- far too few to fit a midrib. The
labelled P4c path replaces the inference that needed it.

`alignment.py` survived the deletion because it is the only metric-scale
machinery in the repo. It is not currently called by any phase.
