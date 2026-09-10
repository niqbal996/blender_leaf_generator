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
    `geometry.py`, `hull.py`, `surfels.py`, `classify2d.py`, `dino.py`, `semantic.py`,
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
| P3x | `pose-geometry` | VGGT / MapAnything on the same masked frames; aligned comparison to COLMAP | `p3/experiments/` |
| P4a | `pose-hull` | visual hull by silhouette carving | `p4/` |
| P4b | `pose-surface` | 2DGS surfels, then a carved thin surface | `p4b/` |
| P4c | `pose-classify` | per-frame organ class maps (DINOv3 or SAM2) | `p4c/class_maps/` |
| P4c | `pose-fuse` | class maps voted onto the 3D points | `p4c/labels.npy` |
| P5 | `pose-structure` | stem centreline + leaf instances | `p5/` |
| P6 | `pose-leaf` | per-leaf midrib, frame, curvature, width | `p6/` |

`run_pipeline.sh` drives the phases; `--stop-after` and `--skip-to` are how
you stop for the two things that need a human, which are picking the P2
plant/holder prompts and picking the P4c organ seeds.

### Learned P3 geometry experiment (VGGT and MapAnything)

P3's masked COLMAP solve remains the baseline.  `pose-geometry` is a separate
experiment, not a replacement: it prepares the exact P2 plant-masked RGB
frames for each learned model, invokes that project's official COLMAP export,
and leaves every result in its own directory.  This keeps a failed learned
pose from quietly contaminating P4--P6.

No checkpoint needs to be downloaded manually.  On first use the adapter
fetches the public exporter code into the Linux-side cache
`~/.cache/blender_leaf_generator/model_code/`, while each official
model's `from_pretrained` call downloads its weights from Hugging Face into
the normal Hugging Face cache.  Accept the gated VGGT model's terms first.
Use an environment variable for the token so it never lands in shell history
or command-line process arguments, then run:

```bash
export HF_TOKEN='hf_...'
pose-solve --workdir runs/plant_9
pose-geometry --workdir runs/plant_9 --backends vggt mapanything \
  --max-images 80 --bundle-adjust \
  --model-python /opt/conda/envs/geometry/bin/python
```

`--max-images` uniformly samples around the capture orbit; use it for the
first VRAM-limited comparison, then re-run the promising backend with `0`
(all frames).  `--bundle-adjust` is forwarded to VGGT's official exporter;
MapAnything ignores it.  Start with `--dry-run` to stage the masked images and
record the exact command without loading a model.  `--model-python` is useful
when the models live in a separate Torch/CUDA environment; otherwise the
adapter uses the Python that launched `pose-geometry`.
`--hf-token` is available when an environment variable is impractical, but
the environment variable is safer. `--vggt-root` and `--mapanything-root`
remain optional overrides for an existing code checkout; they are no longer
required. The exporter cache deliberately does not live in the work directory:
on WSL a dataset under `/mnt/c`, `/mnt/d`, or `/mnt/e` cannot host Git's POSIX
metadata. Use `--code-cache /some/linux/path` to choose another Linux-native
location.

For VGGT's official COLMAP exporter, install its exporter-only dependencies
once into the same environment that launches `pose-geometry`:

```bash
pip install -e ".[vggt]"
```

Two things that extra cannot install for you, both checked before a run
stages anything:

* **Python 3.10 or newer, for the model environment only.** VGGT's exporter
  and its vendored files annotate with `np.ndarray | None` in positions
  Python evaluates on import, so 3.9 fails with `TypeError: unsupported
  operand type(s) for |` from inside `vggt/dependency/projection.py`. This
  project itself does not need 3.10: keep it where it is and pass
  `--model-python /path/to/py310/bin/python`.
* **`pycolmap`**, from whichever build you already use for `pose-solve` --
  plain `pycolmap` or `pycolmap-cuda`. The extra names neither on purpose:
  both install the same `pycolmap` module, so pinning one would clobber the
  other and break P3.

`pose-geometry` asks the exporter's own interpreter what it has before it
clears or stages anything, so a missing dependency costs seconds instead of
surfacing as a traceback from a vendored file after a weight download:

```text
RuntimeError: the vggt exporter cannot run in this environment:
    - /env/bin/python is Python 3.9.18, but the official vggt exporter needs
      3.10 or newer: it annotates with `X | None` in positions Python
      evaluates at import time ...
      Create a newer environment and point --model-python at its python ...
    - pycolmap cannot be imported in /env/bin/python: ModuleNotFoundError ...
      Fix: pip install pycolmap  (or pycolmap-cuda for GPU SIFT). ...
  Nothing was staged or downloaded. Re-run with --skip-env-check to try anyway.
```

`--dry-run` runs exactly this check and stages the inputs without loading a
model, which makes it the quickest way to validate a new machine.  A
`pycolmap-cuda` wheel that imports but cannot find `libcudart` is reported as
its own case, since the fix there is `LD_LIBRARY_PATH`, not an install.

This includes `trimesh`, LightGlue, the tracker configuration libraries, and
the Hugging Face loader used by the upstream exporter. Install your
CUDA-matched `torch`/`torchvision` build before this command if the existing
environment does not already have one; the extra intentionally does not force
a particular CUDA build. It follows the imports and demo requirements of the
[official VGGT exporter](https://github.com/facebookresearch/vggt).

#### Reading the run: memory, stages, and stalls

VGGT's aggregator attends over the tokens of *every* staged frame at once, so
its VRAM grows with frame count rather than staying per-image constant, and
the whole prediction happens inside one CUDA call that prints nothing.  A run
that works at 8 frames can therefore die silently at 27.  `pose-geometry`
makes that stage legible instead of leaving a blank terminal:

```text
  vggt: 27 masked frames; live log .../p3/experiments/vggt/stdout.log
  host RAM: 12.5 GB available of 15.5 GB
  GPU 0 (NVIDIA GeForce RTX 2070 with Max-Q Design): 0.3 GB free of 8.0 GB
  WARNING: 1 other compute process(es) already hold this GPU (pid 63488); ...
  WARNING: VGGT needs roughly 17.5 GB for 27 frames but 0.3 GB is free. ...
  WARNING: consider --max-images 8 (an estimate from this card's 8 GB, ...)
    [vggt 00:12] Loaded 27 images from .../runner/images
    [vggt 02:42] still running, no output for 150s | aggregator + camera/depth
      heads over all frames at once -- the peak-VRAM stage | GPU 7.6/8.0 GB
      used | host 4.1 GB free | RSS 6.2 GB (peak 6.4)
```

Every exporter line is prefixed with elapsed time, and during silence the
last stage reached is reported with live GPU, host, and exporter memory every
`--heartbeat-seconds` (default 30; `0` keeps quiet but still records peaks).
The pre-run figures come from `nvidia-smi` and `/proc/meminfo`; the VRAM
estimate is a coarse empirical fit used only to warn and to propose a frame
count, never to block a run.  Under WSL `nvidia-smi` cannot attribute VRAM
per process, so the GPU figure is device-wide and other compute processes are
listed by pid -- a nearly full card with nothing of yours running usually
means a previous attempt is still holding it.

Output is written to `stdout.log` as it arrives, so an interrupt or a kill
still leaves a complete log, and `resources.json` records the pre-run
snapshot beside the observed peaks.  Ctrl-C stops the exporter rather than
orphaning it on the GPU.  A failure is diagnosed rather than reported as a
bare exit code:

```text
RuntimeError: vggt exporter exited -9 after 12:03 while: aggregator + ...
  Killed by SIGKILL with no traceback, which is normally the host
  out-of-memory killer (common under WSL, whose RAM is capped) ...
  Peak GPU memory in use on the device, all processes together: 7.9 of 8.0 GB.
  Peak exporter host memory: 11.2 GB (host free fell to 0.3 GB).
  This is an out-of-memory failure. 27 frames were staged; retry with
  --max-images 8.
```

A host out-of-memory kill matters under WSL specifically: its VM gets a
fraction of system RAM by default, which a `.wslconfig` `memory=` setting can
raise.

The outputs are deliberately comparable:

```text
p3/experiments/vggt/
  input/images/             P2 plant-masked RGB, original frame names retained
  sparse/best/              official prediction exported as a COLMAP model
  poses.json                registration/orbit report in P3's usual format
  sparse_points.ply         inspectable cloud
p3/experiments/mapanything/  (same layout)
p3/experiments/compare.json  registration and aligned camera-centre agreement
p3/experiments/diag/camera_compare.png
```

`compare.json` aligns matching camera centres by a best-fit similarity before
reporting errors, because learned models and COLMAP use unrelated world axes
and scales.  A low error means the candidate agrees with the baseline; it is
not ground-truth accuracy.  Read that alongside registration fraction, cloud
coverage in `sparse_points.ply`, and the rendered P4 hull.  The official
projects currently expose COLMAP export in their repositories:
[VGGT](https://github.com/facebookresearch/vggt) and
[MapAnything](https://github.com/facebookresearch/map-anything).

To compare silhouette-constrained geometry without overwriting the baseline
P4 artifacts, carve an experimental hull separately:

```bash
pose-hull --workdir runs/plant_9 --geometry-backend vggt
# writes p4/experiments/vggt/{hull.ply,hull.json,diag/}
```

Only promote an experimental reconstruction into the normal `p3/` / `p4/`
path after inspecting it; the semantic-fusion and skeleton stages require one
specific cloud and point order, so mixing artifacts from two backends is not
valid.

### Midribs: fitted from the points, or the straight chord

A leaf's midrib is fitted from that leaf's own points, except where they
cannot describe it. The exception is the small upright leaves at the centre of
a rosette: the cloud closes over the middle of the plant slightly higher than
they attach, so half their length is missing and what survives is a one-sided
sliver. A curve fitted to that waves off the vein, and the straight
crown-to-tip chord is the better midrib.

A leaf takes the chord only when it is **both** steep (above 45 degrees) and
**poorly covered** -- fewer than 85% of the stations along its own chord hold
its own tissue. Steepness alone is not the test: it describes how a leaf is
posed, not how well it was reconstructed, and on sugarbeet_4 those come apart
completely. Its upright blades carry 8k-62k points and cover their whole
chord, and selecting on angle drew every one as a straight stick -- discarding
a real 1.20 arc-over-chord on the largest.

`p5.json` reports `midrib_support` per leaf: elevation, coverage, and which
construction it got. `--strict-midribs` refuses the chord entirely and fits
every midrib from its own points. Use it when the leaves are genuinely upright
and well reconstructed; it is not the default because a sparse leaf fitted
from its own points can wander badly -- thistle3's 470-point leaf produces a
midrib 3.6x longer than the distance it spans.

### Naming a dataset, and pipeline.conf

The shortest correct invocation names the dataset directory and nothing else:

```bash
./run_pipeline.sh /data/2026-09-01/sugarbeet_4
```

Its `pass*/` subdirectories become the capture passes in natural order
(`pass2` before `pass10`), and the workdir is `<dataset>/plant`. A flat
directory of JPEGs is a single pass, which is how `thistle3` is laid out.
`--photos`, `--video` and `--workdir` still override and are still the way to
drive a layout that is not this one.

This is worth doing for correctness, not just brevity. In the long form every
path repeats the same dataset prefix, so one stale component is easy to type
and invisible on review -- which is how sugarbeet_4 came to be reconstructed
from two different plants (see DECISIONS.md, 2026-09-02).

Settings that hold across runs go in a `pipeline.conf` of `key = value` lines
-- keys are the long options with dashes as underscores, and an unknown key is
an error rather than a silent no-op:

```
prompt_bank  = /data/2026-09-01/sugarbeet_3/plant/p2/prompt_bank.npz
seed_bank    = /data/thistle3/plant/p4c/seed_bank.npz
architecture = rosette
low_texture  = 1
```

Read from `~/.config/blender_leaf_generator/pipeline.conf`, `./pipeline.conf`,
`<dataset>/../pipeline.conf` and `<dataset>/pipeline.conf` in that order, each
overriding the last, and any command-line flag overriding all of them.
`--config <file>` uses just that file. `pipeline.conf.example` in the repo
root lists every key.

Keep the HF token out of the command line -- put it in the user-level config
or in `$HF_TOKEN`. A command-line argument is visible in `ps` to every user on
the machine and is written to your shell history.

`--dry-run` prints every resolved path and stops. Run it whenever a path
changed: the paths a config file or the dataset directory supplied are the
ones you did not type on this run, so they are the ones you will not notice
are wrong. It also flags a bank that lives outside the current dataset, which
is legitimate -- that is what banks are for -- and is also what a stale path
looks like.

### Install the pose pipeline

The env this needs is heavier than the leaf-generator one above:

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

DINOv3 is gated on HuggingFace: accept the licence at
[facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
then `export HF_TOKEN=hf_xxx`. Without a token use
`--dino-model facebook/dinov2-base`, which is ungated and behaves similarly.

### GPU use, and why P3 is the exception

Most of the pipeline is on the GPU already and needs no flag: P4b trains
surfels with gsplat, and P4c runs DINOv3 or SAM2 with `--device cuda`
(default). P2's SAM2 propagation is on the GPU too.

**P3 is the exception.** COLMAP's SIFT extraction runs on the CPU unless
pycolmap was *built* against CUDA, and the wheels published on PyPI are not.
There is one authoritative test, and it is not `nvidia-smi` or whether torch
sees your GPU:

```bash
python -c "import pycolmap; print(pycolmap.__version__, pycolmap.has_cuda)"
```

- `True` -- you have a CUDA build; pass `--use-gpu` to move SIFT extraction
  onto it.
- `False` -- `--use-gpu` will not work. Getting it means building COLMAP and
  pycolmap from source with `-DCUDA_ENABLED=ON`; there is no wheel to install.

```bash
./run_pipeline.sh --workdir runs/thistle1 --skip-to p3 --use-gpu   # via the driver
pose-solve --workdir runs/thistle1 --use-gpu                       # or the phase alone
```

Asking for `--use-gpu` without CUDA support used to fail deep inside COLMAP's
own option validation with `Check failed: extraction_options.Check()`, naming
neither the option nor the cause. It is now checked up front and says so.

**It is a smaller win than it sounds.** Only extraction moves; matching and
mapping stay on the CPU whatever you do. Measured on this repo's frames
(1600px, 8 threads, unmasked, on the dev machine's CPU), extraction is linear
in frames and exhaustive matching is linear in *pairs*:

| frames | extract | match | pairs |
|---|---|---|---|
| 8 | 4.0 s | 0.7 s | 28 |
| 16 | 7.6 s | 2.1 s | 120 |

That is ~0.48 s/frame extracting and ~0.018 s/pair matching, so a two-pass
192-frame capture (18,336 pairs) works out at roughly 1.5 minutes extracting
against 5 minutes matching. GPU SIFT takes a bite out of the smaller half.
Masking cuts the real numbers below these, since fewer features survive.

If you want P3 faster without building anything, `--max-image-size` is the
lever that moves both halves at once:

```bash
pose-solve --workdir runs/thistle1 --max-image-size 1200
```

> **Note on the `skeleton-gpu` extra.** `pyproject.toml` declares a
> `skeleton-gpu` extra listing `pycolmap-cuda`, and `pose-all` pulls it in.
> Installing it has broken this environment before -- both packages provide
> the same `pycolmap` module, so whichever unpacks last wins and `import
> pycolmap` can end up raising or missing attributes. `setup_env.sh` detects
> that collision and keeps whichever build imports. If you land in it, the
> way out is to force the CPU wheel back:
>
> ```bash
> pip install --force-reinstall --index-url https://pypi.org/simple pycolmap==4.1.1
> python -c "import pycolmap; print(pycolmap.__version__, pycolmap.has_cuda)"
> ```
>
> The documented install line above (`.[dev,skeleton,segment]`) does not
> include this extra, and P3 works fine on the CPU build.

### Running one plant, start to finish

Worked example: two orbits of a thistle, one at waist height and one from
slightly above, into `runs/thistle1/`. Run the steps in order. There are
**two** points where you stop and click something, and **two** where you stop
and read a QC number; neither is optional, and skipping either wastes the
half hour that follows it.

| step | command | wall time | you do what |
|---|---|---|---|
| 1 | `run_pipeline.sh --stop-after p1p2` | ~10 min | nothing |
| 2 | read `p2/qc.json` | seconds | **decide if P2 tracked the plant** |
| 3 | `pose-pick-prompts` + `pose-segment --reuse-frames` | ~8 min | click the plant, once per pass |
| 4 | `run_pipeline.sh --skip-to p3 --stop-after p4b` | ~30 min | nothing |
| 5 | read `p3/poses.json` | seconds | **check the circle fit** |
| 6 | `pose-pick-seeds` | ~2 min | click leaf / stem / root |
| 7 | `run_pipeline.sh --skip-to p4c --stop-after p4c` | ~5 min | nothing |
| 8 | `run_pipeline.sh --skip-to p5` | ~2 min | nothing |
| 9 | `pose-view-structure` / Blender | — | look at it |

---

**Step 1 -- frames and plant masks (P1+P2).** Several `--video` files are
capture passes of the *same plant* at different camera elevations; they share
one workdir and are solved together in P3. See
[Shooting more than one orbit](#shooting-more-than-one-orbit).

```bash
./run_pipeline.sh --video /data/DSC_0015.MOV /data/DSC_0016.MOV \
    --workdir runs/thistle1 --stop-after p1p2
```

**Step 2 -- check that P2 tracked the plant and not the pliers. Do not skip
this.** SAM2 has to be told which object to follow, and by default that seed
point is chosen by colour, which fails often enough on this rig that it is a
routine step rather than an exception. Nothing errors when it goes wrong: you
get 96 clean masks of a pair of pliers and find out in P4a, half an hour
later, when the hull carves to nothing.

```bash
python -c "import json; d=json.load(open('runs/thistle1/p2/qc.json')); \
print(json.dumps(d['checks'], indent=2)); print('all passed:', d['all_passed'])"
eog runs/thistle1/p2/diag/          # the green overlay must be on the plant
```

Two checks decide it:

- `plant_mask_free_of_holder` -- what fraction of the "plant" mask is sitting
  on holder plastic. Limit 2%. A failure in the 80-90% range means the mask
  *is* the pliers.
- `plant_holder_disjoint` -- the two tracked objects should not overlap.

With more than one pass, check them **per pass**, because one good pass hides
one bad one in the aggregate:

```bash
python - <<'EOF'
import cv2, json, numpy as np
src = json.load(open("runs/thistle1/p1/sources.json"))
rows = {}
for stem, p in sorted(src.items()):
    pm = cv2.imread(f"runs/thistle1/p2/masks/plant/{stem}.png", 0) > 127
    hm = cv2.imread(f"runs/thistle1/p2/masks/holder/{stem}.png", 0) > 127
    rows.setdefault(p, []).append(((pm & hm).sum() / max(pm.sum(), 1), pm.sum()))
for p, r in rows.items():
    ov = np.array([x[0] for x in r]); ar = np.array([x[1] for x in r])
    print(f"pass {p}: {len(r):3d} frames  median plant area {np.median(ar):8.0f} px  "
          f"mean overlap with holder {ov.mean()*100:5.1f}%")
EOF
```

Healthy looks like `mean overlap with holder 0.0%`. A pass reading 70-90% is
tracking the tool. **If every check passes, skip to step 4.**

**Step 3 -- only if step 2 failed: click the plant, once per capture pass.**

```bash
pose-pick-prompts --workdir runs/thistle1 --hf-token hf_xxxxxxxxxxxxxx
```
The hf-token must be added otherwise, the seed bank will not be generated since the DINOv3 is a gated model. 
The window opens on the frame each pass is actually seeded on -- the *first*
frame of that pass, because SAM2 propagates forward from there and a point
clicked anywhere else has nothing to attach to. Faint grey crosses show where
the colour rule put its seeds; when those are on the pliers, that is the bug
you are looking at.

- click **three or more** points on different parts of the plant (one click
  is not enough for the reusable bank -- see the numbers
  [below](#when-the-pliers-get-segmented-as-the-plant))
- **press `3` and click the exposed root, once per pass.** The root is part
  of the plant, but it cannot ride along as an extra `1` = plant click: the
  jaws cut it into a disconnected blob, and SAM2 keeps one temporal memory
  per tracked object, dominated by the object's big connected mass -- root
  points clicked as plant held the root in only 51-77% of thistle1's frames,
  under the ~86% silhouette agreement P4a needs, so the carve deleted it
  anyway. `3` = root seeds the blob as its *own* SAM2 object with its own
  memory, and P2 unions that mask back into the plant mask on write --
  downstream still sees exactly two classes. Miss the click and the root is
  absent from the mask, the hull, the cloud and every phase after -- P4c
  cannot put it back, because P4c only labels points that already exist.
  If SAM2 still drops the root mid-orbit -- the pliers cross in front once
  per rotation, and re-acquiring a small object after occlusion is luck --
  `pose-segment` re-seeds it on its own: stretches of empty root mask are
  scanned with the prompt bank's root examples and the best match becomes a
  new SAM2 conditioning point, logged as `root_reseeds` in
  `p2/prompts.json` so it can be audited against `p2/diag/`
- press `2`, click the plier **jaws** where they grip the stem, not the far
  end of the handle: the tracking crop is sized to the plant, and a point
  outside it is dropped
- press `n` for the next pass and repeat -- **every pass needs its own
  clicks**, including the ones that were already fine
- press `s` to save

Then redo P2 alone, keeping the frames you already extracted:

```bash
pose-segment --workdir runs/thistle1 --reuse-frames
```

Re-check step 2 before going on. That run also writes
`runs/thistle1/p2/prompt_bank.npz`, which makes every later specimen on this
rig seed itself -- see
[When the pliers get segmented as the plant](#when-the-pliers-get-segmented-as-the-plant).

> Do **not** re-run with `--video` to fix this. That extracts a second set of
> frames on top of the first, leaving `p1/frames` disagreeing with
> `p1/sources.json`, and P3 then pools two elevations into one orbit and P4a
> carves an empty hull.

**Step 4 -- camera poses, hull and surface (P3, P4a, P4b).** This is the long
one. P4b is on the GPU already; P3's SIFT is not, unless you have a CUDA
pycolmap -- see [GPU use](#gpu-use-and-why-p3-is-the-exception) before adding
`--use-gpu`.

```bash
./run_pipeline.sh --workdir runs/thistle1 --skip-to p3 --stop-after p4b
```

**Step 5 -- check the circle fit.** The camera is fixed and the plant turns,
so in the plant's frame the camera must trace a circle. Nothing in the solver
enforces that, which is what makes it real evidence rather than a restatement
of COLMAP's own objective.

```bash
python -c "import json; print(json.dumps(json.load(open('runs/thistle1/p3/poses.json'))['checks'], indent=2))"
```

For a single pass the checks are `cameras_lie_on_a_circle`,
`cameras_coplanar` and `full_rotation_covered`. For a multi-pass capture they
become `each_pass_lies_on_a_circle`, `full_rotation_covered`,
`passes_share_a_rotation_axis` (the evidence the passes actually merged into
one coordinate frame -- nothing in the solve enforces it) and
`passes_are_at_different_elevations` (the extra footage only buys anything if
it was shot from somewhere new).

If the circle fit fails, nothing downstream can be right, and the usual cause
is step 2: two passes whose silhouettes describe different objects cannot
agree on an axis.

**Step 6 -- click the organ seeds for P4c.** A different set of clicks from
step 3: those said *which object is the plant*, these say *which parts of the
plant are leaf, stem and root*. This is where `root` becomes a semantic
class -- step 3's `3` = root was a tracking aid that folded straight back
into the plant mask; here the label survives into the coloured cloud. The
window opens on the plant cropped exactly as the classifier crops it, and
clicks off the plant are refused.

```bash
pose-pick-seeds --workdir runs/thistle1
```

Click three or four leaf points **on frames at different angles** (`n` / `p`
to move; a leaf edge-on barely resembles the same leaf face-on), press `2`
and click the stem twice, `3` and click the root, then `s`. Writes
`p4c/seeds.json`.

**Step 7 -- organ labels and coloured clouds (P4c).** The seeds are found at
`p4c/seeds.json` without being named.

```bash
./run_pipeline.sh --workdir runs/thistle1 --skip-to p4c --stop-after p4c
```

Check `p4c/diag/parts_*.jpg` (photograph beside classification) before
blaming the 3D labels for anything -- that image tells you whether a bad
label came from the 2D classifier or from the multi-view voting.

**Step 8 -- structure and per-leaf measurements (P5, P6).** Tell it what kind
of plant this is:

```bash
# upright, with a central stem (the default)
./run_pipeline.sh --workdir runs/thistle1 --skip-to p5

# leaves radiating from a crown at ground level: thistle, sugar beet
./run_pipeline.sh --workdir runs/thistle1 --skip-to p5 --architecture rosette
```


**Step 9 -- look at it.**

```bash
pose-view-structure --workdir runs/thistle1     # matplotlib, rotatable
./scripts/view_in_blender.sh runs/thistle1      # Blender (Windows Blender, from WSL)
```

Measurements land in `runs/thistle1/p6/leaves.json`: arclength, insertion
angle, azimuth and a width profile per leaf, in COLMAP units rather than
millimetres -- no scale reference is solved yet.

### Recovering after a bad P2

If you got as far as P3 or P4 before noticing the masks were wrong, nothing
needs deleting -- every phase overwrites its own output. Fix P2, then re-run
from P3:

```bash
pose-pick-prompts --workdir runs/thistle1        # step 3 above
pose-segment --workdir runs/thistle1 --reuse-frames
./run_pipeline.sh --workdir runs/thistle1 --skip-to p3 --stop-after p4b
```

The one thing that *does* need cleaning is a duplicated frame set from
re-running with `--video` on an existing workdir. Check it before anything
else:

```bash
ls runs/thistle1/p1/frames | wc -l
python -c "import json,collections; \
print(collections.Counter(json.load(open('runs/thistle1/p1/sources.json')).values()))"
```

The count and the per-pass totals must agree (192 frames, `{0: 96, 1: 96}`).
If they do not, delete `runs/thistle1/p1/` and start from step 1.

### Running the rest of the batch

Once one plant is done, its **two** banks carry to the others and there is
nothing left to click. Pass both, and you are never relying on a search
finding the right file:

```bash
./run_pipeline.sh --video /data/DSC_0010.MOV --workdir runs/thistle2 \
    --prompt-bank runs/thistle1/p2/prompt_bank.npz \
    --seed-bank   runs/thistle1/p4c/seed_bank.npz
```

They are different files answering different questions, and **passing only
one of them is the most common way to lose an afternoon**:

| flag | file | answers |
|---|---|---|
| `--prompt-bank` | `p2/prompt_bank.npz` | which object in the scene is the plant (P2) |
| `--seed-bank` | `p4c/seed_bank.npz` | which parts of it are leaf/stem/root (P4c) |

`--seed-bank` alone leaves P2 with nothing, so it falls back to the colour
rule and can track the pliers for a whole pass. That failure is loud -- the
run prints `WARNING: no plant/holder prompts -- falling back to the COLOUR
RULE` before it starts, and stops at the QC gate afterwards with something
like *"89% of the plant mask sits on holder plastic"* -- but only if you are
reading the first ten lines of the log.

**The prompt bank is found automatically only between sibling workdirs.**
`run_pipeline.sh` searches the newest `*/p2/prompt_bank.npz` one level above
the workdir, so `runs/thistle2` finds `runs/thistle1`'s. A bank kept
somewhere else entirely -- another drive, another project folder -- is
**not** found, and nothing says so beyond the warning above. `--prompt-root
<dir>` points the search elsewhere; `--no-prompt-bank` turns it off.

A bank also has to have been built with the same DINO backbone it is loaded
under, since the vectors mean nothing across models; a mismatch is refused
rather than silently used. And note the P2 bank carries **root** examples
when they were clicked -- that is what gives a new specimen's root its own
tracked object, so a bank clicked without roots will lose the root on every
plant it seeds.

Individual phases can also be run directly, which is what to do when
debugging one of them:

```bash
pose-segment   --video /data/DSC_0009.MOV --workdir runs/thistle1/
pose-solve     --workdir runs/thistle1/
pose-hull      --workdir runs/thistle1/ --resolution 256
pose-surface   --workdir runs/thistle1/ --iterations 5000
pose-classify  --workdir runs/thistle1/ --seeds-file runs/thistle1/p4c/seeds.json
pose-fuse      --workdir runs/thistle1/
pose-structure --workdir runs/thistle1/
pose-leaf      --workdir runs/thistle1/
```

The driver stops at a phase whose QC shows a **catastrophic** failure -- P2
tracking the tool instead of the plant, a failed P3 circle fit or rotation
coverage, a hull that does not match the masks on disk. Nothing after such a
failure can be right, and letting the run finish is how a plier hull once
became an empty Blender scene with no error anywhere. Advisory QC failures
(and good runs do have them) never stop a run; `--keep-going` pushes past a
stop when a partial result is wanted knowingly. One related rule of thumb:
**do not re-run a phase while a pipeline is still executing on the same
workdir** -- phases read each other's artifacts from disk, and a concurrent
fix produces a mix no QC number describes (measured: a P3 solved from masks
being rewritten mid-run registered half the frames and fit a 42%-deviant
circle on a perfectly good video).

Other flags on `run_pipeline.sh`: `--skip-p4b` halves the runtime by labelling
the P4a hull instead of a trained surface (blobbier, and P5 cannot
skeletonise it afterwards without going back); `--backend sam
--sam-checkpoint <ckpt>` skips seeds entirely at the cost of a fixed
leaf/stem vocabulary; `--video a.MOV b.MOV` merges two capture passes of the
same plant, covered under
[Shooting more than one orbit](#shooting-more-than-one-orbit).

### When the pliers get segmented as the plant

On some plants P2 tracks the holder instead of the specimen, for the whole
sequence. SAM2 has to be told which object to follow, and the default rule
picks that seed point by colour: the pliers' amber grip is green-dominant in
RGB, so a pass that shows the tool large and the plant small and shadowed
seeds on the tool. Nothing errors. P4a then carves an empty or nonsensical
hull, because the two passes' silhouettes describe different objects.

Catch it at step 1: `p2/qc.json` reports `plant_mask_free_of_holder`, and the
overlays in `p2/diag/` show what the green mask is actually on.

Fix it by clicking the plant once, per capture pass:

```bash
export HF_TOKEN=hf_xxx                      # REQUIRED, see below
pose-pick-prompts --workdir runs/thistle1   # click plant, 2 = plier jaws, 3 = exposed root
pose-segment --workdir runs/thistle1 --reuse-frames    # redo P2 only, keeping the frames
./run_pipeline.sh --workdir runs/thistle1 --skip-to p3
```

> **The token is not optional here.** The reusable bank is DINO feature
> vectors, so building it loads the gated DINOv3 weights. Without `--hf-token`
> (or `HF_TOKEN`) the clicker still writes `p2/prompts_clicked.json` and fixes
> *this* video, prints a one-line note, and writes **no** `p2/prompt_bank.npz`
> -- so the next specimen silently falls back to the colour rule. Confirm you
> got both files before moving on:
>
> ```bash
> ls runs/thistle1/p2/prompts_clicked.json runs/thistle1/p2/prompt_bank.npz
> ```
>
> The success line reads `reusable prompt bank -> .../p2/prompt_bank.npz`.
> Use `--dino-model facebook/dinov2-base` if you have no token; just build
> every later bank with the same model, since a bank refuses to load under a
> different one.

> **`--seed-bank` is not the P2 bank.** There are two, and passing the wrong
> one is silently a no-op:
>
> | flag | file | feeds |
> |---|---|---|
> | `--prompt-bank` | `p2/prompt_bank.npz` | P2: which object is the plant |
> | `--seed-bank` | `p4c/seed_bank.npz` | P4c: which parts are leaf/stem/root |
>
> `run_pipeline.sh` also finds the newest `*/p2/prompt_bank.npz` beside the
> workdir on its own, so usually neither needs naming. If P2 has no prompt
> source at all it now warns loudly before falling back to colour.

**You do this once for a rig, not once per plant.** The clicker writes two
files. `p2/prompts_clicked.json` holds the pixel coordinates, which fix this
video and mean nothing in the next one. `p2/prompt_bank.npz` holds the DINO
feature vectors at those points -- what a thistle and a pair of pliers *look
like* -- and that is what transfers. On a later specimen the bank is compared
against every patch of the first frame and the prompt is placed wherever most
resembles the stored plant, so a different pose, a different distance, or the
plant sitting elsewhere in frame all still work. The prompt only has to land
somewhere inside the right object; SAM2 finds the boundary itself.

```bash
pose-segment --workdir runs/thistle2 --prompt-bank runs/thistle1/p2/prompt_bank.npz
```

Clicking a few specimens into one bank widens it, the same way extra organ
seeds do: the vectors are kept individually rather than averaged, so an
example of a plant shot in shade only adds coverage. The bank records which
backbone built it and refuses to load under a different one, because
dinov2-base and dinov3-vitb16 both emit 768 numbers that mean unrelated
things.

**How well it actually transfers**, measured on the two runs in this repo
(dinov2-base, bank built only from `plant_6`, prompt scored against each
frame's P2 mask, every 8th frame):

| bank | tested on | plant prompt lands on the plant |
|---|---|---|
| 1 example, 1 frame | plant_6 (same video) | 1/12 |
| 3 examples, 1 frame | plant_6 (same video) | 8/12 |
| 9 examples, 3 frames | plant_6 (same video) | 11/12 |
| 9 examples, 3 frames | **plant_9 (different specimen)** | **10/12** |

So: click at least three plant points per pass, on different parts of the
plant. One click is not a bank. With that, the plant prompt does carry to a
new specimen, which is the case this exists for.

The **holder** prompt did not transfer in the same test -- 8/12 within
plant_6, 0/12 on plant_9. Only the plant prompt is required, so a run still
segments; what you lose is P2 subtracting the tracked holder from the plant
mask. Click the holder per specimen if you need that, or check
`plant_mask_free_of_holder` in `p2/qc.json` and only go back when it fails.

**Root examples carry the same way.** A bank whose plant examples are all
foliage will only ever place foliage prompts -- that is what lost thistle2's
root: all three located prompts landed at y 472-549, none below the jaws.
`3` = root clicks store as their own bank label, so on a later specimen the
root prompt is placed at the most root-like patch and seeds the separately
tracked root object. Appearance drift -- dirt stuck to one specimen's root
where the bank's example was clean -- is handled the same way as everything
else in the bank: click that root too, into the same bank. Vectors are kept
individually and scored by best match, so a dirty example *adds* coverage
rather than diluting the clean one, and there is no similarity threshold to
tune. When no patch out-scores the other classes, no root prompt is placed
and `root_tracked_below_the_jaws` in `p2/qc.json` flags the pass -- the
fallback is one click, which also widens the bank.

If the tool changes -- a clamp instead of pliers, a pot instead of a holder --
click a fresh bank rather than editing a threshold. There is no colour rule
left to tune.

### Plant architecture: stem or crown

P5 splits leaves by how far into them you can travel from the plant's base, so
it needs a base to start from. Which one you have is a property of the
specimen, and you pass it in:

| `--architecture` | base | for |
|---|---|---|
| `caulescent` (default) | the stem tissue P4c labelled | an upright plant with a central stem |
| `rosette` | a crown located from the geometry | thistle, sugar beet -- no stem exists |

A rosette has nothing to seed the depth field from: there is no stem to label,
and its roots sit outside the P2 plant mask. Run it as `caulescent` and P5
reports 0 contact points, 0 tips and 0 leaves. With `--architecture rosette`
the crown is found instead, by taking the geodesic extremities of the leaf
tissue and intersecting the paths between them -- every leaf-to-leaf path has
to cross wherever the leaves are joined. `p5/instancing.json` records how
tight the result was (`base_spread_fraction_of_extent`; 1.3% on thistle1),
which is the evidence the leaves really do meet at a point.

Two consequences for a rosette, both visible in Blender:

- **the base is a sphere, not a tube.** `plant_stem` holds a single crown
  node. Left as a curve it came out as a 3-node zigzag spanning 4% of the
  plant -- noise, drawn as an elbow of pipe no thistle has.
- **stem labels are ignored.** P4c calls a rosette's crown "stem", because it
  is thick and not lamina. That is a reasonable thing for a patch classifier
  to say and a bad thing to build a skeleton on: seeding four stem points on
  thistle1 had P4c label 2,214 points stem, and P5 traced a 33-node
  centreline through a plant with no stem.

### Anything the holder hides

Both P4a and P4b read `p2/masks/holder` and treat a pixel the tool covers as
carrying **no evidence**, rather than as evidence the plant is not there.

This matters for the exposed root. The pliers cross in front of it for most
of a rotation, so on thistle1 the root reached the plant mask in only 27% of
frames. A plain silhouette intersection needs 86% agreement and deleted it;
P4b's silhouette loss then trained the same tissue to zero opacity, because
"not in the mask" and "hidden" were the same thing to it. Measured: the P4a
hull now spans z -0.339..0.884 in the plant frame while the P4b surface
spanned 0.000..0.894 -- the carve kept the root and the surfels threw it away.

Neither phase lowers a threshold. Occluded views leave the denominator
instead of voting against, and P4b's silhouette term is weighted to zero on
hidden pixels. `min_judged_views` (default 8) is the floor on unoccluded
views, so a handful of agreeing voters cannot invent geometry.

If the holder was never tracked -- `p2/masks/holder` missing or empty --
both phases say so and fall back to the old behaviour. That happens when the
holder prompt lands outside the tracking crop, which `pose-segment` warns
about at the time.

**This is not inferred.** An earlier version read the crown's spread and
decided for you, and it flipped thistle1 from crown (1.6% of extent) to stem
(15.2%) purely because P4c had started labelling the crown "stem", removing it
from the tissue the test ran on -- same plant, same geometry, opposite answer.
You know which kind of plant you clamped.

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

### Organ seeds for P4c, and how to avoid clicking 50 times

There are two separate sets of clicks in this pipeline and they are easy to
mix up. The P2 *prompts* say which object in the scene is the plant, and are
covered above. The P4c *seeds* below say which parts of the plant are leaf,
stem and root. Both store DINO feature vectors so they carry to later videos;
they live in different files and are picked with different tools.

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

### Still photos instead of video

A directory of JPEGs is a capture pass, exactly like a video:

```bash
./run_pipeline.sh --photos /data/plant_9_shots --workdir runs/plant_9 --stop-after p1p2
pose-segment --photos /data/plant_9_shots --workdir runs/plant_9     # or the phase alone
```

Give several directories for several passes, and mix them with `--video`
freely. After P1 there is no difference at all: every later phase reads
`p1/frames/frame_XXXX.jpg` plus `p1/sources.json` (and `p1/intrinsics.json` /
`p1/manifest.json`, where EXIF supplied a focal and a shutter time) and cannot
tell which kind of capture produced them.

Four things to know, the first three all consequences of stills having no
redundancy:

- **Filename order must be capture order** around the turntable. SAM2
  propagates from one frame to the next and P3 checks the camera traces a
  circle; both assume consecutive frames are neighbouring angles. Cameras
  number shots sequentially, so this is usually free -- but a folder mixing
  two shoots, or renamed files, would break it silently. P1 now checks it
  against the EXIF shutter times rather than trusting it.
- **Every photo is used as it is.** A video gets sampled down to the sharpest
  frame of each angular bin; a photo directory has one shot per angle and no
  alternative to fall back on. Each photo's sharpness is printed on ingest
  and anything far below the median is named. Deleting a hopeless shot is
  your call, and it is a real trade: a dropped photo widens the angular gap
  `full_rotation_covered` measures. A soft frame that still registers is
  usually worth keeping over a hole in the orbit.
- **Photos are resized to a 1920px long edge** by default, matching the video
  path that every downstream default was fitted against. At current settings
  this costs nothing -- SAM2 resizes to 1024 regardless, P3 caps SIFT at
  `--max-image-size`, P4b trains at `--downsample` -- while a 24MP frame is
  11x the pixels through P4a's carve and P4b's rasteriser, which is a VRAM
  wall rather than a slow run on an 8GB card. `--photo-max-edge 0` keeps the
  original size; raise P3's and P4b's settings to match, or the extra
  resolution reaches nothing. Intrinsics are safe either way: COLMAP records
  the camera at the image's real dimensions even when it extracts features
  on a smaller internal copy.
- **Zooming between passes is allowed, because P1 records the lens.** The
  focal length is read from each photo's EXIF into `p1/intrinsics.json`,
  scaled to the size P1 actually wrote, and P3 gives every distinct focal its
  own COLMAP camera seeded with it (`--cameras exif`, the default). This
  matters more than it sounds: P1 re-encodes when it resizes, which drops the
  EXIF from the frame itself, so without that record COLMAP falls back to
  guessing 1.2x the long edge for every frame alike. On sugarbeet_3, whose
  three passes were shot at 48/32/22mm and solved as one shared camera at
  that 2304px guess, the three orbits came back with rotation axes 7.7
  degrees apart and P4a over-carved the hull to 0.53 IoU against masks that
  were themselves clean. Photos with no EXIF, and video, are unaffected:
  with nothing recorded the behaviour is exactly one shared camera as before.

**Several passes must be one shoot of one plant.** P1 writes a provenance
record per frame to `p1/manifest.json` -- source directory and file, EXIF
shutter time, camera body, focal -- and refuses passes that cannot be one
capture: one shot before the previous ended, more than 30 minutes apart, on a
different camera body, or with filenames out of capture order. This is the
last phase that can tell: afterwards every pass is `frame_XXXX.jpg` in one
directory, and a pass of a *different specimen* looks exactly like a second
elevation of this one. Measured on sugarbeet_4, where a mistyped `--photos`
added 13 frames of another plant shot an hour earlier: COLMAP refused to
register them and they still took the camera circle from 0.05% to 7.01% RMS,
because both shoots share a turntable and pliers for the matcher to latch
onto, and the hull fell to 0.471 IoU with no phase reporting a failure.
`--allow-mixed-capture` overrides. Video frames carry no EXIF and are skipped
rather than guessed at.

The blur trade-off is worth stating plainly, because it is the usual reason
to shoot stills in the first place: handheld photos can be *sharper* than
video frames, but they must still be **one static camera and a rotating
subject**. P3 identifies what is rigidly attached to the turntable from
per-pixel temporal variance, so a camera that moves between shots breaks the
assumption the whole solve rests on -- the backdrop stops being static and
there is nothing left to separate subject from scene.

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
