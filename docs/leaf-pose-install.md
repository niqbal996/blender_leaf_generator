# Installing leaf_pose

## One command

```bash
./setup_env.sh --leaf-pose
conda activate leafpose
```

That is the whole install. It creates a conda env called `leafpose`, installs
this repo into it, and verifies the five things `leaf-pose` actually imports.
Re-running it is safe; every step is idempotent.

It does **not** install torch, pycolmap, CUDA or gsplat. `leaf_pose` measures
leaves in one image plane, so it needs no solver and no GPU, and leaving that
stack out is the difference between a two-minute install and a twenty-minute
one. If you already have the pipeline env from `./setup_env.sh`, `leaf-pose`
runs in it as it stands — everything it needs is a subset.

## The modules, and when you need each

Each is additive. Start with the first; add the others only when a capture
asks for them.

### 1. Base — always

```bash
pip install -e ".[leaf-pose]"
```

Masks, midribs, tips, petiole origins, the diagram. Pure NumPy, OpenCV,
scikit-image, rawpy and matplotlib; no model weights, no network, no GPU.

`opencv-**contrib**-python` specifically, and the base install pins it. The
plain `opencv-python` wheel imports as `cv2` all the same, so a mismatch is
silent rather than loud, and two pieces go missing with it:

| piece | what stops working |
|---|---|
| `cv2.aruco` | the fiducial markers, so every size stays in pixels |
| `cv2.ximgproc` | the guided filter, so the mask edge falls back to a bilateral filter and is softer |

`./setup_env.sh --leaf-pose` checks for both by name and says which is
missing. Neither is fatal.

### 2. Scale — when you want millimetres

Nothing to install; measure your markers and pass the number.

```bash
leaf-pose --input <capture> --workdir runs/x --marker-mm 20
```

The markers lie in the same plane as the leaves, so they measure the scale at
the subject, which is the only place it matters. Without `--marker-mm` every
length is reported in pixels and the diagram says so — a guessed millimetre
is worse than an honest pixel.

If your markers are unlabelled but the copy stand is not, measure that
instead and the optics give the scale:

```bash
leaf-pose ... --working-distance-mm 600 --focal-length-mm 50
```

Less accurate, and worth knowing why: that distance is to the lens's rear
principal plane, somewhere inside the barrel, so a tape measure to the front
of the lens is short by an unknown amount — 30 mm out at 600 mm is a 5% error
in every length. `--sensor-width-mm` defaults to 35.7 (full frame); set it
for a crop body. The markers have no such term, so use them where you can.

A quick cross-check: with `--marker-mm` *and* `--photometric`, the run prints
the working distance it derived from your marker size. If that number is not
roughly the height of your stand, the marker size is wrong.

### 3. Photometric normals — when the capture has all its lights

Nothing to install either; it uses the frames you already have.

```bash
leaf-pose --input <capture> --workdir runs/x --photometric
```

Needs the 12-light capture (both polarisation sets) and a `rigdef_cam.xml`
somewhere at or above the capture folder, which is where the rig writes it.
Costs one extra pass over the RAW files. What it buys: a measured normal map
per leaf, and a midrib placed by the crease in that normal field rather than
by brightness. See [leaf-pose.md](leaf-pose.md).

### 4. SAM2 — only when leaves touch

```bash
./setup_env.sh --leaf-pose --with-sam      # torch + SAM2 + the weights
leaf-pose --input <capture> --workdir runs/x --instances sam
```

This is the one module that is large: a torch build, facebookresearch's SAM2
from git, and a ~857 MB checkpoint. Install it only for captures where leaves
overlap or touch, because that is the only thing it does better — on a flat
lay with gaps between the leaves the colour route separates them exactly and
in a second.

SAM2 is installed from git deliberately: the `sam2` name on PyPI is an
unrelated third-party upload.

To keep the 857 MB off this disk:

```bash
export SAM_CHECKPOINT_DIR=/big/disk/checkpoints
```

`--checkpoint-dir` does the same, and `run_pipeline.sh` reads the same
variable.

## Checking it works

```bash
pytest tests/leaf_pose -q
```

31 tests, no data and no GPU needed — they run against synthetic leaves.

Then on a real capture, smallest useful command first:

```bash
leaf-pose --input /mnt/d/PBR_Scans/2026-09-15-Naeem/gaensefuss_31 \
          --workdir runs/gaensefuss_31 --half-size --visualize
```

`--half-size` decodes the RAW files at half resolution: about four times
faster, and enough to see whether the leaves were found and which way round
they are. Drop it for the real run.

## Troubleshooting

**`no such capture`** — `--input` wants the folder of RAW frames, or a single
image file. Nothing was found with a RAW extension (`.ARW`, `.NEF`, `.CR2`,
`.CR3`, `.DNG`, `.RAF`, `.RW2`, `.ORF`) or an image one.

**Leaves missing from the result** — read `detections.json` in the workdir
before changing anything. Every blob that was found is listed with why it was
rejected, so a dropped leaf and a correctly dropped root look different. The
three rules have flags: `--min-area-fraction`, `--min-greenness`,
`--min-solidity`.

**Everything is one blob** — the backing is not dark enough relative to the
leaves, or leaves are touching. Try `--instances sam`.

**The tips and petioles are swapped** — the run prints which leaves it was
unsure about, and those are the ones to look at in `leaf_poses.png`. The
decision is explained in [leaf-pose.md](leaf-pose.md#which-end-is-which).

**`cv2.ximgproc` missing** — `pip install opencv-contrib-python` into the
same env. Something installed plain `opencv-python` over it.
