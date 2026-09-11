# P3 — camera geometry

P3 turns the P2 plant-masked frames into camera poses and a 3D point cloud.
Four backends produce that, and they are not interchangeable in quality or in
what they are good at. This document is what each one does, how to run it, and
what it measured on `thistle3`.

## The contract every backend must satisfy

Whatever produced it, a P3 result is a COLMAP sparse model plus a report:

```text
<workdir>/p3/                       COLMAP baseline (the reference)
  sparse/best/                      cameras, images, points3D
  poses.json                        registration + turntable checks
  sparse_points.ply, camera_centers.ply
<workdir>/p3/experiments/<backend>/ every learned backend, same layout
  input/images/                     the exact masked frames the model saw
  command.json, resources.json      what ran, and what it cost
  stdout.log                        the exporter's own output, as it arrived
```

Three properties are load-bearing, and each has been violated at least once:

1. **Intrinsics in original frame pixels.** P4 carving, P4c label fusion and
   the silhouette metric all project into a full-resolution P2 mask. A model
   exported in a model's internal resolution projects into the corner of that
   mask. MapAnything shipped exactly this: 518×336 cameras against 1920×1280
   masks, which scored *precisely* 0.000 in-silhouette until it was undone.
2. **Original frame names.** `images.bin` names are how P4 finds the matching
   P2 mask.
3. **Points that more than one view agrees about.** A cloud assembled by
   concatenating per-view depth maps is not a reconstruction; see VGGT below.

`rescale_model_to_frames()` enforces (1) after every export, and is a no-op
for a backend that already complies.

## Choosing a backend

Measured on `thistle3` (27 frames, two elevation passes, an RTX A6000).
`points_in_silhouette` is the fraction of projected points landing inside the
P2 plant mask — comparable across backends, higher is better. Coverage is the
fraction of mask area explained, which rewards dense models by construction.

| | registered | points | in-silhouette | coverage | per-pass circle RMS |
|---|---|---|---|---|---|
| colmap | 16/27 | 1,372 | **0.829** | 0.392 | **0.09%** |
| vggt (no BA) | 27/27 | 34,719 | 0.279 | 0.410 | 2.2% |
| mapanything | 27/27 | 189,316 | 0.480 | **0.998** | 3.7–5.2% |

Read it as: COLMAP is by far the most *accurate* and by far the least
*complete*; the learned models are the reverse. Pick COLMAP when its
registration fraction is high, and a learned backend when it is not — or when
you need points on thin structure that SIFT cannot see.

## colmap — the baseline

Masked feature matching and incremental SfM, via pycolmap.

```bash
./run_pipeline.sh --workdir <workdir> --skip-to p3        # or: pose-solve
```

The rig detail that governs everything: the camera orbits the plant and the
backdrop travels **with the rig**, so the backdrop is the one thing motionless
in image space. Match features on it and SfM concludes nothing moved. Matching
is therefore restricted to what is rigidly attached to the table — the disc,
the holder, the plant — which is what P2's variance masking separates.

Useful flags: `--low-texture` (more features, 3–5× extraction time; measured
13/27 → 26/27 frames on one capture), `--cameras exif|single|per-image`,
`--use-gpu` (needs a CUDA pycolmap build).

Its failure mode is silent under-registration. On `thistle3` it kept 16 of 27
frames and those clustered into a ~150° arc, leaving azimuthal gaps of 132.8°
and 221.0°. The cameras it *did* place are essentially perfect — 0.09% circle
RMS per pass — so the answer to a weak COLMAP solve is more coverage, not more
accuracy.

## vggt — the 2025 feed-forward model, via its own exporter

Run through upstream's `demo_colmap.py`, which is why it is the one backend
still using an upstream exporter: it owns the bundle-adjustment path.

```bash
pose-geometry --workdir <workdir> --backends vggt --bundle-adjust \
    --vggt-python ~/miniconda3/envs/vggt/bin/python
```

**Use `--bundle-adjust`.** Without it the exporter takes the
`batch_np_matrix_to_pycolmap_wo_track` path, which unprojects each view's
depth map with that view's pose and writes the union. On `thistle3` that
produced 34,719 points of which *every single one* carried a single-view
track, from 22 contributing views whose centroids scattered over 99% of the
cloud extent — 22 overlapping copies of one plant. Nothing in that path ever
asks two views to agree. The BA path predicts LightGlue tracks and runs
pycolmap bundle adjustment, which is the step that merges them.

## vggt_omega — the newer model, via this project's exporter

VGGT-Omega ships a Gradio demo and no COLMAP exporter, so
`scripts/vggt_omega_colmap.py` is the adapter. It does three things upstream's
VGGT exporter does not:

* **Fuses instead of concatenating.** A point is written only where at least
  `--consistency-views` other views put a surface at the same depth, within
  `--consistency-tolerance` of it. That is what gives the points genuine
  multi-view tracks, and therefore what makes bundle adjustment meaningful.
* **Unprojects only plant pixels**, from the P2 masks, which are passed
  automatically.
* **Writes intrinsics in frame pixels**, scaling out of the model's working
  resolution at the source.

```bash
pose-geometry --workdir <workdir> --backends vggt_omega --bundle-adjust \
    --vggt-omega-python ~/miniconda3/envs/vggt_omega/bin/python
```

Resolution: the default is **512**, which is the resolution the recommended
`vggt_omega_1b_512.pt` checkpoint was trained at, and its model card names it
the checkpoint for real captures. In `balanced` mode a 3:2 frame becomes
624×416 — an exact 3.0769× isotropic reduction from 1920×1280, so mapping the
intrinsics back is a single scale with no crop. `--image-resolution` raises it,
at the cost of leaving the regime the checkpoint was trained in; that is a
gamble, not a free quality knob.

The checkpoints are **gated**: request access, then export `HF_TOKEN`, or pass
a downloaded file with `--omega-checkpoint /path/to.pt`.

## mapanything — via this project's exporter

MapAnything has a COLMAP exporter, but it calls `model.infer()` with
`apply_mask=True, mask_edges=True` and leaves the two controls that matter for
noise at their permissive defaults. `scripts/mapanything_colmap.py` is that
exporter with them exposed, reusing upstream's own
`export_predictions_to_colmap` so the model format stays theirs:

* `apply_confidence_mask` — off upstream; on here, dropping the bottom
  `--confidence-percentile` (default 10) of confidence.
* `use_multiview_confidence` — off upstream; on here, which replaces learned
  per-pixel confidence with confidence derived from *agreement between views*.
  A point no other view corroborates is what a noisy cloud is made of.
* `--plant-masks` intersects the P2 silhouette into the prediction mask, so
  background geometry is never exported.
* `--voxel-fraction` defaults to 0.002 rather than upstream's 0.01: a plant's
  petioles and leaf tips do not survive a voxel sized for a room.

```bash
pose-geometry --workdir <workdir> --backends mapanything \
    --mapanything-python ~/miniconda3/envs/mapanything/bin/python
```

It recovers its own intrinsics, and on `thistle3` predicted a focal of 1,716
frame pixels where COLMAP solves 3,051 — a 1.77× wider field of view than the
lens has, which distorts the shape it reconstructs. MapAnything accepts known
intrinsics as an input; feeding P1's EXIF focals is the obvious next step and
is not wired yet.

## Environments

The backends cannot share one environment, and the reason is not incidental:
**VGGT pins jytime's LightGlue fork and MapAnything pins cvg's**, both of
which install as the module `lightglue`. Whichever lands last wins.

| backend | python | notes |
|---|---|---|
| colmap | this project's env | `pip install -e ".[skeleton]"` (or `skeleton-gpu`) |
| vggt | ≥3.10, its own env | `pip install -e ".[vggt]"`; pins `pycolmap==3.10.0` |
| vggt_omega | ≥3.10, its own env | `pip install -e ".[vggt-omega]"`; gated checkpoint |
| mapanything | ≥3.10, its own env | `pip install -e "<checkout>[colmap]"` |

`pose-geometry` checks the target interpreter before it stages anything —
Python version, every module the exporter imports, and whether the installed
pycolmap can build the model the exporter writes — and names the command that
fixes what is missing. `--dry-run` runs exactly that check without loading a
model, which is the quickest way to validate a new machine -- and it fetches
the exporter checkout first, because a preflight that cannot import the
package it is validating would report a working interpreter as broken.

`--vggt-python`, `--vggt-omega-python` and `--mapanything-python` override
`--model-python` per backend, which is what makes one comparison run across
several environments possible.

## Comparing what came out

```bash
pose-compare-geometry --workdir <workdir>      # discovers what is on disk
```

`p3/experiments/compare.json` carries two independent families of number:

* **Agreement with COLMAP** — camera centres of matching frames aligned by a
  best-fit similarity, then residuals. Low means the candidate agrees with the
  baseline. It is not accuracy: if COLMAP is the one that drifted, the better
  reconstruction scores worse. Read residuals relative to the orbit radius,
  which `poses.json` records.
* **Agreement with the silhouettes** — `points_in_silhouette` and
  `silhouette_coverage`, which need no reference reconstruction because the P2
  masks are independently trusted.

Then carve, which is the first stage that requires the views to agree with
*each other* rather than individually:

```bash
./run_pipeline.sh --workdir <workdir> --skip-to p4a --geometry-backend vggt_omega
```

An empty hull reports how far it missed by: the best voxel's agreement against
the threshold, and whether that gap looks like pose error or like masks from
another capture. Treat `--min-inside-fraction` as a fixed gate and read the
achieved agreement as the measurement — a backend that needs the gate lowered
has told you something, and tuning it per backend destroys the comparison.

Phases after P4a read the baseline `p3/` and `p4/`, so `run_pipeline.sh` stops
after P4a for a learned backend rather than mixing one backend's poses with
another's hull.
