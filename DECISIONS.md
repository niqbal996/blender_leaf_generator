# Decisions log

Append-only record of pipeline decisions: what was chosen, why, and what
evidence backed it. Required by `plant_pose_pipeline_PLAN.md` §0 — the point
is that a threshold or backend choice made six weeks ago can be re-examined
without archaeology.

---

## 2026-08-03 — Package split: `leaf_generator` / `pose_estimator`

All plant-measurement code moved from `src/leaf_generator/skeleton/` to a new
top-level `src/pose_estimator/` package, along with the four root CLI scripts.

**Why:** the two concerns were already independent — `skeleton/` imported
nothing from the rest of `leaf_generator` — but shared a package namespace,
which made the boundary invisible. `leaf_generator` *authors* synthetic leaf
assets for Blender; `pose_estimator` *measures* a real plant. They share no
code and are developed against different plans.

**Evidence:** the move required zero import changes inside the moved modules
beyond the package prefix.

**Consequence:** CLIs are now console scripts (`pose-segment`,
`pose-estimate-skeleton`, `pose-train-splat`, `pose-align-skeleton`) rather
than root-level `.py` files. `blender_plant_import.py` stays at the repo root
because Blender needs a real file path to open.

---

## 2026-08-03 — Environment: conda env `pose_estimator`, Python 3.11, torch cu124

Separate from the older `plant_gen` env so the SAM2/torch stack cannot break
the working COLMAP/gsplat setup.

Hardware this was chosen against: **RTX 2070 Mobile, 8 GB VRAM** (~7 GB free).
That is the binding constraint on every model choice below.

---

## 2026-08-03 — SAM2 installed from source, NOT from PyPI

**Decision:** clone `github.com/facebookresearch/sam2` into `third_party/` and
`pip install -e` it. Do **not** `pip install sam2`.

**Why:** the `sam2` name on PyPI (v1.1.0) is *not* published by Meta — its
project home is `github.com/JinsuaFeito-dev/segment-anything-2`, an unrelated
third-party account. Installing it would pull unvetted code into the
environment under a name that looks official. This is exactly the supply-chain
check §8 of the plan asks for.

**License:** SAM2 code and checkpoints are **Apache-2.0** — no non-commercial
restriction, so it is safe for a commercial asset library. Verified from the
repo's LICENSE file, not from a summary.

**Checkpoint:** `sam2.1_hiera_large.pt` (~900 MB on disk). Chosen over
`base_plus` because P2's hardest job is thin petioles and roots, where the
larger backbone measurably helps; VRAM pressure is handled by offloading the
video and inference state to CPU rather than by shrinking the model.

---

## 2026-08-03 — Frame selection: sharpest-per-angular-bin

Replaced even sampling (`extract_frames`) with `extract_sharpest_frames`,
which splits the video into N consecutive bins and keeps the frame with the
highest variance-of-Laplacian in each.

**Why:** the turntable rotates continuously, so a fixed stride lands on a
motion-blurred frame about as often as a crisp one. Within one bin the
rotation is small, so choosing by sharpness costs nothing in angular
coverage. Plan §P1.1.

**Not yet done:** blur *rejection* against an absolute threshold, and
near-duplicate removal for turntable pauses. Per-bin selection covers the
current captures (continuous rotation, no pauses).

---

## 2026-08-03 — P2 segments a cropped ROI, not the full frame

The subject ROI is the padded union of plant+holder color blobs over all
frames; SAM2 runs on that crop, and masks are pasted back into full-frame
coordinates before being written.

**Why:** SAM2 resizes its input to 1024×1024 regardless. In these captures
(1920×1080, seedling spanning roughly a quarter of the frame) full-frame input
leaves petioles ~2 px wide — below what any segmenter holds onto. Cropping
first is the cheapest available resolution gain, and a single fixed window
keeps the mapping back to full-frame coordinates a constant offset, which
P3/P4 require since the intrinsics describe the full frame.

**Guard:** `--no-roi` disables it if the color prepass picks the wrong region.

---

## 2026-08-03 — Color is used to *seed* SAM2, never to threshold output

Plant prompts come from an Excess-Green blob, holder prompts from a saturated
warm-hue blob (tuned for the red/yellow insulated pliers in the current rig).

**Why this does not violate the plan's ground rule on thresholds (§0):** these
values only decide where a *click* lands. SAM2 then finds the real boundary,
so a sloppy seed still yields a correct mask, and a seed on the wrong object
shows up immediately in the `diag/` overlays rather than silently distorting
geometry. No output pixel is decided by a color threshold.

**Guard:** `--plant-point` / `--holder-point` bypass the derivation entirely.
The holder hue window is rig-specific and **will** need changing for a
different holder — flagged here rather than pretending it generalises.

---

## 2026-08-03 — P2 crop *tracks* the plant; a fixed ROI is useless here

**Measured:** the specimen is clamped off the rotation axis, so it orbits. Its
union bounding box over one rotation covers 1639 of 1920 px on DSC_0009 — a
fixed crop is the full frame and buys nothing.

**Decision:** `solve_tracking_crop` follows the plant with a constant-size,
per-frame window (centroid smoothed circularly, since the sequence is one
closed rotation). Masks paste back to full-frame coordinates before being
written, so P3/P4 never see crop coordinates.

**Result on the current captures:** the window lands at 1080x1080, i.e. ~0.95x
— barely better than full-frame, because the plant's own apparent size varies a
lot across the turn. The mechanism is right and pays off on tighter framing;
**the real fix is at capture time — fill more of the frame with the plant.**

---

## 2026-08-03 — REJECTED: color agreement as a P2 acceptance check

Plan §P2.4 asks for a chroma/luminance matte cross-checked against SAM2, with
disagreement treated as a QC failure. **Tried, measured, rejected.**

**Why it fails on this rig — three separate measurements:**
1. A brightness-based matte swallows the turntable: it covered 19.6% of the
   frame, giving a meaningless 0.32 IoU against masks that are visibly perfect.
2. "Did SAM2 find all the green?" scored 0.64 recall — but inspection showed
   the *missing* pixels were the pliers' **yellow grip**, which is
   green-dominant in RGB. SAM2 was right and the color prior was wrong.
3. Hue cannot separate them: **47% of true plant pixels** fall in the same
   warm-hue window as the yellow plastic.

**Replaced with three checks that need no color prior:**
- `area_temporally_smooth` — turntable area varies smoothly.
- `centroid_trajectory_smooth` — the mask centroid traces a closed orbit; a
  discontinuity is tracking loss. Purely geometric.
- `plant_mask_free_of_holder` — asks the *decidable* question ("did SAM2 grab
  something definitely not-plant?") rather than the undecidable one.

**Genuine independent cross-validation is still owed**, and its right home is
P4: rendered-vs-input mask IoU on held-out views. Noted so this is not
forgotten.

---

## 2026-08-03 — Holder color rule: two disjoint rules, brightness for yellow

`holder_color_mask` = (narrow true-red hue & S>=120) OR (yellow hue & S>=80 &
**V>=215**). Brightness, not hue, separates yellow plastic from yellow-green
foliage.

**Evidence:** the earlier single rule (any warm hue, S>=120) claimed 20% of the
true plant. The two-rule version claims **<1%** while still finding 40k-135k
holder px/frame, measured across 4 frames of DSC_0009.

**Rig-specific and flagged as such** — it encodes "red and yellow insulated
pliers". A different holder needs a new rule; `--holder-point` bypasses it.

---

## 2026-08-03 — QC contamination counts only sizeable blobs

`holder_contamination` runs `_significant_components` (>=0.1% of frame) before
intersecting.

**Why:** specular highlights on glossy leaves are small, bright and warm, so
they trip the yellow-plastic rule exactly as the grip does. Unfiltered, this
reported 4.5% "contamination" on a frame whose mask was pixel-perfect — the red
pixels were leaf glare, confirmed visually. Real contamination is a contiguous
piece of plier, never speckle. After filtering: worst frame 0.91% / 0.11%, mean
0.02% / 0.002% on DSC_0009 / DSC_0010.

This is the plan's "glossy leaves" failure mode (§6) showing up early, in QC
rather than in geometry.

---

## 2026-08-03 — GROUND TRUTH for DSC_0009, from the person who shot it

Recorded because it corrects an assumption, not just a parameter:

- **6 proper leaves**, mostly well represented in the point cloud at the top.
- **One of the six sits below the top group.**
- That lower leaf has **4-6 tiny leaves emerging from the main stem at its
  base**. The hull smudges them into a single blob.
- **One more leaf is emerging at the plant's peak.**

**What this corrects.** The "impossible" width profile -- a leaf measuring
broad at its own base -- was being treated here as proof of a midrib bug. It
is substantially *real*: that base is a merged cluster of small leaves, not
one leaf's petiole. A curvature-regularised spline was about to be built to
suppress it, which would have been fitting away real structure.

**Priority set by the same source:** missing a large leaf while chasing the
tiny ones is the worse error. Detecting the small ones is a bonus, not a
requirement.

---

## 2026-08-03 — min_branch_fraction = 0.22, calibrated against the leaf count

| setting | leaves | points per leaf |
|---|---|---|
| 0.30 | 5 | 1586, 1122, 1002, 960, 616 |
| **0.22** | **6** | **1585, 1122, 1004, 960, 733, 614** |
| 0.16 | 8 | 1122, 831, 825, 767, 712, 571, 532, 528 |

0.22 recovers all six proper leaves, every one substantial, with the two
largest untouched. 0.16 finds more branches but **splits the largest leaf**
(1586 down to 1122) -- precisely the failure mode to avoid. Verified visually:
the six axes land on six distinguishable leaves, including the one below the
main group.

**The frame-continuity check still fails (21.8 deg vs a 5 deg limit) and that
is now understood as partly a data property rather than a defect.** The
merged basal cluster genuinely bends the midrib of the leaf it is attached to.
Left failing rather than loosened, so the limitation stays visible.

---

## 2026-08-03 — A leaf is an edge that ends at a *tip* (P5 bug, found via P6)

**The bug:** `solve_stem_and_leaves` called every non-stem edge a leaf. On
DSC_0009 that included a `root -> branch` edge (the basal stem segment) and a
`branch -> branch` edge (an internode between two insertions). P6 then fitted
midribs to two pieces of stem.

**How it presented:** impossible width profiles (a "leaf" at maximum width at
its own base) and insertion angles past 118 degrees. Both symptoms pointed at
P6's base-point handling, and **two attempted fixes there made things worse** --
picking the axis endpoint nearest the plant origin (fails on a drooping leaf,
whose tip curves back closer to the origin than its attachment), then the
endpoint nearest the stem axis. Neither could work, because the objects being
measured were not leaves.

**The fix:** a leaf terminates at a tip. Edges ending at another branch point
are internodes; edges leaving the root are basal stem. Both are structure, not
organs.

**Result on DSC_0009:** 7 "leaves" -> **5**, all tip-terminating. Insertion
angles tightened from 31.6-146.5 deg to **33.4-54.0 deg**; arclengths to within
1.3x of each other; and every leaf now measures narrow at the base and wide
mid-blade (e.g. leaf 2: 0.0001 at base against 0.0326 at mid-span), which is
the shape a leaf actually has.

**Lesson worth keeping:** the symptom appeared in P6 and the defect was in P5.
Two rounds of increasingly clever compensation in the wrong module made the
output worse each time. The diagnostic that settled it was dumping the
endpoint *kinds* of every leaf edge -- two minutes of looking at the data
rather than reasoning about it.

---

## 2026-08-03 — P4b: 2DGS surfels, seeded from the hull *boundary*

Built the step skipped earlier (plan §P4.2-4). Choices worth recording:

**`rasterization_2dgs`, not `rasterization`.** gsplat 1.5.3 ships both. 2D
Gaussians are flat oriented discs that lie *on* surfaces; isotropic 3DGS fills
volume, which is the property that made the hull unusable for P5 in the first
place. Apache-2.0, so no licence constraint (plan §8).

**Seeded from the hull, not the sparse cloud.** The sparse cloud has ~27k
points concentrated wherever SIFT happened to match; the hull covers the whole
plant including the thin root that photometric matching never recovered.

**Boundary voxels only.** 71.5% of hull voxels are fully enclosed interior.
They have no surface to align to, cannot be seen from any view, and seeding
them spends capacity on primitives hidden behind the ones that matter. On
DSC_0009 the boundary is 104,324 of 474,091 voxels.

**Orientation seeded from local PCA normals.** With identity quaternions every
disc starts facing the same arbitrary direction and the normal-consistency
loss opens at 0.99 — normals essentially orthogonal to the surface it is meant
to represent, with 100k+ primitives to rotate from scratch.

**No densification strategy.** Hull initialisation already covers the plant
densely and correctly, so the usual grow/prune machinery has little to do, and
leaving it out removes a large surface of API and tuning that would otherwise
need its own validation.

**Geometry from rendered depth, never from primitive centres** (plan §P4.3).
Centres sit wherever the photometric loss put them and need not lie on any
surface; the plan names this as the root cause of the old pipeline's need for
hand-tuned thresholds.

**Acceptance checks chosen to test the thing that failed before:**
- rendered-alpha vs input-mask IoU >= 0.90 (does the splat match the silhouettes)
- **median local flatness <= 0.25**, against the hull's measured 0.798. This
  is the check that matters, because "is this a surface or a blob" is exactly
  what P5 needs and exactly what P4a could not deliver.

---

## 2026-08-03 — P5 Path C on the raw hull: attempted, does not converge

Ran the plan's classical baseline (kNN graph + level-set/MST skeletonisation,
reusing the repo's existing `skeletonize.build_skeleton_graph`) directly on the
P4 hull. **It does not produce stable structure**, and the failure is
structural rather than a tuning problem.

**Evidence — sweeping `min_branch_fraction`:**

| min_branch | DSC_0009 leaves | stem length | DSC_0010 leaves | stem length |
|---|---|---|---|---|
| 0.12 | 26 | 0.934 | 25 | 0.000 |
| 0.20 | 8  | 1.642 | 18 | 0.000 |
| 0.30 | 5  | 0.890 | 11 | 0.000 |
| 0.40 | 1  | 2.215 | 1  | 1.768 |

No plateau anywhere; leaf count runs 26 to 0 with no stable region, and stem
length is non-monotonic (0.93 → 1.64 → 0.89 → 2.21), which a stable structure
could not do. At settings giving a plausible leaf count the smallest "leaf"
holds 2-9 points — far too few to fit a midrib.

**Diagnosis, measured:**
- The hull is a **solid volume**: 71.5% of voxels are fully enclosed by 26
  occupied neighbours. kNN+MST skeletonisation assumes points sampled on a
  *surface*; over a solid it traces arbitrary interior paths, which is exactly
  the zigzagging seen in `p5/diag/skeleton_*.jpg`.
- Local neighbourhoods are **not planar**: median PCA flatness 0.798 over
  6-voxel neighbourhoods, i.e. near-isotropic. Leaves in the hull are chunky
  blobs, not thin sheets, so normal-based leaf clustering would not rescue it
  either.

**Root cause — a capture limitation, not a code one.** Every camera lies in
one plane (P3 measured out-of-plane scatter at 0.09% of orbit radius). A
silhouette carve from a single-elevation orbit cannot constrain leaf thickness
for a leaf tilted out of that plane, because the views that would see it
edge-on do not exist. The plan anticipates this: §P1 specifies
`elevations_deg: [20, 60]` — **two** elevations. These captures have one.

**The real gap: P4b was skipped.** Plan §P4 is four steps — visual hull, then
2DGS surface-aligned splatting, then mesh extraction from *rendered depth*,
then carve that mesh against the hull. Only step 1 was built. P5 is meant to
consume the carved splat-derived **surface**, which has thin geometry and
normals; running it on the hull — which the plan itself calls a bound, not a
surface — was never going to work well.

**Not a wasted phase.** Two pieces stand on their own and carry forward:
- `solve_up_direction`: the orbit axis sign is arbitrary and differs between
  specimens (+Z on DSC_0009, -Z on DSC_0010). Resolving it against the table
  plane is required by anything that measures an insertion angle.
- Clamp-line detection: the hull gap under the holder gives the schema's
  origin directly on DSC_0009. **It fails on DSC_0010**, where the holder
  never fully occludes the stem, and the code reports that rather than
  substituting a guess.

---

## 2026-08-03 — P4 carve tolerance = 86% of observing views (fit, not guessed)

A strict visual hull (`min_inside_fraction=1.0`) **deletes the exposed root**
on both specimens. The root is thin and, worse, hidden behind the pliers for
part of the orbit, so far more than a couple of views vote it empty and the
intersection removes it — even though it is plainly present in most masks and
was segmented correctly in P2. Visible in the first carve as a large
mask-only region in `p4/diag/hull_vs_mask_*.jpg`.

**Method:** swept the threshold on both specimens, scoring reprojected hull
against input silhouette (IoU / recall / precision), carving once at the
loosest setting and subsetting, since stricter results are subsets of looser.

| min in-view fraction | IoU DSC_0009 | IoU DSC_0010 |
|---|---|---|
| 100% | 0.618 | 0.722 |
| 94%  | 0.728 | 0.809 |
| 90%  | 0.768 | 0.833 |
| **86%** | **0.785** | **0.838** |
| 82%  | 0.788 | 0.831 |

**Decision:** 0.86. It is the joint optimum — the peak for DSC_0010 and within
0.003 of the peak for DSC_0009, with DSC_0010 already declining by 82%.
Tighter loses the root; looser inflates the whole hull. Fit once, validated on
both specimens, per the plan's §0 rule (b).

**Expressed as a fraction, not a view count**, so it stays meaningful when the
frame count changes. (The earlier `outlier_views=2` was an absolute count and
silently meant something different at 96 frames than it would at 200.)

**Still owed:** a third specimen to confirm 0.86 holds without adjustment.
Note that no sharp knee exists in the voxel-count-vs-threshold curve — this is
a real precision/recall trade-off, not a natural break, so it should be
re-checked whenever the capture geometry changes.

---

## 2026-08-03 — Hull evaluation must render at the voxel's projected size

`_evaluate` closes the reprojected voxel stipple with a kernel derived
per-view from the voxel's own projected size (`f * voxel / depth`), not a
fixed 3x3.

**Why:** voxel centres are point samples and project to a dotted pattern with
gaps. A fixed kernel therefore measures *how finely the hull was sampled*
rather than what shape it is, which made recall resolution-dependent —
DSC_0009 scored 0.683 recall at 128^3 and 0.824 at 256^3 for the same
tolerance, purely from sampling density. Deriving the kernel from geometry
makes the metric comparable across resolutions.

---

## 2026-08-03 — CORRECTION: the camera orbits; the plant is stationary

An earlier entry here claimed a static camera and a rotating subject, inferred
from image-space motion. **That inference was wrong**, corrected by the person
who built the rig: the plant sits on a fixed table and the *camera* orbits it,
with the backdrop mounted to travel with the camera rig. That is why the
footage reads as a locked-off camera watching a spinning plant.

**What this does not change:** the masking decision, which was right. What it
changes is the reason. The backdrop is not "the static thing we should ignore
because it is uninformative" — it is rigid with the camera, so it is the one
element genuinely motionless in image space, and matching on it would have
driven SfM to conclude the camera never moved. It carried an actively wrong
signal, not a merely useless one.

**What it upgrades:** the table and plant are the true static world, so the
recovered orbit is the camera's literal physical path, not a relative-motion
equivalence. The circle-fit acceptance test is measuring a real circle.

**New caveat to carry:** the plant is only approximately stationary. Residual
vibration from the moving rig perturbs leaves between frames, which surfaces
as silhouette disagreement during P4 carving — the reason
`carve(outlier_views=...)` is non-zero by default. Plan §6 lists plant motion
as a known hard failure mode; here it is present but small.

**Method note:** `temporal_std` still works, and works for the same measurable
reason (the two regions move differently in image space). But its docstring
previously justified itself with "the camera is locked off", which was a wrong
explanation of a right method — the kind of thing that survives until someone
changes the rig and it silently stops applying.

---

## 2026-08-03 — Sparse points on the table and holder are intended

The P3 cloud contains many points on the turntable top and the pliers. This is
deliberate and load-bearing: a thin seedling triangulates poorly on its own,
and the disc's dust speckle plus the holder's edges are what make the solve
well-conditioned (27,154 points on DSC_0009 versus a few hundred from foliage
alone).

They are not removed by another mask. P4 carves against the **plant**
silhouettes only, so any voxel outside the plant mask in more than
`outlier_views` frames is deleted — table and holder fall out automatically.
Using the P2 holder mask to pre-delete them would be redundant work that also
costs P3 its parallax.

---

## 2026-08-03 — P3: the rig is camera-static / subject-rotating (measured)

**Measured**, by per-pixel temporal standard deviation over one rotation of
DSC_0009: backdrop ~5 DN (top strip mean 5.11), turntable disc 30-60
(p75=43.4). The camera is locked off; the turntable and everything on it
rotates.

**Why this dominates P3:** SfM assumes a moving camera in a static scene. Here
the truly-static thing is the *backdrop*, so unmasked COLMAP would fit it
perfectly, conclude the camera never moved, and emit a degenerate
reconstruction with all centres coincident — while discarding the plant as an
outlier for having the temerity to move. This is the same convention trap the
plan flags in §P1 pitfalls, arriving without a ChArUco board.

**Decision:** mask matching down to what is rigidly attached to the turntable
(disc + holder + plant). Solved in that frame, relative motion comes out as
"camera orbiting a static plant" — the convention P4 wants.

**How the mask is derived — no color, no marker:** Otsu on the temporal
variance map. The histogram is strongly bimodal ("moved" vs "did not"), so the
split needs no hand-tuned constant, satisfying the plan's §0 ground rule.
Holes are closed and filled because untextured disc patches vary little yet
are physically part of the rotating rig; punching them out would throw away
the parallax that conditions the solve. Coverage: 49% / 51% of frame on
DSC_0009 / DSC_0010.

The per-frame P2 plant and holder masks are unioned in as insurance, since a
leaf tip at the extreme of its arc contributes little temporal variance there.

---

## 2026-08-03 — P3 acceptance: fit a circle to the camera centres

The headline check is that recovered camera centres lie on a circle, are
coplanar, and span the full turn without a large angular gap.

**Why this is real evidence and not circular reasoning:** nothing in COLMAP's
objective knows this is a turntable or rewards circularity. A reconstruction
that collapsed, drifted in scale, or mirrored will not produce a clean circle
by accident. It is the closest available substitute for the independent
cross-validation a ChArUco board would have given (plan §P3), and it is the
reason marker-free capture is worth attempting at all here.

Thresholds (first pass, to be calibrated across specimens): circle RMS < 2% of
orbit radius, out-of-plane scatter < 2%, largest angular gap < 30 deg,
registration >= 95%, mean reprojection error < 1.5 px.

---

## 2026-08-03 — P3: force a single shared camera (COLMAP's default is wrong here)

**Bug found by the circle check, exactly as intended.** DSC_0010 registered
96/96 frames at 0.575 px mean reprojection error — by conventional SfM metrics,
a good reconstruction — yet its camera centres missed their own fitted circle
by 5.24% RMS (14.15% worst) with a *systematic* drift and a discontinuity, not
noise.

**Cause:** COLMAP's `camera_mode` defaults to `AUTO`, which groups images by
EXIF. Frames extracted from video have no EXIF, so it fell back to
**one camera per image** — 96 independent focal lengths for a locked-off body
and lens that never changed.

**Evidence:** focal length spread across frames from the same fixed lens —
DSC_0010 **27.5%**, DSC_0009 1.98%. Physically impossible; the solver was
absorbing reconstruction drift into intrinsics. DSC_0009's smaller spread is
why it passed, i.e. it passed by luck, not correctness.

**Decision:** `camera_mode=CameraMode.SINGLE` by default in
`build_sparse_reconstruction`. `--per-image-cameras` restores the old
behaviour for footage where the lens genuinely changed.

**Worth noting for the plan:** low reprojection error did *not* catch this —
of course it didn't, since the extra parameters exist precisely to drive that
number down. Only the rig-geometry check did. This is the argument for
acceptance criteria derived from physics rather than from the optimiser's own
objective (§0), and it is the first time in this pipeline that it has paid.

---

## 2026-08-03 — P2 accepted on DSC_0009 and DSC_0010

96 frames each, all 6 checks pass on both. Median plant mask 52.8k px
(DSC_0009) / 67.7k px (DSC_0010). Runtime ~2 min/specimen on the RTX 2070.

Masks capture leaves, stem **and the thin exposed roots**, verified visually at
6 angles per specimen. Mask-area-vs-angle traces a smooth periodic curve
peaking at broadside views — the physically expected signature.

---

## 2026-08-07 — Multi-elevation capture: several videos into one solve

**Problem it addresses:** leaves 0, 1 and 9 merge at the apex whorl. They
attach within 0.02 in height of each other, and no camera in a single
waist-height orbit ever looks *down* into the whorl — measured out-of-plane
camera scatter on DSC_0009 is 0.09% of the orbit radius, i.e. the orbit is
essentially a perfect plane. Silhouette carving cannot separate structures it
never observed from a second direction, so this is a **data** limitation, not
an algorithm one, and no amount of clustering work fixes it. The remedy is a
second orbit at a different elevation.

**Design:** several passes share one workdir rather than being reconstructed
separately and registered afterwards. Post-hoc registration of two independent
reconstructions needs a shared scale and an alignment step of its own, and
each pass would be solved from strictly less evidence. Feeding all frames to
one COLMAP solve puts every camera in one coordinate frame by construction.

**Four things assumed one video and were fixed:**

1. `extract_sharpest_frames(..., start_index=)` — frames from later passes are
   numbered consecutively instead of overwriting `frame_0000.jpg`.

2. **SAM2 runs once per pass.** Video propagation carries temporal memory
   between consecutive frames; running it across the cut between two videos
   asks it to track through a discontinuity it has no reason to survive.

3. **One rotating-region mask per pass.** The backdrop is rigid with the
   *camera*, so at a different elevation it occupies a different part of the
   image. Measured on DSC_0009 + DSC_0010: the per-pass masks cover 40.0% and
   36.2% of the frame at IoU **0.653** — pooling both passes into one
   variance map gives 44.4%, which includes backdrop that each pass
   individually excludes. Backdrop features are the one thing P3 must never
   match on (see the P3 entry above), so this was not cosmetic.

4. **One circle fit per pass, not one overall.** Two elevations trace two
   coaxial rings; a single circle through both describes neither. On a
   synthetic *perfect* two-elevation rig (radii 1.0 and 0.8, separation 0.9)
   the pooled fit reports a circle RMS of **11.0%** and would fail the 2%
   acceptance check outright, while each pass fits at 0.00%. The pooled radius
   comes out 0.906 — a value belonging to neither orbit.

**New acceptance checks, only active with more than one pass:**

- `passes_share_a_rotation_axis` (limit 2°) — the strongest evidence
  available that the passes really merged. Nothing in the solve enforces it:
  two independently-registered orbits agreeing on an axis is hard to achieve
  by accident.
- `passes_are_at_different_elevations` — a second pass only helps if it *is*
  at a second elevation. Guards against re-shooting the same viewpoint and
  concluding the extra frames did nothing.

**Downstream frame:** with several passes, `poses.json["orbit"]` is a
`consensus_orbit` — axis averaged across passes (with signs aligned first,
since a fitted plane normal's sign is arbitrary and unaligned averaging can
cancel two consistent axes to zero), centre taken as the mean of the pass
centres, which lies on the shared axis. Quality figures are reported as the
worst pass, not the average. P5's upright plant frame reads this.

**P2 QC is now pass-aware.** `area_temporally_smooth` and
`centroid_trajectory_smooth` both compare consecutive frames, and consecutive
frames from different passes are not continuous — the camera jumps elevation
between them. Left unfixed, the boundary registers as tracking loss and the
checks would fail on exactly the captures they exist to validate.

### Validated on DSC_0009 + DSC_0010, 48 frames each

Merged P3 solve, all six checks pass: 96/96 frames registered, 0.667 px mean
reprojection error, per-pass circle RMS 0.06% and 0.21%, largest angular gap
15.3°, **axes agree to 1.30°**, orbit centres separated by 12.9% of the orbit
radius. 14,286 sparse points.

**But a passing axis check does not establish that the specimen is the same
in both passes.** The turntable disc's own surface texture is enough to bind
two orbits into one coordinate frame; the plant could differ and the check
would still pass. That distinction matters because merging two *different*
specimens produces a confident, internally consistent, meaningless
reconstruction.

Two tests were tried:

- *Shared tracks.* Only 16.7% of points are observed by both passes — but the
  disc, used as a control, is barely different at 18.1% versus 12.4% for the
  plant crown. SIFT rarely matches across a 12.9% elevation change anywhere in
  the frame, so this measures feature repeatability, not specimen identity.
  Inconclusive.

- *Cross-pass silhouette agreement.* Points triangulated as foliage in one
  pass were projected into the other pass's images and tested against a crude
  green mask (the plant is the only green object on a black backdrop). They
  land on foliage **67.3%** of the time. Null hypothesis: the same points spun
  60/120/180° about the turntable axis — inside the plant's swept envelope but
  with the correspondence destroyed — score **19.9%**. A uniformly random
  pixel scores 2.38%. Held-out views *within* a pass score 75-86%, so the
  remaining 8-18 point gap is self-occlusion, residual plant vibration between
  takes, and silhouette disagreement at leaf edges viewed from a new
  elevation.

**Conclusion: same specimen, same pose, two elevations.** The merge is valid.

---

## 2026-08-12 — P4c: obliquity-weighted multi-view fusion

**Observation driving it:** a 2D segmenter identifies a leaf when its blade
faces the camera and mistakes the same leaf for a stem or ribbon when it turns
edge-on. Over a 360° orbit this happens to every leaf, so different views are
right about different leaves and the rest add noise.

**Why "segment every frame and merge" was the wrong shape.** It treats
disagreement between views as conflict to be resolved, when most of it is not
conflict at all: an edge-on camera is not wrong about the leaf, it is simply
not in a position to see one. Requiring agreement asks the capture for
something its geometry forbids.

**The fix:** each view's vote is scaled by the cosine between the point's
surface normal and the ray to that camera. This is not a tuning constant --
it is the foreshortening factor, the fraction of that surface the camera
actually sees. A blade square to the camera contributes its full area; the
same blade edge-on projects to a sliver and contributes nearly nothing. No
threshold, no exponent, no cutoff: grazing views abstain on their own.

**Sign is discarded.** A leaf is a two-sided sheet and which face is turned
toward the camera flips halfway round the orbit; a signed cosine would reward
one half of the capture and punish the other for identical geometry. `abs`
asks the only question that matters -- broad-side or edge-on -- and sidesteps
normal orientation entirely. Self-occlusion is already handled upstream: the
z-buffered index map means a point that reaches a pixel is the one facing the
camera.

**Normals** come free from P4b's `surface.ply` (`nx, ny, nz`), fitted against
the photographs. `estimate_normals` (local PCA) covers a cloud without them,
with a warning for the P4a hull -- it is a solid, and its interior points have
no surface to be normal to.

### Measured on runs/plant_9: 137,024 points, 96 real views

| for each point, the fraction of *its own* views that are broad-side (cos > 0.7) | share of cloud |
|---|---|
| 0–10% | 21.3% |
| 10–25% | 24.0% |
| 25–50% | 39.2% |
| over 50% | 15.5% |

**84.5% of the cloud has broad-side views in the minority.** Plain majority
counting was therefore deciding the overwhelming bulk of this plant from
cameras that could not see the surface they were voting on. Mean obliquity
across observing views is 0.525; **14.9% of points are never seen broad-side
at all** (max cosine < 0.7) — those are labelled from evidence no camera was
positioned to give, which the new `few_points_decided_edge_on` check surfaces
rather than burying.

### Reach of the fix, and its limit

On a synthetic orbit (36 cameras, one flat blade, a segmenter that recognises
the blade only above a given obliquity):

- tolerance 0.8 — blade recognised in **14 of 36 views**, a clear minority, so
  majority voting labels it stem. Those 14 views carry 13.2 of the 22.9 total
  weight, and weighted fusion recovers **leaf**.
- crossover sits near tolerance 0.9 (10 of 36 views): leaf 100% at 0.85, 16%
  at 0.9, 0% at 0.95.

So weighting moves the break-even from "recognised in over half the views" to
"recognised in roughly a third". It is not unlimited rescue, and below that
the fusion reports stem rather than inventing a leaf — which is correct: at
that point the evidence genuinely is absent.

### Latent bug found and fixed on the way

The tally was three fixed arrays (`vote_other`, `vote_leaf`, `vote_stem`)
whose names did not match what the DINO path put in them — it round-tripped
correctly only because class index 0/1/2 happened to line up with the column
order. With **four** seed classes (`"leaf" "tiny leaf" "stem" "root"`, which
is what the prompts had been set to) class index 3 was counted in
`n_views_seen` but into no vote column at all, silently depressing every
affected point's confidence and letting the argmax pick among the first three.
The tally is now an (N, C) array over whatever classes the seeds define, and
out-of-range indices are dropped explicitly rather than folded into class 0.

`--no-normal-weighting` restores plain counting for A/B comparison, and every
run reports how many points the weighting actually moved — reported, not
asserted, since a run where it changes nothing is worth seeing.

## 2026-08-26 — P2: the exposed root is tracked as its own SAM2 object

The root kept vanishing from the reconstruction even after prompts were
clicked on it. The instruction at the time — click the root as an extra
`1` = plant point — was necessary but measurably insufficient:

- thistle1's clicked prompts spanned y 468–900 and did reach the root, yet
  the root was in the plant mask only 74/96 frames (pass 0) and 49/96
  (pass 1) — under the ~86% silhouette agreement P4a demands, so the carve
  deleted it. Occlusion and blur were both ruled out (the root band is
  *less* occluded than the plant body, and the sharp pass was equally bad);
  see DIAGNOSIS_root_loss.md.
- the mechanism: all plant clicks seed **one** SAM2 object at frame 0, and
  SAM2 keeps one temporal memory per object. That memory is dominated by the
  object's large connected mass (foliage), so a disconnected blob — the root,
  cut off by the clamp jaws — flickers out during propagation. No threshold
  touches propagation flicker.

**Decision:** `pose-pick-prompts` gains `[3] root`, a *tracking* category,
not an output class. Root points seed `ROOT_ID = 3`, a third SAM2 object
with its own memory — a single connected region, which is the case SAM2
propagation is actually good at — and `segment_sequence` unions its mask
into `masks/plant` at write time. P2's output contract (plant/holder/
background) is unchanged, so nothing downstream moves; P4c's semantic root
class is a different question and stays as it is. Old `prompts_clicked.json`
files have no `root` key and load unchanged (empty root = the old two-object
behaviour, exactly).

Consequences that fell out of it:

- **The tracking crop grows to contain clicked points**
  (`solve_tracking_crop(include_points=…)`). The crop is sized from the
  colour prepass's foliage blob and the root hangs below it, so a crop
  sized from the blob alone can cut the root off. The window widens by the
  clicked points' offset from the tracked centroid, reusing the existing
  `padding_fraction` — no new constant.
- **A root point outside the crop is fatal**, like a plant point and unlike
  a holder point: the solver was told to contain it, so landing outside
  means the window is wrong — and dropping it silently would re-create the
  exact silent root loss this exists to fix.
- **The prompt bank gains a `root` label.** Labels were already free
  strings, so root clicks store and locate like any class. This is what
  fixes transfer: a bank whose plant examples are all foliage places
  foliage prompts — thistle2's three located prompts all sat at y 472–549,
  none below the jaws, and its root left every phase. Appearance drift
  (dirt on the root) is coverage, not a threshold: vectors are stored
  individually and scored by best match, so clicking one dirty root adds an
  example rather than diluting the clean one. If no patch out-scores the
  other classes, no root prompt is placed and `root_tracked_below_the_jaws`
  flags the pass — the fallback is one click.
- `mask_covers_its_own_prompts` now checks root prompts against the plant
  mask too (the union makes that the right mask to check).

## 2026-08-26 — P2: a lost root object is re-acquired from the prompt bank

The `[3] root` tracking object (previous entry) was necessary but not
sufficient. Measured on thistle1 re-clicked with root prompts (4 per pass):
the root object went **empty for 42 consecutive frames** in pass 0 (frame 54
to the end -- 157° of arc, including the plier-opposite side where the root
is fully visible) and for 8 frames in pass 1. Coverage 74/96 and 79/96,
under P4a's ~86% floor, so the carve still deleted it: root 9.1% of 2D
pixels, 0.0% of 3D points. The mechanism is SAM2's per-object memory
decaying through the occlusion (the pliers cross in front once per
rotation); whether the object is re-acquired afterwards is luck -- pass 1
got lucky, pass 0 did not. The P2 tracking crop was ruled out: it is
full-height (1080 of 1080) on this footage.

**Decision:** after propagation, maximal runs of frames whose root mask is
*exactly empty* (a binary condition, not a threshold) are scanned with the
prompt bank's root examples (`best_patch`: same margin-against-rivals rule
as prompt placement, restricted to the tracking crop). Each round, every
empty run gets one new conditioning point -- its best not-yet-used
positive-margin frame -- then propagation is re-run and the empty runs are
recomputed. Iteration is not optional: a conditioning frame's repair
reaches only ~5-7 neighbouring frames (measured: pass 1's 8-frame gap
needed one seed; pass 0's 42-frame gap recovered just the seeded frame),
so a long stretch converges by rounds as each recovery shrinks its run.
DINO scores are cached per frame and each round consumes a scored
candidate or stops, so the loop terminates. A run with no positive margin
is left alone: that is what genuine occlusion looks like, and holder
pixels already count as no-evidence downstream. Reseeds are logged to
`p2/prompts.json` (`root_reseeds`) because they were placed by appearance
matching rather than a human.

Validated against the failed frames before wiring: the bank finds a
positive-margin root patch on every probed lost frame (margins 0.21-0.34),
and the hits land on root strands or the dirt-covered root ball -- checked
visually, which also settled the "dirt changes the root's colour" worry:
DINO features match the dirty root ball to dirty-root click examples
without any tuning.

**Trap found on the first live run: the reseed was silently a no-op without
`predictor.add_all_frames_to_correct_as_cond = True`.** SAM2 treats a click
on an already-tracked frame as a *correction* and stores its mask as a
non-conditioning output; the next `propagate_in_video` recomputes that frame
from the same decayed memory and discards the click. Verified the hard way:
the reseed points landed on the root ball (checked visually), yet the
re-propagated masks came back byte-identical -- 74/96 and 79/96, medians
equal to three decimals -- including zero root area on the clicked frames
themselves. The flag is SAM2's own switch for promoting correction clicks
to conditioning frames.

**Trap found on the second live run: appearance cannot tell the plant's
dirt-covered root ball from the pile of dirt lying on the turntable.**
Three of pass 0's nine re-seeds landed on that pile (it out-scored the true
root on DINO margin in those frames) and SAM2 grew it into a 41k-147k px
"root" against the true root's ~9k, blowing the mask-area QC to a 224%
frame-to-frame jump. What separates the two is not appearance but
*structure*: the jaws grip the real root, so its mask touches the holder or
the foliage in the image, and a detached pile touches neither. The guard is
applied at exactly one place -- where the risk enters: a machine-placed
seed is accepted only if the mask SAM2 predicts for the click (returned by
`add_new_points_or_box` before any propagation is spent) has a component
attached to the plant or holder (`_attached_root_binary`); otherwise it is
cleared on the spot and the run's next-best candidate is tried. Human
clicks are never second-guessed, and the write path stays the plain union.
An earlier draft also filtered every frame's root mask by attachment at
write time and used attachment as the loop's presence test -- reverted:
real root slivers whose jaw contact is occluded are detached in 2D, and
that draft re-flagged tracked frames and burned a propagation round per
rejected candidate. Cleaner captures (no loose dirt on the table) remain
the better fix; this guard just keeps one bad frame from poisoning an
unattended batch run.

Two adjacent fixes in the same change:

- **The root's mask cedes pixels the holder also claims**
  (`root & ~holder` at write time). The jaws grip the root, so their shared
  boundary is where SAM2 blurs; this run had 8.4% plant/holder overlap and
  19% colour contamination. Holder pixels are occlusion (not
  anti-evidence) in P4a/P4b, so ceding them costs the root nothing.
**Withdrawn from this entry: moving a rosette's crown to the clamp line.**
It shipped here and was reverted the same day, unvalidated -- see
"2026-08-26 -- REVERTED: locating a rosette's crown at the clamp line".
P5's rosette crown is the betweenness hot set's centre, as before.

## 2026-08-26 — P1: still photos as a capture pass

Stills are now a first-class input (`--photos <dir>`, one directory per
pass, mixable with `--video`). The whole integration is at P1: photos are
copied/resized into `p1/frames/frame_XXXX.jpg` and registered in
`p1/sources.json`, after which no later phase can tell a photo capture from
a video one. Nothing downstream changed.

**Filename order is capture order.** SAM2 propagates frame to frame and P3
fits a circle to the camera centres; both assume consecutive frames are
neighbouring angles. Stills carry no other ordering, and a camera's
sequential numbering supplies it for free.

**No sharpest-frame selection, by necessity.** A video gives P1 ~30 frames
per angular bin to choose from; a photo directory has exactly one shot per
angle. So each photo's sharpness is measured (the same
variance-of-Laplacian P1 uses on video, at a common 1600px scale so numbers
compare across sessions) and outliers are *named* rather than dropped:
culling a soft photo also widens the angular gap `full_rotation_covered`
measures, and that trade is the operator's to make. On
`/mnt/d/Turn_table_plant_scans/test_weed_2`, 80 photos scored a median of
10.3 with one shot at 5.1.

**Photos are resized to a 1920px long edge by default**
(`--photo-max-edge 0` opts out). This matches the video path that every
downstream default was fitted against, and at current settings discards
nothing: SAM2 resizes its input to 1024, P3 caps SIFT at
`--max-image-size` (1920), P4b trains at `--downsample` (2). What 24 MP
frames do change is cost -- 11x the pixels through P4a's carve and P4b's
rasteriser, a VRAM wall rather than a slow run on an 8 GB card.

Intrinsics are safe at either setting, which was checked rather than
assumed: COLMAP records the camera at the image's true dimensions even when
`max_image_size` makes it extract features on a smaller internal copy
(verified with pycolmap on these photos -- 6000x4000 in, 6000x4000 camera),
so full-size frames are never silently mismatched against their full-size
masks.

### Measured on test_weed_2 (80 photos, 24 MP, a different specimen and rig
### session from any video capture)

P2 seeded itself from thistle1's `prompt_bank.npz` -- no clicks -- and
tracked the plant, not the pliers: 0.08% holder contamination, 0.31%
plant/holder overlap, and the overlays show clean plant/tool separation
across the orbit. That is the batch-automation claim holding across capture
*style*, not just across specimens.

Two QC checks fail on this specimen and both are false alarms worth
recording, because they are exactly the sort of thing that must not gate an
unattended batch (and do not -- see the gate entry below):

- `area_temporally_smooth` (44% worst jump): a two-leaf seedling's
  projected area genuinely halves between broad-side (26k px) and edge-on
  (12k px) views. The 15% limit was fitted on larger plants; foreshortening
  dominates on a small flat one. The jumps oscillate rather than stepping
  once, which is the signature of real geometry rather than tracking loss.
- `root_tracked_below_the_jaws` (5/80 frames): this seedling has no exposed
  root -- the jaws grip at the base. The bank correctly placed no root
  prompt (`root=0x`), since no patch out-scored the rivals.

### P3 on those same photos: 22/80 registered, and the cause is exposure

The ingest is sound and P2 is clean, but this photo set does not solve.
Sharpness measured *inside the region P3 actually extracts features from*
(the rotating-region mask, not the whole frame, which is mostly black
backdrop) separates the two groups cleanly:

| frames | median Laplacian variance in the P3 mask |
|---|---|
| registered (22) | 23.1 |
| unregistered (58) | 12.6 |

EXIF says why: **0.5 s at f/10, ISO 100**, shots 4 s apart -- a manual
rotate-and-shoot with a half-second exposure. Anything still settling after
the turntable is turned by hand, or any shake from pressing the shutter,
smears the frame. The registered frames come in arcs (19-29, 44-46, 59-65)
with a 70 deg gap, which is what a few lucky still moments look like.

Two things this is *not*, both checked before blaming exposure:

- **Not misfocus.** The plant is the sharpest region in every frame sampled
  (93-121 against 15-38 for the table), so focus was on the subject. Only
  whole-frame sharpness looks bad, because the plant is 0.8% of the frame.
- **Not the camera moving.** Backdrop corners differ by 1-8 DN across
  photos 0-78, so the rig was static as the method requires. (`DSC_0319`
  differs by 74 DN and is numbered outside the 0222-0316 run -- a stray
  shot after the camera moved, and it should be deleted rather than
  ingested.)

Capture-side fixes, in order of leverage: use the self-timer or a remote so
the shutter press cannot shake the rig, and let the table settle before
firing; raise ISO to 800-1600 and open to f/8 to buy a 1/15-1/30 s exposure
(this sensor is clean there, and SIFT cares far more about smear than about
noise); add light if you have it. Keep f/8-f/11 for depth of field -- the
f/16 frames later in the set trade sharpness for it on a 24 MP APS-C.

Also: the turntable in this set carries the same **plain checkerboard** that
defeated thistle2's solve. It is worth replacing with a random,
non-repeating texture (speckle, torn newsprint) rather than removing
outright -- the plant is a small, smooth, feature-poor subject, so most of
P3's usable features come from whatever the turntable surface carries, and
a non-periodic pattern gives abundant features that cannot alias.

## 2026-08-26 — Bank paths resolve from a workdir; --skip-to judges what it skipped

Two usability failures from one real command, both fixed where they happen
rather than documented around.

**1. `--seed-bank <workdir>` died as `IsADirectoryError` from inside
`numpy.load`, after the DINO weights had already loaded.** Pointing at the
run directory is the natural mistake: it is the path already on the command
line from the run that produced the bank, and both banks live at a fixed
place inside one. `banks.resolve_bank` now accepts either the `.npz` or its
run directory, prints the substitution rather than making it silently (the
wrong specimen's workdir is a mistake worth seeing), and is called *before*
the backbone loads. It also names the specific error for a missing path, a
directory with no bank in it, and -- the quiet one -- being handed the
*other* bank: both are `.npz` holding `vectors`/`labels`, so a prompt bank
passed as `--seed-bank` loads cleanly and then classifies leaves as
"plant".

**2. `--skip-to` skipped the QC gate along with the phases.** The same
command reached P4c on a workdir whose P3 had failed its circle fit
(2.14% deviation, a 130 deg angular gap, 16/27 frames registered) and whose
`p4/` and `p4b/` did not exist at all -- the gate added earlier only fires
for phases the invocation actually runs. Skipping in partway is precisely
when the phases below are being taken on trust, so their QC on disk is now
gated before the first phase executes. The same command now stops in a
second with the circle fit named, instead of after a model load and a pass
over the frames.

Also: `--skip-to`/`--stop-after` given twice now print which value won. The
command that prompted this had `--skip-to p3 --skip-to p4c`, and silently
took p4c -- which is why P4a and P4b never ran.

## 2026-08-26 — Why thistle spines are missing, and why no threshold fixes it

Asked whether the smoothed-out leaf prickles are an inference problem in
P2 or P4c. They are neither: they are a resolution budget, and the numbers
say where it is spent. Measured on thistle1 (`frame_0140`):

| stage | detail it can represent |
|---|---|
| **SAM2 mask cell (P2)** | **4.22 px** |
| P4a hull voxel, projected | 2.82 px |
| DINOv3 patch (P4c) | 16 px -- but see below |
| a thistle spine | ~4-10 px |

**P4c is not involved.** It assigns organ labels to points that already
exist and cannot move or add geometry, so its (much coarser) patches
affect only where "leaf" stops and "stem" starts. `--dino-size` sharpens
that boundary; `--stride` is frame subsampling, not spatial resolution.

**P2 is the bottleneck, and it is architectural.** SAM2's mask decoder
emits 256x256 whatever it is given, and P2 stretches that over the
tracking crop. The crop is 1080 px, so one mask cell covers 4.22 image
pixels -- about one spine. Overlaying the mask contour on the photograph
shows exactly that: the boundary follows the leaf *lobes* correctly and
cuts straight across the bases of the spines.

**The crop is the only real lever, and it is nearly exhausted.** Cell size
is crop/256, and the crop cannot be smaller than the subject, so the gain
available is whatever padding wastes. Tested by re-running P2 at
`--roi-padding 0.12` against 0.45:

| | crop | px/cell | boundary complexity (P^2/4piA) |
|---|---|---|---|
| default 0.45 | 1080, 1080 | 4.22 | 7.12 |
| tight 0.12 | 1080, 977 | 4.22 / 3.82 | 7.47 |

A 5% gain in boundary detail, still no spines, and two holder prompts fell
outside the crop. Pass 0 did not shrink at all -- it is clamped by the
frame height. The default is therefore left alone: the cost is real and
the gain is not.

Note the interaction with root clicks: the crop must contain the root
(`solve_tracking_crop(include_points=...)`), and the root hangs well below
the foliage, so including it enlarges the span and *costs* foliage
resolution. That trade is settled in the root's favour -- a missing root
loses an organ, missing spines lose decoration.

**Even a perfect mask would not carry spines all the way through.** The
carve keeps a voxel only where it is inside the silhouette in 86% of the
views that judged it. A spine is a thin protrusion, sharp from a few
angles and edge-on or occluded from most -- the same geometry that had the
root sitting at 0.56 agreement and being deleted. `hull_vs_mask` diagnostics
confirm the foliage silhouette itself is reproduced well (IoU 0.81 on
frame_0114); essentially all of the 19% the hull misses is the root, not
the leaf margins.

**Conclusion: not worth chasing at the current architecture.** Spines
contribute nothing to what P5/P6 actually measure -- midribs, insertion
angles, arclengths, width profiles. Making them appear would need new
machinery (per-leaf SAM2 crops, or a guided-filter/matting refinement that
snaps the boundary to image gradients), and that means new tunables in a
pipeline that has deliberately removed them. Recorded here so the
measurement is not repeated.

## 2026-08-26 — REVERTED: locating a rosette's crown at the clamp line

Attempted and backed out the same day. Recorded because the *problem* is
real and someone will try this again.

**The problem.** `find_clamp_height` looks for the band of stem the pliers
hid, and occlusion-aware carving retired that signal: a holder pixel now
carries no evidence rather than carving the plant away, so the tissue
behind the jaws is reconstructed and the gap closes. thistle3 and thistle4
both report "no clamp gap", so the rosette crown falls back to the
leaf-attachment centroid -- which sat 28% and 48% of plant height above the
base on those two runs. The crown *should* be where the jaws grip.

**What was tried.** Two pieces: `clamp_height_from_holder`, scoring each
point by the fraction of views in which it projects onto the holder mask
and taking the peak height band; and a rosette crown placed at z=0 with x/y
from the centroid of shoot tissue in a thin band around that height.

**Why it was reverted.** The x/y derivation is wrong by construction. At
the clamp line there is almost no shoot tissue -- that is where the plant
emerges -- so the band collects whatever droops to that height, which on a
spreading rosette is outer leaf tips far off-axis. Their centroid is a
point in open space, and that is what it produced. `shoot` also excludes
root points, so the tissue actually at the clamp was not even in the set.

**What survived the test and is worth keeping if this is retried.** The
holder-score signal itself is clean and needs no threshold: on thistle3 the
mean score decays monotonically from 0.125 in the lowest height band to
0.000 above 40% of plant height, and it located a clamp on both specimens
where the gap test found none (thistle3 z=0.076, thistle4 z=0.307, against
plant heights of 1.21 and 3.49). It is also robust to bad organ labels,
which the root-junction alternative is not -- thistle4's transferred seed
bank put "root" on the plier jaws, which would have placed the crown in
mid-air.

**If retried:** take the crown from the gripped tissue in all three
coordinates -- a score-weighted centroid of the peak band, weighting each
point by its own holder fraction -- rather than from a height band over a
tissue class. And validate on a scratch copy of a workdir: this went out
after a partial run on a live one, which left `stem_graph.json` rewritten
and `p5.json` stale.

**Also still open, and not caused by this change:** thistle4's plant frame
is tilted, because `solve_up_direction` resolves "up" against the table
plane and that specimen is held nearly horizontal. Confirmable by checking
whether the tilt predates any of this on a stashed tree.

## 2026-08-28 — P5 midribs: heart leaves, and three open questions

State at the end of this session, and what is deliberately unresolved.

### What changed

- **Leaf ownership follows the branching, not distance.** `own_by_subtree`
  replaces "nearest tip across the surface". The old rule split two leaves
  at the *midpoint between their tips*, which is the crown only when both
  are the same length -- so the shortest leaf annexed its neighbours'
  stalks. On thistle3 one instance held 29% of all leaf tissue and wrapped
  357 degrees around the crown, against 24-43 degrees for a real leaf.
- **Tip acceptance is cut at the widest gap** in the ranked persistence
  ratios rather than at a fixed 0.5. Measured on thistle3 the real leaves
  ran 1.00 0.96 0.90 0.88 0.73 0.57 0.48 0.37 and the noise began at 0.10:
  the old constant sat *inside* the run of real leaves and discarded two of
  them, one by 0.02.
- **`keep_largest_blob`** releases any part of an instance detached from its
  main body, which is tissue taken from a neighbour across a contact.
  Connectivity is judged after cutting edges longer than 3x the instance's
  own median edge -- the kNN graph joins the ten nearest neighbours however
  far apart, so components taken straight off it never split. Identical
  result from 1.5 to 4.0, stops firing at 4.5.
- **Heart leaves are drawn as the straight crown-to-tip chord**, points not
  consulted, selected by `HEART_LEAF_ELEVATION` (45 degrees of crown-to-tip
  rise). Seen from above the cloud closes over the middle of a rosette
  slightly higher than the central leaves attach, so about half their length
  is not reconstructed and what survives is a one-sided sliver. A station is
  the midpoint of the tissue in it, so lopsided tissue puts every station
  off the vein and the fitted curve waves between them. These leaves are
  short and near-upright, so the chord is the better estimate.

### Open question 1: "steep" is not the same as "poorly supported"

`HEART_LEAF_ELEVATION` classifies on angle alone, which is what the operator
specified, and on thistle3 it catches one leaf it arguably should not:

| leaf | points | elevation | chord supported by its own tissue |
|---|---|---|---|
| 1 | 12,761 | +55 deg | **72%** |
| 4 | 2,123 | +56 deg | 49% |
| 7 | 470 | +80 deg | 48% |

`p5/diag/heart_leaf_check.png` plots both candidate midribs against the
points for exactly these leaves. It shows the distinction clearly: on leaf 1
the fitted curve is a gentle 2-voxel bow that tracks its own tissue and is
mildly *better* than the chord; on leaves 4 and 7 the fitted curve arcs 5-8
voxels through regions that are almost entirely *unassigned* tissue
belonging to other leaves. So requiring **steep and poorly supported** would
be more correct -- support splits 72% against 49%/48%, a clean gap.

Not done because the gain is one 2-voxel bow on one leaf, against a second
constant to re-check per species. Revisit if a specimen turns up with large
upright leaves that visibly need their curvature.

**Resolved 2026-09-02** -- sugarbeet_4 is that specimen. See "Midribs follow
the points unless the points are not there" below.

### Open question 2: tissue with no tip of its own is claimed by a neighbour

`own_by_subtree` hands a dead-end branch to whichever leaf its parent
belongs to. The rule it replaced blocked travel through the base contact so
that such tissue stayed unclaimed and `claim_orphans` could promote it. On a
real plant the junction usually has two leaves beyond it and comes out
shared, and `keep_largest_blob` releases detached remnants -- but a bud with
no tip of its own, still connected, will now join a neighbour rather than
becoming its own instance. That is the failure the deleted
`test_growth_stops_at_the_stem` guarded against, and nothing covers it now.

### Open question 3: `min_leaf_points` is the last size-dependent constant

An absolute point count (150). Everything else in this stage is a ratio, a
rank or a comparison. It should become a fraction of total leaf tissue
before the pipeline meets a much denser or sparser cloud.

### Removed as dead

`grow_from_tips` (superseded by `own_by_subtree`), and `get_camera_data` /
`CameraView` in `reconstruction.py`, which served the Gaussian-splat trainer
deleted earlier. `read_tips` is kept: nothing in-repo calls it, but it is
the reader for `p4c/tips3d.json`, which `pose-tips` still writes.

## 2026-09-02 — P1 records the lens; P3 gives each focal its own camera

sugarbeet_3's three passes were shot at 48/32/22mm. P1 re-encodes when it
resizes, which drops EXIF, so COLMAP had no prior and fell back to guessing
1.2x the long edge -- 2304px -- for all 46 frames alike, against true focals
of 3840/2560/1760px. `--cameras single` then forced one shared focal across a
2.2x zoom range.

P1 now reads `FocalLengthIn35mmFilm` into `p1/intrinsics.json`, scaled to the
size it actually wrote, and P3's new default `--cameras exif` groups frames by
focal and seeds one COLMAP camera per group. With no EXIF the behaviour is
byte-for-byte the old `single`, so video is unaffected.

### The measurement that matters is not the IoU

All three modes on the same 46 frames, `--low-texture`, all 46/46 registered:

| mode | reproj px | pass axes differ | per-pass circle RMS | hull IoU | recall |
|---|---|---|---|---|---|
| single | 1.20 | 5.52 deg | 1.36 / 0.95 / 0.53% | 0.630 | 67.5% |
| per-image | 1.05 | 4.74 deg | **4.05** / 1.55 / 1.40% | 0.740 | 78.4% |
| exif | 1.07 | 5.17 deg | **0.39 / 0.46 / 0.26%** | 0.733 | 77.9% |

`per-image` and `exif` score the same IoU, and that number alone would not
choose between them -- run-to-run variance is about 0.10 (the archived
`single` run scored 0.528 where this one scores 0.630, after P2 was re-run).
The per-pass circle fit does choose: `exif` is 3-10x tighter than either
alternative, while `per-image` *fails* the check it was supposed to help.
Given 46 free focals the solver absorbs real geometry into intrinsics, which
is the failure `single_camera`'s docstring already warned about.

The decisive evidence that `exif` recovers true geometry is that the orbit
radii come back in the ratio of the focal lengths, as they must when the
camera zooms rather than moves:

| ratio | focal | exif radii | single radii |
|---|---|---|---|
| pass 0 : 1 | 1.50 | 1.49 | 1.01 |
| pass 1 : 2 | 1.45 | 1.60 | 1.07 |
| pass 0 : 2 | 2.18 | 2.38 | 1.09 |

`single` put all three orbits at nearly the same radius, which is impossible
across a 2.2x zoom: the focal error had been absorbed into camera placement.

### It is not the whole story

`passes_share_a_rotation_axis` still fails at 5.17 deg (limit 2), in every
mode. Each pass is now an excellent circle in isolation, so the passes are
internally consistent and mutually mis-oriented -- the plant or the rig moved
between passes, which no camera model can undo. Intrinsics were worth about
+0.10 IoU here and are not the remaining gap.

### Regression: the case that already worked

thistle3 is one pass at 36mm x14 and 38mm x13, so `exif` builds two cameras
where `single` built one. It does not regress, and it flips
`cameras_lie_on_a_circle` from fail to pass (2.38% -> 1.93% RMS):

| mode | registered | hull IoU | recall | P4a |
|---|---|---|---|---|
| single | 16/27 | 0.814 | 93.5% | all pass |
| exif | 16/27 | 0.822 | 93.8% | all pass |

### Also fixed here: stale frames were being solved as pass 0

`ingest_photos` never cleared `p1/frames`, so an ingest smaller than the last
one left orphans behind. They get no `sources.json` entry and P3 defaulted
them to pass 0. On sugarbeet_3, 39 orphans from an aborted run turned a
46-frame solve into an 85-frame one spanning two different captures, put 43
views in pass 0, and collapsed the hull to 0.234 IoU -- while every phase
reported success. P1 now clears first and P3 refuses to run when a frame is
not listed rather than guessing.

## 2026-09-02 — COLMAP was never the problem; P1 was letting a second plant in

sugarbeet_4 came back with a sparse cloud, a hull that disagreed with its own
masks, and camera centres visibly off the circle -- with P2 and P4c masking
confirmed good and both passes shot at the same 30mm, so neither the previous
entry's focal fix nor a masking fault could explain it.

### What actually happened

`--photos` had been pointed at `sugarbeet_4/pass1` and `sugarbeet_3/pass2`.
Both basenames print as "pass2" in the log, both plants are sugarbeets on the
same turntable with the same pliers, so nothing in the output looked wrong.
53 frames went in: 40 of one plant and 13 of another shot an hour earlier.

COLMAP behaved *correctly* -- it registered the 40 and refused the 13. It was
the matching that did the damage: the two shoots share a turntable, a pair of
pliers and a backdrop, so the cross-shoot matches are on real repeated
structure and survive geometric verification.

Same 40 frames, same masks, with and without the 13 present:

| input | registered | reproj px | circle RMS | angular gap | hull IoU |
|---|---|---|---|---|---|
| 40 own frames | 40/40 | 0.49 | **0.05%** | 84 deg | -- |
| + 13 foreign frames | 40/53 | 0.99 | **7.01%** | 218 deg | 0.471 |

**On clean input this capture solves essentially perfectly** -- 0.05% of
radius off the fitted circle, 0.34% out of plane. COLMAP is not fragile here
and needed no tuning; it was fed two scenes and asked for one.

### Rejected: the masking hypothesis

P3's COLMAP mask is one temporal-variance region per pass, unioned with the
per-frame P2 masks. That region necessarily includes every backdrop pixel the
plant ever swept across, because a pixel changes both when the plant arrives
and when it leaves. Measured on sugarbeet_4: **46% of the mask is static
backdrop wall**, against 10% plant and holder.

That looked like the culprit and is not. Three maskings over the 40 clean
frames:

| mask | coverage | registered | reproj px | circle RMS |
|---|---|---|---|---|
| current (rotating ∪ plant ∪ holder) | 40.9% | 40/40 | 0.49 | 0.05% |
| per-frame plant ∪ holder only | 9.9% | 40/40 | 0.45 | 0.06% |
| plant ∪ holder ∪ derived disc | 15.3% | 40/40 | 0.46 | 0.06% |

Indistinguishable. RANSAC discards the static-backdrop matches without help,
so the 46% is wasted extraction time, not a fault. Left alone: tightening the
mask would have been a plausible-sounding change that fixed nothing, and the
current mask is the one every downstream default was fitted against.

### Rejected for now: fiducial markers

The turntable carries small tag markers, which would give a per-frame
turntable pose directly and remove SfM from the critical path -- the only
genuinely *guaranteed* method. They are unusable as shot: over six frames,
the best-performing dictionary (APRILTAG_16H5, also the most false-positive
prone) found 7 markers total, roughly one per frame, because the tags are
~2% of frame width and near edge-on at this camera elevation. Viable only
after a rig change: larger tags, and enough camera elevation to see them
face-on. Worth doing if turntable capture continues.

### What was added instead

`p1/manifest.json`, one provenance record per frame -- source directory and
file, EXIF shutter time, camera body, focal -- and `check_capture_consistency`
over it, run when more than one pass is given. It rejects, in descending order
of certainty: a pass shot before its predecessor ended (no threshold needed);
consecutive passes more than `max_pass_gap_minutes` apart (measured: 1.0-1.3
min within a real capture, 57 min for the foreign pass, default limit 30);
two camera bodies; and filenames whose order is not capture order, which the
README has always required and nothing checked. Passes with no EXIF time --
video -- are skipped rather than guessed at. `--allow-mixed-capture` overrides.

Verified against the real datasets: it names the sugarbeet_4 failure exactly,
and stays silent on sugarbeet_4 pass1+pass2, sugarbeet_3 pass0+1+2, and
thistle3.

### Still open on sugarbeet_4

The 84 deg angular gap is in the clean solve too: the 40 photos cover about
276 deg, not 360. That is a capture hole, not a solver fault, and no amount
of pose work will fill it -- a visual hull cannot carve what it never saw.

## 2026-09-02 — The rosette crown: nearest foliage above the root, and no column

sugarbeet_4 put its crown at z=0.973 on a plant whose foliage starts at
z=0.068 and whose root ends at z=-0.072 -- 72% of the way up, in the middle
of the canopy. thistle3, same code, put its crown at z=0.368 against a root
top of 0.284, which is right. The interesting question is not why sugarbeet_4
failed but why thistle3 did not, because the two answers are the same bug.

### Two length scales, both attached to the wrong thing

`crown_from_root` searched for the lowest foliage inside a column of
`voxel * 6` about the root's top centroid, over tissue P5 had attached to a
leaf instance.

- **The column was scaled to the voxel.** Voxel size describes how finely the
  cloud was sampled. Nothing about it says how far a crown may sit from the
  middle of its own taproot. thistle3's crown sits on that axis and sugarbeet_4's
  does not -- its crown tissue is 0.045-0.15 off it, against a column of
  0.045 -- so on sugarbeet_4 the column contained no crown at all and the
  search walked up it until it found something, in the canopy.
- **The foliage set was scaled to the instancing.** `own_by_subtree` leaves
  crown tissue unowned *on purpose*: it has two or more leaves beyond it and
  belongs to none of them. Filtering on ownership therefore discards exactly
  what this function looks for, and does so worse the better the crown is. On
  sugarbeet_4, 35% of blade points unassigned, the lowest attached tissue in
  the column was z=0.973; including everything moved it only to z=0.821,
  which is why the ownership filter alone was not the whole story either.

thistle3 escaped both because its crown happens to sit above the middle of
its root, so a narrow column found it, and because at 15% unassigned enough
attached tissue remained inside that column.

### The fix removes the column rather than widening it

Widening was tried and is not the answer: the column was doing double duty as
the drooping-leaf guard, and a wide one admits a blade passing through the
crown's height out at the rim. Measured against plant extent the two
specimens do share a plateau -- from 10% of extent upward sugarbeet_4 holds
z=0.105 and thistle3 z=0.284 -- but that is a knob tuned to two plants.

Asking for the *nearest* foliage above the root's top carries no radius at
all, assumes nothing about where the shoot emerges, and is the question
`root_anchor` already answers for the shoot path. The height guard is then
the whole drooping-leaf exception: a blade hanging below the root is a leaf
tip, and without the guard thistle3's crown falls onto one at z=0.150.

| specimen | root top | foliage starts | crown before | crown after |
|---|---|---|---|---|
| thistle3 | +0.284 | +0.011 | +0.3682 | +0.3070 |
| sugarbeet_4 | -0.072 | +0.068 | **+0.9731** | **+0.0784** |

Both remaining knobs are inert. Across `top_fraction` from 0.02 to 0.35 -- a
factor of 17 -- neither specimen's crown moves at all: thistle3 holds 0.3070
and sugarbeet_4 holds 0.0784 throughout.

### What it cost, and what it recovered

Instancing is untouched on both: identical tips after merge, instances kept,
and unassigned counts. On sugarbeet_4 the reported leaf count went 4 -> 5,
because `num_leaves` is `len(axes)` and axes are fitted after the crown is
known -- a crown 0.9 above the plant's base left one instance without a
usable axis. The wrong crown was costing a leaf, not just mislabelling a
point.

### Note for anyone re-running P5 on a scratch copy

P5 reads `p2/masks/holder` to decide which way is up, and falls back to the
table plane without it. A scratch workdir holding only p3/p4/p4b/p4c
reconstructs the plant frame *upside down* -- shoot z -1.347..-0.068 instead
of +0.068..+1.347 -- and every height in this entry becomes meaningless. Copy
p2 as well.

## 2026-09-02 — Midribs follow the points unless the points are not there

sugarbeet_4's blades came back as straight sticks. They are upright, large and
fully reconstructed, and `HEART_LEAF_ELEVATION` was drawing every one of them
as the straight crown-to-tip chord with its points never consulted. This
closes open question 1 of the 2026-08-28 entry, which predicted exactly this
specimen: "revisit if a specimen turns up with large upright leaves that
visibly need their curvature".

### Steepness was standing in for a question it cannot answer

The chord exists for the small upright leaves at the centre of a rosette. Seen
from above the cloud closes over the middle of the plant slightly higher than
those leaves attach, so about half their length is never reconstructed and
what survives is a one-sided sliver. A station centred on lopsided tissue sits
off the vein, so a fitted curve waves between stations; the straight line is
better than a curve bent toward whichever side happened to be reconstructed.

That is a statement about **reconstruction**, and elevation measures **pose**.
On a rosette the two coincide, which is why the shortcut held. On sugarbeet_4
they come apart completely: its steep blades carry 8k-62k points and cover
their whole chord.

`chord_station_support` asks the question directly -- of the 14 stations
between base and tip, how many hold at least 3 of this leaf's own points --
and the two specimens separate cleanly:

| specimen | steep leaves, by chord coverage |
|---|---|
| thistle3 | 0.79, 0.57, 0.50 |
| sugarbeet_4 | 0.93, 1.00 |

`HEART_LEAF_SUPPORT = 0.85` sits in the middle of that gap. 0.90 was tried
first and is too close to the edge: sugarbeet_4's largest blade, 62,098 points
carrying a 1.20 arc-over-chord, measures 0.93 and would be re-flattened by a
three-point drift.

### What it changes

sugarbeet_4, default rule, before and after:

| leaf | points | elevation | coverage | before | after | arc/chord after |
|---|---|---|---|---|---|---|
| 0 | 62,098 | 68.0 deg | 0.93 | chord | **points** | **1.196** |
| 1 | 18,038 | 59.8 deg | 1.00 | chord | **points** | 1.014 |
| 2 | 16,757 | 23.3 deg | 0.93 | points | points | 1.010 |
| 3 | 2,493 | 38.8 deg | 0.93 | points | points | 1.178 |
| 4 | 1,341 | 39.1 deg | 0.86 | points | points | 1.013 |

thistle3 is untouched: the same three leaves keep the chord and the same five
follow their points.

### `--strict-midribs`, and why it is not the default

The flag refuses the chord outright. It is the right tool for a specimen whose
leaves are genuinely upright and well reconstructed, and it is *not* safe in
general -- thistle3 shows what it costs:

| leaf | points | coverage | default arc/chord | strict arc/chord |
|---|---|---|---|---|
| 1 | 12,761 | 0.79 | 1.000 | 1.013 |
| 4 | 2,123 | 0.57 | 1.000 | 1.261 |
| 7 | 470 | 0.50 | 1.000 | **3.575** |

A 470-point sliver fitted from its own points produces a midrib three and a
half times longer than the distance it spans. That is the failure the chord
was introduced for, still there, and still worth defaulting away from.

On sugarbeet_4 strict changes nothing at all -- no leaf qualifies for the
chord once coverage is consulted -- which is the sign the default is doing the
work rather than the flag.

`p5.json` now carries `midrib_support`: elevation, coverage and which
construction each leaf got, so a leaf reading "chord" with high coverage is
visible without re-deriving anything.

## 2026-09-09 — P3 reports every scene COLMAP built, and retries the leftovers

`build_sparse_reconstruction` ran `incremental_mapping`, took
`max(reconstructions.values(), key=num_reg_images)` and returned it. Every
other scene COLMAP built was discarded without a word.

**Why that is a silent disaster and not a tidy-up.** COLMAP does not fail when
a group of images cannot be tied to the rest -- it starts a *second* scene and
carries on, returning both. So a capture pass whose views do not overlap the
others is not rejected, it is *relocated*, into a scene that was then dropped
on the floor. The only outward sign was a registered count, which reads exactly
like a few soft frames. `most_frames_registered` would report "96/126" for a
run in which 30 perfectly good frames built a complete, internally consistent
scene of their own.

This is the precise shape a handheld top-down pass fails in, which is what
prompted it: the motorised rig cannot be raised high enough to look into the
rosette crown, so the top-down views have to be shot by hand, and whether they
join the orbit is exactly the question that was going unanswered.

### Three layers, and only one of them is new machinery

**Layer 1 -- say what happened.** `describe_connectivity` reports which capture
pass landed in which scene, which scenes were built and dropped, and which
images reached no scene at all. `log_connectivity` prints it during the run.

The acceptance check `all_passes_joined_one_scene` fails when a pass
contributed **zero** images to the winning scene, and is fatal in
`qc_gate.py` -- those views are absent from every phase below, so continuing
spends an hour of carving on evidence that was thrown away.

Deliberately *not* fatal: a stray disconnected clump on a single-pass capture.
The main scene can be complete without it, and gating there would stop good
unattended runs -- the trap this project's QC gate exists to avoid. It is
reported in the check detail instead.

**No threshold anywhere in this check.** "Did pass 1 reach the winning scene"
is a fact about the capture, not a measurement of the plant, so it reads the
same on a thistle seedling and a sugarbeet. It is currently the only check in
P3 of which that is true.

**Layer 2 -- give the leftovers a fairer second attempt.** COLMAP allows each
image three tries to join (`mapper.max_reg_trials = 3`), spent whenever the
mapper happens to reach it. An image is therefore judged against whatever the
scene contained at that moment -- so frames of a viewpoint the scene has barely
covered yet can exhaust their attempts against a half-built scene and never be
reconsidered once it has grown enough to accept them. That is an ordering
accident, not a wrong setting.

`incremental_mapping` accepts an `input_path` (verified against the installed
pycolmap 4.1.1 signature), which loads a finished scene and continues from it
with a fresh mapper and therefore fresh attempt counters. So the finished scene
is handed back and the leftovers try again against the complete thing, with
`multiple_models=False` so it extends rather than wandering off to start
another scene.

**Nothing is loosened.** No threshold moves; the images get their attempt under
the best available conditions instead of the accidental ones. Whether the
mechanism pays is *measured and printed* ("recovered N image(s): 96 -> 126"),
not asserted -- a run where it recovers nothing is worth seeing, and says the
frames genuinely do not overlap.

**Layer 3 -- when the solver is out of moves, name the fix.** Matching here is
exhaustive: every image pair has already been compared, so there is no "try
harder" left in the code. `capture_guidance` prints what actually remains --
`--low-texture` for viewpoint-robust descriptors (offered only when it is off),
and shooting the joining pass as a continuous climb in elevation from the
height of the pass it must join rather than as a separate cluster at the top.
It also names the best bridging surface: the holder (rigid, three-dimensional,
non-repeating), over the disc (foreshortens sharply between a low and a high
camera, and the checkerboard aliases -- the thistle2 failure) and the plant
(smooth, self-occluding, slightly mobile).

Advice rather than another fitted parameter, because both remaining levers are
specimen-independent.

### State

`p3/solve.json` carries the whole report, so `--reuse-sparse` re-reads it
without re-solving and a finished run keeps the evidence. An older workdir has
no such file and the check is *absent* rather than guessed at.

Covered by `tests/pose_estimator/test_scene_connectivity.py` (9 tests),
including the JSON round-trip, which matters because `json` turns the integer
pass keys into strings and the check must still name the right pass afterwards.

**Not yet validated on real frames.** The split-scene path is exercised against
constructed models; layer 2's premise -- that reloading via `input_path` resets
the per-image attempt counters -- follows from COLMAP building a fresh mapper,
but has not been measured here. The first handheld top-down capture is the test,
and the printed before/after count is what settles it.
