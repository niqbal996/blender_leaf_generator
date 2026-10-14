# The apex plan — reconstructing the young leaves at a shoot tip

Started 2026-10-12. Plant: vogelmeere
(`/netscratch/naeem/blender_assets/06-10-2026-Naeem/vogelmeere/plant`).
Every experiment writes into `<plant>/debug/<experiment>/`, never into the repo.

---

## 0. Start here — handoff (updated 2026-10-13, evening)

Sections 1–6 below are the log, including dead ends and numbers that were
later corrected. **This section is the current truth.**
`P=/netscratch/naeem/blender_assets/06-10-2026-Naeem/vogelmeere/plant`, `D=$P/debug`.
Sessions run in replaceable containers: read the repo's `CLAUDE.md` first (the
`pose` env and Claude's memory live on a tmpfs; how to restore both).

### 0.0 Where things stand (2026-10-13)

**The user's look at v2** (`$D/apex_fit`, yesterday's last version) in Blender:
1. counts on the plant (`$P/p1/frames/apex_counts.json`): A 7, B 6, C 7, D 4
   leaves, plus at almost every apex one tiny leaf at the very centre (a bud,
   not counted); may not be complete;
2. blades close, but their curvature is not towards the centre: they spiral
   away from it; midribs better, not there yet;
3. a leaf behind an apex (not part of it) can look like part of it in some
   views and confuse the fit — should all leaves be in the fit?;
4. how is the fit started, can the starting template be better? (answered in
   the chat of 2026-10-13; the template is now §0.1's);
5. Blender: clouds hide the skeleton; collections need collapsing by hand;
   use the P4g-based cloud.

**Measured on v2** (`scripts/apex_fit_check.py`, same scorer for every
version): midribs turned 62° median from base to tip (up to 138°), 10 of 26
towards the shoot axis, the rest away; a sideways bend out of the leaf's own
plane of 0.46 (median change of the unit tangent) — the spiral; rolls from
−238° to +200°, twists to 209°: blades upside down, so their fold cupped
outwards. Nothing in v2's leaf knew where the shoot was or which face is up.

**Built today: v3** (`scripts/apex_fit.py`, design in §0.1) — every apex leaf
lives in its shoot's frame: the apex has a tip and a shoot axis (both fitted;
the axis starts along the P5x branch that ends at the apex); a leaf is placed
by where it joins the axis, its azimuth, its elevation and how far it curls
in; its midrib stays in the plane through the axis, turns only towards it,
its upper face looks at the shoot, its blade is not twisted. A bud at each
tip (structure: never a false positive, no longer than the shortest leaf).
All other P5x leaves are instances in the render, so a leaf behind the apex
claims its own SAM3 mask (point 3). Each fitted leaf proposes its opposite
partner (§3.4); partners are judged one by one before the other candidates.

**Numbers, v2 → v3** (all 74 frames unless said; `$D/apex_fit/recheck/` and
`$D/apex_fit_v3/leaf_support.txt`, same scorer, context leaves in both):

| | v2 | **v3** |
|---|---|---|
| leaves A / B / C / D (user: 7 / 6 / 7 / 4) | 9 / 6 / 7 / 4 | 9 / **4** / **6** / 4, + a bud at each |
| PQ A / B / C / D | 0.40 / 0.40 / 0.29 / 0.36 | 0.41 / 0.41 / 0.37 / 0.35 |
| held out (even frames fit, odd scored) PQ | (old scorer) | 0.40 / 0.41 / 0.36 / 0.37 |
| midrib turn, median | 62° | 32° |
| leaves curling towards the shoot | 10 of 26 | 21 of 23 |
| sideways bend (spiral), median | 0.46 | 0.02 |
| user's lines found (`apex_labels.py --score`) | 52 % | **63 %** (selection 58 %, s2d fair zoom 67 %) |
| tip vs user's lines: across / along / distance | 6.2 / −0.6 / 9.1 px | 5.3 / +0.7 / 9.3 px |

Variant `$D/apex_fit_v3_recurve` (`--recurve`: midribs may also bend away,
to 90°): same counts, PQ 0.42 / 0.41 / 0.37 / 0.37 (held out 0.41 / 0.41 /
0.35 / 0.40), 17 of 23 curl in, lines found 53 %, tips across 5.8 px — no
gain, and the outward bends the user objected to come back. Not used.

**Open, in this order** (each: structural rule, no tuned weight; judge with
`apex_fit_check.py` incl. the counts, `apex_labels.py --score`, the held-out
run, then a bundle):
1. **Counts: B 4 of 6, C 6 of 7, A 9 of 7.** B's four are its two outer pairs;
   the missing two are the young inner pair, turned 90° to the outer one,
   smaller and more upright. A partner proposal (same leaf, other side) cannot
   make them; every one at B lowered J. Next: propose the **next pair up** of
   each fitted pair (azimuth +90°, nearer the tip; size and elevation from the
   pair below — measure the ratio on apices that show two pairs, do not tune
   it). At A, one partner landed 2.3 mm from leaf A_5 (a near-upright leaf's
   partner is its neighbour): a twin; check A_5 / A_8 in Blender.
2. **Growing stops at the first rejected candidate.** Partners are now judged
   one by one; the evidence candidates still stop at the first that does not
   raise J (always a geometry one so far). Cheap to judge the top few.
3. **Midline term** (yesterday's step 1): each matched leaf's visible midrib on
   its mask's 2D centre line (`$D/s2d_curves_zoom_fair/evidence_2d.json`).
   Tips across went 6.2 → 5.3 px from the shape rules alone.
4. Fit C and D together (their squares overlap).

**How to run** (GPU H100; after a container restart see `CLAUDE.md`: rebuild
`pose`, gsplat compiles ~5 min on first use):
- fit: `$D/apex_fit_v3/run.sh` (all frames, the deliverable) and
  `$D/apex_fit_v3_heldout/run.sh` (even frames, scores the odd ones); 3–6 min
  per apex, ~17 min each, both can run at once (the candidate pool is CPU);
- check: `PYTHONPATH=src python scripts/apex_fit_check.py --workdir $P --skeleton $D/apex_fit_v3`
  (counts vs the user's, shape about the shoot axis, PQ, per-leaf support, tips);
- labels: `PYTHONPATH=src python scripts/apex_labels.py --workdir $P --out $D/apex_labels --score $P/p5x $D/apex_select $D/s2d_curves_zoom_fair $D/apex_fit $D/apex_fit_v3 --annotate $D/apex_compare`
  (re-carves `apex_regions.npz`: identical, checked 2026-10-13);
- axis start, drawn: `$D/apex_axes/axis_look.py` (header says how);
- bundle: `$D/blender_bundle/make_bundle.sh` (dated tar.gz + VERSION.txt);
  delete the superseded one, unpack the new one and check what shipped
  (`$D/blender_bundle/viewer_test.py` builds the scene headless with Blender's
  `bpy` module; header says how to get it);
- python is `/root/miniconda3/envs/pose/bin/python`.

**Viewing** (bundle `$D/blender_bundle/vogelmeere_view_core_20261013_1631.tar.gz`):
`./scripts/view_in_blender.sh vogelmeere/plant --p5x p5x_p4g --skeleton2d vogelmeere/plant/debug/apex_compare/p5x,vogelmeere/plant/debug/apex_compare/apex_select,vogelmeere/plant/debug/apex_compare/apex_fit,vogelmeere/plant/debug/apex_compare/apex_fit_v3`
— base scene: P5x's leaf labels on the P4g cloud (`--p5x p5x_p4g`; plain
`--p5x` = the hull-based one), drawn as small translucent points with every
skeleton in front of it (`--cloud-alpha`, `--cloud-size`, `--no-in-front`);
each method one collection (`s2d_apex_fit_v3` > `_blades`, `_midribs`,
`_branches` (v3: each apex's fitted shoot), `_gt_found`, …), the outliner
collapsed to one line per method once Blender's window is up (a timer; not
testable headless — if it fails, the console says so). Buds: blades named
`…_fit_apex_<X>_bud`.

**Folder map of `$D`** (2026-10-13):

| folder | what |
|---|---|
| `apex_labels/` | apex regions carved from the user's boxes, `labels.jpg`, `scores.json` + `score_<method>.jpg` (the yardstick) |
| `apex_zoom_fair/` | zoomed SAM3 masks per apex (`sam3/`), `masks_zoom.npz` |
| `s2d_curves_baseline/`, `s2d_curves_zoom_fair/` | s2d one-curve on P2 / on fair zoomed masks (`evidence_2d.json` feeds the geometry candidates) |
| `apex_select/`, `apex_select_p4g/` | steps 1–2; `apex_select` is the fit's start |
| `apex_fit_v3/` | **current result**: `skeleton.json` (+ `apices`: tip and axis per apex), `fit.json`, `apex_<X>.jpg` (photo / SAM3 / start / fitted; grey = context leaves), `leaf_support.txt`, `run.sh`, `apex_fit_v3_code.py` (the code as run) |
| `apex_fit_v3_heldout/` | same, fitted on even frames; `fit.json` has the held-out scores |
| `apex_fit_v3_recurve{,_heldout}/` | the `--recurve` variant (not used, numbers above) |
| `apex_fit/`, `apex_fit_heldout/` | v2 (yesterday); `apex_fit/recheck/leaf_support.txt` = v2 in today's scorer; `apex_fit/apex_fit_v2.py` the code as it was |
| `apex_fit_prev/` | v1 (split and ghost leaves), for the record |
| `apex_axes/` | the axis start drawn per apex (`axis_look.py`, `.jpg`) |
| `apex_compare/` | label-annotated copies of each method's skeleton (what the bundle ships) |
| `blender_bundle/` | `make_bundle.sh`, `viewer_test.py`, the latest core bundle, `vogelmeere_view_clouds.tar.gz` (P4 clouds, unchanged) |

**Code state** (nothing committed since `812bee9`): `scripts/apex_fit.py` (v3),
`scripts/apex_fit_check.py` (v3 scorer), `apex_select.py` (candidate pool as
a function), `blender_view_plant.py` + `view_in_blender.sh` (nested and
collapsed collections, translucent clouds, skeleton in front, `--p5x
<folder>`), `CLAUDE.md` (new), this doc.

### 0.0b The whole pipeline on every plant (2026-10-13, evening)

The user's ask: one runnable pipeline, run on all five plants of the session
(gf_1, vogelmeere upright; sugarbeet_x, thistle_x, weed_x rosettes), with no
output built by stale code.

**What `./run_pipeline.sh <dataset>` now runs:** P1 → P6, then
`scale p4m p4g p5x s2d apex` (header of `run_pipeline.sh`). `apex` is
`scripts/run_apex.sh`: `apex_regions.py` → `apex_zoom.py` → `skeleton_2d.py`
on the zoomed masks → `apex_select.py` → `apex_fit.py` (+ held-out with
`--apex-heldout`) → `apex_fit_check.py` → `apex_labels.py` (where hand labels
exist) → `make_bundle.sh`. Everything lands in `<workdir>/apex/`; the bundle
in `<workdir>/apex/bundle/<specimen>_view_<stamp>.tar.gz`.

**Stale code, guarded** (`scripts/phase_stamp.py`): each phase stamps the
content hash of every repo file it imports (followed through imports), its
upstream stamps and its settings; a phase is kept only when all match. Legacy
outputs are adopted only if newer than every source file they import and than
their inputs. On 2026-10-13 that rule reruns **everything on all five plants**:
`ply_io.py` (imported by every phase) changed 10-09, after every P1-P4 run of
10-07 -- and every P2 ran without SAM3's clean-up (kernels missing until
10-12), which a fresh P2 now has. `--dry-run` prints the plan per phase.

**Apex regions without hand labels** (`apex_regions.py`): an apex is a shoot's
end -- every stem/branch end of P5x's tree, or a rosette's crown; its region,
the points within one median P5x leaf length of the end and nearer it than
any other end; no P5x leaf mostly inside → no apex. Against vogelmeere's hand
regions: the four ends are the four apices; IoU 0.85 / 0.74 / 0.55 / 0.25 for
A / B / C / D. (Points *beyond* each end held only 0-32 % of the hand
regions.) Rosettes: one apex at the crown, axis = plant up, no stem to join
(leaves start at the crown). Pair proposals only for opposite-leaved species
(`plant_profiles.phyllotaxis`; vogelmeere only).

**Configs:** `/netscratch/naeem/blender_assets/pipeline.conf` (root: sam3,
use_gpu, `architecture = upright` for all!), the session's (`segment_backend`,
`hf_home`), and one per specimen with its `architecture` (rosette for
sugarbeet_x, thistle_x, weed_x; caulescent for gf_1, vogelmeere) -- the old
rosette runs had it typed by hand, and the repo's `pipeline.conf` says rosette.

**Measured costs per plant** (H100, 26 cores): P1-P6 6.5 min; P5x on P4b and
P4g ~20 min; s2d 3 min; apex: zoom 10 min, s2d 4 min, select ~5, fit ~17.
P4m (from the 10-09 logs and a timed sample): 40 s per view and pass, 144 of
them, ~2.5-3 h in all with fusion -- the GPU is busy all of it (100 %, 4.5 GB):
two PatchMatch workers on it (`--gpu-index 0,0`) took 80 s per view each, no
gain. P4g 40-52 min. **~4.5 h per plant, ~22 h for five, one by one.**
A P4m killed mid-kernel hung in uninterruptible exit (`exit_mm`) for > 30 min,
and `ps`/`pgrep` blocked on it.

**Before the campaign:** `vogelmeere/plant_baseline_2026-10-13/` is a frozen
copy of vogelmeere's workdir (no `p4m/dense`): the inputs of `$D/apex_fit_v3`,
which the campaign's fresh vogelmeere run will not reproduce byte for byte.
Tests: `_campaign/test/` (fresh gf_1 P1-P6 in 6.5 min; a re-run keeps all
seven by stamp, `--rerun-from p5` forces from P5: passed);
`_campaign/regress/vogelmeere/` (P5x → apex through the pipeline on
vogelmeere's 10-07 geometry and hand labels, fresh P5x/s2d with the current
code): **reproduces v3** -- leaves 9 / 4 / 6 / 4 (v3 the same), PQ 0.410 /
0.408 / 0.373 / 0.352 (v3 0.412 / 0.410 / 0.374 / 0.348), 22 of 23 curl in,
sideways 0.02, tips across 5.1 px, the user's lines found 62 % (v3 63 %).
Apex stage 22 min for 4 apices (zoom 10, s2d 4, select 1.5, fit 6.5).
The whole apex phase again through `run_pipeline.sh --skip-to apex` (end to
end, stamp written, bundle built): exit 0 -- but leaves 12 / 4 / 7 / 3, and
the fit repeated on *those same inputs* gave 9 / 4 / 6 / 4 again. **The leaf
count is not reproducible run to run** (A ±3): gsplat's backward pass is not
deterministic, J moves by ~0.01 per view, and "add a leaf if J rises at all"
took three leaves at A for +0.006-0.010 each -- less than one view's true
positive. Shape and tips do not move (turn 33-36°, 22-25 of 23-26 curl in,
sideways 0.01-0.02, tips across 5.1 px). **Fix first** (next session): judge
an added or dropped leaf on held-out frames (fit on even, J on odd -- §3.8's
own rule), or demand it be a true positive in more views than it is a false
positive; and seed/determinism where gsplat allows.

### 0.1 v3 as built (2026-10-13) — `scripts/apex_fit.py` → `$D/apex_fit_v3/`

Same renderer, masks, on-plant and away terms, one-to-one instance scores and
add/drop by J as v2 (§0.2). What changed:

| part | v3 | why |
|---|---|---|
| apex | tip O (starts at the selection's node) and shoot axis a, both fitted, shared by its leaves. a starts along the P5x stem or branch that ends nearest the node (its last stretch, over the apex's reach); else the leaves' cone axis | an apex is a shoot's end. The tree apex_select hands on joins nodes sideways (47–90° off at B, D); the cone of the leaves was 61–82° off at A, D (leaves at 3–80° to the axis) — `$D/apex_axes/axis_look.jpg`. The fit turned the axes a further 22–32° |
| leaf | 12 numbers: attach below O (a share of the apex's reach — clamped at 0, as in v2, a leaf touching the tip stuck there for good), azimuth, elevation, incurve 0..180° towards the axis, two Bézier legs; blade start, width, two tapers; one roll within ±90°; fold | the midrib lies in its median plane (the plane holding the axis) and curls only in; the upper face looks at the shoot, so the fold cups inwards; no twist. Removes v2's spirals by construction |
| start | each selected curve read in that frame (its out-of-plane part dropped; incurve from its first and last quarters, at least 10° — a start value), roll 0, fold 4° | the template |
| bud | one more element at O, along the axis, closed, half the shortest leaf; never a false positive, never dropped, no longer than the shortest leaf (`bud` term, unitless like `away`) | the user sees one at almost every apex; SAM3 seldom masks it apart. Judged as a leaf, J dropped it where it is plainly there; never a false positive and unbounded, it grew into an 11 mm leaf on a leaf's mask |
| context | every P5x leaf with most of its points outside the apex's carved region is its own instance (its P5x points): it can claim a SAM3 mask, never a false positive. P5x leaves mostly inside: dropped (the fit replaces them); stem, unassigned: one occluder | the user's point 3: a leaf behind the apex is no longer a target an apex leaf has to cover. 33–44 context leaves per apex; they claim 7–51 masks |
| growing | after the first fit, each leaf's partner (same leaf, turned 180° about the axis) is tried one by one, best first, kept if J rises; then apex_select's candidates as before | §3.4; ranked among all candidates, a partner was never reached (a geometry candidate came first, failed, and ended the growing) |

### 0.2 v2 as built (2026-10-12, night) — `$D/apex_fit/` (kept for the record)

**Design.** Per apex, every leaf is fitted at once to all 74 photos, starting
from the selection (`$D/apex_select`).

| part | choice | why |
|---|---|---|
| leaf | 14 numbers: base sliding along its shoot inside the apex region (1); midrib = quadratic Bézier from the base (6); blade start (1); width at the widest, ≤ ½ blade length (1); outline tᵃ(1−t)ᵇ, base taper a and tip taper b (2); roll at the base, 0 = level (1); twist to the tip (1); fold, 0 flat … 90° closed (1); stalk one voxel wide | the numbers the asset step needs (§3.2). Quadratic = one arc bending one way: a cubic hooked its tips inside the narrow blade end, where the masks hardly see it |
| renderer | gsplat 3D Gaussians, flat: 40 × 11 per leaf, one at the centre of each blade cell, covariance from the grid's own steps (κ = 0.6 of a step, so neighbours close into one surface), antialiased; one colour channel per leaf; the cloud outside the apex region as a fixed occluder channel; pinhole cameras on an undistorted 128 px grid of each apex square, the masks resampled onto it through SIMPLE_RADIAL | depth order and occlusion for free, differentiable, ~25 ms for 74 views |
| masks | per view, leaves and zoomed SAM3 masks matched **one to one** (Hungarian on IoU, every step, no gradient); scored as instances, the panoptic-quality way: a pair above IoU 0.5 is a true positive, a visible leaf in none a false positive, a mask in none a false negative; each pair earns its IoU, each FP and FN costs ½ (the F1 weights); masks reaching the square's edge are leaves passing through (apex_zoom's rule) and no target | no ids across views; two blades on one leaf leave one a false positive, and a blade where SAM3 sees no apex leaf pays in every view it shows in. (The first version let two leaves share a mask and charged nothing for blades on no mask — see below) |
| on plant | every blade point must project inside the P2 plant mask: distance outside, in mm at the leaf, over the leaf's width | sets depth in the side views; a leaf off by its own width is as wrong as one missing its mask — no weight to tune |
| away | distance from the base never decreases along the midrib | "leads away from where it joins" |
| objective | J = Σ views [IoU of the pairs − (FP + FN)/2 − on-plant cost] / views − away | dimensionless; one leaf ≈ one unit |
| leaf count | fit the selection's leaves → add the best-ranked candidate (apex_select's pool: geometry matches + P5x, P5x-on-P4g, both s2d one-curve), re-fit, keep it only if J went up; the first that does not ends the growing → drop any leaf whose removal does not lower J → final fit | §3.8 with the same J; nothing tuned. A raw candidate rarely reaches a true positive before it is fitted, so it is judged fitted |
| output | skeleton.json in the P5x schema (`curve`, `blade_start`, one curve shoot → tip) + per leaf `blade` {length, width, widest_at, tapers, roll, twist, fold} and `blade_mesh` (40 × 11 surface) | what the PBR assets snap to; the viewer draws `blade_mesh` as a surface (`<tag>_blades`) |
| evaluation | (1) held out: fit on even frames, score on odd (`$D/apex_fit_heldout`); (2) your 105 lines (`apex_labels.py --score`); (3) Blender | label-free + labelled + by eye |

**Results** (`$D/apex_fit/run.sh`, ~1–2 min per apex on the H100).

*First version (one evening, kept as `$D/apex_fit_prev`): too many leaves.*
Your Blender look: two blades make one real leaf, and ghost leaves (e.g.
`fit_apex_C_4`, `C_7`) where the plant has none; 54 apex leaves on a plant of
30–35 leaves in all. Measured per leaf over the 74 views
(`leaf_support.txt`): leaves shared their SAM3 mask with another leaf in 56 %
of the views they showed in and were alone on one in only 19 %; 9 leaves lay
on no mask in most views (C_4: 47 of 66, C_7: 54 of 64). Cause: the
objective, not the training — J had stopped moving (< 1 % over the last 200
steps). It let two leaves share a mask (meant for SAM3 masks over two young
leaves; the fit used it everywhere, since two halves cover a mask better than
one blade), and blades on no apex mask -- e.g. over the big leaves, whose
masks are no target -- cost nothing. Fixed by the one-to-one instance scores
above.

*Now* (`$D/apex_fit`, all frames): 26 apex leaves (A 9, B 6, C 7, D 4) + 10
others = 36 leaves; no two fitted leaves within half a blade width of each
other (was 18 pairs); leaves alone on their mask in 71 % of the views they
show in (was 19 %), sharing in 5 % (was 56 %); 2 leaves still on no mask in
most views (A_8, B_4 — check in Blender). Same instance scores on all 74
frames, first version → now: PQ A 0.23 → 0.37, B 0.17 → 0.40, C 0.19 → 0.29,
D 0.26 → 0.34, with false positives 585/457/801/256 → 200/147/235/92.

Held out — fitted on the 37 even frames, scored on the 37 odd ones it never
saw (start = the selection's leaves in the same renderer):

| apex | PQ | F1 | SQ (IoU of matches) | SAM3 mask pixels covered | leaves |
|---|---|---|---|---|---|
| A | 0.16 → 0.37 | 0.26 → 0.55 | 0.60 → 0.67 | 0.44 → 0.60 | 9 → 10 |
| B | 0.12 → 0.40 | 0.21 → 0.58 | 0.58 → 0.70 | 0.37 → 0.65 | 5 → 6 |
| C | 0.10 → 0.29 | 0.16 → 0.45 | 0.60 → 0.66 | 0.46 → 0.65 | 9 → 7 |
| D | 0.14 → 0.33 | 0.22 → 0.51 | 0.62 → 0.65 | 0.33 → 0.46 | 3 → 3 |

Your lines (`$D/apex_labels/scores.json`):

| method | found | tip | tip along your line / across |
|---|---|---|---|
| P5x | 34 % | 6.9 px | |
| selection (start) | 58 % | 5.7 px | +0.5 / 2.6 px |
| s2d one-curve, fair zoom | **67 %** | 5.7 px | +1.5 / 2.8 px |
| fit, first version (54 leaves) | 62 % | 6.2 px | −1.0 / 4.2 px |
| **fit** (`apex_fit`, 26 leaves) | 52 % | 9.1 px | −0.6 / 6.2 px |

What the numbers and pictures say:
- the blades lie on the leaves (`$D/apex_fit/apex_<X>.jpg`, third vs fourth
  column), one blade per leaf, and that holds on frames the fit never saw;
- two earlier fixes, both to the model: tips were 4–6 px short of yours —
  (a) outlines allowed to go square-ended (taper bound now 0.5 = rounded),
  (b) Gaussians on the blade's edge drew it half a Gaussian too big, so the
  fit shrank it (now one Gaussian per cell centre). Tips land on yours along
  the leaf;
- **open — midribs:** against your lines the clean fit is below the
  selection (52 % vs 58 %, tips 6.2 px off *sideways*). The outline is
  matched; where the midrib runs inside it is not pinned by any term. The
  selection's midribs come from SAM3's 2D centre lines, which is what you
  draw. Next: a term pulling each matched leaf's visible midrib onto its
  mask's 2D centre line (s2d's `evidence_2d.json`, identity-free like the
  masks);
- **open — small inner leaves:** C's bud leaves have masks but no leaf (no
  candidate fits them). The opposite-pair prior (§3.4) would propose them
  from the node layout instead of from the candidate pool.

**Viewing** (bundle `$D/blender_bundle/vogelmeere_view_core_<stamp>.tar.gz`,
built by `make_bundle.sh` beside it; VERSION.txt lists md5 + build stamp):
`./scripts/view_in_blender.sh vogelmeere/plant --p5x --skeleton2d vogelmeere/plant/debug/apex_compare/p5x,vogelmeere/plant/debug/apex_compare/apex_select,vogelmeere/plant/debug/apex_compare/apex_fit_prev,vogelmeere/plant/debug/apex_compare/apex_fit`
— the fitted surfaces are the `s2d_apex_fit_blades` collection (`s2d_apex_fit_prev_blades`: the first version).

**Clean-up, same night:** removed from `$D` the step-0 runs (`apex_zoom`,
`apex_zoom_clean`, `apex_zoom_probe`), the matching/lifting dead ends
(`apex_match*`, `apex_lift_p2ev`), the pre-fix sweep (`apex_select_sweep`),
`apex_look`, `stale_backup`, the old s2d without leaf curves
(`apex_zoom_fair/s2d_zoom_fair`, its evidence is byte-identical to
`s2d_curves_zoom_fair`'s), stale score pictures. `apex_select*` were re-run
with only current candidate sources (same 26 leaves but one at A). The
plant's root `skeleton.json`, `evidence_2d.json`, `scores.json`,
`reprojection.jpg`, `p5x_vs_2d.jpg` (s2d of Oct 8, before the one-curve
rules) were replaced by the current s2d run (`$D/s2d_curves_baseline`).
`p5x/` was checked against a fresh retrace: identical. Sections 1–5 below
still name some of the removed folders; they are the record, not paths to use.

### 0.3 Pieces, rules and pitfalls (still valid)

**What exists and works (reuse it):**

| piece | where | what it gives |
|---|---|---|
| apex regions from hand boxes | `scripts/apex_labels.py` → `$D/apex_labels/apex_regions.npz` | 4 apices (A–D), cloud points carved by boxes in 5–6 views, radius 9–15 mm |
| zoomed SAM3 on those regions, clean-up on | `scripts/apex_zoom.py --regions` → `$D/apex_zoom_fair/sam3/apex<X>_pass<p>.npz` | per frame per apex: leaf masks at 1008 px of the square (`m_<i>_<obj>`), square `crop_<i>`, scale `up_<i>`. Good *within* a frame; ids drift *across* frames — never rely on them |
| candidate leaves by geometry | `scripts/apex_match.py` (`match_dense`) | identity solved by epipolar agreement of SAM3 lines over 74 frames: best recall (68–73 %) |
| renderer + selection (steps 1–2) | `scripts/apex_select.py` → `$D/apex_select/` (on P5x), `$D/apex_select_p4g/` (on P5x-on-P4g) | leaf strips drawn into all 74 views on the GPU (torch, depth-tested, SIMPLE_RADIAL camera), greedy subset explaining the zoomed masks; output = P5x with apex leaves replaced, each one smooth curve from its shoot to its tip |
| structure rules | `stem_tree.end_axes`, `leaf_skeleton.smooth_curve`, apex node + "leads away" in `apex_select.py` | axes end at their last leaf / where they turn back; leaves one smooth curve, base at the stem, never folding back |
| yardstick | `scripts/apex_labels.py --score <dirs> --annotate $D/apex_compare` | 105 hand-drawn leaf lines in 6 frames (`$P/p1/frames/frame_00{00,06,14,17,23,25}.json`); identity-free pairing per frame |
| fit (step 3) | `scripts/apex_fit.py` → `$D/apex_fit_v3/` (all frames), `$D/apex_fit_v3_heldout/` (even frames) | §0.1: per apex leaf in its shoot's frame: attach, azimuth, elevation, incurve, width, tapers, roll, fold, `blade_mesh`; a bud per apex; each apex's fitted shoot in the stem tree |
| fit check | `scripts/apex_fit_check.py --skeleton <fit dir>` | any version, same scorer: counts vs the user's, shape about the shoot axis (turn, in, sideways), PQ, F1, TP/FP/FN (TP by context leaves), per-leaf support, twins, tips vs labels |
| Blender bundle | `$D/blender_bundle/vogelmeere_view_core_<stamp>.tar.gz` (+ clouds) | the user views locally; see "Viewing" below |

**Current numbers**: §0.0's table. Older summary, kept (your lines found / tip error; *precision is meaningless*:
the labels are partial): P5x 34 % / 6.9 px · s2d one-curve on fair zoom 67 % /
5.7 px · selection on P5x 58 % / 5.7 px · selection on P5x-on-P4g 57 % / 7.3 px ·
**fit (step 3) 52 % / 9.1 px with 26 apex leaves**, held-out PQ 0.10–0.16 → 0.29–0.40 (§0.1; the first version, 54 leaves with split and ghost blades, 62 % / 6.2 px).

*Written before step 3 was built, kept for the record (§0.1 is what was built):*

**Why the selection stops short** (what step 3 must fix): its leaf shape is a
crude strip (width ½ length, 4 fixed rolls), so real leaves that are wider or
tilted in between spill off their masks and are not chosen; and one SAM3 mask
over two young leaves pays only one leaf.

**Step 3, as agreed** (details in §3.2–3.9):
- start from `apex_select*` leaves (both bases are available; P4g is the
  densest cloud and the user values it);
- per leaf, fit: midrib control points (attachment on the shoot fixed or
  sliding along it), width profile, roll, fold — with a *differentiable*
  renderer (gsplat 2DGS is installed and used by P4g; or a soft rasterizer in
  torch on the same projection code as `apex_select.Views`);
- losses, all identity-free: plant-mask silhouette in each apex square;
  per-view Hungarian-matched overlap with the zoomed SAM3 masks; a mask may
  be shared by two leaves lying side by side; smoothness; leaf stays on its
  shoot and leads away from it;
- then re-run selection with the fitted shapes (add/remove leaves);
- output in the same skeleton schema (`curve`, `blade_start`) plus the new
  per-leaf `width`, `roll`, `fold` — what the PBR leaf assets snap to;
- score: `apex_labels.py --score` (found %, tip px), a label-free held-out
  check (fit on even frames, overlap on odd), and the user's Blender look.
  **Checkpoint**: if the crowns do not look visibly better than P5x and the
  current selection, stop and reconsider (fallback §"other options": a
  parametric bud from the measured apex region).

**How the user works — follow it:**
- experiments and outputs go to `$D/<experiment>/`, never the repo (disk quota);
- **no tuned thresholds**: structural rules only; if a number is needed,
  measure it and say where it came from (the user removed tuned thresholds
  from the P5x tracer on purpose);
- P5x conventions: one smooth curve per leaf, stem → tip, `blade_start`; no
  separate petiole segments; white in Blender means stem only;
- the user judges by looking in Blender locally: every deliverable ends with
  a fresh dated bundle and fetch/view commands; verify *what ships* (check the
  bundled files, the viewer prints each skeleton's path and build stamp);
- keep this doc's checklist current; plain words.

**Pitfalls already paid for:**
- an objective that lets leaves share a mask, or charges nothing for blades
  on no mask, grows split and ghost leaves; more iterations do not help (J
  had converged) — check with `apex_fit_check.py`, not only with IoU;
- label "found %" rewards many leaves; the instance scores (PQ, FP) and the
  per-leaf support show what found % hides — report both;
- Gaussians placed *on* a surface's edge draw it too big by half a Gaussian
  (fit then shrinks it): place them at cell centres;
- `pkill -f`/`pgrep -f "<script>"` also matches the shell running it — use
  `pgrep -f "[a]pex_fit.py"`, or wait on the log (`until grep -q wrote ...`);
- `rm -rf $VAR/...` is refused by Claude Code's safety check — use literal
  absolute paths;
- P5x outputs on disk can predate the tracer (vogelmeere's did): retrace
  (`pose-leaf-instances --retrace`, seconds) before comparing;
- crops for an apex come from the labelled regions, not stem-tree ends;
- SAM3's base/tip call per mask can be swapped; orient by structure;
- two views always triangulate *some* 3D curve — a 2-view fit proves nothing;
- `skeleton_2d.py` needs leaf masks inside the plant mask's box (fixed);
- `pose` env: `./setup_env.sh --with-sam3` keeps `kernels` in transformers'
  window (SAM3 mask clean-up was silently off before);
- other plants still need `--retrace` to get `end_axes`;
- a parameter clamped at a bound (`x.clamp(min=0)`) has no gradient past it:
  once there, it never leaves (v2's leaf base, v3's first attach). Use a
  sigmoid/softplus share instead;
- an element that is never a false positive needs another bound, or it grows
  onto real leaves' masks (the first structural bud);
- `/root` is a tmpfs in these containers: the `pose` env, Claude's memory and
  the scratchpad die with it (see `CLAUDE.md`).

**Suggested first prompt for the next session:** "Read `CLAUDE.md` and
`docs/apex-plan.md` §0. I have looked at the 2026-10-13 bundle in Blender:
<my verdict>. Then do open item 1 (next-pair-up proposals) and show me the
counts and scores before and after."

---

## 1. The problem, in one paragraph

At the tip of every branch (the *apex*) vogelmeere keeps a cluster of young
leaves, 2–10 mm long, folded up against each other. Every 3D point cloud we
build (hull, 2DGS at 960 px, MVS, 2DGS at full resolution) turns that cluster
into one smooth blob, because the leaves touch and hide each other. Tracing a
skeleton through a blob gives "worms". The 2D route (s2d) does not do much
better there, because SAM3's leaf masks are coarse at that size and often draw
one mask over two or three leaves, or swap ids between views.

## 2. What we found on 2026-10-12

- **The photos do show the young leaves separately** — not all of them in any
  one photo, but each leaf is clear in *some* photos. The information is there.
  (`debug/apex_look/apex_*.jpg`: left = photo, right = P2 masks + s2d midribs.)
- **The P2 masks are the weak link.** SAM3 ran on the whole plant, so a mask
  step is ~13 px of the 6000 px original (~0.3 mm). At an apex one mask
  often covers several leaves.
- **SAM3 run zoomed in on one apex** (a 16 mm square cut from the 6000 px
  original) draws finer, better-separated outlines, but still has speckle and
  some merges inside the cluster.
  (`debug/apex_zoom_probe/apex8_pass0.jpg`: photo | P2 masks | zoomed SAM3.)
- **Depth of field is shallow**: at full resolution several views of each
  apex are blurred. They still show the outline, not the inner edges.

So no single photo, and no single mask, settles the apex. The photos have to be
combined with a model of *which leaf hides which*.

---

## 3. Option 3 in plain words: "fit virtual leaves until they look like the photos"

This is called *analysis-by-synthesis*: you explain the photos by
synthesising (rendering) them from a model, and keep adjusting the model
until the renders match.

### 3.1 The idea

Imagine making a few small paper leaves, gluing them onto the real branch tip
where you think they are, and photographing that from the same 74 camera
positions. Compare your photos with the real ones. Where yours show a leaf and
the real ones don't, move the paper leaf. Repeat until every one of the 74
pairs matches. The paper leaves are now where the real leaves are, and you
know each leaf's position, length, width, bend and angle.

The computer does exactly that, with virtual leaves instead of paper ones and
thousands of small adjustments instead of a few.

### 3.2 A virtual leaf

Each young leaf is described by about 12–15 numbers:

| part | what it means | numbers |
|---|---|---|
| attachment | where on the shoot the leaf grows from (a *node*) | 1 (position along the shoot) |
| stalk (petiole) | how long the thin stalk is before the blade | 1 |
| direction | which way the leaf points as it leaves the stem (up/out, around) | 2 angles |
| midrib bend | the centre vein as a smooth curve; how much it curls along its length | 2–3 |
| length | tip distance along the midrib | 1 |
| width profile | how wide the blade is near the base, middle and tip | 3 |
| fold | young leaves are folded along the midrib like a half-closed book | 1 angle |
| roll | how the blade is twisted about its midrib | 1 angle |

These are the same numbers the leaf-asset step needs later: where to attach
the PBR leaf, which curve to bend it along, how wide to scale it, and how
much to fold and twist it. Option 3's result *is* the asset placement.

### 3.3 Where the leaves hang: the anchor

The stem tree (from P5x) ends each branch at a measured point. The apex
leaves hang off a short continuation of that branch, the *shoot axis*. The
shoot axis is fitted too, but it starts from a measured point, so the cluster
cannot drift off the plant.

### 3.4 A prior from botany: opposite pairs

Vogelmeere's leaves come in **opposite pairs**: two leaves at the same node,
pointing away from each other, and the next pair turned by 90°. Going up the
shoot the pairs get smaller and more upright, until the youngest pair is a
closed bud. The fit starts from that pattern and is gently pulled towards it.
It is a *preference*, not a rule: a leaf the photos clearly show elsewhere
wins. The rule lives in the species profile (`plant_profiles.py`), like the
base rule does now, so a rosette plant does not get it.

### 3.5 Drawing the virtual leaves into a photo

Each virtual leaf is turned into a few hundred tiny flat discs (*surfels*)
covering its blade. **gsplat** — the renderer P4g already uses — draws those
discs from each camera, with nearer discs hiding farther ones. Instead of a
real colour, every leaf gets its own colour channel, so the render says, per
pixel, *which virtual leaf is visible there*. That is the per-leaf mask the
model predicts for that photo, with occlusion handled automatically.

The renderer is *differentiable*: it reports not only the picture but also
which way each leaf number should change to make the picture match better.
That is what lets the computer adjust the leaves.

### 3.6 What "matches" means: the scores

Per photo, three comparisons, added up over all 74 photos:

1. **Outline.** The whole rendered apex vs the plant mask. The plant mask is
   reliable (it is the plant/background boundary, not leaf vs leaf), so this
   is the strongest signal.
2. **Per-leaf masks.** The rendered leaf map vs the zoomed SAM3 leaf masks.
   SAM3's ids change between photos, so in each photo the rendered leaves are
   *matched* to the SAM3 masks first (best overlap pairing), then compared.
   A SAM3 mask that covers two leaves just matches one of them partly — it
   does not force them together.
3. **Edges (optional).** Where two rendered leaves overlap, the photo should
   show an edge. Blurred photos get a low weight here.

Plus gentle penalties that keep leaves botanical: smooth midribs, pairs
roughly opposite, younger leaves no bigger than older ones.

### 3.7 The adjusting loop

Start from a first guess (step 1 below), render, score, nudge every number in
the direction that improves the score, repeat — a few thousand times, seconds
to minutes per apex on the H100.

### 3.8 How many leaves

The count is not known in advance. Try the likely counts (from the zoomed
SAM3 masks: e.g. 4, 6 or 8 leaves = 2, 3 or 4 pairs), fit each, and keep the
one that predicts **photos it was not fitted on** best (fit on even frames,
score on odd frames — the same honesty test s2d already uses).

### 3.9 What comes out, and how we know it is good

- Per apex leaf: node, stalk, midrib curve, width profile, fold, roll — written
  into `skeleton.json` beside the other leaves, viewable in Blender.
- **A realism score:** render the fitted leaves into the held-out photos and
  measure the overlap with the real outline (IoU). The same score can be
  applied to the whole digital twin later, which is a measured argument for
  "digital twin vs randomly generated plant" in the thesis.

### 3.10 What it will not do

The very youngest leaves (under ~2 mm, closed inside the bud) are not
recoverable by any method from these photos. They are modelled as a closed
bud — a pair folded together — which is right for a digital twin.

### 3.11 Why not option 2 (leaf ids inside the Gaussians)

It ends in a labelled point cloud that has to be traced again, which is the
exact step that fails at the apex. P4g at full resolution still closes the
apex into a blob, and the labels would come from the same SAM3 ids that swap
between views.

---

## 4. The steps

Each step produces something you can look at before going on.

### Step 0 — Test: do zoomed SAM3 masks alone fix the apex? (today)

**Hypothesis:** if SAM3 is run on zoomed crops of each apex, its leaf masks
become good enough that the *existing* s2d reconstruction produces clean apex
leaves, with no new 3D modelling.

**How:**
1. For every apex of the stem tree, cut a square around it from all 74
   original photos and run SAM3 ("leaf") over each pass as a video.
2. Put those masks back into frame coordinates. Inside each apex square they
   replace the P2 leaf masks; outside, P2's masks stay.
3. Run s2d exactly as for the current skeleton, once on the old masks
   (baseline, to make sure it reproduces) and once on the new masks.
4. Compare: leaf count, ids that failed, held-out reprojection error, the
   apex crops with both skeletons drawn on, and both skeletons in Blender.

**What decides it:**
- apex leaves clean and plausible in Blender → zoomed SAM3 becomes part of P2
  for small leaves, and option 3 can be narrower;
- apex still tangled (expected: masks still merge/speckle inside the
  cluster) → zoomed masks are kept as the *evidence* for option 3, and we go
  on with step 1.

**How to look at step 0 yourself** (`P=/netscratch/naeem/blender_assets/06-10-2026-Naeem/vogelmeere/plant`):

| what | where |
|---|---|
| zoomed SAM3 per apex: photo, P2 masks, zoomed masks (coloured = kept as apex leaf, grey = dropped) | `$P/debug/apex_zoom/sam3/apex<id>_pass<p>.jpg` (and `apex_zoom_clean/sam3/` with SAM3's clean-up on) |
| whole plant, skeleton drawn on 4 photos | `reprojection.jpg` in `$P/debug/apex_zoom/s2d_baseline/`, `.../s2d_zoom/`, `$P/debug/apex_zoom_clean/s2d_zoom_clean/` |
| each apex up close, all skeletons side by side on 8 views | `$P/debug/apex_zoom/compare/apex<id>_<kind>.jpg` |
| numbers per apex and for the whole plant | `$P/debug/apex_zoom/compare/summary.json` |
| all three skeletons in one Blender scene, each its own collections (`s2d_baseline_*`, `s2d_zoom_*`, `s2d_zoom_clean_*`) to toggle | `./scripts/view_in_blender.sh $P --skeleton2d $P/debug/apex_zoom/s2d_baseline,$P/debug/apex_zoom/s2d_zoom,$P/debug/apex_zoom_clean/s2d_zoom_clean` (add `--p5x` to put them over the P5x cloud) |

Reproduce:

```bash
python scripts/apex_zoom.py --workdir $P --out $P/debug/apex_zoom          # SAM3 on GPU, ~10 min
python scripts/apex_zoom.py --workdir $P --out $P/debug/apex_zoom --merge-only   # re-merge only
python scripts/skeleton_2d.py --workdir $P --masks $P/debug/apex_zoom/masks_zoom.npz \
    --tag-mm 8 --out $P/debug/apex_zoom/s2d_zoom
python scripts/apex_zoom.py --workdir $P --out $P/debug/apex_zoom \
    --compare $P/debug/apex_zoom/s2d_baseline $P/debug/apex_zoom/s2d_zoom \
              $P/debug/apex_zoom_clean/s2d_zoom_clean
```

(SAM3 needs `HF_HOME=/netscratch/naeem/hf_cache` and the token. Since the
setup fix the `pose` env always runs SAM3 with its clean-up on, so a rerun
reproduces `apex_zoom_clean`; `apex_zoom` was made with kernels 0.16, i.e.
without it.)

### Step 1 — First guess per apex leaf: lift the key view (see 0b)

Each apex leaf is read in the apex's key view (its 2D midrib), anchored at
the apex in 3D, and given the depth that makes it agree with every other
view *without* knowing which leaf is which there (inside the plant mask,
close to some leaf line). Its output is the starting point of step 3.
Viewable in Blender. (Replaces the first idea — triangulating each leaf's
tip across views — which needed the cross-view identity the apex does not
give.)

### Step 2 — Virtual leaf model and renderer

The leaf of 3.2 as code, the opposite-pair layout of 3.4, the surfel renderer
of 3.5. Check: render the step-1 guess into the photos and draw it over them.

### Step 3 — Fit one apex (apex 08)

The scores of 3.6 and the loop of 3.7, on one apex. Check: held-out IoU
against step 1, the fitted leaves drawn over the photos, Blender.

### Step 4 — All vogelmeere apices

Including leaf-count selection (3.8). Check: the whole plant in Blender,
s2d leaves + fitted apex leaves.

### Step 5 — The other plants, then (optionally) every leaf

The same fit can refine the large leaves too, so the whole plant ends up in
one representation for asset snapping.

---

## 5. Checklist

Ticked as each item is done; results go next to the item.

### Today — 2026-10-12

- [x] Check the weekend queue (`run_all_clouds.sh`).
      gf_1 complete; sugarbeet_x P4m/P4g/compare done, P5x killed 00:14 Sat on
      its first cloud; thistle_x and weed_x never started.
- [x] Look at every apex in full-res photos with P2 masks and s2d drawn on
      → `debug/apex_look/`.
- [x] Zoomed SAM3 probe on one apex (apex 08) → `debug/apex_zoom_probe/`.
- [x] Step 0.1 — zoomed SAM3 on every apex, both passes → `debug/apex_zoom/sam3/`
      7 apices (stem top + 6 branch ends), 18 mm squares, ~35 s per apex per
      pass. Per frame the young leaves come out separated much more often than
      in P2; still: ids drift around the orbit, some masks cover two leaves.
- [x] Step 0.2 — merged masks (zoomed inside apex squares, P2 elsewhere) →
      `debug/apex_zoom/masks_zoom.npz`: 1008 zoomed apex-leaf masks added;
      of 2112 P2 masks 341 trimmed, 577 dropped (the ones drawn over several
      apex leaves).
- [x] Step 0.2b — **found:** SAM3's own mask clean-up (NMS, hole filling,
      sprinkle removal) has been *off* in every P2 run so far: transformers
      5.19 wants `kernels` 0.17–0.18 and `pyproject.toml` pinned 0.16 (the
      warning is in every plant's `pipeline.log`). Same experiment with the
      clean-up on → `debug/apex_zoom_clean/`.
- [x] **Fixed in setup:** the `sam3` extra now asks for
      `transformers[kernels]`, so the kernels window follows transformers;
      `./setup_env.sh --with-sam3` loads the clean-up kernel as a test and its
      final check fails if kernels leaves the window again. Run on `pose`:
      kernels 0.16.2 → 0.17.2, kernel loads. Log: `/netscratch/naeem/setup_env_sam3.log`.
- [x] Step 0.3 — s2d baseline rerun on the old masks reproduces the current
      skeleton exactly (35 leaves, 15 failed ids, 6.9 px held out) →
      `debug/apex_zoom/s2d_baseline/`
- [x] Step 0.4 — s2d on the merged masks → `debug/apex_zoom/s2d_zoom/`,
      `debug/apex_zoom_clean/s2d_zoom_clean/`
- [x] Step 0.5 — before/after comparison → `debug/apex_zoom/compare/`
      (`whole_plant_3way.jpg`, `apex<id>_*.jpg`, `summary.json`):

      |                              | baseline (P2) | zoomed | zoomed + clean-up |
      |---|---|---|---|
      | leaves                       | 35  | 51  | 50  |
      | held-out midrib error        | 6.9 px | 6.6 px | 6.4 px |
      | leaves within 9 mm of an apex| 12  | 22  | 20  |
      | SAM3 ids not reconstructed   | 15 of 86 | 163 of 283 | 166 of 280 |
      | petioles longer than blade   | 5   | 10  | 13  |

- [ ] Blender check of the three skeletons (command in step 0 above) — yours
- [x] Decide: does step 0 alone fix the apex? **Not shown either way — the
      test was confounded by the crops** (your review, 2026-10-12): each square
      was centred on the end of the stem-tree branch, which stops where the
      cloud closes over the cluster, i.e. short of the real apex; its size was
      a fixed 18 mm; and every zoomed mask touching the square's edge was
      dropped (100–200 per pass), which threw out apex leaves the square cut.
      Redo with annotated apex boxes (step 0b). What follows was the first
      reading, kept for the record: the zoomed masks find
      more of the young leaves (12 → ~20 near the apices) and fit slightly
      better, but every apex is still a tangle of crossing midribs and
      petioles, no cleaner than before. The masks are better *per photo*;
      what breaks is *identity across photos*: the zoomed SAM3 ids drift
      around the orbit, so 163 of 283 ids never gather enough agreeing views
      to be triangulated, and pieces of different leaves get joined. s2d
      needs one leaf = one id; the apex does not give that.
      → On to step 1–3, with the zoomed + clean-up masks as their evidence,
      matched to the model photo by photo (3.6), which never needs the ids.

### Step 0b — key views: the apex read where it opens up (vogelmeere, today)

**Why the change** (your labelling, 2026-10-12): keeping one name on one
leaf across views is hard even for a person, at the apex nearly impossible.
But each apex has one or two *key views* where the cluster opens up and its
layout is plain to see. A key view settles which leaves there are, how many,
and their 2D shape — no matching across views needed.

What one view cannot give is **depth** (how far each leaf leans towards or
away from that camera). Depth comes from three things that need no
identity at all:

1. **the anchor** — an apex leaf starts at its apex, whose 3D position is
   known (the boxes below, or the stem tree);
2. **every other view, identity-free** — a leaf at the wrong depth projects
   off the plant in the side views: each projected leaf must lie inside the
   plant mask and close to *some* leaf line found there (whichever one —
   no ids);
3. **the pair prior** — opposite leaves at roughly mirrored angles.

So per leaf the key view fixes the 2D shape and only 2–3 numbers stay
unknown (how deep the tip is, how the midrib bends in depth). This is
option 3 with most of its unknowns pinned by the key view, and it uses
SAM3 only where it was good — inside one photo — never where it failed
(keeping ids across photos).

**How it is evaluated**

| what | how | needs labels? |
|---|---|---|
| leaf count, which leaves, 2D direction, length, attachment | method vs your lines in the key view | yes, key view only |
| depth | project the 3D result into held-out views: on the plant? on some leaf line? | no |
| depth, directly | a second key view where you are sure which leaf is which | optional |

**Oracle test:** your key-view lines → the lifting step. Right in held-out
views ⇒ the 3D step is solved and what is left is finding the leaves in the
key view (where zoomed SAM3 already does fairly well).

- [x] **Labels** (you): 6 frames of pass 0 (0000, 0006, 0014, 0017, 0023,
      0025), 4 apices boxed in 5–6 each, 105 leaf lines drawn base → tip
      (fully visible leaves only, no identity across frames). In `p1/frames/`
      next to the images; nothing in the pipeline reads `*.json` there.
- [x] Apex regions (`scripts/apex_labels.py` → `debug/apex_labels/`): the
      cloud points inside an apex's box in *every* view that boxes it.
      Radii 9–15 mm — the 18 mm squares of step 0 were too small for A and C
      (you were right). Box-centre triangulation was 10–24 px off for the
      lopsided clusters, carving is not. `labels.jpg` shows the labels and
      the carved points.
- [x] Zoomed SAM3 on those regions, clean-up on (`debug/apex_zoom_fair/`):
      1862 apex-leaf masks (was 1008), far fewer cut at the square's edge.
- [x] Fix: `skeleton_2d.py` crashed when a leaf mask reached past the plant
      mask's box (113 of 2537 P2 masks do, by a few px); such pixels now
      count as off the plant. Same results wherever it ran before.
- [x] Scoring against the labels (`apex_labels.py --score`), identity-free:
      per frame, labelled lines and projected skeleton leaves are paired one
      to one (≤ 15 px ≈ 1 mm); each goes to the apex box holding most of it.
      *found* = share of your lines with a partner; *precision* = share of
      skeleton leaves in the box that are one of your lines — a lower bound,
      since you drew only the fully visible ones (~4–5 per apex per frame).
- [x] Lifting from one key view (`scripts/apex_lift.py`, oracle on your
      lines): depth from the cloud's front surface → **21–25 % found** in
      held-out frames. A global push-back peaks at +1 mm: the hull-born
      clouds sit ~1 mm in front of the apex leaves, and per leaf they are
      off by more. **Single-view depth from the cloud is not good enough.**
- [x] Identity by geometry (`scripts/apex_match.py`): a leaf's base and tip
      must agree across frames (epipolar), and a third frame must show a line
      where it projects. On your 6 frames alone: 35 % found (most leaves seen
      in only 2 of 6). On SAM3's lines over all 74 frames (labels only score):

      | method | found | precision | tip px |
      |---|---|---|---|
      | P5x (`p5x/`, current best by eye) | 34 % | 15 % | |
      | P5x on P4g (`p5x_p4g/`) | 37 % | 15 % | |
      | s2d, original P2 masks | 58 % | **33 %** | 7.5 |
      | s2d, zoomed 18 mm squares | 53 % | 20 % | 6.1 |
      | s2d, zoomed fair crops + clean-up | 59 % | 22 % | 5.4 |
      | geometry match, P2 lines (≥ 4 frames) | **73 %** | 9 % | 6.5 |
      | geometry match, fair zoomed lines (≥ 8 frames) | 68 % | 13 % | 5.6 |

      Geometry finds the most of your leaves, with the best tips, and needs
      no ids — but it keeps too many leaves (59 for the 4 apices at ≥ 8
      frames): chance agreements between lines of different leaves survive.
- [x] **Steps 1–2, on P5x** (`scripts/apex_select.py` → `debug/apex_select/`):
      candidates per apex = the geometry-matched leaves (≥ 4 frames) plus
      every apex leaf of P5x, P5x on P4g, s2d and s2d on fair zoom; each a
      leaf-shaped strip (width ½ its length, narrow stalk) at 4 roll angles.
      **Step 1, renderer:** all strips drawn into the apex square of all 74
      views on the GPU, depth-tested, with P5x's other leaves as occluders.
      **Step 2, selection:** greedily add the candidate that best explains the
      zoomed SAM3 masks — per view each leaf earns the pixels of its best mask,
      each mask pays one leaf, pixels off the mask cost ALPHA = ½ each (a
      leaf must lie mostly on one mask). A per-leaf cost of 0 / 150 / 300 px
      kept 26 / 24 / 23 leaves, so it is set to 0: nothing tuned. Seconds per
      apex. Result: P5x with its 34 apex leaves replaced by 26 selected ones
      (25 from the geometry candidates), one curve each, no petiole segments.
      `apex_<X>.jpg`: photo | SAM3 masks | kept leaves as drawn.

      | method | your lines found | its apex leaves found |
      |---|---|---|
      | P5x | 34 % | 17 of 43 |
      | P5x on P4g | 37 % | 17 of 45 |
      | s2d, original masks | 58 % | 22 of 33 |
      | s2d, fair zoom | 59 % | 23 of 50 |
      | geometry match | 68 % | 35 of 92 |
      | **selection on P5x** | 45 % | 18 of 35 |

      By eye (`debug/apex_compare/compare/apex<X>_labelled.jpg`: photo | P5x
      | selection | s2d): the selection draws separate leaves where P5x
      draws loops — but fewer of them. It stops at ~26 because (a) SAM3 often
      draws one mask over two young leaves and a mask pays one leaf, and (b)
      the strip shape is crude (fixed width, 4 rolls): a real leaf that is
      wider or tilted between rolls spills off its mask and is penalised.
- [x] Blender: every method's skeleton, leaves marked against your labels
      (`apex_labels.py --annotate` → `debug/apex_compare/<method>/`): leaves
      that pair with one of your lines are **magenta and thick**, in their own
      `<method>_gt_found` collection. Overlays now also work on the cloud
      panels (`--clouds ... --skeleton2d A,B`).
- [ ] Your look in Blender (commands below)
- [x] **Step 3** (2026-10-12 night): fit each kept leaf's width, roll and bend
      to the masks (differentiable, gsplat), a mask may pay two leaves side by
      side, leaves added from the candidate pool and dropped by the same
      objective → `scripts/apex_fit.py`, `$D/apex_fit/`. Design, numbers and
      viewing in §0.1.

- [x] **Fixes before step 3** (your review):
      - *P5x on disk was stale*: `p5x/` (15:01, Oct 9) and `p5x_p4g/` (14:24)
        predate the "one curve per leaf / curves start on the stem /
        structural rules only" tracer commits. Retraced copies with the
        current tracer: `debug/p5x_current`, `debug/p5x_p4g_current`
        (`pose-leaf-instances --retrace --out debug/...`; `p5x/` untouched).
      - *Stem running on into the apex and hooking back*: new
        `stem_tree.end_axes`, called by the P5x tracer — a stem or branch ends
        where it stops leading away from its own start, or at its last
        attachment (child axis or leaf start), whichever comes first; a leaf
        or child on a cut part is re-attached to the nearest point left.
        No thresholds. Stem + branches: 229 → 161 mm (P4g: 546 → 286 mm).
        Tests: 4 new in `tests/pose_estimator/test_stem_tree.py`, all 26 pass.
      - *Apex leaves diverge from the shoot*: in the selection each apex's
        shoot runs on to its apex node (median of its leaves' bases) and every
        apex leaf is one curve starting on that shoot; base/tip of SAM3 lines
        re-oriented (the base is the end at the stem). Joins 3.0 mm median.
        A last `end_axes` pass removes stubs left beside a shoot.
      - *s2d petiole webbing*: the s2d reference was the old run without
        `--leaf-curves`, and the viewer drew its petioles in stem white. Now
        `debug/s2d_curves_baseline`, `debug/s2d_curves_zoom_fair`
        (`--leaf-curves --stem-tree debug/p5x_current/skeleton.json`): one
        curve per leaf where a 2D stalk walk agrees (25/36, 35/52), the rest
        keep a petiole — drawn light green now, as in the P5x scene; P5x leaves
        (no colour of their own) get per-leaf colours instead of white.

      | method (current versions) | your lines found | its apex leaves found | tip |
      |---|---|---|---|
      | P5x, retraced | 34 % | 17 of 43 | 6.9 px |
      | P5x on P4g, retraced | 35 % | 16 of 45 | 7.7 px |
      | s2d one-curve, original masks | 54 % | 22 of 34 | 6.7 px |
      | s2d one-curve, fair zoom | **60 %** | 23 of 50 | 5.8 px |
      | **selection on retraced P5x** | 53 % | 20 of 34 | **5.6 px** |

- [x] **Fixes after your Blender look** (smooth joins; C and D floating / reversed):
      - *Smooth stalk → blade, every method*: new `leaf_skeleton.smooth_curve`
        draws any leaf polyline the way the P5x tracer does (re-spaced to the
        hull voxel, 24 × 0.5 Laplacian, ends fixed). The selection's join and
        s2d's leaves now go through it: every s2d leaf is one curve from where
        it joins to its tip (before: separate petiole meeting its blade at a
        kink, and `fit_curve`'s few control points drawn as straight pieces).
      - *Selection, C and D*: a leaf's base is the end nearer its apex node,
        and the node is the median of the bases — settled together by
        iterating from the region's centre (the old "nearer any stem" rule
        turned leaves where branches crowd). Every apex leaf joins the nearest
        shoot, unconditionally (the old "gap shorter than the leaf" rule left
        C and D's leaves floating, and the shoot's run-on was then cut as
        unused). A leaf leads away from where it joins — tip farther from it
        than its base — else it is turned (4 were).
      - *s2d, same rule*: a leaf that folded back towards where it joined
        (3 of 36, 5 of 52; mostly 2D stalk walks of 17–55 mm) is re-joined at
        the nearest stem point, turned if needed; `petiole_rule` says so.
      - Now: every leaf of every method one curve, none folding back; worst
        bend per leaf median 6° (selection), 8–10° (s2d), 12° (P5x).

      | method | your lines found | tip |
      |---|---|---|
      | P5x | 34 % | 6.9 px |
      | P5x on P4g | 35 % | 7.7 px |
      | s2d one-curve, original masks | 57 % | 7.2 px |
      | s2d one-curve, fair zoom | **67 %** | 5.7 px |
      | selection on P5x | 58 % | **5.6 px** |

**Blender, all current methods on one plant** (`P=/netscratch/naeem/blender_assets/06-10-2026-Naeem/vogelmeere/plant`,
`C=$P/debug/apex_compare`):

```bash
# over the P5x scene (its cloud, coloured by leaf):
./scripts/view_in_blender.sh $P --p5x --skeleton2d $C/p5x_current,$C/p5x_p4g_current,$C/apex_select,$C/s2d_curves_baseline,$C/s2d_curves_zoom_fair
# on the clouds side by side (each skeleton drawn on every cloud panel):
./scripts/view_in_blender.sh $P --clouds p4a,p4b,p4m,p4g --skeleton2d $C/p5x_current,$C/apex_select
```

Collections are named `s2d_<method>_*` (`..._midribs`, `..._gt_found`, …),
and `s2d_<method>@<cloud>_*` on the cloud panels; toggle them to compare.
Leaves matching one of your lines are magenta and thick (`..._gt_found`).
Pictures: `$C/compare/apex<X>_labelled.jpg` (photo | P5x | selection | s2d),
`$P/debug/apex_select/axes_check.jpg` (stem ends and apex leaves per apex).

**Viewing locally** (2026-10-12): `p5x/` and `p5x_p4g/` were retraced in place
with the current tracer (old files in `debug/stale_backup/`), the comparison
set rebuilt from them, and two bundles written to
`/netscratch/naeem/blender_assets/06-10-2026-Naeem/vogelmeere/plant/debug/blender_bundle/`:
`vogelmeere_view_core.tar.gz` (9 MB: the P5x scene, all skeletons marked
against the labels, the pictures, and `scripts/` with the viewer) and
`vogelmeere_view_clouds.tar.gz` (120 MB: the four clouds, P4m and P4g thinned
to 0.1 mm). Unpacked side by side they are a mini-repo:
`./scripts/view_in_blender.sh vogelmeere/plant --p5x --skeleton2d ...` runs as is.

#### Annotation guide (LabelMe, on `p1/frames/frame_XXXX.jpg`, saved to `debug/annotations/`)

- `apex_A`, `apex_B`, … — rectangle around the whole cluster. Two frames per
  apex, far apart. Same name for the same apex.
- `leaf` — linestrip along each visible leaf's midrib, **base → tip**, in the
  apex's key view(s). Plain label, no attributes, no numbering needed: the
  first point is the base end, the last the tip end. Draw only what you
  see; a hidden base is fine (apex leaves start at the apex anyway).
- Optional: if you use two key views for one apex and are *sure* of a leaf
  in both, label it `leaf_<n>` with the same number in both frames.
- Which frame is a key view is your call per apex; P3 angles for reference
  (azimuth only; ~10° per frame, falling as the frame number rises. The
  passes are ~44° apart in height — pass 0 about level with the plant, pass 1
  from ~45° higher; an earlier "+19° / −25°" here was measured from the centre
  of a circle fitted through both passes, i.e. between them, and was wrong): 0000 0°, 0006 291°, 0012 233°,
  0018 179°, 0024 126°, 0030 50°; pass 1: 0037 2°, 0043 293°, 0049 246°,
  0055 184°, 0061 126°, 0067 68°.

### Next

- [x] Upgrade `kernels` in the `pose` env (done via setup_env, see 0.2b)
- [ ] Rerun P2 on vogelmeere with the clean-up on, into a copy, and compare
      masks before deciding whether to rerun P2 → P5x on all plants
- [ ] Restart the queue for sugarbeet_x (P5x), thistle_x, weed_x
- [x] Step 1 — first guess per apex leaf: the selection (`apex_select.py`),
      not the key-view lifting (dead end, 21–25 %)
- [x] Step 2 — virtual leaf + renderer (`apex_fit.py`, §0.1)
- [x] Step 3 — fit, all four vogelmeere apices at once (§0.1)
- [x] Step 4 — leaf count by the objective, whole plant in Blender (§0.1)
- [x] Opposite-pair prior (§3.4): partner proposals (2026-10-13, §0.1); A gained one
- [x] Leaves in the shoot's frame, bud, context leaves (v3, 2026-10-13, §0.0–0.1)
- [ ] Next-pair-up proposals (B's and C's missing inner leaves, §0.0 open 1)
- [ ] Midline term (§0.0 open 3)
- [ ] Step 5 — other plants

---

## 6. Words used here

| word | meaning |
|---|---|
| apex | the tip of a branch, where new leaves grow |
| node | a point on the stem where leaves attach (here: two, opposite) |
| midrib | the centre vein of a leaf, from its base to its tip |
| petiole | the thin stalk between the stem and the blade |
| s2d | `scripts/skeleton_2d.py`: the skeleton found in the 2D masks, placed in 3D |
| stem tree | the stem and its branches, traced by P5x from the stem cloud |
| surfel | a tiny flat disc, the drawing primitive of 2DGS / gsplat |
| differentiable renderer | a renderer that also says how to change the 3D model to change the picture |
| held-out views | photos not used for fitting, used only for scoring |
| IoU | intersection over union: overlap of two masks, 0 (none) to 1 (identical) |
