# The apex plan — reconstructing the young leaves at a shoot tip

Started 2026-10-12. Plant: vogelmeere
(`/netscratch/naeem/blender_assets/06-10-2026-Naeem/vogelmeere/plant`).
Every experiment writes into `<plant>/debug/<experiment>/`, never into the repo.

---

## 0. Start here — handoff for option 3, step 3 (written 2026-10-12, end of day)

Sections 1–6 below are the day's log, including dead ends and numbers that
were later corrected. **This section is the current truth.**
`P=/netscratch/naeem/blender_assets/06-10-2026-Naeem/vogelmeere/plant`, `D=$P/debug`.

**Goal of the next session.** Option 3, step 3: *fit* the apex leaves to the
photos. Until now the apex leaves were only *chosen* from candidates, never
shaped, and by eye they are not yet better than before. Step 3 is what should
make them better.

**What exists and works (reuse it):**

| piece | where | what it gives |
|---|---|---|
| apex regions from hand boxes | `scripts/apex_labels.py` → `$D/apex_labels/apex_regions.npz` | 4 apices (A–D), cloud points carved by boxes in 5–6 views, radius 9–15 mm |
| zoomed SAM3 on those regions, clean-up on | `scripts/apex_zoom.py --regions` → `$D/apex_zoom_fair/sam3/apex<X>_pass<p>.npz` | per frame per apex: leaf masks at 1008 px of the square (`m_<i>_<obj>`), square `crop_<i>`, scale `up_<i>`. Good *within* a frame; ids drift *across* frames — never rely on them |
| candidate leaves by geometry | `scripts/apex_match.py` (`match_dense`) | identity solved by epipolar agreement of SAM3 lines over 74 frames: best recall (68–73 %) |
| renderer + selection (steps 1–2) | `scripts/apex_select.py` → `$D/apex_select/` (on P5x), `$D/apex_select_p4g/` (on P5x-on-P4g) | leaf strips drawn into all 74 views on the GPU (torch, depth-tested, SIMPLE_RADIAL camera), greedy subset explaining the zoomed masks; output = P5x with apex leaves replaced, each one smooth curve from its shoot to its tip |
| structure rules | `stem_tree.end_axes`, `leaf_skeleton.smooth_curve`, apex node + "leads away" in `apex_select.py` | axes end at their last leaf / where they turn back; leaves one smooth curve, base at the stem, never folding back |
| yardstick | `scripts/apex_labels.py --score <dirs> --annotate $D/apex_compare` | 105 hand-drawn leaf lines in 6 frames (`$P/p1/frames/frame_00{00,06,14,17,23,25}.json`); identity-free pairing per frame |
| Blender bundle | `$D/blender_bundle/vogelmeere_view_core_<stamp>.tar.gz` (+ clouds) | the user views locally; see "Viewing" below |

**Current numbers** (your lines found / tip error; *precision is meaningless*:
the labels are partial): P5x 34 % / 6.9 px · s2d one-curve on fair zoom 67 % /
5.7 px · selection on P5x 58 % / 5.6 px · selection on P5x-on-P4g 57 % / 7.0 px.

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
- P5x outputs on disk can predate the tracer (vogelmeere's did): retrace
  (`pose-leaf-instances --retrace`, seconds) before comparing;
- crops for an apex come from the labelled regions, not stem-tree ends;
- SAM3's base/tip call per mask can be swapped; orient by structure;
- two views always triangulate *some* 3D curve — a 2-view fit proves nothing;
- `skeleton_2d.py` needs leaf masks inside the plant mask's box (fixed);
- `pose` env: `./setup_env.sh --with-sam3` keeps `kernels` in transformers'
  window (SAM3 mask clean-up was silently off before);
- other plants still need `--retrace` to get `end_axes`.

**Suggested first prompt for the new session:** "Read `docs/apex-plan.md`
section 0 and the code it names. Then propose a concrete design for option 3
step 3 on vogelmeere's 4 apices — model parameters, renderer, losses,
evaluation — before writing code."

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
- [ ] **Step 3, next:** fit each kept leaf's width, roll and bend to the masks
      (differentiable, gsplat), and let a mask pay two leaves when both lie
      on it side by side — the two reasons the selection stops early.

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
- [ ] Step 1 — first guess per apex leaf (now: the key-view lifting of 0b)
- [ ] Step 2 — virtual leaf + renderer (adds width, fold, occlusion to the
      lifted midribs)
- [ ] Step 3 — fit apex 08
- [ ] Step 4 — all vogelmeere apices
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
