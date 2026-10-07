# skeleton_2d — a plant skeleton found in the photos, placed in 3D

`scripts/skeleton_2d.py` builds a skeleton — per leaf a tip, a base, a midrib
and a petiole, plus the stem — from SAM3's 2D leaf masks, and uses 3D only to
*position* what the masks already show. It is the third of three skeletons a
run can produce:

| | what it starts from | leaves | stem |
|---|---|---|---|
| P5 (`pose-structure`) | P4c organ labels on the cloud | split by geodesic depth from the stem | traced through the labelled cloud |
| P5x (`pose-leaf-instances`) | SAM3's 2D leaf ids voted onto the cloud | one per id, merged across passes | the stem tree of the stem cloud (`pose_estimator/stem_tree.py`) |
| skeleton_2d | SAM3's 2D masks, per photo | one per id, triangulated | the P5x stem tree, or its own 2D stem (plant profile) |

It is a diagnostic script, not a pipeline phase: it reads a finished run
(P2 masks, P3 poses, P5's plant frame, P5x's labels and stem tree) and writes
only into `--out`.

Why 2D first: SAM3's masks are clean where the 3D labels are not — carved
geometry is thick and labels bleed at leaf borders — so every part is found in
the photos and only triangulated in 3D.

## Run it

```bash
python scripts/skeleton_2d.py --workdir <specimen>/plant --tag-mm 8 --out <specimen>/plant
./scripts/view_in_blender.sh <specimen>/plant --p5x --skeleton2d <specimen>/plant/skeleton.json
```

`--tag-mm` (the AprilTags' printed side) sets the scale; every millimetre
threshold below follows it. Without tags, `--flat-lay` fits the scale against a
flat-lay capture, or `--mm-per-unit` gives it directly. `--passes 0` restricts
the fit to one capture pass, which matters when the plant moved between passes.

## The plant profile decides the rules

The rules that differ between species come from the plant profile the specimen
folder names (`src/pose_estimator/plant_profiles.py`; `--profile` overrides).
`vogelmeere_21/plant` is vogelmeere, `gaensefuss_1/plant` gaensefuss, and a
folder that names no known species keeps the rules gaensefuss was tuned on.

| profile | architecture | base of a 2D mask | petiole |
|---|---|---|---|
| gaensefuss | caulescent | foot | leaf_axis |
| vogelmeere | caulescent | stem_contact | stem_tree |
| sugarbeet, thistle | rosette | foot | crown |
| anything else | caulescent | foot | leaf_axis |

P5 reads the same profile for its `--architecture` default, so one folder name
sets the rules for every skeleton.

## Step by step

**1. Per mask, in 2D** (`extract_2d`; cached in `--out/evidence_2d.json`, rebuilt
when the base rule changes):

- The two ends of the mask are the ends of its longest path through itself.
- Which end is the **base** is the profile's base rule:
  - `foot` — the end nearer the plant's foot, walking through the plant mask.
  - `stem_contact` — the end with stem-mask pixels around it (within a sixth
    of the leaf's length), where the petiole goes in. It falls back to `foot`
    when neither end clearly touches stem — on frames whose stem mask is
    empty, for one. On a bushy plant overlapping blades are shortcuts through
    the plant mask, and the foot rule flips tip and base.
- The **midrib** is the path from base to tip that keeps to the middle of the
  mask.
- Per frame, a 2D **stem** line from the foot to the highest stem-mask pixel.

**2. Per leaf, in 3D** (`reconstruct`):

- Tip and base are triangulated by RANSAC over the frames that show the leaf.
- SAM3 swaps ids between neighbouring leaves mid-pass, so RANSAC is repeated
  on what each model leaves over: every set of views that agrees on a tip is a
  piece of its own.
- The midrib is a 3D curve whose projections lie on the 2D midribs
  (chamfer least squares, distortion removed).
- Pieces are one leaf when their tips are within 6 mm, they never claim the
  same frame, and — within one pass — their bases are within 8 mm. Across
  passes only the tip is compared, because how much petiole SAM3 includes in
  a leaf mask changes with elevation.

**3. Petioles**, by the profile's petiole rule:

- `stem_tree` — on the stem tree P5x traced from the stem cloud: a stalk of
  the tree that ends at this blade's base is its petiole; otherwise the
  nearest stem or branch point; otherwise none, if nothing is within 1.5 blade
  lengths. For plants whose leaves sit on side branches.
- `leaf_axis` — continue the tip→base axis until it meets the one stem
  (within 0.8 blade lengths and a ~19° cone), else the junction found in the
  photos, else the nearest stem point.
- `crown` — a straight line to the crown.

If `p5x/skeleton.json` predates the stem tree it has no `axes`, and the script
re-traces P5x itself (as `--retrace-p5x` does).

## Outputs

| file | what |
|---|---|
| `skeleton.json` | P5x's schema in P5's plant frame: `crown`, `stem`, `axes` (the stem tree, when used), and per leaf `tip`, `base`, `midrib`, `petiole`, `petiole_rule`, the SAM3 ids it came from and its fit error |
| `scores.json` | midrib reprojection error on held-out frames (fit on even frames, scored on odd), leaf count, failed ids and why, midrib lengths in mm, petiole rules used, `petioles_longer_than_blade` |
| `reprojection.jpg` | the skeleton drawn on four photos it was fitted on |
| `p5x_vs_2d.jpg` | P5x's skeleton beside this one, same views |
| `evidence_2d.json`, `masks.npz` | caches of the per-mask evidence and the masks |

In Blender (`--skeleton2d`) each part is its own object: `s2d_stem`,
`s2d_branches`, `s2d_petioles`, `s2d_midribs` (named after the SAM3 ids),
`s2d_tips`, `s2d_crown`.

## What it measured

- gaensefuss_1, pass 0 (2026-10-05): 34 leaves against a flat-lay count of
  34–35; midrib reprojection 3.1 px on held-out frames.
- vogelmeere (2026-10-06, one stem rule for every plant): 56 leaves, 6.7 px
  held out. 45 of 56 petioles were longer than their own blade and 18 attached
  at the clamp — the leaves sit on side branches and the only stem the rule
  knew was the main one. 10 pairs of leaves lay on one blade, 6 of them with
  tip and base swapped. This is what the stem_tree and stem_contact rules are
  for.

## Known limits

- SAM3's stem mask is sometimes nearly empty on a pass (vogelmeere pass 1);
  `stem_contact` then falls back to `foot` on those frames.
- Leaf identity is SAM3's id within a pass. Across passes, two leaves whose
  tips are within 6 mm merge.
- The apical clusters of small leaves are where both the 2D ids and the 3D
  stem tree are least reliable.
