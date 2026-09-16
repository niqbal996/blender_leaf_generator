# leaf_pose — measuring a dissected plant from one photograph

`leaf_pose` takes a flat-lay capture — the leaves detached and laid out on a
dark backing, petioles still attached, photographed from directly above — and
returns, for every leaf: a mask with a sub-pixel outline, the midrib, the
tip, and the petiole origin, which is the cut end that was joined to the stem.

Install: [leaf-pose.md's companion](leaf-pose-install.md). One command.

```bash
leaf-pose --input /mnt/d/PBR_Scans/2026-09-15-Naeem/gaensefuss_31 \
          --workdir runs/gaensefuss_31 --marker-mm 20 --visualize
```

It is a sibling of `pose_estimator`, not a replacement. That package recovers
the same quantities for a plant that is still assembled, and pays for it with
a multi-view solve, a visual hull and a 3D skeleton. Here the plant has been
taken apart and laid flat, so the geometry is already in one plane and the
measurement is direct. If you have the choice, this is the cheaper and far
more accurate way to get leaf shape; it just costs you the plant.

## What a capture is

24 frames of one static scene. Twelve LEDs fire one at a time through
**parallel** polarisers, then the same twelve through **crossed** ones. The
rig writes them in that order — it is also the order `rigdef_cam.xml`
declares, `sidx` 0–11 then `didx` 12–23 — and `leaf-pose` checks itself
against brightness rather than trusting the order blindly, because a capture
saved backwards would invert everything downstream and still run.

Two images come out, and the whole package is written against these rather
than against the frames:

- **flat** — the mean of the crossed set. Crossed polarisers delete the
  surface reflection, and averaging twelve light directions deletes the
  shading with it. What is left is the leaf's own colour, evenly lit. This is
  what gets thresholded, matted, and shown to you.
- **specular** — what the parallel set has that the crossed set does not,
  which is the surface reflection alone. A midrib is a raised ridge, so it
  catches that reflection differently from the flat lamina beside it. This is
  evidence about *shape*, which `flat` has deliberately thrown away.

A single ordinary photograph works too. You lose the specular image and the
photometric normals; everything else runs.

## The four questions, and what answers each

### 1. Where is each leaf?

A grouping problem. On a flat lay the leaves are separated by backing, so a
threshold answers it exactly and in a second — there is nothing here for a
learned model to contribute.

The threshold is on **log luminance**, not on greenness, and that choice was
made from the data rather than from first principles. On `gaensefuss_31` the
felt sits at a luminance of 0.0055 and lit plant tissue at 0.16, a ratio of
30. The colour index does far worse, for a reason that is easy to miss:
excess green divides by R+G+B, so on a nearly unlit backing it divides by
almost nothing and amplifies sensor noise into apparent colour. The felt's
median excess green is 0.12–0.18, not the ~0 a neutral surface implies.

It also does not care what colour the tissue is, which is the reason that
actually mattered. A petiole is paler and yellower than the blade it carries.
Under the colour index it fell below threshold and was cut off: leaf 5's
stalk was truncated by 210 pixels, and the leaf then measured as having
almost no petiole — a wrong number rather than a missing one.

Greenness is still used, but only to say *what* a blob is, never where its
edge is. Averaged over thousands of pixels it is a clean signal, and it is
what rejects the fiducial markers. Three rules, each with a flag, each
rejecting something that was really in the frame:

| rule | rejects | measured on `gaensefuss_31` |
|---|---|---|
| `min_area_fraction` | lint and dust on the backing | specks at 1e-5 of the frame |
| `min_greenness` | the markers, and anything not plant | markers 0.097 and 0.120, against 0.70–0.96 for leaves |
| `min_solidity` | the root system | root 0.08, leaves 0.63–0.95 |

Every blob, accepted or rejected, is written to `detections.json` with its
measurements and its reason. A leaf that goes missing and a root that is
correctly dropped look identical in a count, so the count is not what you
should read.

**When to use `--instances sam` instead.** When leaves touch or overlap,
colour genuinely cannot separate them and SAM2 can. That is the case it is
there for, and it is not the case this capture presents: run on
`gaensefuss_31`, SAM2's automatic mask generator found **20 of the 27 leaves
the colour route found**, missing mostly the small ones and one large one
outright. That is not a criticism of the model — it is being asked to
discover objects with no prompt, on a frame downscaled for detection, where
the alternative gets to exploit a fact about this scene that is simply true.
Reach for it when the colour route's `detections.json` shows two leaves
merged into one blob.

It is not there to find the edges — see below.

### 2. Where exactly is the edge?

A resolution problem, and this is the one structural decision in the package
worth arguing for: **SAM is the wrong instrument for it at this scale.**

SAM's mask decoder emits a 256×256 logit map that is upsampled to the image.
Against a 9568×6376 frame that is about 37 frame pixels per logit pixel, so a
leaf's marginal teeth and a 16-pixel-wide petiole are below the decoder's own
grid before any thresholding happens. No amount of prompting recovers detail
that was never represented.

So whatever produced the coarse mask — colour or SAM — the boundary is
re-decided at native resolution, per leaf, from the image:

1. Score every pixel in the leaf's own crop by the same log-luminance index,
   and turn it into a rough alpha by asking where it falls between this
   blob's interior level and the surrounding backing's. Both levels come from
   *this crop*, so a leaf in a dim corner is judged against its own
   neighbourhood.
2. **Guided-filter that alpha with the photograph as the guide.** This is the
   step that makes the edge sharp. The filter's output is a locally linear
   function of the guide, so the alpha transition is forced to coincide with
   the image's own intensity transition — the real leaf margin — instead of
   with the upsampled coarse mask's idea of it.
3. Take the 0.5 crossing with marching squares. That interpolates *between*
   two pixels, so the contour is not quantised to the pixel grid, which is
   the point of having refined at full resolution at all.

Refinement is allowed to move the boundary but not to annex new territory:
the result is intersected with a dilation of the coarse mask, three detection
pixels wide. The backing carries bright lint fibres, and at full resolution a
fibre touching a leaf is connected to it. That cost more than a ragged
outline — a 200-pixel fibre becomes the furthest point of the shape, so the
leaf's "tip" lands out on the backing and the midrib follows it there.

The outputs are `masks/leaf_NNN.png`, `alpha/leaf_NNN.png` (the matte, for
compositing) and the sub-pixel contour in `leaves.json`. The contour is what
a mesh builder wants: `leaf_generator`'s Blender path builds from a contour.

### 3. Where does the midrib run?

The shape of the mask and the content of the photograph each know something
the other does not, and the midrib is found by making them agree.

**The mask knows the topology.** Its medial axis runs down the middle of the
blade and out along the petiole, and the two ends of the longest path
*through* the leaf's interior are the petiole's cut end and the tip — not
because anything was assumed about which is which, but because those are the
two points furthest apart through the shape. Measuring the distance through
the shape rather than across it is what keeps this right for a curved leaf:
the straight line between the ends of a bent leaf leaves the leaf, so a
principal axis fitted to it is a chord rather than an axis.

**The photograph knows where the rib is.** It is raised, so it is brighter
than the lamina either side and it catches the reflection the crossed
polarisers threw away. A ridge filter responds to exactly that — a bright
line of a given width, at any orientation.

So: endpoints from the shape, then a minimum-cost path between them through a
cost that prefers both the middle of the blade and the visible ridge. For an
unlobed leaf the two agree and the ridge term changes nothing. For a lobed
one the medial axis alone branches into every lobe, and the ridge term is
what keeps the path on the rib.

The filter scales are fractions of each leaf's own half-width, not pixel
counts, because one capture holds leaves from 60 to 1400 pixels across and no
fixed sigma serves both.

**`--photometric` is the strongest version of this.** Twelve photographs
under twelve known light directions over-determine the surface orientation at
every pixel: brightness is albedo times the cosine between the normal and the
light, so three lights already fix the normal and the other nine make it
robust. The crossed set is used, because Lambert's law describes diffuse
reflection and crossed polarisers are what remove everything that is not.

The midrib is then a **crease in the normal field** — normals tip away from
the crest on both sides — which is a far more local and specific signal than
"slightly brighter than its surroundings". The sign matters and carries the
discriminating power: a groove, such as the seam where one leaf overlaps
another, tips its normals inward and scores negative, so it cannot be
mistaken for a midrib.

You also get the normals themselves, at `normals/leaf_NNN.png`, in the
OpenGL convention `leaf_generator` already reads — a *measured* normal map
rather than one inferred from a single photograph.

The cost is one more pass over the RAW files. It needs `rigdef_cam.xml`,
which the rig writes beside the capture session. (That file is not
well-formed XML — it puts raw angle brackets inside attribute values — so it
is matched rather than parsed. It is the rig's output, not ours to correct.)

### 4. Which end is which?

Getting this backwards is not cosmetic. Every quantity measured from the base
outward — the width profile, the insertion angle, the arclength a mesh is
built against — inverts silently, and the result still looks like a leaf.

Both ends are narrow, so comparing widths is close to a coin toss on a young
leaf. What differs is *how* the narrowness is arranged:

- **the petiole is a stalk** — small width, staying small for a long run,
  then stepping up sharply where the lamina begins.
- **the tip is a taper** — width growing from zero more or less steadily,
  with no run where it is both narrow and constant.

So the main vote is "which end has the longer thin run", with mean end width
as a second, weaker vote for the leaf whose petiole was cut off flush.

A third vote comes from **colour**, and it earns its place by being
independent of the other two: a petiole is chlorophyll-poor next to the
lamina it carries. Checked against the leaves whose shape is unambiguous on
`gaensefuss_31`, colour agreed on all 11 of them, with the base reading a
quarter of the tip's excess green on the largest. It is weighted at half, so
it cannot overturn a clear stalk but is decisive where the two shape votes
are near-tied.

**All three fail together on the leaf with no petiole at all** — cut off, or
never attached when the leaf was laid out. Two of them describe the stalk and
the third describes the tissue beside it, so with no stalk there is nothing to
read. Those leaves come back with a low `confidence`, the run names them, and
the diagram is where you settle them by eye. That is the honest state: this
code cannot orient a petiole-less, entire-margined leaf on its own.

### A fourth vote was tried and removed — read this before adding another

A leaf's marginal teeth point toward its apex, so the asymmetry of the margin
profile looks like a direction cue that needs no petiole at all. It validated
**8/8** against gaensefuss's unambiguous leaves, survived a control that
excluded the petiole, and was shipped. It was wrong, and the way it was wrong
is worth recording because the next plausible cue will fail the same way.

What the statistic actually measures on most leaves is the **step where the
blade narrows into the petiole**. That is one large asymmetric slope event; it
dominates a cubed moment, and it always points the "tip" at the stalk. On a
toothed leaf that error happens to agree with the real teeth, so the
validation passed.

On `vogelmeere_1` — *Stellaria*, entire margins, no teeth anywhere — it was
right on **3 of 15** leaves, worse than abstaining, and it overturned six
stalks that were unmistakable (one leaf had 48% of its midrib in a thin run at
one end and 8% at the other).

Nothing about the number warned of this. Its median magnitude was **1.74 on
the untoothed species against 2.81 on the toothed one**, so no threshold
separates "reading teeth" from "reading the petiole step". Two repairs were
tried and measured: trimming the petiole out of the profile first made it
**0/15**, and a quartile-based statistic immune to a single step event dropped
the toothed case to **1/8**.

Scored against every leaf where the stalk settles the answer independently:

| | vogelmeere (entire) | gaensefuss (toothed) |
|---|---|---|
| three votes | **15/15** | **8/8** |
| + margin teeth | 9/15 | 8/8 |

The cue contributed nothing where the answer was already known, and inverted
it where it was not. The lesson generalises: **validate a direction cue on a
species whose margin cannot supply it**, or the petiole step will validate it
for you.

## The diagram

`--visualize` writes `leaf_poses.png`: three panels, because the result
answers three questions and one panel can only answer one of them well.

- **the photograph**, cropped to the leaves, with every contour, midrib, tip
  and petiole origin drawn on. This is the panel that makes an error obvious.
  A midrib that has run into a lobe, or a tip and a base the wrong way round,
  is visible at a glance and invisible in any table.
- **every midrib on a common origin and heading**, which turns the spread of
  curvature and length across one plant into a single picture.
- **petiole and blade length per leaf, ranked** — in millimetres when the
  markers gave a scale, in pixels when they did not, and the axis says which.

Colour means one thing throughout: blade/midrib blue, petiole magenta, tip
yellow. The leaf outline is deliberately not a fourth colour — it is the
object, not a measurement, so it wears the neutral ink the labels do.

Veins are the one thing kept off that overlay, for two reasons: they would be
a fourth simultaneous colour, which no four-hue set on this surface
distinguishes reliably, and forty polylines a leaf would bury the panel that
exists to be scanned. `--veins` puts each leaf in its own `leaves/leaf_NNN.png`
instead, where the midrib is the only other mark.

## Veins — what is reachable, and what is not

`--veins` finds the few **large veins that leave the midrib near the base**
and run out into the basal lobes. It does **not** recover a leaf's secondary
and tertiary venation, and on this capture it cannot be made to. That was
measured, not assumed, and the measurements are worth stating because they
are the difference between a result and a plausible-looking artefact.

**The flat image is the wrong picture to look for a vein in.** It is the mean
of twelve light directions, so it cancels shading by construction — and a vein
is visible precisely *because* it is raised. On leaf 3 the flat image yielded
nothing at all.

**This species' surface competes directly with its veins.** *Chenopodium* is
farinose: the upper face is covered in mealy bladder cells, which are a shape
signal in their own right, at a finer scale than a vein but a comparable
amplitude. In a synthetic leaf carrying both, a ridge filter thresholded at
any level returned *more* candidate curves when the veins were removed than
when they were present — the threshold was tracking the texture.

**What survives is the strong stuff.** Reading the crease in the photometric
normals, smoothed past the granules, leaf 3 gives its two genuine basal veins
(reaching 0.85 and 2.35 times the leaf's half-width) plus one false positive
along the margin rim. Two rules remove that last one: a wider exclusion band
inside the margin, and a minimum angle to the local midrib — the rim artefact
came in at 10°, the two real veins at 24° and 39°.

So: **primary veins yes**, when `--photometric` ran. **A venation network no.**

### How to make the network reachable

Both fixes are on the capture side, and neither is something this code can do
for you:

- **Photograph the leaves underside up.** On the abaxial surface the
  secondaries stand proud instead of being buried under the mealy adaxial
  face. A second pass with the leaves turned over would cost one more capture
  and would very likely give the whole network to the same crease detector.
- **Photograph them in transmitted light**, on a backlit panel. This is the
  classical way venation is imaged: the lamina is translucent and the veins
  are not, so the entire network reads as dark lines with no relief needed at
  all. It would not even need the twelve lights.

Reported per vein: the polyline, the midrib station it departs from, its
length, the ridge strength behind it, and its insertion angle — measured
against the *local* midrib tangent rather than the leaf's long axis, because
that is what stays meaningful on a curved leaf.

## What it writes

```text
<workdir>/
  flat.jpg              the crossed-polariser mean, tone-mapped -- what was measured
  leaves.json           per leaf: contour, midrib, keypoints, sizes, veins
  detections.json       every blob found, accepted or rejected, with the reason
  masks/leaf_NNN.png    the full-resolution binary mask, on the leaf's own crop
  alpha/leaf_NNN.png    the matte behind it, for compositing
  normals/leaf_NNN.png  --photometric: a measured OpenGL normal map
  leaves/leaf_NNN.png   --veins: one leaf large, with its veins drawn
  leaf_poses.png        --visualize: the whole result in one diagram
```

Coordinates in `leaves.json` are full-frame pixels. Lengths and areas are in
millimetres when `--marker-mm` was given and pixels otherwise; every record
carries a `units` field, so nothing has to be inferred from context.

## Things it does not do

- **It does not know a leaf from an inflorescence.** The greenness rule asks
  "is this plant tissue", and a flower cluster passes. It is reported like
  everything else and is obvious in the diagram. Dropping it silently would
  be worse.
- **It does not separate overlapping leaves on its own.** That is what
  `--instances sam` is for, and it has not been measured on a capture that
  needs it — the flat lay this was built against has gaps everywhere.
- **It does not measure curl.** Everything here is in the image plane. The
  photometric normals contain the out-of-plane information and nothing yet
  integrates them into a height field.
- **It does not recover a venation network.** See above for why, and for the
  two capture changes that would fix it.
- **It cannot orient a petiole-less leaf with an entire margin.** Every cue it
  has is about the stalk or the tissue beside it. Such leaves are flagged, not
  guessed at; see the removed tooth vote above for what happens when they are.
- **It has been measured on one capture.** `gaensefuss_31`, 2026-09-15: 24
  frames, 61 MP, 27 leaves on black felt. Every threshold quoted above was
  fitted to it. Another backing, another species or another rig may well move
  them, and `detections.json` is where you would see that first.
