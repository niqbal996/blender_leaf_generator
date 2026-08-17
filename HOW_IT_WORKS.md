# How 2D pixels become a 3D coloured plant

This explains the mechanism at the centre of the pipeline: how a label drawn on
a photograph ends up attached to a point in 3D space, and how those labelled
points become individual leaves.

It is written for someone who knows Python but not computer vision. Every
number quoted is measured from `runs/plant_9`, not estimated.

Read `README.md` first for what each phase does. This document is about *why
the 2D-to-3D step works at all*.

---

## 1. There is only one camera

Physically there is one camera. It orbits the plant while the plant sits
still. You record ~650 frames and keep the sharpest 96 (P1).

So "the camera in frame 31" and "the camera in frame 62" are the same physical
camera standing in two different places. **Camera pose** means: where it stood
and which way it pointed, for one frame. 96 frames, 96 poses.

Everything downstream depends on knowing those 96 poses.

---

## 2. How the poses are recovered — P3

This is *structure from motion*. The reasoning:

1. In every photo, find distinctive small spots — a speck of dust on the
   table, a notch on the pliers. These are SIFT features.
2. Match them between photos. The same speck appears in many frames.
3. That produces a large set of geometric constraints. If one speck appears in
   50 photos, then its 3D position and those 50 camera poses are all tied
   together: a ray from each camera through the matched pixel must pass
   through that one place in space.
4. Solve for everything simultaneously — all poses, all speck positions — so
   the rays meet as closely as possible. That final fitting step is *bundle
   adjustment*.

COLMAP does this (`reconstruction.py::build_sparse_reconstruction`). Per frame
it returns a rotation `R` and translation `t`; across all frames it returns one
shared set of lens parameters `K`.

### Why the poses can be trusted

The camera physically moved in a circle, so the recovered positions must lie on
one. Measured on plant_9: they miss their own fitted circle by **0.04%** of the
orbit radius, with out-of-plane scatter of 0.09%.

Nothing in COLMAP's objective knows about turntables or rewards circularity, so
that agreement is independent evidence rather than a restatement of what the
solver was already minimising. It is the check that once caught a silently
wrong reconstruction whose reprojection error looked excellent — see
`DECISIONS.md`, "force a single shared camera".

### What you do not get

**Absolute size.** Structure from motion recovers shape and camera positions
only up to an unknown scale factor. Everything in this pipeline is therefore in
"COLMAP units", and `p6/leaves.json` says so explicitly. Solving real-world
scale needs an extra measurement — see `alignment.py`, not yet wired in.

---

## 3. The three point clouds, and what a surfel is

Three different clouds exist here. Confusing them makes everything else
confusing.

### Sparse cloud — a by-product of P3

The matched specks from step 2, triangulated. ~27,000 points on plant_9. It
only exists where SIFT found something to match, so it covers the dusty table
well and thin foliage badly. It is **not** used for labelling. It is used to
set the bounding box for carving, and its dust speckle is what makes the pose
solve well-conditioned in the first place.

### Visual hull — P4a

Fill space with a grid of small cubes (voxels). Project each cube into all 96
photos. If it lands outside the plant's silhouette in the views that could see
it, delete it. What survives is the largest shape consistent with every
silhouette.

No feature matching involved, which is why this captures thin stems the sparse
cloud misses. But silhouettes cannot see into concavities, so a cupped leaf
carves as if flat, and leaf thickness is overestimated. The hull is used as a
*bound*, not as the final surface.

### The carved surface — P4b — this is the cloud that gets labelled

A **surfel** is a "surface element": a tiny flat disc. Picture a small circular
sticker floating in space, with a centre, a radius, and a direction it faces.

Compare with ordinary 3D Gaussian splatting, which uses fuzzy *ellipsoid blobs*
that fill volume. A disc lies **on** a surface; a blob fills space. Leaves are
thin sheets, so discs describe them and blobs do not — which is why
`surfels.py` calls `rasterization_2dgs` rather than the isotropic version. In
the code each surfel carries `means` (centre), `scales` (radius — only the
first two components are used, and that is what makes it flat), `quats` (which
way it faces), plus opacity and colour.

~150,000 discs are placed on the hull's skin and then optimised until rendering
them reproduces the actual photographs.

**Then the important part:** the 3D points are *not* the disc centres. Depth is
**rendered** from each camera and those depth pixels are converted back into 3D
points (`surfels.py::render_surface_points`). Disc centres sit wherever the
optimiser put them to make the picture look right, which need not be on any
surface. Rendered depth is a real surface crossing.

That gives `p4b/surface.ply` — **117,414 points on plant_9**, each with a
position and a normal. This is the cloud everything below refers to.

---

## 4. Projection: drawing the cloud from a camera's position

Given a camera pose, there is a formula for where any 3D point lands in that
camera's image. Three steps.

Worked example: point 60000 at `[0.212, 1.336, -0.764]`, frame 31, whose pose
and lens come from COLMAP:

```
K  focal length 3328.3 px, image centre (960.0, 540.0)   -- same for all frames
R  3x3 rotation                                          -- this frame only
t  [-0.961, -1.024, 3.521]                               -- this frame only
```

### Step 1 — move the point into the camera's coordinates

```
R @ [0.212, 1.336, -0.764] + t  =  [0.164, -0.111, 4.080]
```

The world origin is arbitrary. This re-expresses the point as *the camera* sees
it: 0.164 to the right, 0.111 up, and **4.080 in front of the lens**. That third
number is the depth. `R` and `t` do nothing but change the frame of reference.

### Step 2 — divide by the depth

```
0.164 / 4.080  =  0.0403
-0.111 / 4.080 = -0.0271
```

This is perspective itself. A point twice as far away has twice the depth, so
it lands half as far from the image centre. This is the only step that is not a
rigid move — it is where 3D collapses to 2D.

### Step 3 — convert to pixels

```
u = 3328.3 * 0.0403  + 960.0 = 1094.0
v = 3328.3 * -0.0271 + 540.0 =  449.6
```

Multiply by focal length to get pixels rather than a ratio, then shift so that
(0, 0) is the image corner instead of the centre.

**Result: pixel (1094, 450)** in a 1920x1080 image. Checking the plant mask of
that photograph at that pixel: it is plant. The arithmetic put the point where
the plant actually is.

### All points at once

In the code this is two lines operating on the whole array — no loop
(`hull.py::CarveCamera.project`, and identically in
`semantic.py::render_points`):

```python
cam    = points_world @ R.T + t                                    # step 1
pixels = (cam[:, :2] / depth[:, None]) @ K[:2, :2].T + K[:2, 2]     # steps 2, 3
```

`points_world` is 117414x3, so `points_world @ R.T` is a single matrix
multiply. Across 96 frames that is ~11 million projections, and it takes
seconds.

**Whole-cloud check on frame 31:** all 117,414 points land inside the frame,
and **87.2% land on plant pixels**. Not 100%, because from any one viewpoint
some points are hidden behind leaves in front of them and some sit just outside
an eroded mask edge. The large majority landing on plant is what confirms the
poses and the projection are correct *together*.

---

## 5. The index map — the trick that carries labels into 3D

Having computed a pixel for every point, write each point's **row number** into
that pixel:

```python
index_map = np.full((height, width), -1, np.int32)
index_map[py, px] = idx          # this pixel was drawn by point idx
```

Points are written far-to-near, so nearer points overwrite farther ones. That
is a correct depth test for opaque points, and it means a pixel always reports
the point actually facing the camera.

So now there are **two images of the same size, pixel-for-pixel aligned**:

| image | each pixel holds |
|---|---|
| the class map from P4c stage 1 | what DINOv3 called that pixel: leaf / stem / root |
| the index map | which 3D point drew that pixel, or -1 |

They line up because the render used the *same pose and same lens* as the real
photograph.

`index_map[450, 1094] = 60000` means "point 60000 is what you see there". And
DINOv3 labelled that same pixel in that same photo. Therefore point 60000 gets
one vote for that label.

An explicit index map is used rather than "render a surface, then find the
nearest cloud point" because the latter reintroduces an association error
exactly in the sparse regions where small leaves live.

---

## 6. Labelling the points — P4c

### Stage 1: label the pixels (`cli/classify.py`, `dino.py`)

Crop each frame to the plant, zero the background, resize to 896x896. DINOv3
returns a 56x56 grid of feature vectors, one per 16x16 patch. Each patch is
compared to your hand-clicked seed vectors by cosine similarity and takes the
nearest seed's label (`dino.py::classify_frame`). Upsampled to pixels, -1
outside the plant mask.

Written as `p4c/class_maps/frame_XXXX.png`: `0` = not plant, `i+1` = class `i`,
with the class order recorded in `p4c/classify.json`.

Seed vectors are kept **individually**, never averaged into one prototype per
class. Averaging several different-looking leaves produces a generic direction
resembling no individual leaf, while a class with one tight example keeps full
similarity and wins patches it should not.

### Stage 2: vote them onto the points (`cli/fuse.py`, `semantic.py`)

For each of the 96 views:

```python
class_map            = load_class_map(...)                        # 2D labels
_rgb, index_map      = render_points(points, colors, camera)      # section 5
weights              = view_weights(points, normals, camera)      # see below
cast_votes(tally, index_map, class_map, weights)
```

and `cast_votes` is just the lookup:

```python
valid   = index_map >= 0
indices = index_map[valid]     # the 3D points
classes = class_map[valid]     # what the photograph called them
np.add.at(tally["weight"], (indices, classes), per_pixel)
```

**Views are weighted, not required to agree.** A 2D segmenter recognises a leaf
when its blade faces the camera and mistakes it for a stem when it turns
edge-on. Over a full orbit that happens to every leaf, so demanding agreement
asks the capture for something its geometry forbids. Each vote is scaled by
`|cos|` between the point's normal and the ray to that camera — the
foreshortening factor, i.e. how much of that surface the camera actually sees.
Face-on views carry full weight, grazing views approach zero and abstain on
their own. No threshold and no exponent.

Measured on plant_9: **84.5% of the cloud has face-on views in the minority**,
so plain majority counting was deciding most of the plant from cameras that
could not see the surface they were voting on.

### The output

`finalise_votes` takes the winning class per point. Result: **`p4c/labels.npy`,
one label per point, in the same row order as `p4b/surface.ply`.** That row
alignment is the contract everything downstream relies on; `cli/structure.py`
checks the lengths match and refuses to run if they do not.

Two real points from plant_9:

| point | pixel observations | votes | label | confidence |
|---|---|---|---|---|
| 73391 | 326 | root 322, leaf 4 | `root` | 0.99 |
| 64722 | 46 | leaf 26, leaf tip 10, stem 10 | `leaf` | **0.36** |

Point 64722 is a point the views genuinely argued over. Its mean facing was
0.52, so it was often seen at an angle. The confidence is kept per point,
in `p4c/votes.npz`, precisely so such points can be distrusted rather than
silently treated as certain.

---

## 7. From organ type to individual leaves — P5

`labels.npy` says *leaf*, not *which* leaf. Splitting them is P5
(`structure_labels.py`):

1. Take the leaf-labelled points and build a nearest-neighbour graph
   (`leaf_graph`).
2. Measure distance from the stem **through the tissue**, not straight through
   the air (`depth_from_stem`). This is a geodesic distance over that graph.
3. Find the peaks of that distance that survive all the way down to the stem
   before merging with another peak (`tip_persistence`). Those are leaf tips.
   Two bumps on one blade join high up on that blade; two genuinely different
   leaves can only join by descending to where they both meet the stem.
4. Grow inward from each tip. Every leaf point joins whichever tip it reaches
   first (`grow_from_tips`). Growth cannot travel *through* the stem contact
   points, so one leaf's territory stops where the leaf does.

Output `p5/leaf_points.npy` — an instance id per leaf point. That is what gets
coloured per leaf.

The midrib of each leaf is then the centroid of successive shells of geodesic
distance from its tip, carried on past the blade to the fork where it leaves
the stem (`leaf_midrib`, `trunk_and_attachments`).

---

## 8. Which file holds what

| file | coloured by | written by |
|---|---|---|
| `p4b/surface.ply` | nothing — positions + normals | P4b |
| `p4c/class_maps/*.png` | 2D per-pixel organ labels | P4c stage 1 |
| `p4c/labels.npy` | one organ label per 3D point | P4c stage 2 |
| `p4c/labels_vis.ply` | **organ** — leaf red, stem violet, root orange | P4c stage 2 |
| `p4c/confidence.ply` | how sure the vote was | P4c stage 2 |
| `p5/leaf_points.npy` | **instance id** per leaf point | P5 |
| `p5/structure.ply` | **instance** — each leaf its own colour | P5 |

`scripts/blender_view_plant.py` loads the P5 outputs and builds them into a
Blender scene, including one point-cloud object per leaf instance so a single
leaf can be isolated.

---

## Why this way round

The obvious alternative is to classify the 3D points directly from their
geometry. It does not work here, and the reason is worth keeping: a small apex
or basal leaf has too few 3D points to be distinguishable from noise by any
method that decides leaf-versus-stem from 3D shape alone. In a photograph that
same leaf is still obviously a leaf.

So the decision is made in 2D, where the evidence is, and carried back into 3D
by projection and voting. Measured alternatives that were tried and rejected —
2D mask elongation, monocular depth flatness, text-prompted detection — are
recorded in `dino.py`'s module docstring with the numbers that killed them.
