"""Can DINOv3 tell two *touching leaves* apart? A bench for that one question.

`dinov3_organ_lab.py` asks what organ a patch is -- leaf, stem or root. This
asks the next question down, which is the one P5 actually fails at: given that
two patches are both leaf, are they the **same leaf**?

That distinction matters because it is the only one carving destroys. When two
blades touch, the carved surface fuses them into one sheet, and no amount of
graph work on that sheet separates them again -- the evidence is not in it. In
a photograph the two blades are still two objects, with an occlusion edge and
a shading step between them. So the question is not whether a better 3D
algorithm exists. It is whether the *images* still hold the answer, and DINOv3
features are the cheapest way to ask: P4c computes them already, they need no
training, and they carry no threshold of their own.

Nothing here writes into the pipeline. It reads a finished run and produces
pictures and numbers.


================================ WALKTHROUGH =================================

  W=/mnt/d/Turn_table_plant_scans/09-09-2026-Naeem/sugarbeet_1/plant

------------------------------------------------------------------------------
STEP 1  --mode probe        "show me the boundary between these two leaves"
------------------------------------------------------------------------------

    python scripts/leaf_identity_lab.py --mode probe --workdir $W --stride 8

  A window opens. Left half is the plant; right half is empty until you have
  placed two seeds.

  DO: click once in the middle of one leaf, then once on a DIFFERENT leaf that
  touches or overlaps it. Pick the worst pair you have -- two blades crossing
  in the heart of the plant, not two that are obviously far apart.

  THEN GIVE EACH LEAF MORE THAN ONE EXAMPLE. This is the single most useful
  thing you can do here and it is easy to skip. Press `n` a few frames on,
  press `1` to make leaf A active again, and click that same blade where it
  now sits. Press `2`, click leaf B. Keep going round the turn.

  A leaf seen from one angle is one example of that leaf; from three angles it
  is three. They are kept separate and scored by whichever fits best, never
  averaged -- exactly the choice pose-pick-seeds makes for organ class. A
  single example describes the leaf *as it looked in that one view*, which is
  why a mask that is crisp where you clicked goes patchy a few frames later.

  Measured on sugarbeet_1 at --size 1792, scoring against P5's projected leaf
  instances on held-out frames:

      seeded from 1 view    mean IoU 0.217
      seeded from 2 views              0.224
      seeded from 3 views              0.312
      seeded from 5 views              0.382

  Monotonic, and not flattening by 5. So when a mask goes patchy later in the
  turn, the move is another example, not another parameter.

  SEE: the right half fills in immediately. Every patch of the plant is
  painted with whichever seed it resembles more. The header line reports the
  share each seed claimed, the median decision margin, and what percentage of
  patches were near-tied.

  LOOK FOR: does the colour change exactly where the two blades actually meet?
  That is the whole experiment. A boundary on the leaf edge means the features
  can see a distinction the 3D cloud has thrown away. Colour bleeding across
  the join means they cannot.

  THEN press `m`. The right half becomes the decision margin: how far ahead
  the winning seed was, per pixel. Read the colours off the bar drawn in the
  corner -- the scale is FIXED at 0..--margin-scale (0.25 by default), not
  stretched to each frame, so the same colour means the same thing in every
  frame and you can compare them:

      dark blue   ~0.00   the two seeds tied. A coin toss.
      cyan/green  ~0.10   a real but ordinary preference.
      red         0.25+   confident.

  Expect the crown and the petioles to be blue: they genuinely are ambiguous,
  because they look alike on every leaf. Expect blue rings around dew drops
  and specular highlights, for the same reason. What matters is the colour
  along the JOIN between your two blades -- a tidy boundary drawn through a
  blue band is one you should not trust however clean it looks, and that is
  the whole reason this panel exists. An argmax picture always looks decisive.

  THEN press `n`. This is the part worth the trouble. Seeds are stored as
  FEATURE VECTORS, not pixels, so moving to the next frame re-decides that
  frame using the same two seeds -- no re-clicking. Walk the turntable with
  `n` and watch whether the split survives the rotation. Any pipeline use of
  this depends on identity holding up across views, and a single frame cannot
  tell you whether it does.

  On a frame you did not click, there are no seed markers to read the colours
  off, so the status bar carries a SWATCH per seed with its name and the share
  of the plant it claimed. Judge the next frame on three things:

    1. does each blade still come out one colour, or has it gone patchy?
    2. did a share collapse -- "A 95%  B 5%" means one seed has swallowed the
       plant and the split has failed, however plausible the picture looks;
    3. did the median margin drop and the tied % climb? That is the split
       dissolving before it visibly breaks, and it is the early warning.

  Keys:  click     add an example to the ACTIVE leaf
         1-8       make that leaf active (its swatch is outlined white)
         0         next click starts a NEW leaf
         v         cycle DECISION / MARGIN / CONFIDENCE
         u  c      undo last example / clear everything
         n  p      next / previous frame
         s  q      save this frame / quit

  The third view, CONFIDENCE, is how well the winning leaf matched at all,
  rather than how far it won by. They are different failures: a low margin
  means two leaves both fit, a low confidence means NEITHER does, which is
  what an occluded or absent leaf looks like -- the argmax still has to hand
  every pixel to somebody. Read it where a leaf is hidden behind another
  leaf's petiole.

  Do NOT use its per-frame median as a quality gate. That was tried: across
  held-out frames it correlates with actual IoU at only +0.12, because every
  pixel on the plant really does look like some sugarbeet leaf, so confidence
  stays around 0.8 whether the frame worked or not. It is a map of WHERE the
  match failed, not a score for WHETHER this frame is trustworthy.

  `s` writes the panel as a .jpg and the equivalent typed command to
  seeds.txt, so a placement worth keeping can be replayed exactly:

      python scripts/leaf_identity_lab.py --mode probe --workdir $W \\
          --frames 53 --seeds "A:990,300" "B:1300,730"

  Passing --seeds skips the window entirely, which is also the fallback on a
  machine with no display. To read coordinates off a grid for that, use
  --mode reference (below).

  SHARPNESS. The decision is made per pixel, from each seed's score map
  interpolated to full resolution -- not by upsampling a patch-grid argmax --
  so boundaries land sub-patch and follow the leaf edge. On top of that,
  --size sets how fine the features themselves are: it is the square the crop
  is resized to before DINOv3 tiles it into 16px patches.

      --size  896  (default)   56x56 patches    ~7 s/frame
      --size 1344              84x84 patches    ~7 s/frame
      --size 1792             112x112 patches   ~8 s/frame

  Higher is not automatically better, and it is worth looking rather than
  assuming. On sugarbeet_1 frame 53, 1792 traces the serrated leaf margin
  visibly more precisely than 896 -- and also starts assigning the shaded,
  rolled rim of a blade to the other seed, because at that scale the rim
  really does look different from the sunlit middle of its own leaf. Sharper
  outline, more within-leaf noise. Try both on your worst pair.

------------------------------------------------------------------------------
STEP 2  --mode split        "and without me pointing at anything?"
------------------------------------------------------------------------------

    python scripts/leaf_identity_lab.py --mode split --workdir $W \\
        --frames 53 --k-sweep 2,3,4,5,6

  DO: nothing. No clicks. It clusters the plant's patches by feature alone.

  SEE: one .jpg per frame, the original followed by one panel per K.

  LOOK FOR: whether a cluster is ONE blade or SEVERAL. Boundaries that land on
  leaf edges mean the features are spatially sharp. Clusters that each contain
  two or three whole leaves mean the features group by appearance and pose --
  "blade facing this way" -- and not by identity.

  On sugarbeet_1 that is exactly what happens: the edges are crisp, but at
  k=5 the two touching right-hand blades land in the SAME cluster. Which is
  the useful negative -- unsupervised clustering will not do this job, and
  step 1 works only because it is contrastive and you supplied the contrast.

------------------------------------------------------------------------------
STEP 3  --mode affinity     "does this agree with what P5 got wrong?"
------------------------------------------------------------------------------

    python scripts/leaf_identity_lab.py --mode affinity --workdir $W --stride 6

  DO: nothing, but it needs a finished P5 and it takes a few minutes. It
  projects each P5 leaf instance into every view and samples the features
  where that leaf is actually visible.

  SEE: two tables and a control.

    POOLED       averages each leaf's patches into one vector per view.
    PATCH-LEVEL  does not pool: for each patch of leaf i, finds the most
                 similar patch on any OTHER leaf, and tallies which leaf.
    CONTROL      the same leaf seen from two different views. This is the
                 ceiling. Any "these two leaves are alike" score has to be
                 read against how alike one leaf is to itself once the camera
                 has moved, or it means nothing.

  LOOK FOR: a mutual first choice in the patch table -- leaf i's tissue picks
  leaf j out of the whole plant, and leaf j picks i back. That is what one
  blade cut into two instances looks like.

  Expect the two tables to DISAGREE, and believe the patch one. See below.


========================= MEASURED ON sugarbeet_1 ============================

Ground truth from the operator: P5's leaf 3 and leaf 4 are two halves of ONE
physical blade, and a detached petiole with no blade was absorbed into leaf 3.

  Pooled        leaf 3 + leaf 4 scored 0.879, against a 0.825 control and a
                0.797 median -- and at --stride 8 it ranked leaf 0 + leaf 1
                top instead. A statistic whose winner changes with the stride
                is noise. Averaging a leaf's patches answers "what kind of
                thing is this", and for five blades of one sugarbeet that is
                the same answer five times.

  Patch-level   90% of leaf 4's patches pick leaf 3; leaf 3 picks leaf 4 back.
                The merge is visible with no threshold anywhere. It also makes
                leaf 0 and leaf 1 a mutual pair, more weakly, so this is
                evidence to weigh rather than a verdict to act on blindly.

  Probe         one click per blade separates the two touching right-hand
                leaves cleanly, boundary on the occlusion edge, median margin
                0.109 with 8% of patches tied.

  So the identity signal is real, and POOLING IS WHAT DESTROYS IT. Anything
  built on this should compare patches against patches, never leaf averages.

WHERE THIS METHOD RUNS OUT

  Multi-view seeding lifts held-out IoU from 0.217 to 0.382 and was still
  climbing at five views -- a real gain, and not enough to build a pipeline
  stage on. Nor is there a cheap gate: confidence does not tell the good
  frames from the bad ones (+0.12), so "reject the doubtful views and vote the
  rest" is not available from this signal.

  The structural reason is that every frame here is decided ALONE. Nothing
  carries forward, so a blade that disappears behind a petiole for four frames
  comes back with no memory of having been that leaf -- which is exactly the
  reported symptom of masks going patchy and then recovering.

  Fixing that needs temporal state, not a better per-frame descriptor: a video
  model propagating each leaf through the turn, which P2 already does for the
  plant mask, or cross-view voting, which pose-tips already does for tips. The
  value of this bench is that it says so with numbers, and cheaply, before
  anyone plumbs a per-frame descriptor into P5.


=============================== ALSO ========================================

  --mode reference   one frame with a coordinate grid burned in, for reading
                     --seeds off when there is no display to click on.

  Model access: facebook/dinov3-* is gated. If it is already in your HF cache
  -- true on any machine that has run P4c -- it loads from there even when the
  hub refuses. Otherwise accept the licence and `hf auth login`, or use the
  ungated --model facebook/dinov2-base, which behaves the same way here.

  --mode affinity imports pycolmap, so it wants the CUDA runtime on
  LD_LIBRARY_PATH the way the pipeline does. See setup_env.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dinov3_organ_lab import Backbone, patch_index  # noqa: E402

# Distinct, colour-blind-safe enough to read as *identities* rather than as a
# scale -- these label leaves, and nothing here is ordered.
INSTANCE_COLORS = [
    (60, 180, 250), (80, 200, 90), (230, 120, 60), (200, 80, 220),
    (60, 220, 220), (240, 160, 80), (150, 150, 250), (120, 220, 160),
]


# --------------------------------------------------------------------------
# Reading a finished run
# --------------------------------------------------------------------------


def frame_paths(workdir: Path) -> List[Path]:
    frames = sorted((workdir / "p1" / "frames").glob("*.jpg"))
    if not frames:
        frames = sorted((workdir / "p1" / "frames").glob("*.png"))
    if not frames:
        raise SystemExit(f"no frames in {workdir / 'p1' / 'frames'}")
    return frames


def plant_mask(workdir: Path, stem: str, shape) -> Optional[np.ndarray]:
    path = workdir / "p2" / "masks" / "plant" / f"{stem}.png"
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[:2] != shape[:2]:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def crop_box(mask: np.ndarray, shape, pad: int) -> Tuple[int, int, int, int]:
    """The plant's bounding box, padded and clipped. Returned, not applied.

    `dinov3_organ_lab.crop_to_plant` throws the offsets away, and every mode
    here has to map a full-frame pixel onto the patch grid, so the box itself
    is the thing that has to survive.

    The crop is not cosmetic. DINOv3 tiles an image into 16px patches; the
    plant is a small share of these frames, so uncropped it lands on a handful
    of patches and two neighbouring blades fall inside the same one. There is
    no feature quality that recovers a distinction the tiling already erased.
    """
    ys, xs = np.nonzero(mask)
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(shape[1], int(xs.max()) + pad), min(shape[0], int(ys.max()) + pad))


def leaf_owner_per_cloud_point(workdir: Path) -> Tuple[np.ndarray, int]:
    """Which P5 leaf owns each point of the full cloud (-1 = none).

    P5 writes `leaf_points.npy` over the leaf-labelled subset in order, not
    over the whole cloud, so the owner ids have to be scattered back onto the
    full cloud before an index map can be read through them.
    """
    labels = np.load(workdir / "p4c" / "labels.npy")
    with open(workdir / "p4c" / "classify.json") as f:
        order = json.load(f)["class_order"]
    leaf_class = order.index("leaf")
    owner_of_leaf = np.load(workdir / "p5" / "leaf_points.npy")

    leaf_index = np.nonzero(labels == leaf_class)[0]
    if len(leaf_index) != len(owner_of_leaf):
        raise SystemExit(
            f"p5/leaf_points.npy has {len(owner_of_leaf)} entries but p4c/labels.npy "
            f"marks {len(leaf_index)} leaf points -- P5 and P4c are out of step, re-run P5")
    owner = np.full(len(labels), -1, np.int64)
    owner[leaf_index] = owner_of_leaf
    return owner, int(owner_of_leaf.max()) + 1


def load_cloud(workdir: Path):
    from pose_estimator.ply_io import read_ply_vertices

    surface = workdir / "p4b" / "surface.ply"
    path = surface if surface.exists() else workdir / "p4" / "hull_points.ply"
    fields = read_ply_vertices(path)
    return np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)


# --------------------------------------------------------------------------
# Feature helpers
# --------------------------------------------------------------------------


def patch_ids_for_crop(shape, grid) -> np.ndarray:
    """A (h, w) image where each pixel holds the flat patch index covering it.

    Built once per view, then read wherever a pixel needs its feature: the
    alternative is calling `patch_index` per point, which for a leaf of 39k
    points is 39k Python calls.
    """
    h, w = shape[:2]
    gy = np.minimum((np.arange(h) / h * grid[0]).astype(np.int64), grid[0] - 1)
    gx = np.minimum((np.arange(w) / w * grid[1]).astype(np.int64), grid[1] - 1)
    return gy[:, None] * grid[1] + gx[None, :]


def spherical_kmeans(x: np.ndarray, k: int, iters: int = 40, seed: int = 0):
    """k-means on the unit sphere -- cosine distance, which is how DINO features compare.

    Written out rather than imported: sklearn is not a dependency of this repo
    and one bench script is not a reason to make it one. Seeded by k-means++
    so a run is reproducible and does not collapse onto duplicate centres.
    """
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(len(x))]]
    for _ in range(k - 1):
        d = 1.0 - (x @ np.stack(centres).T).max(axis=1)
        d = np.clip(d, 0, None)
        total = d.sum()
        centres.append(x[rng.choice(len(x), p=d / total) if total > 0
                        else rng.integers(len(x))])
    centres = np.stack(centres)

    label = np.zeros(len(x), np.int64)
    for _ in range(iters):
        new = (x @ centres.T).argmax(axis=1)
        if (new == label).all():
            break
        label = new
        for c in range(k):
            member = x[label == c]
            if len(member):
                v = member.mean(axis=0)
                centres[c] = v / max(np.linalg.norm(v), 1e-9)
    return label, centres


def colour_regions(view, region, plant, count) -> np.ndarray:
    out = view.copy()
    for c in range(count):
        hit = (region == c) & plant
        if hit.any():
            colour = np.array(INSTANCE_COLORS[c % len(INSTANCE_COLORS)])
            out[hit] = (0.45 * out[hit] + 0.55 * colour).astype(np.uint8)
    out[~plant] = (view[~plant] * 0.25).astype(np.uint8)
    return out


def banner(panel, text) -> None:
    for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(panel, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, thick)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def mode_reference(args, workdir: Path, out: Path) -> None:
    """One frame with a grid burned in, so --seeds can be read off it."""
    for path in pick_frames(args, workdir):
        bgr = cv2.imread(str(path))
        mask = plant_mask(workdir, path.stem, bgr.shape)
        if mask is None:
            x0, y0, view = 0, 0, bgr
        else:
            x0, y0, x1, y1 = crop_box(mask, bgr.shape, args.pad)
            view = bgr[y0:y1, x0:x1]
        grid_img = view.copy()
        step = 100
        for x in range(0, view.shape[1], step):
            cv2.line(grid_img, (x, 0), (x, view.shape[0]), (0, 255, 255), 1)
            cv2.putText(grid_img, str(x + x0), (x + 3, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
        for y in range(0, view.shape[0], step):
            cv2.line(grid_img, (0, y), (view.shape[1], y), (0, 255, 255), 1)
            cv2.putText(grid_img, str(y + y0), (4, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
        path_out = out / f"reference_{path.stem}.jpg"
        cv2.imwrite(str(path_out), grid_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {path_out}")
    print("\nRead --seeds coordinates off these. They are FULL-FRAME pixels: the grid "
          "labels already add the crop offset back on, so what you read is what you pass.")


SEED_NAMES = "ABCDEFGH"

HELP_LINES = [
    "click add example to active leaf   1-8 pick leaf   0 start a new one",
    "n/p frame   v cycle DECISION/MARGIN/CONFIDENCE   u undo   c clear",
    "s save this frame                  q quit",
]
VIEWS = ("DECISION", "MARGIN", "CONFIDENCE")
# Cosine similarity to the best exemplar. Below ~0.35 nothing on the plant
# resembles any seed; above ~0.85 is as good a match as this backbone gives.
CONFIDENCE_RANGE = (0.35, 0.85)


def prepare_view(workdir: Path, path: Path, backbone: Backbone, pad: int):
    """Crop, mask and featurise one frame. The slow step, so it is cached."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    mask = plant_mask(workdir, path.stem, bgr.shape)
    if mask is None:
        return None
    x0, y0, x1, y1 = crop_box(mask, bgr.shape, pad)
    view, plant = bgr[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
    view[~plant] = 0
    feats, grid = backbone.features(view)
    inside = cv2.resize(plant.astype(np.uint8), (grid[1], grid[0]),
                        interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
    return {"stem": path.stem, "view": view, "plant": plant, "feats": feats,
            "grid": grid, "inside": inside, "origin": (x0, y0)}


def decide(frame, groups: List[np.ndarray]):
    """Per-PIXEL decision, from interpolated scores -- not an upsampled argmax.

    Deciding on the 56x56 patch grid and then blowing the labels up nearest-
    neighbour quantises every boundary to a ~23px block, which is what made
    the masks look blocky. The blockiness was in the *decision*, not the
    features: it threw away where inside a patch the two seeds actually
    crossed over.

    Interpolating each seed's score map to full resolution first and taking
    the argmax per pixel puts the boundary where the evidence changes hands,
    which lands sub-patch and follows the leaf edge. It costs one resize per
    seed and no extra model time -- the features are unchanged, only what is
    done with them.

    Raising --size still helps on top of this, because it gives the features
    themselves a finer grid. This just stops the drawing from being coarser
    than the features already are.
    """
    shape = frame["view"].shape
    maps = []
    for vectors in groups:
        # Best-matching EXEMPLAR, not their average. A leaf seen from one angle
        # is one example of that leaf; from three angles it is three, and
        # keeping them separate widens the description instead of blurring it.
        # This is the same choice P4c makes for organ class -- see
        # pose-pick-seeds -- and the reason a single-exemplar seed stops
        # matching its own leaf a few frames into the turn.
        score = (frame["feats"] @ np.atleast_2d(vectors).T).max(axis=1)
        maps.append(cv2.resize(score.reshape(frame["grid"]).astype(np.float32),
                               (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC))
    maps = np.stack(maps, axis=-1)
    region = maps.argmax(axis=-1).astype(np.uint8)
    # How well the winner matched at all, as opposed to how far it won by.
    # These are different failures and they need telling apart: a tie means
    # two seeds both fit, a low best means NEITHER does -- which is what an
    # occluded or absent leaf looks like, because an argmax must still hand
    # every pixel to somebody.
    confidence = maps.max(axis=-1)
    if len(groups) > 1:
        part = np.partition(maps, -2, axis=-1)
        margin = part[..., -1] - part[..., -2]
    else:
        margin = np.zeros(shape[:2], np.float32)
    return region, margin, confidence


def scalar_overlay(bgr, value, plant, lo: float, hi: float):
    """Margin on a FIXED colour scale, so frames can be compared to each other.

    `heat_overlay` normalises to whatever it was handed, which is right for a
    similarity map read on its own and wrong here: it re-scales per frame, so
    pressing `n` silently changes what red means and the next frame cannot be
    compared with the last. The margin is a difference of cosine similarities
    and already lives in fixed units, so it is pinned to 0..full_scale and a
    colour bar is drawn to say so.
    """
    norm = np.clip((value - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out = (0.45 * bgr + 0.55 * heat).astype(np.uint8)
    out[~plant] = (bgr[~plant] * 0.25).astype(np.uint8)

    bar_w, bar_h, pad = 26, min(260, out.shape[0] // 3), 16
    ramp = cv2.applyColorMap(
        np.repeat(np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None], bar_w, axis=1),
        cv2.COLORMAP_TURBO)
    y, x = pad + 18, out.shape[1] - bar_w - pad
    out[y:y + bar_h, x:x + bar_w] = ramp
    cv2.rectangle(out, (x, y), (x + bar_w, y + bar_h), (255, 255, 255), 1)
    for text, ty in ((f"{hi:.2f}+", y - 6), (f"{lo:.2f}", y + bar_h + 16)):
        for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(out, text, (x - 34, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colour, thick)
    return out


def draw_seed_markers(canvas, groups, frame) -> None:
    """Mark examples clicked on THIS frame; one from another view has no pixel here."""
    for group in groups:
        for shot in group["shots"]:
            if shot["stem"] != frame["stem"]:
                continue
            centre = shot["xy"]
            cv2.drawMarker(canvas, centre, (255, 255, 255), cv2.MARKER_CROSS, 30, 4)
            cv2.drawMarker(canvas, centre, (0, 0, 0), cv2.MARKER_CROSS, 30, 2)
            for colour, thick in (((255, 255, 255), 4), ((0, 0, 0), 1)):
                cv2.putText(canvas, group["name"], (centre[0] + 16, centre[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, thick)


def interactive_probe(args, workdir: Path, out: Path, backbone: Backbone) -> None:
    """Click seeds and watch the split redraw. The clicking is the point.

    Typing coordinates read off a grid is slow enough that nobody tries the
    fourth or fifth placement, and the fourth is usually the informative one --
    whether the boundary holds when the seed sits near the leaf edge rather
    than in its comfortable middle.

    A seed is stored as its FEATURE VECTOR, not as a pixel. So n/p carry the
    seeds onto the next frame and re-decide there, which turns the picker into
    the view-invariance test: click two touching blades once, then walk the
    turntable and watch whether the split survives the rotation. That is the
    property any pipeline use of this would depend on, and it is the one thing
    a single-frame picture cannot tell you.
    """
    from pose_estimator.seed_picker import display_available

    ok, why = display_available()
    if not ok:
        raise SystemExit(
            f"no display available for the interactive picker ({why}).\n"
            f"Fall back to typed coordinates:\n"
            f"  python scripts/leaf_identity_lab.py --mode reference --workdir {workdir} "
            f"--frames 53\n"
            f"then pass --seeds \"name:x,y\" read off that grid.")

    paths = pick_frames(args, workdir)
    cache: dict = {}
    # One group per leaf, each holding as many exemplars as you care to click.
    groups: List[dict] = []
    active = None                       # None = the next click starts a new leaf
    index, view_mode = 0, 0
    message = "click a point on one leaf, then on another"
    window = "leaf identity -- click two touching leaves"
    state = {"click": None}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print(f"  {len(paths)} frame(s). " + "  |  ".join(HELP_LINES))

    try:
        while True:
            if index not in cache:
                frame = prepare_view(workdir, paths[index], backbone, args.pad)
                if frame is None:
                    raise SystemExit(f"could not read {paths[index].name} or its plant mask")
                cache[index] = frame
            frame = cache[index]
            view, plant = frame["view"], frame["plant"]
            scale = min(1.0, args.max_display / max(view.shape[:2]))

            if state["click"] is not None:
                cx, cy = state["click"]
                state["click"] = None
                px, py = int(round(cx / scale)), int(round(cy / scale))
                # A seed off the plant is a seed on the black background, and
                # every background patch would then answer to it.
                if not (0 <= px < view.shape[1] and 0 <= py < view.shape[0]
                        and plant[py, px]):
                    message = "! that click missed the plant"
                elif active is None and len(groups) >= len(SEED_NAMES):
                    message = f"! {len(SEED_NAMES)} leaves is plenty; u to undo"
                else:
                    if active is None:
                        groups.append({"name": SEED_NAMES[len(groups)], "shots": []})
                        active = len(groups) - 1
                    group = groups[active]
                    group["shots"].append({
                        "stem": frame["stem"], "xy": (px, py),
                        "vector": frame["feats"][patch_index((px, py), view.shape,
                                                             frame["grid"])],
                        "full_xy": (px + frame["origin"][0], py + frame["origin"][1]),
                    })
                    message = (f"leaf {group['name']}: {len(group['shots'])} example(s)"
                               f"  [0 = start a new leaf]")

            left = view.copy()
            draw_seed_markers(left, groups, frame)
            if len(groups) >= 2:
                region, margin, confidence = decide(
                    frame, [np.stack([e["vector"] for e in g["shots"]]) for g in groups])
                if VIEWS[view_mode] == "MARGIN":
                    right = scalar_overlay(view, margin, plant, 0.0, args.margin_scale)
                elif VIEWS[view_mode] == "CONFIDENCE":
                    right = scalar_overlay(view, confidence, plant, *CONFIDENCE_RANGE)
                else:
                    right = colour_regions(view, region, plant, len(groups))
                draw_seed_markers(right, groups, frame)
                # Over plant PIXELS, matching what is drawn. The old figure was
                # over patches, so it disagreed with the picture it captioned.
                stat = (f"margin {np.median(margin[plant]):.3f}"
                        f"  tied {100 * (margin[plant] < 0.02).mean():.0f}%"
                        f"  confidence {np.median(confidence[plant]):.3f}"
                        f"  unlike-any {100 * (confidence[plant] < CONFIDENCE_RANGE[0]).mean():.0f}%")
                shares = [float((region[plant] == i).mean()) for i in range(len(groups))]
            else:
                right, shares = view.copy(), []
                stat = f"{len(groups)} leaf/leaves -- two are needed before anything is decided"

            panel = np.hstack([left, right])
            if scale < 1.0:
                panel = cv2.resize(panel, (int(panel.shape[1] * scale),
                                           int(panel.shape[0] * scale)))
            bar = np.zeros((126, panel.shape[1], 3), np.uint8)
            shots = [e for g in groups for e in g["shots"]]
            seeded = ", ".join(sorted({e["stem"] for e in shots}))
            head = (f"{frame['stem']}  ({index + 1}/{len(paths)})   {VIEWS[view_mode]}"
                    + (f"   examples from {seeded}" if shots else "") + f"   {stat}")
            cv2.putText(bar, head, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            # A swatch per leaf. On a frame nobody clicked on there are no
            # markers to read the colours off, which is exactly the frame you
            # are trying to judge after pressing `n`. The count says how many
            # views that leaf has been shown from -- the number to grow when a
            # mask goes patchy later in the turn.
            x = 12
            for i, group in enumerate(groups):
                colour = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
                cv2.rectangle(bar, (x, 32), (x + 22, 50), colour, -1)
                cv2.rectangle(bar, (x, 32), (x + 22, 50),
                              (255, 255, 255) if i == active else (110, 110, 110),
                              3 if i == active else 1)
                text = (f"{i + 1}:{group['name']}x{len(group['shots'])}"
                        + (f" {100 * shares[i]:.0f}%" if shares else ""))
                cv2.putText(bar, text, (x + 28, 47), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (230, 230, 230), 1)
                x += 34 + 13 * len(text)
            cv2.putText(bar, message, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 220, 255) if message.startswith("!") else (150, 255, 150), 1)
            for row, line in enumerate(HELP_LINES[:2]):
                cv2.putText(bar, line, (12, 94 + row * 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.44, (170, 170, 170), 1)
            cv2.imshow(window, np.vstack([panel, bar]))

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u"):
                if not groups:
                    message = "! nothing to undo"
                else:
                    group = groups[active if active is not None else -1]
                    group["shots"].pop()
                    if not group["shots"]:
                        groups.remove(group)
                        active = None
                    message = "removed the last example"
            elif key == ord("c"):
                message, groups, active = f"cleared {len(groups)} leaf/leaves", [], None
            elif key in (ord("v"), ord("m")):
                view_mode = (view_mode + 1) % len(VIEWS)
            elif key == ord("0"):
                active, message = None, "next click starts a new leaf"
            elif ord("1") <= key <= ord("8"):
                want = key - ord("1")
                if want < len(groups):
                    active = want
                    message = (f"adding examples to leaf {groups[want]['name']} "
                               f"({len(groups[want]['shots'])} so far)")
                else:
                    message = f"! there is no leaf {want + 1} yet"
            elif key in (ord("n"), 83):
                index, message = min(index + 1, len(paths) - 1), ""
            elif key in (ord("p"), 81):
                index, message = max(index - 1, 0), ""
            elif key in (ord("s"), 13, 10):
                if len(groups) < 2:
                    message = "! nothing to save until two leaves are seeded"
                    continue
                stamp = out / f"{frame['stem']}_probe.jpg"
                cv2.imwrite(str(stamp), np.vstack([panel, bar]),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                # The equivalent typed command. Repeating a name is how the
                # typed form expresses several examples of one leaf, so a
                # multi-view placement replays exactly as it was clicked.
                spec = " ".join(f'"{g["name"]}:{e["full_xy"][0]},{e["full_xy"][1]}"'
                                for g in groups for e in g["shots"])
                with open(out / "seeds.txt", "w") as f:
                    f.write(f"--seeds {spec}\n")
                message = f"saved {stamp.name} + seeds.txt"
                print(f"  {stamp}\n  replay: --seeds {spec}")
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)

    print(f"\nwrote {out}/")


def mode_probe(args, workdir: Path, out: Path, backbone: Backbone) -> None:
    """Two or more clicked points; every patch goes to whichever it resembles more."""
    seeds = parse_seeds(args.seeds)

    for path in pick_frames(args, workdir):
        bgr = cv2.imread(str(path))
        mask = plant_mask(workdir, path.stem, bgr.shape)
        if mask is None:
            print(f"  {path.stem}: no plant mask -- skipped")
            continue
        x0, y0, x1, y1 = crop_box(mask, bgr.shape, args.pad)
        view, plant = bgr[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
        view[~plant] = 0

        feats, grid = backbone.features(view)
        # A name repeated is several examples of the SAME leaf, kept separate
        # and scored by whichever fits best -- the typed form of clicking one
        # leaf on several frames.
        names = list(dict.fromkeys(n for n, _ in seeds))
        by_name: dict = {n: [] for n in names}
        for name, (px, py) in seeds:
            if not (x0 <= px < x1 and y0 <= py < y1):
                raise SystemExit(f"seed {name} at ({px},{py}) is outside the plant crop "
                                 f"x {x0}-{x1}, y {y0}-{y1}. Re-read it off --mode reference.")
            by_name[name].append(feats[patch_index((px - x0, py - y0), view.shape, grid)])
        groups = [np.stack(by_name[n]) for n in names]

        # Same per-pixel decision the interactive picker makes, so a replayed
        # placement produces the same picture it did on screen.
        frame = {"view": view, "feats": feats, "grid": grid}
        region, margin, confidence = decide(frame, groups)
        overlay = colour_regions(view, region, plant, len(groups))
        for i, (name, (px, py)) in enumerate(seeds):
            centre = (px - x0, py - y0)
            cv2.drawMarker(overlay, centre, (255, 255, 255), cv2.MARKER_CROSS, 30, 4)
            cv2.drawMarker(overlay, centre, (0, 0, 0), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(overlay, name, (centre[0] + 16, centre[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(overlay, name, (centre[0] + 16, centre[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

        panel = np.hstack([view, overlay,
                           scalar_overlay(view, margin, plant, 0.0, args.margin_scale),
                           scalar_overlay(view, confidence, plant, *CONFIDENCE_RANGE)])
        banner(panel, f"{path.stem}   probe: {' vs '.join(names)}   |   "
                      f"margin 0-{args.margin_scale:.2f}   |   "
                      f"confidence {CONFIDENCE_RANGE[0]}-{CONFIDENCE_RANGE[1]}")
        cv2.imwrite(str(out / f"{path.stem}_probe.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

        share = [f"{name} {100 * (region[plant] == i).mean():.0f}%"
                 for i, name in enumerate(names)]
        print(f"{path.stem}: " + "  ".join(share)
              + f"   margin {np.median(margin[plant]):.3f}"
                f"  tied {int((margin[plant] < 0.02).mean() * 100)}%"
                f"  confidence {np.median(confidence[plant]):.3f}"
                f"  unlike-any {int((confidence[plant] < CONFIDENCE_RANGE[0]).mean() * 100)}%")

    print(f"\nwrote {out}/")
    print("Read the middle panel first: does the colour change where the blades "
          "actually meet? Then the right panel -- dark there means the features "
          "were nearly indifferent, and a boundary drawn through a dark band is "
          "one you cannot trust however tidy it looks.")


def mode_split(args, workdir: Path, out: Path, backbone: Backbone) -> None:
    """No clicks: cluster leaf patches by feature and see whether blades fall out."""
    ks = [int(v) for v in args.k_sweep.split(",")] if args.k_sweep else [args.k]
    for path in pick_frames(args, workdir):
        bgr = cv2.imread(str(path))
        mask = plant_mask(workdir, path.stem, bgr.shape)
        if mask is None:
            print(f"  {path.stem}: no plant mask -- skipped")
            continue
        x0, y0, x1, y1 = crop_box(mask, bgr.shape, args.pad)
        view, plant = bgr[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
        view[~plant] = 0
        feats, grid = backbone.features(view)
        inside = cv2.resize(plant.astype(np.uint8), (grid[1], grid[0]),
                            interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
        if inside.sum() < max(ks) * 3:
            print(f"  {path.stem}: only {inside.sum()} plant patches -- crop too small")
            continue

        panels = [view]
        for k in ks:
            label, centres = spherical_kmeans(feats[inside], k, seed=args.seed)
            full = np.full(len(feats), -1, np.int64)
            full[inside] = label
            region = cv2.resize(full.reshape(grid).astype(np.int16) + 1,
                                (view.shape[1], view.shape[0]),
                                interpolation=cv2.INTER_NEAREST) - 1
            panels.append(colour_regions(view, region, plant, k))
            # How separated the clusters are from each other, in the same
            # cosine units the whole method runs on. Rising with k means the
            # extra cluster found something; flat means it split noise.
            off = centres @ centres.T
            np.fill_diagonal(off, np.nan)
            print(f"{path.stem}  k={k}: sizes "
                  f"{sorted((label == c).sum() for c in range(k))[::-1]}"
                  f"   mean between-cluster similarity {np.nanmean(off):.3f}")

        panel = np.hstack(panels)
        banner(panel, f"{path.stem}   unsupervised split   k = {', '.join(map(str, ks))}")
        cv2.imwrite(str(out / f"{path.stem}_split.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"\nwrote {out}/")
    print("If the clusters land on whole blades with no clicks at all, the features "
          "carry leaf identity on their own. If they land on lighting -- sunlit half "
          "one colour, shaded half another -- they carry illumination, and any "
          "instance signal is riding underneath it.")


def mode_affinity(args, workdir: Path, out: Path, backbone: Backbone) -> None:
    """P5 leaf x leaf feature similarity, averaged over views, against a control."""
    import pycolmap

    from pose_estimator.semantic import camera_from_colmap, render_points

    owner, num_leaves = leaf_owner_per_cloud_point(workdir)
    points = load_cloud(workdir)
    if len(points) != len(owner):
        raise SystemExit(f"cloud has {len(points)} points, labels cover {len(owner)}")
    colors = np.full((len(points), 3), 160, np.uint8)

    reconstruction = pycolmap.Reconstruction(str(workdir / "p3" / "sparse" / "best"))
    image_ids = sorted(reconstruction.reg_image_ids())[::args.stride]
    print(f"  {num_leaves} P5 leaves, {len(points)} points, {len(image_ids)} views "
          f"(stride {args.stride})")

    # per_view[v] is a (num_leaves, dim) block, NaN where that leaf was not
    # visible enough in that view to average honestly.
    per_view, view_names = [], []
    # Patch-level tally, kept alongside the pooled one on purpose. Averaging a
    # leaf's patches into one vector answers "what kind of thing is this",
    # which for five blades of one sugarbeet is the same answer five times.
    # The distinction between THIS blade and THAT one lives in local cues --
    # shading, orientation, the occlusion edge where they cross -- and a mean
    # is exactly the operation that removes them. So also ask, patch by patch:
    # of all patches not on leaf i, which leaf holds the one that looks most
    # like this? Same features, no pooling, and it is the same contrastive
    # question --mode probe puts to two clicked points.
    nn_counts = np.zeros((num_leaves, num_leaves))
    for n, image_id in enumerate(image_ids):
        image = reconstruction.images[image_id]
        stem = Path(image.name).stem
        bgr = cv2.imread(str(workdir / "p1" / "frames" / image.name))
        mask = plant_mask(workdir, stem, bgr.shape if bgr is not None else (0, 0))
        if bgr is None or mask is None:
            continue
        x0, y0, x1, y1 = crop_box(mask, bgr.shape, args.pad)
        view, plant = bgr[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
        view[~plant] = 0
        feats, grid = backbone.features(view)

        camera = camera_from_colmap(image, reconstruction.cameras[image.camera_id])
        _rgb, index_map = render_points(points, colors, camera)
        owner_map = np.where(index_map >= 0, owner[index_map.clip(0)], -1)[y0:y1, x0:x1]
        patch_of = patch_ids_for_crop(view.shape, grid)

        block = np.full((num_leaves, feats.shape[1]), np.nan)
        leaf_patches = {}
        for leaf in range(num_leaves):
            here = owner_map == leaf
            if here.sum() < args.min_pixels:
                continue
            # Only patches this leaf *dominates*. A patch straddling two
            # blades holds a blend of both, and blends are exactly what would
            # manufacture agreement between two leaves that touch -- the
            # result this experiment exists to avoid faking.
            ids, counts = np.unique(patch_of[here], return_counts=True)
            total = np.bincount(patch_of[plant].reshape(-1),
                                minlength=grid[0] * grid[1])[ids]
            keep = ids[counts >= args.dominance * np.maximum(total, 1)]
            if len(keep) < args.min_patches:
                continue
            leaf_patches[leaf] = feats[keep]
            v = feats[keep].mean(axis=0)
            block[leaf] = v / max(np.linalg.norm(v), 1e-9)
        per_view.append(block)

        # Within this view only, so lighting and pose are held fixed and what
        # is left between two leaves is identity.
        for i, mine in leaf_patches.items():
            others = [(j, p) for j, p in leaf_patches.items() if j != i]
            if not others:
                continue
            owners = np.concatenate([np.full(len(p), j) for j, p in others])
            sim = mine @ np.concatenate([p for _, p in others]).T
            winner = owners[sim.argmax(axis=1)]
            for j in range(num_leaves):
                nn_counts[i, j] += float((winner == j).sum())
        view_names.append(stem)
        if (n + 1) % 5 == 0 or n + 1 == len(image_ids):
            seen = int(np.isfinite(block[:, 0]).sum())
            print(f"    {n + 1}/{len(image_ids)} views, {seen}/{num_leaves} leaves in {stem}")

    if not per_view:
        raise SystemExit("no usable views -- check p1/frames and p2/masks/plant")
    stack = np.stack(per_view)                                  # (views, leaves, dim)

    # Different leaves, same view. Comparing within a view holds lighting and
    # pose fixed, so what is left is identity -- which is the whole claim.
    pair = np.full((num_leaves, num_leaves), np.nan)
    pair_n = np.zeros((num_leaves, num_leaves), int)
    for i in range(num_leaves):
        for j in range(num_leaves):
            if i == j:
                continue
            both = np.isfinite(stack[:, i, 0]) & np.isfinite(stack[:, j, 0])
            if both.sum():
                pair[i, j] = float((stack[both, i] * stack[both, j]).sum(axis=1).mean())
                pair_n[i, j] = int(both.sum())

    # The control: one leaf, two views. This is the ceiling -- how alike the
    # SAME blade looks to itself once the camera has moved.
    control = []
    for i in range(num_leaves):
        ok = np.nonzero(np.isfinite(stack[:, i, 0]))[0]
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                control.append(float(stack[ok[a], i] @ stack[ok[b], i]))
    control = np.array(control) if control else np.array([np.nan])

    print(f"\n  leaf-to-leaf similarity, same view, averaged over views "
          f"(n views in brackets):\n")
    header = "        " + "".join(f"  leaf {j}    " for j in range(num_leaves))
    print(header)
    for i in range(num_leaves):
        row = f"  leaf {i}"
        for j in range(num_leaves):
            row += "     --     " if i == j else f"  {pair[i, j]:.3f}({pair_n[i, j]:2d}) "
        print(row)

    off = pair[np.isfinite(pair)]
    print(f"\n  CONTROL  same leaf, two different views: median {np.nanmedian(control):.3f} "
          f"({len(control)} pairs)")
    print(f"  different leaves, same view:               median {np.median(off):.3f} "
          f"({len(off)} ordered pairs)")

    order = np.dstack(np.unravel_index(np.argsort(-np.nan_to_num(pair, nan=-1), axis=None),
                                       pair.shape))[0]
    seen, ranked = set(), []
    for i, j in order:
        if i == j or (j, i) in seen or not np.isfinite(pair[i, j]):
            continue
        seen.add((i, j))
        ranked.append((int(i), int(j), float(pair[i, j])))
    print("\n  most-alike leaf pairs (pooled):")
    for i, j, v in ranked[:4]:
        print(f"    leaf {i} + leaf {j}   {v:.3f}")

    share = nn_counts / np.maximum(nn_counts.sum(axis=1, keepdims=True), 1)
    print("\n  PATCH-LEVEL, no pooling. Row i: of leaf i's patches, the share whose")
    print("  most-similar patch on any OTHER leaf belongs to leaf j.\n")
    print("        " + "".join(f" leaf {j}  " for j in range(num_leaves)))
    for i in range(num_leaves):
        row = f"  leaf {i}"
        for j in range(num_leaves):
            row += "   --   " if i == j else f"  {100 * share[i, j]:4.0f}% "
        best = int(np.argmax(np.where(np.arange(num_leaves) == i, -1, share[i])))
        print(row + f"   -> {best}")
    # Mutual first choice is the claim worth acting on: i points at j and j
    # points back. One-way agreement is what a small leaf does to a big
    # neighbour it is partly hidden behind.
    first = [int(np.argmax(np.where(np.arange(num_leaves) == i, -1, share[i])))
             for i in range(num_leaves)]
    mutual = sorted({tuple(sorted((i, j))) for i, j in enumerate(first) if first[j] == i})
    print("\n  mutual first choices: "
          + (", ".join(f"leaf {i} <-> leaf {j}" for i, j in mutual) if mutual else "none"))
    result_nn = {"nn_share": share.tolist(), "mutual_first_choice": mutual}

    result = {
        "views": view_names, "num_leaves": num_leaves,
        "pair_similarity": np.where(np.isfinite(pair), pair, None).tolist(),
        "pair_views": pair_n.tolist(),
        "control_same_leaf_across_views_median": float(np.nanmedian(control)),
        "different_leaves_same_view_median": float(np.median(off)),
        "ranked_pairs": ranked,
        **result_nn,
        # str() every value: --workdir and --out arrive as Path, which json
        # will not encode, and losing a finished sweep to that at the write
        # step is the most annoying possible time to find out.
        "settings": {k: str(v) for k, v in vars(args).items()},
    }
    with open(out / "affinity.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  wrote {out}/affinity.json")
    print("\nHOW TO READ THIS")
    print("  The control is the ceiling. If the top-ranked pair scores near it, those")
    print("  two P5 leaves look as alike as one leaf does to itself across the turn --")
    print("  which is what two halves of a single blade should do.")
    print("  If every pair scores near the control, the features are describing")
    print("  'sugarbeet leaf' and not 'this leaf', and no threshold rescues that.")
    print("  The gap between the top pair and the rest is the usable signal.")
    print("  Then read the patch table, and expect it to disagree with the pooled one.")
    print("  A mutual first choice there is two leaves whose tissue keeps picking each")
    print("  other out of the whole plant -- which is what one blade cut in half does.")


# --------------------------------------------------------------------------


def parse_seeds(items) -> List[Tuple[str, Tuple[int, int]]]:
    seeds = []
    for item in items or []:
        if ":" not in item or "," not in item:
            raise SystemExit(f"--seeds wants name:x,y (got {item!r})")
        name, coords = item.split(":", 1)
        try:
            x, y = (int(v) for v in coords.split(","))
        except ValueError:
            raise SystemExit(f"--seeds wants name:x,y with whole pixels (got {item!r})")
        seeds.append((name.strip(), (x, y)))
    if len({name for name, _ in seeds}) < 2:
        raise SystemExit("--mode probe needs at least two DIFFERENT seed names, one per "
                         "leaf you want told apart. One leaf is a heat map, not a "
                         "decision. (Repeating a name is how you give one leaf several "
                         "examples, which is a different thing.)")
    return seeds


def pick_frames(args, workdir: Path) -> List[Path]:
    paths = frame_paths(workdir)
    if args.frames:
        picks = [int(v) for v in args.frames.split(",")]
        missing = [p for p in picks if p >= len(paths)]
        if missing:
            raise SystemExit(f"--frames {missing} out of range: the run has {len(paths)}")
        return [paths[i] for i in picks]
    return paths[:: args.stride]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True,
                   choices=["reference", "probe", "split", "affinity"])
    p.add_argument("--workdir", required=True, type=Path,
                   help="a finished specimen run (needs p1, p2, p4c; affinity also needs p3, p5)")
    p.add_argument("--frames", help="comma-separated frame indices, e.g. 20,26,53")
    p.add_argument("--stride", type=int, default=8, help="use every Nth frame instead")
    p.add_argument("--seeds", nargs="*",
                   help="probe: name:x,y in full-frame pixels. Omit to click them instead")
    p.add_argument("--margin-scale", type=float, default=0.25,
                   help="probe: margin value painted as full brightness. Display only "
                        "-- it fixes the colour scale so frames can be compared")
    p.add_argument("--max-display", type=int, default=1100,
                   help="longest side of the interactive window's image panel")
    p.add_argument("--k", type=int, default=4, help="split: number of clusters")
    p.add_argument("--k-sweep", help="split: try several, e.g. 2,3,4,5,6")
    p.add_argument("--seed", type=int, default=0, help="split: k-means seed")
    p.add_argument("--min-pixels", type=int, default=200,
                   help="affinity: rendered pixels a leaf needs to count in a view")
    p.add_argument("--min-patches", type=int, default=3,
                   help="affinity: dominated patches a leaf needs to count in a view")
    p.add_argument("--dominance", type=float, default=0.5,
                   help="affinity: share of a patch one leaf must own to use it")
    p.add_argument("--pad", type=int, default=60, help="pixels of context around the plant")
    p.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument("--size", type=int, default=896,
                   help="square the crop is resized to; 896/16 = a 56x56 patch grid")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("/tmp/leaf_identity"))
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "reference":
        mode_reference(args, args.workdir, out)
        return

    # Check the cheap things before spending a minute loading a ViT: a typo in
    # --seeds should cost a second, not a model load.
    pick_frames(args, args.workdir)
    if args.mode == "probe" and args.seeds:
        parse_seeds(args.seeds)

    backbone = Backbone(args.model, args.device, args.size)
    print(f"  {args.model} on {args.device}, {args.size}px crop "
          f"-> {args.size // 16}x{args.size // 16} patches")
    if args.mode == "probe":
        # Clicking is the default. --seeds is for replaying a placement, or for
        # a machine with no display.
        run = mode_probe if args.seeds else interactive_probe
        run(args, args.workdir, out, backbone)
        return
    {"split": mode_split, "affinity": mode_affinity}[args.mode](
        args, args.workdir, out, backbone)


if __name__ == "__main__":
    main()
