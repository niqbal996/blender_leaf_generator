# Plant Turntable → Structured 3D Asset Pipeline

**Implementation plan for Claude Code.** Read §0 before writing any code.

---

## 0. Brief for the agent

### Goal
Turn DSLR turntable video of a single potted (or clamp-held) plant into a **structured plant asset**: a stem skeleton graph plus, for each leaf, a 3D midrib curve with curvature, insertion angle, azimuth, width profile, and a lateral curl profile describing how the two halves of the lamina fold around the midrib. Repeat at scale to build a library.

### Non-goals (do not build these)
- Photorealistic novel-view rendering. Render quality is a *diagnostic*, not a deliverable.
- Field / in-canopy capture. Single isolated plant, controlled turntable, only.
- Real-time anything.
- A GUI. CLI + saved artifacts + matplotlib/Open3D diagnostic dumps only.

### Working style
- **Every phase is a standalone CLI that reads from disk and writes to disk.** No phase holds another phase's state in memory. This lets us swap backends later without rewrites.
- **Phases 1–4 are engineering; phases 5–6 are research.** Do not start 5 until 1–4 pass their acceptance criteria on at least 3 different plants. Most of the pain in the current 3DGS attempt comes from skipping this.
- Write the acceptance-criteria check as an actual script (`scripts/check_pN.py`), not a manual eyeball.
- Commit a diagnostic image/mesh dump per phase per specimen under `runs/<specimen>/<phase>/diag/`.
- When a decision is made (backend choice, threshold, hyperparameter), append it to `DECISIONS.md` with the reason and the evidence.

### Ground rule on thresholds
The existing pipeline fails because it uses *manual, per-plant, appearance-space thresholds*. Any threshold introduced in this pipeline must be either (a) derived from geometry we independently know (e.g. the visual hull), or (b) fit once and validated across all specimens. If you find yourself adding a per-specimen magic number, stop and flag it in `DECISIONS.md` instead.

---

## 1. Output schema — build this first

Define this before any reconstruction code. It is the contract every phase serialises toward.

Create `schema/plant_asset.schema.json` and a matching `src/plantpose/schema.py` with pydantic models.

```jsonc
{
  "asset_id": "specimen_0042",
  "schema_version": "1.0.0",
  "units": "mm",                    // metric scale is mandatory, not optional
  "up_axis": "+Z",
  "origin": "stem base at substrate/clamp line",

  "capture": {
    "camera": "Canon R5 / 100mm macro",
    "n_frames": 96,
    "elevations_deg": [20, 60],
    "scale_source": "charuco",      // charuco | scalebar | none
    "scale_rms_mm": 0.14
  },

  "stem": {
    "nodes": [
      {"id": 0, "p": [0,0,0], "radius_mm": 4.1, "parent": -1, "kind": "base"}
    ],
    "spline": {"degree": 3, "knots": [], "ctrl_pts": [], "arclength_mm": 210.4}
  },

  "leaves": [
    {
      "id": 0,
      "attach_node_id": 12,
      "insertion_angle_deg": 47.3,   // midrib tangent at base vs local stem tangent
      "azimuth_deg": 137.5,          // rotation about stem axis, phyllotaxy
      "midrib": {
        "arclength_mm": 88.2,
        "samples": [
          {
            "s": 0.0,                // normalised arclength 0..1
            "p": [0,0,0],
            "tangent": [0,0,1],
            "normal": [1,0,0],       // rotation-minimising frame, NOT Frenet
            "binormal": [0,1,0],
            "kappa": 0.004,          // 1/mm, midrib bending
            "tau": 0.0002,           // 1/mm, midrib torsion (twist along leaf)
            "half_width_mm": 12.4,
            "curl_kappa": 0.021,     // 1/mm, cross-section curvature; + = adaxial cupping
            "curl_asymmetry": 0.05   // (left - right) / (left + right), -1..1
          }
        ]
      },
      "lamina_model": {
        "kind": "neuraleaf",         // neuraleaf | developable | none
        "shape_latent": [],
        "deform_latent": [],
        "fit_rmse_mm": 0.6
      },
      "mesh_ref": "leaves/leaf_000.ply",
      "confidence": 0.87,
      "occlusion_fraction": 0.11     // fraction of lamina never seen in any view
    }
  ],

  "roots": {"rsml_ref": "roots/specimen_0042.rsml"},   // nullable

  "provenance": {
    "pipeline_version": "git sha",
    "phase_backends": {"pose": "charuco", "geometry": "2dgs+hull", "structure": "gaussianplant"},
    "warnings": []
  }
}
```

**Design notes to honour:**
- Use a **rotation-minimising frame** (double-reflection method, Wang et al. 2008) along the midrib, not a Frenet frame. Frenet frames flip at inflection points, and leaf midribs have inflections constantly. Getting this wrong will silently corrupt every curl measurement.
- `curl_kappa` is defined in the plane spanned by (normal, binormal) at each station — i.e. the cross-section perpendicular to the midrib. Sign convention: positive = adaxial (upper) surface concave.
- `occlusion_fraction` matters more than it looks. A leaf seen from one side only will get a plausible-looking but fabricated curl. Downstream consumers need to be able to filter on it.

---

## 2. Repo layout

```
plantpose/
  README.md
  DECISIONS.md
  pyproject.toml
  schema/
    plant_asset.schema.json
  src/plantpose/
    schema.py
    io.py                  # run directory conventions, artifact naming
    p1_ingest.py           # video → frames → charuco → intrinsics/extrinsics
    p2_segment.py          # SAM2 video propagation → per-frame alpha
    p3_pose.py             # colmap-masked | charuco | vggt fallback
    p4_geometry.py         # 2DGS/gsplat → depth → TSDF mesh; visual hull carve
    p4_hull.py             # silhouette carving, standalone + reusable
    p5_structure.py        # stem graph + leaf instance segmentation
    p6_leaf.py             # midrib extraction, frame, curvature, curl fit
    p7_export.py           # → plant_asset.json (+ RSML for roots)
    diag/                  # plotting + mesh dump helpers
  scripts/
    check_p1.py ... check_p7.py
    run_all.sh
  third_party/             # submodules: gsplat, sam2, etc.
  data/
    raw/<specimen>/         # source video + calibration shots
  runs/<specimen>/<phase>/  # all intermediate artifacts + diag/
  eval/
    pheno4d/               # benchmark harness
    synthetic/             # procedural generation + GT
  tests/
```

---

## 3. Environment

- Python 3.11, CUDA 12.x, PyTorch matched to CUDA.
- `colmap` (system binary) + `pycolmap`.
- `opencv-contrib-python` — the **contrib** build, required for `cv2.aruco`.
- `open3d`, `trimesh`, `scipy`, `numpy`, `networkx`, `pydantic`, `typer`, `rich`.
- Splatting backend: **`gsplat` (nerfstudio-project)**. See §8 on licensing — this choice is deliberate.
- `sam2` (facebookresearch) for mask propagation.

Pin everything in `pyproject.toml`. Record exact versions in `DECISIONS.md` — CUDA/torch/rasterizer combinations in this ecosystem break constantly and "it worked last month" is not reproducible.

**Verify every third-party repo URL by fetching it before adding it as a submodule.** Several of the papers referenced in §9 promised code that may or may not have landed. If a repo is missing, note it and use the fallback listed for that phase; do not stall.

---

## 4. Phase plan

### P0 — Scaffolding
Repo layout, schema module, run-directory conventions, a `Specimen` dataclass, logging, and a `diag` helper that writes a labelled PNG or PLY into the current phase's `diag/`. Stub every phase CLI so `run_all.sh` executes end to end and fails loudly with `NotImplementedError`.

**Accept:** `run_all.sh` runs, creates the full directory tree, fails at P1 with a clear message.

---

### P1 — Ingest and calibration
**In:** `data/raw/<specimen>/turntable.MOV` (+ optional separate calibration clip)
**Out:** `runs/<specimen>/p1/frames/*.png`, `intrinsics.json`, `poses_charuco.json`, `scale.json`

Tasks:
1. Extract frames. Deduplicate near-identical frames (turntable pauses) and **drop blurry frames** by variance-of-Laplacian, keeping the sharpest frame per angular bin. Target 72–120 usable frames per elevation.
2. Detect the ChArUco board on the turntable; calibrate intrinsics (or load them if the lens is pre-calibrated — prefer pre-calibration from a dedicated board clip, it is far more stable).
3. Solve board pose per frame → camera extrinsics in board frame → **metric scale for free**.
4. Fit the turntable rotation axis from the sequence of board poses. Report residual. Define the world frame: origin at the axis/board-plane intersection, +Z along the axis pointing up.
5. Write `scale.json` with mm-per-unit and an RMS residual.

**Accept (`check_p1.py`):**
- ≥ 60 frames retained per elevation.
- Board reprojection RMS < 0.5 px.
- Turntable axis fit residual < 1 mm at the board radius.
- Reprojecting a known board square edge recovers its length to < 1%.

**Pitfalls:** A fixed camera with a rotating subject means the *board rotates with the plant* — poses come out as "camera orbiting a static plant" only after you invert. Get this convention right once, assert it with a synthetic test, and never revisit it. If the pot occludes the board, use a larger board or a ring of markers around the turntable rim.

---

### P2 — Segmentation
**In:** P1 frames
**Out:** `runs/<specimen>/p2/masks/*.png` (binary), `alpha/*.png` (soft matte), `qc.json`

Tasks:
1. SAM2 with video propagation. One or two click prompts on the first frame; propagate through the sequence.
2. Produce **three** mask classes, not one: `plant`, `holder` (pot / clamp / pliers), `background`. The holder mask is needed later — a clamp near the stem base will otherwise be absorbed into the skeleton as a fake branch.
3. Soft matte at leaf edges (thin/backlit edges matter for the visual hull). Erode/dilate consistency check across adjacent frames; flag temporal instability.
4. If a controlled backdrop is used, additionally compute a chroma/luminance matte and **cross-check against SAM2** — disagreement above a threshold is a QC failure, not something to silently average.

**Accept (`check_p2.py`):**
- Mask area varies smoothly with turntable angle (no frame-to-frame jump > 15% of median area).
- Zero frames with empty or full-frame masks.
- Manual spot-check of 5 random frames dumped to `diag/`.

**Pitfalls:** SAM2 will happily include the pot. Thin petioles are the first thing it drops — inspect those specifically. If a specimen consistently loses the stem, that specimen needs a re-shoot with a contrasting backdrop, not a software fix.

---

### P3 — Pose
**In:** P1 frames + intrinsics, P2 masks
**Out:** `runs/<specimen>/p3/cameras.json` (COLMAP-compatible), `sparse/`, `pose_source.txt`

Strategy — try in this order, record which one was used:
1. **ChArUco poses from P1.** If P1 accepted, these are already metrically correct and usually better than SfM on a thin, self-similar subject. Use them directly.
2. **Masked COLMAP.** Feed COLMAP the plant masks so features come only from the subject. Initialise from the ChArUco poses. Refine with bundle adjustment; keep intrinsics fixed.
3. **Feed-forward fallback: VGGT or π³** for specimens where matching collapses (fine foliage, glossy leaves). Use only as an *initialiser* for a subsequent masked BA — not as final poses. Then re-anchor scale from the ChArUco solution.

**Accept (`check_p3.py`):**
- ≥ 95% of frames registered.
- Mean reprojection error < 1.0 px.
- Recovered camera centres lie on a circle (per elevation); RMS deviation from the fitted circle < 2 mm.
- Poses agree with the ChArUco poses to < 1 mm translation / 0.5° rotation.

That last check is the important one. It is a genuine, independent cross-validation, and it is the reason the ChArUco board is worth the setup time.

---

### P4 — Dense geometry and hull constraint
**In:** P3 poses, P1 frames, P2 masks
**Out:** `runs/<specimen>/p4/hull.ply`, `splat.ckpt`, `depth/*.npy`, `mesh.ply`, `points.ply`

This phase is where the current manual-thresholding problem gets solved. Two independent geometry sources, intersected.

Tasks:
1. **Visual hull first.** Voxel-carve from the plant masks using the known poses. Octree or a 512³–1024³ dense grid over the fitted bounding cylinder. Output a watertight hull mesh and an occupancy grid.
   *Why first:* it is deterministic, needs no training, and is robust exactly where MVS/splatting fail (thin stems, petioles). It is also the automatic floater filter.
2. **Surface-aligned splatting.** Train with `gsplat` using a **2DGS / surfel** configuration — flattened primitives that lie on surfaces. Do NOT use vanilla isotropic 3DGS. Supervise with masked images (alpha loss against P2 mattes) plus a normal-consistency term. Optionally add a low-weight monocular normal prior (DSINE / StableNormal); prefer normals over monocular depth here — leaves are flat sheets, so normals are the informative signal and Depth Anything's scale/shift ambiguity contributes noise.
3. **Extract geometry from rendered depth, not from primitive centres.** Render depth + normals from all training views, TSDF-fuse, extract a mesh. Splat centres are optimised for photometric loss and are not surface samples; this is the root cause of the "needs manual thresholding" problem.
4. **Carve.** Reject any mesh vertex / point outside the dilated visual hull (dilation = 1–2 voxels, fit once globally). This replaces the manual opacity/scale threshold entirely.
5. Optionally also run masked COLMAP dense MVS as a third source and report agreement — useful as a trust metric per specimen.

**Accept (`check_p4.py`):**
- Rendered-vs-input mask IoU > 0.97 averaged over held-out views.
- Post-carve floater count (connected components with < 0.1% of total points, farther than 5 mm from the main component) = 0 with no per-specimen tuning.
- Mesh is manifold enough for Open3D to compute per-vertex normals without error.
- Chamfer distance between splat-derived mesh and COLMAP dense cloud reported (no hard threshold; track it across specimens).

**Pitfalls:** The visual hull over-estimates leaf *thickness* — that is fine and expected, it is being used as a bound, not as the geometry. Do not try to make the hull the final surface. Also: concavities (a cupped leaf) are not carvable from silhouettes; the splatting result supplies those.

---

### P5 — Structure: stem graph + leaf instances
**In:** P4 mesh/points, P2 masks, P3 poses
**Out:** `runs/<specimen>/p5/stem_graph.json`, `leaf_instances.npy`, per-leaf point subsets

This is the first research phase. Implement **two** paths behind a common interface `StructureBackend` and compare them empirically; do not commit to one on the basis of a paper abstract.

**Path A — structure-primitive splatting (GaussianPlant-style).**
Represent the plant as coarse *structure primitives* (cylinders for stem/petiole segments, elliptic disks for leaf patches) with a learned branch-vs-leaf probability, plus dense *appearance primitives* bound to those primitives' surfaces. Optimise jointly: photometric loss drives the appearance primitives, and a binding loss propagates that signal to the structure primitives. Add a DINOv3-distilled semantic term to disambiguate slender-but-planar cases, and regularise the branch graph toward a tree with a reweighted MST (penalise oblique connections and edges crossing empty space). Leaf instances then fall out by clustering leaf-labelled disks on position + normal + major-axis alignment.
*This is the closest published match to the target and it emits the branch graph and leaf instances directly, with no post-hoc skeletonisation.* Check whether the authors' code has been released; if not, the paper (arXiv 2512.14087) has enough detail to reimplement, and the pieces (gsplat, DINOv3, MST) all exist.

**Path B — differentiable skeleton fit to silhouettes (Masks-to-Skeleton).**
Model the skeleton as a graph with node positions, radii, and a learned adjacency matrix; render its silhouette differentiably and optimise directly against the P2 multi-view masks, with an MST step in the loop to keep it a tree. This bypasses point-cloud quality entirely, which makes it the right tool for thin petioles that P4 loses. Code: `github.com/huntorochi/Masks-to-Skeleton` (verify).

**Path C — classical baseline (implement anyway, it is cheap).**
kNN graph on the carved point cloud → geodesic distance from the stem base → level-set clustering → MST skeleton, plus L1-medial or Laplacian contraction for comparison. This is the current state of the art *only* when the cloud is clean; it exists here as the number the other two paths must beat.

Additional required step: **holder rejection.** Use the P2 `holder` mask to delete pot/clamp geometry before skeletonisation, then define the asset origin as the lowest surviving stem node.

**Accept (`check_p5.py`):**
- Stem graph is a single connected tree rooted at the base (assert, don't hope).
- Leaf count matches manual annotation on 5 specimens (±1).
- Radius monotonicity: child radius ≤ parent radius along every path, after the sanity-prune.
- Report Chamfer distance of predicted branch points vs a hand-annotated branch cloud on ≥ 3 specimens.

---

### P6 — Leaf model: midrib, curvature, curl
**In:** P5 per-leaf point subsets + P4 mesh, P1 frames
**Out:** per-leaf midrib samples, curl profiles, optional latent codes

Again, two paths, common interface.

**Path A — parametric leaf model fit (NeuraLeaf).**
Fit a neural parametric leaf model that disentangles a flattened 2D base shape from a 3D deformation via skeleton-free skinning. This is the strong option because the midrib is a **fixed curve in the canonical flat template** — so once fitted, the 3D midrib comes out by construction, defined identically across every species in the library, and the lateral curl is not a separate model but simply part of the deformation latent. It is designed to fit to point clouds and depth maps, which is exactly what P4 produces. Project page: `neuraleaf-yang.github.io` (verify code release).

**Path B — explicit geometric fit (implement first; it is the fallback and the sanity check).**
1. **Midrib from appearance, not just geometry.** The midrib is a strong photometric ridge in RGB. Run a Frangi vesselness filter (or a small U-Net) on the leaf region in each frame where that leaf is well-observed, then lift the 2D ridge to 3D by intersecting the back-projected rays with the P4 surface, and robustly fuse across views. This gives the *anatomical* vein. Compare against the purely geometric alternative (geodesic from petiole junction to the farthest lamina point) and report the discrepancy — where they disagree, the appearance-based one is usually right.
2. Fit an arclength-parameterised cubic B-spline to the fused midrib points. Regularise curvature; do not let it wiggle to fit noise.
3. Build a **rotation-minimising frame** along the spline (double-reflection). Assert frame continuity.
4. At N stations along `s`, slice the lamina with the plane spanned by (normal, binormal). Fit each side of the cross-section independently → `half_width_mm`, `curl_kappa`, `curl_asymmetry`.
5. Optionally impose a **developability prior**: leaves are near-inextensible, so the lamina is approximately a developable surface swept along the midrib. This is a strong regulariser for the occluded underside and should measurably reduce fabricated curl. Measure its effect; keep it only if it helps.
6. Derive `insertion_angle_deg` (midrib tangent at s=0 vs local stem tangent) and `azimuth_deg` (rotation about the stem axis) from the P5 graph.
7. Compute `occlusion_fraction` per leaf by projecting the lamina into every view and checking visibility. **This is required, not optional** — an unseen underside produces a confident-looking curl number that is pure prior.

**Accept (`check_p6.py`):**
- Physical validation: flatten-and-scan 10 real leaves (press them, photograph on a flatbed with a ruler). Predicted `arclength_mm` and width profile must match the flattened ground truth within 5%.
- Midrib spline residual to fused ridge points < 1 mm RMS.
- Frame continuity: no discontinuity in the RMF greater than 5° between adjacent stations.
- Curl profiles are stable under leave-one-view-out: re-fit with 20% of views dropped, report variation in `curl_kappa`.

That flatten-and-scan validation is the single most valuable experiment in the whole plan. It is the only place where an independent, physical ground truth is available, and it will catch systematic errors that no amount of self-consistency checking will.

---

### P7 — Export and library
**Out:** `runs/<specimen>/asset/plant_asset.json`, `leaves/*.ply`, `stem.ply`, `preview.png`

Tasks:
1. Serialise to the §1 schema; validate against the JSON schema on write.
2. Emit an **MTG** (OpenAlea multiscale tree graph) alongside the JSON — that is the interchange format the plant-science ecosystem actually reads, and it costs almost nothing once the graph exists.
3. For specimens with exposed roots, emit **RSML** for the root architecture. Roots are thin and tangled; expect the visual hull to be the only thing that survives there. Set root confidence accordingly and do not oversell it.
4. Build a library index (`library/index.parquet`) with one row per asset: species, leaf count, mean occlusion fraction, all acceptance-check values, pipeline git sha. This index is what lets you later ask "which assets are trustworthy enough to train on".
5. A `preview.png` per asset — stem graph + midrib curves overlaid on one input frame. Cheap, and it makes bad assets obvious at a glance.

**Accept:** schema validation passes; a round-trip load reproduces the geometry; 20 random previews inspected.

---

### P8 — Evaluation and generalisation

Run this in parallel with P5/P6, not after.

1. **Pheno4D benchmark.** Labelled maize and tomato point clouds with leaf instance annotations (Bonn / Stachniss group). Feed the point clouds directly into P5/P6, skipping P1–P4. This decouples "is my reconstruction bad" from "is my structure extraction bad" — which is currently the thing you cannot distinguish.
2. **Synthetic supervision for genericity.** This is the answer to "will it generalise". Procedurally generate plants with *known* midribs and curl (Blender + Infinigen's procedural plants, or sample from the fitted NeuraLeaf latent space), render turntable sequences matching the real capture geometry, and get unlimited ground-truth structure for free. Use it to (a) quantify error against exact GT, and (b) pretrain the P5 structure backend. Sim-to-real is unusually tractable here because the capture conditions are controlled.
3. **Cross-species held-out set.** Fit on species A–E, evaluate on F–J untouched. Report degradation. Any per-species tuning that shows up here is a bug in the design, not a hyperparameter to add.
4. For 2D segmentation genericity, prefer zero-shot/foundation-model routes over species-specific training — cross-domain transfer is the property that matters for a library spanning many species (cf. ZeroPlantSeg, `github.com/JunhaoXing/ZeroPlantSeg`).

---

## 5. Suggested order of work

| Week | Work |
|---|---|
| 1 | P0, P1, P2. Shoot 5 specimens covering the range: small rosette, tall stem, large-leaved, fine-foliaged, root-in-clamp. |
| 2 | P3, P4 (hull first, then splatting). Get `check_p4` green on all 5. |
| 3 | P5 Path C (classical baseline) + Pheno4D harness. Establishes the number to beat. |
| 4–5 | P5 Path B, then Path A. Compare on the same 5 specimens + Pheno4D. |
| 6 | P6 Path B + flatten-and-scan validation of 10 leaves. |
| 7 | P6 Path A if code is available; P7 export; synthetic generation. |
| 8 | Scale to 50 specimens, build the library index, review failures. |

---

## 6. Known hard failure modes — expect these, don't be surprised

- **Plant motion during capture.** Leaves move between frames from air currents and from the turntable's own acceleration. Stop-motion capture with a settle delay, not continuous rotation. If a specimen moves anyway, no reconstruction method will save it.
- **Glossy leaves.** Specular highlights are view-dependent and will be reconstructed as geometry. Circular polariser on the lens + polarised light sources if possible.
- **Dense self-occlusion.** Rosettes with overlapping leaves have laminae that are never fully seen. The `occlusion_fraction` field exists for this; assets above ~0.3 should probably be excluded from any downstream training set.
- **Thin petioles.** The single most common structural break. P5 Path B (silhouette-based) exists specifically for this.
- **Scale drift on specimens where ChArUco fails.** Always have a physical scale bar in frame as a backup.

---

## 7. Open questions to resolve empirically

Log answers in `DECISIONS.md` as they land.

1. Is the visual hull constraint alone sufficient to eliminate floaters, or is a learned filter still needed?
2. Does the appearance-based midrib meaningfully differ from the geometric medial axis, and by how much, per species?
3. Does the developability prior improve or bias the occluded-side curl estimate? (Measure with the flatten-and-scan set.)
4. Path A vs Path B for structure: which degrades more gracefully on the fine-foliaged specimen?
5. How many views can be dropped before curl estimates become unstable? This sets the capture cost per asset for the library.

---

## 8. Licensing — flag before scaling up

The original 3D Gaussian Splatting code (Inria/MPII) and several derivatives ship under **non-commercial research licences**. 2DGS inherits constraints from that lineage. If this library is ever going to be a commercial asset set, that matters.

`gsplat` (nerfstudio-project) is Apache-2.0 and is the reason it is specified in §3. SAM2 and COLMAP are permissively licensed. Before adding any third-party repo, read its LICENSE and record the terms in `DECISIONS.md`. Do not discover this at asset #500.

---

## 9. Reference reading

Have the agent fetch and skim these before implementing the corresponding phase. Do not treat any as settled — verify code availability.

- **GaussianPlant** — structure/appearance-disentangled 3DGS emitting branch graphs and leaf instances. arXiv 2512.14087. → P5 Path A.
- **Masks-to-Skeleton** — differentiable graph fit to multi-view silhouettes. Sensors 2025, `github.com/huntorochi/Masks-to-Skeleton`. → P5 Path B.
- **NeuraLeaf** — neural parametric leaf model, 2D base shape ⊗ 3D deformation. ICCV 2025, arXiv 2507.12714, `neuraleaf-yang.github.io`. → P6 Path A.
- **2D Gaussian Splatting** — surface-aligned primitives. SIGGRAPH 2024. → P4.
- **Gaussian Surfels / PGSR** — alternatives to 2DGS for P4; benchmark against each other.
- **TreeFormer** — single-view skeleton estimation; possible initialiser for P5. WACV 2025.
- **ZeroPlantSeg** — zero-shot hierarchical plant/leaf segmentation. arXiv 2509.09116. → P2/P8.
- **Pheno4D** — labelled maize/tomato point clouds. PLoS ONE 2021. → P8.
- **VGGT / π³** — feed-forward pose+geometry, for the P3 fallback path.
- Okura 2022, *3D modeling and reconstruction of plants and trees* (Breeding Science) — the field survey; read this first for orientation.
