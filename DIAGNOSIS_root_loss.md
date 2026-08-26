# Why the root is missing, and what else the runs are hiding

Measured on `runs/thistle1` and `runs/thistle2` as they stood on 2026-08-25.
Every number here is reproducible from the artifacts on disk; the scripts are
in the scratchpad referenced at the end.

## Summary

Four separate failures, only one of which is what it looked like.

| # | failure | where | is it the capture? |
|---|---|---|---|
| 1 | root segmented in 2D, never reconstructed in 3D | P2 masks | **no** -- SAM2 seeding |
| 2 | 221,484-voxel phantom inside the pliers (32% of hull) | P4a, stale artifact | no -- fix existed, never ran |
| 3 | thistle2 lost half its frames (one whole pass) | P3 | **yes** -- blur |
| 4 | every QC failure is advisory; the run completes anyway | `run_pipeline.sh` | no |

Overexposure is **not** a problem on this footage. Blown pixels are 0.00-0.07%
of the plant region and mean luminance is 65-91 of 255. Whatever your colleague
saw on a monitor, the sensor is not clipping. Blur is real, but it is not what
kills the root -- see #1.

## 1. The root: segmented in 2D, absent in 3D

The 2D classifier finds the root in **every frame of thistle1** -- 727,000
pixels, 6.8% of everything it classified. P4c ends up with **4 root points of
58,875** (0.007%). The label had nothing to attach to.

Coverage test -- project the reconstructed geometry back into each frame and
ask what fraction of each class's pixels have any geometry behind them:

```
LEAF pixels covered by p4b surface : 99.8%
ROOT pixels covered by p4b surface : 53.2%
```

So the root is roughly half-reconstructed, and the half that exists is
outvoted (12,462 root votes against 10,004,078 total).

### It is not occlusion

This was the obvious suspect and it is wrong. Sampling voxels in the root band
and in the plant body, using the plant masks with the holder as an occluder:

| band | in frustum | **unoccluded** | **in silhouette** | survives 86%? |
|---|---|---|---|---|
| root `z[-0.39,-0.05]` | 100% | **81%** | **0.560** | 3 of 670 |
| body `z[0.05,0.25]` | 100% | **61%** | **0.981** | 670 of 670 |

The root is *less* occluded than the plant body, and the body reconstructs
perfectly. The occlusion handling in `hull.py` and `surfels.py` is working and
is not the bottleneck. What fails is silhouette agreement: 0.56 against the
0.86 the carve requires.

### It is not blur either

Per pass, for the same root band (pass 1 is 3.9x sharper than pass 0):

| pass | plant sharpness | root silhouette agreement |
|---|---|---|
| 0 (soft) | 85.1 | 0.583 |
| 1 (sharp) | 335.7 | 0.514 |

Equally bad in the sharp pass. Blur does not explain it.

### It is SAM2 seeding

The root is only in the plant mask about two-thirds of the time, because the
jaws cut it off from the foliage and SAM2 propagates from seed points that
were all placed on leaves.

Fraction of the plant mask lying below the jaws, per pass:

```
thistle1  pass 0: root in 74/96 frames    pass 1: root in 49/96 frames
thistle2  pass 0: root in 67/96 frames    pass 1: root in  0/96 frames
```

thistle1's prompts span y 468-900 and reach the root; **thistle2 got 3 prompts,
all at y 472-549**, none below the jaws -- it used the transferred prompt bank,
which places prompts where features most resemble "plant", i.e. foliage. The
bank cannot seed a disconnected blob, which is exactly what the README warns
about and exactly what happened.

Carving from the root class maps alone confirms the root is *there* and merely
under-determined -- 159 voxels at `z -0.389..0.004` at 50% agreement, and
**zero** at 70% or 86%.

**Fix:** re-click prompts with a root click per pass (`pose-pick-prompts`,
`1` = plant, click the root), then `pose-segment --reuse-frames` and re-run
from P3. This is a human step; no threshold change substitutes for it.

## 2. The hull contained a 221,484-voxel phantom

`hull.py` was edited at 21:44; thistle1's hull was carved at 20:11. The
`min_judged_fraction` guard -- written specifically to stop the pliers' shadow
being reconstructed as solid -- **had never run**. The artifact on disk was
32% plier interior.

Re-carving with current code:

| | old (stale) | new |
|---|---|---|
| voxels | 691,849 | 592,908 |
| extent | [2.15, 1.30, 1.32] | [1.55, 0.92, 1.24] |
| voxels below z=-0.05 | 221,484 | **0** |
| hull-vs-mask IoU | 0.440 FAIL | **0.811 PASS** |

`runs/thistle1/p4/` now holds the corrected hull. **P4b onward for thistle1 is
still stale** and must be re-run.

Note the phantom is *why* the earlier evidence looked like an occlusion
problem: those low-z voxels were inside the pliers, so they projected onto the
holder mask 90% of the time. The root was never there.

## 3. thistle2 lost an entire capture pass to blur

P3 registered **96/192** frames -- pass 1 only. Pass 0 registered **0/96**.

| run | pass 0 sharpness | pass 1 sharpness | ratio |
|---|---|---|---|
| thistle1 | 85.1 | 335.7 | 3.9x |
| thistle2 | 53.5 | 338.6 | 6.3x |

The soft pass produces too few stable SIFT features to register. This is the
one finding that genuinely supports changing the capture setup -- and note the
cruel interaction: thistle2's pass 0 is the pass that *did* track the root
(67/96), and it is the one that was thrown away.

thistle2's 3 leaves in `p6/leaves.json` come from half the intended data, one
elevation, and no root.

## 4. Nothing stops a bad run

`run_pipeline.sh` never reads `all_passed`. thistle2 failed P3's
`most_frames_registered` at 50% and still ran P4a, P4b, P4c, P5 and P6 to
completion, producing confident-looking measurements. Both runs have
`all_passed: false` at *every* phase.

## What changed in the code

- `cli/fuse.py` -- new check `classes_survived_into_3d`. Compares each class's
  2D pixel share against its 3D point share; fails when a class with >=1% of
  pixels lands under a tenth of that in 3D. Catches both runs. Replaces the
  old `all_classes_present`, which passed thistle1 on `root 4 > 0`.
- `segmentation_qc.py` -- new check `root_tracked_below_the_jaws`, per pass.
  Threshold is P4a's own `min_inside_fraction` (0.86), so it is predictive of
  the carve rather than a second opinion about it. Flags thistle1 (77%, 51%)
  and thistle2 (70%, 0%).
- `cli/hull.py` -- records `min_judged_views`, `min_judged_fraction` and
  `occlusion_aware` in `hull.json`, so a stale hull is identifiable.
- `tests/pose_estimator/test_root_survival_checks.py` -- 8 tests. Suite is
  128 passing.

## Recommended order

1. Re-click prompts for both specimens with a root click per pass; re-run from
   P2. This is the root fix.
2. Re-run thistle1 from P4b (its hull is corrected but everything after is
   stale).
3. Reshoot thistle2's soft pass, or drop it and accept a single-elevation
   solve. Do not re-run it as-is.
4. Only then consider the capture change. Sharpness is worth fixing (#3);
   exposure is not.
