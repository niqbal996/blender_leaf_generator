#!/usr/bin/env bash
# Plant pose pipeline, P1 -> P6: video in, per-leaf measurements out.
#
#   # 1. frames + masks
#   ./run_pipeline.sh --video <file> --workdir runs/plant_9 --stop-after p1p2
#
#   # 2. click a few leaf/stem/root points (writes p4c/seeds.json)
#   pose-pick-seeds --workdir runs/plant_9
#
#   # 3. everything else. The seeds are found automatically.
#   ./run_pipeline.sh --workdir runs/plant_9 --skip-to p3 --hf-token hf_xxx
#
# Or in one go, if you already have seeds from another specimen:
#
#   ./run_pipeline.sh --video <file> --workdir runs/plant_9 \
#       --seed-bank runs/plant_1/p4c/seed_bank.npz
#
# Or with no seeds at all, using SAM2 instead of DINOv3 (leaf/stem only):
#
#   ./run_pipeline.sh --video <file> --workdir runs/plant_9 \
#       --backend sam --sam-checkpoint checkpoints/sam2.1_hiera_large.pt
#
# Phases: p1p2 p3 p4a p4b p4c p5 p6.  --skip-to <phase> resumes partway on an
# existing workdir; --stop-after <phase> ends early. P4c looks for clicked
# seeds at <workdir>/p4c/seeds.json and uses them without being told.
#
# Several --video files are capture passes of the *same* plant, shot at
# different camera elevations. They share one workdir: frames are numbered
# consecutively, each pass is tracked and variance-masked on its own, and
# COLMAP solves them together into a single coordinate frame. This is the fix
# for leaves that merge at the apex -- a single waist-height orbit never looks
# down into the whorl, so no amount of processing can separate what was never
# seen from two directions.
#
# Stops after P4c. Skeleton and leaf-model stages (P5, P6) are deliberately not
# run -- the deliverable here is coloured point clouds to inspect in Blender.
#
# Picking seeds. P4c classifies every image patch by which labelled example it
# most resembles, so it needs a few clicks on one frame. Get a coordinate grid
# to read them off with:
#
#   python scripts/dinov3_organ_lab.py --mode reference \
#       --images <workdir>/p1/frames --frames 57 \
#       --plant-mask-dir <workdir>/p2/masks/plant --out /tmp/ref
#
# Coordinates are in that cropped image's pixel space. Three or four leaf
# clicks at different angles, two stem, one root works well. Any label is
# allowed; leaf/stem/root get fixed colours.
#
# The DINOv3 weights are gated on HuggingFace: accept the licence at
# https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m then pass
# --hf-token (or export HF_TOKEN). Without a token, pass
# --dino-model facebook/dinov2-base, which is ungated and behaves similarly.
#
# Runtime on an RTX 2070, 96 frames: about 40 minutes, dominated by COLMAP
# (P3) and surfel training (P4b).

set -euo pipefail

VIDEOS=(); WORKDIR=""; SEED_FRAME=""; HF_TOKEN_ARG="${HF_TOKEN:-}"
DINO_MODEL="facebook/dinov3-vitb16-pretrain-lvd1689m"
SEEDS=(); SKIP_TO=""; SEED_BANK=""; SEEDS_FILE=""; SKIP_P4B=0
BACKEND="dino"; SAM_CHECKPOINT=""; STOP_AFTER=""

# Print the comment block at the top of this file, however long it is, so the
# help text cannot drift out of sync with a hard-coded line range.
usage() {
    awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)        shift; while [[ $# -gt 0 && "$1" != --* ]]; do VIDEOS+=("$1"); shift; done ;;
        --workdir)      WORKDIR="$2"; shift 2 ;;
        --seed-frame)   SEED_FRAME="$2"; shift 2 ;;
        --hf-token)     HF_TOKEN_ARG="$2"; shift 2 ;;
        --dino-model)   DINO_MODEL="$2"; shift 2 ;;
        --skip-to)      SKIP_TO="$2"; shift 2 ;;
        --stop-after)   STOP_AFTER="$2"; shift 2 ;;
        --seed-bank)    SEED_BANK="$2"; shift 2 ;;
        --seeds-file)   SEEDS_FILE="$2"; shift 2 ;;
        --backend)      BACKEND="$2"; shift 2 ;;
        --sam-checkpoint) SAM_CHECKPOINT="$2"; shift 2 ;;
        --skip-p4b)     SKIP_P4B=1; shift ;;
        --seeds)        shift; while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
        -h|--help)      usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$WORKDIR" ]] || { echo "--workdir is required" >&2; usage 1; }
[[ ${#VIDEOS[@]} -gt 0 || -d "$WORKDIR/p1/frames" ]] || {
    echo "--video is required unless $WORKDIR/p1/frames already exists" >&2; usage 1; }

# Seeds clicked with pose-pick-seeds land here by default, so finding them is
# not something you should have to tell the script about.
if [[ -z "$SEEDS_FILE" && ${#SEEDS[@]} -eq 0 && -z "$SEED_BANK" && -f "$WORKDIR/p4c/seeds.json" ]]; then
    SEEDS_FILE="$WORKDIR/p4c/seeds.json"
    echo "found clicked seeds at $SEEDS_FILE -- using them for P4c"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/anaconda3/envs/pose_estimator/bin/python"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
[[ -n "$HF_TOKEN_ARG" ]] && export HF_TOKEN="$HF_TOKEN_ARG"

mkdir -p "$WORKDIR"
LOG="$WORKDIR/pipeline.log"
# Everything below is tee'd, so a finished run leaves a readable record.
exec > >(tee -a "$LOG") 2>&1
echo "=== run started $(date '+%Y-%m-%d %H:%M:%S') ==="

# Phases run in order; --skip-to jumps in partway on an existing workdir and
# --stop-after ends early. Both name a phase from this list.
ORDER=(p1p2 p3 p4a p4b p4c p5 p6)
started=0
stopped=0
should_run() {
    [[ "$stopped" == 1 ]] && return 1
    if [[ -n "$SKIP_TO" && "$started" == 0 ]]; then
        [[ "$1" == "$SKIP_TO" ]] || return 1
        started=1
    fi
    # Decided after the phase is allowed to run, so --stop-after p4c runs p4c.
    [[ "$1" == "$STOP_AFTER" ]] && stopped=1
    return 0
}

for name in "$SKIP_TO" "$STOP_AFTER"; do
    [[ -z "$name" ]] && continue
    [[ " ${ORDER[*]} " == *" $name "* ]] || {
        echo "unknown phase '$name' -- expected one of: ${ORDER[*]}" >&2; exit 1; }
done

phase() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

if should_run p1p2; then
    phase "P1+P2  sharpest frames + SAM2 plant/holder masks   -> $WORKDIR/p1, p2"
    [[ ${#VIDEOS[@]} -gt 1 ]] && echo "  ${#VIDEOS[@]} capture passes, tracked separately, solved together in P3"
    $PY -m pose_estimator.cli.segment \
        ${VIDEOS[0]:+--video} ${VIDEOS[@]+"${VIDEOS[@]}"} --workdir "$WORKDIR" \
        --checkpoint "$REPO_ROOT/checkpoints/sam2.1_hiera_large.pt"
fi

if should_run p3; then
    phase "P3     camera poses, masked COLMAP                 -> $WORKDIR/p3"
    $PY -m pose_estimator.cli.pose --workdir "$WORKDIR"
fi

if should_run p4a; then
    phase "P4a    visual hull by silhouette carving           -> $WORKDIR/p4"
    $PY -m pose_estimator.cli.hull --workdir "$WORKDIR" --resolution 256
fi

if should_run p4b && [[ "$SKIP_P4B" == 0 ]]; then
    phase "P4b    2DGS surfels -> carved thin surface         -> $WORKDIR/p4b"
    $PY -m pose_estimator.cli.surface --workdir "$WORKDIR" --iterations 5000
elif [[ "$SKIP_P4B" == 1 ]]; then
    phase "P4b    SKIPPED (--skip-p4b) -- P4c will label the P4a hull instead"
    echo "  Halves the runtime. The hull is a solid, so the cloud is blobbier,"
    echo "  and P5 could not skeletonise it later without re-running P4b."
fi

if should_run p4c; then
    if [[ "$BACKEND" != "sam" && -z "$SEEDS_FILE" && ${#SEEDS[@]} -eq 0 && -z "$SEED_BANK" ]]; then
        phase "P4c    SKIPPED -- no seeds given"
        echo "  The DINO backend needs a few labelled examples. Easiest way to get them:"
        echo "    pose-pick-seeds --workdir $WORKDIR"
        echo "  Click a few leaf/stem/root points, press s. That writes"
        echo "  $WORKDIR/p4c/seeds.json, which this script picks up automatically"
        echo "  on the next run -- no argument needed."
        echo
        echo "  Alternatives:"
        echo "    --seed-bank <path>/seed_bank.npz   reuse an earlier specimen's vectors"
        echo "    --backend sam --sam-checkpoint <ckpt>   no seeds at all (leaf/stem only)"
    else
        phase "P4c    organ labels + coloured clouds            -> $WORKDIR/p4c"
        echo "  two stages: classify (frames -> p4c/class_maps) then fuse (maps -> labels)."
        echo "  Run them separately with pose-classify / pose-fuse when debugging -- the"
        echo "  class maps are what tell you whether a bad label came from the 2D"
        echo "  classifier or from the multi-view voting."
        $PY -m pose_estimator.cli.semantic \
            --workdir "$WORKDIR" --backend "$BACKEND" --dino-model "$DINO_MODEL" \
            ${HF_TOKEN_ARG:+--hf-token "$HF_TOKEN_ARG"} \
            ${SAM_CHECKPOINT:+--checkpoint "$SAM_CHECKPOINT"} \
            ${SEEDS_FILE:+--seeds-file "$SEEDS_FILE"} \
            ${SEED_BANK:+--seed-bank "$SEED_BANK"} \
            ${SEED_FRAME:+--seed-frame "$SEED_FRAME"} \
            ${SEEDS[0]:+--seeds} ${SEEDS[@]+"${SEEDS[@]}"}
    fi
fi

if should_run p5; then
    if [[ -f "$WORKDIR/p4c/labels.npy" ]]; then
        phase "P5     stem centreline + leaf instances           -> $WORKDIR/p5"
        $PY -m pose_estimator.cli.structure --workdir "$WORKDIR"
    else
        phase "P5     SKIPPED -- no $WORKDIR/p4c/labels.npy"
        echo "  P5 is driven by the P4c organ labels; run P4c first."
    fi
fi

if should_run p6; then
    if [[ -f "$WORKDIR/p5/leaf_points.npy" ]]; then
        phase "P6     per-leaf midrib, curvature, width          -> $WORKDIR/p6"
        $PY -m pose_estimator.cli.leaf_model --workdir "$WORKDIR"
    else
        phase "P6     SKIPPED -- no $WORKDIR/p5/leaf_points.npy"
        echo "  P6 fits midribs to P5's per-leaf point subsets; run P5 first."
    fi
fi

phase "done  ($(date '+%H:%M:%S'))"
cat <<EOF
Open in Blender:
  $WORKDIR/p6/midribs.ply           per-leaf midrib curves
  $WORKDIR/p5/structure.ply         leaf / stem / root points, coloured
  $WORKDIR/p4c/labels_vis.ply       cloud coloured by organ (leaf/stem/root)
  $WORKDIR/p4c/leaf_instances.ply   leaf points, one colour per leaf
  $WORKDIR/p4c/confidence.ply       cloud coloured by vote confidence
  $WORKDIR/p4b/surface.ply          the uncoloured surface P4c labelled
  $WORKDIR/p4/hull.ply              visual hull mesh

Measurements: $WORKDIR/p6/leaves.json
  Per leaf: arclength, insertion angle, azimuth, width profile along the midrib.
  Units are COLMAP units, NOT metric -- no scale reference is solved yet.

QC reports -- read before trusting anything:
  $WORKDIR/p2/qc.json   $WORKDIR/p3/poses.json   $WORKDIR/p4/hull.json
  $WORKDIR/p4b/p4b.json $WORKDIR/p4c/qc.json     $WORKDIR/p5/p5.json
  $WORKDIR/p6/p6.json

Check the 2D classification before blaming the 3D labels:
  $WORKDIR/p4c/diag/parts_*.jpg     photograph | classification, side by side
  $WORKDIR/p4c/class_maps/          the per-frame maps the fusion voted on

Is the structure real? These two answer it faster than any number:
  $WORKDIR/p5/diag/skeleton_*.jpg   stem + leaf axes drawn on the photographs
  $WORKDIR/p6/diag/midribs_3d.png   midribs and width profiles

Diagnostics: $WORKDIR/p*/diag/
Full log:    $LOG
EOF
