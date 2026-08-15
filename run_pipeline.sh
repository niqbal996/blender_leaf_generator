#!/usr/bin/env bash
# Plant pose pipeline, P1 -> P4c: video in, organ-labelled point clouds out.
#
#   ./run_pipeline.sh --video <file> [<file2> ...] --workdir <dir> \
#       --seed-frame 57 --seeds "leaf:140,150" "stem:300,300" "root:370,640" \
#       --hf-token hf_xxx
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
SEEDS=(); SKIP_TO=""; SEED_BANK=""; SKIP_P4B=0

usage() {
    sed -n '2,41p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)       shift; while [[ $# -gt 0 && "$1" != --* ]]; do VIDEOS+=("$1"); shift; done ;;
        --workdir)     WORKDIR="$2"; shift 2 ;;
        --seed-frame)  SEED_FRAME="$2"; shift 2 ;;
        --hf-token)    HF_TOKEN_ARG="$2"; shift 2 ;;
        --dino-model)  DINO_MODEL="$2"; shift 2 ;;
        --skip-to)     SKIP_TO="$2"; shift 2 ;;
        --seed-bank)   SEED_BANK="$2"; shift 2 ;;
        --skip-p4b)    SKIP_P4B=1; shift ;;
        --seeds)       shift; while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
        -h|--help)     usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$WORKDIR" ]] || { echo "--workdir is required" >&2; usage 1; }
[[ ${#VIDEOS[@]} -gt 0 || -d "$WORKDIR/p1/frames" ]] || {
    echo "--video is required unless $WORKDIR/p1/frames already exists" >&2; usage 1; }

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

# Phases run in order; --skip-to jumps in partway on an existing workdir.
ORDER=(p1p2 p3 p4a p4b p4c)
started=0
should_run() {
    [[ -z "$SKIP_TO" ]] && return 0
    [[ "$started" == 1 ]] && return 0
    [[ "$1" == "$SKIP_TO" ]] && { started=1; return 0; }
    return 1
}

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
    if [[ ${#SEEDS[@]} -eq 0 && -z "$SEED_BANK" ]]; then
        phase "P4c    SKIPPED -- no --seeds / --seed-frame given"
        echo "  Organ labelling needs a few clicked examples. Generate a coordinate grid:"
        echo "    $PY scripts/dinov3_organ_lab.py --mode reference \\"
        echo "        --images $WORKDIR/p1/frames --frames 57 \\"
        echo "        --plant-mask-dir $WORKDIR/p2/masks/plant --out /tmp/ref"
        echo "  then re-run with --skip-to p4c --seed-frame N --seeds \"leaf:x,y\" ..."
        echo "  Or reuse vectors from a previous specimen:  --seed-bank <path>/seed_bank.npz"
    else
        phase "P4c    organ labels + coloured clouds            -> $WORKDIR/p4c"
        echo "  two stages: classify (frames -> p4c/class_maps) then fuse (maps -> labels)."
        echo "  Run them separately with pose-classify / pose-fuse when debugging -- the"
        echo "  class maps are what tell you whether a bad label came from the 2D"
        echo "  classifier or from the multi-view voting."
        $PY -m pose_estimator.cli.semantic \
            --workdir "$WORKDIR" --dino-model "$DINO_MODEL" \
            ${HF_TOKEN_ARG:+--hf-token "$HF_TOKEN_ARG"} \
            ${SEED_BANK:+--seed-bank "$SEED_BANK"} \
            ${SEED_FRAME:+--seed-frame "$SEED_FRAME"} \
            ${SEEDS[0]:+--seeds} ${SEEDS[@]+"${SEEDS[@]}"}
    fi
fi

phase "done  ($(date '+%H:%M:%S'))"
cat <<EOF
Open in Blender:
  $WORKDIR/p4c/labels_vis.ply       cloud coloured by organ (leaf/stem/root)
  $WORKDIR/p4c/leaf_instances.ply   leaf points, one colour per leaf
  $WORKDIR/p4c/confidence.ply       cloud coloured by vote confidence
  $WORKDIR/p4b/surface.ply          the uncoloured surface P4c labelled
  $WORKDIR/p4/hull.ply              visual hull mesh

QC reports -- read before trusting anything:
  $WORKDIR/p2/qc.json   $WORKDIR/p3/poses.json   $WORKDIR/p4/hull.json
  $WORKDIR/p4b/p4b.json $WORKDIR/p4c/qc.json

Check the 2D classification before blaming the 3D labels:
  $WORKDIR/p4c/diag/parts_*.jpg     photograph | classification, side by side
  $WORKDIR/p4c/class_maps/          the per-frame maps the fusion voted on

Diagnostics: $WORKDIR/p*/diag/
Full log:    $LOG

P5 (structure) and P6 (leaf model) were not run:
  pose-structure --workdir $WORKDIR
  pose-leaf      --workdir $WORKDIR
EOF
