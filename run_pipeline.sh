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
#   --use-gpu   run P3's SIFT on the GPU. Needs a CUDA pycolmap build
#               (pip install pycolmap-cuda) with its bundled CUDA runtime on
#               LD_LIBRARY_PATH -- setup_env.sh arranges both. Without it,
#               COLMAP still runs, just on CPU.
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
# If P2 segments the wrong object. The SAM2 seeds are derived from colour,
# which fails when the holder out-competes the plant on area -- the pliers'
# amber grip is green-dominant in RGB, so a pass that shows it large and
# unlit can seed on the tool and track it for the whole sequence. P4a then
# carves nothing, because two passes' silhouettes describe different objects.
# Check p2/qc.json (plant_mask_free_of_holder) and the p2/diag overlays; if
# the green mask is on the holder, click the plant once:
#
#   ./run_pipeline.sh --video <a> <b> --workdir runs/plant_9 --stop-after p1p2
#   pose-pick-prompts --workdir runs/plant_9     # one plant click per pass
#   pose-segment --workdir runs/plant_9 --reuse-frames    # redo P2 only
#   ./run_pipeline.sh --workdir runs/plant_9 --skip-to p3
#
# You only do that once for a rig. Clicking writes two files: the pixel
# coordinates in p2/prompts_clicked.json, which fix this video, and the DINO
# feature vectors in p2/prompt_bank.npz, which fix every later one. The
# vectors describe what the plant and the plier *look like*, so the next
# specimen is searched for whatever most resembles them and needs no clicks
# however it is posed. Later runs pick the newest bank sitting beside their
# workdir on their own; --prompt-bank <path> names one, --prompt-root <dir>
# says where to look, --no-prompt-bank goes back to the colour rule.
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
# Plant architecture. P5 splits leaves by how far into them you can travel
# from the plant's base, so it needs a base to start from -- and which kind
# you have is a property of the specimen, so you pass it in:
#
#   --architecture caulescent   (default) upright, with a central stem; leaf
#                               depth is measured from the stem tissue P4c
#                               labelled
#   --architecture rosette      leaves radiate from a crown at ground level
#                               and there is no stem at all (thistle, sugar
#                               beet); the crown is located from the geometry
#                               and stem labels are ignored
#
# Run a rosette as caulescent and P5 reports 0 contact points, 0 tips and 0
# leaves: there is nothing to seed the depth field from. It is not inferred --
# an earlier version guessed and flipped thistle1 from crown to stem purely
# because P4c had started labelling the crown "stem".
#
# GPU. P4b (gsplat) and P4c (DINOv3/SAM2) always use the GPU. P3 is the
# exception: COLMAP's SIFT extraction runs on the CPU unless pycolmap was
# built with CUDA, which the PyPI wheels are not. Check with
#
#   python -c "import pycolmap; print(pycolmap.has_cuda)"
#
# True means --use-gpu will work here; False means it raises. Only extraction
# moves -- matching and mapping are CPU either way, and on 192 frames matching
# is the larger share, so this is a smaller win than it sounds.
#
# Runtime on an RTX 2070, 96 frames: about 40 minutes, dominated by COLMAP
# (P3) and surfel training (P4b).

set -euo pipefail

VIDEOS=(); WORKDIR=""; SEED_FRAME=""; HF_TOKEN_ARG="${HF_TOKEN:-}"
DINO_MODEL="facebook/dinov3-vitb16-pretrain-lvd1689m"
SEEDS=(); SKIP_TO=""; SEED_BANK=""; SEEDS_FILE=""; SKIP_P4B=0
BACKEND="dino"; SAM_CHECKPOINT=""; STOP_AFTER=""
PROMPT_BANK=""; PROMPT_ROOT=""; NO_PROMPT_BANK=0; USE_GPU=0; ARCHITECTURE=""; PERSISTENCE=""; PROMPT_POINTS=""

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
        --prompt-bank)  PROMPT_BANK="$2"; shift 2 ;;
        --prompt-points) PROMPT_POINTS="$2"; shift 2 ;;
        --prompt-root)  PROMPT_ROOT="$2"; shift 2 ;;
        --no-prompt-bank) NO_PROMPT_BANK=1; shift ;;
        --use-gpu)      USE_GPU=1; shift ;;
        --architecture) ARCHITECTURE="$2"; shift 2 ;;
        --min-persistence-ratio) PERSISTENCE="$2"; shift 2 ;;
        --seeds-file)   SEEDS_FILE="$2"; shift 2 ;;
        --backend)      BACKEND="$2"; shift 2 ;;
        --sam-checkpoint) SAM_CHECKPOINT="$2"; shift 2 ;;
        --skip-p4b)     SKIP_P4B=1; shift ;;
        --use-gpu)      USE_GPU=1; shift ;;
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

# Which object is the plant, for P2. Three sources, most specific first:
#   1. clicks for this video          <workdir>/p2/prompts_clicked.json
#   2. a bank named with --prompt-bank
#   3. the newest bank beside this workdir
# (3) is what makes a batch work. The bank holds what a plant and a holder
# look like rather than where they sat in one video, so one specimen's clicks
# carry to every later capture of the same rig. Without any of the three, P2
# falls back to the colour rule, which is the thing that seeds on the pliers.
if [[ "$NO_PROMPT_BANK" == 0 && -z "$PROMPT_BANK" && ! -f "$WORKDIR/p2/prompts_clicked.json" ]]; then
    PROMPT_ROOT="${PROMPT_ROOT:-$(dirname "$WORKDIR")}"
    FOUND_BANK="$(ls -t "$PROMPT_ROOT"/*/p2/prompt_bank.npz 2>/dev/null | head -1 || true)"
    if [[ -n "$FOUND_BANK" && "$FOUND_BANK" != "$WORKDIR/p2/prompt_bank.npz" ]]; then
        PROMPT_BANK="$FOUND_BANK"
        echo "reusing plant/holder prompts from $PROMPT_BANK"
        echo "  (pass --prompt-bank <other> to choose, or --no-prompt-bank for the colour rule)"
    fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# --- interpreter -----------------------------------------------------------
# Whatever python is active, so `conda activate <env> && ./run_pipeline.sh`
# is the whole contract. $POSE_PYTHON overrides for a cron job or a wrapper
# that cannot activate an env first.
PY="${POSE_PYTHON:-$(command -v python3 || command -v python)}"
[[ -n "$PY" && -x "$PY" ]] || { echo "no python found on PATH -- activate your env first" >&2; exit 1; }
# Checking the heavy dependencies, not just `import pose_estimator`: this
# script puts src/ on PYTHONPATH, so the package imports from *any*
# interpreter -- including one with none of its dependencies installed. The
# run then dies several minutes in, after extracting frames, on a bare
# ModuleNotFoundError from inside a phase.
MISSING="$("$PY" - <<'PYCHECK' 2>/dev/null
import importlib.util as u
print(" ".join(m for m in ("numpy", "cv2", "torch", "pose_estimator") if u.find_spec(m) is None))
PYCHECK
)"
if [[ -n "${MISSING// /}" ]]; then
    echo "ERROR: $PY is missing: $MISSING" >&2
    echo "  Activate the right environment first:  conda activate pose" >&2
    echo "  Or build one:                          ./setup_env.sh" >&2
    echo "  Override the interpreter with:         POSE_PYTHON=/path/to/python" >&2
    exit 1
fi

# P4b's CUDA toolchain, checked now rather than after COLMAP. gsplat ships no
# prebuilt wheels, so it JIT-compiles on first render -- and if nvcc is too
# old or missing, that surfaces 40 minutes into a run, immediately after the
# expensive phases, with a wall of ninja output. The version matters and not
# just the presence: gsplat compiles with -std=c++20, which nvcc rejects
# before 12.0 ("Value 'c++20' is not defined for option 'std'").
p4b_possible() {
    [[ "$SKIP_P4B" == 1 ]] && return 1
    case "$STOP_AFTER" in p1p2|p3|p4a) return 1 ;; esac
    case "$SKIP_TO" in p4c|p5|p6) return 1 ;; esac
    return 0
}
if p4b_possible; then
    NVCC_BIN="$(command -v nvcc || true)"
    NVCC_MAJOR=""
    [[ -n "$NVCC_BIN" ]] && NVCC_MAJOR="$("$NVCC_BIN" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1)"
    if [[ -z "$NVCC_BIN" ]]; then
        echo "WARNING: no nvcc on PATH -- P4b (gsplat) will fail when it tries to compile." >&2
        echo "  Fix:  ./setup_env.sh          (installs a matching nvcc into the env)" >&2
        echo "  Or skip that phase:  --skip-p4b" >&2
    elif [[ -n "$NVCC_MAJOR" && "$NVCC_MAJOR" -lt 12 ]]; then
        echo "WARNING: nvcc is $("$NVCC_BIN" --version | sed -n 's/.*release \(.*\), .*/\1/p') at $NVCC_BIN," >&2
        echo "  but gsplat compiles with -std=c++20, which needs nvcc 12.0 or newer. P4b will fail." >&2
        echo "  Fix:  ./setup_env.sh          (installs a matching nvcc into the env)" >&2
        echo "  Or skip that phase:  --skip-p4b" >&2
    fi
fi

# --- CUDA ------------------------------------------------------------------
# Located from nvcc rather than assumed: gsplat compiles against whatever
# toolkit is actually installed, and a wrong CUDA_HOME fails at first use
# with an error that never mentions this variable.
if [[ -z "${CUDA_HOME:-}" ]]; then
    if NVCC="$(command -v nvcc)"; then
        CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$NVCC")")")"
    else
        for candidate in /usr/local/cuda /usr/local/cuda-*; do
            [[ -x "$candidate/bin/nvcc" ]] && { CUDA_HOME="$candidate"; break; }
        done
    fi
fi
[[ -n "${CUDA_HOME:-}" ]] && export CUDA_HOME

# --- SAM2 checkpoint -------------------------------------------------------
# Searched across the places it plausibly lives, so a new machine needs no
# edits here. --sam-checkpoint or $SAM2_CHECKPOINT still win.
find_sam_checkpoint() {
    local name="sam2.1_hiera_large.pt"
    local candidates=(
        "${SAM2_CHECKPOINT:-}"
        "$REPO_ROOT/checkpoints/$name"
        "$REPO_ROOT/third_party/sam2/checkpoints/$name"
        "$HOME/.cache/sam2/$name"
    )
    for c in "${candidates[@]}"; do
        [[ -n "$c" && -f "$c" ]] && { echo "$c"; return 0; }
    done
    return 1
}
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
    if ! SAM_CKPT="$(find_sam_checkpoint)"; then
        echo "ERROR: SAM2 checkpoint not found (looked for sam2.1_hiera_large.pt in" >&2
        echo "  \$SAM2_CHECKPOINT, $REPO_ROOT/checkpoints/," >&2
        echo "  $REPO_ROOT/third_party/sam2/checkpoints/, ~/.cache/sam2/)." >&2
        echo "  Fetch it with:  ./setup_env.sh --checkpoint-only" >&2
        exit 1
    fi
    echo "  SAM2 checkpoint: $SAM_CKPT"
    if [[ -n "$PROMPT_BANK" ]]; then
        echo "  plant/holder prompts: $PROMPT_BANK"
    elif [[ -f "$WORKDIR/p2/prompts_clicked.json" ]]; then
        echo "  plant/holder prompts: clicked, $WORKDIR/p2/prompts_clicked.json"
    else
        echo "  WARNING: no plant/holder prompts -- falling back to the COLOUR RULE." >&2
        echo "    That rule picks the largest green-dominant blob, and an orange or" >&2
        echo "    amber plier grip is green-dominant in RGB. It has seeded on the tool" >&2
        echo "    on more than one capture here. Check p2/qc.json before P3." >&2
        echo "    A P4c --seed-bank does NOT feed P2: that bank holds organ classes." >&2
        echo "    The P2 bank is p2/prompt_bank.npz, written by pose-pick-prompts." >&2
    fi
    $PY -m pose_estimator.cli.segment \
        ${VIDEOS[0]:+--video} ${VIDEOS[@]+"${VIDEOS[@]}"} --workdir "$WORKDIR" \
        --checkpoint "$SAM_CKPT" \
        ${PROMPT_BANK:+--prompt-bank "$PROMPT_BANK"} \
        ${PROMPT_BANK:+--dino-model "$DINO_MODEL"} \
        ${PROMPT_POINTS:+--prompt-points "$PROMPT_POINTS"}
fi

if should_run p3; then
    phase "P3     camera poses, masked COLMAP                 -> $WORKDIR/p3"
    $PY -m pose_estimator.cli.pose --workdir "$WORKDIR" \
        $([[ "$USE_GPU" == 1 ]] && echo --use-gpu)
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
        # --backend sam needs a checkpoint too; fall back to the same search.
        SAM_CKPT_P4C="$SAM_CHECKPOINT"
        if [[ -z "$SAM_CKPT_P4C" && "$BACKEND" == "sam" ]]; then
            SAM_CKPT_P4C="$(find_sam_checkpoint || true)"
        fi
        phase "P4c    organ labels + coloured clouds            -> $WORKDIR/p4c"
        echo "  two stages: classify (frames -> p4c/class_maps) then fuse (maps -> labels)."
        echo "  Run them separately with pose-classify / pose-fuse when debugging -- the"
        echo "  class maps are what tell you whether a bad label came from the 2D"
        echo "  classifier or from the multi-view voting."
        $PY -m pose_estimator.cli.semantic \
            --workdir "$WORKDIR" --backend "$BACKEND" --dino-model "$DINO_MODEL" \
            ${HF_TOKEN_ARG:+--hf-token "$HF_TOKEN_ARG"} \
            ${SAM_CKPT_P4C:+--checkpoint "$SAM_CKPT_P4C"} \
            ${SEEDS_FILE:+--seeds-file "$SEEDS_FILE"} \
            ${SEED_BANK:+--seed-bank "$SEED_BANK"} \
            ${SEED_FRAME:+--seed-frame "$SEED_FRAME"} \
            ${SEEDS[0]:+--seeds} ${SEEDS[@]+"${SEEDS[@]}"}
    fi
fi

if should_run p5; then
    if [[ -f "$WORKDIR/p4c/labels.npy" ]]; then
        phase "P5     stem centreline + leaf instances           -> $WORKDIR/p5"
        $PY -m pose_estimator.cli.structure --workdir "$WORKDIR" \
            ${ARCHITECTURE:+--architecture "$ARCHITECTURE"} \
            ${PERSISTENCE:+--min-persistence-ratio "$PERSISTENCE"}
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
