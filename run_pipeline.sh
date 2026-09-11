#!/usr/bin/env bash
# Plant pose pipeline, P1 -> P6: video in, per-leaf measurements out.
#
# Name the dataset directory and the paths follow from it:
#
#   ./run_pipeline.sh /data/2026-09-01/sugarbeet_4
#
# which takes its pass*/ subdirectories as the capture passes, in natural
# order, and works in <dataset>/plant. A flat directory of JPEGs is one pass.
# --photos/--video/--workdir still override, and are still how you point at a
# layout that is not this one.
#
# Settings that hold across runs -- bank paths, architecture, the HF token --
# go in a pipeline.conf of key=value lines rather than in every command:
#
#   prompt_bank  = /data/2026-09-01/sugarbeet_3/plant/p2/prompt_bank.npz
#   seed_bank    = /data/thistle3/plant/p4c/seed_bank.npz
#   architecture = rosette
#   low_texture  = 1
#
# Read from ~/.config/blender_leaf_generator/pipeline.conf, ./pipeline.conf,
# <dataset>/../pipeline.conf and <dataset>/pipeline.conf, in that order --
# each overriding the last, and any flag overriding all of them. --config
# <file> uses just that file. See pipeline.conf.example.
#
# Put the HF token in the user-level file or in $HF_TOKEN, not on the command
# line: an argument is visible in `ps` to every user on the machine and lands
# in your shell history.
#
#   --dry-run   print the resolved inputs and stop. Worth doing whenever a
#               path changed: it prints every path the run will use, including
#               the ones a config file or the dataset directory supplied.
#
#   # 1. frames + masks
#   ./run_pipeline.sh /data/2026-09-01/sugarbeet_4 --stop-after p1p2
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
#   --low-texture  spend more time on features so more frames register: COLMAP's
#               viewpoint- and scale-robust descriptors, weaker maxima kept, and
#               guided matching. For a small, smooth or softly-focused subject.
#               Measured: 13/27 -> 26/27 frames on one capture, 11/54 -> 47/54 on
#               another. Costs roughly 3-5x the feature-extraction time.
#
#   --allow-mixed-capture
#               skip P1's check that the passes are one shoot of one plant --
#               same camera body, consecutive in time, filenames in capture
#               order. The check reads EXIF, so it is only wrong when the EXIF
#               is. Measured on sugarbeet_4: 13 frames of a different plant,
#               shot an hour earlier, reached P3 through a mistyped --photos;
#               COLMAP would not register them and they still took the camera
#               circle from 0.05% to 7.01% RMS, because both shoots share a
#               turntable and pliers for the matcher to latch onto.
#
#   --cameras   how P3 groups camera intrinsics: exif (default), single,
#               per-image or auto. exif reads the focal P1 recorded from each
#               photo into p1/intrinsics.json and gives every distinct lens
#               setting its own camera seeded with that focal; with no EXIF it
#               behaves exactly like single, so video captures are unaffected.
#               single forces one shared camera -- right for one locked-off
#               lens, wrong across a zoom change: measured on sugarbeet_3
#               (passes at 48/32/22mm, solved as one camera at COLMAP's 2304px
#               guess) it gave 1.15px reprojection error and orbit axes 7.7 deg
#               apart, over-carving the P4a hull to 0.53 IoU. per-image is the
#               fallback for photos whose EXIF was stripped.
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
#   --geometry-backend vggt|vggt_omega|mapanything   run P3 with a learned model
#               instead of COLMAP, and run that branch all the way to P6.
#               Everything lands under .../experiments/<b>/, so the baseline
#               is never written to:
#
#                 ./run_pipeline.sh --workdir runs/plant_9 --skip-to p3 \
#                     --geometry-backend vggt \
#                     --model-python ~/miniconda3/envs/vggt/bin/python
#
#               A learned branch skips P4a and P4b. Both exist to manufacture
#               a cloud COLMAP cannot give, and a pointmap already is one.
#               What that costs is real and worth knowing: the hull is an
#               occlusion-aware upper bound -- it can be too big but cannot
#               invent surface -- and P4b's normals are fitted against the
#               photographs, where a bare pointmap's are estimated by local
#               PCA, which weakens the obliquity weighting in label fusion.
#               --carve-check carves the hull anyway, as a validator rather
#               than a dependency: learned points outside it are suspect.
#
#   --compare [b1,b2,b3]    run each branch in turn and put them side by side.
#               Defaults to colmap,vggt_omega,mapanything. The shared work is
#               done once -- P1, P2, the COLMAP model, and the P4c class maps,
#               which are 2D and belong to no branch -- and each branch then
#               runs P3 -> P4c fusion -> P5 -> P6 of its own:
#
#                 ./run_pipeline.sh /data/2026-09-01/thistle3 --compare
#
#               Each branch resumes rather than recomputes: a branch whose P3
#               is already on disk restarts at p4c, so re-running a comparison
#               after re-clicking seeds or re-tuning P5 costs minutes rather
#               than hours -- and needs no model environment, since the
#               exporter never runs. Delete a phase's directory to force it.
#
#               MapAnything is given --poses-from colmap, which is the
#               configuration worth comparing; --branch-flags <b>=<flags>
#               changes what any one branch is run with, e.g.
#               --branch-flags "vggt_omega=--image-resolution 1024".
#
#               It ends by printing the branches side by side on leaf counts
#               and midribs -- the deliverable, rather than the geometry
#               proxies (silhouette IoU, pose residuals) every earlier
#               comparison was judged on -- and the command that draws all
#               three in one Blender scene.
#
#   --reuse-class-maps      skip P4c's classify stage when the class maps are
#               already on disk and go straight to fusion. Implied by
#               --compare for every branch after the first, and the reason
#               those branches differ in geometry alone.
#
# The run STOPS at a phase whose QC shows a catastrophic failure (P2 tracking
# the tool instead of the plant, a failed P3 circle fit, a hull that does not
# match the masks) -- everything after would silently build on garbage.
# Advisory QC failures never stop a run. --keep-going pushes past a stop when
# a partial or salvage result is wanted knowingly.
#
# Still photos instead of video: --photos <dir> takes a directory of JPEGs as
# one capture pass, in filename order (which must be capture order around the
# turntable). Give several directories for several passes, and mix with
# --video freely -- after P1 everything reads frames from disk and cannot tell
# the difference. A video is sampled down to its sharpest frame per angular
# bin; photos have no such redundancy, so each one's sharpness is printed and
# culling a hopeless shot is your call (it widens that angular gap).
#
#   ./run_pipeline.sh --photos /data/plant_9_shots --workdir runs/plant_9 \
#       --stop-after p1p2
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
#   --architecture upright      (default) a central stem with leaves branching
#                               off it; leaf depth is measured out from the
#                               stem tissue P4c labelled. "caulescent" is the
#                               old name for this and still works.
#   --architecture caulescent   (deprecated alias for upright); leaf
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
# Dependencies are checked for the phases this run will actually execute, so a
# carve-only run needs no torch: `pip install -e ".[skeleton]"` covers P3/P4a,
# while P1/P2, P4b and P4c need the segment, splat and semantic extras, and
# ./setup_env.sh installs the lot. A missing package is named together with the
# command that installs it, before any frame is touched.
#
# Runtime on an RTX 2070, 96 frames: about 40 minutes, dominated by COLMAP
# (P3) and surfel training (P4b).

set -euo pipefail

VIDEOS=(); PHOTOS=(); WORKDIR=""; SEED_FRAME=""; HF_TOKEN_ARG="${HF_TOKEN:-}"
DINO_MODEL="facebook/dinov3-vitb16-pretrain-lvd1689m"
SEEDS=(); SKIP_TO=""; SEED_BANK=""; SEEDS_FILE=""; SKIP_P4B=0
BACKEND="dino"; SAM_CHECKPOINT=""; STOP_AFTER=""
PROMPT_BANK=""; PROMPT_ROOT=""; NO_PROMPT_BANK=0; USE_GPU=0; LOW_TEXTURE=0; ARCHITECTURE=""; PERSISTENCE=""; PROMPT_POINTS=""; KEEP_GOING=0
CAMERAS=""; ALLOW_MIXED=0; STRICT_MIDRIBS=0
GEOMETRY_BACKEND="colmap"; MODEL_PYTHON=""; IMAGE_RESOLUTION=""; BUNDLE_ADJUST=0
CARVE_CHECK=0; INTRINSICS_FROM=""; POSES_FROM=""; REUSE_CLASS_MAPS=0; COMPARE=""
declare -A BRANCH_FLAGS=()
DATASET=""; DRY_RUN=0; CONF_FILES=()

# Which settings the command line set explicitly. A config file fills in only
# what is missing, so a flag always beats a file and there is no precedence
# question to remember.
declare -A SET=()

# Print the comment block at the top of this file, however long it is, so the
# help text cannot drift out of sync with a hard-coded line range.
usage() {
    awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"
    exit "${1:-1}"
}

# Flags people reasonably expect this script to have, and what it calls them.
declare -A FLAG_MEANT=(
    [--force]="--keep-going    (only QC stops need overriding; every phase already overwrites its own outputs)"
    [--overwrite]="--keep-going    (phases overwrite their own outputs; nothing needs forcing)"
    [--method]="--geometry-backend vggt|mapanything"
    [--model]="--geometry-backend vggt|mapanything"
    [--geometry]="--geometry-backend vggt|mapanything"
    [--resume]="--skip-to <phase>"
    [--start]="--skip-to <phase>"
    [--start-at]="--skip-to <phase>"
    [--from]="--skip-to <phase>"
    [--stop]="--stop-after <phase>"
    [--until]="--stop-after <phase>"
    [--gpu]="--use-gpu"
    [--conf]="--config <file>"
    [--verbose]="nothing -- every phase already logs to <workdir>/pipeline.log"
)

# An unknown flag used to print one line and then the whole guide, which
# scrolled the one useful line off the screen. The guide stays behind --help.
unknown_option() {
    local given="$1" flag suggestion="" known=()
    # The accepted flags, read out of this script's own parser below, so this
    # message cannot drift from what the parser actually handles.
    mapfile -t known < <(grep -oE '^ +-{1,2}[a-z0-9|-]+\)' "${BASH_SOURCE[0]}" \
        | tr -d ' )' | tr '|' '\n' | grep -E '^--' | sort -u)
    suggestion="${FLAG_MEANT[$given]:-}"
    if [[ -z "$suggestion" ]]; then
        for flag in "${known[@]}"; do
            [[ "$flag" == "$given"* || "$given" == "$flag"* ]] && { suggestion="$flag"; break; }
        done
    fi
    {
        echo "unknown option: $given"
        [[ -z "$suggestion" ]] || echo "  did you mean:  $suggestion"
        echo ""
        echo "  accepted flags:"
        printf '%s\n' "${known[@]}" | column -c 76 | sed 's/^/    /'
        echo ""
        echo "  what each one does:  ${BASH_SOURCE[0]} --help"
    } >&2
    exit 1
}

ORIGINAL_ARGS=("$@")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)        SET[video]=1; shift; while [[ $# -gt 0 && "$1" != --* ]]; do VIDEOS+=("$1"); shift; done ;;
        --photos)       SET[photos]=1; shift; while [[ $# -gt 0 && "$1" != --* ]]; do PHOTOS+=("$1"); shift; done ;;
        --workdir)      SET[workdir]=1; WORKDIR="$2"; shift 2 ;;
        --config)       CONF_FILES+=("$2"); shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --seed-frame)   SEED_FRAME="$2"; shift 2 ;;
        --hf-token) SET[hf_token]=1;     HF_TOKEN_ARG="$2"; shift 2 ;;
        --dino-model) SET[dino_model]=1;   DINO_MODEL="$2"; shift 2 ;;
        --skip-to)      [[ -n "$SKIP_TO" ]] && echo "  note: --skip-to given twice ($SKIP_TO then $2); the last one wins" >&2
                        SKIP_TO="$2"; shift 2 ;;
        --stop-after)   [[ -n "$STOP_AFTER" ]] && echo "  note: --stop-after given twice ($STOP_AFTER then $2); the last one wins" >&2
                        STOP_AFTER="$2"; shift 2 ;;
        --geometry-backend) GEOMETRY_BACKEND="$2"; shift 2 ;;
        --model-python) MODEL_PYTHON="$2"; shift 2 ;;
        --image-resolution) IMAGE_RESOLUTION="$2"; shift 2 ;;
        --bundle-adjust) BUNDLE_ADJUST=1; shift ;;
        --carve-check)  CARVE_CHECK=1; shift ;;
        --reuse-class-maps) REUSE_CLASS_MAPS=1; shift ;;
        --branch-flags) BRANCH_FLAGS["${2%%=*}"]="${2#*=}"; shift 2 ;;
        --compare)      if [[ $# -ge 2 && "$2" != --* ]]; then COMPARE="$2"; shift 2
                        else COMPARE="colmap,vggt_omega,mapanything"; shift; fi ;;
        --intrinsics-from) INTRINSICS_FROM="$2"; shift 2 ;;
        --poses-from)   POSES_FROM="$2"; shift 2 ;;
        --seed-bank) SET[seed_bank]=1;    SEED_BANK="$2"; shift 2 ;;
        --prompt-bank) SET[prompt_bank]=1;  PROMPT_BANK="$2"; shift 2 ;;
        --prompt-points) SET[prompt_points]=1; PROMPT_POINTS="$2"; shift 2 ;;
        --prompt-root) SET[prompt_root]=1;  PROMPT_ROOT="$2"; shift 2 ;;
        --no-prompt-bank) NO_PROMPT_BANK=1; shift ;;
        --use-gpu) SET[use_gpu]=1;      USE_GPU=1; shift ;;
        --low-texture) SET[low_texture]=1;  LOW_TEXTURE=1; shift ;;
        --cameras) SET[cameras]=1;      CAMERAS="$2"; shift 2 ;;
        --allow-mixed-capture) SET[allow_mixed_capture]=1; ALLOW_MIXED=1; shift ;;
        --strict-midribs) SET[strict_midribs]=1; STRICT_MIDRIBS=1; shift ;;
        --architecture) SET[architecture]=1; ARCHITECTURE="$2"; shift 2 ;;
        --min-persistence-ratio) SET[min_persistence_ratio]=1; PERSISTENCE="$2"; shift 2 ;;
        --seeds-file)   SEEDS_FILE="$2"; shift 2 ;;
        --backend) SET[backend]=1;      BACKEND="$2"; shift 2 ;;
        --sam-checkpoint) SET[sam_checkpoint]=1; SAM_CHECKPOINT="$2"; shift 2 ;;
        --skip-p4b) SET[skip_p4b]=1;     SKIP_P4B=1; shift ;;
        --keep-going) SET[keep_going]=1;   KEEP_GOING=1; shift ;;
        --use-gpu)      USE_GPU=1; shift ;;
        --seeds)        shift; while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
        -h|--help)      usage 0 ;;
        --*) unknown_option "$1" ;;
        *)  [[ -z "$DATASET" ]] || { echo "give at most one dataset directory (got $DATASET and $1)" >&2; usage 1; }
            DATASET="$1"; shift ;;
    esac
done

# --------------------------------------------------------------------------
# A dataset directory answers most of the command line by itself
# --------------------------------------------------------------------------
# The long form is the one that goes wrong: every path repeats the same
# dataset prefix, so a single stale component is easy to type and impossible
# to see. Measured cost of exactly that, on sugarbeet_4: one --photos entry
# left pointing at sugarbeet_3 and the solve was of two different plants.
# Naming the dataset once removes the chance to disagree with yourself.
resolve_dataset() {
    local root="$1"
    [[ -d "$root" ]] || { echo "no such dataset directory: $root" >&2; exit 1; }
    root="${root%/}"

    local passes=()
    while IFS= read -r dir; do passes+=("$dir"); done < <(
        find "$root" -mindepth 1 -maxdepth 1 -type d -name 'pass*' | sort -V)

    if [[ ${#passes[@]} -eq 0 ]] && compgen -G "$root"/*.[jJ][pP][gG] > /dev/null; then
        # A flat directory of photos is one pass, which is how thistle3 is laid out.
        passes=("$root")
    fi

    if [[ ${#passes[@]} -gt 0 && -z "${SET[photos]:-}" && -z "${SET[video]:-}" ]]; then
        PHOTOS=("${passes[@]}")
    fi
    [[ -n "${SET[workdir]:-}" ]] || WORKDIR="$root/plant"

    if [[ ${#passes[@]} -eq 0 && ! -d "$WORKDIR/p1/frames" ]]; then
        echo "$root has no pass*/ subdirectories, no *.JPG of its own, and no" >&2
        echo "  frames already at $WORKDIR/p1/frames -- is it a dataset directory?" >&2
        exit 1
    fi
}

# Settings that hold across runs -- bank paths, architecture, the HF token --
# belong in a file rather than in every command. Least specific first, so a
# per-dataset file overrides a per-session one, and a flag overrides both.
load_config() {
    local file key value
    for file in "$@"; do
        [[ -f "$file" ]] || continue
        LOADED_CONF+=("$file")
        local line_no=0
        while IFS= read -r line || [[ -n "$line" ]]; do
            line_no=$((line_no + 1))
            line="${line%%#*}"
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"
            [[ -n "$line" ]] || continue
            [[ "$line" == *=* ]] || {
                echo "$file:$line_no: expected key=value, got: $line" >&2; exit 1; }
            key="${line%%=*}"; value="${line#*=}"
            key="${key%"${key##*[![:space:]]}"}"
            value="${value#"${value%%[![:space:]]*}"}"
            value="${value%\"}"; value="${value#\"}"
            # A flag already given wins; the file only fills in what is missing.
            [[ -z "${SET[$key]:-}" ]] || continue
            case "$key" in
                prompt_bank)   PROMPT_BANK="$value" ;;
                prompt_root)   PROMPT_ROOT="$value" ;;
                prompt_points) PROMPT_POINTS="$value" ;;
                seed_bank)     SEED_BANK="$value" ;;
                architecture)  ARCHITECTURE="$value" ;;
                hf_token)      HF_TOKEN_ARG="$value" ;;
                dino_model)    DINO_MODEL="$value" ;;
                backend)       BACKEND="$value" ;;
                sam_checkpoint) SAM_CHECKPOINT="$value" ;;
                cameras)       CAMERAS="$value" ;;
                min_persistence_ratio) PERSISTENCE="$value" ;;
                low_texture)   LOW_TEXTURE=$([[ "$value" == 1 || "$value" == true ]] && echo 1 || echo 0) ;;
                use_gpu)       USE_GPU=$([[ "$value" == 1 || "$value" == true ]] && echo 1 || echo 0) ;;
                skip_p4b)      SKIP_P4B=$([[ "$value" == 1 || "$value" == true ]] && echo 1 || echo 0) ;;
                keep_going)    KEEP_GOING=$([[ "$value" == 1 || "$value" == true ]] && echo 1 || echo 0) ;;
                strict_midribs) STRICT_MIDRIBS=$([[ "$value" == 1 || "$value" == true ]] && echo 1 || echo 0) ;;
                allow_mixed_capture) ALLOW_MIXED=$([[ "$value" == 1 || "$value" == true ]] && echo 1 || echo 0) ;;
                *) echo "$file:$line_no: unknown setting '$key'. Valid keys are the long" >&2
                   echo "  options with dashes as underscores: prompt_bank, seed_bank," >&2
                   echo "  architecture, hf_token, dino_model, backend, sam_checkpoint," >&2
                   echo "  cameras, prompt_points, prompt_root, min_persistence_ratio," >&2
                   echo "  low_texture, use_gpu, skip_p4b, keep_going, allow_mixed_capture,
                   strict_midribs." >&2
                   exit 1 ;;
            esac
        done < "$file"
    done
}

LOADED_CONF=()
[[ -z "$DATASET" ]] || resolve_dataset "$DATASET"

if [[ ${#CONF_FILES[@]} -gt 0 ]]; then
    load_config "${CONF_FILES[@]}"
else
    CANDIDATES=("$HOME/.config/blender_leaf_generator/pipeline.conf" "$PWD/pipeline.conf")
    # With --workdir there is no $DATASET to hang the per-session and
    # per-specimen files off -- but the workdir is <dataset>/plant by
    # convention, so the same two files are findable from it. Without this a
    # pipeline.conf beside the dataset was silently ignored on exactly the
    # runs that resume with --skip-to, which is most of them.
    CONF_BASE="${DATASET:-$([[ -n "$WORKDIR" ]] && dirname "${WORKDIR%/}")}"
    [[ -z "$CONF_BASE" ]] || CANDIDATES+=("$(dirname "${CONF_BASE%/}")/pipeline.conf" "${CONF_BASE%/}/pipeline.conf")
    load_config "${CANDIDATES[@]}"
fi

[[ -n "$WORKDIR" ]] || { echo "--workdir (or a dataset directory) is required" >&2; usage 1; }
[[ ${#VIDEOS[@]} -gt 0 || ${#PHOTOS[@]} -gt 0 || -d "$WORKDIR/p1/frames" ]] || {
    echo "--video or --photos is required unless $WORKDIR/p1/frames already exists" >&2
    usage 1; }

# Everything resolved, printed before anything runs. A path that came from a
# config file or a dataset directory is one you did not type on this run, so
# it is exactly the kind that goes unnoticed when it is wrong.
print_plan() {
    echo "resolved inputs"
    [[ -z "$DATASET" ]] || echo "  dataset       ${DATASET%/}"
    for f in "${LOADED_CONF[@]+"${LOADED_CONF[@]}"}"; do echo "  config        $f"; done
    if [[ ${#PHOTOS[@]} -gt 0 ]]; then
        for d in "${PHOTOS[@]}"; do
            local n; n=$(find "$d" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | wc -l)
            echo "  photos        ${d%/}  ($n)"
        done
    fi
    for v in "${VIDEOS[@]+"${VIDEOS[@]}"}"; do echo "  video         $v"; done
    echo "  workdir       $WORKDIR"
    [[ -z "$PROMPT_BANK" ]]  || echo "  prompt bank   $PROMPT_BANK"
    [[ -z "$SEED_BANK" ]]    || echo "  seed bank     $SEED_BANK"
    [[ -z "$ARCHITECTURE" ]] || echo "  architecture  $ARCHITECTURE"
    [[ -z "$CAMERAS" ]]      || echo "  cameras       $CAMERAS"
    [[ "$LOW_TEXTURE" == 1 ]] && echo "  low-texture   on"
    [[ "$USE_GPU" == 1 ]]     && echo "  gpu           on"
    [[ -z "$HF_TOKEN_ARG" ]] || echo "  hf token      set (${#HF_TOKEN_ARG} chars)"
    echo "  phases        ${SKIP_TO:-p1p2} -> ${STOP_AFTER:-p6}"
    echo "  geometry      $GEOMETRY_BACKEND$([[ "$BUNDLE_ADJUST" == 1 ]] && echo " + bundle adjustment")"
    [[ -z "$IMAGE_RESOLUTION" ]] || echo "  model res     $IMAGE_RESOLUTION"
    [[ -z "$MODEL_PYTHON" ]] || echo "  model python  $MODEL_PYTHON"

    # A bank belonging to another dataset is legitimate -- that is what banks
    # are for -- but it is also what a stale path looks like, so say so.
    local ds; ds="$(cd "${DATASET:-$WORKDIR}" 2>/dev/null && pwd || echo "")"
    for pair in "prompt bank:$PROMPT_BANK" "seed bank:$SEED_BANK"; do
        local label="${pair%%:*}" path="${pair#*:}"
        [[ -n "$path" ]] || continue
        if [[ ! -e "$path" ]]; then
            echo "  note: $label does not exist: $path" >&2
        elif [[ -n "$ds" && "$(cd "$(dirname "$path")" && pwd)" != "$ds"* ]]; then
            echo "  note: $label comes from another dataset (fine if deliberate)" >&2
        fi
    done
}
# Phases run in this order; --skip-to jumps in partway and --stop-after ends
# early. Declared here because the phase range decides which dependencies,
# config keys and QC gates this run needs, all of which are settled below.
ORDER=(p1p2 p3 p4a p4b p4c p5 p6)

for name in "$SKIP_TO" "$STOP_AFTER"; do
    [[ -z "$name" ]] && continue
    [[ " ${ORDER[*]} " == *" $name "* ]] || {
        echo "unknown phase '$name' -- expected one of: ${ORDER[*]}" >&2; exit 1; }
done

# Whether this run includes <phase>. A pure predicate over the range, kept
# apart from should_run's one-shot state machine, and honouring the two
# phases that skip themselves for reasons unrelated to the range.
will_run_phase() {
    local phase="$1" first="${SKIP_TO:-${ORDER[0]}}" last="${STOP_AFTER:-${ORDER[-1]}}"
    local i target=-1 from=-1 to=-1
    for i in "${!ORDER[@]}"; do
        [[ "${ORDER[$i]}" == "$phase" ]] && target=$i
        [[ "${ORDER[$i]}" == "$first" ]] && from=$i
        [[ "${ORDER[$i]}" == "$last" ]]  && to=$i
    done
    (( target >= 0 && from >= 0 && to >= 0 && target >= from && target <= to )) || return 1
    [[ "$phase" == "p4b" && "$SKIP_P4B" == 1 ]] && return 1
    # P4c skips itself when it has no labelled examples to classify against.
    if [[ "$phase" == "p4c" && "$BACKEND" != "sam" && -z "$SEEDS_FILE" \
          && ${#SEEDS[@]} -eq 0 && -z "$SEED_BANK" ]]; then
        return 1
    fi
    return 0
}

case "$GEOMETRY_BACKEND" in
    colmap|vggt|vggt_omega|mapanything) ;;
    *) echo "unknown --geometry-backend '$GEOMETRY_BACKEND' -- colmap, vggt, vggt_omega or mapanything" >&2
       exit 1 ;;
esac

# Two flags with "backend" in the name land in different phases, and the
# names are close enough that a geometry model passed to the classifier
# would otherwise be accepted here and fail deep inside P4c.
case "$BACKEND" in
    dino|sam) ;;
    colmap|vggt|vggt_omega|mapanything)
        echo "--backend $BACKEND: --backend chooses P4c's organ classifier (dino or sam)." >&2
        echo "  For the P3 geometry model you want:  --geometry-backend $BACKEND" >&2
        exit 1 ;;
    *) echo "unknown --backend '$BACKEND' -- dino or sam (P4c's organ classifier)" >&2
       exit 1 ;;
esac

# A learned backend runs its own branch end to end: every result lands under
# .../experiments/<backend>/ and the baseline is never written to. What it
# does NOT do is carve or splat. P4a and P4b exist to manufacture a cloud
# COLMAP cannot provide, and a pointmap already is one -- so they are skipped
# rather than run on geometry that does not need them. --carve-check puts P4a
# back, not as a dependency but as a validator: the hull is an upper bound, so
# learned points outside it are suspect.
if [[ "$GEOMETRY_BACKEND" != "colmap" ]]; then
    if [[ "$CARVE_CHECK" == 1 ]]; then
        echo "  $GEOMETRY_BACKEND: P4a will carve a hull as a check on the learned cloud"
    fi
fi


# --compare runs the branches one after another in this same script, so the
# three-way comparison is one command rather than three remembered ones.
#
# What is shared is shared exactly once. P1 and P2 are branch-independent, the
# COLMAP model is what --poses-from colmap hands MapAnything, and the P4c
# class maps are 2D -- classification reads frames and masks and never sees a
# cloud. Running the classifier once and fusing it three times is not only
# faster, it is the fairer experiment: the branches then differ in geometry
# alone, which is the whole question being asked.
run_comparison() {
    local requested=() ordered=() learned=() branch status=0
    local -a child=()
    local -a failed=()
    IFS=',' read -r -a requested <<< "$COMPARE"

    for branch in "${requested[@]}"; do
        case "$branch" in
            colmap|vggt|vggt_omega|mapanything) ;;
            *) echo "--compare: unknown backend '$branch' -- colmap, vggt, vggt_omega, mapanything" >&2
               return 1 ;;
        esac
        if [[ "$branch" == "colmap" ]]; then ordered+=("$branch"); else learned+=("$branch"); fi
    done
    ordered+=("${learned[@]+"${learned[@]}"}")
    [[ ${#ordered[@]} -gt 0 ]] || { echo "--compare: no backends given" >&2; return 1; }

    # Everything the parser already consumed is replayed to each child, minus
    # the flags this driver decides for itself.
    local -a orig=("${ORIGINAL_ARGS[@]+"${ORIGINAL_ARGS[@]}"}")
    local i=0 arg
    while (( i < ${#orig[@]} )); do
        arg="${orig[$i]}"
        case "$arg" in
            --geometry-backend|--skip-to|--stop-after|--branch-flags)
                i=$(( i + 2 )); continue ;;
            --compare)
                # Its value is optional, so only a following non-flag is one.
                if (( i + 1 < ${#orig[@]} )) && [[ "${orig[$(( i + 1 ))]}" != --* ]]; then
                    i=$(( i + 2 ))
                else
                    i=$(( i + 1 ))
                fi
                continue ;;
        esac
        child+=("$arg")
        i=$(( i + 1 ))
    done

    echo ""
    echo "=== comparing branches: ${ordered[*]} ==="
    echo "  shared once: P1 frames, P2 masks, the COLMAP model, and the P4c class maps."
    echo "  per branch:  P3 -> P4c fusion -> P5 -> P6, under .../experiments/<backend>/"
    echo ""

    # Where each branch picks up. A comparison is normally run more than once
    # -- new seeds, a re-tuned P5 -- and re-solving a P3 that is already on
    # disk costs many minutes and changes nothing. So a branch starts at the
    # first phase whose input is missing, and says which and why. To force a
    # phase to run again, delete its directory.
    local -a steps=()
    resume_from() {   # resume_from <backend> -> the --skip-to phase, or "" for all
        local b="$1"
        if [[ "$b" == "colmap" ]]; then
            [[ -f "$WORKDIR/p4b/surface.ply" ]] && { echo p4c; return; }
            [[ -d "$WORKDIR/p3/sparse/best" ]] && { echo p4a; return; }
            echo ""
        else
            [[ -d "$WORKDIR/p3/experiments/$b/sparse/best" ]] && { echo p4c; return; }
            echo p3
        fi
    }

    if [[ " ${ordered[*]} " != *" colmap "* && ! -d "$WORKDIR/p3/sparse/best" ]]; then
        # No COLMAP branch was asked for, but the learned backends still need
        # P1, P2 and -- for --poses-from colmap -- the COLMAP model itself.
        steps+=("prefix|--geometry-backend colmap --stop-after p3")
    fi
    # Flags that belong to one branch only. MapAnything defaults to COLMAP's
    # poses because that is the configuration being compared -- solving its
    # own poses is a different experiment, and a markedly worse one here. A
    # --branch-flags entry, or a global --poses-from, replaces the default.
    if [[ -z "${BRANCH_FLAGS[mapanything]:-}" && -z "$POSES_FROM" ]]; then
        BRANCH_FLAGS[mapanything]="--poses-from colmap"
    fi
    for branch in "${ordered[@]}"; do
        local per_branch="${BRANCH_FLAGS[$branch]:-}" from
        from="$(resume_from "$branch")"
        if [[ -n "$from" && "$from" != "p3" ]]; then
            # Its geometry is already solved, so neither COLMAP nor a learned
            # exporter runs -- which is why a re-run needs no model
            # environment for a branch whose P3 is on disk.
            echo "  $branch: geometry already on disk, resuming at $from"
            [[ "$from" == "p3" ]] || per_branch=""
        elif [[ -n "$per_branch" ]]; then
            echo "  $branch: $per_branch"
        fi
        if [[ "$branch" == "colmap" ]]; then
            steps+=("colmap|${from:+--skip-to $from} --geometry-backend colmap $per_branch")
        else
            steps+=("$branch|--skip-to ${from:-p3} --geometry-backend $branch --reuse-class-maps $per_branch")
        fi
    done

    local step name extra
    for step in "${steps[@]}"; do
        name="${step%%|*}"; extra="${step#*|}"
        echo ""
        echo "================================================================"
        echo "  branch $name   ($(date '+%H:%M:%S'))"
        echo "================================================================"
        if [[ "$DRY_RUN" == 1 ]]; then
            echo "  would run: ${BASH_SOURCE[0]} ${child[*]+${child[*]}} $extra"
            continue
        fi
        # shellcheck disable=SC2086
        if ! "${BASH_SOURCE[0]}" ${child[@]+"${child[@]}"} $extra; then
            status=1
            failed+=("$name")
            echo ""
            echo "  branch $name FAILED -- continuing with the rest, so one bad backend" >&2
            echo "  does not cost you the branches that did work." >&2
        fi
    done

    [[ "$DRY_RUN" == 1 ]] && { echo ""; echo "--dry-run: nothing was run."; return 0; }

    echo ""
    echo "================================================================"
    echo "  comparison  ($(date '+%H:%M:%S'))"
    echo "================================================================"
    # Resolved here rather than reused: the driver runs before the script
    # sets REPO_ROOT and PY, and both are cheap to work out again.
    local repo py
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    py="${POSE_PYTHON:-$(command -v python3 || command -v python)}"
    "$py" "$repo/scripts/compare_branches.py" "$WORKDIR" \
        --branches "$(IFS=,; echo "${ordered[*]}")" \
        --json "$WORKDIR/branch_comparison.json" || true

    echo ""
    echo "See all three in one Blender scene, side by side:"
    echo "  ./scripts/view_in_blender.sh $WORKDIR --compare $(IFS=,; echo "${ordered[*]}")"
    echo ""
    echo "One branch on its own:"
    echo "  ./scripts/view_in_blender.sh $WORKDIR --geometry-backend ${ordered[-1]}"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo ""
        echo "  branches that failed: ${failed[*]}   (see $WORKDIR/pipeline.log)" >&2
    fi
    return $status
}

print_plan
if [[ -n "$COMPARE" ]]; then
    run_comparison
    exit $?
fi
if [[ "$DRY_RUN" == 1 ]]; then
    echo
    echo "--dry-run: nothing was run."
    exit 0
fi
echo

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
#
# Only what the phases in *this* run import, though. Demanding the union of
# every phase's dependencies made a torch-free carve-only run impossible on
# a base `pip install -e .`, while never checking pycolmap -- which P3, P4a
# and P5 all need and no base install provides.
phase_modules() {
    case "$1" in
        p1p2) echo "numpy cv2 torch" ;;              # SAM2 masking
        p3)   echo "numpy cv2 pycolmap" ;;           # COLMAP, or scoring a learned model
        p4a)  echo "numpy cv2 pycolmap" ;;           # silhouette carving only
        p4b)  echo "numpy cv2 pycolmap torch gsplat" ;;
        p4c)  echo "numpy cv2 torch $([[ "$BACKEND" == "sam" ]] && echo sam2 || echo transformers)" ;;
        p5)   echo "numpy pycolmap" ;;
        p6)   echo "numpy" ;;
    esac
}

# The extra that installs each one, so the fix is a command and not a hunt.
module_source() {
    case "$1" in
        pycolmap)     echo 'pip install -e ".[skeleton]"   (or ".[skeleton-gpu]" for GPU SIFT)' ;;
        torch)        echo 'a CUDA-matching torch build: https://pytorch.org/get-started/locally/' ;;
        gsplat)       echo 'pip install -e ".[splat]"      (or run with --skip-p4b)' ;;
        transformers) echo 'pip install -e ".[semantic]"' ;;
        sam2)         echo 'see setup_env.sh -- SAM2 installs from its own repository' ;;
        cv2)          echo 'pip install -e ".[segment]"' ;;
        *)            echo 'pip install -e .' ;;
    esac
}

phase_cli() {
    case "$1" in
        p1p2) echo "pose_estimator.cli.segment" ;;
        p3)   [[ "$GEOMETRY_BACKEND" == "colmap" ]] && echo "pose_estimator.cli.pose" \
                                                    || echo "pose_estimator.cli.geometry" ;;
        p4a)  echo "pose_estimator.cli.hull" ;;
        p4b)  echo "pose_estimator.cli.surface" ;;
        p4c)  echo "pose_estimator.cli.semantic" ;;
        p5)   echo "pose_estimator.cli.structure" ;;
        p6)   echo "pose_estimator.cli.leaf_model" ;;
    esac
}

REQUIRED="pose_estimator"
for name in "${ORDER[@]}"; do
    will_run_phase "$name" || continue
    REQUIRED="$REQUIRED $(phase_modules "$name")"
done
CLI_MODULES=""
for name in "${ORDER[@]}"; do
    will_run_phase "$name" || continue
    CLI_MODULES="$CLI_MODULES $(phase_cli "$name")"
done
# Two passes, because neither alone is enough. find_spec catches the heavy
# packages a phase imports *inside* a function -- pycolmap is deliberately
# lazy, so importing the CLI module never reveals it missing. Importing the
# CLI module catches everything the phase pulls in transitively at import
# time, which no hand-maintained list stays in step with.
MISSING="$("$PY" - "$REQUIRED" "--" $CLI_MODULES <<'PYCHECK' 2>/dev/null
import importlib
import importlib.util as u
import sys

wanted = sys.argv[1].split()
clis = sys.argv[sys.argv.index("--") + 1:]
missing = []
for name in wanted:
    try:
        if u.find_spec(name) is None:
            missing.append(name)
    except (ImportError, ValueError):      # broken or namespace-shadowed package
        missing.append(name)
for name in clis:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing.append(exc.name or name)
    except Exception:
        pass    # imports, but objects at import time: not this check's business
print(" ".join(dict.fromkeys(missing)))
PYCHECK
)"
if [[ -n "${MISSING// /}" ]]; then
    echo "ERROR: $PY cannot run ${SKIP_TO:-p1p2} -> ${STOP_AFTER:-p6}; missing: $MISSING" >&2
    for name in $MISSING; do
        printf '  %-14s %s\n' "$name" "$(module_source "$name")" >&2
    done
    echo "" >&2
    echo "  Or build a complete environment:       ./setup_env.sh" >&2
    echo "  Or point at another interpreter:       POSE_PYTHON=/path/to/python" >&2
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
# edits here. --sam-checkpoint, $SAM2_CHECKPOINT (a file or a directory) and
# $SAM_CHECKPOINT_DIR all win -- see scripts/sam_checkpoints.sh, which
# setup_env.sh sources too, so the download and the search cannot point at
# different directories.
# shellcheck source=scripts/sam_checkpoints.sh
source "$REPO_ROOT/scripts/sam_checkpoints.sh"
[[ -n "$HF_TOKEN_ARG" ]] && export HF_TOKEN="$HF_TOKEN_ARG"

mkdir -p "$WORKDIR"
LOG="$WORKDIR/pipeline.log"
# Everything below is tee'd, so a finished run leaves a readable record.
exec > >(tee -a "$LOG") 2>&1
echo "=== run started $(date '+%Y-%m-%d %H:%M:%S') ==="

# Phases run in order; --skip-to jumps in partway on an existing workdir and
# --stop-after ends early. Both name a phase from this list.
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

phase() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

# Where this branch's later phases read and write. The baseline keeps the
# historical paths; a learned backend gets a sibling under experiments/.
if [[ "$GEOMETRY_BACKEND" == "colmap" ]]; then
    P4C_DIR="$WORKDIR/p4c"; P5_DIR="$WORKDIR/p5"; P6_DIR="$WORKDIR/p6"
else
    P4C_DIR="$WORKDIR/p4c/experiments/$GEOMETRY_BACKEND"
    P5_DIR="$WORKDIR/p5/experiments/$GEOMETRY_BACKEND"
    P6_DIR="$WORKDIR/p6/experiments/$GEOMETRY_BACKEND"
fi

# Stop the run when a phase's QC shows a failure nothing downstream can
# absorb. Only catastrophic signatures gate -- advisory failures happen on
# good runs (a root blinking behind the pliers fails area smoothness, a root
# ball in front of the yellow handle fails the colour contamination check)
# and never stop anything. Each signature below was measured in a real
# disaster that previously ran to completion and produced an empty result:
# a "plant" mask that was 99% pliers, and a hull carved at IoU 0.026 against
# masks rewritten mid-run. --keep-going pushes past a gate knowingly.
gate() {  # gate <phase> <qc-json>
    local why=""
    why="$($PY "$REPO_ROOT/scripts/qc_gate.py" "$1" "$2")" || true
    if [[ -z "$why" ]]; then
        return 0
    fi
    echo "" >&2
    echo "  FATAL QC after $1 -- the failures below poison every later phase:" >&2
    while IFS= read -r line; do echo "    $line" >&2; done <<< "$why"
    if [[ "$KEEP_GOING" == 1 ]]; then
        echo "    --keep-going given -- continuing anyway." >&2
        return 0
    fi
    echo "    Fix this phase, then resume with --skip-to <next phase>" >&2
    echo "    (see README: Recovering after a bad P2). To push on anyway: --keep-going" >&2
    exit 1
}

# --skip-to means the earlier phases are being taken on trust from a previous
# run, so their QC on disk is the only evidence about what this run builds on
# -- and it is evidence nothing else would look at, because the gate below
# only fires for phases this invocation actually executes. Skipping into the
# middle of a broken workdir is exactly how a run reaches P4c on a solve that
# failed its circle fit half an hour earlier.
if [[ -n "$SKIP_TO" ]]; then
    for name in "${ORDER[@]}"; do
        [[ "$name" == "$SKIP_TO" ]] && break
        case "$name" in
            p1p2) gate p1p2 "$WORKDIR/p2/qc.json" ;;
            p3)   if [[ "$GEOMETRY_BACKEND" == "colmap" ]]; then
                      gate p3 "$WORKDIR/p3/poses.json"
                  else
                      gate p3 "$WORKDIR/p3/experiments/$GEOMETRY_BACKEND/poses.json"
                  fi ;;
            p4a)  if [[ "$GEOMETRY_BACKEND" == "colmap" ]]; then
                      gate p4a "$WORKDIR/p4/hull.json"
                  else
                      gate p4a "$WORKDIR/p4/experiments/$GEOMETRY_BACKEND/hull.json"
                  fi ;;
        esac
    done
fi

if should_run p1p2; then
    phase "P1+P2  sharpest frames + SAM2 plant/holder masks   -> $WORKDIR/p1, p2"
    n_passes=$(( ${#VIDEOS[@]} + ${#PHOTOS[@]} ))
    [[ $n_passes -gt 1 ]] && echo "  $n_passes capture passes, tracked separately, solved together in P3"
    # --sam-checkpoint applies here too, not just to P4c: it used to be read
    # only by the later phase, so pointing it at relocated weights left P2
    # searching the default locations and failing.
    SAM_CKPT="$SAM_CHECKPOINT"
    if [[ -n "$SAM_CKPT" && -d "$SAM_CKPT" ]]; then
        SAM_CKPT="${SAM_CKPT%/}/$SAM2_CHECKPOINT_NAME"
    fi
    if [[ -n "$SAM_CKPT" && ! -f "$SAM_CKPT" ]]; then
        echo "ERROR: --sam-checkpoint $SAM_CHECKPOINT does not exist." >&2
        exit 1
    fi
    if [[ -z "$SAM_CKPT" ]] && ! SAM_CKPT="$(find_sam_checkpoint)"; then
        sam_checkpoint_error "$SAM2_CHECKPOINT_NAME"
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
        ${VIDEOS[0]:+--video} ${VIDEOS[@]+"${VIDEOS[@]}"} \
        ${PHOTOS[0]:+--photos} ${PHOTOS[@]+"${PHOTOS[@]}"} --workdir "$WORKDIR" \
        --checkpoint "$SAM_CKPT" \
        ${PROMPT_BANK:+--prompt-bank "$PROMPT_BANK"} \
        ${PROMPT_BANK:+--dino-model "$DINO_MODEL"} \
        ${PROMPT_POINTS:+--prompt-points "$PROMPT_POINTS"} \
        $([[ "$ALLOW_MIXED" == 1 ]] && echo --allow-mixed-capture)
    gate p1p2 "$WORKDIR/p2/qc.json"
fi

if should_run p3; then
    if [[ "$GEOMETRY_BACKEND" == "colmap" ]]; then
        phase "P3     camera poses, masked COLMAP                 -> $WORKDIR/p3"
        $PY -m pose_estimator.cli.pose --workdir "$WORKDIR" \
            $([[ "$USE_GPU" == 1 ]] && echo --use-gpu) \
            $([[ -n "$CAMERAS" ]] && echo --cameras "$CAMERAS") \
            $([[ "$LOW_TEXTURE" == 1 ]] && echo --low-texture)
        gate p3 "$WORKDIR/p3/poses.json"
    else
        phase "P3x    $GEOMETRY_BACKEND poses + cloud    -> $WORKDIR/p3/experiments/$GEOMETRY_BACKEND"
        echo "  the COLMAP baseline in $WORKDIR/p3 is read as the comparison reference"
        echo "  and is not modified."
        $PY -m pose_estimator.cli.geometry --workdir "$WORKDIR" \
            --backends "$GEOMETRY_BACKEND" \
            ${MODEL_PYTHON:+--model-python "$MODEL_PYTHON"} \
            ${IMAGE_RESOLUTION:+--image-resolution "$IMAGE_RESOLUTION"} \
            ${INTRINSICS_FROM:+--intrinsics-from "$INTRINSICS_FROM"} \
            ${POSES_FROM:+--poses-from "$POSES_FROM"} \
            $([[ "$BUNDLE_ADJUST" == 1 ]] && echo --bundle-adjust)
        gate p3 "$WORKDIR/p3/experiments/$GEOMETRY_BACKEND/poses.json"
    fi
fi

if should_run p4a; then
    if [[ "$GEOMETRY_BACKEND" == "colmap" ]]; then
        phase "P4a    visual hull by silhouette carving           -> $WORKDIR/p4"
        $PY -m pose_estimator.cli.hull --workdir "$WORKDIR" --resolution 256
        gate p4a "$WORKDIR/p4/hull.json"
    elif [[ "$CARVE_CHECK" == 1 ]]; then
        phase "P4a    hull carved from $GEOMETRY_BACKEND poses (a check, not the geometry)"
        $PY -m pose_estimator.cli.hull --workdir "$WORKDIR" --resolution 256 \
            --geometry-backend "$GEOMETRY_BACKEND"
        gate p4a "$WORKDIR/p4/experiments/$GEOMETRY_BACKEND/hull.json"
        echo ""
        echo "  compare against the baseline hull:"
        echo "    $WORKDIR/p4/hull.json  vs  $WORKDIR/p4/experiments/$GEOMETRY_BACKEND/hull.json"
    else
        phase "P4a    SKIPPED -- $GEOMETRY_BACKEND already produces a dense cloud"
        echo "  Carving exists to manufacture geometry COLMAP cannot give; a pointmap is"
        echo "  already a surface sample, so P4c labels it directly. --carve-check carves"
        echo "  anyway, as an upper bound to judge the learned cloud against."
    fi
fi

if should_run p4b && [[ "$GEOMETRY_BACKEND" != "colmap" ]]; then
    phase "P4b    SKIPPED -- $GEOMETRY_BACKEND needs no surfel pass"
    echo "  P4b refines the P4a hull into a thin surface. The learned cloud is already"
    echo "  a surface sample, so there is nothing to thin. Its normals are estimated by"
    echo "  local PCA in P4c rather than fitted against the photographs, which is the"
    echo "  real cost of skipping this."
elif should_run p4b && [[ "$SKIP_P4B" == 0 ]]; then
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
        phase "P4c    organ labels + coloured clouds            -> $P4C_DIR"
        echo "  two stages: classify (frames -> p4c/class_maps) then fuse (maps -> labels)."
        echo "  Run them separately with pose-classify / pose-fuse when debugging -- the"
        echo "  class maps are what tell you whether a bad label came from the 2D"
        echo "  classifier or from the multi-view voting."
        # The class maps are 2D and belong to no branch: classification reads
        # frames and P2 masks and never touches a cloud. --reuse-class-maps
        # skips straight to fusion when they are already on disk, which is
        # both faster across a comparison and fairer -- the branches then
        # differ in geometry alone rather than in two independent DINO runs.
        if [[ "$REUSE_CLASS_MAPS" == 1 && -f "$WORKDIR/p4c/classify.json" ]]; then
            echo "  --reuse-class-maps: fusing $WORKDIR/p4c/class_maps, classify not re-run."
            $PY -m pose_estimator.cli.fuse \
                --workdir "$WORKDIR" --geometry-backend "$GEOMETRY_BACKEND"
        else
            $PY -m pose_estimator.cli.semantic \
                --workdir "$WORKDIR" --backend "$BACKEND" --dino-model "$DINO_MODEL" \
                --geometry-backend "$GEOMETRY_BACKEND" \
                ${HF_TOKEN_ARG:+--hf-token "$HF_TOKEN_ARG"} \
                ${SAM_CKPT_P4C:+--checkpoint "$SAM_CKPT_P4C"} \
                ${SEEDS_FILE:+--seeds-file "$SEEDS_FILE"} \
                ${SEED_BANK:+--seed-bank "$SEED_BANK"} \
                ${SEED_FRAME:+--seed-frame "$SEED_FRAME"} \
                ${SEEDS[0]:+--seeds} ${SEEDS[@]+"${SEEDS[@]}"}
        fi
    fi
fi

if should_run p5; then
    if [[ -f "$P4C_DIR/labels.npy" ]]; then
        phase "P5     stem centreline + leaf instances           -> $P5_DIR"
        $PY -m pose_estimator.cli.structure --workdir "$WORKDIR" \
            --geometry-backend "$GEOMETRY_BACKEND" \
            ${ARCHITECTURE:+--architecture "$ARCHITECTURE"} \
            $([[ "$STRICT_MIDRIBS" == 1 ]] && echo --strict-midribs) \
            ${PERSISTENCE:+--min-persistence-ratio "$PERSISTENCE"}
    else
        phase "P5     SKIPPED -- no $P4C_DIR/labels.npy"
        echo "  P5 is driven by the P4c organ labels; run P4c first."
    fi
fi

if should_run p6; then
    if [[ -f "$P5_DIR/leaf_points.npy" ]]; then
        phase "P6     per-leaf midrib, curvature, width          -> $P6_DIR"
        $PY -m pose_estimator.cli.leaf_model --workdir "$WORKDIR" \
            --geometry-backend "$GEOMETRY_BACKEND"
    else
        phase "P6     SKIPPED -- no $P5_DIR/leaf_points.npy"
        echo "  P6 fits midribs to P5's per-leaf point subsets; run P5 first."
    fi
fi

phase "done  ($(date '+%H:%M:%S'))"
cat <<EOF
Open in Blender:
  ./scripts/view_in_blender.sh $WORKDIR$([[ "$GEOMETRY_BACKEND" == "colmap" ]] || echo " --geometry-backend $GEOMETRY_BACKEND")
  $P6_DIR/midribs.ply           per-leaf midrib curves
  $P5_DIR/structure.ply         leaf / stem / root points, coloured
  $P4C_DIR/labels_vis.ply       cloud coloured by organ (leaf/stem/root)
  $P4C_DIR/leaf_instances.ply   leaf points, one colour per leaf
  $P4C_DIR/confidence.ply       cloud coloured by vote confidence
  $WORKDIR/p4b/surface.ply          the uncoloured surface P4c labelled
  $WORKDIR/p4/hull.ply              visual hull mesh

Measurements: $P6_DIR/leaves.json
  Per leaf: arclength, insertion angle, azimuth, width profile along the midrib.
  Units are COLMAP units, NOT metric -- no scale reference is solved yet.

QC reports -- read before trusting anything:
  $WORKDIR/p2/qc.json   $WORKDIR/p3/poses.json   $WORKDIR/p4/hull.json
  $WORKDIR/p4b/p4b.json $P4C_DIR/qc.json     $P5_DIR/p5.json
  $P6_DIR/p6.json

Compare this branch against the others on the deliverable, not on geometry:
  ./run_pipeline.sh <dataset> --compare              run all three branches
  ./scripts/compare_branches.py $WORKDIR    leaf counts + midribs
  ./scripts/view_in_blender.sh $WORKDIR --compare    all three, side by side

Check the 2D classification before blaming the 3D labels:
  $WORKDIR/p4c/diag/parts_*.jpg     photograph | classification, side by side
  $WORKDIR/p4c/class_maps/          the per-frame maps the fusion voted on
  (2D and branch-independent: every geometry branch votes on these same maps)

Is the structure real? These two answer it faster than any number:
  $P5_DIR/diag/skeleton_*.jpg   stem + leaf axes drawn on the photographs
  $P6_DIR/diag/midribs_3d.png   midribs and width profiles

Diagnostics: $WORKDIR/p*/diag/
Full log:    $LOG
EOF
