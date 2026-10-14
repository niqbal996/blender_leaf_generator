#!/usr/bin/env bash
# Queue one Slurm job per plant under a data folder, all at once; each runs the whole pipeline
# (scripts/slurm/plant_job.sh -> run_pipeline.sh, P1 -> apex) in its own GPU container.
#
#   scripts/slurm/submit_plants.sh <root> [options] [-- run_pipeline.sh options]
#
#   scripts/slurm/submit_plants.sh /netscratch/naeem/blender_assets/06-10-2026-Naeem --dry-run
#   scripts/slurm/submit_plants.sh /netscratch/naeem/blender_assets/06-10-2026-Naeem \
#       --partition RTXA6000,A100-40GB --mail naeem.iqbal@dfki.de
#   scripts/slurm/submit_plants.sh <root> --only 'thistle*' -- --rerun-from apex
#
# Run it on the login node (it needs sbatch/squeue, not the pipeline's env).
# <root> is a plant folder (it has pass*/ folders, videos or photos), a session folder
# (its children that are plants; <session>/plant_data/* too), or a root of sessions.
# Folders whose name starts with _ or . are never plants.
#
# Options (defaults in brackets):
#   --partition P     Slurm partition(s) [RTXA6000]; a comma list runs each job on
#                     whichever frees first. Any GPU of 40 GB or more is enough
#                     (measured peak of any step: see docs/apex-plan.md section 0.0b).
#   --time T          [1-00:00:00] -- a plant takes ~4.5 h on an H100, more on an A6000/A100;
#                     a job cut off by the limit resumes where it stopped (stamps) when queued again
#   --cpus N          CPUs per GPU [20]
#   --mem MB          [100000]
#   --image SQSH      container image [/netscratch/naeem/nvcr.io_nvidia_pytorch_26.02-py3_blender.sqsh]
#   --mounts M        [/netscratch/naeem and the repo]
#   --mail ADDR       mail when each job ends or fails
#   --only G[,G]      plant folder name globs to keep;  --exclude G[,G]  to drop
#   --log-dir D       one log per job [<root>/_slurm_logs]
#   --mode M          sbatch [default]: queued and detached, nothing to keep open;
#                     srun: your usual `srun ... bash ...` line per plant, started in the
#                     background with nohup -- keep this login shell (tmux) alive
#   --force           queue a plant even if it looks busy (see below)
#   --dry-run         print each command, queue nothing
#
# A plant is skipped as busy when a job named plant_<name> of yours is already queued or
# running (<name> is <session>__<plant> below a root of sessions), or its workdir's
# pipeline.log changed in the last 15 minutes (a run started
# some other way, e.g. in an interactive container): two runs in one workdir would
# overwrite each other. Follow a job: tail -f <log-dir>/<name>_<jobid>.log, or the plant's
# own <plant>/plant/pipeline.log; squeue -u $USER. The result of each plant is a Blender
# bundle in <plant>/plant/apex/bundle/.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PART="RTXA6000"; TIME="1-00:00:00"; CPUS=20; MEM=100000
IMAGE="/netscratch/naeem/nvcr.io_nvidia_pytorch_26.02-py3_blender.sqsh"
MOUNTS="/netscratch/naeem:/netscratch/naeem,/home/iqbal/blender_leaf_generator:/home/iqbal/blender_leaf_generator"
MAIL=""; ONLY=(); EXCLUDE=(); LOG_DIR=""; MODE="sbatch"; FORCE=0; DRY=0; PASS=()

usage() { awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"; exit "${1:-1}"; }
[[ $# -gt 0 && "$1" != -* ]] || usage 1
ROOT="$(cd "$1" && pwd)" || { echo "no such folder: $1" >&2; exit 1; }
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PART="$2"; shift 2 ;;
        --time)      TIME="$2"; shift 2 ;;
        --cpus)      CPUS="$2"; shift 2 ;;
        --mem)       MEM="$2"; shift 2 ;;
        --image)     IMAGE="$2"; shift 2 ;;
        --mounts)    MOUNTS="$2"; shift 2 ;;
        --mail)      MAIL="$2"; shift 2 ;;
        --only)      IFS=',' read -ra g <<< "$2"; ONLY+=("${g[@]}"); shift 2 ;;
        --exclude)   IFS=',' read -ra g <<< "$2"; EXCLUDE+=("${g[@]}"); shift 2 ;;
        --log-dir)   LOG_DIR="$2"; shift 2 ;;
        --mode)      MODE="$2"; shift 2 ;;
        --force)     FORCE=1; shift ;;
        --dry-run)   DRY=1; shift ;;
        -h|--help)   usage 0 ;;
        --)          shift; PASS=("$@"); break ;;
        *)           echo "unknown option $1 (run_pipeline.sh options go after --)" >&2; exit 1 ;;
    esac
done
[[ "$MODE" == sbatch || "$MODE" == srun ]] || { echo "--mode sbatch or srun, got $MODE" >&2; exit 1; }
LOG_DIR="${LOG_DIR:-$ROOT/_slurm_logs}"

# --- the plants -----------------------------------------------------------------
is_plant() {   # a folder run_pipeline.sh can take as a dataset
    local d="$1" n; n="$(basename "$d")"
    [[ "$n" == _* || "$n" == .* || "$n" == plant* || "$n" == pass* || "$n" == calibration ]] && return 1
    compgen -G "$d/pass*/" > /dev/null && return 0
    [[ -n "$(find "$d" -maxdepth 1 -type f \( -iname '*.mov' -o -iname '*.mp4' -o -iname '*.jpg' \) -print -quit)" ]]
}
plants_in() {  # the plants among a folder's children, and in its plant_data/ (never inside a plant)
    local d
    for d in "$1"/*/ "$1"/plant_data/*/; do
        d="${d%/}"
        [[ -d "$d" ]] && is_plant "$d" && echo "$d"
    done
}
PLANTS=()
if is_plant "$ROOT"; then                          # one plant
    PLANTS=("$ROOT")
else                                               # a session, or a root of sessions (or both)
    mapfile -t PLANTS < <(plants_in "$ROOT")
    for s in "$ROOT"/*/; do
        s="${s%/}"; n="$(basename "$s")"
        [[ "$n" == _* || "$n" == .* ]] && continue
        is_plant "$s" || mapfile -t -O "${#PLANTS[@]}" PLANTS < <(plants_in "$s")
    done
fi
mapfile -t PLANTS < <(printf '%s\n' ${PLANTS[@]+"${PLANTS[@]}"} | awk '!seen[$0]++')
matches() { local n="$1" g; shift; for g in "$@"; do [[ "$n" == $g ]] && return 0; done; return 1; }

[[ -f "$IMAGE" ]] || echo "note: no image at $IMAGE" >&2
[[ "$DRY" == 1 ]] || mkdir -p "$LOG_DIR"
have_squeue=0; command -v squeue > /dev/null && have_squeue=1

printf -v PASS_Q ' %q' ${PASS[@]+"${PASS[@]}"}
[[ ${#PASS[@]} -gt 0 ]] || PASS_Q=""
queued=0; skipped=0
echo "plants under $ROOT ($MODE, partition $PART, time $TIME, $CPUS CPUs, ${MEM} MB):"
for d in ${PLANTS[@]+"${PLANTS[@]}"}; do
    name="$(basename "$d")"
    if [[ ${#ONLY[@]} -gt 0 ]] && ! matches "$name" "${ONLY[@]}"; then continue; fi
    if [[ ${#EXCLUDE[@]} -gt 0 ]] && matches "$name" "${EXCLUDE[@]}"; then continue; fi
    # job and log name: the plant, with its session when it is not right under <root>
    # (sugarbeet_1 exists in several sessions)
    rel="${d#"$ROOT"/}"; rel="${rel//plant_data\//}"
    [[ "$d" == "$ROOT" ]] && rel="$name"
    name="${rel//\//__}"
    job="plant_$name"
    busy=""
    if [[ "$have_squeue" == 1 ]] && [[ -n "$(squeue -h -u "$USER" -n "$job" -o %i 2>/dev/null)" ]]; then
        busy="a job named $job is already queued or running ($(squeue -h -u "$USER" -n "$job" -o '%i %T' | head -1))"
    elif [[ -f "$d/plant/pipeline.log" && -n "$(find "$d/plant/pipeline.log" -mmin -15 2>/dev/null)" ]]; then
        busy="its pipeline.log changed $(( ( $(date +%s) - $(stat -c %Y "$d/plant/pipeline.log") ) / 60 )) min ago -- a run is going"
    fi
    if [[ -n "$busy" && "$FORCE" == 0 ]]; then
        printf '  %-14s skipped: %s (--force queues it anyway)\n' "$name" "$busy"
        skipped=$((skipped + 1)); continue
    fi
    inner="bash $REPO/scripts/slurm/plant_job.sh $(printf %q "$d")$PASS_Q"
    container=(--container-image="$IMAGE" --container-mounts="$MOUNTS" --container-workdir="$REPO")
    res=(--ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu="$CPUS" -p "$PART" --mem="$MEM" --time="$TIME"
         --job-name="$job")
    [[ -n "$MAIL" ]] && res+=(--mail-type=END,FAIL --mail-user="$MAIL")
    if [[ "$MODE" == sbatch ]]; then
        cmd=(sbatch "${res[@]}" --output="$LOG_DIR/${name}_%j.log"
             --wrap="srun -K $(printf '%q ' "${container[@]}")$inner")
    else
        cmd=(srun -K "${res[@]}" "${container[@]}" bash -c "$inner")
    fi
    if [[ "$DRY" == 1 ]]; then
        printf '  %-14s would run:' "$name"; printf ' %q' "${cmd[@]}"; echo
    elif [[ "$MODE" == sbatch ]]; then
        out="$("${cmd[@]}" 2>&1)" && printf '  %-14s %s  (log %s)\n' "$name" "$out" "$LOG_DIR/${name}_<jobid>.log" \
            || { printf '  %-14s sbatch FAILED: %s\n' "$name" "$out"; continue; }
    else
        nohup "${cmd[@]}" > "$LOG_DIR/${name}.log" 2>&1 &
        printf '  %-14s srun started in the background (pid %s, log %s)\n' "$name" "$!" "$LOG_DIR/${name}.log"
    fi
    queued=$((queued + 1))
done
echo "$queued queued, $skipped skipped$([[ "$DRY" == 1 ]] && echo ' (dry run: nothing was queued)')."
[[ "$DRY" == 1 || "$queued" == 0 ]] || echo "watch: squeue -u $USER   |   tail -f $LOG_DIR/<plant>_*.log"
