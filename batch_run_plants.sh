#!/usr/bin/env bash
# Run run_pipeline.sh over many specimens -- a whole capture root, a few dated
# sessions, or hand-picked specimens -- one after another or several at once.
#
#   ./batch_run_plants.sh /netscratch/naeem/blender_assets                 every session
#   ./batch_run_plants.sh /netscratch/naeem/blender_assets/28-09-2026-Naeem   one session
#   ./batch_run_plants.sh /netscratch/naeem/blender_assets/2*-09-2026-Naeem   a shell glob
#   ./batch_run_plants.sh <session>/plant_data/maize_1 <session>/plant_data/sugarbeet_2
#
# A path is read at whichever level it names:
#   root      every child holding a plant_data/ is a session. Nothing else at
#             that level is touched -- checkpoints/, older specimens. (A root
#             with no plant_data/ anywhere is read as one flat session.)
#   session   every directory in <session>/plant_data/, or in <session>/ when
#             there is no plant_data/, is a specimen
#   specimen  a directory with pass*/ subdirectories, videos or photos of its
#             own, or frames already at plant/p1/frames
#
# Each specimen runs as `run_pipeline.sh <specimen>`, so its inputs, its
# workdir (<specimen>/plant) and its pipeline.conf files -- every one from the
# root down -- are resolved exactly as a single run resolves them. A directory
# with nothing to process, and any calibration/, is listed as skipped. Paths
# given twice, or a session and a specimen inside it, run once.
#
# Paths come first. After them, these options belong to the batch:
#
#   --jobs N         run N specimens at the same time (default 1: one by one,
#                    with the pipeline's output on the terminal as usual)
#   --only G[,G]     keep only specimens matching a glob, tried against both
#                    <name> and <session>/<name>:   --only 'sugarbeet_*'
#   --exclude G[,G]  drop them:   --exclude 'test_*,28-09*/maize_1'
#   --launcher CMD   run each specimen as  CMD run_pipeline.sh <specimen> ...
#                    An srun line makes every specimen its own cluster job,
#                    and --jobs then caps how many are in the queue at once.
#                    The repo path and the python env must be visible to it
#                    (export POSE_PYTHON=<env>/bin/python to pin the env).
#   --log-dir DIR    batch logs (default runs/batch/<timestamp>/ in this repo)
#   --list           print the queue and stop
#
# Everything else goes to run_pipeline.sh, unchanged, for every specimen:
#
#   ./batch_run_plants.sh <session> --dry-run             resolved inputs, each
#   ./batch_run_plants.sh <session> --stop-after p1p2
#   ./batch_run_plants.sh <session> --skip-to p5 --jobs 4
#
# --workdir, --video and --photos are refused: they name one specimen's files.
# Activate the env first, as for run_pipeline.sh, or set $POSE_PYTHON.
#
# Logs. Each specimen still writes <specimen>/plant/pipeline.log. The batch
# adds, under its log dir:
#   <session>__<name>.log   everything, including errors raised before the
#                           pipeline log opens (a missing package, a bad path)
#   summary.tsv             status, wall time, start and end per specimen
#   resources.csv           GPU utilisation, GPU memory and the batch's RAM
#                           every 15 s -- what tells you whether --jobs can rise
# A failed specimen does not stop the batch; the exit status is 1 if any did.
#
# Running several at once. Each specimen reads and writes only inside its own
# workdir; the shared caches (HuggingFace weights, gsplat's compiled kernel)
# are file-locked. So --jobs N is safe, and it pays because the phases want
# different hardware. Measured on the A100 for 28-09 sugarbeet_1 (119 frames,
# SAM3, 11 min end to end):
#   P1+P2  4.2 min   GPU   SAM3 -- frames kept on the CPU, so VRAM stays small
#   P3     4.5 min   CPU   COLMAP matching and mapping; SIFT on the GPU
#   P4a    0.4 min   CPU
#   P4b    1.2 min   GPU   gsplat
#   P4c    0.5 min   GPU
#   P5+P6  0.4 min   CPU
# One specimen leaves the GPU idle for about half its run, so a second one
# overlapping it is close to free. Start with --jobs 3 and read resources.csv
# and summary.tsv: if GPU memory nears the card's size, or wall time per
# specimen grows more than the throughput does, step back down. CPU cores
# are the likelier limit than the GPU -- COLMAP matching uses every core it
# is given, so several P3s at once slow each other.

set -uo pipefail     # not -e: one failed specimen must not end the batch

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$REPO_ROOT/run_pipeline.sh"

usage() {
    awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"
    exit "${1:-1}"
}

PATHS=(); PASS=(); ONLY=(); EXCLUDE=()
JOBS=1; LAUNCHER=""; LOG_DIR=""; LIST=0; DRY_RUN=0

# Paths first: run_pipeline.sh options take a varying number of values
# (--video a b c), so a bare word after an option belongs to that option.
while [[ $# -gt 0 && "$1" != -* ]]; do PATHS+=("$1"); shift; done
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs)      JOBS="$2"; shift 2 ;;
        --only)      IFS=',' read -ra globs <<< "$2"; ONLY+=("${globs[@]}"); shift 2 ;;
        --exclude)   IFS=',' read -ra globs <<< "$2"; EXCLUDE+=("${globs[@]}"); shift 2 ;;
        --launcher)  LAUNCHER="$2"; shift 2 ;;
        --log-dir)   LOG_DIR="$2"; shift 2 ;;
        --list)      LIST=1; shift ;;
        -h|--help)   usage 0 ;;
        --workdir|--video|--photos)
            echo "$1 names one specimen's inputs and cannot apply to a whole batch." >&2
            echo "  Give the specimen directories as paths instead." >&2
            exit 1 ;;
        -*)          [[ "$1" == --dry-run ]] && DRY_RUN=1
                     PASS+=("$1"); shift
                     while [[ $# -gt 0 && "$1" != -* ]]; do PASS+=("$1"); shift; done ;;
        *)           echo "unexpected '$1' -- give the paths before any option" >&2; exit 1 ;;
    esac
done

[[ ${#PATHS[@]} -gt 0 ]] || { echo "give at least one root, session or specimen directory" >&2; usage 1; }
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs wants a positive integer, got '$JOBS'" >&2; exit 1; }
[[ -x "$PIPELINE" ]] || { echo "run_pipeline.sh not found (or not executable) at $PIPELINE" >&2; exit 1; }

# --------------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------------
# What run_pipeline.sh can take from a directory. Kept in step with its
# resolve_dataset: pass*/ dirs, videos, *.JPG, or frames already extracted.
has_inputs() {
    local d="$1"
    [[ -d "$d/plant/p1/frames" ]] && return 0
    compgen -G "$d/pass*/" > /dev/null && return 0
    [[ -n "$(find "$d" -maxdepth 1 -type f \( -iname '*.mov' -o -iname '*.mp4' \
                -o -iname '*.jpg' \) -print -quit)" ]]
}

matches_any() {   # matches_any <name> <label> <glob>...
    local name="$1" label="$2" g
    shift 2
    for g in "$@"; do
        # shellcheck disable=SC2053   # unquoted on purpose: a glob match
        [[ "$name" == $g || "$label" == $g ]] && return 0
    done
    return 1
}

QUEUE=(); LABELS=(); SKIPPED=(); N_FILTERED=0
declare -A SEEN=()

add_specimen() {   # add_specimen <dir> <session>
    local dir="${1%/}" session="$2" name label real
    name="$(basename "$dir")"
    label="${session:+$session/}$name"
    real="$(realpath "$dir")"
    [[ -z "${SEEN[$real]:-}" ]] || return 0
    SEEN[$real]=1
    if [[ "${name,,}" == calibration ]]; then
        SKIPPED+=("$label|calibration"); return 0
    fi
    if ! has_inputs "$dir"; then
        SKIPPED+=("$label|no pass*/, videos, photos or frames"); return 0
    fi
    if [[ ${#ONLY[@]} -gt 0 ]] && ! matches_any "$name" "$label" "${ONLY[@]}"; then
        N_FILTERED=$((N_FILTERED + 1)); return 0
    fi
    if [[ ${#EXCLUDE[@]} -gt 0 ]] && matches_any "$name" "$label" "${EXCLUDE[@]}"; then
        N_FILTERED=$((N_FILTERED + 1)); return 0
    fi
    QUEUE+=("$dir"); LABELS+=("$label")
}

add_session() {   # add_session <dir>: its plant_data/ children, or its own
    local dir="${1%/}" base child
    base="$dir/plant_data"
    [[ -d "$base" ]] || base="$dir"
    for child in "$base"/*/; do
        [[ -d "$child" ]] && add_specimen "$child" "$(basename "$dir")"
    done
}

session_of() {    # the session a specimen directory belongs to, for its label
    local parent
    parent="$(dirname "$(realpath "$1")")"
    [[ "$(basename "$parent")" == plant_data ]] && parent="$(dirname "$parent")"
    basename "$parent"
}

expand_path() {
    local p="${1%/}" child sessions=()
    [[ -d "$p" ]] || { echo "no such directory: $p" >&2; exit 1; }
    for child in "$p"/*/; do [[ -d "$child/plant_data" ]] && sessions+=("$child"); done
    # Containers are recognised before specimens: a root or a plant_data/ can
    # have stray videos of its own (turn_table_datasets/DSC_0002.MOV), which
    # would otherwise make the whole tree look like one specimen.
    if [[ "$(basename "$(realpath "$p")")" == plant_data ]]; then
        add_session "$(dirname "$(realpath "$p")")"
    elif [[ -d "$p/plant_data" ]]; then
        add_session "$p"
    elif [[ ${#sessions[@]} -gt 0 ]]; then
        for child in "${sessions[@]}"; do add_session "$child"; done
    elif has_inputs "$p"; then
        add_specimen "$p" "$(session_of "$p")"
    else
        add_session "$p"
    fi
}

for p in "${PATHS[@]}"; do expand_path "$p"; done

# How far a workdir already got, so a re-run's queue says what it will redo.
progress() {
    local w="$1/plant" last="" ph
    for ph in p1 p2 p3 p4 p4b p4c p5 p6; do [[ -d "$w/$ph" ]] && last="$ph"; done
    echo "${last:+workdir has up to $last}"
}

inputs() {   # what resolve_dataset will pick, in its order of preference
    local d="$1" n
    n=$(find "$d" -mindepth 1 -maxdepth 1 -type d -name 'pass*' | wc -l)
    (( n > 0 )) && { echo "$n pass dirs"; return; }
    n=$(find "$d" -maxdepth 1 -type f \( -iname '*.mov' -o -iname '*.mp4' \) | wc -l)
    (( n > 0 )) && { echo "$n videos"; return; }
    n=$(find "$d" -maxdepth 1 -type f -iname '*.jpg' | wc -l)
    (( n > 0 )) && { echo "$n photos"; return; }
    echo "frames only"
}

print_queue() {
    local i entry width=8
    for i in "${!LABELS[@]}"; do (( ${#LABELS[$i]} > width )) && width=${#LABELS[$i]}; done
    echo "queue: ${#QUEUE[@]} specimen(s)$([[ $JOBS -gt 1 ]] && echo ", $JOBS at a time")"
    for i in "${!QUEUE[@]}"; do
        printf '  %3d  %-*s  %-13s %s\n' "$((i + 1))" "$width" "${LABELS[$i]}" \
            "$(inputs "${QUEUE[$i]}")" "$(progress "${QUEUE[$i]}")"
    done
    if [[ ${#SKIPPED[@]} -gt 0 ]]; then
        echo "skipped:"
        for entry in "${SKIPPED[@]}"; do
            printf '       %-*s  (%s)\n' "$width" "${entry%%|*}" "${entry#*|}"
        done
    fi
    [[ "$N_FILTERED" == 0 ]] || echo "  ($N_FILTERED more left out by --only/--exclude)"
    [[ ${#PASS[@]} -eq 0 ]] || echo "run_pipeline.sh args: ${PASS[*]}"
}

print_queue
[[ ${#QUEUE[@]} -gt 0 ]] || { echo "nothing to run" >&2; exit 1; }
[[ "$LIST" == 0 ]] || exit 0

pipeline_cmd() {   # pipeline_cmd <specimen>: the command, launcher included
    CMD=("$PIPELINE" "$1" ${PASS[@]+"${PASS[@]}"})
    # Through bash -c so the launcher string may carry its own quoting
    # (--container-mounts="a,b").
    [[ -z "$LAUNCHER" ]] || CMD=(bash -c "$LAUNCHER \"\$@\"" launcher "${CMD[@]}")
}

# --dry-run prints each specimen's resolved inputs and runs nothing, so it
# goes straight to the terminal, in order, with nothing written to disk --
# and runs here, not through the launcher: there is no work to send anywhere.
if [[ "$DRY_RUN" == 1 ]]; then
    for i in "${!QUEUE[@]}"; do
        echo ""
        echo "================ ${LABELS[$i]} ================"
        "$PIPELINE" "${QUEUE[$i]}" ${PASS[@]+"${PASS[@]}"} < /dev/null
    done
    exit 0
fi

# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------
LOG_DIR="${LOG_DIR:-$REPO_ROOT/runs/batch/$(date '+%Y%m%d-%H%M%S')}"
mkdir -p "$LOG_DIR/.status"
print_queue > "$LOG_DIR/queue.txt"

fmt_dur() { printf '%dh%02dm' $(( $1 / 3600 )) $(( $1 % 3600 / 60 )); }

run_one() {   # run_one <index>
    local i="$1" label="${LABELS[$1]}" log start end status
    log="$LOG_DIR/${label//\//__}.log"
    pipeline_cmd "${QUEUE[$i]}"
    start=$(date +%s)
    echo "[$(date '+%H:%M:%S')] start  $label   log: $log"
    if (( JOBS == 1 )); then
        # One at a time: the terminal shows the run as run_pipeline.sh would,
        # and the log gets the same text with the colour stripped.
        "${CMD[@]}" < /dev/null 2>&1 | tee >(sed -u 's/\x1b\[[0-9;]*m//g' > "$log")
        status=${PIPESTATUS[0]}
    else
        NO_COLOR=1 "${CMD[@]}" < /dev/null > "$log" 2>&1
        status=$?
    fi
    end=$(date +%s)
    local verdict="ok"
    (( status == 0 )) || verdict="FAILED (exit $status)"
    echo "[$(date '+%H:%M:%S')] $( (( status == 0 )) && echo 'done  ' || echo 'FAILED') $label   $(fmt_dur $((end - start)))"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$verdict" "$(fmt_dur $((end - start)))" \
        "$(date -d "@$start" '+%H:%M:%S')" "$(date -d "@$end" '+%H:%M:%S')" "$log" \
        > "$LOG_DIR/.status/$i"
}

# GPU and RAM over the whole batch, so the right --jobs is read off the data
# rather than guessed. RAM is the summed RSS of this batch's processes (the
# node's own total includes other people's jobs); shared pages count twice,
# so it overstates a little.
monitor() {
    local pgid gpu rss
    pgid="$(ps -o pgid= $$ | tr -d ' ')"
    echo "time,gpu_util_pct,gpu_mem_used_mib,gpu_mem_total_mib,batch_rss_mib" > "$LOG_DIR/resources.csv"
    while :; do
        gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
                   --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
        rss="$(ps -eo pgid=,rss= | awk -v g="$pgid" '$1 == g { s += $2 } END { print int(s / 1024) }')"
        echo "$(date '+%H:%M:%S'),${gpu:-,,},$rss" >> "$LOG_DIR/resources.csv"
        sleep 15
    done
}

summary() {
    local i
    {
        printf 'specimen\tstatus\twall\tstart\tend\tlog\n'
        for i in "${!QUEUE[@]}"; do
            if [[ -f "$LOG_DIR/.status/$i" ]]; then cat "$LOG_DIR/.status/$i"
            else printf '%s\tnot run\t-\t-\t-\t-\n' "${LABELS[$i]}"; fi
        done
    } > "$LOG_DIR/summary.tsv"
    echo ""
    echo "================================================================"
    echo "  batch summary  ($(date '+%H:%M:%S'))"
    echo "================================================================"
    cut -f1-5 "$LOG_DIR/summary.tsv" | column -t -s $'\t'
    echo ""
    echo "  $LOG_DIR/summary.tsv"
    [[ -n "$LAUNCHER" ]] || echo "  $LOG_DIR/resources.csv"
}

MON_PID=""
if [[ -z "$LAUNCHER" ]]; then   # with a launcher the work runs on other nodes
    monitor &
    MON_PID=$!
fi
# Ctrl-C or a scheduler's SIGTERM stops every running specimen, not just
# this script: they share its process group.
trap '[[ -z "$MON_PID" ]] || { pkill -P "$MON_PID"; kill "$MON_PID"; } 2>/dev/null' EXIT
trap 'echo "batch interrupted -- stopping running specimens" >&2; summary; trap - INT TERM EXIT; kill 0' INT TERM

echo ""
echo "batch logs: $LOG_DIR"
(( JOBS == 1 )) || echo "follow one:  tail -f $LOG_DIR/<session>__<name>.log"
echo ""

PIDS=()
running=0
for i in "${!QUEUE[@]}"; do
    if (( JOBS == 1 )); then
        run_one "$i"
        continue
    fi
    if (( running >= JOBS )); then
        wait -n   # a specimen finished; the monitor never does
        running=$((running - 1))
    fi
    run_one "$i" &
    PIDS+=($!)
    running=$((running + 1))
done
for pid in ${PIDS[@]+"${PIDS[@]}"}; do wait "$pid" 2>/dev/null; done

summary
if grep -q $'\tFAILED' "$LOG_DIR/summary.tsv"; then exit 1; fi
