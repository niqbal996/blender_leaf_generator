#!/usr/bin/env bash
# Fine point clouds for one plant, next to the P4a/P4b baselines (fine_pc branch).
#
#   ./scripts/run_fine_pc.sh <workdir>                       # P4m -> P4g -> comparison
#   ./scripts/run_fine_pc.sh <workdir> --only mvs            # one step: mvs | gs | compare
#   ./scripts/run_fine_pc.sh <workdir> --source-root <dir>   # originals not at the manifest's paths
#   ./scripts/run_fine_pc.sh <workdir> --skeleton2d <dir>    # s2d output (default: the workdir)
#
# Needs a finished pipeline run (P1-P5) and, for the comparison, a
# scripts/skeleton_2d.py run with the current code (its midribs are the
# yardstick). Each step logs to <workdir>/<step dir>/run.log. P4m needs
# pycolmap with CUDA; P4g needs a CUDA GPU with gsplat. Neither writes outside
# its own folder (p4m/, p4g/, compare_clouds/).
#
# Extra flags after `--` go to the one step --only names, e.g.
#   ./scripts/run_fine_pc.sh <workdir> --only gs -- --scale 0.5
set -euo pipefail

WD="${1:?usage: run_fine_pc.sh <workdir> [--only mvs|gs|compare] [--source-root DIR] [--skeleton2d DIR]}"
shift
if [[ "$WD" == -* || ! -d "$WD" ]]; then
    echo "first argument must be the plant workdir, got '$WD' (is \$WD set in this shell?)" >&2
    exit 1
fi
ONLY="" ; SRC=() ; S2D="$WD" ; EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --source-root) SRC=(--source-root "$2"); shift 2 ;;
        --skeleton2d) S2D="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "unknown option $1" >&2; exit 1 ;;
    esac
done
if [[ ${#EXTRA[@]} -gt 0 && ( -z "$ONLY" || "$ONLY" == compare ) ]]; then
    echo "extra flags after -- need --only mvs or --only gs (the two tools take different flags)" >&2
    exit 1
fi
PY="${PY:-python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

if want mvs; then
    mkdir -p "$WD/p4m"
    echo "=== P4m: PatchMatch MVS on the full-resolution originals -> $WD/p4m (log: p4m/run.log)"
    "$PY" -m pose_estimator.cli.mvs --workdir "$WD" ${SRC[@]+"${SRC[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$WD/p4m/run.log"
fi
if want gs; then
    mkdir -p "$WD/p4g"
    echo "=== P4g: 2DGS surface at full resolution -> $WD/p4g (log: p4g/run.log)"
    "$PY" -m pose_estimator.cli.gs_surface --workdir "$WD" ${SRC[@]+"${SRC[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$WD/p4g/run.log"
fi
if want compare; then
    if [[ ! -f "$S2D/skeleton.json" || ! -f "$S2D/evidence_2d.json" ]]; then
        echo "=== comparison skipped: no skeleton_2d output in $S2D (run scripts/skeleton_2d.py first)"
        exit 0
    fi
    mkdir -p "$WD/compare_clouds"
    echo "=== comparison: P4a / P4b / P4m / P4g on the s2d yardstick -> $WD/compare_clouds"
    "$PY" "$REPO/scripts/compare_clouds.py" --workdir "$WD" --skeleton2d "$S2D" ${SRC[@]+"${SRC[@]}"} \
        --out "$WD/compare_clouds" 2>&1 | tee "$WD/compare_clouds/run.log"
fi
