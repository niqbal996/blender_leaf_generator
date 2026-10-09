#!/usr/bin/env bash
# All four clouds and P5x on each, for one or more plants, one plant after another.
#
#   ./scripts/run_all_clouds.sh <workdir> [<workdir> ...]
#   ./scripts/run_all_clouds.sh /netscratch/naeem/blender_assets/06-10-2026-Naeem/{,plant_data/}*/plant
#
# Per plant, each step runs only if its result is missing:
#   P4b        run_pipeline.sh --skip-to p4b --stop-after p4b   (p4b/surface.ply)
#   P4m        run_fine_pc.sh --only mvs                        (p4m/p4m.json)
#   P4g        run_fine_pc.sh --only gs                         (p4g/p4g.json from a full run:
#              scale 1, 30k steps -- a smoke run's is redone)
#   tag scale  scripts/tag_scale.py --tag-mm 8                  (tag_scale.json)
# then the cloud comparison (if a skeleton_2d run is there) and P5x on
# p4a p4b p4m p4g (run_p5x_clouds.sh, always rerun). A P4g already training on
# a workdir is waited for, not started twice. A directory without p4/ (no
# finished P1-P4a) is skipped. A failed step is logged and the plant moves on;
# the exit status is 1 if anything failed. Logs: each step's own, plus
# <workdir>/all_clouds.log.
set -uo pipefail

[[ $# -gt 0 ]] || { echo "usage: run_all_clouds.sh <workdir> [<workdir> ...]" >&2; exit 1; }
PY="${PY:-python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

p4g_done() {
    "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('scale')==1.0 and d.get('iterations')==30000 else 1)" \
        "$1/p4g/p4g.json" 2>/dev/null
}
step() {   # step <name> <command...>: run, report, remember failures
    echo "--- $(date '+%F %T') $1"
    "${@:2}"
    local rc=$?
    echo "--- $(date '+%F %T') $1: exit $rc"
    [[ $rc -eq 0 ]] || failed=1
    return $rc
}

status=0
for WD in "$@"; do
    WD="${WD%/}"
    if [[ ! -d "$WD/p4" ]]; then echo "##### $WD: no p4/ (pipeline not run to P4a), skipped"; continue; fi
    failed=0
    {
        echo "##### $WD  started $(date '+%F %T') on $(hostname), commit $(git -C "$REPO" rev-parse --short HEAD)"
        [[ -s "$WD/p4b/surface.ply" ]] || step P4b "$REPO/run_pipeline.sh" --workdir "$WD" --skip-to p4b --stop-after p4b
        [[ -f "$WD/p4m/p4m.json" ]] || step P4m "$REPO/scripts/run_fine_pc.sh" "$WD" --only mvs
        while pgrep -f "gs_surface --workdir $WD( |$)" >/dev/null; do
            echo "    P4g is already training on $WD; waiting"; sleep 300
        done
        p4g_done "$WD" || step P4g "$REPO/scripts/run_fine_pc.sh" "$WD" --only gs
        [[ -f "$WD/tag_scale.json" ]] || step "tag scale" "$PY" "$REPO/scripts/tag_scale.py" --workdir "$WD" --tag-mm 8
        step compare "$REPO/scripts/run_fine_pc.sh" "$WD" --only compare
        step P5x "$REPO/scripts/run_p5x_clouds.sh" "$WD"
        echo "##### $WD  finished $(date '+%F %T'), $([[ $failed -eq 0 ]] && echo ok || echo 'SOME STEPS FAILED')"
        exit $failed
    } 2>&1 | tee -a "$WD/all_clouds.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || status=1
done
exit $status
