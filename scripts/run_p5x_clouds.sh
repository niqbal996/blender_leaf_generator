#!/usr/bin/env bash
# P5x on each point cloud of one plant, each into its own folder (fine_pc branch).
#
#   ./scripts/run_p5x_clouds.sh <workdir>                    # p4a p4b p4m p4g
#   ./scripts/run_p5x_clouds.sh <workdir> p4m p4g            # just these
#
# Writes <workdir>/p5x_<cloud>/ and logs to <workdir>/p5x_<cloud>.log; p5x/ is
# not touched. P4m and P4g are thinned to 0.15 mm first (thin_cloud.py, scale
# from <workdir>/tag_scale.json) into <cloud dir>/surface_p5x.ply, because P5x
# builds a graph over every point. A missing cloud is skipped, and a failed one
# does not stop the others; the exit status is 1 if any failed or was skipped.
set -uo pipefail

WD="${1:?usage: run_p5x_clouds.sh <workdir> [p4a p4b p4m p4g]}"
shift
if [[ "$WD" == -* || ! -d "$WD" ]]; then
    echo "first argument must be the plant workdir, got '$WD' (is \$WD set in this shell?)" >&2
    exit 1
fi
CLOUDS=("$@")
[[ ${#CLOUDS[@]} -eq 0 ]] && CLOUDS=(p4a p4b p4m p4g)
PY="${PY:-python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
declare -A SRC=([p4a]=p4/hull_points.ply [p4b]=p4b/surface.ply [p4m]=p4m/surface.ply [p4g]=p4g/surface.ply)

status=0
for name in "${CLOUDS[@]}"; do
    src="${SRC[$name]:-}"
    if [[ -z "$src" ]]; then echo "=== $name: unknown cloud (p4a p4b p4m p4g)"; status=1; continue; fi
    cloud="$WD/$src"
    if [[ ! -s "$cloud" ]]; then echo "=== $name: no $cloud, skipped"; status=1; continue; fi
    log="$WD/p5x_$name.log"
    echo "=== $name: $cloud ($(date -r "$cloud" '+%F %T')) -> $WD/p5x_$name (log: $log)"
    (
        echo "started $(date '+%F %T') on $(hostname), commit $(git -C "$REPO" rev-parse --short HEAD)"
        if [[ $name == p4m || $name == p4g ]]; then
            mpu=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['mm_per_unit'])" "$WD/tag_scale.json") &&
            "$PY" "$REPO/scripts/thin_cloud.py" "$cloud" "$(dirname "$cloud")/surface_p5x.ply" \
                --mm 0.15 --mm-per-unit "$mpu" &&
            cloud="$(dirname "$cloud")/surface_p5x.ply"
        fi &&
        "$PY" -m pose_estimator.cli.leaf_instances --workdir "$WD" --cloud "$cloud" --out "p5x_$name"
        rc=$?
        echo "finished $(date '+%F %T'), exit $rc"
        exit $rc
    ) > "$log" 2>&1
    rc=$?
    echo "    exit $rc"
    [[ $rc -ne 0 ]] && status=1
done
exit $status
