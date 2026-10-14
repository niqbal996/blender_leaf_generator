#!/usr/bin/env bash
# The apex stage for one plant: young leaves at every shoot tip, fitted to the photos.
#
#   ./scripts/run_apex.sh <workdir>            (run_pipeline.sh's phase `apex` calls this)
#
# Needs: P5x on the P4b surface (p5x/) and on P4g (p5x_p4g/), the s2d run (s2d/),
# tag_scale.json. Writes <workdir>/apex/ only:
#
#   regions/   apex_regions.py   where the apices are (hand boxes if drawn, else
#                                every shoot end of P5x's tree / a rosette's crown)
#   zoom/      apex_zoom.py      SAM3 re-run on each apex's square, in every frame
#   s2d_zoom/  skeleton_2d.py    2D leaf lines on those masks -> geometry candidates
#   select/    apex_select.py    the starting leaves (steps 1-2 of docs/apex-plan.md)
#   fit/       apex_fit.py       the fitted leaves: skeleton.json, apex_<X>.jpg, fit.json,
#                                leaf_support.txt (apex_fit_check.py)
#   fit_heldout/                 with APEX_HELDOUT=1: fitted on even frames, scored on odd
#   labels/    apex_labels.py    scores against hand-drawn leaf lines, where there are any
#   bundle/    make_bundle.sh    <plant>_view_<stamp>.tar.gz for Blender, and how to open it
#
# Environment: PY (python), TAG_MM (default 8), APEX_HELDOUT (0/1), HF_HOME and
# HF_TOKEN for SAM3. Each step logs to <step>/run.log.
set -euo pipefail

WD="${1:?usage: run_apex.sh <workdir>}"
WD="$(cd "$WD" && pwd)"
PY="${PY:-python}"
TAG_MM="${TAG_MM:-8}"
APEX_HELDOUT="${APEX_HELDOUT:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
A="$WD/apex"
mkdir -p "$A"
step() { echo "--- $(date '+%H:%M:%S') $1"; }

for need in p5x/skeleton.json p5x_p4g/skeleton.json s2d/skeleton.json tag_scale.json; do
    [[ -f "$WD/$need" ]] || { echo "run_apex: no $WD/$need -- run the phases before apex first" >&2; exit 1; }
done

step "regions: where the apices are"
mkdir -p "$A/regions"
"$PY" "$REPO/scripts/apex_regions.py" --workdir "$WD" --out "$A/regions" 2>&1 | tee "$A/regions/run.log"
N=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['apices']))" "$A/regions/apex_regions.json")
if [[ "$N" -eq 0 ]]; then
    echo "  no apex on this plant: nothing to fit"
    exit 0
fi
R="$A/regions/apex_regions.npz"

step "zoom: SAM3 on each apex's square, every frame ($N apices)"
mkdir -p "$A/zoom"
"$PY" "$REPO/scripts/apex_zoom.py" --workdir "$WD" --out "$A/zoom" --regions "$R" 2>&1 | tee "$A/zoom/run.log"

step "s2d_zoom: 2D leaf lines on the zoomed masks"
mkdir -p "$A/s2d_zoom"
"$PY" "$REPO/scripts/skeleton_2d.py" --workdir "$WD" --masks "$A/zoom/masks_zoom.npz" --tag-mm "$TAG_MM" \
    --leaf-curves --stem-tree "$WD/p5x/skeleton.json" --out "$A/s2d_zoom" 2>&1 | tee "$A/s2d_zoom/run.log"

CANDS=("$WD/p5x" "$WD/p5x_p4g" "$WD/s2d" "$A/s2d_zoom")
step "select: starting leaves from the candidates"
mkdir -p "$A/select"
"$PY" "$REPO/scripts/apex_select.py" --workdir "$WD" --regions "$R" --zoom "$A/zoom" \
    --evidence "$A/s2d_zoom/evidence_2d.json" --candidates "${CANDS[@]}" --base "$WD/p5x" \
    --out "$A/select" 2>&1 | tee "$A/select/run.log"

fit() {   # fit <out> <train>
    mkdir -p "$1"
    "$PY" "$REPO/scripts/apex_fit.py" --workdir "$WD" --init "$A/select" --regions "$R" --zoom "$A/zoom" \
        --evidence "$A/s2d_zoom/evidence_2d.json" --candidates "${CANDS[@]}" --train "$2" --out "$1" \
        > "$1/run.log" 2>&1
}
step "fit: every apex leaf shaped to the photos$([[ "$APEX_HELDOUT" == 1 ]] && echo ', and a held-out run beside it')"
held=""
if [[ "$APEX_HELDOUT" == 1 ]]; then
    fit "$A/fit_heldout" even & held=$!
fi
fit "$A/fit" all || { tail -20 "$A/fit/run.log"; exit 1; }
grep -E "^  apex_|fit views|axis turned|wrote" "$A/fit/run.log" | sed 's/; context:.*//' || true
if [[ -n "$held" ]]; then
    wait "$held" || echo "  held-out run failed: $A/fit_heldout/run.log"
    grep -E "HELD-OUT" "$A/fit_heldout/run.log" | sed 's/ (F1.*leaves/ ... leaves/' || true
fi

step "check: counts, shape, instance scores, per-leaf support"
"$PY" "$REPO/scripts/apex_fit_check.py" --workdir "$WD" --skeleton "$A/fit" --regions "$R" --zoom "$A/zoom" \
    > "$A/fit/check.full.log" 2>&1 || { tail -20 "$A/fit/check.full.log"; exit 1; }
grep -v "^  fit_apex" "$A/fit/check.full.log" | tee "$A/fit/check.log"

SKEL=("$WD/p5x" "$A/select" "$A/fit")
if compgen -G "$WD/p1/frames/frame_*.json" > /dev/null; then
    step "labels: scored against the hand-drawn leaf lines"
    mkdir -p "$A/labels"
    "$PY" "$REPO/scripts/apex_labels.py" --workdir "$WD" --out "$A/labels" --score "${SKEL[@]}" \
        --annotate "$A/labels/annotated" 2>&1 | tail -8 | tee "$A/labels/run.log"
    SKEL=("$A/labels/annotated/p5x" "$A/labels/annotated/select" "$A/labels/annotated/fit")
fi

step "bundle: for Blender"
"$REPO/scripts/make_bundle.sh" "$WD" "${SKEL[@]}"
