#!/usr/bin/env bash
# A dated, self-identifying Blender bundle of one plant's apex result, to view on another machine.
#
#   ./scripts/make_bundle.sh <workdir> <skeleton dir> [<skeleton dir> ...]
#
# Writes <workdir>/apex/bundle/<specimen>_view_<stamp>.tar.gz (older bundles there
# are removed: superseded) holding the P5x scene on the P4g cloud and on the P4b
# surface, every given skeleton.json, the apex pictures and numbers, and the
# viewer. Unpacked anywhere it is a mini-repo; VERSION.txt says how to open it
# and lists the md5 and build stamp of every skeleton the viewer loads.
set -euo pipefail
WD="$(cd "${1:?usage: make_bundle.sh <workdir> <skeleton dir>...}" && pwd)"
shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPECIMEN="$(basename "$(dirname "$WD")")"
STAMP=$(date +%Y%m%d_%H%M)
OUT="$WD/apex/bundle"
NAME="${SPECIMEN}_view_$STAMP"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
REL="$SPECIMEN/$(basename "$WD")"
R="$STAGE/$REL"

copy() {    # copy <path under the workdir>, if it is there
    [[ -e "$WD/$1" ]] || return 0
    mkdir -p "$R/$(dirname "$1")"
    cp "$WD/$1" "$R/$1"
}
for f in tag_scale.json p5/stem_graph.json p5x/skeleton.json p5x/segmented.ply \
         p5x_p4g/skeleton.json p5x_p4g/segmented.ply apex/regions/apex_regions.jpg \
         apex/regions/apex_regions.json apex/fit/fit.json apex/fit/leaf_support.txt apex/fit/check.log \
         apex/fit_heldout/fit.json apex/labels/scores.json p1/frames/apex_counts.json \
         apex/margins/margins.json apex/margins/margins.jpg apex/curvature/curvature.json apex/curvature/curvature.jpg \
         apex/flatlay/flatlay_match.json apex/flatlay/flatlay_match.jpg apex/templates/template_fit.json \
         apex/templates/template_fit.jpg; do
    copy "$f"
done
for f in "$WD"/apex/fit/apex_*.jpg "$WD"/apex/labels/score_*.jpg; do
    [[ -e "$f" ]] && copy "${f#$WD/}"
done
SK=()
for d in "$@"; do
    d="$(cd "$d" && pwd)"
    rel="${d#$WD/}"
    copy "$rel/skeleton.json"
    SK+=("$REL/$rel")
done
mkdir -p "$STAGE/scripts"
cp "$REPO/scripts/blender_view_plant.py" "$REPO/scripts/view_in_blender.sh" "$STAGE/scripts/"
LIST=$(IFS=,; echo "${SK[*]}")
BASE=p5x_p4g; [[ -f "$R/p5x_p4g/segmented.ply" ]] || BASE=p5x

{
    echo "$SPECIMEN view bundle $STAMP  (commit $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD -- scripts src 2>/dev/null || echo ' + local edits'))"
    echo
    echo "Open (from the folder you unpacked into):"
    echo "  ./scripts/view_in_blender.sh $REL --p5x $BASE --skeleton2d \\"
    echo "      $LIST"
    echo
    echo "Base scene: P5x's leaf labels on the $( [[ $BASE == p5x_p4g ]] && echo 'P4g cloud' || echo 'P4b surface') (collection $BASE);"
    echo "--p5x p5x gives the P4b-based one. Collections: one per method (s2d_<folder>), closed in the"
    echo "outliner: _blades = fitted leaf surfaces (fit), _midribs, _branches (each apex's fitted shoot),"
    echo "_gt_found = leaves matching a hand-drawn line (only where there are labels). Buds: ..._bud."
    echo "_origin = where the plant's leaves start (a rosette) or the foot of its stem."
    echo
    echo "Pictures and numbers: apex/regions/apex_regions.jpg (where the apices are), apex/fit/apex_<X>.jpg"
    echo "(photo | SAM3 masks | start | fitted, grey = other leaves), apex/fit/check.log (counts, shape, scores),"
    echo "apex/margins/margins.json + .jpg (per leaf: width profile, outline and edge-height waves, in mm)."
    echo
    echo "md5 and build stamp of every skeleton the viewer loads:"
    for s in "${SK[@]}"; do
        f="$s/skeleton.json"
        stamp=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); b=d.get('built') or {}; a=d.get('annotated') or {}; print('built', b.get('at','-'), b.get('by','-'), '| annotated', a.get('at','-'))" "$STAGE/$f")
        echo "  $(md5sum "$STAGE/$f" | cut -c1-32)  $f  ($stamp)"
    done
} > "$STAGE/VERSION.txt"

mkdir -p "$OUT"
rm -f "$OUT/${SPECIMEN}"_view_*.tar.gz
tar czf "$OUT/$NAME.tar.gz" -C "$STAGE" .
echo "  bundle: $OUT/$NAME.tar.gz ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"
sed -n '3,5p' "$STAGE/VERSION.txt"
