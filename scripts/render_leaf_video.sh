#!/usr/bin/env bash
# Render a leaf flythrough to mp4 without tying up your Blender window.
#
#   ./scripts/render_leaf_video.sh /mnt/d/PBR_Scans/2026-09-16-Naeem/gaensefuss_1/maps /mnt/d/clips/gaensefuss
#   ./scripts/render_leaf_video.sh <maps> <out> --mode orbit --focus leaf_17
#   ./scripts/render_leaf_video.sh <maps> <out> --frames-per-leaf 24 --resolution 1920x1080
#
# A few hundred frames takes minutes. Running that from Blender's Text Editor
# blocks the UI for the whole render with no progress bar and no way to
# cancel, so it looks like a hang -- and because FFMPEG opens the output file
# at the start, a plausible-looking mp4 appears immediately and stays
# truncated until the last frame lands. A background Blender avoids all of
# that: progress prints per frame and Ctrl+C stops it.
set -euo pipefail

# view_in_blender.sh guards its own `main`, so sourcing it just lends us
# to_windows_path -- the WSL-to-Windows path rules are fiddly enough (drive
# letters that don't match mount points, UNC for WSL-owned files) to be worth
# having in exactly one place.
source "$(dirname "${BASH_SOURCE[0]}")/view_in_blender.sh"
declare -F to_windows_path >/dev/null || {
    echo "to_windows_path missing -- scripts/view_in_blender.sh changed shape?" >&2
    exit 1; }

main_render() {
    local MAPS="${1:-}" OUT="${2:-}"
    shift 2 2>/dev/null || true
    [[ -n "$MAPS" && -n "$OUT" ]] || { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }
    [[ -d "$MAPS" ]] || { echo "no such maps directory: $MAPS" >&2; exit 1; }

    local REPO
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    # Newest Blender under Program Files wins, so this keeps working after an
    # upgrade without editing the script.
    local BLENDER="${BLENDER_EXE:-}"
    if [[ -z "$BLENDER" ]]; then
        BLENDER="$(ls -d "/mnt/c/Program Files/Blender Foundation/Blender "*/blender.exe 2>/dev/null | sort -V | tail -1 || true)"
    fi
    [[ -n "$BLENDER" ]] || {
        echo "No Windows Blender found under /mnt/c/Program Files/Blender Foundation/." >&2
        echo "Set BLENDER_EXE to its blender.exe." >&2; exit 1; }

    mkdir -p "$(dirname "$OUT")"

    local WIN_REPO WIN_MAPS WIN_OUT WIN_SCRIPT
    WIN_REPO="$(to_windows_path "$REPO")"
    WIN_MAPS="$(to_windows_path "$(cd "$MAPS" && pwd)")"
    WIN_OUT="$(to_windows_path "$(cd "$(dirname "$OUT")" && pwd)")\\$(basename "$OUT")"
    WIN_SCRIPT="$(to_windows_path "$REPO/scripts/render_leaf_video.py")"

    echo "blender : $BLENDER"
    echo "maps    : $WIN_MAPS"
    echo "output  : $WIN_OUT"
    exec "$BLENDER" -b --python "$WIN_SCRIPT" -- \
        --repo "$WIN_REPO" --maps "$WIN_MAPS" --out "$WIN_OUT" "$@"
}

main_render "$@"
