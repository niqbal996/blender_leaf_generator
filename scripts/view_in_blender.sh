#!/usr/bin/env bash
# Open a P5 result in Windows Blender, from WSL.
#
#   ./scripts/view_in_blender.sh runs/plant_9
#   ./scripts/view_in_blender.sh runs/plant_9 --background     # build only, no window
#
# Blender inside WSL is the awkward option here: Ubuntu 20.04 ships a
# libwayland-client too old for recent Blender builds, and Mesa 21.2's d3d12
# driver predates the OpenGL 4.3 that Blender needs, so the GUI falls back and
# refuses to start. Windows Blender has a real GPU and reads the WSL checkout
# over \\wsl.localhost, so it sidesteps both.
#
# Paths are translated here rather than by hand: Windows Blender cannot resolve
# /home/... and gets the UNC form instead.
set -euo pipefail

WORKDIR="${1:-}"
shift || true
[[ -n "$WORKDIR" ]] || { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }
[[ -d "$WORKDIR" ]] || { echo "no such directory: $WORKDIR" >&2; exit 1; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABS_WORKDIR="$(cd "$WORKDIR" && pwd)"

# Newest Blender under Program Files wins, so this keeps working after an
# upgrade without editing the script.
BLENDER="$(ls -d "/mnt/c/Program Files/Blender Foundation/Blender "*/blender.exe 2>/dev/null | sort -V | tail -1 || true)"
[[ -n "$BLENDER" ]] || {
    echo "No Windows Blender found under /mnt/c/Program Files/Blender Foundation/." >&2
    echo "Set BLENDER_EXE to its blender.exe, or install Blender on Windows." >&2
    [[ -n "${BLENDER_EXE:-}" ]] || exit 1; }
BLENDER="${BLENDER_EXE:-$BLENDER}"

to_unc() { printf '\\\\wsl.localhost\\%s%s' "$WSL_DISTRO_NAME" "$(echo "$1" | tr '/' '\\')"; }

echo "blender : $BLENDER"
echo "workdir : $(to_unc "$ABS_WORKDIR")"
exec "$BLENDER" "$@" \
    --python "$(to_unc "$REPO/scripts/blender_view_plant.py")" \
    -- --workdir "$(to_unc "$ABS_WORKDIR")"
