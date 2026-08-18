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

# Match the Windows path rules Blender accepts reliably:
# - /mnt/d/... -> D:\...
# - /home/... -> \\wsl.localhost\<distro>\...
# The UNC path is right for WSL-owned files but a mounted Windows drive can be
# denied when Blender reads it via the WSL localhost share.
to_windows_path() {
    local path="${1:-}"
    [[ -n "$path" ]] || return 1

    if [[ "$path" == /mnt/* ]]; then
        local drive="${path#/mnt/}"
        drive="${drive%%/*}"
        if [[ "$drive" =~ ^[A-Za-z]$ ]]; then
            printf '%s:%s' "${drive^^}" "$(printf '%s' "${path#/mnt/$drive}" | sed 's#/#\\#g')"
            return 0
        fi
    fi

    printf '\\\\wsl.localhost\\%s%s' "$WSL_DISTRO_NAME" "$(printf '%s' "$path" | tr '/' '\\')"
}

main() {
    local WORKDIR="${1:-}"
    shift || true
    [[ -n "$WORKDIR" ]] || { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }
    [[ -d "$WORKDIR" ]] || { echo "no such directory: $WORKDIR" >&2; exit 1; }

    local REPO
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    local ABS_WORKDIR
    ABS_WORKDIR="$(cd "$WORKDIR" && pwd)"

    # Newest Blender under Program Files wins, so this keeps working after an
    # upgrade without editing the script.
    local BLENDER
    BLENDER="$(ls -d "/mnt/c/Program Files/Blender Foundation/Blender "*/blender.exe 2>/dev/null | sort -V | tail -1 || true)"
    [[ -n "$BLENDER" ]] || {
        echo "No Windows Blender found under /mnt/c/Program Files/Blender Foundation/." >&2
        echo "Set BLENDER_EXE to its blender.exe, or install Blender on Windows." >&2
        [[ -n "${BLENDER_EXE:-}" ]] || exit 1; }
    BLENDER="${BLENDER_EXE:-$BLENDER}"

    local WIN_WORKDIR WIN_REPO
    WIN_WORKDIR="$(to_windows_path "$ABS_WORKDIR")"
    WIN_REPO="$(to_windows_path "$REPO/scripts/blender_view_plant.py")"

    echo "blender : $BLENDER"
    echo "workdir : $WIN_WORKDIR"
    exec "$BLENDER" "$@" \
        --python "$WIN_REPO" \
        -- --workdir "$WIN_WORKDIR"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
