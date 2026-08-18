#!/usr/bin/env bash
# Batch-run the plant pose pipeline across many specimen folders.
#
# Example:
#   ./batch_run_plants.sh /mnt/d/PBR_Scans/2026-08-18-Naeem --skip-p4b
#
# The script scans the dataset root for immediate child directories, ignores a
# folder named Calibration/, and runs the existing run_pipeline.sh on every plant
# folder that contains one or more DSC_*.MOV files.
#
# Each plant folder is treated as its own workdir, so outputs land next to the
# videos and you do not need to manually copy/paste the file list for each run.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./batch_run_plants.sh <dataset-root> [extra run_pipeline.sh args]

Example:
  ./batch_run_plants.sh /mnt/d/PBR_Scans/2026-08-18-Naeem --skip-p4b
  ./batch_run_plants.sh /mnt/d/PBR_Scans/2026-08-18-Naeem --backend sam --sam-checkpoint checkpoints/sam2.1_hiera_large.pt

This scans <dataset-root> for immediate subdirectories, skips Calibration/, and
runs the pipeline once per plant folder with all matching DSC_*.MOV videos.
EOF
    exit "${1:-1}"
}

if [[ $# -lt 1 ]]; then
    usage 1
fi

DATASET_ROOT="${1}"
shift || true

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "no such directory: $DATASET_ROOT" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$REPO_ROOT/run_pipeline.sh"

if [[ ! -f "$PIPELINE" ]]; then
    echo "run_pipeline.sh not found at $PIPELINE" >&2
    exit 1
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift || true
fi

# Keep any extra args for the pipeline, e.g. --skip-p4b or --backend sam.
PIPELINE_ARGS=("$@")

find_plants() {
    while IFS= read -r -d '' plant; do
        printf '%s\n' "$plant"
    done < <(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name 'Calibration' -print0 | sort -z)
}

plants=() ; while IFS= read -r plant; do plants+=("$plant"); done < <(find_plants)
if [[ ${#plants[@]} -eq 0 ]]; then
    echo "No plant folders found under $DATASET_ROOT" >&2
    echo "Ignoring: $DATASET_ROOT/Calibration" >&2
    exit 1
fi

for plant in "${plants[@]}"; do
    videos=()
    while IFS= read -r -d '' video; do
        videos+=("$video")
    done < <(find "$plant" -maxdepth 1 -type f \( -iname 'DSC_*.MOV' -o -iname 'DSC_*.mov' \) -print0 | sort -z)

    if [[ ${#videos[@]} -eq 0 ]]; then
        echo "Skipping $plant -- no DSC_*.MOV files found"
        continue
    fi

    echo
    echo "=== Plant: $plant ==="
    printf 'Videos (%d):\n' "${#videos[@]}"
    for v in "${videos[@]}"; do echo "  $v"; done

    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '\nDry run: '\n
        if command -v conda >/dev/null 2>&1; then
            printf '  conda run -n pose --no-capture-output bash %q' "$PIPELINE"
            printf ' --workdir %q --video' "$plant"
            for v in "${videos[@]}"; do printf ' %q' "$v"; done
            for arg in "${PIPELINE_ARGS[@]}"; do printf ' %q' "$arg"; done
            printf '\n'
        else
            printf '  bash %q --workdir %q --video' "$PIPELINE" "$plant"
            for v in "${videos[@]}"; do printf ' %q' "$v"; done
            for arg in "${PIPELINE_ARGS[@]}"; do printf ' %q' "$arg"; done
            printf '\n'
        fi
        continue
    fi

    if command -v conda >/dev/null 2>&1; then
        echo "Running pipeline in pose conda env..."
        conda run -n pose --no-capture-output bash "$PIPELINE" \
            --workdir "$plant" \
            --video "${videos[@]}" \
            "${PIPELINE_ARGS[@]}"
    else
        echo "Running pipeline without conda activation..."
        bash "$PIPELINE" \
            --workdir "$plant" \
            --video "${videos[@]}" \
            "${PIPELINE_ARGS[@]}"
    fi

    echo "=== Finished: $plant ==="
    echo

done

echo "Batch run complete. Scanned: $DATASET_ROOT"
