#!/usr/bin/env bash
# Batch-run the full plant capture -> skeleton -> Gaussian Splat pipeline
# (estimate_plant_skeleton.py -> train_gaussian_splat.py ->
# align_plant_skeleton.py) over every plant folder under a root directory,
# using the GPU throughout. Stops short of Blender import -- that part is
# manual (blender_plant_import.py, run inside Blender).
#
# Layout expected per plant: <ROOT>/<plant_name>/images/*.jpg
# (same convention as every script this repo's README documents), e.g.:
#   /mnt/e/Camera_rig_data/turn_table_datasets/plant_1/images/*.jpg
#   /mnt/e/Camera_rig_data/turn_table_datasets/plant_2/images/*.jpg
#
# Usage:
#   ./sweep_plants.sh /mnt/e/Camera_rig_data/turn_table_datasets
#   ./sweep_plants.sh /mnt/e/Camera_rig_data/turn_table_datasets --iterations 30000
#   ./sweep_plants.sh /mnt/e/Camera_rig_data/turn_table_datasets --force
#   ./sweep_plants.sh /mnt/e/Camera_rig_data/turn_table_datasets --num-gpus 2
#
# --force re-runs a plant even if it already has a splat_blender.ply from a
# previous sweep (default: skip plants that already look done, so the sweep
# is safe to re-run/resume after an interruption or a failure partway
# through the batch).
#
# --num-gpus N (default: 1) runs N plants' *entire* pipelines concurrently,
# one per physical GPU (via CUDA_VISIBLE_DEVICES), round-robin across the
# plant list. This is a per-plant parallelism, not multi-GPU training of a
# single plant -- these scenes (thousands-to-hundreds-of-thousands of
# Gaussians, ~30 views) are too small for splitting one splat's training
# across GPUs to pay for its own communication overhead, but running
# multiple *different* plants' pipelines simultaneously, one per GPU, uses
# the extra hardware for real. CUDA_VISIBLE_DEVICES=<i> makes both
# pycolmap's GPU SIFT extraction and torch/gsplat see that one GPU as
# "cuda:0" inside each worker's subprocesses, so no code changes to either
# tool are needed.
#
# Each plant's full log goes to <plant_dir>/sweep.log (useful since with
# --num-gpus > 1, multiple plants' output would otherwise interleave on one
# terminal).
#
# Requires the "plant_gen" conda env (skeleton-gpu + splat extras already
# installed, per this session's setup) and its CC/CXX/LD_LIBRARY_PATH/
# TORCH_CUDA_ARCH_LIST env vars (already persisted via
# `conda env config vars set`, so activating the env is enough).

set -uo pipefail

ROOT_DIR="${1:?Usage: $0 <root_dir_containing_plant_folders> [--iterations N] [--force] [--num-gpus N]}"
shift

ITERATIONS=30000
FORCE=0
NUM_GPUS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUS_DIR="$(mktemp -d)"
trap 'rm -rf "$STATUS_DIR"' EXIT

# Runs one plant's full pipeline, pinned to $1's GPU index. Meant to be
# called as a background job -- writes its outcome to $STATUS_DIR instead
# of returning it, since a backgrounded subshell can't update the parent
# shell's arrays directly.
run_plant() {
    local gpu_index="$1"
    local plant_dir="$2"
    local plant_name
    plant_name="$(basename "$plant_dir")"
    local images_dir="$plant_dir/images"
    local log_file="$plant_dir/sweep.log"

    export CUDA_VISIBLE_DEVICES="$gpu_index"

    {
        echo "=============================================================="
        echo "[$plant_name] starting on GPU $gpu_index: $(date)"
        echo "=============================================================="

        echo "[$plant_name] 1/3 estimate_plant_skeleton.py (COLMAP + skeleton, GPU SIFT)"
        if ! python3 "$REPO_ROOT/estimate_plant_skeleton.py" \
            --images "$images_dir" \
            --workdir "$plant_dir" \
            --mask-mode none \
            --use-gpu; then
            echo "[$plant_name] FAILED at estimate_plant_skeleton.py"
            echo "failed" > "$STATUS_DIR/$plant_name"
            return
        fi

        echo "[$plant_name] 2/3 train_gaussian_splat.py ($ITERATIONS iterations, GPU $gpu_index)"
        if ! python3 "$REPO_ROOT/train_gaussian_splat.py" \
            --workdir "$plant_dir" \
            --iterations "$ITERATIONS"; then
            echo "[$plant_name] FAILED at train_gaussian_splat.py"
            echo "failed" > "$STATUS_DIR/$plant_name"
            return
        fi

        echo "[$plant_name] 3/3 align_plant_skeleton.py (no --scale-ref-* -- output won't be "
        echo "    metric; re-run this one step later with --scale-ref-a/-b/-distance-m once you"
        echo "    have a real-world measurement, no need to redo reconstruction/training)"
        if ! python3 "$REPO_ROOT/align_plant_skeleton.py" \
            --workdir "$plant_dir"; then
            echo "[$plant_name] FAILED at align_plant_skeleton.py"
            echo "failed" > "$STATUS_DIR/$plant_name"
            return
        fi

        echo "[$plant_name] done: $(date)"
        echo "success" > "$STATUS_DIR/$plant_name"
    } >"$log_file" 2>&1
}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate plant_gen

PLANT_DIRS=()
for plant_dir in "$ROOT_DIR"/*/; do
    plant_dir="${plant_dir%/}"
    plant_name="$(basename "$plant_dir")"

    if [[ ! -d "$plant_dir/images" ]]; then
        echo "[$plant_name] no images/ subfolder -- skipping"
        continue
    fi
    if [[ "$FORCE" -eq 0 && -f "$plant_dir/splat_blender.ply" ]]; then
        echo "[$plant_name] splat_blender.ply already exists -- skipping (use --force to redo)"
        echo "skipped" > "$STATUS_DIR/$plant_name"
        continue
    fi
    PLANT_DIRS+=("$plant_dir")
done

echo "Queued ${#PLANT_DIRS[@]} plant(s) across $NUM_GPUS GPU(s). Per-plant logs: <plant_dir>/sweep.log"

# Round-robin plants across $NUM_GPUS worker slots; within a slot, plants
# run one after another (waiting on that slot's previous job) so at most
# $NUM_GPUS pipelines run at once.
declare -a SLOT_PID=()
for i in "${!PLANT_DIRS[@]}"; do
    slot=$(( i % NUM_GPUS ))
    if [[ -n "${SLOT_PID[$slot]:-}" ]]; then
        wait "${SLOT_PID[$slot]}"
    fi
    run_plant "$slot" "${PLANT_DIRS[$i]}" &
    SLOT_PID[$slot]=$!
done
wait

SUCCESS=()
FAILED=()
SKIPPED=()
for status_file in "$STATUS_DIR"/*; do
    [[ -e "$status_file" ]] || continue
    name="$(basename "$status_file")"
    case "$(cat "$status_file")" in
        success) SUCCESS+=("$name") ;;
        failed) FAILED+=("$name") ;;
        skipped) SKIPPED+=("$name") ;;
    esac
done

echo ""
echo "=============================================================="
echo "Sweep summary"
echo "=============================================================="
echo "Succeeded (${#SUCCESS[@]}): ${SUCCESS[*]:-none}"
echo "Skipped   (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
echo "Failed    (${#FAILED[@]}): ${FAILED[*]:-none}"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
    exit 1
fi
