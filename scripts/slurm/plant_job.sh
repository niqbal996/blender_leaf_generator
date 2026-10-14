#!/usr/bin/env bash
# One plant, start to finish, inside a Slurm job's container. scripts/slurm/submit_plants.sh
# queues it; by hand:
#
#   bash scripts/slurm/plant_job.sh <plant dataset dir> [run_pipeline.sh options ...]
#
# The job's image (nvcr.io_nvidia_pytorch_26.02-py3_blender.sqsh) has the pose env
# installed in /root/miniconda3, so it is only activated here -- no setup step.
# gsplat compiles its CUDA kernels for the GPU this job landed on (A6000 8.6, A100 8.0,
# H100 9.0), ~5 min, into the container's /tmp.
#
# Environment: POSE_ENV (default pose), CONDA_ROOT (default /root/miniconda3).
set -eo pipefail

DATASET="${1:?usage: plant_job.sh <plant dataset dir> [run_pipeline.sh options]}"
shift
DATASET="${DATASET%/}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${POSE_ENV:-pose}"
CONDA="${CONDA_ROOT:-/root/miniconda3}"

echo "=== plant job: $DATASET on $(hostname), Slurm job ${SLURM_JOB_ID:-none}, $(date '+%F %T')"
[[ -d "$DATASET" ]] || { echo "no such dataset directory: $DATASET" >&2; exit 1; }
[[ -f "$CONDA/etc/profile.d/conda.sh" ]] || {
    echo "no conda at $CONDA in this container (CONDA_ROOT=...?)" >&2; exit 1; }
# shellcheck disable=SC1091
source "$CONDA/etc/profile.d/conda.sh"
# the image (nvcr.io_nvidia_pytorch_26.02-py3_blender.sqsh) ships the pose env: no setup step
conda activate "$ENV_NAME" || {
    echo "no conda env '$ENV_NAME' in this image -- is it the _blender.sqsh image?" >&2; exit 1; }
set -u

CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
export TORCH_CUDA_ARCH_LIST="$CAP"                       # the env was pinned to the GPU it was built on
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions_sm$CAP}"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1), compute $CAP"
python - <<'EOF'
import torch, pycolmap
assert torch.cuda.is_available(), "torch sees no GPU -- is the node's driver too old for this torch build?"
p = torch.cuda.get_device_properties(0)
print(f"  torch {torch.__version__} on {p.name} ({p.total_memory / 2**30:.0f} GiB), pycolmap cuda {pycolmap.has_cuda}")
EOF

WORKDIR="$DATASET/plant"
mkdir -p "$WORKDIR"
LOCK="$WORKDIR/.slurm_job"           # submit_plants.sh will not queue a plant twice while this is here
echo "${SLURM_JOB_ID:-none} $(hostname) $(date '+%F %T')" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

export POSE_PYTHON="$(command -v python)" NO_COLOR=1 PYTHONUNBUFFERED=1
cd "$REPO"                           # ./pipeline.conf (the HF token) is read from here
./run_pipeline.sh "$DATASET" "$@"
echo "=== plant job done: $DATASET $(date '+%F %T')"
