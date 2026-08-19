#!/usr/bin/env bash
# One-shot environment setup for this repo's plant pose pipeline (P1-P6)
# plus the Blender leaf-assembly package.
#
#   ./setup_env.sh                 # create/populate the "pose" conda env
#   ./setup_env.sh --name myenv    # different env name
#   ./setup_env.sh --cuda cu121    # override the detected CUDA wheel tag
#   ./setup_env.sh --cpu           # no GPU: CPU torch, CPU pycolmap, no gsplat
#   ./setup_env.sh --check         # detect and report only, install nothing
#   ./setup_env.sh --checkpoint-only  # just fetch the SAM2 weights
#
# Safe to re-run: every step is idempotent, so it doubles as a repair tool.
#
# Three things here are NOT optional extras of a normal pip install, which
# is why this script exists rather than a line in the README:
#   * torch must match the machine's CUDA and must be installed BEFORE
#     gsplat, which JIT-compiles CUDA kernels against whatever torch it finds
#   * SAM2 is installed from facebookresearch's git, because the `sam2` name
#     on PyPI is an unrelated third-party upload
#   * pycolmap-cuda and gsplat both need environment variables set on the
#     env itself (CUDA runtime path, host compiler, GPU arch) or they fail
#     at import/first-use with errors that name neither

set -euo pipefail

ENV_NAME="pose"
CUDA_TAG=""
MODE="gpu"
CHECK_ONLY=0
CKPT_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)  ENV_NAME="$2"; shift 2 ;;
        --cuda)  CUDA_TAG="$2"; shift 2 ;;
        --cpu)   MODE="cpu"; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        --checkpoint-only) CKPT_ONLY=1; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m    WARNING: %s\033[0m\n' "$*"; }
die() { printf '\033[31m    ERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- detect
say "Detecting hardware and toolchain"

GPU_NAME=""; ARCH=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
fi
if [[ -z "$GPU_NAME" && "$MODE" == "gpu" ]]; then
    warn "no NVIDIA GPU detected -- falling back to --cpu"
    MODE="cpu"
fi
[[ -n "$GPU_NAME" ]] && echo "    GPU:              $GPU_NAME (compute capability ${ARCH:-unknown})"

# The torch wheel index is chosen from the CUDA *toolkit* version, since
# that is what gsplat's nvcc will use to build against torch later. Mixing
# them is the classic way to get kernels that compile but never run.
#
# The version comes from the *driver*, not from whatever nvcc happens to be
# on PATH. Reading the system nvcc was wrong twice over, and cost a full
# pipeline run to find out: this machine's apt `nvidia-cuda-toolkit` is 11.5
# while its driver supports 13.x, so the old rule pinned torch to cu118 for
# no reason -- and gsplat 1.5.3 compiles with `-std=c++20`, which nvcc only
# accepts from 12.0 on, so an 11.x toolkit cannot build it at all
# ("nvcc fatal: Value 'c++20' is not defined for option 'std'").
#
# A matching nvcc is then installed *into the env* further down, so the build
# never depends on a system toolkit again.
if [[ "$MODE" == "gpu" && -z "$CUDA_TAG" ]]; then
    # nvidia-smi reports the highest CUDA the driver can run.
    DRIVER_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA[^0-9]*\([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)"
    case "$DRIVER_CUDA" in
        13*)          CUDA_TAG="cu130" ;;
        12[89]|121[0-9]) CUDA_TAG="cu128" ;;
        12[0-7])      CUDA_TAG="cu126" ;;
        11*)          CUDA_TAG="cu118" ;;
        *)            CUDA_TAG="cu128" ;;
    esac
    if [[ -n "$DRIVER_CUDA" ]]; then
        echo "    driver supports:  CUDA ${DRIVER_CUDA:0:2}.${DRIVER_CUDA:2} -> torch wheel $CUDA_TAG"
    else
        warn "could not read the driver's CUDA version; defaulting to $CUDA_TAG (override with --cuda)"
    fi
fi
# The matching toolkit version for the conda nvcc installed below.
case "$CUDA_TAG" in
    cu130) NVCC_VERSION="13.0" ;;
    cu128) NVCC_VERSION="12.8" ;;
    cu126) NVCC_VERSION="12.6" ;;
    cu121) NVCC_VERSION="12.1" ;;
    cu118) NVCC_VERSION="11.8" ;;
    *)     NVCC_VERSION="" ;;
esac
if [[ "$MODE" == "gpu" && "$CUDA_TAG" == "cu118" ]]; then
    warn "cu118 cannot build gsplat (it needs -std=c++20, i.e. nvcc >= 12.0). P4b will not run;"
    warn "everything else will. Pass --cuda cu128 if the driver actually supports it."
fi

# gsplat compiles CUDA through a host compiler nvcc will accept; the system
# default is often too new, which surfaces as a bare "unsupported compiler".
# Preference is driven by the CUDA major version rather than "newest wins":
# each CUDA release caps the host GCC it accepts, and overshooting fails at
# build time with a bare "unsupported compiler". CUDA 11 is happiest on
# gcc-10, CUDA 12 on gcc-12, CUDA 13 on gcc-13.
#
# Setting these is not optional on a machine with no unversioned `c++`: torch
# defaults its host compiler to `c++`, and Ubuntu ships only `g++-11`-style
# names unless build-essential created the symlink. Without CC/CXX the build
# fails on a missing compiler after warning about the wrong one.
case "${CUDA_TAG:-}" in
    cu13*) GCC_PREF="13 12 11 14 10" ;;
    cu12*) GCC_PREF="12 11 10 13 9" ;;
    *)     GCC_PREF="10 11 9 12" ;;
esac
HOST_CC=""; HOST_CXX=""
for v in $GCC_PREF; do
    if [[ -x "/usr/bin/gcc-$v" && -x "/usr/bin/g++-$v" ]]; then
        HOST_CC="/usr/bin/gcc-$v"; HOST_CXX="/usr/bin/g++-$v"; break
    fi
done
if [[ "$MODE" == "gpu" ]]; then
    [[ -n "$HOST_CC" ]] && echo "    host compiler:    $HOST_CC" \
        || warn "no versioned gcc found; gsplat may fail to JIT-compile. apt install gcc-10 g++-10"
fi

command -v conda >/dev/null 2>&1 || die "conda not found. Install miniconda first."
CONDA_BASE="$(conda info --base)"
echo "    conda:            $CONDA_BASE"
echo "    mode:             $MODE"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    say "--check given; nothing installed."
    exit 0
fi

# ---------------------------------------------------------------- checkpoint
# SAM2's weights are a ~900MB file that is not on PyPI and not in this repo,
# so a fresh clone has nothing for P2 to load. Fetched here into a location
# run_pipeline.sh already searches.
fetch_checkpoint() {
    local name="sam2.1_hiera_large.pt"
    local dest="$REPO_ROOT/checkpoints/$name"
    local url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/$name"
    if [[ -f "$dest" ]]; then
        echo "    already present: $dest ($(du -h "$dest" | cut -f1))"
        return 0
    fi
    mkdir -p "$REPO_ROOT/checkpoints"
    echo "    downloading $name (~857 MiB) -> $dest"
    # -C - resumes a partial file; downloading to .part first means an
    # interrupted run never leaves a truncated checkpoint that loads and
    # then fails deep inside SAM2 with a confusing error.
    if command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar -C - -o "$dest.part" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -c -O "$dest.part" "$url"
    else
        die "neither curl nor wget available to download $url"
    fi
    mv "$dest.part" "$dest"
    echo "    done: $dest"
}

if [[ "$CKPT_ONLY" -eq 1 ]]; then
    say "Fetching SAM2 checkpoint only"
    fetch_checkpoint
    exit 0
fi

# ---------------------------------------------------------------- env
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    say "Reusing existing conda env '$ENV_NAME'"
else
    say "Creating conda env '$ENV_NAME' (python 3.10)"
    conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

# conda activate can leave an earlier env's python first on PATH, so every
# step below uses this absolute interpreter rather than bare `python`.
PY="$CONDA_PREFIX/bin/python"
[[ -x "$PY" ]] || die "no python at $PY"
echo "    interpreter:      $PY"
"$PY" -m pip install --upgrade pip -q

# ---------------------------------------------------------------- nvcc
# gsplat ships no prebuilt wheels for any torch/CUDA combination, so it JIT-
# compiles its CUDA kernels the first time P4b renders. That needs an nvcc
# whose version matches the torch build -- and relying on the system's is how
# this broke: apt's was 11.5 against a cu130 torch. Installing it into the
# env pins the pair together and survives anything done to the system.
if [[ "$MODE" == "gpu" && -n "$NVCC_VERSION" ]]; then
    say "Installing nvcc $NVCC_VERSION into the env (gsplat JIT-compiles against it)"
    if "$CONDA_PREFIX/bin/nvcc" --version 2>/dev/null | grep -q "release $NVCC_VERSION"; then
        echo "    already present: $("$CONDA_PREFIX/bin/nvcc" --version | sed -n 's/.*release \(.*\), .*/\1/p')"
    else
        # cuda-nvcc alone lacks the headers its own generated code includes;
        # cuda-cudart-dev supplies them. The pair is ~200MB against ~3GB for
        # the full cuda-toolkit, and nothing here needs the rest of it.
        conda install -n "$ENV_NAME" -c nvidia -y \
            "cuda-nvcc=$NVCC_VERSION" "cuda-cudart-dev=$NVCC_VERSION" \
            || warn "could not install nvcc $NVCC_VERSION -- P4b will fail to build gsplat"
    fi

    # cuda-cudart-dev puts its headers under targets/<arch>/include. nvcc finds
    # them by itself, but gsplat's *host* .cpp files are compiled by plain g++
    # with only `-isystem $CUDA_HOME/include`, so without these links the build
    # dies on "fatal error: cuda_runtime.h: No such file or directory".
    #
    # Worse than dying, on a machine that also has apt's nvidia-cuda-toolkit:
    # g++ then finds /usr/include/cuda_runtime.h instead and the extension
    # builds *successfully* against one CUDA version's headers while nvcc used
    # another's. Observed here -- host files took CUDA 11.5 headers while nvcc
    # was 13.0. Linking them makes the version explicit and identical for both.
    CUDA_TARGET_INCLUDE="$CONDA_PREFIX/targets/x86_64-linux/include"
    if [[ -d "$CUDA_TARGET_INCLUDE" ]]; then
        mkdir -p "$CONDA_PREFIX/include"
        linked=0
        for header in "$CUDA_TARGET_INCLUDE"/*; do
            target="$CONDA_PREFIX/include/$(basename "$header")"
            [[ -e "$target" ]] || { ln -s "$header" "$target" && linked=$((linked + 1)); }
        done
        echo "    linked $linked CUDA header(s) into \$CONDA_PREFIX/include"
    fi
fi

# ---------------------------------------------------------------- repo
# Before torch, deliberately. The extras list a bare `torch`, so pip resolves
# it against PyPI and happily replaces a +cuXXX wheel from the pytorch index
# with the default build -- which is exactly what happened here: a cu118 pin
# came out the far side as 2.13.0+cu130. Installing the repo first and torch
# last means the pinned wheel is the one that survives.
if [[ "$MODE" == "gpu" ]]; then
    say "Installing this repo with the full pipeline extras"
    "$PY" -m pip install -e ".[dev,pose-all]"
else
    say "Installing this repo (CPU: no gsplat/splat, CPU pycolmap)"
    "$PY" -m pip install -e ".[dev,skeleton,segment,semantic]"
fi

# ---------------------------------------------------------------- torch
if [[ "$MODE" == "gpu" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_TAG"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi
say "Settling torch on $CUDA_TAG (last, so the extras cannot replace it)"
WANT_CUDA="cpu"
if [[ "$MODE" == "gpu" ]]; then
    WANT_CUDA="${CUDA_TAG#cu}"
    WANT_CUDA="${WANT_CUDA:0:2}.${WANT_CUDA:2}"   # cu130 -> 13.0
fi
HAVE_CUDA="$("$PY" -c 'import torch; print(torch.version.cuda or "cpu")' 2>/dev/null || echo none)"

if [[ "$HAVE_CUDA" == "$WANT_CUDA" ]]; then
    echo "    torch $("$PY" -c 'import torch; print(torch.__version__)') already targets CUDA $HAVE_CUDA -- left alone"
else
    echo "    torch targets CUDA $HAVE_CUDA but $CUDA_TAG was selected -- reinstalling from the pinned index"
    # With dependencies, deliberately: the wheel needs its own matching
    # nvidia-*-cuXX runtime packages, and a --no-deps reinstall would leave
    # whichever set the previous torch pulled in. That combination imports
    # only by luck -- typically it fails on a missing libcudart.
    "$PY" -m pip install --force-reinstall torch torchvision --index-url "$TORCH_INDEX"
fi

# Verified rather than assumed: a torch whose CUDA does not match the nvcc
# above compiles gsplat fine and then fails at kernel launch, which is a far
# worse failure than not building at all.
TORCH_CUDA="$("$PY" -c 'import torch; print(torch.version.cuda or "cpu")' 2>/dev/null || echo "?")"
echo "    torch $("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null) -- CUDA $TORCH_CUDA"
if [[ "$MODE" == "gpu" && -n "$NVCC_VERSION" && "$TORCH_CUDA" != "$NVCC_VERSION"* ]]; then
    warn "torch is built for CUDA $TORCH_CUDA but nvcc is $NVCC_VERSION -- P4b may build and then"
    warn "fail at run time. Re-run with --cuda matching one of them."
fi

# pycolmap and pycolmap-cuda both install the same `pycolmap` module, so
# having both leaves whichever unpacked last in charge -- and pip reports
# success either way. Worse, if the CUDA build wins but its runtime is not
# loadable, `import pycolmap` raises and the pipeline looks like COLMAP was
# never installed at all. So: verify it actually imports, and if it does
# not, fall back to the CPU build rather than leaving a broken env.
say "Checking pycolmap"
if "$PY" -m pip list 2>/dev/null | grep -q '^pycolmap '; then
    if "$PY" -m pip list 2>/dev/null | grep -q '^pycolmap-cuda '; then
        echo "    both pycolmap and pycolmap-cuda present -- they collide; keeping one"
        # Reinstall the survivor afterwards: the two share the `pycolmap`
        # package directory, so uninstalling either deletes files the other
        # still needs. Skipping this leaves a module that imports but has
        # lost attributes -- observed live, `pycolmap.__version__` gone.
        # In GPU mode the CUDA build is the one worth keeping (it is what
        # makes --use-gpu work); the CPU build only wins as a fallback.
        if [[ "$MODE" == "gpu" ]]; then KEEP="pycolmap-cuda"; DROP="pycolmap";
        else KEEP="pycolmap"; DROP="pycolmap-cuda"; fi
        "$PY" -m pip uninstall -y "$DROP" >/dev/null
        "$PY" -m pip install --force-reinstall --no-deps "$KEEP" >/dev/null
        echo "    kept $KEEP"
    fi
fi
if ! "$PY" -c "import pycolmap" >/dev/null 2>&1; then
    PYCOLMAP_ERR="$("$PY" -c "import pycolmap" 2>&1 | tail -1)"
    warn "pycolmap does not import: $PYCOLMAP_ERR"
    echo "    falling back to the CPU build (GPU SIFT off, everything else identical)"
    "$PY" -m pip uninstall -y pycolmap-cuda >/dev/null 2>&1 || true
    "$PY" -m pip install pycolmap
    "$PY" -c "import pycolmap" >/dev/null 2>&1 \
        && echo "    CPU pycolmap imports cleanly" \
        || warn "pycolmap still not importable -- P3 will fail"
fi

# opencv-contrib-python is now the only opencv this project asks for, but a
# pre-existing plain opencv-python in the env still shadows it -- same `cv2`
# module, last writer wins, and cv2.aruco disappears with it.
if "$PY" -m pip list 2>/dev/null | grep -q '^opencv-python '; then
    say "Removing conflicting opencv-python (contrib is a superset, and carries cv2.aruco)"
    "$PY" -m pip uninstall -y opencv-python
    "$PY" -m pip install --force-reinstall --no-deps opencv-contrib-python
fi

# ---------------------------------------------------------------- sam2
say "Installing SAM2 from facebookresearch (the PyPI 'sam2' is unrelated)"
if [[ ! -d third_party/sam2/.git ]]; then
    mkdir -p third_party
    git clone --depth 1 https://github.com/facebookresearch/sam2.git third_party/sam2
else
    echo "    third_party/sam2 already cloned"
fi
"$PY" -m pip install -e third_party/sam2

say "Fetching SAM2 checkpoint"
fetch_checkpoint

# ---------------------------------------------------------------- vars
say "Pinning environment variables onto '$ENV_NAME'"
VARS=()
if [[ "$MODE" == "gpu" ]]; then
    # pycolmap-cuda links against a CUDA runtime that pip supplies as a
    # package; without its lib/ on the loader path `import pycolmap` dies
    # with "libcudart.so.12: cannot open shared object file".
    CUDART_LIB="$("$PY" -c "import nvidia.cuda_runtime as m, os; print(os.path.join(m.__path__[0],'lib'))" 2>/dev/null || true)"
    [[ -n "$CUDART_LIB" && -d "$CUDART_LIB" ]] && VARS+=("LD_LIBRARY_PATH=$CUDART_LIB")
    [[ -n "$HOST_CC" ]] && VARS+=("CC=$HOST_CC" "CXX=$HOST_CXX")
    [[ -n "$ARCH" ]] && VARS+=("TORCH_CUDA_ARCH_LIST=$ARCH")
    # Points gsplat's build at the env's nvcc rather than whatever the system
    # has. Without it, torch falls back to `which nvcc` and picks up an apt
    # toolkit that may be years older than the torch it is building against.
    [[ -x "$CONDA_PREFIX/bin/nvcc" ]] && VARS+=("CUDA_HOME=$CONDA_PREFIX")
fi
if [[ ${#VARS[@]} -gt 0 ]]; then
    conda env config vars set -n "$ENV_NAME" "${VARS[@]}" >/dev/null
    printf '    %s\n' "${VARS[@]}"
    conda deactivate; conda activate "$ENV_NAME"
else
    echo "    (none needed)"
fi

# ---------------------------------------------------------------- verify
say "Verifying"
"$PY" - <<'PYCODE'
import importlib, sys
ok = True
def check(mod, why, required=True):
    global ok
    try:
        m = importlib.import_module(mod)
        print(f"    [ok]   {mod:16} {getattr(m,'__version__','')}")
        return m
    except Exception as e:
        mark = "FAIL" if required else "skip"
        if required: ok = False
        print(f"    [{mark}] {mod:16} {type(e).__name__}: {str(e)[:70]}  ({why})")
        return None

torch = check("torch", "P4b surface fitting, P2 SAM2")
check("torchvision", "SAM2")
check("transformers", "P4c DINOv2/DINOv3 features")
cv2 = check("cv2", "everywhere")
check("sam2", "P2 mask propagation")
check("pycolmap", "P3 camera solve")
check("gsplat", "P4b", required=False)
check("pytorch_msssim", "P4b", required=False)
check("pose_estimator", "this repo")
check("leaf_generator", "this repo")

if torch is not None:
    print(f"    cuda available:  {torch.cuda.is_available()}"
          f"{'  <-- GPU phases will not run' if not torch.cuda.is_available() else ''}")
if cv2 is not None:
    has = hasattr(cv2, "aruco")
    print(f"    cv2.aruco:       {has}{'' if has else '  <-- P1 ChArUco path will fail'}")
    if not has: ok = False

sys.exit(0 if ok else 1)
PYCODE
STATUS=$?

if [[ $STATUS -ne 0 ]]; then
    warn "Some required imports failed -- see above."
    exit 1
fi

CKPT="$REPO_ROOT/checkpoints/sam2.1_hiera_large.pt"
[[ -f "$CKPT" ]] && echo "    [ok]   sam2 weights    $(du -h "$CKPT" | cut -f1)" \
                 || warn "SAM2 checkpoint missing -- run ./setup_env.sh --checkpoint-only"

say "Done. Activate with:  conda activate $ENV_NAME"
cat <<'NOTE'

    One manual step remains, only if you use P4c (pose-classify / pose-semantic):
    DINOv3 is a gated Hugging Face repo. Accept its license on huggingface.co,
    then authenticate once:

        huggingface-cli login

    Prefer that over passing --hf-token on the command line, where the token
    lands in your shell history and in process listings.
NOTE
