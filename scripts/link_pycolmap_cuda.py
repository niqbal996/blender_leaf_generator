"""Make pycolmap-cuda importable without touching LD_LIBRARY_PATH.

The pycolmap-cuda wheel's extension links libcudart.so.12 and libcurand.so.10
but bundles neither, so `import pycolmap` fails with "libcudart.so.12: cannot
open shared object file" unless a CUDA 12 runtime happens to be on the loader
path. The nvidia-cuda-runtime-cu12 / nvidia-curand-cu12 wheels supply the
files, under directories nothing searches.

The extension's RPATH is $ORIGIN/../pycolmap_cuda.libs, and an RPATH is
searched before LD_LIBRARY_PATH -- so symlinking the two libraries into that
directory fixes the import for every way the interpreter gets started
(activated env, `conda run`, POSE_PYTHON=..., a cron job), which an
activation-time environment variable does not.

    python scripts/link_pycolmap_cuda.py

Safe to re-run; a no-op when the CPU pycolmap is installed. Re-run it after
any `pip install` that reinstalls pycolmap-cuda.
"""

import sys
import sysconfig
from pathlib import Path

# soname -> where its nvidia-*-cu12 wheel puts it, relative to site-packages
NEEDED = {
    "libcudart.so.12": "nvidia/cuda_runtime/lib",
    "libcurand.so.10": "nvidia/curand/lib",
}
PACKAGE = {
    "libcudart.so.12": "nvidia-cuda-runtime-cu12",
    "libcurand.so.10": "nvidia-curand-cu12",
}


def main() -> int:
    site = Path(sysconfig.get_paths()["platlib"])
    libs = site / "pycolmap_cuda.libs"
    if not libs.is_dir():
        print(f"    no {libs.name} in {site} -- pycolmap-cuda not installed, nothing to link")
        return 0

    missing = []
    for soname, subdir in NEEDED.items():
        source = site / subdir / soname
        if not source.exists():
            missing.append(soname)
            print(f"    MISSING {source}  (pip install {PACKAGE[soname]})")
            continue
        link = libs / soname
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source)
        print(f"    linked {soname} -> {source}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
