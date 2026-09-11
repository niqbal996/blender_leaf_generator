"""Where the SAM model weights live, for the Python CLIs.

The bash mirror of this is scripts/sam_checkpoints.sh, which run_pipeline.sh
and setup_env.sh source; the two must stay in step. Duplicating the rule is
deliberate -- the CLIs are run directly as often as through the pipeline
(`pose-segment --workdir ...`), and a hardcoded `checkpoints/sam2.1_...pt`
default resolves against the *current directory*, so running one from
anywhere but the repo root looked for weights that were never there.

A 900 MB checkpoint does not fit in a repo checkout under a disk quota, so
the weights have to be able to live elsewhere. In priority order:

    SAM2_CHECKPOINT     the SAM2 weights: a file, OR a directory to look in
    SAM_CHECKPOINT_DIR  one directory holding every checkpoint this repo uses
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

SAM2_CHECKPOINT_NAME = "sam2.1_hiera_large.pt"

# src/pose_estimator/checkpoints.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]


def checkpoint_candidates(name: str = SAM2_CHECKPOINT_NAME) -> List[Path]:
    """Every path that could hold `name`, most specific first."""
    out: List[Path] = []

    env = os.environ.get("SAM2_CHECKPOINT", "").strip()
    if env and name == SAM2_CHECKPOINT_NAME:
        # Pointing this at the directory the weights were moved to is the
        # obvious reading, and used to fail: a directory is not a file, so
        # the candidate was skipped without comment. A path that does not
        # exist yet is read as a file when it ends in .pt and a directory
        # otherwise -- the extension is the only signal available, and
        # getting it right here is what makes the error message truthful.
        path = Path(env)
        looks_like_dir = path.is_dir() or path.suffix != ".pt"
        out.append(path / name if looks_like_dir else path)

    shared = os.environ.get("SAM_CHECKPOINT_DIR", "").strip()
    if shared:
        out.append(Path(shared) / name)

    out.append(_REPO_ROOT / "checkpoints" / name)
    out.append(_REPO_ROOT / "third_party" / "sam2" / "checkpoints" / name)
    out.append(Path.home() / ".cache" / "sam2" / name)
    return out


def find_checkpoint(name: str = SAM2_CHECKPOINT_NAME) -> Optional[Path]:
    """The first candidate that exists, or None."""
    for candidate in checkpoint_candidates(name):
        if candidate.is_file():
            return candidate
    return None


def default_sam2_checkpoint() -> Path:
    """An argparse default: the checkpoint if it exists, else the first place
    it should be. Returning a path either way keeps `--checkpoint` optional
    and leaves the complaining to `resolve_checkpoint`, which can say where
    it looked."""
    found = find_checkpoint()
    return found if found is not None else checkpoint_candidates()[0]


def resolve_checkpoint(explicit: Optional[Path] = None,
                       name: str = SAM2_CHECKPOINT_NAME) -> Path:
    """The checkpoint to load, or SystemExit naming every path tried.

    `explicit` may be a directory, for the same reason the env var may be.
    """
    if explicit is not None:
        path = Path(explicit)
        if path.is_dir() or path.suffix != ".pt":
            path = path / name
        if path.is_file():
            return path
        raise SystemExit(f"SAM2 checkpoint not found: {explicit}")

    found = find_checkpoint(name)
    if found is not None:
        return found

    tried = "\n".join(
        f"    {c}" + ("   <-- is a directory, not the .pt file" if c.is_dir() else "")
        for c in checkpoint_candidates(name)
    )
    hint = (
        f"  $SAM2_CHECKPOINT is set to '{os.environ['SAM2_CHECKPOINT']}' "
        f"but no {name} is there."
        if os.environ.get("SAM2_CHECKPOINT")
        else "  To keep the weights off this disk:  "
             "export SAM_CHECKPOINT_DIR=/big/disk/checkpoints"
    )
    raise SystemExit(
        f"SAM2 checkpoint not found. Looked for {name} in:\n{tried}\n\n{hint}\n"
        f"  Fetch them with:  ./setup_env.sh --checkpoint-only"
    )
