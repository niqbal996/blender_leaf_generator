"""Where the SAM2 weights are looked for.

A 900 MB checkpoint cannot live in a repo checkout under a disk quota, so it
has to be possible to put it elsewhere and say so. The way that used to fail
is the case pinned first below: `export SAM2_CHECKPOINT=/big/disk/checkpoints/`
names the directory the weights were moved to, the search tested every
candidate with `-f`, a directory is not a file, and the candidate was dropped
without comment -- so the run died naming the repo path it had fallen back
to, which is the one place the user knew the weights were not.

The rule lives twice, in scripts/sam_checkpoints.sh and in
pose_estimator.checkpoints, because run_pipeline.sh and `pose-segment` are
both entry points. Two copies drift, so the last test here is the one that
matters: they must resolve the same case the same way.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHELL_LIB = REPO / "scripts" / "sam_checkpoints.sh"
NAME = "sam2.1_hiera_large.pt"


def shell_resolve(repo_root, env):
    """`find_sam_checkpoint` as run_pipeline.sh calls it: path, or ''."""
    script = f'set -euo pipefail\nREPO_ROOT="{repo_root}"\nsource "{SHELL_LIB}"\nfind_sam_checkpoint || true\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def shell_download_dir(repo_root, env):
    script = f'set -euo pipefail\nREPO_ROOT="{repo_root}"\nsource "{SHELL_LIB}"\nsam_checkpoint_dir\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def python_resolve(repo_root, env):
    """The same question asked of the Python mirror, in a clean interpreter."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "import pose_estimator.checkpoints as ck\n"
        "ck._REPO_ROOT = Path(%r)\n"
        "found = ck.find_checkpoint()\n"
        "print(found if found else '')\n" % (str(REPO / "src"), str(repo_root))
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A machine with no checkpoint anywhere the search knows about."""
    base = dict(PATH="/usr/bin:/bin", HOME=str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return base


@pytest.fixture
def elsewhere(tmp_path):
    """Weights on a big disk, outside the repo."""
    d = tmp_path / "netscratch" / "checkpoints"
    d.mkdir(parents=True)
    (d / NAME).write_bytes(b"weights")
    return d


@pytest.fixture
def repo_root(tmp_path):
    root = tmp_path / "repo"
    (root / "checkpoints").mkdir(parents=True)
    return root


# --- the reported failure ------------------------------------------------


@pytest.mark.parametrize("suffix", ["", "/"], ids=["no-slash", "trailing-slash"])
def test_env_var_may_name_the_directory_the_weights_were_moved_to(
        repo_root, env, elsewhere, suffix):
    env["SAM2_CHECKPOINT"] = str(elsewhere) + suffix
    assert shell_resolve(repo_root, env) == str(elsewhere / NAME)
    assert python_resolve(repo_root, env) == str(elsewhere / NAME)


def test_env_var_may_still_name_the_file(repo_root, env, elsewhere):
    env["SAM2_CHECKPOINT"] = str(elsewhere / NAME)
    assert shell_resolve(repo_root, env) == str(elsewhere / NAME)
    assert python_resolve(repo_root, env) == str(elsewhere / NAME)


def test_shared_dir_var_is_honoured(repo_root, env, elsewhere):
    env["SAM_CHECKPOINT_DIR"] = str(elsewhere)
    assert shell_resolve(repo_root, env) == str(elsewhere / NAME)
    assert python_resolve(repo_root, env) == str(elsewhere / NAME)


def test_relocated_weights_beat_a_copy_still_in_the_repo(repo_root, env, elsewhere):
    """Both exist: the explicit one wins, or moving them changed nothing."""
    (repo_root / "checkpoints" / NAME).write_bytes(b"stale")
    env["SAM2_CHECKPOINT"] = str(elsewhere)
    assert shell_resolve(repo_root, env) == str(elsewhere / NAME)
    assert python_resolve(repo_root, env) == str(elsewhere / NAME)


def test_repo_checkpoints_is_still_the_default(repo_root, env):
    (repo_root / "checkpoints" / NAME).write_bytes(b"weights")
    assert shell_resolve(repo_root, env) == str(repo_root / "checkpoints" / NAME)
    assert python_resolve(repo_root, env) == str(repo_root / "checkpoints" / NAME)


def test_nothing_anywhere_resolves_to_nothing(repo_root, env):
    assert shell_resolve(repo_root, env) == ""
    assert python_resolve(repo_root, env) == ""


# --- fetching and finding must agree -------------------------------------


def test_setup_downloads_where_the_pipeline_looks(repo_root, env, tmp_path):
    """The other half of the bug: weights fetched into a directory nothing
    searches are as missing as no weights at all."""
    target = tmp_path / "big" / "ckpts"
    env["SAM_CHECKPOINT_DIR"] = str(target)
    assert shell_download_dir(repo_root, env) == str(target)

    target.mkdir(parents=True)
    (target / NAME).write_bytes(b"weights")
    assert shell_resolve(repo_root, env) == str(target / NAME)


def test_a_directory_that_does_not_exist_yet_is_read_as_one(repo_root, env, tmp_path):
    """`--checkpoint-dir /not/created/yet` has no `.pt` on the end, so it is a
    directory to create rather than a file to write."""
    env["SAM2_CHECKPOINT"] = str(tmp_path / "not" / "yet")
    assert shell_download_dir(repo_root, env) == str(tmp_path / "not" / "yet")


def test_error_names_the_directory_the_user_actually_set(repo_root, env, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    env["SAM2_CHECKPOINT"] = str(empty)
    script = (f'set -euo pipefail\nREPO_ROOT="{repo_root}"\nsource "{SHELL_LIB}"\n'
              f'find_sam_checkpoint || sam_checkpoint_error "{NAME}"\n')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    # The path searched is reported, not the repo default it fell back to.
    assert str(empty / NAME) in out.stderr
    assert str(empty) in out.stderr
