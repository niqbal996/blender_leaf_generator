"""The command line is where these runs go wrong.

Every path in the long form repeats the same dataset prefix, so one stale
component is easy to type and invisible on review -- which is exactly how
sugarbeet_4 came to be solved as two different plants. These cover the
resolution that removes the repetition, and the config file that removes the
settings which never change.
"""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_pipeline.sh"


def run(*args, cwd=None):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True,
                          cwd=str(cwd) if cwd else None)


@pytest.fixture
def dataset(tmp_path):
    ds = tmp_path / "sugarbeet_4"
    for name in ("pass1", "pass2", "pass10"):
        (ds / name).mkdir(parents=True)
        (ds / name / "DSC_0001.JPG").write_bytes(b"x")
    return ds


def plan(result):
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_dataset_directory_supplies_passes_and_workdir(dataset):
    out = plan(run(str(dataset), "--dry-run"))
    assert f"photos        {dataset}/pass1" in out
    assert f"workdir       {dataset}/plant" in out


def test_passes_are_ordered_naturally_not_lexically(dataset):
    # pass10 after pass2, so frame numbering follows the capture, not ASCII.
    out = plan(run(str(dataset), "--dry-run"))
    order = [line.split()[1] for line in out.splitlines() if line.startswith("  photos")]
    assert [Path(p).name for p in order] == ["pass1", "pass2", "pass10"]


def test_a_flat_directory_of_photos_is_one_pass(tmp_path):
    # thistle3's layout: the JPEGs sit in the dataset directory itself.
    ds = tmp_path / "thistle3"
    ds.mkdir()
    (ds / "DSC_0220.JPG").write_bytes(b"x")
    out = plan(run(str(ds), "--dry-run"))
    assert f"photos        {ds}  (1)" in out
    assert f"workdir       {ds}/plant" in out


def test_explicit_workdir_still_wins(dataset, tmp_path):
    out = plan(run(str(dataset), "--workdir", str(tmp_path / "elsewhere"), "--dry-run"))
    assert f"workdir       {tmp_path}/elsewhere" in out


def test_a_missing_dataset_fails_before_any_work(dataset):
    result = run(str(dataset.parent / "nope"), "--dry-run")
    assert result.returncode != 0
    assert "no such dataset directory" in result.stderr


def test_two_dataset_arguments_are_refused(dataset, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    result = run(str(dataset), str(other), "--dry-run")
    assert result.returncode != 0
    assert "at most one dataset directory" in result.stderr


def test_config_supplies_settings(dataset, tmp_path):
    conf = tmp_path / "pipeline.conf"
    conf.write_text("# comment\narchitecture = rosette\nlow_texture = 1\n\n")
    out = plan(run(str(dataset), "--config", str(conf), "--dry-run"))
    assert "architecture  rosette" in out
    assert "low-texture   on" in out


def test_a_flag_beats_the_config(dataset, tmp_path):
    conf = tmp_path / "pipeline.conf"
    conf.write_text("architecture = rosette\n")
    out = plan(run(str(dataset), "--config", str(conf), "--architecture", "upright", "--dry-run"))
    assert "architecture  upright" in out


def test_a_dataset_config_overrides_a_session_config(dataset):
    (dataset.parent / "pipeline.conf").write_text("architecture = upright\n")
    (dataset / "pipeline.conf").write_text("architecture = rosette\n")
    out = plan(run(str(dataset), "--dry-run", cwd=dataset))
    assert "architecture  rosette" in out


def test_an_unknown_config_key_is_an_error(dataset, tmp_path):
    conf = tmp_path / "pipeline.conf"
    conf.write_text("prompt_bnak = /x\n")
    result = run(str(dataset), "--config", str(conf), "--dry-run")
    assert result.returncode != 0
    assert "unknown setting 'prompt_bnak'" in result.stderr


def test_a_malformed_config_line_is_an_error(dataset, tmp_path):
    conf = tmp_path / "pipeline.conf"
    conf.write_text("architecture rosette\n")
    result = run(str(dataset), "--config", str(conf), "--dry-run")
    assert result.returncode != 0
    assert "expected key=value" in result.stderr


def test_dry_run_does_not_start_the_pipeline(dataset):
    out = plan(run(str(dataset), "--dry-run"))
    assert "nothing was run" in out
    assert not (dataset / "plant" / "p1").exists()


def test_the_token_is_not_echoed(dataset, tmp_path):
    conf = tmp_path / "pipeline.conf"
    conf.write_text("hf_token = hf_secretvalue123\n")
    out = plan(run(str(dataset), "--config", str(conf), "--dry-run"))
    assert "hf_secretvalue123" not in out
    assert "hf token      set" in out
