import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "view_in_blender.sh"


def run_shell(tmp_path, path, wslpath_output=None):
    """Call to_windows_path in a subshell, with a stubbed `wslpath`.

    The stub is what makes these tests describe the script's own logic instead
    of whatever drives happen to be mounted on the machine running them.
    `wslpath_output=None` installs a shim that fails, standing in for a system
    where wslpath is absent or cannot resolve the path.
    """
    env = os.environ.copy()
    env["WSL_DISTRO_NAME"] = "Ubuntu-22.04"

    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "wslpath"
    if wslpath_output is None:
        stub.write_text("#!/usr/bin/env bash\nexit 1\n")
    else:
        # Via a file, so backslash-heavy Windows paths need no shell quoting.
        answer = bindir / "answer.txt"
        answer.write_text(wslpath_output)
        stub.write_text(f"#!/usr/bin/env bash\ncat {answer}\n")
    stub.chmod(0o755)
    env["PATH"] = f"{bindir}:{env['PATH']}"

    return subprocess.run(
        ["bash", "-lc", f"source '{SCRIPT}' && to_windows_path '{path}'"],
        capture_output=True, text=True, env=env,
    )


def test_mnt_drive_uses_the_letter_wslpath_reports(tmp_path):
    # The mount point letter is not the drive letter: an external disk mounted
    # by hand lands wherever there was a free slot, so /mnt/e can be F:.
    # Guessing E: from the path sent Blender to a drive that does not exist.
    result = run_shell(tmp_path, "/mnt/e/Camera_rig_data/thistle3/plant",
                       r"F:\Camera_rig_data\thistle3\plant")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == r"F:\Camera_rig_data\thistle3\plant"


def test_mnt_drive_falls_back_to_the_letter_guess_without_wslpath(tmp_path):
    result = run_shell(tmp_path, "/mnt/d/PBR_Scans/2026-08-18-Naeem/thistle1/plant", None)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == r"D:\PBR_Scans\2026-08-18-Naeem\thistle1\plant"


def test_wsl_home_uses_unc(tmp_path):
    unc = r"\\wsl.localhost\Ubuntu-22.04\home\niqbal\git\blender_leaf_generator"
    result = run_shell(tmp_path, "/home/niqbal/git/blender_leaf_generator", unc)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == unc


def test_wsl_home_falls_back_to_unc_without_wslpath(tmp_path):
    result = run_shell(tmp_path, "/home/niqbal/git/blender_leaf_generator", None)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        r"\\wsl.localhost\Ubuntu-22.04\home\niqbal\git\blender_leaf_generator")
