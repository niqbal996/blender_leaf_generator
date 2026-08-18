import os
import subprocess
from pathlib import Path


def run_shell(path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "view_in_blender.sh"
    env = os.environ.copy()
    env["WSL_DISTRO_NAME"] = "Ubuntu-22.04"
    return subprocess.run(
        ["bash", "-lc", f"source '{script}' && to_windows_path '{path}'"],
        capture_output=True,
        text=True,
        env=env,
    )


def test_to_windows_path_for_mnt_drive_prefers_windows_drive_letter():
    result = run_shell("/mnt/d/PBR_Scans/2026-08-18-Naeem/thistle1/plant")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == r"D:\PBR_Scans\2026-08-18-Naeem\thistle1\plant"


def test_to_windows_path_for_wsl_home_uses_unc():
    result = run_shell("/home/niqbal/git/blender_leaf_generator")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == r"\\wsl.localhost\Ubuntu-22.04\home\niqbal\git\blender_leaf_generator"
