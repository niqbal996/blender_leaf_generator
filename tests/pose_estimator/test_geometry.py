import cv2
import numpy as np

from pose_estimator.geometry import _copy_images, backend_code_cache, find_colmap_model, similarity_transform, stage_masked_images, uniformly_sample


def test_similarity_transform_recovers_scale_rotation_and_translation():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [1., 2., 3.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    target = 2.5 * source @ rotation.T + np.array([3., -4., 7.])

    scale, fitted_rotation, translation = similarity_transform(source, target)
    np.testing.assert_allclose(scale, 2.5, atol=1e-10)
    np.testing.assert_allclose(fitted_rotation, rotation, atol=1e-10)
    np.testing.assert_allclose(scale * source @ fitted_rotation.T + translation, target, atol=1e-10)


def test_uniform_sample_keeps_orbit_endpoints():
    paths = [__import__("pathlib").Path(f"frame_{i:04d}.jpg") for i in range(20)]
    sampled = uniformly_sample(paths, 5)
    assert sampled[0] == paths[0]
    assert sampled[-1] == paths[-1]
    assert len(sampled) == 5


def test_stage_masked_images_retains_frame_names_and_removes_background(tmp_path):
    frames = tmp_path / "p1" / "frames"
    masks = tmp_path / "p2" / "masks" / "plant"
    frames.mkdir(parents=True); masks.mkdir(parents=True)
    image = np.full((16, 20, 3), 200, np.uint8)
    mask = np.zeros((16, 20), np.uint8); mask[4:12, 6:14] = 255
    cv2.imwrite(str(frames / "frame_0000.jpg"), image)
    cv2.imwrite(str(masks / "frame_0000.png"), mask)

    staged = stage_masked_images(tmp_path, tmp_path / "out" / "images")
    result = cv2.imread(str(staged[0]))
    assert staged[0].name == "frame_0000.jpg"
    assert result[0, 0].max() < 4
    assert result[8, 10].mean() > 170


def test_find_colmap_model_accepts_text_and_binary_layouts(tmp_path):
    model = tmp_path / "nested" / "sparse" / "0"
    model.mkdir(parents=True)
    (model / "cameras.txt").write_text("")
    (model / "images.txt").write_text("")
    (model / "points3D.txt").write_text("")
    assert find_colmap_model(tmp_path) == model


def test_explicit_exporter_cache_does_not_depend_on_the_workdir(tmp_path):
    cache = tmp_path / "linux_cache"
    assert backend_code_cache(cache) == cache


def test_model_input_copy_does_not_request_source_metadata(tmp_path, monkeypatch):
    source = tmp_path / "source.jpg"
    source.write_bytes(b"plant pixels")

    def no_metadata(*args, **kwargs):
        raise AssertionError("copy2/copystat must not be used on a WSL mount")

    monkeypatch.setattr("shutil.copystat", no_metadata)
    _copy_images([source], tmp_path / "destination")
    assert (tmp_path / "destination" / source.name).read_bytes() == b"plant pixels"


def test_vggt_estimate_grows_with_frames_and_suggestion_fits_the_card():
    from pose_estimator.geometry import estimate_vggt_vram_gb, suggest_max_images

    assert estimate_vggt_vram_gb(30) > estimate_vggt_vram_gb(8) > estimate_vggt_vram_gb(1)
    assert estimate_vggt_vram_gb(8, bundle_adjust=True) > estimate_vggt_vram_gb(8)
    for capacity in (8.0, 12.0, 24.0, 80.0):
        assert estimate_vggt_vram_gb(suggest_max_images(capacity)) <= capacity
    assert suggest_max_images(2.0) >= 4  # never advise an unusable frame count


def test_exporter_lines_name_the_stage_that_follows_them():
    from pose_estimator.geometry import stage_for_line

    assert "peak-VRAM" in stage_for_line("vggt", "Loaded 27 images from /x/images")
    assert "weights" in stage_for_line("vggt", "Using dtype: torch.float16")
    assert stage_for_line("vggt", "some unremarkable warning") is None


def test_elapsed_labels_stay_short_until_an_hour():
    from pose_estimator.geometry import format_elapsed

    assert format_elapsed(9) == "00:09"
    assert format_elapsed(95) == "01:35"
    assert format_elapsed(3671) == "1:01:11"


def test_gpu_query_survives_wsl_placeholders_and_respects_device_selection(monkeypatch):
    import subprocess as sp

    from pose_estimator import geometry

    def fake_run(command, **kwargs):
        if "--query-compute-apps=pid,process_name,used_memory" in command:
            return sp.CompletedProcess(command, 0, stdout="63488, [Not Found], [N/A]\n", stderr="")
        return sp.CompletedProcess(
            command, 0, stderr="",
            stdout="0, RTX 2070, 7788, 8192\n1, RTX 4090, 512, 24564\n",
        )

    monkeypatch.setattr(geometry.subprocess, "run", fake_run)
    both = geometry.read_gpu_memory()
    assert [gpu["index"] for gpu in both] == ["0", "1"]
    assert abs(both[0]["free_gb"] - (8192 - 7788) / 1024) < 1e-6
    assert [gpu["index"] for gpu in geometry.read_gpu_memory("1")] == ["1"]
    # WSL reports the pid but not its memory; the pid alone is the useful part.
    assert geometry.read_gpu_processes() == [{"pid": 63488, "name": "[Not Found]", "used_gb": None}]


def test_missing_nvidia_smi_is_reported_as_unknown_not_as_zero(monkeypatch):
    from pose_estimator import geometry

    def no_binary(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(geometry.subprocess, "run", no_binary)
    assert geometry.read_gpu_memory() is None
    assert geometry.read_gpu_processes() is None


def test_sigkill_is_diagnosed_as_a_host_oom_with_a_smaller_frame_count(tmp_path):
    from pose_estimator.geometry import diagnose_exporter_failure

    usage = {"stage": "aggregator + camera/depth heads", "last_line": "Loaded 27 images",
             "peak_gpu_used_gb": 7.9, "gpu_total_gb": 8.0, "peak_process_rss_gb": 11.2,
             "min_host_available_gb": 0.3, "elapsed_seconds": 723}
    message = diagnose_exporter_failure("vggt", -9, "no traceback here", 27, usage, tmp_path / "stdout.log")
    assert "out-of-memory" in message
    assert "12:03" in message and "aggregator" in message
    assert "--max-images 8" in message
    assert "7.9 of 8.0 GB" in message


def test_cuda_oom_in_the_output_is_recognised_without_a_signal(tmp_path):
    from pose_estimator.geometry import diagnose_exporter_failure

    usage = {"stage": "aggregator", "last_line": "x", "peak_gpu_used_gb": 7.9, "gpu_total_gb": 8.0,
             "peak_process_rss_gb": 4.0, "min_host_available_gb": 6.0, "elapsed_seconds": 30}
    oom = diagnose_exporter_failure(
        "vggt", 1, "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        27, usage, tmp_path / "log", bundle_adjust=True)
    assert "out-of-memory" in oom and "without --bundle-adjust" in oom

    # An import failure is an environment fault, so it must not read as OOM;
    # a failure matching no pattern falls back to the last line the run printed.
    imports = diagnose_exporter_failure("vggt", 1, "ImportError: no module named lightglue",
                                        27, usage, tmp_path / "log")
    assert "out-of-memory" not in imports and "environment problem" in imports

    other = diagnose_exporter_failure("vggt", 1, "AssertionError: unexpected image shape",
                                      27, usage, tmp_path / "log")
    assert "out-of-memory" not in other and "environment problem" not in other
    assert "Last output: x" in other


def test_silent_exporter_is_reported_live_and_logged_as_it_runs(tmp_path, monkeypatch, capsys):
    import subprocess
    import sys

    from pose_estimator import geometry

    monkeypatch.setattr(geometry, "_SAMPLE_SECONDS", 0.05)
    script = tmp_path / "fake_exporter.py"
    script.write_text("import sys, time\n"
                      "print('Loaded 27 images from /x'); sys.stdout.flush()\n"
                      "time.sleep(0.8)\n"
                      "print('Converting to COLMAP format')\n")
    log = tmp_path / "stdout.log"
    process = subprocess.Popen([sys.executable, "-u", str(script)], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    output, usage = geometry.stream_exporter_output(process, "vggt", log, heartbeat_seconds=0.2)

    assert process.wait() == 0
    printed = capsys.readouterr().out
    assert "still running, no output for" in printed  # the point of the heartbeat
    assert "peak-VRAM stage" in printed
    assert "[vggt 00:00]" in printed
    assert "Converting to COLMAP format" in output
    assert log.read_text() == output  # the log is complete even if the process is killed
    assert "COLMAP sparse model" in usage["stage"]
    assert usage["elapsed_seconds"] >= 0.8


def test_zero_heartbeat_stays_quiet_but_still_records_peaks(tmp_path, monkeypatch, capsys):
    import subprocess
    import sys

    from pose_estimator import geometry

    monkeypatch.setattr(geometry, "_SAMPLE_SECONDS", 0.05)
    monkeypatch.setattr(geometry, "read_gpu_memory",
                        lambda device=None: [{"index": "0", "name": "fake", "used_gb": 6.5,
                                              "total_gb": 8.0, "free_gb": 1.5}])
    script = tmp_path / "quiet.py"
    script.write_text("import time\ntime.sleep(0.5)\n")
    process = subprocess.Popen([sys.executable, "-u", str(script)], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    _, usage = geometry.stream_exporter_output(process, "vggt", tmp_path / "log", heartbeat_seconds=0)

    assert "still running" not in capsys.readouterr().out
    assert usage["peak_gpu_used_gb"] == 6.5


def _probe_output(python, modules, cuda=True):
    import json

    from pose_estimator.geometry import _PROBE_MARKER

    report = {"python": list(python), "executable": "/env/bin/python", "modules": modules}
    if cuda:
        report["torch_cuda"] = {"available": True, "build": "12.4", "devices": ["A100"]}
    return "some import warning\n" + _PROBE_MARKER + json.dumps(report) + "\n"


def test_old_python_is_rejected_with_the_pep604_reason_and_the_model_python_fix(monkeypatch):
    import subprocess as sp

    import pytest

    from pose_estimator import geometry

    ready = {name: "2.0" for name in geometry._REQUIRED_MODULES["vggt"]}
    monkeypatch.setattr(geometry.subprocess, "run",
                        lambda command, **kw: sp.CompletedProcess(
                            command, 0, stdout=_probe_output((3, 9, 18), ready), stderr=""))

    with pytest.raises(RuntimeError) as failure:
        geometry.require_model_environment("vggt")
    message = str(failure.value)
    assert "Python 3.9.18" in message and "3.10 or newer" in message
    assert "--model-python" in message
    assert "Nothing was staged or downloaded" in message


def test_missing_pycolmap_explains_why_the_extra_cannot_pin_it(monkeypatch):
    import subprocess as sp

    import pytest

    from pose_estimator import geometry

    modules = {name: "2.0" for name in geometry._REQUIRED_MODULES["vggt"]}
    modules["pycolmap"] = "MISSING: ModuleNotFoundError: No module named 'pycolmap'"
    monkeypatch.setattr(geometry.subprocess, "run",
                        lambda command, **kw: sp.CompletedProcess(
                            command, 0, stdout=_probe_output((3, 11, 0), modules), stderr=""))

    with pytest.raises(RuntimeError) as failure:
        geometry.require_model_environment("vggt")
    message = str(failure.value)
    assert "pip install pycolmap" in message
    assert "clobber" in message  # the reason it is not simply added to [vggt]


def test_a_pycolmap_missing_its_cuda_runtime_is_explained_differently_from_an_absent_one(monkeypatch):
    import subprocess as sp

    import pytest

    from pose_estimator import geometry

    modules = {name: "2.0" for name in geometry._REQUIRED_MODULES["vggt"]}
    modules["pycolmap"] = "MISSING: ImportError: libcudart.so.12: cannot open shared object file"
    monkeypatch.setattr(geometry.subprocess, "run",
                        lambda command, **kw: sp.CompletedProcess(
                            command, 0, stdout=_probe_output((3, 11, 0), modules), stderr=""))

    with pytest.raises(RuntimeError) as failure:
        geometry.require_model_environment("vggt")
    message = str(failure.value)
    # Same install command as an absent package, but the reason differs: a CUDA
    # build is pointless for an exporter that only writes a model.
    assert 'pip install "pycolmap==3.10.0"' in message
    assert "never extracts features" in message
    assert "clobber" not in message


def test_a_complete_environment_passes_and_reports_its_gpus(monkeypatch):
    import subprocess as sp

    from pose_estimator import geometry

    ready = {name: "2.0" for name in geometry._REQUIRED_MODULES["vggt"]}
    monkeypatch.setattr(geometry.subprocess, "run",
                        lambda command, **kw: sp.CompletedProcess(
                            command, 0, stdout=_probe_output((3, 11, 0), ready), stderr=""))

    report = geometry.require_model_environment("vggt")
    assert report["torch_cuda"]["devices"] == ["A100"]
    assert report["python"] == [3, 11, 0]


def test_the_probe_runs_in_the_interpreter_named_by_model_python(monkeypatch):
    import subprocess as sp

    from pose_estimator import geometry

    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        ready = {name: "2.0" for name in geometry._REQUIRED_MODULES["vggt"]}
        return sp.CompletedProcess(command, 0, stdout=_probe_output((3, 12, 0), ready), stderr="")

    monkeypatch.setattr(geometry.subprocess, "run", fake_run)
    geometry.require_model_environment("vggt", "/opt/conda/envs/geometry/bin/python")
    assert seen["command"][0] == "/opt/conda/envs/geometry/bin/python"
    assert "pycolmap" in seen["command"]  # asked about in that environment, not this one


def test_an_unreadable_probe_reports_the_interpreter_output(monkeypatch):
    import subprocess as sp

    import pytest

    from pose_estimator import geometry

    monkeypatch.setattr(geometry.subprocess, "run",
                        lambda command, **kw: sp.CompletedProcess(
                            command, 1, stdout="", stderr="bad interpreter"))

    with pytest.raises(RuntimeError, match="bad interpreter"):
        geometry.probe_model_environment("vggt")


def test_an_import_time_crash_is_diagnosed_as_environment_not_memory(tmp_path):
    from pose_estimator.geometry import diagnose_exporter_failure

    usage = {"stage": "loading VGGT-1B weights", "last_line": "", "peak_gpu_used_gb": 0.2,
             "gpu_total_gb": 8.0, "peak_process_rss_gb": 0.5,
             "min_host_available_gb": 12.0, "elapsed_seconds": 4}
    pep604 = diagnose_exporter_failure(
        "vggt", 1, "TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'",
        27, usage, tmp_path / "log")
    assert "environment problem" in pep604
    assert "3.10" in pep604 and "--model-python" in pep604
    assert "out-of-memory" not in pep604

    no_pycolmap = diagnose_exporter_failure(
        "vggt", 1, "ModuleNotFoundError: No module named 'pycolmap'", 27, usage, tmp_path / "log")
    assert "pip install pycolmap" in no_pycolmap


def test_pinned_pycolmap_is_read_from_the_checkout_not_hardcoded(tmp_path):
    from pose_estimator.geometry import exporter_pinned_pycolmap

    repo = tmp_path / "vggt"
    repo.mkdir()
    assert exporter_pinned_pycolmap(None) == "3.10.0"          # no checkout yet
    assert exporter_pinned_pycolmap(repo) == "3.10.0"          # no requirements file
    (repo / "requirements_demo.txt").write_text("trimesh\npycolmap==3.11.1\nlightglue\n")
    assert exporter_pinned_pycolmap(repo) == "3.11.1"          # follows an upstream bump


def test_importable_but_incompatible_pycolmap_is_rejected_before_any_work(monkeypatch, tmp_path):
    import subprocess as sp

    import pytest

    from pose_estimator import geometry

    ready = {name: "2.0" for name in geometry._REQUIRED_MODULES["vggt"]}
    stdout = _probe_output((3, 12, 0), ready).replace(
        '"modules"', '"pycolmap_version": "4.1.1", "pycolmap_exporter_api": '
        '"INCOMPATIBLE: AttributeError: \'pycolmap._core.Image\' object has no attribute \'id\'", '
        '"modules"')
    monkeypatch.setattr(geometry.subprocess, "run",
                        lambda command, **kw: sp.CompletedProcess(command, 0, stdout=stdout, stderr=""))
    repo = tmp_path / "vggt"
    repo.mkdir()
    (repo / "requirements_demo.txt").write_text("pycolmap==3.10.0\n")

    with pytest.raises(RuntimeError) as failure:
        geometry.require_model_environment("vggt", None, repo)
    message = str(failure.value)
    assert "pycolmap 4.1.1" in message
    assert "no attribute 'id'" in message
    assert 'pip install "pycolmap==3.10.0"' in message
    assert "Nothing was staged or downloaded" in message


def test_a_pycolmap_wheel_that_cannot_load_its_core_gets_the_plain_build_advice(monkeypatch):
    import subprocess as sp

    import pytest

    from pose_estimator import geometry

    modules = {name: "2.0" for name in geometry._REQUIRED_MODULES["vggt"]}
    modules["pycolmap"] = ("MISSING: RuntimeError: Cannot import the C++ backend pycolmap._core")
    monkeypatch.setattr(geometry.subprocess, "run",
                        lambda command, **kw: sp.CompletedProcess(
                            command, 0, stdout=_probe_output((3, 12, 0), modules), stderr=""))

    with pytest.raises(RuntimeError) as failure:
        geometry.require_model_environment("vggt")
    message = str(failure.value)
    assert 'pip install "pycolmap==3.10.0"' in message
    assert "never extracts features" in message  # why the CUDA build is pointless here


def test_a_post_launch_pycolmap_api_break_is_named_rather_than_reported_as_exit_1(tmp_path):
    from pose_estimator.geometry import diagnose_exporter_failure

    usage = {"stage": "building the COLMAP sparse model (CPU)",
             "last_line": "AttributeError: 'pycolmap._core.Image' object has no attribute 'id'",
             "peak_gpu_used_gb": 10.1, "gpu_total_gb": 45.0, "peak_process_rss_gb": 10.2,
             "min_host_available_gb": 775.6, "elapsed_seconds": 216}
    message = diagnose_exporter_failure(
        "vggt", 1, "AttributeError: 'pycolmap._core.Image' object has no attribute 'id'",
        27, usage, tmp_path / "log")
    assert "environment problem" in message
    assert "requirements_demo.txt" in message
    assert "out-of-memory" not in message  # 10 of 45 GB is not a memory failure
