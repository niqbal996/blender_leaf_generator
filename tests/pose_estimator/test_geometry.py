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

    other = diagnose_exporter_failure("vggt", 1, "ImportError: no module named lightglue",
                                      27, usage, tmp_path / "log")
    assert "out-of-memory" not in other
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
