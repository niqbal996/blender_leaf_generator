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
