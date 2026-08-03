import numpy as np

from pose_estimator.ply_io import read_ply_vertices, write_ply_vertices


def test_round_trip_preserves_names_and_values(tmp_path):
    fields = {
        "x": np.array([0.0, 1.5, -2.25], dtype=np.float32),
        "y": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "z": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "red": np.array([0, 128, 255], dtype=np.uint8),
        "green": np.array([255, 128, 0], dtype=np.uint8),
        "blue": np.array([10, 20, 30], dtype=np.uint8),
    }

    path = tmp_path / "cloud.ply"
    write_ply_vertices(path, fields)
    result = read_ply_vertices(path)

    assert list(result.keys()) == list(fields.keys())
    for name in fields:
        np.testing.assert_array_equal(result[name], fields[name])
        assert result[name].dtype == fields[name].dtype


def test_round_trip_many_named_columns(tmp_path):
    """Mirrors the shape of a Gaussian ply: many float32 columns beyond xyz."""
    n = 5
    fields = {"x": np.zeros(n, dtype=np.float32), "y": np.zeros(n, dtype=np.float32), "z": np.zeros(n, dtype=np.float32)}
    for i in range(6):
        fields[f"f_rest_{i}"] = np.linspace(-1, 1, n, dtype=np.float32) * (i + 1)
    fields["opacity"] = np.array([0.1, 0.5, 0.9, -0.2, 2.0], dtype=np.float32)

    path = tmp_path / "gaussians.ply"
    write_ply_vertices(path, fields)
    result = read_ply_vertices(path)

    assert set(result.keys()) == set(fields.keys())
    for name in fields:
        np.testing.assert_allclose(result[name], fields[name], rtol=0, atol=1e-6)


def test_mismatched_length_raises(tmp_path):
    import pytest

    fields = {"x": np.zeros(3, dtype=np.float32), "y": np.zeros(4, dtype=np.float32)}
    with pytest.raises(ValueError):
        write_ply_vertices(tmp_path / "bad.ply", fields)


def test_binary_round_trip_preserves_names_and_values(tmp_path):
    fields = {
        "x": np.array([0.0, 1.5, -2.25], dtype=np.float32),
        "y": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "z": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "opacity": np.array([-1.5, 0.0, 3.25], dtype=np.float32),
        "red": np.array([0, 128, 255], dtype=np.uint8),
    }

    path = tmp_path / "cloud_binary.ply"
    write_ply_vertices(path, fields, binary=True)
    result = read_ply_vertices(path)

    assert list(result.keys()) == list(fields.keys())
    for name in fields:
        np.testing.assert_array_equal(result[name], fields[name])
        assert result[name].dtype == fields[name].dtype


def test_binary_file_is_smaller_than_ascii_for_many_rows(tmp_path):
    n = 200
    fields = {"x": np.linspace(0, 1, n, dtype=np.float32), "y": np.linspace(1, 2, n, dtype=np.float32)}

    ascii_path = tmp_path / "ascii.ply"
    binary_path = tmp_path / "binary.ply"
    write_ply_vertices(ascii_path, fields, binary=False)
    write_ply_vertices(binary_path, fields, binary=True)

    assert binary_path.stat().st_size < ascii_path.stat().st_size
