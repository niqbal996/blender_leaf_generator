import json

import pytest

from leaf_generator.calibration import find_calibration_log, load_session_calibration


def _write_log(path, pixelsize_mm=0.035, distance_z_mm=450.0):
    data = {
        "metadata": {
            "camera_info": {
                "camera_model": "ILCE-7RM4A",
                "lens_model": "FE 50mm F1.4 GM",
                "focal_length": "50",
            }
        },
        "parameters": {
            "pixelsize_mm": pixelsize_mm,
            "distance_z_mm": distance_z_mm,
        },
    }
    path.write_text(json.dumps(data))


def test_finds_preferred_side_log(tmp_path):
    _write_log(tmp_path / "oberseite_log.json")
    (tmp_path / "unterseite_log.json").write_text("{}")

    log_path = find_calibration_log(tmp_path, side="oberseite")

    assert log_path == tmp_path / "oberseite_log.json"


def test_falls_back_to_any_log_when_preferred_missing(tmp_path):
    _write_log(tmp_path / "unterseite_log.json")

    log_path = find_calibration_log(tmp_path, side="oberseite")

    assert log_path == tmp_path / "unterseite_log.json"


def test_returns_none_when_no_log(tmp_path):
    assert find_calibration_log(tmp_path) is None


def test_load_session_calibration_parses_fields(tmp_path):
    _write_log(tmp_path / "oberseite_log.json", pixelsize_mm=0.035, distance_z_mm=450.0)

    cal = load_session_calibration(tmp_path)

    assert cal is not None
    assert cal.pixel_size_mm == 0.035
    assert cal.distance_z_mm == 450.0
    assert cal.camera_model == "ILCE-7RM4A"
    assert cal.focal_length_mm == 50.0
    assert cal.px_to_mm(100) == pytest.approx(3.5)
    assert cal.px_to_m(100) == pytest.approx(0.0035)


def test_load_session_calibration_missing_pixelsize_returns_none(tmp_path):
    (tmp_path / "oberseite_log.json").write_text(json.dumps({"parameters": {}}))

    assert load_session_calibration(tmp_path) is None


def test_load_session_calibration_no_log_returns_none(tmp_path):
    assert load_session_calibration(tmp_path) is None
