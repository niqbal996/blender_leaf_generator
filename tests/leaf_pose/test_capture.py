"""Reading the capture: which frames are which, where the lights are,
and what the 12 of them say about the surface."""

import numpy as np
import pytest

from leaf_pose import photometric, raw, rig


def make_frames(tmp_path, count, suffix=".ARW"):
    paths = []
    for index in range(count):
        path = tmp_path / f"DSC{index:05d}{suffix}"
        path.write_bytes(b"")
        paths.append(path)
    return paths


def test_split_uses_the_rigs_own_order_when_nothing_contradicts_it():
    frames = [f"f{i}" for i in range(24)]
    capture = raw.split_polarisation(frames)
    assert capture.parallel == frames[:12]
    assert capture.crossed == frames[12:]


def test_split_follows_brightness_when_the_capture_is_saved_backwards():
    """Crossed polarisers cost both the reflection and half the transmission,
    so the crossed half is the darker one whichever order it was written in."""
    frames = [f"f{i}" for i in range(24)]
    dark_first = [2.0] * 12 + [10.0] * 12

    capture = raw.split_polarisation(frames, "auto", dark_first)
    assert capture.crossed == frames[:12]
    assert capture.parallel == frames[12:]


def test_an_odd_or_single_frame_set_is_not_a_polarised_pair_set():
    for count in (1, 3, 7):
        capture = raw.split_polarisation([f"f{i}" for i in range(count)])
        assert capture.parallel == []
        assert len(capture.crossed) == count
        assert not capture.has_polarisation


def test_find_frames_prefers_raw_and_reports_it(tmp_path):
    make_frames(tmp_path, 4)
    (tmp_path / "preview.jpg").write_bytes(b"")

    frames, is_raw = raw.find_frames(tmp_path)
    assert is_raw and len(frames) == 4


def test_find_frames_rejects_a_folder_with_nothing_readable(tmp_path):
    (tmp_path / "notes.txt").write_text("")
    with pytest.raises(raw.CaptureError):
        raw.find_frames(tmp_path)


def test_specular_residual_is_the_difference_the_polarisers_left():
    """The crossed image is scaled to match before subtracting, because the
    two filter pairs do not pass the same amount of light."""
    crossed = np.full((32, 32, 3), 0.2, np.float32)
    parallel = crossed * 1.7
    parallel[10:14, 10:14] += 0.5  # a highlight only the parallel set sees

    residual = raw.specular_residual(parallel, crossed)
    assert residual[12, 12] == pytest.approx(0.5, abs=0.02)
    assert residual[0, 0] == pytest.approx(0.0, abs=0.02)


def test_rig_parses_a_file_that_is_not_well_formed_xml(tmp_path):
    """The rig writes raw angle brackets inside attribute values."""
    path = tmp_path / "rigdef_cam.xml"
    path.write_text(
        '<document><lights origin="0 0 0" framelabel="<cam>_<frame>">\n'
        '  <light sidx="1" spath="<frame>/lit/69.jpg" pos="170 -236 0" />\n'
        '  <light sidx="0" spath="<frame>/lit/68.jpg" pos="255 -15 0" />\n'
        '</lights></document>')

    parsed = rig.load_rig(path)
    assert parsed.num_lights == 2
    # Sorted by sidx, not by file order.
    assert parsed.positions[0].tolist() == [255.0, -15.0, 0.0]


def test_working_distance_from_the_measured_scale():
    """m = f/(z-f), so a 50 mm lens at 0.045 mm/px sits about 650 mm off."""
    assert rig.working_distance_mm(0.045, 50.0, 9504) == pytest.approx(649, abs=5)


def test_working_distance_does_not_change_with_the_decode_resolution():
    """Magnification is a property of the optics, not of how the RAW was read.

    Decoding at half size halves the image width and doubles the millimetres
    per pixel with it; both cancel. Hard-coding the native pixel pitch broke
    this and reported 2036 mm for a copy stand about 600 mm tall.
    """
    full = rig.working_distance_mm(0.0746, 50.0, 9568)
    half = rig.working_distance_mm(0.0746 * 2, 50.0, 9568 // 2)
    assert full == pytest.approx(half, rel=1e-9)


def _synthetic_rib(height=90, width=70, crest=35, amplitude=4.0):
    xs = np.arange(width)[None, :].repeat(height, 0)
    surface = np.exp(-((xs - crest) ** 2) / (2 * 8.0 ** 2)) * amplitude
    dy, dx = np.gradient(surface)
    normals = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def _lit(normals, lights=12):
    ring = np.array([[220 * np.cos(a), 220 * np.sin(a), 0.0]
                     for a in np.linspace(0, 2 * np.pi, lights, endpoint=False)])
    directions = rig.light_directions(rig.Rig(positions=ring),
                                      normals.shape[:2], 0.05, 650.0)
    stack = np.clip(np.einsum("nhwi,hwi->nhw", directions, normals), 0, None) * 0.6
    return stack.astype(np.float32), directions


def test_photometric_stereo_recovers_the_normals():
    normals = _synthetic_rib()
    solved = photometric.solve(*_lit(normals))

    error = np.degrees(np.arccos(np.clip((solved.normals * normals).sum(-1), -1, 1)))
    assert float(np.percentile(error, 95)) < 1.0


def test_the_crease_is_found_at_the_crest_and_not_on_its_flanks():
    """A sign error here does not fail loudly -- it returns the flanks."""
    for crest in (35, 20):
        solved = photometric.solve(*_lit(_synthetic_rib(crest=crest)))
        assert int(solved.ridge.mean(axis=0).argmax()) == pytest.approx(crest, abs=2)


def test_a_groove_does_not_read_as_a_ridge():
    """The seam where one leaf overlaps another must not look like a midrib."""
    solved = photometric.solve(*_lit(_synthetic_rib(amplitude=-4.0)))
    assert float(solved.ridge[:, 35].mean()) < 0.1


def test_extreme_observations_are_dropped_before_the_fit():
    stack = np.tile(np.arange(12, dtype=np.float32)[:, None, None], (1, 4, 4))
    weights = photometric.observation_weights(stack)
    assert weights[:, 0, 0].tolist() == [0, 0, 0] + [1] * 8 + [0]


def test_scale_from_optics_inverts_the_working_distance():
    """The two directions have to agree, or one of them is wrong."""
    distance = rig.working_distance_mm(0.0746, 50.0, 9568)
    assert rig.scale_from_optics(50.0, distance, 9568) == pytest.approx(0.0746,
                                                                       rel=1e-9)


def test_scale_from_optics_refuses_a_subject_inside_the_focal_length():
    with pytest.raises(ValueError):
        rig.scale_from_optics(50.0, 40.0, 9568)
