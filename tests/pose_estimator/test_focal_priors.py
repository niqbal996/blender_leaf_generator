"""P1 records the lens; P3 groups by it.

The failure these guard against is silent: with no focal prior COLMAP guesses
1.2x the long edge, and a capture whose passes were shot at different zooms
then solves as one averaged camera that fits none of them -- registering
happily, and absorbing the mismatch into the camera positions instead.
"""

import json

import pytest
from PIL import Image

from pose_estimator.frames import exif_focal_px
from pose_estimator.reconstruction import _group_by_focal


def write_jpeg(path, size=(60, 40), focal_35mm=None):
    image = Image.new("RGB", size, (10, 120, 40))
    if focal_35mm is None:
        image.save(path, "JPEG")
        return path
    exif = image.getexif()
    # 41989 = FocalLengthIn35mmFilm, in the Exif IFD.
    exif.get_ifd(0x8769)[41989] = focal_35mm
    image.save(path, "JPEG", exif=exif)
    return path


def test_focal_from_35mm_equivalent_scales_to_the_written_size(tmp_path):
    photo = write_jpeg(tmp_path / "a.jpg", focal_35mm=72)
    # 35mm film is 36mm across, so f_px = f35 * long_edge / 36.
    assert exif_focal_px(photo, 1920, 1280) == pytest.approx(3840.0)
    # Scaled to whatever P1 actually wrote, so --photo-max-edge cannot skew it.
    assert exif_focal_px(photo, 960, 640) == pytest.approx(1920.0)


def test_focal_is_none_without_exif(tmp_path):
    photo = write_jpeg(tmp_path / "b.jpg")
    assert exif_focal_px(photo, 1920, 1280) is None


def test_focal_is_none_for_an_unreadable_file(tmp_path):
    broken = tmp_path / "c.jpg"
    broken.write_bytes(b"not a jpeg")
    assert exif_focal_px(broken, 1920, 1280) is None


def test_frames_group_one_camera_per_distinct_focal():
    names = [f"frame_{i:04d}.jpg" for i in range(4)]
    priors = {"frame_0000": 3840.0, "frame_0001": 3840.0,
              "frame_0002": 2560.0, "frame_0003": 2560.0}
    groups = _group_by_focal(names, priors)
    assert len(groups) == 2
    assert {focal for focal, _ in groups} == {3840.0, 2560.0}
    assert sorted(n for _, names in groups for n in names) == names


def test_frames_without_a_prior_share_the_unknown_group_last():
    names = [f"frame_{i:04d}.jpg" for i in range(3)]
    groups = _group_by_focal(names, {"frame_0000": 3840.0})
    assert groups[-1][0] is None
    assert groups[-1][1] == ["frame_0001.jpg", "frame_0002.jpg"]


def test_no_priors_at_all_is_one_shared_camera():
    # The degenerate case must stay identical to `--cameras single`, so video
    # captures and EXIF-stripped photos are no worse off than before.
    names = [f"frame_{i:04d}.jpg" for i in range(3)]
    assert _group_by_focal(names, {}) == [(None, names)]


def test_p3_refuses_frames_that_sources_json_does_not_list(tmp_path):
    from pose_estimator.cli.pose import run

    frames = tmp_path / "p1" / "frames"
    frames.mkdir(parents=True)
    for i in range(3):
        write_jpeg(frames / f"frame_{i:04d}.jpg")
    # Only two of the three are accounted for: the third is a leftover from an
    # earlier ingest, and defaulting it to pass 0 is what silently merges a
    # different capture into this one.
    (tmp_path / "p1" / "sources.json").write_text(
        json.dumps({"frame_0000": 0, "frame_0001": 0}))

    with pytest.raises(ValueError, match="frame_0002.jpg"):
        run(tmp_path)
