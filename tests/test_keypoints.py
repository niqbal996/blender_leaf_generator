import cv2
import numpy as np
import pytest

from leaf_generator.keypoints import NoContourError, estimate_attachment_point, pixel_to_local


def _draw_teardrop(path, width=400, height=200, tip_x=20):
    """Broad round end on the right, narrow tapering tip on the left."""
    img = np.zeros((height, width), dtype=np.uint8)
    cx, cy = width * 0.65, height / 2
    cv2.circle(img, (int(cx), int(cy)), int(height * 0.45), 255, -1)
    triangle = np.array(
        [[tip_x, cy], [cx, cy - height * 0.35], [cx, cy + height * 0.35]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(img, triangle, 255)
    cv2.imwrite(str(path), img)
    return width, height, cy


def test_estimate_attachment_point_finds_narrow_tip(tmp_path):
    mask_path = tmp_path / "teardrop_mask.png"
    width, height, cy = _draw_teardrop(mask_path)

    result = estimate_attachment_point(mask_path)

    x, y = result["pixel"]
    assert x < width * 0.15  # near the tapered tip, not the round end
    assert abs(y - cy) < height * 0.15
    assert result["narrow_width"] < result["broad_width"]
    assert result["confidence"] > 0.3


def test_estimate_attachment_point_empty_mask_raises(tmp_path):
    mask_path = tmp_path / "empty_mask.png"
    cv2.imwrite(str(mask_path), np.zeros((100, 100), dtype=np.uint8))

    with pytest.raises(NoContourError):
        estimate_attachment_point(mask_path)


def test_pixel_to_local_matches_mesh_vertex_formula():
    # Mirrors the (x_pixel/width - 0.5) * scale formula used when building
    # mesh vertices in leaf_generator.blender.mesh, so a keypoint lines up
    # with the mesh it was estimated from.
    local = pixel_to_local(pixel=(0, 0), image_size=(200, 100), scale_x=2.0, scale_y=1.0)
    assert local == pytest.approx((-1.0, -0.5, 0.0))

    local_center = pixel_to_local(pixel=(100, 50), image_size=(200, 100), scale_x=2.0, scale_y=1.0)
    assert local_center == pytest.approx((0.0, 0.0, 0.0))
