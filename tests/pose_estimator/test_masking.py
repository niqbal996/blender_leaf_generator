import cv2
import numpy as np

from pose_estimator.masking import black_background_mask, vegetation_mask


def _synthetic_hand_and_plant_frame(size=300):
    """Gray-ish blurred background, a skin-tone rectangle (finger stand-in),
    and a green rectangle (plant stand-in).
    """
    img = np.full((size, size, 3), (110, 110, 110), dtype=np.uint8)  # BGR gray background
    # skin-tone rectangle (BGR): a typical fair skin tone
    cv2.rectangle(img, (20, 150), (280, 290), (120, 170, 210), -1)
    # green rectangle (plant): BGR with strong green channel
    cv2.rectangle(img, (100, 30), (200, 140), (40, 160, 60), -1)
    return img


def test_vegetation_mask_selects_green_not_skin():
    img = _synthetic_hand_and_plant_frame()
    mask = vegetation_mask(img, exg_threshold=0.08)

    green_region = mask[30:140, 100:200]
    skin_region = mask[150:290, 20:280]

    assert (green_region > 0).mean() > 0.8
    assert (skin_region > 0).mean() < 0.1


def test_black_background_mask_selects_bright_subject():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (140, 140), (200, 200, 200), -1)

    mask = black_background_mask(img, brightness_threshold=30)

    subject_region = mask[60:140, 60:140]
    background_region = mask[0:40, 0:40]

    assert (subject_region > 0).mean() > 0.9
    assert (background_region > 0).mean() < 0.05


def test_masks_are_binary():
    img = _synthetic_hand_and_plant_frame()
    mask = vegetation_mask(img)
    assert set(np.unique(mask)).issubset({0, 255})
