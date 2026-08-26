"""P4b must not train away what the holder hides.

The exposed root is behind the pliers for most of a turntable rotation, so in
~73% of frames the plant mask says "empty" at pixels that are merely hidden.
Fed to the silhouette loss that is absence of evidence taught as evidence of
absence, and opacity there is driven to zero -- which is why the root
survived the P4a carve (hull z -0.339..0.884) and was still missing from the
P4b surface (z 0.000..0.894).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pose_estimator.surfels import TrainView


def weighted_mask_term(alpha, mask, occluder):
    """The loss as train_surfels computes it, in isolation."""
    alpha = torch.tensor(alpha)
    mask = torch.tensor(mask)
    if occluder is None:
        return float((alpha - mask).abs().mean())
    weight = torch.tensor(1.0 - occluder)
    return float(((alpha - mask).abs() * weight).sum() / weight.sum().clamp(min=1.0))


def test_hidden_pixels_do_not_penalise_opacity():
    """A surfel behind the holder must cost nothing in the silhouette term."""
    alpha = np.zeros((4, 4), np.float32)
    alpha[1:3, 1:3] = 1.0                 # the root, rendered opaque
    mask = np.zeros((4, 4), np.float32)   # not in the plant mask: it is hidden
    occluder = np.zeros((4, 4), np.float32)
    occluder[1:3, 1:3] = 1.0

    penalised = weighted_mask_term(alpha, mask, None)
    excused = weighted_mask_term(alpha, mask, occluder)

    assert penalised > 0.2, "fixture must reproduce the penalty"
    assert excused == 0.0, "hidden pixels must carry no silhouette penalty"


def test_visible_disagreement_is_still_penalised():
    """Excusing occluded pixels must not switch the silhouette term off."""
    alpha = np.zeros((4, 4), np.float32)
    alpha[0, 0] = 1.0                     # a floater in plain view
    mask = np.zeros((4, 4), np.float32)
    occluder = np.zeros((4, 4), np.float32)
    occluder[3, 3] = 1.0                  # occlusion elsewhere

    assert weighted_mask_term(alpha, mask, occluder) > 0.0


def test_view_defaults_to_no_occluder():
    """Captures with no holder masks keep the old behaviour exactly."""
    view = TrainView(image=np.zeros((2, 2, 3), np.float32),
                     mask=np.zeros((2, 2), np.float32),
                     K=np.eye(3), world_to_camera=np.eye(4), name="v")
    assert view.occluder is None
