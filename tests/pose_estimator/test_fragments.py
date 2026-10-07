"""A strip of one leaf tracked under a second SAM3 id is folded back into it.

vogelmeere (2026-10-07): SAM3 handed leaf 6's blade from pass0_24 to pass0_64
partway through pass 0; a strip seen broadside only at the end went to 64,
became P5x leaf 39, and drew a second tip on leaf 6.
"""

import numpy as np

from pose_estimator.cli.leaf_instances import (FRAGMENT_CONTACT, absorb_fragments,
                                               fragment_contacts)


def _sheet(x0, x1, y0, y1, step=0.01, z=0.0):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], axis=1)


def test_a_strip_lying_on_a_blade_is_found_and_a_neighbour_is_not():
    blade = _sheet(0.0, 1.0, 0.0, 0.5)
    strip = _sheet(0.6, 0.8, 0.1, 0.3, z=0.004)        # on the blade's surface
    neighbour = _sheet(1.0, 1.3, 0.0, 0.3)             # beside it, touching one edge
    points = np.vstack([blade, strip, neighbour])
    labels = np.concatenate([np.full(len(blade), 0), np.full(len(strip), 1),
                             np.full(len(neighbour), 2)])
    contacts = fragment_contacts(points, labels, tolerance=0.008)
    assert contacts.get(1, (None,))[0] == 0 and contacts[1][1] >= FRAGMENT_CONTACT
    assert 2 not in contacts, "a leaf touching along one edge is not a fragment"


def test_sam3_decides_and_only_the_smaller_is_folded_in():
    labels = np.array([0] * 100 + [1] * 10 + [2] * 10)
    contacts = {1: (0, 1.0), 2: (0, 0.95)}
    votes = {(1, 0): [20, 0],          # SAM3: one object in every photo
             (2, 0): [5, 15]}          # SAM3: mostly two objects -- a real small leaf
    absorbed = absorb_fragments(labels, contacts, votes)
    assert absorbed == {1: 0}
    assert (labels == 1).sum() == 0 and (labels == 0).sum() == 110
    assert (labels == 2).sum() == 10, "the leaf SAM3 keeps apart stays a leaf"


def test_a_fragment_of_a_fragment_ends_in_the_leaf():
    labels = np.array([0] * 100 + [1] * 20 + [2] * 5)
    absorbed = absorb_fragments(labels, {2: (1, 1.0), 1: (0, 0.9)},
                                {(2, 1): [6, 0], (1, 0): [9, 1]})
    assert absorbed == {1: 0, 2: 0}
    assert set(np.unique(labels)) == {0}


def test_one_photo_is_not_enough():
    labels = np.array([0] * 50 + [1] * 5)
    assert absorb_fragments(labels, {1: (0, 1.0)}, {(1, 0): [1, 0]}) == {}
