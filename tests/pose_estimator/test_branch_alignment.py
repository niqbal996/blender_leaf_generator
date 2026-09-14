"""Turning the comparison branches to face the same way.

P3 fixes the plant's axis but not where zero degrees sits on it, so the same
thistle comes out of each geometry branch spun by an arbitrary amount --
measured on thistle3, VGGT-Omega sat 160 degrees off COLMAP and MapAnything
37. Drawn side by side they read as three different plants, and the point of
the side-by-side view is to compare shapes.

The functions under test are pure geometry, so they are lifted out of the
Blender script and run without it -- importing that module needs `bpy`.
"""

from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "blender_view_plant.py"


@pytest.fixture(scope="module")
def blender_geometry():
    """`leaf_azimuths` and `azimuth_turn`, without importing bpy."""
    source = SCRIPT.read_text()
    start = source.index("def leaf_azimuths")
    end = source.index("def build(workdir,")
    namespace = {"np": np}
    exec(compile(source[start:end], str(SCRIPT), "exec"), namespace)
    return namespace


def assert_turn_is(turn, expected_deg):
    """Compare two angles as angles.

    `(degrees(turn) + 50) % 360` looks like it works and does not: a turn that
    comes back as -50.000000000000014 lands on 359.99999999999994, which is
    the right answer and fails any tolerance around zero.
    """
    gap = np.degrees(np.angle(np.exp(1j * (turn - np.radians(expected_deg)))))
    assert abs(gap) < 1e-6, f"expected {expected_deg} deg, got {np.degrees(turn):.6f}"


def plant(bearings_deg, reach=1.0, crown=(0.0, 0.0, 0.0)):
    """A stem graph with one leaf per bearing, all the same length."""
    crown = np.asarray(crown, float)
    leaves = []
    for i, bearing in enumerate(bearings_deg):
        angle = np.radians(bearing)
        tip = crown + np.array([reach * np.cos(angle), reach * np.sin(angle), 0.1])
        leaves.append({"id": i, "tip_xyz": tip.tolist(), "num_points": 100})
    return {"crown_xyz": crown.tolist(), "leaves": leaves, "stem_path_xyz": [crown.tolist()]}


def test_a_branch_already_facing_the_right_way_is_left_alone(blender_geometry):
    reference = plant([0, 90, 180, 270])
    turn = blender_geometry["azimuth_turn"](reference, plant([0, 90, 180, 270]))
    assert_turn_is(turn, 0.0)


@pytest.mark.parametrize("spun", [37.0, 98.0, 160.0, 275.0])
def test_a_spun_branch_is_turned_back(blender_geometry, spun):
    bearings = [0, 70, 145, 250]
    reference = plant(bearings)
    turn = blender_geometry["azimuth_turn"](reference, plant([b + spun for b in bearings]))
    assert_turn_is(turn, -spun)


def test_the_whole_plant_outvotes_its_largest_leaf(blender_geometry):
    """The reason this votes rather than matching the biggest leaf.

    The branches disagree about how many leaves there are -- 8, 5 and 3 on
    thistle3 -- so the largest leaf in one need not exist in another. Here the
    reference's longest leaf was never reconstructed by the branch, and
    matching longest-to-longest would turn the plant to a bearing that nothing
    else supports.
    """
    # Irregular bearings on purpose: leaves an equal angle apart make a plant
    # look the same after a whole-turn's-worth of rotation, so any fixture with
    # evenly spaced leaves has several right answers and tests nothing.
    reference = plant([0, 55, 140, 265])
    reference["leaves"][0]["tip_xyz"] = [3.0, 0.0, 0.1]      # a long leaf at 0 deg

    # The branch has the other three, spun by 50 degrees, and not the long one.
    branch = plant([55 + 50, 140 + 50, 265 + 50])
    branch["leaves"][0]["tip_xyz"] = list(
        np.array(branch["leaves"][0]["tip_xyz"]) * 2.0)       # its own longest, at 105

    turn = blender_geometry["azimuth_turn"](reference, branch)
    assert_turn_is(turn, -50.0)


def test_near_vertical_leaves_do_not_get_a_vote(blender_geometry):
    """A leaf standing up the axis has its tip almost on it, so its bearing is
    a millimetre of noise -- and it must not be allowed to steer the fit."""
    reference = plant([0, 90, 180])
    branch = plant([40, 130, 220])
    branch["leaves"].append({"id": 9, "tip_xyz": [1e-4, -1e-4, 0.9], "num_points": 100})

    angles, reaches = blender_geometry["leaf_azimuths"](branch)
    assert reaches[-1] < 0.1 * reaches.max(), "the upright leaf must be the low-reach one"
    turn = blender_geometry["azimuth_turn"](reference, branch)
    assert_turn_is(turn, -40.0)


def test_a_branch_with_no_leaves_reports_no_turn(blender_geometry):
    """Rotating on no evidence would be worse than leaving it be."""
    assert blender_geometry["azimuth_turn"](plant([0, 90]), {"leaves": []}) is None
    assert blender_geometry["azimuth_turn"]({"leaves": []}, plant([0, 90])) is None


def test_the_real_branches_line_up(blender_geometry):
    """thistle3's own three, if they are on this machine."""
    import json

    base = Path("/mnt/e/Camera_rig_data/turn_table_datasets/thistle3/plant_multipass")
    paths = {"colmap": base / "p5/stem_graph.json",
             "vggt_omega": base / "p5/experiments/vggt_omega/stem_graph.json"}
    if not all(p.exists() for p in paths.values()):
        pytest.skip("thistle3 is not on this machine")

    graphs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    turn = blender_geometry["azimuth_turn"](graphs["colmap"], graphs["vggt_omega"])
    assert turn is not None

    reference, _ = blender_geometry["leaf_azimuths"](graphs["colmap"])
    own, reach = blender_geometry["leaf_azimuths"](graphs["vggt_omega"])
    turned = own[reach > 0.1 * reach.max()] + turn
    gap = np.abs(np.angle(np.exp(1j * (turned[:, None] - reference[None, :])))).min(axis=1)
    matched = int((np.degrees(gap) < 25).sum())
    assert matched >= 3, f"only {matched} of {len(turned)} leaves found a partner"
