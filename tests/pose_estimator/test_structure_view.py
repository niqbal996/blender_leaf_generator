"""Reading a finished P5 run back for inspection.

The viewer must show what the pipeline actually wrote, not a re-derivation of
it -- a viewer that recomputed the instancing could disagree with the run it
is supposed to be checking. So the loader is pure I/O, and these tests pin
that it reconstructs the same arrays and fails loudly when a run is absent.
"""

import json

import numpy as np
import pytest

from pose_estimator.structure_viz import load_structure_view


def write_run(root, n=60, leaves=2, with_tips=True, with_depth=True):
    p5 = root / "p5"
    p5.mkdir(parents=True)
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(n, 3))
    ids = np.repeat(np.arange(leaves), n // leaves).astype(np.int64)
    np.save(p5 / "leaf_points_xyz.npy", pts)
    np.save(p5 / "leaf_points.npy", ids)
    if with_depth:
        np.save(p5 / "leaf_depth.npy", np.linspace(0, 1, n))
    if with_tips:
        np.savez(p5 / "tips.npz",
                 candidate=np.array([0, 1, 2, 30]), accepted=np.array([0, 30]),
                 group=np.array([0, 0, 0, 1]), depth=np.zeros(4))
    graph = {
        "plant_frame": {"origin": [0, 0, 0], "rotation": np.eye(3).tolist(),
                        "table_height": 0.0, "clamp_height_orbit_frame": 0.0},
        "stem_path_xyz": [[0, 0, 0], [0, 0, 0.5], [0, 0, 1.0]],
        "leaves": [{"id": i, "axis_xyz": [[0, 0, 0], [1, 0, 0]],
                    "attachment_xyz": [0, 0, 0], "tip_xyz": [1, 0, 0],
                    "num_points": n // leaves} for i in range(leaves)],
    }
    (p5 / "stem_graph.json").write_text(json.dumps(graph))
    return pts, ids


def test_loads_points_ids_and_tips(tmp_path):
    pts, ids = write_run(tmp_path)
    view = load_structure_view(tmp_path)

    assert np.allclose(view.leaf_points, pts)
    assert np.array_equal(view.leaf_ids, ids)
    assert view.num_leaves == 2
    assert view.accepted_tips.tolist() == [0, 30]
    assert view.candidate_tips.tolist() == [0, 1, 2, 30]
    assert len(view.stem_path) == 3
    assert len(view.axes) == 2


def test_arrays_stay_aligned(tmp_path):
    """Everything the viewer colours by is indexed by leaf point, so a length
    mismatch would silently colour the wrong points."""
    write_run(tmp_path)
    view = load_structure_view(tmp_path)
    assert len(view.leaf_ids) == len(view.leaf_points) == len(view.depth)
    assert view.accepted_tips.max() < len(view.leaf_points)
    assert view.candidate_tips.max() < len(view.leaf_points)


def test_accepted_tips_are_a_subset_of_candidates(tmp_path):
    write_run(tmp_path)
    view = load_structure_view(tmp_path)
    assert set(view.accepted_tips.tolist()) <= set(view.candidate_tips.tolist())


def test_missing_optional_artifacts_still_load(tmp_path):
    """An older run without tips.npz or leaf_depth.npy should still open, just
    with less to show."""
    write_run(tmp_path, with_tips=False, with_depth=False)
    view = load_structure_view(tmp_path)
    assert len(view.accepted_tips) == 0
    assert len(view.depth) == len(view.leaf_points)
    assert np.isnan(view.depth).all()


def test_missing_run_fails_loudly(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        load_structure_view(tmp_path)
    assert "pose-structure" in str(excinfo.value)


def test_stem_and_root_absent_without_a_labelled_cloud(tmp_path):
    """No p4b/p4c in this fixture, so those layers come back empty rather than
    raising -- the viewer just has nothing to draw for them."""
    write_run(tmp_path)
    view = load_structure_view(tmp_path)
    assert len(view.stem_points) == 0
    assert len(view.root_points) == 0
