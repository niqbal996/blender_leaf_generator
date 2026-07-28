from leaf_generator.discovery import find_leaf_groups


def test_find_leaf_groups_both_sides(tmp_path):
    for leaf_id in ("1", "2"):
        for side in ("oberseite", "unterseite"):
            for map_type in ("ALBEDO", "HEIGHT", "mask", "NORMAL_GL", "ROUGHNESS"):
                (tmp_path / f"{leaf_id}_{map_type}_{side}.png").touch()
    (tmp_path / "oberseite_log.json").touch()

    groups = find_leaf_groups(tmp_path)

    assert set(groups.keys()) == {"1", "2"}
    leaf = groups["1"]
    assert leaf.has_both_sides
    assert leaf.primary_side == "oberseite"
    assert set(leaf.maps_for("oberseite").keys()) == {
        "albedo", "height", "mask", "normal", "roughness",
    }


def test_find_leaf_groups_oberseite_only_does_not_break(tmp_path):
    for map_type in ("ALBEDO", "HEIGHT", "mask", "NORMAL_GL", "ROUGHNESS"):
        (tmp_path / f"3_{map_type}_oberseite.png").touch()

    groups = find_leaf_groups(tmp_path)

    leaf = groups["3"]
    assert not leaf.has_both_sides
    assert leaf.primary_side == "oberseite"
    assert "unterseite" not in leaf.sides


def test_find_leaf_groups_sorts_numerically(tmp_path):
    for leaf_id in ("10", "2", "1"):
        (tmp_path / f"{leaf_id}_mask_oberseite.png").touch()

    groups = find_leaf_groups(tmp_path)

    assert list(groups.keys()) == ["1", "2", "10"]
