"""The plant profile is read off the specimen folder's name."""

import pytest

from pose_estimator.plant_profiles import DEFAULT, normalise_architecture, profile_for


@pytest.mark.parametrize("workdir, expected", [
    ("/netscratch/naeem/blender_assets/06-10-2026-Naeem/vogelmeere/plant", "vogelmeere"),
    ("/mnt/e/turn_table_datasets/vogelemeere_x/plant", "vogelmeere"),       # a typo
    ("/data/15-09-2026-Naeem/plant_data/Vogelmeere_21/plant", "vogelmeere"),
    ("/data/16-09-2026-Naeem/plant_data/gaensefuss_1/plant", "gaensefuss"),
    ("/data/x/Gänsefuß_2/plant", "gaensefuss"),
    ("/data/16-09-2026-Naeem/plant_data/sugarbeet_2/plant", "sugarbeet"),
    ("/data/29-09-2026-Naeem/plant_data/thistle3/plant", "thistle"),
    ("/data/29-09-2026-Naeem/plant_data/weed_x1/plant", "weed"),          # a weed of no named species
    ("/netscratch/naeem/blender_assets/06-10-2026-Naeem/weed_x/plant", "weed"),
])
def test_known_species_are_matched(workdir, expected):
    profile, folder = profile_for(workdir)
    assert profile.name == expected, (workdir, folder)


@pytest.mark.parametrize("workdir", ["/data/plant_data/mais_03/plant", "runs/plant_9",
                                     "/data/06-10-2026-Naeem/gf_1/plant"])
def test_unknown_names_keep_the_old_rules(workdir):
    assert profile_for(workdir) == (DEFAULT, None)


def test_rosettes_and_stemmed_plants_get_their_architecture():
    assert profile_for("/d/sugarbeet_1/plant")[0].architecture == "rosette"
    assert profile_for("/d/vogelmeere_1/plant")[0].architecture == "caulescent"
    assert profile_for("/d/weed_x/plant")[0].architecture == "rosette"     # the user, 2026-10-14
    assert profile_for("/d/vogelmeere_1/branch_3/plant")[0].name == "vogelmeere"


def test_rosettes_fit_a_blade_from_the_base_and_chickweed_keeps_its_stalk():
    for name in ("sugarbeet_1", "thistle3", "weed_x"):
        assert profile_for(f"/d/{name}/plant")[0].leaf_shape == "from_base"
    assert profile_for("/d/vogelmeere_1/plant")[0].leaf_shape == "stalked"
    assert DEFAULT.leaf_shape == "stalked"


def test_upright_is_caulescent():
    assert normalise_architecture("upright") == "caulescent"
    assert normalise_architecture("Upright") == "caulescent"
    assert normalise_architecture("rosette") == "rosette"
    assert normalise_architecture(None) is None
