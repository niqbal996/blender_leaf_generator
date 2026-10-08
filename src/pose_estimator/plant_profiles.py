"""Per-plant rules, chosen from the specimen's folder name.

A specimen lives at `<session>/plant_data/<species>_<n>/plant`, and the
species decides things no single rule gets right for every plant: whether
there is a stem at all, whether the leaves sit on side branches, and how a
petiole is found. One generic rule set was tried and is what broke on
vogelmeere (chickweed): the straight-petiole rule tuned on gaensefuss sent
45 of 56 petioles across open air to the one stem it knew about, because
chickweed's leaves hang off side branches.

    from pose_estimator.plant_profiles import profile_for
    profile, folder = profile_for(workdir)       # folder: what it was matched on

Folder names are matched loosely -- case, accents, a trailing `_1`/`_x1` and
small typos are ignored -- so `vogelemeere_x`, `Vogelmeere_21` and
`gänsefuß_3` all resolve. A folder that matches nothing gets `DEFAULT`, which
is the behaviour every plant had before profiles existed. An explicit
`--architecture` still wins over the profile.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

CAULESCENT = "caulescent"
ROSETTE = "rosette"


def normalise_architecture(name: Optional[str]) -> Optional[str]:
    """`upright` is a plainer name for `caulescent`: the same plant, the same code.

    They were once two code paths -- `upright` seeded leaf depth from the top
    of the stem only -- which on a branching plant reached 9.5% of the leaf
    tissue (vogelmeere, 2026-10-06). There is now one.
    """
    if name is None:
        return None
    name = str(name).strip().lower()
    return CAULESCENT if name == "upright" else name


@dataclass(frozen=True)
class PlantProfile:
    name: str
    # P5/P5x/skeleton_2d: "caulescent" (leaves on a stem) or "rosette" (leaves
    # meet at a crown at ground level, no stem).
    architecture: str
    # How skeleton_2d places each petiole:
    #   "stalk_2d"   traced in the photos: a walk along the stem mask from the
    #                blade base to the stem tree, fitted in 3D like a midrib
    #                (petioles_2d) -- leaves on side branches (vogelmeere)
    #   "stem_tree"  a straight line to the nearest axis of the stem cloud's
    #                branched skeleton, or its stalk where one ends at the blade
    #   "leaf_axis"  extend the tip->base axis until it meets the one stem
    #                (gaensefuss, agreed 2026-10-05)
    #   "crown"      rosettes: no petiole at all, each leaf runs from the crown to
    #                its tip (skeleton_2d decides this from the architecture)
    petiole_rule: str
    # How skeleton_2d tells a 2D leaf mask's base from its tip:
    #   "stalk"         the base is where the trimmed stalk meets the blade, the
    #                   tip the blade point farthest from it. For broad blades,
    #                   whose longest path runs corner to corner (vogelmeere).
    #   "stem_contact"  the base is the end touching the stem mask. Needed on a
    #                   bushy plant, where overlapping blades make shortcuts
    #                   through the plant mask and the foot rule flips.
    #   "foot"          the tip is the end farther from the plant's foot,
    #                   walking through the plant mask (gaensefuss).
    base_rule: str
    note: str = ""


PROFILES = {
    "vogelmeere": PlantProfile(
        "vogelmeere", CAULESCENT, petiole_rule="stalk_2d", base_rule="stalk",
        note="chickweed: branched shoots, opposite leaves, long petioles low down"),
    "gaensefuss": PlantProfile(
        "gaensefuss", CAULESCENT, petiole_rule="leaf_axis", base_rule="foot",
        note="goosefoot: one main stem with leaves along it"),
    "sugarbeet": PlantProfile(
        "sugarbeet", ROSETTE, petiole_rule="crown", base_rule="foot",
        note="rosette: leaves meet at a crown, no stem"),
    "thistle": PlantProfile(
        "thistle", ROSETTE, petiole_rule="crown", base_rule="foot",
        note="rosette: leaves meet at a crown, no stem"),
}

# Other names the same species turns up under.
ALIASES = {
    "vogelmiere": "vogelmeere", "chickweed": "vogelmeere", "stellaria": "vogelmeere",
    "gaensefuss": "gaensefuss", "goosefoot": "gaensefuss", "chenopodium": "gaensefuss",
    "zuckerruebe": "sugarbeet", "beet": "sugarbeet",
    "distel": "thistle",
}

# The rules every plant had before profiles: a stem, gaensefuss's petiole rule.
DEFAULT = PlantProfile("default", CAULESCENT, petiole_rule="leaf_axis", base_rule="foot",
                       note="no profile matched the folder name")

# `_1`, `_21`, `_x`, `_x1`, `3` at the end of a folder name number the specimen.
_SPECIMEN_SUFFIX = re.compile(r"(?:[_\-\s]+(?:x\d*|\d+)|\d+)+$")


def _key(text: str) -> str:
    text = _SPECIMEN_SUFFIX.sub("", text.strip().casefold())
    text = text.replace("ß", "ss").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if c.isalpha())


def match(folder: str) -> Optional[PlantProfile]:
    """The profile one folder name names, or None."""
    key = _key(folder)
    if not key:
        return None
    names = {**{k: k for k in PROFILES}, **ALIASES}
    if key in names:
        return PROFILES[names[key]]
    close = difflib.get_close_matches(key, list(names), n=1, cutoff=0.85)
    return PROFILES[names[close[0]]] if close else None


def profile_for(workdir) -> Tuple[PlantProfile, Optional[str]]:
    """(profile, the folder it was matched on) -- DEFAULT and None when nothing matches.

    The innermost folder that names a species wins, so `.../vogelmeere_21/plant`
    is vogelmeere whatever the session folder above it is called.
    """
    for part in reversed(Path(workdir).parts):
        found = match(part)
        if found is not None:
            return found, part
    return DEFAULT, None


def describe(profile: PlantProfile, folder: Optional[str]) -> str:
    """One line for a log: which rules were picked, and from what."""
    source = f"from folder '{folder}'" if folder else "no folder name matched a profile"
    return (f"plant profile {profile.name} ({source}): architecture {profile.architecture}, "
            f"petioles by {profile.petiole_rule}, base by {profile.base_rule}")
