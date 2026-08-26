"""The README must not name commands or QC checks that do not exist.

Documentation drifts silently: a renamed check or a command that was never
written reads perfectly and only fails when someone follows the steps. Both
have happened here -- the walkthrough referenced `camera_path_is_circular`,
which no phase has ever written.
"""

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _readme_without_history() -> str:
    """The README minus the section that documents deleted commands.

    "Removed: the older single-shot prototype" names `pose-estimate-skeleton`
    and friends on purpose -- recording why they went is the point of it.
    """
    text = (REPO / "README.md").read_text()
    marker = "### Removed:"
    return text[: text.index(marker)] if marker in text else text


README = _readme_without_history()


def test_every_pose_command_exists():
    """`pose-foo` in the README has to be a real console script."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    # Extras groups are named `pose-...` too (`pose-all`), and are installed
    # rather than run, so they count as declared for this purpose.
    declared = set(project["scripts"]) | set(project.get("optional-dependencies", {}))

    mentioned = set(re.findall(r"\bpose-[a-z0-9-]+", README))
    # `pose-estimator` is the env/package name, not a command.
    mentioned -= {"pose-estimator"}
    missing = sorted(mentioned - declared)
    assert not missing, f"README names commands that do not exist: {missing}"


def test_every_qc_check_exists():
    """A snake_case name in the README has to be one the code actually writes.

    Checks and report fields both count -- `base_spread_fraction_of_extent`
    is evidence in p5/instancing.json rather than a pass/fail check, and
    documenting it is correct. What this catches is a name that exists
    nowhere, which has happened: `camera_path_is_circular` was invented.
    """
    written = set()
    for path in (REPO / "src" / "pose_estimator").rglob("*.py"):
        text = path.read_text()
        written |= set(re.findall(r'checks\["([a-z_]+)"\]', text))
        # QC reports built as one dict literal rather than by assignment.
        for block in re.findall(r"checks = \{(.*?)\n    \}", text, re.S):
            written |= set(re.findall(r'"([a-z_]+)":\s*\{', block))
        # Any quoted snake_case key the code writes into a report.
        written |= set(re.findall(r'"([a-z]+(?:_[a-z]+)+)":', text))
        # ...and any snake_case name the code defines: parameters and
        # assignments count too, since the README documents flags and tunables
        # as well as report fields.
        written |= set(re.findall(r'\b([a-z]+(?:_[a-z]+)+)\s*[:=]', text))

    assert written, "found no QC check names in the source at all"

    # Backticked identifiers in the README that look like check names.
    mentioned = set(re.findall(r"`([a-z]+(?:_[a-z]+){2,})`", README))
    # Only judge names that are plausibly checks, not any snake_case symbol.
    known_other = {"leaf_points", "sources_json", "min_branch_fraction"}
    suspects = {m for m in mentioned if m not in known_other and not m.endswith("_json")}

    missing = sorted(s for s in suspects if s not in written)
    assert not missing, (
        f"README names QC checks that no phase writes: {missing}. "
        f"Known checks: {sorted(written)}")
