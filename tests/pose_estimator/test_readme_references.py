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
    text = text[: text.index(marker)] if marker in text else text
    # The per-phase documents drift exactly like the README does, so they are
    # held to the same standard rather than being a place names can hide.
    for document in sorted((REPO / "docs").glob("*.md")) if (REPO / "docs").is_dir() else []:
        text += "\n" + document.read_text()
    return text


README = _readme_without_history()


def test_every_pose_command_exists():
    """`pose-foo` in the README has to be a real console script."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    # Extras groups are named `pose-...` too (`pose-all`), and are installed
    # rather than run, so they count as declared for this purpose.
    declared = set(project["scripts"]) | set(project.get("optional-dependencies", {}))

    mentioned = set(re.findall(r"\bpose-[a-z0-9-]+", README))
    # Hyphenated CLI flags of the exporters are not console scripts.
    mentioned -= {"pose-enc"}
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
    sources = list((REPO / "src" / "pose_estimator").rglob("*.py"))
    # The exporters in scripts/ are this project's code too, and the P3 docs
    # name their flags.
    sources += list((REPO / "scripts").glob("*.py"))
    for path in sources:
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
    # Symbols belonging to the upstream projects we drive. Naming them is the
    # point -- "VGGT's no-BA path" is vague where the function name is not --
    # but this repository cannot be expected to define them.
    upstream = {"batch_np_matrix_to_pycolmap_wo_track", "export_predictions_to_colmap",
                "apply_confidence_mask", "use_multiview_confidence", "mask_edges",
                "load_and_preprocess_images", "encoding_to_camera", "confidence_percentile",
                "memory_efficient_inference", "rename_colmap_recons_and_rescale_camera"}
    known_other |= upstream
    suspects = {m for m in mentioned if m not in known_other and not m.endswith("_json")}

    missing = sorted(s for s in suspects if s not in written)
    assert not missing, (
        f"README names QC checks that no phase writes: {missing}. "
        f"Known checks: {sorted(written)}")
