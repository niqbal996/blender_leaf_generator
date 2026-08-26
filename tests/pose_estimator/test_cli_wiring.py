"""Every CLI flag has to reach the function that implements it.

A flag can be declared in argparse, read in the body, and never passed
between the two -- which imports cleanly, passes --help, and then dies with a
NameError several minutes into a run, after frame extraction. That happened
with --prompt-bank. This checks the whole surface rather than that one flag.
"""

import inspect
from importlib import import_module

import pytest

# module, the function main() delegates to, and dests that legitimately do not
# map to a parameter (usually inverted or parsed before the call).
CLIS = [
    ("pose_estimator.cli.segment", "run", {"no_roi", "plant_point", "holder_point", "video"}),
    ("pose_estimator.cli.pick_prompts", "run", {"no_auto", "no_bank"}),
    ("pose_estimator.cli.pick_seeds", "run", set()),
]


@pytest.mark.parametrize("module_name,func_name,exempt", CLIS)
def test_every_flag_reaches_run(module_name, func_name, exempt):
    module = import_module(module_name)
    parser_dests = _argparse_dests(module)
    assert parser_dests, f"{module_name}: found no arguments to check"

    accepted = set(inspect.signature(getattr(module, func_name)).parameters)
    forwarded = inspect.getsource(module.main)

    unwired = sorted(
        dest for dest in parser_dests - exempt - {"help"}
        if f"args.{dest}" not in forwarded
    )
    assert not unwired, (
        f"{module_name}: {unwired} are declared as flags but never passed to "
        f"{func_name}(). {func_name}() accepts {sorted(accepted)}.")


def _argparse_dests(module):
    """Flag names, read by running the module's own parser construction."""
    import argparse

    seen = set()
    real_add = argparse.ArgumentParser.add_argument

    def spy(self, *args, **kwargs):
        action = real_add(self, *args, **kwargs)
        if action.dest not in ("help",):
            seen.add(action.dest)
        return action

    argparse.ArgumentParser.add_argument = spy
    try:
        with pytest.raises(SystemExit):
            module.main(["--help"])
    finally:
        argparse.ArgumentParser.add_argument = real_add
    return seen


def test_prompt_bank_prompts_are_full_frame(monkeypatch):
    """Bank-located prompts must declare full-frame coordinates.

    They are read off the whole frame, but `Prompts` defaults to "crop". With
    the default, segmentation.py both re-reads them as crop-relative AND
    stops passing a seed point to the tracking crop, so the crop reverts to
    the colour prepass -- the exact rule the bank replaces. On thistle2 that
    turned a correct plant prompt into a point on the pliers.
    """
    import numpy as np
    from pathlib import Path
    from pose_estimator.cli import segment as seg

    monkeypatch.setattr(seg, "cv2", type("C", (), {"imread": staticmethod(lambda p: np.zeros((4, 4, 3), np.uint8))}))
    monkeypatch.setattr("pose_estimator.dino.DinoBackbone", lambda *a, **k: object())
    monkeypatch.setattr("pose_estimator.prompt_seeds.load_prompt_bank",
                        lambda path, model: (np.eye(4, 2, dtype=np.float32), ["plant", "holder"]))
    monkeypatch.setattr("pose_estimator.prompt_seeds.locate_prompts",
                        lambda *a, **k: {"plant": [(771, 607)], "holder": [(1731, 646)]})

    out = seg._locate_per_pass(Path("bank.npz"), {0: [Path("frame_0000.jpg")]},
                               "m", 896, "cpu")
    assert out[0].space == "full_frame", "the tracking crop silently reverts without this"
    assert out[0].plant == [(771, 607)]


def test_prompt_coverage_check_reaches_qc_json(tmp_path):
    """A check added after run_qc must be written to disk, not just printed.

    run_qc writes qc.json itself. The prompt-coverage check is added to the
    report afterwards, so without a second write it appeared on the console
    and was absent from the file every reviewer actually reads.
    """
    import json
    import numpy as np
    import cv2
    from pathlib import Path
    from pose_estimator.cli.segment import _check_prompts_are_covered
    from pose_estimator.segmentation import Prompts

    p2 = tmp_path / "p2"
    (p2 / "masks" / "plant").mkdir(parents=True)
    mask = np.zeros((40, 40), np.uint8)
    mask[10:20, 10:20] = 255
    cv2.imwrite(str(p2 / "masks" / "plant" / "frame_0000.png"), mask)

    report = {"checks": {}, "all_passed": True}
    clicked = {0: Prompts(plant=[(15, 15), (35, 35)], holder=[], space="full_frame")}
    _check_prompts_are_covered(report, clicked, None,
                               {0: [Path("frame_0000.jpg")]}, p2)

    check = report["checks"]["mask_covers_its_own_prompts"]
    assert check["pass"] is False, "a prompt outside the mask must fail"
    assert "1 of 2" in check["detail"]
    assert report["all_passed"] is False
