"""Colour for the QC blocks every phase prints.

Six phases printed the same two lines with their own f-strings, so a change to
how a check reads had to be made six times and drifted between them. They call
`print_checks` now.

Colour is a reading aid, not information: a failure says [FAIL] whether or not
the terminal renders it red, and the JSON beside it is what anything
programmatic should read. `NO_COLOR` (the de-facto convention) turns it off.
`run_pipeline.sh` strips the escapes on their way into pipeline.log, so the
log stays greppable while the terminal stays legible.

Three states, because "did it pass" is not the only question worth answering
at a glance:

* red     -- a check failed, or a phase could not run
* green   -- a check passed
* orange  -- the phase ran but on weaker evidence than it wanted: a fallback
             was taken, an input was missing, a result is advisory. These
             never stop a run, which is exactly why they get lost in a wall of
             green without a colour of their own.
"""

from __future__ import annotations

import os
from typing import Mapping

_ENABLED = not os.environ.get("NO_COLOR")

RED = "\033[31m" if _ENABLED else ""
GREEN = "\033[32m" if _ENABLED else ""
ORANGE = "\033[33m" if _ENABLED else ""
BOLD = "\033[1m" if _ENABLED else ""
RESET = "\033[0m" if _ENABLED else ""


def red(text: str) -> str:
    return f"{RED}{text}{RESET}"


def green(text: str) -> str:
    return f"{GREEN}{text}{RESET}"


def orange(text: str) -> str:
    return f"{ORANGE}{text}{RESET}"


def warn(text: str) -> str:
    """A line the reader should notice but that stops nothing."""
    return f"{ORANGE}{text}{RESET}"


def print_checks(phase: str, report: Mapping) -> None:
    """The standard `<phase> checks (...)` block, one line per check.

    `phase` is the label as it should read, e.g. "P5" or "P4 hull".
    """
    passed = bool(report.get("all_passed"))
    headline = green("ALL PASSED") if passed else red("FAILURES PRESENT")
    print(f"\n  {BOLD}{phase} checks{RESET} ({headline}):")
    for name, check in report["checks"].items():
        tag = green("[PASS]") if check["pass"] else red("[FAIL]")
        print(f"    {tag} {name}: {check['detail']}")
