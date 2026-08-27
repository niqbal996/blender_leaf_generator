"""Resolving the two feature banks, and saying so when one cannot be found.

There are two, they are not interchangeable, and the README devotes a table
to keeping them apart:

| flag | file | feeds |
|---|---|---|
| `--prompt-bank` | `p2/prompt_bank.npz` | P2: which object is the plant |
| `--seed-bank`   | `p4c/seed_bank.npz`  | P4c: which parts are leaf/stem/root |

Both are `.npz` files that live at a fixed place inside a specimen's run
directory, which makes the workdir a reasonable thing to point at -- and
pointing at it is what everyone does, because that is the path already on
the command line from the run that produced the bank. Left unresolved, that
mistake surfaces as `IsADirectoryError` from inside `numpy.load`, after the
DINO weights have already been loaded, naming neither the flag nor the
convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

# flag -> where that bank lives inside a run directory.
BANK_LOCATION = {
    "--seed-bank": Path("p4c") / "seed_bank.npz",
    "--prompt-bank": Path("p2") / "prompt_bank.npz",
}


def resolve_bank(path: Union[str, Path], flag: str) -> Path:
    """The `.npz` for `flag`, accepting either the file or its run directory.

    A directory is resolved to the conventional file inside it, and the
    substitution is printed rather than done silently -- pointing at the
    wrong specimen's workdir is a mistake this would otherwise hide.

    Raises SystemExit with the location it looked in, because every caller
    is a CLI and a traceback here tells the reader nothing they can act on.
    """
    path = Path(path)
    relative = BANK_LOCATION[flag]

    if path.is_dir():
        candidate = path / relative
        if candidate.is_file():
            print(f"  {flag} {path} is a run directory -- using {candidate}")
            return candidate
        raise SystemExit(
            f"{flag} {path} is a directory and has no {relative} inside it.\n"
            f"  {flag} wants the .npz file itself, e.g. <workdir>/{relative}\n"
            f"  Nothing there means that specimen never wrote one: "
            f"{'pose-pick-seeds' if flag == '--seed-bank' else 'pose-pick-prompts'} "
            "writes it, and needs the DINO weights (--hf-token) to do so.")

    if not path.exists():
        raise SystemExit(
            f"{flag} {path} does not exist.\n"
            f"  Expected the .npz written by "
            f"{'pose-pick-seeds' if flag == '--seed-bank' else 'pose-pick-prompts'}, "
            f"normally at <workdir>/{relative}")

    # A real file, but is it the *other* bank? They are both .npz and both
    # hold `vectors`/`labels`, so the wrong one loads cleanly and then
    # classifies leaves as "plant" or seeds SAM2 with "stem".
    other = next((f for f, rel in BANK_LOCATION.items()
                  if f != flag and path.name == rel.name), None)
    if other:
        raise SystemExit(
            f"{flag} was given {path.name}, which is the {other} bank.\n"
            f"  {flag} wants {relative.name}. They hold different vocabularies "
            "and are not interchangeable -- see the README's bank table.")
    return path
