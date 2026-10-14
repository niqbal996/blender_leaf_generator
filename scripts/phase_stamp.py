#!/usr/bin/env python
"""Is a phase's output on disk built by the code in this checkout? A stamp per phase says.

    python scripts/phase_stamp.py check <workdir> <phase> --code <entry>... [--after <phase>...] \
        [--args "<what else decides the result>"] --outputs <path>...
    python scripts/phase_stamp.py write <workdir> <phase> (same options)
    python scripts/phase_stamp.py show  <workdir>

run_pipeline.sh asks `check` before each phase and keeps the phase's output
when it is current, and calls `write` after the phase ran. A stamp
(<workdir>/stamps/<phase>.json) records

  code    the content hash of every repo file the phase's entry points import,
          followed through imports (`--code`: a module like
          pose_estimator.cli.mvs, or a script path), found by reading the files
          (ast), so a helper three imports deep counts as much as the entry;
  after   the stamps of the phases it reads (`--after`), so a phase that re-ran
          makes everything built on it stale in turn;
  args    the settings that decide the result (`--args`).

`check` says current only when the stamp exists, the outputs exist, and all
three match what this checkout would stamp now; otherwise it prints why and
exits 1. Stamp-free outputs (made before stamps existed, 2026-10-13) are
adopted once if every output is newer than (a) the last change to every file in
the phase's code -- the git commit time of a committed, unmodified file, else
the file's mtime -- and (b) every output of the phases it reads: the make rule.
It is conservative: a file committed after a run made with it reads as a
change, and the phase re-runs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOTS = (REPO / "src", REPO / "scripts")
STAMPS = None       # --stamps-dir; default <workdir>/stamps


def resolve(name: str) -> Path | None:
    """A module name or script path -> its file in this repo, or None (not ours)."""
    p = Path(name)
    if "/" in name or name.endswith((".py", ".sh")):     # a file: a script, or a shell script
        p = p if p.is_absolute() else REPO / p
        return p.resolve() if p.exists() else None
    parts = name.split(".")
    for root in ROOTS:
        for cand in (root.joinpath(*parts).with_suffix(".py"), root.joinpath(*parts, "__init__.py")):
            if cand.exists():
                return cand.resolve()
    return None


def imports_of(path: Path):
    """Repo files `path` imports (anywhere in it: functions import lazily here); none for a
    file that is not Python (a shell script counts by its own content only)."""
    if path.suffix != ".py":
        return set()
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    pkg = None
    try:
        rel = path.relative_to(REPO / "src")
        pkg = ".".join(rel.with_suffix("").parts[:-1] if rel.name != "__init__.py" else rel.parts[:-1])
    except ValueError:
        pass
    found = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level and pkg is not None:
                up = pkg.split(".")[: len(pkg.split(".")) - (node.level - 1)] if pkg else []
                base = ".".join([*up, base] if base else up)
            names = [base] + [f"{base}.{a.name}" for a in node.names]
        for n in names:
            if not n:
                continue
            # the module itself and every package on its way (their __init__ runs too)
            bits = n.split(".")
            for i in range(1, len(bits) + 1):
                f = resolve(".".join(bits[:i]))
                if f is not None and f != path:
                    found.add(f)
    return found


def closure(entries):
    todo = []
    for e in entries:
        f = resolve(e)
        if f is None:
            raise SystemExit(f"phase_stamp: no repo file for code entry '{e}'")
        todo.append(f)
    seen = set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        todo.extend(imports_of(f) - seen)
    return sorted(seen)


def sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def git_state():
    """(dirty files, {file: last commit time}) of this checkout, or None without git."""
    try:
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout
        dirty = {(REPO / line[3:].strip()).resolve() for line in dirty.splitlines() if line[3:].strip()}
        log = subprocess.run(["git", "-C", str(REPO), "log", "--format=@%ct", "--name-only"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    last, when = {}, None
    for line in log.splitlines():
        if line.startswith("@"):
            when = int(line[1:])
        elif line.strip():
            f = (REPO / line.strip()).resolve()
            last.setdefault(f, when)                 # newest commit first
    return dirty, last


def changed_at(path: Path) -> float:
    st = git_state()
    if st is not None:
        dirty, last = st
        if path not in dirty and path in last:
            return float(last[path])
    return path.stat().st_mtime


def stamp_path(workdir: Path, phase: str) -> Path:
    return (STAMPS or workdir / "stamps") / f"{phase}.json"


def load(workdir: Path, phase: str):
    p = stamp_path(workdir, phase)
    return json.loads(p.read_text()) if p.exists() else None


def digest(code: dict, after: dict, args: str) -> str:
    h = hashlib.sha1()
    h.update(json.dumps([code, after, args], sort_keys=True).encode())
    return h.hexdigest()


def now(workdir, a):
    files = closure(a.code)
    code = {str(f.relative_to(REPO)): sha(f) for f in files}
    after = {}
    for ph in a.after:
        s = load(workdir, ph)
        after[ph] = s["digest"] if s else None
    return files, code, after


def outputs_ok(workdir, a):
    missing = [o for o in a.outputs if not (workdir / o).exists()]
    return missing


def cmd_check(workdir: Path, a) -> int:
    rc = check(workdir, a)
    if rc and a.simulate:
        # --dry-run's plan: this phase will be rebuilt, so what reads it is stale in turn
        p = stamp_path(workdir, a.phase)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"phase": a.phase, "digest": f"rebuild-{time.time()}", "outputs": a.outputs}))
    return rc


def check(workdir: Path, a) -> int:
    missing = outputs_ok(workdir, a)
    if missing:
        print(f"stale: no {missing[0]}")
        return 1
    files, code, after = now(workdir, a)
    s = load(workdir, a.phase)
    if s is not None:
        if s.get("digest") == digest(code, after, a.args):
            print(f"current: built {s.get('at')} by this code" + (" (adopted)" if s.get("adopted") else ""))
            return 0
        why = [f for f in sorted(set(code) | set(s.get("code", {}))) if code.get(f) != s.get("code", {}).get(f)]
        if why:
            print(f"stale: code changed since {s.get('at')}: {', '.join(why[:4])}"
                  + (f" (+{len(why) - 4} more)" if len(why) > 4 else ""))
        elif after != s.get("after"):
            print(f"stale: rebuilt since: {', '.join(p for p in after if after[p] != s.get('after', {}).get(p))}")
        else:
            print(f"stale: settings changed ({s.get('args')!r} -> {a.args!r})")
        return 1
    # no stamp: adopt by the make rule, or not
    if any(v is None for v in after.values()):
        print(f"stale: no stamp, and {', '.join(p for p, v in after.items() if v is None)} has none either")
        return 1
    built = min((workdir / o).stat().st_mtime for o in a.outputs)
    newest = max(files, key=changed_at)
    if changed_at(newest) >= built:
        print(f"stale: no stamp, and {newest.relative_to(REPO)} changed "
              f"{time.strftime('%m-%d %H:%M', time.localtime(changed_at(newest)))}, after the output "
              f"({time.strftime('%m-%d %H:%M', time.localtime(built))})")
        return 1
    for ph in a.after:
        s = load(workdir, ph)
        if str(s.get("digest", "")).startswith("rebuild-"):
            print(f"stale: {ph} will be rebuilt first")
            return 1
        inputs = [workdir / o for o in s.get("outputs", []) if (workdir / o).exists()]
        if inputs and max(i.stat().st_mtime for i in inputs) > built:
            print(f"stale: no stamp, and {ph}'s output is newer than this one")
            return 1
    write(workdir, a, files, code, after, adopted=f"output {time.strftime('%Y-%m-%d %H:%M', time.localtime(built))} "
          f"newer than its code and inputs")
    print(f"current: adopted (output {time.strftime('%m-%d %H:%M', time.localtime(built))} is newer than its "
          f"code, last changed {time.strftime('%m-%d %H:%M', time.localtime(changed_at(newest)))}, and its inputs)")
    return 0


def write(workdir, a, files=None, code=None, after=None, adopted=None):
    if code is None:
        files, code, after = now(workdir, a)
    st = git_state()
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    doc = {"phase": a.phase, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit,
           "dirty": bool(st and any(f in st[0] for f in files)), "digest": digest(code, after, a.args),
           "args": a.args, "after": after, "outputs": a.outputs, "code": code}
    if adopted:
        doc["adopted"] = adopted
    p = stamp_path(workdir, a.phase)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1))


def cmd_show(workdir: Path) -> int:
    d = workdir / "stamps"
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        s = json.loads(p.read_text())
        print(f"  {s['phase']:<6} {s['at']}  commit {s['commit']}{' +local edits' if s.get('dirty') else ''}"
              f"{'  (adopted: ' + s['adopted'] + ')' if s.get('adopted') else ''}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check", "write", "show"))
    ap.add_argument("workdir", type=Path)
    ap.add_argument("phase", nargs="?")
    ap.add_argument("--code", nargs="*", default=[])
    ap.add_argument("--after", nargs="*", default=[])
    ap.add_argument("--args", default="")
    ap.add_argument("--outputs", nargs="*", default=[])
    ap.add_argument("--stamps-dir", type=Path, help="default <workdir>/stamps")
    ap.add_argument("--simulate", action="store_true",
                    help="check: a stale phase leaves a 'will be rebuilt' stamp (use with a scratch --stamps-dir)")
    a = ap.parse_args()
    global STAMPS
    STAMPS = a.stamps_dir
    if a.cmd == "show":
        sys.exit(cmd_show(a.workdir))
    if not a.phase or not a.code:
        ap.error("check/write need a phase and --code")
    sys.exit(cmd_check(a.workdir, a) if a.cmd == "check" else (write(a.workdir, a) or 0))


if __name__ == "__main__":
    main()
