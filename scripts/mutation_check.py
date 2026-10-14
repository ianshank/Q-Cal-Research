#!/usr/bin/env python3
"""Prove each listed defence is tested: break it in a copy of the repository and expect the
named tests to fail.

Every entry of ``tests/mutation/defences.toml`` names one source edit (``old`` -> ``new``, which
must match exactly once) and the tests that should catch it. For each entry the script:

1. copies the working tree's tracked and new unignored files into a temporary directory
   (uncommitted edits included; the repository itself is never written);
2. runs the named tests unmutated there, which must pass, else the entry is an ERROR;
3. applies the edit and runs them again with that copy's ``src`` first on ``sys.path``
   (checked: ``qcal.__file__`` must lie inside the copy).

A mutant counts as KILLED only when pytest exits 1 and the JUnit report shows a failing test
from the named files. Exit 0 is SURVIVED. Anything else (a collection error, an import that
breaks, a timeout) is ERROR, never a kill.

    python3 scripts/mutation_check.py               # every entry
    python3 scripts/mutation_check.py --only S2-ctl  # one or more ids
    python3 scripts/mutation_check.py --list
    python3 scripts/mutation_check.py --json

Report-only: it is not part of ``make check``. Exit status 1 when any entry is not KILLED.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

DEFENCES: Final = Path("tests/mutation/defences.toml")
KILLED: Final = "KILLED"
SURVIVED: Final = "SURVIVED"
ERROR: Final = "ERROR"
#: pytest's exit code for "tests ran and some failed"; every other non-zero code is an error.
TESTS_FAILED: Final = 1
TIMEOUT_S: Final = 900
REQUIRED: Final = ("id", "file", "old", "new", "tests", "finding")


class DefenceError(ValueError):
    """``defences.toml`` is malformed or an entry no longer applies."""


@dataclass(frozen=True)
class Mutant:
    id: str
    file: str
    old: str
    new: str
    tests: tuple[str, ...]
    finding: str


@dataclass(frozen=True)
class Outcome:
    id: str
    verdict: str
    detail: str


def load(path: Path) -> list[Mutant]:
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DefenceError(f"cannot read {path}: {exc}") from exc
    entries = data.get("mutant", [])
    if not isinstance(entries, list) or not entries:
        raise DefenceError(f"{path}: needs at least one [[mutant]] entry")
    mutants: list[Mutant] = []
    for i, entry in enumerate(entries):
        missing = [k for k in REQUIRED if k not in entry]
        if missing:
            raise DefenceError(f"{path}: mutant {i} lacks {missing}")
        tests = entry["tests"]
        if not isinstance(tests, list) or not tests or not all(isinstance(t, str) for t in tests):
            raise DefenceError(f"{path}: mutant {entry['id']!r}: 'tests' must list test paths")
        if entry["old"] == entry["new"]:
            raise DefenceError(f"{path}: mutant {entry['id']!r} changes nothing")
        mutants.append(
            Mutant(
                str(entry["id"]),
                str(entry["file"]),
                str(entry["old"]),
                str(entry["new"]),
                tuple(tests),
                str(entry["finding"]),
            )
        )
    ids = [m.id for m in mutants]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise DefenceError(f"{path}: duplicate mutant ids {duplicates}")
    return mutants


def applicability_problems(root: Path, mutants: Sequence[Mutant]) -> list[str]:
    """Entries whose ``old`` text no longer occurs exactly once, or whose tests are gone."""
    problems: list[str] = []
    for m in mutants:
        source = root / m.file
        count = source.read_text("utf-8").count(m.old) if source.is_file() else -1
        if count != 1:
            found = "file missing" if count < 0 else f"found {count} times"
            problems.append(f"{m.id}: 'old' must occur exactly once in {m.file} ({found})")
        problems.extend(
            f"{m.id}: test file {test} does not exist"
            for test in m.tests
            if not (root / test.split("::", 1)[0]).is_file()
        )
    return problems


def tracked_files(root: Path) -> list[str]:
    """Tracked files plus new, unignored ones: the tree a commit of the work would hold."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def copy_tree(root: Path, files: Sequence[str], dest: Path) -> None:
    """Copy the tracked files and commit them, so tests that read git see a clean tree."""
    for name in files:
        source = root / name
        if not source.is_file():  # deleted in the working tree
            continue
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)
    git = ["git", "-c", "user.name=mutation", "-c", "user.email=mutation@localhost"]
    git += ["-c", "commit.gpgsign=false", "-C", str(dest)]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "copy"]):
        subprocess.run([*git, *args], capture_output=True, check=True)


def _environment(copy: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("QCAL")}
    env["PYTHONPATH"] = str(copy / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def imports_from(copy: Path, python: str) -> bool:
    """Whether ``qcal`` and ``qcal_lab`` resolve to the copy, not an installed package."""
    probe = "import qcal, qcal_lab; print(qcal.__file__); print(qcal_lab.__file__)"
    result = subprocess.run(
        [python, "-c", probe],
        check=False,
        cwd=copy,
        env=_environment(copy),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    inside = copy.resolve()
    return all(Path(line).resolve().is_relative_to(inside) for line in result.stdout.split())


def failing_tests(junit: Path) -> list[str]:
    """``file::name`` of every failed or errored test case in a JUnit report."""
    try:
        tree = ET.parse(junit)  # noqa: S314 - pytest's own report, written in our temp dir
    except (OSError, ET.ParseError):
        return []
    failed: list[str] = []
    for case in tree.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            module = case.get("classname", "").replace(".", "/")
            failed.append(f"{module}.py::{case.get('name', '')}")
    return failed


def classify(returncode: int, failed: Sequence[str], tests: Sequence[str]) -> str:
    if returncode == 0:
        return SURVIVED
    files = {t.split("::", 1)[0] for t in tests}
    named = any(f.split("::", 1)[0] in files or _within(f, files) for f in failed)
    return KILLED if returncode == TESTS_FAILED and named else ERROR


def _within(failed: str, files: set[str]) -> bool:
    """A failing test in a class (``a/b.py::C::t`` reported as ``a/b/C.py::t``)."""
    module = failed.split("::", 1)[0].removesuffix(".py")
    return any(module.startswith(f.removesuffix(".py") + "/") for f in files)


def run_tests(copy: Path, python: str, tests: Sequence[str], junit: Path) -> tuple[int, str]:
    cmd = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", f"--junitxml={junit}"]
    try:
        result = subprocess.run(
            [*cmd, *tests],
            check=False,
            cwd=copy,
            env=_environment(copy),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {TIMEOUT_S}s"
    return result.returncode, (result.stdout + result.stderr)[-1500:]


def check(root: Path, mutant: Mutant, python: str, files: Sequence[str]) -> Outcome:
    with tempfile.TemporaryDirectory(prefix=f"mutant-{mutant.id}-") as tmp:
        copy = Path(tmp) / "repo"
        copy_tree(root, files, copy)
        if not imports_from(copy, python):
            return Outcome(mutant.id, ERROR, "qcal does not import from the copy")
        code, out = run_tests(copy, python, mutant.tests, Path(tmp) / "baseline.xml")
        if code != 0:
            return Outcome(mutant.id, ERROR, f"the named tests fail unmutated:\n{out}")
        target = copy / mutant.file
        text = target.read_text("utf-8")
        if text.count(mutant.old) != 1:
            return Outcome(mutant.id, ERROR, f"'old' does not occur exactly once in {mutant.file}")
        target.write_text(text.replace(mutant.old, mutant.new, 1), "utf-8")
        junit = Path(tmp) / "mutant.xml"
        code, out = run_tests(copy, python, mutant.tests, junit)
        failed = failing_tests(junit)
        verdict = classify(code, failed, mutant.tests)
        detail = ", ".join(failed[:3]) if verdict == KILLED else f"pytest exit {code}\n{out}"
        return Outcome(mutant.id, verdict, detail)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--defences", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None, metavar="ID")
    parser.add_argument("--list", action="store_true", help="list the entries and exit")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        mutants = load(args.defences or root / DEFENCES)
    except DefenceError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if args.only:
        unknown = sorted(set(args.only) - {m.id for m in mutants})
        if unknown:
            sys.stderr.write(f"unknown mutant ids: {unknown}\n")
            return 2
        mutants = [m for m in mutants if m.id in args.only]
    if args.list:
        for m in mutants:
            sys.stdout.write(f"{m.id:<14} {m.file}: {m.finding}\n")
        return 0
    problems = applicability_problems(root, mutants)
    if problems:
        sys.stderr.write("\n".join(problems) + "\n")
        return 2
    files = tracked_files(root)
    outcomes = [check(root, m, args.python, files) for m in mutants]
    if args.json:
        sys.stdout.write(json.dumps([asdict(o) for o in outcomes], indent=2) + "\n")
    else:
        for o in outcomes:
            first = o.detail.splitlines()[0] if o.detail else ""
            sys.stdout.write(f"{o.verdict:<9} {o.id:<14} {first}\n")
    return 0 if all(o.verdict == KILLED for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
