#!/usr/bin/env python3
"""Prove each listed defence is tested: break it in a copy of the repository and expect the
named tests to fail.

Every entry of ``tests/mutation/defences.toml`` names one source edit (``old`` -> ``new``, which
must match exactly once) and the test nodes (``file::test``, a class, or one parametrization)
that should catch it. A whole file is not accepted: a broad test in the same file, such as a
drift check over generated output, would then take credit for every mutant. For each entry the
script:

1. copies the working tree's tracked and new unignored files into a temporary directory
   (uncommitted edits included; the repository itself is never written);
2. runs the named tests unmutated there, which must pass, else the entry is an ERROR;
3. applies the edit and runs them again with that copy's ``src`` first on ``sys.path``
   (checked: ``qcal.__file__`` must lie inside the copy).

A mutant counts as KILLED only when pytest exits 1 and the JUnit report shows one of the named
nodes failing. Exit 0 is SURVIVED. Anything else (a collection error, an import that
breaks, a timeout) is ERROR, never a kill.

    python3 scripts/mutation_check.py               # every entry
    python3 scripts/mutation_check.py --only S2-ctl  # one or more ids
    python3 scripts/mutation_check.py --list
    python3 scripts/mutation_check.py --discover --only S2-ctl  # failing nodes of its files
    python3 scripts/mutation_check.py --json

Report-only: it is not part of ``make check``. Exit status 1 when any entry is not KILLED.
Standard library only.
"""

from __future__ import annotations

import argparse
import ast
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
DISCOVERED: Final = "DISCOVERED"
#: pytest's exit code for "tests ran and some failed"; every other non-zero code is an error.
TESTS_FAILED: Final = 1
TIMEOUT_S: Final = 900
REQUIRED: Final = ("id", "file", "old", "new", "tests", "finding")
NODE: Final = "::"
DEFINITIONS: Final = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


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
            raise DefenceError(f"{path}: mutant {entry['id']!r}: 'tests' must list test nodes")
        files = [t for t in tests if NODE not in t]
        if files:
            raise DefenceError(
                f"{path}: mutant {entry['id']!r}: name test nodes (file::test), not {files}"
            )
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
        for test in m.tests:
            path = root / test_file(test)
            if not path.is_file():
                problems.append(f"{m.id}: test file {test_file(test)} does not exist")
            elif not _defines(path.read_text("utf-8"), test):
                problems.append(f"{m.id}: {test} is not defined in its file")
    return problems


def _defines(source: str, node: str) -> bool:
    """Whether the test (or class) a node names is defined in ``source``: renames are caught
    here, by the unit test of this file, not later as an ERROR of the mutation run."""
    parts = [part.split("[", 1)[0] for part in node.split(NODE)[1:]]
    try:
        body: list[ast.stmt] = ast.parse(source).body
    except SyntaxError:
        return False
    for part in parts:
        found = next((n for n in body if isinstance(n, DEFINITIONS) and n.name == part), None)
        if found is None:
            return False
        body = found.body if isinstance(found, ast.ClassDef) else []
    return True


def test_file(node: str) -> str:
    return node.split(NODE, 1)[0]


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
    git += ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "-C", str(dest)]
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


def node_id(root: Path, classname: str, name: str) -> str:
    """The pytest node id of a JUnit test case: the longest classname prefix that is a file."""
    parts = classname.split(".")
    for i in range(len(parts), 0, -1):
        module = "/".join(parts[:i]) + ".py"
        if (root / module).is_file():
            return NODE.join([module, *parts[i:], name])
    return NODE.join(["/".join(parts) + ".py", name])


def failing_tests(junit: Path, root: Path) -> list[str]:
    """The node id of every failed test case in a JUnit report. An error (a broken fixture or
    setup) is not a failure: no assertion caught the mutant, so it never counts as a kill."""
    try:
        tree = ET.parse(junit)  # noqa: S314 - pytest's own report, written in our temp dir
    except (OSError, ET.ParseError):
        return []
    return [
        node_id(root, case.get("classname", ""), case.get("name", ""))
        for case in tree.iter("testcase")
        if case.find("failure") is not None
    ]


def named(failed: str, node: str) -> bool:
    """``failed`` is ``node``, one of its parametrizations, or a test in the class ``node``."""
    return failed == node or failed.startswith((node + "[", node + NODE))


def classify(returncode: int, failed: Sequence[str], tests: Sequence[str]) -> str:
    if returncode == 0:
        return SURVIVED
    hit = any(named(f, t) for f in failed for t in tests)
    return KILLED if returncode == TESTS_FAILED and hit else ERROR


def run_tests(
    copy: Path, python: str, tests: Sequence[str], junit: Path, *, first_failure: bool = True
) -> tuple[int, str]:
    cmd = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"]
    cmd += ["-x"] if first_failure else []
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


def check(
    root: Path, mutant: Mutant, python: str, files: Sequence[str], *, discover: bool = False
) -> Outcome:
    """Break the defence in a copy and run its named nodes; with ``discover``, run every test
    in the named files instead and report which nodes fail, for choosing the nodes to name."""
    tests = sorted({test_file(t) for t in mutant.tests}) if discover else list(mutant.tests)
    with tempfile.TemporaryDirectory(prefix=f"mutant-{mutant.id}-") as tmp:
        copy = Path(tmp) / "repo"
        copy_tree(root, files, copy)
        if not imports_from(copy, python):
            return Outcome(mutant.id, ERROR, "qcal does not import from the copy")
        code, out = run_tests(copy, python, tests, Path(tmp) / "baseline.xml")
        if code != 0:
            return Outcome(mutant.id, ERROR, f"the named tests fail unmutated:\n{out}")
        target = copy / mutant.file
        text = target.read_text("utf-8")
        if text.count(mutant.old) != 1:
            return Outcome(mutant.id, ERROR, f"'old' does not occur exactly once in {mutant.file}")
        target.write_text(text.replace(mutant.old, mutant.new, 1), "utf-8")
        junit = Path(tmp) / "mutant.xml"
        code, out = run_tests(copy, python, tests, junit, first_failure=not discover)
        failed = failing_tests(junit, copy)
        if discover:  # a collection error is not "nothing caught it"
            verdict = DISCOVERED if code in (0, TESTS_FAILED) else ERROR
            return Outcome(mutant.id, verdict, "\n".join(failed) or f"pytest exit {code}\n{out}")
        verdict = classify(code, failed, mutant.tests)
        detail = ", ".join(failed[:3]) if verdict == KILLED else f"pytest exit {code}\n{out}"
        return Outcome(mutant.id, verdict, detail)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--defences", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None, metavar="ID")
    parser.add_argument("--list", action="store_true", help="list the entries and exit")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="run every test in the named files under each mutant and print the failing nodes",
    )
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
    outcomes = [check(root, m, args.python, files, discover=args.discover) for m in mutants]
    if args.json:
        sys.stdout.write(json.dumps([asdict(o) for o in outcomes], indent=2) + "\n")
    elif args.discover:
        for o in outcomes:
            sys.stdout.write(f"{o.verdict:<10} {o.id}\n")
            sys.stdout.writelines(f"    {line}\n" for line in o.detail.splitlines())
    else:
        for o in outcomes:
            first = o.detail.splitlines()[0] if o.detail else ""
            sys.stdout.write(f"{o.verdict:<9} {o.id:<14} {first}\n")
    good = DISCOVERED if args.discover else KILLED
    return 0 if all(o.verdict == good for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
