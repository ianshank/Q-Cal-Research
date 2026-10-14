"""scripts/mutation_check.py: each listed defence, broken in a copy, fails its named tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.conftest import REPO_ROOT, write


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "mutation_check", REPO_ROOT / "scripts/mutation_check.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


mc = load()

ENTRY = """
[[mutant]]
id = "{id}"
file = "src/qcal/__init__.py"
finding = "f returns one"
old = "{old}"
new = "{new}"
tests = ["tests/test_f.py"]
"""


def test_every_listed_defence_still_applies_to_the_repository() -> None:
    """The file cannot rot: each ``old`` text occurs once and each named test file exists."""
    mutants = mc.load(REPO_ROOT / mc.DEFENCES)
    assert mutants
    assert mc.applicability_problems(REPO_ROOT, mutants) == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "at least one"),
        ('[[mutant]]\nid = "a"\n', "lacks"),
        (ENTRY.format(id="a", old="x", new="x"), "changes nothing"),
        (ENTRY.format(id="a", old="x", new="y") + ENTRY.format(id="a", old="x", new="z"), "dupl"),
        (ENTRY.format(id="a", old="x", new="y").replace('["tests/test_f.py"]', "[]"), "list"),
        ("[[mutant", "cannot read"),
    ],
)
def test_malformed_defence_files_are_refused(tmp_path: Path, text: str, message: str) -> None:
    path = write(tmp_path, "d.toml", text)
    with pytest.raises(mc.DefenceError, match=message):
        mc.load(path)


def test_an_entry_whose_text_moved_is_reported(tmp_path: Path) -> None:
    write(tmp_path, "src/qcal/__init__.py", "x = 1\nx = 1\n")
    (mutant,) = mc.load(write(tmp_path, "d.toml", ENTRY.format(id="a", old="x = 1", new="y")))
    assert mc.applicability_problems(tmp_path, [mutant]) == [
        "a: 'old' must occur exactly once in src/qcal/__init__.py (found 2 times)",
        "a: test file tests/test_f.py does not exist",
    ]


@pytest.mark.parametrize(
    ("returncode", "failed", "verdict"),
    [
        (0, [], "SURVIVED"),
        (1, ["tests/test_f.py::test_one"], "KILLED"),
        (1, ["tests/test_f/TestF.py::test_one"], "KILLED"),  # a test inside a class
        (1, ["tests/test_other.py::test_one"], "ERROR"),  # some other test broke
        (1, [], "ERROR"),
        (2, ["tests/test_f.py::test_one"], "ERROR"),  # interrupted or a collection error
        (5, [], "ERROR"),  # no tests ran
        (-1, [], "ERROR"),  # timed out
    ],
)
def test_only_a_named_failing_test_counts_as_a_kill(
    returncode: int, failed: list[str], verdict: str
) -> None:
    assert mc.classify(returncode, failed, ["tests/test_f.py"]) == verdict


def test_failing_tests_are_read_from_the_junit_report(tmp_path: Path) -> None:
    report = write(
        tmp_path,
        "j.xml",
        '<testsuites><testsuite><testcase classname="tests.test_f" name="test_ok"/>'
        '<testcase classname="tests.test_f" name="test_bad"><failure/></testcase>'
        '<testcase classname="tests.test_f.TestF" name="test_err"><error/></testcase>'
        "</testsuite></testsuites>",
    )
    assert mc.failing_tests(report) == [
        "tests/test_f.py::test_bad",
        "tests/test_f/TestF.py::test_err",
    ]
    assert mc.failing_tests(tmp_path / "absent.xml") == []


def _project(tmp_path: Path, entries: str) -> Path:
    root = tmp_path / "project"
    write(root, "src/qcal/__init__.py", "def f() -> int:\n    return 1  # one\n")
    write(root, "src/qcal_lab/__init__.py", "")
    write(root, "tests/test_f.py", "from qcal import f\n\n\ndef test_one():\n    assert f() == 1\n")
    write(root, "tests/mutation/defences.toml", entries)
    for args in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


@pytest.mark.integration
def test_a_tested_defence_is_killed_and_an_untested_one_survives(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        ENTRY.format(id="tested", old="return 1", new="return 2")
        + ENTRY.format(id="untested", old="# one", new="# two"),
    )
    before = (root / "src/qcal/__init__.py").read_bytes()
    assert mc.main(["--root", str(root), "--json"]) == 1
    outcomes = {o["id"]: o["verdict"] for o in json.loads(capsys.readouterr().out)}
    assert outcomes == {"tested": "KILLED", "untested": "SURVIVED"}
    assert (root / "src/qcal/__init__.py").read_bytes() == before  # mutants touch a copy only


@pytest.mark.integration
def test_a_defence_whose_tests_fail_unmutated_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, ENTRY.format(id="a", old="return 1", new="return 2"))
    write(root, "src/qcal/__init__.py", "def f() -> int:\n    return 3  # one return 1\n")
    assert mc.main(["--root", str(root), "--only", "a"]) == 1
    assert capsys.readouterr().out.startswith("ERROR")


def test_listing_and_unknown_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _project(tmp_path, ENTRY.format(id="a", old="return 1", new="return 2"))
    assert mc.main(["--root", str(root), "--list"]) == 0
    assert "a" in capsys.readouterr().out
    assert mc.main(["--root", str(root), "--only", "zz"]) == 2
    write(root, "tests/mutation/defences.toml", "[[mutant")
    assert mc.main(["--root", str(root)]) == 2
    write(root, "tests/mutation/defences.toml", ENTRY.format(id="a", old="absent", new="b"))
    assert mc.main(["--root", str(root)]) == 2
