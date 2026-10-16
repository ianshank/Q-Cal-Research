"""scripts/traceability.py: integrity rules mapped to the tests that enforce them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from tests.conftest import REPO_ROOT, write


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "traceability", REPO_ROOT / "scripts/traceability.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trace = load()


def test_every_integrity_rule_has_a_test() -> None:
    assert trace.main(["--check", "--root", str(REPO_ROOT)]) == 0


def test_rules_are_read_from_the_documents() -> None:
    rules = trace.rules(REPO_ROOT)
    assert {f"C{i}" for i in range(8)} <= set(rules)
    assert "A1" in rules
    assert rules["C3"].startswith("Never tune on test splits")


def _tree(tmp_path: Path, test_body: str) -> Path:
    write(
        tmp_path,
        "CLAUDE.md",
        "## Integrity\n0. Rule zero.\n1. Rule one.\n## Other\n2. not a rule\n",
    )
    write(tmp_path, "AGENTS.md", "# A\n- First bullet.\n  continued\n")
    write(tmp_path, "tests/unit/test_x.py", test_body)
    return tmp_path


def test_marks_on_modules_classes_and_functions_are_found(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        """
        import pytest
        pytestmark = [pytest.mark.unit, pytest.mark.rule("C0")]

        @pytest.mark.rule("A1")
        def test_a(): ...

        def helper(): ...

        @pytest.mark.rule("C1")
        class TestGroup:
            def test_b(self): ...
        """,
    )
    assert trace.rules(root) == {"C0": "Rule zero.", "C1": "Rule one.", "A1": "First bullet."}
    assert trace.tagged_tests(root) == {
        "A1": ["tests/unit/test_x.py::test_a"],
        "C0": ["tests/unit/test_x.py::TestGroup::test_b", "tests/unit/test_x.py::test_a"],
        "C1": ["tests/unit/test_x.py::TestGroup::test_b"],
    }
    assert trace.main(["--check", "--root", str(root)]) == 0


def test_check_reports_untested_unknown_and_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _tree(tmp_path, 'import pytest\n\n@pytest.mark.rule("C9")\ndef test_a(): ...\n')
    monkeypatch.setattr(trace, "KNOWN_GAPS", {"C9": ("x", "y"), "C1": ("Ian", "why")})
    assert trace.main(["--check", "--json", "--root", str(root)]) == 1
    problems = json.loads(capsys.readouterr().out)["problems"]
    assert "unknown rule id 'C9' in tests/unit/test_x.py::test_a" in problems
    assert any(p.startswith("C0 has no test") for p in problems)
    assert any(p.startswith("A1 has no test") for p in problems)
    assert "KNOWN_GAPS names unknown rule 'C9'" in problems
    assert not any(p.startswith("C1 has no test") for p in problems)  # listed as a gap


def test_text_output_lists_rules_and_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _tree(tmp_path, 'import pytest\n\n@pytest.mark.rule("C0")\ndef test_a(): ...\n')
    monkeypatch.setattr(trace, "KNOWN_GAPS", {"C0": ("Ian", "why")})
    assert trace.main(["--root", str(root)]) == 0  # without --check it only reports
    out = capsys.readouterr().out
    assert "[gap: Ian]" in out
    assert "C0 is listed in KNOWN_GAPS but now has tests" in out
