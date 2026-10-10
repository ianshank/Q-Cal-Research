from __future__ import annotations

import re
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qcal.globs import compile_glob, first_match, glob_match, iter_files, literal_base, normalize


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("Makefile", "Makefile", True),
        ("sub/Makefile", "Makefile", False),
        ("sub/Makefile", "**/Makefile", True),
        ("Makefile", "**/Makefile", True),
        ("src/qcal/hooks/a.py", "src/qcal/hooks/**", True),
        ("src/qcal/hooks", "src/qcal/hooks/**", False),
        ("src/qcal/a.py", "src/qcal/*.py", True),
        ("src/qcal/x/a.py", "src/qcal/*.py", False),
        ("a/handwritten/b/c.py", "**/handwritten/**", True),
        ("handwritten/c.py", "**/handwritten/**", True),
        ("paper/sections/abstract.tex", "paper/sections/abstract*", True),
        ("file1.txt", "file?.txt", True),
        ("file12.txt", "file?.txt", False),
        ("a.py", "[ab].py", True),
        ("c.py", "[!ab].py", True),
        ("a.py", "[!ab].py", False),
        ("[x.py", "[x.py", True),
        ("EXPERIMENTS.YAML", "experiments.yaml", True),
        ("./README.md", "README.md", True),
        ("/README.md", "README.md", True),
        ("dir\\file.md", "dir/file.md", True),
        ("review/gemini-1.md", "review/gemini-*", True),
    ],
)
def test_glob_match(path: str, pattern: str, expected: bool) -> None:
    assert glob_match(path, pattern) is expected


def test_case_sensitivity_is_configurable() -> None:
    assert glob_match("A.md", "a.md", case_insensitive=True)
    assert not glob_match("A.md", "a.md", case_insensitive=False)


def test_first_match_returns_the_matching_pattern() -> None:
    assert first_match("src/a.py", ["*.md", "src/*.py"]) == "src/*.py"
    assert first_match("src/a.py", ["*.md"]) is None


def test_normalize_strips_relative_and_absolute_prefixes() -> None:
    assert normalize("././a/b") == "a/b"
    assert normalize("/a/b") == "a/b"


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("paper/**/*.tex", ("paper", False)),
        ("README.md", ("README.md", True)),
        ("**/x", ("", False)),
        ("a/b/c.py", ("a/b/c.py", True)),
        ("a/[bc]/d", ("a", False)),
    ],
)
def test_literal_base(pattern: str, expected: tuple[str, bool]) -> None:
    assert literal_base(pattern) == expected


@given(
    st.text(
        alphabet=st.characters(blacklist_characters="*?[]/\\", blacklist_categories=["Cs"]),
        min_size=1,
        max_size=20,
    )
)
def test_literal_patterns_match_exactly_themselves(name: str) -> None:
    pattern = normalize(name)
    if not pattern or pattern.startswith("."):
        return
    assert glob_match(pattern, pattern, case_insensitive=False)
    assert not glob_match(pattern + "x", pattern, case_insensitive=False)


def test_compiled_patterns_are_cached() -> None:
    assert compile_glob("a/**") is compile_glob("a/**")
    assert isinstance(compile_glob("a/**"), re.Pattern)


def test_iter_files_walks_only_literal_bases(tmp_path: Path) -> None:
    for rel in ["paper/a.tex", "paper/s/b.tex", "paper/c.md", "README.md", ".venv/x/paper/z.tex"]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    found = iter_files(tmp_path, ["paper/**/*.tex", "README.md", "missing/**", "NOPE.md"])
    assert [p.relative_to(tmp_path).as_posix() for p in found] == [
        "README.md",
        "paper/a.tex",
        "paper/s/b.tex",
    ]


def test_iter_files_handles_empty_inputs(tmp_path: Path) -> None:
    assert iter_files(tmp_path, []) == []
    assert iter_files(tmp_path / "missing", ["*"]) == []


def test_iter_files_root_wide_pattern(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "b.py").write_text("")
    assert len(iter_files(tmp_path, ["**/*.py"])) == 2
