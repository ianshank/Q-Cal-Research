"""Command-reference drift: documented ``qcal`` commands and ``make`` targets must exist."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qcal.cli import build_parser
from qcal.config import Config
from qcal.integrity.agent_layer import check_agent_layer
from qcal.integrity.command_refs import (
    check_command_refs,
    code_lines,
    extract_refs,
    make_targets,
    validate_make,
    validate_qcal,
)
from tests.conftest import write

PARSER = build_parser()


# --- extraction -----------------------------------------------------------------------------


def test_markdown_only_scans_code() -> None:
    text = "Run qcal claims in prose.\n`qcal claims --json`\n```bash\nqcal registry audit\n```\n"

    refs = extract_refs(text, "x.md", markdown=True)

    assert [(r.line, r.tokens) for r in refs] == [
        (2, ("claims", "--json")),
        (4, ("registry", "audit")),
    ]


def test_non_markdown_strips_comments_and_yaml_prose_keys() -> None:
    text = (
        "# qcal Claude Code hooks\n"
        "  - name: Install qcal from the base commit\n"
        "    run: python -I -m qcal claims  # qcal bogus\n"
    )

    refs = extract_refs(text, "ci.yml", markdown=False)

    assert [r.tokens for r in refs] == [("claims",)]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("$(QCAL) registry index --check", ("registry", "index", "--check")),
        (
            "python -I -m qcal ci verify-signatures --base x",
            ("ci", "verify-signatures", "--base", "x"),
        ),
        ("qcal-registry audit", ("registry", "audit")),
        ("qcal-claims --json", ("claims", "--json")),
        ("qcal registry run|run-batch", ("registry", "run")),
        ("qcal leakage --json > out.json", ("leakage", "--json")),
        ("x=$(qcal config)", ("config",)),
    ],
)
def test_invocation_forms(line: str, expected: tuple[str, ...]) -> None:
    (ref,) = extract_refs(line, "Makefile", markdown=False)
    assert ref.tokens == expected


@pytest.mark.parametrize(
    "line", ["src/qcal/cli", "qcal.toml", "qcal:ignore", "pip install qcal-lab", '"qcal env"']
)
def test_non_invocations_are_ignored(line: str) -> None:
    assert extract_refs(line, "Makefile", markdown=False) == []


def test_make_invocations() -> None:
    refs = extract_refs("make test PYTHON=python3.12 && $(MAKE) lint", "x.sh", markdown=False)
    assert [r.tokens for r in refs] == [("test", "PYTHON=python3.12"), ("lint",)]


def test_fences_toggle_and_tilde_fences() -> None:
    text = "~~~\nqcal claims\n~~~\nqcal nothing here\n"
    assert [n for n, _ in code_lines(text, markdown=True)] == [2]


# --- validation -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tokens",
    [
        (),
        ("claims",),
        ("--debug", "claims", "--json"),
        ("--root", "/x", "registry", "audit"),
        ("--root=/x", "registry", "audit", "--strict"),
        ("registry", "run", "C-a", "0"),
        ("registry", "run-batch", "--seeds", "0,1"),
        ("registry", "<subcommand>"),
        ("hook", "--fail-open", "claims"),
        ("policy", "list", "--category", "ian_only"),
        ("ci", "verify-signatures", "--", "--anything"),
    ],
)
def test_valid_command_paths(tokens: tuple[str, ...]) -> None:
    assert validate_qcal(PARSER, tokens) is None


@pytest.mark.parametrize(
    ("tokens", "fragment"),
    [
        (("registry", "licenses"), "has no command 'licenses' (in `qcal registry`)"),
        (("claimz",), "has no command 'claimz' (in `qcal`)"),
        (("registry", "index", "--verify"), "has no option --verify (in `qcal registry index`)"),
        (("--verbose", "claims"), "has no option --verbose (in `qcal`)"),
        (("claims", "--js"), "has no option --js"),  # abbreviations are drift too
    ],
)
def test_drifted_command_paths(tokens: tuple[str, ...], fragment: str) -> None:
    problem = validate_qcal(PARSER, tokens)
    assert problem is not None
    assert fragment in problem


@given(
    st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=8), max_size=6)
)
def test_validation_never_raises(tokens: list[str]) -> None:
    result = validate_qcal(PARSER, tokens)
    assert result is None or isinstance(result, str)


def test_validate_against_a_custom_parser() -> None:
    parser = argparse.ArgumentParser(prog="qcal")
    parser.add_subparsers(dest="cmd").add_parser("only")

    assert validate_qcal(parser, ("only",)) is None
    assert validate_qcal(parser, ("other",)) is not None


def test_make_targets_parsing(tmp_path: Path) -> None:
    makefile = write(
        tmp_path,
        "Makefile",
        "PY ?= python\n.PHONY: a b\na b: ## two\n\techo x: y\nc:\n%.o: %.c\nX := 1\nd::\n",
    )
    assert make_targets(makefile) == {"a", "b", "c"}
    assert make_targets(tmp_path / "missing") == set()


@pytest.mark.parametrize(
    ("tokens", "ok"),
    [
        (("a",), True),
        (("a", "PY=3"), True),
        (("-j4", "b"), True),
        (("zz",), False),
        (("<t>",), True),
    ],
)
def test_validate_make(tokens: tuple[str, ...], *, ok: bool) -> None:
    assert (validate_make({"a", "b"}, tokens) is None) is ok


# --- the whole check --------------------------------------------------------------------------


def test_drift_is_reported_with_file_and_line(config: Config, repo: Path) -> None:
    write(repo, "Makefile", "check:\n\t$(QCAL) claims\n")
    write(
        repo,
        ".claude/skills/s/SKILL.md",
        "---\ndescription: d\n---\nRun `qcal registry licenses`.\nThen `make chek`.\n",
    )

    errors = check_agent_layer(config).errors

    assert errors == [
        (
            "command drift: .claude/skills/s/SKILL.md:4: `qcal registry licenses` has no command "
            "'licenses' (in `qcal registry`)"
        ),
        "command drift: .claude/skills/s/SKILL.md:5: `make chek` has no Makefile target 'chek'",
    ]


def test_make_refs_are_skipped_without_a_makefile(config: Config, repo: Path) -> None:
    write(repo, "AGENTS.md", "`make anything`\n")

    errors, checked = check_command_refs(config)

    assert (errors, checked) == ([], 1)


def test_excluded_and_unreadable_files_are_skipped(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    config = make_config('[agent_layer]\ncommand_ref_exclude_globs = ["docs/plan.md"]\n')
    write(repo, "docs/plan.md", "`qcal future-command`\n")
    (repo / "docs" / "binary.md").write_bytes(b"\xff\xfe`qcal x`")

    assert check_command_refs(config) == ([], 0)


def test_parser_factory_is_injectable(config: Config, repo: Path) -> None:
    write(repo, "AGENTS.md", "`qcal only`\n")
    parser = argparse.ArgumentParser(prog="qcal")
    parser.add_subparsers(dest="cmd").add_parser("only")

    errors, checked = check_command_refs(config, parser_factory=lambda: parser)

    assert (errors, checked) == ([], 1)


def test_error_string_format(config: Config, repo: Path) -> None:
    write(repo, "AGENTS.md", "\n`qcal nope --x`\n")

    (error,), _ = check_command_refs(config)

    assert str(error) == "AGENTS.md:2: `qcal nope --x` has no command 'nope' (in `qcal`)"


def test_fences_of_skipped_languages_are_not_scanned() -> None:
    text = "```mermaid\nqcal tooling\n```\n```bash\nqcal claims\n```\n```\nqcal x\n```\n"

    refs = extract_refs(text, "a.md", markdown=True, skip_languages=frozenset({"mermaid"}))

    assert [r.tokens for r in refs] == [("claims",), ("x",)]


def test_skipped_fence_languages_are_configurable(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    write(repo, "docs/a.md", "```mermaid\nqcal tooling\n```\n")
    assert check_command_refs(make_config("# defaults\n")) == ([], 0)
    config = make_config("[agent_layer]\ncommand_ref_skip_fences = []\n")
    (error,), checked = check_command_refs(config)
    assert checked == 1
    assert "has no command 'tooling'" in str(error)
