"""Wave-3 guard hardening, through the real hook wrapper (docs/changes/qcal-foundations.md, S2).

- ``allow-only`` refuses control characters: shlex reads a newline as a space, bash ends the
  command there.
- The reproducer's exact ``git diff`` / ``git log`` forms are allowed; nothing else is.
- A file with more than one hard link cannot be edited or privately read.
- Clean-room matching folds Unicode: full-width, zero-width, accented and dash variants match.
- ``prior-art-scout`` cannot read ``ian_data``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qcal.config import load_config
from tests.conftest import REPO_ROOT, require_hard_links, run_git, write

pytestmark = pytest.mark.rule("C0", "C6", "A5", "A6")

WRAPPER = REPO_ROOT / ".claude" / "hooks" / "run_hook.sh"


def _wrapper(
    *args: str, payload: dict[str, object], project: Path
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "CLAUDE_PROJECT_DIR": str(project),
        "HOME": str(project),
        "QCAL_PYTHON": sys.executable,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    return subprocess.run(
        [str(WRAPPER), *args],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=project,
        env=env,
        check=False,
        timeout=60,
    )


def _bash(command: str) -> dict[str, object]:
    return {"tool_name": "Bash", "tool_input": {"command": command}}


def _agent_commands(agent: str) -> list[str]:
    """The quoted arguments after the hook name in the agent's Bash hook."""
    import shlex

    text = (REPO_ROOT / ".claude/agents" / f"{agent}.md").read_text("utf-8")
    line = next(line for line in text.splitlines() if "allow-only" in line)
    command = line.split("command:", 1)[1].strip().strip("'")
    words = shlex.split(command)
    return words[words.index("allow-only") + 1 :]


# --- allow-only and the reproducer's commands -------------------------------------------------


@pytest.mark.parametrize(
    "command",
    ["git diff src/qcal_lab", "git diff tests", "git diff --stat", "git log --oneline -20"],
)
def test_the_reproducers_review_commands_are_allowed(command: str) -> None:
    commands = _agent_commands("paper-reproducer")
    result = _wrapper("allow-only", *commands, payload=_bash(command), project=REPO_ROOT)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "git diff\nsrc/qcal_lab",  # tokenises like an allowed command; bash runs two
        "git diff src/qcal_lab\nrm -rf runs",
        "git diff\rsrc/qcal_lab",
        "git diff\tsrc/qcal_lab\x00",
        "git diff --output=/tmp/x src/qcal_lab",
        "git diff src/qcal_lab; rm -rf runs",
        "git diff --no-index /etc/passwd tests",
        "git diff --ext-diff tests",
        "git log --oneline -20 -p",
    ],
)
def test_anything_but_the_exact_commands_is_refused(command: str) -> None:
    commands = _agent_commands("paper-reproducer")
    result = _wrapper("allow-only", *commands, payload=_bash(command), project=REPO_ROOT)
    assert result.returncode == 2, command
    assert (
        "BLOCKED" in result.stderr or "may run only" in result.stderr or "control" in result.stderr
    )


def test_a_trailing_newline_is_still_the_same_command() -> None:
    result = _wrapper("allow-only", "git status", payload=_bash("git status\n"), project=REPO_ROOT)
    assert result.returncode == 0, result.stderr


# --- hard links ---------------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    write(root, "EXPERIMENTS.yaml", "cells: []\n")
    write(root, "notes/free.md", "free\n")
    run_git(root, "init", "-q")
    return root


def test_a_hard_link_to_a_protected_file_cannot_be_edited(project: Path) -> None:
    require_hard_links(project)
    alias = project / "notes/alias.md"
    os.link(project / "EXPERIMENTS.yaml", alias)
    payload = {"tool_name": "Edit", "tool_input": {"file_path": str(alias)}}
    result = _wrapper("guard-paths", payload=payload, project=project)
    assert result.returncode == 2
    assert "hard links" in result.stderr


def test_a_symlink_to_a_hard_link_cannot_be_edited(project: Path) -> None:
    require_hard_links(project)
    alias = project / "notes/alias.md"
    os.link(project / "EXPERIMENTS.yaml", alias)
    (project / "notes/pointer.md").symlink_to(alias)
    payload = {"tool_name": "Write", "tool_input": {"file_path": str(project / "notes/pointer.md")}}
    assert _wrapper("guard-paths", payload=payload, project=project).returncode == 2


def test_an_ordinary_file_and_a_new_file_can_be_edited(project: Path) -> None:
    for name in ("notes/free.md", "notes/new.md"):
        payload = {"tool_name": "Edit", "tool_input": {"file_path": str(project / name)}}
        result = _wrapper("guard-paths", payload=payload, project=project)
        assert result.returncode == 0, (name, result.stderr)


def test_a_hard_link_cannot_be_read_privately(project: Path) -> None:
    require_hard_links(project)
    os.link(project / "EXPERIMENTS.yaml", project / "notes/alias.md")
    payload = {"tool_name": "Read", "tool_input": {"file_path": str(project / "notes/alias.md")}}
    result = _wrapper("deny-read", "ian_only", payload=payload, project=project)
    assert result.returncode == 2
    assert "hard links" in result.stderr


# --- clean-room folding -----------------------------------------------------------------------


def _needle() -> str:
    return load_config(REPO_ROOT).str_list("policy.clean_room_substrings")[0]


def _fullwidth(text: str) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in text)


@pytest.mark.parametrize(
    "spelling",
    [
        _fullwidth,
        lambda s: s[:2] + "\u200b" + s[2:],  # zero-width space (Cf)
        lambda s: s[:1] + "\u00ad" + s[1:],  # soft hyphen (Cf)
        lambda s: s[:1] + "\u0301" + s[1:],  # combining acute accent (Mn)
        str.upper,
    ],
    ids=["fullwidth", "zero-width", "soft-hyphen", "combining-mark", "upper"],
)
def test_a_disguised_clean_room_substring_is_refused(spelling) -> None:
    command = f"echo {spelling(_needle())}"
    result = _wrapper("guard-bash", payload=_bash(command), project=REPO_ROOT)
    assert result.returncode == 2, command
    assert "clean-room" in result.stderr


def test_a_dash_variant_of_a_hyphenated_substring_is_refused() -> None:
    hyphenated = next(
        s for s in load_config(REPO_ROOT).str_list("policy.clean_room_substrings") if "-" in s
    )
    for dash in ("\u2011", "\u2010", "\u2212", "\u2013"):
        command = f"echo {hyphenated.replace('-', dash)}"
        assert _wrapper("guard-bash", payload=_bash(command), project=REPO_ROOT).returncode == 2


# --- the scout cannot read Ian's data ---------------------------------------------------------


@pytest.mark.parametrize(
    "path", ["data/manifests/val.txt", "tests/parity/fixtures/x.json", "docs/reference/notes.md"]
)
def test_the_prior_art_scout_cannot_read_ians_data(path: str) -> None:
    text = (REPO_ROOT / ".claude/agents/prior-art-scout.md").read_text("utf-8")
    line = next(line for line in text.splitlines() if "deny-read" in line)
    targets = line.split("deny-read", 1)[1].strip(" '").split()
    payload = {"tool_name": "Read", "tool_input": {"file_path": str(REPO_ROOT / path)}}
    result = _wrapper("deny-read", *targets, payload=payload, project=REPO_ROOT)
    assert result.returncode == 2, (path, result.stderr)
