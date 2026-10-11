"""The real .claude/hooks/run_hook.sh, driven the way Claude Code drives it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

pytestmark = [pytest.mark.integration, pytest.mark.rule("C0")]
WRAPPER = REPO_ROOT / ".claude" / "hooks" / "run_hook.sh"


def run_hook(
    args: list[str], payload: dict[str, object], project: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    environ = {
        "PATH": os.environ["PATH"],
        "CLAUDE_PROJECT_DIR": str(project),
        "QCAL_PYTHON": sys.executable,
        "HOME": os.environ.get("HOME", "/tmp"),
        **env,
    }
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=environ,
        check=False,
        timeout=60,
    )


def edit(path: Path, tool: str = "Write") -> dict[str, object]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "tool_input": {"file_path": str(path)},
    }


def bash(command: str, cwd: Path) -> dict[str, object]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "cwd": str(cwd),
        "tool_input": {"command": command},
    }


@pytest.mark.parametrize(
    "relative",
    [
        "EXPERIMENTS.yaml",
        "experiments.yaml",
        ".claude/settings.json",
        ".github/workflows/ci.yml",
        "qcal.toml",
        "src/qcal/hooks/guards.py",
        "runs/index.csv",
        "analysis/handwritten/notes.md",
    ],
)
def test_protected_paths_are_blocked_in_the_real_repository(relative: str) -> None:
    result = run_hook(["guard-paths"], edit(REPO_ROOT / relative), REPO_ROOT)
    assert result.returncode == 2, result.stderr
    assert result.stderr.startswith("BLOCKED:")


@pytest.mark.parametrize(
    "relative", ["src/qcal_lab/calib/platt.py", "docs/notes.md", "AMENDMENTS.md"]
)
def test_ordinary_paths_are_allowed(relative: str) -> None:
    result = run_hook(["guard-paths"], edit(REPO_ROOT / relative), REPO_ROOT)
    assert (result.returncode, result.stderr) == (0, "")


def test_symlink_to_a_protected_file_is_blocked(repo: Path) -> None:
    (repo / "EXPERIMENTS.yaml").write_text("x")
    (repo / "e.yaml").symlink_to(repo / "EXPERIMENTS.yaml")
    assert run_hook(["guard-paths"], edit(repo / "e.yaml", "Edit"), repo).returncode == 2


@pytest.mark.parametrize(
    ("command", "code"),
    [
        ("git push --force origin claude/x", 2),
        ("git push origin HEAD:main", 2),
        ("git push -u origin main", 2),
        ("git push origin +main", 2),
        ("git push origin civ", 2),
        ("git push origin claude/x", 0),
        ("git push origin main-fix", 0),
        ("pytest --device cpu > log.txt && cat EXPERIMENTS.yaml", 0),
        ("cp a b && cat handwritten/x.py", 0),
    ],
)
def test_bash_guard_fixtures(command: str, code: int) -> None:
    assert run_hook(["guard-bash"], bash(command, REPO_ROOT), REPO_ROOT).returncode == code


def test_missing_interpreter_fails_closed(repo: Path) -> None:
    result = subprocess.run(
        ["/bin/bash", str(WRAPPER), "guard-paths"],
        input="{}",
        text=True,
        capture_output=True,
        check=False,
        env={"PATH": "/nonexistent", "CLAUDE_PROJECT_DIR": str(repo)},
    )
    assert result.returncode == 2
    assert "no Python interpreter" in result.stderr


def test_interpreter_crash_is_converted_to_a_block(repo: Path, tmp_path: Path) -> None:
    crashing = tmp_path / "python"
    crashing.write_text("#!/bin/sh\nexit 1\n")
    crashing.chmod(0o755)
    result = run_hook(["guard-paths"], edit(repo / "a.md"), repo, QCAL_PYTHON=str(crashing))
    assert result.returncode == 2
    assert "treating it as a failure" in result.stderr


def test_fail_open_for_the_stop_hook(repo: Path, tmp_path: Path) -> None:
    crashing = tmp_path / "python"
    crashing.write_text("#!/bin/sh\nexit 1\n")
    crashing.chmod(0o755)
    result = run_hook(["--fail-open", "claims"], {}, repo, QCAL_PYTHON=str(crashing))
    assert result.returncode == 0
    assert result.stderr.startswith("WARNING")


def test_wrapper_requires_a_hook_name() -> None:
    result = subprocess.run(
        ["bash", str(WRAPPER)], input="{}", text=True, capture_output=True, check=False
    )
    assert result.returncode == 2


def test_stop_hook_passes_on_a_clean_project(repo: Path) -> None:
    result = run_hook(["--fail-open", "claims"], {"hook_event_name": "Stop"}, repo)
    assert (result.returncode, result.stderr) == (0, "")
