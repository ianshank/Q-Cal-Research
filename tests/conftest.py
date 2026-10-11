"""Shared fixtures. Every test runs with a clean qcal/Claude environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import settings

from qcal.config import Config, load_config
from qcal.registry.executor import ExecutionResult, RunSpec
from qcal.registry.records import RunRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_PREFIXES = ("QCAL_", "QCAL__", "CLAUDE_")
# Each top-level test directory is a suite with a marker of the same name: `pytest -m security`.
SUITES = ("unit", "integration", "regression", "security", "e2e", "parity", "oracle", "contract")
# Suites added after pyproject.toml's marker list was signed; registered here instead.
_LATER_SUITES = {
    "oracle": "independent mathematical oracles (scikit-learn, SciPy, pycocotools; tests/oracle)",
    "contract": "the EvalLoop contract, run against the fixture and Ian's loop (tests/contract)",
}
# On CI a missing tool must fail, never skip: a silently skipped signing test proves nothing.
REQUIRE_TOOLS_ENV = "CI"
HYPOTHESIS_PROFILE_ENV = "HYPOTHESIS_PROFILE"

# Hypothesis profiles. CI is derandomised (a signed gate must be reproducible) and has no
# example database; `explore` searches wider and is randomised (weekly job, `make pre-pr`).
# A failure found by `explore` is pinned with `@example` in the fix.
settings.register_profile(
    "ci", max_examples=200, derandomize=True, deadline=None, database=None, print_blob=True
)
settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile("explore", max_examples=500, deadline=None, print_blob=True)
settings.load_profile(
    os.environ.get(HYPOTHESIS_PROFILE_ENV)
    or ("ci" if os.environ.get(REQUIRE_TOOLS_ENV, "").lower() in {"1", "true", "yes"} else "dev")
)


def pytest_configure(config: pytest.Config) -> None:
    for suite, description in _LATER_SUITES.items():
        config.addinivalue_line("markers", f"{suite}: {description}")
    config.addinivalue_line(
        "markers",
        "rule(*ids): the integrity rules a test enforces; scripts/traceability.py maps them",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    tests_dir = Path(__file__).resolve().parent
    for item in items:
        try:
            suite = item.path.resolve().relative_to(tests_dir).parts[0]
        except ValueError:
            continue
        if suite in SUITES:
            item.add_marker(getattr(pytest.mark, suite))


def require_tool(name: str) -> str:
    """Path of ``name``; skip locally when it is missing, fail on CI."""
    import shutil

    path = shutil.which(name)
    if path is None:
        message = f"{name} not available"
        if os.environ.get(REQUIRE_TOOLS_ENV, "").lower() in {"1", "true", "yes"}:
            pytest.fail(
                f"{message}, and {REQUIRE_TOOLS_ENV} requires every tool-backed test to run"
            )
        pytest.skip(message)
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove environment variables that change configuration or hook behaviour."""
    for name in list(os.environ):
        # GIT_* too: a test run from inside a git hook would otherwise use the outer index.
        if name.startswith((*_ENV_PREFIXES, "GIT_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LC_ALL", "C")  # git and tool messages in one language
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)  # no host signing helpers or hooks
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """An empty project root containing a minimal qcal.toml (marks it as the root)."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "qcal.toml").write_text("# test project\n", encoding="utf-8")
    return root


@pytest.fixture
def config(repo: Path) -> Config:
    return load_config(repo, environ={})


@pytest.fixture
def make_config(repo: Path) -> Callable[[str], Config]:
    """Write a qcal.toml body and load it."""

    def _make(body: str) -> Config:
        (repo / "qcal.toml").write_text(textwrap.dedent(body), encoding="utf-8")
        return load_config(repo, environ={})

    return _make


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def git_repo(repo: Path) -> Path:
    """``repo`` initialised as a git repository with one commit on ``main``."""
    run_git(repo, "init", "-q", "-b", "main")
    run_git(repo, "config", "commit.gpgsign", "false")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "initial")
    return repo


def write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def experiments_yaml(
    cells: list[dict[str, Any]], seeds: list[int] | None = None, **extra: Any
) -> str:
    import yaml

    data: dict[str, Any] = {
        "version": 2,
        "seed_role": "calibrator_fit_draw",
        "seeds": seeds if seeds is not None else [0, 1, 2],
        "cells": cells,
    }
    data.update(extra)
    return yaml.safe_dump(data, sort_keys=False)


def make_record(run_id: str = "R1", **overrides: Any) -> RunRecord:
    fields: dict[str, Any] = {
        "run_id": run_id,
        "cell_id": "C-a",
        "seed": 0,
        "status": "ok",
        "started_at": "2026-10-09T00:00:00+00:00",
        "finished_at": "2026-10-09T00:01:00+00:00",
        "seed_role": "calibrator_fit_draw",
        "factors": {"detector": "atss", "precision": "fp32"},
        "metrics": {"LaECE0": 12.5, "AP": 41.0},
    }
    fields.update(overrides)
    return RunRecord(**fields)


class FakeExecutor:
    """Records specs and returns scripted results; never starts a process."""

    def __init__(
        self, results: list[ExecutionResult] | None = None, metrics: dict[str, float] | None = None
    ) -> None:
        self.results = list(results or [])
        self.metrics = metrics or {"LaECE0": 10.0, "AP": 40.0}
        self.specs: list[RunSpec] = []

    def execute(self, spec: RunSpec) -> ExecutionResult:
        self.specs.append(spec)
        if self.results:
            return self.results.pop(0)
        return ExecutionResult(0, metrics=dict(self.metrics))


@pytest.fixture
def fake_executor() -> FakeExecutor:
    return FakeExecutor()


def experiment_script(tmp_path: Path) -> Path:
    """A tiny experiment program honouring the executor's result-file contract."""
    script = tmp_path / "fake_experiment.py"
    script.write_text(
        textwrap.dedent("""
        import json, sys, zlib
        cell, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
        base = (zlib.crc32(cell.encode()) % 1000) / 100.0
        json.dump({"format": "qcal.executor_result", "version": 1, "status": "ok",
                   "failure_kind": None, "error": None, "artifacts": [],
                   "metrics": {"LaECE0": round(base + seed, 4), "AP": round(40 + seed / 10, 4)},
                   "environment": {"trt_version": "test"}}, open(out, "w"))
    """),
        encoding="utf-8",
    )
    return script


def executor_command(script: Path) -> list[str]:
    return [sys.executable, str(script), "{cell_id}", "{seed}", "{result_path}"]


def toml_list(values: list[str]) -> str:
    return json.dumps(values)


@pytest.fixture
def hook_payload() -> Callable[..., str]:
    def _payload(tool: str | None = "Write", **tool_input: Any) -> str:
        data: dict[str, Any] = {"hook_event_name": "PreToolUse", "tool_input": tool_input}
        if tool is not None:
            data["tool_name"] = tool
        return json.dumps(data)

    return _payload


@pytest.fixture
def chdir(monkeypatch: pytest.MonkeyPatch) -> Callable[[Path], None]:
    def _chdir(path: Path) -> None:
        monkeypatch.chdir(path)

    return _chdir


@pytest.fixture
def ssh_keygen() -> str:
    return require_tool("ssh-keygen")


# --- SSH-signed repositories (security and regression suites) -------------------------------

PRINCIPAL = "ian@example.invalid"


@pytest.fixture
def signed_repo(
    repo: Path, ssh_keygen: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str, str]:
    """A repo whose first commit is pre-policy history, then a signed commit adding the policy."""
    for variable in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(variable, PRINCIPAL)  # committer must equal the signing principal
    key = tmp_path / "ian"
    subprocess.run([ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    run_git(repo, "init", "-q", "-b", "main")
    run_git(repo, "config", "user.email", PRINCIPAL)
    run_git(repo, "config", "commit.gpgsign", "false")
    (repo / "qcal.toml").unlink()
    write(repo, "README.md", "old history\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "old history without a policy")
    old = run_git(repo, "rev-parse", "HEAD")
    pub = (tmp_path / "ian.pub").read_text().strip()
    write(repo, "allowed_signers", f'{PRINCIPAL} namespaces="git" {pub}\n')
    write(repo, "qcal.toml", '[signing]\nmode = "enforce"\n')
    write(repo, "EXPERIMENTS.yaml", "version: 2\n")
    run_git(repo, "add", "-A")
    sign(repo, key, "signed policy")
    base = run_git(repo, "rev-parse", "HEAD")
    return repo, key, old, base


def sign(repo: Path, key: Path, message: str) -> None:
    """Commit the staged changes with an SSH signature from ``key``."""
    run_git(
        repo,
        "-c",
        "gpg.format=ssh",
        "-c",
        "gpg.ssh.program=ssh-keygen",
        "-c",
        f"user.signingkey={key}",
        "commit",
        "-q",
        "-S",
        "-m",
        message,
    )
