"""The exact ``qcal ci`` invocations the workflows run, end to end (peer-review finding 13)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from qcal.ci import immutability
from qcal.cli import main
from qcal.hooks.bash import analyze
from tests.conftest import make_record, run_git, write

pytestmark = pytest.mark.integration


@pytest.fixture
def history(git_repo: Path) -> tuple[Path, str, str]:
    base = run_git(git_repo, "rev-parse", "HEAD")
    write(git_repo, "CLAIMS.md", "unsigned edit\n")
    write(git_repo, "runs/registry/R1.json", json.dumps(make_record("R1").to_dict()))
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "unsigned protected change and a record")
    return git_repo, base, run_git(git_repo, "rev-parse", "HEAD")


def ci(repo: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    return main(["--root", str(repo), "ci", *argv], out=out), out.getvalue()


def test_verify_signatures_as_integrity_yml_runs_it(history, tmp_path: Path) -> None:
    repo, base, head = history
    report_file = tmp_path / "signatures.json"

    code, out = ci(
        repo, "verify-signatures", "--base", base, "--head", head, "--github",
        "--json-out", str(report_file),
    )  # fmt: skip

    assert code == 0  # bootstrap mode reports but never fails
    assert "::warning::commit" in out
    assert "::notice::signing.mode is 'bootstrap'" in out
    assert "signatures (bootstrap, 0 allowed key(s)): PASS" in out
    data = json.loads(report_file.read_text())
    assert data["violations"][0]["files"] == ["CLAIMS.md", "runs/registry/R1.json"]


def test_verify_signatures_enforce_override_fails(history) -> None:
    repo, base, head = history

    code, out = ci(repo, "verify-signatures", "--base", base, "--head", head, "--mode", "enforce")

    assert code == 1
    assert "FAIL" in out


def test_verify_signatures_json_flag(history) -> None:
    repo, base, head = history

    code, out = ci(repo, "verify-signatures", "--base", base, "--head", head, "--json")

    assert code == 0
    assert json.loads(out)["mode"] == "bootstrap"


def test_registry_immutable_as_integrity_yml_runs_it(history, tmp_path: Path) -> None:
    repo, base, head = history
    report_file = tmp_path / "imm.json"

    code, out = ci(
        repo, "registry-immutable", "--base", base, "--head", head, "--github",
        "--json-out", str(report_file),
    )  # fmt: skip

    assert code == 0
    assert out.startswith("registry immutability: PASS (1 added)")
    assert json.loads(report_file.read_text())["added"] == ["runs/registry/R1.json"]


def test_registry_immutable_reports_violations_as_errors(history) -> None:
    repo, _base, head = history
    write(repo, "runs/registry/R1.json", "{}")
    run_git(repo, "commit", "-q", "-am", "rewrite a record")
    new_head = run_git(repo, "rev-parse", "HEAD")

    code, out = ci(repo, "registry-immutable", "--base", head, "--head", new_head, "--github")

    assert code == 1
    assert "::error::registry: runs/registry/R1.json: status M" in out


def test_immutability_fails_closed_on_unparseable_diff_output(
    history, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, base, head = history
    real_git = immutability.git

    def odd(args: list[str], cwd: Path) -> str:
        output = real_git(args, cwd)
        return output + "A\0" if args and args[0] == "diff" else output

    monkeypatch.setattr(immutability, "git", odd)

    report = immutability.check_registry_immutable(repo, base, head)

    assert not report.passed
    assert "unparseable git diff output" in report.violations[0]


@pytest.mark.parametrize("bad", ["nope", "HEAD~99"])
def test_bad_refs_are_usage_errors(history, bad: str) -> None:
    repo, _base, head = history

    for command in ("verify-signatures", "registry-immutable"):
        code, _ = ci(repo, command, "--base", bad, "--head", head)
        assert code == 2, command


@pytest.mark.parametrize(
    ("command", "protected"),
    [
        ("git push origin -- main", True),
        ("git push -- origin main", True),
        ("git push origin -- claude/x", False),
        ("git push --repo=origin -- main", True),
    ],
)
def test_double_dash_ends_options_in_push(command: str, *, protected: bool) -> None:
    finding = analyze(
        command,
        protected_branches=["main"],
        deny_flags=["--force"],
        cwd="/w",
        resolve_branch=lambda _: "claude/x",
    )
    assert (finding is not None) is protected
