from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from qcal.hooks import cli
from qcal.hooks.decision import BLOCK_EXIT_CODE, Decision, Mode, emit, resolve_mode
from qcal.registry.index import write_index
from qcal.registry.store import RegistryStore
from tests.conftest import make_record, write


def call(hook: list[str], stdin: str, repo: Path, **env: str) -> tuple[int, str]:
    err = io.StringIO()
    environ = {"CLAUDE_PROJECT_DIR": str(repo), **env}
    code = cli.main(hook, stdin=io.StringIO(stdin), stderr=err, environ=environ)
    return code, err.getvalue()


def write_payload(path: Path, tool: str = "Write") -> str:
    return json.dumps({"tool_name": tool, "tool_input": {"file_path": str(path)}})


def test_guard_denies_with_exit_two_and_reason(repo: Path) -> None:
    code, err = call(["guard-paths"], write_payload(repo / "EXPERIMENTS.yaml"), repo)
    assert code == BLOCK_EXIT_CODE
    assert err.startswith("BLOCKED:")
    assert "Ian-only" in err


def test_guard_allows_silently(repo: Path) -> None:
    assert call(["guard-paths"], write_payload(repo / "docs/a.md"), repo) == (0, "")


@pytest.mark.parametrize(
    ("mode", "code", "prefix"), [("warn", 0, "WARNING"), ("off", 0, ""), ("enforce", 2, "BLOCKED")]
)
def test_human_controlled_mode(repo: Path, mode: str, code: int, prefix: str) -> None:
    result_code, err = call(
        ["guard-paths"], write_payload(repo / "CLAIMS.md"), repo, QCAL_GUARD_MODE=mode
    )
    assert result_code == code
    assert err.startswith(prefix)


def test_unknown_mode_fails_closed(repo: Path) -> None:
    code, err = call(
        ["guard-paths"], write_payload(repo / "docs/a.md"), repo, QCAL_GUARD_MODE="yolo"
    )
    assert code == BLOCK_EXIT_CODE
    assert "could not start" in err


def test_invalid_json_fails_closed_for_guards(repo: Path) -> None:
    code, err = call(["guard-bash"], "{not json", repo)
    assert code == BLOCK_EXIT_CODE
    assert "not valid JSON" in err


def test_internal_errors_fail_closed_for_guards(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> Decision:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli.guards, "guard_paths", boom)
    code, err = call(["guard-paths"], write_payload(repo / "a.md"), repo)
    assert code == BLOCK_EXIT_CODE
    assert "kaboom" in err


def test_decision_log_is_jsonl(repo: Path, tmp_path: Path) -> None:
    log = tmp_path / "decisions.jsonl"
    call(["guard-paths"], write_payload(repo / "CLAIMS.md"), repo, QCAL_HOOK_LOG=str(log))
    call(["guard-paths"], write_payload(repo / "ok.md"), repo, QCAL_HOOK_LOG=str(log))
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["allow"] for e in entries] == [False, True]
    assert entries[0]["rule"] == "paths.ian_only"
    assert entries[0]["tool"] == "Write"


def test_relative_decision_log_resolves_against_root(repo: Path) -> None:
    (repo / "qcal.toml").write_text('[hooks]\nlog_file = "runs/logs/hooks.jsonl"\n')
    call(["guard-paths"], write_payload(repo / "a.md"), repo)
    assert (repo / "runs/logs/hooks.jsonl").is_file()


def test_unwritable_decision_log_never_changes_the_decision(repo: Path, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    code, _ = call(
        ["guard-paths"], write_payload(repo / "a.md"), repo, QCAL_HOOK_LOG=str(blocker / "x.jsonl")
    )
    assert code == 0


def test_scoped_hooks_dispatch(repo: Path) -> None:
    assert call(["scope-write", "review/prior-art"], write_payload(repo / "src/x.py"), repo)[0] == 2
    assert (
        call(["deny-read", "ian_only"], write_payload(repo / "DECISIONS.md", "Read"), repo)[0] == 2
    )
    bash = json.dumps({"tool_name": "Bash", "tool_input": {"command": "make release"}})
    assert call(["allow-only", "make release"], bash, repo)[0] == 0


def _index_with_one_run(repo: Path) -> None:
    from qcal.config import load_config

    config = load_config(repo, environ={})
    RegistryStore(config.path("registry_dir")).write(make_record("R1"))
    write_index(config, RegistryStore(config.path("registry_dir")))


def test_claims_stop_hook_blocks_once_then_warns(repo: Path) -> None:
    _index_with_one_run(repo)
    write(repo, "paper/sections/results.tex", r"LaECE is \qcalval{run:R1:LaECE0}{99.00}.")
    code, err = call(["claims"], "{}", repo)
    assert code == BLOCK_EXIT_CODE
    assert "registry gives 12.50" in err
    code, err = call(["claims"], json.dumps({"stop_hook_active": True}), repo)
    assert code == 0
    assert "not blocking again" in err


def test_claims_stop_hook_passes_on_clean_tree(repo: Path) -> None:
    _index_with_one_run(repo)
    write(repo, "paper/sections/results.tex", r"LaECE is \qcalval{run:R1:LaECE0}{12.50}.")
    assert call(["claims"], "{}", repo) == (0, "")


def test_claims_hook_fails_open_on_errors(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> list[object]:
        raise RuntimeError("index unreadable")

    monkeypatch.setattr("qcal.integrity.claims.check_claims", boom)
    code, err = call(["claims"], "{}", repo)
    assert code == 0
    assert "not blocking" in err
    assert call(["claims"], "{bad", repo)[0] == 0


def test_resolve_mode_and_emit(config) -> None:
    assert resolve_mode(config, {}) is Mode.ENFORCE
    assert resolve_mode(config, {"QCAL_GUARD_MODE": " WARN "}) is Mode.WARN
    err = io.StringIO()
    assert (
        emit(
            Decision.allowed("x"),
            hook="h",
            mode=Mode.ENFORCE,
            config=config,
            stderr=err,
            environ={},
        )
        == 0
    )
    assert (
        emit(
            Decision.denied("why", "r"),
            hook="h",
            mode=Mode.ENFORCE,
            config=config,
            stderr=err,
            environ={},
        )
        == 2
    )
    assert "BLOCKED: why" in err.getvalue()


def test_debug_logging_reaches_stderr(repo: Path) -> None:
    _, err = call(["guard-paths"], write_payload(repo / "a.md"), repo, QCAL_DEBUG="1")
    assert "rule=paths.ok" in err
