"""One regression test (or a few) per finding of the Phase 0 peer review.

Each test names its finding number. A failure here means a fixed defect came back.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from qcal.ci.immutability import check_registry_immutable
from qcal.ci.signatures import RefError, policy_config, verify_signatures
from qcal.cli import main
from qcal.config import Config, ConfigError, load_config
from qcal.hooks import cli as hook_cli
from qcal.integrity.claims import check_claims
from qcal.registry.audit import audit
from qcal.registry.executor import ExecutionResult, RunSpec
from qcal.registry.experiments import load_experiments
from qcal.registry.index import write_index
from qcal.registry.records import RunRecord
from qcal.registry.runner import Runner, RunRefusedError
from qcal.registry.store import RegistryStore
from qcal.registry.tables import TableDataError, build_tables, load_specs
from tests.conftest import (
    FakeExecutor,
    experiments_yaml,
    make_record,
    run_git,
    sign,
    write,
)

# --- finding 1: a pull request that is merely behind base passes ------------------------------


@pytest.mark.requires_ssh_keygen
def test_1_stale_pull_request_is_not_blamed_for_base_changes(signed_repo) -> None:
    repo, key, _old, base = signed_repo
    run_git(repo, "checkout", "-q", "-b", "feature")
    write(repo, "README.md", "feature docs\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "unsigned docs change")
    head = run_git(repo, "rev-parse", "HEAD")
    run_git(repo, "checkout", "-q", "main")
    write(repo, "EXPERIMENTS.yaml", "version: 3\n")
    run_git(repo, "add", "-A")
    sign(repo, key, "signed amendment on base after the branch point")
    new_base = run_git(repo, "rev-parse", "HEAD")

    report = verify_signatures(repo, new_base, head)

    assert report.passed, report.violations
    assert report.to_dict()["checked_net_paths"] == 0
    assert base != new_base


@pytest.mark.requires_ssh_keygen
def test_1_merge_rollback_is_still_caught_against_the_fork_point(signed_repo) -> None:
    repo, _key, old, base = signed_repo
    write(repo, "README.md", "x\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "unsigned")
    tip = run_git(repo, "rev-parse", "HEAD")
    tree = run_git(repo, "rev-parse", f"{old}^{{tree}}")
    merge = run_git(repo, "commit-tree", tree, "-p", tip, "-p", old, "-m", "merge old history")

    assert not verify_signatures(repo, base, merge).passed


# --- finding 2: rendered numbers cannot hide from the claims checker --------------------------

SECTION = "paper/sections/results.tex"


@pytest.fixture
def claims_config(repo: Path) -> Config:
    config = load_config(repo, environ={})
    store = RegistryStore(config.path("registry_dir"))
    store.write(make_record("R1", metrics={"AP": 41.2}))
    write_index(config, store)
    return config


@pytest.mark.parametrize(
    ("line", "fragment"),
    [
        (r"\qcalval{run:R1:AP}{41.2 (vs. 97.3)}", "must be exactly one number"),
        (r"See \url{97.3} for details.", "number 97.3 has no run reference"),
        (r"As shown \citep[Table 2: 97.3]{smith}.", "number 97.3 has no run reference"),
        (r"an improvement of .5\% overall", "number .5 has no run reference"),
        (r"a gain of .5 points", "number .5 has no run reference"),
    ],
    ids=["second-number-in-value", "url", "citation-note", "leading-dot-percent", "points"],
)
def test_2_rendered_numbers_are_checked(claims_config: Config, line: str, fragment: str) -> None:
    write(claims_config.root, SECTION, line + "\n")

    findings = check_claims(claims_config)

    assert any(fragment in f.message for f in findings), [f.message for f in findings]


@pytest.mark.parametrize("value", ["41.2", "$41.2$", r"41.2\,\%", "41.2~%", " 41.2 ", r"41.2\%"])
def test_2_legitimate_value_markup_still_passes(claims_config: Config, value: str) -> None:
    write(claims_config.root, SECTION, rf"AP is \qcalval{{run:R1:AP}}{{{value}}}." + "\n")

    assert check_claims(claims_config) == []


# --- finding 3: table specs select runs by pre-registered factors only ------------------------


@pytest.mark.parametrize("key", ["metric.AP", "env.gpu"])
@pytest.mark.parametrize("place", ["rows", "filter"])
def test_3_table_specs_cannot_select_by_metric_or_environment(
    config: Config, key: str, place: str
) -> None:
    rows = f'["{key}"]' if place == "rows" else '["detector"]'
    extra = f'\n[filter]\n"{key}" = [50.0, 60.0]\n' if place == "filter" else "\n"
    write(
        config.root,
        "configs/tables/t.toml",
        f'name = "t"\nrows = {rows}{extra}[[columns]]\nmetric = "AP"\n',
    )

    with pytest.raises(ConfigError, match="may use only factors columns"):
        load_specs(config)


# --- finding 4: one current run per (cell, seed) --------------------------------------------


@pytest.fixture
def runner(repo: Path) -> Runner:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0, 1]))
    config = load_config(repo, environ={})
    store = RegistryStore(config.path("registry_dir"))
    return Runner(config, load_experiments(config), store, FakeExecutor(), collectors=[])


def test_4_second_run_of_a_completed_pair_is_refused(runner: Runner) -> None:
    first = runner.run("C-a", 0)

    with pytest.raises(RunRefusedError, match=f"already completed as {first.run_id}"):
        runner.run("C-a", 0)


def test_4_supersede_must_name_a_current_run_of_the_same_pair(runner: Runner) -> None:
    other = runner.run("C-a", 1)

    with pytest.raises(RunRefusedError, match="not a current run of C-a@0"):
        runner.run("C-a", 0, supersedes=other.run_id)
    with pytest.raises(RunRefusedError, match="not a current run"):
        runner.run("C-a", 0, supersedes="R-does-not-exist")


def test_4_duplicate_seeds_in_a_batch_are_planned_once(runner: Runner) -> None:
    planned, _ = runner.plan("C-a", seeds=[0, 0, 1, 0])

    assert [p.seed for p in planned] == [0, 1]


def test_4_audit_and_tables_refuse_duplicate_current_runs(repo: Path) -> None:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0, 1]))
    config = load_config(repo, environ={})
    store = RegistryStore(config.path("registry_dir"))
    for run_id in ("R1", "R2"):  # written around the runner, as a hand-made record would be
        store.write(make_record(run_id, metrics={"AP": 40.0 if run_id == "R1" else 60.0}))
    store.write(make_record("R3", seed=1, metrics={"AP": 50.0}))
    write_index(config, store)
    write(
        repo,
        "configs/tables/t.toml",
        'name = "t"\nrows = ["detector"]\n[[columns]]\nmetric = "AP"\n',
    )

    report = audit(config, load_experiments(config), store.load_all())
    assert report.duplicates == ["C-a@0: R1, R2"]
    assert not report.ok()
    with pytest.raises(TableDataError, match="R1 and R2 are both current runs of C-a@0"):
        build_tables(config)


def test_4_audit_flags_supersedes_that_point_nowhere_or_elsewhere(config: Config) -> None:
    write(config.root, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0, 1]))
    records = [
        make_record("R1"),
        make_record("R2", seed=1, supersedes="R1"),  # different seed
        make_record("R3", supersedes="R-typo"),
    ]

    report = audit(config, load_experiments(config), records)

    assert report.bad_supersedes == [
        "R2 (C-a@1) supersedes R1 (C-a@0)",
        "R3 supersedes unknown R-typo",
    ]
    assert not report.ok()


# --- finding 6: a bad policy ref is an error, not a silent downgrade -------------------------


def test_6_unknown_policy_ref_is_a_usage_error(git_repo: Path) -> None:
    with pytest.raises(RefError, match="policy ref 'mian' does not name a commit"):
        policy_config(git_repo, "mian")
    err = io.StringIO()
    code = main(
        [
            "--root",
            str(git_repo),
            "ci",
            "verify-signatures",
            "--base",
            "HEAD",
            "--head",
            "HEAD",
            "--policy-ref",
            "mian",
        ],
        out=err,
    )
    assert code == 2


def test_6_missing_policy_file_falls_back_with_a_warning(
    git_repo: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (git_repo / "qcal.toml").unlink()
    run_git(git_repo, "commit", "-q", "-am", "drop policy")
    with caplog.at_level(logging.WARNING, logger="qcal"):
        config = policy_config(git_repo, "HEAD")
    assert config.str_value("signing.mode") == "bootstrap"
    assert "judging with the packaged defaults" in caplog.text


# --- finding 10: the Stop hook fails open on its own argument errors -------------------------


@pytest.mark.parametrize(("fail_open", "code"), [("1", 0), ("0", 2), (None, 2)])
def test_10_hook_argument_errors_follow_the_fail_mode(fail_open: str | None, code: int) -> None:
    env = {} if fail_open is None else {hook_cli.FAIL_OPEN_ENV: fail_open}
    err = io.StringIO()

    assert hook_cli.main(["claimz"], stdin=io.StringIO("{}"), stderr=err, environ=env) == code
    assert "arguments are invalid" in err.getvalue()


def test_10_hook_help_exits_cleanly() -> None:
    assert hook_cli.main(["--help"], stdin=io.StringIO(""), stderr=io.StringIO(), environ={}) == 0


# --- finding 12: agents cannot overwrite Ian's documents with templates ----------------------


@pytest.mark.parametrize(
    "command",
    [
        "qcal init --force",
        "python -m qcal init --force",
        "cd /x && qcal --root . init --dry-run --force",
    ],
)
def test_12_init_force_is_denied_to_agents(config: Config, command: str) -> None:
    from qcal.hooks.guards import guard_bash
    from qcal.hooks.payload import HookPayload
    from qcal.policy import Policy

    payload = HookPayload.parse(
        json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    )
    decision = guard_bash(payload, config=config, policy=Policy.from_config(config))

    assert decision.rule == "bash.extra"


@pytest.mark.parametrize("command", ["qcal init", "qcal init --dry-run", "make init"])
def test_12_plain_init_is_allowed(config: Config, command: str) -> None:
    from qcal.hooks.guards import guard_bash
    from qcal.hooks.payload import HookPayload
    from qcal.policy import Policy

    payload = HookPayload.parse(
        json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    )

    assert guard_bash(payload, config=config, policy=Policy.from_config(config)).allow


# --- finding 15: exit codes, debug, and executor crashes --------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["registry", "run-batch", "C-*", "--executor", "nope"],
        ["registry", "run", "C-a", "--seed", "0", "--executor", "nope"],
    ],
)
def test_15_unknown_executor_is_a_usage_error(repo: Path, argv: list[str]) -> None:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))

    assert main(["--root", str(repo), *argv], out=io.StringIO()) == 2


def test_15_debug_env_explains_a_config_that_fails_to_load(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "qcal.toml").write_text("[broken\n")
    monkeypatch.setenv("QCAL_DEBUG", "1")

    with pytest.raises(ConfigError):
        main(["--root", str(repo), "config"], out=io.StringIO())


class _Crashing:
    def execute(self, spec: RunSpec) -> ExecutionResult:
        raise OSError(f"cannot start {spec.run_id}")


def test_15_executor_crash_leaves_a_failed_record(repo: Path) -> None:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    config = load_config(repo, environ={})
    store = RegistryStore(config.path("registry_dir"))
    runner = Runner(config, load_experiments(config), store, _Crashing(), collectors=[])

    record = runner.run("C-a", 0)

    assert record.status == config.str_value("registry.failed_status")
    assert "executor raised OSError" in (record.error or "")
    assert [r.run_id for r in store.load_all()] == [record.run_id]


# --- finding 16: the registry directory holds only top-level records -------------------------


@pytest.fixture
def registry_repo(git_repo: Path) -> tuple[Path, str]:
    return git_repo, run_git(git_repo, "rev-parse", "HEAD")


def _add(repo: Path, relative: str, data: dict[str, Any] | str) -> str:
    text = data if isinstance(data, str) else json.dumps(data)
    write(repo, relative, text)
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", f"add {relative}")
    return run_git(repo, "rev-parse", "HEAD")


def _record(run_id: str, **overrides: Any) -> dict[str, Any]:
    return make_record(run_id, **overrides).to_dict()


@pytest.mark.parametrize(
    ("relative", "fragment"),
    [
        ("runs/registry/nested/R1.json", "records live directly in the registry directory"),
        ("runs/registry/notes.txt", "only <run_id>.json records"),
    ],
)
def test_16_junk_in_the_registry_is_a_violation(
    registry_repo: tuple[Path, str], relative: str, fragment: str
) -> None:
    repo, base = registry_repo
    head = _add(repo, relative, _record("R1"))

    report = check_registry_immutable(repo, base, head)

    assert not report.passed
    assert fragment in report.violations[0]


def test_16_supersedes_must_name_an_existing_run_of_the_same_pair(
    registry_repo: tuple[Path, str],
) -> None:
    repo, base = registry_repo
    _add(repo, "runs/registry/R1.json", _record("R1", seed=1))
    _add(repo, "runs/registry/R2.json", _record("R2", supersedes="R1"))
    head = _add(repo, "runs/registry/R3.json", _record("R3", supersedes="R-missing"))

    violations = check_registry_immutable(repo, base, head).violations

    assert violations == [
        "runs/registry/R2.json: supersedes 'R1' of C-a@1, not C-a@0",
        "runs/registry/R3.json: supersedes 'R-missing', which is not in the registry",
    ]


def test_16_valid_supersede_passes(registry_repo: tuple[Path, str]) -> None:
    repo, base = registry_repo
    _add(repo, "runs/registry/R1.json", _record("R1"))
    head = _add(repo, "runs/registry/R2.json", _record("R2", supersedes="R1"))

    assert check_registry_immutable(repo, base, head).passed


# --- finding 17: former hardcoded values are configuration ------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "git.timeout_s",
        "registry.env_command_timeout_s",
        "executor.kind",
        "hooks.max_nesting_depth",
        "agent_layer.skill_required_keys",
    ],
)
def test_17_former_constants_are_configurable(config: Config, key: str) -> None:
    assert config.get(key) not in (None, "")


def test_17_git_timeout_reaches_git(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from qcal import gitutil

    seen: list[float] = []
    real_run = subprocess.run

    def spy(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(kwargs["timeout"])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(gitutil.subprocess, "run", spy)
    (git_repo / "qcal.toml").write_text("[git]\ntimeout_s = 7\n")
    main(["--root", str(git_repo), "policy", "list", "--category", "ian_only"], out=io.StringIO())

    assert seen
    assert set(seen) == {7.0}
    assert gitutil._timeout.get() == gitutil.DEFAULT_TIMEOUT_S  # restored after the command
    with pytest.raises(ValueError, match="must be positive"), gitutil.timeout_scope(0):
        pass


def test_17_record_type_is_unchanged() -> None:
    assert RunRecord.from_dict(make_record("R1").to_dict()) == make_record("R1")
