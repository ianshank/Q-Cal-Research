"""What a registered run may read: environment inputs, closed-world keys, the committed policy."""

from __future__ import annotations

import hashlib
import io
import json
import string
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qcal import cli
from qcal.config import (
    EMPTY_LIST_ELEMENTS,
    OPEN_TABLES,
    REMOVED_KEYS,
    Config,
    config_key_problems,
    environment_problems,
    load_config,
    load_defaults,
    run_neutral_environment,
    without_config_environment,
)
from qcal.registry.audit import audit, mixed_inputs
from qcal.registry.executor import ExecutionResult, RunSpec, SubprocessExecutor
from qcal.registry.experiments import Cell, load_experiments
from qcal.registry.gates import (
    INPUTS_READ_KEY,
    input_mismatches,
    launch_digests,
    policy_state,
    run_input_problems,
)
from qcal.registry.runner import Runner, RunRefusedError
from qcal.registry.store import RegistryStore
from tests.conftest import (
    REPO_ROOT,
    FakeExecutor,
    experiments_yaml,
    make_record,
    run_git,
    write,
)

pytestmark = pytest.mark.rule("C7")

CELLS = [{"id": "C-a", "detector": "atss"}, {"id": "C-b", "detector": "detr"}]


def runner_for(config: Config, executor: Any) -> Runner:
    write(config.root, "EXPERIMENTS.yaml", experiments_yaml(CELLS, seeds=[0, 1]))
    return Runner(
        config,
        load_experiments(config),
        RegistryStore(config.path("registry_dir")),
        executor,
        collectors=[],
    )


# -- environment inputs ------------------------------------------------------------------


def test_environment_inputs_record_overrides_and_locators(repo: Path) -> None:
    env = {
        "QCAL__LOGGING__LEVEL": "DEBUG",
        "QCAL__PATHS__MANIFESTS_DIR": "/elsewhere",
        "QCAL_ROOT": str(repo),
        "QCAL_CONFIG": str(repo / "qcal.toml"),
        "QCAL_DEBUG": "1",
        "HOME": "/home/x",
    }
    config = load_config(repo, environ=env)
    assert dict(config.environment_inputs) == {
        "QCAL_CONFIG": str(repo / "qcal.toml"),
        "QCAL_ROOT": str(repo),
        "QCAL__LOGGING__LEVEL": "DEBUG",
        "QCAL__PATHS__MANIFESTS_DIR": "/elsewhere",
    }
    assert config.path("manifests_dir") == Path("/elsewhere")
    assert load_config(repo, environ={}).environment_inputs == {}


_SECTION = st.text(alphabet=string.ascii_uppercase, min_size=1, max_size=8)


@given(st.dictionaries(st.tuples(_SECTION, _SECTION), st.integers(0, 9), max_size=6))
def test_every_override_is_captured_and_marks_the_sources(
    overrides: dict[tuple[str, str], int],
) -> None:
    env = {f"QCAL__{section}__{key}": str(value) for (section, key), value in overrides.items()}
    root = Path("/nonexistent-qcal-project")
    config = load_config(root, environ=env, use_repo_file=False)
    assert dict(config.environment_inputs) == env
    assert ("environment" in config.sources) == bool(env)


@pytest.mark.parametrize(
    ("env", "problem"),
    [
        ({"QCAL__PATHS__MANIFESTS_DIR": "/x"}, "QCAL__PATHS__MANIFESTS_DIR overrides"),
        ({"QCAL__SIGNING__MODE": '"bootstrap"'}, "QCAL__SIGNING__MODE overrides"),
        ({"QCAL_CONFIG": "/tmp/other.toml"}, "replaces qcal.toml"),
        ({"QCAL_ROOT": "/tmp"}, "names another root"),
    ],
)
def test_environment_problems_refuse_overrides_and_redirects(
    repo: Path, env: dict[str, str], problem: str
) -> None:
    config = load_config(repo, environ=env)
    problems = environment_problems(config)
    assert len(problems) == 1
    assert problem in problems[0]


def test_logging_overrides_and_matching_locators_are_run_neutral(repo: Path) -> None:
    env = {
        "QCAL__LOGGING__LEVEL": "DEBUG",
        "QCAL_ROOT": str(repo),
        "QCAL_CONFIG": str(repo / "qcal.toml"),
    }
    config = load_config(repo, environ=env)
    assert environment_problems(config) == []
    assert run_neutral_environment(config) == {"QCAL__LOGGING__LEVEL": "DEBUG"}


def test_without_config_environment_keeps_everything_else() -> None:
    env = {
        "QCAL__A__B": "1",
        "QCAL_ROOT": "/r",
        "QCAL_CONFIG": "/c",
        "QCAL_DEBUG": "1",
        "QCAL_LOG_LEVEL": "INFO",
        "PATH": "/bin",
    }
    assert without_config_environment(env) == {
        "QCAL_DEBUG": "1",
        "QCAL_LOG_LEVEL": "INFO",
        "PATH": "/bin",
    }


# -- closed-world configuration keys -------------------------------------------------------


def test_the_repository_configuration_passes_the_key_check() -> None:
    assert config_key_problems(load_config(REPO_ROOT, environ={}).data) == []


def test_packaged_defaults_pass_their_own_check() -> None:
    assert config_key_problems(load_defaults()) == []


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ({"signing": {"signed_categorys": ["ian_only"]}}, "unknown configuration key"),
        ({"nosuch": {"x": 1}}, "unknown configuration key 'nosuch'"),
        ({"registry": {"schema_version": 1}}, "registry.schema_version was removed"),
        ({"registry": {"require_clean_tree": "yes"}}, "must be a boolean, got a string"),
        ({"git": {"timeout_s": True}}, "must be a number, got a boolean"),
        ({"paths": {"experiments": 3}}, "must be a string, got a number"),
        ({"paths": "x"}, "must be a table, got a string"),
        ({"git": {"protected_branches": "main"}}, "must be a list, got a string"),
        ({"git": {"protected_branches": ["main", 1]}}, "must hold only a string items"),
        ({"executor": {"command": ["python3", 7]}}, "must hold only a string items"),
        ({"hooks": {"extra_bash_deny": ["x"]}}, "must hold only a table items"),
        ({"policy": {"categories": {"mine": "x/**"}}}, "must be a list, got a string"),
        ({"policy": {"categories": {"mine": [1]}}}, "must hold only a string items"),
        ({"policy": {"messages": {"mine": ["x"]}}}, "must be a string, got a list"),
    ],
)
def test_key_problems(override: dict[str, Any], problem: str) -> None:
    from qcal.config import deep_merge

    problems = config_key_problems(deep_merge(load_defaults(), override))
    assert len(problems) == 1, problems
    assert problem in problems[0]


def test_numbers_are_interchangeable_and_open_tables_accept_new_keys() -> None:
    from qcal.config import deep_merge

    data = deep_merge(
        load_defaults(),
        {
            "git": {"timeout_s": 12.5},
            "executor": {"timeout_s": 3600.0},
            "policy": {"categories": {"mine": ["x/**"]}, "messages": {"mine": "is mine"}},
            "review": {"reviewer_by_branch_prefix": {"codex/": "claude"}},
        },
    )
    assert config_key_problems(data) == []


def test_the_key_registry_names_only_real_keys() -> None:
    defaults = load_defaults()

    def exists(dotted: str) -> bool:
        node: Any = defaults
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return True

    assert all(exists(t) for t in OPEN_TABLES)
    assert all(exists(k) for k in EMPTY_LIST_ELEMENTS)
    assert not any(exists(k) for k in REMOVED_KEYS), "a removed key is still a default"


def _qcal(root: Path, *args: str, environ: dict[str, str] | None = None) -> tuple[int, str]:
    out = io.StringIO()
    code = cli.main(["--root", str(root), *args], out=out, environ=environ or {})
    return code, out.getvalue()


def test_config_check_command(repo: Path) -> None:
    assert _qcal(repo, "config", "--check") == (0, "config: PASS\n")
    (repo / "qcal.toml").write_text('[signing]\nsigned_categorys = ["ian_only"]\n')
    code, out = _qcal(repo, "config", "--check", "--json")
    assert code == 1
    report = json.loads(out)
    assert report["verdict"] == "FAIL"
    assert "signing.signed_categorys" in report["problems"][0]


def test_config_command_shows_environment_inputs(repo: Path) -> None:
    code, out = _qcal(repo, "config", environ={"QCAL__LOGGING__LEVEL": "WARNING"})
    assert code == 0
    assert json.loads(out)["environment_inputs"] == {"QCAL__LOGGING__LEVEL": "WARNING"}


# -- the committed policy ------------------------------------------------------------------


def test_policy_state_without_git(repo: Path) -> None:
    config = load_config(repo, environ={})
    digest = hashlib.sha256((repo / "qcal.toml").read_bytes()).hexdigest()
    assert policy_state(config) == type(policy_state(config))("worktree", digest)
    (repo / "qcal.toml").unlink()
    assert policy_state(load_config(repo, environ={})).source == "defaults"


@pytest.mark.integration
def test_policy_state_compares_the_work_tree_with_head(git_repo: Path) -> None:
    config = load_config(git_repo, environ={})
    state = policy_state(config)
    assert (state.source, state.problem) == ("head", None)
    (git_repo / "qcal.toml").write_text("[signing]\nmode = 'bootstrap'\n")
    assert "differs from HEAD" in str(policy_state(config).problem)
    run_git(git_repo, "commit", "-qam", "policy")
    assert policy_state(config).problem is None
    (git_repo / "qcal.toml").unlink()
    assert "differs from HEAD" in str(policy_state(config).problem)


def test_run_input_problems_refuse_another_pre_registration(repo: Path) -> None:
    config = load_config(repo, environ={})
    write(repo, "drafts/other.yaml", experiments_yaml(CELLS))
    experiments = load_experiments(config, repo / "drafts/other.yaml")
    problems = run_input_problems(config, experiments)
    assert len(problems) == 1
    assert "is not the configured" in problems[0]
    write(repo, "EXPERIMENTS.yaml", experiments_yaml(CELLS))
    assert run_input_problems(config, load_experiments(config)) == []


def test_input_mismatches() -> None:
    launch = {"qcal.toml": "a", "configs/lab.toml": "b"}
    assert input_mismatches(launch, None) == []
    assert input_mismatches(launch, {"qcal.toml": "a", "elsewhere.toml": "z"}) == []
    assert input_mismatches(launch, {"configs/lab.toml": "c"}) == [
        "configs/lab.toml changed while the run was in flight"
    ]


def test_launch_digests_cover_inputs_policy_and_pre_registration(
    make_config: Callable[[str], Config],
) -> None:
    config = make_config('[registry]\nconfig_hash_inputs = ["configs/*.toml"]\n')
    write(config.root, "configs/lab.toml", "x = 1\n")
    write(config.root, "EXPERIMENTS.yaml", experiments_yaml(CELLS))
    experiments = load_experiments(config)
    digests = launch_digests(config, experiments, policy_state(config))
    assert set(digests) == {"configs/lab.toml", "EXPERIMENTS.yaml", "qcal.toml"}
    assert digests["EXPERIMENTS.yaml"] == experiments.sha256


# -- the runner ----------------------------------------------------------------------------


def test_runner_refuses_environment_overrides_before_anything_runs(repo: Path) -> None:
    config = load_config(repo, environ={"QCAL__PATHS__MANIFESTS_DIR": "/elsewhere"})
    executor = FakeExecutor()
    runner = runner_for(config, executor)
    with pytest.raises(RunRefusedError, match="QCAL__PATHS__MANIFESTS_DIR overrides"):
        runner.run("C-a", 0)
    assert executor.specs == []
    assert runner.store.load_all() == []


def test_runner_records_run_neutral_overrides(repo: Path) -> None:
    config = load_config(repo, environ={"QCAL__LOGGING__LEVEL": "DEBUG"})
    record = runner_for(config, FakeExecutor()).run("C-a", 0)
    assert record.status == "ok"
    assert record.provenance["config_environment"] == {"QCAL__LOGGING__LEVEL": "DEBUG"}
    assert record.provenance["config_sources"] == [
        "defaults",
        str(repo / "qcal.toml"),
        "environment",
    ]


def test_a_file_that_changes_during_the_run_fails_it(repo: Path) -> None:
    config = load_config(repo, environ={})
    changed = ExecutionResult(
        0, metrics={"AP": 1.0}, environment={INPUTS_READ_KEY: {"EXPERIMENTS.yaml": "0" * 64}}
    )
    runner = runner_for(config, FakeExecutor([changed]))
    record = runner.run("C-a", 0)
    assert record.status == "failed"
    assert record.metrics == {}
    assert record.error == "EXPERIMENTS.yaml changed while the run was in flight"


def test_matching_input_digests_pass(repo: Path) -> None:
    config = load_config(repo, environ={})
    runner = runner_for(config, FakeExecutor())
    same = ExecutionResult(
        0,
        metrics={"AP": 1.0},
        environment={INPUTS_READ_KEY: {"EXPERIMENTS.yaml": runner.experiments.sha256}},
    )
    runner.executor = FakeExecutor([same])
    record = runner.run("C-a", 0)
    assert record.status == "ok"
    assert record.environment[INPUTS_READ_KEY] == {"EXPERIMENTS.yaml": runner.experiments.sha256}


# -- the executor's child environment ------------------------------------------------------


def test_the_child_never_sees_configuration_variables(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "QCAL__PATHS__MANIFESTS_DIR": "/x",
        "QCAL_CONFIG": "/c.toml",
        "QCAL_ROOT": "/r",
        "QCAL_DEBUG": "1",
    }.items():
        monkeypatch.setenv(name, value)
    executor = SubprocessExecutor(
        ["prog"],
        timeout_s=None,
        env_prefix="QCAL_RUN_",
        root=repo,
        pass_env={"QCAL__LOGGING__LEVEL": "DEBUG"},
    )
    spec = RunSpec("R1", Cell("C-a", {}), 0, "draw", repo / "r.json", repo / "r.log", repo)
    env = executor.environment(spec)
    assert not {"QCAL__PATHS__MANIFESTS_DIR", "QCAL_CONFIG", "QCAL_ROOT"} & set(env)
    assert env["QCAL_DEBUG"] == "1"
    assert env["QCAL__LOGGING__LEVEL"] == "DEBUG"
    assert env["QCAL_RUN_RUN_ID"] == "R1"


def test_from_config_passes_only_run_neutral_overrides(repo: Path) -> None:
    config = load_config(
        repo,
        environ={
            "QCAL__EXECUTOR__COMMAND": '["prog"]',
            "QCAL__LOGGING__FORMAT": "json",
        },
    )
    assert SubprocessExecutor.from_config(config).pass_env == {"QCAL__LOGGING__FORMAT": "json"}


# -- audit: seeds that ran under different configurations ----------------------------------


def _ok(run_id: str, seed: int, digest: str | None) -> Any:
    provenance = {} if digest is None else {"config_inputs_sha256": digest}
    return make_record(run_id, seed=seed, provenance=provenance)


def test_mixed_inputs_lists_cells_whose_seeds_differ() -> None:
    records = [_ok("R1", 0, "a"), _ok("R2", 1, "b"), _ok("R3", 2, None)]
    assert mixed_inputs(records, "ok") == ["C-a: 2 configurations (R1; R2)"]
    assert mixed_inputs([_ok("R1", 0, "a"), _ok("R2", 1, "a")], "ok") == []


def test_strict_audit_fails_on_mixed_inputs(repo: Path) -> None:
    config = load_config(repo, environ={})
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0, 1]))
    report = audit(config, load_experiments(config), [_ok("R1", 0, "a"), _ok("R2", 1, "b")])
    assert report.mixed_inputs
    assert report.ok()
    assert not report.ok(strict=True)


# -- fallbacks -------------------------------------------------------------------------------


def test_key_check_with_custom_defaults_skips_what_it_cannot_type() -> None:
    defaults = {"policy": {"categories": {}}, "x": {"y": []}}
    data = {"policy": {"categories": {"anything": 1}}, "x": {"y": [1, "a"]}}
    assert config_key_problems(data, defaults) == []


def test_git_environment_without_git_variables_drops_nothing() -> None:
    from qcal.gitutil import git_environment

    assert git_environment({"PATH": "/bin"}) == {"PATH": "/bin", "GIT_NO_REPLACE_OBJECTS": "1"}


def test_launch_digests_without_a_policy_file_or_an_in_repo_pre_registration(
    repo: Path, tmp_path: Path
) -> None:
    (repo / "qcal.toml").unlink()
    config = load_config(repo, environ={})
    outside = tmp_path / "elsewhere.yaml"
    outside.write_text(experiments_yaml(CELLS))
    experiments = load_experiments(config, outside)
    assert launch_digests(config, experiments, policy_state(config)) == {}


def test_inputs_read_without_a_policy_or_lab_file(tmp_path: Path) -> None:
    from qcal_lab.config import LabConfig, load_lab_config
    from qcal_lab.experiment import inputs_read
    from tests.lab_support import fixture_project

    root, _ = fixture_project(tmp_path)
    (root / "qcal.toml").unlink()
    config = load_config(root, environ={})
    lab = LabConfig(load_lab_config(root).config, None, "")
    reported = inputs_read(config, lab, load_experiments(config))
    assert set(reported) == {"EXPERIMENTS.yaml"}
