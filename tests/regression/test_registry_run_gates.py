"""Regressions for the cycle-2026-10 audit, PR-A1 (registered-run gates).

A1: configuration from the environment changed a registered run without a trace.
A2: ``--experiments`` recorded one file's factors while the program ran another's.
A3: the experiment program re-read configuration that could differ from the launcher's.
A4: a misspelled policy key fell back to its default silently.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from qcal import cli
from qcal.config import ConfigError, load_config
from qcal.registry.gates import INPUTS_READ_KEY
from qcal_lab.experiment import RunRequest, run_experiment
from tests.conftest import REPO_ROOT, experiments_yaml, run_git, write
from tests.lab_support import fixture_project

pytestmark = pytest.mark.rule("C2", "C7")


def _run(root: Path, *args: str, environ: dict[str, str]) -> int:
    return cli.main(["--root", str(root), *args], out=io.StringIO(), environ=environ)


def test_a1_environment_overrides_are_refused_and_logging_ones_recorded(repo: Path) -> None:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    script = repo / "prog.py"
    script.write_text(
        "import json, sys\n"
        "ENVELOPE = {'format': 'qcal.executor_result', 'version': 1, 'status': 'ok'}\n"
        "json.dump({**ENVELOPE, 'metrics': {'AP': 1.0}}, open(sys.argv[1], 'w'))\n"
    )
    (repo / "qcal.toml").write_text(
        f'[executor]\ncommand = ["{sys.executable}", "{script}", "{{result_path}}"]\n'
        "[registry]\nenv_collectors = []\n"
    )
    refused = {"QCAL__PATHS__MANIFESTS_DIR": "/tmp/elsewhere"}
    assert _run(repo, "registry", "run", "C-a", "--seed", "0", environ=refused) == 2
    assert not (repo / "runs/registry").exists()
    neutral = {"QCAL__LOGGING__LEVEL": "WARNING"}
    out = io.StringIO()
    code = cli.main(
        ["--root", str(repo), "registry", "run", "C-a", "--seed", "0", "--json"],
        out=out,
        environ=neutral,
    )
    assert code == 0
    assert json.loads(out.getvalue())["provenance"]["config_environment"] == neutral


def test_a2_experiments_must_be_the_configured_file(repo: Path) -> None:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a", "detector": "atss"}]))
    write(repo, "other.yaml", experiments_yaml([{"id": "C-a", "detector": "swapped"}]))
    args = ("registry", "run", "C-a", "--seed", "0", "--experiments", str(repo / "other.yaml"))
    assert _run(repo, *args, environ={}) == 2


def test_a3_the_program_refuses_environment_configuration(tmp_path: Path) -> None:
    root, cells = fixture_project(tmp_path)
    request = RunRequest(root, "R-env", cells[0], 0, root / "runs/results/R-env.json")
    config = load_config(root, environ={"QCAL__DATA__SPLITS": '["val", "test"]'})
    with pytest.raises(ConfigError, match="QCAL__DATA__SPLITS overrides configuration"):
        run_experiment(request, qcal_config=config)


def test_a3_the_program_reports_what_it_read(tmp_path: Path) -> None:
    root, cells = fixture_project(tmp_path)
    request = RunRequest(root, "R-read", cells[0], 0, root / "runs/results/R-read.json")
    reported = run_experiment(request).environment[INPUTS_READ_KEY]
    assert set(reported) == {"EXPERIMENTS.yaml", "qcal.toml", "configs/lab.toml"}
    import hashlib

    for path, digest in reported.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path


def test_a4_a_misspelled_policy_key_fails_config_check(repo: Path) -> None:
    (repo / "qcal.toml").write_text('[signing]\nsigned_categorys = ["ian_only"]\n')
    assert _run(repo, "config", "--check", environ={}) == 1


def test_a4_the_repository_policy_passes_config_check() -> None:
    assert _run(REPO_ROOT, "config", "--check", environ={}) == 0


@pytest.mark.e2e
def test_a1_smoke_ignores_configuration_variables_in_the_callers_shell(tmp_path: Path) -> None:
    """``QCAL__REGISTRY__MAX_RUNS_PER_BATCH=1`` would refuse the six-cell batch if it leaked."""
    env = {
        **os.environ,
        "QCAL__REGISTRY__MAX_RUNS_PER_BATCH": "1",
        "QCAL__DATA__SPLITS": '["val"]',
    }
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "qcal_lab",
            "--root",
            str(REPO_ROOT),
            "smoke",
            "--json",
            "--workdir",
            str(tmp_path / "w"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env=env,
    )
    report = json.loads(out.stdout)
    assert out.returncode == 0, report


# --- wave 3: which digests tables compare (N4), one "hashed" predicate (N12) ------------------
# Proposal: docs/changes/qcal-foundations.md, step S5.


def _identity_rows(**differ: str) -> list[dict[str, str]]:
    base = {
        "config_inputs_sha256": "c" * 64,
        "definition_sha256": "d" * 64,
        "policy_sha256": "p" * 64,
        "experiments_sha256": "e" * 64,
        "env.handwritten_sha256": "h" * 64,
    }
    second = {**base, **differ}
    return [{"run_id": "R1", **base}, {"run_id": "R2", **second}]


SCIENCE = ("definition_sha256", "env.handwritten_sha256")
CORE = ("config_inputs_sha256", "definition_sha256", "policy_sha256", "experiments_sha256")


@pytest.mark.parametrize(
    ("differ", "identity", "refused"),
    [
        ({"policy_sha256": "q" * 64}, SCIENCE, False),  # a qcal.toml edit changes no number
        ({"experiments_sha256": "f" * 64}, SCIENCE, False),  # nor does a new cell
        ({"env.handwritten_sha256": "i" * 64}, SCIENCE, True),  # Ian's loop changed
        ({"definition_sha256": "x" * 64}, SCIENCE, True),
        ({"config_inputs_sha256": "x" * 64}, ("config_inputs_sha256",), True),  # the default
        ({"config_inputs_sha256": "x" * 64}, SCIENCE, False),  # narrowed: key is wired
    ],
)
def test_n4_tables_compare_the_configured_identity(
    differ: dict[str, str], identity: tuple[str, ...], refused: bool
) -> None:
    from qcal.integrity.aggregates import unsound_aggregate

    problem = unsound_aggregate(
        "mean", _identity_rows(**differ), environment_prefix="env.", identity=identity, core=CORE
    )
    assert (problem is not None) is refused, problem


def test_n4_a_stale_index_without_an_identity_column_is_refused() -> None:
    from qcal.integrity.aggregates import identity_problem

    rows = [{"run_id": "R1"}, {"run_id": "R2"}]
    assert "regenerate it" in str(identity_problem(rows, ["definition_sha256"], core=CORE))
    # a prefixed column no run reported is refused too: a misspelt identity column, or a
    # renamed environment key, would otherwise switch the comparison off (review B3)
    typo = identity_problem(rows, ["env.raw_prediction_sha256"], core=CORE)
    assert "no run in R1, R2 recorded env.raw_prediction_sha256" in str(typo)
    assert identity_problem(rows[:1], ["env.raw_prediction_sha256"], core=CORE) is None
    half = [{"run_id": "R1", "env.x": "a"}, {"run_id": "R2"}]  # one run did not report it
    assert "runs R2 recorded no env.x" in str(identity_problem(half, ["env.x"], core=CORE))


@pytest.mark.parametrize(
    "rows",
    [
        [{"run_id": "R1", "env.x": ""}, {"run_id": "R2", "env.x": ""}],  # a CSV index: "" == ""
        [{"run_id": "R1", "env.x": "a"}, {"run_id": "R2", "env.x": ""}],
        [{"run_id": "R1", "env.x": None}, {"run_id": "R2", "env.x": None}],  # a short CSV row
    ],
)
def test_s6_an_empty_identity_value_matches_nothing(rows: list[dict[str, Any]]) -> None:
    """Panel round 2: index rows hold "" for a value a run did not record, so two runs that
    both lack it compared equal, and the identity check passed on no evidence."""
    from qcal.integrity.aggregates import identity_problem

    problem = identity_problem(rows, ["env.x"], core=CORE)
    assert "recorded no env.x" in str(problem)
    assert identity_problem(rows[:1], ["env.x"], core=CORE) is None  # one run: nothing to compare
    both_empty = rows[0].get("env.x") in ("", None) and rows[1].get("env.x") in ("", None)
    allowed = identity_problem(rows, ["env.x"], core=CORE, nullable=["env.x"])
    assert (allowed is None) is both_empty  # nullable: empty is a value, compared as such


def test_s6_identity_nullable_must_name_identity_columns(make_config) -> None:
    from qcal.config import config_key_problems
    from qcal.integrity.aggregates import identity_nullable

    config = make_config('[registry]\nidentity_nullable = ["env.handwriten_sha256"]\n')
    with pytest.raises(ConfigError, match=r"env\.handwriten_sha256"):
        identity_nullable(config)
    problems = config_key_problems({"registry": {"identity_nullable": ["env.typo"]}})
    assert any("env.typo" in p and "identity_nullable" in p for p in problems)
    assert identity_nullable(make_config("")) == ()


def test_n4_identity_columns_must_exist_in_an_index(make_config) -> None:
    from qcal.config import config_key_problems
    from qcal.integrity.aggregates import identity_columns

    config = make_config('[registry]\nidentity_columns = ["definiton_sha256"]\n')
    with pytest.raises(ConfigError, match="definiton_sha256"):
        identity_columns(config)
    problems = config_key_problems({"registry": {"identity_columns": ["definiton_sha256"]}})
    assert any("definiton_sha256" in p for p in problems)
    assert config_key_problems({"registry": {"identity_columns": ["env.splits"]}}) == []


def test_n4_the_definition_digest_is_the_same_for_every_seed(make_config) -> None:
    from qcal.registry.experiments import load_experiments
    from qcal.registry.runner import Runner
    from qcal.registry.store import RegistryStore
    from tests.conftest import FakeExecutor

    config = make_config("")
    write(
        config.root,
        "EXPERIMENTS.yaml",
        experiments_yaml(
            [{"id": "C-a", "detector": "atss"}, {"id": "C-b", "detector": "yolox"}], seeds=[0, 1]
        ),
    )
    runner = Runner(
        config,
        load_experiments(config),
        RegistryStore(config.path("registry_dir")),
        FakeExecutor(),
        collectors=[],
    )
    a0, a1, b0 = runner.run("C-a", 0), runner.run("C-a", 1), runner.run("C-b", 0)
    assert a0.provenance["definition_sha256"] == a1.provenance["definition_sha256"]
    assert a0.provenance["config_hash"] != a1.provenance["config_hash"]
    assert a0.provenance["definition_sha256"] != b0.provenance["definition_sha256"]


def test_n4_the_audit_reports_what_tables_refuse_and_what_they_accept(make_config) -> None:
    from qcal.registry.audit import mixed_inputs
    from tests.conftest import make_record

    def record(run_id: str, **provenance: str) -> object:
        base = {"config_inputs_sha256": "c" * 64, "policy_sha256": "p" * 64}
        return make_record(run_id, provenance={**base, **provenance})

    policy = [record("R1"), record("R2", policy_sha256="q" * 64)]
    assert mixed_inputs(policy, "ok")  # reported by the audit, though tables accept it
    loop = [record("R1"), record("R2")]
    loop[1].environment["handwritten_sha256"] = "i" * 64  # type: ignore[attr-defined]
    assert mixed_inputs(loop, "ok") == []
    assert mixed_inputs(
        loop, "ok", identity=["env.handwritten_sha256"], prefixes={"environment": "env."}
    )


# N12: require_hashed and config_inputs share one definition of "hashed"


def _hash_tree(root: Path) -> None:
    write(root, "configs/lab.toml", "a = 1\n")
    write(root, "configs/LAB2.TOML", "a = 2\n")  # the glob is case-sensitive
    write(root, "configs/nested/x.yml", "x: 1\n")
    write(root, "configs/.hidden.toml", "h = 1\n")
    write(root, "configs/tables/main.toml", 'name = "t"\n')  # a table spec: never hashed
    write(root, "configs/notes.md", "not a config\n")
    (root / "configs/link.toml").symlink_to(root / "configs/tables/main.toml")


def test_n12_config_inputs_selects_exactly_these_files(make_config) -> None:
    from qcal.registry.gates import config_inputs

    config = make_config("")
    _hash_tree(config.root)
    # link.toml resolves into the table specs, so it is left out with them
    assert sorted(config_inputs(config)) == [
        "configs/.hidden.toml",
        "configs/lab.toml",
        "configs/nested/x.yml",
    ]


@pytest.mark.parametrize(
    ("path", "hashed"),
    [
        ("configs/lab.toml", True),
        ("configs/LAB2.TOML", False),
        ("configs/tables/main.toml", False),
        ("configs/notes.md", False),
        ("configs/absent.toml", False),
        ("configs/link.toml", False),
    ],
)
def test_n12_hashed_input_is_membership_in_config_inputs(
    make_config, path: str, hashed: bool
) -> None:
    from qcal.registry.gates import config_inputs, hashed_input

    config = make_config("")
    _hash_tree(config.root)
    assert hashed_input(config, path) is hashed
    assert hashed_input(config, config.root / path) is hashed
    assert (path in config_inputs(config)) is hashed
    assert hashed_input(config, "/etc/passwd") is False


# --- S6c: the launcher takes the code's identity from git, not from the program -------------


@pytest.fixture
def coded(git_repo: Path) -> Path:
    """A repository whose policy names src/pkg as the code a run executes, committed."""
    write(git_repo, "src/pkg/model.py", "WEIGHT = 1\n")
    write(git_repo, "qcal.toml", '[registry]\ncode_tree_paths = ["src/pkg"]\n')
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "code")
    return git_repo


def test_s6_the_code_tree_is_read_from_head(coded: Path) -> None:
    from qcal.registry.gates import code_tree, code_tree_digest

    config = load_config(coded, environ={})
    trees, problems = code_tree(config)
    assert problems == []
    assert trees == {"src/pkg": run_git(coded, "rev-parse", "HEAD:src/pkg")}
    assert len(code_tree_digest(trees)) == 64
    assert code_tree_digest({}) == ""  # nothing configured: an empty identity, refused (S6b)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda root: write(root, "src/pkg/model.py", "WEIGHT = 2\n"), "uncommitted changes"),
        (lambda root: write(root, "src/pkg/extra.py", "X = 1\n"), "uncommitted changes"),
        (
            lambda root: write(root, "qcal.toml", '[registry]\ncode_tree_paths = ["src/gone"]\n'),
            "src/gone is not in HEAD",
        ),
    ],
    ids=["edited", "untracked", "missing"],
)
def test_s6_a_registered_run_refuses_code_git_does_not_describe(
    coded: Path, change: Any, message: str
) -> None:
    from qcal.registry.gates import code_tree

    change(coded)
    config = load_config(coded, environ={})
    assert any(message in p for p in code_tree(config)[1])


def test_s6_the_run_gates_include_the_code_tree(coded: Path) -> None:
    from qcal.registry.experiments import load_experiments
    from qcal.registry.gates import policy_state, run_input_problems
    from tests.conftest import experiments_yaml

    write(coded, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    run_git(coded, "add", "-A")
    run_git(coded, "commit", "-q", "-m", "pre-register")
    config = load_config(coded, environ={})
    experiments = load_experiments(config)
    assert not [
        p for p in run_input_problems(config, experiments, policy_state(config)) if "src/pkg" in p
    ]
    write(coded, "src/pkg/model.py", "WEIGHT = 3\n")
    problems = run_input_problems(config, experiments, policy_state(config))
    assert any("src/pkg has uncommitted changes" in p for p in problems)


def test_s6_the_runner_records_the_code_tree(coded: Path) -> None:
    from qcal.registry.experiments import load_experiments
    from qcal.registry.gates import code_tree, code_tree_digest
    from qcal.registry.runner import Runner
    from qcal.registry.store import RegistryStore
    from tests.conftest import FakeExecutor

    write(coded, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    run_git(coded, "add", "-A")
    run_git(coded, "commit", "-q", "-m", "pre-register")
    config = load_config(coded, environ={})
    runner = Runner(
        config,
        load_experiments(config),
        RegistryStore(config.path("registry_dir")),
        FakeExecutor(),
        collectors=[],
    )
    record = runner.run("C-a", 0)
    trees = code_tree(config)[0]
    assert record.provenance["code_tree"] == trees
    assert record.provenance["code_tree_sha256"] == code_tree_digest(trees) != ""
    assert record.flat({})["code_tree_sha256"] == code_tree_digest(trees)  # an index column
