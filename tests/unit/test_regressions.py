"""Regression tests for defects found during self-review."""

from __future__ import annotations

from pathlib import Path

import pytest

from qcal.config import load_config
from qcal.registry.executor import read_result
from qcal.registry.experiments import load_experiments
from qcal.registry.index import write_index
from qcal.registry.records import RecordError
from qcal.registry.runner import Runner, RunRefusedError
from qcal.registry.store import RegistryStore
from qcal.registry.tables import build_tables
from tests.conftest import FakeExecutor, experiments_yaml, make_record, write


def test_boolean_factor_filters_match_the_index_rendering(repo: Path) -> None:
    config = load_config(repo, environ={})
    store = RegistryStore(config.path("registry_dir"))
    store.write(make_record("R1", factors={"detector": "atss", "tf32": False}))
    store.write(
        make_record("R2", factors={"detector": "atss", "tf32": True}, metrics={"LaECE0": 99.0})
    )
    write_index(config, store)
    write(
        repo,
        "configs/tables/t.toml",
        """
        name = "t"
        rows = ["detector"]
        [filter]
        tf32 = false
        [[columns]]
        metric = "LaECE0"
    """,
    )
    build_tables(config)
    table = (repo / "paper/tables/t.tex").read_text()
    assert r"\qcalval{run:R1:LaECE0}{12.50}" in table
    assert "R2" not in table


@pytest.mark.parametrize("name", ["AP:50", "a+b", "has space", "x{y}", "a,b"])
def test_metric_names_that_break_claim_references_are_rejected(name: str, tmp_path: Path) -> None:
    with pytest.raises(RecordError, match="metric name"):
        make_record(metrics={name: 1.0})
    result = tmp_path / "r.json"
    result.write_text(f'{{"metrics": {{"{name}": 1.0}}}}')
    assert "claim references" in (read_result(result, tmp_path).error or "")


def test_metric_names_with_safe_punctuation_are_accepted() -> None:
    assert make_record(metrics={"AP@50": 1.0, "LaECE0": 2.0, "D-ECE": 3.0}).metrics["AP@50"] == 1.0


def test_invalid_run_ids_are_refused_before_execution(repo: Path) -> None:
    write(
        repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C/a", "detector": "atss"}], seeds=[0])
    )
    config = load_config(repo, environ={})
    executor = FakeExecutor()
    runner = Runner(
        config,
        load_experiments(config),
        RegistryStore(config.path("registry_dir")),
        executor,
        collectors=[],
    )
    with pytest.raises(RunRefusedError, match="cannot run C/a"):
        runner.run("C/a", 0)
    assert executor.specs == []


def test_empty_registry_without_an_index_is_consistent(repo: Path) -> None:
    config = load_config(repo, environ={})
    store = RegistryStore(config.path("registry_dir"))
    assert write_index(config, store, check=True).changed is False
    assert write_index(config, store).changed is False
    assert not config.path("index_csv").exists()


def test_empty_registry_with_a_stale_index_is_reported(repo: Path) -> None:
    config = load_config(repo, environ={})
    write(repo, "runs/index.csv", "run_id\nR-old\n")
    assert (
        write_index(config, RegistryStore(config.path("registry_dir")), check=True).changed is True
    )


def _runner(repo: Path, executor: FakeExecutor) -> Runner:
    config = load_config(repo, environ={})
    store = RegistryStore(config.path("registry_dir"))
    return Runner(config, load_experiments(config), store, executor, collectors=[])


def test_failed_rerun_does_not_supersede_a_good_run(repo: Path) -> None:
    from qcal.registry.executor import ExecutionResult

    write(
        repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a", "detector": "atss"}], seeds=[0])
    )
    executor = FakeExecutor(
        results=[ExecutionResult(0, metrics={"AP": 1.0}), ExecutionResult(1, error="crashed")]
    )
    runner = _runner(repo, executor)
    first = runner.run_batch("C-a").completed[0]
    rerun = runner.run_batch("C-a", rerun=True)
    assert rerun.failed[0].supersedes is None
    assert runner.plan("C-a")[1][0][2] == f"already completed as {first.run_id}"


def test_provenance_is_captured_before_the_experiment_runs(repo: Path) -> None:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    config_file = write(repo, "configs/model.yaml", "lr: 0.1\n")

    class MutatingExecutor(FakeExecutor):
        def execute(self, spec):  # type: ignore[no-untyped-def]
            config_file.write_text("lr: 0.2\n")  # an experiment rewriting its config mid-run
            return super().execute(spec)

    runner = _runner(repo, MutatingExecutor())
    expected = runner.config_hash(runner.experiments.cell("C-a"), 0)
    record = runner.run("C-a", 0)
    assert record.provenance["config_hash"] == expected


def test_clean_tree_requirement_ignores_registry_outputs(git_repo: Path) -> None:
    from tests.conftest import run_git

    (git_repo / "qcal.toml").write_text("[registry]\nrequire_clean_tree = true\n")
    write(git_repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0, 1]))
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "prereg")
    batch = _runner(git_repo, FakeExecutor()).run_batch("C-a")
    assert len(batch.completed) == 2
    write(git_repo, "src/x.py", "x = 1\n")
    with pytest.raises(RunRefusedError, match="dirty"):
        _runner(git_repo, FakeExecutor()).run("C-a", 0)


# --- red-team findings -------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["runs/registry/R1.json", "runs/index.csv", "qcal/__init__.py", "setup.py", "conftest.py"],
)
def test_records_and_shadowing_files_require_a_signature(config, path: str) -> None:
    from qcal.policy import Policy

    assert Policy.from_config(config).in_categories(
        path, config.str_list("signing.signed_categories")
    )


def _claims(repo: Path, relative: str, text: str) -> list[str]:
    from qcal.integrity.claims import check_claims

    write(repo, relative, text)
    return [f.message for f in check_claims(load_config(repo, environ={}))]


@pytest.mark.parametrize(
    "text",
    ["Our method improves mAP by 7 points.", r"Accuracy rises 3\% on ID data.", "A gain of 2 pp."],
)
def test_whole_numbers_written_as_results_are_flagged(repo: Path, text: str) -> None:
    assert _claims(repo, "paper/sections/results.tex", text)


@pytest.mark.parametrize("text", ["See Table 2 and Section 3.", "We use 3 seeds on COCO 2017."])
def test_structural_whole_numbers_are_not_flagged(repo: Path, text: str) -> None:
    assert _claims(repo, "paper/sections/results.tex", text) == []


def test_escape_hatches_work_only_in_ian_only_files(repo: Path) -> None:
    hidden = "We reach 0.95 accuracy. % qcal:ignore\nWith \\qcalfixed{0.5} momentum.\n"
    assert len(_claims(repo, "paper/sections/results.tex", hidden)) == 2
    (repo / "paper/sections/results.tex").unlink()
    assert _claims(repo, "paper/sections/abstract.tex", hidden) == []


@pytest.mark.integration
@pytest.mark.requires_ssh_keygen
def test_signature_must_belong_to_the_committer(
    repo: Path, ssh_keygen: str, tmp_path: Path
) -> None:
    import subprocess

    from qcal.ci.signatures import verify_signatures
    from tests.conftest import run_git

    key = tmp_path / "key"
    subprocess.run([ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    pub = (tmp_path / "key.pub").read_text().strip()
    (repo / "allowed_signers").write_text(f'ian@example.invalid namespaces="git" {pub}\n')
    (repo / "qcal.toml").write_text('[signing]\nmode = "enforce"\n')
    run_git(repo, "init", "-q", "-b", "main")
    run_git(repo, "add", "-A")
    run_git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")
    base = run_git(repo, "rev-parse", "HEAD")
    write(repo, "CLAIMS.md", "x\n")
    run_git(repo, "add", "-A")
    run_git(
        repo,
        "-c",
        "gpg.format=ssh",
        "-c",
        f"user.signingkey={key}",
        "-c",
        "user.email=attacker@example.invalid",
        "commit",
        "-q",
        "-S",
        "-m",
        "signed by ian's key",
    )
    head = run_git(repo, "rev-parse", "HEAD")
    report = verify_signatures(repo, base, head)
    assert not report.passed
    assert "committed as" in report.violations[0].detail
