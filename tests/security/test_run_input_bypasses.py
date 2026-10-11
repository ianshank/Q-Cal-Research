"""Attacks on what a registered run reads (cycle 2026-10, PR-A1).

Each test tries one way to make a registered run use configuration that is not the
committed one: an environment override, a locator variable, an uncommitted ``qcal.toml``,
another pre-registration file, or git itself pointed somewhere else. Every attempt must be
refused before the experiment starts, with nothing written to the registry.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from qcal import cli
from qcal.registry.store import RegistryStore
from tests.conftest import executor_command, experiment_script, experiments_yaml, run_git, write

pytestmark = [pytest.mark.integration, pytest.mark.rule("C7")]


@pytest.fixture
def qcal(capsys: pytest.CaptureFixture[str]) -> Callable[..., tuple[int, str, str]]:
    """Run the qcal CLI in-process with an explicit environment; returns (code, out, err)."""

    def run(root: Path, *args: str, env: dict[str, str] | None = None) -> tuple[int, str, str]:
        out = io.StringIO()
        capsys.readouterr()
        code = cli.main(["--root", str(root), *args], out=out, environ=env or {})
        return code, out.getvalue(), capsys.readouterr().err

    return run


QcalRun = Callable[..., tuple[int, str, str]]


@pytest.fixture
def project(git_repo: Path, tmp_path: Path) -> Path:
    """A committed project with a working experiment program, a clean tree and two cells."""
    command = json.dumps(executor_command(experiment_script(tmp_path)))
    (git_repo / "qcal.toml").write_text(
        f"[executor]\ncommand = {command}\n"
        "[registry]\nrequire_clean_tree = true\nenv_collectors = []\n"
    )
    write(git_repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}, {"id": "C-b"}]))
    write(git_repo, ".gitignore", "runs/\n")
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-qm", "pre-register")
    return git_repo


def registry_is_empty(root: Path) -> bool:
    return RegistryStore(root / "runs/registry").load_all() == []


def test_the_honest_run_succeeds(qcal: QcalRun, project: Path) -> None:
    code, out, err = qcal(project, "registry", "run", "C-a", "--seed", "0")
    assert code == 0, err
    assert out.strip().endswith(": ok")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("QCAL__PATHS__MANIFESTS_DIR", "/tmp/other-manifests"),
        ("QCAL__REGISTRY__CONFIG_HASH_INPUTS", '["nothing/*.toml"]'),
        ("QCAL__EXECUTOR__COMMAND", '["python3", "-c", "pass"]'),
        ("QCAL__POLICY__CATEGORIES__IAN_ONLY", '["nothing"]'),
        ("QCAL__REGISTRY__REQUIRE_CLEAN_TREE", "false"),
        ("QCAL__SIGNING__MODE", '"bootstrap"'),
    ],
)
def test_environment_overrides_are_refused(
    qcal: QcalRun, project: Path, name: str, value: str
) -> None:
    code, _, err = qcal(project, "registry", "run", "C-a", "--seed", "0", env={name: value})
    assert code == 2
    assert f"{name} overrides configuration" in err
    assert registry_is_empty(project)


def test_qcal_config_cannot_swap_the_policy_file(
    qcal: QcalRun, project: Path, tmp_path: Path
) -> None:
    other = tmp_path / "permissive.toml"
    other.write_text((project / "qcal.toml").read_text().replace("true", "false"))
    code, _, err = qcal(
        project, "registry", "run", "C-a", "--seed", "0", env={"QCAL_CONFIG": str(other)}
    )
    assert code == 2
    assert "replaces qcal.toml" in err
    assert registry_is_empty(project)


def test_qcal_root_cannot_disagree_with_root(qcal: QcalRun, project: Path, tmp_path: Path) -> None:
    code, _, err = qcal(
        project, "registry", "run", "C-a", "--seed", "0", env={"QCAL_ROOT": str(tmp_path)}
    )
    assert code == 2
    assert "names another root" in err


def test_an_uncommitted_policy_is_refused(qcal: QcalRun, project: Path) -> None:
    text = (
        (project / "qcal.toml")
        .read_text()
        .replace("require_clean_tree = true", "require_clean_tree = false")
    )
    (project / "qcal.toml").write_text(text)
    code, _, err = qcal(project, "registry", "run", "C-a", "--seed", "0")
    assert code == 2
    assert "qcal.toml differs from HEAD" in err
    assert registry_is_empty(project)


def test_another_pre_registration_is_refused_but_a_dry_run_is_not(
    qcal: QcalRun, project: Path
) -> None:
    write(project, "drafts/other.yaml", experiments_yaml([{"id": "C-a", "detector": "swapped"}]))
    args = ("--experiments", str(project / "drafts/other.yaml"))
    code, _, err = qcal(project, "registry", "run", "C-a", "--seed", "0", *args)
    assert code == 2
    assert "is not the configured" in err
    code, _, err = qcal(project, "registry", "run-batch", "C-*", "--dry-run", *args)
    assert code == 0, err
    assert registry_is_empty(project)


def test_a_misspelled_policy_key_is_refused(qcal: QcalRun, project: Path) -> None:
    with (project / "qcal.toml").open("a") as handle:
        handle.write('[signing]\nsigned_categorys = ["ian_only"]\n')
    run_git(project, "commit", "-qam", "typo")
    code, _, err = qcal(project, "registry", "run", "C-a", "--seed", "0")
    assert code == 2
    assert "unknown configuration key 'signing.signed_categorys'" in err


# -- git pointed elsewhere -------------------------------------------------------------------


def _sibling_with_policy(tmp_path: Path, text: str) -> Path:
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    run_git(sibling, "init", "-q", "-b", "main")
    (sibling / "qcal.toml").write_text(text)
    run_git(sibling, "add", "-A")
    run_git(sibling, "-c", "commit.gpgsign=false", "commit", "-qm", "permissive")
    return sibling


def test_git_dir_cannot_point_head_at_another_repository(
    qcal: QcalRun, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edited = (project / "qcal.toml").read_text().replace("true", "false")
    (project / "qcal.toml").write_text(edited)
    sibling = _sibling_with_policy(tmp_path, edited)
    monkeypatch.setenv("GIT_DIR", str(sibling / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(project))
    code, _, err = qcal(project, "registry", "run", "C-a", "--seed", "0")
    assert code == 2
    assert "qcal.toml differs from HEAD" in err


def test_git_replace_cannot_swap_the_committed_policy(qcal: QcalRun, project: Path) -> None:
    edited = (project / "qcal.toml").read_text().replace("true", "false")
    old_blob = run_git(project, "rev-parse", "HEAD:qcal.toml")
    new_blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        input=edited,
        cwd=project,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    run_git(project, "replace", old_blob, new_blob)
    assert run_git(project, "show", "HEAD:qcal.toml") == edited.strip()  # git itself is fooled
    (project / "qcal.toml").write_text(edited)
    code, _, err = qcal(project, "registry", "run", "C-a", "--seed", "0")
    assert code == 2
    assert "qcal.toml differs from HEAD" in err


def test_git_work_tree_cannot_hide_a_dirty_tree(
    qcal: QcalRun, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = tmp_path / "clean-export"
    shutil.copytree(project, clean, ignore=shutil.ignore_patterns(".git"))
    write(project, "notes.txt", "uncommitted\n")
    monkeypatch.setenv("GIT_WORK_TREE", str(clean))
    monkeypatch.setenv("GIT_DIR", str(project / ".git"))
    code, _, err = qcal(project, "registry", "run", "C-a", "--seed", "0")
    assert code == 2
    assert "working tree is dirty" in err


def test_git_config_injection_cannot_hide_a_dirty_tree(
    qcal: QcalRun, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(project, "notes.txt", "uncommitted\n")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "status.showUntrackedFiles")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "no")
    code, _, err = qcal(project, "registry", "run", "C-a", "--seed", "0")
    assert code == 2
    assert "working tree is dirty" in err


# -- the experiment program's own gate -------------------------------------------------------


def test_the_executor_child_sees_no_configuration_variables(
    qcal: QcalRun, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real subprocess: the launcher's shell exports overrides; the child must not see them."""
    dump = tmp_path / "dump_env.py"
    dump.write_text(
        "import json, os, sys\n"
        "json.dump({'metrics': {}, 'environment': {'seen': sorted(k for k in os.environ"
        " if k.startswith('QCAL'))}}, open(sys.argv[1], 'w'))\n"
    )
    command = json.dumps(["python3", str(dump), "{result_path}"])
    text = (project / "qcal.toml").read_text()
    _, _, rest = text.partition("\n[registry]")
    (project / "qcal.toml").write_text(f"[executor]\ncommand = {command}\n[registry]{rest}")
    run_git(project, "commit", "-qam", "dump the child environment")
    monkeypatch.setenv("QCAL_ROOT", str(project))
    code, out, err = qcal(
        project,
        "registry",
        "run",
        "C-a",
        "--seed",
        "0",
        "--json",
        env={"QCAL__LOGGING__LEVEL": "WARNING", "QCAL_ROOT": str(project)},
    )
    assert code == 0, err
    seen = json.loads(out)["environment"]["seen"]
    assert "QCAL_ROOT" not in seen
    assert "QCAL__LOGGING__LEVEL" in seen  # run-neutral overrides are passed on and recorded
    assert all(name.startswith(("QCAL_RUN_", "QCAL__LOGGING__")) for name in seen)
