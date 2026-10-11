"""A lab-produced number, end to end: registered run → index → audit → table → claim.

The smoke test proves the pipeline runs; this proves its numbers can only reach a claim
through the registry. The project is the smoke test's fixture project, committed to git (so
the registered-run gates apply as in the real repository), driven through the real ``qcal``
CLI in subprocesses with ``python -I``.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from qcal_lab.config import load_lab_config
from qcal_lab.smoke import CELL_PREFIX, write_project
from tests.conftest import REPO_ROOT, run_git, write

pytestmark = [pytest.mark.e2e, pytest.mark.rule("C1", "C7")]

METRIC = "smoke_mean_abs_gap"


def qcal(project: Path, *args: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    """Exit code and stdout (stderr is appended only when the command failed)."""
    out = subprocess.run(
        [sys.executable, "-I", "-m", "qcal", "--root", str(project), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=project,
        env=env,
    )
    return out.returncode, out.stdout if out.returncode == 0 else out.stdout + out.stderr


def qcal_json(project: Path, *args: str) -> tuple[int, object]:
    out = subprocess.run(
        [sys.executable, "-I", "-m", "qcal", "--root", str(project), *args, "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=project,
    )
    return out.returncode, json.loads(out.stdout)


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("lab-loop") / "project"
    write_project(root, load_lab_config(REPO_ROOT), sys.executable)
    write(root, ".gitignore", "runs/\n")
    write(
        root,
        "configs/tables/smoke.toml",
        f"""
        [[table]]
        name = "smoke"
        rows = ["calibrator", "calibrator_scope"]
        caption = "Synthetic fixture (no scientific content)"
        label = "tab:smoke"
        [[table.columns]]
        metric = "{METRIC}"
        digits = 4
        """,
    )
    run_git(root, "init", "-q", "-b", "main")
    run_git(root, "add", "-A")
    run_git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "fixture project")
    code, out = qcal(root, "registry", "run-batch", f"{CELL_PREFIX}*")
    assert code == 0, out
    return root


def _index(project: Path) -> list[dict[str, str]]:
    with (project / "runs/index.csv").open() as handle:
        return list(csv.DictReader(handle))


def test_registered_runs_index_audit_and_tabulate(project: Path) -> None:
    assert qcal(project, "registry", "index")[0] == 0
    code, report = qcal_json(project, "registry", "audit", "--strict")
    assert code == 0, report
    assert isinstance(report, dict)
    assert report["completed"] == report["expected"] == 6
    assert report["mixed_inputs"] == []
    assert qcal(project, "registry", "tables")[0] == 0
    table = (project / "paper/tables/smoke.tex").read_text()
    # One seed per cell: each table cell cites its single run directly.
    assert table.count(r"\qcalval{run:") == 6
    assert table.count(f":{METRIC}}}{{") == 6
    assert qcal(project, "claims")[0] == 0
    assert qcal(project, "registry", "tables", "--check")[0] == 0
    assert qcal(project, "registry", "index", "--check")[0] == 0


def test_every_record_carries_the_committed_policy(project: Path) -> None:
    assert qcal(project, "registry", "index")[0] == 0
    records = [
        json.loads(p.read_text()) for p in sorted((project / "runs/registry").glob("*.json"))
    ]
    assert len(records) == 6
    for record in records:
        provenance = record["provenance"]
        assert provenance["policy_source"] == "head"
        assert provenance["git_dirty"] is False
        assert provenance["config_environment"] == {}
        assert set(record["environment"]["inputs_read"]) >= {"EXPERIMENTS.yaml", "qcal.toml"}


def test_a_claim_citing_a_run_verifies_and_a_wrong_one_fails(project: Path) -> None:
    assert qcal(project, "registry", "index")[0] == 0
    row = _index(project)[0]
    value = float(row[f"metric.{METRIC}"])
    claims = project / "CLAIMS.md"
    claims.write_text(f"Gap {value:.4f} (run:{row['run_id']}:{METRIC}).\n")
    code, out = qcal(project, "claims")
    assert code == 0, out
    claims.write_text(f"Gap {value + 0.5:.4f} (run:{row['run_id']}:{METRIC}).\n")
    code, findings = qcal_json(project, "claims")
    assert code == 1
    assert isinstance(findings, list)
    assert findings[0]["kind"] == "mismatch"
    claims.unlink()


def test_tampering_with_a_table_or_a_record_is_caught(project: Path) -> None:
    assert qcal(project, "registry", "index")[0] == 0
    assert qcal(project, "registry", "tables")[0] == 0
    table_path = project / "paper/tables/smoke.tex"
    table = table_path.read_text()
    start = table.index("}{", table.index(r"\qcalval{run:")) + 2
    end = table.index("}", start)
    table_path.write_text(table[:start] + "9.9999" + table[end:])
    assert qcal(project, "claims")[0] == 1
    table_path.write_text(table)

    record_path = min((project / "runs/registry").glob("*.json"))
    original = record_path.read_text()
    record = json.loads(original)
    record["metrics"][METRIC] = record["metrics"][METRIC] + 1.0
    record_path.write_text(json.dumps(record))
    assert qcal(project, "registry", "index", "--check")[0] == 1
    record_path.write_text(original)
    assert qcal(project, "registry", "index", "--check")[0] == 0


def test_the_registry_refuses_inputs_that_are_not_committed(project: Path) -> None:
    import os

    cell = f"{CELL_PREFIX}isotonic-per_class"
    args = ("registry", "run", cell, "--seed", "0", "--supersedes", "x", "--reason", "y")
    env = {**os.environ, "QCAL__PATHS__MANIFESTS_DIR": "/tmp/elsewhere"}
    code, out = qcal(project, *args, env=env)
    assert code == 2
    assert "QCAL__PATHS__MANIFESTS_DIR overrides configuration" in out
    policy = project / "qcal.toml"
    original = policy.read_text()
    policy.write_text(original + "\n# an uncommitted edit\n")
    code, out = qcal(project, *args)
    assert code == 2
    assert "qcal.toml differs from HEAD" in out
    policy.write_text(original)
