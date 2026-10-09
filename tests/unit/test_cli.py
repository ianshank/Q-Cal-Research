from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from qcal import cli
from qcal.registry.store import RegistryStore
from tests.conftest import (
    executor_command,
    experiment_script,
    experiments_yaml,
    make_record,
    run_git,
    write,
)


def qcal(repo: Path, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    code = cli.main(["--root", str(repo), *args], out=out)
    return code, out.getvalue()


def qcal_json(repo: Path, *args: str) -> tuple[int, object]:
    code, text = qcal(repo, *args, "--json")
    return code, json.loads(text)


@pytest.fixture
def registered(repo: Path) -> Path:
    write(
        repo,
        "EXPERIMENTS.yaml",
        experiments_yaml(
            [{"id": "C-a", "detector": "atss"}, {"id": "C-b", "detector": "yolox"}], seeds=[0, 1]
        ),
    )
    return repo


def test_init_creates_then_reports_existing(repo: Path) -> None:
    code, created = qcal_json(repo, "init")
    assert code == 0
    assert {a["action"] for a in created} == {"created"}
    _, again = qcal_json(repo, "init")
    assert {a["action"] for a in again} == {"exists"}
    assert "EXPERIMENTS.yaml" in qcal(repo, "init", "--dry-run")[1]


def test_audit_requires_the_preregistration(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = qcal(repo, "registry", "audit")
    assert code == cli.EXIT_USAGE
    assert "qcal init" in capsys.readouterr().err


def test_audit_reports_coverage(registered: Path) -> None:
    RegistryStore(registered / "runs/registry").write(make_record("R1", cell_id="C-a", seed=0))
    code, report = qcal_json(registered, "registry", "audit")
    assert code == 0
    assert report["expected"] == 4
    assert report["completed"] == 1
    assert qcal(registered, "registry", "audit", "--strict")[0] == cli.EXIT_FAILED
    assert "coverage: 1/4" in qcal(registered, "registry", "audit")[1]


def test_index_write_and_check(registered: Path) -> None:
    RegistryStore(registered / "runs/registry").write(make_record("R1"))
    assert qcal(registered, "registry", "index", "--check")[0] == cli.EXIT_FAILED
    code, text = qcal(registered, "registry", "index")
    assert code == 0
    assert "updated" in text
    assert "up to date" in qcal(registered, "registry", "index", "--check")[1]
    assert "unchanged" in qcal(registered, "registry", "index")[1]


def test_tables_without_specs(repo: Path) -> None:
    assert qcal(repo, "registry", "tables") == (0, "no table specs found\n")
    assert qcal_json(repo, "registry", "tables", "--check") == (0, [])


def test_tables_check_detects_stale_output(registered: Path) -> None:
    RegistryStore(registered / "runs/registry").write(make_record("R1"))
    qcal(registered, "registry", "index")
    write(
        registered,
        "configs/tables/t.toml",
        """
        name = "t"
        rows = ["detector"]
        [[columns]]
        metric = "LaECE0"
    """,
    )
    assert qcal(registered, "registry", "tables", "--check")[0] == cli.EXIT_FAILED
    assert qcal(registered, "registry", "tables")[0] == 0
    assert qcal(registered, "registry", "tables", "--check")[0] == 0


def test_cells_summary_emit_and_missing_design(repo: Path) -> None:
    write(
        repo,
        "EXPERIMENTS.yaml",
        experiments_yaml(
            [],
            design={
                "axes": {"detector": ["atss", "yolox"], "precision": ["fp32", "int8"]},
                "rules": [{"exclude": {"detector": "yolox", "precision": "int8"}}],
            },
        ),
    )
    code, summary = qcal_json(repo, "registry", "cells")
    assert code == 0
    assert summary["cells"] == 3
    code, emitted = qcal(repo, "registry", "cells", "--emit")
    assert emitted.startswith("cells:")
    assert emitted.count("id: C-") == 3
    assert "3 cells" in qcal(repo, "registry", "cells")[1]
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([]))
    assert qcal(repo, "registry", "cells")[0] == cli.EXIT_USAGE


def test_cells_warns_above_threshold(repo: Path, caplog: pytest.LogCaptureFixture) -> None:
    (repo / "qcal.toml").write_text("[experiments]\nmax_cells_warning = 1\n")
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([], design={"axes": {"a": [1, 2]}}))
    qcal(repo, "registry", "cells")
    assert "above the 1 warning threshold" in caplog.text


def test_run_batch_end_to_end_with_a_subprocess_executor(registered: Path, tmp_path: Path) -> None:
    command = json.dumps(executor_command(experiment_script(tmp_path)))
    (registered / "qcal.toml").write_text(
        f'[executor]\ncommand = {command}\n[registry]\nenv_collectors = ["python"]\n'
    )
    code, plan = qcal_json(registered, "registry", "run-batch", "C-*", "--dry-run")
    assert code == 0
    assert len(plan["planned"]) == 4
    assert plan["completed"] == []
    code, result = qcal_json(registered, "registry", "run-batch", "C-a", "--seeds", "0")
    assert code == 0
    assert len(result["completed"]) == 1
    code, text = qcal(registered, "registry", "run-batch", "C-a", "--seeds", "0")
    assert "already completed" in text
    code, record = qcal_json(registered, "registry", "run", "C-b", "--seed", "1")
    assert code == 0
    assert record["status"] == "ok"
    assert record["environment"]["trt_version"] == "test"


def test_run_reports_failure_exit_code(registered: Path, tmp_path: Path) -> None:
    failing = tmp_path / "fail.py"
    failing.write_text("raise SystemExit(3)\n")
    (registered / "qcal.toml").write_text(
        f"[executor]\ncommand = {json.dumps([sys.executable, str(failing)])}\n"
        "[registry]\nenv_collectors = []\n"
    )
    code, text = qcal(registered, "registry", "run", "C-a", "--seed", "0")
    assert code == cli.EXIT_FAILED
    assert "failed" in text
    assert qcal(registered, "registry", "run-batch", "C-b")[0] == cli.EXIT_FAILED


def test_run_without_executor_is_a_usage_error(
    registered: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert qcal(registered, "registry", "run", "C-a", "--seed", "0")[0] == cli.EXIT_USAGE
    assert "executor.command is empty" in capsys.readouterr().err


def test_claims_exit_codes(repo: Path) -> None:
    assert qcal(repo, "claims") == (0, "claims: PASS\n")
    write(repo, "CLAIMS.md", "Calibration improved by 1.5 points.\n")
    code, findings = qcal_json(repo, "claims")
    assert code == cli.EXIT_FAILED
    assert findings[0]["kind"] == "untagged"


def test_leakage_skip_pass_and_fail(repo: Path) -> None:
    assert "SKIP" in qcal(repo, "leakage", "--if-present")[1]
    assert qcal(repo, "leakage")[0] == cli.EXIT_FAILED
    for split, ids in {
        "trt_calib_images": "a",
        "calibrator_fit_split": "b",
        "val": "c",
        "test": "d",
    }.items():
        write(repo, f"data/manifests/{split}.txt", ids + "\n")
    assert qcal_json(repo, "leakage")[1]["verdict"] == "PASS"
    write(repo, "data/manifests/test.txt", "a\n")
    assert qcal(repo, "leakage")[0] == cli.EXIT_FAILED


def test_licenses_and_agent_layer(repo: Path) -> None:
    assert qcal(repo, "licenses", "--no-packages")[0] == 0
    write(repo, "src/pkg/x.py", "import ultralytics\n")
    code, text = qcal(repo, "licenses", "--no-packages")
    assert code == cli.EXIT_FAILED
    assert "ultralytics" in text
    assert "agent layer: PASS" in qcal(repo, "agent-layer")[1]
    write(repo, ".claude/agents/x.md", "---\nname: y\ndescription: d\n---\n")
    assert qcal(repo, "agent-layer")[0] == cli.EXIT_FAILED


def test_policy_list_and_check(git_repo: Path) -> None:
    write(git_repo, "CLAIMS.md", "x")
    write(git_repo, "docs/a.md", "x")
    run_git(git_repo, "add", "-A")
    assert qcal(git_repo, "policy", "list", "--category", "ian_only")[1].split() == ["CLAIMS.md"]
    _code, verdicts = qcal_json(git_repo, "policy", "check", "qcal.toml", "docs/nbcu.md")
    assert verdicts[0]["categories"] == ["enforcement_surface"]
    assert verdicts[1]["clean_room_hits"] == ["nbcu"]
    assert "unprotected" in qcal(git_repo, "policy", "check", "docs/a.md")[1]
    assert qcal(git_repo, "policy", "list", "--category", "nope")[0] == cli.EXIT_USAGE


def test_policy_list_without_git(repo: Path) -> None:
    write(repo, "DECISIONS.md", "x")
    assert qcal(repo, "policy", "list", "--category", "ian_only")[1].split() == ["DECISIONS.md"]


def test_config_command_prints_sources(repo: Path) -> None:
    data = json.loads(qcal(repo, "config")[1])
    assert data["root"] == str(repo.resolve())
    assert data["sources"][0] == "defaults"


def test_debug_reraises(repo: Path) -> None:
    with pytest.raises(Exception, match="does not exist"):
        cli.main(["--root", str(repo), "--debug", "registry", "audit"], out=io.StringIO())


def test_alias_entry_points(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    monkeypatch.setattr(sys, "argv", ["qcal-registry", "index"])
    assert cli.registry_main() == 0
    monkeypatch.setattr(sys, "argv", ["qcal-claims"])
    assert cli.claims_main() == 0
    assert cli.registry_main(["index", "--check"]) == 0


def test_hook_passthrough(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps({"tool_name": "Write", "tool_input": {"file_path": str(repo / "CLAIMS.md")}})
        ),
    )
    assert cli.main(["hook", "guard-paths"]) == 2


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert capsys.readouterr().out.startswith("qcal ")
