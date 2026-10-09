"""The whole Phase 0 loop: pre-register, expand, run, index, audit, tabulate, verify, tamper."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml

from qcal import cli
from tests.conftest import executor_command, experiment_script, write

pytestmark = pytest.mark.integration


def qcal(repo: Path, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    return cli.main(["--root", str(repo), *args], out=out), out.getvalue()


def test_full_loop(repo: Path, tmp_path: Path) -> None:
    command = json.dumps(executor_command(experiment_script(tmp_path)))
    (repo / "qcal.toml").write_text(
        f'[executor]\ncommand = {command}\n[registry]\nenv_collectors = ["python", "packages"]\n'
    )

    # Ian creates the pre-registration from the template and fills in a design.
    assert qcal(repo, "init")[0] == 0
    data = yaml.safe_load((repo / "EXPERIMENTS.yaml").read_text())
    data.update(
        {
            "seeds": [0, 1],
            "seed_role": "calibrator_fit_draw",
            "design": {
                "axes": {"detector": ["atss", "yolox"], "precision": ["fp32", "int8"]},
                "rules": [{"when": {"precision": "fp32"}, "fix": {"note": "baseline"}}],
            },
        }
    )
    (repo / "EXPERIMENTS.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    code, emitted = qcal(repo, "registry", "cells", "--emit")
    data["cells"] = yaml.safe_load(emitted)["cells"]
    (repo / "EXPERIMENTS.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    assert len(data["cells"]) == 4

    # Runs, index, audit.
    assert qcal(repo, "registry", "run-batch", "C-*")[0] == 0
    assert qcal(repo, "registry", "index")[0] == 0
    code, report = qcal(repo, "registry", "audit", "--json")
    audit = json.loads(report)
    assert audit["completed"] == audit["expected"] == 8
    assert audit["placeholders"], "template placeholders are reported until Ian fills them"

    # Tables are generated, and every number in them verifies.
    write(
        repo,
        "configs/tables/main.toml",
        """
        [[table]]
        name = "main"
        rows = ["detector", "precision"]
        caption = "Synthetic"
        label = "tab:main"
        [[table.columns]]
        metric = "LaECE0"
        digits = 3
        [[table.columns]]
        metric = "LaECE0"
        agg = "std"
        digits = 3
    """,
    )
    assert qcal(repo, "registry", "tables")[0] == 0
    table = (repo / "paper/tables/main.tex").read_text()
    assert table.count(r"\qcalval{agg:mean:LaECE0:") == 4
    assert qcal(repo, "claims") == (0, "claims: PASS\n")
    assert qcal(repo, "registry", "tables", "--check")[0] == 0
    assert qcal(repo, "registry", "index", "--check")[0] == 0

    # Tampering with one number is caught.
    first = table.index(r"\qcalval{agg:mean")
    value_start = table.index("}{", first) + 2
    value_end = table.index("}", value_start)
    tampered = table[:value_start] + "0.000" + table[value_end:]
    (repo / "paper/tables/main.tex").write_text(tampered)
    code, findings = qcal(repo, "claims", "--json")
    assert code == cli.EXIT_FAILED
    assert json.loads(findings)[0]["kind"] == "mismatch"
    assert qcal(repo, "registry", "tables", "--check")[0] == cli.EXIT_FAILED

    # A rerun supersedes the old records; claims that cite superseded runs fail.
    assert qcal(repo, "registry", "run-batch", "C-*", "--seeds", "0", "--rerun")[0] == 0
    assert qcal(repo, "registry", "index")[0] == 0
    (repo / "paper/tables/main.tex").write_text(table)
    code, findings = qcal(repo, "claims", "--json")
    assert code == cli.EXIT_FAILED
    assert any("superseded" in f["message"] for f in json.loads(findings))
    assert qcal(repo, "registry", "tables")[0] == 0
    assert qcal(repo, "claims")[0] == 0
