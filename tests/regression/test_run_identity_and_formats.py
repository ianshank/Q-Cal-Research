"""Wave-3 follow-ups on the lab side (docs/changes/qcal-foundations.md, F3).

N1: a run killed outright still leaves a readable failed envelope.
N4: the identity tables compare is the same for every seed of one cell.
N12: the lab's "hashed" is the registry's.
The cache follow-ups (N2, N14, N15) have their own tests next to their code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qcal.config import ConfigError, load_config
from qcal_lab.config import load_lab_config, require_hashed
from qcal_lab.experiment import UNFINISHED_KIND, RunRequest, run_experiment
from tests.conftest import REPO_ROOT, write
from tests.lab_support import fixture_project

pytestmark = pytest.mark.rule("C1", "C7")


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, list[str]]:
    return fixture_project(tmp_path)


def _request(root: Path, cell: str, seed: int, run_id: str) -> RunRequest:
    return RunRequest(root, run_id, cell, seed, root / "runs/results" / f"{run_id}.json")


def _environment_value(environment: dict[str, object], column: str) -> object:
    return environment.get(column.removeprefix("env."))


def test_n4_the_repositorys_identity_set_agrees_across_seeds(
    project: tuple[Path, list[str]],
) -> None:
    """No seed-dependent value may be an identity column, or every cell would be refused."""
    root, cells = project
    text = (root / "EXPERIMENTS.yaml").read_text().replace("seeds:\n- 0\n", "seeds:\n- 0\n- 1\n")
    (root / "EXPERIMENTS.yaml").write_text(text)
    cell = next(c for c in cells if "platt" in c)  # its seed draws the fit subset
    first = run_experiment(_request(root, cell, 0, "R-a"))
    second = run_experiment(_request(root, cell, 1, "R-b"))
    assert first.environment["fit_draw"]["sha256"] != second.environment["fit_draw"]["sha256"]
    columns = load_config(REPO_ROOT).str_list("registry.identity_columns")
    env_columns = [c for c in columns if c.startswith("env.")]
    assert "env.raw_predictions_sha256" in env_columns
    for column in env_columns:
        a, b = (
            _environment_value(first.environment, column),
            _environment_value(second.environment, column),
        )
        assert a is not None, column
        assert a == b, column


def test_n4_raw_predictions_are_digested_per_split(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    result = run_experiment(_request(root, cells[0], 0, "R-a"))
    digests = result.environment["raw_predictions_sha256"]
    raw = {
        a["kind"]: a["sha256"] for a in result.artifacts if a["kind"].startswith("predictions_raw_")
    }
    assert {f"predictions_raw_{split}": sha for split, sha in digests.items()} == raw


def test_n1_a_run_killed_outright_leaves_a_failed_envelope(
    project: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """After every artifact the run writes a pessimistic envelope; SIGKILL cannot run the
    except-clause that writes the real one, so this is what the launcher reads."""
    from qcal_lab import experiment

    root, cells = project
    request = _request(root, cells[0], 0, "R-killed")

    def killed(*_: object, **__: object) -> None:
        raise SystemExit("killed")  # stands in for SIGKILL: nothing after this runs

    monkeypatch.setattr(experiment, "train_calibration", killed)
    with pytest.raises(SystemExit):
        run_experiment(request)
    envelope = json.loads(request.result_path.read_text())
    assert (envelope["status"], envelope["failure_kind"]) == ("failed", UNFINISHED_KIND)
    assert envelope["metrics"] == {}
    kinds = [a["kind"] for a in envelope["artifacts"]]
    assert len(kinds) == 3  # one raw predictions file per role, each listed as it was written
    assert all(kind.startswith("predictions_raw_") for kind in kinds)
    environment = envelope["environment"]
    assert {"program_python", "handwritten_sha256", "qcal_lab_source_sha256"} <= set(environment)
    assert environment["inputs_read"]


def test_n1_a_finished_run_replaces_the_pessimistic_envelope(
    project: tuple[Path, list[str]],
) -> None:
    from qcal_lab.experiment import write_result

    root, cells = project
    request = _request(root, cells[0], 0, "R-a")
    result = run_experiment(request)
    assert json.loads(request.result_path.read_text())["status"] == "failed"  # until the end
    write_result(request.result_path, result)
    assert json.loads(request.result_path.read_text())["status"] == "ok"


def test_n12_the_lab_refuses_a_file_the_registry_does_not_hash(
    project: tuple[Path, list[str]],
) -> None:
    """Table specs are never hashed; a lab configuration under them must be refused."""
    root, _ = project
    with (root / "qcal.toml").open("a") as handle:
        handle.write('\n[paths]\ntable_specs_dir = "configs"\n')
    qcal_config = load_config(root, environ={})
    with pytest.raises(ConfigError, match=r"not covered by registry\.config_hash_inputs"):
        require_hashed(load_lab_config(root), qcal_config)


def test_n12_a_case_mismatched_name_is_refused(project: tuple[Path, list[str]]) -> None:
    root, _ = project
    write(root, "configs/LAB.TOML", (root / "configs/lab.toml").read_text())
    lab = load_lab_config(root, root / "configs/LAB.TOML")
    with pytest.raises(ConfigError, match="not covered"):
        require_hashed(lab, load_config(root, environ={}))
