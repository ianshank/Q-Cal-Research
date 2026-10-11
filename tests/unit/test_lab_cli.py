"""``python -m qcal_lab``: run, smoke, status and splits, with qcal's exit codes."""

from __future__ import annotations

import io
import json
import os
import signal
import time
from pathlib import Path

import pytest

from qcal.config import ConfigError
from qcal.registry.experiments import ExperimentsError
from qcal.registry.records import FAILURE_KINDS
from qcal_lab import cli
from qcal_lab.experiment import PlanError
from tests.conftest import write
from tests.lab_support import coco_doc, fixture_project


def lab(*args: str) -> tuple[int, str]:
    out = io.StringIO()
    return cli.main(list(args), out=out), out.getvalue()


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, list[str]]:
    return fixture_project(tmp_path)


def test_usage_errors_exit_2() -> None:
    assert lab()[0] == cli.EXIT_USAGE
    assert lab("run", "--run-id", "R")[0] == cli.EXIT_USAGE


def test_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert lab("--help")[0] == cli.EXIT_OK
    assert "smoke" in capsys.readouterr().out


def test_run_writes_the_result_file(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    result = root / "runs/results/R1.json"
    args = ["--root", str(root), "run", "--run-id", "R1", "--cell-id", cells[0], "--seed", "0"]
    assert lab(*args, "--result-path", str(result))[0] == cli.EXIT_OK
    data = json.loads(result.read_text())
    assert (data["format"], data["version"], data["status"]) == ("qcal.executor_result", 1, "ok")
    assert set(data) == {
        "format",
        "version",
        "status",
        "failure_kind",
        "error",
        "metrics",
        "artifacts",
        "environment",
    }


def test_run_failures_exit_1_and_write_a_failed_envelope(project: tuple[Path, list[str]]) -> None:
    root, _ = project
    result = root / "runs/results/R1.json"
    args = ["--root", str(root), "run", "--run-id", "R1", "--cell-id", "C-x", "--seed", "0"]
    assert lab(*args, "--result-path", str(result))[0] == cli.EXIT_FAILED
    data = json.loads(result.read_text())
    assert (data["status"], data["failure_kind"], data["metrics"]) == ("failed", "plan", {})
    assert "C-x" in data["error"]


def test_configuration_errors_exit_2(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    (root / "configs/lab.toml").write_text("[broken\n")
    args = ["--root", str(root), "run", "--run-id", "R1", "--cell-id", cells[0], "--seed", "0"]
    assert lab(*args, "--result-path", str(root / "r.json"))[0] == cli.EXIT_USAGE
    assert json.loads((root / "r.json").read_text())["failure_kind"] == "config"


def test_status(project: tuple[Path, list[str]], repo: Path) -> None:
    root, _ = project
    code, text = lab("--root", str(root), "status")
    assert code == cli.EXIT_OK
    assert "TODO oracle parity cases: none yet" in text
    code, text = lab("--root", str(root), "status", "--json", "--strict")
    assert code == cli.EXIT_FAILED
    items = {i["name"]: i["ready"] for i in json.loads(text)["items"]}
    assert items["evaluation loop (Ian)"] is True  # the smoke project uses the fixture stand-in
    assert items["split manifests"] is True
    assert items["detector atss_r50"] is False
    code, text = lab("--root", str(repo), "status")
    assert "TODO evaluation loop (Ian): qcal_lab.handwritten.eval_loop not written" in text


def test_splits_import_partition_and_verify(repo: Path) -> None:
    annotations = write(repo, "ann.json", json.dumps(coco_doc(images=10)))
    root = ["--root", str(repo), "splits"]
    code, text = lab(*root, "import", "--annotations", str(annotations), "--split", "val")
    assert code == cli.EXIT_OK
    assert text.startswith("val: 10 ids, sha256 ")
    sizes = "calibrator_fit_split=3,test=4"
    partition = [*root, "partition", "--annotations", str(annotations), "--seed", "1"]
    assert lab(*partition, "--sizes", sizes)[0] == cli.EXIT_OK
    assert lab(*partition, "--sizes", "x=1,y")[0] == cli.EXIT_FAILED
    # The import put every image in val, so val overlaps both partitions: verify fails on it.
    code, text = lab(*root, "verify", "--annotations", str(annotations))
    assert code == cli.EXIT_FAILED
    assert "trt_calib_images: no manifest" in text
    assert "overlap calibrator_fit_split&val: 3 id(s)" in text
    # An existing split is never replaced from the command line (split shopping).
    code, _ = lab(
        *root,
        "import",
        "--annotations",
        str(write(repo, "b.json", json.dumps(coco_doc(images=4)))),
        "--split",
        "val",
    )
    assert code == cli.EXIT_FAILED
    assert (
        lab(*root, "import", "--annotations", str(annotations), "--split", "val", "--replace")[0]
        == cli.EXIT_USAGE
    )
    smaller = write(repo, "small.json", json.dumps(coco_doc(images=2)))
    code, text = lab(*root, "verify", "--annotations", str(smaller))
    assert code == cli.EXIT_FAILED
    assert "not in small.json" in text


def test_smoke_json_with_a_kept_workdir(tmp_path: Path, repo: Path) -> None:
    code, text = lab("--root", str(repo), "smoke", "--json", "--workdir", str(tmp_path / "w"))
    report = json.loads(text)
    assert code == cli.EXIT_OK, report
    assert report["verdict"] == "PASS"
    assert (tmp_path / "w/project/runs/index.csv").is_file()


def test_smoke_refuses_to_reuse_a_project(tmp_path: Path, repo: Path) -> None:
    (tmp_path / "w/project").mkdir(parents=True)
    (tmp_path / "w/project/qcal.toml").write_text("# someone's project\n")
    code, _ = lab("--root", str(repo), "smoke", "--workdir", str(tmp_path / "w"))
    assert code == cli.EXIT_USAGE
    assert (tmp_path / "w/project/qcal.toml").read_text() == "# someone's project\n"


def test_smoke_text_in_a_temporary_directory(repo: Path) -> None:
    code, text = lab("--root", str(repo), "smoke")
    assert code == cli.EXIT_OK
    assert text.startswith("smoke: PASS")


# -- failure kinds and SIGTERM (PR-A2) ----------------------------------------------------


class OutOfMemoryError(RuntimeError):  # stands in for torch.cuda.OutOfMemoryError
    pass


class AcceleratorError(RuntimeError):  # torch >= 2.8
    pass


class CudaError(RuntimeError):
    pass


class DeviceAssertError(CudaError):  # matched through its base class
    pass


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (KeyboardInterrupt(), "interrupted"),
        (cli.RunInterrupted("signal 15"), "interrupted"),
        (ConfigError("x"), "config"),
        (PlanError("x"), "plan"),
        (ExperimentsError("x"), "plan"),
        (MemoryError(), "host_oom"),
        (OutOfMemoryError("CUDA out of memory"), "cuda_oom"),
        (AcceleratorError("x"), "cuda_error"),
        (DeviceAssertError("x"), "cuda_error"),
        (ValueError("x"), "unknown"),
    ],
)
def test_failure_kind_names_what_ended_the_run(exc: BaseException, kind: str) -> None:
    assert cli.failure_kind(exc) == kind
    assert kind in FAILURE_KINDS


def test_sigterm_becomes_an_interrupt_and_the_handler_is_restored() -> None:
    before = signal.getsignal(signal.SIGTERM)

    def terminated() -> None:
        with cli._sigterm_interrupts():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)  # the handler raises long before this ends

    with pytest.raises(cli.RunInterrupted):
        terminated()
    assert signal.getsignal(signal.SIGTERM) == before


def test_an_unexpected_error_writes_unknown_then_propagates(
    project: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cells = project

    def boom(*_args: object, **_kwargs: object) -> None:
        raise ZeroDivisionError("a bug")

    monkeypatch.setattr(cli, "run_experiment", boom)
    result = root / "runs/results/R1.json"
    args = ["--root", str(root), "run", "--run-id", "R1", "--cell-id", cells[0], "--seed", "0"]
    with pytest.raises(ZeroDivisionError):
        lab(*args, "--result-path", str(result))
    data = json.loads(result.read_text())
    assert (data["status"], data["failure_kind"]) == ("failed", "unknown")
    assert data["error"] == "ZeroDivisionError: a bug"


def test_a_failed_run_reports_the_artifacts_it_wrote(
    project: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cells = project

    def partial(request: object, *, progress: list[dict[str, str]]) -> None:
        progress.append({"path": "runs/results/R1/raw_val.jsonl", "kind": "predictions_raw_val"})
        raise MemoryError

    monkeypatch.setattr(cli, "run_experiment", partial)
    result = root / "runs/results/R1.json"
    args = ["--root", str(root), "run", "--run-id", "R1", "--cell-id", cells[0], "--seed", "0"]
    assert lab(*args, "--result-path", str(result))[0] == cli.EXIT_FAILED
    data = json.loads(result.read_text())
    assert data["failure_kind"] == "host_oom"
    assert data["artifacts"] == [
        {"path": "runs/results/R1/raw_val.jsonl", "kind": "predictions_raw_val"}
    ]
