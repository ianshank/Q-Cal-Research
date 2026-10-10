"""``python -m qcal_lab``: run, smoke, status and splits, with qcal's exit codes."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from qcal_lab import cli
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
    assert set(data) == {"metrics", "artifacts", "environment"}


def test_run_failures_exit_1_and_write_nothing(project: tuple[Path, list[str]]) -> None:
    root, _ = project
    result = root / "runs/results/R1.json"
    args = ["--root", str(root), "run", "--run-id", "R1", "--cell-id", "C-x", "--seed", "0"]
    assert lab(*args, "--result-path", str(result))[0] == cli.EXIT_FAILED
    assert not result.exists()


def test_configuration_errors_exit_2(project: tuple[Path, list[str]]) -> None:
    root, cells = project
    (root / "configs/lab.toml").write_text("[broken\n")
    args = ["--root", str(root), "run", "--run-id", "R1", "--cell-id", cells[0], "--seed", "0"]
    assert lab(*args, "--result-path", str(root / "r.json"))[0] == cli.EXIT_USAGE


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
