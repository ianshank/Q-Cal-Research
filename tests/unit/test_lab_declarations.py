"""Ian's declarations (review B1): read only from a file agents cannot write."""

from __future__ import annotations

from pathlib import Path

import pytest

from qcal.config import Config, load_config
from qcal_lab.declarations import (
    DeclarationsError,
    declarations_path,
    detector_declarations,
    load_declarations,
)
from tests.conftest import REPO_ROOT, write

pytestmark = pytest.mark.rule("C0", "C4")

DECLARED = '[detectors.atss_r50]\nzoo_ap = 0.5\ntolerance = 0.01\nsource = "card"\n'


@pytest.fixture
def project(tmp_path: Path) -> Config:
    write(tmp_path, "qcal.toml", (REPO_ROOT / "qcal.toml").read_text("utf-8"))
    return load_config(tmp_path, environ={})


def _tooling(file: str) -> dict[str, object]:
    return {"declarations": {"file": file}}


def test_the_configured_file_is_ians_data_and_signed() -> None:
    path = declarations_path(load_config(REPO_ROOT, environ={}))
    assert path == REPO_ROOT / "docs/reference/declarations.toml"


@pytest.mark.parametrize(
    "file",
    [
        "configs/declarations.toml",  # agents maintain configs/
        "src/qcal_lab/resources/declarations.toml",  # agent-owned package
        "",
        "/etc/declarations.toml",
        "docs/reference/../../configs/x.toml",
    ],
)
def test_a_file_agents_could_write_is_refused(project: Config, file: str) -> None:
    with pytest.raises(DeclarationsError):
        declarations_path(project, _tooling(file))


def test_a_missing_file_declares_nothing(project: Config) -> None:
    assert load_declarations(project, _tooling("docs/reference/declarations.toml")) == {}
    assert detector_declarations(project, "atss_r50", _tooling("docs/reference/d.toml")) == {}


def test_declarations_are_read_per_detector(project: Config) -> None:
    write(project.root, "docs/reference/declarations.toml", DECLARED)
    tooling = _tooling("docs/reference/declarations.toml")
    declared = detector_declarations(project, "atss_r50", tooling)
    assert dict(declared) == {"zoo_ap": 0.5, "tolerance": 0.01, "source": "card"}
    assert detector_declarations(project, "other", tooling) == {}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[detectors]\natss_r50 = 1\n", "must be a table"),
        ("not toml =", "not valid TOML"),
    ],
)
def test_a_malformed_file_is_refused(project: Config, text: str, message: str) -> None:
    write(project.root, "docs/reference/declarations.toml", text)
    with pytest.raises(DeclarationsError, match=message):
        detector_declarations(project, "atss_r50", _tooling("docs/reference/declarations.toml"))


def test_an_unreadable_file_is_refused(project: Config) -> None:
    (project.root / "docs/reference/declarations.toml").mkdir(parents=True)  # a directory
    with pytest.raises(DeclarationsError, match="cannot read"):
        load_declarations(project, _tooling("docs/reference/declarations.toml"))
