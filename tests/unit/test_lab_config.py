"""qcal_lab.config: packaged defaults < configs/lab.toml, and the hashed-file rule."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from qcal.config import ConfigError, load_config
from qcal_lab.config import (
    LAB_CONFIG_FILE,
    load_lab_config,
    load_lab_defaults,
    parse_lab_config,
    require_hashed,
)
from tests.conftest import write


def test_defaults_load_without_a_repository_layer(repo: Path) -> None:
    lab = load_lab_config(repo)
    assert lab.path is None
    assert lab.sha256 == ""
    assert lab.config.str_value("splits.evaluate") == "test"
    assert lab.config.sources == ("qcal_lab defaults",)
    assert load_lab_defaults() is not load_lab_defaults()  # a fresh copy each time


def test_repository_layer_merges_over_defaults_and_is_hashed_by_bytes(repo: Path) -> None:
    body = b'[splits]\nselect = "val"\n[calibration]\ngrid_step = 0.1\r\n'
    (repo / "configs").mkdir()
    (repo / LAB_CONFIG_FILE).write_bytes(body)
    lab = load_lab_config(repo)
    assert lab.path == repo / LAB_CONFIG_FILE
    assert lab.sha256 == hashlib.sha256(body).hexdigest()
    assert lab.config.float_value("calibration.grid_step") == 0.1
    assert lab.config.float_value("calibration.grid_start") == 0.0  # untouched default


@pytest.mark.parametrize("body", [b"[splits\n", b"\xff\xfe"])
def test_unreadable_layers_are_configuration_errors(repo: Path, body: bytes) -> None:
    (repo / "configs").mkdir()
    (repo / LAB_CONFIG_FILE).write_bytes(body)
    with pytest.raises(ConfigError):
        load_lab_config(repo)


def test_text_and_file_settings(repo: Path) -> None:
    lab = parse_lab_config(repo, '[datasets.id]\nannotations = "a/b.json"\nimages_dir = "/abs"\n')
    assert lab.file("datasets.id.annotations") == repo.resolve() / "a/b.json"
    assert lab.file("datasets.id.images_dir") == Path("/abs")
    with pytest.raises(ConfigError, match=r"detectors\.atss_r50\.config is not set"):
        lab.text("detectors.atss_r50.config")


def test_require_hashed_accepts_files_under_configs(repo: Path) -> None:
    write(repo, LAB_CONFIG_FILE, "")
    require_hashed(load_lab_config(repo), load_config(repo, environ={}))
    require_hashed(load_lab_config(repo / "nowhere"), load_config(repo, environ={}))


def test_require_hashed_refuses_a_file_the_registry_would_not_hash(repo: Path) -> None:
    write(repo, "lab.toml", "")
    lab = load_lab_config(repo, repo / "lab.toml")
    with pytest.raises(ConfigError, match=r"not covered by registry\.config_hash_inputs"):
        require_hashed(lab, load_config(repo, environ={}))


def test_require_hashed_refuses_a_file_outside_the_repository(repo: Path, tmp_path: Path) -> None:
    outside = write(tmp_path, "elsewhere/lab.toml", "")
    lab = load_lab_config(repo, outside)
    with pytest.raises(ConfigError, match="outside the repository"):
        require_hashed(lab, load_config(repo, environ={}))


def test_environment_variables_never_change_science_settings(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QCAL__SPLITS__EVALUATE", '"val"')
    monkeypatch.setenv("QCAL_LAB__SPLITS__EVALUATE", '"val"')
    assert load_lab_config(repo).config.str_value("splits.evaluate") == "test"
