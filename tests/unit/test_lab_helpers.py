"""Lab helpers that only the pipeline exercised before (cycle 2026-10 audit)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from qcal_lab import experiment
from qcal_lab.models.mmdet import set_tf32, tf32_flags
from qcal_lab.predictions import CacheEntry
from tests.lab_support import predictions_header


def fake_torch(*, matmul: bool, cudnn: bool) -> SimpleNamespace:
    return SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=matmul)),
            cudnn=SimpleNamespace(allow_tf32=cudnn),
        )
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_set_tf32_switches_both_backends(enabled: bool) -> None:
    torch = fake_torch(matmul=not enabled, cudnn=not enabled)
    set_tf32(torch, enabled=enabled)
    assert tf32_flags(torch) == {"allow_tf32_matmul": enabled, "allow_tf32_cudnn": enabled}


def test_tf32_flags_report_mixed_states() -> None:
    assert tf32_flags(fake_torch(matmul=True, cudnn=False)) == {
        "allow_tf32_matmul": True,
        "allow_tf32_cudnn": False,
    }


def test_source_digest_covers_code_and_settings_but_not_ians_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "qcal_lab"
    (package / "handwritten").mkdir(parents=True)
    (package / "experiment.py").write_text("x = 1\n")
    (package / "defaults.toml").write_text("a = 1\n")
    (package / "notes.md").write_text("ignored\n")
    (package / "handwritten/eval_loop.py").write_text("Ian's\n")
    monkeypatch.setattr(experiment, "__file__", str(package / "experiment.py"))

    def digest() -> str:
        experiment.source_digest.cache_clear()  # cached per process: the source cannot change
        return experiment.source_digest()

    first = digest()
    (package / "handwritten/eval_loop.py").write_text("Ian's, edited\n")
    (package / "notes.md").write_text("still ignored\n")
    assert digest() == first
    (package / "defaults.toml").write_text("a = 2\n")
    assert digest() != first
    experiment.source_digest.cache_clear()


def test_a_corrupt_cached_prediction_file_is_ignored_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cached = tmp_path / "cached.jsonl"
    cached.write_text("not a predictions file\n")
    header = predictions_header(category_ids=(1, 2))
    with caplog.at_level(logging.WARNING, logger="qcal.lab.experiment"):
        assert experiment._cached_images(CacheEntry(cached, "R1", "now"), header, ["1"]) is None
    assert "ignoring cached predictions cached.jsonl" in caplog.text


def test_smoke_reports_a_project_it_cannot_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qcal import gitutil
    from qcal.config import ConfigError
    from qcal_lab import smoke

    def fail(args: list[str], *_: object, **__: object) -> str:
        raise gitutil.GitError(args, 128, "no git here")

    monkeypatch.setattr(smoke.gitutil, "git", fail)
    with pytest.raises(ConfigError, match="cannot commit the smoke project"):
        smoke._commit_project(tmp_path)
