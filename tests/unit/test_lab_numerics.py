"""qcal_lab.numerics: each precision's torch switches, applied and recorded (PR-D1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from qcal.config import ConfigError
from qcal_lab.numerics import (
    RECORDED_ENV,
    SWITCHES,
    Regime,
    apply_regime,
    device_record,
    effective_switches,
    load_regime,
)
from tests.lab_support import FakeTorch, lab_config


def test_fp32_keeps_torchs_defaults(tmp_path: Path) -> None:
    regime = load_regime(lab_config(tmp_path), "fp32")
    assert regime == Regime("fp32", {})
    torch = FakeTorch()
    before = effective_switches(torch)
    record = apply_regime(torch, regime, {})
    assert record["effective"] == before  # nothing set, everything recorded
    assert set(record["effective"]) == {
        *SWITCHES,
        "deterministic_algorithms",
        "deterministic_warn_only",
    }


def test_fp32_tf32_off_turns_tf32_off_and_determinism_on(tmp_path: Path) -> None:
    regime = load_regime(lab_config(tmp_path), "fp32_tf32_off")
    torch = FakeTorch()
    environ = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": "0"}
    record = apply_regime(torch, regime, environ)
    assert record["effective"] == {
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
    }
    assert record["environment"] == {name: environ[name] for name in RECORDED_ENV}
    assert record["regime"]["required_env"] == ["CUBLAS_WORKSPACE_CONFIG"]


def test_a_regime_refuses_to_run_without_its_environment(tmp_path: Path) -> None:
    regime = load_regime(lab_config(tmp_path), "fp32_tf32_off")
    torch = FakeTorch()
    with pytest.raises(ConfigError, match="needs CUBLAS_WORKSPACE_CONFIG set"):
        apply_regime(torch, regime, {"CUBLAS_WORKSPACE_CONFIG": ""})
    assert torch.backends.cudnn.allow_tf32 is True  # refused before anything was switched


def test_every_supported_precision_has_a_regime(tmp_path: Path) -> None:
    lab = lab_config(tmp_path)
    for precision in lab.config.str_list("factors.supported.precision"):
        assert load_regime(lab, precision).precision == precision


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("", "has no \\[numerics.regimes.int8\\] table"),
        ("[numerics.regimes.int8]\nallow_tf32 = false\n", r"unknown keys \['allow_tf32'\]"),
        ("[numerics.regimes.int8]\ncudnn_benchmark = 0\n", "must be true or false"),
        ('[numerics.regimes.int8]\nrequired_env = "X"\n', "list of variable names"),
        ('[numerics.regimes.int8]\nrequired_env = [""]\n', "list of variable names"),
    ],
)
def test_malformed_regimes_are_refused(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_regime(lab_config(tmp_path, body), "int8")


def test_warn_only_is_a_regime_setting(tmp_path: Path) -> None:
    body = (
        "[numerics.regimes.lax]\ndeterministic_algorithms = true\ndeterministic_warn_only = true\n"
    )
    torch = FakeTorch()
    record = apply_regime(torch, load_regime(lab_config(tmp_path, body), "lax"), {})
    assert (torch.deterministic, torch.warn_only) == (True, True)
    assert record["effective"]["deterministic_warn_only"] is True


def test_device_record_names_the_device_that_ran() -> None:
    record = device_record(FakeTorch(), "cuda:1")
    assert record["name"] == "GPU 1"
    assert record["uuid"] == "GPU-0001"
    assert record["capability"] == "12.0"
    assert (record["cuda"], record["cudnn"]) == ("12.8", 91000)
    assert len(record["torch_build_sha256"]) == 64


@pytest.mark.parametrize(("device", "available"), [("cpu", True), ("cuda:0", False)])
def test_device_record_without_a_gpu_keeps_the_stack(device: str, available: bool) -> None:
    record = device_record(FakeTorch(cuda_available=available), device)
    assert record["device"] == device
    assert "uuid" not in record
    assert record["torch"] == "2.9.0+cu128"
