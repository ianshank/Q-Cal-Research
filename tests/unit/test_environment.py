"""Environment collectors: registry, merge order, and the never-raise contract."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import platform
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qcal.components import ComponentRegistry, UnknownComponentError
from qcal.config import Config
from qcal.registry import environment
from qcal.registry.environment import (
    COLLECTORS,
    Collector,
    collect_environment,
    python_executable,
)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> ComponentRegistry[Collector]:
    """A private collector registry so tests never mutate the global one."""
    fresh: ComponentRegistry[Collector] = ComponentRegistry("environment collector")
    monkeypatch.setattr(environment, "COLLECTORS", fresh)
    return fresh


@pytest.fixture
def fake_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Make ``find_spec`` and ``import_module`` see the modules placed in the returned dict."""
    modules: dict[str, Any] = {}
    real_find_spec = importlib.util.find_spec
    real_import_module = importlib.import_module

    def find_spec(name: str, package: str | None = None) -> Any:
        if name in modules:
            return None if modules[name] is None else SimpleNamespace(name=name)
        return real_find_spec(name, package)

    def import_module(name: str, package: str | None = None) -> Any:
        if name in modules:
            if modules[name] is None:
                raise ModuleNotFoundError(name)
            return modules[name]
        return real_import_module(name, package)

    monkeypatch.setattr(environment.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(environment.importlib, "import_module", import_module)
    return modules


def fake_torch(*, cuda: bool) -> SimpleNamespace:
    return SimpleNamespace(
        __version__="2.5.1",
        version=SimpleNamespace(cuda="12.4" if cuda else None),
        cuda=SimpleNamespace(is_available=lambda: cuda, get_arch_list=lambda: ["sm_86", "sm_89"]),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100)),
    )


# -- registry and merging -----------------------------------------------------------------


def test_every_configured_default_collector_is_registered(config: Config) -> None:
    assert set(config.str_list("registry.env_collectors")) <= set(COLLECTORS.names())


def test_collect_with_no_names_is_empty(tmp_path: Path) -> None:
    assert collect_environment([], tmp_path) == {}


def test_collect_merges_collectors_in_order(
    registry: ComponentRegistry[Collector], tmp_path: Path
) -> None:
    registry.register("first", lambda _root: {"a": 1, "shared": "first"})
    registry.register("second", lambda _root: {"b": 2, "shared": "second"})
    assert collect_environment(["first", "second"], tmp_path) == {
        "a": 1,
        "b": 2,
        "shared": "second",
    }


def test_collect_passes_the_root_to_collectors(
    registry: ComponentRegistry[Collector], tmp_path: Path
) -> None:
    registry.register("where", lambda root: {"root": str(root)})
    assert collect_environment(["where"], tmp_path) == {"root": str(tmp_path)}


def test_failing_collector_contributes_nothing(
    registry: ComponentRegistry[Collector], tmp_path: Path
) -> None:
    def boom(_root: Path) -> dict[str, Any]:
        raise RuntimeError("driver exploded")

    registry.register("ok", lambda _root: {"ok": True})
    registry.register("boom", boom)
    assert collect_environment(["boom", "ok"], tmp_path) == {"ok": True}


def test_failing_collector_is_a_warning_with_a_debug_traceback(
    registry: ComponentRegistry[Collector], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(_root: Path) -> dict[str, Any]:
        raise RuntimeError("driver exploded")

    registry.register("boom", boom)
    with caplog.at_level(logging.DEBUG, logger="qcal"):
        assert collect_environment(["boom"], tmp_path) == {}
    levels = {r.levelno for r in caplog.records if "boom" in r.getMessage()}
    assert levels == {logging.WARNING, logging.DEBUG}
    assert any(r.exc_info for r in caplog.records if r.levelno == logging.DEBUG)


def test_command_timeout_reaches_external_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from qcal.registry import environment

    seen: list[float] = []

    def fake_run(*_args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(kwargs["timeout"])
        return subprocess.CompletedProcess([], 0, "GPU, 1.0\n", "")

    monkeypatch.setattr(environment.shutil, "which", lambda _name: "/bin/true")
    monkeypatch.setattr(environment.subprocess, "run", fake_run)
    collect_environment(["nvidia"], tmp_path, command_timeout_s=2.5)
    collect_environment(["nvidia"], tmp_path)
    assert seen == [2.5, environment.DEFAULT_COMMAND_TIMEOUT_S]


def test_unknown_collector_name_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(UnknownComponentError, match="unknown environment collector 'nope'"):
        collect_environment(["nope"], tmp_path)


def test_real_collector_failure_is_swallowed(fake_modules: dict[str, Any], tmp_path: Path) -> None:
    fake_modules["torch"] = SimpleNamespace()  # no __version__: the collector raises
    result = collect_environment(["torch", "python"], tmp_path)
    assert set(result) == {"python", "python_impl"}


# -- built-in collectors ------------------------------------------------------------------


def test_python_collector_reports_version_and_implementation(tmp_path: Path) -> None:
    assert COLLECTORS.get("python")(tmp_path) == {
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
    }


def test_host_collector_reports_host_platform_and_machine(tmp_path: Path) -> None:
    info = COLLECTORS.get("host")(tmp_path)
    assert set(info) == {"host", "platform", "machine"}
    assert info["machine"] == platform.machine()


@pytest.mark.integration
def test_git_collector_reports_the_branch(git_repo: Path) -> None:
    assert COLLECTORS.get("git")(git_repo) == {"git_branch": "main"}


@pytest.mark.integration
def test_git_collector_outside_a_repository_is_empty(tmp_path: Path) -> None:
    assert COLLECTORS.get("git")(tmp_path) == {}


def test_packages_collector_hashes_installed_distributions(tmp_path: Path) -> None:
    info = COLLECTORS.get("packages")(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{64}", info["env_hash"])
    assert info["package_count"] == len(list(importlib.metadata.distributions()))


def test_packages_collector_is_deterministic(tmp_path: Path) -> None:
    collector = COLLECTORS.get("packages")
    assert collector(tmp_path) == collector(tmp_path)


def test_nvidia_collector_without_nvidia_smi_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(environment.shutil, "which", lambda _name: None)
    assert COLLECTORS.get("nvidia")(tmp_path) == {}


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "NVIDIA RTX 5060 Ti, 575.51\nNVIDIA RTX 5060, 575.51",
            {"gpu": "NVIDIA RTX 5060 Ti; NVIDIA RTX 5060", "driver": "575.51"},
        ),
        ("Tesla P40, 520.61.05\n\n", {"gpu": "Tesla P40", "driver": "520.61.05"}),
        ("Orin", {"gpu": "Orin", "driver": None}),
        (None, {}),
        ("", {}),
    ],
)
def test_nvidia_collector_parses_nvidia_smi_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output: str | None, expected: dict[str, Any]
) -> None:
    monkeypatch.setattr(environment, "_run", lambda _argv: output)
    assert COLLECTORS.get("nvidia")(tmp_path) == expected


def test_torch_collector_without_torch_is_empty(
    fake_modules: dict[str, Any], tmp_path: Path
) -> None:
    fake_modules["torch"] = None
    assert COLLECTORS.get("torch")(tmp_path) == {}


def test_torch_collector_reports_cuda_details_when_available(
    fake_modules: dict[str, Any], tmp_path: Path
) -> None:
    fake_modules["torch"] = fake_torch(cuda=True)
    assert COLLECTORS.get("torch")(tmp_path) == {
        "torch": "2.5.1",
        "cuda": "12.4",
        "cuda_arch_list": "sm_86 sm_89",
        "cudnn": 90100,
    }


def test_torch_collector_omits_device_details_without_cuda(
    fake_modules: dict[str, Any], tmp_path: Path
) -> None:
    fake_modules["torch"] = fake_torch(cuda=False)
    assert COLLECTORS.get("torch")(tmp_path) == {"torch": "2.5.1", "cuda": None}


def test_tensorrt_collector_without_tensorrt_is_empty(
    fake_modules: dict[str, Any], tmp_path: Path
) -> None:
    fake_modules["tensorrt"] = None
    assert COLLECTORS.get("tensorrt")(tmp_path) == {}


def test_tensorrt_collector_reports_version(fake_modules: dict[str, Any], tmp_path: Path) -> None:
    fake_modules["tensorrt"] = SimpleNamespace(__version__="10.3.0")
    assert COLLECTORS.get("tensorrt")(tmp_path) == {"trt_version": "10.3.0"}


def test_jetson_collector_off_device_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(environment, "_TEGRA_RELEASE", tmp_path / "missing")
    assert COLLECTORS.get("jetson")(tmp_path) == {}


def test_jetson_collector_reports_l4t_and_power_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = tmp_path / "nv_tegra_release"
    release.write_text("  # R36 (release), REVISION: 4.3  \nsecond line\n", "utf-8")
    monkeypatch.setattr(environment, "_TEGRA_RELEASE", release)
    monkeypatch.setattr(environment, "_run", lambda _argv: "NV Power Mode:   MAXN_SUPER\n2")
    assert COLLECTORS.get("jetson")(tmp_path) == {
        "l4t": "# R36 (release), REVISION: 4.3",
        "nvpmodel": "NV Power Mode: MAXN_SUPER 2",
    }


def test_jetson_collector_without_nvpmodel_reports_only_l4t(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = tmp_path / "nv_tegra_release"
    release.write_text("# R36\n", "utf-8")
    monkeypatch.setattr(environment, "_TEGRA_RELEASE", release)
    monkeypatch.setattr(environment, "_run", lambda _argv: None)
    assert COLLECTORS.get("jetson")(tmp_path) == {"l4t": "# R36"}


def test_python_executable_is_the_running_interpreter() -> None:
    assert python_executable() == sys.executable


# -- the command helper -------------------------------------------------------------------


def test_run_returns_none_when_the_program_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment.shutil, "which", lambda _name: None)
    assert environment._run(["nvidia-smi"]) is None


@pytest.mark.integration
def test_run_returns_stripped_stdout_on_success() -> None:
    assert environment._run([sys.executable, "-c", "print('  hello  ')"]) == "hello"


@pytest.mark.integration
def test_run_returns_none_on_non_zero_exit() -> None:
    assert environment._run([sys.executable, "-c", "raise SystemExit(3)"]) is None


@pytest.mark.parametrize(
    "error",
    [OSError("exec format error"), subprocess.TimeoutExpired(["x"], 10)],
    ids=["os-error", "timeout"],
)
def test_run_returns_none_when_the_command_cannot_complete(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(environment.shutil, "which", lambda name: name)
    monkeypatch.setattr(environment.subprocess, "run", fail)
    assert environment._run(["anything"]) is None
