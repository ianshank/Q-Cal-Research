"""Pluggable environment collectors recorded with every run.

Each collector returns a flat dict and should not raise; a failure is logged as a
WARNING and the collector contributes nothing. Configuration chooses which run
(``registry.env_collectors``); new collectors register by name.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from qcal import gitutil
from qcal.components import ComponentRegistry
from qcal.config import load_defaults
from qcal.log import get_logger

_log = get_logger("registry.environment")

Collector = Callable[[Path], dict[str, Any]]
COLLECTORS: ComponentRegistry[Collector] = ComponentRegistry("environment collector")
_TEGRA_RELEASE = Path("/etc/nv_tegra_release")
DEFAULT_COMMAND_TIMEOUT_S = float(load_defaults()["registry"]["env_command_timeout_s"])
_command_timeout: ContextVar[float] = ContextVar(
    "qcal_env_command_timeout", default=DEFAULT_COMMAND_TIMEOUT_S
)


def collect_environment(
    names: Sequence[str], root: Path, *, command_timeout_s: float | None = None
) -> dict[str, Any]:
    """Run the named collectors; external probes (``nvidia-smi``...) get ``command_timeout_s``."""
    token = _command_timeout.set(
        DEFAULT_COMMAND_TIMEOUT_S if command_timeout_s is None else command_timeout_s
    )
    try:
        return _collect(names, root)
    finally:
        _command_timeout.reset(token)


def _collect(names: Sequence[str], root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        collector = COLLECTORS.get(name)
        try:
            values = collector(root)
        except Exception as exc:  # noqa: BLE001 - collectors are best-effort by contract
            # WARNING, not DEBUG: a broken collector silently drops versions from the record.
            _log.warning("environment collector %s failed: %s", name, exc)
            _log.debug("collector %s traceback", name, exc_info=True)
            continue
        overlap = set(values) & set(result)
        if overlap:
            _log.debug("collector %s overrides %s", name, sorted(overlap))
        result.update(values)
    return result


def _run(argv: Sequence[str]) -> str | None:
    if shutil.which(argv[0]) is None:
        return None
    try:
        out = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=_command_timeout.get(), check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.debug("%s failed: %s", argv[0], exc)
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@COLLECTORS.register("python")
def _python(_: Path) -> dict[str, Any]:
    return {"python": platform.python_version(), "python_impl": platform.python_implementation()}


@COLLECTORS.register("host")
def _host(_: Path) -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


@COLLECTORS.register("git")
def _git(root: Path) -> dict[str, Any]:
    branch = gitutil.try_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    return {"git_branch": branch} if branch else {}


@COLLECTORS.register("packages")
def _packages(_: Path) -> dict[str, Any]:
    pins = sorted(
        f"{(d.metadata.get_all('Name') or [''])[0].lower()}=={d.version}"
        for d in importlib.metadata.distributions()
    )
    digest = hashlib.sha256("\n".join(pins).encode()).hexdigest()
    return {"env_hash": digest, "package_count": len(pins)}


@COLLECTORS.register("nvidia")
def _nvidia(_: Path) -> dict[str, Any]:
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if not out:
        return {}
    gpus = [line.split(",") for line in out.splitlines() if line.strip()]
    return {
        "gpu": "; ".join(g[0].strip() for g in gpus),
        "driver": gpus[0][1].strip() if len(gpus[0]) > 1 else None,
    }


@COLLECTORS.register("torch")
def _torch(_: Path) -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {}
    torch: Any = importlib.import_module("torch")
    info: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        info["cuda_arch_list"] = " ".join(torch.cuda.get_arch_list())
        info["cudnn"] = torch.backends.cudnn.version()
    return info


@COLLECTORS.register("tensorrt")
def _tensorrt(_: Path) -> dict[str, Any]:
    if importlib.util.find_spec("tensorrt") is None:
        return {}
    return {"trt_version": importlib.import_module("tensorrt").__version__}


@COLLECTORS.register("jetson")
def _jetson(_: Path) -> dict[str, Any]:
    if not _TEGRA_RELEASE.is_file():
        return {}
    info: dict[str, Any] = {"l4t": _TEGRA_RELEASE.read_text("utf-8").splitlines()[0].strip()}
    mode = _run(["nvpmodel", "-q"])
    if mode:
        info["nvpmodel"] = " ".join(mode.split())
    return info


def python_executable() -> str:
    return sys.executable
