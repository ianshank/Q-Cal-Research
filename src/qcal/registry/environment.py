"""Pluggable environment collectors recorded with every run.

Each collector returns a flat dict and should not raise; a failure is logged as a
WARNING and the collector contributes nothing. Configuration chooses which run
(``registry.env_collectors``); new collectors register by name.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Final

import qcal
from qcal import gitutil
from qcal.components import ComponentRegistry
from qcal.config import load_defaults
from qcal.log import get_logger

_log = get_logger("registry.environment")

Collector = Callable[[Path], dict[str, Any]]
COLLECTORS: ComponentRegistry[Collector] = ComponentRegistry("environment collector")
_TEGRA_RELEASE = Path("/etc/nv_tegra_release")
# nvidia-smi fields per GPU, and the record key each becomes. compute_cap needs a recent
# driver; without it the collector falls back to names and drivers only.
_GPU_FIELDS: Final = (
    ("index", "index"),
    ("uuid", "uuid"),
    ("pci.bus_id", "pci_bus_id"),
    ("name", "name"),
    ("compute_cap", "compute_cap"),
    ("memory.total", "memory_total_mib"),
    ("driver_version", "driver"),
)
_GPU_FALLBACK_FIELDS: Final = (("name", "name"), ("driver_version", "driver"))
_INTEGER_GPU_FIELDS: Final = frozenset({"index", "memory_total_mib"})
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


def _query_gpus(fields: Sequence[tuple[str, str]]) -> list[dict[str, Any]] | None:
    query = ",".join(name for name, _ in fields)
    out = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    gpus: list[dict[str, Any]] = []
    for row in csv.reader(line for line in out.splitlines() if line.strip()):
        gpu: dict[str, Any] = {}
        for (_, key), raw in zip(fields, row, strict=False):
            value = raw.strip()
            gpu[key] = int(value) if key in _INTEGER_GPU_FIELDS and value.isdigit() else value
        gpus.append(gpu)
    return gpus


@COLLECTORS.register("nvidia")
def _nvidia(_: Path) -> dict[str, Any]:
    """Every GPU on the host, in nvidia-smi (PCI bus) order. Which one a run used is the
    program's to report: CUDA numbers devices fastest-first unless CUDA_DEVICE_ORDER says
    otherwise, so an index here need not match the program's device index."""
    gpus = _query_gpus(_GPU_FIELDS) or _query_gpus(_GPU_FALLBACK_FIELDS)
    if not gpus:
        return {}
    return {
        "gpus": gpus,
        "gpu": "; ".join(str(g.get("name", "")) for g in gpus),
        "driver": gpus[0].get("driver") or None,
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


def launcher_identity(root: Path) -> dict[str, Any]:
    """The interpreter and qcal that launched a run (``provenance.launcher``)."""
    package = Path(qcal.__file__).resolve().parent
    resolved = root.resolve()
    location = (
        package.relative_to(resolved).as_posix()
        if package.is_relative_to(resolved)
        else str(package)
    )
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "qcal_version": qcal.__version__,
        "qcal_location": location,
    }


def recorded_variables(
    names: Sequence[str], environ: Mapping[str, str] | None = None
) -> dict[str, str | None]:
    """The named environment variables verbatim, ``None`` when unset."""
    source = os.environ if environ is None else environ
    return {name: source.get(name) for name in names}
