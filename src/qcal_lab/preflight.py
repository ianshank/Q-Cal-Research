"""Preflight for the GPU host (rev. 3): what fails a first registered run late if unchecked.

``python -m qcal_lab status`` reports these next to Ian's inputs:

- the lock and cache directories (``paths.inflight_dir``, ``predictions.cache_dir``) are on a
  host-local filesystem: ``flock`` on a network one may refuse (``ENOLCK``) or lock nothing
  across hosts, and a second launcher would not see the first;
- each configured MMDetection detector can run: CUDA is available for a ``cuda`` device, and
  ``mmcv._ext`` (the compiled operators) imports.

Nothing here changes a run; it reports. The network filesystem types and the mount table come
from ``[preflight]`` in ``tooling.toml``.
"""

from __future__ import annotations

import importlib
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from qcal.config import Config
from qcal.log import get_logger
from qcal_lab.config import LabConfig, load_tooling

_log = get_logger("lab.preflight")

SECTION: Final = "preflight"
MMDET_KIND: Final = "mmdet"
CUDA_PREFIX: Final = "cuda"
MMCV_EXT: Final = "mmcv._ext"
#: device, mount point, type: the fields of a mount-table line read here.
MOUNT_FIELDS: Final = 3
#: The kernel writes space, tab, newline and backslash in mount points as \ooo.
OCTAL_ESCAPE: Final = re.compile(r"\\([0-7]{3})")

Importer = Callable[[str], Any]


def settings(tooling: Mapping[str, Any] | None = None) -> tuple[frozenset[str], Path]:
    """The network filesystem types and the mount table to read."""
    section = (tooling if tooling is not None else load_tooling()).get(SECTION, {})
    types = section.get("network_filesystems", []) if isinstance(section, Mapping) else []
    mounts = section.get("mounts_file", "") if isinstance(section, Mapping) else ""
    return frozenset(str(t) for t in types), Path(str(mounts))


def mount_points(text: str) -> list[tuple[str, str]]:
    """``(mount point, filesystem type)`` per line of a ``/proc/mounts``-style table."""
    entries: list[tuple[str, str]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= MOUNT_FIELDS:
            point = OCTAL_ESCAPE.sub(lambda m: chr(int(m[1], 8)), fields[1])  # \040 and the like
            entries.append((point, fields[2]))
    return entries


def filesystem_type(path: Path, mounts: Sequence[tuple[str, str]]) -> str | None:
    """The type of the filesystem ``path`` is on, or would be on once created: links in its
    existing part are resolved, the rest is taken as written."""
    existing, missing = path, []
    while not existing.exists() and existing != existing.parent:
        missing.append(existing.name)
        existing = existing.parent
    resolved = str(Path(os.path.realpath(existing)).joinpath(*reversed(missing)))
    best: tuple[int, str] | None = None
    for point, kind in mounts:
        inside = resolved == point or resolved.startswith(point.rstrip("/") + "/")
        if inside and (best is None or len(point) >= best[0]):
            best = (len(point), kind)
    return best[1] if best else None


def host_local_problems(
    paths: Iterable[tuple[str, Path]], network: frozenset[str], mounts_text: str
) -> list[str]:
    """Each named directory on a network filesystem, as a problem."""
    mounts = mount_points(mounts_text)
    problems: list[str] = []
    for name, path in paths:
        kind = filesystem_type(path, mounts)
        if kind in network:
            problems.append(f"{name} ({path}) is on {kind}: locks need a host-local filesystem")
    return problems


def runtime_problems(device: str, importer: Importer = importlib.import_module) -> list[str]:
    """Why an MMDetection detector on ``device`` cannot run here; empty when it can."""
    problems: list[str] = []
    try:
        importer(MMCV_EXT)
    except Exception as exc:  # noqa: BLE001 - a broken build raises more than ImportError
        problems.append(f"{MMCV_EXT} does not import ({type(exc).__name__}: {exc})")
    if device.startswith(CUDA_PREFIX):
        try:
            torch = importer("torch")
            available = bool(torch.cuda.is_available())
        except Exception as exc:  # noqa: BLE001 - as above
            problems.append(f"torch does not import ({type(exc).__name__}: {exc})")
        else:
            if not available:
                problems.append(f"device {device} but CUDA is not available")
    return problems


def _directories(qcal_config: Config, lab: LabConfig) -> list[tuple[str, Path]]:
    directories = [("paths.inflight_dir", qcal_config.path("inflight_dir"))]
    cache = lab.config.str_value("predictions.cache_dir")
    if cache:
        path = Path(cache)
        directories.append(
            ("predictions.cache_dir", path if path.is_absolute() else lab.config.root / path)
        )
    return directories


def add_preflight(
    report: Any,
    qcal_config: Config,
    lab: LabConfig,
    *,
    tooling: Mapping[str, Any] | None = None,
    importer: Importer = importlib.import_module,
) -> None:
    """Add the preflight items to a :class:`qcal_lab.status.StatusReport`."""
    network, mounts_file = settings(tooling)
    try:
        mounts_text = mounts_file.read_text("utf-8")
    except OSError as exc:
        _log.debug("no mount table at %s (%s): filesystems not checked", mounts_file, exc)
        report.add("lock directories host-local", ok=True, detail=f"not checked ({mounts_file})")
    else:
        problems = host_local_problems(_directories(qcal_config, lab), network, mounts_text)
        detail = "; ".join(problems) or "yes"
        report.add("lock directories host-local", ok=not problems, detail=detail)
    detectors = lab.config.get("detectors", {})
    for name, configured in sorted(detectors.items() if isinstance(detectors, Mapping) else []):
        if not isinstance(configured, Mapping) or configured.get("kind") != MMDET_KIND:
            continue
        problems = runtime_problems(str(configured.get("device", "")), importer)
        report.add(f"detector {name} runtime", ok=not problems, detail="; ".join(problems) or "ok")


__all__ = [
    "add_preflight",
    "filesystem_type",
    "host_local_problems",
    "mount_points",
    "runtime_problems",
    "settings",
]
