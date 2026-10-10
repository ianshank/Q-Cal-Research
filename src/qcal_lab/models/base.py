"""Detector registry: ``detectors.<name>.kind`` in the lab configuration selects a factory."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from qcal.components import ComponentRegistry
from qcal.config import ConfigError
from qcal.protocols import Detector
from qcal.registry.executor import sha256_file
from qcal_lab import __version__
from qcal_lab.config import LabConfig
from qcal_lab.data.coco import GroundTruth
from qcal_lab.data.fixture import is_fixture


class DetectorError(RuntimeError):
    """A detector cannot be built or cannot produce predictions."""


@dataclass(frozen=True)
class DetectorContext:
    name: str
    settings: Mapping[str, Any]
    ground_truth: GroundTruth
    lab: LabConfig
    precision: str
    images_dir: Path | None = None


DetectorFactory = Callable[[DetectorContext], Detector]
DETECTORS: ComponentRegistry[DetectorFactory] = ComponentRegistry("detector kind")


def detector_settings(lab: LabConfig, name: str) -> Mapping[str, Any]:
    detectors = lab.config.get("detectors", {})
    if not isinstance(detectors, Mapping) or not isinstance(detectors.get(name), Mapping):
        known = ", ".join(sorted(detectors)) if isinstance(detectors, Mapping) else "none"
        raise ConfigError(f"detector {name!r} is not configured (detectors.*: {known})")
    settings: Mapping[str, Any] = detectors[name]
    if not isinstance(settings.get("kind"), str):
        raise ConfigError(f"detectors.{name}.kind must name a detector kind")
    return settings


def detector_spec(lab: LabConfig, name: str, precision: str) -> dict[str, Any]:
    """Everything that determines a detector's raw predictions (cache keys, provenance)."""
    spec = json.loads(json.dumps(dict(detector_settings(lab, name)), sort_keys=True))
    return {"name": name, "precision": precision, "settings": spec}


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"


def detector_fingerprint(
    lab: LabConfig,
    name: str,
    precision: str,
    *,
    file_settings: Sequence[str],
    packages: Sequence[str],
) -> dict[str, Any]:
    """The spec plus what it points at: file contents, library versions, this package's version.

    A prediction cache keyed on the spec alone would reuse stale predictions after a config or
    checkpoint was replaced at the same path, or after a library upgrade.
    """
    settings = detector_settings(lab, name)
    files: dict[str, str] = {}
    for key in file_settings:
        value = settings.get(key)
        if isinstance(value, str) and value:
            path = lab.file(f"detectors.{name}.{key}")
            files[key] = sha256_file(path) if path.is_file() else "missing"
    return {
        **detector_spec(lab, name, precision),
        "files": files,
        "packages": {package: _package_version(package) for package in packages},
        "qcal_lab": __version__,
    }


def build_detector(
    lab: LabConfig,
    name: str,
    ground_truth: GroundTruth,
    *,
    precision: str,
    images_dir: Path | None = None,
) -> Detector:
    settings = detector_settings(lab, name)
    factory = DETECTORS.get(str(settings["kind"]))
    # A real detector never sees ground-truth boxes (test boxes included); only the synthetic
    # fixture's stand-in detector, which perturbs them, gets the boxes.
    visible = ground_truth if is_fixture(ground_truth) else ground_truth.without_boxes()
    return factory(DetectorContext(name, settings, visible, lab, precision, images_dir))


def setting_number(settings: Mapping[str, Any], key: str, *, low: float, high: float) -> float:
    value = settings.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ConfigError(f"detector setting {key!r} must be a number")
    if not low <= value <= high:
        raise ConfigError(f"detector setting {key!r} must lie in [{low}, {high}]")
    return float(value)


def setting_int(settings: Mapping[str, Any], key: str) -> int:
    value = settings.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"detector setting {key!r} must be a non-negative integer")
    return value


__all__ = [
    "DETECTORS",
    "DetectorContext",
    "DetectorError",
    "DetectorFactory",
    "build_detector",
    "detector_fingerprint",
    "detector_settings",
    "detector_spec",
    "setting_int",
    "setting_number",
]
