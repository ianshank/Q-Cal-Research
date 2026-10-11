"""Detector registry: ``detectors.<name>.kind`` in the lab configuration selects a factory."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final

from qcal.components import ComponentRegistry
from qcal.config import ConfigError
from qcal.protocols import Detector
from qcal.registry.executor import sha256_file
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
KeyParts = Callable[[LabConfig, str], dict[str, Any]]
#: Per detector kind: what beneath its settings decides its predictions (the compute stack,
#: a resolved model config). Part of every prediction cache key; each kind registers one.
KEY_PARTS: ComponentRegistry[KeyParts] = ComponentRegistry("detector kind (cache key parts)")
#: The deployment targets each detector kind produces predictions for. A cell whose target
#: its detector's kind does not produce is refused, never served another target's predictions.
KIND_TARGETS: Final[Mapping[str, frozenset[str]]] = {
    "fixture": frozenset({"torch_fp32"}),
    "mmdet": frozenset({"torch_fp32"}),
}


#: What a detection's score is, per detector kind; every predictions header records it.
KIND_SCORE_DEFINITIONS: Final[Mapping[str, str]] = {
    "fixture": "fixture: synthetic box quality inflated by overconfidence (no model)",
    "mmdet": "mmdet: pred_instances.scores from inference_detector, after the model's test_cfg",
}
#: Detector settings that override the model's test-time configuration (mmdet test_cfg).
TEST_CFG_SETTINGS: Final = ("score_threshold", "max_per_image")
#: The detector setting naming the model weights, whose sha256 a predictions header records.
CHECKPOINT_SETTING: Final = "checkpoint"
#: Settings a detector kind must have set explicitly. MMDetection's test-time settings are
#: pinned in configuration, so a predictions header's test_cfg is the whole of what applied
#: and never falls back to a model config's own defaults.
KIND_REQUIRED_SETTINGS: Final[Mapping[str, tuple[str, ...]]] = {"mmdet": TEST_CFG_SETTINGS}


def detector_targets(lab: LabConfig, name: str) -> frozenset[str]:
    """The targets the configured detector ``name`` can produce."""
    return KIND_TARGETS.get(str(detector_settings(lab, name)["kind"]), frozenset())


def detector_key_parts(lab: LabConfig, name: str) -> dict[str, Any]:
    """What beneath the configuration decides detector ``name``'s predictions."""
    return KEY_PARTS.get(str(detector_settings(lab, name)["kind"]))(lab, name)


def detector_problems(lab: LabConfig, name: str) -> list[str]:
    """Required settings the configured detector ``name`` lacks (empty when it can run)."""
    settings = detector_settings(lab, name)
    required = KIND_REQUIRED_SETTINGS.get(str(settings["kind"]), ())
    return [f"detectors.{name}.{key} is not set" for key in required if key not in settings]


def score_definition(lab: LabConfig, name: str) -> str:
    """What the configured detector's scores are; refused for a kind that does not say."""
    kind = str(detector_settings(lab, name)["kind"])
    if kind not in KIND_SCORE_DEFINITIONS:
        raise ConfigError(f"detector kind {kind!r} does not define what its scores are")
    return KIND_SCORE_DEFINITIONS[kind]


def detector_test_cfg(lab: LabConfig, name: str) -> dict[str, Any]:
    """The test-time overrides configured for detector ``name`` (absent ones are omitted)."""
    settings = detector_settings(lab, name)
    return {key: settings[key] for key in TEST_CFG_SETTINGS if key in settings}


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
    """The spec plus what it points at: file contents and library versions.

    A prediction cache keyed on the spec alone would reuse stale predictions after a config or
    checkpoint was replaced at the same path, or after a library upgrade. This package's own
    version is not part of it: the cache key covers the code that produces predictions.
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
    problems = detector_problems(lab, name)
    if problems:
        raise ConfigError("; ".join(problems))
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
    "CHECKPOINT_SETTING",
    "DETECTORS",
    "KEY_PARTS",
    "KIND_REQUIRED_SETTINGS",
    "KIND_SCORE_DEFINITIONS",
    "KIND_TARGETS",
    "TEST_CFG_SETTINGS",
    "DetectorContext",
    "DetectorError",
    "DetectorFactory",
    "KeyParts",
    "build_detector",
    "detector_fingerprint",
    "detector_key_parts",
    "detector_problems",
    "detector_settings",
    "detector_spec",
    "detector_targets",
    "detector_test_cfg",
    "score_definition",
    "setting_int",
    "setting_number",
]
