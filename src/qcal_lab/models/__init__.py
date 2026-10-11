"""Detectors behind :class:`qcal.protocols.Detector`.

Importing this package registers every detector kind in :data:`DETECTORS`.
"""

from __future__ import annotations

from qcal_lab.models import fixture, mmdet
from qcal_lab.models.base import (
    DETECTORS,
    KIND_TARGETS,
    DetectorContext,
    DetectorError,
    build_detector,
    detector_fingerprint,
    detector_settings,
    detector_spec,
    detector_targets,
)

__all__ = [
    "DETECTORS",
    "KIND_TARGETS",
    "DetectorContext",
    "DetectorError",
    "build_detector",
    "detector_fingerprint",
    "detector_settings",
    "detector_spec",
    "detector_targets",
    "fixture",
    "mmdet",
]
