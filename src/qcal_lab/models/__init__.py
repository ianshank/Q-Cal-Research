"""Detectors behind :class:`qcal.protocols.Detector`.

Importing this package registers every detector kind in :data:`DETECTORS`.
"""

from __future__ import annotations

from qcal_lab.models import fixture, mmdet
from qcal_lab.models.base import (
    CHECKPOINT_SETTING,
    DETECTORS,
    KEY_PARTS,
    KIND_REQUIRED_SETTINGS,
    KIND_SCORE_DEFINITIONS,
    KIND_TARGETS,
    TEST_CFG_SETTINGS,
    DetectorContext,
    DetectorError,
    build_detector,
    detector_fingerprint,
    detector_key_parts,
    detector_problems,
    detector_settings,
    detector_spec,
    detector_targets,
    detector_test_cfg,
    score_definition,
)

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
    "build_detector",
    "detector_fingerprint",
    "detector_key_parts",
    "detector_problems",
    "detector_settings",
    "detector_spec",
    "detector_targets",
    "detector_test_cfg",
    "fixture",
    "mmdet",
    "score_definition",
]
