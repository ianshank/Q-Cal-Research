"""Post-hoc calibrators and the two-threshold class-wise procedure of arXiv:2405.20459.

Importing this package registers every calibrator kind in :data:`CALIBRATORS`.
"""

from __future__ import annotations

from qcal_lab.calib import isotonic, platt
from qcal_lab.calib.base import (
    CALIBRATORS,
    CalibrationError,
    IdentityCalibrator,
    SerializableCalibrator,
    build_calibrator,
    load_calibrator,
)

__all__ = [
    "CALIBRATORS",
    "CalibrationError",
    "IdentityCalibrator",
    "SerializableCalibrator",
    "build_calibrator",
    "isotonic",
    "load_calibrator",
    "platt",
]
