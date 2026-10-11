"""Oracle suite plumbing: libraries are optional locally, required where ORACLE_REQUIRED is set.

The oracles are public, BSD-licensed libraries used only as *test-time* references for the
mathematics (never imported by ``src/``): scikit-learn's ``IsotonicRegression`` (which
Kuzucu et al. fit, Sec. 4.4), SciPy's L-BFGS-B (the optimiser the paper names for Platt
scaling, App. C.3) and pycocotools. They are not the fiveai reference implementation; that
one is the authoritative G1 oracle and runs only on the P40 (tests/parity). Agreement here
settles TD-13 and TD-16 before those outputs exist; it does not replace them.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import pytest

ORACLE_REQUIRED_ENV = "ORACLE_REQUIRED"  # not QCAL_-prefixed: tests strip those


def oracle(module: str) -> ModuleType:
    """Import an oracle library; skip when it is absent unless ORACLE_REQUIRED is set."""
    if os.environ.get(ORACLE_REQUIRED_ENV, "").lower() in {"1", "true", "yes"}:
        return importlib.import_module(module)
    return pytest.importorskip(module, reason=f"{module} not installed (pip install -e .[oracle])")
