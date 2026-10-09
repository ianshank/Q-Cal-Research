"""Q-Cal research integrity tooling.

The package is split so that the parts Claude Code hooks import (``config``,
``globs``, ``policy``, ``log``, ``hooks`` and ``integrity.claims``) depend only on
the Python standard library. That keeps the guards working before a virtual
environment exists and lets the hook wrapper fail closed when Python itself is
unavailable.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
