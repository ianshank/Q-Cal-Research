"""Command-line interface: ``qcal <command>`` plus the plan's ``qcal-registry`` alias.

Exit codes: 0 success, 1 a check failed or a run failed, 2 usage or
configuration error. ``--debug`` (or ``$QCAL_DEBUG``) prints tracebacks.

The package is split by command group (``registry``, ``checks``, ``ci``,
``inspection``); :mod:`qcal.cli.app` wires the groups into one parser. The names
re-exported here are the stable public surface used by entry points and tests.
"""

from __future__ import annotations

from qcal.cli.app import COMMAND_GROUPS, build_parser, claims_main, main, registry_main
from qcal.cli.common import EXIT_FAILED, EXIT_OK, EXIT_USAGE

__all__ = [
    "COMMAND_GROUPS",
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_USAGE",
    "build_parser",
    "claims_main",
    "main",
    "registry_main",
]
