"""qcal_lab: the agent-owned Phase 1 science code (clean-room FP32 reproduction).

The integrity layer (:mod:`qcal`) is Ian-signed; this package is not. It implements the
shapes in :mod:`qcal.protocols`, is wired by configuration (``configs/lab.toml`` over the
packaged defaults), and runs only through the registry's executor contract, so a result
reaches the run index only as a record that ``qcal registry run`` wrote.

Nothing here writes the run registry, Ian's documents or ``*/handwritten/*``. Ian's
hand-written evaluation loop plugs in through :mod:`qcal_lab.evaluation`.
"""

from __future__ import annotations

__version__ = "0.1.0"
