"""Allow ``python -m qcal``."""

from __future__ import annotations

from qcal.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
