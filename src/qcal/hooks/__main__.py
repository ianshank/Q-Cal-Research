"""Entry point used by ``.claude/hooks/run_hook.sh``: ``python -m qcal.hooks <hook>``."""

from __future__ import annotations

from qcal.hooks.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
