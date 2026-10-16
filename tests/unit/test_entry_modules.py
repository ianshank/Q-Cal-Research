"""``python -m qcal``, ``python -m qcal.hooks`` and ``python -m qcal_lab`` call their CLIs.

Run in-process with ``runpy`` so the entry modules are measured by coverage rather than
excluded from it; tests/integration/test_entry_points.py runs them as real subprocesses.
"""

from __future__ import annotations

import runpy

import pytest


@pytest.mark.parametrize(
    ("module", "target"),
    [
        ("qcal", "qcal.cli.main"),
        ("qcal.hooks", "qcal.hooks.cli.main"),
        ("qcal_lab", "qcal_lab.cli.main"),
    ],
)
def test_entry_module_exits_with_the_cli_status(
    module: str, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(target, lambda *_, **__: 7)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    assert exited.value.code == 7
