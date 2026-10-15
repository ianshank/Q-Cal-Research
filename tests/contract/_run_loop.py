"""Subprocess helper: print a loop's canonical outputs, optionally with every file it opened.

``python -m tests.contract._run_loop <fixture|configured> [--audit]``. Run in a fresh
interpreter so the hash seed, and nothing else, differs between two runs. With ``--audit``
the hook is installed before the loop's module is imported and its factory runs, so files
opened while it is built (``build_events``) are reported apart from those opened while it
evaluates (``events``): a loop could otherwise read a manifest at import and cache it.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable

from qcal.config import load_config
from qcal_lab.config import load_lab_config
from qcal_lab.eval_contract import canonical, fixture_inputs, outputs
from qcal_lab.evaluation import load_eval_loop
from tests.conftest import REPO_ROOT
from tests.lab_support import fixture_ground_truth


def _opened(target: object) -> str:
    """What an ``open`` audit event opened, as text: a bytes path is decoded, so
    ``open(os.fsencode(path))`` is reported like ``open(path)``; a descriptor is ``fd:<n>``."""
    if isinstance(target, int) and not isinstance(target, bool):
        return f"fd:{target}"
    if isinstance(target, str | bytes) or hasattr(target, "__fspath__"):
        return os.fsdecode(target)  # type: ignore[arg-type]
    return repr(target)


def audit_hook(
    events: dict[str, list[list[str]]], phase: list[str]
) -> Callable[[str, tuple[object, ...]], None]:
    """Records every ``open`` and ``socket.*`` event under the current phase."""

    def hook(event: str, args: tuple[object, ...]) -> None:
        if event == "open":  # str, bytes and path-like paths, and descriptors alike
            events[phase[0]].append([event, _opened(args[0] if args else None)])
        elif event.startswith("socket."):
            events[phase[0]].append([event, ""])

    return hook


def main(which: str, audit: bool) -> None:
    lab = load_lab_config(REPO_ROOT)
    qcal_config = load_config(REPO_ROOT, environ={})
    ground_truth = fixture_ground_truth()
    predictions = fixture_inputs(lab, ground_truth)
    events: dict[str, list[list[str]]] = {"build": [], "evaluate": []}
    phase = ["build"]
    if audit:
        sys.addaudithook(audit_hook(events, phase))
    if which == "fixture":
        from qcal_lab.fixture_eval import build as build_fixture_loop  # imported under audit

        loop = build_fixture_loop({})
    else:
        loop = load_eval_loop(lab, qcal_config).loop
    phase[0] = "evaluate"
    result = canonical(outputs(loop, predictions, ground_truth))
    report = {"outputs": result, "events": events["evaluate"], "build_events": events["build"]}
    sys.stdout.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], "--audit" in sys.argv[2:])
