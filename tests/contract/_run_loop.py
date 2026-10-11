"""Subprocess helper: print a loop's canonical outputs, optionally with every file it opened.

``python -m tests.contract._run_loop <fixture|configured> [--audit]``. Run in a fresh
interpreter so the hash seed, and nothing else, differs between two runs.
"""

from __future__ import annotations

import json
import sys

from qcal.config import load_config
from qcal_lab.config import load_lab_config
from qcal_lab.evaluation import load_eval_loop
from qcal_lab.fixture_eval import build as build_fixture_loop
from tests.conftest import REPO_ROOT
from tests.contract.eval_loop_checks import canonical, fixture_inputs, outputs
from tests.lab_support import fixture_ground_truth


def main(which: str, audit: bool) -> None:
    lab = load_lab_config(REPO_ROOT)
    ground_truth = fixture_ground_truth()
    predictions = fixture_inputs(lab, ground_truth)
    if which == "fixture":
        loop = build_fixture_loop({})
    else:
        loop = load_eval_loop(lab, load_config(REPO_ROOT, environ={})).loop
    events: list[list[str]] = []
    if audit:

        def hook(event: str, args: tuple[object, ...]) -> None:
            if event == "open" and args and isinstance(args[0], str):
                events.append([event, args[0]])
            elif event.startswith("socket."):
                events.append([event, ""])

        sys.addaudithook(hook)
    result = canonical(outputs(loop, predictions, ground_truth))
    sys.stdout.write(json.dumps({"outputs": result, "events": events}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], "--audit" in sys.argv[2:])
