"""The EvalLoop contract, checked on the fixture stand-in and on Ian's loop when it exists.

The fixture loop always runs, which validates the contract checks themselves. Ian's loop
(``eval_loop.module``) is reported NOT ASSESSED until ``handwritten/eval_loop.py`` exists;
once it exists, a loop that cannot be loaded or breaks the contract fails here. CLAUDE.md
rule 4: agents test Ian's file and report deviations; they never edit it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from qcal.config import load_config
from qcal_lab.config import load_lab_config
from qcal_lab.evaluation import EvalLoop, HandwrittenMissingError, load_eval_loop
from qcal_lab.fixture_eval import build as build_fixture_loop
from tests.conftest import REPO_ROOT
from tests.contract.eval_loop_checks import fixture_inputs, must_problems
from tests.lab_support import fixture_ground_truth

pytestmark = pytest.mark.rule("C4")

LOOPS = ("fixture", "configured")
#: Files a loop may open while it is imported and built: its code, and qcal_lab's packaged
#: tooling settings (the fixture loop's hit IoU). Nothing a split could be read from.
CODE_SUFFIXES = (".py", ".pyc", ".so", ".pth")
TOOLING_FILE = str(Path("qcal_lab") / "resources" / "tooling.toml")


def loop_factory(which: str) -> Callable[[], EvalLoop]:
    if which == "fixture":
        return lambda: build_fixture_loop({})
    lab = load_lab_config(REPO_ROOT)
    qcal_config = load_config(REPO_ROOT, environ={})
    try:
        load_eval_loop(lab, qcal_config)
    except HandwrittenMissingError:
        pytest.skip("NOT ASSESSED: Ian's evaluation loop is not written yet (CLAUDE.md rule 4)")
    return lambda: load_eval_loop(lab, qcal_config).loop


@pytest.mark.parametrize("which", LOOPS)
def test_the_loop_honours_the_must_contract(which: str) -> None:
    build = loop_factory(which)
    lab = load_lab_config(REPO_ROOT)
    ground_truth = fixture_ground_truth()
    assert must_problems(build, fixture_inputs(lab, ground_truth), ground_truth) == []


def _run(which: str, *flags: str, hash_seed: str = "0") -> dict[str, object]:
    out = subprocess.run(
        [sys.executable, "-m", "tests.contract._run_loop", which, *flags],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.integration
@pytest.mark.parametrize("which", LOOPS)
def test_the_loop_does_not_depend_on_the_hash_seed(which: str) -> None:
    loop_factory(which)  # skips while Ian's loop does not exist
    assert _run(which, hash_seed="0")["outputs"] == _run(which, hash_seed="1")["outputs"]


@pytest.mark.integration
@pytest.mark.parametrize("which", LOOPS)
def test_the_loop_reads_no_files_and_opens_no_sockets(which: str) -> None:
    """Everything the loop needs is in its arguments; reading a manifest or the annotations
    behind the pipeline's back could reach the evaluate split (CLAUDE.md rule 3)."""
    loop_factory(which)
    report = _run(which, "--audit")
    events, built = report["events"], report["build_events"]
    assert isinstance(events, list)
    assert events == [], events
    assert isinstance(built, list)
    # Building may load code and the packaged tooling settings, and nothing else: no
    # manifest, annotations or socket that a loop could read at import and keep.
    unexpected = [e for e in built if not _code_or_tooling(e)]
    assert unexpected == [], unexpected


def _code_or_tooling(event: list[str]) -> bool:
    name, path = event
    return name == "open" and (path.endswith(CODE_SUFFIXES) or path.endswith(TOOLING_FILE))


def test_the_checks_catch_a_broken_loop() -> None:
    """The contract checks themselves: a loop that breaks each MUST rule is caught."""
    lab = load_lab_config(REPO_ROOT)
    ground_truth = fixture_ground_truth()
    predictions = fixture_inputs(lab, ground_truth)

    class Broken:
        calls = 0

        def targets(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
            return {i.image_id: [2.0] * len(i.detections) for i in predictions}

        def threshold_objective(self, predictions, ground_truth, *, label, stage):  # type: ignore[no-untyped-def]
            Broken.calls += 1
            return True

        def metrics(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
            return {"bad name": float(Broken.calls)}

    problems = must_problems(Broken, predictions, ground_truth)
    assert any(p.startswith("targets:") for p in problems)
    assert any("not a number" in p for p in problems)
    assert any(p.startswith("metrics:") for p in problems)
    assert "two calls on the same loop disagree" in problems


def test_a_raising_loop_is_a_finding() -> None:
    lab = load_lab_config(REPO_ROOT)
    ground_truth = fixture_ground_truth()

    class Raises:
        def targets(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        def threshold_objective(self, predictions, ground_truth, *, label, stage):  # type: ignore[no-untyped-def]
            return 0.0

        def metrics(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
            return {"n": 0.0}

    assert must_problems(Raises, fixture_inputs(lab, ground_truth), ground_truth) == [
        "the loop raised RuntimeError: boom"
    ]
