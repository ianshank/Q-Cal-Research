"""``python -m qcal_lab check-eval-loop`` and the contract checks it runs (PR-F, wave 3).

Ian's loop does not exist yet, so the configured loop is NOT ASSESSED; a stand-in replaces
``load_eval_loop`` to exercise the PASS and FAIL paths. No real metric is ever computed.
"""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qcal.config import load_config
from qcal_lab import cli, eval_contract
from qcal_lab.config import load_lab_config
from qcal_lab.fixture_eval import build as build_fixture_loop
from qcal_lab.smoke import smoke_settings
from tests.conftest import REPO_ROOT
from tests.lab_support import ians_loop_absent

pytestmark = pytest.mark.rule("C4")


def _check(**kwargs: Any) -> eval_contract.ContractReport:
    lab = load_lab_config(REPO_ROOT)
    return eval_contract.check_eval_loop(
        lab, load_config(REPO_ROOT, environ={}), smoke_settings(), **kwargs
    )


def _with_loop(monkeypatch: pytest.MonkeyPatch, factory: Any) -> None:
    """Pretend Ian's loop exists and is built by ``factory``."""
    monkeypatch.setattr(eval_contract, "load_eval_loop", lambda *_: SimpleNamespace(loop=factory()))


class NaNObjective:
    """Honours the MUST contract, but its objective is NaN on every class with detections."""

    def __init__(self) -> None:
        self.inner = build_fixture_loop({})

    def targets(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
        return self.inner.targets(predictions, ground_truth)

    def threshold_objective(self, predictions, ground_truth, *, label, stage):  # type: ignore[no-untyped-def]
        return math.nan

    def metrics(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
        return self.inner.metrics(predictions, ground_truth)


def test_the_fixture_stand_in_passes_the_contract() -> None:
    report = _check(fixture=True)
    assert (report.verdict, report.must, report.should) == ("PASS", [], [])
    assert report.version == eval_contract.CONTRACT_VERSION == 1


def test_ians_loop_is_not_assessed_until_it_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    ians_loop_absent(monkeypatch)
    report = _check()
    assert report.verdict == eval_contract.NOT_ASSESSED
    assert report.loop == load_lab_config(REPO_ROOT).text("eval_loop.module")


def test_a_loop_breaking_the_contract_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class Mutable(NaNObjective):
        calls = 0

        def metrics(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
            Mutable.calls += 1
            return {"n": float(Mutable.calls)}  # not stable across calls

    _with_loop(monkeypatch, Mutable)
    report = _check()
    assert report.verdict == "FAIL"
    assert "two calls on the same loop disagree" in report.must
    assert report.should == []  # SHOULD is only reported for a loop that passes MUST


def test_should_findings_are_reported_not_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_loop(monkeypatch, NaNObjective)
    report = _check()
    assert report.verdict == "PASS"
    assert any("is nan on a class with detections" in f for f in report.should)


def test_a_loop_depending_on_the_image_order_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FirstImageOnly(NaNObjective):
        def metrics(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
            return {"n": float(len(predictions[0].detections))}  # order matters

    _with_loop(monkeypatch, FirstImageOnly)
    report = _check()
    assert report.verdict == "FAIL"
    assert report.must == ["the loop's results depend on the order the images are listed in"]


def test_a_sum_taken_in_listed_order_is_not_order_dependence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Panel round 3: a correct loop that adds per-image values in the order listed differs in
    the last bit when the images are reversed (0.1 + 0.2 + 0.3 != 0.3 + 0.2 + 0.1)."""
    assert 0.1 + 0.2 + 0.3 != 0.3 + 0.2 + 0.1  # the premise, in IEEE doubles

    class SumsInListedOrder(NaNObjective):
        def metrics(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
            first = sorted(ground_truth.images)[:3]
            value = dict(zip(first, (0.1, 0.2, 0.3), strict=True))
            total = 0.0
            for image in predictions:  # naive accumulation, in listed order
                total += value.get(image.image_id, 0.0)
            return {"total": total}

    _with_loop(monkeypatch, SumsInListedOrder)
    report = _check()
    assert report.verdict == "PASS", report.must


class Scalar:
    """Stands in for a numpy scalar: a number that is not a Python float."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        ({"a": [0.6000000000000001]}, {"a": [0.6]}, True),  # rounding
        ({"a": 0.6}, {"a": 0.6 + 1e-9}, False),  # a real difference
        ({"a": math.nan}, {"a": math.nan}, True),
        ({"a": math.nan}, {"a": 0.0}, False),
        ({"a": [1.0]}, {"a": [1.0, 2.0]}, False),
        ({"a": 1.0}, {"b": 1.0}, False),
        ({"a": Scalar(0.25)}, {"a": 0.25}, True),
        ({"a": "x"}, {"a": "y"}, False),
    ],
)
def test_reordered_results_are_compared_up_to_rounding(first: Any, second: Any, same: bool) -> None:
    assert eval_contract.close(first, second, *eval_contract.reorder_tolerance()) is same


def test_numpy_like_numbers_do_not_break_the_comparison() -> None:
    assert eval_contract.canonical({"m": Scalar(0.5), "n": [Scalar(math.nan)]}) == (
        '{"m":0.5,"n":["nan"]}'
    )


class Fixture(NaNObjective):
    """The fixture loop's objective too: SHOULD-clean unless a test overrides a part."""

    def threshold_objective(self, predictions, ground_truth, *, label, stage):  # type: ignore[no-untyped-def]
        return self.inner.threshold_objective(predictions, ground_truth, label=label, stage=stage)


@pytest.mark.parametrize(
    ("override", "finding"),
    [
        (
            {
                "threshold_objective": lambda self, predictions, ground_truth, *, label, stage: (
                    float(sum(len(i.detections) for i in predictions))  # nothing kept is best
                )
            },
            "keeping no detections scores better than keeping them all",
        ),
        (
            {
                "targets": lambda self, predictions, ground_truth: {
                    i.image_id: [0.0] * len(i.detections) for i in predictions
                }
            },
            "every target is 0",
        ),
        (
            {
                "metrics": lambda self, predictions, ground_truth: {
                    "images_listed": float(len(predictions))
                }
            },
            "change when images without detections are left out",
        ),
    ],
)
def test_degenerate_but_allowed_loops_are_reported(
    monkeypatch: pytest.MonkeyPatch, override: dict[str, Any], finding: str
) -> None:
    loop = type("Degenerate", (Fixture,), override)
    _with_loop(monkeypatch, loop)
    report = _check()
    assert report.verdict == "PASS"  # SHOULD findings never fail the contract
    assert [f for f in report.should if finding in f]
    assert not [f for f in report.should if re.search(r"\d\.\d", f)]  # no value printed


def test_a_clean_loop_has_no_should_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_loop(monkeypatch, Fixture)
    assert _check().should == []


def test_a_loop_without_metrics_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class Silent(NaNObjective):
        def threshold_objective(self, predictions, ground_truth, *, label, stage):  # type: ignore[no-untyped-def]
            return 0.0

        def metrics(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
            return {}

    _with_loop(monkeypatch, Silent)
    report = _check()
    assert (report.verdict, report.must) == (
        "FAIL",
        ["metrics: the evaluation loop returned no metrics"],
    )


def test_a_factory_returning_something_else_is_caught() -> None:
    lab = load_lab_config(REPO_ROOT)
    ground_truth = eval_contract.contract_fixture(smoke_settings())
    predictions = eval_contract.fixture_inputs(lab, ground_truth)
    assert eval_contract.must_problems(object, predictions, ground_truth) == [
        "the factory did not return an EvalLoop"
    ]
    built = iter([build_fixture_loop({}), object()])
    problems = eval_contract.must_problems(lambda: next(built), predictions, ground_truth)
    assert problems == ["the factory's second call did not return an EvalLoop"]


def test_a_loop_mutating_its_inputs_is_caught() -> None:
    lab = load_lab_config(REPO_ROOT)
    ground_truth = eval_contract.contract_fixture(smoke_settings())  # fresh: the loop edits it
    predictions = eval_contract.fixture_inputs(lab, ground_truth)

    class Mutates(NaNObjective):
        def metrics(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
            ground_truth.info["touched"] = True  # type: ignore[index]
            return self.inner.metrics(predictions, ground_truth)

    assert "the loop mutated its inputs" in eval_contract.must_problems(
        Mutates, predictions, ground_truth
    )


def test_two_loops_from_one_factory_must_agree() -> None:
    lab = load_lab_config(REPO_ROOT)
    ground_truth = eval_contract.contract_fixture(smoke_settings())
    predictions = eval_contract.fixture_inputs(lab, ground_truth)
    builds = iter(range(2))

    class PerBuild(NaNObjective):
        def __init__(self) -> None:
            super().__init__()
            self.build = next(builds)  # stable per loop, different per build

        def metrics(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
            return {"n": float(self.build)}

    assert eval_contract.must_problems(PerBuild, predictions, ground_truth) == [
        "two loops from the same factory disagree"
    ]


def test_a_loop_raising_on_a_later_call_is_a_finding() -> None:
    """Every call is probed, not only the first (Copilot, PR #5)."""
    lab = load_lab_config(REPO_ROOT)
    ground_truth = eval_contract.contract_fixture(smoke_settings())
    predictions = eval_contract.fixture_inputs(lab, ground_truth)

    class SecondCallRaises(NaNObjective):
        calls = 0

        def metrics(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
            SecondCallRaises.calls += 1
            if SecondCallRaises.calls > 1:
                raise RuntimeError("late")
            return self.inner.metrics(predictions, ground_truth)

    assert eval_contract.must_problems(SecondCallRaises, predictions, ground_truth) == [
        "the loop's second call raised RuntimeError: late"
    ]
    builds = [0]

    def factory_fails_second_time() -> object:
        builds[0] += 1
        if builds[0] > 1:
            raise OSError("gone")
        return build_fixture_loop({})

    assert eval_contract.must_problems(factory_fails_second_time, predictions, ground_truth) == [
        "the factory's second call raised OSError: gone"
    ]


def test_a_loop_raising_while_its_should_part_runs_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = [0]

    class ThirdBuildRaises(NaNObjective):
        def __init__(self) -> None:
            calls[0] += 1
            if calls[0] > 3:  # the existence check and two MUST builds; the fourth is SHOULD's
                raise RuntimeError("third")
            super().__init__()

    monkeypatch.setattr(
        eval_contract, "load_eval_loop", lambda *_: SimpleNamespace(loop=ThirdBuildRaises())
    )
    report = _check()
    assert report.verdict == "FAIL"
    assert report.must == ["the loop raised RuntimeError: third (while its SHOULD part ran)"]


def _lab(*args: str) -> tuple[int, str]:
    out = io.StringIO()
    return cli.main(["--root", str(REPO_ROOT), *args], out=out), out.getvalue()


def test_the_command_exits_by_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    ians_loop_absent(monkeypatch)
    code, text = _lab("check-eval-loop")
    assert (code, "NOT ASSESSED" in text) == (cli.EXIT_OK, True)
    assert _lab("check-eval-loop", "--strict")[0] == cli.EXIT_FAILED
    code, text = _lab("check-eval-loop", "--fixture", "--json")
    assert (code, json.loads(text)["verdict"]) == (cli.EXIT_OK, "PASS")

    class Raises(NaNObjective):
        def targets(self, predictions, ground_truth):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    _with_loop(monkeypatch, Raises)
    code, text = _lab("check-eval-loop")
    assert code == cli.EXIT_FAILED
    assert "MUST   the loop raised RuntimeError: boom" in text


def test_status_runs_the_contract_once_ians_loop_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    from qcal_lab import status

    _with_loop(monkeypatch, NaNObjective)
    monkeypatch.setattr(status, "load_eval_loop", lambda *_: None)  # "Ian's loop" loads
    report = status.build_status(load_config(REPO_ROOT, environ={}), load_lab_config(REPO_ROOT))
    (item,) = [i for i in report.to_dict()["items"] if i["name"] == "evaluation loop contract"]
    assert (item["ready"], item["detail"]) == (True, "PASS")


def test_the_checks_no_longer_live_with_their_tests() -> None:
    assert not (REPO_ROOT / "tests/contract/eval_loop_checks.py").exists()
    assert Path(eval_contract.__file__).parent.name == "qcal_lab"
