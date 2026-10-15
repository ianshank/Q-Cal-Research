"""``python -m qcal_lab status``: what still blocks a registered Phase 1 run, item by item.

The report is read-only and changes nothing. ``/reproduce-check`` uses it to say which of
Ian's hand tasks, configuration choices and oracle outputs are still missing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qcal.config import Config, ConfigError
from qcal.integrity.leakage import check_leakage
from qcal.registry.executor import sha256_file
from qcal.reports import FAIL, verdict
from qcal_lab.config import LabConfig
from qcal_lab.evaluation import HandwrittenMissingError, load_eval_loop
from qcal_lab.models import detector_problems
from qcal_lab.parity import KINDS, ParityError, load_cases
from qcal_lab.predictions import PredictionCache


@dataclass
class StatusReport:
    items: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.items)

    def add(self, name: str, *, ok: bool, detail: str) -> None:
        self.items.append((name, ok, detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.passed,
            "items": [{"name": n, "ready": ok, "detail": d} for n, ok, d in self.items],
        }

    def render_text(self) -> str:
        head = "ready" if self.passed else "not ready"
        lines = [f"phase 1 status: {head}"]
        lines += [
            f"  {'ok ' if ok else 'TODO'} {name}: {detail}" for name, ok, detail in self.items
        ]
        return "\n".join(lines)


def _setting_file(lab: LabConfig, dotted: str) -> tuple[bool, str]:
    try:
        path = lab.file(dotted)
    except ConfigError as exc:
        return False, str(exc)
    return path.exists(), str(path) if path.exists() else f"{path} does not exist"


def _detectors(report: StatusReport, lab: LabConfig) -> None:
    detectors = lab.config.get("detectors", {})
    for name, settings in sorted(detectors.items() if isinstance(detectors, Mapping) else []):
        if not isinstance(settings, Mapping) or settings.get("kind") == "fixture":
            continue
        problems = [
            detail
            for key in ("config", "checkpoint")
            for ok, detail in [_setting_file(lab, f"detectors.{name}.{key}")]
            if not ok
        ]
        expected = str(settings.get("checkpoint_sha256", ""))
        if not expected:
            problems.append("checkpoint_sha256 is not set")
        elif not problems:  # the checkpoint exists: its bytes must match the recorded digest
            checkpoint = lab.file(f"detectors.{name}.checkpoint")
            if sha256_file(checkpoint) != expected:
                problems.append(f"checkpoint_sha256 does not match {checkpoint}")
        try:
            problems.extend(detector_problems(lab, name))
        except ConfigError as exc:
            problems.append(str(exc))
        report.add(f"detector {name}", ok=not problems, detail="; ".join(problems) or "configured")
        _sanity(report, name)


def _contract(report: StatusReport, lab: LabConfig, qcal_config: Config) -> None:
    """The EvalLoop contract on Ian's loop, as ``python -m qcal_lab check-eval-loop`` runs it."""
    from qcal_lab.eval_contract import check_eval_loop
    from qcal_lab.smoke import smoke_settings

    contract = check_eval_loop(lab, qcal_config, smoke_settings())
    detail = contract.verdict + (f": {contract.must[0]}" if contract.must else "")
    report.add("evaluation loop contract", ok=contract.passed, detail=detail)


def _sanity(report: StatusReport, name: str) -> None:
    """The detector sanity check (D2) must be declared before a registered run, so it cannot
    be chosen after the fact; until D8 enables the measurement it stays NOT ASSESSED."""
    from qcal_lab.sanity import SanityError, check_detector

    try:
        sanity = check_detector(name)
    except SanityError as exc:
        report.add(f"detector sanity check {name}", ok=False, detail=str(exc))
        return
    ok = sanity.declared and sanity.verdict != FAIL
    report.add(f"detector sanity check {name}", ok=ok, detail=sanity.render_text())


def _quarantine(report: StatusReport, lab: LabConfig) -> None:
    """Quarantined predictions are a conflict nobody explained: a detector that is not
    deterministic, or a key that misses an input. Ian looks before registered runs."""
    configured = lab.config.str_value("predictions.cache_dir")
    if not configured:
        report.add("prediction cache", ok=True, detail="disabled")
        return
    directory = Path(configured)
    moved = PredictionCache(
        directory if directory.is_absolute() else lab.config.root / directory
    ).quarantined()
    detail = (
        f"{len(moved)} quarantined under {moved[0].parent.parent}: unexplained conflicts"
        if moved
        else "no quarantined predictions"
    )
    report.add("prediction cache", ok=not moved, detail=detail)


def build_status(qcal_config: Config, lab: LabConfig) -> StatusReport:
    report = StatusReport()
    report.add(
        "lab configuration",
        ok=lab.path is not None,
        detail=f"{lab.path} ({lab.sha256[:12]})" if lab.path else "configs/lab.toml does not exist",
    )
    experiments = qcal_config.path("experiments")
    report.add("pre-registration", ok=experiments.is_file(), detail=str(experiments))
    command = qcal_config.str_list("executor.command")
    report.add(
        "executor.command", ok=bool(command), detail=" ".join(command) or "empty (qcal.toml)"
    )
    ok, detail = _setting_file(lab, "datasets.id.annotations")
    report.add("in-domain annotations", ok=ok, detail=detail)
    ok, detail = _setting_file(lab, "datasets.id.images_dir")
    report.add("in-domain images", ok=ok, detail=detail)
    leakage = check_leakage(qcal_config)
    detail = f"missing {', '.join(leakage.missing)}" if leakage.missing else verdict(leakage.passed)
    report.add("split manifests", ok=leakage.passed, detail=detail)
    try:
        load_eval_loop(lab, qcal_config)
        report.add("evaluation loop (Ian)", ok=True, detail=lab.text("eval_loop.module"))
        _contract(report, lab, qcal_config)
    except HandwrittenMissingError:
        missing = f"{lab.text('eval_loop.module')} not written"
        report.add("evaluation loop (Ian)", ok=False, detail=missing)
    except (ConfigError, ImportError) as exc:
        report.add("evaluation loop (Ian)", ok=False, detail=f"cannot load: {exc}")
    _detectors(report, lab)
    _quarantine(report, lab)
    try:
        cases = load_cases(lab.config.root / lab.text("parity.fixtures_dir"))
    except ParityError as exc:
        report.add("oracle parity cases", ok=False, detail=str(exc))
    else:
        # Calibrator and metric parity are separate gates: each kind needs at least one case.
        kinds = {k: sum(c.kind == k for c in cases) for k in KINDS}
        shown = ", ".join(f"{n} {k}" for k, n in kinds.items())
        absent = [k for k, n in kinds.items() if not n]
        detail = f"{shown}; none of kind {', '.join(absent)}" if absent else shown
        report.add("oracle parity cases", ok=not absent, detail=detail if cases else "none yet")
    return report


__all__ = ["StatusReport", "build_status"]
