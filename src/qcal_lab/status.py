"""``python -m qcal_lab status``: what still blocks a registered Phase 1 run, item by item.

The report is read-only and changes nothing. ``/reproduce-check`` uses it to say which of
Ian's hand tasks, configuration choices and oracle outputs are still missing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from qcal.config import Config, ConfigError
from qcal.integrity.leakage import check_leakage
from qcal.reports import verdict
from qcal_lab.config import LabConfig
from qcal_lab.evaluation import HandwrittenMissingError, load_eval_loop
from qcal_lab.parity import ParityError, load_cases


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
        if not settings.get("checkpoint_sha256"):
            problems.append("checkpoint_sha256 is not set")
        report.add(f"detector {name}", ok=not problems, detail="; ".join(problems) or "configured")


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
    leakage = check_leakage(qcal_config)
    detail = f"missing {', '.join(leakage.missing)}" if leakage.missing else verdict(leakage.passed)
    report.add("split manifests", ok=leakage.passed, detail=detail)
    try:
        load_eval_loop(lab)
        report.add("evaluation loop (Ian)", ok=True, detail=lab.text("eval_loop.module"))
    except HandwrittenMissingError:
        missing = f"{lab.text('eval_loop.module')} not written"
        report.add("evaluation loop (Ian)", ok=False, detail=missing)
    except (ConfigError, ImportError) as exc:
        report.add("evaluation loop (Ian)", ok=False, detail=f"cannot load: {exc}")
    _detectors(report, lab)
    try:
        cases = load_cases(lab.config.root / lab.text("parity.fixtures_dir"))
    except ParityError as exc:
        report.add("oracle parity cases", ok=False, detail=str(exc))
    else:
        kinds = {k: sum(c.kind == k for c in cases) for k in ("calibrator", "metric")}
        shown = ", ".join(f"{n} {k}" for k, n in kinds.items())
        report.add("oracle parity cases", ok=bool(cases), detail=shown if cases else "none yet")
    return report


__all__ = ["StatusReport", "build_status"]
