"""``make smoke``: the whole Phase 1 loop on a synthetic fixture, through the real registry.

It builds a throwaway project with its own ``qcal.toml``, ``configs/lab.toml``, split
manifests, a 20-image synthetic dataset and an ``EXPERIMENTS.yaml`` of fixture cells. Then it
drives the real commands: ``qcal registry run-batch``, which runs ``python -m qcal_lab run``
per cell through the subprocess executor, then ``qcal registry index``,
``qcal registry audit --strict``, ``qcal leakage``, and a determinism re-run.

It never touches the repository's own registry. The fixture detector and the fixture
evaluation loop refuse real datasets, so nothing here can produce a reportable number.
"""

from __future__ import annotations

import io
import json
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qcal import cli as qcal_cli
from qcal.config import ConfigError, load_config
from qcal.log import get_logger
from qcal.registry.executor import sha256_file
from qcal.registry.store import RegistryStore
from qcal.reports import verdict
from qcal_lab.config import LAB_CONFIG_FILE, LabConfig, parse_lab_config
from qcal_lab.data.fixture import build_fixture, write_fixture
from qcal_lab.data.splits import partition, write_manifest
from qcal_lab.experiment import RunRequest, run_experiment

_log = get_logger("lab.smoke")

FIXTURE_ANNOTATIONS = "data/fixture/annotations.json"
CELL_PREFIX = "C-smoke-"
QcalMain = Callable[[list[str]], tuple[int, str]]
# Every successful smoke run records these (the default split names; see [splits]).
EXPECTED_ARTIFACTS = frozenset(
    {
        "predictions_raw_calibrator_fit_split",
        "predictions_raw_val",
        "predictions_raw_test",
        "calibration",
        "predictions_calibrated_test",
    }
)


@dataclass
class SmokeReport:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    seconds: float = 0.0
    project: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(ok for _, ok, _ in self.checks)

    def add(self, name: str, *, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, ok, detail))
        (_log.info if ok else _log.error)("smoke %s: %s %s", name, verdict(ok), detail)
        return ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": verdict(self.passed),
            "seconds": round(self.seconds, 3),
            "project": self.project,
            "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in self.checks],
        }

    def render_text(self) -> str:
        lines = [f"smoke: {verdict(self.passed)} in {self.seconds:.1f}s ({self.project})"]
        lines += [f"  {verdict(ok):4} {name}{f': {d}' if d else ''}" for name, ok, d in self.checks]
        return "\n".join(lines)


def _toml_list(values: list[str]) -> str:
    return json.dumps(values)  # JSON string arrays are valid TOML arrays


def _qcal(project: Path) -> QcalMain:
    def run(args: list[str]) -> tuple[int, str]:
        out = io.StringIO()
        code = qcal_cli.main(["--root", str(project), *args], out=out)
        return code, out.getvalue()

    return run


def _settings(lab: LabConfig) -> Mapping[str, Any]:
    return lab.table("smoke")


def write_project(project: Path, lab: LabConfig, python: str) -> list[str]:
    """Create the throwaway project; returns the cell ids it pre-registers."""
    s = _settings(lab)
    if project.exists() and any(project.iterdir()):
        raise ConfigError(f"{project} is not empty; the smoke test builds a fresh project")
    project.mkdir(parents=True, exist_ok=True)
    command = [
        python, "-m", "qcal_lab", "--root", "{root}", "run", "--run-id", "{run_id}",
        "--cell-id", "{cell_id}", "--seed", "{seed}", "--result-path", "{result_path}",
    ]  # fmt: skip
    (project / "qcal.toml").write_text(
        "[executor]\n"
        f"command = {_toml_list(command)}\n"
        f"timeout_s = {int(s['max_seconds'])}\n"
        "[registry]\n"
        f"env_collectors = {_toml_list([str(c) for c in s['env_collectors']])}\n"
        'write_parquet = "never"\n',
        encoding="utf-8",
    )
    fixture = build_fixture(
        images=int(s["images"]),
        categories=int(s["categories"]),
        max_objects_per_image=int(s["max_objects_per_image"]),
        seed=int(s["seed"]),
        width=int(s["image_width"]),
        height=int(s["image_height"]),
    )
    write_fixture(project / FIXTURE_ANNOTATIONS, fixture)

    config = load_config(project, environ={})
    sizes = s["split_sizes"]
    splits = config.str_list("data.splits")
    missing = [split for split in splits if split not in sizes]
    if missing:
        raise ConfigError(f"smoke.split_sizes has no size for {missing}")
    ids = [str(image["id"]) for image in fixture["images"]]
    parts = partition(ids, [(split, int(sizes[split])) for split in splits], int(s["seed"]))
    for split, members in parts.items():
        path = config.path("manifests_dir") / config.str_value("data.manifest_pattern").format(
            split=split
        )
        header = {
            "source": FIXTURE_ANNOTATIONS,
            "seed": str(s["seed"]),
            "made_by": "qcal_lab smoke",
        }
        write_manifest(path, members, header, config.str_value("data.comment_prefix"))

    lab_toml = project / LAB_CONFIG_FILE
    lab_toml.parent.mkdir(parents=True, exist_ok=True)
    lab_toml.write_text(
        "# Generated by `python -m qcal_lab smoke`; fixture stand-ins accept only the fixture.\n"
        f'[datasets.id]\nannotations = "{FIXTURE_ANNOTATIONS}"\nimages_dir = ""\n'
        '[eval_loop]\nmodule = "qcal_lab.fixture_eval"\nfactory = "build"\n',
        encoding="utf-8",
    )

    cells = [
        {
            "id": f"{CELL_PREFIX}{calibrator}-{scope}",
            "detector": "fixture",
            "calibrator": str(calibrator),
            "calibrator_scope": scope,
            "calibrator_fit_split_size": int(s["calibrator_fit_split_size"]),
            "precision": "fp32",
        }
        for calibrator in s["calibrators"]
        for scope in ("per_class", "global")
    ]
    experiments = {
        "version": 2,
        "question": "smoke test of the pipeline (synthetic fixture; no scientific content)",
        "seed_role": "calibrator_fit_draw",
        "seeds": [int(seed) for seed in s["seeds"]],
        "cells": cells,
    }
    (project / "EXPERIMENTS.yaml").write_text(
        yaml.safe_dump(experiments, sort_keys=False), encoding="utf-8"
    )
    return [str(c["id"]) for c in cells]


def _check_records(report: SmokeReport, project: Path, expected: int) -> list[Any]:
    config = load_config(project, environ={})
    records = RegistryStore(config.path("registry_dir")).load_all(strict=True)
    ok = [r for r in records if r.status == config.str_value("registry.ok_status")]
    failed = [f"{r.cell_id}: {r.error}" for r in records if r not in ok]
    report.add(
        "every pre-registered run succeeded",
        ok=len(ok) == expected == len(records),
        detail=f"{len(ok)}/{expected} ok" + (f"; {'; '.join(failed)}" if failed else ""),
    )
    bad: list[str] = []
    for record in ok:
        missing = EXPECTED_ARTIFACTS - {a.kind for a in record.artifacts}
        if missing:
            bad.append(f"{record.run_id}: no {', '.join(sorted(missing))}")
        for artifact in record.artifacts:
            path = project / artifact.path
            if not path.is_file() or sha256_file(path) != artifact.sha256:
                bad.append(f"{record.run_id}: {artifact.path} does not match its hash")
    report.add("artifacts recorded and hashed", ok=bool(ok) and not bad, detail="; ".join(bad[:3]))
    return ok


def _check_determinism(report: SmokeReport, project: Path, record: Any) -> None:
    """Re-run one cell with the prediction cache off, so the detector itself runs again."""
    lab_file = project / LAB_CONFIG_FILE
    text = lab_file.read_text("utf-8") + '[predictions]\ncache_dir = ""\n'
    uncached = parse_lab_config(project, text, origin=f"{lab_file} (cache off)")
    rerun_id = f"{record.run_id}-rerun"
    result = run_experiment(
        RunRequest(
            project, rerun_id, record.cell_id, record.seed, project / "runs/smoke" / rerun_id
        ),
        lab=LabConfig(uncached.config, lab_file, uncached.sha256),
    )
    recorded = {a.kind: a.sha256 for a in record.artifacts}
    rerun = {a["kind"]: sha256_file(project / a["path"]) for a in result.artifacts}
    same = result.metrics == dict(record.metrics) and rerun == recorded
    report.add(
        "an uncached re-run reproduces predictions, calibration and metrics byte for byte",
        ok=same and set(rerun) == EXPECTED_ARTIFACTS,
        detail=record.cell_id,
    )


def run_smoke(lab: LabConfig, workdir: Path, *, python: str = sys.executable) -> SmokeReport:
    started = time.monotonic()
    project = workdir / "project"
    report = SmokeReport(project=str(project))
    cells = write_project(project, lab, python)
    seeds = len(_settings(lab)["seeds"])
    qcal = _qcal(project)

    code, out = qcal(["registry", "run-batch", f"{CELL_PREFIX}*"])
    last = out.strip().splitlines()[-1] if out.strip() else ""
    report.add("qcal registry run-batch", ok=code == 0, detail=last)
    records = _check_records(report, project, len(cells) * seeds)
    code, _ = qcal(["registry", "index"])
    report.add("qcal registry index", ok=code == 0)
    code, out = qcal(["registry", "audit", "--strict", "--json"])
    audit = json.loads(out) if out.strip().startswith("{") else {}
    report.add(
        "qcal registry audit --strict: complete coverage",
        ok=code == 0 and audit.get("completed") == audit.get("expected") == len(cells) * seeds,
        detail=f"{audit.get('completed')}/{audit.get('expected')}",
    )
    code, out = qcal(["leakage", "--json"])
    report.add(
        "qcal leakage: splits are disjoint", ok=code == 0, detail=out.strip()[:120] if code else ""
    )
    if records:
        _check_determinism(report, project, records[0])
    report.seconds = time.monotonic() - started
    budget = float(_settings(lab)["max_seconds"])
    report.add(
        "time budget", ok=report.seconds <= budget, detail=f"{report.seconds:.1f}s of {budget:.0f}s"
    )
    return report


__all__ = ["CELL_PREFIX", "FIXTURE_ANNOTATIONS", "SmokeReport", "run_smoke", "write_project"]
