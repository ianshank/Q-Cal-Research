"""The experiment program the registry runs: one pre-registered cell and seed, one result file.

``qcal registry run <cell> --seed <s>`` executes ``executor.command`` from ``qcal.toml``, which
runs ``python -m qcal_lab run``. The program:

1. reads the cell's factors from EXPERIMENTS.yaml, refusing unknown factors and values;
2. loads the dataset and the split manifests, refusing any evaluate split that is also
   used for fitting or selection;
3. runs the detector on the fit, select and evaluate splits (raw predictions are cached);
4. trains the calibration on the fit and select roles only (arXiv:2405.20459 Alg. A.1);
5. applies it to the evaluate split (Alg. A.2) and asks Ian's evaluation loop for the
   metrics;
6. writes the executor contract's result JSON: metrics, artifacts, environment.

The registry, not this program, writes the run record.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from qcal.config import Config, ConfigError, load_config
from qcal.log import get_logger
from qcal.protocols import Detector, ImageDetections
from qcal.registry.experiments import load_experiments
from qcal_lab import __version__
from qcal_lab.calib import CALIBRATORS, build_calibrator
from qcal_lab.calib.thresholds import threshold_grid
from qcal_lab.calib.two_threshold import (
    SCOPES,
    SplitData,
    ThresholdedCalibration,
    train_calibration,
)
from qcal_lab.config import LabConfig, load_lab_config, require_hashed
from qcal_lab.data.coco import GroundTruth, load_coco
from qcal_lab.data.splits import draw, read_split, split_digest
from qcal_lab.evaluation import EvalLoop, check_metrics, load_eval_loop
from qcal_lab.models import build_detector, detector_fingerprint
from qcal_lab.predictions import (
    PredictionCache,
    PredictionsHeader,
    cache_key,
    read_predictions,
    write_predictions,
)

_log = get_logger("lab.experiment")

ROLES: Final = ("fit", "select", "evaluate")
THRESHOLD_ROLES: Final = ("fit", "select")
_FACTOR_ROLES: Final = (
    "detector",
    "calibrator",
    "calibrator_scope",
    "calibrator_fit_split_size",
    "shift",
    "precision",
)


class PlanError(ValueError):
    """The cell asks for something this program does not do, or the splits are unsafe."""


@dataclass(frozen=True)
class RunRequest:
    root: Path
    run_id: str
    cell_id: str
    seed: int
    result_path: Path


@dataclass(frozen=True)
class RunPlan:
    detector: str
    calibrator: str
    scope: str
    fit_size: int | None
    shift: str
    precision: str


@dataclass
class ExperimentResult:
    metrics: dict[str, float]
    artifacts: list[dict[str, str]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "environment": self.environment,
        }


def _factor_names(lab: LabConfig) -> dict[str, str]:
    return {role: lab.text(f"factors.{role}") for role in _FACTOR_ROLES}


def resolve_plan(factors: Mapping[str, Any], lab: LabConfig) -> RunPlan:
    """Map a cell's factors to a plan, refusing anything the program would not honour."""
    names = _factor_names(lab)
    defaults = lab.table("factors.defaults")
    supported = lab.table("factors.supported")
    passthrough = set(lab.config.str_list("factors.passthrough"))
    unknown = sorted(
        k
        for k in factors
        if k not in names.values() and k not in supported and k not in passthrough
    )
    if unknown:
        raise PlanError(f"cell factors this program does not understand: {', '.join(unknown)}")
    effective = {role: factors.get(name, defaults.get(role)) for role, name in names.items()}
    role_of = {name: role for role, name in names.items()}
    for key, allowed in supported.items():
        if not isinstance(allowed, list):
            raise ConfigError(f"factors.supported.{key} must be a list")
        current = effective[role_of[key]] if key in role_of else factors.get(key)
        if current is not None and current not in allowed:
            raise PlanError(f"factor {key}={current!r} is not supported (allowed: {allowed})")

    def value(role: str) -> Any:
        return effective[role]

    detector = value("detector")
    if not isinstance(detector, str) or not detector:
        raise PlanError(f"the cell must name a {names['detector']!r}")
    calibrator = value("calibrator")
    if not isinstance(calibrator, str) or calibrator not in CALIBRATORS:
        raise PlanError(f"unknown calibrator {calibrator!r} (known: {CALIBRATORS.names()})")
    scope = value("calibrator_scope")
    if scope not in SCOPES:
        raise PlanError(f"calibrator scope must be one of {SCOPES}, got {scope!r}")
    size = value("calibrator_fit_split_size")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
        raise PlanError(f"calibrator_fit_split_size must be a positive integer, got {size!r}")
    shift = value("shift")
    datasets = lab.config.get("datasets", {})
    if not isinstance(shift, str) or not isinstance(datasets, Mapping) or shift not in datasets:
        raise PlanError(f"shift {shift!r} has no datasets.{shift} entry")
    precision = value("precision")
    if not isinstance(precision, str):
        raise PlanError("precision must be a string")
    return RunPlan(detector, calibrator, str(scope), size, shift, precision)


def split_roles(lab: LabConfig, qcal_config: Config) -> dict[str, str]:
    """Role -> split name; the evaluate split may never also fit or select."""
    roles = {role: lab.text(f"splits.{role}") for role in ROLES}
    known = qcal_config.str_list("data.splits")
    unknown = [s for s in roles.values() if s not in known]
    if unknown:
        raise PlanError(f"splits {unknown} are not in data.splits {known}")
    if roles["evaluate"] in (roles["fit"], roles["select"]):
        raise PlanError(
            f"the evaluate split {roles['evaluate']!r} is also used for fitting or selection; "
            "that is tuning on the test split (CLAUDE.md rule 3)"
        )
    for key in ("calibration_threshold_role", "operating_threshold_role"):
        if lab.text(f"calibration.{key}") not in THRESHOLD_ROLES:
            raise PlanError(f"calibration.{key} must be one of {THRESHOLD_ROLES}")
    return roles


@dataclass
class _Context:
    request: RunRequest
    qcal_config: Config
    lab: LabConfig
    plan: RunPlan
    dataset: GroundTruth
    artifact_dir: Path
    artifacts: list[dict[str, str]] = field(default_factory=list)

    def relative(self, path: Path) -> str:
        root = self.qcal_config.root
        return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)

    def add_artifact(self, path: Path, kind: str) -> None:
        self.artifacts.append({"path": self.relative(path), "kind": kind})


def _artifact_dir(lab: LabConfig, qcal_config: Config, run_id: str) -> Path:
    template = lab.text("predictions.artifact_dir")
    rendered = template.format(
        results_dir=qcal_config.str_value("paths.results_dir"), run_id=run_id
    )
    path = Path(rendered)
    return path if path.is_absolute() else qcal_config.root / path


def _fingerprint(lab: LabConfig, plan: RunPlan) -> dict[str, Any]:
    return detector_fingerprint(
        lab,
        plan.detector,
        plan.precision,
        file_settings=lab.config.str_list("predictions.fingerprint_file_settings"),
        packages=lab.config.str_list("predictions.fingerprint_packages"),
    )


def _cache(lab: LabConfig) -> PredictionCache:
    configured = lab.config.str_value("predictions.cache_dir")
    if not configured:
        return PredictionCache(None)
    path = Path(configured)
    return PredictionCache(path if path.is_absolute() else lab.config.root / path)


def _raw_predictions(
    ctx: _Context, detector: Detector, split: str, ids: Sequence[str], cache: PredictionCache
) -> tuple[ImageDetections, ...]:
    header = PredictionsHeader(
        detector=ctx.plan.detector,
        split=split,
        stage="raw",
        dataset_sha256=ctx.dataset.sha256,
        split_sha256=split_digest(ids),
        category_ids=ctx.dataset.category_ids,
    )
    key = cache_key(
        detector=_fingerprint(ctx.lab, ctx.plan),
        dataset_sha256=ctx.dataset.sha256,
        split_sha256=header.split_sha256,
    )
    cached = cache.get(key)
    if cached is not None:
        _log.info("using cached raw predictions for %s (%s)", split, cached.name)
        _, images = read_predictions(cached)
        ctx.add_artifact(cached, f"predictions_raw_{split}")
        return images
    _log.info("running %s on %d %s images", ctx.plan.detector, len(ids), split)
    images = tuple(detector.predict(list(ids)))
    if [i.image_id for i in images] != list(ids):
        raise PlanError(f"{ctx.plan.detector} returned images out of order or incomplete")
    path = cache.path(key)
    if path is None:
        path = ctx.artifact_dir / f"raw_{split}.jsonl"
        write_predictions(path, header, images)
    else:
        cache.put(key, header, images)
    ctx.add_artifact(path, f"predictions_raw_{split}")
    return images


def _subset(images: Sequence[ImageDetections], ids: Sequence[str]) -> tuple[ImageDetections, ...]:
    wanted = set(ids)
    return tuple(i for i in images if i.image_id in wanted)


def run_experiment(
    request: RunRequest,
    *,
    qcal_config: Config | None = None,
    lab: LabConfig | None = None,
    eval_loop: EvalLoop | None = None,
) -> ExperimentResult:
    qcal_config = qcal_config or load_config(request.root)
    lab = lab or load_lab_config(qcal_config.root)
    require_hashed(lab, qcal_config)
    experiments = load_experiments(qcal_config)
    cell = experiments.cell(request.cell_id)
    if request.seed not in experiments.seeds_for(cell):
        raise PlanError(f"seed {request.seed} is not pre-registered for {cell.id}")
    plan = resolve_plan(cell.factors, lab)
    roles = split_roles(lab, qcal_config)
    loop = eval_loop or load_eval_loop(lab)  # fail before any detector work if it is missing
    dataset = load_coco(lab.file(f"datasets.{plan.shift}.annotations"))
    ids = {role: read_split(qcal_config, split) for role, split in roles.items()}
    subsets = {role: dataset.subset(role_ids) for role, role_ids in ids.items()}
    fit_ids = draw(ids["fit"], plan.fit_size, request.seed) if plan.fit_size else ids["fit"]
    images_setting = lab.config.str_value(f"datasets.{plan.shift}.images_dir")
    ctx = _Context(
        request, qcal_config, lab, plan, dataset, _artifact_dir(lab, qcal_config, request.run_id)
    )
    detector = build_detector(
        lab,
        plan.detector,
        dataset,
        precision=plan.precision,
        images_dir=lab.file(f"datasets.{plan.shift}.images_dir") if images_setting else None,
    )
    cache = _cache(lab)
    by_split: dict[str, tuple[ImageDetections, ...]] = {}
    for role in ROLES:
        split = roles[role]
        if split not in by_split:
            by_split[split] = _raw_predictions(ctx, detector, split, ids[role], cache)
    raw = {role: by_split[roles[role]] for role in ROLES}
    data = {
        "fit": SplitData(
            roles["fit"], _subset(raw["fit"], fit_ids), subsets["fit"].subset(fit_ids)
        ),
        "select": SplitData(roles["select"], raw["select"], subsets["select"]),
    }
    candidates = threshold_grid(
        lab.config.float_value("calibration.grid_start"),
        lab.config.float_value("calibration.grid_stop"),
        lab.config.float_value("calibration.grid_step"),
    )
    calibration = train_calibration(
        fit=data["fit"],
        calibration_threshold_data=data[lab.text("calibration.calibration_threshold_role")],
        operating_threshold_data=data[lab.text("calibration.operating_threshold_role")],
        build=lambda: build_calibrator(plan.calibrator, lab),
        scope=plan.scope,
        candidates=candidates,
        eval_loop=loop,
    )
    evaluated = calibration.apply(raw["evaluate"])
    metrics = check_metrics(loop.metrics(evaluated, subsets["evaluate"]))
    _write_outputs(ctx, roles["evaluate"], ids["evaluate"], calibration, evaluated)
    environment = {
        "qcal_lab_version": __version__,
        "lab_config_sha256": lab.sha256,
        "eval_loop": f"{lab.text('eval_loop.module')}:{lab.text('eval_loop.factory')}",
        "dataset_sha256": dataset.sha256,
        "splits": {
            role: {
                "split": roles[role],
                "sha256": split_digest(ids[role]),
                "images": len(ids[role]),
            }
            for role in ROLES
        },
        "fit_draw": {
            "size": len(fit_ids),
            "seed": request.seed,
            "sha256": split_digest(fit_ids),
        },
        "detector": _fingerprint(lab, plan),
    }
    return ExperimentResult(metrics, ctx.artifacts, environment)


def _write_outputs(
    ctx: _Context,
    split: str,
    ids: Sequence[str],
    calibration: ThresholdedCalibration,
    evaluated: Sequence[ImageDetections],
) -> None:
    ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = ctx.artifact_dir / "calibration.json"
    calibration_path.write_text(
        json.dumps(calibration.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ctx.add_artifact(calibration_path, "calibration")
    header = PredictionsHeader(
        detector=ctx.plan.detector,
        split=split,
        stage="calibrated",
        dataset_sha256=ctx.dataset.sha256,
        split_sha256=split_digest(ids),
        category_ids=ctx.dataset.category_ids,
    )
    path = ctx.artifact_dir / f"calibrated_{split}.jsonl"
    write_predictions(path, header, evaluated)
    ctx.add_artifact(path, f"predictions_calibrated_{split}")


def write_result(path: Path, result: ExperimentResult) -> None:
    """Write the executor contract's result JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


__all__ = [
    "ROLES",
    "ExperimentResult",
    "PlanError",
    "RunPlan",
    "RunRequest",
    "resolve_plan",
    "run_experiment",
    "split_roles",
    "write_result",
]
