"""The experiment program the registry runs: one pre-registered cell and seed, one result file.

``qcal registry run <cell> --seed <s>`` executes ``executor.command`` from ``qcal.toml``, which
runs ``python -m qcal_lab run``. The program:

1. reads the cell's factors from EXPERIMENTS.yaml, refusing unknown factors and values;
2. loads the dataset and the split manifests, refusing unless ``qcal leakage`` passes and the
   fit, select and evaluate splits share no image;
3. runs the detector on the three splits (raw predictions are cached and verified);
4. fits calibrators on the fit split and selects both thresholds on the select split
   (arXiv:2405.20459 Alg. A.1);
5. applies it to the evaluate split (Alg. A.2) and asks Ian's evaluation loop for the
   metrics;
6. writes the executor contract's result JSON: metrics, artifacts, environment.

The registry, not this program, writes the run record.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import itertools
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from qcal.config import REPO_CONFIG_NAME, Config, ConfigError, environment_problems, load_config
from qcal.integrity.leakage import check_leakage
from qcal.log import get_logger
from qcal.protocols import Detector, ImageDetections
from qcal.registry.executor import sha256_file
from qcal.registry.experiments import Experiments, load_experiments
from qcal.registry.gates import INPUTS_READ_KEY
from qcal_lab import __version__
from qcal_lab.calib import CALIBRATORS, build_calibrator
from qcal_lab.calib.thresholds import threshold_grid
from qcal_lab.calib.two_threshold import (
    SCOPES,
    SplitData,
    ThresholdedCalibration,
    train_calibration,
)
from qcal_lab.config import LabConfig, defaults_path, load_lab_config, require_hashed
from qcal_lab.data.coco import GroundTruth, load_coco
from qcal_lab.data.splits import draw, read_split, split_digest
from qcal_lab.evaluation import EvalLoop, check_metrics, load_eval_loop
from qcal_lab.models import build_detector, detector_fingerprint
from qcal_lab.predictions import (
    CacheEntry,
    PredictionCache,
    PredictionsError,
    PredictionsHeader,
    cache_key,
    read_predictions,
    write_predictions,
)

_log = get_logger("lab.experiment")

ROLES: Final = ("fit", "select", "evaluate")
HANDWRITTEN_DIR: Final = "handwritten/"  # Ian's files; never part of the source digest
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
    """Role -> split name: three different splits (CLAUDE.md rule 3).

    Calibrators fit on the fit split only, and both thresholds are selected on the select
    split only (arXiv:2405.20459 Alg. A.1 lines 2 and 6, with the paper's single val set split
    in two by the pre-registration). The evaluate split is used for nothing but metrics.
    """
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
    if roles["fit"] == roles["select"]:
        raise PlanError(
            f"{roles['fit']!r} would both fit calibrators and select thresholds; calibrators fit "
            "on calibrator_fit_split only and selection uses val only (CLAUDE.md rule 3)"
        )
    return roles


def check_disjoint(qcal_config: Config, ids: Mapping[str, Sequence[str]]) -> None:
    """Refuse unless every manifest passes ``qcal leakage`` and the roles share no image."""
    report = check_leakage(qcal_config)
    if not report.passed:
        raise PlanError(f"split manifests fail the leakage check:\n{report.render_text()}")
    for left, right in itertools.combinations(ROLES, 2):
        shared = sorted(set(ids[left]) & set(ids[right]))
        if shared:
            raise PlanError(
                f"the {left} and {right} splits share {len(shared)} image(s), "
                f"for example {', '.join(shared[:3])}"
            )


@dataclass
class _Context:
    request: RunRequest
    qcal_config: Config
    lab: LabConfig
    plan: RunPlan
    dataset: GroundTruth
    artifact_dir: Path
    images_dir: Path | None
    artifacts: list[dict[str, str]] = field(default_factory=list)
    cache_use: dict[str, dict[str, Any]] = field(default_factory=dict)

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


@functools.cache
def source_digest() -> str:
    """sha256 of this package's code and packaged settings, excluding Ian's files."""
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package).as_posix()
        if path.suffix not in {".py", ".toml"} or relative.startswith(HANDWRITTEN_DIR):
            continue
        digest.update(f"{relative}\0".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _images_digest(ctx: _Context, ids: Sequence[str]) -> str:
    """sha256 over the image files of a split; empty when the detector reads no files."""
    if ctx.images_dir is None:
        return ""
    digest = hashlib.sha256()
    for image_id in ids:
        path = ctx.images_dir / ctx.dataset.images[image_id].file_name
        content = sha256_file(path) if path.is_file() else "missing"
        digest.update(f"{image_id}\0{content}\n".encode())
    return digest.hexdigest()


def _cache(lab: LabConfig) -> PredictionCache:
    configured = lab.config.str_value("predictions.cache_dir")
    if not configured:
        return PredictionCache(None)
    path = Path(configured)
    return PredictionCache(path if path.is_absolute() else lab.config.root / path)


def _cached_images(
    entry: CacheEntry, header: PredictionsHeader, ids: Sequence[str]
) -> tuple[ImageDetections, ...] | None:
    """The cached predictions, if they are exactly what this run would compute."""
    try:
        found, images = read_predictions(entry.path)
    except PredictionsError as exc:
        _log.warning("ignoring cached predictions %s: %s", entry.path.name, exc)
        return None
    if found != header or [i.image_id for i in images] != list(ids):
        _log.warning("ignoring cached predictions %s: header or images differ", entry.path.name)
        return None
    return images


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
        images_dir=str(ctx.images_dir.resolve()) if ctx.images_dir else "",
        images_sha256=_images_digest(ctx, ids),
        source_sha256=source_digest(),
    )
    entry = cache.get(key)
    images = _cached_images(entry, header, ids) if entry is not None else None
    if entry is not None and images is not None:
        _log.info("using raw predictions for %s cached by %s", split, entry.produced_by)
        ctx.cache_use[split] = {"hit": True, "produced_by": entry.produced_by}
        ctx.add_artifact(entry.path, f"predictions_raw_{split}")
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
        cache.put(key, header, images, produced_by=ctx.request.run_id)
    ctx.cache_use[split] = {"hit": False, "produced_by": ctx.request.run_id}
    ctx.add_artifact(path, f"predictions_raw_{split}")
    return images


def _subset(images: Sequence[ImageDetections], ids: Sequence[str]) -> tuple[ImageDetections, ...]:
    wanted = set(ids)
    return tuple(i for i in images if i.image_id in wanted)


def _retained(images: Sequence[ImageDetections]) -> dict[str, Any]:
    per_class = collections.Counter(str(d.label) for i in images for d in i.detections)
    return {"total": sum(per_class.values()), "per_class": dict(sorted(per_class.items()))}


def _effective_digest(lab: LabConfig) -> str:
    canonical = json.dumps(lab.config.data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_experiment(
    request: RunRequest,
    *,
    qcal_config: Config | None = None,
    lab: LabConfig | None = None,
    eval_loop: EvalLoop | None = None,
) -> ExperimentResult:
    """Run one cell and seed. ``eval_loop`` injects a loop for tests (recorded as injected)."""
    qcal_config = qcal_config or load_config(request.root)
    refused = environment_problems(qcal_config)
    if refused:  # the registry strips these; a swapped executor might not
        raise ConfigError("configuration comes from the environment: " + "; ".join(refused))
    lab = lab or load_lab_config(qcal_config.root)
    require_hashed(lab, qcal_config)
    experiments = load_experiments(qcal_config)
    cell = experiments.cell(request.cell_id)
    if request.seed not in experiments.seeds_for(cell):
        raise PlanError(f"seed {request.seed} is not pre-registered for {cell.id}")
    plan = resolve_plan(cell.factors, lab)
    roles = split_roles(lab, qcal_config)
    if eval_loop is None:  # fail before any detector work if Ian's loop is missing
        loaded = load_eval_loop(lab, qcal_config)
        loop, loop_provenance = loaded.loop, loaded.provenance()
    else:
        loop, loop_provenance = eval_loop, {"module": f"injected:{type(eval_loop).__name__}"}
    dataset = load_coco(lab.file(f"datasets.{plan.shift}.annotations"))
    ids = {role: read_split(qcal_config, split) for role, split in roles.items()}
    check_disjoint(qcal_config, ids)
    subsets = {role: dataset.subset(role_ids) for role, role_ids in ids.items()}
    fit_ids = draw(ids["fit"], plan.fit_size, request.seed) if plan.fit_size else ids["fit"]
    images_setting = lab.config.str_value(f"datasets.{plan.shift}.images_dir")
    images_dir = lab.file(f"datasets.{plan.shift}.images_dir") if images_setting else None
    ctx = _Context(
        request,
        qcal_config,
        lab,
        plan,
        dataset,
        _artifact_dir(lab, qcal_config, request.run_id),
        images_dir,
    )
    detector = build_detector(
        lab, plan.detector, dataset, precision=plan.precision, images_dir=images_dir
    )
    cache = _cache(lab)
    raw = {role: _raw_predictions(ctx, detector, roles[role], ids[role], cache) for role in ROLES}
    fit = SplitData(roles["fit"], _subset(raw["fit"], fit_ids), subsets["fit"].subset(fit_ids))
    select = SplitData(roles["select"], raw["select"], subsets["select"])
    candidates = threshold_grid(
        lab.config.float_value("calibration.grid_start"),
        lab.config.float_value("calibration.grid_stop"),
        lab.config.float_value("calibration.grid_step"),
    )
    calibration = train_calibration(
        fit=fit,
        calibration_threshold_data=select,
        operating_threshold_data=select,
        build=lambda: build_calibrator(plan.calibrator, lab),
        scope=plan.scope,
        candidates=candidates,
        eval_loop=loop,
    )
    evaluated = calibration.apply(raw["evaluate"])
    metrics = check_metrics(loop.metrics(evaluated, subsets["evaluate"]))
    _write_outputs(ctx, roles["evaluate"], ids["evaluate"], calibration, evaluated)
    saved = calibration.to_dict()
    environment = {
        "qcal_lab_version": __version__,
        "qcal_lab_source_sha256": source_digest(),
        "lab_config_sha256": lab.sha256,
        "lab_config_effective_sha256": _effective_digest(lab),
        "eval_loop": loop_provenance,
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
            "seeded": plan.fit_size is not None,
            "size": len(fit_ids),
            "seed": request.seed,
            "sha256": split_digest(fit_ids),
        },
        "detector": _fingerprint(lab, plan),
        "detector_runtime": dict(getattr(detector, "runtime", {})),
        "prediction_cache": ctx.cache_use,
        "calibration": {
            key: saved[key]
            for key in ("scope", "calibration_thresholds", "operating_thresholds", "provenance")
        },
        "evaluate_retained_detections": _retained(evaluated),
        "program_python": sys.executable,
        "program_python_version": platform.python_version(),
        INPUTS_READ_KEY: inputs_read(qcal_config, lab, experiments),
    }
    return ExperimentResult(metrics, ctx.artifacts, environment)


def inputs_read(qcal_config: Config, lab: LabConfig, experiments: Experiments) -> dict[str, str]:
    """The sha256 of each configuration file this run read, by repository-relative path.

    The registry compares these with the digests it took at launch and fails the run when a
    file changed in between (``qcal.registry.gates.input_mismatches``).
    """
    root = qcal_config.root
    files: dict[Path, str | None] = {
        experiments.path: experiments.sha256,  # the bytes parsed, not a fresh read
        root / REPO_CONFIG_NAME: qcal_config.repo_sha256,
        defaults_path(): sha256_file(defaults_path()),
    }
    if lab.path is not None:
        files[lab.path] = lab.sha256
    reported: dict[str, str] = {}
    for path, digest in files.items():
        absolute = path if path.is_absolute() else root / path  # not resolved: symlinks are
        if digest and absolute.is_relative_to(root):  # compared under their own name
            reported[absolute.relative_to(root).as_posix()] = digest
    return reported


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
    "HANDWRITTEN_DIR",
    "ROLES",
    "ExperimentResult",
    "PlanError",
    "RunPlan",
    "RunRequest",
    "check_disjoint",
    "resolve_plan",
    "run_experiment",
    "source_digest",
    "split_roles",
    "write_result",
]
