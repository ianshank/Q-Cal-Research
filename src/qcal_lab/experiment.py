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
import dataclasses
import fcntl
import functools
import hashlib
import itertools
import json
import os
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import qcal
from qcal.config import REPO_CONFIG_NAME, Config, ConfigError, environment_problems, load_config
from qcal.integrity.leakage import check_leakage
from qcal.log import get_logger
from qcal.protocols import Detector, ImageDetections
from qcal.registry.executor import STATUS_FAILED, STATUS_OK, result_envelope, sha256_file
from qcal.registry.experiments import Experiments, load_experiments
from qcal.registry.gates import INPUTS_READ_KEY, code_location_problems
from qcal.registry.records import SEED_EFFECTIVE_KEY, RunRecord
from qcal_lab import __version__
from qcal_lab.calib import CALIBRATORS, build_calibrator
from qcal_lab.calib.thresholds import threshold_grid
from qcal_lab.calib.two_threshold import (
    SCOPES,
    SplitData,
    ThresholdedCalibration,
    train_calibration,
    write_calibration,
)
from qcal_lab.config import LabConfig, defaults_path, load_lab_config, require_hashed
from qcal_lab.data.coco import GroundTruth, load_coco
from qcal_lab.data.splits import draw, read_split, split_digest
from qcal_lab.declarations import DeclarationsError, declarations_path, detector_declarations
from qcal_lab.evaluation import EvalLoop, check_metrics, load_eval_loop
from qcal_lab.formats import envelope, json_bytes, write_bytes_atomic
from qcal_lab.models import (
    CHECKPOINT_SETTING,
    build_detector,
    detector_fingerprint,
    detector_key_parts,
    detector_problems,
    detector_targets,
    detector_test_cfg,
    score_definition,
)
from qcal_lab.numerics import Regime, load_regime, recorded_environment
from qcal_lab.predictions import (
    UNREADABLE,
    CacheConflictError,
    CacheEntry,
    PredictionCache,
    PredictionsError,
    PredictionsHeader,
    PredictionSource,
    cache_key,
    loads_predictions,
    read_predictions,
    write_predictions,
)
from qcal_lab.sanity import SanityError, configured, needs_declaration, require_declared

_log = get_logger("lab.experiment")

ROLES: Final = ("fit", "select", "evaluate")
HANDWRITTEN_DIR: Final = "handwritten/"  # Ian's files; never part of the source digest
#: The code that decides a set of raw predictions, relative to this package. Only its source
#: enters the prediction cache key, so editing a calibrator or this module reuses predictions.
PREDICTION_MODULES: Final = (
    "models",
    "data/coco.py",
    "data/fixture.py",
    "predictions.py",
    "numerics.py",
    "formats.py",  # predictions.py writes through it: its bytes decide the cached file's
)
#: Modules the prediction modules import that need not be hashed, each with the reason
#: (tests/unit/test_lab_prediction_modules.py keeps this list complete and current).
PREDICTION_MODULE_EXCLUSIONS: Final[Mapping[str, str]] = {
    "config.py": "settings enter the key by value (fingerprint, source, kind parts)",
    "__init__.py": "the package version only; the key holds the source instead",
    "data/__init__.py": "re-exports only",
}
#: Where qcal.registry.store keeps a run's record, under paths.registry_dir. Science code may
#: not import the store (it writes records), so the producer check reads this one file;
#: tests/unit/test_lab_experiment.py pins it to the store's own layout.
RECORD_FILE: Final = "{run_id}.json"
#: A running launcher's in-flight marker under paths.inflight_dir, locked while it runs; the
#: same test pins it to qcal.registry.store's layout.
MARKER_FILE: Final = "{run_id}.inflight.json"
#: Puts tried under one key: the second follows a quarantine of unvouched bytes.
_PUT_ATTEMPTS: Final = 2
#: The pessimistic envelope's failure kind and error: what a run killed outright reports.
UNFINISHED_KIND: Final = "unknown"
UNFINISHED_ERROR: Final = "the program ended before writing its result (killed or crashed)"
_FACTOR_ROLES: Final = (
    "detector",
    "calibrator",
    "calibrator_scope",
    "calibrator_fit_split_size",
    "shift",
    "precision",
    "target",
    "quant_path",
    "fit_precision",
    "threshold_regime",
    "split_design",
)
#: Where a role's detections come from: the detector itself (Phase 2 adds a replay file).
LIVE_SOURCE: Final = "live"
#: Artifact file names and kinds, inside the run's artifact directory.
RAW_PREDICTIONS_FILE: Final = "raw_{split}.jsonl"
CALIBRATION_FILE: Final = "calibration.json"
CALIBRATED_PREDICTIONS_FILE: Final = "calibrated_{split}.jsonl"
RAW_PREDICTIONS_KIND: Final = "predictions_raw_{split}"
CALIBRATION_KIND: Final = "calibration"
CALIBRATED_PREDICTIONS_KIND: Final = "predictions_calibrated_{split}"
PACKAGES_FILE: Final = "packages.json"
PACKAGES_KIND: Final = "packages"
PACKAGES_FORMAT: Final = "qcal_lab.packages"
PACKAGES_VERSION: Final = 1


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
class RoleSpec:
    """What one role (fit, select, evaluate) reads: Phase 1 gives every role the same."""

    dataset: str
    detector: str
    precision: str
    target: str
    source: str


@dataclass(frozen=True)
class RunPlan:
    detector: str
    calibrator: str
    scope: str
    fit_size: int | None
    shift: str
    precision: str
    target: str = "torch_fp32"
    quant_path: str = "none"
    fit_precision: str = "fp32"
    threshold_regime: str = "reuse_fp32"
    split_design: str = "disjoint"

    def roles(self) -> dict[str, RoleSpec]:
        """Each role's source of detections. Phase 2 lets them differ (for example a fit on
        FP32 detections and an evaluation on INT8 ones); Phase 1 reads one source for all."""
        spec = RoleSpec(self.shift, self.detector, self.precision, self.target, LIVE_SOURCE)
        return dict.fromkeys(ROLES, spec)


@dataclass
class ExperimentResult:
    metrics: dict[str, float]
    artifacts: list[dict[str, str]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    #: Figures the registry records under ``resources`` (``CHILD_RESOURCE_KEYS``).
    resources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The executor's result envelope for a successful run."""
        return result_envelope(
            STATUS_OK,
            metrics=self.metrics,
            artifacts=self.artifacts,
            environment=self.environment,
            resources=self.resources,
        )


def _factor_names(lab: LabConfig) -> dict[str, str]:
    return {role: lab.text(f"factors.{role}") for role in _FACTOR_ROLES}


def resolve_plan(factors: Mapping[str, Any], lab: LabConfig) -> RunPlan:
    """Map a cell's factors to a plan, refusing anything the program would not honour."""
    names = _factor_names(lab)
    defaults = lab.table("factors.defaults")
    supported = lab.table("factors.supported")
    not_roles = sorted((set(defaults) | set(supported)) - set(_FACTOR_ROLES))
    if not_roles:
        raise ConfigError(
            f"factors.defaults/supported name {not_roles}, which are not roles this program "
            f"honours ({', '.join(_FACTOR_ROLES)}); a validated but ignored factor would "
            "label runs it never ran"
        )
    passthrough = set(lab.config.str_list("factors.passthrough"))
    unknown = sorted(k for k in factors if k not in names.values() and k not in passthrough)
    if unknown:
        raise PlanError(f"cell factors this program does not understand: {', '.join(unknown)}")
    effective = {role: factors.get(name, defaults.get(role)) for role, name in names.items()}
    for role, allowed in supported.items():  # keyed by role, whatever the factor is named
        if not isinstance(allowed, list):
            raise ConfigError(f"factors.supported.{role} must be a list")
        current = effective[role]
        if current is not None and current not in allowed:
            raise PlanError(
                f"factor {names[role]}={current!r} is not supported (allowed: {allowed})"
            )

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
    labels = {r: value(r) for r in ("target", "quant_path", "fit_precision", "threshold_regime")}
    labels["split_design"] = value("split_design")
    for role, label in {"precision": precision, **labels}.items():
        if not isinstance(label, str) or not label:
            raise PlanError(f"{names[role]} must be a non-empty string, got {label!r}")
    return RunPlan(detector, calibrator, str(scope), size, shift, str(precision), **labels)


def seed_effective(plan: RunPlan, fit_ids: Sequence[str], fit_split: Sequence[str]) -> bool:
    """Whether this run's seed changed anything: it drew a proper subset of the fit split,
    and the calibrator reads the data it is fit on (the identity does not)."""
    drew = plan.fit_size is not None and len(fit_ids) < len(fit_split)
    return drew and CALIBRATORS.get(plan.calibrator).uses_fit_data


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
    fingerprint: Mapping[str, Any]
    source: PredictionSource
    regime: Regime
    key_parts: Mapping[str, Any]  # the detector kind's, computed once per run
    build: Callable[[], Detector]
    artifacts: list[dict[str, str]] = field(default_factory=list)
    cache_use: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_digests: dict[str, str] = field(default_factory=dict)  # split -> sha256 of its raw file
    built: Detector | None = None
    #: Written as a failed envelope after every artifact, so a run killed outright (SIGKILL,
    #: power) still leaves what it produced; ``write_result`` replaces it at the end.
    partial_environment: dict[str, Any] = field(default_factory=dict)

    def detector(self) -> Detector:
        """The detector, built on first use: a run whose predictions are all cached builds none."""
        if self.built is None:
            self.built = self.build()
        return self.built

    def relative(self, path: Path) -> str:
        root = self.qcal_config.root
        return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)

    def add_artifact(self, path: Path, kind: str, sha256: str) -> None:
        """List an artifact with the sha256 of the bytes this program wrote or read; the
        launcher fails the run if the file holds other bytes when it records it."""
        self.artifacts.append({"path": self.relative(path), "kind": kind, "sha256": sha256})
        self.checkpoint()

    def checkpoint(self) -> None:
        """The pessimistic envelope: failed, with everything produced so far."""
        write_failure(
            self.request.result_path,
            failure_kind=UNFINISHED_KIND,
            error=UNFINISHED_ERROR,
            artifacts=self.artifacts,
            environment=self.partial_environment,
        )

    def header(self, split: str, ids: Sequence[str], stage: str) -> PredictionsHeader:
        return PredictionsHeader(
            detector=self.plan.detector,
            split=split,
            stage=stage,
            dataset_sha256=self.dataset.sha256,
            split_sha256=split_digest(ids),
            category_ids=self.dataset.category_ids,
            source=self.source,
        )


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


def prediction_source(
    lab: LabConfig, plan: RunPlan, fingerprint: Mapping[str, Any]
) -> PredictionSource:
    """What produces this run's predictions, from configuration alone (no detector built)."""
    return PredictionSource(
        precision=plan.precision,
        target=plan.target,
        quant_path=plan.quant_path,
        shift=plan.shift,
        score_definition=score_definition(lab, plan.detector),
        model_sha256=str(fingerprint.get("files", {}).get(CHECKPOINT_SETTING, "")),
        test_cfg=detector_test_cfg(lab, plan.detector),
    )


@functools.cache
def prediction_code_digest() -> str:
    """sha256 of the modules in :data:`PREDICTION_MODULES` (Python sources only)."""
    package = Path(__file__).resolve().parent
    files: set[Path] = set()
    for entry in PREDICTION_MODULES:
        path = package / entry
        files.update(path.rglob("*.py") if path.is_dir() else [path])
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(f"{path.relative_to(package).as_posix()}\0".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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


def handwritten_digest() -> str:
    """sha256 of Ian's hand-written files (read, never written: CLAUDE.md rule 4); "" if none."""
    tree = Path(__file__).resolve().parent / HANDWRITTEN_DIR.rstrip("/")
    files = sorted(p for p in tree.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if not files:
        return ""
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.relative_to(tree).as_posix()}\0".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def program_code(qcal_config: Config) -> dict[str, str]:
    """Where this program's code was imported from.

    With ``registry.require_code_in_root`` both packages must come from ``paths.source_dir``,
    as the launcher requires of qcal: the record's git SHA names the repository's code, and an
    installed copy elsewhere could differ from it.
    """
    root = qcal_config.root
    packages = {
        "qcal": Path(qcal.__file__).resolve().parent,
        "qcal_lab": Path(__file__).resolve().parent,
    }
    problems = code_location_problems(qcal_config)
    source = qcal_config.path("source_dir").resolve()
    if qcal_config.bool_value("registry.require_code_in_root") and not packages[
        "qcal_lab"
    ].is_relative_to(source):
        problems.append(f"qcal_lab is imported from {packages['qcal_lab']}, not from {source}")
    if problems:
        raise ConfigError("; ".join(problems))
    return {
        name: path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        for name, path in packages.items()
    }


def installed_packages() -> dict[str, str]:
    """Every installed distribution and its version, by lower-case name."""
    found = {
        str(dist.metadata["Name"]).lower(): dist.version
        for dist in metadata.distributions()
        if dist.metadata["Name"]
    }
    return dict(sorted(found.items()))


def _write_packages(ctx: _Context) -> None:
    body = {"python": platform.python_version(), "packages": installed_packages()}
    path = ctx.artifact_dir / PACKAGES_FILE
    data = json_bytes(envelope(PACKAGES_FORMAT, PACKAGES_VERSION, body))
    ctx.add_artifact(path, PACKAGES_KIND, write_bytes_atomic(path, data))


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
    entry: CacheEntry, header: PredictionsHeader, ids: Sequence[str], data: bytes | None = None
) -> tuple[ImageDetections, ...] | None:
    """The cached predictions, if they are exactly what this run would compute.

    ``data`` are the entry's bytes as already read and verified; they are parsed, not reread.
    """
    try:
        found, images = (
            read_predictions(entry.path) if data is None else loads_predictions(data, entry.path)
        )
    except PredictionsError as exc:
        _log.warning("ignoring cached predictions %s: %s", entry.path.name, exc)
        return None
    if found != header or [i.image_id for i in images] != list(ids):
        _log.warning("ignoring cached predictions %s: header or images differ", entry.path.name)
        return None
    return images


def _vouched(ctx: _Context, path: Path, sha256: str) -> bool:
    """Whether any run record, of any status, lists ``path`` with these bytes.

    The cache is gitignored and writable outside the registry, so a sidecar alone proves
    nothing; a registered run's record does. Any record counts, not only the sidecar's
    producer: identical bytes re-put by a later run rename the producer, and a failed or
    interrupted run's record lists the bytes it wrote as well as an ok one does.
    """
    listed = ctx.relative(path)
    registry = ctx.qcal_config.path("registry_dir")
    for record_path in sorted(registry.glob(RECORD_FILE.format(run_id="*"))):
        try:
            record = RunRecord.from_dict(json.loads(record_path.read_text("utf-8")))
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError):
            continue
        if any(a.path == listed and a.sha256 == sha256 for a in record.artifacts):
            return True
    return False


def _producer_running(ctx: _Context, run_id: str) -> bool:
    """Whether ``run_id``'s launcher is alive: it holds the lock on its in-flight marker
    until its record is written (qcal.registry.runner), so a held lock means running."""
    if not run_id:
        return False
    marker = ctx.qcal_config.path("inflight_dir") / MARKER_FILE.format(run_id=run_id)
    try:
        handle = marker.open("rb")
    except OSError:
        return False
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


def _quarantine_unvouched(
    ctx: _Context, cache: PredictionCache, key: str, conflict: CacheConflictError
) -> dict[str, Any] | None:
    """Move other bytes under ``key`` aside when nothing vouches for them; what moved.

    Bytes a record lists are evidence and never move. Neither do bytes any of whose
    producers is still running: its record does not exist yet, and moving its file would
    fail that run when the launcher re-hashes its artifacts. Unreadable bytes cannot be
    checked: kept. The producers are checked before the records: a launcher releases its
    marker only after writing its record, so a producer that finished in between is then
    found by :func:`_vouched` (the other order could miss both).
    """
    existing = conflict.existing_sha256
    path = cache.path(key)
    if path is None or existing == UNREADABLE:
        return None
    producers = cache.producers(key)
    running = [run for run in producers if _producer_running(ctx, run)]
    if running:
        _log.info("cached predictions %s belong to running %s; kept", path.name, running)
        return None
    if _vouched(ctx, path, existing):
        return None
    moved = cache.quarantine(key, existing)
    if moved is None:
        return None
    return {
        "sha256": existing,
        "produced_by": producers[-1] if producers else "",
        "producers": producers,
        "moved_to": ctx.relative(moved),
    }


def prediction_key(
    ctx: _Context, split_sha256: str, ids: Sequence[str], header: PredictionsHeader
) -> str:
    """The cache key: everything that decides one split's raw predictions, and nothing else.

    Not where the images live (their bytes count, not their path), nor this package's
    version: the key holds the source of the modules that produce predictions. The header
    the file will carry is part of the key, so code that builds it outside the prediction
    modules cannot change the cached bytes without changing the key.
    """
    return cache_key(
        header=header.to_dict(),
        detector=ctx.fingerprint,
        source=ctx.source.to_dict(),
        numerics={
            "regime": ctx.regime.to_dict(),
            "environment": recorded_environment(os.environ),  # NVIDIA_TF32_OVERRIDE and kin
        },
        kind=ctx.key_parts,
        dataset_sha256=ctx.dataset.sha256,
        split_sha256=split_sha256,
        images_sha256=_images_digest(ctx, ids),
        code_sha256=prediction_code_digest(),
    )


def _raw_predictions(
    ctx: _Context, split: str, ids: Sequence[str], cache: PredictionCache
) -> tuple[ImageDetections, ...]:
    header = ctx.header(split, ids, "raw")
    key = prediction_key(ctx, header.split_sha256, ids, header)
    data = b""
    with cache.lock(key):
        entry = cache.get(key)
        if entry is not None:
            # One read: these bytes are hashed, vouched for by a record and parsed, so a
            # file swapped after the sidecar check is never used.
            try:
                data = entry.path.read_bytes()
            except OSError:
                entry = None
    if entry is not None:
        entry = dataclasses.replace(entry, sha256=hashlib.sha256(data).hexdigest())
        if not _vouched(ctx, entry.path, entry.sha256):
            _log.warning(
                "ignoring cached predictions %s: no record lists them (cached by %s)",
                entry.path.name,
                entry.produced_by,
            )
            entry = None
    images = _cached_images(entry, header, ids, data) if entry is not None else None
    if entry is not None and images is not None:
        _log.info("using raw predictions for %s cached by %s", split, entry.produced_by)
        ctx.cache_use[split] = {"hit": True, "produced_by": entry.produced_by}
        ctx.raw_digests[split] = entry.sha256
        ctx.add_artifact(entry.path, RAW_PREDICTIONS_KIND.format(split=split), entry.sha256)
        return images
    _log.info("running %s on %d %s images", ctx.plan.detector, len(ids), split)
    images = tuple(ctx.detector().predict(list(ids)))
    if [i.image_id for i in images] != list(ids):
        raise PlanError(f"{ctx.plan.detector} returned images out of order or incomplete")
    use: dict[str, Any] = {"hit": False, "produced_by": ctx.request.run_id}
    path, digest = cache.path(key), ""
    if path is not None:
        with cache.lock(key):
            written, use_update = _put(ctx, cache, key, header, images)
        use.update(use_update)
        path, digest = (written.path, written.sha256) if written is not None else (None, "")
    if path is None:
        path = ctx.artifact_dir / RAW_PREDICTIONS_FILE.format(split=split)
        digest = write_predictions(path, header, images)
    ctx.cache_use[split] = use
    ctx.raw_digests[split] = digest
    ctx.add_artifact(path, RAW_PREDICTIONS_KIND.format(split=split), digest)
    return images


def _put(
    ctx: _Context,
    cache: PredictionCache,
    key: str,
    header: PredictionsHeader,
    images: Sequence[ImageDetections],
) -> tuple[CacheEntry | None, dict[str, Any]]:
    """Cache this run's predictions; on other bytes under the key, quarantine them once if
    nothing vouches for them, else keep them and record the conflict."""
    use: dict[str, Any] = {}
    for _attempt in range(_PUT_ATTEMPTS):
        try:
            return cache.put(key, header, images, produced_by=ctx.request.run_id), use
        except CacheConflictError as exc:
            if "quarantined" not in use:
                moved = _quarantine_unvouched(ctx, cache, key, exc)
                if moved is not None:
                    use["quarantined"] = moved
                    continue
            _log.warning("%s; this run keeps its own predictions next to it", exc)
            use["conflict"] = {"key": key, "cached_sha256": exc.existing_sha256}
            return None, use
    return None, use


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
    progress: list[dict[str, str]] | None = None,
    partial_environment: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Run one cell and seed. ``eval_loop`` injects a loop for tests (recorded as injected).

    Artifacts are appended to ``progress`` as they are written, and what the run knows of its
    environment is put in ``partial_environment``, so a run that fails part way can still
    report what it produced and what it read.
    """
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
    code = program_code(qcal_config)
    regime = load_regime(lab, plan.precision)  # refuses a precision without a regime
    declarations_read = _check_detector(qcal_config, lab, plan)
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
    fingerprint = _fingerprint(lab, plan)  # hashes the checkpoint: once per run
    ctx = _Context(
        request,
        qcal_config,
        lab,
        plan,
        dataset,
        _artifact_dir(lab, qcal_config, request.run_id),
        images_dir,
        fingerprint,
        prediction_source(lab, plan, fingerprint),
        regime,
        detector_key_parts(lab, plan.detector),
        functools.partial(
            build_detector,
            lab,
            plan.detector,
            dataset,
            precision=plan.precision,
            images_dir=images_dir,
            declared=functools.partial(detector_declarations, qcal_config, plan.detector),
        ),
        artifacts=progress if progress is not None else [],
        partial_environment=_partial_environment(
            partial_environment, qcal_config, lab, experiments, code, extra=declarations_read
        ),
    )
    ctx.checkpoint()
    cache = _cache(lab)
    raw = {role: _raw_predictions(ctx, roles[role], ids[role], cache) for role in ROLES}
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
    _write_packages(ctx)
    saved = calibration.to_dict()
    environment = {
        "qcal_lab_version": __version__,
        "qcal_lab_source_sha256": source_digest(),
        "handwritten_sha256": handwritten_digest(),
        "program_code": code,
        "numerics": {"regime": regime.to_dict(), "environment": recorded_environment(os.environ)},
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
        SEED_EFFECTIVE_KEY: seed_effective(plan, fit_ids, ids["fit"]),  # tables refuse a
        # spread over seeds that change nothing
        "fit_draw": {
            "seeded": plan.fit_size is not None,
            "size": len(fit_ids),
            "seed": request.seed,
            "sha256": split_digest(fit_ids),
        },
        "detector": fingerprint,
        "predictions_source": ctx.source.to_dict(),
        "detector_built": ctx.built is not None,  # no: every split came from the cache
        "detector_runtime": dict(getattr(ctx.built, "runtime", {})),
        "roles": {role: dataclasses.asdict(spec) for role, spec in plan.roles().items()},
        "prediction_cache": ctx.cache_use,
        # Per split: equal for every seed of one cell, so tables can require it to agree
        # (registry.identity_columns); different raw predictions are a different experiment.
        "raw_predictions_sha256": dict(sorted(ctx.raw_digests.items())),
        "calibration": {
            key: saved[key]
            for key in ("scope", "calibration_thresholds", "operating_thresholds", "provenance")
        },
        "evaluate_retained_detections": _retained(evaluated),
        "program_python": sys.executable,
        "program_python_version": platform.python_version(),
        INPUTS_READ_KEY: inputs_read(qcal_config, lab, experiments, declarations_read),
    }
    hits = sum(bool(use.get("hit")) for use in ctx.cache_use.values())
    resources = {"cache_hits": hits, "cache_misses": len(ctx.cache_use) - hits}
    return ExperimentResult(metrics, ctx.artifacts, environment, resources)


def _check_detector(qcal_config: Config, lab: LabConfig, plan: RunPlan) -> dict[Path, str]:
    """Refuse a detector that is not fully configured, lacks the target, or is undeclared."""
    missing = detector_problems(lab, plan.detector)
    if missing:
        raise ConfigError(f"{'; '.join(missing)}; set it in configs/lab.toml")
    produced = detector_targets(lab, plan.detector)
    if plan.target not in produced:
        raise PlanError(
            f"detector {plan.detector!r} does not produce target {plan.target!r} "
            f"(it produces: {sorted(produced)}); refusing rather than relabelling its predictions"
        )
    return _require_sanity_declaration(qcal_config, lab, plan.detector)


def _require_sanity_declaration(
    qcal_config: Config, lab: LabConfig, detector: str
) -> dict[Path, str]:
    """Refuse a detector whose sanity declaration (D2) is incomplete or about something else.

    Runs before any detector work, so the expectation is fixed before a result exists.
    Returns the declarations file and its digest, for ``inputs_read``.
    """
    if not needs_declaration(lab, detector):
        return {}
    try:
        path = declarations_path(qcal_config)
        require_declared(
            detector, detector_declarations(qcal_config, detector), configured(lab, detector)
        )
    except (DeclarationsError, SanityError) as exc:
        raise PlanError(str(exc)) from exc
    return {path: sha256_file(path)} if path.is_file() else {}


def _partial_environment(
    into: dict[str, Any] | None,
    qcal_config: Config,
    lab: LabConfig,
    experiments: Experiments,
    code: Mapping[str, Any],
    *,
    extra: Mapping[Path, str] | None = None,
) -> dict[str, Any]:
    """What a failed run can still report: its code and the configuration it read."""
    known = {
        "program_python": sys.executable,
        "qcal_lab_source_sha256": source_digest(),
        "handwritten_sha256": handwritten_digest(),
        "program_code": dict(code),
        INPUTS_READ_KEY: inputs_read(qcal_config, lab, experiments, extra),
    }
    if into is None:
        return known
    into.update(known)
    return into


def inputs_read(
    qcal_config: Config,
    lab: LabConfig,
    experiments: Experiments,
    extra: Mapping[Path, str] | None = None,
) -> dict[str, str]:
    """The sha256 of each configuration file this run read, by repository-relative path.

    The registry compares these with the digests it took at launch and fails the run when a
    file changed in between (``qcal.registry.gates.input_mismatches``). ``extra`` adds other
    files the run read, such as Ian's declarations, so the record names the bytes it used.
    """
    root = qcal_config.root
    files: dict[Path, str | None] = {
        experiments.path: experiments.sha256,  # the bytes parsed, not a fresh read
        root / REPO_CONFIG_NAME: qcal_config.repo_sha256,
        defaults_path(): sha256_file(defaults_path()),
    }
    if lab.path is not None:
        files[lab.path] = lab.sha256
    files.update(extra or {})
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
    calibration_path = ctx.artifact_dir / CALIBRATION_FILE
    digest = write_calibration(calibration_path, calibration)
    ctx.add_artifact(calibration_path, CALIBRATION_KIND, digest)
    path = ctx.artifact_dir / CALIBRATED_PREDICTIONS_FILE.format(split=split)
    digest = write_predictions(path, ctx.header(split, ids, "calibrated"), evaluated)
    ctx.add_artifact(path, CALIBRATED_PREDICTIONS_KIND.format(split=split), digest)


def _write_envelope(path: Path, envelope: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_result(path: Path, result: ExperimentResult) -> None:
    """Write the executor contract's result envelope for a successful run, atomically."""
    _write_envelope(path, result.to_dict())


def write_failure(
    path: Path,
    *,
    failure_kind: str,
    error: str,
    artifacts: Sequence[Mapping[str, str]] = (),
    environment: Mapping[str, Any] | None = None,
) -> None:
    """Write the result envelope of a failed run: no metrics, the artifacts written so far,
    and what the run knew of its environment (at least the interpreter)."""
    envelope = result_envelope(
        STATUS_FAILED,
        failure_kind=failure_kind,
        error=error,
        artifacts=artifacts,
        environment={"program_python": sys.executable, **(environment or {})},
    )
    _write_envelope(path, envelope)


__all__ = [
    "CALIBRATED_PREDICTIONS_FILE",
    "CALIBRATED_PREDICTIONS_KIND",
    "CALIBRATION_FILE",
    "CALIBRATION_KIND",
    "HANDWRITTEN_DIR",
    "LIVE_SOURCE",
    "MARKER_FILE",
    "PACKAGES_FILE",
    "PACKAGES_FORMAT",
    "PACKAGES_KIND",
    "PACKAGES_VERSION",
    "PREDICTION_MODULES",
    "RAW_PREDICTIONS_FILE",
    "RAW_PREDICTIONS_KIND",
    "RECORD_FILE",
    "ROLES",
    "ExperimentResult",
    "PlanError",
    "RoleSpec",
    "RunPlan",
    "RunRequest",
    "check_disjoint",
    "handwritten_digest",
    "installed_packages",
    "prediction_code_digest",
    "prediction_key",
    "prediction_source",
    "program_code",
    "resolve_plan",
    "run_experiment",
    "seed_effective",
    "source_digest",
    "split_roles",
    "write_failure",
    "write_result",
]
