"""The seam between the pipeline and Ian's hand-written evaluation loop.

The FP32 evaluation loop and the LaECE0 implementation are Ian's (plan §0 and CLAUDE.md rule
4): ``src/qcal_lab/handwritten/eval_loop.py`` by default (``eval_loop.module``). Its
``build(options)`` factory returns an object with the three methods of :class:`EvalLoop`.
This interface is a proposal; Ian may change it, and the pipeline follows.

The pipeline never computes a reported metric itself. It calls :meth:`EvalLoop.targets` to
get calibration targets on the fit split, :meth:`EvalLoop.threshold_objective` (LRP in the
paper) to select both thresholds on the select split (u_c on its raw scores, v_c on its
calibrated scores), and :meth:`EvalLoop.metrics` once, on the evaluate split. Version 2 of
this interface is proposed in ``docs/changes/evalloop-v2.md``.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from qcal.config import Config, ConfigError
from qcal.policy import Policy
from qcal.protocols import ImageDetections
from qcal.registry.executor import sha256_file
from qcal.registry.records import METRIC_NAME_PATTERN
from qcal_lab.config import LabConfig
from qcal_lab.data.coco import GroundTruth

# Kuzucu et al., arXiv:2405.20459, Alg. A.1: line 2 selects the calibration threshold (LRP with
# tau = 0); line 6 selects the operating threshold after calibration (LRP with tau).
STAGES: Final = ("calibration", "operating")
# The one module besides Ian's that may act as an evaluation loop. It refuses every dataset
# except an unmodified generated fixture and reports only smoke_* metrics.
FIXTURE_LOOP_MODULE: Final = "qcal_lab.fixture_eval"
# The policy category an evaluation loop's source file must belong to (CLAUDE.md rule 4).
IAN_CATEGORY: Final = "ian_only"


class HandwrittenMissingError(ConfigError):
    """Ian's evaluation loop does not exist yet; agents never write it."""


class EvaluationError(ValueError):
    """The evaluation loop returned something the pipeline cannot use."""


@runtime_checkable
class EvalLoop(Protocol):
    def targets(
        self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth
    ) -> Mapping[str, Sequence[float]]:
        """Per image, one calibration target in [0, 1] per detection, in detection order."""
        ...

    def threshold_objective(
        self,
        predictions: Sequence[ImageDetections],
        ground_truth: GroundTruth,
        *,
        label: int,
        stage: str,
    ) -> float:
        """Lower is better; ``predictions`` hold only class ``label`` at one threshold."""
        ...

    def metrics(
        self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth
    ) -> Mapping[str, float]:
        """The run's reported metrics on the evaluate split."""
        ...


@dataclass(frozen=True)
class LoadedEvalLoop:
    loop: EvalLoop
    module: str
    file: str  # repository-relative where possible
    sha256: str

    def provenance(self) -> dict[str, str]:
        return {"module": self.module, "file": self.file, "sha256": self.sha256}


def _check_source(module_name: str, module_file: str | None, qcal_config: Config) -> Path:
    """The loop's source must be one of Ian's files, or the fixture stand-in."""
    if module_file is None:
        raise ConfigError(f"{module_name} has no source file to verify")
    path = Path(module_file).resolve()
    if module_name == FIXTURE_LOOP_MODULE:
        return path
    root = qcal_config.root
    relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else None
    policy = Policy.from_config(qcal_config)
    if relative is None or IAN_CATEGORY not in policy.categories_for(relative):
        shown = relative or str(path)
        raise ConfigError(
            f"evaluation loop {module_name} ({shown}) is not one of Ian's hand-written files; "
            f"reported metrics come only from the {IAN_CATEGORY} category (CLAUDE.md rule 4)"
        )
    return path


def load_eval_loop(lab: LabConfig, qcal_config: Config) -> LoadedEvalLoop:
    """Import ``eval_loop.module``, verify where it lives, and build it with its options."""
    module_name = lab.text("eval_loop.module")
    factory_name = lab.text("eval_loop.factory")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and (module_name == exc.name or module_name.startswith(f"{exc.name}.")):
            raise HandwrittenMissingError(
                f"{module_name} does not exist. The evaluation loop is Ian's to write by hand "
                "(CLAUDE.md rule 4); agents never create it."
            ) from exc
        raise
    source = _check_source(module_name, getattr(module, "__file__", None), qcal_config)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ConfigError(f"{module_name} has no callable {factory_name!r}")
    loop = factory(dict(lab.table("eval_loop.options")))
    if not isinstance(loop, EvalLoop):
        raise ConfigError(
            f"{module_name}.{factory_name}() must return an object with targets, "
            "threshold_objective and metrics"
        )
    root = qcal_config.root
    shown = source.relative_to(root).as_posix() if source.is_relative_to(root) else str(source)
    return LoadedEvalLoop(loop, f"{module_name}:{factory_name}", shown, sha256_file(source))


def check_targets(
    predictions: Sequence[ImageDetections], targets: Mapping[str, Sequence[float]]
) -> dict[str, tuple[float, ...]]:
    """One finite target in [0, 1] per detection, for every image."""
    checked: dict[str, tuple[float, ...]] = {}
    for image in predictions:
        values = targets.get(image.image_id)
        if values is None or len(values) != len(image.detections):
            raise EvaluationError(
                f"image {image.image_id}: expected {len(image.detections)} targets"
            )
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise EvaluationError(f"image {image.image_id}: target {value!r} is not a number")
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise EvaluationError(f"image {image.image_id}: target {value!r} not in [0, 1]")
        checked[image.image_id] = tuple(float(v) for v in values)
    return checked


def check_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Finite numbers under names a claim reference can use (the registry's rule)."""
    checked: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not METRIC_NAME_PATTERN.fullmatch(name):
            raise EvaluationError(f"metric name {name!r} is not usable in a claim reference")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise EvaluationError(f"metric {name} is not a number")
        if not math.isfinite(value):
            raise EvaluationError(f"metric {name} is not finite")
        checked[name] = float(value)
    if not checked:
        raise EvaluationError("the evaluation loop returned no metrics")
    return checked


__all__ = [
    "FIXTURE_LOOP_MODULE",
    "IAN_CATEGORY",
    "STAGES",
    "EvalLoop",
    "EvaluationError",
    "HandwrittenMissingError",
    "LoadedEvalLoop",
    "check_metrics",
    "check_targets",
    "load_eval_loop",
]
