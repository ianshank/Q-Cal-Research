"""Synthetic protocol cases: small inputs on which evaluation protocols that differ in one
choice give different answers (cycle plan PR-F, D3/F5; docs/changes/qcal-foundations.md).

Each case is a COCO annotation document and a COCO results list, plus the alternatives it
separates. Each alternative predicts an *outcome*, never a metric value: what each detection
became (``TP:<annotation id>``, ``FP``, ``ignored``, ``dropped``), which bin a score falls in,
which classes a per-class mean averages, which threshold a selection keeps. Ian runs the
oracle on the cases and commits which alternative it took (``tests/parity/cases/README.md``);
nothing here decides that for him.

``python -m qcal_lab protocol-cases`` writes the cases; ``--check`` reports drift between the
committed files and this generator, so a case cannot change silently after the oracle ran.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

from qcal.log import get_logger
from qcal_lab.calib.thresholds import threshold_grid
from qcal_lab.config import LabConfig, load_tooling
from qcal_lab.formats import write_bytes_atomic

_log = get_logger("lab.protocol_cases")

FORMAT_VERSION: Final = 1
KIND: Final = "protocol"
SUFFIX: Final = ".json"
#: The only keys an alternative may predict: outcomes, never metric values.
PREDICATES: Final = frozenset({"outcomes", "bins", "classes_in_mean", "threshold"})
#: Every case's object side (pixels) is a multiple of this, so IoUs are exact fractions.
_UNIT: Final = 10
#: The IoU of crowd_only_overlap's inside box with its crowd region (a unit box in 5 x 5).
_CROWD_INSIDE_IOU: Final = Fraction(1, 25)
#: The object two equal-score detections compete for: large enough that the worse overlap
#: stays below 1 for any threshold the settings allow.
_COMPETE_SIDE: Final = 10 * _UNIT


@dataclass(frozen=True)
class CaseSettings:
    """``[protocol_cases]`` in ``tooling.toml``; the grid comes from the science settings."""

    directory: str
    image_width: int
    image_height: int
    iou_threshold: float
    max_detections: int
    bin_count: int
    grid: tuple[float, ...]

    @classmethod
    def load(cls, lab: LabConfig, overrides: Mapping[str, Any] | None = None) -> CaseSettings:
        raw = {**load_tooling()["protocol_cases"], **(overrides or {})}
        settings = cls(
            directory=str(raw["directory"]),
            image_width=int(raw["image_width"]),
            image_height=int(raw["image_height"]),
            iou_threshold=float(raw["iou_threshold"]),
            max_detections=int(raw["max_detections"]),
            bin_count=int(raw["bin_count"]),
            grid=threshold_grid(
                lab.config.float_value("calibration.grid_start"),
                lab.config.float_value("calibration.grid_stop"),
                lab.config.float_value("calibration.grid_step"),
            ),
        )
        problems = settings.problems()
        if problems:
            raise ValueError("protocol_cases: " + "; ".join(problems))
        return settings

    def problems(self) -> list[str]:
        problems: list[str] = []
        if not 0.0 < self.iou_threshold < 1.0:
            problems.append("iou_threshold must be in (0, 1)")
        if self.max_detections < 1 or self.bin_count < 2:
            problems.append("max_detections >= 1 and bin_count >= 2 are required")
        if self.grid[-1] >= 1.0:
            problems.append("the threshold grid must stop below 1 (no score lies above it)")
        elif _worse_height(self.iou_threshold) >= _COMPETE_SIDE:
            problems.append("iou_threshold is too close to 1 for equal_score_order")
        elif _exact(self.iou_threshold) <= _CROWD_INSIDE_IOU:
            problems.append("iou_threshold is too small for crowd_only_overlap")
        side = _UNIT * _exact(self.iou_threshold).denominator if not problems else 0
        cells = math.ceil(math.sqrt(self.max_detections + 1))
        needed = max(side + _UNIT, 2 * _COMPETE_SIDE + _UNIT)  # equal_score_order: two objects
        if min(self.image_width, self.image_height) < max(needed, 2 * cells):
            problems.append("the image is too small for the cases")
        return problems


@dataclass
class Case:
    name: str
    separates: str
    categories: list[int] = field(default_factory=lambda: [1])
    annotations: list[dict[str, Any]] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    def gt(self, bbox: list[int], *, category: int = 1, crowd: bool = False) -> int:
        annotation_id = len(self.annotations) + 1
        self.annotations.append(
            {
                "id": annotation_id,
                "image_id": 1,
                "category_id": category,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": int(crowd),
            }
        )
        return annotation_id

    def det(self, bbox: list[int], score: float, *, category: int = 1) -> int:
        self.detections.append(
            {"image_id": 1, "category_id": category, "bbox": bbox, "score": score}
        )
        return len(self.detections) - 1

    def alternative(self, rule: str, **predicts: Any) -> None:
        self.alternatives.append({"rule": rule, "predicts": predicts})

    def document(self, settings: CaseSettings) -> dict[str, Any]:
        return {
            "kind": KIND,
            "format_version": FORMAT_VERSION,
            "case": self.name,
            "separates": self.separates,
            "settings": {
                "iou_threshold": settings.iou_threshold,
                "max_detections": settings.max_detections,
                "bin_count": settings.bin_count,
                "threshold_grid": list(settings.grid),
            },
            "ground_truth": {
                "info": {"description": f"qcal_lab protocol case {self.name}"},
                "images": [
                    {
                        "id": 1,
                        "file_name": f"{self.name}.jpg",
                        "width": settings.image_width,
                        "height": settings.image_height,
                    }
                ],
                "annotations": self.annotations,
                "categories": [{"id": c, "name": f"class_{c}"} for c in self.categories],
            },
            "detections": self.detections,
            "alternatives": self.alternatives,
        }


def _exact(value: float) -> Fraction:
    """The decimal a setting was written as (0.55 is 11/20, not the binary float)."""
    return Fraction(repr(value))


def _worse_height(threshold: float) -> int:
    """The height of a box inside the competed-for object whose IoU is just above ``t``."""
    return math.ceil(_COMPETE_SIDE * threshold) + 1


def iou_xywh(a: list[int], b: list[int]) -> Fraction:
    """Exact IoU of two integer ``[x, y, w, h]`` boxes (COCO: no +1 on widths)."""
    w = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = w * h
    return Fraction(inter, a[2] * a[3] + b[2] * b[3] - inter)


def iou_at_threshold(settings: CaseSettings) -> Case:
    """A detection whose IoU with its object is exactly the threshold."""
    case = Case(
        "iou_at_threshold",
        "a match needs IoU >= t, or IoU > t (IoU of [x, y, w, h] boxes without a +1 on widths, "
        "as COCO computes it; with +1 the IoU here exceeds t and the case does not separate)",
    )
    ratio = _exact(settings.iou_threshold)  # p/q: an object q units wide, a detection p wide
    side = _UNIT * ratio.denominator
    target = case.gt([_UNIT, _UNIT, side, _UNIT])
    found = case.det([_UNIT, _UNIT, _UNIT * ratio.numerator, _UNIT], 0.9)
    case.alternative("IoU >= t matches", outcomes={str(found): f"TP:{target}"})
    case.alternative("IoU > t is required", outcomes={str(found): "FP"})
    return case


def equal_score_order(settings: CaseSettings) -> Case:
    """Two objects, each wanted by two detections of one score. For the first object the
    worse overlap comes first in input order; for the second, the better one does, so input
    order, best IoU and reversed input order each give different matches."""
    case = Case(
        "equal_score_order",
        "greedy matching among equal scores follows input order, the higher IoU, or reversed "
        "input order (e.g. a reversed ascending argsort)",
    )
    side, worse = _COMPETE_SIDE, _worse_height(settings.iou_threshold)
    left = case.gt([0, 0, side, side])
    right = case.gt([side + _UNIT, 0, side, side])
    a_worse = case.det([0, 0, side, worse], 0.8)
    a_better = case.det([0, 0, side, side], 0.8)
    b_better = case.det([side + _UNIT, 0, side, side], 0.8)
    b_worse = case.det([side + _UNIT, 0, side, worse], 0.8)

    def outcomes(*winners: int) -> dict[str, str]:
        target = {a_worse: left, a_better: left, b_better: right, b_worse: right}
        return {str(d): f"TP:{target[d]}" if d in winners else "FP" for d in sorted(target)}

    case.alternative(
        "input order breaks ties (a stable sort)", outcomes=outcomes(a_worse, b_better)
    )
    case.alternative("the higher IoU wins a tie", outcomes=outcomes(a_better, b_better))
    case.alternative("reversed input order breaks ties", outcomes=outcomes(a_better, b_worse))
    return case


def crowd_only_overlap(settings: CaseSettings) -> Case:  # noqa: ARG001 - uniform signature
    """Detections that overlap only a crowd region: one inside it (IoF 1, IoU small), one
    covering it (IoU 1)."""
    case = Case(
        "crowd_only_overlap",
        "a detection on a crowd region is ignored when its intersection over its own area "
        "(IoF) reaches t, ignored only when its IoU does, or a false positive",
    )
    region = 5 * _UNIT
    case.gt([0, 0, region, region], crowd=True)
    covering = case.det([0, 0, region, region], 0.8)
    inside = case.det([_UNIT, _UNIT, _UNIT, _UNIT], 0.7)
    case.alternative(
        "IoF against crowd regions; a match is ignored (pycocotools)",
        outcomes={str(covering): "ignored", str(inside): "ignored"},
    )
    case.alternative(
        "IoU against crowd regions; a match is ignored",
        outcomes={str(covering): "ignored", str(inside): "FP"},
    )
    case.alternative(
        "crowd regions are not ground truth",
        outcomes={str(covering): "FP", str(inside): "FP"},
    )
    return case


def scores_on_bin_edges(settings: CaseSettings) -> Case:
    """Scores exactly on bin edges (as floats), including 1.0."""
    case = Case(
        "scores_on_bin_edges", "bins are [k/B, (k+1)/B) with 1.0 in the last, or (k/B, (k+1)/B]"
    )
    bins = settings.bin_count
    edges = sorted({1, bins // 2, bins - 1, bins})
    left: dict[str, int] = {}
    right: dict[str, int] = {}
    for column, k in enumerate(edges):
        x = column * 2 * _UNIT
        case.gt([x, 0, _UNIT, _UNIT])
        index = case.det([x, 0, _UNIT, _UNIT], k / bins)
        left[str(index)] = min(k, bins - 1)
        right[str(index)] = k - 1
    case.alternative("left-closed bins; 1.0 in the last bin", bins=left)
    case.alternative("right-closed bins", bins=right)
    return case


def class_without_detections(settings: CaseSettings) -> Case:  # noqa: ARG001
    """Class 2 has an object and no detection."""
    case = Case(
        "class_without_detections",
        "a class with objects but no detections is averaged in, or skipped",
        categories=[1, 2],
    )
    target = case.gt([0, 0, _UNIT, _UNIT], category=1)
    case.gt([2 * _UNIT, 0, _UNIT, _UNIT], category=2)
    case.det([0, 0, _UNIT, _UNIT], 0.9, category=1)
    case.alternative("averaged in", classes_in_mean=[1, 2], outcomes={"0": f"TP:{target}"})
    case.alternative("skipped", classes_in_mean=[1], outcomes={"0": f"TP:{target}"})
    return case


def ties_at_max_detections(settings: CaseSettings) -> Case:
    """``max_detections + 1`` detections; the last kept and the first cut share a score."""
    case = Case("ties_at_max_detections", "the cut at max_detections keeps input order, or not")
    count = settings.max_detections + 1
    cells = math.ceil(math.sqrt(count))
    stride = min(settings.image_width, settings.image_height) // cells
    scores = [1 - Fraction(i + 1, count + 1) for i in range(count)]
    scores[-1] = scores[-2]  # ranks max_detections and max_detections + 1 tie
    for i, score in enumerate(scores):
        row, column = divmod(i, cells)
        box = [column * stride, row * stride, stride - 1, stride - 1]
        if i == 0:
            case.gt(box)  # one object, so the class is evaluated
        case.det(box, float(score))
    kept_last, first_cut = count - 2, count - 1
    case.alternative("a stable cut keeps input order", outcomes={str(first_cut): "dropped"})
    case.alternative("the tie goes the other way", outcomes={str(kept_last): "dropped"})
    return case


def threshold_selection_ties(settings: CaseSettings) -> Case:
    """Every score lies above the grid, so every threshold keeps the same detections."""
    case = Case(
        "threshold_selection_ties",
        "a grid search over thresholds (calibration or operating alike) that ties keeps the "
        "lowest tied threshold, or the highest",
    )
    score = (settings.grid[-1] + 1.0) / 2
    for column in range(2):
        x = column * 2 * _UNIT
        case.gt([x, 0, _UNIT, _UNIT])
        case.det([x, 0, _UNIT, _UNIT], score)
    case.alternative("the lowest tied threshold", threshold=settings.grid[0])
    case.alternative("the highest tied threshold", threshold=settings.grid[-1])
    return case


BUILDERS: Final[tuple[Callable[[CaseSettings], Case], ...]] = (
    iou_at_threshold,
    equal_score_order,
    crowd_only_overlap,
    scores_on_bin_edges,
    class_without_detections,
    ties_at_max_detections,
    threshold_selection_ties,
)


def build_cases(settings: CaseSettings) -> dict[str, bytes]:
    """File name -> canonical bytes, for every case."""
    files: dict[str, bytes] = {}
    for builder in BUILDERS:
        case = builder(settings)
        text = json.dumps(case.document(settings), indent=2, sort_keys=True) + "\n"
        files[case.name + SUFFIX] = text.encode("utf-8")
    return files


def drift(directory: Path, files: Mapping[str, bytes]) -> list[str]:
    """How the committed cases differ from the generator's; empty when they agree."""
    present = {p.name for p in directory.glob(f"*{SUFFIX}")} if directory.is_dir() else set()
    problems = [f"{name}: missing" for name in sorted(set(files) - present)]
    problems += [f"{name}: not generated" for name in sorted(present - set(files))]
    problems += [
        f"{name}: differs from the generator"
        for name in sorted(present & set(files))
        if (directory / name).read_bytes() != files[name]
    ]
    return problems


def write_cases(directory: Path, files: Mapping[str, bytes]) -> list[str]:
    """Write every case that changed; returns the names written."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name, data in sorted(files.items()):
        path = directory / name
        if not path.is_file() or path.read_bytes() != data:
            write_bytes_atomic(path, data)
            written.append(name)
    _log.info("protocol cases: %d written, %d unchanged", len(written), len(files) - len(written))
    return written


__all__ = [
    "BUILDERS",
    "FORMAT_VERSION",
    "KIND",
    "PREDICATES",
    "Case",
    "CaseSettings",
    "build_cases",
    "drift",
    "iou_xywh",
    "write_cases",
]
