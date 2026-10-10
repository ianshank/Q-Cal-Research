"""Prediction files: a JSON header line, then one JSON line per image (JSON Lines).

The bytes are deterministic: images sorted by id, keys sorted, floats written with
``repr``. So the same detections always hash the same, and the registry's artifact hash
identifies them. A detection is ``[x1, y1, x2, y2, score, label, logit]``, with ``logit``
null when the detector does not expose one.

:class:`PredictionCache` stores raw predictions under a key derived from everything that
determines them. Calibrator and metric cells then re-read cached predictions instead of
re-running the detector (plan §5, Phase 3).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from qcal.log import get_logger
from qcal.protocols import Detection, ImageDetections

_log = get_logger("lab.predictions")

FORMAT: Final = "qcal_lab.predictions"
VERSION: Final = 1
_FIELDS: Final = 7


class PredictionsError(ValueError):
    """A predictions file or a detector's output is malformed."""


@dataclass(frozen=True)
class PredictionsHeader:
    detector: str
    split: str
    stage: str  # "raw" | "calibrated"
    dataset_sha256: str
    split_sha256: str
    category_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["category_ids"] = list(self.category_ids)
        return {"format": FORMAT, "version": VERSION, **data}


def validate(images: Sequence[ImageDetections], num_classes: int) -> None:
    """Refuse non-finite or out-of-range values, inverted boxes and repeated images."""
    seen: set[str] = set()
    for image in images:
        if image.image_id in seen:
            raise PredictionsError(f"image {image.image_id} appears more than once")
        seen.add(image.image_id)
        for det in image.detections:
            where = f"image {image.image_id}"
            values = (*det.box_xyxy, det.score) + (() if det.logit is None else (det.logit,))
            if not all(isinstance(v, int | float) and math.isfinite(v) for v in values):
                raise PredictionsError(f"{where}: non-finite detection value")
            if not 0.0 <= det.score <= 1.0:
                raise PredictionsError(f"{where}: score {det.score} is outside [0, 1]")
            x1, y1, x2, y2 = det.box_xyxy
            if x2 < x1 or y2 < y1:
                raise PredictionsError(f"{where}: box {det.box_xyxy} is inverted")
            if isinstance(det.label, bool) or not 0 <= det.label < num_classes:
                raise PredictionsError(f"{where}: label {det.label} outside 0..{num_classes - 1}")


def _encode(det: Detection) -> list[Any]:
    return [*map(float, det.box_xyxy), float(det.score), int(det.label), det.logit]


def dumps(header: PredictionsHeader, images: Sequence[ImageDetections]) -> bytes:
    validate(images, len(header.category_ids))
    lines = [json.dumps(header.to_dict(), sort_keys=True, separators=(",", ":"))]
    for image in sorted(images, key=lambda i: i.image_id):
        row = {"image_id": image.image_id, "detections": [_encode(d) for d in image.detections]}
        lines.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_predictions(
    path: Path, header: PredictionsHeader, images: Sequence[ImageDetections]
) -> str:
    """Write atomically and return the sha256 of the bytes written."""
    data = dumps(header, images)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return hashlib.sha256(data).hexdigest()


def _decode(raw: Any, where: str) -> Detection:
    if not isinstance(raw, list) or len(raw) != _FIELDS:
        raise PredictionsError(f"{where}: a detection is a list of {_FIELDS} values")
    *box, score, label, logit = raw
    if isinstance(label, bool) or not isinstance(label, int):
        raise PredictionsError(f"{where}: label must be an integer")
    numbers = [*box, score] + ([] if logit is None else [logit])
    if any(isinstance(v, bool) or not isinstance(v, int | float) for v in numbers):
        raise PredictionsError(f"{where}: box, score and logit must be numbers")
    return Detection(
        (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
        float(score),
        label,
        None if logit is None else float(logit),
    )


def _parse_header(line: str, path: Path) -> PredictionsHeader:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise PredictionsError(f"{path}: header is not JSON: {exc}") from exc
    if not isinstance(data, Mapping) or data.get("format") != FORMAT:
        raise PredictionsError(f"{path} is not a {FORMAT} file")
    if data.get("version") != VERSION:
        raise PredictionsError(f"{path}: unsupported version {data.get('version')!r}")
    try:
        return PredictionsHeader(
            detector=str(data["detector"]),
            split=str(data["split"]),
            stage=str(data["stage"]),
            dataset_sha256=str(data["dataset_sha256"]),
            split_sha256=str(data["split_sha256"]),
            category_ids=tuple(int(c) for c in data["category_ids"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionsError(f"{path}: incomplete header: {exc}") from exc


def read_predictions(path: Path) -> tuple[PredictionsHeader, tuple[ImageDetections, ...]]:
    try:
        lines = path.read_text("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise PredictionsError(f"cannot read {path}: {exc}") from exc
    if not lines:
        raise PredictionsError(f"{path} is empty")
    header = _parse_header(lines[0], path)
    images: list[ImageDetections] = []
    for number, line in enumerate(lines[1:], start=2):
        where = f"{path}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PredictionsError(f"{where}: not JSON: {exc}") from exc
        if not isinstance(row, Mapping) or not isinstance(row.get("image_id"), str):
            raise PredictionsError(f"{where}: needs an image_id")
        dets = row.get("detections")
        if not isinstance(dets, list):
            raise PredictionsError(f"{where}: detections must be a list")
        images.append(ImageDetections(row["image_id"], tuple(_decode(d, where) for d in dets)))
    validate(images, len(header.category_ids))
    return header, tuple(images)


def cache_key(**parts: Any) -> str:
    """A content address for everything that determines a set of raw predictions."""
    blob = json.dumps({"format": FORMAT, "version": VERSION, **parts}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class PredictionCache:
    """Raw predictions keyed by :func:`cache_key`, each stored with its own sha256."""

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory

    def path(self, key: str) -> Path | None:
        return None if self.directory is None else self.directory / f"{key}.jsonl"

    def get(self, key: str) -> Path | None:
        """The cached file if it exists and its bytes still match the stored digest."""
        path = self.path(key)
        if path is None or not path.is_file():
            return None
        sidecar = path.with_suffix(".sha256")
        expected = sidecar.read_text("utf-8").strip() if sidecar.is_file() else ""
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            _log.warning("ignoring cached predictions %s: digest mismatch", path.name)
            return None
        return path

    def put(self, key: str, header: PredictionsHeader, images: Sequence[ImageDetections]) -> Path:
        path = self.path(key)
        if path is None:
            raise PredictionsError("the prediction cache is disabled")
        digest = write_predictions(path, header, images)
        sidecar = path.with_suffix(".sha256")
        tmp = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
        tmp.write_text(digest + "\n", encoding="utf-8")
        tmp.replace(sidecar)
        return path


__all__ = [
    "FORMAT",
    "VERSION",
    "PredictionCache",
    "PredictionsError",
    "PredictionsHeader",
    "cache_key",
    "dumps",
    "read_predictions",
    "validate",
    "write_predictions",
]
