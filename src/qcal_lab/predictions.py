"""Prediction files: a JSON header line, then one JSON line per image (JSON Lines).

The bytes are deterministic: images sorted by id, keys sorted, floats written with
``repr``. So the same detections always hash the same, and the registry's artifact hash
identifies them. A detection is a list in the order of :data:`DETECTION_FIELDS`, which the
header repeats; ``logit`` is null when the detector does not expose one.

Version 2 headers also say what produced the predictions (:class:`PredictionSource`):
precision, deployment target, quantisation path, dataset, what the score means, the model's
sha256 and the test-time overrides. A cached file is reused only when all of it matches.

:class:`PredictionCache` stores raw predictions under a key derived from everything that
determines them. Calibrator and metric cells then re-read cached predictions instead of
re-running the detector (plan §5, Phase 3).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from qcal.log import get_logger
from qcal.protocols import Detection, ImageDetections
from qcal_lab.formats import (
    ENVELOPE_KEYS,
    envelope,
    json_bytes,
    open_envelope,
    write_bytes_atomic,
    write_bytes_exclusive,
)

_log = get_logger("lab.predictions")

FORMAT: Final = "qcal_lab.predictions"
VERSION: Final = 2
#: A detection row, in order. The last three are reserved for Phase 2 (the index of an FP32
#: detection a quantised one matches, and the logits behind a score). They are written as
#: null, and a reader refuses a value there rather than drop it.
DETECTION_FIELDS: Final = (
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
    "label",
    "logit",
    "source_index",
    "class_logit",
    "aux_logit",
)
RESERVED_FIELDS: Final = DETECTION_FIELDS[7:]
STAGES: Final = ("raw", "calibrated")
CACHE_ENTRY_FORMAT: Final = "qcal_lab.prediction_cache_entry"
CACHE_ENTRY_VERSION: Final = 1
_FIELDS: Final = len(DETECTION_FIELDS)
_REQUIRED_SOURCE: Final = ("precision", "target", "quant_path", "shift", "score_definition")


class PredictionsError(ValueError):
    """A predictions file or a detector's output is malformed."""


class CacheConflictError(PredictionsError):
    """A cache key already holds other bytes; the cache never replaces them.

    Records list cached files as artifacts, so replacing one would break every earlier
    record's artifact hash. Other bytes under one key mean the key misses an input that
    changed, the detector is not deterministic, or the entry was tampered with.
    """

    def __init__(self, key: str, existing_sha256: str, new_sha256: str) -> None:
        super().__init__(
            f"cached predictions {key[:12]} hold sha256 {existing_sha256[:12]}, this run "
            f"computed {new_sha256[:12]}; the cache keeps the existing bytes"
        )
        self.key, self.existing_sha256, self.new_sha256 = key, existing_sha256, new_sha256


@dataclass(frozen=True)
class PredictionSource:
    """What produced a set of predictions; every field but the last two must be non-empty."""

    precision: str
    target: str
    quant_path: str
    shift: str  # the dataset entry (datasets.<shift>)
    score_definition: str  # what a detection's score is, per detector kind
    model_sha256: str = ""  # of the checkpoint; empty when the detector has none (the fixture)
    test_cfg: Mapping[str, Any] = field(default_factory=dict)  # test-time overrides applied

    def __post_init__(self) -> None:
        empty = [name for name in _REQUIRED_SOURCE if not _text(getattr(self, name))]
        if empty:
            raise PredictionsError(f"a predictions source needs a non-empty {', '.join(empty)}")
        if not isinstance(self.model_sha256, str) or not isinstance(self.test_cfg, Mapping):
            raise PredictionsError(
                "a predictions source's model_sha256 must be a string and its test_cfg a mapping"
            )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "test_cfg": dict(self.test_cfg)}

    @classmethod
    def from_dict(cls, data: Any, where: str) -> PredictionSource:
        if not isinstance(data, Mapping):
            raise PredictionsError(f"{where}: the header needs a 'source' object")
        try:
            return cls(**{name: data[name] for name in cls.__dataclass_fields__})
        except KeyError as exc:
            raise PredictionsError(f"{where}: the header's source has no {exc}") from exc


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class PredictionsHeader:
    detector: str
    split: str
    stage: str  # one of STAGES
    dataset_sha256: str
    split_sha256: str
    category_ids: tuple[int, ...]
    source: PredictionSource

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise PredictionsError(f"stage must be one of {STAGES}, got {self.stage!r}")

    def to_dict(self) -> dict[str, Any]:
        data = {
            "detector": self.detector,
            "split": self.split,
            "stage": self.stage,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "category_ids": list(self.category_ids),
            "source": self.source.to_dict(),
            "fields": list(DETECTION_FIELDS),
        }
        return envelope(FORMAT, VERSION, data)


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
    reserved = [None] * len(RESERVED_FIELDS)
    return [*map(float, det.box_xyxy), float(det.score), int(det.label), det.logit, *reserved]


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
    return write_bytes_atomic(path, dumps(header, images))


def _decode(raw: Any, where: str) -> Detection:
    if not isinstance(raw, list) or len(raw) != _FIELDS:
        raise PredictionsError(f"{where}: a detection is a list of {_FIELDS} values")
    known = _FIELDS - len(RESERVED_FIELDS)
    *box, score, label, logit = raw[:known]
    used = [n for n, v in zip(RESERVED_FIELDS, raw[known:], strict=True) if v is not None]
    if used:
        raise PredictionsError(f"{where}: reserved field(s) {used} are set; this reader drops them")
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
    body = open_envelope(data, FORMAT, VERSION, str(path), error=PredictionsError)
    if body.get("fields") != list(DETECTION_FIELDS):
        raise PredictionsError(
            f"{path}: detection fields {body.get('fields')!r} are not {list(DETECTION_FIELDS)}"
        )
    try:
        return PredictionsHeader(
            detector=str(body["detector"]),
            split=str(body["split"]),
            stage=str(body["stage"]),
            dataset_sha256=str(body["dataset_sha256"]),
            split_sha256=str(body["split_sha256"]),
            category_ids=tuple(int(c) for c in body["category_ids"]),
            source=PredictionSource.from_dict(body.get("source"), str(path)),
        )
    except PredictionsError:
        raise
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
    """A content address for everything that determines a set of raw predictions.

    The predictions format and version are always part of it, so no part may use their names.
    """
    clash = sorted(ENVELOPE_KEYS & set(parts))
    if clash:
        raise PredictionsError(f"cache key parts may not be named {clash}; the key sets them")
    blob = json.dumps({"format": FORMAT, "version": VERSION, **parts}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    produced_by: str  # the run that computed these predictions
    created_at: str
    sha256: str = ""  # of the cached file's bytes


class PredictionCache:
    """Raw predictions keyed by :func:`cache_key`, each with a metadata sidecar.

    The sidecar records the file's sha256, the key and the run that produced it. A hit only
    proves the bytes are unchanged since that run; the caller must still check the header
    and image ids (:func:`qcal_lab.experiment` does), and the run record names the producer.
    Edit tools may not write under ``runs/cache/`` (``registry_only`` in ``qcal.toml``).
    """

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory

    def path(self, key: str) -> Path | None:
        return None if self.directory is None else self.directory / f"{key}.jsonl"

    @staticmethod
    def _sidecar(path: Path) -> Path:
        return path.with_suffix(".meta.json")

    def get(self, key: str) -> CacheEntry | None:
        """The cached entry if its bytes, key and metadata still agree."""
        path = self.path(key)
        if path is None or not path.is_file():
            return None
        try:
            document = json.loads(self._sidecar(path).read_text("utf-8"))
            meta = open_envelope(
                document, CACHE_ENTRY_FORMAT, CACHE_ENTRY_VERSION, path.name, error=PredictionsError
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, PredictionsError) as exc:
            _log.warning("ignoring cached predictions %s: unreadable metadata (%s)", path.name, exc)
            return None
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if not isinstance(meta, Mapping) or meta.get("sha256") != actual or meta.get("key") != key:
            _log.warning("ignoring cached predictions %s: metadata mismatch", path.name)
            return None
        return CacheEntry(
            path, str(meta.get("produced_by", "")), str(meta.get("created_at", "")), actual
        )

    def put(
        self,
        key: str,
        header: PredictionsHeader,
        images: Sequence[ImageDetections],
        *,
        produced_by: str,
    ) -> CacheEntry:
        """Cache ``images`` under ``key``, never replacing other bytes there.

        Identical bytes already under the key are kept, and the sidecar names ``produced_by``
        from now on (the earlier producer's record may be missing). Other bytes raise
        :class:`CacheConflictError`.
        """
        path = self.path(key)
        if path is None:
            raise PredictionsError("the prediction cache is disabled")
        data = dumps(header, images)
        digest = hashlib.sha256(data).hexdigest()
        try:
            write_bytes_exclusive(path, data)
        except FileExistsError:
            existing = hashlib.sha256(path.read_bytes()).hexdigest()
            if existing != digest:
                raise CacheConflictError(key, existing, digest) from None
            _log.info("cached predictions %s already hold these bytes", path.name)
        created = datetime.now(UTC).isoformat()
        meta = {"sha256": digest, "key": key, "produced_by": produced_by, "created_at": created}
        document = envelope(CACHE_ENTRY_FORMAT, CACHE_ENTRY_VERSION, meta)
        write_bytes_atomic(self._sidecar(path), json_bytes(document))
        return CacheEntry(path, produced_by, created, digest)


__all__ = [
    "CACHE_ENTRY_FORMAT",
    "CACHE_ENTRY_VERSION",
    "DETECTION_FIELDS",
    "FORMAT",
    "RESERVED_FIELDS",
    "STAGES",
    "VERSION",
    "CacheConflictError",
    "CacheEntry",
    "PredictionCache",
    "PredictionSource",
    "PredictionsError",
    "PredictionsHeader",
    "cache_key",
    "dumps",
    "read_predictions",
    "validate",
    "write_predictions",
]
