"""COCO-format ground truth, loaded with the standard library.

Only what detection evaluation needs is kept: image metadata, boxes in ``xyxy`` order,
the crowd flag and area, and the category list. Category ids map to contiguous labels in
ascending id order, which is the order MMDetection's COCO dataset uses for the standard
80 classes [unverified for other annotation files; every predictions file stores the label
map, so a mismatch is visible rather than silent].
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qcal.log import get_logger

_log = get_logger("lab.data.coco")

Box = tuple[float, float, float, float]


class DatasetError(ValueError):
    """The annotation file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ImageInfo:
    image_id: str
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class GroundTruthBox:
    box_xyxy: Box
    label: int
    iscrowd: bool = False
    area: float = 0.0


@dataclass(frozen=True)
class GroundTruth:
    """Images, their boxes and the label map of one annotation file (or a subset of it)."""

    images: Mapping[str, ImageInfo]
    boxes: Mapping[str, tuple[GroundTruthBox, ...]]
    category_ids: tuple[int, ...]  # label -> COCO category id
    category_names: tuple[str, ...]
    sha256: str = ""  # of the source annotation bytes
    description: str = ""  # info.description
    info: Mapping[str, Any] = field(default_factory=dict, repr=False)  # the document's info
    _label_by_category: Mapping[int, int] = field(default_factory=dict, repr=False)

    @property
    def num_classes(self) -> int:
        return len(self.category_ids)

    def image_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.images))

    def label_of(self, category_id: int) -> int:
        try:
            return self._label_by_category[category_id]
        except KeyError:
            raise DatasetError(f"unknown category id {category_id}") from None

    def subset(self, image_ids: Iterable[str]) -> GroundTruth:
        """The same dataset restricted to ``image_ids``; unknown ids are an error."""
        wanted = sorted(set(image_ids))
        missing = [i for i in wanted if i not in self.images]
        if missing:
            shown = ", ".join(missing[:5])
            raise DatasetError(f"{len(missing)} image id(s) are not in the dataset: {shown}")
        return GroundTruth(
            images={i: self.images[i] for i in wanted},
            boxes={i: self.boxes.get(i, ()) for i in wanted},
            category_ids=self.category_ids,
            category_names=self.category_names,
            sha256=self.sha256,
            description=self.description,
            info=self.info,
            _label_by_category=self._label_by_category,
        )

    def without_boxes(self) -> GroundTruth:
        """Image metadata and the label map only: what a real detector may see."""
        return GroundTruth(
            images=self.images,
            boxes=dict.fromkeys(self.images, ()),
            category_ids=self.category_ids,
            category_names=self.category_names,
            sha256=self.sha256,
            description=self.description,
            info=self.info,
            _label_by_category=self._label_by_category,
        )


def _as_id(value: Any, what: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int | str) or value == "":
        raise DatasetError(f"{what} must be an integer or a non-empty string, got {value!r}")
    return str(value)


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise DatasetError(f"{what} must be a finite number, got {value!r}")
    return float(value)


def _records(data: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    raw = data.get(key, [])
    if not isinstance(raw, list) or not all(isinstance(r, Mapping) for r in raw):
        raise DatasetError(f"{key!r} must be a list of objects")
    return list(raw)


def _parse_images(data: Mapping[str, Any]) -> dict[str, ImageInfo]:
    images: dict[str, ImageInfo] = {}
    for entry in _records(data, "images"):
        image_id = _as_id(entry.get("id"), "image id")
        if image_id in images:
            raise DatasetError(f"duplicate image id {image_id}")
        file_name = entry.get("file_name", "")
        width, height = entry.get("width", 0), entry.get("height", 0)
        if not isinstance(file_name, str):
            raise DatasetError(f"image {image_id}: file_name must be a string")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (width, height)):
            raise DatasetError(f"image {image_id}: width and height must be non-negative integers")
        images[image_id] = ImageInfo(image_id, file_name, width, height)
    return images


def _parse_categories(data: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[str, ...]]:
    categories: dict[int, str] = {}
    for entry in _records(data, "categories"):
        cid = entry.get("id")
        if isinstance(cid, bool) or not isinstance(cid, int):
            raise DatasetError(f"category id must be an integer, got {cid!r}")
        if cid in categories:
            raise DatasetError(f"duplicate category id {cid}")
        categories[cid] = str(entry.get("name", cid))
    ordered = sorted(categories)
    return tuple(ordered), tuple(categories[c] for c in ordered)


def _parse_box(entry: Mapping[str, Any], label: int, where: str) -> GroundTruthBox:
    bbox = entry.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise DatasetError(f"{where}: bbox must be [x, y, width, height]")
    x, y, w, h = (_number(v, f"{where} bbox") for v in bbox)
    if w < 0 or h < 0:
        raise DatasetError(f"{where}: bbox width and height must be non-negative")
    crowd = entry.get("iscrowd", 0)
    if crowd not in (0, 1) or isinstance(crowd, float):
        raise DatasetError(f"{where}: iscrowd must be 0 or 1")
    area = _number(entry.get("area", w * h), f"{where} area")
    return GroundTruthBox((x, y, x + w, y + h), label, bool(crowd), area)


def parse_coco(data: Mapping[str, Any], *, sha256: str = "") -> GroundTruth:
    """Validate and convert a decoded COCO annotation document."""
    if not isinstance(data, Mapping):
        raise DatasetError("a COCO annotation file must contain a JSON object")
    images = _parse_images(data)
    category_ids, names = _parse_categories(data)
    label_by_category = {cid: label for label, cid in enumerate(category_ids)}
    boxes: dict[str, list[GroundTruthBox]] = {i: [] for i in images}
    for index, entry in enumerate(_records(data, "annotations")):
        where = f"annotation {entry.get('id', index)}"
        image_id = _as_id(entry.get("image_id"), f"{where} image_id")
        if image_id not in images:
            raise DatasetError(f"{where} refers to unknown image {image_id}")
        cid = entry.get("category_id")
        if isinstance(cid, bool) or not isinstance(cid, int) or cid not in label_by_category:
            raise DatasetError(f"{where} has unknown category_id {cid!r}")
        boxes[image_id].append(_parse_box(entry, label_by_category[cid], where))
    info = data.get("info")
    info = dict(info) if isinstance(info, Mapping) else {}
    description = str(info.get("description", ""))
    return GroundTruth(
        images=images,
        boxes={k: tuple(v) for k, v in boxes.items()},
        category_ids=category_ids,
        category_names=names,
        sha256=sha256,
        description=description,
        info=info,
        _label_by_category=label_by_category,
    )


def load_coco(path: Path) -> GroundTruth:
    """Load and validate a COCO-format annotation file."""
    if not path.is_file():
        raise DatasetError(f"annotation file {path} does not exist")
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"{path} is not a JSON document: {exc}") from exc
    gt = parse_coco(data, sha256=hashlib.sha256(raw).hexdigest())
    _log.info(
        "loaded %s: %d images, %d boxes, %d classes",
        path.name,
        len(gt.images),
        sum(len(b) for b in gt.boxes.values()),
        gt.num_classes,
    )
    return gt


__all__ = [
    "Box",
    "DatasetError",
    "GroundTruth",
    "GroundTruthBox",
    "ImageInfo",
    "load_coco",
    "parse_coco",
]
