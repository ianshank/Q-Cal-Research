"""qcal_lab.data.coco: COCO-format ground truth, validated."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qcal_lab.data.coco import DatasetError, load_coco, parse_coco
from tests.lab_support import coco_doc


def test_boxes_become_xyxy_and_labels_follow_ascending_category_ids() -> None:
    doc = coco_doc(
        categories=(7, 3), boxes={1: [(7, [1.0, 2.0, 3.0, 4.0])], 2: [(3, [0, 0, 1, 1])]}
    )
    gt = parse_coco(doc, sha256="abc")
    assert gt.category_ids == (3, 7)
    assert gt.num_classes == 2
    assert gt.boxes["1"][0].box_xyxy == (1.0, 2.0, 4.0, 6.0)
    assert gt.boxes["1"][0].label == gt.label_of(7) == 1
    assert gt.boxes["2"][0].label == 0
    assert gt.boxes["3"] == ()
    assert gt.image_ids() == ("1", "2", "3")
    assert gt.sha256 == "abc"


def test_subset_keeps_the_label_map_and_refuses_unknown_ids() -> None:
    gt = parse_coco(coco_doc(images=3))
    sub = gt.subset(["2", "1", "1"])
    assert sub.image_ids() == ("1", "2")
    assert sub.label_of(3) == gt.label_of(3)
    with pytest.raises(DatasetError, match="not in the dataset"):
        gt.subset(["1", "99"])
    with pytest.raises(DatasetError, match="unknown category id 42"):
        gt.label_of(42)


def test_load_coco_records_the_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text(json.dumps(coco_doc()))
    gt = load_coco(path)
    assert len(gt.sha256) == 64
    assert gt.description == ""


@pytest.mark.parametrize("content", ["{not json", b"\xff"])
def test_load_coco_refuses_non_json(tmp_path: Path, content: str | bytes) -> None:
    path = tmp_path / "a.json"
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    with pytest.raises(DatasetError, match="not a JSON document"):
        load_coco(path)


def test_load_coco_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="does not exist"):
        load_coco(tmp_path / "missing.json")


def _mutate(path: str, value: Any) -> dict[str, Any]:
    doc = coco_doc()
    section, index, key = path.split(".")
    doc[section][int(index)][key] = value
    return doc


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        (_mutate("images.0.id", True), "image id"),
        (_mutate("images.0.id", ""), "image id"),
        (_mutate("images.0.file_name", 3), "file_name"),
        (_mutate("images.0.width", -1), "width and height"),
        (_mutate("annotations.0.bbox", [1, 2, 3]), "bbox must be"),
        (_mutate("annotations.0.bbox", [1, 2, -3, 4]), "non-negative"),
        (_mutate("annotations.0.bbox", [1, 2, float("nan"), 4]), "finite"),
        (_mutate("annotations.0.image_id", 99), "unknown image"),
        (_mutate("annotations.0.category_id", 2), "unknown category_id"),
        (_mutate("annotations.0.category_id", True), "unknown category_id"),
        (_mutate("annotations.0.iscrowd", 2), "iscrowd"),
        (_mutate("annotations.0.iscrowd", 1.0), "iscrowd"),
        (_mutate("categories.0.id", "1"), "category id must be an integer"),
    ],
)
def test_malformed_documents_are_refused(doc: dict[str, Any], message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        parse_coco(doc)


def test_duplicates_and_shapes_are_refused() -> None:
    doc = coco_doc()
    doc["images"].append(dict(doc["images"][0]))
    with pytest.raises(DatasetError, match="duplicate image id"):
        parse_coco(doc)
    doc = coco_doc()
    doc["categories"].append({"id": 1})
    with pytest.raises(DatasetError, match="duplicate category id"):
        parse_coco(doc)
    with pytest.raises(DatasetError, match="must be a list of objects"):
        parse_coco({"images": {"id": 1}})
    with pytest.raises(DatasetError, match="JSON object"):
        parse_coco([])  # type: ignore[arg-type]


def test_crowd_and_area_are_kept() -> None:
    doc = coco_doc()
    doc["annotations"][0].update({"iscrowd": 1, "area": 12.5})
    box = parse_coco(doc).boxes["1"][0]
    assert box.iscrowd
    assert box.area == 12.5
