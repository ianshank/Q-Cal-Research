"""qcal_lab.predictions: deterministic JSON Lines files and the content-addressed cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qcal.protocols import Detection
from qcal_lab.predictions import (
    PredictionCache,
    PredictionsError,
    PredictionsHeader,
    cache_key,
    dumps,
    read_predictions,
    validate,
    write_predictions,
)
from tests.lab_support import det, image

HEADER = PredictionsHeader("fixture", "val", "raw", "d" * 64, "s" * 64, (1, 3))
IMAGES = (
    image("2", det(0.5, 1, (1, 2, 3, 4))),
    image("1", Detection((0.0, 0.0, 1.5, 2.5), 0.25, 0, logit=-1.1)),
    image("3"),
)


def test_round_trip_and_deterministic_bytes(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    digest = write_predictions(path, HEADER, IMAGES)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    header, images = read_predictions(path)
    assert header == HEADER
    assert [i.image_id for i in images] == ["1", "2", "3"]
    assert images[0].detections[0].logit == -1.1
    assert dumps(HEADER, list(reversed(IMAGES))) == path.read_bytes()


@pytest.mark.parametrize(
    ("images", "message"),
    [
        ((image("1", det(1.5)),), "outside"),
        ((image("1", det(0.5, box=(5, 0, 1, 1))),), "inverted"),
        ((image("1", det(float("nan"))),), "non-finite"),
        ((image("1", det(0.5, label=2)),), "label 2"),
        ((image("1", det(0.5, label=True)),), "label True"),  # type: ignore[arg-type]
        ((image("1"), image("1")), "more than once"),
    ],
)
def test_invalid_detections_are_refused(images: tuple, message: str) -> None:
    with pytest.raises(PredictionsError, match=message):
        validate(images, num_classes=2)


def _write_lines(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def _header(**changes: object) -> str:
    data = {**HEADER.to_dict(), **changes}
    return json.dumps(data)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ((), "is empty"),
        (("not json",), "header is not JSON"),
        ((json.dumps({"format": "other"}),), "is not a qcal_lab.predictions file"),
        ((_header(version=99),), "unsupported version"),
        ((_header(category_ids=None),), "incomplete header"),
        ((_header(), "nope"), "not JSON"),
        ((_header(), json.dumps({"detections": []})), "needs an image_id"),
        ((_header(), json.dumps({"image_id": "1", "detections": {}})), "must be a list"),
        ((_header(), json.dumps({"image_id": "1", "detections": [[1, 2]]})), "list of 7"),
        (
            (
                _header(),
                json.dumps({"image_id": "1", "detections": [[0, 0, 1, 1, 0.5, 1.5, None]]}),
            ),
            "label must be an integer",
        ),
        (
            (
                _header(),
                json.dumps({"image_id": "1", "detections": [[0, 0, 1, "1", 0.5, 0, None]]}),
            ),
            "must be numbers",
        ),
    ],
)
def test_malformed_files_are_refused(tmp_path: Path, lines: tuple[str, ...], message: str) -> None:
    with pytest.raises(PredictionsError, match=message):
        read_predictions(_write_lines(tmp_path, *lines))


def test_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(PredictionsError, match="cannot read"):
        read_predictions(path)


def test_cache_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "cache")
    key = cache_key(detector={"name": "x"}, split_sha256="s")
    assert cache.get(key) is None
    path = cache.put(key, HEADER, IMAGES)
    assert cache.get(key) == path
    path.write_bytes(path.read_bytes() + b"\n")
    assert cache.get(key) is None  # digest mismatch: ignored, recomputed by the caller


def test_cache_keys_depend_on_every_part() -> None:
    assert cache_key(a=1) == cache_key(a=1)
    assert cache_key(a=1) != cache_key(a=2)
    assert cache_key(a=1) != cache_key(a=1, b=0)


def test_disabled_cache(tmp_path: Path) -> None:
    cache = PredictionCache(None)
    assert cache.path("k") is None
    assert cache.get("k") is None
    with pytest.raises(PredictionsError, match="disabled"):
        cache.put("k", HEADER, IMAGES)
