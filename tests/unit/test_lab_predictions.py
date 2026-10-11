"""qcal_lab.predictions: deterministic JSON Lines files and the content-addressed cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qcal.protocols import Detection
from qcal_lab.predictions import (
    CACHE_ENTRY_FORMAT,
    DETECTION_FIELDS,
    CacheConflictError,
    PredictionCache,
    PredictionsError,
    PredictionSource,
    cache_key,
    dumps,
    read_predictions,
    validate,
    write_predictions,
)
from tests.lab_support import SOURCE, det, image, predictions_header

HEADER = predictions_header()
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


def _row(known: list[object], **reserved: object) -> str:
    """One image row with one detection: the known fields, then the reserved ones (null)."""
    tail = [reserved.get(name) for name in DETECTION_FIELDS[len(known) :]]
    return json.dumps({"image_id": "1", "detections": [[*known, *tail]]})


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ((), "is empty"),
        (("not json",), "header is not JSON"),
        ((json.dumps({"format": "other"}),), "is not a qcal_lab.predictions document"),
        ((_header(version=99),), "version 99 is not supported"),
        ((_header(version=1),), "version 1 is not supported"),  # no legacy reader
        ((_header(fields=["x1"]),), "detection fields"),
        ((_header(source=None),), "needs a 'source' object"),
        ((_header(source={**SOURCE.to_dict(), "target": ""}),), "non-empty target"),
        ((_header(source={"precision": "fp32"}),), "source has no 'target'"),
        ((_header(stage="final"),), "stage must be one of"),
        ((_header(category_ids=None),), "incomplete header"),
        ((_header(), "nope"), "not JSON"),
        ((_header(), json.dumps({"detections": []})), "needs an image_id"),
        ((_header(), json.dumps({"image_id": "1", "detections": {}})), "must be a list"),
        ((_header(), json.dumps({"image_id": "1", "detections": [[1, 2]]})), "list of 10"),
        ((_header(), _row([0, 0, 1, 1, 0.5, 1.5, None])), "label must be an integer"),
        ((_header(), _row([0, 0, 1, "1", 0.5, 0, None])), "must be numbers"),
        ((_header(), _row([0, 0, 1, 1, 0.5, 0, None], source_index=3)), r"\['source_index'\]"),
        ((_header(), _row([0, 0, 1, 1, 0.5, 0, None], aux_logit=0.1)), r"\['aux_logit'\]"),
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
    entry = cache.put(key, HEADER, IMAGES, produced_by="R-1")
    found = cache.get(key)
    assert found is not None
    assert (found.path, found.produced_by) == (entry.path, "R-1")
    meta = json.loads(entry.path.with_suffix(".meta.json").read_text())
    assert meta["key"] == key
    assert meta["produced_by"] == "R-1"
    entry.path.write_bytes(entry.path.read_bytes() + b"\n")
    assert cache.get(key) is None  # digest mismatch: ignored, recomputed by the caller


def test_cache_metadata_must_name_the_key(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "cache")
    entry = cache.put("k1", HEADER, IMAGES, produced_by="R-1")
    moved = entry.path.with_name("k2.jsonl")
    entry.path.rename(moved)
    entry.path.with_suffix(".meta.json").rename(moved.with_suffix(".meta.json"))
    assert cache.get("k2") is None  # a copied entry does not answer for another key
    moved.with_suffix(".meta.json").unlink()
    assert cache.get("k2") is None


def test_cache_keys_depend_on_every_part() -> None:
    assert cache_key(a=1) == cache_key(a=1)
    assert cache_key(a=1) != cache_key(a=2)
    assert cache_key(a=1) != cache_key(a=1, b=0)


def test_disabled_cache(tmp_path: Path) -> None:
    cache = PredictionCache(None)
    assert cache.path("k") is None
    assert cache.get("k") is None
    with pytest.raises(PredictionsError, match="disabled"):
        cache.put("k", HEADER, IMAGES, produced_by="R")


# -- version 2 (docs/changes/run-identity-and-formats.md) -------------------------------------


def test_the_header_says_what_produced_the_predictions(tmp_path: Path) -> None:
    source = PredictionSource(
        precision="fp32_tf32_off",
        target="torch_fp32",
        quant_path="none",
        shift="id",
        score_definition="mmdet: pred_instances.scores",
        model_sha256="c" * 64,
        test_cfg={"score_threshold": 0.05, "max_per_image": 100},
    )
    header = predictions_header(source=source)
    write_predictions(tmp_path / "p.jsonl", header, IMAGES)
    found, _ = read_predictions(tmp_path / "p.jsonl")
    assert found == header
    first = json.loads((tmp_path / "p.jsonl").read_text().splitlines()[0])
    assert first["fields"] == list(DETECTION_FIELDS)
    assert first["source"]["test_cfg"] == {"score_threshold": 0.05, "max_per_image": 100}


def test_rows_reserve_three_fields_written_as_null() -> None:
    row = json.loads(dumps(HEADER, IMAGES).decode().splitlines()[1])
    assert row["detections"][0][-3:] == [None, None, None]


@pytest.mark.parametrize(
    "field", ["precision", "target", "quant_path", "shift", "score_definition"]
)
def test_a_source_must_name_what_produced_the_predictions(field: str) -> None:
    with pytest.raises(PredictionsError, match=f"non-empty {field}"):
        PredictionSource(**{**SOURCE.to_dict(), field: " "})


def test_a_source_refuses_mistyped_fields() -> None:
    with pytest.raises(PredictionsError, match="model_sha256 must be a string"):
        PredictionSource(**{**SOURCE.to_dict(), "model_sha256": None})
    with pytest.raises(PredictionsError, match="test_cfg a mapping"):
        PredictionSource(**{**SOURCE.to_dict(), "test_cfg": []})


def test_a_header_with_another_source_is_a_different_header() -> None:
    other = predictions_header(source=PredictionSource(**{**SOURCE.to_dict(), "target": "x"}))
    assert other != HEADER


@pytest.mark.parametrize("part", ["format", "version"])
def test_cache_key_parts_may_not_replace_the_format(part: str) -> None:
    with pytest.raises(PredictionsError, match="may not be named"):
        cache_key(**{part: "x"})


def test_a_cache_sidecar_without_its_envelope_is_a_miss(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cache = PredictionCache(tmp_path / "cache")
    entry = cache.put("k1", HEADER, IMAGES, produced_by="R-1")
    sidecar = entry.path.with_suffix(".meta.json")
    meta = json.loads(sidecar.read_text())
    assert meta["format"] == CACHE_ENTRY_FORMAT
    del meta["format"]
    sidecar.write_text(json.dumps(meta))
    assert cache.get("k1") is None
    assert "unreadable metadata" in caplog.text


def test_the_cache_never_replaces_other_bytes_under_a_key(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "cache")
    first = cache.put("k1", HEADER, IMAGES, produced_by="R-1")
    original = first.path.read_bytes()
    with pytest.raises(CacheConflictError, match="the cache keeps the existing bytes") as caught:
        cache.put("k1", HEADER, IMAGES[:1], produced_by="R-2")
    assert caught.value.existing_sha256 == first.sha256
    assert first.path.read_bytes() == original
    again = cache.put("k1", HEADER, IMAGES, produced_by="R-3")  # identical bytes are kept
    assert (again.sha256, cache.get("k1").produced_by) == (first.sha256, "R-3")  # type: ignore[union-attr]
    assert not list(first.path.parent.glob(".*.tmp"))
