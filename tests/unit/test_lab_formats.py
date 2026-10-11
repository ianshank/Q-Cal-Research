"""qcal_lab.formats: every saved document names its format and version."""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path

import pytest

from qcal_lab.formats import (
    FormatError,
    envelope,
    json_bytes,
    open_envelope,
    read_json,
    write_bytes_atomic,
    write_bytes_exclusive,
)


def test_an_envelope_round_trips_its_body() -> None:
    document = envelope("x.doc", 3, {"a": 1})
    assert document == {"format": "x.doc", "version": 3, "a": 1}
    assert open_envelope(document, "x.doc", 3, "here") == {"a": 1}


@pytest.mark.parametrize("key", ["format", "version"])
def test_a_body_may_not_set_the_envelope(key: str) -> None:
    with pytest.raises(FormatError, match="may not set"):
        envelope("x.doc", 1, {key: "forged"})


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (None, "is not a x.doc document"),
        ({"a": 1}, "is not a x.doc document"),
        ({"format": "y.doc", "version": 1}, "is not a x.doc document"),
        ({"format": "x.doc"}, "version None is not supported"),
        ({"format": "x.doc", "version": 2}, "version 2 is not supported"),
        ({"format": "x.doc", "version": True}, "version True is not supported"),  # True == 1
    ],
)
def test_a_document_of_another_format_or_version_is_refused(data: object, message: str) -> None:
    with pytest.raises(FormatError, match=message):
        open_envelope(data, "x.doc", 1, "here")


def test_the_caller_chooses_the_error_type() -> None:
    with pytest.raises(KeyError):
        open_envelope({}, "x.doc", 1, "here", error=KeyError)


def test_atomic_writes_return_the_digest_and_leave_no_temporary(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "f.json"
    data = json_bytes({"b": 1, "a": [1, 2]})
    assert data == b'{\n  "a": [\n    1,\n    2\n  ],\n  "b": 1\n}\n'
    assert write_bytes_atomic(path, data) == hashlib.sha256(data).hexdigest()
    assert path.read_bytes() == data
    assert [p.name for p in path.parent.iterdir()] == ["f.json"]
    assert read_json(path) == {"a": [1, 2], "b": 1}


def test_unreadable_json_is_a_format_error(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(FormatError, match="cannot read"):
        read_json(tmp_path / "bad.json")
    with pytest.raises(FormatError, match="cannot read"):
        read_json(tmp_path / "missing.json")


def test_exclusive_writes_never_replace_a_file(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "f.bin"
    assert write_bytes_exclusive(path, b"one") == hashlib.sha256(b"one").hexdigest()
    with pytest.raises(FileExistsError):
        write_bytes_exclusive(path, b"two")
    assert path.read_bytes() == b"one"
    assert [p.name for p in path.parent.iterdir()] == ["f.bin"]


@pytest.mark.parametrize("number", [errno.EPERM, errno.EOPNOTSUPP])
def test_exclusive_writes_work_without_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, number: int
) -> None:
    def no_links(*_: object) -> None:
        raise OSError(number, "no hard links here")

    monkeypatch.setattr(os, "link", no_links)
    path = tmp_path / "f.bin"
    assert write_bytes_exclusive(path, b"one") == hashlib.sha256(b"one").hexdigest()
    with pytest.raises(FileExistsError):
        write_bytes_exclusive(path, b"two")
    assert path.read_bytes() == b"one"
    assert [p.name for p in tmp_path.iterdir()] == ["f.bin"]


def test_other_link_errors_are_not_hidden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: object) -> None:
        raise OSError(errno.EIO, "input/output error")

    monkeypatch.setattr(os, "link", broken)
    with pytest.raises(OSError, match="input/output error"):
        write_bytes_exclusive(tmp_path / "f.bin", b"one")
