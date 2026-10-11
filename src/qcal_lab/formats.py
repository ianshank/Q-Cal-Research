"""Saved formats: every file this package writes names its format and version.

A reader refuses a document whose format or version it does not know, so a file written by
another version of the program is never misread. There are no legacy readers: no registered
run or committed artifact exists yet (docs/changes/run-identity-and-formats.md). After the
first registered run a format only gains fields, and a change of meaning bumps its version.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

FORMAT_KEY: Final = "format"
VERSION_KEY: Final = "version"
ENVELOPE_KEYS: Final = frozenset({FORMAT_KEY, VERSION_KEY})


class FormatError(ValueError):
    """A saved document lacks the format or version its reader expects."""


def envelope(name: str, version: int, body: Mapping[str, Any]) -> dict[str, Any]:
    """``body`` with its format name and version; the body may not set either itself."""
    clash = sorted(ENVELOPE_KEYS & set(body))
    if clash:
        raise FormatError(f"a {name} document may not set {clash} in its body")
    return {FORMAT_KEY: name, VERSION_KEY: version, **body}


def open_envelope(
    data: Any,
    name: str,
    version: int,
    where: str,
    *,
    error: type[Exception] = FormatError,
) -> dict[str, Any]:
    """The body of a ``name`` document of exactly ``version``; raises ``error`` otherwise."""
    if not isinstance(data, Mapping) or data.get(FORMAT_KEY) != name:
        raise error(f"{where} is not a {name} document")
    found = data.get(VERSION_KEY)
    if isinstance(found, bool) or found != version:
        raise error(f"{where}: {name} version {found!r} is not supported (this reads {version})")
    return {k: v for k, v in data.items() if k not in ENVELOPE_KEYS}


def write_bytes_atomic(path: Path, data: bytes) -> str:
    """Write ``data`` through a temporary file and a rename; returns its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return hashlib.sha256(data).hexdigest()


def write_bytes_exclusive(path: Path, data: bytes) -> str:
    """Create ``path`` holding ``data``, atomically; :class:`FileExistsError` if it exists.

    The bytes go to a temporary file first and are hard-linked into place, so a reader never
    sees a partial file and two writers never both succeed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()
    return hashlib.sha256(data).hexdigest()


def json_bytes(document: Mapping[str, Any]) -> bytes:
    """Canonical, human-readable JSON: sorted keys, two-space indent, a final newline."""
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_json(path: Path, *, error: type[Exception] = FormatError) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise error(f"cannot read {path}: {exc}") from exc


__all__ = [
    "ENVELOPE_KEYS",
    "FORMAT_KEY",
    "VERSION_KEY",
    "FormatError",
    "envelope",
    "json_bytes",
    "open_envelope",
    "read_json",
    "write_bytes_atomic",
    "write_bytes_exclusive",
]
