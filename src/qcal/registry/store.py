"""Directory of immutable run records. Writes are exclusive: a record is never overwritten."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from qcal.log import get_logger
from qcal.registry.records import RecordError, RunRecord, validate_run_id

_log = get_logger("registry.store")
_SUFFIX = ".json"


class RecordExistsError(FileExistsError):
    """A record with this run id already exists; records are immutable."""


class RegistryStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, run_id: str) -> Path:
        return self.directory / f"{validate_run_id(run_id)}{_SUFFIX}"

    def exists(self, run_id: str) -> bool:
        return self.path_for(run_id).exists()

    def write(self, record: RunRecord) -> Path:
        """Atomically create the record file; refuse to replace an existing one."""
        target = self.path_for(record.run_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        text = json.dumps(record.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=_SUFFIX, dir=self.directory)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(tmp, target)  # atomic, fails if target exists
            except FileExistsError:
                raise RecordExistsError(
                    f"record {record.run_id} already exists at {target}"
                ) from None
            except OSError:  # filesystem without hard links: exclusive create instead
                try:
                    with target.open("x", encoding="utf-8") as handle:
                        handle.write(text)
                except FileExistsError:
                    raise RecordExistsError(
                        f"record {record.run_id} already exists at {target}"
                    ) from None
        finally:
            tmp.unlink(missing_ok=True)
        _log.info("registered run %s (%s) -> %s", record.run_id, record.status, target)
        return target

    def read(self, run_id: str) -> RunRecord:
        return self._load(self.path_for(run_id))

    def iter_records(self, *, strict: bool = False) -> Iterator[RunRecord]:
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob(f"*{_SUFFIX}")):
            if path.name.startswith("."):
                continue
            try:
                record = self._load(path)
            except (RecordError, json.JSONDecodeError, OSError) as exc:
                if strict:
                    raise RecordError(f"{path}: {exc}") from exc
                _log.warning("skipping malformed record %s: %s", path, exc)
                continue
            if record.run_id != path.stem:
                message = f"{path}: run_id {record.run_id!r} does not match the file name"
                if strict:
                    raise RecordError(message)
                _log.warning("skipping record: %s", message)
                continue
            yield record

    def load_all(self, *, strict: bool = False) -> list[RunRecord]:
        return list(self.iter_records(strict=strict))

    @staticmethod
    def _load(path: Path) -> RunRecord:
        data = json.loads(path.read_text("utf-8"))
        if not isinstance(data, dict):
            raise RecordError("record must be a JSON object")
        return RunRecord.from_dict(data)
