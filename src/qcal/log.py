"""Logging setup shared by every qcal entry point.

Resolution order for the level: explicit argument, ``$QCAL_DEBUG`` (forces DEBUG),
``$QCAL_LOG_LEVEL``, then ``logging.level`` from configuration. The environment
variable names themselves come from configuration.

Stdlib-only: imported by Claude Code hooks.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import IO, Any, Final

from qcal.config import Config, ConfigError

PACKAGE_LOGGER: Final = "qcal"
_HANDLER_MARKER: Final = "_qcal_handler"
_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})
_STANDARD_ATTRS: Final = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}
_TEXT_FORMAT: Final = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the package namespace."""
    if name == PACKAGE_LOGGER or name.startswith(f"{PACKAGE_LOGGER}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER}.{name}")


class JsonFormatter(logging.Formatter):
    """One JSON object per line; ``extra=`` fields are included verbatim."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(
            (key, value)
            for key, value in vars(record).items()
            if key not in _STANDARD_ATTRS and not key.startswith("_")
        )
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY


def resolve_level(
    config: Config | None, explicit: str | int | None, environ: Mapping[str, str]
) -> int:
    if explicit is not None:
        return _coerce_level(explicit)
    debug_env = _cfg_str(config, "logging.debug_env")
    if debug_env and is_truthy(environ.get(debug_env)):
        return logging.DEBUG
    level_env = _cfg_str(config, "logging.level_env")
    if level_env and environ.get(level_env):
        return _coerce_level(environ[level_env])
    return _coerce_level(_cfg_str(config, "logging.level") or logging.getLevelName(logging.INFO))


def resolve_format(config: Config | None, explicit: str | None, environ: Mapping[str, str]) -> str:
    if explicit:
        return explicit
    format_env = _cfg_str(config, "logging.format_env")
    if format_env and environ.get(format_env):
        return environ[format_env]
    return _cfg_str(config, "logging.format") or "text"


def configure_logging(
    config: Config | None = None,
    *,
    level: str | int | None = None,
    fmt: str | None = None,
    stream: IO[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> logging.Logger:
    """Install (or replace) the package handler. Safe to call repeatedly."""
    env = os.environ if environ is None else environ
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    setattr(handler, _HANDLER_MARKER, True)
    chosen = resolve_format(config, fmt, env)
    if chosen == "json":
        handler.setFormatter(JsonFormatter())
    elif chosen == "text":
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    else:
        raise ConfigError(f"unknown log format {chosen!r}; expected 'text' or 'json'")
    logger.addHandler(handler)
    logger.setLevel(resolve_level(config, level, env))
    return logger


def _coerce_level(value: str | int) -> int:
    if isinstance(value, int):
        return value
    resolved = logging.getLevelName(value.strip().upper())
    if not isinstance(resolved, int):
        raise ConfigError(f"unknown log level {value!r}")
    return resolved


def _cfg_str(config: Config | None, dotted: str) -> str | None:
    if config is None:
        return None
    value = config.get(dotted, None)
    return value if isinstance(value, str) else None
