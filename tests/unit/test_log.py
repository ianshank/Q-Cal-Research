from __future__ import annotations

import io
import json
import logging

import pytest

from qcal.config import ConfigError
from qcal.log import (
    PACKAGE_LOGGER,
    JsonFormatter,
    configure_logging,
    get_logger,
    is_truthy,
    resolve_format,
    resolve_level,
)


@pytest.fixture(autouse=True)
def _reset_package_logger():
    logger = logging.getLogger(PACKAGE_LOGGER)
    before = list(logger.handlers), logger.level
    yield
    logger.handlers[:] = before[0]
    logger.setLevel(before[1])


@pytest.mark.parametrize(
    ("name", "expected"), [("x", "qcal.x"), ("qcal", "qcal"), ("qcal.y", "qcal.y")]
)
def test_get_logger_namespaces_under_the_package(name: str, expected: str) -> None:
    assert get_logger(name).name == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("TRUE", True),
        (" yes ", True),
        ("on", True),
        ("0", False),
        ("", False),
        (None, False),
    ],
)
def test_is_truthy(value: str | None, expected: bool) -> None:
    assert is_truthy(value) is expected


def test_level_resolution_order(config) -> None:
    assert resolve_level(config, "ERROR", {"QCAL_DEBUG": "1"}) == logging.ERROR
    assert (
        resolve_level(config, None, {"QCAL_DEBUG": "1", "QCAL_LOG_LEVEL": "ERROR"}) == logging.DEBUG
    )
    assert resolve_level(config, None, {"QCAL_LOG_LEVEL": "warning"}) == logging.WARNING
    assert resolve_level(config, None, {}) == logging.INFO
    assert resolve_level(None, None, {}) == logging.INFO
    assert resolve_level(config, logging.CRITICAL, {}) == logging.CRITICAL


def test_format_resolution_order(config) -> None:
    assert resolve_format(config, "json", {"QCAL_LOG_FORMAT": "text"}) == "json"
    assert resolve_format(config, None, {"QCAL_LOG_FORMAT": "json"}) == "json"
    assert resolve_format(config, None, {}) == "text"
    assert resolve_format(None, None, {}) == "text"


def test_unknown_level_and_format_are_errors(config) -> None:
    with pytest.raises(ConfigError, match="log level"):
        resolve_level(config, "LOUD", {})
    with pytest.raises(ConfigError, match="log format"):
        configure_logging(config, fmt="xml", stream=io.StringIO(), environ={})


def test_configure_is_idempotent_and_writes_text(config) -> None:
    stream = io.StringIO()
    configure_logging(config, stream=stream, environ={})
    logger = configure_logging(config, stream=stream, environ={})
    marked = [h for h in logger.handlers if getattr(h, "_qcal_handler", False)]
    assert len(marked) == 1
    get_logger("t").info("hello %s", "world")
    assert "INFO" in stream.getvalue()
    assert "qcal.t: hello world" in stream.getvalue()


def test_json_format_includes_extra_fields_and_exceptions(config) -> None:
    stream = io.StringIO()
    configure_logging(config, fmt="json", stream=stream, environ={})
    log = get_logger("j")
    log.warning("event", extra={"run_id": "R1"})
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("failed")
    first, second = (json.loads(line) for line in stream.getvalue().splitlines())
    assert first["msg"] == "event"
    assert first["run_id"] == "R1"
    assert first["level"] == "WARNING"
    assert "ValueError: boom" in second["exc"]


def test_json_formatter_is_stable_json() -> None:
    record = logging.makeLogRecord({"msg": "m", "levelname": "INFO", "name": "qcal.x"})
    assert json.loads(JsonFormatter().format(record))["logger"] == "qcal.x"
