"""Layered configuration: packaged defaults < repository ``qcal.toml`` < environment.

Only the bootstrap constants below are fixed in code: the name of the repository
config file and the environment variables that locate it. Every other value the
package uses comes from :func:`load_config`.

Stdlib-only: imported by Claude Code hooks.
"""

from __future__ import annotations

import copy
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Final

REPO_CONFIG_NAME: Final = "qcal.toml"
ROOT_ENV: Final = "QCAL_ROOT"
CONFIG_PATH_ENV: Final = "QCAL_CONFIG"
ENV_OVERRIDE_PREFIX: Final = "QCAL__"
ENV_OVERRIDE_SEPARATOR: Final = "__"
ROOT_MARKERS: Final = (REPO_CONFIG_NAME, ".git")
_DEFAULTS_RESOURCE: Final = "defaults.toml"

_MISSING: Final = object()


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or has the wrong type."""


def load_defaults() -> dict[str, Any]:
    """Return a fresh copy of the packaged defaults."""
    text = resources.files("qcal.resources").joinpath(_DEFAULTS_RESOURCE).read_text("utf-8")
    return tomllib.loads(text)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into a copy of ``base``; tables merge, everything else replaces."""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_env_value(raw: str) -> Any:
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def parse_env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    """Turn ``QCAL__SECTION__KEY=value`` variables into a nested override table."""
    overrides: dict[str, Any] = {}
    for name, raw in sorted(environ.items()):
        if not name.startswith(ENV_OVERRIDE_PREFIX):
            continue
        parts = [p.lower() for p in name[len(ENV_OVERRIDE_PREFIX) :].split(ENV_OVERRIDE_SEPARATOR)]
        if not parts or any(not p for p in parts):
            raise ConfigError(f"malformed override variable {name!r}")
        node = overrides
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"override {name!r} conflicts with a scalar override")
            node = child
        node[parts[-1]] = _parse_env_value(raw)
    return overrides


def find_root(start: Path | None = None, environ: Mapping[str, str] | None = None) -> Path:
    """Locate the repository root: ``$QCAL_ROOT``, else the nearest ancestor with a marker."""
    env = os.environ if environ is None else environ
    if env.get(ROOT_ENV):
        return Path(env[ROOT_ENV]).resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if any((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate
    return here


@dataclass(frozen=True)
class Config:
    """Merged configuration with typed accessors that name the offending key on error."""

    root: Path
    data: Mapping[str, Any]
    sources: tuple[str, ...] = field(default=())

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"missing configuration key {dotted!r}")
                return default
            node = node[part]
        return node

    def section(self, dotted: str) -> Mapping[str, Any]:
        value = self.get(dotted)
        if not isinstance(value, Mapping):
            raise ConfigError(f"configuration key {dotted!r} must be a table")
        return value

    def str_value(self, dotted: str) -> str:
        value = self.get(dotted)
        if not isinstance(value, str):
            raise ConfigError(f"configuration key {dotted!r} must be a string")
        return value

    def int_value(self, dotted: str) -> int:
        value = self.get(dotted)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"configuration key {dotted!r} must be an integer")
        return value

    def float_value(self, dotted: str) -> float:
        value = self.get(dotted)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"configuration key {dotted!r} must be a number")
        return float(value)

    def bool_value(self, dotted: str) -> bool:
        value = self.get(dotted)
        if not isinstance(value, bool):
            raise ConfigError(f"configuration key {dotted!r} must be a boolean")
        return value

    def str_list(self, dotted: str) -> list[str]:
        value = self.get(dotted)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"configuration key {dotted!r} must be a list of strings")
        return list(value)

    def table_list(self, dotted: str) -> list[Mapping[str, Any]]:
        value = self.get(dotted)
        if not isinstance(value, list) or not all(isinstance(v, Mapping) for v in value):
            raise ConfigError(f"configuration key {dotted!r} must be a list of tables")
        return list(value)

    def path(self, key: str) -> Path:
        """Resolve ``paths.<key>`` against the repository root."""
        return self.root / self.str_value(f"paths.{key}")


def load_config(
    root: Path | None = None,
    *,
    repo_text: str | None = None,
    environ: Mapping[str, str] | None = None,
    use_repo_file: bool = True,
) -> Config:
    """Build the merged configuration.

    ``repo_text`` supplies the repository layer directly; CI uses it to read the
    policy from the *base* commit rather than from a pull request's head.
    """
    env = os.environ if environ is None else environ
    resolved_root = (root or find_root(environ=env)).resolve()
    data = load_defaults()
    sources = ["defaults"]

    repo_layer: dict[str, Any] | None = None
    if repo_text is not None:
        repo_layer, label = _parse_toml(repo_text, "<repo_text>"), "repo_text"
    elif use_repo_file:
        config_path = Path(env[CONFIG_PATH_ENV]) if env.get(CONFIG_PATH_ENV) else None
        config_path = config_path or resolved_root / REPO_CONFIG_NAME
        if config_path.is_file():
            repo_layer = _parse_toml(config_path.read_text("utf-8"), str(config_path))
            label = str(config_path)
    if repo_layer is not None:
        data = deep_merge(data, repo_layer)
        sources.append(label)

    env_layer = parse_env_overrides(env)
    if env_layer:
        data = deep_merge(data, env_layer)
        sources.append("environment")
    return Config(root=resolved_root, data=data, sources=tuple(sources))


def _parse_toml(text: str, origin: str) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"cannot parse {origin}: {exc}") from exc
