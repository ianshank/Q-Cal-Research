"""Layered configuration: packaged defaults < repository ``qcal.toml`` < environment.

Only the bootstrap constants below are fixed in code: the name of the repository
config file and the environment variables that locate it. Every other value the
package uses comes from :func:`load_config`.

The set of keys is closed: every key must exist in the packaged defaults (or sit in one of
the :data:`OPEN_TABLES`) with a compatible type. :func:`config_key_problems` checks it and
``qcal config --check`` reports it; registered runs refuse a configuration that fails it.

Environment overrides (``QCAL__SECTION__KEY``) and the locator variables are recorded in
:attr:`Config.environment_inputs`. They are a debugging aid: a registered run refuses them
(:func:`environment_problems`), except in :data:`RUN_NEUTRAL_SECTIONS`.

Stdlib-only: imported by Claude Code hooks.
"""

from __future__ import annotations

import copy
import hashlib
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
#: Sections whose environment overrides change nothing a run computes or records, so a
#: registered run accepts them (and records them). Everything else is refused.
RUN_NEUTRAL_SECTIONS: Final = frozenset({"logging"})
#: Tables whose keys are chosen by the repository rather than fixed by the defaults. Their
#: values must have the type of the packaged entries (when the defaults have any).
OPEN_TABLES: Final = frozenset(
    {"policy.categories", "policy.messages", "review.reviewer_by_branch_prefix"}
)
#: Element types of lists whose packaged default is empty (so the default cannot say).
EMPTY_LIST_ELEMENTS: Final[dict[str, type]] = {
    "executor.command": str,
    "licenses.dataset_allowed": str,
    "licenses.dataset_denied": str,
    "agent_layer.required_pretooluse_hooks": str,
}
#: Keys that existed once. A file that still sets one is told what replaced it.
#: Lists whose items must name a ``policy.categories`` table: a misspelled category would
#: otherwise match nothing, and the paths it was meant to protect would go unsigned.
CATEGORY_LISTS: Final = (
    "signing.signed_categories",
    "hooks.deny_categories",
    "claims.escape_hatch_categories",
)
REMOVED_KEYS: Final[dict[str, str]] = {
    "registry.schema_version": (
        "the run-record schema version is a property of the code that reads records "
        "(qcal.registry.records.SCHEMA_VERSION), not a setting; delete the key"
    ),
}

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
    #: Every environment variable that shaped this configuration, verbatim: the
    #: ``QCAL__*`` overrides, plus ``QCAL_ROOT``/``QCAL_CONFIG`` when they were set.
    environment_inputs: Mapping[str, str] = field(default_factory=dict)
    #: sha256 of the exact bytes the repository layer was parsed from (``None``: no layer).
    #: The registry compares this, not the file as it is later, with the committed blob.
    repo_sha256: str | None = None

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
    repo_sha256: str | None = None
    if repo_text is not None:
        repo_layer, label = _parse_toml(repo_text, "<repo_text>"), "repo_text"
        repo_sha256 = hashlib.sha256(repo_text.encode("utf-8")).hexdigest()
    elif use_repo_file:
        config_path = Path(env[CONFIG_PATH_ENV]) if env.get(CONFIG_PATH_ENV) else None
        config_path = config_path or resolved_root / REPO_CONFIG_NAME
        if config_path.is_file():
            raw = config_path.read_bytes()  # one read: what is parsed is what is hashed
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ConfigError(f"{config_path} is not UTF-8: {exc}") from exc
            repo_layer = _parse_toml(text, str(config_path))
            repo_sha256 = hashlib.sha256(raw).hexdigest()
            label = str(config_path)
    if repo_layer is not None:
        data = deep_merge(data, repo_layer)
        sources.append(label)

    env_layer = parse_env_overrides(env)
    if env_layer:
        data = deep_merge(data, env_layer)
        sources.append("environment")
    inputs = {
        name: value
        for name, value in env.items()
        if name.startswith(ENV_OVERRIDE_PREFIX) or (name in _LOCATOR_ENVS and value)
    }
    return Config(
        root=resolved_root,
        data=data,
        sources=tuple(sources),
        environment_inputs=dict(sorted(inputs.items())),
        repo_sha256=repo_sha256,
    )


_LOCATOR_ENVS: Final = (ROOT_ENV, CONFIG_PATH_ENV)


def _override_section(name: str) -> str:
    return name[len(ENV_OVERRIDE_PREFIX) :].split(ENV_OVERRIDE_SEPARATOR, 1)[0].lower()


def environment_problems(config: Config) -> list[str]:
    """Why the environment makes ``config`` unfit for a registered run; empty when it is fit.

    A registered run reads its configuration from committed files only. Overrides outside
    :data:`RUN_NEUTRAL_SECTIONS` are refused, and so are locator variables that point
    anywhere but the repository the run belongs to.
    """
    problems: list[str] = []
    for name, value in config.environment_inputs.items():
        if name == ROOT_ENV:
            if Path(value).resolve() != config.root:
                problems.append(f"{ROOT_ENV}={value} names another root than {config.root}")
        elif name == CONFIG_PATH_ENV:
            if Path(value).resolve() != (config.root / REPO_CONFIG_NAME).resolve():
                problems.append(f"{CONFIG_PATH_ENV}={value} replaces {REPO_CONFIG_NAME}")
        elif _override_section(name) not in RUN_NEUTRAL_SECTIONS:
            problems.append(f"{name} overrides configuration")
    return problems


def run_neutral_environment(config: Config) -> dict[str, str]:
    """The environment inputs a registered run accepts; they go into its record."""
    return {
        name: value
        for name, value in config.environment_inputs.items()
        if name.startswith(ENV_OVERRIDE_PREFIX) and _override_section(name) in RUN_NEUTRAL_SECTIONS
    }


def without_config_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` minus every variable that locates or overrides configuration."""
    return {
        name: value
        for name, value in environ.items()
        if not name.startswith(ENV_OVERRIDE_PREFIX) and name not in _LOCATOR_ENVS
    }


def config_key_problems(
    data: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> list[str]:
    """Unknown, removed and mistyped keys in ``data``, checked against the packaged defaults.

    A misspelled key would otherwise be ignored silently and its default would apply; for
    a policy key (``signing.signed_categories``) that quietly weakens the policy.
    """
    problems: list[str] = []
    _check_table(data, load_defaults() if defaults is None else defaults, "", problems)
    categories = data.get("policy", {}).get("categories", {})
    if isinstance(categories, Mapping):
        for dotted in CATEGORY_LISTS:
            node: Any = data
            for part in dotted.split("."):
                node = node.get(part) if isinstance(node, Mapping) else None
            unknown = [c for c in node or () if isinstance(c, str) and c not in categories]
            if unknown:
                problems.append(
                    f"{dotted} names categories that do not exist: {unknown} "
                    f"(known: {sorted(categories)})"
                )
    return problems


def _check_table(
    node: Mapping[str, Any], defaults: Mapping[str, Any], prefix: str, problems: list[str]
) -> None:
    open_table = prefix.rstrip(".") in OPEN_TABLES
    template = next(iter(defaults.values()), None) if open_table else None
    for key, value in node.items():
        dotted = f"{prefix}{key}"
        if dotted in REMOVED_KEYS:
            problems.append(f"{dotted} was removed: {REMOVED_KEYS[dotted]}")
        elif open_table:
            if template is not None:
                _check_value(value, template, dotted, problems)
        elif key not in defaults:
            problems.append(f"unknown configuration key {dotted!r}")
        else:
            _check_value(value, defaults[key], dotted, problems)


def _check_value(value: Any, default: Any, dotted: str, problems: list[str]) -> None:
    if not _same_kind(default, value):
        problems.append(f"{dotted} must be {_kind(default)}, got {_kind(value)}")
    elif isinstance(default, Mapping):
        _check_table(value, default, f"{dotted}.", problems)
    elif isinstance(default, list):
        element: Any = default[0] if default else EMPTY_LIST_ELEMENTS.get(dotted)
        if element is None:
            return
        bad = [v for v in value if not _same_kind(element, v)]
        if bad:
            problems.append(f"{dotted} must hold only {_kind(element)} items, got {bad[0]!r}")
        elif isinstance(element, Mapping):  # list of tables: each item has the template's keys
            for i, item in enumerate(value):
                unknown = sorted(set(item) - set(element))
                if unknown:
                    problems.append(f"{dotted}[{i}] has unknown keys {unknown}")


def _same_kind(expected: Any, value: Any) -> bool:
    """Same TOML kind: integers and floats are interchangeable; booleans are not numbers."""
    kind = expected if isinstance(expected, type) else type(expected)
    if kind is bool or isinstance(value, bool):
        return kind is bool and isinstance(value, bool)
    if kind in (int, float):
        return isinstance(value, int | float)
    if issubclass(kind, Mapping):
        return isinstance(value, Mapping)
    return isinstance(value, kind)


def _kind(value: Any) -> str:
    kind = value if isinstance(value, type) else type(value)
    if kind is bool:
        return "a boolean"
    if kind in (int, float):
        return "a number"
    if issubclass(kind, Mapping):
        return "a table"
    if kind is list:
        return "a list"
    return "a string" if kind is str else f"a {kind.__name__}"


def _parse_toml(text: str, origin: str) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"cannot parse {origin}: {exc}") from exc
