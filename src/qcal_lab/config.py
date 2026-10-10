"""Layered configuration for the science code: packaged defaults < ``configs/lab.toml``.

Science settings come only from files. The registry hashes ``configs/**`` into every run's
``config_hash`` (``registry.config_hash_inputs``), so a run's settings are recoverable from
its record; an environment variable would not be. :func:`require_hashed` refuses a lab
configuration file the registry would not hash.
"""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Final

from qcal.config import Config, ConfigError, deep_merge
from qcal.globs import first_match

# Bootstrap constant, like qcal's REPO_CONFIG_NAME: where the repository layer lives.
LAB_CONFIG_FILE: Final = "configs/lab.toml"
_DEFAULTS_PACKAGE: Final = "qcal_lab.resources"
_DEFAULTS_RESOURCE: Final = "defaults.toml"


def load_lab_defaults() -> dict[str, Any]:
    """A fresh copy of the packaged defaults."""
    text = resources.files(_DEFAULTS_PACKAGE).joinpath(_DEFAULTS_RESOURCE).read_text("utf-8")
    return tomllib.loads(text)


@dataclass(frozen=True)
class LabConfig:
    """The merged science configuration and the file it came from."""

    config: Config
    path: Path | None = None  # the repository layer, when one exists
    sha256: str = ""  # of the repository layer's bytes

    def text(self, dotted: str) -> str:
        """A required, non-empty string setting."""
        value = self.config.str_value(dotted)
        if not value.strip():
            raise ConfigError(f"{dotted} is not set; set it in {LAB_CONFIG_FILE}")
        return value

    def file(self, dotted: str) -> Path:
        """A required path setting, resolved against the repository root."""
        given = Path(self.text(dotted))
        return given if given.is_absolute() else self.config.root / given

    def table(self, dotted: str) -> Mapping[str, Any]:
        return self.config.section(dotted)


def parse_lab_config(root: Path, text: str | None, *, origin: str = "<text>") -> LabConfig:
    """Merge ``text`` (the repository layer, if any) over the packaged defaults."""
    data = load_lab_defaults()
    sources = ["qcal_lab defaults"]
    digest = ""
    if text is not None:
        try:
            layer = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"cannot parse {origin}: {exc}") from exc
        data = deep_merge(data, layer)
        sources.append(origin)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return LabConfig(Config(root=root.resolve(), data=data, sources=tuple(sources)), None, digest)


def load_lab_config(root: Path, path: Path | None = None) -> LabConfig:
    """Load ``configs/lab.toml`` (or ``path``) over the packaged defaults."""
    target = path or root / LAB_CONFIG_FILE
    if not target.is_file():
        return parse_lab_config(root, None)
    raw = target.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{target} is not UTF-8: {exc}") from exc
    parsed = parse_lab_config(root, text, origin=str(target))
    return LabConfig(parsed.config, target, hashlib.sha256(raw).hexdigest())


def require_hashed(lab: LabConfig, qcal_config: Config) -> None:
    """Refuse a lab configuration file that the registry's config hash would not cover."""
    if lab.path is None:
        return
    resolved = lab.path.resolve()
    if not resolved.is_relative_to(qcal_config.root):
        raise ConfigError(f"{lab.path} is outside the repository, so no run record can hash it")
    relative = resolved.relative_to(qcal_config.root).as_posix()
    if first_match(relative, qcal_config.str_list("registry.config_hash_inputs")) is None:
        raise ConfigError(
            f"{relative} is not covered by registry.config_hash_inputs, so a run's science "
            "settings would not be recorded; move it under configs/"
        )


__all__ = [
    "LAB_CONFIG_FILE",
    "LabConfig",
    "load_lab_config",
    "load_lab_defaults",
    "parse_lab_config",
    "require_hashed",
]
