"""Ian's declarations: values only Ian sets (D2), read from a file agents cannot write.

The detector sanity check's expectation, and the attested label order of a checkpoint that
carries no class names, are Ian's decisions. A value an agent could write would declare
nothing, so they live in the file signed ``qcal.toml`` names (``paths.declarations``). It is
accepted only where ``qcal.toml``'s policy both keeps edit tools out of it
(``hooks.deny_categories``) and requires a signed commit for it (``signing.signed_categories``),
reached through no symbolic link, and committed exactly as read. A missing file declares
nothing, and a non-fixture detector then does not run (:func:`qcal_lab.sanity.require_declared`).

The file's layout, one table per detector; every value is Ian's, so the template is empty::

    [detectors.atss_r50]
    zoo_ap = ""             # the published value, in `units`
    tolerance = ""          # in `units`
    units = ""              # "fraction" (pycocotools) or "percent"
    source = ""             # where the zoo value is published
    checkpoint_sha256 = ""  # the checkpoint it was published for: must equal the configured one
    image_set = ""          # the images the zoo value was measured on
    test_cfg = {}           # its test-time settings: must equal the configured ones
    on_fail = ""            # "stop": a failed check stops for debugging, never swaps checkpoints
    label_order_attested_sha256 = ""  # only for a checkpoint without class names
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from qcal import gitutil
from qcal.config import Config
from qcal.log import get_logger
from qcal.policy import Policy
from qcal.registry.executor import sha256_file

_log = get_logger("lab.declarations")

SECTION: Final = "declarations"
DETECTORS: Final = "detectors"
#: ``paths.<PATH_KEY>`` in signed qcal.toml names the file.
PATH_KEY: Final = "declarations"
#: Categories whose files machines write (and some are untracked): never a declaration.
MACHINE_WRITTEN: Final = ("registry_only",)


class DeclarationsError(ValueError):
    """The declarations file is unprotected, unreadable or malformed."""


def declarations_path(qcal_config: Config, tooling: Mapping[str, Any] | None = None) -> Path:
    """The declarations file, refused unless only Ian's signed commits can change it.

    Its path is ``paths.declarations`` in signed ``qcal.toml``, so an agent cannot point it
    elsewhere. The path must be inside the repository with no symbolic link in it (a link
    would read bytes no signed commit chose), in a category ``qcal.toml`` both denies to edit
    tools and signs, and not a registry-only one (those are machine-written and some are
    untracked). ``tooling`` is accepted for compatibility and ignored.
    """
    del tooling
    relative = qcal_config.str_value(f"paths.{PATH_KEY}")
    if not relative.strip() or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise DeclarationsError(
            f"paths.{PATH_KEY} must be a path inside the repository, got {relative!r}"
        )
    shown = Path(relative).as_posix()
    categories = set(Policy.from_config(qcal_config).categories_for(shown))
    denied = categories & set(qcal_config.str_list("hooks.deny_categories"))
    signed = categories & set(qcal_config.str_list("signing.signed_categories"))
    if not (denied & signed) - set(MACHINE_WRITTEN):
        raise DeclarationsError(
            f"{shown} is not a file agents are kept out of and only signed commits change "
            f"(its categories: {sorted(categories) or 'none'}); Ian's declarations must be"
        )
    root = qcal_config.root.resolve()
    path = root / relative
    if os.path.realpath(path) != os.path.normpath(path):
        raise DeclarationsError(f"{shown} goes through a symbolic link; declarations may not")
    return path


def committed_problem(qcal_config: Config, path: Path) -> str | None:
    """Why the declarations file is not exactly what ``HEAD`` holds, or ``None``.

    Outside a git repository nothing can be compared (registered runs refuse that already).
    """
    root = qcal_config.root.resolve()
    if gitutil.git_marker(root) is None:
        return None
    relative = path.relative_to(root).as_posix()
    committed = gitutil.blob_sha256("HEAD", relative, root)
    if committed is None:
        return f"{relative} is not committed: a declaration is the signed commit that adds it"
    if committed != sha256_file(path):
        return f"{relative} differs from HEAD: commit it (signed) or restore it"
    return None


def load_declarations(
    qcal_config: Config, tooling: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Everything Ian declared; ``{}`` while the file does not exist."""
    path = declarations_path(qcal_config, tooling)
    try:
        text = path.read_text("utf-8")
    except FileNotFoundError:
        _log.debug("no declarations at %s yet", path)
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        raise DeclarationsError(f"cannot read {path}: {exc}") from exc
    problem = committed_problem(qcal_config, path)
    if problem:
        raise DeclarationsError(problem)
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise DeclarationsError(f"{path} is not valid TOML: {exc}") from exc


def detector_declarations(
    qcal_config: Config, name: str, tooling: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """What Ian declared for detector ``name`` (``[detectors.<name>]``); ``{}`` if nothing."""
    detectors = load_declarations(qcal_config, tooling).get(DETECTORS, {})
    declared = detectors.get(name, {}) if isinstance(detectors, Mapping) else {}
    if not isinstance(declared, Mapping):
        raise DeclarationsError(f"{DETECTORS}.{name} must be a table")
    return declared


__all__ = [
    "DETECTORS",
    "MACHINE_WRITTEN",
    "PATH_KEY",
    "SECTION",
    "DeclarationsError",
    "committed_problem",
    "declarations_path",
    "detector_declarations",
    "load_declarations",
]
