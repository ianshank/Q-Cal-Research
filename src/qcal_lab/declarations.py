"""Ian's declarations: values only Ian sets (D2), read from a file agents cannot write.

The detector sanity check's expectation, and the attested label order of a checkpoint that
carries no class names, are Ian's decisions. A value an agent could write would declare
nothing, so they live in ``[declarations] file`` (``tooling.toml``), and that file is accepted
only where ``qcal.toml``'s policy both keeps edit tools out of it (``hooks.deny_categories``)
and requires a signed commit for it (``signing.signed_categories``). A missing file declares
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

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from qcal.config import Config
from qcal.log import get_logger
from qcal.policy import Policy
from qcal_lab.config import load_tooling

_log = get_logger("lab.declarations")

SECTION: Final = "declarations"
DETECTORS: Final = "detectors"


class DeclarationsError(ValueError):
    """The declarations file is unprotected, unreadable or malformed."""


def declarations_path(qcal_config: Config, tooling: Mapping[str, Any] | None = None) -> Path:
    """The declarations file, refused unless only Ian's signed commits can change it."""
    settings = (tooling if tooling is not None else load_tooling()).get(SECTION, {})
    relative = str(settings.get("file", "")) if isinstance(settings, Mapping) else ""
    if not relative.strip() or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise DeclarationsError(
            f"{SECTION}.file must be a path inside the repository, got {relative!r}"
        )
    categories = set(Policy.from_config(qcal_config).categories_for(Path(relative).as_posix()))
    denied = categories & set(qcal_config.str_list("hooks.deny_categories"))
    signed = categories & set(qcal_config.str_list("signing.signed_categories"))
    if not (denied & signed):
        raise DeclarationsError(
            f"{relative} is not a file agents are kept out of and only signed commits change "
            f"(its categories: {sorted(categories) or 'none'}); Ian's declarations must be"
        )
    return qcal_config.root / relative


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
    "SECTION",
    "DeclarationsError",
    "declarations_path",
    "detector_declarations",
    "load_declarations",
]
