"""Materialize Ian's document templates without ever overwriting existing files."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

from qcal.config import Config
from qcal.log import get_logger

_log = get_logger("init")


@dataclass(frozen=True)
class InitAction:
    path: str
    action: str  # created | exists | would-create | would-overwrite | overwritten


def init_documents(
    config: Config, *, force: bool = False, dry_run: bool = False
) -> list[InitAction]:
    templates = resources.files("qcal.resources").joinpath("templates")
    actions: list[InitAction] = []
    for key in config.str_list("init.files"):
        target = config.path(key)
        template = templates.joinpath(target.name)
        if not template.is_file():
            raise FileNotFoundError(f"no packaged template named {target.name} for paths.{key}")
        relative = target.relative_to(config.root).as_posix()
        if target.exists() and not force:
            actions.append(InitAction(relative, "exists"))
            continue
        if dry_run:
            planned = "would-overwrite" if target.exists() else "would-create"
            actions.append(InitAction(relative, planned))
            continue
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(template.read_text("utf-8"), encoding="utf-8")
        actions.append(InitAction(relative, "overwritten" if existed else "created"))
        _log.info("%s %s", "overwrote" if existed else "created", relative)
    return actions
