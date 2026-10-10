"""`qcal init`: packaged templates are materialized without overwriting existing files."""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources
from pathlib import Path

import pytest

from qcal.config import Config, ConfigError
from qcal.initdocs import InitAction, init_documents
from tests.conftest import write

DOCUMENTS = ["EXPERIMENTS.yaml", "DECISIONS.md", "AMENDMENTS.md", "CLAIMS.md", "RESEARCH_LOG.md"]


def template(name: str) -> str:
    return resources.files("qcal.resources").joinpath("templates", name).read_text("utf-8")


def actions_of(actions: list[InitAction]) -> dict[str, str]:
    return {a.path: a.action for a in actions}


def test_every_configured_document_is_created_in_config_order(config: Config) -> None:
    actions = init_documents(config)

    assert actions == [InitAction(name, "created") for name in DOCUMENTS]


@pytest.mark.parametrize("name", DOCUMENTS)
def test_created_document_holds_the_packaged_template(config: Config, name: str) -> None:
    init_documents(config)

    assert (config.root / name).read_text("utf-8") == template(name)


def test_existing_document_is_never_overwritten(config: Config, repo: Path) -> None:
    write(repo, "CLAIMS.md", "Ian's claims\n")

    actions = init_documents(config)

    assert actions_of(actions)["CLAIMS.md"] == "exists"
    assert (repo / "CLAIMS.md").read_text("utf-8") == "Ian's claims\n"
    assert actions_of(actions)["DECISIONS.md"] == "created"


def test_second_run_finds_every_document_existing(config: Config) -> None:
    init_documents(config)

    actions = init_documents(config)

    assert {a.action for a in actions} == {"exists"}


def test_force_overwrites_existing_documents_with_the_template(config: Config, repo: Path) -> None:
    write(repo, "CLAIMS.md", "stale\n")

    actions = init_documents(config, force=True)

    assert actions_of(actions)["CLAIMS.md"] == "overwritten"
    assert actions_of(actions)["DECISIONS.md"] == "created"
    assert (repo / "CLAIMS.md").read_text("utf-8") == template("CLAIMS.md")


def test_dry_run_writes_nothing(config: Config, repo: Path) -> None:
    actions = init_documents(config, dry_run=True)

    assert actions == [InitAction(name, "would-create") for name in DOCUMENTS]
    assert not any((repo / name).exists() for name in DOCUMENTS)


def test_dry_run_still_reports_existing_documents(config: Config, repo: Path) -> None:
    write(repo, "DECISIONS.md", "mine\n")

    actions = init_documents(config, dry_run=True)

    assert actions_of(actions)["DECISIONS.md"] == "exists"


def test_dry_run_with_force_does_not_touch_existing_documents(config: Config, repo: Path) -> None:
    write(repo, "DECISIONS.md", "mine\n")

    actions = init_documents(config, force=True, dry_run=True)

    assert actions_of(actions)["DECISIONS.md"] != "overwritten"
    assert (repo / "DECISIONS.md").read_text("utf-8") == "mine\n"


def test_document_paths_follow_configuration_and_parents_are_created(
    make_config: Callable[[str], Config],
) -> None:
    config = make_config(
        '[paths]\nclaims = "docs/research/CLAIMS.md"\n[init]\nfiles = ["claims"]\n'
    )

    actions = init_documents(config)

    assert actions == [InitAction("docs/research/CLAIMS.md", "created")]
    assert (config.root / "docs/research/CLAIMS.md").read_text("utf-8") == template("CLAIMS.md")


def test_only_configured_documents_are_created(make_config: Callable[[str], Config]) -> None:
    config = make_config('[init]\nfiles = ["decisions"]\n')

    actions = init_documents(config)

    assert actions == [InitAction("DECISIONS.md", "created")]
    assert not (config.root / "CLAIMS.md").exists()


def test_path_without_a_packaged_template_is_an_error(
    make_config: Callable[[str], Config],
) -> None:
    config = make_config('[paths]\nclaims = "claims.md"\n[init]\nfiles = ["claims"]\n')

    with pytest.raises(FileNotFoundError, match=r"no packaged template named claims\.md"):
        init_documents(config)


def test_unknown_path_key_is_a_config_error(make_config: Callable[[str], Config]) -> None:
    config = make_config('[init]\nfiles = ["nope"]\n')

    with pytest.raises(ConfigError, match=r"paths\.nope"):
        init_documents(config)
