"""Ian's declarations (review B1; panel round 2, S6a): read only from the file signed
``qcal.toml`` names, which agents cannot write, through no link, committed as read."""

from __future__ import annotations

from pathlib import Path

import pytest

from qcal.config import Config, load_config
from qcal_lab import declarations
from qcal_lab.declarations import (
    DeclarationsError,
    declarations_path,
    detector_declarations,
    load_declarations,
    read_declarations,
)
from tests.conftest import REPO_ROOT, run_git, write

pytestmark = pytest.mark.rule("C0", "C4")

DECLARED = '[detectors.atss_r50]\nzoo_ap = 0.5\ntolerance = 0.01\nsource = "card"\n'
FILE = "docs/reference/declarations.toml"


def _project(root: Path, declarations: str | None = None) -> Config:
    """A project with the repository's policy, optionally naming another declarations file."""
    text = (REPO_ROOT / "qcal.toml").read_text("utf-8")
    if declarations is not None:
        text = text.replace(f'declarations = "{FILE}"', f'declarations = "{declarations}"')
    write(root, "qcal.toml", text)
    return load_config(root, environ={})


@pytest.fixture
def project(tmp_path: Path) -> Config:
    return _project(tmp_path)


def test_the_configured_file_is_ians_data_and_signed() -> None:
    config = load_config(REPO_ROOT, environ={})
    assert config.str_value("paths.declarations") == FILE  # signed qcal.toml names it
    assert declarations_path(config) == REPO_ROOT.resolve() / FILE


@pytest.mark.parametrize(
    "file",
    [
        "configs/declarations.toml",  # agents maintain configs/
        "src/qcal_lab/resources/declarations.toml",  # agent-owned package
        "runs/cache/declarations.toml",  # denied and signed, but machine-written and untracked
        "",
        "/etc/declarations.toml",
        "docs/reference/../../configs/x.toml",
    ],
)
def test_a_file_agents_could_write_is_refused(tmp_path: Path, file: str) -> None:
    with pytest.raises(DeclarationsError):
        declarations_path(_project(tmp_path, file))


def test_the_packaged_default_declares_nothing(tmp_path: Path) -> None:
    config = load_config(tmp_path, environ={})  # no qcal.toml: paths.declarations is ""
    with pytest.raises(DeclarationsError, match=r"paths\.declarations must be a path"):
        declarations_path(config)


@pytest.mark.parametrize("link", ["docs/reference", f"{FILE}"])
def test_a_path_through_a_symlink_is_refused(project: Config, link: str) -> None:
    """A link would make the protected name read an agent-writable file (panel round 2)."""
    write(project.root, "notes/real.toml", DECLARED)
    target = project.root / link
    target.parent.mkdir(parents=True, exist_ok=True)
    source = project.root / ("notes" if link == "docs/reference" else "notes/real.toml")
    if link == "docs/reference":
        (project.root / "notes/declarations.toml").write_text(DECLARED)
    target.symlink_to(source)
    with pytest.raises(DeclarationsError, match="symbolic link"):
        load_declarations(project)


def test_a_missing_file_declares_nothing(project: Config) -> None:
    assert load_declarations(project) == {}
    assert detector_declarations(project, "atss_r50") == {}


def test_declarations_are_read_per_detector(project: Config) -> None:
    write(project.root, FILE, DECLARED)
    declared = detector_declarations(project, "atss_r50")
    assert dict(declared) == {"zoo_ap": 0.5, "tolerance": 0.01, "source": "card"}
    assert detector_declarations(project, "other") == {}


@pytest.fixture
def committed(project: Config) -> Config:
    """The project in git, with the declarations committed."""
    write(project.root, FILE, DECLARED)
    run_git(project.root, "init", "-q")
    run_git(project.root, "add", "-A")
    run_git(project.root, "commit", "-q", "-m", "declare")
    return project


def test_committed_declarations_are_read(committed: Config) -> None:
    assert dict(detector_declarations(committed, "atss_r50"))["source"] == "card"


def test_declarations_that_differ_from_head_are_refused(committed: Config) -> None:
    write(committed.root, FILE, DECLARED.replace("0.5", "0.6"))  # edited, not committed
    with pytest.raises(DeclarationsError, match="differs from HEAD"):
        load_declarations(committed)


def test_uncommitted_declarations_are_refused(committed: Config) -> None:
    run_git(committed.root, "rm", "-q", "--cached", FILE)
    run_git(committed.root, "commit", "-q", "-m", "untrack")
    with pytest.raises(DeclarationsError, match="is not committed"):
        load_declarations(committed)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[detectors]\natss_r50 = 1\n", "must be a table"),
        ("not toml =", "not valid TOML"),
    ],
)
def test_a_malformed_file_is_refused(project: Config, text: str, message: str) -> None:
    write(project.root, FILE, text)
    with pytest.raises(DeclarationsError, match=message):
        detector_declarations(project, "atss_r50")


def test_an_unreadable_file_is_refused(project: Config) -> None:
    (project.root / FILE).mkdir(parents=True)  # a directory
    with pytest.raises(DeclarationsError, match="cannot read"):
        load_declarations(project)


def test_one_read_serves_the_check_the_parse_and_the_digest(
    committed: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Panel round 3: the bytes compared with HEAD are the bytes parsed and recorded, even if
    the file is replaced between the steps."""
    import hashlib

    path = committed.root / FILE
    original = path.read_bytes()
    real_read = declarations._read_once

    def swap_after_reading(where: Path) -> bytes | None:
        data = real_read(where)
        path.write_text(DECLARED.replace('"card"', '"forged"'))  # replaced right after the read
        return data

    monkeypatch.setattr(declarations, "_read_once", swap_after_reading)
    declared, _, digest = read_declarations(committed)
    assert declared["detectors"]["atss_r50"]["source"] == "card"
    assert digest == hashlib.sha256(original).hexdigest()


def test_a_link_swapped_in_after_the_path_check_is_not_followed(project: Config) -> None:
    write(project.root, "notes/forged.toml", DECLARED)
    link = project.root / FILE
    link.parent.mkdir(parents=True)
    link.symlink_to(project.root / "notes/forged.toml")
    with pytest.raises(DeclarationsError, match="cannot read"):
        declarations._read_once(link)  # what read_declarations opens after the check
