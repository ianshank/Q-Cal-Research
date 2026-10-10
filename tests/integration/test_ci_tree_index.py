"""The cached tree lookups behind the net-content rule agree with ``git rev-parse``."""

from __future__ import annotations

from pathlib import Path

import pytest

from qcal.ci.signatures import TreeIndex, blob_id, tree_entries
from tests.conftest import run_git, write

pytestmark = pytest.mark.integration

ODD_NAMES = ["plain.txt", "dir/nested.md", "tab\there.txt", "new\nline.txt", "ünïcode.md"]


@pytest.fixture
def odd_repo(git_repo: Path) -> tuple[Path, str, str]:
    first = run_git(git_repo, "rev-parse", "HEAD")
    for name in ODD_NAMES:
        write(git_repo, name, f"{name}\n")
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "odd names")
    return git_repo, first, run_git(git_repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("name", ODD_NAMES)
def test_tree_index_matches_rev_parse_for_unusual_names(odd_repo, name: str) -> None:
    repo, _first, head = odd_repo
    assert TreeIndex(repo).blob(head, name) == blob_id(repo, head, name)
    assert TreeIndex(repo).blob(head, name) is not None


def test_missing_paths_and_directories_are_none(odd_repo) -> None:
    repo, first, head = odd_repo
    index = TreeIndex(repo)
    assert index.blob(first, "plain.txt") is None  # added later
    assert index.blob(head, "does/not/exist") is None
    assert index.blob(head, "dir") is None  # trees are not blobs


def test_each_ref_is_listed_once(odd_repo) -> None:
    repo, first, head = odd_repo
    index = TreeIndex(repo)
    for _ in range(3):
        for name in ODD_NAMES:
            index.blob(head, name)
            index.blob(first, name)
    assert index.refs_loaded == 2


def test_tree_entries_lists_every_blob(odd_repo) -> None:
    repo, _first, head = odd_repo
    entries = tree_entries(repo, head)
    assert set(ODD_NAMES) <= set(entries)
    assert all(len(oid) >= 40 for oid in entries.values())


def test_blob_id_of_a_missing_path_is_none(odd_repo) -> None:
    repo, _first, head = odd_repo
    assert blob_id(repo, head, "no/such/file") is None
