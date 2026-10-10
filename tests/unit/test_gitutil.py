"""Thin git wrappers: errors, graceful degradation, HEAD, dirtiness and file contents."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qcal import gitutil
from qcal.gitutil import GitError, git, head_sha, is_dirty, show_file, try_git
from tests.conftest import run_git, write


@pytest.fixture(autouse=True)
def _no_parent_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never discover a repository above tmp_path, so "outside a repo" really is."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A directory that is not inside any git repository."""
    path = tmp_path / "plain"
    path.mkdir()
    return path


def commit_all(root: Path, message: str) -> str:
    run_git(root, "add", "-A")
    run_git(root, "commit", "-q", "-m", message)
    return run_git(root, "rev-parse", "HEAD")


# --- git / GitError -------------------------------------------------------------------------


def test_git_returns_raw_stdout(git_repo: Path) -> None:
    assert git(["rev-parse", "--is-inside-work-tree"], git_repo) == "true\n"


def test_git_raises_git_error_with_returncode_and_stderr(git_repo: Path) -> None:
    with pytest.raises(GitError, match=r"^git show HEAD:nope failed \(128\): fatal: ") as caught:
        git(["show", "HEAD:nope"], git_repo)

    assert caught.value.returncode == 128
    assert "nope" in caught.value.stderr


def test_git_without_check_returns_output_of_a_failing_command(git_repo: Path) -> None:
    output = git(["ls-files", "--error-unmatch", "qcal.toml", "nope"], git_repo, check=False)

    assert output == "qcal.toml\n"


def test_git_error_message_strips_stderr() -> None:
    error = GitError(["status"], 1, "  boom\n")

    assert str(error) == "git status failed (1): boom"
    assert error.stderr == "  boom\n"


def test_git_passes_timeout_and_argv_to_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(argv=argv, **kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(gitutil.subprocess, "run", fake_run)

    assert git(["log"], tmp_path, timeout=5) == "ok"
    assert seen["argv"] == ["git", "log"]
    assert (seen["cwd"], seen["timeout"], seen["check"]) == (tmp_path, 5, False)


# --- try_git ------------------------------------------------------------------------------


def test_try_git_returns_stripped_stdout(git_repo: Path) -> None:
    assert try_git(["rev-parse", "--abbrev-ref", "HEAD"], git_repo) == "main"


def test_try_git_returns_none_when_git_fails(outside: Path) -> None:
    assert try_git(["rev-parse", "HEAD"], outside) is None


def test_try_git_returns_none_when_cwd_does_not_exist(tmp_path: Path) -> None:
    assert try_git(["status"], tmp_path / "missing") is None


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(["git"], 60),
        FileNotFoundError("git"),
    ],
    ids=["timeout", "git-not-installed"],
)
def test_try_git_returns_none_when_git_cannot_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception
) -> None:
    def fake_run(*_: Any, **__: Any) -> None:
        raise error

    monkeypatch.setattr(gitutil.subprocess, "run", fake_run)

    assert try_git(["status"], tmp_path) is None


# --- head_sha -----------------------------------------------------------------------------


def test_head_sha_is_the_current_commit(git_repo: Path) -> None:
    sha = head_sha(git_repo)

    assert sha == run_git(git_repo, "rev-parse", "HEAD")
    assert sha is not None
    assert len(sha) == 40


def test_head_sha_follows_new_commits(git_repo: Path) -> None:
    write(git_repo, "a.txt", "a\n")
    new = commit_all(git_repo, "second")

    assert head_sha(git_repo) == new


def test_head_sha_outside_a_repository_is_none(outside: Path) -> None:
    assert head_sha(outside) is None


def test_head_sha_of_a_repository_without_commits_is_none(outside: Path) -> None:
    run_git(outside, "init", "-q")

    assert head_sha(outside) is None


# --- is_dirty -----------------------------------------------------------------------------


def test_clean_work_tree_is_not_dirty(git_repo: Path) -> None:
    assert is_dirty(git_repo) is False


def _untracked(root: Path) -> None:
    write(root, "new.txt", "x\n")


def _modified(root: Path) -> None:
    write(root, "qcal.toml", "# changed\n")


def _staged(root: Path) -> None:
    write(root, "new.txt", "x\n")
    run_git(root, "add", "new.txt")


def _deleted(root: Path) -> None:
    (root / "qcal.toml").unlink()


@pytest.mark.parametrize(
    "change", [_untracked, _modified, _staged, _deleted], ids=lambda f: f.__name__.strip("_")
)
def test_any_change_makes_the_work_tree_dirty(
    git_repo: Path, change: Callable[[Path], None]
) -> None:
    change(git_repo)

    assert is_dirty(git_repo) is True


def test_is_dirty_outside_a_repository_is_none(outside: Path) -> None:
    assert is_dirty(outside) is None


def test_excluded_paths_do_not_make_the_tree_dirty(git_repo: Path) -> None:
    write(git_repo, "runs/index.csv", "run_id\n")
    write(git_repo, "runs/registry/R1.json", "{}\n")

    assert is_dirty(git_repo, exclude=["runs/index.csv", "runs/registry"]) is False


def test_changes_outside_the_exclusions_still_count(git_repo: Path) -> None:
    write(git_repo, "runs/index.csv", "run_id\n")
    write(git_repo, "src/model.py", "x = 1\n")

    assert is_dirty(git_repo, exclude=["runs/index.csv"]) is True


def test_excluding_a_modified_tracked_file_ignores_it(git_repo: Path) -> None:
    write(git_repo, "qcal.toml", "# changed\n")

    assert is_dirty(git_repo, exclude=["qcal.toml"]) is False


def test_empty_exclusion_list_behaves_like_no_exclusions(git_repo: Path) -> None:
    write(git_repo, "new.txt", "x\n")

    assert is_dirty(git_repo, exclude=[]) is True


def test_is_dirty_with_exclusions_outside_a_repository_is_none(outside: Path) -> None:
    assert is_dirty(outside, exclude=["runs/index.csv"]) is None


def test_exclusions_are_repo_relative_when_called_from_a_subdirectory(git_repo: Path) -> None:
    sub = git_repo / "src"
    write(git_repo, "src/keep.py", "x = 1\n")
    commit_all(git_repo, "add src")
    write(git_repo, "README.md", "changed outside src\n")

    assert is_dirty(sub, exclude=["runs/index.csv"]) is True


# --- show_file ----------------------------------------------------------------------------


def test_show_file_returns_contents_at_a_ref(git_repo: Path) -> None:
    assert show_file("HEAD", "qcal.toml", git_repo) == "# test project\n"


def test_show_file_reads_the_commit_not_the_work_tree(git_repo: Path) -> None:
    first = run_git(git_repo, "rev-parse", "HEAD")
    write(git_repo, "qcal.toml", "# second\n")
    commit_all(git_repo, "second")
    write(git_repo, "qcal.toml", "# uncommitted\n")

    assert show_file(first, "qcal.toml", git_repo) == "# test project\n"
    assert show_file("HEAD", "qcal.toml", git_repo) == "# second\n"


def test_show_file_reads_nested_paths(git_repo: Path) -> None:
    write(git_repo, "a/b/c.txt", "deep\n")
    commit_all(git_repo, "nested")

    assert show_file("HEAD", "a/b/c.txt", git_repo) == "deep\n"


@pytest.mark.parametrize(
    ("ref", "path"),
    [("HEAD", "missing.txt"), ("no-such-ref", "qcal.toml"), ("HEAD~5", "qcal.toml")],
)
def test_show_file_returns_none_when_ref_or_path_is_absent(
    git_repo: Path, ref: str, path: str
) -> None:
    assert show_file(ref, path, git_repo) is None


# --- commit resolution, merge base, timeouts ------------------------------------------------


def test_resolve_commit_accepts_refs_and_rejects_non_commits(git_repo: Path) -> None:
    head = run_git(git_repo, "rev-parse", "HEAD")
    assert gitutil.resolve_commit("HEAD", git_repo) == head
    tree = run_git(git_repo, "rev-parse", "HEAD^{tree}")
    for bad in ("nope", tree, "--all"):
        with pytest.raises(GitError):
            gitutil.resolve_commit(bad, git_repo)


def test_merge_base_of_unrelated_histories_is_none(git_repo: Path) -> None:
    head = run_git(git_repo, "rev-parse", "HEAD")
    run_git(git_repo, "checkout", "-q", "--orphan", "other")
    write(git_repo, "o.txt", "o\n")
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "orphan")
    other = run_git(git_repo, "rev-parse", "HEAD")

    assert gitutil.merge_base(head, other, git_repo) is None
    assert gitutil.merge_base(head, head, git_repo) == head


def test_timeout_scope_restores_the_previous_value() -> None:
    before = gitutil._timeout.get()
    with gitutil.timeout_scope(3):
        assert gitutil._timeout.get() == 3
    assert gitutil._timeout.get() == before


def test_set_timeout_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        gitutil.set_timeout(-1)
