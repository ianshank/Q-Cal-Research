"""Regression tests for the security review of the signed-commit control (real SSH signing)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from qcal.ci.immutability import check_registry_immutable
from qcal.ci.signatures import signature_kind, verify_signatures
from tests.conftest import run_git, write

pytestmark = [pytest.mark.integration, pytest.mark.requires_ssh_keygen]
PRINCIPAL = "ian@example.invalid"


@pytest.fixture
def signed_repo(
    repo: Path, ssh_keygen: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str, str]:
    """A repo whose first commit is pre-policy history, then a signed commit adding the policy."""
    for variable in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(variable, PRINCIPAL)  # committer must equal the signing principal
    key = tmp_path / "ian"
    subprocess.run([ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    run_git(repo, "init", "-q", "-b", "main")
    run_git(repo, "config", "user.email", PRINCIPAL)
    run_git(repo, "config", "commit.gpgsign", "false")
    (repo / "qcal.toml").unlink()
    write(repo, "README.md", "old history\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "old history without a policy")
    old = run_git(repo, "rev-parse", "HEAD")
    pub = (tmp_path / "ian.pub").read_text().strip()
    write(repo, "allowed_signers", f'{PRINCIPAL} namespaces="git" {pub}\n')
    write(repo, "qcal.toml", '[signing]\nmode = "enforce"\n')
    write(repo, "EXPERIMENTS.yaml", "version: 2\n")
    run_git(repo, "add", "-A")
    sign(repo, key, "signed policy")
    base = run_git(repo, "rev-parse", "HEAD")
    return repo, key, old, base


def sign(repo: Path, key: Path, message: str) -> None:
    run_git(
        repo,
        "-c",
        "gpg.format=ssh",
        "-c",
        "gpg.ssh.program=ssh-keygen",
        "-c",
        f"user.signingkey={key}",
        "commit",
        "-q",
        "-S",
        "-m",
        message,
    )


def test_crafted_merge_cannot_delete_protected_files(signed_repo) -> None:
    repo, _key, old, base = signed_repo
    write(repo, "docs/note.md", "harmless\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "harmless unsigned change")
    tip = run_git(repo, "rev-parse", "HEAD")
    run_git(repo, "rm", "-q", "qcal.toml", "allowed_signers", "EXPERIMENTS.yaml")
    tree = run_git(repo, "write-tree")
    merge = run_git(repo, "commit-tree", tree, "-p", tip, "-p", old, "-m", "innocent merge")

    report = verify_signatures(repo, base, merge)

    assert not report.passed
    flagged = {f for v in report.violations for f in v.protected_files}
    assert {"qcal.toml", "allowed_signers", "EXPERIMENTS.yaml"} <= flagged


def test_rolling_back_to_old_signed_content_is_a_violation(signed_repo) -> None:
    repo, key, _old, base = signed_repo
    write(repo, "EXPERIMENTS.yaml", "version: 3\n")
    run_git(repo, "add", "-A")
    sign(repo, key, "signed amendment")
    new_base = run_git(repo, "rev-parse", "HEAD")
    tip_tree_with_old = run_git(repo, "rev-parse", f"{base}^{{tree}}")
    rollback = run_git(
        repo, "commit-tree", tip_tree_with_old, "-p", new_base, "-p", base, "-m", "merge"
    )

    report = verify_signatures(repo, new_base, rollback)

    assert not report.passed
    assert any("EXPERIMENTS.yaml" in v.protected_files for v in report.violations)


def test_signed_side_branch_merged_by_an_unsigned_merge_passes(signed_repo) -> None:
    repo, key, _old, base = signed_repo
    run_git(repo, "checkout", "-q", "-b", "side")
    write(repo, "EXPERIMENTS.yaml", "version: 3\n")
    run_git(repo, "add", "-A")
    sign(repo, key, "signed amendment on a side branch")
    run_git(repo, "checkout", "-q", "main")
    run_git(repo, "checkout", "-q", "-b", "feature")
    write(repo, "docs/x.md", "x\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "unsigned docs")
    run_git(repo, "merge", "-q", "--no-edit", "--no-gpg-sign", "side")

    assert verify_signatures(repo, base, run_git(repo, "rev-parse", "HEAD")).passed


@pytest.mark.parametrize(
    "path", [".github/workflows/é.yml", ".claude/agents/é.md", '.claude/agents/a"b.md']
)
def test_quoted_paths_cannot_dodge_the_patterns(signed_repo, path: str) -> None:
    repo, _key, _old, base = signed_repo
    write(repo, path, "x\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "unsigned addition with an unusual name")

    report = verify_signatures(repo, base, run_git(repo, "rev-parse", "HEAD"))

    assert not report.passed
    assert report.violations[0].protected_files == [path]


def test_quoted_record_names_are_still_validated(signed_repo) -> None:
    repo, _key, _old, base = signed_repo
    write(repo, "runs/registry/é.json", "{}\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "odd record")

    report = check_registry_immutable(repo, base, run_git(repo, "rev-parse", "HEAD"))

    assert not report.passed
    assert "invalid record" in report.violations[0]


def test_signature_kind_reads_the_commit_header(signed_repo) -> None:
    repo, _key, old, base = signed_repo
    assert signature_kind(repo, base) == "ssh"
    assert signature_kind(repo, old) == "no"
