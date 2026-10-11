"""Signed-commit CI check with real SSH signatures.

Every commit in ``base..head`` that touches a protected category must carry a good
signature from a key in the *base* ref's ``allowed_signers``, judged by the *base*
ref's ``qcal.toml``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from qcal.ci.signatures import (
    CommitVerdict,
    SignatureReport,
    changed_files,
    count_keys,
    github_annotations,
    policy_config,
    verify_signatures,
)
from tests.conftest import require_tool, run_git, write

pytestmark = [pytest.mark.integration, pytest.mark.rule("C0")]
signing = pytest.mark.requires_ssh_keygen

ENFORCE = '[signing]\nmode = "enforce"\n'
BOOTSTRAP = '[signing]\nmode = "bootstrap"\n'
NO_KEYS = "allowed_signers has no keys on the base ref"
COMMITTER = "test@example.invalid"  # GIT_COMMITTER_EMAIL set by conftest; the signing principal
FAKE_VERIFIER = """#!/bin/sh
# Stands in for ssh-keygen and vouches for every signature.
for a in "$@"; do case "$a" in find-principals) echo "test@example.invalid"; exit 0;; esac; done
cat >/dev/null
echo 'Good "git" signature for test@example.invalid with ED25519 key SHA256:fake'
"""


@dataclass(frozen=True)
class Keys:
    allowed: Path
    intruder: Path


def generate_key(ssh_keygen: str, path: Path) -> Path:
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-C", path.name, "-f", str(path)],
        check=True,
        capture_output=True,
    )
    return path


def signer_line(key: Path, principal: str = COMMITTER) -> str:
    public = Path(f"{key}.pub").read_text("utf-8").strip()
    return f'{principal} namespaces="git" {public}\n'


def commit(
    root: Path, message: str, files: Mapping[str, str] | None = None, *, key: Path | None = None
) -> str:
    """Write ``files``, stage everything and commit; SSH-sign with ``key`` when given."""
    for relative, text in (files or {}).items():
        write(root, relative, text)
    run_git(root, "add", "-A")
    if key is None:
        run_git(root, "commit", "-q", "--allow-empty", "-m", message)
    else:
        run_git(
            root,
            "-c",
            "gpg.format=ssh",
            "-c",
            f"user.signingkey={key}",
            "commit",
            "-q",
            "-S",
            "--allow-empty",
            "-m",
            message,
        )
    return run_git(root, "rev-parse", "HEAD")


def verify(root: Path, **kwargs: Any) -> SignatureReport:
    return verify_signatures(root, "main", "feature", **kwargs)


def violations(report: SignatureReport) -> list[tuple[str, list[str]]]:
    return [(v.sha, v.protected_files) for v in report.violations]


@pytest.fixture
def keys(ssh_keygen: str, tmp_path: Path) -> Keys:
    directory = tmp_path / "keys"
    directory.mkdir()
    return Keys(
        generate_key(ssh_keygen, directory / "allowed"),
        generate_key(ssh_keygen, directory / "intruder"),
    )


@pytest.fixture
def make_project(git_repo: Path) -> Callable[[str, str | None], Path]:
    """Commit a base policy (and ``allowed_signers`` unless ``None``) on main, then branch."""

    def _make(policy: str, signers: str | None) -> Path:
        files = {"qcal.toml": policy, "CLAIMS.md": "# claims\n", "notes/todo.md": "- none\n"}
        if signers is not None:
            files["allowed_signers"] = signers
        commit(git_repo, "base policy", files)
        run_git(git_repo, "checkout", "-q", "-b", "feature")
        return git_repo

    return _make


@pytest.fixture
def project(make_project: Callable[[str, str | None], Path], keys: Keys) -> Path:
    """Enforcing base policy whose allowed_signers lists only ``keys.allowed``."""
    return make_project(ENFORCE, signer_line(keys.allowed))


# --- enforce mode -------------------------------------------------------------------------


@signing
def test_unsigned_protected_change_is_a_violation_in_enforce_mode(project: Path) -> None:
    sha = commit(project, "edit claims", {"CLAIMS.md": "# claims\nnew\n"})

    report = verify(project)

    assert report.mode == "enforce"
    assert not report.passed
    assert report.violations == [CommitVerdict(sha, ["CLAIMS.md"], False, "unsigned")]


@signing
def test_protected_change_signed_by_an_allowed_key_passes(project: Path, keys: Keys) -> None:
    sha = commit(project, "edit claims", {"CLAIMS.md": "# claims\nnew\n"}, key=keys.allowed)

    report = verify(project)

    assert report.passed
    assert report.verdicts == [CommitVerdict(sha, ["CLAIMS.md"], True, "good signature")]


@signing
def test_unprotected_unsigned_change_passes(project: Path) -> None:
    sha = commit(project, "notes", {"notes/todo.md": "- more\n", "src/model.py": "x = 1\n"})

    report = verify(project)

    assert report.passed
    assert report.verdicts == [CommitVerdict(sha, [], True, "no protected paths")]


@signing
def test_protected_change_signed_by_an_unlisted_key_is_a_violation(
    project: Path, keys: Keys
) -> None:
    sha = commit(project, "edit claims", {"CLAIMS.md": "# claims\nx\n"}, key=keys.intruder)

    report = verify(project)

    assert violations(report) == [(sha, ["CLAIMS.md"])]
    assert "principal" in report.violations[0].detail.lower()


@signing
def test_only_protected_files_are_listed_for_a_commit(project: Path) -> None:
    files = {"CLAIMS.md": "x\n", "Makefile": "all:\n", "notes/todo.md": "y\n", "src/m.py": "z\n"}
    sha = commit(project, "mixed", files)

    report = verify(project)

    assert violations(report) == [(sha, ["CLAIMS.md", "Makefile"])]


@signing
def test_each_commit_is_judged_on_its_own(project: Path, keys: Keys) -> None:
    signed = commit(project, "signed", {"CLAIMS.md": "a\n"}, key=keys.allowed)
    unsigned = commit(project, "unsigned", {"CLAIMS.md": "b\n"})
    plain = commit(project, "plain", {"notes/todo.md": "c\n"})

    report = verify(project)

    assert [v.sha for v in report.verdicts] == [signed, unsigned, plain]
    assert violations(report) == [(unsigned, ["CLAIMS.md"])]


@signing
def test_deleting_a_protected_file_needs_a_signature(project: Path) -> None:
    run_git(project, "rm", "-q", "CLAIMS.md")
    sha = commit(project, "drop claims")

    assert violations(verify(project)) == [(sha, ["CLAIMS.md"])]


@signing
def test_renaming_a_protected_file_away_needs_a_signature(project: Path) -> None:
    run_git(project, "mv", "CLAIMS.md", "notes/claims.md")
    sha = commit(project, "move claims")

    assert violations(verify(project)) == [(sha, ["CLAIMS.md"])]


@signing
@pytest.mark.parametrize("path", ["runs/registry/R1.json", "runs/index.csv"])
def test_registry_paths_need_a_signature(project: Path, path: str) -> None:
    sha = commit(project, "record", {path: "{}\n"})

    assert violations(verify(project)) == [(sha, [path])]


@signing
def test_signed_registry_record_passes(project: Path, keys: Keys) -> None:
    commit(project, "record", {"runs/registry/R1.json": "{}\n"}, key=keys.allowed)

    assert verify(project).passed


@signing
def test_commits_already_on_base_are_not_checked(project: Path) -> None:
    sha = commit(project, "notes", {"notes/todo.md": "x\n"})

    report = verify(project)

    assert [v.sha for v in report.verdicts] == [sha]
    assert report.to_dict()["checked_commits"] == 1


@signing
def test_empty_range_checks_nothing_and_passes(project: Path) -> None:
    report = verify(project)

    assert report.verdicts == []
    assert report.passed


# --- bootstrap and mode selection ----------------------------------------------------------


@signing
def test_bootstrap_mode_reports_violations_but_passes(
    make_project: Callable[[str, str | None], Path], keys: Keys
) -> None:
    root = make_project(BOOTSTRAP, signer_line(keys.allowed))
    sha = commit(root, "edit claims", {"CLAIMS.md": "x\n"})

    report = verify(root)

    assert report.mode == "bootstrap"
    assert report.passed
    assert violations(report) == [(sha, ["CLAIMS.md"])]


@signing
def test_mode_argument_overrides_the_base_policy(
    make_project: Callable[[str, str | None], Path], keys: Keys
) -> None:
    root = make_project(BOOTSTRAP, signer_line(keys.allowed))
    commit(root, "edit claims", {"CLAIMS.md": "x\n"})

    report = verify(root, mode="enforce")

    assert report.mode == "enforce"
    assert not report.passed


def test_unknown_mode_argument_is_rejected(git_repo: Path) -> None:
    run_git(git_repo, "branch", "feature")

    with pytest.raises(ValueError, match=r"signing\.mode must be one of .*'strict'"):
        verify(git_repo, mode="strict")


def test_unknown_mode_in_the_base_policy_is_rejected(git_repo: Path) -> None:
    commit(git_repo, "policy", {"qcal.toml": '[signing]\nmode = "audit"\n'})
    run_git(git_repo, "branch", "feature")

    with pytest.raises(ValueError, match=r"signing\.mode must be one of .*'audit'"):
        verify(git_repo)


def test_base_without_qcal_toml_is_judged_by_packaged_defaults(git_repo: Path) -> None:
    run_git(git_repo, "rm", "-q", "qcal.toml")
    commit(git_repo, "no policy file")
    run_git(git_repo, "checkout", "-q", "-b", "feature")
    sha = commit(git_repo, "edit claims", {"CLAIMS.md": "x\n"})

    report = verify(git_repo)

    assert (report.mode, report.keys, report.passed) == ("bootstrap", 0, True)
    assert report.violations == [CommitVerdict(sha, ["CLAIMS.md"], False, NO_KEYS)]


# --- the base ref, not the head, decides --------------------------------------------------


@signing
@pytest.mark.parametrize(
    "relaxed",
    [
        pytest.param(BOOTSTRAP, id="bootstrap-mode"),
        pytest.param(ENFORCE + "signed_categories = []\n", id="no-signed-categories"),
        pytest.param(
            ENFORCE + "[policy.categories]\nian_only = []\nenforcement_surface = []\n",
            id="empty-categories",
        ),
        pytest.param(ENFORCE + 'allowed_signers = "my_signers"\n', id="own-signers-file"),
    ],
)
def test_head_relaxing_qcal_toml_is_judged_by_the_base_policy(
    project: Path, keys: Keys, relaxed: str
) -> None:
    files = {"qcal.toml": relaxed, "my_signers": signer_line(keys.intruder)}
    relax = commit(project, "relax policy", files, key=keys.intruder)
    edit = commit(project, "edit claims", {"CLAIMS.md": "unreviewed\n"}, key=keys.intruder)

    report = verify(project)

    assert (report.mode, report.keys, report.passed) == ("enforce", 1, False)
    assert violations(report) == [(relax, ["qcal.toml"]), (edit, ["CLAIMS.md"])]


@signing
def test_head_adding_its_own_key_is_judged_by_the_base_signers(project: Path, keys: Keys) -> None:
    signers = signer_line(keys.allowed) + signer_line(keys.intruder)
    add_key = commit(project, "add my key", {"allowed_signers": signers}, key=keys.intruder)
    edit = commit(project, "edit claims", {"CLAIMS.md": "unreviewed\n"}, key=keys.intruder)

    report = verify(project)

    assert report.keys == 1
    assert violations(report) == [(add_key, ["allowed_signers"]), (edit, ["CLAIMS.md"])]


@signing
def test_policy_ref_selects_whose_policy_and_signers_apply(project: Path, keys: Keys) -> None:
    signers = signer_line(keys.allowed) + signer_line(keys.intruder)
    commit(project, "add my key", {"allowed_signers": signers}, key=keys.intruder)
    commit(project, "edit claims", {"CLAIMS.md": "unreviewed\n"}, key=keys.intruder)

    report = verify(project, policy_ref="feature")

    assert report.keys == 2
    assert report.passed


@signing
@pytest.mark.parametrize(
    "signers",
    [None, "", "# Ian: add your signing key here\n\n   \n"],
    ids=["absent", "empty", "comments-only"],
)
def test_allowed_signers_without_keys_rejects_even_signed_commits(
    make_project: Callable[[str, str | None], Path], keys: Keys, signers: str | None
) -> None:
    root = make_project(ENFORCE, signers)
    sha = commit(root, "edit claims", {"CLAIMS.md": "x\n"}, key=keys.allowed)

    report = verify(root)

    assert report.keys == 0
    assert report.violations == [CommitVerdict(sha, ["CLAIMS.md"], False, NO_KEYS)]
    assert not report.passed


# --- signer identity and verifier ---------------------------------------------------------


@signing
@pytest.mark.parametrize("committer", ["other@example.invalid", "test@example.invalid.evil"])
def test_signature_by_an_allowed_key_for_another_committer_is_a_violation(
    project: Path, keys: Keys, monkeypatch: pytest.MonkeyPatch, committer: str
) -> None:
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", committer)
    sha = commit(project, "edit claims", {"CLAIMS.md": "x\n"}, key=keys.allowed)

    report = verify(project)

    assert report.violations == [
        CommitVerdict(
            sha, ["CLAIMS.md"], False, f"signed by {COMMITTER!r} but committed as {committer!r}"
        )
    ]


@signing
def test_committer_email_is_matched_case_insensitively(
    project: Path, keys: Keys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "TEST@Example.Invalid")
    commit(project, "edit claims", {"CLAIMS.md": "x\n"}, key=keys.allowed)

    assert verify(project).passed


@signing
def test_committer_match_can_be_disabled_by_the_base_policy(
    make_project: Callable[[str, str | None], Path],
    keys: Keys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(ENFORCE + "require_committer_match = false\n", signer_line(keys.allowed))
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "other@example.invalid")
    commit(root, "edit claims", {"CLAIMS.md": "x\n"}, key=keys.allowed)

    assert verify(root).passed


@signing
def test_principal_other_than_the_committer_is_a_violation(
    make_project: Callable[[str, str | None], Path], keys: Keys
) -> None:
    root = make_project(ENFORCE, signer_line(keys.allowed, principal="ian@example.invalid"))
    commit(root, "edit claims", {"CLAIMS.md": "x\n"}, key=keys.allowed)

    report = verify(root)

    assert [v.detail for v in report.violations] == [
        f"signed by 'ian@example.invalid' but committed as {COMMITTER!r}"
    ]


def _repo_config(root: Path, fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_git(root, "config", "gpg.ssh.program", str(fake))


def _env_config(root: Path, fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "gpg.ssh.program")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(fake))


@signing
@pytest.mark.parametrize("install", [_repo_config, _env_config], ids=["repo-config", "env-config"])
def test_git_config_cannot_swap_the_verifier(
    project: Path,
    keys: Keys,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    install: Callable[[Path, Path, pytest.MonkeyPatch], None],
) -> None:
    sha = commit(project, "edit claims", {"CLAIMS.md": "x\n"}, key=keys.intruder)
    fake = write(tmp_path, "fake-ssh-keygen", FAKE_VERIFIER)
    fake.chmod(0o755)
    install(project, fake, monkeypatch)

    assert violations(verify(project)) == [(sha, ["CLAIMS.md"])]


@signing
def test_verifier_program_comes_from_the_base_policy(
    make_project: Callable[[str, str | None], Path], keys: Keys, tmp_path: Path
) -> None:
    fake = write(tmp_path, "fake-ssh-keygen", FAKE_VERIFIER)
    fake.chmod(0o755)
    root = make_project(ENFORCE + f'ssh_program = "{fake}"\n', signer_line(keys.allowed))
    commit(root, "edit claims", {"CLAIMS.md": "x\n"}, key=keys.intruder)

    assert verify(root).passed


# --- merges -------------------------------------------------------------------------------


def _diverge_and_change_main(root: Path) -> None:
    """Feature gets an unprotected commit; main gets an unsigned protected change."""
    commit(root, "feature work", {"notes/todo.md": "- feature\n"})
    run_git(root, "checkout", "-q", "main")
    commit(root, "direct change on main", {"CLAIMS.md": "# claims\nmain\n", "Makefile": "all:\n"})
    run_git(root, "checkout", "-q", "feature")


@signing
def test_merging_the_base_branch_into_the_feature_creates_no_violation(project: Path) -> None:
    _diverge_and_change_main(project)
    run_git(project, "merge", "-q", "--no-edit", "main")
    merge = run_git(project, "rev-parse", "HEAD")

    report = verify(project)

    assert len(run_git(project, "rev-list", "--parents", "-n", "1", merge).split()) == 3
    assert report.passed
    assert [v for v in report.verdicts if v.sha == merge] == [
        CommitVerdict(merge, [], True, "no protected paths")
    ]


@signing
@pytest.mark.parametrize(("signed", "passed"), [(False, False), (True, True)])
def test_merge_that_rewrites_a_protected_file_needs_a_signature(
    project: Path, keys: Keys, signed: bool, passed: bool
) -> None:
    _diverge_and_change_main(project)
    run_git(project, "merge", "-q", "--no-commit", "--no-ff", "main")
    write(project, "CLAIMS.md", "# claims\nrewritten while merging\n")
    merge = commit(project, "merge main", key=keys.allowed if signed else None)

    report = verify(project)

    assert next(v for v in report.verdicts if v.sha == merge).protected_files == ["CLAIMS.md"]
    assert report.passed is passed


# --- changed_files ------------------------------------------------------------------------


def test_changed_files_of_a_root_commit_lists_its_tree(git_repo: Path) -> None:
    assert changed_files(git_repo, "HEAD") == ["qcal.toml"]


def test_changed_files_lists_added_modified_and_deleted_paths(git_repo: Path) -> None:
    commit(git_repo, "seed", {"a.txt": "a\n", "b.txt": "b\n"})
    (git_repo / "a.txt").unlink()
    sha = commit(git_repo, "change", {"b.txt": "B\n", "c/d.txt": "d\n"})

    assert changed_files(git_repo, sha) == ["a.txt", "b.txt", "c/d.txt"]


def test_changed_files_reports_both_sides_of_a_rename(git_repo: Path) -> None:
    commit(git_repo, "seed", {"old.txt": "same content\n"})
    run_git(git_repo, "mv", "old.txt", "new.txt")
    sha = commit(git_repo, "rename")

    assert changed_files(git_repo, sha) == ["new.txt", "old.txt"]


def test_changed_files_of_a_clean_merge_is_empty(git_repo: Path) -> None:
    run_git(git_repo, "checkout", "-q", "-b", "side")
    commit(git_repo, "side", {"side.txt": "s\n"})
    run_git(git_repo, "checkout", "-q", "main")
    commit(git_repo, "main", {"main.txt": "m\n"})
    run_git(git_repo, "merge", "-q", "--no-edit", "side")

    assert changed_files(git_repo, "HEAD") == []


# --- policy_config and count_keys -----------------------------------------------------------


def test_policy_config_reads_the_committed_file_not_the_work_tree(git_repo: Path) -> None:
    commit(git_repo, "policy", {"qcal.toml": BOOTSTRAP})
    write(git_repo, "qcal.toml", ENFORCE)

    assert policy_config(git_repo, "HEAD").str_value("signing.mode") == "bootstrap"


def test_policy_config_ignores_environment_overrides(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit(git_repo, "policy", {"qcal.toml": ENFORCE})
    monkeypatch.setenv("QCAL__SIGNING__MODE", '"bootstrap"')

    assert policy_config(git_repo, "HEAD").str_value("signing.mode") == "enforce"


def test_policy_config_without_a_committed_file_is_the_packaged_defaults(git_repo: Path) -> None:
    run_git(git_repo, "rm", "-q", "qcal.toml")
    commit(git_repo, "drop policy")
    write(git_repo, "qcal.toml", ENFORCE)

    config = policy_config(git_repo, "HEAD")

    assert config.sources == ("defaults",)
    assert config.str_value("signing.mode") == "bootstrap"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, 0),
        ("", 0),
        ("# comment\n\n   \n  # indented comment\n", 0),
        ('a@x namespaces="git" ssh-ed25519 AAAA\n', 1),
        ('a@x ssh-ed25519 AAAA\n# note\nb@x namespaces="git" ssh-ed25519 BBBB\n', 2),
    ],
)
def test_count_keys_counts_non_comment_lines(text: str | None, expected: int) -> None:
    assert count_keys(text) == expected


# --- report and annotations ---------------------------------------------------------------


@signing
def test_annotations_for_an_enforced_violation(project: Path) -> None:
    sha = commit(project, "edit", {"CLAIMS.md": "x\n", "qcal.toml": ENFORCE + "# touched\n"})

    lines = github_annotations(verify(project))

    assert lines == [
        (
            f"::error::commit {sha[:12]} changes protected paths (CLAIMS.md, qcal.toml) "
            "without an allowed signature: unsigned"
        )
    ]


def test_bootstrap_annotations_warn_and_explain_how_to_enforce() -> None:
    sha = "0123456789abcdef0123"
    report = SignatureReport("bootstrap", 0, [CommitVerdict(sha, ["CLAIMS.md"], False, NO_KEYS)])

    assert github_annotations(report) == [
        (
            f"::warning::commit 0123456789ab changes protected paths (CLAIMS.md) "
            f"without an allowed signature: {NO_KEYS}"
        ),
        (
            "::notice::signing.mode is 'bootstrap'; set it to 'enforce' in qcal.toml "
            "(in a signed commit) to make this check blocking."
        ),
    ]


@pytest.mark.parametrize("mode", ["bootstrap", "enforce"])
def test_no_violations_means_no_annotations(mode: str) -> None:
    report = SignatureReport(mode, 1, [CommitVerdict("a" * 40, [], True, "no protected paths")])

    assert github_annotations(report) == []


def test_report_dict_lists_only_violations() -> None:
    bad = CommitVerdict("b" * 40, ["Makefile"], False, "unsigned")
    report = SignatureReport(
        "enforce", 2, [CommitVerdict("a" * 40, ["CLAIMS.md"], True, "good signature"), bad]
    )

    assert report.to_dict() == {
        "mode": "enforce",
        "allowed_keys": 2,
        "verdict": "FAIL",
        "violations": [{"sha": "b" * 40, "files": ["Makefile"], "detail": "unsigned"}],
        "checked_commits": 2,
        "checked_net_paths": 0,
    }


def test_bootstrap_report_with_violations_still_passes() -> None:
    report = SignatureReport("bootstrap", 0, [CommitVerdict("a" * 40, ["X"], False, "unsigned")])

    assert report.passed
    assert report.to_dict()["verdict"] == "PASS"


# --- signatures that are not SSH signatures from allowed_signers -------------------------------


@pytest.fixture
def gpg_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A throwaway OpenPGP key whose UID is the committer email, trusted in a private keyring."""
    gpg, gpgconf = require_tool("gpg"), require_tool("gpgconf")
    home = tmp_path / "gpg"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("GNUPGHOME", str(home))
    created = subprocess.run(
        [gpg, "--batch", "--pinentry-mode", "loopback", "--passphrase", "", "--quick-gen-key",
         COMMITTER, "ed25519", "sign", "never"],
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    if created.returncode != 0:
        pytest.skip(f"cannot create a gpg key: {created.stderr.strip()[-200:]}")
    listing = subprocess.run(
        [gpg, "--batch", "--with-colons", "--list-secret-keys"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    yield next(line.split(":")[9] for line in listing.splitlines() if line.startswith("fpr:"))
    subprocess.run([gpgconf, "--kill", "all"], capture_output=True, check=False)


@signing
def test_openpgp_signature_outside_allowed_signers_is_a_violation(
    project: Path, gpg_key: str
) -> None:
    write(project, "CLAIMS.md", "# claims\ngpg-signed\n")
    run_git(project, "add", "-A")
    run_git(
        project, "-c", "gpg.format=openpgp", "-c", f"user.signingkey={gpg_key}",
        "commit", "-q", "-S", "-m", "gpg signed",
    )  # fmt: skip
    sha = run_git(project, "rev-parse", "HEAD")

    report = verify(project)

    assert violations(report) == [(sha, ["CLAIMS.md"])]
