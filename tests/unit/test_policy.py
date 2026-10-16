from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from qcal.config import ConfigError
from qcal.policy import Policy

pytestmark = pytest.mark.rule("C5", "C6")


@pytest.fixture
def policy(config) -> Policy:
    return Policy.from_config(config)


@pytest.mark.parametrize(
    ("path", "categories"),
    [
        ("EXPERIMENTS.yaml", ("ian_only",)),
        ("experiments.yaml", ("ian_only",)),
        ("CLAIMS.md", ("ian_only",)),
        ("DECISIONS.md", ("ian_only",)),
        ("src/qcal/handwritten/laece.py", ("ian_only", "enforcement_surface")),
        ("src/qcal_lab/handwritten/laece.py", ("ian_only",)),
        ("src/qcal/cli/__init__.py", ("enforcement_surface",)),  # shadowing sub-package
        ("analysis/handwritten/error_analysis.md", ("ian_only",)),
        ("paper/sections/abstract.tex", ("ian_only",)),
        ("runs/index.csv", ("registry_only",)),
        ("runs/registry/R1.json", ("registry_only",)),
        (".claude/settings.json", ("enforcement_surface",)),
        (".github/workflows/ci.yml", ("enforcement_surface",)),
        ("qcal.toml", ("enforcement_surface",)),
        ("pyproject.toml", ("enforcement_surface",)),
        ("src/qcal/cli.py", ("enforcement_surface",)),
        ("src/qcal/registry/tables.py", ("enforcement_surface",)),
        ("review/gemini-x.md", ("enforcement_surface",)),
        ("src/qcal_lab/calib/platt.py", ()),
        ("paper/sections/method.tex", ()),
        ("docs/notes.md", ()),
        ("AMENDMENTS.md", ()),
    ],
)
def test_default_categories(policy: Policy, path: str, categories: tuple[str, ...]) -> None:
    assert policy.categories_for(path) == categories


def test_evaluate_absolute_and_relative_paths(policy: Policy, repo: Path) -> None:
    assert policy.evaluate(repo / "EXPERIMENTS.yaml", repo).categories == ("ian_only",)
    assert policy.evaluate("EXPERIMENTS.yaml", repo).relative == "EXPERIMENTS.yaml"
    assert policy.evaluate("src/../EXPERIMENTS.yaml", repo).categories == ("ian_only",)


def test_symlink_to_a_protected_file_is_protected(policy: Policy, repo: Path) -> None:
    (repo / "EXPERIMENTS.yaml").write_text("x")
    (repo / "innocent.yaml").symlink_to(repo / "EXPERIMENTS.yaml")
    verdict = policy.evaluate(repo / "innocent.yaml", repo)
    assert verdict.relative == "innocent.yaml"
    assert verdict.resolved_relative == "EXPERIMENTS.yaml"
    assert verdict.categories == ("ian_only",)


def test_symlinked_directory_is_resolved(policy: Policy, repo: Path) -> None:
    (repo / ".claude").mkdir()
    (repo / "notes").symlink_to(repo / ".claude")
    assert policy.evaluate(repo / "notes" / "settings.json", repo).categories == (
        "enforcement_surface",
    )


def test_paths_outside_the_root(policy: Policy, repo: Path, tmp_path: Path) -> None:
    verdict = policy.evaluate(tmp_path / "elsewhere.txt", repo)
    assert verdict.outside_root
    assert verdict.categories == ()


def test_root_itself_is_relative_empty(policy: Policy, repo: Path) -> None:
    assert policy.evaluate(repo, repo).relative == ""


@pytest.mark.parametrize(
    "text", ["docs/NBCU-notes.md", "x/Edge_DIT/a.py", "edge-dit", "EdgeDit.txt"]
)
def test_clean_room_substrings_are_case_insensitive(policy: Policy, repo: Path, text: str) -> None:
    assert policy.evaluate(text, repo).clean_room_hits


def test_case_sensitive_policy() -> None:
    strict = Policy({"x": ["A.md"]}, clean_room_substrings=["Secret"], case_insensitive=False)
    assert strict.categories_for("a.md") == ()
    assert strict.clean_room_hits("secret") == ()
    assert strict.clean_room_hits("Secret") == ("Secret",)


def test_categories_are_dynamic_from_configuration(make_config) -> None:
    config = make_config('[policy.categories]\nreviewers_only = ["review/**"]\n')
    policy = Policy.from_config(config)
    assert "reviewers_only" in policy.category_names
    assert policy.categories_for("review/a.md") == ("reviewers_only",)
    assert policy.patterns("reviewers_only") == ("review/**",)
    assert policy.patterns("nope") == ()


def test_messages_and_fallback(policy: Policy) -> None:
    assert "Ian-only" in policy.message_for("ian_only")
    assert "protected (other)" in policy.message_for("other")


def test_blocked_by_and_in_categories(policy: Policy, repo: Path) -> None:
    verdict = policy.evaluate("runs/index.csv", repo)
    assert verdict.blocked_by(["ian_only"]) == ()
    assert verdict.blocked_by(["registry_only"]) == ("registry_only",)
    assert policy.in_categories("qcal.toml", ["enforcement_surface"])
    assert not policy.in_categories("docs/a.md", ["enforcement_surface"])


# --- clean-room matching folds Unicode (docs/changes/qcal-foundations.md, S2) ----------------

FOLDING = Policy({}, clean_room_substrings=["secret-lab"])
# Long-stable code points only: the matrix spans Unicode 14 to 15.1.
STABLE = st.characters(min_codepoint=0x20, max_codepoint=0x24F) | st.sampled_from(
    ["\u200b", "\u00ad", "\ufeff", "\u0301", "\u0327", "\u2010", "\u2011", "\u2212", "\uff41"]
)


@given(st.text(alphabet=STABLE, max_size=12))
@example("e\u0301")
def test_folding_is_idempotent(text: str) -> None:
    assert FOLDING.fold(FOLDING.fold(text)) == FOLDING.fold(text)


@given(
    st.text(alphabet=STABLE, max_size=12),
    st.integers(0, 12),
    st.sampled_from(["\u200b", "\u00ad", "\ufeff"]),
)
def test_a_format_character_anywhere_changes_nothing(text: str, at: int, cf: str) -> None:
    assert FOLDING.fold(text[:at] + cf + text[at:]) == FOLDING.fold(text)


@given(st.text(alphabet=STABLE, max_size=12))
@example("caf\u00e9")
def test_composed_and_decomposed_spellings_fold_alike(text: str) -> None:
    nfc, nfd = unicodedata.normalize("NFC", text), unicodedata.normalize("NFD", text)
    assert FOLDING.fold(nfc) == FOLDING.fold(nfd)


def test_fullwidth_ascii_folds_like_ascii() -> None:
    fullwidth = "".join(chr(ord(c) + 0xFEE0) for c in "Secret-Lab")
    assert FOLDING.fold(fullwidth) == FOLDING.fold("secret-lab") == "secret-lab"


@pytest.mark.parametrize(
    "text",
    [
        "a SECRET-LAB note",
        "secret\u2011lab",  # non-breaking hyphen
        "secret\u2212lab",  # minus sign
        "secr\u00e9t-lab",  # precomposed accent
        "se\u200bcret-lab",  # zero-width space
    ],
)
def test_disguised_spellings_are_clean_room_hits(text: str) -> None:
    assert FOLDING.clean_room_hits(text) == ("secret-lab",)


def test_folding_never_widens_the_needles() -> None:
    assert FOLDING.clean_room_hits("secret lab") == ()
    assert FOLDING.clean_room_hits("secretlab") == ()


def test_case_sensitive_matching_does_not_casefold() -> None:
    strict = Policy({}, clean_room_substrings=["Lab"], case_insensitive=False)
    assert strict.clean_room_hits("lab") == ()
    assert strict.clean_room_hits("\uff2c\uff41\uff42") == ("Lab",)  # full-width "Lab"


def test_folding_can_be_switched_off() -> None:
    plain = Policy(
        {}, clean_room_substrings=["lab"], unicode_normalization="", strip_unicode_categories=[]
    )
    assert plain.clean_room_hits("\uff4c\uff41\uff42") == ()
    assert plain.clean_room_hits("LAB") == ("lab",)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unicode_normalization": "NFKX"}, "unicode_normalization must be one of"),
        ({"strip_unicode_categories": ["Cf", "Zz"]}, "unknown categories"),
    ],
)
def test_a_misconfigured_fold_is_a_configuration_error(kwargs: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Policy({}, **kwargs)


def test_folding_is_configured_from_the_policy_section(make_config) -> None:
    config = make_config('[policy]\nunicode_normalization = "NFC"\nstrip_unicode_categories = []\n')
    policy = Policy.from_config(config)
    assert policy.fold("\uff41") == "\uff41"  # NFC keeps full-width forms
