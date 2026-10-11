"""License audit: installed distributions, forbidden imports and dataset cards."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from qcal.config import Config, ConfigError
from qcal.integrity.licenses import (
    FrontmatterError,
    LicenseReport,
    check_dataset_cards,
    check_imports,
    check_licenses,
    check_packages,
    distribution_license,
    imported_modules,
    metadata_value,
    read_frontmatter,
)
from tests.conftest import write

pytestmark = pytest.mark.rule("C6", "A6")

Headers = Sequence[tuple[str, str]]


@pytest.fixture
def make_dist(tmp_path: Path) -> Callable[..., importlib.metadata.Distribution]:
    """Build a real ``*.dist-info`` directory and return its PathDistribution."""
    site = tmp_path / "site-packages"

    def _make(name: str, *headers: tuple[str, str], version: str = "1.0") -> Any:
        info = site / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        body = [
            "Metadata-Version: 2.1",
            f"Name: {name}",
            f"Version: {version}",
            *(f"{k}: {v}" for k, v in headers),
        ]
        (info / "METADATA").write_text("\n".join(body) + "\n", encoding="utf-8")
        return importlib.metadata.PathDistribution(info)

    return _make


def audit(config: Config, *dists: importlib.metadata.Distribution) -> LicenseReport:
    report = LicenseReport()
    check_packages(config, report, list(dists))
    return report


def scan_imports(config: Config) -> LicenseReport:
    report = LicenseReport()
    check_imports(config, report)
    return report


def card(root: Path, name: str, front: str | None) -> Path:
    path = root / "docs" / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Dataset\n" if front is None else f"---\n{front}---\n# Dataset\n"
    path.write_text(text, encoding="utf-8")
    return path


def cards(config: Config) -> LicenseReport:
    report = LicenseReport()
    check_dataset_cards(config, report)
    return report


# --- distribution metadata ----------------------------------------------------------------


def test_metadata_value_returns_the_first_value(make_dist: Callable[..., Any]) -> None:
    dist = make_dist("pkg", ("Classifier", "First"), ("Classifier", "Second"))

    assert metadata_value(dist, "Classifier") == "First"


def test_metadata_value_of_absent_field_is_empty(make_dist: Callable[..., Any]) -> None:
    assert metadata_value(make_dist("pkg"), "License") == ""


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        pytest.param((("License-Expression", "MIT"),), "MIT", id="expression"),
        pytest.param((("License", "  BSD-3-Clause  "),), "BSD-3-Clause", id="stripped"),
        pytest.param(
            (
                ("License-Expression", "Apache-2.0"),
                ("License", "Apache License 2.0"),
                ("Classifier", "Programming Language :: Python"),
                ("Classifier", "License :: OSI Approved :: Apache Software License"),
            ),
            "Apache-2.0 | Apache License 2.0 | License :: OSI Approved :: Apache Software License",
            id="all-sources-in-order",
        ),
        pytest.param((("License", "   "),), "", id="blank"),
        pytest.param((), "", id="none"),
    ],
)
def test_distribution_license_joins_every_license_source(
    make_dist: Callable[..., Any], headers: Headers, expected: str
) -> None:
    assert distribution_license(make_dist("pkg", *headers)) == expected


# --- check_packages -----------------------------------------------------------------------


def test_permissive_packages_pass(config: Config, make_dist: Callable[..., Any]) -> None:
    report = audit(
        config,
        make_dist("alpha", ("License-Expression", "MIT")),
        make_dist("beta", ("Classifier", "License :: OSI Approved :: BSD License")),
        make_dist("gamma"),
    )

    assert report.passed
    assert (report.errors, report.warnings, report.checked_packages) == ([], [], 3)


@pytest.mark.parametrize("name", ["ultralytics", "Ultralytics", "ULTRALYTICS"])
def test_denied_package_is_an_error_whatever_its_case(
    config: Config, make_dist: Callable[..., Any], name: str
) -> None:
    report = audit(config, make_dist(name, ("License", "AGPL-3.0")))

    assert report.errors == ["package ultralytics is denied by policy"]
    assert report.checked_packages == 1


@pytest.mark.parametrize(
    ("header", "text", "hit"),
    [
        ("License", "GNU AGPLv3", "AGPL"),
        ("License-Expression", "agpl-3.0-only", "AGPL"),
        ("License-Expression", "SSPL-1.0", "SSPL"),
        ("License", "Apache-2.0 with Commons Clause", "Commons Clause"),
    ],
)
def test_denied_license_substring_is_an_error(
    config: Config, make_dist: Callable[..., Any], header: str, text: str, hit: str
) -> None:
    report = audit(config, make_dist("pkg", (header, text)))

    assert report.errors == [f"package pkg has a denied license ({hit}): {text}"]
    assert report.warnings == []


@pytest.mark.parametrize(
    ("header", "text"),
    [
        ("License-Expression", "GPL-3.0-or-later"),
        ("License", "LGPL-2.1"),
        ("License", "CC-BY-NC-4.0"),
        ("License", "Creative Commons NonCommercial"),
        ("Classifier", "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"),
    ],
)
def test_review_license_substring_is_a_warning(
    config: Config, make_dist: Callable[..., Any], header: str, text: str
) -> None:
    report = audit(config, make_dist("pkg", (header, text)))

    assert report.passed
    assert report.warnings == [f"package pkg license needs review: {text}"]


def test_license_lists_and_denied_packages_are_configurable(
    make_config: Callable[[str], Config], make_dist: Callable[..., Any]
) -> None:
    config = make_config(
        """
        [licenses]
        deny_license_substrings = ["Proprietary"]
        warn_license_substrings = ["MPL"]
        deny_packages = ["leftpad"]
        """
    )

    report = audit(
        config,
        make_dist("closed", ("License", "Proprietary")),
        make_dist("moz", ("License", "MPL-2.0")),
        make_dist("leftpad", ("License", "MIT")),
        make_dist("agpl-ok-here", ("License", "AGPL-3.0")),
    )

    assert report.errors == [
        "package closed has a denied license (Proprietary): Proprietary",
        "package leftpad is denied by policy",
    ]
    assert report.warnings == ["package moz license needs review: MPL-2.0"]


def test_installed_distributions_are_audited_by_default(
    config: Config, make_dist: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = [make_dist("ultralytics"), make_dist("fine", ("License", "MIT"))]
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: iter(fake))
    report = LicenseReport()

    check_packages(config, report)

    assert report.errors == ["package ultralytics is denied by policy"]
    assert report.checked_packages == 2


# --- imports ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import a, b.c", {"a", "b.c"}),
        ("import numpy as np", {"numpy"}),
        ("from x.y import z", {"x.y"}),
        ("from . import sibling", set()),
        ("from .pkg import thing", set()),
        ("def f():\n    import lazy\n", {"lazy"}),
        ("try:\n    import fast\nexcept ImportError:\n    fast = None\n", {"fast"}),
        ("text = 'import hidden'  # import also_hidden", set()),
    ],
)
def test_imported_modules_lists_absolute_imports(source: str, expected: set[str]) -> None:
    assert imported_modules(source) == expected


def test_imported_modules_rejects_invalid_source() -> None:
    with pytest.raises(SyntaxError):
        imported_modules("import (")


@pytest.mark.parametrize(
    "source",
    [
        b"import json\n",
        b"# -*- coding: latin-1 -*-\nimport json\nname = '\xe9'\n",
        "import json\nname = '\u00e9'\n".encode(),
    ],
    ids=["ascii", "pep263-latin1", "utf8"],
)
def test_imported_modules_accepts_encoded_source(source: bytes) -> None:
    assert imported_modules(source) == {"json"}


@pytest.mark.parametrize(
    "source",
    [
        "import ultralytics",
        "import ultralytics.engine.model",
        "from ultralytics import YOLO",
        "from ultralytics.utils import ops",
        "import numpy, ultralytics as u",
        "def build():\n    from ultralytics import YOLO\n    return YOLO\n",
    ],
)
def test_forbidden_import_is_an_error(config: Config, repo: Path, source: str) -> None:
    write(repo, "src/pkg/model.py", source)

    report = scan_imports(config)

    assert report.errors == ["src/pkg/model.py imports ultralytics, which policy forbids here"]


@pytest.mark.parametrize(
    "source",
    [
        "import ultralyticsx",
        "from . import ultralytics",
        "from .ultralytics import YOLO",
        "name = 'ultralytics'",
        "# import ultralytics",
    ],
)
def test_lookalike_imports_are_not_forbidden(config: Config, repo: Path, source: str) -> None:
    write(repo, "src/pkg/model.py", source)

    assert scan_imports(config).errors == []


def test_forbidden_import_is_allowed_inside_its_allowed_globs(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    config = make_config('[licenses]\nimport_scan_globs = ["src/**/*.py", "tests/**/*.py"]\n')
    write(repo, "tests/parity/test_oracle.py", "import detection_calibration")
    write(repo, "src/qcal/metrics.py", "from detection_calibration import laece")

    report = scan_imports(config)

    assert report.errors == [
        "src/qcal/metrics.py imports detection_calibration, which policy forbids here"
    ]
    assert report.checked_files == 2


def test_only_files_matching_scan_globs_are_checked(config: Config, repo: Path) -> None:
    write(repo, "notebooks/explore.py", "import ultralytics")
    write(repo, "scripts/tool.py", "import json")
    write(repo, "src/readme.txt", "import ultralytics")

    report = scan_imports(config)

    assert report.errors == []
    assert report.checked_files == 1


def test_unparsable_source_is_a_warning(config: Config, repo: Path) -> None:
    write(repo, "src/broken.py", "def f(:\n")
    write(repo, "src/bad.py", "import ultralytics")

    report = scan_imports(config)

    assert len(report.warnings) == 1
    assert report.warnings[0].startswith("src/broken.py: cannot parse (")
    assert report.errors == ["src/bad.py imports ultralytics, which policy forbids here"]
    assert report.checked_files == 2


def test_forbidden_import_rule_without_module_is_a_config_error(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    config = make_config("[licenses]\nforbidden_imports = [{ allowed_globs = [] }]\n")
    write(repo, "src/x.py", "import os")

    with pytest.raises(ConfigError, match="need a string 'module'"):
        scan_imports(config)


def test_source_with_a_declared_encoding_is_scanned(config: Config, repo: Path) -> None:
    path = repo / "src" / "legacy.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"# -*- coding: latin-1 -*-\nimport ultralytics\nname = '\xe9'\n")

    report = scan_imports(config)

    assert report.errors == ["src/legacy.py imports ultralytics, which policy forbids here"]


@pytest.mark.parametrize(
    "payload",
    [b"name = '\xff\xfe'\n", b"# -*- coding: no-such-codec -*-\nimport os\n", b"import os\0\n"],
    ids=["undecodable-utf8", "unknown-codec", "null-byte"],
)
def test_undecodable_source_is_a_warning_not_a_crash(
    config: Config, repo: Path, payload: bytes
) -> None:
    path = repo / "src" / "odd.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    report = scan_imports(config)

    assert len(report.warnings) == 1
    assert report.warnings[0].startswith("src/odd.py: cannot parse (")
    assert report.errors == []


# --- frontmatter --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "---\nlicense: MIT\nname: coco\n---\nbody\n", {"license": "MIT", "name": "coco"}
        ),
        pytest.param("--- \nlicense: MIT\n  ---  \n", {"license": "MIT"}, id="padded-fences"),
        pytest.param(
            "---\nkey: [a, b]\n---\n---\nother: 1\n---\n", {"key": ["a", "b"]}, id="first"
        ),
        pytest.param("", {}, id="empty-text"),
        pytest.param("# Title\nlicense: MIT\n", {}, id="no-fence"),
        pytest.param("\n---\nlicense: MIT\n---\n", {}, id="fence-not-first-line"),
        pytest.param("---\nlicense: MIT\n", {}, id="unclosed"),
        pytest.param("---\n---\nbody\n", {}, id="empty-block"),
        pytest.param("---\n# only a comment\n---\n", {}, id="comment-only"),
        pytest.param("---\n- a\n- b\n---\n", {}, id="list"),
        pytest.param("---\njust text\n---\n", {}, id="scalar"),
    ],
)
def test_read_frontmatter(text: str, expected: dict[str, Any]) -> None:
    assert read_frontmatter(text) == expected


def test_malformed_frontmatter_yaml_raises_frontmatter_error() -> None:
    with pytest.raises(FrontmatterError, match="invalid YAML frontmatter") as caught:
        read_frontmatter("---\nlicense: [unclosed\n---\n")

    assert isinstance(caught.value.__cause__, yaml.YAMLError)


# --- dataset cards ------------------------------------------------------------------------


def test_no_dataset_card_directory_checks_nothing(config: Config) -> None:
    report = cards(config)

    assert report.passed
    assert report.checked_cards == 0


@pytest.mark.parametrize(
    "front",
    [None, "name: coco\n", "license: ''\n", "license: [MIT]\n", "license: 4.0\n"],
    ids=["no-frontmatter", "no-key", "empty", "list", "number"],
)
def test_card_without_a_license_string_is_an_error(
    config: Config, repo: Path, front: str | None
) -> None:
    card(repo, "coco.md", front)

    assert cards(config).errors == ["dataset card coco.md has no 'license' in its frontmatter"]


def test_card_with_malformed_frontmatter_is_an_error(config: Config, repo: Path) -> None:
    card(repo, "broken.md", "license: [MIT\n")
    card(repo, "fine.md", "license: MIT\n")

    report = cards(config)

    assert len(report.errors) == 1
    assert report.errors[0].startswith("dataset card broken.md: invalid YAML frontmatter:")
    assert report.checked_cards == 2


@pytest.mark.parametrize("value", ["BDD100K", "bdd100k"])
def test_denied_dataset_license_is_an_error_whatever_its_case(
    make_config: Callable[[str], Config], repo: Path, value: str
) -> None:
    config = make_config('[licenses]\ndataset_denied = ["BDD100K"]\n')
    card(repo, "bdd.md", f"license: {value}\n")

    assert cards(config).errors == [f"dataset card bdd.md uses denied license {value}"]


def test_license_missing_from_a_nonempty_allowlist_is_an_error(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    config = make_config('[licenses]\ndataset_allowed = ["CC-BY-4.0"]\n')
    card(repo, "ok.md", "license: CC-BY-4.0\n")
    card(repo, "other.md", "license: MIT\n")

    report = cards(config)

    assert report.errors == ["dataset card other.md license MIT is not on the allowlist"]
    assert report.checked_cards == 2


def test_empty_allowlist_accepts_any_license_that_is_not_denied(config: Config, repo: Path) -> None:
    card(repo, "any.md", "license: Some-Custom-Terms\n")

    assert cards(config).passed


def test_only_markdown_cards_at_the_top_level_are_checked(config: Config, repo: Path) -> None:
    card(repo, "real.md", "license: MIT\n")
    write(repo, "docs/data/notes.txt", "no frontmatter\n")
    write(repo, "docs/data/nested/deep.md", "no frontmatter\n")

    report = cards(config)

    assert report.passed
    assert report.checked_cards == 1


def test_license_key_and_directory_are_configurable(
    make_config: Callable[[str], Config], repo: Path
) -> None:
    config = make_config(
        '[paths]\ndataset_cards_dir = "cards"\n[licenses]\ndataset_license_key = "licence"\n'
    )
    write(repo, "cards/a.md", "---\nlicence: MIT\n---\n")
    write(repo, "cards/b.md", "---\nlicense: MIT\n---\n")

    assert cards(config).errors == ["dataset card b.md has no 'licence' in its frontmatter"]


# --- whole audit and report ---------------------------------------------------------------


def test_check_licenses_without_packages_skips_installed_distributions(
    config: Config, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _forbidden() -> None:
        raise AssertionError("distributions() must not be called")

    monkeypatch.setattr(importlib.metadata, "distributions", _forbidden)
    write(repo, "src/a.py", "import ultralytics")
    card(repo, "c.md", None)

    report = check_licenses(config, packages=False)

    assert report.errors == [
        "src/a.py imports ultralytics, which policy forbids here",
        "dataset card c.md has no 'license' in its frontmatter",
    ]
    assert report.checked_packages == 0


def test_check_licenses_combines_every_check(
    config: Config,
    repo: Path,
    make_dist: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = [make_dist("gpl-thing", ("License", "GPL-2.0"))]
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: iter(fake))
    write(repo, "src/a.py", "import os")
    card(repo, "c.md", "license: MIT\n")

    data = check_licenses(config).to_dict()

    assert data == {
        "verdict": "PASS",
        "errors": [],
        "warnings": ["package gpl-thing license needs review: GPL-2.0"],
        "checked": {"packages": 1, "files": 1, "dataset_cards": 1},
    }


def test_report_with_errors_fails() -> None:
    report = LicenseReport(errors=["x"])

    assert not report.passed
    assert report.to_dict()["verdict"] == "FAIL"
