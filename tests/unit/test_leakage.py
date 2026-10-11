"""Split-manifest leakage check: per-split hashes, duplicates, missing splits and overlaps."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from qcal.config import Config, ConfigError
from qcal.integrity.leakage import LeakageReport, check_leakage, manifest_path, read_manifest

SPLITS = ("trt_calib_images", "calibrator_fit_split", "val", "test")


def digest(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(ids))).encode()).hexdigest()


def manifest(config: Config, split: str, text: str) -> Path:
    pattern = config.str_value("data.manifest_pattern")
    path = config.path("manifests_dir") / pattern.format(split=split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def lines(*ids: str) -> str:
    return "".join(f"{i}\n" for i in ids)


@pytest.fixture
def disjoint(config: Config) -> Config:
    """All four default splits present and pairwise disjoint."""
    for split in SPLITS:
        manifest(config, split, lines(f"{split}-1", f"{split}-2"))
    return config


# --- read_manifest ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "prefix", "expected"),
    [
        ("a\nb\n", "#", ["a", "b"]),
        ("  a  \n\tb\n", "#", ["a", "b"]),
        ("# header\na\n  # indented comment\nb\n", "#", ["a", "b"]),
        ("a\n\n   \nb", "#", ["a", "b"]),
        ("a\r\nb\r\n", "#", ["a", "b"]),
        ("; note\na\n# kept\n", ";", ["a", "# kept"]),
        ("", "#", []),
        ("b\na\nb\n", "#", ["b", "a", "b"]),
    ],
)
def test_read_manifest_keeps_ids_in_order_without_comments_or_blanks(
    text: str, prefix: str, expected: list[str]
) -> None:
    assert read_manifest(text, prefix) == expected


def test_empty_comment_prefix_disables_comments() -> None:
    assert read_manifest("a\n# b\n", "") == ["a", "# b"]


def test_empty_comment_prefix_still_detects_overlaps(
    make_config: Callable[[str], Config],
) -> None:
    config = make_config('[data]\nsplits = ["val", "test"]\ncomment_prefix = ""\n')
    manifest(config, "val", lines("#1", "x"))
    manifest(config, "test", lines("#1", "y"))

    report = check_leakage(config)

    assert report.overlaps == {"test&val": ["#1"]}


# --- check_leakage ------------------------------------------------------------------------


def test_disjoint_manifests_pass(disjoint: Config) -> None:
    report = check_leakage(disjoint)

    assert report.passed
    assert (report.missing, report.duplicates, report.overlaps) == ([], {}, {})


def test_each_present_split_is_hashed_and_sized(disjoint: Config) -> None:
    report = check_leakage(disjoint)

    assert report.present == {s: digest([f"{s}-1", f"{s}-2"]) for s in SPLITS}
    assert report.sizes == dict.fromkeys(SPLITS, 2)


def test_hash_ignores_order_comments_blanks_and_whitespace(config: Config) -> None:
    manifest(config, "val", "# exported 2026-10-09\n\n  b  \na\n")

    report = check_leakage(config)

    assert report.present["val"] == digest(["a", "b"])


def test_hash_changes_when_an_id_changes(config: Config) -> None:
    manifest(config, "val", lines("a", "c"))

    report = check_leakage(config)

    assert report.present["val"] != digest(["a", "b"])


def test_missing_split_fails(disjoint: Config) -> None:
    (disjoint.path("manifests_dir") / "test.txt").unlink()

    report = check_leakage(disjoint)

    assert not report.passed
    assert report.missing == ["test"]
    assert "test" not in report.present


def test_missing_directory_reports_every_split_in_config_order(config: Config) -> None:
    report = check_leakage(config)

    assert report.missing == list(SPLITS)
    assert not report.passed


def test_duplicates_fail_and_are_counted(disjoint: Config) -> None:
    manifest(disjoint, "val", lines("a", "a", "b", "a"))

    report = check_leakage(disjoint)

    assert not report.passed
    assert report.duplicates == {"val": 2}
    assert report.sizes["val"] == 2
    assert report.present["val"] == digest(["a", "b"])


def test_overlap_between_two_splits_fails_with_sorted_shared_ids(disjoint: Config) -> None:
    manifest(disjoint, "val", lines("v1", "c", "b"))
    manifest(disjoint, "test", lines("b", "t1", "c"))

    report = check_leakage(disjoint)

    assert not report.passed
    assert report.overlaps == {"test&val": ["b", "c"]}


def test_every_overlapping_pair_is_reported_once(config: Config) -> None:
    for split in SPLITS:
        manifest(config, split, lines("shared", f"{split}-only"))

    report = check_leakage(config)

    assert sorted(report.overlaps) == [
        "calibrator_fit_split&test",
        "calibrator_fit_split&trt_calib_images",
        "calibrator_fit_split&val",
        "test&trt_calib_images",
        "test&val",
        "trt_calib_images&val",
    ]
    assert all(ids == ["shared"] for ids in report.overlaps.values())


def test_comment_lines_do_not_count_as_overlap(disjoint: Config) -> None:
    manifest(disjoint, "val", "# source: coco\nv1\n")
    manifest(disjoint, "test", "# source: coco\nt1\n")

    assert check_leakage(disjoint).passed


def test_splits_directory_pattern_and_comment_prefix_are_configurable(
    make_config: Callable[[str], Config],
) -> None:
    config = make_config(
        """
        [paths]
        manifests_dir = "splits"
        [data]
        splits = ["fit", "holdout"]
        manifest_pattern = "{split}.lst"
        comment_prefix = ";"
        """
    )
    manifest(config, "fit", "; header\nx\n")
    manifest(config, "holdout", "; header\nx\n")

    report = check_leakage(config)

    assert (config.root / "splits" / "fit.lst").is_file()
    assert report.overlaps == {"fit&holdout": ["x"]}
    assert report.sizes == {"fit": 1, "holdout": 1}


def test_no_configured_splits_trivially_passes(make_config: Callable[[str], Config]) -> None:
    config = make_config("[data]\nsplits = []\n")

    report = check_leakage(config)

    assert report.passed
    assert report.present == {}


# --- report -------------------------------------------------------------------------------


def test_empty_report_passes() -> None:
    assert LeakageReport().passed


def test_report_dict_for_a_passing_check(disjoint: Config) -> None:
    data = check_leakage(disjoint).to_dict()

    assert data["verdict"] == "PASS"
    assert data["splits"]["val"] == {"sha256": digest(["val-1", "val-2"]), "size": 2}
    assert (data["missing"], data["duplicates"], data["overlaps"]) == ([], {}, {})


def test_report_dict_truncates_overlap_examples_to_ten(disjoint: Config) -> None:
    shared = [f"img{i:02d}" for i in range(12)]
    manifest(disjoint, "val", lines(*shared))
    manifest(disjoint, "test", lines(*shared))

    data = check_leakage(disjoint).to_dict()

    assert data["verdict"] == "FAIL"
    assert data["overlaps"] == {"test&val": {"count": 12, "examples": shared[:10]}}


def test_report_dict_lists_missing_and_duplicates(config: Config) -> None:
    manifest(config, "val", lines("a", "a"))

    data = check_leakage(config).to_dict()

    assert data["verdict"] == "FAIL"
    assert data["missing"] == ["trt_calib_images", "calibrator_fit_split", "test"]
    assert data["duplicates"] == {"val": 1}


# -- manifest paths (PR-A2) ---------------------------------------------------------------


def test_manifest_path_defaults_to_the_first_dataset(make_config: Callable[[str], Config]) -> None:
    config = make_config(
        '[data]\nmanifest_pattern = "{dataset}/{split}.txt"\ndatasets = ["id", "fog"]\n'
    )
    assert manifest_path(config, "val") == config.path("manifests_dir") / "id" / "val.txt"
    assert manifest_path(config, "val", "fog") == config.path("manifests_dir") / "fog" / "val.txt"


def test_manifest_path_refuses_unknown_placeholders(make_config: Callable[[str], Config]) -> None:
    config = make_config('[data]\nmanifest_pattern = "{split}-{seed}.txt"\n')
    with pytest.raises(ConfigError, match="seed"):
        manifest_path(config, "val")


def test_leakage_reads_dataset_scoped_manifests(make_config: Callable[[str], Config]) -> None:
    config = make_config('[data]\nmanifest_pattern = "{dataset}/{split}.txt"\n')
    for split, ids in (
        ("trt_calib_images", "1"),
        ("calibrator_fit_split", "2"),
        ("val", "3"),
        ("test", "4"),
    ):
        path = manifest_path(config, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ids + "\n")
    assert check_leakage(config).passed
