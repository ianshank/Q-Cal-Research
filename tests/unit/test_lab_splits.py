"""Split manifests (plan §5 Phase 1 ``tests/test_splits.py``): deterministic, disjoint, and
hashed exactly like ``qcal leakage``."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qcal.config import Config
from qcal.integrity.leakage import check_leakage
from qcal_lab.data.splits import (
    SplitError,
    draw,
    manifest_path,
    partition,
    read_split,
    render_manifest,
    split_digest,
    write_manifest,
)
from tests.conftest import write

IDS = [str(i) for i in range(1, 51)]


def test_digest_matches_the_leakage_check(config: Config) -> None:
    sizes = [("trt_calib_images", 5), ("calibrator_fit_split", 10), ("val", 10), ("test", 10)]
    parts = partition(IDS, sizes, seed=1)
    for split, ids in parts.items():
        write_manifest(manifest_path(config, split), ids, {"seed": "1"}, "#")
    report = check_leakage(config)
    assert report.passed
    assert report.present == {split: split_digest(ids) for split, ids in parts.items()}


def test_draw_is_seeded_sorted_and_independent_of_input_order() -> None:
    first = draw(IDS, 7, seed=3)
    assert first == tuple(sorted(first))
    assert draw(list(reversed(IDS)) + IDS, 7, seed=3) == first
    assert draw(IDS, 7, seed=4) != first
    assert set(draw(IDS, 8, seed=3)) > set(first)  # a larger draw extends a smaller one


@pytest.mark.parametrize("size", [0, -1, True, 51, 2.5])
def test_draw_refuses_impossible_sizes(size: object) -> None:
    with pytest.raises(SplitError):
        draw(IDS, size, seed=0)  # type: ignore[arg-type]


def test_partition_is_disjoint_and_exact() -> None:
    parts = partition(IDS, [("a", 5), ("b", 20), ("c", 0)], seed="x")
    assert [len(parts[k]) for k in "abc"] == [5, 20, 0]
    assert not set(parts["a"]) & set(parts["b"])
    assert partition(reversed(IDS), [("a", 5), ("b", 20), ("c", 0)], seed="x") == parts


@pytest.mark.parametrize(
    ("sizes", "message"),
    [
        ([("a", 30), ("b", 30)], "need 60 images"),
        ([("a", 1), ("a", 1)], "unique"),
        ([("a", -1)], "non-negative"),
    ],
)
def test_partition_refuses_bad_requests(sizes: list[tuple[str, int]], message: str) -> None:
    with pytest.raises(SplitError, match=message):
        partition(IDS, sizes, seed=0)


def test_ids_must_be_non_empty_strings() -> None:
    with pytest.raises(SplitError, match="non-empty strings"):
        draw(["a", ""], 1, seed=0)


@given(
    st.sets(st.text(alphabet="abcdef0123456789", min_size=1, max_size=6), min_size=1, max_size=40),
    st.integers(0, 5),
    st.data(),
)
def test_partition_properties(ids: set[str], seed: int, data: st.DataObject) -> None:
    first = data.draw(st.integers(0, len(ids)))
    second = data.draw(st.integers(0, len(ids) - first))
    parts = partition(ids, [("a", first), ("b", second)], seed)
    assert len(parts["a"]) == first
    assert len(parts["b"]) == second
    assert not set(parts["a"]) & set(parts["b"])
    assert set(parts["a"]) | set(parts["b"]) <= ids


def test_write_manifest_is_idempotent_and_refuses_a_different_split(tmp_path: Path) -> None:
    path = tmp_path / "m/val.txt"
    digest = write_manifest(path, ["2", "1"], {"source": "x"}, "#")
    assert path.read_text() == "# source: x\n1\n2\n"
    assert write_manifest(path, ["1", "2", "2"], {"source": "other"}, "#") == digest
    assert path.read_text().startswith("# source: x")  # unchanged
    with pytest.raises(SplitError, match="split shopping"):
        write_manifest(path, ["1", "3"], {}, "#")
    assert write_manifest(path, ["1", "3"], {}, "#", replace=True) == split_digest(["1", "3"])
    with pytest.raises(SplitError, match="empty split"):
        write_manifest(tmp_path / "e.txt", [], {}, "#")


def test_read_split(config: Config) -> None:
    with pytest.raises(SplitError, match="has no manifest"):
        read_split(config, "val")
    write(config.root, "data/manifests/val.txt", "# header\n3\n1\n")
    assert read_split(config, "val") == ("1", "3")
    write(config.root, "data/manifests/val.txt", "1\n1\n")
    with pytest.raises(SplitError, match="more than once"):
        read_split(config, "val")
    write(config.root, "data/manifests/val.txt", "# only a comment\n")
    with pytest.raises(SplitError, match="is empty"):
        read_split(config, "val")


def test_render_manifest_sorts_and_deduplicates() -> None:
    assert render_manifest(["b", "a", "b"], {}, "#") == "a\nb\n"
