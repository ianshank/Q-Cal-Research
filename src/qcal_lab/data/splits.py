"""Split manifests: deterministic, order-independent, and hashed exactly like ``qcal leakage``.

A manifest is one image id per line at ``paths.manifests_dir / data.manifest_pattern``, with
optional comment lines (``data.comment_prefix``) recording where the split came from. Draws
and partitions rank ids by ``sha256(seed, id)``, so the result depends only on the set of ids
and the seed: never on file order, platform or Python's hash seed.

Writing refuses to replace a manifest with a different set of ids: choosing a split after
seeing results is split shopping. Ian decides any change, through ``AMENDMENTS.md``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from qcal.config import Config
from qcal.integrity.leakage import manifest_path as leakage_manifest_path
from qcal.integrity.leakage import read_manifest
from qcal.log import get_logger

_log = get_logger("lab.data.splits")


class SplitError(ValueError):
    """A split request is impossible or would replace an existing split."""


def split_digest(ids: Iterable[str]) -> str:
    """The digest ``qcal leakage`` reports for a split: sha256 of the sorted unique ids."""
    return hashlib.sha256("\n".join(sorted(set(ids))).encode()).hexdigest()


def rank_key(seed: int | str, image_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{image_id}".encode()).hexdigest()


def _ranked(ids: Iterable[str], seed: int | str) -> list[str]:
    unique = set(ids)
    if any(not isinstance(i, str) or not i for i in unique):
        raise SplitError("image ids must be non-empty strings")
    return sorted(unique, key=lambda i: (rank_key(seed, i), i))


def draw(ids: Iterable[str], size: int, seed: int | str) -> tuple[str, ...]:
    """A seeded subset of ``size`` ids, returned sorted."""
    ranked = _ranked(ids, seed)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SplitError(f"subset size must be a positive integer, got {size!r}")
    if size > len(ranked):
        raise SplitError(f"cannot draw {size} images from a split of {len(ranked)}")
    return tuple(sorted(ranked[:size]))


def partition(
    ids: Iterable[str], sizes: Sequence[tuple[str, int]], seed: int | str
) -> dict[str, tuple[str, ...]]:
    """Disjoint seeded splits, filled in the order given; leftover ids belong to no split."""
    ranked = _ranked(ids, seed)
    names = [name for name, _ in sizes]
    if len(set(names)) != len(names):
        raise SplitError("split names must be unique")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for _, n in sizes):
        raise SplitError("split sizes must be non-negative integers")
    total = sum(n for _, n in sizes)
    if total > len(ranked):
        raise SplitError(f"the splits need {total} images but only {len(ranked)} exist")
    result: dict[str, tuple[str, ...]] = {}
    start = 0
    for name, size in sizes:
        result[name] = tuple(sorted(ranked[start : start + size]))
        start += size
    return result


def render_manifest(ids: Iterable[str], header: Mapping[str, str], comment_prefix: str) -> str:
    lines = [f"{comment_prefix} {key}: {value}" for key, value in header.items()]
    lines.extend(sorted(set(ids)))
    return "\n".join(lines) + "\n"


def write_manifest(
    path: Path,
    ids: Iterable[str],
    header: Mapping[str, str],
    comment_prefix: str,
    *,
    replace: bool = False,
) -> str:
    """Write a manifest and return its digest; refuse to change an existing split's ids."""
    unique = sorted(set(ids))
    if not unique:
        raise SplitError(f"refusing to write an empty split to {path}")
    digest = split_digest(unique)
    if path.is_file():
        existing = read_manifest(path.read_text("utf-8"), comment_prefix)
        if split_digest(existing) == digest:
            _log.info("%s already holds this split (%s)", path, digest[:12])
            return digest
        if not replace:
            raise SplitError(
                f"{path} already holds a different split; replacing a split after results "
                "exist is split shopping. Record the change in AMENDMENTS.md first."
            )
        _log.warning("replacing the split in %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_manifest(unique, header, comment_prefix), encoding="utf-8")
    _log.info("wrote %s: %d ids (%s)", path, len(unique), digest[:12])
    return digest


def manifest_path(config: Config, split: str) -> Path:
    """The split's manifest for the default dataset (``qcal``'s rule, shared with leakage)."""
    return leakage_manifest_path(config, split)


def read_split(config: Config, split: str) -> tuple[str, ...]:
    """The sorted unique ids of a split's manifest (``qcal`` data settings)."""
    path = manifest_path(config, split)
    if not path.is_file():
        raise SplitError(f"split {split!r} has no manifest at {path}")
    ids = read_manifest(path.read_text("utf-8"), config.str_value("data.comment_prefix"))
    if len(set(ids)) != len(ids):
        raise SplitError(f"{path} lists some image ids more than once")
    if not ids:
        raise SplitError(f"{path} is empty")
    return tuple(sorted(ids))


__all__ = [
    "SplitError",
    "draw",
    "manifest_path",
    "partition",
    "rank_key",
    "read_split",
    "render_manifest",
    "split_digest",
    "write_manifest",
]
