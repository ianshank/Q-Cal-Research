"""Split-manifest leakage check: hashes per split and pairwise disjointness.

Manifests are plain text, one image id per line, at
``paths.manifests_dir / data.manifest_pattern``. Comment lines start with
``data.comment_prefix``. This is the deterministic half of the
``data-leakage-checker`` agent's job.
"""

from __future__ import annotations

import hashlib
import itertools
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from qcal.config import Config, ConfigError
from qcal.log import get_logger
from qcal.reports import verdict

_log = get_logger("integrity.leakage")
#: The names data.manifest_pattern may use.
MANIFEST_PLACEHOLDERS: Final = frozenset({"split", "dataset"})


@dataclass
class LeakageReport:
    present: dict[str, str] = field(default_factory=dict)  # split -> sha256 of sorted ids
    sizes: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    duplicates: dict[str, int] = field(default_factory=dict)
    overlaps: dict[str, list[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.overlaps and not self.duplicates and not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": verdict(self.passed),
            "splits": {k: {"sha256": v, "size": self.sizes[k]} for k, v in self.present.items()},
            "missing": self.missing,
            "duplicates": self.duplicates,
            "overlaps": {
                k: {"count": len(v), "examples": v[:10]} for k, v in self.overlaps.items()
            },
        }

    def render_text(self) -> str:
        sizes = ", ".join(f"{k}={self.sizes[k]}" for k in self.present)
        lines = [f"leakage: {verdict(self.passed)} ({sizes or 'no splits'})"]
        lines += [f"  missing split: {name}" for name in self.missing]
        lines += [f"  duplicate ids in {name}: {n}" for name, n in self.duplicates.items()]
        lines += [
            f"  overlap {pair}: {len(ids)} id(s), e.g. {', '.join(ids[:3])}"
            for pair, ids in self.overlaps.items()
        ]
        return "\n".join(lines)


def read_manifest(text: str, comment_prefix: str) -> list[str]:
    lines = (line.strip() for line in text.splitlines())
    return [
        line for line in lines if line and not (comment_prefix and line.startswith(comment_prefix))
    ]


def manifest_path(config: Config, split: str, dataset: str | None = None) -> Path:
    """Where a split's manifest lives: ``data.manifest_pattern`` in ``paths.manifests_dir``.

    The pattern may name ``{split}`` and ``{dataset}`` (default: the first of
    ``data.datasets``), so shifted datasets can keep their own manifests.
    """
    pattern = config.str_value("data.manifest_pattern")
    unknown = sorted(
        {name for _, name, _, _ in string.Formatter().parse(pattern) if name}
        - MANIFEST_PLACEHOLDERS
    )
    if unknown:
        raise ConfigError(
            f"data.manifest_pattern uses {unknown}; allowed: {sorted(MANIFEST_PLACEHOLDERS)}"
        )
    if dataset is None:
        datasets = config.str_list("data.datasets")
        if not datasets:
            raise ConfigError("data.datasets is empty; name at least the in-domain dataset")
        dataset = datasets[0]
    name = dataset
    return config.path("manifests_dir") / pattern.format(split=split, dataset=name)


def check_leakage(config: Config) -> LeakageReport:
    comment = config.str_value("data.comment_prefix")
    report = LeakageReport()
    ids: dict[str, set[str]] = {}
    for split in config.str_list("data.splits"):
        path = manifest_path(config, split)
        if not path.is_file():
            report.missing.append(split)
            continue
        entries = read_manifest(path.read_text("utf-8"), comment)
        unique = set(entries)
        if len(unique) != len(entries):
            report.duplicates[split] = len(entries) - len(unique)
        ids[split] = unique
        report.sizes[split] = len(unique)
        report.present[split] = hashlib.sha256("\n".join(sorted(unique)).encode()).hexdigest()
    for left, right in itertools.combinations(sorted(ids), 2):
        common = sorted(ids[left] & ids[right])
        if common:
            report.overlaps[f"{left}&{right}"] = common
    if not report.present:
        _log.info("leakage check: no split manifests found in %s", config.path("manifests_dir"))
    else:
        _log.info("leakage check: %s", verdict(report.passed))
    return report
