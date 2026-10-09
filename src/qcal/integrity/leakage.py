"""Split-manifest leakage check: hashes per split and pairwise disjointness.

Manifests are plain text, one image id per line, at
``paths.manifests_dir / data.manifest_pattern``. Comment lines start with
``data.comment_prefix``. This is the deterministic half of the
``data-leakage-checker`` agent's job.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Any

from qcal.config import Config
from qcal.log import get_logger

_log = get_logger("integrity.leakage")


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
            "verdict": "PASS" if self.passed else "FAIL",
            "splits": {k: {"sha256": v, "size": self.sizes[k]} for k, v in self.present.items()},
            "missing": self.missing,
            "duplicates": self.duplicates,
            "overlaps": {
                k: {"count": len(v), "examples": v[:10]} for k, v in self.overlaps.items()
            },
        }


def read_manifest(text: str, comment_prefix: str) -> list[str]:
    lines = (line.strip() for line in text.splitlines())
    return [
        line for line in lines if line and not (comment_prefix and line.startswith(comment_prefix))
    ]


def check_leakage(config: Config) -> LeakageReport:
    directory = config.path("manifests_dir")
    pattern = config.str_value("data.manifest_pattern")
    comment = config.str_value("data.comment_prefix")
    report = LeakageReport()
    ids: dict[str, set[str]] = {}
    for split in config.str_list("data.splits"):
        path = directory / pattern.format(split=split)
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
    _log.info("leakage check: %s", "PASS" if report.passed else "FAIL")
    return report
