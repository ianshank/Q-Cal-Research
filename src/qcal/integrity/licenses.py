"""License and provenance audit (replaces the v1 ``license-auditor`` agent).

Checks installed distributions against configured license substrings and denied
packages, scans source files for forbidden imports (e.g. the CC BY-NC-SA oracle
outside its test directory), and validates dataset cards against the allowlist.
"""

from __future__ import annotations

import ast
import importlib.metadata
from dataclasses import dataclass, field
from typing import Any

from qcal.config import Config, ConfigError
from qcal.globs import first_match, iter_files
from qcal.log import get_logger

_log = get_logger("integrity.licenses")
_FRONTMATTER = "---"


@dataclass
class LicenseReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_packages: int = 0
    checked_files: int = 0
    checked_cards: int = 0

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": "PASS" if self.passed else "FAIL",
            "errors": self.errors,
            "warnings": self.warnings,
            "checked": {
                "packages": self.checked_packages,
                "files": self.checked_files,
                "dataset_cards": self.checked_cards,
            },
        }


def metadata_value(dist: importlib.metadata.Distribution, key: str) -> str:
    """First value of a core-metadata field, or ``""`` (no deprecated ``__getitem__`` access)."""
    values = dist.metadata.get_all(key) or []
    return str(values[0]) if values else ""


def distribution_license(dist: importlib.metadata.Distribution) -> str:
    meta = dist.metadata
    parts = [metadata_value(dist, "License-Expression"), metadata_value(dist, "License")]
    parts += [c for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
    return " | ".join(p.strip() for p in parts if p and p.strip())


def check_packages(
    config: Config,
    report: LicenseReport,
    distributions: list[importlib.metadata.Distribution] | None = None,
) -> None:
    deny = config.str_list("licenses.deny_license_substrings")
    warn = config.str_list("licenses.warn_license_substrings")
    denied_packages = {p.lower() for p in config.str_list("licenses.deny_packages")}
    for dist in (
        distributions if distributions is not None else list(importlib.metadata.distributions())
    ):
        name = metadata_value(dist, "Name").lower()
        report.checked_packages += 1
        if name in denied_packages:
            report.errors.append(f"package {name} is denied by policy")
            continue
        text = distribution_license(dist)
        lowered = text.lower()
        hit = next((d for d in deny if d.lower() in lowered), None)
        if hit:
            report.errors.append(f"package {name} has a denied license ({hit}): {text}")
        elif any(w.lower() in lowered for w in warn):
            report.warnings.append(f"package {name} license needs review: {text}")


def imported_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def check_imports(config: Config, report: LicenseReport) -> None:
    rules = config.table_list("licenses.forbidden_imports")
    root = config.root
    for path in iter_files(root, config.str_list("licenses.import_scan_globs")):
        relative = path.relative_to(root).as_posix()
        report.checked_files += 1
        try:
            modules = imported_modules(path.read_text("utf-8"))
        except SyntaxError as exc:
            report.warnings.append(f"{relative}: cannot parse ({exc.msg})")
            continue
        for rule in rules:
            module = rule.get("module")
            if not isinstance(module, str):
                raise ConfigError("licenses.forbidden_imports entries need a string 'module'")
            allowed = [str(g) for g in rule.get("allowed_globs", [])]
            used = any(m == module or m.startswith(f"{module}.") for m in modules)
            if used and first_match(relative, allowed) is None:
                report.errors.append(f"{relative} imports {module}, which policy forbids here")


def read_frontmatter(text: str) -> dict[str, Any]:
    import yaml

    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER:
        return {}
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == _FRONTMATTER)
    except StopIteration:
        return {}
    data = yaml.safe_load("\n".join(lines[1:end])) or {}
    return data if isinstance(data, dict) else {}


def check_dataset_cards(config: Config, report: LicenseReport) -> None:
    directory = config.path("dataset_cards_dir")
    if not directory.is_dir():
        return
    key = config.str_value("licenses.dataset_license_key")
    allowed = config.str_list("licenses.dataset_allowed")
    denied = {d.lower() for d in config.str_list("licenses.dataset_denied")}
    for card in sorted(directory.glob("*.md")):
        report.checked_cards += 1
        license_value = read_frontmatter(card.read_text("utf-8")).get(key)
        name = card.name
        if not isinstance(license_value, str) or not license_value:
            report.errors.append(f"dataset card {name} has no {key!r} in its frontmatter")
        elif license_value.lower() in denied:
            report.errors.append(f"dataset card {name} uses denied license {license_value}")
        elif allowed and license_value not in allowed:
            report.errors.append(
                f"dataset card {name} license {license_value} is not on the allowlist"
            )


def check_licenses(config: Config, *, packages: bool = True) -> LicenseReport:
    report = LicenseReport()
    if packages:
        check_packages(config, report)
    check_imports(config, report)
    check_dataset_cards(config, report)
    _log.info(
        "license check: %s (%d errors, %d warnings)",
        "PASS" if report.passed else "FAIL",
        len(report.errors),
        len(report.warnings),
    )
    return report


__all__ = [
    "LicenseReport",
    "check_licenses",
    "distribution_license",
    "imported_modules",
    "read_frontmatter",
]
