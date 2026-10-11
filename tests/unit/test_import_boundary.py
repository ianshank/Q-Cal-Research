"""One-way imports: qcal never imports qcal_lab; qcal_lab uses an allowlisted qcal API.

qcal is the signed integrity layer; qcal_lab is agent-owned science code. If qcal imported
qcal_lab, agent-owned code would run inside the trusted checks (integrity.yml runs base qcal
on pull requests). The allowlist makes every new dependency of qcal_lab on qcal a visible,
reviewed change.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.conftest import REPO_ROOT

#: The qcal modules qcal_lab may import. Extending it is a reviewed change.
QCAL_API_FOR_LAB = frozenset(
    {
        "qcal",
        "qcal.components",
        "qcal.config",
        "qcal.globs",
        "qcal.gitutil",
        "qcal.integrity.leakage",
        "qcal.log",
        "qcal.policy",
        "qcal.protocols",
        "qcal.registry.executor",
        "qcal.registry.experiments",
        "qcal.registry.gates",
        "qcal.registry.records",
        "qcal.registry.store",
        "qcal.reports",
    }
)


def imported_modules(package: Path) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text("utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module)
        found[path.relative_to(REPO_ROOT).as_posix()] = names
    return found


def test_qcal_never_imports_qcal_lab() -> None:
    offenders = {
        path: sorted(n for n in names if n == "qcal_lab" or n.startswith("qcal_lab."))
        for path, names in imported_modules(REPO_ROOT / "src/qcal").items()
    }
    assert {p: n for p, n in offenders.items() if n} == {}


def test_qcal_lab_imports_only_the_allowlisted_qcal_api() -> None:
    used = {
        name
        for names in imported_modules(REPO_ROOT / "src/qcal_lab").values()
        for name in names
        if name == "qcal" or name.startswith("qcal.")
    }
    assert used <= QCAL_API_FOR_LAB, sorted(used - QCAL_API_FOR_LAB)
