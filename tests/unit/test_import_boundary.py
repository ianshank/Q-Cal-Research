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
        "qcal.cli",  # smoke.py drives `qcal registry ...` in-process, as a user would
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


def _is_local_module(dotted: str) -> bool:
    """Whether ``dotted`` names a module or package under ``src/``."""
    base = REPO_ROOT / "src" / Path(*dotted.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def modules_in(tree: ast.AST) -> set[str]:
    """Every absolute module an AST imports, including submodules named by a from-import."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            # `from qcal import cli` imports the module qcal.cli; an ordinary attribute
            # (`from qcal.config import load_config`) is not a module and is not added.
            names |= {
                f"{node.module}.{alias.name}"
                for alias in node.names
                if _is_local_module(f"{node.module}.{alias.name}")
            }
    return names


def imported_modules(package: Path) -> dict[str, set[str]]:
    return {
        path.relative_to(REPO_ROOT).as_posix(): modules_in(ast.parse(path.read_text("utf-8")))
        for path in sorted(package.rglob("*.py"))
    }


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


def test_a_from_import_of_a_submodule_counts_as_that_module() -> None:
    """`from qcal import cli` depends on qcal.cli, not merely on qcal (Copilot, PR #4)."""
    source = (
        "from qcal import cli as c\n"
        "from qcal import gitutil\n"
        "from qcal.config import load_config\n"
        "import qcal.log\n"
    )
    assert modules_in(ast.parse(source)) == {
        "qcal",
        "qcal.cli",
        "qcal.gitutil",
        "qcal.config",  # load_config is an attribute, not a module
        "qcal.log",
    }
