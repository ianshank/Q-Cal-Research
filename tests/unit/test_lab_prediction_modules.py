"""The prediction cache key covers every module that decides prediction bytes (N15).

``PREDICTION_MODULES`` lists the ``qcal_lab`` sources whose digest is in the cache key. This
test follows their imports: every ``qcal_lab`` module they reach must be listed or excluded
with a reason, no exclusion may be stale, and the ``qcal`` modules they reach must be ones
that cannot change the bytes on their own. docs/changes/qcal-foundations.md (F3).
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from qcal_lab import experiment
from qcal_lab.predictions import PredictionsHeader, PredictionSource

PACKAGE = Path(experiment.__file__).resolve().parent
#: qcal modules the prediction code may import, and why each cannot change prediction bytes.
QCAL_ALLOWED = {
    "qcal": "the package itself (version only)",
    "qcal.log": "logging only",
    "qcal.protocols": "the Detection and ImageDetections types the bytes serialise",
    "qcal.config": "settings, which enter the key by value",
    "qcal.registry.executor": "sha256_file, a digest of the inputs",
    "qcal.registry.records": "run-id validation",
    "qcal.components": "the detector registry: names to builders",
    "qcal.registry.gates": "the lab configuration's hashed-input check, not prediction bytes",
}

pytestmark = pytest.mark.rule("C7")


def _module_file(dotted: str) -> Path | None:
    """The source file of a ``qcal_lab`` module, or ``None`` for another package."""
    parts = dotted.split(".")
    if parts[0] != "qcal_lab":
        return None
    base = PACKAGE.joinpath(*parts[1:])
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imports(path: Path) -> set[str]:
    package = ".".join(("qcal_lab", *path.relative_to(PACKAGE).parent.parts))
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)  # submodules
    return found


def _listed(relative: str) -> bool:
    return any(
        relative == entry or relative.startswith(f"{entry}/")
        for entry in experiment.PREDICTION_MODULES
    )


def _closure() -> tuple[set[str], set[str]]:
    """(qcal_lab files, qcal modules) reachable from the prediction modules."""
    start = {
        path
        for entry in experiment.PREDICTION_MODULES
        for path in (
            (PACKAGE / entry).rglob("*.py") if (PACKAGE / entry).is_dir() else [PACKAGE / entry]
        )
    }
    seen: set[Path] = set()
    qcal: set[str] = set()
    todo = list(start)
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in _imports(path):
            if name == "qcal" or name.startswith("qcal."):
                qcal.add(name)
            target = _module_file(name)
            if target is not None:
                todo.append(target)
                # importing a submodule runs every parent package's __init__ first
                parent = target.parent
                while parent != PACKAGE.parent and parent.is_relative_to(PACKAGE):
                    todo.append(parent / "__init__.py")
                    parent = parent.parent
    return {p.relative_to(PACKAGE).as_posix() for p in seen}, qcal


def test_every_reachable_lab_module_is_hashed_or_excluded_with_a_reason() -> None:
    reached, _ = _closure()
    unaccounted = sorted(
        r for r in reached if not _listed(r) and r not in experiment.PREDICTION_MODULE_EXCLUSIONS
    )
    assert unaccounted == [], "add them to PREDICTION_MODULES or PREDICTION_MODULE_EXCLUSIONS"


def test_no_exclusion_is_stale() -> None:
    reached, _ = _closure()
    assert sorted(set(experiment.PREDICTION_MODULE_EXCLUSIONS) - reached) == []
    assert all(reason.strip() for reason in experiment.PREDICTION_MODULE_EXCLUSIONS.values())


def test_the_prediction_code_reaches_only_harmless_qcal_modules() -> None:
    _, qcal = _closure()
    # "from qcal.x import name" also yields "qcal.x.name"; only real modules count
    modules = {name for name in qcal if _is_module(name)}
    assert sorted(modules - set(QCAL_ALLOWED)) == []


def _is_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def test_formats_is_hashed() -> None:
    assert _listed("formats.py")


SOURCE = PredictionSource("fp32", "torch_fp32", "none", "id", "golden")
HEADER = PredictionsHeader("golden", "val", "raw", "d" * 64, "e" * 64, (1, 2), SOURCE)


@pytest.mark.parametrize(
    "change",
    [
        {"detector": "other"},
        {"split": "test"},
        {"dataset_sha256": "f" * 64},
        {"split_sha256": "0" * 64},
        {"category_ids": (1, 3)},
        {"source": dataclasses.replace(SOURCE, precision="int8")},
    ],
    ids=lambda c: next(iter(c)),
)
def test_every_header_field_is_in_the_key(
    change: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from qcal_lab.numerics import Regime

    monkeypatch.setattr(experiment, "prediction_code_digest", lambda: "c" * 64)
    ctx = SimpleNamespace(
        fingerprint={"name": "golden"},
        source=SOURCE,
        regime=Regime("fp32", {}),
        key_parts={},
        dataset=SimpleNamespace(sha256="d" * 64),
        images_dir=None,
    )
    before = experiment.prediction_key(ctx, "e" * 64, ["1"], HEADER)  # type: ignore[arg-type]
    after = experiment.prediction_key(  # type: ignore[arg-type]
        ctx, "e" * 64, ["1"], dataclasses.replace(HEADER, **change)
    )
    assert before != after
