#!/usr/bin/env python3
"""Map the integrity rules to the tests that enforce them; fail when a rule has none.

Rules are read from the repository's own text, so a rule added there is checked at once:

- ``C<n>``: the numbered list under ``## Integrity`` in CLAUDE.md (``C0`` .. ``C7``);
- ``A<n>``: the top-level bullets of AGENTS.md, numbered from 1.

Tests declare the rules they enforce with ``@pytest.mark.rule("C3", ...)`` or a module-level
``pytestmark``. The scan is static (``ast``), so it needs neither pytest nor the packages.

    python3 scripts/traceability.py            # rule -> tests, as text
    python3 scripts/traceability.py --json
    python3 scripts/traceability.py --check    # exit 1: an untested rule, an unknown id, or a
                                               # known gap that now has tests

``KNOWN_GAPS`` lists rules without a direct test, each with an owner and the reason; the
check keeps the list honest in both directions. Standard library only.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

RULE_MARK: Final = "rule"
CLAUDE_SECTION: Final = "## Integrity"
NUMBERED: Final = re.compile(r"^(\d+)\.\s+(.*)")

#: Rules with no direct test yet: id -> (owner, reason). Remove an entry when its test lands.
KNOWN_GAPS: Final[dict[str, tuple[str, str]]] = {}


def claude_rules(text: str) -> dict[str, str]:
    rules: dict[str, str] = {}
    inside = False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line.startswith(CLAUDE_SECTION)
            continue
        match = NUMBERED.match(line) if inside else None
        if match:
            rules[f"C{match.group(1)}"] = match.group(2).strip()
    return rules


def agents_rules(text: str) -> dict[str, str]:
    bullets = [line[2:].strip() for line in text.splitlines() if line.startswith("- ")]
    return {f"A{i}": bullet for i, bullet in enumerate(bullets, start=1)}


def rules(root: Path) -> dict[str, str]:
    found = claude_rules((root / "CLAUDE.md").read_text("utf-8"))
    found.update(agents_rules((root / "AGENTS.md").read_text("utf-8")))
    return found


def _rule_ids(node: ast.AST) -> list[str]:
    """String arguments of every ``<...>.mark.rule(...)`` call inside ``node``."""
    ids: list[str] = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == RULE_MARK
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "mark"
        ):
            ids += [
                a.value
                for a in call.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            ]
    return ids


def _tests(tree: ast.Module) -> Iterator[tuple[str, list[ast.expr]]]:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            yield node.name, node.decorator_list
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    yield f"{node.name}::{item.name}", [*node.decorator_list, *item.decorator_list]


def tagged_tests(root: Path, tests_dir: str = "tests") -> dict[str, list[str]]:
    """Rule id -> sorted test ids (``path::name``) that declare it."""
    by_rule: dict[str, set[str]] = {}
    for path in sorted((root / tests_dir).rglob("test_*.py")):
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        module_ids = [
            rule_id
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
            for rule_id in _rule_ids(node.value)
        ]
        relative = path.relative_to(root).as_posix()
        for name, decorators in _tests(tree):
            ids = module_ids + [i for d in decorators for i in _rule_ids(d)]
            for rule_id in ids:
                by_rule.setdefault(rule_id, set()).add(f"{relative}::{name}")
    return {k: sorted(v) for k, v in sorted(by_rule.items())}


def problems(known: dict[str, str], tagged: dict[str, list[str]]) -> list[str]:
    found = [f"unknown rule id {r!r} in {tagged[r][0]}" for r in tagged if r not in known]
    for rule_id in known:
        has_tests = bool(tagged.get(rule_id))
        if not has_tests and rule_id not in KNOWN_GAPS:
            found.append(f"{rule_id} has no test: {known[rule_id][:80]}")
        if has_tests and rule_id in KNOWN_GAPS:
            found.append(f"{rule_id} is listed in KNOWN_GAPS but now has tests; remove the entry")
    found += [f"KNOWN_GAPS names unknown rule {r!r}" for r in KNOWN_GAPS if r not in known]
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    known = rules(args.root)
    tagged = tagged_tests(args.root)
    found = problems(known, tagged)
    out = sys.stdout
    if args.json:
        payload = {
            "rules": {
                r: {"text": text, "tests": tagged.get(r, []), "gap": KNOWN_GAPS.get(r)}
                for r, text in known.items()
            },
            "problems": found,
        }
        out.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        for rule_id, text in known.items():
            count = len(tagged.get(rule_id, []))
            gap = f"  [gap: {KNOWN_GAPS[rule_id][0]}]" if rule_id in KNOWN_GAPS else ""
            out.write(f"{rule_id:>3} {count:4d} test(s)  {text[:70]}{gap}\n")
        for problem in found:
            out.write(f"  problem: {problem}\n")
    return 1 if args.check and found else 0


if __name__ == "__main__":
    sys.exit(main())
