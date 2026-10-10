"""Adversarial review of the hardening: B3, B4 and non-blocking N1, N2, N3.

B1, B2 and B5 are attacks on controls and live in ``tests/security``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from qcal.ci.signatures import TreeIndex, verify_signatures
from qcal.config import Config, load_config
from qcal.integrity.agent_layer import check_agent_layer
from qcal.integrity.claims import check_claims
from qcal.integrity.command_refs import validate_make
from qcal.registry.experiments import load_experiments
from qcal.registry.index import write_index
from qcal.registry.records import RunRecord
from qcal.registry.runner import Runner, RunRefusedError
from qcal.registry.store import RegistryStore
from qcal.registry.tables import TableDataError, build_tables
from tests.conftest import FakeExecutor, experiments_yaml, make_record, run_git, sign, write

SECTION = "paper/sections/results.tex"

# --- B3: numbers that used to hide in strict files ------------------------------------------


@pytest.fixture
def indexed(config: Config) -> Config:
    store = RegistryStore(config.path("registry_dir"))
    store.write(make_record("R1", cell_id="C-a", metrics={"AP": 41.2, "delta": 0.5}))
    write_index(config, store)
    return config


@pytest.mark.parametrize(
    ("line", "number"),
    [
        ("latency drops to 2.3ms", "2.3"),
        ("a 1.5x speedup", "1.5"),
        ("AP improves by +4.1pt", "4.1"),
        ("between 0.03--0.05 on COCO", "0.05"),
        (r"\qcalval{run:R1:AP}{41.2}/97.3 on two splits", "97.3"),
        ("a 3x speedup", "3x"),
        ("only 12 ms per frame", "12 ms"),
    ],
)
def test_b3_numbers_with_units_ranges_and_pairs_are_flagged(
    indexed: Config, line: str, number: str
) -> None:
    write(indexed.root, SECTION, line + "\n")

    messages = [f.message for f in check_claims(indexed)]

    assert f"number {number} has no run reference" in messages, messages


@pytest.mark.parametrize(
    "line",
    [
        r"the gap is $-$\qcalval{run:R1:delta}{0.5}",
        r"the gap is -\qcalval{run:R1:delta}{0.5}",
        r"the gap is \textminus\qcalval{run:R1:delta}{0.5}",
        r"AP reaches 1\qcalval{run:R1:AP}{41.2}",
        r"AP reaches \qcalval{run:R1:AP}{41.2}5",
    ],
)
def test_b3_signs_and_digits_touching_a_tagged_value_are_flagged(
    indexed: Config, line: str
) -> None:
    write(indexed.root, SECTION, line + "\n")

    assert any("changes the number the reader sees" in f.message for f in check_claims(indexed))


@pytest.mark.parametrize(
    "line",
    [
        r"AP is $\qcalval{run:R1:AP}{41.2}$",
        r"Table 2 \qcalval{run:R1:AP}{41.2}",
        r"\qcalval{run:R1:delta}{0.5}--\qcalval{run:R1:delta}{0.5}",
        r"$\qcalval{run:R1:AP}{41.2} \pm \qcalval{run:R1:delta}{0.5}$",
        r"Licensed \texttt{Apache-2.0}; see \includegraphics[width=0.5\linewidth]{fig/1.5x.pdf}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\vspace{2.5pt} and \url{https://arxiv.org/abs/2401.12345}",
    ],
)
def test_b3_legitimate_markup_still_passes(indexed: Config, line: str) -> None:
    write(indexed.root, SECTION, line + "\n")

    assert check_claims(indexed) == []


@pytest.mark.parametrize("relative", ["paper/main.tex", "paper/appendix/a.tex", "README.md"])
def test_b3_every_paper_file_and_the_readme_are_strict(indexed: Config, relative: str) -> None:
    write(indexed.root, relative, "AP improves to 41.0 on COCO\n")

    assert [f.kind for f in check_claims(indexed)] == ["untagged"]


# --- B4: references and tables cover whole cells -------------------------------------------


@pytest.fixture
def seeds(config: Config) -> Config:
    store = RegistryStore(config.path("registry_dir"))
    for run_id, cell, seed, ap in [
        ("R1", "C-a", 0, 40.0),
        ("R2", "C-a", 1, 50.0),
        ("R3", "C-a", 2, 60.0),
        ("R4", "C-b", 0, 30.0),
    ]:
        store.write(make_record(run_id, cell_id=cell, seed=seed, metrics={"AP": ap}))
    write_index(config, store)
    return config


@pytest.mark.parametrize(
    ("ref", "value", "fragment"),
    [
        ("run:R3:AP", "60.0", "must name every current run of C-a (leaves out R1, R2)"),
        ("agg:mean:AP:R2+R3", "55.0", "leaves out R1"),
        ("agg:max:AP:R1+R2", "50.0", "leaves out R3"),
        ("agg:mean:AP:R1+R2+R3+R4", "45.0", "pools runs of cells C-a, C-b"),
    ],
)
def test_b4_references_cannot_pick_seeds_or_pool_cells(
    seeds: Config, ref: str, value: str, fragment: str
) -> None:
    write(seeds.root, SECTION, rf"\qcalval{{{ref}}}{{{value}}}" + "\n")

    messages = [f.message for f in check_claims(seeds)]

    assert any(fragment in m for m in messages), messages


@pytest.mark.parametrize(
    ("ref", "value"),
    [("agg:mean:AP:R1+R2+R3", "50.0"), ("agg:max:AP:R3+R1+R2", "60.0"), ("run:R4:AP", "30.0")],
)
def test_b4_whole_cells_pass(seeds: Config, ref: str, value: str) -> None:
    write(seeds.root, SECTION, rf"\qcalval{{{ref}}}{{{value}}}" + "\n")

    assert check_claims(seeds) == []


def test_b4_pooling_can_be_allowed_explicitly(make_config: Callable[[str], Config]) -> None:
    config = make_config("[tables]\nallow_pooled_cells = true\n")
    store = RegistryStore(config.path("registry_dir"))
    store.write(make_record("R1", cell_id="C-a", metrics={"AP": 40.0}))
    store.write(make_record("R2", cell_id="C-b", metrics={"AP": 50.0}))
    write_index(config, store)
    write(config.root, SECTION, r"\qcalval{agg:mean:AP:R1+R2}{45.0}" + "\n")

    assert check_claims(config) == []


def test_b4_a_table_row_may_not_pool_cells(seeds: Config) -> None:
    write(
        seeds.root,
        "configs/tables/t.toml",
        'name = "t"\nrows = ["detector"]\n[[columns]]\nmetric = "AP"\n',
    )

    with pytest.raises(TableDataError, match="pools cells C-a, C-b"):
        build_tables(seeds)


# --- N1: a merge that only changes a protected file's mode is a change ---------------------


@pytest.mark.requires_ssh_keygen
def test_n1_mode_only_merge_is_caught(signed_repo) -> None:
    repo, key, _old, base = signed_repo
    write(repo, ".claude/hooks/run_hook.sh", "#!/bin/sh\nexit 0\n").chmod(0o755)
    run_git(repo, "add", "-A")
    sign(repo, key, "signed executable hook")
    signed_tip = run_git(repo, "rev-parse", "HEAD")
    run_git(repo, "update-index", "--chmod=-x", ".claude/hooks/run_hook.sh")
    tree = run_git(repo, "write-tree")
    merge = run_git(repo, "commit-tree", tree, "-p", signed_tip, "-p", base, "-m", "merge")

    report = verify_signatures(repo, base, merge, mode="enforce")

    assert not report.passed
    assert TreeIndex(repo).entry(merge, ".claude/hooks/run_hook.sh").startswith("100644 ")


# --- N2: agent-layer gaps ---------------------------------------------------------------------

AGENT = ".claude/agents/a.md"


def test_n2_a_narrowing_disallow_does_not_remove_the_tool(config: Config, repo: Path) -> None:
    write(
        repo,
        AGENT,
        "---\nname: a\ndescription: d\ntools: Bash, WebFetch\ndisallowedTools: Bash(curl:*)\n---\n",
    )

    assert any("combines write-capable" in e for e in check_agent_layer(config).errors)


def test_n2_required_guards_must_be_wired(make_config: Callable[[str], Config], repo: Path) -> None:
    config = make_config('[agent_layer]\nrequired_pretooluse_hooks = ["guard-paths"]\n')
    write(repo, ".claude/settings.json", json.dumps({"hooks": {"Stop": []}}))

    assert check_agent_layer(config).errors == [
        ".claude/settings.json: required PreToolUse hook 'guard-paths' is not wired"
    ]


def test_n2_hook_scripts_outside_the_project_are_reported(config: Config, repo: Path) -> None:
    command = "$CLAUDE_PROJECT_DIR/../outside.sh"
    write(
        repo,
        ".claude/settings.json",
        json.dumps({"hooks": {"Stop": [{"hooks": [{"command": command}]}]}}),
    )

    assert check_agent_layer(config).errors == [
        f".claude/settings.json: Stop hook {command!r} points outside the project"
    ]


@pytest.mark.parametrize(
    ("tokens", "ok"),
    [
        (("-C", "docs", "html"), True),
        (("--directory=docs", "html"), True),
        (("-f", "Other.mk", "check"), True),
        (("-f", "Other.mk", "nope"), False),
    ],
)
def test_n2_make_options_with_values(tokens: tuple[str, ...], *, ok: bool) -> None:
    assert (validate_make({"check"}, tokens) is None) is ok


# --- N3: superseding a completed run needs a recorded reason --------------------------------


@pytest.fixture
def runner(repo: Path) -> Runner:
    write(repo, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    config = load_config(repo, environ={})
    return Runner(
        config,
        load_experiments(config),
        RegistryStore(config.path("registry_dir")),
        FakeExecutor(),
        collectors=[],
    )


def test_n3_supersede_without_a_reason_is_refused(runner: Runner) -> None:
    first = runner.run("C-a", 0)

    with pytest.raises(RunRefusedError, match="needs a reason"):
        runner.run("C-a", 0, supersedes=first.run_id)
    with pytest.raises(RunRefusedError, match="needs a reason"):
        runner.run_batch("C-a", rerun=True)


def test_n3_the_reason_is_recorded(runner: Runner) -> None:
    first = runner.run("C-a", 0)

    second: RunRecord = runner.run("C-a", 0, supersedes=first.run_id, reason=" driver 550 ")

    assert second.provenance["supersede_reason"] == "driver 550"


def test_n3_reason_requirement_is_configurable(make_config: Callable[[str], Config]) -> None:
    config = make_config("[registry]\nrequire_supersede_reason = false\n")
    write(config.root, "EXPERIMENTS.yaml", experiments_yaml([{"id": "C-a"}], seeds=[0]))
    runner = Runner(
        config,
        load_experiments(config),
        RegistryStore(config.path("registry_dir")),
        FakeExecutor(),
        collectors=[],
    )
    first = runner.run("C-a", 0)

    assert runner.run("C-a", 0, supersedes=first.run_id).supersedes == first.run_id
