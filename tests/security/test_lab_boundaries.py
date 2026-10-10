"""Phase 1 boundaries: the science code cannot fabricate, leak test data, or reach Ian's files."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from qcal.config import Config, load_config
from qcal.hooks import guards
from qcal.hooks.payload import HookPayload
from qcal.policy import Policy
from qcal_lab.data.fixture import NotFixtureError
from qcal_lab.experiment import PlanError, resolve_plan, split_roles
from qcal_lab.fixture_eval import FixtureEvalLoop
from qcal_lab.models import build_detector
from tests.conftest import REPO_ROOT
from tests.lab_support import ground_truth, lab_config

LAB_SRC = REPO_ROOT / "src/qcal_lab"


def test_fixture_stand_ins_refuse_a_real_dataset(tmp_path: Path) -> None:
    """Even a misconfigured lab file cannot score a real dataset with the stand-ins."""
    from qcal_lab.data.fixture import FIXTURE_DESCRIPTION

    real = ground_truth(description=FIXTURE_DESCRIPTION)  # a copied label is not enough
    with pytest.raises(NotFixtureError):
        build_detector(lab_config(tmp_path), "fixture", real, precision="fp32")
    loop = FixtureEvalLoop()
    for call in (
        lambda: loop.targets([], real),
        lambda: loop.metrics([], real),
        lambda: loop.threshold_objective([], real, label=0, stage="operating"),
    ):
        with pytest.raises(NotFixtureError):
            call()


@pytest.mark.parametrize("role", ["fit", "select"])
def test_no_configuration_lets_the_test_split_fit_or_select(
    repo: Path, tmp_path: Path, role: str
) -> None:
    lab = lab_config(tmp_path, f'[splits]\n{role} = "test"\n')
    with pytest.raises(PlanError, match="tuning on the test split"):
        split_roles(lab, load_config(repo, environ={}))


def test_a_cell_cannot_smuggle_an_override_through_a_factor(tmp_path: Path) -> None:
    lab = lab_config(tmp_path)
    for factor in ({"eval_loop": "x"}, {"splits.evaluate": "val"}, {"grid_step": 0.5}):
        with pytest.raises(PlanError, match="does not understand"):
            resolve_plan({"detector": "atss_r50", **factor}, lab)


def _modules() -> list[tuple[Path, ast.Module]]:
    return [(p, ast.parse(p.read_text(), str(p))) for p in sorted(LAB_SRC.rglob("*.py"))]


def test_science_code_never_imports_the_oracle_or_writes_through_the_registry_store() -> None:
    banned = {"detection_calibration", "qcal.registry.store"}
    allowed_store_readers = {LAB_SRC / "smoke.py"}  # reads the throwaway project's records
    for path, tree in _modules():
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not name.startswith("detection_calibration"), f"{path}: imports the oracle"
                if name in banned and path not in allowed_store_readers:
                    pytest.fail(f"{path}: imports {name}")


def test_only_ians_files_or_the_fixture_stand_in_can_be_the_evaluation_loop(
    repo_config: Config,
) -> None:
    from qcal.config import ConfigError
    from qcal_lab.evaluation import load_eval_loop

    for module in ("qcal_lab.calib.platt", "qcal_lab.experiment", "json"):
        lab = lab_config(REPO_ROOT, f'[eval_loop]\nmodule = "{module}"\nfactory = "build"\n')
        with pytest.raises(ConfigError, match="not one of Ian's hand-written files"):
            load_eval_loop(lab, repo_config)


def test_no_handwritten_module_is_shipped_by_agents() -> None:
    assert not (LAB_SRC / "handwritten").exists() or all(
        p.name == "__pycache__" for p in (LAB_SRC / "handwritten").iterdir()
    ), "src/qcal_lab/handwritten/ is Ian's; agents never create files there"


@pytest.fixture
def repo_config() -> Config:
    return load_config(REPO_ROOT, environ={})


def _decide(hook: str, path: str, args: list[str], config: Config) -> bool:
    payload = HookPayload.parse(
        json.dumps({"tool_name": "Write", "tool_input": {"file_path": str(REPO_ROOT / path)}})
    )
    policy = Policy.from_config(config)
    if hook == "scope-write":
        decision = guards.scope_write(payload, prefixes=args, config=config, root=REPO_ROOT)
    else:
        decision = guards.guard_paths(payload, config=config, policy=policy, root=REPO_ROOT)
    return decision.allow


def _agent_hooks(agent: str) -> dict[str, tuple[str, list[str]]]:
    """matcher -> (hook name, arguments) for an agent's frontmatter PreToolUse hooks."""
    import shlex

    import yaml

    text = (REPO_ROOT / f".claude/agents/{agent}.md").read_text()
    meta = yaml.safe_load(text.split("---")[1])
    hooks: dict[str, tuple[str, list[str]]] = {}
    for group in meta["hooks"]["PreToolUse"]:
        wrapper, hook, *args = shlex.split(group["hooks"][0]["command"])
        assert wrapper.endswith("/.claude/hooks/run_hook.sh")
        hooks[group["matcher"]] = (hook, args)
    return hooks


def _agent_hook_args(agent: str) -> list[str]:
    return next(iter(_agent_hooks(agent).values()))[1]


def test_paper_reproducer_writes_only_science_code_tests_and_configs(repo_config: Config) -> None:
    args = _agent_hook_args("paper-reproducer")
    assert args == ["src/qcal_lab", "tests", "configs"]
    for path in ("src/qcal_lab/calib/platt.py", "tests/unit/test_x.py", "configs/lab.toml"):
        assert _decide("scope-write", path, args, repo_config)
    for path in ("src/qcal/config.py", ".claude/settings.json", "EXPERIMENTS.yaml", "paper/a.tex"):
        assert not _decide("scope-write", path, args, repo_config)


@pytest.mark.parametrize(
    "path",
    [
        "tests/parity/fixtures/isotonic_case.json",  # the oracle it is graded against
        "docs/reference/kuzucu_eccv24.md",  # gate-2 published values
        "data/manifests/test.txt",  # the split decision
        "runs/cache/predictions/abc.jsonl",  # predictions that feed registered runs
        "runs/cache/predictions/abc.meta.json",
    ],
)
def test_no_agent_may_write_the_oracle_splits_references_or_cache(
    repo_config: Config, path: str
) -> None:
    assert not _decide("guard-paths", path, [], repo_config)
    policy = Policy.from_config(repo_config)
    signed = repo_config.str_list("signing.signed_categories")
    assert policy.in_categories(path, signed), "only Ian's signed commits may change it"


def test_paper_reproducer_bash_is_an_allow_list() -> None:
    from qcal.hooks.guards import allow_only

    hook, commands = _agent_hooks("paper-reproducer")["Bash"]
    assert hook == "allow-only"
    for command, allowed in [
        ("make smoke", True),
        ("make parity", True),
        ("git status", True),
        ('python -c \'open("tests/parity/fixtures/x.json", "w")\'', False),
        ("make smoke && rm -rf runs", False),
        ("curl https://example.com", False),
    ]:
        payload = HookPayload.parse(
            json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
        )
        assert allow_only(payload, commands=commands).allow is allowed, command


def test_handwritten_files_stay_ians_even_inside_the_reproducers_scope(repo_config: Config) -> None:
    path = "src/qcal_lab/handwritten/eval_loop.py"
    assert _decide("scope-write", path, ["src/qcal_lab"], repo_config)
    assert not _decide("guard-paths", path, [], repo_config)  # the session guard still refuses


def test_prior_art_scout_cannot_read_ians_documents(repo_config: Config) -> None:
    assert _agent_hook_args("prior-art-scout") == ["ian_only"]
    policy = Policy.from_config(repo_config)
    for path in ("EXPERIMENTS.yaml", "DECISIONS.md", "src/qcal_lab/handwritten/laece.py"):
        payload = HookPayload.parse(
            json.dumps({"tool_name": "Read", "tool_input": {"file_path": str(REPO_ROOT / path)}})
        )
        decision = guards.deny_read(
            payload, targets=["ian_only"], config=repo_config, policy=policy, root=REPO_ROOT
        )
        assert not decision.allow, path
