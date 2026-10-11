"""What a registered run may read, checked before it starts and after it finishes.

A registered run reads its configuration from committed files: the packaged defaults, the
``qcal.toml`` at ``HEAD`` and the configured ``EXPERIMENTS.yaml``. Before launch,
:func:`run_input_problems` refuses environment overrides, locator variables pointing
elsewhere, unknown or mistyped configuration keys, a ``qcal.toml`` that differs from
``HEAD``, and a pre-registration file other than the configured one. After the run,
:func:`input_mismatches` compares the digests the experiment program reports for the files
it read (``environment.inputs_read``) with the digests taken at launch, so a file that
changed while the run was in flight fails the run.

The run's record carries :class:`PolicyState` and :func:`config_inputs`, so an auditor can
tell which committed policy and which configuration files produced it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from qcal import gitutil
from qcal.config import REPO_CONFIG_NAME, Config, config_key_problems, environment_problems
from qcal.globs import iter_files
from qcal.log import get_logger
from qcal.registry.experiments import Experiments

_log = get_logger("registry.gates")

#: The key an experiment program uses in its result ``environment`` to report the digest of
#: every configuration file it read, by repository-relative path.
INPUTS_READ_KEY: Final = "inputs_read"


@dataclass(frozen=True)
class PolicyState:
    """Where the repository policy (``qcal.toml``) came from, and its digest.

    ``source`` is ``head`` (a git work tree whose ``qcal.toml`` equals ``HEAD``'s),
    ``worktree`` (no git ``HEAD`` to compare with, as in the smoke test's project) or
    ``defaults`` (no ``qcal.toml`` at all). ``problem`` explains why a run must not start.
    """

    source: str
    sha256: str | None
    problem: str | None = None


def policy_state(config: Config) -> PolicyState:
    root = config.root
    path = root / REPO_CONFIG_NAME
    worktree = path.read_bytes() if path.is_file() else None
    digest = hashlib.sha256(worktree).hexdigest() if worktree is not None else None
    if gitutil.head_sha(root) is None:
        return PolicyState("worktree" if worktree is not None else "defaults", digest)
    committed = gitutil.show_file("HEAD", REPO_CONFIG_NAME, root)
    current = worktree.decode("utf-8", errors="replace") if worktree is not None else None
    if committed != current:
        return PolicyState(
            "head",
            digest,
            f"{REPO_CONFIG_NAME} differs from HEAD; commit it first "
            "(registered runs read the committed policy)",
        )
    return PolicyState("head", digest)


def run_input_problems(config: Config, experiments: Experiments) -> list[str]:
    """Why a registered run must not start with this configuration; empty when it may."""
    problems = list(environment_problems(config))
    problems += [f"configuration: {p}" for p in config_key_problems(config.data)]
    configured = config.path("experiments")
    if experiments.path.resolve() != configured.resolve():
        problems.append(
            f"pre-registration {experiments.path} is not the configured {configured}; "
            "registered runs use the configured EXPERIMENTS.yaml (--experiments is for "
            "--dry-run, audit and cells)"
        )
    state = policy_state(config)
    if state.problem:
        problems.append(state.problem)
    return problems


def config_inputs(config: Config) -> dict[str, str]:
    """``registry.config_hash_inputs`` files by repository-relative path, with their sha256."""
    root = config.root
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in iter_files(root, config.str_list("registry.config_hash_inputs"))
    }


def digest_of(mapping: Mapping[str, Any]) -> str:
    canonical = json.dumps(mapping, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def launch_digests(config: Config, experiments: Experiments, policy: PolicyState) -> dict[str, str]:
    """The digest of every file a run may read, taken at launch, by relative path."""
    root = config.root
    digests = dict(config_inputs(config))
    path = experiments.path.resolve()
    if path.is_relative_to(root) and experiments.sha256:
        digests[path.relative_to(root).as_posix()] = experiments.sha256
    if policy.sha256:
        digests[REPO_CONFIG_NAME] = policy.sha256
    return digests


def input_mismatches(launch: Mapping[str, str], reported: Any) -> list[str]:
    """Files the experiment read whose digest differs from the one taken at launch."""
    if not isinstance(reported, Mapping):
        return []
    return [
        f"{path} changed while the run was in flight"
        for path, digest in sorted(reported.items())
        if path in launch and launch[path] != digest
    ]
