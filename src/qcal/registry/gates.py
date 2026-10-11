"""What a registered run may read, checked before it starts and after it finishes.

A registered run reads its configuration from committed files: the packaged defaults, the
``qcal.toml`` at ``HEAD`` and the ``EXPERIMENTS.yaml`` at ``HEAD``. Before launch,
:func:`run_input_problems` refuses:

- environment overrides and locator variables pointing elsewhere;
- unknown or mistyped configuration keys;
- a ``--root`` that is not the top level of its git repository (a copy of the inputs in an
  ignored subdirectory would otherwise pass every other check);
- a git repository that git cannot read (it would otherwise look like "no git");
- a ``qcal.toml`` whose *parsed bytes* differ from the ``HEAD`` blob: the configuration in
  memory is what is compared, so restoring the file after it was read changes nothing;
- a pre-registration that is not the configured file, or differs from ``HEAD``.

After the run, :func:`input_mismatches` compares the digests the experiment program reports
for the files it read (``environment.inputs_read``) with the digests taken at launch. That
is a tripwire, not a control: the reporter is the experiment program itself. Files it did
not report are listed in the record (``provenance.inputs_unverified``).

Outside git (a throwaway project) the policy is the work tree's and the record says so
(``policy_source``); ``qcal registry audit --strict`` flags such records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
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
#: ``policy_source`` values. Only ``head`` means the committed policy was verified.
HEAD, WORKTREE, DEFAULTS = "head", "worktree", "defaults"


@dataclass(frozen=True)
class PolicyState:
    """Where the repository policy (``qcal.toml``) came from, and the digest of its bytes.

    ``source`` is ``head`` (a git work tree whose parsed ``qcal.toml`` equals ``HEAD``'s),
    ``worktree`` (no git repository: the policy is the file on disk) or ``defaults`` (no
    ``qcal.toml``). ``problems`` explain why a registered run must not start.
    """

    source: str
    sha256: str | None
    problems: tuple[str, ...] = field(default=())

    @property
    def committed(self) -> bool:
        """Whether ``HEAD`` could be read, so committed files can be compared with it."""
        return self.source != WORKTREE and not self.problems


def _repository_problems(root: Path) -> tuple[bool, list[str]]:
    """``(inside git, problems)`` for a run rooted at ``root``."""
    marker = gitutil.git_marker(root)
    if marker is None:
        return False, []
    top = gitutil.toplevel(root)
    if top is None:
        return True, [
            (
                f"{marker} exists but git cannot read the repository (owner, a broken .git "
                "file, or no git); registered runs need the committed policy"
            )
        ]
    if top != root:
        return True, [
            (
                f"--root {root} is not the top level of its git repository ({top}); "
                "registered runs read the committed files of the repository they belong to"
            )
        ]
    if gitutil.head_sha(root) is None:
        return True, ["the repository has no commit at HEAD"]
    return True, []


def policy_state(config: Config) -> PolicyState:
    """The policy the configuration was parsed from, checked against ``HEAD`` when in git."""
    root = config.root
    parsed = config.repo_sha256
    inside_git, problems = _repository_problems(root)
    if not inside_git:
        return PolicyState(WORKTREE if parsed else DEFAULTS, parsed)
    if not problems and gitutil.blob_sha256("HEAD", REPO_CONFIG_NAME, root) != parsed:
        problems.append(
            f"{REPO_CONFIG_NAME} differs from HEAD; commit it first "
            "(registered runs read the committed policy)"
        )
    state = PolicyState(HEAD if parsed else DEFAULTS, parsed, tuple(problems))
    _log.debug("policy: source=%s sha256=%s problems=%s", state.source, parsed, problems)
    return state


def run_input_problems(
    config: Config, experiments: Experiments, policy: PolicyState | None = None
) -> list[str]:
    """Why a registered run must not start with this configuration; empty when it may."""
    policy = policy or policy_state(config)
    problems = list(environment_problems(config))
    problems += [f"configuration: {p}" for p in config_key_problems(config.data)]
    problems += list(policy.problems)
    configured = config.path("experiments")
    if experiments.path.resolve() != configured.resolve():
        problems.append(
            f"pre-registration {experiments.path} is not the configured {configured}; "
            "registered runs use the configured EXPERIMENTS.yaml (--experiments is for "
            "--dry-run, audit and cells)"
        )
    elif policy.committed:
        relative = _relative(configured, config.root)
        if relative is None:
            problems.append(f"pre-registration {configured} is outside the repository")
        elif gitutil.blob_sha256("HEAD", relative, config.root) != experiments.sha256:
            problems.append(
                f"{relative} differs from HEAD; commit the pre-registration first "
                "(registered runs use the committed one)"
            )
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


def _relative(path: Path, root: Path) -> str | None:
    absolute = path if path.is_absolute() else root / path
    return absolute.relative_to(root).as_posix() if absolute.is_relative_to(root) else None


def launch_digests(
    config: Config,
    experiments: Experiments,
    policy: PolicyState,
    inputs: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The digest of every file a run may read, taken at launch, by relative path.

    The pre-registration and the policy are the bytes already parsed, not a fresh read.
    """
    digests = dict(config_inputs(config) if inputs is None else inputs)
    relative = _relative(experiments.path, config.root)
    if relative is not None and experiments.sha256:
        digests[relative] = experiments.sha256
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


def unverified_inputs(launch: Mapping[str, str], reported: Any) -> list[str]:
    """Launch inputs the experiment did not report reading (all of them if it reported none)."""
    seen = set(reported) if isinstance(reported, Mapping) else set()
    return sorted(set(launch) - seen)
