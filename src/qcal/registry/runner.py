"""Launch pre-registered runs and record them. No free-form overrides exist by design."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from qcal import gitutil
from qcal.config import Config, run_neutral_environment
from qcal.log import get_logger
from qcal.registry.environment import collect_environment, launcher_identity, recorded_variables
from qcal.registry.executor import ExecutionResult, Executor, RunSpec
from qcal.registry.experiments import Cell, Experiments, ExperimentsError
from qcal.registry.gates import (
    INPUTS_READ_KEY,
    PolicyState,
    config_inputs,
    digest_of,
    input_mismatches,
    launch_digests,
    policy_state,
    run_input_problems,
    unverified_inputs,
)
from qcal.registry.records import (
    SCHEMA_VERSION,
    RecordError,
    RunRecord,
    effective,
    validate_run_id,
)
from qcal.registry.store import MARKER_SUFFIX, RegistryStore, pair_lock_name

_log = get_logger("registry.runner")

Clock = Callable[[], datetime]
#: Environment keys the launcher collected itself; the program's keys keep their own names.
LAUNCHER_PREFIX: Final = "launcher."
#: Where a program key that would fall inside a reserved namespace is kept instead.
REPORTED_PREFIX: Final = "reported."


class RunRefusedError(RuntimeError):
    """A run was refused before execution (unregistered seed, dirty tree, batch too large,
    configuration from the environment or from uncommitted files)."""


@dataclass(frozen=True)
class PlannedRun:
    cell: Cell
    seed: int
    supersedes: str | None = None


@dataclass
class BatchResult:
    planned: list[PlannedRun] = field(default_factory=list)
    completed: list[RunRecord] = field(default_factory=list)
    failed: list[RunRecord] = field(default_factory=list)
    skipped: list[tuple[str, int, str]] = field(default_factory=list)
    batch_id: str | None = None

    @property
    def ok(self) -> bool:
        return not self.failed


class Runner:
    def __init__(
        self,
        config: Config,
        experiments: Experiments,
        store: RegistryStore,
        executor: Executor,
        *,
        clock: Clock | None = None,
        nonce: Callable[[], str] | None = None,
        collectors: Sequence[str] | None = None,
    ) -> None:
        self.config = config
        self.experiments = experiments
        self.store = store
        self.executor = executor
        self.clock = clock or (lambda: datetime.now(tz=UTC))
        nonce_bytes = config.int_value("registry.nonce_bytes")
        self.nonce = nonce or (lambda: secrets.token_hex(nonce_bytes))
        self.collectors = (
            list(collectors)
            if collectors is not None
            else config.str_list("registry.env_collectors")
        )

    # -- identity -----------------------------------------------------------------
    def new_run_id(self, cell_id: str, seed: int, when: datetime) -> str:
        template = self.config.str_value("registry.run_id_template")
        stamp = when.strftime(self.config.str_value("registry.timestamp_format"))
        return template.format(timestamp=stamp, cell_id=cell_id, seed=seed, nonce=self.nonce())

    def config_hash(self, cell: Cell, seed: int, inputs: Mapping[str, str] | None = None) -> str:
        """Hash of everything that defines the run: factors, seed, command, config files."""
        payload = {
            "cell": cell.id,
            "factors": dict(cell.factors),
            "seed": seed,
            "command": self.config.get("executor.command"),
            "inputs": dict(config_inputs(self.config) if inputs is None else inputs),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def check_inputs(self, policy: PolicyState | None = None) -> PolicyState:
        """Refuse a run whose configuration does not come from committed files only."""
        policy = policy or policy_state(self.config)
        problems = run_input_problems(self.config, self.experiments, policy)
        if problems:
            _log.error("refusing a registered run: %s", "; ".join(problems))
            raise RunRefusedError(
                "registered runs read committed configuration only:\n  - " + "\n  - ".join(problems)
            )
        return policy

    def _output_paths(self) -> list[str]:
        """Registry outputs, which a run itself creates, never count as a dirty tree."""
        root = self.config.root
        keys = (
            "registry_dir",
            "index_csv",
            "index_parquet",
            "logs_dir",
            "results_dir",
            "inflight_dir",
        )
        paths = [self.config.path(k) for k in keys]
        return [p.relative_to(root).as_posix() for p in paths if p.is_relative_to(root)]

    # -- single run ---------------------------------------------------------------
    def run(
        self,
        cell_id: str,
        seed: int,
        *,
        supersedes: str | None = None,
        reason: str = "",
        batch_id: str | None = None,
    ) -> RunRecord:
        cell = self.experiments.cell(cell_id)
        allowed = self.experiments.seeds_for(cell)
        if seed not in allowed:
            raise RunRefusedError(
                f"seed {seed} is not pre-registered for {cell_id} (allowed: {list(allowed)})"
            )
        policy = self.check_inputs()
        with self._pair_lock(cell.id, seed):  # from the duplicate check to the record write
            return self._run_locked(
                cell, seed, policy, supersedes=supersedes, reason=reason, batch_id=batch_id
            )

    def _run_locked(
        self,
        cell: Cell,
        seed: int,
        policy: PolicyState,
        *,
        supersedes: str | None,
        reason: str,
        batch_id: str | None,
    ) -> RunRecord:
        retry_of = self._check_not_duplicate(cell.id, seed, supersedes)
        if (
            supersedes
            and not reason.strip()
            and self.config.bool_value("registry.require_supersede_reason")
        ):
            raise RunRefusedError(
                f"superseding {supersedes} needs a reason (--reason); a rerun without one "
                "looks like rerunning until the result is favourable"
            )
        root = self.config.root
        dirty = gitutil.is_dirty(root, self._output_paths())
        if self.config.bool_value("registry.require_clean_tree") and dirty is not False:
            state = "dirty" if dirty else "of unknown state (no git work tree?)"
            raise RunRefusedError(f"working tree is {state} and registry.require_clean_tree is set")
        if dirty:
            _log.warning("working tree is dirty; the record will say git_dirty=true")

        started = self.clock()
        run_id = self.new_run_id(cell.id, seed, started)
        try:
            validate_run_id(run_id)
        except RecordError as exc:
            raise RunRefusedError(f"cannot run {cell.id}: {exc}") from exc
        names = {"run_id": run_id}
        spec = RunSpec(
            run_id=run_id,
            cell=cell,
            seed=seed,
            seed_role=self.experiments.seed_role,
            result_path=self.config.path("results_dir")
            / self.config.str_value("executor.result_filename").format_map(names),
            log_path=self.config.path("logs_dir")
            / self.config.str_value("executor.log_filename").format_map(names),
            root=root,
            python=sys.executable,
        )
        inputs = config_inputs(self.config)  # read once: hashed, digested and compared
        launch = launch_digests(self.config, self.experiments, policy, inputs)
        provenance = {
            "git_sha": gitutil.head_sha(root),
            "git_dirty": dirty,
            "config_hash": self.config_hash(cell, seed, inputs),
            "config_inputs_sha256": digest_of(inputs),
            "config_sources": list(self.config.sources),
            "config_environment": run_neutral_environment(self.config),
            "policy_source": policy.source,
            "policy_sha256": policy.sha256,
            "experiments_sha256": self.experiments.sha256,
            "executor": type(self.executor).__name__,
            "launcher": launcher_identity(root),
            "environment_variables": recorded_variables(
                self.config.str_list("registry.recorded_env_vars")
            ),
            "batch_id": batch_id,
            "retry_of": retry_of,
        }
        if supersedes:
            provenance["supersede_reason"] = reason.strip()
        marker = self._mark_inflight(run_id, cell.id, seed, started)
        try:
            return self._execute_and_record(
                spec, started, provenance, launch, supersedes=supersedes
            )
        finally:
            marker.unlink(missing_ok=True)

    def _execute_and_record(
        self,
        spec: RunSpec,
        started: datetime,
        provenance: dict[str, Any],
        launch: Mapping[str, str],
        *,
        supersedes: str | None,
    ) -> RunRecord:
        run_id, cell, root = spec.run_id, spec.cell, self.config.root
        _log.info("starting %s (cell=%s seed=%s)", run_id, cell.id, spec.seed)
        result, interrupted = self._execute(spec)
        reported = result.environment.get(INPUTS_READ_KEY)
        changed = input_mismatches(launch, reported)
        if changed and result.ok:
            result.error = "; ".join(changed)
            result.failure_kind = "config"
            _log.error("run %s: %s", run_id, result.error)
        provenance["inputs_unverified"] = unverified_inputs(launch, reported)
        finished = self.clock()
        environment = merge_environment(
            collect_environment(
                self.collectors,
                root,
                command_timeout_s=self.config.float_value("registry.env_command_timeout_s"),
            ),
            result.environment,
        )
        if supersedes and not result.ok:
            _log.warning("rerun %s failed; it does not supersede %s", run_id, supersedes)
            supersedes = None
        status_key = "registry.ok_status" if result.ok else "registry.failed_status"
        record = RunRecord(
            run_id=run_id,
            cell_id=cell.id,
            seed=spec.seed,
            seed_role=self.experiments.seed_role,
            status=self.config.str_value(status_key),
            supersedes=supersedes,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            duration_s=round((finished - started).total_seconds(), 3),
            provenance=provenance,
            factors=dict(cell.factors),
            metrics=result.metrics if result.ok else {},
            environment=environment,
            artifacts=tuple(result.artifacts),  # a failed run keeps its partial artifacts
            log_path=_relative(spec.log_path, root),
            error=result.error,
            failure_kind=None if result.ok else (result.failure_kind or "unknown"),
            resources=dict(result.resources),
            schema_version=SCHEMA_VERSION,
        )
        self.store.write(record)
        if not result.ok:
            _log.error("run %s failed (%s): %s", run_id, record.failure_kind, result.error)
        if interrupted is not None:
            raise interrupted
        return record

    # -- concurrency --------------------------------------------------------------
    @contextlib.contextmanager
    def _pair_lock(self, cell_id: str, seed: int) -> Iterator[None]:
        """An advisory lock per (cell, seed), released by the kernel if the launcher dies,
        so two launchers cannot both pass the duplicate check for one pair."""
        directory = self.config.path("inflight_dir")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / pair_lock_name(cell_id, seed)
        with path.open("a", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RunRefusedError(
                    f"{cell_id}@{seed} is being run by another launcher (lock {path})"
                ) from None
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _mark_inflight(self, run_id: str, cell_id: str, seed: int, started: datetime) -> Path:
        """A marker naming the running run; the audit reports markers a crash left behind."""
        marker = self.config.path("inflight_dir") / f"{run_id}{MARKER_SUFFIX}"
        payload = {
            "run_id": run_id,
            "cell_id": cell_id,
            "seed": seed,
            "started_at": started.isoformat(),
            "pid": os.getpid(),
        }
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return marker

    def _execute(self, spec: RunSpec) -> tuple[ExecutionResult, BaseException | None]:
        """Run the program; a crash becomes a failed result, and an interrupt (Ctrl-C,
        ``SystemExit``) is returned so the caller records the run before re-raising it."""
        try:
            return self.executor.execute(spec), None
        except Exception as exc:  # noqa: BLE001 - a crash is recorded as a failed run, never lost
            _log.exception("executor %s raised for %s", type(self.executor).__name__, spec.run_id)
            error = f"executor raised {type(exc).__name__}: {exc}"
            return ExecutionResult(-1, error=error, failure_kind="unknown"), None
        except BaseException as exc:  # noqa: BLE001 - recorded here, re-raised by run()
            _log.error(
                "run %s interrupted (%s); recording it first", spec.run_id, type(exc).__name__
            )
            error = f"interrupted by {type(exc).__name__}"
            return ExecutionResult(-1, error=error, failure_kind="interrupted"), exc

    def _check_not_duplicate(self, cell_id: str, seed: int, supersedes: str | None) -> str | None:
        """One current ok run per (cell, seed): a second one needs an explicit supersede.

        Returns the latest current failed run of the pair, which this run retries (``None``
        when superseding, or when the pair never failed)."""
        ok_status = self.config.str_value("registry.ok_status")
        current = {
            r.run_id: r
            for r in effective(self.store.load_all())
            if (r.cell_id, r.seed) == (cell_id, seed)
        }
        if supersedes is not None:
            if supersedes not in current:
                raise RunRefusedError(
                    f"cannot supersede {supersedes}: it is not a current run of {cell_id}@{seed}"
                )
            return None
        done = sorted(r.run_id for r in current.values() if r.status == ok_status)
        if done:
            raise RunRefusedError(
                f"{cell_id}@{seed} already completed as {done[0]}; "
                "replace it with `qcal registry run --supersedes` or `run-batch --rerun`"
            )
        failed = sorted(current.values(), key=lambda r: (r.started_at, r.run_id))
        return failed[-1].run_id if failed else None

    def new_batch_id(self, when: datetime) -> str:
        template = self.config.str_value("registry.batch_id_template")
        stamp = when.strftime(self.config.str_value("registry.timestamp_format"))
        return template.format(timestamp=stamp, nonce=self.nonce())

    # -- batches ------------------------------------------------------------------
    def plan(
        self, patterns: str, *, seeds: Sequence[int] | None = None, rerun: bool = False
    ) -> tuple[list[PlannedRun], list[tuple[str, int, str]]]:
        cells = self.experiments.match(patterns)
        if not cells:
            raise ExperimentsError(f"no pre-registered cell matches {patterns!r}")
        ok_status = self.config.str_value("registry.ok_status")
        done: dict[tuple[str, int], str] = {
            (r.cell_id, r.seed): r.run_id
            for r in effective(self.store.load_all())
            if r.status == ok_status
        }
        planned: list[PlannedRun] = []
        skipped: list[tuple[str, int, str]] = []
        for cell in cells:
            allowed = self.experiments.seeds_for(cell)
            for seed in dict.fromkeys(seeds) if seeds is not None else allowed:
                if seed not in allowed:
                    raise RunRefusedError(f"seed {seed} is not pre-registered for {cell.id}")
                previous = done.get((cell.id, seed))
                if previous and not rerun:
                    skipped.append((cell.id, seed, f"already completed as {previous}"))
                    continue
                planned.append(PlannedRun(cell, seed, previous if rerun else None))
        return planned, skipped

    def run_batch(
        self,
        patterns: str,
        *,
        seeds: Sequence[int] | None = None,
        max_runs: int | None = None,
        keep_going: bool = False,
        rerun: bool = False,
        dry_run: bool = False,
        reason: str = "",
    ) -> BatchResult:
        planned, skipped = self.plan(patterns, seeds=seeds, rerun=rerun)
        limit = (
            max_runs
            if max_runs is not None
            else self.config.int_value("registry.max_runs_per_batch")
        )
        if len(planned) > limit:
            raise RunRefusedError(
                f"batch would launch {len(planned)} runs, above the limit of {limit}; "
                "narrow the pattern or pass --max-runs explicitly"
            )
        batch = BatchResult(planned=planned, skipped=skipped)
        _log.info("batch %r: %d planned, %d skipped", patterns, len(planned), len(skipped))
        if dry_run:
            return batch
        batch.batch_id = self.new_batch_id(self.clock())
        for item in planned:
            record = self.run(
                item.cell.id,
                item.seed,
                supersedes=item.supersedes,
                reason=reason,
                batch_id=batch.batch_id,
            )
            if record.status == self.config.str_value("registry.ok_status"):
                batch.completed.append(record)
                continue
            batch.failed.append(record)
            if not keep_going:
                _log.error("stopping batch after first failure (%s)", record.run_id)
                break
        return batch


def merge_environment(collected: Mapping[str, Any], reported: Mapping[str, Any]) -> dict[str, Any]:
    """The launcher's values under ``launcher.``, the program's under their own names.

    The two run in different interpreters, so their versions may differ; both are kept. A
    program key inside a reserved namespace is kept under ``reported.``, so nothing the
    program says can pass for a value the launcher collected, and nothing is dropped.
    """
    merged = {f"{LAUNCHER_PREFIX}{key}": value for key, value in collected.items()}
    for key, value in reported.items():
        if str(key).startswith((LAUNCHER_PREFIX, REPORTED_PREFIX)):
            _log.warning(
                "experiment reported %s inside a reserved namespace; kept as %s%s",
                key,
                REPORTED_PREFIX,
                key,
            )
            merged[f"{REPORTED_PREFIX}{key}"] = value
        else:
            merged[key] = value
    return merged


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
