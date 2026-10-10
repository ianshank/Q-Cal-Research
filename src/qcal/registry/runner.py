"""Launch pre-registered runs and record them. No free-form overrides exist by design."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qcal import gitutil
from qcal.config import Config
from qcal.globs import iter_files
from qcal.log import get_logger
from qcal.registry.environment import collect_environment
from qcal.registry.executor import ExecutionResult, Executor, RunSpec
from qcal.registry.experiments import Cell, Experiments, ExperimentsError
from qcal.registry.records import RecordError, RunRecord, effective, validate_run_id
from qcal.registry.store import RegistryStore

_log = get_logger("registry.runner")

Clock = Callable[[], datetime]


class RunRefusedError(RuntimeError):
    """A run was refused before execution (unregistered seed, dirty tree, batch too large)."""


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

    def config_hash(self, cell: Cell, seed: int) -> str:
        """Hash of everything that defines the run: factors, seed, command, config files."""
        root = self.config.root
        inputs = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in iter_files(root, self.config.str_list("registry.config_hash_inputs"))
        }
        payload = {
            "cell": cell.id,
            "factors": dict(cell.factors),
            "seed": seed,
            "command": self.config.get("executor.command"),
            "inputs": inputs,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _output_paths(self) -> list[str]:
        """Registry outputs, which a run itself creates, never count as a dirty tree."""
        root = self.config.root
        keys = ("registry_dir", "index_csv", "index_parquet", "logs_dir", "results_dir")
        paths = [self.config.path(k) for k in keys]
        return [p.relative_to(root).as_posix() for p in paths if p.is_relative_to(root)]

    # -- single run ---------------------------------------------------------------
    def run(self, cell_id: str, seed: int, *, supersedes: str | None = None) -> RunRecord:
        cell = self.experiments.cell(cell_id)
        allowed = self.experiments.seeds_for(cell)
        if seed not in allowed:
            raise RunRefusedError(
                f"seed {seed} is not pre-registered for {cell_id} (allowed: {list(allowed)})"
            )
        self._check_not_duplicate(cell.id, seed, supersedes)
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
        )
        provenance = {
            "git_sha": gitutil.head_sha(root),
            "git_dirty": dirty,
            "config_hash": self.config_hash(cell, seed),
            "experiments_sha256": self.experiments.sha256,
            "executor": type(self.executor).__name__,
        }
        _log.info("starting %s (cell=%s seed=%s)", run_id, cell.id, seed)
        try:
            result = self.executor.execute(spec)
        except Exception as exc:  # noqa: BLE001 - a crash is recorded as a failed run, never lost
            _log.exception("executor %s raised for %s", type(self.executor).__name__, run_id)
            result = ExecutionResult(-1, error=f"executor raised {type(exc).__name__}: {exc}")
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
            seed=seed,
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
            artifacts=tuple(result.artifacts) if result.ok else (),
            log_path=_relative(spec.log_path, root),
            error=result.error,
            schema_version=self.config.int_value("registry.schema_version"),
        )
        self.store.write(record)
        if not result.ok:
            _log.error("run %s failed: %s", run_id, result.error)
        return record

    def _check_not_duplicate(self, cell_id: str, seed: int, supersedes: str | None) -> None:
        """One current ok run per (cell, seed): a second one needs an explicit supersede."""
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
            return
        done = sorted(r.run_id for r in current.values() if r.status == ok_status)
        if done:
            raise RunRefusedError(
                f"{cell_id}@{seed} already completed as {done[0]}; "
                "replace it with `qcal registry run --supersedes` or `run-batch --rerun`"
            )

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
        for item in planned:
            record = self.run(item.cell.id, item.seed, supersedes=item.supersedes)
            if record.status == self.config.str_value("registry.ok_status"):
                batch.completed.append(record)
                continue
            batch.failed.append(record)
            if not keep_going:
                _log.error("stopping batch after first failure (%s)", record.run_id)
                break
        return batch


def merge_environment(collected: Mapping[str, Any], reported: Mapping[str, Any]) -> dict[str, Any]:
    """Experiment-reported values never overwrite what qcal collected itself."""
    merged = dict(collected)
    for key, value in reported.items():
        if key in collected and collected[key] != value:
            _log.warning(
                "experiment reported %s=%r; keeping collected %r", key, value, collected[key]
            )
            merged[f"reported_{key}"] = value
        else:
            merged[key] = value
    return merged


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
