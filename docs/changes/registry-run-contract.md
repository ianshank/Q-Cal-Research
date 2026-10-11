# change: registry-run-contract

## Status
Proposed, Oct 11, 2026. Edits the enforcement surface (`src/qcal/`, `qcal.toml`, the packaged
defaults), so it needs Ian's signed commits. It is meant for signing session 1 (Oct 17),
together with `docs/changes/registry-run-gates.md` (PR-A1). Cycle plan:
`docs/changes/cycle-2026-10-g1-readiness.md` (PR-A2). Its commits sit next to the A1 commits
on draft PR #4, so `qcal ci verify-signatures` lists them together.

## Gate served
G1/K1 (Nov 8). Before the first registered run writes a record, the record must be able to
say:
- which interpreter and GPU ran the program;
- how the run failed, if it failed;
- what it cost.

Records are immutable, so any field added later is missing from every earlier record.

## Why
Verified against `8c76044`:

- **Interpreter.** The executor runs whatever `python3` is first on `PATH` (`qcal.toml:44`), not
  the launcher's interpreter. Nothing records which one ran.
- **Environment is collected in the wrong process, at the wrong time.**
  - Collectors run in the launcher after the child exits (`runner.py:205-213`).
  - The `nvidia` collector joins every GPU's name into one string and keeps only the first
    driver (`environment.py:114-123`).
  - `CUDA_VISIBLE_DEVICES` is recorded nowhere.
  - Launcher and program keys share one namespace; on a collision the program's value is
    renamed (`runner.py:328-339`).
- **Failure handling.**
  - `executor.timeout_s = 0` means no timeout.
  - A timeout kills only the direct child (`executor.py:171-185`), so its children survive and
    keep writing to the log.
  - Ctrl-C in the launcher writes no record (`runner.py:194-198` catches only `Exception`).
  - A failed run keeps neither the program's environment nor its partial artifacts. The child's
    result has no status field, and it is not read after a non-zero exit.
- **Cost.** A record has `duration_s` only: no CPU time, memory, GPU time or cache use.
- **Bookkeeping.**
  - No batch id.
  - A rerun of a failed run is not linked to it.
  - No lock per (cell, seed): two launchers can both pass the duplicate check.
- **Identity.**
  - Nothing checks that the `qcal` running a registered run is the repository's own.
  - Tables average seeds without checking that they ran under the same configuration files,
    and they report a spread over seeds that drew nothing.
- **Datasets.** `data.manifest_pattern` cannot name a dataset (`{split}` only), which Phase 2's
  shifted datasets need.

## What changes
Each step is a separate commit:

1. **Interpreter, launcher namespace, GPU inventory, code location.**
   - A `{python}` placeholder resolves to the launcher's interpreter, and `qcal.toml` uses it.
     `config_hash` still hashes the template, so it stays the same across machines.
   - `provenance.launcher` records the interpreter, its version, and the `qcal` version and
     location.
   - Launcher collectors write under `launcher.`, and `nvidia` lists every GPU (index, uuid,
     PCI bus id, name, compute capability, memory, driver).
   - `registry.recorded_env_vars` lists the variables recorded verbatim, null when unset:
     `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`, `NVIDIA_TF32_OVERRIDE`,
     `CUBLAS_WORKSPACE_CONFIG`, `PYTORCH_CUDA_ALLOC_CONF`, `CUDA_MODULE_LOADING`,
     `LD_LIBRARY_PATH` and `OMP_NUM_THREADS`.
   - A program key that would collide with the launcher namespace is kept under `reported.`;
     nothing is dropped.
   - `registry.require_code_in_root`: a registered run refuses unless `qcal` is imported from
     `<root>/src`. It is off in the packaged defaults and on in `qcal.toml`.
2. **Failure handling and the result envelope.**
   - The program runs in its own process group. On a timeout or a launcher exception the group
     gets SIGTERM, then SIGKILL after `executor.kill_grace_s`.
   - The program writes a result envelope (`qcal.executor_result`, version 1) on every exit:
     status, failure kind, error, metrics, artifacts and environment.
   - The envelope is read on a non-zero exit too. A failed run keeps its partial artifacts and
     the program's environment, never its metrics.
   - `RunRecord.failure_kind` takes one of: timeout, cuda_oom, host_oom, cuda_error, plan,
     config, interrupted, unknown.
   - A record is written on any `BaseException`, then the exception is re-raised.
3. **Bookkeeping.**
   - `provenance.batch_id` is shared by the runs of one batch.
   - `provenance.retry_of` names the latest failed run of the same cell and seed.
   - A file lock per (cell, seed) is held from the duplicate check to the record write, and an
     in-flight marker under `paths.inflight_dir` is reported as stale by the audit.
   - An interrupted batch exits 130.
4. **Resources.**
   - `RunRecord.resources`, flattened as `res.*`, holds wall time, CPU user and system time,
     and peak memory from `os.wait4`.
   - It also holds the device and cache counts the program reports, through an allowlist.
   - `cloud_usd` is reserved.
5. **Identity, tables, manifests.**
   - The index gains `config_inputs_sha256`, `failure_kind` and `batch_id` as core columns.
   - Tables refuse to aggregate runs of one cell that ran under different configuration files.
     They also refuse a spread (std, or a CI later) over seeds whose draw changed nothing,
     which the program reports as `seed_effective`.
   - `data.manifest_pattern` may use `{dataset}` through one shared `manifest_path`.
   - `qcal.toml` narrows the lab hash input to the lab defaults file, so unhashed tooling
     settings can live next to it.
   - Table specs (`paths.table_specs_dir`) are no longer run inputs, even under
     `configs/**`. They say how results are shown, so adding a table after the runs must not
     make every earlier run look stale. The end-to-end test found this once tables began
     refusing mixed inputs.
   - `environment.container_image`, `inputs[]` and `environment.cost` are reserved.

## Decisions
- **The launcher's interpreter, not `python3`.** Alternative: an absolute interpreter path in
  `qcal.toml`. Rejected: it is hashed, so the same run would hash differently on every machine.
- **Prefix, do not merge, launcher keys.** The launcher and the program run in different
  interpreters, so their versions can legitimately differ. Keeping both under separate names
  records the difference instead of hiding it.
- **The device actually used comes from the program.** `nvidia-smi` orders GPUs by PCI bus,
  while CUDA by default orders them fastest first. So the launcher records the inventory and
  the CUDA variables, and the lab program records the device it used (PR-D1).
- **A process group, not the child alone.** MMDetection and DataLoader workers are
  grandchildren; killing only the child left them running.
- **Fail closed on a malformed envelope.** A result without the declared format and version is
  a failed run, so a program that predates the envelope cannot record a success by accident.
- **Locks are advisory files** (`fcntl.flock`), released by the kernel if the launcher dies.
  Alternative: a database. Rejected: the registry is plain files by design.
- **Tables refuse rather than warn.** A mean over runs with different configuration files, or a
  spread over identical runs, is a wrong number. The run-input audit already flags the first;
  tables now refuse both.

## Compatibility
- Records: schema 1 only gains optional fields (`failure_kind`, `resources`, provenance keys).
  Launcher environment keys move under `launcher.`. No registered record exists yet.
- Index: new core columns and `res.*` columns; `env.*` columns from the launcher become
  `env.launcher.*`.
- Config keys added:
  - `executor.kill_grace_s`, `executor.poll_interval_s`;
  - `registry.recorded_env_vars`, `registry.require_code_in_root`, `registry.batch_id_template`;
  - `paths.inflight_dir`, `data.datasets`;
  - `registry.column_prefixes.resources`.

  None is removed.
- Executor: programs must write the envelope. The lab program and the test scripts do, in the
  same commit.
- CLI: an interrupted run or batch exits 130 after writing its record.

## Out of scope
- Automatic retries (never: a retry is a new, linked run).
- A budget report (report-only, later).
- Artifact storage (PR-G).
- The device-used fields, which the lab program reports (PR-D1).

## Adversarial review (advisory) and what changed
`adversarial-reviewer` on the wave-2 head (`f57816f`): **block**. The full report, with how
each finding was handled, is in `review/claude/claude-sdlc-agents-implementation-plan-gmb29t.md`.
The launcher's three blocking findings, each fixed with a test that fails without its fix:
- **B1, a record vouching for bytes the program never wrote.** The launcher hashed artifacts
  only when it wrote the record, possibly hours after the program wrote them. The program now
  reports each artifact's sha256 in the envelope (optional key, `[0-9a-f]{64}`). A file that
  differs fails the run, and a failed run drops it from its partial artifacts.
- **B5, an interrupt after the program finished.** A Ctrl-C while the collectors ran (torch,
  `nvidia-smi`, git) lost the record, and the `finally` deleted the in-flight marker. SIGINT,
  SIGTERM and SIGHUP are now held from the program's exit to the record write, then delivered.
  They are held by swapping handlers, not by a thread mask, which other live threads defeat.
  The marker is removed only once the record exists.
- **B6, a launcher killed by `kill` or a closed terminal.** The program runs in its own
  session, so it outlived the launcher unrecorded and kept the GPU. While a program runs,
  SIGTERM and SIGHUP now raise `LauncherSignal` (a `KeyboardInterrupt`): the group is
  terminated, the run recorded, and the launcher exits 130. SIGKILL of the launcher still
  orphans the program; `PR_SET_PDEATHSIG` would cover it and is left out (it needs `ctypes`
  and a pre-exec hook).

Non-blocking findings applied:
- N6: envelopes must be self-consistent (an ok status with a failure kind or error is
  refused; a boolean version is refused; resource figures are finite numbers, strings or
  null).
- N7: processes the program leaves in its group are terminated before artifacts are hashed.
- N16: `executor.kill_grace_s` and `poll_interval_s` must be positive.
- N5: an empty `data.datasets` is a configuration error.

Follow-ups, not in this change:
- N1: an interrupted run drops the program's envelope (partial artifacts, environment).
- N4: tables compare only `config_inputs_sha256`, not the policy and pre-registration
  digests the audit also compares.
- N5: the leakage check and split reads use the first dataset only; Phase 2 needs per-dataset
  manifests read by the cells that use them.
- N8: the audit's lock probe can make a launcher's non-blocking lock fail.
- N12: a table-spec directory covering `configs/` can exclude the lab file from the hash
  while `require_hashed` accepts it.

## Ian decisions requested
1. Sign this change in session 1 (Oct 17).
2. Set `executor.timeout_s` in `qcal.toml`; 14400 (4 h) is suggested. Agents do not set it.
3. Pin GPUs by UUID in `CUDA_VISIBLE_DEVICES`, or set `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the
   recorded inventory and the device used agree. CLAUDE.md's Compute section would say so; that
   wording is Ian's.
4. Accept that records carry `LD_LIBRARY_PATH` and the host name. Both expose local paths if
   the repository becomes public (D7). The alternative is to record their digests.

## Ian hours
About 1.5 hours to read the diff and sign.

## Acceptance
- [ ] `make check` green (lint, types, tests with coverage gate)
- [ ] `make smoke` passes with the envelope and `{python}`
- [ ] a timed-out run kills its grandchildren and records `failure_kind = timeout` with the
      program's environment
- [ ] Ctrl-C during a run leaves a record
- [ ] two launchers racing for one (cell, seed) produce one run
- [ ] tables refuse a cell mixing configuration digests
- [ ] advisory adversarial review saved under `review/claude/<branch-slug>.md`
- [ ] Ian's signed commit; RESEARCH_LOG.md entry (Ian)
- [ ] the other model's review at `review/gemini/<branch-slug>.md`

## Tests
- unit: `test_executor.py`, `test_runner.py`, `test_environment.py`, `test_records.py`,
  `test_index.py`, `test_tables.py`, `test_audit.py`, `test_run_gates.py`, `test_leakage.py`
- integration: process-group kill, SIGTERM escalation, a lock race, `KeyboardInterrupt`
- security: `qcal` imported from outside the root is refused
- regression: `tests/regression/test_registry_run_contract.py`, one test per gap above

## Tasks
Proposal → interpreter and environment → failure handling → bookkeeping → resources →
identity and tables, each a commit with its tests; advisory review; Ian signs in session 1.
