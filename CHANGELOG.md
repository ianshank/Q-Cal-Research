# Changelog

All notable changes to the `qcal` tooling and the agent layer. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package follows
[Semantic Versioning](https://semver.org/); the version lives in `qcal.__version__`.

## [Unreleased]

### Cycle 2026-10: G0 → G1 readiness

Plan: `docs/changes/cycle-2026-10-g1-readiness.md` (reviewed by the adversarial reviewer and a
four-lens expert panel).

Process (PR-0):
- The cycle plan is committed as a change document; `docs/changes/README.md` indexes every
  proposal; the template gains Status, Decisions, Compatibility, Ian decisions requested and
  Ian hours.
- `SECURITY.md` (private reporting; what counts as a bypass) and `CONTRIBUTING.md` (the
  workflow, how Ian signs an agent branch, what the hooks tell agents, the test suites).
- `docs/AGENT_LAYER.md` lists the scoped hooks each subagent carries, `allow-only` included.

Proposed, for Ian to accept by Oct 17: `docs/changes/evalloop-v2.md`, a two-layer evaluation
interface (matching once, then objectives and metrics), three detection sets, a bootstrap
over resampled datasets (copies under fresh ids, no weights), and a replay detector.

Registered-run gates (`docs/changes/registry-run-gates.md`, signed surface):
- A registered run reads committed configuration only. `qcal registry run|run-batch` refuses
  `QCAL__*` overrides (except `logging`), a `QCAL_CONFIG`/`QCAL_ROOT` that points away from
  the repository, a `qcal.toml` that differs from `HEAD`, unknown or mistyped configuration
  keys, and an `--experiments` file other than the configured one (`--dry-run` excepted).
- The experiment program's environment no longer carries configuration variables; it reports
  the digest of every configuration file it read, and a file that changed during the run
  fails it.
- Records gain `provenance.{config_inputs_sha256, config_sources, config_environment,
  policy_source, policy_sha256}`; `qcal registry audit --strict` fails when a cell's seeds ran
  under different configuration files.
- Git runs without redirecting `GIT_*` variables, with replacement objects ignored and
  `core.fsmonitor` off.
- `qcal config --check` rejects unknown, removed and mistyped keys (run by `make integrity`
  and CI); `qcal config` shows the environment inputs.
- The record schema version is a code constant (`records.SCHEMA_VERSION`); records claiming a
  newer schema are refused. `registry.schema_version` is removed from the defaults.
- After Copilot's review: every `registry.config_hash_inputs` file must equal `HEAD`, read
  once for the gate and the hash; a project outside git is never compared with `HEAD`; the
  fields of list-of-table keys are type-checked.
- After the adversarial review: the run root must be the repository's top level;
  `EXPERIMENTS.yaml` must equal `HEAD`; the parsed bytes, not a later read, are compared;
  the registry's git queries are isolated from user and system git configuration; a
  repository git cannot read is refused; `audit --strict` fails on records whose policy was
  not verified (`uncommitted_policy`); category names in policy lists must exist.

Registered-run contract (`docs/changes/registry-run-contract.md`, signed surface, PR-A2):
- The experiment command names `{python}`, the launcher's own interpreter, run with `-I`;
  `config_hash` still hashes the template. `provenance.launcher` records the interpreter and
  the `qcal` that launched; `registry.require_code_in_root` refuses a `qcal` imported from
  outside `paths.source_dir`.
- Launcher environment keys live under `launcher.`; a program key that would collide is kept
  under `reported.`. The `nvidia` collector lists every GPU (index, uuid, PCI bus, name,
  capability, memory, driver). `registry.recorded_env_vars` records CUDA and threading
  variables verbatim, null when unset.
- The program runs in its own process group; a timeout or a launcher exception sends SIGTERM
  to the group, then SIGKILL after `executor.kill_grace_s`, and a second interrupt kills at
  once. The program writes a `qcal.executor_result` v1 envelope on every exit; it is read on
  a non-zero exit too, and a malformed one fails the run. Records gain `failure_kind`
  (timeout, cuda_oom, host_oom, cuda_error, plan, config, interrupted, unknown) and are
  written on any `BaseException`, after which it is re-raised. An interrupted batch exits 130.
- One `flock` per (cell, seed) from the duplicate check to the record; in-flight markers under
  `paths.inflight_dir`, reported as stale by the audit; `provenance.batch_id` and `retry_of`.
- `RunRecord.resources` (`res.*` columns): wall and CPU time and peak memory from `wait4`, and
  the device and cache counts the program reports through an allowlist.
- Index columns `config_inputs_sha256`, `batch_id`, `failure_kind`. Tables and claims refuse
  to average runs of one row under different configuration files, and refuse a spread over
  seeds the program reports as `seed_effective = false`. Table specs are no longer run inputs.
- `data.manifest_pattern` may name `{dataset}` through one shared `leakage.manifest_path`.
- guard-bash no longer crashes on a segment that is only a redirection.

Run identity and formats (`docs/changes/run-identity-and-formats.md`, agent-owned, PR-D1):
- Golden digests pin every run identity (`tests/regression/test_run_identity_goldens.py`);
  each re-pin names the proposal step that moved it.
- `target`, `quant_path`, `fit_precision`, `threshold_regime` and `split_design` are roles;
  the factor tables may name roles only; a detector kind declares the targets it produces.
  Smoke and fixture-loop settings move to the unhashed `resources/tooling.toml`.
- Every saved format declares its name and version (`qcal_lab/formats.py`): predictions v2
  with a source header and reserved row fields, calibrator (with Platt's solver settings and
  boundary flag), calibration, cache sidecar, parity case (with `oracle.environment`) and a
  package list.
- `[numerics.regimes.*]`: `fp32` keeps torch's defaults, `fp32_tf32_off` turns TF32 and
  autotuning off and determinism on; every switch in effect is recorded, with the device,
  CUDA, cuDNN and torch's build digest. The default stays `fp32` (D5 is Ian's).
- MMDetection detectors must set `score_threshold` and `max_per_image` (`configs/lab.toml`
  pins 100; the threshold is Ian's, F10); the model's class names must equal the dataset's.
- The prediction cache key covers the prediction modules' source, the source header, the
  numerics and each kind's compute stack and resolved config, not the package version or the
  images' path. A hit needs the producer's record to list the file; the cache never replaces
  bytes; the detector is built on the first miss only.
- Smoke runs the program with the registered command, `{python} -I`.
- A Platt property test bounds parameter recovery by the loss's curvature (found by the
  `explore` profile); Platt takes a full Newton step once the predicted decrease is below the
  loss's resolution.

Verification (PR-E):
- Hypothesis invariants for isotonic regression, Platt scaling and Alg. A.1/A.2 against slow
  reference implementations; scikit-learn and SciPy oracles (`tests/oracle`, extra `oracle`)
  agree to 1e-12 and 1e-10 in loss (evidence for TD-13 and TD-16).
- `tests/contract`: the EvalLoop contract on the fixture loop and, once it exists, Ian's. The
  file and socket audit starts before the loop is imported, so a loop cannot read data while
  it is built and keep it.
- CI job `oracle` installs the `oracle` extra and runs the properties, oracles and contract
  with `ORACLE_REQUIRED=1`, so a missing library fails instead of skipping.
- `tests/e2e/test_lab_registry_loop.py`: a lab number through the registry to a claim.
- `scripts/traceability.py --check`: every integrity rule in CLAUDE.md and AGENTS.md maps to
  tests (`@pytest.mark.rule`); Hypothesis profiles `ci`, `dev`, `explore`.

### Phase 1 scaffold

Change proposal: `docs/changes/reproduce-kuzucu-eccv24-baselines.md`. Reference:
`docs/LAB.md`. It holds generic, published-method infrastructure only, with no results. The
plan gates Phase 1 on G0, which is not yet recorded.

Added:
- `qcal_lab`, the agent-owned science package:
  - configuration: packaged defaults < `configs/lab.toml`, hashed into every run and never
    read from the environment;
  - a COCO loader;
  - split manifests whose digests equal `qcal leakage`'s, and which refuse to replace a
    different split;
  - a synthetic fixture;
  - deterministic prediction files with a content-addressed cache.
- Calibrators following arXiv:2405.20459:
  - Platt scaling (Eq. 8–9, soft IoU targets, a ≥ 0);
  - isotonic regression (PAVA);
  - the two-threshold class-wise procedure (Alg. A.1/A.2), with identity for empty
    classes.
- Detectors: a fixture detector, and an MMDetection 3.x adapter (lazy import; unverified
  until the environment spike).
- `python -m qcal_lab run`, the experiment program behind `executor.command`. It:
  - refuses unknown factors and values;
  - refuses an evaluate split that also fits or selects;
  - fails before any detector work when Ian's evaluation loop is missing.
- `make smoke`: six fixture cells through `qcal registry run-batch`, then index, audit
  `--strict`, leakage, and a byte-for-byte re-run, in seconds.
- `make lab-status` and `make parity`, with a `tests/parity/` harness for oracle outputs.
- The agents `paper-reproducer` (worktree; writes only `src/qcal_lab`, `tests`, `configs`)
  and `prior-art-scout` (read-only; Ian's documents read-denied; anonymous Hugging Face MCP).
  The skills `/reproduce-check` and `/prior-art`.
- A `smoke` CI job, the `parity` test suite, and the dataset card `docs/data/coco.md`.

Fixed after the adversarial review (verdict: block; details in the change proposal):
- leakage is now checked by image id before any detector work;
- both thresholds are selected on val only;
- only Ian's files, or the digest-verified fixture stand-in, can report metrics;
- the prediction-cache key is complete, and hits are verified with their producer recorded;
- the oracle outputs, reference values and manifests are `ian_data`;
- splits cannot be replaced from the command line;
- the packaged defaults are hashed into `config_hash`.

Changed:
- `qcal.toml` sets `executor.command`.
- mypy and coverage include `qcal_lab`.
- `.mcp.json` adds the Hugging Face server (no token).

### Phase 0 hardening

Change proposal: `docs/changes/phase-0-hardening.md`.

### Added

- `qcal ci review-check`. It enforces the cross-review merge rule from git objects: the
  other model's review must be at `review/<reviewer>/<branch-slug>.md`, approving, name a
  `reviewed_sha` inside the pull request, have no review-irrelevant change after it, and
  have no open blocking finding. `integrity.yml` runs it. It is report-only until
  `review.mode = "enforce"`.
- `qcal agent-layer` checks:
  - command-reference drift: every `qcal` command and `make` target in docs, skills,
    agents, workflows and scripts must exist;
  - tool policy: explicit `tools`; no write tool alongside a network tool;
  - model allowlist;
  - skill and agent cross-references;
  - typed keys;
  - hook events, matchers, wrapped hook names and guard coverage, including hooks in agent
    frontmatter.
- `qcal registry run --supersedes RUN_ID`. `--executor` defaults to `executor.kind`.
- `--json-out` and `--github` on every `qcal ci` command.
- Skills `/pre-pr` and `/change-proposal`, both user-invoked only.
- A frontmatter `allow-only` hook that limits the data-leakage-checker's Bash access.
- SessionStart now reports the guard, signing and review modes.
- Test suites `unit`, `integration`, `regression`, `security` and `e2e`. Each has a pytest
  marker applied by directory and a make target.
- Make targets `pre-pr`, `validate`, `test-reports` (JUnit and coverage XML), `gitleaks`,
  `docker-build`, `docker-test` and `nightly`. `scripts/nightly.sh` writes a JSON-lines
  report.
- Infrastructure:
  - `Dockerfile` (non-root, offline editable install, optional build CA secret) and
    `.dockerignore`;
  - `.github/dependabot.yml` (pip, GitHub Actions, Docker);
  - `.gitleaks.toml` with a pinned, checksum-verified CI scan and a pre-commit hook;
  - CI jobs for secrets, the container, and the parquet extra.
- Documentation:
  - `README.md`, this changelog;
  - C4 diagrams in `docs/architecture/`;
  - `docs/AGENT_LAYER.md`;
  - `docs/NEXT_STEPS.md` with the tech-debt register.

### Changed

- `qcal.cli` is a package (`app`, `common`, `registry`, `checks`, `ci`, `inspection`). The
  public names `main`, `registry_main`, `claims_main`, `build_parser` and the exit codes are
  unchanged.
- The signature check's net-content rule now diffs from `merge-base(base, head)`. A pull
  request that is only behind base no longer fails for base's own signed edits.
- The signature check reads each ref's tree once (`TreeIndex`), not once per path.
- The claims checker:
  - a tagged value must be exactly one number;
  - `\url{}` and citation notes are scanned;
  - leading-dot decimals such as `.5%` are detected.
- Table rows and filters accept only factor columns.
- Every check prints readable text (a shared `qcal.reports.CheckReport` protocol); `--json`
  is unchanged.
- The hook wrapper runs Python with `-P`. Hook argument errors follow the hook's fail mode.
- Configuration replaces constants: `git.timeout_s`, `registry.env_command_timeout_s`,
  `executor.kind`, `hooks.max_nesting_depth`, `agent_layer.*` lists.
- Agents must declare `tools`.
- Environment-collector failures are logged at WARNING.
- The coverage gate goes from 90 to 95.
- Ruff adds the rule sets NPY, PERF, FURB, ERA, T20, PTH, ARG, BLE, FBT, PIE, G, LOG, FLY
  and ISC.
- `case_insensitive` is keyword-only in `qcal.globs`.

### Fixed

Each fix has a regression test in `tests/regression/` or `tests/security/`.

- A second run of a completed (cell, seed) was averaged into tables silently. It is now
  refused unless it supersedes a current run of the same pair. `--seeds` is
  de-duplicated. Audit and tables fail on duplicates.
- Dangling or cross-pair `supersedes`, non-JSON files, and nested files in the registry
  were accepted.
- A mistyped `--policy-ref` (or base or head) silently fell back to `bootstrap` defaults.
  It is now a usage error (exit 2).
- Table specs could cherry-pick runs by metric or environment values.
- Rendered numbers could hide from the claims checker.
- An executor that raised left no record of the failed run.
- Hook guard bypasses:
  - a `qcal/` package in the working directory;
  - a hand-made `.git` marker;
  - `HEAD:heads/main`;
  - `cd` or `-C` to another worktree;
  - command substitution (`git -C $(pwd) push origin main`);
  - variable and wildcard refspecs;
  - `xargs`-fed pushes.
- The Stop hook trapped sessions when its own arguments were invalid.
- Agents could overwrite Ian's documents with `qcal init --force`. Guard-bash and the
  settings deny list now refuse it.
- The weekly-review skill read signatures through host-dependent `git log %G?`.

- Adversarial review of this round (verdict: block, six findings; each has a test in
  `tests/security/test_review_and_worktree_bypasses.py`,
  `tests/security/test_guard_bypasses.py` or
  `tests/regression/test_adversarial_review_findings.py`):
  - an author could approve their own pull request. Every review file is now signed;
    reviewer names match exactly; `resolved_in` must be a reviewed commit of the PR;
  - forged `.git` files could move the policy root. Edit tools may not write git
    internals, and symlinks are judged at their target;
  - numbers with units, ranges, pairs or an adjacent sign hid from the claims checker.
    All of `paper/` and `README.md` are strict;
  - references and table rows could pick seeds or pool cells;
  - further push bypasses: `-c remote.*` and `alias.*`, persistent and shell aliases,
    `GIT_DIR`, `cd -P`, subshells, and a tokenizer gap (`);`);
  - a merge that only changed a file's mode passed the net-content rule;
  - a narrowing `disallowedTools` entry removed the whole tool from the policy check.
- Superseding a run needs a recorded `--reason`.
- Copilot review:
  - a record may not supersede itself;
  - required record fields must be non-empty strings (no `str()` coercion of `null`);
  - undecodable record files are skipped as malformed;
  - distinct cells sharing a content-addressed id are an error;
  - `allowed_signers` is no longer exempt from secret scanning.

  Second Copilot review:
  - supersede cycles are rejected by `registry-immutable` and fail the audit;
  - `registry-immutable` requires `provenance.supersede_reason` when the base policy does;
  - an aggregate reference may not name a run twice;
  - design axis and rule `fix` values must be scalars;
  - `seed_role` must be a string.

  Third Copilot review (Phase 1 scaffold):
  - a design may not use the cell id key or the seeds key as a factor name (`qcal registry
    cells` refused nothing and the emitted factor overwrote the content-addressed id);
  - the oracle-parity metric test called the evaluation-loop loader with the old signature;
  - `make lab-status` verifies the checkpoint against its recorded sha256 and reports the
    image directory.

  Fourth Copilot review (Phase 1 scaffold):
  - `python -m qcal_lab splits verify` takes its verdict from `qcal leakage`: a missing
    manifest or a duplicated id now fails, as an overlap already did, and duplicates are
    reported;
  - `make lab-status` counts oracle parity as ready only when both calibrator and metric
    cases exist.

  Tests: `tests/regression/test_copilot_review_findings.py`.

### Security

- Gitleaks scans history and the working tree in CI with a pinned, checksum-verified
  binary.
- `.gitleaks.toml`, the Gemini review directory and Dependabot configuration are on the
  enforcement surface; every committed review (`review/*/**`) is signed.
- GitHub Actions: `actions/checkout@v5` (the v4 runtime, Node 20, is deprecated).

## [0.1.0]: Phase 0 integrity layer

### Added

- Layered configuration, logging, a glob path policy, and a component registry for
  dependency injection.
- The run registry:
  - records, an atomic store, pre-registered experiments, design expansion;
  - environment collectors and a subprocess executor;
  - the runner, the index (CSV, optional Parquet), the audit, and LaTeX tables.
- Claims checker with display-precision verification and aggregate references.
- Leakage, license, forbidden-import and agent-layer checks.
- Claude Code hooks behind a fail-closed wrapper: `guard-paths`, `guard-bash`,
  `scope-write`, `deny-read`, `allow-only`, and the `claims` Stop hook.
- CI checks:
  - SSH-signed commits for protected paths (per-commit and net-content rules, policy from
    the base ref);
  - an append-only registry.
- `qcal init` templates for Ian's documents.
- `ci.yml` and `integrity.yml` workflows.
