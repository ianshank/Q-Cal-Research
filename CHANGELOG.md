# Changelog

All notable changes to the `qcal` tooling and the agent layer. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package follows
[Semantic Versioning](https://semver.org/); the version lives in `qcal.__version__`.

## [Unreleased]

Phase 0 hardening. Change proposal: `docs/changes/phase-0-hardening.md`.

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
