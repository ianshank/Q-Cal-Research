# change: registry-run-gates

## Status
Proposed, Oct 11, 2026. Edits the enforcement surface: needs Ian's signed commit.
Cycle plan: `docs/changes/cycle-2026-10-g1-readiness.md` (PR-A1).

## Gate served
G0 (Oct 23): the control plane. A registered run must be what its record says it was before
the first G1 run writes a record; records are immutable, so a gap closed later leaves every
earlier record ambiguous.

## Why
Four ways to change a registered run without a trace, all verified against `d84b9e0`:
- **A1, environment.** `qcal registry run` loaded configuration with `QCAL__*` overrides
  (`cli/app.py`), passed the whole environment to the experiment program
  (`registry/executor.py`), and the program reloaded configuration the same way
  (`qcal_lab/experiment.py`). `QCAL__PATHS__MANIFESTS_DIR` moved the manifests the run and its
  leakage check read; `QCAL__REGISTRY__CONFIG_HASH_INPUTS` narrowed what `config_hash` covers;
  `QCAL__POLICY__CATEGORIES__IAN_ONLY` could make an agent-written module pass as Ian's
  evaluation loop. The record named none of it.
- **A2, `--experiments`.** The launcher recorded factors from any YAML file while the program
  ran the configured `EXPERIMENTS.yaml`'s factors for the same cell id; tables select rows by
  the recorded factors. `experiments_sha256` was written and never checked.
- **A3, uncommitted policy and git redirection.** `qcal.toml` is not a `config_hash` input and
  `require_clean_tree` is off, so an uncommitted policy edit was a legal run input. Git ran
  with the caller's environment: `GIT_DIR`, `GIT_WORK_TREE`, `GIT_CONFIG_*` and `git replace`
  objects could answer "what is at HEAD" and "is the tree dirty" for another repository.
- **A4, open-world keys.** Unknown keys were accepted silently, so a misspelled
  `signing.signed_categories` in `qcal.toml` would fall back to the default, which lacks
  `ian_data`, and manifests and oracle outputs would pass unsigned.

## What changes
Enforcement surface (signed):
- `qcal.config`: `Config.environment_inputs` (every `QCAL__*` variable, plus `QCAL_ROOT` and
  `QCAL_CONFIG` when set); `environment_problems`, `run_neutral_environment`,
  `without_config_environment`; `config_key_problems`, the closed-world key check typed by the
  packaged defaults, with `OPEN_TABLES`, `EMPTY_LIST_ELEMENTS` and `REMOVED_KEYS` as code
  constants (they describe the schema, so configuration cannot loosen them).
- `qcal.gitutil`: git runs with redirecting `GIT_*` variables removed (identity and
  `GIT_CONFIG_NOSYSTEM`/`GIT_CONFIG_GLOBAL` kept), `GIT_NO_REPLACE_OBJECTS=1`, and
  `-c core.fsmonitor=false` on every call.
- New `qcal.registry.gates`: `run_input_problems` (environment, keys, `qcal.toml` equals `HEAD`,
  the configured pre-registration), `policy_state`, `config_inputs`, `launch_digests`,
  `input_mismatches`.
- `qcal.registry.runner`: refuses (exit 2, nothing written) before launch; records
  `provenance.{config_inputs_sha256, config_sources, config_environment, policy_source,
  policy_sha256}`; fails a run whose program reports a configuration file that changed
  since launch (`environment.inputs_read`).
- `qcal.registry.executor`: the child's environment drops `QCAL__*`, `QCAL_ROOT` and
  `QCAL_CONFIG`, then re-adds only the run-neutral overrides the record carries.
- `qcal.registry.records`: `SCHEMA_VERSION = 1` in code; records outside 1..1 are refused.
  `registry.schema_version` is removed from the defaults (a format version is not a setting).
- `qcal.registry.audit`: `mixed_inputs` lists cells whose current ok seeds ran under different
  configuration files; `--strict` fails on it.
- `qcal.cli`: `main(..., environ=)`; `qcal config --check [--json]`; `qcal config` shows
  `environment_inputs`.
- `Makefile` (`integrity` runs `qcal config --check`), `ci.yml` (the same step),
  `CLAUDE.md` (the override rule).

Agent-owned:
- `qcal_lab.experiment`: refuses environment configuration itself (a swapped executor might
  not strip it) and reports `environment.inputs_read`.
- `qcal_lab.smoke`: its in-process `qcal` calls and its uncached re-run pass an environment
  without configuration variables, so a stray `QCAL__*` in the caller's shell cannot fail
  `make smoke`.

## Decisions
- **Refuse, do not hash, environment overrides.** Hashing them into `config_hash` would make
  two records with the same committed inputs differ without saying why, and would prevent
  nothing. The hash keeps identifying committed inputs; the gate refuses everything else.
  Alternative rejected: an `allow_environment_overrides` switch. A switch read from the same
  environment authorises itself; read from `HEAD` it is one more signed key to get wrong.
- **`logging` is the only run-neutral section** (`RUN_NEUTRAL_SECTIONS`, code constant). Its
  overrides are passed to the child and recorded.
- **Compare `qcal.toml` and `EXPERIMENTS.yaml` with `HEAD` rather than requiring a clean
  tree.** A dirty tree elsewhere is already recorded (`git_dirty`) and refused when
  `require_clean_tree` is on; the policy decides what the gates themselves do and the
  pre-registration decides what runs, so both must equal `HEAD` always. A project outside git
  records `policy_source: worktree`, and `audit --strict` flags it.
- **`--experiments` stays for `--dry-run`, `audit` and `cells`**; registered runs use the
  configured file.
- **Input digests are verified after the run, not locked during it.** Locking files across
  platforms is fragile; comparing the program's reported digests with launch digests catches
  the same change and fails the run.

## Compatibility
- Records: additive provenance keys under schema 1; existing records read unchanged. Records
  claiming a schema above 1 are now refused instead of misread.
- Config: `registry.schema_version` is removed; a file that still sets it fails
  `qcal config --check` and registered runs, with a message naming the replacement.
- CLI: `qcal config` gains `--check` and `--json` and an `environment_inputs` field;
  registered runs refuse inputs they used to accept (the point of the change).
- Experiment programs: may report `environment.inputs_read`; programs that do not are
  unaffected.

## Out of scope
- `PATH`, `LD_PRELOAD`, `.pth` files, `NVIDIA_TF32_OVERRIDE` and other variables that change
  execution rather than configuration. PR-A2 records the ones that matter.
- An agent with Bash can still edit `~/.gitconfig` or `.gitignore`; the dirty check uses git's
  ignore rules. Signed commits plus CI remain the control.
- Hooks keep reading their mode from the environment by design.

## Ian decisions requested
1. Sign this change (session 1, Oct 17).
2. Turn on `registry.require_clean_tree = true` before the first G1 run (S0).
3. Confirm `audit --strict` should fail on `mixed_inputs` (proposed: yes).

## Ian hours
About 1 hour to read the diff and sign.

## Adversarial review (advisory) and what changed
`adversarial-reviewer` on `d707031`: **block**, 4 blocking, 10 non-blocking. All blocking
findings are fixed in the follow-up commit, each with a security test that fails when its
defence is disabled (checked by hand):
- **B1, a nested root.** A copy of the inputs in an ignored subdirectory (`--root build/x`
  or `QCAL_ROOT`) passed every check and wrote into the real registry. Now the run's root
  must be the top level of its git repository.
- **B2, check-then-use.** The gate re-read `qcal.toml` after the configuration had been
  parsed, so a file restored in between passed. `Config.repo_sha256` records the digest of
  the bytes parsed; the gate compares that with the `HEAD` blob read as bytes (no text
  decoding). Policy state and input digests are computed once per run and reused.
- **B3, the pre-registration.** `EXPERIMENTS.yaml` was checked by path only. It must now
  equal `HEAD` too (rule 2).
- **B4, git configuration and unreadable repositories.** The registry's git queries now run
  isolated (no global or system configuration, no global ignore or attributes file, the C
  locale) and `status` counts every untracked file. A `.git` that git cannot read is refused
  instead of being treated as "no git".

Non-blocking, applied: N1 (`provenance.inputs_unverified`; both sides compared under their
own names; `inputs_read` is a tripwire, not a control), N2 (`mixed_inputs` also compares
policy and pre-registration digests; `uncommitted_policy` fails `audit --strict`), N3
(category names in `signing.signed_categories`, `hooks.deny_categories` and
`claims.escape_hatch_categories` must exist; list-of-table items may use only the
template's keys), N4 (refusals logged at ERROR), N5 (the program's refusal is a
`ConfigError`, exit 2), N6 (the alias lookup sees the agent's own environment), N7
(`defaults` label), N8 (child-environment and `GIT_INDEX_FILE` tests), N9 (defaults header),
N10 (`SCHEMA_VERSION` default). The smoke project is now its own committed repository, so
the top-level rule holds wherever its work directory lives.

Residual, documented: an agent with Bash can still edit the repository's own `.git/config`
(for example a clean filter) or `.gitignore`; the dirty check uses the repository's own
ignore rules. `qcal.toml` must be byte-identical to `HEAD` (no CRLF conversion on checkout).
Signed commits plus CI remain the control.

## Copilot review on PR #4 and what changed
Three findings on this change held up against the code and are fixed in one follow-up
commit, each with a test that fails when its fix is reverted:
- **Uncommitted science configuration.** Only `qcal.toml` and `EXPERIMENTS.yaml` were
  compared with `HEAD`. A run could hash a work-tree `configs/lab.toml` (or lab defaults)
  that git never kept, and `registry.require_clean_tree` is off by default. Every
  `registry.config_hash_inputs` file must now equal `HEAD`. The runner reads the inputs once
  and passes the same mapping to the gate and the hash, so nothing can change in between.
- **No git, no `qcal.toml`.** The policy state inferred "`HEAD` is readable" from its source
  label, so such a project was refused against a `HEAD` that does not exist. `PolicyState`
  now records whether the root is in git.
- **List-of-table fields** (`hooks.extra_bash_deny`) were checked for unknown keys but not
  for kinds; each field is now checked against the template.

## Acceptance
- [x] `make check` green (lint, types, tests with coverage gate)
- [x] `tests/security/test_run_input_bypasses.py`: 24 attacks refused; each defence's tests
      fail with that defence disabled (checked by hand)
- [x] `make smoke` passes with `QCAL__REGISTRY__MAX_RUNS_PER_BATCH=1` exported
- [x] `qcal config --check` passes on the repository and fails on a misspelled key
- [ ] Ian's signed commit
- [ ] RESEARCH_LOG.md entry (Ian)
- [ ] the other model's review at `review/gemini/<branch-slug>.md`

## Tests
- unit: `tests/unit/test_run_gates.py` (41), `test_runner.py`, `test_records.py`,
  `test_audit.py`, `test_gitutil.py`
- security: `tests/security/test_run_input_bypasses.py`
- regression: `tests/regression/test_registry_run_gates.py` (A1–A4, including the smoke e2e)
